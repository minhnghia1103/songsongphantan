import os
import torch
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.distributed as dist
import math
import time
import gc
import wandb
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from huggingface_hub import login
import subprocess

# Thiết lập hạt giống ngẫu nhiên
torch.manual_seed(42)
np.random.seed(42)

# Lấy thông tin từ biến môi trường
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
rank = int(os.environ.get('RANK', 0))

print(f"Rank {rank}: Initializing with local_rank={local_rank}, world_size={world_size}")

# Thiết lập CUDA device trước
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')

# Khởi tạo distributed training theo thứ tự đúng
if not dist.is_initialized():
    # Sử dụng NCCL backend cho GPU
    dist.init_process_group(
        backend='nccl',
        init_method='env://',  # Sử dụng environment variables
        world_size=world_size,
        rank=rank
    )
    print(f"Rank {rank}: Process group initialized")

# Khởi tạo DeepSpeed distributed - QUAN TRỌNG: phải sau init_process_group
deepspeed.init_distributed(
    dist_backend='nccl',
    auto_mpi_discovery=False,  # Tắt auto MPI discovery
    distributed_port=29500,    # Explicit port
    verbose=True
)

# Đồng bộ tất cả processes
dist.barrier()

# Hàm kiểm tra GPU
def check_gpu_status():
    gpu_info = f"Rank {rank}: Local rank: {local_rank}, Device: {torch.cuda.current_device()}\n"
    gpu_info += f"Rank {rank}: Device name: {torch.cuda.get_device_name(local_rank)}\n"
    gpu_info += f"Rank {rank}: Total GPUs: {torch.cuda.device_count()}\n"
    gpu_info += f"Rank {rank}: Memory allocated: {torch.cuda.memory_allocated(local_rank) / 1024**3:.2f} GB\n"
    gpu_info += f"Rank {rank}: World size: {dist.get_world_size()}\n"
    return gpu_info

if rank == 0:
    print("GPU Status at initialization:")
    print(check_gpu_status())

# Cấu hình
class Config:
    model_name = "Qwen/Qwen2.5-0.5B"
    dataset_name = "vietgpt/wikipedia_vi"
    output_dir = "/kaggle/working/qwen-vietnamese-wiki-pipeline"
    num_train_epochs = 3
    per_device_train_batch_size = 1  # Micro batch size per GPU
    per_device_valid_batch_size = 1
    gradient_accumulation_steps = 8  # Giảm xuống để tránh OOM
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_length = 512
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    logging_steps = 10
    save_strategy = "epoch"
    fp16 = True
    num_workers = 0
    use_wandb = True
    wandb_project = "PARADIS-Qwen-Pipeline"
    wandb_run_name = "Kaggle-DeepSpeed-Pipeline-2GPU-Fixed"
    use_hf = True
    hf_repo = "MinhNghia/PARADIS-Qwen-Pipeline-Kaggle"
    train_size = 1000  # Giảm size để test
    valid_size = 100
    min_text_length = 50
    random_seed = 42
    num_stages = 2  # 2 stages cho 2 GPU
    partition_method = "uniform"

config = Config()

# Khởi tạo wandb và HF chỉ trên rank 0
if rank == 0:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        
        if config.use_wandb:
            WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
            wandb.login(key=WANDB_API_KEY)
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__,
            )
        
        if config.use_hf:
            HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
            login(HF_TOKEN)
    except Exception as e:
        print(f"Warning: Could not initialize wandb/hf: {e}")

# Tải tokenizer và config
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
vocab_size = model_config.vocab_size
hidden_size = model_config.hidden_size
num_layers = model_config.num_hidden_layers
num_heads = model_config.num_attention_heads
intermediate_size = model_config.intermediate_size

if rank == 0:
    print(f"Model config: vocab_size={vocab_size}, hidden_size={hidden_size}, num_layers={num_layers}")

# Dataset class
class WikiViDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        combined_text = f"Tiêu đề: {item['title']}\n\nNội dung: {item['text']}"
        
        tokenized = self.tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

# Load dataset
if rank == 0:
    print("Loading dataset...")

dataset = load_dataset(config.dataset_name, split="train")
dataset = dataset.select_columns(['title', 'text'])

def filter_function(example):
    return (
        example['text'] is not None and 
        example['title'] is not None and
        len(example['text'].strip()) > config.min_text_length
    )

dataset = dataset.filter(filter_function)
dataset = dataset.shuffle(seed=config.random_seed)

train_split = dataset.select(range(config.train_size))
valid_split = dataset.select(range(config.train_size, config.train_size + config.valid_size))

if rank == 0:
    print(f"Train samples: {len(train_split)}")
    print(f"Valid samples: {len(valid_split)}")

train_ds = WikiViDataset(train_split, tokenizer, config.max_length)
valid_ds = WikiViDataset(valid_split, tokenizer, config.max_length)

# Pipeline components - cải thiện implementation
class PipelineEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_dim = hidden_size
        
    def forward(self, x):
        # Handle both tuple and tensor input
        if isinstance(x, tuple):
            input_ids = x[0]
        else:
            input_ids = x
            
        # Ensure input_ids is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        embeddings = self.embed_tokens(input_ids)
        return embeddings

class PipelineTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(0.1)
        )
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        
    def forward(self, x):
        if isinstance(x, tuple):
            hidden_states = x[0]
        else:
            hidden_states = x
            
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        try:
            attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
            hidden_states = residual + attn_output
        except Exception as e:
            print(f"Attention error: {e}")
            hidden_states = residual
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class PipelineLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, x):
        if isinstance(x, tuple):
            hidden_states = x[0]
        else:
            hidden_states = x
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

# Loss function với cải thiện
class PipelineLoss(nn.Module):
    def __init__(self, vocab_size, pad_token_id):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='mean')
        self.vocab_size = vocab_size
        
    def forward(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        if isinstance(labels, tuple):
            labels = labels[0]
            
        # Ensure same device
        if logits.device != labels.device:
            labels = labels.to(logits.device)
            
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss calculation
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        
        loss = self.loss_fn(shift_logits, shift_labels)
        return loss

# Tạo pipeline model
if rank == 0:
    print("Creating pipeline model...")

# Tạo layers cho pipeline
layers = []

# Embedding layer
layers.append(LayerSpec(PipelineEmbedding, vocab_size, hidden_size))

# Transformer layers
for i in range(num_layers):
    layers.append(LayerSpec(PipelineTransformerLayer, hidden_size, num_heads, intermediate_size))

# Output layer
layers.append(LayerSpec(PipelineLMHead, hidden_size, vocab_size))

# Tạo pipeline model
model = PipelineModule(
    layers=layers,
    num_stages=config.num_stages,
    loss_fn=PipelineLoss(vocab_size, tokenizer.pad_token_id),
    partition_method=config.partition_method,
    activation_checkpoint_interval=1
)

# Tính toán batch size chính xác cho DeepSpeed Pipeline
# Với Pipeline Parallelism, data_parallel_size = world_size / pipe_parallel_size
data_parallel_size = world_size // config.num_stages
train_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps * data_parallel_size

if rank == 0:
    print(f"Batch size calculation:")
    print(f"  per_device_train_batch_size: {config.per_device_train_batch_size}")
    print(f"  gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    print(f"  world_size: {world_size}")
    print(f"  num_stages: {config.num_stages}")
    print(f"  data_parallel_size: {data_parallel_size}")
    print(f"  train_batch_size: {train_batch_size}")

# DeepSpeed config với batch size được tính đúng
ds_config = {
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": config.per_device_train_batch_size,
    "gradient_accumulation_steps": config.gradient_accumulation_steps,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "eps": config.adam_epsilon,
            "betas": [0.9, 0.999]
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": config.learning_rate,
            "warmup_num_steps": 50
        }
    },
    "fp16": {
        "enabled": config.fp16,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "gradient_clipping": config.max_grad_norm,
    "steps_per_print": config.logging_steps,
    "pipeline": {
        "activation_checkpoint_interval": 1,
        "pipe_parallel_size": config.num_stages,
        "data_parallel_size": data_parallel_size,
    },
    "zero_optimization": {
        "stage": 0  # Disable ZeRO for pipeline parallelism
    },
    "wall_clock_breakdown": False,
    "memory_breakdown": False
}

# Collate function
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return (input_ids, labels)

# DataLoaders
train_dataloader = DataLoader(
    train_ds,
    batch_size=config.per_device_train_batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    collate_fn=collate_fn,
    pin_memory=False,
    drop_last=True
)

valid_dataloader = DataLoader(
    valid_ds,
    batch_size=config.per_device_valid_batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    collate_fn=collate_fn,
    pin_memory=False,
    drop_last=False
)

# Initialize DeepSpeed
if rank == 0:
    print("Initializing DeepSpeed...")

model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    config=ds_config
)

# FIXED: Training functions với proper timer handling
def train_pipeline_step(model_engine, data_iterator):
    """
    Khắc phục lỗi timer bằng cách sử dụng data iterator thay vì tạo mới
    """
    try:
        # Không tạo iterator mới, sử dụng iterator đã có
        loss = model_engine.train_batch(data_iter=data_iterator)
        return loss.item() if loss is not None else None
    except Exception as e:
        if rank == 0:
            print(f"Training step error: {e}")
        return None

def validate_pipeline_model(model_engine, dataloader, max_steps=20):
    model_engine.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        # Tạo iterator một lần cho validation
        data_iter = iter(dataloader)
        
        for batch_idx in range(min(max_steps, len(dataloader))):
            try:
                batch_data = next(data_iter)
                single_batch_iter = iter([batch_data])
                loss = model_engine.eval_batch(data_iter=single_batch_iter)
                
                if loss is not None and not torch.isnan(torch.tensor(loss)):
                    total_loss += loss.item()
                    num_batches += 1
                    
            except StopIteration:
                break
            except Exception as e:
                if rank == 0:
                    print(f"Validation step {batch_idx} error: {e}")
                continue
    
    model_engine.train()
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 20))
        return avg_loss, perplexity
    return float('inf'), float('inf')

# FIXED: Training loop với proper data handling
if rank == 0:
    print("Starting training...")
    
os.makedirs(config.output_dir, exist_ok=True)

best_valid_loss = float('inf')

for epoch in range(config.num_train_epochs):
    if rank == 0:
        print(f"\nEpoch {epoch + 1}/{config.num_train_epochs}")
    
    model_engine.train()
    total_loss = 0
    successful_steps = 0
    
    # Tạo data iterator cho toàn bộ epoch
    train_iter = iter(train_dataloader)
    
    progress_bar = tqdm(range(len(train_dataloader)), desc=f"Training Epoch {epoch + 1}", disable=rank != 0)
    
    for step in progress_bar:
        try:
            # Lấy batch từ iterator
            batch_data = next(train_iter)
            
            # Tạo single batch iterator cho train_batch
            single_batch_iter = iter([batch_data])
            
            # Gọi training step với iterator
            loss = train_pipeline_step(model_engine, single_batch_iter)
            
            if loss is not None and not math.isnan(loss):
                total_loss += loss
                successful_steps += 1
                
                if rank == 0:
                    progress_bar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'avg_loss': f"{total_loss/successful_steps:.4f}" if successful_steps > 0 else "N/A"
                    })
                    
                    if config.use_wandb and (step + 1) % config.logging_steps == 0:
                        wandb.log({
                            "train_loss": loss,
                            "step": epoch * len(train_dataloader) + step + 1,
                            "epoch": epoch + 1
                        })
            
            # Clean up memory periodically
            if step % 10 == 0:
                torch.cuda.empty_cache()
                
            # Limit steps for testing
            if step >= 50:
                break
                
        except StopIteration:
            break
        except Exception as e:
            if rank == 0:
                print(f"Step {step} error: {e}")
            continue
    
    avg_train_loss = total_loss / successful_steps if successful_steps > 0 else float('inf')
    
    if rank == 0:
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Successful steps: {successful_steps}")
    
    # Validation
    if rank == 0:
        print("Running validation...")
    
    valid_loss, perplexity = validate_pipeline_model(model_engine, valid_dataloader)
    
    if rank == 0:
        print(f"Validation loss: {valid_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        
        if config.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "valid_loss": valid_loss,
                "perplexity": perplexity
            })
    
    # Save best model
    if valid_loss < best_valid_loss and not math.isnan(valid_loss):
        best_valid_loss = valid_loss
        if rank == 0:
            print(f"New best model! Validation loss: {valid_loss:.4f}")
            checkpoint_path = os.path.join(config.output_dir, "best_checkpoint")
            model_engine.save_checkpoint(checkpoint_path)
            tokenizer.save_pretrained(config.output_dir)
    
    torch.cuda.empty_cache()
    gc.collect()

# Final cleanup
if rank == 0:
    print("Training completed!")
    final_checkpoint = os.path.join(config.output_dir, "final_checkpoint")
    model_engine.save_checkpoint(final_checkpoint)
    
    if config.use_wandb:
        wandb.finish()
    
    print("Final GPU Status:")
    print(check_gpu_status())