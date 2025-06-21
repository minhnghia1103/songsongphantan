# train.py
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

# Thiết lập hạt giống ngẫu nhiên
torch.manual_seed(42)
np.random.seed(42)

# Lấy thông tin từ biến môi trường do deepspeed.launch cung cấp
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))  # Dùng torch.cuda.device_count()
rank = int(os.environ.get('RANK', local_rank))

# Kiểm tra GPU
print(f"Rank {rank}: Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.device_count() < 2 and rank == 0:
    print("Warning: Need 2 GPUs for pipeline parallelism. Check Kaggle settings.")

# Thiết lập biến môi trường cho wandb
os.environ["WANDB_DIR"] = "/kaggle/working/wandb"
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

# Khởi tạo môi trường phân tán
if not dist.is_initialized():
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=torch.distributed.default_pg_timeout
    )
torch.cuda.set_device(local_rank)
deepspeed.init_distributed(dist_backend='nccl')

# Cấu hình huấn luyện
class Config:
    model_name = "Qwen/Qwen2.5-0.5B"
    dataset_name = "vietgpt/wikipedia_vi"
    output_dir = "/kaggle/working/qwen-vietnamese-wiki-pipeline"  # Sử dụng /kaggle/working/
    num_train_epochs = 3
    per_device_train_batch_size = 2
    per_device_valid_batch_size = 2
    gradient_accumulation_steps = 8
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
    wandb_run_name = "Kaggle-DeepSpeed-Pipeline-2T4"
    use_hf = True
    hf_repo = "h9art/PARADIS-Qwen-Pipeline-Kaggle"
    train_size = 5000
    valid_size = 500
    min_text_length = 50
    random_seed = 42
    num_stages = 2
    partition_method = "uniform"

config = Config()

# Khởi tạo wandb chỉ trên rank 0
if rank == 0:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__,
        )

# Đăng nhập Hugging Face
if config.use_hf and rank == 0:
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    login(HF_TOKEN)

# Tải tokenizer và cấu hình mô hình
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
    print(f"Cấu hình mô hình: vocab_size={vocab_size}, hidden_size={hidden_size}, num_layers={num_layers}")

# Lớp Dataset
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

# Tải dataset
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
    print(f"Số mẫu huấn luyện: {len(train_split)}")
    print(f"Số mẫu kiểm tra: {len(valid_split)}")

train_ds = WikiViDataset(train_split, tokenizer, config.max_length)
valid_ds = WikiViDataset(valid_split, tokenizer, config.max_length)

# Định nghĩa các lớp cho Pipeline
class PipelineEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
    def forward(self, input_ids):
        if isinstance(input_ids, tuple):
            input_ids = input_ids[0]
        return self.embed_tokens(input_ids)

class PipelineTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
            
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class PipelineLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, hidden_states):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

class PipelineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
    def forward(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        if isinstance(labels, tuple):
            labels = labels[0]
            
        if logits.device != labels.device:
            labels = labels.to(logits.device)
            
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        loss = self.loss_fn(shift_logits, shift_labels)
        return loss

# Tạo mô hình Pipeline
layers = [
    LayerSpec(PipelineEmbedding, vocab_size, hidden_size),
    *[LayerSpec(PipelineTransformerLayer, hidden_size, num_heads, intermediate_size) for _ in range(num_layers)],
    LayerSpec(PipelineLMHead, hidden_size, vocab_size)
]

model = PipelineModule(
    layers=layers,
    num_stages=config.num_stages,
    loss_fn=PipelineLoss(),
    partition_method=config.partition_method,
    activation_checkpoint_interval=1
)

# Cấu hình DeepSpeed
ds_config = {
    "train_batch_size": config.per_device_train_batch_size * config.gradient_accumulation_steps * 1,  # 2 * 8 * 1 = 16
    "train_micro_batch_size_per_gpu": config.per_device_train_batch_size,  # 2
    "gradient_accumulation_steps": config.gradient_accumulation_steps,  # 8
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
            "warmup_num_steps": 100
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
        "pipe_parallel_size": config.num_stages,  # 2
        "data_parallel_size": 1,
    },
    "zero_optimization": {
        "stage": 0
    },
    "verbose": True
}

# Hàm collate
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    if torch.cuda.is_available():
        input_ids = input_ids.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
    return (input_ids, labels)

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

# Khởi tạo DeepSpeed Engine
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=[p for p in model.parameters() if p.requires_grad],
    config=ds_config
)

# Hàm huấn luyện
def train_pipeline_step(model_engine, data_iter):
    try:
        loss = model_engine.train_batch(data_iter=data_iter)
        return loss.item() if loss is not None else 0.0
    except Exception as e:
        if rank == 0:
            print(f"Lỗi trong bước huấn luyện: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return None

# Hàm kiểm tra
def validate_pipeline_model(model_engine, dataloader):
    model_engine.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Đang kiểm tra", disable=rank != 0)):
            try:
                data_iter = iter([batch_data])
                loss = model_engine.eval_batch(data_iter=data_iter)
                
                if loss is not None and not torch.isnan(torch.tensor(loss)) and not torch.isinf(torch.tensor(loss)):
                    total_loss += loss.item()
                    num_batches += 1
                    
                if batch_idx >= 50:
                    break
                    
            except Exception as e:
                if rank == 0:
                    print(f"Lỗi bước kiểm tra {batch_idx}: {e}")
                continue
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 20))
        return avg_loss, perplexity
    return float('inf'), float('inf')

# Vòng lặp huấn luyện
if rank == 0:
    print("Bắt đầu huấn luyện DeepSpeed Pipeline...")
os.makedirs(config.output_dir, exist_ok=True)

training_history = {
    'train_losses': [],
    'valid_losses': [],
    'valid_perplexities': []
}

best_valid_loss = float('inf')

for epoch in range(config.num_train_epochs):
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{config.num_train_epochs}")
        print(f"{'=' * 60}")
    
    model_engine.train()
    start_time = time.time()
    total_loss = 0
    num_steps = 0
    successful_steps = 0
    
    progress_bar = tqdm(train_dataloader, desc=f"Huấn luyện Epoch {epoch + 1}", disable=rank != 0)
    
    for step, batch_data in enumerate(progress_bar):
        data_iter = iter([batch_data])
        loss = train_pipeline_step(model_engine, data_iter)
        
        if loss is not None and not math.isnan(loss) and not math.isinf(loss):
            total_loss += loss
            successful_steps += 1
            
            if rank == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'avg_loss': f"{total_loss/successful_steps:.4f}",
                    'success_rate': f"{successful_steps}/{step+1}"
                })
                
                if config.use_wandb and (step + 1) % config.logging_steps == 0:
                    wandb.log({
                        "train_loss": loss,
                        "train_step": epoch * len(train_dataloader) + step + 1,
                        "epoch": epoch + 1,
                        "success_rate": successful_steps / (step + 1)
                    })
        
        num_steps += 1
        
        if step % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    avg_train_loss = total_loss / successful_steps if successful_steps > 0 else float('inf')
    training_history['train_losses'].append(avg_train_loss)
    
    train_time = time.time() - start_time
    if rank == 0:
        print(f"Huấn luyện hoàn tất trong {train_time/60:.1f} phút")
        print(f"Loss huấn luyện trung bình: {avg_train_loss:.4f}")
        print(f"Số bước thành công: {successful_steps}/{num_steps}")
    
    if rank == 0:
        print("Đang chạy kiểm tra...")
    start_time = time.time()
    
    valid_loss, perplexity = validate_pipeline_model(model_engine, valid_dataloader)
    
    valid_time = time.time() - start_time
    training_history['valid_losses'].append(valid_loss)
    training_history['valid_perplexities'].append(perplexity)
    
    if rank == 0:
        print(f"Kiểm tra hoàn tất trong {valid_time/60:.1f} phút")
        print(f"Loss kiểm tra: {valid_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        
        if config.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "valid_loss": valid_loss,
                "perplexity": perplexity,
                "train_time_mins": train_time/60,
                "valid_time_mins": valid_time/60,
                "successful_steps": successful_steps,
                "total_steps": num_steps
            })
    
    if valid_loss < best_valid_loss and not math.isnan(valid_loss):
        best_valid_loss = valid_loss
        if rank == 0:
            print(f"🎉 Mô hình tốt nhất! Loss kiểm tra: {valid_loss:.4f}")
            checkpoint_path = os.path.join(config.output_dir, "best_checkpoint")
            model_engine.save_checkpoint(checkpoint_path)
            tokenizer.save_pretrained(config.output_dir)
            torch.save(training_history, os.path.join(config.output_dir, 'training_history.pt'))
            print(f"Mô hình đã lưu tại {checkpoint_path}")
    
    torch.cuda.empty_cache()
    gc.collect()

# Lưu checkpoint cuối cùng
if rank == 0:
    final_checkpoint = os.path.join(config.output_dir, "final_checkpoint")
    model_engine.save_checkpoint(final_checkpoint)
    print(f"Mô hình pipeline cuối cùng đã lưu tại {final_checkpoint}")

if config.use_wandb and rank == 0:
    wandb.finish()

if rank == 0:
    print("🎊 Hoàn tất huấn luyện DeepSpeed Pipeline!")