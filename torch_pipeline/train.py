from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.distributed.pipeline.sync import Pipeline
import os

# Khởi tạo Accelerator
accelerator = Accelerator(
    gradient_accumulation_steps=16,
    mixed_precision="bf16"
)

# --- Step 1: Load tokenizer & EOS token ---
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

EOS_TOKEN = tokenizer.eos_token

# --- Step 2: Prompt formatting function (Giữ nguyên) ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}""" + EOS_TOKEN

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_, output in zip(instructions, inputs, outputs):
        if input_ and input_.strip():
            text = alpaca_prompt.format(instruction, input_, output)
        else:
            prompt_no_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}""" + EOS_TOKEN
            text = prompt_no_input.format(instruction, output)
        texts.append(text)
    return {"text": texts}

# --- Step 3: Load dataset & preprocess (Giữ nguyên) ---
dataset_name = "bkai-foundation-models/vi-alpaca"
full_dataset = load_dataset(dataset_name, split="train")

# Chia dataset thành tập train và validation
split_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"].select(range(min(10000, len(split_dataset["train"]))))
eval_dataset = split_dataset["test"].select(range(min(10000, len(split_dataset["test"]))))

# Áp dụng hàm định dạng và token hóa
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)

# --- Step 4: Load model ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# --- Step 5: Chia mô hình thành các stage cho Pipeline Parallelism ---
# Giả sử sử dụng 2 stage như trong cấu hình DeepSpeed
num_stages = 2
device_map = {i: i % accelerator.num_processes for i in range(len(model.transformer.h))}
model = torch.nn.Sequential(*[model.transformer.h[i] for i in range(len(model.transformer.h))])
pipeline = Pipeline(
    modules=model,
    partition_points=[len(model) // num_stages],
    device=torch.device(f"cuda:{accelerator.local_process_index}")
)

# --- Step 6: Chuẩn bị model và optimizer với Accelerator ---
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    pipeline,  # Sử dụng pipeline thay vì model trực tiếp
    torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01),
    tokenized_train_dataset,
    tokenized_eval_dataset
)

# --- Step 7: Cấu hình TrainingArguments ---
training_args = TrainingArguments(
    output_dir="./qwen2-7b-pipeline-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    warmup_steps=100,
    max_grad_norm=1.0,
    logging_steps=10,
    report_to="wandb",
    run_name="qwen2.5-finetune-pipeline",
    gradient_checkpointing=True,
)

# --- Step 8: Trainer ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- Step 9: Bắt đầu huấn luyện ---
accelerator.print("Bắt đầu huấn luyện với PyTorch Pipeline Parallelism và Accelerate...")
trainer.train()

# Lưu mô hình
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)