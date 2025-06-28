from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_dataset
import torch
import os

# --- Step 1: Load tokenizer & EOS token ---
# Sửa nhỏ: Tên model chính thức là Qwen2
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
        # Bộ dữ liệu này có thể có một số mẫu không có 'input'.
        # Xử lý trường hợp 'input' rỗng.
        if input_ and input_.strip():
            text = alpaca_prompt.format(instruction, input_, output)
        else:
            # Dùng một biến thể prompt khác cho các tác vụ không cần input
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

split_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)

# Lấy tối đa 10.000 mẫu cho train và 10.000 cho validation
train_dataset = split_dataset["train"].select(range(min(10000, len(split_dataset["train"]))))
eval_dataset = split_dataset["test"].select(range(min(10000, len(split_dataset["test"]))))

# Áp dụng hàm định dạng và token hóa cho cả hai tập
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)


def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
# --- Step 6: Sửa lại TrainingArguments cho Full Fine-Tuning ---
training_args = TrainingArguments(
    output_dir="./Fine-Tuning Qwen 2.5 with DeepSpeed Pipeline/qwen2-7b-pipeline-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    eval_strategy ="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    warmup_steps=100,
    max_grad_norm=1.0,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    deepspeed="./ds_config.json",
    report_to="wandb",                         # << Bật wandb
    run_name="qwen2.5-finetune-pipeline",      # << Tên run
    gradient_checkpointing=True,
)

# --- Step 7: Trainer (Giữ nguyên) ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    # Thêm callback vào đây
    callbacks=[early_stopping_callback],
)

print("Bắt đầu huấn luyện Full-Tuning với DeepSpeed Pipeline...")
trainer.train()


trainer.save_model("./Fine-Tuning Qwen 2.5 with DeepSpeed_Zero3/qwen2.5b-full-finetune-vi-alpaca-best")