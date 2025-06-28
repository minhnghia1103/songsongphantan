from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
# Import thêm EarlyStoppingCallback
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling, EarlyStoppingCallback 
)


# --- Step 1: Load tokenizer & EOS token ---
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
EOS_TOKEN = tokenizer.eos_token

# --- Step 2: Prompt formatting function ---
# Prompt Alpaca phù hợp với bộ dữ liệu này.
# Bạn có thể dịch prompt sang tiếng Việt nếu muốn, nhưng giữ nguyên cũng không sao.
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

# --- Step 3: Load dataset & preprocess (ĐÃ THAY ĐỔI) ---
# Tên bộ dữ liệu đã được cập nhật
dataset_name = "bkai-foundation-models/vi-alpaca" 
full_dataset = load_dataset(dataset_name, split="train")

# Chia dataset thành tập train và validation để theo dõi overfitting
# Chia dataset thành tập train và validation
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


# --- Step 4: Load Qwen model ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# --- Step 5: BỎ QUA PHẦN ÁP DỤNG LORA ---


# --- Step 6: TrainingArguments (Thêm tham số cho Early Stopping) ---
training_args = TrainingArguments(
    output_dir="./Fine-Tuning Qwen 2.5 with DeepSpeed_Zero3/qwen2.5b-full-finetune-vi-alpaca",
    num_train_epochs=5, # << Tăng số epoch lên để Early Stopping có cơ hội kích hoạt
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    
    eval_strategy ="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    
    # Rất quan trọng: tham số này sẽ tải lại checkpoint tốt nhất (dựa trên val_loss)
    # khi quá trình huấn luyện kết thúc (dù là kết thúc bình thường hay do Early Stopping).
    load_best_model_at_end=True,
    metric_for_best_model="loss", # << Chỉ định metric để xác định "model tốt nhất" là loss
    greater_is_better=False,     # << Vì là loss, nên giá trị nhỏ hơn là tốt hơn
    
    logging_steps=10,
    bf16=False,
    fp16=True,
    deepspeed=r"./ds_config_zero3.json", 
    report_to="wandb",
    run_name="qwen2.5-finetune-vi-alpaca-early-stopping",
    gradient_checkpointing=True,
    
    # Tham số này cũng hữu ích: nó sẽ giới hạn tổng số checkpoint được lưu
    # để tránh làm đầy ổ cứng. Ví dụ, chỉ lưu 3 checkpoint gần nhất.
    save_total_limit=2,
)

# --- Step 7: Trainer (Thêm callback) ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Tạo callback
# early_stopping_patience: Chờ bao nhiêu lần đánh giá trước khi dừng
# nếu metric không cải thiện. Ví dụ, nếu loss không giảm trong 3 lần
# đánh giá liên tiếp (3 * 50 = 150 bước), quá trình huấn luyện sẽ dừng.
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

# Bắt đầu quá trình full fine-tuning
trainer.train()

# Lưu model cuối cùng (sẽ là model tốt nhất đã được tự động tải lại)
trainer.save_model("./Fine-Tuning Qwen 2.5 with DeepSpeed_Zero3/qwen2.5b-full-finetune-vi-alpaca-best")