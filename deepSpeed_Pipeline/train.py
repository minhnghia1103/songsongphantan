import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from datasets import load_dataset, DatasetDict
import wandb
import math

# --- Step 1: Cấu hình ---
class Config:
    """Cấu hình cho quá trình fine-tuning"""
    model_name = "Qwen/Qwen3-0.6B"
    dataset_name = "yahma/alpaca-cleaned"
    output_dir = "./qwen3-0.6B-pipeline-finetuned"
    train_size = 10000
    valid_size = 10000
    test_size = 5000
    min_text_length = 50
    max_length = 512
    num_train_epochs = 1
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_steps = 100
    logging_steps = 10
    save_strategy = "epoch"
    evaluation_strategy = "epoch"
    wandb_project = "PARADIS-Qwen3_0.6B"
    deepspeed_config = "./ds_config.json"

# --- Step 2: Thiết lập biến môi trường và W&B ---
def setup_environment():
    """Thiết lập biến môi trường và đăng nhập W&B"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY không được thiết lập trong biến môi trường")
    wandb.login(key=wandb_api_key)

# --- Step 3: Load tokenizer ---
config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
EOS_TOKEN = tokenizer.eos_token

# --- Step 4: Prompt formatting function ---
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
        text = alpaca_prompt.format(instruction, input_, output)
        texts.append(text)
    return {"text": texts}

# --- Step 5: Load và preprocess dataset ---
def filter_function(example):
    """Lọc các mẫu có output quá ngắn"""
    return example['output'] is not None and len(example['output'].strip()) > config.min_text_length

def load_and_preprocess_data():
    """Tải và tiền xử lý dataset"""
    dataset = load_dataset(config.dataset_name, split="train")
    dataset = dataset.select_columns(['instruction', 'input', 'output'])
    dataset = dataset.filter(filter_function)
    dataset = dataset.shuffle(seed=42)
    
    # Tạo train, valid, test splits
    train_split = dataset.select(range(config.train_size))
    valid_split = dataset.select(range(config.train_size, config.train_size + config.valid_size))
    test_split = dataset.select(range(config.train_size + config.valid_size, 
                                    config.train_size + config.valid_size + config.test_size))
    
    return DatasetDict({
        "train": train_split,
        "valid": valid_split,
        "test": test_split
    })

# --- Step 6: Tokenize dataset ---
def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=config.max_length, return_overflowing_tokens=False)

# --- Step 7: Main processing ---
def main():
    # Thiết lập môi trường
    setup_environment()
    
    # Tải dataset và tạo splits
    print("Đang tải và tiền xử lý dataset...")
    dataset = load_and_preprocess_data()
    print(f"Kích thước dataset: Train={len(dataset['train'])}, Valid={len(dataset['valid'])}, Test={len(dataset['test'])}")
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(formatting_prompts_func, batched=True).map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    
    # Load model
    print("Đang tải model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Thiết lập TrainingArguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        warmup_steps=config.warmup_steps,
        max_grad_norm=1.0,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.logging_steps,
        bf16=True,
        deepspeed=config.deepspeed_config,
        report_to="wandb",
        run_name=config.wandb_project,
        gradient_checkpointing=True,
    )
    
    # Khởi tạo Trainer với callback để log perplexity và thời gian
    class CustomCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.train_start_time = time.time()

        def on_train_end(self, args, state, control, **kwargs):
            train_time = (time.time() - self.train_start_time) / 60  # Chuyển sang phút
            wandb.log({"train_time (m)": train_time})

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            if "eval_loss" in metrics:
                perplexity = math.exp(metrics["eval_loss"])
                valid_time = (time.time() - getattr(self, "eval_start_time", time.time())) / 60  # Chuyển sang phút
                wandb.log({"perplexity": perplexity, "valid_time (m)": valid_time})
            self.eval_start_time = time.time()  # Cập nhật thời gian bắt đầu đánh giá

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[CustomCallback()],  # Thêm callback tùy chỉnh
    )
    
    # Bắt đầu huấn luyện
    print("Bắt đầu huấn luyện Full-Tuning với DeepSpeed Pipeline...")
    trainer.train()
    
    # Đánh giá trên tập test
    print("Đánh giá trên tập test...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Kết quả trên tập test: {test_results}")
    # Log test results to W&B
    if "eval_loss" in test_results:
        wandb.log({"test_loss": test_results["eval_loss"], "test_perplexity": math.exp(test_results["eval_loss"])})

if __name__ == "__main__":
    main()