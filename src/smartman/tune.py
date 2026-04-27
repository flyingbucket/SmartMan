import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from datetime import datetime

# config
model_dir = "./base_models/Qwen2.5-Coder-1.5B"
data_path = "./data/processed/nl2bash/all.jsonl"

run_name = "qwen"
timestamp = datetime.now().strftime("%m%d-%H%M")
output_dir = f"./output/{run_name}-{timestamp}"

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

dataset = load_dataset("json", data_files=data_path, split="train")

sft_config = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="messages",
    max_length=1024,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_steps=100,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    peft_config=lora_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(output_dir)
