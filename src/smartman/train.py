import os

# ============================
# 在所有 import 之前设置，否则不生效
# ============================
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  

import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# ============================
# 路径配置
# ============================
MODEL_PATH = "/home/by26/qy/workspace/agc2/models/Qwen2.5-Coder-1.5B"
TRAIN_DATA = "/home/by26/qy/workspace/agc2/data/processed/nl2bash/train_final.jsonl"
OUTPUT_DIR = "/home/by26/qy/workspace/agc2/output/lora"

# ============================
# 4-bit 量化配置
# ============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # ← float16 → bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ============================
# 加载 Tokenizer
# ============================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================
# 加载模型
# ⚠️ 多卡关键改动1：device_map 从 "cuda:0" 改成 "auto"
# ============================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",               # ← 改这里，自动分配到多卡
    trust_remote_code=True,
    local_files_only=True,
)
model.config.use_cache = False

# ============================
# LoRA 配置（不需要改）
# ============================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# ============================
# 数据集
# ============================
dataset = load_dataset("json", data_files=TRAIN_DATA, split="train")
print(f"训练数据: {len(dataset)} 条")

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    # 直接tokenize好再返回
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(
    format_chat,
    remove_columns=dataset.column_names,  # 删掉原始列
    num_proc=4,                            # 4进程并行处理
)

# ============================
# 训练参数
# ⚠️ 多卡关键改动2：batch和worker数量翻倍
# ============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,   # 每张卡8，4卡总batch=32
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    bf16=True,
    logging_steps=20,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=4,        # 每张卡一个worker
    ddp_find_unused_parameters=False, # ← 多卡必须加，避免报错
)

# ============================
# 开始训练
# ============================
from transformers import DataCollatorForSeq2Seq

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    ),
)

print("开始训练！")
trainer.model.print_trainable_parameters()
trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR + "/final")
tokenizer.save_pretrained(OUTPUT_DIR + "/final")
print(f"训练完成！保存在 {OUTPUT_DIR}/final")