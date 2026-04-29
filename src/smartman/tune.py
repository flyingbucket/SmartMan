import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from datetime import datetime

model_id = "./base_models/Qwen2.5-1.5B-Instruct"
data_path = "data/processed/nl2bash"

timestemp = datetime.now().strftime("%m%d_%H%M")
run_name = "qwenInsctruct"
output_dir = f"output/{run_name}{timestemp}"


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # use bf16
    device_map="auto",
    attn_implementation="flash_attention_2",
)

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
    ],  # all linear layers in model
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# data preparation
dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(data_path, "train.jsonl"),
        "eval": os.path.join(data_path, "eval.jsonl"),  # 用 eval 做验证
    },
)

training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=5,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    bf16=True,
    push_to_hub=False,
    dataset_text_field="messages",
    max_length=512,
    packing=True,
    # tensorboard
    report_to="tensorboard",
    logging_dir=f"{output_dir}/runs",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    peft_config=lora_config,
    args=training_args,
    processing_class=tokenizer,
)

print("开始微调...")
trainer.train()
trainer.save_model(output_dir)
print(f"微调完成，适配器已保存至: {output_dir}")
