import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from trl.trainer.sft_config import SFTConfig
from ..core import registry


@registry.register_model("bf16")
def build_model_bf16(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # use bf16
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    return model


@registry.register_lora("stable-qwen-all-linear")
def build_lora_all_linear(r: int = 16, lora_dropout: float = 0.05):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=2 * r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # all linear layers in model
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return lora_config


@registry.register_sft("stable-qwen-bf16")
def buildsft_bf16(
    output_dir: str,
    lr: float = 2e-4,
    n_epochs: int = 5,
    max_length: int = 512,
    packing: bool = True,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
):
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(lr),
        num_train_epochs=n_epochs,
        max_length=max_length,
        packing=packing,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        bf16=True,
        push_to_hub=False,
        dataset_text_field="messages",
        # tensorboard
        report_to="tensorboard",
        logging_dir=f"{output_dir}/runs",
    )
    return training_args
