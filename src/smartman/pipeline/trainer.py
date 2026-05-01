from typing import Dict
from datetime import datetime

from datasets import DatasetDict
from transformers import AutoTokenizer
from trl.trainer.sft_trainer import SFTTrainer
from quicktune import registry, register_default_recipes


def load_recipe(model_id: str, recipe: Dict, output_dir: str):
    model_builder = registry.model_zoo[recipe["model"]]
    model = model_builder(model_id)
    lora_builder = registry.lora_zoo[recipe["lora"]]
    lora_args: Dict | None = recipe.get("lora_args")
    lora_config = lora_builder(**lora_args) if lora_args else lora_builder()
    sft_builder = registry.sft_zoo[recipe["sft"]]
    sft_args: Dict | None = recipe.get("sft_args")
    sft_config = (
        sft_builder(output_dir, **sft_args) if sft_args else sft_builder(output_dir)
    )
    return model, lora_config, sft_config


def train(model_id: str, run_name: str, recipe: Dict, data_dict: DatasetDict):
    timestemp = datetime.now().strftime("%m%d_%H%M")
    output_dir = f"output/{run_name}{timestemp}"

    # load recipe
    register_default_recipes()
    model, lora_config, sft_config = load_recipe(model_id, recipe, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    trainer = SFTTrainer(
        model=model,
        train_dataset=data_dict["train"],
        eval_dataset=data_dict["eval"],
        peft_config=lora_config,
        args=sft_config,
        processing_class=tokenizer,
    )

    # train
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
