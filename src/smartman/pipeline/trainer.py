import argparse
from typing import Dict
import yaml
from datetime import datetime

from transformers import AutoTokenizer
from trl.trainer.sft_trainer import SFTTrainer
from datasets import load_dataset, DatasetDict, concatenate_datasets
from quicktune import registry, register_default_recipes


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, help="Path to config file")
    args = p.parse_args()
    return args


def load_data(paths: Dict):
    data_files = {"train": "train.jsonl", "eval": "eval.jsonl", "test": "test.jsonl"}
    data_dicts = [
        load_dataset("json", data_dir, data_files=data_files)
        for data_dir in paths["data_path"]
    ]

    all_splits = data_dicts[0].keys()
    merged_dict = {}
    for split in all_splits:
        datasets_to_combine = [dd[split] for dd in data_dicts]
        merged_dict[split] = concatenate_datasets(datasets_to_combine)

    return DatasetDict(merged_dict)


def load_recipe(paths: Dict, recipe: Dict):
    model_builder = registry.model_zoo[recipe["model"]]
    model = model_builder(paths["model_id"])
    lora_builder = registry.lora_zoo[recipe["lora"]]
    lora_args: Dict | None = recipe.get("lora_args")
    lora_config = lora_builder(**lora_args) if lora_args else lora_builder()
    sft_builder = registry.sft_zoo[recipe["sft"]]
    sft_args: Dict | None = recipe.get("sft_args")
    sft_config = sft_builder(**sft_args) if sft_args else sft_builder()
    return model, lora_config, sft_config


def train():
    # load config
    args = get_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    timestemp = datetime.now().strftime("%m%d_%H%M")
    run_name = paths["run_name"]
    paths["output_dir"] = f"output/{run_name}{timestemp}"
    recipe: Dict = config["recipe"]

    # load data
    data_dict = load_data(paths)

    # load recipe
    register_default_recipes()
    model, lora_config, sft_config = load_recipe(paths, recipe)

    tokenizer = AutoTokenizer.from_pretrained(paths["model_id"], trust_remote_code=True)
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
    trainer.save_model(paths["output_dir"])
    return paths["output_dir"]
