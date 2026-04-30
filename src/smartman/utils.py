import os
from typing import List
from datasets import load_dataset, DatasetDict, concatenate_datasets


def load_data(data_dirs: List):
    data_dicts = []
    for data_dir in data_dirs:
        data_files = {
            "train": os.path.join(data_dir, "train.jsonl"),
            "eval": os.path.join(data_dir, "eval.jsonl"),
            "test": os.path.join(data_dir, "test.jsonl"),
        }
        dd = load_dataset("json", data_files=data_files)
        data_dicts.append(dd)

    all_splits = data_dicts[0].keys()
    merged_dict = {}
    for split in all_splits:
        datasets_to_combine = [dd[split] for dd in data_dicts]
        merged_dict[split] = concatenate_datasets(datasets_to_combine)

    return DatasetDict(merged_dict)
