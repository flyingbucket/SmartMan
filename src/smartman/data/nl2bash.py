import json
import random
from pathlib import Path

random.seed(42)

nl_path = Path("./data/raw/nl2bash/all.nl")
cm_path = Path("./data/raw/nl2bash/all.cm")

data_dir = Path("./data/processed/nl2bash")
data_dir.mkdir(exist_ok=True)


system_msg = {
    "role": "system",
    "content": "你是一个Linux专家，请根据用户的描述，输出对应的Bash命令。只输出命令本身，不要任何解释。",
}

dataset = []
skipped = 0

with open(nl_path, "r") as nl_f, open(cm_path, "r") as cm_f:
    for nl, cm in zip(nl_f, cm_f):
        nl = nl.strip()
        cm = cm.strip()

        # 过滤掉空行、命令太长或太短的
        if not nl or not cm:
            skipped += 1
            continue
        if len(cm) > 300:
            skipped += 1
            continue

        dataset.append(
            {
                "messages": [
                    system_msg,
                    {"role": "user", "content": nl},
                    {"role": "assistant", "content": cm},
                ]
            }
        )

print(f"有效数据: {len(dataset)} 条")
print(f"过滤掉:   {skipped} 条")


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


save_jsonl(dataset, data_dir / "all.jsonl")
print(f"全量数据已保存 → {data_dir / 'all.jsonl'}")


random.shuffle(dataset)

# 8:1:1
total = len(dataset)
train_end = int(total * 0.8)
val_end = int(total * 0.9)
assert train_end > 0 and val_end > train_end

train_data = dataset[:train_end]
val_data = dataset[train_end:val_end]
test_data = dataset[val_end:]

save_jsonl(train_data, data_dir / "train.jsonl")
save_jsonl(val_data, data_dir / "eval.jsonl")
save_jsonl(test_data, data_dir / "test.jsonl")

print(
    f"划分完成：Train({len(train_data)}), Eval({len(val_data)}), Test({len(test_data)})"
)
