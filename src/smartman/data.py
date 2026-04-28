import json
import random
from pathlib import Path

from transformers import data

nl_path = Path("./data/raw/nl2bash/all.nl")
cm_path = Path("./data/raw/nl2bash/all.cm")
data_dir = Path("./data/processed/nl2bash")
data_dir.mkdir(exist_ok=True)
file_path = data_dir / "all.jsonl"
random.seed(42)

# 输出路径
Path("./data/processed/nl2bash").mkdir(parents=True, exist_ok=True)
all_path = Path("./data/processed/nl2bash/all.jsonl")

# ============================
# system prompt 改成中文任务描述
# ============================
system_msg = {  # 系统提示词,每一个对话样本都会带上这句话
    "role": "system",
    "content": "你是一个Linux专家，请根据用户的中文描述，输出对应的Bash命令。只输出命令本身，不要任何解释。",
}

# ============================
# 读取 + 过滤
# ============================
dataset = []
skipped = 0

with open(nl_path, "r") as nl_f, open(cm_path, "r") as cm_f:
    for nl, cm in zip(nl_f, cm_f):
        nl = nl.strip()
        cm = cm.strip()

        # 过滤掉空行、命令太短或太长的
        if not nl or not cm:
            skipped += 1
            continue
        if len(cm) < 3 or len(cm) > 300:
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


# ============================
# 保存
# ============================
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


save_jsonl(dataset, all_path)
print(f"全量数据已保存 → {all_path}")
