import json
import random
from pathlib import Path

random.seed(42)

all_path     = Path("./data/processed/nl2bash/all.jsonl")       # 英文全量
chinese_path = Path("./data/processed/nl2bash/train_chinese.jsonl")  # 中文数据
train_path   = Path("./data/processed/nl2bash/train_final.jsonl")
test_path    = Path("./data/processed/nl2bash/test.jsonl")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

all_data = load_jsonl(all_path)
chinese  = load_jsonl(chinese_path)

print(f"英文原始数据: {len(all_data)} 条")
print(f"中文数据:     {len(chinese)} 条 × 5 = {len(chinese)*5} 条")

# ============================
# 先从英文数据里划出测试集（划分在合并之前）
# 保证测试集是纯净的英文原始数据，不混入中文
# ============================
random.shuffle(all_data)
split    = int(len(all_data) * 0.9)
train_en = all_data[:split]
test     = all_data[split:]     # 测试集：只用英文原始数据

# ============================
# 训练集 = 英文训练部分 + 中文×5，再打乱
# ============================
train_final = train_en + chinese * 5
random.shuffle(train_final)

# ============================
# 保存
# ============================
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl(train_final, train_path)
save_jsonl(test,        test_path)

print(f"\n最终训练集: {len(train_final)} 条 → {train_path}")
print(f"最终测试集: {len(test)}  条 → {test_path}")