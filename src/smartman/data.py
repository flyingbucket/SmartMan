import json
from pathlib import Path

nl_path = Path("./data/raw/nl2bash/all.nl")
cm_path = Path("./data/raw/nl2bash/all.cm")
file_path = Path("./data/processed/nl2bash/all.jsonl")

system_msg = {"role": "system", "content": "You are a helpful assistant."}
dataset = []

with open(nl_path, "r") as nl_f:
    with open(cm_path, "r") as cm_f:
        for nl, cm in zip(nl_f, cm_f):
            user_msg = {"role": "user", "content": nl}
            assistant_msg = {"role": "assistant", "content": cm}
            message = [system_msg, user_msg, assistant_msg]
            dataset.append(message)


with open(file_path, "w", encoding="utf-8") as f:
    for entry in dataset:
        json_record = {"messages": entry}
        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")

print(f"Saved to {file_path}")
