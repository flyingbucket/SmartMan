from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument("--base_model_name", type=str)
p.add_argument("--adapter_name", type=str)
args = p.parse_args()
base_model_name = args.base_model_name
adapter_name = args.adapter_name

merged_path = f"./dist/{base_model_name}_{adapter_name}"

# 显式指定使用 CPU 加载
base_model = AutoModelForCausalLM.from_pretrained(
    f"./base_models/{base_model_name}",
    torch_dtype=torch.float32,
    device_map="cpu",
)

model = PeftModel.from_pretrained(
    base_model,
    f"./output/{adapter_name}",
    device_map="cpu",
)

# 合并并保存
merged_model = model.merge_and_unload()  # type:ignore
merged_model.save_pretrained(merged_path, safe_serialization=True)

# 处理 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"./base_models/{base_model_name}")
tokenizer.save_pretrained(merged_path)

print(f"Merged and saved to: {merged_path}")
