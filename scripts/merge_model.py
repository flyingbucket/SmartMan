from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

baase_model_name = "Qwen2.5-Coder-1.5B"
adapter_name = "qwenInstruct_short0501_0107"
merged_path = f"./dist/{baase_model_name}_{adapter_name}"

# 显式指定使用 CPU 加载
base_model = AutoModelForCausalLM.from_pretrained(
    f"./base_models/{baase_model_name}",
    torch_dtype=torch.float32,
    device_map="cpu",
)

model = PeftModel.from_pretrained(
    base_model,
    f"./output/{adapter_name}",
    device_map="cpu",
)

# 合并并保存
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_path, safe_serialization=True)

# 处理 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"./base_models/{baase_model_name}")
tokenizer.save_pretrained(merged_path)

print(f"成功合并并保存至: {merged_path}")
