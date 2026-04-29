import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# --- 配置区 ---
model_dir = "./base_models/Qwen2.5-Coder-1.5B"
# 找到你最新的输出文件夹，或者直接填路径
adapter_dir = "./output/qwen-0428-0034"


print(f"正在加载基础模型: {model_dir}")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer.padding_side = "left"

# 1. 显式定义量化配置 (必须这样写才能开启 4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 3. 挂载 LoRA 适配器
# print(f"正在挂载 LoRA 权重: {adapter_dir}")
# model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()


# ... 后面的对话循环逻辑保持不变 ...
def generate_bash(user_input, max_new_tokens=20):

    # 构造标准 Qwen 对话格式
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides precise Linux Bash commands.",
        },
        {"role": "user", "content": user_input},
    ]

    # 应用 Chat Template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # 生成答案
    with torch.no_grad():
        generated_ids = model.generate(  # type: ignore
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, im_end_id],
        )

    # 只截取模型新生成的回答部分
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return response


test_cases = [
    "列出当前目录下所有的.py文件",
    "查看系统内存使用情况",
    "把file.txt的内容按行倒序输出",
    "递归删除当前目录下所有__pycache__文件夹",
    "统计当前目录下有多少个.sh文件",
]


for i, case in enumerate(test_cases, 1):
    result = generate_bash(case)
    print(f"\n[{i}] 输入: {case}")
    print(f"    输出: {result}")
