import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- 配置区 ---
model_dir = "./base_models/Qwen2.5-Coder-1.5B"
# 找到你最新的输出文件夹，或者直接填路径
adapter_dir = "./output/qwen-0428-0034"


print(f"正在加载基础模型: {model_dir}")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

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
print(f"正在挂载 LoRA 权重: {adapter_dir}")
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

# ... 后面的对话循环逻辑保持不变 ...
while True:
    user_input = input("User ❯ ")
    if user_input.lower() in ["exit", "quit", "q"]:
        break

    # 构造标准 Qwen 对话格式
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides precise Bash commands.",
        },
        {"role": "user", "content": user_input},
    ]

    # 应用 Chat Template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成答案
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            # top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,  # 结束符
        )

    # 只截取模型新生成的回答部分
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"\nAssistant ❯ \033[32m{response}\033[0m\n")
