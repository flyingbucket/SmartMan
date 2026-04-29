import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "./base_models/Qwen2.5-Coder-1.5B"


def test_loading_strategies(prompt):
    configs = [
        {"name": "FP16_Original", "bnb": None, "use_template": True},
        {
            "name": "4bit_NF4",
            "bnb": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            ),
            "use_template": True,
        },
        {
            "name": "4bit_RawText",
            "bnb": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            ),
            "use_template": False,
        },
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for cfg in configs:
        print(f"\n--- 测试模式: {cfg['name']} ---")

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=cfg["bnb"],
            device_map="auto",
            dtype=torch.bfloat16 if cfg["bnb"] is None else None,
        )

        # 构造输入
        if cfg["use_template"]:
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # 模拟 base 模型的续写逻辑，不使用 ChatML 标签
            input_text = f"# Task: {prompt}\n# Bash command:\n"

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(  # type: ignore
                **inputs,
                max_new_tokens=30,
                do_sample=True,  # 使用 Greedy Search 排除采样随机性
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )

        result = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=False
        )
        print(f"生成的 Response:\n{result}")

        # 释放显存
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_loading_strategies("统计当前目录下有多少个.sh文件")
