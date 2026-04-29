import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# --- 路径配置 ---
model_id = "./base_models/Qwen2.5-Coder-1.5B"
# 指向你最新的微调输出文件夹
adapter_id = "./output/qwen0429_0120"


# 1. 定义一个强制拦截器类
class ExactTokenIDCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # 获取刚刚生成的最后一个 Token 的 ID
        last_token_id = input_ids[0][-1].item()
        # 如果这个 ID 在我们的停止列表中，强制停止！
        if last_token_id in self.stop_token_ids:
            return True
        return False


stop_words_ids = [151645, 151643]
stopping_criteria = StoppingCriteriaList([ExactTokenIDCriteria(stop_words_ids)])


def test_lora_loading_strategies(prompt):
    configs = [
        {
            "name": "LoRA_4bit_NF4_ChatML",
            "bnb": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            "use_template": True,
        },
        {"name": "LoRA_FP16_ChatML", "bnb": None, "use_template": True},
    ]

    # 加载 Tokenizer (建议直接从基础模型加载)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 确保 pad_token 正确对齐，避免生成循环 [cite: 582, 635]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 诊断代码
    print(f"Tokenizer EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    test_str = "<|im_end|>"
    print(f"Test Encode: {tokenizer.encode(test_str, add_special_tokens=False)}")

    for cfg in configs:
        print("\n" + "=" * 50)
        print(f"--- 测试模式: {cfg['name']} ---")
        print("=" * 50)

        # 1. 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=cfg["bnb"],
            device_map="auto",
            torch_dtype=torch.bfloat16 if cfg["bnb"] is None else None,
            trust_remote_code=True,
        )

        # 2. 挂载 LoRA 适配器
        print(f"正在挂载 LoRA 权重: {adapter_id}")
        model = PeftModel.from_pretrained(base_model, adapter_id)
        model.eval()

        # 3. 构造输入 (使用 Qwen2.5 标准 ChatML 模板) [cite: 197, 681]
        if cfg["use_template"]:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个Linux专家，请根据用户的描述，输出对应的Bash命令。",
                },
                {"role": "user", "content": prompt},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = f"User: {prompt}\nAssistant: "

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                # 移除 eos_token_id=stop_words_ids，改用 stopping_criteria
                stopping_criteria=stopping_criteria,
                repetition_penalty=1.1,
            )

        # 3. 极其关键的排查步骤：打印原始 ID！
        # 不要只打印 tokenizer.decode(outputs[0])，一定要看底层的数字
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]  # 截取新生成的部分
        print(f"\n[Debug] 原始生成的 Token IDs: {generated_ids.tolist()}")
        print(
            f"[Debug] 最终解码文本: {tokenizer.decode(generated_ids, skip_special_tokens=False)}"
        )

        # 6. 释放显存，准备下一次测试
        del model
        del base_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_prompt = "统计当前目录下有多少个.sh文件"
    test_lora_loading_strategies(test_prompt)
