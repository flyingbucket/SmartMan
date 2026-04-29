from argparse import ArgumentParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import PeftModel

model_id = "./base_models/Qwen2.5-1.5B-Instruct"
arg_parser = ArgumentParser()
arg_parser.add_argument("--adapter_dir", type=str, default=None)
arg_parser.add_argument("--max_new_tokens", type=int, default=30)
args = arg_parser.parse_args()


def test_lora_loading_strategies(prompt):
    configs = [
        {
            "name": "4bit_NF4_ChatML",
            "bnb": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            "use_template": True,
        },
        {"name": "FP16_ChatML", "bnb": None, "use_template": True},
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    for cfg in configs:
        print("\n" + "=" * 50)
        print(f"--- 测试模式: {cfg['name']} ---")
        print("=" * 50)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=cfg["bnb"],
            device_map="auto",
            dtype=torch.bfloat16 if cfg["bnb"] is None else None,
            trust_remote_code=True,
        )
        if args.adapter_dir:
            print(f"Loading LoRA layer: {args.adapter_dir}")
            model = PeftModel.from_pretrained(base_model, args.adapter_dir)
            model.eval()
        else:
            print("Using base model")
            model = base_model
            model.eval()

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
            outputs = model.generate(  # type: ignore
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["<|im_end|>", "<|endoftext|>"],
                repetition_penalty=1.1,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]  # 截取新生成的部分
        generated_ids = generated_ids.tolist()
        eos_found: bool = 151643 in generated_ids or 151645 in generated_ids
        print(f"\n[Debug]EOS in generated token ids:{eos_found}")
        print(f"[Debug] 原始生成的 Token IDs: {generated_ids}")
        print(
            f"[Debug] 最终解码文本: {tokenizer.decode(generated_ids, skip_special_tokens=False)}"
        )

        del model
        del base_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_prompt = "统计当前目录下有多少个.sh文件"
    test_lora_loading_strategies(test_prompt)
