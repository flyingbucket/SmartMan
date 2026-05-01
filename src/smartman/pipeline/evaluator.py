from pathlib import Path
import shlex
from typing import List
from datetime import datetime

import bashlex
import torch
import pandas as pd
from bashlex import errors as bash_errors
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

from quicktune import register_default_recipes, registry


class BashEvaluator:
    def __init__(self):
        # 预加载 NLTK 平滑函数，防止短命令计算 BLEU 时得分为 0
        self.smoothie = SmoothingFunction().method4

    def normalize_bash_to_tokens(self, command: str) -> list:
        """
        利用标准库 shlex 进行规范化预处理
        自动完成：压缩空格、处理各种嵌套引号、提取纯净的参数 Token
        """
        try:
            # split in POSIX standard
            return shlex.split(command)
        except ValueError:
            # 捕获 shlex 解析错误（如引号未闭合等模型生成的残缺代码）
            # 退化为按普通空格分割
            return command.strip().split()

    def calc_sem(self, pred: str, ref: str) -> int:
        """
        1. S-EM (Stripped Exact Match)
        通过对比规范化后的 Token 列表，忽略格式和引号差异
        """
        pred_tokens = self.normalize_bash_to_tokens(pred)
        ref_tokens = self.normalize_bash_to_tokens(ref)

        return 1 if pred_tokens == ref_tokens else 0

    def calc_bleu(self, pred: str, ref: str) -> float:
        """
        2. Token-level BLEU (平替 CodeBLEU 的词法部分)
        基于规范化的 Token 计算 N-gram 相似度
        """
        pred_tokens = self.normalize_bash_to_tokens(pred)
        ref_tokens = self.normalize_bash_to_tokens(ref)

        if not pred_tokens or not ref_tokens:
            return 0.0

        # NLTK sentence_bleu 需要 ref 是 list of list
        score = sentence_bleu(
            [ref_tokens], pred_tokens, smoothing_function=self.smoothie
        )
        return float(score)  # type: ignore

    def calc_syntax_pass(self, pred: str) -> int:
        """
        3. Syntax Pass
        使用 bashlex 将字符串尝试构建为 AST，不报错即为语法合法
        """
        # 过滤掉空的预测
        if not pred.strip():
            return 0

        try:
            bashlex.parse(pred)
            return 1
        except bash_errors.ParsingError:
            return 0
        except NotImplementedError:
            return 1
        except Exception:
            return 0


def eval_nl2bash(batch_inference_fn, dataset: list, batch_size: int = 8) -> dict:
    evaluator = BashEvaluator()
    total_samples = len(dataset)
    metrics = {"S-EM": 0.0, "BLEU": 0.0, "Syntax_Pass_Rate": 0.0}

    # 按 batch_size 步进
    for i in tqdm(range(0, total_samples, batch_size)):
        batch_items = dataset[i : i + batch_size]
        prompts = [item["prompt"] for item in batch_items]
        ref_cmds = [item["ground_truth"] for item in batch_items]

        # 批量推理
        pred_cmds = batch_inference_fn(prompts)

        # 批量计算指标
        for pred_cmd, ref_cmd in zip(pred_cmds, ref_cmds):
            metrics["S-EM"] += evaluator.calc_sem(pred_cmd, ref_cmd)
            metrics["BLEU"] += evaluator.calc_bleu(pred_cmd, ref_cmd)
            metrics["Syntax_Pass_Rate"] += evaluator.calc_syntax_pass(pred_cmd)
    metrics["S-EM"] = (metrics["S-EM"] / total_samples) * 100
    metrics["BLEU"] = (metrics["BLEU"] / total_samples) * 100
    metrics["Syntax_Pass_Rate"] = (metrics["Syntax_Pass_Rate"] / total_samples) * 100
    return metrics


def generate(prompts, tokenizer, model, max_new_tokens, system_prompt=None):
    batch_messages = []
    for prompt in prompts:
        msg = []
        if system_prompt:
            msg.append({"role": "system", "content": system_prompt})
        msg.append({"role": "user", "content": prompt})
        batch_messages.append(msg)

    input_texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch_messages
    ]

    tokenizer.padding_side = "left"
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i in range(len(prompts)):
        # 只取出模型生成的新 token 部分（跳过 input 部分）
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[i][input_len:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if "<|im_start|>assistant" in decoded:
            decoded = decoded.split("<|im_start|>assistant")[-1]
        results.append(decoded.strip())

    return results


def evaluate(model_id: str, eval_conf, data_dict, data_dirs: List[str]):
    max_new_tokens = eval_conf.get("max_new_tokens", 128)
    batch_size = eval_conf.get("batch_size", 8)
    ds_test = data_dict["test"]
    default_sys = ds_test[0]["messages"][0]["content"]
    sys_content = eval_conf.get("system_prompt", default_sys)
    register_default_recipes()
    results_list = []
    for causal_conf_name in eval_conf["model"]:
        for lora in eval_conf["lora_dir"]:
            model_builder = registry.model_zoo[causal_conf_name]
            model = model_builder(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = PeftModel.from_pretrained(model, lora)
            model.eval()

            def model_inference_wrapper(prompts):
                return generate(prompts, tokenizer, model, max_new_tokens, sys_content)

            formatted_dataset = []
            for item in ds_test:
                formatted_dataset.append(
                    {
                        "prompt": item["messages"][1]["content"],
                        "ground_truth": item["messages"][-1]["content"],
                    }
                )

            metrics = eval_nl2bash(
                model_inference_wrapper, formatted_dataset, batch_size
            )
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "base_model": model_id,
                "data_dirs": data_dirs,
                "quant_config": causal_conf_name,
                "lora_name": lora.split("/")[-1],
                "lora_path": lora,
                **metrics,
            }
            results_list.append(record)
            print(
                f"\n[Progress] Evaluated: Config={causal_conf_name}, LoRA={record['lora_name']}"
            )
            print(
                f" >> S-EM: {metrics['S-EM']:.2f}% | BLEU: {metrics['BLEU']:.2f} | Syntax Pass: {metrics['Syntax_Pass_Rate']:.2f}%\n"
            )
    df = pd.DataFrame(results_list)

    print("\n" + "=" * 30 + " EVALUATION SUMMARY " + "=" * 30)
    display_cols = ["quant_config", "lora_name", "S-EM", "BLEU", "Syntax_Pass_Rate"]
    summary_df = df[display_cols].sort_values(by="S-EM", ascending=False)  # type: ignore
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("=" * 80)

    best_row = summary_df.iloc[0]
    print(
        f"\nBEST PERFORMER: {best_row['lora_name']} ({best_row['quant_config']}) with S-EM: {best_row['S-EM']:.2f}%"
    )
    eval_res_dir = Path("./eval_results")
    eval_res_dir.mkdir(exist_ok=True)
    output_path = f"{eval_res_dir}/{datetime.now().strftime('%m%d_%H%M')}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nAll evaluation finished, result stored to: {output_path}")
