import difflib
import json
import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import bashlex
import pandas as pd
import torch
from bashlex import errors as bash_errors
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

from quicktune import register_default_recipes, registry


class BashEvaluator:
    def __init__(self):
        self.smoothie = SmoothingFunction().method4

    def _clean(self, command: str) -> str:
        # 统一空白 & 引号
        cmd = command.strip()
        cmd = cmd.replace('"', "'")
        cmd = re.sub(r"\s+", " ", cmd)
        return cmd.strip()

    def normalize_bash_to_tokens(self, command: str) -> List[str]:
        command = self._clean(command)
        try:
            return shlex.split(command)
        except ValueError:
            return command.strip().split()

    def calc_sem(self, pred: str, ref: str) -> int:
        pred_tokens = self.normalize_bash_to_tokens(pred)
        ref_tokens = self.normalize_bash_to_tokens(ref)
        return 1 if pred_tokens == ref_tokens else 0

    def calc_bleu(self, pred: str, ref: str) -> float:
        pred_tokens = self.normalize_bash_to_tokens(pred)
        ref_tokens = self.normalize_bash_to_tokens(ref)
        if not pred_tokens or not ref_tokens:
            return 0.0
        score = sentence_bleu(
            [ref_tokens], pred_tokens, smoothing_function=self.smoothie
        )
        return float(score)  # type: ignore

    def calc_syntax_pass(self, pred: str) -> int:
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

    def calc_soft_em(self, pred: str, ref: str) -> float:
        """Token-level F1 (soft EM)"""
        pred_tokens = self.normalize_bash_to_tokens(pred)
        ref_tokens = self.normalize_bash_to_tokens(ref)
        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_set = pred_tokens
        ref_set = ref_tokens
        common = 0
        used = [False] * len(ref_set)

        for pt in pred_set:
            for i, rt in enumerate(ref_set):
                if not used[i] and pt == rt:
                    used[i] = True
                    common += 1
                    break

        precision = common / len(pred_set)
        recall = common / len(ref_set)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def calc_edit_similarity(self, pred: str, ref: str) -> float:
        """Char-level similarity (0~1)"""
        pred_clean = self._clean(pred)
        ref_clean = self._clean(ref)
        return difflib.SequenceMatcher(None, pred_clean, ref_clean).ratio()

    def calc_length_ratio(self, pred: str, ref: str) -> float:
        """Length ratio (pred/ref)"""
        pred_tokens = self.normalize_bash_to_tokens(pred)
        ref_tokens = self.normalize_bash_to_tokens(ref)
        if not ref_tokens:
            return 0.0
        return len(pred_tokens) / max(1, len(ref_tokens))

    def eval_all(self, pred: str, ref: str) -> Dict[str, float]:
        return {
            "S-EM": float(self.calc_sem(pred, ref)),
            "BLEU": self.calc_bleu(pred, ref),
            "Syntax_Pass": float(self.calc_syntax_pass(pred)),
            "Soft-EM_F1": self.calc_soft_em(pred, ref),
            "Edit_Sim": self.calc_edit_similarity(pred, ref),
            "Len_Ratio": self.calc_length_ratio(pred, ref),
        }


def eval_nl2bash(batch_inference_fn, dataset: list, batch_size: int = 8) -> dict:
    evaluator = BashEvaluator()
    total_samples = len(dataset)

    metrics = {
        "S-EM": 0.0,
        "BLEU": 0.0,
        "Syntax_Pass": 0.0,
        "Soft-EM_F1": 0.0,
        "Edit_Sim": 0.0,
        "Len_Ratio": 0.0,
    }

    for i in tqdm(range(0, total_samples, batch_size)):
        batch_items = dataset[i : i + batch_size]
        prompts = [item["prompt"] for item in batch_items]
        ref_cmds = [item["ground_truth"] for item in batch_items]

        pred_cmds = batch_inference_fn(prompts)

        for pred_cmd, ref_cmd in zip(pred_cmds, ref_cmds):
            res = evaluator.eval_all(pred_cmd, ref_cmd)
            for k in metrics:
                metrics[k] += res[k]

    for k in metrics:
        metrics[k] = (metrics[k] / total_samples) * 100

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


def evaluate_single(model_id, eval_conf, data_dict, data_dirs, causal_conf_name, lora):

    max_new_tokens = eval_conf.get("max_new_tokens", 128)
    batch_size = eval_conf.get("batch_size", 8)
    ds_test = data_dict["test"]
    default_sys = ds_test[0]["messages"][0]["content"]
    sys_content = eval_conf.get("system_prompt", default_sys)

    register_default_recipes()
    model_builder = registry.model_zoo[causal_conf_name]
    model = model_builder(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora)
    model.eval()

    def model_inference_wrapper(prompts):
        return generate(prompts, tokenizer, model, max_new_tokens, sys_content)

    formatted_dataset = [
        {
            "prompt": item["messages"][1]["content"],
            "ground_truth": item["messages"][-1]["content"],
        }
        for item in ds_test
    ]

    metrics = eval_nl2bash(model_inference_wrapper, formatted_dataset, batch_size)
    record = {
        "base_model": model_id,
        "data_dirs": data_dirs,
        "quant_config": causal_conf_name,
        "lora_name": lora.split("/")[-1],
        "lora_path": lora,
        **metrics,
    }
    return record


def _stable_json(x) -> str:
    """把 list/dict/str 等转成稳定的字符串，方便做去重 key。"""
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def _load_history(eval_res_dir: Path) -> pd.DataFrame:
    csvs = sorted(eval_res_dir.glob("*.csv"))
    if not csvs:
        return pd.DataFrame()

    dfs = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
            df["__source_csv"] = str(p)
            dfs.append(df)
        except Exception:
            continue

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _make_key(base_model: str, quant_config: str, lora_path: str, data_dirs) -> str:
    if isinstance(data_dirs, str):
        data_dirs_key = data_dirs
    else:
        data_dirs_key = _stable_json(list(data_dirs))
    return "||".join([base_model, quant_config, lora_path, data_dirs_key])


def evaluate(model_id: str, eval_conf, data_dict, data_dirs):
    max_new_tokens = eval_conf.get("max_new_tokens", 128)
    batch_size = eval_conf.get("batch_size", 8)

    ds_test = data_dict["test"]
    default_sys = ds_test[0]["messages"][0]["content"]
    sys_content = eval_conf.get("system_prompt", default_sys)

    register_default_recipes()

    eval_res_dir = Path("./eval_results")
    eval_res_dir.mkdir(exist_ok=True)

    hist_df = _load_history(eval_res_dir)
    done_keys = set()
    if not hist_df.empty:
        for _, r in hist_df.iterrows():
            try:
                base_model = str(r.get("base_model", ""))
                quant_config = str(r.get("quant_config", ""))
                lora_path = str(r.get("lora_path", ""))
                dd = r.get("data_dirs", "")
                done_keys.add(_make_key(base_model, quant_config, lora_path, dd))
            except Exception:
                continue

    new_records = []
    skipped = 0
    total = 0

    for causal_conf_name in eval_conf["model"]:
        for lora in eval_conf["lora_dir"]:
            total += 1
            lora_path = lora
            key = _make_key(model_id, causal_conf_name, lora_path, data_dirs)

            if key in done_keys:
                skipped += 1
                print(f"[Skip] already evaluated: {causal_conf_name} + {lora_path}")
                continue

            model_builder = registry.model_zoo[causal_conf_name]
            model = model_builder(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = PeftModel.from_pretrained(model, lora_path)
            model.eval()

            def model_inference_wrapper(prompts):
                return generate(prompts, tokenizer, model, max_new_tokens, sys_content)

            formatted_dataset = [
                {
                    "prompt": item["messages"][1]["content"],
                    "ground_truth": item["messages"][-1]["content"],
                }
                for item in ds_test
            ]

            metrics = eval_nl2bash(
                model_inference_wrapper, formatted_dataset, batch_size
            )

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "base_model": model_id,
                "data_dirs": _stable_json(list(data_dirs)),
                "quant_config": causal_conf_name,
                "lora_name": lora_path.split("/")[-1],
                "lora_path": lora_path,
                "max_new_tokens": max_new_tokens,
                "batch_size": batch_size,
                **metrics,
            }
            new_records.append(record)
            done_keys.add(key)  # 防止同一次 run 重复

            print(
                f"\n[Progress] Evaluated: Config={causal_conf_name}, LoRA={record['lora_name']}"
            )
            print(
                f" >> S-EM: {metrics['S-EM']:.2f}% | BLEU: {metrics['BLEU']:.2f} | "
                f"Syntax: {metrics['Syntax_Pass']:.2f}% | Soft-EM: {metrics['Soft-EM_F1']:.2f}% | "
                f"EditSim: {metrics['Edit_Sim']:.2f}% | LenRatio: {metrics['Len_Ratio']:.2f}%\n"
            )

    print(
        f"\n[Done] total combos={total}, skipped={skipped}, newly evaluated={len(new_records)}"
    )

    # ---------- 3) 只落盘新结果 ----------
    if not new_records:
        print("[Info] No new results. Nothing to write.")
        return

    new_df = pd.DataFrame(new_records)

    # 追加写入一个“本次增量”文件（推荐）
    out_path = eval_res_dir / f"{datetime.now().strftime('%m%d_%H%M')}_incremental.csv"
    new_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Info] Incremental results saved to: {out_path}")

    # ---------- 4) 打印汇总（历史 + 本次增量合并） ----------
    merged_df = pd.concat(
        [hist_df.drop(columns=["__source_csv"], errors="ignore"), new_df],
        ignore_index=True,
    )
    display_cols = [
        "quant_config",
        "lora_name",
        "S-EM",
        "BLEU",
        "Syntax_Pass",
        "Soft-EM_F1",
        "Edit_Sim",
        "Len_Ratio",
    ]
    summary_df = merged_df[display_cols].sort_values(by="S-EM", ascending=False)  # type: ignore
    print("\n" + "=" * 30 + " EVALUATION SUMMARY (ALL) " + "=" * 30)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("=" * 80)
