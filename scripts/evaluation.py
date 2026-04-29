import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
import difflib

# =========================
# CONFIG
# =========================
base_model_path = "./base_models/Qwen2.5-1.5B-Instruct"
lora_path = "./output/qwenInstruct0429_1852"
data_dir = "data/processed/nl2bash"

max_new_tokens = 64


# =========================
# LOAD MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, lora_path)
model.eval()


# =========================
# UTIL FUNCTIONS
# =========================


def extract_bash(text):
    """
    clean model output
    """
    text = text.strip()

    # remove markdown
    text = re.sub(r"```.*?```", "", text, flags=re.S)

    # remove extra whitespace
    return text.strip()


def normalize(cmd):
    """
    normalize bash for comparison
    """
    cmd = cmd.strip()

    # unify quotes
    cmd = cmd.replace('"', "'")

    # remove multiple spaces
    cmd = re.sub(r"\s+", " ", cmd)

    return cmd.strip()


def exact_match(pred, gold):
    return normalize(pred) == normalize(gold)


def similarity(pred, gold):
    return difflib.SequenceMatcher(None, pred, gold).ratio()


# =========================
# INFERENCE
# =========================


def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(  # type: ignore
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=False)

    # extract assistant output
    if "<|im_start|>assistant" in decoded:
        decoded = decoded.split("<|im_start|>assistant")[-1]

    return extract_bash(decoded)


# =========================
# EVALUATION LOOP
# =========================
def main():
    results = []

    exact_matches = 0
    total = 0
    similarities = []
    length_errors = 0
    dd = load_dataset("json", data_dir=data_dir, data_files={"test": "test.jsonl"})
    ds_test = dd["test"]
    for item in tqdm(ds_test, total=len(ds_test)):
        prompt = item["messages"][1]["content"]  # type: ignore
        gold = item["messages"][-1]["content"]  # type: ignore

        pred = generate(prompt)

        em = exact_match(pred, gold)
        sim = similarity(pred, gold)

        if em:
            exact_matches += 1

        similarities.append(sim)

        # stop behavior check
        if len(pred.split()) > len(gold.split()) * 1.5:
            length_errors += 1

        results.append(
            {
                "prompt": prompt,
                "gold": gold,
                "pred": pred,
                "em": em,
                "sim": sim,
            }
        )

        total += 1

    # =========================
    # SUMMARY
    # =========================

    print("\n================ RESULTS ================\n")

    print(f"Total samples: {total}")
    print(f"Exact Match Accuracy: {exact_matches / total:.4f}")
    print(f"Avg Similarity: {sum(similarities) / total:.4f}")
    print(f"Over-length outputs: {length_errors / total:.4f}")

    # save results
    with open("eval_results/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to eval_results.json")


if __name__ == "__main__":
    main()
