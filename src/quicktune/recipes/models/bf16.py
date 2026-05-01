"""命名规则: <精度>/<风格>"""

import torch
from transformers import AutoModelForCausalLM
from ...core import registry


@registry.register_model("bf16")
def build_model_bf16(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    return model


@registry.register_model("bf16/eager")
def build_model_bf16_eager(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    return model


@registry.register_model("bf16/remote_code")
def build_model_bf16_remote(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    return model
