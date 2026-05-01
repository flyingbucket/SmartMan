"""命名规则: <量化方式>/<风格>"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from ...core import registry


@registry.register_model("bnb/4bit")
def build_model_4bit(model_id: str):
    """标准的 QLoRA 4-bit 加载配置"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # 使用 NormalFloat4
        bnb_4bit_use_double_quant=True,  # 二次量化，进一步省内存
        bnb_4bit_compute_dtype=torch.bfloat16,  # A30 建议使用 bf16 计算
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    return model
