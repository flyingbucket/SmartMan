"""命名规则: <量化方式>/<风格>"""

from transformers import AutoModelForCausalLM
from ...core import registry


@registry.register_model("int8")
def build_model_8bit(model_id: str):
    """标准的 8-bit 加载 (通常用于推理或不追求极致内存占用)"""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    return model
