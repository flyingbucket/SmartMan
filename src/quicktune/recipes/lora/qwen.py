"""命名规则: <架构>/<微调层>"""

from peft import LoraConfig
from ...core import registry


@registry.register_lora("qwen/all_linear")
def build_lora_all_linear(r: int = 16, lora_dropout: float = 0.05):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=2 * r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return lora_config


@registry.register_lora("qwen/minimal")
def build_lora_minimal(r: int = 8, lora_dropout: float = 0.05):
    """最节省参数的方案，仅更新 Q 和 V"""
    return LoraConfig(
        r=r,
        lora_alpha=2 * r,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


@registry.register_lora("qwen/attention_only")
def build_lora_attention(r: int = 16, lora_dropout: float = 0.05):
    """专注提升模型对输入指令的解析能力"""
    return LoraConfig(
        r=r,
        lora_alpha=2 * r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


@registry.register_lora("qwen/mlp_only")
def build_lora_mlp(r: int = 16, lora_dropout: float = 0.05):
    """适合需要模型背诵更多 Bash 命令选项的场景"""
    return LoraConfig(
        r=r,
        lora_alpha=2 * r,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
