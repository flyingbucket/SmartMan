from typing import Callable, Dict, TypeAlias

from peft import LoraConfig
from transformers import PreTrainedModel
from trl.trainer.sft_config import SFTConfig

ModelBuilder: TypeAlias = Callable[..., PreTrainedModel]
LoraBuilder: TypeAlias = Callable[..., LoraConfig]
SFTBuilder: TypeAlias = Callable[..., SFTConfig]


class TuneRegistry:
    def __init__(self) -> None:
        self.model_zoo: Dict[str, Callable] = {}
        self.lora_zoo: Dict[str, Callable] = {}
        self.sft_zoo: Dict[str, Callable] = {}

    def register_model(self, name: str):
        def wrapper(builder: ModelBuilder) -> ModelBuilder:
            if name in self.model_zoo:
                raise ValueError(f"Model {name} is already registered")
            self.model_zoo[name] = builder
            return builder

        return wrapper

    def register_lora(self, name: str):
        def wrapper(builder: LoraBuilder) -> LoraBuilder:
            if name in self.lora_zoo:
                raise ValueError(f"Lora config {name} is already registered")
            self.lora_zoo[name] = builder
            return builder

        return wrapper

    def register_sft(self, name: str):
        def wrapper(builder: SFTBuilder) -> SFTBuilder:
            if name in self.sft_zoo:
                raise ValueError(f"SFT config {name} is already registered")
            self.sft_zoo[name] = builder
            return builder

        return wrapper


registry = TuneRegistry()
