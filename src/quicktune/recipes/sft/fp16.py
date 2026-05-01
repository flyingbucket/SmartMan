"""命名规则: <精度>/<风格>"""

from trl.trainer.sft_config import SFTConfig
from ...core import registry


@registry.register_sft("fp16/stable")
def buildsft_fp16(output_dir: str, **kwargs):
    training_args = SFTConfig(
        output_dir=output_dir,
        fp16=True,  # 明确使用 fp16
        learning_rate=float(kwargs.get("lr", 5e-5)),  # 全参或普通LoRA通常学习率略低
        lr_scheduler_type="cosine",
        per_device_train_batch_size=kwargs.get("batch_size", 8),
        gradient_accumulation_steps=kwargs.get("grad_acc", 2),
        max_length=kwargs.get("max_length", 512),
        num_train_epochs=kwargs.get("n_epochs", 3),
        packing=kwargs.get("packing", False),
        report_to="tensorboard",
        logging_dir=f"{output_dir}/runs",
    )
    return training_args
