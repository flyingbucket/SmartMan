"""命名规则: <精度>/<风格>"""

from trl.trainer.sft_config import SFTConfig
from ...core import registry


@registry.register_sft("bf16/flexible")
def build_sft_flexible(output_dir: str, **kwargs):
    default_args = {
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.0,
        "num_train_epochs": 3,
    }
    default_args.update(kwargs)

    float_fields = ["learning_rate", "weight_decay", "warmup_ratio", "warmup_steps"]
    int_fields = [
        "num_train_epochs",
        "max_length",
        "eval_steps",
        "save_steps",
        "per_device_train_batch_size",
    ]

    for field in float_fields:
        if field in default_args:
            default_args[field] = float(default_args[field])

    for field in int_fields:
        if field in default_args:
            default_args[field] = int(default_args[field])

    return SFTConfig(output_dir=output_dir, **default_args)


@registry.register_sft("bf16/stable")
def buildsft_bf16(
    output_dir: str,
    lr: float = 2e-4,
    n_epochs: int = 5,
    max_length: int = 512,
    packing: bool = True,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
):
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(lr),
        num_train_epochs=n_epochs,
        max_length=max_length,
        packing=packing,
        bf16=True,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        push_to_hub=False,
        dataset_text_field="messages",
        # 自动化最佳模型选择
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_total_limit=3,
        # tensorboard
        report_to="tensorboard",
        logging_dir=f"{output_dir}/runs",
    )
    return training_args


@registry.register_sft("bf16/stable-epoch_wise")
def build_sft_epoch_wise(
    output_dir: str, lr: float = 1e-4, n_epochs: int = 5, **kwargs
):
    """
    专门按 Epoch 评估和保存的配置，适合 NL2Bash 等精密任务
    """
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=float(lr),
        num_train_epochs=n_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        # 自动化最佳模型选择
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_total_limit=3,
        bf16=True,
        packing=True,
        dataset_text_field="messages",
        **kwargs,
    )
    return training_args
