# SmartMan

SmartMan 是一个从微调到落地的 NL2Bash 项目，提供完整的微调训练链路、评测流程，以及 Rust 实现的本地命令行客户端。项目核心目标是将自然语言描述稳定转换为可执行的 Bash 命令，并支持训练、评估与本地推理使用。

## 核心特性

- **QuickTune 微调框架**：提供轻量且可扩展的微调注册与配方机制（模型、LoRA、SFT 配置统一管理）。
- **微调 Pipeline**：包含数据准备、训练、评估、模型合并的完整流程。
- **smartman-cli 客户端**：Rust 实现的交互式 CLI，可连接本地 LLM 引擎并在终端内进行 NL2Bash 交互。

## 目录结构

```
.
├── src/
│   ├── quicktune/          # QuickTune 微调框架
│   └── smartman/           # 训练/评估 pipeline
├── scripts/                # 数据处理、微调、评测、合并脚本
├── configs/                # 训练/评估配置示例
├── data/                   # 数据目录（原始与处理后）
└── smartman-cli/           # Rust CLI 客户端
```

## 环境准备

推荐使用 Conda 环境：

```bash
conda env create -f environment.yaml
conda activate smartman
```

或直接使用 pip：

```bash
pip install -r requirements.txt
pip install -e .
```

## QuickTune 微调框架

QuickTune 通过注册表机制管理模型、LoRA 与 SFT 配置：

- `src/quicktune/core/manager.py` 提供统一注册入口。
- `src/quicktune/recipes/` 目录下自动加载可用的配方。

训练时通过配置文件引用 recipe，实现快速组合与复用。

## 微调 Pipeline

训练入口在 `src/smartman/main.py`，支持 `train` / `eval` / `all` 三种模式。

### 配置示例

参考 `configs/example.yaml`，定义：

- 基座模型 `model_id`
- 训练/评估数据路径 `data_dirs`
- QuickTune `recipe`
- 评估配置 `eval_conf`

### 训练与评估

```bash
python -m smartman.main --config configs/example.yaml --mode train
python -m smartman.main --config configs/example.yaml --mode eval
python -m smartman.main --config configs/example.yaml --mode all
```

### 关键脚本

- `scripts/nl2bash.py`：构建英文 NL2Bash 数据集
- `scripts/make_chinese_data.py`：生成中文 NL2Bash 训练数据
- `scripts/merge_data.py`：合并中英文训练集
- `scripts/tune.py`：独立微调脚本示例
- `scripts/test_model.py`：推理测试与 LoRA 加载测试
- `scripts/merge_model.py`：合并 LoRA 权重到基座模型

## 评估指标

评估流程支持多种指标：

- S-EM（严格匹配）
- BLEU
- Syntax Pass
- Soft-EM F1
- Edit Similarity
- Length Ratio

评估结果会保存到 `eval_results/` 目录。

## smartman-cli（Rust 客户端）

smartman-cli 提供本地交互式终端体验，默认连接本地 `llamafile --server` 服务。

推荐从 GitHub Release 获取对应系统的预编译版本进行安装,详见release页面。Release 包含一个默认的 `llamafile-thin` 引擎（体积小，CPU-only），也可以从 llamafile 官方仓库的 Release 下载完整引擎（支持更多算子/硬件），再通过 CLI 的安装命令接入。量化后的 GGUF 模型可从本仓库的 Release（ModelRelease）下载并安装使用。

引擎与模型安装（使用 CLI 安装到本地 assets）：

```bash
# 安装完整 llamafile 引擎（来自 llamafile 官方 Release）
smartman-cli install engine /path/to/llamafile --link

# 安装量化 GGUF 模型（来自本仓库 ModelRelease）
smartman-cli install model /path/to/model.gguf
```

### 交互使用

- 输入自然语言指令，返回 Bash 命令
- 支持直接执行、复制到剪贴板
- 通过 `!` 前缀执行本地 shell 命令

## License

MIT
