# 环境与依赖安装

将本目录中的文件按原目录结构复制到 TITAN/Deep-B3 仓库根目录即可。本依赖配置兼容本会话中的原TITAN整合包和Deep-B3适配包，不修改数据、训练配置、结构规则或实验结果。

## 支持环境

- Linux x86_64 / NVIDIA GPU；宿主机或容器已能访问GPU，NVIDIA驱动可用。
- 已安装Conda（Miniconda、Miniforge或Anaconda）。
- Python **3.10.18**；PyTorch **2.7.1+cu118**；torchvision **0.22.1+cu118**。
- 全部18项核心包的版本以 `requirements.txt` 为准，原整合包会严格校验这些版本。
- 环境包安装需要访问conda-forge、PyPI和PyTorch官方wheel源。

## 安装

以下命令均在**仓库根目录**执行。已有同名环境时，使用新的环境名，不要删除或覆盖正在运行的环境。

```bash
# 1. 创建Python与系统运行库环境。
conda env create -f environment.yml
conda activate titan_bbb

# 2. 先从PyTorch官方源安装指定CUDA构建。
python -m pip install -r requirements-torch-cu118.txt

# 3. 安装并固定完整核心依赖及实际用到的间接依赖。
python -m pip install -r requirements.txt

# 4. 检查包依赖、版本、关键功能与CUDA。
python -m pip check
python scripts/check_environment.py
```

如果已有 `titan_bbb`，可将上面的创建与激活两行换成：

```bash
conda env create -f environment.yml -n titan_bbb_new
conda activate titan_bbb_new
```

安装CUDA版PyTorch和完整依赖是两个步骤：`requirements.txt`锁定了`+cu118`版本，预先从官方源安装可以避免误装其他CUDA构建或CPU构建。第二步的临时依赖版本可能被第三步按原环境记录重新解析，最终以全部命令执行成功、`pip check`和检查脚本通过为准。不要在执行到一半时启动实验。

`environment.yml`负责Python和Linux运行库，pip负责Python工具包；没有把同一个PyTorch同时交给Conda和pip安装。文件里没有原服务器的绝对`prefix`路径。

## 文件用途

| 文件 | 用途 |
|---|---|
| `environment.yml` | 创建Python3.10.18及Linux C++/OpenMP运行库 |
| `requirements-torch-cu118.txt` | 从官方源取得指定CUDA版torch和torchvision |
| `requirements.txt` | 锁定run_pipeline.py校验的18个核心包，适用于本项目的复现安装 |
| `constraints.txt` | 对实际依赖到的间接包使用原环境记录的版本；不会主动安装所有条目 |
| `.python-version` | 给pyenv和编辑器标记Python版本，不替代Conda环境创建 |
| `scripts/check_environment.py` | 检查18包、Python、分子处理、原生LightGBM、torchvision算子、Roberta、Excel及CUDA，输出environment_check.json |
| `build_titan_bbb_env.py` | 可选的一键创建/检查入口，替代上面的手工安装步骤 |

检查脚本会做很小的功能测试，不读取实验数据，不训练正式模型，不下载预训练权重。GPU不可用时会明确失败，因为当前完整流程的特征提取要求CUDA。

可选一键入口（不需要先创建Conda环境）：

```bash
python build_titan_bbb_env.py --name titan_bbb
```

一键入口在发现已有同名环境时只检查，不自动改包。它会将两个CUDA wheel先从官方源取到本地，再和其他核心包一起按版本约束安装，最终执行环境检查。手工步骤和一键入口二选一即可。

## 模型资产

环境工具包与预训练权重分开管理：

- ChemBERTa：`pretrained/ChemBERTa-100M-MLM/`，保持整合包要求的完整离线文件。
- ResNet50：Torch缓存中的 `checkpoints/resnet50-0676ba61.pth`，也可按流程参数指定本地权重。
- NVIDIA驱动由服务器/容器宿主环境提供，pip安装不替代驱动安装。

不要把预训练权重、数据和实验输出追加到requirements文件。不要直接用开发机器的完整`pip freeze`覆盖本项目的核心依赖清单。原环境记录用于提供可审计的版本约束，不能保证不同驱动/GPU/系统运行库下数值逐字节一致。

## 验证范围

文件结构、18项版本一致性、requirements与constraints一致性、YAML语法和检查脚本语法已在本地验证。当前本地运行环境不是目标CUDA环境，没有在这里重新完成全部依赖下载和GPU安装；正式运行前请在目标服务器执行最后两条检查命令。

## 官方参考

- Conda环境文件：https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
- pip requirements格式：https://pip.pypa.io/en/stable/reference/requirements-file-format/
- PyTorch2.7.1 CUDA11.8安装：https://pytorch.org/get-started/previous-versions/#v271
