# 当前版本：恢复原结构筛选协议

默认结构配置已改为 `legacy_review_v1`，请先阅读 `LEGACY_RESTORE_CN.md`。它恢复原主要身份、家族内校正和S1–S5情景，重放明确保存的历史审查角色；连接合并后显著性不再是核心组门槛。可用 `--reuse-upstream-from` 在核验后复用已完成上游，避免再次GPU训练。

以下是上一版自动协议说明；涉及默认自动分层的描述请以 `LEGACY_RESTORE_CN.md` 为准。

# TITAN-BBB：原始数据、当前结构发现、四组权重搜索与消融

当前默认入口是 `run_pipeline.py --mode rebuild --anchor-policy refit --structure-source discover`。先阅读 `STRUCTURE_DISCOVERY_CN.md` 中的新自动候选标准；它与此前人工审查的 V01/V03 分组不是同一套筛选流程，新的结构与性能可能不同。

## 依赖和模型

使用原 `titan_bbb` 环境。原始 TITAN 分类数据、Cornelissen 原始表、S2清洗与身份去重、三个辅助模型训练代码均在包内。

ChemBERTa 整个目录放在 `pretrained/ChemBERTa-100M-MLM/`，默认从包内读取。ResNet50 继续使用已确认的 Torch 缓存 `checkpoints/resnet50-0676ba61.pth`，也可用 `--resnet-weights` 指定本地文件。PyTorch 安装本身不包含这份权重；本包不联网下载权重。

环境版本、CUDA与原始输入文件校验仍沿用原包要求。默认重训不需要旧实验模型、旧特征缓存、旧搜索目录或包外结构分析脚本。

## 运行

在整合包根目录执行：

```bash
conda activate titan_bbb
python run_pipeline.py --output /root/project/titanBBB/reproduction_with_discovery_seed42
```

结果目录必须是整合包和原项目之外的新目录。更新代码后不能续跑旧版本结果。新版本运行中断后，重复同一命令并加 `--resume`；代码、环境、设备、候选配置和运行选项不能改变。

可以先运行到结构发现：

```bash
python run_pipeline.py --output /root/project/titanBBB/reproduction_with_discovery_seed42 --stage structures
```

确认本次输出的候选及审计后，使用相同输出位置续跑后续阶段：

```bash
python run_pipeline.py --output /root/project/titanBBB/reproduction_with_discovery_seed42 --resume
```

## 默认流程

1. `prepare`：校验并准备包内原始数据和代码。
2. `exclusion`：S2清洗、统一标准化及辅助池与TITAN身份去重。
3. `auxiliary`：在辅助池内部选择轮数，训练三个辅助模型，生成固定原始机制分数。
4. `modalities`：生成表格、分子图像和文本特征。
5. `fusion`：训练未输入机制分数的神经baseline，冻结并抽取H2048。
6. `structures`：以H2048+M3训练未加结构权重的LightGBM；仅用其验证集预测进行结构发现、FDR校正、敏感性分析、重叠处理和训练覆盖审计，生成本次候选规则。
7. `stage2`：四种机制组合分别选择两组结构权重，锁定验证结果后测试，完成四套九机制消融。

Stage 1 使用原始种子2026，神经baseline和Stage 2使用42；沿用固定类别权重和既有LightGBM参数。机制分数不进入神经attention，保留原值与H2048拼接。

## 输出

- `structure_discovery/`：本次未加权LightGBM、验证集TP/TN/FP/FN、全部检验、敏感性、候选决策、规则定义、训练覆盖、成员、重叠、Excel与SVG。
- `stage2/`：本次四组搜索、验证选择记录、权重、测试结果。
- `ablation/`：36个模型、72行阈值策略结果、逐分子预测、错误转换、Excel与SVG。
- `historical_comparison.json`：历史中间输出差异；默认只报告。
- `ablation/reported_result_verification.json`：本次最终结果与历史六位小数的差异。

搜索默认每个模型200个不同权重组合，前40个为锚点和随机初始点。权重1–3，步长0.05，采用Optuna4.5.0的三目标TPE。若某组没有符合标准的结构，其权重固定为1，实际搜索空间和预算相应缩减，不强行补入旧结构。也可预先设置 `--search-budget` 和 `--search-startup`。

## 历史兼容路线

`--structure-source fixed` 才会显式沿用此前固定规则。
`--strict-history` 才把历史输出差异恢复成停止条件。
`--anchor-policy archived` 需要 `--structure-source fixed` 并提供旧模型所在的 `--assets-root`。
`--mode replay` 始终是独立标记的历史模型回放，需要旧模型与旧特征，并严格核对历史结果。

普通新流程不应直接调用保留的旧 `run_priority_bayes.py` 等历史入口。请统一使用 `run_pipeline.py`。

## 验证与局限

当前结构模块经过真实RDKit、SciPy、LightGBM和Optuna的合成数据测试，覆盖新候选传递、无候选、空分组、36组消融、72行导出和续跑。测试特征与结果仅是软件测试，不是BBB性能估计。尚未在此执行真实GPU特征提取、原数据神经重训或完整800组合搜索。

同一验证集用于结构发现和权重选择，属于模型开发。原测试集已多次查看，不能据此声称独立泛化验证或因果机制发现。本包不声称自动完成了此前的人工化合物身份核查和文献审查。
