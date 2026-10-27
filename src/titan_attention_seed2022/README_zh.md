# TITAN-BBB：机制条件注意力与完整消融，神经网络 seed=42

包内代码目录名沿用上一版 `titan_attention_seed2022`，实际默认 seed 已改为 **42**；结果目录按 `--seed` 自动命名。

本包回到你原来的 **TITAN 分类数据**：train=7593、val=727、test=942。沿用原划分和 TITAN 标签，不做标签反转。三维分数复用 `mechanism_features/`，其来源仍是之前排除了 TITAN 重叠的三个固定辅助模型。辅助模型和分子特征提取器不重新训练。

## 1. 上传与运行

把整个 `titan_attention_seed2022` 文件夹上传到：

```
/root/project/titanBBB/TITAN-BBB-main/src/titan_attention_seed2022/
```

目录应为：

```
TITAN-BBB-main/
  dataset/                         原 TITAN CSV
  features/                        原三模态 .npy
  mechanism_features/              原三维机制分数、audit、manifest.json
  src/
    utils/model.py                 作者原模型
    utils/utils.py                 作者原 seed 函数
    titan_attention_seed2022/
      __init__.py
      common.py
      model.py
      trainer.py
      run_titan_attention.py
      README_zh.md
```

在服务器终端执行：

```bash
conda activate titan_bbb
cd /root/project/titanBBB/TITAN-BBB-main/src
python titan_attention_seed2022/run_titan_attention.py --project-root .. --seed 42
```

`--project-root ..` 是从当前 `src` 指向上一层 `TITAN-BBB-main`。默认 seed 已是 42，命令仍显式写出便于核对。

只核对数据、不训练：

```bash
python titan_attention_seed2022/run_titan_attention.py --project-root .. --check-only
```

已完成的组继续使用、中断组从头重新训练：

```bash
python titan_attention_seed2022/run_titan_attention.py --project-root .. --seed 42 --resume
```

`--resume` 不是恢复中断 epoch 的优化器状态。程序校验已完成组的配置、验证选择锁和检查点后复用；未完成的组从相同 seed 开始下一次 attempt。更改输入、代码、环境或设置后应另设 `--output-dir`。

如果分数目录改过名字，追加 `--mechanism-dir /实际路径`。此目录必须包含原生成器生成的 `manifest.json`、三个 `.npy` 和对应 audit CSV；仅有三个裸数组不够。代码不需要网络下载。

## 2. 默认比较十组

| experiment | 外排 | 摄取 | PAMPA | 结构 |
|---|:---:|:---:|:---:|---|
| baseline | — | — | — | 原 TITAN，头输入2048维 |
| zero_control | 0 | 0 | 0 | 新机制注意力＋2051维头 |
| efflux_only | ✓ | 0 | 0 | 新机制注意力＋2051维头 |
| influx_only | 0 | ✓ | 0 | 新机制注意力＋2051维头 |
| pampa_only | 0 | 0 | ✓ | 新机制注意力＋2051维头 |
| efflux_influx | ✓ | ✓ | 0 | 新机制注意力＋2051维头 |
| efflux_pampa | ✓ | 0 | ✓ | 新机制注意力＋2051维头 |
| influx_pampa | 0 | ✓ | ✓ | 新机制注意力＋2051维头 |
| full_fusion | ✓ | ✓ | ✓ | 新机制注意力＋2051维头 |
| late_concat | ✓ | ✓ | ✓ | 旧注意力＋2051维头，仅末端拼接 |

前九行是完整九组消融，最后一行是旧融合结构的补充对照。十组都在 TITAN 原训练集从头训练，神经网络 seed=42，不复用之前的神经网络检查点。

✓ 使用该机制的真实固定分数，0 表示固定零列。**同一个掩码同时用于注意力上下文与末端预测头**，不会只从一条路径中移除机制。所有消融组都保留三个位置和相同结构，不删除/交换列，不改变输入维数。训练、验证、测试均使用同一组掩码。

例如 `efflux_only` 使用 `[p_efflux,0,0]`；此向量同时进入 `U m` 和最终2051维拼接。`zero_control` 的 U 及末端三列仍为可训练参数，但输入恒为零，无法携带机制信息。

每个消融组都独立重训、独立早停、独立在验证集选择阈值。它们不是只在已训练全融合模型上临时把分数置零。`late_concat` 是原来的末端拼接方法；`full_fusion` 在这个方法上增加机制条件注意力。

仅运行新模型也可以：

```bash
python titan_attention_seed2022/run_titan_attention.py --project-root .. --seed 42 \
  --groups full_fusion --output-dir ../titan_attention_only_seed42_results
```

## 3. 新注意力如何计算

三种模态仍分别投影至 2048 维，得到 `h_tab`、`h_img`、`h_txt`。机制向量 `m=[p_efflux,p_influx,p_pampa]` 的顺序固定。

原注意力：

```
e_j = vᵀ tanh(W h_j + b)
```

新注意力：

```
c   = U m                           U：Linear(3,2048)，无偏置
e_j = vᵀ tanh(W h_j + b + c)
α   = softmax([e_tab,e_img,e_txt])
h   = α_tab*h_tab + α_img*h_img + α_txt*h_txt
预测 = MLP(concat(h,m))              2051→2048→1
```

`c` 放在 **Tanh 之前**。如果给三个最终注意力分数加同一个标量再做 softmax，该标量会抵消，不能实现本需求。这里非线性层使机制上下文能够改变不同模态的相对分数。

这不是新增第四个注意力 token。注意力权重仍有三个，分别对应表格、图像、文本；机制分数负责调节这三个权重，也继续进入末端预测头。

新参数 U 初始化为零，末端增加的三列也按上一版初始化为零。所有公共参数与同 seed 基线初始化对齐，新增层构造后恢复 CPU RNG，避免额外随机初始化改变后续 dropout/loader 的随机序列。U 可以通过注意力路径获得梯度；程序在训练前执行真实 Torch 梯度与敏感性检查，临时探针参数会完整恢复。

十组的所有非新增训练规则保持一致：

- 三模态 StandardScaler 仅拟合训练集；机制分数保持原 0～1 值。
- 投影维度 2048、dropout=0.1、batch=32。
- AdamW，lr=1e-4、weight_decay=0.01。
- 训练集负类/正类比作为 BCEWithLogitsLoss 的 pos_weight。
- 最多 1000 epoch；验证集加权 BCE 连续 10 次无改善早停，重载最佳检查点。
- 每组在验证集从 0.1、0.3、0.5、0.7、0.9 选择 BACC 最优阈值，平局取较小阈值，阳性规则严格为 `p > threshold`。
- 十组全部锁定检查点和阈值后才评估测试集。另报告固定 0.5 和 0.7 的补充结果，保留自动选择阈值的原结果。这里的自动选择是指定五个候选值的网格选择，不是任意连续阈值搜索。

原 TITAN 划分中已报告的标准化身份重叠（train/val=1、train/test=2）随原划分保留，本代码不重新划分。训练数据、特征和旧结果均不覆盖。

## 4. 输出

默认输出：

```
/root/project/titanBBB/TITAN-BBB-main/titan_attention_ablation_seed42_results/
```

- `test_comparison.csv`、`val_comparison.csv`：各组验证选阈值结果。
- `test_fixed_threshold05.csv`、`test_fixed_threshold07.csv`，以及对应 val 文件：固定阈值补充结果。
- `test_all_thresholds.csv`、`val_all_thresholds.csv`：保留所选阈值并追加不同的固定阈值，避免重复行。
- `comparison.csv`：验证和测试完整指标，包含 AUROC、AP、ACC、BACC、SN、SP、F1、MCC 和混淆矩阵。
- `ablation_table.md`：十组验证/测试表；`contrast_summary.csv`：预先指定的差值比较，包括单机制vs零机制、全融合vs移除某项机制、全融合vs旧末端拼接。差值为描述性结果，不声称统计显著。
- `mechanism_score_summary.csv`：各划分机制分数分布，不读取测试标签选择参数。
- `experiments/<组名>/attempt_001/<mode>_seed42/`：检查点、配置、`training.log`、`history.csv`、`selection_lock.json`、逐样本预测及 `val_attention.csv` / `test_attention.csv`。
- 各新注意力组的 `val_context_diagnostic.csv`：固定该组已训练模型，比较真实上下文、关闭注意力上下文、置换注意力上下文。末端头仍使用该组掩码后的分数，因此只干预注意力路径。这些只是诊断，不用于重新选模型、种子或阈值。
- `val_attention.csv` / `test_attention.csv` 中 `p_efflux/p_influx/p_pampa` 是该组真正使用的掩码后分数，`raw_p_*` 是对应原始缓存分数。注意区分，避免把已屏蔽列误认为仍在使用。
- 新注意力组的 `learned_context_summary.json`：U 的列范数及权重顺序。参数范数、注意力权重本身均不等同于机制的因果重要性。
- `run_manifest.json`、`all_selections_locked.json`、`titan_modality_scalers.joblib`：输入、来源、设置、选择锁和预处理记录。

分类模型的输出是 BBB 阳性 logit/probability；本包不以二分类标签计算回归 MAE/MSE/RMSE。

更新版默认使用新的结果目录，上一版三组的 `titan_attention_seed2022_results` 不覆盖，也不自动复用。旧名称 `--groups mechanism_attention` 仍接受并映射到 `full_fusion`。`--resume` 仅适用于本次更新版创建且输入/源码/设置均一致的结果目录。

## 5. 验证范围

交付环境没有 Torch/CUDA，因此没有在这里进行真实训练，也没有生成新的性能数字。本地进行了实际源码的 NumPy 前向、有限差分敏感性、数据对齐/训练集专用标准化和替代训练后端流程检查。服务器启动时另执行真实 Torch 前向/反向与机制注意力敏感性检查。新结构是否改善泛化，需以这轮训练结果判断。
