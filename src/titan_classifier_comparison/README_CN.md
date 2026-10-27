# TITAN：神经网络预测头与 LightGBM 的四组对照

默认 **TITAN 原分类数据划分、原标签、seed=42**。本包只做分类，
不改变已完成的回归实验，也不采用 DeepB3 的标签转换。

## 1. 这次具体比较什么

从你已经训练好的 **原始 baseline（预测头输入为2048维）** 中，
抽取注意力加权汇总后的2048维向量 H。抽取时关闭 Dropout，冻结整个模型。
四组共用同一份 H，再训练各自的分类器：

| experiment | 分类器输入 | 新训练的分类器 |
|---|---|---|
| nn_baseline | H，2048维 | 神经网络：2048→2048→1 |
| nn_mechanism | H + 外排/摄取/PAMPA分数，2051维 | 神经网络：2051→2048→1 |
| lightgbm_baseline | H，2048维 | LightGBM |
| lightgbm_mechanism | H + 外排/摄取/PAMPA分数，2051维 | LightGBM |

这里的两个神经网络预测头是**重新训练**的；投影层和注意力层始终冻结。
所以 `nn_baseline` 不必等于你之前端到端训练 TITAN 得到的0.933731。
原 baseline 检查点的结果另存为 `original_baseline_reference_only.csv`，便于核对，
但不混入这四行对照。

这次机制分数仅追加到分类器输入，**不参与注意力权重计算**。
这样四组的上游表征完全一致，便于判断变化来自分类器还是机制输入。
这与前一次“机制参与注意力”的实验是两个不同的问题。

LightGBM 训练不会把梯度传回神经网络，因此不能像原神经网络预测头一样联合更新注意力。
本实验回答的是“在同一套已学好的 TITAN 表征上，换分类器是否更有效”。
它不能证明 LightGBM 在重新设计的所有特征方案中都优于神经网络。
MechBBB 使用 LightGBM，但这里的输入特征和训练设计并非完整复刻 MechBBB。

## 2. 上传到哪里

在本地解压后，把整个 `titan_classifier_comparison` 文件夹通过 MobaXterm 的 SFTP
上传到服务器的：

```text
/root/project/titanBBB/TITAN-BBB-main/src/
```

上传后入口文件应该位于：

```text
/root/project/titanBBB/TITAN-BBB-main/src/titan_classifier_comparison/run_classifier_comparison.py
```

不要覆盖原来的 `train.py`、`utils/model.py` 或此前的实验文件。

## 3. 开始运行

在服务器终端中输入：

```bash
conda activate titan_bbb
cd /root/project/titanBBB/TITAN-BBB-main/src
python titan_classifier_comparison/run_classifier_comparison.py --project-root .. --seed 42
```

`--project-root ..` 表示项目根目录是当前 `src` 的上一层。
默认自动使用 CUDA；LightGBM 在 CPU 上训练，使用4个线程。
本包沿用当前环境，需要 torch、lightgbm、numpy、pandas、scikit-learn、joblib。
无需重新创建环境或下载 ChemBERTa、ResNet。
如果出现 `No module named lightgbm`，先在 `titan_bbb` 环境中单独安装缺少的 LightGBM；
不要为了本实验批量升级原来的依赖。

如果想先检查已有文件，可先运行（不会训练、不会创建输出文件）：

```bash
python titan_classifier_comparison/run_classifier_comparison.py --project-root .. --seed 42 --check-only
```

检查通过后，再运行不带 `--check-only` 的命令。
真正运行时还会从 baseline 重新计算验证集概率，与保存的验证集概率比较，
确认检查点、源码和预处理能复现后，才训练四组分类器。

## 4. 读取哪些旧文件

默认按下列顺序寻找已有 baseline 实验，找到第一个后检查其 seed 和完整性；
不按测试集指标挑选来源：

1. `titan_attention_ablation_seed42_results/`
2. `titan_attention_seed42_results/`
3. `titan_attention_seed2022_results/`（只有清单记录的 seed 实际为42才接受）
4. `titan_mechanism_2051_results/` 中的 **baseline_seed42** 子模型。

读取该实验 `run_manifest.json` 记录的 baseline 文件夹、模型配置、
选择锁、`best_model.pth`、`titan_modality_scalers.joblib` 和验证集预测。
不使用该实验中的机制模型检查点。

如果程序未找到自动路径，显式指定**实验结果根目录**：

```bash
python titan_classifier_comparison/run_classifier_comparison.py --project-root .. --seed 42 --baseline-run ../titan_attention_ablation_seed42_results
```

其余输入为：

- `dataset/{train,val,test}_classification.csv`：7593/727/942行；标签保持TITAN原编码。
- `features/{train,val,test}-{tabular,image,text}-classification.npy`：384/2048/768维。
- `mechanism_features/manifest.json`、三个机制分数 `.npy` 及行审计 `.csv`。
- `src/utils/model.py`：原TITAN模型源码，需与 baseline 记录的哈希一致。

代码只调用 baseline 保存的 StandardScaler 的 `transform`，不重新拟合预处理。
三个机制分数顺序为 `p_efflux, p_influx, p_pampa`，保持原始[0,1]数值。
不会重训辅助模型，也不会修改原始输入、检查点或划分。
原划分已有的分子身份交叉记录保留在运行清单中；本次不额外改变划分。

## 5. 模型、训练轮数和阈值怎样确定

- 两组神经网络：AdamW，lr=1e-4，weight_decay=0.01，batch=32，Dropout=0.1；
  最多1000轮，验证损失连续10轮未改善则停止，恢复最佳检查点。
- 两组 LightGBM：一套预先固定的起始配置；lr=0.03、最多31叶、深度5、
  min_data_in_leaf=40、L2=1、feature_fraction=0.9、bagging_fraction=0.8；
  最多2000轮，验证损失连续50轮未改善则停止，保存最佳迭代。
- 两类分类器都只从训练标签计算 `scale_pos_weight=N_negative/N_positive`。
  轮数选择统一采用**按样本数取平均的加权二元交叉熵**。
  LightGBM 显式使用自定义验证指标，避免其默认指标与神经网络损失加权口径不一致。
- 两个神经网络头的共有权重从相同随机初始化出发；机制头新增的3列权重以0初始化，
  保留相同的后续随机数状态，新增权重可正常训练。
- 各组分别在验证集上从 **0.1、0.3、0.5、0.7、0.9** 选择 BACC 最高的阈值；
  并列时采用先出现的较小阈值。判断规则与原TITAN一致：`probability > threshold`。
- 四组模型和阈值全部写入选择锁后，才计算测试集指标。
  不把辅助模型原来的训练轮数照搬到新分类器，也不用测试集选参数。
- 固定0.5、0.7只是另外两份报告，不再训练，不会替换验证集自动选择的结果。

这是一套起始对照设置，不是对两类分类器做了全面超参数搜索。
LightGBM 不保证提高；尤其2048维稠密表征未必适合树的单特征切分。
单个seed的微小差值只能视为初步观察。

## 6. 结果在哪里

默认目录：

```text
/root/project/titanBBB/TITAN-BBB-main/titan_classifier_comparison_seed42_results/
```

| 文件 | 含义 |
|---|---|
| test_comparison.csv | 四组测试指标；阈值分别来自验证集 |
| val_comparison.csv | 四组验证指标 |
| test_fixed_threshold05.csv / test_fixed_threshold07.csv | 测试集固定0.5、0.7的补充结果 |
| val_fixed_threshold05.csv / val_fixed_threshold07.csv | 验证集对应补充结果 |
| test_all_thresholds.csv / val_all_thresholds.csv | 各12行；自动选择、固定0.5、固定0.7全部保留 |
| comparison_table.md | 可直接阅读的全部指标表 |
| contrast_summary.csv | 加机制前后、换分类器前后的指标差值（A减B） |
| original_baseline_reference_only.csv | 旧端到端baseline检查点的指标，单独参考 |
| run_manifest.json / all_selections_locked.json | 输入、软件版本、配置、检查点及选择锁 |

每张比较表均有 **AUROC、AUPRC_AP、ACC、BACC、SN、SP、F1、MCC**，
以及阈值、样本数和混淆矩阵计数。AUPRC_AP 使用 average precision 口径。
SN对应TITAN原标签1的召回率，SP对应原标签0的召回率，标签含义未翻转。

`features/attempt_001/` 保存冻结2048维向量；
`models/<experiment>/attempt_001/` 保存分类器、最佳轮数/阈值、日志、训练历史和逐样本预测。
LightGBM 另有 `feature_importance.csv`，可查看三个机制分数的使用情况；
特征重要性不能直接解释为生物机制的因果作用。

建议首先比较 `lightgbm_mechanism` 与 `lightgbm_baseline`，
再比较 `nn_mechanism` 与 `nn_baseline`，检查换分类器是否改变机制分数的增益。

## 7. 中断后继续

使用完全相同的命令加 `--resume`：

```bash
python titan_classifier_comparison/run_classifier_comparison.py --project-root .. --seed 42 --resume
```

已完成并通过哈希核对的特征和组会复用；尚未完成的组从头训练，使用新的 attempt 文件夹。
不支持恢复到中断前的某一个epoch。若之前显式指定了 baseline 路径或其他参数，
继续时必须保留这些参数。输入、源码、依赖版本或配置改变时会拒绝复用，
可用新的 `--output-dir ../另一个结果目录` 单独运行。

## 8. 已完成的代码检查及限制

包内 `tests/` 检查输入哈希、行顺序、训练集scaler复用、错误检查点拒绝、
阈值并列和严格大于规则、加权损失、共有初始化、2048/2051维前向数学、
前向钩子抽取、LightGBM调用约定、四组选定后才测试、完成组复用及缓存篡改拒绝。

开发工具环境未安装 Torch / LightGBM；这些检查包括NumPy数值验证与模拟后端，
**不代表已经完成真实神经网络训练或真实LightGBM训练**。
真实运行时，代码会在服务器上执行神经网络预测头的维度和反向传播检查。
实际指标以服务器生成的CSV为准。

可选的本地代码自检（从 `src` 运行，不需要Torch或LightGBM）：

```bash
python -m unittest discover -s titan_classifier_comparison/tests -v
```

LightGBM官方接口参考：

- https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html
- https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html
