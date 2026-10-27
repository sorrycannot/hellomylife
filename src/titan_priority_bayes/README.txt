V01 V03 优先结构贝叶斯权重搜索

一 运行前提
本程序用于原 TITAN 分类划分及原未加权 LightGBM 全融合模型。
输入保持为冻结的2048维融合向量，加上efflux、influx、pampa三个原始机制概率，共2051维。
固定模型与搜索随机种子42，只改变训练时的逐样本权重。
程序不训练上游神经融合模块，也不重新训练三个辅助机制模型。

把本文件夹 titan_priority_bayes 整体放到服务器：
/root/project/titanBBB/TITAN-BBB-main/src/
应存在：src/titan_priority_bayes/run_priority_bayes.py
及同文件夹里的 priority_bayes 子目录，请不要只上传入口文件。

在服务器执行：
conda activate titan_bbb
cd /root/project/titanBBB/TITAN-BBB-main/src
python titan_priority_bayes/run_priority_bayes.py --project-root ..

默认执行完整200组搜索。运行期间保持服务器任务存活；每完成一组即保存其模型和结果。

二 依赖
沿用现有 titan_bbb 环境。贝叶斯代理模型使用scikit-learn高斯过程，不需要Optuna。
程序会检查 numpy、pandas、scipy、scikit-learn、LightGBM、RDKit、joblib和openpyxl。
之前训练环境已有的大部分依赖不应升级。若仅缺openpyxl：
python -m pip install openpyxl
requirements.txt供核对或补装缺少的依赖；不要为了运行本程序升级现有环境。
本地化学匹配验证使用RDKit 2025.03.6；合成流程训练测试使用LightGBM 4.6.0。
实际运行时记录服务器版本，并要求恢复运行时版本不变。
本程序针对当前Linux服务器，使用Linux文件锁防止同一attempt并行写入。

三 原始输入来源与校验
默认来源目录为项目根目录下：
titan_structure_weighting_seed42_results/attempt_003
只从该运行的input_audit.json读取锁定的train、val输入路径与哈希；
从reference_audit.json读取原未加权全融合参考模型及既定训练轮数上限、早停设置；
参考预测读取 predictions/reference_saved_full_fusion/fixed_0p5/val_all.csv。
不读取其旧结构定义、旧结构权重或候选选择结果。
不按验证性能从多个输入来源中选择，也不重新扫描不同版本的机制分数。

若这个原始目录改过名字，只修改来源参数，例如：
python titan_priority_bayes/run_priority_bayes.py --project-root .. --source-run 原始目录相对于项目根目录的路径
--source-run必须指向有上述原始输入与参考模型审计记录的目录，不能指向仅有新结构覆盖表的目录。

程序依次核查：
1 当前train、val CSV与历史输入记录的SHA256一致，行数为7593、727，标签保持BBB+=1。
2 冻结H和机制M的文件字节、数组形状、列顺序及可用的行身份见证一致。
3 原参考模型在这些输入上逐行复现保存的验证概率。
4 这些概率与本代码包中用于发现V01/V03的原未加权模型概率逐行一致。
5 用原始723个验证身份校准全部17条结构规则的精确成员集合。
6 训练主要身份覆盖与上一轮审计一致：
   V01：1142个身份，阳性220、阴性922、连接关系组514。
   V03：530个身份，阳性257、阴性273、连接关系组237。
7 正式搜索的第一组(1,1)重新训练未加权对照，要求逐行验证概率与原参考模型最大差异<=1e-10，
   且固定0.5阈值下的判断一致。失败即停止，不放宽容差，不继续加权搜索。

没有行身份见证的裸数组沿用原导出顺序并锁定历史字节。
字节锁定与验证概率复现是强一致性检查，但不是对历史训练数组行身份的独立证明。
共享NPZ只加载指定的train/val数组键。共享机制CSV先读分区名称，再跳过其他分区的数据行；
原测试CSV、测试特征数组键和测试预测不被加载。共享容器的整体文件哈希仍会读取其文件字节。

四 结构与权重
仅启用V01、V03。分子身份保留立体信息；Morgan规则仍按原设定不包含手性。
局部环境构建与成功执行的训练覆盖审计相同，联合稀疏ID、半径与精确环境字符串匹配。
不会用宽泛的“有羟基”或“有羧基”替代原规则。

V01权重w01、V03权重w03分别从1.00到3.00，每次相隔0.05，均有41个可能值。
命中V01的阳性、阴性样本都按相同规则加权，V03亦同。
未命中权重1；仅命中一条使用对应权重；两条均命中使用max(w01,w03)。
不累乘，不作额外归一化，原scale_pos_weight保持不变。
结构权重通过LightGBM训练Dataset的weight实现。
验证评价不应用结构权重；早停沿用原始类别加权BCE，选权重的指标为验证AUROC。

训练覆盖按保留立体信息的主要身份统计；实际模型仍使用原始7593行，保持原始特征对齐。
一致重复记录使用同一结构权重，并保留全部原始训练行。
解析失败、多组分及身份标签冲突记录不用于规则加权，结构权重为1；
它们同样保留在未加权和加权组中，不在某一组单独删除。
这些处理在train_structure_matches.csv及最终训练权重表中完整记录。

五 贝叶斯搜索
默认200个不同参数组合，包含以下4个固定参照点：
(1,1)、(1.5,1)、(1,1.5)、(1.5,1.5)。
随后36个组合采用seed42的无重复随机探索，形成前40组初始观察。
后160组根据已观察的验证AUROC，拟合高斯过程并在未试过的网格点上最大化期望改进EI。
不会遍历全部1681组；不重复训练同一权重组合。

代理模型：ConstantKernel × Matern(nu=2.5) + WhiteKernel；normalize_y=True；
权重网格坐标缩放到[0,1]；GP随机种子42；优化核超参数时额外重启1次。
EI探索偏移xi=0.0001，单位为AUROC；在对数空间比较EI，避免极小EI下溢为零而失去排序。
高斯负尾采用Mills比渐近展开保持数值稳定；普通EI与logEI都记录在每组PROPOSAL.json中。
仅当所有候选均无有限logEI时，才用后验标准差最大的未试组合，并明确记录回退原因。

只按验证AUROC选择本预算内最佳组合，不用BACC、MCC、测试表现作次级性能筛选。
AUROC完全相同时，优先最大权重更小、再总权重更小、再V01/V03更小的组合。
固定0.5和验证BACC自动阈值结果都保存；自动阈值沿用0.1、0.3、0.5、0.7、0.9，
严格使用 probability > threshold，相同BACC取较早的阈值。
最终最佳checkpoint直接保存，不把train和val合并重训。

六 输出位置
项目根目录/titan_priority_bayes_seed42_results/attempt_bayes001
后续新运行自动创建attempt_bayes002等，不覆盖旧运行。

主要输出：
protocol_lock.json                     输入、代码、规则、环境、参数与选择策略锁定
input_audit.json                       原始输入来源与行对齐范围
structure_audit.json                   17条校准结果及V01/V03覆盖核对
control_reproduction_check.json         未加权对照逐行复现检查
train_structure_matches.csv            原始训练行的结构命中和身份审计
val_structure_matches.csv              原始验证行的结构命中和身份审计
train_val_structure_coverage.csv        两条规则的原始行及主要身份覆盖
search_history.csv                     所有组合的全部验证分类指标及实际权重总量
search_progress.csv                    每一步AUROC及当前最佳值
search_progress.svg                    可直接用于汇报的英文搜索进度图
excel/priority_weight_search.xlsx       搜索记录、覆盖、参照及最佳结果、子组和错误转移
trials/trial_0001/...                  每组模型、验证概率NPY、早停历史、提案及完成记录
best_trial.json                        本预算内选中的完整记录
selected_model/best_model.txt          选中的LightGBM模型
selected_model/selection_lock.json     选权重依据、阈值及模型哈希
selected_train_weights.csv             每个原始训练行的最终结构和有效损失权重
selected_upweighted_train_rows.csv     最终被提高权重的训练行
validation_comparison.csv              原参考、未加权重训、三个固定参照及贝叶斯最佳模型
predictions/模型名/阈值策略/val_all.csv  验证逐化合物结果，并另存TP/TN/FP/FN四张表
validation_paired_changes_fixed05.csv  最佳模型对比原参考的逐化合物错误转移
validation_corrected_errors_fixed05.csv 被纠正的原FN/FP
validation_new_errors_fixed05.csv      从正确变成错误的样本
validation_subgroups_primary_fixed05.csv 按723个主要验证身份的结构子组指标
selection_history/                    保留每次完成预算对应的选择结果

主要整体指标基于原始727行；结构子组基于723个主要身份，两个分母不会混用。
单标签或空子组的AUROC等不可计算项留空，不写成0。
CSV、SVG每完成一组更新；Excel每10组及最终完成时更新。

七 先检查再启动与中断恢复
如果只想先检查输入：
python titan_priority_bayes/run_priority_bayes.py --project-root .. --stage inspect
该命令不拟合新模型，只完成输入与结构校准、原参考模型预测一致性检查。
检查完成后，在同一目录继续：
python titan_priority_bayes/run_priority_bayes.py --project-root .. --resume titan_priority_bayes_seed42_results/attempt_bayes001

运行中断时也使用上面的--resume命令。
--budget表示总预算，不是新增次数。已有120组，默认预算200时只完成其余80组。
若明确要扩展到300组：
python titan_priority_bayes/run_priority_bayes.py --project-root .. --resume titan_priority_bayes_seed42_results/attempt_bayes001 --budget 300
历史模型与完成记录保留，新选择结果按总完成数存档。

恢复时校验已完成文件哈希、验证指标、输入、代码、规则、环境及训练设置。
已提交的组合不会重新训练；中断在保存完成记录前的最后一组会重新执行。
不要在一个attempt里更换输入、代码、RDKit或LightGBM版本；需要变更时新开attempt。
FAILED.json和INTERRUPTED.json保留失败/中断记录，最新完成情况见COMPLETE.json及终端输出。

八 自检与证据边界
在服务器src/titan_priority_bayes目录可运行：
python -m unittest discover -s tests -v
测试使用临时合成工程和真实RDKit/LightGBM，不使用服务器的真实训练数据，也不产生研究结论。
包含17条原验证成员校准、立体身份、权重重叠、AUROC选择、GP唯一提案、
输入列顺序、测试输入读取拦截、真实训练、恢复运行、结果文件与篡改停止检查。

当前验证集已参与结构发现，本轮结果属于开发选权重，不能作为独立泛化收益的确认。
程序不读取原测试数据。独立评价与任何后续测试应在本轮方案锁定后另行开展。
“最佳”仅指预算内验证AUROC最高，不保证全局最优，也不保证其他数据上提升。
本代码未接入随机样本加权负对照；那是后续单独的验证实验，不混入本轮200组参数搜索。

算法参考：
Snoek J, Larochelle H, Adams RP. Practical Bayesian Optimization of Machine Learning Algorithms. 2012. https://arxiv.org/abs/1206.2944
Ament S, Daulton S, Eriksson D, et al. Unexpected Improvements to Expected Improvement for Bayesian Optimization. 2023. https://arxiv.org/abs/2310.20708
scikit-learn GaussianProcessRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
