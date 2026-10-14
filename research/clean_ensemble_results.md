# 2026-10-13：Extra Trees、外部数据与简洁融合实验

## 结论

正式方案仍为自训练 Public 0.96105；本轮未产生可靠的新提升，没有新增线上提交。完成了 28 项融合比较（含当前基线）和 Extra Trees LightGBM 前两折训练。所有预测来自自己训练。

## Extra Trees 是什么

Extra Trees 是极端随机树，增加切分阈值的随机性以降低树之间的相关性。这里试的是 LightGBM 的 `extra_trees=True`：评估特征切分时仅检查一个随机阈值，仍属于 LightGBM boosting；不等同于 sklearn 的独立 ExtraTreesClassifier 森林。

官方参数：https://lightgbm.readthedocs.io/en/latest/Parameters.html#extra_trees

本轮沿用我们自己的教师及特征，采用 31 leaves、特征比例 0.4、min_child_samples 60、extra_trees=True。固定原有 seed 42 五折划分，仅执行第 1/2 折；内部早停后按选定轮数重训外折。分别训练 1728/1535 轮，单折 AUC 0.96102424/0.95989945。不是完整 OOF，也不是只开关 extra_trees 的严格单参数消融。

为了比较不完整训练，所有分量均在各验证折内重新排名：加入 10% XT 后增益分别约 +0.000002/-0.000007，20%/30% 均下降；替换原有 30% 教师分量也下降。停止扩展此配置，但不推断所有 Extra Trees 配置都无效。明细：`../experiments/teacher_lgbm_xt/blend_screen.csv`；源码：`../code/train_teacher_lgbm_xt.py`。

## 教师数据为何相关

官方比赛数据页明确说明比赛数据参考 arseniyshutko/binary-aviation-satisfaction-129k，分布接近但不完全相同：https://www.kaggle.com/competitions/playground-series-s6e10/data 。这正是本项目使用的原始数据来源。

处理链：129,880 原始行 → 按既定规则排除 465 条与比赛数据近重复的行 → 129,415 行训练 XGBoost → 对比赛训练与测试样本预测 → logit 概率作为新 LightGBM 的一个输入特征。并未把原始行追加到比赛训练集。主模型仍使用 699,635 条比赛训练样本。没有使用他人生成的预测。

教师单独应用于比赛训练样本 AUC 0.955241，与目标具有预测关联；它可能提供原始分布中的可迁移结构，不表示原始数据就是测试集答案。近重复排除规则有明确界限，不能宣称消除全部可能的相似记录。当前成绩提升来自整条新训练方案，未通过消融单独证明教师特征贡献了全部提升。

## 融合筛选

固定 8 组粗权重，分别比较 rank、概率和 logit 融合；另比较 2/3/4 分量的固定 C=0.1 Logistic stacking，加上已验证基线共 28 项。历史折 1/2 排序选择，3/4/5 诊断确认；stacking 的预测按元模型折交叉拟合。历史基础 OOF 多次复用，因此不属于独立留出测试或完整嵌套交叉验证。

| 配置 | 全量历史 OOF AUC（约） | 结论 |
|---|---:|---|
| 当前四分量 rank | 0.96145159 | 正式版，Public 0.96105 |
| 相同权重 logit | 0.961462 | 总体仅高约 0.000011，折间不一致 |
| 四分量 Logistic | 0.961449 | 选择折略优，三个确认折均下降 |
| 三分量 logit，移除 TabM | 0.961432 | 更少训练模型，但确认折下降 |
| RealMLP / 教师两分量等权 logit | 0.961382 | 更简单，但总体损失约 0.000070 |

完整记录：`../clean_blends/screen.csv`。选出的 2/3/4 分量候选 CSV、OOF 和配置保留用于审计，均未提交、未晋升为正式模型。

“干净”优先落实为单层加权、单一正式配置、来源可追踪、实验与正式结果分开；不是为了少一个模型接受未经衡量的性能损失。`../ensemble_spec.json` 集中管理正式权重，`../reproduce.py` 为复现入口，改造后 SHA-256 校验仍通过。下一步若继续提高上限，比细扫权重更值得的是加强基础模型的互补特征；模型蒸馏可以降低部署成本，但需要独立验证，不能预先承诺无损压缩。
