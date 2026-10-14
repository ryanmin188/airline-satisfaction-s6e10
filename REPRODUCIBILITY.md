# Airline Satisfaction — 自训练 Public ROC-AUC 0.96114

2026-10-13，`submissions/add_compact30.csv` 在 Kaggle Playground Series S6E10 获得 **Public ROC-AUC 0.96114**，超过目标 0.96110。本地 OOF AUC 为 **0.9614951671283594**。线上、线下是不同数据集的指标。

全部 OOF/test 预测来自自己的训练。公开 Notebook 仅作为方法参考，没有加载他人的预测、模型权重或 ensemble。原始公开航空数据用来自行训练教师，来源和排除规则均有记录。

![Kaggle 成绩](evidence/kaggle_096114.png)

## 固定的最终配方

| 自训练配置 | 权重 | 源码 |
|---|---:|---|
| Conditional TE LightGBM | 15.68% | src/historical/feature_5_conditional_frequency.py |
| RealMLP V1 | 23.52% | src/historical/mlp_lean_1.py |
| TabM M3 三种子概率平均 | 9.80% | src/historical/tabm_m3_full_oof.py、tabm_m3_multiseed.py |
| 原始数据教师特征 + LightGBM | 21.00% | src/train_teacher_lgbm.py |
| Compact RealMLP，seed 33 | 30.00% | src/train_realmlp_compact.py |

各分量分别取平均秩除以样本数，然后一次加权求和。TabM 先平均三个种子的概率再取秩。五条配置覆盖三个最终预测模型族；TabM 的三个种子和交叉验证各折不另称模型族。另有一个自己训练的 XGBoost 提供教师特征。

新 RealMLP 使用 47 个特征、512/256/128 隐藏层、n_ens=8、固定 3 epochs、统一五折 seed=42。删除两组高基数交互的类别输入，保留其内层交叉拟合 TE。单模型 OOF 0.9611721453。新增分量让此前 0.96105 的自训练融合提高到线上 0.96114；没有逐特征消融，不能把提升完全归因于某一特征。

## 快速复现

```sh
pip install -r requirements-reproduce.txt
python src/unpack_artifacts.py
python reproduce.py
```

`reproduce.py` 读取冻结的自训练预测和 `ensemble_spec.json`，写出 `submissions/reproduced.csv`，核验 OOF 和提交 SHA-256。已验证重建与实际提交逐字节一致：

`9700c827f0dd1b8d46098b19c9c3de83c123fb6de96d5342b76f98628c3308ca`

## 数据和训练

```sh
python src/unpack_data.py
pip install -r requirements-train.txt
python src/train_realmlp_compact.py --seed 33
python src/train_teacher_lgbm.py
```

新训练脚本用项目相对路径，已有折预测时恢复检查点。完全重新训练请先将对应 `experiments` 子目录移到备份位置。RealMLP/TabM 需要兼容的 PyTorch GPU 环境；GPU 重训不承诺位级一致。部分神经网络只保存了 OOF/test 预测，未序列化拟合后的网络参数。教师 XGBoost 和教师 LightGBM 五折权重已保存。

历史训练代码原样保留在 `src/historical`。用 `python src/run_historical.py feature_5_conditional_frequency.py` 可将其旧路径映射到当前项目内 `retraining/`；运行顺序为 conditional LightGBM、RealMLP V1、TabM full OOF、TabM multiseed。这些耗时历史训练没有在归档阶段全部重跑，当前精确复现基于冻结预测。

比赛 train/test/sample_submission 和原始 `original.csv` 以无损 gzip 保存于 `data/`。原始航空数据 129,880 行，排除 465 个预定义近重复后训练教师；规则匹配服务评分及类别，Age 差≤1、航程差≤10、延迟差≤5，不查看比赛标签。详细审计见解包后的 `experiments/teacher_lgbm/teacher_audit.json`。

## 实验记录和验证边界

- 前一自训练冠军：Public 0.96105，OOF 0.961451589。
- 28 种较简洁融合及 Extra Trees LightGBM 两折筛选未发现稳定改进；保留摘要和相关源码。
- 本轮 enhanced LGBM、XGBoost 两折结果不够稳定，没有进入最终配方；CatBoost 分支未运行。
- Compact RealMLP 完成五折。九种固定融合候选按历史折 1/2 选择，折 3/4/5 作诊断确认。最终五折增益均正，比较表为 `experiments/compact_blend_screen.csv`。
- 历史 OOF 多次复用，确认折不是全新独立留出集，也不是完整嵌套验证；Public 分数不保证 Private 分数。
- 此前包含他人预测的 0.96111 不属于本项目成果，也未纳入本目录。

`artifacts/` 无损归档预测、权重和日志，`src/unpack_artifacts.py` 在原目录结构解包；`MANIFEST.json` 记录仓库文件 SHA-256。`research/` 保留来源和早期比较记录，旧记录中的路径、指标为历史状态；当前最终结果以 `ensemble_spec.json` 为准。

## 来源

- [比赛](https://www.kaggle.com/competitions/playground-series-s6e10)
- [原始航空数据](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k)
- [路线/距离特征方法参考](https://www.kaggle.com/code/busyaprime/s6e10-route-or-distance-both-lb-0-96169)
- [RealMLP 配置方法参考](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit)
- [教师特征方法参考](https://www.kaggle.com/code/sachith7/s6e10-what-each-step-was-worth)

数据及第三方方法仍受各来源条款约束；没有为第三方数据重新授予许可证。项目展示仓库公开，个人学习与备面文稿不包含在发布清单中。
