# 分布预测头（HL-Gauss）：五组实验

原版 DLF 在融合特征 `z` 上直接输出一个标量 `ŷ = wᵀz + b`。本改进把最后一个预测头
换成"各分数区间的概率"，再用概率加权平均得到连续分数，训练时可选增加高斯软标签监督。

五组实验共用同一套骨干与 `L_other`（原有 hetero / c / 重构 / 解耦 / ort / shared
损失，保持不变），只改**最终主预测头**。通过 `head_mode` 五选一启动。

## 公式

分布头：

```
ℓ = Wz + b,   p = softmax(ℓ),   ŷ = Σ_k p_k c_k
```

`p_k` 为第 k 个区间概率，`c_k` 为区间中心，`ŷ` 仍是连续值（如 1.27）。

高斯软标签 `q(y)`：把真实标签视为 `N(y, σ²)`，计算每个区间内的概率质量并归一化。

```
q_k(y) = Φ((e_{k+1} − y)/σ) − Φ((e_k − y)/σ)
```

两端边界取 `∓∞`，因此 `Σ_k q_k = 1`，无需再归一化；但截断到支持范围后
`Σ_k q_k c_k ≈ y` 而非严格相等，这也是完整版本保留主 L1 的理由。

## 五组对照

| 实验 | `head_mode` | 最终输出 | 主 L1 | CE 目标 |
|---|---|:---:|:---:|---|
| ① 原始 DLF | `original` | 标量 | 有 | 无 CE |
| ② 分布头，仅 L1 | `dist_l1` | `Σp_k c_k` | 有 | 无 CE |
| ③ 硬标签 CE + L1 | `dist_hard` | `Σp_k c_k` | 有 | 标签所在区间 one-hot |
| ④ HL-Gauss，仅 CE | `dist_gauss` | `Σp_k c_k` | **无** | 高斯软标签 |
| ⑤ HL-Gauss + L1 | `dist_gauss_l1` | `Σp_k c_k` | 有 | 高斯软标签 |

主头损失（其余 `L_other` 不变）：

```
①  L1(ŷ, y)
②  L1(ŷ, y)
③  L1(ŷ, y) + λ·CE(ℓ, onehot)
④  λ·CE(ℓ, q(y))
⑤  L1(ŷ, y) + λ·CE(ℓ, q(y))
```

注意 ④ 只是**主头**去掉 L1；原有辅助头里的 L1 仍包含在 `L_other` 中。
不同实验训练总损失的项目数不同，**不能直接比较总损失大小**，只比较评估指标。

## 实验解释性

②—⑤ 使用相同预测头结构、区间数和区间中心；③④⑤ 使用相同 λ；④⑤ 使用相同 σ。于是：

- ② vs ①：更换输出参数化本身是否有效；
- ③ vs ②：加入硬标签区间分类是否有效；
- ⑤ vs ②：加入高斯软标签分类是否有效；
- ⑤ vs ③：高斯软标签是否优于硬标签；
- ⑤ vs ④：保留主 L1 是否有帮助。

## 评估一致性

五组验证集 `Loss`（驱动 `ReduceLROnPlateau` 与 `KeyEval='Loss'` 选模）**统一用
`L1(ŷ, y)`**，与训练目标解耦，保证早停与选模规则完全相同；只让训练目标变化。
测试仍用相同的 MAE / Corr / Acc-2 / F1 / Acc-7。

## 配置

`config/config.json` → `DLF.commonParams`：

| 键 | 默认 | 说明 |
|---|---:|---|
| `head_mode` | `original` | 五选一，见上表 |
| `head_num_bins` | 66 | 区间数 K |
| `head_bin_min` | -3.3 | 支持范围下界 |
| `head_bin_max` | 3.3 | 支持范围上界 |
| `head_gauss_sigma` | 0.075 | 高斯软标签标准差 σ |
| `head_ce_weight` | 0.1 | 交叉熵权重 λ |

区间等宽，宽度 `Δ = (bin_max−bin_min)/num_bins = 0.1`。

## 运行

```bash
python train.py --dataset mosi --head-mode original
python train.py --dataset mosi --head-mode dist_l1
python train.py --dataset mosi --head-mode dist_hard
python train.py --dataset mosi --head-mode dist_gauss
python train.py --dataset mosi --head-mode dist_gauss_l1
```

测试使用相同参数。checkpoint 后缀自动带模式，互不覆盖，也可与其他开关组合：

```
pt/DLFmosi.pth
pt/DLF_dist_l1_mosi.pth
pt/DLF_dist_hard_mosi.pth
pt/DLF_dist_gauss_mosi.pth
pt/DLF_dist_gauss_l1_mosi.pth
```

例如与选择性空读取组合：`--lfa-mode selective_null --head-mode dist_gauss_l1`
对应 `DLF_dist_gauss_l1_selective_null_mosi.pth`。

`test.py` 会校验 checkpoint 与 `head_mode` 是否匹配，不匹配直接报错，避免
`strict=False` 静默使用随机初始化的预测头。

## 训练诊断日志

启用分布头时，每个 epoch 额外打印（batch 均值）：

| 指标 | 含义 |
|---|---|
| `l1` | 期望值 L1（始终计算，便于观察） |
| `ce` | 当前 CE 项（无 CE 的模式为 0） |
| `entropy` | 预测分布熵（nats），越小越集中 |
| `bin_acc` | argmax 概率区间命中真实标签所在区间的比例 |

## 实现位置

- 分布头、软/硬标签、损失：[trains/singleTask/distribution_head.py](trains/singleTask/distribution_head.py)
- 模型侧开关与三元组输出：[trains/singleTask/model/DLF.py](trains/singleTask/model/DLF.py)
- 训练器主头损失与诊断：[trains/singleTask/DLF.py](trains/singleTask/DLF.py)
- 变体后缀与 checkpoint 命名：[run.py](run.py)
- 单元测试：[tests/test_distribution_head.py](tests/test_distribution_head.py)
