# 选择性空读取：两版实现

主实验用于对比原始 DLF、原始 DLF + 选择性空读取，以及在其基础上再加收益监督
的版本。原始 DLF 自带的 `0.01 * HingeLoss` 保持不变，后加的 ATP、SENT 默认关闭。
通过 `lfa_mode` 三选一：

- `original`：原始交叉注意力，默认模式；
- `selective_null`：仅在 A→L 和 V→L 路径增加可学习空 Key，空 Value 固定为零；
  空 Key 只加在**最后一层**，并带 per-head `null_bias` 与 warm-up 门控；
- `selective_null_gain`：在 `selective_null` 基础上加入逐样本收益监督（EMA Teacher）。

```bash
python train.py --dataset mosi --lfa-mode original
python train.py --dataset mosi --lfa-mode selective_null
python train.py --dataset mosi --lfa-mode selective_null_gain
```

测试时使用相同的参数。三种 checkpoints 互不覆盖：
`pt/DLFmosi.pth`、`pt/DLF_selective_null_mosi.pth`、
`pt/DLF_selective_null_gain_mosi.pth`。

已有的 ATP/SENT 实现仍然保留。例如 `--shared-loss-mode atp_sent` 可运行
ATP_SENT 消融，其 checkpoint 会带有相应后缀，不会覆盖上述主实验。

## 空读取的最后一层 + warm-up

`selective_null` 的空 Key 只注入 A→L 和 V→L 的**最后一层**交叉注意力，
门控作用于最终的跨模态读取，而非每一层。

`null_bias`（per-head）控制 Null 位置的 logit 偏置，训练器按 epoch 控制门控：

- 前 `null_warmup_epochs`（默认 5）个 epoch：`null_bias = -30`，Null 概率≈0，
  门控基本关闭；同时冻结 `null_key` 与 `null_bias`，不参与学习。
- 之后：`null_bias` 重置为 0 并解冻，`null_key` / `null_bias` 正常学习，
  由任务损失自行决定是否读取 Null。

即：先让模态分支学出可用的跨模态表示，再放开 Null 门控供模型选择是否跳过。

## 收益监督（`selective_null_gain`）

对每个样本判断：如果强制某跨模态路径只读取 Null，预测误差是否变大？
误差变大说明该模态有帮助，应减少空读取；误差变小说明该模态有干扰，应增加
空读取。仅屏蔽对应 A→L / V→L 路径，`c_fusion` 等共享分支保持不变。

- 用逐样本 Smooth L1 计算 `e_full`、`e_A-null`、`e_V-null`，
  收益 `Δ = e_null - e_full`。
- 软目标 `r = sigmoid((Δ - δ) / τ)`，空读取目标 `t_null = 1 - r`。
- 置信度权重 `w = min(|Δ| / c, 1)`，做加权 BCE：`L_gain`。
- 收益标签由 **EMA Teacher** 无梯度产生，避免学生自监督坍塌。
- 前 `warmup` 个 epoch 只训练原始 DLF 损失，之后 `λ_gain` 线性 ramp 到上限。

总损失：`L = L_DLF + λ_gain · L_gain`。推理时不运行 Teacher，也不计算反事实
分支，因此没有额外推理成本。

超参默认值（`config/config.json`）：

| 参数 | 键 | 默认 |
|---|---|---:|
| EMA decay | `gain_ema_decay` | 0.995 |
| warmup epochs | `gain_warmup_epochs` | 5 |
| ramp epochs | `gain_ramp_epochs` | 5 |
| margin δ | `gain_margin` | 0.01 |
| temperature τ | `gain_temperature` | 0.1 |
| λ_gain 上限 | `gain_weight` | 0.05 |
| 置信度尺度 c | `gain_confidence_scale` | 0.1 |

`selective_null` 与 `selective_null_gain` 均输出 `null_attn_a`、`null_attn_v`，
形状 `(batch, text_steps)`，表示最后一层交叉注意力选择空位置的概率均值。
`selective_null_gain` 在训练时额外返回 `output_logit_a_null`、`output_logit_v_null`
两个反事实预测（仅训练使用）。

## 推荐实验（三组）

| 实验 | 空读取 | 收益监督 | lfa_mode |
|---|---:|---:|---|
| 原始 DLF | × | × | `original` |
| DLF + Null | √ | × | `selective_null` |
| DLF + Null + Gain | √ | √ | `selective_null_gain` |

可额外统计：有益/有害样本的平均空读取率、`Δ>0` 与 `Δ<0` 两组的性能、
音频与视频分别被拒绝的比例。三组均从头训练，并保持数据、种子、原始 HingeLoss、
训练预算和验证集选模规则相同。

## 融合门控（`fusion_gate_mode`）

在最终融合前对 A、V 两条路径各加一个样本级标量门控，文本路径与共享路径不加门：

\[
z_{\rm gate}=[u_L;\ g_V u_V;\ g_A u_A;\ c],\qquad
g_A=\sigma(w_A^\top[h_L;h_A]+b_A),\quad
g_V=\sigma(w_V^\top[h_L;h_V]+b_V)
\]

`u_A,u_V` 是 LFA 之后的文本引导交互结果（`t` 在 sigmoid 投影之后）；`h_L,h_A,h_V`
用融合前的 `hs_proj_*_high`（维度 d=50，拼接 100 维），新增参数 `4d+2`。门控与
`lfa_mode` 正交，可与 `selective_null` / `selective_null_gain` 组合；组合时反事实
分支也乘同一门控。

`fusion_gate_mode` 四选一（config 键，默认 `none`）：

- `none`：不启用，现有三版行为逐位不变；
- `fixed`：`g_A=g_V=fusion_gate_init`（默认 0.9）常数，用于检查“仅是缩小特征”的对照；
- `global`：两个全局可学习标量（2 个参数），检查是否需要逐样本；
- `sample`：逐样本两个门（`4d+2` 个参数），weight 置零、bias 取 σ⁻¹(0.9)，
  训练起点与 `fixed` 一致。

```bash
python train.py --dataset mosi --fusion-gate sample
python train.py --dataset mosi --lfa-mode selective_null_gain --fusion-gate sample
```

checkpoint 后缀自动带上模式，例如 `DLF_sample_mosi.pth`、与 gain 组合时
`DLF_selective_null_gain_sample_mosi.pth`，互不覆盖。启用门控时训练器每 epoch 记录
门控均值与跨样本标准差（日志 + `result/normal/gate_stats<variant>.csv`）：若 std
接近 0，说明模型没有真正按样本选择模态。四种模式都应与原版 DLF 在相同数据、种子
和训练设置下比较 MAE、相关系数及分类指标。
