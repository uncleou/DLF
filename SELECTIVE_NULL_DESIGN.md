# 选择性空读取：第一版

主实验用于对比原始 DLF 与原始 DLF + 选择性空读取，不增加收益监督。
原始 DLF 自带的 `0.01 * HingeLoss` 保持不变，后加的 ATP、SENT 默认关闭。
通过 `lfa_mode` 选择 LFA：

- `original`：原始交叉注意力，默认模式；
- `selective_null`：仅在 A→L 和 V→L 路径增加可学习空 Key，空 Value 固定为零。

```bash
python train.py --dataset mosi --shared-loss-mode original --lfa-mode original
python train.py --dataset mosi --shared-loss-mode original --lfa-mode selective_null
```

测试时使用相同的三个参数。原版继续保存为
`pt/DLFmosi.pth`，新版保存为 `pt/DLF_selective_null_mosi.pth`，互不覆盖。

已有的 ATP/SENT 实现仍然保留。例如 `--shared-loss-mode atp_sent` 可运行
ATP_SENT 消融，其 checkpoint 会带有 `atp_sent` 后缀，不会覆盖上述主实验。

新版额外输出 `null_attn_a` 和 `null_attn_v`，形状均为
`(batch, text_steps)`。它们用于观察最后一层交叉注意力选择空位置的程度，
不应解释为完整模态贡献，因为共享分支仍使用全部模态。

第一轮实验只比较 `original` 与 `selective_null`，两组均从头训练，并保持
数据、种子、原始 HingeLoss、训练预算和验证集选模规则相同。
