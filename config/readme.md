# Shared representation constraints

`config.json` now defaults to replacing DLF's triplet loss with ATP-all and
sentiment-conditioned centroid KL. The model architecture and checkpoint keys
are unchanged. Both losses use `c_l_sim`, `c_a_sim`, and `c_v_sim` (shared
projections), not the private `s_*` features.

Edit `DLF.commonParams`, or override these fields in `DLF.datasetParams.mosi`
or `DLF.datasetParams.mosei` for dataset-specific experiments:

| `shared_loss_mode` | Shared constraint added to the base objective |
| --- | --- |
| `original` | `0.01 * L_m` |
| `none` | None |
| `atp` | `atp_weight * L_ATP` |
| `sent` | `sent_weight * L_sent` |
| `atp_sent` (default) | `atp_weight * L_ATP + sent_weight * L_sent` |
| `original_atp_sent` | All three terms above |

The base objective stays `L_task + 0.1*L_r + 0.1*L_s + 0.01*L_o`.
For older external configurations without `shared_loss_mode`, the trainer falls
back to `original`. The original triplet implementation is preserved.

| Parameter | Initial value | Meaning |
| --- | --- | --- |
| `atp_weight` | `0.01` | Final ATP coefficient in total loss |
| `sent_weight` | `0.01` | Final KL coefficient in total loss |
| `sent_sigma` | `1.0` | Label-distance temperature, strictly positive |
| `sent_tau` | `1.0` | Squared centroid-distance temperature, strictly positive |

These are **untuned starting values**, not recommended optimal settings. Select
them on the validation set. Weights must be finite and nonnegative; temperatures
must be finite and positive. New weights are not multiplied by another `0.1`.

For the new losses only, normalize each projected vector with L2 norm and
`eps=1e-8`. ATP sums squared distances across features and averages across the
three modality pairs and the batch. The centroid is the mean of the three unit
vectors, **without a second normalization**. KL is `KL(q || p)`, with
`q = softmax(-abs(y_i-y_j)/sent_sigma)` and
`p = softmax(-squared_distance(centroid_i,centroid_j)/sent_tau)` over `j != i`.
Continuous labels are training-only targets; all three feature branches retain
gradients. For batches smaller than three, KL returns a differentiable zero.
Gradient accumulation does not enlarge the set of neighbors beyond the current
minibatch.

Training logs record the chosen mode and hyperparameters, plus epoch means of
the raw ATP, KL and triplet losses and their weighted total. Inactive terms are
logged as zero. Evaluation continues to use final-prediction MAE.

Run the numerical and gradient checks in an environment with PyTorch installed:

```bash
python -m unittest discover -s tests -v
```

Training still uses `python train.py` and the existing DLF dataset/dependency
setup. Existing checkpoint filenames are shared by all modes: use separate run
directories or archive checkpoints/results between ablations.
