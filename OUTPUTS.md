# Run Outputs

Record all benchmark and training results here as you run each script.

---

## Task 1 — Data Preparation

### Hidden state generation
- Samples: 3000
- Sequence length: 2048
- Disk usage (`data/hidden_states/`): _TODO_

---

## Task 2 — EAGLE-3 Training

Best checkpoint epoch: _TODO_

| Metric | Reference | Yours |
|---|---:|---:|
| `val/loss_0_epoch` | 2.509 | _TODO_ |
| `val/full_acc_0_epoch` | 0.463 | _TODO_ |
| `val/cond_acc_0_epoch` | 0.463 | _TODO_ |
| `val/loss_1_epoch` | 3.778 | _TODO_ |
| `val/full_acc_1_epoch` | 0.181 | _TODO_ |
| `val/cond_acc_1_epoch` | 0.364 | _TODO_ |
| `val/loss_2_epoch` | 4.550 | _TODO_ |
| `val/full_acc_2_epoch` | 0.069 | _TODO_ |
| `val/cond_acc_2_epoch` | 0.320 | _TODO_ |
| `val/loss_epoch` | 10.837 | _TODO_ |

---

## Task 3 — FP8 Quantization

Quantized model directory: `Qwen3-8B-FP8-Dynamic`

config.json `quantization_config` output:
```
TODO — paste output of 04_quantize_fp8.py here
```

---

## Task 4 — Benchmark Results

### 1. Baseline

```
TODO — paste full vllm bench serve output here
```

### 2. Speculative Decoding (EAGLE-3)

Draft tokens used: _TODO_

```
TODO
```

### 3. FP8 Quantization Only

```
TODO
```

### 4. FP8 + Speculative Decoding

Draft tokens used: _TODO_

```
TODO
```

---

## Draft Token Sweep (Task 4 tuning)

| Draft tokens | Output tok/s | Acceptance rate | Acceptance length | TPOT ms |
|---:|---:|---:|---:|---:|
| 1 | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| 2 | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| 3 | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

Chosen value and justification: _TODO_

---

## Summary Table

| Configuration | Duration s | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | Acceptance rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | N/A |
| Speculative decoding | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| FP8 quantization | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | N/A |
| FP8 + speculative decoding | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

---

## Main Question Answer

**Which should be done first: speculative decoding training or quantization?**

_TODO — fill in after running experiments_
