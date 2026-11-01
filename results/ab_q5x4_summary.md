# A/B applicatif Q5_K T=4 x4 (GGML_CUDA_VOLTA_Q5_X4) — MTP ON config prod — 23/10

date : 2026-10-23 20:26 UTC
reps : 3 · service restauré : OUI
.so prod fa3e75c2 (A=x4 ON defaut, B=x4 OFF)
ctx 36312 (= run9b CUPTI) · -b 2048 · MTP ON (spec-draft-n-max 3) · 64 gen

| leg | rep | decode t/s | Δ vs A | acc | mean len | acc rate/pos |
|---|---|---|---|---|---|---|
| A-x4on | 1 | 32.64 | ? | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |
| B-x4off | 1 | 31.03 | ? | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |
| A-x4on | 2 | 32.58 | ? | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |
| B-x4off | 2 | 32.53 | ? | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |
| A-x4on | 3 | 32.72 | ? | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |
| B-x4off | 3 | 32.62 | ? | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |

## médianes (toutes reps)
  - A-x4on: decode 32.64 t/s
  - B-x4off: decode 32.53 t/s
