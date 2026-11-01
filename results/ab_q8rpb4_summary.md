# A/B applicatif Q8_0 T=4 rpb4 (candidat q8x4) — MTP ON config prod — 23/10

date : 2026-10-23 19:17 UTC
reps : 2 · service restauré : OUI
.so prod fa3e75c2 (A) · .so cand 8425c68b (B,C)
ctx 36312 (= run9b CUPTI) · -b 2048 · MTP ON (spec-draft-n-max 3) · 64 gen

| leg | rep | decode t/s | Δ vs A | acc | mean len | acc rate/pos |
|---|---|---|---|---|---|---|
| A-prod | 1 | 30.44 | +0.0% | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |
| B-rpb4 | 1 | 32.69 | +7.4% | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |
| C-rpb4x4 | 1 | 32.65 | +7.3% | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |
| A-prod | 2 | 32.62 | +0.0% | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |
| B-rpb4 | 2 | 32.62 | +0.0% | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |
| C-rpb4x4 | 2 | 32.7 | +0.2% | 0.83333 | 3.5 | (1.000, 0.862, 0.759) |

## médianes (toutes reps)
  - A-prod: decode 32.62 t/s
  - B-rpb4: decode 32.69 t/s
  - C-rpb4x4: decode 32.7 t/s
