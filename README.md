# mmvq — GEMV lecture des poids Q6_K W4R4 (H3 / W2R4 / PA) → CLOS

**Verdict : CLOS (29/10) — 3 pistes testées, toutes < seuil 5 %, périmètre fermé.**
Le `.so` prod `8035caf4` (de l'époque) est optimal sur ce périmètre.

## Question

Le kernel GEMV `mmvq` (dépaquetage Q6_K, `q6_f17408x5120`) représente **28,9 % du
GPU kernel** du decode. NCU montre **59,3 % DRAM / 505 GB/s** = quasi mur HBM2
(V100 ~794 GB/s mesurés, mais le Q6_K lit des octets utiles + overhead). 3 pistes
restantes listées dans `DECISION_H3` ont été tranchées en une session : H3,
W2R4, PA.

## Les 3 pistes, 3 NO-GO

### H3 — hoist du dépaquetage Q6_K (désimbrication des boucles) — **NO-GO**
Hoister le dépaquetage `q6_K` (les `qk_scale`/`qs` par super-block) hors la boucle
interne, pour réduire la pression register (le kernel tourne à **96 regs**,
2 blocks/SM max). Mesure microbench `mb_mmvq` (T=4, GEMV Q6_K, .so h3 vs prod) :

| shape | prod 185.3 µs | H3 | Δ |
|---|---|---|---|
| q6_f17408x5120 | 185,3 | 181,1 | **−2,3 %** (le meilleur des 3) |

−2,3 % **< seuil GO 5 %** → NO-GO. Le hoist aide marginalement mais le kernel
reste au mur HBM2 (lit les mêmes octets). `DECISION_H3_20261029.md`.

### W2R4 — nwarps 4→2 sur Q6_K — **NO-GO**
Réduire le parallélisme par block (nwarps 4→2) pour augmenter l'occupancy
(moins de regs/warp → plus de blocks). Hypothèse : le kernel est register-bound
(96 regs). Mesure : **−0,76 %** (bruit) → NO-GO. **Prouve que les 96 regs ne
viennent pas d'un paramètre simple (nwarps) mais des 16 accumulateurs `tmp[4][4]`
+ du dépaquetage q6_K** — un floor structurel, pas un réglage. `DECISION_W2R4_20261029.md`.

### PA — pacc BF16 vs FP32 (attention, hors scope « poids » mais vérifié) — **GO garder BF16**
Vérification que l'accumulateur BF16 de l'attention (path `q8v14partial`) est déjà
optimal (pas de régression par rapport à FP32) → garder BF16. Confirme que l'attention
n'est pas un levier poids. `DECISION_PA_20261029.md`.

## Pourquoi CLOS (le raisonnement fondamental)

1. **Q6_K / Q5_K = mur HBM2** (59-66 % DRAM) : aucun micro-kernel ne peut lire
   moins d'octets sans changer le format du poids (hors périmètre) → neutre par
   construction. Q5_K x4 déjà testé = **neutre** (`ab_q5x4_summary.md`).
2. **Q8_0 = L1-throughput-bound (90,4 %), non DRAM-bound** : seul non au mur HBM2,
   MAIS regs 72 = floor des 16 accumulateurs, q8rpb4 (rows=4) déjà testé = **neutre**
   (`ab_q8rpb4_summary.md`), nwarps Q8_0 = même prédiction NO-GO que W2R4.
3. **7+ variantes testées sur le périmètre**, toutes neutres ou lentes :
   42d9581 (nwarps), q6x4, q6rp1/2/8 (rows-per-block), q5x4, q8rpb4, W2R4, H3.
4. Le decode est au **mur HBM2** (94-100 % du BW théo) — l'unique levier avec
   headroom mesurable est **hors périmètre poids** : réduire les steps/token
   (= acceptance MTP, cf `repos/mtp-nmax/`) ou la réécriture structurelle
   <128 regs (classe « projet, ROI incertain »).

## Contenu

```
src/   # patch_h3.py, patch_w2r4.py, patch_pacc_fp32.py (les 3 patchs), q6_mmvq_proto.cu (proto)
results/  # DECISION_H3/W2R4/PA, mmvq_CLOS_note.md, mmvq_weightload_CLOS_final_20261029.md,
          # RESOLUTION_q8v14_20261029.md, mb_w2_ab_*.jsonl, val_w2b_*.jsonl,
          # mb_mmvq_*.jsonl, ab_q5x4_*, ab_q8rpb4_*, ab_confirm_results.jsonl
bench/  # mb_mmvq(.cu/.bin), ab_mmvq2.py, ab_q5x4.py, ab_q8rpb4.py, ncu_mmvq_gemv_recon2.py, ab_confirm.py
```

## `.so` associés

| md5 | état |
|---|---|
| `8035caf4` | prod de l'époque (T3+vldg, avant cvec) — référence « optimal » de ce périmètre |
| `*.h3-20261029` | H3 candidat (NO-GO) |
| `*.w2-20261029` | W2R4 candidat (NO-GO) |

## Re-validation

1. **Microbench** (rapide, dlopen du .so, pas de rebuild) :
   `python3 bench/ab_mmvq2.py --so <md5> --shapes q6_f17408x5120` → comparer à
   `results/mb_w2_ab_20261029-174553.jsonl` (prod 185,3 µs ; H3 −2,3 %).
2. **NCU** (pour le diagnostic, pas le verdict) :
   `python3 bench/ncu_mmvq_gemv_recon2.py --so <md5>` → DRAM % / regs / SM %.
   Verdict = le DRAM % (si ~60 % → mur HBM2, pas de levier kernel).

## Caveats

- Le seuil GO est **5 % e2e sur la shape dominante** ; −2,3 % microbench H3 est
  déjà < 5 % et ne se transférerait pas tel quel en e2e (le GEMV est 28,9 % du
  kernel, pas du e2e).
- Le périmètre est **strictement « lecture des poids »** (contrainte de la
  mission) : l'attention, le shortlist, le quantize (KV) sont hors scope.
- « CLOS » ≠ « parfait » : c'est « aucun levier micro-kernel > 5 % trouvé sur ce
  périmètre, 7+ variantes testées ». Un changement de format de poids (Q5_0,
  Q4_0, etc.) rouvrirait la question, mais c'est un autre sujet.
