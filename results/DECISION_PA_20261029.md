# DECISION PA (pacc BF16 vs FP32) — 29/10/2026

## Question

Le buffer `pacc` de l'attention 2-pass (ggml_q8v, route H2) est stocké en BF16 sur
Volta. Coût du BF16 (troncature 16 bas bits) vs gain (bande-passante halve) ?

## Méthode

- **Candidat PA** : patch de `fattn-q8-volta.cuh` (fork dual-arch), gate
  `GGML_CUDA_Q8V_PA32=1` → pacc en FP32 (160 bits de flottants par élément au lieu
  de 8+8 packés en int16).
- **Validation** (n_kv=107351, T=4, mode h2) :
  - cand_bf16 vs prod : **bit-exact** (0.000e+00) — candidat ne change pas le chemin prod.
  - cand_fp32 vs prod : **relmax=1.66e-03** — bruit de troncature BF16, pas d'erreur.
- **A/B microbench** `mb_fa h2` (T=4, 3 rounds alternés, 100 iters/round, médiane) :

| n_kv | prod BF16 | cand_bf16 (ctrl) | cand_fp32 | Δ FP32 vs prod |
|------|-----------|-------------------|-----------|----------------|
| 107520 | 1556.5 µs | 1554.4 µs (−0.13 %) | 1649.7 µs | **+5.99 %** |
| 2048   | 188.4 µs  | 185.3 µs (−1.65 %) | 258.0 µs  | **+36.94 %** |
| 512    | 49.2 µs   | 49.2 µs  (±0 %)   | 67.6 µs   | **+37.40 %** |

Contrôle de stabilité (3 rounds, ordre inversé fp32→prod, n_kv=107520) : prod 1553.4 µs,
fp32 1655.8 µs → **+6.59 %** — signal reproductible, pas un artefact d'ordre.

## Verdict

**GO pour garder BF16.** Le BF16 est une optimisation DRAM mesurable, pas un artéfact.

- **Grand n_kv (prod, 107 520)** : **+6 %** (±0.6) pour FP32 — au-dessus du seuil de 5 %
  et reproductible dans les 2 ordres ; le pacc = ~11 % du trafic d'attention, doubler
  ce trafic se paie même quand le KV domine.
- **Petit n_kv (512, 2048)** : +37 % — régime où le pacc write/read est un gros
  pourcentage du trafic total ; doubler le buffer pacc coûte 37 % de latence.
- **Précision** : le coût BF16 = troncature des 16 bas bits des partial sums
  (relmax ≈ 1.7e-3, non détectable en pratique ; absorbé par l'online-softmax fp32
  final). Pas d'erreur, juste un bruit de quantification borné.
- **Coût SASS** : PRMT 0x5410 (packing 2 fp32 → 2 bf16) ≈ 30 instructions par warp
  dans `partial_kernel` ; quasi nul face aux 576 HMMA et au load K/V. Le gain vient
  de la DRAM, pas de l'ALU.

**Conclusion** : BF16 = bon choix sur Volta (range dynamique suffisante pour les
partial sums, bande-passante halve). FP32 pacc est un NO-GO : +6 à +37 % de latence
pour un gain de précision non perceptible (relmax 1.7e-3). Le prod est optimal tel
quel.

## Artefacts

- Candidat préservé : `v100/dual-arch/lib/libggml-cuda-dual.so.pa-20261029` (md5 `34c87c5eb`)
- Patch : `v100/dual-arch/scratch/proto_v2/patch_pacc_fp32.py`
- Backup source : `bench/fattn-q8-volta.cuh.fork-8035caf4.bak` (md5 `36f564e2f`)
- JSONL A/B : `bench/mb_pa_ab_20261029-162450.jsonl`
- JSONL A/B contrôle ordre : `bench/mb_pa_ab2_*.jsonl`
- JSONL validation : `bench/val_pa_20261029-162032.jsonl`
- SASS : `/tmp/pa_sass/all47_pa.sass` (46 469 lignes)

## État final

- Fork : reverté (md5 source = `36f564e2f`)
- Prod `.so` : `8035caf41` intact
- GPU : libre
