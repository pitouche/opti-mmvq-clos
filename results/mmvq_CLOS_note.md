# mmvq — levier CLOS (24/10, 03:40 UTC) : le decode est au mur HBM2, aucune variante ne bat prod

## Contexte
run9d (CUPTI, ctx 90k, decode 512 tok) : mmvq = 69.8 % du busy GPU (7 870 ms/512), dont
Q6_K 4 172 ms (52.9 %), Q8_0 1 634 ms, Q5_K 1 205 ms. → « LE levier kernel » à valider.

## Décision : CLOS, pas de candidate à déployer

### 1. Le pas de decode est au mur de bande passante (preuve par les nombres)
- Blob modèle qwen3.8-27b-q6 : **23 088 MB** (vérifié).
- HBM2 V100 : 877 MHz × 3072 bits ≈ **900 GB/s** (max théorique 909).
- Toit BW (poids seuls) : 23 088 / 900 = **25.65 ms/step**.
- KV q8_0 @ 90k ajoutée (~2 × 475 MB) : ≈ +1.05 ms → toit ≈ **26.7 ms/step**.
- **Mesuré run9d : 27.27 ms/step → 94–100 % du toit HBM2.**

Conséquence : chaque µs gagnée dans un kernel mmvq doit être payée en moins d'octets lus.
Tant que le poids q6/q8 est lu une fois par step (déjà le cas), aucun micro-kernel
n'a de headroom au-dessus du BW. Seul un changement de régime réduit le pas :
moins de steps/token (acceptance MTP) ou moins d'octets/step (quantization, risque qualité).

### 2. Toutes les variantes mmvq testées sont NEUTRES ou PLUS LENTES (preuves)
| variante | cible | résultat | source |
|---|---|---|---|
| 42d9581 (nwarps K-quant 4→2) | mmvq T=4 | **neutre** (A/B 17/10, 10 pts ctx ; variance baseline 10-24 % >> Δ) | ab_mmvq2_CONSOLIDE.md |
| q6x4 (vec_dot Q6_K x4, decode 1x/reuse 4 colonnes) | Q6_K T=4 | **neutre-lent** (même run ncu : 202.8 vs 204.8 µs ; le gain de trafic poids est annulé par perte de parallélisme) | ncu_q6x4_20261023-151542.txt |
| q6rp1 / rp2 / rp8 (rows-per-block sweep) | Q6_K T=4 | **toutes lentes** : 219.1 / 197.6 / 195.6 vs prod 185.3 µs | ncu_q6rp_20261023-153550.txt |
| Q5_K x4 (q8x4, gate GGML_CUDA_VOLTA_Q5_X4) | Q5_K T=4 | **neutre** : 32.64 vs 32.53 t/s (A/B env sur .so prod, 3 reps) ; microbench 124.9 vs 128.0/127.0 µs (sous bruit) | ab_q5x4_summary.md, mb_mmvq_q5_*.jsonl |
| Q6_W4R4 (4 lines/rows, 4 warps) | Q6_K T=4 | **déjà en prod** (env GGML_CUDA_VOLTA_Q6_W4R4=1 + .so fa3e75c2 avec le branch dans mmvq.cu) — c'est le baseline de tous les tests ci-dessus | ollama-v100.service |

Note : les nombres ncu_* (185-205 µs) diffèrent du jsonl pur (135 µs) car ncu flush/serialise ;
la comparaison A/B est faite DANS le même fichier ncu → le verdict (candidats lents) est robuste.

### 3. Pourquoi les variantes échouent (cohérent avec le mur BW)
- q6x4 factorise la lecture du poids sur 4 colonnes : théoriquement −75 % de trafic poids
  Q6_K… mais le microbench montre +50 % lent → le gain de trafic est annulé par
  (a) la perte de parallélisme (moins de blocks actifs), (b) le padding/shared du vec_dot x4.
  Sur V100 (L2 61 Mo, poids partiellement L2-warm entre appels), le re-read n'était pas le goulou.
- 42d9581 réduit les warps K-quant : le quant est déjà en sous-capacité (BW-bound, pas FLOP-bound).
- W4R4 (prod) : c'était le seul gain réel trouvé sur mmvq (≈ +1 % court ctx, BILAN 18/10) → déjà exploité.

## Conséquences stratégiques
1. **mmvq CLOS en tant que levier kernel.** Ne pas retravailler vecdotq/mmvq sur cette V100
   tant que la quantization du modèle reste q6/q8 (le mur BW est invariant).
2. L'unique levier decode avec headroom mesurable = **réduire les steps/token** :
   **acceptance MTP** (prod 2.67-2.90 ; ×1.47 vs un hypothétique 4.0). C'est `mtp-accept`.
   Si 2.68 → 3.5 : +30 % t/s sans toucher un seul kernel.
3. Seconde piste (hors kernel) : préfill (735 t/s, pas le goulot) — rien.
4. Tout bench mmvq futur doit d'abord répondre : « est-ce que ça lit moins d'octets ? »
   Sinon → neutre par construction.

## Artefacts
- ncu_q6x4_20261023-151542.txt, ncu_q6rp_20261023-153550.txt (microbench sous ncu, mêmes runs prod/candidat)
- mb_mmvq_prod_20261023.jsonl, mb_mmvq_q5_*.jsonl (microbench pur)
- ab_mmvq2_CONSOLIDE.md (A/B 42d9581), ab_q5x4_summary.md (A/B Q5 x4)
- .so candidats (gardés, hors prod) : libggml-cuda-42d9581.so (0500fb9a), -q6x4 (6c211f0a),
  -q6rp1/2/8, -q8x4 (8425c68b)
