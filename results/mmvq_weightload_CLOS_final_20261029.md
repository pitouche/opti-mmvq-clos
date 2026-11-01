# mmvq / lecture-des-poids GEMV — CLOS FINAL (29/10) : plus aucun levier kernel > 5 %

> Suite immédiate de `DECISION_W2R4_20261029.md`. Après le NO-GO du nwarps 4→2,
> j'ai passé en revue les **3 directions restantes** listées dans
> `DECISION_H3` (§ « Prochaine direction ») et la note de scope
> `RESOLUTION_q8v14` (périmètre = lecture des poids, pas du KV).
> **Conclusion : le périmètre « lecture des poids » est CLOS définitivement.**

## Les 3 options restantes, tranchées

### (a) Attaquer les 96 regs (partition K / réécriture structurelle) — CLOS
C'était la tâche `pk_*` (patch partition-K `GGML_CUDA_Q6_PK`, gate SASS regs<64).
**Abandonné en amont du build, avec des raisons :**
1. **Ce n'est pas un A/B de paramètre** : casser les 96 regs exige de changer le
   schéma de partition de K (moins de kbx par lane, ou un autre découpage de la
   ligne) — une réécriture du kernel, hors périmètre « bench A/B d'un paramètre »
   de la mission.
2. **ROI incertain sur un kernel au mur HBM2** : `ncu_mmvq_gemv_recon2` montre
   q6_f17408x5120 à **59.3 % DRAM, 66 % SM, 505 GB/s** = mur de bande passante.
   Moins de regs ne fait pas lire moins d'octets. (C'est exactement le raisonnement
   de `mmvq_CLOS_note` §3 : « est-ce que ça lit moins d'octets ? Sinon → neutre par
   construction ».)
3. **Précédent direct** : le sweep `q6rp1/rp2/rp8` (rows-per-block, qui modifie
   aussi regs) a donné **toutes plus lentes** (219.1/197.6/195.6 vs prod 185.3 µs)
   — `ncu_q6rp_20261023-153550.txt`. Jouer sur la forme du block a déjà été
   réfuté ; la partition K est le même type de changement.
4. W2R4 (même jour) a **prouvé** que les 96 regs ne viennent pas d'un paramètre
   simple (nwarps) mais des **16 accumulateurs `tmp[4][4]` + le dépaquetage q6_K**
   — un floor structurel.

### (b) Re-scope la cible (Q8_0 11.2 % / Q5_K 8.3 %) — CLOS
Les 3 cibles in-scope au T=4 (ncu recon2) :

| shape | regs | DRAM % | L1/TEX % | L2 % | SM % | limiter |
|---|---:|---:|---:|---:|---:|---|
| q6_f17408x5120 | 96 | 59.3 | 42.9 | 22.8 | 66.0 | **BW-wall HBM2** |
| q5_f17408x5120 | 96 | 53.5 | 59.4 | 20.1 | 64.5 | **BW-wall HBM2** |
| q8_f17408x5120 | 72 | **14.5** | **90.4** | 23.1 | 63.0 | **L1/latence** |

- **Q6_K / Q5_K** = mur HBM2 → aucun micro-kernel n'a de headroom au-dessus du BW.
  (Q5_K x4 déjà testé 23/10 = **neutre**, `ab_q5x4_summary.md`.)
- **Q8_0** = **L1-throughput-bound (90.4 %), non DRAM-bound** : c'est le seul qui
  n'est pas au mur HBM2. MAIS : (i) regs 72 = floor des **16 accumulateurs
  `tmp[4][4]`** (l'unpack q8_0 est minuscule, W2R4 prouve que c'est l'accumulateur
  qui fait les regs, pas le dépaquetage) ; (ii) **q8rpb4 (rows=4) déjà testé 23/10
  = NEUTRE** (`ab_q8rpb4_summary.md` : 32.69 vs 32.62 t/s, +0.2 % médiane) ;
  (iii) nwarps Q8_0 = même prédiction NO-GO que W2R4 (les regs viennent des
  accumulateurs). → **Aucun levier Q8_0 plausible > 5 %.**
- **Q8_0 n'est que 11.2 % du GPU kernel** : même un x2 (improbable) ne ferait
  qu'~5.6 % du kernel, et e2e c'est noyé par la variance d'acceptance MTP.

### (c) Le reste (~30 %) est hors périmètre « poids »
Par `RESOLUTION_q8v14` + contrainte user (« Cible mémoire : lecture des poids,
pas du KV ») : l'attention (q8v14partial 30.1 % + reduce + volta_small ≈ 34 %),
le shortlist et la quantize **lisent du KV / ne sont pas des GEMV poids** → hors
scope. (PA a confirmé ce matin que le pacc BF16 de l'attention est déjà optimal.)

## Verdict consolidé
**Le périmètre « lecture des poids » (mmvq GEMV : Q6_K 28.9 %, Q8_0 11.2 %,
Q5_K 8.3 %) est CLOS.** Sur ce périmètre, **7+ variantes** ont été testées et
toutes sont neutres ou lentes (42d9581, q6x4, q6rp1/2/8, q5x4, q8rpb4, W2R4,
H3). La pression register est **structurelle** (16 accumulateurs + dépaquetage),
pas un paramètre réglable. Le decode est au **mur HBM2** (94–100 % du BW théo).

**Aucun levier micro-kernel in-scope n'atteint 5 %.**

## Conséquence stratégique
L'unique levier decode avec headroom mesurable reste **hors périmètre poids** :
**réduire les steps/token = acceptance MTP** (cf `mtp_defer_diagnostic`,
`mmvq_CLOS_note` §2.2) — ou, plus loin, la réécriture structurelle <128 regs /
KV q4_0 déjà classées « projet, ROI incertain / hors périmètre » dans
`BILAN_GLOBAL_3SEM` §sem.5.

**Recommandation : ne pas relancer de bench mmvq/GEMV in-scope.** La mission
kernel « lecture des poids » est terminée ; le `.so` prod `8035caf4` est optimal
sur ce périmètre.

## Artefacts de référence
- `DECISION_W2R4_20261029.md` (nwarps 4→2, NO-GO)
- `DECISION_H3_20261029.md` (hoist unpack, NO-GO)
- `DECISION_PA_20261029.md` (pacc BF16 vs FP32, attention hors scope)
- `ncu_mmvq_gemv_recon2_summary.md` (regs/bound des 3 cibles in-scope)
- `ncu_q6rp_20261023-153550.txt`, `ab_q5x4_summary.md`, `ab_q8rpb4_summary.md`
- `mmvq_CLOS_note.md` (la clôture initiale 24/10)
