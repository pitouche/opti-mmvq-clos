#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ncu_mmvq_gemv_recon.py — Phase 1 « attaque GEMV verify » : NCU fraîche sur le .so PROD 8035caf4.

Cible mémoire : LECTURE DES POIDS (GEMV), pas du KV.
Objet profilé : les kernels mul_mat_vec_q T=4 des 6 shapes FFN réelles du 27B,
exécutés par le microbench mb_mmvq (dlopen du .so prod via MB_MMVQ_SO, CVD=0, W4R4=1).

Pourquoi mb_mmvq et pas le serveur : mêmes kernels (mul_mat_vec_q<type,4,...>),
mêmes shapes, mêmes configs de lancement, contexte GPU exclusif sous NCU
(sans replay des ~40 000 lances du serveur). La durée sous NCU est recalibrée
vs le jsonl non profilé (133-135 µs) pour extraire le surcoût du profiler.

Protocole superviseur (supervisor_bench_README.md) :
  - lock tenu par le superviseur (pas de relock ici) ;
  - stop ollama-v100 → NCU (sudo, V100 seule, CVD=0) → restauration → vérif active ;
  - AUCUN swap du .so prod : le .so est lu via dlopen dans mb_mmvq, jamais copié.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run ncu_mmvq_gemv_recon.py --detach
"""
import json
import os
import re
import sys
import time
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "ncu_recon_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]
exec(src, ns)

NCU = "/home/guile/ncu-2024.1.1/usr/lib/nsight-compute/ncu"
MB = os.path.join(D, "mb_mmvq")
SO = ns["SO_PROD"]  # 8035caf4

OUT_DIR = "/tmp/ncu_mmvq_recon"
OUT_SUMMARY = os.path.join(D, "ncu_mmvq_gemv_recon_summary.md")

# NOTE : LC_ALL=C forcé pour des décimales à point (sinon virgule locale FR).
METRICS = (
    "dram__bytes_read.sum,"
    "dram__bytes_write.sum,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
    "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,"
    "sm__warps_active.avg.pct_of_peak_sustained_active,"
    "gpu__time_duration.sum,"
    "gpc__cycles_elapsed.max"
)

# métriques attendues : clé = nom de base (préfixe), le suffixe d'agrégation varie (.sum/.avg)
WANT = [
    "dram__bytes_read",
    "dram__bytes_write",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__time_duration",
    "gpc__cycles_elapsed",
]

# NCU exprime les valeurs dans une "unité jolie" (Mbyte, Gbyte, usecond, ...) : il faut
# la lire pour ramener en unité canonique (octets, µs, %).
UNIT_SCALE = {
    "B": 1, "byte": 1, "bytes": 1,
    "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
    "usecond": 1, "ns": 1e-3, "us": 1, "ms": 1e3, "s": 1e6,
    "%": 1, "cycle": 1, "cycles": 1, "": 1,
}

# (shape_filter mb_mmvq, famille, K, rows) — les 6 shapes FFN réelles, T=4 uniquement.
SHAPES = [
    ("q6_f17408x5120", "Q6_K", 5120, 17408),
    ("q6_f5120x17408", "Q6_K", 17408, 5120),
    ("q8_f17408x5120", "Q8_0", 5120, 17408),
    ("q8_f5120x17408", "Q8_0", 17408, 5120),
    ("q5_f17408x5120", "Q5_K", 5120, 17408),
    ("q5_f5120x17408", "Q5_K", 17408, 5120),
]

# volumes théoriques des poids (lecture minimale, B/élément)
BPE = {"Q6_K": 210.0 / 256, "Q8_0": 34.0 / 32, "Q5_K": 176.0 / 256}

log = ns["log"]


def ncu_one(shape):
    raw = os.path.join(OUT_DIR, "ncu_%s.raw.txt" % shape)
    # sudo env CVD=0 = V100 ; MB_MMVQ_SO = prod (dlopen, pas de copie).
    full = (
        ["sudo", "env", "LC_ALL=C", "CUDA_VISIBLE_DEVICES=" + str(v100_cuda_ordinal()), "MB_MMVQ_SO=" + SO, NCU,
         "--launch-skip", "5", "--launch-count", "5",
         "--kernel-name", "regex:mul_mat_vec_q",
         "--metrics", METRICS]
        + [MB, "60", shape, "4"]
    )
    log("  · NCU %s (5 lances)" % shape)
    r = ns["run"](full, timeout=900)
    with open(raw, "w") as f:
        f.write("CMD : " + " ".join(full) + "\nRC=%d\n" % r.returncode + r.stdout + "\nSTDERR:\n" + r.stderr)
    if r.returncode != 0:
        log("    !! NCU rc=%d" % r.returncode)
    return raw


def parse_raw(raw, shape):
    txt = open(raw).read()
    if "Profiling" not in txt:
        log("    !! pas de section Profiling dans %s" % shape)
        return []
    out = []
    blocks = re.split(r"==PROF== Profiling", txt)[1:]
    for b in blocks:
        lines = b.splitlines()
        # header info : chercher dans les 6 premieres lignes (le nom de kernel
        # peut etre separe par des lignes de JSON/progress du harnais)
        head = "\n".join(lines[:6])
        gtype = ggrid = gblock = "?"
        mh = re.search(r"mul_mat_vec_q<\(ggml_type\)(\d+), \(int\)(\d+)", head)
        if mh:
            gtype = mh.group(1)
        mg = re.search(r"\)\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*x\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", head)
        if mg:
            ggrid = "%s,%s,%s" % (mg.group(1), mg.group(2), mg.group(3))
            gblock = "%s,%s,%s" % (mg.group(4), mg.group(5), mg.group(6))
        metrics = {}
        for line in lines:
            # ex : "    dram__bytes_read.sum      Mbyte      73.21"
            mm = re.match(r"^\s+([a-z][a-z0-9_.]+)\s+([A-Za-z%_]+)\s+(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*$", line)
            if not mm:
                continue
            name, unit, val = mm.group(1), mm.group(2), mm.group(3)
            try:
                v = float(val.replace(",", "")) * UNIT_SCALE.get(unit, 1)
            except ValueError:
                continue
            for w in WANT:
                if name == w or name.startswith(w + "."):
                    metrics[w] = v
                    break
        if "gpu__time_duration" in metrics:
            out.append({"shape": shape, "type": gtype, "grid": ggrid, "block": gblock, "m": metrics})
    return out


def main():
    import signal
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])

    log("NCU recon GEMV T=4 — .so prod (md5 %s) — lock tenu par le superviseur"
        % __import__("hashlib").md5(open(SO, "rb").read()).hexdigest()[:8])
    os.makedirs(OUT_DIR, exist_ok=True)

    ns["stop_and_free"]()

    results = []
    try:
        for shape, fam, K, rowsn in SHAPES:
            raw = ncu_one(shape)
            if not raw:
                continue
            results.extend(parse_raw(raw, shape))
    finally:
        ns["restore_service"]()

    # ---------- analyse ----------
    per_shape = {}
    for shape, fam, K, rowsn in SHAPES:
        recs = [r for r in results if r["shape"] == shape]
        if not recs:
            log("  !! %s : 0 lance analysée" % shape)
            continue

        def med(k):
            vs = sorted(r["m"][k] for r in recs if k in r["m"])
            return vs[len(vs) // 2] if vs else None

        dur = med("gpu__time_duration")
        rb = med("dram__bytes_read")
        wb = med("dram__bytes_write")
        theo_mb = K * rowsn * BPE[fam] / 1e6
        bw = (rb + wb) / dur / 1e6 if (dur and rb is not None) else None  # Mo/µs = GB/s
        per_shape[shape] = {
            "shape": shape, "fam": fam, "K": K, "rows": rowsn, "n": len(recs),
            "grid": recs[0]["grid"], "block": recs[0]["block"],
            "dur_us": dur,
            "dram_r_mb": (rb / 1e6) if rb is not None else None,
            "dram_w_mb": (wb / 1e6) if wb is not None else None,
            "bw_gbs": bw,
            "dram_pct": med("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            "occ_pct": med("sm__warps_active.avg.pct_of_peak_sustained_active"),
            "l2miss_ld_mb": (med("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld") or 0) / 1e6,
            "ghz": (med("gpc__cycles_elapsed") / dur / 1e3) if (dur and med("gpc__cycles_elapsed")) else None,
            "theo_w_mb": theo_mb,
            "dram_vs_theo": (rb / (theo_mb * 1e6)) if (rb and theo_mb) else None,
        }

    # ---------- outputs ----------
    with open(os.path.join(OUT_DIR, "ncu_mmvq_recon.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT_DIR, "ncu_mmvq_recon_summary.jsonl"), "w") as f:
        for shape in sorted(per_shape):
            f.write(json.dumps(per_shape[shape]) + "\n")

    L = ["# NCU recon GEMV T=4 — .so prod 8035caf4 (mb_mmvq, 5 lances/shape)", "",
         "date : " + time.strftime("%Y-%m-%d %H:%M UTC"),
         "service restauré : %s" % ("OUI" if ns["_restored"] else "NON"), "",
         "Cible : LECTURE DES POIDS (GEMV). T=4 (verify MTP). 6 shapes FFN réelles 27B.", "",
         "| shape | type | grid | block | dur µs | DRAM_r MB | BW GB/s | %pic DRAM | occ % | L2miss MB | théo poids MB | DRAM/théo | GHz |",
         "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for shape in sorted(per_shape):
        s = per_shape[shape]
        L.append("| %s | %s | %s | %s | %.0f | %.1f | %.0f | %.1f | %.1f | %.1f | %.1f | %.2f | %.2f |" % (
            s["shape"], s["fam"], s["grid"], s["block"],
            s["dur_us"] or 0, s["dram_r_mb"] or 0, s["bw_gbs"] or 0,
            s["dram_pct"] or 0, s["occ_pct"] or 0, s["l2miss_ld_mb"] or 0,
            s["theo_w_mb"] or 0, s["dram_vs_theo"] or 0, s["ghz"] or 0))
    L.append("")
    L.append("Référence : pic HBM2 V100 ≈ 900 GB/s. Durée non profilée (mb_mmvq) : "
             "Q6_K ~133-135 µs, Q8_0 ~100 µs, Q5_K ~115-128 µs (T=4).")
    L.append("Le ratio durée_NCU / durée_non_profilée = surcoût du profiler (≈7 % sur le KOLBO E1).")
    L.append("")
    L.append("Artefacts : %s/ncu_<shape>.raw.txt + ncu_mmvq_recon.jsonl" % OUT_DIR)
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")

    log("FIN — %d lances profilées — %s" % (len(results), OUT_SUMMARY))
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
