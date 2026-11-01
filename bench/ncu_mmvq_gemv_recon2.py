#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ncu_mmvq_gemv_recon2.py — Phase 1b : NCU SECTIONNES (full) sur les 6 shapes GEMV T=4.

Recon1 : DRAM/duree/occ. Recon2 : LIMITERS d'occupation (regs/smem/warps), WCPI,
SpeedOfLight complet, pour choisir la variante kernel.

Recon1 a montre : Q6_K T4 DRAM=59% pic, occ theo 31% (limite regs+smem), "well-balanced".
Recon2 confirme et etend aux 6 shapes.

Meme harnais (mb_mmvq + .so prod 8035caf4 via MB_MMVQ_SO, CVD=0). 3 lances/shape.
Sorties : /tmp/ncu_mmvq_recon2/ + summary.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run ncu_mmvq_gemv_recon2.py --detach
"""
import json
import os
import re
import sys
import time
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "ncu_recon2_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]
exec(src, ns)

NCU = "/home/guile/ncu-2024.1.1/usr/lib/nsight-compute/ncu"
MB = os.path.join(D, "mb_mmvq")
SO = ns["SO_PROD"]

OUT_DIR = "/tmp/ncu_mmvq_recon2"
OUT_SUMMARY = os.path.join(D, "ncu_mmvq_gemv_recon2_summary.md")

SECTIONS = [
    "LaunchStats", "Occupancy", "SpeedOfLight", "WarpStateStats",
    "MemoryWorkloadAnalysis", "ComputeWorkloadAnalysis", "SchedulerStats",
]

SHAPES = [
    "q6_f17408x5120", "q6_f5120x17408",
    "q8_f17408x5120", "q8_f5120x17408",
    "q5_f17408x5120", "q5_f5120x17408",
]

# (cle, label EXACT NCU, section) — la section sert a ne pas confondre
# "Memory Throughput" (SOL, %) vs "Memory Throughput Gbyte/second" (MemWA).
FIELDS = [
    ("dur_us",        "Duration",                          "solsol"),
    ("sm_pct",        "Compute (SM) Throughput",           "solsol"),
    ("mem_pct",       "Memory Throughput",                 "solsol"),
    ("dram_pct",      "DRAM Throughput",                   "solsol"),
    ("l1_pct",        "L1/TEX Cache Throughput",           "solsol"),
    ("lts_pct",       "L2 Cache Throughput",               "solsol"),
    ("sm_ghz",        "SM Frequency",                      "solsol"),
    ("bw_gbs",        "Memory Throughput Gbyte/second",    "memwa"),
    ("mem_busy",      "Mem Busy",                          "memwa"),
    ("maxbw_pct",     "Max Bandwidth",                     "memwa"),
    ("l1_hit",        "L1/TEX Hit Rate",                   "memwa"),
    ("l2_hit",        "L2 Hit Rate",                       "memwa"),
    ("one_or_more",   "One or More Eligible",              "sched"),
    ("issued_warp",   "Issued Warp Per Scheduler",         "sched"),
    ("no_elig",       "No Eligible",                       "sched"),
    ("act_warps_sched","Active Warps Per Scheduler",       "sched"),
    ("wcpi",          "Warp Cycles Per Issued Instruction","warp"),
    ("avg_act_thr",   "Avg. Active Threads Per Warp",      "warp"),
    ("block_size",    "Block Size",                        "launch"),
    ("grid_size",     "Grid Size",                         "launch"),
    ("regs",          "Registers Per Thread",              "launch"),
    ("smem_static_kb","Static Shared Memory Per Block",    "launch"),
    ("n_sms",         "# SMs",                             "launch"),
    ("waves",         "Waves Per SM",                      "launch"),
    ("lim_sm",        "Block Limit SM",                    "occ"),
    ("lim_regs",      "Block Limit Registers",             "occ"),
    ("lim_smem",      "Block Limit Shared Mem",            "occ"),
    ("lim_warps",     "Block Limit Warps",                 "occ"),
    ("theo_warps",    "Theoretical Active Warps per SM",   "occ"),
    ("theo_occ",      "Theoretical Occupancy",             "occ"),
    ("ach_occ",       "Achieved Occupancy",                "occ"),
    ("ach_warps",     "Achieved Active Warps Per SM",      "occ"),
]

# section header (texte apres "Section:") -> tag
SEC_TAG = {
    "GPU Speed Of Light Throughput": "solsol",
    "Memory Workload Analysis": "memwa",
    "Scheduler Statistics": "sched",
    "Warp State Statistics": "warp",
    "Launch Statistics": "launch",
    "Occupancy": "occ",
    "Compute Workload Analysis": "comp",
}

log = ns["log"]


def ncu_one(shape):
    raw = os.path.join(OUT_DIR, "ncu_%s.full.txt" % shape)
    # reutilisation d'un fichier deja profile (run precedent crash, fichier valide)
    if os.path.exists(raw):
        t = open(raw).read()
        if "RC=0" in t.split("\n")[1] if len(t.split("\n")) > 1 else False and t.count("Section: Launch Statistics") >= 3:
            log("  · %s : fichier existant reutilise" % shape)
            return raw
    cmd = ["sudo", "env", "LC_ALL=C", "CUDA_VISIBLE_DEVICES=" + str(v100_cuda_ordinal()), "MB_MMVQ_SO=" + SO, NCU,
           "--launch-skip", "2", "--launch-count", "3",
           "--kernel-name", "regex:mul_mat_vec_q"]
    for s in SECTIONS:
        cmd += ["--section", s]
    cmd += [MB, "50", shape, "4"]
    log("  · NCU full %s (3 lances)" % shape)
    r = ns["run"](cmd, timeout=1800)
    with open(raw, "w") as f:
        f.write("CMD : " + " ".join(cmd) + "\nRC=%d\n" % r.returncode + r.stdout + "\nSTDERR:\n" + r.stderr)
    if r.returncode != 0:
        log("    !! NCU rc=%d" % r.returncode)
    return raw


def grab(b, label, tag):
    """Valeur premiere occurrence de `label` DANS la section `tag` du block b."""
    # isole le chunk de section
    idx = [m.start() for m in re.finditer(r"Section: ([^\n]+)", b)]
    chunks = []
    for i, m in enumerate(re.finditer(r"Section: ([^\n]+)", b)):
        name = m.group(1).strip()
        end = idx[i + 1] if i + 1 < len(idx) else len(b)
        chunks.append((SEC_TAG.get(name, "?"), b[m.start():end]))
    for ctag, chunk in chunks:
        if ctag != tag:
            continue
        m = re.search(re.escape(label) + r"\s+[^0-9\-]*?(-?\d[\d,]*(?:\.\d+)?)", chunk)
        if not m:
            return None
        v = m.group(1).replace(",", "")
        try:
            return float(v)
        except ValueError:
            return None
    return None


def parse_full(raw, shape):
    txt = open(raw).read()
    # chaque lance = un chunk demarre par la signature du kernel ; chaque chunk
    # contient UNE serie de sections (SOL, MemWA, Scheduler, Warp, Launch, Occ, Comp).
    chunks = re.split(r"void mul_mat_vec_q<", txt)[1:]
    recs = []
    for b in chunks:
        if "Section: Launch Statistics" not in b:
            continue
        rec = {"shape": shape}
        mh = re.search(r"mul_mat_vec_q<\(ggml_type\)(\d+), \(int\)(\d+)", b)
        if mh:
            rec["type"] = mh.group(1)
        mg = re.search(r"\)\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*x\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", b)
        if mg:
            rec["grid"] = ",".join(mg.groups()[:3])
            rec["block"] = ",".join(mg.groups()[3:])
        for key, label, tag in FIELDS:
            rec[key] = grab(b, label, tag)
        recs.append(rec)
    return recs


def main():
    import signal
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])

    log("NCU recon2 (sections) GEMV T=4 — .so prod (md5 %s)"
        % __import__("hashlib").md5(open(SO, "rb").read()).hexdigest()[:8])
    os.makedirs(OUT_DIR, exist_ok=True)

    ns["stop_and_free"]()

    results = []
    try:
        for shape in SHAPES:
            raw = ncu_one(shape)
            if not raw:
                continue
            results.extend(parse_full(raw, shape))
    finally:
        ns["restore_service"]()

    with open(os.path.join(OUT_DIR, "ncu_mmvq_recon2.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    def g(r, k):
        v = r.get(k)
        return ("%.2f" % v) if isinstance(v, float) else ("?" if v is None else str(v))

    L = ["# NCU recon2 (sections) GEMV T=4 — .so prod 8035caf4", "",
         "date : " + time.strftime("%Y-%m-%d %H:%M UTC"),
         "service restauré : %s" % ("OUI" if ns["_restored"] else "NON"), "",
         "1 ligne par shape (1er lance). Cible : LECTURE DES POIDS, T=4.", "",
         "| shape | dur µs | BW GB/s | DRAM % | SM % | occ theo | occ ach | lim R/S/W | regs | smem KB | WCPI | NoElig % | waves |",
         "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    seen = set()
    for r in results:
        if r["shape"] in seen:
            continue
        seen.add(r["shape"])
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s/%s/%s | %s | %s | %s | %s | %s |" % (
            r["shape"], g(r, "dur_us"), g(r, "bw_gbs"), g(r, "dram_pct"), g(r, "sm_pct"),
            g(r, "theo_occ"), g(r, "ach_occ"),
            g(r, "lim_regs"), g(r, "lim_smem"), g(r, "lim_warps"),
            g(r, "regs"), g(r, "smem_static_kb"), g(r, "wcpi"), g(r, "no_elig"), g(r, "waves")))
    L.append("")
    L.append("Legende : BW = MemWA Gbyte/s ; DRAM/SM = % pic ; lim R/S/W = block limit regs/smem/warps ; "
             "WCPI = Warp Cycles Per Issued Instruction (haut = latence non masquee) ; NoElig = % scheduler sans warp elig.")
    L.append("")
    L.append("Artefacts : %s/ncu_<shape>.full.txt (toutes sections) + ncu_mmvq_recon2.jsonl" % OUT_DIR)
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")

    log("FIN — %d lances sectionnees — %s" % (len(results), OUT_SUMMARY))
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
