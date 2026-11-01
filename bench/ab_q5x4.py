#!/usr/bin/env python3
"""
ab_q5x4.py — A/B APPLICATIF Q5_K T=4 x4 (lever GGML_CUDA_VOLTA_Q5_X4) en MTP ON (config prod), 23/10.

Cible : mul_mat_vec_q<Q8_0, ncols=4> (verify MTP T=4) — 15.8% du decode prod.
Microbench mb_mmvq (3 rounds alternés) :
  q8_f17408x5120 : prod 100.4us → rpb4 82.0us (-18.3%)
  q8_f5120x17408 : prod 101.4us → rpb4 95.2us (-6.1%)
NCU : Q8_0 T4 est L1-bound (L1/TEX 90%, DRAM 12.5%) ; rpb4 (grid 2560→1280)
      L1/TEX 90→72.8%, SM 55.5→43.9%, durée 107.0→99.8us.

Candidat : libggml-cuda-q8x4.so (8425c68b) — 1 seul fichier recompilé (mmvq.cu),
relank des 150 objets prod. Env : GGML_CUDA_VOLTA_Q8_RPB4 (defaut ON, =1 ici),
GGML_CUDA_VOLTA_Q8_X4 (opt-in, testé en leg C). Arithmetique identique → bit-identique.

3 legs (interleave, serveur frais par leg), MTP ON = config prod exacte :
  A-prod    : .so prod fa3e75c2 (rpb=2 générique)
  B-rpb4    : .so q8x4 + GGML_CUDA_VOLTA_Q8_RPB4=1
  C-rpb4x4  : .so q8x4 + RPB4=1 + GGML_CUDA_VOLTA_Q8_X4=1
Point : ctx 36312 (exactement le run9b CUPTI — là où la décomposition a ciblé Q8),
64 gen. Mesure : decode t/s + acceptance MTP (mean len, acc rate/pos) — l'acceptance
doit rester identique (bit-identique). 36k : pas de drift de contexte long.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run ab_q8rpb4.py --detach
"""
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "night_bench_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]

# MTP ON : config prod actuelle (-c 128000) + flags spec
src = src.replace("MAX_CTX = 196672", "MAX_CTX = 128000")
src = src.replace(
    '        # MTP OFF : pas de --spec-type (consigne 18/10 : MTP exclu)\n',
    '        # MTP ON : config prod exacte (23/10 — bench Q8 rpb4)\n'
    '        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-backend-sampling",\n'
)
exec(src, ns)

SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
OUT_JSONL = os.path.join(D, "ab_q5x4_results.jsonl")
OUT_SUMMARY = os.path.join(D, "ab_q5x4_summary.md")
OUT_SERVER_LOG = os.path.join(D, "ab_q5x4_server.log")
OUT_GPUTS = os.path.join(D, "ab_q5x4_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
PORT = ns["PORT"]
N_GEN = ns["N_GEN"]

# env prod complet (unit ollama-v100) + le levier du bench
ENV_BASE = {
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "LLAMA_ARG_FIT_TARGET": "1024",
    "GGML_CUDA_QWEN35_MTP_SHORTLIST":
        "/home/guile/llama-experiments/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32",
    "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
}

CTX = 36312   # = run9b CUPTI (cible Q8_0 T4)

# Q5_K T=4 : x4 (vec_dot_q5_K_q8_1_x4) est ON par DEFaut en prod (Volta, cc==7.0).
# Legs : A = prod tel quel (x4 ON), B = x4 OFF (opt-out). Meme .so prod.
EXP = [
    ("A-x4on",  SO_PROD, dict(ENV_BASE)),
    ("B-x4off", SO_PROD, dict(ENV_BASE, GGML_CUDA_VOLTA_Q5_X4="0")),
]


def parse_mtp(logf, since_pos):
    """acceptance / mean len / acc rate/pos depuis le log (dernières lignes du point)."""
    try:
        with open(logf, "r", errors="replace") as f:
            data = f.read()[since_pos:]
    except OSError:
        return {}
    out = {}
    m = re.findall(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)", data)
    if m:
        out["mtp_acceptance"] = float(m[-1][0])
        out["mtp_mean_len"] = float(m[-1][3])
    m = re.findall(r"#mean acc len = ([\d.]+), #acc rate/pos = \(([\d.]+), ([\d.]+), ([\d.]+)\)", data)
    if m:
        out["acc_mean_len"] = float(m[-1][0])
        out["acc_rate_pos"] = (float(m[-1][1]), float(m[-1][2]), float(m[-1][3]))
    return out


orig_parse = ns["parse_point"]
def parse_point2(logf, since_pos):
    out = orig_parse(logf, since_pos)
    out.update(parse_mtp(OUT_SERVER_LOG, since_pos))
    return out
ns["parse_point"] = parse_point2


def so_prefix(p):
    try:
        return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    except OSError:
        return "?"


def main():
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    log(f"BENCH Q5 X4 (MTP ON, config prod) — reps={reps} — ctx={CTX}")
    if not ns["port_free"](PORT):
        sys.exit(f"port {PORT} occupé — abandon")
    if so_prefix(SO_PROD) != "fa3e75c2":
        sys.exit(f"ABORT : .so prod {so_prefix(SO_PROD)} != fa3e75c2 — prod dérivé, ne pas bench")
    log(f"  · SO PROD md5 = {so_prefix(SO_PROD)} (2 legs, x4 on/off)")

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()

    sampler = subprocess.Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    allrows = []
    try:
        for rep in range(1, reps + 1):
            legs = EXP if rep % 2 == 1 else list(reversed(EXP))
            for name, so, extra_env in legs:
                try:
                    log(f"=== Q5rep{rep} :: LEG {name} (so={so_prefix(so)})")
                    ns["start_server"](so, 2048, None, extra_env)
                    wu = ns["warmup"](ns["log_size"]())
                    since = ns["log_size"]()
                    log(f"  · point {CTX} → {N_GEN} gen (warmup={wu})")
                    api, met = ns["run_point"](CTX, N_GEN, since)
                    row = {"leg": name, "rep": rep, "ctx": CTX, "mtp": "on",
                           "so": so_prefix(so), **api, **met}
                    allrows.append(row)
                    log(f"    → decode {met.get('eval_tps','?')} t/s | "
                        f"acc {met.get('mtp_acceptance','?')} mean_len {met.get('mtp_mean_len','?')} | "
                        f"rate/pos {met.get('acc_rate_pos','?')} | {api.get('error','ok')}")
                except Exception as e:
                    log(f"  · {name} rep{rep} ECHEC : {e}")
                finally:
                    ns["stop_server"]()
                    time.sleep(12)
    finally:
        try:
            sampler.terminate(); sampler.wait(timeout=10)
        except Exception:
            pass
        ns["restore_service"]()

    with open(OUT_JSONL, "a") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")

    med = ns["med"]
    L = ["# A/B applicatif Q5_K T=4 x4 (GGML_CUDA_VOLTA_Q5_X4) — MTP ON config prod — 23/10", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"reps : {reps} · service restauré : {'OUI' if ns['_restored'] else 'NON'}",
         f".so prod {so_prefix(SO_PROD)} (A=x4 ON defaut, B=x4 OFF)",
         f"ctx {CTX} (= run9b CUPTI) · -b 2048 · MTP ON (spec-draft-n-max 3) · 64 gen",
         "",
         "| leg | rep | decode t/s | Δ vs A | acc | mean len | acc rate/pos |",
         "|---|---|---|---|---|---|---|"]
    for rep in range(1, reps + 1):
        ref = next((r.get("eval_tps") for r in allrows
                    if r["leg"] == "A-prod" and r.get("rep") == rep), None)
        for name, _, _ in EXP:
            r = next((r for r in allrows if r["leg"] == name and r.get("rep") == rep), None)
            if not r:
                continue
            tps = r.get("eval_tps")
            d = f"{(tps-ref)/ref*100:+.1f}%" if (tps and ref) else "?"
            rp = r.get("acc_rate_pos")
            rps = f"({rp[0]:.3f}, {rp[1]:.3f}, {rp[2]:.3f})" if rp else "?"
            L.append(f"| {name} | {rep} | {tps} | {d} | {r.get('mtp_acceptance','?')} | "
                     f"{r.get('mtp_mean_len','?')} | {rps} |")
    L += ["", "## médianes (toutes reps)"]
    for name, _, _ in EXP:
        v = [r["eval_tps"] for r in allrows if r["leg"] == name and r.get("eval_tps")]
        L.append(f"  - {name}: decode {med(v)} t/s")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
