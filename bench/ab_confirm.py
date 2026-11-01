#!/usr/bin/env python3
"""
ab_confirm.py — A/B/C de LEVIERS ENV (pas de rebuild, meme .so prod, MTP OFF).

Contexte : le patch 42d9581 est rejete (MTP off -> pas de gain decode, legerement
arriere). La vraie cible restante = les env VOLTA ABSENTS de la prod mais presents
dans le .so (activables par env, zero rebuild, zero touch au .so) :
  - GGML_CUDA_VOLTA_Q5_X4     -> Q5_K (efficacité 47% du pic = la plus grosse marge)
  - GGML_CUDA_VOLTA_Q8_FATTN_TC -> Q8 flash-attn tensor cores
MTP volontairement DESACTIVE (le cycle draft/verify bruitait les mesures).

Legs (meme .so prod a4b6f440, MTP off, KV q8, contexte identique) :
  A-base          : prod = Q6_W4R4=1 seule
  B-q5x4          : A + Q5_X4=1
  C-q5x4+q8fa     : B + Q8_FATTN_TC=1
Delta A->B isole Q5_X4 ; B->C ajoute Q8_FATTN_TC.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run ab_confirm.py --detach
"""
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

D = os.path.dirname(os.path.abspath(__file__))

LLAMA_SERVER = "/home/guile/llama-experiments/ollama-v100/lib/ollama/llama-server"
GGUF = ("/usr/share/ollama/.ollama/models/blobs/"
        "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
MMPROJ = ("/usr/share/ollama/.ollama/models/blobs/"
          "sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e")

PORT = 8917
SERVICE = "ollama-v100"
OUT_JSONL = os.path.join(D, "ab_confirm_results.jsonl")
OUT_SUMMARY = os.path.join(D, "ab_confirm_summary.md")
OUT_SERVER_LOG = os.path.join(D, "ab_confirm_server.log")

CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
]
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"

# base = la prod (hors MTP) : Q6_W4R4 seule. KV q8 via --cache-type.
ENV_BASE = {"GGML_CUDA_VOLTA_Q6_W4R4": "1", "LLAMA_ARG_FIT_TARGET": "1024"}
# levier : -b/-ub (chunk de prefill). prod V100 tourne a 512.
LEGS = [
    ("A-b512",  dict(ENV_BASE, _B="512")),
    ("B-b2048", dict(ENV_BASE, _B="2048")),
]

# ladder 16k -> 3x64k, palier 32k au-delà de 64k (même que l'A/B 42d9581)
POINTS = [
    ("warm", 114688, 64),   # regne reel prod
    ("warm", 196608, 64),
]
REPS = 3
MAX_CTX = 196672
TIMEOUT_S = 900
GPU_SMI_IDX = v100_smi_index()  # résolu par UUID (ne pas hardcoder)
GPU_CUDA_IDX = v100_cuda_ordinal()  # résolu par UUID (ne pas hardcoder)
FILL = ("La tour de contrôle surveille le passage des avions au crépuscule "
        "pendant que le chef de piste organise les départs de la soirée. ")

_state = {}
_restored = False


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def svc_state():
    return run(["systemctl", "is-active", SERVICE]).stdout.strip()


def vram_mib():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1


def v100_proc_gone():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    for line in r.stdout.splitlines():
        if line.strip():
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip().replace("MiB", "").replace(" ", "").isdigit():
                if int(parts[1].strip().split()[0]) > 8000:
                    return False
    return True


def restore_service():
    global _restored
    try:
        if svc_state() != "active":
            log(f"  · restauration : systemctl start {SERVICE}")
            run(["sudo", "systemctl", "start", SERVICE], timeout=120)
        for _ in range(90):
            if svc_state() == "active":
                break
            time.sleep(2)
        _restored = svc_state() == "active"
        log(f"  · service {SERVICE} = {svc_state()}")
    except Exception as e:
        log(f"  · ERREUR restauration : {e}")
        _restored = False


def on_sig(sig, frm):
    restore_service()
    sys.exit(128 + sig)


def stop_and_free():
    log(f"  · stop {SERVICE}")
    run(["sudo", "systemctl", "stop", SERVICE], timeout=120)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024 and v100_proc_gone():
            log(f"  · V100 libérée ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  · ATTENTION V100 pas totalement libérée (VRAM={vram_mib()} MiB)")
    return False


def build_env(extra):
    env = dict(os.environ)
    for d in CUDA_LIBS:
        if os.path.isdir(d) and d not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(extra)
    env["GGML_BACKEND_PATH"] = SO_PROD
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_CUDA_IDX)
    return env


def server_args(batch):
    return [
        "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline",
        "-c", str(MAX_CTX), "-np", "1",
        "-b", batch, "-ub", batch,
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
        "--mmproj", MMPROJ,
        # MTP OFF : pas de --spec-type (c'est ce qui bruitait)
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--flash-attn", "auto",
    ]


def port_free(port):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port)); s.close(); return True
    except OSError:
        s.close(); return False


def http_json(url, data=None, timeout=10):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def start_server(extra):
    batch = extra.get("_B", "512")
    env = {k:v for k,v in extra.items() if k != "_B"}
    if vram_mib() > 8192:
        raise RuntimeError(f"V100 pas vide (VRAM={vram_mib()} MiB) — refus de démarrer")
    env = build_env(env)
    lf = open(OUT_SERVER_LOG, "a")
    proc = subprocess.Popen([LLAMA_SERVER] + server_args(batch), stdout=lf,
                            stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server mort au démarrage (exit {proc.returncode})")
        try:
            http_json(f"http://127.0.0.1:{PORT}/health", timeout=3); break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("llama-server : pas de /health en 600 s")
    time.sleep(3)
    if vram_mib() < 1024:
        raise RuntimeError(f"ABORT : VRAM={vram_mib()} MiB — model en CPU")
    d = http_json(f"http://127.0.0.1:{PORT}/v1/models", timeout=10)
    _state["model_id"] = d["data"][0]["id"]
    log(f"  · server prêt ({time.time()-t0:.0f}s, VRAM {vram_mib()} MiB) -b {batch}")
    return proc


def stop_server():
    p = _state.get("proc")
    if p and p.poll() is None:
        p.terminate()
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(timeout=10)
    time.sleep(8)
    for _ in range(60):
        if vram_mib() < 4096: break
        time.sleep(3)
    log(f"  · server stop, VRAM {vram_mib()} MiB")


def prompt_text(ctx_target_tokens):
    n_words = ctx_target_tokens * 4 // 5   # 1.222 tok/mot mesure -> ~0.98*cible
    words = FILL.split() * max(1, n_words // FILL.count(" ") + 1)
    return " ".join(words[: n_words])


def parse_point(logf, since_pos, n_gen):
    with open(logf, "r", errors="replace") as f:
        data = f.read()[since_pos:]
    out = {}
    m = re.search(r"prompt eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["prompt_tps"] = round(n / (ms / 1000.0), 1) if ms else 0
        out["prompt_tokens"] = n
    m = re.search(r"(?<!prompt )eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["eval_tps"] = round(n / (ms / 1000.0), 2) if ms else 0
        out["eval_tokens"] = n
        out["eval_ms"] = round(ms, 1)
    return out


def run_point(mode, ctx_target, n_gen, since_pos):
    text = prompt_text(ctx_target)
    data = {
        "model": _state["model_id"], "stream": False,
        "messages": [{"role": "user", "content": text + "\nRéponds en une phrase."}],
        "max_tokens": n_gen, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
    }
    t0 = time.time()
    try:
        r = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", data, timeout=TIMEOUT_S)
        u = (r or {}).get("usage", {})
        api = {"prompt_tokens_api": u.get("prompt_tokens"),
               "gen_tokens_api": u.get("completion_tokens"),
               "wall_s": round(time.time() - t0, 1)}
    except Exception as e:
        api = {"error": str(e)}
    return api, parse_point(OUT_SERVER_LOG, since_pos, n_gen)


def log_size():
    try: return os.path.getsize(OUT_SERVER_LOG)
    except OSError: return 0


def leg(name, extra):
    log(f"=== LEG {name} : {list(extra.keys())}")
    start_server(extra)
    rows = []
    try:
        for mode, ctx, ng in POINTS:
            since = log_size()
            log(f"  · point {mode} {ctx} → {ng} gen")
            api, met = run_point(mode, ctx, ng, since)
            row = {"leg": name, "mode": mode, "ctx": ctx, "mtp": "off", **api, **met}
            rows.append(row)
            log(f"    → prefill {met.get('prompt_tps','?')} t/s | decode {met.get('eval_tps','?')} t/s | {api.get('error','ok')}")
    finally:
        stop_server()
        time.sleep(15)
    return rows


def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    log("AB CONFIRMATION (b512 vs b2048, decode) — lancé (lock tenu par le superviseur)")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")
    stop_and_free()
    open(OUT_SERVER_LOG, "w").close()
    allrows = []
    # CONFIRMATION : 3 reps par contexte, ordre ALTERNÉ (A,B / B,A / A,B)
    # pour eliminer tout biais d'ordre/thermique systematique.
    for mode, ctx, ng in POINTS:
        for rep in range(1, REPS+1):
            order = [l for l in LEGS] if rep % 2 == 1 else list(reversed([l for l in LEGS]))
            for name, extra in order:
                try:
                    log(f"=== {mode} {ctx} rep{rep} :: LEG {name} {list(extra.keys())}")
                    start_server(extra)
                    since = log_size()
                    api, met = run_point(mode, ctx, ng, since)
                    row = {"leg": name, "mode": mode, "ctx": ctx, "rep": rep, "mtp": "off", **api, **met}
                    allrows.append(row)
                    log(f"    → prefill {met.get('prompt_tps','?')} t/s | decode {met.get('eval_tps','?')} t/s | {api.get('error','ok')}")
                except Exception as e:
                    log(f"  · {mode} {ctx} {name} rep{rep} ÉCHEC : {e}")
                finally:
                    stop_server()
                    time.sleep(12)
    with open(OUT_JSONL, "a") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")

    lines = ["# AB BATCH PREFILL (-b/-ub 512/1024/2048/4096) — MTP OFF, même .so prod", "",
             f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
             f"service restauré : {'OUI' if _restored else 'NON'}",
             "",
             "## PREFILL (prompt t/s) — la cible du levier -b",
             "| point | A b512 | B b1024 | C b2048 | D b4096 |",
             "|---|---|---|---|---|",
             ]
    idx = {n: {} for n, _ in LEGS}
    for r in allrows:
        idx[r["leg"]].setdefault((r["mode"], r["ctx"]), r)
    def med(vals):
        vals = sorted(v for v in vals if v)
        return vals[len(vals)//2] if vals else None
    lines.append("| contexte | batch | decode rep1/2/3 | médiane |")
    lines.append("|---|---|---|---|")
    for mode, ctx, ng in POINTS:
        for n in names:
            rs = sorted([r for r in allrows if r["ctx"]==ctx and r["leg"]==n], key=lambda r: r.get("rep",0))
            vals = [r.get("eval_tps") for r in rs]
            lines.append(f"| {ctx} | {n} | {' / '.join(str(v) for v in vals)} | {med(vals)} |")
    lines.append("")
    lines.append("## Δ decode par rep (B-b2048 vs A-b512) — le test de confirmation")
    lines.append("| contexte | rep | A b512 | B b2048 | Δ |")
    lines.append("|---|---|---|---|---|")
    for mode, ctx, ng in POINTS:
        for rep in range(1, REPS+1):
            a = next((r for r in allrows if r["ctx"]==ctx and r["leg"]=="A-b512" and r.get("rep")==rep), None)
            b = next((r for r in allrows if r["ctx"]==ctx and r["leg"]=="B-b2048" and r.get("rep")==rep), None)
            if a and b and a.get("eval_tps") and b.get("eval_tps"):
                d = (b["eval_tps"]-a["eval_tps"])/a["eval_tps"]*100
                lines.append(f"| {ctx} | {rep} | {a['eval_tps']} | {b['eval_tps']} | {d:+.1f}% |")
    lines.append("")
    lines.append("Verdict : B> A sur les 3 reps avec Δ>10% = gain decode CONFIRMÉ.")
    lines.append("")
    lines.append("## PREFILL (t/s) — rappel, 3 reps")
    lines.append("| contexte | A b512 | B b2048 |")
    lines.append("|---|---|---|")
    for mode, ctx, ng in POINTS:
        a = [r.get("prompt_tps") for r in allrows if r["ctx"]==ctx and r["leg"]=="A-b512"]
        b = [r.get("prompt_tps") for r in allrows if r["ctx"]==ctx and r["leg"]=="B-b2048"]
        lines.append(f"| {ctx} | {med(a)} | {med(b)} |")
    # restauration d'ABORD, puis le summary (sinon il affiche NON)
    restore_service()
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    sys.exit(0 if _restored else 3)


if __name__ == "__main__":
    main()
