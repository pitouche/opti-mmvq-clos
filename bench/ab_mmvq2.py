#!/usr/bin/env python3
"""
ab_mmvq2.py — A/B SÛR : lib prod (a4b6f440) vs lib 42d9581 (nwarps T=1 K-quant 4->2).

PROTOCOLE SÛR (leçon de l'incident du 17/10 08:14) :
  * Jamais d'unload par API. La seule méthode = `sudo systemctl stop ollama-v100`
    + attendre VRAM<1GB + process parti, puis mesurer, puis `sudo systemctl start`
    dans un trap EXIT. Zéro cohabitation runner prod / serveur de bench => zéro
    race d'éviction du scheduler => zéro split CPU.
  * 1 seul llama-server à la fois (port 8917), V100 seule.
  * Le .so est injecté via GGML_BACKEND_PATH (plugin isolé) — on ne touche JAMAIS
    au .so prod sur disque.
  * flock : 1 seul bench à la fois.

Ce que ça mesure (3 métriques croisées, prélevées du log llama-server) :
  * prompt_eval t/s  -> préfill (T>1) : CONTRÔLE, ne doit pas bouger
  * eval t/s         -> decode (mix T=1 draft + T=4 verify) : le gain du patch y vit
  * draft mean len + acceptance -> QUALITÉ : ne doit pas bouger (sinon patch suspect)

USAGE (via superviseur, pour survivre à la fin de tir) :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run ab_mmvq2.py --detach
"""
import ctypes
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
OUT_JSONL = os.path.join(D, "ab_mmvq2_results.jsonl")
OUT_SUMMARY = os.path.join(D, "ab_mmvq2_summary.md")
OUT_SERVER_LOG = os.path.join(D, "ab_mmvq2_server.log")
OUT_RUNLOG = os.path.join(D, "ab_mmvq2_run.log")

CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
]
GGML_ENV = {
    "GGML_CUDA_ALLREDUCE": "internal",
    "GGML_CUDA_AR_COPY_THRESHOLD": "131072",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "1",
    "GGML_CUDA_VOLTA_Q5_X4": "1",
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_GQA8_NCOLS2": "2",
    "LLAMA_ARG_FIT_TARGET": "1024",
}
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_42D  = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-42d9581.so"

# (mode, contexte cible, nb gen) — contexte réduit (98k) car on mesure du decode,
# le préfill long n'apporte rien au patch et gonfle la VRAM/temps.
POINTS = [
    ("cold", 16384, 64),
    ("warm", 32768, 64),
    ("warm", 65536, 64),
    ("warm", 98304, 64),
    ("warm", 131072, 64),
    ("warm", 163840, 64),
    ("warm", 196608, 64),
]
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
    # le runner ollama (32GB) est le seul gros process sur la V100
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
    """trap de sécurité : ramène ollama-v100 actif. Idempotent."""
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
    """stoppe le service prod et attend que la V100 soit vide (méthode sûre)."""
    log(f"  · stop {SERVICE}")
    run(["sudo", "systemctl", "stop", SERVICE], timeout=120)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024 and v100_proc_gone():
            log(f"  · V100 libérée ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  · ATTENTION V100 pas totalement libérée (VRAM={vram_mib()} MiB) — on continue")
    return False


def build_env(so_path):
    env = dict(os.environ)
    for d in CUDA_LIBS:
        if os.path.isdir(d) and d not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(GGML_ENV)
    env["GGML_BACKEND_PATH"] = so_path
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_CUDA_IDX)
    return env


def server_args():
    return [
        "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline",
        "-c", str(MAX_CTX), "-np", "1",
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
        "--mmproj", MMPROJ,
        # MTP DESACTIVE volontairement (variance du cycle draft/verify)
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--flash-attn", "auto",
    ]


def port_free(port):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except OSError:
        s.close()
        return False


def http_json(url, data=None, timeout=10):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def start_server(so_path):
    if vram_mib() > 8192:
        raise RuntimeError(f"V100 pas vide (VRAM={vram_mib()} MiB) — refus de démarrer (anti-OOM)")
    env = build_env(so_path)
    lf = open(OUT_SERVER_LOG, "a")
    proc = subprocess.Popen([LLAMA_SERVER] + server_args(), stdout=lf,
                            stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server mort au démarrage (exit {proc.returncode})")
        try:
            http_json(f"http://127.0.0.1:{PORT}/health", timeout=3)
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("llama-server : pas de /health en 600 s")
    time.sleep(3)
    if vram_mib() < 1024:
        raise RuntimeError(f"ABORT : VRAM={vram_mib()} MiB — model en CPU (CUDA non résolu)")
    d = http_json(f"http://127.0.0.1:{PORT}/v1/models", timeout=10)
    _state["model_id"] = d["data"][0]["id"]
    log(f"  · server prêt ({time.time()-t0:.0f}s, VRAM {vram_mib()} MiB)")
    return proc


def stop_server():
    p = _state.get("proc")
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=10)
    time.sleep(8)
    # attendre la VRAM qui redescende (le server direct libère sa 28GB)
    for _ in range(60):
        if vram_mib() < 4096:
            break
        time.sleep(3)
    log(f"  · server stop, VRAM {vram_mib()} MiB")


def prompt_text(ctx_target_tokens):
    n_words = ctx_target_tokens * 4 // 5   # 1.222 tok/mot mesure (run 11:02) -> ~0.98*cible
    words = FILL.split() * max(1, n_words // FILL.count(" ") + 1)
    return " ".join(words[: n_words])


def parse_point(logf, since_pos, n_gen):
    """Extrait DU LOG (source de vérité) les 3 métriques du dernier point."""
    with open(logf, "r", errors="replace") as f:
        data = f.read()[since_pos:]
    out = {}
    m = re.search(r"prompt eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["prompt_tps"] = round(n / (ms / 1000.0), 1) if ms else 0
        out["prompt_tokens"] = n
        out["prompt_ms"] = round(ms, 1)
    # NEGATIVE LOOKBEHIND : ne pas matcher "prompt eval time" (le \b avant eval suffit pas)
    m = re.search(r"(?<!prompt )eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["eval_tps"] = round(n / (ms / 1000.0), 2) if ms else 0
        out["eval_tokens"] = n
        out["eval_ms"] = round(ms, 1)
    m = re.search(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted / \s*(\d+) generated\), mean len =\s+([\d.]+)", data)
    if m:
        out["draft_acc"] = float(m.group(1))
        out["draft_mean_len"] = float(m.group(4))
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
    logf_metrics = parse_point(OUT_SERVER_LOG, since_pos, n_gen)
    return api, logf_metrics


def log_size():
    try:
        return os.path.getsize(OUT_SERVER_LOG)
    except OSError:
        return 0


def leg(name, so_path):
    if not os.path.exists(so_path):
        raise RuntimeError(f".so introuvable : {so_path}")
    log(f"=== LEG {name} : {os.path.basename(so_path)}")
    start_server(so_path)
    rows = []
    try:
        for mode, ctx, ng in POINTS:
            since = log_size()
            log(f"  · point {mode} {ctx} → {ng} gen")
            api, met = run_point(mode, ctx, ng, since)
            row = {"leg": name, "mode": mode, "ctx": ctx, "mtp": "off", **api, **met}
            rows.append(row)
            log(f"    → prefill {met.get('prompt_tps','?')} t/s | decode {met.get('eval_tps','?')} t/s "
                f"| draft acc {met.get('draft_acc','?')} / mean len {met.get('draft_mean_len','?')} "
                f"| {api.get('error','ok')}")
    finally:
        stop_server()
        time.sleep(15)
    return rows


def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    # NB : le superviseur_bench.py tient déjà /tmp/kolbo-supervisor-bench.lock
    # (1 bench à la fois). L'enfant NE doit PAS re-tenter le lock (il se bloquerait
    # tout seul). Si on lance ce bench SANS superviseur, il faut alors garantir
    # soi-même qu'aucun autre bench tourne (flock manuel avant).
    log("AB mmvq2 (protocole SÛR stop/start) — lancé (lock tenu par le superviseur)")

    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")

    state0 = svc_state()
    log(f"état initial service = {state0}, VRAM {vram_mib()} MiB")

    stop_and_free()

    open(OUT_SERVER_LOG, "w").close()
    allrows = []
    for name, so in (("A-prod", SO_PROD), ("B-42d9581", SO_42D)):
        try:
            allrows.extend(leg(name, so))
        except Exception as e:
            log(f"  · LEG {name} ÉCHEC : {e}")
            stop_server()
            continue

    with open(OUT_JSONL, "a") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")

    # synthèse
    lines = ["# AB SÛR : prod vs 42d9581 (nwarps T=1 K-quant 4->2)", "",
             f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
             f"service restauré : {'OUI' if _restored else 'NON'}",
             "",
             "| point | A prefill | B prefill | A decode | B decode | Δdecode | A acc/len | B acc/len |",
             "|---|---|---|---|---|---|---|---|"]
    for mode, ctx, ng in POINTS:
        a = [r for r in allrows if r["leg"] == "A-prod" and r["mode"] == mode and r["ctx"] == ctx]
        b = [r for r in allrows if r["leg"] == "B-42d9581" and r["mode"] == mode and r["ctx"] == ctx]
        if not a or not b:
            lines.append(f"| {mode} {ctx} | — (leg manquant) |")
            continue
        a, b = a[0], b[0]
        da = a.get("eval_tps"); db = b.get("eval_tps")
        delta = f"{(db-da)/da:+.1%}" if (da and db) else "—"
        al = f"{a.get('draft_acc','?')}/{a.get('draft_mean_len','?')}"
        bl = f"{b.get('draft_acc','?')}/{b.get('draft_mean_len','?')}"
        lines.append(f"| {mode} {ctx} | {a.get('prompt_tps','?')} | {b.get('prompt_tps','?')} | "
                     f"{da} | {db} | {delta} | {al} | {bl} |")
    lines.append("")
    lines.append("Interprétation : Δdecode > 0 avec prefill stable et acc/len identiques = patch gagnant et sûr.")
    # restauration d'ABORD, puis le summary (sinon il affiche NON)
    restore_service()
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    sys.exit(0 if _restored else 3)


if __name__ == "__main__":
    main()
