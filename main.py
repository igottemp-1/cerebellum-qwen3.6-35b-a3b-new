import os
import sys
import time
import stat
import shutil
import subprocess
import requests
import kagglehub
from pathlib import Path
from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────────────────────
PORT            = 4041
MODEL_DIR       = Path("/persistent-storage/models")
BINARY_DIR      = Path("/tmp/beellama")
SERVER_BIN      = BINARY_DIR / "llama-server"

MAIN_MODEL_REPO = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
MAIN_MODEL_FILE = "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"

DFLASH_REPO     = "Anbeeld/Qwen3.6-35B-A3B-DFlash-GGUF"
DFLASH_FILE     = "qwen36-35b-a3b-dflash-Q6_K.gguf"

HF_TOKEN        = os.environ["HF_TOKEN"]
API_TOKEN       = os.environ["API_TOKEN"]
CF_TOKEN        = os.environ["CLOUDFLARE_TUNNEL_TOKEN"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

def ensure_model(repo, filename, dest_dir):
    dest = dest_dir / filename
    if dest.exists():
        log(f"Cached: {filename}")
        return dest
    log(f"Downloading {filename} from {repo} ...")
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(dest_dir),
        token=HF_TOKEN,
    )
    log(f"Done: {filename}")
    return dest

# ── Step 1: Pull BeeLlama binaries from Kaggle ────────────────────────────────
log("Pulling BeeLlama binaries from Kaggle...")
BINARY_DIR.mkdir(parents=True, exist_ok=True)
kaggle_path = Path(kagglehub.dataset_download("igottempmail/beellama-l4-sw89"))
for f in kaggle_path.iterdir():
    shutil.copy2(f, BINARY_DIR / f.name)
# make all files executable that need it
SERVER_BIN.chmod(SERVER_BIN.stat().st_mode | stat.S_IEXEC)
log("BeeLlama binaries ready")

# ── Step 2: Ensure models cached in persistent storage ───────────────────────
MODEL_DIR.mkdir(parents=True, exist_ok=True)
main_model  = ensure_model(MAIN_MODEL_REPO, MAIN_MODEL_FILE, MODEL_DIR)
dflash_model = ensure_model(DFLASH_REPO,    DFLASH_FILE,     MODEL_DIR)

# ── Step 3: Install cloudflared ───────────────────────────────────────────────
cf_bin = Path("/usr/local/bin/cloudflared")
if not cf_bin.exists():
    log("Installing cloudflared...")
    arch_url = (
        "https://github.com/cloudflare/cloudflared/releases/latest"
        "/download/cloudflared-linux-amd64"
    )
    subprocess.run(
        ["curl", "-fsSL", arch_url, "-o", str(cf_bin)],
        check=True
    )
    cf_bin.chmod(cf_bin.stat().st_mode | stat.S_IEXEC)
    log("cloudflared installed")
else:
    log("cloudflared already present")

# ── Step 4: Start Cloudflare tunnel ──────────────────────────────────────────
log("Starting Cloudflare tunnel...")
cf_proc = subprocess.Popen(
    [
        str(cf_bin),
        "tunnel", "--no-autoupdate", "run",
        "--token", CF_TOKEN,
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
log(f"Cloudflare tunnel PID: {cf_proc.pid}")

# ── Step 5: Launch llama-server ───────────────────────────────────────────────
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = f"{BINARY_DIR}:{env.get('LD_LIBRARY_PATH', '')}"

cmd = [
    str(SERVER_BIN),
    "--model",              str(main_model),
    "--host",               "0.0.0.0",
    "--port",               str(PORT),
    "--alias",              "qwen3.6-35b-a3b",
    "--api-key",            API_TOKEN,

    # GPU
    "-ngl",                 "999",

    # Context
    "--ctx-size",           "128000",
    "--parallel",           "1",
    "--kv-unified",

    # KV cache
    "--cache-type-k",       "turbo3_tcq",
    "--cache-type-v",       "turbo3_tcq",
    "--cache-ram",          "4096",

    # Batch
    "--batch-size",         "2048",
    "--ubatch-size",        "512",

    # Flash attn
    "--flash-attn",

    # Sampling
    "--temp",               "0.3",
    "--top-k",              "20",
    "--top-p",              "0.95",
    "--min-p",              "0.0",
    "--repeat-penalty",     "1.0",

    # Reasoning
    "--reasoning",          "on",
    "--jinja",
    "--chat-template-kwargs", '{"preserve_thinking":true}',

    # DFlash speculative decoding
    "--spec-type",          "dflash",
    "--spec-draft-model",   str(dflash_model),
    "--spec-draft-ngl",     "all",
    "--spec-dflash-cross-ctx", "1024",
    "--spec-draft-n-max",   "16",
    "--spec-branch-budget", "2",
    "--spec-draft-temp",    "0",
    "--spec-dm-adaptive",

    # MTP on top
    "--spec-type",          "mtp",
    "--spec-draft-n-max",   "2",

    # Logging
    "--log-timestamps",
    "--log-prefix",
    "-v",                   "0",
]

log(f"Starting llama-server on port {PORT}...")
server_proc = subprocess.Popen(cmd, env=env)
log(f"llama-server PID: {server_proc.pid}")

# ── Step 6: Wait for server to be healthy ─────────────────────────────────────
log("Waiting for llama-server /health ...")
for attempt in range(120):
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
        if r.status_code == 200:
            log("llama-server healthy, ready for traffic")
            break
    except Exception:
        pass
    time.sleep(2)
else:
    log("ERROR: llama-server did not become healthy in time, exiting")
    server_proc.terminate()
    cf_proc.terminate()
    sys.exit(1)

# ── Step 7: Park — keep alive, crash if server dies ──────────────────────────
log("Parking. Server is live.")
exit_code = server_proc.wait()
log(f"llama-server exited with code {exit_code}, shutting down")
cf_proc.terminate()
sys.exit(exit_code)
