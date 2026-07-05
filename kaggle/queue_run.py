"""Queue manager for Kaggle GPU kernels under the 2-concurrent-session limit.

Launches queued configs whenever a slot frees, downloads each kernel's outputs
(copying probability arrays into oof/) as it finishes, and exits when everything
is done. Designed to run in the background and drive the whole training sweep.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "oof"
OOF.mkdir(exist_ok=True)
USER = "nakuls1nha"
MAX_SLOTS = 2
os.environ.setdefault("KAGGLE_API_TOKEN", "KGAT_3e5b52454fe8f2c89b7dae5ccfa4413b")
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# configs still to launch (order matters), plus slugs already running externally
PENDING = ["mdeberta", "xlmr_large_a", "xlmr_large_b"]
CONFIG_SLUG = {
    "xlmr_base": "clinspan-xlmr-base",
    "mdeberta": "clinspan-mdeberta",
    "xlmr_large_a": "clinspan-xlmr-large-a",
    "xlmr_large_b": "clinspan-xlmr-large-b",
}
ALREADY_ACTIVE = ["clinspan-valmean", "clinspan-xlmr-base"]
DOWNLOAD = {"clinspan-xlmr-base", "clinspan-mdeberta",
            "clinspan-xlmr-large-a", "clinspan-xlmr-large-b"}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def status(slug):
    try:
        r = subprocess.run(["kaggle", "kernels", "status", f"{USER}/{slug}"],
                           capture_output=True, text=True, timeout=60)
        out = (r.stdout + r.stderr).replace("\r", "")
        m = re.search(r'status "?KernelWorkerStatus\.(\w+)"?', out)
        return m.group(1).upper() if m else "UNKNOWN"
    except Exception as e:
        return "UNKNOWN"


def is_terminal(st):
    return st in ("COMPLETE", "ERROR", "CANCELACKNOWLEDGED", "CANCELREQUESTED", "CANCELLED")


def push(config):
    r = subprocess.run([sys.executable, str(ROOT / "kaggle" / "push.py"),
                        str(ROOT / "kaggle" / "configs" / f"{config}.json")],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    if "successfully pushed" in out:
        return True
    log(f"  push {config} failed: {out.strip().splitlines()[-1] if out.strip() else '??'}")
    return False


def download(slug):
    out = ROOT / "kaggle_out" / slug
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", str(out)],
                   capture_output=True, text=True)
    n = 0
    for f in out.glob("oof_*.npy"):
        shutil.copy(f, OOF / f.name); n += 1
    for f in out.glob("test_*.npy"):
        shutil.copy(f, OOF / f.name); n += 1
    cv = list(out.glob("cv_*.json"))
    macro = None
    if cv:
        try:
            macro = json.load(open(cv[0], encoding="utf-8")).get("macro")
        except Exception:
            pass
    log(f"  downloaded {slug}: {n} arrays copied, CV macro={macro}")


def main():
    active = list(ALREADY_ACTIVE)          # slugs that occupy a slot
    done = set()
    pending = list(PENDING)
    while True:
        # refresh statuses
        st = {s: status(s) for s in active}
        # download newly-finished
        for s in list(active):
            if is_terminal(st.get(s, "UNKNOWN")):
                if s in DOWNLOAD and s not in done:
                    download(s)
                done.add(s)
                active.remove(s)
        running = len(active)
        log(f"active={ {s: st.get(s) for s in st} } running={running} pending={pending} done={sorted(done)}")
        # launch to fill slots
        while running < MAX_SLOTS and pending:
            cfg = pending[0]
            log(f"launching {cfg} ...")
            if push(cfg):
                active.append(CONFIG_SLUG[cfg])
                pending.pop(0)
                running += 1
                time.sleep(10)
            else:
                break  # slot not actually free yet; retry next cycle
        if not pending and all(is_terminal(status(s)) for s in active) and not active:
            break
        # done if nothing pending and all download targets fetched
        if not pending and DOWNLOAD.issubset(done):
            break
        time.sleep(90)
    log("QUEUE COMPLETE. done=", sorted(done))


if __name__ == "__main__":
    main()
