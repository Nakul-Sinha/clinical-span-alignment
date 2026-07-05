"""Inject a CONFIG into train_mc.py and push it as a Kaggle GPU kernel.

Usage: python kaggle/push.py <config_json_file>
The config file must contain the full CONFIG dict plus a "slug" key for the
kernel id. Requires KAGGLE_API_TOKEN in the environment.
"""
import json
import os
import pprint
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "train_mc.py"
PUSH_ROOT = HERE / "_push"
USER = "nakuls1nha"
DATASET = "nakuls1nha/clinspan-data"


def build(cfg: dict):
    slug = cfg.pop("slug")
    machine = cfg.pop("machine_shape", "NvidiaTeslaT4")
    template = cfg.pop("template", "train_mc.py")
    global TEMPLATE, CODE_FILE
    TEMPLATE = HERE / template
    CODE_FILE = template
    src = TEMPLATE.read_text(encoding="utf-8")
    # pprint emits valid Python literals (True/False/None), unlike json.dumps
    block = "# === CONFIG_JSON_START ===\nCONFIG = " + pprint.pformat(cfg, sort_dicts=False, width=100) + "\n# === CONFIG_JSON_END ==="
    src = re.sub(r"# === CONFIG_JSON_START ===.*?# === CONFIG_JSON_END ===",
                 block, src, flags=re.DOTALL)
    return slug, machine, src


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    slug, machine, src = build(cfg)
    pdir = PUSH_ROOT / slug
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / CODE_FILE).write_text(src, encoding="utf-8")
    meta = {
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": machine,
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (pdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("pushing", slug, "...")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(pdir)],
                       capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr)


if __name__ == "__main__":
    main()
