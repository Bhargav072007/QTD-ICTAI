"""Rebuild MANIFEST.json (SHA-256 of every tracked file except the manifest itself) and
verify that every .py compiles and every .json parses.  Run from the repository root."""
from __future__ import annotations
import hashlib, json, py_compile, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
files = sorted(f for f in subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split("\n") if f and f != "MANIFEST.json")
bad = []
for f in files:
    p = ROOT / f
    try:
        if f.endswith(".py"):
            py_compile.compile(str(p), doraise=True)
        elif f.endswith(".json"):
            json.loads(p.read_text(encoding="utf-8"))
        if b"\x00" in p.read_bytes() and not f.endswith((".npz", ".pdf")):
            bad.append(f + ": NUL bytes")
    except Exception as exc:  # noqa: BLE001
        bad.append(f"{f}: {exc}")
if bad:
    sys.exit("Refusing to write manifest:\n" + "\n".join(bad))
(ROOT / "MANIFEST.json").write_text(json.dumps({f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}, indent=2, sort_keys=True))
print(f"MANIFEST.json written: {len(files)} files, all .py compile, all .json parse")
