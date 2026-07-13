"""Build an offline, clean-room anonymous artifact from the audited repository."""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, subprocess, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT.parent / "qaa_codex_anonymous_artifact"
TEXT_EXTENSIONS = {".py", ".json", ".md", ".txt", ".ini", ".lock", ".rst", ""}
IDENTITY_PATTERNS = [r"<REDACTED>", r"<REDACTED>", r"<REDACTED>", r"<REDACTED>", r"<REDACTED>", r"<REDACTED>", r"[A-Za-z]:\\\\[^\r\n\"']+"]

def scrub_text(text: str) -> str:
    text = re.sub(r"[A-Za-z]:\\\\[^\r\n\"']+", "<REPOSITORY_ROOT>", text)
    for term in ("<REDACTED>", "<REDACTED>", "<REDACTED>", "<REDACTED>", "<REDACTED>", "<REDACTED>"):
        text = re.sub(re.escape(term), "<REDACTED>", text, flags=re.IGNORECASE)
    return text

def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_EXTENSIONS:
        destination.write_text(scrub_text(source.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    else:
        shutil.copy2(source, destination)

def copy_tree(relative: str, destination: Path) -> None:
    for source in (ROOT / relative).rglob("*"):
        if source.is_dir() or "__pycache__" in source.parts or source.name.startswith("pytest-cache-files-"):
            continue
        copy_file(source, destination / source.relative_to(ROOT))

def write_readme(destination: Path) -> None:
    (destination / "README_ARTIFACT.md").write_text("# Anonymous reproduction artifact\n\nOffline source and persisted JSON evidence.\n\n## Setup\n\n```bash\npython -m venv .venv\n.venv\\Scripts\\python -m pip install -r requirements.lock\n```\n\n## Reproduce\n\n```bash\npython run_mc_no_replacement_analysis.py\npython run_split_experiment.py --shots 1024 --output-name split_results_1024.json\npython run_mechanism_controls.py\npython run_qaoa_multiseed.py --seed 42\npython run_qaoa_multiseed.py --assemble\npython run_traceability_checks.py\n```\n\nRepeat QAOA per seed before assembly.\n", encoding="utf-8")

def scan(destination: Path) -> list[str]:
    hits = []
    for path in destination.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in IDENTITY_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                hits.append(f"{path.relative_to(destination)}: {pattern}")
    return hits

def manifest(destination: Path) -> int:
    rows = {str(path.relative_to(destination)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(destination.rglob("*")) if path.is_file() and ".git" not in path.parts and path.name != "MANIFEST.json"}
    (destination / "MANIFEST.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return len(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Export anonymous QTD artifact without uploading")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    destination = parser.parse_args().destination.resolve()
    if destination.exists(): raise FileExistsError(f"Refusing to overwrite existing export: {destination}")
    destination.mkdir(parents=True)
    for folder in ("phase1", "phase2_qaoa", "phase3_quantum_tree", "analysis"): copy_tree(folder, destination)
    for source in ROOT.glob("run_*.py"): copy_file(source, destination / source.name)
    for source in (ROOT / "outputs").rglob("*.json"): copy_file(source, destination / source.relative_to(ROOT))
    for name in ("requirements.txt", "requirements.lock", "pytest.ini", "LICENSE"): copy_file(ROOT / name, destination / name)
    copy_file(Path(__file__), destination / "scripts" / "export_anonymous_artifact.py")
    write_readme(destination)
    hits = scan(destination)
    if hits: raise RuntimeError("Anonymity scan failed:\n" + "\n".join(hits))
    count = manifest(destination)
    for command in (("git", "init", "-q"), ("git", "config", "user.name", "artifact"), ("git", "config", "user.email", "artifact@invalid"), ("git", "add", "."), ("git", "commit", "-q", "-m", "artifact")): subprocess.run(command, cwd=destination, check=True)
    zip_path = destination.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in destination.rglob("*"):
            if path.is_file() and ".git" not in path.parts: archive.write(path, path.relative_to(destination))
    print(json.dumps({"destination": str(destination), "zip": str(zip_path), "anonymity_hits": 0, "manifest_file_count": count}, indent=2))

if __name__ == "__main__": main()
