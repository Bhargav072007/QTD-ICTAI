"""Explicit, non-destructive destinations for reproduction scripts."""
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def output_options(names, group):
    parser = argparse.ArgumentParser(description="Write reproduction artifacts separately from historical outputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reproduction" / group)
    parser.add_argument("--force", action="store_true", help="Explicitly replace files in the selected destination")
    args = parser.parse_args()
    for name in names:
        path = args.output_dir / name
        if path.exists() and not args.force:
            parser.error(f"Refusing to overwrite {path}; select a fresh --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args.output_dir
