"""Bring the results of a Kaggle/Colab run into results/, then re-render the paper tables and numbers.

    python -m scripts.import_results ~/Downloads/mhrag_results.zip     # the notebook's output zip
    python -m scripts.import_results path/to/mhrag_results             # or an unzipped folder
    python -m scripts.import_results ... --dry-run                     # only list what would change

Only experiments the run finished (a marker in _checkpoints/) are imported, so a partial or stale copy never replaces
newer local results. latency.json is merged per configuration: new measurements are added and a pending entry never
replaces a measured one. Checkpoints, partial files and the notebook's copies of paper files are skipped; the tables
and numbers are regenerated here instead.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from mhrag.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
# notebook step -> the files and folders it produces (relative to the results folder)
STEP_OUTPUTS = {
    "classifier": ["risk_classifier.json", "safety_gate_v2_devset.json", "safety_gate_v2_heldout.json"],
    "retrieval_main": ["retrieval_main.json"],
    "retrieval_chunks": ["retrieval_chunks.json"],
    "generation_local": ["generation_local.json", "generation/local"],
    "generation_gpu": ["generation_gpu.json", "generation/gpu"],
}
EXPERIMENTS = ["retrieval_main", "retrieval_chunks", "generation_local", "generation_gpu"]


def find_root(src: Path) -> Path:
    """The folder that holds the results files (the zip's root, or a nested mhrag_results/)."""
    for cand in [src, src / "mhrag_results", *src.glob("*/mhrag_results")]:
        if (cand / "_checkpoints").is_dir() or list(cand.glob("*.json")):
            return cand
    raise SystemExit(f"no results found in {src}")


def _measured(entry) -> bool:
    return isinstance(entry, dict) and "summary" in entry


def merge_latency(src: Path, dst: Path, dry: bool) -> list[str]:
    if not src.exists():
        return []
    new = json.loads(src.read_text())
    cur = json.loads(dst.read_text()) if dst.exists() else {}
    changed = [k for k, v in new.items() if _measured(v) and v != cur.get(k)]
    if changed and not dry:
        cur.update({k: new[k] for k in changed})
        dst.write_text(json.dumps(cur, indent=2))
    return [f"latency.json: {k}" for k in changed]


def copy_path(src: Path, dst: Path, dry: bool) -> list[str]:
    """Copy a file or folder; returns the files that are new or different."""
    files = [src] if src.is_file() else [p for p in src.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    changed = []
    for f in files:
        target = dst if src.is_file() else dst / f.relative_to(src)
        if target.exists() and filecmp.cmp(f, target, shallow=False):
            continue
        changed.append(str(target.relative_to(RESULTS)))
        if not dry:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
    return changed


def copy_path_differs(root: Path, outputs: list[str]) -> bool:
    return any(copy_path(root / o, RESULTS / o, dry=True) for o in outputs)


def import_results(src: Path, dry: bool = False) -> dict:
    root = find_root(src)
    done = {p.stem for p in (root / "_checkpoints").glob("*.json")}
    report = {"finished_steps": sorted(done), "imported": [], "not_finished": []}
    for step, outputs in STEP_OUTPUTS.items():
        present = [o for o in outputs if (root / o).exists()]
        if not present:
            continue
        if step not in done:  # switched off or unfinished: report it only if its files differ from ours
            if copy_path_differs(root, present):
                report["not_finished"].append(step)
            continue
        for o in present:
            report["imported"] += copy_path(root / o, RESULTS / o, dry)
    report["imported"] += merge_latency(root / "latency.json", RESULTS / "latency.json", dry)
    for log in (root / "logs").glob("kaggle_*.log"):
        report["imported"] += copy_path(log, RESULTS / "logs" / log.name, dry)
    return report


def render(py: str = sys.executable) -> None:
    for exp in EXPERIMENTS:
        if (RESULTS / f"{exp}.json").exists():
            subprocess.run([py, "-m", "scripts.run_eval", "--tables-only", "--config",
                            f"configs/experiments/{exp}.yaml"], cwd=PROJECT_ROOT, check=True)
    subprocess.run([py, "-m", "scripts.paper_numbers"], cwd=PROJECT_ROOT, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="mhrag_results.zip or an unzipped mhrag_results folder")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        src = args.source.expanduser()
        if src.suffix == ".zip":
            zipfile.ZipFile(src).extractall(tmp)
            src = Path(tmp)
        report = import_results(src, args.dry_run)
    print("finished on Kaggle:", ", ".join(report["finished_steps"]) or "none")
    if report["not_finished"]:
        print("not finished, so left out (the next session continues them):", ", ".join(report["not_finished"]))
    print(f"{'would import' if args.dry_run else 'imported'} {len(report['imported'])} new or changed file(s):")
    for f in report["imported"]:
        print("  ", f)
    if report["imported"] and not args.dry_run:
        render()
        print("\nTables and paper/numbers.tex regenerated. Review with `git status` / `git diff`, then commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
