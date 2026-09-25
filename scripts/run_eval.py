"""Run an experiment described by a YAML file and write results/ + paper/tables/.

    python -m scripts.run_eval --config configs/experiments/retrieval_main.yaml
    python -m scripts.run_eval --config configs/experiments/retrieval_chunks.yaml
    python -m scripts.run_eval --config configs/experiments/generation_local.yaml
    python -m scripts.run_eval --config configs/experiments/generation_gpu.yaml     # Colab / GPU
    python -m scripts.run_eval --config configs/experiments/generation_gpu.yaml --models qwen2.5-7b-instruct
    python -m scripts.run_eval --tables-only --config ...                           # re-render LaTeX

Generation runs checkpoint every answer to results/generation/<exp>/<model>__<profile>__<dataset>.jsonl and
resume from it, so an interrupted Colab session can continue.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from mhrag.config import PROJECT_ROOT, get_settings

log = logging.getLogger("run_eval")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--limit", type=int, help="use only the first N queries of each dataset (smoke runs)")
    ap.add_argument("--models", nargs="+", help="generation only: run just these model keys (e.g. one per Kaggle session); "
                                               "results are merged with earlier runs of the same experiment")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "sentence_transformers", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    path = Path(args.config)
    cfg = yaml.safe_load((path if path.is_absolute() else PROJECT_ROOT / path).read_text())
    s = get_settings()
    for k, v in (cfg.get("settings") or {}).items():  # experiment-level overrides, e.g. retrieval.top_k
        section, key = k.split(".", 1)
        setattr(getattr(s, section), key, v)
    out = s.results_dir / f"{cfg['kind']}_{cfg['name']}.json"

    if cfg["kind"] == "retrieval":
        from eval.report import retrieval_tables
        from eval.retrieval_eval import run_retrieval_experiment

        if not args.tables_only:
            if args.limit:
                cfg["limit"] = args.limit
            ckpt = out.with_suffix(".partial.jsonl")  # finished runs; an interrupted experiment resumes from here
            res = run_retrieval_experiment(cfg, s, checkpoint=None if args.limit else ckpt)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res))
            ckpt.unlink(missing_ok=True)
        retrieval_tables(json.loads(out.read_text()))
    elif cfg["kind"] == "generation":
        from eval.generation_eval import run_generation_experiment
        from eval.report import generation_tables

        if args.models:
            unknown = [m for m in args.models if m not in cfg["models"]]
            if unknown:
                raise SystemExit(f"--models not in the experiment config: {unknown}")
            cfg["run_models"] = args.models
        if not args.tables_only:
            res = run_generation_experiment(cfg, s, limit=args.limit)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1))
        generation_tables(json.loads(out.read_text()))
    else:
        raise SystemExit(f"unknown kind {cfg['kind']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
