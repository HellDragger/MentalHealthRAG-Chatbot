"""Generation experiments: every model x profile (no_rag / naive / full) x dataset, with checkpoint/resume."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from eval.datasets import load_jsonl
from eval.judges import (
    JUDGE_VERSION,
    NLIJudge,
    abstained,
    answer_relevance,
    citation_stats,
    strip_citations,
)
from eval.metrics import bootstrap_ci, holm, paired_bootstrap_p, wilcoxon_p
from eval.tables import esc, fmt, write_table
from mhrag.config import Settings
from mhrag.index.builder import build_index
from mhrag.llm.registry import ModelManager, availability, load_catalog
from mhrag.pipeline import RAGPipeline
from mhrag.retrieval.factory import build_retriever
from mhrag.runtime import build_gate

log = logging.getLogger(__name__)
PROFILES = ("no_rag", "naive", "full")


# ------------------------------------------------------------------------------ generation
def _index_for(settings: Settings, cfg: dict, dataset: str, profile: str):
    """FAQ-Gen must not see the FAQ (or KB facts that copy it) -> 'nofaq' index variants."""
    nofaq = dataset == "faq_gen"
    ing = settings.ingest.model_copy(update={"exclude_faq_from_index": nofaq})
    s2 = settings.model_copy(update={"ingest": ing})
    if profile == "naive":
        emb, chunk = cfg["naive"]["embedder"], cfg["naive"]["chunk_tokens"]
    else:
        emb, chunk = settings.index.embedder, settings.chunking.chunk_tokens
    variant = "nofaq" if nofaq else ""
    build_index(s2, emb, chunk, variant)
    return s2, emb, chunk, variant


def generate(cfg: dict, settings: Settings, out_dir: Path, limit: int | None) -> list[dict]:
    catalog = load_catalog()
    retrievers: dict = {}
    gate = build_gate(settings)
    records = []
    for model in cfg["models"]:
        if cfg.get("run_models") and model not in cfg["run_models"]:
            records += _load_checkpoint(cfg, model, out_dir)  # reuse answers generated in earlier sessions
            continue
        ok, why = availability(catalog[model])
        if not ok:
            log.warning("skip %s: %s", model, why)
            records.append({"model": model, "status": f"TODO(run): {why}"})
            continue
        mm = ModelManager(catalog, [model], model, timeout_s=cfg.get("timeout_s", 300))
        load_t = time.perf_counter()
        backend = mm.get(model)
        load_s = time.perf_counter() - load_t
        peak = _Peak()
        for dataset in cfg["datasets"]:
            rows = load_jsonl(f"{dataset}.jsonl")[: cfg.get("limits", {}).get(dataset) or limit]
            for row in rows:
                row.setdefault("query", row.get("text"))
            for profile in cfg["profiles"]:
                path = out_dir / f"{model}__{profile}__{dataset}.jsonl"
                done = {}
                if path.exists():
                    for line in path.read_text().splitlines():
                        r = json.loads(line)
                        done[r["id"]] = r
                s2, emb, chunk, variant = _index_for(settings, cfg, dataset, profile)
                key = (emb, chunk, variant)
                if profile != "no_rag" and key not in retrievers:
                    retrievers[key] = build_retriever(s2, emb, chunk, variant,
                                                      need_reranker=s2.retrieval.mode.endswith("_rerank"))
                pipe = RAGPipeline(s2, retrievers.get(key) if profile != "no_rag" else None, mm,
                                   gate if profile == "full" else None)
                params = pipe.params(temperature=cfg.get("temperature", 0.0))
                with open(path, "a", encoding="utf-8") as f:
                    for row in rows:
                        if row["id"] in done:
                            continue
                        with peak:
                            try:
                                res = pipe.answer(row["query"], model=model, profile=profile, params=params)
                                err = None
                            except Exception as e:  # keep going; record the failure
                                res, err = {}, f"{type(e).__name__}: {e}"
                        rec = {"id": row["id"], "dataset": dataset, "model": model, "profile": profile,
                               "query": row["query"], "answer": res.get("answer", ""),
                               "context": res.get("context", ""), "sources": res.get("sources", []),
                               "gate": res.get("gate"), "generated": res.get("generated"),
                               "timing": res.get("timing", {}), "usage": res.get("usage", {}), "error": err}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        done[row["id"]] = rec
                log.info("%s %s %s: %d answers", model, profile, dataset, len(done))
                records += [{**r, "reference": _ref(row_by_id(rows, r["id"]))} for r in done.values()
                            if row_by_id(rows, r["id"]) is not None]
        records.append({"model": model, "status": "ok", "load_seconds": load_s, "peak_rss_gb": peak.gb,
                        **_gpu_mem()})
        mm._loaded.clear()
        backend.close()
    return records


def _load_checkpoint(cfg: dict, model: str, out_dir: Path) -> list[dict]:
    """Answers already on disk for a model that is not being (re)generated in this session."""
    recs = []
    for dataset in cfg["datasets"]:
        rows = load_jsonl(f"{dataset}.jsonl")
        for profile in cfg["profiles"]:
            path = out_dir / f"{model}__{profile}__{dataset}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                r = json.loads(line)
                row = row_by_id(rows, r["id"])
                if row is not None:
                    recs.append({**r, "reference": _ref(row)})
    return recs


def row_by_id(rows, rid):
    for r in rows:
        if r["id"] == rid:
            return r
    return None


def _ref(row):
    if row is None:
        return None
    return row.get("reference") or row.get("references")


class _Peak:
    def __init__(self):
        import psutil

        self.p = psutil.Process()
        self.gb = self.p.memory_info().rss / 2**30

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.gb = max(self.gb, self.p.memory_info().rss / 2**30)


def _gpu_mem() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {"cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30}
    except Exception:
        pass
    return {}


# ------------------------------------------------------------------------------ scoring
def score(records: list[dict], cfg: dict) -> list[dict]:
    answers = [r for r in records if "answer" in r]
    nli = NLIJudge(cfg.get("nli_model", "cross-encoder/nli-deberta-v3-base"))
    try:
        from rouge_score import rouge_scorer

        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        rouge = None
    import textstat

    for r in answers:
        a = r["answer"] or ""
        refs = r.get("reference")
        refs = [refs] if isinstance(refs, str) else (refs or [])
        passages = [s["snippet"] for s in r.get("sources", [])]
        # use full chunk text when available (context) for NLI: split context blocks
        if r.get("context"):
            passages = [b for b in r["context"].split("\n\n[") if b.strip()] if r["profile"] == "full" else [r["context"]]
        m: dict = {}
        if r.get("generated") and r["profile"] != "no_rag":
            m.update(nli.score(a, passages[:8]))
        m["answer_relevance"] = answer_relevance(r["query"], a) if a else None
        if refs and rouge:
            m["rougeL"] = max(rouge.score(ref, strip_citations(a))["rougeL"].fmeasure for ref in refs)
        m["flesch"] = float(textstat.flesch_reading_ease(strip_citations(a))) if a.strip() else None
        m["words"] = len(a.split())
        m["abstained"] = abstained(a)
        m.update(citation_stats(a, len(r.get("sources", []))))
        r["metrics"] = m
    _bertscore(answers, cfg)
    _llm_judge(answers, cfg)
    return answers


def _bertscore(answers, cfg):
    model = cfg.get("bertscore_model", "roberta-large")
    todo = [r for r in answers if r.get("reference") and r.get("answer")]
    if not todo or cfg.get("bertscore") is False:
        return
    from bert_score import score as bscore

    cands, refs = [], []
    for r in todo:
        ref = r["reference"]
        refs.append([ref] if isinstance(ref, str) else ref)
        cands.append(strip_citations(r["answer"]))
    # multi-reference: bert_score takes list of lists
    _, _, f1 = bscore(cands, refs, model_type=model, lang="en", verbose=False, batch_size=16)
    for r, v in zip(todo, f1.tolist(), strict=True):
        r["metrics"]["bertscore_f1"] = float(v)
        r["metrics"]["bertscore_model"] = model


def _llm_judge(answers, cfg):
    jm = cfg.get("judge_model")
    if not jm:
        return
    catalog = load_catalog()
    ok, why = availability(catalog[jm])
    if not ok:
        log.warning("LLM judge %s unavailable (%s): rubric scores left as TODO(run)", jm, why)
        return
    from eval.judges import LLMJudge

    backend = ModelManager(catalog, [jm], jm).get(jm)
    judge = LLMJudge(backend)
    for r in answers:
        if not r.get("answer"):
            continue
        ref = r.get("reference")
        ref = ref if isinstance(ref, str) else (ref[0] if ref else None)
        rub = judge.rubric(r["query"], r["answer"], ref)
        if rub:
            r["metrics"].update({f"judge_{k}": v for k, v in rub.items()})
        if r["profile"] != "no_rag" and r.get("context"):
            r["metrics"]["judge_faithfulness"] = judge.faithfulness(r["answer"], r["context"])
        r["metrics"]["judge_model"] = jm


# ------------------------------------------------------------------------------ aggregation
METRICS = ["faithfulness", "contradiction_rate", "answer_relevance", "rougeL", "bertscore_f1", "flesch", "words",
           "has_citation", "abstained", "judge_empathy", "judge_safety", "judge_helpfulness", "judge_correctness",
           "judge_faithfulness"]


def aggregate(answers: list[dict], model_info: list[dict], cfg: dict) -> dict:
    groups: dict = {}
    for r in answers:
        groups.setdefault((r["model"], r["profile"], r["dataset"]), []).append(r)
    summary = []
    for (model, profile, dataset), rs in sorted(groups.items()):
        row = {"model": model, "profile": profile, "dataset": dataset, "n": len(rs),
               "errors": sum(bool(r.get("error")) for r in rs)}
        for m in METRICS:
            vals = [float(r["metrics"][m]) for r in rs if r.get("metrics", {}).get(m) is not None]
            if vals:
                mean, lo, hi = bootstrap_ci(vals, n_boot=1000)
                row[m] = {"mean": mean, "lo": lo, "hi": hi, "n": len(vals)}
        t = [r["timing"] for r in rs if r.get("generated")]
        for k in ("ttft_ms", "total_ms", "tokens_per_s"):
            v = [x[k] for x in t if x.get(k) is not None]
            if v:
                row[k] = {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(np.mean(v))}
        row["gate_labels"] = {g: sum(r.get("gate") == g for r in rs) for g in {r.get("gate") for r in rs}}
        usage = [r.get("usage") or {} for r in rs]
        pin = sum(u.get("prompt_tokens") or 0 for u in usage)
        pout = sum(u.get("completion_tokens") or 0 for u in usage)
        spec = load_catalog().get(model)
        if spec and spec.backend == "openai_compatible":
            pi, po = spec.raw.get("price_per_m_input"), spec.raw.get("price_per_m_output")
            row["tokens_per_query"] = {"input": pin / len(rs), "output": pout / len(rs)}
            row["cost_per_1k_queries_usd"] = (
                1000 * (pin / len(rs) * pi + pout / len(rs) * po) / 1e6 if pi is not None and po is not None
                else "TODO(run): set price_per_m_input/output in configs/models.yaml"
            )
        summary.append(row)
    return {"summary": summary, "models": model_info, "significance": _significance(groups),
            "verbosity_bias": _verbosity(answers), "judge_version": JUDGE_VERSION}


def _significance(groups) -> dict:
    out = {}
    for model, dataset in {(m, d) for m, _, d in groups}:
        for metric in ("faithfulness", "rougeL", "bertscore_f1", "answer_relevance", "judge_helpfulness"):
            raw, wil, delta = {}, {}, {}
            for a, b in (("full", "naive"), ("full", "no_rag"), ("naive", "no_rag")):
                ga, gb = groups.get((model, a, dataset)), groups.get((model, b, dataset))
                if not ga or not gb:
                    continue
                va = {r["id"]: r["metrics"].get(metric) for r in ga if r.get("metrics")}
                vb = {r["id"]: r["metrics"].get(metric) for r in gb if r.get("metrics")}
                ids = [i for i in va if va[i] is not None and vb.get(i) is not None]
                if len(ids) < 5:
                    continue
                x, y = [va[i] for i in ids], [vb[i] for i in ids]
                k = f"{a}>{b}"
                raw[k], wil[k], delta[k] = paired_bootstrap_p(x, y), wilcoxon_p(x, y), float(np.mean(x) - np.mean(y))
            if raw:
                out[f"{model}|{dataset}|{metric}"] = {"delta": delta, "p_bootstrap": raw, "p_bootstrap_holm": holm(raw),
                                                      "p_wilcoxon": wil, "p_wilcoxon_holm": holm(wil)}
    return out


def _verbosity(answers) -> dict:
    from scipy.stats import spearmanr

    out = {}
    for k in ("judge_helpfulness", "judge_empathy", "judge_correctness"):
        pairs = [(r["metrics"]["words"], r["metrics"][k]) for r in answers if r.get("metrics", {}).get(k) is not None]
        if len(pairs) >= 10:
            rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            out[k] = {"spearman_rho_vs_length": float(rho), "p": float(p), "n": len(pairs)}
    return out


def run_generation_experiment(cfg: dict, settings: Settings, limit: int | None = None) -> dict:
    out_dir = settings.results_dir / "generation" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    records = generate(cfg, settings, out_dir, limit)
    model_info = [r for r in records if "status" in r]
    answers = score([r for r in records if "answer" in r], cfg)
    with open(out_dir / "scored.jsonl", "w", encoding="utf-8") as f:
        for r in answers:
            f.write(json.dumps({k: v for k, v in r.items() if k != "context"}, ensure_ascii=False) + "\n")
    res = aggregate(answers, model_info, cfg)
    res.update(name=cfg["name"], config=cfg)
    return res


# ------------------------------------------------------------------------------ tables
PROFILE_NAMES = {"no_rag": "No RAG", "naive": "Naive RAG (v1)", "full": "Full pipeline"}


def _cell(row, m, nd=3):
    v = row.get(m)
    if not v:
        return "--"
    return f"{v['mean']:.{nd}f}"


def render_tables(res: dict) -> None:
    name = res["name"]
    rows = res["summary"]
    for dataset in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == dataset]
        if dataset == "oos_questions":
            table = [[esc(r["model"]), PROFILE_NAMES.get(r["profile"], r["profile"]), _cell(r, "abstained"),
                      f"{1 - r['abstained']['mean']:.3f}" if r.get("abstained") else "--",
                      _cell(r, "judge_safety", 2), str(r["n"])] for r in rs]
            write_table(f"generation_{name}_oos", ["Model", "Setting", "Abstain/refuse", "Answered (halluc.)",
                                                  "Judge safety", "n"], table,
                        caption="Out-of-scope questions (50): rate at which the system declines or says it lacks "
                                "information (higher is better; answering is counted as a potential hallucination). "
                                "Abstention is detected with a fixed phrase list (eval/judges.py).",
                        label=f"tab:gen-{name}-oos", align="llrrrr", source=f"results/generation_{name}.json")
            continue
        table = []
        for r in rs:
            table.append([esc(r["model"]), PROFILE_NAMES.get(r["profile"], r["profile"]), _cell(r, "faithfulness"),
                          _cell(r, "answer_relevance"), _cell(r, "rougeL"), _cell(r, "bertscore_f1"),
                          _cell(r, "flesch", 1), _cell(r, "has_citation", 2), _cell(r, "judge_helpfulness", 2),
                          fmt(r["ttft_ms"]["p50"] / 1000, 2) if r.get("ttft_ms") else "--"])
        write_table(f"generation_{name}_{dataset}",
                    ["Model", "Setting", "Faithful.", "Relev.", "ROUGE-L", "BERTScore", "Flesch", "Cites", "Judge help.",
                     "TTFT s"], table,
                    caption=f"Generation quality on {dataset.replace('_', '-')}. Faithfulness = share of answer sentences "
                            "entailed by a retrieved passage (NLI, DeBERTa-v3); Relev. = question-answer embedding "
                            "cosine; Cites = share of answers with an inline citation; Judge = LLM-as-judge (1--5), "
                            "`--' where not run.",
                    label=f"tab:gen-{name}-{dataset.replace('_', '-')}", align="llrrrrrrrr",
                    source=f"results/generation_{name}.json")
