"""Experiments interrupted at a session time limit resume from their checkpoints."""

import json
from types import SimpleNamespace

import eval.retrieval_eval as re_mod
from eval.generation_eval import read_checkpoint


def test_read_checkpoint_drops_truncated_line(tmp_path):
    p = tmp_path / "m__full__faq_gen.jsonl"
    p.write_text(json.dumps({"id": 1, "answer": "a"}) + "\n" + '{"id": 2, "answ')  # killed mid-write
    rows = read_checkpoint(p)
    assert [r["id"] for r in rows] == [1]
    assert p.read_text().endswith("\n")  # the next append starts on a clean line
    with open(p, "a") as f:
        f.write(json.dumps({"id": 3}) + "\n")
    assert [r["id"] for r in read_checkpoint(p)] == [1, 3]
    assert read_checkpoint(tmp_path / "missing.jsonl") == []


def test_retrieval_experiment_resumes(tmp_path, monkeypatch):
    queries = {"heading_qa": [{"id": "q1"}, {"id": "q2"}], "synth_qa": [{"id": "s1"}]}
    calls = []

    def fake_score(retriever, qs, mode):
        calls.append(mode)
        if len(calls) > limit[0]:
            raise KeyboardInterrupt  # simulate the session being killed
        per = {"nDCG@10": [1.0] * len(qs), "MRR@10": [1.0] * len(qs)}
        return {"summary": {m: {"mean": 1.0} for m in ("R@1", "R@10", "MRR@10", "nDCG@10")}, "per_query": per}

    monkeypatch.setattr(re_mod, "load_jsonl", lambda name: queries[name.removesuffix(".jsonl")])
    monkeypatch.setattr(re_mod, "build_index", lambda *a, **k: None)
    monkeypatch.setattr(re_mod, "build_retriever",
                        lambda *a, **k: SimpleNamespace(index=SimpleNamespace(chunks=[0] * 5)))
    monkeypatch.setattr(re_mod, "score_run", fake_score)
    cfg = {"name": "t", "datasets": ["heading_qa", "synth_qa"], "chunk_sizes": [128, 256], "embedders": ["a", "b"],
           "bm25_on": "a", "systems": [{"name": "bm25", "mode": "bm25"}, {"name": "dense", "mode": "dense"}]}
    total = 2 * (1 + 2) * 2  # chunk sizes x (bm25 once + dense per embedder) x datasets
    ckpt = tmp_path / "retrieval_t.partial.jsonl"

    limit = [5]
    try:
        re_mod.run_retrieval_experiment(cfg, None, checkpoint=ckpt)
    except KeyboardInterrupt:
        pass
    assert len(ckpt.read_text().splitlines()) == 5

    calls.clear()
    limit[0] = 10**6
    res = re_mod.run_retrieval_experiment(cfg, None, checkpoint=ckpt)
    assert len(calls) == total - 5  # only the unfinished runs were scored again
    assert len(res["runs"]) == total
    assert all("_key" not in r for r in res["runs"])


def test_generation_skips_models_already_complete(tmp_path, monkeypatch):
    import eval.generation_eval as ge

    rows = {"faq_gen": [{"id": 1, "question": "q", "answer": "a"}], "oos_questions": [{"id": 7, "text": "q"}]}
    monkeypatch.setattr(ge, "load_jsonl", lambda name: [dict(r) for r in rows[name.removesuffix(".jsonl")]])
    monkeypatch.setattr(ge, "load_catalog", lambda: {})
    monkeypatch.setattr(ge, "build_gate", lambda s: None)

    def must_not_load(*a, **k):
        raise AssertionError("a completed model must not be loaded again")

    monkeypatch.setattr(ge, "ModelManager", must_not_load)
    monkeypatch.setattr(ge, "availability", must_not_load)
    cfg = {"models": ["m"], "datasets": ["faq_gen", "oos_questions"], "profiles": ["no_rag", "full"]}
    for d, rs in rows.items():
        for p in cfg["profiles"]:
            (tmp_path / f"m__{p}__{d}.jsonl").write_text(
                "".join(json.dumps({"id": r["id"], "model": "m", "profile": p, "dataset": d, "answer": "x"}) + "\n"
                        for r in rs))
    (tmp_path / "m__status.json").write_text(json.dumps({"model": "m", "status": "ok", "load_seconds": 3.0}))
    recs = ge.generate(cfg, None, tmp_path, limit=None)
    assert sum("answer" in r for r in recs) == 4
    assert {"model": "m", "status": "ok", "load_seconds": 3.0} in recs


def test_failed_answers_are_retried(tmp_path, monkeypatch):
    import eval.generation_eval as ge

    rows = {"faq_gen": [{"id": 1}, {"id": 2}]}
    monkeypatch.setattr(ge, "load_jsonl", lambda name: [dict(r) for r in rows[name.removesuffix(".jsonl")]])
    cfg = {"datasets": ["faq_gen"], "profiles": ["full"]}
    p = tmp_path / "m__full__faq_gen.jsonl"
    p.write_text(json.dumps({"id": 1, "answer": "ok"}) + "\n" + json.dumps({"id": 2, "answer": "", "error": "E"}) + "\n")
    assert not ge._complete(cfg, "m", tmp_path, None)  # id 2 failed -> the model is not finished
    with open(p, "a") as f:  # the retry succeeds; a later failure must not replace a success
        f.write(json.dumps({"id": 2, "answer": "fixed"}) + "\n" + json.dumps({"id": 1, "answer": "", "error": "E"}) + "\n")
    assert {i: r["answer"] for i, r in ge._answers(p).items()} == {1: "ok", 2: "fixed"}
    assert ge._complete(cfg, "m", tmp_path, None)


def test_judge_client_error_disables_judge_without_crashing(tmp_path, monkeypatch):
    import eval.generation_eval as ge
    import eval.judges as judges
    from mhrag.llm.base import BackendError

    class Mgr:
        def __init__(self, *a, **k):
            pass

        def get(self, key):
            return object()

    calls = []

    class Judge:
        def __init__(self, backend):
            pass

        def rubric(self, q, a, ref):
            calls.append(q)
            if len(calls) > 1:
                raise BackendError("llama-3.3-70b-groq: HTTP 404 (model not found)")
            return {"empathy": 4}

        def faithfulness(self, a, ctx):
            return 1.0

    monkeypatch.setattr(ge, "load_catalog", lambda: {"j": None})
    monkeypatch.setattr(ge, "availability", lambda spec: (True, "ok"))
    monkeypatch.setattr(ge, "ModelManager", Mgr)
    monkeypatch.setattr(judges, "LLMJudge", Judge)
    answers = [{"query": f"q{i}", "answer": "a", "profile": "no_rag", "metrics": {"words": 1}} for i in range(3)]
    ge._llm_judge(answers, {"judge_model": "j"}, tmp_path)
    assert len(calls) == 2  # stopped at the first client error
    assert all(not any(k.startswith("judge_") for k in a["metrics"]) for a in answers)  # no partial judging
    assert (tmp_path / "judge_cache.jsonl").read_text().count("\n") == 1  # the verdict obtained is kept


def _fake_judge(monkeypatch, fail_after=None, error="HTTP 429"):
    import eval.generation_eval as ge
    import eval.judges as judges
    from mhrag.llm.base import BackendError

    calls = []

    class Mgr:
        def __init__(self, *a, **k):
            pass

        def get(self, key):
            return object()

    class Judge:
        def __init__(self, backend):
            pass

        def rubric(self, q, a, ref):
            calls.append(q)
            if fail_after is not None and len(calls) > fail_after:
                raise BackendError(f"judge: {error} (rate limit reached)")
            return {"empathy": 4}

        def faithfulness(self, a, ctx):
            return 1.0

    monkeypatch.setattr(ge, "load_catalog", lambda: {"j": None})
    monkeypatch.setattr(ge, "availability", lambda spec: (True, "ok"))
    monkeypatch.setattr(ge, "ModelManager", Mgr)
    monkeypatch.setattr(judges, "LLMJudge", Judge)
    return calls


def _answers_for_judging(n=6):
    return [{"id": i, "dataset": "faq_gen", "query": f"q{i}", "answer": "a", "profile": "no_rag", "metrics": {}}
            for i in range(n)]


def test_judge_quota_exhausted_withholds_scores_then_resumes(tmp_path, monkeypatch):
    import eval.generation_eval as ge

    calls = _fake_judge(monkeypatch, fail_after=2)
    answers = _answers_for_judging()
    assert ge._llm_judge(answers, {"judge_model": "j"}, tmp_path) == "incomplete"
    assert all(not a["metrics"] for a in answers)  # no partial judging in the tables
    assert (tmp_path / "judge_cache.jsonl").read_text().count("\n") == 2  # but the verdicts are kept

    calls = _fake_judge(monkeypatch)  # quota back
    answers = _answers_for_judging()
    assert ge._llm_judge(answers, {"judge_model": "j"}, tmp_path) == "complete"
    assert len(calls) == 4  # only the 4 answers not judged before
    assert all(a["metrics"]["judge_empathy"] == 4 for a in answers)


def test_judge_sample_is_fixed_and_shared(monkeypatch, tmp_path):
    import eval.generation_eval as ge

    ids = list(range(40))
    a, b = ge.judge_sample_ids({"judge_sample": 10}, "faq_gen", ids), ge.judge_sample_ids({"judge_sample": 10}, "faq_gen", ids[::-1])
    assert a == b and len(a) == 10  # the same questions whatever the order (so for every model and setting)
    assert ge.judge_sample_ids({}, "faq_gen", ids) is None
    calls = _fake_judge(monkeypatch)
    answers = _answers_for_judging(40)
    assert ge._llm_judge(answers, {"judge_model": "j", "judge_sample": 10}, tmp_path) == "complete"
    assert len(calls) == 10 and sum("judge_empathy" in x["metrics"] for x in answers) == 10


def test_generation_skips_model_the_key_cannot_use_and_stops_on_rate_limit(tmp_path, monkeypatch):
    import pytest

    import eval.generation_eval as ge

    rows = {"faq_gen": [{"id": 1, "query": "q"}, {"id": 2, "query": "q2"}]}
    monkeypatch.setattr(ge, "load_jsonl", lambda name: [dict(r) for r in rows[name.removesuffix(".jsonl")]])
    monkeypatch.setattr(ge, "load_catalog", lambda: {"api": None})
    monkeypatch.setattr(ge, "availability", lambda spec: (True, "ok"))
    monkeypatch.setattr(ge, "build_gate", lambda s: None)
    monkeypatch.setattr(ge, "_index_for", lambda *a: (None, "e", 1, ""))

    class Backend:
        def close(self):
            pass

    class Mgr:
        def __init__(self, *a, **k):
            self._loaded = {}

        def get(self, key):
            return Backend()

    error = ["BackendError: api: HTTP 404 (no access)"]

    class Pipe:
        def __init__(self, *a):
            pass

        def params(self, **k):
            return None

        def answer(self, *a, **k):
            raise RuntimeError(error[0])

    monkeypatch.setattr(ge, "ModelManager", Mgr)
    monkeypatch.setattr(ge, "RAGPipeline", Pipe)
    cfg = {"models": ["api"], "datasets": ["faq_gen"], "profiles": ["no_rag", "full"]}
    recs = ge.generate(cfg, None, tmp_path, limit=None)
    assert recs == [{"model": "api", "status": "TODO(run): no access with this API key: RuntimeError: "
                                               "BackendError: api: HTTP 404 (no access)"}]
    assert (tmp_path / "api__no_rag__faq_gen.jsonl").read_text().count("\n") == 1  # stopped at the first failure

    error[0] = "BackendError: api: HTTP 429 (rate limit)"
    with pytest.raises(ge.RateLimited):
        ge.generate(cfg, None, tmp_path, limit=None)


def test_generate_only_saves_answers_without_scoring(tmp_path, monkeypatch):
    import yaml

    from scripts import run_eval

    cfg = {"kind": "generation", "name": "t", "models": ["mock"], "profiles": ["no_rag"], "datasets": ["faq_gen"],
           "temperature": 0.0, "judge_model": None}
    path = tmp_path / "gen.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("MHRAG_PATHS__RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("MHRAG_PATHS__ARTIFACTS", str(tmp_path / "artifacts"))  # never the repo's own folders
    monkeypatch.setenv("MHRAG_EMBEDDER_BACKEND", "hashing")  # offline, no model download
    monkeypatch.setenv("MHRAG_SAFETY__CLASSIFIER", "none")
    assert run_eval.main(["--config", str(path), "--generate-only", "--limit", "2"]) == 0
    out = tmp_path / "results" / "generation" / "t"
    assert (out / "mock__no_rag__faq_gen.jsonl").read_text().count("\n") == 2
    assert (out / "mock__status.json").exists()
    assert not (out / "scored.jsonl").exists() and not (tmp_path / "results" / "generation_t.json").exists()
    assert not (tmp_path / "artifacts" / "index").exists()  # no retrieval, so no index was built
