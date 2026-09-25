"""scripts.import_results only brings in experiments the Kaggle run finished, and never loses a local measurement."""

import json

import scripts.import_results as ir


def test_import_only_finished_steps_and_merge_latency(tmp_path, monkeypatch):
    local, run = tmp_path / "results", tmp_path / "out" / "mhrag_results"
    local.mkdir()
    (run / "_checkpoints").mkdir(parents=True)
    (run / "generation" / "local").mkdir(parents=True)
    monkeypatch.setattr(ir, "RESULTS", local)

    (local / "retrieval_main.json").write_text('{"v": "laptop"}')
    (run / "retrieval_main.json").write_text('{"v": "laptop"}')              # seeded copy, step switched off
    (run / "retrieval_chunks.json").write_text('{"v": "kaggle"}')            # finished on Kaggle
    (run / "_checkpoints" / "retrieval_chunks.json").write_text("{}")
    (run / "generation_local.json").write_text("{}")                          # unfinished: no marker
    (run / "generation" / "local" / "m__full__faq_gen.jsonl").write_text("{}\n")
    (run / "retrieval_chunks.partial.jsonl").write_text("x")                  # never imported
    (local / "latency.json").write_text(json.dumps({"cpu": {"summary": 1}, "cuda": {"status": "TODO(run)"}}))
    (run / "latency.json").write_text(json.dumps({"cpu": {"status": "TODO(run)"}, "cuda": {"summary": 2}}))

    rep = ir.import_results(tmp_path / "out")
    assert "retrieval_chunks.json" in rep["imported"]
    assert json.loads((local / "retrieval_chunks.json").read_text()) == {"v": "kaggle"}
    assert rep["not_finished"] == ["generation_local"]
    assert not (local / "generation_local.json").exists() and not (local / "generation").exists()
    assert not (local / "retrieval_chunks.partial.jsonl").exists()
    lat = json.loads((local / "latency.json").read_text())
    assert lat == {"cpu": {"summary": 1}, "cuda": {"summary": 2}}  # measured entries kept and added
    assert ir.import_results(tmp_path / "out")["imported"] == []  # idempotent
