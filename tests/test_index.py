import json

import pytest

from mhrag.index.builder import build_index
from mhrag.index.embedders import make_embedder
from mhrag.index.store import IndexMismatchError, IndexMissingError, load_index


def test_index_roundtrip_persists(built_index):
    settings, path, stats = built_index
    emb = make_embedder("minilm", settings.embedder_cfg("minilm"), "hashing")
    idx = load_index(path, emb.identity())
    assert len(idx.chunks) == stats["chunking"]["chunks"] > 0
    assert idx.dense.emb.shape[0] == len(idx.chunks)
    m = json.loads((path / "manifest.json").read_text())
    assert m["embedder"]["model"] == settings.embedder_cfg("minilm").model
    assert m["n_chunks"] == len(idx.chunks)


def test_build_is_idempotent(built_index):
    settings, path, _ = built_index
    _, _, built = build_index(settings)
    assert not built
    _, _, built = build_index(settings, force=True)
    assert built


def test_rebuild_when_corpus_config_changes(built_index):
    settings, _, _ = built_index
    settings.ingest.include_faq = False
    _, _, built = build_index(settings)
    assert built


def test_missing_index_raises_actionable_error(tmp_path):
    with pytest.raises(IndexMissingError, match="build_index"):
        load_index(tmp_path / "nope")


def test_embedder_mismatch_detected(built_index):
    settings, path, _ = built_index
    other = make_embedder("e5-base", settings.embedder_cfg("e5-base"), "hashing")
    with pytest.raises(IndexMismatchError):
        load_index(path, other.identity())


def test_every_chunk_has_citation_metadata(built_index):
    settings, path, _ = built_index
    idx = load_index(path)
    for c in idx.chunks:
        assert c.source_file and c.title and c.source_type
        assert c.section is not None
    assert any(c.url and c.url.startswith("https://www.mind.org.uk") for c in idx.chunks)
