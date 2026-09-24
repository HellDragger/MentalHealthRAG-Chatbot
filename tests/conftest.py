from __future__ import annotations

import os

import pytest

from mhrag.config import Settings
from tests.fixtures import build_fixture_raw_data

# Tests never touch the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture()
def settings(tmp_path) -> Settings:
    raw = build_fixture_raw_data(tmp_path)
    s = Settings()
    s.paths.raw_data = str(raw)
    s.paths.artifacts = str(tmp_path / "artifacts")
    s.paths.results = str(tmp_path / "results")
    s.embedder_backend = "hashing"
    s.index.embedder = "minilm"
    s.chunking.chunk_tokens = 128
    s.chunking.min_chunk_tokens = 5
    s.safety.classifier = "none"
    s.llm.model = "mock"
    s.llm.served_models = ["mock"]
    s.server.warmup = False
    s.ingest.boilerplate_min_docs = 2
    return s


@pytest.fixture()
def built_index(settings):
    from mhrag.index.builder import build_index

    path, stats, built = build_index(settings)
    assert built
    return settings, path, stats
