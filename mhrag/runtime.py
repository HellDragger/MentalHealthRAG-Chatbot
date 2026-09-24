"""Wires config -> index -> retriever -> safety gate -> model manager -> pipeline. Used by the server, the
evaluation scripts and the latency benchmark so they all run exactly the same system."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mhrag.config import Settings, get_settings
from mhrag.index.store import LoadedIndex
from mhrag.llm.registry import ModelManager, load_catalog
from mhrag.pipeline import RAGPipeline
from mhrag.retrieval.factory import build_retriever
from mhrag.retrieval.retriever import Retriever
from mhrag.safety.classifier import load_classifier
from mhrag.safety.gate import SafetyGate

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    index: LoadedIndex
    retriever: Retriever
    gate: SafetyGate
    models: ModelManager
    pipeline: RAGPipeline


def build_gate(settings: Settings) -> SafetyGate:
    clf = load_classifier(settings.safety.classifier, settings.artifacts_dir)
    return SafetyGate(clf, settings.safety.crisis_threshold, settings.safety.elevated_threshold)


def resolve_auto_model(s: Settings, catalog) -> None:
    """MHRAG_LLM__MODEL=auto: use an API model when its key is set, else fall back to the local GGUF model."""
    from mhrag.llm.registry import availability

    if s.llm.model != "auto":
        return
    for key in s.llm.auto_preference:
        if key in catalog and availability(catalog[key])[0]:
            s.llm.model = key
            break
    else:
        s.llm.model = "mock"
    log.info("llm.model=auto resolved to %s", s.llm.model)
    s.llm.served_models = [m for m in s.llm.served_models if m != "auto"]


def build_runtime(settings: Settings | None = None, warmup: bool | None = None) -> Runtime:
    s = settings or get_settings()
    catalog = load_catalog()
    resolve_auto_model(s, catalog)
    retriever = build_retriever(s, need_reranker=s.retrieval.mode.endswith("_rerank"))
    gate = build_gate(s)
    models = ModelManager(catalog, s.llm.served_models, s.llm.model, timeout_s=s.llm.timeout_s)
    pipeline = RAGPipeline(s, retriever, models, gate)
    if warmup if warmup is not None else s.server.warmup:
        retriever.retrieve("warm up query about anxiety", 1, s.retrieval.mode)
        retriever._rcache.clear()
        try:
            models.get().warmup()
        except Exception as e:  # the default model may be an API model without a key
            log.warning("Default model %s not warmed up: %s", s.llm.model, e)
    return Runtime(s, retriever.index, retriever, gate, models, pipeline)
