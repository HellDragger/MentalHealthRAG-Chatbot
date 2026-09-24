"""Model catalogue + a manager that loads each backend once and keeps local models bounded in memory."""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path

from mhrag.config import load_yaml
from mhrag.llm.base import Backend, BackendError, ModelSpec

log = logging.getLogger(__name__)
LOCAL_BACKENDS = {"hf", "llamacpp"}


def load_catalog(path: str | Path = "models.yaml") -> dict[str, ModelSpec]:
    data = load_yaml(path)["models"]
    return {
        k: ModelSpec(key=k, backend=v["backend"], label=v.get("label", k), family=v.get("family", "generic"),
                     context=int(v.get("context", 8192)), raw=v)
        for k, v in data.items()
    }


def availability(spec: ModelSpec) -> tuple[bool, str]:
    """Cheap check (no downloads, no network) of whether a model can be served here."""
    raw = spec.raw
    if spec.backend == "mock":
        return True, "ok"
    if spec.backend == "openai_compatible":
        key_env = raw.get("api_key_env")
        if key_env and not os.environ.get(key_env) and not raw.get("api_key_optional"):
            return False, f"set {key_env}"
        for f in ("base_url", "api_model"):
            if not (raw.get(f) or os.environ.get(raw.get(f"{f}_env", "") or "_none_")):
                return False, f"set {raw.get(f + '_env')}"
        return True, "ok"
    if spec.backend == "ollama":
        return True, "requires a running Ollama server"
    if spec.backend == "llamacpp":
        if importlib.util.find_spec("llama_cpp") is None:
            return False, "pip install llama-cpp-python"
        return True, "ok"
    if spec.backend == "hf":
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
            return False, "pip install '.[gpu]'"
        if raw.get("gated") and not os.environ.get("HF_TOKEN"):
            return False, "gated: accept licence on HF and set HF_TOKEN"
        return True, "ok"
    return False, f"unknown backend {spec.backend}"


def make_backend(spec: ModelSpec, timeout_s: float = 120.0) -> Backend:
    if spec.backend == "mock":
        from mhrag.llm.mock import MockBackend

        return MockBackend(spec)
    if spec.backend == "openai_compatible":
        from mhrag.llm.remote import OpenAICompatibleBackend

        return OpenAICompatibleBackend(spec, timeout_s=timeout_s)
    if spec.backend == "ollama":
        from mhrag.llm.remote import OllamaBackend

        return OllamaBackend(spec, timeout_s=timeout_s)
    if spec.backend == "llamacpp":
        from mhrag.llm.llamacpp import LlamaCppBackend

        return LlamaCppBackend(spec)
    if spec.backend == "hf":
        from mhrag.llm.hf import HFBackend

        return HFBackend(spec)
    raise BackendError(f"unknown backend {spec.backend}")


class ModelManager:
    """Loads backends lazily and once. At most `max_local` heavy local models stay resident (LRU)."""

    def __init__(self, catalog: dict[str, ModelSpec], served: list[str], default: str, max_local: int = 1,
                 timeout_s: float = 120.0):
        unknown = [k for k in served + [default] if k not in catalog]
        if unknown:
            raise KeyError(f"Models not in configs/models.yaml: {unknown}")
        self.catalog = catalog
        self.served = list(dict.fromkeys([default] + served))
        self.default = default
        self.max_local = max_local
        self.timeout_s = timeout_s
        self._loaded: OrderedDict[str, Backend] = OrderedDict()
        self._lock = threading.Lock()

    def list(self) -> list[dict]:
        out = []
        for k in self.served:
            spec = self.catalog[k]
            ok, why = availability(spec)
            out.append({"key": k, "label": spec.label, "backend": spec.backend, "available": ok,
                        "status": why, "loaded": k in self._loaded, "default": k == self.default})
        return out

    def get(self, key: str | None = None) -> Backend:
        key = key or self.default
        if key not in self.served:
            raise BackendError(f"model {key!r} is not served by this server")
        with self._lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._loaded[key]
            spec = self.catalog[key]
            ok, why = availability(spec)
            if not ok:
                raise BackendError(f"model {key!r} unavailable: {why}")
            if spec.backend in LOCAL_BACKENDS:
                local = [k for k, b in self._loaded.items() if self.catalog[k].backend in LOCAL_BACKENDS]
                while len(local) >= self.max_local:
                    old = local.pop(0)
                    log.info("Unloading %s to make room for %s", old, key)
                    self._loaded.pop(old).close()
            backend = make_backend(spec, self.timeout_s)
            self._loaded[key] = backend
            return backend
