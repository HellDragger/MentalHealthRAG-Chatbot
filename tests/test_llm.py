import json

import httpx
import pytest

from mhrag.llm import remote
from mhrag.llm.base import BackendError, GenerationParams, ModelSpec
from mhrag.llm.mock import FAIL_TRIGGER, MockBackend
from mhrag.llm.registry import ModelManager, availability, load_catalog
from mhrag.llm.templates import (
    llama2_prompt,
    merge_system,
    prepare_messages,
    strip_reasoning,
    template_kwargs,
)

MSGS = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}]


def test_catalog_entries_are_well_formed():
    cat = load_catalog()
    required = {"mistral-7b-instruct-v0.3", "llama-3.1-8b-instruct", "llama-3.2-3b-instruct", "qwen2.5-7b-instruct",
                "qwen2.5-1.5b-instruct", "qwen3-4b", "qwen3-8b", "phi-3.5-mini-instruct", "phi-4-mini-instruct",
                "mentallama-chat-7b", "llama-3.3-70b-groq", "gemma-2-2b-it", "gemma-3-12b-it", "mock"}
    assert required <= set(cat)
    for k, spec in cat.items():
        assert spec.backend in {"hf", "llamacpp", "ollama", "openai_compatible", "mock"}, k
        if spec.backend == "hf":
            assert "hf_id" in spec.raw and "license" in spec.raw and "gated" in spec.raw, k
        if spec.backend == "openai_compatible":
            assert "api_key_env" in spec.raw, k
            assert "api_key" not in spec.raw, "API keys must never be hard-coded"


def test_gemma_has_no_system_role():
    out = prepare_messages(MSGS, "gemma")
    assert [m["role"] for m in out] == ["user"] and out[0]["content"].startswith("SYS")
    assert prepare_messages(MSGS, "qwen2")[0]["role"] == "system"


def test_merge_system_without_user():
    assert merge_system([{"role": "system", "content": "S"}]) == [{"role": "user", "content": "S"}]


def test_llama2_prompt_format():
    p = llama2_prompt(MSGS + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "q2"}])
    assert p.startswith("<s>[INST] <<SYS>>\nSYS\n<</SYS>>\n\nhello [/INST] hi </s><s>[INST] q2 [/INST]")


def test_qwen3_non_thinking():
    assert template_kwargs("qwen3") == {"enable_thinking": False}
    assert strip_reasoning("<think>hmm</think>Answer") == "Answer"


def test_sampling_flag_follows_temperature():
    assert not GenerationParams(temperature=0).do_sample
    assert GenerationParams(temperature=0.7).do_sample


def test_mock_backend_streams_and_fails_on_request():
    b = MockBackend(ModelSpec("mock", "mock"), delay_s=0)
    out = b.generate([{"role": "user", "content": "Context:\n[1] T\nSleep helps.\n\nQuestion: ?"}], GenerationParams())
    assert "[1]" in out
    with pytest.raises(BackendError):
        b.generate([{"role": "user", "content": FAIL_TRIGGER}], GenerationParams())


def _sse_body(chunks, usage=None):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}" for c in chunks]
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}")
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


@pytest.fixture()
def fake_http(monkeypatch):
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        if request.headers.get("authorization") != "Bearer sk-test":
            return httpx.Response(401, json={"error": "bad key"})
        return httpx.Response(200, text=_sse_body(["Hel", "lo"], {"prompt_tokens": 5, "completion_tokens": 2}),
                              headers={"content-type": "text/event-stream"})

    real = httpx.Client

    def client(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    monkeypatch.setattr(remote.httpx, "Client", client)
    return calls


def test_openai_compatible_streams_and_reads_key_from_env(fake_http, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    spec = load_catalog()["llama-3.3-70b-groq"]
    b = remote.OpenAICompatibleBackend(spec)
    assert b.generate(MSGS, GenerationParams(temperature=0)) == "Hello"
    body = json.loads(fake_http[0].content)
    assert body["model"] == "llama-3.3-70b-versatile" and body["stream"] is True
    assert b.last_usage["completion_tokens"] == 2


def test_openai_compatible_bad_key_is_backend_error(fake_http, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "wrong")
    b = remote.OpenAICompatibleBackend(load_catalog()["llama-3.3-70b-groq"])
    with pytest.raises(BackendError) as e:
        b.generate(MSGS, GenerationParams())
    assert "wrong" not in str(e.value)  # never leak the key


def test_missing_api_key_marks_model_unavailable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    ok, why = availability(load_catalog()["llama-3.3-70b-groq"])
    assert not ok and "GROQ_API_KEY" in why


def test_model_manager_serves_only_configured_models():
    mm = ModelManager(load_catalog(), ["mock"], "mock")
    assert mm.get("mock") is mm.get()  # loaded once
    with pytest.raises(BackendError):
        mm.get("qwen3-8b")
    assert mm.list()[0]["key"] == "mock"
