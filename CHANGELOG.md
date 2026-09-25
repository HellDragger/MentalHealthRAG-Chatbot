# Changelog

## v2.0.0 (branch `v2-research`)

v2 is a rewrite. The v1 Flask app (`app.py`, `rag.py`), `python_scripts/`, `knowledge_base.ipynb`, `templates/` and `static/` were
removed in the Phase 1 commit; git history keeps them. Each row maps a v1 defect to where it is fixed in v2.
B1–B28 are from the author's audit (all confirmed); B29–B45 were found during the v2 audit.

### Core app (`rag.py`, `app.py`)

| # | Bug (v1) | Fix (v2) |
|---|---|---|
| B1 | `chromadb.Client()` is in-memory, so the `"mental"` collection is empty on every start; retrieval returns an empty context. `vectordb.py` upserts are lost on exit. | Chroma removed. `mhrag/index/store.py` persists a normalised float32 matrix + `chunks.jsonl` + `manifest.json` to `artifacts/index/<name>/`. Regression test: `tests/test_index.py::test_index_roundtrip_persists`. |
| B2 | Nothing builds or loads a persistent index at startup; no non-empty check. (Also: rag.py reads collection `mental`, database.py writes LangChain's default `langchain` collection in another directory, so they never meet.) | `python -m scripts.build_index` builds everything (idempotent, hash-checked). `mhrag/index/store.py::load_index` validates the manifest; `server/app.py` refuses to start with an actionable message if the index is missing, empty or built with another embedder (`tests/test_pipeline_server.py::test_refuses_without_index`). |
| B3 | Index built with `HuggingFaceEmbeddings(MiniLM)`; queries embedded by Chroma's default function via `query_texts`. | One `Embedder` object (`mhrag/index/embedders.py`) is recorded in the manifest and used for both passages and queries, with model-specific prefixes (e5 `query:`/`passage:`, bge query instruction). Mismatch raises `IndexMismatchError`. |
| B4 | Mistral-7B loaded in fp32 without quantisation; `device=-1` silently uses CPU (no MPS path). | `mhrag/llm/hf.py`: bf16/fp16, `device_map="auto"`, optional NF4 4-bit, SDPA/flash-attn, MPS on Apple Silicon, logs the device. CPU deployments use `mhrag/llm/llamacpp.py` (GGUF Q4_K_M). |
| B5 | Mistral `[INST]` chat template not applied; markdown template used. | All backends receive chat messages; `hf` uses `tokenizer.apply_chat_template`, llama.cpp uses the GGUF's template, APIs use native chat. Per-family quirks (Gemma: no system role; Qwen3: `enable_thinking=False`) in `mhrag/llm/templates.py`. |
| B6 | `return_full_text` not False, so the prompt is echoed. | `hf` backend decodes only newly generated tokens (`TextIteratorStreamer(skip_prompt=True)`). |
| B7 | `temperature`/`top_p` without `do_sample=True` (ignored, warnings). | `GenerationParams` sets `do_sample = temperature > 0`; greedy by default for reproducible eval. |
| B8 | `n_results=2`; chunks joined with a space; no sources. | Hybrid BM25+dense (RRF) over top-30 candidates → cross-encoder rerank → top-k (default 5) trimmed to a token budget; numbered `[1]…[k]` context blocks with title/section/URL; citations returned to the UI. |
| B9 | No system prompt, grounding or safety behaviour. | `mhrag/prompts.py`: grounded, empathetic, non-diagnostic system prompt with citation and "say you don't know" instructions; `mhrag/safety/` gate + output checks. |
| B10 | `app.run(debug=True)` loads the model twice and ships debug mode. | FastAPI + uvicorn, no reloader in production; model loaded once in the lifespan hook. |
| B11 | Model loaded at import; one blocking request; no streaming/timeout/concurrency. | Lifespan loading + warm-up; SSE streaming; generation runs in a worker thread with a per-request timeout and a bounded concurrency semaphore. |
| B12 | Raw `str(e)` returned to the client. | Errors logged server-side with a request id; the client gets a generic message + the id. |
| B13 | No conversation history. | `/api/chat` accepts the message history (last 6 turns, each capped at 1,500 characters); optional history-aware query rewriting (`mhrag/retrieval/rewrite.py`). |
| B14 | Deprecated LangChain imports; unnecessary `nest_asyncio`. | LangChain and `nest_asyncio` removed; the pipeline is ~plain Python. |

### Frontend (`templates/frontend.html`)

| # | Bug (v1) | Fix (v2) |
|---|---|---|
| B15 | XSS: model output inserted via `innerHTML`. | `web/app.js` renders markdown with `marked` and sanitises with DOMPurify; user text uses `textContent`. Test: `tests/test_web_static.py` checks there's no raw `innerHTML =` of untrusted strings. |
| B16 | No try/catch around fetch (loading dots stuck), no Enter-to-send, button not disabled, input cleared late. | Full error handling with a retry button, AbortController, Enter-to-send (Shift+Enter newline), disabled send while streaming, input cleared immediately. |
| B17 | No disclaimer, crisis info, sources or model selector. | Persistent disclaimer, crisis banner with region-selectable helplines, collapsible citations, model selector fed by `/api/models`. |

### Scripts (`python_scripts/`)

| # | Bug (v1) | Fix (v2) |
|---|---|---|
| B18 | `choma_online.py` broken throughout (invalid `Client(host,port)`, wrong `query()` args, `add()` without ids/documents, nonexistent `query`/`context` columns, BART-CNN summariser used for QA, `model.encode` on a seq2seq model, whole PDF as one 256-token-truncated vector, France test query). | Deleted. Replaced by `mhrag/ingest/*` + `scripts/build_index.py`. |
| B19 | `bart.py` loads GPT-J-6B; `bart.py`/`gpt2.py` call `input()` at import; base GPT-2 isn't instruction-tuned. | Deleted. Models are chosen from `configs/models.yaml`. GPT-2, GPT-J-6B and BART are kept only as clearly labelled v1 baselines, run with correct completion/seq2seq handling (`mhrag/llm/hf.py`). |
| B20 | `query_mistral.py`: wrong model id (`Mistral-8B-Instruct-2410` → `Ministral-8B-Instruct-2410`), `FAISS`/`RetrievalQA` never imported, FAISS built from raw strings with `len(str)` as dim, LLM reloaded per query, `max_tokens=8192`, invalid `RetrievalQA(prompt_template=…)`, vLLM object passed as a LangChain LLM. | Deleted. `mhrag/index/store.py::DenseIndex` (exact inner product over real embeddings; optional FAISS `IndexFlatIP`); LLM loaded once; `max_new_tokens` capped at 400. vLLM is reachable through the `openai_compatible` backend. |
| B21 | `query_rag.py`: base GPT-2, keyword-match "evaluation", reads `../data/chroma` while `database.py` writes `../data/chromaprocessed`; `database.py` and `knowledge_base_creation_chroma.py` are near-duplicates reading different folders. | Deleted. One index path from config; real evaluation in `eval/` + `scripts/run_eval.py`. |
| B22 | `text_conversion.py` embeds JSON syntax (braces, keys, indentation); `csv_to_json.py` strips `[`/`]` from every value. | Loaders emit clean natural-language documents (FAQ/fact question as the section heading, answer as the text; embedded as `title — heading` + text). No JSON is ever embedded. |
| B23 | 200-char chunks with 50 overlap (~40 tokens). | Token-based, heading-aware chunking (default 256 tokens, 15 % overlap; 128/256/512 in experiments), measured with the embedder's tokenizer. |
| B24 | Raw Reddit/Twitter posts (dreaddit, depression-reddit, Mental-Health-Twitter, mental_health.csv) indexed as "facts". | Excluded from the KB. Used only to train/evaluate the risk classifier (`scripts/train_risk_classifier.py`). |
| B25 | Path bugs: `processed_json` vs `processed_JSON`; hardcoded `../data/...`; notebook uses `"PDF Files"` (folder is `PDF_Files`) and passes directories to `CSVLoader`/`JSONLoader`. | All paths resolved from the repo root via `mhrag.config.PROJECT_ROOT`; the ingester reads `data/raw_data.zip` directly (or an extracted `data/raw_data/`). |
| B26 | `json_preprocessing.py`/`pdf_preprocessing.py` empty; `csv_preprocessing.py` relies on a global `output_dir`. | Deleted; replaced by typed loader functions with tests. |
| B27 | `requirements.txt` missing flask, langchain-huggingface, langchain-chroma, PyPDF2, pandas, faiss, vllm, accelerate; pins Windows-only `pyreadline3`, `torch==2.4.0` (no Py3.13 wheel); pulls unused kubernetes/unstructured/opentelemetry. | `pyproject.toml` with extras (`cpu`, `gpu`, `eval`, `dev`); `requirements.txt` (slim CPU server) and `requirements-eval.txt`. |
| B28 | `__pycache__/`, `.idea/`, `.vscode/` committed; `.gitignore` lacks Python artefacts; no `.env.example`; empty README. | Untracked (Phase 0 commit), new `.gitignore`, `.env.example`, full README. |

### Found during the v2 audit

| # | Bug (v1) | Fix (v2) |
|---|---|---|
| B29 | `database.py`, `knowledge_base_creation_chroma.py`, `query_mistral.py` `shutil.rmtree` the index on every run (not idempotent); call `db.persist()` (removed in langchain-chroma). | `scripts/build_index.py` is idempotent: it skips when corpus hash + embedder + chunk config match the manifest, and writes atomically (temp dir + rename). |
| B30 | `text_extraction_cleaning.py` collapses all whitespace, destroying headings; browser headers/URLs/page counters remain in the text. | `mhrag/ingest/pdf.py` parses headers into metadata (title, URL, date) and strips them; newlines are kept so headings become `section` metadata. |
| B31 | `csv_preprocessing.py` writes classifier labels into KB text ("not ptsd", "depression"/"not depression"). | Label datasets never enter the KB (see B24). |
| B32 | `vectordb.py` `CharacterTextSplitter` splits on `\n\n` which PDF text lacks → page-sized chunks silently truncated by MiniLM's 256-token limit. | Chunker enforces a token limit ≤ the embedder's `max_seq_length`; a test asserts no chunk exceeds it. |
| B33 | `query_rag.py` thresholds "relevance" 0.7 derived from L2 distance on unnormalised vectors. | Vectors are L2-normalised; cosine = inner product. No fixed relevance threshold is used for gating. |
| B34 | Frontend: body starts `light-theme` and a stored `dark-theme` is added without removing it (both classes apply); fixed `id="loading"` duplicates on concurrent sends; server `error` field never shown; timestamp via `innerHTML`; no labels/ARIA. | Rebuilt `web/`: `data-theme` attribute, per-message state, error surfacing, `aria-live` chat log, labelled controls. |
| B35 | `.gitignore` ignored all of `data/` while the 89 MB `raw_data.zip` is force-tracked (above the HF Spaces 10 MB non-LFS limit). | Explicit ignore rules; the Space sync uploads the prebuilt index and excludes the raw archive (`scripts/deploy_space.py`, run by `.github/workflows/ci.yml`). |
| B36 | MIT `LICENSE` implicitly presented as covering third-party data (© Mind, Kaggle, CounselChat). | README "Data & licences" section separates code licence from each dataset's terms. |
| B37 | Flask app but FastAPI/uvicorn pinned; unrelated heavy pins. | See B27. |
| B38 | `app.py`: `request.json.get` raises on non-JSON bodies; no input length limit. | Pydantic request model with `max_length`, 422 on bad input; per-IP rate limit (slowapi). |
| B39 | KB.json `fact-10` ("What causes mental illness?") has the answer to fact-9 (prevalence). | Dropped at ingestion and logged in `results/ingest_stats.json` (`dropped_intents`). |
| B40 | KB.json `scared` intent uses `response` whose value is a *stringified* Python list. | `mhrag/ingest/qa.py::load_intents` accepts `response`/`responses`, parses string lists with `ast.literal_eval`. (Chit-chat intents are not indexed; only `fact-*`.) |
| B41 | KB.json chit-chat (Pandora persona, "created by Akash Nath", unverified helpline 9152987821) was embedded as knowledge. | Only `fact-*` intents are indexed; helplines come solely from `configs/helplines.yaml`, each with a verified source. |
| B42 | `train.csv` duplicates `context_response_train.csv`; `mentalhealth.csv` duplicates 97/98 FAQ rows; `mentalhealth.json` duplicates FAQ 1–10; KB `fact-8…24` duplicate FAQ answers. | Duplicate files are not loaded (`train.csv`, `mentalhealth.csv`, `mentalhealth.json` are used for evaluation only); exact and near-duplicate chunks are removed across sources (`mhrag/ingest/dedup.py`); counts in `results/ingest_stats.json`. |
| B43 | `mental_health.csv` is pre-normalised (lowercased, stop-words and punctuation removed, so "not"/"no" are gone while "dont" survives), so a classifier trained on it sees a different distribution from raw chat text. | The risk classifier applies the same normaliser at inference (`mhrag/safety/normalize.py`), and the lexicon runs on raw text with a negation guard. The shift is reported in the paper. |
| B44 | `tips-to-improve-your-wellbeing-2020-easy-read.pdf` has no text layer (image-only); silently contributed nothing. | Detected and reported as skipped (`results/ingest_stats.json`). |
| B45 | No Apple-Silicon (MPS) support; torch device logic only CUDA/CPU. | `mhrag/llm/hf.py::pick_device` handles cuda → mps → cpu. |

### Issues found and fixed while building and testing v2

| # | Issue | Fix |
|---|---|---|
| D1 | A short new question right after a crisis turn ("What is OCD?") was carried over as crisis. | The carry-over rule applies only to short, non-question replies (`mhrag/safety/gate.py`). |
| D2 | Risk-gate v1 routed third-party disclosures ("my friend wants to kill herself") to the user-crisis protocol, missed method requests phrased as "…is lethal?", and let the classifier escalate explicitly negated messages ("I'm not suicidal…"). Found on the development red-team set. | Gate v2: third-party routing takes priority, a structural method-request detector, the negation guard applies to classifier decisions, and two thresholds (recall-oriented "elevated", precision-oriented "crisis"). Evaluated on a held-out set committed before evaluation (`results/safety_gate_v{1,2}_{devset,heldout}.json`). |
| D3 | The high-precision threshold could come out *below* the recall threshold for well-separated classifiers. | `tau_p = max(tau_r, tau_prec)` (`scripts/train_risk_classifier.py`). |
| D4 | The inference normaliser dropped apostrophe contractions ("don't"), while the training corpus keeps them as "dont". | Apostrophes are removed before stop-word filtering (`mhrag/safety/normalize.py`). |
| D5 | Answers that hit `max_new_tokens` stopped mid-sentence (reported by a user). | Backends report `finish_reason`. Truncated answers are trimmed to the last complete sentence or bullet, marked as shortened, and the UI offers **Continue** (`mhrag/pipeline.py`, `web/app.js`). |
| D6 | GPT-2 / GPT-J / BART / FLAN-T5 (the original v1 models) cannot use chat templates, and GPT-2 has a 1,024-token context. | `completion` and `seq2seq` model families, per-model context budgets and prompt truncation (`mhrag/llm/hf.py`, `mhrag/pipeline.py`). |
| D7 | Semaphore slots could leak if a client disconnected before streaming started. | The slot is acquired inside the SSE generator and always released in `finally` (`server/app.py`). |
| D8 | With `max_concurrent_generations=2`, two requests could run on the same llama.cpp / transformers model instance at once (not thread-safe). | Per-model `generation_lock` held for the whole stream in the local backends (`mhrag/llm/base.py`, `llamacpp.py`, `hf.py`). |
| D9 | The rate limiter trusted the left-most `X-Forwarded-For` entry, which clients control. | Use the connection address; deployments set `--proxy-headers` in uvicorn (`server/app.py`, `Dockerfile`). |
| D10 | The paper failed to compile in CI: table names with `_` in the pending-table caption (and `\todorun` arguments) are invalid in text mode; retrieval tables ran about 130 pt past the margin; two references were undefined. | `underscore` package, wrapping TODO markers, generated tabulars wrapped in `adjustbox` (`eval/tables.py`), `\tableref` for tables still pending (`paper/main.tex`). |
| D11 | The Kaggle notebook ran every experiment in one session; a run longer than Kaggle's 12-hour limit could lose all output, because the results were only zipped by the last cell. | Resumable steps with done markers (keyed to a fingerprint of each step's config and eval data), a zip after every step, and a session time budget that stops the running step cleanly (`notebooks/kaggle_experiments.ipynb`). |
| D12 | Interrupted experiments could not resume cleanly: a truncated last line crashed the generation checkpoint loader, retrieval runs were only saved at the end, completed models were reloaded, and LLM-judge verdicts were recomputed. | Tolerant checkpoint reader that repairs the file; per-run retrieval checkpoint (`*.partial.jsonl`); completed models skipped without loading; judge verdicts cached per answer (`eval/generation_eval.py`, `eval/retrieval_eval.py`). |
| D13 | A model that failed to load (out of memory or disk) aborted the whole generation benchmark. | Recorded as `TODO(run): failed to load` and the run continues with the next model. |
| D14 | `bench_latency` wrote to `results/latency.json` relative to the working directory and ignored `MHRAG_PATHS__RESULTS`, so the Kaggle latency results never reached the results zip. The Hugging Face cache also kept every model's weights, filling the disk after four or five 7–8B models. | Default output follows the configured results directory; `MHRAG_FREE_MODEL_CACHE=1` deletes a model's weights once all its answers are saved. |
