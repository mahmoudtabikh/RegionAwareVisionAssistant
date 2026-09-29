# Region-Aware Vision Assistant

Anomaly detection on material surfaces (leather, wood), plus a local LLM that explains the result to a QA reviewer. Grounded in real methodology docs, not generic RAG filler — the LLM can only say what's actually documented, and it's told explicitly not to invent confidence percentages or defect types.

Built to extend production CV/edge-AI experience (EfficientAD, ONNX, edge inference) with a modern GenAI stack (RAG, vector search, local LLM). The RAG corpus is my own documentation of the system's own methodology, not external knowledge or LLM-generated text — I wanted every claim in the corpus to be something I could personally defend.

## What it does

1. **CV**: two EfficientAD-S models (leather, wood, trained separately on MVTec AD) flag defects and return structured region-level output — polygons, bbox, area, compactness per region.
2. **Explanation**: a RAG pipeline pulls the relevant docs for the result (category performance, known limitations, explanation rules), and a local LLM (Qwen3-8B via Ollama) explains it in plain language — grounded in those docs, with rules against stating fake probabilities, referencing file names, or inventing region detail.
3. **Serving**: FastAPI (`/predict/`, `/explain/`), containerized with Docker Compose alongside Qdrant and Ollama.

## Architecture

```
                    ┌─────────────┐
   image + category │   FastAPI   │
  ─────────────────▶│   /predict/ │──▶ ONNX Runtime (EfficientAD-S, CPU)
                     │             │       │
                     │             │       ▼
                     │             │  region extraction (OpenCV contours)
                     │             │       │
                     │             │       ▼
                     │             │  structured JSON (score, regions,
                     │             │  polygon/bbox/area/compactness)
                     └─────────────┘
                            │
                            │ predict() output
                            ▼
                     ┌─────────────┐
                     │   FastAPI   │──▶ Qdrant (vector search over
                     │   /explain/ │       methodology docs, category-filtered)
                     │             │       │
                     │             │       ▼
                     │             │  Ollama (Qwen3-8B) — grounded,
                     │             │  rule-constrained explanation
                     └─────────────┘
```

## Stack

- **CV**: PyTorch, anomalib (EfficientAD-S), ONNX Runtime, OpenCV
- **Serving**: FastAPI, Uvicorn
- **RAG**: LangChain, Qdrant, HuggingFace embeddings (`BAAI/bge-small-en-v1.5`)
- **LLM**: Ollama (Qwen3-8B), local, no cloud dependency
- **Infra**: Docker Compose (`api`, `qdrant`, `ollama`), WSL2/Ubuntu dev environment, GPU passthrough

## Why these choices

- **EfficientAD-S** over PatchCore/PaDiM: built for millisecond latency, which matches the edge-AI work this project extends. Trade-off is slightly weaker pixel-level localization for a real speed gain — not chasing a benchmark number.
- **ONNX export**: verified against the original PyTorch checkpoint, not assumed correct. Max deviation ~6e-5 across score, anomaly map, and mask.
- **Threshold via F1-sweep on a held-out validation split**, not the test set the final metrics get reported on. Chose this over a default threshold specifically to keep the reported numbers honest.
- **RAG corpus = my own methodology docs**, not external domain knowledge or LLM-generated content. This was the main credibility decision on the project — every fact the LLM can cite is something I actually verified while building the system.
- **Local LLM over a cloud API**: consistent with the rest of the project, and a genuinely more interesting technical problem — running a quantized 7-8B model inside an 8GB VRAM budget.

## Results

| Category | Threshold | Precision | Recall | F1 | Test set size |
|---|---|---|---|---|---|
| Leather | 0.5046 | 1.00 | 0.92 | 0.958 | 100 (held-out) |
| Wood | 0.5002 | 0.94 | 0.98 | 0.959 | 64 (held-out) |

Thresholds picked on a validation split (20% of MVTec's test set, stratified, seed-fixed), applied unmodified to a disjoint held-out test split. Verified zero image overlap between val and test.

Leather and wood land on different points of the precision/recall trade-off with the same methodology — leather favors precision (no false alarms, ~8% of defects missed), wood favors recall (nearly all defects caught, ~6% false-alarm rate). That's the two materials' score distributions, not a methodology difference.

**Known limitations**, documented not hidden: leather struggles most with subtle discoloration; wood struggles most with liquid-type defects and is inconsistent on scratches. Details in `docs/category_performance_leather.md` and `docs/category_performance_wood.md`.

## Performance

Measured end-to-end, over HTTP, against the running Docker Compose stack (50 runs each, 3 warmup runs discarded, on the RTX 3070 / WSL2 dev machine):

| | mean | median | p95 | min | max |
|---|---|---|---|---|---|
| `/predict/` only | 155.5 ms | 152.4 ms | 187.3 ms | 132.2 ms | 210.8 ms |
| `/predict/` + `/explain/` | 16.7 s | 16.7 s | 24.6 s | 6.4 s | 34.8 s |

`/predict/` is full-stack HTTP latency (request handling, image decode/preprocessing, ONNX inference on CPU) — not raw model forward-pass time, which is faster; EfficientAD-S itself is a millisecond-scale model, the 150ms here is the full request cycle around it.

The `/explain/` step dominates total latency and has high variance. Qwen3's "thinking" mode (extended reasoning before the final answer) is enabled and is the likely main driver of both the high mean and the spread — disabling it is a known, not-yet-done optimization (see below).

## Running it

```bash
docker compose up -d
docker exec -it <ollama-container-name> ollama pull qwen3:8b   # first run only
```

Then `http://localhost:8000/docs` for the API. `POST /predict/` with an image + category (`leather` or `wood`) returns the structured detection output; feed that into `POST /explain/` for the explanation.

## Local development / testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Covers: region extraction (`extract_regions` — thresholding, contours, bbox/area/compactness), image preprocessing (`process_image` — colour conversion, resize, normalization), RAG doc loading (`load_documents`, category/doc-type classification), and both API endpoints with the CV/LLM/vector-store dependencies mocked out. 18 tests, all passing.

This is unit/integration coverage of the code paths — not a systematic evaluation of retrieval or generation quality. See below.

Linting: `ruff check .` / `ruff format .`.

## Project structure

```
src/
  training/   — training, validation, testing scripts (anomalib/EfficientAD)
  release/    — ONNX export + numerical parity check
  lib/        — shared inference and region-extraction logic
  rag/        — doc loading, one-time Qdrant indexing
  llm/        — RAG retrieval + LLM explanation generation
  api/        — FastAPI service (predict, explain)
docs/         — the RAG corpus: methodology, per-category performance, explanation rules, examples
results/      — checkpoints, ONNX exports, eval outputs (gitignored)
data/         — MVTec AD + Imagenette (gitignored, auto-downloaded on first training run)
```

## What's verified vs. not yet

**Verified**: ONNX export parity (quantitative), val/test split independence (checked directly, zero overlap), threshold methodology (leakage-free), retrieval correctness (spot-checked across categories and query types), the full containerized stack (all three services on real service-name networking, not host-networking luck).

**Not done yet** — deliberate scope boundary, not an oversight:
- A systematic evaluation of retrieval quality and LLM rule-compliance at scale (tests exist and pass — see above — but generation/retrieval quality itself is spot-checked, not exhaustively evaluated)
- Calibrated confidence (Platt/isotonic). Right now it's threshold-based binary classification only; the raw score is explicitly documented as not a probability
- Disabling Qwen3's thinking mode for the explanation step — likely the main lever on the `/explain/` latency and variance noted above, not yet applied

## Background

Extends production CV/edge-AI work — including a patented AI-driven material authentication system — into RAG and local LLM serving, to close the gap between production CV experience and current ML/Applied AI role requirements.