# Region-Aware Vision Assistant

An end-to-end system combining production computer vision with a locally-hosted GenAI explanation layer: anomaly detection models flag defects on material surfaces, and a RAG-grounded local LLM explains *why* to a QA reviewer — in plain language, with rules against hallucinating confidence levels or defect types the model can't actually determine.

Built as a portfolio project extending professional production CV/edge-AI experience (EfficientAD-based anomaly detection, ONNX export, edge inference) with a modern GenAI stack (RAG, vector search, local LLM serving) — deliberately grounded in the author's own verified methodology documentation rather than generic or LLM-generated content, to avoid the credibility problems common in "chat with your docs" portfolio projects.

## What it does

1. **Computer vision**: two EfficientAD-S anomaly detection models (trained separately on MVTec AD's `leather` and `wood` categories) flag surface defects and return structured, region-level output — polygon boundaries, bounding boxes, area, and compactness per flagged region.
2. **Explanation**: a RAG pipeline retrieves the relevant methodology documentation for the result (category-specific performance data, known limitations, explanation rules), and a locally-hosted LLM (Qwen3-8B via Ollama) generates a natural-language explanation for a QA operative — grounded in that documentation, with explicit rules against stating fabricated probabilities, referencing internal file names, or inventing region detail that wasn't actually detected.
3. **Serving**: both are exposed through a FastAPI service (`/predict/`, `/explain/`), fully containerized via Docker Compose alongside Qdrant (vector store) and Ollama (LLM inference).

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
- **LLM**: Ollama (Qwen3-8B), served locally, no cloud dependency
- **Infra**: Docker Compose (three services: `api`, `qdrant`, `ollama`), WSL2/Ubuntu dev environment with GPU passthrough

## Why these choices

- **EfficientAD-S** over PatchCore/PaDiM: purpose-built for millisecond-level inference latency, consistent with the author's production edge-AI background — a deliberate trade-off (slightly lower pixel-level localization precision) in exchange for speed, matched to a real deployment constraint rather than chasing benchmark leaderboard numbers.
- **ONNX export**: production-realistic serving path; export verified against the original PyTorch checkpoint with a quantitative sanity check (max output deviation ~6e-5 across score, anomaly map, and predicted mask) rather than assumed correct.
- **Threshold via F1-sweep on a held-out validation split** (not the test set the final metrics are reported on) — chosen deliberately over a default/library threshold to keep the reported test metrics honest and leakage-free.
- **RAG corpus = the author's own methodology documentation**, not external domain knowledge or LLM-generated filler — the project's central credibility decision. Every fact the LLM can cite is something the author personally verified while building the system (calibration approach, what the anomaly score does and doesn't mean, category-specific known failure modes).
- **Local LLM (Ollama) over a cloud API**: consistent with the project's local/self-hosted framing, and a legitimately more interesting technical story than "called an API" — running a quantized 7-8B model within an 8GB VRAM budget.

## Results

| Category | Threshold | Precision | Recall | F1 | Test set size |
|---|---|---|---|---|---|
| Leather | 0.5046 | 1.00 | 0.92 | 0.958 | 100 (held-out) |
| Wood | 0.5002 | 0.94 | 0.98 | — | 64 (held-out) |

Thresholds were selected on a validation split (20% of MVTec's test set, stratified, seed-fixed) and applied unmodified to a disjoint held-out test split — verified to have zero image overlap with validation.

The two categories land on different points of the precision/recall trade-off despite identical methodology: leather's threshold favors precision (no false alarms, ~8% of defects missed), wood's favors recall (nearly all defects caught, ~6% false-alarm rate) — reflecting how each material's normal-vs-defect score distributions actually separate, not a methodology difference.

**Known limitations** (documented, not hidden): leather struggles most with subtle discoloration defects; wood struggles most with liquid-type defects and shows inconsistent confidence on scratches. Full detail in `docs/category_performance_leather.md` and `docs/category_performance_wood.md`.

## Running it

```bash
docker compose up -d
docker exec -it <ollama-container-name> ollama pull qwen3:8b   # first run only
```

Then visit `http://localhost:8000/docs` for the interactive API (Swagger UI). `POST /predict/` with an image + category (`leather` or `wood`) returns structured detection output; feed that output into `POST /explain/` for a grounded natural-language explanation.

## Project structure

```
src/
  training/   — model training, validation, testing scripts (anomalib/EfficientAD)
  release/    — ONNX export + numerical parity verification
  lib/        — shared inference and region-extraction logic (used by API + scripts)
  rag/        — document loading and one-time Qdrant indexing
  llm/        — RAG retrieval + LLM explanation generation
  api/        — FastAPI service (predict, explain endpoints)
docs/         — the RAG corpus: methodology, per-category performance, explanation rules, examples
results/      — trained checkpoints, ONNX exports, evaluation outputs (gitignored)
data/         — MVTec AD + Imagenette (gitignored, auto-downloaded on first training run)
```

## What's verified vs. not yet

**Verified**: ONNX export parity (quantitative), val/test split independence (zero overlap, checked directly), threshold selection methodology (leakage-free), retrieval correctness (spot-checked across categories and query types), end-to-end containerized stack (all three services, service-name networking, not host-networking luck).

**Not yet done** (deliberate scope boundary, not an oversight):
- Formal, systematic evaluation of retrieval quality and LLM rule-compliance across a large test set (currently spot-checked, not exhaustively tested)
- Calibrated confidence (Platt/isotonic regression) — the system currently reports threshold-based binary classification only; raw anomaly score is explicitly documented as *not* a probability
- Automated test suite
- Measured end-to-end latency numbers (predict alone vs. predict+explain)

## Background

Extends production experience in computer vision and edge AI — including a patented AI-driven material authentication system — into a self-directed exploration of retrieval-augmented generation and local LLM serving, aimed at closing the gap between traditional production CV work and current ML/Applied AI role requirements.