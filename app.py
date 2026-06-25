"""
Financial Sentiment Analyzer — FastAPI + Mangum (AWS Lambda)

Model is loaded once at module import time so Lambda reuse hits the warm path.
ONNX path is preferred when model/onnx/model_quantized.onnx or model/onnx/model.onnx
exists; falls back to PyTorch pipeline otherwise.
"""
import json
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading — happens at cold-start, cached for warm invocations
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/var/task/model"))

# Prefer INT8-quantized model (half the size, ~2× faster on Lambda's x86)
_ONNX_DIR = MODEL_DIR / "onnx"
_ONNX_QUANTIZED = _ONNX_DIR / "model_quantized.onnx"
_ONNX_BASE = _ONNX_DIR / "model.onnx"

if _ONNX_QUANTIZED.exists():
    ONNX_PATH: Path | None = _ONNX_QUANTIZED
elif _ONNX_BASE.exists():
    ONNX_PATH = _ONNX_BASE
else:
    ONNX_PATH = None


def _load_id2label() -> dict[int, str]:
    """Read label mapping from the fine-tuned model's config.json.

    Avoids the fragile assumption that FinancialPhraseBank always uses
    {0: Negative, 1: Neutral, 2: Positive} — the actual order depends on
    how HuggingFace assigned integer ids during fine-tuning.
    """
    cfg_path = MODEL_DIR / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        if "id2label" in cfg:
            return {int(k): v for k, v in cfg["id2label"].items()}
    logger.warning("config.json missing or has no id2label; using default FinancialPhraseBank mapping")
    return {0: "Negative", 1: "Neutral", 2: "Positive"}


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


def _load_onnx_session(path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1    # Lambda vCPU is single-threaded; extra threads add lock overhead
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    logger.info("Loading ONNX session from %s", path)
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _load_pytorch_pipeline():
    from transformers import pipeline
    logger.info("Loading PyTorch pipeline from %s", MODEL_DIR)
    return pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        device=-1,      # force CPU
        top_k=None,
    )


# Eager global load — intentional module-level side-effect for Lambda warmth
ID2LABEL = _load_id2label()
TOKENIZER = _load_tokenizer()

if ONNX_PATH is not None:
    _USE_ONNX = True
    SESSION = _load_onnx_session(ONNX_PATH)
    PIPE = None
    logger.info("ONNX backend active (%s)", ONNX_PATH.name)
else:
    _USE_ONNX = False
    SESSION = None
    PIPE = _load_pytorch_pipeline()
    logger.info("PyTorch backend active")

# ---------------------------------------------------------------------------
# Cold-start warmup — runs one dummy inference so the first real request
# doesn't pay the JIT graph-compilation cost.
# ---------------------------------------------------------------------------
_WARMUP_TEXT = "Revenue increased 12% year over year."


def _warmup() -> None:
    try:
        if _USE_ONNX:
            enc = TOKENIZER(
                _WARMUP_TEXT, return_tensors="np",
                truncation=True, max_length=128, padding="max_length",
            )
            inputs = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in {inp.name for inp in SESSION.get_inputs()}:
                inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
            SESSION.run(["logits"], inputs)
        else:
            PIPE(_WARMUP_TEXT, truncation=True, max_length=128)
        logger.info("Warmup inference complete")
    except Exception as exc:
        logger.warning("Warmup inference failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Financial Sentiment Analyzer", version="1.0.0")

_ValidLabel = Literal["Positive", "Neutral", "Negative"]


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: _ValidLabel
    score: float
    scores: dict[str, float]


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _predict_onnx(text: str) -> SentimentResponse:
    encoding = TOKENIZER(
        text,
        return_tensors="np",
        truncation=True,
        max_length=128,
        padding="max_length",     # fixed shape avoids ORT re-tracing
    )
    inputs = {
        "input_ids": encoding["input_ids"].astype(np.int64),
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in {inp.name for inp in SESSION.get_inputs()}:
        inputs["token_type_ids"] = encoding["token_type_ids"].astype(np.int64)

    logits = SESSION.run(["logits"], inputs)[0][0]
    probs = _softmax(logits)
    idx = int(probs.argmax())
    scores = {ID2LABEL[i]: float(probs[i]) for i in range(len(ID2LABEL))}
    return SentimentResponse(label=ID2LABEL[idx], score=float(probs[idx]), scores=scores)


def _predict_pytorch(text: str) -> SentimentResponse:
    raw = PIPE(text, truncation=True, max_length=128)
    # Pipeline label strings come from config.json id2label; remap to our ID2LABEL
    # in case they differ in capitalisation or format (e.g. "LABEL_0" → "Negative").
    label_map = {v.upper(): v for v in ID2LABEL.values()}
    all_scores = {label_map.get(r["label"].upper(), r["label"]): r["score"] for r in raw[0]}
    top = max(raw[0], key=lambda r: r["score"])
    top_label = label_map.get(top["label"].upper(), top["label"])
    return SentimentResponse(label=top_label, score=top["score"], scores=all_scores)


@app.post("/predict", response_model=SentimentResponse)
def predict(req: SentimentRequest) -> SentimentResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")
    if len(text) > 2000:
        raise HTTPException(status_code=422, detail="text exceeds 2000 character limit")
    return _predict_onnx(text) if _USE_ONNX else _predict_pytorch(text)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "onnx" if _USE_ONNX else "pytorch",
        "onnx_model": ONNX_PATH.name if ONNX_PATH else None,
        "labels": ID2LABEL,
    }


# Mangum wraps FastAPI for the Lambda event bridge
handler = Mangum(app, lifespan="off")
