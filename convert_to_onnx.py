"""
Convert a fine-tuned DistilBERT model to ONNX and quantize to INT8.

Run this ONCE on your dev machine (not inside Lambda) after fine-tuning:

    python convert_to_onnx.py --model_dir ./model --output_dir ./model/onnx

The script produces:
    model/onnx/model.onnx            — FP32 baseline
    model/onnx/model_quantized.onnx  — INT8 (preferred; ~50% smaller, ~2× faster)

app.py will automatically pick up model_quantized.onnx over model.onnx.
"""
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def export_and_quantize(model_dir: Path, output_dir: Path, task: str = "text-classification") -> None:
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Export to FP32 ONNX ──────────────────────────────────────────
    logger.info("Exporting %s to ONNX (FP32)…", model_dir)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(model_dir),
        export=True,
    )
    ort_model.save_pretrained(str(output_dir))
    logger.info("FP32 model saved to %s/model.onnx", output_dir)

    # ── Step 2: Dynamic INT8 quantization ────────────────────────────────────
    # Dynamic quantization needs no calibration dataset, making it ideal for
    # serverless deployments. It quantizes weight matrices to INT8 at save time
    # and quantizes activations to INT8 at inference time.
    #
    # avx2() targets Lambda's x86_64 instances.
    # If you deploy to Graviton (arm64), use AutoQuantizationConfig.arm64() instead.
    logger.info("Quantizing to INT8 (dynamic, AVX2)…")
    quantizer = ORTQuantizer.from_pretrained(str(output_dir))
    qconfig = AutoQuantizationConfig.avx2(
        is_static=False,     # dynamic quantization — no calibration data needed
        per_channel=False,   # per-tensor is safer for small models like DistilBERT
    )
    quantizer.quantize(
        save_dir=str(output_dir),
        quantization_config=qconfig,
    )
    quantized_path = output_dir / "model_quantized.onnx"
    if quantized_path.exists():
        fp32_size = (output_dir / "model.onnx").stat().st_size / 1024 / 1024
        int8_size = quantized_path.stat().st_size / 1024 / 1024
        logger.info(
            "INT8 model saved to %s  (%.1f MB → %.1f MB, %.0f%% reduction)",
            quantized_path, fp32_size, int8_size, (1 - int8_size / fp32_size) * 100,
        )
    else:
        logger.warning("Quantized model file not found at expected path %s", quantized_path)


def verify(output_dir: Path) -> None:
    """Sanity-check that the exported model produces valid logits."""
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_path = output_dir / "model_quantized.onnx"
    if not model_path.exists():
        model_path = output_dir / "model.onnx"

    tokenizer = AutoTokenizer.from_pretrained(str(output_dir.parent))
    enc = tokenizer(
        "The company reported record profits this quarter.",
        return_tensors="np",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}
    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in input_names:
        inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

    logits = session.run(["logits"], inputs)[0][0]
    probs = np.exp(logits) / np.exp(logits).sum()
    logger.info("Verification logits: %s", logits.tolist())
    logger.info("Verification probs:  %s", probs.tolist())
    logger.info("Predicted class index: %d", int(probs.argmax()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DistilBERT → ONNX + INT8 quantize")
    parser.add_argument("--model_dir", type=Path, default=Path("model"),
                        help="Path to the fine-tuned PyTorch model directory")
    parser.add_argument("--output_dir", type=Path, default=Path("model/onnx"),
                        help="Where to write model.onnx and model_quantized.onnx")
    parser.add_argument("--skip_verify", action="store_true",
                        help="Skip the post-export sanity check")
    args = parser.parse_args()

    export_and_quantize(args.model_dir, args.output_dir)
    if not args.skip_verify:
        verify(args.output_dir)
