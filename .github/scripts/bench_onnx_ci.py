"""
CI-only benchmark: PyTorch (FP32) vs ONNX INT8-quantized DistilBERT latency,
run on a real x86_64 GitHub Actions runner (native hardware, no emulation).

Builds a model with the exact same architecture as the fine-tuned checkpoint
(distilbert-base-uncased, 3-class sequence classification head) using public
pretrained weights — latency depends on the compute graph and quantization
scheme, not the specific fine-tuned weight values, so this is a faithful
stand-in without uploading the real checkpoint to GitHub.

Mirrors convert_to_onnx.py's export/quantization config and bench_onnx.py's
timing methodology so the numbers are comparable to what Lambda actually runs.
"""
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

BASE_MODEL = "distilbert-base-uncased"
N = 100
WARMUP = 15

TEXTS = [
    "Revenue increased 12% year over year, beating consensus estimates.",
    "The company missed on both top and bottom line, and guidance was cut for the full year.",
    "Margins were flat as input costs offset pricing gains, management reiterated its prior outlook.",
    "We are pleased to report record free cash flow generation this quarter, driven by strong demand across all segments and disciplined cost management.",
    "Shares fell sharply after the CFO flagged softening enterprise demand and elevated churn in the SMB segment heading into next quarter.",
    "Guidance was reiterated for the full year despite macro headwinds in EMEA.",
]


def build_reference_model(tmp_dir: Path) -> Path:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    config = AutoConfig.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
        label2id={"Negative": 0, "Neutral": 1, "Positive": 2},
    )
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model_dir = tmp_dir / "model"
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return model_dir


def export_and_quantize(model_dir: Path, output_dir: Path) -> Path:
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    ort_model = ORTModelForSequenceClassification.from_pretrained(str(model_dir), export=True)
    ort_model.save_pretrained(str(output_dir))

    quantizer = ORTQuantizer.from_pretrained(str(output_dir))
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(output_dir), quantization_config=qconfig)
    return output_dir / "model_quantized.onnx"


def load_pytorch_pipeline(model_dir: Path):
    from transformers import pipeline
    return pipeline("text-classification", model=str(model_dir), tokenizer=str(model_dir), device=-1, top_k=None)


def load_onnx_session(onnx_path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def bench_pytorch(pipe, n, warmup):
    for i in range(warmup):
        pipe(TEXTS[i % len(TEXTS)], truncation=True, max_length=128)
    latencies = []
    for i in range(n):
        t0 = time.perf_counter()
        pipe(TEXTS[i % len(TEXTS)], truncation=True, max_length=128)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def bench_onnx(session, tokenizer, n, warmup):
    input_names = {inp.name for inp in session.get_inputs()}

    def run(text):
        enc = tokenizer(text, return_tensors="np", truncation=True, max_length=128, padding="max_length")
        inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
        return session.run(["logits"], inputs)

    for i in range(warmup):
        run(TEXTS[i % len(TEXTS)])
    latencies = []
    for i in range(n):
        t0 = time.perf_counter()
        run(TEXTS[i % len(TEXTS)])
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def summarize(name, latencies):
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=100)[94]
    p99 = statistics.quantiles(latencies, n=100)[98]
    print(f"{name:>12}  mean={mean:7.2f}ms  median={median:7.2f}ms  p95={p95:7.2f}ms  p99={p99:7.2f}ms  n={len(latencies)}")
    return mean, median


if __name__ == "__main__":
    from transformers import AutoTokenizer

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print(f"Building reference model ({BASE_MODEL}, 3-class head)...")
        model_dir = build_reference_model(tmp_dir)

        print("Exporting + INT8-quantizing (AVX2, dynamic)...")
        onnx_path = export_and_quantize(model_dir, tmp_dir / "onnx")

        print(f"\nBenchmarking on native x86_64 ({WARMUP} warmup + {N} timed calls per backend)...\n")
        pt_pipe = load_pytorch_pipeline(model_dir)
        onnx_session = load_onnx_session(onnx_path)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        pt_latencies = bench_pytorch(pt_pipe, N, WARMUP)
        onnx_latencies = bench_onnx(onnx_session, tokenizer, N, WARMUP)

        print()
        pt_mean, pt_median = summarize("PyTorch", pt_latencies)
        onnx_mean, onnx_median = summarize("ONNX INT8", onnx_latencies)

        reduction_mean = (1 - onnx_mean / pt_mean) * 100
        print(f"\nLatency reduction (mean):   {reduction_mean:+.1f}%")
        print(f"For the resume bullet: X ~= {round(onnx_median)}ms (ONNX INT8 median), N = {N} calls")
