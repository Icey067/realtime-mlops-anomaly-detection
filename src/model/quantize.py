"""Export PyTorch Autoencoder to ONNX format and apply Dynamic INT8 Quantization.

Achieves substantial memory footprint reduction and sub-millisecond CPU inference
suitable for sub-15ms end-to-end streaming latency.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch

from src.config import Settings, get_settings
from src.model.autoencoder import Autoencoder

# Force UTF-8 on Windows stdout to prevent emoji character map errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentinel.quantize")


def export_pytorch_to_onnx(
    checkpoint_path: Path,
    onnx_output_path: Path,
    input_dim: int = 30,
    hidden_dim: int = 16,
    latent_dim: int = 8,
) -> Path:
    """Export PyTorch Autoencoder checkpoint to ONNX format with dynamic batch axes."""
    logger.info("Loading PyTorch model from: %s", checkpoint_path)
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to ONNX at opset 14: %s", onnx_output_path)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

    # Validate exported ONNX graph
    onnx_model = onnx.load(str(onnx_output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph successfully verified.")
    return onnx_output_path


def quantize_to_int8(
    onnx_input_path: Path,
    quantized_output_path: Path,
) -> Path:
    """Perform Dynamic INT8 Quantization on ONNX graph."""
    logger.info("Applying Dynamic INT8 Quantization to: %s", onnx_input_path)
    quantized_output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(onnx_input_path),
        model_output=str(quantized_output_path),
        weight_type=QuantType.QInt8,
    )

    raw_size_kb = onnx_input_path.stat().st_size / 1024
    quant_size_kb = quantized_output_path.stat().st_size / 1024
    reduction = (1.0 - quant_size_kb / max(raw_size_kb, 0.001)) * 100

    logger.info(
        "Quantization complete! Model size: %.2f KB -> %.2f KB (%.1f%% reduction)",
        raw_size_kb,
        quant_size_kb,
        reduction,
    )
    return quantized_output_path


def benchmark_sessions(
    onnx_path: Path,
    quantized_path: Path,
    input_dim: int = 30,
    iterations: int = 500,
) -> Tuple[float, float]:
    """Benchmark inference latency between float32 ONNX and quantized INT8 sessions."""
    logger.info("Running CPU latency benchmark (%d iterations)...", iterations)
    dummy_data = np.random.randn(1, input_dim).astype(np.float32)

    sess_fp32 = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])

    input_name_fp32 = sess_fp32.get_inputs()[0].name
    input_name_int8 = sess_int8.get_inputs()[0].name

    # Warmup
    for _ in range(50):
        _ = sess_fp32.run(None, {input_name_fp32: dummy_data})
        _ = sess_int8.run(None, {input_name_int8: dummy_data})

    # Benchmark FP32
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = sess_fp32.run(None, {input_name_fp32: dummy_data})
    fp32_latency_ms = ((time.perf_counter() - t0) / iterations) * 1000

    # Benchmark INT8
    t1 = time.perf_counter()
    for _ in range(iterations):
        _ = sess_int8.run(None, {input_name_int8: dummy_data})
    int8_latency_ms = ((time.perf_counter() - t1) / iterations) * 1000

    logger.info(
        "Latency Benchmark: FP32 = %.3fms / sample | INT8 = %.3fms / sample",
        fp32_latency_ms,
        int8_latency_ms,
    )
    return fp32_latency_ms, int8_latency_ms


def run_export_and_quantization(settings: Optional[Settings] = None) -> Path:
    """Orchestrate full ONNX export, quantization, and benchmarking flow."""
    settings = settings or get_settings()

    onnx_path = export_pytorch_to_onnx(
        checkpoint_path=settings.MODEL_CHECKPOINT_PATH,
        onnx_output_path=settings.ONNX_MODEL_PATH,
        input_dim=settings.FEATURE_DIM,
        hidden_dim=settings.HIDDEN_DIM,
        latent_dim=settings.LATENT_DIM,
    )

    quantized_path = quantize_to_int8(
        onnx_input_path=onnx_path,
        quantized_output_path=settings.ONNX_QUANTIZED_PATH,
    )

    benchmark_sessions(
        onnx_path=onnx_path,
        quantized_path=quantized_path,
        input_dim=settings.FEATURE_DIM,
    )

    return quantized_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Export and Quantize Autoencoder to ONNX INT8")
    parser.parse_args()
    run_export_and_quantization()


if __name__ == "__main__":
    main()
