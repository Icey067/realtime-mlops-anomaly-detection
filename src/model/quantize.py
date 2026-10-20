"""Export the autoencoder to ONNX, quantize it to INT8, and benchmark CPU latency."""

from __future__ import annotations

import argparse
import inspect
import logging
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.config import Settings, get_settings
from src.logging_config import configure_logging
from src.model.autoencoder import Autoencoder

logger = logging.getLogger(__name__)


def export_pytorch_to_onnx(
    checkpoint_path: Path,
    onnx_output_path: Path,
    input_dim: int = 30,
    hidden_dim: int = 16,
    latent_dim: int = 8,
) -> Path:
    """Export a PyTorch checkpoint to ONNX with a dynamic batch axis."""
    logger.info("Loading PyTorch model from %s", checkpoint_path)
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting ONNX (opset 14) to %s", onnx_output_path)
    export_kwargs = {
        "export_params": True,
        "opset_version": 14,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, dummy_input, str(onnx_output_path), **export_kwargs)

    onnx_model = onnx.load(str(onnx_output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph validated")
    return onnx_output_path


def quantize_to_int8(
    onnx_input_path: Path,
    quantized_output_path: Path,
) -> Path:
    """Quantize ONNX weights to INT8."""
    logger.info("Quantizing %s to INT8", onnx_input_path)
    quantized_output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(onnx_input_path),
        model_output=str(quantized_output_path),
        weight_type=QuantType.QInt8,
    )

    raw_size_kb = onnx_input_path.stat().st_size / 1024
    quant_size_kb = quantized_output_path.stat().st_size / 1024
    reduction = (1.0 - quant_size_kb / max(raw_size_kb, 0.001)) * 100

    logger.info("Quantized size: %.2f KB -> %.2f KB (%.1f%%)", raw_size_kb, quant_size_kb, reduction)
    return quantized_output_path


def benchmark_sessions(
    onnx_path: Path,
    quantized_path: Path,
    input_dim: int = 30,
    iterations: int = 500,
) -> tuple[float, float]:
    """Measure per-sample CPU latency for FP32 and INT8 sessions."""
    logger.info("Benchmarking %d iterations", iterations)
    dummy_data = np.random.randn(1, input_dim).astype(np.float32)

    sess_fp32 = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])

    input_name_fp32 = sess_fp32.get_inputs()[0].name
    input_name_int8 = sess_int8.get_inputs()[0].name

    for _ in range(50):
        sess_fp32.run(None, {input_name_fp32: dummy_data})
        sess_int8.run(None, {input_name_int8: dummy_data})

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = sess_fp32.run(None, {input_name_fp32: dummy_data})
    fp32_latency_ms = ((time.perf_counter() - t0) / iterations) * 1000

    t1 = time.perf_counter()
    for _ in range(iterations):
        sess_int8.run(None, {input_name_int8: dummy_data})
    int8_latency_ms = ((time.perf_counter() - t1) / iterations) * 1000

    logger.info("Latency: FP32 %.3f ms/sample | INT8 %.3f ms/sample", fp32_latency_ms, int8_latency_ms)
    return fp32_latency_ms, int8_latency_ms


def run_export_and_quantization(settings: Settings | None = None) -> Path:
    """Run export, quantization, and benchmarking."""
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
    configure_logging()
    parser = argparse.ArgumentParser(description="Export and quantize the autoencoder to ONNX INT8")
    parser.parse_args()
    run_export_and_quantization()


if __name__ == "__main__":
    main()
