"""
=============================================================================
BharatSRM-Net v4: ONNX Export & INT8 Quantization Benchmark
=============================================================================
Section 7.2 & 7.3 Specification:
  - Exports PyTorch model to ONNX with dynamic spatial axes.
  - Applies INT8 dynamic range quantization for fast CPU deployment on Intel Core i7.
  - Benchmarks latency, memory reduction, and numerical fidelity.
=============================================================================
"""

import os

import torch


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str = "checkpoints/bharatsrm_net_v4.onnx",
    input_size: tuple[int, int] = (256, 256),
) -> str:
    """Exports BharatSRMNetV4 to ONNX with dynamic spatial dimensions."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    dummy_lr = torch.randn(1, 10, input_size[0], input_size[1])
    dummy_mask = torch.ones(1, 1, input_size[0], input_size[1])
    dummy_dem = torch.zeros(1, 2, input_size[0], input_size[1])

    # Simplified wrapper for standard ONNX export
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, core_model: torch.nn.Module):
            super().__init__()
            self.core = core_model

        def forward(self, lr: torch.Tensor, mask: torch.Tensor, dem: torch.Tensor):
            out = self.core(lr, mask, dem)
            return out["sr_image"], out["variance"]

    wrapper = ONNXWrapper(model)

    torch.onnx.export(
        wrapper,
        (dummy_lr, dummy_mask, dummy_dem),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["lr_input", "validity_mask", "context_dem"],
        output_names=["sr_image", "variance"],
        dynamic_axes={
            "lr_input": {0: "batch", 2: "height", 3: "width"},
            "validity_mask": {0: "batch", 2: "height", 3: "width"},
            "context_dem": {0: "batch", 2: "height", 3: "width"},
            "sr_image": {0: "batch", 2: "hr_height", 3: "hr_width"},
            "variance": {0: "batch", 2: "hr_height", 3: "hr_width"},
        },
    )
    return output_path


def quantize_pytorch_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    """Uses torch.quantization.quantize_dynamic() for INT8 dynamic quantization of Conv2d and Linear layers."""
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Conv2d, torch.nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model


def quantize_onnx_dynamic(input_onnx_path: str, output_onnx_path: str) -> str:
    """Uses onnxruntime.quantization.quantize_dynamic() for ONNX INT8 quantization."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        input_onnx_path,
        output_onnx_path,
        weight_type=QuantType.QUInt8,
    )
    return output_onnx_path


def benchmark_quantization(
    fp32_model: torch.nn.Module,
    int8_model: torch.nn.Module,
    test_input: tuple[torch.Tensor, ...],
    target: torch.Tensor,
) -> dict[str, float]:
    """Compares FP32 vs INT8 model size, latency per tile, and PSNR degradation after quantization."""
    import tempfile
    import time
    from src.evaluation.metrics import calculate_psnr
    
    # Model size
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(fp32_model.state_dict(), f.name)
        fp32_size = os.path.getsize(f.name) / (1024 * 1024)
    os.remove(f.name)
        
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(int8_model.state_dict(), f.name)
        int8_size = os.path.getsize(f.name) / (1024 * 1024)
    os.remove(f.name)
        
    # Latency
    fp32_model.eval()
    int8_model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            fp32_model(*test_input)
            int8_model(*test_input)
            
    # Measure FP32
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            fp32_out = fp32_model(*test_input)
    fp32_latency = (time.perf_counter() - start) / 5.0
    
    # Measure INT8
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            int8_out = int8_model(*test_input)
    int8_latency = (time.perf_counter() - start) / 5.0
    
    if isinstance(fp32_out, dict):
        fp32_pred = fp32_out["sr_image"]
        int8_pred = int8_out["sr_image"]
    else:
        fp32_pred = fp32_out
        int8_pred = int8_out
        
    fp32_psnr = calculate_psnr(fp32_pred, target)["PSNR_mean"]
    int8_psnr = calculate_psnr(int8_pred, target)["PSNR_mean"]
    
    return {
        "fp32_size_mb": fp32_size,
        "int8_size_mb": int8_size,
        "size_reduction_factor": fp32_size / int8_size if int8_size > 0 else 0,
        "fp32_latency_s": fp32_latency,
        "int8_latency_s": int8_latency,
        "speedup": fp32_latency / int8_latency if int8_latency > 0 else 0,
        "fp32_psnr": fp32_psnr,
        "int8_psnr": int8_psnr,
        "psnr_degradation": fp32_psnr - int8_psnr
    }
