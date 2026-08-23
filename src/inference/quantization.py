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
