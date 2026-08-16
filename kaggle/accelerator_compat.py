from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class AcceleratorProfile:
    name: str
    vram_gb: float
    compute_capability: float
    supported_cuda_versions: List[str]
    supported_pytorch_versions: List[str]
    kaggle_api_name: str
    notes: str = ''

ACCELERATOR_MATRIX = {
    'T4': AcceleratorProfile(
        name='NVIDIA Tesla T4',
        vram_gb=16.0,
        compute_capability=7.5,
        supported_cuda_versions=['11.8', '12.1', '12.2', '12.4'],
        supported_pytorch_versions=['2.0+'],
        kaggle_api_name='gpu',
        notes='Default Kaggle GPU. Turing architecture. Full compatibility with modern PyTorch.'
    ),
    'P100': AcceleratorProfile(
        name='NVIDIA Tesla P100',
        vram_gb=16.0,
        compute_capability=6.0,
        supported_cuda_versions=['11.8'],
        supported_pytorch_versions=['1.13', '2.0'],
        kaggle_api_name='gpu',
        notes='Pascal architecture. WARNING: Not compatible with default Kaggle Docker image for PyTorch 2.1+ (Pascal kernels not included). Use T4 instead for modern PyTorch.'
    ),
    'TPU_V3': AcceleratorProfile(
        name='Google TPU v3-8',
        vram_gb=128.0,
        compute_capability=0.0,
        supported_cuda_versions=[],
        supported_pytorch_versions=['2.0+ (via torch_xla)'],
        kaggle_api_name='tpu',
        notes='Requires torch_xla. Different programming model from CUDA.'
    ),
}

def check_compatibility(accelerator: str, pytorch_version: str, cuda_version: Optional[str] = None) -> Dict:
    profile = ACCELERATOR_MATRIX.get(accelerator.upper())
    if not profile:
        return {"compatible": False, "warnings": [f"Unknown accelerator {accelerator}"], "recommendation": "Use T4"}

    warnings = []
    compatible = True

    # Check PyTorch version roughly
    pt_major = pytorch_version.split('.')[0]
    if accelerator.upper() == 'P100' and float(pt_major + '.' + pytorch_version.split('.')[1]) >= 2.1:
        warnings.append("P100 is not compatible with PyTorch 2.1+ in Kaggle docker environments due to missing Pascal kernels.")
        compatible = False

    if cuda_version and cuda_version not in profile.supported_cuda_versions and profile.compute_capability > 0:
        warnings.append(f"CUDA version {cuda_version} may not be natively supported on {accelerator}.")

    return {
        "compatible": compatible,
        "warnings": warnings,
        "recommendation": "T4" if not compatible else accelerator.upper()
    }

def recommend_accelerator(model_vram_estimate_gb: float, pytorch_version: str) -> str:
    # Estimate VRAM
    if model_vram_estimate_gb > 16:
        # Might need TPU or multi-GPU (multi-T4 is 2x16 = 32)
        # Kaggle gives 2xT4 often
        return 'T4' # returning T4 representing 2xT4 since it's default
    
    # Check if modern pytorch
    pt_major = float(pytorch_version.split('.')[0] + '.' + pytorch_version.split('.')[1])
    if pt_major >= 2.1:
        return 'T4'
        
    return 'T4'
