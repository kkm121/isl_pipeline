# BharatSRM-Net v4 — Master Agentic Engineering Architecture
**Physically-Consistent, Uncertainty-Aware Super-Resolution Framework for Indian Satellite Imagery**
*SIH 2026 — Problem Statement ID 26142 (NTRO - National Technical Research Organisation)*

---

## 🏛️ 4-Plane Engineering Architecture

1. **Control Plane** — High-level reasoning, physical invariance enforcement, subagent orchestration.
2. **Integration Plane** — Fast MCP protocols for dataset streaming, linter execution, and GPU dispatch.
3. **Execution Plane** — Sealed containers (`--network=none --read-only`) and remote Kaggle dual Tesla T4 GPUs.
4. **Verification Plane** — Deterministic gate enforcement with strict mathematical and quantitative verification.

---

## 👥 Specialized Multi-Agent System (9 Specialized Roles)

| # | Agent Role | Primary Model | Reasoning Level | Core Responsibility | Tool Scope |
|:---|:---|:---|:---|:---|:---|
| 1 | **Principal Architect** | **Claude Opus / Pro** | **High** | End-to-end framework orchestration, NTRO PS compliance, physical consistency validation | `advance_pipeline`, `git`, read tools |
| 2 | **Adversarial Critic** | **Claude Opus / Pro** | **High** | Audits sensor MTF/PSF cycle consistency, flags data leakage, rejects unearned claims | Read-only tools, diff inspector |
| 3 | **Independent Reviewer** | **Claude Opus / Pro** | **High** | Clean-session diff verification, mathematical correctness of losses, gate sign-off | Read-only tools, AST analyzer |
| 4 | **EO Data Engineer** | **Pro / Flash** | **Medium-High** | Sentinel-2 L2A 10m ingestion, COG windowed streaming, AKAZE/RANSAC co-registration, CartoDEM | `data/`, rasterio, geospatial tools |
| 5 | **Deep Learning Architect** | **Pro / Flash** | **Medium-High** | PartialConv2d, Dilated Residual Blocks, AC-FEM, PixelShuffle s=4, Heteroscedastic Uncertainty Head | `src/models/`, PyTorch |
| 6 | **Loss & Physics Specialist** | **Pro / Flash** | **Medium-High** | 5-term composite loss ($\mathcal{L}_{rec}, \mathcal{L}_{spec}, \mathcal{L}_{degrade}, \mathcal{L}_{struct}, \mathcal{L}_{conf}$), loss warmup | `src/training/`, loss math |
| 7 | **Downstream RS Specialist** | **Pro / Flash** | **Medium** | PMGSY rural road extraction, LULC boundary disaggregation, bi-temporal disaster damage head | `src/models/downstream_heads.py` |
| 8 | **EO Metrics Profiler** | **Flash** | **Medium** | PSNR, SSIM, SAM, ERGAS, RMSE, Cloud-stratified evaluation, Uncertainty reliability calibration | `src/evaluation/`, metrics |
| 9 | **GIS Web & Edge Engineer** | **Flash** | **Medium** | Tiled 2D Hanning window CPU engine, ONNX/INT8 quantization, Interactive Web GIS Studio | `src/web/`, `src/inference/` |

---

## 🔒 Strict Scientific Mandates

1. **Physical & Spectral Invariance**: Reconstructed high-resolution imagery MUST satisfy cycle-consistency degradation constraints against observed Sentinel-2 inputs ($\hat{x}_{LR} = \text{Downsample}_s(\text{Blur}_{PSF}(\hat{y}_{SR}))$).
2. **Uncertainty Accounting**: High-frequency textures inferred by deep networks must be quantified via per-pixel, per-band variance $\sigma^2 = \exp(s)$.
3. **No Target Leakage**: Input context features (CartoDEM) must never encode downstream labels (e.g. road vectors).
4. **Cloud-Stratified Evaluation**: Performance must be reported across Clear, Cloud-Edge, and Cloud-Shadow strata to prevent clear pixels from masking cloud-corrupted degradation.
5. **Deterministic Testing**: 100% test pass rate required in `tests/` before committing changes.
