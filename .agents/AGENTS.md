# BharatSRM-Net v4 — Master 12-Agent Specialized Swarm
**Physically-Consistent, Uncertainty-Aware Super-Resolution Framework for Indian Satellite Imagery**
*SIH 2026 — Problem Statement ID 26142 (NTRO - National Technical Research Organisation)*

---

## 🏛️ 4-Plane Engineering Architecture

1. **Control Plane** — High-level reasoning, physical invariance enforcement, multi-agent task dispatch.
2. **Integration Plane** — Fast MCP protocols for dataset streaming, linter execution, and GPU dispatch.
3. **Execution Plane** — Sealed containers (`--network=none --read-only`) and remote Kaggle dual Tesla T4 GPUs.
4. **Verification Plane** — Deterministic gate enforcement with strict mathematical, physical, and quantitative proofs.

---

## 👥 The 12 Specialized Subagent Roles

| # | Agent Name | Primary Model Tier | Reasoning Level | Core Domain & Specialization | Tool Scope |
|:---|:---|:---|:---|:---|:---|
| 1 | **`principal_architect`** | **Claude Opus / Pro** | **High** | End-to-end framework orchestration, NTRO PS compliance, roadmap & gate execution | `advance_pipeline`, `git`, read tools |
| 2 | **`physics_sensor_auditor`** | **Claude Opus / Pro** | **High** | Sensor MTF/PSF point spread function modeling, cycle-consistency $\mathcal{L}_{degrade}$, optical reflectance limits | Read-only tools, physics math |
| 3 | **`adversarial_critic`** | **Claude Opus / Pro** | **High** | Target leakage detection (DEM/road vectors), honest disjoint splits, anti-fabrication enforcement | Read-only tools, diff inspector |
| 4 | **`independent_reviewer`** | **Claude Opus / Pro** | **High** | Clean-session diff verification, loss sign derivations (e.g. $\exp(-s)$ in NLL), gate sign-off | Read-only tools, AST analyzer |
| 5 | **`eo_data_streaming_engineer`** | **Pro** | **Medium-High** | Sentinel-2 L2A 10m ingestion, COG windowed streaming, WorldStrat, PlanetScope, Bhuvan, CartoDEM | `data/`, rasterio, COG tools |
| 6 | **`cross_sensor_registration_engineer`** | **Pro** | **Medium-High** | AKAZE/SIFT keypoints, RANSAC homography, sub-pixel phase correlation, per-band radiometric histogram matching | `src/data/registration.py` |
| 7 | **`deep_learning_architect`** | **Pro** | **Medium-High** | PartialConv2d, Dilated Residual Blocks ($r \in \{1,2,4,8\}$), AC-FEM cross-attention, PixelShuffle $s=4$, Uncertainty Head | `src/models/`, PyTorch |
| 8 | **`loss_optimization_specialist`** | **Pro** | **Medium-High** | 5-term composite loss ($\mathcal{L}_{rec}, \mathcal{L}_{spec}, \mathcal{L}_{degrade}, \mathcal{L}_{struct}, \mathcal{L}_{conf}$), 3-epoch warmup, AMP FP16, AdamW | `src/training/`, loss math |
| 9 | **`downstream_rs_specialist`** | **Pro / Flash** | **Medium** | PMGSY rural road extraction (SRSNet skeletonization to GeoJSON), LULC urban boundary disaggregation, disaster change head | `src/models/downstream_heads.py` |
| 10 | **`eo_metrics_profiler`** | **Flash** | **Medium** | PSNR, SSIM, SAM ($^\circ$), ERGAS, RMSE, cloud-stratified evaluation (Clear, Cloud-Edge, Cloud-Shadow), Leave-One-Region-Out CV | `src/evaluation/`, metrics |
| 11 | **`uncertainty_calibration_engineer`** | **Flash** | **Medium** | Binned reliability curves ($\sigma^2$ vs empirical MSE), spread-skill correlation, per-band uncertainty validation | `src/evaluation/uncertainty_calibration.py` |
| 12 | **`gis_web_edge_developer`** | **Flash** | **Medium** | Tiled 2D Hanning window CPU engine, ONNX/INT8 dynamic quantization, FastAPI + Mapbox/Leaflet Split-View interactive GIS Studio | `src/web/`, `src/inference/` |

---

## 🔒 The 6 Implementation Gates (Before Model Deployment)

1. **Gate 1 (WorldStrat Modality Confirmation)**: Verify exact SPOT 6/7 1.5m pansharpened RGBN product tier before pretraining.
2. **Gate 2 (Indian Pair Validation)**: Confirm real Sentinel-2 $\leftrightarrow$ Bhuvan HR pairs across the 4 AOIs before full ISTD-SR dataset assembly.
3. **Gate 3 (Sensor PSF Kernel Sensitivity Pilot)**: Validate Gaussian MTF vs Sinc-windowed approximation consistency for $\mathcal{L}_{degrade}$.
4. **Gate 4 (Trained Control Baseline)**: Benchmark against Bicubic and off-the-shelf EDSR/SRResNet on the same pairs before claiming AC-FEM/PConv superiority.
5. **Gate 5 (Cloud-Stratified Evaluation)**: Clear, Cloud-Edge, and Cloud-Shadow metrics wired into evaluation before final benchmark claims.
6. **Gate 6 (Uncertainty Equation Derivation)**: Verify exact negative log-likelihood $\frac{1}{N}\sum_i [\exp(-s_i)\|y_i - \hat{y}_i\|^2 + s_i]$ sign implementation.
