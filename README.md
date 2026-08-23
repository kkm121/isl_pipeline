# BharatSRM-Net v4
**A Physically-Consistent, Uncertainty-Aware Deep Learning Super-Resolution Framework for Indian Medium-Resolution Satellite Imagery**

*SIH 2026 — Problem Statement 26142 (NTRO - National Technical Research Organisation)*

---

## 🛰️ Problem Statement & Objective
Medium-resolution satellite imagery (Sentinel-2 at 10m Ground Sampling Distance) offers broad spatial coverage and frequent revisit intervals, but lacks fine details required for detecting small rural roads, individual agricultural parcels, and localized disaster damage.

**BharatSRM-Net v4** super-resolves 10m Sentinel-2 L2A imagery to **<4m (nominal 2.5m, scale factor $s=4$)** multi-band RGBN (Red, Green, Blue, NIR) products while rigorously enforcing **sensor MTF/PSF cycle consistency** and predicting per-pixel **heteroscedastic uncertainty maps**.

---

## 🏛️ Core Architecture Components

```
┌──────────────────────────────────────────────────────────────────┐
│              Sentinel-2 L2A (10 Bands, 10m GSD)                  │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
               Cloud & Validity Mask (S2cloudless + QA60)
                                │
    ┌───────────────────────────┴───────────────────────────┐
    ▼                                                       ▼
Masked Multispectral Encoder                       Context Encoder (Optional)
Partial Conv2D + Dilated Blocks (r ∈ {1,2,4,8})    CartoDEM 30m Elevation & Slope
+ Window Attention                                 (Topographic illumination prior)
    │                                                       │
    └───────────────────────────┬───────────────────────────┘
                                ▼
                AC-FEM: Mask-Gated Feature Fusion
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
       Reconstruction Head               Uncertainty Head
   (PixelShuffle s=4, 4-Band RGBN)      (Per-pixel log-variance)
                │                               │
                ▼                               ▼
     2.5m Super-Resolved Image          Uncertainty Map (σ²)
```

---

## 🧮 5-Term Composite Loss Function
$$\mathcal{L}_{core} = \lambda_1 \mathcal{L}_{rec} + \lambda_2 \mathcal{L}_{spec} + \lambda_3 \mathcal{L}_{degrade} + \lambda_4 \mathcal{L}_{struct} + \lambda_5 \mathcal{L}_{conf}$$

1. **Reconstruction Loss ($\mathcal{L}_{rec}$)**: Charbonnier smooth L1 loss ($\epsilon = 10^{-3}$).
2. **Spectral Consistency Loss ($\mathcal{L}_{spec}$)**: Spectral Angle Mapper (SAM) angular deviation.
3. **Degradation Consistency Loss ($\mathcal{L}_{degrade}$)**: Sensor MTF/PSF cycle consistency ($\hat{x}_{LR} = \text{Downsample}_s(\text{Blur}_{PSF}(\hat{y}_{SR}))$).
4. **Structural Loss ($\mathcal{L}_{struct}$)**: SSIM + Sobel gradient edge preservation.
5. **Heteroscedastic Uncertainty Loss ($\mathcal{L}_{conf}$)**: Jointly trains the per-pixel log-variance $s_i = \log \sigma_i^2$.

---

## 🗺️ Indian Stratified Terrain Dataset (ISTD-SR) AOIs
1. **Indo-Gangetic Plains (Punjab/Haryana)**: Smallholder agriculture (<0.5 ha parcels), Kharif/Rabi rotation.
2. **Peri-Urban Corridors (Bengaluru, Delhi-NCR)**: Rapid built-up expansion, informal settlements.
3. **Western Ghats**: Steep canopy, persistent monsoon cloud cover.
4. **Semi-Arid Scrubland (Rajasthan)**: Sparse vegetation, unpaved rural tracks, high bare-soil reflectance.

---

## ⚡ Deployment & Web GIS Studio
* **Tiled CPU Inference Engine**: Overlapping $256 \times 256$ patches blended with **2D Hanning window**.
* **Interactive GIS Web Studio**: FastAPI backend with Leaflet.js / Mapbox GL split-view before/after slider, uncertainty heatmaps, and 1-click GeoTIFF / GeoJSON vector export.
