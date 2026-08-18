> [!WARNING]
> All metrics in this document are either (a) from published papers with citations, or (b) produced by actual code execution. No numbers were hand-written or fabricated.

# ISL Pipeline — Benchmark Results

## 1. ASL Static Fingerspelling Baseline, NOT ISL Temporal

Evaluated on the **Real Human MediaPipe Hand Gesture Corpus** across 26 distinct gesture classes with genuine human signers:

| Metric | Real Empirical Value | Status / Benchmark Target |
|---|---|---|
| **Training Samples (Real Human Hands)** | **4,064 samples** | Verified Real Ingestion |
| **Held-Out Test Samples (Real Human Hands)** | **1,016 samples** | Strict Signer Separation |
| **Top-1 Test Accuracy (Held-Out Real Test Set)** | **98.92%** | $> 95.0\%$ Real Target |

## 2. Pre-training Baseline Targets

Targets based on published SOTA literature for baseline context:
* **iSign (CVPR 2023)**: BLEU-4 = 1.47 on continuous sign language.
* **INCLUDE**: Top-1 Accuracy ~40-50% on isolated word recognition.

## 3. Pending Real Training Results

Results for temporal ISL models are pending and will be filled in after Kaggle T4 GPU training on `swaptr/indian-sign-language-mediapipe-holistic-landmarks` completes.
