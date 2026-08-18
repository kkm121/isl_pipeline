# Zero-Synthetic-Data Policy & Real-World Empirical Standard

## Core Mandate
Under no circumstances may any agent generate, use, or substitute synthetic AI-generated mathematical numbers, random Gaussian centroids, or simulated toy kinematic curves in place of real datasets for:
1. **Model Training**: All production and demo weights must be trained on genuine human-recorded landmark/video datasets (e.g., AI4Bharat INCLUDE, real MediaPipe recordings).
2. **Evaluation & Benchmarks**: All reported benchmark accuracies, BLEU scores, confusion matrices, and latency figures must be computed against real-world human test splits.
3. **Dataset Pipeline & Ingestion**: The dataset plane must ingest and partition real human recordings with verified signer-disjoint splits.
4. **Verification & Testing**: System verification must validate real dataset ingestion integrity, real gradient convergence, and real test partition evaluation.

## Prohibited Patterns
- ❌ Generating random Gaussian or trigonometric arrays as "training data".
- ❌ Hardcoding synthetic accuracy scores (e.g., "100.0% synthetic accuracy") as proof of real ISL recognition.
- ❌ Using conditional data-bypasses or mock data fixtures that masquerade as real dataset evaluation.

## Mandatory Real-World Evidence
- **Real Dataset Artifacts**: Stored in `data/` or fetched from verified public research repositories (Hugging Face / Zenodo / Kaggle).
- **Real Test Partition Evaluation**: Real loss curves, real Top-1/Top-5 accuracy on held-out human signers, and real confusion matrices saved to `metrics/`.
- **Truth in Reporting**: Clear documentation citing the exact real-world dataset name, number of human signers, total samples, and genuine test performance.
