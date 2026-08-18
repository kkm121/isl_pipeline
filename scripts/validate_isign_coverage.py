#!/usr/bin/env python3
"""
iSign Dataset Strategy: Validate Kaggle keypoints against official iSign CSV.

This script implements the recommended approach:
  1. Stream iSign_v1.1.csv from HuggingFace (requires HF token + accepted terms)
  2. Download isign-mediapipe-keypoints train.csv from Kaggle (requires kaggle.json)
  3. Compare UIDs to determine how complete the Kaggle derivative is
  4. Print a coverage report + recommendation

Usage:
    python scripts/validate_isign_coverage.py

Prerequisites:
    A) HuggingFace token (accepted iSign research-use agreement):
       https://huggingface.co/datasets/Exploration-Lab/iSign
       Save token to: C:/Users/muthu/.huggingface/token
       OR set env: HF_TOKEN=hf_xxx...

    B) Kaggle credentials (if checking Kaggle dataset):
       https://www.kaggle.com/settings/account -> Create New Token
       Save to: ~/.kaggle/kaggle.json

Why this matters:
    The Kaggle isign-mediapipe-keypoints dataset may be an incomplete
    derivative of the official 118K+ segment iSign release. Before using
    it in research, we verify what fraction of the official UIDs it covers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _load_hf_token() -> str | None:
    """Load HuggingFace token from standard locations."""
    for loc in [
        os.environ.get("HF_TOKEN"),
        os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        str(Path.home() / ".huggingface" / "token"),
        str(Path.home() / ".cache" / "huggingface" / "token"),
    ]:
        if loc and os.path.isfile(loc):
            return Path(loc).read_text().strip()
        if loc and len(loc) > 8 and loc.startswith("hf_"):
            return loc
    return None


def step1_fetch_official_isign_csv(token: str) -> set[str]:
    """Stream iSign_v1.1.csv from HuggingFace and return set of official UIDs."""
    import io
    import urllib.request

    print("\n[Step 1] Fetching official iSign_v1.1.csv from HuggingFace...")
    url = (
        "https://huggingface.co/datasets/Exploration-Lab/iSign"
        "/resolve/main/iSign_v1.1.csv"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "isl-pipeline/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()

    import csv

    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    uid_col = None
    uids: set[str] = set()
    for row in reader:
        if uid_col is None:
            # Discover the UID column (typically 'uid' or 'id')
            for col in row:
                if "uid" in col.lower() or col.lower() == "id":
                    uid_col = col
                    break
            if uid_col is None:
                # Fall back: first column
                uid_col = next(iter(row))
            print(f"  UID column identified: '{uid_col}'")

        uid = row[uid_col].strip()
        if uid:
            uids.add(uid)

    # Extract unique video_ids (prefix before the last '-N')
    video_ids = {uid.rsplit("-", 1)[0] for uid in uids}
    print(f"  Official UIDs (segments): {len(uids):,}")
    print(f"  Official unique video_ids: {len(video_ids):,}")
    return uids


def step2_fetch_kaggle_ids() -> set[str]:
    """Load isign-mediapipe-keypoints train.csv and return its UID/sequence_id set."""
    print("\n[Step 2] Loading Kaggle isign-mediapipe-keypoints metadata...")

    # Try local cache first
    local_paths = [
        Path("data/isign_kaggle/train.csv"),
        Path("data/isign_kaggle/metadata.csv"),
    ]
    for p in local_paths:
        if p.exists():
            import pandas as pd

            df = pd.read_csv(p)
            print(f"  Loaded from local cache: {p} ({len(df):,} rows)")
            print(f"  Columns: {list(df.columns)}")
            # Try to extract sequence IDs
            id_col = next(
                (c for c in df.columns if "uid" in c.lower() or "id" in c.lower()), None
            )
            if id_col:
                return set(df[id_col].astype(str).tolist())
            return set(df.iloc[:, 0].astype(str).tolist())

    # Try Kaggle API
    kg_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kg_json.exists():
        print("  No ~/.kaggle/kaggle.json found.")
        print(
            "  Create one at: https://www.kaggle.com/settings/account -> Create New Token"
        )
        return set()

    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            "unicreator/isign-mediapipe-keypoints",
            "--path",
            "data/isign_kaggle/",
            "--unzip",
            "-q",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Kaggle download failed: {result.stderr[:300]}")
        return set()

    for p in local_paths:
        if p.exists():
            import pandas as pd

            df = pd.read_csv(p)
            id_col = next(
                (c for c in df.columns if "uid" in c.lower() or "id" in c.lower()), None
            )
            if id_col:
                return set(df[id_col].astype(str).tolist())

    return set()


def step3_compute_coverage(official_uids: set[str], kaggle_ids: set[str]) -> None:
    """Compare and report coverage."""
    print("\n[Step 3] Coverage Analysis")
    print("=" * 60)

    if not official_uids:
        print("  Cannot compute: official UIDs not loaded.")
        return
    if not kaggle_ids:
        print("  Cannot compute: Kaggle IDs not loaded.")
        print("  Run with Kaggle credentials configured to get Kaggle coverage.")
        return

    # Normalize both sets (strip whitespace, lowercase)
    o = {uid.strip().lower() for uid in official_uids}
    k = {uid.strip().lower() for uid in kaggle_ids}

    in_both = o & k
    only_official = o - k
    only_kaggle = k - o

    coverage_pct = (len(in_both) / len(o) * 100) if o else 0.0

    print(f"  Official iSign segments:     {len(o):>8,}")
    print(f"  Kaggle dataset IDs:          {len(k):>8,}")
    print(f"  Matched (in both):           {len(in_both):>8,}")
    print(f"  Only in official (missing):  {len(only_official):>8,}")
    print(f"  Only in Kaggle (extra):      {len(only_kaggle):>8,}")
    print(f"\n  >>> COVERAGE: {coverage_pct:.1f}% of official iSign <<<")

    if coverage_pct >= 95:
        verdict = "EXCELLENT — Kaggle dataset is essentially complete. Safe for research."
    elif coverage_pct >= 80:
        verdict = "GOOD — Kaggle dataset covers most of iSign. Note the missing segments."
    elif coverage_pct >= 50:
        verdict = "PARTIAL — Kaggle covers about half. Use official videos for full results."
    else:
        verdict = "POOR — Kaggle is a heavily reduced subset. Use official iSign videos."

    print(f"\n  Verdict: {verdict}")

    # Save report
    report = {
        "official_segment_count": len(o),
        "kaggle_id_count": len(k),
        "matched": len(in_both),
        "missing_from_kaggle": len(only_official),
        "extra_in_kaggle": len(only_kaggle),
        "coverage_percent": round(coverage_pct, 2),
        "verdict": verdict,
    }
    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/isign_kaggle_coverage.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  Report saved to: metrics/isign_kaggle_coverage.json")


def main() -> None:
    print("=" * 60)
    print("iSign Dataset Coverage Validator")
    print("=" * 60)

    token = _load_hf_token()
    if not token:
        print("\n[BLOCKED] No HuggingFace token found.")
        print("\nTo get the official iSign CSV:")
        print("  1. Accept research-use agreement at:")
        print("     https://huggingface.co/datasets/Exploration-Lab/iSign")
        print("  2. Create a Read token at:")
        print("     https://huggingface.co/settings/tokens")
        print("  3. Save it to C:/Users/muthu/.huggingface/token")
        print("  4. Re-run this script.")
        print("\nFor now, proceeding with Kaggle dataset only (no official comparison).")
        kaggle_ids = step2_fetch_kaggle_ids()
        print(f"\nKaggle dataset IDs loaded: {len(kaggle_ids):,}")
        print("Cannot compute official coverage without HF token.")
        sys.exit(0)

    official_uids = step1_fetch_official_isign_csv(token)
    kaggle_ids = step2_fetch_kaggle_ids()
    step3_compute_coverage(official_uids, kaggle_ids)

    print("\n[Done] Run the Kaggle T4 training kernel once coverage >= 70%:")
    print("  kaggle kernels push -p kaggle/submit_include/")


if __name__ == "__main__":
    main()
