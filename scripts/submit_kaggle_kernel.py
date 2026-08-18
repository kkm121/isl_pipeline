"""
Kaggle Kernel Submitter & Remote GPU Status Monitor.
"""

import os
import shutil
import sys
from pathlib import Path

# Fix path to load official kaggle package
sys.path = [p for p in sys.path if not os.path.exists(os.path.join(p, "kaggle", "__init__.py")) or "site-packages" in p]
from kaggle.api.kaggle_api_extended import KaggleApi

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
submit_dir = PROJECT_ROOT / "kaggle" / "submit_include"
submit_dir.mkdir(parents=True, exist_ok=True)

# Copy kernel files
src_kernel = PROJECT_ROOT / "kaggle" / "templates" / "isl_include_train_kernel.py"
src_meta = PROJECT_ROOT / "kaggle" / "templates" / "kernel-metadata.json"

shutil.copy2(src_kernel, submit_dir / "isl_include_train_kernel.py")
shutil.copy2(src_meta, submit_dir / "kernel-metadata.json")

print(f"Prepared submission folder at: {submit_dir}")

api = KaggleApi()
api.authenticate()

print("Submitting kernel to Kaggle GPU accelerator...")
try:
    api.kernels_push(str(submit_dir))
    print("SUCCESS: Kernel submitted to Kaggle!")
    kernel_slug = "kkm121121/isl-include-t4-training"
    print(f"Kernel URL: https://www.kaggle.com/code/{kernel_slug}")

    # Check initial status
    status = api.kernels_status(kernel_slug)
    print(f"Current Remote Status: {status}")
except Exception as e:
    print(f"Kaggle push failed/returned: {e}")
