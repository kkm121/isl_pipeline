"""
=============================================================================
Generate High-Quality Business & Architecture Diagrams for SIH Presentation
=============================================================================
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

OUT_DIR = "outputs/ppt_diagrams"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Slide 2 Diagram: Business Value Proposition Flow
# -------------------------------------------------------------
def make_diagram_slide2():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    # Card 1: Input
    rect1 = patches.FancyBboxPatch((0.05, 0.2), 0.26, 0.65, boxstyle="round,pad=0.03", ec="#38bdf8", fc="#1e293b", lw=2)
    ax.add_patch(rect1)
    ax.text(0.18, 0.72, "FREE INPUT DATA", color="#38bdf8", weight="bold", fontsize=11, ha="center")
    ax.text(0.18, 0.58, "10m Sentinel-2 L2A\n+ CartoDEM Elevation", color="white", weight="bold", fontsize=10, ha="center")
    ax.text(0.18, 0.35, "• Global 5-Day Revisit\n• 100% Free / Open API\n• ₹0 Cost to Government", color="#94a3b8", fontsize=9, ha="center")

    # Arrow 1
    ax.annotate("", xy=(0.38, 0.52), xytext=(0.32, 0.52), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    # Card 2: AI Engine
    rect2 = patches.FancyBboxPatch((0.39, 0.15), 0.28, 0.75, boxstyle="round,pad=0.03", ec="#818cf8", fc="#1e1b4b", lw=2.5)
    ax.add_patch(rect2)
    ax.text(0.53, 0.76, "BHARATSRM-NET v4", color="#a5b4fc", weight="bold", fontsize=12, ha="center")
    ax.text(0.53, 0.62, "Physics-Consistent AI Core", color="#38bdf8", weight="bold", fontsize=10, ha="center")
    ax.text(0.53, 0.38, "• PartialConv Cloud Masking\n• AC-FEM Context Fusion\n• Zero-DC Residual Learning\n• Calibrated Uncertainty", color="#cbd5e1", fontsize=9, ha="center")

    # Arrow 2
    ax.annotate("", xy=(0.74, 0.52), xytext=(0.68, 0.52), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    # Card 3: Output Products
    rect3 = patches.FancyBboxPatch((0.75, 0.12), 0.22, 0.82, boxstyle="round,pad=0.03", ec="#4ade80", fc="#064e3b", lw=2)
    ax.add_patch(rect3)
    ax.text(0.86, 0.82, "COMMERCIAL VALUE", color="#4ade80", weight="bold", fontsize=11, ha="center")
    ax.text(0.86, 0.68, "2.5m Sub-Meter Output", color="white", weight="bold", fontsize=10, ha="center")
    ax.text(0.86, 0.38, "• 16x Pixel Density\n• PMGSY Rural Roads\n• ISRO 5-Class LULC\n• Defense Confidence Map\n• Saves ₹100s of Crores", color="#cbd5e1", fontsize=8.5, ha="center")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/diagram_slide2.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[OK] Generated diagram_slide2.png")

# -------------------------------------------------------------
# 2. Slide 3 Diagram: 3-Stage End-to-End Architecture
# -------------------------------------------------------------
def make_diagram_slide3():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    stages = [
        ("STAGE 1: INGESTION", "#0284c7", "#0c4a6e", ["10-Band Sentinel-2 (B2-B12)", "CartoDEM Slope & Aspect", "S2cloudless / QA60 Mask", "Windowed COG Streaming"]),
        ("STAGE 2: AI CORE", "#6366f1", "#312e81", ["PartialConv2d Cloud Inpainting", "AC-FEM Cross-Attention", "Dilated Residual Blocks (r=1,2,4,8)", "ICNR PixelShuffle (s=4)"]),
        ("STAGE 3: MULTI-TASK DELIVERY", "#059669", "#064e3b", ["2.5m True-Color Super-Res", "Calibrated Uncertainty (σ²)", "PMGSY Rural Road Vectors", "ISRO 5-Class LULC Map"]),
    ]

    for i, (title, ec, fc, bullets) in enumerate(stages):
        x = 0.05 + i * 0.32
        rect = patches.FancyBboxPatch((x, 0.15), 0.27, 0.75, boxstyle="round,pad=0.03", ec=ec, fc=fc, lw=2)
        ax.add_patch(rect)
        ax.text(x + 0.135, 0.78, title, color="white", weight="bold", fontsize=10, ha="center")
        
        y_text = 0.62
        for b in bullets:
            ax.text(x + 0.02, y_text, f"► {b}", color="#e2e8f0", fontsize=8.5, va="top")
            y_text -= 0.13

        if i < 2:
            ax.annotate("", xy=(x + 0.315, 0.52), xytext=(x + 0.275, 0.52), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/diagram_slide3.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[OK] Generated diagram_slide3.png")

# -------------------------------------------------------------
# 3. Slide 4 Diagram: Feasibility & Deployment Pipeline
# -------------------------------------------------------------
def make_diagram_slide4():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    boxes = [
        ("OPEN SATELLITE APIS", "#0284c7", "#0c4a6e", ["Copernicus Data Space", "ISRO Bhuvan CartoDEM", "Direct COG Windowing"]),
        ("SEALED CONTAINER", "#8b5cf6", "#4c1d95", ["Docker Sandboxed", "PyTorch AMP FP16", "INT8 Edge Quantization"]),
        ("DEPLOYMENT TARGETS", "#10b981", "#064e3b", ["Air-Gapped Defense HQ", "Cloud Web GIS Studio", "Edge Field Drones / Tablets"]),
    ]

    for i, (title, ec, fc, bullets) in enumerate(boxes):
        x = 0.05 + i * 0.32
        rect = patches.FancyBboxPatch((x, 0.18), 0.27, 0.70, boxstyle="round,pad=0.03", ec=ec, fc=fc, lw=2)
        ax.add_patch(rect)
        ax.text(x + 0.135, 0.76, title, color="white", weight="bold", fontsize=10.5, ha="center")
        
        y_text = 0.60
        for b in bullets:
            ax.text(x + 0.02, y_text, f"✔ {b}", color="#f1f5f9", fontsize=9, va="top")
            y_text -= 0.14

        if i < 2:
            ax.annotate("", xy=(x + 0.315, 0.52), xytext=(x + 0.275, 0.52), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/diagram_slide4.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[OK] Generated diagram_slide4.png")

# -------------------------------------------------------------
# 4. Slide 5 Diagram: ROI & Competitive Advantage Matrix
# -------------------------------------------------------------
def make_diagram_slide5():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    # Table Header
    headers = ["Feature / Capability", "Foreign Commercial (SPOT 6/7)", "Standard Bicubic / GANs", "BharatSRM-Net v4 (Our Solution)"]
    x_positions = [0.03, 0.30, 0.56, 0.81]
    widths = [0.25, 0.24, 0.23, 0.26]

    for j, (h, x, w) in enumerate(zip(headers, x_positions, widths)):
        header_color = "#38bdf8" if j == 3 else "#94a3b8"
        bg = "#1e1b4b" if j == 3 else "#1e293b"
        rect = patches.FancyBboxPatch((x, 0.78), w - 0.02, 0.15, boxstyle="round,pad=0.01", ec=header_color, fc=bg, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + (w-0.02)/2, 0.84, h, color="white", weight="bold", fontsize=8.5, ha="center")

    rows = [
        ("Cost per 1,000 km²", "₹12,00,000+ ($15/km²)", "₹0 (Free Interpolation)", "₹0 (100% Free Open Data)"),
        ("Spatial Resolution", "1.5m - 2.5m Commercial", "10m Smeared (Blurry)", "2.5m Super-Resolved (<4m)"),
        ("Anti-Hallucination", "None (Physical Optical)", "High Hallucination Risk", "Calibrated Uncertainty (σ²)"),
        ("Multi-Task Analytics", "None (Raw Image Only)", "None", "PMGSY Roads + ISRO LULC"),
        ("Sovereign Security", "Foreign Satellite Dependency", "N/A", "100% Indigenous & Air-Gapped"),
    ]

    for i, (f, c1, c2, c3) in enumerate(rows):
        y = 0.63 - i * 0.13
        for j, (val, x, w) in enumerate(zip([f, c1, c2, c3], x_positions, widths)):
            color = "#4ade80" if j == 3 else ("#f87171" if j == 2 and i == 2 else "#cbd5e1")
            weight = "bold" if j == 3 else "normal"
            rect = patches.FancyBboxPatch((x, y), w - 0.02, 0.11, boxstyle="round,pad=0.01", ec="#334155", fc="#0f172a" if j != 3 else "#064e3b", lw=1)
            ax.add_patch(rect)
            ax.text(x + (w-0.02)/2, y + 0.045, val, color=color, weight=weight, fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/diagram_slide5.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[OK] Generated diagram_slide5.png")

if __name__ == "__main__":
    make_diagram_slide2()
    make_diagram_slide3()
    make_diagram_slide4()
    make_diagram_slide5()
    print("\n[SUCCESS] All 4 slide diagrams generated successfully!")
