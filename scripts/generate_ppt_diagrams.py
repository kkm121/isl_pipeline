"""
=============================================================================
Generate Executive Business Pitch Diagrams for SIH Presentation
=============================================================================
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = "outputs/ppt_diagrams"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Slide 2: The Core Business Value Proposition
def make_diagram_slide2():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    # Card 1: Free Input
    rect1 = patches.FancyBboxPatch((0.05, 0.2), 0.26, 0.65, boxstyle="round,pad=0.03", ec="#38bdf8", fc="#1e293b", lw=2)
    ax.add_patch(rect1)
    ax.text(0.18, 0.72, "FREE RAW SATELLITE DATA", color="#38bdf8", weight="bold", fontsize=10, ha="center")
    ax.text(0.18, 0.58, "10m Sentinel-2 (Copernicus)\n+ ISRO Elevation Maps", color="white", weight="bold", fontsize=9.5, ha="center")
    ax.text(0.18, 0.35, "• Global 5-Day Revisit\n• 100% Free Public Data\n• ₹0 Satellite Acquisition Cost", color="#94a3b8", fontsize=8.5, ha="center")

    # Arrow 1
    ax.annotate("", xy=(0.38, 0.52), xytext=(0.32, 0.52), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    # Card 2: AI Engine
    rect2 = patches.FancyBboxPatch((0.39, 0.15), 0.28, 0.75, boxstyle="round,pad=0.03", ec="#818cf8", fc="#1e1b4b", lw=2.5)
    ax.add_patch(rect2)
    ax.text(0.53, 0.76, "BHARATSRM-NET ENGINE", color="#a5b4fc", weight="bold", fontsize=11, ha="center")
    ax.text(0.53, 0.62, "AI-Powered Spatial Enhancement", color="#38bdf8", weight="bold", fontsize=9.5, ha="center")
    ax.text(0.53, 0.38, "• 4x Super-Resolution Core\n• True-Color Physics Engine\n• Built-in AI Confidence Meter\n• Cloud & Shadow Recovery", color="#cbd5e1", fontsize=8.5, ha="center")

    # Arrow 2
    ax.annotate("", xy=(0.74, 0.52), xytext=(0.68, 0.52), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    # Card 3: Actionable Value
    rect3 = patches.FancyBboxPatch((0.75, 0.12), 0.22, 0.82, boxstyle="round,pad=0.03", ec="#4ade80", fc="#064e3b", lw=2)
    ax.add_patch(rect3)
    ax.text(0.86, 0.82, "COMMERCIAL VALUE", color="#4ade80", weight="bold", fontsize=10.5, ha="center")
    ax.text(0.86, 0.68, "2.5m Actionable Output", color="white", weight="bold", fontsize=9.5, ha="center")
    ax.text(0.86, 0.38, "• 16x Sharper Ground Detail\n• Village Road Mapping\n• Farm Boundary Tracking\n• Defense Security Insights\n• Saves ₹100s of Crores", color="#cbd5e1", fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/diagram_slide2.png", bbox_inches="tight", dpi=300)
    plt.close()

# 2. Slide 3: End-to-End System Workflow
def make_diagram_slide3():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    stages = [
        ("1. INGESTION", "#0284c7", "#0c4a6e", ["Free 10m Satellite Imagery", "ISRO Elevation & Slope Maps", "Automatic Cloud & Haze Filter", "Instant Seamless Streaming"]),
        ("2. AI ENHANCEMENT", "#6366f1", "#312e81", ["Multi-Spectral Deep Neural Net", "Terrain & Elevation Context Fusion", "Sub-Pixel Detail Reconstruction", "True-Color Physics Guarantee"]),
        ("3. ACTIONABLE INSIGHTS", "#059669", "#064e3b", ["2.5m Commercial-Grade Image", "AI Trust & Confidence Map", "PMGSY Village Road Extraction", "ISRO Land-Use Classification"]),
    ]

    for i, (title, ec, fc, bullets) in enumerate(stages):
        x = 0.05 + i * 0.32
        rect = patches.FancyBboxPatch((x, 0.15), 0.27, 0.75, boxstyle="round,pad=0.03", ec=ec, fc=fc, lw=2)
        ax.add_patch(rect)
        ax.text(x + 0.135, 0.78, title, color="white", weight="bold", fontsize=10, ha="center")
        
        y_text = 0.62
        for b in bullets:
            ax.text(x + 0.02, y_text, f"✔ {b}", color="#e2e8f0", fontsize=8.5, va="top")
            y_text -= 0.13

        if i < 2:
            ax.annotate("", xy=(x + 0.315, 0.52), xytext=(x + 0.275, 0.52), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/diagram_slide3.png", bbox_inches="tight", dpi=300)
    plt.close()

# 3. Slide 4: Deployment & Operations
def make_diagram_slide4():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    boxes = [
        ("DATA SOURCING", "#0284c7", "#0c4a6e", ["Copernicus Open Access Hub", "ISRO Bhuvan Portal", "100% Free & Open-Source"]),
        ("SECURE AI PLATFORM", "#8b5cf6", "#4c1d95", ["High-Speed GPU/CPU Engine", "Offline Air-Gapped Ready", "Sub-Second Processing Time"]),
        ("END USERS & MINISTRIES", "#10b981", "#064e3b", ["NTRO & Defense Command", "Rural Development (PMGSY)", "Agriculture & Disaster Teams"]),
    ]

    for i, (title, ec, fc, bullets) in enumerate(boxes):
        x = 0.05 + i * 0.32
        rect = patches.FancyBboxPatch((x, 0.18), 0.27, 0.70, boxstyle="round,pad=0.03", ec=ec, fc=fc, lw=2)
        ax.add_patch(rect)
        ax.text(x + 0.135, 0.76, title, color="white", weight="bold", fontsize=10, ha="center")
        
        y_text = 0.60
        for b in bullets:
            ax.text(x + 0.02, y_text, f"✔ {b}", color="#f1f5f9", fontsize=8.5, va="top")
            y_text -= 0.14

        if i < 2:
            ax.annotate("", xy=(x + 0.315, 0.52), xytext=(x + 0.275, 0.52), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/diagram_slide4.png", bbox_inches="tight", dpi=300)
    plt.close()

# 4. Slide 5: Market ROI & Competitive Advantage
def make_diagram_slide5():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    headers = ["Evaluation Metric", "Commercial Satellites", "Standard AI Tools", "BharatSRM-Net (Our Pitch)"]
    x_positions = [0.03, 0.30, 0.56, 0.81]
    widths = [0.25, 0.24, 0.23, 0.26]

    for j, (h, x, w) in enumerate(zip(headers, x_positions, widths)):
        header_color = "#38bdf8" if j == 3 else "#94a3b8"
        bg = "#1e1b4b" if j == 3 else "#1e293b"
        rect = patches.FancyBboxPatch((x, 0.78), w - 0.02, 0.15, boxstyle="round,pad=0.01", ec=header_color, fc=bg, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + (w-0.02)/2, 0.84, h, color="white", weight="bold", fontsize=8.5, ha="center")

    rows = [
        ("Procurement Cost", "₹12,00,000+ per 1,000 km²", "Free (Poor Quality)", "₹0 (100% Free Open Data)"),
        ("Ground Detail Quality", "2.5m Commercial Grade", "10m Blurry / Unclear", "2.5m Super-Resolved (<4m)"),
        ("AI Trust & Confidence", "No Confidence Scoring", "High AI Guesswork Risk", "Built-in Trust Map (σ²)"),
        ("Built-in Analytics", "None (Raw Images Only)", "None", "Roads + Farm Boundaries"),
        ("Sovereign Security", "Foreign Vendor Dependent", "Non-Compliant", "100% Indigenous & Secure"),
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

if __name__ == "__main__":
    make_diagram_slide2()
    make_diagram_slide3()
    make_diagram_slide4()
    make_diagram_slide5()
    print("[SUCCESS] Business pitch diagrams refreshed!")
