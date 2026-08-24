"""
=============================================================================
Generate Clean, Plain-Language Pitch Diagrams (Zero Clipping, 100% Human Words)
=============================================================================
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = "outputs/ppt_diagrams"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Slide 2: Plain-Language Solution Flow
# -------------------------------------------------------------
def make_diagram_slide2():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    # Card 1: Free Input
    rect1 = patches.FancyBboxPatch((0.03, 0.15), 0.28, 0.72, boxstyle="round,pad=0.02", ec="#38bdf8", fc="#1e293b", lw=2)
    ax.add_patch(rect1)
    ax.text(0.17, 0.75, "FREE SATELLITE DATA", color="#38bdf8", weight="bold", fontsize=10.5, ha="center")
    ax.text(0.17, 0.62, "10m Sentinel-2 Images\n+ ISRO Height Maps", color="white", weight="bold", fontsize=9.5, ha="center")
    ax.text(0.17, 0.35, "• Updated every 5 days\n• Free for everyone to use\n• Cost: ₹0", color="#94a3b8", fontsize=9, ha="center")

    # Arrow 1
    ax.annotate("", xy=(0.35, 0.51), xytext=(0.31, 0.51), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    # Card 2: Our Software
    rect2 = patches.FancyBboxPatch((0.36, 0.12), 0.28, 0.78, boxstyle="round,pad=0.02", ec="#818cf8", fc="#1e1b4b", lw=2.5)
    ax.add_patch(rect2)
    ax.text(0.50, 0.78, "OUR SOFTWARE TOOL", color="#a5b4fc", weight="bold", fontsize=11, ha="center")
    ax.text(0.50, 0.64, "BharatSRM-Net", color="#38bdf8", weight="bold", fontsize=10, ha="center")
    ax.text(0.50, 0.36, "• Makes images 4x sharper\n• Keeps original colors accurate\n• Highlights clear vs blurry zones\n• Cleans up cloud haze", color="#cbd5e1", fontsize=8.5, ha="center")

    # Arrow 2
    ax.annotate("", xy=(0.68, 0.51), xytext=(0.64, 0.51), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    # Card 3: Practical Results
    rect3 = patches.FancyBboxPatch((0.69, 0.10), 0.28, 0.82, boxstyle="round,pad=0.02", ec="#4ade80", fc="#064e3b", lw=2)
    ax.add_patch(rect3)
    ax.text(0.83, 0.80, "PRACTICAL RESULTS", color="#4ade80", weight="bold", fontsize=10.5, ha="center")
    ax.text(0.83, 0.66, "Sharp 2.5m Details", color="white", weight="bold", fontsize=9.5, ha="center")
    ax.text(0.83, 0.36, "• See narrow village roads\n• Trace farm plot boundaries\n• Identify water & buildings\n• Saves government money", color="#cbd5e1", fontsize=8.5, ha="center")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/diagram_slide2.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[OK] Generated diagram_slide2.png")

# -------------------------------------------------------------
# 2. Slide 3: 3-Step Work Process
# -------------------------------------------------------------
def make_diagram_slide3():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    stages = [
        ("STEP 1: GET THE DATA", "#0284c7", "#0c4a6e", ["Fetch free 10m satellite images", "Add ISRO ground elevation data", "Filter out heavy clouds automatically", "Stream into software seamlessly"]),
        ("STEP 2: ENHANCE THE IMAGE", "#6366f1", "#312e81", ["Fill in missing details from height maps", "Sharpen edges and narrow tracks", "Keep natural surface colors exact", "Check confidence for every pixel"]),
        ("STEP 3: EXTRACT USEFUL MAPS", "#059669", "#064e3b", ["Output sharp 2.5m map layers", "Show confidence heatmap for safety", "Trace village roads automatically", "Mark farms, forests, and water"]),
    ]

    for i, (title, ec, fc, bullets) in enumerate(stages):
        x = 0.03 + i * 0.33
        rect = patches.FancyBboxPatch((x, 0.15), 0.30, 0.72, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=2)
        ax.add_patch(rect)
        ax.text(x + 0.15, 0.76, title, color="white", weight="bold", fontsize=10, ha="center")
        
        y_text = 0.60
        for b in bullets:
            ax.text(x + 0.02, y_text, f"✔ {b}", color="#e2e8f0", fontsize=8.5, va="top")
            y_text -= 0.125

        if i < 2:
            ax.annotate("", xy=(x + 0.325, 0.51), xytext=(x + 0.295, 0.51), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/diagram_slide3.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[OK] Generated diagram_slide3.png")

# -------------------------------------------------------------
# 3. Slide 4: Real-World Use & Deployment
# -------------------------------------------------------------
def make_diagram_slide4():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    boxes = [
        ("DATA SOURCES", "#0284c7", "#0c4a6e", ["European Space Agency Portal", "ISRO Bhuvan Open Data", "100% Free Public Sources"]),
        ("HOW IT RUNS", "#8b5cf6", "#4c1d95", ["Runs in web browser or desktop", "Works completely offline / secure", "Takes under 2 seconds per area"]),
        ("WHO USES IT", "#10b981", "#064e3b", ["College students & researchers", "Rural development officers (PMGSY)", "Defense and disaster relief teams"]),
    ]

    for i, (title, ec, fc, bullets) in enumerate(boxes):
        x = 0.03 + i * 0.33
        rect = patches.FancyBboxPatch((x, 0.16), 0.30, 0.70, boxstyle="round,pad=0.02", ec=ec, fc=fc, lw=2)
        ax.add_patch(rect)
        ax.text(x + 0.15, 0.75, title, color="white", weight="bold", fontsize=10, ha="center")
        
        y_text = 0.58
        for b in bullets:
            ax.text(x + 0.02, y_text, f"✔ {b}", color="#f1f5f9", fontsize=8.5, va="top")
            y_text -= 0.13

        if i < 2:
            ax.annotate("", xy=(x + 0.325, 0.51), xytext=(x + 0.295, 0.51), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/diagram_slide4.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[OK] Generated diagram_slide4.png")

# -------------------------------------------------------------
# 4. Slide 5: Clean Comparison Table (100% In Bounds, No Clipping)
# -------------------------------------------------------------
def make_diagram_slide5():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.axis("off")

    headers = ["Feature", "Commercial Satellites", "Basic Image Upscalers", "BharatSRM-Net (Our Tool)"]
    x_positions = [0.02, 0.26, 0.50, 0.74]
    widths = [0.23, 0.23, 0.23, 0.24]

    for j, (h, x, w) in enumerate(zip(headers, x_positions, widths)):
        header_color = "#38bdf8" if j == 3 else "#94a3b8"
        bg = "#1e1b4b" if j == 3 else "#1e293b"
        rect = patches.FancyBboxPatch((x, 0.78), w, 0.14, boxstyle="round,pad=0.01", ec=header_color, fc=bg, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, 0.84, h, color="white", weight="bold", fontsize=8.5, ha="center")

    rows = [
        ("Cost to Use", "Over ₹12 Lakhs per area", "Free (but blurry)", "₹0 (Uses free open data)"),
        ("Image Sharpness", "Sharp (2.5m)", "Blurry (10m stretched)", "Sharp (2.5m enhanced)"),
        ("Error Checking", "None provided", "Guesses without checking", "Shows where errors may be"),
        ("Automatic Mapping", "No (just raw image)", "No", "Finds roads, farms & water"),
        ("Security & Privacy", "Foreign vendor dependent", "Not secure", "100% Local and Private"),
    ]

    for i, (f, c1, c2, c3) in enumerate(rows):
        y = 0.63 - i * 0.125
        for j, (val, x, w) in enumerate(zip([f, c1, c2, c3], x_positions, widths)):
            color = "#4ade80" if j == 3 else ("#f87171" if j == 2 and i == 2 else "#cbd5e1")
            weight = "bold" if j == 3 else "normal"
            rect = patches.FancyBboxPatch((x, y), w, 0.105, boxstyle="round,pad=0.01", ec="#334155", fc="#0f172a" if j != 3 else "#064e3b", lw=1)
            ax.add_patch(rect)
            ax.text(x + w/2, y + 0.042, val, color=color, weight=weight, fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/diagram_slide5.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[OK] Generated diagram_slide5.png (100% within margins, zero clipping)")

if __name__ == "__main__":
    make_diagram_slide2()
    make_diagram_slide3()
    make_diagram_slide4()
    make_diagram_slide5()
    print("[SUCCESS] All human-language diagrams regenerated with zero clipping!")
