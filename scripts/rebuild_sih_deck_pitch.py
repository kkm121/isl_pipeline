"""
=============================================================================
BharatSRM-Net: Plain-Language Human Pitch Deck Generator (SIH 2026)
=============================================================================
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

PPT_FILES = [
    r"C:\Users\muthu\Downloads\SIH2026-BharatSRM-Net-SmartEducation.pptx",
    r"C:\Users\muthu\Downloads\SIH2026-IDEA-Presentation-Format (2).pptx",
    r"C:\Users\muthu\Downloads\SIH2026-BharatSRM-Net-PitchDeck.pptx",
    r"C:\Users\muthu\Downloads\SIH2026-IDEA-Presentation-ChaosToCode.pptx",
]
DIAG_DIR = "outputs/ppt_diagrams"

def style_bullet(p, title, desc, font_size=10.5, space_after=5, bold_color=RGBColor(15, 23, 42), text_color=RGBColor(51, 65, 85)):
    p.text = ""
    p.space_after = Pt(space_after)
    p.font.name = "Segoe UI"
    
    r1 = p.add_run()
    r1.text = f"• {title}: "
    r1.font.bold = True
    r1.font.size = Pt(font_size)
    r1.font.name = "Segoe UI"
    r1.font.color.rgb = bold_color
    
    r2 = p.add_run()
    r2.text = desc
    r2.font.bold = False
    r2.font.size = Pt(font_size)
    r2.font.name = "Segoe UI"
    r2.font.color.rgb = text_color

def build_human_pitch(out_path):
    template_path = r"C:\Users\muthu\Downloads\SIH2026-IDEA-Presentation-Format (2).pptx"
    if not os.path.exists(template_path):
        return
    try:
        prs = pptx.Presentation(template_path)
        print(f"\nBuilding Plain-Language Pitch for: {out_path}...")

        # -------------------------------------------------------------
        # SLIDE 1: Title Slide (Clear & Human)
        # -------------------------------------------------------------
        s1 = prs.slides[0]
        for shape in s1.shapes:
            if shape.has_text_frame:
                if "SMART INDIA HACKATHON" in shape.text:
                    shape.top = Inches(0.35)
                    shape.left = Inches(0.5)
                    shape.width = Inches(10.0)
                    shape.height = Inches(0.7)
                    shape.text_frame.paragraphs[0].font.size = Pt(22)
                elif "TITLE PAGE" in shape.text or "BharatSRM" in shape.text:
                    shape.top = Inches(0.95)
                    shape.left = Inches(0.5)
                    shape.width = Inches(9.0)
                    shape.height = Inches(0.9)
                    shape.text_frame.clear()
                    p1 = shape.text_frame.paragraphs[0]
                    p1.text = "BharatSRM-Net"
                    p1.font.size = Pt(26)
                    p1.font.bold = True
                    p1.font.color.rgb = RGBColor(15, 23, 42)
                    p2 = shape.text_frame.add_paragraph()
                    p2.text = "Software that turns free satellite images into sharp, high-detail maps for education, farming, and defense"
                    p2.font.size = Pt(11.5)
                    p2.font.color.rgb = RGBColor(71, 85, 105)
                elif "Problem Statement ID" in shape.text:
                    shape.top = Inches(1.95)
                    shape.left = Inches(0.5)
                    shape.width = Inches(6.8)
                    shape.height = Inches(4.5)
                    tf = shape.text_frame
                    tf.clear()
                    fields = [
                        ("Problem Statement ID", "26142"),
                        ("Problem Statement Title", "Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries"),
                        ("Theme", "Smart Education (Space Tech & Geography Learning)"),
                        ("Category", "Software"),
                        ("Organization / Ministry", "National Technical Research Organisation (NTRO)"),
                        ("Team Name", "ChaosToCode"),
                        ("Team ID", "[To be filled by team]"),
                    ]
                    for idx, (k, v) in enumerate(fields):
                        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                        style_bullet(p, k, v, font_size=11, space_after=3, bold_color=RGBColor(2, 132, 199))

        # -------------------------------------------------------------
        # SLIDE 2: What We Built & Why It Matters
        # -------------------------------------------------------------
        s2 = prs.slides[1]
        for s in list(s2.shapes):
            if s.name.startswith("Added_Diagram"):
                s2.shapes._spTree.remove(s._element)

        for shape in s2.shapes:
            if shape.has_text_frame:
                if "PROPOSED SOLUTION" in shape.text or "IDEA TITLE" in shape.text:
                    shape.top = Inches(0.2)
                    shape.left = Inches(0.5)
                    shape.width = Inches(10.0)
                    shape.height = Inches(0.8)
                    shape.text_frame.paragraphs[0].text = "WHAT WE BUILT & WHY IT MATTERS"
                elif "The Problem" in shape.text or "Proposed Solution" in shape.text or "The Core" in shape.text or "The Cost" in shape.text:
                    shape.top = Inches(1.25)
                    shape.left = Inches(0.5)
                    shape.width = Inches(5.3)
                    shape.height = Inches(5.2)
                    tf = shape.text_frame
                    tf.clear()
                    bullets = [
                        ("The Problem", "High-resolution satellite images are very expensive. Free satellites give blurry pictures where village roads and farm edges are hard to see."),
                        ("What We Built", "A software tool that takes free 10m satellite pictures and makes them 4x sharper (2.5m resolution) for ₹0 extra cost."),
                        ("Error Checking You Can Trust", "Standard AI tools often guess and invent fake details. Our tool gives an error heatmap showing where the image is 100% reliable."),
                        ("Real Practical Uses", "Finds rural village roads, marks crop fields for farmer insurance, tracks floods, and helps college students study satellite maps."),
                    ]
                    for idx, (k, v) in enumerate(bullets):
                        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                        style_bullet(p, k, v, font_size=10.5, space_after=6)
                elif "ChaosToCode" in shape.text or "Your Team Name" in shape.text:
                    shape.text_frame.paragraphs[0].text = "ChaosToCode"

        diag2 = f"{DIAG_DIR}/diagram_slide2.png"
        if os.path.exists(diag2):
            pic = s2.shapes.add_picture(diag2, Inches(6.0), Inches(1.45), width=Inches(6.8), height=Inches(4.8))
            pic.name = "Added_Diagram_2"

        # -------------------------------------------------------------
        # SLIDE 3: How the Tool Works
        # -------------------------------------------------------------
        s3 = prs.slides[2]
        for s in list(s3.shapes):
            if s.name.startswith("Added_Diagram"):
                s3.shapes._spTree.remove(s._element)

        for shape in s3.shapes:
            if shape.has_text_frame:
                if "TECHNICAL APPROACH" in shape.text or "PRODUCT WORKFLOW" in shape.text:
                    shape.top = Inches(0.2)
                    shape.left = Inches(0.5)
                    shape.width = Inches(10.0)
                    shape.height = Inches(0.7)
                    shape.text_frame.paragraphs[0].text = "HOW THE SOFTWARE WORKS"
                elif "End-to-End" in shape.text or "Technologies to be used" in shape.text or "Multi-Modal" in shape.text:
                    shape.top = Inches(0.95)
                    shape.left = Inches(0.5)
                    shape.width = Inches(12.3)
                    shape.height = Inches(1.0)
                    tf = shape.text_frame
                    tf.clear()
                    bullets = [
                        ("Simple 3-Step Process", "Loads free satellite pictures + ISRO height data -> Sharpens details using deep learning -> Produces sharp maps and road vectors in seconds."),
                        ("Handles Huge Areas", "Smoothly processes entire districts and states without any tiling seams, borders, or distortion."),
                    ]
                    for idx, (k, v) in enumerate(bullets):
                        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                        style_bullet(p, k, v, font_size=10.5, space_after=2)
                elif "ChaosToCode" in shape.text or "Your Team Name" in shape.text:
                    shape.text_frame.paragraphs[0].text = "ChaosToCode"

        diag3 = f"{DIAG_DIR}/diagram_slide3.png"
        if os.path.exists(diag3):
            pic = s3.shapes.add_picture(diag3, Inches(0.5), Inches(2.1), width=Inches(12.3), height=Inches(4.6))
            pic.name = "Added_Diagram_3"

        # -------------------------------------------------------------
        # SLIDE 4: Real-World Use & Easy Deployment
        # -------------------------------------------------------------
        s4 = prs.slides[3]
        for s in list(s4.shapes):
            if s.name.startswith("Added_Diagram"):
                s4.shapes._spTree.remove(s._element)

        for shape in s4.shapes:
            if shape.has_text_frame:
                if "FEASIBILITY" in shape.text:
                    shape.top = Inches(0.2)
                    shape.left = Inches(0.5)
                    shape.width = Inches(10.0)
                    shape.height = Inches(0.8)
                    shape.text_frame.paragraphs[0].text = "HOW IT RUNS & WHO CAN USE IT"
                elif "Verified Working" in shape.text or "Analysis of the feasibility" in shape.text:
                    shape.top = Inches(1.25)
                    shape.left = Inches(0.5)
                    shape.width = Inches(5.3)
                    shape.height = Inches(5.2)
                    tf = shape.text_frame
                    tf.clear()
                    bullets = [
                        ("Ready Right Now", "A fully working web tool tested across agricultural plains, mountains, cities, and deserts with 1-second response times."),
                        ("Zero Extra Cost", "Uses only free, open public satellite data from ISRO and Europe. No subscriptions or hidden fees."),
                        ("Works Everywhere", "Runs in standard web browsers for students, and works completely offline on secure defense computers."),
                        ("Handles Clouds", "Automatically removes cloud haze so ground details come through clearly."),
                    ]
                    for idx, (k, v) in enumerate(bullets):
                        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                        style_bullet(p, k, v, font_size=10.5, space_after=6)
                elif "ChaosToCode" in shape.text or "Your Team Name" in shape.text:
                    shape.text_frame.paragraphs[0].text = "ChaosToCode"

        diag4 = f"{DIAG_DIR}/diagram_slide4.png"
        if os.path.exists(diag4):
            pic = s4.shapes.add_picture(diag4, Inches(6.0), Inches(1.45), width=Inches(6.8), height=Inches(4.8))
            pic.name = "Added_Diagram_4"

        # -------------------------------------------------------------
        # SLIDE 5: Practical Value & Smart Education (No Clipping Table!)
        # -------------------------------------------------------------
        s5 = prs.slides[4]
        for s in list(s5.shapes):
            if s.name.startswith("Added_Diagram"):
                s5.shapes._spTree.remove(s._element)

        for shape in s5.shapes:
            if shape.has_text_frame:
                if "IMPACT AND BENEFITS" in shape.text or "BUSINESS IMPACT" in shape.text or "NATIONAL IMPACT" in shape.text or "PRACTICAL VALUE" in shape.text:
                    shape.top = Inches(0.2)
                    shape.left = Inches(0.5)
                    shape.width = Inches(10.0)
                    shape.height = Inches(0.8)
                    shape.text_frame.paragraphs[0].text = "PRACTICAL VALUE FOR INDIA & EDUCATION"
                elif "Smart Education" in shape.text or "Massive Cost Savings" in shape.text or "Potential impact" in shape.text or "Helps Students" in shape.text:
                    shape.top = Inches(1.25)
                    shape.left = Inches(0.5)
                    shape.width = Inches(4.9)
                    shape.height = Inches(5.2)
                    tf = shape.text_frame
                    tf.clear()
                    bullets = [
                        ("Helps Students & Colleges", "Lets college students and researchers study high-detail satellite maps in labs without paying expensive commercial fees."),
                        ("Saves Government Money", "Replaces costly foreign satellite image purchases, saving crores for public projects."),
                        ("Builds Rural Roads (PMGSY)", "Automatically maps unpaved roads and village paths to help rural road planning."),
                        ("Farming & Disaster Relief", "Measures farm plot sizes for crop insurance and maps flooded areas during monsoon emergencies."),
                    ]
                    for idx, (k, v) in enumerate(bullets):
                        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                        style_bullet(p, k, v, font_size=10.5, space_after=6)
                elif "ChaosToCode" in shape.text or "Your Team Name" in shape.text:
                    shape.text_frame.paragraphs[0].text = "ChaosToCode"

        diag5 = f"{DIAG_DIR}/diagram_slide5.png"
        if os.path.exists(diag5):
            pic = s5.shapes.add_picture(diag5, Inches(5.6), Inches(1.4), width=Inches(7.2), height=Inches(5.0))
            pic.name = "Added_Diagram_5"

        # -------------------------------------------------------------
        # SLIDE 6: Research & Data Sources
        # -------------------------------------------------------------
        s6 = prs.slides[5]
        for shape in s6.shapes:
            if shape.has_text_frame:
                if "RESEARCH" in shape.text or "VALIDATION" in shape.text:
                    shape.top = Inches(0.2)
                    shape.left = Inches(0.5)
                    shape.width = Inches(10.0)
                    shape.height = Inches(0.8)
                    shape.text_frame.paragraphs[0].text = "RESEARCH & DATA SOURCES"
                elif "Rigorous Academic" in shape.text or "Trained on Real" in shape.text or "Rigorous Dataset" in shape.text or "Details / Links" in shape.text:
                    shape.top = Inches(1.25)
                    shape.left = Inches(0.6)
                    shape.width = Inches(12.0)
                    shape.height = Inches(5.2)
                    tf = shape.text_frame
                    tf.clear()
                    bullets = [
                        ("Trained on Real Satellite Pairs", "Pre-trained on 3,900+ real satellite pairs across Indian and global regions."),
                        ("Follows ISRO Standards", "Uses standard ISRO land-use categories (Water, Forest, Farmland, Buildings, Barren)."),
                        ("Proven Scientific Foundation", "Built on peer-reviewed research in computer vision and satellite image processing."),
                        ("Open Public Data", "Uses European Space Agency (Copernicus) and ISRO Bhuvan open datasets."),
                    ]
                    for idx, (k, v) in enumerate(bullets):
                        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                        style_bullet(p, k, v, font_size=11.5, space_after=8)
                elif "ChaosToCode" in shape.text or "Your Team Name" in shape.text:
                    shape.text_frame.paragraphs[0].text = "ChaosToCode"

        prs.save(out_path)
        print(f"[SUCCESS] Plain-language deck saved to: {out_path}")
    except Exception as e:
        print(f"[NOTE] Could not update {out_path}: {e}")

if __name__ == "__main__":
    for p in PPT_FILES:
        build_human_pitch(p)
