"""
=============================================================================
Rebuild SIH 2026 Pitch Deck: Executive Business Style with Flow Diagrams
=============================================================================
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

PPT_PATH = r"C:\Users\muthu\Downloads\SIH2026-IDEA-Presentation-Format (2).pptx"
DIAG_DIR = r"outputs/ppt_diagrams"

def style_bullet(p, title, desc, font_size=11, space_after=4, bold_color=RGBColor(15, 23, 42), text_color=RGBColor(51, 65, 85)):
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

def rebuild_pitch_deck():
    prs = pptx.Presentation(PPT_PATH)
    print(f"Loaded {len(prs.slides)} slides.")

    # -------------------------------------------------------------
    # SLIDE 1: Title Page
    # -------------------------------------------------------------
    slide1 = prs.slides[0]
    for shape in slide1.shapes:
        if shape.has_text_frame:
            if "TITLE PAGE" in shape.text or "BharatSRM" in shape.text:
                shape.text_frame.clear()
                p = shape.text_frame.paragraphs[0]
                p.text = "BharatSRM-Net v4"
                p.font.size = Pt(28)
                p.font.bold = True
                p.font.name = "Segoe UI"
                p.font.color.rgb = RGBColor(15, 23, 42)
                
                p2 = shape.text_frame.add_paragraph()
                p2.text = "Physically-Consistent, Uncertainty-Aware Super-Resolution Framework for Indian Satellite Imagery"
                p2.font.size = Pt(13)
                p2.font.name = "Segoe UI"
                p2.font.color.rgb = RGBColor(71, 85, 105)
                
            elif "SMART INDIA HACKATHON" in shape.text:
                p = shape.text_frame.paragraphs[0]
                p.text = "SMART INDIA HACKATHON 2026"
                p.font.size = Pt(22)
                p.font.bold = True
                p.font.name = "Segoe UI"
                p.font.color.rgb = RGBColor(2, 132, 199)
                
            elif "Problem Statement ID" in shape.text:
                tf = shape.text_frame
                tf.clear()
                fields = [
                    ("Problem Statement ID", "26142"),
                    ("Problem Statement Title", "Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries"),
                    ("Theme", "Smart Education / Space Technology & Defense"),
                    ("PS Category", "Software"),
                    ("Organization / Ministry", "National Technical Research Organisation (NTRO)"),
                    ("Team Name", "ChaosToCode"),
                    ("Team ID", "[To be filled by team]"),
                ]
                for idx, (k, v) in enumerate(fields):
                    p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                    style_bullet(p, k, v, font_size=12, bold_color=RGBColor(2, 132, 199))

    # -------------------------------------------------------------
    # SLIDE 2: Proposed Solution (Business Pitch + Diagram)
    # -------------------------------------------------------------
    slide2 = prs.slides[1]
    # Remove existing image shapes if any from previous run
    for s in list(slide2.shapes):
        if s.name.startswith("Added_Diagram"):
            slide2.shapes._spTree.remove(s._element)

    for shape in slide2.shapes:
        if shape.has_text_frame:
            if "PROPOSED SOLUTION" in shape.text or "IDEA TITLE" in shape.text:
                shape.text_frame.clear()
                p = shape.text_frame.paragraphs[0]
                p.text = "PROPOSED SOLUTION: BHARATSRM-NET v4"
                p.font.size = Pt(18)
                p.font.bold = True
                p.font.color.rgb = RGBColor(15, 23, 42)
            elif "4x Physical Super-Resolution" in shape.text or "Proposed Solution" in shape.text:
                tf = shape.text_frame
                tf.clear()
                shape.top = Inches(1.2)
                shape.height = Inches(2.2)
                shape.width = Inches(12.5)
                shape.left = Inches(0.4)
                
                bullets = [
                    ("The Problem", "Free 10m Sentinel-2 satellite data is too blurry for narrow village roads, farm parcel boundary tracking, and defense surveillance, while commercial 2.5m data costs crores (> ₹12 Lakh per 1,000 km²)."),
                    ("The Innovation", "BharatSRM-Net v4 transforms free 10m Sentinel-2 data into 2.5m commercial-grade imagery (<4m target requirement) with 16x higher spatial sampling density and zero color drift."),
                    ("Trust & Defense Anti-Hallucination", "Unlike standard black-box AI upscalers, our model outputs a Calibrated Uncertainty Heatmap (σ²) proving to intelligence analysts exactly which pixels are 100% trustworthy."),
                ]
                for idx, (k, v) in enumerate(bullets):
                    p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                    style_bullet(p, k, v, font_size=11)
            elif "ChaosToCode" in shape.text or "Your Team Name" in shape.text:
                shape.text_frame.clear()
                shape.text_frame.paragraphs[0].text = "ChaosToCode"

    # Add Diagram Image on Slide 2
    diag2_path = f"{DIAG_DIR}/diagram_slide2.png"
    if os.path.exists(diag2_path):
        pic = slide2.shapes.add_picture(diag2_path, Inches(0.4), Inches(3.55), width=Inches(12.5), height=Inches(3.2))
        pic.name = "Added_Diagram_2"

    # -------------------------------------------------------------
    # SLIDE 3: Technical Approach (Architecture Flow + Diagram)
    # -------------------------------------------------------------
    slide3 = prs.slides[2]
    for s in list(slide3.shapes):
        if s.name.startswith("Added_Diagram"):
            slide3.shapes._spTree.remove(s._element)

    for shape in slide3.shapes:
        if shape.has_text_frame:
            if "TECHNICAL APPROACH" in shape.text:
                shape.text_frame.clear()
                p = shape.text_frame.paragraphs[0]
                p.text = "TECHNICAL APPROACH & SYSTEM ARCHITECTURE"
                p.font.size = Pt(18)
                p.font.bold = True
                p.font.color.rgb = RGBColor(15, 23, 42)
            elif "4-Plane Swarm" in shape.text or "Technologies to be used" in shape.text:
                tf = shape.text_frame
                tf.clear()
                shape.top = Inches(1.2)
                shape.height = Inches(2.2)
                shape.width = Inches(12.5)
                shape.left = Inches(0.4)
                
                bullets = [
                    ("Multi-Modal Ingestion", "Ingests 10 Sentinel-2 multispectral bands + CartoDEM terrain slope/aspect with S2cloudless mask."),
                    ("Physics-Informed Deep Learning Core", "PartialConv2d (cloud inpainting) + AC-FEM Cross-Attention + Dilated Residual Blocks (r=1,2,4,8) + ICNR PixelShuffle (s=4)."),
                    ("Tiled 2D Hanning Engine & Web Studio", "Seamless processing of regional 10,000x10,000 scenes with zero tile seams and real-time GIS split-screen slider."),
                ]
                for idx, (k, v) in enumerate(bullets):
                    p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                    style_bullet(p, k, v, font_size=11)
            elif "ChaosToCode" in shape.text or "Your Team Name" in shape.text:
                shape.text_frame.clear()
                shape.text_frame.paragraphs[0].text = "ChaosToCode"

    # Add Diagram Image on Slide 3
    diag3_path = f"{DIAG_DIR}/diagram_slide3.png"
    if os.path.exists(diag3_path):
        pic = slide3.shapes.add_picture(diag3_path, Inches(0.4), Inches(3.55), width=Inches(12.5), height=Inches(3.2))
        pic.name = "Added_Diagram_3"

    # -------------------------------------------------------------
    # SLIDE 4: Feasibility & Deployment Pipeline (Diagram)
    # -------------------------------------------------------------
    slide4 = prs.slides[3]
    for s in list(slide4.shapes):
        if s.name.startswith("Added_Diagram"):
            slide4.shapes._spTree.remove(s._element)

    for shape in slide4.shapes:
        if shape.has_text_frame:
            if "FEASIBILITY" in shape.text:
                shape.text_frame.clear()
                p = shape.text_frame.paragraphs[0]
                p.text = "FEASIBILITY, VIABILITY & DEPLOYMENT"
                p.font.size = Pt(18)
                p.font.bold = True
                p.font.color.rgb = RGBColor(15, 23, 42)
            elif "Technical Feasibility" in shape.text or "Analysis of the feasibility" in shape.text:
                tf = shape.text_frame
                tf.clear()
                shape.top = Inches(1.2)
                shape.height = Inches(2.2)
                shape.width = Inches(12.5)
                shape.left = Inches(0.4)
                
                bullets = [
                    ("Verified Working Prototype", "Full-stack prototype with live sub-second FastAPI GPU inference validated across 4 Indian biomes."),
                    ("100% Free Data Pipeline", "Uses openly accessible Copernicus Sentinel-2 L2A BOA and ISRO Bhuvan CartoDEM data (zero API licensing cost)."),
                    ("Air-Gapped Defense Ready", "Sealed Docker containerization (--network=none --read-only) ensuring zero-leakage air-gapped defense deployment."),
                ]
                for idx, (k, v) in enumerate(bullets):
                    p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                    style_bullet(p, k, v, font_size=11)
            elif "ChaosToCode" in shape.text or "Your Team Name" in shape.text:
                shape.text_frame.clear()
                shape.text_frame.paragraphs[0].text = "ChaosToCode"

    # Add Diagram Image on Slide 4
    diag4_path = f"{DIAG_DIR}/diagram_slide4.png"
    if os.path.exists(diag4_path):
        pic = slide4.shapes.add_picture(diag4_path, Inches(0.4), Inches(3.55), width=Inches(12.5), height=Inches(3.2))
        pic.name = "Added_Diagram_4"

    # -------------------------------------------------------------
    # SLIDE 5: Impact & Competitive ROI Table (Diagram)
    # -------------------------------------------------------------
    slide5 = prs.slides[4]
    for s in list(slide5.shapes):
        if s.name.startswith("Added_Diagram"):
            slide5.shapes._spTree.remove(s._element)

    for shape in slide5.shapes:
        if shape.has_text_frame:
            if "IMPACT AND BENEFITS" in shape.text:
                shape.text_frame.clear()
                p = shape.text_frame.paragraphs[0]
                p.text = "BUSINESS IMPACT, ROI & COMPETITIVE ADVANTAGE"
                p.font.size = Pt(18)
                p.font.bold = True
                p.font.color.rgb = RGBColor(15, 23, 42)
            elif "Strategic Defense Impact" in shape.text or "Potential impact" in shape.text:
                tf = shape.text_frame
                tf.clear()
                shape.top = Inches(1.2)
                shape.height = Inches(2.2)
                shape.width = Inches(12.5)
                shape.left = Inches(0.4)
                
                bullets = [
                    ("Massive Cost Savings (ROI)", "Saves ₹100s of Crores annually by replacing expensive foreign commercial satellite procurement ($15/km²)."),
                    ("PMGSY Rural Connectivity", "Automated vectorization of unpaved village roads and connectivity corridors for PM Gati Shakti."),
                    ("Food Security & Disaster Response", "Individual crop parcel boundary tracking (PM Fasal Bima) + rapid flood inundation extent mapping."),
                ]
                for idx, (k, v) in enumerate(bullets):
                    p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                    style_bullet(p, k, v, font_size=11)
            elif "ChaosToCode" in shape.text or "Your Team Name" in shape.text:
                shape.text_frame.clear()
                shape.text_frame.paragraphs[0].text = "ChaosToCode"

    # Add Diagram Image on Slide 5
    diag5_path = f"{DIAG_DIR}/diagram_slide5.png"
    if os.path.exists(diag5_path):
        pic = slide5.shapes.add_picture(diag5_path, Inches(0.4), Inches(3.55), width=Inches(12.5), height=Inches(3.2))
        pic.name = "Added_Diagram_5"

    # -------------------------------------------------------------
    # SLIDE 6: Research & References
    # -------------------------------------------------------------
    slide6 = prs.slides[5]
    for shape in slide6.shapes:
        if shape.has_text_frame:
            if "RESEARCH" in shape.text:
                shape.text_frame.clear()
                p = shape.text_frame.paragraphs[0]
                p.text = "RESEARCH, DATASETS & BENCHMARKS"
                p.font.size = Pt(18)
                p.font.bold = True
                p.font.color.rgb = RGBColor(15, 23, 42)
            elif "Bayesian Uncertainty" in shape.text or "Details / Links" in shape.text:
                tf = shape.text_frame
                tf.clear()
                bullets = [
                    ("Bayesian Deep Learning & Uncertainty", "Kendall & Gal (NeurIPS) - 'What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?'"),
                    ("Irregular Cloud & Hole Inpainting", "Liu et al. (ECCV) - 'Image Inpainting for Irregular Holes Using Partial Convolutions.'"),
                    ("Sub-Pixel Anti-Aliasing", "Odena et al. - 'Deconvolution and Checkerboard Artifacts' (ICNR Sub-Pixel Convolution Formulation)."),
                    ("Earth Observation Super-Resolution Benchmark", "Cornebise et al. (NeurIPS) - 'WorldStrat: A Dataset for Spatial Super-Resolution in Earth Observation.'"),
                    ("Operational Portals & Ground Truth", "Copernicus Open Access Hub (Sentinel-2 BOA), ISRO NRSC Bhuvan (CartoDEM & LULC), SPOT 6/7 1.5m Pansharpened RGBN."),
                ]
                for idx, (k, v) in enumerate(bullets):
                    p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                    style_bullet(p, k, v, font_size=12, space_after=6)
            elif "ChaosToCode" in shape.text or "Your Team Name" in shape.text:
                shape.text_frame.clear()
                shape.text_frame.paragraphs[0].text = "ChaosToCode"

    out_path = r"C:\Users\muthu\Downloads\SIH2026-IDEA-Presentation-ChaosToCode.pptx"
    prs.save(out_path)
    print(f"\n[SUCCESS] Rebuilt pitch deck saved to: {out_path}")
    
    try:
        prs.save(PPT_PATH)
        print(f"[SUCCESS] Also updated original file: {PPT_PATH}")
    except Exception as e:
        print(f"[NOTE] Original file was open in PowerPoint, saved to: {out_path}")

if __name__ == "__main__":
    rebuild_pitch_deck()
