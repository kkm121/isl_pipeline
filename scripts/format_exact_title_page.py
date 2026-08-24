"""
=============================================================================
Update Slide 1 to match the exact SIH Title Page format shown in user's image
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

def format_title_page(path):
    if not os.path.exists(path):
        return
    try:
        prs = pptx.Presentation(path)
        s1 = prs.slides[0]
        
        # Hide or remove redundant extra subtitle/title placeholders if needed
        for shape in s1.shapes:
            if shape.has_text_frame:
                if "TITLE PAGE" in shape.text or "SMART INDIA" in shape.text or "BharatSRM" in shape.text:
                    if shape.name == "Subtitle 3":
                        shape.text_frame.clear() # Clear redundant subtitle
                    else:
                        shape.top = Inches(0.4)
                        shape.left = Inches(0.5)
                        shape.width = Inches(8.0)
                        shape.height = Inches(0.8)
                        shape.text_frame.clear()
                        p = shape.text_frame.paragraphs[0]
                        p.text = "TITLE PAGE"
                        p.font.size = Pt(28)
                        p.font.bold = True
                        p.font.name = "Arial"
                        p.font.color.rgb = RGBColor(0, 0, 0)
                elif "Problem Statement ID" in shape.text:
                    shape.top = Inches(1.5)
                    shape.left = Inches(0.5)
                    shape.width = Inches(7.5)
                    shape.height = Inches(5.2)
                    tf = shape.text_frame
                    tf.clear()
                    
                    fields = [
                        ("Problem Statement ID \u2013 ", "26142"),
                        ("Problem Statement Title- ", "Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries"),
                        ("Theme- ", "Smart Education"),
                        ("PS Category- ", "Software"),
                        ("Team ID- ", "1"),
                        ("Team Name (Registered on portal)- ", "CHAOS TO CODE"),
                    ]
                    
                    for idx, (label, val) in enumerate(fields):
                        p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                        p.space_after = Pt(14)
                        p.font.name = "Arial"
                        
                        r1 = p.add_run()
                        r1.text = label
                        r1.font.bold = True
                        r1.font.size = Pt(13)
                        r1.font.name = "Arial"
                        r1.font.color.rgb = RGBColor(0, 0, 0)
                        
                        r2 = p.add_run()
                        r2.text = val
                        r2.font.bold = True
                        r2.font.size = Pt(13)
                        r2.font.name = "Arial"
                        r2.font.color.rgb = RGBColor(16, 56, 107) # Exact SIH Navy Blue

        prs.save(path)
        print(f"[SUCCESS] Updated Title Page for: {path}")
    except Exception as e:
        print(f"[NOTE] Could not update {path}: {e}")

if __name__ == "__main__":
    for p in PPT_FILES:
        format_title_page(p)
