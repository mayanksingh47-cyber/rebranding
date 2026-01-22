"""
DXC Logo Rebranding Solution
Standalone solution for replacing old DXC logos and converting purple to orange

Features:
- Replaces old DXC logo (purple/black) with new logo in header, footer, and anywhere it appears
- Converts purple shapes/colors to orange
- Preserves all content and positioning
- Same UI/UX as existing rebranding platform

Author: Mayank Kumar
Cross Functional Capabilities - AI and Automation
Organization: DXC Technology
Version: 1.0.0 - Logo Rebranding Solution
"""

import streamlit as st
import os
import io
import zipfile
import tempfile
import base64
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# ============================================================================
# PDF REBRANDING CLASS (Logo & Color Replacement Focus)
# ============================================================================

# Check for PyMuPDF first (independent check - no poppler needed)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Check for pdf2image (requires poppler)
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Check for other dependencies
try:
    from PyPDF2 import PdfReader
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch, cm
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    from PIL import Image, ImageDraw
    PDF_OTHER_DEPS_AVAILABLE = True
except ImportError:
    PDF_OTHER_DEPS_AVAILABLE = False

# Export availability flags
PDF_AVAILABLE = PYMUPDF_AVAILABLE or PDF2IMAGE_AVAILABLE
PDF_ALL_DEPS_AVAILABLE = PDF_AVAILABLE and PDF_OTHER_DEPS_AVAILABLE


class DXCLogoRebrander:
    """
    Rebrands PDF documents by replacing old DXC logos and converting purple to orange.
    
    CRITICAL SCOPE:
    - REPLACES old DXC logos (purple/black) with new logo in header, footer, and anywhere
    - REPLACES purple colors/shapes with orange (#EF8900 or #FF7E51)
    - PRESERVES all content, positioning, and formatting
    """
    
    def __init__(self, poppler_path: Optional[str] = None):
        # DXC Brand Colors
        self.DXC_DARK_NAVY = HexColor('#0E1020')
        self.DXC_ORANGE = HexColor('#EF8900')
        self.DXC_MELON = HexColor('#FF7E51')  # Melon color for replacing purple
        self.DXC_GRAY = HexColor('#6B7280')
        self.DXC_LIGHT_BG = HexColor('#F6F3F0')
        
        # Poppler path (for Windows users who need to specify it - only if using pdf2image)
        self.poppler_path = poppler_path
        
        # Use PyMuPDF if available (preferred - no poppler needed)
        self.use_pymupdf = PYMUPDF_AVAILABLE
        
        # Page dimensions (A4)
        self.PAGE_WIDTH = A4[0]
        self.PAGE_HEIGHT = A4[1]
        
        # Header/Footer dimensions
        self.HEADER_HEIGHT = 1.0 * inch
        self.FOOTER_HEIGHT = 0.6 * inch
        self.LOGO_WIDTH = 4.5 * cm
        self.LOGO_HEIGHT = 1.0 * cm
        
        # Logo detection thresholds
        self.MAX_LOGO_WIDTH_CM = 10
        self.MAX_LOGO_HEIGHT_CM = 4
        
    def _find_logo(self) -> Optional[str]:
        """Find new DXC logo file"""
        logo_paths = [
            'extracted_image1.png',
            'DXCLogo.png',
            'dxc_brand_logo.png',
            'AI Agent 1 - System Network and Services Agent/DXCLogo.png'
        ]
        
        for logo_path in logo_paths:
            if os.path.exists(logo_path):
                return logo_path
        return None
    
    def _is_purple_color(self, r: int, g: int, b: int) -> bool:
        """
        Detect purple colors.
        Returns True if color is purple/violet/magenta.
        """
        # Convert to int to avoid overflow warnings
        r, g, b = int(r), int(g), int(b)
        
        # Clamp values to 0-255
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        # Classic purple: high R and B, low G
        is_classic_purple = (r > 80 and b > 80 and g < 100 and (r + b) > (2 * g + 50))
        
        # Violet: R and B close, G lower
        is_violet = (r > 100 and b > 100 and abs(r - b) < 80 and g < max(r, b) * 0.6)
        
        # Magenta: high R and B, very low G
        is_magenta = (r > 100 and b > 100 and abs(r - b) < 80 and g < max(r, b) * 0.6)
        
        # Office purple: specific RGB range
        is_office_purple = (r >= 112 and r <= 142 and g >= 48 and g <= 78 and b >= 160 and b <= 190)
        
        # DXC old brand purple
        is_dxc_purple = (r >= 85 and r <= 105 and g >= 25 and g <= 55 and b >= 140 and b <= 180)
        
        return is_classic_purple or is_violet or is_magenta or is_office_purple or is_dxc_purple
    
    def _is_old_logo_color(self, r: int, g: int, b: int) -> bool:
        """Detect old DXC logo colors (black, purple, or orange - old orange logo)"""
        r, g, b = int(r), int(g), int(b)
        # Black/dark colors (old black logo or TECHNOLOGY text)
        is_black = (r < 50 and g < 50 and b < 50)
        # Purple colors (old purple logo)
        is_purple = self._is_purple_color(r, g, b)
        # Orange colors (old orange DXC logo - RGB around 239, 137, 0 or similar)
        # Old orange logo: high R, medium G, low B
        is_old_orange = (r > 200 and r < 255 and g > 100 and g < 200 and b < 50)
        return is_black or is_purple or is_old_orange
    
    def _replace_purple_colors_in_image(self, img: Image.Image, replacement_color: str = 'orange') -> Image.Image:
        """
        Replace ALL purple colors (text, headings, shapes, borders, icons) with specified color.
        AGGRESSIVE: Catches ALL purple shades including light, dark, and gradient purples.
        Uses FAST vectorized NumPy operations to convert EVERY purple pixel to the replacement color.
        
        Args:
            img: Input image
            replacement_color: 'orange' (default) or 'black' - color to replace purple with
        """
        img_array = np.array(img.convert('RGB'))
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # COMPREHENSIVE purple detection - catch ALL purple shades
        
        # Classic purple: high R and B, low G
        is_classic_purple = (r > 70) & (b > 70) & (g < 120) & ((r + b) > (2 * g + 30))
        
        # Violet: R and B close, G lower
        is_violet = (r > 90) & (b > 90) & (np.abs(r - b) < 100) & (g < np.maximum(r, b) * 0.7)
        
        # Magenta: high R and B, very low G
        is_magenta = (r > 90) & (b > 90) & (np.abs(r - b) < 100) & (g < np.maximum(r, b) * 0.7)
        
        # Office purple: specific RGB range
        is_office_purple = (r >= 100) & (r <= 160) & (g >= 40) & (g <= 100) & (b >= 140) & (b <= 220)
        
        # DXC purple: common in documents
        is_dxc_purple = (r >= 75) & (r <= 120) & (g >= 20) & (g <= 70) & (b >= 120) & (b <= 200)
        
        # Dark purple (for text/headings)
        is_dark_purple = (r > 50) & (b > 80) & (g < 80) & ((r + b) > (2 * g + 30))
        
        # Light purple/pink-purple (gradients, light shades)
        is_light_purple = (r > 150) & (b > 150) & (g > 100) & (g < 180) & (np.abs(r - b) < 60) & (b > g + 20)
        
        # Purple-pink (common in gradients)
        is_purple_pink = (r > 180) & (b > 160) & (g > 120) & (g < 200) & (np.abs(r - b) < 50) & (b > g + 10)
        
        # Medium purple (shapes, borders)
        is_medium_purple = (r > 100) & (r < 200) & (b > 120) & (b < 220) & (g > 50) & (g < 150) & (b > r + 20) & (b > g + 30)
        
        # Combine ALL purple types (text, headings, shapes, borders, icons, backgrounds)
        purple_mask = (
            is_classic_purple | is_violet | is_magenta | is_office_purple | 
            is_dxc_purple | is_dark_purple | is_light_purple | is_purple_pink | is_medium_purple
        )
        
        # Replace ALL purple with specified color
        if replacement_color.lower() == 'black':
            # Black = RGB 0, 0, 0
            img_array[purple_mask, 0] = 0   # R
            img_array[purple_mask, 1] = 0   # G
            img_array[purple_mask, 2] = 0   # B
        else:
            # Default: Orange (#EF8900 = RGB 239, 137, 0)
            # This converts purple text, headings, shapes, borders, icons, and ALL purple elements to orange
            img_array[purple_mask, 0] = 239  # R
            img_array[purple_mask, 1] = 137  # G
            img_array[purple_mask, 2] = 0    # B
        
        return Image.fromarray(img_array, mode='RGB')
    
    def _detect_old_logos_in_image(self, img: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Detect ONLY the old DXC logo on extreme left top.
        VERY PRECISE: Only detects the logo, NOT any content like "SPARK IIoT".
        Returns list of (x, y, width, height) bounding boxes.
        """
        width, height = img.size
        img_array = np.array(img.convert('RGB'))
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Estimate DPI
        estimated_dpi = height / (A4[1] / 72)
        if estimated_dpi < 100:
            estimated_dpi = 150
        
        # Focus ONLY on extreme left top corner (where old logo is)
        # Top 1.5 inches, left 4cm - VERY SMALL area to avoid content
        header_zone_height = int(1.5 * inch * estimated_dpi / 72)
        left_side_width = int(4 * cm * estimated_dpi / 2.54)  # Left 4cm only - very small
        
        # Extract only the extreme left top area
        left_top_area = img_array[:min(header_zone_height, height), :min(left_side_width, width)]
        
        if left_top_area.size == 0:
            return []
        
        r_lt, g_lt, b_lt = left_top_area[:, :, 0], left_top_area[:, :, 1], left_top_area[:, :, 2]
        
        # Detect old orange logo (orange DXC - RGB around 239, 137, 0)
        # STRICT: High R (200-255), medium G (100-200), low B (0-50)
        is_old_orange = (r_lt > 200) & (r_lt < 255) & (g_lt > 100) & (g_lt < 200) & (b_lt < 50)
        
        # Only proceed if orange logo is found
        if not np.any(is_old_orange):
            return []
        
        # Find orange logo bounding box first
        orange_coords = np.where(is_old_orange)
        orange_y_min, orange_y_max = int(orange_coords[0].min()), int(orange_coords[0].max())
        orange_x_min, orange_x_max = int(orange_coords[1].min()), int(orange_coords[1].max())
        
        # Detect black text (TECHNOLOGY) - but ONLY directly below orange logo
        # Black text must be within 1cm below orange logo and aligned horizontally
        black_search_y_start = orange_y_max
        black_search_y_end = min(header_zone_height, orange_y_max + int(1.0 * cm * estimated_dpi / 2.54))
        black_search_x_start = max(0, orange_x_min - int(0.3 * cm * estimated_dpi / 2.54))
        black_search_x_end = min(left_side_width, orange_x_max + int(0.3 * cm * estimated_dpi / 2.54))
        
        if black_search_y_end > black_search_y_start:
            black_area = img_array[black_search_y_start:black_search_y_end, black_search_x_start:black_search_x_end]
            if black_area.size > 0:
                r_b, g_b, b_b = black_area[:, :, 0], black_area[:, :, 1], black_area[:, :, 2]
                is_black = (r_b < 60) & (g_b < 60) & (b_b < 60)
                
                if np.any(is_black):
                    # Find black text bounding box
                    black_coords = np.where(is_black)
                    black_y_min = int(black_coords[0].min()) + black_search_y_start
                    black_y_max = int(black_coords[0].max()) + black_search_y_start
                    black_x_min = int(black_coords[1].min()) + black_search_x_start
                    black_x_max = int(black_coords[1].max()) + black_search_x_start
                    
                    # Combine orange logo and black text bounding boxes
                    x_min = min(orange_x_min, black_x_min)
                    y_min = min(orange_y_min, black_y_min)
                    x_max = max(orange_x_max, black_x_max)
                    y_max = max(orange_y_max, black_y_max)
                else:
                    # Only orange logo, no black text
                    x_min, y_min = orange_x_min, orange_y_min
                    x_max, y_max = orange_x_max, orange_y_max
            else:
                # Only orange logo
                x_min, y_min = orange_x_min, orange_y_min
                x_max, y_max = orange_x_max, orange_y_max
        else:
            # Only orange logo
            x_min, y_min = orange_x_min, orange_y_min
            x_max, y_max = orange_x_max, orange_y_max
        
        # Very small expansion - just to catch logo edges
        expand_x = int(0.1 * cm * estimated_dpi / 2.54)
        expand_y = int(0.05 * cm * estimated_dpi / 2.54)
        
        x_min = max(0, x_min - expand_x)
        y_min = max(0, y_min - expand_y)
        x_max = min(left_side_width, x_max + expand_x)
        y_max = min(header_zone_height, y_max + expand_y)
        
        logo_w = x_max - x_min
        logo_h = y_max - y_min
        
        # Verify it's a reasonable logo size (small - logo only, not content)
        logo_w_cm = logo_w * 2.54 / estimated_dpi
        logo_h_cm = logo_h * 2.54 / estimated_dpi
        
        # Logo should be small (0.5-5cm width, 0.3-2cm height) - NOT large like "SPARK IIoT"
        if 0.5 <= logo_w_cm <= 5 and 0.3 <= logo_h_cm <= 2:
            return [(x_min, y_min, logo_w, logo_h)]
        
        return []
    
    def _remove_old_logos_from_image(self, img: Image.Image) -> Image.Image:
        """
        Remove ONLY the old DXC logo from extreme left top.
        PRECISE: Only removes exact logo bounding box, no content cutting.
        """
        # Use detection-based removal ONLY - this is precise and won't cut content
        logo_regions = self._detect_old_logos_in_image(img)
        
        if not logo_regions:
            return img
        
        result_img = img.copy()
        if result_img.mode != 'RGB':
            result_img = result_img.convert('RGB')
        
        img_array = np.array(result_img)
        width, height = img.size
        
        # Remove ONLY the detected logo regions - precise bounding boxes
        # NO expansion, NO large area clearing - just exact logo
        for x, y, w, h in logo_regions:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(width, int(x + w)), min(height, int(y + h))
            if x2 > x1 and y2 > y1:
                # Clear ONLY the exact logo bounding box - no expansion
                img_array[y1:y2, x1:x2] = [255, 255, 255]
        
        return Image.fromarray(img_array)
    
    def _add_new_logo_to_image(self, img: Image.Image, logo_path: Optional[str], 
                                old_logo_positions: List[Tuple[int, int, int, int]]) -> Image.Image:
        """
        Add new DXC logo to image.
        If old logo was on left, replace it at same position (moved to right).
        Otherwise, place on right side of header.
        Ensures only one logo exists.
        """
        if not logo_path or not os.path.exists(logo_path):
            return img
        
        result_img = img.copy()
        width, height = img.size
        
        try:
            logo_img = Image.open(logo_path)
            if logo_img.mode != 'RGBA':
                logo_img = logo_img.convert('RGBA')
            
            # Calculate DPI
            estimated_dpi = height / (A4[1] / 72)
            if estimated_dpi < 100:
                estimated_dpi = 200
            
            # Logo size: 4.5cm width, maintain aspect ratio
            logo_width_cm = 4.5
            logo_width_px = int(logo_width_cm * estimated_dpi / 2.54)
            
            original_width, original_height = logo_img.size
            aspect_ratio = original_height / original_width
            logo_height_px = int(logo_width_px * aspect_ratio)
            
            # Max height for header
            max_height_px = int(1.5 * cm * estimated_dpi / 2.54)
            if logo_height_px > max_height_px:
                logo_height_px = max_height_px
                logo_width_px = int(logo_height_px / aspect_ratio)
            
            logo_img = logo_img.resize((logo_width_px, logo_height_px), Image.Resampling.LANCZOS)
            
            # Determine logo position
            header_zone_height = int(1.0 * inch * estimated_dpi / 72)
            logo_y = int(0.2 * inch * estimated_dpi / 72)
            
            # If old logo was detected on left, place new logo on right at similar vertical position
            # Otherwise, default to right side
            if old_logo_positions:
                # Use vertical position from old logo if available
                for x, y, w, h in old_logo_positions:
                    if y > 0 and h > 0:
                        logo_y = y + (h - logo_height_px) // 2
                        logo_y = max(0, min(logo_y, header_zone_height - logo_height_px))
                        break
            
            # ALWAYS place on RIGHT side (replace old left logo position)
            logo_x = width - logo_width_px - int(0.5 * inch * estimated_dpi / 72)
            
            # Ensure within bounds
            logo_x = max(0, min(logo_x, width - logo_width_px))
            logo_y = max(0, min(logo_y, height - logo_height_px))
            
            # Ensure logo stays in header zone
            if logo_y + logo_height_px > header_zone_height:
                logo_y = header_zone_height - logo_height_px - int(0.1 * inch * estimated_dpi / 72)
                if logo_y < 0:
                    logo_y = int(0.1 * inch * estimated_dpi / 72)
            
            # Remove black background from logo
            if logo_img.mode == 'RGBA':
                logo_array = np.array(logo_img, dtype=np.uint8)
                r, g, b, a = logo_array[:, :, 0], logo_array[:, :, 1], logo_array[:, :, 2], logo_array[:, :, 3]
                is_black = (r < 50) & (g < 50) & (b < 50)
                a[is_black] = 0
                logo_array[:, :, 3] = a
                logo_img_clean = Image.fromarray(logo_array, mode='RGBA')
                result_img.paste(logo_img_clean, (logo_x, logo_y), logo_img_clean.split()[3])
        
        except Exception as e:
            import sys
            sys.stderr.write(f"Warning: Could not add logo: {str(e)}\n")
        
        return result_img
    
    def rebrand_pdf(self, source_file: bytes, source_filename: str, replacement_color: str = 'orange') -> Tuple[bytes, str]:
        """
        Rebrand a PDF document:
        - Replace old DXC logos (purple/black) with new logo
        - Convert purple colors to specified color (orange or black)
        - Preserve all content and positioning
        
        Args:
            source_file: PDF file bytes
            source_filename: Original filename
            replacement_color: 'orange' (default) or 'black' - color to replace purple shapes with
        """
        # Runtime dependency checks
        runtime_pymupdf = False
        runtime_pdf2image = False
        runtime_other_deps = False
        
        try:
            import fitz
            runtime_pymupdf = True
        except ImportError:
            pass
        
        try:
            from pdf2image import convert_from_path
            runtime_pdf2image = True
        except ImportError:
            pass
        
        try:
            from PyPDF2 import PdfReader
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import inch, cm
            from reportlab.pdfgen import canvas
            from reportlab.lib.colors import HexColor
            from PIL import Image, ImageDraw
            runtime_other_deps = True
        except ImportError:
            pass
        
        if not runtime_pymupdf and not runtime_pdf2image:
            raise RuntimeError(
                "No PDF conversion library available. Please install one:\n"
                "pip install PyMuPDF (recommended - no poppler needed)\n"
                "OR: pip install pdf2image (requires poppler)"
            )
        
        if not runtime_other_deps:
            raise RuntimeError(
                "Missing required dependencies. Please install:\n"
                "pip install PyPDF2 reportlab Pillow numpy"
            )
        
        # Save source to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(source_file)
            source_path = tmp.name
        
        try:
            logo_path = self._find_logo()
            
            # Convert PDF to images
            page_images = []
            if runtime_pymupdf:
                doc = fitz.open(source_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    page_images.append(img)
                doc.close()
            elif runtime_pdf2image:
                images = convert_from_path(
                    source_path,
                    dpi=150,
                    poppler_path=self.poppler_path
                )
                page_images = images
            
            # Process each page
            processed_images = []
            for img in page_images:
                # Step 1: Remove old logo FIRST (before any processing)
                # This ensures logo is removed even if other steps interfere
                img = self._remove_old_logos_from_image(img)
                
                # Step 2: Replace purple colors/text with specified color (orange or black)
                img = self._replace_purple_colors_in_image(img, replacement_color)
                
                # Step 3: Remove old logo AGAIN (in case purple conversion created orange pixels)
                img = self._remove_old_logos_from_image(img)
                
                # Step 4: Add new logo ONLY on right side (ensures only one logo exists)
                # Don't pass old logo positions - always place on right
                img = self._add_new_logo_to_image(img, logo_path, [])
                
                processed_images.append(img)
            
            # Convert processed images back to PDF
            output_buffer = io.BytesIO()
            c = canvas.Canvas(output_buffer, pagesize=A4)
            
            for img in processed_images:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                    img.save(tmp_img.name, format='PNG')
                    tmp_img_path = tmp_img.name
                
                try:
                    c.drawImage(tmp_img_path, 0, 0, width=A4[0], height=A4[1], preserveAspectRatio=True)
                    c.showPage()
                finally:
                    if os.path.exists(tmp_img_path):
                        os.unlink(tmp_img_path)
            
            c.save()
            output_buffer.seek(0)
            
            # Generate output filename
            output_name = f"DXC_Logo_Rebranded_{Path(source_filename).stem}_{datetime.now().strftime('%Y%m%d')}.pdf"
            
            return output_buffer.getvalue(), output_name
            
        finally:
            if os.path.exists(source_path):
                os.unlink(source_path)


# ============================================================================
# UI STYLING (Same as existing platform)
# ============================================================================

def get_logo_base64():
    """Load DXC logo as base64"""
    logo_path = Path(__file__).parent / "dxc_brand_logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def load_css():
    """Load enterprise CSS styling - EXACT SAME as existing files"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;500;600;700&display=swap');
    
    :root {
        --dxc-dark: #0E1020;
        --dxc-orange: #EF8900;
        --dxc-gradient-start: #5B8DEF;
        --dxc-gradient-end: #EF8068;
        --dxc-light-bg: #F8F9FA;
        --success-green: #198754;
        --error-red: #DC3545;
        --warning-amber: #FFC107;
        --text-primary: #212529;
        --text-secondary: #6C757D;
        --bg-white: #FFFFFF;
        --border-light: #DEE2E6;
    }
    
    /* Hide Streamlit branding and unnecessary elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Remove default Streamlit container padding */
    .block-container {
        padding-top: 5rem;
        padding-bottom: 5rem;
    }
    
    /* Main app styling */
    .main .block-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Enterprise Header - Fixed to top */
    .enterprise-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: linear-gradient(135deg, var(--dxc-dark) 0%, #1C1E30 50%, #2A2D45 100%);
        padding: 0.75rem 1rem;
        border-radius: 0;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 2px solid var(--dxc-orange);
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .header-logo img {
        height: 36px;
        width: auto;
    }
    
    .header-content {
        border-left: 1px solid rgba(255,255,255,0.2);
        padding-left: 0.75rem;
    }
    
    .enterprise-title {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .enterprise-subtitle {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.75rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.1rem;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--dxc-dark);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--dxc-orange);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem 0;
    }
    
    .status-ready {
        background: rgba(25, 135, 84, 0.1);
        color: var(--success-green);
        border: 1px solid rgba(25, 135, 84, 0.3);
    }
    
    .status-unavailable {
        background: rgba(220, 53, 69, 0.1);
        color: var(--error-red);
        border: 1px solid rgba(220, 53, 69, 0.3);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .status-dot.green {
        background: var(--success-green);
    }
    
    .status-dot.red {
        background: var(--error-red);
    }
    
    /* File upload success message */
    .upload-success {
        background: linear-gradient(135deg, rgba(25, 135, 84, 0.08) 0%, rgba(25, 135, 84, 0.04) 100%);
        border: 1px solid rgba(25, 135, 84, 0.2);
        border-left: 4px solid var(--success-green);
        color: #0F5132;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    /* Error message */
    .upload-error {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.08) 0%, rgba(220, 53, 69, 0.04) 100%);
        border: 1px solid rgba(220, 53, 69, 0.2);
        border-left: 4px solid var(--error-red);
        color: #842029;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    
    /* Feature cards grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .feature-card {
        background: var(--bg-white);
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid var(--border-light);
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .feature-card:hover {
        border-color: var(--dxc-orange);
        box-shadow: 0 4px 12px rgba(14, 16, 32, 0.08);
    }
    
    .feature-icon {
        width: 40px;
        height: 40px;
        margin: 0 auto 0.75rem;
        background: linear-gradient(135deg, var(--dxc-gradient-start), var(--dxc-gradient-end));
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .feature-icon svg {
        width: 20px;
        height: 20px;
        fill: white;
    }
    
    .feature-title {
        font-weight: 600;
        color: var(--dxc-dark);
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    
    .feature-desc {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    
    /* Stats cards */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        flex: 1;
        background: var(--bg-white);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--dxc-dark);
        line-height: 1;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #F6F3F0;
        border-right: none;
        box-shadow: 2px 0 8px rgba(14, 16, 32, 0.06);
        z-index: auto !important;
    }
    
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }
    
    section[data-testid="stSidebar"] .section-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--dxc-dark);
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--dxc-orange);
    }
    
    /* Sidebar section cards */
    .sidebar-section {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(14, 16, 32, 0.08);
        box-shadow: 0 1px 3px rgba(14, 16, 32, 0.04);
    }
    
    .sidebar-section-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--dxc-dark);
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--dxc-orange);
    }
    
    /* Sidebar status pills */
    .sidebar-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-status.ready {
        background: linear-gradient(135deg, rgba(25, 135, 84, 0.12) 0%, rgba(25, 135, 84, 0.06) 100%);
        color: #0D6E3F;
        border: 1px solid rgba(25, 135, 84, 0.2);
    }
    
    .sidebar-status.unavailable {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.12) 0%, rgba(220, 53, 69, 0.06) 100%);
        color: #A52834;
        border: 1px solid rgba(220, 53, 69, 0.2);
    }
    
    .sidebar-status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    
    .sidebar-status-dot.green {
        background: var(--success-green);
        box-shadow: 0 0 4px rgba(25, 135, 84, 0.5);
    }
    
    .sidebar-status-dot.red {
        background: var(--error-red);
        box-shadow: 0 0 4px rgba(220, 53, 69, 0.5);
    }
    
    /* Sidebar list items */
    .sidebar-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .sidebar-list-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.4rem 0;
        font-size: 0.82rem;
        color: var(--text-primary);
        line-height: 1.4;
    }
    
    .sidebar-list-item::before {
        content: "";
        width: 4px;
        height: 4px;
        background: var(--dxc-orange);
        border-radius: 50%;
        margin-top: 0.45rem;
        flex-shrink: 0;
    }
    
    .sidebar-list-item.check::before {
        content: "✓";
        width: auto;
        height: auto;
        background: none;
        color: var(--success-green);
        font-weight: 700;
        font-size: 0.75rem;
        margin-top: 0;
    }
    
    .sidebar-list-item.update::before {
        content: "→";
        width: auto;
        height: auto;
        background: none;
        color: var(--dxc-orange);
        font-weight: 700;
        font-size: 0.75rem;
        margin-top: 0;
    }
    
    /* Sidebar template info */
    .sidebar-template {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
        font-size: 0.82rem;
    }
    
    .sidebar-template-label {
        font-weight: 600;
        color: var(--text-secondary);
        min-width: 45px;
    }
    
    .sidebar-template-value {
        color: var(--dxc-dark);
        font-weight: 500;
    }
    
    /* Sidebar divider */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(14, 16, 32, 0.1), transparent);
        margin: 1rem 0;
    }
    
    /* File list styling */
    .file-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border-light);
        font-size: 0.9rem;
    }
    
    .file-item:last-child {
        border-bottom: none;
    }
    
    .file-type-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-right: 0.75rem;
        color: white;
    }
    
    .file-type-badge.docx {
        background: linear-gradient(135deg, #2B579A, #1E3A5F);
    }
    
    .file-type-badge.pdf {
        background: linear-gradient(135deg, #DC143C, #B22222);
    }
    
    /* Footer - Fixed to bottom */
    .enterprise-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 999;
        text-align: center;
    }
    
    .footer-top {
        background: #000000;
        border-top: none;
        padding: 0.4rem 1rem;
    }
    
    .footer-credits {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.8);
    }
    
    .footer-credits strong {
        color: #ffffff;
        font-weight: 600;
    }
    
    .footer-separator {
        color: rgba(255,255,255,0.5);
    }
    
    .footer-bottom {
        background: #000000;
        padding: 0.25rem 1rem;
    }
    
    .footer-copyright {
        font-size: 0.65rem;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .footer-developer a {
        color: var(--dxc-orange);
        text-decoration: none;
    }
    
    
    /* Button styling overrides */
    .stButton > button {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--dxc-dark), #1C1E30);
        border: none;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1C1E30, var(--dxc-dark));
        box-shadow: 0 4px 12px rgba(14, 16, 32, 0.3);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        border-radius: 6px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--dxc-dark), var(--dxc-orange));
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: var(--bg-white);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 1rem;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--dxc-dark);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border-light);
        border-radius: 8px;
        padding: 1rem;
        background: var(--bg-white);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--dxc-orange);
    }
    
    /* Remove expander arrows and streamlit decorations */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    
    /* Info/warning/error boxes - cleaner look */
    .stAlert {
        border-radius: 6px;
    }
    
    /* Divider styling */
    hr {
        border: none;
        border-top: 1px solid var(--border-light);
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# BULK REBRANDER
# ============================================================================

class LogoRebrandingBulkProcessor:
    """Handles bulk rebranding of PDF documents"""
    
    def __init__(self):
        self.pdf_rebrander = DXCLogoRebrander() if PDF_ALL_DEPS_AVAILABLE else None
        self.results = []
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set progress callback function"""
        self.progress_callback = callback
    
    def update_progress(self, message: str, progress: int, file_name: str = ""):
        """Update progress"""
        if self.progress_callback:
            self.progress_callback(message, progress, file_name)
    
    def process_files(self, uploaded_files: List, replacement_color: str = 'orange') -> Dict:
        """
        Process multiple uploaded PDF files for logo rebranding.
        
        Args:
            uploaded_files: List of uploaded file objects
            replacement_color: 'orange' (default) or 'black' - color to replace purple shapes with
        
        Returns dict with:
        - successful: list of (filename, bytes, new_filename)
        - failed: list of (filename, error_message)
        - stats: processing statistics
        """
        results = {
            'successful': [],
            'failed': [],
            'stats': {
                'total': len(uploaded_files),
                'pdf_count': 0,
                'success_count': 0,
                'fail_count': 0,
                'processing_time': 0
            }
        }
        
        start_time = time.time()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            file_ext = Path(file_name).suffix.lower()
            
            progress = int(((idx + 1) / len(uploaded_files)) * 100)
            self.update_progress(f"Processing {file_name}...", progress, file_name)
            
            try:
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)
                
                if file_ext == '.pdf':
                    results['stats']['pdf_count'] += 1
                    
                    if not self.pdf_rebrander:
                        results['failed'].append((file_name, "PDF rebranding not available"))
                        results['stats']['fail_count'] += 1
                        continue
                    
                    self.update_progress(f"Rebranding {file_name}...", progress, file_name)
                    
                    rebranded_bytes, new_filename = self.pdf_rebrander.rebrand_pdf(
                        file_bytes, file_name, replacement_color
                    )
                    
                    self.update_progress(f"Completed {file_name}", progress, file_name)
                    results['successful'].append((file_name, rebranded_bytes, new_filename))
                    results['stats']['success_count'] += 1
                else:
                    results['failed'].append((file_name, f"Unsupported file type: {file_ext}. Only PDF files are supported."))
                    results['stats']['fail_count'] += 1
                    
            except Exception as e:
                results['failed'].append((file_name, str(e)))
                results['stats']['fail_count'] += 1
        
        results['stats']['processing_time'] = round(time.time() - start_time, 2)
        
        self.update_progress("Processing complete!", 100, "")
        return results
    
    def create_zip_download(self, successful_files: List[Tuple]) -> bytes:
        """Create a ZIP file containing all successfully rebranded documents"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for original_name, file_bytes, new_name in successful_files:
                zip_file.writestr(new_name, file_bytes)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    load_css()
    
    # Get logo
    logo_base64 = get_logo_base64()
    
    # Header with logo
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="DXC Technology">' if logo_base64 else '<span style="color: white; font-weight: 700; font-size: 1.5rem;">DXC</span>'
    
    st.markdown(f"""
    <div class="enterprise-header">
        <div class="header-left">
            <div class="header-logo">
                {logo_html}
            </div>
            <div class="header-content">
                <div class="enterprise-title">DXC Logo Rebranding</div>
                <div class="enterprise-subtitle">Replace old logos and convert purple to orange</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check dependencies
    if not PDF_ALL_DEPS_AVAILABLE:
        st.markdown("""
        <div class="upload-error">
            <strong>Configuration Error:</strong> PDF rebranding not available. 
            Please install required dependencies:
            <br><br>
            <strong>For PDF:</strong> pip install PyMuPDF PyPDF2 reportlab Pillow numpy
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Feature cards
    st.markdown("""
    <div class="feature-grid" style="grid-template-columns: repeat(3, 1fr);">
        <div class="feature-card">
            <div class="feature-icon">
                <svg viewBox="0 0 24 24" fill="white"><path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/></svg>
            </div>
            <div class="feature-title">PDF Documents</div>
            <div class="feature-desc">Rebrand .pdf files</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">
                <svg viewBox="0 0 24 24" fill="white"><path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4C16.41,4 20,7.59 20,12C20,16.41 16.41,20 12,20C7.59,20 4,16.41 4,12C4,7.59 7.59,4 12,4M12,6A6,6 0 0,0 6,12A6,6 0 0,0 12,18A6,6 0 0,0 18,12A6,6 0 0,0 12,6Z"/></svg>
            </div>
            <div class="feature-title">Logo Replacement</div>
            <div class="feature-desc">Old → New DXC logo</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">
                <svg viewBox="0 0 24 24" fill="white"><path d="M7.5,21L5.5,19L18.5,6L20.5,8M17,2L22,7L17,12L15.59,10.59L18.17,8L10.5,8L10.5,6L18.17,6L15.59,3.41L17,2Z"/></svg>
            </div>
            <div class="feature-title">Color Conversion</div>
            <div class="feature-desc">Purple → Orange</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize bulk rebrander
    if 'logo_rebrander' not in st.session_state:
        st.session_state.logo_rebrander = LogoRebrandingBulkProcessor()
    
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    
    # Sidebar
    with st.sidebar:
        # System Status Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">System Status</div>
        """, unsafe_allow_html=True)
        
        if PDF_ALL_DEPS_AVAILABLE:
            st.markdown("""
            <div class="sidebar-status ready">
                <span class="sidebar-status-dot green"></span>
                PDF Rebranding Ready
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-status unavailable">
                <span class="sidebar-status-dot red"></span>
                PDF Not Available
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Supported Formats Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">Supported Formats</div>
            <div class="sidebar-list-item">PDF Documents (.pdf)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Rebranding Process Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">Scope</div>
            <div style="font-size: 0.75rem; font-weight: 600; color: #6C757D; margin-bottom: 0.4rem; text-transform: uppercase;">Preserved</div>
            <div class="sidebar-list-item check">All content and text</div>
            <div class="sidebar-list-item check">Images and graphics</div>
            <div class="sidebar-list-item check">Page layout and formatting</div>
            <div class="sidebar-list-item check">Document structure</div>
            <div style="font-size: 0.75rem; font-weight: 600; color: #6C757D; margin: 0.75rem 0 0.4rem 0; text-transform: uppercase;">Updated</div>
            <div class="sidebar-list-item update">Old DXC logos → New logo</div>
            <div class="sidebar-list-item update">Purple colors → Orange</div>
            <div class="sidebar-list-item update">Purple shapes → Orange</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<div class="section-header">Upload PDF Documents</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Select PDF (.pdf) files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF documents to replace old DXC logos and convert purple to orange. All content and positioning will be preserved.",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        # Show uploaded files summary
        st.markdown(f"""
        <div class="upload-success">
            <strong>{len(uploaded_files)}</strong> PDF file(s) uploaded and ready for logo rebranding
        </div>
        """, unsafe_allow_html=True)
        
        # File list
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Uploaded Files:**")
            for f in uploaded_files:
                st.markdown(f"""
                <div class="file-item">
                    <span class="file-type-badge pdf">PDF</span>
                    {f.name}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Summary:**")
            st.markdown(f"PDF Documents: **{len(uploaded_files)}**")
        
        st.markdown("---")
        
        # Color selection option
        st.markdown("**Shape/Box Color Option:**")
        replacement_color = st.radio(
            "Choose the color for converted purple shapes/boxes:",
            options=['Orange', 'Black'],
            index=0,  # Default to Orange
            horizontal=True,
            help="Select whether purple shapes/boxes should be converted to Orange (default) or Black"
        )
        replacement_color_lower = replacement_color.lower()
        
        st.markdown("---")
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            process_btn = st.button(
                "Rebrand Logos & Colors",
                use_container_width=True,
                type="primary"
            )
        
        if process_btn:
            # Progress tracking
            progress_container = st.empty()
            progress_bar = st.progress(0)
            current_file_container = st.empty()
            
            def update_progress(message, progress, file_name):
                progress_container.info(message)
                progress_bar.progress(progress / 100)
                if file_name:
                    current_file_container.caption(f"Processing: {file_name}")
            
            st.session_state.logo_rebrander.set_progress_callback(update_progress)
            
            # Process files
            with st.spinner("Processing documents..."):
                results = st.session_state.logo_rebrander.process_files(uploaded_files, replacement_color_lower)
                st.session_state.processing_results = results
            
            # Clear progress
            progress_container.empty()
            progress_bar.empty()
            current_file_container.empty()
            
            st.rerun()
    
    # Display results
    if st.session_state.processing_results:
        results = st.session_state.processing_results
        
        st.markdown("---")
        st.markdown('<div class="section-header">Processing Results</div>', unsafe_allow_html=True)
        
        # Statistics
        stats = results['stats']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", stats['total'])
        with col2:
            st.metric("Successful", stats['success_count'])
        with col3:
            st.metric("Failed", stats['fail_count'])
        with col4:
            st.metric("Time (sec)", stats['processing_time'])
        
        # Success rate
        if stats['total'] > 0:
            success_rate = (stats['success_count'] / stats['total']) * 100
            st.progress(success_rate / 100)
            st.caption(f"Success Rate: {success_rate:.1f}%")
        
        # Successful files
        if results['successful']:
            st.markdown("---")
            st.markdown('<div class="section-header">Successfully Rebranded</div>', unsafe_allow_html=True)
            
            # Download all as ZIP
            if len(results['successful']) > 1:
                zip_data = st.session_state.logo_rebrander.create_zip_download(results['successful'])
                zip_filename = f"DXC_Logo_Rebranded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                
                st.download_button(
                    label=f"Download All ({len(results['successful'])} files) as ZIP",
                    data=zip_data,
                    file_name=zip_filename,
                    mime="application/zip",
                    use_container_width=True,
                    type="primary"
                )
                
                st.markdown("---")
                st.markdown("**Or download individually:**")
            
            # Individual downloads
            for original_name, file_bytes, new_name in results['successful']:
                st.download_button(
                    label=f"Download: {new_name}",
                    data=file_bytes,
                    file_name=new_name,
                    mime="application/pdf",
                    key=f"download_{original_name}"
                )
        
        # Failed files
        if results['failed']:
            st.markdown("---")
            st.markdown('<div class="section-header">Failed Files</div>', unsafe_allow_html=True)
            
            for file_name, error_msg in results['failed']:
                st.markdown(f"""
                <div class="upload-error">
                    <strong>{file_name}</strong><br>
                    {error_msg}
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown(f"""
    <div class="enterprise-footer">
        <div class="footer-top">
            <div class="footer-credits">
                <span>Developed by <strong>Mayank Kumar</strong></span>
                <span class="footer-separator">|</span>
                <span>Cross Functional Capabilities - AI and Automation</span>
                <span class="footer-separator">|</span>
                <span>DXC Technology</span>
            </div>
        </div>
        <div class="footer-bottom">
            <div class="footer-copyright">DXC Technology &copy; {datetime.now().year}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
