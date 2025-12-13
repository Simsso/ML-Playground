#!/usr/bin/env python3
"""
Airport Diagram Processing Pipeline
Converts PDF airport diagrams to cropped images (PNG and JPG)
"""

import sys
import os
import re
from pathlib import Path
from pdf2image import convert_from_path
import cv2
import numpy as np
import pdfplumber


def extract_header_text_bbox(pdf_path):
    """
    Extract bounding box of airport header text in PDF coordinates.

    Returns:
        dict with 'bbox' (x0, top, x1, bottom in PDF points) and 'text' info
        or None if text cannot be extracted
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return None

            page = pdf.pages[0]
            words = page.extract_words()

            # Find ICAO code first
            icao_pattern = re.compile(r'^[A-Z0-9]{3,4}:$')
            icao_word = None

            for word in words:
                if word['top'] < 35 and icao_pattern.match(word['text']):
                    icao_word = word
                    break

            if not icao_word:
                return None

            # Find words on the same line as ICAO (line 1)
            line1_y_tolerance = 3  # points
            line1_words = [icao_word]
            for word in words:
                if word != icao_word and abs(word['top'] - icao_word['top']) < line1_y_tolerance:
                    # Must be to the right of ICAO and be uppercase text
                    if word['x0'] > icao_word['x1'] and any(c.isupper() for c in word['text']) and len(word['text']) > 2:
                        line1_words.append(word)

            # Find words on line 2 (location line, below line 1)
            line1_bottom = max(w['bottom'] for w in line1_words)
            line2_words = []
            for word in words:
                # Location line should be 2-15 points below line 1
                y_diff = word['top'] - line1_bottom
                if 2 < y_diff < 15:
                    # Should be uppercase text and not coordinates
                    if any(c.isupper() for c in word['text']) and len(word['text']) > 2:
                        # Not coordinates (no degree symbols or numbers with apostrophes)
                        if '°' not in word['text'] and '′' not in word['text']:
                            # Should be roughly aligned horizontally (within 2 points of first location word)
                            if not line2_words or abs(word['top'] - line2_words[0]['top']) < 2:
                                line2_words.append(word)

            if not line2_words:
                return None

            # Calculate bounding box for both lines
            all_words = line1_words + line2_words
            x0 = min(w['x0'] for w in all_words)
            top = min(w['top'] for w in all_words)
            x1 = max(w['x1'] for w in all_words)
            bottom = max(w['bottom'] for w in all_words)

            # Extract text for display
            line1_text = ' '.join(w['text'] for w in sorted(line1_words, key=lambda w: w['x0']))
            line2_text = ' '.join(w['text'] for w in sorted(line2_words, key=lambda w: w['x0']))

            print(f"Extracted header text: {line1_text}")
            print(f"Location: {line2_text}")
            print(f"PDF bbox: x0={x0:.1f}, top={top:.1f}, x1={x1:.1f}, bottom={bottom:.1f}")

            return {
                'bbox': (x0, top, x1, bottom),
                'text': f"{line1_text}\n{line2_text}"
            }

    except Exception as e:
        print(f"Warning: Could not extract header text bbox: {e}")
        import traceback
        traceback.print_exc()
        return None


def pdf_to_image(pdf_path, dpi=450):
    """Convert PDF to high-resolution image."""
    print(f"Converting PDF to image at {dpi} DPI...")
    images = convert_from_path(pdf_path, dpi=dpi)
    if len(images) == 0:
        raise ValueError("No pages found in PDF")
    # Return first page as numpy array
    img = np.array(images[0])
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def detect_and_crop_border(image):
    """
    Detect black rectangular border and crop to inside of it.
    Returns:
        tuple: (cropped_image, border_coords)
        border_coords: dict with keys 'x', 'y', 'w', 'h'
    """
    print("Detecting border...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to find dark regions (black border)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Warning: No border detected, returning original image")
        return image, None

    # Find the largest contour (should be the border)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)

    print(f"Border detected at: x={x}, y={y}, w={w}, h={h}")
    print(f"Cropping to inside of border...")

    # Add small margin to crop inside the border (adjust as needed)
    margin = 5
    x_crop = x + margin
    y_crop = y + margin
    w_crop = w - 2 * margin
    h_crop = h - 2 * margin

    # Ensure we don't go out of bounds
    x_crop = max(0, x_crop)
    y_crop = max(0, y_crop)
    w_crop = min(w_crop, image.shape[1] - x_crop)
    h_crop = min(h_crop, image.shape[0] - y_crop)

    # Crop the image
    cropped = image[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]

    border_coords = {'x': x, 'y': y, 'w': w, 'h': h}

    return cropped, border_coords


def find_best_corner(image, dpi):
    """
    Analyze top corners to find which has more white space.
    Returns 'left' or 'right'.

    Args:
        image: The cropped diagram image
        dpi: DPI used for conversion (to scale corner region size)
    """
    # Calculate corner region size based on DPI
    # Make regions larger to capture more diagram content
    # At 300 DPI: ~900×600px, At 450 DPI: ~1350×900px
    corner_width = int(dpi * 3.0)
    corner_height = int(dpi * 2.0)

    height, width = image.shape[:2]

    # Ensure we don't exceed image dimensions
    corner_width = min(corner_width, width // 2)
    corner_height = min(corner_height, height // 4)

    # Extract top-left and top-right corner regions
    top_left = image[0:corner_height, 0:corner_width]
    top_right = image[0:corner_height, width-corner_width:width]

    # Convert to grayscale for analysis
    gray_left = cv2.cvtColor(top_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(top_right, cv2.COLOR_BGR2GRAY)

    # Count dark/mid-tone pixels (grayscale < 200)
    # Ignore light gray and white pixels to better detect actual diagram content
    dark_threshold = 200
    left_dark_count = np.sum(gray_left < dark_threshold)
    right_dark_count = np.sum(gray_right < dark_threshold)

    # Calculate percentages
    left_total = gray_left.size
    right_total = gray_right.size
    left_dark_pct = (left_dark_count / left_total) * 100
    right_dark_pct = (right_dark_count / right_total) * 100

    print(f"Corner analysis (region: {corner_width}×{corner_height}px):")
    print(f"  Top-left content density: {left_dark_pct:.1f}%")
    print(f"  Top-right content density: {right_dark_pct:.1f}%")

    # Place text in corner with LESS content (more empty space)
    return 'left' if left_dark_pct < right_dark_pct else 'right'


def extract_text_region(original_image, bbox_pdf, dpi):
    """
    Extract text region from original image based on PDF bounding box.

    Args:
        original_image: The full rendered PDF image (before cropping)
        bbox_pdf: Bounding box in PDF coordinates (x0, top, x1, bottom in points)
        dpi: DPI used for rendering

    Returns:
        Extracted text region as image
    """
    # Convert PDF points to pixels
    # 1 PDF point = 1/72 inch, so at N DPI: 1 point = N/72 pixels
    points_to_pixels = dpi / 72.0

    x0_pdf, top_pdf, x1_pdf, bottom_pdf = bbox_pdf

    # Convert to pixel coordinates
    x0_px = int(x0_pdf * points_to_pixels)
    y0_px = int(top_pdf * points_to_pixels)
    x1_px = int(x1_pdf * points_to_pixels)
    y1_px = int(bottom_pdf * points_to_pixels)

    # Add minimal padding (2 points) to avoid cutting off anti-aliasing
    padding = int(2 * points_to_pixels)
    x0_px = max(0, x0_px - padding)
    y0_px = max(0, y0_px - padding)
    x1_px = min(original_image.shape[1], x1_px + padding)
    y1_px = min(original_image.shape[0], y1_px + padding)

    print(f"Extracting text region: ({x0_px}, {y0_px}) to ({x1_px}, {y1_px}) pixels")

    # Extract region
    text_region = original_image[y0_px:y1_px, x0_px:x1_px].copy()

    # Trim excess white space from bottom and right to avoid including border elements
    text_region = trim_whitespace(text_region)

    # Make white background transparent
    text_region = make_white_transparent(text_region)

    return text_region


def make_white_transparent(image, threshold=245):
    """
    Convert white/near-white pixels to transparent.

    Args:
        image: BGR image
        threshold: Brightness threshold for "white" pixels (default 245)

    Returns:
        RGBA image with transparent background
    """
    # Convert BGR to BGRA (add alpha channel)
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Convert to grayscale to check brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find white/near-white pixels
    white_mask = gray > threshold

    # Set alpha channel to 0 (transparent) for white pixels
    bgra[:, :, 3] = np.where(white_mask, 0, 255)

    return bgra


def trim_whitespace(image, threshold=250):
    """
    Trim excess white space from edges of image.

    Args:
        image: Image to trim
        threshold: Pixel brightness threshold (default 250, near-white)

    Returns:
        Trimmed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find rows and columns that contain non-white pixels
    non_white_rows = np.where(np.any(gray < threshold, axis=1))[0]
    non_white_cols = np.where(np.any(gray < threshold, axis=0))[0]

    if len(non_white_rows) == 0 or len(non_white_cols) == 0:
        # Image is all white, return small portion
        return image[:10, :10]

    # Get bounding box of non-white content
    y_min, y_max = non_white_rows[0], non_white_rows[-1] + 1
    x_min, x_max = non_white_cols[0], non_white_cols[-1] + 1

    # Add small margin back (5 pixels)
    margin = 5
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(image.shape[0], y_max + margin)
    x_max = min(image.shape[1], x_max + margin)

    return image[y_min:y_max, x_min:x_max]


def place_text_region(image, text_region, corner, dpi):
    """
    Place extracted text region on the image at the specified corner.

    Args:
        image: The cropped diagram image (BGR)
        text_region: Extracted text image region (BGRA with transparency)
        corner: 'left' or 'right'
        dpi: DPI used for conversion (for scaling)

    Returns:
        Image with text placed
    """
    result = image.copy()

    # Calculate padding from edges
    padding = int(dpi / 10)  # ~30-45px at 300-450 DPI

    text_height, text_width = text_region.shape[:2]

    # Calculate position based on corner
    if corner == 'left':
        x_pos = padding
    else:  # right
        x_pos = image.shape[1] - text_width - padding

    y_pos = padding

    # Ensure we don't exceed image boundaries
    if x_pos + text_width > image.shape[1]:
        x_pos = image.shape[1] - text_width
    if y_pos + text_height > image.shape[0]:
        y_pos = image.shape[0] - text_height

    x_pos = max(0, x_pos)
    y_pos = max(0, y_pos)

    # Alpha blend the text region onto the result
    # Extract the region where we'll place the text
    roi = result[y_pos:y_pos+text_height, x_pos:x_pos+text_width]

    # Extract alpha channel from text_region (BGRA)
    alpha = text_region[:, :, 3] / 255.0
    alpha_3channel = np.stack([alpha, alpha, alpha], axis=2)

    # Extract BGR channels from text_region
    text_bgr = text_region[:, :, :3]

    # Alpha blend: result = foreground * alpha + background * (1 - alpha)
    blended = (text_bgr * alpha_3channel + roi * (1 - alpha_3channel)).astype(np.uint8)

    # Place blended result back
    result[y_pos:y_pos+text_height, x_pos:x_pos+text_width] = blended

    print(f"Text region placed at top-{corner} corner ({x_pos}, {y_pos}) with transparency")

    return result


def save_images(image, base_name, output_dir=".", formats=['png']):
    """
    Save image in specified format(s).

    Args:
        image: Image to save
        base_name: Base filename (without extension)
        output_dir: Output directory
        formats: List of formats to save ('png', 'jpg', or both)

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    if 'png' in formats:
        png_path = output_dir / f"{base_name}.png"
        print(f"Saving PNG: {png_path}")
        cv2.imwrite(str(png_path), image)
        saved_paths.append(png_path)

    if 'jpg' in formats or 'jpeg' in formats:
        jpg_path = output_dir / f"{base_name}.jpg"
        print(f"Saving JPG: {jpg_path}")
        cv2.imwrite(str(jpg_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(jpg_path)

    return saved_paths


def process_pdf(pdf_path, output_dir=".", dpi=450, overlay_header=False, formats=['png']):
    """
    Main pipeline to process a single PDF.

    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory to save output images
        dpi: DPI for PDF conversion (higher = better quality)
        overlay_header: If True, extract header text and overlay on cropped image
        formats: List of output formats ('png', 'jpg', or both)
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")

    # Step 1: Extract header text bounding box if overlay is enabled
    header_bbox_data = None
    if overlay_header:
        header_bbox_data = extract_header_text_bbox(pdf_path)
        if not header_bbox_data:
            print("Warning: Could not extract header text, proceeding without overlay")

    # Step 2: Convert PDF to image
    original_image = pdf_to_image(pdf_path, dpi=dpi)
    print(f"Image size: {original_image.shape[1]}x{original_image.shape[0]} pixels")

    # Step 3: Extract text region from original image if overlay is enabled
    text_region = None
    if overlay_header and header_bbox_data:
        text_region = extract_text_region(original_image, header_bbox_data['bbox'], dpi)

    # Step 4: Detect border and crop
    cropped_image, border_coords = detect_and_crop_border(original_image)
    print(f"Cropped size: {cropped_image.shape[1]}x{cropped_image.shape[0]} pixels")

    # Step 5: Place text region if extracted
    if overlay_header and text_region is not None:
        # Find best corner for text placement
        best_corner = find_best_corner(cropped_image, dpi)
        # Place text region
        cropped_image = place_text_region(cropped_image, text_region, best_corner, dpi)

    # Step 6: Save in specified format(s)
    base_name = pdf_path.stem
    saved_paths = save_images(cropped_image, base_name, output_dir, formats)

    print(f"✓ Successfully processed {pdf_path.name}")
    for path in saved_paths:
        print(f"  {path.suffix.upper()[1:]}: {path}")

    return saved_paths


def main():
    """Process all PDF files in current directory or specified files."""
    import argparse

    parser = argparse.ArgumentParser(description='Process airport diagram PDFs')
    parser.add_argument('files', nargs='*', help='PDF files to process (default: all PDFs in current directory)')
    parser.add_argument('--output', '-o', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--dpi', type=int, default=450, help='DPI for PDF conversion (default: 450)')
    parser.add_argument('--overlay-text', action='store_true', help='Extract and overlay header text (airport ID and location) on cropped image')
    parser.add_argument('--format', '-f', choices=['png', 'jpg', 'both'], default='png',
                       help='Output format: png, jpg, or both (default: png)')

    args = parser.parse_args()

    # Parse formats
    if args.format == 'both':
        formats = ['png', 'jpg']
    else:
        formats = [args.format]

    # Get list of PDFs to process
    if args.files:
        pdf_files = [Path(f) for f in args.files]
    else:
        # Process all PDFs in current directory
        pdf_files = list(Path('.').glob('*.pdf'))

    if not pdf_files:
        print("No PDF files found to process")
        return 1

    print(f"Found {len(pdf_files)} PDF file(s) to process")

    # Process each PDF
    success_count = 0
    for pdf_file in pdf_files:
        try:
            process_pdf(pdf_file, output_dir=args.output, dpi=args.dpi,
                       overlay_header=args.overlay_text, formats=formats)
            success_count += 1
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(pdf_files)} files processed successfully")
    print(f"{'='*60}")

    return 0 if success_count == len(pdf_files) else 1


if __name__ == '__main__':
    sys.exit(main())
