# Airport Diagram Processing Pipeline

Automated pipeline to process airport diagram PDFs by converting them to high-resolution images and cropping to the diagram content area.

## Features

- Converts PDF to high-DPI image (default 450 DPI)
- Automatically detects and crops to the inside of the black border outline
- Optional text overlay: Extracts airport identifier and location text, overlays it on the cropped diagram
- Intelligent text placement: Analyzes diagram content to place text in the corner with more empty space
- Transparent text background: Blends seamlessly with colored diagram areas
- Saves results in PNG format (default), with options for JPG or both
- Batch processes multiple PDFs
- Preserves original filename with new extensions

## Requirements

- Python 3.7+
- poppler-utils (for PDF rendering)

### Install poppler-utils

**macOS:**
```bash
brew install poppler
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
Download from: https://github.com/oschwartz10612/poppler-windows/releases

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Process all PDFs in current directory:
```bash
python process_diagrams.py
```

### Process specific PDF files:
```bash
python process_diagrams.py KCDW-FF-TaxiDiagram.pdf KSFO-FF-TaxiDiagram.pdf
```

### Specify output directory:
```bash
python process_diagrams.py --output ./processed
```

### Adjust DPI (higher = better quality, larger files):
```bash
python process_diagrams.py --dpi 600
```

### Enable text overlay (extracts and places airport ID and location):
```bash
python process_diagrams.py --overlay-text
```

### Choose output format (png, jpg, or both):
```bash
python process_diagrams.py --format png    # default
python process_diagrams.py --format jpg    # JPEG only
python process_diagrams.py --format both   # both PNG and JPG
```

### Combined options:
```bash
python process_diagrams.py KCDW-FF-TaxiDiagram.pdf --output ./output --dpi 450 --overlay-text --format png
```

## Output

For each input PDF (e.g., `KCDW-FF-TaxiDiagram.pdf`), the pipeline generates:
- `KCDW-FF-TaxiDiagram.png` - High-quality PNG (default)
- `KCDW-FF-TaxiDiagram.jpg` - JPEG (95% quality, if `--format jpg` or `--format both` is specified)

All files contain the cropped diagram without the border. If `--overlay-text` is used, the airport identifier and location text will be placed in the optimal corner.

## How It Works

1. **PDF to Image**: Converts PDF to high-resolution image using pdf2image
2. **Text Extraction** (optional): Extracts airport identifier and location text from PDF header with original styling
3. **Border Detection**: Uses OpenCV to detect the black rectangular border
4. **Cropping**: Crops to inside of the border with a small margin
5. **Corner Analysis** (optional): Analyzes content density in top corners to find optimal text placement
6. **Text Overlay** (optional): Places extracted text with transparent background in the corner with more empty space
7. **Save**: Exports as PNG (default) or JPG with original filename

## Troubleshooting

**Error: "Unable to get page count"**
- Make sure poppler-utils is installed

**Border not detected correctly**
- Try adjusting the DPI (higher values may help)
- Check if the PDF has a clear black border outline

**Poor image quality**
- Increase DPI with `--dpi 600` or higher
- Note: Higher DPI = larger file sizes

**Text not extracted or placed incorrectly**
- Ensure the PDF has the standard header format: "KXXX: AIRPORT NAME" on first line
- The corner detection analyzes a 1350×900px region (at 450 DPI) to determine placement
- Text placement chooses the corner with less diagram content (more white space)
