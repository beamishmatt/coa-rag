"""OCR a scanned PDF and save as searchable text."""
import sys
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract

def ocr_pdf(pdf_path: Path, output_path: Path = None) -> str:
    """Convert scanned PDF to text using OCR."""
    print(f"Processing: {pdf_path}")
    
    # Convert PDF pages to images
    print("Converting PDF to images...")
    images = convert_from_path(pdf_path, dpi=300)
    print(f"Found {len(images)} pages")
    
    # OCR each page
    full_text = []
    for i, image in enumerate(images):
        print(f"  OCR page {i+1}/{len(images)}...", end=" ")
        text = pytesseract.image_to_string(image)
        full_text.append(f"--- Page {i+1} ---\n{text}")
        print(f"({len(text)} chars)")
    
    result = "\n\n".join(full_text)
    
    # Save to file if output path provided
    if output_path:
        output_path.write_text(result)
        print(f"\nSaved to: {output_path}")
        print(f"Total: {len(result):,} characters")
    
    return result

if __name__ == "__main__":
    pdf_file = Path("data/docs/Crime Scene Services - MSP (10-18-2011).pdf")
    output_file = Path("data/docs/Crime Scene Services - MSP (10-18-2011).txt")
    
    text = ocr_pdf(pdf_file, output_file)
    
    # Show a preview
    print("\n" + "="*60)
    print("PREVIEW (first 2000 chars):")
    print("="*60)
    print(text[:2000])




