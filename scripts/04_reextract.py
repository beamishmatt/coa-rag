"""
Re-extract data from documents with updated extraction prompts.

This script re-processes all documents to extract:
- Entities (people, organizations, locations, etc.)
- Claims/statements
- Events/timeline
- Relationships between people (NEW)
- Key facts

After extraction, it runs cross-document analysis to generate:
- Conflicts/inconsistencies
- Investigative notes (NEW) - factual observations for follow-up

Usage:
  python scripts/04_reextract.py              # Re-extract all documents
  python scripts/04_reextract.py --analyze    # Only run cross-document analysis (faster)
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from src.config import OPENAI_API_KEY, DEFAULT_MODEL
from src.extract import (
    load_extracted, save_extracted, _empty_extraction,
    extract_from_document, merge_extraction, detect_conflicts,
    deduplicate_extracted_data, analyze_investigative_notes, get_extraction_summary
)

# Check for flags
ANALYZE_ONLY = "--analyze" in sys.argv

client = OpenAI(api_key=OPENAI_API_KEY)
docs_dir = Path("data/docs")


def get_document_text(file_path: Path) -> str:
    """Extract text from a document file."""
    
    if file_path.suffix.lower() == '.txt':
        return file_path.read_text()
    
    elif file_path.suffix.lower() == '.pdf':
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            text_parts = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            
            # If very little text, might be scanned - try OCR
            if len(full_text.strip()) < 500:
                print(f"    Low text content, attempting OCR...")
                ocr_text = ocr_pdf(file_path)
                if len(ocr_text) > len(full_text):
                    return ocr_text
            
            return full_text
            
        except ImportError:
            print(f"    Warning: PyMuPDF not installed, trying pdfplumber...")
            try:
                import pdfplumber
                text_parts = []
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            text_parts.append(f"--- Page {i + 1} ---\n{text}")
                return "\n\n".join(text_parts)
            except ImportError:
                print(f"    Error: No PDF library available")
                return ""
    
    else:
        # Try to read as text
        try:
            return file_path.read_text()
        except:
            print(f"    Cannot read file type: {file_path.suffix}")
            return ""


def ocr_pdf(file_path: Path) -> str:
    """OCR a PDF file."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        images = convert_from_path(file_path, dpi=300)
        text_parts = []
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            text_parts.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"    OCR failed: {e}")
        return ""


def main():
    print("=" * 60)
    print("🔄 RE-EXTRACTING DOCUMENT DATA")
    print("=" * 60)
    
    if ANALYZE_ONLY:
        print("\n⚡ Running cross-document analysis only (--analyze flag)\n")
        all_data = load_extracted()
        
        if not all_data.get("documents"):
            print("❌ No documents found in extracted data. Run without --analyze first.")
            return
            
    else:
        # Full re-extraction
        all_data = _empty_extraction()
        
        # Find all document files
        doc_files = []
        for ext in ['*.pdf', '*.txt']:
            doc_files.extend(docs_dir.glob(ext))
        
        # Filter out .txt files that have matching .pdf (those are OCR outputs)
        doc_files = [f for f in doc_files 
                     if not (f.suffix == '.txt' and (docs_dir / f"{f.stem}.pdf").exists())]
        
        if not doc_files:
            print(f"❌ No documents found in {docs_dir}")
            return
        
        print(f"\n📂 Found {len(doc_files)} document(s) to process:\n")
        for f in doc_files:
            print(f"   - {f.name}")
        
        # Process each document
        print("\n" + "-" * 60)
        
        for i, file_path in enumerate(doc_files, 1):
            print(f"\n[{i}/{len(doc_files)}] Processing: {file_path.name}")
            
            # Get document text
            print("   📖 Extracting text...")
            doc_text = get_document_text(file_path)
            
            if not doc_text or len(doc_text.strip()) < 100:
                print(f"   ⚠️  Skipping - insufficient text content")
                continue
            
            print(f"   📊 Text length: {len(doc_text):,} characters")
            
            # Run extraction
            print("   🔍 Running LLM extraction (entities, claims, events, relationships)...")
            extraction = extract_from_document(client, DEFAULT_MODEL, doc_text, file_path.name)
            
            # Report what was extracted
            print(f"   ✅ Extracted:")
            print(f"      - {len(extraction.get('entities', []))} entities")
            print(f"      - {len(extraction.get('claims', []))} claims")
            print(f"      - {len(extraction.get('events', []))} events")
            print(f"      - {len(extraction.get('relationships', []))} relationships")
            print(f"      - {len(extraction.get('key_facts', []))} key facts")
            
            # Merge into all_data
            all_data = merge_extraction(all_data, extraction, file_path.name)
        
        # Deduplicate entities
        print("\n" + "-" * 60)
        print("\n🔗 Deduplicating entities...")
        all_data = deduplicate_extracted_data(all_data)
    
    # Cross-document analysis
    print("\n" + "-" * 60)
    print("\n🔬 Running cross-document analysis...")
    
    # Detect conflicts
    print("   📋 Detecting conflicts/inconsistencies...")
    conflicts = detect_conflicts(all_data, client, DEFAULT_MODEL)
    all_data["conflicts"] = conflicts
    print(f"   ✅ Found {len(conflicts)} potential conflicts")
    
    # Generate investigative notes
    print("   📝 Generating investigative notes (factual observations)...")
    notes = analyze_investigative_notes(client, DEFAULT_MODEL, all_data)
    all_data["investigative_notes"] = notes
    print(f"   ✅ Generated {len(notes)} investigative notes")
    
    # Save results
    print("\n" + "-" * 60)
    print("\n💾 Saving extracted data...")
    save_extracted(all_data)
    
    # Print summary
    summary = get_extraction_summary(all_data)
    print("\n" + "=" * 60)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"""
Summary:
  📄 Documents:           {summary['documents']}
  👤 Entities:            {summary['entities']}
  💬 Claims:              {summary['claims']}
  📅 Events:              {summary['events']}
  🔗 Relationships:       {summary['relationships']}
  📋 Investigative Notes: {summary['investigative_notes']}
  ⚠️  Conflicts:          {summary['conflicts']}
  📌 Key Facts:           {summary['key_facts']}
""")
    
    print("Data saved to: data/extracted.json")
    print("\nYou can now ask questions like:")
    print('  - "What relationships exist between people?"')
    print('  - "How is Seth Green connected to Amanda?"')
    print('  - "Show me investigative notes"')
    print('  - "What should we investigate further?"')


if __name__ == "__main__":
    main()




