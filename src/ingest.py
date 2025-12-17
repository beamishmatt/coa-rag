from pathlib import Path
from openai import OpenAI
import tempfile
import shutil

def is_scanned_pdf(pdf_path: Path) -> bool:
    """
    Check if a PDF is scanned (image-based) by trying to extract text.
    Returns True if the PDF has little/no extractable text (likely scanned).
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_length = 0
        page_count = len(doc)
        for page in doc:
            text_length += len(page.get_text().strip())
        doc.close()
        
        # If very little text extracted, it's likely scanned
        # Threshold: less than 100 chars per page on average
        avg_chars_per_page = text_length / max(page_count, 1)
        return avg_chars_per_page < 100
    except ImportError:
        # PyMuPDF not installed, try pdfplumber as fallback
        try:
            import pdfplumber
            text_length = 0
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    text_length += len(text.strip())
                avg_chars_per_page = text_length / max(len(pdf.pages), 1)
            return avg_chars_per_page < 100
        except ImportError:
            # No PDF text extraction library, assume not scanned
            print(f"  Warning: Cannot check if PDF is scanned (install pymupdf or pdfplumber)")
            return False


def ocr_pdf_to_text(pdf_path: Path) -> str:
    """OCR a scanned PDF and return the extracted text."""
    from pdf2image import convert_from_path
    import pytesseract
    
    print(f"  📄 OCR processing: {pdf_path.name}")
    images = convert_from_path(pdf_path, dpi=300)
    
    full_text = []
    for i, image in enumerate(images):
        print(f"    Page {i+1}/{len(images)}...", end=" ", flush=True)
        text = pytesseract.image_to_string(image)
        full_text.append(f"--- Page {i+1} ---\n{text}")
        print(f"({len(text)} chars)")
    
    return "\n\n".join(full_text)


def prepare_file_for_upload(file_path: Path, temp_dir: Path) -> Path:
    """
    Prepare a file for upload. If it's a scanned PDF, OCR it first.
    Returns the path to upload (original or OCR'd version).
    """
    if file_path.suffix.lower() != '.pdf':
        return file_path
    
    # Check if it's a scanned PDF
    if is_scanned_pdf(file_path):
        print(f"  🔍 Detected scanned PDF: {file_path.name}")
        
        # OCR it
        text = ocr_pdf_to_text(file_path)
        
        # Save as text file
        txt_path = temp_dir / f"{file_path.stem}.txt"
        txt_path.write_text(text)
        print(f"  ✅ OCR complete: {len(text):,} characters extracted")
        
        return txt_path
    else:
        print(f"  📄 Text-based PDF: {file_path.name}")
        return file_path


def upload_files(client: OpenAI, docs_dir: Path, auto_ocr: bool = True) -> list[str]:
    """
    Upload files to OpenAI. Automatically OCRs scanned PDFs if auto_ocr=True.
    """
    file_ids = []
    
    # Create temp directory for OCR'd files
    temp_dir = Path(tempfile.mkdtemp(prefix="ocr_"))
    
    try:
        for p in sorted(docs_dir.glob("*")):
            if not p.is_file():
                continue
            
            # Skip already-processed text versions of PDFs
            if p.suffix == '.txt' and (docs_dir / f"{p.stem}.pdf").exists():
                print(f"  ⏭️  Skipping {p.name} (PDF version exists)")
                continue
                
            print(f"\nProcessing: {p.name}")
            
            # Prepare file (OCR if needed)
            if auto_ocr:
                upload_path = prepare_file_for_upload(p, temp_dir)
            else:
                upload_path = p
            
            # Upload to OpenAI
            f = client.files.create(file=open(upload_path, "rb"), purpose="assistants")
            file_ids.append(f.id)
            print(f"  ☁️  Uploaded: {f.id}")
            
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return file_ids

def attach_files_to_vector_store(client: OpenAI, vector_store_id: str, file_ids: list[str]) -> None:
    # simplest: attach individually (cookbook does this, parallelizable if you want)
    for fid in file_ids:
        client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=fid)

def wait_until_ready(client: OpenAI, vector_store_id: str, max_checks: int = 60) -> None:
    # The docs recommend waiting until file status is `completed` before querying.
    import time
    for _ in range(max_checks):
        vs = client.vector_stores.retrieve(vector_store_id)
        if vs.file_counts.in_progress == 0:
            return
        time.sleep(2)
    raise TimeoutError("Vector store still indexing after wait period")