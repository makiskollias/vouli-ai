# pip install pymupdf python-dotenv
# (optional OCR) pip install pytesseract pillow
# and install tesseract on your OS if you enable OCR

import re
import json
from pathlib import Path

import fitz  # PyMuPDF

USE_OCR_FALLBACK = False  # set True if you want OCR when a page has no extractable text

def normalize_text(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"[ \t]+", " ", s)  # collapse spaces
    s = re.sub(r"\n{3,}", "\n\n", s)  # collapse blank lines
    # De-hyphenate line breaks (very common in Greek PDFs):
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    # Join broken lines (keeps paragraph breaks)
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)
    return s.strip()

def ocr_page(doc: fitz.Document, page_index: int) -> str:
    # OCR fallback (slow). Only used if USE_OCR_FALLBACK=True.
    import pytesseract
    from PIL import Image

    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img, lang="ell")  # needs Greek language pack
    return text

def chunk_text(text: str, target_chars: int = 1800, overlap_chars: int = 200):
    """
    Simple character-based chunker with overlap.
    Good enough for MVP; later you can do sentence/section-aware chunking.
    """
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + target_chars, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap_chars)
    return chunks

def pdf_to_chunks(
    pdf_path: str,
    source_id: str,
    source_url: str | None = None,
    chunk_chars: int = 1800,
    overlap_chars: int = 200
):
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    out = []
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        text = page.get_text("text") or ""
        if not text.strip() and USE_OCR_FALLBACK:
            text = ocr_page(doc, page_idx)

        text = normalize_text(text)
        if not text:
            continue

        page_chunks = chunk_text(text, target_chars=chunk_chars, overlap_chars=overlap_chars)
        for k, ch in enumerate(page_chunks):
            out.append({
                "id": f"{source_id}:p{page_idx+1}:c{k+1}",
                "source_id": source_id,
                "source_url": source_url,
                "page": page_idx + 1,
                "chunk_index": k + 1,
                "text": ch,
            })

    doc.close()
    return out

def save_jsonl(rows, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    pdf = "nomos.pdf"  # your PDF
    chunks = pdf_to_chunks(
        pdf_path=pdf,
        source_id="NOMOS_XXXX_YYYY",
        source_url=None,  # put official link if you have it
        chunk_chars=1800,
        overlap_chars=200
    )
    save_jsonl(chunks, "chunks.jsonl")
    print(f"Saved {len(chunks)} chunks to chunks.jsonl")
