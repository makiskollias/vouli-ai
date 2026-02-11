import re
import json
import os
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize_text(s: str) -> str:
    s = s.replace("\u00ad", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)
    return s.strip()

def get_embedding(text):
    # Δημιουργία διανύσματος για το κείμενο
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def chunk_text(text, target_chars=1200, overlap_chars=200):
    chunks = []
    i = 0
    while i < len(text):
        j = min(i + target_chars, len(text))
        chunks.append(text[i:j].strip())
        if j == len(text): break
        i = j - overlap_chars
    return chunks

def process_pdf(pdf_path, source_id):
    doc = fitz.open(pdf_path)
    all_chunks = []
    
    print(f"Επεξεργασία {pdf_path}...")
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        text = normalize_text(page.get_text("text"))
        if not text: continue
        
        page_chunks = chunk_text(text)
        for k, ch in enumerate(page_chunks):
            print(f"Δημιουργία embedding για σελίδα {page_idx+1}, chunk {k+1}...")
            all_chunks.append({
                "id": f"{source_id}:p{page_idx+1}:c{k+1}",
                "page": page_idx + 1,
                "text": ch,
                "embedding": get_embedding(ch) # Αποθήκευση εδώ!
            })
    doc.close()
    return all_chunks

if __name__ == "__main__":
    data = process_pdf("nomos.pdf", "NOMOS_5083_2024")
    with open("chunks.jsonl", "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Έτοιμο! Σώθηκαν {len(data)} chunks με embeddings.")
