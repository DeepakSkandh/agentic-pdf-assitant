"""Reader agent: PDF -> clean text chunks.
Digital pages: text extracted directly with PyMuPDF (fast, exact).
Scanned pages: rendered to an image and transcribed by Gemma 4 (no Tesseract needed)."""
import pymupdf as fitz

from config import CHUNK_CHARS, MIN_TEXT_CHARS, OCR_DPI
from llm import chat

OCR_PROMPT = ("Transcribe all the text on this page exactly as written. "
              "Keep headings and paragraph breaks. Output only the text, nothing else.")


def chunk_pages(pages: list[str], max_chars: int) -> list[str]:
    """Group pages into chunks of at most max_chars, splitting long pages at line breaks."""
    chunks, cur = [], ""
    for page in pages:
        while len(page) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            cut = page.rfind("\n", 0, max_chars)
            if cut < max_chars // 2:
                cut = max_chars
            chunks.append(page[:cut])
            page = page[cut:]
        if cur and len(cur) + len(page) > max_chars:
            chunks.append(cur)
            cur = ""
        cur += page + "\n\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


def reader_agent(state):
    doc = fitz.open(state["pdf_path"])
    pages_text, ocr_pages = [], 0
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if len(text.strip()) < MIN_TEXT_CHARS:
            png = page.get_pixmap(dpi=OCR_DPI).tobytes("png")
            text = chat(OCR_PROMPT, images=[png], temperature=0)
            ocr_pages += 1
        pages_text.append(f"[Page {i + 1}]\n{text.strip()}")
    n_pages = len(doc)
    doc.close()

    chunks = chunk_pages(pages_text, CHUNK_CHARS)
    return {
        "pages": n_pages,
        "chunks": chunks,
        "log": [f"Reader: {n_pages} pages ({ocr_pages} via OCR) -> {len(chunks)} chunks"],
    }
