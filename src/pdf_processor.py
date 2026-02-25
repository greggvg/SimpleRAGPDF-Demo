"""PDF extraction and text processing utilities"""

from typing import List
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from all pages
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        pages.append(doc.load_page(i).get_text())
    return "\n".join(pages)


def _split_by_separator(text: str, sep: str) -> List[str]:
    """
    Split text by a separator while preserving the separator.
    
    Args:
        text: Text to split
        sep: Separator string
        
    Returns:
        List of text parts
    """
    if sep == "":
        return list(text)
    parts = text.split(sep)
    out = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            out.append(p + sep)
        else:
            out.append(p)
    return out


def recursive_text_splitter(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    separators: List[str] = None,
    min_chunk_chars: int = 200
) -> List[str]:
    """
    Split text into chunks with overlap using recursive separator-based splitting.
    
    This mimics LangChain's RecursiveCharacterTextSplitter behavior without requiring
    the library.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        separators: List of separators to try in order (default: ["\n\n", "\n", ". ", " ", ""])
        min_chunk_chars: Minimum characters to keep a chunk (otherwise merge with previous)
        
    Returns:
        List of text chunks
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    text = (text or "").strip()
    if not text:
        return []

    def _recurse(t: str, seps: List[str]) -> List[str]:
        if len(t) <= chunk_size or not seps:
            return [t]

        sep = seps[0]
        parts = _split_by_separator(t, sep)

        if len(parts) == 1:
            return _recurse(t, seps[1:])

        out, buf = [], ""
        for part in parts:
            if len(part) > chunk_size and seps[1:]:
                if buf.strip():
                    out.append(buf)
                    buf = ""
                out.extend(_recurse(part, seps[1:]))
                continue

            if len(buf) + len(part) <= chunk_size:
                buf += part
            else:
                if buf.strip():
                    out.append(buf)
                buf = part

        if buf.strip():
            out.append(buf)

        return out

    pieces = _recurse(text, separators)

    cleaned = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if cleaned and len(p) < min_chunk_chars:
            cleaned[-1] = (cleaned[-1].rstrip() + " " + p).strip()
        else:
            cleaned.append(p)

    chunks = []
    for p in cleaned:
        if not chunks:
            chunks.append(p)
            continue
        overlap_text = chunks[-1][-chunk_overlap:] if chunk_overlap > 0 else ""
        merged = (overlap_text + p).strip()
        if len(merged) > chunk_size:
            merged = merged[-chunk_size:]
        chunks.append(merged)

    return chunks
