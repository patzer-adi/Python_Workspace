"""Safe, in-memory processing of uploaded invention descriptions.

Nothing here touches disk or the patent database. To add PDF or DOCX support
later, write an extractor ``(bytes) -> str`` and register it in ``EXTRACTORS``.
"""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Callable

MAX_UPLOAD_BYTES = 200 * 1024  # 200 KB is plenty for a TXT invention description
MIN_TEXT_CHARS = 20
DEFAULT_CHUNK_CHARS = 1000  # ~250 tokens: fits MiniLM's 256-token window
DEFAULT_OVERLAP_CHARS = 150
MAX_SECTIONS_PER_SEARCH = 8  # caps retrieval cost on shared free hosting


class DocumentError(ValueError):
    """Raised with a user-presentable message when a document can't be used."""


def decode_text(raw: bytes) -> str:
    """Decode bytes to text, trying common encodings. Rejects binary content."""
    if not raw:
        raise DocumentError("The file is empty.")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16", errors="replace")
    else:
        if raw.count(b"\x00") > max(4, len(raw) // 50):
            raise DocumentError("This does not look like a plain-text file.")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
    return text


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[^\S\n]+", " ", text)  # collapse runs of spaces/tabs
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", text)  # control chars
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_txt(raw: bytes) -> str:
    return normalize_text(decode_text(raw))


EXTRACTORS: dict[str, Callable[[bytes], str]] = {
    ".txt": extract_txt,
    # ".pdf": extract_pdf,   # future
    # ".docx": extract_docx, # future
}


def extract_text(filename: str, raw: bytes) -> str:
    """Validate and extract text from an uploaded file's bytes."""
    if len(raw) > MAX_UPLOAD_BYTES:
        raise DocumentError(
            f"The file is {len(raw) / 1024:.0f} KB; the limit is {MAX_UPLOAD_BYTES // 1024} KB."
        )
    suffix = PurePath(filename or "").suffix.lower()
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        supported = ", ".join(sorted(EXTRACTORS))
        raise DocumentError(f"Unsupported file type '{suffix or 'unknown'}'. Supported: {supported}.")
    text = extractor(raw)
    if len(text) < MIN_TEXT_CHARS:
        raise DocumentError("The file contains too little text to search with.")
    return text


def _tail(text: str, n: int) -> str:
    """Last ~n characters of text, starting on a word boundary."""
    if n <= 0 or len(text) <= n:
        return ""
    piece = text[-n:]
    cut = piece.find(" ")
    return piece[cut + 1 :] if cut != -1 else ""


def _split_long(unit: str, max_chars: int) -> list[str]:
    """Split an over-long paragraph at sentence (then word) boundaries."""
    out: list[str] = []
    buf = ""
    for sentence in re.split(r"(?<=[.!?;])\s+", unit):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if buf:
                out.append(buf)
                buf = ""
            out.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if not sentence:
            continue
        if buf and len(buf) + 1 + len(sentence) > max_chars:
            out.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        out.append(buf)
    return out


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Paragraph-aware chunking with a small overlap between neighbouring chunks."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        units.extend([paragraph] if len(paragraph) <= max_chars else _split_long(paragraph, max_chars))

    chunks: list[str] = []
    buf = ""
    for unit in units:
        if buf and len(buf) + 2 + len(unit) > max_chars:
            chunks.append(buf)
            tail = _tail(buf, overlap)
            buf = f"{tail} {unit}".strip() if tail else unit
            if len(buf) > max_chars:
                buf = unit
        else:
            buf = f"{buf}\n\n{unit}" if buf else unit
    if buf:
        chunks.append(buf)
    return chunks


def select_sections(chunks: list[str], limit: int = MAX_SECTIONS_PER_SEARCH) -> list[str]:
    """Evenly sample at most ``limit`` chunks so long documents stay affordable."""
    if limit <= 0 or len(chunks) <= limit:
        return list(chunks)
    step = (len(chunks) - 1) / (limit - 1) if limit > 1 else 0
    indices = sorted({round(i * step) for i in range(limit)})
    return [chunks[i] for i in indices]
