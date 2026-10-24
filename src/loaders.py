"""Document loading module for the Personal AI Knowledge Assistant.

Supports loading plain text (.txt), Markdown (.md), and text-based PDF (.pdf) documents.
Extracts document records preserving relative source paths, page numbers (for PDFs),
and metadata while reporting unreadable or unextractable content.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from src.config import PROJECT_ROOT


@dataclass(frozen=True)
class DocumentRecord:
    """Represents extracted text from a source file or an individual page of a PDF.
    
    Attributes:
        text: Extracted text content.
        source: Normalized relative path from project root (e.g. 'data/documents/sample.md').
        page: 1-indexed page number for multi-page documents (like PDFs); None for text/markdown.
        metadata: Optional dictionary for additional file/context metadata.
    """
    text: str
    source: str
    page: Optional[int] = None
    metadata: dict = field(default_factory=dict)


def _normalize_source_path(file_path: Path, root: Optional[Path] = None) -> str:
    """Convert an absolute or relative path to a normalized relative path from project root."""
    base = root or PROJECT_ROOT
    try:
        rel = file_path.resolve().relative_to(base.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        # If path is outside base root (e.g. temp directory in tests), use its clean name or relative form
        return file_path.name


def load_text_file(file_path: Path, root: Optional[Path] = None) -> DocumentRecord:
    """Load a UTF-8 text or Markdown file.
    
    Args:
        file_path: Path to the .txt or .md file.
        root: Optional project root for relative path calculation.
        
    Returns:
        DocumentRecord containing the file's text, relative source path, and page=None.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or contains only whitespace.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding, exc.object, exc.start, exc.end,
            f"Failed to decode '{path.name}' as UTF-8: {exc.reason}"
        ) from exc

    stripped = content.strip()
    if not stripped:
        raise ValueError(f"File is empty or contains only whitespace: {path.name}")

    rel_source = _normalize_source_path(path, root)
    return DocumentRecord(
        text=stripped,
        source=rel_source,
        page=None,
        metadata={"file_size": path.stat().st_size, "extension": path.suffix.lower()},
    )


def load_pdf_file(
    file_path: Path, root: Optional[Path] = None
) -> tuple[list[DocumentRecord], list[str]]:
    """Load a text-based PDF file, extracting text page-by-page.

    Preserves original 1-indexed page numbers. Pages with no extractable text are
    reported as per-page warnings rather than causing the whole file to fail.
    Reasons for empty pages may include: blank pages, purely image-based (scanned)
    content, or pages with only vector graphics. OCR is not performed and is not
    available in this version.

    Args:
        file_path: Path to the .pdf file.
        root: Optional project root for relative path calculation.

    Returns:
        Tuple of:
          - List of DocumentRecord objects, one per page with extractable text.
          - List of warning strings for pages with no extractable text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the PDF has zero extractable text across all pages.
        pypdf.errors.PdfReadError: If the file is corrupted or cannot be parsed.
        pypdf.errors.PdfStreamError: If the PDF stream is malformed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    rel_source = _normalize_source_path(path, root)
    reader = PdfReader(str(path))

    records: list[DocumentRecord] = []
    page_warnings: list[str] = []
    total_pages = len(reader.pages)

    for index, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        cleaned = extracted.strip()
        if cleaned:
            records.append(
                DocumentRecord(
                    text=cleaned,
                    source=rel_source,
                    page=index,
                    metadata={"page_count": total_pages, "page_number": index},
                )
            )
        else:
            page_warnings.append(
                f"Page {index} of '{path.name}' yielded no extractable text. "
                "The page may be blank, image-only, or require OCR (not supported)."
            )

    if not records:
        raise ValueError(
            f"No extractable text found in any page of PDF '{path.name}' "
            f"({total_pages} page(s) checked). "
            "Pages may be blank or image-only. OCR is not supported in this version."
        )

    return records, page_warnings


def load_document(
    file_path: Path, root: Optional[Path] = None
) -> tuple[list[DocumentRecord], list[str]]:
    """Load any supported document (.txt, .md, .pdf) by extension.

    Args:
        file_path: Path to the document.
        root: Optional project root for relative path calculation.

    Returns:
        Tuple of:
          - List of DocumentRecord objects.
          - List of warning strings (e.g. empty PDF pages).

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in {".txt", ".md"}:
        return [load_text_file(path, root)], []
    elif ext == ".pdf":
        return load_pdf_file(path, root)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}' for file '{path.name}'. "
            "Supported formats are .txt, .md, and .pdf."
        )


class LoadDirResult(tuple):
    """Result of loading a document directory.

    Unpacks as `(records, errors)` for 2-element tuple compatibility with
    existing code and tests, while also exposing:
      - .records: list[DocumentRecord]
      - .errors: list[str] (file-level loading failures that should abort rebuilds)
      - .warnings: list[str] (page-level skipped-page warnings from valid multi-page PDFs)
    """

    def __new__(
        cls,
        records: list[DocumentRecord],
        errors: list[str],
        warnings: Optional[list[str]] = None,
    ):
        warns = warnings if warnings is not None else []
        inst = super().__new__(cls, (records, errors))
        inst.records = records
        inst.errors = errors
        inst.warnings = warns
        return inst


def load_documents_dir(
    dir_path: Path, root: Optional[Path] = None
) -> LoadDirResult:
    """Load all supported documents from a directory.

    Successful records, file-level errors, and page-level warnings are tracked
    distinctly:
      - records: Successfully parsed pages/documents.
      - errors: File-level load failures (corrupt files, encrypted PDFs, missing dir).
      - warnings: Non-fatal page-level reports (e.g. textless pages in valid PDFs).

    Args:
        dir_path: Path to the directory containing documents.
        root: Optional project root for relative path calculation.

    Returns:
        LoadDirResult unpacking as (records, errors) with .warnings attribute.
    """
    path = Path(dir_path)
    if not path.is_dir():
        return LoadDirResult([], [f"Directory not found: {path}"], [])

    records: list[DocumentRecord] = []
    errors: list[str] = []
    warnings: list[str] = []

    supported_extensions = {".txt", ".md", ".pdf"}
    files = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in supported_extensions]
    )

    if not files:
        errors.append(f"No supported documents found in directory: {path}")
        return LoadDirResult(records, errors, warnings)

    for file in files:
        try:
            docs, file_warnings = load_document(file, root)
            records.extend(docs)
            warnings.extend(file_warnings)  # page-level warnings kept separate
        except Exception as exc:
            errors.append(f"Failed to load '{file.name}': {exc}")

    return LoadDirResult(records, errors, warnings)
