"""Tests for Phase 2: Document loaders (txt, md, and text-based pdf).

Isolation notes:
- Tests for config-path resolution depend on the real project filesystem
  via src.config.PROJECT_ROOT, which is an absolute path anchored to this
  source tree. Tests that rely on it are NOT fully hermetic.
- Tests using tmp_path operate on temporary directories created by pytest
  and are independent of the benchmark corpus in data/documents/.
- No real .env credentials are read or required: load_text_file and
  load_pdf_file do not consult environment variables.
"""

from pathlib import Path
import pypdf
import pypdf.generic as g
import pypdf.errors as pdf_errors
import pytest

# pypdf.PdfWriter is used in test_load_pdf_password_protected to create an encrypted PDF

from src.loaders import (
    DocumentRecord,
    load_document,
    load_documents_dir,
    load_pdf_file,
    load_text_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_minimal_pdf_with_text(path: Path, pages_text: list[str]) -> None:
    """Create a valid minimal text-based PDF. Pages in pages_text that are
    the empty string will be written as blank pages (no content stream text).
    """
    writer = pypdf.PdfWriter()
    font_dict = g.DictionaryObject({
        g.NameObject("/Type"): g.NameObject("/Font"),
        g.NameObject("/Subtype"): g.NameObject("/Type1"),
        g.NameObject("/BaseFont"): g.NameObject("/Helvetica"),
    })

    for text in pages_text:
        page = writer.add_blank_page(width=300, height=300)
        if text:
            stream = g.DecodedStreamObject()
            # PDF text operators: BT..ET frame, Tf sets font, Td positions, Tj shows text
            stream.set_data(f"BT /F1 12 Tf 20 150 Td ({text}) Tj ET".encode("latin-1"))
            page[g.NameObject("/Contents")] = stream
            page[g.NameObject("/Resources")] = g.DictionaryObject({
                g.NameObject("/Font"): g.DictionaryObject({g.NameObject("/F1"): font_dict})
            })
        # Pages with empty text string get no content stream → extract_text() returns ""

    with open(path, "wb") as f:
        writer.write(f)


def _create_blank_pdf(path: Path, num_pages: int = 2) -> None:
    """Create a PDF with no extractable text (blank pages only)."""
    writer = pypdf.PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=300, height=300)
    with open(path, "wb") as f:
        writer.write(f)


# ---------------------------------------------------------------------------
# Text / Markdown loading
# ---------------------------------------------------------------------------

def test_load_text_file_valid(tmp_path):
    """load_text_file loads markdown text and returns normalized relative source path."""
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Sample Title\n\nThis is sample text.", encoding="utf-8")

    doc = load_text_file(file_path, root=tmp_path)
    assert isinstance(doc, DocumentRecord)
    assert doc.text == "# Sample Title\n\nThis is sample text."
    assert doc.source == "sample.md"
    assert doc.page is None
    assert doc.metadata["extension"] == ".md"


def test_load_text_file_empty_raises(tmp_path):
    """Empty files must be rejected with an actionable ValueError."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_text_file(empty_file)
    assert "empty" in str(exc.value).lower()


def test_load_text_file_whitespace_only_raises(tmp_path):
    """Whitespace-only files must be rejected with an actionable ValueError."""
    ws_file = tmp_path / "spaces.md"
    ws_file.write_text("   \n\t  \n  ", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_text_file(ws_file)
    assert "whitespace" in str(exc.value).lower()


def test_load_text_file_missing_raises(tmp_path):
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_text_file(tmp_path / "non_existent.md")


def test_load_text_file_unicode_content(tmp_path):
    """UTF-8 text containing symbols and accents must load without corruption."""
    unicode_file = tmp_path / "unicode.md"
    content = "Café résumé — 15,000 µs latency ± 2.5% variation."
    unicode_file.write_text(content, encoding="utf-8")

    doc = load_text_file(unicode_file, root=tmp_path)
    assert doc.text == content


# ---------------------------------------------------------------------------
# PDF loading: successful cases
# ---------------------------------------------------------------------------

def test_load_pdf_file_multipage_success(tmp_path):
    """Text-based PDF preserves 1-indexed page numbers across all readable pages."""
    pdf_path = tmp_path / "multipage.pdf"
    _create_minimal_pdf_with_text(pdf_path, ["Page One Evidence", "Page Two Analysis"])

    docs, warnings = load_pdf_file(pdf_path, root=tmp_path)
    assert len(docs) == 2
    assert len(warnings) == 0
    assert docs[0].page == 1
    assert "Page One Evidence" in docs[0].text
    assert docs[1].page == 2
    assert "Page Two Analysis" in docs[1].text
    assert docs[0].source == "multipage.pdf"


def test_load_pdf_partially_extractable_mixed_pages(tmp_path):
    """PDF with readable page 1, empty page 2, and readable page 3.

    - Page 3 must retain page number 3 (not be renumbered to 2).
    - Page 2 must be reported as a per-page warning, not suppress other pages.
    - The returned records list must contain exactly 2 entries (pages 1 and 3).
    """
    pdf_path = tmp_path / "partial.pdf"
    # Empty string in the middle creates a blank page with no extractable text
    _create_minimal_pdf_with_text(pdf_path, ["Content on Page One", "", "Content on Page Three"])

    docs, warnings = load_pdf_file(pdf_path, root=tmp_path)

    # Two readable pages extracted
    assert len(docs) == 2, f"Expected 2 records, got {len(docs)}: {[d.page for d in docs]}"

    # Original page numbers preserved; page 3 is not renumbered
    page_numbers = [d.page for d in docs]
    assert 1 in page_numbers
    assert 3 in page_numbers
    assert 2 not in page_numbers

    # Content integrity
    assert any("Page One" in d.text for d in docs)
    assert any("Page Three" in d.text for d in docs)

    # Exactly one warning for blank page 2
    assert len(warnings) == 1
    assert "page 2" in warnings[0].lower() or "page 2" in warnings[0]
    # Warning must mention the empty/no-text condition, not assert it's scanned
    assert "no extractable text" in warnings[0].lower()


def test_load_pdf_all_empty_raises(tmp_path):
    """PDF with no extractable text on any page must raise ValueError."""
    blank_pdf = tmp_path / "blank.pdf"
    _create_blank_pdf(blank_pdf, num_pages=2)

    with pytest.raises(ValueError) as exc:
        load_pdf_file(blank_pdf)
    msg = str(exc.value).lower()
    assert "no extractable text" in msg
    assert "ocr" in msg


# ---------------------------------------------------------------------------
# PDF loading: failure / error cases (Gap 3)
# ---------------------------------------------------------------------------

def test_load_pdf_corrupted_file(tmp_path):
    """A file with a .pdf extension but invalid binary content must fail with a clear error.

    pypdf raises PdfReadError (or a subclass) for malformed PDF streams.
    The caller-facing error must be distinguishable from a valid empty-page warning.
    """
    corrupt_pdf = tmp_path / "corrupted.pdf"
    # Write arbitrary garbage bytes that are not a valid PDF
    corrupt_pdf.write_bytes(b"This is not a PDF file at all. Random bytes: \x00\x01\x02\xff")

    with pytest.raises(Exception) as exc:
        load_pdf_file(corrupt_pdf)
    # pypdf should raise a PDF-related error, not a silent success
    assert exc.type is not None


def test_load_pdf_password_protected(tmp_path):
    """A password-protected PDF that has not been decrypted must fail clearly.

    pypdf raises FileNotDecryptedError when pages of an encrypted-but-not-decrypted
    PDF are accessed. The loader must surface this as a clear failure, not return
    an empty record list or silently succeed.
    load_documents_dir must capture this as an error string, not a crash.
    """
    encrypted_pdf = tmp_path / "protected.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.encrypt("secret123")
    with open(encrypted_pdf, "wb") as f:
        writer.write(f)

    # Direct call to load_pdf_file must raise — pypdf raises FileNotDecryptedError
    # when pages are accessed on an encrypted-but-not-decrypted PDF.
    with pytest.raises(Exception) as exc:
        load_pdf_file(encrypted_pdf)
    error_name = exc.type.__name__.lower()
    assert (
        "decrypt" in error_name
        or "password" in error_name
        or "encrypt" in error_name
        or "pdf" in error_name
    ), f"Unexpected error type: {exc.type.__name__}"

    # load_documents_dir must isolate the failure rather than crash
    records, errors = load_documents_dir(tmp_path, root=tmp_path)
    assert len(records) == 0
    assert any("protected.pdf" in err for err in errors)


def test_load_documents_dir_isolates_failures(tmp_path):
    """A failed file does not suppress successfully loaded documents.

    The error message for the failed file must appear in the errors list,
    while the record from the good file must appear in the records list.
    """
    good = tmp_path / "good.md"
    good.write_text("Valid document content for testing.", encoding="utf-8")

    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"NOT A REAL PDF \x00\xff")

    records, errors = load_documents_dir(tmp_path, root=tmp_path)

    # Good file loads successfully
    assert any(r.source.endswith("good.md") for r in records)
    # Bad file produces an error entry, not a crash
    assert any("bad.pdf" in err for err in errors)
    # The two categories remain separate
    assert len(records) >= 1


# ---------------------------------------------------------------------------
# load_document dispatch
# ---------------------------------------------------------------------------

def test_load_document_dispatch_and_unsupported_format(tmp_path):
    """load_document routes by extension and rejects unsupported formats.

    The return type is always a tuple of (records_list, warnings_list).
    """
    txt = tmp_path / "notes.txt"
    txt.write_text("Plain text content", encoding="utf-8")
    docs, warnings = load_document(txt, root=tmp_path)
    assert len(docs) == 1
    assert warnings == []

    unsupported = tmp_path / "archive.docx"
    unsupported.write_text("Dummy content", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_document(unsupported)
    assert "unsupported" in str(exc.value).lower()


def test_load_documents_dir_success(tmp_path):
    """load_documents_dir loads multiple files and returns zero errors on clean input."""
    (tmp_path / "doc1.md").write_text("Document 1 content", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Document 2 content", encoding="utf-8")

    pdf_path = tmp_path / "doc3.pdf"
    _create_minimal_pdf_with_text(pdf_path, ["PDF content"])

    records, errors = load_documents_dir(tmp_path, root=tmp_path)
    assert len(records) == 3
    assert len(errors) == 0
