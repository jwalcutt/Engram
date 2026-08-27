from __future__ import annotations

import json
from pathlib import Path

import pytest

pypdf = pytest.importorskip("pypdf")  # PDF ingestion genuinely needs the [pdf] extra

from crossmodalrag.capabilities import MissingModalityBackend  # noqa: E402
from crossmodalrag.db import connect, init_db  # noqa: E402
from crossmodalrag.ingest import pdf as pdf_module  # noqa: E402
from crossmodalrag.ingest.pdf import ingest_pdf  # noqa: E402

FIXTURE_PDF = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "sample_seed"
    / "vault"
    / "documents"
    / "spec.pdf"
)


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "memory.db")
    init_db(connection)
    yield connection
    connection.close()


def _chunks(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT c.chunk_index, c.chunk_text, c.metadata_json, s.source_type, s.source_uri
        FROM evidence_chunks c JOIN sources s ON s.id = c.source_id
        ORDER BY c.chunk_index
        """
    ).fetchall()
    return [dict(r) for r in rows]


# --- happy path ---------------------------------------------------------------


def test_ingest_fixture_pdf_creates_pdf_source_with_page_locators(conn):
    inserted = ingest_pdf(conn, pdf_path=FIXTURE_PDF)

    assert inserted >= 1
    sources = conn.execute("SELECT source_type, source_uri FROM sources").fetchall()
    assert len(sources) == 1
    assert sources[0]["source_type"] == "pdf"
    assert sources[0]["source_uri"].endswith("/spec.pdf")

    chunks = _chunks(conn)
    assert chunks
    full_text = " ".join(c["chunk_text"] for c in chunks)
    assert "0.15" in full_text
    for c in chunks:
        meta = json.loads(c["metadata_json"])
        assert meta["modality"] == "pdf-page"
        assert meta["source_type"] == "pdf"
        assert isinstance(meta["page"], int) and meta["page"] >= 1


def test_directory_ingestion_finds_pdf(conn, tmp_path):
    # Copy the fixture into a directory tree and ingest the directory.
    nested = tmp_path / "docs" / "sub"
    nested.mkdir(parents=True)
    (nested / "copy.pdf").write_bytes(FIXTURE_PDF.read_bytes())

    inserted = ingest_pdf(conn, pdf_path=tmp_path / "docs")

    assert inserted >= 1
    (src,) = conn.execute("SELECT source_uri FROM sources").fetchall()
    assert src["source_uri"].endswith("/copy.pdf")


# --- page locator correctness -------------------------------------------------


def test_page_numbers_are_1_based_and_ordered(conn, tmp_path):
    pdf_path = _multi_page_pdf(tmp_path / "multi.pdf", ["Alpha page one", "Beta page two", "Gamma page three"])

    ingest_pdf(conn, pdf_path=pdf_path)

    chunks = _chunks(conn)
    pages_by_marker = {}
    for c in chunks:
        meta = json.loads(c["metadata_json"])
        for marker in ("Alpha", "Beta", "Gamma"):
            if marker in c["chunk_text"]:
                pages_by_marker[marker] = meta["page"]
    assert pages_by_marker == {"Alpha": 1, "Beta": 2, "Gamma": 3}


# --- determinism / idempotency ------------------------------------------------


def test_reingest_unchanged_pdf_is_noop(conn):
    ingest_pdf(conn, pdf_path=FIXTURE_PDF)
    before = _snapshot(conn)

    inserted = ingest_pdf(conn, pdf_path=FIXTURE_PDF)

    assert inserted == 0
    assert _snapshot(conn) == before


def test_extractor_version_bump_triggers_reingest(conn, monkeypatch):
    ingest_pdf(conn, pdf_path=FIXTURE_PDF)
    first_fp = conn.execute("SELECT source_fingerprint FROM sources").fetchone()[0]

    monkeypatch.setattr(pdf_module, "_extractor_version", lambda: "999.0-test")
    inserted = ingest_pdf(conn, pdf_path=FIXTURE_PDF)

    assert inserted >= 1  # re-derived because the fingerprint changed
    second_fp = conn.execute("SELECT source_fingerprint FROM sources").fetchone()[0]
    assert second_fp != first_fp
    # still a single source row (historical duplicate collapsed)
    assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1


# --- empty / encrypted edge cases --------------------------------------------


def test_blank_page_registers_source_without_chunks(conn, tmp_path):
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    blank = tmp_path / "blank.pdf"
    writer.write(str(blank))

    inserted = ingest_pdf(conn, pdf_path=blank)

    assert inserted == 0
    assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM evidence_chunks").fetchone()[0] == 0
    # re-ingest of the unchanged blank PDF stays a no-op
    assert ingest_pdf(conn, pdf_path=blank) == 0


def test_empty_password_encrypted_pdf_ingests(conn, tmp_path):
    src = _multi_page_pdf(tmp_path / "plain.pdf", ["Secret marker AAA"])
    reader = pypdf.PdfReader(str(src))
    writer = pypdf.PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt("")  # owner/empty user password
    enc = tmp_path / "enc.pdf"
    writer.write(str(enc))

    inserted = ingest_pdf(conn, pdf_path=enc)

    assert inserted >= 1
    assert any("AAA" in c["chunk_text"] for c in _chunks(conn))


def test_password_protected_pdf_raises_clear_error(conn, tmp_path):
    src = _multi_page_pdf(tmp_path / "plain2.pdf", ["text"])
    reader = pypdf.PdfReader(str(src))
    writer = pypdf.PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt("hunter2")
    enc = tmp_path / "locked.pdf"
    writer.write(str(enc))

    with pytest.raises(ValueError, match="password-protected"):
        ingest_pdf(conn, pdf_path=enc)


# --- CLI graceful degradation -------------------------------------------------


def test_ingest_pdf_cmd_exits_cleanly_when_backend_missing(monkeypatch, tmp_path):
    from crossmodalrag import cli

    monkeypatch.setenv("CMRAG_DB_PATH", str(tmp_path / "mem.db"))

    def _raise():
        raise MissingModalityBackend('PDF backend not installed. Run: pip install -e ".[pdf]"')

    # The connector calls require_pdf() first; simulate the extra being absent.
    monkeypatch.setattr("crossmodalrag.ingest.pdf.require_pdf", _raise)

    with pytest.raises(SystemExit, match=r"\[pdf\]"):
        cli.ingest_pdf_cmd([FIXTURE_PDF])


# --- helpers ------------------------------------------------------------------


def _multi_page_pdf(path: Path, pages_text: list[str]) -> Path:
    """Build a real multi-page PDF whose pages contain extractable marker text."""
    writer = pypdf.PdfWriter()
    for text in pages_text:
        reader = pypdf.PdfReader(_single_page_bytes(text))
        writer.add_page(reader.pages[0])
    writer.write(str(path))
    return path


def _single_page_bytes(text: str):
    import io

    # Reuse the dev-only fixture generator's hand-rolled minimal PDF (no reportlab).
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_xmodal_fixtures.py"
    spec = importlib.util.spec_from_file_location("xmodal_fixture_gen", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return io.BytesIO(module.build_minimal_pdf([text]))


def _snapshot(conn):
    sources = conn.execute(
        "SELECT source_type, source_uri, source_fingerprint, title, metadata_json FROM sources ORDER BY id"
    ).fetchall()
    chunks = conn.execute(
        "SELECT source_id, chunk_index, chunk_text, metadata_json FROM evidence_chunks ORDER BY source_id, chunk_index"
    ).fetchall()
    return ([tuple(r) for r in sources], [tuple(r) for r in chunks])
