from __future__ import annotations

import json
from pathlib import Path

import pytest

pytesseract = pytest.importorskip("pytesseract")  # OCR genuinely needs the [ocr] extra
PIL_Image = pytest.importorskip("PIL.Image")

# Skip the whole module when the tesseract binary itself is unavailable.
try:
    pytesseract.get_tesseract_version()
except Exception:  # pragma: no cover - environment without the tesseract binary
    pytest.skip("tesseract binary not installed", allow_module_level=True)

from crossmodalrag.capabilities import MissingModalityBackend  # noqa: E402
from crossmodalrag.db import connect, init_db  # noqa: E402
from crossmodalrag.ingest import image as image_module  # noqa: E402
from crossmodalrag.ingest.image import ingest_images  # noqa: E402

DOCUMENTS = (
    Path(__file__).resolve().parent / "fixtures" / "sample_seed" / "vault" / "documents"
)
SCREENSHOT = DOCUMENTS / "notes-screenshot.png"
DIAGRAM = DOCUMENTS / "architecture-diagram.png"


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
        ORDER BY s.source_uri, c.chunk_index
        """
    ).fetchall()
    return [dict(r) for r in rows]


# --- happy path ---------------------------------------------------------------


def test_ingest_screenshot_creates_image_source_with_ocr_metadata(conn):
    inserted = ingest_images(conn, image_path=SCREENSHOT)

    assert inserted >= 1
    sources = conn.execute("SELECT source_type, source_uri FROM sources").fetchall()
    assert len(sources) == 1
    assert sources[0]["source_type"] == "image"
    assert sources[0]["source_uri"].endswith("/notes-screenshot.png")

    chunks = _chunks(conn)
    text = " ".join(c["chunk_text"] for c in chunks)
    assert "reindex" in text.lower()
    for c in chunks:
        meta = json.loads(c["metadata_json"])
        assert meta["modality"] == "ocr"
        assert meta["source_type"] == "image"
        assert 0.0 < meta["ocr_confidence"] <= 1.0


def test_directory_ingestion_picks_images_and_ignores_pdf(conn):
    inserted = ingest_images(conn, image_path=DOCUMENTS)

    assert inserted >= 1
    source_types = {r["source_type"] for r in conn.execute("SELECT source_type FROM sources")}
    assert source_types == {"image"}  # spec.pdf in the same dir is NOT ingested here
    uris = {r["source_uri"] for r in conn.execute("SELECT source_uri FROM sources")}
    assert any(u.endswith("notes-screenshot.png") for u in uris)
    assert any(u.endswith("architecture-diagram.png") for u in uris)
    assert not any(u.endswith(".pdf") for u in uris)


# --- determinism / idempotency ------------------------------------------------


def test_reingest_unchanged_image_is_noop(conn):
    ingest_images(conn, image_path=SCREENSHOT)
    before = _snapshot(conn)

    inserted = ingest_images(conn, image_path=SCREENSHOT)

    assert inserted == 0
    assert _snapshot(conn) == before


def test_engine_version_bump_triggers_reingest(conn, monkeypatch):
    ingest_images(conn, image_path=SCREENSHOT)
    first_fp = conn.execute("SELECT source_fingerprint FROM sources").fetchone()[0]

    monkeypatch.setattr(image_module, "_ocr_engine_version", lambda: "999.0-test")
    inserted = ingest_images(conn, image_path=SCREENSHOT)

    assert inserted >= 1
    second_fp = conn.execute("SELECT source_fingerprint FROM sources").fetchone()[0]
    assert second_fp != first_fp
    assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1


# --- no-OCR-text image --------------------------------------------------------


def test_blank_image_registers_source_without_chunks(conn, tmp_path):
    blank = tmp_path / "blank.png"
    PIL_Image.new("RGB", (200, 80), "white").save(blank)

    inserted = ingest_images(conn, image_path=blank)

    assert inserted == 0
    assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM evidence_chunks").fetchone()[0] == 0
    # unchanged blank image re-ingest stays a no-op
    assert ingest_images(conn, image_path=blank) == 0


# --- CLI graceful degradation -------------------------------------------------


def test_ingest_images_cmd_exits_cleanly_when_backend_missing(monkeypatch, tmp_path):
    from crossmodalrag import cli

    monkeypatch.setenv("CMRAG_DB_PATH", str(tmp_path / "mem.db"))

    def _raise():
        raise MissingModalityBackend('OCR backend not installed. Run: pip install -e ".[ocr]"')

    monkeypatch.setattr("crossmodalrag.ingest.image.require_ocr", _raise)

    with pytest.raises(SystemExit, match=r"\[ocr\]"):
        cli.ingest_images_cmd([SCREENSHOT])


def _snapshot(conn):
    sources = conn.execute(
        "SELECT source_type, source_uri, source_fingerprint, title, metadata_json FROM sources ORDER BY id"
    ).fetchall()
    chunks = conn.execute(
        "SELECT source_id, chunk_index, chunk_text, metadata_json FROM evidence_chunks ORDER BY source_id, chunk_index"
    ).fetchall()
    return ([tuple(r) for r in sources], [tuple(r) for r in chunks])
