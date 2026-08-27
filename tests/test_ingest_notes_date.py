from __future__ import annotations

from pathlib import Path


from crossmodalrag.db import connect, init_db
from crossmodalrag.ingest.notes import _parse_note_date, ingest_notes


def _note_timestamp(conn, source_uri: str) -> str:
    row = conn.execute(
        "SELECT timestamp FROM sources WHERE source_type = 'note' AND source_uri = ?",
        (source_uri,),
    ).fetchone()
    assert row is not None
    return str(row["timestamp"])


# --- pure parser --------------------------------------------------------------


def test_parse_frontmatter_date():
    text = "---\ntitle: Note\ndate: 2026-02-01\n---\n\n# Body\n"
    assert _parse_note_date(text) == "2026-02-01T00:00:00+00:00"


def test_parse_frontmatter_created_with_time():
    text = "---\ncreated: 2026-05-15T09:30:00+00:00\n---\nbody\n"
    assert _parse_note_date(text) == "2026-05-15T09:30:00+00:00"


def test_parse_body_date_line():
    text = "# Chunking Strategy\n\nDate: 2026-03-01\n\nbody\n"
    assert _parse_note_date(text) == "2026-03-01T00:00:00+00:00"


def test_parse_returns_none_without_date():
    assert _parse_note_date("# Just a heading\n\nNo date here.\n") is None


def test_parse_returns_none_on_malformed_date():
    assert _parse_note_date("---\ndate: not-a-date\n---\n") is None


def test_body_date_only_scanned_near_top():
    # A `Date:` far down the body is ignored (only the head is scanned).
    body = "# Heading\n" + "\n".join(f"line {i}" for i in range(20)) + "\nDate: 2026-01-01\n"
    assert _parse_note_date(body) is None


# --- ingestion wiring ---------------------------------------------------------


def test_ingest_uses_frontmatter_date_as_timestamp(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "dated.md"
    note.write_text("---\ndate: 2026-02-01\n---\n\nContent about chunking.\n", encoding="utf-8")

    conn = connect(tmp_path / "mem.db")
    try:
        init_db(conn)
        ingest_notes(conn, vault)
        assert _note_timestamp(conn, str(note.resolve())) == "2026-02-01T00:00:00+00:00"
    finally:
        conn.close()


def test_ingest_falls_back_to_mtime_when_dateless(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "plain.md"
    note.write_text("# Plain\n\nNo declared date.\n", encoding="utf-8")

    conn = connect(tmp_path / "mem.db")
    try:
        init_db(conn)
        ingest_notes(conn, vault)
        ts = _note_timestamp(conn, str(note.resolve()))
        # mtime-derived timestamp: an ISO string that is NOT the (absent) declared date.
        assert ts and "T" in ts
    finally:
        conn.close()


def test_ingest_with_date_is_idempotent(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "dated.md"
    note.write_text("---\ndate: 2026-02-01\n---\n\nContent.\n", encoding="utf-8")

    conn = connect(tmp_path / "mem.db")
    try:
        init_db(conn)
        assert ingest_notes(conn, vault) > 0
        # Unchanged content (incl. the date) → skipped, timestamp stable.
        assert ingest_notes(conn, vault) == 0
        assert _note_timestamp(conn, str(note.resolve())) == "2026-02-01T00:00:00+00:00"
    finally:
        conn.close()
