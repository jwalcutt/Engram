from __future__ import annotations

import json

import pytest

from crossmodalrag.db import connect, init_db
from crossmodalrag.modality import (
    MODALITY_PDF_PAGE,
    build_chunk_metadata,
    parse_locator,
)
from crossmodalrag.retrieve import lexical
from crossmodalrag.retrieve.lexical import RetrievalHit
from crossmodalrag.retrieve.rerank import dedupe_hits, resolve_source_types


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "memory.db")
    init_db(connection)
    yield connection
    connection.close()


def _add_chunk(conn, *, source_type, source_uri, text, metadata: dict) -> int:
    cur = conn.execute(
        "INSERT INTO sources (source_type, source_uri, timestamp, title) VALUES (?, ?, ?, ?)",
        (source_type, source_uri, "2026-01-01T00:00:00+00:00", source_uri),
    )
    src = int(cur.lastrowid)
    cur = conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text, metadata_json) VALUES (?, ?, ?, ?)",
        (src, 0, text, json.dumps(metadata)),
    )
    return int(cur.lastrowid)


# --- modality -> source_type mapping ------------------------------------------


def test_resolve_source_types_maps_user_terms():
    assert resolve_source_types(None) is None
    assert resolve_source_types(["pdf"]) == {"pdf"}
    assert resolve_source_types(["text", "image"]) == {"note", "image"}
    with pytest.raises(ValueError, match="Unknown modality"):
        resolve_source_types(["holadeck"])


# --- chunk metadata surfaced on hits ------------------------------------------


def test_retrieval_surfaces_chunk_metadata_and_locator(conn):
    _add_chunk(
        conn,
        source_type="pdf",
        source_uri="/abs/spec.pdf",
        text="the retrieval gate abstains below the minimum score",
        metadata=build_chunk_metadata(modality=MODALITY_PDF_PAGE, source_type="pdf", page=4),
    )

    hits = lexical.retrieve(conn, query="retrieval gate abstains", top_k=5)

    assert hits
    loc = parse_locator(hits[0].chunk_metadata_json)
    assert loc is not None
    assert loc.modality == "pdf-page"
    assert loc.page == 4


# --- --modality (source_type) filter ------------------------------------------


def test_restrict_source_types_filters_by_modality(conn):
    shared = "embeddings backfill workflow detail"
    _add_chunk(
        conn,
        source_type="note",
        source_uri="/abs/note.md",
        text=shared + " in a note",
        metadata={"modality": "text", "source_type": "note"},
    )
    _add_chunk(
        conn,
        source_type="pdf",
        source_uri="/abs/spec.pdf",
        text=shared + " in a pdf page",
        metadata=build_chunk_metadata(modality=MODALITY_PDF_PAGE, source_type="pdf", page=1),
    )

    all_hits = lexical.retrieve(conn, query=shared, top_k=5)
    assert {h.source_type for h in all_hits} == {"note", "pdf"}

    pdf_only = lexical.retrieve(
        conn, query=shared, top_k=5, restrict_source_types={"pdf"}
    )
    assert pdf_only
    assert {h.source_type for h in pdf_only} == {"pdf"}


# --- conservative cross-modal dedupe ------------------------------------------


def _hit(chunk_id, source_type, text, score) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_id=chunk_id,
        source_type=source_type,
        source_uri=f"/abs/{chunk_id}",
        source_timestamp=None,
        title=None,
        chunk_index=0,
        chunk_text=text,
        score=score,
        lexical_score=score,
        recency_score=0.0,
    )


def test_dedupe_drops_near_identical_cross_modal_duplicate():
    same = "the quarterly plan covers retrieval evaluation and chunking work"
    hits = [
        _hit(1, "image", same, 0.9),     # OCR screenshot of the note (highest score)
        _hit(2, "note", same, 0.6),       # the note itself — near-identical text
        _hit(3, "pdf", "an entirely separate specification about rate limits", 0.4),
    ]

    kept = dedupe_hits(hits, threshold=0.95)

    assert [h.chunk_id for h in kept] == [1, 3]  # duplicate #2 dropped, distinct #3 kept


def test_dedupe_keeps_distinct_evidence():
    hits = [
        _hit(1, "note", "alpha beta gamma delta epsilon", 0.9),
        _hit(2, "pdf", "completely unrelated zeta eta theta iota", 0.5),
    ]

    kept = dedupe_hits(hits, threshold=0.95)

    assert [h.chunk_id for h in kept] == [1, 2]


def test_dedupe_max_kept_equals_full_dedupe_then_slice():
    # A pool with duplicates interleaved: early stop must yield exactly the same
    # top slice as deduping everything and slicing afterwards.
    dup = "identical duplicated content shared across two modalities exactly"
    hits = [
        _hit(1, "note", dup, 0.9),
        _hit(2, "image", dup, 0.8),  # near-duplicate of #1 — dropped either way
        _hit(3, "pdf", "unique alpha beta gamma", 0.7),
        _hit(4, "note", "unique delta epsilon zeta", 0.6),
        _hit(5, "git_commit", "unique eta theta iota", 0.5),
    ]

    for k in (1, 2, 3, 5):
        assert dedupe_hits(hits, threshold=0.95, max_kept=k) == dedupe_hits(
            hits, threshold=0.95
        )[:k]


def test_dedupe_max_kept_stops_scanning_after_quota():
    hits = [_hit(i, "note", f"unique text {i} tokens {i * 7}", 1.0 - i * 0.01) for i in range(50)]
    kept = dedupe_hits(hits, threshold=0.95, max_kept=5)
    assert len(kept) == 5
    assert [h.chunk_id for h in kept] == [0, 1, 2, 3, 4]
