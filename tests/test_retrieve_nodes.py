from __future__ import annotations

import hashlib
import re

import pytest

from crossmodalrag.db import connect, init_db
from crossmodalrag.embed.store import embed_pending_nodes
from crossmodalrag.memory.store import add_edge, insert_node
from crossmodalrag.retrieve.nodes import (
    candidate_chunk_ids,
    drilldown_source_uris,
    retrieve_nodes,
)

WORD_RE = re.compile(r"[a-z0-9]+")


class StubEmbedProvider:
    def __init__(self, dim: int = 64, name: str = "stub-embed-v1") -> None:
        self.dim = dim
        self.name = name

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in WORD_RE.findall(text.lower()):
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            out.append(vec)
        return out


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "memory.db")
    init_db(connection)
    yield connection
    connection.close()


def _event(conn, title: str, uri: str) -> tuple[int, int]:
    cur = conn.execute("INSERT INTO sources (source_type, source_uri) VALUES ('note', ?)", (uri,))
    sid = int(cur.lastrowid)
    cur = conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, 0, ?)", (sid, title)
    )
    chunk_id = int(cur.lastrowid)
    event_id = insert_node(conn, level=1, node_type="event", title=title)
    add_edge(conn, 1, event_id, 0, chunk_id, "derived_from")
    return event_id, chunk_id


def _concept(conn, title: str, members: list[int], centrality: float | None = None) -> int:
    cid = insert_node(conn, level=3, node_type="concept", title=title)
    for ev in members:
        add_edge(conn, 3, cid, 1, ev, "contains")
    if centrality is not None:
        conn.execute("UPDATE memory_nodes SET centrality = ? WHERE id = ?", (centrality, cid))
    conn.commit()
    return cid


def test_retrieve_nodes_returns_target_level_ranked(conn) -> None:
    e1, c1 = _event(conn, "parser bounds fix", "/v/a.md")
    e2, c2 = _event(conn, "cooking pasta recipe", "/v/b.md")
    cParser = _concept(conn, "parser bounds work", [e1])
    _concept(conn, "cooking recipes", [e2])  # distractor: must not outrank the parser concept
    embed_pending_nodes(conn, StubEmbedProvider(), level=3, node_type="concept")

    hits = retrieve_nodes(conn, "parser bounds", level="concept", top_k=5, provider=StubEmbedProvider())
    assert hits
    assert all(h.node_type == "concept" for h in hits)
    assert hits[0].node_id == cParser
    assert hits[0].vector_score > 0.0


def test_retrieve_nodes_lexical_fallback_without_vectors(conn) -> None:
    e1, _ = _event(conn, "fingerprint deterministic ingestion", "/v/a.md")
    e2, _ = _event(conn, "weather forecast", "/v/b.md")
    cA = _concept(conn, "fingerprint deterministic ingestion", [e1])
    _concept(conn, "weather forecast", [e2])
    # No node embeddings + provider=None -> lexical-only ranking.
    hits = retrieve_nodes(conn, "fingerprint", level="concept", provider=None)
    assert hits[0].node_id == cA
    assert all(h.vector_score == 0.0 for h in hits)


def test_centrality_breaks_ties(conn) -> None:
    e1, _ = _event(conn, "alpha topic", "/v/a.md")
    e2, _ = _event(conn, "alpha topic", "/v/b.md")
    low = _concept(conn, "alpha topic", [e1], centrality=0.1)
    high = _concept(conn, "alpha topic", [e2], centrality=0.9)
    # Identical titles (equal vector+lexical); centrality should order high first.
    hits = retrieve_nodes(conn, "alpha topic", level="concept", provider=None)
    assert [h.node_id for h in hits][:2] == [high, low]


def test_drilldown_recovers_evidence(conn) -> None:
    e1, c1 = _event(conn, "parser bounds fix", "/v/a.md")
    e2, c2 = _event(conn, "parser bounds check", "/v/b.md")
    _concept(conn, "parser bounds work", [e1, e2])
    embed_pending_nodes(conn, StubEmbedProvider(), level=3, node_type="concept")

    hits = retrieve_nodes(conn, "parser bounds", level="concept", provider=StubEmbedProvider())
    assert candidate_chunk_ids(conn, hits) >= {c1, c2}
    assert set(drilldown_source_uris(conn, hits)) == {"/v/a.md", "/v/b.md"}


def test_unknown_level_rejected(conn) -> None:
    with pytest.raises(ValueError):
        retrieve_nodes(conn, "q", level="galaxy")
