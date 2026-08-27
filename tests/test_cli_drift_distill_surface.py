from __future__ import annotations

import hashlib
import json
import re
import sys

import pytest

from crossmodalrag import cli
from crossmodalrag.db import connect, init_db
from crossmodalrag.embed.store import embed_pending_chunks
from crossmodalrag.memory.distill import build_distilled
from crossmodalrag.memory.drift import build_drift
from crossmodalrag.memory.store import add_edge

WORD_RE = re.compile(r"[a-z0-9]+")
WIN0 = "2026-01-01T00:00:00+00:00"
WIN1 = "2026-02-10T00:00:00+00:00"


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


def _add_event(conn, title: str, time_start: str) -> int:
    uri = f"/v/{title.replace(' ', '_')}-{time_start[:10]}.md"
    cur = conn.execute("INSERT INTO sources (source_type, source_uri) VALUES ('note', ?)", (uri,))
    sid = int(cur.lastrowid)
    cur = conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, 0, ?)", (sid, title)
    )
    chunk_id = int(cur.lastrowid)
    cur = conn.execute(
        "INSERT INTO memory_nodes (level, node_type, title, time_start, time_end) "
        "VALUES (1, 'event', ?, ?, ?)",
        (title, time_start, time_start),
    )
    event_id = int(cur.lastrowid)
    add_edge(conn, 1, event_id, 0, chunk_id, "derived_from")
    conn.commit()
    return event_id


@pytest.fixture
def built_db(tmp_path, monkeypatch):
    """A DB with a drifting concept whose drift + distillation layers are built."""
    db_path = tmp_path / "memory.db"
    conn = connect(db_path)
    init_db(conn)
    e1 = _add_event(conn, "alpha beta chunk one", WIN0)
    e2 = _add_event(conn, "gamma delta chunk two", WIN1)
    cur = conn.execute("INSERT INTO memory_nodes (level, node_type, title) VALUES (3, 'concept', ?)", ("surface concept",))
    cid = int(cur.lastrowid)
    add_edge(conn, 3, cid, 1, e1, "contains")
    add_edge(conn, 3, cid, 1, e2, "contains")
    conn.commit()
    embed_pending_chunks(conn, StubEmbedProvider())
    build_drift(conn, StubEmbedProvider(), window_days=30.0)
    build_distilled(conn, StubEmbedProvider(), None, target_ratio=0.5)
    conn.close()
    monkeypatch.setenv("CMRAG_DB_PATH", str(db_path))
    return db_path, cid


def _run(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["mem", *argv])
    cli.main()


def test_memory_stats_shows_distilled_and_drift_counts(built_db, monkeypatch, capsys):
    _run(monkeypatch, ["memory-stats"])
    out = capsys.readouterr().out
    assert "Distilled nodes: 1" in out
    assert "Drift snapshots: 2" in out


def test_drift_json_emits_contract(built_db, monkeypatch, capsys):
    _, cid = built_db
    _run(monkeypatch, ["drift", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "drift" in payload and payload["drift"]
    entry = payload["drift"][0]
    assert entry["concept_id"] == cid
    assert len(entry["windows"]) == 2  # the movement trajectory
    assert entry["evidence_source_uri"]


def test_distill_json_emits_contract(built_db, monkeypatch, capsys):
    _run(monkeypatch, ["distill", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["distilled"]
    assert "overall_compression_ratio" in payload
    assert payload["distilled"][0]["core_count"] <= payload["distilled"][0]["full_count"]


def test_distill_text_view_runs(built_db, monkeypatch, capsys):
    _run(monkeypatch, ["distill"])
    out = capsys.readouterr().out
    assert "Distilled nodes" in out
    assert "ratio=" in out
