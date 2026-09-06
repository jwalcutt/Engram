"""Stable JSON output contracts (the UI/API boundary).

Pins the documented key sets for every read-view `--json` contract — unit tests on the library
`*_to_dict` / `list_*` / `memory_stats` helpers, plus end-to-end shape tests through `cli.main`.
"""

from __future__ import annotations

import json
import sys

import pytest

from crossmodalrag import cli
from crossmodalrag.db import connect, init_db
from crossmodalrag.evaluation import EvalSummary, EvalQueryResult, eval_summary_to_dict
from crossmodalrag.memory.concepts import list_concept_views
from crossmodalrag.memory.episodes import list_episode_timeline
from crossmodalrag.memory.forgetting import ForgettingRisk, forgetting_risk_to_dict
from crossmodalrag.memory.integrity import memory_stats
from crossmodalrag.memory.recall import RecallCard, recall_card_to_dict
from crossmodalrag.memory.store import add_edge
from crossmodalrag.usage.store import record_usage_event
from crossmodalrag.usage.strength import UsageSummary, usage_summary_to_dict


# --- unit: exact key sets (additive-only guard) -------------------------------


def test_forgetting_risk_to_dict_keys():
    item = ForgettingRisk(
        node_id=1, level=3, node_type="concept", title="c", importance=0.5, staleness=0.5,
        risk=0.25, confidence=1.0, support=2, last_touch=None, evidence_source_uris=["/a"],
    )
    assert set(forgetting_risk_to_dict(item)) == {
        "node_id", "level", "node_type", "title", "risk", "importance", "staleness",
        "confidence", "support", "last_touch", "evidence_source_uris",
    }


def test_recall_card_to_dict_keys():
    card = RecallCard(
        node_id=1, level=3, node_type="concept", title="c", question="q?", answer="a",
        risk=0.1, confidence=1.0, evidence_source_uris=["/a"], generated_by="fallback",
    )
    assert set(recall_card_to_dict(card)) == {
        "node_id", "level", "node_type", "title", "question", "answer", "risk",
        "confidence", "generated_by", "evidence_source_uris",
    }


def test_usage_summary_to_dict_keys():
    s = UsageSummary(target_kind="chunk", target_id=1, count=2, last_event_at="2026-01-01", strength=1.5)
    assert set(usage_summary_to_dict(s)) == {
        "target_kind", "target_id", "count", "last_event_at", "strength",
    }


def test_eval_summary_to_dict_keys_and_misses():
    miss = EvalQueryResult("q-miss", ["/g"], [], None, False, False)
    hit = EvalQueryResult("q-hit", ["/g"], ["/g"], 1, True, True)
    summary = EvalSummary(2, 5, 0.5, 0.5, 0.5, [hit, miss])
    payload = eval_summary_to_dict(summary)
    assert set(payload) == {
        "query_count", "top_k", "recall_at_k", "mrr_at_k", "citation_hit_rate", "misses",
    }
    assert payload["misses"] == ["q-miss"]  # only the no-hit query


# --- DB fixture for the read views + e2e --------------------------------------


def _add_event_with_chunk(conn, text: str) -> tuple[int, int, str]:
    uri = f"/v/{text.replace(' ', '_')}.md"
    cur = conn.execute(
        "INSERT INTO sources (source_type, source_uri, timestamp) VALUES ('note', ?, ?)",
        (uri, "2026-01-01T00:00:00+00:00"),
    )
    sid = int(cur.lastrowid)
    cur = conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, 0, ?)", (sid, text)
    )
    chunk_id = int(cur.lastrowid)
    cur = conn.execute(
        "INSERT INTO memory_nodes (level, node_type, title, time_start) VALUES (1, 'event', ?, ?)",
        (text, "2026-01-01T00:00:00+00:00"),
    )
    eid = int(cur.lastrowid)
    add_edge(conn, 1, eid, 0, chunk_id, "derived_from")
    return eid, chunk_id, uri


@pytest.fixture
def built_db(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    conn = connect(db_path)
    init_db(conn)
    e1, c1, uri1 = _add_event_with_chunk(conn, "parser bounds check fix")
    e2, c2, _ = _add_event_with_chunk(conn, "parser overflow guard")
    # L3 concept + L2 episode over the events.
    cur = conn.execute(
        "INSERT INTO memory_nodes (level, node_type, title, centrality) VALUES (3, 'concept', ?, 0.9)",
        ("Parser hardening",),
    )
    cid = int(cur.lastrowid)
    add_edge(conn, 3, cid, 1, e1, "contains")
    add_edge(conn, 3, cid, 1, e2, "contains")
    cur = conn.execute(
        "INSERT INTO memory_nodes (level, node_type, title, time_start, time_end) "
        "VALUES (2, 'episode', ?, ?, ?)",
        ("Parser session", "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"),
    )
    ep = int(cur.lastrowid)
    add_edge(conn, 2, ep, 1, e1, "contains")
    add_edge(conn, 2, ep, 1, e2, "contains")
    # Usage history + an eval query with gold.
    record_usage_event(conn, "chunk", c1, "retrieval_hit", event_at="2026-01-01T00:00:00+00:00")
    conn.execute(
        "INSERT INTO queries_eval (query_text, expected_source_uris) VALUES (?, ?)",
        ("[t] parser bounds check fix", json.dumps([uri1])),
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("CMRAG_DB_PATH", str(db_path))
    return db_path, cid, ep


# --- unit: library read functions ---------------------------------------------


def test_list_concept_views(built_db):
    db_path, cid, _ = built_db
    conn = connect(db_path)
    try:
        views = list_concept_views(conn, top=10)
    finally:
        conn.close()
    assert views and set(views[0]) == {"node_id", "title", "centrality", "members"}
    assert views[0]["node_id"] == cid and views[0]["members"] == 2


def test_list_episode_timeline(built_db):
    db_path, _, ep = built_db
    conn = connect(db_path)
    try:
        rows = list_episode_timeline(conn, limit=10)
    finally:
        conn.close()
    assert rows and set(rows[0]) == {"node_id", "title", "time_start", "time_end", "members"}
    assert rows[0]["node_id"] == ep


def test_memory_stats_contract(built_db):
    db_path, _, _ = built_db
    conn = connect(db_path)
    try:
        stats = memory_stats(conn)
    finally:
        conn.close()
    assert set(stats) == {
        "total_nodes", "nodes_by_level", "nodes_by_type", "edges", "relates_edges",
        "node_embeddings", "distilled_nodes", "drift_snapshots", "top_central", "integrity",
    }
    assert set(stats["integrity"]) == {
        "unsupported_count", "unsupported_ids", "dangling_count", "dangling_ids",
    }
    assert stats["nodes_by_level"]["3"] == 1  # the concept


# --- e2e: cli.main --json shape (offline) -------------------------------------


def _run_json(monkeypatch, capsys, argv: list[str]) -> dict:
    monkeypatch.setattr(sys, "argv", ["mem", *argv])
    cli.main()
    return json.loads(capsys.readouterr().out)


def test_cli_forgetting_json(built_db, monkeypatch, capsys):
    payload = _run_json(monkeypatch, capsys, ["forgetting", "--level", "concept", "--json"])
    assert payload["level"] == "concept"
    assert payload["forgetting"] and "evidence_source_uris" in payload["forgetting"][0]


def test_cli_usage_json(built_db, monkeypatch, capsys):
    payload = _run_json(monkeypatch, capsys, ["usage", "--json"])
    assert set(payload) == {"tracking_enabled", "total_events", "by_type", "top_targets"}
    assert payload["total_events"] == 1


def test_cli_concepts_json(built_db, monkeypatch, capsys):
    payload = _run_json(monkeypatch, capsys, ["concepts", "--json"])
    assert payload["concepts"][0]["members"] == 2


def test_cli_timeline_json(built_db, monkeypatch, capsys):
    payload = _run_json(monkeypatch, capsys, ["timeline", "--json"])
    assert payload["timeline"][0]["members"] == 2
    assert set(payload["timeline"][0]) == {"node_id", "title", "time_start", "time_end", "members"}


def test_cli_memory_stats_json(built_db, monkeypatch, capsys):
    payload = _run_json(monkeypatch, capsys, ["memory-stats", "--json"])
    assert payload["nodes_by_level"]["3"] == 1
    assert "integrity" in payload


def test_cli_eval_json(built_db, monkeypatch, capsys):
    payload = _run_json(monkeypatch, capsys, ["eval", "--query-prefix", "[t]", "--json"])
    assert {"recall_at_k", "mrr_at_k", "citation_hit_rate", "query_count", "misses"} <= set(payload)


def test_cli_recall_json_offline(built_db, monkeypatch, capsys):
    # Force the deterministic fallback (no Ollama) so the recall JSON path runs offline.
    import crossmodalrag.service as svc

    monkeypatch.setattr(svc, "get_default_llm_provider", lambda *a, **k: None)
    payload = _run_json(monkeypatch, capsys, ["recall", "--level", "concept", "--json"])
    assert payload["level"] == "concept"
    assert payload["recall"] and payload["recall"][0]["generated_by"] == "fallback"


# --- chat-history contracts (conversations/messages) ---------------------------------------


def test_message_and_conversation_to_dict_keys(tmp_path):
    import json as _json

    from crossmodalrag.conversations.contract import conversation_to_dict, message_to_dict
    from crossmodalrag.conversations.store import (
        create_conversation,
        get_conversation,
        list_messages,
        record_message,
    )
    from crossmodalrag.db import connect, init_db

    conn = connect(tmp_path / "m.db")
    init_db(conn)
    cid = create_conversation(conn, started_at="2026-07-11T10:00:00+00:00", title="t")
    record_message(conn, cid, turn_index=0, role="user", text="q")
    record_message(
        conn, cid, turn_index=0, role="assistant", text="a [E1]",
        evidence_json=_json.dumps([{"evidence_id": "E1"}]), model="stub",
    )
    conn.commit()

    msg = message_to_dict(list_messages(conn, cid)[1])
    assert set(msg) == {
        "id", "role", "turn_index", "text", "abstention_reason", "truncated",
        "model", "created_at", "evidence",
    }
    conv = conversation_to_dict(conn, get_conversation(conn, cid))
    assert set(conv) == {"id", "started_at", "updated_at", "title", "message_count", "messages"}
    conv_no_msgs = conversation_to_dict(conn, get_conversation(conn, cid), include_messages=False)
    assert set(conv_no_msgs) == {"id", "started_at", "updated_at", "title", "message_count"}
    conn.close()


def test_stored_evidence_shape_matches_ask_contract():
    # The shared-helper guarantee: what the recorder snapshots is element-for-element
    # the ask contract's evidence array.
    from crossmodalrag.generate.answer import evidence_payload, generated_answer_to_dict
    from crossmodalrag.generate.synthesize import GeneratedAnswer
    from crossmodalrag.retrieve.lexical import RetrievalHit

    hit = RetrievalHit(
        chunk_id=1, source_id=1, source_type="note", source_uri="/abs/x.md",
        source_timestamp="2026-06-01T00:00:00+00:00", title="x", chunk_index=0,
        chunk_text="alpha", score=0.9, lexical_score=0.9, recency_score=0.5, vector_score=0.0,
    )
    gen = GeneratedAnswer(
        query="q", answer_text="a [E1]", cited_evidence_ids=["E1"], invalid_citations=[],
        evidence=[hit], abstained=False, model="stub", id_map={"E1": hit},
    )
    assert generated_answer_to_dict(gen)["evidence"] == evidence_payload(gen)


# --- service layer: one envelope per read view, shared by CLI, API and MCP -------------------


def _service_payload(name: str, **kwargs) -> dict:
    import crossmodalrag.service as svc

    with svc.open_store() as conn:
        return getattr(svc, f"{name}_payload")(conn, **kwargs)


@pytest.mark.parametrize(
    ("name", "kwargs", "argv"),
    [
        ("concepts", {"top": 20}, ["concepts", "--json"]),
        ("timeline", {"limit": 50}, ["timeline", "--json"]),
        ("forgetting", {"level": "concept"}, ["forgetting", "--level", "concept", "--json"]),
        ("recall", {"level": "concept"}, ["recall", "--level", "concept", "--json"]),
    ],
)
def test_service_payloads_match_cli_json(built_db, monkeypatch, capsys, name, kwargs, argv):
    """The CLI renders the service envelope verbatim: one source of truth per contract."""
    from datetime import datetime, timezone

    import crossmodalrag.service as svc

    monkeypatch.setattr(svc, "get_default_llm_provider", lambda *a, **k: None)  # offline recall
    # Forgetting/recall scores decay with the clock; freeze it so both calls see the same instant.
    monkeypatch.setattr(svc, "_now", lambda: datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert _service_payload(name, **kwargs) == _run_json(monkeypatch, capsys, argv)


def test_forgetting_and_recall_payloads_reject_unknown_level(built_db):
    with pytest.raises(ValueError, match="bogus"):
        _service_payload("forgetting", level="bogus")
    with pytest.raises(ValueError, match="bogus"):
        _service_payload("recall", level="bogus")


def test_open_store_inits_schema(tmp_path, monkeypatch):
    import crossmodalrag.service as svc

    db_path = tmp_path / "fresh" / "memory.db"
    monkeypatch.setenv("CMRAG_DB_PATH", str(db_path))
    with svc.open_store() as conn:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sources", "evidence_chunks", "memory_nodes"} <= tables
    assert db_path.exists()

    explicit = tmp_path / "explicit.db"
    with svc.open_store(explicit) as conn:
        assert conn.execute("PRAGMA database_list").fetchone()["file"] == str(explicit)
