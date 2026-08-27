from __future__ import annotations

import json

from crossmodalrag.db import connect, init_db
from crossmodalrag.memory.extract import (
    EVENT_PROMPT_VERSION,
    extract_events_for_source,
    extract_pending_sources,
)
from crossmodalrag.memory.integrity import find_dangling_edges, find_unsupported_nodes
from crossmodalrag.memory.store import list_nodes, resolve_to_evidence


class StubLLMProvider:
    """Deterministic stub: returns a fixed JSON event array (no Ollama needed)."""

    def __init__(self, output: str | None = None, name: str = "stub-extract") -> None:
        self.name = name
        self._output = (
            output
            if output is not None
            else json.dumps(
                [
                    {"title": "Decided on schema", "summary": "Chose polymorphic edges."},
                    {"title": "Added drill-down", "summary": "Implemented resolve_to_evidence."},
                ]
            )
        )
        self.calls = 0

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        return self._output


def _new_db(tmp_path):
    conn = connect(tmp_path / "memory.db")
    init_db(conn)
    return conn


def _add_source(conn, uri: str, chunks: list[str], ts: str = "2026-06-01T00:00:00+00:00") -> int:
    cur = conn.execute(
        "INSERT INTO sources (source_type, source_uri, timestamp, title) VALUES (?, ?, ?, ?)",
        ("note", uri, ts, uri),
    )
    sid = int(cur.lastrowid)
    for idx, text in enumerate(chunks):
        conn.execute(
            "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
            (sid, idx, text),
        )
    conn.commit()
    return sid


def _snapshot(conn):
    nodes = conn.execute(
        "SELECT level, node_type, title, content, derivation_fingerprint, model, prompt_version "
        "FROM memory_nodes ORDER BY id"
    ).fetchall()
    edges = conn.execute(
        "SELECT parent_level, parent_id, child_level, child_id, relation FROM memory_edges ORDER BY id"
    ).fetchall()
    return [tuple(n) for n in nodes], [tuple(e) for e in edges]


def test_extract_creates_grounded_events(tmp_path) -> None:
    conn = _new_db(tmp_path)
    sid = _add_source(conn, "note-a", ["first body", "second body"])
    provider = StubLLMProvider()

    created = extract_events_for_source(conn, provider, sid)
    assert created == 2

    events = list_nodes(conn, level=1, node_type="event")
    assert len(events) == 2
    chunk_ids = [int(r["id"]) for r in conn.execute(
        "SELECT id FROM evidence_chunks WHERE source_id = ? ORDER BY id", (sid,)
    )]
    for event in events:
        assert resolve_to_evidence(conn, 1, event.id) == sorted(chunk_ids)
        assert event.model == "stub-extract"
        assert event.prompt_version == EVENT_PROMPT_VERSION
        assert event.derivation_fingerprint
        assert json.loads(event.metadata_json)["source_id"] == sid

    assert find_unsupported_nodes(conn) == []
    assert find_dangling_edges(conn) == []


def test_extract_pending_is_deterministic_no_op_on_rerun(tmp_path) -> None:
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["alpha"])
    _add_source(conn, "note-b", ["beta"])
    provider = StubLLMProvider()

    first = extract_pending_sources(conn, provider)
    assert first.sources_processed == 2
    assert first.events_created == 4
    snap1 = _snapshot(conn)

    second = extract_pending_sources(conn, provider)
    assert second.sources_processed == 0
    assert second.sources_skipped == 2
    assert second.events_created == 0
    assert _snapshot(conn) == snap1  # identical nodes/edges


def test_changed_source_triggers_reextraction(tmp_path) -> None:
    conn = _new_db(tmp_path)
    sid = _add_source(conn, "note-a", ["original body"])
    provider = StubLLMProvider()
    extract_pending_sources(conn, provider)

    # Change the source content -> fingerprint changes -> re-extract.
    conn.execute("UPDATE evidence_chunks SET chunk_text = ? WHERE source_id = ?", ("new body", sid))
    conn.commit()

    result = extract_pending_sources(conn, provider)
    assert result.sources_processed == 1
    assert find_dangling_edges(conn) == []
    # Still exactly the stub's two events for the one source (old ones replaced).
    assert len(list_nodes(conn, level=1, node_type="event")) == 2


def test_prompt_version_change_triggers_reextraction(tmp_path) -> None:
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["alpha"])
    provider = StubLLMProvider()
    extract_pending_sources(conn, provider, prompt_version="v1")
    again = extract_pending_sources(conn, provider, prompt_version="v2")
    assert again.sources_processed == 1


def test_malformed_output_is_graceful(tmp_path) -> None:
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["alpha"])
    provider = StubLLMProvider(output="I could not produce JSON, sorry.")

    result = extract_pending_sources(conn, provider)
    assert result.events_created == 0
    assert result.parse_failures == 1
    assert list_nodes(conn, level=1, node_type="event") == []
    assert find_dangling_edges(conn) == []

    # Unparseable sources are not recorded, so they are retried (not skipped) next run.
    retry = extract_pending_sources(conn, provider)
    assert retry.sources_processed == 1


def test_empty_event_array_is_recorded_and_skipped(tmp_path) -> None:
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["alpha"])
    provider = StubLLMProvider(output="[]")

    first = extract_pending_sources(conn, provider)
    assert first.events_created == 0
    assert first.parse_failures == 0
    # A valid empty result IS recorded, so re-running skips it (no wasted LLM call).
    second = extract_pending_sources(conn, provider)
    assert second.sources_processed == 0
    assert second.sources_skipped == 1


def test_limit_bounds_sources_processed(tmp_path) -> None:
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["a"])
    _add_source(conn, "note-b", ["b"])
    _add_source(conn, "note-c", ["c"])
    provider = StubLLMProvider()

    result = extract_pending_sources(conn, provider, limit=1)
    assert result.sources_processed == 1
    # Remaining two are picked up on subsequent runs.
    result2 = extract_pending_sources(conn, provider, limit=5)
    assert result2.sources_processed == 2


def test_parse_repairs_unquoted_string_values(tmp_path) -> None:
    # Regression: llama3.2 (temp 0) emits well-shaped objects whose string
    # values are missing quotes ('"summary": Cache exists to reduce latency').
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["alpha"])
    raw = (
        '[\n  {\n    "title": "High-level purpose of cache",\n'
        '    "summary": Cache exists to reduce latency/bandwidth\n  },\n'
        '  {\n    "title": "Core problem",\n'
        '    "summary": Modern systems have a memory hierarchy\n  }\n]'
    )

    result = extract_pending_sources(conn, StubLLMProvider(output=raw))

    assert result.parse_failures == 0
    assert result.events_created == 2
    titles = {n.title for n in list_nodes(conn, level=1, node_type="event")}
    assert titles == {"High-level purpose of cache", "Core problem"}


def test_parse_repairs_braceless_bracket_blocks(tmp_path) -> None:
    # Regression: llama3.2 sometimes writes each event as its own bracket block
    # without braces ('[ "title": ..., "summary": ... ], [ ... ]').
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["alpha"])
    raw = (
        '[\n  "title": "Added journal entry",\n'
        '  "summary": "A new journal entry was added"\n],\n \n'
        '[\n  "title": "Updated spring-journal.pdf",\n'
        '  "summary": "The size was updated"\n],'
    )

    result = extract_pending_sources(conn, StubLLMProvider(output=raw))

    assert result.parse_failures == 0
    assert result.events_created == 2
    titles = {n.title for n in list_nodes(conn, level=1, node_type="event")}
    assert titles == {"Added journal entry", "Updated spring-journal.pdf"}


def test_parse_repair_does_not_requote_quoted_values(tmp_path) -> None:
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["alpha"])
    raw = '[\n  {\n    "title": "Quoted title",\n    "summary": bare value here\n  }\n]'

    extract_pending_sources(conn, StubLLMProvider(output=raw))

    node = list_nodes(conn, level=1, node_type="event")[0]
    assert node.title == "Quoted title"  # not '"Quoted title"'
    assert node.content == "bare value here"


def test_unparseable_sources_are_identified(tmp_path) -> None:
    conn = _new_db(tmp_path)
    sid = _add_source(conn, "note-a", ["alpha"])
    provider = StubLLMProvider(output="I could not produce JSON, sorry.")

    result = extract_pending_sources(conn, provider)

    assert result.parse_failures == 1
    assert result.unparseable_sources == [(sid, "note-a")]


def test_parse_recovers_truncated_array(tmp_path) -> None:
    # Regression: the model runs out of tokens and never closes the array —
    # every fully-emitted object is recovered, the trailing partial one dropped.
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["alpha"])
    raw = (
        '[\n  {"title": "DHCP client checks database", "summary": "Checks static db"},\n'
        '  {"title": "Pool exhausted", "summary": "Available IPs depleted"},\n'
        '  {"title": "Partial object", "summary": "cut off mid-'
    )

    result = extract_pending_sources(conn, StubLLMProvider(output=raw))

    assert result.parse_failures == 0
    titles = {n.title for n in list_nodes(conn, level=1, node_type="event")}
    assert titles == {"DHCP client checks database", "Pool exhausted"}


def test_parse_repairs_latex_escapes(tmp_path) -> None:
    # Regression: LaTeX in summaries ('\\implies', '\\land') is not a legal JSON
    # escape sequence; it must survive as literal text, not fail the whole array.
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["alpha"])
    raw = '[{"title": "Eliminate biconditional", "summary": "(A $\\implies$ B) $\\land$ (B $\\implies$ A)"}]'

    result = extract_pending_sources(conn, StubLLMProvider(output=raw))

    assert result.parse_failures == 0
    node = list_nodes(conn, level=1, node_type="event")[0]
    assert "$\\implies$" in node.content


def test_parse_accepts_capitalized_keys(tmp_path) -> None:
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["alpha"])
    raw = '[\n  "Title": "Twos complement",\n  "Summary": "Definition"\n],\n[\n  "Title": "Derivation",\n  "Summary": "From minterms"\n]'

    result = extract_pending_sources(conn, StubLLMProvider(output=raw))

    assert result.parse_failures == 0
    titles = {n.title for n in list_nodes(conn, level=1, node_type="event")}
    assert titles == {"Twos complement", "Derivation"}


def test_parse_repairs_unquoted_keys(tmp_path) -> None:
    # Regression: keys emitted without quotes ('{"title": "x", summary: "y"}').
    conn = _new_db(tmp_path)
    _add_source(conn, "note-a", ["alpha"])
    raw = (
        '[\n  {"title": "Hot Air Gun turned OFF", summary: "Mode changed from ON to OFF"},\n'
        '  {"title": "Fan turned ON", summary: "Mode changed from OFF to ON"}\n]'
    )

    result = extract_pending_sources(conn, StubLLMProvider(output=raw))

    assert result.parse_failures == 0
    titles = {n.title for n in list_nodes(conn, level=1, node_type="event")}
    assert titles == {"Hot Air Gun turned OFF", "Fan turned ON"}
