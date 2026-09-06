"""The local MCP server: Engram's read views as tools over stdio.

Pins the thin-client guarantee a third time (tool payload == service payload == CLI `--json`),
the read-only tool surface (names, annotations, no usage writes), the provenance every `ask`
evidence item carries (`locator`, `source_uri`, `chunk_id`), and the two failure modes a
client-spawned server must make loud: a database that does not exist, and stray stdout.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

import pytest

pytest.importorskip("mcp")  # skips the whole module when the [mcp] extra is absent
from mcp import Client

import crossmodalrag.service as svc
from crossmodalrag import cli, mcp_server
from crossmodalrag.db import connect, init_db
from crossmodalrag.memory.forgetting import LEVEL_NAMES
from crossmodalrag.memory.store import add_edge
from crossmodalrag.modality import build_chunk_metadata
from crossmodalrag.retrieve.hybrid import PROFILE_WEIGHTS
from crossmodalrag.retrieve.nodes import LEVEL_TO_NODE
from crossmodalrag.retrieve.rerank import MODALITY_SOURCE_TYPES
from crossmodalrag.usage.store import record_usage_event

pytestmark = pytest.mark.anyio

TOOLS = {"ask", "concepts", "timeline", "forgetting", "recall", "history", "conversation", "status"}
FROZEN_NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def built_db(tmp_path, monkeypatch):
    """A small memory graph plus one PDF page chunk, wired as CMRAG_DB_PATH, fully offline."""
    db = tmp_path / "memory.db"
    conn = connect(db)
    init_db(conn)

    def _event(text):
        cur = conn.execute("INSERT INTO sources (source_type, source_uri) VALUES ('note', ?)", (f"/v/{text}.md",))
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
        return eid, chunk_id

    e1, c1 = _event("parser bounds fix")
    e2, _ = _event("parser overflow guard")
    cur = conn.execute(
        "INSERT INTO memory_nodes (level, node_type, title, centrality) VALUES (3, 'concept', ?, 0.9)",
        ("Parser hardening",),
    )
    cid = int(cur.lastrowid)
    add_edge(conn, 3, cid, 1, e1, "contains")
    add_edge(conn, 3, cid, 1, e2, "contains")
    cur = conn.execute(
        "INSERT INTO memory_nodes (level, node_type, title, time_start, time_end) VALUES "
        "(2, 'episode', ?, ?, ?)",
        ("Parser session", "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"),
    )
    add_edge(conn, 2, int(cur.lastrowid), 1, e1, "contains")
    record_usage_event(conn, "chunk", c1, "retrieval_hit", event_at="2026-01-01T00:00:00+00:00")
    # One PDF page so an `ask` can surface a `file.pdf p.N` locator through the MCP path.
    cur = conn.execute("INSERT INTO sources (source_type, source_uri) VALUES ('pdf', '/docs/spec.pdf')")
    conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text, metadata_json) VALUES (?, 0, ?, ?)",
        (
            int(cur.lastrowid),
            "The rate limit is defined as 100 requests per minute per token.",
            json.dumps(build_chunk_metadata(modality="pdf-page", source_type="pdf", page=4)),
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("CMRAG_DB_PATH", str(db))
    monkeypatch.delenv("CMRAG_USAGE_TRACKING", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    # Offline and deterministic: no Ollama, frozen clock for the decay-based views.
    monkeypatch.setattr(svc, "ping_ollama", lambda: False)
    monkeypatch.setattr(svc, "get_default_llm_provider", lambda *a, **k: None)
    monkeypatch.setattr(svc, "_now", lambda: FROZEN_NOW)
    return db, cid


@pytest.fixture
async def client(built_db):
    async with Client(mcp_server.build_server()) as c:
        yield c


def _run_json(monkeypatch, capsys, argv: list[str]) -> dict:
    monkeypatch.setattr(sys, "argv", ["mem", *argv])
    cli.main()
    return json.loads(capsys.readouterr().out)


def _service(name: str, **kwargs) -> dict:
    with svc.open_store() as conn:
        return getattr(svc, f"{name}_payload")(conn, **kwargs)


def _error_text(result) -> str:
    """Tool failures reach the client as an error result carrying the message, not an exception."""
    assert result.is_error is True
    assert result.structured_content is None
    return "".join(getattr(block, "text", "") for block in result.content)


# --- tool surface -------------------------------------------------------------


async def test_tool_list_and_annotations(client):
    listed = await client.list_tools()
    tools = {t.name: t for t in listed.tools}
    assert set(tools) == TOOLS
    for name, tool in tools.items():
        assert tool.description, name
        assert tool.annotations is not None, name
        assert tool.annotations.read_only_hint is True, name
        assert tool.annotations.open_world_hint is False, name
    assert tools["recall"].annotations.idempotent_hint is True
    assert client.instructions and "read-only" in client.instructions.lower()


def test_literal_params_track_library_tables():
    """The schema-level enums must follow the library's own tables, not drift from them."""
    assert set(get_args(mcp_server.MemoryLevel)) == set(LEVEL_NAMES)
    assert set(get_args(mcp_server.Profile)) == set(PROFILE_WEIGHTS)
    assert set(get_args(mcp_server.Modality)) == set(MODALITY_SOURCE_TYPES)
    assert set(get_args(mcp_server.AskLevel)) == {"evidence", *LEVEL_TO_NODE}


async def test_bad_level_rejected_by_schema(client, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("the library must not run for an argument the schema rejects")

    monkeypatch.setattr(svc, "forgetting_payload", _boom)
    text = _error_text(await client.call_tool("forgetting", {"level": "bogus"}))
    assert "bogus" not in text or "Input should be" in text  # a validation message, not a traceback
    assert "'event', 'episode', 'concept' or 'all'" in text


# --- thin-client guarantee: tool == service == CLI --json -----------------------


@pytest.mark.parametrize(
    ("tool", "args", "service", "kwargs", "argv"),
    [
        ("concepts", {}, "concepts", {"top": 20}, ["concepts", "--json"]),
        ("timeline", {}, "timeline", {"limit": 50}, ["timeline", "--json"]),
        ("forgetting", {"level": "concept"}, "forgetting", {"level": "concept"},
         ["forgetting", "--level", "concept", "--json"]),
        ("recall", {"level": "concept"}, "recall", {"level": "concept"},
         ["recall", "--level", "concept", "--json"]),
        ("history", {}, "conversations", {"top": 10}, ["history", "--json"]),
    ],
)
async def test_read_tools_match_service_and_cli(client, monkeypatch, capsys, tool, args, service, kwargs, argv):
    result = await client.call_tool(tool, args)
    assert result.is_error is False
    assert result.structured_content == _service(service, **kwargs)
    assert result.structured_content == _run_json(monkeypatch, capsys, argv)


async def test_recall_uses_fallback_cards_offline(client):
    result = await client.call_tool("recall", {})
    cards = result.structured_content["recall"]
    assert cards and all(c["generated_by"] == "fallback" for c in cards)
    assert all(c["evidence_source_uris"] for c in cards)  # every card cites its source


async def test_status_is_the_health_report(client):
    result = await client.call_tool("status", {})
    assert set(result.structured_content) == set(svc.health_report())
    assert result.structured_content["ollama"]["reachable"] is False
    assert result.structured_content["db"]["exists"] is True


# --- ask: provenance through the MCP path -------------------------------------------


async def test_ask_matches_cli_minus_timing(client, monkeypatch, capsys):
    result = await client.call_tool("ask", {"query": "parser bounds", "use_llm": False})
    tool_payload = dict(result.structured_content)
    cli_payload = _run_json(monkeypatch, capsys, ["ask", "parser bounds", "--no-llm", "--json"])
    tool_payload.pop("timing")
    cli_payload.pop("timing")
    assert tool_payload == cli_payload
    assert tool_payload["abstained"] is False
    first = tool_payload["evidence"][0]
    assert {"evidence_id", "source_uri", "locator", "chunk_id", "page", "modality"} <= set(first)


async def test_ask_locator_renders_pdf_page(client):
    result = await client.call_tool("ask", {"query": "rate limit per minute", "use_llm": False})
    locators = [e["locator"] for e in result.structured_content["evidence"]]
    assert any(loc.endswith("/docs/spec.pdf p.4") for loc in locators), locators


async def test_ask_never_records_usage(client, built_db, monkeypatch):
    monkeypatch.setenv("CMRAG_USAGE_TRACKING", "on")
    db, _ = built_db
    await client.call_tool("ask", {"query": "parser", "use_llm": False})
    conn = connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM usage_events").fetchone()["n"] == 1  # the fixture's
    finally:
        conn.close()


# --- conversations ------------------------------------------------------------


async def test_conversation_roundtrip(client, built_db):
    from crossmodalrag.conversations.store import create_conversation, record_message

    db, _ = built_db
    conn = connect(db)
    cid = create_conversation(conn, started_at="2026-07-11T10:00:00+00:00", title="t")
    record_message(conn, cid, turn_index=0, role="user", text="q")
    record_message(
        conn, cid, turn_index=0, role="assistant", text="a [E1]",
        evidence_json=json.dumps([{"evidence_id": "E1"}]), model="stub",
    )
    conn.commit()
    conn.close()

    result = await client.call_tool("conversation", {"conversation_id": cid})
    with svc.open_store() as conn:
        assert result.structured_content == svc.conversation_payload(conn, cid)
    assert result.structured_content["messages"][1]["evidence"] == [{"evidence_id": "E1"}]


async def test_conversation_not_found_is_tool_error(client):
    text = _error_text(await client.call_tool("conversation", {"conversation_id": 999}))
    assert "999" in text


# --- hygiene: missing DB, explicit DB, stdout, writes --------------------------------


async def test_missing_db_is_clear_tool_error(tmp_path, monkeypatch):
    missing = tmp_path / "nope" / "memory.db"
    monkeypatch.setenv("CMRAG_DB_PATH", str(missing))
    async with Client(mcp_server.build_server()) as c:
        text = _error_text(await c.call_tool("concepts", {}))
    assert str(missing) in text and "CMRAG_DB_PATH" in text
    assert not missing.exists()  # never silently created


async def test_build_server_db_path_overrides_env(built_db, tmp_path):
    other = tmp_path / "other.db"
    conn = connect(other)
    init_db(conn)
    conn.close()
    async with Client(mcp_server.build_server(db_path=other)) as c:
        result = await c.call_tool("concepts", {})
    assert result.structured_content == {"concepts": []}


async def test_tools_write_nothing_to_stdout(client, capsys):
    await client.call_tool("ask", {"query": "parser", "use_llm": False})
    for name in ("concepts", "timeline", "forgetting", "recall", "history", "status"):
        await client.call_tool(name, {})
    assert capsys.readouterr().out == ""


async def test_recall_only_writes_its_card_cache(client, built_db):
    db, _ = built_db

    def _counts() -> dict[str, int]:
        conn = connect(db)
        try:
            names = [r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            return {n: conn.execute(f"SELECT COUNT(*) AS n FROM {n}").fetchone()["n"] for n in names}
        finally:
            conn.close()

    before = _counts()
    await client.call_tool("recall", {})
    after = _counts()
    assert after["recall_cards"] > before["recall_cards"]
    assert {k: v for k, v in after.items() if k != "recall_cards"} == {
        k: v for k, v in before.items() if k != "recall_cards"
    }


# --- real stdio: spawn the server the way a client does ---------------------------------


async def test_stdio_subprocess_roundtrip(built_db, tmp_path):
    """`python -m crossmodalrag.mcp_server --db …` speaks MCP over its own stdin/stdout.

    Spawned from an unrelated working directory with no CMRAG_DB_PATH, the way Claude Desktop
    launches servers, so `--db` alone must be enough.
    """
    import os

    import crossmodalrag
    from mcp import StdioServerParameters

    db, _ = built_db
    env = {k: v for k, v in os.environ.items() if k != "CMRAG_DB_PATH"}
    env["PYTHONPATH"] = str(Path(crossmodalrag.__file__).resolve().parents[1])
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "crossmodalrag.mcp_server", "--db", str(db)],
        env=env,
        cwd=str(tmp_path / "elsewhere"),
    )
    (tmp_path / "elsewhere").mkdir()
    async with Client(params) as c:
        listed = await c.list_tools()
        assert {t.name for t in listed.tools} == TOOLS
        result = await c.call_tool("concepts", {})
        assert result.structured_content["concepts"][0]["title"] == "Parser hardening"
        result = await c.call_tool("status", {})
        assert result.structured_content["db"]["path"] == str(db)
    assert not (tmp_path / "elsewhere" / "data").exists()  # nothing created in the spawn cwd
