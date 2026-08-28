from __future__ import annotations

import json
import sys

import pytest

fastapi = pytest.importorskip("fastapi")  # skips the whole module when the [ui] extra is absent
from fastapi.testclient import TestClient  # noqa: E402

from crossmodalrag import cli  # noqa: E402
from crossmodalrag.api import MissingUIBackend, create_app  # noqa: E402
from crossmodalrag.db import connect, init_db  # noqa: E402
from crossmodalrag.memory.store import add_edge  # noqa: E402
from crossmodalrag.usage.store import record_usage_event  # noqa: E402


@pytest.fixture
def built_db(tmp_path, monkeypatch):
    """A small DB with a concept + events + episode + one usage event, wired as CMRAG_DB_PATH."""
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
    conn.commit()
    conn.close()

    monkeypatch.setenv("CMRAG_DB_PATH", str(db))
    # Keep /health deterministic + offline.
    import crossmodalrag.service as svc
    monkeypatch.setattr(svc, "ping_ollama", lambda: False)
    return db, cid


@pytest.fixture
def client(built_db):
    # `base_url` matters: the API rejects any Host that is not a loopback name (the
    # DNS-rebinding guard in `api/guard.py`), and TestClient otherwise sends `testserver`.
    # Guard-specific coverage lives in tests/test_api_guard.py.
    return TestClient(create_app(), base_url="http://127.0.0.1:8765")


# --- endpoint contracts -------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"db", "extras", "ollama", "models", "config", "connectors", "memory"}
    assert body["ollama"]["reachable"] is False


def test_memory_stats(client):
    body = client.get("/memory-stats").json()
    assert body["nodes_by_level"]["3"] == 1 and "integrity" in body


def test_concepts_and_timeline(client, built_db):
    _, cid = built_db
    concepts = client.get("/concepts").json()["concepts"]
    assert concepts and concepts[0]["node_id"] == cid and concepts[0]["members"] == 2
    timeline = client.get("/timeline").json()["timeline"]
    assert timeline and set(timeline[0]) == {"node_id", "title", "time_start", "time_end", "members"}


def test_forgetting(client):
    body = client.get("/forgetting", params={"level": "concept"}).json()
    assert body["level"] == "concept"
    assert body["forgetting"] and "evidence_source_uris" in body["forgetting"][0]


def test_drift_distill_usage_shapes(client):
    assert "drift" in client.get("/drift").json()
    distill = client.get("/distill").json()
    assert set(distill) == {"distilled", "overall_compression_ratio"}
    usage = client.get("/usage").json()
    assert usage["total_events"] == 1 and set(usage) == {
        "tracking_enabled", "total_events", "by_type", "top_targets"
    }


def test_ask_offline_template(client):
    body = client.get("/ask", params={"q": "parser", "use_llm": "false"}).json()
    assert set(body) >= {"query", "model", "abstained", "answer", "evidence"}
    assert body["model"] is None  # no-LLM template path
    if body["evidence"]:
        assert "source_uri" in body["evidence"][0] and "locator" in body["evidence"][0]


def test_ask_stream_no_llm_emits_single_answer_event(client):
    with client.stream("GET", "/ask/stream", params={"q": "parser", "use_llm": "false"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert [e["type"] for e in events] == ["answer"]
    assert set(events[0]["data"]) >= {"query", "model", "abstained", "answer", "evidence"}


def test_ask_stream_tokens_then_final_answer_matches_ask(client, monkeypatch):
    import crossmodalrag.service as svc

    class _StreamStub:
        name = "stub-stream"

        def generate(self, prompt, system=None):
            return "Grounded answer [E1]."

        def generate_stream(self, prompt, system=None):
            yield "Grounded "
            yield "answer [E1]."

    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    monkeypatch.setattr(svc, "get_default_llm_provider", lambda: _StreamStub())

    with client.stream("GET", "/ask/stream", params={"q": "parser"}) as r:
        assert r.status_code == 200
        events = [json.loads(line) for line in r.iter_lines() if line]

    assert [e["type"] for e in events] == ["token", "token", "answer"]
    assert "".join(e["text"] for e in events[:-1]) == "Grounded answer [E1]."

    # The final event carries the exact /ask contract (timing wall-clock aside).
    buffered = client.get("/ask", params={"q": "parser"}).json()
    final = events[-1]["data"]
    final.pop("timing"), buffered.pop("timing")
    assert final == buffered


def test_ask_stream_early_disconnect_leaves_server_healthy(client, monkeypatch):
    """Abandoning the NDJSON stream mid-flight (the console's Stop button) must not
    corrupt server state — follow-up requests keep working."""
    import crossmodalrag.service as svc

    class _ManyTokens:
        name = "stub-stream"

        def generate(self, prompt, system=None):
            return "x [E1]."

        def generate_stream(self, prompt, system=None):
            for i in range(50):
                yield f"tok{i} "

    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    monkeypatch.setattr(svc, "get_default_llm_provider", lambda: _ManyTokens())

    with client.stream("GET", "/ask/stream", params={"q": "parser"}) as r:
        assert r.status_code == 200
        next(r.iter_lines())  # read one event, then drop the connection

    assert client.get("/health").status_code == 200
    assert client.get("/ask", params={"q": "parser", "use_llm": "false"}).status_code == 200


def test_bad_level_is_400(client):
    assert client.get("/forgetting", params={"level": "nonsense"}).status_code == 400


def test_routes_are_get_only(client):
    assert client.post("/concepts").status_code == 405  # read-only API


def test_web_ui_served_at_root_when_built(client):
    from crossmodalrag.api.app import STATIC_DIR

    if not STATIC_DIR.is_dir():
        pytest.skip("web UI not built (run `cd web && npm run build`)")
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # The SPA mount must not shadow the JSON API.
    assert client.get("/memory-stats").status_code == 200


# --- thin-client guarantee: API payload == CLI --json payload -----------------


def test_api_matches_cli_concepts(client, built_db, monkeypatch, capsys):
    api_body = client.get("/concepts").json()
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["mem", "concepts", "--json"])
    cli.main()
    cli_body = json.loads(capsys.readouterr().out)
    assert api_body == cli_body  # same library contract, two surfaces


# --- serve degradation --------------------------------------------------------


def test_serve_cmd_translates_missing_backend(monkeypatch):
    import crossmodalrag.api as api

    def _boom(**kwargs):
        raise MissingUIBackend("need the [ui] extra")

    monkeypatch.setattr(api, "create_app", _boom)
    with pytest.raises(cli.CLIError):
        cli.serve_cmd()


# --- conversations + chat (step 13) --------------------------------------------------------


class _ChatStub:
    name = "stub-chat"

    def generate(self, prompt, system=None):
        if prompt.startswith("Title this conversation"):
            return "Parser Bounds Chat"
        return "Grounded answer [E1]."

    def generate_stream(self, prompt, system=None):
        if prompt.startswith("Title this conversation"):
            yield "Parser Bounds Chat"
            return
        yield "Grounded "
        yield "answer [E1]."


def _chat_events(client, body):
    with client.stream("POST", "/chat/stream", json=body) as r:
        assert r.status_code == 200
        return [json.loads(line) for line in r.iter_lines() if line]


@pytest.fixture
def chat_env(built_db, monkeypatch):
    import crossmodalrag.service as svc

    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    monkeypatch.delenv("CMRAG_SAVE_HISTORY", raising=False)
    monkeypatch.setattr(svc, "get_default_llm_provider", lambda: _ChatStub())
    return built_db


def test_conversations_empty_list(client, chat_env):
    r = client.get("/conversations")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"save_enabled", "total", "conversations"}
    assert body["total"] == 0


def test_chat_stream_creates_conversation_and_persists(client, chat_env):
    events = _chat_events(client, {"q": "parser bounds fix?"})
    assert [e["type"] for e in events] == ["token", "token", "answer"]
    final = events[-1]
    assert final["conversation_id"] is not None
    assert final["conversation"]["title"] == "Parser Bounds Chat"  # LLM-titled
    assert final["data"]["answer"] == "Grounded answer [E1]."

    cid = final["conversation_id"]
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    conv = r.json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["messages"][1]["evidence"][0]["evidence_id"] == "E1"


def test_chat_stream_resumes_conversation_with_context(client, chat_env, monkeypatch):
    first = _chat_events(client, {"q": "parser bounds fix?"})[-1]
    cid = first["conversation_id"]

    captured: list[str] = []
    import crossmodalrag.service as svc

    class _Recording(_ChatStub):
        def generate_stream(self, prompt, system=None):
            if not prompt.startswith("Title this conversation"):
                captured.append(prompt)
            yield from super().generate_stream(prompt, system=system)

    monkeypatch.setattr(svc, "get_default_llm_provider", lambda: _Recording())
    second = _chat_events(client, {"q": "parser again?", "conversation_id": cid})[-1]
    assert second["conversation_id"] == cid  # appended, not a new conversation

    from crossmodalrag.chat import HISTORY_HEADER

    assert HISTORY_HEADER in captured[0]
    assert "User: parser bounds fix?" in captured[0]

    conv = client.get(f"/conversations/{cid}").json()
    assert conv["message_count"] == 4


def test_chat_stream_unknown_conversation_404(client, chat_env):
    r = client.post("/chat/stream", json={"q": "hi", "conversation_id": 424242})
    assert r.status_code == 404


def test_chat_stream_missing_query_400(client, chat_env):
    r = client.post("/chat/stream", json={})
    assert r.status_code == 400


def test_chat_stream_save_false_persists_nothing(client, chat_env):
    final = _chat_events(client, {"q": "parser bounds fix?", "save": False})[-1]
    assert final["conversation_id"] is None
    assert final["data"]["answer"] == "Grounded answer [E1]."
    assert client.get("/conversations").json()["total"] == 0


def test_conversation_endpoint_matches_cli_history_json(client, chat_env, monkeypatch, capsys):
    cid = _chat_events(client, {"q": "parser bounds fix?"})[-1]["conversation_id"]
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["mem", "history", "--show", str(cid), "--json"])
    cli.main()
    cli_payload = json.loads(capsys.readouterr().out)
    api_payload = client.get(f"/conversations/{cid}").json()
    assert api_payload == cli_payload  # thin-client guarantee: one contract, two surfaces


def test_delete_conversation_scoped(client, chat_env):
    cid1 = _chat_events(client, {"q": "parser bounds fix?"})[-1]["conversation_id"]
    cid2 = _chat_events(client, {"q": "parser overflow guard?"})[-1]["conversation_id"]
    assert cid1 != cid2

    r = client.delete(f"/conversations/{cid1}")
    assert r.status_code == 200
    assert r.json() == {"deleted": 1}
    assert client.get(f"/conversations/{cid1}").status_code == 404  # gone
    assert client.get(f"/conversations/{cid2}").status_code == 200  # untouched
    assert client.get("/conversations").json()["total"] == 1


def test_delete_conversation_unknown_or_repeat_404(client, chat_env):
    assert client.delete("/conversations/424242").status_code == 404
    cid = _chat_events(client, {"q": "parser bounds fix?"})[-1]["conversation_id"]
    assert client.delete(f"/conversations/{cid}").status_code == 200
    assert client.delete(f"/conversations/{cid}").status_code == 404  # double delete


def test_rename_conversation_endpoint(client, chat_env):
    cid = _chat_events(client, {"q": "parser bounds fix?"})[-1]["conversation_id"]
    r = client.patch(f"/conversations/{cid}", json={"title": "  Parser deep dive  "})
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Parser deep dive"  # stripped
    assert body["id"] == cid
    assert client.get(f"/conversations/{cid}").json()["title"] == "Parser deep dive"


def test_rename_conversation_endpoint_errors(client, chat_env):
    assert client.patch("/conversations/424242", json={"title": "x"}).status_code == 404
    cid = _chat_events(client, {"q": "parser bounds fix?"})[-1]["conversation_id"]
    assert client.patch(f"/conversations/{cid}", json={}).status_code == 400
    assert client.patch(f"/conversations/{cid}", json={"title": "   "}).status_code == 400


def test_rename_conversation_404_when_row_disappears_after_update(client, chat_env, monkeypatch):
    """A conversation that vanishes between the UPDATE and the read-back yields 404.

    The rename handler re-reads the row to build its response. A concurrent delete, or an
    interpreter running with assertions stripped (where an ``assert`` guard is never compiled),
    leaves that read returning None. The handler has to answer 404 rather than pass None into
    the contract layer, which would surface as an opaque AttributeError.
    """
    cid = _chat_events(client, {"q": "parser bounds fix?"})[-1]["conversation_id"]

    import crossmodalrag.conversations.store as store

    monkeypatch.setattr(store, "get_conversation", lambda conn, conversation_id: None)
    response = client.patch(f"/conversations/{cid}", json={"title": "Parser deep dive"})
    assert response.status_code == 404
    assert str(cid) in response.json()["detail"]


def test_route_annotations_resolve(built_db):
    """Every handler annotation must resolve against the module namespace.

    ``from __future__ import annotations`` keeps annotations as strings, so a handler annotated
    with a name imported only inside its own body still parses and still serves traffic.
    Anything that later reads those hints (``typing.get_type_hints``, schema generation, a docs
    build) raises NameError. This pins the handlers to names that exist at module level.
    """
    import typing

    app = create_app()
    checked = 0
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None or getattr(endpoint, "__module__", None) != create_app.__module__:
            continue
        typing.get_type_hints(endpoint)
        checked += 1
    assert checked > 10  # guards against the loop silently matching nothing
