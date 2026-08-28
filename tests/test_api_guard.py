"""Browser-facing guards on the loopback API (issue #3, CWE-352).

`mem serve` binds 127.0.0.1 with no auth, so the browser is the only realistic attacker: a
page the user already has open can address the loopback service directly. Two request headers
close that door, and both are set by the browser and unforgeable from page script.

`Host` closes DNS rebinding — a name the attacker controls that resolves to 127.0.0.1 makes
their page *same-origin* with the API, which is what would turn unreadable drive-by requests
into readable memory exfiltration. `Sec-Fetch-Site` closes the plain drive-by: `<img src>` and
`fetch(..., {mode:'no-cors'})` are simple cross-origin GETs that neither a preflight nor the
absence of CORS headers can stop, and each one costs a local LLM inference run.

Non-browser clients (curl, the CLI, scripts) send no `Sec-Fetch-*` headers and are unaffected.
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")  # skips the whole module when the [ui] extra is absent
from fastapi.testclient import TestClient  # noqa: E402

from crossmodalrag.api.guard import (  # noqa: E402
    LOOPBACK_HOSTNAMES,
    normalize_hostname,
    reject_reason,
)
from crossmodalrag.db import connect, init_db  # noqa: E402

SERVED = "http://127.0.0.1:8765"


@pytest.fixture
def db(tmp_path, monkeypatch):
    """One note chunk, wired as CMRAG_DB_PATH, with /health kept offline and deterministic."""
    path = tmp_path / "memory.db"
    conn = connect(path)
    init_db(conn)
    cur = conn.execute("INSERT INTO sources (source_type, source_uri) VALUES ('note', '/v/parser.md')")
    conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, 0, ?)",
        (int(cur.lastrowid), "parser bounds fix"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("CMRAG_DB_PATH", str(path))
    import crossmodalrag.service as svc

    monkeypatch.setattr(svc, "ping_ollama", lambda: False)
    return path


@pytest.fixture
def guarded(db):
    from crossmodalrag.api import create_app

    return TestClient(create_app(), base_url=SERVED)


# --- Host header parsing ------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("127.0.0.1:8765", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1"),
        ("localhost:5173", "localhost"),
        ("LocalHost", "localhost"),
        ("  localhost:8765  ", "localhost"),
        ("[::1]:8765", "::1"),
        ("[::1]", "::1"),
        ("::1", "::1"),  # unbracketed IPv6 literal: colons are the address, not a port
        ("evil.test:8765", "evil.test"),
        ("", ""),
    ],
)
def test_normalize_hostname_strips_port_and_brackets(header, expected):
    assert normalize_hostname(header) == expected


def test_loopback_hostnames_are_the_three_names_a_browser_can_send():
    assert LOOPBACK_HOSTNAMES == frozenset({"localhost", "127.0.0.1", "::1"})


# --- Host check (DNS rebinding) -----------------------------------------------


def _reason(host="127.0.0.1:8765", method="GET", allowed=None, **headers):
    """`reject_reason` with a loopback Host and no fetch metadata unless overridden."""
    return reject_reason(
        method=method,
        headers={"host": host, **headers},
        allowed_hostnames=LOOPBACK_HOSTNAMES if allowed is None else frozenset(allowed),
    )


@pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765", "[::1]:8765", "localhost:5173"])
def test_loopback_hosts_are_accepted_on_any_port(host):
    """The port is not the boundary: rebinding needs a *name*, and the Vite dev proxy forwards
    the browser's own `localhost:5173` unchanged."""
    assert _reason(host=host) is None


@pytest.mark.parametrize("host", ["evil.test:8765", "rebind.evil.test", "192.168.1.5:8765", ""])
def test_non_loopback_host_is_rejected(host):
    reason = _reason(host=host)
    assert reason is not None
    assert "Host" in reason


def test_rebound_host_is_rejected_even_when_the_browser_calls_it_same_origin():
    """The rebinding payoff: once evil.test resolves to 127.0.0.1 the attacker's page IS
    same-origin, so fetch metadata alone waves it through and responses become readable.
    The Host check has to run first and independently."""
    assert _reason(host="evil.test:8765", **{"sec-fetch-site": "same-origin"}) is not None


def test_explicitly_allowed_host_is_accepted():
    assert _reason(host="engram.lan:8765", allowed=LOOPBACK_HOSTNAMES | {"engram.lan"}) is None


def test_wildcard_allows_any_host_but_keeps_the_fetch_metadata_check():
    assert _reason(host="engram.lan", allowed={"*"}) is None
    assert _reason(host="engram.lan", allowed={"*"}, **{"sec-fetch-site": "cross-site"}) is not None


# --- Fetch metadata (drive-by) ------------------------------------------------


def test_missing_fetch_metadata_is_accepted():
    """curl, the CLI, and every other non-browser client send no Sec-Fetch-* headers."""
    assert _reason() is None


@pytest.mark.parametrize("site", ["same-origin", "same-site", "none"])
def test_same_origin_and_user_initiated_requests_are_accepted(site):
    assert _reason(**{"sec-fetch-site": site}) is None


@pytest.mark.parametrize(
    ("dest", "mode"),
    [
        ("image", "no-cors"),  # <img src="http://127.0.0.1:8765/ask?q=…">
        ("empty", "no-cors"),  # fetch(…, {mode: 'no-cors'})
        ("empty", "cors"),  # fetch(…) — CORS hides the response, but the query already ran
        ("script", "no-cors"),
        ("iframe", "navigate"),  # a navigation, but not a top-level one
    ],
)
def test_cross_site_subresource_requests_are_rejected(dest, mode):
    reason = _reason(**{"sec-fetch-site": "cross-site", "sec-fetch-dest": dest, "sec-fetch-mode": mode})
    assert reason is not None
    assert "cross-site" in reason


def test_cross_site_top_level_navigation_is_accepted():
    """Clicking a link to the console is visible and one-shot, not a drive-by."""
    assert (
        _reason(
            **{"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "document"}
        )
        is None
    )


def test_cross_site_form_post_navigation_is_rejected():
    """A cross-site form submit is also navigate/document; only safe methods ride that path."""
    assert (
        _reason(
            method="POST",
            **{"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "document"},
        )
        is not None
    )


# --- Wired into the app -------------------------------------------------------


def test_guarded_app_serves_normal_loopback_requests(guarded):
    assert guarded.get("/health").status_code == 200


def test_guarded_app_rejects_a_rebound_host(guarded):
    r = guarded.get("/health", headers={"Host": "engram.evil.test:8765"})
    assert r.status_code == 403
    assert "Host" in r.json()["detail"]


def test_cross_site_drive_by_is_refused_before_any_retrieval_runs(db, monkeypatch):
    """The cost of a drive-by is a local inference run, so the guard has to fire ahead of the
    handler — patched before `create_app`, which binds the service functions into its closure."""
    import crossmodalrag.service as svc

    def _boom(*args, **kwargs):  # pragma: no cover - reaching it is the failure
        raise AssertionError("retrieval ran for a cross-site request")

    monkeypatch.setattr(svc, "answer_payload", _boom)
    from crossmodalrag.api import create_app

    client = TestClient(create_app(), base_url=SERVED)
    r = client.get(
        "/ask",
        params={"q": "parser"},
        headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Dest": "image", "Sec-Fetch-Mode": "no-cors"},
    )
    assert r.status_code == 403
    assert "cross-site" in r.json()["detail"]


def test_cross_site_delete_is_rejected(guarded):
    assert guarded.delete("/conversations/1", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_same_origin_ask_still_reaches_the_handler(guarded):
    """The mirror of the drive-by test: the console's own request is untouched."""
    r = guarded.get(
        "/ask",
        params={"q": "parser", "use_llm": "false"},
        headers={"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Dest": "empty", "Sec-Fetch-Mode": "cors"},
    )
    assert r.status_code == 200
    assert "answer" in r.json()


def test_guard_leaves_streaming_responses_intact(guarded):
    """The guard is pure-ASGI pass-through, so NDJSON still streams line by line."""
    with guarded.stream("GET", "/ask/stream", params={"q": "parser", "use_llm": "false"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in r.iter_lines() if line.strip()]
    assert events[-1]["type"] == "answer"


def test_create_app_accepts_extra_hosts_for_a_deliberate_non_loopback_bind(db):
    from crossmodalrag.api import create_app

    client = TestClient(create_app(allowed_hosts=["192.168.1.5"]), base_url=SERVED)
    assert client.get("/health", headers={"Host": "192.168.1.5:8765"}).status_code == 200
    assert client.get("/health", headers={"Host": "evil.test:8765"}).status_code == 403


def test_allowed_hosts_env_var_extends_the_loopback_set(db, monkeypatch):
    from crossmodalrag.api import create_app

    monkeypatch.setenv("CMRAG_API_ALLOWED_HOSTS", "engram.lan, 192.168.1.5")
    client = TestClient(create_app(), base_url=SERVED)
    assert client.get("/health", headers={"Host": "engram.lan:8765"}).status_code == 200
    assert client.get("/health", headers={"Host": "192.168.1.5:8765"}).status_code == 200
    assert client.get("/health", headers={"Host": "evil.test:8765"}).status_code == 403


def test_serve_cmd_allows_the_host_it_was_told_to_bind(monkeypatch):
    """`mem serve --host 192.168.1.5` must not 403 the very address it advertises."""
    import uvicorn

    from crossmodalrag import api, cli

    captured: dict = {}

    def _create_app(*, allowed_hosts=None):
        captured["allowed_hosts"] = allowed_hosts
        return object()

    monkeypatch.setattr(api, "create_app", _create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    cli.serve_cmd(host="192.168.1.5", port=8765)
    assert captured["allowed_hosts"] == ["192.168.1.5"]


def test_serve_cmd_on_loopback_adds_no_extra_hosts(monkeypatch):
    import uvicorn

    from crossmodalrag import api, cli

    captured: dict = {}

    def _create_app(*, allowed_hosts=None):
        captured["allowed_hosts"] = allowed_hosts
        return object()

    monkeypatch.setattr(api, "create_app", _create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    cli.serve_cmd()
    assert captured["allowed_hosts"] == []
