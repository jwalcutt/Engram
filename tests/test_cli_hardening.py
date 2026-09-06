from __future__ import annotations

import io
import json
import os
import sys

import pytest

from crossmodalrag import cli
from crossmodalrag.progress import ProgressReporter, make_progress

CONNECTOR_PREFIXES = ("OBSIDIAN_VAULT_PATH", "REPO_PATH", "PDF_PATH", "IMAGE_PATH")


def _isolate_env(monkeypatch) -> None:
    """Clear every numbered connector env var and stop `main()` from reloading the project .env.

    `cli.main` calls `load_dotenv`, which would re-populate the developer's real `REPO_PATH_*` etc.
    process-wide; no-op it so a test only sees the env it sets explicitly.
    """
    for key in list(os.environ):
        if any(key.startswith(f"{p}_") for p in CONNECTOR_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)


def _run(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["mem", *argv])
    cli.main()


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


# --- progress utility ---------------------------------------------------------


def test_make_progress_noop_when_not_tty():
    assert make_progress("x", stream=io.StringIO()) is None  # not a TTY -> disabled


def test_make_progress_disabled_flag():
    assert make_progress("x", enabled=False, stream=_FakeTTY()) is None


def test_progress_reporter_writes_on_tty():
    s = _FakeTTY()
    reporter = ProgressReporter("embed", stream=s)
    for i in range(1, 4):
        reporter(i, 3)
    assert s.getvalue() == "\rembed: 1/3\rembed: 2/3\rembed: 3/3\n"


def test_progress_callback_invoked_per_item():
    s = _FakeTTY()
    reporter = make_progress("ingest", stream=s)
    assert reporter is not None
    calls = 0
    for i in range(1, 6):
        reporter(i, 5)
        calls += 1
    assert calls == 5
    assert s.getvalue().endswith("ingest: 5/5\n")


# --- error handling + exit codes ----------------------------------------------


def test_bad_ingest_path_clean_error_exit_1(monkeypatch, capsys, tmp_path):
    _isolate_env(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["ingest-notes", str(tmp_path / "nope")])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "Traceback" not in err  # no stack trace for an expected failure


def test_unknown_command_exits_2(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["frobnicate"])
    assert exc.value.code == 2  # argparse usage error


# --- mem sync -----------------------------------------------------------------


@pytest.fixture
def sync_env(monkeypatch, tmp_path):
    """A configured notes vault + a (capability-disabled) pdf path; other connectors unset."""
    _isolate_env(monkeypatch)
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nalpha content\n", encoding="utf-8")
    (vault / "b.md").write_text("# B\n\nbeta content\n", encoding="utf-8")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH_1", str(vault))
    monkeypatch.setenv("PDF_PATH_1", str(pdf_dir))
    monkeypatch.setenv("CMRAG_DB_PATH", str(tmp_path / "memory.db"))
    # Keep it offline/fast: no embedder, and force the pdf extra to look absent.
    monkeypatch.setattr(cli, "get_default_provider", lambda *a, **k: None)
    import crossmodalrag.capabilities as caps
    monkeypatch.setattr(caps, "has_pdf", lambda: False)
    monkeypatch.setattr(caps, "has_ocr", lambda: False)
    return tmp_path


def _sync_json(monkeypatch, capsys, *extra) -> dict:
    _run(monkeypatch, ["sync", "--json", *extra])
    return json.loads(capsys.readouterr().out)


def test_sync_ingests_notes_and_is_idempotent(sync_env, monkeypatch, capsys):
    payload = _sync_json(monkeypatch, capsys)
    by = {c["connector"]: c for c in payload["connectors"]}
    assert by["notes"]["inserted"] > 0 and by["notes"]["status"] == "ok"
    assert by["pdf"]["status"].startswith("skipped")  # extra forced absent
    assert by["git"]["status"] == "no-paths" and by["image"]["status"] == "no-paths"
    assert payload["total_inserted"] == by["notes"]["inserted"]

    # Re-running re-chunks nothing (fingerprint-skip idempotency).
    again = _sync_json(monkeypatch, capsys)
    assert {c["connector"]: c["inserted"] for c in again["connectors"]}["notes"] == 0
    assert again["total_inserted"] == 0


def test_sync_only_filter(sync_env, monkeypatch, capsys):
    payload = _sync_json(monkeypatch, capsys, "--only", "git")
    names = {c["connector"] for c in payload["connectors"]}
    assert names == {"git"}  # only the selected connector is reported


# --- mem doctor ---------------------------------------------------------------


def test_doctor_json_reports_health_offline(monkeypatch, capsys, tmp_path):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("CMRAG_DB_PATH", str(tmp_path / "memory.db"))
    import crossmodalrag.service as svc
    monkeypatch.setattr(svc, "ping_ollama", lambda: False)  # deterministic, offline
    _run(monkeypatch, ["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"db", "extras", "ollama", "models", "config", "connectors", "memory"}
    assert report["ollama"]["reachable"] is False  # reported, not raised
    assert set(report["extras"]) == {"embeddings", "pdf", "ocr"}


def test_doctor_text_runs(monkeypatch, capsys, tmp_path):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("CMRAG_DB_PATH", str(tmp_path / "memory.db"))
    import crossmodalrag.service as svc
    monkeypatch.setattr(svc, "ping_ollama", lambda: False)
    _run(monkeypatch, ["doctor"])
    out = capsys.readouterr().out
    assert "doctor" in out and "Ollama: reachable=False" in out


# --- mem mcp: the stdio MCP server entry point ----------------------------------


def test_mcp_cmd_refuses_missing_db(tmp_path, monkeypatch, capsys):
    """A client-spawned server has an arbitrary cwd; a missing DB must fail loudly, not be created."""
    missing = tmp_path / "nope" / "memory.db"
    with pytest.raises(cli.CLIError, match=str(missing)):
        cli.mcp_cmd(db=str(missing))
    assert not missing.exists()
    assert capsys.readouterr().out == ""


def test_mcp_cmd_missing_extra_is_clierror(tmp_path, monkeypatch):
    from crossmodalrag.db import connect, init_db

    db = tmp_path / "memory.db"
    init_db(connect(db))
    monkeypatch.setitem(sys.modules, "mcp", None)  # `import mcp` now raises ModuleNotFoundError
    with pytest.raises(cli.CLIError, match=r"\[mcp\]"):
        cli.mcp_cmd(db=str(db))


def test_mcp_cmd_runs_server_with_stdout_untouched(tmp_path, monkeypatch, capsys):
    pytest.importorskip("mcp")
    import crossmodalrag.mcp_server as mcp_server
    from crossmodalrag.db import connect, init_db

    db = tmp_path / "memory.db"
    init_db(connect(db))
    calls = []
    monkeypatch.setattr(mcp_server, "run_stdio", lambda **kw: calls.append(kw))
    _run(monkeypatch, ["mcp", "--db", str(db)])
    assert calls == [{"db_path": db.resolve()}]
    out, err = capsys.readouterr()
    assert out == ""  # stdout is the protocol channel
    assert str(db) in err
