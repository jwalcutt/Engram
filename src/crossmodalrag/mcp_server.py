"""Local MCP server: Engram's read views as tools an AI assistant can call over stdio.

A third thin client over ``service.py``, next to the CLI and the HTTP API. Each tool returns the
matching ``--json`` contract verbatim as structured content, so every surface renders the same
payload and ``tests/test_json_contracts.py`` stays the single shape pin.

Posture:

- **stdio only.** The client spawns ``mem mcp`` and talks over its stdin/stdout; there is no
  network listener, so the loopback ``Host``/fetch-metadata guards the HTTP API needs do not
  apply here. Nothing leaves the machine.
- **Read tools only.** No ingestion, no history mutation, no usage tracking. The one write is
  ``recall``'s card cache (``recall_cards``), a derived, fingerprint-keyed cache the HTTP API's
  ``GET /recall`` fills the same way; it never touches ingestion or history state.
- **stdout is the wire.** Nothing in this module prints; logs go to stderr.
- **A missing database is an error, not an empty store.** The spawning client chooses the working
  directory (Claude Desktop uses ``/``), so ``./data/memory.db`` and ``./.env`` rarely resolve.
  Tools refuse to open a path that does not exist instead of creating an empty one there.

Requires the opt-in ``[mcp]`` extra; the module imports without it, and ``build_server`` raises
``MissingMCPBackend`` when the SDK is absent.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from crossmodalrag.config import get_db_path, get_default_profile, get_default_top_k
from crossmodalrag.retrieve.hybrid import DEFAULT_PROFILE

# Schema-level enums. Each mirrors a library table (pinned by tests) so a bad value is rejected
# by argument validation before any database work.
AskLevel = Literal["evidence", "event", "episode", "concept"]
MemoryLevel = Literal["event", "episode", "concept", "all"]
Profile = Literal["balanced", "relevant", "recent", "usage"]
Modality = Literal["text", "code", "pdf", "image"]

SERVER_NAME = "engram"

INSTRUCTIONS = (
    "Engram is the user's local, evidence-grounded memory store (notes, commits, PDFs, images). "
    "All tools are read-only and nothing leaves the machine. Use `ask` to answer questions from "
    "the user's own material: every item in `evidence` carries `evidence_id`, `source_uri`, "
    "`locator` (the citable location, e.g. `spec.pdf p.4` or `repo@sha`), and `chunk_id`; cite "
    "`locator` when you repeat a claim. `abstained: true` means the store holds no grounded "
    "answer, not that the answer is no. `use_llm=false` returns the ranked evidence without "
    "synthesis and needs no local model; it is the fast path. `concepts`, `timeline`, "
    "`forgetting` and `recall` browse the derived memory hierarchy; `history` and `conversation` "
    "read saved chat sessions; `status` reports the database path, installed extras and whether "
    "the local model is reachable."
)


class MissingMCPBackend(RuntimeError):
    """Raised when the MCP server is requested without the ``[mcp]`` extra installed."""


def resolve_db_path(db_path: Path | None = None) -> Path:
    """The database this server will read: an explicit path, else the configured one."""
    return Path(db_path).expanduser().resolve() if db_path is not None else get_db_path()


def build_server(*, db_path: Path | None = None):
    """Build the MCP server. Raises ``MissingMCPBackend`` if the ``[mcp]`` extra is not installed.

    ``db_path`` pins the store to serve; ``None`` follows ``CMRAG_DB_PATH`` / the default. Config-
    derived defaults (`top_k`, `profile`) are read once here so they appear as real defaults in
    the published tool schema.
    """
    try:
        from mcp.server import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as exc:
        raise MissingMCPBackend(
            "The MCP server requires the [mcp] extra. Run: pip install -e \".[mcp]\""
        ) from exc

    from crossmodalrag.service import (
        ConversationNotFound,
        answer_payload,
        concepts_payload,
        conversation_payload,
        conversations_payload,
        forgetting_payload,
        health_report,
        open_store,
        recall_payload,
        timeline_payload,
    )

    default_top_k = get_default_top_k(5)
    default_profile = get_default_profile(DEFAULT_PROFILE)
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    read_only_idempotent = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

    @contextmanager
    def _store() -> Iterator[Any]:
        # One connection per call, opened on the worker thread that runs the tool, so sqlite
        # handles never cross threads. Refuse (rather than create) a missing file.
        path = resolve_db_path(db_path)
        if not path.exists():
            raise ToolError(
                f"No memory database at {path}. Start the server with `mem mcp --db /path/to/memory.db` "
                "or set CMRAG_DB_PATH in the MCP client configuration."
            )
        with open_store(path) as conn:
            yield conn

    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)

    @server.tool(annotations=read_only)
    def ask(
        query: str,
        top_k: int = default_top_k,
        profile: Profile = default_profile,
        level: AskLevel = "evidence",
        modalities: list[Modality] | None = None,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        """Answer a question from the user's memory store, grounded in retrieved evidence.

        Returns the answer (or `abstained: true` with a reason when the evidence is too weak) and
        an `evidence` list; each item carries `evidence_id`, `source_uri`, `locator` (the citable
        location such as `spec.pdf p.4` or `repo@sha`), `chunk_id`, `modality`, scores and an
        excerpt. `level` enters retrieval at a memory level (event/episode/concept) and drills down
        to evidence. `modalities` restricts source types. `use_llm=false` skips synthesis and
        returns ranked evidence only (fast, needs no local model). Never records usage.
        """
        with _store() as conn:
            return answer_payload(
                conn, query=query, top_k=top_k, profile=profile, level=level,
                modalities=list(modalities) if modalities else None, use_llm=use_llm,
            )

    @server.tool(annotations=read_only)
    def concepts(top: int = 20) -> dict[str, Any]:
        """List the most central concepts (L3 memory nodes) with `node_id`, title, centrality and member count."""
        with _store() as conn:
            return concepts_payload(conn, top=top)

    @server.tool(annotations=read_only)
    def timeline(limit: int = 50) -> dict[str, Any]:
        """List episodes (L2 memory nodes) oldest first, with `node_id`, title, time window and member count."""
        with _store() as conn:
            return timeline_payload(conn, limit=limit)

    @server.tool(annotations=read_only)
    def forgetting(level: MemoryLevel = "concept", top: int = 10, min_support: int = 1) -> dict[str, Any]:
        """Rank what the user is most likely forgetting: important memories not revisited recently.

        Each item exposes the estimate's components (`importance`, `staleness`, `confidence`,
        `support`) and `evidence_source_uris`; treat low `confidence` or `support` as weak.
        """
        with _store() as conn:
            return forgetting_payload(conn, level=level, top=top, min_support=min_support)

    @server.tool(annotations=read_only_idempotent)
    def recall(level: MemoryLevel = "concept", top: int = 10, min_support: int = 1) -> dict[str, Any]:
        """Active-recall practice cards for the memories most at risk of being forgotten.

        Each card cites `evidence_source_uris`; `generated_by` is `llm` or `fallback` (no local
        model). Cards are cached in the store's `recall_cards` table, the only write any tool
        performs; it is a derived cache and never touches ingested or chat-history data.
        """
        with _store() as conn:
            return recall_payload(conn, level=level, top=top, min_support=min_support)

    @server.tool(annotations=read_only)
    def history(top: int = 10) -> dict[str, Any]:
        """List saved chat conversations, newest first (`id`, title, timestamps, message count)."""
        with _store() as conn:
            return conversations_payload(conn, top=top)

    @server.tool(annotations=read_only)
    def conversation(conversation_id: int) -> dict[str, Any]:
        """One saved conversation with its ordered messages; assistant messages keep their cited evidence."""
        with _store() as conn:
            try:
                return conversation_payload(conn, conversation_id)
            except ConversationNotFound as exc:
                raise ToolError(str(exc)) from exc

    @server.tool(annotations=read_only)
    def status() -> dict[str, Any]:
        """Health report: database path and size, installed extras, local model reachability, memory stats."""
        return health_report(resolve_db_path(db_path))

    return server


def run_stdio(*, db_path: Path | None = None) -> None:
    """Serve over the process's stdin/stdout until the client disconnects. Logs go to stderr."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    build_server(db_path=db_path).run()


def main(argv: list[str] | None = None) -> None:
    """`python -m crossmodalrag.mcp_server [--db PATH]` (the `mem mcp` command is the documented entry)."""
    parser = argparse.ArgumentParser(description="Serve Engram's read-only memory tools over stdio (MCP).")
    parser.add_argument("--db", default=None, help="Memory DB to serve (default: CMRAG_DB_PATH or ./data/memory.db).")
    args = parser.parse_args(argv)
    path = resolve_db_path(Path(args.db) if args.db else None)
    if not path.exists():
        print(f"error: No memory database at {path}. Pass --db or set CMRAG_DB_PATH.", file=sys.stderr)
        raise SystemExit(1)
    run_stdio(db_path=path)


if __name__ == "__main__":
    main()
