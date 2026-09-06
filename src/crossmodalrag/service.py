"""Library service layer: orchestration shared by the CLI and the local HTTP API.

Keeping the `ask` retrieval+synthesis and the `doctor` health report here (rather than inline in the
CLI) means the API is a *thin client* — it calls these functions and returns the existing JSON
contracts, adding no retrieval/derivation logic of its own.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from crossmodalrag.config import (
    CONNECTOR_ENV_PREFIX,
    get_config_path,
    get_connector_paths,
    get_db_path,
    get_extract_model,
    get_llm_base_url,
    get_llm_model,
    get_llm_timeout,
    get_usage_halflife_days,
    load_config,
)
from crossmodalrag.db import connect, init_db
from crossmodalrag.embed.provider import get_default_provider
from crossmodalrag.generate.answer import generated_answer_to_dict, template_answer_to_dict
from crossmodalrag.generate.provider import LLMUnavailable, get_default_llm_provider
from crossmodalrag.generate.synthesize import synthesize_answer, synthesize_answer_stream
from crossmodalrag.memory.concepts import list_concept_views
from crossmodalrag.memory.episodes import list_episode_timeline
from crossmodalrag.memory.forgetting import LEVEL_NAMES, compute_forgetting_risk, forgetting_risk_to_dict
from crossmodalrag.memory.integrity import memory_stats
from crossmodalrag.memory.recall import generate_recall_cards, recall_card_to_dict
from crossmodalrag.retrieve.hybrid import DEFAULT_PROFILE, retrieve
from crossmodalrag.retrieve.nodes import candidate_chunk_ids, retrieve_nodes
from crossmodalrag.retrieve.rerank import resolve_source_types


def retrieve_for_answer(
    conn: sqlite3.Connection,
    *,
    query: str,
    top_k: int = 5,
    profile: str = DEFAULT_PROFILE,
    level: str = "evidence",
    modalities: list[str] | None = None,
):
    """Retrieve evidence for a query, drilling memory-level entry points down to L0. Read-only.

    Returns ``(hits, matched_nodes)`` — the L0 evidence hits and (for non-`evidence` levels) the
    matched memory nodes that were drilled down.
    """
    restrict_source_types = resolve_source_types(modalities)
    matched_nodes = []
    if level == "evidence":
        hits = retrieve(
            conn, query=query, top_k=top_k, profile=profile, restrict_source_types=restrict_source_types
        )
    else:
        matched_nodes = retrieve_nodes(conn, query, level=level, top_k=top_k, profile=profile)
        chunk_ids = candidate_chunk_ids(conn, matched_nodes)
        hits = (
            retrieve(
                conn,
                query=query,
                top_k=top_k,
                profile=profile,
                restrict_chunk_ids=chunk_ids,
                restrict_source_types=restrict_source_types,
            )
            if chunk_ids
            else []
        )
    return hits, matched_nodes


def matched_nodes_payload(matched_nodes) -> list[dict]:
    return [
        {
            "node_id": n.node_id,
            "level": n.level,
            "node_type": n.node_type,
            "title": n.title,
            "centrality": n.centrality,
            "score": n.score,
        }
        for n in matched_nodes
    ]


def answer_payload(
    conn: sqlite3.Connection,
    *,
    query: str,
    top_k: int = 5,
    profile: str = DEFAULT_PROFILE,
    level: str = "evidence",
    modalities: list[str] | None = None,
    use_llm: bool = True,
) -> dict:
    """The `mem ask --json` payload for a query (no printing, no usage tracking).

    Synthesizes a grounded answer when the LLM is available, else returns the deterministic evidence
    template. Either way the payload carries provenance for each evidence item.
    """
    start = time.monotonic()
    hits, matched_nodes = retrieve_for_answer(
        conn, query=query, top_k=top_k, profile=profile, level=level, modalities=modalities
    )
    provider = get_default_llm_provider() if use_llm else None
    if provider is not None:
        try:
            gen = synthesize_answer(query, hits, provider)
        except LLMUnavailable:
            provider = None
        else:
            data = generated_answer_to_dict(gen, total_seconds=time.monotonic() - start)
            if matched_nodes:
                data["matched_nodes"] = matched_nodes_payload(matched_nodes)
            return data

    data = template_answer_to_dict(query, hits, total_seconds=time.monotonic() - start)
    if matched_nodes:
        data["matched_nodes"] = matched_nodes_payload(matched_nodes)
    return data


def answer_stream_events(
    conn: sqlite3.Connection,
    *,
    query: str,
    top_k: int = 5,
    profile: str = DEFAULT_PROFILE,
    level: str = "evidence",
    modalities: list[str] | None = None,
    use_llm: bool = True,
):
    """Streaming variant of :func:`answer_payload`: an event iterator for live UIs.

    Yields zero or more ``{"type": "token", "text": ...}`` events as the LLM
    generates, then exactly one ``{"type": "answer", "data": ...}`` event whose
    ``data`` is the same payload :func:`answer_payload` returns (identical
    contract — citations/abstention computed on the full output). Gate
    abstentions, ``use_llm=False``, and an unreachable LLM (including
    mid-stream failure) all still end with the final ``answer`` event, so
    consumers can rely on it unconditionally.

    Retrieval happens on the first ``next()``; the DB is not touched after
    that. Callers that consume the stream outside the connection's scope (or
    thread) should retrieve first and use :func:`stream_answer_events`.
    """
    start = time.monotonic()
    hits, matched_nodes = retrieve_for_answer(
        conn, query=query, top_k=top_k, profile=profile, level=level, modalities=modalities
    )
    yield from stream_answer_events(
        query=query, hits=hits, matched_nodes=matched_nodes, use_llm=use_llm, start=start
    )


def stream_answer_events(
    *,
    query: str,
    hits,
    matched_nodes,
    use_llm: bool = True,
    start: float | None = None,
    history: str | None = None,
):
    """DB-free core of :func:`answer_stream_events`, fed with already-retrieved evidence.

    Holding no sqlite objects, this generator may be consumed — and closed —
    from any thread and long after the retrieval connection is gone. That is
    exactly the API streaming-response situation: the ASGI server iterates
    (and, on client disconnect, closes) the generator on arbitrary worker
    threads, while sqlite connections are bound to their creating thread.

    ``history`` is an optional pre-rendered conversation block
    (``chat.render_history``) threaded into the synthesis prompt for
    multi-turn sessions; ``None`` is byte-identical to single-turn.
    """
    start = time.monotonic() if start is None else start
    provider = get_default_llm_provider() if use_llm else None
    if provider is not None:
        gen = None
        try:
            stream = synthesize_answer_stream(query, hits, provider, history=history)
            while True:
                try:
                    fragment = next(stream)
                except StopIteration as stop:
                    gen = stop.value
                    break
                yield {"type": "token", "text": fragment}
        except LLMUnavailable:
            provider = None
        if gen is not None:
            data = generated_answer_to_dict(gen, total_seconds=time.monotonic() - start)
            if matched_nodes:
                data["matched_nodes"] = matched_nodes_payload(matched_nodes)
            yield {"type": "answer", "data": data}
            return

    data = template_answer_to_dict(query, hits, total_seconds=time.monotonic() - start)
    if matched_nodes:
        data["matched_nodes"] = matched_nodes_payload(matched_nodes)
    yield {"type": "answer", "data": data}


@contextmanager
def open_store(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the memory DB (default: the configured path), apply the idempotent schema, close on exit.

    Every thin client opens one connection per request/tool call through this, so sqlite handles
    never cross threads. ``connect`` creates a missing file; callers that must not do that (an MCP
    server spawned in an arbitrary directory) check ``path.exists()`` first.
    """
    conn = connect(db_path if db_path is not None else get_db_path())
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _levels(level: str) -> tuple[int, ...]:
    levels = LEVEL_NAMES.get(level)
    if levels is None:
        raise ValueError(f"Unknown level '{level}'. Expected one of: {', '.join(LEVEL_NAMES)}.")
    return levels


def concepts_payload(conn: sqlite3.Connection, *, top: int = 20) -> dict:
    """The `mem concepts --json` contract (shared by GET /concepts and the MCP `concepts` tool)."""
    return {"concepts": list_concept_views(conn, top=top)}


def timeline_payload(conn: sqlite3.Connection, *, limit: int = 50) -> dict:
    """The `mem timeline --json` contract (GET /timeline, MCP `timeline`)."""
    return {"timeline": list_episode_timeline(conn, limit=limit)}


def forgetting_payload(
    conn: sqlite3.Connection,
    *,
    level: str = "concept",
    top: int = 10,
    min_support: int = 1,
    now: datetime | None = None,
) -> dict:
    """The `mem forgetting --json` contract. ``now`` is injectable for frozen-clock reproducibility.

    Raises ``ValueError`` for a level outside ``LEVEL_NAMES``.
    """
    items = compute_forgetting_risk(
        conn,
        now=now if now is not None else _now(),
        halflife_days=get_usage_halflife_days(),
        levels=_levels(level),
        min_support=min_support,
        top=top,
    )
    return {"level": level, "forgetting": [forgetting_risk_to_dict(i) for i in items]}


def recall_payload(
    conn: sqlite3.Connection,
    *,
    level: str = "concept",
    top: int = 10,
    min_support: int = 1,
    regenerate: bool = False,
    now: datetime | None = None,
) -> dict:
    """The `mem recall --json` contract.

    Cards come from the extraction model (``CMRAG_EXTRACT_MODEL``) when Ollama is reachable and
    from the deterministic fallback otherwise; either way they are cached in ``recall_cards``
    (a derived, fingerprint-keyed cache, never ingestion or history state). Raises ``ValueError``
    for an unknown level.
    """
    levels = _levels(level)
    provider = get_default_llm_provider(get_extract_model())
    cards = generate_recall_cards(
        conn,
        provider,
        now=now if now is not None else _now(),
        halflife_days=get_usage_halflife_days(),
        levels=levels,
        top=top,
        min_support=min_support,
        regenerate=regenerate,
    )
    return {"level": level, "recall": [recall_card_to_dict(c) for c in cards]}


class ConversationNotFound(LookupError):
    """Raised when a chat turn or read targets a conversation id that doesn't exist."""


def conversations_payload(conn: sqlite3.Connection, *, top: int | None = None) -> dict:
    """The `mem history --json` list contract (shared verbatim by GET /conversations)."""
    from crossmodalrag.config import save_history_enabled
    from crossmodalrag.conversations.contract import conversation_to_dict
    from crossmodalrag.conversations.store import list_conversations

    return {
        "save_enabled": save_history_enabled(),
        "total": len(list_conversations(conn)),
        "conversations": [
            conversation_to_dict(conn, c, include_messages=False)
            for c in list_conversations(conn, top=top)
        ],
    }


def conversation_payload(conn: sqlite3.Connection, conversation_id: int) -> dict:
    """One conversation with its ordered messages (GET /conversations/{id})."""
    from crossmodalrag.conversations.contract import conversation_to_dict
    from crossmodalrag.conversations.store import get_conversation

    conversation = get_conversation(conn, conversation_id)
    if conversation is None:
        raise ConversationNotFound(f"No saved conversation with id {conversation_id}.")
    return conversation_to_dict(conn, conversation)


def chat_stream_events(
    *,
    query: str,
    conversation_id: int | None = None,
    top_k: int = 5,
    profile: str = DEFAULT_PROFILE,
    level: str = "evidence",
    modalities: list[str] | None = None,
    use_llm: bool = True,
    save: bool = True,
):
    """One persisted multi-turn chat turn as an NDJSON-able event stream (the web chat).

    The server-side twin of the CLI chat loop: resumes the stored conversation's
    context (`turns_from_messages` + the session cap), retrieves independently for
    THIS turn, streams tokens, then persists the turn — including abstained ones —
    via `SessionRecorder` (best-effort, auto-titled on first turn). Yields
    `{"type":"token",...}` events then one final
    `{"type":"answer","data":…, "conversation_id":…, "conversation":…}` event;
    `conversation_id` is None when saving is disabled (context is then not carried
    server-side) or the turn wasn't persisted (template path).

    Setup (resume + retrieval, including the ConversationNotFound check) runs
    EAGERLY on the caller's thread with a connection opened and closed here;
    the returned generator holds no sqlite objects (the recorder opens its own
    per-call connection), so it may be consumed from any worker thread.
    """
    from crossmodalrag.chat import ChatSession, render_history
    from crossmodalrag.config import save_history_enabled
    from crossmodalrag.conversations.contract import conversation_to_dict
    from crossmodalrag.conversations.naming import generate_conversation_title
    from crossmodalrag.conversations.recorder import SessionRecorder
    from crossmodalrag.conversations.resume import next_turn_index, turns_from_messages
    from crossmodalrag.conversations.store import get_conversation, list_messages

    start = time.monotonic()
    db_path = get_db_path()
    saving = save and save_history_enabled()

    history = None
    resume_index = 0
    conn = connect(db_path)
    try:
        init_db(conn)
        if conversation_id is not None:
            if get_conversation(conn, conversation_id) is None:
                raise ConversationNotFound(f"No saved conversation with id {conversation_id}.")
            messages = list_messages(conn, conversation_id)
            session = ChatSession()
            for turn in turns_from_messages(messages):
                session.add_turn(turn.query, turn.answer_text)  # context cap applies
            history = render_history(session.turns) or None
            resume_index = next_turn_index(messages)
        hits, matched_nodes = retrieve_for_answer(
            conn, query=query, top_k=top_k, profile=profile, level=level, modalities=modalities
        )
    finally:
        conn.close()

    def _title_fn(first_query: str, answer_text: str) -> str | None:
        provider = get_default_llm_provider() if use_llm else None
        if provider is None:
            return None
        return generate_conversation_title(provider, query=first_query, answer_text=answer_text)

    recorder = SessionRecorder(db_path, enabled=saving, title_fn=_title_fn)
    if conversation_id is not None:
        recorder.attach(conversation_id, next_turn_index=resume_index)

    def _events():
        provider = get_default_llm_provider() if use_llm else None
        gen = None
        if provider is not None:
            try:
                stream = synthesize_answer_stream(query, hits, provider, history=history)
                while True:
                    try:
                        fragment = next(stream)
                    except StopIteration as stop:
                        gen = stop.value
                        break
                    yield {"type": "token", "text": fragment}
            except LLMUnavailable:
                pass

        if gen is not None:
            recorder.record_turn(query, gen)
            data = generated_answer_to_dict(gen, total_seconds=time.monotonic() - start)
        else:
            # Template/no-LLM turns are not persisted (not a synthesized answer).
            data = template_answer_to_dict(query, hits, total_seconds=time.monotonic() - start)
        if matched_nodes:
            data["matched_nodes"] = matched_nodes_payload(matched_nodes)

        final: dict = {"type": "answer", "data": data, "conversation_id": recorder.conversation_id}
        if recorder.conversation_id is not None:
            inner = connect(db_path)
            try:
                conversation = get_conversation(inner, recorder.conversation_id)
                if conversation is not None:
                    final["conversation"] = conversation_to_dict(
                        inner, conversation, include_messages=False
                    )
            finally:
                inner.close()
        yield final

    return _events()


def ping_ollama() -> bool:
    """Best-effort reachability check for the local Ollama server (never raises)."""
    import urllib.request

    url = f"{get_llm_base_url()}/api/tags"
    timeout = min(get_llm_timeout(), 2.0)
    try:
        # urlopen targets the locally configured Ollama base URL, never user input.
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def health_report(db_path: Path | None = None) -> dict:
    """Read-only health report: DB, installed extras, Ollama reachability, models, config, memory.

    The payload behind `mem doctor`, the API `/health` endpoint and the MCP `status` tool. ``db_path``
    names the store to report on; ``None`` means the configured one.
    """
    from crossmodalrag.capabilities import has_ocr, has_pdf

    db_path = db_path if db_path is not None else get_db_path()
    db_exists = db_path.exists()
    provider = get_default_provider()
    embed_model = provider.name if provider is not None else None

    stats = None
    if db_exists:
        conn = connect(db_path)
        try:
            init_db(conn)
            stats = memory_stats(conn)
        finally:
            conn.close()

    config_path = get_config_path()
    return {
        "db": {
            "path": str(db_path),
            "exists": db_exists,
            "size_bytes": (db_path.stat().st_size if db_exists else 0),
        },
        "extras": {"embeddings": provider is not None, "pdf": has_pdf(), "ocr": has_ocr()},
        "ollama": {"base_url": get_llm_base_url(), "reachable": ping_ollama()},
        "models": {"embed": embed_model, "llm": get_llm_model(), "extract": get_extract_model()},
        "config": {"path": (str(config_path) if config_path is not None else None), "loaded": bool(load_config())},
        "connectors": {name: len(get_connector_paths(name)) for name in CONNECTOR_ENV_PREFIX},
        "memory": stats,
    }
