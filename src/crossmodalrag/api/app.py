"""Local HTTP API exposing the existing JSON contracts (the UI/Obsidian boundary).

A thin FastAPI wrapper over the library functions — it adds no retrieval/derivation logic.
Read-first: every memory/engine surface is GET-only and never writes. The explicit exceptions
touch ONLY the user-owned, additive chat-history tables (``conversations``/``messages`` — never
ingestion or derivation state): ``POST /chat/stream`` (the web chat; appends a turn, respecting
``CMRAG_SAVE_HISTORY`` and the per-request ``save`` flag), ``PATCH /conversations/{id}``
(rename), and ``DELETE /conversations/{id}`` (delete one saved conversation — the API twin of
``mem history --clear --id``).
Requires the opt-in ``[ui]`` extra; the module imports without it, and ``create_app`` raises
``MissingUIBackend`` when FastAPI is absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Built web UI, produced by `web/` (Vite) and committed here. Served at the API
# root when present; the API routes above take precedence over the SPA's static mount.
STATIC_DIR = Path(__file__).resolve().parent / "static"


class MissingUIBackend(RuntimeError):
    """Raised when the local API is requested without the ``[ui]`` extra installed."""


@contextmanager
def _conn():
    from crossmodalrag.config import get_db_path
    from crossmodalrag.db import connect, init_db

    conn = connect(get_db_path())
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()


def create_app(*, allowed_hosts: Sequence[str] | None = None):
    """Build the FastAPI app. Raises ``MissingUIBackend`` if the ``[ui]`` extra is not installed.

    ``allowed_hosts`` names additional hostnames the ``Host`` guard accepts, on top of loopback
    and ``CMRAG_API_ALLOWED_HOSTS`` — `mem serve` passes the address it was told to bind.
    """
    try:
        from fastapi import FastAPI, HTTPException, Query
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise MissingUIBackend(
            "The local API requires the [ui] extra. Run: pip install -e \".[ui]\""
        ) from exc

    from crossmodalrag.config import get_usage_halflife_days
    from crossmodalrag.evaluation import distilled_compression_ratio
    from crossmodalrag.memory.concepts import list_concept_views
    from crossmodalrag.memory.distill import distilled_summaries, distilled_summary_to_dict
    from crossmodalrag.memory.drift import concept_drift_summaries, drift_summary_to_dict
    from crossmodalrag.memory.episodes import list_episode_timeline
    from crossmodalrag.memory.forgetting import (
        LEVEL_NAMES,
        compute_forgetting_risk,
        forgetting_risk_to_dict,
    )
    from crossmodalrag.memory.integrity import memory_stats
    from crossmodalrag.memory.recall import generate_recall_cards, recall_card_to_dict
    from crossmodalrag.service import (
        ConversationNotFound,
        answer_payload,
        chat_stream_events,
        conversation_payload,
        conversations_payload,
        health_report,
        retrieve_for_answer,
        stream_answer_events,
    )
    from crossmodalrag.usage.store import usage_summaries
    from crossmodalrag.usage.strength import usage_summary_to_dict

    app = FastAPI(
        title="Engram local API",
        version="1",
        description="Read-only access to the local memory engine. Localhost-only; no auth.",
    )

    # Loopback with no auth means the browser, not the network, is the attacker: see
    # `api/guard.py` for why `Host` and `Sec-Fetch-Site` are the two headers that matter.
    # Added first so it wraps everything, the static UI mount included, and rejects before
    # routing — a cross-site `/ask` must not cost a local inference run.
    from crossmodalrag.api.guard import LOOPBACK_HOSTNAMES, LocalOriginGuard
    from crossmodalrag.config import get_api_allowed_hosts

    app.add_middleware(
        LocalOriginGuard,
        allowed_hosts=[*LOOPBACK_HOSTNAMES, *get_api_allowed_hosts(), *(allowed_hosts or [])],
    )

    def _levels(level: str):
        levels = LEVEL_NAMES.get(level)
        if levels is None:
            raise HTTPException(status_code=400, detail=f"Unknown level '{level}'.")
        return levels

    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @app.get("/health")
    def health() -> dict:
        return health_report()

    @app.get("/ask")
    def ask(
        q: str = Query(..., description="The query."),
        top_k: int = 5,
        profile: str = "balanced",
        level: str = "evidence",
        modality: list[str] | None = Query(None),
        use_llm: bool = True,
    ) -> dict:
        with _conn() as conn:
            return answer_payload(
                conn, query=q, top_k=top_k, profile=profile, level=level,
                modalities=modality, use_llm=use_llm,
            )

    @app.get("/ask/stream")
    def ask_stream(
        q: str = Query(..., description="The query."),
        top_k: int = 5,
        profile: str = "balanced",
        level: str = "evidence",
        modality: list[str] | None = Query(None),
        use_llm: bool = True,
    ):
        """Streaming `/ask`: NDJSON events — `{"type":"token","text":…}` per LLM fragment,
        then one final `{"type":"answer","data":…}` carrying the exact `/ask` payload.
        The final event always arrives (gate abstentions, `use_llm=false`, and LLM
        failures included), so clients can rely on it unconditionally.
        """
        import json
        import time

        from fastapi.responses import StreamingResponse

        # Retrieve inside this handler so the sqlite connection opens and closes on
        # one thread. The response generator below holds NO sqlite objects: the ASGI
        # server may iterate/close it on a different worker thread (e.g. on client
        # disconnect), where a thread-bound connection dies with ProgrammingError.
        start = time.monotonic()
        with _conn() as conn:
            hits, matched_nodes = retrieve_for_answer(
                conn, query=q, top_k=top_k, profile=profile, level=level, modalities=modality
            )

        def _ndjson():
            for event in stream_answer_events(
                query=q, hits=hits, matched_nodes=matched_nodes, use_llm=use_llm, start=start
            ):
                yield json.dumps(event) + "\n"

        return StreamingResponse(_ndjson(), media_type="application/x-ndjson")

    @app.get("/conversations")
    def conversations(top: int | None = None) -> dict:
        with _conn() as conn:
            return conversations_payload(conn, top=top)

    @app.get("/conversations/{conversation_id}")
    def conversation(conversation_id: int) -> dict:
        with _conn() as conn:
            try:
                return conversation_payload(conn, conversation_id)
            except ConversationNotFound as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch("/conversations/{conversation_id}")
    def rename_conversation_route(conversation_id: int, body: dict) -> dict:
        """Rename ONE saved conversation (user-owned data; overrides the auto-title).
        One of the API's explicit write paths — see the module docstring."""
        from crossmodalrag.conversations.contract import conversation_to_dict
        from crossmodalrag.conversations.store import get_conversation, rename_conversation

        title = str(body.get("title") or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="Missing or empty 'title'.")
        title = title[:200]
        with _conn() as conn:
            if not rename_conversation(conn, conversation_id, title=title):
                raise HTTPException(
                    status_code=404, detail=f"No saved conversation with id {conversation_id}."
                )
            conversation = get_conversation(conn, conversation_id)
            if conversation is None:
                # The rename above succeeded, so this only fires if the row went away in
                # between. Kept as a real branch, not an assert: assertions are stripped
                # under `python -O`, which would hand None to the contract layer instead.
                raise HTTPException(
                    status_code=404, detail=f"No saved conversation with id {conversation_id}."
                )
            return conversation_to_dict(conn, conversation, include_messages=False)

    @app.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: int) -> dict:
        """Delete ONE saved conversation (the user's own private data; no undo).
        One of the API's two explicit write paths — see the module docstring."""
        from crossmodalrag.conversations.store import clear_conversations

        with _conn() as conn:
            deleted = clear_conversations(conn, conversation_id=conversation_id)
        if deleted == 0:
            raise HTTPException(
                status_code=404, detail=f"No saved conversation with id {conversation_id}."
            )
        return {"deleted": deleted}

    @app.post("/chat/stream")
    # No return annotation: `StreamingResponse` is imported in the body (below), so naming it
    # here would leave an unresolvable annotation. `ask_stream` above does the same.
    def chat_stream(body: dict):
        """One persisted multi-turn chat turn (the web chat): NDJSON token events, then a
        final `{"type":"answer","data":…, "conversation_id":…}` event. The API's single
        write path — it appends only to the user-owned chat-history tables (see module
        docstring); pass `"save": false` (or set CMRAG_SAVE_HISTORY=off) to disable, at
        the cost of server-side context carry."""
        import json

        from fastapi.responses import StreamingResponse

        q = str(body.get("q") or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Missing 'q'.")
        conversation_id = body.get("conversation_id")
        if conversation_id is not None and not isinstance(conversation_id, int):
            raise HTTPException(status_code=400, detail="'conversation_id' must be an integer.")
        try:
            events = chat_stream_events(
                query=q,
                conversation_id=conversation_id,
                top_k=int(body.get("top_k") or 5),
                profile=str(body.get("profile") or "balanced"),
                level=str(body.get("level") or "evidence"),
                modalities=body.get("modality"),
                use_llm=bool(body.get("use_llm", True)),
                save=bool(body.get("save", True)),
            )
        except ConversationNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        def _ndjson():
            for event in events:
                yield json.dumps(event) + "\n"

        return StreamingResponse(_ndjson(), media_type="application/x-ndjson")

    @app.get("/concepts")
    def concepts(top: int = 20) -> dict:
        with _conn() as conn:
            return {"concepts": list_concept_views(conn, top=top)}

    @app.get("/timeline")
    def timeline(limit: int = 50) -> dict:
        with _conn() as conn:
            return {"timeline": list_episode_timeline(conn, limit=limit)}

    @app.get("/memory-stats")
    def memory_stats_route() -> dict:
        with _conn() as conn:
            return memory_stats(conn)

    @app.get("/forgetting")
    def forgetting(level: str = "concept", top: int = 10, min_support: int = 1) -> dict:
        with _conn() as conn:
            items = compute_forgetting_risk(
                conn, now=_now(), halflife_days=get_usage_halflife_days(),
                levels=_levels(level), min_support=min_support, top=top,
            )
            return {"level": level, "forgetting": [forgetting_risk_to_dict(i) for i in items]}

    @app.get("/recall")
    def recall(level: str = "concept", top: int = 10, min_support: int = 1) -> dict:
        from crossmodalrag.config import get_extract_model
        from crossmodalrag.generate.provider import get_default_llm_provider

        with _conn() as conn:
            provider = get_default_llm_provider(get_extract_model())
            cards = generate_recall_cards(
                conn, provider, now=_now(), halflife_days=get_usage_halflife_days(),
                levels=_levels(level), top=top, min_support=min_support,
            )
            return {"level": level, "recall": [recall_card_to_dict(c) for c in cards]}

    @app.get("/drift")
    def drift(top: int = 10, min_support: int = 1) -> dict:
        with _conn() as conn:
            items = concept_drift_summaries(conn, top=top, min_support=min_support)
            return {"drift": [drift_summary_to_dict(conn, i) for i in items]}

    @app.get("/distill")
    def distill(top: int = 10) -> dict:
        with _conn() as conn:
            items = distilled_summaries(conn, top=top)
            return {
                "distilled": [distilled_summary_to_dict(i) for i in items],
                "overall_compression_ratio": {
                    "episode": distilled_compression_ratio(conn, level="episode"),
                    "concept": distilled_compression_ratio(conn, level="concept"),
                },
            }

    @app.get("/usage")
    def usage(top: int = 10) -> dict:
        from crossmodalrag.config import usage_tracking_enabled

        with _conn() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM usage_events").fetchone()["n"]
            by_type = conn.execute(
                "SELECT event_type, COUNT(*) AS n FROM usage_events GROUP BY event_type ORDER BY event_type"
            ).fetchall()
            summaries = usage_summaries(conn, now=_now(), halflife_days=get_usage_halflife_days())
        top_targets = sorted(summaries.values(), key=lambda s: s.strength, reverse=True)[:top]
        return {
            "tracking_enabled": usage_tracking_enabled(),
            "total_events": int(total),
            "by_type": {r["event_type"]: int(r["n"]) for r in by_type},
            "top_targets": [usage_summary_to_dict(s) for s in top_targets],
        }

    # Serve the built web UI at the root (vendored, no external calls). Mounted LAST so the JSON API
    # routes above win; absent when the UI hasn't been built (the API still works headless).
    if STATIC_DIR.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

    return app
