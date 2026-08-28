from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from crossmodalrag.config import (
    get_connector_paths,
    get_db_path,
    get_default_profile,
    get_default_top_k,
    get_extract_model,
    load_dotenv,
    save_history_enabled,
    usage_tracking_enabled,
)
from crossmodalrag.db import connect, init_db
from crossmodalrag.embed.provider import (
    DEFAULT_EMBED_MODEL,
    MissingEmbeddingBackend,
    get_default_provider,
    require_default_provider,
)
from crossmodalrag.embed.store import (
    count_embeddings,
    count_node_embeddings,
    embed_pending_chunks,
    embed_pending_nodes,
)
from crossmodalrag.evaluation import load_eval_queries_file, run_eval, upsert_eval_queries
from crossmodalrag.generate.answer import (
    format_answer_stream_header,
    format_generated_answer,
    format_generated_answer_footer,
    format_grounded_answer,
    generated_answer_to_dict,
    template_answer_to_dict,
)
from crossmodalrag.generate.provider import LLMUnavailable, get_default_llm_provider
from crossmodalrag.generate.synthesize import GeneratedAnswer, synthesize_answer
from crossmodalrag.generation_eval import run_generation_eval
from crossmodalrag.capabilities import MissingModalityBackend
from crossmodalrag.ingest.git import ingest_git
from crossmodalrag.ingest.image import ingest_images
from crossmodalrag.ingest.notes import ingest_notes
from crossmodalrag.ingest.pdf import ingest_pdf
from crossmodalrag.progress import make_progress
from crossmodalrag.service import retrieve_for_answer
from crossmodalrag.retrieve.rerank import MODALITY_SOURCE_TYPES, resolve_source_types
from crossmodalrag.memory.concepts import build_concepts
from crossmodalrag.memory.episodes import build_episodes
from crossmodalrag.memory.extract import extract_pending_sources
from crossmodalrag.memory.graph import build_graph
from crossmodalrag.retrieve.hybrid import DEFAULT_PROFILE, PROFILE_WEIGHTS
from crossmodalrag.sample_data import default_sample_db_path, seed_sample_data

def init_db_cmd() -> None:
    db_path = get_db_path()
    conn = connect(db_path)
    try:
        init_db(conn)
    finally:
        conn.close()
    print(f"Initialized database at {db_path}")


def ingest_notes_cmd(vault_paths: list[Path]) -> None:
    db_path = get_db_path()
    conn = connect(db_path)
    embedder = get_default_provider()
    try:
        init_db(conn)
        total_inserted = 0
        for vault_path in vault_paths:
            inserted = ingest_notes(
                conn, vault_path=vault_path, embedder=embedder,
                progress=make_progress(f"notes {vault_path.name}"),
            )
            total_inserted += inserted
            print(f"Ingested notes from {vault_path} into {db_path}. Inserted chunks: {inserted}")
    finally:
        conn.close()
    print(
        f"Completed note ingestion for {len(vault_paths)} vault(s) into {db_path}. "
        f"Total inserted chunks: {total_inserted}"
    )


def ingest_git_cmd(repo_paths: list[Path], max_commits: int = 300) -> None:
    db_path = get_db_path()
    conn = connect(db_path)
    embedder = get_default_provider()
    try:
        init_db(conn)
        total_inserted = 0
        for repo_path in repo_paths:
            inserted = ingest_git(
                conn, repo_path=repo_path, max_commits=max_commits, embedder=embedder,
                progress=make_progress(f"git {repo_path.name}"),
            )
            total_inserted += inserted
            print(
                f"Ingested git history from {repo_path} into {db_path}. Inserted chunks: {inserted}"
            )
    finally:
        conn.close()
    print(
        f"Completed git ingestion for {len(repo_paths)} repo(s) into {db_path}. "
        f"Total inserted chunks: {total_inserted}"
    )


def ingest_pdf_cmd(pdf_paths: list[Path]) -> None:
    db_path = get_db_path()
    conn = connect(db_path)
    embedder = get_default_provider()
    try:
        init_db(conn)
        total_inserted = 0
        for pdf_path in pdf_paths:
            inserted = ingest_pdf(
                conn, pdf_path=pdf_path, embedder=embedder,
                progress=make_progress(f"pdf {pdf_path.name}"),
            )
            total_inserted += inserted
            print(f"Ingested PDF(s) from {pdf_path} into {db_path}. Inserted chunks: {inserted}")
    except MissingModalityBackend as exc:
        raise SystemExit(str(exc))
    finally:
        conn.close()
    print(
        f"Completed PDF ingestion for {len(pdf_paths)} path(s) into {db_path}. "
        f"Total inserted chunks: {total_inserted}"
    )


def ingest_images_cmd(image_paths: list[Path]) -> None:
    db_path = get_db_path()
    conn = connect(db_path)
    embedder = get_default_provider()
    try:
        init_db(conn)
        total_inserted = 0
        for image_path in image_paths:
            inserted = ingest_images(
                conn, image_path=image_path, embedder=embedder,
                progress=make_progress(f"images {image_path.name}"),
            )
            total_inserted += inserted
            print(f"Ingested image(s) from {image_path} into {db_path}. Inserted chunks: {inserted}")
    except MissingModalityBackend as exc:
        raise SystemExit(str(exc))
    finally:
        conn.close()
    print(
        f"Completed image ingestion for {len(image_paths)} path(s) into {db_path}. "
        f"Total inserted chunks: {total_inserted}"
    )


def ask_cmd(
    query: str,
    top_k: int = 5,
    profile: str = DEFAULT_PROFILE,
    explain: bool = False,
    use_llm: bool = True,
    as_json: bool = False,
    debug: bool = False,
    level: str = "evidence",
    modalities: list[str] | None = None,
    accept: bool = False,
    track: bool | None = None,
    stream: bool = True,
) -> None:
    # Usage tracking is opt-in: off unless enabled by env / --track / --accept, and never by --no-track.
    track_enabled = False if track is False else (bool(track) or accept or usage_tracking_enabled())
    _run_ask_turn(
        query,
        top_k=top_k,
        profile=profile,
        explain=explain,
        use_llm=use_llm,
        as_json=as_json,
        debug=debug,
        level=level,
        modalities=modalities,
        accept=accept,
        track_enabled=track_enabled,
        stream=stream,
    )


def _run_ask_turn(
    query: str,
    *,
    top_k: int = 5,
    profile: str = DEFAULT_PROFILE,
    explain: bool = False,
    use_llm: bool = True,
    as_json: bool = False,
    debug: bool = False,
    level: str = "evidence",
    modalities: list[str] | None = None,
    accept: bool = False,
    track_enabled: bool = False,
    stream: bool = True,
    history: str | None = None,
) -> "GeneratedAnswer | None":
    """One complete ask turn: retrieve → synthesize → render → track.

    Shared by one-shot `mem ask` (history=None, byte-identical behavior) and
    the interactive chat loop (history = the session's rendered prior turns).
    Returns the ``GeneratedAnswer`` on the LLM path so a chat session can carry
    the turn as context; ``None`` on the template/no-LLM paths (a template
    render is not a synthesized answer and is never carried).
    """
    db_path = get_db_path()
    ask_start = time.monotonic()
    conn = connect(db_path)
    try:
        hits, matched_nodes = retrieve_for_answer(
            conn, query=query, top_k=top_k, profile=profile, level=level, modalities=modalities
        )
    finally:
        conn.close()

    matched_payload = [
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

    provider = get_default_llm_provider() if use_llm else None
    if provider is not None:
        # Stream tokens live only for human-readable TTY output: --json stays a
        # buffered contract, and piped/redirected output stays clean.
        do_stream = stream and not as_json and sys.stdout.isatty()
        streamed_any = False

        def _print_token(fragment: str) -> None:
            nonlocal streamed_any
            if not streamed_any:
                # Lazy header: the weak-retrieval gate abstains before the LLM,
                # so nothing is printed until the first real token arrives.
                _print_matched_nodes(matched_payload, level)
                print(format_answer_stream_header(query, provider.name))
                streamed_any = True
            sys.stdout.write(fragment)
            sys.stdout.flush()

        try:
            gen = synthesize_answer(
                query, hits, provider, on_token=_print_token if do_stream else None, history=history
            )
        except LLMUnavailable as exc:
            if streamed_any:
                sys.stdout.write("\n")
                sys.stdout.flush()
            print(f"[notice] LLM unavailable, falling back to evidence template: {exc}", file=sys.stderr)
            provider = None
        else:
            if as_json:
                data = generated_answer_to_dict(gen, total_seconds=time.monotonic() - ask_start)
                if matched_payload:
                    data["matched_nodes"] = matched_payload
                print(json.dumps(data, indent=2))
            elif streamed_any:
                # Answer text already on screen; close the line and render the
                # trailer (status/citations/evidence/debug) from the full output.
                sys.stdout.write("\n")
                sys.stdout.flush()
                footer = format_generated_answer_footer(gen, explain=explain, debug=debug)
                if footer:
                    print(footer)
            else:
                _print_matched_nodes(matched_payload, level)
                print(format_generated_answer(gen, explain=explain, debug=debug))
            if track_enabled and not gen.abstained:
                cited = [gen.id_map[eid].chunk_id for eid in gen.cited_evidence_ids if eid in gen.id_map]
                _track_ask(
                    db_path,
                    hits=hits,
                    matched_nodes=matched_nodes,
                    accepted_chunk_ids=(cited or [h.chunk_id for h in hits]) if accept else [],
                )
            return gen

    # No LLM (disabled or unavailable): deterministic evidence template.
    if as_json:
        data = template_answer_to_dict(query, hits, total_seconds=time.monotonic() - ask_start)
        if matched_payload:
            data["matched_nodes"] = matched_payload
        print(json.dumps(data, indent=2))
    else:
        _print_matched_nodes(matched_payload, level)
        print(format_grounded_answer(query, hits, explain=explain or debug))

    # No-LLM / fallback path: the template shows `hits` directly; no citations to scope `--accept`.
    if track_enabled and hits:
        _track_ask(
            db_path,
            hits=hits,
            matched_nodes=matched_nodes,
            accepted_chunk_ids=[h.chunk_id for h in hits] if accept else [],
        )
    return None


def _track_ask(db_path, *, hits, matched_nodes, accepted_chunk_ids) -> None:
    """Best-effort usage tracking for an `ask` (never raises into the command path)."""
    from datetime import datetime, timezone

    from crossmodalrag.usage.tracking import record_ask_interaction

    try:
        conn = connect(db_path)
        try:
            init_db(conn)
            record_ask_interaction(
                conn,
                now=datetime.now(timezone.utc),
                retrieved_chunk_ids=[h.chunk_id for h in hits],
                accepted_chunk_ids=accepted_chunk_ids,
                opened_node_ids=[n.node_id for n in matched_nodes],
            )
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - tracking must never break a query
        print(f"[notice] usage tracking skipped: {exc}", file=sys.stderr)


def _print_matched_nodes(matched_payload: list[dict], level: str) -> None:
    if not matched_payload:
        return
    print(f"Matched {level}s:")
    for node in matched_payload:
        title = node["title"] or "untitled"
        print(f"  #{node['node_id']} (score={node['score']:.3f}, centrality={node['centrality']:.3f}): {title}")
    print()


def chat_cmd(
    *,
    top_k: int = 5,
    profile: str = DEFAULT_PROFILE,
    explain: bool = False,
    use_llm: bool = True,
    debug: bool = False,
    level: str = "evidence",
    modalities: list[str] | None = None,
    track: bool | None = None,
    stream: bool = True,
    save: bool = True,
    resume: int | None = None,
) -> None:
    """Interactive multi-turn ask session (`mem chat`, or `mem ask` with no query).

    A thin REPL over :func:`_run_ask_turn`: every turn runs retrieval
    independently and cites the CURRENT turn's evidence; prior answered turns
    are carried only as conversation context (`chat.render_history`, citations
    stripped). Abstained and template (`--no-llm` / LLM-unavailable) turns are
    never carried. Exit with /exit, /quit, or Ctrl-D; /clear resets the context
    without leaving; /new also starts a new saved conversation. Piped stdin
    works as a batch mode (one query per line, no prompt/banner).

    With ``save`` (default; `--no-save` / CMRAG_SAVE_HISTORY=off disables) each
    LLM turn — including abstained ones — is persisted locally to the
    conversations/messages history (browse with `mem history`, wipe with
    `mem history --clear`). Template/no-LLM turns are not persisted. New
    conversations are auto-titled by the local LLM from their first exchange
    (deterministic first-query fallback when the LLM is unavailable).

    ``resume`` continues a stored conversation: its answered turns are loaded
    as carried context (context cap + abstained-skip rules apply) and new turns
    append to the same conversation (0 = the most recent conversation).
    """
    from crossmodalrag.chat import ChatSession, render_history
    from crossmodalrag.conversations.naming import generate_conversation_title
    from crossmodalrag.conversations.recorder import SessionRecorder
    from crossmodalrag.conversations.resume import next_turn_index, turns_from_messages
    from crossmodalrag.conversations.store import get_conversation, list_conversations, list_messages

    try:  # line editing/arrow keys when available; purely optional
        import readline  # noqa: F401
    except ImportError:  # pragma: no cover - platform-dependent
        pass

    # Usage tracking mirrors ask_cmd (no --accept in a session).
    track_enabled = False if track is False else (bool(track) or usage_tracking_enabled())
    session = ChatSession()

    def _title_fn(query: str, answer_text: str) -> str | None:
        provider = get_default_llm_provider() if use_llm else None
        if provider is None:
            return None
        return generate_conversation_title(provider, query=query, answer_text=answer_text)

    recorder = SessionRecorder(get_db_path(), enabled=save, title_fn=_title_fn)
    interactive = sys.stdin.isatty()

    if resume is not None:
        conn = connect(get_db_path())
        try:
            init_db(conn)
            if resume == 0:  # bare --resume: the most recent conversation
                recent = list_conversations(conn, top=1)
                if not recent:
                    raise CLIError("No saved conversations to resume. Start one with `mem chat`.")
                conversation = recent[0]
            else:
                conversation = get_conversation(conn, resume)
                if conversation is None:
                    raise CLIError(f"No saved conversation with id {resume}.")
            messages = list_messages(conn, conversation.id)
        finally:
            conn.close()
        for turn in turns_from_messages(messages):
            session.add_turn(turn.query, turn.answer_text)  # context cap applies
        recorder.attach(conversation.id, next_turn_index=next_turn_index(messages))
        print(
            f'Resuming conversation #{conversation.id}: "{conversation.title or "untitled"}" '
            f"({len(messages)} messages; {len(session.turns)} turns carried as context)"
        )
        for turn in session.turns:
            print(f"  you> {turn.query}")

    if interactive:
        print("Interactive session — /exit (or Ctrl-D) to quit, /clear to reset context.")
        if save:
            print(
                "History is saved locally (/new starts a fresh conversation; "
                "`mem history --clear` wipes; --no-save disables)."
            )
    try:
        while True:
            try:
                line = input("ask> " if interactive else "")
            except EOFError:
                break
            text = line.strip()
            if not text:
                continue
            if text in {"/exit", "/quit"}:
                break
            if text in {"/clear", "/new"}:
                session.clear()
                if text == "/new":
                    recorder.new_conversation()
                    print("[new conversation]")
                else:
                    print("[context cleared]")
                continue
            gen = _run_ask_turn(
                text,
                top_k=top_k,
                profile=profile,
                explain=explain,
                use_llm=use_llm,
                level=level,
                modalities=modalities,
                debug=debug,
                track_enabled=track_enabled,
                stream=stream,
                history=render_history(session.turns) or None,
            )
            if gen is not None:
                session.add_turn(text, gen.answer_text, abstained=gen.abstained)
                recorder.record_turn(text, gen)
            print()
    except KeyboardInterrupt:
        # Clean session exit whether at the prompt or mid-stream; close any
        # partially streamed line first.
        sys.stdout.write("\n")
        sys.stdout.flush()
    if interactive:
        print("bye")


def history_cmd(
    show_id: int | None = None,
    top: int = 10,
    clear: bool = False,
    clear_id: int | None = None,
    rename_id: int | None = None,
    new_title: str | None = None,
    as_json: bool = False,
) -> None:
    """List/show/clear/rename locally saved chat conversations (`mem history`)."""
    from crossmodalrag.conversations.contract import conversation_to_dict
    from crossmodalrag.conversations.store import (
        clear_conversations,
        count_messages,
        get_conversation,
        list_conversations,
        list_messages,
        rename_conversation,
    )

    if sum(1 for flag in (clear, show_id is not None, rename_id is not None) if flag) > 1:
        raise CLIError("--clear, --show, and --rename cannot be combined.")

    db_path = get_db_path()
    conn = connect(db_path)
    try:
        init_db(conn)

        if rename_id is not None:
            title = (new_title or "").strip()
            if not title:
                raise CLIError('--rename needs a name: mem history --rename <id> --title "…"')
            if not rename_conversation(conn, rename_id, title=title[:200]):
                raise CLIError(f"No saved conversation with id {rename_id}.")
            if as_json:
                print(json.dumps({"renamed": rename_id, "title": title[:200]}, indent=2))
            else:
                print(f'Renamed conversation #{rename_id} to "{title[:200]}".')
            return

        if clear:
            deleted = clear_conversations(conn, conversation_id=clear_id)
            if as_json:
                print(json.dumps({"cleared": deleted}, indent=2))
            else:
                scope = f"conversation #{clear_id}" if clear_id is not None else "all conversations"
                print(f"Cleared {deleted} conversation(s) ({scope}) from {db_path}.")
            return

        if show_id is not None:
            conversation = get_conversation(conn, show_id)
            if conversation is None:
                raise CLIError(f"No saved conversation with id {show_id}.")
            if as_json:
                print(json.dumps(conversation_to_dict(conn, conversation), indent=2))
                return
            print(f'Conversation #{conversation.id}: "{conversation.title or "untitled"}"')
            print(f"{conversation.started_at} → {conversation.updated_at}")
            for message in list_messages(conn, show_id):
                print()
                if message.role == "user":
                    print(f"you> {message.text}")
                    continue
                header = message.model or "assistant"
                marks = []
                if message.abstention_reason:
                    marks.append(f"abstained: {message.abstention_reason}")
                if message.truncated:
                    marks.append("truncated")
                suffix = f" [{'; '.join(marks)}]" if marks else ""
                print(f"{header}>{suffix} {message.text}")
                if message.evidence_json:
                    cited = [
                        ev for ev in json.loads(message.evidence_json) if ev.get("cited")
                    ]
                    for ev in cited:
                        print(f"      [{ev['evidence_id']}] {ev.get('locator') or ev['source_uri']}")
            return

        conversations = list_conversations(conn, top=top)
        if as_json:
            print(
                json.dumps(
                    {
                        "save_enabled": save_history_enabled(),
                        "total": len(list_conversations(conn)),
                        "conversations": [
                            conversation_to_dict(conn, c, include_messages=False)
                            for c in conversations
                        ],
                    },
                    indent=2,
                )
            )
            return
        if not conversations:
            print("No saved conversations. Start one with `mem chat`.")
            return
        print(f"Saved conversations (newest first, top {top}):")
        for c in conversations:
            n = count_messages(conn, c.id)
            print(f"  #{c.id}  {c.updated_at}  ({n} messages)  {c.title or 'untitled'}")
        print('Show one with `mem history --show <id>`; wipe with `mem history --clear`.')
    finally:
        conn.close()


def eval_cmd(
    top_k: int = 5,
    query_prefix: str | None = None,
    load_queries_path: Path | None = None,
    profile: str = DEFAULT_PROFILE,
    level: str = "evidence",
    modalities: list[str] | None = None,
    as_json: bool = False,
) -> None:
    from crossmodalrag.evaluation import eval_summary_to_dict

    db_path = get_db_path()
    restrict_source_types = resolve_source_types(modalities)
    conn = connect(db_path)
    try:
        init_db(conn)
        loaded = 0
        if load_queries_path is not None:
            from crossmodalrag.evaluation import validate_eval_queries

            queries = load_eval_queries_file(load_queries_path)
            known_uris = {
                str(row["source_uri"])
                for row in conn.execute("SELECT DISTINCT source_uri FROM sources").fetchall()
            }
            for warning in validate_eval_queries(queries, known_source_uris=known_uris):
                print(
                    f"warning: eval row #{warning.row} ({warning.query_text!r}): "
                    f"{warning.issue} uri {warning.uri!r}",
                    file=sys.stderr,
                )
            loaded = upsert_eval_queries(conn, queries)
        summary = run_eval(
            conn,
            top_k=top_k,
            query_prefix=query_prefix,
            profile=profile,
            level=level,
            restrict_source_types=restrict_source_types,
        )
    finally:
        conn.close()

    if as_json:
        payload = eval_summary_to_dict(summary)
        payload.update({"level": level, "profile": profile})
        print(json.dumps(payload, indent=2))
        return

    print(f"Evaluation DB: {db_path}")
    print(f"Retrieval level: {level} | profile: {profile}")
    if load_queries_path is not None:
        print(f"Eval queries loaded/upserted: {loaded} from {load_queries_path}")
    if query_prefix:
        print(f"Query prefix filter: {query_prefix}")
    if summary.query_count == 0:
        print("No evaluation queries found. Load queries into 'queries_eval' and run again.")
        return
    print(f"Queries evaluated: {summary.query_count}")
    print(f"Recall@{summary.top_k}: {summary.recall_at_k:.3f}")
    print(f"MRR@{summary.top_k}: {summary.mrr_at_k:.3f}")
    print(f"Citation hit-rate (top-1): {summary.citation_hit_rate:.3f}")
    misses = [r.query_text for r in summary.results if r.first_correct_rank is None]
    if misses:
        print(f"Queries with no correct hit in top-{summary.top_k} ({len(misses)}):")
        for query_text in misses:
            print(f"  - {query_text}")


def reindex_embeddings_cmd(batch_size: int = 64, model: str | None = None) -> None:
    db_path = get_db_path()
    try:
        provider = require_default_provider(model)
    except MissingEmbeddingBackend as exc:
        raise SystemExit(str(exc))
    conn = connect(db_path)
    try:
        init_db(conn)
        embedded = embed_pending_chunks(
            conn, provider, batch_size=batch_size, progress=make_progress("embed chunks")
        )
        total = count_embeddings(conn, model=provider.name)
        nodes_embedded = 0
        for node_type, lvl in (("event", 1), ("episode", 2), ("concept", 3)):
            nodes_embedded += embed_pending_nodes(
                conn, provider, level=lvl, node_type=node_type, batch_size=batch_size,
                progress=make_progress(f"embed {node_type}s"),
            )
        node_total = count_node_embeddings(conn, model=provider.name)
    finally:
        conn.close()
    print(f"Reindex DB: {db_path}")
    print(f"Model: {provider.name} (dim={provider.dim})")
    print(f"Chunks embedded this run: {embedded}")
    print(f"Total chunks with current-model embeddings: {total}")
    print(f"Memory nodes embedded this run: {nodes_embedded}")
    print(f"Total nodes with current-model embeddings: {node_total}")


def eval_generation_cmd(
    top_k: int = 5,
    query_prefix: str | None = None,
    profile: str = DEFAULT_PROFILE,
    model: str | None = None,
    level: str = "evidence",
) -> None:
    db_path = get_db_path()
    provider = get_default_llm_provider()
    if provider is None:
        raise SystemExit("No LLM provider configured (set CMRAG_LLM_PROVIDER).")
    if model:
        provider.name = model
    conn = connect(db_path)
    try:
        init_db(conn)
        try:
            summary = run_generation_eval(
                conn, provider, top_k=top_k, query_prefix=query_prefix, profile=profile, level=level
            )
        except LLMUnavailable as exc:
            raise SystemExit(str(exc))
    finally:
        conn.close()

    print(f"Evaluation DB: {db_path}")
    print(f"Model: {summary.model} | level: {summary.level} | profile: {summary.profile}")
    if query_prefix:
        print(f"Query prefix filter: {query_prefix}")
    if summary.query_count == 0:
        print("No evaluation queries found. Load queries into 'queries_eval' and run again.")
        return
    print(f"Queries evaluated: {summary.query_count}")
    print(f"Citation validity: {summary.citation_validity:.3f}")
    print(f"Source-grounding hit: {summary.source_grounding_hit:.3f}")
    print(f"Source coverage: {summary.source_coverage:.3f}")
    print(f"Abstention correct: {summary.abstention_correct:.3f}")
    bad = [r.query_text for r in summary.results if not r.abstention_correct or not r.citation_valid]
    if bad:
        print(f"Queries with citation/abstention issues ({len(bad)}):")
        for query_text in bad:
            print(f"  - {query_text}")


def build_memory_cmd(level: str = "all", limit: int | None = None, model: str | None = None) -> None:
    build_events = level in ("event", "all")
    build_eps = level in ("episode", "all")
    build_concept = level in ("concept", "all")
    build_graph_layer = level in ("graph", "all")
    build_drift_layer = level in ("drift", "all")
    build_distill_layer = level in ("distill", "all")
    db_path = get_db_path()
    conn = connect(db_path)
    try:
        init_db(conn)
        print(f"Memory DB: {db_path}")

        if build_events:
            # Deterministic self-heal before extraction: a re-chunk re-issues chunk ids,
            # and text-identical sources are fingerprint-skipped by extraction, which
            # would leave their events anchored to dead ids.
            from crossmodalrag.memory.integrity import repair_evidence_edges

            repair = repair_evidence_edges(conn)
            if repair.events_repaired or repair.orphaned_event_ids:
                print("L1 evidence re-anchor")
                print(
                    f"  Events re-anchored: {repair.events_repaired} "
                    f"(dead edges removed: {repair.edges_removed}, edges added: {repair.edges_added})"
                )
                if repair.orphaned_event_ids:
                    shown = ", ".join(str(i) for i in repair.orphaned_event_ids[:10])
                    more = len(repair.orphaned_event_ids) - 10
                    suffix = f" (+{more} more)" if more > 0 else ""
                    print(f"  Events with no recoverable source (left in place): {shown}{suffix}")

            provider = get_default_llm_provider(model or get_extract_model())
            if provider is None:
                raise SystemExit("No LLM provider configured (set CMRAG_LLM_PROVIDER).")
            try:
                result = extract_pending_sources(
                    conn, provider, limit=limit, progress=make_progress("L1 events")
                )
            except LLMUnavailable as exc:
                raise SystemExit(str(exc))
            print(f"L1 events | model: {provider.name}")
            print(f"  Sources processed: {result.sources_processed}")
            print(f"  Sources skipped (up to date): {result.sources_skipped}")
            print(f"  Events created: {result.events_created}")
            if result.parse_failures:
                print(f"  Unparseable sources (will retry next run): {result.parse_failures}")
                for source_id, source_uri in result.unparseable_sources[:10]:
                    print(f"    #{source_id} {source_uri}")
                if len(result.unparseable_sources) > 10:
                    print(f"    … (+{len(result.unparseable_sources) - 10} more)")

        if build_eps:
            episodes = build_episodes(conn)
            print("L2 episodes")
            print(f"  Episodes created: {episodes.episodes_created}")
            print(f"  Episodes kept (up to date): {episodes.episodes_kept}")
            print(f"  Episodes deleted (stale): {episodes.episodes_deleted}")
            print(f"  Events grouped: {episodes.events_grouped}")

        if build_concept:
            embed_provider = get_default_provider()
            if embed_provider is None:
                raise SystemExit(
                    "L3 concepts require the embeddings extra. Run: pip install -e \".[embeddings]\""
                )
            llm_provider = get_default_llm_provider(model or get_extract_model())
            concepts = build_concepts(conn, embed_provider, llm_provider)
            print(f"L3 concepts | embed: {embed_provider.name}")
            print(f"  Concepts created: {concepts.concepts_created}")
            print(f"  Concepts kept (up to date): {concepts.concepts_kept}")
            print(f"  Concepts deleted (stale): {concepts.concepts_deleted}")
            print(f"  Events clustered: {concepts.events_clustered} (unclustered: {concepts.events_unclustered})")
            if concepts.concepts_created:
                print(f"  Named by LLM: {concepts.named_by_llm} | by fallback: {concepts.named_by_fallback}")

        if build_graph_layer:
            graph = build_graph(conn)
            print("Graph")
            print(f"  Concept co-occurrence edges: {graph.relates_edges_created} "
                  f"(replaced {graph.relates_edges_deleted})")
            print(f"  Nodes scored (centrality): {graph.nodes_scored}")

        if build_drift_layer:
            embed_provider = get_default_provider()
            if embed_provider is None:
                raise SystemExit(
                    "Concept drift requires the embeddings extra. Run: pip install -e \".[embeddings]\""
                )
            from crossmodalrag.memory.drift import build_drift

            drift = build_drift(conn, embed_provider)
            print(f"Concept drift | embed: {embed_provider.name}")
            print(f"  Concepts analyzed: {drift.concepts_analyzed}")
            print(f"  Snapshots created: {drift.snapshots_created}")
            print(f"  Snapshots kept (up to date): {drift.snapshots_kept}")
            print(f"  Snapshots deleted (stale): {drift.snapshots_deleted}")

        if build_distill_layer:
            embed_provider = get_default_provider()
            if embed_provider is None:
                raise SystemExit(
                    "Distillation requires the embeddings extra. Run: pip install -e \".[embeddings]\""
                )
            from crossmodalrag.memory.distill import build_distilled

            llm_provider = get_default_llm_provider(model or get_extract_model())
            distilled = build_distilled(conn, embed_provider, llm_provider)
            print(f"Distillation | embed: {embed_provider.name}")
            print(f"  Nodes distilled: {distilled.nodes_distilled}")
            print(f"  Nodes kept (up to date): {distilled.nodes_kept}")
            print(f"  Nodes deleted (stale): {distilled.nodes_deleted}")
            if distilled.nodes_distilled:
                print(f"  Summaries by LLM: {distilled.named_by_llm} | by fallback: {distilled.named_by_fallback}")
    finally:
        conn.close()
    print("Run `mem memory-stats` to inspect node/edge counts and integrity.")


def memory_stats_cmd(as_json: bool = False) -> None:
    from crossmodalrag.memory.integrity import memory_stats

    db_path = get_db_path()
    conn = connect(db_path)
    try:
        init_db(conn)
        stats = memory_stats(conn)
    finally:
        conn.close()

    if as_json:
        print(json.dumps(stats, indent=2))
        return

    by_level = stats["nodes_by_level"]
    print(f"Memory DB: {db_path}")
    print(f"Memory nodes (L1-L3): {stats['total_nodes']}")
    for level in (1, 2, 3):
        print(f"  L{level}: {by_level.get(str(level), 0)}")
    if stats["nodes_by_type"]:
        print("By type: " + ", ".join(f"{name}={count}" for name, count in stats["nodes_by_type"].items()))
    print(f"Memory edges: {stats['edges']}")
    print(f"Concept co-occurrence edges (relates_to): {stats['relates_edges']}")
    print(f"Node embeddings: {stats['node_embeddings']}")
    print(f"Distilled nodes: {stats['distilled_nodes']}")
    print(f"Drift snapshots: {stats['drift_snapshots']}")
    if stats["top_central"]:
        print("Top central nodes:")
        for row in stats["top_central"]:
            print(f"  L{row['level']} #{row['node_id']} ({row['centrality']:.3f}): {row['title'] or 'untitled'}")
    integ = stats["integrity"]
    print("Integrity:")
    print(f"  unsupported nodes (no L0 evidence): {integ['unsupported_count']}")
    print(f"  dangling edges (missing endpoint): {integ['dangling_count']}")
    if integ["unsupported_ids"]:
        print(f"  unsupported node ids: {integ['unsupported_ids']}")
    if integ["dangling_ids"]:
        print(f"  dangling edge ids: {integ['dangling_ids']}")


def usage_cmd(clear: bool = False, top: int = 10, as_json: bool = False) -> None:
    from datetime import datetime, timezone

    from crossmodalrag.config import get_usage_halflife_days, usage_tracking_enabled
    from crossmodalrag.usage.store import clear_usage_events, usage_summaries
    from crossmodalrag.usage.strength import usage_summary_to_dict

    db_path = get_db_path()
    conn = connect(db_path)
    try:
        init_db(conn)
        if clear:
            deleted = clear_usage_events(conn)
            if as_json:
                print(json.dumps({"cleared": deleted}, indent=2))
            else:
                print(f"Cleared {deleted} usage event(s) from {db_path}.")
            return

        total = conn.execute("SELECT COUNT(*) AS n FROM usage_events").fetchone()["n"]
        by_type = conn.execute(
            "SELECT event_type, COUNT(*) AS n FROM usage_events GROUP BY event_type ORDER BY event_type"
        ).fetchall()
        summaries = usage_summaries(
            conn, now=datetime.now(timezone.utc), halflife_days=get_usage_halflife_days()
        )
    finally:
        conn.close()

    top_targets = sorted(summaries.values(), key=lambda s: s.strength, reverse=True)[:top]
    if as_json:
        print(json.dumps(
            {
                "tracking_enabled": usage_tracking_enabled(),
                "total_events": int(total),
                "by_type": {r["event_type"]: int(r["n"]) for r in by_type},
                "top_targets": [usage_summary_to_dict(s) for s in top_targets],
            },
            indent=2,
        ))
        return

    print(f"Usage DB: {db_path}")
    print(f"Tracking enabled (env): {usage_tracking_enabled()}")
    print(f"Total usage events: {total}")
    if by_type:
        print("By type: " + ", ".join(f"{r['event_type']}={r['n']}" for r in by_type))
    if top_targets:
        print(f"Top {len(top_targets)} targets by rehearsal strength:")
        for s in top_targets:
            print(f"  {s.target_kind} #{s.target_id}: strength={s.strength:.3f} "
                  f"events={s.count} last={s.last_event_at}")


def forgetting_cmd(level: str = "concept", top: int = 10, min_support: int = 1, as_json: bool = False) -> None:
    from datetime import datetime, timezone

    from crossmodalrag.config import get_usage_halflife_days
    from crossmodalrag.memory.forgetting import (
        LEVEL_NAMES,
        compute_forgetting_risk,
        forgetting_risk_to_dict,
    )

    db_path = get_db_path()
    conn = connect(db_path)
    try:
        init_db(conn)
        items = compute_forgetting_risk(
            conn,
            now=datetime.now(timezone.utc),
            halflife_days=get_usage_halflife_days(),
            levels=LEVEL_NAMES[level],
            min_support=min_support,
            top=top,
        )
    finally:
        conn.close()

    if as_json:
        print(json.dumps({"level": level, "forgetting": [forgetting_risk_to_dict(i) for i in items]}, indent=2))
        return

    print(f"Forgetting risk (level={level}) — DB: {db_path}")
    if not items:
        print(
            f"No {level} memory nodes with grounding found. Run `mem build-memory` first, "
            "or try `--level all`."
        )
        return
    print("What you're most likely forgetting (important but not recently revisited):")
    for item in items:
        title = item.title or "untitled"
        print(
            f"  [risk={item.risk:.3f}] L{item.level} {item.node_type}: {title}\n"
            f"      importance={item.importance:.3f} staleness={item.staleness:.3f} "
            f"confidence={item.confidence:.3f} (support={item.support}, last_touch={item.last_touch})"
        )
        if item.evidence_source_uris:
            print(f"      evidence: {', '.join(item.evidence_source_uris)}")


def drift_cmd(top: int = 10, min_support: int = 1, as_json: bool = False) -> None:
    from crossmodalrag.memory.drift import concept_drift_summaries, drift_summary_to_dict

    db_path = get_db_path()
    conn = connect(db_path)
    try:
        init_db(conn)
        items = concept_drift_summaries(conn, top=top, min_support=min_support)
        if as_json:
            payload = {"drift": [drift_summary_to_dict(conn, item) for item in items]}
    finally:
        conn.close()

    if as_json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Concept drift — DB: {db_path}")
    if not items:
        print(
            "No drift snapshots found. Run `mem build-memory` (or `mem build-memory --level drift`) "
            "first; concept drift needs the embeddings extra."
        )
        return
    print("How your concepts have moved across time windows (highest drift first):")
    for item in items:
        title = item.title or "untitled"
        tags = " [relearning]" if item.relearning else ""
        print(
            f"  [drift={item.overall_drift:.3f}] concept #{item.concept_id}: {title}{tags}\n"
            f"      windows={item.window_count} support={item.support} "
            f"confidence={item.confidence:.3f} span={item.span_start[:10]}..{item.span_end[:10]}"
        )
        if item.evidence_source_uri:
            print(f"      evidence: {item.evidence_source_uri}")


def distill_cmd(top: int = 10, as_json: bool = False) -> None:
    from crossmodalrag.evaluation import distilled_compression_ratio
    from crossmodalrag.memory.distill import distilled_summaries, distilled_summary_to_dict

    db_path = get_db_path()
    conn = connect(db_path)
    try:
        init_db(conn)
        items = distilled_summaries(conn, top=top)
        if as_json:
            payload = {
                "distilled": [distilled_summary_to_dict(item) for item in items],
                "overall_compression_ratio": {
                    "episode": distilled_compression_ratio(conn, level="episode"),
                    "concept": distilled_compression_ratio(conn, level="concept"),
                },
            }
    finally:
        conn.close()

    if as_json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Distilled nodes — DB: {db_path}")
    if not items:
        print(
            "No distilled nodes found. Run `mem build-memory --level distill` first "
            "(needs the embeddings extra)."
        )
        return
    print("Retrieval-preserving stand-ins (most compressed first):")
    for item in items:
        title = item.title or "untitled"
        conf = f" confidence={item.confidence:.3f}" if item.confidence is not None else ""
        print(
            f"  L{item.level} #{item.node_id}: {title}\n"
            f"      core/full evidence={item.core_count}/{item.full_count} "
            f"ratio={item.compression_ratio:.3f}{conf}"
        )
        if item.evidence_source_uri:
            print(f"      evidence: {item.evidence_source_uri}")


def recall_cmd(
    level: str = "concept",
    top: int = 10,
    min_support: int = 1,
    regenerate: bool = False,
    as_json: bool = False,
) -> None:
    from datetime import datetime, timezone

    from crossmodalrag.config import get_usage_halflife_days
    from crossmodalrag.memory.forgetting import LEVEL_NAMES
    from crossmodalrag.memory.recall import generate_recall_cards, recall_card_to_dict

    db_path = get_db_path()
    provider = get_default_llm_provider(get_extract_model())
    conn = connect(db_path)
    try:
        init_db(conn)
        cards = generate_recall_cards(
            conn,
            provider,
            now=datetime.now(timezone.utc),
            halflife_days=get_usage_halflife_days(),
            levels=LEVEL_NAMES[level],
            top=top,
            min_support=min_support,
            regenerate=regenerate,
        )
    finally:
        conn.close()

    if as_json:
        print(json.dumps({"level": level, "recall": [recall_card_to_dict(c) for c in cards]}, indent=2))
        return

    print(f"Active-recall cards (level={level}) — DB: {db_path}")
    if not cards:
        print(
            f"No {level} memory nodes with grounding found. Run `mem build-memory` first, "
            "or try `--level all`."
        )
        return
    print("Quiz yourself on what you're most likely forgetting:")
    for card in cards:
        title = card.title or "untitled"
        print(f"  [risk={card.risk:.3f} | {card.generated_by}] L{card.level} {card.node_type}: {title}")
        print(f"      Q: {card.question}")
        if card.answer:
            print(f"      A: {card.answer}")
        if card.evidence_source_uris:
            print(f"      evidence: {', '.join(card.evidence_source_uris)}")


def concepts_cmd(top: int = 20, as_json: bool = False) -> None:
    from crossmodalrag.memory.concepts import list_concept_views

    db_path = get_db_path()
    conn = connect(db_path)
    try:
        init_db(conn)
        items = list_concept_views(conn, top=top)
    finally:
        conn.close()

    if as_json:
        print(json.dumps({"concepts": items}, indent=2))
        return
    if not items:
        print("No concepts yet. Run `mem build-memory` (needs the embeddings extra).")
        return
    print(f"Concepts (top {len(items)} by centrality):")
    for item in items:
        print(
            f"  #{item['node_id']} (centrality={item['centrality']:.3f}, {item['members']} events): "
            f"{item['title'] or 'untitled'}"
        )


def timeline_cmd(limit: int = 50, as_json: bool = False) -> None:
    from crossmodalrag.memory.episodes import list_episode_timeline

    db_path = get_db_path()
    conn = connect(db_path)
    try:
        init_db(conn)
        items = list_episode_timeline(conn, limit=limit)
    finally:
        conn.close()

    if as_json:
        print(json.dumps({"timeline": items}, indent=2))
        return
    if not items:
        print("No episodes yet. Run `mem build-memory`.")
        return
    print(f"Timeline ({len(items)} episodes, oldest first):")
    for item in items:
        start = (item["time_start"] or "?")[:10]
        end = (item["time_end"] or "?")[:10]
        span = start if start == end else f"{start}..{end}"
        print(f"  #{item['node_id']} [{span}] ({item['members']} events): {item['title'] or 'untitled'}")


def seed_sample_cmd(
    workspace_dir: Path,
    force: bool = False,
    db_path: Path | None = None,
) -> None:
    db_path = (db_path or default_sample_db_path()).expanduser().resolve()
    conn = connect(db_path)
    try:
        init_db(conn)
        result = seed_sample_data(conn, workspace_dir=workspace_dir, force=force)
    finally:
        conn.close()
    print(f"Seeded sample data into sample DB: {db_path}")
    print("Main DB was not modified.")
    print(f"Workspace: {result.workspace_dir}")
    print(f"Sample vault: {result.vault_dir}")
    print(f"Sample git repo: {result.repo_dir}")
    print(
        "Inserted chunks "
        f"(notes={result.notes_chunks_inserted}, git={result.git_chunks_inserted}, "
        f"pdf={result.pdf_chunks_inserted}, image={result.image_chunks_inserted}); "
        f"eval queries upserted={result.eval_queries_upserted}; "
        f"usage events seeded={result.usage_events_seeded}"
    )


# Connector families for `mem sync` / `mem doctor`: (name, env prefix).
_SYNC_CONNECTORS = (
    ("notes", "OBSIDIAN_VAULT_PATH"),
    ("git", "REPO_PATH"),
    ("pdf", "PDF_PATH"),
    ("image", "IMAGE_PATH"),
)


def sync_cmd(max_commits: int = 300, only: list[str] | None = None, as_json: bool = False) -> None:
    """Incrementally (re-)ingest every connector configured in the environment, in one pass.

    Idempotent: ingestion fingerprint-skips unchanged sources, so re-running only re-chunks what
    changed. PDF/image are skipped (not errored) when their extra is absent. A bad path is recorded
    per-connector and does not abort the rest of the sync.
    """
    from crossmodalrag.capabilities import has_ocr, has_pdf

    available = {"notes": True, "git": True, "pdf": has_pdf(), "ocr_ok": has_ocr()}
    db_path = get_db_path()
    embedder = get_default_provider()
    report: list[dict] = []
    total_inserted = 0

    conn = connect(db_path)
    try:
        init_db(conn)
        for name, prefix in _SYNC_CONNECTORS:
            if only and name not in only:
                continue
            paths = get_connector_paths(name)
            if not paths:
                report.append({"connector": name, "paths": 0, "inserted": 0, "status": "no-paths"})
                continue
            if name == "pdf" and not available["pdf"]:
                report.append({"connector": name, "paths": len(paths), "inserted": 0,
                               "status": "skipped: install the [pdf] extra"})
                continue
            if name == "image" and not available["ocr_ok"]:
                report.append({"connector": name, "paths": len(paths), "inserted": 0,
                               "status": "skipped: install the [ocr] extra + tesseract"})
                continue

            inserted = 0
            errors: list[str] = []
            for p in paths:
                try:
                    if name == "notes":
                        inserted += ingest_notes(conn, vault_path=p, embedder=embedder,
                                                 progress=make_progress(f"notes {p.name}"))
                    elif name == "git":
                        inserted += ingest_git(conn, repo_path=p, max_commits=max_commits,
                                               embedder=embedder, progress=make_progress(f"git {p.name}"))
                    elif name == "pdf":
                        inserted += ingest_pdf(conn, pdf_path=p, embedder=embedder,
                                               progress=make_progress(f"pdf {p.name}"))
                    elif name == "image":
                        inserted += ingest_images(conn, image_path=p, embedder=embedder,
                                                  progress=make_progress(f"images {p.name}"))
                except (FileNotFoundError, MissingModalityBackend, sqlite3.Error) as exc:
                    errors.append(f"{p}: {exc}")
            total_inserted += inserted
            status = "ok" if not errors else f"ok with {len(errors)} error(s)"
            entry = {"connector": name, "paths": len(paths), "inserted": inserted, "status": status}
            if errors:
                entry["errors"] = errors
            report.append(entry)
    finally:
        conn.close()

    if as_json:
        print(json.dumps({"db": str(db_path), "connectors": report, "total_inserted": total_inserted}, indent=2))
        return
    print(f"Sync DB: {db_path}")
    if not report:
        print("No connectors configured. Set OBSIDIAN_VAULT_PATH_*, REPO_PATH_*, PDF_PATH_*, IMAGE_PATH_* in .env.")
        return
    for r in report:
        print(f"  {r['connector']}: {r['paths']} path(s), {r['inserted']} chunk(s) inserted [{r['status']}]")
        for err in r.get("errors", []):
            print(f"      error: {err}")
    print(f"Total inserted chunks (changed sources): {total_inserted}")


def doctor_cmd(as_json: bool = False) -> None:
    """Read-only health check: DB, installed extras, Ollama reachability, models, connectors, memory."""
    from crossmodalrag.service import health_report

    report = health_report()
    stats = report["memory"]

    if as_json:
        print(json.dumps(report, indent=2))
        return

    print(f"Engram doctor. DB: {report['db']['path']}")
    print(f"  DB exists: {report['db']['exists']} ({report['db']['size_bytes']} bytes)")
    ex = report["extras"]
    print(f"  Extras: embeddings={ex['embeddings']} pdf={ex['pdf']} ocr={ex['ocr']}")
    print(f"  Ollama: reachable={report['ollama']['reachable']} ({report['ollama']['base_url']})")
    m = report["models"]
    print(f"  Models: embed={m['embed']} llm={m['llm']} extract={m['extract']}")
    cfg = report["config"]
    print(f"  Config file: {cfg['path'] or '(none)'} (loaded={cfg['loaded']})")
    print("  Connectors (effective paths): " + ", ".join(
        f"{name}={count}" for name, count in report["connectors"].items()))
    if stats is not None:
        integ = stats["integrity"]
        print(
            f"  Memory: nodes={stats['total_nodes']} embeddings={stats['node_embeddings']} "
            f"distilled={stats['distilled_nodes']} drift={stats['drift_snapshots']} | "
            f"integrity: unsupported={integ['unsupported_count']} dangling={integ['dangling_count']}"
        )
    else:
        print("  Memory: no database yet (run `mem init-db` / `mem sync`).")


def backup_cmd(dest: Path | None = None) -> None:
    """Write a consistent single-file copy of the local DB (uses SQLite's online backup, WAL-safe)."""
    from datetime import datetime

    db_path = get_db_path()
    if not db_path.exists():
        raise CLIError(f"No database to back up at {db_path}.")
    if dest is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = db_path.with_name(f"{db_path.name}.backup-{ts}")
    dest = dest.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    src = sqlite3.connect(db_path)
    try:
        out = sqlite3.connect(dest)
        try:
            src.backup(out)  # checkpoints WAL into a complete single file
        finally:
            out.close()
    finally:
        src.close()
    print(f"Backed up {db_path} -> {dest} ({dest.stat().st_size} bytes)")


def restore_cmd(src: Path, force: bool = False) -> None:
    """Replace the local DB with a backup. Destructive: requires --force to overwrite an existing DB."""
    import shutil

    src = src.expanduser().resolve()
    if not src.exists():
        raise CLIError(f"Backup file not found: {src}")
    # Validate it's actually a SQLite database before clobbering anything.
    try:
        probe = sqlite3.connect(src)
        try:
            probe.execute("PRAGMA schema_version").fetchone()
        finally:
            probe.close()
    except sqlite3.Error:
        raise CLIError(f"Not a valid SQLite database: {src}")

    db_path = get_db_path()
    if db_path.exists() and not force:
        raise CLIError(
            f"Refusing to overwrite the existing database at {db_path}. Re-run with --force to confirm."
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop stale WAL/SHM sidecars so they can't shadow the restored file.
    for suffix in ("-wal", "-shm"):
        side = db_path.with_name(db_path.name + suffix)
        if side.exists():
            side.unlink()
    shutil.copy2(src, db_path)
    print(f"Restored {db_path} from {src} ({db_path.stat().st_size} bytes)")


def serve_cmd(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the local read-only HTTP API (requires the `[ui]` extra; binds localhost by default)."""
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise CLIError(
            "`mem serve` requires the [ui] extra. Run: pip install -e \".[ui]\""
        ) from exc

    from crossmodalrag.api import MissingUIBackend, create_app

    loopback = host in ("127.0.0.1", "localhost", "::1")
    # The API rejects any request whose Host header is not a loopback name (DNS-rebinding
    # guard), so a deliberate non-loopback bind has to declare the address it advertises.
    # Wildcard binds resolve to no single name, hence the CMRAG_API_ALLOWED_HOSTS pointer.
    try:
        app = create_app(allowed_hosts=[] if loopback else [host])
    except MissingUIBackend as exc:
        raise CLIError(str(exc)) from exc

    if not loopback:
        print(
            f"WARNING: binding to {host} exposes the unauthenticated, read-only API beyond localhost.",
            file=sys.stderr,
        )
        if host in ("0.0.0.0", "::"):
            print(
                "         A wildcard bind answers on names this process cannot know: list them in "
                "CMRAG_API_ALLOWED_HOSTS or requests will be refused with 403.",
                file=sys.stderr,
            )
    print(f"Serving the Engram local API on http://{host}:{port}. Press Ctrl-C to stop.")
    uvicorn.run(app, host=host, port=port, log_level="info")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Engram local memory CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Initialize SQLite database schema.")

    p_sync = sub.add_parser(
        "sync",
        help="Incrementally (re-)ingest every connector configured in .env (notes/git/pdf/image). "
        "Idempotent: only changed sources are re-chunked.",
    )
    p_sync.add_argument("--max-commits", type=int, default=300, help="Max commits per repo (git).")
    p_sync.add_argument(
        "--only",
        choices=["notes", "git", "pdf", "image"],
        action="append",
        default=None,
        help="Restrict sync to one or more connectors (repeatable).",
    )
    p_sync.add_argument("--json", dest="as_json", action="store_true", help="Emit a sync summary as JSON.")

    p_doctor = sub.add_parser(
        "doctor",
        help="Read-only health check: DB, installed extras, Ollama reachability, models, connectors, memory.",
    )
    p_doctor.add_argument("--json", dest="as_json", action="store_true", help="Emit the report as JSON.")

    p_backup = sub.add_parser(
        "backup", help="Write a consistent single-file copy of the local database (WAL-safe)."
    )
    p_backup.add_argument(
        "dest", type=Path, nargs="?", default=None,
        help="Backup destination (default: alongside the DB with a timestamp suffix).",
    )

    p_restore = sub.add_parser(
        "restore", help="Replace the local database with a backup file (destructive; needs --force)."
    )
    p_restore.add_argument("src", type=Path, help="Backup file to restore from.")
    p_restore.add_argument(
        "--force", action="store_true", help="Overwrite the existing database (required when one exists).",
    )

    p_serve = sub.add_parser(
        "serve",
        help="Run the local read-only HTTP API for the web UI / Obsidian (requires the [ui] extra; "
        "binds localhost by default).",
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1 / loopback).")
    p_serve.add_argument("--port", type=int, default=8765, help="Bind port (default 8765).")

    p_notes = sub.add_parser(
        "ingest-notes",
        help="Ingest markdown notes from one or more vault paths (or use OBSIDIAN_VAULT_PATH_* from .env).",
    )
    p_notes.add_argument("vault_paths", nargs="*", type=Path)

    p_git = sub.add_parser(
        "ingest-git",
        help="Ingest git commits and diffs from one or more repos (or use REPO_PATH_* from .env).",
    )
    p_git.add_argument("repo_paths", nargs="*", type=Path)
    p_git.add_argument("--max-commits", type=int, default=300)

    p_pdf = sub.add_parser(
        "ingest-pdf",
        help="Ingest PDF text (per-page) from one or more files/dirs (or use PDF_PATH_* from .env). "
        'Requires the [pdf] extra: pip install -e ".[pdf]".',
    )
    p_pdf.add_argument("pdf_paths", nargs="*", type=Path)

    p_img = sub.add_parser(
        "ingest-images",
        help="Ingest image OCR text from one or more files/dirs (or use IMAGE_PATH_* from .env). "
        'Requires the [ocr] extra (pip install -e ".[ocr]") and a local tesseract binary.',
    )
    p_img.add_argument("image_paths", nargs="*", type=Path)

    profile_choices = sorted(PROFILE_WEIGHTS)

    level_choices = ["evidence", "event", "episode", "concept"]
    modality_choices = sorted(MODALITY_SOURCE_TYPES)

    def _add_ask_options(p, *, oneshot_only: bool) -> None:
        """Shared `ask`/`chat` options; oneshot_only adds the flags that only
        make sense for a single buffered query (--json contract, --accept)."""
        p.add_argument("--top-k", type=int, default=get_default_top_k())
        p.add_argument(
            "--level",
            choices=level_choices,
            default="evidence",
            help="Retrieval level: 'evidence' (L0 chunks, default) or a memory level "
            "(event/episode/concept), which drills matched nodes down to L0 for grounding.",
        )
        p.add_argument(
            "--profile",
            choices=profile_choices,
            default=get_default_profile(DEFAULT_PROFILE),
            help="Hybrid retrieval profile (vector/lexical/recency blend).",
        )
        p.add_argument(
            "--explain",
            action="store_true",
            help="Show per-hit score components (vector/lexical/recency/combined).",
        )
        p.add_argument(
            "--no-llm",
            action="store_true",
            help="Skip LLM synthesis; return the deterministic evidence template.",
        )
        if oneshot_only:
            p.add_argument(
                "--json",
                action="store_true",
                dest="as_json",
                help="Emit a structured JSON answer (stable contract for UIs); one-shot only.",
            )
        p.add_argument(
            "--debug",
            action="store_true",
            help="Include retrieval diagnostics, the raw prompt, and raw model output.",
        )
        p.add_argument(
            "--modality",
            choices=modality_choices,
            action="append",
            default=None,
            help="Restrict evidence to one or more modalities (repeatable). "
            "Maps to source types: text=notes, code=git, pdf=PDFs, image=OCR'd images.",
        )
        if oneshot_only:
            p.add_argument(
                "--accept",
                action="store_true",
                help="Record this answer as accepted (usage feedback on the cited evidence); "
                "enables tracking. One-shot only.",
            )
        p.add_argument(
            "--track",
            action="store_true",
            help="Log usage events for this query (overrides CMRAG_USAGE_TRACKING for the call).",
        )
        p.add_argument(
            "--no-track",
            action="store_true",
            help="Do not log usage events for this query, even if tracking is enabled by env.",
        )
        p.add_argument(
            "--no-stream",
            action="store_true",
            help="Print the answer only once generation finishes instead of streaming tokens "
            "live (streaming applies to interactive terminals only; --json is always buffered).",
        )
        p.add_argument(
            "--no-save",
            action="store_true",
            help="Don't persist this session to local chat history (interactive sessions only; "
            "overrides CMRAG_SAVE_HISTORY; one-shot queries are never saved).",
        )
        p.add_argument(
            "--resume",
            type=int,
            nargs="?",
            const=0,
            default=None,
            metavar="ID",
            help="Resume a saved conversation in an interactive session: load its turns as "
            "context and append new turns to it. Omit ID to resume the most recent one.",
        )

    p_ask = sub.add_parser("ask", help="Query indexed evidence.")
    p_ask.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Question to ask. Omit to start an interactive multi-turn session "
        "(/exit or Ctrl-D to quit, /clear to reset context).",
    )
    _add_ask_options(p_ask, oneshot_only=True)

    p_chat = sub.add_parser(
        "chat",
        help="Interactive multi-turn ask session (same as `mem ask` with no query).",
    )
    _add_ask_options(p_chat, oneshot_only=False)

    p_history = sub.add_parser(
        "history",
        help="List/show saved chat conversations (local), or --clear to wipe history.",
    )
    p_history.add_argument("--show", type=int, metavar="ID", help="Show one conversation in full.")
    p_history.add_argument("--top", type=int, default=10, help="How many conversations to list.")
    p_history.add_argument(
        "--clear", action="store_true", help="Delete saved conversations (local, no confirmation)."
    )
    p_history.add_argument(
        "--id", type=int, default=None, dest="clear_id",
        help="With --clear: delete only this conversation.",
    )
    p_history.add_argument(
        "--rename", type=int, default=None, dest="rename_id", metavar="ID",
        help="Rename one conversation (requires --title).",
    )
    p_history.add_argument(
        "--title", type=str, default=None, dest="new_title",
        help='With --rename: the new conversation title, e.g. --title "Parser deep dive".',
    )
    p_history.add_argument("--json", dest="as_json", action="store_true")

    p_eval = sub.add_parser(
        "eval",
        help="Run retrieval evaluation using queries stored in queries_eval (optionally load from JSON).",
    )
    p_eval.add_argument("--top-k", type=int, default=get_default_top_k())
    p_eval.add_argument(
        "--query-prefix",
        type=str,
        default=None,
        help="Only evaluate queries whose text starts with this prefix (e.g. '[sample]').",
    )
    p_eval.add_argument(
        "--load-queries",
        type=Path,
        default=None,
        help="JSON file of eval query rows to upsert before running evaluation.",
    )
    p_eval.add_argument(
        "--profile",
        choices=profile_choices,
        default=get_default_profile(DEFAULT_PROFILE),
        help="Hybrid retrieval profile (vector/lexical/recency blend).",
    )
    p_eval.add_argument(
        "--level",
        choices=level_choices,
        default="evidence",
        help="Retrieval level: 'evidence' (default) or a memory level evaluated via drill-down.",
    )
    p_eval.add_argument(
        "--modality",
        choices=modality_choices,
        action="append",
        default=None,
        help="Restrict evidence to one or more modalities (repeatable): text/code/pdf/image.",
    )
    p_eval.add_argument("--json", dest="as_json", action="store_true", help="Emit metrics as JSON.")

    p_reindex = sub.add_parser(
        "reindex-embeddings",
        help="Backfill/repair chunk embeddings for the active model (requires the 'embeddings' extra).",
    )
    p_reindex.add_argument("--batch-size", type=int, default=64)
    p_reindex.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Embedding model id (defaults to CMRAG_EMBED_MODEL or {DEFAULT_EMBED_MODEL}).",
    )

    p_eval_gen = sub.add_parser(
        "eval-generation",
        help="Evaluate grounded answer synthesis (citation validity, source grounding, abstention).",
    )
    p_eval_gen.add_argument("--top-k", type=int, default=5)
    p_eval_gen.add_argument(
        "--query-prefix",
        type=str,
        default=None,
        help="Only evaluate queries whose text starts with this prefix (e.g. '[sample]').",
    )
    p_eval_gen.add_argument(
        "--profile",
        choices=profile_choices,
        default=DEFAULT_PROFILE,
        help="Hybrid retrieval profile (vector/lexical/recency blend).",
    )
    p_eval_gen.add_argument(
        "--level",
        choices=level_choices,
        default="evidence",
        help="Retrieval level: 'evidence' (default) or a memory level; higher levels drill "
        "matched nodes down to L0 before synthesis (answers still cite L0).",
    )
    p_eval_gen.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model id (defaults to CMRAG_LLM_MODEL).",
    )

    p_build = sub.add_parser(
        "build-memory",
        help="Derive hierarchical memory from L0 evidence (L1 events via Ollama; L2 episodes deterministic).",
    )
    p_build.add_argument(
        "--level",
        choices=["event", "episode", "concept", "graph", "drift", "distill", "all"],
        default="all",
        help="Memory level to build: 'event' (LLM), 'episode' (no LLM), "
        "'concept' (embeddings extra; LLM naming optional), 'graph' (no LLM/embeddings), "
        "'drift' (concept drift over time windows; embeddings extra, no LLM), "
        "'distill' (compact retrieval-preserving node stand-ins; embeddings extra, LLM summary "
        "optional), or 'all' (default).",
    )
    p_build.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of sources to (re)process for event extraction this run (resumable).",
    )
    p_build.add_argument(
        "--model",
        type=str,
        default=None,
        help="Extraction model id (defaults to CMRAG_EXTRACT_MODEL).",
    )

    p_stats = sub.add_parser(
        "memory-stats",
        help="Show hierarchical memory node/edge counts and structural integrity status.",
    )
    p_stats.add_argument("--json", dest="as_json", action="store_true", help="Emit stats as JSON.")

    p_concepts = sub.add_parser("concepts", help="List L3 concepts ranked by centrality.")
    p_concepts.add_argument("--top", type=int, default=20)
    p_concepts.add_argument("--json", dest="as_json", action="store_true", help="Emit concepts as JSON.")

    p_timeline = sub.add_parser("timeline", help="List L2 episodes chronologically.")
    p_timeline.add_argument("--limit", type=int, default=50)
    p_timeline.add_argument("--json", dest="as_json", action="store_true", help="Emit the timeline as JSON.")

    p_usage = sub.add_parser(
        "usage",
        help="Show local usage-tracking stats (read-only), or --clear to wipe usage history.",
    )
    p_usage.add_argument("--clear", action="store_true", help="Delete all usage events (local).")
    p_usage.add_argument("--top", type=int, default=10, help="How many top targets to show.")
    p_usage.add_argument("--json", dest="as_json", action="store_true", help="Emit usage stats as JSON.")

    p_forgetting = sub.add_parser(
        "forgetting",
        help="Rank important-but-stale memories ('what am I likely forgetting?'). Read-only.",
    )
    p_forgetting.add_argument(
        "--level",
        choices=["concept", "episode", "event", "all"],
        default="concept",
        help="Which memory level to score (default concept).",
    )
    p_forgetting.add_argument("--top", type=int, default=10)
    p_forgetting.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Minimum grounded L0 chunks for a node to be considered.",
    )
    p_forgetting.add_argument("--json", dest="as_json", action="store_true", help="Emit risks as JSON.")

    p_drift = sub.add_parser(
        "drift",
        help="Show how L3 concepts have moved across time windows (concept drift). Read-only; "
        "run `mem build-memory --level drift` first.",
    )
    p_drift.add_argument("--top", type=int, default=10)
    p_drift.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Minimum total member events (across windows) for a concept to be shown.",
    )
    p_drift.add_argument(
        "--json",
        action="store_true",
        help="Emit the drift summaries as JSON (stable contract incl. the per-window trajectory).",
    )

    p_distill = sub.add_parser(
        "distill",
        help="List distilled node stand-ins (core vs full evidence, compression ratio). Read-only; "
        "run `mem build-memory --level distill` first.",
    )
    p_distill.add_argument("--top", type=int, default=10)
    p_distill.add_argument(
        "--json",
        action="store_true",
        help="Emit the distilled summaries as JSON (stable contract + overall compression ratio).",
    )

    p_recall = sub.add_parser(
        "recall",
        help="Generate grounded active-recall study cards for the memories you're most likely "
        "forgetting (local LLM, cached; deterministic fallback when Ollama is absent).",
    )
    p_recall.add_argument(
        "--level",
        choices=["concept", "episode", "event", "all"],
        default="concept",
        help="Which memory level to generate cards for (default concept).",
    )
    p_recall.add_argument("--top", type=int, default=10)
    p_recall.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Minimum grounded L0 chunks for a node to be considered.",
    )
    p_recall.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of cards (e.g. to upgrade fallback cards once Ollama is available).",
    )
    p_recall.add_argument("--json", dest="as_json", action="store_true", help="Emit cards as JSON.")

    p_seed = sub.add_parser(
        "seed-sample",
        help="Create deterministic synthetic notes/git fixtures and ingest them into the DB.",
    )
    p_seed.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("data") / "sample-seed-workspace",
        help="Directory for generated synthetic sample vault and git repo.",
    )
    p_seed.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="DB path for synthetic sample data (defaults to a temp DB, not the main memory DB).",
    )
    p_seed.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the sample workspace directory if it already exists (destructive).",
    )

    return parser


class CLIError(Exception):
    """An expected, user-facing CLI failure: reported as `error: <msg>` with exit code 1 (no traceback)."""


# Expected failures are surfaced as a clean message + exit 1 (not a traceback). Argparse usage errors
# keep argparse's own exit code 2; unexpected bugs still raise so they are visible.
_EXPECTED_ERRORS = (
    CLIError,
    FileNotFoundError,
    MissingModalityBackend,
    MissingEmbeddingBackend,
    LLMUnavailable,
    sqlite3.Error,
)


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    try:
        _dispatch(parser, args)
    except _EXPECTED_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command == "init-db":
        init_db_cmd()
        return
    if args.command == "sync":
        sync_cmd(max_commits=args.max_commits, only=args.only, as_json=args.as_json)
        return
    if args.command == "doctor":
        doctor_cmd(as_json=args.as_json)
        return
    if args.command == "backup":
        backup_cmd(dest=args.dest)
        return
    if args.command == "restore":
        restore_cmd(src=args.src, force=args.force)
        return
    if args.command == "serve":
        serve_cmd(host=args.host, port=args.port)
        return
    if args.command == "ingest-notes":
        vault_paths = _resolve_ingest_paths(
            args.vault_paths,
            connector="notes",
            command_name="ingest-notes",
        )
        if not vault_paths:
            parser.error(
                "No vault paths provided. Use `mem ingest-notes <vault_path> [<vault_path> ...]` "
                "or define `OBSIDIAN_VAULT_PATH_1`, `OBSIDIAN_VAULT_PATH_2`, ... in your local .env "
                "for default ingestion targets."
            )
        ingest_notes_cmd(vault_paths)
        return
    if args.command == "ingest-git":
        repo_paths = _resolve_ingest_paths(
            args.repo_paths,
            connector="git",
            command_name="ingest-git",
        )
        if not repo_paths:
            parser.error(
                "No repo paths provided. Use `mem ingest-git <repo_path> [<repo_path> ...] [--max-commits N]` "
                "or define `REPO_PATH_1`, `REPO_PATH_2`, ... in your local .env "
                "for default ingestion targets."
            )
        ingest_git_cmd(repo_paths, max_commits=args.max_commits)
        return
    if args.command == "ingest-pdf":
        pdf_paths = _resolve_ingest_paths(
            args.pdf_paths,
            connector="pdf",
            command_name="ingest-pdf",
        )
        if not pdf_paths:
            parser.error(
                "No PDF paths provided. Use `mem ingest-pdf <path> [<path> ...]` "
                "or define `PDF_PATH_1`, `PDF_PATH_2`, ... in your local .env "
                "for default ingestion targets."
            )
        ingest_pdf_cmd(pdf_paths)
        return
    if args.command == "ingest-images":
        image_paths = _resolve_ingest_paths(
            args.image_paths,
            connector="image",
            command_name="ingest-images",
        )
        if not image_paths:
            parser.error(
                "No image paths provided. Use `mem ingest-images <path> [<path> ...]` "
                "or define `IMAGE_PATH_1`, `IMAGE_PATH_2`, ... in your local .env "
                "for default ingestion targets."
            )
        ingest_images_cmd(image_paths)
        return
    if args.command in {"ask", "chat"}:
        track = True if args.track else (False if args.no_track else None)
        query = getattr(args, "query", None)
        if query is None:
            # Interactive multi-turn session (`mem chat`, or `mem ask` with no query).
            if getattr(args, "as_json", False):
                raise CLIError('--json requires a one-shot query: mem ask --json "<question>"')
            if getattr(args, "accept", False):
                raise CLIError('--accept requires a one-shot query: mem ask --accept "<question>"')
            chat_cmd(
                top_k=args.top_k,
                profile=args.profile,
                explain=args.explain,
                use_llm=not args.no_llm,
                debug=args.debug,
                level=args.level,
                modalities=args.modality,
                track=track,
                stream=not args.no_stream,
                save=False if args.no_save else save_history_enabled(),
                resume=args.resume,
            )
            return
        if getattr(args, "resume", None) is not None:
            raise CLIError("--resume starts an interactive session; omit the query to use it.")
        ask_cmd(
            query,
            top_k=args.top_k,
            profile=args.profile,
            explain=args.explain,
            use_llm=not args.no_llm,
            as_json=args.as_json,
            debug=args.debug,
            level=args.level,
            modalities=args.modality,
            accept=args.accept,
            track=track,
            stream=not args.no_stream,
        )
        return
    if args.command == "history":
        history_cmd(
            show_id=args.show,
            top=args.top,
            clear=args.clear,
            clear_id=args.clear_id,
            rename_id=args.rename_id,
            new_title=args.new_title,
            as_json=args.as_json,
        )
        return
    if args.command == "eval":
        eval_cmd(
            top_k=args.top_k,
            query_prefix=args.query_prefix,
            load_queries_path=args.load_queries,
            profile=args.profile,
            level=args.level,
            modalities=args.modality,
            as_json=args.as_json,
        )
        return
    if args.command == "reindex-embeddings":
        reindex_embeddings_cmd(batch_size=args.batch_size, model=args.model)
        return
    if args.command == "eval-generation":
        eval_generation_cmd(
            top_k=args.top_k,
            query_prefix=args.query_prefix,
            profile=args.profile,
            model=args.model,
            level=args.level,
        )
        return
    if args.command == "build-memory":
        build_memory_cmd(level=args.level, limit=args.limit, model=args.model)
        return
    if args.command == "memory-stats":
        memory_stats_cmd(as_json=args.as_json)
        return
    if args.command == "concepts":
        concepts_cmd(top=args.top, as_json=args.as_json)
        return
    if args.command == "timeline":
        timeline_cmd(limit=args.limit, as_json=args.as_json)
        return
    if args.command == "usage":
        usage_cmd(clear=args.clear, top=args.top, as_json=args.as_json)
        return
    if args.command == "forgetting":
        forgetting_cmd(level=args.level, top=args.top, min_support=args.min_support, as_json=args.as_json)
        return
    if args.command == "drift":
        drift_cmd(top=args.top, min_support=args.min_support, as_json=args.json)
        return
    if args.command == "distill":
        distill_cmd(top=args.top, as_json=args.json)
        return
    if args.command == "recall":
        recall_cmd(
            level=args.level,
            top=args.top,
            min_support=args.min_support,
            regenerate=args.regenerate,
            as_json=args.as_json,
        )
        return
    if args.command == "seed-sample":
        seed_sample_cmd(args.workspace_dir, force=args.force, db_path=args.db_path)
        return
    parser.error(f"Unknown command: {args.command}")


def _resolve_ingest_paths(
    explicit_paths: list[Path],
    *,
    connector: str,
    command_name: str,
) -> list[Path]:
    if explicit_paths:
        return [path.expanduser().resolve() for path in explicit_paths]
    # Fall back to configured paths: environment (.env) first, then the config file.
    paths = get_connector_paths(connector)
    if paths:
        print(
            f"No explicit paths provided for `{command_name}`. "
            f"Using {len(paths)} configured path(s) (environment / config file)."
        )
        return paths
    return []


if __name__ == "__main__":
    main()
