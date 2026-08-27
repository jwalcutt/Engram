"""Library-level tests for the multi-turn chat session state and the additive
``history`` threading through prompt building, synthesis, and the service
stream. No DB, no CLI — pure logic plus stub providers."""

from __future__ import annotations

import pytest

from crossmodalrag.chat import (
    HISTORY_HEADER,
    ChatSession,
    ChatTurn,
    render_history,
    strip_citations,
)
from crossmodalrag.generate.synthesize import (
    CHAT_SYSTEM_ADDENDUM,
    CITATION_RE,
    INSUFFICIENT_EVIDENCE_TEXT,
    build_evidence_prompt,
    synthesize_answer,
    synthesize_answer_stream,
)
from crossmodalrag.retrieve.lexical import RetrievalHit
from crossmodalrag import service


def _hit(chunk_id: int, text: str, uri: str = "/abs/note.md", score: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_id=chunk_id,
        source_type="note",
        source_uri=uri,
        source_timestamp="2026-06-01T00:00:00+00:00",
        title="note",
        chunk_index=0,
        chunk_text=text,
        score=score,
        lexical_score=score,
        recency_score=0.5,
        vector_score=0.0,
    )


class RecordingProvider:
    """Stub capturing every (prompt, system) pair it is asked to generate for."""

    def __init__(self, output: str = "Answer [E1].", name: str = "stub-llm") -> None:
        self.name = name
        self._output = output
        self.prompts: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append((prompt, system))
        return self._output

    def generate_stream(self, prompt: str, system: str | None = None):
        self.prompts.append((prompt, system))
        yield self._output


# --- session state ------------------------------------------------------------------


def test_render_history_format() -> None:
    turns = [ChatTurn("what is X?", "X is a thing."), ChatTurn("and Y?", "Y differs.")]
    rendered = render_history(turns)
    assert rendered.splitlines() == [
        HISTORY_HEADER,
        "User: what is X?",
        "Assistant: X is a thing.",
        "User: and Y?",
        "Assistant: Y differs.",
    ]


def test_render_history_empty_is_empty_string() -> None:
    assert render_history([]) == ""


def test_strip_citations_removes_stale_evidence_ids() -> None:
    assert strip_citations("Claim [E1]. Another [E12][E3].") == "Claim . Another ."


def test_turns_store_raw_answer_but_render_stripped() -> None:
    # Stored raw (persistence-ready); sanitized only in the prompt view, so a
    # stale [E#] can never collide with the current turn's evidence numbering.
    session = ChatSession(max_turns=8)
    session.add_turn("q", "Grounded claim [E2].")
    assert session.turns[0].answer_text == "Grounded claim [E2]."
    rendered = render_history(session.turns)
    # No digit-bearing citation id survives rendering (the literal "[E#]" in the
    # header is instructional text, not a citation — CITATION_RE needs digits).
    assert CITATION_RE.search(rendered) is None
    assert "Grounded claim" in rendered


def test_cap_evicts_oldest_first() -> None:
    session = ChatSession(max_turns=2)
    for i in range(3):
        session.add_turn(f"q{i}", f"a{i}")
    assert [t.query for t in session.turns] == ["q1", "q2"]


def test_zero_cap_disables_carried_context() -> None:
    session = ChatSession(max_turns=0)
    session.add_turn("q", "a")
    assert session.turns == []


def test_abstained_turn_is_skipped_entirely() -> None:
    session = ChatSession(max_turns=8)
    session.add_turn("unanswerable?", INSUFFICIENT_EVIDENCE_TEXT, abstained=True)
    assert session.turns == []


def test_clear_resets_turns() -> None:
    session = ChatSession(max_turns=8)
    session.add_turn("q", "a")
    session.clear()
    assert session.turns == []


def test_default_cap_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMRAG_CHAT_CONTEXT_TURNS", "3")
    assert ChatSession().max_turns == 3


# --- prompt threading ----------------------------------------------------------------


def test_build_evidence_prompt_ablation_byte_identical() -> None:
    hits = [_hit(1, "alpha"), _hit(2, "beta")]
    base = build_evidence_prompt("q", hits)
    assert build_evidence_prompt("q", hits, history=None) == base
    assert build_evidence_prompt("q", hits, history="") == base


def test_build_evidence_prompt_places_history_before_evidence() -> None:
    hits = [_hit(1, "alpha")]
    history = render_history([ChatTurn("earlier q", "earlier a")])
    system, prompt, id_map = build_evidence_prompt("q", hits, history=history)
    _, base_prompt, base_map = build_evidence_prompt("q", hits)
    assert prompt.startswith(HISTORY_HEADER)
    assert prompt.index(HISTORY_HEADER) < prompt.index("Evidence:")
    # The tail (evidence + question + answer cue) is exactly the single-turn prompt.
    assert prompt.endswith(base_prompt)
    assert id_map == base_map
    # Chat turns extend the system prompt with the conversation-meta addendum;
    # the step-10 SYSTEM_PROMPT itself is untouched (it stays the prefix).
    base_system = build_evidence_prompt("q", hits)[0]
    assert system.startswith(base_system)
    assert CHAT_SYSTEM_ADDENDUM in system


def test_synthesize_answer_threads_history() -> None:
    provider = RecordingProvider()
    history = render_history([ChatTurn("earlier q", "earlier a")])
    gen = synthesize_answer("q", [_hit(1, "alpha")], provider, min_evidence_score=0.0, history=history)
    assert not gen.abstained
    prompt = provider.prompts[0][0]
    assert prompt.startswith(HISTORY_HEADER)
    assert "earlier a" in prompt


def test_synthesize_answer_stream_threads_history() -> None:
    provider = RecordingProvider()
    history = render_history([ChatTurn("earlier q", "earlier a")])
    stream = synthesize_answer_stream(
        "q", [_hit(1, "alpha")], provider, min_evidence_score=0.0, history=history
    )
    while True:
        try:
            next(stream)
        except StopIteration:
            break
    assert provider.prompts[0][0].startswith(HISTORY_HEADER)


def test_gate_with_history_still_short_circuits_before_llm() -> None:
    provider = RecordingProvider()
    gen = synthesize_answer(
        "q", [], provider, min_evidence_score=0.5, history=render_history([ChatTurn("a", "b")])
    )
    assert gen.abstained
    assert provider.prompts == []  # never reached the LLM


def test_service_stream_answer_events_threads_history(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = RecordingProvider()
    monkeypatch.setattr(service, "get_default_llm_provider", lambda: provider)
    history = render_history([ChatTurn("earlier q", "earlier a")])
    events = list(
        service.stream_answer_events(
            query="q", hits=[_hit(1, "alpha")], matched_nodes=[], history=history
        )
    )
    assert events[-1]["type"] == "answer"
    assert provider.prompts[0][0].startswith(HISTORY_HEADER)


def test_render_history_caps_long_answers() -> None:
    from crossmodalrag.chat import HISTORY_ANSWER_MAX_CHARS

    long_answer = "x" * (HISTORY_ANSWER_MAX_CHARS + 500)
    rendered = render_history([ChatTurn("q", long_answer)])
    assistant_line = [line for line in rendered.splitlines() if line.startswith("Assistant:")][0]
    assert assistant_line.endswith("[…truncated]")
    assert len(assistant_line) < HISTORY_ANSWER_MAX_CHARS + 100
    # Short answers are untouched.
    assert render_history([ChatTurn("q", "short")]).endswith("Assistant: short")
