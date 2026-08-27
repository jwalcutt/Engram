"""Tests for LLM conversation naming and session resume."""

from __future__ import annotations


from crossmodalrag.conversations.naming import (
    TITLE_MAX_WORDS,
    generate_conversation_title,
    sanitize_title,
)
from crossmodalrag.conversations.resume import next_turn_index, turns_from_messages
from crossmodalrag.conversations.store import MessageRow
from crossmodalrag.generate.provider import LLMUnavailable


class _TitleProvider:
    name = "stub-llm"

    def __init__(self, output: str) -> None:
        self._output = output
        self.prompts: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append((prompt, system))
        return self._output

    def generate_stream(self, prompt: str, system: str | None = None):
        yield self.generate(prompt, system=system)


# --- naming -------------------------------------------------------------------------


def test_generate_title_uses_first_exchange_and_sanitizes():
    provider = _TitleProvider('  "Parser Bounds Fix Discussion."  ')
    title = generate_conversation_title(provider, query="why did x?", answer_text="because y")
    assert title == "Parser Bounds Fix Discussion"
    prompt, system = provider.prompts[0]
    assert "why did x?" in prompt
    assert "because y" in prompt
    assert "at most 6 words" in system


def test_generate_title_none_on_llm_unavailable():
    class _Down:
        name = "down"

        def generate(self, prompt, system=None):
            raise LLMUnavailable("no ollama")

        def generate_stream(self, prompt, system=None):
            raise LLMUnavailable("no ollama")

    assert generate_conversation_title(_Down(), query="q", answer_text="a") is None


def test_generate_title_none_on_empty_output():
    assert generate_conversation_title(_TitleProvider("   \n"), query="q", answer_text="a") is None


def test_sanitize_title_clamps_words_and_chars():
    long = " ".join(["word"] * 20)
    sane = sanitize_title(long)
    assert len(sane.split()) == TITLE_MAX_WORDS
    assert sanitize_title("Title: The Actual Name") == "The Actual Name"
    assert sanitize_title("First line\nsecond line") == "First line"
    assert sanitize_title("Ends with punctuation!?") == "Ends with punctuation"
    assert sanitize_title("x" * 100).endswith("…")


# --- resume turn reconstruction --------------------------------------------------------


def _msg(mid, turn, role, text, reason=None):
    return MessageRow(
        id=mid, conversation_id=1, turn_index=turn, role=role, text=text,
        evidence_json=None, abstention_reason=reason, truncated=False,
        model=None if role == "user" else "stub", created_at="2026-07-12T00:00:00+00:00",
    )


def test_turns_from_messages_pairs_and_skips_abstained():
    messages = [
        _msg(1, 0, "user", "q0"),
        _msg(2, 0, "assistant", "a0 [E1]"),
        _msg(3, 1, "user", "q1-abstained"),
        _msg(4, 1, "assistant", "Insufficient…", reason="llm_insufficient"),
        _msg(5, 2, "user", "q2"),
        _msg(6, 2, "assistant", "a2"),
    ]
    turns = turns_from_messages(messages)
    assert [(t.query, t.answer_text) for t in turns] == [("q0", "a0 [E1]"), ("q2", "a2")]
    assert next_turn_index(messages) == 3
    assert next_turn_index([]) == 0
