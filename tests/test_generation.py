from __future__ import annotations

import json

import pytest

from crossmodalrag.db import connect, init_db
from crossmodalrag.generate import provider as provider_mod
from crossmodalrag.generate.answer import format_generated_answer, generated_answer_to_dict
from crossmodalrag.generate.provider import LLMUnavailable, OllamaProvider
from crossmodalrag.generate.synthesize import (
    INSUFFICIENT_EVIDENCE_TEXT,
    parse_citations,
    synthesize_answer,
)
from crossmodalrag.generation_eval import run_generation_eval
from crossmodalrag.memory.store import add_edge, insert_node
from crossmodalrag.retrieve.lexical import RetrievalHit


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


class StubLLMProvider:
    def __init__(self, output: str = "Synthesized finding [E1].", name: str = "stub-llm") -> None:
        self.name = name
        self._output = output
        self.calls = 0

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        return self._output


def test_parse_citations_dedupes_and_orders() -> None:
    assert parse_citations("a [E2] b [E1] c [E2]") == ["E2", "E1"]


def test_synthesize_splits_valid_and_invalid_citations() -> None:
    hits = [_hit(1, "alpha"), _hit(2, "beta")]
    provider = StubLLMProvider(output="Claim one [E1]. Claim two [E5].")
    gen = synthesize_answer("q", hits, provider, min_evidence_score=0.0)
    assert gen.cited_evidence_ids == ["E1"]
    assert gen.invalid_citations == ["E5"]
    assert not gen.abstained


def test_weak_retrieval_gate_abstains_without_calling_llm() -> None:
    provider = StubLLMProvider()
    # Top score below threshold -> abstain before any generation.
    gen = synthesize_answer("q", [_hit(1, "alpha", score=0.05)], provider, min_evidence_score=0.15)
    assert gen.abstained
    assert gen.answer_text == INSUFFICIENT_EVIDENCE_TEXT
    assert provider.calls == 0


def test_no_hits_abstains() -> None:
    provider = StubLLMProvider()
    gen = synthesize_answer("q", [], provider)
    assert gen.abstained
    assert provider.calls == 0


def test_render_plain_text_includes_evidence_map() -> None:
    hits = [_hit(7, "evidence body")]
    gen = synthesize_answer("why", hits, StubLLMProvider(output="Because [E1]."), min_evidence_score=0.0)
    text = format_generated_answer(gen)
    assert "[E1]" in text
    assert "chunk_id=7" in text
    assert "Because [E1]." in text


def test_json_contract_invariants() -> None:
    hits = [_hit(7, "evidence body")]
    gen = synthesize_answer("why", hits, StubLLMProvider(output="Because [E1]."), min_evidence_score=0.0)
    data = generated_answer_to_dict(gen)
    assert data["model"] == "stub-llm"
    assert data["abstained"] is False
    assert data["evidence"][0]["evidence_id"] == "E1"
    assert data["evidence"][0]["cited"] is True
    # Round-trips as JSON (stable contract).
    json.dumps(data)


def _modal_hit(chunk_id, source_type, uri, metadata: dict, text="evidence body") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_id=chunk_id,
        source_type=source_type,
        source_uri=uri,
        source_timestamp="2026-06-01T00:00:00+00:00",
        title=source_type,
        chunk_index=0,
        chunk_text=text,
        score=0.9,
        lexical_score=0.9,
        recency_score=0.5,
        chunk_metadata_json=json.dumps(metadata),
    )


def test_json_adds_modality_locator_without_dropping_existing_keys() -> None:
    from crossmodalrag.modality import MODALITY_OCR, MODALITY_PDF_PAGE, build_chunk_metadata

    hits = [
        _modal_hit(1, "pdf", "/abs/spec.pdf", build_chunk_metadata(modality=MODALITY_PDF_PAGE, source_type="pdf", page=4)),
        _modal_hit(2, "image", "/abs/diagram.png", build_chunk_metadata(modality=MODALITY_OCR, source_type="image", ocr_confidence=0.95)),
    ]
    gen = synthesize_answer("q", hits, StubLLMProvider(output="See [E1] and [E2]."), min_evidence_score=0.0)
    data = generated_answer_to_dict(gen)

    pdf, img = data["evidence"][0], data["evidence"][1]
    # New additive fields.
    assert pdf["modality"] == "pdf-page"
    assert pdf["locator"] == "/abs/spec.pdf p.4"
    assert pdf["page"] == 4
    assert pdf["ocr_confidence"] is None
    assert img["modality"] == "ocr"
    assert img["ocr_confidence"] == 0.95
    # Existing keys remain present and stable.
    for key in ("evidence_id", "cited", "source_id", "chunk_id", "source_type", "source_uri", "title", "scores", "excerpt"):
        assert key in pdf
    json.dumps(data)


def test_plain_text_renders_page_locator_and_ocr_confidence() -> None:
    from crossmodalrag.modality import MODALITY_OCR, MODALITY_PDF_PAGE, build_chunk_metadata

    hits = [
        _modal_hit(1, "pdf", "/abs/spec.pdf", build_chunk_metadata(modality=MODALITY_PDF_PAGE, source_type="pdf", page=4)),
        _modal_hit(2, "image", "/abs/diagram.png", build_chunk_metadata(modality=MODALITY_OCR, source_type="image", ocr_confidence=0.95)),
    ]
    gen = synthesize_answer("q", hits, StubLLMProvider(output="See [E1] and [E2]."), min_evidence_score=0.0)
    text = format_generated_answer(gen)

    assert "spec.pdf p.4" in text
    assert "ocr_conf=0.95" in text


class _FakeNDJSONResp:
    """Fake urlopen response emitting Ollama's streaming NDJSON protocol."""

    def __init__(self, lines: list[dict]) -> None:
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(json.dumps(line).encode("utf-8") + b"\n" for line in self._lines)


def test_ollama_provider_parses_response(monkeypatch) -> None:
    resp = _FakeNDJSONResp(
        [{"response": "hello ", "done": False}, {"response": "world", "done": True}]
    )
    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", lambda *a, **k: resp)
    assert OllamaProvider(model="x").generate("prompt") == "hello world"


def test_ollama_provider_streams_fragments_and_stops_at_done(monkeypatch) -> None:
    resp = _FakeNDJSONResp(
        [
            {"response": "a", "done": False},
            {"response": "b", "done": False},
            {"response": "", "done": True},
            {"response": "IGNORED-AFTER-DONE", "done": False},
        ]
    )
    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", lambda *a, **k: resp)
    assert list(OllamaProvider(model="x").generate_stream("prompt")) == ["a", "b"]


def test_ollama_provider_stream_payload_requests_streaming(monkeypatch) -> None:
    captured: dict = {}

    def _fake_urlopen(request, timeout=None):
        captured.update(json.loads(request.data.decode("utf-8")))
        return _FakeNDJSONResp([{"response": "ok", "done": True}])

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", _fake_urlopen)
    OllamaProvider(model="x").generate("prompt")
    assert captured["stream"] is True


def test_ollama_provider_stream_raises_on_error_line(monkeypatch) -> None:
    resp = _FakeNDJSONResp([{"error": "model not found"}])
    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", lambda *a, **k: resp)
    with pytest.raises(LLMUnavailable, match="model not found"):
        list(OllamaProvider(model="x").generate_stream("prompt"))


def test_ollama_provider_raises_llm_unavailable(monkeypatch) -> None:
    def _boom(*a, **k):
        raise provider_mod.urllib.error.URLError("connection refused")

    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", _boom)
    with pytest.raises(LLMUnavailable):
        OllamaProvider(model="x").generate("prompt")


def test_run_generation_eval_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    conn = connect(tmp_path / "memory.db")
    init_db(conn)
    cur = conn.execute(
        "INSERT INTO sources (source_type, source_uri, timestamp, title) VALUES (?, ?, ?, ?)",
        ("note", "/abs/parser.md", "2026-06-01T00:00:00+00:00", "parser"),
    )
    sid = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
        (sid, 0, "parser bounds check fix details"),
    )
    # Answerable query (token overlap -> retrieved) and an unanswerable one (no overlap).
    conn.execute(
        "INSERT INTO queries_eval (query_text, expected_source_uris) VALUES (?, ?)",
        ("[test] parser bounds check", json.dumps(["/abs/parser.md"])),
    )
    conn.execute(
        "INSERT INTO queries_eval (query_text, expected_source_uris) VALUES (?, ?)",
        ("[test] zzzz unrelated tokyo revenue", json.dumps([])),
    )
    conn.commit()

    provider = StubLLMProvider(output="Grounded answer [E1].")
    summary = run_generation_eval(conn, provider, top_k=5, query_prefix="[test]")

    assert summary.query_count == 2
    assert summary.level == "evidence"            # default flat retrieval
    assert summary.citation_validity == 1.0       # no hallucinated ids
    assert summary.source_grounding_hit == 1.0    # answerable cites the expected source
    assert summary.source_coverage == 1.0         # cites the single expected source
    assert summary.abstention_correct == 1.0      # answered the answerable, abstained the other


def _seed_event(conn, title: str, uri: str) -> tuple[int, int]:
    cur = conn.execute(
        "INSERT INTO sources (source_type, source_uri, timestamp, title) VALUES (?, ?, ?, ?)",
        ("note", uri, "2026-06-01T00:00:00+00:00", title),
    )
    sid = int(cur.lastrowid)
    cur = conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, 0, ?)",
        (sid, title),
    )
    chunk_id = int(cur.lastrowid)
    event_id = insert_node(conn, level=1, node_type="event", title=title)
    add_edge(conn, 1, event_id, 0, chunk_id, "derived_from")
    return event_id, chunk_id


def test_source_coverage_partial_when_one_of_two_expected_cited(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    conn = connect(tmp_path / "memory.db")
    init_db(conn)
    _seed_event(conn, "parser bounds check fix", "/abs/a.md")
    _seed_event(conn, "parser bounds check guard", "/abs/b.md")
    # Two-source gold; the stub answer cites only one evidence id -> coverage 0.5.
    conn.execute(
        "INSERT INTO queries_eval (query_text, expected_source_uris) VALUES (?, ?)",
        ("[t] parser bounds", json.dumps(["/abs/a.md", "/abs/b.md"])),
    )
    conn.commit()

    summary = run_generation_eval(conn, StubLLMProvider(output="Answer [E1]."), query_prefix="[t]")
    assert summary.source_grounding_hit == 1.0     # cited at least one expected source
    assert summary.source_coverage == 0.5          # but only one of the two


def test_run_generation_eval_concept_level_grounds_in_l0(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    conn = connect(tmp_path / "memory.db")
    init_db(conn)
    event_id, _chunk_id = _seed_event(conn, "fingerprint deterministic ingestion", "/abs/ingest.md")
    concept = insert_node(conn, level=3, node_type="concept", title="fingerprint deterministic ingestion")
    add_edge(conn, 3, concept, 1, event_id, "contains")
    conn.execute(
        "INSERT INTO queries_eval (query_text, expected_source_uris) VALUES (?, ?)",
        ("[t] fingerprint ingestion", json.dumps(["/abs/ingest.md"])),
    )
    conn.commit()

    # No node embeddings -> concept retrieval is lexical; drill-down still recovers L0,
    # and the synthesized answer cites that L0 chunk (provenance holds at concept level).
    provider = StubLLMProvider(output="Grounded at concept level [E1].")
    summary = run_generation_eval(conn, provider, query_prefix="[t]", level="concept")
    assert summary.level == "concept"
    assert summary.query_count == 1
    assert summary.source_grounding_hit == 1.0
    assert summary.source_coverage == 1.0
    assert provider.calls == 1


def test_concept_level_abstains_when_no_nodes_match(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    conn = connect(tmp_path / "memory.db")
    init_db(conn)
    _seed_event(conn, "unrelated topic", "/abs/x.md")
    conn.execute(
        "INSERT INTO queries_eval (query_text, expected_source_uris) VALUES (?, ?)",
        ("[t] zzz nonmatching query", json.dumps(["/abs/x.md"])),
    )
    conn.commit()

    # No concept nodes exist -> empty candidate set -> abstain, no LLM call, no crash.
    provider = StubLLMProvider(output="should not be used [E1].")
    summary = run_generation_eval(conn, provider, query_prefix="[t]", level="concept")
    assert summary.query_count == 1
    assert summary.results[0].abstained is True
    assert provider.calls == 0


def test_ask_falls_back_to_template_when_llm_unavailable(tmp_path, monkeypatch, capsys) -> None:
    from crossmodalrag import cli

    db = tmp_path / "memory.db"
    monkeypatch.setenv("CMRAG_DB_PATH", str(db))
    conn = connect(db)
    init_db(conn)
    cur = conn.execute(
        "INSERT INTO sources (source_type, source_uri, timestamp, title) VALUES (?, ?, ?, ?)",
        ("note", "/abs/x.md", "2026-06-01T00:00:00+00:00", "x"),
    )
    conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
        (int(cur.lastrowid), 0, "fingerprint deterministic ingestion"),
    )
    conn.commit()
    conn.close()

    class _Raising:
        name = "raise-llm"

        def generate(self, prompt, system=None):
            raise LLMUnavailable("down")

    monkeypatch.setattr(cli, "get_default_llm_provider", lambda: _Raising())
    cli.ask_cmd("fingerprint", top_k=3)
    out = capsys.readouterr()
    assert "Evidence-grounded findings" in out.out
    assert "falling back" in out.err


# --- abstention_reason: gate vs model refusals must be distinguishable --------------------


def test_gate_abstention_reason_is_weak_retrieval() -> None:
    gen = synthesize_answer(
        "q", [_hit(1, "alpha", score=0.05)], StubLLMProvider(), min_evidence_score=0.15
    )
    assert gen.abstained
    assert gen.abstention_reason == "weak_retrieval"


def test_no_hits_abstention_reason_is_weak_retrieval() -> None:
    gen = synthesize_answer("q", [], StubLLMProvider())
    assert gen.abstention_reason == "weak_retrieval"


def test_llm_abstention_reason_is_llm_insufficient() -> None:
    provider = StubLLMProvider(output=INSUFFICIENT_EVIDENCE_TEXT)
    gen = synthesize_answer("q", [_hit(1, "alpha")], provider, min_evidence_score=0.0)
    assert gen.abstained
    assert provider.calls == 1  # the model was consulted and refused
    assert gen.abstention_reason == "llm_insufficient"


def test_answered_query_has_no_abstention_reason() -> None:
    gen = synthesize_answer("q", [_hit(1, "alpha")], StubLLMProvider(), min_evidence_score=0.0)
    assert not gen.abstained
    assert gen.abstention_reason is None


def test_abstention_reason_in_json_contract() -> None:
    provider = StubLLMProvider(output=INSUFFICIENT_EVIDENCE_TEXT)
    gen = synthesize_answer("q", [_hit(1, "alpha")], provider, min_evidence_score=0.0)
    data = generated_answer_to_dict(gen)
    assert data["abstention_reason"] == "llm_insufficient"
    answered = synthesize_answer("q", [_hit(1, "alpha")], StubLLMProvider(), min_evidence_score=0.0)
    assert generated_answer_to_dict(answered)["abstention_reason"] is None


def test_abstention_status_line_shows_reason_and_top_score() -> None:
    provider = StubLLMProvider(output=INSUFFICIENT_EVIDENCE_TEXT)
    gen = synthesize_answer("q", [_hit(1, "alpha", score=0.9)], provider, min_evidence_score=0.0)
    text = format_generated_answer(gen)
    assert "abstained (llm_insufficient; top retrieval score 0.900)" in text


def test_partial_answer_instruction_present_in_system_prompt() -> None:
    # The Q6-class fix: the prompt must offer a partial-answer middle path and
    # reserve the exact-sentence refusal for the no-relevant-evidence case only.
    from crossmodalrag.generate.synthesize import SYSTEM_PROMPT

    assert "PARTIALLY" in SYSTEM_PROMPT
    assert "NONE" in SYSTEM_PROMPT
    assert INSUFFICIENT_EVIDENCE_TEXT in SYSTEM_PROMPT


# --- Gap D: breadth synthesis -----------------------------------------------------------


def test_breadth_instruction_present_in_system_prompt() -> None:
    # Multi-source synthesis: the prompt must direct the model to integrate every
    # materially relevant item and multi-cite agreeing sources, not pad citations.
    from crossmodalrag.generate.synthesize import SYSTEM_PROMPT

    assert "EVERY evidence item" in SYSTEM_PROMPT
    assert "[E1][E3]" in SYSTEM_PROMPT
    assert "Never cite an item that did not contribute" in SYSTEM_PROMPT


# --- Synthesis rebalancing: experience-aware answers ------------------------------------


def test_experience_aware_instruction_present_in_system_prompt() -> None:
    # Direct, experience-aware synthesis: general knowledge is permitted but
    # subordinate — never conflated with the user's records, never a substitute
    # for evidence, and never grounds to answer an unanswerable query.
    from crossmodalrag.generate.synthesize import SYSTEM_PROMPT

    assert "authoritative record" in SYSTEM_PROMPT
    assert "never invent personal history" in SYSTEM_PROMPT
    # Knowledge can answer conceptual sub-questions but never fabricates history:
    assert "conceptual parts of the question" in SYSTEM_PROMPT
    assert "General knowledge alone never justifies a claim about the user's history" in SYSTEM_PROMPT
    # The boxing-in clause must stay gone (the rebalancing itself):
    assert "Do not use any outside knowledge" not in SYSTEM_PROMPT
    # Evidence-derived claims still carry mandatory citation:
    assert "MUST cite" in SYSTEM_PROMPT


def test_source_coverage_counts_distinct_sources_not_citations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    conn = connect(tmp_path / "memory.db")
    init_db(conn)
    # Two chunks from the SAME source: citing both must count the source once.
    cur = conn.execute(
        "INSERT INTO sources (source_type, source_uri, timestamp, title) VALUES (?, ?, ?, ?)",
        ("note", "/abs/a.md", "2026-06-01T00:00:00+00:00", "parser"),
    )
    sid = int(cur.lastrowid)
    for i in range(2):
        conn.execute(
            "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
            (sid, i, f"parser bounds check part {i}"),
        )
    conn.execute(
        "INSERT INTO queries_eval (query_text, expected_source_uris) VALUES (?, ?)",
        ("[t2] parser bounds", json.dumps(["/abs/a.md", "/abs/b.md"])),
    )
    conn.commit()

    # The stub cites two evidence ids, but both resolve to /abs/a.md.
    summary = run_generation_eval(
        conn, StubLLMProvider(output="Answer [E1][E2]."), query_prefix="[t2]"
    )
    assert summary.source_coverage == 0.5  # one distinct gold source of two, not 2/2


# --- streaming: live token output must not change the grounded-answer contract ------------


class StreamingStubProvider:
    def __init__(self, fragments: list[str], name: str = "stub-stream") -> None:
        self.name = name
        self._fragments = fragments
        self.generate_calls = 0
        self.stream_calls = 0

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.generate_calls += 1
        return "".join(self._fragments).strip()

    def generate_stream(self, prompt: str, system: str | None = None):
        self.stream_calls += 1
        yield from self._fragments


def test_on_token_receives_fragments_and_answer_matches_joined_stream() -> None:
    fragments = ["Claim ", "one ", "[E1].", " Claim two ", "[E5]."]
    provider = StreamingStubProvider(fragments)
    seen: list[str] = []
    gen = synthesize_answer(
        "q", [_hit(1, "alpha"), _hit(2, "beta")], provider,
        min_evidence_score=0.0, on_token=seen.append,
    )
    assert seen == fragments
    assert provider.stream_calls == 1
    assert provider.generate_calls == 0
    assert gen.answer_text == "".join(fragments).strip()


def test_streamed_and_buffered_paths_produce_identical_answers() -> None:
    fragments = ["Claim one ", "[E1].", " Claim two ", "[E5]."]
    hits = [_hit(1, "alpha"), _hit(2, "beta")]
    streamed = synthesize_answer(
        "q", hits, StreamingStubProvider(fragments), min_evidence_score=0.0,
        on_token=lambda _fragment: None,
    )
    buffered = synthesize_answer(
        "q", hits, StubLLMProvider(output="".join(fragments)), min_evidence_score=0.0
    )
    assert streamed.answer_text == buffered.answer_text
    assert streamed.cited_evidence_ids == buffered.cited_evidence_ids == ["E1"]
    assert streamed.invalid_citations == buffered.invalid_citations == ["E5"]
    assert streamed.abstained == buffered.abstained is False


def test_streamed_llm_abstention_still_detected() -> None:
    # The exact-sentence refusal arrives in pieces; abstention is judged on the whole.
    midpoint = len(INSUFFICIENT_EVIDENCE_TEXT) // 2
    fragments = [INSUFFICIENT_EVIDENCE_TEXT[:midpoint], INSUFFICIENT_EVIDENCE_TEXT[midpoint:]]
    gen = synthesize_answer(
        "q", [_hit(1, "alpha")], StreamingStubProvider(fragments),
        min_evidence_score=0.0, on_token=lambda _fragment: None,
    )
    assert gen.abstained
    assert gen.abstention_reason == "llm_insufficient"


def test_on_token_not_called_when_gate_abstains() -> None:
    provider = StreamingStubProvider(["should not stream"])
    seen: list[str] = []
    gen = synthesize_answer(
        "q", [_hit(1, "alpha", score=0.05)], provider,
        min_evidence_score=0.15, on_token=seen.append,
    )
    assert gen.abstained
    assert seen == []
    assert provider.stream_calls == 0


def test_llm_unavailable_propagates_mid_stream() -> None:
    class _FailsMidStream:
        name = "fail-stream"

        def generate(self, prompt: str, system: str | None = None) -> str:
            return ""

        def generate_stream(self, prompt: str, system: str | None = None):
            yield "partial "
            raise LLMUnavailable("connection dropped mid-stream")

    seen: list[str] = []
    with pytest.raises(LLMUnavailable, match="mid-stream"):
        synthesize_answer(
            "q", [_hit(1, "alpha")], _FailsMidStream(),
            min_evidence_score=0.0, on_token=seen.append,
        )
    assert seen == ["partial "]  # fragments before the failure were observable


def _seed_ask_db(tmp_path, monkeypatch) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("CMRAG_DB_PATH", str(db))
    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    conn = connect(db)
    init_db(conn)
    cur = conn.execute(
        "INSERT INTO sources (source_type, source_uri, timestamp, title) VALUES (?, ?, ?, ?)",
        ("note", "/abs/x.md", "2026-06-01T00:00:00+00:00", "x"),
    )
    conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
        (int(cur.lastrowid), 0, "fingerprint deterministic ingestion"),
    )
    conn.commit()
    conn.close()


def test_ask_streams_tokens_on_tty_with_footer(tmp_path, monkeypatch, capsys) -> None:
    import sys as _sys

    from crossmodalrag import cli

    _seed_ask_db(tmp_path, monkeypatch)
    provider = StreamingStubProvider(["Grounded ", "answer [E1]."])
    monkeypatch.setattr(cli, "get_default_llm_provider", lambda: provider)
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: True)

    cli.ask_cmd("fingerprint", top_k=3)
    out = capsys.readouterr().out

    assert provider.stream_calls == 1  # streamed, not buffered
    assert "Answer:\nGrounded answer [E1]." in out
    assert out.count("Answer:") == 1  # answer text not re-printed after streaming
    assert "Evidence:" in out  # provenance footer still rendered
    assert "chunk_id=" in out


def test_ask_no_stream_flag_uses_buffered_path(tmp_path, monkeypatch, capsys) -> None:
    import sys as _sys

    from crossmodalrag import cli

    _seed_ask_db(tmp_path, monkeypatch)
    provider = StreamingStubProvider(["Grounded ", "answer [E1]."])
    monkeypatch.setattr(cli, "get_default_llm_provider", lambda: provider)
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: True)

    cli.ask_cmd("fingerprint", top_k=3, stream=False)
    out = capsys.readouterr().out

    assert provider.stream_calls == 0
    assert provider.generate_calls == 1
    assert "Grounded answer [E1]." in out
    assert "Evidence:" in out


def test_ask_does_not_stream_when_stdout_is_not_a_tty(tmp_path, monkeypatch, capsys) -> None:
    from crossmodalrag import cli

    _seed_ask_db(tmp_path, monkeypatch)
    provider = StreamingStubProvider(["Grounded ", "answer [E1]."])
    monkeypatch.setattr(cli, "get_default_llm_provider", lambda: provider)
    # capsys stdout is not a TTY -> piped/redirected output stays buffered.

    cli.ask_cmd("fingerprint", top_k=3)
    out = capsys.readouterr().out

    assert provider.stream_calls == 0
    assert provider.generate_calls == 1
    assert "Grounded answer [E1]." in out


def test_ask_json_stays_buffered_even_on_tty(tmp_path, monkeypatch, capsys) -> None:
    import sys as _sys

    from crossmodalrag import cli

    _seed_ask_db(tmp_path, monkeypatch)
    provider = StreamingStubProvider(["Grounded ", "answer [E1]."])
    monkeypatch.setattr(cli, "get_default_llm_provider", lambda: provider)
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: True)

    cli.ask_cmd("fingerprint", top_k=3, as_json=True)
    out = capsys.readouterr().out

    assert provider.stream_calls == 0
    data = json.loads(out)  # a single valid JSON document, no interleaved tokens
    assert data["answer"] == "Grounded answer [E1]."


def test_synthesize_answer_stream_returns_answer_as_generator_value() -> None:
    fragments = ["Claim ", "[E1]."]
    from crossmodalrag.generate.synthesize import synthesize_answer_stream

    stream = synthesize_answer_stream(
        "q", [_hit(1, "alpha")], StreamingStubProvider(fragments), min_evidence_score=0.0
    )
    seen: list[str] = []
    while True:
        try:
            seen.append(next(stream))
        except StopIteration as stop:
            gen = stop.value
            break
    assert seen == fragments
    assert gen.answer_text == "Claim [E1]."
    assert gen.cited_evidence_ids == ["E1"]


# --- service layer: answer_stream_events (the /ask/stream backbone) -----------------------


def _service_conn(tmp_path):
    conn = connect(tmp_path / "memory.db")
    init_db(conn)
    cur = conn.execute(
        "INSERT INTO sources (source_type, source_uri, timestamp, title) VALUES (?, ?, ?, ?)",
        ("note", "/abs/x.md", "2026-06-01T00:00:00+00:00", "x"),
    )
    conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
        (int(cur.lastrowid), 0, "fingerprint deterministic ingestion"),
    )
    conn.commit()
    return conn


def test_answer_stream_events_tokens_then_final_answer(tmp_path, monkeypatch) -> None:
    import crossmodalrag.service as svc

    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    monkeypatch.setattr(
        svc, "get_default_llm_provider",
        lambda: StreamingStubProvider(["Grounded ", "answer [E1]."]),
    )
    conn = _service_conn(tmp_path)
    events = list(svc.answer_stream_events(conn, query="fingerprint"))
    conn.close()

    assert [e["type"] for e in events] == ["token", "token", "answer"]
    assert "".join(e["text"] for e in events[:-1]) == "Grounded answer [E1]."
    final = events[-1]["data"]
    assert final["answer"] == "Grounded answer [E1]."
    assert final["cited_evidence_ids"] == ["E1"]
    assert final["evidence"]  # provenance present, same contract as answer_payload


def test_answer_stream_events_final_data_matches_answer_payload(tmp_path, monkeypatch) -> None:
    import crossmodalrag.service as svc

    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    monkeypatch.setattr(
        svc, "get_default_llm_provider",
        lambda: StreamingStubProvider(["Grounded ", "answer [E1]."]),
    )
    conn = _service_conn(tmp_path)
    streamed = list(svc.answer_stream_events(conn, query="fingerprint"))[-1]["data"]
    buffered = svc.answer_payload(conn, query="fingerprint")
    conn.close()

    streamed.pop("timing"), buffered.pop("timing")  # wall-clock differs per call
    assert streamed == buffered  # identical contract, streaming is display-only


def test_answer_stream_events_no_llm_emits_only_final_template(tmp_path, monkeypatch) -> None:
    import crossmodalrag.service as svc

    conn = _service_conn(tmp_path)
    events = list(svc.answer_stream_events(conn, query="fingerprint", use_llm=False))
    conn.close()

    assert [e["type"] for e in events] == ["answer"]
    assert events[0]["data"]["model"] is None  # deterministic template path


def test_answer_stream_events_llm_failure_falls_back_to_template(tmp_path, monkeypatch) -> None:
    import crossmodalrag.service as svc

    class _FailsMidStream:
        name = "fail-stream"

        def generate(self, prompt, system=None):
            raise LLMUnavailable("down")

        def generate_stream(self, prompt, system=None):
            yield "partial "
            raise LLMUnavailable("connection dropped mid-stream")

    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    monkeypatch.setattr(svc, "get_default_llm_provider", lambda: _FailsMidStream())
    conn = _service_conn(tmp_path)
    events = list(svc.answer_stream_events(conn, query="fingerprint"))
    conn.close()

    # Tokens before the failure are emitted, and the stream STILL ends with a
    # final answer event (the template fallback) so consumers never hang.
    assert [e["type"] for e in events] == ["token", "answer"]
    assert events[-1]["data"]["model"] is None


def test_stream_answer_events_works_after_retrieval_conn_closed(tmp_path, monkeypatch) -> None:
    """The API consumes the stream long after the request-scoped conn is gone."""
    import crossmodalrag.service as svc

    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    monkeypatch.setattr(
        svc, "get_default_llm_provider",
        lambda: StreamingStubProvider(["Grounded ", "answer [E1]."]),
    )
    conn = _service_conn(tmp_path)
    hits, matched_nodes = svc.retrieve_for_answer(conn, query="fingerprint")
    conn.close()  # DB gone before a single event is consumed

    events = list(
        svc.stream_answer_events(query="fingerprint", hits=hits, matched_nodes=matched_nodes)
    )
    assert [e["type"] for e in events] == ["token", "token", "answer"]
    assert events[-1]["data"]["answer"] == "Grounded answer [E1]."


def test_stream_answer_events_close_from_another_thread_is_clean(tmp_path, monkeypatch) -> None:
    """Client-disconnect regression: ASGI servers close abandoned response generators
    from arbitrary worker threads. Holding a thread-bound sqlite connection inside the
    generator died there with sqlite3.ProgrammingError ('Exception ignored in ...')."""
    import threading

    import crossmodalrag.service as svc

    monkeypatch.setenv("CMRAG_MIN_EVIDENCE_SCORE", "0.0")
    monkeypatch.setattr(
        svc, "get_default_llm_provider",
        lambda: StreamingStubProvider(["a", "b", "c", "d [E1]."]),
    )
    conn = _service_conn(tmp_path)
    hits, matched_nodes = svc.retrieve_for_answer(conn, query="fingerprint")
    conn.close()

    events = svc.stream_answer_events(query="fingerprint", hits=hits, matched_nodes=matched_nodes)
    assert next(events)["type"] == "token"  # mid-stream, like a user hitting Stop

    errors: list[BaseException] = []

    def _close() -> None:
        try:
            events.close()
        except BaseException as exc:  # the regression IS "close raised"
            errors.append(exc)

    t = threading.Thread(target=_close)
    t.start()
    t.join()
    assert errors == []


def test_streamed_output_matches_buffered_output_when_answered(tmp_path, monkeypatch, capsys) -> None:
    import sys as _sys

    from crossmodalrag import cli

    _seed_ask_db(tmp_path, monkeypatch)
    monkeypatch.setattr(_sys.stdout, "isatty", lambda: True)

    monkeypatch.setattr(
        cli, "get_default_llm_provider",
        lambda: StreamingStubProvider(["Grounded ", "answer [E1]."]),
    )
    cli.ask_cmd("fingerprint", top_k=3)
    streamed = capsys.readouterr().out

    monkeypatch.setattr(
        cli, "get_default_llm_provider",
        lambda: StreamingStubProvider(["Grounded ", "answer [E1]."]),
    )
    cli.ask_cmd("fingerprint", top_k=3, stream=False)
    buffered = capsys.readouterr().out

    assert streamed == buffered  # rendering parity: streaming is display-only
