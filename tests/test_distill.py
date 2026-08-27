from __future__ import annotations

import hashlib
import json
import re

import pytest

from crossmodalrag.db import connect, init_db
from crossmodalrag.embed.store import embed_pending_chunks, pack_vector
from crossmodalrag.evaluation import (
    EvalSummary,
    distill_gate_fires,
    distilled_compression_ratio,
    run_distilled_eval,
)
from crossmodalrag.memory.distill import (
    build_distilled,
    distilled_summaries,
    distilled_summary_to_dict,
)
from crossmodalrag.memory.store import add_edge, resolve_to_evidence
from crossmodalrag.retrieve.distilled import distilled_drilldown_source_uris, retrieve_distilled

WORD_RE = re.compile(r"[a-z0-9]+")


class StubEmbedProvider:
    def __init__(self, dim: int = 64, name: str = "stub-embed-v1") -> None:
        self.dim = dim
        self.name = name

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in WORD_RE.findall(text.lower()):
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            out.append(vec)
        return out


class StubLLMProvider:
    def __init__(self, label: str = "a faithful distilled summary", name: str = "stub-llm") -> None:
        self.name = name
        self._label = label
        self.calls = 0

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        return self._label


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "memory.db")
    init_db(connection)
    yield connection
    connection.close()


def _add_event_with_chunk(conn, text: str) -> tuple[int, int, str]:
    uri = f"/v/{text.replace(' ', '_')}.md"
    cur = conn.execute("INSERT INTO sources (source_type, source_uri) VALUES ('note', ?)", (uri,))
    sid = int(cur.lastrowid)
    cur = conn.execute(
        "INSERT INTO evidence_chunks (source_id, chunk_index, chunk_text) VALUES (?, 0, ?)", (sid, text)
    )
    chunk_id = int(cur.lastrowid)
    cur = conn.execute(
        "INSERT INTO memory_nodes (level, node_type, title) VALUES (1, 'event', ?)", (text,)
    )
    event_id = int(cur.lastrowid)
    add_edge(conn, 1, event_id, 0, chunk_id, "derived_from")
    conn.commit()
    return event_id, chunk_id, uri


def _make_concept(conn, title: str, member_ids: list[int], content: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO memory_nodes (level, node_type, title, content) VALUES (3, 'concept', ?, ?)",
        (title, content),
    )
    concept_id = int(cur.lastrowid)
    for mid in member_ids:
        add_edge(conn, 3, concept_id, 1, mid, "contains")
    conn.commit()
    return concept_id


def _distilled_rows(conn):
    return [
        tuple(r)
        for r in conn.execute(
            "SELECT node_id, level, summary, core_evidence_json, derivation_fingerprint FROM distilled_nodes "
            "ORDER BY node_id"
        ).fetchall()
    ]


# --- build_distilled: core-evidence is a real, sized L0 subset --------------------------


def test_core_evidence_is_real_subset_sized_to_ratio(conn):
    members = [_add_event_with_chunk(conn, f"alpha topic chunk {i}") for i in range(4)]
    cid = _make_concept(conn, "alpha concept", [e for e, _, _ in members])
    embed_pending_chunks(conn, StubEmbedProvider())

    build_distilled(conn, StubEmbedProvider(), StubLLMProvider(), target_ratio=0.5)

    full = set(resolve_to_evidence(conn, 3, cid))
    row = conn.execute("SELECT core_evidence_json FROM distilled_nodes WHERE node_id = ?", (cid,)).fetchone()
    core = set(json.loads(row["core_evidence_json"]))
    assert core <= full  # a real subset, never invented
    assert len(core) == 2  # round(0.5 * 4)


def test_distilled_drilldown_is_subset_of_full(conn):
    members = [_add_event_with_chunk(conn, f"beta topic chunk {i}") for i in range(4)]
    _make_concept(conn, "beta concept", [e for e, _, _ in members])
    embed_pending_chunks(conn, StubEmbedProvider())
    build_distilled(conn, StubEmbedProvider(), StubLLMProvider(), target_ratio=0.5)

    hit = retrieve_distilled(conn, "beta topic", level="concept", top_k=5, provider=StubEmbedProvider())
    distilled_uris = set(distilled_drilldown_source_uris(conn, hit))

    full_uris = {uri for _, _, uri in members}
    assert distilled_uris  # something retrieved + grounded
    assert distilled_uris <= full_uris  # never a source outside the concept's evidence


# --- determinism + reconcile ------------------------------------------------------------


def test_rebuild_is_a_noop(conn):
    members = [_add_event_with_chunk(conn, f"gamma chunk {i}") for i in range(4)]
    _make_concept(conn, "gamma concept", [e for e, _, _ in members])
    embed_pending_chunks(conn, StubEmbedProvider())

    llm = StubLLMProvider()
    first = build_distilled(conn, StubEmbedProvider(), llm, target_ratio=0.5)
    before = _distilled_rows(conn)
    calls_after_first = llm.calls
    second = build_distilled(conn, StubEmbedProvider(), llm, target_ratio=0.5)
    after = _distilled_rows(conn)

    assert first.nodes_distilled == 1
    assert second.nodes_distilled == 0 and second.nodes_kept == 1
    assert llm.calls == calls_after_first  # matched node not re-summarized
    assert after == before


def test_changing_ratio_rederives_with_smaller_core(conn):
    members = [_add_event_with_chunk(conn, f"delta chunk {i}") for i in range(4)]
    cid = _make_concept(conn, "delta concept", [e for e, _, _ in members])
    embed_pending_chunks(conn, StubEmbedProvider())

    build_distilled(conn, StubEmbedProvider(), StubLLMProvider(), target_ratio=0.5)
    result = build_distilled(conn, StubEmbedProvider(), StubLLMProvider(), target_ratio=0.25)

    assert result.nodes_distilled == 1  # re-derived (fingerprint includes the ratio)
    row = conn.execute("SELECT core_evidence_json FROM distilled_nodes WHERE node_id = ?", (cid,)).fetchone()
    assert len(json.loads(row["core_evidence_json"])) == 1  # round(0.25 * 4)


# --- fallback summary -------------------------------------------------------------------


def test_fallback_summary_without_llm(conn):
    members = [_add_event_with_chunk(conn, f"epsilon chunk {i}") for i in range(2)]
    cid = _make_concept(conn, "epsilon concept", [e for e, _, _ in members], content="epsilon digest")
    embed_pending_chunks(conn, StubEmbedProvider())

    result = build_distilled(conn, StubEmbedProvider(), None, target_ratio=0.5)
    assert result.named_by_fallback == 1 and result.named_by_llm == 0
    row = conn.execute("SELECT summary FROM distilled_nodes WHERE node_id = ?", (cid,)).fetchone()
    assert row["summary"] == "epsilon digest"  # deterministic fallback to node content


def test_llm_unavailable_falls_back(conn):
    from crossmodalrag.generate.provider import LLMUnavailable

    class DeadLLM:
        name = "dead"

        def generate(self, prompt, system=None):
            raise LLMUnavailable("ollama down")

    members = [_add_event_with_chunk(conn, f"zeta chunk {i}") for i in range(2)]
    _make_concept(conn, "zeta concept", [e for e, _, _ in members], content="zeta digest")
    embed_pending_chunks(conn, StubEmbedProvider())

    result = build_distilled(conn, StubEmbedProvider(), DeadLLM(), target_ratio=0.5)
    assert result.nodes_distilled == 1 and result.named_by_fallback == 1  # never crashes


def test_build_distilled_only_writes_distilled_nodes(conn):
    members = [_add_event_with_chunk(conn, f"eta chunk {i}") for i in range(2)]
    _make_concept(conn, "eta concept", [e for e, _, _ in members])
    embed_pending_chunks(conn, StubEmbedProvider())

    def _mem_snapshot():
        nodes = conn.execute("SELECT id, level, node_type, title FROM memory_nodes ORDER BY id").fetchall()
        edges = conn.execute(
            "SELECT parent_level, parent_id, child_level, child_id, relation FROM memory_edges ORDER BY id"
        ).fetchall()
        return [tuple(n) for n in nodes], [tuple(e) for e in edges]

    before = _mem_snapshot()
    build_distilled(conn, StubEmbedProvider(), StubLLMProvider(), target_ratio=0.5)
    assert _mem_snapshot() == before  # additive/separable: memory_nodes/edges untouched


# --- gate measurement (run_distilled_eval + ratio + gate) -------------------------------


def _seed_distilled_concept(conn, *, query_tokens: str, gold_chunk_id: int, core_chunk_ids: list[int]):
    """Insert a concept node + a distilled row directly so core membership is fully controlled."""
    cur = conn.execute(
        "INSERT INTO memory_nodes (level, node_type, title, content) VALUES (3, 'concept', ?, ?)",
        (query_tokens, query_tokens),
    )
    cid = int(cur.lastrowid)
    conn.execute(
        """
        INSERT INTO distilled_nodes (node_id, level, summary, model, prompt_version, dim, vector,
                                     core_evidence_json, derivation_fingerprint, confidence)
        VALUES (?, 3, ?, 'stub-embed-v1', 'distill-summary-v1', 1, ?, ?, 'fp', 1.0)
        """,
        (cid, query_tokens, pack_vector([0.0]), json.dumps(sorted(core_chunk_ids))),
    )
    conn.commit()
    return cid


def test_run_distilled_eval_recall_preserved_when_core_keeps_gold(conn):
    e, gold_chunk, gold_uri = _add_event_with_chunk(conn, "rate limit spec answer")
    _seed_distilled_concept(conn, query_tokens="rate limit spec", gold_chunk_id=gold_chunk, core_chunk_ids=[gold_chunk])
    conn.execute(
        "INSERT INTO queries_eval (query_text, expected_source_uris) VALUES (?, ?)",
        ("[t] what is the rate limit spec", json.dumps([gold_uri])),
    )
    conn.commit()

    summary = run_distilled_eval(conn, top_k=5, query_prefix="[t]", level="concept")
    assert summary.recall_at_k == pytest.approx(1.0)  # lexical match on summary -> drills to gold


def test_run_distilled_eval_recall_lost_when_core_drops_gold(conn):
    _e1, gold_chunk, gold_uri = _add_event_with_chunk(conn, "rate limit spec answer")
    _e2, other_chunk, _other_uri = _add_event_with_chunk(conn, "unrelated filler material")
    # Core excludes the gold chunk -> drill-down cannot reach the gold source.
    _seed_distilled_concept(conn, query_tokens="rate limit spec", gold_chunk_id=gold_chunk, core_chunk_ids=[other_chunk])
    conn.execute(
        "INSERT INTO queries_eval (query_text, expected_source_uris) VALUES (?, ?)",
        ("[t] what is the rate limit spec", json.dumps([gold_uri])),
    )
    conn.commit()

    summary = run_distilled_eval(conn, top_k=5, query_prefix="[t]", level="concept")
    assert summary.recall_at_k == pytest.approx(0.0)


def test_distilled_compression_ratio(conn):
    members = [_add_event_with_chunk(conn, f"theta chunk {i}") for i in range(4)]
    _make_concept(conn, "theta concept", [e for e, _, _ in members])
    embed_pending_chunks(conn, StubEmbedProvider())
    build_distilled(conn, StubEmbedProvider(), StubLLMProvider(), target_ratio=0.5)

    # 2 core chunks kept out of 4 full -> 0.5.
    assert distilled_compression_ratio(conn, level="concept") == pytest.approx(0.5)


def test_gate_fires_when_preserved_and_compressed(conn):
    # Recall preserved (delta 0) and a real compression ratio -> adopt.
    full = EvalSummary(query_count=1, top_k=5, recall_at_k=1.0, mrr_at_k=1.0, citation_hit_rate=1.0, results=[])
    distilled = EvalSummary(query_count=1, top_k=5, recall_at_k=1.0, mrr_at_k=1.0, citation_hit_rate=1.0, results=[])
    assert distill_gate_fires(full, distilled, compression_ratio=0.5) is True
    # Same recall but no compression -> hold.
    assert distill_gate_fires(full, distilled, compression_ratio=0.9) is False


# --- read view + JSON contract ----------------------------------------------------------

DISTILL_JSON_KEYS = {
    "node_id",
    "level",
    "node_type",
    "title",
    "summary",
    "core_count",
    "full_count",
    "compression_ratio",
    "confidence",
    "evidence_source_uri",
}


def test_distilled_summaries_report_compression_and_grounding(conn):
    members = [_add_event_with_chunk(conn, f"iota chunk {i}") for i in range(4)]
    cid = _make_concept(conn, "iota concept", [e for e, _, _ in members])
    embed_pending_chunks(conn, StubEmbedProvider())
    build_distilled(conn, StubEmbedProvider(), StubLLMProvider(), target_ratio=0.5)

    summaries = distilled_summaries(conn)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.node_id == cid
    assert s.core_count <= s.full_count
    assert s.full_count == 4 and s.core_count == 2
    assert s.compression_ratio == pytest.approx(s.core_count / s.full_count)
    assert s.evidence_source_uri in {uri for _, _, uri in members}  # grounded to a real source


def test_distilled_summary_to_dict_is_stable_contract(conn):
    members = [_add_event_with_chunk(conn, f"kappa chunk {i}") for i in range(2)]
    _make_concept(conn, "kappa concept", [e for e, _, _ in members])
    embed_pending_chunks(conn, StubEmbedProvider())
    build_distilled(conn, StubEmbedProvider(), StubLLMProvider(), target_ratio=0.5)

    payload = distilled_summary_to_dict(distilled_summaries(conn)[0])
    assert set(payload) == DISTILL_JSON_KEYS  # additive-only contract


def test_distilled_summaries_empty_when_nothing_distilled(conn):
    assert distilled_summaries(conn) == []
