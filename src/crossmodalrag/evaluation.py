from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from crossmodalrag.retrieve.hybrid import DEFAULT_PROFILE, retrieve


@dataclass(frozen=True)
class EvalQuery:
    id: int | None
    query_text: str
    expected_source_uris: list[str]


@dataclass(frozen=True)
class EvalQueryResult:
    query_text: str
    expected_source_uris: list[str]
    retrieved_source_uris: list[str]
    first_correct_rank: int | None
    recall_hit: bool
    citation_hit: bool


@dataclass(frozen=True)
class EvalSummary:
    query_count: int
    top_k: int
    recall_at_k: float
    mrr_at_k: float
    citation_hit_rate: float
    results: list[EvalQueryResult]


def eval_summary_to_dict(summary: "EvalSummary") -> dict:
    """Stable JSON contract for `mem eval --json`. Keep field names backward-compatible."""
    return {
        "query_count": summary.query_count,
        "top_k": summary.top_k,
        "recall_at_k": summary.recall_at_k,
        "mrr_at_k": summary.mrr_at_k,
        "citation_hit_rate": summary.citation_hit_rate,
        "misses": [r.query_text for r in summary.results if r.first_correct_rank is None],
    }


# The pre-committed native-embedding gate. Native (CLIP-class) image embeddings
# become justified when, measured against the OCR-text-first baseline, the
# visually-dominant slice trails the text-dominant slice on Recall@K by at least
# this much.
XMODAL_GATE_THRESHOLD = 0.30


def xmodal_gate_delta(text_summary: EvalSummary, visual_summary: EvalSummary) -> float:
    """Recall@K gap between the text-heavy and visual-heavy cross-modal slices."""
    return text_summary.recall_at_k - visual_summary.recall_at_k


def xmodal_gate_fires(
    text_summary: EvalSummary,
    visual_summary: EvalSummary,
    threshold: float = XMODAL_GATE_THRESHOLD,
) -> bool:
    """True when the OCR-text-first shortfall justifies the native-embedding spike."""
    return xmodal_gate_delta(text_summary, visual_summary) >= threshold


# distillation gate (pre-committed).
# A distilled (compact) representation of L2/L3 nodes is ADOPTED only when it preserves retrieval
# quality within EPSILON of the full nodes AND meets the size target. Semantics are INVERTED vs the
# xmodal gate: xmodal FIRES on a large gap (justifying a spike); distill FIRES (adopts) only when the
# recall loss is small enough AND compression is achieved. The authoritative reading is deferred to
# the later step that builds the distilled retrieval path — until then there is no distilled
# summary to compare, so the gate HOLDs (mirrors the xmodal "no signal yet" precedent).
DISTILL_GATE_EPSILON = 0.05            # max Recall@K the distilled path may lose vs full nodes
DISTILL_GATE_COMPRESSION_RATIO = 0.5   # target distilled-size / full-size (<= is good)


def distill_gate_delta(full_summary: EvalSummary, distilled_summary: EvalSummary) -> float:
    """Recall@K LOST by distilling: full minus distilled (>= 0 means the distilled path is worse)."""
    return full_summary.recall_at_k - distilled_summary.recall_at_k


def distill_gate_fires(
    full_summary: EvalSummary,
    distilled_summary: EvalSummary,
    *,
    compression_ratio: float,
    epsilon: float = DISTILL_GATE_EPSILON,
    target_ratio: float = DISTILL_GATE_COMPRESSION_RATIO,
) -> bool:
    """True when adoption is justified: recall preserved within ``epsilon`` AND size target met.

    ``compression_ratio`` is the achieved distilled-size / full-size (smaller is better). Both
    conditions must hold — a distillation that keeps recall but doesn't shrink, or shrinks but loses
    too much recall, does not fire.
    """
    return (
        distill_gate_delta(full_summary, distilled_summary) <= epsilon
        and compression_ratio <= target_ratio
    )


def run_eval(
    conn: sqlite3.Connection,
    *,
    top_k: int = 5,
    query_prefix: str | None = None,
    profile: str = DEFAULT_PROFILE,
    level: str = "evidence",
    restrict_source_types: set[str] | None = None,
    now=None,
) -> EvalSummary:
    queries = list_eval_queries(conn, query_prefix=query_prefix)
    results: list[EvalQueryResult] = []
    for query in queries:
        # Retrieval metrics need gold sources; abstain-only (negative) cases
        # carry no expected URIs and are evaluated by generation eval instead.
        if not query.expected_source_uris:
            continue
        if level == "evidence":
            hits = retrieve(
                conn,
                query=query.query_text,
                top_k=top_k,
                profile=profile,
                restrict_source_types=restrict_source_types,
                now=now,
            )
            retrieved_source_uris = _unique_source_uris_in_order([hit.source_uri for hit in hits])
        else:
            # Level-targeted retrieval + drill-down: recover L0 sources via the matched nodes.
            from crossmodalrag.retrieve.nodes import drilldown_source_uris, retrieve_nodes

            node_hits = retrieve_nodes(conn, query.query_text, level=level, top_k=top_k, profile=profile)
            retrieved_source_uris = drilldown_source_uris(conn, node_hits)
        first_correct_rank = _first_correct_rank(retrieved_source_uris, set(query.expected_source_uris))
        results.append(
            EvalQueryResult(
                query_text=query.query_text,
                expected_source_uris=query.expected_source_uris,
                retrieved_source_uris=retrieved_source_uris,
                first_correct_rank=first_correct_rank,
                recall_hit=first_correct_rank is not None and first_correct_rank <= top_k,
                # Approximation for pre-LLM phase: did the first cited/retrieved source match?
                citation_hit=first_correct_rank == 1,
            )
        )

    if not results:
        return EvalSummary(
            query_count=0,
            top_k=top_k,
            recall_at_k=0.0,
            mrr_at_k=0.0,
            citation_hit_rate=0.0,
            results=[],
        )

    query_count = len(results)
    recall = sum(1 for r in results if r.recall_hit) / query_count
    mrr = sum((1.0 / r.first_correct_rank) if r.first_correct_rank else 0.0 for r in results) / query_count
    citation_hit_rate = sum(1 for r in results if r.citation_hit) / query_count
    return EvalSummary(
        query_count=query_count,
        top_k=top_k,
        recall_at_k=recall,
        mrr_at_k=mrr,
        citation_hit_rate=citation_hit_rate,
        results=results,
    )


def run_distilled_eval(
    conn: sqlite3.Connection,
    *,
    top_k: int = 5,
    query_prefix: str | None = None,
    profile: str = DEFAULT_PROFILE,
    level: str = "concept",
    now=None,  # accepted for run_eval symmetry; distilled recency uses the wall clock
) -> EvalSummary:
    """Retrieval eval through the *distilled* stand-ins: rank distilled nodes, drill to their
    core-evidence subset, score against the gold sources. Same metric as ``run_eval`` so the two are
    directly comparable for the distillation gate."""
    from crossmodalrag.retrieve.distilled import distilled_drilldown_source_uris, retrieve_distilled

    queries = list_eval_queries(conn, query_prefix=query_prefix)
    results: list[EvalQueryResult] = []
    for query in queries:
        if not query.expected_source_uris:
            continue
        hits = retrieve_distilled(conn, query.query_text, level=level, top_k=top_k, profile=profile)
        retrieved_source_uris = distilled_drilldown_source_uris(conn, hits)
        first_correct_rank = _first_correct_rank(retrieved_source_uris, set(query.expected_source_uris))
        results.append(
            EvalQueryResult(
                query_text=query.query_text,
                expected_source_uris=query.expected_source_uris,
                retrieved_source_uris=retrieved_source_uris,
                first_correct_rank=first_correct_rank,
                recall_hit=first_correct_rank is not None and first_correct_rank <= top_k,
                citation_hit=first_correct_rank == 1,
            )
        )

    if not results:
        return EvalSummary(
            query_count=0, top_k=top_k, recall_at_k=0.0, mrr_at_k=0.0, citation_hit_rate=0.0, results=[]
        )
    query_count = len(results)
    recall = sum(1 for r in results if r.recall_hit) / query_count
    mrr = sum((1.0 / r.first_correct_rank) if r.first_correct_rank else 0.0 for r in results) / query_count
    citation_hit_rate = sum(1 for r in results if r.citation_hit) / query_count
    return EvalSummary(
        query_count=query_count,
        top_k=top_k,
        recall_at_k=recall,
        mrr_at_k=mrr,
        citation_hit_rate=citation_hit_rate,
        results=results,
    )


def distilled_compression_ratio(conn: sqlite3.Connection, *, level: str = "concept") -> float:
    """Evidence-footprint shrinkage of the distilled level: Σ|core_evidence| ÷ Σ|full evidence|.

    This is the "size" the distilled node stands in for (≤ target means real compression). Returns
    1.0 (no compression) when nothing has been distilled at the level."""
    from crossmodalrag.memory.store import resolve_to_evidence
    from crossmodalrag.retrieve.distilled import LEVEL_TO_NODE

    node_level, _ = LEVEL_TO_NODE[level]
    rows = conn.execute(
        "SELECT node_id, core_evidence_json FROM distilled_nodes WHERE level = ?", (node_level,)
    ).fetchall()
    if not rows:
        return 1.0
    core_total = full_total = 0
    for row in rows:
        core_total += len(json.loads(row["core_evidence_json"] or "[]"))
        full_total += len(resolve_to_evidence(conn, node_level, int(row["node_id"])))
    return (core_total / full_total) if full_total else 1.0


def list_eval_queries(
    conn: sqlite3.Connection,
    *,
    query_prefix: str | None = None,
) -> list[EvalQuery]:
    sql = """
        SELECT id, query_text, expected_source_uris
        FROM queries_eval
    """
    params: list[object] = []
    if query_prefix:
        sql += " WHERE query_text LIKE ?"
        params.append(f"{query_prefix}%")
    sql += " ORDER BY id ASC"
    rows = conn.execute(sql, tuple(params)).fetchall()
    return [
        EvalQuery(
            id=int(row["id"]),
            query_text=str(row["query_text"]),
            expected_source_uris=parse_expected_source_uris(row["expected_source_uris"]),
        )
        for row in rows
    ]


def load_eval_queries_file(path: Path) -> list[EvalQuery]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation query file must be a JSON list.")

    out: list[EvalQuery] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Eval row #{idx} must be an object.")
        query_text = str(item.get("query_text", "")).strip()
        if not query_text:
            raise ValueError(f"Eval row #{idx} is missing non-empty 'query_text'.")
        expected = item.get("expected_source_uris", [])
        if not isinstance(expected, list):
            raise ValueError(f"Eval row #{idx} field 'expected_source_uris' must be a list.")
        expected_uris = [str(x).strip() for x in expected if str(x).strip()]
        out.append(EvalQuery(id=None, query_text=query_text, expected_source_uris=expected_uris))
    return out


@dataclass(frozen=True)
class EvalQueryWarning:
    """A suspicious-but-not-fatal problem in a loaded eval query row.

    Malformed gold URIs never match a retrieved source, which silently zeroes
    grounding metrics — so loading warns instead of failing (a query file may
    legitimately predate ingestion of its sources).
    """

    row: int
    query_text: str
    uri: str
    issue: str  # "not_absolute" | "doubled_slash" | "unknown_source"


def validate_eval_queries(
    queries: list[EvalQuery],
    *,
    known_source_uris: set[str] | None = None,
) -> list[EvalQueryWarning]:
    """Check expected_source_uris for problems that would silently break URI matching.

    - ``doubled_slash``: an empty path segment (e.g. ``//Users/...``) — almost always a typo.
    - ``not_absolute``: neither an absolute path nor a ``scheme://`` URI.
    - ``unknown_source``: not ingested (checked only when ``known_source_uris`` is a
      non-empty set, so an empty/fresh DB doesn't flag every row). Git gold URIs use the
      ``<repo>@<sha>`` form, which matches the stored source URI exactly.
    """
    warnings: list[EvalQueryWarning] = []
    for row, query in enumerate(queries, start=1):
        for uri in query.expected_source_uris:
            issue = _uri_issue(uri)
            if issue is None and known_source_uris and uri not in known_source_uris:
                issue = "unknown_source"
            if issue is not None:
                warnings.append(
                    EvalQueryWarning(row=row, query_text=query.query_text, uri=uri, issue=issue)
                )
    return warnings


def _uri_issue(uri: str) -> str | None:
    scheme, sep, rest = uri.partition("://")
    path = rest if sep else uri
    if sep and not scheme:
        return "not_absolute"  # e.g. "://x" — no scheme
    if not sep and not uri.startswith("/"):
        return "not_absolute"
    if "//" in path:
        return "doubled_slash"
    return None


def upsert_eval_queries(conn: sqlite3.Connection, queries: list[EvalQuery]) -> int:
    for q in queries:
        expected_json = json.dumps(q.expected_source_uris)
        rows = conn.execute(
            """
            SELECT id, expected_source_uris
            FROM queries_eval
            WHERE query_text = ?
            ORDER BY id ASC
            """,
            (q.query_text,),
        ).fetchall()
        if not rows:
            conn.execute(
                """
                INSERT INTO queries_eval (query_text, expected_source_uris)
                VALUES (?, ?)
                """,
                (q.query_text, expected_json),
            )
            continue

        canonical_id = int(rows[0]["id"])
        if str(rows[0]["expected_source_uris"] or "") != expected_json:
            conn.execute(
                """
                UPDATE queries_eval
                SET expected_source_uris = ?
                WHERE id = ?
                """,
                (expected_json, canonical_id),
            )
        for dup in rows[1:]:
            conn.execute("DELETE FROM queries_eval WHERE id = ?", (int(dup["id"]),))
    conn.commit()
    return len(queries)


def parse_expected_source_uris(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not isinstance(raw, str):
        return [str(raw)]
    text = raw.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [s.strip() for s in text.split(",") if s.strip()]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, str):
        return [parsed]
    return []


def _unique_source_uris_in_order(source_uris: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for uri in source_uris:
        if uri in seen:
            continue
        seen.add(uri)
        out.append(uri)
    return out


def _first_correct_rank(retrieved_source_uris: list[str], expected: set[str]) -> int | None:
    if not expected:
        return None
    for idx, uri in enumerate(retrieved_source_uris, start=1):
        if uri in expected:
            return idx
    return None
