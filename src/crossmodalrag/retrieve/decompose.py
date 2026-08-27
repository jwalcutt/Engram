"""Deterministic decomposition of comparative queries.

"Difference between X and Y" needs evidence for *both* X and Y, but a single
top-k pass over the whole query lets the better-represented side crowd the
other out entirely (observed: a CNN-vs-RNN query retrieved only CNN chunks, so
synthesis could not compare and abstained). A matched comparative query is
retrieved as two sub-queries with reserved result slots per side.

Conservative by design: a small set of explicit patterns, no LLM, and any
query that does not match retrieves exactly as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from crossmodalrag.retrieve.lexical import RetrievalHit

# Each pattern captures the two compared subjects. Kept intentionally narrow:
# "difference in performance" or a bare "compare notes" must not match.
_COMPARATIVE_PATTERNS = (
    re.compile(r"\bdifferences?\s+between\s+(.+?)\s+and\s+(.+?)(?:[.?!]|$)", re.IGNORECASE),
    re.compile(r"\bcompare\s+(.+?)\s+(?:with|to|and)\s+(.+?)(?:[.?!]|$)", re.IGNORECASE),
    re.compile(r"^(?:.*?\b)?(\S.*?)\s+(?:vs\.?|versus)\s+(.+?)(?:[.?!]|$)", re.IGNORECASE),
)

# Leading noise words stripped from a captured subject ("a", "the", ...).
_SUBJECT_PREFIX_RE = re.compile(r"^(?:a|an|the|using)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class ComparativeQuery:
    left: str
    right: str


def split_comparative_query(query: str) -> ComparativeQuery | None:
    """Return the two compared subjects, or None when the query is not comparative."""
    for pattern in _COMPARATIVE_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        left = _clean_subject(match.group(1))
        right = _clean_subject(match.group(2))
        if left and right and left.lower() != right.lower():
            return ComparativeQuery(left=left, right=right)
    return None


def _clean_subject(text: str) -> str:
    return _SUBJECT_PREFIX_RE.sub("", text.strip().strip(",;:")).strip()


def merge_side_hits(
    left_hits: list[RetrievalHit],
    right_hits: list[RetrievalHit],
    top_k: int,
    full_hits: list[RetrievalHit] | None = None,
) -> list[RetrievalHit]:
    """Merge full-query and per-side results with reserved slots.

    The full-query ranking stays the base — a source that covers *both*
    subjects (the ideal comparative evidence) keeps its joint-context rank —
    while each side holds a small reserved share so the thinner subject cannot
    be crowded out entirely. Unfilled shares are backfilled best-score-first
    from all pools. All inputs must be score-sorted (descending); duplicates
    count toward the first pool that claims them. The final list is
    score-sorted for display; slot reservation only decides membership.
    """
    side_share = max(1, top_k // 4)
    full_share = top_k - 2 * side_share if full_hits is not None else 0

    seen: set[int] = set()
    picked: list[RetrievalHit] = []

    def take(hits: list[RetrievalHit], limit: int) -> None:
        taken = 0
        for hit in hits:
            if taken >= limit or len(picked) >= top_k:
                break
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            picked.append(hit)
            taken += 1

    take(full_hits or [], full_share)
    take(left_hits, side_share)
    take(right_hits, side_share)
    # Backfill unfilled slots from every pool's remainder, best-scored first.
    if len(picked) < top_k:
        remainder = sorted(
            (full_hits or []) + left_hits + right_hits, key=lambda h: h.score, reverse=True
        )
        take(remainder, top_k - len(picked))

    picked.sort(key=lambda hit: hit.score, reverse=True)
    return picked
