#!/usr/bin/env python3
"""native-embedding measurement spike (throwaway, not product).

The authoritative gate FIRED: OCR-text-first cannot retrieve the visually-dominant
diagram (visual-slice Recall@5 = 0). This spike measures whether native CLIP image
embeddings rescue that slice, and whether a dual-space rank-time merge is feasible
without regressing the text slice. It answers a decision; it does NOT ship anything.

It compares three rankers on the seeded `[sample-xmodal-*]` eval slices:
  A  OCR-text-first      — the existing semantic-over-text retriever (bge-small)
  B  CLIP image space    — Qdrant/clip-ViT-B-32 vision(images) + text(queries), cosine
  A⊕B  dual-space merge  — Reciprocal Rank Fusion (scale-free; needs only ranks)

Requires the [image-embeddings] + [ocr] extras and a seeded+reindexed DB. First run
downloads the CLIP ONNX towers (~150-350 MB). NEVER imported by src/crossmodalrag/.

Usage:
  mem seed-sample --db-path /tmp/x6.db --workspace-dir /tmp/x6-ws --force
  CMRAG_DB_PATH=/tmp/x6.db mem reindex-embeddings
  python scripts/spikes/clip_dual_space_spike.py --db-path /tmp/x6.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crossmodalrag.config import get_db_path, load_dotenv  # noqa: E402
from crossmodalrag.db import connect, init_db  # noqa: E402
from crossmodalrag.evaluation import list_eval_queries  # noqa: E402
from crossmodalrag.retrieve.hybrid import retrieve as hybrid_retrieve  # noqa: E402

VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"
TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
TOP_K = 5            # gate-consistent retrieval depth (run_eval uses top-K chunks)
RRF_K = 60           # standard RRF constant


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def ocr_text_first_ranking(conn, query: str) -> list[str]:
    """Ranker A: existing semantic-over-text retrieval -> ordered unique source_uris.

    Uses top_k=TOP_K chunks to stay **consistent with the authoritative gate**
    (`run_eval` measures unique sources among the top-K chunks). A deeper pool would
    over-credit OCR-text-first: the diagram's "Pipeline Overview" OCR text lands at
    chunk-rank ~8 for the visual queries — reachable, but not in the top-K the gate
    (and thus the whole decision) is built on.
    """
    hits = hybrid_retrieve(conn, query=query, top_k=TOP_K, profile="relevant")
    return _ordered_unique([h.source_uri for h in hits])


class ClipImageRanker:
    """Ranker B: rank image source_uris by CLIP text<->image cosine."""

    def __init__(self, image_uris: list[str]) -> None:
        import numpy as np
        from fastembed import ImageEmbedding, TextEmbedding

        self._np = np
        self._text = TextEmbedding(TEXT_MODEL)
        vision = ImageEmbedding(VISION_MODEL)
        self.image_uris = [u for u in image_uris if Path(u).exists()]
        mats = list(vision.embed(self.image_uris)) if self.image_uris else []
        self._img = np.array(mats, dtype=np.float32) if mats else np.zeros((0, 512), np.float32)

    def rank(self, query: str) -> list[tuple[str, float]]:
        np = self._np
        if self._img.shape[0] == 0:
            return []
        q = np.array(list(self._text.embed([query]))[0], dtype=np.float32)
        qn = float(np.linalg.norm(q)) or float("inf")
        inorms = np.linalg.norm(self._img, axis=1)
        inorms[inorms == 0.0] = np.inf
        sims = (self._img @ q) / (inorms * qn)
        order = np.argsort(-sims)
        return [(self.image_uris[i], float(sims[i])) for i in order]

    def ranked_uris(self, query: str) -> list[str]:
        return [u for u, _ in self.rank(query)]


def rrf_merge(*rankings: list[str], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion over several ranked uri lists (scale-free)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, uri in enumerate(ranking, start=1):
            scores[uri] = scores.get(uri, 0.0) + 1.0 / (k + rank)
    return [uri for uri, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def recall_at_k(ranked: list[str], gold: set[str], k: int = TOP_K) -> float:
    if not gold:
        return 0.0
    return 1.0 if any(uri in gold for uri in ranked[:k]) else 0.0


def slice_recall(rankings_per_query: list[tuple[list[str], set[str]]]) -> float:
    if not rankings_per_query:
        return 0.0
    return sum(recall_at_k(r, g) for r, g in rankings_per_query) / len(rankings_per_query)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="CLIP dual-space measurement spike (non-product).")
    parser.add_argument("--db-path", type=Path, default=None)
    args = parser.parse_args()
    db_path = (args.db_path or get_db_path()).expanduser().resolve()

    conn = connect(db_path)
    init_db(conn)
    image_uris = [
        str(r["source_uri"])
        for r in conn.execute("SELECT source_uri FROM sources WHERE source_type = 'image'").fetchall()
    ]
    text_q = list_eval_queries(conn, query_prefix="[sample-xmodal-text]")
    visual_q = list_eval_queries(conn, query_prefix="[sample-xmodal-visual]")

    print(f"DB: {db_path}")
    print(f"image sources: {len(image_uris)} | text-slice queries: {len(text_q)} | "
          f"visual-slice queries: {len(visual_q)}")
    if not image_uris:
        print("No image sources — seed with the [ocr] extra installed, then reindex. Aborting.")
        return

    print(f"Loading CLIP towers ({VISION_MODEL} / {TEXT_MODEL})… (first run downloads weights)")
    clip = ClipImageRanker(image_uris)

    rows = []  # (slice_name, queries)
    for name, queries in (("text-heavy", text_q), ("visual-heavy", visual_q)):
        a_set, b_set, ab_set = [], [], []
        diag = []
        for q in queries:
            gold = set(q.expected_source_uris)
            a = ocr_text_first_ranking(conn, q.query_text)
            b = clip.ranked_uris(q.query_text)
            ab = rrf_merge(a, b)
            a_set.append((a, gold))
            b_set.append((b, gold))
            ab_set.append((ab, gold))
            # diagnostic: CLIP rank of each gold image uri (1-based) if present
            for g in gold:
                if g in b:
                    diag.append((q.query_text[:48], g.split("/")[-1], b.index(g) + 1))
        rows.append((name, slice_recall(a_set), slice_recall(b_set), slice_recall(ab_set), diag))
    conn.close()

    print(f"\nRecall@{TOP_K} by slice (n=queries):")
    print(f"{'slice':<14}{'OCR-text-first (A)':>20}{'CLIP-only (B)':>16}{'dual RRF (A⊕B)':>16}")
    for name, ra, rb, rab, _ in rows:
        print(f"{name:<14}{ra:>20.3f}{rb:>16.3f}{rab:>16.3f}")

    print("\nDiagnostic — CLIP rank of gold image per query:")
    for name, _, _, _, diag in rows:
        for qtext, fname, rank in diag:
            print(f"  [{name}] '{qtext}…' -> {fname} at CLIP rank {rank}")

    # Verdict (feasibility + benefit), recorded from this output.
    text_a = next(r[1] for r in rows if r[0] == "text-heavy")
    text_ab = next(r[3] for r in rows if r[0] == "text-heavy")
    vis_a = next(r[1] for r in rows if r[0] == "visual-heavy")
    vis_b = next(r[2] for r in rows if r[0] == "visual-heavy")
    vis_ab = next(r[3] for r in rows if r[0] == "visual-heavy")
    print("\nVerdict:")
    print(f"  feasibility (RRF merge held text slice): {'YES' if text_ab >= text_a else 'NO'} "
          f"(text A={text_a:.3f} -> A⊕B={text_ab:.3f})")
    lift = vis_ab - vis_a
    print(f"  benefit (visual lift from native CLIP): A={vis_a:.3f} CLIP-only={vis_b:.3f} "
          f"A⊕B={vis_ab:.3f} (lift {lift:+.3f})")
    print("\nNOTE: tiny synthetic corpus — directional only. Measurement, not adoption.")


if __name__ == "__main__":
    main()
