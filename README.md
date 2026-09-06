# Engram

Engram is a local-first memory engine for your notes, code history, PDFs, and screenshots. Every
answer cites the exact sources it drew from, and when the evidence is too weak the system says so
instead of guessing.

Current scope:

- Ingest markdown notes into SQLite.
- Ingest git commits and diffs into SQLite.
- Ingest PDF text (per-page, with page locators) into SQLite; optional `[pdf]` extra.
- Ingest image or diagram text via local OCR (with an OCR-confidence signal); optional `[ocr]` extra.
- Create structure-aware searchable chunks from evidence.
- Query with a hybrid (semantic vector, lexical, and recency) retriever and get cited evidence.
- Optional local embeddings via `fastembed` (no torch); falls back to lexical when not installed.
- Synthesize direct, grounded answers with a local LLM (Ollama) that treat the retrieved evidence as
the record of your own experience and cite it inline, abstaining when evidence is weak. Without
Ollama, answers fall back to a deterministic template.



## Quickstart

1. Create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For semantic (vector) retrieval, install the optional embeddings extra. The core stays
dependency-free; without this extra everything still works using lexical retrieval only.

```bash
pip install -e ".[embeddings]"   # adds fastembed (ONNX, local, no torch) + numpy
```

1. Initialize the local database:

```bash
mem init-db
```

Optional: seed deterministic synthetic notes + git history for smoke testing:

```bash
mem seed-sample
```

1. Ingest notes:

```bash
mem ingest-notes /path/to/obsidian/vault
```

Or ingest multiple vaults in one command:

```bash
mem ingest-notes /path/to/vault-a /path/to/vault-b
```

If you omit paths, `mem ingest-notes` will use all `OBSIDIAN_VAULT_PATH_<n>` values from your local `.env`.

Note chunking is markdown header-aware. Every chunk is prefixed with a one-line context breadcrumb
(`note title > heading > subheading`), so a chunk's subject is part of its searchable and embedded
text even when the body is mostly formulas or tables. Oversized sections are split on sentence or
whitespace boundaries, never mid-word.

1. Ingest git history:

```bash
mem ingest-git /path/to/repo --max-commits 300
```

Or ingest multiple repos in one command:

```bash
mem ingest-git /path/to/repo-a /path/to/repo-b --max-commits 300
```

If you omit paths, `mem ingest-git` will use all `REPO_PATH_<n>` values from your local `.env`.

### Ingest PDFs (optional, cross-modal)

PDF ingestion is text-first: each page's extractable text becomes searchable chunks that carry a
1-based page locator, so evidence can be cited as `file.pdf p.N`. It requires the optional `[pdf]`
extra (`pip install -e ".[pdf]"`); without it the command exits with an install hint and the core
stays dependency-free.

```bash
mem ingest-pdf /path/to/file.pdf /path/to/dir-of-pdfs
```

A path may be a single `.pdf` file or a directory (searched recursively). One source row is created
per file; re-ingesting an unchanged file is a no-op. The fingerprint folds in the extractor version,
so an extractor upgrade re-derives intentionally. Pages with no extractable text (scanned or
image-only pages, for example) register the source but produce no chunks; use image OCR (below) for
those. If you omit paths, `mem ingest-pdf` uses all `PDF_PATH_<n>` values from your local `.env`.

### Ingest images and diagrams (optional, OCR)

Image ingestion is OCR-text-first: the recognized text of each image becomes searchable chunks
tagged `modality=ocr` and carrying an `ocr_confidence` signal, since low-confidence OCR is weak
evidence. It requires the optional `[ocr]` extra (`pip install -e ".[ocr]"`) plus a local
[Tesseract](https://github.com/tesseract-ocr/tesseract) binary (`brew install tesseract`, for
example). Without them the command exits with a hint and the core stays dependency-free.

```bash
mem ingest-images /path/to/diagram.png /path/to/dir-of-images
```

A path may be a single image (`.png/.jpg/.jpeg/.gif/.bmp/.tif/.tiff/.webp`) or a directory (searched
recursively; non-image files are ignored). One source row per image; re-ingesting an unchanged image
is a no-op, because the fingerprint folds in the OCR engine version. An image whose OCR yields no
text registers the source with no chunks. Purely visual content (a diagram whose meaning lives in its
layout, not its words) is intentionally hard to retrieve from OCR text alone. Measuring that gap is
what `scripts/xmodal_gate.py` is for. If you omit paths, `mem ingest-images` uses all
`IMAGE_PATH_<n>` values from your local `.env`.

### Forcing a re-chunk (after chunker changes)

Ingestion is idempotent: a source whose content fingerprint is unchanged is skipped, and its
existing chunks are left untouched. Source fingerprints fold in the chunker version, so when a
release changes the chunking logic (and bumps `CHUNKER_VERSION`), simply re-running ingestion
(`mem sync`, or the individual `ingest-*` commands) re-chunks and re-embeds every existing source
in place. Unchanged sources under an unchanged chunker still never churn.

Alternatively, to re-chunk from a clean slate, rebuild the database from scratch:

```bash
# Rebuild the default ./data/memory.db in place
rm -f ./data/memory.db ./data/memory.db-wal ./data/memory.db-shm
mem init-db
mem ingest-notes   # explicit paths, or OBSIDIAN_VAULT_PATH_* from .env
mem ingest-git     # explicit paths, or REPO_PATH_* from .env
```

To rebuild into a separate database without disturbing your current one, point `CMRAG_DB_PATH`
at a fresh path first, then ingest there and compare before swapping:

```bash
export CMRAG_DB_PATH=/absolute/path/to/memory-rechunked.db
mem init-db
mem ingest-notes
mem ingest-git
```

Note: a full rebuild also clears the `queries_eval` table. If you have custom evaluation
queries, re-add them with `mem eval --load-queries file.json` (see the query file format below)
after re-ingesting.

1. (Optional) Build semantic embeddings for ingested chunks:

```bash
mem reindex-embeddings
```

This embeds any chunks that don't yet have a vector for the active model (resumable and
idempotent). It requires the `[embeddings]` extra. Ingestion also embeds inline when the extra
is installed, so `reindex-embeddings` is mainly for backfilling existing data or after changing
the model. Vectors are tagged with the model that produced them; changing `CMRAG_EMBED_MODEL`
means you should re-run this command.

1. Ask a question:

```bash
mem ask "What have I learned so far about gradient descent?" --top-k 5 --profile relevant --explain
```

By default `mem ask` synthesizes a grounded answer with a local LLM via Ollama, treating the
retrieved evidence as the authoritative record of your own experience. Every claim about your work,
history, or notes cites its evidence inline as `[E#]`. The model integrates across all materially
relevant evidence items, multi-citing sources that independently agree, rather than answering from
the single top hit. It may draw on general knowledge to explain and connect the evidence, but it
never presents that knowledge as coming from your records, and never treats it as grounds to answer
a question your records don't address.

If retrieval is too weak (top score below `CMRAG_MIN_EVIDENCE_SCORE`), `ask` abstains instead of
guessing. When the evidence only partially covers the question, it answers what the evidence
supports and states what it doesn't cover, rather than refusing outright. Every abstention carries a
reason: `weak_retrieval` (the gate short-circuited before the LLM) or `llm_insufficient` (the model
judged the evidence irrelevant). The status line shows that reason alongside the top retrieval
score, and `--json` exposes it as `abstention_reason`. If Ollama is unreachable, `ask` falls back to
the deterministic evidence template automatically.

If generation stops because the model's context window filled, the answer is flagged as truncated
rather than presented as complete: a visible warning in the CLI and web console, plus an additive
`truncated` key in the `--json` and API contract. Raise `CMRAG_LLM_NUM_CTX`, lower `--top-k`, or
narrow the question.

In an interactive terminal the answer streams token-by-token as it is generated, and the citations
and evidence footer render once generation completes (citation validation and abstention detection
still run on the full output). Pass `--no-stream` to print the answer only when finished. Piped or
redirected output and `--json` are always buffered, so scripting contracts are unchanged.

### Interactive multi-turn sessions (`mem chat`)

Run `mem chat` (or `mem ask` with no query) to start an interactive chat session. Enter successive
questions until you quit, and ask follow-ups ("expand on that", "how does that relate to…") that
resolve against the conversation so far.

- **In-session commands:** `/exit` or `/quit` (or Ctrl-D / Ctrl-C) ends the session; `/clear` (or
`/new`) resets the carried context without leaving; blank lines are ignored.
- **Provenance is per-turn:** every turn runs retrieval independently against your memory store,
the weak-evidence gate and abstention apply per turn, and `[E#]` citations always refer to the
*current* turn's evidence. Prior turns inform phrasing and reference resolution only, and their
old citation markers are stripped from the carried context. Requests about the conversation
itself ("rephrase that last answer") are answered from the conversation and carry no citations.
- **Context window:** the last `CMRAG_CHAT_CONTEXT_TURNS` (default 8) answered turns are carried.
Older turns are dropped deterministically, and abstained or template (`--no-llm` or LLM-offline)
turns are never carried. Set it to 0 to disable carried context.
- Ask options (`--profile`, `--level`, `--modality`, `--track`/`--no-track`, `--no-stream`,
`--explain`, `--debug`) apply to every turn; `--json` and `--accept` are one-shot-only.
- Piped stdin works as a batch mode (one query per line, no prompt or banner). With lexical-only
retrieval (no embeddings extra), a purely referential follow-up ("expand on that") may retrieve
nothing and abstain; semantic retrieval resolves this in practice.



#### Chat history

Engram saves interactive sessions locally by default to the memory DB (`conversations` and
`messages`), so you can browse past exchanges later. Each saved answer keeps its full evidence
snapshot (so provenance drill-down survives re-chunking), its abstention reason, and its truncation
flag. The local LLM auto-titles new conversations from their first exchange, producing a short
few-word name; without Ollama the title falls back to the first question. History is private and
local-only, never sent anywhere. It is also additive and separable, meaning it never enters any
ingestion or derivation fingerprint, so dropping it changes nothing else. You control all of it:

- `mem chat --resume [<id>]` resumes a saved conversation. Its answered turns load as carried
context (the context cap and abstained-skip rules apply) and new turns append to the same
conversation. Omit the id to resume the most recent one.
- `mem history` lists saved conversations (newest first, `--top N`, `--json`).
- `mem history --show <id>` replays one conversation chat-style with cited-evidence refs
(`--json` emits the full contract, evidence snapshots included).
- `mem history --rename <id> --title "<name>"` renames one conversation, overriding the auto-title.
- `mem history --clear [--id <id>]` wipes everything, or one conversation.
- `--no-save` on `mem chat` or bare `mem ask`, or `CMRAG_SAVE_HISTORY=off`, disables saving.
- In-session: `/new` starts a fresh saved conversation (and clears context); `/clear` only resets
the carried context and stays in the same conversation.

Engram records only interactive sessions. One-shot `mem ask "<query>"`, `--no-llm` template turns,
and eval or scripted paths never write history.

Requires [Ollama](https://ollama.com) running locally with a model pulled
(`ollama pull gemma4`). Swap models anytime via `CMRAG_LLM_MODEL`.

- `--profile` selects the hybrid retrieval blend of semantic, lexical, recency, and usage signals:
  - `balanced` (default): 0.55 vector + 0.30 lexical + 0.15 recency
  - `relevant`: 0.70 vector + 0.25 lexical + 0.05 recency
  - `recent`: 0.35 vector + 0.20 lexical + 0.45 recency
  - `usage`: 0.55 vector + 0.25 lexical + 0.05 recency + 0.15 usage (rehearsal strength). This
  opt-in, time-aware profile promotes memories you've used recently or often. The usage term is 0
  in every other profile, so they are unchanged. It re-ranks only already-relevant candidates
  (never surfacing irrelevant ones) and needs embeddings plus a usage history
  (`CMRAG_USAGE_HALFLIFE_DAYS`, `CMRAG_USAGE_SATURATION`). `--explain` and `--json` expose a
  `usage` score component.
- **Usage tracking** is opt-in and private. The `usage` profile's history comes from real `mem ask`
interactions, but tracking is off by default. Enable it with `CMRAG_USAGE_TRACKING=on`, or
per-call with `mem ask … --track` (logs `retrieval_hit` for the evidence shown) and
`mem ask … --accept` (also logs `accepted_answer` for the cited evidence; implies `--track`).
`mem ask … --level …` additionally logs an `open` event for the memory nodes drilled into.
`--no-track` suppresses logging for a call. Only the target id, event type, and time are stored,
never your query text, and everything stays local. Inspect with `mem usage`; wipe with
`mem usage --clear`.
- **Forgetting risk** (`mem forgetting`): surfaces important-but-stale memories, answering "what am
I likely forgetting that's still relevant?" Each memory node is scored `risk = importance × staleness`. Importance is its graph centrality (run `mem build-memory` first). Staleness grows with
time since you last touched it, whether through a recent `retrieval` or `open` event via tracking,
or through its content age. Rehearsing a memory therefore lowers its risk. Every item is grounded to its L0
evidence and shows its importance, staleness, and confidence components. Use `--level all` to rank
across events, episodes, and concepts. Read-only.
- **Active recall** (`mem recall`): turns the highest forgetting-risk memories into grounded study
cards, each a question plus a one-sentence answer drawn strictly from that memory's L0 evidence.
The local LLM generates the cards (`CMRAG_EXTRACT_MODEL`, temp 0) and caches them, regenerating
only when the underlying memory changes or when you pass `--regenerate`. If Ollama is unavailable
it falls back to a deterministic template question plus an evidence excerpt, so it never fails.
Each card cites its evidence. `--level all` covers events, episodes, and concepts.
- **Concept drift** (`mem drift`): shows how your L3 concepts have moved over time. For each
concept, its member events are bucketed into time windows (`CMRAG_DRIFT_WINDOW_DAYS`, default 30),
a per-window prototype (centroid) is computed, and the drift between consecutive windows is scored
as `1 − cosine`. Concepts re-engaged after an empty window are flagged as relearning. Each item
shows window count, span, support, confidence, and a grounding URI. Build the snapshots first with
`mem build-memory --level drift` (deterministic, no LLM; needs the `[embeddings]` extra); `mem drift`
is a read-only view. Timestamps drive the windows, so see "date-aware note dates" below.
`mem drift --json` emits a stable contract: per concept, `concept_id`, `overall_drift`,
`relearning`, a grounding URI, and a `windows` array holding the movement trajectory for a UI to
plot.
- **Distilled stand-ins** (`mem distill`): lists the compact, retrieval-preserving representations
built by `mem build-memory --level distill`. For each L2 or L3 node it shows the summary, the kept
(core) versus full L0 evidence counts, the achieved compression ratio, confidence, and a grounding
URI. Read-only. `mem distill --json` adds a stable per-node contract plus the per-level
`overall_compression_ratio`. Whether to actually retrieve via these stand-ins in production is
decided by the distillation gate (`scripts/distill_gate.py`), not enabled by default. See the
distillation gate section below.
- `--level` chooses the retrieval level: `evidence` (default, L0 chunks) or a memory level
(`event`/`episode`/`concept`). At a memory level, `ask` retrieves the matching nodes, prints them,
then drills them down to their L0 evidence and answers grounded in (and citing) that L0, so
provenance holds regardless of entry level. Memory-level retrieval needs the hierarchy built
(`mem build-memory`) and benefits from node embeddings (`mem reindex-embeddings`); without the
embeddings extra it ranks nodes lexically.
- **Comparative queries** ("difference between X and Y", "compare X with Y", "X vs Y") are
detected deterministically, without an LLM. Retrieval then runs three passes: the full query plus
one per subject, merged with reserved slots. The better-represented subject therefore cannot crowd
the other out of the evidence entirely. If one side has no evidence, the answer covers the present side
and says what's missing. Non-comparative queries retrieve exactly as before; `--explain` and
`--json` show which sub-query produced each hit (`subquery`).
- `--explain` prints per-hit score components (vector, lexical, recency, usage, title).
Ranking includes a small additive title boost (`CMRAG_TITLE_BOOST_WEIGHT`, default 0.05;
0 disables): query-term overlap with the source *title*, so a note literally named for the
query's terms wins near-ties against incidental mentions in commit diffs. Lexical matching
also treats underscore notation and its plain spelling as equivalent (a note's `F_1`
matches a query's `f1`).
Retrieval also applies a source-diversity cap (`CMRAG_MAX_CHUNKS_PER_SOURCE`, default 2;
0 disables): a single source contributes at most that many chunks to a result, so one strongly
matching note cannot fill the whole evidence list and crowd out every other source. Drill-down
retrieval from memory nodes is never capped, because it deliberately ranks within one node's
evidence.
- `--no-llm` skips synthesis and returns the deterministic evidence template.
- `--no-stream` disables live token streaming and prints the finished answer in one block
(streaming only applies to interactive terminals; piped output and `--json` are always buffered).
- `--json` emits a structured answer (stable contract for UIs). It includes `matched_nodes` at
memory levels, plus a `timing` block carrying `total_seconds` for the whole ask and
`generation_seconds` for the LLM call alone, so latency is measurable per answer.
Each evidence entry also carries cross-modal provenance: `modality`, a rendered `locator`
(`spec.pdf p.4`, for example), and `page` or `ocr_confidence` when applicable (additive; existing
fields unchanged).
- `--modality` restricts evidence to one or more modalities (repeatable): `text` (notes), `code`
(git), `pdf`, `image` (OCR'd images). Citations render the modality and locator inline.
- `--debug` adds retrieval diagnostics plus the raw prompt and model output.



```bash
mem ask "what are the themes of this project?" --level concept   # retrieve concepts, answer from their L0 evidence
mem ask "what does the spec say about rate limits?" --modality pdf --json   # only PDF evidence, cited as file.pdf p.N
```

Near-identical evidence is de-duplicated across modalities (an OCR'd screenshot of a note and the
note itself, for example) so the same content isn't cited twice (`CMRAG_DEDUPE_THRESHOLD`, default
0.95).

1. Run retrieval evaluation (using seeded sample queries or your own `queries_eval` rows):

```bash
mem eval --top-k 5 --profile relevant                 # evidence-level recall/MRR
mem eval --top-k 5 --level concept                    # drill-down recall via the concept layer
```

1. Evaluate grounded answer quality (citation faithfulness, requires Ollama):

```bash
mem eval-generation --query-prefix "[sample]" --profile relevant            # flat L0 baseline
mem eval-generation --query-prefix "[sample-synth]" --level concept          # synthesis at a memory level
```

This reports `citation_validity` (no hallucinated `[E#]`), `source_grounding_hit` (cites at
least one expected source), `source_coverage` (fraction of an answerable query's expected
sources actually cited, which rewards multi-source synthesis), and `abstention_correct` (answers
answerable queries, abstains on unanswerable ones). Queries with empty `expected_source_uris`
are treated as negative (should-abstain) cases. Swap `CMRAG_LLM_MODEL` to compare models.

`--level` runs the same citation-faithfulness eval over a memory level (`event`/`episode`/
`concept`): it retrieves matching nodes, drills them down to their L0 evidence, and synthesizes
from that L0, so answers still cite L0 `[E#]` regardless of entry level. Compare a memory level
against `--level evidence` to gauge the synthesis benefit. The seeded `[sample-synth]` queries
have multi-source gold for exactly this; the `[sample]` queries are single-source specific-fact
checks for measuring no regression.

## CLI Commands

- `mem init-db`
- `mem doctor [--json]` (read-only health check: DB, installed extras, Ollama reachability, active models, config file, configured connectors, memory build and integrity)
- `mem sync [--max-commits N] [--only notes|git|pdf|image ...] [--json]` (incrementally re-ingest every connector configured in `.env`/the config file; idempotent, so only changed sources are re-chunked)
- `mem backup [<dest>]` (write a consistent single-file copy of the local DB; WAL-safe)
- `mem restore <src> [--force]` (replace the local DB with a backup; destructive, so `--force` is required to overwrite an existing DB)
- `mem serve [--host 127.0.0.1] [--port 8765]` (local web console + HTTP API serving the JSON contracts; requires the `[ui]` extra; localhost-only by default)
- `mem mcp [--db PATH]` (serve the read views to an AI assistant as Model Context Protocol tools over stdio; requires the `[mcp]` extra; no network listener)
- `mem seed-sample [--workspace-dir PATH] [--force]`
- `mem ingest-notes [<vault_path> ...]` (falls back to `.env` `OBSIDIAN_VAULT_PATH_*`)
- `mem ingest-git [<repo_path> ...] [--max-commits N]` (falls back to `.env` `REPO_PATH_*`)
- `mem ingest-pdf [<path> ...]` (file or directory; falls back to `.env` `PDF_PATH_*`; requires the `[pdf]` extra)
- `mem ingest-images [<path> ...]` (file or directory; falls back to `.env` `IMAGE_PATH_*`; requires the `[ocr]` extra + a tesseract binary)
- `mem ask "<query>" [--top-k N] [--level evidence|event|episode|concept] [--profile balanced|relevant|recent|usage] [--modality text|code|pdf|image ...] [--explain] [--no-llm] [--no-stream] [--json] [--debug] [--track|--no-track] [--accept]`
- `mem chat [--resume [<id>]] [--no-save] [ask options ...]` (interactive multi-turn session; `mem ask` with no query does the same)
- `mem history [--top N] [--show <id>] [--rename <id> --title "<name>"] [--clear [--id <id>]] [--json]` (browse, rename, or wipe saved chat conversations)
- `mem usage [--clear] [--top N] [--json]` (local usage-tracking stats; `--clear` wipes the history)
- `mem forgetting [--level concept|episode|event|all] [--top N] [--min-support N] [--json]` ("what am I likely forgetting?"; important-but-stale memories, grounded to evidence)
- `mem recall [--level concept|episode|event|all] [--top N] [--min-support N] [--regenerate] [--json]` (grounded active-recall study cards for the highest forgetting-risk memories)
- `mem drift [--top N] [--min-support N] [--json]` (concept drift over time windows; read-only, so run `mem build-memory --level drift` first, needs the `[embeddings]` extra)
- `mem distill [--top N] [--json]` (list distilled node stand-ins: core versus full evidence, plus compression ratio; read-only, so run `mem build-memory --level distill` first)
- `mem eval [--top-k N] [--query-prefix PREFIX] [--load-queries PATH.json] [--profile ...] [--level ...] [--modality text|code|pdf|image ...] [--json]`
- `mem eval-generation [--top-k N] [--query-prefix PREFIX] [--profile ...] [--level evidence|event|episode|concept] [--model ID]` (requires Ollama)
- `mem reindex-embeddings [--batch-size N] [--model ID]` (requires the `[embeddings]` extra)
- `mem build-memory [--level event|episode|concept|graph|drift|distill|all] [--limit N] [--model ID]` (events/concept-naming use Ollama; concepts + drift + distill need the `[embeddings]` extra; episode/graph need neither)
- `mem memory-stats [--json]` (node and edge counts, integrity, plus distilled-node + drift-snapshot counts)
- `mem concepts [--top N] [--json]` (L3 concepts by centrality)
- `mem timeline [--limit N] [--json]` (L2 episodes, oldest first)



### Operations (sync, doctor, progress, exit codes)

- `mem sync` re-ingests every connector configured in your `.env` (`OBSIDIAN_VAULT_PATH_*`,
`REPO_PATH_*`, `PDF_PATH_*`, `IMAGE_PATH_*`) in one pass. It is incremental and idempotent:
ingestion fingerprint-skips unchanged sources, so a second run re-chunks only what changed. PDF and
image connectors are skipped with a note (not an error) when their extra is absent, and a bad path
is reported per-connector without aborting the rest of the sync. `--only` restricts to specific
connectors; `--json` emits a summary.
- `mem doctor` is a read-only health check covering DB path and size, which optional extras are
installed, whether Ollama is reachable, the active embed, LLM, and extract models, configured
connector counts, and memory build and integrity. Use it to diagnose "why is retrieval
lexical-only?" or "why did `ask` fall back to the template?".
- **Progress.** Long operations (`ingest-`*, `sync`, `reindex-embeddings`, `build-memory`) print a
progress line to stderr only when stderr is an interactive terminal, so piping or `--json` output is
never polluted.
- **Exit codes.** `0` success; `1` an expected, reported failure (printed as `error: <message>`, no
stack trace); `2` a command-line usage error.
- **Backup and restore.** `mem backup [<dest>]` writes a WAL-safe single-file copy of the DB
(default: alongside it with a timestamp). `mem restore <src>` replaces the active DB from a backup.
Restoring destroys the current DB, so the command refuses to overwrite an existing one unless you
pass `--force`.



### Config file (optional)

Beyond `.env`, an optional TOML config file can hold connector paths and retrieval defaults.
Engram looks for `$CMRAG_CONFIG`, else `./crossmodalrag.toml`. Resolution precedence is always
CLI flag, then environment or `.env`, then config file, then built-in default. The config therefore
never overrides something you pass explicitly, and the eval baselines (which pass an explicit
profile) do not move.

```toml
[connectors]                       # used by `mem sync` / `mem ingest-*` fallback
notes = ["/path/to/vault"]
git   = ["/path/to/repo"]

[retrieval]                        # applied when you omit the flag
profile = "relevant"               # balanced | relevant | recent | usage
top_k = 5
```

(Requires Python 3.11+ for the stdlib `tomllib` parser; the core install stays dependency-free.)

### JSON output contracts

The `--json` modes are stable, machine-readable contracts intended for tooling and UI integration.
`ask`, `eval`, `concepts`, `timeline`, `memory-stats`, `forgetting`, `recall`, `usage`, `drift`,
`distill`, and `history` all support `--json`. Each contract is additive-only: field names and
shapes stay backward-compatible, and changes only *add* keys, never renaming or removing them.
Every contract that returns memory items carries stable identifiers (`node_id`) and provenance
(`evidence_source_uris`, or evidence ids plus L0 locators) so a consumer can drill back down to the
source. The library owns the shapes (the `*_to_dict`, `list_*`, and `memory_stats` helpers), and
`tests/test_json_contracts.py` pins them, so the CLI, API, UI, and MCP tools render exactly the same
payloads.

### Web UI and local API (`mem serve`)

`mem serve` runs a local HTTP API and serves a web console whose primary surface is a grounded chat.
Conversations are multi-turn and carry context, and each saved conversation gets its own route,
navigable and resumable from the sidebar's history panel and auto-titled by the local LLM. User
messages and cited assistant replies stack above a bottom composer, and every reply carries a
per-message evidence ledger with claim to exact-evidence drill-down. A Settings page consolidates
the retrieval and generation parameters
(profile, memory level, evidence per turn, synthesis, history saving). The console also keeps the
read views: concepts, timeline, concept-drift movement, and forgetting/recall. It requires the
opt-in `[ui]` extra (`pip install -e ".[ui]"`, adds FastAPI + uvicorn) and binds `127.0.0.1` by
default, with no auth on loopback, nothing leaving the machine, and all assets (including fonts)
vendored. Without the extra, `mem serve` exits with an install hint.

```bash
pip install -e ".[ui]"
mem serve                       # web UI + API at http://127.0.0.1:8765  (Ctrl-C to stop)
# then open http://127.0.0.1:8765 in a browser
```

The API is read-first: every memory and engine route is GET and read-only (no writes; `/ask` does
not record usage). Those routes are `/health`, `/ask`, `/conversations`, `/conversations/{id}`,
`/concepts`, `/timeline`, `/memory-stats`, `/forgetting`, `/recall`, `/drift`, `/distill`, and
`/usage`, each returning exactly the corresponding `--json` payload. The explicit write paths touch
only the user-owned chat-history tables (the same store as `mem chat`). `POST /chat/stream` runs one
persisted multi-turn chat turn (NDJSON token events, then a final answer event carrying the
conversation id), respecting `CMRAG_SAVE_HISTORY` and a per-request `save` flag. `PATCH /conversations/{id}` renames one conversation, and `DELETE /conversations/{id}` deletes one (the API
twins of `mem history --rename` and `mem history --clear --id <id>`; the web sidebar exposes both
behind a per-row menu, with delete requiring a second confirming click). Everything remains wipeable
with `mem history --clear`. The web console is served at `/`, and interactive API docs are at
`/docs`. Use `--host` or `--port` to change the bind; binding to a non-loopback `--host` exposes the
unauthenticated API on your network and is warned against.

Loopback with no auth is safe against the network but not against the browser, so the API refuses
two classes of request before they reach a handler. It answers only to a loopback `Host`
(`localhost`, `127.0.0.1`, `::1`, any port), which blocks DNS rebinding: without that check, a name
the attacker controls that resolves to `127.0.0.1` would make their page same-origin with your
memory store and able to read the replies. It also refuses cross-site browser requests, identified
by `Sec-Fetch-Site`, which blocks the drive-by that no CORS policy can stop: `<img src>` and
`fetch(…, {mode: 'no-cors'})` reach a loopback service whatever the response headers say, and each
one spends a local inference run on someone else's query. Clicking a link to the console still
works, and curl, scripts and the CLI send no `Sec-Fetch-*` headers and are unaffected. Set
`CMRAG_API_ALLOWED_HOSTS` to the names you will reach it by when you deliberately bind
non-loopback; refused requests get a 403 explaining which check fired.

`/ask/stream` is the streaming variant of `/ask`: NDJSON events (`{"type":"token","text":…}` per
generated fragment), then one final `{"type":"answer","data":…}` carrying the exact `/ask` payload.
Citations are validated on the full output, and the final event always arrives, including on
abstention or LLM failure.

```bash
curl -s localhost:8765/health
curl -s "localhost:8765/ask?q=why+did+I+change+the+parser&use_llm=true"
curl -sN "localhost:8765/ask/stream?q=why+did+I+change+the+parser"   # NDJSON token stream
```

The web console is a React app under `web/`, built to `src/crossmodalrag/api/static/` (committed, so
`mem serve` needs no Node). To develop it: `cd web && npm install && npm run dev` (proxies to a running
`mem serve`); `npm run build` refreshes the shipped bundle. See `web/README.md`.

### MCP server for AI assistants (`mem mcp`)

`mem mcp` lets an AI assistant query your memory store directly. It serves the read views as
Model Context Protocol tools over the process's stdin and stdout, so Claude Code, Claude Desktop,
or any other MCP client can ground its answers in your notes, commits, PDFs, and images. There is
no network listener, no auth to configure, and nothing leaves the machine. It requires the opt-in
`[mcp]` extra (`pip install -e ".[mcp]"`, adds the official `mcp` SDK); without it, `mem mcp` exits
with an install hint.

```bash
pip install -e ".[mcp]"
claude mcp add engram -- mem mcp --db /absolute/path/to/data/memory.db   # Claude Code
```

For Claude Desktop, add the same command to `claude_desktop_config.json` (use the absolute path of
the `mem` binary in your virtualenv):

```json
{
  "mcpServers": {
    "engram": {
      "command": "/absolute/path/to/.venv/bin/mem",
      "args": ["mcp", "--db", "/absolute/path/to/data/memory.db"],
      "env": {"CMRAG_LLM_MODEL": "gemma4"}
    }
  }
}
```

Pass `--db` (or set `CMRAG_DB_PATH` in the client's `env`). The client chooses the server's
working directory, and Claude Desktop uses `/`, so neither `./data/memory.db` nor a project `.env`
resolves the way they do at your shell. Any other setting you rely on (`CMRAG_LLM_MODEL`,
`CMRAG_LLM_BASE_URL`, a `CMRAG_CONFIG` path) goes in that `env` block for the same reason. A
database that does not exist is refused with a message naming the path rather than created empty.

The tools are `ask`, `concepts`, `timeline`, `forgetting`, `recall`, `history`, `conversation`
(one saved conversation by id), and `status` (the `mem doctor` report). Each returns exactly the
corresponding `--json` payload as structured content, so every `ask` evidence item carries
`evidence_id`, `source_uri`, `locator` (`spec.pdf p.4`, `repo@sha`), and `chunk_id`, and an
abstention is reported as `abstained: true` with its reason. Every tool is marked read-only. `ask`
never records usage. The one write is `recall`'s card cache (`recall_cards`), the same derived
cache `mem recall` and `GET /recall` fill; it never touches ingested or chat-history data. `ask`
with `use_llm` on calls the local Ollama model and can take up to `CMRAG_LLM_TIMEOUT` seconds;
`use_llm: false` returns the ranked evidence without synthesis, needs no model, and is the fast
path. Bad arguments (an unknown level or profile) are rejected by the tool schema. Logs go to
stderr; stdout is the protocol channel.

## Hierarchical Memory (experimental)

Beyond flat evidence retrieval, Engram can build higher-level memory layers on top of the
L0 evidence chunks: L1 atomic events → L2 episodes → L3 concepts. Every higher-level node is
traceable down to its L0 evidence.

Currently implemented: the node and edge substrate plus L1 atomic-event extraction, L2 episode
grouping, and L3 concept clustering. `mem build-memory` derives all three:

- **L1 events** (requires Ollama): a local LLM extracts atomic events ("what happened": a decision,
learning, fix, task, or change) from each source, including PDF pages and OCR'd images, linking
each event to its L0 evidence. The hierarchy therefore spans modalities and drills back to a cited
page or region locator. Before extracting, `build-memory` deterministically re-anchors existing
events whose evidence links point at chunk ids re-issued by a re-chunk (`mem sync` after a chunker
upgrade). Extraction skips text-identical sources, so their events are re-linked to the source's
current chunks instead, with no LLM involved, and `mem memory-stats` integrity comes back clean.
Events whose source no longer exists are reported, never silently deleted.
Model replies that are *almost* JSON (unquoted values or keys, LaTeX escapes, truncated arrays)
are repaired syntactically rather than discarded. Sources whose output still can't be parsed
are listed by id and path in the command output, then retried on the next run.
- **L2 episodes** (no LLM): events are grouped into "sessions of related work" by project (git repo,
or containing folder for notes, PDFs, and images) and time gap. A new episode starts when
consecutive events in a project are more than `CMRAG_EPISODE_GAP_HOURS` apart (default 24). A folder
mixing notes, PDFs, and screenshots forms one cross-modal episode. Each episode links to its members.
- **L3 concepts** (requires the `[embeddings]` extra; LLM naming optional): events are clustered by
semantic similarity (cosine ≥ `CMRAG_CONCEPT_SIM_THRESHOLD`, default 0.80) into recurring topics
that span episodes. Each new concept is named by the local LLM (`CMRAG_EXTRACT_MODEL`, temp 0)
from a representative sample of member events; a response that isn't a short topic label (or an
unavailable Ollama) falls back to a deterministic name.
- **Graph** (no LLM or embeddings): computes PageRank centrality per node (an importance signal
stored on each node) and concept co-occurrence links. Two concepts get a `relates_to` edge
(weighted by shared-episode count) when they have events in the same episode.
- **Drift** (no LLM; needs the `[embeddings]` extra): buckets each concept's member events into
time windows and scores how far the concept's prototype (centroid) moved between windows; surfaced
via `mem drift`. See the `mem drift` bullet above.
- **Distill** (needs the `[embeddings]` extra; LLM summary optional): derives a compact,
retrieval-preserving stand-in for each L2 or L3 node. A stand-in is a short summary (plus its
embedding) and a minimal subset of the node's real L0 evidence chunks, namely the most
representative ones, sized to `CMRAG_DISTILL_COMPRESSION_RATIO`. This is a research and measurement
feature: it does not change
`mem ask` ranking. Whether a distilled stand-in is good enough to adopt is decided by
`scripts/distill_gate.py` (below), not by default. Provenance is preserved, because the kept chunks
are a real subset rather than a generated paraphrase.

Every higher-level node drills down to its L0 evidence through its members.

```bash
mem build-memory --limit 50          # L1 events (up to 50 sources) + L2 episodes + L3 concepts + graph + drift
mem build-memory --level event       # only L1 events (LLM)
mem build-memory --level episode     # only L2 episodes (no Ollama needed)
mem build-memory --level concept     # only L3 concepts (needs the embeddings extra; run reindex-embeddings first)
mem build-memory --level graph       # only centrality + concept co-occurrence (no LLM/embeddings)
mem build-memory --level drift       # only concept-drift snapshots (needs the embeddings extra; build concepts first)
mem build-memory --level distill     # only distilled node stand-ins (needs the embeddings extra; build concepts first)
mem memory-stats                     # node/edge counts, co-occurrence edges, top central nodes, integrity
```

**Distillation gate (research and measurement).** `scripts/distill_gate.py` measures whether the
distilled stand-ins preserve retrieval quality against the full nodes, under a pre-committed budget
(`CMRAG_DISTILL_EPSILON`, default 0.05 max Recall@K loss; `CMRAG_DISTILL_COMPRESSION_RATIO`, default
0.5 target footprint). It prints `GATE: FIRE` only when recall is preserved within ε and the size
target is met. Run it on an embedded corpus after building concepts and distilling:

```bash
mem build-memory --db-path <db>                       # events/episodes/concepts (Ollama)
CMRAG_DB_PATH=<db> mem reindex-embeddings             # needs the [embeddings] extra
CMRAG_DB_PATH=<db> mem build-memory --level distill   # derive the distilled stand-ins
python scripts/distill_gate.py --db-path <db>         # full vs distilled Recall@K, ratio, FIRE/HOLD
```

Adopting distillation in the live retrieval path is a separate, explicitly-gated decision. The gate
must fire on a representative corpus first.

**Date-aware note dates.** A note's timestamp drives time-aware layers (episodes, drift). If a note
declares an explicit date (YAML frontmatter `date:` or `created:`, or a leading `Date: YYYY-MM-DD`
line near the top), that date becomes its timestamp; otherwise the file's modification time is used.
This keeps windowing deterministic across machines and checkouts.

Once built and embedded (`mem reindex-embeddings` now also embeds memory nodes), the hierarchy is
queryable: `mem ask --level concept|episode|event` retrieves at that level and answers grounded in
the drilled-down L0 evidence, and `mem concepts` and `mem timeline` browse the concept and episode
layers. Node ranking blends semantic, lexical, recency, and centrality signals.

All layers are deterministic and incremental: L1 sources are re-extracted only when content,
model, or prompt version changes; L2 and L3 are reconciled by membership, so re-running on unchanged
data is a no-op (and concepts are not re-named). L1 uses `CMRAG_EXTRACT_MODEL` (default `llama3.2`,
separate from the synthesis model so bulk extraction stays fast); `--model` overrides per run.

## Synthetic Sample Seed Workflow

Use `mem seed-sample` to create a tiny deterministic sample vault + sample git repo and ingest them into an isolated sample DB.

- Creates a local workspace at `./data/sample-seed-workspace` by default
- Writes to a separate temp sample DB by default (does not modify your main `./data/memory.db`)
- Seeds synthetic markdown notes and git commits (no personal data)
- Populates namespaced sample rows in `queries_eval`: `[sample]` single-source specific-fact
queries (including one negative case that should abstain), `[sample-synth]` multi-source synthesis
queries, `[sample-xmodal-text]` and `[sample-xmodal-visual]` cross-modal slices, and a
`[sample-drift]` slice. The drift slice holds a deliberately drifting "chunking" concept, built
from two notes whose approach shifts across time windows, plus a stable provenance control
- Materializes tiny synthetic cross-modal fixtures (a 1-page PDF + two PNGs) under the sample vault.
The PDF is ingested when the `[pdf]` extra is present and the images when `[ocr]` (+ tesseract) is
present; otherwise those slices stay at a ~0 baseline. The two slices are designed so an
strategy that reads OCR or PDF text first answers the *text-heavy* slice but fails the *visual-heavy* one (a
layout or colour-only diagram), which is the gap the pre-committed native-embedding gate measures.
Read the gate semantically, since it compares against semantic image embeddings:
`mem reindex-embeddings` first, then `python scripts/xmodal_gate.py` (it warns if no vectors are
stored). Regenerate the fixtures with `scripts/generate_xmodal_fixtures.py`.
- Safe to re-run; unchanged content is reused and ingestion remains idempotent

Run the sample retrieval benchmark:

```bash
mem eval --query-prefix "[sample]" --top-k 5
```

Use `--force` to rebuild the sample workspace directory from scratch.
Use `--db-path` if you want the sample dataset in a specific non-main database path.

## Data Location

By default, data is stored at:

- `./data/memory.db`

Set `CMRAG_DB_PATH` to override:

```bash
export CMRAG_DB_PATH=/absolute/path/to/memory.db
```

Select the embedding model (defaults to `BAAI/bge-small-en-v1.5`, 384-dim):

```bash
export CMRAG_EMBED_MODEL=BAAI/bge-small-en-v1.5
```

Configure the local LLM used for answer synthesis (defaults shown):

```bash
export CMRAG_LLM_PROVIDER=ollama
export CMRAG_LLM_MODEL=gemma4              # swap for any model you have in `ollama list`
export CMRAG_LLM_BASE_URL=http://localhost:11434
export CMRAG_LLM_TIMEOUT=120
export CMRAG_LLM_KEEP_ALIVE=30m            # keep the model loaded between calls ("30m", "1h",
                                           # seconds, or -1 = until Ollama exits); cold-loads
                                           # dominate tail latency
export CMRAG_LLM_NUM_CTX=8192              # context window requested from Ollama; its 4096
                                           # default truncates evidence-heavy prompts and cuts
                                           # answers mid-generation (raise if warned; 0 = server default)
export CMRAG_MIN_EVIDENCE_SCORE=0.15       # abstain below this top retrieval score
export CMRAG_CHAT_CONTEXT_TURNS=8          # prior turns carried in `mem chat` (0 disables)
export CMRAG_SAVE_HISTORY=on               # save chat sessions locally (off disables; `mem history --clear` wipes)
export CMRAG_EXTRACT_MODEL=llama3.2        # model for `mem build-memory` event extraction
export CMRAG_EPISODE_GAP_HOURS=24          # L2 episode session gap (deterministic, no LLM)
export CMRAG_CONCEPT_SIM_THRESHOLD=0.80    # L3 concept clustering cosine threshold (embeddings extra)
```

Distillation and drift scaffolding (additive, opt-in; defaults shown):

```bash
export CMRAG_DRIFT_WINDOW_DAYS=30          # concept-drift snapshot window length (days)
export CMRAG_DISTILL_EPSILON=0.05          # max Recall@K a distilled rep may lose vs full nodes (gate)
export CMRAG_DISTILL_COMPRESSION_RATIO=0.5 # target distilled-size / full-size for the distillation gate
```

See `.env.example` for the full list of supported variables.

## Evaluation Query File Format (`mem eval --load-queries`)

Use a JSON array of rows:

```json
[
  {
    "query_text": "What fixed the parser bounds check bug?",
    "expected_source_uris": [
      "/abs/path/to/repo@abc123",
      "/abs/path/to/note/http-parser.md"
    ]
  }
]
```

`mem eval --load-queries file.json` upserts rows into `queries_eval` and then runs metrics (`Recall@K`, `MRR@K`, and an approximate citation hit-rate based on the top retrieved source).

Loading also validates each `expected_source_uris` entry, printing a warning to stderr (never
failing the load) for gold URIs that would silently break exact-match scoring. Three cases trigger a
warning: a doubled path segment such as `//Users/...`, a non-absolute path that isn't a `scheme://`
URI, or a URI that matches no ingested source. That last check is skipped when the DB has no sources
yet, since a query file may legitimately predate ingestion. Warnings go to stderr so `--json` output
stays a clean contract.