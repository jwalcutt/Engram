from __future__ import annotations

import os
import re
from pathlib import Path


def load_dotenv(dotenv_path: Path | None = None) -> None:
    path = dotenv_path or (Path.cwd() / ".env")
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def get_db_path() -> Path:
    raw = os.getenv("CMRAG_DB_PATH")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd() / "data" / "memory.db"


def get_llm_provider_name() -> str:
    return os.getenv("CMRAG_LLM_PROVIDER", "ollama").strip() or "ollama"


def get_llm_model() -> str:
    return os.getenv("CMRAG_LLM_MODEL", "gemma4").strip() or "gemma4"


def get_llm_base_url() -> str:
    return os.getenv("CMRAG_LLM_BASE_URL", "http://localhost:11434").strip().rstrip("/")


def get_llm_timeout() -> float:
    raw = os.getenv("CMRAG_LLM_TIMEOUT", "120").strip()
    try:
        return float(raw)
    except ValueError:
        return 120.0


def get_llm_keep_alive() -> float | str:
    """How long Ollama keeps the synthesis model loaded after a call. Default "30m".

    Cold-loading the model between calls dominated tail latency (30 s vs 18 min for
    near-identical prompts), so requests pin it for a while by default. Accepts an
    Ollama duration string ("30m", "1h"), a number of seconds, or -1 to keep the
    model loaded until Ollama shuts down.
    """
    raw = os.getenv("CMRAG_LLM_KEEP_ALIVE", "30m").strip()
    if not raw:
        return "30m"
    try:
        return float(raw)
    except ValueError:
        return raw


def get_llm_num_ctx() -> int:
    """Context window (tokens) requested from Ollama for every LLM call. Default 8192.

    Ollama's own server default (4096) is too small for evidence-heavy synthesis
    prompts: it silently truncates the prompt head (destroying the system prompt)
    and stops generation when the window fills, yielding empty or mid-sentence-cut
    answers. 0 defers to the server default (not recommended).
    """
    raw = os.getenv("CMRAG_LLM_NUM_CTX", "8192").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 8192


def get_extract_model() -> str:
    return os.getenv("CMRAG_EXTRACT_MODEL", "llama3.2").strip() or "llama3.2"


def get_episode_gap_seconds() -> int:
    raw = os.getenv("CMRAG_EPISODE_GAP_HOURS", "24").strip()
    try:
        hours = float(raw)
    except ValueError:
        hours = 24.0
    return int(hours * 3600)


def get_concept_sim_threshold() -> float:
    """Cosine threshold for L3 concept clustering. Default 0.80.

    bge-class embeddings are anisotropic: cosines between *unrelated* short texts
    commonly sit near 0.6, so a 0.60 threshold chains the whole event set into a
    couple of mega-clusters. Measured on a 2.5k-event corpus, 0.80 yields coherent
    single-topic concepts (max cluster ~2% of events) while covering ~70% of
    events; unclustered events remain retrievable at L1/L0.
    """
    raw = os.getenv("CMRAG_CONCEPT_SIM_THRESHOLD", "0.80").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.80


def get_usage_halflife_days() -> float:
    """Half-life (days) for the usage rehearsal-strength decay. Default 30."""
    raw = os.getenv("CMRAG_USAGE_HALFLIFE_DAYS", "30").strip()
    try:
        value = float(raw)
    except ValueError:
        return 30.0
    return value if value > 0 else 30.0


def get_usage_saturation() -> float:
    """Half-saturation point mapping usage strength into [0,1) for the `usage` profile. Default 3.0."""
    raw = os.getenv("CMRAG_USAGE_SATURATION", "3").strip()
    try:
        value = float(raw)
    except ValueError:
        return 3.0
    return value if value > 0 else 3.0


def usage_tracking_enabled() -> bool:
    """Whether `mem ask` logs interaction events to usage_events. OFF by default (opt-in).

    Privacy/local-first: only target id + event type + time are stored (never query text), and
    history is clearable via `mem usage --clear`. Enable with CMRAG_USAGE_TRACKING=on (or per-call
    `mem ask --track` / `--accept`).
    """
    return os.getenv("CMRAG_USAGE_TRACKING", "off").strip().lower() in {"on", "1", "true", "yes"}


def save_history_enabled() -> bool:
    """Whether interactive chat sessions persist to conversations/messages. ON by default.

    Privacy: this stores raw query/answer text — LOCAL-ONLY, never sent anywhere. Disable
    with CMRAG_SAVE_HISTORY=off (or per-session `--no-save`); wipe any time with
    `mem history --clear`. One-shot `mem ask "<q>"` and eval/scripted paths never write
    history regardless of this setting.
    """
    return os.getenv("CMRAG_SAVE_HISTORY", "on").strip().lower() not in {"off", "0", "false", "no"}


def get_api_allowed_hosts() -> list[str]:
    """Extra hostnames the local API (`mem serve`) will answer to, beyond loopback.

    The API rejects any request whose `Host` is not `localhost`, `127.0.0.1` or `::1`, which is
    what stops a DNS-rebinding page from becoming same-origin with the loopback service. Binding
    a non-loopback `--host` needs the names you will actually reach it by:
    `CMRAG_API_ALLOWED_HOSTS=engram.lan,192.168.1.5` (comma-separated; `*` disables the check).
    """
    raw = os.getenv("CMRAG_API_ALLOWED_HOSTS", "").strip()
    return [part.strip() for part in raw.split(",") if part.strip()]


def get_drift_window_days() -> float:
    """Window length (days) for concept-drift snapshots. Default 30."""
    raw = os.getenv("CMRAG_DRIFT_WINDOW_DAYS", "30").strip()
    try:
        value = float(raw)
    except ValueError:
        return 30.0
    return value if value > 0 else 30.0


def get_distill_epsilon() -> float:
    """Max Recall@K a distilled representation may lose vs the full nodes. Default 0.05."""
    raw = os.getenv("CMRAG_DISTILL_EPSILON", "0.05").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.05
    return value if value >= 0 else 0.05


def get_distill_compression_ratio() -> float:
    """Target distilled-size / full-size for the distillation gate (0 < x <= 1). Default 0.5."""
    raw = os.getenv("CMRAG_DISTILL_COMPRESSION_RATIO", "0.5").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.5
    return value if 0 < value <= 1.0 else 0.5


def get_title_boost_weight() -> float:
    """Additive bonus weight for query-token overlap with the source *title*. Default 0.05.

    Small by design: it re-orders near-ties in favor of sources literally named for
    the query's terms (a note titled "Fourier Transform" over an incidental mention
    in a diff) without overriding semantic relevance. 0 disables the boost.
    """
    raw = os.getenv("CMRAG_TITLE_BOOST_WEIGHT", "0.05").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.05
    return value if value >= 0 else 0.05


def get_min_evidence_score() -> float:
    raw = os.getenv("CMRAG_MIN_EVIDENCE_SCORE", "0.15").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.15


def get_chat_context_turns() -> int:
    """Max prior (question, answer) turns carried into each interactive-ask
    turn's prompt. Oldest turns are evicted deterministically past the cap;
    0 disables carried context entirely."""
    raw = os.getenv("CMRAG_CHAT_CONTEXT_TURNS", "8").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 8


def get_numbered_env_paths(prefix: str) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    indexed: list[tuple[int, Path]] = []
    for key, raw in os.environ.items():
        match = pattern.match(key)
        if not match:
            continue
        value = raw.strip()
        if not value:
            continue
        indexed.append((int(match.group(1)), Path(value).expanduser().resolve()))
    indexed.sort(key=lambda item: item[0])
    return [path for _, path in indexed]


# --- Optional TOML config file ------------------------------------------------------------
# A `crossmodalrag.toml` (or `$CMRAG_CONFIG`) supplies connector paths + retrieval defaults.
# Precedence is always: explicit CLI flag > environment/.env > config file > built-in default.
# The file is optional and never required; a missing/malformed file degrades to "{}" silently.

# Connector name -> the numbered env-var prefix it falls back from.
CONNECTOR_ENV_PREFIX: dict[str, str] = {
    "notes": "OBSIDIAN_VAULT_PATH",
    "git": "REPO_PATH",
    "pdf": "PDF_PATH",
    "image": "IMAGE_PATH",
}


def get_config_path() -> Path | None:
    """Resolve the active config file: ``$CMRAG_CONFIG`` if set, else ``./crossmodalrag.toml``.

    Returns the path only when it exists (for `$CMRAG_CONFIG`, the explicit path is returned even if
    missing so `doctor` can report a misconfiguration); otherwise None.
    """
    raw = os.getenv("CMRAG_CONFIG")
    if raw and raw.strip():
        return Path(raw).expanduser()
    default = Path.cwd() / "crossmodalrag.toml"
    return default if default.exists() else None


def load_config() -> dict:
    """Parse the TOML config file into a dict (``{}`` when absent, unreadable, or malformed)."""
    path = get_config_path()
    if path is None or not path.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - 3.11+ per requires-python
        return {}
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def get_connector_paths(name: str) -> list[Path]:
    """Resolved ingestion paths for a connector: environment first, else the config file."""
    env_paths = get_numbered_env_paths(CONNECTOR_ENV_PREFIX[name])
    if env_paths:
        return env_paths
    raw = load_config().get("connectors", {}).get(name, [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [Path(str(p)).expanduser().resolve() for p in raw if str(p).strip()]


def get_default_profile(builtin: str) -> str:
    """Default retrieval profile: config `[retrieval].profile` if set, else ``builtin``."""
    value = load_config().get("retrieval", {}).get("profile")
    return value if isinstance(value, str) and value.strip() else builtin


def get_default_top_k(builtin: int = 5) -> int:
    """Default retrieval top-k: config `[retrieval].top_k` if a positive int, else ``builtin``."""
    value = load_config().get("retrieval", {}).get("top_k")
    return value if isinstance(value, int) and value > 0 else builtin
