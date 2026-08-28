"""Browser-facing guards for the local API (the loopback threat model).

`mem serve` binds 127.0.0.1 with no auth, which is safe against the network but not against
the browser: any page the user has open can address the loopback service directly, and the
absence of CORS headers only stops the attacker from *reading* the reply. Two request headers
close the two paths that survive that, and a page cannot forge either — the browser sets them.

``Host`` closes DNS rebinding. A name the attacker controls that resolves to 127.0.0.1 makes
their page same-origin with the API, at which point CORS stops applying and the responses
(the user's memory) become readable. Rejecting any ``Host`` that is not a loopback name means
a rebound request never reaches a handler.

``Sec-Fetch-Site`` closes the plain drive-by. ``<img src="…/ask?q=…">`` and
``fetch(…, {mode: 'no-cors'})`` are simple cross-origin GETs: no preflight, nothing for the
missing CORS policy to block, and each one spends a local LLM inference run on an
attacker-chosen query. Fetch metadata is the only signal that distinguishes them, because a
no-cors GET carries no ``Origin`` header at all.

Non-browser clients (curl, the CLI, scripts) send no ``Sec-Fetch-*`` headers, so they pass the
second check untouched and are gated only on ``Host``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

#: The names a browser can put in ``Host`` for a loopback bind. Ports are irrelevant here:
#: rebinding needs an attacker-controlled *name*, and requiring a port would break the Vite
#: dev proxy, which forwards the browser's own ``localhost:5173``.
LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})

#: ``Sec-Fetch-Site`` values that are not cross-site: the console's own requests
#: (``same-origin``) and user-initiated ones such as a typed URL or a bookmark (``none``).
_SAFE_FETCH_SITES = frozenset({"same-origin", "same-site", "none"})

#: Methods allowed to arrive as a cross-site top-level navigation (a clicked link). Anything
#: else on that path is a cross-site form submit.
_SAFE_METHODS = frozenset({"GET", "HEAD"})


def normalize_hostname(host_header: str) -> str:
    """The hostname from a ``Host`` header: lowercased, port and IPv6 brackets removed."""
    host = host_header.strip().lower()
    if host.startswith("["):  # bracketed IPv6, with or without a port: [::1] / [::1]:8765
        end = host.find("]")
        return host[1:end] if end != -1 else host[1:]
    if host.count(":") > 1:  # bare IPv6 literal — every colon belongs to the address
        return host
    return host.split(":", 1)[0]


def reject_reason(
    *,
    method: str,
    headers: Mapping[str, str],
    allowed_hostnames: frozenset[str],
) -> str | None:
    """Why this request must not reach a handler, or ``None`` to let it through.

    ``headers`` must be lowercase-keyed. ``allowed_hostnames`` containing ``"*"`` disables the
    host check only; the fetch-metadata check always applies.
    """
    host = headers.get("host", "")
    if "*" not in allowed_hostnames and normalize_hostname(host) not in allowed_hostnames:
        return (
            f"Refused: Host header {host!r} is not a host this local API answers to "
            f"({', '.join(sorted(allowed_hostnames))}). Reach it on its loopback address, or "
            "set CMRAG_API_ALLOWED_HOSTS for a deliberate non-loopback bind."
        )

    site = headers.get("sec-fetch-site")
    if site is None or site in _SAFE_FETCH_SITES:
        return None
    is_top_level_navigation = (
        headers.get("sec-fetch-mode") == "navigate" and headers.get("sec-fetch-dest") == "document"
    )
    if is_top_level_navigation and method.upper() in _SAFE_METHODS:
        return None
    return (
        f"Refused: {site} request to a local-only API. Engram answers the console it serves "
        "and non-browser clients, not requests made by other pages you have open."
    )


class LocalOriginGuard:
    """ASGI middleware applying :func:`reject_reason` ahead of routing.

    Written against the raw ASGI interface rather than Starlette's ``BaseHTTPMiddleware`` so a
    permitted request is passed through byte-for-byte, leaving the NDJSON streaming responses
    (`/ask/stream`, `/chat/stream`) and their disconnect handling exactly as they were.
    """

    def __init__(self, app, *, allowed_hosts: Iterable[str] | None = None) -> None:
        self.app = app
        names = LOOPBACK_HOSTNAMES if allowed_hosts is None else allowed_hosts
        self.allowed_hostnames = frozenset(n.strip().lower() for n in names if n and n.strip())

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        reason = reject_reason(
            method=scope.get("method", "GET"),
            headers=headers,
            allowed_hostnames=self.allowed_hostnames,
        )
        if reason is None:
            await self.app(scope, receive, send)
            return
        await _forbidden(send, reason)


async def _forbidden(send, reason: str) -> None:
    """A 403 in FastAPI's own error shape, so clients parse it the same way."""
    import json

    body = json.dumps({"detail": reason}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
