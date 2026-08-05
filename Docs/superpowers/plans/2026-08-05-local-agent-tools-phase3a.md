# Local Agent Tools — Phase 3a (Research Tools) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the research cluster to the Console agent catalog: `web_fetch` (ported from tldw_server with a purpose-built SSRF guard), `web_search` (migrated from the legacy tool), and `todo_write` (session-scoped todos with transcript rendering) — plus a tool-discovery hint in the agent system prompt.

**Architecture:** New sync-core module `Tools/web_tool_impls.py` (SSRF guard + fetcher + rate limiter + in-memory TTL cache); `web_search` spec delegating to `Web_Scraping/WebSearch_APIs.perform_websearch`; `todo_write` state held per Console session with the provider receiving a per-run store + change callback at composition time (the provider stays Textual-free and context-free per-call). All register as `LocalToolSpec`s per ADR-032.

**Tech Stack:** Python ≥3.11, httpx (project dep), trafilatura (project dep), pytest.

**Specs:** `Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md` (phase 3a) + base spec `2026-08-04-local-agent-tools-design.md` §3 (approval discipline)
**ADRs:** 032 (permission boundary), 033 (process boundary — not exercised this phase)

**Reference source for ports:** tldw_server @ `5605b9d9906322c2e6b5342b48c391ae674d315e`. A sparse clone may exist at `/tmp/tldw_server_mcp/tldw_server`; if missing: `git clone --depth 1 --filter=blob:none --sparse https://github.com/rmusser01/tldw_server /tmp/tldw_server_ref && cd /tmp/tldw_server_ref && git sparse-checkout set tldw_Server_API/app/core/MCP_unified/modules/implementations tldw_Server_API/app/core/Web_Scraping` then reference `tldw_Server_API/app/core/MCP_unified/modules/implementations/{web_fetch_module,web_tool_base,web_rate_limit,web_cache,web_search_module}.py` and `tldw_Server_API/app/core/Web_Scraping/outbound_policy.py`. **Attribution (binding per re-plan §5):** every ported file gets a header comment with source repo, source path, and that commit SHA.

---

## Verified facts (from phases 1-2 — do not re-derive)

- `LocalToolSpec{name, description, parameters, handler, tags}`; handlers are sync `Callable[[dict], str]` raising `LocalToolError`; provider byte-fits results (32 KiB) and converts all exceptions to error strings (`Agents/local_tool_provider.py`).
- New specs register in `_default_specs(workspace_root)`; catalog ids `local:<name>`; the catalog exact-id test in `Tests/Agents/test_local_tool_provider.py` must be extended per new tool.
- Network tools (`web_fetch`, `web_search`) are network-classed: default `ask` per the global permission default; NO risk tags (they don't mutate); `todo_write` carries `tags=("mutates",)`.
- `_compose_local_provider` (`Chat/console_chat_controller.py:~1000`) builds the provider per run — session-scoped seams (todo store) are injected there.
- `perform_websearch` (`Web_Scraping/WebSearch_APIs.py:1137`) is sync, ~15 positional args with a FIXME; the wrapper supplies config defaults.
- trafilatura usage precedent: `Web_Scraping/Article_Extractor_Lib.py:538` (`trafilatura.extract(html, include_comments=False, include_tables=False, include_images=False)`).
- `ConsoleChatSession` (`Chat/console_chat_store.py:123-134`) is a mutable dataclass — add the todo field there.
- Tests: cores in `Tests/Tools/`, provider in `Tests/Agents/`, integration in `Tests/Agents/test_local_tools_integration.py`; use `ws = tmp_path/"ws"` for workspace fixtures (autouse fixture pollutes tmp_path); run with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`.
- Known pre-existing failures to deselect: `Tests/Chat/test_anthropic_native_tools.py::test_anthropic_shaped_tools_pass_through_untouched`, `Tests/Utils/test_github_api_client.py::TestGitHubAPIClient::test_client_property_without_token`.

---

## Task 0: Backlog task

- [ ] **Step 1:** Create via CLI (fallback: markdown) — title "Local agent tools phase 3a: research tools (web_fetch/web_search/todo_write)". ACs:
  1. web_fetch refuses private/loopback/link-local targets and non-http(s) schemes, including on redirect hops
  2. web_fetch enforces redirect cap, timeout, byte caps, per-domain rate limit, and TTL cache
  3. web_search delegates to perform_websearch with bounded per-result size
  4. todo_write mutates per-session state and renders in the transcript
  5. Agent system prompt hints at find_tools/load_tools discovery
  6. All new tests pass
  Plan pointer: this file. Then `git add backlog/ && git commit -m "docs: create phase-3a backlog task"`.

---

## Task 1: SSRF guard

**Files:**
- Create: `tldw_chatbook/Tools/web_tool_impls.py` (starts here; attribution header: inspired by / checklist from tldw_server `tldw_Server_API/app/core/Web_Scraping/outbound_policy.py` @ 5605b9d9 — written fresh per re-plan §2.1, not a line port)
- Test: `Tests/Tools/test_web_tool_impls.py`

- [ ] **Step 1: Failing tests**

```python
import pytest
from tldw_chatbook.Tools.web_tool_impls import LocalToolError, validate_outbound_url

def test_accepts_public_https():
    assert validate_outbound_url("https://example.com/page") == "https://example.com/page"

def test_rejects_bad_schemes():
    for url in ("file:///etc/passwd", "ftp://x/y", "gopher://x", "javascript:alert(1)", "data:text/html,hi"):
        with pytest.raises(LocalToolError):
            validate_outbound_url(url)

def test_rejects_loopback_and_private_literals():
    for url in ("http://127.0.0.1/", "http://localhost/", "http://10.0.0.5/", "http://172.16.0.1/",
                "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data", "http://[::1]/",
                "http://0.0.0.0/"):
        with pytest.raises(LocalToolError):
            validate_outbound_url(url)

def test_rejects_private_dns_answer(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("10.1.2.3", 80))])
    with pytest.raises(LocalToolError, match="private|internal|not allowed"):
        validate_outbound_url("http://evil.internal.example.com/")

def test_rejects_unresolvable_host(monkeypatch):
    import socket
    def boom(*a, **k): raise socket.gaierror("no")
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(LocalToolError):
        validate_outbound_url("http://does-not-exist.invalid/")
```

- [ ] **Step 2: Verify failure** (ModuleNotFoundError)

- [ ] **Step 3: Implement** `validate_outbound_url(url: str) -> str` in `web_tool_impls.py`:

```python
"""Sync cores for web_* agent tools.

The SSRF guard below is written fresh for tldw_chatbook, using tldw_server's
tldw_Server_API/app/core/Web_Scraping/outbound_policy.py @ 5605b9d9906322c2e6b5342b48c391ae674d315e
(https://github.com/rmusser01/tldw_server, GPL-3.0-only) as the requirements
checklist — see re-plan spec §2.1 (2026-08-05).
"""

import ipaddress
import socket
from urllib.parse import urlsplit

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DNS_CACHE_TTL_SECONDS = 300.0


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def validate_outbound_url(url: str) -> str:
    """Return ``url`` if it's safe to fetch; raise LocalToolError otherwise.

    Checks: scheme allowlist (http/https), host resolves, and EVERY resolved
    IP is public (loopback/private/link-local/reserved refused). Called for
    the initial URL AND every redirect hop (DNS-rebinding window lives
    between hops). Reuses LocalToolError from local_tool_impls.
    """
    from .local_tool_impls import LocalToolError  # or module-level import

    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise LocalToolError(f"URL scheme not allowed (http/https only): {url!r}")
    host = parts.hostname
    if not host:
        raise LocalToolError(f"URL has no host: {url!r}")
    try:
        ipaddress.ip_address(host)  # literal IP: check directly
        candidates = [host]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80),
                                       proto=socket.IPPROTO_TCP)
        except (socket.gaierror, UnicodeError, OSError) as exc:
            raise LocalToolError(f"host does not resolve: {host!r}") from exc
        candidates = [info[4][0] for info in infos]
    if not candidates or not all(_is_public_ip(ip) for ip in candidates):
        raise LocalToolError(f"host resolves to a private/internal address: {host!r}")
    return url
```

(DNS rebinding caveat: resolution happens again inside httpx; the guard re-checks every redirect hop, which bounds the window. Document this in the docstring. Do NOT attempt to pin the connection IP — that's the heavy version, not needed here.)

- [ ] **Step 4:** tests pass
- [ ] **Step 5:** `git commit -m "feat: SSRF guard for outbound web tools"`

---

## Task 2: `web_fetch` core

**Files:**
- Modify: `tldw_chatbook/Tools/web_tool_impls.py`
- Test: `Tests/Tools/test_web_tool_impls.py`

Reference (read before writing): tldw_server `web_fetch_module.py` + `web_tool_base.py` + `web_rate_limit.py` + `web_cache.py` at the pinned SHA. Port the *behaviors*, not the code shape.

- [ ] **Step 1: Failing tests** (use `httpx.MockTransport`):

```python
def test_fetch_extracts_text(transport): ...   # HTML in -> extracted text out, byte-fitted
def test_fetch_validates_each_redirect_hop(transport):
    # hop 1 public 302 -> http://169.254.169.254/ must raise LocalToolError
def test_fetch_redirect_cap(transport): ...    # 6 chained redirects -> error "redirect"
def test_fetch_byte_cap_sets_truncated(transport): ...  # >1MB body -> 1MB + "truncated" marker
def test_fetch_rate_limits_per_domain(transport, monkeypatch): ...
def test_fetch_caches_within_ttl(transport): ...  # 2nd call hits cache (transport called once)
```

- [ ] **Step 2: Implement** in `web_tool_impls.py`:

```python
FETCH_MAX_REDIRECTS = 5
FETCH_TIMEOUT_SECONDS = 30.0
FETCH_MAX_BYTES = 1 * 1024 * 1024          # default cap
FETCH_HARD_MAX_BYTES = 5 * 1024 * 1024     # absolute ceiling for max_bytes arg
FETCH_CACHE_TTL_SECONDS = 900.0
RATE_LIMIT_INTERVAL_SECONDS = 1.0          # per-domain min interval

_fetch_cache: dict[str, tuple[float, str]] = {}
_domain_last_fetch: dict[str, float] = {}


def web_fetch(url: str, *, max_bytes: int = FETCH_MAX_BYTES) -> str:
    """Fetch ``url`` and return extracted text (trafilatura), byte-capped.

    SSRF-guarded per hop (validate_outbound_url), redirect-capped,
    rate-limited per domain, cached in-memory for FETCH_CACHE_TTL_SECONDS.
    Result ends with a truncation marker when capped. All failures raise
    LocalToolError with structured reasons ("invalid-url", "ssrf",
    "redirect-limit", "timeout", "http-<status>", "too-large", "rate-limited").
    """
```

Implementation notes: manual redirect following (httpx `follow_redirects=False`, loop with per-hop `validate_outbound_url`); read response streaming with a byte cap (`response.iter_bytes` bound, or read then cap — bounded reads preferred); rate limiter = min interval per `urlsplit(url).hostname` (sleep the remainder; tests monkeypatch `time.monotonic`/`time.sleep`); cache keyed by url. `max_bytes` arg clamped to `FETCH_HARD_MAX_BYTES`. Keep the module's cache/rate-limit dicts module-level but expose a `_reset_state_for_tests()` helper.

- [ ] **Step 3:** tests pass
- [ ] **Step 4:** `git commit -m "feat: web_fetch core with SSRF guard, caps, rate limit, cache"`

---

## Task 3: `web_fetch` + `web_search` specs

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (`_default_specs`)
- Test: `Tests/Agents/test_local_tool_provider.py`

- [ ] **Step 1: Failing tests** — catalog includes `local:web_fetch` and `local:web_search`; both `tags==()`; web_fetch schema requires `url` (max_bytes optional int); web_search schema requires `query`. A web_search handler test with `perform_websearch` monkeypatched: returns bounded text (each result ≤ ~4 KiB), error from the backend becomes a result-string.

- [ ] **Step 2: Implement.**
  - `web_fetch` spec → handler calls `web_tool_impls.web_fetch(args["url"], max_bytes=args.get("max_bytes", FETCH_MAX_BYTES))`.
  - `web_search` spec → handler delegates to `Web_Scraping/WebSearch_APIs.perform_websearch`. FIRST read `perform_websearch`'s signature (:1137) and how the legacy `Tools/web_search_tool.py` calls it — copy the legacy tool's config-default wiring, add per-result bounding (~4 KiB/result) and a total cap. Do NOT reuse the legacy tool's Tool ABC; the spec handler calls perform_websearch directly.

- [ ] **Step 3:** tests pass; extend the catalog exact-id test
- [ ] **Step 4:** `git commit -m "feat: web_fetch + web_search tool specs"`

---

## Task 4: `todo_write` (session-scoped state + transcript rendering)

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (session field)
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (todo store seam + spec)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`_compose_local_provider` wiring)
- Test: `Tests/Agents/test_local_tool_provider.py`, `Tests/Chat/test_console_local_review_hook.py`

Design (binding): the provider is per-run and context-free per call, so todo state flows in at composition:
- `ConsoleChatSession` gains `todos: list[dict] = field(default_factory=list)` (items: `{content, status, activeForm}` — status in `pending|in_progress|completed`, claude-code TodoWrite shape; session-lifetime only, no persistence).
- `LocalToolProvider.__init__` gains optional `todo_store: list | None = None` and `on_todo_change: Callable[[list], None] | None = None`. When `todo_store is None`, the `todo_write` spec is NOT registered (catalog unchanged for non-Console/test constructions — document this).
- `todo_write` handler validates the incoming list (each item needs `content` and a valid `status`; exactly one `in_progress` allowed — enforce), replaces the store's contents in place (`store[:] = new`), calls `on_todo_change(store)` if set, returns a short confirmation ("N todos (1 in progress)").
- `_compose_local_provider` passes the current session's `todos` list and an `on_todo_change` that posts the transcript update via the existing `app.call_from_thread` pattern (find how other session-UI updates are surfaced — e.g. `_append_marker` on the bridge; reuse the lightest existing mechanism).
- `tags=("mutates",)`.

- [ ] **Step 1: Failing tests** — provider-level: todo_write replaces store, validates shape (missing content → error, two in_progress → error), calls on_todo_change, spec absent when todo_store=None. Controller-level: composed provider for a session wires the session's actual list.
- [ ] **Step 2: Implement.**
- [ ] **Step 3:** tests pass
- [ ] **Step 4:** `git commit -m "feat: todo_write session todos with transcript rendering"`

---

## Task 5: Discovery hint + integration tests

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`compose_agent_system_prompt`, :103)
- Test: `Tests/Agents/test_local_tools_integration.py`

- [ ] **Step 1:** Add to the composed system prompt one line (verbatim): `Additional tools (file, web, git, and more) are available but not shown; use find_tools to search the catalog and load_tools to load their schemas before calling them.` — placed so it only appears when the registry catalog exceeds the direct-disclosure threshold (check how compose happens vs when disclosure is decided; if the prompt is composed before registry size is known, include the line unconditionally with a "when available" framing — pick the cleaner option and note it). Test: composed prompt contains the hint.
- [ ] **Step 2: Integration tests** — extend the phase-2 padded find/load e2e: a scripted run where the model uses `find_tools` for "fetch", loads and calls `web_fetch` (handler monkeypatched to avoid network — or mock at the httpx layer), asserting one approval round trip; and a `todo_write` e2e asserting the session list mutated and one approval round trip (mutates tag floors inherited allow → ask; the test's resolve_state returns allow-inherited or ask to exercise this — verify which).
- [ ] **Step 3:** full `Tests/Agents Tests/Tools Tests/Chat` green (minus known pre-existing failures)
- [ ] **Step 4:** `git commit -m "feat: tool-discovery hint + research-tool e2e coverage"`

---

## Task 6: Close-out (controller-led)

- [ ] Backlog task: check ACs, Implementation Notes, status Done.
- [ ] Final whole-implementation review subagent, then superpowers:finishing-a-development-branch.
