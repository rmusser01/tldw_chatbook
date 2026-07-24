# Remote Skill Fetch (install from a GitHub link) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install a skill from a pasted GitHub or zip URL: classify → SSRF-hardened fetch → bounded re-root → the EXISTING hardened `import_skill_file` → trust-pending.

**Architecture:** One new module `Skills_Interop/skill_remote_fetch.py` owns classification (pure), the hardened fetcher (httpx, manual redirects, resolve-and-reject), the bounded extraction bridge, and the install seam. Token/branches reuse the existing `Utils/github_api_client.GitHubAPIClient`. Policy enforced via a new public `SkillsScopeService.enforce_install_remote()`. The UI is a URL branch at the top of the existing import flow.

**Tech Stack:** Python 3.11+, httpx (existing dep), `ipaddress`/`socket` stdlib, pytest + `httpx.MockTransport`. No new dependencies.

**Design doc:** `Docs/superpowers/specs/2026-07-24-skills-remote-fetch-design.md` (governs on ambiguity).

## Global Constraints

- **Fetch policy (verbatim):** https-only; `validate_url` first; resolve-and-reject EVERY resolved address (A+AAAA; mixed public+private → reject; IP-literal hosts checked directly) — private/loopback/link-local/reserved/multicast/unspecified all rejected; `follow_redirects=False` with a manual ≤3-hop loop re-running ALL checks per hop; auth header (`token <t>`) ONLY while the current hop's host ∈ {`github.com`, `api.github.com`, `codeload.github.com`, `objects.githubusercontent.com`}; streamed download aborting past **30 MB** compressed; timeouts 10 s connect / 60 s total.
- **GitHub normalization:** `api.github.com/repos/{owner}/{repo}/zipball/{ref}` (`HEAD` default) — never codeload-direct.
- **Ref-split:** one-segment tail = ref; multi-segment → longest-prefix match against `get_branches` (which gains `per_page=100` — one additive edit); no match OR API failure → first-segment heuristic (never a wrong silent match).
- **Bridge is bounded:** candidate discovery is central-directory-only (no decompression); re-root synthesis reuses `LocalSkillsService._read_zip_member_bounded(archive, member, member_name, max_bytes)` and enforces `MAX_SUPPORTING_FILE_BYTES` (5 MB)/`MAX_SUPPORTING_FILES_TOTAL_BYTES` (25 MB)/`MAX_SUPPORTING_FILES_COUNT` (500) DURING synthesis; many candidates → error listing ≤20 paths.
- **Policy:** REQUIRED registry entry `_resource("skills.install_remote", actions=(LAUNCH,))` (engine fails closed); enforced via the NEW public `SkillsScopeService.enforce_install_remote()` BEFORE any network I/O; the import path's own `skills.import.launch.local` gate remains (intentional double-gate).
- Always trust-pending (`trust_approved=False`). Network stays OUT of `LocalSkillsService`.
- **Tests:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <dirs> -q` (directories only — never a file plus its parent dir); Skills/UI suites need `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`. Known baselines: `Tests/Chat/test_anthropic_native_tools.py::test_anthropic_shaped_tools_pass_through_untouched`; flaky `Tests/Skills/test_skills_library_flow.py::test_skill_editor_canvas_scrolls_trust_panel_into_view` (isolate, don't chase).
- **Commit hygiene:** `git add` ONLY named files — NEVER `git add -A`. SUBAGENTS MUST NOT touch/checkout `.superpowers/sdd/progress.md`.
- Anchors verified at `c8082f9cf` — re-grep before editing.

## File Structure

| File | Change |
|---|---|
| `tldw_chatbook/Skills_Interop/skill_remote_fetch.py` | NEW: classifier, ref-split, fetcher, bridge, `install_skill_from_url` |
| `tldw_chatbook/runtime_policy/registry.py` | `_resource("skills.install_remote", actions=(LAUNCH,))` |
| `tldw_chatbook/Skills_Interop/skills_scope_service.py` | public `enforce_install_remote()` |
| `tldw_chatbook/Utils/github_api_client.py` | `get_branches` gains `per_page=100` (one line) |
| `tldw_chatbook/UI/Screens/library_screen.py` | URL branch at top of `_run_library_skills_import` |
| Tests | `Tests/Skills/test_skill_remote_fetch.py` (new; classifier/fetcher/bridge/seam), UI test in `Tests/Skills/test_skills_import.py` or the file's existing import-row suite |

---

### Task 1: Policy entry + public enforcement passthrough + branches pagination

**Files:**
- Modify: `tldw_chatbook/runtime_policy/registry.py` (`server_skills` block, sibling of `skills.read_file`); `tldw_chatbook/Skills_Interop/skills_scope_service.py`; `tldw_chatbook/Utils/github_api_client.py` (`get_branches` ~:200)
- Test: `Tests/Skills/test_skill_remote_fetch.py` (create)

**Interfaces:**
- Produces: action id `skills.install_remote.launch.local` registered; `SkillsScopeService.enforce_install_remote() -> None` (public; raises `PolicyDeniedError` on denial, no-op without an enforcer); `get_branches` requests `params={"per_page": 100}`.

- [ ] **Step 1: Failing tests** (create the test file)

```python
import pytest


def test_policy_action_id_registered():
    # Engine fails CLOSED on unknown ids — pin registration. Mirror the access
    # idiom Tests/ already uses for skills.read_file.launch.local (grep it).
    from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY

    assert "skills.install_remote.launch.local" in CAPABILITY_REGISTRY


def test_scope_service_public_enforce_passthrough():
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    calls = []

    class _Enforcer:
        def require_allowed(self, *, action_id):
            calls.append(action_id)

    scope = SkillsScopeService(
        local_service=None, server_service=None, policy_enforcer=_Enforcer()
    )
    scope.enforce_install_remote()
    assert calls == ["skills.install_remote.launch.local"]
    # No enforcer wired -> no-op, no raise (matches _enforce_policy semantics).
    SkillsScopeService(local_service=None, server_service=None).enforce_install_remote()


@pytest.mark.asyncio
async def test_get_branches_requests_full_page(monkeypatch):
    from tldw_chatbook.Utils.github_api_client import GitHubAPIClient

    client = GitHubAPIClient(token="t")
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"name": "main"}]

    class _Client:
        async def get(self, url, **kwargs):
            captured["url"] = url
            captured["params"] = kwargs.get("params")
            return _Resp()

    monkeypatch.setattr(type(client), "client", property(lambda self: _Client()))
    branches = await client.get_branches("o", "r")
    assert branches == ["main"]
    assert captured["params"] == {"per_page": 100}
```

(Adapt the registry access and SkillsScopeService construction to the real signatures — READ both files first; the assertions are the contract. If `client` is a plain property returning a cached httpx client, monkeypatch whatever the file's other tests patch — grep `Tests/` for existing `GitHubAPIClient` test idioms and mirror.)

- [ ] **Step 2: RED** — `KeyError`/`AttributeError`/missing params. **Step 3: Implement**: registry line `_resource("skills.install_remote", actions=(LAUNCH,)),` after the `skills.read_file` entry; scope-service method:

```python
    def enforce_install_remote(self) -> None:
        """Gate a remote skill install (public seam for skill_remote_fetch).

        Enforces ``skills.install_remote.launch.local`` BEFORE any network
        I/O happens. Public by design: the fetch module must not reach the
        private ``_enforce_policy`` across the class boundary.

        Raises:
            PolicyDeniedError: When a wired policy enforcer denies the action.
        """
        self._enforce_policy("skills.install_remote.launch.local")
```

`get_branches`: change `response = await self.client.get(url)` → `response = await self.client.get(url, params={"per_page": 100})` (cache key unaffected — verify `_get_cache_key(url)` signature; if it accepts params, pass them for key correctness).

- [ ] **Step 4: GREEN** + regression: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/ Tests/RuntimePolicy/ -q` (RuntimePolicy has 2 known pre-existing failures — watchlists/study drift; stash-prove if unsure). **Step 5: Commit**

```bash
git add tldw_chatbook/runtime_policy/registry.py tldw_chatbook/Skills_Interop/skills_scope_service.py tldw_chatbook/Utils/github_api_client.py Tests/Skills/test_skill_remote_fetch.py
git commit -m "feat(skills): install_remote policy entry, public enforce seam, paged branches"
```

---

### Task 2: Pure URL classifier + ref-split

**Files:**
- Create: `tldw_chatbook/Skills_Interop/skill_remote_fetch.py`
- Test: `Tests/Skills/test_skill_remote_fetch.py` (extend)

**Interfaces:**
- Produces (module-level, all pure):

```python
@dataclass(frozen=True)
class GitHubZipSource:
    owner: str
    repo: str
    tree_tail: tuple[str, ...]   # raw /tree/ segments; ref/subdir split later
    suggested_name: str          # subdir basename or repo name (raw; import normalizes)

@dataclass(frozen=True)
class DirectZipSource:
    url: str
    github_auth: bool            # release-asset URLs on github.com
    suggested_name: str          # asset/zip basename sans .zip

class RemoteSkillError(ValueError):
    """User-presentable remote-install failure."""

def classify_skill_source_url(url: str) -> GitHubZipSource | DirectZipSource:
    ...  # raises RemoteSkillError on anything unsupported

async def resolve_ref_and_subdir(
    source: GitHubZipSource,
    list_branches,  # async callable (owner, repo) -> list[str]
) -> tuple[str, str]:
    ...  # -> (ref, subdir-posix-or-""), per the spec's §2 rules
```

- [ ] **Step 1: Failing tests** (append)

```python
from tldw_chatbook.Skills_Interop.skill_remote_fetch import (
    DirectZipSource,
    GitHubZipSource,
    RemoteSkillError,
    classify_skill_source_url,
    resolve_ref_and_subdir,
)


def test_classify_repo_root():
    src = classify_skill_source_url("https://github.com/obra/superpowers")
    assert src == GitHubZipSource(
        owner="obra", repo="superpowers", tree_tail=(), suggested_name="superpowers"
    )


def test_classify_tree_subdir():
    src = classify_skill_source_url(
        "https://github.com/obra/superpowers/tree/main/skills/brainstorming"
    )
    assert isinstance(src, GitHubZipSource)
    assert src.tree_tail == ("main", "skills", "brainstorming")
    assert src.suggested_name == "brainstorming"


def test_classify_release_asset_and_direct_zip():
    rel = classify_skill_source_url(
        "https://github.com/o/r/releases/download/v1/my-skill.zip"
    )
    assert rel == DirectZipSource(
        url="https://github.com/o/r/releases/download/v1/my-skill.zip",
        github_auth=True,
        suggested_name="my-skill",
    )
    direct = classify_skill_source_url("https://example.com/pkg/my-skill.zip")
    assert direct.github_auth is False and direct.suggested_name == "my-skill"


@pytest.mark.parametrize("bad", [
    "http://github.com/o/r",                    # http
    "https://example.com/not-a-zip",            # non-github non-zip
    "git@github.com:o/r.git",                   # git protocol
    "https://github.com/onlyowner",             # no repo
    "ftp://example.com/x.zip",
])
def test_classify_rejects(bad):
    with pytest.raises(RemoteSkillError):
        classify_skill_source_url(bad)


@pytest.mark.asyncio
async def test_ref_split_single_segment_is_ref():
    src = classify_skill_source_url("https://github.com/o/r/tree/v1.2")
    async def _boom(o, r):  # must NOT be called for single-segment tails
        raise AssertionError("branch listing not needed")
    assert await resolve_ref_and_subdir(src, _boom) == ("v1.2", "")


@pytest.mark.asyncio
async def test_ref_split_longest_prefix_branch_wins():
    src = classify_skill_source_url("https://github.com/o/r/tree/feature/foo/skills/x")
    async def _branches(o, r):
        return ["main", "feature/foo", "feature"]
    ref, subdir = await resolve_ref_and_subdir(src, _branches)
    assert (ref, subdir) == ("feature/foo", "skills/x")


@pytest.mark.asyncio
async def test_ref_split_no_match_falls_back_to_first_segment():
    src = classify_skill_source_url("https://github.com/o/r/tree/main/skills/x")
    async def _branches(o, r):
        return ["dev"]  # capped/truncated list without the real branch
    assert await resolve_ref_and_subdir(src, _branches) == ("main", "skills/x")


@pytest.mark.asyncio
async def test_ref_split_api_failure_falls_back():
    src = classify_skill_source_url("https://github.com/o/r/tree/main/skills/x")
    async def _branches(o, r):
        raise RuntimeError("offline")
    assert await resolve_ref_and_subdir(src, _branches) == ("main", "skills/x")
```

- [ ] **Step 2: RED** (ImportError). **Step 3: Implement** (new module; complete):

```python
"""Remote skill fetch: URL classification, hardened download, bounded re-root.

The install path for "paste a GitHub link": classify -> fetch (SSRF-hardened)
-> re-root the archive to a single-skill zip -> the EXISTING hardened
``import_skill_file`` seam. This module owns all skill-install network I/O;
``LocalSkillsService`` stays network-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class GitHubZipSource:
    """A GitHub repo/tree URL normalized for zipball download."""

    owner: str
    repo: str
    tree_tail: tuple[str, ...]
    suggested_name: str


@dataclass(frozen=True)
class DirectZipSource:
    """A direct https zip URL (GitHub release asset or third-party)."""

    url: str
    github_auth: bool
    suggested_name: str


class RemoteSkillError(ValueError):
    """User-presentable remote-install failure."""


def _zip_basename(path_segment: str) -> str:
    name = path_segment.rsplit("/", 1)[-1]
    return name[:-4] if name.lower().endswith(".zip") else name


def classify_skill_source_url(url: str) -> GitHubZipSource | DirectZipSource:
    """Classify a pasted URL into a fetchable skill source.

    Args:
        url: The pasted URL.

    Returns:
        A ``GitHubZipSource`` (repo/tree form, zipball-normalized later) or a
        ``DirectZipSource`` (release asset / any https ``.zip``).

    Raises:
        RemoteSkillError: Non-https, non-GitHub non-zip, or malformed input.
    """
    # Deferred import: input_validation pulls metrics plumbing.
    from ..Utils.input_validation import validate_url

    if not validate_url(url):
        raise RemoteSkillError("That doesn't look like a valid URL.")
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise RemoteSkillError("Only https:// URLs are supported.")
    host = (parsed.hostname or "").lower()
    path = parsed.path.strip("/")
    segments = tuple(s for s in path.split("/") if s)

    if host == "github.com":
        if len(segments) >= 4 and segments[2] == "releases" and segments[3] == "download":
            if not path.lower().endswith(".zip"):
                raise RemoteSkillError("Only .zip release assets are supported.")
            return DirectZipSource(
                url=url, github_auth=True, suggested_name=_zip_basename(path)
            )
        if len(segments) == 2:
            owner, repo = segments
            return GitHubZipSource(
                owner=owner, repo=repo, tree_tail=(), suggested_name=repo
            )
        if len(segments) >= 4 and segments[2] == "tree":
            owner, repo = segments[0], segments[1]
            tail = segments[3:]
            return GitHubZipSource(
                owner=owner,
                repo=repo,
                tree_tail=tail,
                suggested_name=tail[-1] if len(tail) > 1 else repo,
            )
        raise RemoteSkillError(
            "Unsupported GitHub URL — use a repo, /tree/<branch>/<subdir>, "
            "or a .zip release asset."
        )

    if path.lower().endswith(".zip"):
        return DirectZipSource(
            url=url, github_auth=False, suggested_name=_zip_basename(path)
        )
    raise RemoteSkillError(
        "Unsupported URL — paste a GitHub repo/subdirectory URL or a direct .zip link."
    )


async def resolve_ref_and_subdir(
    source: GitHubZipSource, list_branches
) -> tuple[str, str]:
    """Split a ``/tree/`` tail into (ref, subdir) per the spec's §2 rules.

    Single-segment tails are the ref outright. Multi-segment tails try a
    longest-prefix match against the repo's branch list (possibly capped at
    100 — a missing match degrades to the first-segment heuristic, never a
    wrong silent match). Branch-list failures degrade the same way.

    Args:
        source: The classified GitHub source (``tree_tail`` may be empty).
        list_branches: Async callable ``(owner, repo) -> list[str]``.

    Returns:
        ``(ref, subdir)`` — subdir is ``""`` or a POSIX relative path.
    """
    tail = source.tree_tail
    if not tail:
        return "HEAD", ""
    if len(tail) == 1:
        return tail[0], ""
    try:
        branches = await list_branches(source.owner, source.repo)
    except Exception:
        branches = []
    joined = "/".join(tail)
    best = ""
    for branch in branches:
        if (joined == branch or joined.startswith(branch + "/")) and len(branch) > len(best):
            best = branch
    if best:
        subdir = joined[len(best):].strip("/")
        return best, subdir
    return tail[0], "/".join(tail[1:])
```

- [ ] **Step 4: GREEN**: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/ -q`. **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/skill_remote_fetch.py Tests/Skills/test_skill_remote_fetch.py
git commit -m "feat(skills): pure remote-source classifier + slash-ref splitting"
```

---

### Task 3: SSRF-hardened fetcher

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skill_remote_fetch.py`
- Test: `Tests/Skills/test_skill_remote_fetch.py` (extend)

**Interfaces:**
- Produces:

```python
REMOTE_FETCH_MAX_BYTES = 30 * 1024 * 1024
REMOTE_FETCH_MAX_HOPS = 3
GITHUB_AUTH_HOSTS = frozenset(
    {"github.com", "api.github.com", "codeload.github.com", "objects.githubusercontent.com"}
)

async def fetch_zip_bytes(
    url: str,
    *,
    token: str | None = None,
    transport=None,           # httpx transport injection for tests
    resolver=None,            # (host: str) -> list[str] of IPs; None = socket.getaddrinfo
) -> bytes:
    ...  # raises RemoteSkillError on every rejection class
```

- [ ] **Step 1: Failing tests** (append; `httpx.MockTransport` + an injected fake resolver — full code)

```python
import httpx

from tldw_chatbook.Skills_Interop.skill_remote_fetch import (
    GITHUB_AUTH_HOSTS,
    REMOTE_FETCH_MAX_BYTES,
    fetch_zip_bytes,
)

_PUB = lambda host: ["93.184.216.34"]          # public resolver


def _transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_fetch_happy_path_streams_bytes():
    def handler(request):
        return httpx.Response(200, content=b"PK\x03\x04zipbytes")
    out = await fetch_zip_bytes(
        "https://api.github.com/repos/o/r/zipball/HEAD",
        transport=_transport(handler), resolver=_PUB,
    )
    assert out.startswith(b"PK")


@pytest.mark.asyncio
async def test_private_and_mixed_resolution_rejected():
    with pytest.raises(RemoteSkillError):
        await fetch_zip_bytes("https://internal.example/x.zip",
                              transport=_transport(lambda r: httpx.Response(200)),
                              resolver=lambda h: ["10.0.0.5"])
    with pytest.raises(RemoteSkillError):  # mixed public+private -> reject
        await fetch_zip_bytes("https://mixed.example/x.zip",
                              transport=_transport(lambda r: httpx.Response(200)),
                              resolver=lambda h: ["93.184.216.34", "fd00::1"])
    with pytest.raises(RemoteSkillError):  # IP-literal host
        await fetch_zip_bytes("https://169.254.169.254/x.zip",
                              transport=_transport(lambda r: httpx.Response(200)),
                              resolver=_PUB)


@pytest.mark.asyncio
async def test_redirect_hop_revalidated_and_capped():
    def handler(request):
        host = request.url.host
        if host == "a.example":
            return httpx.Response(302, headers={"location": "https://b.example/x.zip"})
        if host == "b.example":
            return httpx.Response(302, headers={"location": "http://c.example/x.zip"})
        raise AssertionError("unexpected host")
    with pytest.raises(RemoteSkillError):   # https->http hop rejected
        await fetch_zip_bytes("https://a.example/x.zip",
                              transport=_transport(handler), resolver=_PUB)

    def loop_handler(request):
        return httpx.Response(302, headers={"location": str(request.url)})
    with pytest.raises(RemoteSkillError):   # >3 hops
        await fetch_zip_bytes("https://a.example/x.zip",
                              transport=_transport(loop_handler), resolver=_PUB)

    def private_hop(request):
        if request.url.host == "a.example":
            return httpx.Response(302, headers={"location": "https://evil.internal/x.zip"})
        raise AssertionError
    with pytest.raises(RemoteSkillError):   # hop host resolves private
        await fetch_zip_bytes(
            "https://a.example/x.zip", transport=_transport(private_hop),
            resolver=lambda h: ["10.1.1.1"] if h == "evil.internal" else ["93.184.216.34"],
        )


@pytest.mark.asyncio
async def test_auth_scoped_to_github_family():
    seen = {}
    def handler(request):
        seen[request.url.host] = request.headers.get("authorization")
        if request.url.host == "api.github.com":
            return httpx.Response(
                302, headers={"location": "https://codeload.github.com/o/r/zip/HEAD"})
        if request.url.host == "codeload.github.com":
            return httpx.Response(
                302, headers={"location": "https://cdn.example.com/x.zip"})
        return httpx.Response(200, content=b"PK\x03\x04")
    await fetch_zip_bytes("https://api.github.com/repos/o/r/zipball/HEAD",
                          token="SECRET", transport=_transport(handler), resolver=_PUB)
    assert seen["api.github.com"] == "token SECRET"
    assert seen["codeload.github.com"] == "token SECRET"
    assert seen["cdn.example.com"] is None    # STRIPPED off-family


@pytest.mark.asyncio
async def test_stream_cap_aborts():
    big = b"x" * (REMOTE_FETCH_MAX_BYTES + 1024)
    def handler(request):
        return httpx.Response(200, content=big)
    with pytest.raises(RemoteSkillError, match="too large"):
        await fetch_zip_bytes("https://example.com/x.zip",
                              transport=_transport(handler), resolver=_PUB)
```

- [ ] **Step 2: RED**. **Step 3: Implement** (append to the module):

```python
import ipaddress
import socket

import httpx

REMOTE_FETCH_MAX_BYTES = 30 * 1024 * 1024
REMOTE_FETCH_MAX_HOPS = 3
_FETCH_CHUNK = 65536
GITHUB_AUTH_HOSTS = frozenset(
    {"github.com", "api.github.com", "codeload.github.com",
     "objects.githubusercontent.com"}
)


def _default_resolver(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    return [info[4][0] for info in infos]


def _assert_host_allowed(host: str, resolver) -> None:
    """Reject unless EVERY resolved address is public (mixed sets reject)."""
    try:
        literal = ipaddress.ip_address(host)
        addresses = [str(literal)]
    except ValueError:
        try:
            addresses = resolver(host)
        except OSError as exc:
            raise RemoteSkillError(f"Could not resolve {host}.") from exc
    if not addresses:
        raise RemoteSkillError(f"Could not resolve {host}.")
    for raw in addresses:
        addr = ipaddress.ip_address(raw.split("%", 1)[0])
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            raise RemoteSkillError("That host is not reachable from here.")


async def fetch_zip_bytes(
    url: str, *, token: str | None = None, transport=None, resolver=None
) -> bytes:
    """Download a zip with SSRF hardening, manual redirects, and a size cap.

    Args:
        url: The (already classified/normalized) https URL to download.
        token: Optional GitHub token; attached ONLY on GITHUB_AUTH_HOSTS hops.
        transport: httpx transport override (tests).
        resolver: ``(host) -> list[str]`` override (tests); defaults to
            ``socket.getaddrinfo``.

    Returns:
        The raw (compressed) zip bytes, at most ``REMOTE_FETCH_MAX_BYTES``.

    Raises:
        RemoteSkillError: On every rejection class (scheme, host, hops, cap,
            HTTP status, timeouts).
    """
    from ..Utils.input_validation import validate_url

    resolver = resolver or _default_resolver
    timeout = httpx.Timeout(60.0, connect=10.0)
    current = url
    async with httpx.AsyncClient(
        transport=transport, timeout=timeout, follow_redirects=False
    ) as client:
        for _hop in range(REMOTE_FETCH_MAX_HOPS + 1):
            if not validate_url(current):
                raise RemoteSkillError("Invalid URL after redirect.")
            parsed = httpx.URL(current)
            if parsed.scheme != "https":
                raise RemoteSkillError("Only https:// is allowed (redirect downgraded).")
            host = (parsed.host or "").lower()
            _assert_host_allowed(host, resolver)
            headers = {}
            if token and host in GITHUB_AUTH_HOSTS:
                headers["Authorization"] = f"token {token}"
            try:
                async with client.stream("GET", current, headers=headers) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            raise RemoteSkillError("Redirect without a location.")
                        current = str(parsed.join(location))
                        continue
                    if response.status_code != 200:
                        raise RemoteSkillError(
                            f"Download failed (HTTP {response.status_code})."
                        )
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in response.aiter_bytes(_FETCH_CHUNK):
                        received += len(chunk)
                        if received > REMOTE_FETCH_MAX_BYTES:
                            raise RemoteSkillError(
                                "Download too large (over 30 MB compressed)."
                            )
                        chunks.append(chunk)
                    return b"".join(chunks)
            except httpx.TimeoutException as exc:
                raise RemoteSkillError("Download timed out.") from exc
            except httpx.HTTPError as exc:
                raise RemoteSkillError(f"Download failed: {exc}") from exc
        raise RemoteSkillError("Too many redirects.")
```

- [ ] **Step 4: GREEN** (same suite command). **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/skill_remote_fetch.py Tests/Skills/test_skill_remote_fetch.py
git commit -m "feat(skills): SSRF-hardened streaming zip fetcher with scoped GitHub auth"
```

---

### Task 4: Bounded extraction bridge

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skill_remote_fetch.py`
- Test: `Tests/Skills/test_skill_remote_fetch.py` (extend)

**Interfaces:**
- Consumes: `LocalSkillsService._read_zip_member_bounded(archive, member, member_name, max_bytes) -> bytes` (staticmethod, exists — import the CLASS and call the staticmethod); caps from `tldw_api.skills_schemas`.
- Produces: `re_root_skill_zip(zip_bytes: bytes, *, subdir: str, suggested_name: str) -> tuple[bytes, str]` → (single-skill zip bytes, final name). Raises `RemoteSkillError` for zero candidates, many candidates (message lists ≤20 paths + "paste a subdirectory URL"), corrupt zip, or cap violations during synthesis.

- [ ] **Step 1: Failing tests** (append — build archives with zipfile in-memory)

```python
import io
import zipfile

from tldw_chatbook.Skills_Interop.skill_remote_fetch import re_root_skill_zip


def _zipball(entries, wrapper="repo-abc123/"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries:
            z.writestr(wrapper + name, data)
    return buf.getvalue()


def test_reroot_subdir_install():
    data = _zipball([
        ("skills/brainstorm/SKILL.md", "---\nname: brainstorm\n---\nbody"),
        ("skills/brainstorm/references/api.md", "# api"),
        ("skills/other/SKILL.md", "x"),
        ("README.md", "top"),
    ])
    out, name = re_root_skill_zip(data, subdir="skills/brainstorm", suggested_name="brainstorm")
    with zipfile.ZipFile(io.BytesIO(out)) as z:
        assert set(z.namelist()) == {"SKILL.md", "references/api.md"}
    assert name == "brainstorm"


def test_reroot_root_is_skill_and_unwrapped_asset():
    wrapped = _zipball([("SKILL.md", "body"), ("notes.md", "n")])
    out, _ = re_root_skill_zip(wrapped, subdir="", suggested_name="repo")
    with zipfile.ZipFile(io.BytesIO(out)) as z:
        assert "SKILL.md" in z.namelist()
    unwrapped = _zipball([("SKILL.md", "body")], wrapper="")   # release-asset shape
    out2, _ = re_root_skill_zip(unwrapped, subdir="", suggested_name="asset")
    with zipfile.ZipFile(io.BytesIO(out2)) as z:
        assert "SKILL.md" in z.namelist()


def test_reroot_single_candidate_auto_installs():
    data = _zipball([("skills/only/SKILL.md", "body"), ("README.md", "r")])
    out, name = re_root_skill_zip(data, subdir="", suggested_name="repo")
    assert name == "only"


def test_reroot_many_candidates_error_lists_paths():
    entries = [(f"skills/s{i}/SKILL.md", "b") for i in range(25)] 
    data = _zipball(entries)
    with pytest.raises(RemoteSkillError) as exc:
        re_root_skill_zip(data, subdir="", suggested_name="repo")
    msg = str(exc.value)
    assert "skills/s0" in msg and "subdirectory URL" in msg
    assert msg.count("skills/s") <= 20


def test_reroot_zero_candidates_and_corrupt():
    with pytest.raises(RemoteSkillError, match="No SKILL.md"):
        re_root_skill_zip(_zipball([("README.md", "r")]), subdir="", suggested_name="x")
    with pytest.raises(RemoteSkillError):
        re_root_skill_zip(b"not a zip", subdir="", suggested_name="x")


def test_reroot_lying_header_bomb_aborts_bounded():
    # Forge a member whose declared size is tiny but whose stream is huge:
    # simplest honest proxy — a member whose ACTUAL content exceeds the
    # per-file cap; the bounded reader must abort during synthesis.
    from tldw_chatbook.tldw_api.skills_schemas import MAX_SUPPORTING_FILE_BYTES
    big = "x" * (MAX_SUPPORTING_FILE_BYTES + 100)
    data = _zipball([("SKILL.md", "body"), ("references/huge.md", big)])
    with pytest.raises(RemoteSkillError, match="too large"):
        re_root_skill_zip(data, subdir="", suggested_name="repo")
```

- [ ] **Step 2: RED**. **Step 3: Implement** (append):

```python
import zipfile as _zipfile
from io import BytesIO

_CANDIDATE_SCAN_DEPTH = 3
_CANDIDATE_LIST_LIMIT = 20


def _archive_names(archive: "_zipfile.ZipFile") -> list[str]:
    return [m.filename for m in archive.infolist() if not m.is_dir()]


def _detect_root(names: list[str]) -> str:
    """Root prefix: '' when SKILL.md sits at top; descend ONE wrapper dir."""
    if "SKILL.md" in names:
        return ""
    tops = {n.split("/", 1)[0] for n in names if "/" in n}
    loose = [n for n in names if "/" not in n]
    if len(tops) == 1 and not loose:
        return next(iter(tops)) + "/"
    return ""


def re_root_skill_zip(
    zip_bytes: bytes, *, subdir: str, suggested_name: str
) -> tuple[bytes, str]:
    """Re-root a fetched archive to a single-skill zip (BOUNDED synthesis).

    Candidate discovery is central-directory-only (no decompression). Member
    extraction reuses ``LocalSkillsService._read_zip_member_bounded`` and
    enforces the import caps DURING synthesis, so a lying-header bomb aborts
    here exactly as it would in the importer.

    Args:
        zip_bytes: The fetched archive (already download-capped).
        subdir: POSIX subdir to install from ('' = auto-detect at root).
        suggested_name: Fallback skill name (subdir basename wins upstream).

    Returns:
        ``(single_skill_zip_bytes, final_name)``.

    Raises:
        RemoteSkillError: Corrupt archive, zero/many candidates, cap abort.
    """
    from .local_skills_service import LocalSkillsService
    from ..tldw_api.skills_schemas import (
        MAX_SUPPORTING_FILE_BYTES,
        MAX_SUPPORTING_FILES_COUNT,
        MAX_SUPPORTING_FILES_TOTAL_BYTES,
    )

    try:
        archive = _zipfile.ZipFile(BytesIO(zip_bytes))
    except _zipfile.BadZipFile as exc:
        raise RemoteSkillError("The download was not a valid zip archive.") from exc
    with archive:
        names = _archive_names(archive)
        root = _detect_root(names)
        base = root + (subdir.strip("/") + "/" if subdir.strip("/") else "")

        if base and not any(n.startswith(base) for n in names):
            raise RemoteSkillError(f"No such folder in the archive: {subdir}")

        skill_root = base
        final_name = suggested_name
        if (skill_root + "SKILL.md") not in names:
            # Depth-limited candidate scan under the current base.
            candidates = sorted(
                n[len(base):-len("/SKILL.md")]
                for n in names
                if n.startswith(base)
                and n.endswith("/SKILL.md")
                and n[len(base):].count("/") <= _CANDIDATE_SCAN_DEPTH
            )
            if not candidates:
                raise RemoteSkillError("No SKILL.md found in that archive.")
            if len(candidates) > 1:
                shown = ", ".join(candidates[:_CANDIDATE_LIST_LIMIT])
                more = len(candidates) - min(len(candidates), _CANDIDATE_LIST_LIMIT)
                suffix = f" (+{more} more)" if more > 0 else ""
                raise RemoteSkillError(
                    f"Found {len(candidates)} skills in that repository: "
                    f"{shown}{suffix}. Paste a subdirectory URL to install one."
                )
            skill_root = base + candidates[0] + "/"
            final_name = candidates[0].rsplit("/", 1)[-1]

        members = [
            m for m in archive.infolist()
            if not m.is_dir() and m.filename.startswith(skill_root)
        ]
        if len(members) > MAX_SUPPORTING_FILES_COUNT + 1:
            raise RemoteSkillError("That skill bundle has too many files.")
        out = BytesIO()
        total = 0
        with _zipfile.ZipFile(out, "w", compression=_zipfile.ZIP_DEFLATED) as dest:
            for member in members:
                relative = member.filename[len(skill_root):]
                if not relative:
                    continue
                if member.file_size > MAX_SUPPORTING_FILE_BYTES:
                    raise RemoteSkillError(
                        f"File too large in bundle: {relative}"
                    )
                try:
                    data = LocalSkillsService._read_zip_member_bounded(
                        archive, member, relative, MAX_SUPPORTING_FILE_BYTES
                    )
                except ValueError as exc:
                    raise RemoteSkillError(f"File too large in bundle: {relative}") from exc
                total += len(data)
                if total > MAX_SUPPORTING_FILES_TOTAL_BYTES:
                    raise RemoteSkillError("That skill bundle is too large.")
                info = _zipfile.ZipInfo(relative)
                info.external_attr = member.external_attr
                dest.writestr(info, data)
        return out.getvalue(), final_name
```

(Check `_read_zip_member_bounded`'s raise type for over-cap — it raises `ValueError("local_skill_file_too_large:…")`; the wrap above maps it to `RemoteSkillError` matching "too large". Adjust the match string if the helper's message differs — READ it first.)

- [ ] **Step 4: GREEN**. **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/skill_remote_fetch.py Tests/Skills/test_skill_remote_fetch.py
git commit -m "feat(skills): bounded archive re-rooting bridge with candidate discovery"
```

---

### Task 5: Install seam + UI URL branch

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skill_remote_fetch.py`; `tldw_chatbook/UI/Screens/library_screen.py` (`_run_library_skills_import` head ~:7248; placeholder copy in `Widgets/Library/library_skills_canvas.py` import-row Input ~:766)
- Test: `Tests/Skills/test_skill_remote_fetch.py` (seam) + the import-row UI suite in `Tests/Skills/test_skills_import.py` (URL routing)

**Interfaces:**
- Produces: `async install_skill_from_url(url: str, *, scope_service, overwrite: bool = False, transport=None, resolver=None) -> dict` — order: classify (pure, may raise) → `scope_service.enforce_install_remote()` → token via `GitHubAPIClient` → (GitHub source) `resolve_ref_and_subdir(source, api.get_branches)` → normalized `https://api.github.com/repos/{owner}/{repo}/zipball/{ref}` → `fetch_zip_bytes` → `re_root_skill_zip` → `await scope_service.import_skill_file(bytes, mode="local", filename=f"{name}.zip", content_type="application/zip", overwrite=overwrite, trust_approved=False)`; returns the import result dict. The `GitHubAPIClient` instance is closed (`await api.close()`) in a finally.
- UI: at the TOP of `_run_library_skills_import`, before `validate_path_simple`:

```python
        if raw_path.startswith(("http://", "https://")):
            await self._install_library_skill_from_url(raw_path)
            return
```

with a new `_install_library_skill_from_url` mirroring `_import_library_skill_from_loose_file` (~:7303): resolve `skills_scope_service`; call `install_skill_from_url`; success → `self._apply_library_skills_import_success(result.get("name", ""))`; `RemoteSkillError` → `self._apply_library_skills_import_status(str(exc))` (user-presentable by construction); other exceptions → `self._apply_library_skills_import_outcome_from_exception(name_guess, exc)`. Placeholder copy: append "or GitHub/zip URL".

- [ ] **Step 1: Failing tests**: seam test with a fake scope service (records `enforce_install_remote` called BEFORE any transport request; import_skill_file receives the re-rooted bytes + `trust_approved=False` + `filename="brainstorm.zip"`), MockTransport serving a zipball for `/repos/o/r/zipball/main`; an http:// URL raises without touching enforce (classify-first is fine — but enforcement MUST precede network: assert transport saw zero requests when enforce raises PolicyDeniedError-like). UI test: a URL draft routes to the fetch path (monkeypatch `install_skill_from_url` in the screen module's namespace) and success primes the Review button — mirror the existing import-row test idioms in `Tests/Skills/test_skills_import.py`.
- [ ] **Step 2: RED** → **Step 3: implement seam** (compose the Task 2–4 pieces exactly per the Interfaces block; GitHubAPIClient used for `.token` + `get_branches`, closed in finally) **+ UI branch** → **Step 4: GREEN** + `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/ -q`. **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/skill_remote_fetch.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/library_skills_canvas.py Tests/Skills/test_skill_remote_fetch.py Tests/Skills/test_skills_import.py
git commit -m "feat(skills): install_skill_from_url seam + Library URL import branch"
```

---

### Task 6: E2E + full regression

**Files:**
- Test: `Tests/Skills/test_skill_remote_fetch.py` (extend)

- [ ] **Step 1: E2E test** — MockTransport serving: `api.github.com/repos/obra/superpowers/zipball/main` → 302 to `codeload.github.com/...` → 200 with a real zipball-shaped archive (`superpowers-abc/skills/demo/SKILL.md` + `references/api.md`); a REAL `LocalSkillsService(store_dir=tmp_path)` behind a REAL `SkillsScopeService`; `install_skill_from_url("https://github.com/obra/superpowers/tree/main/skills/demo", scope_service=..., transport=..., resolver=fake_public)` → skill "demo" exists on disk with `references/api.md`, trust-pending (not trusted), and the auth-scoping/302 path exercised end to end.
- [ ] **Step 2: Full gate:**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/ Tests/RuntimePolicy/ Tests/UI/test_library_skills_canvas.py -q`
Expected: 0 failed beyond the known baselines (RuntimePolicy's 2 pre-existing; the flaky canvas test).

- [ ] **Step 3: Commit**

```bash
git add Tests/Skills/test_skill_remote_fetch.py
git commit -m "test(skills): e2e — install a skill from a GitHub tree URL via mocked transports"
```

---

## Notes for the executor

- The spec governs; its §3 fetch policy and §4 bounded-bridge rules are load-bearing security.
- `RemoteSkillError` messages are user-presentable BY DESIGN — the UI shows them verbatim; keep them human.
- The DNS-rebinding residual is documented in the spec — do NOT attempt transport-level IP pinning in this plan.
- Re-grep every anchor; files shift.
