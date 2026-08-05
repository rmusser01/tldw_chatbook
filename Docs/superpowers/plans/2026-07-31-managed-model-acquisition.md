# Managed Model Acquisition (TASK-595) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consent-driven managed acquisition (preflight → consent → resumable verified download → install → activate) composed over the sealed TASK-594 artifact core.

**Architecture:** New async `ArtifactAcquisitionService` (`Model_Artifacts/acquisition.py`) + streaming guarded fetch (`Model_Artifacts/fetch.py`), composing the existing sync `ModelArtifactService` via executor hops. Two small sync core additions only: `install(consume_source=)` and orphans-only staging GC in `reconcile()`. One exclusive acquisition-session lease serializes managed acquisition across processes.

**Tech Stack:** Python ≥3.11, httpx (async streaming), portalocker leases (existing), pytest + pytest-asyncio, stdlib `http.server` fixture.

**Spec:** `Docs/superpowers/specs/2026-07-30-managed-model-acquisition-design.md` — read it first. Binding: ADR-025; TASK-595 ACs in full.

## Global Constraints

- The TASK-594 suite (`Tests/Model_Artifacts/test_service.py`, `test_operation_leases*.py`) must pass UNMODIFIED at every commit.
- Core changes are limited to Tasks 1–2 exactly as specified; everything else composes.
- Tests touch the network ONLY via the localhost fixture server; no external hosts, ever.
- Constants: `ACQUISITION_SAFETY_MARGIN_BYTES = 256 * 1024 * 1024` and `MAX_FILE_REFETCHES = 1` live in `acquisition.py`. `ACQUISITION_SESSION_LEASE_KEY = ArtifactLeaseKey("#managed-acquisition", "session", "global")` is defined ONCE in `service.py` (Task 2 needs it and importing from acquisition would be circular); Task 6 makes it public there and `acquisition.py` imports it.
- Secrets (tokens/headers) never appear in logs, error messages, manifests, or sidecars — several tasks assert this.
- Async tests need explicit `@pytest.mark.asyncio`.
- Test command prefix (venv is uv-managed in the MAIN checkout; worktree code must win):
  `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.claude/worktrees/wizard-loose-ends /Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.venv/bin/pytest`
- Google-style docstrings with Args/Returns/Raises on every new public callable; type hints throughout.
- Backlog hygiene: TASK-595 is In Progress; final task closes it with notes.

---

### Task 1: Core — `install(consume_source=)`

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/service.py` (`install` ~:1680, `_copy_payload` ~:2257)
- Test: `Tests/Model_Artifacts/test_install_consume_source.py` (create)

**Interfaces:**
- Consumes: existing `install(descriptor, source_directory) -> ArtifactRef`, `_copy_payload`, staging layout.
- Produces: `install(descriptor, source_directory, *, consume_source: bool = False) -> ArtifactRef`. With `consume_source=True`: per-file `os.replace` into install staging when `source_directory` is inside the service root; source outside the root raises `ArtifactPathError`; per-file `EXDEV` falls back to copy+delete for that file. Default `False` = today's behavior, byte-for-byte.

- [ ] **Step 1: Write the failing tests**

Read `Tests/Model_Artifacts/test_service.py` first for the established fixtures (service root tmp_path fixture, descriptor builders) and reuse them. Create `Tests/Model_Artifacts/test_install_consume_source.py`:

```python
"""TASK-595 Task 1: consume_source install semantics."""

import os
from pathlib import Path

import pytest

from tldw_chatbook.Model_Artifacts.service import (
    ArtifactPathError,
    ModelArtifactService,
)

# Reuse the descriptor/payload builders from test_service.py — import them
# if they are module-level helpers; otherwise copy the minimal single-file
# descriptor builder here (one file "model.onnx", correct size and sha256).
from Tests.Model_Artifacts.test_service import (  # adjust to actual names
    make_descriptor,
    write_payload,
)


@pytest.fixture()
def service(tmp_path):
    return ModelArtifactService(tmp_path / "root")


def test_consume_source_moves_files_and_installs(service, tmp_path):
    """Files are moved (source emptied), install verifies and promotes."""
    descriptor = make_descriptor()
    source = Path(service.staging_path) / "managed" / "src"
    source.mkdir(parents=True)
    write_payload(source, descriptor)
    ref = service.install(descriptor, source, consume_source=True)
    installed = service.artifact_path(ref)
    assert installed.exists()
    # Moved, not copied: the declared payload files are gone from source.
    for file in descriptor.files:
        assert not (source / file.path).exists()


def test_consume_source_outside_root_raises(service, tmp_path):
    descriptor = make_descriptor()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    write_payload(outside, descriptor)
    with pytest.raises(ArtifactPathError):
        service.install(descriptor, outside, consume_source=True)
    # Nothing consumed on refusal.
    for file in descriptor.files:
        assert (outside / file.path).exists()


def test_consume_source_exdev_falls_back_to_copy(service, monkeypatch):
    """EXDEV inside the root degrades to copy+delete, still installing."""
    descriptor = make_descriptor()
    source = Path(service.staging_path) / "managed" / "src"
    source.mkdir(parents=True)
    write_payload(source, descriptor)

    real_replace = os.replace

    def exdev_replace(src, dst, *a, **k):
        raise OSError(18, "Invalid cross-device link")  # errno.EXDEV

    monkeypatch.setattr("tldw_chatbook.Model_Artifacts.service.os.replace", exdev_replace)
    ref = service.install(descriptor, source, consume_source=True)
    monkeypatch.setattr("tldw_chatbook.Model_Artifacts.service.os.replace", real_replace)
    assert service.artifact_path(ref).exists()


def test_default_copy_behavior_unchanged(service, tmp_path):
    """consume_source=False keeps today's copy semantics: source intact."""
    descriptor = make_descriptor()
    source = tmp_path / "root" / "staging" / "src2"
    source.mkdir(parents=True)
    write_payload(source, descriptor)
    service.install(descriptor, source)
    for file in descriptor.files:
        assert (source / file.path).exists()
```

If `test_service.py`'s builders are not importable module-level helpers, define local `make_descriptor()`/`write_payload()` mirroring its construction exactly (single `ArtifactFile` with real computed sha256 via `hashlib.sha256`) — do not weaken any assertion to avoid the plumbing.

- [ ] **Step 2: Run to verify failure**

Run: `<prefix> Tests/Model_Artifacts/test_install_consume_source.py -q`
Expected: FAIL — `install() got an unexpected keyword argument 'consume_source'`.

- [ ] **Step 3: Implement**

In `service.py`: add the keyword to `install`, thread it to the payload-transfer step. Where `_copy_payload` iterates declared files, a `consume_source` transfer does:

```python
def _transfer_payload(self, descriptor, source_directory, staging_directory, *, consume_source):
    """Copy or move the declared payload into install staging.

    Args:
        consume_source: Move files with os.replace when the source lies
            inside this service's root (both stagings share it). EXDEV
            (bind-mount inside the root) degrades to copy+delete for that
            file — correctness over the disk optimization.

    Raises:
        ArtifactPathError: consume_source with a source outside the root.
    """
    if consume_source:
        resolved = source_directory.resolve(strict=True)
        if not resolved.is_relative_to(self._root):   # match existing containment idiom
            raise ArtifactPathError(
                "consume_source requires a source inside the service root"
            )
    for file in descriptor.files:
        src = source_directory / file.path
        dst = staging_directory / file.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if consume_source:
            try:
                os.replace(src, dst)
                continue
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                logger.debug("EXDEV moving %s; falling back to copy", file.path)
        shutil.copyfile(src, dst)
        if consume_source:
            src.unlink()
```

Match the existing code's naming, containment helpers (`_assert_managed_path` / root identity checks — read them and use the same idiom, not `is_relative_to` if the file uses inode-identity checks), logging style, and error types. `import errno` at module top if absent.

- [ ] **Step 4: Run new tests + the sealed-core suite**

Run: `<prefix> Tests/Model_Artifacts/test_install_consume_source.py Tests/Model_Artifacts/test_service.py -q`
Expected: ALL PASS (594 suite untouched and green).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/service.py Tests/Model_Artifacts/test_install_consume_source.py
git commit -m "feat(artifacts): consume_source install — in-root move with EXDEV copy fallback (TASK-595)"
```

---

### Task 2: Core — orphans-only staging GC in `reconcile()`

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/service.py` (`reconcile` ~:1036, `ReconcileReport` ~:786)
- Test: `Tests/Model_Artifacts/test_reconcile_staging_gc.py` (create)

**Interfaces:**
- Consumes: existing `reconcile() -> ReconcileReport`, staging layout, lease machinery.
- Produces: `ReconcileReport` gains `staging_removed: tuple[str, ...] = ()` (additive, default keeps compat). GC rule: delete staging entries that are **true orphans** — any entry that is NOT a `managed/<id>/<rev>/<variant>` directory containing a parseable `fetch-state.json`. Managed entries WITH a valid sidecar are never touched. GC of managed entries requires the acquisition-session lease non-blocking (skip GC of managed space entirely when busy); non-managed orphans (dead `install-*` tmpdirs) are removed under whatever exclusive lease `reconcile` already holds — READ the current `reconcile` locking first and mirror it.

- [ ] **Step 1: Write the failing tests**

```python
"""TASK-595 Task 2: reconcile deletes staging orphans, preserves resumable state."""

import json
from pathlib import Path

import pytest

from tldw_chatbook.Model_Artifacts.service import ModelArtifactService


@pytest.fixture()
def service(tmp_path):
    return ModelArtifactService(tmp_path / "root")


def _managed_dir(service, artifact="m1", rev="r1", variant="int8") -> Path:
    d = Path(service.staging_path) / "managed" / artifact / rev / variant
    d.mkdir(parents=True)
    return d


def test_orphan_install_staging_is_removed(service):
    orphan = Path(service.staging_path) / "install-deadbeef"
    orphan.mkdir()
    (orphan / "partial.bin").write_bytes(b"x" * 10)
    report = service.reconcile()
    assert not orphan.exists()
    assert any("install-deadbeef" in item for item in report.staging_removed)


def test_managed_entry_without_sidecar_is_removed(service):
    d = _managed_dir(service)
    (d / "model.onnx").write_bytes(b"partial")
    report = service.reconcile()
    assert not d.exists()
    assert report.staging_removed


def test_managed_entry_with_valid_sidecar_survives(service):
    d = _managed_dir(service)
    (d / "model.onnx").write_bytes(b"partial")
    (d / "fetch-state.json").write_text(json.dumps({
        "files": {"model.onnx": {"etag": "\"abc\"", "last_modified": None,
                                   "bytes_done": 7, "complete": False}}
    }))
    report = service.reconcile()
    assert d.exists()
    assert (d / "model.onnx").exists()


def test_managed_entry_with_corrupt_sidecar_is_removed(service):
    d = _managed_dir(service)
    (d / "fetch-state.json").write_text("{not json")
    service.reconcile()
    assert not d.exists()


def test_gc_never_escapes_staging(service, tmp_path):
    """Containment: a symlink inside staging pointing outside must not
    cause deletion outside the root (extends the 594 containment tests)."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep")
    link = Path(service.staging_path) / "managed" / "evil"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(victim)
    service.reconcile()
    assert (victim / "keep.txt").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `<prefix> Tests/Model_Artifacts/test_reconcile_staging_gc.py -q`
Expected: FAIL — orphans survive and `ReconcileReport` has no `staging_removed`.

- [ ] **Step 3: Implement**

Add the field to `ReconcileReport`; in `reconcile()`, after the existing staging scan (~:1339 where `staging_entries` is collected), classify each entry: parse `managed/<id>/<rev>/<variant>/fetch-state.json` with `json.loads` guarded (missing/invalid ⇒ orphan); for managed-space deletions, first try `ACQUISITION_SESSION_LEASE_KEY` non-blocking (import from `acquisition` would be circular — define the key TUPLE locally in service.py as `_ACQUISITION_SESSION_KEY = ArtifactLeaseKey("#managed-acquisition", "session", "global")` with a cross-reference comment; Task 6 imports it FROM service to guarantee a single definition). Delete with the file's existing containment-checked deletion idiom (find how `delete()` removes trees safely — reuse it; symlinked entries are unlinked, never followed). Record removed entry names (relative to staging root) in `staging_removed`.

- [ ] **Step 4: Run new + sealed suites**

Run: `<prefix> Tests/Model_Artifacts/test_reconcile_staging_gc.py Tests/Model_Artifacts/test_service.py -q`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/service.py Tests/Model_Artifacts/test_reconcile_staging_gc.py
git commit -m "feat(artifacts): reconcile GCs staging orphans only; resumable sidecar state survives (TASK-595)"
```

---

### Task 3: Fixture HTTP server + `fetch.stream_fetch`

**Files:**
- Create: `Tests/Model_Artifacts/fixture_http.py`
- Create: `tldw_chatbook/Model_Artifacts/fetch.py`
- Test: `Tests/Model_Artifacts/test_stream_fetch.py` (create)

**Interfaces:**
- Consumes: `tldw_chatbook.Utils.egress.check_url_or_raise_async(url, *, trusted_origins)`, `evaluate_url_policy_async`, `MAX_REDIRECT_HOPS` (public); httpx.
- Produces:
  - `FetchValidators(etag: str | None, last_modified: str | None)` frozen; property `strong: bool` (False when etag is None-or-weak `W/` and last_modified is None).
  - `FetchResult(bytes_written: int, validators: FetchValidators, resumed: bool)` frozen.
  - `class FetchRestartRequired(Exception)` — validators mismatch / no Range support while resuming; caller restarts from zero.
  - `class FetchTooLargeError(Exception)`; `class FetchTransportError(Exception)` (wraps httpx errors; message NEVER includes headers).
  - `async def stream_fetch(url: str, destination: Path, *, client: httpx.AsyncClient, max_bytes: int, resume_from: int = 0, validators: FetchValidators | None = None, headers: Mapping[str, str] | None = None, trusted_origins: frozenset[str] = frozenset(), on_chunk: Callable[[int], None] | None = None) -> FetchResult` — appends to `destination`, `fsync`s before returning, `resume_from + written ≤ max_bytes` enforced, per-hop egress checks, credential headers stripped on cross-origin redirects (local `_STRIP_HEADERS` tuple mirroring egress's, with a drift-guard test).
  - Fixture server: `class FixtureArtifactServer` — context manager; `serve(path: str, body: bytes, *, etag: str | None = '"v1"', weak_etag: bool = False, support_range: bool = True, disconnect_after: int | None = None, require_token: str | None = None)`; `.url(path) -> str`; records request headers per path for assertions (`.requests[path] -> list[dict]`).

- [ ] **Step 1: Write the fixture server**

```python
"""Localhost HTTP fixture for artifact-fetch tests. stdlib only."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class _Route:
    body: bytes
    etag: str | None
    support_range: bool
    disconnect_after: int | None
    require_token: str | None


class FixtureArtifactServer:
    """Configurable localhost server: Range, ETag, faults, auth."""

    def __init__(self) -> None:
        self._routes: dict[str, _Route] = {}
        self.requests: dict[str, list[dict]] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _route(self):
                return outer._routes.get(self.path)

            def do_HEAD(self):
                self._respond(head_only=True)

            def do_GET(self):
                self._respond(head_only=False)

            def _respond(self, *, head_only: bool):
                route = self._route()
                outer.requests.setdefault(self.path, []).append(dict(self.headers))
                if route is None:
                    self.send_response(404); self.end_headers(); return
                if route.require_token and (
                    self.headers.get("Authorization") != f"Bearer {route.require_token}"
                ):
                    self.send_response(401); self.end_headers(); return
                body = route.body
                start = 0
                status = 200
                range_header = self.headers.get("Range")
                if range_header and route.support_range:
                    start = int(range_header.split("=")[1].split("-")[0])
                    body = body[start:]
                    status = 206
                self.send_response(status)
                if route.etag:
                    self.send_header("ETag", route.etag)
                if route.support_range:
                    self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if head_only:
                    return
                cut = route.disconnect_after
                if cut is not None and cut < len(body):
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.connection.close()  # simulate mid-body drop
                    return
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def serve(self, path, body, *, etag='"v1"', weak_etag=False,
              support_range=True, disconnect_after=None, require_token=None):
        if etag and weak_etag:
            etag = f"W/{etag}"
        self._routes[path] = _Route(body, etag, support_range, disconnect_after, require_token)

    def url(self, path: str) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}{path}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        return False
```

- [ ] **Step 2: Write the failing fetch tests**

```python
"""TASK-595 Task 3: streaming guarded fetch."""

import hashlib

import httpx
import pytest

from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer
from tldw_chatbook.Model_Artifacts.fetch import (
    FetchRestartRequired,
    FetchTooLargeError,
    FetchValidators,
    stream_fetch,
)

BODY = b"0123456789" * 1000  # 10 KB


@pytest.mark.asyncio
async def test_full_fetch_writes_and_reports(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY)
        dest = tmp_path / "f.bin"
        async with httpx.AsyncClient() as client:
            result = await stream_fetch(
                srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY) + 10,
                trusted_origins=frozenset({srv.url("/").rstrip("/")}),
            )
    assert dest.read_bytes() == BODY
    assert result.bytes_written == len(BODY)
    assert result.resumed is False
    assert result.validators.etag == '"v1"'


@pytest.mark.asyncio
async def test_resume_uses_range_and_appends(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY)
        dest = tmp_path / "f.bin"
        dest.write_bytes(BODY[:4000])
        async with httpx.AsyncClient() as client:
            result = await stream_fetch(
                srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                resume_from=4000,
                validators=FetchValidators(etag='"v1"', last_modified=None),
                trusted_origins=frozenset({srv.url("/").rstrip("/")}),
            )
    assert dest.read_bytes() == BODY
    assert result.resumed is True
    assert result.bytes_written == len(BODY) - 4000
    # The server actually saw a Range request.
    assert any("Range" in r for r in srv.requests["/f.bin"])


@pytest.mark.asyncio
async def test_changed_validator_raises_restart(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY, etag='"v2"')
        dest = tmp_path / "f.bin"
        dest.write_bytes(BODY[:100])
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchRestartRequired):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                    resume_from=100,
                    validators=FetchValidators(etag='"v1"', last_modified=None),
                    trusted_origins=frozenset({srv.url("/").rstrip("/")}),
                )


@pytest.mark.asyncio
async def test_weak_etag_never_resumes(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY, weak_etag=True)
        dest = tmp_path / "f.bin"
        dest.write_bytes(BODY[:100])
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchRestartRequired):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                    resume_from=100,
                    validators=FetchValidators(etag='W/"v1"', last_modified=None),
                    trusted_origins=frozenset({srv.url("/").rstrip("/")}),
                )


@pytest.mark.asyncio
async def test_no_range_support_raises_restart(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY, support_range=False)
        dest = tmp_path / "f.bin"
        dest.write_bytes(BODY[:100])
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchRestartRequired):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                    resume_from=100,
                    validators=FetchValidators(etag='"v1"', last_modified=None),
                    trusted_origins=frozenset({srv.url("/").rstrip("/")}),
                )


@pytest.mark.asyncio
async def test_max_bytes_bounds_final_size(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY)
        dest = tmp_path / "f.bin"
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchTooLargeError):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=100,
                    trusted_origins=frozenset({srv.url("/").rstrip("/")}),
                )


def test_strip_headers_mirror_matches_egress():
    """Drift guard: our local strip tuple must equal egress's."""
    from tldw_chatbook.Utils import egress
    from tldw_chatbook.Model_Artifacts import fetch

    assert set(fetch._STRIP_HEADERS) == set(egress._STRIP_HEADERS)
```

- [ ] **Step 3: Run to verify failure**

Run: `<prefix> Tests/Model_Artifacts/test_stream_fetch.py -q`
Expected: FAIL — `ModuleNotFoundError: ... fetch`.

- [ ] **Step 4: Implement `fetch.py`**

```python
"""Streaming guarded fetch for managed artifact acquisition (TASK-595).

The egress hop loop (SSRF policy, hop cap, credential stripping) re-shaped
to stream to disk under a hard byte bound. The fetch-state sidecar is owned
by acquisition, NOT this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import httpx
from loguru import logger

from tldw_chatbook.Utils.egress import (
    MAX_REDIRECT_HOPS,
    check_url_or_raise_async,
)

# Mirrors egress._STRIP_HEADERS (private there); the drift-guard test in
# test_stream_fetch.py fails if the two sets ever diverge.
_STRIP_HEADERS = ("authorization", "cookie", "proxy-authorization", "x-goog-api-key")

_CHUNK_BYTES = 1024 * 1024


class FetchError(Exception):
    """Base for streaming-fetch failures. Messages never carry headers."""


class FetchRestartRequired(FetchError):
    """Resume impossible (validators changed/weak, or no Range support)."""


class FetchTooLargeError(FetchError):
    """The transfer would exceed the declared byte bound."""


class FetchTransportError(FetchError):
    """Network-level failure, wrapping httpx errors without header data."""


@dataclass(frozen=True)
class FetchValidators:
    """HTTP validators captured from a response, used for Range resume."""

    etag: str | None
    last_modified: str | None

    @property
    def strong(self) -> bool:
        """True when resuming on these validators is safe (strong ETag or
        a Last-Modified date; weak `W/` ETags never qualify)."""
        if self.etag and not self.etag.startswith("W/"):
            return True
        return self.etag is None and self.last_modified is not None


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one stream_fetch call."""

    bytes_written: int
    validators: FetchValidators
    resumed: bool


def _same_origin(a: httpx.URL, b: httpx.URL) -> bool:
    return (a.scheme, a.host, a.port) == (b.scheme, b.host, b.port)


async def stream_fetch(
    url: str,
    destination: Path,
    *,
    client: httpx.AsyncClient,
    max_bytes: int,
    resume_from: int = 0,
    validators: FetchValidators | None = None,
    headers: Mapping[str, str] | None = None,
    trusted_origins: frozenset[str] = frozenset(),
    on_chunk: Callable[[int], None] | None = None,
) -> FetchResult:
    """Stream a URL to disk with egress guards, byte bounds, and resume.

    Args:
        url: Source URL (http/https; egress policy enforced per hop).
        destination: File appended to (created if absent).
        client: Caller-owned AsyncClient (connection reuse, test injection).
        max_bytes: Hard bound on the FINAL file size (resume_from + written).
        resume_from: Durable bytes already on disk; sends a Range request.
        validators: Validators the existing bytes were fetched under; resume
            requires them strong and matching the server's current ones.
        headers: Extra headers (credentials); stripped on cross-origin hops.
        trusted_origins: Egress-policy trust additions (fixture servers).
        on_chunk: Called with each chunk's byte count (progress).

    Returns:
        FetchResult with bytes written THIS call, captured validators, and
        whether a Range continuation was used.

    Raises:
        FetchRestartRequired: Resume requested but unsafe/unsupported.
        FetchTooLargeError: Bound exceeded (pre-declared or mid-stream).
        FetchTransportError: Connection/protocol failures.
        EgressBlockedError: Policy rejection (from egress helpers).
    """
    if resume_from and not (validators and validators.strong):
        raise FetchRestartRequired("resume requires strong validators")
    if resume_from >= max_bytes:
        raise FetchTooLargeError("resume offset already at or past the bound")

    current = httpx.URL(url)
    origin = current
    request_headers: dict[str, str] = dict(headers or {})
    if resume_from:
        request_headers["Range"] = f"bytes={resume_from}-"
        if validators and validators.etag:
            request_headers["If-Range"] = validators.etag

    written = 0
    for _hop in range(MAX_REDIRECT_HOPS + 1):
        await check_url_or_raise_async(str(current), trusted_origins=trusted_origins)
        send_headers = dict(request_headers)
        if not _same_origin(origin, current):
            for name in list(send_headers):
                if name.lower() in _STRIP_HEADERS:
                    del send_headers[name]
        try:
            async with client.stream("GET", current, headers=send_headers,
                                     follow_redirects=False) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        raise FetchTransportError("redirect without location")
                    current = current.join(location)
                    continue
                if resume_from and response.status_code != 206:
                    # Server ignored Range (200 full body / no support).
                    raise FetchRestartRequired("server did not honor Range")
                if response.status_code == 401 or response.status_code == 403:
                    raise FetchTransportError(f"HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise FetchTransportError(f"HTTP {response.status_code}")
                got = FetchValidators(
                    etag=response.headers.get("etag"),
                    last_modified=response.headers.get("last-modified"),
                )
                if resume_from and validators and validators.etag and got.etag:
                    if got.etag != validators.etag:
                        raise FetchRestartRequired("validators changed upstream")
                mode = "ab" if resume_from else "wb"
                with open(destination, mode) as fh:
                    async for chunk in response.aiter_bytes(_CHUNK_BYTES):
                        if resume_from + written + len(chunk) > max_bytes:
                            raise FetchTooLargeError("byte bound exceeded")
                        fh.write(chunk)
                        written += len(chunk)
                        if on_chunk:
                            on_chunk(len(chunk))
                    fh.flush()
                    os.fsync(fh.fileno())
                return FetchResult(written, got, resumed=bool(resume_from))
        except httpx.HTTPError as exc:
            raise FetchTransportError(type(exc).__name__) from exc
    raise FetchTransportError("redirect hop limit exceeded")
```

- [ ] **Step 5: Run tests to verify pass**

Run: `<prefix> Tests/Model_Artifacts/test_stream_fetch.py -q`
Expected: PASS. Note: the egress policy may block 127.0.0.1 by classification — the `trusted_origins` passed by tests must satisfy `check_url_or_raise_async`; read `egress.evaluate_url_policy`'s trusted-origin format (`_normalize_trusted` :167) and adjust the tests' origin strings to that exact format if the first run rejects them (fix the TEST format, not the policy).

- [ ] **Step 6: Commit**

```bash
git add Tests/Model_Artifacts/fixture_http.py tldw_chatbook/Model_Artifacts/fetch.py Tests/Model_Artifacts/test_stream_fetch.py
git commit -m "feat(artifacts): streaming guarded fetch with Range resume + localhost fixture server (TASK-595)"
```

---

### Task 4: Acquisition types + catalog closure walk (pure)

**Files:**
- Create: `tldw_chatbook/Model_Artifacts/acquisition.py` (types + walk only this task)
- Test: `Tests/Model_Artifacts/test_acquisition_types.py` (create)

**Interfaces:**
- Consumes: `ArtifactRef`, `ArtifactDescriptor`, `closure_fingerprint` from `.service`.
- Produces (exact names later tasks import):
  - `ACQUISITION_SAFETY_MARGIN_BYTES = 256 * 1024 * 1024`; `MAX_FILE_REFETCHES = 1`
  - `class ArtifactCatalog(Protocol): def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor`
  - Errors: `AcquisitionError(ArtifactError)`; subclasses `CatalogError`, `ConsentMismatchError`, `PreflightNotGrantableError`, `AcquisitionBusyError`, `InsufficientSpaceError`, `GatedRepositoryError`, `TransferError(retryable: bool attr)`
  - `resolve_catalog_closure(root: ArtifactRef, catalog: ArtifactCatalog) -> tuple[ArtifactDescriptor, ...]` — stable order sorted by ref; cycle detection (`CatalogError`), conflict detection (same artifact_id at two different revisions ⇒ `CatalogError`), unknown ref ⇒ `CatalogError` (wrapping the catalog's exception)
  - `ArtifactPreflightEntry`, `PreflightReport` (all fields per spec incl. `already_staged_bytes`), `AcquisitionConsent`, `AcquisitionProgress` with `phase: Literal["fetch", "pre-verify", "verify-install", "activate"]` — frozen dataclasses exactly as the spec's API section
  - `PreflightReport.grant() -> AcquisitionConsent` — raises `PreflightNotGrantableError` when `gating_errors` is non-empty or `sufficient_space` is False

- [ ] **Step 1: Write the failing tests**

```python
"""TASK-595 Task 4: pure acquisition types and the catalog closure walk."""

import pytest

from tldw_chatbook.Model_Artifacts.acquisition import (
    AcquisitionConsent,
    ArtifactPreflightEntry,
    CatalogError,
    PreflightNotGrantableError,
    PreflightReport,
    resolve_catalog_closure,
)
from tldw_chatbook.Model_Artifacts.service import ArtifactRef, closure_fingerprint

# Build minimal descriptors via the same helper approach as Task 1's tests
from Tests.Model_Artifacts.test_service import make_descriptor  # adjust name


class DictCatalog:
    def __init__(self, mapping):
        self._m = mapping

    def descriptor(self, ref):
        return self._m[ref]


def _ref(a="root", r="r1", v="int8"):
    return ArtifactRef(a, r, v)


def test_closure_walk_resolves_dependencies_in_stable_order():
    dep = _ref("aaa-dep")
    root = _ref("root")
    catalog = DictCatalog({
        root: make_descriptor(ref=root, dependencies=(dep,)),
        dep: make_descriptor(ref=dep),
    })
    closure = resolve_catalog_closure(root, catalog)
    ids = [d.artifact_id for d in closure]
    assert ids == sorted(ids)
    assert len(closure) == 2


def test_closure_walk_detects_cycles():
    a, b = _ref("a"), _ref("b")
    catalog = DictCatalog({
        a: make_descriptor(ref=a, dependencies=(b,)),
        b: make_descriptor(ref=b, dependencies=(a,)),
    })
    with pytest.raises(CatalogError):
        resolve_catalog_closure(a, catalog)


def test_closure_walk_detects_revision_conflicts():
    dep1, dep2 = _ref("dep", "r1"), _ref("dep", "r2")
    a, b = _ref("a"), _ref("b")
    root = _ref("root")
    catalog = DictCatalog({
        root: make_descriptor(ref=root, dependencies=(a, b)),
        a: make_descriptor(ref=a, dependencies=(dep1,)),
        b: make_descriptor(ref=b, dependencies=(dep2,)),
        dep1: make_descriptor(ref=dep1),
        dep2: make_descriptor(ref=dep2),
    })
    with pytest.raises(CatalogError):
        resolve_catalog_closure(root, catalog)


def test_unknown_ref_is_a_typed_error():
    with pytest.raises(CatalogError):
        resolve_catalog_closure(_ref("missing"), DictCatalog({}))


def _report(**overrides):
    defaults = dict(
        root=_ref(), closure_fingerprint="f" * 64,
        entries=(), download_bytes=0, already_staged_bytes=0,
        staging_overhead_bytes=0, retained_bytes=0,
        destination=__import__("pathlib").Path("/tmp/x"),
        free_bytes=10**12, required_bytes=10**6, sufficient_space=True,
        gating_errors=(),
    )
    defaults.update(overrides)
    return PreflightReport(**defaults)


def test_grant_returns_consent_with_fingerprint():
    consent = _report().grant()
    assert isinstance(consent, AcquisitionConsent)
    assert consent.closure_fingerprint == "f" * 64


def test_grant_refuses_gating_errors_and_insufficient_space():
    with pytest.raises(PreflightNotGrantableError):
        _report(gating_errors=("token required",)).grant()
    with pytest.raises(PreflightNotGrantableError):
        _report(sufficient_space=False).grant()
```

`make_descriptor` must accept `ref=` and `dependencies=` for these tests — if the existing helper doesn't, write a local one constructing `ArtifactDescriptor` with all mandatory fields (copy the field list from an existing test's construction).

- [ ] **Step 2: Run to verify failure** — `<prefix> Tests/Model_Artifacts/test_acquisition_types.py -q` → ImportError.

- [ ] **Step 3: Implement** the types exactly as the spec's API section (copy field lists verbatim), the error family, and:

```python
def resolve_catalog_closure(
    root: ArtifactRef, catalog: ArtifactCatalog
) -> tuple[ArtifactDescriptor, ...]:
    """Resolve the full dependency closure from CATALOG descriptors.

    Deliberately not the core's _resolve_closure (which reads installed
    manifests): at preflight, dependencies may not be installed at all.
    Same rules: cycle and revision-conflict detection; stable sorted order.

    Raises:
        CatalogError: Unknown ref, dependency cycle, or two revisions of
            one artifact_id in the same closure.
    """
    resolved: dict[ArtifactRef, ArtifactDescriptor] = {}
    revisions: dict[str, ArtifactRef] = {}
    visiting: set[ArtifactRef] = set()

    def visit(ref: ArtifactRef) -> None:
        if ref in resolved:
            return
        if ref in visiting:
            raise CatalogError(f"dependency cycle at {ref.artifact_id}")
        seen = revisions.get(ref.artifact_id)
        if seen is not None and seen != ref:
            raise CatalogError(
                f"conflicting revisions for {ref.artifact_id}: {seen.revision} vs {ref.revision}"
            )
        visiting.add(ref)
        try:
            descriptor = catalog.descriptor(ref)
        except Exception as exc:
            raise CatalogError(f"unknown artifact {ref.artifact_id}@{ref.revision}") from exc
        revisions[ref.artifact_id] = ref
        for dep in descriptor.dependencies:
            visit(dep)
        visiting.discard(ref)
        resolved[ref] = descriptor

    visit(root)
    return tuple(resolved[ref] for ref in sorted(resolved))
```

- [ ] **Step 4: Run** — all Task 4 tests PASS.
- [ ] **Step 5: Commit** — `feat(artifacts): acquisition types, errors, catalog closure walk (TASK-595)`

---

### Task 5: `preflight()`

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/acquisition.py`
- Test: `Tests/Model_Artifacts/test_preflight.py` (create)

**Interfaces:**
- Consumes: Task 4 types/walk; `ModelArtifactService.list_installed()`, `.artifact_path()`, `.disk_usage()`, `.staging_path`; Task 3 fixture server; `FetchValidators`.
- Produces: `ArtifactAcquisitionService(core, *, client_factory=None, credential_resolver=None, free_bytes_probe=None)` and `async def preflight(self, root: ArtifactRef, catalog: ArtifactCatalog) -> PreflightReport`.

**Behavior to implement (all spec'd):** closure via `resolve_catalog_closure` (executor not needed — pure); per-entry `already_installed` from `core.list_installed()` refs; `download_bytes` = Σ total_bytes of not-installed entries minus `already_staged_bytes` (sidecar `bytes_done` sums read from `staging/managed/...`; best-effort); `staging_overhead_bytes = 0` under consume_source semantics (document why in a comment: moves, not copies — keep the field for honesty if semantics change); `retained_bytes` = installed size of the currently active version of the root's artifact_id when an upgrade would retain it (from `list_installed()` + active state); `required_bytes = download_bytes + staging_overhead_bytes + retained_bytes + ACQUISITION_SAFETY_MARGIN_BYTES`; `free_bytes` from injected probe or `core.disk_usage().free_bytes`; one bounded HEAD per unique repository host over the first not-installed entry's URL (client from `client_factory` or a short-lived `httpx.AsyncClient`), 401/403 ⇒ a `gating_errors` entry naming the repository and the credential env var to set — never the token value.

- [ ] **Step 1: Write the failing tests** (fixture server + `DictCatalog` from Task 4's test module; inject `free_bytes_probe`):

```python
@pytest.mark.asyncio
async def test_preflight_aggregates_and_grants(tmp_path):
    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    with FixtureArtifactServer() as srv:
        body = b"m" * 2048
        srv.serve("/m.onnx", body)
        root = ArtifactRef("root-model", "r1", "int8")
        catalog = DictCatalog({root: make_descriptor(
            ref=root, files_body=body, source_url=srv.url("/m.onnx"))})
        report = await svc.preflight(root, catalog)
    assert report.download_bytes == 2048
    assert report.sufficient_space is True
    assert report.entries[0].already_installed is False
    report.grant()  # must not raise


@pytest.mark.asyncio
async def test_preflight_counts_staged_credit(tmp_path):
    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    root = ArtifactRef("root-model", "r1", "int8")
    staged = Path(core.staging_path) / "managed" / "root-model" / "r1" / "int8"
    staged.mkdir(parents=True)
    (staged / "fetch-state.json").write_text(json.dumps(
        {"files": {"m.onnx": {"etag": '"v1"', "last_modified": None,
                                "bytes_done": 500, "complete": False}}}))
    with FixtureArtifactServer() as srv:
        body = b"m" * 2048
        srv.serve("/m.onnx", body)
        catalog = DictCatalog({root: make_descriptor(
            ref=root, files_body=body, source_url=srv.url("/m.onnx"))})
        report = await svc.preflight(root, catalog)
    assert report.already_staged_bytes == 500
    assert report.download_bytes == 2048 - 500


@pytest.mark.asyncio
async def test_preflight_insufficient_space_blocks_grant(tmp_path):
    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10)
    with FixtureArtifactServer() as srv:
        body = b"m" * 2048
        srv.serve("/m.onnx", body)
        root = ArtifactRef("root-model", "r1", "int8")
        catalog = DictCatalog({root: make_descriptor(
            ref=root, files_body=body, source_url=srv.url("/m.onnx"))})
        report = await svc.preflight(root, catalog)
    assert report.sufficient_space is False
    with pytest.raises(PreflightNotGrantableError):
        report.grant()


@pytest.mark.asyncio
async def test_preflight_gated_repo_reports_instructions(tmp_path):
    core = ModelArtifactService(tmp_path / "root")
    svc = ArtifactAcquisitionService(core, free_bytes_probe=lambda p: 10**12)
    with FixtureArtifactServer() as srv:
        body = b"m" * 2048
        srv.serve("/m.onnx", body, require_token="tok-secret")
        root = ArtifactRef("root-model", "r1", "int8")
        catalog = DictCatalog({root: make_descriptor(
            ref=root, files_body=body, source_url=srv.url("/m.onnx"))})
        report = await svc.preflight(root, catalog)
    assert report.gating_errors, "401 repo must surface a gating error"
    assert all("tok-secret" not in message for message in report.gating_errors)
    with pytest.raises(PreflightNotGrantableError):
        report.grant()
```

- [ ] **Step 2: Run** → ImportError/AttributeError. **Step 3: Implement** per Behavior above. **Step 4: Run** → PASS. **Step 5: Commit** — `feat(artifacts): consent preflight with staged credit, space math, gating probe (TASK-595)`

---

### Task 6: Session lease + `provision()` skeleton

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/acquisition.py`; `tldw_chatbook/Model_Artifacts/service.py` (export `_ACQUISITION_SESSION_KEY` created in Task 2 as public `ACQUISITION_SESSION_LEASE_KEY`)
- Test: `Tests/Model_Artifacts/test_provision_serialization.py` (create)

**Interfaces:**
- Consumes: `ArtifactOperationLease` (`leases.py:132`, `timeout_seconds`), Task 4/5 surfaces.
- Produces: `async def provision(self, consent: AcquisitionConsent, catalog: ArtifactCatalog, *, progress=None) -> ArtifactRef` — this task delivers: in-process `asyncio.Lock`; session lease acquired non-blocking in an executor (`timeout_seconds=0.1` ⇒ `AcquisitionBusyError` on `ArtifactLeaseTimeoutError`); fingerprint recompute + `ConsentMismatchError`; free-space recheck (`InsufficientSpaceError`); then per-artifact phases as METHOD STUBS (`_fetch_artifact`, `_preverify_artifact`, `_install_artifact`) raising `NotImplementedError` — Tasks 7–8 fill them; `activate` call wiring; the already-installed skip path (skips straight past stubs, so the idempotent-completion test passes NOW).

- [ ] **Step 1: Failing tests**: (a) two concurrent `provision()` calls in one process — second waits on the asyncio.Lock (assert serialized via event ordering); (b) cross-process busy: hold the session lease from a spawned process (reuse `Tests/Model_Artifacts/lease_processes.py` helpers) → `AcquisitionBusyError`; (c) fingerprint drift: consent minted from one catalog, provision with a mutated catalog → `ConsentMismatchError`; (d) fully-installed closure: pre-install via `core.install` + `core.activate`, then provision → returns root without touching the stubs (assert stubs not called via monkeypatch counters).
- [ ] **Step 2: Run** → fail. **Step 3: Implement.** **Step 4: Run** → pass, plus `Tests/Model_Artifacts/test_operation_leases*.py` still green. **Step 5: Commit** — `feat(artifacts): provision skeleton — session lease, busy semantics, consent drift check, idempotent completion (TASK-595)`

---

### Task 7: Provision fetch phase (durable staging, sidecar, resume)

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/acquisition.py` (`_fetch_artifact`)
- Test: `Tests/Model_Artifacts/test_provision_fetch.py` (create)

**Interfaces:**
- Consumes: `stream_fetch` + its errors; `atomic_write_json` (same import the core uses); sidecar shape `{"files": {name: {"etag", "last_modified", "bytes_done", "complete"}}}`.
- Produces: `_fetch_artifact(descriptor, staging_dir, progress_cb)` — for each declared file: read sidecar; complete files skipped; partial with strong matching validators → `stream_fetch(resume_from=...)`; `FetchRestartRequired` → truncate file, reset sidecar entry, fetch from zero; sidecar written (atomic, fsynced) only AFTER data fsync (stream_fetch fsyncs); `max_bytes = file.size_bytes` exactly (the descriptor IS the bound — a longer body is corrupt by definition); progress `phase="fetch"` events with closure-wide `bytes_done/bytes_total`; cancellation honored between chunks (asyncio-native).

- [ ] **Step 1: Failing tests** (fixture server): resume-after-partial (pre-seed file+sidecar; assert Range request seen + final bytes correct); changed-ETag restart (assert refetched from zero, sidecar updated); mid-body disconnect leaves durable sidecar ≤ file bytes and a re-provision completes; ENOSPC (monkeypatch the file write to raise `OSError(errno.ENOSPC, ...)`) → `TransferError` with `retryable=True` and staging retained; **cancellation** — start `provision()` as a task over a slow route (`disconnect_after=None`, large body, throttled by a chunk-callback event), `task.cancel()` mid-fetch, assert `asyncio.CancelledError` propagates, the session lease is released, sidecar reflects only durable bytes, and the prior active artifact (pre-seed one via `core.install`+`activate`) is still active and `acquire()`-able (AC #3's cancellation guarantee, spec's chunk-boundary semantics).
- [ ] **Steps 2–4:** red → implement → green. **Step 5: Commit** — `feat(artifacts): resumable per-artifact fetch with durable sidecars (TASK-595)`

---

### Task 8: Pre-verify + install + activate phases

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/acquisition.py` (`_preverify_artifact`, `_install_artifact`, activate wiring)
- Test: `Tests/Model_Artifacts/test_provision_install.py` (create)

**Interfaces:**
- Consumes: Task 1's `install(..., consume_source=True)`; `core.activate`; `hashlib.sha256` streaming.
- Produces: `_preverify_artifact` — streaming SHA-256 per file with `phase="pre-verify"` byte progress; mismatch → delete that file, reset its sidecar entry, refetch up to `MAX_FILE_REFETCHES`, then `TransferError(retryable=True)`; `_install_artifact` — executor hop to `core.install(descriptor, staging_dir, consume_source=True)`, then remove the (now-emptied) staging dir + sidecar; `phase="verify-install"` per-artifact events; after all artifacts: executor `core.activate(root_ref)`, `phase="activate"` event; return the root ref.

- [ ] **Step 1: Failing tests:** end-to-end provision happy path over the fixture server (2-artifact closure: root + dependency; assert installed, activated, staging gone, progress phases seen in order fetch→pre-verify→verify-install→activate); corrupt-payload (serve wrong bytes; assert exactly one refetch attempt — count server requests — then `TransferError`, nothing installed); crash-after-install completion (pre-install both artifacts via core, no activate; provision → activates without any fetch — zero fixture requests).
- [ ] **Steps 2–4:** red → implement → green, plus full `Tests/Model_Artifacts/ -q` green. **Step 5: Commit** — `feat(artifacts): pre-verified consume-source install and activation phases (TASK-595)`

---

### Task 9: Credentials, gated repos, secret hygiene, import boundary

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/acquisition.py` (CredentialResolver protocol + default resolver)
- Test: `Tests/Model_Artifacts/test_credentials_and_boundaries.py` (create)

**Interfaces:**
- Consumes: `config.get_cli_setting`; existing env precedence (`HUGGINGFACE_API_KEY` per `config.py:960`, `HF_TOKEN` per `Constants.py:1792`).
- Produces: `class CredentialResolver(Protocol): def resolve(self, repository: str) -> str | None`; `EnvConfigCredentialResolver` (env → config; keyring deliberately deferred with a comment citing the spec's "where available"); acquisition attaches `Authorization: Bearer <token>` ONLY for the entry's own origin (fetch strips on cross-origin anyway — defense in depth).

- [ ] **Step 1: Failing tests:** 401-until-token fixture — without resolver → preflight `gating_errors`; with a resolver returning "tok" → provision succeeds AND `caplog` (all levels) contains no "tok" AND the sidecar file bytes contain no "tok" AND no `TransferError`/gating string contains "tok"; import boundary — walk the modules the existing `Tests/STT/test_boundaries.py` guards (read it and reuse its module list/mechanism) asserting none of them import `tldw_chatbook.Model_Artifacts.acquisition` or `.fetch`.
- [ ] **Steps 2–4:** red → implement → green. **Step 5: Commit** — `feat(artifacts): credential resolution without persistence; worker import boundary (TASK-595)`

---

### Task 10: Crash recovery, containment, exports, close-out

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/__init__.py` (export the new public surface)
- Test: `Tests/Model_Artifacts/test_provision_crash_recovery.py` (create)
- Modify: `backlog/tasks/task-595 - *.md`

**Interfaces:**
- Consumes: everything; the process harness (`lease_processes.py` — read it; it spawns real subprocesses and kills them).

- [ ] **Step 1: Failing tests:** subprocess provisions against a slow fixture route (`disconnect_after` high, big body), parent `kill -9`s it mid-fetch → assert: valid-sidecar staging survives `reconcile()` (Task 2's rule), session lease is free (OS release), a fresh provision resumes (Range request observed) and completes; a second scenario kills between install and activate → fresh provision activates with zero fetch requests. Containment: after the crash, `reconcile()`'s `staging_removed` names only orphans, nothing outside staging (assert unrelated tmp_path files untouched).
- [ ] **Step 2–4:** red → implement fixes if any surface → green; then the full gate:

Run: `<prefix> Tests/Model_Artifacts/ Tests/STT/test_boundaries.py -q`
Expected: ALL PASS (including the untouched 594 suites).

- [ ] **Step 5: Exports + backlog close-out**

Add to `Model_Artifacts/__init__.py.__all__`: `ArtifactAcquisitionService`, `ArtifactCatalog`, `PreflightReport`, `ArtifactPreflightEntry`, `AcquisitionConsent`, `AcquisitionProgress`, `AcquisitionError`, `CatalogError`, `ConsentMismatchError`, `PreflightNotGrantableError`, `AcquisitionBusyError`, `InsufficientSpaceError`, `GatedRepositoryError`, `TransferError`, `stream_fetch`, `FetchValidators`, `FetchResult`, `ACQUISITION_SESSION_LEASE_KEY`.

```bash
backlog task edit 595 -s Done --notes "ArtifactAcquisitionService (async preflight/consent/provision) + stream_fetch over the sealed 594 core; consume_source install + orphans-only staging GC as the only core changes; session-lease serialization with busy semantics; resume/pre-verify/crash-recovery/credential fixtures per AC #6. Spec: Docs/superpowers/specs/2026-07-30-managed-model-acquisition-design.md."
git add -A && git commit -m "feat(artifacts): managed acquisition crash recovery, exports, TASK-595 close-out"
```

If implementation surfaced a generalizable trap, add it to `backlog/docs/lessons-*.md` with the incident, per CLAUDE.md.

---

## Plan Self-Review Notes

- **Spec coverage:** preflight math incl. staged credit + gating probe (T5), consent/grant/drift (T4/T6), session lease + busy + in-process lock (T6), durable sidecar fetch with resume/validators/bounds (T3/T7), pre-verify + refetch-once + consume_source + activate-last + idempotent completion (T8), credentials + secret hygiene + import boundary (T9), crash recovery + containment + GC rules (T2/T10), core additions exactly two (T1/T2), never-trap error taxonomy (T4, exercised across T5–T9).
- **Deliberate simplifications, documented in code:** `staging_overhead_bytes = 0` under consume_source semantics (field retained for honesty); keyring resolver deferred with comment; sequential file transfers (spec non-goal).
- **Verify-don't-assume points named in tasks:** test_service helper import names (T1/T4), containment idiom in `service.py` (T1/T2), `reconcile`'s existing lease usage (T2), egress trusted-origin format (T3), `test_boundaries.py` mechanism (T9), process-harness API (T10).
