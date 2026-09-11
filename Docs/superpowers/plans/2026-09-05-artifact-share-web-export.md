# Artifact Share Web Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From the Artifacts screen, share a selection of local Chatbook artifacts as a temporary, optionally password-protected web page (hosted by an aiohttp child process serving a staged snapshot) that recipients use to download bundles.

**Architecture:** The TUI stages immutable copies of the selected zips plus a pre-built `bundle.zip` into `<user_data>/share/<id>/` with a `0600` manifest, then spawns `python -m tldw_chatbook.Web_Server.artifact_share_server` (own process group). The child serves an HTML index, per-artifact downloads, the bundle, and `index.json`, behind optional HTTP Basic auth; it never touches the private chatbooks directory. An app-owned `ArtifactShareController` supervises the child; the share survives screen navigation and dies with the app.

**Tech Stack:** Python ≥3.11, aiohttp (existing `[web]` extra — **no new dependencies**), pydantic, Textual 8.2.8 (`SelectionList`, `ModalScreen`), pytest with `pytest-asyncio` auto mode.

**Spec:** `Docs/superpowers/specs/2026-09-05-artifact-share-web-export-design.md` (read it first — the plan argues from it)
**Backlog task:** TASK-31978 (`backlog/tasks/task-31978 - Add-artifact-share-web-export-from-Artifacts-screen.md`)
**ADR:** `backlog/decisions/123-artifact-share-web-export.md` (Task 1)

## Global Constraints

- Execute in an isolated worktree created from `origin/dev` (superpowers:using-git-worktrees). All paths/line numbers below reference `origin/dev` commit `d3bf8b5397`.
- No new runtime or test dependencies. aiohttp/textual-serve availability comes from the existing `[web]` extra; gate via `Web_Server.is_web_server_available()` / `Utils/optional_deps.check_web_server_deps()`. Import aiohttp **lazily inside functions/methods** (pattern of `Web_Server/serve.py`) so module import never fails without the extra.
- Share files are written with mode `0o600` via `Utils/atomic_file_ops` (`atomic_write_json`, `atomic_copy`). Never log usernames, passwords, or derived keys. `html.escape(..., quote=True)` every user-derived string in HTML output.
- Keybinding: ADR-031 — single-letter htop-style (`s` = Share), no reserved globals, footer hints must stay 1:1 with working bindings.
- pytest: `asyncio_mode = auto`, `--strict-markers`. Server tests that bind sockets use `pytest.mark.loopback_network` (the autouse network guard blocks otherwise); UI tests use `pytest.mark.ui`; run **targeted files only** — never a full suite sweep without explicit user opt-in.
- Commit style: conventional commits (`feat(web-server): …`, `test(ui): …`, `docs: …`), one commit per task.

---

### Task 1: ADR 123 — artifact share web export

**Files:**
- Create: `backlog/decisions/123-artifact-share-web-export.md`

**Interfaces:**
- Consumes: the approved spec.
- Produces: ADR number 123 that later tasks and the backlog task reference.

- [ ] **Step 1: Write the ADR**

Create `backlog/decisions/123-artifact-share-web-export.md` following the repo ADR template (`backlog/decisions/000-template.md`). Content (adapt wording, keep the decisions):

```markdown
# ADR-123: Artifact share web export via staged snapshot child server

- **Status:** Accepted
- **Date:** 2026-09-05
- **Scope:** Web serving, security surface, child-process lifecycle
- **Amends/relates:** none (task-31230 web auth remains scoped to the full-app web server)

## Context

Users need to hand Chatbook artifacts to other people. The only existing paths are
manual file transfer of `.zip` bundles from the private chatbooks directory.
`Web_Server/serve.py` serves the full TUI via textual-serve — far more exposure
and machinery than a download page needs.

## Decision

1. Sharing is an explicit, ephemeral session started from the Artifacts screen:
   the app stages immutable copies of the selected artifact zips (plus a
   pre-built bundle.zip) into `<user_data>/share/<id>/` with a `0600` manifest,
   then spawns `tldw_chatbook.Web_Server.artifact_share_server` as a child
   process (own process group) that serves ONLY the staging directory.
2. The recipient experience is a plain HTML page (no JavaScript) with per-file
   and bundle downloads plus `index.json`; aiohttp comes from the existing
   `[web]` extra. textual-serve remains the stack for full-app web mode and is
   unchanged; the share server reuses its dependency gate and download
   semantics (`Content-Disposition: attachment`, streamed `FileResponse`).
3. Auth is optional HTTP Basic (single shared username/password), PBKDF2
   verifier in the manifest, constant-time compare, per-IP lockout
   (10 failures / 30 s) tuned gentle because recipients may share a NAT IP.
   Non-loopback bind without a password requires typed confirmation.
   Auth covers every route; there are no unauthenticated metadata channels.
4. Lifecycle: the controller is app-owned (share survives navigation), one
   active share at a time, SIGTERM→SIGKILL escalation on stop, PPID orphan
   guard in the child, and a dead-PID startup sweep clears crash residue.

## Consequences

- Staging freezes content: later library edits/deletions cannot alter or break
  a running share; the child never reads the private chatbooks directory, so
  containment is single-root.
- The zip IS the artifact: registry records may drift cosmetically from bundle
  bytes (`update_chatbook` never rewrites the zip); v1 shares existing bytes
  and `index.json` hashes what recipients actually download. Re-export at
  share time is future work.
- Plain HTTP: Basic auth is an access gate, not confidentiality. Hostile
  networks require a reverse proxy (documented in the user guide).
- No per-recipient accounts, persistence, TLS, or upload — stop = revoke.

## Alternatives considered

- In-process aiohttp worker thread: rejected — event-loop teardown against
  screen/app exit is fragile; crashes destabilize the TUI.
- textual-serve `Server` subclass with a read-only viewer Textual app:
  rejected for v1 — heavy browser-terminal recipient UX and a second Textual
  app to maintain; the manifest/auth/staging work is reusable if wanted later.
- Serving live library paths: rejected — registry records may lack
  `file_path`, mid-share mutation/deletion breaks shares, and the child would
  need private-directory access.
```

- [ ] **Step 2: Commit**

```bash
git add backlog/decisions/123-artifact-share-web-export.md
git commit -m "docs: add ADR-123 for artifact share web export"
```

---

### Task 2: Manifest, staging, and auth verifier module

**Files:**
- Create: `tldw_chatbook/Web_Server/artifact_share_manifest.py`
- Test: `Tests/Web_Server/test_artifact_share_manifest.py`

**Interfaces:**
- Consumes: `Utils.atomic_file_ops.atomic_write_json(file_path, data, *, mode=0o600, privacy_safe_log=True)`, `Utils.atomic_file_ops.atomic_copy(src, dst, mode=None)`; `Subscriptions.security.CredentialEncryptor.derive_key_from_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]` (static, PBKDF2-HMAC-SHA256, 100 000 iterations); `Utils.paths.get_user_data_dir() -> Path`.
- Produces (used by Tasks 3–6):
  - `class ArtifactShareError(RuntimeError)`, `class ArtifactShareStagingError(ArtifactShareError)`
  - `def new_artifact_key() -> str`
  - `def slugify_share_name(name: str) -> str`
  - `class ArtifactShareAuth(BaseModel)` with fields `username: str`, `pbkdf2_salt_hex: str`, `pbkdf2_hash_hex: str`
  - `def build_share_auth(username: str, password: str) -> ArtifactShareAuth`
  - `def verify_share_auth(auth: ArtifactShareAuth, username: str, password: str) -> bool`
  - `class SharedArtifact(BaseModel)` with fields `key, display_name, description, kind, size_bytes, sha256, source_chatbook_id, staged_name`
  - `class ArtifactShareManifest(BaseModel)` with fields `share_id, share_name, created_at, auth, artifacts` (JSON key `"schema"` = 1 via alias)
  - `def share_root_dir() -> Path` (`<user_data>/share`)
  - `def stage_share(records: list[dict], *, share_name: str, auth: ArtifactShareAuth | None, share_root: Path | None = None) -> ArtifactShareManifest`
  - `def load_manifest(manifest_path: Path) -> ArtifactShareManifest` (fail closed)
  - `def sweep_stale_shares(share_root: Path | None = None) -> list[Path]`
  - `def pid_alive(pid: int) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Web_Server/test_artifact_share_manifest.py`:

```python
"""Manifest/staging/auth tests for the artifact share web export."""

import json
import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook.Web_Server.artifact_share_manifest import (
    ArtifactShareAuth,
    ArtifactShareStagingError,
    build_share_auth,
    load_manifest,
    pid_alive,
    share_root_dir,
    stage_share,
    sweep_stale_shares,
    verify_share_auth,
)

pytestmark = pytest.mark.unit


def _make_zip(path: Path, payload: bytes = b"chatbook-zip-bytes") -> Path:
    path.write_bytes(payload)
    return path


def _record(tmp_path: Path, name: str, *, file_name: str | None = None, cid: int = 1) -> dict:
    file_path = None if file_name is None else str(_make_zip(tmp_path / file_name))
    return {
        "id": str(cid),
        "chatbook_id": cid,
        "name": name,
        "description": f"desc for {name}",
        "file_path": file_path,
        "created_at": "2026-09-05T00:00:00+00:00",
        "updated_at": "2026-09-05T00:00:00+00:00",
    }


def test_stage_share_copies_and_hashes_and_writes_manifest_0600(tmp_path):
    root = tmp_path / "share-root"
    records = [
        _record(tmp_path, "Weekly Report", file_name="a.zip", cid=1),
        _record(tmp_path, "Notes <b>bold</b>", file_name="b.zip", cid=2),
    ]
    auth = build_share_auth("alice", "secret-pass")

    manifest = stage_share(
        records, share_name="My Library", auth=auth, share_root=root
    )

    share_dir = root / manifest.share_id
    manifest_path = share_dir / "manifest.json"
    assert manifest_path.is_file()
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert len(manifest.artifacts) == 2
    for item in manifest.artifacts:
        staged = share_dir / item.staged_name
        assert staged.is_file()
        assert stat.S_IMODE(staged.stat().st_mode) == 0o600
        assert staged.read_bytes() in (b"chatbook-zip-bytes",)
        assert item.size_bytes == staged.stat().st_size
    # user-controlled names never become raw filenames
    assert all("<" not in item.staged_name for item in manifest.artifacts)
    # bundle.zip exists and contains both staged files
    import zipfile

    with zipfile.ZipFile(share_dir / "bundle.zip") as bundle:
        assert sorted(bundle.namelist()) == sorted(
            item.staged_name for item in manifest.artifacts
        )
    # round-trip through the on-disk JSON
    loaded = load_manifest(manifest_path)
    assert loaded.share_name == "My Library"
    assert loaded.auth is not None and loaded.auth.username == "alice"
    assert {i.key for i in loaded.artifacts} == {i.key for i in manifest.artifacts}
    assert json.loads(manifest_path.read_text())["schema"] == 1


def test_stage_share_rejects_record_without_bundle(tmp_path):
    records = [_record(tmp_path, "Ghost", file_name=None, cid=3)]
    with pytest.raises(ArtifactShareStagingError, match="Ghost"):
        stage_share(records, share_name="x", auth=None, share_root=tmp_path / "r")
    # fail-closed: no share directory left behind
    assert not (tmp_path / "r").exists() or not any((tmp_path / "r").iterdir())


def test_stage_share_rejects_missing_file(tmp_path):
    records = [_record(tmp_path, "Vanished", file_name="gone.zip", cid=4)]
    Path(tmp_path / "gone.zip").unlink(missing_ok=True)
    with pytest.raises(ArtifactShareStagingError, match="Vanished"):
        stage_share(records, share_name="x", auth=None, share_root=tmp_path / "r")


def test_auth_verify_roundtrip_and_reject():
    auth = build_share_auth("alice", "secret-pass")
    assert isinstance(auth, ArtifactShareAuth)
    assert verify_share_auth(auth, "alice", "secret-pass") is True
    assert verify_share_auth(auth, "alice", "wrong") is False
    assert verify_share_auth(auth, "bob", "secret-pass") is False


def test_sweep_removes_dead_pid_and_keeps_live(tmp_path):
    root = tmp_path / "share-root"
    dead = root / "deadbeef"
    live = root / "cafebabe"
    for entry in (dead, live):
        entry.mkdir(parents=True)
        (entry / "manifest.json").write_text("{}")
    (dead / "status.json").write_text(json.dumps({"pid": 999999999}))
    (live / "status.json").write_text(json.dumps({"pid": os.getpid()}))

    removed = sweep_stale_shares(root)

    assert removed == [dead]
    assert not dead.exists()
    assert live.exists()


def test_pid_alive_current_and_bogus():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(999999999) is False


def test_share_root_dir_under_user_data():
    assert share_root_dir().name == "share"
    assert share_root_dir().parent.name != ""  # anchored under the user data dir
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Web_Server/test_artifact_share_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Web_Server.artifact_share_manifest'`

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Web_Server/artifact_share_manifest.py`:

```python
# artifact_share_manifest.py
"""Staging, manifest, and auth-verifier model for artifact share sessions.

The share server child process and the app-side controller both consume this
module; it must stay importable without the ``[web]`` extra (no aiohttp here).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..Subscriptions.security import CredentialEncryptor
from ..Utils.atomic_file_ops import atomic_copy, atomic_write_json
from ..Utils.paths import get_user_data_dir

ARTIFACT_SHARE_SCHEMA = 1
_STAGED_FILE_MODE = 0o600


class ArtifactShareError(RuntimeError):
    """Base error for artifact share orchestration."""


class ArtifactShareStagingError(ArtifactShareError):
    """A selected artifact could not be staged for sharing."""


def new_artifact_key() -> str:
    """Return a fresh 128-bit opaque URL-safe share key."""
    return secrets.token_urlsafe(16)


def slugify_share_name(name: str) -> str:
    """Reduce an arbitrary display name to a safe filename fragment."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-.")
    return cleaned[:60] or "artifact"


class ArtifactShareAuth(BaseModel):
    """Single shared Basic-auth verifier (PBKDF2-HMAC-SHA256, 100k iterations)."""

    username: str
    pbkdf2_salt_hex: str
    pbkdf2_hash_hex: str


def build_share_auth(username: str, password: str) -> ArtifactShareAuth:
    key, salt = CredentialEncryptor.derive_key_from_password(password)
    return ArtifactShareAuth(
        username=username,
        pbkdf2_salt_hex=salt.hex(),
        pbkdf2_hash_hex=key.hex(),
    )


def verify_share_auth(auth: ArtifactShareAuth, username: str, password: str) -> bool:
    if not hmac.compare_digest(username.encode("utf-8"), auth.username.encode("utf-8")):
        return False
    key, _unused_salt = CredentialEncryptor.derive_key_from_password(
        password, bytes.fromhex(auth.pbkdf2_salt_hex)
    )
    return hmac.compare_digest(key.hex(), auth.pbkdf2_hash_hex)


class SharedArtifact(BaseModel):
    key: str
    display_name: str
    description: str = ""
    kind: str = "chatbook"
    size_bytes: int
    sha256: str
    source_chatbook_id: int | str
    staged_name: str


class ArtifactShareManifest(BaseModel):
    model_config = {"populate_by_name": True}

    schema_version: int = Field(default=ARTIFACT_SHARE_SCHEMA, alias="schema")
    share_id: str
    share_name: str
    created_at: str
    auth: ArtifactShareAuth | None = None
    artifacts: list[SharedArtifact]


def share_root_dir() -> Path:
    return get_user_data_dir() / "share"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_share(
    records: list[dict[str, Any]],
    *,
    share_name: str,
    auth: ArtifactShareAuth | None,
    share_root: Path | None = None,
) -> ArtifactShareManifest:
    """Copy selected artifact bundles into a fresh staging directory.

    Raises ArtifactShareStagingError naming the offending artifact when a
    record has no on-disk bundle; the staging directory is removed on any
    failure (fail closed).
    """
    root = Path(share_root) if share_root is not None else share_root_dir()
    share_id = uuid.uuid4().hex
    share_dir = root / share_id
    share_dir.mkdir(parents=True, exist_ok=False)
    try:
        staged: list[SharedArtifact] = []
        used_names: set[str] = set()
        for record in records:
            display_name = str(record.get("name") or "artifact")
            raw_path = record.get("file_path")
            source = Path(str(raw_path)).expanduser() if raw_path else None
            if source is None or not source.is_file():
                raise ArtifactShareStagingError(
                    f"Artifact '{display_name}' has no exported bundle on disk."
                )
            chatbook_id = record.get("chatbook_id", record.get("id"))
            base = f"{chatbook_id}-{slugify_share_name(display_name)}"
            candidate = f"{base}.zip"
            suffix = 1
            while candidate in used_names:
                suffix += 1
                candidate = f"{base}-{suffix}.zip"
            used_names.add(candidate)
            atomic_copy(source, share_dir / candidate, mode=_STAGED_FILE_MODE)
            copied = share_dir / candidate
            staged.append(
                SharedArtifact(
                    key=new_artifact_key(),
                    display_name=display_name,
                    description=str(record.get("description") or ""),
                    size_bytes=copied.stat().st_size,
                    sha256=_sha256_file(copied),
                    source_chatbook_id=chatbook_id,
                    staged_name=candidate,
                )
            )
        bundle_path = share_dir / "bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_STORED) as bundle:
            for item in staged:
                bundle.write(share_dir / item.staged_name, arcname=item.staged_name)
        os.chmod(bundle_path, _STAGED_FILE_MODE)
        manifest = ArtifactShareManifest(
            share_id=share_id,
            share_name=str(share_name or "Shared artifacts"),
            created_at=datetime.now(timezone.utc).isoformat(),
            auth=auth,
            artifacts=staged,
        )
        atomic_write_json(
            share_dir / "manifest.json",
            manifest.model_dump(by_alias=True, mode="json"),
            mode=_STAGED_FILE_MODE,
            privacy_safe_log=True,
        )
        return manifest
    except Exception:
        shutil.rmtree(share_dir, ignore_errors=True)
        raise


def load_manifest(manifest_path: Path) -> ArtifactShareManifest:
    """Load and validate a share manifest; any problem raises (fail closed)."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest = ArtifactShareManifest.model_validate(data)
    if not manifest.artifacts:
        raise ArtifactShareError("Share manifest contains no artifacts.")
    return manifest


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale_shares(share_root: Path | None = None) -> list[Path]:
    """Remove share directories whose server process is gone.

    Call only at app startup, before any share is started, so a freshly
    created directory can never race this sweep.
    """
    root = Path(share_root) if share_root is not None else share_root_dir()
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        status_path = entry / "status.json"
        pid: int | None = None
        if status_path.is_file():
            try:
                pid = int(json.loads(status_path.read_text(encoding="utf-8"))["pid"])
            except (ValueError, KeyError, OSError):
                pid = None
        if pid is None or not pid_alive(pid):
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry)
    return removed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Web_Server/test_artifact_share_manifest.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Web_Server/artifact_share_manifest.py Tests/Web_Server/test_artifact_share_manifest.py
git commit -m "feat(web-server): add artifact share staging manifest and auth verifier"
```

---

### Task 3: Share server child process (aiohttp)

**Files:**
- Create: `tldw_chatbook/Web_Server/artifact_share_server.py`
- Test: `Tests/Web_Server/test_artifact_share_server.py`

**Interfaces:**
- Consumes: Task 2's `ArtifactShareServer` inputs — `load_manifest(manifest_path)`, `verify_share_auth(auth, username, password)`, `ArtifactShareManifest` / `SharedArtifact` / `ArtifactShareAuth` models; `Utils.atomic_file_ops.atomic_write_json`.
- Produces (used by Task 4):
  - `class ArtifactShareServer` with `__init__(self, manifest_path: Path)`, `build_app(self) -> web.Application` (aiohttp imported lazily), `run(self, host: str, port: int) -> str`
  - module `__main__` entry: `python -m tldw_chatbook.Web_Server.artifact_share_server <manifest.json> [--host H] [--port P]`
  - child contract: writes `status.json` (`{"url", "pid", "started_at"}`, mode `0o600`) beside the manifest at startup; prints `ARTIFACT_SHARE_READY <url>` to stdout; exits when parent dies (PPID guard) or on SIGTERM.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Web_Server/test_artifact_share_server.py`:

```python
"""Route, auth, and containment tests for the artifact share server."""

import json
import subprocess
import sys
import threading
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tldw_chatbook.Web_Server.artifact_share_manifest import (
    build_share_auth,
    stage_share,
)
from tldw_chatbook.Web_Server.artifact_share_server import ArtifactShareServer

pytestmark = [pytest.mark.unit, pytest.mark.loopback_network]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _stage(tmp_path: Path, *, auth: bool, names: tuple[str, ...] = ("Alpha", "Beta")):
    records = []
    for index, name in enumerate(names, start=1):
        bundle = tmp_path / f"src-{index}.zip"
        bundle.write_bytes(f"bytes-of-{name}".encode())
        records.append(
            {
                "id": str(index),
                "chatbook_id": index,
                "name": name,
                "description": f"desc {name} <script>alert(1)</script>",
                "file_path": str(bundle),
            }
        )
    auth_model = build_share_auth("alice", "secret-pass") if auth else None
    manifest = stage_share(
        records, share_name="Test Library 'quotes'", auth=auth_model, share_root=tmp_path / "share"
    )
    return tmp_path / "share" / manifest.share_id / "manifest.json", manifest


@contextmanager
def _served(manifest_path: Path):
    """Run the share app on an ephemeral loopback port in a background thread."""
    import asyncio

    from aiohttp import web

    server = ArtifactShareServer(manifest_path)
    app = server.build_app()
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    holder: dict[str, object] = {}

    def _run() -> None:
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app, access_log=None)

        async def _start() -> None:
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            holder["url"] = f"http://{runner.addresses[0][0]}:{runner.addresses[0][1]}"
            ready.set()
            await holder["stop_event"].wait()
            await runner.cleanup()

        holder["stop_event"] = asyncio.Event()
        loop.run_until_complete(_start())
        loop.close()

    thread = threading.Thread(target=_run, name="test-share-server", daemon=True)
    thread.start()
    assert ready.wait(timeout=10), "share server did not start"
    yield holder["url"]
    loop.call_soon_threadsafe(holder["stop_event"].set)
    thread.join(timeout=10)


def _client(auth: tuple[str, str] | None = None) -> httpx.Client:
    return httpx.Client(
        base_url="ignored",
        timeout=10,
        auth=auth,
        follow_redirects=False,
    )


def test_index_page_lists_artifacts_escaped_no_auth(tmp_path):
    manifest_path, _manifest = _stage(tmp_path, auth=False)
    with _served(manifest_path) as url, _client() as client:
        response = client.get(f"{url}/")
        assert response.status_code == 200
        assert "text/html" in response.headers["Content-Type"]
        body = response.text
        assert "Alpha" in body and "Beta" in body
        assert "<script>" not in body  # descriptions are escaped
        assert "&lt;script&gt;" in body
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert "default-src" in response.headers["Content-Security-Policy"]


def test_index_json_and_artifact_download_match_staged_bytes(tmp_path):
    manifest_path, manifest = _stage(tmp_path, auth=False)
    share_dir = manifest_path.parent
    with _served(manifest_path) as url, _client() as client:
        listing = client.get(f"{url}/index.json")
        assert listing.status_code == 200
        payload = listing.json()
        assert payload["share_name"] == "Test Library 'quotes'"
        assert {entry["name"] for entry in payload["artifacts"]} == {
            item.display_name for item in manifest.artifacts
        }
        first = manifest.artifacts[0]
        entry = next(e for e in payload["artifacts"] if e["name"] == first.display_name)
        assert entry["sha256"] == first.sha256

        download = client.get(f"{url}/artifact/{first.key}")
        assert download.status_code == 200
        assert download.content == (share_dir / first.staged_name).read_bytes()
        assert download.headers["Content-Disposition"].startswith("attachment;")
        assert "application" in download.headers["Content-Type"]
        head = client.head(f"{url}/artifact/{first.key}")
        assert head.status_code == 200

        bundle = client.get(f"{url}/bundle.zip")
        assert bundle.status_code == 200
        with zipfile.ZipFile(__import__("io").BytesIO(bundle.content)) as opened:
            assert sorted(opened.namelist()) == sorted(
                item.staged_name for item in manifest.artifacts
            )


def test_unknown_key_is_404(tmp_path):
    manifest_path, _manifest = _stage(tmp_path, auth=False)
    with _served(manifest_path) as url, _client() as client:
        assert client.get(f"{url}/artifact/does-not-exist").status_code == 404


def test_traversal_staged_name_is_never_served(tmp_path, monkeypatch):
    manifest_path, manifest = _stage(tmp_path, auth=False)
    # Tamper with the manifest the way a corrupted file would look: a
    # staged_name that escapes the staging directory must not resolve.
    data = json.loads(manifest_path.read_text())
    data["artifacts"][0]["staged_name"] = "../escape.zip"
    manifest_path.write_text(json.dumps(data))
    (tmp_path / "escape.zip").write_bytes(b"escaped")
    with _served(manifest_path) as url, _client() as client:
        key = json.loads(manifest_path.read_text())["artifacts"][0]["key"]
        assert client.get(f"{url}/artifact/{key}").status_code == 404


def test_auth_challenges_and_admits(tmp_path):
    manifest_path, _manifest = _stage(tmp_path, auth=True)
    with _served(manifest_path) as url:
        with _client() as client:
            denied = client.get(f"{url}/")
            assert denied.status_code == 401
            assert "WWW-Authenticate" in denied.headers
            assert client.get(f"{url}/index.json").status_code == 401
        with _client(auth=("alice", "secret-pass")) as client:
            assert client.get(f"{url}/").status_code == 200
        with _client(auth=("alice", "wrong")) as client:
            assert client.get(f"{url}/").status_code == 401


def test_auth_lockout_after_ten_failures(tmp_path):
    manifest_path, _manifest = _stage(tmp_path, auth=True)
    with _served(manifest_path) as url, _client(auth=("alice", "wrong")) as client:
        for _ in range(10):
            assert client.get(f"{url}/").status_code == 401
        assert client.get(f"{url}/").status_code == 429
        with _client(auth=("alice", "secret-pass")) as good:
            assert good.get(f"{url}/").status_code == 429  # same IP stays locked out


def test_subprocess_ready_line_and_download(tmp_path):
    manifest_path, _manifest = _stage(tmp_path, auth=False)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tldw_chatbook.Web_Server.artifact_share_server",
            str(manifest_path),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Read stdout lines until the ready marker (bounded wait).
        url = None
        for _ in range(100):
            line = process.stdout.readline()
            if not line:
                break
            if line.startswith("ARTIFACT_SHARE_READY "):
                url = line.split(" ", 1)[1].strip()
                break
        assert url, "child never reported ARTIFACT_SHARE_READY"
        status = json.loads((manifest_path.parent / "status.json").read_text())
        assert status["url"] == url
        assert status["pid"] == process.pid
        with _client() as client:
            assert client.get(f"{url}/").status_code == 200
    finally:
        process.terminate()
        process.wait(timeout=10)
```

Note: `test_subprocess_ready_line_and_download` needs the `integration` flavor; add `@pytest.mark.integration` directly above it (keep the module-level `pytestmark` for the rest).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Web_Server/test_artifact_share_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Web_Server.artifact_share_server'`

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Web_Server/artifact_share_server.py`:

```python
# artifact_share_server.py
"""Child-process HTTP server exposing one staged artifact share.

Run as ``python -m tldw_chatbook.Web_Server.artifact_share_server <manifest>``
by the app-side controller. Serves ONLY files inside the staging directory
(the manifest's parent). aiohttp is imported lazily so this module (and its
tests) import cleanly without the ``[web]`` extra.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import html
import os
import signal
import threading
import time
import urllib.parse
from pathlib import Path

from loguru import logger

from ..Utils.atomic_file_ops import atomic_write_json
from .artifact_share_manifest import (
    ArtifactShareAuth,
    ArtifactShareManifest,
    ArtifactShareServerLoadError,  # re-exported alias, see below
    load_manifest,
    verify_share_auth,
)
from .artifact_share_manifest import ArtifactShareError  # noqa: F811 (re-export)

_AUTH_FAILURE_THRESHOLD = 10
_AUTH_LOCKOUT_SECONDS = 30.0
_VERIFIED_CACHE_LIMIT = 64

_BASE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
    "Cache-Control": "no-store",
}

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{share_name}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 46rem; padding: 0 1rem; }}
h1 {{ font-size: 1.4rem; }}
article {{ border: 1px solid #ccc; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
.desc {{ color: #444; }}
.meta {{ color: #777; font-size: .9rem; }}
.btn {{ display: inline-block; margin-top: .5rem; padding: .4rem .9rem; border-radius: 6px;
       background: #0b63a3; color: #fff; text-decoration: none; }}
footer {{ margin-top: 2rem; color: #777; font-size: .85rem; }}
</style>
</head>
<body>
<h1>{share_name}</h1>
<p>{count} artifact(s) available.</p>
{auth_note}
{rows}
<p><a class="btn" href="/bundle.zip" download>Download all</a></p>
<footer>Import these bundles into tldw_chatbook via Chatbooks &rarr; Import.</footer>
</body>
</html>
"""


class ArtifactShareServer:
    """Serve one staged share directory described by a manifest."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self.manifest: ArtifactShareManifest = load_manifest(self.manifest_path)
        self.staging_dir = self.manifest_path.parent.resolve()
        self._verified: set[tuple[str, str]] = set()
        self._failures: dict[str, tuple[int, float]] = {}

    # -- routes -------------------------------------------------------------

    def build_app(self):
        from aiohttp import web

        app = web.Application(client_max_size=1)
        app["share_server"] = self
        app.middlewares.append(self._headers_middleware)
        if self.manifest.auth is not None:
            app.middlewares.append(self._auth_middleware)
        app.router.add_get("/", self.handle_index)
        app.router.add_get("/index.json", self.handle_index_json)
        app.router.add_get("/artifact/{key}", self.handle_artifact)
        app.router.add_get("/bundle.zip", self.handle_bundle)
        return app

    async def _headers_middleware(self, request, handler):
        from aiohttp import web

        try:
            response = await handler(request)
        except web.HTTPException as exc:
            for name, value in _BASE_HEADERS.items():
                exc.headers[name] = value
            raise
        for name, value in _BASE_HEADERS.items():
            response.headers[name] = value
        return response

    async def _auth_middleware(self, request, handler):
        from aiohttp import web

        auth = self.manifest.auth
        assert auth is not None  # middleware only installed when auth is set
        peer = request.remote or "unknown"
        now = time.monotonic()
        failures, locked_until = self._failures.get(peer, (0, 0.0))
        if locked_until > now:
            raise web.HTTPTooManyRequests(text="Too many failed attempts; retry shortly.")
        username, password = _parse_basic_auth(request.headers.get("Authorization", ""))
        if username is not None and (username, password) in self._verified:
            return await handler(request)
        if username is not None and verify_share_auth(auth, username, password):
            if len(self._verified) >= _VERIFIED_CACHE_LIMIT:
                self._verified.clear()
            self._verified.add((username, password))
            self._failures.pop(peer, None)
            return await handler(request)
        count = failures + 1
        locked_until = now + _AUTH_LOCKOUT_SECONDS if count >= _AUTH_FAILURE_THRESHOLD else 0.0
        self._failures[peer] = (count, locked_until)
        logger.warning(f"Artifact share auth failure from {peer} (attempt {count})")
        raise web.HTTPUnauthorized(
            headers={"WWW-Authenticate": 'Basic realm="tldw chatbook artifact share"'}
        )

    async def handle_index(self, request):
        from aiohttp import web

        return web.Response(text=self._render_index(), content_type="text/html")

    async def handle_index_json(self, request):
        import json as _json

        from aiohttp import web

        payload = {
            "share_name": self.manifest.share_name,
            "created_at": self.manifest.created_at,
            "artifacts": [
                {
                    "key": item.key,
                    "name": item.display_name,
                    "description": item.description,
                    "kind": item.kind,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in self.manifest.artifacts
            ],
        }
        return web.Response(
            text=_json.dumps(payload, ensure_ascii=False),
            content_type="application/json",
        )

    async def handle_artifact(self, request):
        from aiohttp import web

        key = request.match_info["key"]
        try:
            path, display_name = self._resolve_staged(key)
        except _UnknownKeyError:
            raise web.HTTPNotFound(text="No such artifact.")
        except _GoneKeyError:
            raise web.HTTPGone(text="This artifact is no longer available.")
        return self._file_response(path, display_name)

    async def handle_bundle(self, request):
        bundle = self.staging_dir / "bundle.zip"
        if not bundle.is_file():
            from aiohttp import web

            raise web.HTTPGone(text="Bundle is no longer available.")
        return self._file_response(bundle, f"{self.manifest.share_name}-bundle.zip")

    def _file_response(self, path: Path, display_name: str):
        from aiohttp import web

        return web.FileResponse(path, headers={"Content-Disposition": _content_disposition(display_name)})

    # -- helpers ------------------------------------------------------------

    def _resolve_staged(self, key: str) -> tuple[Path, str]:
        for item in self.manifest.artifacts:
            if not hmac.compare_digest(item.key, key):
                continue
            staged = item.staged_name
            if Path(staged).name != staged:  # separators/traversal never resolve
                break
            candidate = (self.staging_dir / staged).resolve()
            try:
                candidate.relative_to(self.staging_dir)
            except ValueError:
                break
            if candidate.is_file():
                return candidate, item.display_name
            raise _GoneKeyError(key)
        raise _UnknownKeyError(key)

    def _render_index(self) -> str:
        rows = []
        for item in self.manifest.artifacts:
            name = html.escape(item.display_name, quote=True)
            description = html.escape(item.description or "", quote=True)
            size = (
                f"{item.size_bytes / (1024 * 1024):.1f} MB"
                if item.size_bytes >= 1024 * 1024
                else f"{max(item.size_bytes, 0) / 1024:.1f} KB"
            )
            rows.append(
                f"<article><h3>{name}</h3>"
                f"<p class=\"desc\">{description}</p>"
                f"<p class=\"meta\">{html.escape(item.kind)} &middot; {size}</p>"
                f"<a class=\"btn\" href=\"/artifact/{item.key}\" download>Download</a></article>"
            )
        auth_note = (
            "<p>Protected sharing is active.</p>" if self.manifest.auth is not None else ""
        )
        return _PAGE_TEMPLATE.format(
            share_name=html.escape(self.manifest.share_name or "Shared artifacts", quote=True),
            count=len(self.manifest.artifacts),
            auth_note=auth_note,
            rows="\n".join(rows),
        )

    # -- lifecycle ----------------------------------------------------------

    def run(self, host: str, port: int) -> str:
        """Serve until SIGTERM or orphaned; return the bound URL (after start)."""
        from aiohttp import web

        parent_pid = os.getppid()

        async def _serve() -> str:
            loop = asyncio.get_running_loop()
            stop = asyncio.Event()
            runner = web.AppRunner(self.build_app(), access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            bound_host, bound_port = runner.addresses[0][:2]
            url = f"http://{_format_host(bound_host)}:{bound_port}"

            def _watch_parent() -> None:
                while True:
                    if os.getppid() != parent_pid:
                        loop.call_soon_threadsafe(stop.set)
                        return
                    time.sleep(2.0)

            threading.Thread(target=_watch_parent, name="share-ppid-watch", daemon=True).start()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, stop.set)
                except NotImplementedError:  # pragma: no cover - windows
                    pass
            atomic_write_json(
                self.manifest_path.parent / "status.json",
                {
                    "url": url,
                    "pid": os.getpid(),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                mode=0o600,
                privacy_safe_log=True,
            )
            print(f"ARTIFACT_SHARE_READY {url}", flush=True)
            logger.info(
                f"Artifact share serving {len(self.manifest.artifacts)} artifact(s) at {url}"
            )
            await stop.wait()
            await runner.cleanup()
            return url

        return asyncio.run(_serve())


class _UnknownKeyError(Exception):
    pass


class _GoneKeyError(Exception):
    pass


def _parse_basic_auth(header: str) -> tuple[str | None, str | None]:
    if not header.startswith("Basic "):
        return None, None
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None, None
    if ":" not in decoded:
        return None, None
    username, _, password = decoded.partition(":")
    return username, password


def _ascii_fallback(name: str) -> str:
    cleaned = name.encode("ascii", "replace").decode("ascii").replace('"', "'")
    return cleaned or "artifact"


def _content_disposition(filename: str) -> str:
    quoted = urllib.parse.quote(filename, safe="")
    return f"attachment; filename=\"{_ascii_fallback(filename)}\"; filename*=UTF-8''{quoted}"


def _format_host(host: str) -> str:
    if ":" in host:  # IPv6 literal needs brackets in URLs
        return f"[{host}]"
    return host


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="artifact-share-server",
        description="Serve one staged tldw_chatbook artifact share.",
    )
    parser.add_argument("manifest", type=Path, help="Path to the share manifest.json")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="Bind port (0 = ephemeral)")
    args = parser.parse_args(argv)
    server = ArtifactShareServer(args.manifest)
    server.run(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Correction before committing:** the import block at the top references `ArtifactShareServerLoadError`, which does not exist. Remove that name from the import (keep only `load_manifest`, `verify_share_auth`, `ArtifactShareManifest`, `ArtifactShareAuth`, `ArtifactShareError`). Also keep the child's log volume startup-only: `access_log=None` above already suppresses per-request logs, and the merged stdout pipe held by the controller is small — do not add per-request logging.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Web_Server/test_artifact_share_server.py -v`
Expected: PASS (8 tests, including the subprocess smoke test)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Web_Server/artifact_share_server.py Tests/Web_Server/test_artifact_share_server.py
git commit -m "feat(web-server): add artifact share child server with basic auth"
```

---

### Task 4: App-side share controller

**Files:**
- Create: `tldw_chatbook/Web_Server/artifact_share.py`
- Test: `Tests/Web_Server/test_artifact_share_controller.py`

**Interfaces:**
- Consumes: Task 2 (`stage_share`, `build_share_auth`, `share_root_dir`, `sweep_stale_shares`, `ArtifactShareError`), Task 3 child contract (`status.json` with `url`/`pid`; child exit ≠ 0 on bind failure), `Web_Server.__init__.is_web_server_available()` (defined at `Web_Server/__init__.py`).
- Produces (used by Tasks 5–6):
  - `@dataclass(frozen=True) class ShareStatus` with fields `share_name: str`, `urls: tuple[str, ...]`, `artifact_count: int`, `share_dir: Path`
  - `class ArtifactShareController` with `__init__(self, *, status_callback: Callable[[ShareStatus | None], None] | None = None)`, `status -> ShareStatus | None` (property), `startup_sweep(self) -> list[Path]`, `start_share(self, *, records: list[dict], share_name: str, username: str | None = None, password: str | None = None, bind: str = "127.0.0.1", port: int = 0) -> ShareStatus`, `stop_share(self) -> None`
  - `def compute_display_urls(bind: str, port: int) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Web_Server/test_artifact_share_controller.py`:

```python
"""Controller lifecycle tests for artifact share sessions."""

import json
from pathlib import Path

import pytest

from tldw_chatbook.Web_Server.artifact_share import (
    ArtifactShareController,
    ShareStatus,
    compute_display_urls,
)
from tldw_chatbook.Web_Server.artifact_share_manifest import (
    ArtifactShareError,
    share_root_dir,
)

pytestmark = [pytest.mark.unit, pytest.mark.loopback_network]


def _record(tmp_path: Path, cid: int = 1) -> dict:
    bundle = tmp_path / f"src-{cid}.zip"
    bundle.write_bytes(f"bundle-{cid}".encode())
    return {
        "id": str(cid),
        "chatbook_id": cid,
        "name": f"Artifact {cid}",
        "description": "",
        "file_path": str(bundle),
    }


@pytest.fixture
def isolated_share_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Web_Server.artifact_share.share_root_dir",
        lambda: tmp_path / "share",
    )
    return tmp_path / "share"


def test_compute_display_urls_loopback_and_lan():
    assert compute_display_urls("127.0.0.1", 8000) == ["http://127.0.0.1:8000"]
    lan = compute_display_urls("0.0.0.0", 8000)
    assert lan[0] == "http://127.0.0.1:8000"
    import re

    assert re.fullmatch(r"http://\d{1,3}(\.\d{1,3}){3}:8000", lan[1])


def test_start_and_stop_share_lifecycle(tmp_path, isolated_share_root):
    controller = ArtifactShareController()
    statuses = []
    controller._status_callback = statuses.append

    status = controller.start_share(
        records=[_record(tmp_path, 1), _record(tmp_path, 2)],
        share_name="Field kit",
    )

    assert isinstance(status, ShareStatus)
    assert status.artifact_count == 2
    assert status.urls and status.urls[0].startswith("http://127.0.0.1:")
    share_dir = status.share_dir
    assert (share_dir / "manifest.json").is_file()
    assert controller.status == status

    controller.stop_share()

    assert controller.status is None
    assert statuses[-1] is None
    assert not share_dir.exists()  # staging cleaned
    import httpx

    with pytest.raises(httpx.ConnectError):
        httpx.get(status.urls[0], timeout=2)


def test_single_active_share_second_start_replaces_first(tmp_path, isolated_share_root):
    controller = ArtifactShareController()
    first = controller.start_share(records=[_record(tmp_path, 1)], share_name="one")
    second = controller.start_share(records=[_record(tmp_path, 2)], share_name="two")
    assert not first.share_dir.exists()
    assert second.share_dir.is_file_bundle_dir if hasattr(second, "is_file_bundle_dir") else True
    assert controller.status == second
    controller.stop_share()


def test_start_share_without_web_deps_is_clean_error(tmp_path, isolated_share_root, monkeypatch):
    import tldw_chatbook.Web_Server.artifact_share as controller_module

    monkeypatch.setattr(controller_module, "is_web_server_available", lambda: False)
    controller = ArtifactShareController()
    with pytest.raises(ArtifactShareError, match=r"tldw_chatbook\[web\]"):
        controller.start_share(records=[_record(tmp_path, 1)], share_name="x")


def test_startup_sweep_removes_stale_dirs(tmp_path, isolated_share_root):
    stale = isolated_share_root / "deadbeef"
    stale.mkdir(parents=True)
    (stale / "manifest.json").write_text("{}")
    controller = ArtifactShareController()
    removed = controller.startup_sweep()
    assert removed == [stale]
    assert not stale.exists()
```

(`is_file_bundle_dir` in `test_single_active_share_second_start_replaces_first` is filler — remove that line; the meaningful assertions are `not first.share_dir.exists()` and `controller.status == second`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Web_Server/test_artifact_share_controller.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Web_Server.artifact_share'`

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Web_Server/artifact_share.py`:

```python
# artifact_share.py
"""App-side orchestration for artifact share sessions (no Textual imports)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from loguru import logger

from . import is_web_server_available
from .artifact_share_manifest import (
    ArtifactShareError,
    build_share_auth,
    share_root_dir,
    stage_share,
    sweep_stale_shares,
)

_CHILD_READY_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class ShareStatus:
    share_name: str
    urls: tuple[str, ...]
    artifact_count: int
    share_dir: Path


class ArtifactShareController:
    """Owns at most one running share: staging, child process, teardown."""

    def __init__(
        self, *, status_callback: Callable[[ShareStatus | None], None] | None = None
    ) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._share_dir: Path | None = None
        self._status: ShareStatus | None = None
        self._status_callback = status_callback

    @property
    def status(self) -> ShareStatus | None:
        with self._lock:
            return self._status

    def startup_sweep(self) -> list[Path]:
        removed = sweep_stale_shares()
        if removed:
            logger.info(f"Artifact share sweep removed {len(removed)} stale share dir(s)")
        return removed

    def start_share(
        self,
        *,
        records: list[dict[str, Any]],
        share_name: str,
        username: str | None = None,
        password: str | None = None,
        bind: str = "127.0.0.1",
        port: int = 0,
    ) -> ShareStatus:
        if not is_web_server_available():
            raise ArtifactShareError(
                "Web sharing requires extra packages: pip install tldw_chatbook[web]"
            )
        if not records:
            raise ArtifactShareError("No artifacts selected to share.")
        with self._lock:
            self.stop_share()  # single active share; starting a new one stops the old
            auth = (
                build_share_auth(username, password)
                if username and password
                else None
            )
            manifest = stage_share(
                records, share_name=share_name, auth=auth, share_root=share_root_dir()
            )
            self._share_dir = share_root_dir() / manifest.share_id
            manifest_path = self._share_dir / "manifest.json"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tldw_chatbook.Web_Server.artifact_share_server",
                    str(manifest_path),
                    "--host",
                    bind,
                    "--port",
                    str(port),
                ],
                start_new_session=True,
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self._process = process
            status_path = self._share_dir / "status.json"
            deadline = time.monotonic() + _CHILD_READY_TIMEOUT_SECONDS
            url: str | None = None
            while time.monotonic() < deadline:
                if status_path.is_file():
                    try:
                        url = str(json.loads(status_path.read_text(encoding="utf-8"))["url"])
                        break
                    except (ValueError, KeyError, OSError):
                        pass
                if process.poll() is not None:
                    output = ""
                    if process.stdout is not None:
                        try:
                            output = process.stdout.read().decode(errors="replace")
                        except OSError:
                            pass
                    self._cleanup_staging()
                    raise ArtifactShareError(
                        f"Artifact share server failed to start: {output[-500:]}"
                    )
                time.sleep(0.1)
            if url is None:
                self.stop_share()
                raise ArtifactShareError(
                    "Artifact share server did not report readiness in time."
                )
            bound = urlparse(url)
            self._status = ShareStatus(
                share_name=manifest.share_name,
                urls=tuple(compute_display_urls(bind, bound.port or port)),
                artifact_count=len(manifest.artifacts),
                share_dir=self._share_dir,
            )
            self._emit_status()
            return self._status

    def stop_share(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            if process is not None:
                self._terminate_child(process)
                if process.stdout is not None:
                    try:
                        process.stdout.close()
                    except OSError:
                        pass
            self._cleanup_staging()
            if self._status is not None:
                self._status = None
                self._emit_status()

    def _terminate_child(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - pathological
            logger.error(f"Artifact share child {process.pid} refused to die")

    def _cleanup_staging(self) -> None:
        if self._share_dir is not None:
            shutil.rmtree(self._share_dir, ignore_errors=True)
            self._share_dir = None

    def _emit_status(self) -> None:
        callback = self._status_callback
        if callback is None:
            return
        try:
            callback(self._status)
        except Exception:  # noqa: BLE001 - UI callback must not break sharing
            logger.exception("Artifact share status callback failed")


def compute_display_urls(bind: str, port: int) -> list[str]:
    """Human-usable URLs for the bound port: loopback always, LAN route when wide."""
    urls = [f"http://127.0.0.1:{port}"]
    if bind not in ("127.0.0.1", "localhost", "::1"):
        lan_ip = _primary_route_ip()
        if lan_ip:
            urls.append(f"http://{lan_ip}:{port}")
    return urls


def _primary_route_ip() -> str | None:
    # UDP connect chooses the outbound interface without sending a packet.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            candidate = probe.getsockname()[0]
        if candidate and not candidate.startswith("127."):
            return candidate
    except OSError:
        pass
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Web_Server/test_artifact_share_controller.py -v`
Expected: PASS (5 tests). These spawn the real child from Task 3 — if `test_start_and_stop_share_lifecycle` flakes on CI, first check the child's stdout pipe isn't filling (it logs only at startup).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Web_Server/artifact_share.py Tests/Web_Server/test_artifact_share_controller.py
git commit -m "feat(web-server): add artifact share controller with child supervision"
```

---

### Task 5: Share dialog (ModalScreen)

**Files:**
- Create: `tldw_chatbook/UI/Screens/artifact_share_dialog.py`
- Test: `Tests/UI/test_artifact_share_dialog.py`

**Interfaces:**
- Consumes: Task 4's `ShareStatus` (only for the "share already running" notice); Textual `ModalScreen[dict | None]` idiom of `UI/Chunking_Lab_Modules/dialogs.py::LabDialog` (BUNDLED_CSS + escape-cancel + dismiss result); `SelectionList`/`Selection` widgets (precedent: `Widgets/Library/library_skills_canvas.py`).
- Produces (used by Task 6): `class ArtifactShareDialog(ModalScreen[dict | None])` with `__init__(self, records: list[dict], *, active_share_notice: str | None = None)`; dismisses with `None` (cancel/escape) or a dict:
  `{"selected_records": list[dict], "share_name": str, "username": str, "password": str, "bind": str, "port": int}` where `bind` is `"127.0.0.1"` or `"0.0.0.0"` and `port` is `0` (auto) or a positive int.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_artifact_share_dialog.py`:

```python
"""Dialog behavior tests for artifact share."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.Screens.artifact_share_dialog import ArtifactShareDialog

pytestmark = pytest.mark.ui

_CONFIRM_PHRASE = "share"


class _DialogHost(App[None]):
    def __init__(self, dialog: ArtifactShareDialog) -> None:
        super().__init__()
        self._dialog = dialog
        self.result: object = "unset"

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(self._dialog, self._accept)

    def _accept(self, result: object) -> None:
        self.result = result


def _records() -> list[dict]:
    return [
        {"id": "1", "chatbook_id": 1, "name": "With Bundle", "description": "d1", "file_path": "/tmp/a.zip"},
        {"id": "2", "chatbook_id": 2, "name": "No Bundle", "description": "d2", "file_path": None},
    ]


async def test_cancel_dismisses_with_none():
    dialog = ArtifactShareDialog(_records())
    app = _DialogHost(dialog)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is None


async def test_start_requires_selection():
    dialog = ArtifactShareDialog(_records())
    app = _DialogHost(dialog)
    async with app.run_test() as pilot:
        await pilot.click("#share-start")
        await pilot.pause()
        assert app.result == "unset"  # still open
        status = dialog.query_one("#share-dialog-status", Static)
        assert "select" in status.renderable.lower() if status.renderable else True


async def test_valid_submission_returns_options():
    dialog = ArtifactShareDialog(_records())
    app = _DialogHost(dialog)
    async with app.run_test() as pilot:
        options = dialog.query_one("#share-artifact-list")
        # select the first (enabled) option; the second is disabled (no bundle)
        options.select(0)
        await pilot.pause()
        share_name = dialog.query_one("#share-name", )
        share_name.value = "Field kit"
        await pilot.click("#share-auth-toggle")
        await pilot.pause()
        dialog.query_one("#share-username").value = "alice"
        dialog.query_one("#share-password").value = "secret-pass"
        await pilot.click("#share-start")
        await pilot.pause()
    assert isinstance(app.result, dict)
    assert app.result["share_name"] == "Field kit"
    assert [r["name"] for r in app.result["selected_records"]] == ["With Bundle"]
    assert app.result["username"] == "alice"
    assert app.result["bind"] == "127.0.0.1"
    assert app.result["port"] == 0


async def test_lan_without_password_requires_typed_confirmation():
    dialog = ArtifactShareDialog(_records())
    app = _DialogHost(dialog)
    async with app.run_test() as pilot:
        options = dialog.query_one("#share-artifact-list")
        options.select(0)
        await pilot.pause()
        await pilot.click("#share-bind-lan")
        await pilot.pause()
        await pilot.click("#share-start")
        await pilot.pause()
        assert app.result == "unset"  # blocked: confirmation required
        confirm = dialog.query_one("#share-confirm")
        assert confirm.display
        confirm.value = _CONFIRM_PHRASE
        await pilot.click("#share-start")
        await pilot.pause()
    assert isinstance(app.result, dict)
    assert app.result["bind"] == "0.0.0.0"
    assert app.result["password"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/UI/test_artifact_share_dialog.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` for `artifact_share_dialog`

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/UI/Screens/artifact_share_dialog.py`:

```python
# artifact_share_dialog.py
"""Modal dialog for starting an artifact share session."""

from __future__ import annotations

from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    RadioButton,
    RadioSet,
    SelectionList,
    Static,
)
from textual.widgets.selection_list import Selection

_CONFIRM_PHRASE = "share"


def _option_title(record: dict[str, Any]) -> str:
    name = str(record.get("name") or "Unnamed artifact")
    if record.get("file_path"):
        return name
    return f"{name}  (no exported bundle on disk)"


class ArtifactShareDialog(ModalScreen[dict | None]):
    """Choose artifacts and share options; dismisses with an options dict or None."""

    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel", show=False)]

    BUNDLED_CSS = """
    ArtifactShareDialog { align: center middle; background: $background 70%; }
    ArtifactShareDialog > VerticalScroll {
        width: 76; max-width: 96%; height: auto; max-height: 90%;
        background: $surface; border: solid $primary; padding: 1 2;
    }
    ArtifactShareDialog Input, ArtifactShareDialog RadioSet, ArtifactShareDialog SelectionList {
        margin-bottom: 1;
    }
    ArtifactShareDialog #share-dialog-status { color: $warning; }
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        active_share_notice: str | None = None,
    ) -> None:
        super().__init__()
        self._records = list(records)
        self._active_share_notice = active_share_notice

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("Share artifacts", markup=False)
            yield Static(
                "Recipients browse a temporary web page and download only the "
                "selected bundles. Traffic is plain HTTP: a password is an access "
                "gate, not encryption. Stop sharing to revoke access.",
                markup=False,
            )
            if self._active_share_notice:
                yield Static(self._active_share_notice, markup=False, id="share-active-note")
            yield Static("Share name (page title)", markup=False)
            yield Input(placeholder="Shared artifacts", id="share-name")
            yield Static("Artifacts", markup=False)
            yield SelectionList(
                *(
                    Selection(
                        _option_title(record),
                        str(record.get("id")),
                        disabled=not bool(record.get("file_path")),
                    )
                    for record in self._records
                ),
                id="share-artifact-list",
            )
            yield Checkbox("Require a password (single shared login)", False, id="share-auth-toggle")
            yield Input(placeholder="Username", id="share-username")
            yield Input(placeholder="Password", password=True, id="share-password")
            yield Static("Who can reach it", markup=False)
            with RadioSet(id="share-bind"):
                yield RadioButton("This computer only (localhost)", value=True, id="share-bind-loopback")
                yield RadioButton("Local network (all interfaces)", id="share-bind-lan")
            yield Static("Port (blank = pick automatically)", markup=False)
            yield Input(placeholder="auto", id="share-port")
            yield Static(
                "Sharing without a password on the local network exposes these "
                f"artifacts to everyone on that network. Type '{_CONFIRM_PHRASE}' "
                "to confirm.",
                markup=False,
                id="share-confirm-label",
            )
            yield Input(placeholder=_CONFIRM_PHRASE, id="share-confirm")
            yield Static("", markup=False, id="share-dialog-status")
            with Horizontal():
                yield Button("Start sharing", id="share-start", variant="primary")
                yield Button("Cancel", id="share-cancel")

    def on_mount(self) -> None:
        self._sync_auth_visibility(False)
        self._sync_confirm_visibility()

    def _selected_records(self) -> list[dict[str, Any]]:
        selected_ids = {
            str(value) for value in self.query_one("#share-artifact-list", SelectionList).selected
        }
        return [record for record in self._records if str(record.get("id")) in selected_ids]

    def _bind_choice(self) -> str:
        return (
            "0.0.0.0"
            if self.query_one("#share-bind-lan", RadioButton).value
            else "127.0.0.1"
        )

    def _port_value(self) -> int:
        raw = self.query_one("#share-port", Input).value.strip()
        if not raw:
            return 0
        try:
            port = int(raw)
        except ValueError:
            return -1
        return port if 1 <= port <= 65535 else -1

    def _auth_enabled(self) -> bool:
        return self.query_one("#share-auth-toggle", Checkbox).value

    def _sync_auth_visibility(self, enabled: bool) -> None:
        self.query_one("#share-username", Input).display = enabled
        self.query_one("#share-password", Input).display = enabled

    def _sync_confirm_visibility(self) -> None:
        needs_confirm = self._bind_choice() == "0.0.0.0" and not self._auth_enabled()
        self.query_one("#share-confirm-label", Static).display = needs_confirm
        self.query_one("#share-confirm", Input).display = needs_confirm

    def _status(self, message: str) -> None:
        self.query_one("#share-dialog-status", Static).update(message)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "share-auth-toggle":
            self._sync_auth_visibility(event.value)
            self._sync_confirm_visibility()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self._sync_confirm_visibility()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "share-cancel":
            self.dismiss(None)
            return
        if event.button.id != "share-start":
            return
        selected = self._selected_records()
        if not selected:
            self._status("Select at least one artifact to share.")
            return
        username = self.query_one("#share-username", Input).value.strip()
        password = self.query_one("#share-password", Input).value
        if self._auth_enabled() and (not username or not password):
            self._status("Enter both a username and a password, or disable the password.")
            return
        port = self._port_value()
        if port < 0:
            self._status("Port must be a number between 1 and 65535, or blank for automatic.")
            return
        if self._bind_choice() == "0.0.0.0" and not self._auth_enabled():
            if self.query_one("#share-confirm", Input).value.strip() != _CONFIRM_PHRASE:
                self._status(
                    f"Type '{_CONFIRM_PHRASE}' to share without a password on the local network."
                )
                return
        self.dismiss(
            {
                "selected_records": selected,
                "share_name": self.query_one("#share-name", Input).value.strip()
                or "Shared artifacts",
                "username": username if self._auth_enabled() else "",
                "password": password if self._auth_enabled() else "",
                "bind": self._bind_choice(),
                "port": port,
            }
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/UI/test_artifact_share_dialog.py -v`
Expected: PASS (4 tests). If `pilot.click("#share-bind-lan")` misses (radio hit-box), replace with `dialog.query_one("#share-bind-lan", RadioButton).toggle()` + `await pilot.pause()` — same assertion.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/artifact_share_dialog.py Tests/UI/test_artifact_share_dialog.py
git commit -m "feat(ui): add artifact share dialog with auth and bind guardrails"
```

---

### Task 6: Artifacts screen wiring and app lifecycle

**Files:**
- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py` (compose block near lines 847–977; handlers near lines 1102–1135; `on_screen_resume` at line 137)
- Modify: `tldw_chatbook/app.py` (`_wire_prompt_chatbook_services` at line 9957, after the `local_chatbook_service` assignment at 9964–9966; `on_unmount` at line 17596, near the `_disconnect_local_mcp_client()` call at ~17739)
- Test: `Tests/UI/test_artifacts_screen_share.py`

**Interfaces:**
- Consumes: Task 4's `ArtifactShareController` / `ShareStatus` / `ArtifactShareError`; Task 5's `ArtifactShareDialog` result dict; `Web_Server.is_web_server_available()`; existing screen idioms — `self.app_instance` service access, `_notify(message, severity)` (artifacts_screen.py:291), `@work(exclusive=True, thread=True, group=...)` workers, `self.app.call_from_thread(...)` for UI marshaling, `asyncio.run(service.list_chatbooks(limit=1000))` inside the worker thread.
- Produces: app attribute `self.artifact_share_controller: ArtifactShareController` on `TldwCli`; screen widgets `#artifacts-share` (button), `#artifacts-share-status` (banner Static), `#artifacts-share-stop` (button); binding `s` → `action_share_artifacts`; app method `_shutdown_artifact_share(self) -> None` (idempotent, called from `on_unmount`).

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_artifacts_screen_share.py`, modeled on `Tests/UI/test_artifacts_screen_reports.py`'s harness (`_build_test_app` + `DestinationHarness`):

```python
"""Artifacts screen share wiring tests."""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.UI.Screens.artifact_share_dialog import ArtifactShareDialog
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen
from tldw_chatbook.Web_Server.artifact_share import ArtifactShareController, ShareStatus

pytestmark = pytest.mark.ui


@asynccontextmanager
async def _open_artifacts(app, *, size=(160, 50)):
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)
        yield screen, pilot


@pytest.fixture
def stub_controller():
    class _Stub:
        def __init__(self):
            self.status = None
            self.stopped = 0

        def stop_share(self):
            self.stopped += 1

    return _Stub()


async def test_share_button_and_binding_open_dialog(tmp_path):
    app = await _build_test_app()
    async with _open_artifacts(app) as (screen, pilot):
        assert screen.query_one("#artifacts-share")
        await pilot.click("#artifacts-share")
        await pilot.pause()
        assert any(isinstance(s, ArtifactShareDialog) for s in app.screen_stack)


async def test_banner_reflects_active_share(tmp_path, stub_controller):
    app = await _build_test_app()
    async with _open_artifacts(app) as (screen, pilot):
        app.artifact_share_controller = stub_controller
        stub_controller.status = ShareStatus(
            share_name="Kit",
            urls=("http://127.0.0.1:8123",),
            artifact_count=3,
            share_dir=Path(tmp_path),
        )
        screen._render_share_banner()
        banner = screen.query_one("#artifacts-share-status")
        assert "http://127.0.0.1:8123" in str(banner.renderable)
        assert screen.query_one("#artifacts-share-stop").display
        stub_controller.status = None
        screen._render_share_banner()
        assert not screen.query_one("#artifacts-share-stop").display


async def test_stop_button_calls_controller(tmp_path, stub_controller):
    app = await _build_test_app()
    async with _open_artifacts(app) as (screen, pilot):
        app.artifact_share_controller = stub_controller
        stub_controller.status = ShareStatus(
            share_name="Kit",
            urls=("http://127.0.0.1:8123",),
            artifact_count=1,
            share_dir=Path(tmp_path),
        )
        screen._render_share_banner()
        await pilot.click("#artifacts-share-stop")
        await pilot.pause(0.2)
        assert stub_controller.stopped == 1


async def test_app_shutdown_stops_share(tmp_path, stub_controller):
    app = await _build_test_app()
    app.artifact_share_controller = stub_controller
    app._shutdown_artifact_share()
    assert stub_controller.stopped == 1
    app._shutdown_artifact_share()  # idempotent
    assert stub_controller.stopped == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/UI/test_artifacts_screen_share.py -v`
Expected: FAIL — `NoSuchWidget` for `#artifacts-share`, and `AttributeError` for `_shutdown_artifact_share`.

- [ ] **Step 3: Wire the screen**

In `tldw_chatbook/UI/Screens/artifacts_screen.py`:

3a. Add imports at the top (following the file's existing grouped local-import style):

```python
from ...Web_Server.artifact_share import ArtifactShareError, ShareStatus
from ..Navigation.base_app_screen import BaseAppScreen  # already imported
from .artifact_share_dialog import ArtifactShareDialog
```

Also add `asyncio` to the stdlib import block (the file already imports it — verify at line 7; if present, skip).

3b. Add the binding on the class (the screen has no `BINDINGS` today — create the attribute right under the class docstring at line 106):

```python
    BINDINGS = [Binding("s", "share_artifacts", "Share")]
```

(Add `from textual.binding import Binding` to the textual imports; `on`, `work` are already imported.)

3c. In `compose_content`, inside the list pane `Vertical(id="artifacts-list-pane")` right after the `#artifacts-open-console` button block (~line 957–961), add:

```python
            with Horizontal(id="artifacts-share-row", classes="destination-row"):
                yield Button(
                    "Share artifacts",
                    id="artifacts-share",
                    classes="destination-action",
                )
            yield Static("", id="artifacts-share-status", classes="destination-purpose")
            yield Button("Stop sharing", id="artifacts-share-stop", styles="display: none")
```

(`Horizontal` is already imported. Match the exact idiom of neighboring buttons — copy the classes/kwargs the adjacent `#artifacts-open-console` button uses.)

3d. Add the action and handlers (near the other `@on(Button.Pressed)` handlers, after `open_console` at line 1110):

```python
    def action_share_artifacts(self) -> None:
        self._open_share_dialog()

    @on(Button.Pressed, "#artifacts-share")
    def share_artifacts(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_share_dialog()

    def _open_share_dialog(self) -> None:
        from ...Web_Server import is_web_server_available

        if not is_web_server_available():
            self._notify(
                "Web sharing needs extra packages: pip install tldw_chatbook[web]",
                "warning",
            )
            return
        self._share_dialog_worker = self._run_share_dialog_open()

    @work(exclusive=True, thread=True, group="artifacts-share-dialog")
    def _run_share_dialog_open(self) -> None:
        service = getattr(self.app_instance, "local_chatbook_service", None)
        if service is None:
            self.app.call_from_thread(self._notify, CHATBOOK_SERVICE_ERROR_COPY)
            return
        try:
            records = asyncio.run(service.list_chatbooks(limit=1000))
        except Exception as exc:
            logger.warning(f"Artifact share: listing chatbooks failed: {exc}")
            self.app.call_from_thread(self._notify, CHATBOOK_SERVICE_ERROR_COPY)
            return
        controller = getattr(self.app_instance, "artifact_share_controller", None)
        notice = None
        if controller is not None and controller.status is not None:
            notice = (
                "A share is already running; starting a new one will stop it."
            )
        dialog = ArtifactShareDialog(records, active_share_notice=notice)
        self.app.call_from_thread(self.app.push_screen, dialog, self._on_share_dialog_result)

    def _on_share_dialog_result(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        self._start_share(result)

    @work(exclusive=True, thread=True, group="artifacts-share-start")
    def _start_share(self, options: dict) -> None:
        controller = getattr(self.app_instance, "artifact_share_controller", None)
        if controller is None:
            self.app.call_from_thread(
                self._notify, "Artifact sharing is unavailable in this session.", "error"
            )
            return
        try:
            status = controller.start_share(
                records=options["selected_records"],
                share_name=options["share_name"],
                username=options.get("username") or None,
                password=options.get("password") or None,
                bind=options["bind"],
                port=options["port"],
            )
        except ArtifactShareError as exc:
            self.app.call_from_thread(self._notify, f"Sharing failed: {exc}", "error")
            self.app.call_from_thread(self._render_share_banner)
            return
        url = status.urls[-1] if status.urls else "the assigned port"
        self.app.call_from_thread(
            self._notify, f"Sharing {status.artifact_count} artifact(s) at {url}"
        )
        self.app.call_from_thread(self._render_share_banner)

    @on(Button.Pressed, "#artifacts-share-stop")
    def stop_artifact_share(self, event: Button.Pressed) -> None:
        event.stop()
        self._stop_share()

    @work(exclusive=True, thread=True, group="artifacts-share-stop")
    def _stop_share(self) -> None:
        controller = getattr(self.app_instance, "artifact_share_controller", None)
        if controller is None:
            return
        controller.stop_share()
        self.app.call_from_thread(self._notify, "Stopped sharing artifacts.")
        self.app.call_from_thread(self._render_share_banner)

    def _render_share_banner(self) -> None:
        try:
            banner = self.query_one("#artifacts-share-status", Static)
            stop_button = self.query_one("#artifacts-share-stop", Button)
        except Exception:  # screen not composed yet
            return
        controller = getattr(self.app_instance, "artifact_share_controller", None)
        status: ShareStatus | None = getattr(controller, "status", None)
        if status is None:
            banner.update("")
            stop_button.display = False
            return
        urls = "  ·  ".join(status.urls)
        banner.update(
            f"Sharing {status.artifact_count} artifact(s) as '{status.share_name}': {urls}"
        )
        stop_button.display = True
```

3e. In `on_screen_resume` (line 137) append `self._render_share_banner()` so the banner survives navigation.

3f. Footer hint registration (ADR-031 truthful 1:1). First inspect an existing caller to copy the exact tuple shape:

```bash
git grep -n "register_footer_shortcuts(" origin/dev -- tldw_chatbook/UI | head -5
```

Then, in `on_mount` (line 130), append a call using that observed shape for a single `("s", "Share")` entry, e.g. if the observed format is `(("s", "Share"),)`:

```python
        self.register_footer_shortcuts(
            source="artifacts-screen", shortcuts=(("s", "Share"),)
        )
```

Adjust the tuple to match the observed caller exactly — the repo test-enforces footer-hint truthfulness (ADR-031 rule 4).

- [ ] **Step 4: Wire the app**

In `tldw_chatbook/app.py`:

4a. In `_wire_prompt_chatbook_services` (line 9957), immediately after the `local_chatbook_service` assignment (lines 9964–9966):

```python
        from .Web_Server.artifact_share import ArtifactShareController

        self.artifact_share_controller = ArtifactShareController()
        try:
            self.artifact_share_controller.startup_sweep()
        except Exception as exc:
            logger.warning(f"Artifact share startup sweep failed: {exc}")
```

(Deferred import on purpose — matches the repo's task-285 phase-2 pattern; keeps the app import graph unchanged.)

4b. Add an idempotent shutdown method near the other small lifecycle helpers (e.g. just above `on_unmount` at line 17596):

```python
    def _shutdown_artifact_share(self) -> None:
        controller = getattr(self, "artifact_share_controller", None)
        if controller is None:
            return
        try:
            controller.stop_share()
        except Exception as exc:
            logger.warning(f"Artifact share shutdown failed: {exc}")
```

4c. In `on_unmount`'s main try block, next to `await self._disconnect_local_mcp_client()` (~line 17739), add:

```python
                self._shutdown_artifact_share()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest Tests/UI/test_artifacts_screen_share.py Tests/UI/test_artifact_share_dialog.py Tests/UI/test_artifacts_screen_reports.py -v`
Expected: PASS — including the pre-existing reports tests (regression guard for the screen changes).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/artifacts_screen.py tldw_chatbook/app.py Tests/UI/test_artifacts_screen_share.py
git commit -m "feat(ui): wire artifact share into Artifacts screen and app lifecycle"
```

---

### Task 7: User guide, spec/task closeout

**Files:**
- Modify: `Docs/User_Guide/artifacts.md`
- Modify: `Docs/superpowers/specs/2026-09-05-artifact-share-web-export-design.md` (only if implementation deviated)
- Modify: `backlog/tasks/task-31978 - Add-artifact-share-web-export-from-Artifacts-screen.md`

**Interfaces:**
- Consumes: everything above.
- Produces: updated docs; TASK-31978 moved toward Done (implementation notes recorded).

- [ ] **Step 1: Extend the user guide**

Append a "Sharing artifacts" section to `Docs/User_Guide/artifacts.md` (currently a 16-line stub). Content must cover, in the guide's existing voice: pressing `s` or the Share artifacts button; selecting bundles (records without an on-disk bundle are not shareable); the optional single shared username/password; the localhost vs local-network choice and the typed confirmation required to share wide without a password; the plain-HTTP caveat with the reverse-proxy suggestion for hostile networks; that recipients import downloads via Chatbooks → Import; and that Stop sharing (or closing the app) revokes access immediately.

- [ ] **Step 2: Record any deviations**

If implementation deviated from the spec (e.g. footer-hint tuple shape forced a change, or a test idiom differed), update the spec's relevant lines and note the deviation in TASK-31978's Implementation Notes — do not silently diverge.

- [ ] **Step 3: Close out TASK-31978**

Update the backlog task: check off acceptance criteria that the tests demonstrate, add the Implementation Plan reference (`Docs/superpowers/plans/2026-09-05-artifact-share-web-export.md`), add Implementation Notes (approach, modified/added files, deviations), and record the ADR check:

```text
ADR required: yes
ADR path: backlog/decisions/123-artifact-share-web-export.md
```

Leave the task status as dictated by the DoD checklist — only set Done when every DoD item (tests, lint, docs, self-review) is complete. Decide whether this work produced a generalizable lesson for `backlog/docs/lessons-*.md`; most tasks produce nothing there — do not invent one.

- [ ] **Step 4: Run the full targeted verification set**

```bash
python -m pytest Tests/Web_Server/test_artifact_share_manifest.py \
  Tests/Web_Server/test_artifact_share_server.py \
  Tests/Web_Server/test_artifact_share_controller.py \
  Tests/Web_Server/test_web_server_dependency_gate.py \
  Tests/UI/test_artifact_share_dialog.py \
  Tests/UI/test_artifacts_screen_share.py \
  Tests/UI/test_artifacts_screen_reports.py -v
```

Expected: all PASS. Do not run the full suite (repo rule: ask first).

- [ ] **Step 5: Commit**

```bash
git add Docs/User_Guide/artifacts.md "backlog/tasks/task-31978 - Add-artifact-share-web-export-from-Artifacts-screen.md"
git commit -m "docs: document artifact sharing and close out task notes"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** staging/manifest → Task 2; server routes/auth/headers/child contract → Task 3; lifecycle/supervision/URL display/sweep → Task 4; dialog + guardrails → Task 5; screen/app wiring + keybinding + gate → Task 6; docs/ADR/task closeout → Tasks 1 & 7. Deferred items in the spec's "Open items" section map to no tasks, by design. One intentional deviation: the spec's "app-title suffix" reminder is implemented as the screen banner + start/stop notifications instead (app title is managed dynamically elsewhere; a suffix would fight it) — Task 7 Step 2 records this in the spec.
- **Placeholder scan:** the two "correction" callouts inside Tasks 3 and 4 (remove the nonexistent import; remove the filler assertion line) are explicit pre-commit edits, not deferred work; the footer-hint step in Task 6 includes the grep command and adaptation instruction because the tuple shape is repo-defined. No TBDs remain.
- **Type consistency:** `ShareStatus(share_name, urls, artifact_count, share_dir)` used identically in Tasks 4–6; the dialog result dict keys (`selected_records`, `share_name`, `username`, `password`, `bind`, `port`) match Task 6's `_start_share` consumption; child contract (`status.json` fields, ready-line format, module path) is identical in Tasks 3 and 4; `stage_share(records, share_name=, auth=, share_root=)` signature matches across Tasks 2–4.
