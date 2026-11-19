# Served-Mode Auth Foundation (Phase 1a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End "type the long token on the phone, every restart" — persistent device sessions, a device-manager page with logout/revocation, QR device pairing, and a `tldw-serve --pair` CLI, plus the Phase 1b engine-binding spike that gates the mobile plan.

**Architecture:** `WebAuthManager` (`tldw_chatbook/Canvas/web_auth.py`) keeps its single-process admission model but gains a disk-backed session store (`Web_Server/session_store.py`), a wall-clock basis, and day-scale timeouts. `Web_Server/serve.py` gains the `/auth/devices` page family, QR pairing assets, and CLI plumbing. No child-process or DB-layer changes.

**Tech Stack:** Python ≥3.12, aiohttp (already a `web` extra dep), `qrcode` (new, `web` extra only), vendored `qrcodegen.js` (MIT) for browser QR rendering. pytest with aiohttp `TestClient`.

**Spec:** `Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md` (§3 = this plan; §4.3 spike = Task 8; §5 sequencing)

## Global Constraints

- Targeted pytest runs only — never a full-suite sweep without explicit operator opt-in (repo rule).
- Raw cookie values never touch disk; only `_cookie_digest()` digests (hex-encoded for JSON keys).
- `web_sessions.json` is owned by exactly one process (the served parent); atomic write via temp file + `os.replace`, mode `0600`.
- No credential value (token, cookie, CSRF) in logs or error text.
- All new `[web_server]` config reads use the existing `get_cli_setting("web_server", key, default=…)` pattern in `create_server`.
- Imports: stdlib → third-party → local; Google-style docstrings on public functions.
- Execute on one feature branch off `origin/dev` (suggested: `feat/served-auth-phase1a`), one commit per step-group; Phase 1a ships as one PR.
- ADR required before code: Task 1 creates it (`backlog/decisions/233-served-web-session-persistence-and-device-pairing.md` — 233 is the next free number after 232).

---

### Task 1: ADR-233 and Backlog task

**Files:**
- Create: `backlog/decisions/233-served-web-session-persistence-and-device-pairing.md`
- Create via CLI: one Backlog task for Phase 1a

**Interfaces:**
- Consumes: nothing.
- Produces: ADR-233 path and the Backlog task id, referenced by later commits and the final Implementation Notes.

- [ ] **Step 1: Write ADR-233**

```markdown
# ADR-233: Served web session persistence and device pairing

Date: 2026-10-15
Status: Accepted (implements spec Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md §3)

## Context

Served mode admitted remote browsers with one shared long access token and
in-memory sessions: 30-minute idle / 8-hour absolute expiry, and every process
restart revoked every device. For a single operator on a phone this meant
re-typing the long token constantly.

## Decision

1. Sessions persist to `get_user_data_dir()/web_sessions.json` (atomic
   tmp+rename writes, mode 0600, cookie digests only — never raw cookie
   values). Defaults: 30-day sliding idle, 180-day absolute.
2. The auth clock basis switches from `time.monotonic` to `time.time()`
   (wall clock) so persisted timestamps survive restarts. Wall-clock steps
   at day-scale granularity are accepted.
3. Device pairing: the existing one-use bootstrap nonce gains a QR face —
   "Add a device" on `/auth/devices` and `tldw-serve --pair`. Nonce TTL for
   issuance is 120 s; exchange stays one-use and rate-limited.
4. `/auth/devices` lists devices (User-Agent-derived labels, editable),
   supports sign-out, per-device revoke, and revoke-all.
5. Token-form login defaults to persistent sessions; unchecking
   "remember this device" downgrades to the old short in-memory session.

## Consequences

- Stopping the process no longer revokes sessions. Incident response:
  Revoke all on /auth/devices, or stop the process AND delete
  web_sessions.json, or rotate the access token.
- A stolen cookie is long-lived. Mitigations: digest-only disk storage,
  0600 file, per-device revocation with last-seen visibility. No
  client-IP binding — Wi-Fi/cellular roaming would break sessions;
  accepted for a single-operator deployment.
- Loopback auto-login, login rate limiting, TLS/proxy fail-closed rules,
  SameSite=Strict + CSRF posture, and the session capacity cap are unchanged.

## Alternatives considered

- Passkeys/WebAuthn as primary: deferred to Phase 2 (LAN-by-IP origins
  cannot be WebAuthn RP IDs; QR covers every path).
- Network-layer trust (tailnet/proxy): rejected as primary — pushes setup
  burden, leaves the WebSocket session layer half-designed.
```

- [ ] **Step 2: Create the Backlog task and capture its id**

```bash
backlog task create "Served-mode auth foundation: persistent sessions, QR pairing, device manager (Phase 1a)" \
  -d "Implements spec 2026-10-15 §3: persistent device sessions, /auth/devices, QR pairing, tldw-serve --pair" \
  --ac "Restarting the served process keeps phone sessions logged in,Q R pairing works from /auth/devices and --pair,Logout and per-device/all-device revocation work,Web_Server README incident response updated,ADR-233 linked" \
  -s "In Progress"
```

Capture the printed task id as `<TASK-ID>` for later steps. Link ADR-233 in the task notes:

```bash
backlog task edit <TASK-ID> --notes "ADR: backlog/decisions/233-served-web-session-persistence-and-device-pairing.md"
```

- [ ] **Step 3: Commit**

```bash
git add backlog/decisions/233-served-web-session-persistence-and-device-pairing.md
git commit -m "docs(adr): ADR-233 served web session persistence and device pairing"
```

---

### Task 2: Session store module

**Files:**
- Create: `tldw_chatbook/Web_Server/session_store.py`
- Test: `Tests/Web_Server/test_session_store.py`

**Interfaces:**
- Consumes: nothing (framework-neutral; must not import `Canvas.web_auth` — that direction is reserved).
- Produces:

```python
@dataclass(frozen=True)
class StoredSession:
    digest_hex: str      # hex of BrowserSession.cookie_digest
    session_id: str
    csrf_token: str
    client_ip: str
    device_label: str
    created_at: float    # wall clock (time.time())
    last_seen_at: float
    absolute_expires_at: float

class WebSessionStore:
    def __init__(self, path: Path) -> None: ...
    def load(self) -> dict[str, StoredSession]: ...   # {} if missing or corrupt (log warning)
    def save(self, sessions: Mapping[str, StoredSession]) -> None: ...  # atomic, 0600
```

- [ ] **Step 1: Write failing tests**

```python
# Tests/Web_Server/test_session_store.py
from __future__ import annotations

import json
from pathlib import Path

from tldw_chatbook.Web_Server.session_store import StoredSession, WebSessionStore


def _record(digest_hex: str = "ab12", **overrides) -> StoredSession:
    fields = dict(
        digest_hex=digest_hex,
        session_id="sid-1",
        csrf_token="csrf-1",
        client_ip="127.0.0.1",
        device_label="iPhone · Safari",
        created_at=1000.0,
        last_seen_at=1000.0,
        absolute_expires_at=2000.0,
    )
    fields.update(overrides)
    return StoredSession(**fields)


def test_round_trip_preserves_records(tmp_path: Path) -> None:
    store = WebSessionStore(tmp_path / "web_sessions.json")
    store.save({"ab12": _record(), "cd34": _record("cd34", device_label="Mac · Chrome")})
    loaded = WebSessionStore(tmp_path / "web_sessions.json").load()
    assert set(loaded) == {"ab12", "cd34"}
    assert loaded["cd34"].device_label == "Mac · Chrome"


def test_save_is_atomic_and_private(tmp_path: Path) -> None:
    store = WebSessionStore(tmp_path / "web_sessions.json")
    store.save({"ab12": _record()})
    files = list(tmp_path.iterdir())
    assert [f.name for f in files] == ["web_sessions.json"]  # no temp leftovers
    assert (tmp_path / "web_sessions.json").stat().st_mode & 0o777 == 0o600


def test_file_never_contains_raw_cookie_only_hex_digests(tmp_path: Path) -> None:
    store = WebSessionStore(tmp_path / "web_sessions.json")
    store.save({"ab12": _record()})
    text = (tmp_path / "web_sessions.json").read_text()
    assert "ab12" in text  # digest hex is the only key material


def test_load_missing_or_corrupt_file_fails_safe(tmp_path: Path) -> None:
    assert WebSessionStore(tmp_path / "absent.json").load() == {}
    corrupt = tmp_path / "web_sessions.json"
    corrupt.write_text("{not json")
    assert WebSessionStore(corrupt).load() == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/Web_Server/test_session_store.py -v`
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Web_Server.session_store`

- [ ] **Step 3: Implement the store**

```python
# tldw_chatbook/Web_Server/session_store.py
"""Disk-backed persistence for served web sessions (ADR-233).

Framework-neutral on purpose: this module must not import
``tldw_chatbook.Canvas.web_auth`` — the manager imports the store, never the
reverse. Records carry cookie *digests* (hex) only; raw cookie values are
never written to disk.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

_FIELDS = (
    "digest_hex",
    "session_id",
    "csrf_token",
    "client_ip",
    "device_label",
    "created_at",
    "last_seen_at",
    "absolute_expires_at",
)


@dataclass(frozen=True)
class StoredSession:
    digest_hex: str
    session_id: str
    csrf_token: str
    client_ip: str
    device_label: str
    created_at: float
    last_seen_at: float
    absolute_expires_at: float


class WebSessionStore:
    """Single-owner JSON session store with atomic, private writes."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def load(self) -> dict[str, StoredSession]:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            logger.warning("discarding unreadable web session store: %s", error)
            return {}
        records: dict[str, StoredSession] = {}
        for key, value in raw.items():
            try:
                records[key] = StoredSession(**{name: value[name] for name in _FIELDS})
            except (KeyError, TypeError):
                continue  # skip malformed entries, keep the rest
        return records

    def save(self, sessions: Mapping[str, StoredSession]) -> None:
        payload = {key: asdict(record) for key, record in sessions.items()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=self._path.parent, prefix=".web_sessions-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle)
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, self._path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/Web_Server/test_session_store.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Web_Server/session_store.py Tests/Web_Server/test_session_store.py
git commit -m "feat(web-server): disk-backed session store (ADR-233)"
```

---

### Task 3: Manager persistence, wall clock, config timeouts, cookie max-age

**Files:**
- Modify: `tldw_chatbook/Canvas/web_auth.py` (constructor ~406-433, `_new_session` 497-524, `_prune_sessions` 526+, `touch_session` 699+, `_revoke_session` 709+, `revoke_all` 716+, defaults at 28-30)
- Modify: `tldw_chatbook/Web_Server/serve.py` (`ChatbookWebServerMixin.__init__` ~582-598, `_set_session_cookie` 737-747, `create_server` 1476-1590)
- Modify: `tldw_chatbook/Web_Server/README.md` (Remote access and authentication + Incident response sections)
- Test: `Tests/Canvas/test_web_auth.py`

**Interfaces:**
- Consumes: `WebSessionStore`, `StoredSession` (Task 2).
- Produces:
  - `BrowserSession` gains fields `device_label: str = ""` and `persist: bool = True`.
  - `WebAuthManager.__init__(..., session_store: WebSessionStore | None = None)`; default clock becomes `time.time`.
  - `login_with_access_token(supplied, *, client_ip, device_label="", persist=True)`; same two kwargs on `exchange_bootstrap` and `authenticate_local`.
  - `WebAuthManager.sessions_snapshot() -> list[BrowserSession]` (ordered oldest-first), `revoke_by_session_id(session_id: str) -> bool`, `rename_device(session_id: str, label: str) -> bool`.
  - Mixin ctor kwargs: `session_store=None`, `session_idle_timeout_seconds=None`, `session_absolute_timeout_seconds=None` (None → current manager defaults).
  - Config keys `[web_server] session_idle_timeout_days = 30`, `session_absolute_timeout_days = 180` (read in `create_server`, multiplied by 86400).

- [ ] **Step 1: Write failing manager tests** (append to `Tests/Canvas/test_web_auth.py`; reuse the file's existing `_manager()`-style fixture pattern — the existing helper builds `WebAuthManager(policy, clock=clock, idle_timeout_seconds=30, absolute_timeout_seconds=120, ...)` with a controllable clock)

```python
def _persisted_manager(tmp_path, **overrides):
    policy = build_web_auth_policy(
        host="127.0.0.1", port=8000, access_token="tok",
        public_url="https://chatbook.example", allow_insecure_remote_http=False,
        trusted_proxy_addresses=[], direct_tls=True,
    )
    store = WebSessionStore(tmp_path / "web_sessions.json")
    now = 1_000_000.0
    fields = dict(
        policy=policy,
        clock=lambda: now,
        idle_timeout_seconds=30 * 86400,
        absolute_timeout_seconds=180 * 86400,
        session_store=store,
    )
    fields.update(overrides)
    return WebAuthManager(**fields), store, lambda value: now.__setattr__...  # see note
```

Note: bind the fake clock through a small mutable holder (the existing tests already inject clocks this way — follow their local idiom instead of the pseudo-line above). Then:

```python
def test_session_survives_manager_restart(tmp_path) -> None:
    manager, store, _ = _persisted_manager(tmp_path)
    grant = manager.login_with_access_token("tok", client_ip="192.168.1.9")
    revived, _, _ = _persisted_manager(tmp_path)  # fresh manager, same store file
    session = revived.authenticate_request(_facts(cookie=grant.cookie_value))
    assert session.device_label == ""


def test_revocation_is_persisted(tmp_path) -> None:
    manager, store, _ = _persisted_manager(tmp_path)
    grant = manager.login_with_access_token("tok", client_ip="192.168.1.9")
    manager.revoke(grant.cookie_value)
    revived, _, _ = _persisted_manager(tmp_path)
    with pytest.raises(AuthenticationError):
        revived.authenticate_request(_facts(cookie=grant.cookie_value))


def test_short_login_is_not_persisted(tmp_path) -> None:
    manager, store, _ = _persisted_manager(tmp_path)
    manager.login_with_access_token(
        "tok", client_ip="192.168.1.9", persist=False
    )
    assert store.load() == {}


def test_default_clock_is_wall_clock() -> None:
    import time as _time
    policy = build_web_auth_policy(
        host="127.0.0.1", port=8000, access_token="tok",
        public_url="https://chatbook.example", allow_insecure_remote_http=False,
        trusted_proxy_addresses=[], direct_tls=True,
    )
    manager = WebAuthManager(policy)
    assert manager._clock is _time.time
```

(`_facts(cookie=...)` = the test file's existing `RequestFacts` builder for an authenticated GET; reuse the local idiom.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/Canvas/test_web_auth.py -v -k "survives_manager_restart or revocation_is_persisted or short_login or default_clock"`
Expected: FAIL — unexpected kwargs / no persistence.

- [ ] **Step 3: Implement manager changes**

1. Defaults (top of `web_auth.py`): `_DEFAULT_BOOTSTRAP_TTL_SECONDS = 120.0` (spec §3.3); add `_TOUCH_PERSIST_INTERVAL_SECONDS = 300.0`.
2. `BrowserSession`: add `device_label: str = ""`, `persist: bool = True`.
3. `WebAuthManager.__init__`: `clock: Callable[[], float] = time.time`; new kw `session_store=None` stored as `self._session_store`; after state init call `self._load_persisted_sessions(self._clock())`.
4. `_load_persisted_sessions(now)`: load store; drop records with `absolute_expires_at <= now` or `last_seen_at + idle_timeout <= now`; convert `StoredSession` → `BrowserSession` (digest bytes `bytes.fromhex(digest_hex)`, `persist=True`); order by `last_seen_at`; keep newest `max_sessions`; on any store write-back after prune (delete_expired → `self._persist_all()`).
5. Conversion helpers `_to_stored(session) -> StoredSession` / `_from_stored(record) -> BrowserSession` (module-private).
6. `_new_session(client_ip, now, *, device_label="", persist=True)`: set new fields; if `persist and self._session_store`: `self._persist_all()`.
7. `_persist_all()`: when `self._session_store` is set, save `{digest.hex(): _to_stored(s) for digest, s in self._sessions.items() if s.persist}`.
8. `_revoke_session` / `revoke` / `revoke_all` / `_prune_sessions` / capacity eviction in `_new_session`: call `self._persist_all()` after mutation.
9. `touch_session`: after updating `last_seen_at`, `if now - previous_last_seen >= _TOUCH_PERSIST_INTERVAL_SECONDS: self._persist_all()` (throttle; a crash losing minutes only shortens a session).
10. `login_with_access_token`, `exchange_bootstrap`, `authenticate_local`: accept `device_label: str = ""`, `persist: bool = True`; thread into `_new_session`.
11. Add `sessions_snapshot()`, `revoke_by_session_id()`, `rename_device()`:

```python
def sessions_snapshot(self) -> list[BrowserSession]:
    return sorted(self._sessions.values(), key=lambda s: s.last_seen_at)

def revoke_by_session_id(self, session_id: str) -> bool:
    session = next(
        (s for s in self._sessions.values() if s.session_id == session_id), None
    )
    if session is None:
        return False
    self._revoke_session(session)
    return True

def rename_device(self, session_id: str, label: str) -> bool:
    label = label.strip()[:64]
    session = next(
        (s for s in self._sessions.values() if s.session_id == session_id), None
    )
    if session is None or not label:
        return False
    session.device_label = label
    self._persist_all()
    return True
```

- [ ] **Step 4: Wire serve.py**

1. `ChatbookWebServerMixin.__init__`: accept `session_store=None`, `session_idle_timeout_seconds=None`, `session_absolute_timeout_seconds=None`; build `WebAuthManager(web_auth_policy, session_store=session_store, idle_timeout_seconds=… or default, absolute_timeout_seconds=… or default)` (pass only non-None overrides to keep test fixtures on defaults).
2. `_set_session_cookie`: `max_age = max(1, int(grant.expires_at - time.time()))` (import `time`; delete the `asyncio.get_running_loop()` use).
3. `create_server`: read the two day keys, clamp to ≥1, ensure absolute ≥ idle (else raise `ValueError` with the key names), multiply by `86400`, pass plus `session_store=WebSessionStore(get_user_data_dir() / "web_sessions.json")` into the server ctor (import `get_user_data_dir` from `tldw_chatbook.Utils.paths`).

- [ ] **Step 5: Update `Web_Server/README.md`**

- Replace the 30-minute/8-hour sentence with the new defaults and the two config keys.
- Incident response step 1 becomes: stop the served process **and** revoke devices — "Revoke all" on `/auth/devices`, or delete `web_sessions.json` from the user data dir while stopped; token rotation additionally invalidates pairing.

- [ ] **Step 6: Run targeted tests**

Run: `.venv/bin/pytest Tests/Canvas/test_web_auth.py Tests/Web_Server/ -v`
Expected: all PASS (existing tests included — the clock default change must not break them; fixtures inject clocks explicitly).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Canvas/web_auth.py tldw_chatbook/Web_Server/serve.py tldw_chatbook/Web_Server/README.md Tests/Canvas/test_web_auth.py
git commit -m "feat(web-auth): persistent device sessions on wall clock (ADR-233)"
```

---

### Task 4: Device labels, /auth/devices page, logout and revocation

**Files:**
- Modify: `tldw_chatbook/Web_Server/serve.py` (routes block ~663-683, handlers near `handle_login` 840-875)
- Create: `tldw_chatbook/Web_Server/static/devices.js`
- Test: `Tests/Web_Server/test_web_auth_integration.py`

**Interfaces:**
- Consumes: `request["chatbook_browser_session"]` and `request["chatbook_csrf"]` set by `_web_auth_middleware`; manager methods from Task 3.
- Produces: `GET /auth/devices` (HTML page), `POST /auth/logout`, `POST /auth/devices/revoke` (JSON body `{"session_id": …}`), `POST /auth/devices/rename` (JSON body `{"session_id": …, "label": …}`); `_device_label_from_user_agent(ua: str) -> str` module function.

- [ ] **Step 1: Write failing integration tests** (follow the file's `_FakeBaseServer`/`build_chatbook_web_server_class` fixture pattern)

```python
async def test_devices_page_lists_and_revokes(tmp_path, aiohttp_client_factory):
    # login via token form (follows existing token-login test in this file)
    client = await _logged_in_client(tmp_path, remember="1")
    page = await client.get("/auth/devices")
    assert page.status == 200
    assert "Sign out this device" in await page.text()

    resp = await client.post(
        "/auth/devices/revoke",
        json={"session_id": _current_session_id(client)},
        headers={"X-Chatbook-CSRF": _csrf(client)},   # use the repo's CSRF_HEADER_NAME value
    )
    assert resp.status == 200
    # current session revoked → next request bounces to login
    after = await client.get("/")
    assert after.status == 303


async def test_logout_clears_cookie(tmp_path, aiohttp_client_factory):
    client = await _logged_in_client(tmp_path)
    resp = await client.post("/auth/logout", headers={"X-Chatbook-CSRF": _csrf(client)})
    assert resp.status == 303
    assert SESSION_COOKIE_NAME not in SimpleCookie(resp.headers.get("Set-Cookie", ""))
```

(Reuse the file's existing helpers for policy/server/client construction; `_logged_in_client` wraps them plus the `/auth/login` POST from the current token tests; look up `CSRF_HEADER_NAME` in `Canvas/web_auth.py` imports and use the literal header it defines.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/Web_Server/test_web_auth_integration.py -v -k devices`
Expected: FAIL — 404 on `/auth/devices`.

- [ ] **Step 3: Implement**

1. `_device_label_from_user_agent(ua)`: platform markers `iPhone|iPad|Android|Macintosh|Windows|Linux` × browser markers `Edg/|Firefox|Chrome|Safari`, joined `" · "`, fallback `"Unknown device"`, stripped to 64 chars; HTML-escaped by the page renderer.
2. `handle_login` POST: derive label + `remember = data.get("remember") == "1"`, pass to `login_with_access_token`. `handle_login_bootstrap`: derive label, pass (always persist). Middleware's loopback auto-login path: pass `device_label=_device_label_from_user_agent(request.headers.get("User-Agent", ""))`.
3. `handle_devices_page`: read `request["chatbook_browser_session"]`; render inline HTML (same style as `handle_login`) — CSRF in `<meta name="chatbook-csrf">`, one `<section>` per `sessions_snapshot()` entry showing escaped label, created/last-seen (ISO via `datetime.fromtimestamp(..., UTC)`), current-device marker when `session_id` matches, buttons wired by `devices.js`; `<script src="/static/chatbook/devices.js" defer>`.
4. `handle_logout`: `self._web_auth.revoke(request.cookies.get(SESSION_COOKIE_NAME, ""))`; 303 to `/auth/login`; `response.del_cookie(SESSION_COOKIE_NAME, path="/")`.
5. `handle_devices_revoke` / `handle_devices_rename`: parse JSON (≤4096 bytes guard like `handle_login`), call `revoke_by_session_id` / `rename_device`, return `{"ok": bool}` with `Cache-Control: no-store`.
6. Register routes in the `routes` list (authed — NOT in `_public_auth_paths`):

```python
web.get("/auth/devices", self.handle_devices_page, name="devices"),
web.post("/auth/logout", self.handle_logout, name="logout"),
web.post("/auth/devices/revoke", self.handle_devices_revoke, name="devices_revoke"),
web.post("/auth/devices/rename", self.handle_devices_rename, name="devices_rename"),
```

7. `Web_Server/static/devices.js`: read CSRF from the meta tag, `fetch` POSTs with the CSRF header and `credentials: "same-origin"`, reload on success. No QR code yet (Task 5).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/Web_Server/test_web_auth_integration.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Web_Server/serve.py tldw_chatbook/Web_Server/static/devices.js Tests/Web_Server/test_web_auth_integration.py
git commit -m "feat(web-auth): /auth/devices manager, logout, per-device revoke/rename"
```

---

### Task 5: QR pairing and remember checkbox

**Files:**
- Create: `tldw_chatbook/Web_Server/static/qrcodegen.js` (vendored, MIT — Project Nayuki "QR Code generator library", single-file `qrcodegen.js`)
- Modify: `tldw_chatbook/Web_Server/static/devices.js`, `tldw_chatbook/Web_Server/serve.py` (`handle_login` GET form; new `handle_devices_pair`)
- Test: `Tests/Web_Server/test_web_auth_integration.py`

**Interfaces:**
- Consumes: `issue_browser_bootstrap_url()` (exists, serve.py ~896); TTL already raised to 120 s in Task 3.
- Produces: `POST /auth/devices/pair` → `{"url": "https://…/auth/bootstrap?nonce=…"}` (one-use).

- [ ] **Step 1: Write failing tests**

```python
async def test_pair_issues_one_use_bootstrap_url(tmp_path, aiohttp_client_factory):
    client = await _logged_in_client(tmp_path)
    resp = await client.post(
        "/auth/devices/pair", headers={"X-Chatbook-CSRF": _csrf(client)}
    )
    assert resp.status == 200
    url = (await resp.json())["url"]
    assert "/auth/bootstrap?nonce=" in url
    # exchanging it grants a session; a second exchange is denied
    phone = await _fresh_client(tmp_path)
    first = await phone.get(url, allow_redirects=False)
    assert first.status == 303
    assert SESSION_COOKIE_NAME in SimpleCookie(first.headers["Set-Cookie"])
    second = await phone.get(url, allow_redirects=False)
    assert second.status == 401


async def test_login_without_remember_stays_short(tmp_path, aiohttp_client_factory):
    client = await _logged_in_client(tmp_path, remember=None)
    cookie = SimpleCookie(client.session.cookie_jar.filter_cookies("")["..."].value)  # follow file idiom
    # short session is absent from the store file
    assert (tmp_path / "web_sessions.json").exists() is False or <store empty>
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/Web_Server/test_web_auth_integration.py -v -k "pair or remember"`
Expected: FAIL — 404 on `/auth/devices/pair`.

- [ ] **Step 3: Implement**

1. Vendor `qrcodegen.js` from Project Nayuki (MIT; pin the exact version and record the source URL + hash in a `Web_Server/static/VENDORED.md` note).
2. `handle_devices_pair`: return `{"url": self.issue_browser_bootstrap_url()}` with `Cache-Control: no-store`.
3. Route: `web.post("/auth/devices/pair", self.handle_devices_pair, name="devices_pair")`.
4. `devices.js`: "Add a device" button → POST pair → render `url` as QR into a `<canvas>` via `qrcodegen.QrCode.encodeText(url, Ecc.MEDIUM)`; show the raw URL as selectable fallback text; auto-refresh via a new pair request after 100 s.
5. `handle_login` GET form: add

```html
<label><input type="checkbox" name="remember" value="1" checked> Remember this
device (persistent session; uncheck on shared machines)</label>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/Web_Server/test_web_auth_integration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Web_Server/static/qrcodegen.js tldw_chatbook/Web_Server/static/VENDORED.md tldw_chatbook/Web_Server/static/devices.js tldw_chatbook/Web_Server/serve.py Tests/Web_Server/test_web_auth_integration.py
git commit -m "feat(web-auth): QR device pairing and remember-device checkbox (ADR-233)"
```

---

### Task 6: tldw-serve --pair CLI

**Files:**
- Modify: `tldw_chatbook/Web_Server/serve.py` (`main()` ~1642-1661, `run_web_server`), `pyproject.toml` (`web` extra + line 364/477 area)
- Test: `Tests/Web_Server/test_serve_cli_pair.py`

**Interfaces:**
- Consumes: `issue_browser_bootstrap_url()` on the built server.
- Produces: `--pair` flag; `_print_pair_qr(server, *, echo=print) -> bool`.

- [ ] **Step 1: Write failing tests**

```python
# Tests/Web_Server/test_serve_cli_pair.py
from tldw_chatbook.Web_Server.serve import _print_pair_qr


class _Stub:
    public_url = "https://chatbook.example"
    def issue_browser_bootstrap_url(self) -> str:
        return "https://chatbook.example/auth/bootstrap?nonce=abc123"


def test_pair_prints_qr_and_url(capsys) -> None:
    qrcode = pytest.importorskip("qrcode")
    assert _print_pair_qr(_Stub()) is True
    out = capsys.readouterr().out
    assert "https://chatbook.example/auth/bootstrap?nonce=abc123" in out
    assert "█" in out or "▀" in out  # ascii qr block output


def test_pair_skips_without_public_url(capsys) -> None:
    class _NoUrl(_Stub):
        public_url = None
    assert _print_pair_qr(_NoUrl()) is False
    assert "public_url" in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/Web_Server/test_serve_cli_pair.py -v`
Expected: FAIL — no `_print_pair_qr`.

- [ ] **Step 3: Implement**

```python
def _print_pair_qr(server, *, echo=print) -> bool:
    """Print a one-use pairing QR for a server about to start (ADR-233 §3.3)."""
    if not getattr(server, "public_url", None):
        echo("--pair skipped: [web_server] public_url is not configured.")
        return False
    url = server.issue_browser_bootstrap_url()
    try:
        import qrcode
    except ImportError:
        echo(f"Pair with this one-use link (install 'qrcode' for a QR): {url}")
        return True
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    echo("Scan with your phone to sign in (valid once, for 120 seconds):")
    qr.print_ascii(invert=True)
    echo(url)
    return True
```

- `main()`: `parser.add_argument("--pair", action="store_true", help="Print a QR to pair a phone before serving")`; plumb `pair=args.pair` into `run_web_server`, which calls `_print_pair_qr(server)` between `create_server(...)` and `server.serve(...)`.
- `pyproject.toml`: add `"qrcode"` to the `web` extra (both dependency blocks that list `textual-serve`).

- [ ] **Step 4: Run tests, install extra, verify**

Run: `.venv/bin/pip install -e ".[web]" && .venv/bin/pytest Tests/Web_Server/test_serve_cli_pair.py -v`
Expected: PASS (both tests once `qrcode` is installed).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Web_Server/serve.py pyproject.toml Tests/Web_Server/test_serve_cli_pair.py
git commit -m "feat(serve): tldw-serve --pair prints one-use phone pairing QR (ADR-233)"
```

---

### Task 7: Extract the approval capture cap out of the widget module

**Files:**
- Create: `tldw_chatbook/Chat/approval_constants.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py` (move the literal out, import it back), `tldw_chatbook/Chat/console_chat_controller.py:51`, `tldw_chatbook/Chat/permission_summary_service.py:15`, `tldw_chatbook/Agents/local_tool_provider.py:46` (import-site flips)
- Test: `Tests/Chat/test_approval_constants.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `from tldw_chatbook.Chat.approval_constants import TOOL_DESCRIPTION_CAPTURE_CAP` — importable with zero Textual imports (spec §4.3 shared requirement; prerequisite for the headless parent import graph).

- [ ] **Step 1: Write failing test**

```python
# Tests/Chat/test_approval_constants.py
from pathlib import Path

import tldw_chatbook.Chat.approval_constants as module


def test_module_has_no_textual_dependency() -> None:
    source = Path(module.__file__).read_text()
    assert "textual" not in source


def test_value_matches_widget_re_export() -> None:
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        TOOL_DESCRIPTION_CAPTURE_CAP as widget_value,
    )
    assert module.TOOL_DESCRIPTION_CAPTURE_CAP == widget_value
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest Tests/Chat/test_approval_constants.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

1. New module holds the literal verbatim (copy the exact value now at the top of `chat_approval_card.py`) plus a one-line docstring noting it is the shared capture cap for tool descriptions (widget, controller, permission summary, local tool provider).
2. `chat_approval_card.py`: delete the literal, import from the new module (keeps the name re-exported for any external importers).
3. Flip the three other import sites listed above to the new module.
4. `grep -rn "TOOL_DESCRIPTION_CAPTURE_CAP" tldw_chatbook Tests --include="*.py"` — confirm no importer still points at the widget except the widget itself.

- [ ] **Step 4: Run targeted tests**

Run: `.venv/bin/pytest Tests/Chat/test_approval_constants.py Tests/Console -k "approval or tool" -v` (plus the touched modules' existing approval tests)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/approval_constants.py tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/permission_summary_service.py tldw_chatbook/Agents/local_tool_provider.py Tests/Chat/test_approval_constants.py
git commit -m "refactor(chat): approval capture cap lives in a textual-free module (spec 4.3)"
```

---

### Task 8: Phase 1b engine-binding spike (throwaway probe)

**Files:**
- Create (scratch, NOT committed): `/tmp/mobile_binding_spike.py`
- Create (committed): `Docs/superpowers/spikes/2026-10-15-mobile-engine-binding-spike.md`

**Interfaces:**
- Consumes: `ConsoleChatStore`, `ConsoleChatController` constructor surface (`Chat/console_chat_controller.py:3909-3998`), `Tests/Chat/conftest.py::StreamingGateway` (fake provider gateway), Task 7's textual-free constant.
- Produces: a written go/no-go on Binding 1 (parent-process service) vs Binding 2 (headless child App) that the Phase 1b plan argues from — spec §4.3 spike criteria.

- [ ] **Step 1: Run the probe**

Spike criteria (spec §4.3): construct store + controller against a tmp SQLite DB with a **minimal, App-free context**, complete one **agent-mode** round-trip (a tool call + one approval decision) using `StreamingGateway` as the provider gateway. Plain-chat-only success does not qualify. Record: which of the ~50 constructor seams had to be non-None, whether approval escalation is observable/resolvable without widgets, and whether importing the controller in a plain process (no Textual app) pulls Textual transitively (`python -c "import sys; import tldw_chatbook.Chat.console_chat_controller as m; print([n for n in sys.modules if n.startswith('textual')])"`).

- [ ] **Step 2: Write the findings doc**

`Docs/superpowers/spikes/2026-10-15-mobile-engine-binding-spike.md` records: environment, what was constructed, the seam inventory (non-None requirements), the agent round-trip result, the transitive-import result, and the recommendation (Binding 1 / Binding 1-scoped-to-plain-chat / Binding 2) with reasoning. Label the probe script throwaway and include its path for reference only.

- [ ] **Step 3: Close out**

- Mark all Backlog ACs checked, add the `## Implementation Notes` section to `<TASK-ID>` (approach, files, ADR-233 link, spike pointer), set status Done via `backlog task edit <TASK-ID> -s Done`.
- Lessons check: only add a `backlog/docs/lessons-*.md` entry if something generalizable surfaced (e.g. the monotonic-clock persistence trap); state the incident, not just the rule.

- [ ] **Step 4: Commit**

```bash
git add Docs/superpowers/spikes/2026-10-15-mobile-engine-binding-spike.md
git commit -m "docs(spike): mobile engine-binding feasibility findings (spec 4.3)"
```

---

## Self-Review

**Spec coverage (Phase 1a):** §3.1 persistence+clock+timeouts+cookie → Task 2+3; §3.2 devices page/logout/revoke → Task 4; §3.3 QR + `--pair` + remember checkbox + 120 s TTL → Tasks 3/5/6; §3.4 README incident response → Task 3 Step 5; ADR → Task 1; §4.3 shared constant-extraction requirement → Task 7; §4.3 spike → Task 8. §3.5 (passkeys) and the rest of §4 are Phase 1b/2 by the spec's own sequencing (§5) — out of this plan by design.

**Placeholder scan:** the `_persisted_manager` fixture and `_logged_in_client`/`_csrf`/`_facts` helpers defer to the existing test files' local idioms with exact anchors (file + pattern named); the Nayuki vendoring names source, license, and pinning requirements. No TBD/TODO steps.

**Type consistency:** `StoredSession`/`WebSessionStore` names are identical in Tasks 2/3; `device_label`/`persist` kwargs thread Tasks 3→4→5; `sessions_snapshot`/`revoke_by_session_id`/`rename_device` (Task 3) match Task 4 handler calls; `_print_pair_qr` signature matches Task 6 tests.
