# Skills Foundation — Trust Isolation, Recovery & Discoverability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local skill trust profile-isolated, always recoverable (no UI dead-end), and discoverable/reachable from the Skills list — so a new user can set up trust in one click and an upgraded user is never stuck.

**Architecture:** Profile-scope the two OS-keyring entries (generation marker + derived-key cache) by a per-profile account suffix; add `clear()`/delete primitives to those stores and a public manifest delete; add `reset_trust()` and a structured `trust_posture()` to `SkillTrustService`; render an adaptive trust header on the Skills list canvas and wire its actions (reusing the existing bootstrap/passphrase modal flows) with a confirm-gated Reset that also serves as the forgot-passphrase escape hatch.

**Tech Stack:** Python 3.11+, Textual, pytest (+ pytest-asyncio), the existing `Skills_Interop` trust subsystem, the Library screen + `library_skills_canvas` widget.

## Global Constraints

- Spec: `Docs/superpowers/specs/2026-07-21-skills-foundation-trust-design.md`. Layer 0, Spec 1.
- **Out of scope** (do NOT touch): nested supporting-file import (#3, Spec 2), pack/batch import, remote fetch, agent tool.
- **No trust *data* migration:** old global keyring entries are never read again; prior users re-setup once. We DO add code to detect + cleanly present the orphaned-manifest state.
- **Safety invariant:** never auto-destroy trust data on marker absence. Only a clean `load_marker() == None` counts as orphaned; a raised `SkillTrustMarkerUnavailable` is "transient/unavailable" and must never trigger a destructive path.
- **Reset is destructive and user-initiated:** always confirmation-gated; copy must state "skills are not deleted — only trust is reset."
- Run tests with the venv: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`.
- Tests must NOT touch the real OS keyring — use the in-repo fake keyring backend pattern already used in `Tests/Skills/` (a dict-backed object with `get_password`/`set_password`/`delete_password`).
- `Settings` reads `SkillTrustService.overall_status()` (`settings_screen.py:3607`) — do **not** change `overall_status()`'s signature or return contract; add the new `trust_posture()` alongside it.
- Do not run broad `pkill pytest` — scope test runs to this worktree.

---

### Task 1: Profile-scope the keyring marker + key-cache accounts

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_store.py` (`KeyringSkillTrustGenerationMarkerStore`, `KeyringSkillTrustKeyCache`, `build_skill_trust_marker_store_with_fallback`, `build_default_skill_trust_key_cache`)
- Modify: `tldw_chatbook/app.py:4497-4514` (derive + pass the scope)
- Test: `Tests/Skills/test_skill_trust_store_scoping.py` (new)

**Interfaces:**
- Produces: `KeyringSkillTrustGenerationMarkerStore(service_name=..., keyring_backend=..., account_scope="")`; `KeyringSkillTrustKeyCache(..., account_scope="")`. When `account_scope` is non-empty, the keyring account becomes `f"{_MARKER_USERNAME}:{account_scope}"` / `f"{_KEY_CACHE_USERNAME}:{account_scope}"`.
- Produces: `build_skill_trust_marker_store_with_fallback(*, fallback_marker_path, keyring_backend=None, account_scope="")`; `build_default_skill_trust_key_cache(keyring_backend=None, account_scope="")`.
- Produces: module-level `skill_trust_account_scope(store_dir: Path) -> str` returning the first 16 hex chars of `sha256(str(store_dir.resolve()))`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Skills/test_skill_trust_store_scoping.py`:

```python
from pathlib import Path

from tldw_chatbook.Skills_Interop.skill_trust_store import (
    KeyringSkillTrustGenerationMarkerStore,
    KeyringSkillTrustKeyCache,
    skill_trust_account_scope,
    _MARKER_USERNAME,
    _KEY_CACHE_USERNAME,
)


class _FakeKeyring:
    """Secure-looking dict-backed keyring for tests."""

    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    # is_secure_keyring_backend checks the class/module; see conftest note.
    def get_password(self, service, account):
        return self.store.get((service, account))

    def set_password(self, service, account, value):
        self.store[(service, account)] = value

    def delete_password(self, service, account):
        self.store.pop((service, account), None)


def test_account_scope_is_stable_and_dir_specific(tmp_path):
    a = skill_trust_account_scope(tmp_path / "profileA" / "skills" / "trust")
    b = skill_trust_account_scope(tmp_path / "profileB" / "skills" / "trust")
    assert a and b and a != b
    assert a == skill_trust_account_scope(tmp_path / "profileA" / "skills" / "trust")
    assert len(a) == 16


def test_scoped_marker_accounts_do_not_cross_read(monkeypatch):
    kr = _FakeKeyring()
    monkeypatch.setattr(
        "tldw_chatbook.Skills_Interop.skill_trust_store.is_secure_keyring_backend",
        lambda backend: True,
    )
    a = KeyringSkillTrustGenerationMarkerStore(keyring_backend=kr, account_scope="aaaa")
    b = KeyringSkillTrustGenerationMarkerStore(keyring_backend=kr, account_scope="bbbb")
    a.save_marker(generation=1, manifest_digest="digestA")
    assert a.load_marker() is not None
    assert b.load_marker() is None  # scoped: B cannot see A's marker
    # And the legacy global account is untouched by scoped writes:
    assert (KeyringSkillTrustGenerationMarkerStore.__dataclass_fields__)  # sanity
    assert kr.store.get(("tldw_chatbook.skill_trust", _MARKER_USERNAME)) is None


def test_scoped_key_cache_accounts_do_not_cross_read(monkeypatch):
    kr = _FakeKeyring()
    monkeypatch.setattr(
        "tldw_chatbook.Skills_Interop.skill_trust_store.is_secure_keyring_backend",
        lambda backend: True,
    )
    ca = KeyringSkillTrustKeyCache(keyring_backend=kr, account_scope="aaaa")
    cb = KeyringSkillTrustKeyCache(keyring_backend=kr, account_scope="bbbb")
    # Write a raw payload under A's scoped account, prove B can't read it.
    kr.set_password("tldw_chatbook.skill_trust.keys", f"{_KEY_CACHE_USERNAME}:aaaa", "x")
    assert kr.get_password("tldw_chatbook.skill_trust.keys", f"{_KEY_CACHE_USERNAME}:bbbb") is None
    assert kr.get_password("tldw_chatbook.skill_trust.keys", _KEY_CACHE_USERNAME) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_trust_store_scoping.py -q`
Expected: FAIL — `ImportError: cannot import name 'skill_trust_account_scope'` and `account_scope` unknown kwarg.

- [ ] **Step 3: Add the scope helper + account_scope fields**

In `tldw_chatbook/Skills_Interop/skill_trust_store.py`, add near the top-level helpers (after the `_KEY_CACHE_FIELDS` block):

```python
import hashlib


def skill_trust_account_scope(store_dir: Path) -> str:
    """Per-profile keyring account suffix derived from the resolved store dir.

    Isolates each profile's trust marker + key cache in the shared OS keyring
    so profiles/users on one machine cannot read each other's entries. Moving
    the data dir changes the scope (a one-time re-setup, by design).
    """
    resolved = str(Path(store_dir).resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
```

In `KeyringSkillTrustGenerationMarkerStore` add the field and use it:

```python
@dataclass(slots=True, repr=False)
class KeyringSkillTrustGenerationMarkerStore:
    service_name: str = _DEFAULT_MARKER_SERVICE_NAME
    keyring_backend: Any | None = None
    account_scope: str = ""

    # ... __post_init__ unchanged ...

    @property
    def _account(self) -> str:
        return f"{_MARKER_USERNAME}:{self.account_scope}" if self.account_scope else _MARKER_USERNAME
```

Then replace the two `_MARKER_USERNAME` uses in `load_marker`/`save_marker` with `self._account`.

In `KeyringSkillTrustKeyCache` add `account_scope: str = ""` and a `_account` property returning `f"{_KEY_CACHE_USERNAME}:{self.account_scope}"` when scoped, and replace the two `_KEY_CACHE_USERNAME` uses in `load_keys`/`save_keys` with `self._account`.

Thread the scope through the builders:

```python
def build_skill_trust_marker_store_with_fallback(
    *,
    fallback_marker_path: Path,
    keyring_backend: Any | None = None,
    account_scope: str = "",
):
    try:
        keyring_store = KeyringSkillTrustGenerationMarkerStore(
            keyring_backend=keyring_backend, account_scope=account_scope
        )
        return keyring_store, False
    except Exception:
        return FileSkillTrustGenerationMarkerStore(fallback_marker_path), True


def build_default_skill_trust_key_cache(
    keyring_backend: Any | None = None,
    account_scope: str = "",
) -> "KeyringSkillTrustKeyCache | None":
    try:
        return KeyringSkillTrustKeyCache(
            keyring_backend=keyring_backend, account_scope=account_scope
        )
    except Exception:
        return None
```

(Preserve each builder's existing body/branches; only add the `account_scope` param + pass-through. Verify the real current body before editing — the snippet shows the shape, not a verbatim replacement.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_trust_store_scoping.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire the scope in app.py**

In `tldw_chatbook/app.py`, at the skills-trust wiring (~4497-4514), compute the scope once and pass it to both builders:

```python
local_skills_store_dir = get_user_data_dir() / "skills"
trust_store_dir = local_skills_store_dir / "trust"
trust_account_scope = skill_trust_account_scope(trust_store_dir)
skill_trust_marker_store, reduced_rollback_protection = (
    build_skill_trust_marker_store_with_fallback(
        fallback_marker_path=trust_store_dir / "generation_marker.json",
        account_scope=trust_account_scope,
    )
)
self.local_skill_trust_service = SkillTrustService(
    skills_dir=local_skills_store_dir / "skills",
    trust_store=SkillTrustStore(
        store_dir=trust_store_dir,
        marker_store=skill_trust_marker_store,
    ),
    key_cache=build_default_skill_trust_key_cache(account_scope=trust_account_scope),
    keyring_convenience_enabled=False,
    reduced_rollback_protection=reduced_rollback_protection,
)
```

Add `skill_trust_account_scope` to the existing `from ...skill_trust_store import (...)` block in `app.py`.

- [ ] **Step 6: Run the broader trust suite for regressions + commit**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/ -q`
Expected: PASS (existing trust tests unaffected; new scoping tests pass).

```bash
git add tldw_chatbook/Skills_Interop/skill_trust_store.py tldw_chatbook/app.py Tests/Skills/test_skill_trust_store_scoping.py
git commit -m "fix(skills): profile-scope trust keyring marker + key cache (Spec 1 T1)"
```

---

### Task 2: Store `clear()`/delete primitives (reset foundation)

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_store.py` (marker Protocol + `FileSkillTrustGenerationMarkerStore`, `KeyringSkillTrustGenerationMarkerStore`, `UnavailableSkillTrustGenerationMarkerStore`, `KeyringSkillTrustKeyCache`, `SkillTrustStore`)
- Test: `Tests/Skills/test_skill_trust_store_reset.py` (new)

**Interfaces:**
- Produces: `SkillTrustGenerationMarkerStore.clear() -> None` on every marker variant.
- Produces: `KeyringSkillTrustKeyCache.clear() -> None`.
- Produces: `SkillTrustStore.delete_manifest() -> None` (removes manifest file + snapshots dir; missing-ok).

- [ ] **Step 1: Write the failing test**

Create `Tests/Skills/test_skill_trust_store_reset.py`:

```python
from pathlib import Path

from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    KeyringSkillTrustGenerationMarkerStore,
    KeyringSkillTrustKeyCache,
    SkillTrustStore,
    _MARKER_USERNAME,
)


class _FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, s, a):
        return self.store.get((s, a))

    def set_password(self, s, a, v):
        self.store[(s, a)] = v

    def delete_password(self, s, a):
        self.store.pop((s, a), None)


def test_file_marker_clear_is_idempotent(tmp_path):
    store = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store.save_marker(generation=1, manifest_digest="d")
    assert store.load_marker() is not None
    store.clear()
    assert store.load_marker() is None
    store.clear()  # idempotent, no raise
    assert store.load_marker() is None


def test_keyring_marker_clear_deletes_scoped_account(monkeypatch):
    kr = _FakeKeyring()
    monkeypatch.setattr(
        "tldw_chatbook.Skills_Interop.skill_trust_store.is_secure_keyring_backend",
        lambda backend: True,
    )
    store = KeyringSkillTrustGenerationMarkerStore(keyring_backend=kr, account_scope="aaaa")
    store.save_marker(generation=1, manifest_digest="d")
    store.clear()
    assert store.load_marker() is None
    assert kr.store == {}


def test_keyring_key_cache_clear(monkeypatch):
    kr = _FakeKeyring()
    monkeypatch.setattr(
        "tldw_chatbook.Skills_Interop.skill_trust_store.is_secure_keyring_backend",
        lambda backend: True,
    )
    cache = KeyringSkillTrustKeyCache(keyring_backend=kr, account_scope="aaaa")
    kr.set_password("tldw_chatbook.skill_trust.keys", cache._account, "payload")
    cache.clear()
    assert kr.store == {}


def test_store_delete_manifest_removes_file_and_snapshots(tmp_path):
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path, marker_store=marker)
    store.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    store.manifest_path.write_text("{}", encoding="utf-8")
    store.snapshots_dir.mkdir(parents=True, exist_ok=True)
    (store.snapshots_dir / "snap").write_text("x", encoding="utf-8")
    store.delete_manifest()
    assert not store.manifest_path.exists()
    assert not store.snapshots_dir.exists()
    store.delete_manifest()  # idempotent
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_trust_store_reset.py -q`
Expected: FAIL — `AttributeError: 'FileSkillTrustGenerationMarkerStore' object has no attribute 'clear'`.

- [ ] **Step 3: Implement clear()/delete_manifest()**

Add to the marker `Protocol` (`SkillTrustGenerationMarkerStore`): `def clear(self) -> None: ...`.

`FileSkillTrustGenerationMarkerStore.clear`:

```python
    def clear(self) -> None:
        """Remove the on-disk marker file (missing-ok, no raise)."""
        try:
            self.marker_path.unlink(missing_ok=True)
        except OSError:
            pass
```

`KeyringSkillTrustGenerationMarkerStore.clear`:

```python
    def clear(self) -> None:
        """Delete the scoped keyring marker entry (best-effort)."""
        deleter = getattr(self.keyring_backend, "delete_password", None)
        if callable(deleter):
            try:
                deleter(self.service_name, self._account)
            except Exception:
                pass
```

`UnavailableSkillTrustGenerationMarkerStore.clear`: `def clear(self) -> None: return None`.

`KeyringSkillTrustKeyCache.clear`: same shape as the keyring marker `clear` but using `self._account` on `self.service_name`.

`SkillTrustStore.delete_manifest`:

```python
    def delete_manifest(self) -> None:
        """Remove the manifest payload and all snapshots (missing-ok)."""
        import shutil
        try:
            self.manifest_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            shutil.rmtree(self.snapshots_dir, ignore_errors=True)
        except OSError:
            pass
```

- [ ] **Step 4: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_trust_store_reset.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/skill_trust_store.py Tests/Skills/test_skill_trust_store_reset.py
git commit -m "feat(skills): add clear()/delete_manifest primitives for trust reset (Spec 1 T2)"
```

---

### Task 3: `reset_trust()` + structured `trust_posture()` on the service

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_service.py` (add `reset_trust`, `trust_posture`)
- Test: `Tests/Skills/test_skill_trust_service_reset_posture.py` (new)

**Interfaces:**
- Consumes: `SkillTrustStore.delete_manifest()`, `marker_store.clear()`, `key_cache.clear()` (Task 2); `skill_trust_account_scope` (Task 1, indirectly via wiring).
- Produces: `SkillTrustService.reset_trust() -> None` — clears manifest+snapshots, marker, key cache; resets in-memory `_keys`/`_salt`/`keyring_convenience_enabled`; best-effort, never raises.
- Produces: `SkillTrustService.trust_posture() -> str` returning exactly one of:
  `"needs_setup"` | `"needs_resetup"` | `"unavailable"` | `"locked"` | `"error"` | `"ready"`.
  Contract:
  - no manifest, `load_marker()` is `None` → `"needs_setup"`
  - no manifest, `load_marker()` returns a value → `"needs_resetup"` (poisoned fresh profile: inherited/foreign marker, no local manifest)
  - manifest present, `load_marker()` raises `SkillTrustMarkerUnavailable` → `"unavailable"`
  - manifest present, `load_marker()` is `None` → `"needs_resetup"` (orphaned manifest — the upgrade case)
  - manifest present, marker present, `self._keys is None` → `"locked"`
  - manifest present, marker present, keys loaded, `_load_valid_manifest()` raises → `"error"`
  - otherwise → `"ready"`

- [ ] **Step 1: Write the failing test**

Create `Tests/Skills/test_skill_trust_service_reset_posture.py`:

```python
import secrets
from pathlib import Path

import pytest

from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
    SkillTrustMarkerUnavailable,
)


def _service(tmp_path, marker=None):
    marker = marker or FileSkillTrustGenerationMarkerStore(tmp_path / "trust" / "marker.json")
    (tmp_path / "trust").mkdir(parents=True, exist_ok=True)
    (tmp_path / "skills").mkdir(parents=True, exist_ok=True)
    return SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker),
        key_cache=None,
    )


def test_posture_needs_setup_on_pristine(tmp_path):
    svc = _service(tmp_path)
    assert svc.trust_posture() == "needs_setup"


def test_posture_needs_resetup_when_marker_present_but_no_manifest(tmp_path):
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "trust" / "marker.json")
    (tmp_path / "trust").mkdir(parents=True, exist_ok=True)
    marker.save_marker(generation=1, manifest_digest="d")  # foreign/inherited marker
    svc = _service(tmp_path, marker=marker)
    assert svc.trust_posture() == "needs_resetup"


def test_posture_unavailable_when_marker_load_raises(tmp_path):
    class _Raising:
        def load_marker(self):
            raise SkillTrustMarkerUnavailable("locked keychain")

        def save_marker(self, **k):
            pass

        def clear(self):
            pass

    svc = _service(tmp_path, marker=_Raising())
    # Force has_manifest True by writing a manifest file placeholder path.
    svc.trust_store.manifest_path.write_text("{}", encoding="utf-8")
    assert svc.trust_posture() == "unavailable"


def test_reset_then_bootstrap_recovers_from_poison(tmp_path):
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "trust" / "marker.json")
    (tmp_path / "trust").mkdir(parents=True, exist_ok=True)
    marker.save_marker(generation=9, manifest_digest="stale")  # poison
    svc = _service(tmp_path, marker=marker)
    assert svc.trust_posture() == "needs_resetup"
    svc.reset_trust()
    assert svc.trust_posture() == "needs_setup"
    svc.bootstrap_trust("passphrase", salt=secrets.token_bytes(32))
    assert svc.trust_posture() == "ready"


def test_reset_trust_is_non_crashing_and_idempotent(tmp_path):
    svc = _service(tmp_path)
    svc.reset_trust()
    svc.reset_trust()  # idempotent, no raise
    assert svc.trust_posture() == "needs_setup"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_trust_service_reset_posture.py -q`
Expected: FAIL — `AttributeError: 'SkillTrustService' object has no attribute 'trust_posture'`.

- [ ] **Step 3: Implement `reset_trust` + `trust_posture`**

In `tldw_chatbook/Skills_Interop/skill_trust_service.py`, add these methods to `SkillTrustService` (near `overall_status`):

```python
    def reset_trust(self) -> None:
        """Clear all local trust state, returning the profile to first-run.

        Destructive: drops the trusted baseline (every skill returns to
        "needs review"). Skills themselves are untouched. Best-effort and
        non-raising so a partially-available keyring never blocks recovery.
        """
        try:
            self.trust_store.delete_manifest()
        except Exception:
            pass
        for target in (self.trust_store.marker_store, self.key_cache):
            clear = getattr(target, "clear", None)
            if callable(clear):
                try:
                    clear()
                except Exception:
                    pass
        self._keys = None
        self._salt = None
        self.keyring_convenience_enabled = False

    def _safe_load_marker(self):
        """Return (marker, available). available=False iff the marker store raised."""
        try:
            return self.trust_store.marker_store.load_marker(), True
        except SkillTrustMarkerUnavailable:
            return None, False
        except Exception:
            return None, False

    def trust_posture(self) -> str:
        """Structured global trust posture for the Skills list header.

        See the plan's Task 3 interface contract for the exact mapping.
        """
        has_manifest = self.trust_store.has_manifest()
        marker, available = self._safe_load_marker()
        if not has_manifest:
            return "needs_resetup" if marker else "needs_setup"
        if not available:
            return "unavailable"
        if marker is None:
            return "needs_resetup"  # orphaned manifest (upgrade case)
        if self._keys is None:
            return "locked"
        try:
            self._load_valid_manifest()
        except (SkillTrustMarkerUnavailable, ValueError):
            return "error"
        return "ready"
```

- [ ] **Step 4: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_trust_service_reset_posture.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Full trust-suite regression + commit**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/ -q`
Expected: PASS.

```bash
git add tldw_chatbook/Skills_Interop/skill_trust_service.py Tests/Skills/test_skill_trust_service_reset_posture.py
git commit -m "feat(skills): reset_trust + structured trust_posture (Spec 1 T3)"
```

---

### Task 4: Adaptive trust header on the Skills list (state + canvas)

**Files:**
- Modify: `tldw_chatbook/Library/library_skills_state.py` (`SkillsListState`, `build_skills_list_state`, add `skill_trust_header_line`)
- Modify: `tldw_chatbook/Widgets/Library/library_skills_canvas.py` (`_compose_list` header render; `LibrarySkillsListCanvas.__init__` gains `trust_posture`)
- Test: `Tests/UI/test_library_skills_canvas.py` (extend), `Tests/Library/test_library_skills_state.py` (extend)

**Interfaces:**
- Consumes: `trust_posture()` string from Task 3 (passed in by the screen, Task 5).
- Produces: `skill_trust_header_line(posture: str, blocked_count: int) -> tuple[str, str] | None` — returns `(copy, action_id)` or `None` when the header should be hidden. `action_id` ∈ {`"setup"`,`"resetup"`,`"retry"`,`"unlock"`,`"review"`,`""`}.
- Produces: `LibrarySkillsListCanvas(..., trust_posture: str = "")` rendering `#library-skills-trust-header` (a `Static`) + an inline action `Button` with id `library-skills-trust-action` when `action_id` is non-empty.

- [ ] **Step 1: Write the failing test (state helper)**

Add to `Tests/Library/test_library_skills_state.py`:

```python
def test_skill_trust_header_line_maps_postures():
    from tldw_chatbook.Library.library_skills_state import skill_trust_header_line

    assert skill_trust_header_line("needs_setup", 0)[1] == "setup"
    assert "isn't set up" in skill_trust_header_line("needs_setup", 0)[0]
    assert skill_trust_header_line("needs_resetup", 0)[1] == "resetup"
    assert "again after an update" in skill_trust_header_line("needs_resetup", 0)[0]
    assert skill_trust_header_line("unavailable", 0)[1] == "retry"
    assert skill_trust_header_line("locked", 0)[1] == "unlock"
    # ready + blocked skills -> review; ready + none -> quiet 'ready'
    assert skill_trust_header_line("ready", 3)[1] == "review"
    assert "3 skill" in skill_trust_header_line("ready", 3)[0]
    assert skill_trust_header_line("ready", 0)[1] == ""
    # disabled/empty posture -> hidden
    assert skill_trust_header_line("", 0) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_skills_state.py -q -k trust_header_line`
Expected: FAIL — `ImportError: cannot import name 'skill_trust_header_line'`.

- [ ] **Step 3: Implement the state helper**

In `tldw_chatbook/Library/library_skills_state.py`:

```python
def skill_trust_header_line(posture: str, blocked_count: int) -> tuple[str, str] | None:
    """Return (copy, action_id) for the Skills-list trust header, or None to hide.

    action_id: "setup" | "resetup" | "retry" | "unlock" | "review" | "".
    """
    if posture == "needs_setup":
        return ("Skill trust isn't set up — set it up to review and use skills.", "setup")
    if posture == "needs_resetup":
        return ("Skill trust needs to be set up again after an update.", "resetup")
    if posture == "unavailable":
        return ("Skill trust is temporarily unavailable — try again.", "retry")
    if posture == "locked":
        return ("Skill trust is locked for this session.", "unlock")
    if posture == "ready":
        if blocked_count > 0:
            noun = "skill needs" if blocked_count == 1 else "skills need"
            return (f"{blocked_count} {noun} review before use.", "review")
        return ("Skill trust: ready.", "")
    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_skills_state.py -q -k trust_header_line`
Expected: PASS.

- [ ] **Step 5: Write the failing canvas render test**

Add to `Tests/UI/test_library_skills_canvas.py` (uses the existing `_CanvasHost`/`_two_row_state` helpers in that file):

```python
@pytest.mark.asyncio
async def test_skills_list_renders_trust_header_setup():
    app = _CanvasHost(_two_row_state(), trust_posture="needs_setup")
    async with app.run_test() as pilot:
        header = pilot.app.query_one("#library-skills-trust-header", Static)
        assert "isn't set up" in str(header.renderable)
        action = pilot.app.query_one("#library-skills-trust-action", Button)
        assert action.trust_action == "setup"


@pytest.mark.asyncio
async def test_skills_list_trust_header_hidden_when_ready_and_clean():
    app = _CanvasHost(_two_row_state(), trust_posture="ready")
    async with app.run_test() as pilot:
        # _two_row_state has a blocked row -> "review"; use an all-clean state instead:
        pass  # replaced below


@pytest.mark.asyncio
async def test_skills_list_trust_header_hidden_when_posture_absent():
    app = _CanvasHost(_two_row_state(), trust_posture="")
    async with app.run_test() as pilot:
        assert not pilot.app.query("#library-skills-trust-action")
```

(Replace the middle stubbed test with a clean-state fixture if `_two_row_state` has a blocked row; assert the action button is absent and the header shows "ready" quietly. Check the existing `_two_row_state` in the file and build a no-blocked variant.)

- [ ] **Step 6: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_skills_canvas.py -q -k trust_header`
Expected: FAIL — no `#library-skills-trust-header`; `trust_posture` unknown kwarg.

- [ ] **Step 7: Render the header in the canvas**

In `library_skills_canvas.py`, add `trust_posture: str = ""` to `LibrarySkillsListCanvas.__init__` (store `self.trust_posture`). In `_compose_list`, right after the `Static(f"Skills ({state.count})", ...)` header line, insert:

```python
from tldw_chatbook.Library.library_skills_state import skill_trust_header_line

blocked_count = sum(1 for row in state.rows if getattr(row, "blocked", False))
header = skill_trust_header_line(self.trust_posture, blocked_count)
if header is not None:
    copy, action_id = header
    yield Static(copy, id="library-skills-trust-header", markup=False)
    if action_id:
        button = Button(
            {"setup": "Set up skill trust", "resetup": "Set up skill trust",
             "retry": "Retry", "unlock": "Unlock", "review": "Review"}[action_id],
            id="library-skills-trust-action",
            classes="library-canvas-action",
            compact=True,
        )
        button.trust_action = action_id  # read by the screen handler
        yield button
```

(Import `skill_trust_header_line` at module top instead of inline. Add `trust_action: str = ""` as a class attribute on a small `Button` subclass or set it dynamically as shown — match the existing pattern used for `skill_name` on the row buttons in this same file.)

- [ ] **Step 8: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_skills_canvas.py -q -k trust_header`
Expected: PASS.

- [ ] **Step 9: Full skills-canvas + state regression + commit**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_skills_canvas.py Tests/Library/test_library_skills_state.py -q`
Expected: PASS.

```bash
git add tldw_chatbook/Library/library_skills_state.py tldw_chatbook/Widgets/Library/library_skills_canvas.py Tests/UI/test_library_skills_canvas.py Tests/Library/test_library_skills_state.py
git commit -m "feat(skills): adaptive trust header on the Skills list (Spec 1 T4)"
```

---

### Task 5: Wire header actions, reset confirmation, and editor recovery in the screen

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (compute + pass `trust_posture` to the list canvas; header action handler; reset confirm state + handlers; Set-up = reset-then-bootstrap when a stale manifest exists; editor `manifest_error`/`locked` panel → Reset action)
- Modify: `tldw_chatbook/Widgets/Library/library_skills_canvas.py` (reset-confirm inline row; trust-panel remediation copy → Reset button)
- Test: `Tests/Skills/test_skills_library_flow.py` (extend — real-service harness), `Tests/UI/test_library_skills_canvas.py` (extend)

**Interfaces:**
- Consumes: `trust_posture()` (T3); `skill_trust_header_line` (T4); `reset_trust()` (T3); existing `_bootstrap_library_skill_trust`, `_request_library_skill_trust_bootstrap_passphrase`, `_call_library_skill_trust_service`, `_unlock_library_skill_trust` in `library_screen.py`.
- Produces: `LibraryScreen.handle_library_skills_trust_action` (`@on(Button.Pressed, "#library-skills-trust-action")`) dispatching on `event.button.trust_action`.
- Produces: `LibraryScreen._reset_library_skill_trust()` (worker) — confirm-gated; runs `reset_trust()` off-thread; for the "setup"/"resetup" flow, if `has_manifest()`, reset first then bootstrap.

- [ ] **Step 1: Write the failing harness test**

Add to `Tests/Skills/test_skills_library_flow.py` (uses `_real_skills_scope_service`, `_real_trust_service`, `_open_skill_editor`, `LibraryHarness`, `_wait_for_library_shell`):

```python
@pytest.mark.asyncio
async def test_orphaned_manifest_is_one_click_resetup(tmp_path):
    """Manifest present but scoped marker cleanly absent -> header offers a single
    Set-up that reset-then-bootstraps, never the locked/unlock/error detour."""
    trust = _real_uninitialized_trust_service(tmp_path)
    local_service, service = _real_skills_scope_service(tmp_path, trust_service=trust)
    await local_service.create_skill(
        name="demo", content=_skill_content(title="D", description="d"),
    )
    # Bootstrap, then simulate the upgrade: clear ONLY the marker, leaving the manifest.
    trust.bootstrap_trust("pw", salt=b"7" * 32)
    trust.trust_store.marker_store.clear()
    trust._keys = None  # fresh session
    assert trust.trust_posture() == "needs_resetup"

    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    app.local_skill_trust_service = trust
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        await pilot.pause(); await pilot.pause()
        header = screen.query_one("#library-skills-trust-header", Static)
        assert "again after an update" in str(header.renderable)
        action = screen.query_one("#library-skills-trust-action", Button)
        assert action.trust_action == "resetup"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skills_library_flow.py -q -k orphaned_manifest`
Expected: FAIL — `#library-skills-trust-header` not found (screen doesn't pass `trust_posture` yet).

- [ ] **Step 3: Compute + pass the posture to the canvas**

In `library_screen.py`, add a helper that reads the posture off-thread and caches it on the screen (mirror the existing `_build_local_source_records` off-thread pattern; do NOT call the trust service on the compose thread). Store `self._library_skills_trust_posture: str = ""`, refresh it whenever the skills snapshot refreshes, and pass `trust_posture=self._library_skills_trust_posture` in **both** `LibrarySkillsListCanvas(...)` list-view constructions (~line 3713). Guard: when `local_skill_trust_service` is missing or the scope service is in server/allow-untrusted mode, leave posture `""` (header hidden).

```python
def _refresh_library_skills_trust_posture(self) -> None:
    service = getattr(self.app_instance, "local_skill_trust_service", None)
    posture_fn = getattr(service, "trust_posture", None)
    if not callable(posture_fn):
        self._library_skills_trust_posture = ""
        return
    self.run_worker(
        self._load_library_skills_trust_posture(posture_fn),
        exclusive=True, group="library_skills_trust_posture", exit_on_error=False,
    )

async def _load_library_skills_trust_posture(self, posture_fn) -> None:
    try:
        posture = await asyncio.to_thread(posture_fn)
    except Exception:
        posture = ""
    self._library_skills_trust_posture = posture if isinstance(posture, str) else ""
    if self.is_mounted and self._library_selected_row_id == LIBRARY_ROW_BROWSE_SKILLS:
        self.refresh(recompose=True)
```

Call `_refresh_library_skills_trust_posture()` from wherever the skills snapshot is refreshed and on entering the skills row.

- [ ] **Step 4: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skills_library_flow.py -q -k orphaned_manifest`
Expected: PASS.

- [ ] **Step 5: Write the failing action-dispatch + reset-confirm tests**

Add to `Tests/UI/test_library_skills_canvas.py` (direct-method style with `SimpleNamespace`, matching the existing handler tests in that file):

```python
@pytest.mark.asyncio
async def test_trust_action_setup_dispatches_bootstrap():
    calls = []
    fake = SimpleNamespace(
        _library_selected_row_id=LIBRARY_ROW_BROWSE_SKILLS,
        _begin_library_skill_trust_setup=lambda: calls.append("setup"),
        _unlock_library_skill_trust=lambda: None,
        run_worker=lambda coro, **k: None,
    )
    button = SimpleNamespace(trust_action="setup")
    LibraryScreen.handle_library_skills_trust_action(fake, SimpleNamespace(stop=lambda: None, button=button))
    assert calls == ["setup"]


def test_reset_requires_confirmation():
    fake = SimpleNamespace(
        _library_skill_trust_confirming_reset=False,
        refresh=lambda recompose=False: None,
        is_mounted=True,
    )
    LibraryScreen.handle_library_skills_trust_reset_request(fake, SimpleNamespace(stop=lambda: None))
    assert fake._library_skill_trust_confirming_reset is True
```

- [ ] **Step 6: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_skills_canvas.py -q -k "trust_action or requires_confirmation"`
Expected: FAIL — handlers don't exist.

- [ ] **Step 7: Implement the header action dispatch, reset-confirm, and reset-then-bootstrap**

In `library_screen.py`:

```python
@on(Button.Pressed, "#library-skills-trust-action")
def handle_library_skills_trust_action(self, event: Button.Pressed) -> None:
    event.stop()
    action = getattr(event.button, "trust_action", "")
    if action in ("setup", "resetup"):
        self._begin_library_skill_trust_setup()
    elif action == "unlock":
        self.run_worker(self._unlock_library_skill_trust(), exclusive=True,
                        group="library_skill_trust")
    elif action == "retry":
        self._refresh_library_skills_trust_posture()
    elif action == "review":
        self._open_first_blocked_skill()

def _begin_library_skill_trust_setup(self) -> None:
    self.run_worker(self._setup_library_skill_trust(), exclusive=True,
                    group="library_skill_trust")

async def _setup_library_skill_trust(self) -> None:
    """Set up trust; reset-then-bootstrap when a stale/orphaned manifest exists."""
    service = getattr(self.app_instance, "local_skill_trust_service", None)
    if service is None:
        return
    passphrase = await self._request_library_skill_trust_bootstrap_passphrase()
    if passphrase is None:
        return
    if getattr(service, "trust_store", None) and service.trust_store.has_manifest():
        await self._call_library_skill_trust_service("reset_trust")
    _, ok = await self._call_library_skill_trust_service("bootstrap_trust", passphrase)
    if ok:
        self._refresh_library_skills_trust_posture()
        self._refresh_local_source_snapshot()

@on(Button.Pressed, "#library-skills-trust-reset")
def handle_library_skills_trust_reset_request(self, event: Button.Pressed) -> None:
    event.stop()
    self._library_skill_trust_confirming_reset = True
    if self.is_mounted:
        self.refresh(recompose=True)

@on(Button.Pressed, "#library-skills-trust-reset-cancel")
def handle_library_skills_trust_reset_cancel(self, event: Button.Pressed) -> None:
    event.stop()
    self._library_skill_trust_confirming_reset = False
    if self.is_mounted:
        self.refresh(recompose=True)

@on(Button.Pressed, "#library-skills-trust-reset-confirm")
def handle_library_skills_trust_reset_confirm(self, event: Button.Pressed) -> None:
    event.stop()
    self._library_skill_trust_confirming_reset = False
    self.run_worker(self._do_library_skill_trust_reset(), exclusive=True,
                    group="library_skill_trust")

async def _do_library_skill_trust_reset(self) -> None:
    await self._call_library_skill_trust_service("reset_trust")
    self._refresh_library_skills_trust_posture()
    self._refresh_local_source_snapshot()
    if self.is_mounted:
        self.refresh(recompose=True)
```

Add `self._library_skill_trust_confirming_reset: bool = False` to `LibraryScreen.__init__`. Implement `_open_first_blocked_skill()` reusing the existing row-open flow (find the first row with `blocked` true in `_build_library_skills_state()`, open its editor).

In the canvas, render the reset-confirm inline row (when `confirming_reset`) and the standalone **Reset** button. Pass `confirming_reset=self._library_skill_trust_confirming_reset` and surface a Reset button whenever the posture is `needs_resetup`, `locked`, or on the `manifest_error` editor panel. The confirm copy: *"Reset skill trust? Every skill will need re-approval. Your skills are not deleted."* with `#library-skills-trust-reset-confirm` / `#library-skills-trust-reset-cancel`.

Replace the editor trust-panel `manifest_error` remediation (the task-421 `skill_trust_remediation_copy` "delete the trust store" text) with a short line + the in-UI Reset button (`#library-skills-trust-reset`).

- [ ] **Step 8: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_skills_canvas.py -q -k "trust_action or requires_confirmation"`
Expected: PASS.

- [ ] **Step 9: Full regression + commit**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_skills_canvas.py Tests/Skills/ Tests/Library/test_library_skills_state.py -q`
Expected: PASS. (Also update any existing task-421 remediation-copy test whose expectation changed — search for `skill_trust_remediation_copy` in tests and adjust to the new Reset-based copy.)

```bash
git add tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/library_skills_canvas.py Tests/UI/test_library_skills_canvas.py Tests/Skills/test_skills_library_flow.py
git commit -m "feat(skills): list-header trust actions + confirm-gated reset + editor recovery (Spec 1 T5)"
```

---

### Task 6: Minor — palette entry disambiguation (finding #6)

**Files:**
- Modify: `tldw_chatbook/app.py` (`TabNavigationProvider` — the generic Library command's help text vs the `LIBRARY_SUBROUTE_COMMANDS` "Library — Skills" entry)
- Test: `Tests/UI/test_command_palette_shell_routes.py` (extend)

**Interfaces:**
- Consumes: existing `TabNavigationProvider.LIBRARY_SUBROUTE_COMMANDS` and the generic Library destination command.

- [ ] **Step 1: Write the failing test**

Add to `Tests/UI/test_command_palette_shell_routes.py`:

```python
def test_library_and_skills_palette_entries_are_distinguishable():
    from tldw_chatbook.app import TabNavigationProvider

    subroutes = TabNavigationProvider.LIBRARY_SUBROUTE_COMMANDS
    skills = next(item for item in subroutes if item[0] == "skills")
    # The skills deep-link command text must not be a bare "Library" that
    # collides with the generic destination command.
    assert "Skills" in skills[1]
    assert skills[1] != "Library"
```

- [ ] **Step 2: Run to verify it fails/passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_command_palette_shell_routes.py -q -k distinguishable`
Expected: likely PASS already (the deep-link text is "Library — Skills"). If it passes, the finding is already covered by Spec-1 predecessor work; **do nothing further** (YAGNI) beyond confirming the generic Library command's help text doesn't read identically. If it fails, make the minimal copy tweak so the two are distinct.

- [ ] **Step 3: Commit (only if a change was needed)**

```bash
git add tldw_chatbook/app.py Tests/UI/test_command_palette_shell_routes.py
git commit -m "polish(skills): distinguish Library vs Library-Skills palette entries (Spec 1 T6)"
```

---

### Task 7: Runtime verification checkpoint (live app)

**Files:** none (manual verification against the running TUI).

Use the `verify` skill recipe. Run against a **fresh** scratch profile so trust state is genuinely first-run, and against a **simulated orphaned** profile.

- [ ] **Step 1: Fresh-profile happy path.** Launch with a scratch `TLDW_CONFIG_PATH` and a never-used `users_name`. Import a single superpowers skill (the `test-driven-development` folder), confirm the list header reads *"Skill trust isn't set up — Set up skill trust"*, click it, set a passphrase, and confirm the skill goes to ✓ trusted and the header goes quiet ("ready"). Expected: one-click setup, no `manifest cannot be verified`.

- [ ] **Step 2: Orphaned-manifest path (the finding that bit us).** In the same profile, bootstrap trust, then (via a short REPL against the profile's trust service, keyring disabled → file marker) `marker_store.clear()` to simulate the upgrade. Relaunch. Confirm the header reads *"Skill trust needs to be set up again after an update."* with a **Set up skill trust** action, that clicking it (passphrase) recovers to ready in one step, and that no `locked → unlock → manifest_error` detour appears.

- [ ] **Step 3: Reset recovery + forgot-passphrase.** From a `locked` state, confirm the header/editor offers **Reset** (with the confirmation naming "skills are not deleted"), and that Reset returns to first-run.

- [ ] **Step 4:** Record findings; if any behavior diverges from the design, file a fix task and address before the PR. Delete the scratch profile data afterward.

---

## Self-Review

**Spec coverage:**
- Component 1 (trust isolation) → Task 1 (scoping) + Task 3 (`trust_posture` orphaned detection, safety None-vs-raise). ✓
- Component 2 (recovery / `reset_trust`, store primitives, reset-then-bootstrap, no dead-end, editor Reset) → Task 2 (primitives) + Task 3 (`reset_trust`) + Task 5 (reset-then-bootstrap, editor panel Reset, confirm). ✓
- Component 3 (adaptive header, all posture rows, hidden-when-unavailable/zero, quiet-when-ready, forgot-passphrase Reset from locked) → Task 4 (states + render) + Task 5 (locked→Reset, wiring). ✓
- Component 4 (palette) → Task 6. ✓
- Testing section → per-task tests + Task 7 runtime checkpoint. ✓
- Security fix (key-cache scoping) → Task 1. ✓

**Placeholder scan:** No "TBD"/"handle edge cases". A few steps say "verify the real current body before editing" / "match the existing pattern" — these are deliberate guards against verbatim-replacing large existing methods, and each still shows the exact new code to add. The Task 4 middle test is explicitly flagged to be finalized against `_two_row_state`'s real shape (blocked vs clean) — resolve when writing it.

**Type consistency:** `trust_posture()` returns the 6 strings used by `skill_trust_header_line` and the header table. `action_id`/`trust_action` values (`setup`/`resetup`/`retry`/`unlock`/`review`) are consistent across Task 4 (produced) and Task 5 (consumed). `account_scope` naming is consistent across Task 1 stores + builders + app wiring. `clear()`/`delete_manifest()` names match between Task 2 (defined) and Task 3 (consumed).

## Execution Handoff

Two execution options:
1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks.
2. **Inline Execution** — execute in this session with checkpoints.
