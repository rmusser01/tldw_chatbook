# Local Skill Trust Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Protect Chatbook-managed local skills from offline tampering by adding passphrase-rooted trust bootstrap, authenticated baseline verification, logical quarantine, reviewed re-trust, and visible recovery states.

**Architecture:** Add focused trust modules beside `LocalSkillsService`: models/errors, crypto helpers, scanner, authenticated store, and orchestration service. Wire the service into `LocalSkillsService`, Skills UI, Settings privacy posture, runtime policy, and tests without touching Codex runtime skills or server-managed skills.

**Tech Stack:** Python 3.11+, dataclasses, pathlib, json, hashlib/hmac, Cryptodome AES-GCM/scrypt, existing `keyring` secure-backend checks, pytest/pytest-asyncio, Textual mounted UI tests, Backlog.md, ADRs.

---

## Source Documents

- Backlog task: `backlog/tasks/task-137 - Add-local-skill-trust-integrity-controls.md`
- ADR: `backlog/decisions/009-local-skill-trust-boundary.md`
- Spec: `Docs/superpowers/specs/2026-06-25-local-skill-trust-integrity-design.md`

ADR required: yes

ADR path: `backlog/decisions/009-local-skill-trust-boundary.md`

Reason: This changes a security boundary, trust root, local storage semantics, runtime policy IDs, and the `LocalSkillsService` contract. ADR-009 is already accepted and must remain linked from the task and implementation notes.

## Pre-Execution Review Corrections

These corrections are mandatory and supersede any lower-level code snippet that appears to conflict:

1. Production rollback protection must use a secure OS keyring-backed generation marker store. File-backed marker storage is allowed only for tests or an explicit reduced-rollback-protection mode, and that mode must be reflected in Settings posture.
2. A manifest with a missing, unavailable, stale, or mismatched marker is a global `quarantined_manifest_error`/`rollback_marker_unavailable` state. It must not be accepted as trusted.
3. `status_for_skill()` and list/context paths must not raise just because trust is locked or the manifest is unverifiable. They must return visible blocked trust status. `ensure_skill_trusted()` is the use-time raise path.
4. Chatbook-approved create/update/import flows may rebaseline only after explicit approval text. Out-of-band file writes remain the tamper signal and must quarantine.
5. Review approval uses canonical JSON digests of captured fingerprints, not Python `str(...)`, before updating the baseline.
6. Skills UI recovery actions must be functional, not only rendered: bootstrap, unlock/retry, review diff, and trust reviewed version must call the trust service through app-owned handlers and refresh state.
7. `SkillTrustService` must expose Settings posture fields used by the UI: `overall_status()`, `keyring_convenience_enabled`, and `reduced_rollback_protection`.
8. Production unlock must persist and load a non-secret KDF salt from trust metadata. Tests may inject deterministic salts, but app/bootstrap flows must not require callers to remember a salt.
9. Optional keyring convenience must store derived trust material or a wrapped trust root in a secure keyring, never the passphrase, and must be disabled by default until the user explicitly enables it.

## File Structure

- Create `tldw_chatbook/Skills_Interop/skill_trust_models.py`: trust status constants, reason codes, dataclasses, and `SkillTrustBlockedError`.
- Create `tldw_chatbook/Skills_Interop/skill_trust_crypto.py`: key derivation, canonical JSON, HMAC, AES-GCM encryption/decryption, and hash helpers.
- Create `tldw_chatbook/Skills_Interop/skill_trust_scanner.py`: deterministic skill-directory scanning, file fingerprinting, unsafe path detection, text snapshot capture.
- Create `tldw_chatbook/Skills_Interop/skill_trust_store.py`: manifest/snapshot persistence, secure keyring generation marker store, explicit reduced-protection file marker for tests/recovery, keyring convenience helpers, authenticated audit records.
- Create `tldw_chatbook/Skills_Interop/skill_trust_service.py`: bootstrap, unlock, status classification, review capture, approval, key rotation, Settings posture, and use-time enforcement.
- Modify `tldw_chatbook/Skills_Interop/local_skills_service.py`: inject trust service, add trust fields to list/detail/context responses, block execution when trust fails, and atomically re-trust only explicitly approved Chatbook mutations.
- Modify `tldw_chatbook/Skills_Interop/skills_scope_service.py`: preserve trust fields during local response normalization. Recovery actions use `app.local_skill_trust_service` directly in v1.
- Modify `tldw_chatbook/Skills_Interop/__init__.py`: export trust service/types.
- Modify `tldw_chatbook/runtime_policy/registry.py`: add local `skills.trust.*.local` policy actions.
- Modify `tldw_chatbook/app.py`: construct and wire `SkillTrustService` into `LocalSkillsService`.
- Modify `tldw_chatbook/UI/Screens/skills_screen.py`: render trust state, block attach for trust-blocked skills, add unlock/bootstrap/review/approve/reject recovery controls.
- Modify `tldw_chatbook/UI/Screens/settings_privacy_security.py`: include redacted skill trust posture and keyring convenience/reduced rollback posture rows.
- Modify `backlog/tasks/task-137 - Add-local-skill-trust-integrity-controls.md`: add plan link, ADR link, and closeout notes during final task completion.
- Add tests under `Tests/Skills/`, `Tests/RuntimePolicy/`, and `Tests/UI/`.

## Task 1: Register Trust Policy Actions And ADR/Task Links

**Files:**
- Modify: `tldw_chatbook/runtime_policy/registry.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_core.py`
- Modify: `Tests/RuntimePolicy/test_domain_edge_contracts.py`
- Modify: `backlog/tasks/task-137 - Add-local-skill-trust-integrity-controls.md`

- [ ] **Step 1: Put the Backlog task in progress and add plan metadata**

Run:

```bash
backlog task edit 137 -s "In Progress" --plan "ADR required: yes
ADR path: backlog/decisions/009-local-skill-trust-boundary.md
Reason: Local skill trust changes a security boundary, trust root, local storage semantics, runtime policy IDs, and the LocalSkillsService contract.

1. Register local skill trust runtime-policy actions.
2. Add trust models, crypto, scanner, store, and service substrate.
3. Integrate trust checks into LocalSkillsService and app wiring.
4. Add Skills and Settings trust posture UI.
5. Run focused trust, Skills, runtime-policy, and UI tests.

Superpowers plan: Docs/superpowers/plans/2026-06-25-local-skill-trust-integrity.md"
```

Expected: `TASK-137` status becomes `In Progress` and the task contains an `## Implementation Plan` section with the ADR check.

- [ ] **Step 2: Add failing runtime-policy tests**

In `Tests/RuntimePolicy/test_domain_edge_contracts.py`, extend `test_skills_and_kanban_have_local_policy_actions()` with these action IDs:

```python
    for action_id in [
        "skills.list.local",
        "skills.execute.launch.local",
        "skills.trust.unlock.local",
        "skills.trust.review.local",
        "skills.trust.approve.local",
        "skills.trust.reject.local",
        "skills.trust.rebootstrap.local",
        "skills.trust.rotate_key.local",
        "skills.trust.audit.local",
        "kanban.boards.list.local",
        "kanban.cards.create.local",
        "kanban.card_links.delete.local",
    ]:
        entry = CAPABILITY_REGISTRY[action_id]
        assert entry.required_source == "local"
        assert entry.authority_owner == "local"
```

In `Tests/RuntimePolicy/test_runtime_policy_core.py`, extend the `"server_skills"` expected action block with:

```text
        skills.trust.approve.local
        skills.trust.audit.local
        skills.trust.rebootstrap.local
        skills.trust.reject.local
        skills.trust.review.local
        skills.trust.rotate_key.local
        skills.trust.unlock.local
```

- [ ] **Step 3: Run the failing policy tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/RuntimePolicy/test_domain_edge_contracts.py::test_skills_and_kanban_have_local_policy_actions Tests/RuntimePolicy/test_runtime_policy_core.py::test_capability_registry_contains_expected_actions --tb=short
```

Expected: fail with missing `skills.trust.*.local` action IDs.

- [ ] **Step 4: Implement registry actions**

In `tldw_chatbook/runtime_policy/registry.py`, add action constants near the existing action constants:

```python
UNLOCK = _action("unlock", "launch")
REVIEW = _action("review", "detail")
REJECT = _action("reject", "update")
REBOOTSTRAP = _action("rebootstrap", "update")
ROTATE_KEY = _action("rotate_key", "update")
AUDIT = _action("audit", "detail")
```

In the `"server_skills"` capability resources, add the local-only trust resource:

```python
            _resource(
                "skills.trust",
                actions=(UNLOCK, REVIEW, APPROVE, REJECT, REBOOTSTRAP, ROTATE_KEY, AUDIT),
                sources=LOCAL_ONLY_SOURCES,
            ),
```

- [ ] **Step 5: Re-run policy tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/RuntimePolicy/test_domain_edge_contracts.py::test_skills_and_kanban_have_local_policy_actions Tests/RuntimePolicy/test_runtime_policy_core.py::test_capability_registry_contains_expected_actions --tb=short
```

Expected: pass.

- [ ] **Step 6: Commit policy and task metadata**

Run:

```bash
git add tldw_chatbook/runtime_policy/registry.py Tests/RuntimePolicy/test_domain_edge_contracts.py Tests/RuntimePolicy/test_runtime_policy_core.py "backlog/tasks/task-137 - Add-local-skill-trust-integrity-controls.md"
git commit -m "feat: register local skill trust policy actions"
```

Expected: commit succeeds.

## Task 2: Add Trust Models And Crypto Helpers

**Files:**
- Create: `tldw_chatbook/Skills_Interop/skill_trust_models.py`
- Create: `tldw_chatbook/Skills_Interop/skill_trust_crypto.py`
- Modify: `tldw_chatbook/Skills_Interop/__init__.py`
- Test: `Tests/Skills/test_skill_trust_crypto.py`

- [ ] **Step 1: Write failing crypto/model tests**

Create `Tests/Skills/test_skill_trust_crypto.py`:

```python
import pytest

from tldw_chatbook.Skills_Interop.skill_trust_crypto import (
    decrypt_json_blob,
    derive_skill_trust_keys,
    encrypt_json_blob,
    manifest_mac,
)
from tldw_chatbook.Skills_Interop.skill_trust_models import (
    SkillTrustBlockedError,
    SkillTrustStatus,
    TRUST_STATUS_TRUSTED,
)


def test_skill_trust_key_derivation_separates_key_purposes():
    keys = derive_skill_trust_keys("correct horse battery staple", salt=b"0" * 32)

    assert len(keys.manifest_mac_key) == 32
    assert len(keys.snapshot_key) == 32
    assert len(keys.audit_mac_key) == 32
    assert keys.manifest_mac_key != keys.snapshot_key
    assert keys.snapshot_key != keys.audit_mac_key


def test_manifest_mac_rejects_tampered_manifest_payload():
    keys = derive_skill_trust_keys("passphrase", salt=b"1" * 32)
    manifest = {"version": 1, "generation": 1, "skills": {"demo": {"status": "trusted"}}}
    tag = manifest_mac(manifest, keys.manifest_mac_key)

    tampered = {"version": 1, "generation": 2, "skills": {"demo": {"status": "trusted"}}}

    assert manifest_mac(manifest, keys.manifest_mac_key) == tag
    assert manifest_mac(tampered, keys.manifest_mac_key) != tag


def test_snapshot_encryption_round_trips_and_authenticates_associated_data():
    keys = derive_skill_trust_keys("passphrase", salt=b"2" * 32)
    payload = {"files": {"SKILL.md": "# Demo\nTrusted"}}
    encrypted = encrypt_json_blob(payload, keys.snapshot_key, associated_data=b"demo:generation:1")

    assert decrypt_json_blob(encrypted, keys.snapshot_key, associated_data=b"demo:generation:1") == payload

    with pytest.raises(ValueError, match="authentication failed"):
        decrypt_json_blob(encrypted, keys.snapshot_key, associated_data=b"demo:generation:2")


def test_trust_blocked_error_carries_safe_fields():
    status = SkillTrustStatus(
        skill_name="demo",
        trust_status=TRUST_STATUS_TRUSTED,
        trust_reason_code=None,
        trust_blocked=False,
        changed_files=(),
        manifest_generation=3,
        last_verified_at="2026-06-25T00:00:00+00:00",
    )

    error = SkillTrustBlockedError(
        skill_name="demo",
        reason_code="skill_modified",
        trust_status="quarantined_modified",
        changed_files=("SKILL.md",),
    )

    assert status.skill_name == "demo"
    assert str(error) == "Local skill demo is trust-blocked: skill_modified"
    assert error.changed_files == ("SKILL.md",)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_skill_trust_crypto.py --tb=short
```

Expected: fail with `ModuleNotFoundError` for `skill_trust_crypto`.

- [ ] **Step 3: Add trust models**

Create `tldw_chatbook/Skills_Interop/skill_trust_models.py`:

```python
"""Local skill trust state models and exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TRUST_STATUS_TRUSTED = "trusted"
TRUST_STATUS_UNINITIALIZED = "trust_uninitialized"
TRUST_STATUS_LOCKED = "trust_locked"
TRUST_STATUS_QUARANTINED_MODIFIED = "quarantined_modified"
TRUST_STATUS_QUARANTINED_ADDED = "quarantined_added"
TRUST_STATUS_QUARANTINED_DELETED = "quarantined_deleted"
TRUST_STATUS_QUARANTINED_MANIFEST_ERROR = "quarantined_manifest_error"
TRUST_STATUS_QUARANTINED_UNSUPPORTED_PATH = "quarantined_unsupported_path"

TRUST_REASON_SKILL_MODIFIED = "skill_modified"
TRUST_REASON_SKILL_ADDED = "skill_added"
TRUST_REASON_SKILL_DELETED = "skill_deleted"
TRUST_REASON_UNSUPPORTED_PATH = "unsupported_path"
TRUST_REASON_TRUST_LOCKED = "trust_locked"
TRUST_REASON_TRUST_UNINITIALIZED = "trust_uninitialized"
TRUST_REASON_MANIFEST_INVALID = "manifest_invalid"
TRUST_REASON_ROLLBACK_MARKER_UNAVAILABLE = "rollback_marker_unavailable"
TRUST_REASON_SNAPSHOT_MISMATCH = "snapshot_mismatch"
TRUST_REASON_STORE_WRITE_FAILED = "trust_store_write_failed"
TRUST_REASON_HISTORY_UNRECOVERABLE = "trust_history_unrecoverable"


@dataclass(frozen=True, slots=True)
class SkillFileFingerprint:
    relative_path: str
    file_type: str
    byte_length: int
    sha256: str

    def as_manifest_entry(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "file_type": self.file_type,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SkillDirectorySnapshot:
    skill_name: str
    fingerprints: tuple[SkillFileFingerprint, ...]
    text_files: dict[str, str]
    unsupported_paths: tuple[str, ...] = ()

    @property
    def fingerprint_map(self) -> dict[str, SkillFileFingerprint]:
        return {item.relative_path: item for item in self.fingerprints}


@dataclass(frozen=True, slots=True)
class SkillTrustStatus:
    skill_name: str
    trust_status: str
    trust_reason_code: str | None
    trust_blocked: bool
    changed_files: tuple[str, ...]
    manifest_generation: int | None
    last_verified_at: str | None

    def response_fields(self) -> dict[str, Any]:
        return {
            "trust_status": self.trust_status,
            "trust_reason_code": self.trust_reason_code,
            "trust_blocked": self.trust_blocked,
            "trust_changed_files": list(self.changed_files),
            "trust_manifest_generation": self.manifest_generation,
            "trust_last_verified_at": self.last_verified_at,
        }


class SkillTrustBlockedError(RuntimeError):
    """Raised when a trust-blocked local skill is staged or executed."""

    def __init__(
        self,
        *,
        skill_name: str,
        reason_code: str,
        trust_status: str,
        changed_files: tuple[str, ...] = (),
    ) -> None:
        super().__init__(f"Local skill {skill_name} is trust-blocked: {reason_code}")
        self.skill_name = skill_name
        self.reason_code = reason_code
        self.trust_status = trust_status
        self.changed_files = changed_files
```

- [ ] **Step 4: Add crypto helpers**

Create `tldw_chatbook/Skills_Interop/skill_trust_crypto.py`:

```python
"""Cryptographic helpers for local skill trust state."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import Any

from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import scrypt


SKILL_TRUST_KDF_N = 16384
SKILL_TRUST_KDF_R = 8
SKILL_TRUST_KDF_P = 1
SKILL_TRUST_KEY_SIZE = 32
SKILL_TRUST_NONCE_SIZE = 12


@dataclass(frozen=True, slots=True)
class SkillTrustKeys:
    manifest_mac_key: bytes
    snapshot_key: bytes
    audit_mac_key: bytes
    wrapped_root_key: bytes


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive_skill_trust_keys(passphrase: str, *, salt: bytes) -> SkillTrustKeys:
    if len(salt) != 32:
        raise ValueError("skill trust salt must be 32 bytes")
    root = scrypt(
        passphrase.encode("utf-8"),
        salt,
        key_len=SKILL_TRUST_KEY_SIZE,
        N=SKILL_TRUST_KDF_N,
        r=SKILL_TRUST_KDF_R,
        p=SKILL_TRUST_KDF_P,
    )
    manifest_mac_key = hmac.new(root, b"tldw-chatbook-skill-trust-manifest-v1", hashlib.sha256).digest()
    snapshot_key = hmac.new(root, b"tldw-chatbook-skill-trust-snapshot-v1", hashlib.sha256).digest()
    audit_mac_key = hmac.new(root, b"tldw-chatbook-skill-trust-audit-v1", hashlib.sha256).digest()
    wrapped_root_key = hmac.new(root, b"tldw-chatbook-skill-trust-wrapped-root-v1", hashlib.sha256).digest()
    return SkillTrustKeys(
        manifest_mac_key=manifest_mac_key,
        snapshot_key=snapshot_key,
        audit_mac_key=audit_mac_key,
        wrapped_root_key=wrapped_root_key,
    )


def manifest_mac(manifest_payload: dict[str, Any], key: bytes) -> str:
    return hmac.new(key, canonical_json(manifest_payload), hashlib.sha256).hexdigest()


def encrypt_json_blob(payload: dict[str, Any], key: bytes, *, associated_data: bytes) -> dict[str, str]:
    nonce = os.urandom(SKILL_TRUST_NONCE_SIZE)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(associated_data)
    ciphertext, tag = cipher.encrypt_and_digest(canonical_json(payload))
    return {
        "alg": "AES-256-GCM",
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "tag": base64.b64encode(tag).decode("ascii"),
    }


def decrypt_json_blob(blob: dict[str, str], key: bytes, *, associated_data: bytes) -> dict[str, Any]:
    try:
        nonce = base64.b64decode(blob["nonce"])
        ciphertext = base64.b64decode(blob["ciphertext"])
        tag = base64.b64decode(blob["tag"])
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        cipher.update(associated_data)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        payload = json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise ValueError("snapshot authentication failed") from exc
    if not isinstance(payload, dict):
        raise ValueError("snapshot payload must be an object")
    return payload
```

- [ ] **Step 5: Export trust types**

Update `tldw_chatbook/Skills_Interop/__init__.py`:

```python
"""Local and server SKILL.md interoperability services."""

from .local_skills_service import LocalSkillsService
from .server_skills_service import ServerSkillsService
from .skill_trust_models import SkillTrustBlockedError, SkillTrustStatus
from .skills_scope_service import SkillsBackend, SkillsScopeService

__all__ = [
    "LocalSkillsService",
    "ServerSkillsService",
    "SkillTrustBlockedError",
    "SkillTrustStatus",
    "SkillsBackend",
    "SkillsScopeService",
]
```

- [ ] **Step 6: Run crypto/model tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_skill_trust_crypto.py --tb=short
```

Expected: pass.

- [ ] **Step 7: Commit trust models and crypto**

Run:

```bash
git add tldw_chatbook/Skills_Interop/skill_trust_models.py tldw_chatbook/Skills_Interop/skill_trust_crypto.py tldw_chatbook/Skills_Interop/__init__.py Tests/Skills/test_skill_trust_crypto.py
git commit -m "feat: add local skill trust crypto models"
```

Expected: commit succeeds.

## Task 3: Add Skill Directory Scanner

**Files:**
- Create: `tldw_chatbook/Skills_Interop/skill_trust_scanner.py`
- Test: `Tests/Skills/test_skill_trust_scanner.py`

- [ ] **Step 1: Write failing scanner tests**

Create `Tests/Skills/test_skill_trust_scanner.py`:

```python
from pathlib import Path

from tldw_chatbook.Skills_Interop.skill_trust_scanner import scan_skill_directory


def test_scan_skill_directory_fingerprints_skill_and_supporting_files(tmp_path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Demo\nUse safely.\n", encoding="utf-8")
    (skill_dir / "notes.md").write_text("Trusted notes.\n", encoding="utf-8")

    snapshot = scan_skill_directory("demo", skill_dir)

    assert snapshot.skill_name == "demo"
    assert [item.relative_path for item in snapshot.fingerprints] == ["SKILL.md", "notes.md"]
    assert snapshot.text_files["SKILL.md"] == "# Demo\nUse safely.\n"
    assert snapshot.text_files["notes.md"] == "Trusted notes.\n"
    assert snapshot.unsupported_paths == ()


def test_scan_skill_directory_marks_symlink_nested_tmp_and_binary_paths_unsupported(tmp_path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (skill_dir / "SKILL.md.tmp").write_text("partial", encoding="utf-8")
    (skill_dir / "nested").mkdir()
    (skill_dir / "binary.dat").write_bytes(b"\xff\xfe\x00")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (skill_dir / "linked.md").symlink_to(outside)

    snapshot = scan_skill_directory("demo", skill_dir)

    assert snapshot.unsupported_paths == (
        "SKILL.md.tmp",
        "binary.dat",
        "linked.md",
        "nested",
    )
    assert [item.relative_path for item in snapshot.fingerprints] == ["SKILL.md"]
```

- [ ] **Step 2: Run scanner tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_skill_trust_scanner.py --tb=short
```

Expected: fail with `ModuleNotFoundError` for `skill_trust_scanner`.

- [ ] **Step 3: Implement scanner**

Create `tldw_chatbook/Skills_Interop/skill_trust_scanner.py`:

```python
"""Deterministic local skill directory scanning for trust verification."""

from __future__ import annotations

from pathlib import Path

from ..tldw_api.skills_schemas import SUPPORTING_FILE_NAME_PATTERN
from .skill_trust_crypto import sha256_hex
from .skill_trust_models import SkillDirectorySnapshot, SkillFileFingerprint


_SKILL_FILENAME = "SKILL.md"
_TEMP_SUFFIXES = (".tmp", ".swp", ".part")


def _is_safe_skill_file_name(filename: str) -> bool:
    if filename == _SKILL_FILENAME:
        return True
    if filename.endswith(_TEMP_SUFFIXES):
        return False
    return bool(SUPPORTING_FILE_NAME_PATTERN.match(filename))


def scan_skill_directory(skill_name: str, skill_dir: Path) -> SkillDirectorySnapshot:
    fingerprints: list[SkillFileFingerprint] = []
    text_files: dict[str, str] = {}
    unsupported_paths: list[str] = []

    for path in sorted(skill_dir.iterdir(), key=lambda item: item.name):
        relative_path = path.name
        if path.is_symlink() or path.is_dir() or not _is_safe_skill_file_name(relative_path):
            unsupported_paths.append(relative_path)
            continue
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            unsupported_paths.append(relative_path)
            continue
        fingerprints.append(
            SkillFileFingerprint(
                relative_path=relative_path,
                file_type="skill" if relative_path == _SKILL_FILENAME else "supporting_text",
                byte_length=len(raw),
                sha256=sha256_hex(raw),
            )
        )
        text_files[relative_path] = text

    return SkillDirectorySnapshot(
        skill_name=skill_name,
        fingerprints=tuple(fingerprints),
        text_files=text_files,
        unsupported_paths=tuple(sorted(unsupported_paths)),
    )
```

- [ ] **Step 4: Run scanner tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_skill_trust_scanner.py --tb=short
```

Expected: pass.

- [ ] **Step 5: Commit scanner**

Run:

```bash
git add tldw_chatbook/Skills_Interop/skill_trust_scanner.py Tests/Skills/test_skill_trust_scanner.py
git commit -m "feat: scan local skill files for trust"
```

Expected: commit succeeds.

## Task 4: Add Trust Store, Manifest, Snapshots, Generation Marker, And Audit Records

**Files:**
- Create: `tldw_chatbook/Skills_Interop/skill_trust_store.py`
- Test: `Tests/Skills/test_skill_trust_store.py`

- [ ] **Step 1: Write failing store tests**

Create `Tests/Skills/test_skill_trust_store.py`:

```python
import json

import pytest

from tldw_chatbook.Skills_Interop.skill_trust_crypto import derive_skill_trust_keys
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    KeyringSkillTrustKeyCache,
    KeyringSkillTrustGenerationMarkerStore,
    SkillTrustMarkerUnavailable,
    SkillTrustStore,
)


class FakeSecureKeyring:
    __module__ = "keyring.backends.macOS"
    priority = 1

    def __init__(self):
        self.values = {}

    def get_password(self, service_name, username):
        return self.values.get((service_name, username))

    def set_password(self, service_name, username, password):
        self.values[(service_name, username)] = password


class FakePlaintextKeyring(FakeSecureKeyring):
    __module__ = "keyring.backends.file"


def test_trust_store_round_trips_manifest_snapshot_and_marker(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"3" * 32)
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)

    manifest = {
        "version": 1,
        "generation": 1,
        "skills": {
            "demo": {
                "files": [{"relative_path": "SKILL.md", "file_type": "skill", "byte_length": 6, "sha256": "abc"}],
                "snapshot_id": "demo-1",
                "trusted_at": "2026-06-25T00:00:00+00:00",
            }
        },
        "audit": [],
    }

    store.save_manifest(manifest, keys, salt=b"3" * 32)
    store.save_snapshot("demo-1", {"files": {"SKILL.md": "# Demo"}}, keys, generation=1)

    loaded = store.load_manifest(keys)
    snapshot = store.load_snapshot("demo-1", keys, generation=1)

    assert loaded["generation"] == 1
    assert snapshot == {"files": {"SKILL.md": "# Demo"}}
    assert store.load_salt() == b"3" * 32
    assert marker.load_marker() == {"generation": 1, "manifest_digest": store.manifest_digest(loaded)}


def test_trust_store_rejects_tampered_manifest(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"4" * 32)
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)

    store.save_manifest({"version": 1, "generation": 1, "skills": {}, "audit": []}, keys, salt=b"4" * 32)
    payload = json.loads((tmp_path / "trust" / "skill_trust_manifest.json").read_text(encoding="utf-8"))
    payload["manifest"]["generation"] = 2
    (tmp_path / "trust" / "skill_trust_manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest authentication failed"):
        store.load_manifest(keys)


def test_trust_store_rejects_marker_mismatch(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"5" * 32)
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)

    store.save_manifest({"version": 1, "generation": 3, "skills": {}, "audit": []}, keys, salt=b"5" * 32)
    marker.save_marker(generation=2, manifest_digest="old")

    with pytest.raises(ValueError, match="manifest generation marker mismatch"):
        store.load_manifest(keys)


def test_trust_store_rejects_missing_marker_after_manifest_exists(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"5" * 32)
    marker_path = tmp_path / "marker.json"
    marker = FileSkillTrustGenerationMarkerStore(marker_path)
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)

    store.save_manifest({"version": 1, "generation": 1, "skills": {}, "audit": []}, keys, salt=b"5" * 32)
    marker_path.unlink()

    with pytest.raises(ValueError, match="manifest generation marker mismatch"):
        store.load_manifest(keys)


def test_keyring_generation_marker_store_round_trips_with_secure_backend():
    marker = KeyringSkillTrustGenerationMarkerStore(keyring_backend=FakeSecureKeyring())

    marker.save_marker(generation=3, manifest_digest="digest")

    assert marker.load_marker() == {"generation": 3, "manifest_digest": "digest"}


def test_keyring_generation_marker_store_rejects_insecure_backend():
    with pytest.raises(SkillTrustMarkerUnavailable, match="secure OS-backed"):
        KeyringSkillTrustGenerationMarkerStore(keyring_backend=FakePlaintextKeyring())


def test_keyring_key_cache_round_trips_without_storing_passphrase():
    fake = FakeSecureKeyring()
    keys = derive_skill_trust_keys("passphrase", salt=b"8" * 32)
    cache = KeyringSkillTrustKeyCache(keyring_backend=fake)

    cache.save_keys(keys)
    loaded = cache.load_keys()

    assert loaded == keys
    assert "passphrase" not in "\n".join(fake.values.values())
```

- [ ] **Step 2: Run store tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_skill_trust_store.py --tb=short
```

Expected: fail with `ModuleNotFoundError` for `skill_trust_store`.

- [ ] **Step 3: Implement store**

Create `tldw_chatbook/Skills_Interop/skill_trust_store.py` with these public methods:

```python
"""Persistence for local skill trust manifests, snapshots, markers, and audit events."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..runtime_policy.server_credentials import is_secure_keyring_backend
from .skill_trust_crypto import canonical_json, decrypt_json_blob, encrypt_json_blob, manifest_mac, sha256_hex


_MANIFEST_FILENAME = "skill_trust_manifest.json"
_SNAPSHOTS_DIRNAME = "snapshots"
_DEFAULT_MARKER_SERVICE_NAME = "tldw_chatbook.skill_trust"
_DEFAULT_KEY_CACHE_SERVICE_NAME = "tldw_chatbook.skill_trust.keys"
_MARKER_USERNAME = "local-skills:generation-marker:v1"
_KEY_CACHE_USERNAME = "local-skills:trust-root:v1"


class SkillTrustMarkerUnavailable(RuntimeError):
    reason_code = "rollback_marker_unavailable"


class SkillTrustGenerationMarkerStore(Protocol):
    def load_marker(self) -> dict[str, Any] | None: ...
    def save_marker(self, *, generation: int, manifest_digest: str) -> None: ...


@dataclass(slots=True)
class FileSkillTrustGenerationMarkerStore:
    """Reduced-protection/test marker store. Do not use for default production wiring."""

    marker_path: Path

    def load_marker(self) -> dict[str, Any] | None:
        if not self.marker_path.exists():
            return None
        payload = json.loads(self.marker_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def save_marker(self, *, generation: int, manifest_digest: str) -> None:
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.marker_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps({"generation": generation, "manifest_digest": manifest_digest}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.marker_path)


class UnavailableSkillTrustGenerationMarkerStore:
    def __init__(self, message: str) -> None:
        self.message = message

    def _raise_unavailable(self) -> None:
        raise SkillTrustMarkerUnavailable(self.message)

    def load_marker(self) -> dict[str, Any] | None:
        self._raise_unavailable()

    def save_marker(self, *, generation: int, manifest_digest: str) -> None:
        self._raise_unavailable()


@dataclass(slots=True)
class KeyringSkillTrustGenerationMarkerStore:
    service_name: str = _DEFAULT_MARKER_SERVICE_NAME
    keyring_backend: Any | None = None

    def __post_init__(self) -> None:
        keyring_backend = self.keyring_backend
        if keyring_backend is None:
            import keyring

            keyring_backend = keyring.get_keyring()
        get_keyring = getattr(keyring_backend, "get_keyring", None)
        if callable(get_keyring):
            keyring_backend = get_keyring()
        if not is_secure_keyring_backend(keyring_backend):
            raise SkillTrustMarkerUnavailable("No secure OS-backed generation marker store is available.")
        self.keyring_backend = keyring_backend

    def load_marker(self) -> dict[str, Any] | None:
        payload = self.keyring_backend.get_password(self.service_name, _MARKER_USERNAME)
        if not payload:
            return None
        marker = json.loads(payload)
        return marker if isinstance(marker, dict) else None

    def save_marker(self, *, generation: int, manifest_digest: str) -> None:
        payload = json.dumps({"generation": generation, "manifest_digest": manifest_digest}, sort_keys=True)
        self.keyring_backend.set_password(self.service_name, _MARKER_USERNAME, payload)


def build_default_skill_trust_marker_store(keyring_backend: Any | None = None) -> SkillTrustGenerationMarkerStore:
    try:
        return KeyringSkillTrustGenerationMarkerStore(keyring_backend=keyring_backend)
    except SkillTrustMarkerUnavailable as exc:
        return UnavailableSkillTrustGenerationMarkerStore(str(exc))


@dataclass(slots=True)
class KeyringSkillTrustKeyCache:
    service_name: str = _DEFAULT_KEY_CACHE_SERVICE_NAME
    keyring_backend: Any | None = None

    def __post_init__(self) -> None:
        keyring_backend = self.keyring_backend
        if keyring_backend is None:
            import keyring

            keyring_backend = keyring.get_keyring()
        get_keyring = getattr(keyring_backend, "get_keyring", None)
        if callable(get_keyring):
            keyring_backend = get_keyring()
        if not is_secure_keyring_backend(keyring_backend):
            raise SkillTrustMarkerUnavailable("No secure OS-backed key cache is available.")
        self.keyring_backend = keyring_backend

    def save_keys(self, keys: Any) -> None:
        payload = {
            "version": 1,
            "manifest_mac_key": base64.b64encode(keys.manifest_mac_key).decode("ascii"),
            "snapshot_key": base64.b64encode(keys.snapshot_key).decode("ascii"),
            "audit_mac_key": base64.b64encode(keys.audit_mac_key).decode("ascii"),
            "wrapped_root_key": base64.b64encode(keys.wrapped_root_key).decode("ascii"),
        }
        self.keyring_backend.set_password(self.service_name, _KEY_CACHE_USERNAME, json.dumps(payload, sort_keys=True))

    def load_keys(self) -> Any | None:
        from .skill_trust_crypto import SkillTrustKeys

        payload = self.keyring_backend.get_password(self.service_name, _KEY_CACHE_USERNAME)
        if not payload:
            return None
        data = json.loads(payload)
        return SkillTrustKeys(
            manifest_mac_key=base64.b64decode(data["manifest_mac_key"]),
            snapshot_key=base64.b64decode(data["snapshot_key"]),
            audit_mac_key=base64.b64decode(data["audit_mac_key"]),
            wrapped_root_key=base64.b64decode(data["wrapped_root_key"]),
        )


def build_default_skill_trust_key_cache(keyring_backend: Any | None = None) -> KeyringSkillTrustKeyCache | None:
    try:
        return KeyringSkillTrustKeyCache(keyring_backend=keyring_backend)
    except SkillTrustMarkerUnavailable:
        return None


@dataclass(slots=True)
class SkillTrustStore:
    store_dir: Path
    marker_store: SkillTrustGenerationMarkerStore

    @property
    def manifest_path(self) -> Path:
        return self.store_dir / _MANIFEST_FILENAME

    @property
    def snapshots_dir(self) -> Path:
        return self.store_dir / _SNAPSHOTS_DIRNAME

    def has_manifest(self) -> bool:
        return self.manifest_path.exists()

    def manifest_digest(self, manifest: dict[str, Any]) -> str:
        return sha256_hex(canonical_json(manifest))

    def load_salt(self) -> bytes:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        encoded = payload.get("kdf_salt")
        if not isinstance(encoded, str):
            raise ValueError("skill trust salt missing")
        salt = base64.b64decode(encoded.encode("ascii"))
        if len(salt) != 32:
            raise ValueError("skill trust salt invalid")
        return salt

    def load_manifest(self, keys: Any) -> dict[str, Any]:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest = payload.get("manifest")
        tag = payload.get("mac")
        if not isinstance(manifest, dict) or not isinstance(tag, str):
            raise ValueError("manifest authentication failed")
        if manifest_mac(manifest, keys.manifest_mac_key) != tag:
            raise ValueError("manifest authentication failed")
        marker = self.marker_store.load_marker()
        digest = self.manifest_digest(manifest)
        if marker is None:
            raise ValueError("manifest generation marker mismatch")
        if int(marker.get("generation", -1)) != int(manifest.get("generation", -2)):
            raise ValueError("manifest generation marker mismatch")
        if marker.get("manifest_digest") != digest:
            raise ValueError("manifest generation marker mismatch")
        return manifest

    def save_manifest(self, manifest: dict[str, Any], keys: Any, *, salt: bytes | None = None) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        if salt is None:
            salt = self.load_salt()
        if len(salt) != 32:
            raise ValueError("skill trust salt invalid")
        payload = {
            "kdf_salt": base64.b64encode(salt).decode("ascii"),
            "manifest": manifest,
            "mac": manifest_mac(manifest, keys.manifest_mac_key),
        }
        temp_path = self.manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(self.manifest_path)
        self.marker_store.save_marker(
            generation=int(manifest["generation"]),
            manifest_digest=self.manifest_digest(manifest),
        )

    def save_snapshot(self, snapshot_id: str, payload: dict[str, Any], keys: Any, *, generation: int) -> None:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        associated_data = f"snapshot:{snapshot_id}:generation:{generation}".encode("utf-8")
        encrypted = encrypt_json_blob(payload, keys.snapshot_key, associated_data=associated_data)
        temp_path = (self.snapshots_dir / f"{snapshot_id}.json").with_suffix(".tmp")
        temp_path.write_text(json.dumps(encrypted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(self.snapshots_dir / f"{snapshot_id}.json")

    def load_snapshot(self, snapshot_id: str, keys: Any, *, generation: int) -> dict[str, Any]:
        encrypted = json.loads((self.snapshots_dir / f"{snapshot_id}.json").read_text(encoding="utf-8"))
        associated_data = f"snapshot:{snapshot_id}:generation:{generation}".encode("utf-8")
        return decrypt_json_blob(encrypted, keys.snapshot_key, associated_data=associated_data)
```

- [ ] **Step 4: Run store tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_skill_trust_store.py --tb=short
```

Expected: pass.

- [ ] **Step 5: Commit trust store**

Run:

```bash
git add tldw_chatbook/Skills_Interop/skill_trust_store.py Tests/Skills/test_skill_trust_store.py
git commit -m "feat: persist local skill trust manifests"
```

Expected: commit succeeds.

## Task 5: Add SkillTrustService Bootstrap, Classification, Review, And Approval

**Files:**
- Create: `tldw_chatbook/Skills_Interop/skill_trust_service.py`
- Test: `Tests/Skills/test_skill_trust_service.py`

- [ ] **Step 1: Write failing service tests**

Create `Tests/Skills/test_skill_trust_service.py`:

```python
import pytest

from tldw_chatbook.Skills_Interop.skill_trust_crypto import derive_skill_trust_keys
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import FileSkillTrustGenerationMarkerStore, SkillTrustStore


def _service(tmp_path, passphrase="passphrase"):
    skills_dir = tmp_path / "skills"
    trust_store = SkillTrustStore(
        store_dir=tmp_path / "trust",
        marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
    )
    service = SkillTrustService(skills_dir=skills_dir, trust_store=trust_store)
    service.unlock_with_passphrase(passphrase, salt=b"6" * 32)
    return service, skills_dir


def test_uninitialized_service_blocks_until_bootstrap(tmp_path):
    service, skills_dir = _service(tmp_path)
    (skills_dir / "demo").mkdir(parents=True)
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\n", encoding="utf-8")

    status = service.status_for_skill("demo")

    assert status.trust_status == "trust_uninitialized"
    assert status.trust_blocked is True
    with pytest.raises(SkillTrustBlockedError, match="trust_uninitialized"):
        service.ensure_skill_trusted("demo")


def test_bootstrap_trusts_current_files_and_detects_modification(tmp_path):
    service, skills_dir = _service(tmp_path)
    (skills_dir / "demo").mkdir(parents=True)
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\n", encoding="utf-8")

    service.bootstrap_trust()
    trusted = service.status_for_skill("demo")
    assert trusted.trust_status == "trusted"
    service.ensure_skill_trusted("demo")

    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\nBackdoor\n", encoding="utf-8")
    modified = service.status_for_skill("demo")

    assert modified.trust_status == "quarantined_modified"
    assert modified.changed_files == ("SKILL.md",)
    with pytest.raises(SkillTrustBlockedError, match="skill_modified"):
        service.ensure_skill_trusted("demo")


def test_existing_manifest_without_unlock_reports_locked_status(tmp_path):
    service, skills_dir = _service(tmp_path)
    (skills_dir / "demo").mkdir(parents=True)
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    service.bootstrap_trust()

    locked_service = SkillTrustService(
        skills_dir=skills_dir,
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
    )
    status = locked_service.status_for_skill("demo")

    assert status.trust_status == "trust_locked"
    assert status.trust_blocked is True


def test_missing_marker_reports_global_manifest_error(tmp_path):
    service, skills_dir = _service(tmp_path)
    (skills_dir / "demo").mkdir(parents=True)
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    service.bootstrap_trust()
    (tmp_path / "marker.json").unlink()

    status = service.status_for_skill("demo")

    assert status.trust_status == "quarantined_manifest_error"
    assert status.trust_blocked is True


def test_review_approval_requires_live_files_to_match_reviewed_snapshot(tmp_path):
    service, skills_dir = _service(tmp_path)
    (skills_dir / "demo").mkdir(parents=True)
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    service.bootstrap_trust()
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\nChanged\n", encoding="utf-8")

    review = service.capture_review("demo")
    assert review["changed_files"] == ["SKILL.md"]
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\nChanged again\n", encoding="utf-8")

    with pytest.raises(ValueError, match="snapshot_mismatch"):
        service.trust_reviewed_snapshot(review["review_id"])


def test_review_approval_restores_trust_for_reviewed_snapshot(tmp_path):
    service, skills_dir = _service(tmp_path)
    (skills_dir / "demo").mkdir(parents=True)
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    service.bootstrap_trust()
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\nChanged\n", encoding="utf-8")

    review = service.capture_review("demo")
    service.trust_reviewed_snapshot(review["review_id"])

    assert service.status_for_skill("demo").trust_status == "trusted"
```

- [ ] **Step 2: Run service tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_skill_trust_service.py --tb=short
```

Expected: fail with `ModuleNotFoundError` for `skill_trust_service`.

- [ ] **Step 3: Implement the service substrate**

Create `tldw_chatbook/Skills_Interop/skill_trust_service.py` with these public methods and fields:

```python
"""Orchestration service for Chatbook-managed local skill trust."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .skill_trust_crypto import canonical_json, derive_skill_trust_keys, sha256_hex
from .skill_trust_models import (
    SkillDirectorySnapshot,
    SkillTrustBlockedError,
    SkillTrustStatus,
    TRUST_REASON_MANIFEST_INVALID,
    TRUST_REASON_ROLLBACK_MARKER_UNAVAILABLE,
    TRUST_REASON_SKILL_ADDED,
    TRUST_REASON_SKILL_DELETED,
    TRUST_REASON_SKILL_MODIFIED,
    TRUST_REASON_TRUST_LOCKED,
    TRUST_REASON_TRUST_UNINITIALIZED,
    TRUST_REASON_UNSUPPORTED_PATH,
    TRUST_STATUS_QUARANTINED_ADDED,
    TRUST_STATUS_QUARANTINED_DELETED,
    TRUST_STATUS_QUARANTINED_MANIFEST_ERROR,
    TRUST_STATUS_QUARANTINED_MODIFIED,
    TRUST_STATUS_QUARANTINED_UNSUPPORTED_PATH,
    TRUST_STATUS_TRUSTED,
    TRUST_STATUS_LOCKED,
    TRUST_STATUS_UNINITIALIZED,
)
from .skill_trust_scanner import scan_skill_directory
from .skill_trust_store import SkillTrustMarkerUnavailable


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillTrustService:
    def __init__(
        self,
        *,
        skills_dir: Path,
        trust_store: Any,
        key_cache: Any | None = None,
        keyring_convenience_enabled: bool = False,
        reduced_rollback_protection: bool = False,
    ) -> None:
        self.skills_dir = skills_dir
        self.trust_store = trust_store
        self.key_cache = key_cache
        self.keyring_convenience_enabled = keyring_convenience_enabled
        self.reduced_rollback_protection = reduced_rollback_protection
        self._keys: Any | None = None
        self._salt: bytes | None = None
        self._reviews: dict[str, dict[str, Any]] = {}

    def unlock_with_passphrase(self, passphrase: str, *, salt: bytes | None = None) -> None:
        if salt is None:
            salt = self.trust_store.load_salt()
        self._keys = derive_skill_trust_keys(passphrase, salt=salt)
        self._salt = salt

    def enable_keyring_convenience(self) -> None:
        if self.key_cache is None:
            raise SkillTrustMarkerUnavailable("No secure OS-backed key cache is available.")
        self.key_cache.save_keys(self._require_keys())
        self.keyring_convenience_enabled = True

    def unlock_from_keyring_convenience(self) -> bool:
        if self.key_cache is None:
            return False
        keys = self.key_cache.load_keys()
        if keys is None:
            return False
        self._keys = keys
        self._salt = self.trust_store.load_salt()
        self.keyring_convenience_enabled = True
        return True

    def _require_keys(self) -> Any:
        if self._keys is None:
            raise SkillTrustBlockedError(
                skill_name="<all>",
                reason_code="trust_locked",
                trust_status="trust_locked",
            )
        return self._keys

    def _require_salt(self) -> bytes:
        if self._salt is None:
            raise ValueError("skill trust salt missing")
        return self._salt

    def _empty_manifest(self) -> dict[str, Any]:
        return {"version": 1, "generation": 0, "skills": {}, "audit": []}

    def _load_manifest_or_uninitialized(self) -> dict[str, Any] | None:
        if not self.trust_store.has_manifest():
            return None
        return self.trust_store.load_manifest(self._require_keys())

    def _locked_status(self, skill_name: str) -> SkillTrustStatus:
        return SkillTrustStatus(skill_name, TRUST_STATUS_LOCKED, TRUST_REASON_TRUST_LOCKED, True, (), None, _now_iso())

    def _manifest_error_status(self, skill_name: str, reason_code: str) -> SkillTrustStatus:
        return SkillTrustStatus(skill_name, TRUST_STATUS_QUARANTINED_MANIFEST_ERROR, reason_code, True, (), None, _now_iso())

    def _fingerprints_digest(self, snapshot: SkillDirectorySnapshot) -> str:
        entries = [item.as_manifest_entry() for item in snapshot.fingerprints]
        return sha256_hex(canonical_json(entries))

    def overall_status(self) -> str:
        if not self.trust_store.has_manifest():
            return TRUST_STATUS_UNINITIALIZED
        if self._keys is None:
            return TRUST_STATUS_LOCKED
        try:
            self.trust_store.load_manifest(self._keys)
        except SkillTrustMarkerUnavailable:
            return TRUST_STATUS_QUARANTINED_MANIFEST_ERROR
        except ValueError:
            return TRUST_STATUS_QUARANTINED_MANIFEST_ERROR
        return TRUST_STATUS_TRUSTED

    def bootstrap_trust(self, passphrase: str | None = None, *, salt: bytes | None = None) -> None:
        if passphrase is not None:
            self.unlock_with_passphrase(passphrase, salt=salt or secrets.token_bytes(32))
        keys = self._require_keys()
        manifest_salt = self._require_salt()
        skills: dict[str, Any] = {}
        generation = 1
        for skill_dir in sorted(self.skills_dir.iterdir(), key=lambda item: item.name) if self.skills_dir.exists() else []:
            if not skill_dir.is_dir():
                continue
            snapshot = scan_skill_directory(skill_dir.name, skill_dir)
            if snapshot.unsupported_paths:
                raise ValueError("unsupported_path")
            snapshot_id = f"{skill_dir.name}-{generation}"
            self.trust_store.save_snapshot(
                snapshot_id,
                {"files": snapshot.text_files},
                keys,
                generation=generation,
            )
            skills[skill_dir.name] = {
                "files": [item.as_manifest_entry() for item in snapshot.fingerprints],
                "snapshot_id": snapshot_id,
                "trusted_at": _now_iso(),
            }
        manifest = {
            "version": 1,
            "generation": generation,
            "skills": skills,
            "audit": [{"event": "trust_bootstrap", "at": _now_iso(), "skill_count": len(skills)}],
        }
        self.trust_store.save_manifest(manifest, keys, salt=manifest_salt)

    def status_for_skill(self, skill_name: str) -> SkillTrustStatus:
        if not self.trust_store.has_manifest():
            return SkillTrustStatus(skill_name, TRUST_STATUS_UNINITIALIZED, TRUST_REASON_TRUST_UNINITIALIZED, True, (), None, None)
        if self._keys is None:
            return self._locked_status(skill_name)
        try:
            manifest = self.trust_store.load_manifest(self._keys)
        except SkillTrustMarkerUnavailable:
            return self._manifest_error_status(skill_name, TRUST_REASON_ROLLBACK_MARKER_UNAVAILABLE)
        except ValueError:
            return self._manifest_error_status(skill_name, TRUST_REASON_MANIFEST_INVALID)
        skill_dir = self.skills_dir / skill_name
        current = scan_skill_directory(skill_name, skill_dir)
        if current.unsupported_paths:
            return SkillTrustStatus(
                skill_name,
                TRUST_STATUS_QUARANTINED_UNSUPPORTED_PATH,
                TRUST_REASON_UNSUPPORTED_PATH,
                True,
                current.unsupported_paths,
                int(manifest["generation"]),
                _now_iso(),
            )
        trusted = manifest.get("skills", {}).get(skill_name)
        if trusted is None:
            return SkillTrustStatus(skill_name, TRUST_STATUS_QUARANTINED_ADDED, TRUST_REASON_SKILL_ADDED, True, tuple(item.relative_path for item in current.fingerprints), int(manifest["generation"]), _now_iso())
        trusted_files = {item["relative_path"]: item for item in trusted.get("files", [])}
        current_files = {item.relative_path: item.as_manifest_entry() for item in current.fingerprints}
        changed = tuple(sorted(set(trusted_files) ^ set(current_files) | {path for path in trusted_files.keys() & current_files.keys() if trusted_files[path] != current_files[path]}))
        if changed:
            reason = TRUST_REASON_SKILL_DELETED if any(path not in current_files for path in changed) else TRUST_REASON_SKILL_MODIFIED
            status = TRUST_STATUS_QUARANTINED_DELETED if reason == TRUST_REASON_SKILL_DELETED else TRUST_STATUS_QUARANTINED_MODIFIED
            return SkillTrustStatus(skill_name, status, reason, True, changed, int(manifest["generation"]), _now_iso())
        return SkillTrustStatus(skill_name, TRUST_STATUS_TRUSTED, None, False, (), int(manifest["generation"]), _now_iso())

    def ensure_skill_trusted(self, skill_name: str) -> None:
        status = self.status_for_skill(skill_name)
        if not status.trust_blocked:
            return
        raise SkillTrustBlockedError(
            skill_name=skill_name,
            reason_code=status.trust_reason_code or "trust_blocked",
            trust_status=status.trust_status,
            changed_files=status.changed_files,
        )

    def capture_review(self, skill_name: str) -> dict[str, Any]:
        status = self.status_for_skill(skill_name)
        current = scan_skill_directory(skill_name, self.skills_dir / skill_name)
        review_id = secrets.token_hex(16)
        review = {
            "review_id": review_id,
            "skill_name": skill_name,
            "current_digest": self._fingerprints_digest(current),
            "current_files": current.text_files,
            "changed_files": list(status.changed_files),
        }
        self._reviews[review_id] = review
        return review

    def trust_reviewed_snapshot(self, review_id: str) -> None:
        review = self._reviews[review_id]
        skill_name = review["skill_name"]
        current = scan_skill_directory(skill_name, self.skills_dir / skill_name)
        current_digest = self._fingerprints_digest(current)
        if current_digest != review["current_digest"]:
            raise ValueError("snapshot_mismatch")
        self.trust_current_skill(skill_name, audit_event="trust_approved", snapshot=current)

    def trust_current_skill(
        self,
        skill_name: str,
        *,
        audit_event: str = "trust_chatbook_mutation",
        snapshot: SkillDirectorySnapshot | None = None,
    ) -> None:
        keys = self._require_keys()
        manifest = self.trust_store.load_manifest(keys)
        generation = int(manifest["generation"]) + 1
        current = snapshot or scan_skill_directory(skill_name, self.skills_dir / skill_name)
        if current.unsupported_paths:
            raise ValueError("unsupported_path")
        snapshot_id = f"{skill_name}-{generation}"
        self.trust_store.save_snapshot(snapshot_id, {"files": current.text_files}, keys, generation=generation)
        manifest["generation"] = generation
        manifest.setdefault("skills", {})[skill_name] = {
            "files": [item.as_manifest_entry() for item in current.fingerprints],
            "snapshot_id": snapshot_id,
            "trusted_at": _now_iso(),
        }
        manifest.setdefault("audit", []).append({"event": audit_event, "at": _now_iso(), "skill_name": skill_name})
        self.trust_store.save_manifest(manifest, keys)
```

- [ ] **Step 4: Run service tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_skill_trust_service.py --tb=short
```

Expected: pass.

- [ ] **Step 5: Commit service substrate**

Run:

```bash
git add tldw_chatbook/Skills_Interop/skill_trust_service.py Tests/Skills/test_skill_trust_service.py
git commit -m "feat: add local skill trust service"
```

Expected: commit succeeds.

## Task 6: Integrate Trust Enforcement Into LocalSkillsService

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py`
- Modify: `tldw_chatbook/Skills_Interop/skills_scope_service.py`
- Test: `Tests/Skills/test_local_skills_service.py`
- Test: `Tests/Skills/test_skills_scope_service.py`

- [ ] **Step 1: Add failing local service trust tests**

Append to `Tests/Skills/test_local_skills_service.py`:

```python
import pytest

from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import FileSkillTrustGenerationMarkerStore, SkillTrustStore


def _trusted_local_service(tmp_path):
    trust_service = SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
    )
    trust_service.unlock_with_passphrase("passphrase", salt=b"7" * 32)
    return LocalSkillsService(store_dir=tmp_path, trust_service=trust_service), trust_service


@pytest.mark.asyncio
async def test_local_skills_service_exposes_trust_state_and_blocks_uninitialized_context(tmp_path):
    service, _trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nRender {{args}}")

    listed = await service.list_skills()
    context = await service.get_context()

    assert listed["skills"][0]["trust_status"] == "trust_uninitialized"
    assert listed["skills"][0]["trust_blocked"] is True
    assert context["available_skills"] == []
    assert context["blocked_skills"][0]["name"] == "demo-skill"


@pytest.mark.asyncio
async def test_local_skills_service_blocks_execute_when_skill_changes_on_disk_after_bootstrap(tmp_path):
    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nRender {{args}}")
    trust.bootstrap_trust()
    (tmp_path / "skills" / "demo-skill" / "SKILL.md").write_text("# Demo\nChanged {{args}}", encoding="utf-8")

    with pytest.raises(SkillTrustBlockedError, match="skill_modified"):
        await service.execute_skill("demo-skill", args="x")


@pytest.mark.asyncio
async def test_local_skills_service_retrusts_explicitly_approved_update(tmp_path):
    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nRender {{args}}")
    trust.bootstrap_trust()

    await service.update_skill("demo-skill", content="# Demo\nChanged {{args}}", trust_approved=True)

    listed = await service.list_skills()
    assert listed["skills"][0]["trust_status"] == "trusted"
```

- [ ] **Step 2: Run failing local service trust tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_local_skills_service.py::test_local_skills_service_exposes_trust_state_and_blocks_uninitialized_context Tests/Skills/test_local_skills_service.py::test_local_skills_service_blocks_execute_when_skill_changes_on_disk_after_bootstrap Tests/Skills/test_local_skills_service.py::test_local_skills_service_retrusts_explicitly_approved_update --tb=short
```

Expected: fail because `LocalSkillsService` does not accept `trust_service`.

- [ ] **Step 3: Modify LocalSkillsService constructor and response helpers**

In `tldw_chatbook/Skills_Interop/local_skills_service.py`, update the constructor:

```python
    def __init__(
        self,
        *,
        store_dir: str | Path,
        policy_enforcer: Any | None = None,
        trust_service: Any | None = None,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.skills_dir = self.store_dir / _SKILLS_DIRNAME
        self.index_path = self.store_dir / _INDEX_FILENAME
        self.policy_enforcer = policy_enforcer
        self.trust_service = trust_service
        self._lock = asyncio.Lock()
```

Add helpers near `_summary_for_record`:

```python
    def _trust_fields_for_record(self, record: dict[str, Any]) -> dict[str, Any]:
        if self.trust_service is None:
            return {
                "trust_status": "trusted",
                "trust_reason_code": None,
                "trust_blocked": False,
                "trust_changed_files": [],
                "trust_manifest_generation": None,
                "trust_last_verified_at": None,
            }
        return self.trust_service.status_for_skill(str(record["name"])).response_fields()

    def _require_trusted_skill(self, skill_name: str) -> None:
        if self.trust_service is not None:
            self.trust_service.ensure_skill_trusted(skill_name)

    def _trust_after_approved_mutation(self, skill_name: str, *, trust_approved: bool) -> None:
        if self.trust_service is not None and trust_approved:
            self.trust_service.trust_current_skill(skill_name, audit_event="trust_chatbook_mutation")
```

Update `_response_for_record()` and `_summary_for_record()` call sites so trust fields are merged into detail/list dictionaries:

```python
        payload = self._dump(response)
        payload.update(self._trust_fields_for_record(record))
        return payload
```

Change `_summary_for_record` from `@staticmethod` to an instance method and merge trust fields:

```python
        summary.update(self._trust_fields_for_record(record))
```

- [ ] **Step 4: Exclude trust-blocked skills from context and block execution**

In `get_context()`, build `available_skills` only from non-blocked records:

```python
        available: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        for _, record in sorted(records.items()):
            summary = self._summary_for_record(record)
            if summary.get("trust_blocked"):
                blocked.append(summary)
                continue
            available.append(summary)
```

Return `blocked_skills` as an extra field:

```python
        payload = self._dump(
            SkillContextPayload(
                available_skills=available,
                context_text="\n".join(context_lines),
            )
        )
        payload["blocked_skills"] = blocked
        return payload
```

At the start of `execute_skill()`, add:

```python
        self._require_trusted_skill(skill_name)
```

- [ ] **Step 5: Rebaseline only explicitly approved Chatbook mutations**

Add a keyword-only `trust_approved: bool = False` argument to `create_skill()`, `update_skill()`, `import_skill()`, and `import_skill_file()`. After the skill file, supporting files, and index entry are written successfully, call:

```python
            self._trust_after_approved_mutation(skill_name, trust_approved=trust_approved)
```

The UI layer must pass `trust_approved=True` only from Save/Import confirmations that explicitly say the trusted baseline will be updated. Direct filesystem writes, background sync, or API calls without that flag must remain quarantined as added/modified/deleted.

- [ ] **Step 6: Run local service trust tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_local_skills_service.py::test_local_skills_service_exposes_trust_state_and_blocks_uninitialized_context Tests/Skills/test_local_skills_service.py::test_local_skills_service_blocks_execute_when_skill_changes_on_disk_after_bootstrap Tests/Skills/test_local_skills_service.py::test_local_skills_service_retrusts_explicitly_approved_update --tb=short
```

Expected: pass.

- [ ] **Step 7: Run full local skills tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_local_skills_service.py Tests/Skills/test_skills_scope_service.py --tb=short
```

Expected: pass. Legacy tests that instantiate `LocalSkillsService` without `trust_service` continue to use the explicit no-trust-service compatibility path, but app wiring tests must prove production constructs a real trust service.

- [ ] **Step 8: Commit local service integration**

Run:

```bash
git add tldw_chatbook/Skills_Interop/local_skills_service.py tldw_chatbook/Skills_Interop/skills_scope_service.py Tests/Skills/test_local_skills_service.py Tests/Skills/test_skills_scope_service.py
git commit -m "feat: enforce trust for local skills"
```

Expected: commit succeeds.

## Task 7: Wire Trust Service In App Construction

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Add failing app wiring assertion**

In `Tests/UI/test_screen_navigation.py`, extend the local skills service wiring test with:

```python
    assert app.local_skill_trust_service is app.local_skills_service.trust_service
    assert app.local_skill_trust_service.skills_dir == app.local_skills_service.skills_dir
```

- [ ] **Step 2: Run failing app wiring test**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py::test_app_wires_local_and_server_skills_services --tb=short
```

Expected: fail because `local_skill_trust_service` is not wired.

- [ ] **Step 3: Wire trust store and service in `app.py`**

Near the existing `LocalSkillsService` construction, import:

```python
from .Skills_Interop.skill_trust_service import SkillTrustService
from .Skills_Interop.skill_trust_store import (
    SkillTrustStore,
    build_default_skill_trust_key_cache,
    build_default_skill_trust_marker_store,
)
```

Replace the local skills construction with:

```python
        local_skills_store_dir = get_user_data_dir() / "skills"
        self.local_skill_trust_service = SkillTrustService(
            skills_dir=local_skills_store_dir / "skills",
            trust_store=SkillTrustStore(
                store_dir=local_skills_store_dir / "trust",
                marker_store=build_default_skill_trust_marker_store(),
            ),
            key_cache=build_default_skill_trust_key_cache(),
            keyring_convenience_enabled=False,
            reduced_rollback_protection=False,
        )
        self.local_skills_service = LocalSkillsService(
            store_dir=local_skills_store_dir,
            policy_enforcer=self.service_policy_enforcer,
            trust_service=self.local_skill_trust_service,
        )
```

- [ ] **Step 4: Run app wiring test**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py::test_app_wires_local_and_server_skills_services --tb=short
```

Expected: pass.

- [ ] **Step 5: Commit app wiring**

Run:

```bash
git add tldw_chatbook/app.py Tests/UI/test_screen_navigation.py
git commit -m "feat: wire local skill trust service"
```

Expected: commit succeeds.

## Task 8: Add Skills Screen Trust States And Recovery Actions

**Files:**
- Modify: `tldw_chatbook/UI/Screens/skills_screen.py`
- Test: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Add failing Skills UI trust-state tests**

Add `from unittest.mock import AsyncMock` near the other imports if it is not already present, then append to the Skills destination tests:

```python
@pytest.mark.asyncio
async def test_skills_destination_blocks_metadata_valid_trust_blocked_skill():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "record_id": "local:skill:summarize-notes",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "quarantined_modified",
                "trust_reason_code": "skill_modified",
                "trust_blocked": True,
                "trust_changed_files": ["SKILL.md"],
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        text = _visible_text(screen)
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "Trust: changed since trusted baseline" in text
        assert "Reason code: skill_modified" in text
        assert "Changed files: SKILL.md" in text
        assert button.disabled is True


@pytest.mark.asyncio
async def test_skills_destination_shows_uninitialized_bootstrap_action():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "new-skill",
                "description": "New local skill",
                "record_id": "local:skill:new-skill",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "trust_uninitialized",
                "trust_reason_code": "trust_uninitialized",
                "trust_blocked": True,
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        text = _visible_text(screen)

        assert "Trust: not initialized" in text
        assert screen.query_one("#skills-bootstrap-trust", Button).disabled is False


@pytest.mark.asyncio
async def test_skills_destination_bootstrap_action_calls_trust_service():
    app = _build_test_app()
    app.local_skill_trust_service = RecordingSkillTrustService()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "new-skill",
                "description": "New local skill",
                "record_id": "local:skill:new-skill",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "trust_uninitialized",
                "trust_reason_code": "trust_uninitialized",
                "trust_blocked": True,
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        screen._request_skill_trust_passphrase = AsyncMock(return_value="passphrase")
        await pilot.click("#skills-bootstrap-trust")

        assert app.local_skill_trust_service.bootstrap_calls == 1


@pytest.mark.asyncio
async def test_skills_destination_review_action_enables_trust_reviewed_version():
    app = _build_test_app()
    app.local_skill_trust_service = RecordingSkillTrustService(review_id="review-1")
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "record_id": "local:skill:summarize-notes",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "quarantined_modified",
                "trust_reason_code": "skill_modified",
                "trust_blocked": True,
                "trust_changed_files": ["SKILL.md"],
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        await pilot.click("#skills-review-diff")

        assert app.local_skill_trust_service.reviewed_skill == "summarize-notes"
        assert screen.query_one("#skills-trust-reviewed-version", Button).disabled is False

        await pilot.click("#skills-trust-reviewed-version")
        assert app.local_skill_trust_service.trusted_review_id == "review-1"
```

Add a small fake near the destination test helpers:

```python
class RecordingSkillTrustService:
    def __init__(self, review_id="review-id"):
        self.review_id = review_id
        self.bootstrap_calls = 0
        self.reviewed_skill = None
        self.trusted_review_id = None

    def bootstrap_trust(self, *args, **kwargs):
        self.bootstrap_calls += 1

    def capture_review(self, skill_name):
        self.reviewed_skill = skill_name
        return {"review_id": self.review_id, "skill_name": skill_name, "changed_files": ["SKILL.md"]}

    def trust_reviewed_snapshot(self, review_id):
        self.trusted_review_id = review_id
```

- [ ] **Step 2: Run failing Skills UI tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py::test_skills_destination_blocks_metadata_valid_trust_blocked_skill Tests/UI/test_destination_shells.py::test_skills_destination_shows_uninitialized_bootstrap_action Tests/UI/test_destination_shells.py::test_skills_destination_bootstrap_action_calls_trust_service Tests/UI/test_destination_shells.py::test_skills_destination_review_action_enables_trust_reviewed_version --tb=short
```

Expected: fail because the screen does not render trust state.

- [ ] **Step 3: Add trust helpers to SkillsScreen**

In `tldw_chatbook/UI/Screens/skills_screen.py`, add methods:

```python
    def _skill_trust_status(self, record: Mapping[str, Any]) -> str:
        return self._skill_field(record, "trust_status", "trusted")

    def _skill_trust_reason(self, record: Mapping[str, Any]) -> str:
        return self._skill_field(record, "trust_reason_code")

    def _skill_trust_blocked(self, record: Mapping[str, Any]) -> bool:
        return bool(record.get("trust_blocked"))

    def _skill_trust_changed_files(self, record: Mapping[str, Any]) -> list[str]:
        files = record.get("trust_changed_files")
        if not isinstance(files, list):
            return []
        return [
            self._safe_skill_text(file_name, max_length=100)
            for file_name in files
            if self._safe_skill_text(file_name, max_length=100)
        ]

    def _skill_trust_copy(self, record: Mapping[str, Any]) -> str:
        status = self._skill_trust_status(record)
        if status == "trusted":
            return "Trust: trusted baseline"
        if status == "trust_uninitialized":
            return "Trust: not initialized"
        if status == "trust_locked":
            return "Trust: locked"
        if status == "quarantined_modified":
            return "Trust: changed since trusted baseline"
        if status == "quarantined_added":
            return "Trust: new untrusted file"
        if status == "quarantined_deleted":
            return "Trust: trusted file missing"
        if status == "quarantined_manifest_error":
            return "Trust: manifest cannot be verified"
        if status == "quarantined_unsupported_path":
            return "Trust: unsupported file path"
        return "Trust: blocked"
```

Update `_is_skill_valid()`:

```python
    def _is_skill_valid(self, record: Mapping[str, Any]) -> bool:
        return self._skill_validation_status(record) == "valid" and not self._skill_trust_blocked(record)
```

- [ ] **Step 4: Render trust rows and recovery buttons**

In the skill-list render loop, after validation rows, add:

```python
                            yield Static(
                                self._plain_text(self._skill_trust_copy(record)),
                                id=f"skills-trust-status-{index}",
                            )
                            trust_reason = self._skill_trust_reason(record)
                            if trust_reason:
                                yield Static(
                                    self._plain_text(f"Reason code: {trust_reason}"),
                                    id=f"skills-trust-reason-{index}",
                                )
                            changed_files = ", ".join(self._skill_trust_changed_files(record))
                            if changed_files:
                                yield Static(
                                    self._plain_text(f"Changed files: {changed_files}"),
                                    id=f"skills-trust-files-{index}",
                                )
```

In the inspector actions, add disabled-safe recovery controls:

```python
                    yield Button(
                        "Bootstrap Trust",
                        id="skills-bootstrap-trust",
                        disabled=not any(self._skill_trust_status(record) == "trust_uninitialized" for record in self._local_skill_records),
                        tooltip="Create the first trusted baseline after reviewing current local skills.",
                    )
                    yield Button(
                        "Unlock Trust",
                        id="skills-unlock-trust",
                        disabled=not any(self._skill_trust_status(record) == "trust_locked" for record in self._local_skill_records),
                        tooltip="Unlock local skill trust with the trust passphrase for this session.",
                    )
                    yield Button(
                        "Review Diff",
                        id="skills-review-diff",
                        disabled=not (selected_metadata and selected_metadata.get("validation_status") == "valid"),
                        tooltip="Capture the current skill files and compare them with the trusted baseline.",
                    )
                    yield Button(
                        "Trust Reviewed Version",
                        id="skills-trust-reviewed-version",
                        disabled=self._active_trust_review is None,
                        tooltip="Enabled after a captured diff still matches live files.",
                    )
```

- [ ] **Step 5: Wire Skills recovery button handlers**

Add `self._active_trust_review: dict[str, Any] | None = None` to the screen state. In the existing button handler, route the trust buttons through `app.local_skill_trust_service`:

```python
    async def _handle_skill_trust_action(self, button_id: str) -> None:
        trust_service = getattr(self.app_instance, "local_skill_trust_service", None)
        if trust_service is None:
            return
        if button_id == "skills-bootstrap-trust":
            passphrase = await self._request_skill_trust_passphrase(confirm_bootstrap=True)
            if passphrase is None:
                return
            trust_service.bootstrap_trust(passphrase)
            await self._refresh_skills()
            return
        if button_id == "skills-unlock-trust":
            passphrase = await self._request_skill_trust_passphrase(confirm_bootstrap=False)
            if passphrase is None:
                return
            trust_service.unlock_with_passphrase(passphrase)
            await self._refresh_skills()
            return
        selected = self._selected_skill_record()
        if selected is None:
            return
        skill_name = str(selected.get("name") or "")
        if button_id == "skills-review-diff":
            self._active_trust_review = trust_service.capture_review(skill_name)
            await self._refresh_skills()
            return
        if button_id == "skills-trust-reviewed-version" and self._active_trust_review:
            trust_service.trust_reviewed_snapshot(str(self._active_trust_review["review_id"]))
            self._active_trust_review = None
            await self._refresh_skills()
```

Implement `_request_skill_trust_passphrase()` with existing password-dialog primitives or a small modal: no default passphrase, no logging, no path/secrets in notifications. When `confirm_bootstrap=True`, the modal copy must state that current local skill files will become the trusted baseline.

- [ ] **Step 6: Run Skills UI trust tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py::test_skills_destination_blocks_metadata_valid_trust_blocked_skill Tests/UI/test_destination_shells.py::test_skills_destination_shows_uninitialized_bootstrap_action Tests/UI/test_destination_shells.py::test_skills_destination_bootstrap_action_calls_trust_service Tests/UI/test_destination_shells.py::test_skills_destination_review_action_enables_trust_reviewed_version --tb=short
```

Expected: pass.

- [ ] **Step 7: Run focused Skills UI tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py -k "skills_destination or skills_attach" --tb=short
```

Expected: pass.

- [ ] **Step 8: Commit Skills UI trust state**

Run:

```bash
git add tldw_chatbook/UI/Screens/skills_screen.py Tests/UI/test_destination_shells.py
git commit -m "feat: show local skill trust states"
```

Expected: commit succeeds.

## Task 9: Add Settings Privacy Trust Posture

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_privacy_security.py`
- Modify: Settings rendering file that consumes `build_privacy_posture_rows`
- Test: `Tests/UI/test_settings_privacy_security.py`

- [ ] **Step 1: Add failing posture helper test**

Append to `Tests/UI/test_settings_privacy_security.py`:

```python
def test_privacy_posture_reports_skill_trust_without_leaking_paths():
    posture = build_settings_privacy_posture(
        {"encryption": {"enabled": True}},
        environ={},
        skill_trust={
            "enabled": True,
            "trust_status": "quarantined_modified",
            "keyring_convenience_enabled": True,
            "reduced_rollback_protection": False,
            "skills_dir": "/Users/example/private/skills",
        },
    )

    text = "\n".join(build_privacy_posture_rows(posture))

    assert "Skill trust: quarantined_modified" in text
    assert "Skill trust keyring convenience: enabled" in text
    assert "Skill trust rollback protection: full" in text
    assert "/Users/example/private/skills" not in text
```

- [ ] **Step 2: Run failing posture test**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_settings_privacy_security.py::test_privacy_posture_reports_skill_trust_without_leaking_paths --tb=short
```

Expected: fail because `build_settings_privacy_posture()` does not accept `skill_trust`.

- [ ] **Step 3: Extend posture dataclass and builder**

In `settings_privacy_security.py`, extend `SettingsPrivacyPosture`:

```python
    skill_trust_enabled: bool = False
    skill_trust_status: str = "unavailable"
    skill_trust_keyring_convenience_enabled: bool = False
    skill_trust_reduced_rollback_protection: bool = False
```

Update `build_settings_privacy_posture()` signature:

```python
def build_settings_privacy_posture(
    app_config: object,
    *,
    environ: Mapping[str, str] | None = None,
    skill_trust: Mapping[str, object] | None = None,
) -> SettingsPrivacyPosture:
```

Add safe extraction before return:

```python
    trust = skill_trust if isinstance(skill_trust, Mapping) else {}
```

Pass fields into `SettingsPrivacyPosture`:

```python
        skill_trust_enabled=bool(trust.get("enabled")),
        skill_trust_status=str(trust.get("trust_status") or "unavailable")[:80],
        skill_trust_keyring_convenience_enabled=bool(trust.get("keyring_convenience_enabled")),
        skill_trust_reduced_rollback_protection=bool(trust.get("reduced_rollback_protection")),
```

Update `build_privacy_posture_rows()`:

```python
        f"Skill trust: {posture.skill_trust_status if posture.skill_trust_enabled else 'disabled'}",
        (
            "Skill trust keyring convenience: enabled"
            if posture.skill_trust_keyring_convenience_enabled
            else "Skill trust keyring convenience: disabled"
        ),
        (
            "Skill trust rollback protection: reduced"
            if posture.skill_trust_reduced_rollback_protection
            else "Skill trust rollback protection: full"
        ),
```

- [ ] **Step 4: Wire SettingsScreen posture input**

In `tldw_chatbook/UI/Screens/settings_screen.py`, add this helper near `_settings_privacy_posture()`:

```python
    def _skill_trust_posture(self) -> dict[str, object]:
        skill_trust_service = getattr(self.app_instance, "local_skill_trust_service", None)
        if skill_trust_service is None:
            return {
                "enabled": False,
                "trust_status": "unavailable",
                "keyring_convenience_enabled": False,
                "reduced_rollback_protection": False,
            }
        overall_status = getattr(skill_trust_service, "overall_status", None)
        return {
            "enabled": True,
            "trust_status": overall_status() if callable(overall_status) else "locked",
            "keyring_convenience_enabled": bool(getattr(skill_trust_service, "keyring_convenience_enabled", False)),
            "reduced_rollback_protection": bool(getattr(skill_trust_service, "reduced_rollback_protection", False)),
        }
```

Update `_settings_privacy_posture()` so it passes the mapping:

```python
        return build_settings_privacy_posture(
            app_config,
            skill_trust=self._skill_trust_posture(),
        )
```

In the Privacy & Security renderer, add these rows after the existing `Recovery actions` detail row:

```python
                yield self._detail_row(
                    "Skill trust",
                    posture.skill_trust_status if posture.skill_trust_enabled else "disabled",
                )
                yield self._detail_row(
                    "Skill trust keyring convenience",
                    "enabled" if posture.skill_trust_keyring_convenience_enabled else "disabled",
                )
                yield self._detail_row(
                    "Skill trust rollback protection",
                    "reduced" if posture.skill_trust_reduced_rollback_protection else "full",
                )
```

- [ ] **Step 5: Run Settings posture tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_settings_privacy_security.py --tb=short
```

Expected: pass.

- [ ] **Step 6: Commit Settings trust posture**

Run:

```bash
git add tldw_chatbook/UI/Screens/settings_privacy_security.py Tests/UI/test_settings_privacy_security.py
git commit -m "feat: report local skill trust posture"
```

Expected: commit succeeds.

## Task 10: Final Verification, Documentation, And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-137 - Add-local-skill-trust-integrity-controls.md`
- Modify: `Docs/superpowers/specs/2026-06-25-local-skill-trust-integrity-design.md` only if implementation deviated from the approved spec.
- Test: focused verification commands below.

- [ ] **Step 1: Run focused trust and Skills tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Skills/test_skill_trust_crypto.py Tests/Skills/test_skill_trust_scanner.py Tests/Skills/test_skill_trust_store.py Tests/Skills/test_skill_trust_service.py Tests/Skills/test_local_skills_service.py Tests/Skills/test_skills_scope_service.py --tb=short
```

Expected: pass.

- [ ] **Step 2: Run runtime policy tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/RuntimePolicy/test_domain_edge_contracts.py Tests/RuntimePolicy/test_runtime_policy_core.py --tb=short
```

Expected: pass.

- [ ] **Step 3: Run focused UI tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py -k "skills_destination or skills_attach" --tb=short
.venv/bin/python -m pytest -q Tests/UI/test_settings_privacy_security.py Tests/UI/test_screen_navigation.py --tb=short
```

Expected: pass.

- [ ] **Step 4: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 5: Update Backlog task acceptance criteria and implementation notes**

Use `backlog task edit` to mark the task Done only after every acceptance criterion is met. The notes must include this content:

```text
Implemented local skill trust integrity for Chatbook-managed local skills. Added ADR-009-backed trust modules for passphrase-derived keys, authenticated manifests, encrypted trusted snapshots, generation markers, directory scanning, logical quarantine, reviewed re-trust, and deterministic trust-blocked service behavior. Wired trust state into LocalSkillsService, app construction, Skills UI, Settings Privacy posture, and runtime-policy action IDs. Focused trust, Skills, runtime-policy, and UI tests pass. ADR path: backlog/decisions/009-local-skill-trust-boundary.md. Superpowers plan: Docs/superpowers/plans/2026-06-25-local-skill-trust-integrity.md.
```

Then run:

```bash
backlog task edit 137 -s Done --notes "Implemented local skill trust integrity for Chatbook-managed local skills. Added ADR-009-backed trust modules for passphrase-derived keys, authenticated manifests, encrypted trusted snapshots, generation markers, directory scanning, logical quarantine, reviewed re-trust, and deterministic trust-blocked service behavior. Wired trust state into LocalSkillsService, app construction, Skills UI, Settings Privacy posture, and runtime-policy action IDs. Focused trust, Skills, runtime-policy, and UI tests pass. ADR path: backlog/decisions/009-local-skill-trust-boundary.md. Superpowers plan: Docs/superpowers/plans/2026-06-25-local-skill-trust-integrity.md."
```

Expected: task is Done and contains implementation notes.

- [ ] **Step 6: Commit closeout**

Run:

```bash
git add "backlog/tasks/task-137 - Add-local-skill-trust-integrity-controls.md"
git commit -m "docs: close local skill trust task"
```

Expected: commit succeeds.

## Self-Review Notes

Spec coverage:

- Explicit bootstrap and `trust_uninitialized`: Task 5, Task 6, Task 8.
- Authenticated manifest/snapshots/generation marker: Task 2, Task 4, Task 5.
- Logical quarantine and use-time blocking: Task 5, Task 6.
- Reviewed re-trust with snapshot mismatch protection: Task 5, Task 8.
- Keyring convenience and reduced rollback posture: Task 4, Task 9.
- UI and Settings recovery visibility: Task 8, Task 9.
- Runtime policy IDs: Task 1.
- ADR/task hygiene: Task 1, Task 10.

No implementation work should include Codex runtime skill folders, server skills, skill sync, or physical file moves.
