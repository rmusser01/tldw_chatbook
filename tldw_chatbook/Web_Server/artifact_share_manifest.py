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
import tempfile
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
_STAGED_DIR_MODE = 0o700


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
    root_existed = root.is_dir()
    root.mkdir(parents=True, exist_ok=True)
    if not root_existed:  # harden only what we create; tolerate pre-existing
        os.chmod(root, _STAGED_DIR_MODE)
    share_id = uuid.uuid4().hex
    share_dir = root / share_id
    share_dir.mkdir(exist_ok=False)
    os.chmod(share_dir, _STAGED_DIR_MODE)  # mkdir honors umask; enforce explicitly
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
        # Build the bundle through a 0600 mkstemp file in the same directory
        # and atomically move it into place: zipfile writing the final name
        # directly would leave it 0644 (umask-default) until a later chmod,
        # briefly exposing member names to other local users.
        bundle_fd, bundle_tmp = tempfile.mkstemp(prefix=".bundle-", dir=share_dir)
        os.close(bundle_fd)
        with zipfile.ZipFile(bundle_tmp, "w", zipfile.ZIP_STORED) as bundle:
            for item in staged:
                bundle.write(share_dir / item.staged_name, arcname=item.staged_name)
        os.replace(bundle_tmp, bundle_path)  # mkstemp mode 0600 is preserved
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
    if manifest.schema_version != ARTIFACT_SHARE_SCHEMA:
        # A missing "schema" key defaults to ARTIFACT_SHARE_SCHEMA and still
        # loads; any other value (including decorative future numbers) is
        # refused rather than guessed at.
        raise ArtifactShareError(
            f"Unsupported share manifest schema: {manifest.schema_version}"
        )
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
