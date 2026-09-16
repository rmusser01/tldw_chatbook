"""Local Notes pairing claims; imported history never supplies authorization."""

from __future__ import annotations

import hashlib
import json
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import (
    ActivationStore,
    _flush_existing,
    _private,
    _write,
)
from tldw_chatbook.Backup_Recovery.profile_paths import lexical_path
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


class _Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    version: int = Field(default=1, ge=1, le=1)
    owner: str
    binding: str
    generation: str
    installation: str
    nonce: str = Field(pattern=r"^[0-9a-f]{32}$")
    review: str


@dataclass(frozen=True)
class NotesRecoveryReview:
    """Complete local comparison without note contents or imported authority."""

    owner: str
    fingerprint: str
    root: Path
    entries: tuple[tuple[str, str], ...]
    historical_owners: tuple[str, ...]
    issues: tuple[str, ...]


def _identity(path):
    if path is None:
        raise ValueError("notes_pairing_source_unavailable")
    selected = lexical_path(path)
    info = selected.lstat()
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)) or (
        stat.S_ISREG(info.st_mode) and info.st_nlink != 1
    ):
        raise ValueError("notes_pairing_source_unsafe")
    if selected.resolve() != selected:
        raise ValueError("notes_pairing_source_unsafe")
    return (str(selected), info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))


@contextmanager
def _scope(owner, paths):
    # This existing pure reader validates actual admission plus paired generations.
    from tldw_chatbook.Backup_Recovery.generation_witnesses import _witnesses
    from tldw_chatbook.Backup_Recovery.isolated_restore import installation_client_id

    with ExitStack() as stack:
        witnesses = []
        for path in dict.fromkeys(paths):
            lease = stack.enter_context(acquire_storage(path))
            for witness in _witnesses(path, lease):
                if witness not in witnesses:
                    witnesses.append(witness)
        if not witnesses:
            yield (), None, None
            return
        active_installation = installation_client_id()
        installation = _verified_installation(witnesses)
        if active_installation != installation:
            raise ValueError("notes_pairing_fresh_process_required")
        if len(installation) != 32:
            raise ValueError("notes_pairing_identity_unavailable")
        binding = _digest(
            (
                owner,
                str(bootstrap.effective_config_path()),
                installation,
                tuple(_identity(path) for path in paths),
                witnesses,
            )
        )
        yield tuple(witnesses), binding, installation


def _verified_installation(witnesses):
    """Re-read the current committed identity instead of trusting startup cache."""
    from tldw_chatbook.Backup_Recovery.activation import replacement_installation_id
    from tldw_chatbook.Backup_Recovery.isolated_restore import _launch_descriptor
    from tldw_chatbook.Backup_Recovery.journal import Journal, _Prepared

    try:
        identity = replacement_installation_id()
    except bootstrap.RecoveryRequired as error:
        if error.args != ("isolated_profile_selector_required",):
            raise ValueError("notes_pairing_identity_unavailable") from None
        selector = str(bootstrap.effective_config_path())
        _, profiles, _ = bootstrap._control_records(bootstrap.default_bootstrap_root())
        witness = next(
            row["activation"] for row in profiles if row["selector"] == selector
        )
        if witness not in witnesses:
            raise ValueError("notes_pairing_identity_unavailable")
        control = Path(witness["store_root"]).parent
        journal = Journal(control, witness["operation_id"])
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        prepared = _Prepared.model_validate(
            next(row.evidence for row in rows if row.event == "prepared")
        )
        profile = next(
            row for row in prepared.isolated_profiles if row.config == selector
        )
        identity = _launch_descriptor(profile.profile_id, control).installation_id
    if identity is None:
        raise ValueError("notes_pairing_identity_unavailable")
    return identity


def _name(owner, binding):
    return "notes-pairing-" + _digest((owner, binding)) + ".json"


def _claims(owner, witnesses, binding, installation):
    claims = []
    for witness in witnesses:
        store = ActivationStore(Path(witness["store_root"]))
        with (
            _private(store.root),
            _private(store._generation(witness["generation"])) as parent,
        ):
            claim = _Claim.model_validate(
                bootstrap._read(parent, _name(owner, binding))
            )
        if (
            claim.owner != owner
            or claim.binding != binding
            or claim.generation != witness["generation"]
            or claim.installation != installation
            or not store.allowed(claim.generation, owner)
        ):
            raise ValueError("notes_pairing_review_required")
        claims.append(claim)
    if len({claim.nonce for claim in claims}) > 1:
        raise ValueError("notes_pairing_review_required")
    return claims


def require_pairing(owner: str, paths: tuple[Path | None, ...]) -> str | None:
    """Return a current local nonce, or None for an ordinary installation."""
    try:
        with _scope(owner, paths) as (witnesses, binding, installation):
            claims = _claims(owner, witnesses, binding, installation)
            return claims[0].nonce if claims else None
    except (OSError, ValueError, RuntimeError):
        raise PermissionError("notes_pairing_review_required") from None


def file_paths(service) -> tuple[Path, Path | None]:
    return (
        service.root,
        None
        if service._replica is None or service._replica.is_memory_db
        else Path(service._replica.db_path),
    )


def _comparison(disk, stored):
    entries = []
    for name in sorted(set(disk) | set(stored)):
        local, old = disk.get(name), stored.get(name)
        if local is None:
            state = "stored_only"
        elif old is None:
            state = "disk_only"
        elif local == old:
            state = "unchanged"
        else:
            state = "changed"
        entries.append((name, state))
    return tuple(entries)


def _observe_files(service):
    """Use the File Notes owner's exact member policy and raw-byte reader."""
    identity = _identity(service.root)
    observed, uncertain, failed = service._walk_candidates()
    disk, facts = {}, []
    issues = {"file_notes_scan_incomplete"} if uncertain or failed else set()
    for name in observed:
        try:
            opened = service._load_file(name)
            repeated = service._load_file(name)
            if (opened.content_hash, opened.size, opened.mtime_ns) != (
                repeated.content_hash,
                repeated.size,
                repeated.mtime_ns,
            ):
                raise ValueError("file_notes_source_changed")
            disk[name] = opened.content_hash
            facts.append((name, opened.content_hash, opened.size, opened.mtime_ns))
        except (OSError, ValueError):
            issues.add("file_notes_scan_incomplete")
    if (
        service._walk_candidates() != (observed, uncertain, failed)
        or _identity(service.root) != identity
    ):
        issues.add("file_notes_scan_incomplete")
    return disk, facts, issues


def _observe(target):
    """Compare the current File Notes root with its recovered replica."""
    disk, facts, scan_issues = _observe_files(target)
    stored = {
        row.relative_path: row.content_hash
        for row in target._replica.list_active_files(target.root_key)
    }
    facts.append(tuple(sorted(stored.items())))
    if target._pending_replica_moves:
        scan_issues.add("notes_pairing_pending_history")
    if target._session_owner.current_binding() != target._session_binding:
        scan_issues.add("notes_pairing_session_changed")
    return _comparison(disk, stored), (), tuple(sorted(scan_issues)), _digest(facts)


def review_pairing(
    owner: str,
    target,
    root: Path,
    user_id: str = "",
    *,
    expected: NotesRecoveryReview | None = None,
) -> NotesRecoveryReview:
    """Perform/recheck a complete owner comparison before a durable local claim."""
    if owner != "notes.file_notes":
        raise ValueError("notes_pairing_owner_unsupported")
    paths = file_paths(target)
    with _scope(owner, paths) as (witnesses, binding, installation):
        if not witnesses:
            raise ValueError("notes_recovery_not_required")
        entries, owners, issues, content = _observe(target)
        review = NotesRecoveryReview(
            owner,
            _digest((binding, content, owners, issues)),
            Path(root),
            entries,
            owners,
            issues,
        )
        if expected is None:
            return review
        if expected != review or issues:
            raise ValueError("notes_pairing_review_changed")
        nonce = None
        existing = []
        for witness in witnesses:
            store = ActivationStore(Path(witness["store_root"]))
            with (
                _private(store.root),
                _private(store._generation(witness["generation"])) as parent,
            ):
                if owner not in store._required(parent, witness["generation"]).owners:
                    raise ValueError("notes_pairing_owner_not_required")
                try:
                    record = _Claim.model_validate(
                        bootstrap._read(parent, _name(owner, binding))
                    )
                except FileNotFoundError:
                    record = None
                if record is not None:
                    if (
                        record.owner != owner
                        or record.binding != binding
                        or record.generation != witness["generation"]
                        or record.installation != installation
                        or nonce is not None
                        and nonce != record.nonce
                    ):
                        raise ValueError("notes_pairing_review_changed")
                    nonce = record.nonce
                existing.append((store, witness, record))
        nonce = nonce or uuid4().hex
        for store, witness, record in existing:
            claim = record or _Claim(
                owner=owner,
                binding=binding,
                generation=witness["generation"],
                installation=installation,
                nonce=nonce,
                review=review.fingerprint,
            )
            with (
                _private(store.root),
                _private(store._generation(claim.generation)) as parent,
            ):
                if record is None:
                    _write(parent, _name(owner, binding), claim)
                else:
                    _flush_existing(parent, _name(owner, binding), claim)
            store.approve(claim.generation, owner)
        return review
