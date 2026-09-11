"""Fresh local Skills trust roots selected by actual recovery generation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, _private
from tldw_chatbook.Backup_Recovery.profile_paths import lexical_path
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

from .skill_trust_models import SkillDirectorySnapshot


class RootBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    version: int = 1
    owner: str = "skills"
    binding: str
    generation: str
    root: str
    device: int
    inode: int


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def observed(paths, *, retained=None):
    """Use actual native admitted witnesses without importing an owner service."""
    from tldw_chatbook.Backup_Recovery.generation_witnesses import _witnesses

    with ExitStack() as stack:
        witnesses = []
        for path in dict.fromkeys(map(lexical_path, paths)):
            lease = (retained or {}).get(path)
            if lease is None:
                lease = stack.enter_context(acquire_storage(path))
            for witness in _witnesses(path, lease):
                if witness not in witnesses:
                    witnesses.append(witness)
        yield tuple(witnesses)


def is_recovered(skills_dir: Path, trust_root: Path) -> bool:
    """Passive app-factory selection before any keyring backend discovery."""
    with observed((skills_dir, trust_root)) as witnesses:
        return bool(witnesses)


def _sources(service):
    original = service._recovery_original_store
    marker = original.marker_store
    return tuple(
        path
        for path in (
            service.skills_dir,
            original.store_dir,
            getattr(marker, "marker_path", None),
            getattr(marker, "store_dir", None),
        )
        if path is not None
    )


def _identity(path):
    from tldw_chatbook.Backup_Recovery.native_files import pinned_directory

    path = lexical_path(path)
    try:
        with pinned_directory(path.parent) as parent:
            try:
                info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return str(path), None
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError("skills_recovery_source_unsafe")
            if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                raise ValueError("skills_recovery_source_unsafe")
            return str(path), info.st_dev, info.st_ino
    except FileNotFoundError:
        return str(path), None


def _selection(service, witnesses):
    sources = tuple(_identity(path) for path in _sources(service))
    binding = _digest((sources, witnesses))
    generations = sorted({w["generation"] for w in witnesses})
    suffix = generations[0] if len(generations) == 1 else _digest(generations)
    root = lexical_path(service._recovery_original_store.store_dir) / (
        "recovery-" + suffix
    )
    return binding, root


def select(service):
    """Select a passive fresh store; retain the original only as historical bytes."""
    from .skill_trust_store import FileSkillTrustGenerationMarkerStore, SkillTrustStore

    with observed(_sources(service)) as witnesses:
        if not witnesses:
            return
        _, root = _selection(service, witnesses)
        if service.trust_store.store_dir != root:
            service.trust_store = SkillTrustStore(
                root,
                FileSkillTrustGenerationMarkerStore(
                    root / "generation_marker.json", root
                ),
            )
            service.key_cache = None
            service.keyring_convenience_enabled = False
            service.reduced_rollback_protection = True
            service._keys = None
            service._salt = None


def _name(binding):
    return "skills-root-" + binding + ".json"


def _check_record(record, binding, root, witness):
    if (
        record.version != 1
        or record.owner != "skills"
        or record.binding != binding
        or record.generation != witness["generation"]
        or record.root != str(root)
    ):
        raise ValueError("skills_recovery_binding_changed")
    with _private(root) as fd:
        current = os.fstat(fd)
    if (current.st_dev, current.st_ino) != (record.device, record.inode):
        raise ValueError("skills_recovery_root_changed")


def _records(service, witnesses):
    binding, root = _selection(service, witnesses)
    if lexical_path(service.trust_store.store_dir) != root:
        raise ValueError("skills_recovery_root_changed")
    info = None
    for witness in witnesses:
        store = ActivationStore(Path(witness["store_root"]))
        with (
            _private(store.root),
            _private(store._generation(witness["generation"])) as parent,
        ):
            record = RootBinding.model_validate(bootstrap._read(parent, _name(binding)))
        _check_record(record, binding, root, witness)
        info = record
    return info


def allowed(service, *, retained=None):
    """A global owner flag cannot make an unbound historical root live."""
    trust = getattr(service, "trust_service", service)
    if trust is None or not hasattr(trust, "_recovery_original_store"):
        # Ordinary injected services still use their existing behavior; real
        # restored local services cannot obtain a fresh root through injection.
        with observed((service.skills_dir,), retained=retained) as witnesses:
            return not witnesses
    try:
        with observed(_sources(trust), retained=retained) as witnesses:
            if not witnesses:
                return True
            _records(trust, witnesses)
            return True
    except (OSError, ValueError, RuntimeError, TypeError):
        return False


def require_write(service):
    from .skill_trust_service import _active_execution, _execution_identity

    active = _active_execution.get()
    retained = (
        active[1] if active is not None and active[0] == _execution_identity() else None
    )
    if not allowed(service, retained=retained):
        raise ValueError("skills_recovery_root_review_required")


def needs_review(service) -> bool:
    """Read only the current local root binding and Skills review requirement."""
    with observed(_sources(service)) as witnesses:
        if not witnesses:
            return False
        try:
            _records(service, witnesses)
        except (OSError, ValueError):
            return True
        return any(
            not ActivationStore(Path(witness["store_root"])).allowed(
                witness["generation"], "skills"
            )
            for witness in witnesses
        )


@dataclass(frozen=True)
class RecoveryReview:
    """Actual current bundles and inactive historical grants for local review."""

    binding: str
    source_identities: tuple[tuple, ...]
    existing_trust_parent: tuple[str, int, int]
    missing_trust_parents: tuple[str, ...]
    skills: tuple[SkillDirectorySnapshot, ...]
    historical_script_grants: tuple[str, ...]
    historical_grants_digest: str
    reduced_rollback_protection: bool = True


def _parent_plan(service):
    """Record only the owner's missing trust-parent chain, without creating it."""
    path = lexical_path(service._recovery_original_store.store_dir)
    missing = []
    while len(identity := _identity(path)) == 2:
        missing.append(str(path))
        path = path.parent
    with _private(path):
        pass
    return identity, tuple(reversed(missing))


def _snapshots(service):
    from tldw_chatbook.Backup_Recovery.native_files import pinned_directory

    from .skill_trust_scanner import SUPPORTING_JUNK_DIRS

    result = []
    for name, root in service._iter_skill_dirs():
        # The normal scanner's os.walk is intentionally tolerant. A complete
        # recovery review must surface unreadable supported subdirectories.
        def failed(error):
            raise error

        for directory, children, _ in os.walk(root, followlinks=False, onerror=failed):
            with pinned_directory(Path(directory)) as fd:
                if os.fstat(fd).st_mode & 0o500 != 0o500:
                    raise ValueError("skills_recovery_scan_incomplete")
            children[:] = [
                child for child in children if child not in SUPPORTING_JUNK_DIRS
            ]
        snapshot = service._scan_skill(name)
        if snapshot.unsupported_paths or not snapshot.fingerprints:
            raise ValueError("skills_recovery_scan_incomplete")
        for item in snapshot.fingerprints:
            path = root / item.relative_path
            with pinned_directory(path.parent) as parent:
                fd = os.open(
                    path.name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=parent,
                )
                try:
                    before = os.fstat(fd)
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or before.st_nlink != 1
                        or not before.st_mode & 0o400
                    ):
                        raise ValueError("skills_recovery_scan_incomplete")
                    digest = hashlib.sha256()
                    remaining = item.byte_length + 1
                    while remaining:
                        chunk = os.read(fd, min(1024**2, remaining))
                        if not chunk:
                            break
                        digest.update(chunk)
                        remaining -= len(chunk)
                    after = os.fstat(fd)
                    if (
                        (
                            before.st_dev,
                            before.st_ino,
                            before.st_size,
                            before.st_mtime_ns,
                            before.st_ctime_ns,
                        )
                        != (
                            after.st_dev,
                            after.st_ino,
                            after.st_size,
                            after.st_mtime_ns,
                            after.st_ctime_ns,
                        )
                        or before.st_size != item.byte_length
                        or digest.hexdigest() != item.sha256
                        or bool(before.st_mode & stat.S_IXUSR) != item.executable
                    ):
                        raise ValueError("skills_recovery_scan_changed")
                finally:
                    os.close(fd)
        result.append(snapshot)
    return tuple(result)


def _review(service, witnesses):
    from tldw_chatbook.Backup_Recovery.storage_admission import _read_recovery_file

    binding, _ = _selection(service, witnesses)
    # Reading this old sidecar is disclosure only, never a grant or key probe.
    path = service._recovery_original_store.store_dir / "skill_script_grants.json"
    parent, missing = _parent_plan(service)
    raw = b""
    if not missing:
        try:
            raw = _read_recovery_file("skills", path, max_bytes=16 * 1024**2)
        except FileNotFoundError:
            pass
    grants = json.loads(raw) if raw else {}
    if not isinstance(grants, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in grants.items()
    ):
        raise ValueError("skills_recovery_historical_grants_unavailable")
    first = _snapshots(service)
    if first != _snapshots(service):
        raise ValueError("skills_recovery_scan_changed")
    if (parent, missing) != _parent_plan(service):
        raise ValueError("skills_recovery_review_changed")
    return RecoveryReview(
        binding,
        tuple(_identity(path) for path in _sources(service)),
        parent,
        missing,
        first,
        tuple(sorted(grants)),
        hashlib.sha256(raw).hexdigest(),
    )


def capture(service):
    with observed(_sources(service)) as witnesses:
        if not witnesses:
            raise ValueError("skills_recovery_not_required")
        return _review(service, witnesses)


def approve(service, review, passphrase):
    from tldw_chatbook.Backup_Recovery.activation import _flush_existing, _write
    from tldw_chatbook.Backup_Recovery.native_files import (
        _flush_private_tree,
        create_private_directory,
    )

    if not isinstance(passphrase, str) or not passphrase:
        raise ValueError("skills_recovery_passphrase_required")
    with observed(_sources(service)) as witnesses:
        if (
            not witnesses
            or type(review) is not RecoveryReview
            or _review(service, witnesses) != review
        ):
            raise ValueError("skills_recovery_review_changed")
        binding, root = _selection(service, witnesses)
        if lexical_path(service.trust_store.store_dir) != root:
            raise ValueError("skills_recovery_root_changed")
        existing = []
        for witness in witnesses:
            store = ActivationStore(Path(witness["store_root"]))
            with (
                _private(store.root),
                _private(store._generation(witness["generation"])) as parent,
            ):
                if (
                    "skills"
                    not in store._required(parent, witness["generation"]).owners
                ):
                    raise ValueError("skills_recovery_owner_not_required")
                try:
                    record = RootBinding.model_validate(
                        bootstrap._read(parent, _name(binding))
                    )
                except FileNotFoundError:
                    record = None
                existing.append((store, witness, record))
        if any(record is not None for _, _, record in existing):
            # A matching durable first binding identifies our interrupted setup.
            # Validate every existing witness; only its missing peers may be filled.
            for _, witness, record in existing:
                if record is not None:
                    _check_record(record, binding, root, witness)
            service.unlock_with_passphrase(passphrase)
        else:
            # A foreign or interrupted unbound root is evidence, not reusable
            # scratch space. Never overwrite it or call reset_trust on history.
            created = {}
            for name in review.missing_trust_parents:
                if (
                    _identity(Path(review.existing_trust_parent[0]))
                    != review.existing_trust_parent
                ):
                    raise ValueError("skills_recovery_review_changed")
                create_private_directory(Path(name))
                created[name] = _identity(Path(name))
            if created:
                current = _review(service, witnesses)
                expected_sources = tuple(
                    created.get(row[0], row) for row in review.source_identities
                )
                expected = replace(
                    review,
                    binding=current.binding,
                    source_identities=expected_sources,
                    existing_trust_parent=created[review.missing_trust_parents[-1]],
                    missing_trust_parents=(),
                )
                if (
                    current != expected
                    or _identity(Path(review.existing_trust_parent[0]))
                    != review.existing_trust_parent
                ):
                    raise ValueError("skills_recovery_review_changed")
                review = current
                binding, root = _selection(service, witnesses)
            create_private_directory(root)
            service._bootstrap_trust(passphrase)
        manifest = service._load_valid_manifest()
        expected = {
            snapshot.skill_name: [
                item.as_manifest_entry() for item in snapshot.fingerprints
            ]
            for snapshot in review.skills
        }
        actual = {name: row["files"] for name, row in manifest["skills"].items()}
        if actual != expected or _review(service, witnesses) != review:
            raise ValueError("skills_recovery_review_changed")
        with _private(root) as fd:
            info = os.fstat(fd)
            _flush_private_tree(fd, info.st_dev)
        for store, witness, record in existing:
            bound = RootBinding(
                binding=binding,
                generation=witness["generation"],
                root=str(root),
                device=info.st_dev,
                inode=info.st_ino,
            )
            with (
                _private(store.root),
                _private(store._generation(witness["generation"])) as parent,
            ):
                if record is None:
                    _write(parent, _name(binding), bound)
                else:
                    _flush_existing(parent, _name(binding), bound)
        for store, witness, _ in existing:
            store.approve(witness["generation"], "skills")
