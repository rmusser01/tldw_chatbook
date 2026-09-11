"""Locally issued fixed recovery associations; never an archive import API."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .admission import Admission, AdmissionTimeout, fcntl
from .bootstrap import (
    RecoveryRequired,
    _activation_witness,
    _binding,
    _control_records,
    _fingerprint,
    _key,
    _overlap,
    _read,
    _records,
    _registry,
)
from .native_files import (
    create_private_directory,
    flush_directory,
    pinned_directory,
    publish_new,
)
from .profile_paths import lexical_path
from .qualification import qualified_for

UNBOUND_NAMESPACE = "bootstrap.unbound"


class _Pending(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    operation_id: str = Field(min_length=1, max_length=256)
    namespaces: list[str] = Field(min_length=1, max_length=4096)
    control_root: str
    selectors: list[str] = Field(min_length=1, max_length=4096)


class _Profile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    selector: str
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    namespaces: list[str] = Field(min_length=1, max_length=4096)
    roots: list[str] = Field(min_length=1, max_length=4096)
    activation: dict | None = None

    @field_validator("activation")
    @classmethod
    def _validate_activation(cls, value):
        _activation_witness(value)
        return value


@contextmanager
def _enrollment(authority: Admission, *, session=None, names=()):
    if session is not None:
        from .storage_admission import MaintenanceSession

        if type(session) is not MaintenanceSession:
            raise ValueError("enrollment_session_required")
        session._check()
        if (
            session._control != authority.control_root
            or session._control_identity != authority._identity
            or not {UNBOUND_NAMESPACE, *names} <= set(session._names)
        ):
            raise ValueError("enrollment_session_scope")
        yield
        return
    try:
        with authority.maintenance((UNBOUND_NAMESPACE,), 0.1):
            yield
    except AdmissionTimeout:
        raise RecoveryRequired("close_unenrolled_clients_and_restart") from None


def _ensure(root: Path) -> None:
    """Registration only: create missing private ancestors, never fix existing ones."""
    if not root.exists():
        _ensure(root.parent)
        try:
            create_private_directory(root)
        except FileExistsError:
            pass
    with pinned_directory(root) as parent:
        if os.fstat(parent).st_uid != os.geteuid() or os.fstat(parent).st_mode & 0o077:
            # Ancestors may be normal trusted home/config dirs; only final bootstrap
            # is checked private by _records, not chmodded during registration.
            return


def _write(root: Path, name: str, data: bytes) -> None:
    if len(data) > 1048576:
        raise ValueError("record_too_large")
    with pinned_directory(root) as parent:
        # Publish exclusively in place: an incomplete/failed write is itself a fence.
        Admission._write_new_record(parent, name, data)
        flush_directory(parent)


def admission_authority(bootstrap_root: Path) -> Admission:
    """Initialize only qualified fixed authority, outside every replacement target."""
    existing = bootstrap_root.parent
    while not existing.exists():
        existing = existing.parent
    allowed, reason = qualified_for("admission", existing)
    if not allowed:
        raise RecoveryRequired(reason)
    _ensure(bootstrap_root)
    _records(bootstrap_root)
    marker = bootstrap_root / "unbound-owner"
    created_marker = False
    if not marker.exists():
        if (bootstrap_root / "admission").exists():
            raise RecoveryRequired("recovery_scope_uncertain")
        try:
            _write(bootstrap_root, marker.name, b"local enrollment owner\n")
            created_marker = True
        except FileExistsError:
            pass
    if created_marker:
        authority = Admission(bootstrap_root / "admission")
        authority.register(UNBOUND_NAMESPACE, (marker,))
        return authority
    return _existing_admission_authority(bootstrap_root)


def _existing_admission_authority(bootstrap_root):
    """Open existing native authority for the operation-bound recovery executor."""
    marker = bootstrap_root / "unbound-owner"
    try:
        authority = Admission.open_existing(bootstrap_root / "admission")
        with authority._directory() as parent:
            with authority._lock(parent, "registry.lock", fcntl.LOCK_SH):
                registry = authority._read(parent)
                # Validate the marker's private regular-file posture and the exact
                # registered physical identity. Never repair missing/replaced state.
                _registry(bootstrap_root)
                entry = registry.entries.get(UNBOUND_NAMESPACE)
                if (
                    entry is None
                    or entry.roots != [str(marker)]
                    or entry.pending is not None
                    or entry.proposed
                    or not authority._tokens((marker,)) <= set(entry.historical)
                ):
                    raise RecoveryRequired("recovery_scope_uncertain")
        return authority
    except (OSError, ValueError, RuntimeError):
        raise RecoveryRequired("recovery_scope_uncertain") from None


def _verify_pending_selector(selector: Path) -> None:
    """Verify existing bytes or genuine absence for a local pre-publication fence."""
    try:
        selector.lstat()
    except FileNotFoundError:
        pass
    else:
        with pinned_directory(selector.parent):
            _fingerprint(selector)  # Existing bytes stay strict; no TOML parsing.
        return

    ancestor = selector
    missing = None
    while True:
        try:
            ancestor.lstat()
            break
        except FileNotFoundError:
            missing = ancestor.name
            ancestor = ancestor.parent
    # This also rejects links (including dangling links) and non-directories
    # in the existing prefix. Never create destination directories to enroll it.
    with pinned_directory(ancestor) as parent:
        if missing is not None:
            try:
                os.stat(missing, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return
    raise ValueError("selector_absence_changed")


def register_pending(
    bootstrap_root: Path,
    operation_id: str,
    namespaces: tuple[str, ...],
    control_root: Path,
    selectors: tuple[Path, ...],
) -> None:
    """Durably fence local selectors before publication; never clears evidence."""
    names = Admission._names(namespaces)
    if type(selectors) is not tuple or not selectors:
        raise ValueError("invalid_selectors")
    selected = tuple(lexical_path(p) for p in selectors)
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate_selectors")
    for selector in selected:
        _verify_pending_selector(selector)
        if _overlap(selector, bootstrap_root) or _overlap(selector, control_root):
            raise ValueError("control_root_overlaps_target")
    registry = _registry(bootstrap_root)
    if registry is not None:
        for namespace in names:
            if namespace not in registry:
                raise ValueError("namespace_unregistered")
            for raw in registry[namespace]["roots"]:
                if _overlap(Path(raw), bootstrap_root) or _overlap(
                    Path(raw), control_root
                ):
                    raise ValueError("control_root_overlaps_target")
    record = _Pending(
        operation_id=operation_id,
        namespaces=list(names),
        control_root=str(lexical_path(control_root)),
        selectors=[str(p) for p in selected],
    )
    existing = bootstrap_root.parent
    while not existing.exists():
        existing = existing.parent
    allowed, reason = qualified_for("admission", existing)
    if not allowed:
        raise RecoveryRequired(reason)
    _ensure(bootstrap_root)
    _records(bootstrap_root)
    _write(
        bootstrap_root,
        "pending-" + _key(operation_id) + ".json",
        record.model_dump_json().encode(),
    )


def bind_profile(
    bootstrap_root: Path,
    config_selector: Path,
    namespaces: tuple[str, ...],
    authority_root: Path,
    *,
    session=None,
) -> None:
    """Enroll an intact local mapping only after unbound owners have retired.

    Namespaces must already be installed in the fixed authority. This operation
    neither parses an archive nor chooses storage from serialized archive locators.
    Changed mappings require an explicit future reconciliation/refresh operation.
    """
    if lexical_path(authority_root) != lexical_path(bootstrap_root / "admission"):
        raise RecoveryRequired("conflicting_admission_authority")
    authority = admission_authority(bootstrap_root)
    selected = lexical_path(config_selector)
    names = Admission._names(namespaces)
    with _enrollment(authority, session=session, names=names):
        pending, profiles = _records(bootstrap_root)
        if pending:
            raise RecoveryRequired("recovery_pending")
        registry = _registry(bootstrap_root)
        if registry is None or any(n not in registry for n in names):
            raise ValueError("namespace_unregistered")
        roots = sorted({p for n in names for p in registry[n]["roots"]})
        if any(_overlap(Path(p), bootstrap_root) for p in roots + [str(selected)]):
            raise ValueError("control_root_overlaps_target")
        if not any(_overlap(selected, Path(p)) for p in roots):
            raise ValueError("selector_not_in_admission_scope")
        record = {
            "version": 1,
            "selector": str(selected),
            "fingerprint": _fingerprint(selected),
            "namespaces": list(names),
            "roots": roots,
        }
        if _binding(selected, [record], registry) is None:
            raise ValueError("binding_unverified")
        validated = _Profile.model_validate(record)

        _write(
            bootstrap_root,
            "profile-" + _key(str(selected)) + ".json",
            validated.model_dump_json(exclude_none=True).encode(),
        )


def _activation_record_identity(parent, name, expected):
    """Capture only the exact checked record or verified absence."""
    try:
        identity = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        if expected is not None:
            raise ValueError("activation_record_changed") from None
        return None
    if (
        expected is None
        or _read(parent, name) != expected
        or os.stat(name, dir_fd=parent, follow_symlinks=False) != identity
    ):
        raise ValueError("activation_record_changed")
    return identity


def _publish_activation_record(root, parent, name, before, after, temporary, identity):
    """Replace only the checked captured record, retaining failed-write evidence."""
    if _activation_record_identity(parent, name, before) != identity:
        raise ValueError("activation_record_changed")
    Admission._write_new_record(parent, temporary, json.dumps(after).encode())
    if before is None:
        info = os.fstat(parent)
        publish_new(
            root / temporary,
            root / name,
            parent_identities=((info.st_dev, info.st_ino),) * 2,
        )
    else:
        if (
            _read(parent, name) != before
            or os.stat(name, dir_fd=parent, follow_symlinks=False) != identity
        ):
            raise ValueError("activation_record_changed")
        os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        flush_directory(parent)
    if _read(parent, name) != after:
        raise ValueError("activation_publication_changed")


def _bind_activation(
    bootstrap_root, operation_id, config_selector, generation, owners, session
):
    """Install paired generation evidence while exact native maintenance is held.

    No namespace remap, enrollment refresh, journal commit or fence clearance is
    performed here. The pending operation and an explicit before/after intent
    retain ambiguous writes for the eventual recovery executor.
    """
    from .activation import ActivationStore, _identifier, _private
    from .journal import Journal
    from .storage_admission import MaintenanceSession, _contains_owned_path

    if type(session) is not MaintenanceSession:
        raise ValueError("maintenance_session_required")
    MaintenanceSession._check(session)
    root = lexical_path(bootstrap_root)
    selected = lexical_path(config_selector)
    if lexical_path(session._control) != root / "admission":
        raise ValueError("conflicting_admission_authority")
    with pinned_directory(session._control) as parent:
        info = os.fstat(parent)
        if (info.st_dev, info.st_ino) != session._control_identity:
            raise ValueError("activation_authority_changed")
    pending, profiles, associations = _control_records(root)
    operation = next((p for p in pending if p["operation_id"] == operation_id), None)
    if (
        operation is None
        or str(selected) not in operation["selectors"]
        or not set(operation["namespaces"]) <= set(session._names)
    ):
        raise ValueError("activation_pending_scope_mismatch")
    registry = _registry(root)
    names = operation["namespaces"]
    previous = next((p for p in profiles if p["selector"] == str(selected)), None)
    if previous is not None:
        names = previous["namespaces"]
    if (
        registry is None
        or not set(names) <= set(operation["namespaces"])
        or any(n not in registry for n in names)
    ):
        raise ValueError("activation_profile_scope_mismatch")
    roots = sorted({p for n in names for p in registry[n]["roots"]})
    if previous is not None and previous["roots"] != roots:
        raise ValueError("activation_profile_mapping_changed")
    if not any(_contains_owned_path(Path(p), selected) for p in roots) or any(
        not any(_contains_owned_path(r, Path(p)) for r in session._roots) for p in roots
    ):
        raise ValueError("activation_profile_outside_maintenance")
    control = Path(operation["control_root"])
    if any(
        _overlap(Path(p), r) for p in roots + [str(selected)] for r in (root, control)
    ):
        raise ValueError("activation_control_overlaps_profile")
    with _private(control):
        pass
    allowed, reason = qualified_for("admission", root)
    if not allowed:
        raise ValueError(reason)
    if type(owners) is not tuple or not owners or len(owners) > 4096:
        raise ValueError("activation_owners_invalid")
    witness = {
        "operation_id": _identifier(operation_id),
        "generation": _identifier(generation),
        "owners": sorted({_identifier(owner) for owner in owners}),
        "namespaces": sorted(names),
        "store_root": str(control / "activation"),
    }
    _activation_witness(witness)
    prior_association = next(
        (a for a in associations if a["selector"] == str(selected)), None
    )
    old_witness = previous.get("activation") if previous else None
    if old_witness != (prior_association["activation"] if prior_association else None):
        raise ValueError("activation_pair_inconsistent")
    if old_witness and (
        old_witness["operation_id"] == operation_id
        and old_witness != witness
        and not _rollback_activation_successor(
            control, operation_id, old_witness, witness, selected
        )
        or old_witness["operation_id"] != operation_id
        and old_witness["generation"] == generation
    ):
        raise ValueError("activation_generation_conflict")
    store = ActivationStore(control / "activation")
    if (
        store._generation(generation).exists()
        and not any(a["activation"] == witness for a in associations)
        and not _operation_activation_generation(
            control, operation_id, witness, selected
        )
    ):
        raise ValueError("activation_generation_already_used")
    profile = _Profile(
        selector=str(selected),
        fingerprint=_fingerprint(selected),
        namespaces=list(names),
        roots=roots,
        activation=witness,
    ).model_dump(exclude_none=True)
    association = {"version": 1, "selector": str(selected), "activation": witness}
    key = _key(str(selected))
    profile_name, association_name = (
        "profile-" + key + ".json",
        "activation-" + key + ".json",
    )
    with _private(root) as parent:
        root_info = os.fstat(parent)
        identities = {
            name: _activation_record_identity(parent, name, record)
            for name, record in (
                (profile_name, previous),
                (association_name, prior_association),
            )
        }
    store.require(generation, tuple(witness["owners"]))
    with _private(root) as parent:
        MaintenanceSession._check(session)
        info = os.fstat(parent)
        if (info.st_dev, info.st_ino) != (root_info.st_dev, root_info.st_ino):
            raise ValueError("activation_authority_changed")
        for name, record in (
            (profile_name, previous),
            (association_name, prior_association),
        ):
            if _activation_record_identity(parent, name, record) != identities[name]:
                raise ValueError("activation_record_changed")
        if _read(parent, "pending-" + _key(operation_id) + ".json") != operation:
            raise ValueError("activation_pending_changed")
        if previous == profile and prior_association == association:
            for name, record in (
                (profile_name, profile),
                (association_name, association),
            ):
                Journal._flush_record(parent, name, record)
            flush_directory(parent)
            return store.root
        intent_name = "activation-update-" + key + ".json"
        intent = {
            "version": 1,
            "operation_id": operation_id,
            "selector": str(selected),
            "before": [previous, prior_association],
            "after": [profile, association],
        }
        encoded = json.dumps(intent).encode()
        if len(encoded) > 1048576:
            raise ValueError("activation_update_too_large")
        Admission._write_new_record(parent, intent_name, encoded)
        flush_directory(parent)
        for index, (name, before, after) in enumerate(
            (
                (association_name, prior_association, association),
                (profile_name, previous, profile),
            )
        ):
            _publish_activation_record(
                root,
                parent,
                name,
                before,
                after,
                f"activation-stage-{key}-{index}.json",
                identities[name],
            )
        if (
            _read(parent, profile_name) != profile
            or _read(parent, association_name) != association
            or _read(parent, intent_name) != intent
            or _read(parent, "pending-" + _key(operation_id) + ".json") != operation
        ):
            raise ValueError("activation_publication_changed")
        MaintenanceSession._check(session)
        os.unlink(intent_name, dir_fd=parent)
        flush_directory(parent)
    return store.root


def _rollback_activation_successor(
    control, operation_id, previous, requested, selector
):
    """Verify the same operation's durable reverse generation under held admission.

    The executor retains its journal lock during activation; read its pinned chain
    directly here instead of trying to acquire the same non-reentrant flock again.
    """
    from .journal import Journal, _matches, _Object, _Prepared
    from .owner_registry import install_adapters

    path = control / ("operation-" + _key(operation_id))
    if not path.is_dir():
        return False
    journal = Journal(control, operation_id)
    with pinned_directory(journal.root) as parent:
        rows = journal._records(parent)
    from .rollback_credentials import current_phase

    phase = current_phase(rows)
    start = next((row for row in rows if row.event == "rollback_started"), None)
    validated = next(
        (row for row in reversed(phase) if row.event == "originals_validated"), None
    )
    if (
        start is None
        or validated is None
        or rows[-1].event
        not in {"originals_validated", "rollback_activation_recorded", "rolled_back"}
    ):
        return False
    prepared = _Prepared.model_validate(
        next(row.evidence for row in rows if row.event == "prepared")
    )
    return (
        previous["generation"] == prepared.generation
        and requested["generation"] == start.evidence["generation"]
        and str(selector) in {row["config"] for row in start.evidence["profiles"]}
        and requested["owners"]
        == sorted(
            owner.owner_id for owner in install_adapters() if owner.activation_required
        )
        and all(
            _matches(_Object.model_validate(row), row["path"], metadata=True)
            for row in validated.evidence["artifacts"]
        )
    )


class _ActivationUpdate(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: int = Field(ge=1, le=1)
    operation_id: str = Field(min_length=1, max_length=256)
    selector: str
    before: list[dict | None] = Field(min_length=2, max_length=2)
    after: list[dict] = Field(min_length=2, max_length=2)


def _operation_activation_generation(control, operation_id, witness, selector):
    """Allow only a generation already named by actual installed local evidence."""
    from .journal import Journal, _Prepared
    from .owner_registry import install_adapters

    if not (control / ("operation-" + _key(operation_id))).is_dir():
        return False
    journal = Journal(control, operation_id)
    with pinned_directory(journal.root) as parent:
        rows = journal._records(parent)
    prepared = _Prepared.model_validate(
        next(row.evidence for row in rows if row.event == "prepared")
    )
    rolling = next((row for row in rows if row.event == "rollback_started"), None)
    generation = rolling.evidence["generation"] if rolling else prepared.generation
    validated = "originals_validated" if rolling else "installed_validated"
    from .rollback_credentials import current_phase

    phase = current_phase(rows) if rolling else rows
    return (
        (
            prepared.mode == "replace" and bool(prepared.replacement_profiles)
            or prepared.mode == "isolated"
            and rolling is None
            and any(row.config == str(selector) for row in prepared.isolated_profiles)
            and witness["namespaces"] == prepared.publication.namespaces
        )
        and any(row.event == validated for row in phase)
        and witness["generation"] == generation
        and witness["operation_id"] == operation_id
        and witness["store_root"] == str(control / "activation")
        and str(selector) in prepared.publication.selectors
        and set(witness["namespaces"]) <= set(prepared.publication.namespaces)
        and witness["owners"]
        == sorted(
            owner.owner_id for owner in install_adapters() if owner.activation_required
        )
    )


def _recovery_pending(root, journal, prepared):
    """Read the exact fixed pointer even while paired activation reads are fenced."""
    expected = _Pending(
        operation_id=journal.operation_id,
        control_root=str(journal.root.parent),
        namespaces=prepared.publication.namespaces,
        selectors=prepared.publication.selectors,
    )
    with pinned_directory(root) as parent:
        record = _Pending.model_validate(
            _read(parent, "pending-" + _key(journal.operation_id) + ".json")
        )
    if record != expected:
        raise ValueError("publication_pending_mismatch")
    return record


def _recover_activation_pairs(journal, prepared, session):
    """Finish exact local before/after pairs; ordinary readers never repair them."""
    from .activation import ActivationStore
    from .publication import _finalization_session

    root = Path(prepared.publication.bootstrap_root)
    _finalization_session(session, prepared.publication, prepared)
    pending = _recovery_pending(root, journal, prepared)
    registry = _registry(root)
    with journal._locked(exclusive=True), pinned_directory(root) as parent:
        names = sorted(
            name for name in os.listdir(parent) if name.startswith("activation-update-")
        )
        for name in names:
            update = _ActivationUpdate.model_validate(_read(parent, name))
            selector = Path(update.selector)
            key = _key(update.selector)
            after_profile = _Profile.model_validate(update.after[0]).model_dump(
                exclude_none=True
            )
            after_association = update.after[1]
            witness = after_profile.get("activation")
            if (
                name != "activation-update-" + key + ".json"
                or update.operation_id != journal.operation_id
                or update.selector not in pending.selectors
                or after_profile["selector"] != update.selector
                or after_profile["fingerprint"] != _fingerprint(selector)
                or after_association
                != {"version": 1, "selector": update.selector, "activation": witness}
                or not _operation_activation_generation(
                    journal.root.parent, journal.operation_id, witness, selector
                )
                or after_profile["namespaces"] != witness["namespaces"]
                or after_profile["roots"]
                != sorted(
                    {
                        path
                        for scope in witness["namespaces"]
                        for path in registry[scope]["roots"]
                    }
                )
            ):
                raise ValueError("activation_recovery_context_invalid")
            if update.before[0] is not None:
                previous = _Profile.model_validate(update.before[0]).model_dump(
                    exclude_none=True
                )
                if (
                    previous["selector"] != update.selector
                    or previous["namespaces"] != after_profile["namespaces"]
                    or previous["roots"] != after_profile["roots"]
                ):
                    raise ValueError("activation_recovery_context_invalid")
                old = previous.get("activation")
            else:
                old = None
            if update.before[1] != (
                None
                if old is None
                else {"version": 1, "selector": update.selector, "activation": old}
            ):
                raise ValueError("activation_pair_inconsistent")
            store = ActivationStore(Path(witness["store_root"]))
            with pinned_directory(
                store._generation(witness["generation"])
            ) as generation_parent:
                if (
                    store._required(generation_parent, witness["generation"]).owners
                    != witness["owners"]
                ):
                    raise ValueError("activation_recovery_context_invalid")
            intent_identity = os.stat(name, dir_fd=parent, follow_symlinks=False)
            for index, prefix in ((1, "activation-"), (0, "profile-")):
                target = prefix + key + ".json"
                temporary = "activation-stage-" + key + "-" + str(1 - index) + ".json"
                _resume_activation_record(
                    root,
                    parent,
                    target,
                    temporary,
                    update.before[index],
                    update.after[index],
                )
            _finalization_session(session, prepared.publication, prepared)
            _recovery_pending(root, journal, prepared)
            if (
                _read(parent, name) != update.model_dump()
                or os.stat(name, dir_fd=parent, follow_symlinks=False)
                != intent_identity
            ):
                raise ValueError("activation_record_changed")
            os.unlink(name, dir_fd=parent)
            flush_directory(parent)
    _control_records(root)


def _resume_activation_record(root, parent, name, temporary, before, after):
    """Reconcile an actual paired write using only its durable exact JSON states."""
    try:
        current = _read(parent, name)
    except FileNotFoundError:
        current = None
    if current not in (before, after):
        raise ValueError("activation_record_changed")
    identity = _activation_record_identity(parent, name, current)
    try:
        staged = _read(parent, temporary)
    except FileNotFoundError:
        staged = None
    if staged is not None and staged != after:
        raise ValueError("activation_record_changed")
    staged_identity = (
        os.stat(temporary, dir_fd=parent, follow_symlinks=False)
        if staged is not None
        else None
    )
    if current != after:
        if staged is None:
            Admission._write_new_record(parent, temporary, json.dumps(after).encode())
        if _activation_record_identity(parent, name, before) != identity:
            raise ValueError("activation_record_changed")
        if _read(parent, temporary) != after or (
            staged_identity is not None
            and os.stat(temporary, dir_fd=parent, follow_symlinks=False)
            != staged_identity
        ):
            raise ValueError("activation_record_changed")
        if before is None:
            info = os.fstat(parent)
            publish_new(
                root / temporary,
                root / name,
                parent_identities=((info.st_dev, info.st_ino),) * 2,
            )
        else:
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        flush_directory(parent)
    elif staged is not None:
        if (
            _read(parent, temporary) != after
            or os.stat(temporary, dir_fd=parent, follow_symlinks=False)
            != staged_identity
        ):
            raise ValueError("activation_record_changed")
        os.unlink(temporary, dir_fd=parent)
        flush_directory(parent)
    if _read(parent, name) != after:
        raise ValueError("activation_record_changed")
