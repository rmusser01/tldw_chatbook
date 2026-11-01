"""Fixed unchanged installed builtin members; never restoration authority."""

from contextlib import contextmanager

from .mobile_preview_budget import path_cost

_OWNER = "persona.visual_identity_builtin"


def _preservation_budget(_read_budget=None):
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    from .mobile_preview_budget import current

    scope = current()
    if _read_budget is not None:
        if type(_read_budget) is not _TraceReadBudget:
            raise TypeError("invalid_mobile_preservation_budget")
        if scope is not None and scope.read_budget is not _read_budget:
            raise ValueError("mobile_preview_budget_changed")
        return _read_budget
    return scope.read_budget if scope is not None else _TraceReadBudget()


def _ordinary_builtin_members(document, target, budget):
    """Match native archive topology to the actual current installed package.

    This pure mapping grants no source read or publication authority. Public
    preview and admitted replacement validation retain their distinct gates.
    """
    from tldw_chatbook.Persona_Visual import recovery as assets

    from .archive_models import ArchiveManifest
    from .models import Inventory

    if (
        type(document) is not ArchiveManifest
        or type(target) is not Inventory
        or not target.complete
    ):
        raise ValueError("mobile_canonical_current_target_required")
    budget.reserve(8192 + 16 * len(assets.__file__))
    owner = next(row for row in assets.recovery_adapters() if row.owner_id == _OWNER)
    package = owner._root({})
    budget.container(
        5
        * (
            len(document.producer_inventory)
            + len(document.directories)
            + len(document.files)
            + len(target.items)
        )
    )
    producers = {
        row.logical_id: row
        for row in document.producer_inventory
        if row.owner_id == _OWNER and row.status in {"included", "included_directory"}
    }
    if not producers:
        return {}
    records = {row.logical_id: row for row in (*document.directories, *document.files)}
    directories = {row.logical_id: row for row in document.directories}
    current = {item.logical_id: item for item in target.items}
    roots, matched = {}, {}
    for key, producer in producers.items():
        item, record = current.get(key), records.get(key)
        if item is None or record is None or item.owner != _OWNER or item.path is None:
            raise ValueError("mobile_canonical_builtin_members_changed")
        meta = item.metadata
        if (
            meta is None
            or item.status != producer.status
            or item.dependencies != producer.dependencies
            or item.shared_group != producer.shared_group
            or record.metadata is None
            or (
                meta.root_id,
                meta.parent_id,
                meta.relative_path,
                meta.mode,
                meta.mtime_ns,
            )
            != (
                record.root_id,
                record.parent_id,
                record.relative_path,
                record.metadata.mode,
                record.metadata.mtime_ns,
            )
            or meta.kind != ("directory" if key in directories else "file")
            or key not in directories
            and record.owner_id != _OWNER
        ):
            raise ValueError("mobile_canonical_builtin_members_changed")
        budget.value(meta.root_id)
        budget.container(4)
        parts = meta.root_id.split(":")
        root = directories.get(meta.root_id)
        root_item = current.get(meta.root_id)
        root_producer = producers.get(meta.root_id)
        if (
            len(parts) != 3
            or parts[0] != "profile"
            or parts[1] not in document.profile_ids
            or parts[2] != _OWNER
            or root is None
            or root.synthetic
            or root.parent_id is not None
            or root.relative_path != ""
            or root.root_id != root.logical_id
            or root_item is None
            or root_item.path != package
            or root_producer is None
            or root_producer.status != "included_directory"
        ):
            raise ValueError("mobile_canonical_builtin_history_unsupported")
        budget.reserve(path_cost(package, copies=2) + 16 * len(meta.relative_path))
        if item.path != package / meta.relative_path:
            raise ValueError("mobile_canonical_builtin_members_changed")
        if key != meta.root_id:
            parent = current.get(meta.parent_id)
            if (
                parent is None
                or parent.owner != _OWNER
                or parent.status != "included_directory"
                or parent.metadata is None
                or parent.metadata.root_id != meta.root_id
                or parent.path != item.path.parent
                or meta.parent_id not in producers
            ):
                raise ValueError("mobile_canonical_builtin_members_changed")
        roots[meta.root_id] = root_item
        matched[key] = item
    # Additional current referenced members are a changed native closure, not
    # historical evidence that this archive may silently preserve.
    budget.container(len(target.items) + len(matched))
    actual = {
        item.logical_id
        for item in target.items
        if item.owner == _OWNER
        and item.metadata is not None
        and item.metadata.root_id in roots
        and item.status in {"included", "included_directory"}
    }
    if actual != set(matched):
        raise ValueError("mobile_canonical_builtin_members_changed")
    return matched


def _verify_current_builtin_files(document, members, budget, *, cancel=None):
    """Read current bytes through the installed source admission and digest."""
    from threading import Event

    from .storage_admission import _digest_recovery_file

    budget.reserve(1024)
    if cancel is None:
        cancel = Event()
    for record in document.files:
        item = members.get(record.logical_id)
        if item is None:
            continue
        # The native limit must stay positive even for an authenticated empty
        # member. Equality below rejects any growth through that EOF probe.
        actual = _digest_recovery_file(
            item.owner,
            item.path,
            max_bytes=max(1, record.size),
            cancel=cancel,
            _read_budget=budget,
        )
        if actual != (record.size, record.sha256):
            raise ValueError("mobile_canonical_builtin_content_changed")


def _member_observation(item, record, budget, custody, cancel, read_buffer):
    """Use native framed file proof or the native metadata-only directory proof."""
    import stat

    from tldw_chatbook.Utils.platform_files import os

    from .archive_reader import _check
    from .journal import (
        _artifact_component_limit,
        _ArtifactObservationLimits,
        _evidence_digest,
        observe_artifact,
    )
    from .mobile_runs_goal_budget import _serialization
    from .publication import _observe_safety_source

    _check(cancel)
    custody.scope.check()
    # Fixed stat/Object fields, path/parent traversal and metadata-only directory
    # serialization. Native stat integer widths are bounded; no file text enters
    # the directory hash and the path is excluded from its JSON payload.
    budget.reserve(16384 + path_cost(item.path, copies=16 + len(item.path.parts)))
    info = os.stat(item.path, follow_symlinks=False)
    directory = item.status == "included_directory"
    if (
        item.metadata is None
        or record.metadata is None
        or (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
        is False
        or stat.S_IMODE(info.st_mode) != record.metadata.mode
        or info.st_mtime_ns != record.metadata.mtime_ns
        or item.metadata.mode != record.metadata.mode
        or item.metadata.mtime_ns != record.metadata.mtime_ns
    ):
        raise ValueError("mobile_canonical_builtin_metadata_changed")
    if directory:
        framing = {
            "safety_directory_metadata": {
                "device": info.st_dev,
                "inode": info.st_ino,
                "mode": stat.S_IMODE(info.st_mode),
                "mtime_ns": info.st_mtime_ns,
                "ctime_ns": info.st_ctime_ns,
            }
        }
        _serialization(framing, budget, encoded=True)
        _serialization(framing, budget, encoded=True)
        expected = _evidence_digest(framing)
        actual = _observe_safety_source(item.path)
        _check(cancel)
        custody.scope.check()
        if (
            actual["sha256"] != expected
            or actual["device"] != info.st_dev
            or actual["inode"] != info.st_ino
        ):
            raise ValueError("mobile_canonical_builtin_metadata_changed")
        return actual
    if info.st_size != record.size:
        raise ValueError("mobile_canonical_builtin_content_changed")
    with custody.pinned_directory(item.path.parent) as parent:
        component = _artifact_component_limit(parent)
    budget.reserve(1024 + 512 * (record.size // 65536 + 3))
    # This is exactly the native single-file metadata framing; price its JSON
    # and encoding before observe_artifact creates either representation.
    framing = (
        (
            "",
            "file",
            record.size,
            record.sha256,
            stat.S_IMODE(info.st_mode),
            info.st_mtime_ns,
        ),
    )
    # Native observation creates its own rows/encoding; expected evidence is a
    # second actual serialization/hash, independently priced before either call.
    _serialization(framing, budget, encoded=True)
    _serialization(framing, budget, encoded=True)
    expected = _evidence_digest(framing)
    limits = _ArtifactObservationLimits(1, record.size, 0, 0, 0, component)
    actual = observe_artifact(
        item.path,
        metadata=True,
        _custody=custody,
        _limits=limits,
        _read_buffer=read_buffer,
    )
    _check(cancel)
    custody.scope.check()
    if (
        actual["sha256"] != expected
        or actual["device"] != info.st_dev
        or actual["inode"] != info.st_ino
    ):
        raise ValueError("mobile_canonical_builtin_content_changed")
    return actual


def _mobile_source_members(
    plan, proof, document, manifest_digest, session, *, _read_budget=None, _custody=None
):
    """Fixed authenticated mobile-copy sources under actual new native admission.

    Public listing/preview keeps its strict no-pending wrapper. These two fixed
    native callers already bind their acquired archive or installed receipt to
    this plan and manifest, and hold the new replacement's exclusion.
    """
    from pathlib import Path

    from . import bootstrap
    from .control_records import UNBOUND_NAMESPACE
    from .mobile_canonical_copy import _CanonicalCopy
    from .models import Inventory
    from .restore_plan import RestorePlan
    from .storage_admission import (
        MaintenanceSession,
        _CaptureFileDescriptors,
    )

    if (
        type(plan) is not RestorePlan
        or type(proof) is not _CanonicalCopy
        or type(session) is not MaintenanceSession
        or plan.local_snapshot is None
        or plan.mode != "replace"
        or manifest_digest != proof.manifest_digest
        or type(plan.target) is not Inventory
        or not plan.target.complete
    ):
        raise ValueError("local_snapshot_builtin_native_session_required")
    if _custody is not None and (
        type(_custody) is not _CaptureFileDescriptors
        or _custody not in session._scopes
        or _custody.close_failed
    ):
        raise ValueError("mobile_canonical_custody_changed")
    session._check()
    if (
        session._control != bootstrap.default_bootstrap_root() / "admission"
        or UNBOUND_NAMESPACE not in session._names
        or proof.prepared.bootstrap_root != str(bootstrap.default_bootstrap_root())
    ):
        raise ValueError("local_snapshot_builtin_scope_unverified")
    budget = _preservation_budget(_read_budget)
    budget.reserve(2048)
    budget.container(
        5 * len(plan.target.items)
        + len(plan.preserve)
        + len(plan.restore)
        + len(plan.safety_scope)
    )
    current_configs = {
        item.path
        for item in plan.target.items
        if item.owner == "config" and item.path is not None
    }
    budget.value(
        (proof.prepared.source_selector, proof.prepared.target_selector), text_copies=4
    )
    expected_configs = {
        Path(proof.prepared.source_selector),
        Path(proof.prepared.target_selector),
    }
    if current_configs != expected_configs:
        raise ValueError("mobile_canonical_selector_review_required")
    custody = _CaptureFileDescriptors(None) if _custody is None else _custody
    try:
        _, profiles = bootstrap._records(
            bootstrap.default_bootstrap_root(), _budget=budget, _custody=custody
        )
        registry = bootstrap._registry(
            bootstrap.default_bootstrap_root(), _budget=budget, _custody=custody
        )
        for selector in current_configs:
            binding = bootstrap._binding(
                selector, profiles, registry, _budget=budget, _custody=custody
            )
            reviewed = next(
                (
                    row
                    for row in proof.prepared.selector_review
                    if row.selector == str(selector)
                ),
                None,
            )
            if (
                binding is None
                or reviewed is None
                or binding["fingerprint"] != reviewed.fingerprint
                or any(name not in session._names for name in binding["namespaces"])
            ):
                raise ValueError("mobile_canonical_selector_review_required")
    finally:
        if _custody is None:
            custody.retire()
    for store in proof.prepared.targets:
        budget.reserve(512 + 16 * len(store.witnesses[-1].path))
        path = Path(store.witnesses[-1].path)
        if (
            sum(
                item.owner == store.owner_id and item.path == path
                for item in plan.target.items
            )
            != 1
        ):
            raise ValueError("mobile_canonical_store_review_required")
    members = _ordinary_builtin_members(document, plan.target, budget)
    budget.container(len(document.files) + len(document.directories) + 5 * len(members))
    records = {row.logical_id: row for row in (*document.files, *document.directories)}
    budget.container(5)
    return members, records, profiles, registry, budget


@contextmanager
def _mobile_bound_sources(
    plan, proof, document, manifest_digest, work, session, cancel, *, _read_budget=None
):
    """Issue only the existing fixed native source scope after fresh matching."""
    from pathlib import Path

    from . import bootstrap
    from .limits import ArchiveLimits
    from .storage_admission import (
        _CaptureFileDescriptors,
        _CaptureScope,
        _contains_owned_path,
        _local,
    )

    members, records, profiles, registry, budget = _mobile_source_members(
        plan,
        proof,
        document,
        manifest_digest,
        session,
        _read_budget=_read_budget,
    )
    restored = dict(plan.restore)
    files = []
    for key, item in members.items():
        if (
            key not in plan.safety_scope
            or (key, item.path) not in plan.preserve
            or key in restored
            or item.path in restored.values()
            or not any(_contains_owned_path(root, item.path) for root in session._roots)
        ):
            raise ValueError("local_snapshot_builtin_scope_unverified")
        if item.status == "included":
            budget.reserve(1024 + path_cost(item.path, copies=3 + len(item.path.parts)))
            path = item.path.resolve(strict=True)
            info = path.stat()
            files.append((path, info.st_dev, info.st_ino))
    from .archive_reader import _check
    from .later_rollback import _stage_read_sources

    budget.container(
        8 * len(plan.target.items) + len(plan.retire) + len(plan.safety_scope)
    )
    sources = _stage_read_sources(plan)
    for item in sources:
        _check(cancel)
        if item.owner == _OWNER:
            if members.get(item.logical_id) != item:
                raise ValueError("mobile_canonical_builtin_members_changed")
            continue
        # Fixed native plan-derived non-package sources retain positive current
        # profile ownership; the package mapping never grants that ownership.
        if item.path is None or not any(
            _contains_owned_path(root, item.path) for root in session._roots
        ):
            raise ValueError("local_snapshot_builtin_scope_unverified")
        covered = False
        for row in profiles:
            budget.reserve(path_cost(item.path, copies=len(item.path.parts) + 2))
            budget.container(len(row["roots"]))
            for raw in row["roots"]:
                budget.reserve(512 + 32 * len(raw))
            if not any(
                item.path == Path(raw) or Path(raw) in item.path.parents
                for raw in row["roots"]
            ):
                continue
            budget.reserve(512 + 16 * len(row["selector"]))
            budget.reserve(512)
            check_custody = _CaptureFileDescriptors(None)
            try:
                binding = bootstrap._binding(
                    Path(row["selector"]),
                    profiles,
                    registry,
                    _budget=budget,
                    _custody=check_custody,
                )
            finally:
                check_custody.retire()
            if binding is None or any(
                name not in session._names for name in binding["namespaces"]
            ):
                raise ValueError("local_snapshot_builtin_scope_unverified")
            covered = True
        if not covered:
            raise ValueError("local_snapshot_builtin_scope_unverified")
        if item.status == "included":
            budget.reserve(1024 + path_cost(item.path, copies=3 + len(item.path.parts)))
            path = item.path.resolve(strict=True)
            info = path.stat()
            files.append((path, info.st_dev, info.st_ino))
    _check(cancel)
    budget.reserve(1024 + 32 * len(files))
    limits = ArchiveLimits()
    with session._capture_bound_sources(
        tuple(files), work, limits, limits.expanded_bytes
    ):
        scope = _local.capture_scope
        if (
            type(scope) is not _CaptureScope
            or scope.session is not session
            or scope not in session._scopes
        ):
            raise ValueError("local_snapshot_builtin_scope_unverified")
        scope.check()
        budget.reserve(256)
        yield members, records, budget, scope
        scope.check()
    session._check()


def _mobile_builtin_candidates(
    plan, proof, document, manifest_digest, work, session, cancel
):
    """Produce private validation candidates in the fixed native source scope."""
    import hashlib

    from .native_files import create_private_directory
    from .storage_admission import (
        _CaptureFileDescriptors,
        _digest_recovery_file,
        copy_capture_file,
    )

    with _mobile_bound_sources(
        plan, proof, document, manifest_digest, work, session, cancel
    ) as (members, records, budget, scope):
        if not members:
            return {}, {}
        budget.container(2 * len(members))
        candidates, observed = {}, {}
        # Use only the actual scope installed by that native context. It gains
        # neither an accounting field nor a reusable-buffer method.
        budget.reserve(65536 + 1024)
        raw = bytearray(65536)
        view = memoryview(raw)
        read_buffer = (raw, view, [view])
        custody = _CaptureFileDescriptors(scope)
        try:
            for key, item in members.items():
                observed[key] = _member_observation(
                    item, records[key], budget, custody, cancel, read_buffer
                )
            for key, item in members.items():
                budget.reserve(1024 + path_cost(work, copies=3) + 8 * len(key))
                destination = work / (
                    "builtin-" + hashlib.sha256(key.encode()).hexdigest()
                )
                if item.status == "included_directory":
                    create_private_directory(destination)
                else:
                    record = records[key]
                    copy_capture_file(
                        item.owner,
                        item.path,
                        destination,
                        cancel,
                        max_bytes=max(1, record.size),
                        _read_budget=budget,
                    )
                    actual = _digest_recovery_file(
                        item.owner,
                        destination,
                        max_bytes=max(1, record.size),
                        cancel=cancel,
                        _read_budget=budget,
                    )
                    if actual != (record.size, record.sha256):
                        raise ValueError("mobile_canonical_builtin_content_changed")
                candidates[key] = destination
            for key, item in members.items():
                if (
                    _member_observation(
                        item, records[key], budget, custody, cancel, read_buffer
                    )
                    != observed[key]
                ):
                    raise ValueError("local_snapshot_builtin_members_changed")
        finally:
            custody.retire()
    session._check()
    return members, candidates


def _mobile_stage_names(plan, archive, authority, cancel):
    """Acquire only actual current native source names for a sealed mobile copy.

    Called under the existing original-copy lock, before new maintenance. The
    installed package remains a readonly source, never a private profile root.
    """
    from pathlib import Path

    from . import bootstrap
    from .archive_reader import _check
    from .capture_service import _capture_names
    from .later_rollback import _stage_read_sources, verify_snapshot_source
    from .mobile_canonical_copy import (
        _CanonicalCopy,
        _current_canonical_target,
        _verify_canonical_archive,
    )
    from .restore_plan import RestorePlan
    from .storage_admission import _preview_reads

    if (
        type(plan) is not RestorePlan
        or plan.local_snapshot is None
        or plan.mode != "replace"
    ):
        raise ValueError("local_snapshot_source_required")
    budget = _preservation_budget()
    with _preview_reads(_read_budget=budget):
        from .mobile_preview_budget import current

        scope = current()
        _check(cancel)
        _, proof = verify_snapshot_source(plan, cancel=cancel)
        if type(proof) is not _CanonicalCopy:
            raise ValueError("mobile_canonical_archive_required")
        document = _verify_canonical_archive(proof, archive, plan.archive_digest)
        _current_canonical_target(proof, plan.target, _read_budget=budget)
        members = _ordinary_builtin_members(document, plan.target, budget)
        _verify_current_builtin_files(document, members, budget, cancel=cancel)
        budget.container(3 * len(plan.target.items))
        sources = _stage_read_sources(plan)
        root = bootstrap.default_bootstrap_root()
        if authority.control_root.parent != root:
            raise ValueError("local_snapshot_builtin_scope_unverified")
        _, profiles = bootstrap._records(root, _budget=budget, _custody=scope.custody)
        registry = bootstrap._registry(root, _budget=budget, _custody=scope.custody)
        budget.container(len(registry) + len(sources))
        names = set()
        for item in sources:
            _check(cancel)
            if item.owner == _OWNER:
                if members.get(item.logical_id) != item:
                    raise ValueError("mobile_canonical_builtin_members_changed")
                continue
            # Every non-package source retains the original positive native
            # profile-ownership rule. Package membership cannot exempt it.
            budget.container(len(profiles))
            for row in profiles:
                for raw in row["roots"]:
                    budget.reserve(512 + 16 * len(raw))
            bindings = [
                row
                for row in profiles
                if any(
                    item.path == Path(raw) or Path(raw) in item.path.parents
                    for raw in row["roots"]
                )
            ]
            if not bindings:
                raise ValueError("local_snapshot_builtin_scope_unverified")
            for row in bindings:
                budget.reserve(512 + 16 * len(row["selector"]))
                binding = bootstrap._binding(
                    Path(row["selector"]),
                    profiles,
                    registry,
                    _budget=budget,
                    _custody=scope.custody,
                )
                if binding is None:
                    raise ValueError("local_snapshot_builtin_scope_unverified")
                budget.container(len(binding["namespaces"]))
                names.update(binding["namespaces"])
        names.update(
            _capture_names(
                authority,
                plan.target,
                include_absent_sqlite="settings" in plan.effective_groups,
                _read_budget=budget,
                _custody=scope.custody,
                cancel=cancel,
            )
        )
        budget.container(2 * len(names))
        return tuple(sorted(names))


def _mobile_safety_members(
    plan,
    journal,
    parent,
    prepared,
    receipt,
    session,
    archive,
    original,
    budget,
    cancel,
):
    """Reprove only unchanged installed preserve-only members at this boundary.

    The caller still performs every original SafetyCopy observation and all
    generic admission checks. This result is local evidence, never a scope.
    """
    import hashlib
    import stat

    from tldw_chatbook.Utils.platform_files import os

    from . import bootstrap
    from .admission import _price_paths
    from .archive_models import SealedArchive
    from .archive_reader import _check
    from .control_records import UNBOUND_NAMESPACE
    from .effective_roots import effective_roots
    from .journal import Journal, _evidence_digest, _matches
    from .later_rollback import _accounted_snapshot_evidence
    from .mobile_canonical_copy import _CanonicalCopy, _verify_canonical_archive
    from .mobile_runs_goal_budget import _serialization
    from .storage_admission import _CaptureFileDescriptors, _contains_owned_path

    if (
        budget is None
        or type(archive) is not SealedArchive
        or type(original) is not Journal
        or plan.local_snapshot is None
    ):
        raise ValueError("mobile_safety_source_required")
    budget = _preservation_budget(budget)
    _check(cancel)
    session._check()
    if (
        UNBOUND_NAMESPACE not in session._names
        or session._control != bootstrap.default_bootstrap_root() / "admission"
    ):
        raise ValueError("mobile_safety_source_required")
    source = plan.local_snapshot
    budget.reserve(
        4096 + path_cost(source.control_root, copies=4) + 32 * len(source.operation_id)
    )
    if (
        original.operation_id != source.operation_id
        or original.root.parent != source.control_root
        or original.root.name
        != "operation-" + hashlib.sha256(source.operation_id.encode()).hexdigest()
    ):
        raise ValueError("mobile_safety_source_journal_changed")
    budget.container(len(session._scopes) + 1)
    custody = _CaptureFileDescriptors(None)
    session._scopes.append(custody)
    try:
        # Reject actual self/physical lock identity before entering the source SH.
        # Supported source writers require the already-held unbound exclusion.
        with custody.pinned_directory(original.root) as source_parent:
            original_info = os.fstat(source_parent)
            current_info = os.fstat(parent)
            original_lock = os.stat(
                "journal.lock", dir_fd=source_parent, follow_symlinks=False
            )
            current_lock = os.stat("journal.lock", dir_fd=parent, follow_symlinks=False)
            if (
                (original_info.st_dev, original_info.st_ino) != original._identity
                or (current_info.st_dev, current_info.st_ino) != journal._identity
                or original._identity == journal._identity
                or (original_lock.st_dev, original_lock.st_ino)
                == (current_lock.st_dev, current_lock.st_ino)
                or not stat.S_ISREG(original_lock.st_mode)
                or original_lock.st_nlink != 1
                or original_lock.st_uid != os.geteuid()
                or original_lock.st_mode & 0o077
            ):
                raise ValueError("mobile_safety_source_journal_changed")
            proof, digest, terminal = _accounted_snapshot_evidence(
                original, budget, _custody=custody
            )
            _check(cancel)
            if (
                type(proof) is not _CanonicalCopy
                or terminal != "mobile_committed"
                or digest != source.rollback_digest
                or receipt.archive_digest != plan.archive_digest
                or archive.digest != receipt.archive_digest
                or receipt.manifest_digest != proof.manifest_digest
                or not _matches(
                    proof.ciphertext,
                    proof.ciphertext.path,
                    cancel=cancel,
                    _read_budget=budget,
                    _custody=custody,
                )
            ):
                raise ValueError("mobile_safety_source_changed")
            budget.reserve(512)
            if (
                hashlib.sha256(archive.manifest_bytes).hexdigest()
                != receipt.manifest_digest
            ):
                raise ValueError("mobile_safety_source_changed")
            document = _verify_canonical_archive(
                proof,
                archive,
                plan.archive_digest,
                _read_budget=budget,
                cancel=cancel,
                _custody=custody,
            )
            members, records, _profiles, registry, budget = _mobile_source_members(
                plan,
                proof,
                document,
                receipt.manifest_digest,
                session,
                _read_budget=budget,
                _custody=custody,
            )
            budget.container(4 * len(members) + len(prepared.safety_sources))
            observations = {row.logical_id: row for row in prepared.safety_sources}
            if not members:
                return members
            roots = {members[item.metadata.root_id].path for item in members.values()}
            if len(roots) != 1:
                raise ValueError("mobile_safety_source_binding_unverified")
            root = next(iter(roots))
            budget.container(len(plan.restore) + len(plan.retire))
            writes = (*plan.restore, *plan.retire)
            budget.container()
            seen_writes = set()
            for _, path in writes:
                # Price possible native Path hash/equality string materialization.
                budget.reserve(path_cost(path))
                if path in seen_writes:
                    continue
                budget.container(len(seen_writes) + 1)
                seen_writes.add(path)
                budget.reserve(512)
                _price_paths((root, path), budget)
                if bootstrap._overlap(root, path):
                    raise ValueError("mobile_safety_source_not_preserved")
            # Select current held native directory geometry, never a namespace
            # prefix or profile grant. Only fixed installed members use this path.
            budget.reserve(1024 + path_cost(root, copies=2))
            covering = None
            for name, entry in registry.items():
                _check(cancel)
                if (
                    name not in prepared.publication.namespaces
                    or name not in session._names
                    or entry not in session._root_entries
                    or entry["pending"] is not None
                    or entry["proposed"]
                ):
                    continue
                budget.reserve(512)
                native_roots = effective_roots(
                    entry["roots"],
                    (entry,),
                    _read_budget=budget,
                    _custody=custody,
                )
                for bound in native_roots:
                    if bound not in session._roots:
                        continue
                    budget.reserve(512)
                    _price_paths((bound, root), budget, copies=2)
                    # Missing roots cannot cover this installed source. Other
                    # real roots in a multi-root namespace remain candidates.
                    if not os.path.lexists(bound):
                        continue
                    resolved = bound.resolve(strict=True)
                    if (resolved, True) not in session._publication_roots:
                        continue
                    budget.container(1)
                    _price_paths((bound,), budget)
                    budget.reserve(
                        path_cost(bound)
                        + 4096
                        + 16 * (len(".bootstrap-reader") + 1) * (len(bound.parts) + 2)
                    )
                    with custody.pinned_directory(bound) as bound_fd:
                        bound_info = os.fstat(bound_fd)
                        if not stat.S_ISDIR(
                            bound_info.st_mode
                        ) or not _contains_owned_path(bound, root, _custody=custody):
                            continue
                        budget.container(3)
                        covering = bound, bound_info.st_dev, bound_info.st_ino
                    break
                if covering is not None:
                    break
            if covering is None:
                raise ValueError("mobile_safety_source_binding_unverified")
            # Group only existing member paths for this one verification call.
            # No descriptor, resolved path or source proof survives this call.
            budget.container(len(members))
            parents = {}
            for key, item in members.items():
                _check(cancel)
                budget.reserve(512 + path_cost(item.path, copies=2))
                item_parent_path = item.path.parent
                if item_parent_path not in parents:
                    budget.container()
                    parents[item_parent_path] = []
                budget.container(2)
                budget.container(len(parents[item_parent_path]) + 1)
                parents[item_parent_path].append((key, item))
            budget.reserve(path_cost(root, copies=4) + 4096)
            with custody.pinned_directory(root) as root_fd:
                root_info = os.fstat(root_fd)
                for item_parent_path, children in parents.items():
                    _check(cancel)
                    budget.container(1)
                    _price_paths((item_parent_path,), budget)
                    budget.reserve(
                        path_cost(item_parent_path)
                        + 16
                        * (len(".bootstrap-reader") + 1)
                        * (len(item_parent_path.parts) + 2)
                    )
                    budget.container(os.stat_result.n_fields)
                    budget.container(2)
                    with custody.pinned_directory(item_parent_path) as item_parent:
                        parent_info = os.fstat(item_parent)
                        for key, item in children:
                            _check(cancel)
                            # The new leaf stat, fixed comparison tuples and native
                            # expected-frame model are distinct from the parent walk.
                            budget.container(os.stat_result.n_fields)
                            budget.reserve(64 * (os.stat_result.n_fields + 12))
                            budget.container(16)
                            record = records[key]
                            if (
                                key not in plan.safety_scope
                                or (key, item.path) not in plan.preserve
                            ):
                                raise ValueError("mobile_safety_source_not_preserved")
                            info = os.stat(
                                item.path.name,
                                dir_fd=item_parent,
                                follow_symlinks=False,
                            )
                            observed = observations.get(key)
                            directory = item.status == "included_directory"
                            if (
                                observed is None
                                or observed.owner_id != _OWNER
                                or observed.source.path != str(item.path)
                                or observed.source.kind
                                != ("directory" if directory else "file")
                                or (observed.source.device, observed.source.inode)
                                != (info.st_dev, info.st_ino)
                                or (
                                    stat.S_ISDIR(info.st_mode)
                                    if directory
                                    else stat.S_ISREG(info.st_mode)
                                )
                                is False
                                or stat.S_IMODE(info.st_mode) != record.metadata.mode
                                or info.st_mtime_ns != record.metadata.mtime_ns
                            ):
                                raise ValueError("mobile_safety_source_changed")
                            if directory:
                                framing = {
                                    "safety_directory_metadata": {
                                        "device": info.st_dev,
                                        "inode": info.st_ino,
                                        "mode": stat.S_IMODE(info.st_mode),
                                        "mtime_ns": info.st_mtime_ns,
                                        "ctime_ns": info.st_ctime_ns,
                                    }
                                }
                                expected_size = 0
                            else:
                                if info.st_size != record.size:
                                    raise ValueError("mobile_safety_source_changed")
                                framing = (
                                    (
                                        "",
                                        "file",
                                        record.size,
                                        record.sha256,
                                        stat.S_IMODE(info.st_mode),
                                        info.st_mtime_ns,
                                    ),
                                )
                                expected_size = record.size
                            # Native _safety_source_matches just reobserved this exact
                            # prepared Object; bind it to the authenticated raw member.
                            _serialization(framing, budget, encoded=True)
                            if (
                                observed.source.size != expected_size
                                or observed.source.sha256 != _evidence_digest(framing)
                            ):
                                raise ValueError("mobile_safety_source_changed")
                        _check(cancel)
                        budget.container(1)
                        _price_paths((item_parent_path,), budget)
                        budget.reserve(
                            path_cost(item_parent_path)
                            + 16
                            * (len(".bootstrap-reader") + 1)
                            * (len(item_parent_path.parts) + 2)
                        )
                        budget.container(os.stat_result.n_fields)
                        budget.container(2)
                        with custody.pinned_directory(
                            item_parent_path
                        ) as current_parent:
                            parent_after = os.fstat(current_parent)
                            if (parent_info.st_dev, parent_info.st_ino) != (
                                parent_after.st_dev,
                                parent_after.st_ino,
                            ):
                                raise ValueError("mobile_safety_source_changed")
                budget.reserve(path_cost(root, copies=2) + 512)
                with custody.pinned_directory(root) as current_root:
                    current_root_info = os.fstat(current_root)
                    if (current_root_info.st_dev, current_root_info.st_ino) != (
                        root_info.st_dev,
                        root_info.st_ino,
                    ):
                        raise ValueError("mobile_safety_source_changed")
            # A session records publication path/kind, not a root inode. Reopen
            # the exact covering path to prove this call's physical continuity.
            bound, device, inode = covering
            budget.reserve(512)
            _price_paths((bound,), budget)
            budget.reserve(
                path_cost(bound)
                + 4096
                + 16 * (len(".bootstrap-reader") + 1) * (len(bound.parts) + 2)
            )
            with custody.pinned_directory(bound) as current_bound:
                info = os.fstat(current_bound)
                if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (
                    device,
                    inode,
                ):
                    raise ValueError("mobile_safety_source_binding_unverified")
            _check(cancel)
            session._check()
            return members
    finally:
        custody.retire()
        session._scopes.remove(custody)
