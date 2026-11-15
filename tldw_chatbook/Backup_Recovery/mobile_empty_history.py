"""R176: one native empty default history container, preserved without writes."""

from contextlib import contextmanager

from .mobile_preview_budget import path_cost

_OWNER = "agents.history"


@contextmanager
def _history_custody(budget, supplied=None):
    """Use original scope custody or the native bounded standalone observer."""
    from .mobile_preview_budget import current
    from .storage_admission import _CaptureFileDescriptors

    scope = current()
    if scope is not None and scope.read_budget is not budget:
        raise ValueError("mobile_preview_budget_changed")
    custody = (
        supplied
        if supplied is not None
        else scope.custody
        if scope is not None
        else None
    )
    owned = custody is None
    if owned:
        budget.reserve(512)
        custody = _CaptureFileDescriptors(None)
    if type(custody) is not _CaptureFileDescriptors or custody.close_failed:
        raise ValueError("mobile_canonical_history_custody_required")
    if custody.scope is not None:
        custody.scope.check()
        if custody not in custody.scope.resources:
            raise ValueError("mobile_canonical_history_custody_required")
    try:
        yield custody
    finally:
        if owned:
            custody.retire()


def _native_default_history_root(
    config, proof, budget, *, _custody=None, _config_bytes=None
):
    """Read this authenticated ordinary profile's native default selection."""
    from .inventory import _parse_profile_config, read_profile_config
    from .journal import _MobilePrepared
    from .mobile_canonical_copy import _CanonicalCopy
    from .mobile_consolidation import _selector_observation

    if (
        type(proof) is not _CanonicalCopy
        or type(proof.prepared) is not _MobilePrepared
        or config.owner != "config"
        or config.status != "included"
        or config.path is None
    ):
        raise ValueError("mobile_canonical_history_binding_changed")
    budget.reserve(4096 + path_cost(config.path, copies=16 + len(config.path.parts)))
    budget.container(len(proof.prepared.selector_review))
    selected = str(config.path)
    if selected not in (proof.prepared.source_selector, proof.prepared.target_selector):
        raise ValueError("mobile_canonical_history_binding_changed")
    rows = [row for row in proof.prepared.selector_review if row.selector == selected]
    if len(rows) != 1 or rows[0].fingerprint is None:
        raise ValueError("mobile_canonical_history_binding_changed")
    row = rows[0]
    if _config_bytes is not None:
        import hashlib

        from .bootstrap import MAX_RECORD

        # Only semantic bytes from the installed owner's private copy arrive here.
        # Physical publication proof stays at that owner, never in this mapper.
        if type(_config_bytes) is not bytes or len(_config_bytes) > MAX_RECORD:
            raise ValueError("mobile_canonical_config_data_changed")
        budget.reserve(512)
        if hashlib.sha256(_config_bytes).hexdigest() != row.fingerprint:
            raise ValueError("mobile_canonical_config_data_changed")
        return _default_history_root(
            _parse_profile_config(_config_bytes, _read_budget=budget), budget
        )
    budget.container(len(row.witnesses))
    expected = (
        row.fingerprint,
        tuple((w.path, w.device, w.inode, w.mode) for w in row.witnesses),
    )
    with _history_custody(budget, _custody) as custody:
        if (
            _selector_observation(config.path, _budget=budget, _custody=custody)
            != expected
        ):
            raise ValueError("mobile_canonical_selector_review_required")
        data = read_profile_config(config.path, _read_budget=budget, _custody=custody)
        if (
            _selector_observation(config.path, _budget=budget, _custody=custody)
            != expected
        ):
            raise ValueError("mobile_canonical_selector_review_required")
    return _default_history_root(data, budget)


def _default_history_root(data, budget):
    """Pure native selection from this profile's already owned configuration."""
    # Ordinary _RunLogs.discover passes stores=None, regardless of DB overrides.
    # Charge selection/string/path work even outside an ambient accounting scope.
    from tldw_chatbook.Utils.platform_files import os

    from .profile_paths import lexical_path, setting, user_data_dir

    budget.value(data, text_copies=16)
    budget.value(os.environ.get("HOME"), text_copies=16)
    budget.reserve(8192)
    root = user_data_dir(data) / "tool_sandbox"
    configured = setting(data, "tools", "file_sandbox_root")
    selected_root = lexical_path(configured) if configured else root
    return root if selected_root == root else None


def _empty_history_members(
    document, target, budget, proof, *, _custody=None, _config_data=None
):
    """Match only native default-container topology; this grants no read scope."""
    from .archive_models import ArchiveManifest
    from .models import DiscoveryContext, Inventory
    from .recovery_files import _tree_member_id

    if (
        type(document) is not ArchiveManifest
        or type(target) is not Inventory
        or not target.complete
    ):
        raise ValueError("mobile_canonical_current_target_required")
    budget.container(
        5
        * (
            len(document.producer_inventory)
            + len(document.directories)
            + len(document.files)
            + len(target.items)
        )
    )
    producers = {row.logical_id: row for row in document.producer_inventory}
    records = {row.logical_id: row for row in (*document.directories, *document.files)}
    current = {item.logical_id: item for item in target.items}
    if (
        len(producers) != len(document.producer_inventory)
        or len(records) != len(document.directories) + len(document.files)
        or len(current) != len(target.items)
    ):
        raise ValueError("mobile_canonical_history_members_changed")
    used_configs = None
    if _config_data is not None:
        if type(_config_data) is not tuple:
            raise ValueError("mobile_canonical_config_data_changed")
        budget.container(len(_config_data))
        budget.container(len(_config_data))
        used_configs, seen_configs = set(), set()
        for pair in _config_data:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not bytes
                or pair[0] in seen_configs
            ):
                raise ValueError("mobile_canonical_config_data_changed")
            seen_configs.add(pair[0])
    matched = {}
    for record in document.directories:
        producer = producers.get(record.logical_id)
        if (
            producer is None
            or producer.owner_id != _OWNER
            or record.parent_id is not None
            or record.synthetic
        ):
            continue
        # Nonempty history remains on ordinary deferred-owner restore/refusal.
        if any(
            row.root_id == record.logical_id and row.logical_id != record.logical_id
            for row in records.values()
        ):
            continue
        key = record.logical_id
        budget.value(key, text_copies=4)
        parts = key.split(":")
        item = current.get(key)
        if (
            len(parts) != 4
            or parts[0] != "profile"
            or parts[1] not in document.profile_ids
            or parts[2] != _OWNER
            or item is None
            or item.owner != _OWNER
            or item.path is None
        ):
            raise ValueError("mobile_canonical_history_members_changed")
        config_key = f"profile:{parts[1]}:config"
        runs_key = f"profile:{parts[1]}:db.agent_runs"
        config, runs = current.get(config_key), current.get(runs_key)
        if (
            config is None
            or config.owner != "config"
            or config.status != "included"
            or config.path is None
            or runs is None
            or runs.owner != "db.agent_runs"
            or runs.status != "included"
            or runs.path is None
            or config_key not in runs.dependencies
        ):
            raise ValueError("mobile_canonical_history_binding_changed")
        budget.reserve(
            2048 + path_cost(runs.path, copies=5) + path_cost(config.path, copies=2)
        )
        config_bytes = None
        if _config_data is not None:
            for key_config, raw_config in _config_data:
                if key_config == config_key:
                    config_bytes = raw_config
                    used_configs.add(config_key)
                    break
        if config_bytes is None:
            root = _native_default_history_root(
                config, proof, budget, _custody=_custody
            )
        else:
            root = _native_default_history_root(
                config, proof, budget, _custody=_custody, _config_bytes=config_bytes
            )
        if root is None or item.path != root:
            continue
        expected = _tree_member_id(
            DiscoveryContext(config.path, parts[1]), _OWNER, root, root
        )
        meta = item.metadata
        if (
            key != expected
            or record.root_id != key
            or record.relative_path != ""
            or producer.status != "included_directory"
            or item.status != producer.status
            or producer.dependencies != (config_key,)
            or item.dependencies != producer.dependencies
            or item.shared_group != producer.shared_group
            or item.shared_group is not None
            or meta is None
            or meta.version != 1
            or meta.policy != "private"
            or (
                meta.root_id,
                meta.parent_id,
                meta.relative_path,
                meta.kind,
                meta.mode,
                meta.mtime_ns,
            )
            != (
                key,
                None,
                "",
                "directory",
                record.metadata.mode,
                record.metadata.mtime_ns,
            )
        ):
            raise ValueError("mobile_canonical_history_members_changed")
        if any(
            other.logical_id != key
            and other.status in {"included", "included_directory"}
            and (
                other.path is not None
                and (other.path == root or root in other.path.parents)
                or other.metadata is not None
                and other.metadata.root_id == key
            )
            for other in target.items
        ):
            raise ValueError("mobile_canonical_history_not_empty")
        matched[key] = item
    if _config_data is not None and len(used_configs) != len(_config_data):
        raise ValueError("mobile_canonical_config_data_changed")
    return matched


def _empty_history_observation(item, record, budget, custody, cancel):
    """Observe zero entries and exact native directory/ancestor continuity."""
    import stat

    from tldw_chatbook.Utils.platform_files import os

    from .archive_reader import _check
    from .bootstrap import _bounded_names
    from .native_files import pinned_directory
    from .storage_admission import _CaptureFileDescriptors

    if (
        type(custody) is not _CaptureFileDescriptors
        or custody.close_failed
        or item.owner != _OWNER
        or item.status != "included_directory"
        or item.path is None
        or item.metadata is None
    ):
        raise ValueError("mobile_canonical_history_custody_required")
    _check(cancel)
    budget.reserve(8192 + path_cost(item.path, copies=16 + len(item.path.parts)))
    ancestors = []

    def observe_ancestor(fd):
        budget.reserve(256)
        info = os.fstat(fd)
        ancestors.append((info.st_dev, info.st_ino))

    with pinned_directory(
        item.path, _custody=custody, _visit=observe_ancestor
    ) as parent:
        before = os.fstat(parent)
        if (
            not stat.S_ISDIR(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o077
            or stat.S_IMODE(before.st_mode) != record.metadata.mode
            or before.st_mtime_ns != record.metadata.mtime_ns
        ):
            raise ValueError("mobile_canonical_history_metadata_changed")
        _bounded_names(parent, 0, budget, custody)
        after = os.fstat(parent)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("mobile_canonical_history_changed")
        previous_ancestors = ancestors
        ancestors = []
        with pinned_directory(
            item.path, _custody=custody, _visit=observe_ancestor
        ) as named:
            _bounded_names(named, 0, budget, custody)
            info = os.fstat(named)
            if ancestors != previous_ancestors or identity != (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_uid,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ):
                raise ValueError("mobile_canonical_history_changed")
    _check(cancel)
    return identity


def _verify_empty_history_members(document, members, budget, cancel, *, custody=None):
    """Repeat physical empty proof within the caller's existing native custody."""
    budget.container(len(members))
    history = {key: item for key, item in members.items() if item.owner == _OWNER}
    if not history:
        return
    from threading import Event

    cancel = Event() if cancel is None else cancel
    budget.container(len(history) + len(document.directories))
    records = {row.logical_id: row for row in document.directories}
    with _history_custody(budget, custody) as observed_custody:
        for key, item in history.items():
            _empty_history_observation(
                item, records[key], budget, observed_custody, cancel
            )


def _publication_source_context(
    journal, plan, root, namespaces, archive, budget, original, session, cancel
):
    """Validate the fixed private native tuple before any new journal acquisition."""
    from hashlib import sha256
    from threading import Event

    from tldw_chatbook.Backup_Recovery.models import _TraceReadBudget

    from . import bootstrap
    from .archive_models import SealedArchive
    from .archive_reader import _check
    from .control_records import UNBOUND_NAMESPACE
    from .journal import Journal
    from .mobile_builtin_preservation import _preservation_budget
    from .restore_plan import LocalSnapshotSource, RestorePlan
    from .storage_admission import MaintenanceSession

    if all(value is None for value in (archive, budget, original, session, cancel)):
        return False
    if (
        type(archive) is not SealedArchive
        or type(budget) is not _TraceReadBudget
        or type(original) is not Journal
        or type(session) is not MaintenanceSession
        or type(cancel) is not Event
        or type(plan) is not RestorePlan
        or type(plan.local_snapshot) is not LocalSnapshotSource
        or plan.mode != "replace"
        or type(journal) is not Journal
    ):
        raise ValueError("mobile_safety_source_required")
    _preservation_budget(budget)
    _check(cancel)
    session._check()
    budget.reserve(4096 + path_cost(root, copies=4))
    budget.container(len(namespaces))
    if (
        root != bootstrap.default_bootstrap_root()
        or session._control != root / "admission"
        or UNBOUND_NAMESPACE not in session._names
        or any(name not in session._names for name in namespaces)
        or archive.digest != plan.archive_digest
    ):
        raise ValueError("mobile_safety_source_required")
    source = plan.local_snapshot
    budget.reserve(
        path_cost(source.control_root, copies=4) + 32 * len(source.operation_id)
    )
    if (
        original is journal
        or original._identity == journal._identity
        or original.operation_id != source.operation_id
        or original.root.parent != source.control_root
        or original.root.name
        != "operation-" + sha256(source.operation_id.encode()).hexdigest()
    ):
        raise ValueError("mobile_safety_source_journal_changed")
    return True


def _publication_history_observation(item, plan, context, session, budget, cancel):
    """Observe one held native selected directory provisionally, without payload IO.

    This is not source authentication. The caller must run complete native mobile
    safety authentication against the proposed Prepared record before appending it.
    """
    import stat
    from pathlib import Path

    from tldw_chatbook.Utils.platform_files import os

    from . import bootstrap
    from .archive_reader import _check
    from .effective_roots import effective_roots
    from .publication import _observe_safety_directory
    from .storage_admission import _CaptureFileDescriptors, _contains_owned_path

    _check(cancel)
    session._check()
    budget.reserve(512)
    budget.container(len(session._scopes) + 1)
    custody = _CaptureFileDescriptors(None)
    session._scopes.append(custody)
    try:
        budget.reserve(16384 + path_cost(item.path, copies=24 + len(item.path.parts)))
        if (
            item.owner != _OWNER
            or item.status != "included_directory"
            or item.metadata is None
            or item.metadata.root_id != item.logical_id
            or item.metadata.relative_path != ""
            or item.metadata.parent_id is not None
            or (item.logical_id, item.path) not in plan.preserve
            or item.logical_id not in plan.safety_scope
        ):
            raise ValueError("mobile_safety_source_binding_unverified")
        budget.value(item.logical_id, text_copies=4)
        budget.container(len(plan.target.items))
        config_key = ":".join(item.logical_id.split(":")[:2]) + ":config"
        configs = [
            row
            for row in plan.target.items
            if row.logical_id == config_key and row.owner == "config"
        ]
        if len(configs) != 1 or str(configs[0].path) not in context.selectors:
            raise ValueError("mobile_safety_source_binding_unverified")
        root = Path(context.bootstrap_root)
        _, profiles = bootstrap._records(root, _budget=budget, _custody=custody)
        registry = bootstrap._registry(root, _budget=budget, _custody=custody)
        binding = bootstrap._binding(
            configs[0].path, profiles, registry, _budget=budget, _custody=custody
        )
        if binding is None or any(
            name not in session._names for name in binding["namespaces"]
        ):
            raise ValueError("mobile_safety_source_binding_unverified")
        covering = None
        for name in binding["namespaces"]:
            _check(cancel)
            entry = registry[name]
            if (
                name not in context.namespaces
                or entry not in session._root_entries
                or entry["pending"] is not None
                or entry["proposed"]
            ):
                continue
            budget.reserve(512)
            roots = effective_roots(
                entry["roots"], (entry,), _read_budget=budget, _custody=custody
            )
            for bound in roots:
                if bound not in session._roots:
                    continue
                budget.reserve(8192 + path_cost(bound, copies=16 + len(bound.parts)))
                if (
                    not os.path.lexists(bound)
                    or (bound.resolve(strict=True), True)
                    not in session._publication_roots
                ):
                    continue
                with custody.pinned_directory(bound) as parent:
                    before = os.fstat(parent)
                    if not stat.S_ISDIR(before.st_mode) or not _contains_owned_path(
                        bound, item.path, _custody=custody
                    ):
                        continue
                    session._check()
                    # This primitive cannot open/read a substituted file body.
                    observation = _observe_safety_directory(item.path, _custody=custody)
                    with custody.pinned_directory(item.path) as selected:
                        info = os.fstat(selected)
                        if (
                            info.st_mode & 0o077
                            or stat.S_IMODE(info.st_mode) != item.metadata.mode
                            or info.st_mtime_ns != item.metadata.mtime_ns
                            or (info.st_dev, info.st_ino)
                            != (observation["device"], observation["inode"])
                        ):
                            raise ValueError("mobile_safety_source_changed")
                    with custody.pinned_directory(bound) as current:
                        after = os.fstat(current)
                        if (before.st_dev, before.st_ino) != (
                            after.st_dev,
                            after.st_ino,
                        ):
                            raise ValueError("mobile_safety_source_binding_unverified")
                    covering = observation
                break
            if covering is not None:
                break
        if covering is None:
            raise ValueError("mobile_safety_source_binding_unverified")
        _check(cancel)
        session._check()
        return covering
    finally:
        custody.retire()
        session._scopes.remove(custody)
