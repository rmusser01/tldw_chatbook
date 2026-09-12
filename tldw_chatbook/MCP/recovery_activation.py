"""Finite current-generation MCP roots; retained policy is never live authority."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import tomllib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, _private
from tldw_chatbook.Backup_Recovery.profile_paths import lexical_path
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
from tldw_chatbook.Utils.platform_files import os

_FRESH = frozenset({"mcp.local", "mcp.permissions", "mcp.context"})
_OWNERS = _FRESH | {"mcp.targets"}
_LIMIT = 16 * 1024**2


class RootBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: int = 1
    owner: str
    source: str
    target: str
    scope: str
    root: str
    device: int
    inode: int
    workspace: str
    workspace_device: int
    workspace_inode: int
    review_digest: str


@dataclass(frozen=True)
class SourceReview:
    owner: str
    path: str
    target: str
    identity: tuple
    parent_identity: tuple
    witnesses: tuple[str, ...]
    payload: bytes | None = field(repr=False)


@dataclass(frozen=True)
class RecoveryReview:
    sources: tuple[SourceReview, ...]
    workspace: str
    workspace_identity: tuple


def _directory(path):
    from tldw_chatbook.Backup_Recovery.native_files import pinned_directory

    with pinned_directory(path) as parent:
        info = os.fstat(parent)
        if info.st_uid != os.geteuid():
            raise ValueError("mcp_recovery_root_unsafe")
        return info.st_dev, info.st_ino


def _read(path):
    from tldw_chatbook.Backup_Recovery.archive_reader import _identity, _regular

    try:
        with _regular(path) as stream:
            info = os.fstat(stream.fileno())
            before = _identity(info)
            if info.st_nlink != 1 or info.st_uid != os.geteuid():
                raise ValueError("mcp_recovery_source_unsafe")
            payload = stream.read(_LIMIT + 1)
            if len(payload) > _LIMIT or before != _identity(os.fstat(stream.fileno())):
                raise ValueError("mcp_recovery_source_changed")
        if before != _identity(os.stat(path, follow_symlinks=False)):
            raise ValueError("mcp_recovery_source_changed")
        return before, payload
    except FileNotFoundError:
        return (), None


def _workspace(payload):
    if payload is None:
        raise ValueError("mcp_recovery_config_required")
    config = tomllib.loads(payload.decode("utf-8"))
    raw = config.get("console", {}).get("workspace_root", "")
    if not isinstance(raw, str):
        raise TypeError("mcp_recovery_workspace_invalid")
    path = (Path(raw.strip()).expanduser() if raw.strip() else Path.cwd()).resolve()
    return str(path), _directory(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def observed(path, *, retained=None):
    from tldw_chatbook.Backup_Recovery.generation_witnesses import _witnesses

    path = lexical_path(path)
    with ExitStack() as stack:
        lease = (retained or {}).get(path)
        if lease is None:
            lease = stack.enter_context(acquire_storage(path))
        yield tuple(_witnesses(path, lease))


def _root(path, witnesses):
    generations = sorted({w["generation"] for w in witnesses})
    suffix = generations[0] if len(generations) == 1 else _digest(generations)
    return path.parent / ("mcp-recovery-" + suffix)


def select(store):
    """Select an absent fresh file before any reset-capable ordinary read."""
    path = lexical_path(store.path)
    store._recovery_original_path = path
    with observed(path) as witnesses:
        if witnesses:
            store.path = _root(path, witnesses) / path.name


def selected_path(canonical):
    """Resolve only this declared store's current admitted generation."""
    with observed(canonical) as witnesses:
        return _root(canonical, witnesses) / canonical.name if witnesses else canonical


def _installed_items(witness):
    """Read the actual operation's receipt-bound local destination mapping."""
    from tldw_chatbook.Backup_Recovery.archive_reader import _manifest
    from tldw_chatbook.Backup_Recovery.journal import (
        Journal,
        _CandidateReceipt,
        _evidence_digest,
        _Prepared,
    )
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.plan_records import load_plan
    from tldw_chatbook.Backup_Recovery.staging import _items

    control = Path(witness["store_root"]).parent
    operation = witness["operation_id"]
    with _private(control / ("operation-" + bootstrap._key(operation))):
        journal = Journal(control, operation)
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        if not rows or rows[-1].event not in {"committed", "rolled_back"}:
            raise ValueError("mcp_recovery_mapping_unverified")
        rollback = rows[-1].event == "rolled_back"
        event = next(
            row
            for row in reversed(rows)
            if row.event
            == ("rollback_activation_recorded" if rollback else "activation_recorded")
        )
        prepared = _Prepared.model_validate(
            next(row.evidence for row in rows if row.event == "prepared")
        )
        if (
            event.evidence["generation"] != witness["generation"]
            or event.evidence["owners"] != witness["owners"]
            or rows[-1].evidence["activation_digest"]
            != _evidence_digest(event.evidence)
            or prepared.publication.bootstrap_root
            != str(bootstrap.default_bootstrap_root())
            or not set(witness["namespaces"]) <= set(prepared.publication.namespaces)
        ):
            raise ValueError("mcp_recovery_mapping_unverified")
        plan = load_plan(journal)
        if rollback:
            items = tuple(plan.target.items) if plan.target else ()
            return items, items, tuple(event.evidence["selectors"])
        receipt = _CandidateReceipt.model_validate(rows[0].evidence)
        encoded = _read(journal.root / "verified-manifest.json")[1]
        if (
            encoded is None
            or hashlib.sha256(encoded).hexdigest() != receipt.manifest_digest
        ):
            raise ValueError("mcp_recovery_mapping_unverified")
        document = _manifest(encoded, ArchiveLimits(), encrypted=True)
        installed = tuple(_items(document, plan).values())
        original = tuple(plan.target.items) if plan.target else ()
        preserved = dict(plan.preserve)
        items = installed + tuple(
            row for row in original if preserved.get(row.logical_id) == row.path
        )
        return items, installed + original, tuple(event.evidence["selectors"])


def _imported(owner, canonical, witnesses, selector):
    """Select only explicit retained MCP payloads from the current local plan."""
    if owner not in _FRESH:
        return ()
    found = {}
    for witness in witnesses:
        items, relations, selectors = _installed_items(witness)
        if str(selector) not in selectors:
            continue
        configs = {
            row.logical_id
            for row in relations
            if row.owner == "config" and row.path == selector
        }
        originals = [
            row
            for row in items
            if row.owner == owner and row.path == canonical and row.status == "included"
        ]
        for original in originals:
            if not set(original.dependencies).intersection(configs):
                continue
            prefix = original.logical_id + ":"
            for row in items:
                if not row.logical_id.startswith(prefix):
                    continue
                suffix = row.logical_id[len(prefix) :]
                if owner == "mcp.permissions" and suffix == "bak":
                    continue
                if not (
                    suffix == "fresh"
                    or owner == "mcp.permissions"
                    and suffix == "fresh.bak"
                    or re.fullmatch(r"history\.[0-9a-f]{64}", suffix)
                ):
                    raise ValueError("mcp_recovery_mapping_unverified")
                leaves = {canonical.name} | (
                    {canonical.name + ".bak"} if owner == "mcp.permissions" else set()
                )
                if (
                    row.owner != owner
                    or row.status != "included"
                    or row.path is None
                    or row.dependencies != original.dependencies
                    or row.path.name not in leaves
                    or row.path.parent == canonical.parent
                    or suffix == "fresh"
                    and row.path.name != canonical.name
                    or suffix == "fresh.bak"
                    and row.path.name != canonical.name + ".bak"
                    or row.metadata is None
                    or row.metadata.kind != "file"
                ):
                    raise ValueError("mcp_recovery_mapping_unverified")
                previous = found.setdefault(suffix, row.path)
                if previous != row.path:
                    raise ValueError("mcp_recovery_mapping_unverified")
    return tuple(sorted(found.items()))


def _inventory_witnesses(context, canonical):
    from tldw_chatbook.Backup_Recovery.generation_witnesses import _paired_witnesses

    root = bootstrap.default_bootstrap_root()
    before = bootstrap._control_records(root)
    _, profiles, associations = before
    selector = lexical_path(context.config_path)
    binding = bootstrap._binding(selector, profiles, bootstrap._registry(root))
    if binding is None:
        if any(row["selector"] == str(selector) for row in profiles + associations):
            raise ValueError("mcp_recovery_binding_changed")
        return ()
    witnesses = _paired_witnesses(canonical, root, binding["namespaces"], selector)
    if bootstrap._control_records(root) != before:
        raise ValueError("mcp_recovery_binding_changed")
    return witnesses


def inventory_history(context, owner, canonical):
    """Retain exact former fresh files without treating them as permissions."""
    if owner not in _FRESH:
        return ()
    rows = _imported(
        owner,
        canonical,
        _inventory_witnesses(context, canonical),
        lexical_path(context.config_path),
    )
    return tuple(
        (
            suffix
            if suffix.startswith("history.")
            else "history." + _digest((owner, suffix, str(path))),
            path,
        )
        for suffix, path in rows
    )


def inventory_path(context, owner, canonical):
    """Declare one verified fresh file without borrowing execution admission."""
    from tldw_chatbook.Backup_Recovery.generation_witnesses import _paired_witnesses

    if owner not in _FRESH:
        return None
    root = bootstrap.default_bootstrap_root()
    before = bootstrap._control_records(root)
    _, profiles, associations = before
    selector = lexical_path(context.config_path)
    binding = bootstrap._binding(selector, profiles, bootstrap._registry(root))
    if binding is None:
        if any(row["selector"] == str(selector) for row in profiles + associations):
            raise ValueError("mcp_recovery_binding_changed")
        return None
    witnesses = _paired_witnesses(canonical, root, binding["namespaces"], selector)
    if not witnesses:
        return None
    target = _root(canonical, witnesses) / canonical.name
    try:
        _directory(target.parent)
    except FileNotFoundError:
        return None
    for witness in witnesses:
        record = _record(owner, canonical, witness)
        _check(
            record,
            owner,
            canonical,
            witnesses,
            witness,
            (record.workspace, (record.workspace_device, record.workspace_inode)),
        )
    if bootstrap._control_records(root) != before:
        raise ValueError("mcp_recovery_binding_changed")
    return target


def inventory_container(context, root):
    """Recognize exact current and receipt-mapped historical MCP containers."""
    from tldw_chatbook.Backup_Recovery.native_files import pinned_directory

    names = {
        "mcp.local": "local_mcp_store.json",
        "mcp.permissions": "mcp_permissions.json",
        "mcp.context": "unified_mcp_context.json",
    }
    targets = tuple(
        inventory_path(context, owner, root / name) for owner, name in names.items()
    )
    present = tuple(path for path in targets if path is not None)
    if present and (
        len(present) != len(targets) or len({path.parent for path in present}) != 1
    ):
        raise ValueError("mcp_recovery_container_changed")
    containers = {}
    for owner, name in names.items():
        for _, path in inventory_history(context, owner, root / name):
            containers.setdefault(path.parent, set()).add(path.name)
    if present:
        if present[0].parent in containers:
            raise ValueError("mcp_recovery_container_changed")
        containers[present[0].parent] = set(names.values())
    for parent, required in containers.items():
        optional = (
            {"mcp_permissions.json.bak"}
            if present and parent == present[0].parent
            else set()
        )
        with pinned_directory(parent) as fd:
            before = os.fstat(fd)
            children = set(os.listdir(fd))
            if not required <= children <= required | optional:
                raise ValueError("mcp_recovery_container_changed")
            for name in children:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != os.geteuid()
                ):
                    raise ValueError("mcp_recovery_container_changed")
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_mtime_ns,
            ):
                raise ValueError("mcp_recovery_container_changed")
        if _directory(parent) != (before.st_dev, before.st_ino):
            raise ValueError("mcp_recovery_container_changed")
    return tuple(containers)


def _record_name(owner, path):
    return "mcp-root-" + _digest((owner, str(path))) + ".json"


def _record(owner, path, witness):
    store = ActivationStore(Path(witness["store_root"]))
    with _private(store._generation(witness["generation"])) as parent:
        return RootBinding.model_validate(
            bootstrap._read(parent, _record_name(owner, path))
        )


def _check(record, owner, path, witnesses, witness, workspace):
    target = _root(path, witnesses) / path.name if owner in _FRESH else path
    root = target.parent
    if (
        record.version != 1
        or record.owner != owner
        or record.source != str(path)
        or record.target != str(target)
        or record.scope != _digest(witness)
        or record.root != str(root)
        or _directory(root) != (record.device, record.inode)
        or workspace
        != (record.workspace, (record.workspace_device, record.workspace_inode))
    ):
        raise ValueError("mcp_recovery_binding_changed")


def allowed(sources, *, retained):
    """Owner flags alone cannot authorize a restored MCP source."""
    workspace = None
    for owner, path in sources:
        if owner not in _OWNERS:
            continue
        with observed(path, retained=retained) as witnesses:
            if witnesses and workspace is None:
                workspace = _workspace(_read(bootstrap.effective_config_path())[1])
            for witness in witnesses:
                record = _record(owner, path, witness)
                _check(record, owner, path, witnesses, witness, workspace)
    return True


def require_store_write(store, owner):
    from .activation import MCPActivationRequired

    try:
        _require_store_write(store, owner)
    except (OSError, ValueError, TypeError):
        raise MCPActivationRequired() from None


def readable(store, owner):
    """Inactive stores expose defaults without reading unreviewed policy."""
    from .activation import MCPActivationRequired

    try:
        require_store_write(store, owner)
    except MCPActivationRequired:
        return False
    return True


def _require_store_write(store, owner):
    path = getattr(store, "_recovery_original_path", lexical_path(store.path))
    with observed(path) as witnesses:
        if not witnesses:
            return
        expected = _root(path, witnesses) / path.name if owner in _FRESH else path
        if lexical_path(store.path) != expected:
            raise ValueError("mcp_recovery_binding_changed")
        workspace = _workspace(_read(bootstrap.effective_config_path())[1])
        for witness in witnesses:
            _check(
                _record(owner, path, witness),
                owner,
                path,
                witnesses,
                witness,
                workspace,
            )
            activation = ActivationStore(Path(witness["store_root"]))
            if any(
                not activation.allowed(witness["generation"], name)
                for name in _OWNERS.intersection(witness["owners"])
            ):
                raise ValueError("mcp_recovery_review_required")


@contextmanager
def _review_scope(service):
    from .activation import _sources

    sources = _sources(service)
    with ExitStack() as stack:
        leases = {
            path: stack.enter_context(acquire_storage(path))
            for path in dict.fromkeys(path for _, path in sources)
        }
        definitions = set()
        for owner, path in sources:
            if owner != "mcp.local":
                continue
            with observed(path, retained=leases) as witnesses:
                definitions.update(
                    value
                    for suffix, value in _imported(
                        owner,
                        path,
                        witnesses,
                        lexical_path(bootstrap.effective_config_path()),
                    )
                    if suffix == "fresh"
                )
        if len(definitions) > 1:
            raise ValueError("mcp_recovery_mapping_unverified")
        sources = tuple(sources) + tuple(("definition", path) for path in definitions)
        for path in definitions:
            if path not in leases:
                leases[path] = stack.enter_context(acquire_storage(path))
        yield sources, leases


def _review(sources, leases):
    rows = []
    for owner, path in sources:
        with observed(path, retained=leases) as witnesses:
            if owner != "config" and not witnesses:
                raise ValueError("mcp_recovery_source_unbound")
            target = (
                _root(path, witnesses) / path.name
                if owner in _FRESH and witnesses
                else path
            )
            identity, payload = _read(path)
            rows.append(
                SourceReview(
                    owner,
                    str(path),
                    str(target),
                    identity,
                    _directory(path.parent),
                    tuple(json.dumps(w, sort_keys=True) for w in witnesses),
                    payload,
                )
            )
            if owner == "mcp.permissions":
                backup = path.with_name(path.name + ".bak")
                identity, payload = _read(backup)
                rows.append(
                    SourceReview(
                        "history",
                        str(backup),
                        str(backup),
                        identity,
                        _directory(path.parent),
                        (),
                        payload,
                    )
                )
    if not any(row.witnesses for row in rows):
        raise ValueError("mcp_recovery_not_required")
    workspace, identity = _workspace(
        next(row.payload for row in rows if row.owner == "config")
    )
    return RecoveryReview(tuple(rows), workspace, identity)


def capture(service):
    with _review_scope(service) as (sources, leases):
        return _review(sources, leases)


def _fresh_payloads(review):
    from .local_store import LocalMCPStoreState
    from .permission_store import _fresh_payload
    from .unified_control_models import UnifiedMCPContext

    original = next(
        (row.payload for row in review.sources if row.owner == "definition"),
        next(row.payload for row in review.sources if row.owner == "mcp.local"),
    )
    payload = json.loads(original) if original is not None else {}
    if not isinstance(payload, dict):
        raise TypeError("mcp_recovery_definitions_invalid")
    profiles = LocalMCPStoreState.from_dict(payload).profiles
    return {
        "mcp.local": LocalMCPStoreState(profiles=profiles).to_dict(),
        "mcp.permissions": _fresh_payload(),
        "mcp.context": UnifiedMCPContext().to_dict(),
    }


def approve(service, review):
    from tldw_chatbook.Backup_Recovery.activation import _flush_existing, _write
    from tldw_chatbook.Backup_Recovery.native_files import (
        _flush_private_tree,
        create_private_directory,
        create_private_file,
    )

    if type(review) is not RecoveryReview:
        raise ValueError("mcp_recovery_review_changed")
    with _review_scope(service) as (sources, leases):
        if _review(sources, leases) != review:
            raise ValueError("mcp_recovery_review_changed")
        payloads = _fresh_payloads(review)
        digest = _digest(
            [
                (
                    r.owner,
                    r.path,
                    r.target,
                    r.identity,
                    r.parent_identity,
                    r.witnesses,
                    hashlib.sha256(r.payload).hexdigest()
                    if r.payload is not None
                    else None,
                )
                for r in review.sources
            ]
        )
        records = []
        for row in review.sources:
            if row.owner not in _OWNERS:
                continue
            witnesses = tuple(json.loads(value) for value in row.witnesses)
            for witness in witnesses:
                if row.owner not in witness["owners"]:
                    raise ValueError("mcp_recovery_owner_not_required")
                try:
                    record = _record(row.owner, Path(row.path), witness)
                except FileNotFoundError:
                    record = None
                if record is not None:
                    _check(
                        record,
                        row.owner,
                        Path(row.path),
                        witnesses,
                        witness,
                        (review.workspace, review.workspace_identity),
                    )
                    if record.review_digest != digest:
                        raise ValueError("mcp_recovery_review_changed")
                records.append((row, witness, record))
        owned = {record.root for _, _, record in records if record is not None}
        for root in dict.fromkeys(
            Path(row.target).parent for row in review.sources if row.owner in _FRESH
        ):
            with acquire_storage(root):
                with _private(root.parent):
                    pass
                if str(root) not in owned:
                    expected = {
                        Path(row.target).name: json.dumps(
                            payloads[row.owner], sort_keys=True
                        ).encode()
                        for row in review.sources
                        if row.owner in _FRESH and Path(row.target).parent == root
                    }
                    try:
                        create_private_directory(root)
                    except FileExistsError:
                        pass
                    with _private(root) as parent:
                        identity = _directory(root)
                        present = set(os.listdir(parent))
                        if not present <= expected.keys():
                            raise ValueError("mcp_recovery_fresh_policy_changed")
                        for name in present:
                            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                            if (
                                info.st_mode & 0o077
                                or _read(root / name)[1] != expected[name]
                            ):
                                raise ValueError("mcp_recovery_fresh_policy_changed")
                    for name, encoded in expected.items():
                        if name not in present:
                            with (
                                create_private_file(root / name) as fd,
                                os.fdopen(fd, "wb", closefd=False) as output,
                            ):
                                output.write(encoded)
                    with _private(root) as parent:
                        if (
                            _directory(root) != identity
                            or set(os.listdir(parent)) != expected.keys()
                        ):
                            raise ValueError("mcp_recovery_fresh_policy_changed")
                        if any(
                            _read(root / name)[1] != encoded
                            for name, encoded in expected.items()
                        ):
                            raise ValueError("mcp_recovery_fresh_policy_changed")
                with _private(root) as parent:
                    _flush_private_tree(parent, os.fstat(parent).st_dev)
        if _review(sources, leases) != review:
            raise ValueError("mcp_recovery_review_changed")
        completed = all(
            record is not None
            and ActivationStore(Path(witness["store_root"])).allowed(
                witness["generation"], row.owner
            )
            for row, witness, record in records
        )
        if not completed:
            for row in review.sources:
                if (
                    row.owner in _FRESH
                    and json.loads(_read(Path(row.target))[1]) != payloads[row.owner]
                ):
                    raise ValueError("mcp_recovery_fresh_policy_changed")
        for row, witness, record in records:
            root = Path(row.target).parent
            device, inode = _directory(root)
            bound = RootBinding(
                owner=row.owner,
                source=row.path,
                target=row.target,
                scope=_digest(witness),
                root=str(root),
                device=device,
                inode=inode,
                workspace=review.workspace,
                workspace_device=review.workspace_identity[0],
                workspace_inode=review.workspace_identity[1],
                review_digest=digest,
            )
            activation = ActivationStore(Path(witness["store_root"]))
            with _private(activation._generation(witness["generation"])) as parent:
                if record is None:
                    _write(parent, _record_name(row.owner, row.path), bound)
                else:
                    _flush_existing(parent, _record_name(row.owner, row.path), bound)
        for row, witness, _ in records:
            ActivationStore(Path(witness["store_root"])).approve(
                witness["generation"], row.owner
            )
        service.context = service.context_store.load()
        service.clear_session_approvals()
