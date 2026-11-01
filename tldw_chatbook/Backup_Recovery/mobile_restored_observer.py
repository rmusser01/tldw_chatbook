"""Ordinary observation of one direct native restore of a mobile canonical copy."""

import hashlib
import stat
from contextlib import contextmanager
from pathlib import Path
from threading import Event

from tldw_chatbook.Utils.platform_files import os

from . import bootstrap
from .archive_reader import _manifest, _regular
from .journal import Journal, _evidence_digest, _mobile_ref, _Prepared
from .limits import ArchiveLimits
from .mobile_cutover_observer import _observe_committed_selectors
from .mobile_preview_budget import _completion_observation, path_cost
from .mobile_rollback_io import observation
from .mobile_runs_goal_budget import _serialization
from .plan_records import _load_plan
from .storage_admission import _mobile_check_witnesses


def _digest(value, budget):
    _serialization(value, budget, encoded=True)
    return _evidence_digest(value)


def _reference(record, budget):
    budget.container(6)
    budget.value(record.operation_id, text_copies=2)
    budget.value(record.previous, text_copies=2)
    budget.value(record.event, text_copies=2)
    budget.value(record.evidence, text_copies=2)
    data = record.model_dump()
    _serialization(data, budget, encoded=True)
    return _mobile_ref(record)


def _manifest_bytes(journal, expected, budget, custody):
    budget.reserve(4096 + path_cost(journal.root, copies=4))
    path = journal.root / "verified-manifest.json"
    with _regular(path, _custody=custody) as stream:
        before = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
        ):
            raise ValueError("verified_manifest_not_private")
        if before.st_size > ArchiveLimits().manifest_bytes:
            raise ValueError("manifest_limit")
        budget.reserve(256 + before.st_size + 1)
        raw = stream.read(before.st_size + 1)
        if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("verified_manifest_changed")
        after = os.fstat(stream.fileno())
        named = path.lstat()
        facts = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_uid,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if facts(before) != facts(after) or facts(before) != facts(named):
            raise ValueError("verified_manifest_changed")
    budget.json(raw)
    budget.reserve(16384 + 32 * len(raw))
    return _manifest(raw, ArchiveLimits(), True)


def _physical(targets, installed, artifacts, budget, custody):
    """Current DB bytes may change; the published file and its ancestors may not."""
    budget.container(len(targets) * 20)
    states = []
    for target in targets:
        original = target.witnesses[-1]
        path = Path(original.path)
        budget.reserve(4096 + path_cost(path, copies=8))
        saved = [row for row in installed["artifacts"] if row["path"] == str(path)]
        declared = [row for row in artifacts if row.target == str(path)]
        if (
            len(saved) != 1
            or len(declared) != 1
            or declared[0].action != "publish"
            or declared[0].previous is None
            or declared[0].candidate is None
            or (declared[0].previous.device, declared[0].previous.inode)
            != (original.device, original.inode)
            or (declared[0].candidate.device, declared[0].candidate.inode)
            != (saved[0]["device"], saved[0]["inode"])
            or saved[0]["kind"] != "file"
        ):
            raise ValueError("mobile_restored_target_changed")
        # Direct replacement of a DB leaf preserves the original authenticated
        # full ancestor chain. No arbitrary new path is admitted by equality.
        for witness in target.witnesses[:-1]:
            budget.reserve(512 + path_cost(Path(witness.path), copies=2))
            with custody.pinned_directory(Path(witness.path)) as parent:
                info = os.fstat(parent)
                if (info.st_dev, info.st_ino, info.st_mode) != (
                    witness.device,
                    witness.inode,
                    witness.mode,
                ):
                    raise ValueError("mobile_restored_ancestor_changed")
        with custody.pinned_directory(path.parent) as parent:
            current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (
                not stat.S_ISREG(current.st_mode)
                or stat.S_IMODE(current.st_mode) != 0o600
                or current.st_uid != os.geteuid()
                or current.st_nlink != 1
                or (current.st_dev, current.st_ino)
                != (saved[0]["device"], saved[0]["inode"])
            ):
                raise ValueError("mobile_restored_target_changed")
            states.append((str(path), current.st_dev, current.st_ino, current.st_mode))
            # Sidecars are ephemeral. Check each observation's native safety,
            # without requiring their lifetime or content to remain constant.
            budget.container(3)
            for suffix in ("-wal", "-shm", "-journal"):
                budget.reserve(128 + 8 * (len(path.name) + len(suffix)))
                name = path.name + suffix
                try:
                    info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077
                    or info.st_nlink != 1
                ):
                    raise ValueError("mobile_restored_target_changed")
    return states


@contextmanager
def _direct_restoration(
    mobile, rows, prepared, witnesses, claim, terminal, configuration, budget, custody
):
    """Read one real native chain under the caller's source and ordinary leases."""
    try:
        _mobile_check_witnesses(rows, witnesses)
    except bootstrap.RecoveryRequired as error:
        if str(error) != "mobile_object_changed":
            raise
    else:
        yield False
        return
    cancel = Event()
    selected = next(
        (row[2] for row in configuration[0] if row[0] == prepared.target_selector), None
    )
    activation = selected.get("activation") if selected else None
    if activation is None or activation["store_root"] != str(
        mobile.root.parent / "activation"
    ):
        raise ValueError("mobile_restored_generation_required")
    budget.value(activation, text_copies=4)
    with _completion_observation(budget, custody):
        journal = Journal._open_existing(
            mobile.root.parent,
            activation["operation_id"],
            _budget=budget,
            _custody=custody,
        )
        if (
            journal.operation_id == mobile.operation_id
            or journal._identity == mobile._identity
        ):
            raise ValueError("mobile_restored_generation_required")
        with journal._locked(exclusive=False, _custody=custody) as parent:
            records = journal._records(parent, _budget=budget, _custody=custody)
            if not records or records[-1].event != "committed":
                raise ValueError("mobile_restored_generation_required")
            terminal_ref = _reference(records[-1], budget)
            plan = _load_plan(journal, records)
            budget.container(len(records) * 4)
            preparation = next(row for row in records if row.event == "prepared")
            budget.value(preparation.evidence, text_copies=4)
            replacement = _Prepared.model_validate(preparation.evidence)
            published = next(
                row for row in records if row.event == "installed_validated"
            )
            activated = next(
                row for row in records if row.event == "activation_recorded"
            )
            installed = published.evidence
            canonical = next(
                row
                for row in rows
                if row.event == "mobile_capture_verified"
                and row.evidence["purpose"] == "canonical"
            )
            source = plan.local_snapshot
            if (
                plan.mode != "replace"
                or source is None
                or source.control_root != mobile.root.parent
                or source.operation_id != mobile.operation_id
                or source.rollback_digest != _digest(canonical.evidence, budget)
                or replacement.mode != "replace"
                or replacement.isolated_profiles
                or replacement.publication is None
                or replacement.publication.bootstrap_root != prepared.bootstrap_root
                or replacement.generation != activation["generation"]
                or records[-1].evidence["generation"] != activation["generation"]
                or prepared.target_selector not in activated.evidence["selectors"]
                or activated.evidence["owners"] != activation["owners"]
                or not set(activation["namespaces"])
                <= set(replacement.publication.namespaces)
                or records[-1].evidence["activation_digest"]
                != _digest(activated.evidence, budget)
                or activated.evidence["installed_digest"] != _digest(installed, budget)
                or activated.evidence["plan_digest"] != installed["plan_digest"]
                or installed["plan_digest"] != replacement.publication.plan_digest
                or plan.archive_digest != replacement.publication.archive_digest
                or installed["manifest_digest"] != canonical.evidence["manifest_digest"]
            ):
                raise ValueError("mobile_restored_provenance_changed")
            document = _manifest_bytes(
                journal, installed["manifest_digest"], budget, custody
            )
            budget.container(
                len(document.files) + len(plan.restore) + len(terminal.targets) * 4
            )
            destinations = dict(plan.restore)
            for target in terminal.targets:
                matches = [
                    row
                    for row in document.files
                    if row.owner_id == target.owner_id
                    and destinations.get(row.logical_id)
                    == Path(target.witnesses[-1].path)
                ]
                if len(matches) != 1:
                    raise ValueError("mobile_restored_owner_coverage_changed")
            from .activation import ActivationStore

            store = ActivationStore(Path(activation["store_root"]))
            required_path = store._generation(replacement.generation) / "required.json"
            budget.reserve(4096 + path_cost(required_path, copies=8))
            required = [
                row
                for row in activated.evidence["records"]
                if row["path"] == str(required_path)
            ]
            if (
                len(required) != 1
                or observation(required_path, budget, custody, cancel) != required[0]
            ):
                raise ValueError("mobile_restored_activation_changed")
            with custody.pinned_directory(required_path.parent) as required_parent:
                if (
                    store._required(required_parent, replacement.generation).owners
                    != activation["owners"]
                ):
                    raise ValueError("mobile_restored_activation_changed")
            # Current paired config records were validated by the native selector
            # observer. Their original leaf hashes are not permanent invariants.
            before = _physical(
                terminal.targets, installed, replacement.artifacts, budget, custody
            )
            from .mobile_consolidation_recovery import _open_completion_read
            from .owner_registry import install_adapters
            from .sqlite_validation import _check, _restrict_connection

            owners = install_adapters()
            budget.container(len(owners) + 3)
            for target in terminal.targets:
                owner = next(row for row in owners if row.owner_id == target.owner_id)
                budget.container(12)
                with _open_completion_read(
                    Path(target.witnesses[-1].path)
                ) as connection:
                    connection.execute("BEGIN")
                    restrictions = _restrict_connection(connection, cancel)
                    issues, version = _check(
                        connection, owner, owner.schema_policy(), restrictions
                    )
                    if (
                        issues
                        or version != target.schema_version
                        or restrictions.expired()
                    ):
                        raise ValueError("mobile_restored_schema_changed")
            if (
                _physical(
                    terminal.targets, installed, replacement.artifacts, budget, custody
                )
                != before
            ):
                raise ValueError("mobile_restored_target_changed")
            yield True
            current = journal._records(parent, _budget=budget, _custody=custody)
            if _reference(current[-1], budget) != terminal_ref:
                raise ValueError("mobile_restored_generation_changed")
            # Receipt-bound plan and required/manifest bytes are immutable; current
            # data is checked by generation/physical/schema, not old payload hash.
            from .journal import _CandidateReceipt, _matches

            budget.value(current[0].evidence, text_copies=2)
            receipt = _CandidateReceipt.model_validate(current[0].evidence)
            if not _matches(
                receipt.local_plan,
                str(journal.root / "restore-plan.json"),
                cancel=cancel,
                _read_budget=budget,
                _custody=custody,
            ):
                raise ValueError("mobile_restored_provenance_changed")
            if observation(required_path, budget, custody, cancel) != required[0]:
                raise ValueError("mobile_restored_activation_changed")
            _manifest_bytes(journal, installed["manifest_digest"], budget, custody)
            if (
                _observe_committed_selectors(prepared, claim, terminal, budget, custody)
                != configuration
            ):
                raise ValueError("mobile_cutover_changed")
            if (
                _physical(
                    terminal.targets, installed, replacement.artifacts, budget, custody
                )
                != before
            ):
                raise ValueError("mobile_restored_target_changed")
