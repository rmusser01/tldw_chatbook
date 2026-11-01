"""Current native configuration proof for the ordinary committed observer only."""

import stat
from contextlib import contextmanager
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from . import bootstrap
from .admission import Admission, fcntl
from .capture_service import _selectors
from .inventory import read_profile_config
from .mobile_consolidation import (
    CanonicalConsoleStores,
    _path_witnesses,
    _selector_observation,
    _source_group_identity,
)
from .profile_paths import canonical_console_paths, legacy_mobile_dir


@contextmanager
def _claim_read_lock(lock, budget, custody):
    """Hold the existing original source flock throughout ordinary observation."""
    from .native_files import _close_owned

    budget.value(str(lock), text_copies=8)
    budget.reserve(4096)
    with custody.pinned_directory(lock.parent) as parent:
        descriptor = Admission._open(parent, lock.name, os.O_RDONLY)
        custody.fds.append(descriptor)
        try:
            initial = os.fstat(descriptor)

            def check():
                current = os.fstat(descriptor)
                named = os.stat(lock.name, dir_fd=parent, follow_symlinks=False)
                for info in (current, named):
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or stat.S_IMODE(info.st_mode) != 0o600
                        or info.st_uid != os.geteuid()
                        or info.st_nlink != 1
                        or (info.st_dev, info.st_ino)
                        != (initial.st_dev, initial.st_ino)
                    ):
                        raise ValueError("mobile_source_lock_changed")

            check()
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            check()
            yield
            check()
        finally:
            _close_owned(descriptor, custody)


def _retained_cutover_evidence(records, terminal, budget, custody):
    """Check original encrypted captures and published history without mutation."""
    from threading import Event

    from .mobile_rollback_io import observation

    budget.container(len(terminal.captures) + 2)
    cancel = Event()
    for capture in terminal.captures:
        artifact = records[capture.verified.sequence].evidence["archive"]
        budget.value(artifact["path"], text_copies=4)
        if observation(Path(artifact["path"]), budget, custody, cancel) != artifact:
            raise ValueError("mobile_capture_changed")
    if terminal.history is not None:
        artifact = records[terminal.history.sequence].evidence["artifact"]
        budget.value(artifact["path"], text_copies=4)
        if observation(Path(artifact["path"]), budget, custody, cancel) != artifact:
            raise ValueError("mobile_history_changed")


def _chain(rows):
    return tuple((row.path, row.device, row.inode, row.mode) for row in rows)


def _observe_committed_selectors(prepared, claim, terminal, budget, custody):
    """Return one call-local observation; never refresh an installed binding."""
    root = Path(prepared.bootstrap_root)
    pending, profiles, associations = bootstrap._control_records(
        root, _budget=budget, _custody=custody
    )
    if pending:
        raise bootstrap.RecoveryRequired("recovery_pending")
    registry = bootstrap._registry(root, _budget=budget, _custody=custody)
    budget.container(len(prepared.selector_review) * 8 + len(prepared.namespaces))
    selected = (Path(prepared.source_selector), Path(prepared.target_selector))
    selectors = _selectors(
        selected, include_known_profiles=True, _budget=budget, _custody=custody
    )
    if tuple(str(path) for path in selectors) != tuple(
        row.selector for row in prepared.selector_review
    ):
        raise ValueError("mobile_selector_review_changed")
    source = Path(claim.source_dir)
    budget.container(len(claim.source_witnesses) * 5)
    if _path_witnesses((source,), _budget=budget)[0][1] != _chain(
        claim.source_witnesses
    ):
        raise ValueError("mobile_source_claim_changed")
    budget.reserve(4096 + 64 * len(str(source)))
    source_group = _source_group_identity(source)
    observations = []
    admitted = set(prepared.namespaces)
    for row, selector in zip(prepared.selector_review, selectors, strict=True):
        budget.container(len(row.witnesses) * 5 + 16)
        observed = _selector_observation(selector, _budget=budget, _custody=custody)
        original = (row.fingerprint, _chain(row.witnesses))
        changed = observed != original
        if changed:
            # Native publication may replace the leaf, never an ancestor or mode.
            if (
                observed[1][:-1] != original[1][:-1]
                or observed[1][-1][0] != original[1][-1][0]
                or observed[1][-1][3] != original[1][-1][3]
                or observed[0] is None
                or row.fingerprint is None
            ):
                raise ValueError("mobile_selector_review_changed")
            with custody.pinned_directory(selector.parent) as parent:
                info = os.stat(selector.name, dir_fd=parent, follow_symlinks=False)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_mode & 0o077
                    or info.st_uid != os.geteuid()
                    or info.st_nlink != 1
                    or (str(selector), info.st_dev, info.st_ino, info.st_mode)
                    != observed[1][-1]
                ):
                    raise ValueError("mobile_selector_review_changed")
        binding = None
        if changed or selector in selected:
            binding = bootstrap._binding(
                selector, profiles, registry, _budget=budget, _custody=custody
            )
            if binding is None or binding["fingerprint"] != observed[0]:
                raise ValueError("mobile_cutover_profile_changed")
            budget.container(len(binding["namespaces"]))
            if not set(binding["namespaces"]) <= admitted:
                raise ValueError("mobile_cutover_profile_changed")
            association = next(
                (item for item in associations if item["selector"] == str(selector)),
                None,
            )
            if (association["activation"] if association else None) != binding.get(
                "activation"
            ):
                raise ValueError("mobile_cutover_profile_changed")
        if observed[0] is not None:
            config = read_profile_config(
                selector, _read_budget=budget, _custody=custody
            )
            # Price fixed resolver/path/identity representations before construction.
            budget.value(config, text_copies=16)
            budget.reserve(32768)
            current_source = legacy_mobile_dir(config)
            candidate = _source_group_identity(current_source) == source_group
            if candidate != (row.destination_identity is not None):
                raise ValueError("mobile_selector_review_changed")
            if str(selector) == prepared.source_selector and current_source != source:
                raise ValueError("mobile_source_claim_changed")
            destination = CanonicalConsoleStores(*canonical_console_paths(config))
            if (
                str(selector) == prepared.target_selector
                and destination.identity != terminal.destination_identity
            ):
                raise ValueError("mobile_cutover_target_changed")
            if candidate:
                if destination.identity != row.destination_identity:
                    raise ValueError("mobile_selector_review_changed")
                if destination.identity != terminal.destination_identity:
                    for path, target in zip(
                        destination.paths, row.targets, strict=True
                    ):
                        budget.container(
                            (len(target.witnesses) + len(target.sidecars)) * 5
                        )
                        if _path_witnesses((path,), _budget=budget)[0][1] != _chain(
                            target.witnesses
                        ):
                            raise ValueError("mobile_selector_review_changed")
                        for sidecar in target.sidecars:
                            if (
                                _path_witnesses((Path(sidecar.path),), _budget=budget)[
                                    0
                                ][1][-1]
                                != _chain((sidecar,))[0]
                            ):
                                raise ValueError("mobile_selector_review_changed")
            if (
                _selector_observation(selector, _budget=budget, _custody=custody)
                != observed
            ):
                raise ValueError("mobile_selector_review_changed")
        observations.append((str(selector), observed, binding))
    return observations, profiles, associations, registry
