"""Bounded retention policy; eligibility never proves deletion authority."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta

MANAGED_BYTES = 2 * 1024 * 1024 * 1024
FREE_RESERVE_BYTES = 100 * 1024 * 1024
INACTIVE_REVISIONS = 2
REVISION_AGE = timedelta(days=30)
STAGING_AGE = timedelta(hours=24)
TERMINAL_RECEIPTS = 1000
RECEIPT_AGE = timedelta(days=30)


def retention_candidates(
    revisions: tuple[dict, ...], protected: frozenset[str], now: datetime
) -> tuple[str, ...]:
    """Return expired/excess inactive digests, excluding every protected identity.

    The caller supplies reconciled ownership and advisory creation timestamps.
    Current records protect their digest even if duplicated under another owner.
    Missing dates defer age pruning; only dated records take part in count policy.
    A candidate is not authorization to unlink its authority/package references.
    """
    if now.tzinfo is None:
        raise ValueError("retention requires timezone-aware time")
    protected = protected | frozenset(
        row["revision_digest"] for row in revisions if row.get("current")
    )
    groups = defaultdict(list)
    for row in revisions:
        created = row.get("created_at")
        if (
            row.get("current")
            or not isinstance(created, datetime)
            or created.tzinfo is None
        ):
            continue
        groups[row["installation_id"]].append(row)
    candidates = set()
    for rows in groups.values():
        rows.sort(
            key=lambda row: (row["created_at"], row["revision_digest"]), reverse=True
        )
        for index, row in enumerate(rows):
            if row["revision_digest"] not in protected and (
                index >= INACTIVE_REVISIONS or now - row["created_at"] > REVISION_AGE
            ):
                candidates.add(row["revision_digest"])
    return tuple(sorted(candidates))


def prune_transition_history(coordinator) -> None:
    """Reclaim a contiguous authenticated prefix under the existing storage owner.

    Reconciliation has already succeeded. Live aborted reviews and every current
    or unresolved target remain protected; metadata timestamps establish only age.
    """
    import time

    from . import authority_store

    coordinator._require_worker()
    authority, registry = coordinator.authority, coordinator.registry
    current = authority.load_marker()
    evidence = []
    offset = 0
    while True:
        page = authority.list_transitions(limit=50, offset=offset)
        evidence.extend(page)
        if len(page) < 50:
            break
        offset += len(page)
    by_id = {item.new.operation_id: item for item in evidence}
    lineage = []
    marker = current
    while marker.operation_id in by_id:
        item = by_id[marker.operation_id]
        if item.new != marker:
            raise ValueError("retention lineage mismatch")
        lineage.append(item)
        marker = item.old
    lineage.reverse()
    chain_ids = {item.new.operation_id for item in lineage}
    aborted = [item for item in evidence if item.new.operation_id not in chain_ids]
    live = {
        review.operation_id
        for review in coordinator._reviews.values()
        if review.expires_at > time.monotonic()
    }
    authority.retire_orphan_snapshots(
        protected_operation_ids=frozenset(
            item.new.operation_id for item in aborted if item.new.operation_id in live
        )
    )
    now = datetime.now(UTC)
    count = len(evidence)
    for item in aborted:
        if item.committed or registry.read_operation(item.new.operation_id) is not None:
            raise ValueError("unresolved retention transition")
        if item.new.operation_id not in live:
            registry.forget_operation_hints((item.new.operation_id,))
            authority.retire_transition(item)
            count -= 1
    retained_aborts = [item for item in aborted if item.new.operation_id in live]
    for item in lineage:
        if item.new == current:
            break
        if item.snapshot["operation_result"][
            "kind"
        ] == "retain" and cleanup_retained_operation(coordinator, item, remove=False):
            break
        observed = registry.operation_observed_at(item.new.operation_id)
        old_enough = observed is not None and now - observed > RECEIPT_AGE
        if count < authority_store.MAX_TRANSITIONS and not old_enough:
            break
        # Removing an endpoint needed by a still-live aborted review would turn
        # that exact prepared branch into disconnected evidence on restart.
        if any(side.old == item.old for side in retained_aborts):
            break
        registry.forget_operation_hints((item.new.operation_id,))
        authority.retire_transition(item)
        count -= 1


def managed_usage(root) -> int:
    """Count every package/cache/staging byte; unknown ownership never frees quota."""
    import os
    import stat
    from pathlib import Path

    def size(path):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise OSError("unqualified managed storage link")
        if stat.S_ISREG(info.st_mode):
            return info.st_size
        if not stat.S_ISDIR(info.st_mode):
            raise OSError("unqualified managed storage type")
        return sum(size(Path(entry.path)) for entry in os.scandir(path))

    return sum(
        size(root / name)
        for name in ("packages", "packages-revisions", "cache", "staging")
        if os.path.lexists(root / name)
    )


def require_capacity(root, additional_bytes: int) -> None:
    """Refuse quota/reserve pressure without deleting protected material."""
    import shutil

    if type(additional_bytes) is not int or additional_bytes < 0:
        raise ValueError("invalid managed allocation")
    if managed_usage(root) + additional_bytes > MANAGED_BYTES:
        raise OSError("managed plugin quota exceeded")
    if shutil.disk_usage(root).free < additional_bytes + FREE_RESERVE_BYTES:
        raise OSError("insufficient plugin storage reserve")


def eligible_revisions(
    coordinator, installation_id: str, *, snapshot=None
) -> tuple[str, ...]:
    """Qualify policy candidates against current, live and reconstruction owners."""
    import time

    coordinator._require_worker()
    snapshot = snapshot if snapshot is not None else coordinator.published_snapshot()
    # An unreconciled process can still own any revision of this installation.
    if coordinator.owner.unsettled_tokens(installation_id):
        return ()
    with coordinator.fences.live_lock:
        if any(
            record.installation_id == installation_id and not record.completed.is_set()
            for record in coordinator.fences.runs.values()
        ):
            return ()
    protected = set()
    for review in coordinator._reviews.values():
        if (
            review.installation_id == installation_id
            and review.expires_at > time.monotonic()
            and review.kind != "retain"
        ):
            receipt = coordinator.registry.read_operation(review.operation_id)
            if receipt is None or receipt["phase"] != "complete":
                protected.add(review.inspection.effective_digest)
    offset = 0
    while True:
        evidence = coordinator.authority.list_transitions(limit=50, offset=offset)
        if any(not item.committed for item in evidence):
            return ()
        if len(evidence) < 50:
            break
        offset += len(evidence)
    current = {row["revision_digest"] for row in snapshot["installations"]}
    rows = tuple(
        dict(
            row,
            current=row["revision_digest"] in current,
            created_at=coordinator.registry.revision_observed_at(
                installation_id, row["revision_digest"]
            ),
        )
        for row in snapshot["revisions"]
        if row["installation_id"] == installation_id
    )
    return retention_candidates(rows, frozenset(protected), datetime.now(UTC))


def transition_inventory(authority):
    rows = []
    while True:
        page = authority.list_transitions(limit=50, offset=len(rows))
        rows.extend(page)
        if len(page) < 50:
            return rows


def remove_revision_root(coordinator, installation_id, row, *, remove=True) -> bool:
    """Qualify original namespace/anchor/directory; absence is checked by descriptors."""
    import os
    import shutil
    from pathlib import Path

    coordinator._require_worker()
    coordinator.owner.require_owner(coordinator.owner.root)
    path = Path(row["materialized_identity"])
    digest = row["revision_digest"]
    if any(
        item["materialized_identity"] == str(path)
        for item in coordinator.published_snapshot()["revisions"]
    ):
        raise ValueError("revision remains referenced")
    if coordinator.owner.unsettled_tokens(installation_id):
        raise ValueError("revision owner unresolved")
    with coordinator.fences.live_lock:
        if any(
            item.installation_id == installation_id and not item.completed.is_set()
            for item in coordinator.fences.runs.values()
        ):
            raise ValueError("revision owner still active")
    relative = path.relative_to(coordinator.owner.root)
    if relative.parts not in {
        ("packages", installation_id),
        ("packages-revisions", installation_id, digest),
    }:
        raise ValueError("revision path is not owned")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    opened = []
    try:
        fd = os.open(coordinator.owner.root, flags)
        opened.append(fd)
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) != (row["root_device"], row["root_inode"]):
            raise ValueError("retention owner root changed")
        for component in relative.parts[:-1]:
            try:
                fd = os.open(component, flags, dir_fd=fd)
            except FileNotFoundError as error:
                # Absence is only qualified beneath the captured exact anchor.
                # A missing ancestor may have moved with the original package.
                raise ValueError("retention ancestor unavailable") from error
            opened.append(fd)
        anchor = os.fstat(fd)
        if (anchor.st_dev, anchor.st_ino) != (
            row["anchor_device"],
            row["anchor_inode"],
        ):
            raise ValueError("retention anchor changed")
        name = relative.parts[-1]
        try:
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            os.fsync(fd)
            return False
        if (info.st_dev, info.st_ino) != (row["device"], row["inode"]):
            raise ValueError("revision physical identity changed")
        if remove:
            shutil.rmtree(name, dir_fd=fd)
            os.fsync(fd)
        return not remove
    finally:
        for fd in reversed(opened):
            os.close(fd)


def cleanup_retained_operation(
    coordinator, evidence, *, remove=True
) -> tuple[str, ...]:
    """Reconcile cleanup from the original authenticated retained result."""
    if not evidence.committed:
        raise ValueError("committed retention required")
    result = evidence.snapshot["operation_result"]
    if result["kind"] != "retain":
        return ()
    if coordinator.authority.verify_transition(result["operation_id"]) != evidence:
        raise ValueError("retention evidence changed")
    errors = []
    for row in result["retired_revisions"]:
        try:
            pending = remove_revision_root(
                coordinator, result["installation_id"], row, remove=remove
            )
            if pending:
                errors.append("package_cleanup_pending")
        except (OSError, ValueError, PermissionError):
            errors.append("package_cleanup_pending")
    return tuple(errors)


from dataclasses import dataclass


@dataclass(frozen=True)
class StagingCustody:
    """Producer-supplied reconciled custody; unknown directories cannot qualify."""

    name: str
    owner_id: str
    created_at: datetime
    device: int
    inode: int
    reconciled: bool
    terminal: bool
    recovery_referenced: bool


def cleanup_staging(owner, custody: StagingCustody, now: datetime) -> bool:
    """Narrow I3 adapter; ownership/restart qualification belongs to its producer."""
    import os
    import shutil

    if (
        not custody.owner_id
        or not custody.reconciled
        or not custody.terminal
        or custody.recovery_referenced
    ):
        return False
    if (
        now.tzinfo is None
        or custody.created_at.tzinfo is None
        or now - custody.created_at < STAGING_AGE
    ):
        return False
    if (
        not custody.name
        or custody.name in {".", ".."}
        or "/" in custody.name
        or "\\" in custody.name
    ):
        raise ValueError("invalid staging identity")
    owner.require_owner(owner.root)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(owner.root, flags)
    try:
        current = owner.root.lstat()
        opened = os.fstat(root_fd)
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("staging root changed")
        staging_fd = os.open("staging", flags, dir_fd=root_fd)
        try:
            current = os.stat(custody.name, dir_fd=staging_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (custody.device, custody.inode):
                raise ValueError("staging physical identity changed")
            shutil.rmtree(custody.name, dir_fd=staging_fd)
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
    finally:
        os.close(root_fd)
    return True
