"""Explicit manual recovery of opaque bytes; never profile or schema restoration."""

import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from uuid import uuid4

from . import archive_reader as reader
from .archive_models import DependencyGroup, Directory, Owner, Payload, SealedArchive
from .limits import ArchiveLimits
from .native_files import (
    create_private_directory,
    create_private_file,
    pinned_directory,
    publish_new,
)
from .qualification import qualified_for
from .space import require_capacity


@dataclass(frozen=True)
class InertExtractionPlan:
    archive_digest: str
    manifest_digest: str
    group_ids: tuple[str, ...]
    destination: Path
    parent_identity: tuple[int, int]
    protected_roots: tuple[Path, ...]
    limits: ArchiveLimits
    files: tuple[Payload, ...]
    directories: tuple[Directory, ...]
    groups: tuple[DependencyGroup, ...]
    owners: tuple[Owner, ...]
    unselected_dependencies: tuple[tuple[str, str], ...]
    payload_bytes: int


@dataclass(frozen=True)
class InertExtractionResult:
    destination: Path
    archive_digest: str
    group_ids: tuple[str, ...]
    report_path: Path
    state: str = "inert_extracted"


def _path(path):
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute_output_required")
    return path


def _overlap(left, right):
    return left == right or left in right.parents or right in left.parents


def _destination(archive, destination, protected_roots):
    from .bootstrap import default_bootstrap_root
    from .service_storage import default_control_root

    _path(destination)
    protected = (
        archive.path.parent,
        default_bootstrap_root(),
        default_control_root().parent,
        *protected_roots,
    )
    if any(_overlap(destination, _path(root)) for root in protected):
        raise ValueError("output_overlap")
    with pinned_directory(destination.parent) as parent:
        try:
            os.stat(destination.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("destination_exists")
        info = os.fstat(parent)
        identity = info.st_dev, info.st_ino
    for operation in ("publish_new", "publish_directory"):
        allowed, reason = qualified_for(operation, destination.parent)
        if not allowed:
            raise ValueError(reason)
    return identity


def _output(payload):
    return "payload/" + hashlib.sha256(payload.logical_id.encode()).hexdigest() + ".bin"


def _report(plan):
    return {
        "version": 1,
        "state": "inert_extracted",
        "archive_digest": plan.archive_digest,
        "manifest_digest": plan.manifest_digest,
        "groups": [
            {"group_id": group.group_id, "complete": group.complete}
            for group in plan.groups
        ],
        "files": [
            {
                "logical_id": row.logical_id,
                "root_id": row.root_id,
                "archive_relative_path": row.relative_path,
                "owner_id": row.owner_id,
                "output": _output(row),
                "size": row.size,
                "sha256": row.sha256,
            }
            for row in plan.files
        ],
        "directories": [
            {
                "logical_id": row.logical_id,
                "root_id": row.root_id,
                "archive_relative_path": row.relative_path,
            }
            for row in plan.directories
        ],
        "declared_owners": [row.model_dump(mode="json") for row in plan.owners],
        "unselected_dependencies": plan.unselected_dependencies,
        "validation": "bytes_verified_only",
        "metadata": "private_modes_only",
    }


def _encoded(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode()


def preview_inert_extraction(
    archive: SealedArchive,
    *,
    group_ids: tuple[str, ...],
    destination: Path,
    limits: ArchiveLimits,
    cancel: Event,
    protected_roots: tuple[Path, ...] = (),
) -> InertExtractionPlan:
    """Review explicit available groups and one absent manual output directory."""
    if (
        type(group_ids) is not tuple
        or not group_ids
        or any(type(key) is not str for key in group_ids)
        or len(set(group_ids)) != len(group_ids)
    ):
        raise ValueError("extraction_groups_required")
    if type(protected_roots) is not tuple or type(limits) is not ArchiveLimits:
        raise ValueError("extraction_options_invalid")
    for root in protected_roots:
        _path(root)
    doc = reader.verify_sealed(archive, cancel)
    with reader._regular(archive.path) as stream:
        size = os.fstat(stream.fileno()).st_size
    input_size = (
        archive.encrypted_source.identity[2] if archive.encrypted_source else size
    )
    if input_size > limits.input_bytes:
        raise ValueError("input_limit")
    if size > limits.decrypted_bytes:
        raise ValueError("decrypted_limit")
    checked = reader._inspect(
        archive.path,
        limits,
        archive.encrypted_source is not None,
        cancel,
        archive.digest,
    )
    if checked != archive.manifest_bytes:
        raise ValueError("sealed_changed")
    reader.verify_sealed(archive, cancel)
    by_group = {row.group_id: row for row in doc.dependency_groups}
    if not set(group_ids) <= by_group.keys():
        raise ValueError("extraction_group_unknown")
    groups = tuple(by_group[key] for key in sorted(group_ids))
    selected = {key for group in groups for key in group.members}
    files = tuple(row for row in doc.files if row.logical_id in selected)
    directories = {row.logical_id: row for row in doc.directories}
    ancestors = selected & directories.keys()
    for row in files:
        parent = row.parent_id
        while parent is not None:
            ancestors.add(parent)
            parent = directories[parent].parent_id
    plan = InertExtractionPlan(
        archive.digest,
        hashlib.sha256(checked).hexdigest(),
        tuple(sorted(group_ids)),
        destination,
        _destination(archive, destination, protected_roots),
        tuple(sorted(set(protected_roots))),
        limits,
        files,
        tuple(directories[key] for key in sorted(ancestors)),
        groups,
        tuple(
            row
            for row in doc.owners
            if row.owner_id in {file.owner_id for file in files}
        ),
        tuple(
            sorted(
                (row.logical_id, dep)
                for row in doc.producer_inventory
                if row.logical_id in selected
                for dep in row.dependencies
                if dep not in selected
            )
        ),
        sum(row.size for row in files),
    )
    require_capacity(
        {destination: plan.payload_bytes + 2 * len(_encoded(_report(plan))) + 4096}
    )
    return plan


def _write(path, data):
    with create_private_file(path) as fd, os.fdopen(os.dup(fd), "wb") as output:
        output.write(data)


def extract_inert(
    archive: SealedArchive, plan: InertExtractionPlan, *, cancel: Event
) -> InertExtractionResult:
    """Copy unchanged selected payloads, then publish only a complete opaque folder."""
    if type(plan) is not InertExtractionPlan:
        raise ValueError("extraction_preview_required")
    if (
        preview_inert_extraction(
            archive,
            group_ids=plan.group_ids,
            destination=plan.destination,
            limits=plan.limits,
            cancel=cancel,
            protected_roots=plan.protected_roots,
        )
        != plan
    ):
        raise ValueError("extraction_preview_changed")
    report = _encoded(_report(plan))
    work = plan.destination.parent / (".inert-extraction-" + uuid4().hex)
    create_private_directory(work)
    info = work.stat(follow_symlinks=False)
    identity = info.st_dev, info.st_ino
    publishing = False
    try:
        candidate = work / "candidate"
        create_private_directory(candidate)
        create_private_directory(candidate / "payload")
        copied = 0
        with reader._regular(archive.path) as stream:
            central = reader._central_preflight(stream, plan.limits)
            with zipfile.ZipFile(stream) as container:
                offsets = reader._local_ranges(stream, container.infolist(), central)
                for row in plan.files:
                    count = 0
                    digest = hashlib.sha256()
                    with (
                        create_private_file(candidate / _output(row)) as fd,
                        os.fdopen(os.dup(fd), "wb") as output,
                    ):
                        info = container.getinfo(row.payload)
                        for chunk in reader._member_chunks(
                            stream, info, offsets[row.payload], cancel
                        ):
                            count += len(chunk)
                            copied += len(chunk)
                            if count > min(
                                row.size, plan.limits.member_bytes
                            ) or copied > min(
                                plan.payload_bytes, plan.limits.expanded_bytes
                            ):
                                raise ValueError("expanded_limit")
                            require_capacity(
                                {
                                    work: plan.payload_bytes
                                    - copied
                                    + 2 * len(report)
                                    + 4096
                                }
                            )
                            output.write(chunk)
                            digest.update(chunk)
                    if count != row.size or digest.hexdigest() != row.sha256:
                        raise ValueError("payload_digest_mismatch")
        _write(candidate / "inert-mapping.json", report)
        reader.verify_sealed(archive, cancel)
        for row in plan.files:
            if reader._hash(candidate / _output(row), cancel) != row.sha256:
                raise ValueError("extraction_bytes_changed")
        if (
            _destination(archive, plan.destination, plan.protected_roots)
            != plan.parent_identity
        ):
            raise ValueError("publication_parent_changed")
        _write(
            work / "publication-intent.json",
            _encoded(
                {
                    "version": 1,
                    "destination": str(plan.destination),
                    "parent_identity": plan.parent_identity,
                    "archive_digest": plan.archive_digest,
                    "report_digest": hashlib.sha256(report).hexdigest(),
                    "group_ids": plan.group_ids,
                }
            ),
        )
        reader._check(cancel)
        publishing = True
        publish_new(
            candidate,
            plan.destination,
            parent_identities=(identity, plan.parent_identity),
        )
        return InertExtractionResult(
            plan.destination,
            archive.digest,
            plan.group_ids,
            plan.destination / "inert-mapping.json",
        )
    finally:
        # Once native publication starts, preserve intent even on an ambiguous barrier.
        if not publishing:
            with pinned_directory(work) as parent:
                info = os.fstat(parent)
                if (info.st_dev, info.st_ino) != identity:
                    raise ValueError("extraction_workspace_changed")
            shutil.rmtree(work)
