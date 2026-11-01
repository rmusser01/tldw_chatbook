"""Fixed prospective history projections and actual publication observations."""

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tldw_chatbook.Utils.platform_files import os

from . import archive_reader, native_files
from .bootstrap import MAX_RECORD
from .inventory import _parse_profile_config
from .mobile_history_files import _root_component
from .profile_paths import user_data_dir
from .storage_admission import _CaptureFileDescriptors, _contains_capture_path


def _target_history_location(captured, budget):
    """Read the publisher's existing fixed target selector with retained custody."""
    captured.check()
    selector = captured.preview.target_selector
    provenance = next(
        row for row in captured.preview.selector_provenance if row.selector == selector
    )
    custody = _CaptureFileDescriptors(None)
    captured.session._scopes.append(custody)
    try:
        with native_files.pinned_directory(selector.parent, _custody=custody) as parent:
            fd = os.open(
                selector.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent,
            )
            custody.fds.append(fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= MAX_RECORD:
                raise ValueError("mobile_history_selector_changed")
            budget.reserve(128 + info.st_size + 1)
            raw = os.read(fd, info.st_size + 1)
            if (
                len(raw) != info.st_size
                or archive_reader._identity(info)
                != archive_reader._identity(os.fstat(fd))
                or archive_reader._identity(info)
                != archive_reader._identity(
                    os.stat(selector.name, dir_fd=parent, follow_symlinks=False)
                )
                or hashlib.sha256(raw).hexdigest() != provenance.fingerprint
            ):
                raise ValueError("mobile_history_selector_changed")
    finally:
        custody.retire()
    config = _parse_profile_config(raw, _read_budget=budget)
    data_root = user_data_dir(config)
    location = (
        data_root
        / "agent_history"
        / "imports"
        / captured.preview.destination.identity
        / captured.preview.source_fingerprint
    )
    if not _contains_capture_path(captured.session._roots, location):
        raise ValueError("mobile_history_outside_scope")
    captured.check()
    return data_root, location


def _history_shapes(plan, location, budget):
    """Qualify complete future receipt shapes; no prospective inode is authority."""
    from tldw_chatbook.Agents.run_log_paging import _SEGMENT

    budget.container(len(plan.history_members) * 12 + len(plan.history) * 2)
    proofs = {proof.logical_id: proof for proof in plan.history}
    members = {member.logical_id: member for member in plan.history_members}
    if len(members) != len(plan.history_members):
        raise ValueError("mobile_history_duplicate_member")
    if {
        member.logical_id
        for member in plan.history_members
        if member.sha256 is not None
    } != set(proofs):
        raise ValueError("mobile_history_incomplete_native_qualification")
    roots, children, expected = {}, [], {}
    for member in plan.history_members:
        if member.root_id not in members:
            raise ValueError("mobile_history_root_missing")
        root = members[member.root_id]
        if (
            root.relative_path != ""
            or root.sha256 is not None
            or root.location != _root_component(root.logical_id)
        ):
            raise ValueError("mobile_history_root_changed")
        if member.location != root.location + (
            "/" + member.relative_path if member.relative_path else ""
        ):
            raise ValueError("mobile_history_location_changed")
        if member.relative_path:
            parent = str(PurePosixPath(member.relative_path).parent)
            parent = "" if parent == "." else parent
            if not any(
                candidate.root_id == member.root_id
                and candidate.relative_path == parent
                and candidate.sha256 is None
                for candidate in plan.history_members
            ):
                raise ValueError("mobile_history_parent_missing")
        budget.container(len(plan.history_members) * 3)
        rows, total = [], 0
        descendants = [
            candidate
            for candidate in plan.history_members
            if candidate.root_id == member.root_id
            and (
                not member.relative_path
                or candidate.relative_path == member.relative_path
                or candidate.relative_path.startswith(member.relative_path + "/")
            )
        ]

        # Native observe_artifact uses sorted depth-first traversal, not lexical
        # sorting of whole paths (a sibling may sort before a child's slash).
        def visit(current):
            nonlocal total
            suffix = (
                ""
                if current is member  # noqa: B023 -- Synchronous walk finishes within this iteration.
                else "/"
                + current.relative_path[
                    len(member.relative_path)  # noqa: B023
                    + (
                        1 if member.relative_path else 0  # noqa: B023
                    ) :
                ]
            )
            budget.value(suffix)
            if current.sha256 is None:
                rows.append((suffix, "directory"))  # noqa: B023 -- Synchronous walk finishes before this loop advances.
                direct = [
                    candidate
                    for candidate in descendants  # noqa: B023 -- Synchronous walk finishes within this iteration.
                    if candidate is not current
                    and str(PurePosixPath(candidate.relative_path).parent)
                    == (current.relative_path or ".")
                ]
                for candidate in sorted(
                    direct, key=lambda value: PurePosixPath(value.relative_path).name
                ):
                    visit(candidate)
            else:
                rows.append((suffix, "file", current.size, current.sha256))  # noqa: B023 -- Synchronous walk finishes before this loop advances.
                total += current.size

        visit(member)
        budget.value(rows, text_copies=8)
        digest = hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode()
        ).hexdigest()
        path = str(location / member.location)
        budget.value(path, text_copies=3)
        shape = {
            "path": path,
            "device": 2**64 - 1,
            "inode": 2**64 - 1,
            "kind": "directory" if member.sha256 is None else "file",
            "size": total,
            "sha256": digest,
        }
        expected[member.logical_id] = shape
        proof = proofs.get(member.logical_id)
        budget.container(24)
        data = {
            "logical_id": member.logical_id,
            "root_id": member.root_id,
            "relative_path": member.relative_path,
            "location": member.location,
            "size": member.size,
            "sha256": member.sha256,
            "owner": list(proof.owner_association)
            if proof is not None and proof.owner_association is not None
            else None,
            "framed_runs": [
                list(pair) for pair in proof.associations if pair[0] in proof.run_ids
            ]
            if proof is not None
            and _SEGMENT.fullmatch(PurePosixPath(member.relative_path).name)
            else [],
            "historical_run_ids": list(proof.historical_run_ids)
            if proof is not None
            else [],
            "segment_ids": list(proof.segment_ids) if proof is not None else [],
            "artifact": shape,
        }
        children.append({"version": 1, "kind": "runlog", "data": data})
        if not member.relative_path:
            roots[member.root_id] = shape
            children.append(
                {
                    "version": 1,
                    "kind": "runlog_root",
                    "data": {
                        "root_id": member.root_id,
                        "location": member.location,
                        "artifact": shape,
                    },
                }
            )
    return children, expected


def _observe_history_children(scope, children, expected, budget):
    """Replace prospective numbers exclusively with actual native observations."""
    scope.check()
    budget.container(len(scope.members) * 3)
    observed = {}
    for member in scope.members:
        # All subtree observation representation was reserved before publication.
        actual = scope._observe(scope.location / member.location)
        shape = expected[member.logical_id]
        if any(actual[key] != shape[key] for key in ("path", "kind", "size", "sha256")):
            raise ValueError("mobile_history_published_member_changed")
        observed[member.logical_id] = actual
    result = []
    budget.container(len(children) * 3)
    for child in children:
        if child["kind"] not in {"runlog", "runlog_root"}:
            raise ValueError("mobile_history_prospective_kind_changed")
        data = dict(child["data"])
        identity = data["logical_id"] if child["kind"] == "runlog" else data["root_id"]
        data["artifact"] = observed[identity]
        result.append({"version": 1, "kind": child["kind"], "data": data})
    records = scope._records()
    evidence = next(
        row.evidence
        for row in reversed(records)
        if row.event == "mobile_history_verified"
    )
    result.append(
        {
            "version": 1,
            "kind": "history_publication",
            "data": {
                "destination_identity": scope.captured.preview.destination.identity,
                "source_fingerprint": scope.captured.preview.source_fingerprint,
                "location": str(scope.location),
                "artifact": evidence["artifact"],
            },
        }
    )
    scope.check()
    return result


def _original_history_inventory(
    session, partition, current, selections, approved_scope
):
    """Invert only this pending operation's witnessed first-root creation.

    This internal call consumes installed current discovery and both original
    authenticated manifests in the actual live reader. It grants no new source.
    File-byte qualification remains the later physically separate AFiles phase.
    """
    from dataclasses import replace

    from . import control_records
    from .inventory import _selection_scope_digest, classify_entries
    from .journal import _evidence_digest, _mobile_state
    from .mobile_captured_sqlite import _CapturedSQLite
    from .mobile_history_files import _HistoryMember
    from .storage_admission import _local, _mobile_admission, _mobile_witness

    reads = getattr(_local, "mobile_sqlite_scope", None)
    if type(reads) is not _CapturedSQLite or reads.session is not session:
        raise ValueError("scope_changed")
    reads.check()
    partition.check()
    budget = reads.read_budget
    admission = _mobile_admission(session)
    if (
        budget is None
        or admission[2] is not budget
        or partition.claim is not reads.captured.claim
        or partition.journal is not reads.journal
        or partition.scope_digest != approved_scope
        or not current.complete
        or current.issues
        or reads._canonical_archive is None
        or reads._source_archive is None
    ):
        raise ValueError("scope_changed")
    originals = (reads._source_archive, reads._canonical_archive)
    if not any(row[0] is partition for row in originals):
        raise ValueError("scope_changed")
    for original_partition, document, sealed in originals:
        original_partition.check()
        source = (
            reads.captured.source
            if original_partition.purpose == "legacy"
            else reads.captured.canonical
        )
        if (
            source is None
            or sealed.manifest_bytes != source.manifest_bytes
            or sealed.encrypted_source is None
            or sealed.encrypted_source.digest != source.digest
        ):
            raise ValueError("mobile_sqlite_capture_changed")
    with reads.journal._locked(exclusive=False) as fd:
        records = reads.journal._records(fd)
    prepared, _witnesses = _mobile_state(records)
    control_records._mobile_pending(session, reads.journal, prepared)
    budget.container(len(records) * 2 + len(current.items) * 8)
    intents = [
        row.evidence for row in records if row.event == "mobile_history_intended"
    ]
    if len(intents) != 1:
        raise ValueError("scope_changed")
    intent = intents[0]
    budget.value(intent, text_copies=4)
    data_root, location = _target_history_location(reads.captured, budget)
    publication = data_root / "agent_history"
    temporary = data_root / (
        ".mobile-history-"
        + hashlib.sha256(reads.journal.operation_id.encode()).hexdigest()
    )
    if (
        intent["publication"] != str(publication)
        or intent["temporary"] != str(temporary)
        or intent["location"] != str(location)
        or intent["parent"]["path"] != str(data_root)
        or intent["source_fingerprint"] != prepared.source_fingerprint
        or intent["source_fingerprint"] != reads.captured.preview.source_fingerprint
        or intent["destination_identity"] != reads.captured.preview.destination.identity
        or intent["manifest_digest"]
        != hashlib.sha256(reads._source_archive[2].manifest_bytes).hexdigest()
    ):
        raise ValueError("scope_changed")
    roots = [
        item
        for item in current.items
        if item.owner == "agents.history" and item.path == publication
    ]
    if len(roots) != 1:
        raise ValueError("scope_changed")
    root = roots[0]
    metadata = root.metadata
    if (
        root.status != "included_directory"
        or root.shared_group is not None
        or root.deletion_validated
        or metadata is None
        or (
            metadata.version,
            metadata.root_id,
            metadata.relative_path,
            metadata.parent_id,
            metadata.kind,
            metadata.policy,
        )
        != (1, root.logical_id, "", None, "directory", "private")
    ):
        raise ValueError("scope_changed")
    canonical = reads._canonical_archive[1]
    budget.container(
        len(canonical.producer_inventory)
        + len(canonical.files)
        + len(canonical.directories)
    )
    before = [
        item
        for item in canonical.producer_inventory
        if item.logical_id == root.logical_id
    ]
    if (
        len(before) != 1
        or before[0].owner_id != root.owner
        or before[0].status != "unused"
        or before[0].shared_group is not None
        or before[0].dependencies != root.dependencies
        or any(
            item.logical_id == root.logical_id
            for item in (*canonical.files, *canonical.directories)
        )
    ):
        raise ValueError("scope_changed")

    # Reproduce the fixed native member grammar from authenticated original
    # records plus the still-current installed declarations, never a saved plan.
    source = reads._source_archive[1]
    budget.container(
        len(source.producer_inventory) * 2 + len(source.files) + len(source.directories)
    )
    producers = {
        item.logical_id
        for item in source.producer_inventory
        if item.owner_id == "agents.history"
        and item.status in {"included", "included_directory"}
    }
    declarations = {item.logical_id: item for item in current.items}
    members = []
    for item in (*source.directories, *source.files):
        if item.logical_id not in producers:
            continue
        declared = declarations.get(item.logical_id)
        original_root = declarations.get(item.root_id)
        if (
            declared is None
            or original_root is None
            or original_root.path is None
            or declared.owner != "agents.history"
            or declared.metadata is None
            or declared.metadata.root_id != item.root_id
            or declared.metadata.relative_path != item.relative_path
            or declared.path != original_root.path / item.relative_path
            or original_root.path == publication
        ):
            raise ValueError("scope_changed")
        budget.container(9)
        relative = (Path(_root_component(item.root_id)) / item.relative_path).as_posix()
        budget.value((str(original_root.path), relative), text_copies=3)
        members.append(
            _HistoryMember(
                item.logical_id,
                item.root_id,
                str(original_root.path),
                item.relative_path,
                relative,
                getattr(item, "size", 0),
                getattr(item, "sha256", None),
            )
        )
    if producers != {member.logical_id for member in members}:
        raise ValueError("scope_changed")
    budget.container(len(members) * 10 + 1)
    member_evidence = {
        "members": [
            member.__dict__ for member in sorted(members, key=lambda row: row.location)
        ]
    }
    budget.value(member_evidence, text_copies=4)
    if _evidence_digest(member_evidence) != intent["members_digest"]:
        raise ValueError("scope_changed")
    digest = _evidence_digest(intent)
    phases = [
        row
        for row in records
        if row.event
        in {
            "mobile_history_staged",
            "mobile_history_observed",
            "mobile_history_verified",
        }
    ]
    if not phases or phases[0].event != "mobile_history_staged":
        raise ValueError("scope_changed")
    artifact = phases[0].evidence["artifact"]
    budget.value(artifact, text_copies=3)
    if artifact["path"] != str(temporary) or artifact["kind"] != "directory":
        raise ValueError("scope_changed")
    for phase in phases:
        if phase.evidence["intent_digest"] != digest or phase.evidence["artifact"] != {
            **artifact,
            "path": str(
                temporary if phase.event == "mobile_history_staged" else publication
            ),
        }:
            raise ValueError("scope_changed")
    custody = _CaptureFileDescriptors(reads)
    try:
        with native_files.pinned_directory(data_root, _custody=custody) as parent:
            if _mobile_witness(data_root) != intent["parent"]:
                raise ValueError("mobile_history_parent_changed")
            held_parent = os.fstat(parent)
            if (held_parent.st_dev, held_parent.st_ino) != (
                intent["parent"]["device"],
                intent["parent"]["inode"],
            ):
                raise ValueError("mobile_history_parent_changed")
            # lstat semantics: a broken link is an existing temporary name.
            try:
                os.stat(temporary.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError("mobile_history_custody_ambiguous")
            fd = os.open(
                publication.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent,
            )
            custody.fds.append(fd)
            held = os.fstat(fd)
            named = os.stat(publication.name, dir_fd=parent, follow_symlinks=False)
            if (
                not stat.S_ISDIR(named.st_mode)
                or not stat.S_ISDIR(held.st_mode)
                or (held.st_dev, held.st_ino) != (artifact["device"], artifact["inode"])
                or (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino)
            ):
                raise ValueError("mobile_history_object_changed")
    finally:
        custody.retire()
    budget.container(len(current.items) * 16)
    for item in current.items:
        budget.value(item, text_copies=4)
        if item.path is not None:
            budget.value(str(item.path), text_copies=8)
    restored = []
    for item in current.items:
        if item is root:
            restored.append(replace(root, status="unused", metadata=None))
        elif item.metadata is not None and item.metadata.root_id == root.logical_id:
            if (
                item.owner != root.owner
                or item.path is None
                or not item.path.is_relative_to(publication)
                or item.path == publication
                or item.shared_group is not None
            ):
                raise ValueError("scope_changed")
        else:
            restored.append(item)
    # Native classification supplies the original None resolved slot for unused.
    restored = classify_entries(tuple(restored))
    result = replace(
        restored,
        scope_digest=_selection_scope_digest(restored.scope_digest, selections),
    )
    if not result.complete or result.scope_digest != approved_scope:
        raise ValueError("scope_changed")
    reads.check()
    return result


@dataclass(frozen=True)
class _HistoryObservationBounds:
    """Fixed live-operation numeric schedule; never observed content authority."""

    publication: Path
    temporary: Path
    whole: object
    members: tuple


def _history_observation_bounds(
    captured, plan, data_root, location, publication, budget
):
    from .journal import _ArtifactObservationLimits, _qualify_artifact_enumeration

    temporary = publication.parent / (
        ".mobile-history-"
        + hashlib.sha256(captured.journal.operation_id.encode()).hexdigest()
    )
    budget.value(str(data_root), text_copies=4)
    budget.reserve(4096 + 128 * len(str(data_root)))
    custody = _CaptureFileDescriptors(None)
    captured.session._scopes.append(custody)
    try:
        with native_files.pinned_directory(data_root, _custody=custody) as fd:
            component_limit = _qualify_artifact_enumeration(fd, custody)
    finally:
        custody.retire()
    # Admit each path representation before constructing the fixed target census.
    budget.reserve(1024 + 32 * (len(str(location)) + len(str(publication))))
    prefix = str(location.relative_to(publication))
    prefix = "" if prefix == "." else prefix
    budget.container(
        len(plan.history_members) * 8 + len(PurePosixPath(prefix).parts) * 8 + 16
    )
    for member in plan.history_members:
        budget.reserve(
            256 + 16 * (len(prefix) + len(member.location) + len(str(location)) + 4)
        )
        budget.container((len(PurePosixPath(member.location).parts) + 1) * 4)
    whole_rows = {"": 0}
    parts = PurePosixPath(prefix).parts
    for index in range(1, len(parts) + 1):
        whole_rows["/" + "/".join(parts[:index])] = 0
    for member in plan.history_members:
        relative = "/" + (prefix + "/" if prefix else "") + member.location
        if relative in whole_rows:
            raise ValueError("mobile_history_duplicate_observation_path")
        whole_rows[relative] = member.size
    for relative in whole_rows:
        for component in PurePosixPath(relative).parts:
            if component == "/":
                continue
            budget.reserve(64 + 4 * len(component))
            if len(os.fsencode(component)) > component_limit:
                raise ValueError("artifact_bounded_enumeration_unavailable")

    def limits(rows):
        budget.container(16)
        return _ArtifactObservationLimits(
            entry_count=len(rows),
            total_file_bytes=sum(rows.values()),
            total_relative_characters=sum(len(path) for path in rows),
            maximum_relative_length=max(map(len, rows), default=0),
            maximum_depth=max((path.count("/") for path in rows), default=0),
            maximum_component_length=component_limit,
        )

    def price(bounds, absolute_length):
        return (
            131072
            + 2048
            + 12 * bounds.maximum_component_length
            + 4096 * bounds.entry_count
            + 128 * bounds.total_relative_characters
            + 64 * bounds.entry_count * (bounds.maximum_depth + 1)
            + 128 * absolute_length
            + 4096
        )

    whole = limits(whole_rows)
    absolute_width = max(len(str(publication)), len(str(temporary)))
    budget.reserve(3 * price(whole, absolute_width) + 4 * (128 * absolute_width + 4096))
    member_bounds = []
    for member in plan.history_members:
        budget.container(len(plan.history_members) * 4 + 4)
        rows = {}
        for candidate in plan.history_members:
            if candidate.root_id != member.root_id:
                continue
            if candidate.location == member.location:
                relative = ""
            elif candidate.location.startswith(member.location + "/"):
                budget.reserve(64 + 4 * len(candidate.location))
                relative = candidate.location[len(member.location) :]
            else:
                continue
            rows[relative] = candidate.size
        bounds = limits(rows)
        target = location / member.location
        budget.reserve(price(bounds, len(str(target))))
        member_bounds.append((target, bounds))
    budget.container(len(member_bounds) * 2 + 12)
    return _HistoryObservationBounds(
        publication, temporary, whole, tuple(member_bounds)
    )
