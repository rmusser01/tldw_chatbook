"""Pure source-owned RunLog qualification within the captured SQLite lifetime."""

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from tldw_chatbook.Agents.run_log_format import decode_record_header
from tldw_chatbook.Agents.run_log_paging import _SEGMENT

from . import archive_reader
from .limits import ArchiveLimits
from .mobile_content_graph import _ContentBlocker
from .mobile_history_files import _members


@dataclass(frozen=True)
class _RunHistory:
    logical_id: str
    root_id: str
    original_root: str
    relative_path: str
    size: int
    sha256: str
    run_ids: tuple[str, ...]
    associations: tuple[tuple[str, str], ...]
    historical_run_ids: tuple[str, ...] = ()
    owner_run_id: str | None = None
    owner_association: tuple[str, str] | None = None
    segment_ids: tuple[str, ...] = ()


def _admit_header(raw, start, end, budget):
    """Account for the installed split/partition parser before its first slice."""
    size = end - start
    budget.reserve(64 + size)  # Byte slice passed to the native decoder.
    budget.reserve(64 + 4 * size)  # UTF-8 replacement decode.
    tokens = raw.count(b" ", start, end) + 1
    budget.container(tokens)  # split list, including repeated empty tokens.
    budget.container(max(0, tokens - 2))  # Native parts[2:] slice.
    budget.container(tokens)  # Prospective dictionary entries, including extras.
    position = start
    for _ in range(tokens):
        stop = raw.find(b" ", position, end)
        if stop < 0:
            stop = end
        length = stop - position
        budget.reserve(64 + 4 * length)  # Split token.
        budget.container(3)  # partition('=') result tuple.
        budget.reserve(128 + 4 * length)  # Partition key and value strings.
        position = stop + 1
    budget.container(12)  # Native record fields and integer conversion results.


def _frame_runs(raw, runs, primary, budget):
    """Account for every byte; the native recovery iterator intentionally skips."""
    position = 0
    budget.container(len(runs))
    found = set()
    while position < len(raw):
        newline = raw.find(b"\n", position)
        if newline < 0 or not raw.startswith(b"#@# ", position):
            raise ValueError("native_history_partial")
        _admit_header(raw, position, newline, budget)
        decoded = decode_record_header(raw[position:newline])
        if decoded is None:
            raise ValueError("native_history_partial")
        record, size = decoded
        end = newline + 1 + size
        if end >= len(raw) or raw[end] != 10:
            raise ValueError("native_history_partial")
        run = runs.get(record.run_id)
        if run is None:
            raise ValueError("native_history_foreign_run")
        budget.container(len(runs))
        visited = set()
        while run["id"] != primary:
            if (
                run["id"] in visited
                or run["conversation_id"] != runs[primary]["conversation_id"]
            ):
                raise ValueError("native_history_owner_unproved")
            visited.add(run["id"])
            run = runs.get(run["parent_run_id"])
            if run is None:
                raise ValueError("native_history_owner_unproved")
        found.add(record.run_id)
        position = end + 1
    budget.container(len(found))
    return tuple(sorted(found))


def _qualify_history(reads, runs, identities, *, _read_budget):
    from .mobile_consolidation_apply import _MobileCapturePartition
    from .mobile_trace_inputs import _check_trace_members

    reads.check()
    budget = _read_budget
    partition, document, sealed = reads._source_archive
    source = reads.captured.source
    if (
        type(partition) is not _MobileCapturePartition
        or partition not in reads.session._scopes
        or not partition.active
        or partition.review_only
        or partition.purpose != "legacy"
        or partition.claim is not reads.captured.claim
        or partition.journal is not reads.journal
        or partition.destination != source.path
        or sealed.encrypted_source is None
        or sealed.encrypted_source.digest != source.digest
        or sealed.manifest_bytes != source.manifest_bytes
    ):
        raise ValueError("mobile_runs_history_capture_changed")
    budget.json(sealed.manifest_bytes)
    checked = archive_reader.verify_sealed(sealed, reads.cancel, _custody=reads.custody)
    if checked != document:
        raise ValueError("mobile_runs_history_capture_changed")
    members, names = _members(
        reads.captured, document, _read_budget=budget, _scope=reads
    )
    budget.container((len(members) + len(document.files)) * 5)
    payloads = {p.logical_id: p for p in document.files}
    proofs, blockers = [], []
    budget.container(len(members) * 3)
    by_path = {}
    manifest_segments = {}
    for member in members:
        budget.container(2)
        by_path[member.root_id, member.relative_path] = member
    with archive_reader._regular(sealed.path, _custody=reads.custody) as stream:
        central = archive_reader._central_preflight(stream, ArchiveLimits())
        budget.container((len(document.files) + 1) * 16)
        for payload in document.files:
            budget.value(payload.payload, text_copies=3)
        with zipfile.ZipFile(stream) as archive:
            offsets = archive_reader._local_ranges(stream, archive.infolist(), central)
            for member in members:
                if member.sha256 is None:
                    continue
                reads.check()
                payload = payloads[member.logical_id]
                budget.reserve(128 + 2 * member.size)
                raw = bytearray()
                digest = hashlib.sha256()
                info = archive.getinfo(payload.payload)
                for chunk in archive_reader._member_chunks(
                    stream, info, offsets[info.filename], reads.cancel
                ):
                    if len(raw) + len(chunk) > member.size:
                        raise ValueError("mobile_runs_history_payload_changed")
                    digest.update(chunk)
                    raw.extend(chunk)
                if len(raw) != member.size or digest.hexdigest() != member.sha256:
                    raise ValueError("mobile_runs_history_payload_changed")
                try:
                    budget.value(member.relative_path, text_copies=4)
                    parts = PurePosixPath(member.relative_path).parts
                    if PurePosixPath(member.original_root).name in names:
                        pass
                    elif parts and parts[0] in names:
                        parts = parts[1:]
                    else:
                        raise ValueError("native_history_owner_unproved")
                    if parts == (".gitignore",):
                        budget.value(member)
                        proofs.append(
                            _RunHistory(
                                member.logical_id,
                                member.root_id,
                                member.original_root,
                                member.relative_path,
                                member.size,
                                member.sha256,
                                (),
                                (),
                            )
                        )
                        continue
                    if (
                        len(parts) != 2
                        or parts[0] not in runs
                        or runs[parts[0]]["parent_run_id"] is not None
                        or runs[parts[0]]["agent_kind"] != "primary"
                    ):
                        raise ValueError("native_history_owner_unproved")
                    primary, filename = parts
                    historical_run_ids = ()
                    if _SEGMENT.fullmatch(filename):
                        run_ids = _frame_runs(raw, runs, primary, budget)
                    elif filename == "MANIFEST":
                        budget.json(raw)
                        metadata = json.loads(raw)
                        if (
                            not isinstance(metadata, dict)
                            or metadata.get("run_id") != primary
                        ):
                            raise ValueError("native_history_owner_unproved")
                        run_ids = (primary,)
                        superseded = metadata.get("superseded_run_id")
                        if superseded:
                            previous = runs.get(superseded)
                            if (
                                previous is None
                                or superseded == primary
                                or previous["parent_run_id"] is not None
                                or previous["agent_kind"] != "primary"
                                or previous["conversation_id"]
                                != runs[primary]["conversation_id"]
                            ):
                                raise ValueError("native_history_owner_unproved")
                            # The authenticated declaration proves a recorded reference,
                            # not successful supersession or ownership of this file.
                            budget.container(1)
                            historical_run_ids = (superseded,)
                        segments = metadata.get("segments")
                        if not isinstance(segments, list):
                            raise ValueError("native_history_missing_member")
                        budget.container(len(segments))
                        budget.container(len(segments))
                        declared = []
                        for segment in segments:
                            if not isinstance(segment, str) or not _SEGMENT.fullmatch(
                                segment
                            ):
                                raise ValueError("native_history_segment_unproved")
                            budget.value(member.relative_path, text_copies=4)
                            budget.value(segment, text_copies=2)
                            budget.container(3)  # rpartition result.
                            budget.container(2)  # Lookup key.
                            relative = (
                                member.relative_path.rpartition("/")[0] + "/" + segment
                            )
                            actual = by_path.get((member.root_id, relative))
                            if actual is None or actual.sha256 is None:
                                raise ValueError("native_history_missing_member")
                            declared.append(actual.logical_id)
                        manifest_segments[member.logical_id] = tuple(declared)
                    else:
                        raise ValueError("native_history_format_unproved")
                    budget.value(member)
                    budget.container((len(run_ids) + len(historical_run_ids)) * 4)
                    associations = tuple(
                        (r, identities.mapped("run", r))
                        for group in (run_ids, historical_run_ids)
                        for r in group
                    )
                    budget.container(2)
                    owner_association = (primary, identities.mapped("run", primary))
                    proofs.append(
                        _RunHistory(
                            member.logical_id,
                            member.root_id,
                            member.original_root,
                            member.relative_path,
                            member.size,
                            member.sha256,
                            run_ids,
                            associations,
                            historical_run_ids,
                            primary,
                            owner_association,
                            manifest_segments.get(member.logical_id, ()),
                        )
                    )
                except (ValueError, KeyError, TypeError) as error:
                    reason = (
                        str(error)
                        if str(error)
                        in {
                            "native_history_partial",
                            "native_history_foreign_run",
                            "native_history_owner_unproved",
                            "native_history_missing_member",
                            "native_history_format_unproved",
                            "native_history_segment_unproved",
                        }
                        else "native_history_metadata_invalid"
                    )
                    budget.container(3)
                    blockers.append(
                        _ContentBlocker("agents.history", (member.logical_id,), reason)
                    )
    budget.container(len(proofs) * 3)
    qualified = {proof.logical_id for proof in proofs}
    retained = []
    for proof in proofs:
        if any(
            segment not in qualified
            for segment in manifest_segments.get(proof.logical_id, ())
        ):
            budget.container(3)
            blockers.append(
                _ContentBlocker(
                    "agents.history",
                    (proof.logical_id,),
                    "native_history_segment_unproved",
                )
            )
        else:
            retained.append(proof)
    proofs = retained
    budget.json(sealed.manifest_bytes)
    archive_reader.verify_sealed(sealed, reads.cancel, _custody=reads.custody)
    reads.check()
    _check_trace_members(reads)
    budget.container(len(proofs) + len(blockers))
    return tuple(proofs), tuple(blockers)
