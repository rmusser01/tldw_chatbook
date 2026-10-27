"""Pure source-owned RunLog qualification within the captured SQLite lifetime."""

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from tldw_chatbook.Agents.run_log_format import decode_record_header

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


def _frame_runs(raw, runs, primary, budget):
    """Account for every byte; the native recovery iterator intentionally skips."""
    position = 0
    budget.container(len(runs))
    found = set()
    while position < len(raw):
        newline = raw.find(b"\n", position)
        if newline < 0 or not raw.startswith(b"#@# ", position):
            raise ValueError("native_history_partial")
        budget.reserve(1024 + 12 * (newline - position))
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
                    ):
                        raise ValueError("native_history_owner_unproved")
                    primary, filename = parts
                    if (
                        filename.startswith("logs.")
                        and filename.endswith(".txt")
                        and filename[5:-4].isdigit()
                    ):
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
                            if superseded not in runs:
                                raise ValueError("native_history_foreign_run")
                            run_ids += (superseded,)
                        segments = metadata.get("segments")
                        if not isinstance(segments, list) or any(
                            not isinstance(s, str)
                            or not any(
                                m.root_id == member.root_id
                                and PurePosixPath(m.relative_path).parent
                                == PurePosixPath(member.relative_path).parent
                                and PurePosixPath(m.relative_path).name == s
                                for m in members
                            )
                            for s in segments
                        ):
                            raise ValueError("native_history_missing_member")
                    else:
                        raise ValueError("native_history_format_unproved")
                    budget.value(member)
                    budget.container(len(run_ids) * 4)
                    associations = tuple(
                        (r, identities.mapped("run", r)) for r in run_ids
                    )
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
                        }
                        else "native_history_metadata_invalid"
                    )
                    budget.container(3)
                    blockers.append(
                        _ContentBlocker("agents.history", (member.logical_id,), reason)
                    )
    budget.json(sealed.manifest_bytes)
    archive_reader.verify_sealed(sealed, reads.cancel, _custody=reads.custody)
    reads.check()
    _check_trace_members(reads)
    budget.container(len(proofs) + len(blockers))
    return tuple(proofs), tuple(blockers)
