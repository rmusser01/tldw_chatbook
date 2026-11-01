"""One native-issued, continuously locked mobile journal decoding frame."""

import hashlib
import stat
import threading
from contextlib import contextmanager

from tldw_chatbook.Utils.platform_files import os

from .admission import Admission
from .archive_reader import _check
from .journal import MAX_EVENTS, _encoded, _Event, _validate
from .limits import RECOVERY_RECORD_BYTES
from .native_files import _close_owned, flush_directory
from .storage_admission import MaintenanceSession, _mobile_coordinator_budget

_local = threading.local()


def _current(journal, parent=None):
    frame = getattr(_local, "frame", None)
    if frame is None:
        return None
    if frame.journal is not journal or parent is not None and parent != frame.parent:
        raise ValueError("mobile_journal_frame_changed")
    frame.check()
    return frame


class _Frame:
    def __init__(self):
        raise TypeError("mobile_journal_frame_is_native_issued")

    def check(self):
        if (
            type(self) is not _Frame
            or not self.active
            or getattr(_local, "frame", None) is not self
        ):
            raise ValueError("mobile_journal_frame_inactive")
        self.session._check()
        _check(self.cancel)
        if (
            self.session._mobile_journal is not self.journal
            or _mobile_coordinator_budget(self.session) is not self.budget
        ):
            raise ValueError("mobile_journal_frame_changed")
        for fd, name, expected in (
            (self.parent, None, self.journal._identity),
            (self.lock, "journal.lock", self.lock_identity),
        ):
            # The installed pinned_directory owns its parent FD lexically and
            # routes close through custody.close_descriptor; it does not insert
            # that parent into custody.fds. The separately opened flock does.
            if name is not None and fd not in self.custody.fds:
                raise ValueError("mobile_journal_frame_retired")
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != expected:
                raise ValueError("mobile_journal_frame_changed")
            if (
                info.st_uid != os.geteuid()
                or info.st_mode & 0o077
                or name is not None
                and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)
            ):
                raise ValueError("mobile_journal_frame_changed")
            named = (
                os.stat(name, dir_fd=self.parent, follow_symlinks=False)
                if name
                else self.journal.root.lstat()
            )
            if (named.st_dev, named.st_ino) != expected:
                raise ValueError("mobile_journal_frame_changed")

    def census(self):
        self.check()
        iterator = os.scandir(self.parent)
        self.custody.directory_iterators.append(iterator)
        names = set()
        try:
            for entry in iterator:
                if len(names) >= MAX_EVENTS + 4:
                    raise ValueError("journal_limit")
                self.budget.value(entry.name, text_copies=2)
                self.budget.container(1)
                names.add(entry.name)
        finally:
            self.custody.close_directory_iterator(iterator)
        return names

    def fingerprint(self, name):
        self.check()
        self.budget.reserve(2048)
        fd = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.parent
        )
        self.custody.fds.append(fd)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_nlink != 1
                or info.st_mode & 0o077
                or not 0 <= info.st_size <= RECOVERY_RECORD_BYTES
            ):
                raise ValueError("journal_record_invalid")
            digest = hashlib.sha256()
            remaining = info.st_size
            while remaining:
                _check(self.cancel)
                # Both the bytearray and its iovec/view are owned once by this
                # frame. Only the read count and remaining scalar change here.
                self.budget.reserve(256)
                count = os.readv(fd, self.iovec)
                if (
                    not count
                    or count > remaining
                    or count < min(len(self.buffer), remaining)
                ):
                    raise ValueError("journal_record_changed")
                remaining -= count
                if count == len(self.buffer):
                    digest.update(self.buffer_view)
                else:
                    self.budget.reserve(256)
                    digest.update(self.buffer_view[:count])
            identity = lambda row: (
                row.st_dev,
                row.st_ino,
                row.st_mode,
                row.st_nlink,
                row.st_uid,
                row.st_size,
                row.st_mtime_ns,
                row.st_ctime_ns,
            )
            if identity(info) != identity(os.fstat(fd)) or identity(info) != identity(
                os.stat(name, dir_fd=self.parent, follow_symlinks=False)
            ):
                raise ValueError("journal_record_changed")
            return identity(info), digest.hexdigest()
        finally:
            _close_owned(fd, self.custody)

    def verify(self):
        if self.census() != set(self.files) | {"journal.lock"}:
            raise ValueError("journal_artifacts_changed")
        for name, expected in self.files.items():
            if self.fingerprint(name) != expected:
                raise ValueError("journal_record_changed")

    def verification_cost(self):
        """Bound the unchanged full census/hash pass without spending it twice."""
        self.check()
        # Fingerprint rejects a short nonfinal read, so each valid fixed chunk
        # costs one scalar reservation and the final partial view costs one.
        self.budget.reserve(1024 + 256 * len(self.files))
        return (
            64
            + 8 * len("journal.lock")
            + 96
            + sum(
                64
                + 8 * len(name)
                + 96
                + 2048
                + 256 * ((value[0][5] + len(self.buffer) - 1) // len(self.buffer))
                + (256 if value[0][5] % len(self.buffer) else 0)
                for name, value in self.files.items()
            )
        )

    def require_capacity(self, amount):
        """Check foreseeable effect completion; real reservations still follow."""
        self.check()
        self.budget.require_capacity(amount)

    def artifact_buffer(self):
        """Own one byte workspace for repeated full native artifact hashes."""
        self.check()
        if self._artifact_buffer is None:
            self.budget.reserve(65536 + 1024)
            buffer = bytearray(65536)
            view = memoryview(buffer)
            self._artifact_buffer = buffer, view, [view]
        return self._artifact_buffer

    def append(self, event, evidence):
        """Append through exact native grammar while this owner retains the lock."""
        self.verify()
        if not event.startswith("mobile_") or len(self.records) >= MAX_EVENTS:
            raise ValueError("mobile_journal_event_required")
        self.budget.value(evidence, text_copies=8)
        self.budget.reserve(8192)
        validated = _validate(event, evidence, self.records)
        record = _Event(
            operation_id=self.journal.operation_id,
            sequence=len(self.records),
            previous=self.files[f"{len(self.records) - 1:06d}.json"][1]
            if self.records
            else "",
            event=event,
            evidence=validated,
        )
        encoded = _encoded(record)
        if len(encoded) > RECOVERY_RECORD_BYTES:
            raise ValueError("journal_record_limit")
        name = f"{len(self.records):06d}.json"
        Admission._write_new_record(self.parent, name, encoded)
        flush_directory(self.parent)
        actual = self.fingerprint(name)
        if actual[1] != hashlib.sha256(encoded).hexdigest():
            raise ValueError("journal_record_changed")
        self.files[name] = actual
        self.records.append(record)
        return record


@contextmanager
def _locked_mobile(journal, session, custody, cancel):
    """Issue only from actual maintenance and keep one exclusive native lock."""
    with _locked_frame(journal, session, custody, cancel, None) as frame:
        yield frame


@contextmanager
def _verify_current_mobile(journal, session, custody, cancel):
    """Verify only the existing frame of a fixed public owner transaction."""
    from tldw_chatbook.Chat.console_trace_read_budget import _mobile_budget_pair

    from .storage_admission import _mobile_admission

    frame = _current(journal)
    if frame is None:
        raise ValueError("mobile_current_frame_required")
    if (
        frame.session is not session
        or frame.custody is not custody
        or frame.cancel is not cancel
    ):
        raise ValueError("mobile_verification_frame_changed")
    admission = _mobile_admission(session)
    if admission[4] is not frame.budget or not _mobile_budget_pair(
        frame.budget, admission[2]
    ):
        raise ValueError("mobile_verification_frame_changed")
    frame.verify()
    yield frame
    frame.verify()


@contextmanager
def _locked_mobile_rollback(journal, session, claim, cancel):
    """Authenticate the retained native claim before associating a restart."""
    from .mobile_consolidation_apply import _SourceClaim

    if type(claim) is not _SourceClaim or claim.session is not session:
        raise ValueError("mobile_source_claim_inactive")
    _SourceClaim._check_binding(claim)
    with _locked_frame(journal, session, claim.custody, cancel, claim) as frame:
        yield frame


def _bind_rollback(journal, session, claim, records, budget):
    """Bind only the actual lock's freshly validated history and current claim."""
    from pathlib import Path

    from . import bootstrap, control_records
    from .journal import _mobile_state, _MobileSourceClaim
    from .mobile_consolidation import CanonicalConsoleStores, _path_witnesses
    from .mobile_consolidation_apply import _SourceClaim

    budget.value(records[0].evidence, text_copies=4)
    prepared, _ = _mobile_state(records)
    control_records._mobile_pending(
        session, journal, prepared, _budget=budget, _custody=claim.custody
    )
    with claim.custody.pinned_directory(claim.path.parent) as parent:
        raw = bootstrap._read(
            parent, claim.path.name, _budget=budget, _custody=claim.custody
        )
    budget.value(raw, text_copies=8)
    source = _MobileSourceClaim.model_validate(raw)
    if (
        source.operation_id != journal.operation_id
        or Path(source.control_root) != journal.root.parent
        or source.claim_path != str(claim.path)
        or next(row.evidence for row in records if row.event == "mobile_claim_intended")
        != raw
    ):
        raise ValueError("mobile_source_claim_changed")
    claim.receipt = next(
        row.evidence for row in records if row.event == "mobile_claim_observed"
    )
    _SourceClaim._check_binding(claim, _budget=budget, _custody=claim.custody)
    budget.container(len(source.source_witnesses) * 5)
    if _path_witnesses((Path(source.source_dir),), _budget=budget)[0][1] != tuple(
        (row.path, row.device, row.inode, row.mode) for row in source.source_witnesses
    ):
        raise ValueError("mobile_source_changed")
    budget.reserve(32768)
    destination = CanonicalConsoleStores(
        *(Path(row.witnesses[-1].path) for row in prepared.targets)
    )
    if (
        prepared.source_fingerprint != source.source_fingerprint
        or destination.identity != source.destination_identity
    ):
        raise ValueError("mobile_rollback_scope_changed")
    session._mobile_journal = journal
    claim.journal = journal


@contextmanager
def _locked_frame(journal, session, custody, cancel, rollback_claim):
    if (
        type(session) is not MaintenanceSession
        or rollback_claim is None
        and session._mobile_journal is not journal
        or getattr(_local, "frame", None) is not None
    ):
        raise ValueError("mobile_native_journal_required")
    session._check()
    budget = _mobile_coordinator_budget(session)
    if budget is None:
        raise ValueError("mobile_native_budget_required")
    if not hasattr(os, "readv"):
        raise ValueError("mobile_journal_stream_unavailable")
    budget.reserve(8192)
    with journal._locked(exclusive=True, _custody=custody) as parent:
        records = journal._records(parent, _budget=budget, _custody=custody)
        if not records or records[0].event != "mobile_prepared":
            raise ValueError("mobile_native_journal_required")
        if rollback_claim is not None:
            _bind_rollback(journal, session, rollback_claim, records, budget)
        frame = object.__new__(_Frame)
        frame.journal, frame.session, frame.parent = journal, session, parent
        frame.budget, frame.custody, frame.cancel = budget, custody, cancel
        lock_info = os.stat("journal.lock", dir_fd=parent, follow_symlinks=False)
        frame.lock_identity = lock_info.st_dev, lock_info.st_ino
        locks = [
            fd
            for fd in custody.fds
            if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == frame.lock_identity
        ]
        if len(locks) != 1:
            raise ValueError("mobile_native_journal_lock_required")
        frame.lock, frame.records, frame.files = locks[0], records, {}
        frame.buffer = bytearray(4096)
        frame.buffer_view = memoryview(frame.buffer)
        budget.reserve(128)
        frame.iovec = [frame.buffer_view]
        frame._artifact_buffer = None
        frame.active = True
        _local.frame = frame
        try:
            names = frame.census()
            expected = {f"{index:06d}.json" for index in range(len(records))} | {
                "journal.lock"
            }
            if records[0].evidence.get("predecessor") is not None and any(
                row.event == "mobile_claim_observed" for row in records
            ):
                # _records just authenticated this exact local claim and receipt.
                expected.add("mobile-source-claim.json")
                frame.files["mobile-source-claim.json"] = frame.fingerprint(
                    "mobile-source-claim.json"
                )
            if names != expected:
                raise ValueError("mobile_journal_artifacts_changed")
            for index, record in enumerate(records):
                name = f"{index:06d}.json"
                actual = frame.fingerprint(name)
                # Native JSON escaping can expand each source byte to six ASCII
                # characters. Price Unicode + encoded bytes before serialization.
                budget.reserve(512 + 32 * actual[0][5])
                if actual[1] != hashlib.sha256(_encoded(record)).hexdigest():
                    raise ValueError("mobile_journal_encoding_changed")
                frame.files[name] = actual
            yield frame
            frame.verify()
        finally:
            frame.active = False
            _local.frame = None
