"""Fixed ancestor reads owned by one native source flock, never decoded caches."""

import hashlib
import stat
import threading
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from . import bootstrap
from .admission import Admission
from .journal import MAX_EVENTS, _encoded, _mobile_state
from .limits import RECOVERY_RECORD_BYTES
from .mobile_lineage import CLAIM_NAME, _read_lineage
from .mobile_preview_budget import path_cost
from .recovery_copies import _journal


def _identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _raw(fd, parent, name, budget, custody):
    """Hash the actual held descriptor and its current exact native name."""
    if fd not in custody.fds or parent not in custody.fds or custody.close_failed:
        raise ValueError("mobile_ancestor_retired")
    budget.reserve(4096)
    before = os.fstat(fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
        or before.st_mode & 0o077
        or not 0 <= before.st_size <= RECOVERY_RECORD_BYTES
    ):
        raise ValueError("mobile_ancestor_file_changed")
    digest = hashlib.sha256()
    position = 0
    while position < before.st_size:
        size = min(65536, before.st_size - position)
        budget.reserve(size + 512)
        block = os.pread(fd, size, position)
        if len(block) != size:
            raise ValueError("mobile_ancestor_file_changed")
        digest.update(block)
        position += size
    expected = _identity(before)
    if expected != _identity(os.fstat(fd)) or expected != _identity(
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    ):
        raise ValueError("mobile_ancestor_file_changed")
    return expected, digest.hexdigest()


class _AncestorClosure:
    def __init__(self):
        raise TypeError("mobile_ancestor_custody_is_native_issued")

    def check(self):
        from .mobile_consolidation_apply import _SourceClaim
        from .mobile_journal_frame import _local
        from .storage_admission import _mobile_admission

        claim = self.claim
        if (
            type(self) is not _AncestorClosure
            or getattr(claim, "_ancestors", None) is not self
            or type(claim) is not _SourceClaim
            or threading.get_ident() != self.thread
            or claim.journal is not self.child
            or claim.operation_id != self.child.operation_id
            or claim.session._mobile_journal is not self.child
            or claim.custody is not self.custody
            or _mobile_admission(claim.session)[4] is not self.budget
        ):
            raise ValueError("mobile_ancestor_scope_changed")
        _SourceClaim._check_native_binding(claim)
        frame = getattr(_local, "frame", None)
        if frame is not None:
            frame.check()
            if (
                frame.journal is not self.child
                or frame.session is not claim.session
                or frame.custody is not self.custody
            ):
                raise ValueError("mobile_ancestor_frame_changed")

    def verify(self):
        self.check()
        for path, (parent, directory, files, census) in self.roots.items():
            self.budget.reserve(8192 + path_cost(path, copies=4))
            if parent not in self.custody.fds:
                raise ValueError("mobile_ancestor_retired")
            with self.custody.pinned_directory(path) as current:
                held, named = os.fstat(parent), os.fstat(current)
                identities = (
                    (_identity(held), _identity(named))
                    if census
                    else (
                        (held.st_dev, held.st_ino, held.st_mode, held.st_uid),
                        (named.st_dev, named.st_ino, named.st_mode, named.st_uid),
                    )
                )
                if any(value != directory for value in identities):
                    raise ValueError("mobile_ancestor_directory_changed")
            if census:
                names = bootstrap._bounded_names(
                    parent, MAX_EVENTS + 4, self.budget, self.custody
                )
                self.budget.container(2 * len(names))
                if set(names) != set(files):
                    raise ValueError("mobile_ancestor_census_changed")
            for name, (fd, expected) in files.items():
                if _raw(fd, parent, name, self.budget, self.custody) != expected:
                    raise ValueError("mobile_ancestor_file_changed")

    def read(self, control, operation):
        """Fresh complete raw/native parsing, only for this fixed ancestor."""
        self.verify()
        key = (str(control), operation)
        if key not in self.journals:
            raise ValueError("mobile_ancestor_unknown")
        journal = self.journals[key]
        parent, _, files, _ = self.roots[journal.root]
        records = journal._parse_records(
            parent, _budget=self.budget, _custody=self.custody
        )
        for index, row in enumerate(records):
            expected = files[f"{index:06d}.json"][1]
            self.budget.reserve(512 + 32 * expected[0][5])
            if hashlib.sha256(_encoded(row)).hexdigest() != expected[1]:
                raise ValueError("mobile_ancestor_encoding_changed")
        if (
            len(records) < 2
            or records[-2].event != "mobile_rolled_back"
            or records[-1].event != "mobile_successor_intended"
        ):
            raise ValueError("mobile_ancestor_not_immutable")
        self.verify()
        return journal, records


def _retain(closure, path, parent, names, *, census):
    if path in closure.roots:
        raise ValueError("mobile_ancestor_directory_alias")
    closure.budget.reserve(4096 + path_cost(path, copies=4))
    held = os.dup(parent)
    closure.custody.fds.append(held)
    files = {}
    for name in names:
        closure.budget.reserve(1024 + 16 * len(name))
        fd = Admission._open(parent, name, os.O_RDONLY)
        closure.custody.fds.append(fd)
        files[name] = fd, _raw(fd, held, name, closure.budget, closure.custody)
    info = os.fstat(parent)
    directory = (
        _identity(info)
        if census
        else (info.st_dev, info.st_ino, info.st_mode, info.st_uid)
    )
    closure.roots[path] = held, directory, files, census


def _issue_ancestors(claim, child, source):
    """After claim publication, before child EX, retain immutable ancestors."""
    from .mobile_consolidation_apply import _SourceClaim
    from .mobile_journal_frame import _local
    from .storage_admission import _mobile_admission

    _SourceClaim._check_native_binding(claim)
    if claim.path == claim.original_path:
        return
    if (
        getattr(_local, "frame", None) is not None
        or getattr(claim, "_ancestors", None) is not None
    ):
        raise ValueError("mobile_ancestor_order_invalid")
    budget = _mobile_admission(claim.session)[4]
    if budget is None:
        raise ValueError("mobile_public_successor_required")
    leaf = _read_lineage(claim.original_path, source, budget, claim.custody)
    if leaf is None or leaf[0].root != child.root or leaf[2] is None:
        raise ValueError("mobile_lineage_changed")
    budget.value(leaf[1][0].evidence, text_copies=4)
    prepared, _ = _mobile_state(leaf[1])
    budget.reserve(8192)
    closure = object.__new__(_AncestorClosure)
    closure.claim, closure.child, closure.custody = claim, child, claim.custody
    closure.budget, closure.thread = budget, threading.get_ident()
    closure.roots, closure.journals = {}, {}
    predecessor = prepared.predecessor
    seen = set()
    while predecessor is not None:
        budget.reserve(4096)
        if len(seen) >= bootstrap.MAX_RECORDS or predecessor.operation_id in seen:
            raise ValueError("mobile_lineage_limit")
        seen.add(predecessor.operation_id)
        budget.value(predecessor.control_root, text_copies=8)
        journal = _journal(Path(predecessor.control_root), predecessor.operation_id)
        with journal._locked(exclusive=False, _custody=claim.custody) as parent:
            rows = journal._records(parent, _budget=budget, _custody=claim.custody)
            if (
                len(rows) < 2
                or rows[-2].event != "mobile_rolled_back"
                or rows[-1].event != "mobile_successor_intended"
            ):
                raise ValueError("mobile_ancestor_not_immutable")
            names = bootstrap._bounded_names(
                parent, MAX_EVENTS + 4, budget, claim.custody
            )
            budget.reserve(2048 + 512 * len(rows))
            expected = {f"{index:06d}.json" for index in range(len(rows))} | {
                "journal.lock"
            }
            if rows[0].evidence.get("predecessor") is not None:
                expected.add(CLAIM_NAME)
            if set(names) != expected:
                raise ValueError("mobile_ancestor_census_changed")
            _retain(closure, journal.root, parent, names, census=True)
            files = closure.roots[journal.root][2]
            for index, row in enumerate(rows):
                receipt = files[f"{index:06d}.json"][1]
                budget.reserve(512 + 32 * receipt[0][5])
                if hashlib.sha256(_encoded(row)).hexdigest() != receipt[1]:
                    raise ValueError("mobile_ancestor_encoding_changed")
            closure.journals[str(journal.root.parent), journal.operation_id] = journal
            budget.value(rows[0].evidence, text_copies=4)
            prepared, _ = _mobile_state(rows)
            predecessor = prepared.predecessor
        # The actual native SH descriptor retires here. Retained lock FDs are
        # separate opens with no flock; no ancestor flock survives into child EX.
    budget.reserve(8192 + path_cost(claim.original_path, copies=4))
    with claim.custody.pinned_directory(claim.original_path.parent) as parent:
        _retain(
            closure,
            claim.original_path.parent,
            parent,
            (claim.original_path.name,),
            census=False,
        )
    claim._ancestors = closure
    closure.verify()


def _ancestor_records(control, operation):
    """Dispatch only through the actual current frame's native source owner."""
    from .mobile_consolidation_apply import _SourceClaim
    from .mobile_journal_frame import _local

    frame = getattr(_local, "frame", None)
    if frame is None:
        raise ValueError("mobile_ancestor_frame_required")
    frame.check()
    frame.budget.container(len(frame.session._scopes))
    claims = [
        row
        for row in frame.session._scopes
        if type(row) is _SourceClaim
        and getattr(row, "journal", None) is frame.journal
        and row.custody is frame.custody
        and getattr(row, "_ancestors", None) is not None
    ]
    if len(claims) != 1:
        raise ValueError("mobile_ancestor_scope_changed")
    return claims[0]._ancestors.read(control, operation)
