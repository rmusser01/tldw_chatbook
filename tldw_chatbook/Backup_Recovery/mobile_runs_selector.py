"""Fixed source-selector accounting for the actual pure RunLog operation."""

import hashlib
import stat

from tldw_chatbook.Utils.platform_files import os

from . import archive_reader, native_files


def _charge_toml(payload, budget):
    """Reserve possible native structures; tomllib alone validates the grammar.

    Each lexical value/key pays for its retained string/scalar and a container
    entry. Each structural separator pays for a container/parser-prefix record;
    in particular dots pay for implicit dictionaries and native flag prefixes.
    Quoted text/comment punctuation does not create structures. Multiline closing
    quote runs include the native permitted fourth/fifth content quotes.
    """
    budget.reserve(64 + 4 * len(payload))  # UTF-8 decoding before tomllib.
    budget.container(8)  # Native root, parser flags and initial namespace.
    position = 0
    length = len(payload)
    span = maximum_span = 0
    after_dot = True
    while position < length:
        code = payload[position]
        if code in b" \t\r\n":
            if code in (10, 13):
                span, after_dot = 0, True
            position += 1
        elif code == 35:
            end = payload.find(b"\n", position)
            position = length if end < 0 else end + 1
            span, after_dot = 0, True
        elif code in (34, 39):
            start = position
            triple = payload[position : position + 3] == bytes((code,)) * 3
            position += 3 if triple else 1
            while position < length:
                if code == 34 and payload[position] == 92:
                    position += 2
                elif payload[position] == code:
                    end = position + 1
                    if triple:
                        while end < length and payload[end] == code:
                            end += 1
                        if end - position < 3:
                            position = end
                            continue
                    position = end
                    break
                else:
                    position += 1
            budget.reserve(64 + 4 * (position - start))
            budget.container(4)  # Value/key, key tuple and native namespace entry.
            span = span + 1 if after_dot else 1
            maximum_span = max(maximum_span, span)
            after_dot = False
        elif code in b"[].{},=":
            # A dot may create an implicit table AND its parser flags. Array-table
            # brackets pay on every occurrence, even when prefixes already exist.
            budget.container(8)
            if code == 61:
                # Native assignment materializes header + key-parent tuples.
                # Each is bounded by an observed lexical dotted-path span;
                # pay again for every assignment, including reused headers.
                budget.container(2 * maximum_span)
            if code == 46:
                after_dot = True
            else:
                span, after_dot = 0, True
            position += 1
        else:
            start = position
            while position < length and payload[position] not in b" \t\r\n#\"'[].{},=":
                position += 1
            budget.reserve(64 + 4 * (position - start))
            budget.container(4)
            span = span + 1 if after_dot else 1
            maximum_span = max(maximum_span, span)
            after_dot = False


def _source_config(reads, budget):
    """Read only this live scope's native admitted selector, without chmod."""
    from .bootstrap import MAX_RECORD
    from .inventory import _parse_profile_config
    from .mobile_captured_sqlite import _CapturedSQLite
    from .mobile_consolidation import _path_witnesses

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_runs_selector_scope_required")
    reads.check()
    preview = reads.captured.preview
    selector = preview.source_selector
    provenance = next(
        (p for p in preview.selector_provenance if p.selector == selector), None
    )
    if (
        provenance is None
        or provenance.fingerprint is None
        or sum(p.selector == selector for p in preview.selector_provenance) != 1
    ):
        raise ValueError("mobile_runs_selector_provenance_unproved")
    budget.value(provenance.witnesses, text_copies=3)
    if _path_witnesses((selector,))[0][1] != provenance.witnesses:
        raise ValueError("mobile_runs_selector_changed")
    with native_files.pinned_directory(
        selector.parent, _custody=reads.custody
    ) as parent:
        fd = os.open(
            selector.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        reads.custody.fds.append(fd)
        try:
            info = os.fstat(fd)
            before = archive_reader._identity(info)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or not 0 <= info.st_size <= MAX_RECORD
            ):
                raise ValueError("mobile_runs_selector_unverified")
            budget.reserve(64 + info.st_size + 1)
            raw = os.read(fd, info.st_size + 1)
            if (
                len(raw) != info.st_size
                or before != archive_reader._identity(os.fstat(fd))
                or before
                != archive_reader._identity(
                    os.stat(selector.name, dir_fd=parent, follow_symlinks=False)
                )
                or hashlib.sha256(raw).hexdigest() != provenance.fingerprint
            ):
                raise ValueError("mobile_runs_selector_changed")
        finally:
            native_files._close_owned(fd, reads.custody)
    if _path_witnesses((selector,))[0][1] != provenance.witnesses:
        raise ValueError("mobile_runs_selector_changed")
    reads.check()
    return _parse_profile_config(raw, _read_budget=budget)
