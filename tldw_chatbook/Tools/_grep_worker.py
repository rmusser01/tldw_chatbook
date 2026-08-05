"""Standalone child-process worker for `GrepFiles`'s regex search step.

TASK-843. Spawned by `Tools.file_operation_tools._run_grep_subprocess` as
``sys.executable -S <this file>``. Deliberately a PLAIN SCRIPT with NO
import of `tldw_chatbook` itself, and no import of anything beyond the
Python standard library: the whole point of running in a separate
process is to be trivially killable (`Popen.kill()`, SIGKILL on POSIX)
without dragging along whatever state the running `tldw_chatbook`
process happens to hold (a live Textual app, open DB connections, loaded
config) into a new process whose only job is to run a possibly-
adversarial regular expression against file content. Importing the
parent package here would reintroduce exactly the "child inherits
surprising state" risk this design exists to avoid, and would slow down
every single invocation for no benefit -- this worker needs none of it.

Trust boundary: every path handed to this worker via stdin MUST already
be fully validated by the parent (containment against the sandbox/
workspace roots, the sensitive-path denylist, the hidden-component rule
-- see `file_operation_tools._iter_candidates_across_roots`). This worker
performs NONE of that validation itself; it trusts its input completely
and exists to do exactly one thing: run a regex over lines read from a
fixed list of files, under the same bounds `GrepFiles` already enforces
in-process (line-length cap, per-file byte cap, total-lines-scanned cap).

Protocol: a single JSON object read from stdin --
``{"pattern": str, "file_paths": [str, ...], "max_matches": int,
"max_line_search_chars": int, "max_lines_scanned": int,
"max_file_bytes": int}`` -- and a single JSON object written to stdout --
either ``{"matches": [{"path", "line_number", "line"}, ...],
"lines_scanned": int}`` or ``{"error": str}``. Always exits 0: a bad
pattern or an unreadable file is reported as part of the JSON payload,
never as a nonzero exit code (a nonzero exit means something
unanticipated escaped every try/except below, which `_run_grep_subprocess`
treats as a hard failure).

What this process's own `RLIMIT_CPU` self-limit (see `_apply_cpu_limit`)
does and does not add: it is a defense-in-depth measure for the case
where the PARENT itself dies before it can ever call `Popen.kill()` on
this worker (e.g. the whole host application crashes) -- without it, an
orphaned worker would have nothing left to stop it. It is POSIX-only
(the `resource` module does not exist on Windows) and is NOT the primary
guarantee: `_run_grep_subprocess`'s `communicate(timeout=...)` +
`kill()` is what actually bounds this worker's runtime on every
platform, including Windows, in the ordinary (non-orphaned) case.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


#: Generous relative to `GrepFiles`'s own ~18s subprocess ceiling
#: (`_GREP_SUBPROCESS_TIMEOUT_SECONDS`) -- this is a backstop for the
#: orphaned-parent case, not the primary timeout, so it deliberately does
#: not try to match that ceiling exactly.
_CPU_LIMIT_SECONDS = 60


def _apply_cpu_limit() -> None:
    """Best-effort self-imposed CPU cap (POSIX only). See module docstring.

    Silently does nothing if the `resource` module is unavailable
    (Windows) or if the limit cannot be set for any reason -- this is
    additive defense in depth, never the primary guarantee, so a failure
    here must not stop the worker from running the actual search.
    """
    try:
        import resource
    except ImportError:
        return
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        # A process cannot raise its own hard limit without privilege, so
        # keep whatever hard limit is already in force unless it is
        # infinite (the common case), in which case cap it to something
        # generous. The soft limit -- the one that actually triggers
        # SIGXCPU -- is the smaller of our desired cap and that hard
        # limit, since setrlimit rejects soft > hard.
        new_hard = (
            _CPU_LIMIT_SECONDS + 5 if hard == resource.RLIM_INFINITY else hard
        )
        new_soft = min(_CPU_LIMIT_SECONDS, new_hard)
        resource.setrlimit(resource.RLIMIT_CPU, (new_soft, new_hard))
    except (ValueError, OSError):
        pass


def run_search(request: dict) -> dict:
    """Execute one grep request in-process. Never raises.

    Args:
        request: Parsed JSON request; see this module's docstring for
            the required keys.

    Returns:
        ``{"matches": [...], "lines_scanned": int}`` on success
        (including a legitimate zero-match result), or ``{"error": str}``.
    """
    try:
        pattern = request["pattern"]
        file_paths = request["file_paths"]
        max_matches = int(request["max_matches"])
        max_line_search_chars = int(request["max_line_search_chars"])
        max_lines_scanned = int(request["max_lines_scanned"])
        max_file_bytes = int(request["max_file_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"error": f"malformed grep worker request: {exc}"}

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return {"error": f"invalid regular expression: {exc}"}

    matches: list[dict] = []
    lines_scanned = 0
    for path_str in file_paths:
        if len(matches) >= max_matches or lines_scanned >= max_lines_scanned:
            break
        path = Path(path_str)
        try:
            if path.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue
        # Streamed line-by-line rather than `read_text()` + `splitlines()`
        # -- same rationale as the in-process implementation this replaces:
        # a single pathological file with no newlines would otherwise force
        # a large peak allocation. `max_file_bytes` still bounds that worst
        # case independent of this.
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for number, line in enumerate(fh, start=1):
                    lines_scanned += 1
                    # Search only a length-capped slice of the line, never
                    # the full line -- this is the FIRST line of defence
                    # against a catastrophic-backtracking pattern; the
                    # subprocess boundary this worker runs inside of is
                    # what actually bounds CPU exposure once a pattern gets
                    # past this cap (see `_run_grep_subprocess`'s
                    # docstring in `file_operation_tools.py`).
                    if regex.search(line[:max_line_search_chars]):
                        matches.append(
                            {
                                "path": path_str,
                                "line_number": number,
                                "line": line.rstrip("\n")[:500],
                            }
                        )
                    if (
                        len(matches) >= max_matches
                        or lines_scanned >= max_lines_scanned
                    ):
                        break
        except OSError:
            continue

    return {"matches": matches, "lines_scanned": lines_scanned}


def main() -> int:
    """Entry point: read one JSON request from stdin, write one to stdout."""
    _apply_cpu_limit()
    raw_request = sys.stdin.read()
    try:
        request = json.loads(raw_request)
    except json.JSONDecodeError as exc:
        json.dump({"error": f"malformed grep worker request: {exc}"}, sys.stdout)
        return 0
    if not isinstance(request, dict):
        json.dump({"error": "malformed grep worker request"}, sys.stdout)
        return 0
    result = run_search(request)
    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
