"""TASK-26020: @-references expanded into the Console prompt before send.

``@path`` (optionally ``#L10-20``), ``@folder/``, ``@diff`` and ``@staged`` are
expanded inline before the message is sent. The parser and expander here are
pure: filesystem/git access is injected, so the security guarantees are tested
directly.

Two invariants carry the safety (AC#7 + AC#3):
- An ``@`` preceded by a word character is an email/handle, not a reference
  (``bob@example.com`` is left untouched).
- A candidate that resolves to nothing (a decorator like ``@property``, a typo)
  is left as literal text -- only a real, allowed file/folder or the ``@diff``/
  ``@staged`` tokens are expanded. Resolution is delegated to a resolver that
  applies the SAME allowed-roots + sensitive-path authority the file tools use,
  so a reference can never read what the tools cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# A reference candidate: `@` not preceded by a word char, then a run of
# non-space characters. The lookbehind excludes emails/handles (AC#7).
_REFERENCE_RE = re.compile(r"(?<!\w)@([^\s@]+)")

_SPECIAL_TOKENS = ("diff", "staged")


@dataclass(frozen=True)
class ReferenceCandidate:
    raw: str      # e.g. "@src/main.py#L10-20"
    token: str    # e.g. "src/main.py#L10-20"
    start: int
    end: int


@dataclass(frozen=True)
class ReferenceRecord:
    """What happened to one reference (shown in the transcript -- AC#6)."""

    raw: str
    kind: str      # "file" | "folder" | "diff" | "staged" | "refused"
    ok: bool
    detail: str    # short human-readable summary / refusal reason


@dataclass(frozen=True)
class ReferenceExpansion:
    expanded_text: str
    records: List[ReferenceRecord] = field(default_factory=list)


def find_reference_candidates(text: str) -> List[ReferenceCandidate]:
    """Return every ``@`` reference candidate in ``text`` (AC#7 aware)."""
    out: List[ReferenceCandidate] = []
    for m in _REFERENCE_RE.finditer(text or ""):
        # Trim trailing punctuation that is clearly not part of a path.
        token = m.group(1).rstrip(".,;:)]}\"'")
        if not token:
            continue
        out.append(
            ReferenceCandidate(
                raw="@" + token,
                token=token,
                start=m.start(),
                end=m.start() + 1 + len(token),
            )
        )
    return out


#: resolve(token) -> None (not a reference / leave literal)
#:               or ("file",  content:str, line_range:(int,int)|None)
#:               or ("folder", listing:str, None)
#:               or ("refused", reason:str, None)
Resolver = Callable[[str], Optional[Tuple[str, str, Optional[Tuple[int, int]]]]]
GitRunner = Callable[[str], str]


def _slice_lines(content: str, line_range: Optional[Tuple[int, int]]) -> str:
    if not line_range:
        return content
    start, end = line_range
    lines = content.splitlines()
    lo = max(1, start)
    hi = min(len(lines), end if end else start)
    return "\n".join(lines[lo - 1 : hi])


def _block(header: str, body: str) -> str:
    body = body.rstrip("\n")
    return f"\n<{header}>\n{body}\n</{header}>\n"


def expand_references(
    text: str,
    *,
    resolve: Resolver,
    git_runner: GitRunner,
) -> ReferenceExpansion:
    """Expand references in ``text`` before send.

    Each candidate is resolved: a real allowed file/folder is inlined (with an
    optional line range), ``@diff``/``@staged`` run git, an explicit refusal is
    reported (never read), and anything that resolves to nothing is left as
    literal text. Returns the rewritten text plus a record per expansion.
    """
    records: List[ReferenceRecord] = []
    candidates = find_reference_candidates(text)
    if not candidates:
        return ReferenceExpansion(expanded_text=text, records=[])

    # Rebuild the string left-to-right, replacing only expanded candidates.
    pieces: List[str] = []
    cursor = 0
    for cand in candidates:
        pieces.append(text[cursor:cand.start])
        replacement = cand.raw  # default: leave literal
        low = cand.token.lower()

        if low in _SPECIAL_TOKENS:
            try:
                out = git_runner(low)
            except Exception as exc:  # noqa: BLE001 - a git failure is a refusal, not a crash
                records.append(ReferenceRecord(cand.raw, "refused", False, f"git {low} failed: {exc}"))
            else:
                if out and out.strip():
                    replacement = cand.raw + _block(f"git-{low}", out)
                    records.append(ReferenceRecord(cand.raw, low, True, f"included git {low}"))
                else:
                    records.append(ReferenceRecord(cand.raw, low, True, f"git {low} was empty"))
            pieces.append(replacement)
            cursor = cand.end
            continue

        resolved = resolve(cand.token)
        if resolved is None:
            # not a reference (decorator/typo/outside vocabulary) -> literal
            pieces.append(replacement)
            cursor = cand.end
            continue

        kind, payload, line_range = resolved
        if kind == "refused":
            records.append(ReferenceRecord(cand.raw, "refused", False, payload))
            # leave the literal token; do NOT inject the refused content
            pieces.append(cand.raw)
        elif kind == "file":
            body = _slice_lines(payload, line_range)
            rng = f"#L{line_range[0]}-{line_range[1]}" if line_range else ""
            replacement = cand.raw + _block(f"file:{cand.token}", body)
            records.append(ReferenceRecord(cand.raw, "file", True, f"included {cand.token}{rng}"))
            pieces.append(replacement)
        elif kind == "folder":
            replacement = cand.raw + _block(f"folder:{cand.token}", payload)
            records.append(ReferenceRecord(cand.raw, "folder", True, f"listed {cand.token}"))
            pieces.append(replacement)
        else:
            pieces.append(cand.raw)
        cursor = cand.end

    pieces.append(text[cursor:])
    return ReferenceExpansion(expanded_text="".join(pieces), records=records)


# --- impure resolver: reuses the file tools' allowed-roots + sensitive-path
# authority so a reference can never read what the tools cannot (AC#3) --------

import subprocess  # noqa: E402  (kept local to the impure section)
from pathlib import Path  # noqa: E402

#: Inline size ceiling for a referenced file (AC#4).
MAX_REFERENCE_BYTES = 256 * 1024


def parse_token(token: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    """Split ``path#L10-20`` / ``path#L5`` into (path, (lo, hi)) or (path, None)."""
    if "#l" in token.lower():
        idx = token.lower().rfind("#l")
        path_part = token[:idx]
        range_part = token[idx + 2 :]
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", range_part)
        if m and path_part:
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) else lo
            if hi >= lo:
                return path_part, (lo, hi)
    return token, None


def _looks_binary(data: bytes) -> bool:
    if b"\x00" in data[:8192]:
        return True
    sample = data[:8192]
    if not sample:
        return False
    text_bytes = bytes(range(0x20, 0x7F)) + b"\n\r\t\f\b"
    nontext = sum(1 for b in sample if b not in text_bytes)
    return nontext / len(sample) > 0.30


def _expand_path(path: "Path", line_range, max_bytes: int):
    if path.is_dir():
        try:
            entries = sorted(
                x.name + ("/" if x.is_dir() else "")
                for x in path.iterdir()
                if not x.name.startswith(".")  # hidden entries are refused by the tools
            )
        except OSError as exc:
            return ("refused", f"could not list folder: {exc}", None)
        return ("folder", "\n".join(entries), None)
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > max_bytes:
        return ("refused", f"file exceeds the {max_bytes}-byte inline size limit", None)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return ("refused", f"could not read file: {exc}", None)
    if _looks_binary(data):
        return ("refused", "file is binary and was not injected", None)
    return ("file", data.decode("utf-8", errors="replace"), line_range)


def _exists_under_roots(path_str: str, roots) -> bool:
    p = Path(path_str).expanduser()
    for root in roots:
        candidate = p if p.is_absolute() else (Path(root) / path_str)
        try:
            if candidate.exists():
                return True
        except OSError:
            continue
    try:
        return p.is_absolute() and p.exists()
    except OSError:
        return False


def resolve_reference(token, *, roots, max_bytes: int = MAX_REFERENCE_BYTES):
    """Resolve one reference token against the allowed roots (impure).

    Lane-6/8 review C1: resolution goes through the SAME choke point the file
    tools use -- ``validate_path_multi`` (which enforces allowed-roots
    containment, traversal + symlink checks, and ``allow_hidden=False``) plus
    ``is_sensitive_path`` -- so a reference can never read what
    ``ReadFileTool`` would refuse (e.g. ``.env`` / ``.git/config``).

    Returns None to leave the token literal (nonexistent -> decorator/typo),
    ("refused", reason, None) for a real-but-disallowed/hidden/sensitive/
    binary/oversized target, or ("file"|"folder", payload, line_range) for an
    allowed one.
    """
    from ..Utils.path_validation import validate_path_multi
    from ..Utils.sensitive_paths import is_sensitive_path

    if not roots:
        return ("refused", "no allowed workspace roots are configured", None) if _exists_under_roots(token, roots) else None

    path_str, line_range = parse_token(token)
    try:
        validated = Path(validate_path_multi(path_str, list(roots)))
    except ValueError:
        # Disallowed by the file-tool gate (outside roots, traversal, or a
        # hidden component). Refuse honestly if it actually exists; otherwise
        # leave it literal (a decorator/typo, AC#7).
        if _exists_under_roots(path_str, roots):
            return ("refused", "outside the allowed workspace roots, hidden, or a sensitive path", None)
        return None

    if is_sensitive_path(validated):
        return ("refused", "a protected/sensitive path", None)
    if not validated.exists():
        return None  # validated but nonexistent -> literal (AC#7)
    return _expand_path(validated, line_range, max_bytes)


def build_console_reference_resolver() -> Resolver:
    """Build a resolver bound to the file tools' allowed roots + sensitive set."""
    from ..Tools.file_operation_tools import _tool_sandbox_root
    from ..Tools.workspace_file_roots import allowed_file_roots

    try:
        roots = allowed_file_roots(write=False, sandbox_root=_tool_sandbox_root())
    except Exception:  # noqa: BLE001 - fail safe to no roots
        roots = ()

    def _resolve(token: str):
        return resolve_reference(token, roots=roots)

    return _resolve


def _git_reference_cwd() -> "Path":
    try:
        from ..Tools.workspace_file_roots import get_launch_cwd
        return Path(get_launch_cwd())
    except Exception:  # noqa: BLE001
        return Path.cwd()


def _cwd_within_roots(cwd: "Path", roots) -> bool:
    from ..Tools.file_operation_tools import is_within
    for root in roots or ():
        try:
            if is_within(cwd, Path(root)):
                return True
        except OSError:
            continue
    return False


def run_git_reference(kind: str, *, roots=None, timeout: float = 5.0) -> str:
    """Run ``git diff`` / ``git diff --staged`` for @diff/@staged (AC#2/#3).

    Lane-8 review I1: the diff is confined to a repo INSIDE the allowed roots
    (so it can't surface a repo the file tools can't reach), and the sensitive
    -path denylist is spliced in as git ``:(exclude)`` pathspecs -- exactly as
    ``git_tool_impls`` does -- so a tracked ``.env``/``~/.ssh/*`` never leaks
    its diff content.
    """
    cwd = _git_reference_cwd()
    if roots is not None and not _cwd_within_roots(cwd, roots):
        raise RuntimeError("the git repository is outside the allowed workspace roots")

    try:
        from ..Tools.git_tool_impls import _denylist_pathspecs
        excludes = list(_denylist_pathspecs(cwd))
    except Exception:  # noqa: BLE001 - if we can't build excludes, fail closed
        raise RuntimeError("could not build the sensitive-path exclusion list")

    args = ["git", "diff"]
    if kind == "staged":
        args.append("--staged")
    if excludes:
        args.append("--")
        args.extend(excludes)
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - a git failure is a refusal upstream
        raise RuntimeError(str(exc))
    out = proc.stdout or ""
    if len(out) > MAX_REFERENCE_BYTES:
        out = out[:MAX_REFERENCE_BYTES] + "\n… (diff truncated)"
    return out


def build_console_git_runner() -> GitRunner:
    """Build a git_runner bound to the allowed roots (for the send-path wiring)."""
    from ..Tools.file_operation_tools import _tool_sandbox_root
    from ..Tools.workspace_file_roots import allowed_file_roots

    try:
        roots = allowed_file_roots(write=False, sandbox_root=_tool_sandbox_root())
    except Exception:  # noqa: BLE001
        roots = ()

    def _run(kind: str) -> str:
        return run_git_reference(kind, roots=roots)

    return _run
