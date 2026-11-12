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

import asyncio
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

# A reference candidate: `@` not preceded by a word char, then a run of
# non-space characters. The lookbehind excludes emails/handles (AC#7).
_REFERENCE_RE = re.compile(r"(?<!\w)@([^\s@]+)")

_SPECIAL_TOKENS = ("diff", "staged")


@dataclass(frozen=True)
class ReferenceCandidate:
    """One ``@`` reference candidate found in composer text.

    Attributes:
        raw: The candidate as typed, including the sigil (``"@src/main.py#L10-20"``).
        token: ``raw`` without the sigil (``"src/main.py#L10-20"``).
        start: Index of the ``@`` in the source text.
        end: Index one past the candidate's last character.
    """

    raw: str
    token: str
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
    offsets: tuple[tuple[int, int], ...] = ()


def _native_reference_candidates(text: str) -> List[ReferenceCandidate]:
    """Scan native syntax for preflight/refusal without invoking callbacks."""
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


def find_reference_candidates(text: str) -> list[ReferenceCandidate]:
    """Return every ``@`` reference candidate in ``text`` (AC#7 aware)."""
    return _native_reference_candidates(text)


#: resolve(token) -> None (not a reference / leave literal)
#:               or ("file",  content:str, line_range:(int,int)|None)
#:               or ("folder", listing:str, None)
#:               or ("refused", reason:str, None)
#: One resolved reference: (kind, payload, line_range) — kind is "file"/
#: "folder"/"refused", payload is content/listing/refusal-reason, and
#: line_range only applies to files. ``None`` = leave the token literal.
ResolvedReference = Optional[Tuple[str, str, Optional[Tuple[int, int]]]]
Resolver = Callable[[str], ResolvedReference]
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
    candidates: Iterable[ReferenceCandidate] | None = None,
) -> ReferenceExpansion:
    """Expand references in ``text`` before send.

    Each candidate is resolved: a real allowed file/folder is inlined (with an
    optional line range), ``@diff``/``@staged`` run git, an explicit refusal is
    reported (never read), and anything that resolves to nothing is left as
    literal text. Returns the rewritten text plus a record per expansion.
    """
    records: List[ReferenceRecord] = []
    candidates = list(
        find_reference_candidates(text) if candidates is None else candidates
    )
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
    offsets = tuple(
        (candidate.end, len(pieces[2 * index + 1]) - len(candidate.raw))
        for index, candidate in enumerate(candidates)
    )
    return ReferenceExpansion(
        expanded_text="".join(pieces), records=records, offsets=offsets
    )


# --- impure resolver: reuses the file tools' allowed-roots + sensitive-path
# authority so a reference can never read what the tools cannot (AC#3) --------


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
    """Decide whether a referenced file is binary and must not be injected.

    TASK-38599: a strict UTF-8 decode is tried first — any file that
    decodes is text regardless of how non-ASCII its first 8 KB look, so
    CJK/Cyrillic notes are no longer refused by the printable-ASCII ratio.
    The whole payload is decoded (it is already capped at
    ``MAX_REFERENCE_BYTES``) so a multibyte character straddling the 8 KB
    sample boundary cannot fake a decode failure. Only when the decode
    fails does the original printable-ASCII heuristic apply. Null bytes
    stay an immediate refusal either way: they decode as valid UTF-8 but
    never appear in text worth injecting.
    """
    if b"\x00" in data[:8192]:
        return True
    sample = data[:8192]
    if not sample:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        text_bytes = bytes(range(0x20, 0x7F)) + b"\n\r\t\f\b"
        nontext = sum(1 for b in sample if b not in text_bytes)
        return nontext / len(sample) > 0.30
    return False


def _expand_path(path: "Path", line_range, max_bytes: int, *, roots=(), context=None):
    from ..Utils.path_validation import validate_path_multi
    from ..Utils.sensitive_paths import is_sensitive_path

    def visible(child):
        try:
            validated = Path(validate_path_multi(str(child), list(roots or (path,))))
        except ValueError:
            return False
        return not is_sensitive_path(validated, context=context)

    if path.is_dir():
        try:
            entries = sorted(
                x.name + ("/" if x.is_dir() else "")
                for x in path.iterdir()
                if not x.name.startswith(".") and visible(x)
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


def _lexically_pathlike(path_str: str) -> bool:
    """Whether a validation-refused token was clearly MEANT as a path.

    Qodo #1 (PR #2313): the previous existence probe called ``exists()`` on
    raw user-controlled paths BEFORE validation -- probing the filesystem
    outside the allowed roots. The refused-vs-literal split is now decided
    lexically, with zero filesystem contact: absolute paths, ``..``
    traversals, ``~`` expansions, and hidden dot-components are unmistakably
    path attempts (a decorator like ``@property`` or an email handle is
    none of these), so they earn an honest refusal; anything else stays a
    literal token.

    Args:
        path_str: The reference token's path portion, unvalidated.

    Returns:
        True when the token should be refused rather than left literal.
    """
    if path_str.startswith(("/", "~")):
        return True
    parts = Path(path_str).parts
    return any(part == ".." or part.startswith(".") for part in parts)


def resolve_reference(
    token: str,
    *,
    roots: Iterable[Path],
    max_bytes: int = MAX_REFERENCE_BYTES,
    context=None,
) -> ResolvedReference:
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
        return (
            ("refused", "no allowed workspace roots are configured", None)
            if _lexically_pathlike(token)
            else None
        )

    path_str, line_range = parse_token(token)
    try:
        validated = Path(validate_path_multi(path_str, list(roots)))
    except ValueError:
        # Disallowed by the file-tool gate (outside roots, traversal, or a
        # hidden component). The refused-vs-literal split is lexical -- no
        # filesystem probe of an unvalidated path (Qodo #1, PR #2313).
        if _lexically_pathlike(path_str):
            return ("refused", "outside the allowed workspace roots, hidden, or a sensitive path", None)
        return None

    if is_sensitive_path(validated, context=context):
        return ("refused", "a protected/sensitive path", None)
    if not validated.exists():
        return None  # validated but nonexistent -> literal (AC#7)
    return _expand_path(validated, line_range, max_bytes, roots=roots, context=context)


def build_console_reference_resolver(*, authority=None) -> Resolver:
    """Bind file reads to one original explicit-reference authority."""
    from .console_reference_preparation import ReferenceAuthority

    authority = authority or ReferenceAuthority()

    def _resolve(token: str):
        authority.check()
        result = resolve_reference(token, roots=authority.roots, context=authority.context)
        authority.check()
        return result

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


def run_git_reference(
    kind: str,
    *,
    roots=None,
    timeout: float = 5.0,
    cwd=None,
    context=None,
    producer=None,
    executable=None,
    require_current=None,
    environment=None,
) -> str:
    """Read the whole repository diff only inside the original readable maximum."""
    from ..Tools.git_tool_impls import _denylist_pathspecs
    from ..Workspaces.native_git import qualified_git_command, run_native_git

    cwd = Path(cwd) if cwd is not None else _git_reference_cwd()
    if roots is None or not _cwd_within_roots(cwd, roots):
        raise RuntimeError("the git repository is outside the allowed workspace roots")
    if kind not in _SPECIAL_TOKENS:
        raise RuntimeError("unsupported git reference")
    executable = executable or shutil.which("git")
    if executable is None:
        raise RuntimeError("git executable unavailable")
    environment = (
        dict(environment)
        if environment is not None
        else {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    )
    environment["GIT_TERMINAL_PROMPT"] = "0"

    def run_process(cmd, env, *, binary=False):
        if require_current is not None:
            require_current()
        if producer is None:
            return subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                timeout=timeout,
                text=not binary,
                check=False,
            )
        from ..Agents.execution_capacity import current_execution_owner

        if (
            producer.operation is None
            or current_execution_owner() is not producer.owner
        ):
            raise RuntimeError("original reference Git custody unavailable")
        producer.check()
        with asyncio.Runner() as runner:
            code, stdout, stderr = runner.run(
                run_native_git(
                    producer,
                    cmd,
                    env,
                    None,
                    cwd=str(cwd),
                    timeout_seconds=timeout,
                )
            )
        producer.check()
        if not binary:
            stdout, stderr = (
                data.decode("utf-8", errors="replace") for data in (stdout, stderr)
            )
        return subprocess.CompletedProcess(cmd, code, stdout, stderr)

    def run_at(root, args):
        cmd = qualified_git_command(
            executable,
            ["-C", str(root)],
            args,
            environment,
            run_process,
            RuntimeError,
        )
        result = run_process(cmd, environment)
        if result.returncode:
            raise RuntimeError("git reference command failed")
        return result.stdout

    top = Path(run_at(cwd, ("rev-parse", "--show-toplevel")).strip()).resolve(
        strict=True
    )
    if not _cwd_within_roots(top, roots):
        raise RuntimeError("the git repository is outside the allowed workspace roots")
    excludes = _denylist_pathspecs(top, context=context)
    cwd = top
    args = ("diff", "--staged") if kind == "staged" else ("diff",)
    out = run_at(top, (*args, "--", *excludes))
    if len(out) > MAX_REFERENCE_BYTES:
        out = out[:MAX_REFERENCE_BYTES] + "\n… (diff truncated)"
    return out


def build_console_git_runner(*, authority=None, producer=None) -> GitRunner:
    """Bind Git to the same roots, cwd and denials as this expansion's files."""
    from .console_reference_preparation import ReferenceAuthority

    authority = authority or ReferenceAuthority()
    executable = authority.executable

    def _run(kind: str) -> str:
        authority.check()
        result = run_git_reference(
            kind,
            roots=authority.roots,
            cwd=authority.cwd,
            context=authority.context,
            producer=producer,
            executable=executable,
            require_current=authority.check_git,
            environment=authority.environment,
        )
        authority.check()
        return result

    return _run


def reference_callbacks():
    """Pin the actual parser, factories and body callables used by this worker."""
    return tuple(
        (globals()[name], getattr(globals()[name], "__code__", None))
        for name in (
            "find_reference_candidates",
            "expand_references",
            "resolve_reference",
            "_expand_path",
            "build_console_reference_resolver",
            "build_console_git_runner",
            "run_git_reference",
        )
    )


_STOCK_REFERENCE_CALLBACKS = reference_callbacks()
