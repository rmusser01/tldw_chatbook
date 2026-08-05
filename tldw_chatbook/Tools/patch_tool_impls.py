"""fs_patch core: unified-diff parser/applier + workspace wrapper.

The parser/applier below (``parse_unified_diff``, ``apply_patch_to_text``
and their helpers/dataclasses, ``FilesystemPatchError``) is a near-verbatim
port of tldw_server's
tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py
@ 5605b9d9906322c2e6b5342b48c391ae674d315e
(https://github.com/rmusser01/tldw_server, GPL-3.0-only). Reason codes are
kept exactly so error handling stays in lockstep with the reference.

The workspace wrapper (``patch_files``) is written fresh for tldw_chatbook:
it enforces ADR-032 confinement via resolve_workspace_path and phase-2
write discipline (encode-before-write, newline-preserving reads), and
translates FilesystemPatchError into the shared LocalToolError.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tldw_chatbook.Tools.local_tool_impls import LocalToolError, resolve_workspace_path

PATCH_MAX_BYTES = 256 * 1024
PATCH_MAX_FILES = 20
PATCH_MAX_HUNKS = 200

PatchLineKind = Literal["context", "add", "remove"]
PatchFileAction = Literal["modify", "create"]

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")
_NO_NEWLINE_MARKER = r"\ No newline at end of file"
_HEADER_TIMESTAMP_METADATA = re.compile(
    r"\s+\d{4}-\d{2}-\d{2}"
    r"(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?"
    r"(?:\s*(?:[+-]\d{4}|[+-]\d{2}:?\d{2}|Z))?$"
)


class FilesystemPatchError(ValueError):
    """Raised when a unified diff cannot be parsed or applied safely."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class PatchHunkLine:
    """One context, addition, or removal line inside a unified-diff hunk."""

    kind: PatchLineKind
    text: str
    has_trailing_newline: bool = True


@dataclass(frozen=True, slots=True)
class PatchHunk:
    """Parsed unified-diff hunk with old and new line ranges."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[PatchHunkLine, ...]


@dataclass(frozen=True, slots=True)
class PatchFile:
    """One file-level patch from a unified diff."""

    old_path: str | None
    new_path: str | None
    action: PatchFileAction
    hunks: tuple[PatchHunk, ...]


def parse_unified_diff(
    diff_text: str,
    *,
    max_files: int,
    max_hunks: int,
    max_bytes: int,
) -> tuple[PatchFile, ...]:
    """Parse bounded unified diff text into file-level patch plans."""

    if not isinstance(diff_text, str) or not diff_text.strip():
        raise FilesystemPatchError("invalid_diff")
    if len(diff_text.encode("utf-8")) > max(1, int(max_bytes)):
        raise FilesystemPatchError("diff_too_large")

    lines = diff_text.splitlines()
    files: list[PatchFile] = []
    hunk_count = 0
    index = 0

    while index < len(lines):
        if not lines[index].startswith("--- "):
            index += 1
            continue

        old_path = _parse_header_path(lines[index][4:])
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise FilesystemPatchError("invalid_diff")
        new_path = _parse_header_path(lines[index][4:])
        index += 1

        if old_path is None and new_path is None:
            raise FilesystemPatchError("invalid_patch_path")
        if new_path is None:
            raise FilesystemPatchError("delete_not_supported")
        if old_path is None:
            action: PatchFileAction = "create"
        else:
            if old_path != new_path:
                raise FilesystemPatchError("rename_not_supported")
            action = "modify"

        hunks: list[PatchHunk] = []
        while index < len(lines) and not lines[index].startswith("--- "):
            if not lines[index].startswith("@@ "):
                raise FilesystemPatchError("invalid_diff")
            hunk, index = _parse_hunk(lines, index)
            hunks.append(hunk)
            hunk_count += 1
            if hunk_count > max(1, int(max_hunks)):
                raise FilesystemPatchError("diff_hunk_limit_exceeded")

        if not hunks:
            raise FilesystemPatchError("invalid_diff")
        files.append(PatchFile(old_path=old_path, new_path=new_path, action=action, hunks=tuple(hunks)))
        if len(files) > max(1, int(max_files)):
            raise FilesystemPatchError("diff_file_limit_exceeded")

    if not files:
        raise FilesystemPatchError("invalid_diff")
    return tuple(files)


def apply_patch_to_text(original: str, patch_file: PatchFile) -> str:
    """Apply one parsed file patch to original text without touching the filesystem."""

    original_lines = original.splitlines(keepends=True)
    newline = _detect_output_newline(original_lines)
    output: list[str] = []
    cursor = 0

    for hunk in patch_file.hunks:
        hunk_start = max(0, hunk.old_start - 1)
        if hunk_start < cursor or hunk_start > len(original_lines):
            raise FilesystemPatchError("patch_context_mismatch")
        output.extend(original_lines[cursor:hunk_start])
        cursor = hunk_start

        for hunk_line in hunk.lines:
            if hunk_line.kind == "add":
                output.append(hunk_line.text)
                if hunk_line.has_trailing_newline:
                    output.append(newline)
                continue

            if cursor >= len(original_lines):
                raise FilesystemPatchError("patch_context_mismatch")
            if _line_body(original_lines[cursor]) != hunk_line.text:
                raise FilesystemPatchError("patch_context_mismatch")
            if _line_has_trailing_newline(original_lines[cursor]) != hunk_line.has_trailing_newline:
                raise FilesystemPatchError("patch_context_mismatch")
            if hunk_line.kind == "context":
                output.append(original_lines[cursor])
            cursor += 1

    output.extend(original_lines[cursor:])
    return "".join(output)


def _parse_hunk(lines: list[str], start_index: int) -> tuple[PatchHunk, int]:
    """Parse one unified-diff hunk and return the next unread line index."""

    match = _HUNK_HEADER.match(lines[start_index])
    if match is None:
        raise FilesystemPatchError("invalid_hunk_header")

    old_start = int(match.group(1))
    old_count = int(match.group(2) or "1")
    new_start = int(match.group(3))
    new_count = int(match.group(4) or "1")
    hunk_lines: list[PatchHunkLine] = []
    old_seen = 0
    new_seen = 0
    index = start_index + 1

    while index < len(lines) and not lines[index].startswith("@@ ") and not lines[index].startswith("--- "):
        raw_line = lines[index]
        index += 1
        if raw_line == _NO_NEWLINE_MARKER:
            if not hunk_lines:
                raise FilesystemPatchError("invalid_no_newline_marker")
            previous = hunk_lines[-1]
            if not previous.has_trailing_newline:
                raise FilesystemPatchError("invalid_no_newline_marker")
            hunk_lines[-1] = PatchHunkLine(
                kind=previous.kind,
                text=previous.text,
                has_trailing_newline=False,
            )
            continue
        if not raw_line:
            raise FilesystemPatchError("invalid_hunk_line")

        prefix = raw_line[0]
        text = raw_line[1:]
        if prefix == " ":
            hunk_lines.append(PatchHunkLine(kind="context", text=text))
            old_seen += 1
            new_seen += 1
        elif prefix == "-":
            hunk_lines.append(PatchHunkLine(kind="remove", text=text))
            old_seen += 1
        elif prefix == "+":
            hunk_lines.append(PatchHunkLine(kind="add", text=text))
            new_seen += 1
        else:
            raise FilesystemPatchError("invalid_hunk_line")

    if old_seen != old_count or new_seen != new_count:
        raise FilesystemPatchError("invalid_hunk_line_count")
    return (
        PatchHunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            lines=tuple(hunk_lines),
        ),
        index,
    )


def _parse_header_path(raw_path: str) -> str | None:
    """Normalize a unified-diff file header path while stripping safe metadata.

    Tab-separated metadata is removed first because GNU/Git-style diffs commonly
    place timestamps after a tab. When no tab exists, only a trailing
    timestamp-shaped suffix is stripped so paths containing spaces remain intact.
    Returns None for `/dev/null` create/delete sentinels.
    """

    candidate = raw_path.rstrip()
    if "\t" in candidate:
        candidate = candidate.split("\t", 1)[0]
    else:
        candidate = _strip_space_separated_header_metadata(candidate)
    candidate = candidate.strip()
    if candidate == "/dev/null":
        return None
    if candidate.startswith("a/") or candidate.startswith("b/"):
        candidate = candidate[2:]
    return _normalize_patch_path(candidate)


def _strip_space_separated_header_metadata(candidate: str) -> str:
    """Strip common space-separated timestamp metadata from a diff header path."""

    stripped = _HEADER_TIMESTAMP_METADATA.sub("", candidate).rstrip()
    return stripped or candidate


def _normalize_patch_path(raw_path: str) -> str:
    candidate = raw_path.strip().replace("\\", "/")
    if not candidate or candidate in {".", "/"}:
        raise FilesystemPatchError("invalid_patch_path")
    if candidate.startswith("/") or candidate.startswith("//"):
        raise FilesystemPatchError("invalid_patch_path")
    if len(candidate) >= 2 and candidate[1] == ":" and candidate[0].isalpha():
        raise FilesystemPatchError("invalid_patch_path")

    parts = candidate.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise FilesystemPatchError("invalid_patch_path")
    return "/".join(parts)


def _detect_output_newline(lines: list[str]) -> str:
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
        if line.endswith("\r"):
            return "\r"
    return "\n"


def _line_body(line: str) -> str:
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1]
    return line


def _line_has_trailing_newline(line: str) -> bool:
    """Return whether a split line retained an LF, CRLF, or CR terminator."""

    return line.endswith(("\n", "\r"))


def patch_files(diff_text: str, *, workspace_root: Path, dry_run: bool = False) -> str:
    """Parse and apply a unified diff to workspace files.

    Every target is confined via resolve_workspace_path. Modify targets must
    exist; create targets must not. dry_run validates and reports without
    writing. Returns a per-file summary ("patched X", "would patch X").
    Files are applied sequentially; if a later file fails, earlier files stay
    patched — the error names the failed file so the model can recover
    (atomic multi-file apply is a documented non-goal for this phase).
    """

    try:
        parsed = parse_unified_diff(
            diff_text,
            max_files=PATCH_MAX_FILES,
            max_hunks=PATCH_MAX_HUNKS,
            max_bytes=PATCH_MAX_BYTES,
        )
    except FilesystemPatchError as exc:
        raise LocalToolError(f"fs_patch failed [{exc.reason_code}]") from exc

    summaries: list[str] = []
    for patch_file in parsed:
        rel_path = patch_file.new_path
        assert rel_path is not None  # guaranteed by parse_unified_diff
        try:
            root = resolve_workspace_path(rel_path, workspace_root)
            if patch_file.action == "modify":
                if not root.is_file():
                    raise LocalToolError(f"file not found: {rel_path}")
                try:
                    with open(root, encoding="utf-8", newline="") as fh:
                        original = fh.read()
                except UnicodeDecodeError as exc:
                    raise LocalToolError(
                        f"'{rel_path}' is not valid UTF-8; fs_patch only patches text files"
                    ) from exc
            else:  # create
                if root.exists():
                    raise LocalToolError(f"file already exists: {rel_path}")
                if not root.parent.is_dir():
                    raise LocalToolError(
                        f"parent directory does not exist for: {rel_path}"
                    )
                original = ""

            updated = apply_patch_to_text(original, patch_file)
            # Encode BEFORE writing — a failed encode must never truncate an
            # existing file. dry_run still validates encodability.
            try:
                data = updated.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise LocalToolError(
                    f"patched content for '{rel_path}' is not UTF-8 encodable "
                    f"(lone surrogate?): {exc}"
                ) from exc
            if not dry_run:
                root.write_bytes(data)
        except FilesystemPatchError as exc:
            raise LocalToolError(
                f"fs_patch failed [{exc.reason_code}]: {rel_path}"
            ) from exc
        summaries.append(
            f"{'would patch' if dry_run else 'patched'} {rel_path}"
        )
    return "\n".join(summaries)
