"""Immutable review evidence under admitted workspace and scratch ownership.

This host owner never grants a tool permission or loads project instructions.
The provider uses original_source_path before content dispatch through the
existing instruction ledger. Discovery returns names/refs/ranges, never bodies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator

from pydantic import TypeAdapter, ValidationError

from tldw_chatbook.Agents.code_review_models import (
    CoverageGap,
    ReviewRefused,
    ReviewTarget,
    ReviewWireModel,
    SourceQuery,
    canonical_record_bytes,
    review_result_char_limit,
)
from tldw_chatbook.Agents.code_review_store import CapturedSource, ReviewStore
from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
from tldw_chatbook.Tools import git_tool_impls as _git
from tldw_chatbook.Tools.local_tool_impls import (
    MAX_GREP_RESULTS,
    MAX_LIST_ENTRIES,
    MAX_READ_FILE_BYTES,
    LocalToolError,
    _relative_target_is_safe,
)
from tldw_chatbook.Tools.patch_tool_impls import (
    FilesystemPatchError,
    apply_patch_to_text,
    parse_patch_targets,
)
from tldw_chatbook.Tools.remote_root_types import RemoteRoot, local_root_path
from tldw_chatbook.Utils.sensitive_paths import (
    SensitiveExclusion,
    sensitive_exclusions_under,
)

_QUERY_SECONDS = 5.0
_QUERY_READ_BYTES = 2 * 1024 * 1024


class ReviewPage(ReviewWireModel):
    """Complete model-bound envelope; adapters must not truncate its JSON."""

    review_id: str
    target_digest: str
    source_epoch: int
    inventory_revision: int
    revision: int = 0
    submission_revision: int | None = None
    text: str = ""
    records: tuple[dict, ...] = ()
    source_ref: str | None = None
    source_digest: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    cursor: str | None = None
    gaps: tuple[CoverageGap, ...] = ()


@dataclass(frozen=True)
class ReviewSourceAuthority:
    """Host-only admitted root; patch references resolve only accepted inputs."""

    admitted: RunAdmittedWorkspaceRoot
    source_epoch: int
    resolve_patch: Callable[[str], bytes] | None = field(default=None, repr=False)

    def __post_init__(self):
        if not isinstance(self.admitted, RunAdmittedWorkspaceRoot):
            raise ReviewRefused("source_authority_unavailable")
        if type(self.source_epoch) is not int or self.source_epoch < 0:
            raise ReviewRefused("source_epoch_mismatch")

    @contextmanager
    def scope(self):
        if isinstance(self.admitted.root, RemoteRoot):
            raise ReviewRefused("target_acquisition_unavailable")
        try:
            if not self.admitted.guard(False):
                raise ReviewRefused("source_authority_unavailable")
            root = local_root_path(self.admitted.root, site="review source").resolve()
            with (
                self.admitted.authority_scope()
                if self.admitted.authority_scope
                else nullcontext()
            ):
                exclusions = list(sensitive_exclusions_under(root))
                if self.admitted.exclusions_provider:
                    for path in map(Path, self.admitted.exclusions_provider()):
                        # A resolved exclusion outside the root (an escaping
                        # symlink) matches no in-root path; _safe refuses the
                        # escaping link itself.
                        if not path.is_relative_to(root):
                            continue
                        relative = path.relative_to(root).as_posix()
                        # Exact excluded files remain excluded if deleted/replaced.
                        exclusions.append(SensitiveExclusion("subtree", relative))
                yield root, tuple(exclusions)
                if not self.admitted.guard(False):
                    raise ReviewRefused("source_authority_unavailable")
        except ReviewRefused:
            raise
        except (OSError, RuntimeError) as exc:
            raise ReviewRefused("source_authority_unavailable") from exc


@dataclass(frozen=True)
class SourceDocument:
    source_ref: str
    relative_path: str
    side: str
    digest: str
    byte_count: int
    line_count: int
    encoding: str = "utf-8"
    newline: str = "none"


@dataclass(frozen=True)
class SourceManifest:
    review_id: str
    target_digest: str
    binding_ref: str
    source_epoch: int
    kind: str
    scope: str
    comparison: str
    revisions: tuple[tuple[str, str], ...]
    initial_sources: tuple[SourceDocument, ...]
    gaps: tuple[CoverageGap, ...]
    absent_sources: tuple[tuple[str, str], ...]


@dataclass
class _SourceState:
    manifest: SourceManifest
    authority: ReviewSourceAuthority
    documents: dict[str, SourceDocument]
    by_path: dict[tuple[str, str], str]
    inventory_revision: int = 1
    cursors: dict[str, Path] = field(default_factory=dict)
    pages: dict[str, Path] = field(default_factory=dict)


_BOUND: ContextVar[tuple[ReviewStore, ReviewSourceAuthority] | None] = ContextVar(
    "review_source", default=None
)


@contextmanager
def use_review_source_authority(
    store: ReviewStore, authority: ReviewSourceAuthority
) -> Iterator[None]:
    """Host adapter binds the owning store, in addition to the actual run ID."""
    token = _BOUND.set((store, authority))
    try:
        yield
    finally:
        _BOUND.reset(token)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe(root, exclusions, path, *, directory=False):
    if (
        not isinstance(path, str)
        or not path
        or "\x00" in path
        or "\\" in path
        or path.startswith("/")
        or ".." in PurePosixPath(path).parts
    ):
        raise ReviewRefused("invalid_source_path")
    if not _relative_target_is_safe(
        Path(path), root, exclusions, is_directory=directory
    ):
        raise ReviewRefused("source_path_unavailable")
    # Refuse every symlink component, including internal aliases, before opening.
    candidate = root
    for part in PurePosixPath(path).parts:
        candidate /= part
        if candidate.is_symlink():
            raise ReviewRefused("unsupported_source_type")
    return root / path


def _listed_names(directory) -> frozenset[str]:
    """Names stored in ``directory``, spelled exactly as the filesystem lists them.

    A directory that can be entered but not listed cannot prove a spelling:
    that fails closed as unavailable source, not as a variant spelling.
    """
    try:
        return frozenset(os.listdir(directory))
    except OSError as exc:
        raise ReviewRefused("source_unavailable") from exc


def _spelled_as_stored(root, path, listings) -> bool:
    """Whether an existing worktree selection names each component as stored.

    Case- or normalization-insensitive filesystems (APFS, NTFS) open
    ``MAIN.PY`` or an NFD ``naïve.py`` as another stored file, which may be
    tracked or excluded; only the stored spelling is listed. A missing path is
    left to the caller's missing-file handling. ``listings`` memoizes one
    pass's directory listings; the drift re-check passes a fresh one.
    """
    if not os.path.lexists(root / path):
        return True
    directory = root
    for part in PurePosixPath(path).parts:
        if directory not in listings:
            listings[directory] = _listed_names(directory)
        if part not in listings[directory]:
            return False
        directory = directory / part
    return True


def _in_scope(path, scope):
    return scope == "." or path == scope or path.startswith(scope.rstrip("/") + "/")


def _text(data):
    if b"\0" in data:
        raise ReviewRefused("binary_source")
    try:
        return data.decode("utf-8")
    except UnicodeError as exc:
        raise ReviewRefused("unsupported_source_encoding") from exc


def _read_mutable(root, exclusions, path):
    target = _safe(root, exclusions, path)
    try:
        # Walk through directory descriptors so a symlink swap cannot redirect
        # the open after lexical/resolved admission.
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            parts = PurePosixPath(path).parts
            for part in parts[:-1]:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = next_fd
            file_fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        finally:
            os.close(directory_fd)
        with os.fdopen(file_fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ReviewRefused("unsupported_source_type")
            if before.st_size > MAX_READ_FILE_BYTES:
                raise ReviewRefused("source_truncated")
            data = stream.read(MAX_READ_FILE_BYTES + 1)
            after = os.fstat(stream.fileno())
        current = target.stat()

        def identity(item):
            return (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        if identity(before) != identity(after) or identity(after) != identity(current):
            raise ReviewRefused("source_drift")
        if len(data) > MAX_READ_FILE_BYTES:
            raise ReviewRefused("source_truncated")
        return data
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReviewRefused("source_unavailable") from exc


def _document(path, side, data):
    text = _text(data)
    return SourceDocument(
        secrets.token_hex(16),
        path,
        side,
        _hash(data),
        len(data),
        len(text.splitlines()),
        newline=("crlf" if b"\r\n" in data else "lf" if b"\n" in data else "none"),
    )


def _git_error(exc):
    allowed = {
        "history_unavailable",
        "source_truncated",
        "ambiguous_comparison",
        "unmerged_source",
        "unsupported_source_type",
        "source_drift",
        "source_path_unavailable",
    }
    return ReviewRefused(
        str(exc) if str(exc) in allowed else "source_acquisition_failed"
    )


def capture_review_source(
    target: ReviewTarget,
    authority: ReviewSourceAuthority,
    store: ReviewStore,
    review_id: str,
) -> SourceManifest:
    """Reserve, freeze and recheck required inputs before sealing their identity."""
    try:
        target = TypeAdapter(ReviewTarget).validate_python(target)
    except ValidationError as exc:
        raise ReviewRefused("invalid_review_target") from exc
    normalized = {
        "scope": PurePosixPath(target.scope).as_posix(),
        "context_paths": tuple(
            PurePosixPath(path).as_posix() for path in target.context_paths
        ),
    }
    if target.kind == "working_tree":
        normalized["untracked_paths"] = tuple(
            PurePosixPath(path).as_posix() for path in target.untracked_paths
        )
    target = target.model_copy(update=normalized)
    if target.binding_ref != authority.admitted.binding_id:
        raise ReviewRefused("source_binding_mismatch")
    with store.lease(review_id) as context, authority.scope() as (root, exclusions):
        # The guard may wait on the UI thread, which takes the store lock on
        # Stop/close, so capture holds that lock only for these checks;
        # reservations and finalize_capture re-check publication under it.
        with store._tab.lock:
            if context.source_finalized or context.active_attempts:
                raise ReviewRefused("source_already_published")
            if context.source_epoch != authority.source_epoch:
                raise ReviewRefused("source_epoch_mismatch")
        _safe(root, exclusions, target.scope, directory=True)
        try:
            return _capture(target, authority, store, context, root, exclusions)
        except LocalToolError as exc:
            raise _git_error(exc) from exc
        except (UnicodeError, OSError, ValueError) as exc:
            if isinstance(exc, ReviewRefused):
                raise
            raise ReviewRefused("source_acquisition_failed") from exc


# A changed file the view cannot hold as text is a named coverage gap, not a
# refusal of the whole target (spec: excluded/binary material is coverage).
_UNREADABLE_CHANGE = frozenset(
    {
        "binary_source",
        "unsupported_source_type",
        "unsupported_source_encoding",
        "source_truncated",
    }
)
#: Capture gaps shown to the primary (prepare) and on a reviewer's first scope
#: listing are bounded; withheld sensitive changes never carry names.
GAP_PREVIEW_LIMIT = 8
_GAP_REASON_CHARS = 160


def capture_gap_preview(gaps) -> tuple[tuple[str, ...], int]:
    """First capture gap reasons, each clipped, and how many were left out."""
    reasons = tuple(
        reason
        if len(reason) <= _GAP_REASON_CHARS
        else reason[: _GAP_REASON_CHARS - 1] + "…"
        for reason in (gap.reason for gap in gaps[:GAP_PREVIEW_LIMIT])
    )
    return reasons, max(0, len(gaps) - GAP_PREVIEW_LIMIT)


def _within_file_quota(store, files):
    """Refuse a change set before reading entries when its files cannot fit."""
    if files + 2 > store._tab.limits.review_max_files:
        raise ReviewRefused("storage_quota_exceeded")


def _capture(target, authority, store, context, root, exclusions):
    reader = None
    revisions = {}
    selected = {}  # (side, relative path) -> immutable Git entry or mutable marker
    mutable = set()
    gaps = []
    patch_files = ()
    patch_bytes = b""
    comparison = "captured"
    head = None
    layers = None
    changed_paths = set()
    withheld = 0
    listings = {}  # this pass's directory listings
    spelled = []  # worktree selections whose stored spelling was required
    if target.kind in ("pr", "commit_range", "working_tree") or getattr(
        target, "preimage_commit", None
    ):
        reader = ReviewGitReader(root, sensitive_exclusions=exclusions)
    if target.kind in ("pr", "commit_range"):
        base, head = (
            reader.commit(target.base_commit),
            reader.commit(target.head_commit),
        )
        comparison = target.comparison
        if comparison == "merge_base":
            base = reader.merge_base(base, head)
        revisions = {"base": base, "head": head}
        changed = reader.changed(base, head, target.scope)
        _within_file_quota(store, len(changed) * len(revisions))
        withheld = reader.change_count(base, head, target.scope) - len(changed)
        changed_paths.update(changed)
        for path in changed:
            for side, revision in revisions.items():
                selected[side, path] = reader.entry(revision, path)
    elif target.kind == "working_tree":
        head = reader.head()
        revisions = {"base": head}
        staged = (
            reader.changed(head, None, target.scope, staged=True)
            if target.staged
            else ()
        )
        unstaged = reader.changed(None, None, target.scope) if target.unstaged else ()
        layers = (staged, unstaged)
        changed_paths.update((*staged, *unstaged))
        _within_file_quota(
            store,
            len(changed_paths) * (2 + int(target.unstaged))
            + len(target.untracked_paths),
        )
        if target.staged:
            withheld += reader.change_count(head, None, target.scope, staged=True)
            withheld -= len(staged)
        if target.unstaged:
            withheld += reader.change_count(None, None, target.scope) - len(unstaged)
        for path in target.untracked_paths:
            _safe(root, exclusions, path)
            if not _in_scope(path, target.scope) or not _spelled_as_stored(
                root, path, listings
            ):
                raise ReviewRefused("invalid_untracked_selection")
            spelled.append(path)
        paths = tuple(dict.fromkeys((*staged, *unstaged)))
        indexed = reader.index_entries((*paths, *target.untracked_paths))
        for path in paths:
            selected["base", path] = reader.entry(head, path)
            selected["staged", path] = indexed[path]
            if target.unstaged:
                selected["worktree", path] = "mutable"
                mutable.add(path)
        for path in target.untracked_paths:
            if indexed[path] is not None:
                raise ReviewRefused("invalid_untracked_selection")
            selected["worktree", path] = "mutable"
            mutable.add(path)
    else:
        if authority.resolve_patch is None:
            raise ReviewRefused("target_acquisition_unavailable")
        patch_bytes = authority.resolve_patch(target.patch_ref)
        if not isinstance(patch_bytes, bytes):
            raise ReviewRefused("target_acquisition_unavailable")
        if len(patch_bytes) > MAX_READ_FILE_BYTES:
            raise ReviewRefused("source_truncated")
        try:
            patch_files = parse_patch_targets(_text(patch_bytes))
        except FilesystemPatchError as exc:
            raise ReviewRefused("invalid_patch") from exc
        touched_paths = set()
        for patch in patch_files:
            paths = {patch.old_path, patch.new_path} - {None}
            if touched_paths & paths:
                raise ReviewRefused("duplicate_patch_target")
            touched_paths.update(paths)
        if target.preimage_commit:
            revisions = {"base": reader.commit(target.preimage_commit)}
        for patch in patch_files:
            for path in (patch.old_path, patch.new_path):
                if path:
                    _safe(root, exclusions, path)
                    if not _in_scope(path, target.scope):
                        raise ReviewRefused("source_outside_scope")
            preimage_path = patch.old_path or patch.new_path
            selected["base", preimage_path] = (
                reader.entry(revisions["base"], preimage_path)
                if revisions
                else "mutable"
            )
            if not revisions:
                # A variant spelling would read another stored file as preimage.
                if not _spelled_as_stored(root, preimage_path, listings):
                    raise ReviewRefused("patch_preimage_mismatch")
                spelled.append(preimage_path)
                mutable.add(preimage_path)
    if withheld > 0:
        gaps.append(
            CoverageGap(reason=f"excluded_source: {withheld} changed file(s) withheld")
        )

    def unreadable(path, reason):
        gap = CoverageGap(reason=f"{reason}: {path}")
        if gap not in gaps:
            gaps.append(gap)
        mutable.discard(path)

    required = set(selected)
    for path in target.context_paths:
        if any(path in (patch.old_path, patch.new_path) for patch in patch_files):
            continue
        try:
            _safe(root, exclusions, path)
            if not _in_scope(path, target.scope):
                raise ReviewRefused("source_outside_scope")
            if target.kind in ("pr", "commit_range"):
                for side, revision in revisions.items():
                    selected.setdefault((side, path), reader.entry(revision, path))
            else:
                key = ("worktree" if target.kind == "working_tree" else "head", path)
                if key not in selected:
                    if not _spelled_as_stored(root, path, listings):
                        raise ReviewRefused("source_path_unavailable")
                    spelled.append(path)
                selected.setdefault(key, "mutable")
                mutable.add(path)
        except (ReviewRefused, LocalToolError):
            gaps.append(CoverageGap(reason="optional_context_unavailable"))
    sizes = {}
    for (side, path), entry in tuple(selected.items()):
        try:
            _safe(root, exclusions, path)
        except ReviewRefused as exc:
            if path not in changed_paths or exc.reason not in _UNREADABLE_CHANGE:
                raise
            unreadable(path, exc.reason)
            selected.pop((side, path))
            continue
        if entry == "mutable":
            try:
                size = (root / path).stat().st_size
            except FileNotFoundError:
                size = 0
        else:
            size = entry.size if entry else 0
        if size > MAX_READ_FILE_BYTES:
            if path in changed_paths:
                unreadable(path, "source_truncated")
                selected.pop((side, path))
                continue
            if (side, path) in required:
                raise ReviewRefused("source_truncated")
            gaps.append(CoverageGap(reason="optional_context_unavailable"))
            selected.pop((side, path))
            mutable.discard(path)
            continue
        sizes[side, path] = size
    # Patch output can retain the preimage and add every byte of patch text.
    estimate = sum(sizes.values()) + len(patch_bytes)
    if patch_files:
        estimate += sum(sizes.values()) + len(patch_bytes)
    metadata_allowance = 4096 + (len(selected) + len(patch_files) * 2) * 1024
    with store.reserve(
        context.review_id,
        source_bytes=estimate,
        report_bytes=metadata_allowance,
        files=len(selected) + len(patch_files) + 2,
    ) as reservation:
        documents = {}
        captures = {}
        data_by_path = {}
        missing = []

        def add(side, path, data):
            document = _document(path, side, data)
            capture_path = reservation.write(data)
            documents[document.source_ref] = document
            captures[document.source_ref] = CapturedSource(
                path=capture_path,
                digest=document.digest,
                line_count=document.line_count,
            )
            data_by_path[side, path] = data

        for (side, path), entry in selected.items():
            try:
                data = (
                    _read_mutable(root, exclusions, path)
                    if entry == "mutable"
                    else reader.blob(entry, max_bytes=MAX_READ_FILE_BYTES)
                    if entry
                    else None
                )
                if data is None:
                    if (side, path) not in required:
                        gaps.append(CoverageGap(reason="optional_context_unavailable"))
                    elif (
                        target.kind == "working_tree" and path in target.untracked_paths
                    ):
                        raise ReviewRefused("source_unavailable")
                    missing.append((side, path))
                    continue
                add(side, path, data)
            except (ReviewRefused, LocalToolError) as exc:
                reason = exc.reason if isinstance(exc, ReviewRefused) else str(exc)
                if path in changed_paths and reason in _UNREADABLE_CHANGE:
                    unreadable(path, reason)
                    continue
                if (side, path) in required:
                    raise
                gaps.append(CoverageGap(reason="optional_context_unavailable"))
                mutable.discard(path)
        for patch in patch_files:
            if not patch.old_path and ("base", patch.new_path) in data_by_path:
                raise ReviewRefused("patch_preimage_mismatch")
            preimage = data_by_path.get(("base", patch.old_path), b"")
            if patch.old_path and ("base", patch.old_path) not in data_by_path:
                raise ReviewRefused("patch_preimage_mismatch")
            try:
                output = apply_patch_to_text(_text(preimage), patch).encode("utf-8")
            except FilesystemPatchError as exc:
                raise ReviewRefused("patch_preimage_mismatch") from exc
            if patch.new_path:
                add("head", patch.new_path, output)
        if patch_bytes:
            reservation.write(patch_bytes)
        # Re-read all mutable inputs, not merely their mtimes. Index and selected
        # change lists are independently pinned; a drift publishes no manifest.
        for path in mutable:
            previous = next(
                (
                    data
                    for (side, name), data in data_by_path.items()
                    if name == path and selected.get((side, name)) == "mutable"
                ),
                None,
            )
            if _read_mutable(root, exclusions, path) != previous:
                raise ReviewRefused("source_drift")
        # A fresh listing: a respelled selection (same bytes) is still drift.
        # Only for what is still read; a gap'd path is ignored, as its bytes are.
        fresh = {}
        if any(
            path in mutable and not _spelled_as_stored(root, path, fresh)
            for path in spelled
        ):
            raise ReviewRefused("source_drift")
        if target.kind == "working_tree":
            now = (
                reader.changed(head, None, target.scope, staged=True)
                if target.staged
                else (),
                reader.changed(None, None, target.scope) if target.unstaged else (),
            )
            if now != layers or reader.head() != head:
                raise ReviewRefused("source_drift")
            # A fresh index read: drift between the passes is still seen.
            staged_now = reader.index_entries(
                (
                    *(path for side, path in selected if side == "staged"),
                    *target.untracked_paths,
                )
            )
            for (side, path), entry in selected.items():
                if side == "staged" and staged_now[path] != entry:
                    raise ReviewRefused("source_drift")
            for path in target.untracked_paths:
                if staged_now[path] is not None:
                    raise ReviewRefused("source_drift")
        if not authority.admitted.guard(False):
            raise ReviewRefused("source_authority_unavailable")
        identity = {
            "target": target.model_dump(mode="json"),
            "binding": {
                "workspace_id": authority.admitted.workspace_id,
                "locator_fingerprint": authority.admitted.locator_fingerprint,
                "root_identity": authority.admitted.root_identity,
            },
            "revisions": revisions,
            "sources": [
                (doc.side, doc.relative_path, doc.digest) for doc in documents.values()
            ],
            "missing": missing,
            "patch_digest": _hash(patch_bytes) if patch_bytes else None,
        }
        digest = _hash(canonical_record_bytes(identity))
        manifest = SourceManifest(
            context.review_id,
            digest,
            target.binding_ref,
            context.source_epoch,
            target.kind,
            target.scope,
            comparison,
            tuple(revisions.items()),
            tuple(documents.values()),
            tuple(gaps),
            tuple(missing),
        )
        state = _SourceState(
            manifest,
            authority,
            documents,
            {
                (doc.side, doc.relative_path): doc.source_ref
                for doc in documents.values()
            },
        )
        metadata = canonical_record_bytes(
            identity
            | {
                "target_digest": digest,
                "inventory": [doc.__dict__ for doc in documents.values()],
                "gaps": [gap.model_dump(mode="json") for gap in gaps],
            }
        )
        reservation.write(metadata, category="report")
        actual_source = sum(doc.byte_count for doc in documents.values()) + len(
            patch_bytes
        )
        store.finalize_capture(
            context.review_id,
            target_digest=digest,
            source_state=state,
            sources=captures,
            reservation=reservation,
            actual_source_bytes=actual_source,
            actual_report_bytes=len(metadata),
            actual_files=len(documents) + int(bool(patch_bytes)) + 1,
        )
        return manifest


@contextmanager
def _access(review_id):
    bound = _BOUND.get()
    if bound is None:
        raise ReviewRefused("source_authority_unavailable")
    store, authority = bound
    with store.lease(review_id) as context:
        with store._tab.lock:
            binding = store._bound(context)
            state = context.source_state
            revision = context.revision
            if not isinstance(state, _SourceState) or authority is not state.authority:
                raise ReviewRefused("source_authority_mismatch")
            if authority.source_epoch != context.source_epoch:
                raise ReviewRefused("source_epoch_mismatch")
        # Guard entry/exit can synchronously dispatch to the UI. The lease pins
        # bytes, not authority: revalidate the exact snapshot after that callback.
        with authority.scope() as (root, exclusions):
            with store._tab.lock:
                store._live(context)
                if (
                    store._bound(context) != binding
                    or context.source_state is not state
                    or context.revision != revision
                    or context.source_epoch != authority.source_epoch
                ):
                    raise ReviewRefused("source_authority_changed")
                yield store, context, state, binding, root, exclusions


def _actor(binding):
    return binding.model_dump(mode="json")


def _load_cursor(state, cursor, binding, selection):
    path = state.cursors.get(cursor)
    if path is None:
        raise ReviewRefused("invalid_source_cursor")
    payload = json.loads(path.read_bytes())
    if payload["actor"] != _actor(binding) or payload["selection"] != selection:
        raise ReviewRefused("source_cursor_conflict")
    return payload


def _fits(page, max_result_chars, max_tokens, estimate=None):
    encoded = page.model_dump_json()
    # Conservative byte estimate works without a provider tokenizer; Task 4
    # still performs its model-specific complete-request admission check.
    return len(encoded) <= review_result_char_limit(max_result_chars) and (
        estimate(encoded) <= max_tokens
        if estimate is not None
        else len(encoded.encode("utf-8")) <= max_tokens * 4
    )


def _base_page(state, context, binding, **kwargs):
    manifest = state.manifest
    return ReviewPage(
        review_id=manifest.review_id,
        target_digest=manifest.target_digest,
        source_epoch=manifest.source_epoch,
        inventory_revision=state.inventory_revision,
        revision=context.revision,
        submission_revision=(
            context.streams[(binding.slot_id, binding.attempt_id)].revision
            if binding.role != "primary"
            else None
        ),
        **kwargs,
    )


def _persist_page(
    store, context, state, binding, selection, page, next_position, incoming
):
    payload = {
        "actor": _actor(binding),
        "selection": selection,
        "inventory_revision": page.inventory_revision,
        "position": next_position,
    }
    page_bytes = page.model_dump_json().encode("utf-8")
    cursor_bytes = canonical_record_bytes(payload) if page.cursor else b""
    with store.reserve(
        context.review_id,
        source_bytes=0,
        report_bytes=len(page_bytes) + len(cursor_bytes),
        files=1 + bool(cursor_bytes),
    ) as reservation:
        page_path = reservation.write(page_bytes, category="report")
        cursor_path = (
            reservation.write(cursor_bytes, category="report") if cursor_bytes else None
        )
        reservation.commit(
            0, len(page_bytes) + len(cursor_bytes), 1 + bool(cursor_bytes)
        )
        if page.cursor:
            state.cursors[page.cursor] = cursor_path
        if incoming:
            state.pages[incoming] = page_path
    return page


def _replay(state, cursor, max_result_chars, max_tokens, estimate=None):
    path = state.pages.get(cursor)
    if path is None:
        return None
    page = ReviewPage.model_validate_json(path.read_bytes())
    if not _fits(page, max_result_chars, max_tokens, estimate):
        raise ReviewRefused("source_page_capacity_changed")
    return page


def original_source_path(review_id: str, source_ref: str) -> Path:
    """Pure authority mapping for provider path_targets; no body read/advance."""
    with _access(review_id) as (_, context, state, _, root, exclusions):
        document = state.documents.get(source_ref)
        if document is None or source_ref not in context.sources:
            raise ReviewRefused("invalid_source_reference")
        return _safe(root, exclusions, document.relative_path)


def _captured_bytes(context, document):
    captured = context.sources.get(document.source_ref)
    if captured is None:
        raise ReviewRefused("invalid_source_reference")
    try:
        with captured.path.open("rb") as stream:
            data = stream.read(document.byte_count + 1)
    except OSError as exc:
        raise ReviewRefused("source_unavailable") from exc
    if len(data) != document.byte_count or _hash(data) != document.digest:
        raise ReviewRefused("source_drift")
    return data


def read_review_source(
    manifest: SourceManifest,
    source_ref: str,
    *,
    start_line: int,
    max_tokens: int,
    end_line: int | None = None,
    cursor: str | None = None,
    max_result_chars: int = 16000,
    estimate: Callable[[str], int] | None = None,
) -> ReviewPage:
    """Deliver whole lines from a captured range, with replayable bound cursors."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ReviewRefused("source_capacity_unavailable")
    max_tokens = min(max_tokens, 1024)
    with _access(manifest.review_id) as (
        store,
        context,
        state,
        binding,
        root,
        exclusions,
    ):
        if manifest is not state.manifest:
            raise ReviewRefused("invalid_source_manifest")
        document = state.documents.get(source_ref)
        if document is None:
            raise ReviewRefused("invalid_source_reference")
        _safe(root, exclusions, document.relative_path)
        if (
            document.line_count == 0
            and type(start_line) is int
            and start_line == 1
            and end_line is None
            and cursor is None
        ):
            _captured_bytes(context, document)
            page = _base_page(
                state,
                context,
                binding,
                source_ref=source_ref,
                source_digest=document.digest,
            )
            if not _fits(page, max_result_chars, max_tokens, estimate):
                raise ReviewRefused("review_control_too_large_for_tool_result")
            return page
        end = document.line_count if end_line is None else end_line
        if (
            type(start_line) is not int
            or type(end) is not int
            or start_line < 1
            or end < start_line
            or end > document.line_count
        ):
            refusal = ReviewRefused("invalid_source_range")
            refusal.source_range_details = {
                "source_ref": source_ref,
                "line_count": document.line_count,
            }
            raise refusal
        selection = {
            "op": "content",
            "source_ref": source_ref,
            "start_line": start_line,
            "end_line": end,
        }
        offset = start_line
        if cursor:
            saved = _load_cursor(state, cursor, binding, selection)
            replay = _replay(state, cursor, max_result_chars, max_tokens, estimate)
            if replay:
                return replay
            offset = saved["position"]
        lines = _text(_captured_bytes(context, document)).splitlines(keepends=True)
        token = secrets.token_hex(16)
        page = _base_page(
            state,
            context,
            binding,
            source_ref=source_ref,
            source_digest=document.digest,
            start_line=offset,
            end_line=offset,
            cursor=token,
        )
        if not _fits(page, max_result_chars, max_tokens, estimate):
            raise ReviewRefused("review_control_too_large_for_tool_result")
        delivered = []
        for number in range(offset, end + 1):
            candidate = page.model_copy(
                update={
                    "text": "".join((*delivered, lines[number - 1])),
                    "end_line": number,
                    "cursor": token if number < end else None,
                }
            )
            if not _fits(candidate, max_result_chars, max_tokens, estimate):
                break
            delivered.append(lines[number - 1])
            page = candidate
        if not delivered:
            refusal = ReviewRefused("source_span_too_large_for_tool_result")
            refusal.gap = CoverageGap(reason=refusal.reason, source_ref=source_ref)
            raise refusal
        return _persist_page(
            store, context, state, binding, selection, page, page.end_line + 1, cursor
        )


def _materialize(store, context, state, reader, path, side, root, exclusions):
    _safe(root, exclusions, path)
    if (side, path) in state.manifest.absent_sources:
        raise ReviewRefused("source_absent")
    existing = state.by_path.get((side, path))
    if existing:
        return state.documents[existing]
    if state.manifest.kind not in ("pr", "commit_range"):
        raise ReviewRefused("context_not_captured")
    revision = dict(state.manifest.revisions).get(side)
    if revision is None:
        raise ReviewRefused("invalid_source_side")
    entry = reader.entry(revision, path)
    if entry is None:
        raise ReviewRefused("source_unavailable")
    if entry.size > MAX_READ_FILE_BYTES:
        raise ReviewRefused("source_truncated")
    with store.reserve(
        context.review_id,
        source_bytes=entry.size,
        report_bytes=2048 + len(path.encode("utf-8")),
        files=2,
    ) as reservation:
        data = reader.blob(entry, max_bytes=MAX_READ_FILE_BYTES)
        document = _document(path, side, data)
        source_path = reservation.write(data)
        metadata = canonical_record_bytes(document.__dict__)
        reservation.write(metadata, category="report")
        reservation.commit(len(data), len(metadata), 2)
        context.sources[document.source_ref] = CapturedSource(
            path=source_path, digest=document.digest, line_count=document.line_count
        )
        state.documents[document.source_ref] = document
        state.by_path[side, path] = document.source_ref
        state.inventory_revision += 1
        return document


def _candidates(state, reader, query, position):
    """Yield bounded pinned-tree/frozen-inventory names with resumable positions."""
    if state.manifest.kind not in ("pr", "commit_range"):
        names = sorted(
            doc.relative_path
            for doc in state.manifest.initial_sources
            if doc.side == query.side
            and _in_scope(doc.relative_path, query.relative_dir)
        )
        start = position.get("index", 0)
        for index in range(start, min(len(names), start + MAX_LIST_ENTRIES)):
            yield names[index], {"index": index + 1}, index + 1 < len(names)
        return
    revision = dict(state.manifest.revisions)[query.side]
    pending = [
        list(item) for item in position.get("pending", [[query.relative_dir, 0]])
    ]
    count = 0
    deadline = time.monotonic() + _QUERY_SECONDS
    while pending and count < MAX_LIST_ENTRIES and time.monotonic() < deadline:
        directory, offset = pending.pop()
        entries = reader.tree(revision, directory)
        if offset >= len(entries):
            yield None, {"pending": pending}, bool(pending)
        for index in range(offset, len(entries)):
            entry = entries[index]
            remaining = (
                [*pending, [directory, index + 1]]
                if index + 1 < len(entries)
                else list(pending)
            )
            count += 1
            if entry.mode == "040000":
                pending = [*remaining, [entry.path, 0]]
                yield None, {"pending": pending}, True
                break
            yield entry.path, {"pending": remaining}, bool(remaining)
            if count >= MAX_LIST_ENTRIES or time.monotonic() >= deadline:
                return
        else:
            continue


def _capture_gap_notes(state, context, binding, max_result_chars, max_tokens, estimate):
    """Capture gaps for a first scope listing, within half the page budget."""
    total = len(state.manifest.gaps)
    if not total:
        return []
    shown, omitted = capture_gap_preview(state.manifest.gaps)
    notes = [CoverageGap(reason=reason) for reason in shown]
    if omitted:
        notes.append(CoverageGap(reason=f"capture_gaps_omitted: {omitted} more"))
    for candidate in (notes, [CoverageGap(reason=f"capture_gaps: {total} recorded")]):
        page = _base_page(state, context, binding, gaps=tuple(candidate))
        if _fits(page, max_result_chars // 2, max_tokens // 2, estimate):
            return candidate
    return []


def query_review_source(
    review_id: str,
    query: SourceQuery,
    *,
    cursor: str | None = None,
    max_result_chars: int = 16000,
    max_tokens: int = 1024,
    estimate: Callable[[str], int] | None = None,
) -> ReviewPage:
    """List, literal-search or resolve only an admitted captured comparison side."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ReviewRefused("source_capacity_unavailable")
    max_tokens = min(1024, max_tokens)
    try:
        query = TypeAdapter(SourceQuery).validate_python(query)
    except ValidationError as exc:
        raise ReviewRefused("invalid_source_query") from exc
    field_name = "relative_path" if query.op == "resolve" else "relative_dir"
    query = query.model_copy(
        update={field_name: PurePosixPath(getattr(query, field_name)).as_posix()}
    )
    with _access(review_id) as (store, context, state, binding, root, exclusions):
        selection = query.model_dump(mode="json")
        position = {}
        if cursor:
            saved = _load_cursor(state, cursor, binding, selection)
            replay = _replay(state, cursor, max_result_chars, max_tokens, estimate)
            if replay:
                requested = (
                    query.relative_path if query.op == "resolve" else query.relative_dir
                )
                _safe(root, exclusions, requested, directory=query.op != "resolve")
                for record in replay.records:
                    _safe(root, exclusions, record["relative_path"])
                for gap in replay.gaps:
                    if gap.source_ref:
                        _safe(
                            root,
                            exclusions,
                            state.documents[gap.source_ref].relative_path,
                        )
                return replay
            position = saved["position"]
        path = query.relative_path if query.op == "resolve" else query.relative_dir
        _safe(root, exclusions, path, directory=query.op != "resolve")
        if not _in_scope(path, state.manifest.scope):
            raise ReviewRefused("source_outside_scope")
        committed = state.manifest.kind in ("pr", "commit_range")
        sides = (
            set(dict(state.manifest.revisions))
            if committed
            else {"base", "head"}
            if state.manifest.kind == "patch"
            else {"base", "staged", "worktree"}
        )
        if query.side not in sides:
            raise ReviewRefused("invalid_source_side")
        try:
            reader = (
                ReviewGitReader(root, sensitive_exclusions=exclusions)
                if committed
                else None
            )
            return _query(
                store,
                context,
                state,
                binding,
                root,
                exclusions,
                reader,
                query,
                selection,
                position,
                cursor,
                max_result_chars,
                max_tokens,
                estimate,
            )
        except LocalToolError as exc:
            raise _git_error(exc) from exc
        except (UnicodeError, OSError) as exc:
            raise ReviewRefused("source_acquisition_failed") from exc


def _query(
    store,
    context,
    state,
    binding,
    root,
    exclusions,
    reader,
    query,
    selection,
    position,
    cursor,
    max_result_chars,
    max_tokens,
    estimate,
):
    token = secrets.token_hex(16)
    page = _base_page(state, context, binding, cursor=token)
    if not _fits(page, max_result_chars, max_tokens, estimate):
        raise ReviewRefused("review_control_too_large_for_tool_result")
    candidates = (
        [(query.relative_path, {}, False)]
        if query.op == "resolve"
        else _candidates(state, reader, query, position)
    )
    records = []
    final_position = position
    more = False
    read_bytes = 0
    gaps = (
        _capture_gap_notes(
            state, context, binding, max_result_chars, max_tokens, estimate
        )
        if query.op == "list"
        and cursor is None
        and query.relative_dir == state.manifest.scope
        else []
    )

    def gap_page(gap, next_position, has_more):
        nonlocal page, gaps, final_position, more
        updated_gaps = list(dict.fromkeys((*gaps, gap)))
        candidate = _base_page(
            state,
            context,
            binding,
            records=tuple(records),
            gaps=tuple(updated_gaps),
            cursor=token if has_more else None,
        )
        if not _fits(candidate, max_result_chars, max_tokens, estimate):
            if not records:
                raise ReviewRefused("review_control_too_large_for_tool_result")
            return _persist_page(
                store,
                context,
                state,
                binding,
                selection,
                page.model_copy(update={"cursor": token}),
                final_position,
                cursor,
            )
        page, gaps = candidate, updated_gaps
        final_position, more = next_position, has_more
        return None

    for path, next_position, has_more in candidates:
        if path is None:
            final_position, more = next_position, has_more
            continue
        try:
            _safe(root, exclusions, path)
            document = _materialize(
                store, context, state, reader, path, query.side, root, exclusions
            )
        except (ReviewRefused, LocalToolError) as exc:
            if query.op == "resolve":
                raise
            reason = exc.reason if isinstance(exc, ReviewRefused) else str(exc)
            if reason not in {
                "binary_source",
                "unsupported_source_type",
                "source_truncated",
                "unsupported_source_encoding",
                "source_unavailable",
                "source_path_unavailable",
            }:
                raise
            completed = gap_page(
                CoverageGap(
                    reason=reason
                    if reason
                    in {
                        "binary_source",
                        "unsupported_source_type",
                        "source_truncated",
                        "unsupported_source_encoding",
                    }
                    else "source_unavailable"
                ),
                next_position,
                has_more,
            )
            if completed is not None:
                return completed
            continue
        entry = {
            "source_ref": document.source_ref,
            "relative_path": path,
            "side": query.side,
        }
        hit_offset = (
            position.get("hit_offset", 0) if position.get("hit_path") == path else 0
        )
        entries = [entry]
        if query.op == "search":
            if document.byte_count > _QUERY_READ_BYTES:
                completed = gap_page(
                    CoverageGap(reason="search_read_limit"), next_position, has_more
                )
                if completed is not None:
                    return completed
                continue
            read_bytes += document.byte_count
            if read_bytes > _QUERY_READ_BYTES:
                more = True
                break
            hits = (
                entry | {"start_line": number, "end_line": number}
                for number, line in enumerate(
                    _text(_captured_bytes(context, document)).splitlines(), 1
                )
                if query.text in line
            )
            # One lookahead preserves the next hit without allocating every match.
            entries = list(islice(hits, hit_offset, hit_offset + MAX_GREP_RESULTS + 1))
        # File-local search hits are independently resumable; don't lose a tail
        # when a whole-file hit list cannot fit the complete envelope.
        for index in range(len(entries)):
            after = (
                next_position
                if index + 1 == len(entries)
                else final_position
                | {"hit_path": path, "hit_offset": hit_offset + index + 1}
            )
            has_tail = has_more or index + 1 < len(entries)
            candidate = _base_page(
                state,
                context,
                binding,
                records=tuple((*records, entries[index])),
                cursor=token if has_tail else None,
                gaps=tuple(dict.fromkeys(gaps)),
            )
            if len(records) >= MAX_GREP_RESULTS or not _fits(
                candidate, max_result_chars, max_tokens, estimate
            ):
                more = True
                if not records:
                    refusal = ReviewRefused("review_record_too_large_for_tool_result")
                    refusal.gap = CoverageGap(reason=refusal.reason)
                    raise refusal
                return _persist_page(
                    store,
                    context,
                    state,
                    binding,
                    selection,
                    page.model_copy(update={"cursor": token}),
                    final_position,
                    cursor,
                )
            records.append(entries[index])
            page = candidate
            final_position, more = after, has_tail
        if not entries:
            final_position, more = next_position, has_more
    if more and not records:
        gaps.append(CoverageGap(reason="source_query_incomplete"))
    page = _base_page(
        state,
        context,
        binding,
        records=tuple(records),
        cursor=token if more else None,
        gaps=tuple(dict.fromkeys(gaps)),
    )
    if not _fits(page, max_result_chars, max_tokens, estimate):
        raise ReviewRefused("review_control_too_large_for_tool_result")
    if more and final_position == position and not records:
        raise ReviewRefused("source_query_capacity_unavailable")
    return _persist_page(
        store, context, state, binding, selection, page, final_position, cursor
    )


def source_cursor_selection(review_id: str, cursor: str) -> dict:
    """Map a valid actor cursor to its original selection, without advancing it.

    The provider can omit repeated wire fields on continuation and prepare the
    original instruction target before dispatch. This never reads source bodies.
    """
    with _access(review_id) as (_, context, state, binding, root, exclusions):
        path = state.cursors.get(cursor)
        if path is None:
            raise ReviewRefused("invalid_source_cursor")
        payload = json.loads(path.read_bytes())
        if payload["actor"] != _actor(binding):
            raise ReviewRefused("source_cursor_conflict")
        selection = payload["selection"]
        if selection["op"] == "content":
            document = state.documents.get(selection["source_ref"])
            if document is None or document.source_ref not in context.sources:
                raise ReviewRefused("invalid_source_reference")
            _safe(root, exclusions, document.relative_path)
        else:
            _safe(
                root,
                exclusions,
                selection.get("relative_path", selection.get("relative_dir")),
                directory=selection["op"] != "resolve",
            )
        return dict(selection)


def _argv_chunks(paths, limit: int = 64 * 1024):
    """Split pathspec arguments so one git process stays within argv limits."""
    chunk: list[str] = []
    size = 0
    for path in paths:
        if chunk and size + len(path) > limit:
            yield chunk
            chunk, size = [], 0
        chunk.append(path)
        size += len(path) + 24
    if chunk:
        yield chunk


@dataclass(frozen=True, slots=True)
class ReviewGitEntry:
    """Host-selected tree/index entry; callers cannot select a raw blob object."""

    path: str
    oid: str
    mode: str
    size: int


class ReviewGitReader:
    """Narrow internal immutable-object reader; no checkout/index mutations."""

    def __init__(self, workspace_root: Path, *, sensitive_exclusions=None):
        self.exclusions = (
            sensitive_exclusions
            if sensitive_exclusions is not None
            else sensitive_exclusions_under(workspace_root)
        )
        self.root = _git.prepare_repository(
            workspace_root, sensitive_exclusions=self.exclusions
        )
        if self.root != workspace_root.resolve():
            raise LocalToolError("review repository must match admitted root")
        self.object_format = self._read(["rev-parse", "--show-object-format"]).strip()
        if self.object_format not in (b"sha1", b"sha256"):
            raise LocalToolError("unsupported_object_format")
        self.oid_length = 40 if self.object_format == b"sha1" else 64
        if self._read(["rev-parse", "--is-shallow-repository"]).strip() != b"false":
            raise LocalToolError("history_unavailable")
        # Older Git ignores NO_LAZY_FETCH; refuse promisor configuration too.
        partial = _git._run_git_bytes(
            [
                "git",
                "config",
                "--get-regexp",
                r"^(extensions\.partialclone|remote\..*\.promisor)$",
            ],
            cwd=self.root,
            immutable_objects=True,
        )
        if partial.truncated or partial.returncode not in (0, 1) or partial.stdout:
            raise LocalToolError("history_unavailable")
        # Commit and tree objects are immutable: verify each commit once and
        # list each pinned directory once per reader (one capture or query).
        self._commits: set[str] = set()
        self._listings: dict[tuple[str, str], dict[str, ReviewGitEntry]] = {}
        self._object_sizes: dict[str, int] = {}  # an object's size never changes

    def _read(
        self, args: list[str], *, limit: int = _git.GIT_MAX_OUTPUT_BYTES
    ) -> bytes:
        result = _git._run_git_bytes(
            ["git", *args],
            cwd=self.root,
            max_output_bytes=limit,
            immutable_objects=True,
        )
        if result.truncated:
            raise LocalToolError("source_truncated")
        if result.returncode:
            raise LocalToolError("history_unavailable")
        return result.stdout

    def _oid(self, value: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(
            "[0-9a-f]{" + str(self.oid_length) + "}", value
        ):
            raise LocalToolError("invalid_object_id")
        return value

    def safe_path(self, value: str, *, directory: bool = False) -> str:
        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
            or "\\" in value
            or value.startswith("/")
            or ".." in PurePosixPath(value).parts
        ):
            raise LocalToolError("invalid_source_path")
        if not _relative_target_is_safe(
            Path(value), self.root, self.exclusions, is_directory=directory
        ):
            raise LocalToolError("source_path_unavailable")
        return value

    def commit(self, value: str) -> str:
        oid = self._oid(value)
        if oid in self._commits:
            return oid
        resolved = (
            self._read(["rev-parse", "--verify", oid + "^{commit}"])
            .strip()
            .decode("ascii")
        )
        if resolved != oid:
            raise LocalToolError("invalid_commit_id")
        self._commits.add(oid)
        return oid

    def head(self) -> str:
        return self.commit(
            self._read(["rev-parse", "--verify", "HEAD"]).strip().decode("ascii")
        )

    def merge_base(self, base: str, head: str) -> str:
        values = self._read(
            ["merge-base", "--all", self.commit(base), self.commit(head)]
        ).splitlines()
        if len(values) != 1:
            raise LocalToolError("ambiguous_comparison")
        return self.commit(values[0].decode("ascii"))

    def _changed_paths(
        self, base, head, scope, staged, *, withhold_sensitive
    ) -> list[bytes]:
        args = [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--name-only",
            "-z",
        ]
        if staged:
            args.append("--cached")
        if base is not None:
            args.append(self.commit(base))
        if head is not None:
            args.append(self.commit(head))
        args.extend(
            ["--", _git._literal_pathspec(self.safe_path(scope, directory=True))]
        )
        if withhold_sensitive:
            args.extend(
                _git._denylist_pathspecs(
                    self.root,
                    sensitive_exclusions=self.exclusions,
                    workspace_root=self.root,
                )
            )
        return [part for part in self._read(args).split(b"\0") if part]

    def changed(
        self, base: str | None, head: str | None, scope: str = ".", *, staged=False
    ) -> tuple[str, ...]:
        """Admitted changed paths; withheld ones are only counted, never named."""
        paths = []
        for part in self._changed_paths(
            base, head, scope, staged, withhold_sensitive=True
        ):
            try:
                paths.append(self.safe_path(part.decode("utf-8")))
            except LocalToolError:
                continue  # unsafe in the checkout; change_count still counts it
        return tuple(paths)

    def change_count(
        self, base: str | None, head: str | None, scope: str = ".", *, staged=False
    ) -> int:
        """Count every changed path, withheld ones included, naming none."""
        return len(
            self._changed_paths(base, head, scope, staged, withhold_sensitive=False)
        )

    def _listing(self, commit: str, directory: str) -> dict[str, ReviewGitEntry]:
        """Every blob/tree/gitlink entry of one pinned directory.

        Callers check paths. A submodule (gitlink, mode 160000) is kept so a
        changed one is reported as unsupported instead of absent; blob()
        refuses it and tree() omits it from listings.
        """
        key = (self.commit(commit), directory)
        listing = self._listings.get(key)
        if listing is not None:
            return listing
        treeish = commit if directory == "." else commit + ":" + directory
        listing = {}
        for record in self._read(["ls-tree", "-z", "-l", treeish]).split(b"\0"):
            if not record:
                continue
            meta, name = record.split(b"\t", 1)
            mode, kind, oid, size = meta.split()
            if kind not in (b"blob", b"tree", b"commit"):
                continue
            path = (
                name.decode("utf-8")
                if directory == "."
                else directory + "/" + name.decode("utf-8")
            )
            listing[path] = ReviewGitEntry(
                path,
                self._oid(oid.decode("ascii")),
                mode.decode("ascii"),
                int(size) if size != b"-" else 0,
            )
            if kind == b"blob":
                self._object_sizes[listing[path].oid] = listing[path].size
        self._listings[key] = listing
        return listing

    def tree(self, commit: str, directory: str = ".") -> tuple[ReviewGitEntry, ...]:
        self.commit(commit)
        self.safe_path(directory, directory=True)
        entries = []
        for entry in self._listing(commit, directory).values():
            if entry.mode == "160000":
                continue
            try:
                self.safe_path(entry.path, directory=entry.mode == "040000")
            except LocalToolError:
                continue
            entries.append(entry)
        return tuple(entries)

    def entry(self, commit: str, path: str) -> ReviewGitEntry | None:
        # Admit the requested path and each ancestor directory walked, never
        # every sibling of every listed directory (tree() filters listings).
        self.safe_path(path)
        directory = "."
        for part in PurePosixPath(path).parts:
            candidate = part if directory == "." else directory + "/" + part
            entry = self._listing(commit, directory).get(candidate)
            if entry is None:
                return None
            if candidate == path:
                return entry
            if entry.mode != "040000":
                return None
            try:
                self.safe_path(candidate, directory=True)
            except LocalToolError:
                return None
            directory = candidate
        return None

    def index(self, path: str) -> ReviewGitEntry | None:
        return self.index_entries((path,))[path]

    def index_entries(self, paths) -> dict[str, ReviewGitEntry | None]:
        """Stage-0 index entries for ``paths`` from one fresh index read.

        Every call re-reads the index (drift checks stay real); only object
        sizes, which an object ID fixes, are memoized. A path that is a
        directory or has several stages refuses as before.
        """
        paths = tuple(dict.fromkeys(paths))
        for path in paths:
            self.safe_path(path)
        found: dict[str, list[tuple[bytes, str]]] = {path: [] for path in paths}
        for chunk in _argv_chunks(paths):
            raw = self._read(
                ["ls-files", "--stage", "-z", "--", *map(_git._literal_pathspec, chunk)]
            )
            for record in raw.split(b"\0"):
                if not record:
                    continue
                meta, name = record.split(b"\t", 1)
                name = name.decode("utf-8")
                parts = PurePosixPath(name).parts
                for depth in range(1, len(parts) + 1):
                    owner = "/".join(parts[:depth])
                    if owner in found:
                        found[owner].append((meta, name))
        entries = {}
        for path, records in found.items():
            if not records:
                entries[path] = None
                continue
            if len(records) != 1:
                raise LocalToolError("unmerged_source")
            meta, name = records[0]
            mode, oid, stage = meta.split()
            if stage != b"0" or name != path:
                raise LocalToolError("unmerged_source")
            selected = self._oid(oid.decode("ascii"))
            if selected not in self._object_sizes:
                self._object_sizes[selected] = int(
                    self._read(["cat-file", "-s", selected]).strip()
                )
            entries[path] = ReviewGitEntry(
                path, selected, mode.decode("ascii"), self._object_sizes[selected]
            )
        return entries

    def blob(
        self, entry: ReviewGitEntry, *, max_bytes: int = _git.GIT_MAX_OUTPUT_BYTES
    ) -> bytes:
        self.safe_path(entry.path)
        if entry.mode not in ("100644", "100755"):
            raise LocalToolError("unsupported_source_type")
        if entry.size > max_bytes:
            raise LocalToolError("source_truncated")
        data = self._read(["cat-file", "blob", self._oid(entry.oid)], limit=max_bytes)
        if len(data) != entry.size:
            raise LocalToolError("source_drift")
        return data
