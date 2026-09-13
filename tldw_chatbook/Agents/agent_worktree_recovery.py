"""Exact-authority confirmed recovery of durable, physically drained agent work."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..DB.agent_worktrees import AgentWorktreeRepository
from .agent_models import TERMINAL_RUN_STATUSES
from .agent_worktree import WorktreeRefusal, _worktree_root_identity
from .agent_worktree_git import MAX_OUTPUT, OperationError, run_git

if TYPE_CHECKING:
    from ..DB.AgentRuns_DB import AgentRunsDB
    from .local_tool_provider import RunAdmittedWorkspaceRoot

MAX_PREVIEW = 8192
MAX_ENTRIES = 10000


@dataclass(frozen=True)
class WorktreeRecoveryOutcome:
    """Durably confirmed receipt; the source checkout is always retained."""

    action: str
    message: str
    state: str
    commit_sha: str | None = None


def _refuse(code, message):
    raise OperationError(code, message)


def _authority(record, authority):
    if authority is None or not authority.allow_write or not authority.guard(True):
        _refuse(
            "source_authority_revoked",
            "A current writable selected repository is required.",
        )
    for key, value in (
        ("workspace_id", authority.workspace_id),
        ("binding_id", authority.binding_id),
        ("locator_fingerprint", authority.locator_fingerprint),
        ("repo_root", str(authority.root)),
        ("repo_identity", authority.root_identity),
    ):
        if record[key] != value:
            _refuse(
                "ownership_mismatch", "The selected repository does not own this work."
            )
    for path, key in (
        (authority.root, "repo_identity"),
        (Path(record["child_path"]), "child_identity"),
        (Path(record["git_common_dir"]), "git_common_identity"),
    ):
        if _worktree_root_identity(path) != record[key]:
            _refuse(
                "identity_changed",
                "Repository or checkout identity changed; work is retained.",
            )


def _metadata(record, authority):
    _authority(record, authority)
    if not _read_primitives():
        _refuse("unsupported_primitives", "No-follow snapshot reads are unavailable.")
    common = Path(record["git_common_dir"])
    child = Path(record["child_path"])
    for root in (authority.root, child):
        raw = run_git(root, "rev-parse", "--git-common-dir").decode().strip()
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if path.resolve() != common:
            _refuse(
                "metadata_changed", "Git common directory no longer matches its record."
            )
        if run_git(root, "rev-parse", "--show-toplevel").decode().strip() != str(root):
            _refuse(
                "unsupported_layout",
                "Recovery requires exact repository checkout roots.",
            )
    gitdir = Path(run_git(child, "rev-parse", "--absolute-git-dir").decode().strip())
    if (
        gitdir.parent != common / "worktrees"
        or gitdir.resolve() != gitdir
        or not stat.S_ISREG(os.lstat(child / ".git").st_mode)
        or any(
            not stat.S_ISREG(os.lstat(gitdir / name).st_mode)
            for name in ("gitdir", "commondir")
        )
    ):
        _refuse("unsupported_layout", "Recovery requires a standard linked worktree.")
    with os.fdopen(
        os.open(gitdir / "gitdir", os.O_RDONLY | os.O_NOFOLLOW), "rb"
    ) as link:
        link_value = link.read(4097)
    if len(link_value) > 4096 or os.fsdecode(link_value).strip() != str(child / ".git"):
        _refuse("unsupported_layout", "Linked checkout metadata does not match.")
    if (
        run_git(child, "symbolic-ref", "HEAD").decode().strip()
        != "refs/heads/" + record["branch"]
    ):
        _refuse("branch_changed", "The child branch no longer matches its record.")
    run_git(child, "merge-base", "--is-ancestor", record["base_sha"], "HEAD")
    _authority(record, authority)


def _read_primitives():
    return (
        hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and all(fn in os.supports_dir_fd for fn in (os.open, os.stat, os.readlink))
        and os.listdir in os.supports_fd
    )


def _primitives():
    return (
        _read_primitives()
        and os.name == "posix"
        and all(fn in os.supports_dir_fd for fn in (os.unlink, os.rmdir))
    )


def _tree_digest(root: Path, *, cleanup=False, selected=None, inventory_only=False):
    """Hash every non-administrative entry through no-follow directory handles."""
    if not (_primitives() if cleanup else _read_primitives()):
        _refuse(
            "unsupported_primitives",
            "No-follow descriptor-relative filesystem operations are unavailable.",
        )
    digest = hashlib.sha256()
    size = 0
    entries = 0

    def visit(fd, prefix):
        nonlocal size, entries
        for name in sorted(os.listdir(fd)):
            relative = prefix + name
            if name == ".git":
                if prefix:
                    _refuse(
                        "nested_repository",
                        "Nested repositories cannot be discarded or captured.",
                    )
                continue
            if selected is not None and not any(
                p == relative
                or p.startswith(relative + "/")
                or (p.endswith("/") and relative.startswith(p))
                for p in selected
            ):
                continue
            entries += 1
            if entries > MAX_ENTRIES:
                _refuse(
                    "snapshot_limit",
                    "Checkout contains too many entries for a bounded preview.",
                )
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            digest.update(
                os.fsencode(prefix + name) + b"\0" + str(info.st_mode).encode() + b"\0"
            )
            if stat.S_ISDIR(info.st_mode):
                nested = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                try:
                    visit(nested, prefix + name + "/")
                finally:
                    os.close(nested)
                if cleanup:
                    os.rmdir(name, dir_fd=fd)
            elif stat.S_ISLNK(info.st_mode):
                digest.update(os.fsencode(os.readlink(name, dir_fd=fd)))
                if cleanup:
                    os.unlink(name, dir_fd=fd)
            elif stat.S_ISREG(info.st_mode):
                if not cleanup and not inventory_only:
                    file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
                    with os.fdopen(file_fd, "rb") as stream:
                        while chunk := stream.read(65536):
                            size += len(chunk)
                            if size > MAX_OUTPUT:
                                _refuse(
                                    "snapshot_limit",
                                    "Checkout contents exceed the 32 MiB preview limit.",
                                )
                            digest.update(chunk)
                if cleanup:
                    os.unlink(name, dir_fd=fd)
            else:
                _refuse(
                    "unsupported_entry",
                    "Special filesystem entries cannot be recovered automatically.",
                )
            digest.update(b"\0")

    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        visit(fd, "")
    finally:
        os.close(fd)
    return digest.digest()


def _snapshot(record, authority):
    _metadata(record, authority)
    child = Path(record["child_path"])
    if record.get("_action") == "discard":
        _tree_digest(child, inventory_only=True)
    digest = hashlib.sha256()
    for root in (child, authority.root):
        for args in (
            ("rev-parse", "HEAD"),
            ("ls-files", "--stage", "-z"),
            ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        ):
            digest.update(run_git(root, *args))
        changed = run_git(
            root, "ls-files", "--modified", "--others", "--exclude-standard", "-z"
        )
        selected = {os.fsdecode(p) for p in changed.split(b"\0") if p}
        if root == child and record.get("_action") == "discard":
            ignored = run_git(
                root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"
            )
            selected.update(os.fsdecode(p) for p in ignored.split(b"\0") if p)
        digest.update(_tree_digest(root, selected=selected))
    for args in (
        (
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            record["base_sha"],
        ),
    ):
        digest.update(run_git(child, *args))
    tracked = run_git(child, "ls-files", "--stage", "-z")
    if any(entry.startswith(b"160000 ") for entry in tracked.split(b"\0")):
        _refuse(
            "unsupported_layout",
            "Submodule checkouts cannot be recovered automatically.",
        )
    preview = run_git(
        child,
        "diff",
        "--stat",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        record["base_sha"],
    ).decode("utf-8", "replace")
    untracked = run_git(child, "ls-files", "--others", "--exclude-standard", "-z")
    if untracked:
        preview += "\nNew files: " + ", ".join(
            repr(os.fsdecode(p)) for p in untracked.split(b"\0") if p
        )
    if record.get("_action") == "discard":
        ignored = run_git(
            child, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"
        )
        if ignored:
            preview += "\nIgnored entries to discard: " + ", ".join(
                repr(os.fsdecode(p)) for p in ignored.split(b"\0") if p
            )
    _authority(record, authority)
    return digest.digest(), preview[:MAX_PREVIEW]


def _operation_in_progress(root):
    for name in (
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "rebase-merge",
        "rebase-apply",
        "sequencer",
    ):
        path = Path(run_git(root, "rev-parse", "--git-path", name).decode().strip())
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return True
    return False


def recover_agent_worktree(
    db: AgentRunsDB,
    *,
    authority: RunAdmittedWorkspaceRoot,
    conversation_id: str,
    run_id: str,
    action: str,
    request_confirmation: Callable[[dict], dict],
    should_cancel: Callable[[], bool],
) -> WorktreeRecoveryOutcome | WorktreeRefusal:
    """Confirm and perform one recovery on the caller's owning worker thread."""
    repository = AgentWorktreeRepository(db)
    operation_id = None
    destination_effect = False
    source_effect = False
    try:
        if action not in ("apply", "merge", "discard"):
            _refuse(
                "invalid_action", "Recovery action must be apply, merge or discard."
            )
        if not callable(request_confirmation):
            _refuse(
                "confirmation_unavailable",
                "No worktree confirmation surface is available.",
            )
        record = repository.get_for_conversation(run_id, conversation_id)
        if (
            record is None
            or record["writer_state"] != "drained"
            or record["mutation_state"] != "unresolved"
            or record["operation_id"] is not None
            or record["run_status"] not in TERMINAL_RUN_STATUSES
        ):
            _refuse(
                "worktree_unavailable",
                "Work must be owned, terminal, unresolved and physically drained.",
            )
        if should_cancel():
            _refuse("cancelled", "Recovery cancelled before mutation.")
        if action == "discard" and not _primitives():
            _refuse(
                "unsupported_primitives",
                "Logical discard requires POSIX no-follow primitives.",
            )
        record["_action"] = action
        fingerprint, preview = _snapshot(record, authority)
        root, child = authority.root, Path(record["child_path"])
        if action == "discard" and not _primitives():
            _refuse(
                "unsupported_primitives",
                "Logical discard requires POSIX no-follow primitives.",
            )
        if action == "merge" and (
            run_git(root, "status", "--porcelain=v1", "-z")
            or _operation_in_progress(root)
        ):
            _refuse(
                "destination_busy",
                "Merge requires a clean destination with no operation in progress.",
            )
        payload = {
            "run_id": run_id,
            "action": action,
            "branch": record["branch"],
            "worktree": str(child),
            "source": str(child),
            "destination": str(root),
            "diffstat": preview,
        }
        if action == "discard":
            payload.update(
                retains_checkout=True,
                retained_baseline="Discard removes changes and the branch; a detached baseline checkout is retained.",
            )
        try:
            decision = request_confirmation(payload)
        except Exception:  # noqa: BLE001 - callback failure denies consent
            _refuse(
                "confirmation_failed", "Confirmation failed; source work is unchanged."
            )
        if not isinstance(decision, dict) or decision.get("allow") is not True:
            _refuse(
                "confirmation_denied",
                "Recovery was not allowed; source work is unchanged.",
            )
        if should_cancel():
            _refuse("cancelled", "Recovery cancelled before mutation.")
        if _snapshot(record, authority)[0] != fingerprint:
            _refuse(
                "preview_changed",
                "Work changed during confirmation; request a fresh preview.",
            )
        if (
            action != "discard"
            and not run_git(
                child,
                "diff",
                "--name-only",
                "--no-ext-diff",
                "--no-textconv",
                "--no-color",
                record["base_sha"],
            )
            and not run_git(child, "ls-files", "--others", "--exclude-standard", "-z")
        ):
            _refuse(
                "nothing_to_merge", "The child has no changes beyond its original base."
            )
        operation_id = uuid.uuid4().hex
        if not repository.claim(
            run_id, conversation_id, operation_id=operation_id, action=action
        ):
            operation_id = None
            _refuse(
                "claim_refused",
                "Another operation owns this work or it is no longer eligible.",
            )
        _authority(record, authority)
        if should_cancel():
            if not repository.finish_operation(
                run_id, operation_id, state="unresolved"
            ):
                _refuse(
                    "persistence_failed", "Cancellation receipt could not be saved."
                )
            operation_id = None
            _refuse("cancelled", "Recovery cancelled before mutation.")
        commit_sha = None
        if action == "discard":
            expected = run_git(child, "rev-parse", "HEAD").decode().strip()
            _authority(record, authority)
            source_effect = True
            # Delete descendants through pinned descriptors before checkout; keep .git and root.
            _tree_digest(child, cleanup=True)
            _authority(record, authority)
            run_git(child, "checkout", "--detach", "--force", record["base_sha"])
            _authority(record, authority)
            run_git(
                root, "update-ref", "-d", "refs/heads/" + record["branch"], expected
            )
            result_state = "discarded_cleanup_pending"
            message = "Discarded agent changes and branch; detached baseline checkout retained."
        else:
            if run_git(child, "status", "--porcelain=v1", "-z"):
                source_effect = True
                run_git(child, "add", "-A")
                _authority(record, authority)
                run_git(
                    child,
                    "-c",
                    "user.name=tldw-agent",
                    "-c",
                    "user.email=agent@tldw.local",
                    "commit",
                    "-m",
                    "Capture confirmed agent work",
                )
            head = run_git(child, "rev-parse", "HEAD").decode().strip()
            _authority(record, authority)
            if action == "apply":
                with tempfile.NamedTemporaryFile(suffix=".patch") as patch:
                    run_git(
                        child,
                        "diff",
                        "--binary",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--no-color",
                        record["base_sha"],
                        head,
                        output=patch,
                    )
                    patch.flush()
                    run_git(root, "apply", "--check", patch.name)
                    _authority(record, authority)
                    destination_effect = True
                    run_git(root, "apply", patch.name)
                result_state = "applied"
                message = (
                    "Applied agent changes as unstaged edits; source checkout retained."
                )
            else:
                if run_git(
                    root, "status", "--porcelain=v1", "-z"
                ) or _operation_in_progress(root):
                    _refuse(
                        "destination_busy",
                        "Destination changed before merge; source capture is retained.",
                    )
                previous = run_git(root, "rev-parse", "HEAD")
                destination_effect = True
                try:
                    run_git(
                        root,
                        "merge",
                        "--no-ff",
                        "--no-edit",
                        head,
                        "-m",
                        "Merge confirmed agent work",
                    )
                except OperationError as exc:
                    # Only our MERGE_HEAD proves this operation owns the merge.
                    if (
                        exc.code == "git_failed"
                        and _operation_in_progress(root)
                        and run_git(root, "rev-parse", "MERGE_HEAD").strip()
                        == head.encode()
                    ):
                        _authority(record, authority)
                        run_git(root, "merge", "--abort")
                        if (
                            run_git(root, "rev-parse", "HEAD") == previous
                            and not run_git(root, "status", "--porcelain=v1", "-z")
                            and not _operation_in_progress(root)
                        ):
                            destination_effect = False
                    raise
                commit_sha = run_git(root, "rev-parse", "HEAD").decode().strip()
                result_state = "merged"
                message = "Merged agent work with an explicit merge commit; source checkout retained."
        _authority(record, authority)
        if not repository.finish_operation(run_id, operation_id, state=result_state):
            _refuse(
                "persistence_failed",
                "Git completed but its receipt could not be persisted; do not retry automatically.",
            )
        operation_id = None
        return WorktreeRecoveryOutcome(action, message, result_state, commit_sha)
    except Exception as exc:  # noqa: BLE001 - all ambiguous outcomes stay protected
        code = exc.code if isinstance(exc, OperationError) else "recovery_uncertain"
        message = (
            str(exc)
            if isinstance(exc, OperationError)
            else "Recovery could not establish completion; work is retained."
        )
        if operation_id is not None:
            known_no_effect = (
                not destination_effect
                and action != "discard"
                and isinstance(exc, OperationError)
                and code
                in (
                    "git_failed",
                    "output_limit",
                    "destination_busy",
                    "source_authority_revoked",
                    "identity_changed",
                )
            )
            try:
                completed = repository.finish_operation(
                    run_id,
                    operation_id,
                    state="unresolved" if known_no_effect else "uncertain",
                )
                if not completed:
                    code, message = (
                        "persistence_failed",
                        "Operation receipt is unconfirmed; automatic retry is disabled.",
                    )
            except Exception:  # noqa: BLE001 - failed persistence must never permit replay
                code, message = (
                    "persistence_failed",
                    "Operation receipt could not be saved; automatic retry is disabled.",
                )
            if source_effect:
                message += " A source capture or partial cleanup may have occurred; checkout retained."
        return WorktreeRefusal(code, message)
