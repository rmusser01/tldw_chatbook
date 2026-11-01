"""TASK-34351: a sub-agent worktree retired mid-dispatch gets honest copy.

A sub-agent run with isolation is routed to its own git worktree
(``admit_run_workspace_root``) and un-routed when that run ends
(``retire_run_workspace_root``). A path-tool call from the run can lose the
race with its own retirement in two places. Before the approval gate, the
provider refused with the private-scratch copy, which names the wrong thing.
After the gate, it raised a bare ``KeyError``, so the model saw the alias as
its error text. These tests drive both windows with the real provider, the
real subprocess executor and a real retire, triggered at the exact point by
the provider's own callbacks rather than by editing its internals. Each runs
in a fresh real profile (``private_profile_test``) so the local
backup-recovery bootstrap state of the developer's profile cannot interfere.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL,
    LOCAL_RUN_WORKTREE_RELEASED_REFUSAL,
    LocalToolProvider,
    RunAdmittedWorkspaceRoot,
)
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor

ALLOW = EffectiveToolState(state="allow", origin="tool_override")
ASK = EffectiveToolState(state="ask", origin="global_default")
RUN = "run-child"


def _identity(root: Path) -> tuple[tuple[str, int, int, int], ...]:
    rows = []
    for component in (*reversed(root.parents), root):
        value = os.lstat(component)
        rows.append((str(component), value.st_dev, value.st_ino, value.st_mode))
    return tuple(rows)


def _worktree_authority(root: Path, *, guard) -> RunAdmittedWorkspaceRoot:
    """Build the authority exactly as ``AgentService`` admits a worktree."""
    root = root.resolve()
    return RunAdmittedWorkspaceRoot(
        workspace_id="workspace-1",
        binding_id="folder-1",
        alias=f"agent-{RUN}",
        root=root,
        locator_fingerprint="f" * 64,
        root_identity=_identity(root),
        allow_write=True,
        guard=guard,
        workspace_executor=WorkspaceToolExecutor(root),
    )


def _provider(shared: Path, *, state, approval_callback=None) -> LocalToolProvider:
    return LocalToolProvider(
        workspace_root=shared.resolve(),
        allow_write=True,
        resolve_state=lambda _hub: state,
        kill_switch=lambda: False,
        approval_callback=approval_callback,
    )


def _assert_honest_release_refusal(result, worktree: Path) -> None:
    assert result.ok is False
    assert result.error == LOCAL_RUN_WORKTREE_RELEASED_REFUSAL
    assert "scratch" not in result.error.lower()
    assert f"agent-{RUN}" not in result.error
    assert not (worktree / "out.txt").exists()


@pytest.fixture
def roots(tmp_path):
    shared = tmp_path / "shared"
    worktree = tmp_path / "wt"
    shared.mkdir()
    worktree.mkdir()
    return shared, worktree


@pytest.mark.asyncio
@private_profile_test
def test_retire_before_the_gate_names_the_released_worktree(request, roots):
    """The run ends between root selection and the dispatch-spec lookup."""
    shared, worktree = roots
    provider_box: list[LocalToolProvider] = []
    retired = []

    def guard(_write: bool) -> bool:
        # The first validity check runs right after root selection; the run
        # ends at exactly that moment.
        if not retired:
            retired.append(True)
            provider_box[0].retire_run_workspace_root(RUN)
        return worktree.is_dir()

    provider = _provider(shared, state=ALLOW)
    provider_box.append(provider)
    provider.admit_run_workspace_root(RUN, _worktree_authority(worktree, guard=guard))

    with use_run_id(RUN):
        result = provider.invoke(
            "local:fs_write", {"path": "out.txt", "content": "late\n"}
        )

    assert retired == [True]
    _assert_honest_release_refusal(result, worktree)
    assert not (shared / "out.txt").exists()


@pytest.mark.asyncio
@private_profile_test
def test_run_ending_while_its_approval_card_waits_is_refused_honestly(request, roots):
    """The realistic window: the approval card outlives the run."""
    shared, worktree = roots
    provider_box: list[LocalToolProvider] = []
    cards = []

    def approve_after_the_run_ended(pending):
        cards.append(pending)
        provider_box[0].retire_run_workspace_root(RUN)
        return {"fs_write": "approve_once"}

    provider = _provider(
        shared, state=ASK, approval_callback=approve_after_the_run_ended
    )
    provider_box.append(provider)
    provider.admit_run_workspace_root(
        RUN, _worktree_authority(worktree, guard=lambda _write: worktree.is_dir())
    )

    with use_run_id(RUN):
        result = provider.invoke(
            "local:fs_write", {"path": "out.txt", "content": "late\n"}
        )

    assert len(cards) == 1
    _assert_honest_release_refusal(result, worktree)
    assert not (shared / "out.txt").exists()


@pytest.mark.asyncio
@private_profile_test
def test_unavailable_private_scratch_keeps_the_scratch_copy(request, tmp_path):
    """A real closed scratch space is still reported as scratch."""
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = manager.snapshot("session-1")
    provider = LocalToolProvider(
        workspace_root=snapshot.root,
        allow_write=True,
        resolve_state=lambda _hub: ALLOW,
        kill_switch=lambda: False,
        authority_scope=functools.partial(manager.lease, snapshot),
    )
    manager.close("session-1")
    try:
        with use_run_id(RUN):
            result = provider.invoke(
                "local:fs_write", {"path": "out.txt", "content": "x\n"}
            )
    finally:
        manager.dispose()

    assert result.ok is False
    assert result.error == LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL
