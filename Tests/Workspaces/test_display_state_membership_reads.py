"""The Console workspace rail reads memberships once per build.

TASK-32901 (tier-2 S13 P2): ``build_console_workspace_state`` runs on the
Textual event loop by design, and called ``list_workspace_memberships`` twice
-- once for conversation rows, once for handoff rows. That accessor is an
unpaged ``SELECT * ... WHERE workspace_id = ?`` with ``fetchall()`` and no
``LIMIT``; its paged siblings (``list_workspace_source_memberships``,
``list_workspace_note_memberships``) prove the table is expected to grow.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Workspaces import display_state
from tldw_chatbook.Workspaces.models import (
    DEFAULT_WORKSPACE_ID,
    WorkspaceAuthority,
    WorkspaceRecord,
)


class _CountingRegistry:
    def __init__(self, active, memberships=()):
        self.active = active
        self.memberships = tuple(memberships)
        self.membership_reads = 0

    def get_active_workspace(self):
        return self.active

    def list_workspace_memberships(self, workspace_id):
        self.membership_reads += 1
        return self.memberships

    def list_workspaces(self):
        return (self.active,)


@pytest.fixture()
def workspace():
    return WorkspaceRecord(
        workspace_id="workspace-alpha",
        name="Alpha",
        description="",
        authority=WorkspaceAuthority.LOCAL_ONLY,
    )


def test_console_state_build_reads_memberships_once(workspace):
    registry = _CountingRegistry(workspace)

    display_state.build_console_workspace_state(
        registry_service=registry,
        current_conversation=None,
    )

    assert registry.membership_reads == 1


def test_supplied_conversations_still_read_memberships_only_once(workspace):
    registry = _CountingRegistry(workspace)

    display_state.build_console_workspace_state(
        registry_service=registry,
        current_conversation=None,
        conversations=(),
    )

    assert registry.membership_reads <= 1


def test_membership_read_failure_still_degrades_to_empty_rows(workspace):
    """A failing registry read must not escape the pure UI-loop state build."""

    class _Failing(_CountingRegistry):
        def list_workspace_memberships(self, workspace_id):
            self.membership_reads += 1
            raise RuntimeError("workspace storage unavailable")

    registry = _Failing(workspace)
    state = display_state.build_console_workspace_state(
        registry_service=registry,
        current_conversation=None,
    )

    assert registry.membership_reads == 1
    assert state.conversation_rows == ()
    assert state.handoff_rows == ()
