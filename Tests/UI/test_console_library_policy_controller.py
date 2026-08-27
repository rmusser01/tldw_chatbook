"""Controller coverage for per-conversation Console Library policy edits."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyHolder,
    ConsoleLibraryPolicySnapshot,
    ConsoleLibraryPolicyWriteResult,
    ConsoleLibraryPolicyWriteStatus,
)
from tldw_chatbook.UI.Console_Modules.library_policy import (
    ConsoleLibraryPolicyController,
)


def _snapshot() -> ConsoleLibraryPolicySnapshot:
    return ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=7,
        source="durable",
    )


def _candidate() -> ConsoleLibraryPolicyCandidate:
    return ConsoleLibraryPolicyCandidate(
        auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
    )


class _Store:
    def __init__(self, session, result=None, *, raises: bool = False) -> None:
        self.session = session
        self.result = result
        self.raises = raises
        self.save_calls = 0

    def stage_session_library_policy(self, _session_id, candidate) -> None:
        current = self.session.library_policy_holder.snapshot
        self.session.library_policy_holder.snapshot = ConsoleLibraryPolicySnapshot(
            auto_retrieve=candidate.auto_retrieve,
            assistant_access=candidate.assistant_access,
            policy_revision=current.policy_revision,
            source=current.source,
        )
        self.session.library_policy_holder.explicitly_staged = True

    async def save_session_library_policy(self, _session_id):
        self.save_calls += 1
        if self.raises:
            raise RuntimeError("storage failed")
        return self.result


def _controller(session, store, sync_calls) -> ConsoleLibraryPolicyController:
    return ConsoleLibraryPolicyController(
        app_instance=SimpleNamespace(notify=lambda *args, **kwargs: None),
        active_session=lambda: session,
        ensure_store=lambda: store,
        direct_library_tools=lambda: True,
        push_screen=lambda _modal: None,
        request_control_bar_sync=lambda: sync_calls.append("sync"),
    )


@pytest.mark.asyncio
async def test_temporary_policy_edit_stays_local_without_durable_write() -> None:
    session = SimpleNamespace(
        id="temp",
        ephemeral=True,
        persisted_conversation_id=None,
        library_policy_holder=ConsoleLibraryPolicyHolder(_snapshot()),
    )
    store = _Store(session)
    sync_calls: list[str] = []

    outcome = await _controller(session, store, sync_calls)._save(_candidate())

    assert outcome.status == "saved"
    assert outcome.snapshot.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
    assert store.save_calls == 0
    assert sync_calls == ["sync"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "raises", "expected_status"),
    [
        (None, True, "error"),
        (
            ConsoleLibraryPolicyWriteResult(
                status=ConsoleLibraryPolicyWriteStatus.CONFLICT,
                snapshot=_snapshot(),
            ),
            False,
            "conflict",
        ),
    ],
)
async def test_failed_durable_save_restores_the_prior_committed_holder(
    result, raises: bool, expected_status: str
) -> None:
    prior = _snapshot()
    session = SimpleNamespace(
        id="durable",
        ephemeral=False,
        persisted_conversation_id="conversation-1",
        library_policy_holder=ConsoleLibraryPolicyHolder(prior),
    )
    store = _Store(session, result, raises=raises)
    sync_calls: list[str] = []

    outcome = await _controller(session, store, sync_calls)._save(_candidate())

    assert outcome.status == expected_status
    assert session.library_policy_holder.snapshot == prior
    assert session.library_policy_holder.explicitly_staged is False
    assert sync_calls == ["sync"]
