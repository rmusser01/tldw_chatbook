"""Shared Console capture-policy dialog behavior and compact geometry."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_chat_controller import (
    CapturePolicyMutationResult,
    CapturePolicyMutationStatus,
    CapturePolicySnapshot,
    CapturePurgeResult,
)
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    CapturePolicyResolution,
    CapturePolicySource,
)
from tldw_chatbook.Widgets.Console.console_capture_policy_dialog import (
    CapturePolicyBindings,
    CaptureScope,
    ConsoleCapturePolicyDialog,
)


def _snapshot(**changes: object) -> CapturePolicySnapshot:
    base = CapturePolicySnapshot(
        session_id="session-a",
        conversation_id="conversation-a",
        conversation_title="Pinned chat",
        enabled=True,
        next_detail=None,
        conversation_detail=CaptureDetail.SAFE,
        global_detail=CaptureDetail.SAFE,
        effective=CapturePolicyResolution(
            True, CaptureDetail.SAFE, CapturePolicySource.CONVERSATION, ()
        ),
        policy_revision=3,
        config_generation=4,
        capture_revision=5,
        active_run_detail=None,
        queued_consumer=False,
        save_pending=False,
        error_code=None,
    )
    return replace(base, **changes)


class _Harness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield Static("opener", id="opener")


class _PolicyHost:
    def __init__(self, snapshot: CapturePolicySnapshot) -> None:
        self.snapshot = snapshot
        self.calls: list[tuple[str, object]] = []
        self.confirmations: list[str] = []
        self.full_count = 2
        self.capture_revision = snapshot.capture_revision

    def bindings(self) -> CapturePolicyBindings:
        async def conversation(
            detail: CaptureDetail | None, revision: int
        ) -> CapturePolicyMutationResult:
            self.calls.append(("conversation", detail))
            assert revision == self.snapshot.policy_revision
            self.snapshot = replace(
                self.snapshot,
                conversation_detail=detail,
                effective=CapturePolicyResolution(
                    self.snapshot.enabled,
                    detail or self.snapshot.global_detail,
                    CapturePolicySource.CONVERSATION
                    if detail is not None
                    else CapturePolicySource.GLOBAL,
                    (),
                ),
                policy_revision=revision + 1,
            )
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.APPLIED,
                self.snapshot,
                False,
                None,
            )

        def next_send(
            detail: CaptureDetail | None, revision: int
        ) -> CapturePolicyMutationResult:
            self.calls.append(("next", detail))
            self.snapshot = replace(
                self.snapshot,
                next_detail=detail,
                policy_revision=revision + 1,
            )
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.APPLIED,
                self.snapshot,
                False,
                None,
            )

        def global_apply(
            enabled: bool,
            detail: CaptureDetail,
            config_generation: int,
            policy_revision: int,
        ) -> CapturePolicyMutationResult:
            self.calls.append(("global", (enabled, detail)))
            self.snapshot = replace(
                self.snapshot,
                enabled=enabled,
                global_detail=detail,
                config_generation=config_generation + 1,
                policy_revision=policy_revision + 1,
                effective=CapturePolicyResolution(
                    enabled,
                    detail,
                    CapturePolicySource.GLOBAL
                    if enabled
                    else CapturePolicySource.DISABLED,
                    (),
                ),
            )
            return CapturePolicyMutationResult(
                CapturePolicyMutationStatus.APPLIED,
                self.snapshot,
                False,
                None,
            )

        async def count() -> int:
            return self.full_count

        async def purge(revision: int) -> CapturePurgeResult:
            self.calls.append(("purge", revision))
            self.full_count = 0
            self.capture_revision += 1
            return CapturePurgeResult.deleted(2, self.capture_revision)

        return CapturePolicyBindings(
            target_session_id=self.snapshot.session_id,
            target_conversation_id=self.snapshot.conversation_id,
            read=lambda: self.snapshot,
            apply_next=next_send,
            apply_conversation=conversation,
            apply_global=global_apply,
            count_full=count,
            purge_full=purge,
            capture_revision=lambda: self.capture_revision,
        )


@pytest.mark.asyncio
async def test_apply_mutates_only_selected_scope_and_next_full_skips_confirmation() -> None:
    host = _PolicyHost(_snapshot())
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleCapturePolicyDialog(host.bindings())
        await app.push_screen(dialog)
        await pilot.pause()
        confirmations = 0

        async def confirm(_message: str, **_kwargs: str) -> bool:
            nonlocal confirmations
            confirmations += 1
            return True

        dialog._confirm = confirm
        result = await dialog.apply(CaptureScope.NEXT_SEND, CaptureDetail.FULL)

        assert result is not None and result.status is CapturePolicyMutationStatus.APPLIED
        assert host.calls == [("next", CaptureDetail.FULL)]
        assert confirmations == 0


@pytest.mark.asyncio
async def test_inherit_that_reveals_full_requires_confirmation() -> None:
    host = _PolicyHost(
        _snapshot(
            conversation_detail=CaptureDetail.SAFE,
            global_detail=CaptureDetail.FULL,
        )
    )
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleCapturePolicyDialog(host.bindings())
        await app.push_screen(dialog)
        await pilot.pause()
        messages: list[str] = []

        async def decline(message: str, **_kwargs: str) -> bool:
            messages.append(message)
            return False

        dialog._confirm = decline
        assert dialog.preview_for(CaptureScope.CONVERSATION, None).requires_confirmation
        assert await dialog.apply(CaptureScope.CONVERSATION, None) is None
        assert host.calls == []
        assert messages and "Full" in messages[0]


@pytest.mark.asyncio
async def test_off_blocks_full_scope_edits_but_preserves_dormant_global_full() -> None:
    host = _PolicyHost(
        _snapshot(
            enabled=False,
            global_detail=CaptureDetail.FULL,
            effective=CapturePolicyResolution(
                False, CaptureDetail.SAFE, CapturePolicySource.DISABLED, ()
            ),
        )
    )
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleCapturePolicyDialog(host.bindings())
        await app.push_screen(dialog)
        await pilot.pause()

        assert await dialog.apply(CaptureScope.CONVERSATION, CaptureDetail.FULL) is None
        assert host.calls == []
        assert "Capture Off" in dialog.status_text
        assert "Dormant Full" in str(
            dialog.query_one("#capture-policy-effective", Static).render()
        )


@pytest.mark.asyncio
async def test_off_allows_safe_edit_and_resume_warns_once_for_dormant_conversation_full() -> None:
    host = _PolicyHost(
        _snapshot(
            enabled=False,
            conversation_detail=CaptureDetail.FULL,
            global_detail=CaptureDetail.SAFE,
            effective=CapturePolicyResolution(
                False, CaptureDetail.SAFE, CapturePolicySource.DISABLED, ()
            ),
        )
    )
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleCapturePolicyDialog(host.bindings())
        await app.push_screen(dialog)
        await pilot.pause()
        confirmations: list[str] = []

        async def confirm(message: str, **_kwargs: str) -> bool:
            confirmations.append(message)
            return True

        dialog._confirm = confirm
        assert await dialog.apply(CaptureScope.CONVERSATION, CaptureDetail.SAFE)
        host.snapshot = replace(
            host.snapshot,
            conversation_detail=CaptureDetail.FULL,
            enabled=False,
            effective=CapturePolicyResolution(
                False, CaptureDetail.SAFE, CapturePolicySource.DISABLED, ()
            ),
        )
        dialog.snapshot = host.snapshot

        assert await dialog.set_capture_enabled(True)
        assert len(confirmations) == 1
        assert "dormant Full" in confirmations[0]


@pytest.mark.asyncio
async def test_stale_and_partial_success_results_use_literal_status() -> None:
    host = _PolicyHost(_snapshot())
    bindings = host.bindings()

    def stale(
        _detail: CaptureDetail | None, _revision: int
    ) -> CapturePolicyMutationResult:
        return CapturePolicyMutationResult(
            CapturePolicyMutationStatus.STALE,
            host.snapshot,
            True,
            "stale_policy_revision",
        )

    bindings = replace(bindings, apply_next=stale)
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleCapturePolicyDialog(bindings)
        await app.push_screen(dialog)
        await pilot.pause()

        assert await dialog.apply(CaptureScope.NEXT_SEND, CaptureDetail.SAFE)
        assert dialog.status_text == "Failed — policy changed; reopen and try again"

        degraded = CapturePolicyMutationResult(
            CapturePolicyMutationStatus.APPLIED,
            host.snapshot,
            False,
            "cache_refresh_degraded",
        )
        dialog._consume_mutation(degraded)
        assert dialog.status_text == "Saved and active — settings cache refresh degraded"


@pytest.mark.asyncio
async def test_purge_confirmation_names_logical_deletion_wal_and_policy() -> None:
    host = _PolicyHost(_snapshot())
    app = _Harness()
    async with app.run_test() as pilot:
        dialog = ConsoleCapturePolicyDialog(host.bindings())
        await app.push_screen(dialog)
        await pilot.pause()
        messages: list[str] = []

        async def confirm(message: str, **_kwargs: str) -> bool:
            messages.append(message)
            return True

        dialog._confirm = confirm
        result = await dialog.delete_full_captures()

        assert result is not None and result.removed_count == 2
        assert "logical record deletion" in messages[0]
        assert "WAL" in messages[0]
        assert "capture policy remains Full" in messages[0]
        assert "Deleted 2" in dialog.status_text


@pytest.mark.asyncio
async def test_compact_dialog_keeps_status_and_fixed_actions_visible_and_escape_restores_focus() -> None:
    host = _PolicyHost(_snapshot(queued_consumer=True))
    app = _Harness()
    async with app.run_test(size=(80, 24)) as pilot:
        opener = app.query_one("#opener", Static)
        opener.can_focus = True
        opener.focus()
        dialog = ConsoleCapturePolicyDialog(host.bindings())
        await app.push_screen(dialog)
        await pilot.pause()

        assert "queued exchange" in str(
            dialog.query_one("#capture-policy-effective", Static).render()
        )
        assert dialog.query_one("#capture-policy-status").region.height > 0
        actions = dialog.query_one("#capture-policy-actions")
        assert actions.region.height > 0 and actions.region.bottom <= 24

        await pilot.press("escape")
        await pilot.pause()
        assert app.focused is opener
