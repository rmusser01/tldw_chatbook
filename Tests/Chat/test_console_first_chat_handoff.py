"""First-chat handoff contract (TASK-21145, UAT H-2/H-3).

The UAT incident: on a fresh profile, the very first "Hello" was
intercepted by a "Project instructions need a folder" modal exposing the
raw ``no_eligible_binding`` code, and with a broken provider the composer
sat on "Validating provider." for 30s+ with no terminal state.
"""

from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    project_recovery_should_skip_send_interception,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)


class _HangingGateway:
    """resolve_for_send that never returns (the H-3 hang shape)."""

    async def resolve_for_send(self, selection):
        await asyncio.sleep(3600)

    async def stream_chat(self, resolution, messages, **kwargs):
        yield ""


def test_fresh_session_no_eligible_binding_never_intercepts_send():
    state = ProjectInstructionControlState.new_session()
    assert state.project_instructions_enabled
    assert state.working_folder_binding_id is None
    assert project_recovery_should_skip_send_interception(
        "no_eligible_binding", state
    )


@pytest.mark.parametrize(
    "code", ["binding_unavailable", "binding_retargeted", "choose_binding"]
)
def test_real_recovery_codes_still_intercept(code):
    state = ProjectInstructionControlState.new_session()
    assert not project_recovery_should_skip_send_interception(code, state)


def test_broken_existing_binding_still_intercepts_even_for_no_eligible():
    bound = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-1",
        working_folder_locator_fingerprint="fp",
        project_instruction_notice_key=None,
    )
    assert not project_recovery_should_skip_send_interception(
        "no_eligible_binding", bound
    )


@pytest.mark.asyncio
async def test_provider_validation_reaches_terminal_state_in_bounded_time():
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=_HangingGateway()
    )
    controller.PROVIDER_VALIDATION_TIMEOUT_SECONDS = 0.05
    resolution = await asyncio.wait_for(
        controller._resolve_for_send_bounded(object()), timeout=5.0
    )
    assert resolution.ready is False
    assert "timed out" in resolution.visible_copy
    # The copy carries no raw internal codes.
    assert "_" not in resolution.visible_copy


# ---------------------------------------------------------------------------
# TASK-21150 item (c): every resolve_for_send await must carry the H-3
# deadline, not just the send path. A hang in continuation replay, dispatch
# retry, the instruction preview, or compaction wedges its own surface just
# as badly — the source-level check is the guard that keeps a NEW await from
# reintroducing the hazard.
# ---------------------------------------------------------------------------


def test_no_unbounded_resolve_for_send_awaits_remain():
    """Every gateway resolve must go through the bounded seam.

    A source-level assertion (not a behavioral one) because the hazard is
    "someone adds await self.provider_gateway.resolve_for_send(...) again":
    only the text can catch that before it ships.
    """
    import ast
    from pathlib import Path

    source = Path("tldw_chatbook/Chat/console_chat_controller.py").read_text()
    tree = ast.parse(source)
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "resolve_for_send"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "provider_gateway"
        ):
            offenders.append(node.lineno)

    # The single sanctioned call is inside _resolve_for_send_bounded itself,
    # where asyncio.wait_for supplies the deadline.
    bounded = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef)
        and n.name == "_resolve_for_send_bounded"
    )
    allowed = {
        n.lineno
        for n in ast.walk(bounded)
        if isinstance(n, ast.Await)
    }
    unbounded = sorted(set(offenders) - allowed)
    assert not unbounded, (
        "unbounded provider_gateway.resolve_for_send await(s) at line(s) "
        f"{unbounded} — route them through self._resolve_for_send_bounded "
        "so no surface can hang (TASK-21150 item c / UAT H-3)"
    )


@pytest.mark.asyncio
async def test_compaction_reaches_a_terminal_answer_when_the_gateway_hangs():
    """Behavioral half of item (c): the source guard proves the seam is
    used; this proves the seam works on a surface other than send."""
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=_HangingGateway()
    )
    controller.PROVIDER_VALIDATION_TIMEOUT_SECONDS = 0.05
    session = controller.store.ensure_session()
    ok, message = await asyncio.wait_for(
        controller.compact_context_now(session.id), timeout=5.0
    )
    assert ok is False
    assert message, "a blocked compaction must say something actionable"
    assert "_" not in message, f"raw internal code leaked: {message!r}"


# ---------------------------------------------------------------------------
# Qodo review of PR #2131 (bug): bounding continuation replay made
# _resolve_for_send_bounded return a not-ready stand-in instead of raising,
# and the caller's single combined condition relabelled that timeout as
# "Prepared destination changed." — the wrong reason, and it threw away the
# recovery guidance the bound exists to deliver.
# ---------------------------------------------------------------------------


def _resolution(**kwargs):
    from types import SimpleNamespace

    return SimpleNamespace(**kwargs)


def test_prepared_continuation_surfaces_timeout_copy_not_destination_changed():
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=_HangingGateway()
    )
    timeout_copy = (
        "Provider validation timed out. Check the server or your connection, "
        "then try again."
    )
    copy = controller._prepared_continuation_block_copy(
        _resolution(ready=False, visible_copy=timeout_copy),
        expected_destination=object(),
    )
    assert "timed out" in copy, f"timeout guidance was discarded: {copy!r}"
    assert "destination changed" not in copy.lower()


def test_prepared_continuation_not_ready_without_copy_still_explains_itself():
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=_HangingGateway()
    )
    copy = controller._prepared_continuation_block_copy(
        _resolution(ready=False, visible_copy=""), expected_destination=object()
    )
    assert copy, "a blocked continuation must always say something"
    assert "_" not in copy, f"raw internal code leaked: {copy!r}"


def test_prepared_continuation_still_reports_a_real_destination_change():
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=_HangingGateway()
    )
    copy = controller._prepared_continuation_block_copy(
        _resolution(ready=True, resolved_destination=object()),
        expected_destination=object(),
    )
    assert copy == "Prepared destination changed."


def test_prepared_continuation_allows_an_unchanged_destination():
    from tldw_chatbook.Chat.console_chat_controller import ConsoleResolvedDestination

    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=_HangingGateway()
    )
    destination = ConsoleResolvedDestination.__new__(ConsoleResolvedDestination)
    copy = controller._prepared_continuation_block_copy(
        _resolution(ready=True, resolved_destination=destination),
        expected_destination=destination,
    )
    assert copy == "", f"an unchanged destination must proceed, got {copy!r}"
