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
