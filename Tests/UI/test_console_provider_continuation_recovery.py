"""Keyboard and privacy coverage for Console continuation recovery."""

from __future__ import annotations

from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    ProviderContinuationRecoveryCallout,
    provider_continuation_recovery_state,
)


def _message(
    *, call_state: str = "pending", remote: bool = False
) -> ConsoleChatMessage:
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k2",
        api_base_url="https://api.moonshot.ai/v1",
        state="active",
        rounds=(
            ContinuationRound(
                "PRIVATE_REASONING_CANARY",
                ("PRIVATE_REASONING_BLOCK",),
                (
                    ContinuationCall(
                        "PRIVATE_CALL_ID",
                        "calculator",
                        '{"secret":"PRIVATE_ARGUMENT_CANARY"}',
                        call_state,  # type: ignore[arg-type]
                    ),
                ),
            ),
        ),
    )
    return ConsoleChatMessage(
        id="assistant-owner",
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        provider_continuation=checkpoint,
        provider_continuation_remote=remote,
        provider_continuation_message_version=1,
    )


class _RecoveryApp(App):
    def __init__(self, message: ConsoleChatMessage) -> None:
        super().__init__()
        self.message = message
        self.actions: list[tuple[str, str, int]] = []

    def compose(self) -> ComposeResult:
        yield ProviderContinuationRecoveryCallout(
            state=provider_continuation_recovery_state(self.message),
            on_action=self._record,
        )

    async def _record(self, action: str, message_id: str, version: int) -> bool:
        self.actions.append((action, message_id, version))
        return True


async def test_pending_recovery_is_private_wrapped_and_keyboard_reachable() -> None:
    app = _RecoveryApp(_message())
    async with app.run_test(size=(38, 16)) as pilot:
        await pilot.pause()
        rendered = "\n".join(
            str(widget.render())
            for widget in app.screen.query("ProviderContinuationRecoveryCallout *")
            if hasattr(widget, "render")
        )
        assert "Interrupted tool run" in rendered
        assert "may not have finished" in rendered
        assert "PRIVATE_" not in rendered
        assert app.actions == []

        await pilot.press("tab")
        await pilot.press("shift+tab")
        await pilot.press("enter")
        await pilot.pause()
        assert app.actions == [("resume", "assistant-owner", 1)]


async def test_executing_recovery_blocks_resume_and_explains_why() -> None:
    app = _RecoveryApp(_message(call_state="executing"))
    async with app.run_test(size=(42, 16)) as pilot:
        await pilot.pause()
        assert len(app.screen.query("#console-continuation-resume")) == 0
        detail = str(app.screen.query_one("#console-continuation-impact").render())
        assert "may already have run" in detail
        assert app.actions == []


async def test_remote_recovery_offers_take_over_not_resume() -> None:
    app = _RecoveryApp(_message(remote=True))
    async with app.run_test(size=(46, 18)) as pilot:
        await pilot.pause()
        assert len(app.screen.query("#console-continuation-resume")) == 0
        assert len(app.screen.query("#console-continuation-take-over")) == 1
        warning = str(app.screen.query_one("#console-continuation-impact").render())
        assert "other device may still be running" in warning
        assert "exactly-once" not in warning
