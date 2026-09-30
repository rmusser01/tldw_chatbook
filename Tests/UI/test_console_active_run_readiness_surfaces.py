"""A healthy active run is a run-state fact, not a provider blocker (TASK-33620.4).

Review finding G2-02 (2026-09-29): on every healthy send the Inspect rail read
"Run: Blocked", "Provider: blocked - Provider setup needed: current run is
active", the Conversation settings line read "Not ready - current run is
active", the collapsed Inspect edge badge read "setup", and -- during the
pre-acceptance "Preparing..." window -- the composer strip read "Send blocked -
finish provider setup to continue" as a link that opened the first-run wizard
over the running Console.

Two causes, both exercised here through the REAL mounted ``ChatScreen``:

* the screen fed ``active_run`` into provider readiness, so an otherwise
  fully configured provider became ``blocker="active_run"`` and flowed through
  the "Provider setup needed: {blocker}" template into every display consumer
  that did not carry its own ``wait_for_active_run`` guard (TASK-32345 fixed
  one consumer; this pins them together);
* the prompt queue's "Preparing..."/"Queue full" tooltip rode the composer's
  ``setup_blocked_reason`` slot, whose fallback copy blames provider setup and
  whose CSS class turns the strip into a setup-wizard link.

The run is held at the provider twice -- once during validation (the queue's
Preparing window) and once mid-stream (an accepted turn) -- and every surface
is asserted TOGETHER at each hold, because the TASK-32345 lesson is that a
guard applied to one consumer leaves the others lying.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_console_native_chat_flow import (
    DelayedWaitingGateway,
    _build_console_send_test_app,
    _configure_native_ready_console,
    _configure_openai_missing_api_key,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _static_text, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.Chat.console_prompt_queue import MAX_CONSOLE_QUEUE_ENTRIES
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_rail_handle import ConsoleRailHandle

_SIZE = (235, 52)
_SETTLE_TIMEOUT = 10.0
_STRIP = "#console-send-disabled-reason"
#: Inside the strip's text, past its padding cell. Pre-fix, a click here on the
#: "finish provider setup" strip DID fire ``app.run_setup_wizard`` (verified);
#: the widget's (0, 0) corner does not, so a default click proves nothing.
_STRIP_TEXT_OFFSET = (5, 0)


class _WizardRecordingHarness(ConsoleHarness):
    """The real Console host, recording every setup-wizard activation.

    The composer strip's setup link is ``@click=app.run_setup_wizard``; the
    production app opens the first-run wizard there. Recording the action on
    the host is what lets a click on the strip PROVE it never opens setup
    mid-run, rather than inferring it from a CSS class.
    """

    def __init__(self, app_instance) -> None:
        super().__init__(app_instance)
        self.setup_wizard_activations = 0

    def action_run_setup_wizard(self) -> None:
        self.setup_wizard_activations += 1


def _run_line(console) -> str:
    return _static_text(console.query_one("#console-send-authority-run", Static))


def _provider_row(console) -> str:
    return _static_text(console.query_one("#console-inspector-provider", Static))


def _settings_readiness_line(console) -> str:
    return _static_text(
        console.query_one("#console-settings-readiness-row", Static)
    )


def _edge_badge(console) -> str:
    return console.query_one(
        "#console-inspector-rail-handle", ConsoleRailHandle
    ).badge


def _right_rail_text(console) -> str:
    rail = console.query_one("#console-right-rail")
    return " ".join(
        _static_text(widget)
        for widget in rail.query(Static)
        if widget.display and hasattr(widget, "renderable")
    )


async def _open_inspect(console, pilot) -> None:
    await _wait_for_selector(console, pilot, "#console-inspector-rail-open")
    await pilot.click("#console-inspector-rail-open")
    for _ in range(60):
        rail = console.query_one("#console-right-rail")
        if rail.display and rail.styles.display != "none":
            break
        await pilot.pause(0.05)
    await _wait_for_selector(console, pilot, "#console-send-authority-run")
    await _wait_for_selector(console, pilot, "#console-inspector-provider")


async def _settle(console, pilot) -> None:
    # The 0.2s tick races assertions; force the exact sync it performs.
    await console._sync_native_console_chat_ui()
    await pilot.pause()
    await pilot.pause()


def _run_surface_violations(console) -> list[str]:
    """Every run-truth surface that lies about a healthy active run."""
    run_line = _run_line(console)
    provider_row = _provider_row(console)
    settings_line = _settings_readiness_line(console)
    badge = _edge_badge(console)
    rail_text = _right_rail_text(console)
    violations = []
    if run_line != "Run: Running":
        violations.append(f"AC#1 pinned summary {run_line!r} (want 'Run: Running')")
    if provider_row != "Provider: ready":
        violations.append(f"AC#2 Provider row {provider_row!r} (want ready)")
    if "Not ready" in settings_line:
        violations.append(f"AC#2 Conversation settings {settings_line!r}")
    if "Provider setup needed" in rail_text:
        violations.append("AC#2 'Provider setup needed' rendered in the rail")
    if badge != "running":
        violations.append(f"AC#3 edge badge {badge!r} (want 'running')")
    return violations


def _composer_queue_violations(
    composer: ConsoleComposerBar, *, expected: str
) -> list[str]:
    """Every way the composer misreports a queue state as provider setup."""
    reason = str(composer._send_disabled_reason or "")
    tooltip = str(composer.query_one("#console-send-message", Button).tooltip or "")
    violations = []
    if "provider setup" in reason.lower():
        violations.append(f"AC#4 strip blames provider setup: {reason!r}")
    if "provider setup" in tooltip.lower():
        violations.append(f"AC#4 Send tooltip blames provider setup: {tooltip!r}")
    if expected not in reason:
        violations.append(f"AC#4 strip lacks queue copy {expected!r}: {reason!r}")
    if composer.has_class("console-composer-setup-blocked"):
        violations.append("AC#4 strip is the setup-wizard link")
    return violations


@pytest.mark.asyncio
async def test_held_healthy_run_reads_running_on_every_surface_and_never_setup():
    """AC#1-#5: drive a real held provider run with Inspect open."""
    gateway = DelayedWaitingGateway()
    app = _build_console_send_test_app()
    _configure_native_ready_console(app, model="test-model")
    app.console_provider_gateway_factory = lambda: gateway
    host = _WizardRecordingHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _open_inspect(console, pilot)
        await _settle(console, pilot)
        # Baseline: the idle, ready Console the run starts from.
        assert _run_line(console) == "Run: Ready", _run_line(console)
        assert _provider_row(console) == "Provider: ready", _provider_row(console)
        assert _edge_badge(console) == ""

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()

        # --- Hold 1: provider validation (the queue's Preparing window) ----
        await asyncio.wait_for(
            gateway.validation_started.wait(), timeout=_SETTLE_TIMEOUT
        )
        controller = console._ensure_console_chat_controller()
        assert controller.run_state.status is ConsoleRunStatus.VALIDATING
        await _settle(console, pilot)
        send_button = console.query_one("#console-send-message", Button)
        assert send_button.label.plain.startswith("Preparing"), send_button.label
        violations = _run_surface_violations(console) + _composer_queue_violations(
            composer, expected="this turn"
        )
        assert not violations, "[preparing] " + "; ".join(violations)
        # Activating the strip mid-run must never open first-run setup.
        await pilot.click(_STRIP, offset=_STRIP_TEXT_OFFSET)
        await pilot.pause()
        assert host.setup_wizard_activations == 0, (
            "AC#4: clicking the composer strip mid-run opened the setup wizard"
        )

        # --- Hold 2: an accepted turn, stalled mid-stream -------------------
        gateway.validation_release.set()
        await asyncio.wait_for(gateway.started.wait(), timeout=_SETTLE_TIMEOUT)
        await _wait_for_text(console, pilot, "partial")
        await _settle(console, pilot)
        assert controller.run_state.status is ConsoleRunStatus.STREAMING
        violations = _run_surface_violations(console)
        assert not violations, "[streaming] " + "; ".join(violations)

        # Fill the queue behind the live turn through the real Queue action.
        session_id = controller.store.active_session_id
        for index in range(MAX_CONSOLE_QUEUE_ENTRIES):
            composer.load_draft(f"follow-up {index}")
            for _ in range(40):
                if send_button.label.plain.startswith("Queue"):
                    break
                await pilot.pause(0.05)
            await console.handle_console_send_message(Button.Pressed(send_button))
            for _ in range(40):
                count = controller.prompt_queue_registry.snapshot(
                    session_id
                ).total_count
                if count == index + 1:
                    break
                await pilot.pause(0.05)
            else:
                raise AssertionError(f"queue admission stalled at {count}")
        composer.load_draft("one too many")
        await _settle(console, pilot)
        assert send_button.label.plain.startswith("Queue full"), send_button.label
        violations = _run_surface_violations(console) + _composer_queue_violations(
            composer, expected="Queue full"
        )
        assert not violations, "[queue-full] " + "; ".join(violations)
        await pilot.click(_STRIP, offset=_STRIP_TEXT_OFFSET)
        await pilot.pause()
        assert host.setup_wizard_activations == 0, (
            "AC#4: clicking the Queue-full strip opened the setup wizard"
        )

        # Release the held turn and let it finish; the harness teardown
        # cancels whatever the queue does next.
        composer.load_draft("")
        gateway.release.set()
        await _wait_for_text(console, pilot, "partial done")


@pytest.mark.asyncio
async def test_missing_api_key_still_reads_blocked_with_a_recovery_action():
    """AC#5 negative control: a genuinely unconfigured provider keeps its
    blocker, its 'Provider setup needed' copy, a recovery action, the
    'setup' edge badge and the composer's setup link."""
    app = _build_console_send_test_app()
    _configure_openai_missing_api_key(app)
    host = _WizardRecordingHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _open_inspect(console, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")
        await _settle(console, pilot)

        provider_row = _provider_row(console)
        assert provider_row.startswith("Provider: blocked"), provider_row
        assert "Provider setup needed" in provider_row, provider_row
        assert "API key" in provider_row, provider_row
        next_action = _static_text(
            console.query_one("#console-inspector-next-action", Static)
        )
        assert next_action.startswith("Next action: "), next_action
        assert next_action.removeprefix("Next action: ").strip(), (
            f"a genuine blocker must name its recovery action: {next_action!r}"
        )
        run_line = _run_line(console)
        assert run_line in {"Run: Blocked", "Run: Recovery required"}, run_line
        assert _edge_badge(console) == "setup"
        assert "Not ready" in _settings_readiness_line(console)
        reason = str(composer._send_disabled_reason or "")
        assert "api key" in reason.lower(), reason
        assert composer.has_class("console-composer-setup-blocked"), (
            "a genuine setup blocker keeps its way out (TASK-21145)"
        )
