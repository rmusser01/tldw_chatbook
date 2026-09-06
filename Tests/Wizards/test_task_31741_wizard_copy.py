"""task-31741 copy contract: the wizard must not promise a blocked continue.

Release UAT found the Provider step showing "Couldn't discover models for
OpenAI. You can continue anyway." while Next was simultaneously hard-blocked
by commit()'s "API key required." -- both on screen at once. The Welcome
step compounded it by promising "every step can be skipped with Next",
which a keyed provider without a credential refuses. Copy only; the
readiness gate itself is untouched.
"""

import pytest
from textual.widgets import Static

from Tests.Wizards.test_first_run_setup_wizard import (
    _HostApp,
    _make_wizard,
    _provider_step,
    _StepHost,
)


@pytest.mark.asyncio
async def test_blocked_provider_discovery_failure_never_says_continue_anyway():
    """Keyed provider, no credential: Next refuses, so the copy must too."""
    step = _provider_step()
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("openai")
        await pilot.pause()
        copy = step._discovery_failure_status("OpenAI")
        assert "continue anyway" not in copy.lower(), (
            "discovery-failure copy promises continuing while commit() is"
            f" hard-blocked on the missing key: {copy!r}"
        )
        assert "API key" in copy, (
            f"blocked-state copy should name the unblock (an API key): {copy!r}"
        )


@pytest.mark.asyncio
async def test_ready_provider_discovery_failure_still_offers_continue_anyway():
    """No-key local provider: continuing IS allowed, keep saying so."""
    step = _provider_step()
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.select_provider("ollama")
        await pilot.pause()
        copy = step._discovery_failure_status("Ollama")
        assert "You can continue anyway." in copy, (
            f"a provider Next would accept lost its reassurance: {copy!r}"
        )


@pytest.mark.asyncio
async def test_welcome_does_not_promise_every_step_is_skippable():
    """The Provider step refuses Next for a keyed provider without a key,
    so Welcome must not teach "every step can be skipped with Next" -- and
    should name the exit that always works instead (Esc)."""
    wizard = _make_wizard()
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        subtitles = " ".join(
            str(static.renderable)
            for static in wizard.query(".setup-welcome .setup-subtitle").results(
                Static
            )
        )
        assert "every step can be" not in subtitles, (
            f"Welcome still over-promises skipping: {subtitles!r}"
        )
        assert "Esc" in subtitles, (
            "Welcome should point at the universal out (Esc) once the"
            f" every-step promise is gone: {subtitles!r}"
        )
