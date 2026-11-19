"""TASK-34100.6 slice A: skipping the Provider step is a real, honest skip.

Review findings new-a11y-terminal-visual-02, new-provider-01, provider-01 and
provider-26 (2026-10-02). On base the only skip was Enter in the empty key
field, which cleared the provider choice but not the wizard's staged draft:
Model then showed the skipped provider and its template model, and Next looped
on 'Saving the provider and model setup failed'. After the skip, Back showed
the skipped provider's form, and a key typed there was silently discarded.

These drive the real wizard with real key presses.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.Wizards.provider_step_harness import (
    SIZE,
    WizardHost,
    current_step,
    key_input,
    pick_provider,
    pinned_error,
    provider_list,
    resumed_on_provider,
    static_text,
    step_id,
    tracker_states,
    wait_for_step,
    wait_until,
)
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_MODEL,
    STEP_PROVIDER,
    STEP_VOICE,
)

# Selecting a provider reads the atomic config snapshot (TASK-32873).
pytestmark = pytest.mark.bootstrap_profile


def _without_openai_env(monkeypatch) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


async def _type_into(pilot, field: Input, text: str) -> None:
    field.focus()
    await pilot.pause()
    await pilot.press(*text)
    await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
async def test_skip_after_a_staged_key_releases_it_and_model_never_loops(
    request, monkeypatch
):
    """AC#1: rejected key -> Back -> clear -> skip -> Next reaches Voice."""
    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        await _type_into(pilot, key_input(provider), "sk-rejected-0001")
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _container, model = current_step(wizard)
        custom = model.query_one("#setup-model-custom", Input)
        custom.value = ""
        await _type_into(pilot, custom, "gpt-4o-mini")

        await pilot.press("ctrl+b")
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        field = key_input(provider)
        field.focus()
        await pilot.pause()
        field.value = ""
        await pilot.pause()
        await pilot.press("enter")  # the documented skip
        await wait_for_step(pilot, wizard, STEP_MODEL)

        assert wizard.query_one("SetupWizardContainer").staged_provider_draft is None
        line = static_text(model, "#setup-model-provider-line")
        assert "OpenAI" not in line, line
        assert model.query_one("#setup-model-custom", Input).value == ""

        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_VOICE)
        assert "Saving the provider and model setup failed" not in pinned_error(wizard)
        states = tracker_states(wizard)
        assert states[STEP_PROVIDER] == "skipped"
        assert states[STEP_MODEL] == "skipped"


@pytest.mark.asyncio
@private_profile_test
async def test_back_after_skip_with_no_repick_still_reaches_voice(request, monkeypatch):
    """AC#1 (review case b): Back -> Ctrl+N without a re-pick never loops."""
    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        await _type_into(pilot, key_input(provider), "sk-rejected-0001")
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        await pilot.press("ctrl+b")
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        key_input(provider).value = ""
        await pilot.pause()
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        await pilot.press("ctrl+b")
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_VOICE)


@pytest.mark.asyncio
@private_profile_test
async def test_skip_keeps_the_choice_so_a_key_typed_after_back_is_staged(
    request, monkeypatch
):
    """AC#2: skip -> Back -> paste key -> Next stages it for that provider."""
    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        field = key_input(provider)
        field.focus()
        await pilot.pause()
        await pilot.press("enter")  # skip with an empty key
        await wait_for_step(pilot, wizard, STEP_MODEL)

        await pilot.press("ctrl+b")
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        assert provider.selected_provider_key == "openai"
        await _type_into(pilot, key_input(provider), "sk-after-back-0001")
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _container, model = current_step(wizard)
        staged = wizard.query_one("SetupWizardContainer").staged_provider_draft
        assert staged is not None and staged.provider == "openai"
        assert static_text(model, "#setup-model-provider-line") == "Models for OpenAI."


@pytest.mark.asyncio
@private_profile_test
async def test_repicking_the_shown_provider_keeps_a_typed_draft_key(
    request, monkeypatch
):
    """AC#2: re-picking the provider already shown keeps the typed key."""
    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        key_input(provider).focus()
        await pilot.pause()
        await pilot.press("enter")  # skip
        await wait_for_step(pilot, wizard, STEP_MODEL)
        await pilot.press("ctrl+b")
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        await _type_into(pilot, key_input(provider), "sk-draft-0001")
        await pick_provider(pilot, provider, "openai")
        assert key_input(provider).value == "sk-draft-0001"


@pytest.mark.asyncio
@private_profile_test
async def test_next_with_an_empty_key_skips_like_enter(request, monkeypatch):
    """AC#3: Next with an empty key on a keyed provider skips."""
    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        assert pinned_error(wizard) == ""
        assert wizard.query_one("SetupWizardContainer").staged_provider_draft is None
        _container, model = current_step(wizard)
        assert "OpenAI" not in static_text(model, "#setup-model-provider-line")


@pytest.mark.asyncio
@private_profile_test
async def test_the_skip_button_skips_for_mouse_users(request, monkeypatch):
    """AC#3: a visible 'Skip — connect later' button does the same."""
    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        skip = provider.query_one("#setup-provider-skip", Button)
        assert skip.display and "Skip" in str(skip.label)
        assert "connect later" in str(skip.label)
        await pilot.click("#setup-provider-skip")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        assert wizard.query_one("SetupWizardContainer").staged_provider_draft is None


@pytest.mark.asyncio
@private_profile_test
async def test_skip_button_hides_once_the_provider_is_ready(request, monkeypatch):
    """The skip control shows only while a keyed provider is not ready."""
    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        skip = provider.query_one("#setup-provider-skip", Button)
        await pick_provider(pilot, provider, "openai")
        assert skip.display
        await _type_into(pilot, key_input(provider), "sk-valid-shape-0001")
        assert not skip.display
        await pick_provider(pilot, provider, "ollama")
        assert not skip.display


@pytest.mark.asyncio
@private_profile_test
async def test_a_refusal_names_the_fix_and_clears_on_provider_change(
    request, monkeypatch
):
    """AC#3: no impossible 'Retry with Next'; the strip clears on a switch."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "azure")
        await _type_into(pilot, key_input(provider), "azure-key-0001")
        await pilot.press("ctrl+n")
        await wait_until(pilot, lambda: pinned_error(wizard) != "")
        refusal = pinned_error(wizard)
        assert step_id(wizard) == STEP_PROVIDER
        assert "Retry with Next" not in refusal
        assert "API key required" not in refusal
        assert "Azure" in refusal and "URL" in refusal
        await pick_provider(pilot, provider, "anthropic")
        assert pinned_error(wizard) == ""


@pytest.mark.asyncio
async def test_a_no_op_repick_never_clears_a_contained_switch_error(monkeypatch):
    """AC#28 (provider-26): the double dispatch's no-op re-pick keeps the line.

    Once the list has been interacted with, one arrow press runs the pick
    twice (OptionHighlighted, then Interacted). A pick that fails after
    selecting its provider leaves the second pass a no-op re-pick of that same
    provider; on base that no-op's success path cleared the line at once.
    """
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer

    armed: list[bool] = []
    real_capture = SetupWizardContainer.capture_provider_config_precondition

    def _snapshot_fails_once(discovery_key):
        if armed and discovery_key.provider_key == "anthropic":
            armed.clear()
            raise OSError("config snapshot unreadable")
        return real_capture(discovery_key)

    monkeypatch.setattr(
        SetupWizardContainer,
        "capture_provider_config_precondition",
        staticmethod(_snapshot_fails_once),
    )
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    app._keep_screen_alive_on_handler_error = True
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")  # the list is interacted
        provider_list(provider).focus()  # the pick moved focus to the key field
        await pilot.pause()
        armed.append(True)
        await pilot.press("down")  # onto Anthropic: selected, then it raises
        await pilot.pause()
        assert not armed, "the failing snapshot was never reached"
        assert provider.selected_provider_key == "anthropic"
        assert "went wrong" in pinned_error(wizard)


@pytest.mark.asyncio
@private_profile_test
async def test_an_exported_key_with_no_resource_url_is_refused_not_skipped(
    request, monkeypatch
):
    """Review round 1, F4: Azure with AZURE_OPENAI_API_KEY exported and no URL
    skipped silently on Next, though the step asked for the URL. The key is
    there; the URL is what's missing, so Next names it. Skip still skips."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key-" + "0" * 30)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "azure")
        ok, refusal = await provider.commit()
        assert not ok
        assert "Azure OpenAI resource URL" in refusal
        assert provider.get_step_data()["skipped"] is False
        skip = provider.query_one("#setup-provider-skip", Button)
        skip.scroll_visible(animate=False)  # below the URL field and env choice
        await pilot.pause()
        await pilot.click("#setup-provider-skip")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        assert provider.get_step_data()["skipped"] is True


@pytest.mark.asyncio
@private_profile_test
async def test_the_pinned_refusal_starts_at_the_steps_content_column(
    request, monkeypatch
):
    """AC#20; review round 1, live F8: the strip's '✗' sat two columns left
    of the Provider step's content column."""
    from textual.widgets import Static

    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "azure")
        await _type_into(pilot, key_input(provider), "azure-key-0001")
        await pilot.press("ctrl+n")
        await wait_until(pilot, lambda: pinned_error(wizard) != "")
        await pilot.pause()
        strip = wizard.query_one("#setup-step-error-pinned", Static)
        column = provider.query_one(".setup-provider").content_region.x
        assert strip.content_region.x == column


def _container_stub(*, staged: bool = True, result=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        staged_provider_draft=object() if staged else None,
        _provider_last_config_result=result,
    )


@pytest.mark.parametrize(
    ("container", "expected"),
    [
        pytest.param(
            "no_staged_draft",
            "No provider is connected, so there is no model to save. Go Back to "
            "connect one, or clear the model name and press Next to skip.",
            id="no-staged-draft",
        ),
        pytest.param(
            "conflict",
            "Your settings changed in another window while saving. Press Next to "
            "save again.",
            id="conflict",
        ),
        pytest.param(
            "before_replace",
            "Couldn't write config.toml, so nothing was saved. Check the disk "
            "and the file's permissions, then press Next to try again.",
            id="before-replace",
        ),
        pytest.param(
            "cache_reload",
            "config.toml was written, but the app couldn't reload it. Press Next "
            "to try again.",
            id="cache-reload",
        ),
        pytest.param(
            "generic",
            "The provider settings couldn't be saved. Go Back to Provider to check "
            "them, or clear the model name and press Next to skip.",
            id="generic",
        ),
    ],
)
def test_a_failed_model_save_names_its_cause_and_a_working_fix(container, expected):
    """AC#1 ('a failed save names its real cause and an action that works');
    review round 1, TE-02: none of the five branches had a test."""
    from tldw_chatbook.config import ConfigMutationResult
    from tldw_chatbook.UI.Wizards.first_run_provider_skip import (
        model_save_failure_message,
    )

    stubs = {
        "no_staged_draft": _container_stub(staged=False),
        "conflict": _container_stub(
            result=ConfigMutationResult(False, False, None, conflict=True)
        ),
        "before_replace": _container_stub(
            result=ConfigMutationResult(False, False, "before_replace")
        ),
        "cache_reload": _container_stub(
            result=ConfigMutationResult(True, False, "cache_reload")
        ),
        "generic": _container_stub(result=None),
    }
    message = model_save_failure_message(stubs[container])
    assert message == expected
    assert "Saving the provider and model setup failed" not in message


@pytest.mark.asyncio
@private_profile_test
async def test_model_names_a_config_write_failure_on_next(request, monkeypatch):
    """TE-02: the Model step's own Next, when the provider+model save fails
    before config.toml is replaced, pins the cause and a fix that can work --
    never the old generic 'Saving the provider and model setup failed'."""
    from unittest.mock import MagicMock

    from tldw_chatbook.config import ConfigMutationResult

    _without_openai_env(monkeypatch)
    persist = MagicMock(return_value=ConfigMutationResult(False, False, "before_replace"))
    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_setup_persistence.persist_provider_setup", persist
    )
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        await _type_into(pilot, key_input(provider), "sk-proj-0001")
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _container, model = current_step(wizard)
        custom = model.query_one("#setup-model-custom", Input)
        custom.value = ""
        await _type_into(pilot, custom, "gpt-4o-mini")
        await pilot.press("ctrl+n")
        await wait_until(pilot, lambda: bool(pinned_error(wizard)))
        assert persist.called, "the save really ran"
        assert step_id(wizard) == STEP_MODEL
        pinned = pinned_error(wizard)
        assert "Couldn't write config.toml, so nothing was saved" in pinned
        assert "press Next to try again" in pinned
        assert "Saving the provider and model setup failed" not in pinned


@pytest.mark.asyncio
async def test_a_retryable_refusal_gets_one_single_spaced_retry_suffix(monkeypatch):
    """AC#24 ('refusal suffixes use single spaces'); review round 1, TE-05: a
    mutant giving RETRY_SUFFIX a double leading space survived. On a step
    that doesn't name its own fix, a refused Next reads '<reason> Retry with
    Next, or go Back.' -- one space, once."""
    from unittest.mock import AsyncMock

    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        await pilot.press("ctrl+n")  # nothing picked: a skip
        await wait_for_step(pilot, wizard, STEP_MODEL)
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_VOICE)
        _container, voice = current_step(wizard)
        assert not getattr(voice, "REFUSAL_NAMES_ITS_FIX", False)
        monkeypatch.setattr(
            voice, "commit", AsyncMock(return_value=(False, "The voice couldn't be saved."))
        )
        await pilot.press("ctrl+n")
        await wait_until(pilot, lambda: bool(pinned_error(wizard)))
        pinned = pinned_error(wizard)
        assert pinned.endswith("The voice couldn't be saved. Retry with Next, or go Back.")
        assert "  " not in pinned
        assert pinned.count("Retry with Next") == 1


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("how", ["mouse", "keyboard"])
async def test_back_after_a_skip_puts_focus_on_the_key_so_a_paste_lands(
    request, monkeypatch, how
):
    """Review round 2, G6-VR2-03: after Back from a skip, focus sat on the
    provider list, so a terminal paste was dropped silently and Next skipped
    again -- the literal AC#2 flow 'Skip -> Back -> paste key -> Next' failed
    unless the user clicked into the field first."""
    from textual import events

    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        if how == "mouse":
            await pilot.click("#setup-provider-skip")
        else:
            provider_list(provider).focus()
            await pilot.pause()
            await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        await pilot.press("ctrl+b")
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        await pilot.pause(0.2)
        assert app.focused is key_input(provider)
        app.post_message(events.Paste("sk-proj-" + "p" * 40))
        await pilot.pause()
        assert key_input(provider).value == "sk-proj-" + "p" * 40
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _container, model = current_step(wizard)
        assert static_text(model, "#setup-model-provider-line") == "Models for OpenAI."


@pytest.mark.asyncio
@private_profile_test
async def test_a_paste_on_the_list_goes_to_the_key_field_never_the_filter(
    request, monkeypatch
):
    """Review round 2, G6-VR2-03: a key pasted (or typed) while the list had
    focus went to the filter -- in plain text, unmasked."""
    from textual import events

    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        provider_list(provider).focus()
        await pilot.pause()
        app.post_message(events.Paste("sk-proj-" + "q" * 40))
        await pilot.pause(0.2)
        assert provider.query_one("#setup-provider-filter", Input).value == ""
        assert key_input(provider).value == "sk-proj-" + "q" * 40
        assert app.focused is key_input(provider)


def _key_field_on_screen(provider) -> bool:
    """The key field can be seen: its section shown and expanded, laid out."""
    auth = provider.query_one("#setup-provider-auth-toggle")
    field = key_input(provider)
    return bool(
        auth.display and not auth.has_class("hidden") and not auth.collapsed
        and field.display and field.region.height > 0
    )


def _resumed_with(provider_key: str):
    """A run resumed (after a kill) onto Provider with ``provider_key`` saved."""
    from Tests.Wizards.provider_step_harness import app_instance
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        SETUP_DRAFT_VERSION,
        STEP_WELCOME,
        TRACK_QUICK,
        SetupDraft,
    )

    draft = SetupDraft(
        version=SETUP_DRAFT_VERSION,
        track=TRACK_QUICK,
        active_step_id=STEP_PROVIDER,
        values={
            STEP_WELCOME: {"track": TRACK_QUICK},
            STEP_PROVIDER: {"provider_key": provider_key, "provider_value": provider_key},
        },
    )
    return FirstRunSetupWizard(app_instance(), resume_draft=draft)


@pytest.mark.asyncio
@private_profile_test
async def test_next_on_a_resumed_keyed_provider_asks_for_the_key_instead_of_skipping(
    request, monkeypatch
):
    """Review round 2, G6-VR2-05: after a kill at Model and 'Resume', Provider
    restored Anthropic with its key section hidden (keys aren't kept) and Next
    silently skipped it -- the empty-key skip, for a key field the user never
    saw. The first Next shows the key section and asks for the key."""
    _without_openai_env(monkeypatch)
    wizard = _resumed_with("anthropic")
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        assert provider.selected_provider_key == "anthropic"
        await pilot.press("ctrl+n")
        await pilot.pause(0.3)
        assert step_id(wizard) == STEP_PROVIDER, "a key field never shown was skipped"
        refusal = pinned_error(wizard)
        assert "Anthropic API key" in refusal and "aren't kept" in refusal
        assert "Skip" in refusal
        assert _key_field_on_screen(provider)
        assert provider.query_one("#setup-provider-skip", Button).display
        assert app.focused is key_input(provider)
        await pilot.press("ctrl+n")  # now it is the visible empty-key skip
        await wait_for_step(pilot, wizard, STEP_MODEL)


@pytest.mark.asyncio
@private_profile_test
async def test_enter_on_a_resumed_keyed_row_shows_its_key_field(request, monkeypatch):
    """Review round 2, G6-VR2-05 / AC#26: a click or Enter on the restored row
    was a no-op (the provider was already 'selected'), so the key section
    never appeared."""
    _without_openai_env(monkeypatch)
    wizard = _resumed_with("anthropic")
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        choices = provider_list(provider)
        choices.focus()
        await pilot.pause()
        assert choices.highlighted_option.id == "provider-anthropic"
        assert not _key_field_on_screen(provider)  # provider-04 (TASK-34100.10)
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert _key_field_on_screen(provider)
        assert app.focused is key_input(provider)


@pytest.mark.asyncio
@private_profile_test
async def test_skip_skips_while_the_field_holds_a_key_that_cannot_be_sent(
    request, monkeypatch
):
    """Review v3r1, R1-02: with a smart dash in the typed key, readiness reads
    the field as no key, so 'Skip — connect later' shows -- and clicking it
    left the step on Provider with the non-ASCII refusal. An explicit skip
    always advances; the unusable key is never staged."""
    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        key_input(provider).value = "sk-proj-abc–ddd0001"
        await pilot.pause()
        skip = provider.query_one("#setup-provider-skip", Button)
        assert skip.display, "the unusable key reads as no key, so Skip shows"
        await pilot.click("#setup-provider-skip")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        assert pinned_error(wizard) == ""
        assert wizard.query_one("SetupWizardContainer").staged_provider_draft is None


@pytest.mark.asyncio
@private_profile_test
async def test_a_skip_the_wizard_never_ran_does_not_turn_a_later_refusal_into_a_skip(
    request, monkeypatch
):
    """Review v3r1, R1-07: the one-shot flag was consumed only by commit(), so
    a Skip pressed while an advance was already running survived to the next
    Next -- which then skipped past a missing Azure resource URL instead of
    asking for it."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "azure")
        await _type_into(pilot, key_input(provider), "azure-key-0001")
        container._advancing = True  # an advance is already in flight
        provider.skip_without_key()
        container.action_next()  # fenced: nothing commits
        container._advancing = False
        await pilot.pause()
        await pilot.press("ctrl+n")
        await wait_until(pilot, lambda: pinned_error(wizard) != "")
        assert step_id(wizard) == STEP_PROVIDER, "a stale skip skipped the URL"
        assert "URL" in pinned_error(wizard)


@pytest.mark.asyncio
@private_profile_test
async def test_a_provider_change_or_a_key_edit_drops_a_pending_skip(request, monkeypatch):
    """Review v3r1, R1-07: neither a provider change nor a key edit cleared the
    flag, so a skip asked for one form could fire on another."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    _without_openai_env(monkeypatch)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _container, provider = current_step(wizard)
        await pick_provider(pilot, provider, "openai")
        provider._skip_requested = True
        await pick_provider(pilot, provider, "azure")
        assert provider._skip_requested is False
        provider._skip_requested = True
        await _type_into(pilot, key_input(provider), "azure-key-0001")
        assert provider._skip_requested is False
