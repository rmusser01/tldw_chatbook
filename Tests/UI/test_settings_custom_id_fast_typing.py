"""Settings ▸ Providers & Models ▸ Model: keys typed faster than the screen
hears of them do not throw the field out of Custom ID mode (review of
#3057, R3 #3).

Each Custom ID keystroke reaches the screen as a message. When a second key
landed first (key repeat, an IME, a paste sent as keys), the screen compared
its adapter (one key behind) with the live field, took the difference for a
stale picker and pushed the older text back: the field left Custom ID mode,
was rewritten and selected.
"""

from __future__ import annotations

import pytest
from textual import events
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_default_model_picker import (
    _app,
    _open_providers,
    _settle,
    _widgets,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness

_SIZE = (211, 44)


@pytest.mark.asyncio
@private_profile_test
async def test_two_queued_keys_keep_the_field_in_custom_id_mode(request):
    host = _SettingsCssHarness(
        _app("openai", "gpt-4o", catalog=("gpt-4o",)), "settings"
    )

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, _results, adapter = _widgets(screen)
        field.focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert host.focused is picker.query_one("#model-search-picker-custom", Button)
        await pilot.press("enter")
        await pilot.pause()
        assert picker.custom_mode and host.focused is field
        await pilot.press(*"my-model-")
        await _settle(host, pilot)
        assert adapter.value == "my-model-"

        # Both keys are in the field's queue before the screen hears of "a".
        field.post_message(events.Key("a", "a"))
        field.post_message(events.Key("b", "b"))
        await _settle(host, pilot)
        await pilot.pause()

        assert picker.custom_mode
        assert field.value == "my-model-ab"
        assert field.selection.is_empty
        assert adapter.value == picker.value == "my-model-ab"
        assert screen._provider_setting_values_mapping()["model"] == "my-model-ab"


@pytest.mark.asyncio
@private_profile_test
async def test_a_late_mount_echo_keeps_the_handoffs_model_and_typing_still_stages(
    request, monkeypatch
):
    """TASK-38975: the vLLM default handoff stages provider, model and endpoint
    and writes the model and endpoint fields quietly. If a field's mount echo
    (a ``Changed`` carrying the saved value) were handled after it, the change
    handler took it for an edit and staged the saved value again: the draft
    would save vLLM with the old model or endpoint while the fields showed the
    new ones. The model, endpoint and provider echoes are held here and
    delivered after the handoff (review r11g-rev1 I1 added the endpoint)."""
    from textual.widgets import Select

    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _console_app,
        _install_ready_vllm_target,
        _SettingsFlowHarness,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmDefaultIntent

    held: dict[str, object] = {}
    held_ids = {
        "settings-model-value": Input.Changed,
        "settings-provider-endpoint-value": Input.Changed,
        "settings-provider-value": Select.Changed,
    }
    real_posts = {Input: Input.post_message, Select: Select.post_message}

    def holding(widget_type):
        def post(self, message):
            kind = held_ids.get(self.id)
            if kind is not None and self.id not in held and isinstance(message, kind):
                held[self.id] = message
                return True
            return real_posts[widget_type](self, message)

        return post

    monkeypatch.setattr(Input, "post_message", holding(Input))
    monkeypatch.setattr(Select, "post_message", holding(Select))
    app = _console_app()
    _, target = _install_ready_vllm_target(app)
    app.pending_handoffs.stage(
        HandoffChannel.VLLM_DEFAULT, VllmDefaultIntent.from_target(target)
    )
    host = _SettingsFlowHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#settings-provider-save-result")
        await _settle(host, pilot)
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_DEFAULT)
        picker, field, _results, adapter = _widgets(screen)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        assert held["settings-model-value"].value == "model-a"  # the saved model
        assert held["settings-provider-endpoint-value"].value == (
            "http://127.0.0.1:9099"  # the saved endpoint
        )
        assert adapter.value == "chatbook-vllm"
        assert endpoint.value == target.api_url

        for widget_id, message in held.items():
            widget = screen.query_one(f"#{widget_id}")
            real_posts[Select if isinstance(widget, Select) else Input](widget, message)
        await _settle(host, pilot)
        await pilot.pause()

        draft = screen._provider_draft()
        assert draft is not None
        assert (
            draft.values["provider"],
            draft.values["model"],
            draft.values["endpoint"],
        ) == ("vllm", "chatbook-vllm", target.api_url)
        assert adapter.value == "chatbook-vllm"
        assert endpoint.value == target.api_url

        # A later edit still stages: a Custom ID typed in the field.
        field.focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert host.focused is picker.query_one("#model-search-picker-custom", Button)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press(*"typed-model")
        await _settle(host, pilot)
        assert adapter.value == "typed-model"
        assert screen._provider_draft().values["model"] == "typed-model"
