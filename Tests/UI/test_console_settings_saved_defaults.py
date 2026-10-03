"""Chat settings names itself, its scope and its actions (TASK-33006.5).

The title names the chat and counts its unsaved edits, the scope line says
where defaults live, and the footer offers Use saved defaults, Save as model
default, Default for new chats (Ctrl+N) and Apply to this chat (Ctrl+Enter),
each key working. Use saved defaults stages, for the draft's own
provider·model, what a new chat on that pair resolves, through the
controller's rebaser, and applies nothing. Every test mounts the real modal
under the production stylesheets and drives it with real keys.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Select, Static
from textual.widgets._select import SelectOverlay

from Tests.UI.test_console_settings_core_first import (
    CoreFirstHarness,
    FULL_SCREEN_SIZES,
    _painted,
    _settings,
)
from Tests.UI.test_console_settings_model_change import pick_modal, real_rebase, settle
from tldw_chatbook.Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsAction,
)
from tldw_chatbook.Widgets.Console.console_settings_field_row import field_control_id
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    CONSOLE_SETTINGS_MODEL_SCOPE_COPY,
    ConsoleSettingsModal,
)
from tldw_chatbook.Widgets.Console.console_settings_saved_defaults import (
    APPLY_LABEL,
    NEW_CHAT_DEFAULT_LABEL,
    SAVE_MODEL_DEFAULT_LABEL,
    SAVED_DEFAULTS_MATCH_LABEL,
    USE_SAVED_DEFAULTS_ID,
    USE_SAVED_DEFAULTS_LABEL,
)

# Census-gated (scripts/ui_pr_gate_census.txt): Tests/UI/conftest.py imports
# tldw_chatbook.app per test, which fails closed with
# RecoveryRequired("raw_source_selection_changed") under the per-test sandbox.
pytestmark = pytest.mark.bootstrap_profile

#: The harness's saved chain for llama_cpp·model-a: the model profile's
#: Temperature 0.4 and chat_defaults' Max tokens 2048 (``_settings()``).
EDITED = {"temperature": 0.9, "max_tokens": 512, "top_k": 7}


async def _open(pilot, app, modal, results=None) -> None:
    await app.push_screen(modal, callback=None if results is None else results.append)
    await settle(pilot, app)


def _value(modal, name: str) -> str:
    return modal.query_one(f"#{field_control_id(name)}", Input).value


def _source(modal, name: str) -> str:
    return str(modal.query_one(f"#{field_control_id(name)}-source", Static).content)


async def _type(pilot, modal, name: str, text: str) -> None:
    field = modal.query_one(f"#{field_control_id(name)}", Input)
    field.focus()
    await pilot.pause()
    await pilot.press("end", *["backspace"] * len(field.value), *text)
    await pilot.pause()


@pytest.mark.asyncio
async def test_title_names_the_chat_and_counts_its_unsaved_edits() -> None:
    """AC#1: 'Chat settings · <chat title>' plus the unsaved count, and the
    scope line that says where defaults live."""
    app = CoreFirstHarness()
    modal = pick_modal(app, _settings(), chat_title="Refactor plan")
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        title = modal.query_one("#console-settings-modal-title", Static)
        painted = _painted(app.screen)
        assert "Chat settings · Refactor plan" in painted[title.region.y]
        assert "unsaved" not in painted[title.region.y]
        assert CONSOLE_SETTINGS_MODEL_SCOPE_COPY == (
            "Applies to this chat only · saved with the conversation · "
            "defaults live in Settings ▸ Providers & Models (F4)"
        )
        scope = modal.query_one("#console-settings-scope", Static)
        assert CONSOLE_SETTINGS_MODEL_SCOPE_COPY in painted[scope.region.y]

        await _type(pilot, modal, "temperature", "0.9")
        await settle(pilot, app)
        assert str(title.content) == "Chat settings · Refactor plan · 1 unsaved edit"
        await _type(pilot, modal, "max_tokens", "512")
        await settle(pilot, app)
        assert str(title.content) == "Chat settings · Refactor plan · 2 unsaved edits"


@pytest.mark.parametrize("size", FULL_SCREEN_SIZES)
@pytest.mark.asyncio
async def test_footer_offers_the_four_actions_in_order(size) -> None:
    """AC#2: the Esc hint, then Use saved defaults, Save as model default,
    Default for new chats (Ctrl+N) and Apply to this chat (Ctrl+Enter), on
    one painted row; Cancel is the Context view's, not the Model view's.

    Plan ruling R14: the Context view keeps its own scope copy and offers
    no defaults action (its defaults line:
    ``test_defaults_line_follows_the_view_switch``)."""
    app = CoreFirstHarness()
    modal = pick_modal(app, _settings(**EDITED))
    async with app.run_test(size=size) as pilot:
        await _open(pilot, app, modal)
        default_scope = modal.query_one("#console-settings-default-scope", Static)
        assert default_scope.display and str(default_scope.content)
        apply = modal.query_one("#console-settings-save", Button)
        row = _painted(app.screen)[apply.region.y]
        labels = (
            "Esc close",
            USE_SAVED_DEFAULTS_LABEL,
            SAVE_MODEL_DEFAULT_LABEL,
            NEW_CHAT_DEFAULT_LABEL,
            APPLY_LABEL,
        )
        positions = [row.find(label) for label in labels]
        assert -1 not in positions and positions == sorted(positions), row
        assert "Cancel" not in row
        assert not modal.query_one("#console-settings-cancel", Button).display

        modal.query_one("#console-settings-view-context", Button).press()
        await settle(pilot, app)
        assert modal.query_one("#console-settings-cancel", Button).display
        for action in (
            USE_SAVED_DEFAULTS_ID,
            "console-settings-save-default",
            "console-settings-make-default",
        ):
            assert not modal.query_one(f"#{action}", Button).display, action
        assert str(modal.query_one("#console-settings-scope", Static).content) == (
            "Use: this conversation only. Defaults: F4 Settings > Console Behavior."
        )


@pytest.mark.parametrize("edited", [True, False], ids=["edited", "unchanged"])
@pytest.mark.asyncio
async def test_defaults_line_follows_the_view_switch(edited) -> None:
    """TASK-33006.6 AC#2 (plan R14): the defaults-scope line shows only
    beside a defaults action. The Context view offers none, so switching to
    it hides the line; switching back re-derives Save as model default and
    its line from the draft, so an unchanged chat shows neither. Rewritten
    on purpose from TASK-33006.5's strict xfail
    ``test_context_view_paints_no_defaults_line``."""
    app = CoreFirstHarness()
    settings = _settings(**EDITED) if edited else _settings()
    if not edited:  # save the chat's own values as every default it compares
        app.app_config["api_settings"]["llama_cpp"]["model"] = "model-a"
        app.app_config["chat_defaults"].update(
            provider="llama_cpp", model="model-a", streaming=settings.streaming
        )
        app.app_config["console"] = {
            "provider_defaults": {
                "llama_cpp": {
                    name: getattr(settings, name)
                    for name in FULL_MODEL_DEFAULT_FIELDS - {"streaming"}
                    if getattr(settings, name) is not None
                }
            }
        }
    modal = pick_modal(app, settings)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        default_scope = modal.query_one("#console-settings-default-scope", Static)
        save_default = modal.query_one("#console-settings-save-default", Button)
        line = "Used by future conversations for llama.cpp."
        assert save_default.display is default_scope.display is edited
        modal.query_one("#console-settings-view-context", Button).press()
        await settle(pilot, app)
        assert not save_default.display and not default_scope.display
        assert not any(line in row for row in _painted(app.screen))
        modal.query_one("#console-settings-view-model", Button).press()
        await settle(pilot, app)
        assert save_default.display is default_scope.display is edited
        assert any(line in row for row in _painted(app.screen)) is edited


@pytest.mark.parametrize(
    ("key", "action", "mask"),
    (
        ("ctrl+n", ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT, FULL_MODEL_DEFAULT_FIELDS),
        ("ctrl+enter", ConsoleSettingsAction.APPLY_TO_CHAT, frozenset()),
    ),
)
@pytest.mark.asyncio
async def test_every_advertised_key_works_from_a_field(key, action, mask) -> None:
    """AC#2/#3/#7: Ctrl+N and Ctrl+Enter submit their buttons' actions from
    a focused field; Apply carries no default mask (it writes no
    configuration), and the modal binds no ADR-031 rule 2 key."""
    banned = {f"ctrl+{letter}" for letter in "cvxsdzarw"}
    bound = {
        key.strip()
        for binding in ConsoleSettingsModal.BINDINGS
        for key in (binding[0] if isinstance(binding, tuple) else binding.key).split(",")
    }
    assert not bound & banned
    assert "ctrl+n" in bound

    app = CoreFirstHarness()
    results: list = []
    modal = pick_modal(app, _settings(**EDITED))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal, results)
        modal.query_one("#console-settings-temperature", Input).focus()
        await pilot.press(key)
        await settle(pilot, app)
        assert len(results) == 1, results
        submission = results[0].submission
        assert submission.action is action
        assert submission.default_field_mask == mask


@pytest.mark.asyncio
async def test_ctrl_n_works_inside_an_open_select_dropdown() -> None:
    """AC#2: the open dropdown (Textual's SelectOverlay) binds no Ctrl+N and
    types only printable keys, so Ctrl+N reaches the modal. The highlighted
    option is not a choice, so the submission keeps the committed value."""
    app = CoreFirstHarness()
    results: list = []
    modal = pick_modal(app, _settings(**EDITED))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal, results)
        streaming = modal.query_one("#console-settings-streaming", Select)
        committed = streaming.value
        streaming.focus()
        await pilot.press("enter", "down")
        await pilot.pause()
        assert streaming.expanded and isinstance(app.focused, SelectOverlay)
        await pilot.press("ctrl+n")
        await settle(pilot, app)
        assert len(results) == 1, results
        submission = results[0].submission
        assert submission.action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT
        assert streaming.value == committed


@pytest.mark.asyncio
async def test_save_as_model_default_keeps_the_full_field_mask() -> None:
    """AC#7 (ADR-095:74-77): Save as model default patches every field."""
    app = CoreFirstHarness()
    results: list = []
    modal = pick_modal(app, _settings(**EDITED))
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal, results)
        save = modal.query_one("#console-settings-save-default", Button)
        assert str(save.label) == SAVE_MODEL_DEFAULT_LABEL
        await pilot.click(save)
        await settle(pilot, app)
        submission = results[0].submission
        assert submission.action is ConsoleSettingsAction.SAVE_MODEL_DEFAULT
        assert submission.default_field_mask == FULL_MODEL_DEFAULT_FIELDS


@pytest.mark.asyncio
async def test_use_saved_defaults_stages_the_new_chat_values_and_keeps_the_pair() -> None:
    """AC#4: an unapplied edit is replaced, each field that now differs from
    the conversation reads 'edited *', the pair stays, the controller's
    rebaser is called for that same pair, and nothing is applied."""
    app = CoreFirstHarness()
    results: list = []
    rebases: list = []

    def recording_rebase(state, **kwargs):
        rebases.append((state.settings.provider, state.settings.model, kwargs))
        return real_rebase(state, **kwargs)

    modal = pick_modal(app, _settings(**EDITED), draft_rebaser=recording_rebase)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal, results)
        await _type(pilot, modal, "top_p", "0.5")  # an unapplied edit
        await settle(pilot, app)
        assert _source(modal, "top_p") == "edited *"

        await pilot.click(f"#{USE_SAVED_DEFAULTS_ID}")
        await settle(pilot, app)
        assert [(provider, model) for provider, model, _kwargs in rebases] == [
            ("llama_cpp", "model-a")
        ]
        assert rebases[0][2]["provider"] == "llama_cpp"
        assert rebases[0][2]["model"] == "model-a"
        assert (_value(modal, "temperature"), _value(modal, "max_tokens")) == (
            "0.4",
            "2048",
        )
        assert (_value(modal, "top_k"), _value(modal, "top_p")) == ("", "0.95")
        for name in ("temperature", "max_tokens", "top_k"):
            assert _source(modal, name) == "edited *", name
        assert _source(modal, "top_p") != "edited *"  # back to the chat's value
        assert _source(modal, "streaming") != "edited *"
        assert modal._draft.settings.provider == "llama_cpp"
        assert modal._draft.settings.model == "model-a"
        assert str(
            modal.query_one("#console-settings-model-source", Static).content
        ) == "this chat"
        assert results == [] and app.screen is modal  # nothing applied
        assert str(modal.query_one("#console-settings-modal-title", Static).content) == (
            "Chat settings · 3 unsaved edits"
        )


@pytest.mark.asyncio
async def test_use_saved_defaults_is_disabled_with_its_reason_while_the_draft_matches() -> None:
    """AC#6: matching the saved defaults disables the button, and its
    painted label says why; an edit enables it, and using it disables it
    again with focus moved on to Apply."""
    app = CoreFirstHarness()
    modal = pick_modal(app, _settings())  # the chat already holds the defaults
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(pilot, app, modal)
        button = modal.query_one(f"#{USE_SAVED_DEFAULTS_ID}", Button)
        assert button.disabled
        assert SAVED_DEFAULTS_MATCH_LABEL in _painted(app.screen)[button.region.y]

        # A cleared Temperature is an edit too (found live): the defaults
        # would put 0.4 back, though the draft falls back to it on Apply.
        await _type(pilot, modal, "temperature", "")
        await settle(pilot, app)
        assert not button.disabled
        await _type(pilot, modal, "temperature", "0.9")
        await settle(pilot, app)
        assert not button.disabled
        assert str(button.label) == USE_SAVED_DEFAULTS_LABEL

        await pilot.click(button)
        await settle(pilot, app)
        assert button.disabled
        assert str(button.label) == SAVED_DEFAULTS_MATCH_LABEL
        assert _value(modal, "temperature") == "0.4"
        assert app.focused is modal.query_one("#console-settings-save", Button)
