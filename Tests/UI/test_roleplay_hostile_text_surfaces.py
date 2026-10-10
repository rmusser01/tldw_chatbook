"""Every other Roleplay surface that shows untrusted text paints it literally (TASK-34400).

``test_roleplay_hostile_names.py`` drives whole Roleplay flows. This module
pins the remaining sinks a static sweep of the Roleplay code found, one widget
at a time: each gets hostile text through its own public seam and must paint
it as typed, raise nothing and carry no ``@click`` meta. A ``MarkupError``
raised while drawing ends ``run_test`` and fails the test; in the real app it
bypasses the TASK-32533 keep-alive and exits the whole app.

The notification-only sinks (CCP helpers, Buddy coordinators) are pinned at
the call: they must pass ``markup=False``, because a toast parses markup when
it renders.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
from pydantic import BaseModel
from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Select, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.roleplay_frame_harness import click_meta_cells, painted_rows
from Tests.UI.test_roleplay_hostile_names import HOSTILE_NAMES
from tldw_chatbook.UI.CCP_Modules.ccp_character_handler import CCPCharacterHandler
from tldw_chatbook.UI.CCP_Modules.ccp_loading_indicators import (
    LoadingManager,
    with_loading,
)
from tldw_chatbook.UI.CCP_Modules.ccp_persona_handler import CCPPersonaHandler
from tldw_chatbook.UI.CCP_Modules.ccp_validation_decorators import (
    validate_file_import,
    validate_input,
)
from tldw_chatbook.UI.Navigation.buddy_management import BuddyManagementCoordinator
from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
    PersonasPreviewController,
)
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from tldw_chatbook.Widgets.Persona_Widgets.buddy_character_review import (
    BuddyCharacterReviewDialog,
)
from tldw_chatbook.Widgets.Persona_Widgets.buddy_workspace_modal import (
    BuddyWorkspaceModal,
)
from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_tts_widget import (
    CharacterTTSPresentationState,
    CharacterTTSProfileOption,
    PersonasCharacterTTSWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import (
    PersonasInspectorPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    VisualIdentityAssetMetadata,
    VisualIdentityPackMetadata,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
    PersonasPreviewPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_visual_identity_pack_widget import (
    PersonasVisualIdentityPackWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.petdex_import_review import (
    PetdexImportReviewDialog,
)

#: The flow test's names plus two backslash cases. ``a\[/]b`` defeats an
#: escaper written for Textual's parser once Rich's parser reads the result;
#: ``\[x^2\]`` is an ordinary LaTeX reply from a model.
HOSTILE_TEXT = (*HOSTILE_NAMES, "a\\[/]b", "\\[x^2\\]")
SIZE = (200, 60)


class _Host(ConsolidatedCSSApp):
    """Mounts one widget under the app stylesheets."""

    def __init__(self, widget) -> None:
        super().__init__()
        self._widget = widget

    def compose(self) -> ComposeResult:
        yield self._widget


def _painted(app) -> str:
    return "\n".join(painted_rows(app.screen))


async def _settle(pilot) -> None:
    await pilot.pause()
    await pilot.pause()


async def _render_like_a_tooltip(app, pilot, tooltip) -> str:
    """Paint ``tooltip`` the way Textual's tooltip timer does: ``Static.update``.

    ``Screen._handle_tooltip_timer`` calls ``tooltip.update(widget.tooltip)``
    on a markup-on ``Static``, so a ``str`` tooltip is parsed as markup.
    """
    probe = Static(id="tooltip-probe")
    await app.mount(probe)
    probe.update(tooltip)
    await _settle(pilot)
    text = str(probe.render())
    await probe.remove()
    return text


def _select_label(select: Select) -> str:
    """The text a ``Select`` shows for its current value (it may be clipped
    on screen, so read the label widget, which renders the whole prompt)."""
    return str(select.query_one("#label", Static).render())


async def _open_overlay(pilot, select: Select) -> None:
    """Open the dropdown: its option list parses a ``str`` prompt as markup."""
    select.focus()
    await pilot.press("enter")
    await _settle(pilot)
    assert select.expanded
    assert click_meta_cells(pilot.app.screen) == []


def _assert_literal(app, *texts: str) -> None:
    painted = _painted(app)
    for text in texts:
        assert text in painted, painted
    assert click_meta_cells(app.screen) == []


@pytest.mark.parametrize("text", HOSTILE_TEXT)
async def test_the_preview_transcript_status_readout_and_greetings(text):
    """Speaker label, an *action* span, a model reply, the status and provider
    lines and the greeting dropdown (card text)."""
    pane = PersonasPreviewPane(id="personas-preview-pane")
    app = _Host(pane)
    async with app.run_test(size=SIZE) as pilot:
        pane.expand()
        pane.set_speakers(character=text)
        await pane.seed_greeting(f"Hi *{text}* there")
        pane.append_reply(text)
        pane.set_status(f"Running via Console default: {text}")
        pane.set_provider_readout(f"Provider: {text} / {text}")
        pane.set_greetings([text, "Second"], 0)
        await _settle(pilot)
        _assert_literal(
            app,
            f"{text}: {text}",
            f"Running via Console default: {text}",
            f"Provider: {text} / {text}",
        )
        if "*" not in text:
            assert f"{text}: Hi {text} there" in _painted(app)
        select = pane.query_one("#personas-preview-greeting-select", Select)
        prompts = [str(prompt) for prompt, _value in select._options]
        assert f"Greeting 1 (default): {text}" in prompts
        await _open_overlay(pilot, select)


@pytest.mark.parametrize("text", HOSTILE_TEXT)
async def test_the_inspector_readiness_line_and_chat_now_tooltip(text):
    """Provider block reasons quote a custom endpoint's display name."""
    inspector = PersonasInspectorPane(id="personas-inspector-pane")
    app = _Host(inspector)
    async with app.run_test(size=SIZE) as pilot:
        inspector.show_selection(name="Sam", kind="character", entity_id="1")
        inspector.set_console_actions_enabled(True, provider_block_reason=text)
        await _settle(pilot)
        _assert_literal(app, f"Chat now blocked: {text}")
        inspector.show_validation((f"name: {text}",))
        summary = inspector.query_one("#personas-validation-summary", Static)
        assert str(summary.render()) == f"Validation errors:\nname: {text}"
        start = inspector.query_one("#personas-start-chat")
        assert await _render_like_a_tooltip(app, pilot, start.tooltip) == (
            f"Chat now blocked: {text}"
        )
        inspector.set_console_actions_enabled(False, reason=text)
        await _settle(pilot)
        _assert_literal(app, f"Chat now and Send to Console draft blocked: {text}")
        attach = inspector.query_one("#personas-attach-to-console")
        assert await _render_like_a_tooltip(app, pilot, attach.tooltip) == (
            f"Chat now and Send to Console draft blocked: {text}"
        )


@pytest.mark.parametrize("text", HOSTILE_TEXT)
async def test_the_character_editor_greetings_style_and_validation(text):
    """Alternate greetings (an imported card), a style template name and a
    validation error that echoes card text."""
    editor = PersonasCharacterEditorWidget()
    app = _Host(editor)
    async with app.run_test(size=SIZE) as pilot:
        editor.load_character({"name": "A", "alternate_greetings": [text]})
        editor.set_style_readout(f"Style: {text}")
        editor.show_validation((f"character_book: entry {text}",))
        await _settle(pilot)
        table = editor.query_one("#personas-char-editor-greetings-table")
        table.scroll_visible()
        editor.query_one("#personas-char-editor-style-readout").scroll_visible()
        await _settle(pilot)
        _assert_literal(app, f"Style: {text}")
        # A str cell is parsed as markup when the row renders; the cell must
        # already be literal text.
        cell = table.get_row_at(0)[0]
        assert isinstance(cell, Text) and cell.plain == text
        validation = editor.query_one("#personas-char-editor-validation")
        assert f"character_book: entry {text}" in str(validation.render())


@pytest.mark.parametrize("text", HOSTILE_TEXT)
async def test_the_persona_portrait_choices_name_characters_literally(text):
    editor = PersonaProfileEditorWidget()
    app = _Host(editor)
    async with app.run_test(size=SIZE) as pilot:
        editor.begin_actor_pack_creation(((text, 1), ("Plain", 2)))
        await _settle(pilot)
        select = editor.query_one("#personas-editor-character-portrait", Select)
        assert _select_label(select) == text
        await _open_overlay(pilot, select)


@pytest.mark.parametrize("text", HOSTILE_TEXT)
async def test_the_tts_profile_choices_name_profiles_literally(text):
    widget = PersonasCharacterTTSWidget(context="card")
    app = _Host(widget)
    profile_id = uuid.uuid4()
    async with app.run_test(size=SIZE) as pilot:
        widget.apply_state(
            CharacterTTSPresentationState(
                profiles=(
                    CharacterTTSProfileOption(
                        profile_id=profile_id,
                        display_name=text,
                        availability="available",
                    ),
                ),
                selected_profile_id=profile_id,
                status="Assigned.",
                controls_enabled=True,
            )
        )
        await _settle(pilot)
        select = widget.query_one(Select)
        assert _select_label(select).startswith(f"{text} · ")
        await _open_overlay(pilot, select)


@pytest.mark.parametrize("text", HOSTILE_TEXT)
async def test_the_visual_identity_pack_title_and_asset_list(text):
    """An imported actor pack names itself and its assets."""
    asset = VisualIdentityAssetMetadata(
        asset_id=1,
        expression_key="neutral",
        original_label=text,
        display_label=text,
        content_type="image/png",
        is_animated=False,
    )
    pack = VisualIdentityPackMetadata(
        binding_id=1,
        pack_id=1,
        pack_version_id=1,
        title=text,
        source_kind="imported",
        default_expression_key="neutral",
        assets=(asset,),
    )
    widget = PersonasVisualIdentityPackWidget(pack)
    app = _Host(widget)
    async with app.run_test(size=SIZE) as pilot:
        await _settle(pilot)
        widget.apply_filter("")
        await _settle(pilot)
        _assert_literal(app, text)
        title = widget.query_one("#personas-visual-identity-title")
        assert str(title.render()) == text
        assert painted_rows(app.screen) and _painted(app).count(text) >= 2


@pytest.mark.parametrize("text", HOSTILE_TEXT)
async def test_the_buddy_review_and_petdex_state_choices(text):
    """Source state names come from an archive or are typed into JSON."""
    snapshot = SimpleNamespace(
        title="Buddy", source_sha256="a" * 64, artwork=None, is_current=lambda: True
    )
    rows = [
        SimpleNamespace(
            source_state=text, expression_key="neutral", fallback=False, frame_count=1
        )
    ]
    buddy = BuddyCharacterReviewDialog(
        snapshot,
        rows,
        db=object(),
        local_service=object(),
        profile_root=None,
        authority_guard=lambda: True,
        config={},
    )
    app = ConsolidatedCSSApp()
    async with app.run_test(size=SIZE) as pilot:
        await app.push_screen(buddy)
        await _settle(pilot)
        portrait = buddy.query_one("#buddy-portrait-state", Select)
        assert _select_label(portrait) == text
        await _open_overlay(pilot, portrait)
        await pilot.press("escape")
        await _settle(pilot)
        app.pop_screen()
        await _settle(pilot)
        petdex = PetdexImportReviewDialog(authority_guard=lambda: True, config={})
        await app.push_screen(petdex)
        await _settle(pilot)
        petdex._set_state_options((SimpleNamespace(name=text),))
        await _settle(pilot)
        preview = petdex.query_one("#petdex-preview-state", Select)
        await _open_overlay(pilot, preview)


@pytest.mark.parametrize("text", HOSTILE_TEXT)
async def test_agent_question_options_on_the_buddy_and_console_card(text):
    """Option labels are model-authored; the card is shared with Console."""
    app = _Host(ChatTaskCards(id="chat-task-cards"))
    payload = {
        "request_id": "r1",
        "session_id": "s1",
        "timeout_seconds": 0.0,
        "deadline_monotonic": None,
        "asked_by": "agent",
        "questions": [
            {
                "question": "Pick one?",
                "header": "Q",
                "multiSelect": multi,
                "options": [{"label": text, "description": text}],
            }
            for multi in (False, True)
        ],
    }
    async with app.run_test(size=SIZE) as pilot:
        app.query_one(ChatTaskCards).sync_state(
            TaskResumeState(pending_question=payload)
        )
        await _settle(pilot)
        _assert_literal(app, f"{text} — {text}")
        assert _painted(app).count(f"{text} — {text}") >= 2


def _hostile_error() -> ValueError:
    return ValueError("Server said [/] for [@click=app.quit]x")


def test_ccp_character_and_persona_handlers_notify_literally():
    for handler_class in (CCPCharacterHandler, CCPPersonaHandler):
        handler = object.__new__(handler_class)
        handler.window = Mock()
        handler._notify(str(_hostile_error()))
        handler.window.notify.assert_called_once_with(
            str(_hostile_error()), severity="warning", markup=False
        )


async def test_ccp_loading_notifications_are_literal():
    window = Mock()
    manager = LoadingManager(window)
    await manager.start_loading("Loading [/]", "op-1", notify=True)
    assert window.notify.call_args.kwargs["markup"] is False

    class _Owner:
        def __init__(self) -> None:
            self.window = Mock()
            self.loading_manager = SimpleNamespace(
                start_loading=AsyncMock(), stop_loading=AsyncMock()
            )

        @with_loading("Saving", "Saved [/]", "Failed")
        async def succeed(self):
            return 1

        @with_loading("Saving", "Saved", "Failed")
        async def fail(self):
            raise _hostile_error()

    owner = _Owner()
    await owner.succeed()
    assert owner.window.notify.call_args.kwargs["markup"] is False
    owner.window.notify.reset_mock()
    with pytest.raises(ValueError):
        await owner.fail()
    assert owner.window.notify.call_args.kwargs["markup"] is False


async def test_ccp_validation_notifications_are_literal():
    class _Named(BaseModel):
        name: int

    class _Owner:
        def __init__(self) -> None:
            self.window = Mock()

        @validate_input(_Named)
        async def save(self, validated):
            return validated

        @validate_file_import
        async def load(self, file_path, file_type=None):
            return file_path

    owner = _Owner()
    assert await owner.save({"name": "[/] not a number"}) is None
    notify = owner.window.app_instance.notify
    assert notify.call_args.args[0].startswith("Validation Error:")
    assert notify.call_args.kwargs["markup"] is False
    notify.reset_mock()
    assert await owner.load("/no/such/dir[/]/card.json") is None
    assert notify.call_args.args[0].startswith("Invalid file:")
    assert notify.call_args.kwargs["markup"] is False


async def test_buddy_settings_and_workspace_errors_notify_literally():
    coordinator = object.__new__(BuddyManagementCoordinator)
    coordinator.app = Mock()
    coordinator.apply_choice = AsyncMock(side_effect=_hostile_error())
    await coordinator._apply_and_report(object())
    coordinator.app.notify.assert_called_once_with(
        str(_hostile_error()), severity="error", markup=False
    )

    modal = object.__new__(BuddyWorkspaceModal)
    modal._fresh = True
    modal._is_mounted = False
    modal._acknowledge = Mock(side_effect=_hostile_error())
    app = Mock()
    with patch.object(BuddyWorkspaceModal, "app", new_callable=PropertyMock) as prop:
        prop.return_value = app
        await modal._mark_seen(object())
    assert app.notify.call_args.args[0] == str(_hostile_error())
    assert app.notify.call_args.kwargs["markup"] is False


@pytest.mark.parametrize("text", HOSTILE_TEXT)
async def test_roleplay_file_picker_titles_paint_untrusted_text_literally(
    text, monkeypatch
):
    """The shared picker paints ``title`` as a border title, which parses
    markup; Roleplay builds three titles from user or imported text."""
    import tldw_chatbook.Widgets.enhanced_file_picker as picker_module

    titles: list[str] = []

    class _RecordingPicker:
        def __init__(self, *args, title: str = "", **kwargs) -> None:
            titles.append(title)

    monkeypatch.setattr(picker_module, "EnhancedFileOpen", _RecordingPicker)
    screen = object.__new__(PersonasScreen)
    screen._local_character_actions_allowed = lambda: True
    host_app = Mock()
    host_app.push_screen_wait = AsyncMock(return_value=None)
    asset = VisualIdentityAssetMetadata(
        asset_id=1,
        expression_key="neutral",
        original_label=text,
        display_label=text,
        content_type="image/png",
        is_animated=False,
    )
    with patch.object(PersonasScreen, "app", new_callable=PropertyMock) as prop:
        prop.return_value = host_app
        await screen._visual_identity_replace_dialog(asset)
        await screen._persona_visual_replace_dialog(text)
        await screen._expression_upload_dialog_worker(1, text)
    assert len(titles) == 3

    framed = [Static("x") for _ in titles]
    for widget, title in zip(framed, titles):
        widget.styles.border = ("round", "white")
        widget.border_title = title
    app = ConsolidatedCSSApp()
    async with app.run_test(size=SIZE) as pilot:
        await app.mount_all(framed)
        await _settle(pilot)
        _assert_literal(
            app,
            f"Replace {text} reaction",
            f"Replace {text} Persona Visual",
            f"Upload {text.capitalize()} Expression Image",
        )


@pytest.mark.parametrize("text", HOSTILE_TEXT)
def test_the_preview_provider_label_is_plain_text(text):
    """The readout and status lines are literal now, so the label they quote
    must not arrive pre-escaped (a stray backslash would show)."""
    assert PersonasPreviewController._provider_label(text) == text


@pytest.mark.parametrize("text", HOSTILE_TEXT)
def test_the_header_view_keeps_an_unsaved_item_out_of_markup(text):
    """Roleplay frame B1 replaced the subtitle that named an unsaved item: the
    shared header's markup-parsing subtitle names only the kind, the item goes
    to the literal item label as plain text, and a server label reaches the
    markup-parsing status chip only escaped (it parses back to what it shows)."""
    from textual.content import Content

    from tldw_chatbook.UI.Persona_Modules import roleplay_frame_state as fs

    view = fs.build_header_view(
        fs.RoleplayHeaderInputs(
            mode="characters",
            item_name=text,
            unsaved=True,
            runtime_source="server",
            server_label=text,
        ),
        220,
    )
    assert Content.from_markup(view.state.subtitle).plain == "Characters"
    assert view.item == (text, False)
    assert view.unsaved_chip == "Unsaved changes"
    assert Content.from_markup(view.state.status_label).plain == view.status_plain
    assert view.status_plain.startswith(f"Server: {text}")
