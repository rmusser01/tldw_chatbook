# Tests/UI/test_personas_workbench.py
"""Mounted tests for the destination-native Personas workbench."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import inspect
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, call
from uuid import UUID

import pytest
from textual.app import App

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Checkbox, Input, ListView, Select, Static, TextArea

from Tests.UI.background_signals import wait_for_background_signal, wait_for_signal
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
import tldw_chatbook.UI.Persona_Modules.personas_conversations_controller as conversations_controller_module
import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
import tldw_chatbook.UI.Screens.personas_screen as personas_screen_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    CharacterCardImportOutcome,
    CharacterCardTTSInspection,
)
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Constants import (
    CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID,
    LIBRARY_MODE_CONVERSATIONS,
    LIBRARY_NAV_CONTEXT_CONVERSATION_ID,
    LIBRARY_NAV_CONTEXT_MODE,
    TAB_CHAT,
    TAB_LIBRARY,
)
from tldw_chatbook.Persona_Buddy import (
    PersonaBuddyController,
    PersonaBuddyPreferences,
    PersonaBuddySelection,
)
from tldw_chatbook.tldw_api import PersonaProfileCreate
from tldw_chatbook.TTS import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    LoadedCharacterTTSAssignment,
    LoadedTTSProfile,
    PortableProfileAvailabilityObservation,
    PortableProfileImportPlan,
    PortableProfileImportResult,
    ProfileRepositoryError,
    TTSGenerationProfile,
    TTSPlaygroundSelectionPreset,
    TTSProfileAvailability,
    TTSProfileAvailabilitySnapshot,
    TTSProfileDraft,
    TTSProfilePageSnapshot,
)
from tldw_chatbook.TTS.profile_portability import PortableTTSProfile
from tldw_chatbook.TTS.profile_service import TTSProfileDependencyProjection
from tldw_chatbook.tldw_api.character_persona_schemas import (
    LocalPersonaProfileCreate,
    LocalPersonaProfileUpdate,
    PersonaProfileUpdate,
)
from tldw_chatbook.UI.Navigation.shortcut_context import ShortcutAction, ShortcutContext
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.tts_profile_recovery import dependency_recovery_actions
from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy_widget import (
    PersonaBuddyWidget,
)
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PersonaActionRequested,
    PersonaBuddyActionRequested,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import (
    PersonasInspectorPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_tts_widget import (
    CharacterTTSProfileOption,
    CharacterTTSPresentationState,
    PersonasCharacterTTSWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    CharacterTTSActionRequested,
    CharacterImageUploadRequested,
    EditPersonaProfileRequested,
    PersonaProfileSaveRequested,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
    PersonasPreviewPane,
)

pytestmark = pytest.mark.asyncio

CHARACTERS = [
    {
        "id": 1,
        "name": "Detective Sam",
        "description": "Noir detective",
        "first_message": "The name's {{char}}. Who's asking?",
        "alternate_greetings": ["An alternate opener.", "A third opener."],
        "version": 1,
    },
    {
        "id": 2,
        "name": "Lab Assistant",
        "description": "Helpful scientist",
        "version": 1,
    },
]

PROFILE = {
    "id": "p-1",
    "name": "Archivist",
    "description": "Preserve and retrieve",
    "system_prompt": "You are a meticulous archivist.",
}


def _portable_tts_profile(
    profile: TTSGenerationProfile,
    *,
    profile_id: UUID | None = None,
) -> PortableTTSProfile:
    """Project a stored test profile into the sanitized portable value."""

    return PortableTTSProfile(
        profile_id=profile.profile_id if profile_id is None else profile_id,
        draft=TTSProfileDraft(
            display_name=profile.display_name,
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
        ),
    )


@pytest.fixture
def stub_scope_service(mock_app_instance):
    """Replace the MagicMock scope service with explicit AsyncMock methods."""
    service = Mock()
    service.list_persona_profiles = AsyncMock(
        return_value={"items": [dict(PROFILE)], "total": 1}
    )
    service.get_persona_profile = AsyncMock(return_value=dict(PROFILE))
    service.create_persona_profile = AsyncMock(
        return_value={"id": "p-9", "name": "Mentor"}
    )
    service.update_persona_profile = AsyncMock(
        return_value={"id": "p-1", "name": "Archivist 2"}
    )
    service.delete_persona_profile = AsyncMock(
        return_value={"status": "deleted", "persona_id": "p-1"}
    )
    mock_app_instance.character_persona_scope_service = service
    return service


@pytest.fixture
def stub_characters(monkeypatch):
    from Tests.UI.test_personas_dictionaries import patch_character_paging

    monkeypatch.setattr(
        character_handler_module,
        "fetch_all_characters",
        lambda: [dict(c) for c in CHARACTERS],
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: next(
            dict(c) for c in CHARACTERS if str(c["id"]) == str(character_id)
        ),
    )
    # Task 4: the library pages from the DB seam now; mirror fetch_all_characters
    # (read live so tests that swap it mid-run for create/delete stay consistent).
    patch_character_paging(monkeypatch)


class PersonasTestApp(ConsolidatedCSSApp):
    def __init__(self, mock_app_instance):
        super().__init__()
        self._mock = mock_app_instance
        self.character_persona_scope_service = (
            mock_app_instance.character_persona_scope_service
        )

    # Delegating these to a MagicMock would make Textual see phantom dynamic
    # hooks (``compute_*``/``watch_*``/...) on the App and crash at mount.
    _NON_DELEGATED_PREFIXES = (
        "_",
        "watch_",
        "compute_",
        "validate_",
        "action_",
        "key_",
        "on_",
    )

    def __getattr__(self, name):
        if name.startswith(self._NON_DELEGATED_PREFIXES):
            raise AttributeError(name)
        return getattr(self.__dict__["_mock"], name)

    def compose(self):
        # Mirrors the real app: an `AppFooterStatus` composed directly on
        # the app's own default screen (see app.py's `compose()`).
        # Task-264: `PersonasScreen` (via `BaseAppScreen.compose()`) now
        # mounts its OWN `AppFooterStatus` too, and
        # `PersonasScreen._register_footer_shortcuts()` resolves that
        # screen-owned instance via ``self.query_one("AppFooterStatus")`` --
        # so this default-screen widget is only kept around as a foil (the
        # tests below assert the registration does NOT land here).
        yield AppFooterStatus(id="app-footer-status")

    async def _ensure_tts_profile_service(self):
        """Delegate the real app's private lazy loader when a test provides it."""

        loader = self.__dict__["_mock"].__dict__.get("_ensure_tts_profile_service")
        if not callable(loader):
            return None
        result = loader()
        if inspect.isawaitable(result):
            result = await result
        return result

    def on_mount(self) -> None:
        self.push_screen(PersonasScreen(self))


class StyledPersonasTestApp(PersonasTestApp):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )


class PersonaBuddyWorkbenchApp(PersonasTestApp):
    """Workbench harness using the real app-to-screen Buddy reconciliation."""

    reconcile_persona_buddy_view = TldwCli.reconcile_persona_buddy_view
    _persona_buddy_authority = staticmethod(TldwCli._persona_buddy_authority)
    is_persona_buddy_confirmed_unavailable = (
        TldwCli.is_persona_buddy_confirmed_unavailable
    )
    confirm_persona_buddy_unavailable = TldwCli.confirm_persona_buddy_unavailable

    def __init__(self, mock_app_instance) -> None:
        super().__init__(mock_app_instance)
        self._persona_buddy_unavailable_authority = None


def _row_text(item) -> str:
    """Visible text of a library/conversation row (the ListItem's inner Static)."""
    return str(item.query_one(Static).renderable)


_SHARED_MODE_OWNER_CASES = (
    # The count slot snapshots the pane's count line, which F-033 emptied for
    # unfiltered lists (the total now lives in the merged header purpose
    # line), so both cases expect "".
    pytest.param(
        "dictionaries",
        "dictionary",
        "New Dictionary Owner",
        "2 entries · on",
        "",
        id="dictionaries",
    ),
    pytest.param(
        "lore",
        "lore",
        "New Lore Owner",
        "3 entries · on",
        "",
        id="lore",
    ),
)


def _configure_shared_mode_sources(mock_app_instance, monkeypatch) -> None:
    """Install complete Dictionary/Lore list seams for shared-pane race tests."""
    mock_app_instance.runtime_backend = "local"
    mock_app_instance.chat_dictionary_scope_service = SimpleNamespace(
        list_dictionaries=AsyncMock(
            return_value={
                "dictionaries": [
                    {
                        "id": 91,
                        "name": "New Dictionary Owner",
                        "entry_count": 2,
                        "enabled": True,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        PersonasScreen,
        "_lore_manager",
        lambda self: object(),
    )


def _shared_pane_publication(screen) -> dict[str, object]:
    """Snapshot the shared row/count/label fields that establish pane ownership."""
    library = screen.query_one("#personas-library-pane")
    rendered_rows = tuple(
        (
            str(row.id),
            tuple(str(static.renderable) for static in row.query(Static).results()),
        )
        for row in library.query(".personas-library-row").results()
    )
    return {
        "mode": screen.state.active_mode,
        "rows": rendered_rows,
        "count": str(
            screen.query_one(
                "#personas-library-count",
                Static,
            ).renderable
        ),
        "sort": str(
            screen.query_one(
                "#personas-library-sort",
                Button,
            ).label
        ),
        "tag": str(
            screen.query_one(
                "#personas-library-tag",
                Button,
            ).label
        ),
    }


def _observe_shared_mode_render(
    monkeypatch,
    screen,
    mode: str,
    started,
) -> tuple[str, str]:
    """Mark a real mode renderer as started and stamp owner-distinct labels."""
    library = screen.query_one("#personas-library-pane")
    method_name = (
        "_render_dictionary_rows" if mode == "dictionaries" else "_render_lore_rows"
    )
    original_render = getattr(screen, method_name)
    owner_sort = f"Sort: {mode} owner"
    owner_tag = f"Tag: {mode} owner"

    async def observed_render(*args, **kwargs):
        started.set()
        await original_render(*args, **kwargs)
        library.set_sort_label(owner_sort)
        library.set_tag_label(owner_tag)

    monkeypatch.setattr(screen, method_name, observed_render)
    return owner_sort, owner_tag


def _right_edge(widget) -> int:
    """Right edge of a mounted widget region."""
    return widget.region.x + widget.region.width


def _relative_luminance(color) -> float:
    """Return WCAG relative luminance for a Rich color."""
    triplet = color.get_truecolor()

    def channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(triplet.red)
        + 0.7152 * channel(triplet.green)
        + 0.0722 * channel(triplet.blue)
    )


def _contrast_ratio(first, second) -> float:
    """Return WCAG contrast for two Rich colors."""
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_style_of_text(app: App, region, needle: str):
    """Return the compositor style that actually paints ``needle``."""
    strips = list(app.screen._compositor.render_strips())
    for y in range(region.y, region.bottom):
        if y >= len(strips):
            break
        segments = list(strips[y]._segments)
        row_text = "".join(segment.text for segment in segments)
        index = row_text.find(needle)
        if index == -1:
            continue
        x = 0
        for segment in segments:
            if x + len(segment.text) > index:
                return segment.style
            x += len(segment.text)
    return None


async def _mounted(pilot):
    await pilot.pause()
    return pilot.app.screen


class TestWorkbenchShell:
    async def test_route_renders_destination_workbench(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            header = screen.query_one("#personas-header")
            assert "ds-destination-header" in header.classes
            title = screen.query_one("#personas-header #workbench-header-title", Static)
            assert "Roleplay" in str(title.renderable)
            assert screen.query_one("#personas-mode-strip")
            assert screen.query_one("#personas-library-pane")
            assert screen.query_one("#personas-work-area")
            assert screen.query_one("#personas-inspector-pane")
            assert (
                screen.query_one("#personas-library-rail-open", Button).tooltip
                == "Open Library rail"
            )
            assert (
                screen.query_one("#personas-inspector-rail-open", Button).tooltip
                == "Open Inspector rail"
            )

    async def test_personas_screen_sets_up_reused_ccp_enhancements(
        self,
        mock_app_instance,
        stub_characters,
    ):
        """Verify PersonasScreen installs loading/decorator support for CCP handlers."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)

            assert hasattr(screen, "loading_manager")
            assert (
                getattr(
                    screen.character_handler.__class__,
                    "_personas_character_enhanced",
                    False,
                )
                is True
            )

    async def test_workbench_columns_fit_80_column_terminal(
        self, mock_app_instance, stub_characters
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(80, 40)) as pilot:
            screen = await _mounted(pilot)
            workbench = screen.query_one("#personas-workbench")
            library = screen.query_one("#personas-library-pane")
            work_area = screen.query_one("#personas-work-area")
            inspector = screen.query_one("#personas-inspector-pane")
            readiness = screen.query_one("#personas-readiness-console", Static)

            assert workbench.has_class("personas-workbench-compact")
            assert library.size.width >= 12
            assert work_area.size.width >= 34
            assert inspector.size.width >= 18
            assert _right_edge(inspector) <= _right_edge(workbench)
            # task-440 honesty contract: with no provider configured the
            # readiness line never claims ready (F-031 auto-select means a
            # selection exists, so this is the provider gate talking).
            assert "blocked" in str(readiness.renderable).lower()

    async def test_header_band_merges_purpose_and_count(
        self, mock_app_instance, stub_characters
    ):
        """F-033: purpose + count share one line; the separate status strip
        and the library pane's duplicate count line are gone."""
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(170, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # The five-strip band lost the standalone count strip entirely.
            assert not screen.query("#personas-status-row")
            purpose = screen.query_one("#personas-purpose", Static)
            assert str(purpose.renderable) == "Characters — who the AI plays · 2"
            # ...so the workbench starts one row higher than the old layout.
            assert screen.query_one("#personas-workbench").region.y == 10
            # The count renders once (header line), not again under the list.
            count = screen.query_one("#personas-library-count", Static)
            assert str(count.renderable) == ""

    async def test_header_purpose_count_updates_per_mode(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        """F-033: the merged line carries the live count for each mode."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            purpose = screen.query_one("#personas-purpose", Static)
            assert str(purpose.renderable) == "Characters — who the AI plays · 2"
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert "Personas" in str(purpose.renderable)
            assert "· 1" in str(purpose.renderable)

    async def test_library_rail_collapses_and_reopens_from_handle(
        self,
        mock_app_instance,
        stub_characters,
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)

            await pilot.click("#personas-library-rail-collapse")
            await pilot.pause()

            assert screen.query_one("#personas-library-pane").display is False
            assert screen.query_one("#personas-library-rail-handle").display is True
            assert screen.query_one("#personas-work-area").display is True

            await pilot.click("#personas-library-rail-open")
            await pilot.pause()

            assert screen.query_one("#personas-library-pane").display is True
            assert screen.query_one("#personas-library-rail-handle").display is False

    async def test_inspector_rail_collapses_and_reopens_from_handle(
        self,
        mock_app_instance,
        stub_characters,
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)

            await pilot.click("#personas-inspector-rail-collapse")
            await pilot.pause()

            assert screen.query_one("#personas-inspector-pane").display is False
            assert screen.query_one("#personas-inspector-rail-handle").display is True
            assert screen.query_one("#personas-work-area").display is True

            await pilot.click("#personas-inspector-rail-open")
            await pilot.pause()

            assert screen.query_one("#personas-inspector-pane").display is True
            assert screen.query_one("#personas-inspector-rail-handle").display is False

    async def test_collapsed_inspector_rail_handle_is_keyboard_reachable(
        self,
        mock_app_instance,
        stub_characters,
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-inspector-rail-collapse")
            await pilot.pause()

            open_button = screen.query_one("#personas-inspector-rail-open", Button)
            await pilot.press("shift+f6")
            await pilot.pause()
            assert pilot.app.focused is open_button

            await pilot.press("enter")
            await pilot.pause()

            assert screen.query_one("#personas-inspector-pane").display is True
            assert screen.query_one("#personas-inspector-rail-handle").display is False

    async def test_resize_sync_skips_work_when_compact_state_is_unchanged(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(80, 40)) as pilot:
            screen = await _mounted(pilot)
            assert screen.query_one("#personas-workbench").has_class(
                "personas-workbench-compact"
            )

            # Guard only the pane/workbench selectors `_sync_responsive_workbench()`
            # itself would touch if it (incorrectly) did work on an unchanged
            # compact state. A blanket fail-on-any-call patch is too broad now
            # that the screen's own `on_unmount` teardown legitimately calls
            # `self.query_one("AppFooterStatus")` (task-264, per-screen footer)
            # -- that unrelated call happens when this `async with` block exits
            # and would otherwise trip this guard as a false positive.
            original_query_one = screen.query_one

            def fail_query(selector, *args, **kwargs):
                if isinstance(selector, str) and selector.startswith("#personas-"):
                    raise AssertionError(
                        "unchanged compact state should not query panes"
                    )
                return original_query_one(selector, *args, **kwargs)

            monkeypatch.setattr(screen, "query_one", fail_query)
            screen._sync_responsive_workbench()

    async def test_resize_hook_has_google_style_docstring(self):
        docstring = inspect.getdoc(PersonasScreen.on_resize)

        assert docstring is not None
        assert "Args:" in docstring
        assert "event:" in docstring

    async def test_characters_mode_lists_library_rows(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam", "Lab Assistant"]

    async def test_footer_shortcut_context_set_and_cleared(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            context = screen._shortcut_context()
            rendered = context.render()
            assert "new" in rendered.lower()
            assert "search" in rendered.lower()
            # task-445: unavailable actions are dropped from the rendered
            # hint entirely rather than shown with a literal "unavailable"
            # suffix. F-031 auto-selects the first row on first paint, so
            # draft IS available here; "save" (no editor open) is the
            # remaining dropped hint.
            assert "save" not in rendered.lower()
            # task-2232: the footer names the one secondary CTA verbatim.
            assert "attach" not in rendered.lower()
            assert "ctrl+enter send to console draft" in rendered.lower()
            # F-038: the always-on accelerators are advertised, not hidden.
            assert "f6 pane" in rendered.lower()
            assert "ctrl+1-4 mode" in rendered.lower()
            assert "[ ]" in rendered
            # space toggle is dictionaries-only, so it stays hidden here.
            assert "space" not in rendered.lower()
            assert context.source == "personas"
            # task-264: the registration lands on the SCREEN's own footer,
            # not the harness's default-screen stand-in.
            footer = screen.query_one(AppFooterStatus)
            assert "new" in footer.shortcut_text.lower()
            assert "search" in footer.shortcut_text.lower()
            await pilot.app.pop_screen()
            await pilot.pause()
            # task-264: the context dies WITH the screen -- its footer is
            # detached from the DOM along with it (Textual's `is_mounted`
            # flag is stale after removal; `parent is None` is the reliable
            # signal), so no stale personas hints can leak to another
            # surface. The default-screen stand-in (never registered
            # against) shows the default shortcuts.
            assert footer.parent is None
            default_footer = pilot.app.query_one(AppFooterStatus)
            assert default_footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT

    async def test_footer_advertises_space_toggle_in_dictionaries_mode(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        """F-038: the dictionary row toggle key is disclosed only in its mode."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert "space" not in screen._shortcut_context().render().lower()
            await screen._apply_mode("dictionaries")
            await pilot.pause()
            rendered = screen._shortcut_context().render().lower()
            assert "space toggle" in rendered
            await screen._apply_mode("characters")
            await pilot.pause()
            assert "space" not in screen._shortcut_context().render().lower()

    async def test_mode_chips_advertise_their_ctrl_shortcut(
        self, mock_app_instance, stub_characters
    ):
        """F-038: each mode chip tooltip carries its Ctrl+N jump key."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            expected = {
                "characters": ("Characters — who the AI plays.", "(Ctrl+1)"),
                "personas": ("Personas — who you play in the chat.", "(Ctrl+2)"),
                "dictionaries": ("Dictionaries — text find/replace rules.", "(Ctrl+3)"),
                "lore": ("Lore — world facts injected on keywords.", "(Ctrl+4)"),
            }
            for mode, (descriptor, hint) in expected.items():
                tooltip = screen.query_one(f"#personas-mode-{mode}", Button).tooltip
                assert descriptor in tooltip, mode
                assert hint in tooltip, mode

    async def test_unmount_clear_does_not_stomp_other_screens_context(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            # task-264: the registration lands on the SCREEN's own footer,
            # not the harness's default-screen stand-in.
            footer = screen.query_one(AppFooterStatus)
            # Another screen registers its context (switch_screen mounts the
            # incoming screen before unmounting the outgoing one).
            footer.set_shortcut_context(
                ShortcutContext(
                    source="console",
                    actions=(ShortcutAction("ctrl+enter", "send"),),
                )
            )
            screen._clear_footer_shortcuts()
            assert "ctrl+enter send" in footer.shortcut_text
            assert footer.shortcut_text != AppFooterStatus.DEFAULT_SHORTCUT_TEXT

    async def test_purpose_line_shows_live_counts_per_mode(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        """F-033: the merged purpose line carries each mode's live count."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            purpose = screen.query_one("#personas-purpose", Static)
            assert "· 2" in str(purpose.renderable)
            assert "Characters" in str(purpose.renderable)
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert "· 1" in str(purpose.renderable)
            assert "Personas" in str(purpose.renderable)
            await pilot.click("#personas-mode-dictionaries")
            await pilot.pause()
            assert "Dictionaries" in str(purpose.renderable)

    async def test_placeholder_modes_show_placeholder_panel(
        self, mock_app_instance, stub_characters
    ):
        """ "prompts" is the one remaining placeholder mode: dictionaries shipped
        in Roleplay P1a and lore shipped in Roleplay P2a (Task 6); prompts is
        retired to the Library (Task 7) and has no chip, so it is reached
        directly via ``_apply_mode`` rather than a chip click."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await screen._apply_mode("prompts")
            await pilot.pause()
            assert screen.state.active_mode == "prompts"
            placeholder = screen.query_one("#personas-mode-placeholder", Static)
            assert placeholder.display is True
            assert "moving to the Library" in str(placeholder.renderable)

    async def test_mode_chips_are_self_explaining_and_mark_coming_soon(
        self, mock_app_instance, stub_characters
    ):
        """All chip modes are live now (Lore shipped in Roleplay P2a Task 6);
        no chip carries the "soon" suffix anymore."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            lore_chip = screen.query_one("#personas-mode-lore", Button)
            # F-038: chip tooltips carry their Ctrl+N jump key.
            assert (
                lore_chip.tooltip == "Lore — world facts injected on keywords. (Ctrl+4)"
            )
            assert "soon" not in str(lore_chip.label).lower()
            char_chip = screen.query_one("#personas-mode-characters", Button)
            assert "soon" not in str(char_chip.label).lower()

    async def test_coming_soon_mode_shows_inviting_copy(
        self, mock_app_instance, stub_characters
    ):
        """ "prompts" is the last mode still rendered as a placeholder (retired
        to the Library, Task 7) - lore now has its own live detail view."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await screen._apply_mode("prompts")
            await pilot.pause()
            body = str(
                screen.query_one("#personas-mode-placeholder", Static).renderable
            )
            assert "moving to the library" in body.lower()
            assert "not available yet" not in body.lower()

    async def test_title_reframed_to_roleplay_keeps_state_suffix(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            title = screen.query_one("#personas-header #workbench-header-title", Static)
            assert str(title.renderable).startswith("Roleplay")
            status = screen.query_one(
                "#personas-header #workbench-header-status", Static
            )
            # F-031 auto-selects the first row on first paint, which makes
            # the provider gate operative (task-440): the mock config has no
            # ready provider, so the header honestly reads Blocked.
            assert str(status.renderable) == "Blocked"
            # dynamic suffix still appends in create mode
            screen._edit_mode = "create"
            screen._update_title()
            await pilot.pause()
            subtitle = str(
                screen.query_one(
                    "#personas-header #workbench-header-subtitle", Static
                ).renderable
            )
            assert "New character" in subtitle

    async def test_purpose_shows_active_mode_descriptor_and_updates_on_switch(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            purpose = screen.query_one("#personas-purpose", Static)
            assert "who the AI plays" in str(
                purpose.renderable
            )  # characters is the default mode
            await screen._apply_mode("personas")
            await pilot.pause()
            # F-033: the purpose line also carries the live count.
            assert str(screen.query_one("#personas-purpose", Static).renderable) == (
                "Personas — who you play in the chat · 0"
            )


class TestCharacterSelectionAndEdit:
    async def test_row_selection_shows_card_and_inspector(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            assert screen.state.selected_entity_id == "1"
            assert screen._edit_mode == "view"
            assert "Selected: Detective Sam" in str(
                screen.query_one("#personas-selected-name", Static).renderable
            )

    async def test_card_body_populates_on_selection(
        self, mock_app_instance, stub_characters
    ):
        """The card's BODY must populate: placeholder hidden, fields filled.

        Mirrors the screenshot QA defect where the inspector and preview
        populated but the center card kept its 'No character loaded.'
        placeholder with an empty details area.
        """
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.query_one("#ccp-character-card-view").display is True
            placeholder = screen.query_one("#personas-character-card-empty")
            assert placeholder.display is False
            body = screen.query_one("#personas-character-card-body")
            assert body.display is True
            name = screen.query_one("#personas-character-card-name", Static)
            assert "Detective Sam" in str(name.renderable)

    async def test_new_button_opens_editor_in_create_mode(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            # Select a character first: entering create mode must not leave
            # the previous selection's identity in the inspector.
            await pilot.click("#personas-library-row-character-1")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            assert screen._edit_mode == "create"
            editor = screen.query_one("#ccp-character-editor-view")
            assert editor.display is True
            selected_name = str(
                screen.query_one("#personas-selected-name", Static).renderable
            )
            assert "Detective Sam" not in selected_name
            # Unsaved gating must survive the identity reset: no Console
            # action is offered for a pristine create session (F-031: the
            # readiness line falls back to the no-selection guidance).
            readiness = str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert readiness == "Pick a character or persona to start chatting."
            assert screen._console_action_allowed() is False

    async def test_ctrl_n_opens_editor_in_create_mode(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert screen._edit_mode == "create"
            assert screen.query_one("#ccp-character-editor-view").display is True

    async def test_ctrl_f_focuses_library_search(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            await _mounted(pilot)
            await pilot.pause()
            await pilot.press("ctrl+f")
            await pilot.pause()
            focused = pilot.app.focused
            assert focused is not None
            assert focused.id == "personas-library-search"

    async def test_save_with_missing_name_blocks_and_shows_validation(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """Blocked saves render in the editor footer; the inspector says
        "editing..." while an editor is open (the footer is the single
        in-editor validation surface)."""
        created = []
        monkeypatch.setattr(
            character_handler_module,
            "create_character",
            lambda data: created.append(data) or 99,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
                CharacterSaveRequested,
            )

            screen.post_message(
                CharacterSaveRequested({"name": "", "first_message": "Hi"})
            )
            await pilot.pause()
            editor_validation = screen.query_one(
                "#personas-char-editor-validation", Static
            )
            assert "name: required" in str(editor_validation.renderable)
            summary = screen.query_one("#personas-validation-summary", Static)
            assert "Validation: editing..." in str(summary.renderable)
            assert "OK" not in str(summary.renderable)
        assert created == []

    async def test_character_book_errors_render_in_editor_footer(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """Screen-side validation (character_book) renders in the editor footer."""
        created = []
        monkeypatch.setattr(
            character_handler_module,
            "create_character",
            lambda data: created.append(data) or 99,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
                CharacterSaveRequested,
            )

            # entries is required (and must be a list) when a book is present.
            screen.post_message(
                CharacterSaveRequested(
                    {"name": "Bookish", "character_book": {"entries": "nope"}}
                )
            )
            await pilot.pause()
            editor_validation = screen.query_one(
                "#personas-char-editor-validation", Static
            )
            assert "character_book" in str(editor_validation.renderable)
            summary = screen.query_one("#personas-validation-summary", Static)
            assert "Validation: editing..." in str(summary.renderable)
        assert created == []

    async def test_inspector_validation_reads_editing_while_editor_open(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """Editor open -> inspector "editing..."; save success -> back to OK."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            EditCharacterRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            CharacterSaveRequested,
        )

        monkeypatch.setattr(
            character_handler_module, "update_character", lambda cid, data: True
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            summary = screen.query_one("#personas-validation-summary", Static)
            assert "Validation: OK" in str(summary.renderable)
            screen.post_message(EditCharacterRequested("1"))
            await pilot.pause()
            assert "Validation: editing..." in str(summary.renderable)
            screen.post_message(
                CharacterSaveRequested({"name": "Detective Sam", "version": 1})
            )
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "Validation: OK" in str(summary.renderable)

    async def test_inspector_validation_back_to_ok_on_editor_cancel(
        self, mock_app_instance, stub_characters
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            EditCharacterRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            CharacterEditorCancelled,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen.post_message(EditCharacterRequested("1"))
            await pilot.pause()
            summary = screen.query_one("#personas-validation-summary", Static)
            assert "Validation: editing..." in str(summary.renderable)
            screen.post_message(CharacterEditorCancelled())
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "Validation: OK" in str(summary.renderable)

    async def test_save_calls_create_and_refreshes(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        created = []
        monkeypatch.setattr(
            character_handler_module,
            "create_character",
            lambda data: created.append(data) or 99,
        )

        def fetch_all_with_created():
            characters = [dict(c) for c in CHARACTERS]
            if created:
                characters.append({"id": 99, "name": "New Hero", "version": 1})
            return characters

        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", fetch_all_with_created
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
                CharacterSaveRequested,
            )

            screen.post_message(
                CharacterSaveRequested({"name": "New Hero", "first_message": "Hi"})
            )
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_id == "99"
            assert screen._edit_mode == "view"
        assert created and created[0]["name"] == "New Hero"

    async def test_edit_requested_for_mismatched_character_is_ignored(
        self, mock_app_instance, stub_characters
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            EditCharacterRequested,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            screen.post_message(EditCharacterRequested(2))
            await pilot.pause()
            assert screen._edit_mode == "view"
            assert screen.query_one("#ccp-character-editor-view").display is False

    async def test_mode_switch_during_save_does_not_render_character_into_other_mode(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        import threading

        release = threading.Event()
        created = []

        def blocking_create(data):
            release.wait(timeout=5)
            created.append(data)
            return 99

        monkeypatch.setattr(
            character_handler_module, "create_character", blocking_create
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
                CharacterSaveRequested,
            )

            screen.post_message(
                CharacterSaveRequested({"name": "New Hero", "first_message": "Hi"})
            )
            await pilot.pause()  # Save worker is now blocked inside create_character.
            await screen._apply_mode("prompts")
            await pilot.pause()
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert created and created[0]["name"] == "New Hero"
            assert screen.state.active_mode == "prompts"
            assert screen.state.selected_entity_id is None
            placeholder = screen.query_one("#personas-mode-placeholder", Static)
            assert placeholder.display is True
            assert screen.query_one("#ccp-character-card-view").display is False
            assert "New Hero" not in str(
                screen.query_one("#personas-selected-name", Static).renderable
            )


class TestPersonasMode:
    async def _enter_personas_mode(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-mode-personas")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def test_personas_mode_lists_profiles(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Archivist"]

    async def test_personas_mode_copy_avoids_human_identity_actions(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            chip = screen.query_one("#personas-mode-personas", Button)
            assert str(chip.label) == "Personas"

            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            # F-034: the descriptor teaches the genre convention (who YOU
            # play) - and F-033 merged the live count into the same line.
            assert (
                str(screen.query_one("#personas-purpose", Static).renderable)
                == "Personas — who you play in the chat · 1"
            )
            visible_copy = "\n".join(
                [
                    str(widget.label)
                    if isinstance(widget, Button)
                    else str(widget.renderable)
                    for widget in screen.query("Button, Static")
                    if widget.display
                ]
            )
            for forbidden in (
                "User" + " Profiles",
                "Set as my name",
                "Clear my name",
                "Chatting as",
            ):
                assert forbidden not in visible_copy

    async def test_personas_mode_service_failure_shows_recovery_state(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        stub_scope_service.list_persona_profiles.side_effect = RuntimeError(
            "scope offline"
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)

            recovery = screen.query_one("#personas-service-error", Static)
            copy = str(recovery.renderable)
            assert "Personas unavailable" in copy
            assert "Unavailable:" in copy
            assert "Recovery:" in copy
            assert "scope offline" in copy
            assert not list(screen.query("#personas-library-empty"))

    async def test_personas_mode_service_failure_replaces_stale_rows(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            assert [_row_text(r) for r in screen.query(".personas-library-row")] == [
                "Archivist"
            ]

            stub_scope_service.list_persona_profiles.side_effect = RuntimeError(
                "scope offline"
            )
            screen._refresh_profile_rows_worker()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert screen.query_one("#personas-service-error", Static)
            assert not list(screen.query(".personas-library-row"))

    async def test_personas_mode_empty_state_copy_unchanged(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        stub_scope_service.list_persona_profiles.return_value = {
            "items": [],
            "total": 0,
        }

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)

            empty = screen.query_one("#personas-library-empty", Static)
            assert str(empty.renderable) == "No personas yet - use New to add one."
            assert not list(screen.query("#personas-service-error"))

    async def test_profile_selection_shows_card(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-row-persona-p-1")
            await pilot.pause()
            assert screen.state.selected_entity_kind == "persona"
            assert screen.query_one("#ccp-persona-card-view").display is True
            assert "Selected: Archivist" in str(
                screen.query_one("#personas-selected-name", Static).renderable
            )

    async def test_profile_save_calls_scope_service(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            assert screen._edit_mode == "create"
            screen.post_message(PersonaProfileSaveRequested({"name": "Mentor"}))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            stub_scope_service.create_persona_profile.assert_awaited_once()
            # Save-in-place: create -> edit, the editor stays open (not the
            # read-only card).
            assert screen._edit_mode == "edit"
            assert screen.query_one("#ccp-persona-editor-view").display is True

    async def test_profile_save_refresh_failure_updates_purpose_line_and_recovery(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            assert "· 1" in str(
                screen.query_one("#personas-purpose", Static).renderable
            )

            stub_scope_service.list_persona_profiles.side_effect = RuntimeError(
                "scope offline"
            )
            await pilot.click("#personas-library-new")
            await pilot.pause()
            screen.post_message(PersonaProfileSaveRequested({"name": "Mentor"}))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert screen.query_one("#personas-service-error", Static)
            assert not list(screen.query(".personas-library-row"))
            assert "· 0" in str(
                screen.query_one("#personas-purpose", Static).renderable
            )

    async def test_profile_edit_save_calls_update(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-row-persona-p-1")
            await pilot.pause()
            screen.post_message(EditPersonaProfileRequested("p-1"))
            await pilot.pause()
            assert screen._edit_mode == "edit"
            assert screen.query_one("#ccp-persona-editor-view").display is True
            screen.post_message(
                PersonaProfileSaveRequested({"id": "p-1", "name": "Archivist 2"})
            )
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            stub_scope_service.update_persona_profile.assert_awaited_once()
            assert stub_scope_service.update_persona_profile.await_args.args[0] == "p-1"
            # Save-in-place: the editor stays open after an edit save too.
            assert screen._edit_mode == "edit"
            assert screen.query_one("#ccp-persona-editor-view").display is True

    async def test_local_profile_editor_roundtrip_builds_local_update_schema(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        raw_local_record = {
            **PROFILE,
            "description": "RAW-LOCAL-DESCRIPTION-SENTINEL",
            "personality_traits": "RAW-LOCAL-TRAITS-SENTINEL",
        }
        stub_scope_service.list_persona_profiles.return_value = {
            "items": [dict(raw_local_record)],
            "total": 1,
        }
        stub_scope_service.get_persona_profile.return_value = dict(raw_local_record)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-row-persona-p-1")
            await pilot.pause()
            screen.post_message(EditPersonaProfileRequested("p-1"))
            await pilot.pause()

            description = screen.query_one("#personas-editor-description", TextArea)
            personality_traits = screen.query_one(
                "#personas-editor-personality-traits", TextArea
            )
            assert description.text == "RAW-LOCAL-DESCRIPTION-SENTINEL"
            assert personality_traits.text == "RAW-LOCAL-TRAITS-SENTINEL"

            screen.query_one("#personas-editor-name", Input).value = "Archivist Revised"
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            request = stub_scope_service.update_persona_profile.await_args.args[1]
            assert isinstance(request, LocalPersonaProfileUpdate)
            assert request.name == "Archivist Revised"
            assert request.description == "RAW-LOCAL-DESCRIPTION-SENTINEL"
            assert request.personality_traits == "RAW-LOCAL-TRAITS-SENTINEL"

    async def test_profile_save_failure_keeps_editor_open(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        stub_scope_service.create_persona_profile.side_effect = RuntimeError("boom")
        notifications: list[tuple[str, str]] = []
        app = PersonasTestApp(mock_app_instance)
        app.notify = lambda message, severity="information", **kwargs: (
            notifications.append((str(message), severity))
        )
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            screen.post_message(PersonaProfileSaveRequested({"name": "Mentor"}))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any(
                "Save failed" in message and severity == "error"
                for message, severity in notifications
            )
            assert screen._edit_mode == "create"
            assert screen.query_one("#ccp-persona-editor-view").display is True

    async def test_local_profile_save_passes_local_schema_with_local_only_fields(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            screen.post_message(
                PersonaProfileSaveRequested(
                    {
                        "name": "Mentor",
                        "description": "Guides new users",
                        "personality_traits": "patient, curious",
                    }
                )
            )
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            stub_scope_service.create_persona_profile.assert_awaited_once()
            request = stub_scope_service.create_persona_profile.await_args.args[0]
            assert isinstance(request, LocalPersonaProfileCreate)
            assert request.name == "Mentor"
            assert request.description == "Guides new users"
            assert request.personality_traits == "patient, curious"

    async def test_server_profile_save_omits_local_only_fields(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await screen.handle_runtime_backend_changed("server")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            screen.post_message(
                PersonaProfileSaveRequested(
                    {
                        "name": "Mentor",
                        "description": "Must remain local",
                        "personality_traits": "must remain local",
                    }
                )
            )
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            request = stub_scope_service.create_persona_profile.await_args.args[0]
            assert isinstance(request, PersonaProfileCreate)
            assert (
                request.model_dump()
                .keys()
                .isdisjoint({"description", "personality_traits"})
            )
            assert stub_scope_service.create_persona_profile.await_args.kwargs[
                "mode"
            ] == ("server")

    async def test_server_profile_edit_blocks_and_omits_local_only_fields(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await screen.handle_runtime_backend_changed("server")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-library-row-persona-p-1")
            await pilot.pause()
            screen.post_message(EditPersonaProfileRequested("p-1"))
            await pilot.pause()

            assert screen.query_one("#personas-editor-description").disabled is True
            assert (
                screen.query_one("#personas-editor-personality-traits").disabled is True
            )
            screen.post_message(
                PersonaProfileSaveRequested(
                    {
                        "id": "p-1",
                        "version": 1,
                        "name": "Archivist 2",
                        "description": "Must remain local",
                        "personality_traits": "must remain local",
                    }
                )
            )
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            request = stub_scope_service.update_persona_profile.await_args.args[1]
            assert isinstance(request, PersonaProfileUpdate)
            assert (
                request.model_dump()
                .keys()
                .isdisjoint({"description", "personality_traits"})
            )
            assert stub_scope_service.update_persona_profile.await_args.kwargs[
                "mode"
            ] == ("server")

    async def test_double_save_creates_once(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-library-new")
            await pilot.pause()
            screen.post_message(PersonaProfileSaveRequested({"name": "Mentor"}))
            screen.post_message(PersonaProfileSaveRequested({"name": "Mentor"}))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            stub_scope_service.create_persona_profile.assert_awaited_once()

    async def test_character_mode_unaffected(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._enter_personas_mode(pilot)
            await pilot.click("#personas-mode-characters")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.active_mode == "characters"
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam", "Lab Assistant"]


PROFILES_FOR_SEARCH = [
    {
        "id": "p-1",
        "name": "Archivist",
        "description": "Preserve and retrieve",
        "system_prompt": "You are a meticulous archivist.",
    },
    {
        "id": "p-2",
        "name": "Navigator",
        "description": "Charts the course",
        "system_prompt": "You guide the user.",
    },
]


class TestSearch:
    async def _wait_for_search_render(self, pilot: Any) -> None:
        await pilot.pause(
            personas_screen_module.PERSONAS_SEARCH_DEBOUNCE_SECONDS + 0.05
        )
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_search_input_debounces_rapid_changes(
        self,
        mock_app_instance: Any,
        stub_characters: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rapid search edits render only the final query after the debounce window."""

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()

            rendered_queries: list[str] = []
            original_render = screen._render_library_rows

            async def observe_render(
                *,
                expected_query: str | None = None,
                expected_mode: str | None = None,
            ) -> None:
                rendered_queries.append(screen.state.search_query)
                await original_render(
                    expected_query=expected_query,
                    expected_mode=expected_mode,
                )

            monkeypatch.setattr(screen, "_render_library_rows", observe_render)

            search_input = screen.query_one("#personas-library-search")
            search_input.value = "s"
            search_input.value = "sa"
            search_input.value = "sam"

            await pilot.pause(0.05)
            assert rendered_queries == []

            await self._wait_for_search_render(pilot)
            assert rendered_queries == ["sam"]
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam"]

    async def test_stale_fts_search_result_does_not_update_library_rows(
        self,
        mock_app_instance: Any,
        stub_characters: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A slower search result is dropped if the query changes while it awaits."""

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()

            screen.LIBRARY_FTS_THRESHOLD = 2
            library = screen.query_one("#personas-library-pane")
            original_update_rows = library.update_rows
            rendered_rows: list[tuple[str, ...]] = []

            async def observe_update_rows(rows: tuple[Any, ...], **kwargs: Any) -> None:
                rendered_rows.append(tuple(row.name for row in rows))
                await original_update_rows(rows, **kwargs)

            async def fake_to_thread(
                function: Any,
                query: str,
                *args: Any,
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                screen.state.search_query = "lab"
                return [{"id": 1, "name": "Detective Sam"}]

            monkeypatch.setattr(library, "update_rows", observe_update_rows)
            monkeypatch.setattr(
                personas_screen_module.asyncio, "to_thread", fake_to_thread
            )

            screen.state.search_query = "sam"
            await screen._render_search_query(query="sam", mode="characters")
            await pilot.pause()

            assert rendered_rows == []

    async def test_search_filters_loaded_characters_locally(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            # Type into the search input
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "sam"
            await self._wait_for_search_render(pilot)
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam"]
            # F-033: the match count now lives in the merged header purpose
            # line; the pane's duplicate count line stays empty.
            count = str(screen.query_one("#personas-library-count", Static).renderable)
            assert count == ""
            assert "· 1" in str(
                screen.query_one("#personas-purpose", Static).renderable
            )

    async def test_clearing_search_restores_all_rows(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            search_input = screen.query_one("#personas-library-search")
            # Filter first
            search_input.value = "sam"
            await self._wait_for_search_render(pilot)
            # Then clear
            search_input.value = ""
            await self._wait_for_search_render(pilot)
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam", "Lab Assistant"]
            count = str(screen.query_one("#personas-library-count", Static).renderable)
            # F-033: unfiltered count renders once, in the header purpose line.
            assert count == ""
            assert "· 2" in str(
                screen.query_one("#personas-purpose", Static).renderable
            )

    async def test_search_is_case_insensitive(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "LAB"
            await self._wait_for_search_render(pilot)
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Lab Assistant"]

    async def test_search_filters_profiles_in_personas_mode(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        # Replace the scope service stub with two profiles
        stub_scope_service.list_persona_profiles = AsyncMock(
            return_value={"items": [dict(p) for p in PROFILES_FOR_SEARCH], "total": 2}
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # Two profiles loaded; now search for "nav"
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "nav"
            await self._wait_for_search_render(pilot)
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Navigator"]
            # F-033: the pane's duplicate count line stays empty; the merged
            # header line carries the personas library total.
            count = str(screen.query_one("#personas-library-count", Static).renderable)
            assert count == ""
            assert "· 2" in str(
                screen.query_one("#personas-purpose", Static).renderable
            )

    async def test_mode_switch_clears_search(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            # Search in characters mode
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "sam"
            await self._wait_for_search_render(pilot)
            assert len(screen.query(".personas-library-row")) == 1
            # Switch to personas mode and back
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-mode-characters")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # All rows visible after round-trip
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam", "Lab Assistant"]
            # Search input is cleared
            assert screen.query_one("#personas-library-search").value == ""

    # NOTE (P3a Task 4): the old in-memory-vs-FTS hybrid search (gated on
    # LIBRARY_FTS_THRESHOLD, using ccp_character_handler.search_characters_fts and
    # the "Showing N ... from full library" copy) was replaced by a single paged
    # DB search (get_character_page_for_ui / count_character_page). The former
    # tests test_fts_path_used_for_large_libraries,
    # test_fts_search_count_uses_unbounded_full_library_copy, and
    # test_fts_search_runs_off_the_event_loop covered that removed path; the
    # off-thread guarantee for the new path is covered in
    # Tests/UI/test_personas_library_scale.py.

    async def test_concurrent_renders_do_not_duplicate_rows(
        self, mock_app_instance, stub_characters
    ):
        """Two back-to-back renders must serialize instead of double-mounting."""
        import asyncio

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await asyncio.gather(
                screen._render_library_rows(), screen._render_library_rows()
            )
            await pilot.pause()
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Detective Sam", "Lab Assistant"]


class TestImportExport:
    """Path-based import/export flows; file dialogs are never exercised."""

    @pytest.fixture
    def stub_db(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: sentinel
        )
        return sentinel

    @staticmethod
    def _capture_notifications(app) -> list[tuple[str, str]]:
        """Shadow App.notify with an instance attribute, like _notify resolves it."""
        captured: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: captured.append(
            (str(message), severity)
        )
        return captured

    async def test_json_export_path_failure_never_logs_sensitive_destination(
        self,
        tmp_path,
    ):
        from loguru import logger as loguru_logger

        secret = "credential-private-origin-message-text"
        hidden_parent = tmp_path / f".{secret}"
        hidden_parent.mkdir()
        target = hidden_parent / "character.json"
        messages: list[str] = []
        sink = loguru_logger.add(
            lambda message: messages.append(str(message)),
            level="DEBUG",
        )
        try:
            with pytest.raises(ValueError):
                PersonasScreen._write_text_file(str(target), "{}")
        finally:
            loguru_logger.remove(sink)

        assert secret not in "".join(messages)

    async def test_import_success_refreshes_selects_and_clears_search(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        card_bytes = b'{"name":"Imported Hero"}'
        card_path = tmp_path / "card.json"
        card_path.write_bytes(card_bytes)
        imported_sources: list[bytes] = []

        def fake_import(source_bytes):
            imported_sources.append(source_bytes)
            return CharacterCardImportOutcome(3, True, None, None)

        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _source_bytes: CharacterCardTTSInspection(),
        )
        monkeypatch.setattr(
            character_handler_module, "import_character_card_with_outcome", fake_import
        )

        def fetch_all_with_imported():
            characters = [dict(c) for c in CHARACTERS]
            if imported_sources:
                characters.append({"id": 3, "name": "Imported Hero", "version": 1})
            return characters

        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", fetch_all_with_imported
        )
        monkeypatch.setattr(
            character_handler_module,
            "fetch_character_by_id",
            lambda character_id: next(
                (
                    dict(c)
                    for c in fetch_all_with_imported()
                    if str(c["id"]) == str(character_id)
                ),
                None,
            ),
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            search_input = screen.query_one("#personas-library-search")
            search_input.value = "sam"
            await pilot.pause()
            await screen._import_character_from_path(str(card_path))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert imported_sources == [card_bytes]
            assert screen.state.selected_entity_id == "3"
            assert screen.query_one("#personas-library-search").value == ""
            rows = screen.query(".personas-library-row")
            assert "Imported Hero" in [_row_text(r) for r in rows]

    async def test_import_success_notification_lingers_past_the_app_default(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        """task-445: the review saw the import-success toast flash by --
        it fired at the same instant the card view/inspector swapped in,
        against the app's plain 5s default. It must now request an explicit
        timeout longer than that default so it reads at normal pace."""
        card_bytes = b'{"name":"Imported Hero"}'
        card_path = tmp_path / "card.json"
        card_path.write_bytes(card_bytes)
        imported_sources: list[bytes] = []

        def fake_import(source_bytes):
            imported_sources.append(source_bytes)
            return CharacterCardImportOutcome(3, True, None, None)

        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _source_bytes: CharacterCardTTSInspection(),
        )
        monkeypatch.setattr(
            character_handler_module, "import_character_card_with_outcome", fake_import
        )

        def fetch_all_with_imported():
            characters = [dict(c) for c in CHARACTERS]
            if imported_sources:
                characters.append({"id": 3, "name": "Imported Hero", "version": 1})
            return characters

        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", fetch_all_with_imported
        )
        monkeypatch.setattr(
            character_handler_module,
            "fetch_character_by_id",
            lambda character_id: next(
                (
                    dict(c)
                    for c in fetch_all_with_imported()
                    if str(c["id"]) == str(character_id)
                ),
                None,
            ),
        )
        app = PersonasTestApp(mock_app_instance)
        calls: list[tuple[str, str, dict]] = []
        app.notify = lambda message, severity="information", **kwargs: calls.append(
            (str(message), severity, kwargs)
        )
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await screen._import_character_from_path(str(card_path))
            await pilot.pause()
        matches = [c for c in calls if c[0] == "Character imported."]
        assert matches, f"no 'Character imported.' notify call in {calls}"
        message, severity, kwargs = matches[-1]
        assert severity == "information"
        timeout = kwargs.get("timeout")
        assert timeout is not None and timeout > 5, (
            f"import-success notify must linger past the 5s app default, got {timeout!r}"
        )

    async def test_import_markdown_routes_through_character_import_helper(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        imported_sources: list[bytes] = []
        card_path = tmp_path / "card.md"
        card_path.write_text("# Character Card\n", encoding="utf-8")
        card_bytes = card_path.read_bytes()

        def fake_import(source_bytes):
            imported_sources.append(source_bytes)
            return CharacterCardImportOutcome(3, True, None, None)

        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _source_bytes: CharacterCardTTSInspection(),
        )
        monkeypatch.setattr(
            character_handler_module, "import_character_card_with_outcome", fake_import
        )

        def fetch_all_with_imported():
            characters = [dict(c) for c in CHARACTERS]
            if imported_sources:
                characters.append({"id": 3, "name": "Markdown Hero", "version": 1})
            return characters

        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", fetch_all_with_imported
        )
        monkeypatch.setattr(
            character_handler_module,
            "fetch_character_by_id",
            lambda character_id: next(
                (
                    dict(c)
                    for c in fetch_all_with_imported()
                    if str(c["id"]) == str(character_id)
                ),
                None,
            ),
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await screen._import_character_from_path(str(card_path))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert imported_sources == [card_bytes]
            assert screen.state.selected_entity_id == "3"

    async def test_import_invalid_markdown_uses_failure_path_without_selection_change(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        bad_path = tmp_path / "bad.md"
        bad_path.write_text("# Not a character card\n", encoding="utf-8")
        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _source_bytes: CharacterCardTTSInspection(),
        )
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            lambda _source_bytes: None,
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await screen._import_character_from_path(str(bad_path))
            await pilot.pause()
            assert screen.state.selected_entity_id == "1"
            assert any(
                "valid character card" in message and severity == "error"
                for message, severity in notifications
            )

    async def test_second_import_request_ignored_while_dialog_active(
        self, mock_app_instance, stub_characters
    ):
        """A queued import action must not start a second dialog worker."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            calls: list[int] = []

            def counting_worker():
                calls.append(1)

                async def _noop():
                    pass

                return _noop()

            screen._import_dialog_worker = counting_worker
            screen._io_dialog_active = True
            screen.post_message(PersonaActionRequested(action="import"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert calls == []
            # Sanity check: with the flag cleared the same wiring does fire.
            screen._io_dialog_active = False
            screen.post_message(PersonaActionRequested(action="import"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert calls == [1]

    async def test_import_existing_character_notifies_already_existed(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        """A name-conflict import returns an existing id; the copy must say so
        honestly (task-429: re-import does not update the existing character)."""
        card_path = tmp_path / "dupe.json"
        card_path.write_bytes(b'{"name":"Detective Sam"}')
        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _source_bytes: CharacterCardTTSInspection(),
        )
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            lambda _source_bytes: CharacterCardImportOutcome(1, False, None, None),
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await screen._import_character_from_path(str(card_path))
            await pilot.pause()
            assert screen.state.selected_entity_id == "1"
            assert any(
                "already existed" in message
                and "does not update an existing character" in message
                and severity == "information"
                for message, severity in notifications
            )
            assert not any(
                "Character imported." in message for message, _ in notifications
            )

    async def test_imported_lorebook_note_names_book(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """The helper turns a saved character's world-book extension into
        readable copy naming the book and its entry count (task-429)."""
        monkeypatch.setattr(
            character_handler_module,
            "fetch_character_by_id",
            lambda character_id: {
                "extensions": {
                    "character_world_books": [
                        {"name": "Second Chance Lore", "entries": [{}, {}]}
                    ]
                }
            },
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            note = await screen._imported_lorebook_note("7")
            assert "Second Chance Lore" in note
            assert "2 entries" in note

    async def test_imported_lorebook_note_coerces_json_string_extensions(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """Extensions saved/returned as a JSON string must still be readable."""
        monkeypatch.setattr(
            character_handler_module,
            "fetch_character_by_id",
            lambda character_id: {
                "extensions": json.dumps(
                    {"character_world_books": [{"name": "Old Lore"}]}
                )
            },
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            note = await screen._imported_lorebook_note("9")
            assert "Old Lore" in note
            assert "0 entries" in note

    async def test_imported_lorebook_note_empty_without_a_book(
        self, mock_app_instance, stub_characters
    ):
        """No world-book extension on the saved character yields no note."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            # stub_characters' fetch_character_by_id returns bare CHARACTERS
            # records with no `extensions` key at all.
            note = await screen._imported_lorebook_note("1")
            assert note == ""

    async def test_imported_lorebook_note_swallows_fetch_errors(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """A DB error while re-fetching the character must never raise; the
        toast should simply omit the lorebook note (guard-every-read)."""

        def boom(character_id):
            raise RuntimeError("db offline")

        monkeypatch.setattr(character_handler_module, "fetch_character_by_id", boom)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            note = await screen._imported_lorebook_note("1")
            assert note == ""

    async def test_import_success_names_lorebook_in_toast(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        """A freshly imported card carrying an embedded book gets named in
        the success toast, end to end (task-429)."""
        card_path = tmp_path / "card.json"
        card_path.write_bytes(b'{"name":"Imported Hero"}')
        imported_sources: list[bytes] = []

        def fake_import(source_bytes):
            imported_sources.append(source_bytes)
            return CharacterCardImportOutcome(3, True, None, None)

        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _source_bytes: CharacterCardTTSInspection(),
        )
        monkeypatch.setattr(
            character_handler_module, "import_character_card_with_outcome", fake_import
        )

        def fetch_all_with_imported():
            characters = [dict(c) for c in CHARACTERS]
            if imported_sources:
                characters.append({"id": 3, "name": "Imported Hero", "version": 1})
            return characters

        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", fetch_all_with_imported
        )

        def fetch_with_book(character_id):
            record = next(
                (
                    dict(c)
                    for c in fetch_all_with_imported()
                    if str(c["id"]) == str(character_id)
                ),
                None,
            )
            if record and str(character_id) == "3":
                record["extensions"] = {
                    "character_world_books": [
                        {"name": "Second Chance Lore", "entries": [{}, {}]}
                    ]
                }
            return record

        monkeypatch.setattr(
            character_handler_module, "fetch_character_by_id", fetch_with_book
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await screen._import_character_from_path(str(card_path))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any(
                message.startswith("Character imported.")
                and "Second Chance Lore" in message
                and "2 entries" in message
                and severity == "information"
                for message, severity in notifications
            )

    async def test_duplicate_copies_character_under_disambiguated_name(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """Task-443 AC2: characters gained a Duplicate seam (the library-rail
        button now shows in Characters mode too), reusing the same
        ``create_character`` seam a normal Save-as-new already calls -
        mirrors ``test_duplicate_copies_entries_and_strategy`` (dictionaries)
        and the lore equivalent."""
        created_payloads: list[dict] = []
        characters = [dict(c) for c in CHARACTERS]

        def fake_create(data):
            record = dict(data)
            record["id"] = 99
            record["version"] = 1
            characters.append(record)
            created_payloads.append(data)
            return 99

        monkeypatch.setattr(character_handler_module, "create_character", fake_create)
        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", lambda: list(characters)
        )
        monkeypatch.setattr(
            character_handler_module,
            "fetch_character_by_id",
            lambda character_id: next(
                (dict(c) for c in characters if str(c["id"]) == str(character_id)),
                None,
            ),
        )
        app = PersonasTestApp(mock_app_instance)
        # Button-press tests need a real-terminal-sized layout: at the
        # default 80x24 the center panels overlap the library toolbar's
        # coordinates, so a Duplicate click silently lands on the
        # character-dictionaries panel instead (same reason
        # TestConsoleActions runs at 160x50).
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            # Duplicate now applies to characters (task-443) - the library
            # rail button must be visible in the default characters mode.
            assert screen.query_one("#personas-library-duplicate", Button).display
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await pilot.click("#personas-library-duplicate")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(created_payloads) == 1
            payload = created_payloads[0]
            assert payload["name"] == "Detective Sam (copy)"
            assert payload["description"] == "Noir detective"
            assert payload["first_message"] == "The name's {{char}}. Who's asking?"
            assert payload["alternate_greetings"] == [
                "An alternate opener.",
                "A third opener.",
            ]
            assert screen.state.selected_entity_id == "99"
            rows = screen.query(".personas-library-row")
            assert "Detective Sam (copy)" in [_row_text(r) for r in rows]

    async def test_duplicate_name_conflict_notifies_error(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError

        def fake_create(data):
            raise ConflictError(
                "already exists", entity="character_cards", entity_id=data["name"]
            )

        monkeypatch.setattr(character_handler_module, "create_character", fake_create)
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        # 200x60: see test_duplicate_copies_character_under_disambiguated_name.
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await pilot.click("#personas-library-duplicate")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_id == "1"
            assert any(
                "already exists" in message and severity == "error"
                for message, severity in notifications
            )

    async def test_stage_character_avatar_from_path_updates_editor_and_dirty_state(
        self, mock_app_instance, stub_characters, tmp_path
    ):
        avatar = tmp_path / "avatar.png"
        avatar.write_bytes(b"\x89PNG staged avatar")
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()

            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert editor.get_character_data()["image"] == b"\x89PNG staged avatar"
            assert (
                str(
                    screen.query_one(
                        "#personas-char-editor-avatar-status", Static
                    ).renderable
                )
                == "Avatar: embedded"
            )
            assert screen.state.has_unsaved_changes is True

    async def test_stage_character_avatar_rejects_unsupported_extension_without_mutation(
        self, mock_app_instance, stub_characters, tmp_path
    ):
        bad = tmp_path / "avatar.txt"
        bad.write_text("not an image")
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()

            await screen._stage_character_avatar_from_path(str(bad))
            await pilot.pause()

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert "image" not in editor.get_character_data()
            assert screen.state.has_unsaved_changes is False
            assert any(
                "Unsupported avatar image type" in msg for msg, _ in notifications
            )

    async def test_stage_character_avatar_rejects_oversize_without_mutation(
        self, mock_app_instance, stub_characters, tmp_path
    ):
        avatar = tmp_path / "avatar.png"
        with avatar.open("wb") as avatar_file:
            avatar_file.truncate(personas_screen_module.PERSONAS_AVATAR_MAX_BYTES + 1)
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()

            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert "image" not in editor.get_character_data()
            assert screen.state.has_unsaved_changes is False
            assert any("5 MB or smaller" in msg for msg, _ in notifications)

    async def test_stage_character_avatar_drops_stale_read_after_editor_restarts(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        avatar = tmp_path / "avatar.png"
        avatar.write_bytes(b"\x89PNG original avatar")
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()

            async def fake_to_thread(function: Any, path: str) -> bytes:
                screen._finish_cancel_edit()
                await screen._begin_create_character()
                return b"\x89PNG stale avatar"

            monkeypatch.setattr(
                personas_screen_module.asyncio, "to_thread", fake_to_thread
            )

            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert "image" not in editor.get_character_data()
            assert screen.state.has_unsaved_changes is False
            assert not any("Avatar staged" in msg for msg, _ in notifications)

    async def test_stage_character_avatar_requires_open_editor(
        self, mock_app_instance, stub_characters, tmp_path
    ):
        avatar = tmp_path / "avatar.png"
        avatar.write_bytes(b"\x89PNG staged avatar")
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()

            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()

            assert screen.state.has_unsaved_changes is False
            assert any("Open a character editor" in msg for msg, _ in notifications)

    @staticmethod
    async def _open_persona_editor(pilot, mode: str):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-mode-personas")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        if mode == "create":
            await pilot.click("#personas-library-new")
            await pilot.pause()
        else:
            await pilot.click("#personas-library-row-persona-p-1")
            await pilot.pause()
            screen.post_message(EditPersonaProfileRequested("p-1"))
            await pilot.pause()
        assert screen._edit_mode == mode
        assert screen.state.has_unsaved_changes is False
        return screen

    @pytest.mark.parametrize("mode", ["create", "edit"])
    async def test_stage_character_avatar_ignores_persona_editor_session(
        self, mock_app_instance, stub_characters, stub_scope_service, tmp_path, mode
    ):
        avatar = tmp_path / f"persona-{mode}.png"
        avatar.write_bytes(b"\x89PNG persona editor avatar")
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await self._open_persona_editor(pilot, mode)

            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()

            editor = screen.query_one(PersonasCharacterEditorWidget)
            assert "image" not in editor.get_character_data()
            assert screen.state.has_unsaved_changes is False
            assert any("Open a character editor" in msg for msg, _ in notifications)

    @pytest.mark.parametrize("mode", ["create", "edit"])
    async def test_avatar_upload_request_ignores_persona_editor_session(
        self, mock_app_instance, stub_characters, stub_scope_service, mode
    ):
        calls: list[int] = []
        app = PersonasTestApp(mock_app_instance)
        notifications = TestImportExport._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await self._open_persona_editor(pilot, mode)

            def worker():
                calls.append(1)

                async def _noop():
                    pass

                return _noop()

            screen._avatar_upload_dialog_worker = worker
            screen.post_message(CharacterImageUploadRequested())
            await pilot.pause()
            await app.workers.wait_for_complete()

            assert calls == []
            assert screen._io_dialog_active is False
            assert screen.state.has_unsaved_changes is False
            assert any("Open a character editor" in msg for msg, _ in notifications)

    async def test_avatar_upload_request_launches_dialog_worker(
        self, mock_app_instance, stub_characters
    ):
        calls: list[int] = []
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()

            def worker():
                calls.append(1)

                async def _noop():
                    pass

                return _noop()

            screen._avatar_upload_dialog_worker = worker
            screen.post_message(CharacterImageUploadRequested())
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert calls == [1]

            screen.post_message(CharacterImageUploadRequested())
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert calls == [1]

    async def test_export_json_writes_file(
        self, mock_app_instance, stub_characters, stub_db, monkeypatch, tmp_path
    ):
        calls: list[tuple] = []

        def fake_export(db, character_id, include_image=True):
            calls.append((db, character_id, include_image))
            return '{"name": "Detective Sam"}'

        monkeypatch.setattr(
            personas_screen_module, "export_character_card_to_json", fake_export
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            target = tmp_path / "detective_sam.json"
            await screen._export_selected_character(str(target), fmt="json")
            await pilot.pause()
            assert calls and calls[0][0] is stub_db and calls[0][1] == 1
            assert target.read_text(encoding="utf-8") == '{"name": "Detective Sam"}'
            assert any(
                "Exported to" in message and severity == "information"
                for message, severity in notifications
            )

    async def test_export_png_delegates(
        self, mock_app_instance, stub_characters, stub_db, monkeypatch, tmp_path
    ):
        captured: dict = {}

        def fake_export_png(db, character_id, output_path, base_directory=None):
            captured.update(
                db=db,
                character_id=character_id,
                output_path=output_path,
                base_directory=base_directory,
            )
            return True

        monkeypatch.setattr(
            personas_screen_module, "export_character_card_to_png", fake_export_png
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            target = tmp_path / "detective_sam.png"
            await screen._export_selected_character(str(target), fmt="png")
            await pilot.pause()
            assert captured["db"] is stub_db
            assert captured["character_id"] == 1
            assert captured["output_path"] == str(target)
            assert any(
                "Exported to" in message and severity == "information"
                for message, severity in notifications
            )

    async def test_character_export_includes_voice_only_after_explicit_opt_in(
        self,
        mock_app_instance,
        stub_characters,
        stub_db,
        monkeypatch,
        tmp_path,
    ):
        exported_profiles: list[PortableTTSProfile | None] = []

        def fake_export(
            db,
            character_id,
            include_image=True,
            portable_tts_profile=None,
        ):
            assert db is stub_db
            assert character_id == 1
            exported_profiles.append(portable_tts_profile)
            return '{"spec": "chara_card_v2"}'

        monkeypatch.setattr(
            personas_screen_module,
            "export_character_card_to_json",
            fake_export,
        )
        profile = _character_tts_profile(1)
        portable = PortableTTSProfile(
            profile_id=profile.profile_id,
            draft=TTSProfileDraft(
                display_name=profile.display_name,
                provider_id=profile.provider_id,
                model_id=profile.model_id,
                voice_id=profile.voice_id,
                response_format=profile.response_format,
                speed=profile.speed,
                options=profile.options,
            ),
        )
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            monkeypatch.setattr(screen, "_queue_character_tts_refresh", lambda: None)
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            inspector = screen.query_one(PersonasInspectorPane)
            checkbox = inspector.query_one("#personas-export-include-tts", Checkbox)
            assert checkbox.value is False
            assert checkbox.disabled is True
            # task-2233: hidden while the character has no assigned profile.
            assert checkbox.display is False

            inspector.set_tts_export_available(True)
            assert checkbox.disabled is False
            # task-2233: an assignment makes it reappear.
            assert checkbox.display is True
            monkeypatch.setattr(
                screen,
                "_portable_tts_profile_for_export",
                lambda: portable,
            )

            await screen._export_selected_character(
                str(tmp_path / "ordinary.json"),
                fmt="json",
            )
            checkbox.value = True
            await screen._export_selected_character(
                str(tmp_path / "portable.json"),
                fmt="json",
            )

        assert exported_profiles == [None, portable]

    async def test_portable_import_preflights_before_character_write_and_commits(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        tmp_path,
    ):
        events: list[str] = []
        profile = _character_tts_profile(1)
        portable = PortableTTSProfile(
            profile_id=profile.profile_id,
            draft=TTSProfileDraft(
                display_name=profile.display_name,
                provider_id=profile.provider_id,
                model_id=profile.model_id,
                voice_id=profile.voice_id,
                response_format=profile.response_format,
                speed=profile.speed,
                options=profile.options,
            ),
        )
        observation = PortableProfileAvailabilityObservation(
            repository_generation=7,
            configuration_revision=3,
            profile=portable,
            availability="available",
        )
        plan = PortableProfileImportPlan(
            observation=observation,
            allowed_choices=("create",),
            reuse_profile=None,
            copy_candidate=portable,
        )
        character_ref = CharacterRef(
            source="local",
            authority_id="local-test-authority",
            character_id="1",
        )

        class _Service:
            async def observe_portable_profile(self, candidate):
                events.append("observe")
                assert candidate == portable
                return observation

            async def inspect_portable_profile_import(self, candidate):
                events.append("collisions")
                assert candidate == observation
                return plan

            async def get_assigned_profile(self, candidate):
                events.append("assignment")
                assert candidate == character_ref
                return LoadedCharacterTTSAssignment(7, None)

            async def commit_portable_profile_import(
                self,
                candidate_plan,
                choice,
                candidate_ref,
                *,
                expected_current,
            ):
                events.append("commit")
                assert (candidate_plan, choice, candidate_ref, expected_current) == (
                    plan,
                    "create",
                    character_ref,
                    None,
                )
                return PortableProfileImportResult(
                    created=True,
                    availability="available",
                    loaded=LoadedTTSProfile(7, profile),
                    assignment=CharacterTTSAssignment(
                        character_ref, profile.profile_id
                    ),
                )

        source = tmp_path / "portable-card.json"
        source.write_bytes(b"same immutable card bytes")
        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda source_bytes: (
                events.append("inspect") or CharacterCardTTSInspection(portable)
            ),
        )
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            lambda source_bytes: (
                events.append("character_write")
                or CharacterCardImportOutcome(1, True, portable, None)
            ),
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            monkeypatch.setattr(screen, "_queue_character_tts_refresh", lambda: None)
            monkeypatch.setattr(
                screen,
                "_character_tts_profile_service",
                AsyncMock(return_value=_Service()),
            )
            monkeypatch.setattr(
                screen,
                "_local_character_ref_for_import",
                AsyncMock(return_value=character_ref),
            )

            await screen._import_character_from_path(str(source))
            await pilot.pause()

        assert events == [
            "inspect",
            "observe",
            "character_write",
            "collisions",
            "assignment",
            "commit",
        ]
        assert any(
            "voice profile applied" in message.casefold()
            for message, _severity in notifications
        )

    async def test_portable_collision_cancellation_writes_no_profile_or_assignment(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        tmp_path,
    ):
        profile = _character_tts_profile(1)
        portable = PortableTTSProfile(
            profile.profile_id,
            TTSProfileDraft(
                display_name=profile.display_name,
                provider_id=profile.provider_id,
                model_id=profile.model_id,
                voice_id=profile.voice_id,
                response_format=profile.response_format,
                speed=profile.speed,
                options=profile.options,
            ),
        )
        observation = PortableProfileAvailabilityObservation(
            7,
            3,
            portable,
            "available",
        )
        plan = PortableProfileImportPlan(
            observation,
            ("reuse", "copy"),
            profile,
            PortableTTSProfile(
                UUID("33333333-3333-4333-8333-333333333333"),
                portable.draft,
            ),
        )
        events: list[str] = []
        commits: list[object] = []
        character_ref = CharacterRef(
            source="local",
            authority_id="local-test-authority",
            character_id="1",
        )

        class _Service:
            async def observe_portable_profile(self, _profile):
                return observation

            async def inspect_portable_profile_import(self, _observation):
                return plan

            async def get_assigned_profile(self, _character_ref):
                events.append("assignment_snapshot")
                return LoadedCharacterTTSAssignment(7, None)

            async def commit_portable_profile_import(self, *args, **kwargs):
                commits.append((args, kwargs))

        source = tmp_path / "collision.json"
        source.write_bytes(b"immutable collision card")
        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _bytes: CharacterCardTTSInspection(portable),
        )
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            lambda _bytes: CharacterCardImportOutcome(1, True, portable, None),
        )
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            monkeypatch.setattr(screen, "_queue_character_tts_refresh", lambda: None)
            monkeypatch.setattr(
                screen,
                "_character_tts_profile_service",
                AsyncMock(return_value=_Service()),
            )
            monkeypatch.setattr(
                screen,
                "_local_character_ref_for_import",
                AsyncMock(return_value=character_ref),
            )
            monkeypatch.setattr(
                screen,
                "_resolve_import_collision_choice",
                AsyncMock(
                    side_effect=lambda _plan: events.append("collision_choice") or None
                ),
            )

            await screen._import_character_from_path(str(source))

        assert events == ["assignment_snapshot", "collision_choice"]
        assert commits == []

    async def test_reused_character_requires_confirmation_before_profile_work(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        tmp_path,
    ):
        profile = _character_tts_profile(1)
        portable = PortableTTSProfile(
            profile.profile_id,
            TTSProfileDraft(
                display_name=profile.display_name,
                provider_id=profile.provider_id,
                model_id=profile.model_id,
                voice_id=profile.voice_id,
                response_format=profile.response_format,
                speed=profile.speed,
                options=profile.options,
            ),
        )
        observation = PortableProfileAvailabilityObservation(
            7, 3, portable, "available"
        )

        class _Service:
            async def observe_portable_profile(self, _profile):
                return observation

            async def inspect_portable_profile_import(self, _observation):
                raise AssertionError(
                    "declined apply must stop before profile collision reads"
                )

            async def get_assigned_profile(self, _character_ref):
                raise AssertionError("declined apply must preserve current assignment")

            async def commit_portable_profile_import(self, *args, **kwargs):
                raise AssertionError("declined apply must not commit")

        source = tmp_path / "reused.json"
        source.write_bytes(b"immutable reused card")
        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _bytes: CharacterCardTTSInspection(portable),
        )
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            lambda _bytes: CharacterCardImportOutcome(1, False, portable, None),
        )
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            monkeypatch.setattr(screen, "_queue_character_tts_refresh", lambda: None)
            monkeypatch.setattr(
                screen,
                "_character_tts_profile_service",
                AsyncMock(return_value=_Service()),
            )
            confirmation = AsyncMock(return_value=False)
            monkeypatch.setattr(
                screen,
                "_confirm_reused_character_tts_apply",
                confirmation,
            )

            await screen._import_character_from_path(str(source))

        confirmation.assert_awaited_once()

    async def test_unavailable_imported_voice_is_saved_for_repair_not_assigned(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        tmp_path,
    ):
        profile = _character_tts_profile(1)
        portable = _portable_tts_profile(profile)
        observation = PortableProfileAvailabilityObservation(
            7,
            3,
            portable,
            "unavailable",
        )
        plan = PortableProfileImportPlan(
            observation,
            ("create",),
            None,
            portable,
        )
        character_ref = CharacterRef(
            source="local",
            authority_id="local-test-authority",
            character_id="1",
        )
        commit_calls: list[tuple[object, ...]] = []

        class _Service:
            async def observe_portable_profile(self, _profile):
                return observation

            async def inspect_portable_profile_import(self, _observation):
                return plan

            async def get_assigned_profile(self, _character_ref):
                return LoadedCharacterTTSAssignment(7, None)

            async def commit_portable_profile_import(
                self,
                candidate_plan,
                choice,
                candidate_ref,
                *,
                expected_current,
            ):
                commit_calls.append(
                    (candidate_plan, choice, candidate_ref, expected_current)
                )
                return PortableProfileImportResult(
                    created=True,
                    availability="unavailable",
                    loaded=LoadedTTSProfile(7, profile),
                    assignment=None,
                )

        source = tmp_path / "unavailable.json"
        source.write_bytes(b"one immutable character card")
        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _bytes: CharacterCardTTSInspection(portable),
        )
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            lambda _bytes: CharacterCardImportOutcome(1, True, portable, None),
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            monkeypatch.setattr(screen, "_queue_character_tts_refresh", lambda: None)
            monkeypatch.setattr(
                screen,
                "_character_tts_profile_service",
                AsyncMock(return_value=_Service()),
            )
            monkeypatch.setattr(
                screen,
                "_local_character_ref_for_import",
                AsyncMock(return_value=character_ref),
            )

            await screen._import_character_from_path(str(source))

        assert commit_calls == [(plan, "create", character_ref, None)]
        assert any(
            "saved for repair" in message.casefold()
            and "not assigned" in message.casefold()
            and severity == "information"
            for message, severity in notifications
        )
        # Task-6c (TASK-2450 AC#9): this copy is only ever reached for a
        # genuinely unavailable profile now (an unverified one auto-applies
        # instead) -- it must say so plainly, not the vaguer "not currently
        # available" that used to also (inaccurately) describe "unverified".
        assert any(
            "unavailable" in message.casefold() for message, _severity in notifications
        )

    async def test_unverified_imported_voice_auto_applies_with_honest_copy(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        tmp_path,
    ):
        """Task-6c (TASK-2450 AC#8/#9): a legacy-provider imported profile is
        always observed as 'unverified' (never 'available') -- it must
        auto-apply, and the resulting toast must read as a success, never
        mention 'unavailable', and never launder the state as verified
        either (it simply reports the ordinary 'applied' outcome, matching
        the honesty convention Gap 2 established: the Voice Profiles
        library and Roleplay status line are where 'Unverified' is shown,
        not a one-shot import toast)."""

        profile = _character_tts_profile(1)
        portable = _portable_tts_profile(profile)
        observation = PortableProfileAvailabilityObservation(
            7,
            3,
            portable,
            "unverified",
        )
        plan = PortableProfileImportPlan(
            observation,
            ("create",),
            None,
            portable,
        )
        character_ref = CharacterRef(
            source="local",
            authority_id="local-test-authority",
            character_id="1",
        )
        commit_calls: list[tuple[object, ...]] = []

        class _Service:
            async def observe_portable_profile(self, _profile):
                return observation

            async def inspect_portable_profile_import(self, _observation):
                return plan

            async def get_assigned_profile(self, _character_ref):
                return LoadedCharacterTTSAssignment(7, None)

            async def commit_portable_profile_import(
                self,
                candidate_plan,
                choice,
                candidate_ref,
                *,
                expected_current,
            ):
                commit_calls.append(
                    (candidate_plan, choice, candidate_ref, expected_current)
                )
                return PortableProfileImportResult(
                    created=True,
                    availability="unverified",
                    loaded=LoadedTTSProfile(7, profile),
                    assignment=CharacterTTSAssignment(
                        candidate_ref, profile.profile_id
                    ),
                )

        source = tmp_path / "unverified.json"
        source.write_bytes(b"one immutable character card")
        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _bytes: CharacterCardTTSInspection(portable),
        )
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            lambda _bytes: CharacterCardImportOutcome(1, True, portable, None),
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            monkeypatch.setattr(screen, "_queue_character_tts_refresh", lambda: None)
            monkeypatch.setattr(
                screen,
                "_character_tts_profile_service",
                AsyncMock(return_value=_Service()),
            )
            monkeypatch.setattr(
                screen,
                "_local_character_ref_for_import",
                AsyncMock(return_value=character_ref),
            )

            await screen._import_character_from_path(str(source))

        assert commit_calls == [(plan, "create", character_ref, None)]
        assert any(
            "applied successfully" in message.casefold() and severity == "information"
            for message, severity in notifications
        )
        assert all(
            "unavailable" not in message.casefold()
            for message, _severity in notifications
        )

    async def test_unavailable_reused_voice_reports_new_character_is_unassigned(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        tmp_path,
    ):
        profile = _character_tts_profile(1)
        portable = _portable_tts_profile(profile)
        observation = PortableProfileAvailabilityObservation(
            7,
            3,
            portable,
            "unavailable",
        )
        plan = PortableProfileImportPlan(
            observation,
            ("reuse", "copy"),
            profile,
            _portable_tts_profile(
                profile,
                profile_id=UUID("33333333-3333-4333-8333-333333333333"),
            ),
        )
        character_ref = CharacterRef(
            source="local",
            authority_id="local-test-authority",
            character_id="1",
        )

        class _Service:
            async def observe_portable_profile(self, _profile):
                return observation

            async def inspect_portable_profile_import(self, _observation):
                return plan

            async def get_assigned_profile(self, _character_ref):
                return LoadedCharacterTTSAssignment(7, None)

            async def commit_portable_profile_import(self, *args, **kwargs):
                del args, kwargs
                return PortableProfileImportResult(
                    created=False,
                    availability="unavailable",
                    loaded=LoadedTTSProfile(7, profile),
                    assignment=None,
                )

        source = tmp_path / "unavailable-reuse.json"
        source.write_bytes(b"one immutable character card")
        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _bytes: CharacterCardTTSInspection(portable),
        )
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            lambda _bytes: CharacterCardImportOutcome(1, True, portable, None),
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            monkeypatch.setattr(screen, "_queue_character_tts_refresh", lambda: None)
            monkeypatch.setattr(
                screen,
                "_character_tts_profile_service",
                AsyncMock(return_value=_Service()),
            )
            monkeypatch.setattr(
                screen,
                "_local_character_ref_for_import",
                AsyncMock(return_value=character_ref),
            )
            monkeypatch.setattr(
                screen,
                "_resolve_import_collision_choice",
                AsyncMock(return_value="reuse"),
            )

            await screen._import_character_from_path(str(source))

        matching = [
            message.casefold()
            for message, _severity in notifications
            if "voice" in message.casefold()
        ]
        assert any("remains unassigned" in message for message in matching)
        assert all(
            "existing voice assignment was preserved" not in message
            for message in matching
        )
        # Task-6c (TASK-2450 AC#9): same honesty pin as the saved-for-repair
        # case -- only reachable for a genuinely unavailable profile now.
        assert any("unavailable" in message for message in matching)

    async def test_profile_commit_failure_keeps_character_and_hides_sensitive_detail(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        tmp_path,
    ):
        from loguru import logger as loguru_logger

        secret = "credential-private-origin-message-text"
        profile = _character_tts_profile(1)
        portable = _portable_tts_profile(profile)
        observation = PortableProfileAvailabilityObservation(
            7,
            3,
            portable,
            "available",
        )
        plan = PortableProfileImportPlan(
            observation,
            ("create",),
            None,
            portable,
        )
        character_writes: list[bytes] = []

        class _Service:
            async def observe_portable_profile(self, _profile):
                return observation

            async def inspect_portable_profile_import(self, _observation):
                return plan

            async def get_assigned_profile(self, _character_ref):
                return LoadedCharacterTTSAssignment(7, None)

            async def commit_portable_profile_import(self, *args, **kwargs):
                del args, kwargs
                raise RuntimeError(secret)

        private_dir = tmp_path / secret
        private_dir.mkdir()
        source = private_dir / "card.json"
        source.write_bytes(b"immutable card with private roleplay text")
        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _bytes: CharacterCardTTSInspection(portable),
        )

        def persist_character(source_bytes):
            character_writes.append(source_bytes)
            return CharacterCardImportOutcome(1, True, portable, None)

        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            persist_character,
        )
        mock_app_instance.chat_dictionary_scope_service = SimpleNamespace(
            list_character_dictionaries=AsyncMock(
                return_value={"dictionaries": []}
            )
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        log_messages: list[str] = []
        sink = loguru_logger.add(
            lambda message: log_messages.append(str(message)),
            level="DEBUG",
        )
        try:
            async with app.run_test() as pilot:
                screen = await _mounted(pilot)
                fetch_calls = 0

                def fetch_after_import(character_id):
                    nonlocal fetch_calls
                    fetch_calls += 1
                    if fetch_calls == 1:
                        raise RuntimeError(secret)
                    return next(
                        dict(character)
                        for character in CHARACTERS
                        if str(character["id"]) == str(character_id)
                    )

                monkeypatch.setattr(
                    character_handler_module,
                    "fetch_character_by_id",
                    fetch_after_import,
                )
                monkeypatch.setattr(
                    screen,
                    "_queue_character_tts_refresh",
                    lambda: None,
                )
                monkeypatch.setattr(
                    screen,
                    "_character_tts_profile_service",
                    AsyncMock(return_value=_Service()),
                )
                monkeypatch.setattr(
                    screen,
                    "_local_character_ref_for_import",
                    AsyncMock(
                        return_value=CharacterRef(
                            source="local",
                            authority_id="local-test-authority",
                            character_id="1",
                        )
                    ),
                )

                await screen._import_character_from_path(str(source))
        finally:
            loguru_logger.remove(sink)

        assert character_writes == [source.read_bytes()]
        assert any(
            "character was kept" in message.casefold()
            and "could not be saved or assigned" in message.casefold()
            and severity == "warning"
            for message, severity in notifications
        )
        observable_copy = " ".join(
            [*(message for message, _severity in notifications), *log_messages]
        )
        assert secret not in observable_copy
        assert str(source) not in observable_copy

    async def test_export_json_rejects_hidden_directory_destination(
        self, mock_app_instance, stub_characters, stub_db, monkeypatch, tmp_path
    ):
        """The JSON write path validates the destination like the PNG path."""
        monkeypatch.setattr(
            personas_screen_module,
            "export_character_card_to_json",
            lambda db, character_id, include_image=True: '{"name": "Detective Sam"}',
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            hidden_dir = tmp_path / ".sneaky"
            hidden_dir.mkdir()
            target = hidden_dir / "out.json"
            await screen._export_selected_character(str(target), fmt="json")
            await pilot.pause()
            assert not target.exists()
            assert any(
                "Export failed" in message and severity == "error"
                for message, severity in notifications
            )

    async def test_export_json_rejects_missing_destination_directory(
        self, mock_app_instance, stub_characters, stub_db, monkeypatch, tmp_path
    ):
        """A destination in a nonexistent directory fails readably."""
        monkeypatch.setattr(
            personas_screen_module,
            "export_character_card_to_json",
            lambda db, character_id, include_image=True: '{"name": "Detective Sam"}',
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            target = tmp_path / "missing" / "out.json"
            await screen._export_selected_character(str(target), fmt="json")
            await pilot.pause()
            assert not target.exists()
            assert any(
                "Export failed" in message and severity == "error"
                for message, severity in notifications
            )

    async def test_export_profile_json(
        self, mock_app_instance, stub_characters, stub_scope_service, tmp_path
    ):
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-library-row-persona-p-1")
            await pilot.pause()
            target = tmp_path / "archivist.json"
            await screen._export_selected_character(str(target), fmt="json")
            await pilot.pause()
            assert "Archivist" in target.read_text(encoding="utf-8")
            assert any(
                "Exported to" in message and severity == "information"
                for message, severity in notifications
            )

    async def test_import_requires_characters_mode(
        self, mock_app_instance, stub_characters, stub_scope_service, monkeypatch
    ):
        import_card = Mock()
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            import_card,
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen.post_message(PersonaActionRequested(action="import"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            import_card.assert_not_called()
            assert not any(severity == "error" for _, severity in notifications)


class _FtsStubDB:
    """Captures the MATCH term handed to search_character_cards.

    TASK-19558: `search_character_cards` used to compute a `safe_search_term`
    and bind the RAW one, so a caller-built prefix expression reached MATCH
    only through the plain-text parameter. The plain-text parameter now
    quotes what it is given, and a caller-built expression travels through
    `fts_match_query` -- which is what `calls` records here, so these tests
    still assert on the expression SQLite actually sees.
    """

    def __init__(self):
        self.calls: list[tuple[str, int]] = []
        self.plain_terms: list[str] = []

    def search_character_cards(self, search_term, limit=10, fts_match_query=None):
        self.plain_terms.append(search_term)
        self.calls.append((fts_match_query, limit))
        return [{"id": 1, "name": "Match"}]


_CURSOR_OMITTED = object()


def _conversation_record(index: int, *, title: str | None = None) -> dict[str, Any]:
    """Return the complete conversation shape the mounted inspector consumes."""
    return {
        "id": f"conv-{index}",
        "title": title or f"Case {index}",
        "last_modified": (
            datetime(2026, 8, 27, 12, tzinfo=UTC) - timedelta(minutes=index)
        ).isoformat(),
    }


class _ConversationPageDB:
    """Cursor-aware character DB double at the screen's production seam."""

    def __init__(self, *pages: object) -> None:
        self.pages = list(pages) or [[]]
        self.calls: list[tuple[int, int, int, dict[str, object]]] = []

    def replace_pages(self, *pages: object) -> None:
        """Replace queued responses; the last response repeats if read again."""
        self.pages = list(pages) or [[]]

    def get_character_card_by_id(self, character_id: int) -> dict[str, Any] | None:
        """Support the mounted inspector's sibling portrait read."""
        return next(
            (deepcopy(row) for row in CHARACTERS if row["id"] == character_id),
            None,
        )

    def get_conversations_for_character(
        self,
        character_id: int,
        limit: int = 50,
        offset: int = 0,
        *,
        before_last_modified: object = _CURSOR_OMITTED,
        before_id: object = _CURSOR_OMITTED,
    ) -> list[dict[str, Any]]:
        cursor: dict[str, object] = {}
        if before_last_modified is not _CURSOR_OMITTED:
            cursor["before_last_modified"] = before_last_modified
        if before_id is not _CURSOR_OMITTED:
            cursor["before_id"] = before_id
        self.calls.append((character_id, limit, offset, cursor))
        response = self.pages.pop(0) if len(self.pages) > 1 else self.pages[0]
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            response = response(
                character_id,
                limit,
                offset,
                **cursor,
            )
        return deepcopy(response)


def _install_conversation_db(
    monkeypatch: pytest.MonkeyPatch, *pages: object
) -> _ConversationPageDB:
    db = _ConversationPageDB(*pages)
    monkeypatch.setattr(PersonasScreen, "_character_db", lambda self: db)
    return db


class TestFtsTermSafety:
    """Unit tests for the MATCH expression built by search_characters_fts."""

    @pytest.fixture
    def stub_db(self, monkeypatch):
        stub = _FtsStubDB()
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: stub
        )
        return stub

    async def test_normal_term_becomes_quoted_prefix_query(self, stub_db):
        results = character_handler_module.search_characters_fts("sam")
        assert [term for term, _ in stub_db.calls] == ['"sam"*']
        assert results and results[0]["name"] == "Match"
        # The raw term is still passed positionally, but when
        # `fts_match_query` is supplied `search_term` is UNUSED by
        # `search_character_cards` -- not even in its error message, which
        # reports the expression that was actually run. Recorded here only so
        # a future change that starts using it is visible.
        assert stub_db.plain_terms == ["sam"]

    async def test_apostrophe_term_is_safe(self, stub_db):
        character_handler_module.search_characters_fts("O'Brien")
        assert [term for term, _ in stub_db.calls] == ['"O\'Brien"*']

    async def test_embedded_double_quote_is_escaped(self, stub_db):
        character_handler_module.search_characters_fts('sam"')
        assert [term for term, _ in stub_db.calls] == ['"sam"""*']

    async def test_fts_operator_characters_are_quoted(self, stub_db):
        character_handler_module.search_characters_fts("(")
        assert [term for term, _ in stub_db.calls] == ['"("*']
        character_handler_module.search_characters_fts("sam-")
        assert [term for term, _ in stub_db.calls][-1] == '"sam-"*'

    async def test_empty_term_returns_empty_without_db_call(self, stub_db):
        assert character_handler_module.search_characters_fts("") == []
        assert character_handler_module.search_characters_fts("   ") == []
        assert stub_db.calls == []


class _NavCaptureApp(PersonasTestApp):
    """Test app that records NavigateToScreen routes bubbled from the screen."""

    def __init__(self, mock_app_instance):
        super().__init__(mock_app_instance)
        self.nav_routes: list[str] = []
        self.nav_contexts: list[dict[str, object]] = []

    def on_navigate_to_screen(self, message) -> None:
        self.nav_routes.append(message.screen_name)
        self.nav_contexts.append(dict(getattr(message, "screen_context", {}) or {}))


class _StyledNavCaptureApp(_NavCaptureApp):
    """Navigation-capture harness using the production consolidated CSS."""

    CSS_PATH = StyledPersonasTestApp.CSS_PATH


class TestConversationsPanel:
    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        """Stub the DB resolver, conversation listing, and message retrieval."""
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        db = _install_conversation_db(
            monkeypatch, [_conversation_record(1, title="First case")]
        )
        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            lambda db, conversation_id, character_name, user_name, **kwargs: [
                ("Hello there", "Greetings, detective."),
            ],
        )
        return db

    async def _select_first_character(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def _open_conversation(self, pilot):
        screen = await self._select_first_character(pilot)
        await pilot.click("#personas-conversation-row-conv-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def test_selecting_character_lists_conversations(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            rows = screen.query(".personas-conversation-row")
            assert [_row_text(r) for r in rows] == ["First case"]

    async def test_first_page_uses_sentinel_without_rendering_it(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        page = [_conversation_record(index) for index in range(1, 22)]
        stub_conversations.replace_pages(page)
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert stub_conversations.calls == [(1, 21, 0, {})]
            rows = list(screen.query(".personas-conversation-row"))
            assert [_row_text(row) for row in rows] == [
                f"Case {index}" for index in range(1, 21)
            ]
            assert not screen.query("#personas-conversation-row-conv-21")
            tail = screen.query_one(".personas-conversations-tail")
            assert _row_text(tail) == "Load 20 older conversations"

    async def test_enter_loads_next_page_from_twentieth_visible_cursor(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        first_page = [_conversation_record(index) for index in range(1, 22)]
        second_page = [_conversation_record(index) for index in range(21, 42)]
        stub_conversations.replace_pages(first_page, second_page)
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            conversation_list = screen.query_one(
                "#personas-conversations-list", ListView
            )
            conversation_list.focus()
            conversation_list.index = len(conversation_list.children) - 1
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            twentieth = _conversation_record(20)
            assert stub_conversations.calls == [
                (1, 21, 0, {}),
                (
                    1,
                    21,
                    0,
                    {
                        "before_last_modified": twentieth["last_modified"],
                        "before_id": twentieth["id"],
                    },
                ),
            ]
            rows = list(screen.query(".personas-conversation-row"))
            assert [_row_text(row) for row in rows] == [
                f"Case {index}" for index in range(1, 41)
            ]
            assert _row_text(rows[20]) == "Case 21"
            assert not screen.query("#personas-conversation-row-conv-41")

    async def test_appended_row_keeps_preview_and_all_conversation_actions(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        first_page = [_conversation_record(index) for index in range(1, 22)]
        second_page = [_conversation_record(21), _conversation_record(22)]
        stub_conversations.replace_pages(first_page, second_page)
        app = _NavCaptureApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.click("#personas-conversation-row-conv-1")
            await pilot.app.workers.wait_for_complete()
            assert screen.conversations._open_conversation_id == "conv-1"

            conversation_list = screen.query_one(
                "#personas-conversations-list", ListView
            )
            conversation_list.focus()
            conversation_list.index = len(conversation_list.children) - 1
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.conversations._open_conversation_id == "conv-1"
            assert screen.query_one(
                "#personas-conversation-transcript-view"
            ).display

            oldest = screen.query_one("#personas-conversation-row-conv-22")
            conversation_list.index = list(conversation_list.children).index(oldest)
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert screen.conversations._open_conversation_id == "conv-22"
            assert str(
                screen.query_one("#personas-transcript-title", Static).renderable
            ) == "Case 22"
            assert str(
                screen.query_one("#personas-conversation-resume", Button).label
            ) == "Resume chat"
            assert str(
                screen.query_one(
                    "#personas-conversation-continue-console", Button
                ).label
            ) == "Send transcript to Console draft"
            assert str(
                screen.query_one(
                    "#personas-conversation-open-library", Button
                ).label
            ) == "Open in Library"

            await pilot.click("#personas-conversation-resume")
            await pilot.click("#personas-conversation-continue-console")
            await pilot.click("#personas-conversation-open-library")
            await pilot.pause()

        assert app.nav_routes == [TAB_CHAT, TAB_LIBRARY]
        assert app.nav_contexts[0] == {
            CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "conv-22"
        }
        assert app.nav_contexts[1] == {
            LIBRARY_NAV_CONTEXT_MODE: LIBRARY_MODE_CONVERSATIONS,
            LIBRARY_NAV_CONTEXT_CONVERSATION_ID: "conv-22",
        }
        app.open_chat_with_handoff.assert_called_once()

    async def test_append_loading_is_single_flight_for_repeated_enter(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        started = asyncio.Event()
        release = threading.Event()
        app = PersonasTestApp(mock_app_instance)

        def gated_page(character_id, limit=50, offset=0, **cursor):
            app.call_from_thread(started.set)
            release.wait(timeout=5)
            return [_conversation_record(21)]

        stub_conversations.replace_pages(
            [_conversation_record(index) for index in range(1, 22)], gated_page
        )
        try:
            async with app.run_test(size=(160, 50)) as pilot:
                screen = await _mounted(pilot)
                await pilot.app.workers.wait_for_complete()
                conversation_list = screen.query_one(
                    "#personas-conversations-list", ListView
                )
                conversation_list.focus()
                conversation_list.index = len(conversation_list.children) - 1
                await pilot.press("enter")
                await wait_for_signal(started, what="the gated older-page read")

                tail = screen.query_one(".personas-conversations-tail")
                assert _row_text(tail) == "Loading older conversations..."
                await pilot.press("enter")
                await pilot.pause()
                assert len(stub_conversations.calls) == 2

                release.set()
                await pilot.app.workers.wait_for_complete()
        finally:
            release.set()

    async def test_initial_loading_render_exception_becomes_keyboard_retry(
        self,
        mock_app_instance,
        stub_characters,
        stub_conversations,
        monkeypatch,
    ):
        stub_conversations.replace_pages([_conversation_record(90)])
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            calls_before = list(stub_conversations.calls)
            inspector = screen.query_one(PersonasInspectorPane)
            original_loading = inspector.show_conversations_loading
            fail_once = True

            async def one_shot_loading_failure(render_attempt=None):
                nonlocal fail_once
                rendered = await original_loading(render_attempt)
                if fail_once:
                    fail_once = False
                    raise RuntimeError("initial loading render failed")
                return rendered

            monkeypatch.setattr(
                inspector, "show_conversations_loading", one_shot_loading_failure
            )

            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()

            assert stub_conversations.calls == calls_before
            assert screen.conversations._conversation_list_attempt is None
            assert screen.conversations._conversation_list_phase == "initial-retry"
            tail = screen.query_one(".personas-conversations-tail")
            assert "Retry conversations" in _row_text(tail)
            assert not tail.disabled

            conversation_list = screen.query_one(
                "#personas-conversations-list", ListView
            )
            conversation_list.focus()
            conversation_list.index = 0
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert stub_conversations.calls[-1] == (2, 21, 0, {})
            assert [_row_text(row) for row in screen.query(".personas-conversation-row")] == [
                "Case 90"
            ]

    async def test_append_loading_render_exception_preserves_boundary_for_retry(
        self,
        mock_app_instance,
        stub_characters,
        stub_conversations,
        monkeypatch,
    ):
        first_page = [_conversation_record(index) for index in range(1, 22)]
        stub_conversations.replace_pages(first_page, [_conversation_record(21)])
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            inspector = screen.query_one(PersonasInspectorPane)
            original_loading = inspector.show_older_conversations_loading
            fail_once = True

            async def one_shot_loading_failure(render_attempt=None):
                nonlocal fail_once
                rendered = await original_loading(render_attempt)
                if fail_once:
                    fail_once = False
                    raise RuntimeError("append loading render failed")
                return rendered

            monkeypatch.setattr(
                inspector,
                "show_older_conversations_loading",
                one_shot_loading_failure,
            )
            cursor = screen.conversations._next_conversation_cursor
            rows_before = dict(screen.conversations._conversation_rows)
            ids_before = set(screen.conversations._loaded_conversation_ids)
            conversation_list = screen.query_one(
                "#personas-conversations-list", ListView
            )
            conversation_list.focus()
            conversation_list.index = len(conversation_list.children) - 1

            await pilot.press("enter")
            await pilot.pause()

            assert len(stub_conversations.calls) == 1
            assert screen.conversations._conversation_list_attempt is None
            assert screen.conversations._conversation_list_phase == "append-retry"
            assert screen.conversations._conversation_rows == rows_before
            assert screen.conversations._loaded_conversation_ids == ids_before
            assert screen.conversations._next_conversation_cursor == cursor
            tail = screen.query_one(".personas-conversations-tail")
            assert "Retry older conversations" in _row_text(tail)
            assert not tail.disabled

            conversation_list.index = len(conversation_list.children) - 1
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert stub_conversations.calls[-1] == (
                1,
                21,
                0,
                {
                    "before_last_modified": cursor[0],
                    "before_id": cursor[1],
                },
            )
            assert screen.conversations._next_conversation_cursor != cursor
            assert screen.query_one("#personas-conversation-row-conv-21")

    async def test_append_result_render_exception_rolls_back_dom_and_page_state(
        self,
        mock_app_instance,
        stub_characters,
        stub_conversations,
        monkeypatch,
    ):
        first_page = [_conversation_record(index) for index in range(1, 22)]
        stub_conversations.replace_pages(first_page, [_conversation_record(21)])
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            inspector = screen.query_one(PersonasInspectorPane)
            original_append = inspector.append_conversations
            fail_once = True

            async def one_shot_result_failure(*args, **kwargs):
                nonlocal fail_once
                rendered = await original_append(*args, **kwargs)
                if fail_once:
                    fail_once = False
                    raise RuntimeError("append result render failed")
                return rendered

            monkeypatch.setattr(
                inspector, "append_conversations", one_shot_result_failure
            )
            cursor = screen.conversations._next_conversation_cursor
            rows_before = dict(screen.conversations._conversation_rows)
            ids_before = set(screen.conversations._loaded_conversation_ids)
            conversation_list = screen.query_one(
                "#personas-conversations-list", ListView
            )
            conversation_list.focus()
            conversation_list.index = len(conversation_list.children) - 1

            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert screen.conversations._conversation_list_attempt is None
            assert screen.conversations._conversation_list_phase == "append-retry"
            assert screen.conversations._conversation_rows == rows_before
            assert screen.conversations._loaded_conversation_ids == ids_before
            assert screen.conversations._next_conversation_cursor == cursor
            assert not screen.query("#personas-conversation-row-conv-21")
            assert "Retry older conversations" in _row_text(
                screen.query_one(".personas-conversations-tail")
            )

            conversation_list.index = len(conversation_list.children) - 1
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            expected_cursor = {
                "before_last_modified": cursor[0],
                "before_id": cursor[1],
            }
            assert stub_conversations.calls[1:] == [
                (1, 21, 0, expected_cursor),
                (1, 21, 0, expected_cursor),
            ]
            assert list(screen.query("#personas-conversation-row-conv-21"))
            assert "conv-21" in screen.conversations._loaded_conversation_ids

    async def test_db_failure_retries_retry_tail_render_once(
        self,
        mock_app_instance,
        stub_characters,
        stub_conversations,
        monkeypatch,
    ):
        from loguru import logger as loguru_logger

        read_started = asyncio.Event()
        release_read = threading.Event()
        app = PersonasTestApp(mock_app_instance)
        retry_records = []
        sink_id = loguru_logger.add(
            lambda message: retry_records.append(message.record),
            filter=lambda record: record["message"]
            == "Could not render the conversations retry state.",
        )

        def gated_failure(character_id, limit=50, offset=0, **cursor):
            app.call_from_thread(read_started.set)
            release_read.wait(timeout=5)
            raise RuntimeError("database failed")

        stub_conversations.replace_pages(
            gated_failure, [_conversation_record(91, title="Retry succeeded")]
        )
        try:
            async with app.run_test(size=(160, 50)) as pilot:
                screen = await _mounted(pilot)
                await wait_for_signal(read_started, what="the gated failed DB read")
                inspector = screen.query_one(PersonasInspectorPane)
                original_failure = inspector.show_conversations_failure
                fail_once = True

                async def one_shot_retry_tail_failure(*args, **kwargs):
                    nonlocal fail_once
                    if fail_once:
                        fail_once = False
                        raise RuntimeError("retry tail render failed")
                    return await original_failure(*args, **kwargs)

                monkeypatch.setattr(
                    inspector,
                    "show_conversations_failure",
                    one_shot_retry_tail_failure,
                )
                release_read.set()
                await pilot.app.workers.wait_for_complete()
                await pilot.pause()

                assert screen.conversations._conversation_list_attempt is None
                assert screen.conversations._conversation_list_phase == "initial-retry"
                tail = screen.query_one(".personas-conversations-tail")
                assert "Retry conversations" in _row_text(tail)
                assert not tail.disabled

                conversation_list = screen.query_one(
                    "#personas-conversations-list", ListView
                )
                conversation_list.focus()
                conversation_list.index = 0
                await pilot.press("enter")
                await pilot.app.workers.wait_for_complete()
                await pilot.pause()

                assert stub_conversations.calls == [
                    (1, 21, 0, {}),
                    (1, 21, 0, {}),
                ]
                assert _row_text(
                    screen.query_one("#personas-conversation-row-conv-91")
                ) == "Retry succeeded"
        finally:
            release_read.set()
            loguru_logger.remove(sink_id)

        assert len(retry_records) == 1
        expected_context = {
            "character_id": "1",
            "cursor": None,
            "phase": "initial-retry",
            "operation": "render-owned-retry",
        }
        assert {
            key: retry_records[0]["extra"][key] for key in expected_context
        } == expected_context

    async def test_initial_and_append_failures_retry_the_identical_boundary(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        first_page = [_conversation_record(index) for index in range(1, 22)]
        stub_conversations.replace_pages(RuntimeError("initial failed"), first_page)
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            conversation_list = screen.query_one(
                "#personas-conversations-list", ListView
            )
            assert list(screen.query(".personas-conversation-row")) == []
            assert "Retry conversations" in _row_text(
                screen.query_one(".personas-conversations-tail")
            )

            conversation_list.focus()
            conversation_list.index = 0
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            assert stub_conversations.calls[:2] == [
                (1, 21, 0, {}),
                (1, 21, 0, {}),
            ]

            append_cursor = {
                "before_last_modified": _conversation_record(20)["last_modified"],
                "before_id": "conv-20",
            }
            stub_conversations.replace_pages(
                RuntimeError("append failed"), [_conversation_record(21)]
            )
            conversation_list.index = len(conversation_list.children) - 1
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            rows_before_retry = [
                _row_text(row) for row in screen.query(".personas-conversation-row")
            ]
            assert rows_before_retry == [f"Case {index}" for index in range(1, 21)]
            assert "Retry older conversations" in _row_text(
                screen.query_one(".personas-conversations-tail")
            )

            conversation_list.index = len(conversation_list.children) - 1
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert stub_conversations.calls[2:] == [
                (1, 21, 0, append_cursor),
                (1, 21, 0, append_cursor),
            ]
            assert [
                _row_text(row) for row in screen.query(".personas-conversation-row")
            ] == [f"Case {index}" for index in range(1, 22)]

    @pytest.mark.parametrize(
        ("page", "expected_rows", "tail_copy"),
        (
            ([], [], "No saved conversations."),
            (
                [_conversation_record(1), _conversation_record(2)],
                ["Case 1", "Case 2"],
                "All conversations shown.",
            ),
        ),
        ids=("empty", "exhausted"),
    )
    async def test_successful_bounded_first_page_has_explicit_terminal_state(
        self,
        mock_app_instance,
        stub_characters,
        stub_conversations,
        page,
        expected_rows,
        tail_copy,
    ):
        stub_conversations.replace_pages(page)
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert [
                _row_text(row) for row in screen.query(".personas-conversation-row")
            ] == expected_rows
            assert _row_text(
                screen.query_one(".personas-conversations-tail")
            ) == tail_copy

    async def test_duplicate_shadow_page_auto_advances_to_unseen_rows(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        first_page = [_conversation_record(index) for index in range(1, 22)]
        retained_cursor_time = datetime.fromisoformat(first_page[19]["last_modified"])
        duplicate_shadow = []
        for index in range(1, 21):
            duplicate = _conversation_record(index)
            duplicate["last_modified"] = (
                retained_cursor_time - timedelta(seconds=index)
            ).isoformat()
            duplicate_shadow.append(duplicate)
        duplicate_shadow.append(_conversation_record(21))
        continued_page = [_conversation_record(index) for index in range(21, 42)]
        continued_read_started = asyncio.Event()
        app = PersonasTestApp(mock_app_instance)

        def continued_read(character_id, limit=50, offset=0, **cursor):
            app.call_from_thread(continued_read_started.set)
            return continued_page

        stub_conversations.replace_pages(
            first_page, duplicate_shadow, continued_read
        )

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            conversation_list = screen.query_one(
                "#personas-conversations-list", ListView
            )
            conversation_list.focus()
            conversation_list.index = len(conversation_list.children) - 1
            await pilot.press("enter")
            await wait_for_signal(
                continued_read_started, what="the duplicate-shadow continuation"
            )
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            original_twentieth = first_page[19]
            raw_shadow_boundary = duplicate_shadow[19]
            assert stub_conversations.calls == [
                (1, 21, 0, {}),
                (
                    1,
                    21,
                    0,
                    {
                        "before_last_modified": original_twentieth["last_modified"],
                        "before_id": original_twentieth["id"],
                    },
                ),
                (
                    1,
                    21,
                    0,
                    {
                        "before_last_modified": raw_shadow_boundary[
                            "last_modified"
                        ],
                        "before_id": raw_shadow_boundary["id"],
                    },
                ),
            ]
            rows = list(screen.query(".personas-conversation-row"))
            assert [_row_text(row) for row in rows] == [
                f"Case {index}" for index in range(1, 41)
            ]
            assert len({row.id for row in rows}) == 40
            assert _row_text(
                screen.query_one(".personas-conversations-tail")
            ) == "Load 20 older conversations"

    async def test_duplicate_shadow_auto_traversal_yields_at_hop_budget(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        first_page = [_conversation_record(index) for index in range(1, 22)]
        retained_cursor_time = datetime.fromisoformat(first_page[19]["last_modified"])
        expected_auto_hops = (
            conversations_controller_module._CONVERSATIONS_MAX_AUTO_HOPS
        )
        duplicate_reads = 0
        generated_boundaries: list[tuple[str, str]] = []
        unbounded_read_started = asyncio.Event()
        abort_unbounded_read = threading.Event()
        app = PersonasTestApp(mock_app_instance)

        def duplicate_page(read_number: int) -> list[dict[str, Any]]:
            page = []
            for index in range(1, 21):
                duplicate = _conversation_record(index)
                duplicate["last_modified"] = (
                    retained_cursor_time
                    - timedelta(hours=read_number, seconds=index)
                ).isoformat()
                page.append(duplicate)
            page.append(_conversation_record(1))
            generated_boundaries.append(
                (page[19]["last_modified"], page[19]["id"])
            )
            return page

        def moving_duplicates(character_id, limit=50, offset=0, **cursor):
            nonlocal duplicate_reads
            duplicate_reads += 1
            page = duplicate_page(duplicate_reads)
            if duplicate_reads > expected_auto_hops + 1:
                app.call_from_thread(unbounded_read_started.set)
                abort_unbounded_read.wait()
                raise RuntimeError("test stopped unbounded duplicate traversal")
            return page

        stub_conversations.replace_pages(first_page, moving_duplicates)

        try:
            async with app.run_test(size=(160, 50)) as pilot:
                screen = await _mounted(pilot)
                await pilot.app.workers.wait_for_complete()
                conversation_list = screen.query_one(
                    "#personas-conversations-list", ListView
                )
                conversation_list.focus()
                conversation_list.index = len(conversation_list.children) - 1
                await pilot.press("enter")

                workers_done = asyncio.create_task(
                    pilot.app.workers.wait_for_complete()
                )
                unbounded = asyncio.create_task(unbounded_read_started.wait())
                done, pending = await asyncio.wait(
                    {workers_done, unbounded},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                try:
                    assert workers_done in done, (
                        "duplicate-only traversal scheduled beyond its hop budget"
                    )
                finally:
                    abort_unbounded_read.set()
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                await pilot.app.workers.wait_for_complete()
                await pilot.pause()

                latest_safe_boundary = generated_boundaries[-1]
                assert duplicate_reads == expected_auto_hops + 1
                assert screen.conversations._next_conversation_cursor == (
                    latest_safe_boundary
                )
                assert screen.conversations._conversation_list_attempt is None
                assert screen.conversations._conversation_list_phase == "ready"
                assert _row_text(
                    screen.query_one(".personas-conversations-tail")
                ) == "Load 20 older conversations"
                assert len(list(screen.query(".personas-conversation-row"))) == 20

                next_duplicate_page = duplicate_page(duplicate_reads + 1)
                next_boundary = generated_boundaries[-1]
                progress_row = _conversation_record(21)
                progress_row["last_modified"] = (
                    datetime.fromisoformat(next_boundary[0])
                    - timedelta(seconds=1)
                ).isoformat()
                stub_conversations.replace_pages(
                    next_duplicate_page, [progress_row]
                )
                calls_before_new_attempt = len(stub_conversations.calls)

                conversation_list.index = len(conversation_list.children) - 1
                await pilot.press("enter")
                await pilot.app.workers.wait_for_complete()
                await pilot.pause()

                assert len(stub_conversations.calls) == calls_before_new_attempt + 2
                assert stub_conversations.calls[-2][3] == {
                    "before_last_modified": latest_safe_boundary[0],
                    "before_id": latest_safe_boundary[1],
                }
                assert stub_conversations.calls[-1][3] == {
                    "before_last_modified": next_boundary[0],
                    "before_id": next_boundary[1],
                }
                assert list(screen.query("#personas-conversation-row-conv-21"))
                assert _row_text(
                    screen.query_one(".personas-conversations-tail")
                ) == "All conversations shown."
        finally:
            abort_unbounded_read.set()

    async def test_mixed_page_commits_raw_boundary_after_last_accepted_row(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        first_page = [_conversation_record(index) for index in range(1, 22)]
        row_21 = _conversation_record(21)
        row_21_time = datetime.fromisoformat(row_21["last_modified"])
        trailing_duplicates = []
        for index in range(1, 20):
            duplicate = _conversation_record(index)
            duplicate["last_modified"] = (
                row_21_time - timedelta(seconds=index)
            ).isoformat()
            trailing_duplicates.append(duplicate)
        mixed_page = [row_21, *trailing_duplicates, _conversation_record(22)]
        stub_conversations.replace_pages(
            first_page, mixed_page, [_conversation_record(22)]
        )
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            conversation_list = screen.query_one(
                "#personas-conversations-list", ListView
            )
            conversation_list.focus()
            conversation_list.index = len(conversation_list.children) - 1
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            raw_boundary = trailing_duplicates[-1]
            assert screen.conversations._next_conversation_cursor == (
                raw_boundary["last_modified"],
                raw_boundary["id"],
            )
            assert list(screen.query("#personas-conversation-row-conv-21"))

            conversation_list.index = len(conversation_list.children) - 1
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert stub_conversations.calls[-1] == (
                1,
                21,
                0,
                {
                    "before_last_modified": raw_boundary["last_modified"],
                    "before_id": raw_boundary["id"],
                },
            )
            assert list(screen.query("#personas-conversation-row-conv-22"))

    async def test_nonadvancing_duplicate_page_terminates_without_looping(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        first_page = [_conversation_record(index) for index in range(1, 22)]
        duplicate_page = [_conversation_record(index) for index in range(1, 22)]
        stub_conversations.replace_pages(first_page, duplicate_page)
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            conversation_list = screen.query_one(
                "#personas-conversations-list", ListView
            )
            conversation_list.focus()
            conversation_list.index = len(conversation_list.children) - 1
            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            rows = list(screen.query(".personas-conversation-row"))
            assert len(rows) == 20
            assert len({row.id for row in rows}) == 20
            assert _row_text(
                screen.query_one(".personas-conversations-tail")
            ) == "All conversations shown."
            await pilot.press("enter")
            await pilot.pause()
            assert len(stub_conversations.calls) == 2

    @pytest.mark.parametrize(
        "stale_action",
        ("other-character", "mode-switch", "same-character-reset"),
    )
    async def test_gated_stale_append_is_ignored_after_context_changes(
        self,
        mock_app_instance,
        stub_characters,
        stub_conversations,
        stub_scope_service,
        stale_action,
    ):
        started = asyncio.Event()
        release = threading.Event()
        app = PersonasTestApp(mock_app_instance)

        def gated_page(character_id, limit=50, offset=0, **cursor):
            app.call_from_thread(started.set)
            release.wait(timeout=5)
            return [_conversation_record(21, title="Stale older row")]

        newer_page = [_conversation_record(90, title="Current context row")]
        stub_conversations.replace_pages(
            [_conversation_record(index) for index in range(1, 22)],
            gated_page,
            newer_page,
        )
        try:
            async with app.run_test(size=(160, 50)) as pilot:
                screen = await _mounted(pilot)
                await pilot.app.workers.wait_for_complete()
                conversation_list = screen.query_one(
                    "#personas-conversations-list", ListView
                )
                conversation_list.focus()
                conversation_list.index = len(conversation_list.children) - 1
                await pilot.press("enter")
                await wait_for_signal(started, what="the stale older-page read")

                if stale_action == "other-character":
                    await pilot.click("#personas-library-row-character-2")
                elif stale_action == "mode-switch":
                    await pilot.click("#personas-mode-personas")
                else:
                    await pilot.click("#personas-library-row-character-1")
                release.set()
                await pilot.app.workers.wait_for_complete()
                await pilot.pause()

                visible = [
                    _row_text(row)
                    for row in screen.query(".personas-conversation-row")
                ]
                assert "Stale older row" not in visible
                if stale_action == "mode-switch":
                    assert screen.state.active_mode == "personas"
                    assert visible == []
                else:
                    assert visible == ["Current context row"]
        finally:
            release.set()

    async def test_mode_switch_wins_after_stale_initial_rows_enter_dom_mount(
        self,
        mock_app_instance,
        stub_characters,
        stub_conversations,
        stub_scope_service,
        monkeypatch,
    ):
        """A reset during the inspector's mount await owns the final list."""
        stale_page = [_conversation_record(90, title="Stale DOM row")]
        initial_mount_started = asyncio.Event()
        render_invalidated = asyncio.Event()
        release_initial_mount = asyncio.Event()
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            stub_conversations.replace_pages(stale_page)
            inspector = screen.query_one(PersonasInspectorPane)
            conversation_list = screen.query_one(
                "#personas-conversations-list", ListView
            )
            real_extend = conversation_list.extend

            async def gated_extend(items):
                items = tuple(items)
                if any(
                    item.id == "personas-conversation-row-conv-90" for item in items
                ):
                    initial_mount_started.set()
                    await release_initial_mount.wait()
                return await real_extend(items)

            monkeypatch.setattr(conversation_list, "extend", gated_extend)
            real_invalidate = inspector.invalidate_conversation_render

            def observed_invalidate(*args, **kwargs):
                real_invalidate(*args, **kwargs)
                render_invalidated.set()
                release_initial_mount.set()

            monkeypatch.setattr(
                inspector, "invalidate_conversation_render", observed_invalidate
            )

            await pilot.click("#personas-library-row-character-2")
            await wait_for_signal(
                initial_mount_started, what="the stale initial-row DOM mount"
            )

            mode_switch = asyncio.create_task(screen._apply_mode("personas"))
            try:
                await wait_for_signal(
                    render_invalidated, what="the synchronous list-render invalidation"
                )
                await mode_switch
                await pilot.app.workers.wait_for_complete()
                await pilot.pause()

                assert screen.state.active_mode == "personas"
                assert list(screen.query(".personas-conversation-row")) == []
                assert not any(
                    "Retry" in _row_text(tail)
                    for tail in screen.query(".personas-conversations-tail")
                )
            finally:
                release_initial_mount.set()
                if not mode_switch.done():
                    await mode_switch

    async def test_append_completion_preserves_other_focus_and_highlight(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        started = asyncio.Event()
        release = threading.Event()
        app = PersonasTestApp(mock_app_instance)

        def gated_page(character_id, limit=50, offset=0, **cursor):
            app.call_from_thread(started.set)
            release.wait(timeout=5)
            return [_conversation_record(21)]

        stub_conversations.replace_pages(
            [_conversation_record(index) for index in range(1, 22)], gated_page
        )
        try:
            async with app.run_test(size=(160, 50)) as pilot:
                screen = await _mounted(pilot)
                await pilot.app.workers.wait_for_complete()
                conversation_list = screen.query_one(
                    "#personas-conversations-list", ListView
                )
                conversation_list.focus()
                conversation_list.index = len(conversation_list.children) - 1
                await pilot.press("enter")
                await wait_for_signal(started, what="the focus-preservation read")

                conversation_list.index = 4
                search = screen.query_one("#personas-library-search", Input)
                search.focus()
                await pilot.pause()
                release.set()
                await pilot.app.workers.wait_for_complete()
                await pilot.pause()

                assert pilot.app.focused is search
                assert conversation_list.index == 4
                assert _row_text(conversation_list.highlighted_child) == "Case 5"
        finally:
            release.set()

    async def test_conversations_panel_shows_loading_then_rows(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        """While the listing worker runs the panel says it is loading."""
        started = asyncio.Event()
        release = threading.Event()
        app = PersonasTestApp(mock_app_instance)

        def gated_listing(character_id, limit=50, offset=0, **cursor):
            app.call_from_thread(started.set)
            release.wait(timeout=5)
            return [_conversation_record(1, title="First case")]

        stub_conversations.replace_pages(gated_listing)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await wait_for_signal(started, what="the gated conversation listing")
            # F-031: first-paint auto-select already started the (gated)
            # listing during mount - the loading placeholder is up while the
            # worker thread waits on the gate.
            panel = screen.query_one("#personas-conversations-list")
            texts: list[str] = []
            for _ in range(200):
                await pilot.pause(0.05)
                texts = [str(s.renderable) for s in panel.query(Static)]
                if any("Loading conversations..." in text for text in texts):
                    break
            assert any("Loading conversations..." in text for text in texts)
            release.set()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            rows = screen.query(".personas-conversation-row")
            assert [_row_text(r) for r in rows] == ["First case"]
            texts = [str(s.renderable) for s in panel.query(Static)]
            assert not any("Loading" in text for text in texts)

    async def test_conversations_panel_empty_shows_copy(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        stub_conversations.replace_pages([])
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            panel = screen.query_one("#personas-conversations-list")
            texts = [str(s.renderable) for s in panel.query(Static)]
            assert any("No saved conversations." in text for text in texts)
            assert list(screen.query(".personas-conversation-row")) == []

    async def test_open_conversation_shows_loading_placeholder(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        """Clicking a conversation gives instant feedback: the transcript view
        opens immediately with a loading placeholder."""
        import threading

        release = threading.Event()

        def gated_messages(db, conversation_id, character_name, user_name, **kwargs):
            release.wait(timeout=5)
            return [("Hello there", "Greetings, detective.")]

        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            gated_messages,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            await pilot.click("#personas-conversation-row-conv-1")
            await pilot.pause()
            view = screen.query_one("#personas-conversation-transcript-view")
            assert view.display is True
            texts = [str(s.renderable) for s in view.query(Static)]
            assert any("Loading transcript..." in text for text in texts)
            note = screen.query_one("#personas-transcript-preview-note", Static)
            assert str(note.renderable) == (
                "Preview shows up to 200 messages. Resume opens the saved chat "
                "in Console."
            )
            assert note.parent is view
            resume = screen.query_one("#personas-conversation-resume", Button)
            assert resume.disabled is False
            assert str(resume.label) == "Resume chat"
            release.set()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            texts = [str(s.renderable) for s in view.query(Static)]
            assert not any("Loading transcript..." in text for text in texts)
            assert any("Greetings, detective." in text for text in texts)

    async def test_back_during_preview_success_load_stays_on_card(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        started = threading.Event()
        release = threading.Event()

        def gated_messages(db, conversation_id, character_name, user_name, **kwargs):
            started.set()
            release.wait(timeout=5)
            return [("Delayed question", "Delayed answer")]

        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            gated_messages,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            completion_seen = asyncio.Event()
            original_show = screen.conversations.show_conversation_view

            async def observed_show(*args, **kwargs):
                await original_show(*args, **kwargs)
                completion_seen.set()

            monkeypatch.setattr(
                screen.conversations, "show_conversation_view", observed_show
            )
            try:
                await pilot.click("#personas-conversation-row-conv-1")
                assert await asyncio.to_thread(started.wait, 2)
                await pilot.click("#personas-conversation-back")
                await pilot.pause()
                release.set()
                await asyncio.wait_for(completion_seen.wait(), 2)
                await pilot.pause()

                assert screen.query_one("#ccp-character-card-view").display is True
                assert (
                    screen.query_one(
                        "#personas-conversation-transcript-view"
                    ).display
                    is False
                )
                assert (
                    screen.query_one("#personas-conversation-actions").display
                    is False
                )
                assert screen.query_one("#personas-inspector-actions").display is True
                assert pilot.app.focused.id == "personas-conversations-list"
            finally:
                release.set()

    async def test_back_during_preview_error_load_stays_on_card(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        started = threading.Event()
        release = threading.Event()

        def gated_failure(db, conversation_id, character_name, user_name, **kwargs):
            started.set()
            release.wait(timeout=5)
            raise RuntimeError("delayed preview failure")

        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            gated_failure,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            completion_seen = asyncio.Event()
            original_show_error = screen.conversations.show_conversation_error

            async def observed_show_error(*args, **kwargs):
                await original_show_error(*args, **kwargs)
                completion_seen.set()

            monkeypatch.setattr(
                screen.conversations,
                "show_conversation_error",
                observed_show_error,
            )
            try:
                await pilot.click("#personas-conversation-row-conv-1")
                assert await asyncio.to_thread(started.wait, 2)
                await pilot.click("#personas-conversation-back")
                await pilot.pause()
                release.set()
                await asyncio.wait_for(completion_seen.wait(), 2)
                await pilot.pause()

                assert screen.query_one("#ccp-character-card-view").display is True
                assert (
                    screen.query_one(
                        "#personas-conversation-transcript-view"
                    ).display
                    is False
                )
                assert (
                    screen.query_one("#personas-conversation-actions").display
                    is False
                )
                assert screen.query_one("#personas-inspector-actions").display is True
                assert pilot.app.focused.id == "personas-conversations-list"
            finally:
                release.set()

    async def test_stale_same_row_completion_cannot_replace_newer_preview(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        first_started = threading.Event()
        first_release = threading.Event()
        calls = 0

        def gated_first_load(
            db, conversation_id, character_name, user_name, **kwargs
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                first_release.wait(timeout=5)
                return [("Old question", "Stale completion")]
            return [("New question", "Current preview")]

        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            gated_first_load,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            stale_completion_seen = asyncio.Event()
            original_show = screen.conversations.show_conversation_view

            async def observed_show(*args, **kwargs):
                await original_show(*args, **kwargs)
                if "Stale completion" in args[2]:
                    stale_completion_seen.set()

            monkeypatch.setattr(
                screen.conversations, "show_conversation_view", observed_show
            )
            try:
                await pilot.click("#personas-conversation-row-conv-1")
                assert await asyncio.to_thread(first_started.wait, 2)

                await pilot.click("#personas-conversation-row-conv-1")
                for _ in range(200):
                    await pilot.pause(0.01)
                    text = "\n".join(
                        str(line.renderable)
                        for line in screen.query(".personas-transcript-line")
                    )
                    if "Current preview" in text:
                        break
                assert "Current preview" in text

                first_release.set()
                await asyncio.wait_for(stale_completion_seen.wait(), 2)
                await pilot.pause()
                text = "\n".join(
                    str(line.renderable)
                    for line in screen.query(".personas-transcript-line")
                )
                assert "Current preview" in text
                assert "Stale completion" not in text
                assert (
                    screen.query_one(
                        "#personas-conversation-transcript-view"
                    ).display
                    is True
                )
                assert screen.query_one("#ccp-character-card-view").display is False
                assert pilot.app.focused.id == "personas-conversations-list"
            finally:
                first_release.set()

    async def test_character_switch_invalidates_preview_before_detail_load_finishes(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            prior_attempt = screen.conversations._preview_attempt
            detail_started = asyncio.Event()
            release_detail = asyncio.Event()
            original_load_character = screen.character_handler.load_character

            async def gated_load_character(character_id):
                if str(character_id) == "2":
                    detail_started.set()
                    await release_detail.wait()
                await original_load_character(character_id)

            monkeypatch.setattr(
                screen.character_handler, "load_character", gated_load_character
            )
            selection = asyncio.create_task(
                screen._select_character("2", "Lab Assistant")
            )
            try:
                await asyncio.wait_for(detail_started.wait(), 2)
                assert screen.state.selected_entity_id == "2"
                assert screen.conversations._preview_attempt is None

                await screen.conversations.show_conversation_view(
                    "conv-1",
                    [{"role": "assistant", "content": "Stale character preview"}],
                    "Stale character preview",
                    False,
                    prior_attempt,
                )
                rendered = "\n".join(
                    str(line.renderable)
                    for line in screen.query(".personas-transcript-line")
                )
                assert "Stale character preview" not in rendered
            finally:
                release_detail.set()
                await selection

    async def test_stale_transcript_mount_cannot_replace_newer_loading_state(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            controller = screen.conversations
            prior_attempt = controller._preview_attempt
            controller._conversation_rows["conv-2"] = "Second case"
            monkeypatch.setattr(
                controller, "load_conversation_messages", lambda *args, **kwargs: None
            )

            view = screen.query_one("#personas-conversation-transcript-view")
            scroll = screen.query_one("#personas-transcript-scroll")
            stale_mount_started = asyncio.Event()
            release_stale_mount = asyncio.Event()
            loading_complete = asyncio.Event()
            original_mount_all = scroll.mount_all
            original_show_loading = view.show_loading
            mount_calls = 0

            async def gated_mount_all(widgets, *args, **kwargs):
                nonlocal mount_calls
                mount_calls += 1
                if mount_calls == 1:
                    stale_mount_started.set()
                    await release_stale_mount.wait()
                return await original_mount_all(widgets, *args, **kwargs)

            async def observed_show_loading(*args, **kwargs):
                result = await original_show_loading(*args, **kwargs)
                loading_complete.set()
                return result

            monkeypatch.setattr(scroll, "mount_all", gated_mount_all)
            monkeypatch.setattr(view, "show_loading", observed_show_loading)

            stale_render = asyncio.create_task(
                controller.show_conversation_view(
                    "conv-1",
                    [{"role": "assistant", "content": "Stale completion"}],
                    "Stale completion",
                    False,
                    prior_attempt,
                )
            )
            await asyncio.wait_for(stale_mount_started.wait(), 2)
            current_open = asyncio.create_task(controller.open_conversation("conv-2"))
            try:
                for _ in range(100):
                    if controller._preview_attempt is not prior_attempt:
                        break
                    await pilot.pause(0.01)
                assert controller._preview_attempt is not prior_attempt
                try:
                    await asyncio.wait_for(loading_complete.wait(), 0.2)
                except TimeoutError:
                    pass
                release_stale_mount.set()
                await asyncio.gather(stale_render, current_open)
                await pilot.pause()

                assert screen.query_one("#personas-transcript-loading", Static)
                rendered = "\n".join(
                    str(line.renderable)
                    for line in screen.query(".personas-transcript-line")
                )
                assert "Stale completion" not in rendered
            finally:
                release_stale_mount.set()
                await asyncio.gather(stale_render, current_open, return_exceptions=True)

    async def test_empty_conversation_is_distinct_and_can_resume(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            lambda *args, **kwargs: [],
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            assert str(
                screen.query_one("#personas-transcript-empty", Static).renderable
            ) == "No messages to display."
            assert screen.query_one(
                "#personas-conversation-resume", Button
            ).disabled is False

    async def test_conversation_load_failure_is_distinct_and_does_not_stage(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        def fail(*args, **kwargs):
            raise RuntimeError("preview unavailable")

        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            fail,
        )
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            error = screen.query_one("#personas-transcript-error", Static)
            assert str(error.renderable) == (
                "Couldn't load this preview. You can still resume the saved chat."
            )
            assert screen.conversations._loaded_conversation_id is None
            assert screen.conversations._failed_conversation_id == "conv-1"
            assert screen.query_one(
                "#personas-conversation-resume", Button
            ).disabled is False
            await pilot.click("#personas-conversation-continue-console")
            await pilot.pause()
        app.open_chat_with_handoff.assert_not_called()

    async def test_transcript_lines_use_speaker_names(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The read-only transcript uses You/<character name>, not user/assistant."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            view = screen.query_one("#personas-conversation-transcript-view")
            texts = [
                str(line.renderable) for line in view.query(".personas-transcript-line")
            ]
            assert texts == [
                "You: Hello there",
                "Detective Sam: Greetings, detective.",
            ]

    async def test_conversation_listing_failure_is_tolerant(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        stub_conversations.replace_pages(RuntimeError("listing failed"))
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            assert list(screen.query(".personas-conversation-row")) == []
            # Selection itself still succeeded.
            assert screen.state.selected_entity_id == "1"
            assert screen.query_one("#ccp-character-card-view").display is True

    async def test_conversation_row_opens_readonly_view(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            assert (
                screen.query_one("#personas-conversation-transcript-view").display
                is True
            )
            assert screen.query_one("#ccp-character-card-view").display is False

    async def test_back_returns_to_card(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            await pilot.click("#personas-conversation-back")
            await pilot.pause()
            assert screen.query_one("#ccp-character-card-view").display is True
            assert (
                screen.query_one("#personas-conversation-transcript-view").display
                is False
            )

    async def test_conversation_preview_hides_card_actions_shortcut_and_footer_then_back_restores(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            inspector_actions = screen.query_one("#personas-inspector-actions")
            assert inspector_actions.display is False
            assert "ctrl+enter" not in screen._shortcut_context().render().lower()
            await pilot.press("ctrl+enter")
            await pilot.pause()
            app.open_chat_with_handoff.assert_not_called()

            await pilot.click("#personas-conversation-back")
            await pilot.pause()
            assert inspector_actions.display is True
            assert "ctrl+enter send to console draft" in (
                screen._shortcut_context().render().lower()
            )

    async def test_conversation_actions_have_three_row_hierarchy(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            actions = screen.query_one("#personas-conversation-actions")
            assert [child.id for child in actions.children] == [
                "personas-conversation-resume",
                "personas-conversation-continue-console",
                "personas-conversation-navigation-actions",
            ]
            navigation = screen.query_one("#personas-conversation-navigation-actions")
            assert [child.id for child in navigation.children] == [
                "personas-conversation-back",
                "personas-conversation-open-library",
            ]
            resume = screen.query_one("#personas-conversation-resume", Button)
            send = screen.query_one(
                "#personas-conversation-continue-console", Button
            )
            back = screen.query_one("#personas-conversation-back", Button)
            library = screen.query_one(
                "#personas-conversation-open-library", Button
            )
            assert resume.has_class("console-action-primary")
            assert send.has_class("console-action-secondary")
            assert back.has_class("console-action-subdued")
            assert library.has_class("console-action-subdued")
            assert str(resume.label) == "Resume chat"
            assert str(send.label) == "Send transcript to Console draft"
            assert str(back.label) == "Back to card"
            assert str(library.label) == "Open in Library"
            assert actions.region.height == 9
            assert resume.region.width == actions.content_region.width
            assert send.region.width == actions.content_region.width
            assert back.region.width == library.region.width
            assert resume.region.y < send.region.y < back.region.y

    @pytest.mark.parametrize("size", ((80, 24), (160, 50)), ids=("compact", "standard"))
    async def test_conversation_actions_fit_production_css(
        self, mock_app_instance, stub_characters, stub_conversations, size
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=size) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await screen.conversations.open_conversation("conv-1")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            actions = screen.query_one("#personas-conversation-actions")
            for button in actions.query(Button):
                assert button.region.x >= actions.content_region.x
                assert button.region.right <= actions.content_region.right
                assert button.region.y >= actions.content_region.y
                assert button.region.bottom <= actions.content_region.bottom
            assert screen.query_one("#personas-transcript-scroll").region.height > 0

            resume = screen.query_one("#personas-conversation-resume", Button)
            send = screen.query_one(
                "#personas-conversation-continue-console", Button
            )
            navigation = screen.query_one(
                "#personas-conversation-navigation-actions"
            )
            back = screen.query_one("#personas-conversation-back", Button)
            library = screen.query_one(
                "#personas-conversation-open-library", Button
            )
            assert [
                resume.region.height,
                send.region.height,
                navigation.region.height,
                back.region.height,
                library.region.height,
            ] == [3, 3, 3, 3, 3]
            assert send.region.y - resume.region.y == 3
            assert navigation.region.y - send.region.y == 3
            assert back.region.y == navigation.region.y == library.region.y

    async def test_conversation_preview_f6_and_tab_order_start_at_resume(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            screen.query_one("#personas-library-rows").focus()
            await pilot.press("f6")
            await pilot.pause()
            assert pilot.app.focused.id == "personas-conversation-resume"
            expected = (
                "personas-conversation-continue-console",
                "personas-conversation-back",
                "personas-conversation-open-library",
                "personas-transcript-scroll",
            )
            for focus_id in expected:
                await pilot.press("tab")
                await pilot.pause()
                assert pilot.app.focused.id == focus_id

    async def test_conversation_resume_posts_normalized_id_only(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        callbacks = []
        app = _NavCaptureApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            screen.conversations._open_conversation_id = "  conv-1  "
            monkeypatch.setattr(
                screen,
                "set_timer",
                lambda _delay, callback, **_kwargs: callbacks.append(callback),
            )
            screen.conversations.resume_in_console()
            await pilot.pause()
            resume = screen.query_one("#personas-conversation-resume", Button)
            assert resume.disabled is True
            assert str(resume.label) == "Opening Console…"
            assert app.nav_routes == [TAB_CHAT]
            assert app.nav_contexts == [
                {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "conv-1"}
            ]
            app.open_chat_with_handoff.assert_not_called()

            assert len(callbacks) == 1
            callbacks[0]()
            assert resume.disabled is False
            assert str(resume.label) == "Resume chat"

    async def test_conversation_resume_reselection_keeps_same_target_single_flight(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        stub_conversations.replace_pages(
            [
                _conversation_record(1, title="First case"),
                _conversation_record(2, title="Second case"),
            ]
        )
        callbacks = []
        app = _StyledNavCaptureApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            monkeypatch.setattr(
                screen,
                "set_timer",
                lambda _delay, callback, **_kwargs: callbacks.append(callback),
            )

            await pilot.click("#personas-conversation-row-conv-1")
            await pilot.app.workers.wait_for_complete()
            await pilot.click("#personas-conversation-resume")
            await pilot.pause()
            assert app.nav_contexts == [
                {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "conv-1"}
            ]

            await pilot.click("#personas-conversation-row-conv-1")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            resume = screen.query_one("#personas-conversation-resume", Button)
            assert resume.disabled is True
            assert str(resume.label) == "Opening Console…"
            await pilot.click("#personas-conversation-resume")
            await pilot.pause()
            assert app.nav_contexts == [
                {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "conv-1"}
            ]

            await pilot.click("#personas-conversation-row-conv-2")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert resume.disabled is False
            await pilot.click("#personas-conversation-resume")
            await pilot.pause()
            assert app.nav_contexts == [
                {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "conv-1"},
                {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "conv-2"},
            ]

            assert len(callbacks) == 2
            callbacks[0]()
            assert resume.disabled is True
            assert str(resume.label) == "Opening Console…"
            callbacks[1]()
            assert resume.disabled is False
            assert str(resume.label) == "Resume chat"

    async def test_conversation_resume_single_flight_survives_browsing_away_and_back(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        stub_conversations.replace_pages(
            [
                _conversation_record(1, title="First case"),
                _conversation_record(2, title="Second case"),
            ]
        )
        callbacks = []
        app = _StyledNavCaptureApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            monkeypatch.setattr(
                screen,
                "set_timer",
                lambda _delay, callback, **_kwargs: callbacks.append(callback),
            )

            await pilot.click("#personas-conversation-row-conv-1")
            await pilot.app.workers.wait_for_complete()
            await pilot.click("#personas-conversation-resume")
            await pilot.pause()

            await pilot.click("#personas-conversation-row-conv-2")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            resume = screen.query_one("#personas-conversation-resume", Button)
            assert resume.disabled is False

            await pilot.click("#personas-conversation-row-conv-1")
            await pilot.app.workers.wait_for_complete()
            await pilot.click("#personas-conversation-resume")
            await pilot.pause()
            assert app.nav_contexts == [
                {CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID: "conv-1"}
            ]
            assert len(callbacks) == 1
            assert resume.disabled is True
            assert str(resume.label) == "Opening Console…"

    async def test_conversation_stale_same_target_fallback_keeps_new_attempt_busy(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        callbacks = []
        app = _StyledNavCaptureApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            monkeypatch.setattr(
                screen,
                "set_timer",
                lambda _delay, callback, **_kwargs: callbacks.append(callback),
            )
            resume = screen.query_one("#personas-conversation-resume", Button)

            await pilot.click("#personas-conversation-resume")
            await pilot.pause()
            assert len(callbacks) == 1
            callbacks[0]()
            assert resume.disabled is False
            await pilot.pause()

            await pilot.click("#personas-conversation-resume")
            await pilot.pause()
            assert len(callbacks) == 2
            callbacks[0]()
            assert resume.disabled is True
            assert str(resume.label) == "Opening Console…"
            callbacks[1]()
            assert resume.disabled is False
            assert str(resume.label) == "Resume chat"

    async def test_conversation_resume_stale_row_stays_in_roleplay_with_exact_copy(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        notifications: list[str] = []
        app = _NavCaptureApp(mock_app_instance)
        app.notify = lambda message, **kwargs: notifications.append(str(message))
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            screen.conversations._conversation_rows = {}
            screen.conversations.resume_in_console()
            await pilot.pause()
        assert app.nav_routes == []
        assert notifications == [
            "This conversation is no longer available. Refresh conversations and "
            "try again."
        ]

    async def test_conversation_disabled_resume_busy_label_meets_contrast_floor(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(80, 24)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await screen.conversations.open_conversation("conv-1")
            await pilot.app.workers.wait_for_complete()
            resume = screen.query_one("#personas-conversation-resume", Button)
            resume.label = "Opening Console…"
            resume.disabled = True
            await pilot.pause()
            assert resume.styles.opacity == 1.0
            style = _painted_style_of_text(app, resume.region, "Opening Console…")
            assert style is not None
            assert style.color is not None and style.bgcolor is not None
            assert _contrast_ratio(style.color, style.bgcolor) >= 3.0

    async def test_continue_in_console_stages_payload(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            await self._open_conversation(pilot)
            await pilot.click("#personas-conversation-continue-console")
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.item_type == "character-conversation"
        assert payload.metadata["conversation_id"] == "conv-1"
        assert payload.metadata["selected_kind"] == "character"
        assert payload.metadata["selected_record_id"] == "1"
        assert "Detective Sam" in payload.title
        assert "First case" in payload.title
        assert "Greetings, detective." in payload.body

    async def test_open_in_library_navigates(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = _NavCaptureApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            await self._open_conversation(pilot)
            await pilot.click("#personas-conversation-open-library")
            await pilot.pause()
            assert app.nav_routes == [TAB_LIBRARY]
            assert app.nav_contexts == [
                {
                    LIBRARY_NAV_CONTEXT_MODE: LIBRARY_MODE_CONVERSATIONS,
                    LIBRARY_NAV_CONTEXT_CONVERSATION_ID: "conv-1",
                }
            ]

    async def test_open_in_library_requires_open_conversation(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        notifications: list[tuple[str, str]] = []
        app = _NavCaptureApp(mock_app_instance)
        app.notify = lambda message, severity="information", **kwargs: (
            notifications.append((str(message), severity))
        )
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            screen.conversations.open_in_library()
            await pilot.pause()

        assert app.nav_routes == []
        assert any(
            "Open a conversation" in message and severity == "warning"
            for message, severity in notifications
        )

    async def test_stale_conversation_view_is_skipped(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A view continuation for a superseded conversation id is dropped."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            assert screen.conversations._open_conversation_id == "conv-1"
            await screen.conversations.show_conversation_view(
                "conv-stale",
                [{"role": "user", "content": "stale"}],
                "stale",
                False,
                object(),
            )
            await pilot.pause()
            assert screen.conversations._loaded_conversation_id == "conv-1"
            assert screen.conversations._open_conversation_transcript != "stale"

    async def test_long_transcript_sets_body_truncated(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            lambda db, conversation_id, character_name, user_name, **kwargs: [
                ("u" * 500, "b" * 500) for _ in range(20)
            ],
        )
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            await self._open_conversation(pilot)
            await pilot.click("#personas-conversation-continue-console")
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.body_truncated is True
        assert (
            len(payload.body)
            <= conversations_controller_module._HANDOFF_TRANSCRIPT_CHAR_LIMIT
        )
        assert payload.source_id == "conv-1"

    async def test_continue_blocked_while_loading(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Send to Console draft refuses to stage a transcript still in flight."""
        notifications: list[tuple[str, str]] = []
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        app.notify = lambda message, severity="information", **kwargs: (
            notifications.append((str(message), severity))
        )
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            screen.conversations._open_conversation_id = "conv-2"
            screen.conversations._loaded_conversation_id = "conv-1"
            await pilot.click("#personas-conversation-continue-console")
            await pilot.pause()
        app.open_chat_with_handoff.assert_not_called()
        assert any(
            "still loading" in message and severity == "warning"
            for message, severity in notifications
        )

    async def test_continue_blocked_during_same_conversation_reload(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        """Re-selecting the same conversation clears loaded state so Continue is blocked
        until the reload worker delivers its results.

        Regression: open_conversation() previously only reset _open_conversation_transcript
        but left _loaded_conversation_id and _open_conversation_truncated intact.  That meant
        the guard in continue_in_console() saw _loaded_conversation_id ==
        _open_conversation_id immediately after the reload started and would stage an empty
        body with a stale truncation flag.
        """
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            ConversationRowSelected as _CRS,
        )

        notifications: list[tuple[str, str]] = []
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        app.notify = lambda message, severity="information", **kwargs: (
            notifications.append((str(message), severity))
        )

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._open_conversation(pilot)
            # conv-1 is fully loaded at this point.
            assert screen.conversations._loaded_conversation_id == "conv-1"
            assert screen.conversations._open_conversation_transcript != ""

            # Stub the worker entry so it never calls show_conversation_view,
            # thus simulating an in-flight reload whose result hasn't arrived
            # yet. We patch on the controller instance so subsequent calls
            # within this test are bypassed.
            screen.conversations.load_conversation_messages = lambda *args, **kwargs: (
                None
            )

            # Re-select the same conversation by posting the message directly to the
            # screen — this exercises _handle_conversation_row_selected →
            # open_conversation() without relying on the inspector-pane button being
            # click-reachable while the conversation view is displayed on top.
            screen.post_message(_CRS("conv-1"))
            await pilot.pause()

            # open_conversation() should have cleared _loaded_conversation_id.
            assert screen.conversations._loaded_conversation_id is None, (
                "open_conversation() must reset _loaded_conversation_id so that "
                "re-selecting the same conversation doesn't bypass the "
                "still-loading guard"
            )

            # Try to continue — the reload is in flight so it must be blocked.
            await pilot.click("#personas-conversation-continue-console")
            await pilot.pause()

        app.open_chat_with_handoff.assert_not_called()
        assert any(
            "still loading" in message and severity == "warning"
            for message, severity in notifications
        ), f"Expected 'still loading' warning; got: {notifications}"

    async def test_profile_selection_shows_no_conversations(
        self, mock_app_instance, stub_characters, stub_conversations, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            assert len(screen.query(".personas-conversation-row")) == 1
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-library-row-persona-p-1")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_kind == "persona"
            assert list(screen.query(".personas-conversation-row")) == []


class TestConsoleActions:
    """Send to Console draft and Chat now from the inspector (Task 12, F-032)."""

    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        _install_conversation_db(
            monkeypatch, [_conversation_record(1, title="First case")]
        )
        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            lambda db, conversation_id, character_name, user_name, **kwargs: [
                ("Hello there", "Greetings, detective."),
            ],
        )

    async def _select_first_character(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def _select_profile(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-mode-personas")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#personas-library-row-persona-p-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def test_attach_stages_selected_character_payload(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.query_one("#personas-attach-to-console", Button).press()
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.runtime_backend == "local"
        assert payload.source_owner == "local"
        assert payload.source_selector_state == "local"
        assert payload.metadata["selected_kind"] == "character"
        assert payload.metadata["selected_record_id"] == "1"
        assert payload.metadata["selected_target_id"] == "local:character:1"
        assert payload.metadata["backend"] == "local"
        assert "Detective Sam" in payload.title
        assert "Noir detective" in payload.body
        assert "Detective Sam" in payload.suggested_prompt

    async def test_attach_blocked_without_selection(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        # No-selection requires an empty library now: F-031 auto-selects the
        # first row on a fresh mount when rows exist.
        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", lambda: []
        )
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("ctrl+enter")
            await pilot.pause()
        app.open_chat_with_handoff.assert_not_called()

    async def test_screen_gate_controls_visible_console_actions_after_selection(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            assert (
                screen.query_one("#personas-attach-to-console", Button).disabled
                is False
            )

            screen._console_action_allowed = lambda: False
            screen._console_action_block_reason = lambda: "prompts are not attachable"
            screen._sync_title_and_console_actions()
            await pilot.pause()

            assert (
                screen.query_one("#personas-attach-to-console", Button).disabled is True
            )
            assert screen.query_one("#personas-start-chat", Button).disabled is True
            assert (
                "Chat now and Send to Console draft blocked: prompts are not attachable"
                in str(
                    screen.query_one("#personas-readiness-console", Static).renderable
                )
            )

    async def test_readiness_surfaces_reflect_unready_character_provider(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Task-440: honest readiness copy when the handoff provider is unready.

        The configured chat_defaults provider (which a fresh Chat-now
        Console session resolves - the native Console never reads
        character_defaults) has no API key - the handoff send would fail, so
        neither readiness surface may claim things are ready. Per-intent gating
        (task-523): Chat now is DISABLED (it needs an immediate reply) while
        Send to Console draft stays enabled (it stages context; the reply is
        deferred).
        """
        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)

            readiness_text = str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert readiness_text != "Ready to chat in Console."
            assert readiness_text.startswith("Chat now blocked:")
            assert "anthropic" in readiness_text.lower()
            assert (
                "api key" in readiness_text.lower()
                or "api_settings" in readiness_text.lower()
            )
            # Per-intent gating (task-523): an unready handoff provider blocks
            # Chat now (it needs an immediate reply) but NOT Send to Console
            # draft (it only stages context; the reply is deferred). The user
            # can still stage the card and fix the provider before sending.
            assert screen.query_one("#personas-start-chat", Button).disabled is True
            assert (
                screen.query_one("#personas-attach-to-console", Button).disabled
                is False
            )

            header_status = str(
                screen.query_one(
                    "#personas-header #workbench-header-status", Static
                ).renderable
            )
            assert header_status != "Ready"

    async def test_readiness_surfaces_stay_ready_with_a_configured_provider(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Provider ready -> "Ready to chat in Console."/"Ready" copy shows."""
        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "api_settings": {"anthropic": {"api_key": "unit-test-placeholder-key"}},
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)

            assert "Ready to chat in Console." in str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert (
                str(
                    screen.query_one(
                        "#personas-header #workbench-header-status", Static
                    ).renderable
                )
                == "Ready"
            )
            assert (
                screen.query_one("#personas-attach-to-console", Button).disabled
                is False
            )
            assert screen.query_one("#personas-start-chat", Button).disabled is False

    async def test_readiness_blocked_when_handoff_provider_unready_despite_ready_character_provider(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Task-440 review: readiness mirrors the Chat-now HANDOFF resolution.

        Send to Console draft/Chat now create a fresh native-Console session
        resolved from
        chat_defaults (chat_screen._start_character_console_session ->
        _default_console_session_settings); the native Console never reads
        character_defaults. Shipped-defaults failure shape: only an Anthropic
        key configured (character_defaults=anthropic READY) while
        chat_defaults points at OpenAI (UNREADY) - the real handoff send
        would fail, so neither surface may claim ready. A
        character_defaults-first readiness short-circuits ready here, which
        is exactly the dishonesty under review.
        """
        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "openai", "model": "gpt-4o"},
            "api_settings": {"anthropic": {"api_key": "unit-test-placeholder-key"}},
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)

            readiness_text = str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert readiness_text != "Ready to chat in Console."
            assert readiness_text.startswith("Chat now blocked:")
            assert "openai" in readiness_text.lower()
            assert (
                str(
                    screen.query_one(
                        "#personas-header #workbench-header-status", Static
                    ).renderable
                )
                != "Ready"
            )
            # Per-intent gating (task-523): Chat now disabled, Send to Console
            # draft enabled.
            assert screen.query_one("#personas-start-chat", Button).disabled is True
            assert (
                screen.query_one("#personas-attach-to-console", Button).disabled
                is False
            )

    async def test_readiness_ready_when_handoff_provider_ready_despite_unready_character_provider(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Handoff provider ready => ready surfaces, whatever character_defaults says."""
        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "openai", "model": "gpt-4o"},
            "api_settings": {"openai": {"api_key": "unit-test-placeholder-key"}},
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)

            assert "Ready to chat in Console." in str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert (
                str(
                    screen.query_one(
                        "#personas-header #workbench-header-status", Static
                    ).renderable
                )
                == "Ready"
            )

    async def test_action_gate_precedes_provider_readiness_on_both_surfaces(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Qodo #824-2: with the action gate closed (unsaved edits), the
        provider axis is NOT operative -- the inspector shows the ACTION
        reason (never provider copy) and the header keeps its pre-task-440
        semantics rather than claiming a conflicting provider-"Blocked".
        One precedence rule on both surfaces: action gate first, provider
        readiness only once the gate opens."""
        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "openai", "model": "gpt-4o"},
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.state.has_unsaved_changes = True
            screen._sync_title_and_console_actions()
            await pilot.pause()

            readiness_text = str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert readiness_text == "Save or discard your edits to chat in Console."
            assert "openai" not in readiness_text.lower()  # no provider copy
            header_status = str(
                screen.query_one(
                    "#personas-header #workbench-header-status", Static
                ).renderable
            )
            assert header_status == "Ready"  # pre-task-440 header semantics

    async def test_selection_pushes_console_gate_before_async_followup(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        """Selection should not render as blocked before follow-up work completes."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            inspector = screen.query_one(PersonasInspectorPane)

            observed_enabled: list[bool] = []
            original_loading = inspector.show_conversations_loading

            async def assert_gate_synced_before_loading(render_attempt=None):
                observed_enabled.append(
                    not screen.query_one("#personas-attach-to-console", Button).disabled
                )
                return await original_loading(render_attempt)

            monkeypatch.setattr(
                inspector,
                "show_conversations_loading",
                assert_gate_synced_before_loading,
            )

            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

        assert observed_enabled == [True]

    async def test_character_save_updates_console_gate_without_detached_reload(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        """Save completion owns presentation without an unfenced reload worker."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()

            load_character = AsyncMock()
            monkeypatch.setattr(
                screen.character_handler, "load_character", load_character
            )

            await screen._after_character_save("1", "Detective Sam")
            await pilot.pause()

            assert not screen.query_one(
                "#personas-attach-to-console", Button
            ).disabled
            load_character.assert_not_awaited()

    async def test_profile_save_pushes_console_gate_before_row_render(
        self,
        mock_app_instance,
        stub_characters,
        stub_conversations,
        stub_scope_service,
        monkeypatch,
    ):
        """Profile save completion should not wait for row rendering to sync gates."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            observed_enabled: list[bool] = []

            async def observe_render_rows():
                observed_enabled.append(
                    not screen.query_one("#personas-attach-to-console", Button).disabled
                )

            monkeypatch.setattr(screen, "_render_profile_rows", observe_render_rows)

            await screen._after_profile_save({"id": "p-1", "name": "Archivist"})
            await pilot.pause()

        assert observed_enabled == [True]

    async def test_attach_blocked_with_unsaved_edits(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            EditCharacterRequested,
        )

        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.post_message(EditCharacterRequested("1"))
            await pilot.pause()
            assert screen._edit_mode == "edit"
            # Change-based dirty tracking: make a real edit first.
            from textual.widgets import TextArea

            screen.query_one(
                "#personas-char-editor-description", TextArea
            ).text = "edited"
            await pilot.pause()
            assert screen.state.has_unsaved_changes is True
            await pilot.press("ctrl+enter")
            await pilot.pause()
        app.open_chat_with_handoff.assert_not_called()

    async def test_start_chat_uses_real_mechanism(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Chat now stages a handoff with start_chat intent metadata.

        The legacy CCP route launched a blank tab directly via the main chat
        tab container (`#chat-window` lookup), which is not mounted while a
        destination screen is active; the workbench therefore uses the
        app-level ``open_chat_with_handoff`` API with an intent marker.
        """
        # Chat now needs a ready handoff provider (task-523 per-intent);
        # give a keyless local provider so the button is enabled and the guard
        # passes.
        mock_app_instance.app_config = {
            "chat_defaults": {"provider": "llama_cpp", "model": "local.gguf"},
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:8181"}},
        }
        mock_app_instance.runtime_backend = "local"
        mock_app_instance.active_server_id = "configured-but-not-selected"
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.query_one("#personas-start-chat", Button).press()
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.runtime_backend == "local"
        assert payload.source_owner == "local"
        assert payload.source_selector_state == "local"
        assert payload.active_server_profile_id is None
        assert payload.metadata["intent"] == "start_chat"
        assert payload.metadata["selected_target_id"] == "local:character:1"
        assert payload.metadata["backend"] == "local"
        assert payload.suggested_prompt == "Respond as Detective Sam."

    async def test_server_start_chat_carries_exact_source_and_active_target(
        self,
        mock_app_instance,
        stub_characters,
        stub_conversations,
        monkeypatch,
    ):
        """Server browsing, detail, preview, and handoff share one proven card."""
        local_card = {
            "id": 1,
            "name": "Local collision",
            "description": "Local-only description",
            "first_message": "Local-only greeting",
        }
        server_row = {
            "id": 1,
            "name": "Remote Elara",
            "description": "Server summary",
        }
        server_card = {
            "id": 1,
            "name": "Remote Elara",
            "description": "Remote authoritative description",
            "first_message": "Remote hello from {{char}}.",
            "system_prompt": "Use the server card.",
        }
        local_list = Mock(return_value=[dict(local_card)])
        local_detail = Mock(return_value=dict(local_card))
        local_page = Mock(return_value=[dict(local_card)])
        local_count = Mock(return_value=1)
        local_conversations = _install_conversation_db(monkeypatch, [])
        local_dictionaries = AsyncMock(return_value={"dictionaries": []})
        local_worldbooks = Mock(return_value=[])
        local_avatar = AsyncMock()
        monkeypatch.setattr(
            character_handler_module,
            "fetch_all_characters",
            local_list,
        )
        monkeypatch.setattr(
            character_handler_module,
            "fetch_character_by_id",
            local_detail,
        )
        monkeypatch.setattr(
            personas_screen_module,
            "get_character_page_for_ui",
            local_page,
        )
        monkeypatch.setattr(
            personas_screen_module,
            "count_character_page",
            local_count,
        )
        monkeypatch.setattr(
            PersonasScreen,
            "_render_inspector_avatar",
            local_avatar,
        )
        monkeypatch.setattr(
            PersonasScreen,
            "_lore_manager",
            lambda self: SimpleNamespace(
                get_world_books_for_character=local_worldbooks
            ),
        )

        row_dto = SimpleNamespace(model_dump=Mock(return_value=dict(server_row)))
        detail_dto = SimpleNamespace(model_dump=Mock(return_value=dict(server_card)))
        scope_service = SimpleNamespace(
            list_characters=AsyncMock(return_value=[row_dto]),
            search_characters=AsyncMock(return_value={"items": [row_dto], "total": 1}),
            get_character=AsyncMock(return_value=detail_dto),
        )
        mock_app_instance.app_config = {
            "chat_defaults": {"provider": "llama_cpp", "model": "local.gguf"},
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:8181"}},
        }
        mock_app_instance.runtime_backend = "server"
        mock_app_instance.active_server_id = "configured-target-7"
        mock_app_instance.character_persona_scope_service = scope_service
        mock_app_instance.chat_dictionary_scope_service = SimpleNamespace(
            list_character_dictionaries=local_dictionaries
        )
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # F-031: first-paint auto-select already fetched the mounted row
            # (server detail + card render) once; reset those provenance
            # mocks so the strict assertions below pin only the click-driven
            # selection path they were written for.
            scope_service.get_character.reset_mock()
            detail_dto.model_dump.reset_mock()
            row = screen.query_one("#personas-library-row-character-1")
            assert "Remote Elara" in _row_text(row)
            assert "Local collision" not in _row_text(row)

            screen.state.search_query = "remote"
            await screen._reload_character_page(reset_offset=True)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert "Remote Elara" in str(
                screen.query_one("#personas-character-card-name", Static).renderable
            )
            assert "Remote authoritative description" in str(
                screen.query_one(
                    "#personas-character-card-description", Static
                ).renderable
            )
            assert "Remote Elara: Remote hello from Remote Elara." in (
                screen.query_one(PersonasPreviewPane).transcript_text()
            )
            assert screen.query_one("#personas-character-attachments").display is False
            assert (
                screen.query_one("#personas-card-edit-character", Button).disabled
                is True
            )
            screen.query_one("#personas-start-chat", Button).press()
            await pilot.pause()

        page_size = personas_screen_module.PERSONAS_LIBRARY_PAGE_SIZE
        scope_service.list_characters.assert_awaited_once_with(
            mode="server",
            limit=page_size + 1,
            offset=0,
        )
        scope_service.search_characters.assert_awaited_once_with(
            "remote",
            mode="server",
            limit=page_size + 1,
        )
        scope_service.get_character.assert_awaited_once_with(1, mode="server")
        row_dto.model_dump.assert_called_with(mode="json")
        detail_dto.model_dump.assert_called_once_with(mode="json")
        local_list.assert_not_called()
        local_page.assert_not_called()
        local_count.assert_not_called()
        local_detail.assert_not_called()
        local_avatar.assert_not_awaited()
        assert local_conversations.calls == []
        local_dictionaries.assert_not_awaited()
        local_worldbooks.assert_not_called()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.runtime_backend == "server"
        assert payload.source_owner == "server"
        assert payload.source_selector_state == "server"
        assert payload.active_server_profile_id == "configured-target-7"
        assert payload.metadata["intent"] == "start_chat"
        assert payload.metadata["selected_target_id"] == "server:character:1"
        assert payload.metadata["backend"] == "server"
        assert "Remote authoritative description" in payload.body
        assert "Local-only description" not in payload.body

    async def test_start_chat_action_guard_blocks_unready_provider(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Task-523: the Start-Chat action path refuses to stage a handoff
        while the resolved provider is unready. The visible button is disabled,
        so this defends against a press racing a config change - invoked via the
        action path directly since ``.press()`` on a disabled button is a no-op.
        """
        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
        }
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        captured: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: captured.append(
            (str(message), severity)
        )
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            await screen._attach_selection_to_console(intent="start_chat")
            await pilot.pause()
        app.open_chat_with_handoff.assert_not_called()
        assert any(
            msg.startswith("Chat now blocked:") and severity == "warning"
            for msg, severity in captured
        )

    async def test_attach_action_not_gated_on_provider_readiness(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Task-523: the per-intent guard is Start-Chat-only. Attach still
        stages the card with an unready provider (its reply is deferred), so
        no ``intent=start_chat`` marker and a real handoff is created."""
        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
        }
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.query_one("#personas-attach-to-console", Button).press()
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.metadata.get("intent") != "start_chat"

    async def test_header_carries_blocked_class_when_provider_unready(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Task-523: the header carries the ``status-blocked`` class (the red
        cue's CSS hook) while the staged handoff provider is unready, and drops
        it once the provider becomes ready."""
        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            header = screen.query_one("#personas-header")
            assert header.has_class("status-blocked") is True

            mock_app_instance.app_config["api_settings"] = {
                "anthropic": {"api_key": "unit-test-placeholder-key"}
            }
            screen._sync_title_and_console_actions()
            await pilot.pause()
            assert header.has_class("status-blocked") is False

    async def test_blocked_header_badge_renders_red_under_real_bundle(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Task-523 regression guard for the red cue's CSS cascade.

        The colour rule MUST live in app-tier CSS: a widget ``DEFAULT_CSS``
        rule is outranked by the bundle's ``.ds-status-badge`` (color:
        $ds-text-primary) regardless of selector specificity, so the badge
        would stay primary and the cue would never render. Uses
        ``StyledPersonasTestApp`` (loads the real bundle) and asserts the
        blocked-state badge colour DIFFERS from the ready-state colour - if the
        rule were outranked, both states would render the identical primary
        colour and this fails.
        """
        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
        }
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            badge = screen.query_one(
                "#personas-header #workbench-header-status", Static
            )
            assert screen.query_one("#personas-header").has_class("status-blocked")
            blocked_color = badge.styles.color

            mock_app_instance.app_config["api_settings"] = {
                "anthropic": {"api_key": "unit-test-placeholder-key"}
            }
            screen._sync_title_and_console_actions()
            await pilot.pause()
            assert not screen.query_one("#personas-header").has_class("status-blocked")
            ready_color = badge.styles.color

        assert blocked_color != ready_color

    async def test_attach_stages_profile_payload(
        self, mock_app_instance, stub_characters, stub_conversations, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_profile(pilot)
            screen.query_one("#personas-attach-to-console", Button).press()
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.metadata["selected_kind"] == "persona"
        assert payload.metadata["selected_target_id"] == "local:persona:p-1"
        assert "Archivist" in payload.title
        assert "You are a meticulous archivist." in payload.body

    async def test_attach_aborts_when_profile_fetch_degraded(
        self, mock_app_instance, stub_characters, stub_conversations, stub_scope_service
    ):
        """A fallback (list-row) profile record must not stage silently."""
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        captured: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: captured.append(
            (str(message), severity)
        )
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_profile(pilot)
            # The service degrades after listing/selection succeeded.
            stub_scope_service.get_persona_profile = AsyncMock(
                side_effect=RuntimeError("service down")
            )
            screen.query_one("#personas-attach-to-console", Button).press()
            await pilot.pause()
        app.open_chat_with_handoff.assert_not_called()
        assert (
            "Persona is not fully loaded; try reselecting it.",
            "warning",
        ) in captured
        assert not any(severity == "information" for _msg, severity in captured)

    async def test_ctrl_enter_attaches_selected_character(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Ctrl+Enter stages the selected character, same as the Attach button."""
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            await self._select_first_character(pilot)
            await pilot.press("ctrl+enter")
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.metadata["selected_kind"] == "character"
        assert payload.metadata["selected_record_id"] == "1"

    async def test_attach_warns_when_handoff_unavailable(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A missing app handoff API warns instead of toasting success."""
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = None
        captured: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: captured.append(
            (str(message), severity)
        )
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.query_one("#personas-attach-to-console", Button).press()
            await pilot.pause()
        assert ("Console handoff is unavailable.", "warning") in captured
        assert not any(severity == "information" for _msg, severity in captured)

    async def test_conversation_continue_still_works(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The refactored shared seam preserves the Continue-in-Console contract."""
        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            await self._select_first_character(pilot)
            await pilot.click("#personas-conversation-row-conv-1")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-conversation-continue-console")
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.item_type == "character-conversation"
        assert payload.source_id == "conv-1"
        assert payload.metadata["conversation_id"] == "conv-1"
        assert payload.metadata["selected_kind"] == "character"
        assert payload.metadata["selected_record_id"] == "1"
        assert payload.metadata["selected_target_id"] == "local:character:1"
        assert payload.metadata["backend"] == "local"
        assert "Greetings, detective." in payload.body

    async def test_footer_shortcut_attach_available(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The attach action is truthful: allowed only with a saved, clean selection."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            # F-031: first paint auto-selects the first row, so attach is
            # allowed from the start...
            screen = await self._select_first_character(pilot)
            assert screen._console_action_allowed() is True
            # ...but the gate still closes the moment the selection is dirty.
            screen.state.has_unsaved_changes = True
            assert screen._console_action_allowed() is False


class TestServerCharacterSourceIsolation:
    """Server Characters never exposes or crosses into local character seams."""

    @staticmethod
    def _server_service(
        *,
        rows: list[dict] | None = None,
        detail: dict | None = None,
    ) -> SimpleNamespace:
        records = rows if rows is not None else [{"id": 7, "name": "Remote Elara"}]
        card = (
            detail
            if detail is not None
            else {
                "id": 7,
                "name": "Remote Elara",
                "description": "Server-owned card",
                "first_message": "Hello from the server.",
            }
        )
        return SimpleNamespace(
            list_characters=AsyncMock(
                return_value={
                    "items": [dict(row) for row in records],
                    "total": len(records),
                }
            ),
            search_characters=AsyncMock(
                return_value={
                    "items": [dict(row) for row in records],
                    "total": len(records),
                }
            ),
            get_character=AsyncMock(return_value=dict(card)),
        )

    async def test_source_switch_clears_local_rows_before_server_load_and_gates_actions(
        self, mock_app_instance, stub_characters
    ):
        """A blocked server fetch cannot leave the local page or local controls live."""
        import asyncio

        started = asyncio.Event()
        release = asyncio.Event()
        service = self._server_service()

        async def blocked_list(*, mode, limit, offset):
            assert mode == "server"
            started.set()
            await release.wait()
            return {"items": [{"id": 7, "name": "Remote Elara"}], "total": 1}

        service.list_characters.side_effect = blocked_list
        mock_app_instance.runtime_backend = "local"
        mock_app_instance.active_server_id = "server-a"
        mock_app_instance.character_persona_scope_service = service
        mock_app_instance.app_config = {
            "chat_defaults": {"provider": "llama_cpp", "model": "local.gguf"},
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:8181"}},
        }
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            assert screen.query_one("#personas-library-row-character-1")

            switch = asyncio.create_task(
                screen.handle_runtime_backend_changed("server")
            )
            await wait_for_background_signal(
                started, switch, what="the runtime-backend switch"
            )
            await pilot.pause()

            assert screen._characters == []
            assert screen._character_total == 0
            assert not list(screen.query("#personas-library-row-character-1"))

            release.set()
            await switch
            await pilot.pause()
            await pilot.click("#personas-library-row-character-7")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            for control_id in (
                "#personas-library-new",
                "#personas-library-import",
                "#personas-library-duplicate",
                "#personas-library-tag",
                "#personas-card-edit-character",
                "#personas-export-json",
                "#personas-export-png",
                "#personas-delete",
            ):
                control = screen.query_one(control_id, Button)
                assert control.disabled or not control.display
            assert (
                screen.query_one("#personas-attach-to-console", Button).disabled
                is False
            )
            assert screen.query_one("#personas-start-chat", Button).disabled is False

    async def test_server_browsing_disables_actions_with_reason_tooltips(
        self, mock_app_instance, stub_characters
    ):
        """F-037: every action disabled by server browsing says why."""
        mock_app_instance.runtime_backend = "server"
        mock_app_instance.active_server_id = "server-a"
        mock_app_instance.character_persona_scope_service = self._server_service()
        mock_app_instance.chat_dictionary_scope_service = None
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # F-031 auto-selects the first server row; the local-only actions
            # must be disabled AND explained.
            assert screen.state.selected_entity_id == "7"
            for control_id in (
                "#personas-card-edit-character",
                "#personas-export-json",
                "#personas-export-png",
                "#personas-delete",
            ):
                control = screen.query_one(control_id, Button)
                assert control.disabled is True, control_id
                assert control.tooltip == ("Server characters are read-only here."), (
                    control_id
                )

    async def test_server_footer_does_not_advertise_local_character_creation(
        self, mock_app_instance, stub_characters
    ):
        """Switching to server Characters removes the unsupported Ctrl+N hint."""
        mock_app_instance.runtime_backend = "local"
        mock_app_instance.active_server_id = "server-a"
        mock_app_instance.character_persona_scope_service = self._server_service()
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            footer = screen.query_one(AppFooterStatus)
            assert "ctrl+n new" in footer.shortcut_text.lower()

            await screen.handle_runtime_backend_changed("server")
            await pilot.pause()
            rendered = screen._shortcut_context().render().lower()
            footer_copy = footer.shortcut_text.lower()

            assert "ctrl+n new" not in rendered
            assert "ctrl+n new" not in footer_copy

    async def test_server_tag_request_does_not_schedule_local_worker(
        self, mock_app_instance
    ):
        """A queued Tag event cannot start local tag discovery on the server."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
            PersonaTagFilterRequested,
        )

        screen = PersonasScreen(mock_app_instance)
        screen.state.runtime_source = "server"
        screen.state.active_mode = "characters"
        screen.run_worker = Mock()

        await screen._handle_tag_filter(PersonaTagFilterRequested())

        screen.run_worker.assert_not_called()
        assert screen._io_dialog_active is False

    async def test_server_action_dispatch_and_direct_local_seams_fail_closed(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """Queued events and worker continuations are harmless after a source flip."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            CharacterSaveRequested,
            EditCharacterRequested,
        )

        service = self._server_service()
        mock_app_instance.runtime_backend = "server"
        mock_app_instance.active_server_id = "server-a"
        mock_app_instance.character_persona_scope_service = service
        app = PersonasTestApp(mock_app_instance)

        local_fetch = Mock(return_value=dict(CHARACTERS[0]))
        local_create = Mock(return_value=99)
        local_update = Mock(return_value=True)
        local_inspect = Mock(return_value=CharacterCardTTSInspection())
        local_import = Mock(
            return_value=CharacterCardImportOutcome(99, True, None, None)
        )
        local_delete = Mock(return_value=True)
        monkeypatch.setattr(
            character_handler_module, "fetch_character_by_id", local_fetch
        )
        monkeypatch.setattr(character_handler_module, "create_character", local_create)
        monkeypatch.setattr(character_handler_module, "update_character", local_update)
        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            local_inspect,
        )
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            local_import,
        )
        monkeypatch.setattr(character_handler_module, "delete_character", local_delete)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-library-row-character-7")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            begin_create_impl = screen._begin_create_character
            open_import_impl = screen._open_import_dialog
            duplicate_impl = screen._duplicate_selected_character
            begin_create = AsyncMock()
            open_import = AsyncMock()
            duplicate = AsyncMock()
            screen._begin_create_character = begin_create
            screen._open_import_dialog = open_import
            screen._duplicate_selected_character = duplicate
            for action in ("create", "import", "duplicate"):
                await screen._handle_action_requested(
                    PersonaActionRequested(action=action)
                )
            begin_create.assert_not_awaited()
            open_import.assert_not_awaited()
            duplicate.assert_not_awaited()
            screen._begin_create_character = begin_create_impl
            screen._open_import_dialog = open_import_impl
            screen._duplicate_selected_character = duplicate_impl

            run_worker = Mock()
            screen.run_worker = run_worker
            await begin_create_impl()
            await open_import_impl()
            await duplicate_impl()
            assert screen._edit_mode == "view"
            run_worker.assert_not_called()

            full_record = Mock(return_value=dict(CHARACTERS[0]))
            save_worker = Mock()
            screen._full_character_record = full_record
            screen._save_character_worker = save_worker
            screen._handle_edit_requested(EditCharacterRequested("7"))
            screen._handle_save_requested(
                CharacterSaveRequested({"name": "Must stay remote"})
            )
            full_record.assert_not_called()
            save_worker.assert_not_called()

            await PersonasScreen._save_character_worker.__wrapped__(
                screen,
                {"name": "Must stay remote"},
                "7",
                "edit",
            )
            await screen._import_character_from_path("/tmp/remote-card.json")
            await screen._export_selected_character("/tmp/remote-card.json", fmt="json")
            await screen._export_selected_character("/tmp/remote-card.png", fmt="png")
            await screen._delete_entity("character", "7", 1)

        local_fetch.assert_not_called()
        local_create.assert_not_called()
        local_update.assert_not_called()
        local_inspect.assert_not_called()
        local_import.assert_not_called()
        local_delete.assert_not_called()

    async def test_server_source_blocks_attachment_and_editor_mutation_messages(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """Hidden card/editor controls also fail closed when their messages race."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_character_dictionaries import (
            CharacterDictionaryAttachRequested,
            CharacterDictionaryDetachRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_character_world_books import (
            CharacterWorldBookAttachRequested,
            CharacterWorldBookDetachRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            CharacterAvatarGenerateRequested,
            CharacterExpressionClearRequested,
            CharacterExpressionGenerateAllRequested,
            CharacterExpressionGenerateRequested,
            CharacterExpressionSetExportRequested,
            CharacterExpressionSetImportRequested,
            CharacterExpressionStylePickRequested,
            CharacterExpressionUploadRequested,
            CharacterImageRemoveRequested,
            CharacterImageUploadRequested,
        )

        dictionary_service = SimpleNamespace(
            attach_to_character=AsyncMock(),
            detach_from_character=AsyncMock(),
        )
        lore_manager = SimpleNamespace(
            attach_world_book_to_character=Mock(),
            detach_world_book_from_character=Mock(),
        )
        mock_app_instance.runtime_backend = "server"
        mock_app_instance.active_server_id = "server-a"
        mock_app_instance.character_persona_scope_service = self._server_service()
        mock_app_instance.chat_dictionary_scope_service = dictionary_service
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.click("#personas-library-row-character-7")
            await pilot.pause()
            screen.run_worker = Mock()
            monkeypatch.setattr(
                PersonasScreen, "_lore_manager", lambda self: lore_manager
            )

            await screen._handle_character_dictionary_attach(
                CharacterDictionaryAttachRequested()
            )
            await screen._handle_character_dictionary_detach(
                CharacterDictionaryDetachRequested("facts")
            )
            await screen._handle_character_worldbook_attach(
                CharacterWorldBookAttachRequested()
            )
            await screen._handle_character_worldbook_detach(
                CharacterWorldBookDetachRequested("setting")
            )
            for message, handler in (
                (
                    CharacterImageUploadRequested(),
                    screen._handle_character_image_upload_requested,
                ),
                (
                    CharacterImageRemoveRequested(),
                    screen._handle_character_image_remove,
                ),
                (
                    CharacterExpressionUploadRequested("thinking"),
                    screen._handle_character_expression_upload_requested,
                ),
                (
                    CharacterExpressionGenerateRequested("thinking"),
                    screen._handle_character_expression_generate_requested,
                ),
                (
                    CharacterAvatarGenerateRequested(),
                    screen._handle_character_avatar_generate_requested,
                ),
                (
                    CharacterExpressionGenerateAllRequested(),
                    screen._handle_character_expression_generate_all_requested,
                ),
                (
                    CharacterExpressionStylePickRequested(),
                    screen._handle_expression_style_pick_requested,
                ),
                (
                    CharacterExpressionClearRequested("thinking"),
                    screen._handle_character_expression_clear_requested,
                ),
                (
                    CharacterExpressionSetImportRequested(),
                    screen._handle_expression_set_import_requested,
                ),
                (
                    CharacterExpressionSetExportRequested(),
                    screen._handle_expression_set_export_requested,
                ),
            ):
                handler(message)

            screen.run_worker.assert_not_called()
            dictionary_service.attach_to_character.assert_not_awaited()
            dictionary_service.detach_from_character.assert_not_awaited()
            lore_manager.attach_world_book_to_character.assert_not_called()
            lore_manager.detach_world_book_from_character.assert_not_called()

    async def test_local_page_result_is_dropped_after_source_switch(
        self, mock_app_instance, monkeypatch
    ):
        """A local DB read already in flight cannot publish into server mode."""
        import asyncio

        screen = PersonasScreen(mock_app_instance)
        screen.state.runtime_source = "local"
        screen._character_db = lambda: object()
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_to_thread(function, *args, **kwargs):
            if function is personas_screen_module.count_character_page:
                started.set()
                await release.wait()
                return 1
            return [{"id": 1, "name": "Late local row"}]

        monkeypatch.setattr(
            personas_screen_module.asyncio, "to_thread", blocked_to_thread
        )
        display = AsyncMock()
        screen._display_character_page = display

        load = asyncio.create_task(screen._reload_character_page())
        await wait_for_background_signal(
            started, load, what="the character page reload"
        )
        screen.state.runtime_source = "server"
        release.set()
        await load

        display.assert_not_awaited()

    async def test_local_page_failure_is_silent_after_source_switch(
        self, mock_app_instance, monkeypatch
    ):
        """A stale local failure cannot notify or restore rows after server switch."""
        import asyncio

        screen = PersonasScreen(mock_app_instance)
        screen.state.runtime_source = "local"
        screen._character_db = lambda: object()
        screen._characters = [{"id": 1, "name": "Old local row"}]
        screen._character_total = 1
        started = asyncio.Event()
        release = asyncio.Event()

        async def failing_to_thread(function, *args, **kwargs):
            started.set()
            await release.wait()
            raise RuntimeError("stale local failure")

        monkeypatch.setattr(
            personas_screen_module.asyncio, "to_thread", failing_to_thread
        )
        notify = Mock()
        screen._notify = notify

        load = asyncio.create_task(screen._reload_character_page())
        await wait_for_background_signal(
            started, load, what="the character page reload"
        )
        await screen.handle_runtime_backend_changed("server")
        assert screen._characters == []
        assert screen._character_total == 0

        release.set()
        await load

        notify.assert_not_called()
        assert screen._characters == []
        assert screen._character_total == 0

    async def test_stale_local_offset_clamp_cannot_mutate_newer_page(
        self, mock_app_instance, monkeypatch
    ):
        """A stale offset-100 count cannot clamp or render over offset 50."""
        import asyncio

        screen = PersonasScreen(mock_app_instance)
        screen.state.runtime_source = "local"
        screen.state.page_offset = 100
        screen._character_db = lambda: object()
        started = asyncio.Event()
        release = asyncio.Event()
        count_calls = 0

        async def interleaved_to_thread(function, *args, **kwargs):
            nonlocal count_calls
            if function is personas_screen_module.count_character_page:
                count_calls += 1
                if count_calls == 1:
                    started.set()
                    await release.wait()
                    return 1
                return 60
            if kwargs["offset"] == 50:
                return [{"id": 50, "name": "Newer offset winner"}]
            return [{"id": 1, "name": "Stale clamped row"}]

        monkeypatch.setattr(
            personas_screen_module.asyncio, "to_thread", interleaved_to_thread
        )
        notify = Mock()
        screen._notify = notify
        display = AsyncMock(wraps=screen._display_character_page)
        screen._display_character_page = display

        stale = asyncio.create_task(screen._reload_character_page())
        await wait_for_background_signal(
            started, stale, what="the stale character page reload"
        )
        screen.state.page_offset = 50
        await screen._reload_character_page()
        assert screen._characters == [{"id": 50, "name": "Newer offset winner"}]

        release.set()
        await stale

        assert screen.state.page_offset == 50
        assert screen._characters == [{"id": 50, "name": "Newer offset winner"}]
        assert screen._character_total == 60
        assert display.await_count == 1
        notify.assert_not_called()

    async def test_local_query_aba_drops_older_success(
        self, mock_app_instance, monkeypatch
    ):
        """Local X→Y→X keeps the newer X page when the older X returns last."""
        import asyncio

        screen = PersonasScreen(mock_app_instance)
        screen.state.runtime_source = "local"
        screen._character_db = lambda: object()
        started = asyncio.Event()
        release = asyncio.Event()
        count_calls = 0
        x_fetch_calls = 0

        async def interleaved_to_thread(function, *args, **kwargs):
            nonlocal count_calls, x_fetch_calls
            if function is personas_screen_module.count_character_page:
                count_calls += 1
                if count_calls == 1:
                    started.set()
                    await release.wait()
                return 1
            if kwargs["search_term"] is not None:
                return [{"id": 2, "name": "Y page"}]
            x_fetch_calls += 1
            if x_fetch_calls == 1:
                return [{"id": 3, "name": "Newer X winner"}]
            return [{"id": 1, "name": "Older X result"}]

        monkeypatch.setattr(
            personas_screen_module.asyncio, "to_thread", interleaved_to_thread
        )

        older_x = asyncio.create_task(screen._reload_character_page())
        await wait_for_background_signal(
            started, older_x, what="the older character page reload"
        )
        screen.state.search_query = "y"
        await screen._reload_character_page()
        screen.state.search_query = ""
        await screen._reload_character_page()
        assert screen._characters == [{"id": 3, "name": "Newer X winner"}]

        release.set()
        await older_x

        assert screen._characters == [{"id": 3, "name": "Newer X winner"}]

    async def test_mid_render_invalidation_clears_partial_stale_character_page(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """A newer failed reload leaves no older page published after its await."""
        import asyncio

        mock_app_instance.runtime_backend = "local"
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            library = screen.query_one("#personas-library-pane")
            original_update_rows = library.update_rows
            stale_render_started = asyncio.Event()
            release_stale_render = asyncio.Event()
            count_calls = 0

            async def interleaved_to_thread(function, *args, **kwargs):
                nonlocal count_calls
                if function is personas_screen_module.count_character_page:
                    count_calls += 1
                    if count_calls == 1:
                        return 1
                    raise RuntimeError("newer page failed")
                return [{"id": 7, "name": "Older X"}]

            async def block_after_row_publication(rows, **kwargs):
                await original_update_rows(rows, **kwargs)
                if tuple(row.name for row in rows) == ("Older X",):
                    stale_render_started.set()
                    await release_stale_render.wait()

            monkeypatch.setattr(
                personas_screen_module.asyncio,
                "to_thread",
                interleaved_to_thread,
            )
            monkeypatch.setattr(library, "update_rows", block_after_row_publication)
            screen._character_db = lambda: object()
            screen._count_cache_key = None
            screen._notify = Mock()
            screen.state.sort_key = "modified_desc"
            screen.state.tag_filter = "older-tag"

            stale = asyncio.create_task(screen._reload_character_page())
            await wait_for_background_signal(
                stale_render_started, stale, what="the stale character page render"
            )

            screen.state.sort_key = "name_asc"
            screen.state.tag_filter = None
            await screen._reload_character_page()

            release_stale_render.set()
            await stale
            await pilot.pause()

            assert screen._characters == []
            assert screen._character_total == 0
            assert not list(screen.query(".personas-library-row"))
            # F-033: the pane count line is empty for unfiltered lists; the
            # (zero) total shows in the merged header purpose line instead.
            assert (
                str(
                    screen.query_one(
                        "#personas-library-count",
                        Static,
                    ).renderable
                )
                == ""
            )
            assert "· 0" in str(
                screen.query_one("#personas-purpose", Static).renderable
            )
            assert (
                str(screen.query_one("#personas-library-sort", Button).label)
                == "Sort: Name"
            )
            assert (
                str(screen.query_one("#personas-library-tag", Button).label)
                == "Tag: All"
            )

    @pytest.mark.parametrize(
        (
            "new_mode",
            "new_kind",
            "new_name",
            "new_meta",
            "new_count",
        ),
        (
            # F-033: the count slot snapshots the pane's count line, which is
            # empty for unfiltered lists (the total moved to the merged
            # header purpose line).
            (
                "dictionaries",
                "dictionary",
                "New Dictionary Owner",
                "2 entries · on",
                "",
            ),
            (
                "lore",
                "lore",
                "New Lore Owner",
                "3 entries · on",
                "",
            ),
        ),
    )
    async def test_stale_character_render_preserves_newer_shared_mode_publication(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        new_mode,
        new_kind,
        new_name,
        new_meta,
        new_count,
    ):
        """A mode renderer started after Characters publishes must finish last."""
        import asyncio

        _configure_shared_mode_sources(mock_app_instance, monkeypatch)
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            library = screen.query_one("#personas-library-pane")
            original_update_rows = library.update_rows
            stale_render_started = asyncio.Event()
            release_stale_render = asyncio.Event()
            mode_render_started = asyncio.Event()
            owner_sort, owner_tag = _observe_shared_mode_render(
                monkeypatch,
                screen,
                new_mode,
                mode_render_started,
            )

            async def interleaved_to_thread(function, *args, **kwargs):
                if function is personas_screen_module.count_character_page:
                    return 1
                if function is personas_screen_module.get_character_page_for_ui:
                    return [{"id": 7, "name": "Older Character Publication"}]
                if function is PersonasScreen._list_world_books_with_counts:
                    return [
                        {
                            "id": 91,
                            "name": "New Lore Owner",
                            "entry_count": 3,
                            "enabled": True,
                        }
                    ]
                raise AssertionError(f"Unexpected to_thread function: {function!r}")

            async def block_after_character_publication(rows, **kwargs):
                await original_update_rows(rows, **kwargs)
                if tuple(row.name for row in rows) == ("Older Character Publication",):
                    stale_render_started.set()
                    await release_stale_render.wait()

            monkeypatch.setattr(
                personas_screen_module.asyncio,
                "to_thread",
                interleaved_to_thread,
            )
            monkeypatch.setattr(
                library,
                "update_rows",
                block_after_character_publication,
            )
            screen._character_db = lambda: object()
            screen._count_cache_key = None
            screen.state.sort_key = "modified_desc"
            screen.state.tag_filter = "older-tag"

            stale = asyncio.create_task(screen._reload_character_page())
            await wait_for_background_signal(
                stale_render_started, stale, what="the stale character page render"
            )

            newer_mode = asyncio.create_task(screen._apply_mode(new_mode))
            await wait_for_background_signal(
                mode_render_started, newer_mode, what="the newer mode render"
            )
            await pilot.pause()

            release_stale_render.set()
            await stale
            await newer_mode
            await pilot.pause()

            assert _shared_pane_publication(screen) == {
                "mode": new_mode,
                "rows": (
                    (
                        f"personas-library-row-{new_kind}-91",
                        (new_name, new_meta),
                    ),
                ),
                "count": new_count,
                "sort": owner_sort,
                "tag": owner_tag,
            }
            assert screen._characters == []
            assert screen._character_total == 0
            assert screen._count_cache_key is None

    @pytest.mark.parametrize(
        ("new_mode", "new_kind", "new_name", "new_meta", "new_count"),
        _SHARED_MODE_OWNER_CASES,
    )
    async def test_new_mode_wins_when_stale_character_waits_before_publication(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        new_mode,
        new_kind,
        new_name,
        new_meta,
        new_count,
    ):
        """A mode renderer started behind Characters must publish last."""
        import asyncio

        _configure_shared_mode_sources(mock_app_instance, monkeypatch)
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            library = screen.query_one("#personas-library-pane")
            original_update_rows = library.update_rows
            character_writer_entered = asyncio.Event()
            release_character_writer = asyncio.Event()
            mode_render_started = asyncio.Event()
            owner_sort, owner_tag = _observe_shared_mode_render(
                monkeypatch,
                screen,
                new_mode,
                mode_render_started,
            )

            async def interleaved_to_thread(function, *args, **kwargs):
                if function is personas_screen_module.count_character_page:
                    return 1
                if function is personas_screen_module.get_character_page_for_ui:
                    return [{"id": 7, "name": "Older Character Publication"}]
                if function is PersonasScreen._list_world_books_with_counts:
                    return [
                        {
                            "id": 91,
                            "name": "New Lore Owner",
                            "entry_count": 3,
                            "enabled": True,
                        }
                    ]
                raise AssertionError(f"Unexpected to_thread function: {function!r}")

            async def block_before_character_publication(rows, **kwargs):
                if tuple(row.name for row in rows) == ("Older Character Publication",):
                    character_writer_entered.set()
                    await release_character_writer.wait()
                await original_update_rows(rows, **kwargs)

            monkeypatch.setattr(
                personas_screen_module.asyncio,
                "to_thread",
                interleaved_to_thread,
            )
            monkeypatch.setattr(
                library,
                "update_rows",
                block_before_character_publication,
            )
            screen._character_db = lambda: object()
            screen._count_cache_key = None

            stale_character = asyncio.create_task(screen._reload_character_page())
            await wait_for_background_signal(
                character_writer_entered,
                stale_character,
                what="the stale character page reload",
            )

            newer_mode = asyncio.create_task(screen._apply_mode(new_mode))
            await wait_for_background_signal(
                mode_render_started, newer_mode, what="the newer mode render"
            )
            await pilot.pause()

            release_character_writer.set()
            await stale_character
            await newer_mode
            await pilot.pause()

            assert _shared_pane_publication(screen) == {
                "mode": new_mode,
                "rows": (
                    (
                        f"personas-library-row-{new_kind}-91",
                        (new_name, new_meta),
                    ),
                ),
                "count": new_count,
                "sort": owner_sort,
                "tag": owner_tag,
            }
            assert screen._characters == []
            assert screen._character_total == 0

    @pytest.mark.parametrize(
        ("new_mode", "new_kind", "new_name", "new_meta", "new_count"),
        _SHARED_MODE_OWNER_CASES,
    )
    async def test_new_mode_wins_when_stale_character_cleanup_is_suspended(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        new_mode,
        new_kind,
        new_name,
        new_meta,
        new_count,
    ):
        """A mode renderer started inside stale cleanup must publish last."""
        import asyncio

        _configure_shared_mode_sources(mock_app_instance, monkeypatch)
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            library = screen.query_one("#personas-library-pane")
            original_update_rows = library.update_rows
            initial_character_published = asyncio.Event()
            release_initial_character = asyncio.Event()
            cleanup_writer_entered = asyncio.Event()
            release_cleanup_writer = asyncio.Event()
            mode_render_started = asyncio.Event()
            owner_sort, owner_tag = _observe_shared_mode_render(
                monkeypatch,
                screen,
                new_mode,
                mode_render_started,
            )
            count_calls = 0

            async def interleaved_to_thread(function, *args, **kwargs):
                nonlocal count_calls
                if function is personas_screen_module.count_character_page:
                    count_calls += 1
                    if count_calls == 1:
                        return 1
                    raise RuntimeError("newer character page failed")
                if function is personas_screen_module.get_character_page_for_ui:
                    return [{"id": 7, "name": "Older Character Publication"}]
                if function is PersonasScreen._list_world_books_with_counts:
                    return [
                        {
                            "id": 91,
                            "name": "New Lore Owner",
                            "entry_count": 3,
                            "enabled": True,
                        }
                    ]
                raise AssertionError(f"Unexpected to_thread function: {function!r}")

            async def suspend_character_writers(rows, **kwargs):
                row_names = tuple(row.name for row in rows)
                if row_names == ("Older Character Publication",):
                    await original_update_rows(rows, **kwargs)
                    initial_character_published.set()
                    await release_initial_character.wait()
                    return
                if not rows and kwargs.get("noun") == "characters":
                    cleanup_writer_entered.set()
                    await release_cleanup_writer.wait()
                await original_update_rows(rows, **kwargs)

            monkeypatch.setattr(
                personas_screen_module.asyncio,
                "to_thread",
                interleaved_to_thread,
            )
            monkeypatch.setattr(
                library,
                "update_rows",
                suspend_character_writers,
            )
            screen._character_db = lambda: object()
            screen._count_cache_key = None
            screen.state.tag_filter = "older-tag"

            stale_character = asyncio.create_task(screen._reload_character_page())
            await wait_for_background_signal(
                initial_character_published,
                stale_character,
                what="the initial character publication",
            )

            screen.state.tag_filter = None
            await screen._reload_character_page()
            release_initial_character.set()
            await wait_for_background_signal(
                cleanup_writer_entered,
                stale_character,
                what="the stale reload's cleanup writer",
            )

            newer_mode = asyncio.create_task(screen._apply_mode(new_mode))
            await wait_for_background_signal(
                mode_render_started, newer_mode, what="the newer mode render"
            )
            await pilot.pause()

            release_cleanup_writer.set()
            await stale_character
            await newer_mode
            await pilot.pause()

            assert _shared_pane_publication(screen) == {
                "mode": new_mode,
                "rows": (
                    (
                        f"personas-library-row-{new_kind}-91",
                        (new_name, new_meta),
                    ),
                ),
                "count": new_count,
                "sort": owner_sort,
                "tag": owner_tag,
            }
            assert screen._characters == []
            assert screen._character_total == 0
            assert screen._count_cache_key is None

    @pytest.mark.parametrize(
        ("source_mode", "owner_change"),
        (
            pytest.param("dictionaries", "query", id="dictionaries-query"),
            pytest.param("dictionaries", "mode", id="dictionaries-mode"),
            pytest.param("lore", "query", id="lore-query"),
            pytest.param("lore", "mode", id="lore-mode"),
        ),
    )
    async def test_slow_shared_mode_fetch_cannot_overwrite_changed_owner(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        source_mode,
        owner_change,
    ):
        """Dictionary/Lore results publish only for their captured mode/query."""
        import asyncio

        fetch_started = asyncio.Event()
        release_fetch = asyncio.Event()
        dictionary_calls = 0

        async def list_dictionaries(*, mode, include_inactive):
            nonlocal dictionary_calls
            assert (mode, include_inactive) == ("local", True)
            dictionary_calls += 1
            if source_mode == "dictionaries" and dictionary_calls == 1:
                fetch_started.set()
                await release_fetch.wait()
                name = "Old Dictionary Result"
            else:
                name = "new Dictionary Owner"
            return {
                "dictionaries": [
                    {
                        "id": 91,
                        "name": name,
                        "entry_count": 2,
                        "enabled": True,
                    }
                ]
            }

        mock_app_instance.runtime_backend = "local"
        mock_app_instance.chat_dictionary_scope_service = SimpleNamespace(
            list_dictionaries=list_dictionaries
        )
        monkeypatch.setattr(
            PersonasScreen,
            "_lore_manager",
            lambda self: object(),
        )
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            lore_calls = 0

            async def interleaved_to_thread(function, *args, **kwargs):
                nonlocal lore_calls
                if function is PersonasScreen._list_world_books_with_counts:
                    lore_calls += 1
                    if source_mode == "lore" and lore_calls == 1:
                        fetch_started.set()
                        await release_fetch.wait()
                        name = "Old Lore Result"
                    else:
                        name = "new Lore Owner"
                    return [
                        {
                            "id": 91,
                            "name": name,
                            "entry_count": 3,
                            "enabled": True,
                        }
                    ]
                raise AssertionError(f"Unexpected to_thread function: {function!r}")

            monkeypatch.setattr(
                personas_screen_module.asyncio,
                "to_thread",
                interleaved_to_thread,
            )

            stale_fetch = asyncio.create_task(screen._apply_mode(source_mode))
            await wait_for_background_signal(
                fetch_started, stale_fetch, what="the stale shared-mode fetch"
            )

            library = screen.query_one("#personas-library-pane")
            if owner_change == "query":
                screen.state.search_query = "new"
                if source_mode == "dictionaries":
                    await screen._render_dictionary_rows(query="new")
                    expected_rows = (
                        (
                            "personas-library-row-dictionary-91",
                            ("new Dictionary Owner", "2 entries · on"),
                        ),
                    )
                    expected_count = "1 of 1 dictionary"
                else:
                    await screen._render_lore_rows(query="new")
                    expected_rows = (
                        (
                            "personas-library-row-lore-91",
                            ("new Lore Owner", "3 entries · on"),
                        ),
                    )
                    expected_count = "1 of 1 lore book"
                expected_mode = source_mode
            else:
                await screen._apply_mode("prompts")
                expected_mode = "prompts"
                expected_rows = ()
                # F-033: unfiltered lists leave the pane count line empty;
                # the total lives in the merged header purpose line.
                expected_count = ""

            owner_sort = f"Sort: {expected_mode} current owner"
            owner_tag = f"Tag: {expected_mode} current owner"
            library.set_sort_label(owner_sort)
            library.set_tag_label(owner_tag)
            await pilot.pause()
            current_publication = _shared_pane_publication(screen)

            release_fetch.set()
            await stale_fetch
            await pilot.pause()

            assert current_publication == {
                "mode": expected_mode,
                "rows": expected_rows,
                "count": expected_count,
                "sort": owner_sort,
                "tag": owner_tag,
            }
            assert _shared_pane_publication(screen) == current_publication

    @pytest.mark.parametrize("source_mode", ("dictionaries", "lore"))
    @pytest.mark.parametrize("older_outcome", ("success", "failure"))
    async def test_shared_mode_query_aba_drops_older_completion(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        source_mode,
        older_outcome,
    ):
        """Dictionary/Lore X→Y→X keeps the newer X rows and cache."""
        import asyncio

        fetch_started = asyncio.Event()
        release_fetch = asyncio.Event()
        call_count = 0
        kind = "dictionary" if source_mode == "dictionaries" else "lore"
        cache_name = (
            "_dictionaries_cache"
            if source_mode == "dictionaries"
            else "_lore_books_cache"
        )
        entry_count = 2 if source_mode == "dictionaries" else 3

        def record(item_id: int, name: str) -> dict:
            return {
                "id": item_id,
                "name": name,
                "entry_count": entry_count,
                "enabled": True,
            }

        old_x = record(71, "X older result")
        y_result = record(72, "Y intermediate result")
        newer_x = record(73, "X newer winner")

        async def fetch_records() -> list[dict]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                fetch_started.set()
                await release_fetch.wait()
                if older_outcome == "failure":
                    raise RuntimeError("older X failed")
                return [old_x]
            if call_count == 2:
                return [y_result]
            if call_count == 3:
                return [newer_x]
            raise AssertionError(f"Unexpected fetch call: {call_count}")

        async def list_dictionaries(*, mode, include_inactive):
            assert (mode, include_inactive) == ("local", True)
            return {"dictionaries": await fetch_records()}

        _configure_shared_mode_sources(mock_app_instance, monkeypatch)
        mock_app_instance.chat_dictionary_scope_service.list_dictionaries = (
            list_dictionaries
        )
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)

            async def interleaved_to_thread(function, *args, **kwargs):
                if function is PersonasScreen._list_world_books_with_counts:
                    return await fetch_records()
                raise AssertionError(f"Unexpected to_thread function: {function!r}")

            monkeypatch.setattr(
                personas_screen_module.asyncio,
                "to_thread",
                interleaved_to_thread,
            )
            library = screen.query_one("#personas-library-pane")
            screen.state.switch_mode(source_mode)
            screen.state.search_query = "x"
            library.set_mode(source_mode)
            render = (
                screen._render_dictionary_rows
                if source_mode == "dictionaries"
                else screen._render_lore_rows
            )

            older_x_request = asyncio.create_task(render(query="x"))
            await wait_for_background_signal(
                fetch_started, older_x_request, what="the older render request"
            )

            screen.state.search_query = "y"
            await render(query="y")
            screen.state.search_query = "x"
            await render(query="x")
            await pilot.pause()

            newer_publication = _shared_pane_publication(screen)
            newer_cache = deepcopy(getattr(screen, cache_name))
            assert newer_publication["rows"] == (
                (
                    f"personas-library-row-{kind}-73",
                    ("X newer winner", f"{entry_count} entries · on"),
                ),
            )
            assert newer_cache == [newer_x]

            release_fetch.set()
            await older_x_request
            await pilot.pause()

            assert _shared_pane_publication(screen) == newer_publication
            assert getattr(screen, cache_name) == newer_cache

    @pytest.mark.parametrize("source_mode", ("dictionaries", "lore"))
    async def test_shared_mode_away_and_back_drops_older_completion(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        source_mode,
    ):
        """Dictionary/Lore mode-away→back keeps the newer request owner."""
        import asyncio

        fetch_started = asyncio.Event()
        release_fetch = asyncio.Event()
        call_count = 0
        kind = "dictionary" if source_mode == "dictionaries" else "lore"
        cache_name = (
            "_dictionaries_cache"
            if source_mode == "dictionaries"
            else "_lore_books_cache"
        )
        entry_count = 2 if source_mode == "dictionaries" else 3

        def record(item_id: int, name: str) -> dict:
            return {
                "id": item_id,
                "name": name,
                "entry_count": entry_count,
                "enabled": True,
            }

        older_result = record(81, "Older before mode return")
        newer_result = record(82, "Newer after mode return")

        async def fetch_records() -> list[dict]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                fetch_started.set()
                await release_fetch.wait()
                return [older_result]
            if call_count == 2:
                return [newer_result]
            raise AssertionError(f"Unexpected fetch call: {call_count}")

        async def list_dictionaries(*, mode, include_inactive):
            assert (mode, include_inactive) == ("local", True)
            return {"dictionaries": await fetch_records()}

        _configure_shared_mode_sources(mock_app_instance, monkeypatch)
        mock_app_instance.chat_dictionary_scope_service.list_dictionaries = (
            list_dictionaries
        )
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)

            async def interleaved_to_thread(function, *args, **kwargs):
                if function is PersonasScreen._list_world_books_with_counts:
                    return await fetch_records()
                raise AssertionError(f"Unexpected to_thread function: {function!r}")

            monkeypatch.setattr(
                personas_screen_module.asyncio,
                "to_thread",
                interleaved_to_thread,
            )
            library = screen.query_one("#personas-library-pane")
            screen.state.switch_mode(source_mode)
            screen.state.search_query = ""
            library.set_mode(source_mode)
            render = (
                screen._render_dictionary_rows
                if source_mode == "dictionaries"
                else screen._render_lore_rows
            )

            older_request = asyncio.create_task(render())
            await wait_for_background_signal(
                fetch_started, older_request, what="the older render request"
            )

            await screen._apply_mode("prompts")
            await screen._apply_mode(source_mode)
            await pilot.pause()

            newer_publication = _shared_pane_publication(screen)
            newer_cache = deepcopy(getattr(screen, cache_name))
            assert newer_publication["rows"] == (
                (
                    f"personas-library-row-{kind}-82",
                    ("Newer after mode return", f"{entry_count} entries · on"),
                ),
            )
            assert newer_cache == [newer_result]

            release_fetch.set()
            await older_request
            await pilot.pause()

            assert _shared_pane_publication(screen) == newer_publication
            assert getattr(screen, cache_name) == newer_cache

    @pytest.mark.parametrize("valid_mode", ("dictionaries", "lore"))
    @pytest.mark.parametrize(
        "stale_callback",
        ("wrong-mode", "wrong-query"),
    )
    @pytest.mark.parametrize("valid_outcome", ("success", "failure"))
    async def test_invalid_shared_mode_callback_does_not_supersede_valid_owner(
        self,
        mock_app_instance,
        stub_characters,
        monkeypatch,
        valid_mode,
        stale_callback,
        valid_outcome,
    ):
        """Rejected callbacks cannot invalidate an accepted suspended owner."""
        import asyncio

        fetch_started = asyncio.Event()
        release_fetch = asyncio.Event()
        fetch_calls = 0
        kind = "dictionary" if valid_mode == "dictionaries" else "lore"
        cache_name = (
            "_dictionaries_cache"
            if valid_mode == "dictionaries"
            else "_lore_books_cache"
        )
        entry_count = 2 if valid_mode == "dictionaries" else 3
        owner_record = {
            "id": 93,
            "name": f"Owner {kind.title()} Result",
            "entry_count": entry_count,
            "enabled": True,
        }
        prior_cache = [
            {
                "id": 90,
                "name": f"Prior {kind.title()} Cache",
                "entry_count": 1,
                "enabled": True,
            }
        ]

        async def fetch_records() -> list[dict]:
            nonlocal fetch_calls
            fetch_calls += 1
            if fetch_calls != 1:
                raise AssertionError(f"Unexpected fetch call: {fetch_calls}")
            fetch_started.set()
            await release_fetch.wait()
            if valid_outcome == "failure":
                raise RuntimeError("accepted owner failed")
            return [owner_record]

        async def list_dictionaries(*, mode, include_inactive):
            assert (mode, include_inactive) == ("local", True)
            return {"dictionaries": await fetch_records()}

        _configure_shared_mode_sources(mock_app_instance, monkeypatch)
        mock_app_instance.chat_dictionary_scope_service.list_dictionaries = (
            list_dictionaries
        )
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)

            async def interleaved_to_thread(function, *args, **kwargs):
                if function is PersonasScreen._list_world_books_with_counts:
                    return await fetch_records()
                raise AssertionError(f"Unexpected to_thread function: {function!r}")

            monkeypatch.setattr(
                personas_screen_module.asyncio,
                "to_thread",
                interleaved_to_thread,
            )
            library = screen.query_one("#personas-library-pane")
            screen.state.switch_mode(valid_mode)
            screen.state.search_query = "owner"
            library.set_mode(valid_mode)
            setattr(screen, cache_name, deepcopy(prior_cache))

            valid_render = (
                screen._render_dictionary_rows
                if valid_mode == "dictionaries"
                else screen._render_lore_rows
            )
            generation_before = screen._dictionary_lore_request_generation
            valid_request = asyncio.create_task(valid_render(query="owner"))
            await wait_for_background_signal(
                fetch_started, valid_request, what="the valid render request"
            )
            accepted_generation = screen._dictionary_lore_request_generation

            if stale_callback == "wrong-mode":
                invalid_render = (
                    screen._render_lore_rows
                    if valid_mode == "dictionaries"
                    else screen._render_dictionary_rows
                )
                await invalid_render(query="owner")
            else:
                await valid_render(query="stale")
            generation_after_invalid = screen._dictionary_lore_request_generation

            release_fetch.set()
            await valid_request
            await pilot.pause()

            publication = _shared_pane_publication(screen)
            recovery_rows = list(
                screen.query(".personas-library-recovery-row").results()
            )
            recovery_copy = (
                str(recovery_rows[0].query_one(Static).renderable)
                if recovery_rows
                else None
            )
            if valid_outcome == "success":
                expected_rows = (
                    (
                        f"personas-library-row-{kind}-93",
                        (
                            f"Owner {kind.title()} Result",
                            f"{entry_count} entries · on",
                        ),
                    ),
                )
                expected_count = (
                    "1 of 1 dictionary"
                    if valid_mode == "dictionaries"
                    else "1 of 1 lore book"
                )
                expected_cache = [owner_record]
                expected_recovery = None
            else:
                expected_rows = ()
                expected_count = (
                    "Dictionaries unavailable"
                    if valid_mode == "dictionaries"
                    else "Lore books unavailable"
                )
                expected_cache = prior_cache
                expected_recovery = (
                    "Dictionaries could not be loaded.\nSwitch modes and back to retry."
                    if valid_mode == "dictionaries"
                    else "Lore books could not be loaded.\n"
                    "Switch modes and back to retry."
                )

            assert (
                accepted_generation,
                generation_after_invalid,
                fetch_calls,
                publication["rows"],
                publication["count"],
                getattr(screen, cache_name),
                recovery_copy,
            ) == (
                generation_before + 1,
                accepted_generation,
                1,
                expected_rows,
                expected_count,
                expected_cache,
                expected_recovery,
            )

    @pytest.mark.parametrize("older_outcome", ("success", "failure"))
    async def test_server_target_aba_drops_older_request(
        self, mock_app_instance, older_outcome
    ):
        """Server X→Y→X keeps newer X rows and ignores older X completion."""
        import asyncio

        started = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def list_characters(*, mode, limit, offset):
            nonlocal call_count
            assert mode == "server"
            call_count += 1
            if call_count == 1:
                started.set()
                await release.wait()
                if older_outcome == "failure":
                    raise RuntimeError("older X failed")
                return {"items": [{"id": 1, "name": "Older X"}], "total": 1}
            if mock_app_instance.active_server_id == "target-y":
                return {"items": [{"id": 2, "name": "Y page"}], "total": 1}
            return {"items": [{"id": 3, "name": "Newer X winner"}], "total": 1}

        mock_app_instance.runtime_backend = "server"
        mock_app_instance.active_server_id = "target-x"
        mock_app_instance.character_persona_scope_service = SimpleNamespace(
            list_characters=AsyncMock(side_effect=list_characters)
        )
        screen = PersonasScreen(mock_app_instance)
        notify = Mock()
        screen._notify = notify

        older_x = asyncio.create_task(screen._reload_server_character_page())
        await wait_for_background_signal(
            started, older_x, what="the older server character page load"
        )
        mock_app_instance.active_server_id = "target-y"
        await screen._reload_server_character_page()
        mock_app_instance.active_server_id = "target-x"
        await screen._reload_server_character_page()
        assert screen._characters == [{"id": 3, "name": "Newer X winner"}]

        release.set()
        await older_x

        assert screen._characters == [{"id": 3, "name": "Newer X winner"}]
        assert screen._character_total == 1
        notify.assert_not_called()

    @pytest.mark.parametrize(
        "changed_dimension",
        ("source", "target", "mode", "query", "sort", "tag", "offset"),
    )
    async def test_server_page_result_is_dropped_after_request_snapshot_changes(
        self, mock_app_instance, changed_dimension
    ):
        """Every source/target/view facet fences an in-flight server result."""
        import asyncio

        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_list(*, mode, limit, offset):
            assert mode == "server"
            started.set()
            await release.wait()
            return {"items": [{"id": 9, "name": "Late server row"}], "total": 1}

        mock_app_instance.runtime_backend = "server"
        mock_app_instance.active_server_id = "target-a"
        mock_app_instance.character_persona_scope_service = SimpleNamespace(
            list_characters=AsyncMock(side_effect=blocked_list)
        )
        screen = PersonasScreen(mock_app_instance)
        screen._characters = [{"id": 1, "name": "Old row"}]
        screen._character_total = 1

        load = asyncio.create_task(screen._reload_server_character_page())
        await wait_for_background_signal(
            started, load, what="the server character page load"
        )
        if changed_dimension == "source":
            screen.state.runtime_source = "local"
        elif changed_dimension == "target":
            mock_app_instance.active_server_id = "target-b"
        elif changed_dimension == "mode":
            screen.state.active_mode = "personas"
        elif changed_dimension == "query":
            screen.state.search_query = "new query"
        elif changed_dimension == "sort":
            screen.state.sort_key = "date"
        elif changed_dimension == "tag":
            screen.state.tag_filter = "new-tag"
        else:
            screen.state.page_offset = personas_screen_module.PERSONAS_LIBRARY_PAGE_SIZE
        release.set()
        await load

        assert screen._characters == []
        assert screen._character_total == 0

    async def test_target_b_winner_survives_stale_target_a_failure(
        self, mock_app_instance
    ):
        """A failed A request cannot clear the newer B page."""
        import asyncio

        started_a = asyncio.Event()
        release_a = asyncio.Event()

        async def list_characters(*, mode, limit, offset):
            assert mode == "server"
            if mock_app_instance.active_server_id == "target-a":
                started_a.set()
                await release_a.wait()
                raise RuntimeError("target A went away")
            return {"items": [{"id": 2, "name": "Target B"}], "total": 1}

        mock_app_instance.runtime_backend = "server"
        mock_app_instance.active_server_id = "target-a"
        mock_app_instance.character_persona_scope_service = SimpleNamespace(
            list_characters=AsyncMock(side_effect=list_characters)
        )
        screen = PersonasScreen(mock_app_instance)
        screen._characters = [{"id": 1, "name": "Old A"}]
        screen._character_total = 1

        load_a = asyncio.create_task(screen._reload_server_character_page())
        await wait_for_background_signal(
            started_a, load_a, what="the target-A server character page load"
        )
        mock_app_instance.active_server_id = "target-b"
        await screen._reload_server_character_page()
        assert screen._characters == [{"id": 2, "name": "Target B"}]

        release_a.set()
        await load_a
        assert screen._characters == [{"id": 2, "name": "Target B"}]
        assert screen._character_total == 1

    @pytest.mark.parametrize(
        "failure",
        ("missing-target", "missing-service", "network"),
    )
    async def test_server_load_failures_replace_old_rows_with_empty_page(
        self, mock_app_instance, failure
    ):
        mock_app_instance.runtime_backend = "server"
        mock_app_instance.active_server_id = (
            None if failure == "missing-target" else "target-a"
        )
        if failure == "missing-service":
            service = SimpleNamespace()
        elif failure == "network":
            service = SimpleNamespace(
                list_characters=AsyncMock(side_effect=RuntimeError("offline"))
            )
        else:
            service = SimpleNamespace(
                list_characters=AsyncMock(return_value={"items": [], "total": 0})
            )
        mock_app_instance.character_persona_scope_service = service
        screen = PersonasScreen(mock_app_instance)
        screen._characters = [{"id": 1, "name": "Stale row"}]
        screen._character_total = 1
        screen._count_cache_key = ("stale", None)

        await screen._reload_server_character_page()

        assert screen._characters == []
        assert screen._character_total == 0
        assert screen._count_cache_key is None

    async def test_server_search_pages_at_api_limit_without_third_page(
        self, mock_app_instance
    ):
        rows = [{"id": index, "name": f"Remote {index:03d}"} for index in range(1, 121)]

        async def search_characters(query, *, mode, limit):
            assert query == "remote"
            assert mode == "server"
            return {"items": rows[:limit], "total": len(rows)}

        service = self._server_service(rows=[])
        service.search_characters.side_effect = search_characters
        mock_app_instance.runtime_backend = "server"
        mock_app_instance.active_server_id = "target-a"
        mock_app_instance.character_persona_scope_service = service
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            screen.state.search_query = "remote"
            await screen._reload_character_page(reset_offset=True)
            assert [row["id"] for row in screen._characters] == list(range(1, 51))

            await screen._on_page_changed_delta(1)
            assert [row["id"] for row in screen._characters] == list(range(51, 101))
            assert screen._character_total == 100
            assert screen.query_one("#personas-library-next", Button).disabled is True

            await screen._on_page_changed_delta(1)
            assert screen.state.page_offset == 50

        assert [
            call.kwargs["limit"] for call in service.search_characters.await_args_list
        ] == [51, 100]


class _FakePreviewGateway:
    """In-memory gateway double: ready resolution + scripted stream.

    ``gate`` holds the stream BEFORE the first chunk; ``mid_gate`` holds it
    between the first chunk and the rest; ``error`` raises before any chunk
    unless ``error_after_first`` is set, in which case the first chunk is
    yielded and the error raised on the next pull. ``stream_failures`` makes
    that many stream_chat calls raise before any chunk, then later calls
    succeed (exercises the non-streaming retry). ``selections`` records every
    selection passed to resolve_for_send so tests can assert the retry's
    ``streaming`` flag.
    """

    def __init__(
        self,
        chunks=("Hello, ", "world."),
        gate=None,
        mid_gate=None,
        error=None,
        resolve_error=None,
        error_after_first=False,
        stream_failures=0,
    ):
        self.chunks = chunks
        self.gate = gate
        self.mid_gate = mid_gate
        self.error = error
        self.resolve_error = resolve_error
        self.error_after_first = error_after_first
        self.stream_failures = stream_failures
        self.requests: list[list[dict]] = []
        self.selections: list = []
        self.closed = False

    async def resolve_for_send(self, selection):
        from tldw_chatbook.Chat.console_provider_gateway import (
            ConsoleProviderResolution,
        )

        self.selections.append(selection)
        if self.resolve_error is not None:
            raise self.resolve_error
        return ConsoleProviderResolution(
            provider="openai", base_url="", model="test-model", ready=True
        )

    async def stream_chat(self, resolution, messages, **_kwargs):
        self.requests.append([dict(m) for m in messages])
        if self.gate is not None:
            await self.gate.wait()
        if self.stream_failures > 0:
            self.stream_failures -= 1
            raise RuntimeError("stream transport failed")
        if self.error is not None and not self.error_after_first:
            raise self.error
        for index, chunk in enumerate(self.chunks):
            if index == 1 and self.mid_gate is not None:
                await self.mid_gate.wait()
            yield chunk
            if index == 0 and self.error is not None and self.error_after_first:
                raise self.error

    async def aclose(self):
        self.closed = True


class ReadinessMapPreviewGateway(_FakePreviewGateway):
    """Fake gateway whose readiness depends on the selection's provider.

    Providers in ``ready_providers`` resolve ready; anything else resolves
    blocked with Console-style visible copy, so tests can drive the
    character-defaults -> chat_defaults fallback (task-425).
    """

    def __init__(self, ready_providers, resolve_error_providers=None, **kwargs):
        super().__init__(**kwargs)
        self.ready_providers = {p.lower() for p in ready_providers}
        self.resolve_error_providers = {
            p.lower() for p in (resolve_error_providers or set())
        }

    async def resolve_for_send(self, selection):
        from tldw_chatbook.Chat.console_provider_gateway import (
            ConsoleProviderResolution,
        )

        self.selections.append(selection)
        provider = (selection.provider or "").lower()
        if provider in self.resolve_error_providers:
            raise RuntimeError(f"{provider} resolution exploded")
        if provider in self.ready_providers:
            return ConsoleProviderResolution(
                provider=provider,
                base_url="",
                model=selection.explicit_model or "test-model",
                ready=True,
            )
        return ConsoleProviderResolution(
            provider=provider,
            base_url="",
            model=selection.explicit_model or "",
            ready=False,
            visible_copy=f"{provider or 'provider'} is not ready: Missing API key.",
        )


class TestPreviewIntegration:
    """Ephemeral preview-conversation pane wiring on the screen (Task 13)."""

    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        _install_conversation_db(monkeypatch, [])

    async def _select_first_character(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def test_preview_logic_is_owned_by_controller(
        self, mock_app_instance, stub_characters
    ):
        from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
            PersonasPreviewController,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            assert isinstance(screen.preview, PersonasPreviewController)
            assert screen.preview.screen is screen

    async def test_preview_pane_is_mounted_in_work_area(
        self, mock_app_instance, stub_characters
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            pane = screen.query_one("#personas-preview-pane", PersonasPreviewPane)
            work_area = screen.query_one("#personas-work-area")
            assert pane in work_area.children

    async def test_mode_switch_resets_stale_character_speaker_label(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        # task-437: after selecting a character then leaving Characters mode,
        # the preview must not keep the previous character's speaker label
        # (else a persona Test Reply would render under the stale name).
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            pane = screen.query_one("#personas-preview-pane", PersonasPreviewPane)
            assert pane._character_label == "Detective Sam"
            await screen._apply_mode("personas")
            await pilot.pause()
            assert pane._character_label == "character"

    async def test_delete_selected_character_resets_speaker_label(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        # task-437 review: deleting the selected character drops its speaker
        # label so a later Test Reply isn't labelled with the deleted name (the
        # preview stays live/visible in Characters mode).
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            pane = screen.query_one("#personas-preview-pane", PersonasPreviewPane)
            assert pane._character_label == "Detective Sam"
            await screen._after_delete("character")
            await pilot.pause()
            assert pane._character_label == "character"

    async def test_greeting_seeds_after_character_load(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The greeting seeds once the load worker delivers the full card.

        ``load_character`` only schedules a thread worker, so the full record
        (with ``first_message``) is not available synchronously at selection
        time; the screen must seed from the load-completion message instead.
        """
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert (
                "Detective Sam: The name's Detective Sam. Who's asking?"
                in pane.transcript_text()
            )

    async def test_reselect_does_not_duplicate_greeting(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Re-selecting a character seeds exactly one greeting line."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        async def _select(pilot, row_id):
            await pilot.click(row_id)
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            await pilot.pause()
            await _select(pilot, "#personas-library-row-character-2")
            await _select(pilot, "#personas-library-row-character-1")
            pane = screen.query_one(PersonasPreviewPane)
            greeting_line = "Detective Sam: The name's Detective Sam. Who's asking?"
            lines = [line for line in pane.transcript_text().splitlines() if line]
            assert lines == [greeting_line]

    async def test_reset_after_character_reload_uses_updated_greeting(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A same-character reload refreshes Reset's stored greeting seed.

        Args:
            mock_app_instance: Fixture providing the app object used by the
                mounted Personas test app.
            stub_characters: Fixture stubbing local character list/load data.
            stub_conversations: Fixture stubbing character conversation data.
        """
        from tldw_chatbook.UI.CCP_Modules.ccp_messages import CharacterMessage
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            pane = screen.query_one(PersonasPreviewPane)
            assert (
                "Detective Sam: The name's Detective Sam. Who's asking?"
                in pane.transcript_text()
            )

            screen.post_message(
                CharacterMessage.Loaded(
                    "1",
                    {
                        "id": 1,
                        "name": "Detective Sam",
                        "first_message": "Edited opener from {{char}} to {{user}}.",
                        "version": 2,
                    },
                )
            )
            await pilot.pause()
            assert pane.transcript_text() == (
                "Detective Sam: The name's Detective Sam. Who's asking?"
            )
            screen.query_one("#personas-preview-reset", Button).press()
            await pilot.pause()

            assert pane.transcript_text() == (
                "Detective Sam: Edited opener from Detective Sam to User."
            )

    async def test_alternate_greeting_selector_seeds_and_reset_returns_to_choice(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """TASK-438: picking an alternate greeting re-seeds the preview, and
        Reset returns to the CHOSEN greeting, not the primary."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            pane = screen.query_one("#personas-preview-pane", PersonasPreviewPane)
            assert screen.query_one("#personas-preview-greeting-row").display is True
            # choose alternate index 1
            await screen.preview.handle_greeting_selected(1)
            await pilot.pause()
            assert "An alternate opener." in pane.transcript_text()
            # send a turn, then Reset -> returns to the CHOSEN alternate, not primary
            pane.append_user("hi")
            pane.append_reply("hello")
            await pilot.pause()
            await pane.reset()
            await pilot.pause()
            assert "An alternate opener." in pane.transcript_text()
            assert "hi" not in pane.transcript_text()

    async def test_reload_preserves_chosen_alternate_greeting(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """TASK-438 review: a same-character reload (edit+save) that preserves the
        in-progress transcript keeps the CHOSEN alternate greeting, so Reset still
        returns to it rather than silently reverting to the primary."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            pane = screen.query_one("#personas-preview-pane", PersonasPreviewPane)
            await screen.preview.handle_greeting_selected(1)  # choose alternate
            await pilot.pause()
            pane.append_user("hi")
            pane.append_reply("hello")
            await pilot.pause()
            # Same-character reload (the task-437 preserve path): CharacterMessage.Loaded
            await screen.preview.handle_character_loaded(
                character_id="1",
                card_data={
                    "name": "Detective Sam",
                    "first_message": "The name's {{char}}. Who's asking?",
                    "alternate_greetings": ["An alternate opener.", "A third opener."],
                },
            )
            await pilot.pause()
            # the chosen index is preserved (selector + Reset seed), not reset to 0
            assert screen.preview._current_greeting_index == 1
            await pane.reset()
            await pilot.pause()
            assert "An alternate opener." in pane.transcript_text()
            assert "Who's asking" not in pane.transcript_text()

    async def test_reload_via_real_select_does_not_wipe_transcript(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """TASK-438 review: choosing an alternate through the REAL Select widget
        then a same-character reload must NOT wipe the in-progress transcript —
        the programmatic set_options() must not fire a spurious re-seed."""
        from textual.widgets import Select
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            pane = screen.query_one("#personas-preview-pane", PersonasPreviewPane)
            # pick alternate index 1 via the real widget (fires Select.Changed)
            pane.query_one("#personas-preview-greeting-select", Select).value = 1
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            pane.append_user("hi")
            pane.append_reply("hello")
            await pilot.pause()
            # same-character reload (edit+save path)
            await screen.preview.handle_character_loaded(
                character_id="1",
                card_data={
                    "name": "Detective Sam",
                    "first_message": "The name's {{char}}. Who's asking?",
                    "alternate_greetings": ["An alternate opener.", "A third opener."],
                },
            )
            await pilot.pause()
            text = pane.transcript_text()
            assert "hi" in text and "hello" in text  # conversation survived
            assert screen.preview._current_greeting_index == 1

    async def test_greeting_selector_hidden_without_alternates(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A character without ``alternate_greetings`` keeps the row hidden."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            screen.query_one("#personas-preview-pane", PersonasPreviewPane)
            assert screen.query_one("#personas-preview-greeting-row").display is False

    async def _readout_text(self, screen):
        from textual.widgets import Static as _Static

        return str(screen.query_one("#personas-preview-provider", _Static).renderable)

    async def test_provider_readout_shows_character_and_fallback(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The readout names the character provider/model and fallback target
        (task-426)."""
        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "llama_cpp", "model": "local.gguf"},
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            text = await self._readout_text(screen)
            assert "Anthropic" in text
            assert "claude-3-haiku" in text
            # The Console-default fallback target is named too.
            assert "llama.cpp" in text

    async def test_provider_readout_no_fallback_note_when_same_provider(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """No fallback note when chat_defaults matches character_defaults."""
        mock_app_instance.app_config = {
            "character_defaults": {"provider": "openai", "model": "gpt-4o"},
            "chat_defaults": {"provider": "openai", "model": "gpt-4o"},
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            text = await self._readout_text(screen)
            assert "OpenAI" in text
            assert "gpt-4o" in text
            assert "Console default" not in text

    async def test_provider_readout_falls_to_chat_when_no_character_provider(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """With no character provider, the chat default is what answers."""
        mock_app_instance.app_config = {
            "character_defaults": {},
            "chat_defaults": {"provider": "llama_cpp", "model": "local.gguf"},
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            text = await self._readout_text(screen)
            assert "llama.cpp" in text
            assert "local.gguf" in text
            assert "Console default" in text

    async def test_provider_readout_uses_effective_provider_model(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The readout and send share the model inherited from provider config."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )

        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic"},
            "api_settings": {
                "anthropic": {"model": "claude-3-5-haiku-latest"},
            },
        }
        fake = ReadinessMapPreviewGateway(ready_providers={"anthropic"})
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            text = await self._readout_text(screen)
            assert text == "Provider: Anthropic / claude-3-5-haiku-latest"

            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert fake.selections[0].provider == "anthropic"
            assert fake.selections[0].explicit_model is None
            assert fake.selections[0].configured_model == "claude-3-5-haiku-latest"

    async def test_provider_readout_normalizes_whitespace_defaults(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Readout and send normalize whitespace before Console resolution."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )

        mock_app_instance.app_config = {
            "character_defaults": {"provider": "   ", "model": " ignored "},
            "chat_defaults": {
                "provider": "  llama_cpp  ",
                "model": "  local.gguf  ",
            },
        }
        fake = ReadinessMapPreviewGateway(ready_providers={"llama_cpp"})
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            text = await self._readout_text(screen)
            assert text == "Provider: llama.cpp / local.gguf (Console default)"
            assert screen.preview._readout_nav_provider == "llama_cpp"

            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert [selection.provider for selection in fake.selections] == [
                "",
                "llama_cpp",
            ]
            fallback = fake.selections[-1]
            assert fallback.explicit_model == "local.gguf"
            assert fake.requests, "Normalized fallback selection should answer"

    async def test_configure_button_navigates_to_settings_providers(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The Configure affordance deep-links to Settings > Providers & Models
        with the character provider preselected (task-426)."""
        from textual.widgets import Button

        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
        from tldw_chatbook.UI.Screens.settings_config_models import (
            SettingsCategoryId,
        )

        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "llama_cpp", "model": "local.gguf"},
        }
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            posted: list[NavigateToScreen] = []
            original = screen.post_message

            def _record(message):
                if isinstance(message, NavigateToScreen):
                    posted.append(message)
                return original(message)

            screen.post_message = _record
            screen.query_one("#personas-preview-configure", Button).press()
            await pilot.pause()
            assert posted, "Configure should post a navigation message"
            nav = posted[-1]
            assert nav.screen_name == "settings"
            category = nav.screen_context.get("category")
            category_value = getattr(category, "value", category)
            assert category_value == SettingsCategoryId.PROVIDERS_MODELS.value
            assert nav.screen_context.get("provider") == "anthropic"

    async def test_post_send_readout_reflects_resolved_fallback_provider(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """After a fallback reply, the readout reflects the provider that
        actually answered (task-425/426)."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )

        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "llama_cpp", "model": "local.gguf"},
        }
        fake = ReadinessMapPreviewGateway(ready_providers={"llama_cpp"})
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            text = await self._readout_text(screen)
            assert "llama.cpp" in text
            assert "Console default" in text
            # Configure still targets the character-configured provider (the
            # one to make ready), not the fallback that happened to answer.
            assert screen.preview._readout_nav_provider == "anthropic"

    async def test_blocked_provider_shows_readable_status(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """An unconfigured provider yields readable copy, never a traceback."""
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )

        # No character_defaults in config -> empty provider -> blocked.
        mock_app_instance.app_config = {}
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            status = str(
                screen.query_one("#personas-preview-status", _Static).renderable
            )
            assert status.strip()
            assert "Traceback" not in status

    async def test_reply_flow_appends_reply_and_history(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway()
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi there"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert "Detective Sam: Hello, world." in pane.transcript_text()
            assert screen.preview.history == [
                {"role": "user", "content": "Hi there"},
                {"role": "assistant", "content": "Hello, world."},
            ]
            from textual.widgets import Static as _Static

            assert (
                str(screen.query_one("#personas-preview-status", _Static).renderable)
                == "Ready"
            )
            # The provider saw the system prompt followed by the history.
            assert fake.requests and fake.requests[0][0]["role"] == "system"
            assert fake.requests[0][1] == {"role": "user", "content": "Hi there"}

    async def test_reply_streams_progressively_into_one_line(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Chunks render as they arrive, updating ONE growing character line."""
        import asyncio

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        mid_gate = asyncio.Event()
        fake = _FakePreviewGateway(mid_gate=mid_gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            pane = screen.query_one(PersonasPreviewPane)
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            # The first chunk must be visible WHILE the stream is held open.
            for _ in range(50):
                if "Detective Sam: Hello, " in pane.transcript_text():
                    break
                await pilot.pause()
            assert "Detective Sam: Hello, " in pane.transcript_text()
            # History gets the consolidated entry only at the end.
            assert not any(
                entry["role"] == "assistant" for entry in screen.preview.history
            )
            mid_gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            lines = pane.transcript_text().splitlines()
            assert lines.count("Detective Sam: Hello, world.") == 1
            assert "Detective Sam: Hello, " not in [
                line for line in lines if line != "Detective Sam: Hello, world."
            ]
            assert screen.preview.history[-1] == {
                "role": "assistant",
                "content": "Hello, world.",
            }

    async def test_reset_mid_stream_removes_partial_line(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Reset after the first chunk landed must drop the partial line."""
        import asyncio

        from textual.widgets import Button as _Button

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        mid_gate = asyncio.Event()
        fake = _FakePreviewGateway(mid_gate=mid_gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            pane = screen.query_one(PersonasPreviewPane)
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            for _ in range(50):
                if "Detective Sam: Hello, " in pane.transcript_text():
                    break
                await pilot.pause()
            assert "Detective Sam: Hello, " in pane.transcript_text()
            screen.query_one("#personas-preview-reset", _Button).press()
            await pilot.pause()
            mid_gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "Hello" not in pane.transcript_text()
            assert pane.transcript_text() == (
                "Detective Sam: The name's Detective Sam. Who's asking?"
            )
            assert not any(
                entry["role"] == "assistant" for entry in screen.preview.history
            )

    async def test_selection_change_mid_stream_removes_partial_line(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A selection move after the first chunk must drop the partial line."""
        import asyncio

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        mid_gate = asyncio.Event()
        fake = _FakePreviewGateway(mid_gate=mid_gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            pane = screen.query_one(PersonasPreviewPane)
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            for _ in range(50):
                if "Detective Sam: Hello, " in pane.transcript_text():
                    break
                await pilot.pause()
            assert "Detective Sam: Hello, " in pane.transcript_text()
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            mid_gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "Hello" not in pane.transcript_text()
            assert not any(
                entry["role"] == "assistant" for entry in screen.preview.history
            )

    async def test_error_mid_stream_removes_partial_line(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A provider error after the first chunk must not leave a dangling
        partial line; status shows the recovery copy and the orphaned user
        history entry is popped."""
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway(
            error=RuntimeError("provider exploded"), error_after_first=True
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert "Hello" not in pane.transcript_text()
            assert screen.preview.history == []
            status = str(
                screen.query_one("#personas-preview-status", _Static).renderable
            )
            assert "Provider error" in status

    async def test_provider_failure_logs_native_exception_with_preview_context(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Provider failures include Loguru-native exceptions and safe context."""
        from loguru import logger as loguru_logger

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )

        records: list[dict[str, Any]] = []
        sink_id = loguru_logger.add(
            lambda message: records.append(message.record), level="ERROR"
        )
        mock_app_instance.app_config = {
            "character_defaults": {
                "provider": "openai",
                "model": "gpt-4o-mini",
            }
        }
        fake = _FakePreviewGateway(error=RuntimeError("provider exploded"))
        app = PersonasTestApp(mock_app_instance)
        try:
            async with app.run_test(size=(160, 50)) as pilot:
                screen = await self._select_first_character(pilot)
                screen.preview.ensure_gateway = lambda: fake
                screen.post_message(PreviewReplyRequested("Hi"))
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
        finally:
            loguru_logger.remove(sink_id)

        stream_failure = next(
            record
            for record in records
            if record["message"]
            == "Preview provider call failed; retrying without streaming."
        )
        retry_failure = next(
            record
            for record in records
            if record["message"] == "Preview non-streaming retry failed."
        )
        assert stream_failure["exception"] is not None
        assert retry_failure["exception"] is not None
        extra = stream_failure["extra"]
        assert extra["attempt"] == "streaming"
        assert extra["streaming"] is True
        assert retry_failure["extra"]["attempt"] == "non_streaming"
        assert retry_failure["extra"]["streaming"] is False
        assert extra["provider"] == "openai"
        assert extra["model"] == "gpt-4o-mini"
        assert extra["selection_kind"] == "character"
        assert extra["selection_id"] == "1"
        assert extra["resolved_provider"] == "openai"
        assert extra["resolved_model"] == "test-model"
        assert isinstance(extra["generation"], int)

    async def test_resolution_failure_logs_safe_preview_context(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Resolution exceptions retain the standard structured preview context."""
        from loguru import logger as loguru_logger

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )

        records: list[dict[str, Any]] = []
        sink_id = loguru_logger.add(
            lambda message: records.append(message.record), level="ERROR"
        )
        mock_app_instance.app_config = {
            "character_defaults": {
                "provider": "anthropic",
                "model": "claude-3-haiku",
            },
            "chat_defaults": {
                "provider": "llama_cpp",
                "model": "local.gguf",
                "streaming": False,
            },
        }
        fake = ReadinessMapPreviewGateway(
            ready_providers=set(),
            resolve_error_providers={"llama_cpp"},
        )
        app = PersonasTestApp(mock_app_instance)
        try:
            async with app.run_test(size=(160, 50)) as pilot:
                screen = await self._select_first_character(pilot)
                screen.preview.ensure_gateway = lambda: fake
                screen.post_message(PreviewReplyRequested("Hi"))
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
        finally:
            loguru_logger.remove(sink_id)

        failure = next(
            record
            for record in records
            if record["message"] == "Preview provider resolution failed."
        )
        assert failure["exception"] is not None
        extra = failure["extra"]
        assert extra["operation"] == "personas_preview_reply"
        assert extra["provider"] == "llama_cpp"
        assert extra["model"] == "local.gguf"
        assert extra["selection_kind"] == "character"
        assert extra["selection_id"] == "1"
        assert extra["streaming"] is False
        assert extra["attempt"] == "resolve"
        assert isinstance(extra["generation"], int)

    async def test_unready_character_provider_falls_back_to_chat_defaults(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """The preview falls back to a ready chat_defaults provider (task-425).

        First-run configs carry the shipped ``[character_defaults]`` template
        (Anthropic) verbatim, so the fallback keys on readiness, not on the
        section's presence.
        """
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {
                "provider": "llama_cpp",
                "model": "local.gguf",
                "streaming": False,
                "temperature": 0.42,
                "top_p": 0.77,
                "max_tokens": 321,
            },
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:8181/v1/chat/completions"}
            },
        }
        fake = ReadinessMapPreviewGateway(ready_providers={"llama_cpp"})
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert "Detective Sam: Hello, world." in pane.transcript_text()
            assert [s.provider for s in fake.selections] == ["anthropic", "llama_cpp"]
            fallback = fake.selections[-1]
            assert fallback.explicit_model == "local.gguf"
            assert fallback.base_url == "http://127.0.0.1:8181"
            assert fallback.streaming is False
            assert fallback.temperature == 0.42
            assert fallback.top_p == 0.77
            assert fallback.max_tokens == 321
            status = str(
                screen.query_one("#personas-preview-status", _Static).renderable
            )
            assert "via Console default" in status
            assert "llama_cpp" in status

    async def test_ready_character_provider_wins_over_chat_defaults(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A usable character_defaults provider is used without fallback."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        mock_app_instance.app_config = {
            "character_defaults": {
                "provider": "anthropic",
                "model": "claude-3-haiku",
                "streaming": False,
                "temperature": 0.81,
                "top_p": 0.88,
                "max_tokens": 2048,
            },
            "chat_defaults": {
                "provider": "llama_cpp",
                "model": "local.gguf",
                "streaming": True,
                "temperature": 0.2,
            },
        }
        fake = ReadinessMapPreviewGateway(ready_providers={"anthropic", "llama_cpp"})
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert "Detective Sam: Hello, world." in pane.transcript_text()
            assert [s.provider for s in fake.selections] == ["anthropic"]
            character_selection = fake.selections[0]
            assert character_selection.streaming is False
            assert character_selection.temperature == 0.81
            assert character_selection.top_p == 0.88
            assert character_selection.max_tokens == 2048

    async def test_both_providers_unready_names_settings_remedy(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """When neither provider is ready the status points at Settings."""
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )

        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "llama_cpp", "model": "local.gguf"},
        }
        fake = ReadinessMapPreviewGateway(ready_providers=set())
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            status = str(
                screen.query_one("#personas-preview-status", _Static).renderable
            )
            assert "Settings" in status
            assert "Traceback" not in status
            # Both providers were attempted before giving up.
            assert [s.provider for s in fake.selections] == ["anthropic", "llama_cpp"]
            # The surfaced blocker is the chat_defaults provider the user
            # actually configured, not the shipped character default.
            assert "llama_cpp" in status
            assert "anthropic" not in status

    async def test_fallback_provider_survives_non_streaming_retry(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A streaming failure re-resolves the fallback provider, not the
        character default (task-425)."""
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        mock_app_instance.app_config = {
            "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
            "chat_defaults": {"provider": "llama_cpp", "model": "local.gguf"},
        }
        # First stream_chat raises -> non-streaming retry re-resolves.
        fake = ReadinessMapPreviewGateway(
            ready_providers={"llama_cpp"}, stream_failures=1
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert "Detective Sam: Hello, world." in pane.transcript_text()
            # anthropic (unready) -> llama_cpp (ready, streaming) ->
            # llama_cpp (non-streaming retry re-resolve).
            assert [s.provider for s in fake.selections] == [
                "anthropic",
                "llama_cpp",
                "llama_cpp",
            ]
            assert fake.selections[-1].streaming is False

    async def test_draft_aware_system_prompt_uses_editor_data(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from textual.widgets import TextArea as _TextArea

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            EditCharacterRequested,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.post_message(EditCharacterRequested("1"))
            await pilot.pause()
            assert screen._edit_mode == "edit"
            editor = screen.query_one(PersonasCharacterEditorWidget)
            editor.query_one(
                "#personas-char-editor-description", _TextArea
            ).text = "Draft noir vibes, unsaved."
            await pilot.pause()
            prompt = screen.preview.system_prompt()
            assert "Draft noir vibes, unsaved." in prompt

    async def test_open_in_console_stages_preview_transcript(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from textual.widgets import Button as _Button

        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        app = PersonasTestApp(mock_app_instance)
        app.open_chat_with_handoff = Mock()
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            pane = screen.query_one(PersonasPreviewPane)
            pane.append_user("Hi")
            pane.append_reply("Hello.")
            await pilot.pause()
            screen.query_one("#personas-preview-open-console", _Button).press()
            await pilot.pause()
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "personas"
        assert payload.item_type == "preview-conversation"
        assert payload.title == "Personas preview conversation"
        assert "User: Hi" in payload.body
        assert "Detective Sam: Hello." in payload.body
        assert payload.suggested_prompt == "Continue this conversation in character."

    async def test_stale_reply_is_dropped_after_selection_change(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        import asyncio

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        gate = asyncio.Event()
        fake = _FakePreviewGateway(gate=gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            # Let the worker reach the gated stream deterministically.
            for _ in range(50):
                if fake.requests:
                    break
                await pilot.pause()
            assert fake.requests
            # Selection changes while the stream is in flight. The gated
            # preview worker is still running, so release the gate BEFORE
            # waiting for workers (waiting first deadlocks the test).
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert "Detective Sam: Hello, world." not in pane.transcript_text()
            assert not any(
                entry["role"] == "assistant" for entry in screen.preview.history
            )
            from textual.widgets import Static as _Static

            assert (
                str(screen.query_one("#personas-preview-status", _Static).renderable)
                != "Ready"
            )

    async def test_reset_and_mode_switch_clear_history(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewResetRequested,
        )

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.history.append({"role": "user", "content": "Hi"})
            screen.post_message(PreviewResetRequested())
            await pilot.pause()
            assert screen.preview.history == []
            screen.preview.history.append({"role": "user", "content": "Hi again"})
            await screen._apply_mode("prompts")
            await pilot.pause()
            assert screen.preview.history == []

    async def test_reset_mid_stream_drops_late_reply(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Reset while a reply streams must invalidate the in-flight worker.

        The selection key alone cannot catch this: Reset keeps the same
        (kind, id), so without a generation bump (and group cancel) the late
        reply would land in the freshly cleared history/transcript.
        """
        import asyncio

        from textual.widgets import Button as _Button

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        gate = asyncio.Event()
        fake = _FakePreviewGateway(gate=gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            # Let the worker reach the gated stream deterministically.
            for _ in range(50):
                if fake.requests:
                    break
                await pilot.pause()
            assert fake.requests
            # Reset while the stream is held at the gate.
            screen.query_one("#personas-preview-reset", _Button).press()
            await pilot.pause()
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert "Detective Sam: Hello, world." not in pane.transcript_text()
            assert not any(
                entry["role"] == "assistant" for entry in screen.preview.history
            )

    async def test_error_pops_orphaned_user_history_entry(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A provider error must not leave an unanswered user turn in history.

        The transcript keeps the user's line (they did say it), but the
        history entry is popped so a retry does not send [user, user].
        """
        from textual.widgets import Button as _Button, Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway(error=RuntimeError("provider exploded"))
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            pane = screen.query_one(PersonasPreviewPane)
            pane.expand()
            await pilot.pause()
            screen.query_one("#personas-preview-input", Input).value = "Hi"
            screen.query_one("#personas-preview-test-reply", _Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # History: no trailing unanswered user entry.
            assert not any(entry["role"] == "user" for entry in screen.preview.history)
            # Transcript: the user line stays visible.
            assert "User: Hi" in pane.transcript_text()
            status = str(
                screen.query_one("#personas-preview-status", _Static).renderable
            )
            assert status.strip()
            assert "Traceback" not in status

    async def test_stream_failure_falls_back_to_non_streaming_once(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """A failed stream retries exactly once with streaming disabled."""
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway(stream_failures=1)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # Both attempts resolved; the retry disabled streaming.
            assert [s.streaming for s in fake.selections] == [True, False]
            assert len(fake.requests) == 2
            pane = screen.query_one(PersonasPreviewPane)
            assert "Detective Sam: Hello, world." in pane.transcript_text()
            assert screen.preview.history[-1] == {
                "role": "assistant",
                "content": "Hello, world.",
            }
            assert (
                str(screen.query_one("#personas-preview-status", _Static).renderable)
                == "Ready"
            )

    async def test_both_attempts_failing_keeps_error_semantics(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """When the non-streaming retry also fails the existing error
        semantics apply: orphan user turn popped, readable error status."""
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway(error=RuntimeError("provider exploded"))
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # Streaming attempt + one non-streaming retry, no more.
            assert [s.streaming for s in fake.selections] == [True, False]
            assert len(fake.requests) == 2
            assert screen.preview.history == []
            pane = screen.query_one(PersonasPreviewPane)
            assert "Hello" not in pane.transcript_text()
            status = str(
                screen.query_one("#personas-preview-status", _Static).renderable
            )
            assert "Provider error" in status

    async def test_empty_reply_sets_status_without_bare_line(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """An empty stream must not append a bare transcript line or history entry."""
        from textual.widgets import Static as _Static

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )
        from tldw_chatbook.Widgets.Persona_Widgets.personas_preview_pane import (
            PersonasPreviewPane,
        )

        fake = _FakePreviewGateway(chunks=())
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            pane = screen.query_one(PersonasPreviewPane)
            assert all(
                line.strip() != "character:"
                for line in pane.transcript_text().splitlines()
            )
            assert not any(
                entry["role"] == "assistant" for entry in screen.preview.history
            )
            assert (
                str(screen.query_one("#personas-preview-status", _Static).renderable)
                == "No reply received"
            )

    async def test_unmount_closes_gateway(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Leaving the screen releases the preview gateway's HTTP client."""
        fake = _FakePreviewGateway()
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.gateway = fake
        assert fake.closed is True

    async def test_double_fire_coalesces_user_turns(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """An exclusive-cancelled predecessor leaves back-to-back user turns;
        the replacement worker must coalesce them so strict providers never
        see [user, user]."""
        import asyncio

        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            PreviewReplyRequested,
        )

        gate = asyncio.Event()
        fake = _FakePreviewGateway(gate=gate)
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            screen.preview.ensure_gateway = lambda: fake
            screen.post_message(PreviewReplyRequested("Hi"))
            await pilot.pause()
            for _ in range(50):
                if fake.requests:
                    break
                await pilot.pause()
            assert len(fake.requests) == 1
            # Second request while the first is gated: exclusive=True cancels
            # the first worker, and the second sees history [user, user].
            screen.post_message(PreviewReplyRequested("Again"))
            await pilot.pause()
            for _ in range(50):
                if len(fake.requests) >= 2:
                    break
                await pilot.pause()
            assert len(fake.requests) == 2
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            user_messages = [m for m in fake.requests[1] if m["role"] == "user"]
            assert user_messages == [{"role": "user", "content": "Hi\nAgain"}]


class TestDelete:
    """Confirmed delete for characters and persona profiles (Task 14).

    The confirmation dialog itself is bypassed by replacing the screen's
    ``_confirm_delete`` helper, the same way the import/export tests bypass
    the file dialogs by calling the path-based methods directly.
    """

    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        _install_conversation_db(
            monkeypatch, [_conversation_record(1, title="First case")]
        )

    @staticmethod
    def _capture_notifications(app) -> list[tuple[str, str]]:
        captured: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: captured.append(
            (str(message), severity)
        )
        return captured

    @staticmethod
    def _bypass_confirm(screen, result: bool) -> None:
        async def _confirm(name: str) -> bool:
            return result

        screen._confirm_delete = _confirm

    async def _select_first_character(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def _select_profile(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-mode-personas")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#personas-library-row-persona-p-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    @staticmethod
    async def _press_delete(pilot, screen):
        screen.query_one("#personas-delete", Button).press()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_delete_character_soft_deletes_and_clears(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        deleted: list[tuple] = []

        def fake_delete(character_id, expected_version):
            deleted.append((character_id, expected_version))
            return True

        monkeypatch.setattr(character_handler_module, "delete_character", fake_delete)

        def fetch_all_post_delete():
            characters = [dict(c) for c in CHARACTERS]
            if deleted:
                characters = [c for c in characters if str(c["id"]) != "1"]
            return characters

        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", fetch_all_post_delete
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            # Sanity: the conversations panel has rows before the delete.
            assert screen.query_one("#personas-conversations-list").children
            self._bypass_confirm(screen, True)
            await self._press_delete(pilot, screen)
            # delete_character received the id and the FULL record's version.
            assert deleted == [("1", 1)]
            # Selection cleared, view mode, center pane empty.
            assert screen.state.selected_entity_id is None
            assert screen.state.selected_entity_kind is None
            assert screen._edit_mode == "view"
            assert screen.query_one("#ccp-character-card-view").display is False
            assert "Selected: none" in str(
                screen.query_one("#personas-selected-name", Static).renderable
            )
            # Conversations panel emptied.
            assert not screen.query_one("#personas-conversations-list").children
            # Library refreshed without the deleted record.
            rows = screen.query(".personas-library-row")
            assert [_row_text(r) for r in rows] == ["Lab Assistant"]
        assert ("Deleted.", "information") in notifications

    async def test_delete_conflict_shows_recovery_copy(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        monkeypatch.setattr(
            character_handler_module,
            "delete_character",
            lambda character_id, expected_version: False,
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            self._bypass_confirm(screen, True)
            await self._press_delete(pilot, screen)
            assert any(
                "changed since it was loaded" in message and severity == "error"
                for message, severity in notifications
            )
            # The selection is retained so the user can reselect/retry.
            assert screen.state.selected_entity_id == "1"
            assert not any(message == "Deleted." for message, _ in notifications)

    async def test_delete_cancelled_is_noop(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        deleted: list[tuple] = []
        monkeypatch.setattr(
            character_handler_module,
            "delete_character",
            lambda character_id, expected_version: (
                deleted.append((character_id, expected_version)) or True
            ),
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            self._bypass_confirm(screen, False)
            await self._press_delete(pilot, screen)
            assert deleted == []
            assert screen.state.selected_entity_id == "1"
            assert screen.query_one("#ccp-character-card-view").display is True
            assert not any(message == "Deleted." for message, _ in notifications)

    async def test_delete_profile_calls_scope_service(
        self, mock_app_instance, stub_characters, stub_conversations, stub_scope_service
    ):
        # The full record (with version) comes from get_persona_profile.
        stub_scope_service.get_persona_profile = AsyncMock(
            return_value={**PROFILE, "version": 3}
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_profile(pilot)
            self._bypass_confirm(screen, True)
            await self._press_delete(pilot, screen)
            stub_scope_service.delete_persona_profile.assert_awaited_once()
            await_args = stub_scope_service.delete_persona_profile.await_args
            assert await_args.args[0] == "p-1"
            assert await_args.kwargs == {"expected_version": 3, "mode": "local"}
            assert screen.state.selected_entity_id is None
            assert screen.query_one("#ccp-persona-card-view").display is False
        assert ("Deleted.", "information") in notifications

    async def test_delete_blocked_when_full_record_missing(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        deleted: list[tuple] = []
        monkeypatch.setattr(
            character_handler_module,
            "delete_character",
            lambda character_id, expected_version: (
                deleted.append((character_id, expected_version)) or True
            ),
        )
        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            # Simulate a stale handler: the loaded record is another character,
            # so the optimistic-lock version cannot be sourced.
            screen.character_handler.current_character_id = "999"
            self._bypass_confirm(screen, True)
            await self._press_delete(pilot, screen)
            assert deleted == []
            assert screen.state.selected_entity_id == "1"
            assert any(
                "not loaded" in message and severity == "warning"
                for message, severity in notifications
            )


@pytest.fixture
def legacy_human_config(tmp_path, monkeypatch):
    """Seed and snapshot a retired human-pointer value without mutating it."""
    legacy_key = "active_user_profile"
    config_path = tmp_path / "legacy-persona-config.toml"
    serialized = (
        "[character_defaults]\n"
        f'{legacy_key} = "Archivist"\n'
        'provider = "anthropic"\n'
        'model = "claude-3-haiku"\n'
        "\n[chat_defaults]\n"
        'provider = "anthropic"\n'
        'model = "claude-3-haiku"\n'
        "\n[api_settings.anthropic]\n"
        'api_key = "unit-test-placeholder-key"\n'
    ).encode()
    config_path.write_bytes(serialized)

    from tldw_chatbook import config as config_module

    previous_config_path = config_module.get_cli_config_path()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    mapping = config_module.load_cli_config_and_ensure_existence(force_reload=True)
    assert mapping["character_defaults"][legacy_key] == "Archivist"
    mapping_snapshot = deepcopy(mapping)
    bytes_snapshot = config_path.read_bytes()
    mutation_calls: list[str] = []

    def _forbid_mutation(label):
        def _callback(*args, **kwargs):
            mutation_calls.append(label)
            raise AssertionError(f"legacy config mutation callback ran: {label}")

        return _callback

    for label, callback_name in (
        ("save", "save_setting_to_cli_config"),
        ("save-or-clear", "save_settings_to_cli_config"),
        ("clear", "delete_settings_from_cli_config"),
        ("repair", "replace_cli_config"),
    ):
        monkeypatch.setattr(
            config_module,
            callback_name,
            _forbid_mutation(label),
        )
    for callback_name in (
        "save_setting_to_cli_config",
        "save_settings_to_cli_config",
        "delete_settings_from_cli_config",
    ):
        monkeypatch.setattr(
            chat_screen_module,
            callback_name,
            _forbid_mutation(f"chat-screen.{callback_name}"),
        )

    def _assert_unchanged() -> None:
        assert config_module.get_cli_config_path() == config_path
        assert mapping == mapping_snapshot
        assert config_path.read_bytes() == bytes_snapshot
        assert mutation_calls == []

    try:
        yield SimpleNamespace(
            mapping=mapping,
            path=config_path,
            assert_unchanged=_assert_unchanged,
        )
    finally:
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(previous_config_path))
        config_module.load_cli_config_and_ensure_existence(force_reload=True)


class TestBulkLibraryActions:
    """F-040: the library pane's mark set drives bulk delete/export."""

    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        _install_conversation_db(
            monkeypatch, [_conversation_record(1, title="First case")]
        )

    @staticmethod
    def _capture_notifications(app) -> list[tuple[str, str]]:
        captured: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: captured.append(
            (str(message), severity)
        )
        return captured

    async def _mount_with_marks(self, pilot, row_indexes=(0, 1)):
        """Mount and mark rows through the pane's m key, like a user."""
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        list_view = screen.query_one("#personas-library-rows", ListView)
        list_view.focus()
        await pilot.pause()
        for index in row_indexes:
            list_view.index = index
            await pilot.press("m")
            await pilot.pause()
        return screen

    async def test_marks_retarget_delete_and_export_affordances(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._mount_with_marks(pilot, (0, 1))
            inspector = screen.query_one(PersonasInspectorPane)
            delete = inspector.query_one("#personas-delete", Button)
            export_json = inspector.query_one("#personas-export-json", Button)
            export_png = inspector.query_one("#personas-export-png", Button)
            assert delete.disabled is False
            assert delete.tooltip == "Delete the 2 marked items."
            assert export_json.disabled is False
            assert export_json.tooltip == "Export the 2 marked items as JSON."
            assert export_png.disabled is True
            assert export_png.tooltip == "Bulk export is JSON only."
            # Clearing the marks restores the selection-owned gates.
            screen.query_one("#personas-library-pane").clear_marks()
            await pilot.pause()
            assert export_png.disabled is False
            assert delete.tooltip is None

    async def test_bulk_delete_marked_characters(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        deleted: list[tuple[str, int]] = []

        def _delete(character_id, expected_version):
            deleted.append((str(character_id), expected_version))
            return True

        monkeypatch.setattr(character_handler_module, "delete_character", _delete)
        # The live read shrinks as deletes land, so the refresh renders empty.
        monkeypatch.setattr(
            character_handler_module,
            "fetch_all_characters",
            lambda: [
                dict(c)
                for c in CHARACTERS
                if str(c["id"]) not in {did for did, _ in deleted}
            ],
        )
        confirm_calls: list[str] = []

        async def _confirm(name: str) -> bool:
            confirm_calls.append(name)
            return True

        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._mount_with_marks(pilot, (0, 1))
            screen._confirm_delete = _confirm
            screen.query_one("#personas-delete", Button).press()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # One confirm for the whole batch; each item deleted once.
            assert confirm_calls == ["2 characters"]
            assert sorted(deleted) == [("1", 1), ("2", 1)]
            assert not list(screen.query(".personas-library-row"))
            assert screen.state.selected_entity_id is None
            assert screen._marked_rows == ()
        assert ("Deleted 2 characters.", "information") in notifications

    async def test_bulk_delete_keeps_unmarked_selection(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        deleted: list[str] = []

        def _delete(character_id, expected_version):
            deleted.append(str(character_id))
            return True

        monkeypatch.setattr(character_handler_module, "delete_character", _delete)
        monkeypatch.setattr(
            character_handler_module,
            "fetch_all_characters",
            lambda: [dict(c) for c in CHARACTERS if str(c["id"]) not in set(deleted)],
        )

        async def _confirm(name: str) -> bool:
            return True

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            # F-031 auto-selected row 1; only row 2 is marked for deletion.
            screen = await self._mount_with_marks(pilot, (1,))
            screen._confirm_delete = _confirm
            screen.query_one("#personas-delete", Button).press()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert deleted == ["2"]
            assert screen.state.selected_entity_id == "1"

    async def test_bulk_export_marked_characters_writes_one_file_per_card(
        self, mock_app_instance, stub_characters, stub_conversations, tmp_path
    ):
        exports: list[tuple[int, str]] = []

        def _fake_export(character_id, target_path, portable_profile):
            exports.append((character_id, target_path))

        app = PersonasTestApp(mock_app_instance)
        notifications = self._capture_notifications(app)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._mount_with_marks(pilot, (0, 1))
            screen._export_character_json_sync = _fake_export
            pilot.app.push_screen_wait = AsyncMock(return_value=str(tmp_path))
            screen.query_one("#personas-export-json", Button).press()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert [character_id for character_id, _ in exports] == [1, 2]
            paths = sorted(target for _, target in exports)
            assert paths[0].endswith("Detective Sam.json")
            assert paths[1].endswith("Lab Assistant.json")
        assert any(
            message.startswith("Exported 2 items") and severity == "information"
            for message, severity in notifications
        )

    async def test_bulk_export_marked_pushes_enhanced_directory_picker(
        self, mock_app_instance, stub_characters, stub_conversations, tmp_path
    ):
        """The marked-rows JSON export must use the enhanced picker family
        (TASK-16477): same chrome as every other Roleplay dialog, and it
        remembers its start directory per context."""
        from tldw_chatbook.Widgets.enhanced_file_picker import (
            EnhancedSelectDirectory,
        )

        pushed: list[object] = []

        async def _fake_push_screen_wait(picker):
            pushed.append(picker)
            return tmp_path

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._mount_with_marks(pilot, (0, 1))
            pilot.app.push_screen_wait = AsyncMock(side_effect=_fake_push_screen_wait)
            screen.query_one("#personas-export-json", Button).press()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

        assert len(pushed) == 1
        picker = pushed[0]
        assert isinstance(picker, EnhancedSelectDirectory)
        assert picker._title == "Export 2 items as JSON"
        assert picker.context == "character_export_dir"

    async def test_footer_discloses_sort_key_in_sortable_modes(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert "s sort" in screen._shortcut_context().render().lower()
            await screen._apply_mode("dictionaries")
            await pilot.pause()
            assert "s sort" not in screen._shortcut_context().render().lower()


class TestPersonaHumanIdentityRemoval:
    """Personas never identify the human user (F-034: "who you play" copy
    teaches the play-identity convention without reviving that framing)."""

    async def _select_profile(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-mode-personas")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#personas-library-row-persona-p-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def test_preview_ignores_seeded_legacy_human_value(
        self,
        mock_app_instance,
        stub_characters,
        legacy_human_config,
    ):
        profile_lister = Mock()
        profile_lister.list_persona_profiles.return_value = [dict(PROFILE)]
        mock_app_instance.app_config = legacy_human_config.mapping
        mock_app_instance.local_character_persona_service = profile_lister
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            pane = screen.query_one(PersonasPreviewPane)
            greeting = screen.preview._load_greetings(
                {"first_message": "Hello {{user}}, I am {{char}}."},
                "Elara",
            )
            await pane.seed_greeting(greeting)
            pane.append_user("Hi")
            await pilot.pause()

            assert greeting == "Hello User, I am Elara."
            assert pane._user_label == "User"
            assert pane.transcript_text().endswith("User: Hi")
            profile_lister.list_persona_profiles.assert_not_called()

        legacy_human_config.assert_unchanged()

    async def test_workbench_exposes_no_human_identity_controls_or_marker(
        self,
        mock_app_instance,
        stub_characters,
        stub_scope_service,
        legacy_human_config,
    ):
        mock_app_instance.app_config = legacy_human_config.mapping
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await self._select_profile(pilot)
            row = screen.query_one("#personas-library-row-persona-p-1")
            exposed = {
                "active summary": bool(
                    list(screen.query("#personas-active-profile-summary"))
                ),
                "set/clear button": bool(list(screen.query("#personas-set-my-name"))),
                "active marker": "●" in _row_text(row),
                "pointer toggle handler": hasattr(
                    PersonasScreen, "_handle_set_my_name"
                ),
            }
            assert exposed == {
                "active summary": False,
                "set/clear button": False,
                "active marker": False,
                "pointer toggle handler": False,
            }

        legacy_human_config.assert_unchanged()

    @pytest.mark.parametrize("runtime_source", ("local", "server"))
    async def test_character_handoff_uses_user_without_profile_override(
        self,
        monkeypatch,
        legacy_human_config,
        runtime_source,
    ):
        local_profile_service = Mock()
        local_profile_service.list_persona_profiles.return_value = [dict(PROFILE)]
        server_profile_service = Mock()
        server_profile_service.list_persona_profiles = AsyncMock(
            return_value={"items": [dict(PROFILE)], "total": 1}
        )
        server_profile_service.get_character = AsyncMock(
            return_value={
                "id": 7,
                "name": "Elara",
                "first_message": "Hello {{user}}, I am {{char}}.",
                "system_prompt": "Stay curious.",
            }
        )
        db = SimpleNamespace(
            get_local_authority_id=Mock(return_value="local-authority"),
            get_character_card_by_id=Mock(
                side_effect=AssertionError(
                    "Chat now must use the source-aware scope service"
                )
            ),
        )
        server_target_id = "configured-target-7"
        server_authority_id = "server-user-v1:" + ("d" * 64)
        authority_resolver = AsyncMock(return_value=server_authority_id)
        authority_capture = object()

        async def resolve_authority_for_capture(
            *,
            expected_server_id,
            context_capture,
        ):
            assert context_capture is authority_capture
            return await authority_resolver(expected_server_id=expected_server_id)

        class _CapturingStore(ConsoleChatStore):
            """Real in-memory Console store that records what was appended.

            task-14920: this was a hand-rolled stub implementing only
            ``create_session``/``append_message``. Once the handoff started
            seeding the greeting through
            ``ConsoleChatStore.seed_character_roleplay`` (commit a6cc05d8b),
            the stub no longer had the method production calls, and the
            handoff's ``except Exception`` swallowed the ``AttributeError``
            -- so this test silently asserted "no greeting" instead of
            failing on a stale double. Subclassing the real store keeps the
            greeting expansion under production's control.
            """

            def __init__(self):
                super().__init__()
                self.session = None
                self.messages = []

            def create_session(self, **kwargs):
                self.session = super().create_session(**kwargs)
                return self.session

            def append_message(
                self, session_id, *, role, content, persist=False, **kwargs
            ):
                self.messages.append(
                    {
                        "session_id": session_id,
                        "role": role,
                        "content": content,
                        "persist": persist,
                    }
                )
                return super().append_message(
                    session_id,
                    role=role,
                    content=content,
                    persist=persist,
                    **kwargs,
                )

        runtime_app = SimpleNamespace(
            app_config=legacy_human_config.mapping,
            active_server_id=(server_target_id if runtime_source == "server" else None),
            chachanotes_db=db,
            character_persona_scope_service=server_profile_service,
            local_character_persona_service=local_profile_service,
            server_context_provider=SimpleNamespace(
                capture_character_authority_context=Mock(
                    return_value=authority_capture
                ),
                is_character_authority_context_current=Mock(
                    side_effect=lambda capture: capture is authority_capture
                ),
                resolve_character_authority_id=resolve_authority_for_capture,
            ),
        )
        screen = ChatScreen(runtime_app)
        store = _CapturingStore()
        baseline = ConsoleSessionSettings(
            provider="anthropic",
            model="claude-3-haiku",
        )

        monkeypatch.setattr(
            ChatScreen,
            "_ensure_console_chat_store",
            lambda self: store,
        )
        monkeypatch.setattr(
            ConsoleSessionController,
            "_default_console_session_settings",
            lambda self: baseline,
        )
        monkeypatch.setattr(
            ChatScreen,
            "_sync_native_console_chat_ui",
            AsyncMock(),
        )
        monkeypatch.setattr(
            ChatScreen,
            "_focus_console_composer_if_needed",
            lambda self, **kwargs: None,
        )
        payload = ChatHandoffPayload(
            source="personas",
            item_type="character-card",
            title="Elara",
            body="Character summary",
            runtime_backend=runtime_source,
            source_owner=runtime_source,
            source_selector_state=runtime_source,
            active_server_profile_id=(
                server_target_id if runtime_source == "server" else None
            ),
            metadata={
                "intent": "start_chat",
                "selected_kind": "character",
                "selected_record_id": "7",
                "selected_name": "Elara",
                "selected_target_id": f"{runtime_source}:character:7",
                "backend": runtime_source,
            },
        )

        assert await screen._session._start_character_console_session(payload) is True
        assert store.session is not None
        assert not hasattr(store.session.settings, "user_profile_label")
        assert [message["content"] for message in store.messages] == [
            "Hello User, I am Elara."
        ]
        local_profile_service.list_persona_profiles.assert_not_called()
        server_profile_service.list_persona_profiles.assert_not_awaited()
        server_profile_service.get_character.assert_awaited_once_with(
            7, mode=runtime_source
        )
        db.get_character_card_by_id.assert_not_called()
        if runtime_source == "local":
            db.get_local_authority_id.assert_called_once_with()
            authority_resolver.assert_not_awaited()
            assert store.session.assistant_authority_id == "local-authority"
        else:
            db.get_local_authority_id.assert_not_called()
            authority_resolver.assert_awaited_once_with(
                expected_server_id=server_target_id
            )
            assert store.session.assistant_authority_id == server_authority_id
        legacy_human_config.assert_unchanged()

    async def test_profile_rename_does_not_follow_legacy_pointer(
        self,
        mock_app_instance,
        stub_characters,
        stub_scope_service,
        legacy_human_config,
    ):
        mock_app_instance.app_config = legacy_human_config.mapping
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await self._select_profile(pilot)
            renamed = {**PROFILE, "name": "Chronicler"}
            stub_scope_service.list_persona_profiles = AsyncMock(
                return_value={"items": [renamed], "total": 1}
            )
            await screen._after_profile_save({"id": "p-1", "name": "Chronicler"})
            await pilot.pause()
            assert screen.state.selected_entity_name == "Chronicler"

        legacy_human_config.assert_unchanged()

    async def test_profile_delete_does_not_clear_legacy_pointer(
        self,
        mock_app_instance,
        stub_characters,
        stub_scope_service,
        legacy_human_config,
    ):
        mock_app_instance.app_config = legacy_human_config.mapping
        stub_scope_service.get_persona_profile = AsyncMock(
            return_value={**PROFILE, "version": 1}
        )
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await self._select_profile(pilot)
            await screen._delete_entity("persona", "p-1", 1)
            await pilot.pause()
            stub_scope_service.delete_persona_profile.assert_awaited_once()

        legacy_human_config.assert_unchanged()


class TestCharactersEmptyStateGuidance:
    """task-436: Characters mode shows onboarding guidance when nothing is
    selected, instead of a blank center pane that reads as broken.

    F-031 (task-2082) layers first-paint auto-select on top: a non-empty
    library mounts with its first row already selected (guidance hidden,
    card shown), so the guidance state is only reachable with an empty
    library, after a delete, or after a mode round-trip.
    """

    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        """Mirrors TestDelete.stub_conversations: stub the DB resolver and
        conversation listing so a real character selection/delete round-trip
        doesn't hit an unstubbed DB."""
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        _install_conversation_db(
            monkeypatch, [_conversation_record(1, title="First case")]
        )

    @staticmethod
    def _bypass_confirm(screen, result: bool) -> None:
        async def _confirm(name: str) -> bool:
            return result

        screen._confirm_delete = _confirm

    async def _select_first_character(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        return screen

    async def test_first_paint_auto_selects_first_library_row(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """F-031: a non-empty library mounts with its first row selected -
        card loaded, inspector awake - instead of a void center."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_id == "1"
            assert screen.state.selected_entity_kind == "character"
            assert screen.state.selected_entity_name == "Detective Sam"
            assert screen.query_one("#ccp-character-card-view").display is True
            assert (
                screen.query_one("#personas-characters-empty", Static).display is False
            )
            assert "Selected: Detective Sam" in str(
                screen.query_one("#personas-selected-name", Static).renderable
            )
            # Auto-select must not move focus (focus-steal guards, F-031).
            search = screen.query_one("#personas-library-search", Input)
            assert not search.has_focus

    async def test_first_paint_auto_select_keeps_unsaved_guards_quiet(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """F-031: the auto-selected row is a clean selection - no unsaved
        state, no guard dialog on a follow-up selection."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_id == "1"
            assert screen.state.has_unsaved_changes is False
            # A second selection runs the clean fast path (no confirm).
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_id == "2"

    async def test_guidance_shown_when_library_empty(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """With no rows there is nothing to auto-select: the no-selection
        guidance paints (the state first-time users with no card see until
        they New/Import one)."""
        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", lambda: []
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.active_mode == "characters"
            assert not screen.state.selected_entity_id
            guidance = screen.query_one("#personas-characters-empty", Static)
            assert guidance.display is True
            body = str(guidance.renderable)
            # F-035: the truly-empty copy names the creation actions (and no
            # longer claims there is a list to pick from).
            assert "No characters yet" in body
            assert "New" in body and "Import" in body
            assert "Pick one from the list" not in body
            assert screen.query_one("#ccp-character-card-view").display is False

    async def test_guidance_adapts_when_library_has_items(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """F-035: with characters in the library, a cleared selection asks
        for a pick - the New/Import onboarding copy would be wrong here."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # F-031 auto-selected row 1; a mode round-trip clears the
            # selection while the non-empty library stays.
            await screen._apply_mode("lore")
            await pilot.pause()
            await screen._apply_mode("characters")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert not screen.state.selected_entity_id
            assert screen._character_total > 0
            guidance = screen.query_one("#personas-characters-empty", Static)
            assert guidance.display is True
            body = str(guidance.renderable)
            assert body == "Pick a character from the list to see it here."
            assert "Import" not in body

    async def test_guidance_uses_left_aligned_empty_state_convention(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        """F-035: empty copy aligns like the app's other empty states
        (left/top, cf. .chat-empty-state), not centered in a void."""
        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", lambda: []
        )
        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            guidance = screen.query_one("#personas-characters-empty", Static)
            assert guidance.display is True
            assert str(guidance.styles.text_align) == "left"
            assert guidance.styles.content_align_horizontal == "left"
            assert guidance.styles.content_align_vertical == "top"

    async def test_guidance_hidden_after_selection(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # First paint already auto-selected row 1 (F-031), so the
            # guidance is hidden from the start...
            assert screen.state.selected_entity_id == "1"
            assert (
                screen.query_one("#personas-characters-empty", Static).display is False
            )
            # ...and stays hidden when the selection moves to another row.
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_id == "2"
            assert (
                screen.query_one("#personas-characters-empty", Static).display is False
            )
            assert screen.query_one("#ccp-character-card-view").display is True

    async def test_guidance_hidden_in_other_modes(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await screen._apply_mode("lore")
            await pilot.pause()
            assert (
                screen.query_one("#personas-characters-empty", Static).display is False
            )
            await screen._apply_mode("characters")
            await pilot.pause()
            assert (
                screen.query_one("#personas-characters-empty", Static).display is True
            )

    async def test_guidance_returns_after_delete(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        monkeypatch.setattr(
            character_handler_module,
            "delete_character",
            lambda character_id, expected_version: True,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._select_first_character(pilot)
            assert (
                screen.query_one("#personas-characters-empty", Static).display is False
            )
            self._bypass_confirm(screen, True)
            screen.query_one("#personas-delete", Button).press()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert not screen.state.selected_entity_id
            assert (
                screen.query_one("#personas-characters-empty", Static).display is True
            )


class TestKeyboardInteraction:
    """UX-E2: context-sensitive Escape, real Ctrl+S, mode keys, managed focus."""

    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        """Stub the DB resolver, conversation listing, and message retrieval."""
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        _install_conversation_db(
            monkeypatch, [_conversation_record(1, title="First case")]
        )
        monkeypatch.setattr(
            conversations_controller_module,
            "retrieve_conversation_messages_for_ui",
            lambda db, conversation_id, character_name, user_name, **kwargs: [
                ("Hello there", "Greetings, detective."),
            ],
        )

    @staticmethod
    def _bypass_confirm(screen, answer: bool) -> list[bool]:
        """Stub the unsaved-changes confirm; returns a call log."""
        calls: list[bool] = []

        async def fake_confirm() -> bool:
            calls.append(answer)
            return answer

        screen._confirm_discard_unsaved = fake_confirm
        return calls

    async def _open_create_editor(self, pilot):
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert screen._edit_mode == "create"
        return screen

    async def test_escape_cancels_editor_via_guard(
        self, mock_app_instance, stub_characters
    ):
        """Esc in the editor takes the SAME guarded cancel path as the button."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._open_create_editor(pilot)
            # Type into the focused Name input first: dirty tracking is
            # change-based, so a pristine editor would cancel dialog-free.
            await pilot.press("x")
            await pilot.pause()
            confirms = self._bypass_confirm(screen, True)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # The guard was consulted (real edit present), not bypassed.
            assert confirms == [True]
            assert screen._edit_mode == "view"
            assert screen.query_one("#ccp-character-editor-view").display is False

    async def test_escape_keeps_editor_when_guard_declined(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._open_create_editor(pilot)
            await pilot.press("x")  # real edit; the guard must fire
            await pilot.pause()
            confirms = self._bypass_confirm(screen, False)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert confirms == [False]
            assert screen._edit_mode == "create"
            assert screen.query_one("#ccp-character-editor-view").display is True

    async def test_escape_in_conversation_transcript_returns_to_card(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-row-character-1")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-conversation-row-conv-1")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            transcript = screen.query_one("#personas-conversation-transcript-view")
            assert transcript.display is True
            # Selection and asynchronous preview completion keep arrow-key
            # browsing anchored in the conversations list.
            focused = pilot.app.focused
            assert focused is not None and focused.id == "personas-conversations-list"
            await pilot.press("escape")
            await pilot.pause()
            assert transcript.display is False
            assert screen.query_one("#ccp-character-card-view").display is True
            # Back returns focus to the conversations list in the inspector.
            focused = pilot.app.focused
            assert focused is not None and focused.id == "personas-conversations-list"

    async def test_escape_blurs_search_input(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.press("ctrl+f")
            await pilot.pause()
            assert pilot.app.focused.id == "personas-library-search"
            await pilot.press("escape")
            await pilot.pause()
            focused = pilot.app.focused
            assert focused is not None and focused.id == "personas-library-rows"
            # Still in view mode; nothing else changed.
            assert screen._edit_mode == "view"

    async def test_ctrl_s_saves_from_editor(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        created = []
        monkeypatch.setattr(
            character_handler_module,
            "create_character",
            lambda data: created.append(data) or 99,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._open_create_editor(pilot)
            screen.query_one("#personas-char-editor-name", Input).value = "New Hero"
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen._edit_mode == "view"
        assert created and created[0]["name"] == "New Hero"

    async def test_save_persists_staged_avatar_bytes(
        self, mock_app_instance, stub_characters, monkeypatch, tmp_path
    ):
        avatar = tmp_path / "avatar.png"
        avatar.write_bytes(b"\x89PNG staged avatar")
        created: list[dict[str, Any]] = []
        monkeypatch.setattr(
            character_handler_module,
            "create_character",
            lambda data: created.append(dict(data)) or 99,
        )
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            screen.query_one("#personas-char-editor-name", Input).value = "Avatar Hero"
            await screen._stage_character_avatar_from_path(str(avatar))
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

        assert created
        assert created[0]["name"] == "Avatar Hero"
        assert created[0]["image"] == b"\x89PNG staged avatar"

    async def test_ctrl_s_noop_in_view_mode(
        self, mock_app_instance, stub_characters, monkeypatch
    ):
        created = []
        monkeypatch.setattr(
            character_handler_module,
            "create_character",
            lambda data: created.append(data) or 99,
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen._edit_mode == "view"
        assert created == []

    async def test_footer_save_hint_flips_with_edit_mode(
        self, mock_app_instance, stub_characters
    ):
        # ctrl+s stays hidden in the native Footer; edit-mode transitions still
        # gate whether saving is meaningful (no-op in view mode).
        bindings_by_key = {binding.key: binding for binding in PersonasScreen.BINDINGS}
        assert bindings_by_key["ctrl+s"].show is False
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            assert screen._edit_mode == "view"

            def _save_action(context):
                return next(a for a in context.actions if a.label == "save")

            assert _save_action(screen._shortcut_context()).available is False
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert screen._edit_mode == "create"
            assert _save_action(screen._shortcut_context()).available is True
            # The footer was re-registered on the transition.
            # task-264: the registration lands on the SCREEN's own footer,
            # not the harness's default-screen stand-in.
            footer = screen.query_one(AppFooterStatus)
            assert "ctrl+s save unavailable" not in footer.shortcut_text
            assert "ctrl+s save" in footer.shortcut_text
            confirms = self._bypass_confirm(screen, True)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # A pristine create session cancels without the discard dialog
            # (change-based dirty tracking).
            assert confirms == []
            assert screen._edit_mode == "view"
            assert _save_action(screen._shortcut_context()).available is False
            # task-445: unavailable hints are dropped entirely rather than
            # rendered with a literal "unavailable" suffix.
            assert "ctrl+s save" not in footer.shortcut_text

    async def test_mode_keys_switch_modes(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.press("ctrl+2")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.active_mode == "personas"
            # ]/[ cycle through the strip order from the active mode.
            # "prompts" is retired from the strip (Task 7), so "dictionaries"
            # is next after "personas".
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert screen.state.active_mode == "dictionaries"
            await pilot.press("left_square_bracket")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.active_mode == "personas"

    async def test_focus_lands_in_editor_name_on_create(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            await self._open_create_editor(pilot)
            await pilot.pause()
            focused = pilot.app.focused
            assert focused is not None
            assert focused.id == "personas-char-editor-name"

    async def test_focus_returns_to_library_after_cancel(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await self._open_create_editor(pilot)
            self._bypass_confirm(screen, True)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            focused = pilot.app.focused
            assert focused is not None
            assert focused.id == "personas-library-rows"


class TestDirtyTracking:
    """UX-E3: change-based dirty tracking, live header state, row badges."""

    @pytest.fixture
    def stub_conversations(self, monkeypatch):
        monkeypatch.setattr(
            character_handler_module, "_default_character_db", lambda: object()
        )
        _install_conversation_db(monkeypatch, [])

    @staticmethod
    def _bypass_confirm(screen, answer: bool) -> list[bool]:
        """Stub the unsaved-changes confirm; returns a call log."""
        calls: list[bool] = []

        async def fake_confirm() -> bool:
            calls.append(answer)
            return answer

        screen._confirm_discard_unsaved = fake_confirm
        return calls

    async def _edit_first_character(self, pilot):
        from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
            EditCharacterRequested,
        )

        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        screen.post_message(EditCharacterRequested("1"))
        await pilot.pause()
        assert screen._edit_mode == "edit"
        return screen

    @staticmethod
    async def _type_in_description(pilot, screen):
        from textual.widgets import TextArea

        screen.query_one("#personas-char-editor-description", TextArea).focus()
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()

    async def test_edit_without_changes_switches_without_dialog(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Edit then click away with zero keystrokes: no discard dialog."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            assert screen.state.has_unsaved_changes is False
            confirms = self._bypass_confirm(screen, True)
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert confirms == []
            assert screen.state.selected_entity_id == "2"
            assert screen._edit_mode == "view"

    async def test_typing_marks_dirty_and_guard_fires(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            await self._type_in_description(pilot, screen)
            assert screen.state.has_unsaved_changes is True
            confirms = self._bypass_confirm(screen, True)
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert confirms == [True]
            assert screen.state.selected_entity_id == "2"

    async def test_persona_editor_typing_marks_dirty_and_guard_fires(
        self, mock_app_instance, stub_characters, stub_conversations, stub_scope_service
    ):
        """Carryover: PersonaProfileEditorWidget._field_changed parity with the
        character editor — typing posts EditorContentChanged exactly once, the
        screen marks the session unsaved, and leaving consults the guard."""
        from textual.widgets import TextArea

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-mode-personas")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#personas-library-row-persona-p-1")
            await pilot.pause()
            screen.post_message(EditPersonaProfileRequested("p-1"))
            await pilot.pause()
            assert screen._edit_mode == "edit"
            # Programmatic population must not have marked the session dirty.
            assert screen.state.has_unsaved_changes is False
            screen.query_one("#personas-editor-description", TextArea).focus()
            await pilot.pause()
            await pilot.press("x")
            await pilot.pause()
            assert screen.state.has_unsaved_changes is True
            readiness = str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            # F-032 intent copy for the unsaved gate.
            assert readiness == "Save or discard your edits to chat in Console."
            confirms = self._bypass_confirm(screen, True)
            await pilot.click("#personas-mode-characters")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert confirms == [True]
            assert screen.state.active_mode == "characters"

    async def test_programmatic_load_does_not_mark_dirty(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Populating the editor (load/new) must not count as a user change."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.has_unsaved_changes is False
            # Same for a fresh create session (new_character population).
            await pilot.press("ctrl+n")
            await pilot.pause()
            await pilot.pause()
            assert screen._edit_mode == "create"
            assert screen.state.has_unsaved_changes is False

    async def test_title_reflects_editing_state(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        monkeypatch.setattr(
            character_handler_module, "update_character", lambda cid, data: True
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            subtitle = screen.query_one(
                "#personas-header #workbench-header-subtitle", Static
            )
            text = str(subtitle.renderable)
            assert "Editing Detective Sam" in text
            assert "unsaved" not in text
            await self._type_in_description(pilot, screen)
            assert "Editing Detective Sam - unsaved" in str(subtitle.renderable)
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # Save-in-place: the editor stays open, so the header keeps
            # showing "Editing <name>" (just without the "- unsaved" suffix
            # now that the save cleared it).
            assert str(subtitle.renderable) == "Editing Detective Sam"
            title = screen.query_one("#personas-header #workbench-header-title", Static)
            # F-034: the screen's one public name matches the nav label.
            assert str(title.renderable) == "Roleplay"

    async def test_active_row_gets_unsaved_badge(
        self, mock_app_instance, stub_characters, stub_conversations, monkeypatch
    ):
        monkeypatch.setattr(
            character_handler_module, "update_character", lambda cid, data: True
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            row = screen.query_one("#personas-library-row-character-1")
            assert "is-unsaved" not in row.classes
            await self._type_in_description(pilot, screen)
            assert "is-unsaved" in row.classes
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # The save refresh rebuilds the rows; the badge must be gone.
            row = screen.query_one("#personas-library-row-character-1")
            assert "is-unsaved" not in row.classes

    async def test_unsaved_badge_cleared_on_discarded_switch(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Discarding edits while switching rows must drop the stale badge."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._edit_first_character(pilot)
            await self._type_in_description(pilot, screen)
            row = screen.query_one("#personas-library-row-character-1")
            assert "is-unsaved" in row.classes
            self._bypass_confirm(screen, True)
            await pilot.click("#personas-library-row-character-2")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.state.selected_entity_id == "2"
            assert not screen.query(".personas-library-row.is-unsaved")

    async def test_import_refreshes_attach_action(
        self,
        mock_app_instance,
        stub_characters,
        stub_conversations,
        monkeypatch,
        tmp_path,
    ):
        """UX-E2 carryover: import-selection must enable the attach action."""
        source = tmp_path / "card.json"
        source.write_bytes(b'{"name":"Detective Sam"}')
        # F-031 auto-selects the first row on a fresh mount when rows exist;
        # the "no prior selection" baseline this test pins needs an empty
        # library.
        monkeypatch.setattr(
            character_handler_module, "fetch_all_characters", lambda: []
        )
        monkeypatch.setattr(
            character_handler_module,
            "inspect_character_card_tts_attachment",
            lambda _source_bytes: CharacterCardTTSInspection(),
        )
        monkeypatch.setattr(
            character_handler_module,
            "import_character_card_with_outcome",
            lambda _source_bytes: CharacterCardImportOutcome(1, False, None, None),
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            assert screen._console_action_allowed() is False  # no prior selection
            draft = next(
                a
                for a in screen._shortcut_context().actions
                if a.label == "Send to Console draft"
            )
            assert draft.available is False  # no prior selection
            # task-264: the registration lands on the SCREEN's own footer,
            # not the harness's default-screen stand-in.
            footer = screen.query_one(AppFooterStatus)
            # task-445: unavailable hints are dropped entirely rather than
            # rendered with a literal "unavailable" suffix.
            assert (
                "ctrl+enter send to console draft" not in footer.shortcut_text.lower()
            )
            await screen._import_character_from_path(str(source))
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            assert screen._console_action_allowed() is True


class TestCharacterTTSPortabilityDialogs:
    """Mounted contracts for the two explicit imported-voice decisions."""

    @staticmethod
    def _collision_plan(
        allowed_choices: tuple[str, ...],
    ) -> PortableProfileImportPlan:
        profile = _character_tts_profile(1)
        portable = _portable_tts_profile(profile)
        return PortableProfileImportPlan(
            observation=PortableProfileAvailabilityObservation(
                7,
                3,
                portable,
                "available",
            ),
            allowed_choices=allowed_choices,
            reuse_profile=profile if "reuse" in allowed_choices else None,
            copy_candidate=_portable_tts_profile(
                profile,
                profile_id=UUID("33333333-3333-4333-8333-333333333333"),
            ),
        )

    async def test_collision_dialog_returns_explicit_reuse_choice(self):
        from tldw_chatbook.Widgets.Persona_Widgets.character_tts_portability_dialogs import (
            CharacterTTSProfileCollisionDialog,
        )

        app = App()
        results: list[str | None] = []
        plan = self._collision_plan(("reuse", "copy"))
        async with app.run_test(size=(100, 35)) as pilot:
            dialog = CharacterTTSProfileCollisionDialog(plan)
            await app.push_screen(dialog, callback=results.append)
            await pilot.pause()

            assert dialog.query_one("#character-tts-collision-reuse", Button)
            assert dialog.query_one("#character-tts-collision-copy-profile", Button)
            await pilot.click("#character-tts-collision-reuse")
            await pilot.pause()

        assert results == ["reuse"]

    async def test_collision_dialog_returns_copy_when_reuse_is_unsafe(self):
        from tldw_chatbook.Widgets.Persona_Widgets.character_tts_portability_dialogs import (
            CharacterTTSProfileCollisionDialog,
        )

        app = App()
        results: list[str | None] = []
        plan = self._collision_plan(("copy",))
        async with app.run_test(size=(100, 35)) as pilot:
            dialog = CharacterTTSProfileCollisionDialog(plan)
            await app.push_screen(dialog, callback=results.append)
            await pilot.pause()

            assert not dialog.query("#character-tts-collision-reuse")
            await pilot.click("#character-tts-collision-copy-profile")
            await pilot.pause()

        assert results == ["copy"]

    async def test_existing_character_dialog_requires_explicit_apply(self):
        from tldw_chatbook.Widgets.Persona_Widgets.character_tts_portability_dialogs import (
            CharacterTTSExistingAssignmentDialog,
        )

        app = App()
        results: list[bool] = []
        async with app.run_test(size=(100, 35)) as pilot:
            dialog = CharacterTTSExistingAssignmentDialog()
            await app.push_screen(dialog, callback=results.append)
            await pilot.pause()

            assert dialog.query_one("#character-tts-existing-cancel", Button)
            await pilot.click("#character-tts-existing-confirm")
            await pilot.pause()

        assert results == [True]


class TestConfirmationDialogEscape:
    """Keyboard users must be able to dismiss the shared confirm dialog."""

    async def test_confirmation_dialog_escape_cancels(self):
        from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

        results: list[bool] = []

        class DialogApp(ConsolidatedCSSApp):
            def on_mount(self) -> None:
                self.push_screen(ConfirmationDialog(), callback=results.append)

        app = DialogApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(pilot.app.screen, ConfirmationDialog)
            await pilot.press("escape")
            await pilot.pause()
            assert results == [False]
            assert not isinstance(pilot.app.screen, ConfirmationDialog)


class TestImportExportFilters:
    """The file-picker filters must use callable testers, not glob strings.

    Regression guard for the P0 crash: ``Filter.__call__`` does
    ``self.tester(path)``; a glob STRING tester ("*.json") raises
    ``TypeError: 'str' object is not callable`` and tears down the session
    when Import / Export JSON / Export PNG is pressed. We drive the real
    import/export workers, capture the picker actually built, and assert
    every filter is callable and returns a bool without raising.
    """

    @staticmethod
    def _assert_filters_callable(filters) -> None:
        from pathlib import Path

        # ``selections`` enumerates every registered filter; each must be a
        # callable tester that survives being invoked on a Path.
        assert bool(filters)
        for _name, filter_id in filters.selections:
            entry = filters[filter_id]
            for sample in (Path("x.json"), Path("x.png"), Path("x.txt")):
                result = entry(sample)
                assert isinstance(result, bool)

    async def _capture_picker(self, pilot, screen, launch):
        from unittest.mock import AsyncMock

        captured: dict = {}

        async def _fake_push_screen_wait(picker):
            captured["picker"] = picker
            return None  # user cancels; the worker returns cleanly

        pilot.app.push_screen_wait = AsyncMock(side_effect=_fake_push_screen_wait)
        await launch()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert "picker" in captured, "picker was never pushed"
        return captured["picker"]

    async def test_import_filters_are_callable(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            picker = await self._capture_picker(
                pilot, screen, screen._import_dialog_worker
            )
            self._assert_filters_callable(picker.filters)

    async def test_import_filters_include_markdown(
        self, mock_app_instance, stub_characters
    ):
        from pathlib import Path

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            picker = await self._capture_picker(
                pilot, screen, screen._import_dialog_worker
            )
            filter_by_name = {
                name: picker.filters[filter_id]
                for name, filter_id in picker.filters.selections
            }

            # Markdown stays importable via its own dedicated sub-filter, but
            # is NOT part of the broad "Character Cards" default (task-431
            # AC#1): a plain docs folder full of .md files should not read
            # as a folder of character cards.
            assert "Markdown Files" in filter_by_name
            assert filter_by_name["Character Cards"](Path("character.md")) is False
            assert (
                filter_by_name["Character Cards"](Path("character.markdown")) is False
            )
            assert filter_by_name["Markdown Files"](Path("character.md")) is True
            assert filter_by_name["Markdown Files"](Path("character.markdown")) is True
            assert filter_by_name["Markdown Files"](Path("character.json")) is False

    async def test_import_filters_character_cards_accepts_webp_not_md(
        self, mock_app_instance, stub_characters
    ):
        """task-431 AC#1: primary filter accepts .webp, drops .md as a card."""
        from pathlib import Path

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            picker = await self._capture_picker(
                pilot, screen, screen._import_dialog_worker
            )
            filter_by_name = {
                name: picker.filters[filter_id]
                for name, filter_id in picker.filters.selections
            }

            character_cards = filter_by_name["Character Cards"]
            assert character_cards(Path("x.webp")) is True
            assert character_cards(Path("x.png")) is True
            assert character_cards(Path("x.json")) is True
            assert character_cards(Path("README.md")) is False
            assert character_cards(Path("x.markdown")) is False

            assert "Card Images (PNG/WebP)" in filter_by_name
            card_images = filter_by_name["Card Images (PNG/WebP)"]
            assert card_images(Path("x.png")) is True
            assert card_images(Path("x.webp")) is True
            assert card_images(Path("x.json")) is False

    async def test_export_json_filters_are_callable(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            picker = await self._capture_picker(
                pilot, screen, lambda: screen._export_dialog_worker("json")
            )
            self._assert_filters_callable(picker.filters)

    async def test_export_png_filters_are_callable(
        self, mock_app_instance, stub_characters
    ):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            picker = await self._capture_picker(
                pilot, screen, lambda: screen._export_dialog_worker("png")
            )
            self._assert_filters_callable(picker.filters)

    @pytest.mark.parametrize("outcome", ["cancel", "error", "selection"])
    async def test_avatar_upload_worker_resets_flag_and_uses_avatar_context(
        self, mock_app_instance, stub_characters, tmp_path, outcome
    ):
        avatar = tmp_path / "avatar.png"
        avatar.write_bytes(b"\x89PNG selected avatar")
        app = PersonasTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await pilot.pause()
            await pilot.click("#personas-library-new")
            await pilot.pause()
            captured: dict = {}

            async def _fake_push_screen_wait(picker):
                captured["picker"] = picker
                if outcome == "error":
                    raise RuntimeError("dialog failed")
                if outcome == "selection":
                    return avatar
                return None

            pilot.app.push_screen_wait = AsyncMock(side_effect=_fake_push_screen_wait)
            screen._io_dialog_active = True

            await screen._avatar_upload_dialog_worker()
            await pilot.pause()

            assert screen._io_dialog_active is False
            picker = captured["picker"]
            assert picker.context == "character_avatar_upload"
            self._assert_filters_callable(picker.filters)
            editor = screen.query_one(PersonasCharacterEditorWidget)
            if outcome == "selection":
                assert (
                    editor.get_character_data()["image"] == b"\x89PNG selected avatar"
                )
            else:
                assert "image" not in editor.get_character_data()


async def test_character_import_filters_helper_accepts_webp_not_md():
    """Unit-level guard on the module-level filter helper itself (task-431 AC#1).

    Exercises ``_character_import_filters`` directly (no screen mount) so the
    primary "Character Cards" tester's behavior is pinned independently of
    the dialog-worker integration test in ``TestImportExportFilters``. Marked
    ``async`` (with no ``await``) only to match this module's file-wide
    ``pytestmark = pytest.mark.asyncio`` and avoid its inapplicable-mark
    warning on sync functions.
    """
    from pathlib import Path

    filters = personas_screen_module._character_import_filters()
    filter_by_name = {
        name: filters[filter_id] for name, filter_id in filters.selections
    }
    character_cards = filter_by_name["Character Cards"]

    assert character_cards(Path("x.webp")) is True
    assert character_cards(Path("x.png")) is True
    assert character_cards(Path("x.json")) is True
    assert character_cards(Path("README.md")) is False
    assert character_cards(Path("x.markdown")) is False


# --- Roleplay UAT: character rows were identified only by a date ---
# Live repro (origin/dev @ f384a2807): every character row carried a
# YYYY-MM-DD last-modified line and nothing else. Both characters in the
# library showed the SAME date, so the only secondary information on the row
# discriminated nothing. In a roleplay library that grows to dozens of
# characters, a one-line description is what makes a row recognizable.


async def test_character_row_meta_prefers_a_description_snippet_over_the_date() -> None:
    """A character row should say who the character is, not when it was touched."""
    rows = personas_screen_module.PersonasScreen._build_library_rows(
        [
            {
                "id": 2,
                "name": "Seraphina",
                "last_modified": "2026-07-26T10:00:00",
                "description": (
                    "Seraphina is the last archivist of a drowned library. "
                    "She is guarded, dryly funny, and speaks in careful, "
                    "deliberate sentences."
                ),
            }
        ],
        "character",
    )

    assert len(rows) == 1
    assert "archivist of a drowned library" in rows[0].meta


async def test_character_row_meta_falls_back_to_the_date_without_a_description() -> (
    None
):
    """Characters with no description keep their previous date meta line."""
    rows = personas_screen_module.PersonasScreen._build_library_rows(
        [{"id": 3, "name": "Blank", "last_modified": "2026-07-26T10:00:00"}],
        "character",
    )

    assert rows[0].meta == "2026-07-26"


async def test_character_row_meta_is_a_single_bounded_line() -> None:
    """A long description must not blow the row height out."""
    rows = personas_screen_module.PersonasScreen._build_library_rows(
        [{"id": 4, "name": "Verbose", "description": "word " * 200}],
        "character",
    )

    assert "\n" not in rows[0].meta
    assert len(rows[0].meta) <= 80


# --- Roleplay UAT: the inspector never showed the character's portrait ---
# Selecting a character surfaced its name, type, validation, conversations and
# actions, but no picture -- for a roleplay user the portrait is a primary
# identifying attribute, and the machinery to render one already existed for
# the editor thumbnail.


async def test_inspector_pane_exposes_an_avatar_thumbnail_holder() -> None:
    """The inspector must have somewhere to render the selected portrait."""
    from textual.containers import Container

    from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import (
        PersonasInspectorPane,
    )

    class _Host(ConsolidatedCSSApp):
        def compose(self):
            yield PersonasInspectorPane()

    app = _Host()
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        holder = app.query_one("#personas-inspector-avatar-thumb", Container)
        assert holder is not None


async def test_inspector_avatar_thumbnail_mounts_and_clears() -> None:
    """A prepared renderable mounts; None clears it back to empty."""
    from textual.containers import Container
    from textual.widgets import Static as _S

    from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import (
        PersonasInspectorPane,
    )

    class _Host(ConsolidatedCSSApp):
        def compose(self):
            yield PersonasInspectorPane()

    app = _Host()
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        pane = app.query_one(PersonasInspectorPane)

        pane.set_avatar_thumbnail(_S("portrait"))
        await pilot.pause()
        holder = app.query_one("#personas-inspector-avatar-thumb", Container)
        assert len(holder.children) == 1

        pane.set_avatar_thumbnail(None)
        await pilot.pause()
        assert len(holder.children) == 0


async def test_debounced_validation_does_not_erase_a_blocked_save_message(
    mock_app_instance, stub_characters, monkeypatch
):
    """A gated re-validation must not wipe the footer the save path wrote.

    `_run_validation` is debounced and, on an untouched form, rendered nothing
    by calling `show_validation(())` -- which CLEARS. A blocked save writes
    "name: required" into that same footer, so any later field/validation churn
    made the blocker silently vanish while the save stayed blocked. The
    freshly-opened case is already handled by `load_character`, which clears
    the footer explicitly.
    """
    from textual.widgets import Static

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-new")
        await pilot.pause()
        editor = screen.query_one(PersonasCharacterEditorWidget)
        editor.show_validation(("name: required",))
        await pilot.pause()

        # An untouched form re-validating (the debounced path) must leave it be.
        editor._user_touched = False
        editor._run_validation()
        await pilot.pause()

        footer = screen.query_one("#personas-char-editor-validation", Static)
        assert "name: required" in str(footer.renderable)


# --- Character Voice & Speech assignment controls (TASK-617.5) ---


def _character_tts_profile(index: int) -> TTSGenerationProfile:
    timestamp = datetime(2026, 7, 31, tzinfo=UTC)
    display_name = f"Roleplay Voice {index}"
    return TTSGenerationProfile(
        profile_id=UUID(int=index),
        display_name=display_name,
        normalized_name=display_name.casefold(),
        provider_id="audio_cpp",
        model_id=f"model-{index}",
        voice_id=f"voice-{index}",
        response_format="wav",
        speed=1.0,
        options={},
        revision=1,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _character_tts_availability(
    page: TTSProfilePageSnapshot,
    *,
    state: str = "available",
    configuration_revision: int = 4,
    catalog_revision: int | None = 8,
    dependency: TTSProfileDependencyProjection | None = None,
) -> TTSProfileAvailabilitySnapshot:
    recovery = {
        "available": "none",
        "unavailable": "edit",
        "unverified": "refresh",
    }[state]
    return TTSProfileAvailabilitySnapshot(
        repository_generation=page.repository_generation,
        configuration_revision=configuration_revision,
        catalog_revision=catalog_revision,
        profiles=tuple(
            TTSProfileAvailability(
                profile_id=profile.profile_id,
                state=state,  # type: ignore[arg-type]
                recovery_action=recovery,  # type: ignore[arg-type]
                dependency=dependency or TTSProfileDependencyProjection(),
            )
            for profile in page.profiles
        ),
    )


class _CharacterTTSProfileService:
    def __init__(
        self,
        *,
        page: TTSProfilePageSnapshot,
        assigned: LoadedCharacterTTSAssignment,
        availability_state: str = "available",
        dependency: TTSProfileDependencyProjection | None = None,
    ) -> None:
        self.page = page
        self.assigned = assigned
        self.availability_state = availability_state
        self.dependency = dependency
        self.assignment_count_value = 1
        self.get_calls: list[CharacterRef] = []
        self.availability_calls: list[TTSProfilePageSnapshot] = []
        self.set_calls: list[
            tuple[CharacterRef, LoadedTTSProfile, CharacterTTSAssignment | None]
        ] = []
        self.detach_calls: list[tuple[CharacterTTSAssignment, int]] = []
        self.update_calls: list[tuple[LoadedTTSProfile, TTSProfileDraft]] = []
        self.set_error: BaseException | None = None
        self.extra_profiles: dict[UUID, LoadedTTSProfile] = {}
        self.get_profile_calls: list[UUID] = []

    async def get_assigned_profile(
        self, character_ref: CharacterRef
    ) -> LoadedCharacterTTSAssignment:
        self.get_calls.append(character_ref)
        return self.assigned

    async def list_profiles(
        self, *, search: str | None = None, offset: int = 0
    ) -> TTSProfilePageSnapshot:
        assert search is None
        assert offset == 0
        return self.page

    async def get_profile(self, profile_id: UUID) -> LoadedTTSProfile:
        self.get_profile_calls.append(profile_id)
        loaded = self.extra_profiles.get(profile_id)
        if loaded is None:
            raise ProfileRepositoryError("missing")
        return loaded

    async def observe_availability(
        self, page: TTSProfilePageSnapshot
    ) -> TTSProfileAvailabilitySnapshot:
        self.availability_calls.append(page)
        return _character_tts_availability(
            page,
            state=self.availability_state,
            dependency=self.dependency,
        )

    async def assignment_count(self, loaded: LoadedTTSProfile) -> int:
        return self.assignment_count_value

    async def set_assignment(
        self,
        character_ref: CharacterRef,
        loaded: LoadedTTSProfile,
        expected_current: CharacterTTSAssignment | None,
    ) -> CharacterTTSAssignment:
        self.set_calls.append((character_ref, loaded, expected_current))
        if self.set_error is not None:
            raise self.set_error
        return CharacterTTSAssignment(
            character_ref=character_ref,
            profile_id=loaded.profile.profile_id,
        )

    async def detach_assignment(
        self,
        assignment: CharacterTTSAssignment,
        repository_generation: int,
    ) -> None:
        self.detach_calls.append((assignment, repository_generation))

    async def update_profile(
        self,
        loaded: LoadedTTSProfile,
        draft: TTSProfileDraft,
    ) -> LoadedTTSProfile:
        self.update_calls.append((loaded, draft))
        return loaded

    def preview_preset(
        self,
        loaded: LoadedTTSProfile,
        availability: TTSProfileAvailability,
    ) -> TTSPlaygroundSelectionPreset:
        profile = loaded.profile
        return TTSPlaygroundSelectionPreset(
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
            availability=availability.state,
        )


class _CharacterTTSWidgetHost(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.actions: list[CharacterTTSActionRequested] = []

    def compose(self):
        yield PersonasCharacterTTSWidget(context="card")

    def on_character_ttsaction_requested(
        self, message: CharacterTTSActionRequested
    ) -> None:
        self.actions.append(message)


async def test_character_tts_widget_renders_disabled_global_and_broken_assignment() -> (
    None
):
    profile = _character_tts_profile(1)
    available = CharacterTTSProfileOption(
        profile_id=profile.profile_id,
        display_name=profile.display_name,
        availability="available",
    )
    unavailable = CharacterTTSProfileOption(
        profile_id=profile.profile_id,
        display_name=profile.display_name,
        availability="unavailable",
    )
    app = _CharacterTTSWidgetHost()
    async with app.run_test() as pilot:
        widget = app.query_one(PersonasCharacterTTSWidget)
        selector = widget.query_one(Select)

        widget.apply_state(CharacterTTSPresentationState.disabled())
        await pilot.pause()
        assert selector.disabled is True
        assert "Save/reopen before assigning" in str(
            widget.query_one(".personas-character-tts-status", Static).renderable
        )

        widget.apply_state(
            CharacterTTSPresentationState(
                profiles=(available,),
                selected_profile_id=None,
                status="Using the global speech default.",
                controls_enabled=True,
            )
        )
        await pilot.pause()
        assert selector.value == "__global__"
        assert selector.disabled is False

        widget.apply_state(
            CharacterTTSPresentationState(
                profiles=(unavailable,),
                selected_profile_id=profile.profile_id,
                status="Unavailable · repair the profile or remove this assignment.",
                controls_enabled=True,
                assignment_count=3,
            )
        )
        await pilot.pause()
        assert selector.value == str(profile.profile_id)
        assert profile.display_name in str(
            selector.query_one("#label", Static).renderable
        )
        assert "Unavailable" in str(
            widget.query_one(".personas-character-tts-status", Static).renderable
        )
        assert (
            widget.query_one(".personas-character-tts-remove", Button).disabled is False
        )
        assert (
            str(widget.query_one(".personas-character-tts-edit", Button).label)
            == "Repair"
        )


async def test_character_tts_widget_emits_id_only_intents_for_available_profiles() -> (
    None
):
    available = _character_tts_profile(1)
    unavailable = _character_tts_profile(2)
    app = _CharacterTTSWidgetHost()
    async with app.run_test() as pilot:
        widget = app.query_one(PersonasCharacterTTSWidget)
        widget.apply_state(
            CharacterTTSPresentationState(
                profiles=(
                    CharacterTTSProfileOption(
                        available.profile_id,
                        available.display_name,
                        "available",
                    ),
                    CharacterTTSProfileOption(
                        unavailable.profile_id,
                        unavailable.display_name,
                        "unavailable",
                    ),
                ),
                selected_profile_id=None,
                status="Using the global speech default.",
                controls_enabled=True,
            )
        )
        await pilot.pause()
        selector = widget.query_one(Select)

        selector.value = str(unavailable.profile_id)
        await pilot.pause()
        assert app.actions == []
        assert selector.value == "__global__"

        selector.value = str(available.profile_id)
        await pilot.pause()
        assert len(app.actions) == 1
        assert app.actions[0].action == "assign"
        assert app.actions[0].profile_id == available.profile_id
        assert vars(app.actions[0]).keys() >= {"action", "profile_id"}
        assert "authority" not in vars(app.actions[0])


async def test_character_tts_widget_refuses_dependency_blocked_inactive_profile() -> (
    None
):
    profile = _character_tts_profile(1)
    blocked = CharacterTTSProfileOption(
        profile.profile_id,
        profile.display_name,
        "available",
        dependency=TTSProfileDependencyProjection(
            reason="recipe_missing",
            display="Needs compatible model",
            action="open_audio_cpp_settings",
        ),
    )
    app = _CharacterTTSWidgetHost()
    async with app.run_test(size=(80, 24)) as pilot:
        widget = app.query_one(PersonasCharacterTTSWidget)
        widget.apply_state(
            CharacterTTSPresentationState(
                profiles=(blocked,),
                selected_profile_id=None,
                status="Using the global speech default.",
                controls_enabled=True,
            )
        )
        await pilot.pause()
        selector = widget.query_one(Select)

        assert "Needs compatible model" in next(
            str(label)
            for label, value in selector._options
            if value == str(profile.profile_id)
        )
        selector.value = str(profile.profile_id)
        await pilot.pause()

        assert app.actions == []
        assert selector.value == "__global__"


async def test_character_tts_widget_renders_and_dispatches_shared_recovery_truth() -> (
    None
):
    profile = _character_tts_profile(1)
    dependency = TTSProfileDependencyProjection(
        reason="recipe_mismatch",
        display="Needs compatible model",
        action="open_audio_cpp_settings",
        advisory="recipe_provenance_unavailable",
        advisory_display="Recipe provenance unavailable",
        advisory_action="generate_new_profile",
    )
    option = CharacterTTSProfileOption(
        profile.profile_id,
        profile.display_name,
        "available",
        dependency=dependency,
    )
    app = _CharacterTTSWidgetHost()

    async with app.run_test(size=(40, 24)) as pilot:
        widget = app.query_one(PersonasCharacterTTSWidget)
        widget.apply_state(
            CharacterTTSPresentationState(
                profiles=(option,),
                selected_profile_id=profile.profile_id,
                status="Needs compatible model. Recipe provenance unavailable.",
                controls_enabled=True,
            )
        )
        await pilot.pause()
        actions = dependency_recovery_actions(dependency)
        blocker = widget.query_one(".personas-character-tts-dependency-primary", Button)
        advisory = widget.query_one(
            ".personas-character-tts-dependency-advisory", Button
        )
        assert (str(blocker.label), blocker.tooltip) == (
            actions[0].label,
            actions[0].tooltip,
        )
        assert (str(advisory.label), advisory.tooltip) == (
            actions[1].label,
            actions[1].tooltip,
        )
        action_area = widget.query_one(".personas-character-tts-actions")
        assert widget.region.width == 40
        for button in (blocker, advisory):
            assert button.region.width == action_area.region.width
            assert action_area.region.contains_region(button.region), str(button.label)

        selector = widget.query_one(Select)
        selector.focus()
        focused_recovery: list[str] = []
        for _ in range(5):
            await pilot.press("tab")
            focused = app.focused
            if focused in (blocker, advisory):
                assert isinstance(focused, Button)
                focused_recovery.append(str(focused.label))
                await pilot.press("enter")

        await pilot.pause()

        assert focused_recovery == [actions[0].label, actions[1].label]
        assert [(message.action, message.profile_id) for message in app.actions] == [
            ("open_audio_cpp_settings", profile.profile_id),
            ("generate_new_profile", profile.profile_id),
        ]


async def test_character_tts_suggestion_is_guidance_not_assignment() -> None:
    profile = _character_tts_profile(1)
    option = CharacterTTSProfileOption(
        profile.profile_id,
        profile.display_name,
        "available",
    )
    app = _CharacterTTSWidgetHost()
    async with app.run_test() as pilot:
        widget = app.query_one(PersonasCharacterTTSWidget)
        widget.apply_state(
            CharacterTTSPresentationState(
                profiles=(option,),
                selected_profile_id=None,
                suggested_profile_id=profile.profile_id,
                status="Using the global speech default.",
                controls_enabled=True,
            )
        )
        await pilot.pause()
        selector = widget.query_one(Select)

        assert selector.value == "__global__"
        assert app.actions == []
        assert "Suggested" in next(
            str(label)
            for label, value in selector._options
            if value == str(profile.profile_id)
        )

        selector.value = str(profile.profile_id)
        await pilot.pause()

        assert [(message.action, message.profile_id) for message in app.actions] == [
            ("assign", profile.profile_id)
        ]


async def test_character_tts_widget_accepts_unverified_profile_assignment_without_laundering_it() -> (
    None
):
    """A legacy-provider profile is always classified 'unverified' this slice
    (task-2450 amendment). Refusing to assign it made no legacy profile ever
    assignable through this UI (task-2453) -- it must be assignable, but the
    option row and the post-assignment Edit/Repair label must keep saying so
    honestly rather than presenting it as a confirmed-working 'available'
    profile."""

    unverified = _character_tts_profile(1)
    unavailable = _character_tts_profile(2)
    app = _CharacterTTSWidgetHost()
    async with app.run_test() as pilot:
        widget = app.query_one(PersonasCharacterTTSWidget)
        widget.apply_state(
            CharacterTTSPresentationState(
                profiles=(
                    CharacterTTSProfileOption(
                        unverified.profile_id,
                        unverified.display_name,
                        "unverified",
                        recovery_action="none",
                    ),
                    CharacterTTSProfileOption(
                        unavailable.profile_id,
                        unavailable.display_name,
                        "unavailable",
                    ),
                ),
                selected_profile_id=None,
                status="Using the global speech default.",
                controls_enabled=True,
            )
        )
        await pilot.pause()
        selector = widget.query_one(Select)

        # This profile's provider has no catalog to preflight
        # (recovery_action == "none"), so the option row must say so
        # honestly -- selecting it must never present it as a
        # confirmed-working "available" profile.
        option_labels = [str(prompt) for prompt, _value in selector._options]
        unverified_label = next(
            label for label in option_labels if unverified.display_name in label
        )
        assert "no catalog check" in unverified_label
        assert "available" not in unverified_label

        # Still refused: genuinely unavailable stays refused, unchanged.
        selector.value = str(unavailable.profile_id)
        await pilot.pause()
        assert app.actions == []
        assert selector.value == "__global__"

        # Newly accepted: unverified must be assignable.
        selector.value = str(unverified.profile_id)
        await pilot.pause()
        assert len(app.actions) == 1
        assert app.actions[0].action == "assign"
        assert app.actions[0].profile_id == unverified.profile_id
        assert selector.value == str(unverified.profile_id)

        # The assignment must render honestly, not as a verified/available
        # one: the Edit button stays "Edit" (not "Repair" -- that copy is
        # reserved for a genuinely unavailable profile), and the status text
        # passed in by the caller (asserted separately in
        # personas_screen.py's own tests) is rendered verbatim.
        widget.apply_state(
            CharacterTTSPresentationState(
                profiles=(
                    CharacterTTSProfileOption(
                        unverified.profile_id,
                        unverified.display_name,
                        "unverified",
                        recovery_action="none",
                    ),
                ),
                selected_profile_id=unverified.profile_id,
                status=(
                    f"{unverified.display_name} · No catalog check · Used by 1 "
                    "character. The exact selection is used as-is; the "
                    "assignment is preserved."
                ),
                controls_enabled=True,
                assignment_count=1,
            )
        )
        await pilot.pause()
        assert (
            str(widget.query_one(".personas-character-tts-edit", Button).label)
            == "Edit"
        )
        status_text = str(
            widget.query_one(".personas-character-tts-status", Static).renderable
        )
        assert "No catalog check" in status_text
        assert "unverified" not in status_text.casefold()


def _configure_character_tts_app(
    mock_app_instance: Any,
    service: _CharacterTTSProfileService,
) -> None:
    mock_app_instance.runtime_backend = "local"
    mock_app_instance.chachanotes_db = SimpleNamespace(
        get_local_authority_id=lambda: "local-test-authority",
        get_character_card_by_id=lambda _character_id: {
            **CHARACTERS[0],
            "extensions": {},
        },
    )
    mock_app_instance._ensure_tts_profile_service = AsyncMock(return_value=service)


async def test_character_tts_population_requires_one_generation_and_observes_off_page_assignment(
    mock_app_instance,
    stub_characters,
    monkeypatch,
) -> None:
    # F-031: an empty library keeps first-paint auto-select from consuming
    # this state machine's exact-call seams before the explicit select below.
    monkeypatch.setattr(character_handler_module, "fetch_all_characters", lambda: [])
    first_page_profile = _character_tts_profile(1)
    assigned_profile = _character_tts_profile(51)
    character_ref = CharacterRef(
        source="local",
        authority_id="local-test-authority",
        character_id="1",
    )
    assignment = CharacterTTSAssignment(
        character_ref=character_ref,
        profile_id=assigned_profile.profile_id,
    )
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(first_page_profile,),
            total=51,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=AssignedTTSProfileSnapshot(
                assignment=assignment,
                profile=assigned_profile,
            ),
        ),
        availability_state="unavailable",
    )
    service.assignment_count_value = 4
    _configure_character_tts_app(mock_app_instance, service)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()

        assert [len(call.profiles) for call in service.availability_calls] == [1, 1]
        assert service.availability_calls[1].profiles == (assigned_profile,)
        assert screen._character_tts_snapshot is not None
        assert screen._character_tts_snapshot.repository_generation == 7
        card_control = screen.query_one("#personas-character-card-tts")
        editor_control = screen.query_one("#personas-character-editor-tts")
        assert card_control.presentation_state.selected_profile_id == (
            assigned_profile.profile_id
        )
        assert card_control.presentation_state is editor_control.presentation_state
        assert card_control.display is True
        assert editor_control.display is True
        assert screen.query_one("#ccp-character-card-view").display is True
        assert screen.query_one("#ccp-character-editor-view").display is False
        assert card_control.presentation_state.assignment_count == 4
        assert "Unavailable" in card_control.presentation_state.status
        actions = card_control.query_one(".personas-character-tts-actions")
        for button in actions.query(Button):
            if button.display:
                assert actions.region.contains_region(button.region), str(button.label)


async def test_roleplay_profile_suggestion_requires_character_choice_and_exact_fresh_revision(
    mock_app_instance,
    stub_characters,
) -> None:
    profile = replace(_character_tts_profile(1), revision=2)
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(7, None),
    )
    _configure_character_tts_app(mock_app_instance, service)
    suggestion = personas_screen_module.CharacterTTSProfileSuggestion(
        profile_id=profile.profile_id,
        repository_generation=7,
        profile_revision=2,
    )
    app = PersonasTestApp(mock_app_instance)
    screen = PersonasScreen(mock_app_instance)
    screen.apply_navigation_context(
        {"view": "characters", "voice_profile_suggestion": suggestion}
    )

    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert screen.state.selected_entity_id is None
        assert service.set_calls == []

        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        state = screen._character_tts_presentation
        assert state.selected_profile_id is None
        assert state.suggested_profile_id == profile.profile_id
        assert service.set_calls == []
        selector = screen.query_one(
            "#personas-character-card-tts .personas-character-tts-profile",
            Select,
        )
        assert selector.value == "__global__"
        assert app.focused is selector
        actions = screen.query_one(
            "#personas-character-card-tts .personas-character-tts-actions"
        )
        for button in actions.query(Button):
            if button.display:
                assert actions.region.contains_region(button.region), str(button.label)

        selector.value = str(profile.profile_id)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        assert service.set_calls == [
            (
                screen._character_tts_snapshot.character_ref,
                LoadedTTSProfile(7, profile),
                None,
            )
        ]
        assert screen._character_tts_profile_suggestion is None


async def test_roleplay_profile_suggestion_clears_when_generation_or_revision_is_stale(
    mock_app_instance,
    stub_characters,
) -> None:
    profile = replace(_character_tts_profile(1), revision=3)
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(7, (profile,), 1),
        assigned=LoadedCharacterTTSAssignment(7, None),
    )
    _configure_character_tts_app(mock_app_instance, service)
    suggestion = personas_screen_module.CharacterTTSProfileSuggestion(
        profile_id=profile.profile_id,
        repository_generation=7,
        profile_revision=2,
    )
    app = PersonasTestApp(mock_app_instance)
    screen = PersonasScreen(mock_app_instance)
    screen.apply_navigation_context(
        {"view": "characters", "voice_profile_suggestion": suggestion}
    )

    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()

        assert screen._character_tts_profile_suggestion is None
        assert screen._character_tts_presentation.suggested_profile_id is None
        assert service.set_calls == []


async def test_roleplay_profile_suggestion_resolves_exact_profile_beyond_first_page(
    mock_app_instance,
    stub_characters,
) -> None:
    first_page = _character_tts_profile(1)
    suggested_profile = replace(_character_tts_profile(51), revision=2)
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(7, (first_page,), 51),
        assigned=LoadedCharacterTTSAssignment(7, None),
    )
    service.extra_profiles[suggested_profile.profile_id] = LoadedTTSProfile(
        7,
        suggested_profile,
    )
    _configure_character_tts_app(mock_app_instance, service)
    suggestion = personas_screen_module.CharacterTTSProfileSuggestion(
        profile_id=suggested_profile.profile_id,
        repository_generation=7,
        profile_revision=2,
    )
    app = PersonasTestApp(mock_app_instance)
    screen = PersonasScreen(mock_app_instance)
    screen.apply_navigation_context(
        {"view": "characters", "voice_profile_suggestion": suggestion}
    )

    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()

        assert service.get_profile_calls == [suggested_profile.profile_id]
        assert (
            screen._character_tts_presentation.suggested_profile_id
            == suggested_profile.profile_id
        )
        assert screen._character_tts_presentation.selected_profile_id is None
        assert service.set_calls == []


async def test_character_tts_population_rejects_mixed_repository_generations(
    mock_app_instance,
    stub_characters,
) -> None:
    profile = _character_tts_profile(1)
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=8,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=None,
        ),
    )
    _configure_character_tts_app(mock_app_instance, service)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()

        assert screen._character_tts_snapshot is None
        assert (
            screen.query_one(
                "#personas-character-card-tts"
            ).presentation_state.controls_enabled
            is False
        )


async def test_character_tts_off_page_assignment_requires_matching_capability_revisions(
    mock_app_instance,
    stub_characters,
    monkeypatch,
) -> None:
    # F-031: empty library - see the population-generation test above.
    monkeypatch.setattr(character_handler_module, "fetch_all_characters", lambda: [])
    first_page_profile = _character_tts_profile(1)
    assigned_profile = _character_tts_profile(51)
    character_ref = CharacterRef(
        source="local",
        authority_id="local-test-authority",
        character_id="1",
    )

    class _RevisionChangingService(_CharacterTTSProfileService):
        async def observe_availability(
            self, page: TTSProfilePageSnapshot
        ) -> TTSProfileAvailabilitySnapshot:
            self.availability_calls.append(page)
            return _character_tts_availability(
                page,
                configuration_revision=3 + len(self.availability_calls),
                catalog_revision=8,
            )

    service = _RevisionChangingService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(first_page_profile,),
            total=51,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=AssignedTTSProfileSnapshot(
                assignment=CharacterTTSAssignment(
                    character_ref=character_ref,
                    profile_id=assigned_profile.profile_id,
                ),
                profile=assigned_profile,
            ),
        ),
    )
    _configure_character_tts_app(mock_app_instance, service)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()

        assert len(service.availability_calls) == 4
        assert screen._character_tts_snapshot is None
        assert screen._character_tts_presentation.controls_enabled is False


@pytest.mark.parametrize("final_authority_check", ["changed", "error"])
async def test_character_tts_server_principal_change_rejects_late_population(
    mock_app_instance,
    stub_characters,
    final_authority_check: str,
) -> None:
    profile = _character_tts_profile(1)
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=None,
        ),
    )
    started = asyncio.Event()
    release = asyncio.Event()

    class _ServerAuthorityProvider:
        def __init__(self) -> None:
            self.capture = object()
            self.current = True
            self.raise_on_check = False

        def capture_character_authority_context(
            self, *, expected_server_id: str
        ) -> object:
            assert expected_server_id == "server-a"
            return self.capture

        def is_character_authority_context_current(self, capture: object) -> bool:
            if self.raise_on_check:
                raise RuntimeError("sensitive authority detail")
            return capture is self.capture and self.current

        async def resolve_character_authority_id(
            self,
            *,
            expected_server_id: str,
            context_capture: object,
        ) -> str:
            assert expected_server_id == "server-a"
            assert context_capture is self.capture
            started.set()
            await release.wait()
            return "server-user-v1:" + ("a" * 64)

    provider = _ServerAuthorityProvider()
    mock_app_instance.runtime_backend = "server"
    mock_app_instance.active_server_id = "server-a"
    mock_app_instance.server_context_provider = provider
    mock_app_instance._ensure_tts_profile_service = AsyncMock(return_value=service)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        screen.state.select_entity(
            entity_kind="character",
            entity_id="1",
            entity_name="Detective Sam",
        )
        screen._selected_server_character = (
            "server-a",
            dict(CHARACTERS[0]),
        )
        screen._character_tts_request_generation += 1
        request_generation = screen._character_tts_request_generation
        task = asyncio.create_task(
            screen._character_tts_refresh_worker(
                request_generation,
                "1",
                "server",
            )
        )
        await wait_for_background_signal(
            started, task, what="the character TTS refresh worker"
        )
        if final_authority_check == "error":
            provider.raise_on_check = True
        else:
            provider.current = False
            mock_app_instance.active_server_id = "server-b"
        release.set()
        await task

        assert service.get_calls == []
        assert screen._character_tts_snapshot is None
        assert screen._character_tts_presentation.controls_enabled is False
        assert (
            screen._character_tts_presentation.status == "Save/reopen before assigning."
        )


async def test_character_tts_local_authority_change_rejects_late_population(
    mock_app_instance,
    stub_characters,
    monkeypatch,
) -> None:
    # F-031: empty library - see the population-generation test above.
    monkeypatch.setattr(character_handler_module, "fetch_all_characters", lambda: [])
    profile = _character_tts_profile(1)
    original_ref = CharacterRef(
        source="local",
        authority_id="local-authority-before-restore",
        character_id="1",
    )
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=AssignedTTSProfileSnapshot(
                assignment=CharacterTTSAssignment(
                    character_ref=original_ref,
                    profile_id=profile.profile_id,
                ),
                profile=profile,
            ),
        ),
    )
    authority_reader = Mock(
        side_effect=(
            "local-authority-before-restore",
            "local-authority-after-restore",
        )
    )
    mock_app_instance.runtime_backend = "local"
    mock_app_instance.chachanotes_db = SimpleNamespace(
        get_local_authority_id=authority_reader,
        get_character_card_by_id=lambda _character_id: {
            **CHARACTERS[0],
            "extensions": {},
        },
    )
    mock_app_instance._ensure_tts_profile_service = AsyncMock(return_value=service)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        screen.state.select_entity(
            entity_kind="character",
            entity_id="1",
            entity_name="Detective Sam",
        )
        screen._character_tts_request_generation += 1
        await screen._character_tts_refresh_worker(
            screen._character_tts_request_generation,
            "1",
            "local",
        )

        assert authority_reader.call_count == 2
        assert service.get_calls == [original_ref]
        assert screen._character_tts_snapshot is None
        assert screen._character_tts_presentation.controls_enabled is False


async def test_character_tts_missing_local_authority_disables_without_profile_reads(
    mock_app_instance,
    stub_characters,
) -> None:
    profile = _character_tts_profile(1)
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=None,
        ),
    )
    mock_app_instance.runtime_backend = "local"
    mock_app_instance.chachanotes_db = object()
    mock_app_instance._ensure_tts_profile_service = AsyncMock(return_value=service)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        screen.state.select_entity(
            entity_kind="character",
            entity_id="1",
            entity_name="Detective Sam",
        )
        screen._character_tts_request_generation += 1
        await screen._character_tts_refresh_worker(
            screen._character_tts_request_generation,
            "1",
            "local",
        )

        assert service.get_calls == []
        assert screen._character_tts_snapshot is None
        assert screen._character_tts_presentation.controls_enabled is False
        assert "Save/reopen before assigning" in (
            screen._character_tts_presentation.status
        )


async def test_character_tts_assign_and_detach_use_exact_observed_tokens(
    mock_app_instance,
    stub_characters,
) -> None:
    profile = _character_tts_profile(1)
    character_ref = CharacterRef(
        source="local",
        authority_id="local-test-authority",
        character_id="1",
    )
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=None,
        ),
    )
    _configure_character_tts_app(mock_app_instance, service)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()

        screen.post_message(CharacterTTSActionRequested("assign", profile.profile_id))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        assert service.set_calls == [
            (
                character_ref,
                LoadedTTSProfile(
                    repository_generation=7,
                    profile=profile,
                ),
                None,
            )
        ]

        assignment = CharacterTTSAssignment(
            character_ref=character_ref,
            profile_id=profile.profile_id,
        )
        service.assigned = LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=AssignedTTSProfileSnapshot(
                assignment=assignment,
                profile=profile,
            ),
        )
        screen._queue_character_tts_refresh()
        await pilot.app.workers.wait_for_complete()
        screen.post_message(CharacterTTSActionRequested("assign", None))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        assert service.detach_calls == [(assignment, 7)]


async def test_character_tts_assignment_worker_accepts_unverified_profile(
    mock_app_instance,
    stub_characters,
) -> None:
    """The screen-side assignment worker has its OWN availability gate,
    independent of the widget's (`personas_character_tts_widget.py`). Fixing
    only the widget still silently drops the assignment here -- this pins the
    worker actually calling `set_assignment` for a legacy-provider profile
    classified 'unverified' (task-2450 amendment), not just letting the
    widget's own message through."""

    profile = _character_tts_profile(1)
    character_ref = CharacterRef(
        source="local",
        authority_id="local-test-authority",
        character_id="1",
    )
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=None,
        ),
        availability_state="unverified",
    )
    _configure_character_tts_app(mock_app_instance, service)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()

        screen.post_message(CharacterTTSActionRequested("assign", profile.profile_id))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        assert service.set_calls == [
            (
                character_ref,
                LoadedTTSProfile(
                    repository_generation=7,
                    profile=profile,
                ),
                None,
            )
        ]


@pytest.mark.parametrize(
    ("recovery_action", "expected_status", "forbidden"),
    [
        (
            "refresh",
            (
                "{name} · Unverified · Used by 1 character. Refresh or repair "
                "the profile; the assignment is preserved."
            ),
            (),
        ),
        (
            "none",
            (
                "{name} · No catalog check · Used by 1 character. The exact "
                "selection is used as-is; the assignment is preserved."
            ),
            ("refresh", "retry", "unverified"),
        ),
    ],
)
async def test_character_tts_unverified_status_never_promises_an_impossible_refresh(
    recovery_action: str,
    expected_status: str,
    forbidden: tuple[str, ...],
) -> None:
    """The Roleplay status line must follow the availability's own recovery.

    A legacy-provider profile is permanently "unverified" (no catalog to
    preflight), so telling the user to Refresh names a control that can
    never change the state -- ADR-031. audio.cpp keeps its refresh copy.
    """

    profile = _character_tts_profile(1)
    loaded = LoadedTTSProfile(repository_generation=7, profile=profile)
    assignment = CharacterTTSAssignment(
        character_ref=CharacterRef(
            source="local",
            authority_id="local-test-authority",
            character_id="1",
        ),
        profile_id=profile.profile_id,
    )
    availability = TTSProfileAvailability(
        profile_id=profile.profile_id,
        state="unverified",
        recovery_action=recovery_action,  # type: ignore[arg-type]
    )
    snapshot = personas_screen_module._CharacterTTSControlSnapshot(
        request_generation=1,
        runtime_source="local",
        character_id="1",
        character_ref=assignment.character_ref,
        repository_generation=7,
        loaded_profiles=(loaded,),
        availability=(availability,),
        current=AssignedTTSProfileSnapshot(
            assignment=assignment,
            profile=profile,
        ),
        assignment_count=1,
        configuration_revision=4,
        catalog_revision=None,
        expected_server_id=None,
        server_context_capture=None,
    )

    state = PersonasScreen._character_tts_presentation_from_snapshot(snapshot)

    assert state.status == expected_status.format(name=profile.display_name)
    for word in forbidden:
        assert word not in state.status.casefold()
    # Pin the threading, not just its downstream effect: the status line
    # above is derived from `current_availability.recovery_action`, a local
    # read off `snapshot.availability` -- it says nothing about whether the
    # *option row* for this same profile (`state.profiles[0]`) carries the
    # real value too. personas_screen.py:1753-1756 is the only call site
    # that threads it into `CharacterTTSProfileOption`; the dataclass
    # default ("refresh") would silently paper over a dropped argument
    # there, leaving the Select option reading "· unverified" beside a
    # status line that correctly says "· No catalog check" -- with every
    # other test in this suite (which construct options directly) still
    # green.
    assert state.profiles[0].recovery_action == recovery_action


async def test_character_tts_assignment_worker_still_refuses_unavailable_profile(
    mock_app_instance,
    stub_characters,
) -> None:
    """The complementary pin: a genuinely unavailable profile stays refused
    at the worker layer, unchanged."""

    profile = _character_tts_profile(1)
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=None,
        ),
        availability_state="unavailable",
    )
    _configure_character_tts_app(mock_app_instance, service)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()

        screen.post_message(CharacterTTSActionRequested("assign", profile.profile_id))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        assert service.set_calls == []


async def test_character_tts_preview_create_and_edit_reuse_existing_speech_surfaces(
    mock_app_instance,
    stub_characters,
) -> None:
    profile = _character_tts_profile(1)
    character_ref = CharacterRef(
        source="local",
        authority_id="local-test-authority",
        character_id="1",
    )
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=AssignedTTSProfileSnapshot(
                assignment=CharacterTTSAssignment(
                    character_ref=character_ref,
                    profile_id=profile.profile_id,
                ),
                profile=profile,
            ),
        ),
        dependency=TTSProfileDependencyProjection(
            reason="recipe_missing",
            display="Needs compatible model",
            action="open_audio_cpp_settings",
            advisory="recipe_provenance_unavailable",
            advisory_display="Recipe provenance unavailable",
            advisory_action="generate_new_profile",
        ),
    )
    _configure_character_tts_app(mock_app_instance, service)
    app = _NavCaptureApp(mock_app_instance)

    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()

        character_tts = screen.query_one(
            "#personas-character-card-tts", PersonasCharacterTTSWidget
        )
        action_area = character_tts.query_one(".personas-character-tts-actions")
        blocker = character_tts.query_one(
            ".personas-character-tts-dependency-primary", Button
        )
        advisory = character_tts.query_one(
            ".personas-character-tts-dependency-advisory", Button
        )
        character_tts.query_one(
            ".personas-character-tts-recovery-actions"
        ).scroll_visible()
        await pilot.pause()
        assert character_tts.region.width <= 40
        for button in (blocker, advisory):
            assert button.display is True
            assert button.region.width == action_area.region.width
            assert action_area.region.contains_region(button.region), str(button.label)

        screen.post_message(CharacterTTSActionRequested("preview", profile.profile_id))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert app.nav_routes[-1] == "stts"
        preset = app.nav_contexts[-1]["profile_preset"]
        assert type(preset) is TTSPlaygroundSelectionPreset
        assert preset.model_id == profile.model_id
        assert preset.voice_id == profile.voice_id

        blocker.press()
        await pilot.pause()
        assert app.nav_routes[-1] == "settings"
        assert app.nav_contexts[-1]["category"] == "speech-tts"

        advisory.press()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        assert app.nav_routes[-1] == "stts"
        recovery_preset = app.nav_contexts[-1]["profile_preset"]
        assert type(recovery_preset) is TTSPlaygroundSelectionPreset
        assert recovery_preset.model_id == profile.model_id

        snapshot = screen._character_tts_snapshot
        assert snapshot is not None
        pending = TTSProfileDependencyProjection(
            reason="recipe_pending_apply",
            display="Compatible model saved; apply settings",
            action="open_speech_lab_apply",
        )
        screen._character_tts_snapshot = replace(
            snapshot,
            availability=tuple(
                replace(item, dependency=pending) for item in snapshot.availability
            ),
        )
        for operation in ("open_speech_lab_apply",):
            screen.post_message(
                CharacterTTSActionRequested(operation, profile.profile_id)
            )
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            assert app.nav_routes[-1] == "stts"
            recovery_preset = app.nav_contexts[-1]["profile_preset"]
            assert type(recovery_preset) is TTSPlaygroundSelectionPreset
            assert recovery_preset.model_id == profile.model_id
            assert recovery_preset.voice_id == profile.voice_id
        assert service.set_calls == []
        assert service.detach_calls == []

        screen.post_message(CharacterTTSActionRequested("create", None))
        await pilot.pause()
        assert app.nav_contexts[-1] == {"view": "playground"}

        draft = TTSProfileDraft(
            display_name="Edited roleplay voice",
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
        )
        pilot.app.push_screen_wait = AsyncMock(return_value=draft)
        screen.post_message(CharacterTTSActionRequested("edit", profile.profile_id))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        assert service.update_calls == [
            (
                LoadedTTSProfile(
                    repository_generation=7,
                    profile=profile,
                ),
                draft,
            )
        ]


async def test_character_tts_preview_rechecks_local_authority(
    mock_app_instance,
    stub_characters,
) -> None:
    profile = _character_tts_profile(1)
    character_ref = CharacterRef(
        source="local",
        authority_id="local-test-authority",
        character_id="1",
    )
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=AssignedTTSProfileSnapshot(
                assignment=CharacterTTSAssignment(
                    character_ref=character_ref,
                    profile_id=profile.profile_id,
                ),
                profile=profile,
            ),
        ),
    )
    _configure_character_tts_app(mock_app_instance, service)
    authority_reader = Mock(return_value="local-test-authority")
    mock_app_instance.chachanotes_db.get_local_authority_id = authority_reader
    app = _NavCaptureApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()
        assert screen._character_tts_snapshot is not None

        authority_reader.return_value = "different-local-authority"
        screen.post_message(CharacterTTSActionRequested("preview", profile.profile_id))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()

        assert app.nav_contexts == []


async def test_character_tts_conflict_refreshes_and_stale_selection_cannot_publish(
    mock_app_instance,
    stub_characters,
) -> None:
    profile = _character_tts_profile(1)
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=None,
        ),
    )
    service.set_error = ProfileRepositoryError("conflict")
    _configure_character_tts_app(mock_app_instance, service)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()
        reads_before = len(service.get_calls)

        screen.post_message(CharacterTTSActionRequested("assign", profile.profile_id))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        assert len(service.get_calls) > reads_before

        request_generation = screen._character_tts_request_generation
        screen.state.select_entity(
            entity_kind="character",
            entity_id="2",
            entity_name="Lab Assistant",
        )
        assert (
            screen._character_tts_request_is_current(
                request_generation,
                "1",
                "local",
            )
            is False
        )


async def test_character_soft_delete_never_detaches_tts_assignment(
    mock_app_instance,
    stub_characters,
    monkeypatch,
) -> None:
    profile = _character_tts_profile(1)
    character_ref = CharacterRef(
        source="local",
        authority_id="local-test-authority",
        character_id="1",
    )
    assignment = CharacterTTSAssignment(
        character_ref=character_ref,
        profile_id=profile.profile_id,
    )
    service = _CharacterTTSProfileService(
        page=TTSProfilePageSnapshot(
            repository_generation=7,
            profiles=(profile,),
            total=1,
        ),
        assigned=LoadedCharacterTTSAssignment(
            repository_generation=7,
            snapshot=AssignedTTSProfileSnapshot(
                assignment=assignment,
                profile=profile,
            ),
        ),
        availability_state="unverified",
    )
    _configure_character_tts_app(mock_app_instance, service)
    monkeypatch.setattr(character_handler_module, "delete_character", lambda *_: True)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.app.workers.wait_for_complete()
        await screen._select_character("1", "Detective Sam")
        await pilot.app.workers.wait_for_complete()
        await screen._delete_entity("character", "1", 1)
        await pilot.app.workers.wait_for_complete()

        assert service.detach_calls == []


def _configure_persona_buddy(
    mock_app_instance,
    records: dict[str, dict],
    *,
    preferences: PersonaBuddyPreferences | None = None,
) -> PersonaBuddyController:
    def local_record(persona_id: str):
        record = records.get(str(persona_id))
        return dict(record) if record is not None else None

    async def scoped_record(persona_id: str, *, mode: str):
        assert mode == "local"
        record = local_record(persona_id)
        if record is None:
            raise ValueError("persona missing")
        return record

    scope = SimpleNamespace(
        local_service=SimpleNamespace(get_persona_profile=local_record),
        list_persona_profiles=AsyncMock(
            return_value={
                "items": [dict(item) for item in records.values()],
                "total": len(records),
            }
        ),
        get_persona_profile=AsyncMock(side_effect=scoped_record),
        delete_persona_profile=AsyncMock(
            return_value={"status": "deleted", "persona_id": "p-1"}
        ),
    )
    controller = PersonaBuddyController(
        preferences=preferences,
        local_persona_service=scope.local_service,
        preference_writer=lambda _preferences: True,
    )
    mock_app_instance.runtime_backend = "local"
    mock_app_instance.character_persona_scope_service = scope
    mock_app_instance.persona_buddy_controller = controller
    mock_app_instance.reconcile_persona_buddy_view = AsyncMock(return_value=True)
    return controller


async def test_workbench_highlight_never_retargets_buddy(
    mock_app_instance,
    stub_characters,
) -> None:
    records = {
        "p-1": {**PROFILE, "version": 2, "is_active": True, "deleted": False},
        "p-2": {
            **PROFILE,
            "id": "p-2",
            "name": "Navigator",
            "version": 5,
            "is_active": True,
            "deleted": False,
        },
    }
    controller = _configure_persona_buddy(
        mock_app_instance,
        records,
        preferences=PersonaBuddyPreferences(
            enabled=True,
            selection=PersonaBuddySelection("local", "p-1"),
        ),
    )
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("p-2", "Navigator")
        await pilot.pause()

        assert controller.snapshot().selection == PersonaBuddySelection("local", "p-1")
        mock_app_instance.reconcile_persona_buddy_view.assert_not_awaited()


async def test_floating_buddy_close_refreshes_active_personas_inspector(
    mock_app_instance,
    stub_characters,
) -> None:
    records = {"p-1": {**PROFILE, "version": 2, "is_active": True, "deleted": False}}
    controller = _configure_persona_buddy(
        mock_app_instance,
        records,
        preferences=PersonaBuddyPreferences(
            enabled=True,
            open=True,
            selection=PersonaBuddySelection("local", "p-1"),
        ),
    )
    persisted: list[PersonaBuddyPreferences] = []
    controller._preference_writer = lambda preferences: (
        persisted.append(preferences) or True
    )

    async def unresolved_until_closed(*, cols: int, lines: int):
        return None

    controller.resolve_current_visual = unresolved_until_closed
    app = PersonaBuddyWorkbenchApp(mock_app_instance)

    async with app.run_test(size=(100, 30)) as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("p-1", "Archivist")
        await app.reconcile_persona_buddy_view()
        await pilot.pause()

        assert screen.query_one(PersonaBuddyWidget).is_attached
        assert screen.query_one("#personas-buddy-close", Button).disabled is False
        assert screen.query_one("#personas-buddy-show", Button).disabled is True

        await pilot.click("#persona-buddy-close")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert controller.current_preferences().open is False
        assert persisted[-1].open is False
        assert not list(screen.query(PersonaBuddyWidget))
        show = screen.query_one("#personas-buddy-show", Button)
        close = screen.query_one("#personas-buddy-close", Button)
        assert show.disabled is False
        assert close.disabled is True
        assert close.tooltip == "Buddy is already closed."


async def test_stale_personas_screen_reconcile_skips_screen_local_buddy_hook(
    mock_app_instance,
    stub_characters,
) -> None:
    _configure_persona_buddy(
        mock_app_instance,
        {},
        preferences=PersonaBuddyPreferences(open=False),
    )
    app = PersonaBuddyWorkbenchApp(mock_app_instance)

    async with app.run_test(size=(100, 30)) as pilot:
        stale = await _mounted(pilot)
        await app.switch_screen(PersonasScreen(app))
        hook = Mock(wraps=stale.sync_persona_buddy_reconciled_state)
        stale.sync_persona_buddy_reconciled_state = hook

        await stale.reconcile_persona_buddy_view()

        hook.assert_not_called()


@pytest.mark.parametrize("compact", (False, True), ids=("normal", "compact"))
async def test_real_80x24_workbench_scrolls_each_buddy_action_into_view_and_runs_it(
    mock_app_instance,
    stub_characters,
    compact: bool,
) -> None:
    records = {"p-1": {**PROFILE, "version": 2, "is_active": True, "deleted": False}}
    controller = _configure_persona_buddy(
        mock_app_instance,
        records,
        preferences=PersonaBuddyPreferences(
            enabled=True,
            open=True,
            selection=PersonaBuddySelection("local", "p-1"),
        ),
    )
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("p-1", "Archivist")
        workbench = screen.query_one("#personas-workbench")
        workbench.set_class(compact, "personas-workbench-compact")
        for pane_id in (
            "#personas-library-pane",
            "#personas-work-area",
            "#personas-inspector-pane",
        ):
            screen.query_one(pane_id).set_class(
                compact, "personas-workbench-compact-pane"
            )
        await pilot.pause()

        inspector = screen.query_one("#personas-inspector-pane")
        expectations = (
            ("#personas-buddy-close", "Close Buddy", True, False),
            ("#personas-buddy-show", "Show Buddy", True, True),
            ("#personas-buddy-disable", "Disable Buddy", False, True),
            ("#personas-buddy-use", "Use for Buddy", True, True),
        )
        for button_id, label, enabled, opened in expectations:
            button = screen.query_one(button_id, Button)
            assert str(button.label) == label
            assert button.disabled is False
            button.focus(scroll_visible=True)
            await pilot.pause(0.5)

            assert pilot.app.focused is button
            assert button.region.y >= max(0, inspector.content_region.y)
            assert button.region.bottom <= min(24, inspector.content_region.bottom)

            await pilot.press("enter")
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            preferences = controller.current_preferences()
            assert preferences.enabled is enabled
            assert preferences.open is opened


async def test_explicit_replacement_is_required(
    mock_app_instance,
    stub_characters,
) -> None:
    records = {
        "p-1": {**PROFILE, "version": 2, "is_active": True, "deleted": False},
        "p-2": {
            **PROFILE,
            "id": "p-2",
            "name": "Navigator",
            "version": 5,
            "is_active": True,
            "deleted": False,
        },
    }
    controller = _configure_persona_buddy(
        mock_app_instance,
        records,
        preferences=PersonaBuddyPreferences(
            enabled=True,
            selection=PersonaBuddySelection("local", "p-1"),
        ),
    )
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("p-2", "Navigator")
        await pilot.pause()
        assert not screen.query_one("#personas-buddy-use", Button).disabled
        for button_id in (
            "#personas-buddy-show",
            "#personas-buddy-close",
            "#personas-buddy-disable",
        ):
            button = screen.query_one(button_id, Button)
            assert button.disabled is True
            assert button.tooltip == "Select the Persona currently used by Buddy"

        await screen._select_profile("p-1", "Archivist")
        assert screen.query_one("#personas-buddy-use", Button).disabled is False
        assert screen.query_one("#personas-buddy-close", Button).disabled is False
        assert screen.query_one("#personas-buddy-disable", Button).disabled is False
        show = screen.query_one("#personas-buddy-show", Button)
        assert show.disabled is True
        assert show.tooltip == "Buddy is already open."

        await screen._select_profile("p-2", "Navigator")

        screen.post_message(
            PersonaBuddyActionRequested(
                action="show", source="local", persona_id="p-2", revision=5
            )
        )
        await pilot.pause()
        assert controller.snapshot().selection == PersonaBuddySelection("local", "p-1")
        mock_app_instance.reconcile_persona_buddy_view.assert_not_awaited()

        screen.post_message(
            PersonaBuddyActionRequested(
                action="use", source="local", persona_id="p-2", revision=5
            )
        )
        await pilot.pause()

        snapshot = controller.snapshot()
        assert snapshot.selection == PersonaBuddySelection("local", "p-2")
        assert snapshot.enabled is True
        assert snapshot.open is True
        mock_app_instance.reconcile_persona_buddy_view.assert_awaited_once()


@pytest.mark.parametrize(
    ("action", "expected_enabled", "expected_open"),
    (
        ("show", True, True),
        ("close", True, False),
        ("disable", False, True),
    ),
)
async def test_buddy_visibility_actions_preserve_explicit_selection(
    mock_app_instance,
    stub_characters,
    action: str,
    expected_enabled: bool,
    expected_open: bool,
) -> None:
    record = {**PROFILE, "version": 2, "is_active": True, "deleted": False}
    controller = _configure_persona_buddy(
        mock_app_instance,
        {"p-1": record},
        preferences=PersonaBuddyPreferences(
            enabled=action != "show",
            open=action != "show",
            selection=PersonaBuddySelection("local", "p-1"),
        ),
    )
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("p-1", "Archivist")
        screen.post_message(
            PersonaBuddyActionRequested(
                action=action, source="local", persona_id="p-1", revision=2
            )
        )
        await pilot.pause()

        preferences = controller.current_preferences()
        assert preferences.selection == PersonaBuddySelection("local", "p-1")
        assert preferences.enabled is expected_enabled
        assert preferences.open is expected_open


@pytest.mark.parametrize("failure", ("false", "raise", "cancel"))
async def test_buddy_action_writer_failure_leaves_memory_and_durable_state_unchanged(
    mock_app_instance,
    stub_characters,
    failure: str,
) -> None:
    record = {**PROFILE, "version": 2, "is_active": True, "deleted": False}
    initial = PersonaBuddyPreferences(
        enabled=True,
        open=True,
        selection=PersonaBuddySelection("local", "p-1"),
    )
    controller = _configure_persona_buddy(
        mock_app_instance,
        {"p-1": record},
        preferences=initial,
    )
    durable = initial

    def writer(preferences: PersonaBuddyPreferences) -> bool:
        nonlocal durable
        if failure == "false":
            return False
        if failure == "raise":
            raise RuntimeError("writer failed")
        raise asyncio.CancelledError

    controller._preference_writer = writer
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("p-1", "Archivist")
        screen.post_message(
            PersonaBuddyActionRequested(
                action="close", source="local", persona_id="p-1", revision=2
            )
        )
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert durable == initial
        assert controller.current_preferences() == initial
        assert screen.state.has_unsaved_changes is False
        mock_app_instance.reconcile_persona_buddy_view.assert_not_awaited()


async def test_buddy_action_persists_before_applying_memory_or_reconciling(
    mock_app_instance,
    stub_characters,
) -> None:
    record = {**PROFILE, "version": 2, "is_active": True, "deleted": False}
    initial = PersonaBuddyPreferences(
        enabled=True,
        open=True,
        selection=PersonaBuddySelection("local", "p-1"),
    )
    controller = _configure_persona_buddy(
        mock_app_instance,
        {"p-1": record},
        preferences=initial,
    )
    entered = threading.Event()
    release = threading.Event()
    durable = initial

    def writer(preferences: PersonaBuddyPreferences) -> bool:
        nonlocal durable
        entered.set()
        release.wait(timeout=5)
        durable = preferences
        return True

    controller._preference_writer = writer
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("p-1", "Archivist")
        screen.post_message(
            PersonaBuddyActionRequested(
                action="close", source="local", persona_id="p-1", revision=2
            )
        )
        assert await asyncio.to_thread(entered.wait, 2)

        assert durable == initial
        assert controller.current_preferences() == initial
        mock_app_instance.reconcile_persona_buddy_view.assert_not_awaited()

        release.set()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert durable.open is False
        assert controller.current_preferences() == durable
        mock_app_instance.reconcile_persona_buddy_view.assert_awaited_once()


async def test_disabled_deleted_missing_persona_hides_but_preserves_enabled_selection(
    mock_app_instance,
    stub_characters,
) -> None:
    record = {**PROFILE, "version": 2, "is_active": True, "deleted": False}
    records = {"p-1": record}
    controller = _configure_persona_buddy(
        mock_app_instance,
        records,
        preferences=PersonaBuddyPreferences(
            enabled=True,
            selection=PersonaBuddySelection("local", "p-1"),
        ),
    )
    resolved = []

    async def reconcile():
        resolved.append(await controller.resolve_current_visual(cols=80, lines=24))
        return True

    mock_app_instance.reconcile_persona_buddy_view.side_effect = reconcile
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        before = controller.snapshot().profile_generation

        for unavailable_record in (
            {**record, "is_active": False},
            {**record, "deleted": True},
            None,
        ):
            if unavailable_record is None:
                records.pop("p-1")
            else:
                records["p-1"] = unavailable_record
            await screen._refresh_persona_buddy_lifecycle("p-1")

        snapshot = controller.snapshot()
        assert snapshot.profile_generation == before + 3
        assert snapshot.enabled is True
        assert snapshot.selection == PersonaBuddySelection("local", "p-1")
        assert mock_app_instance.reconcile_persona_buddy_view.await_count == 3
        assert [visual.available for visual in resolved] == [False, False, False]
        assert {visual.reason for visual in resolved} == {
            "persona_buddy_persona_unavailable"
        }


async def test_restore_reresolves_same_selection(
    mock_app_instance,
    stub_characters,
) -> None:
    record = {**PROFILE, "version": 3, "is_active": False, "deleted": False}
    records = {"p-1": record}
    controller = _configure_persona_buddy(
        mock_app_instance,
        records,
        preferences=PersonaBuddyPreferences(
            enabled=True,
            open=True,
            collapsed=True,
            selection=PersonaBuddySelection("local", "p-1"),
        ),
    )
    resolved = []

    async def reconcile():
        resolved.append(await controller.resolve_current_visual(cols=80, lines=24))
        return True

    mock_app_instance.reconcile_persona_buddy_view.side_effect = reconcile
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        before = controller.snapshot().profile_generation
        await screen._refresh_persona_buddy_lifecycle("p-1")
        records["p-1"] = {
            **record,
            "version": 4,
            "is_active": True,
            "deleted": False,
        }
        await screen._refresh_persona_buddy_lifecycle("p-1")

        preferences = controller.current_preferences()
        assert preferences.selection == PersonaBuddySelection("local", "p-1")
        assert preferences.enabled is True
        assert preferences.open is True
        assert preferences.collapsed is True
        assert controller.snapshot().profile_generation == before + 2
        assert [visual.reason for visual in resolved] == [
            "persona_buddy_persona_unavailable",
            "persona_buddy_binding_unavailable",
        ]


async def test_buddy_action_fetch_aba_cannot_apply_or_reconcile(
    mock_app_instance,
    stub_characters,
) -> None:
    records = {
        "p-1": {**PROFILE, "version": 2, "is_active": True, "deleted": False},
        "p-2": {
            **PROFILE,
            "id": "p-2",
            "name": "Navigator",
            "version": 5,
            "is_active": True,
            "deleted": False,
        },
    }
    controller = _configure_persona_buddy(
        mock_app_instance,
        records,
        preferences=PersonaBuddyPreferences(
            enabled=True,
            selection=PersonaBuddySelection("local", "p-1"),
        ),
    )
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("p-1", "Archivist")
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def fetch(persona_id: str, *, mode: str):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
            return dict(records[persona_id])

        mock_app_instance.character_persona_scope_service.get_persona_profile = fetch
        screen.post_message(
            PersonaBuddyActionRequested(
                action="close", source="local", persona_id="p-1", revision=2
            )
        )
        await wait_for_signal(started, what="Buddy action fetch start")
        await screen._select_profile("p-2", "Navigator")
        await screen._select_profile("p-1", "Archivist")
        release.set()
        await pilot.pause()

        assert controller.current_preferences().open is True
        mock_app_instance.reconcile_persona_buddy_view.assert_not_awaited()


async def test_incomplete_profile_fetch_disables_cached_buddy_actions(
    mock_app_instance,
    stub_characters,
) -> None:
    records = {"p-1": {**PROFILE, "version": 2, "is_active": True, "deleted": False}}
    _configure_persona_buddy(mock_app_instance, records)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        mock_app_instance.character_persona_scope_service.get_persona_profile = (
            AsyncMock(side_effect=RuntimeError("service unavailable"))
        )
        await screen._select_profile("p-1", "Archivist")

        for button in screen.query(".persona-buddy-action").results(Button):
            assert button.disabled is True
            assert (
                button.tooltip
                == "Persona details are unavailable. Refresh and try again."
            )


async def test_local_save_and_delete_refresh_only_the_same_buddy_selection(
    monkeypatch,
    mock_app_instance,
    stub_characters,
) -> None:
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        mock_app_instance.character_persona_scope_service.delete_persona_profile = (
            AsyncMock()
        )
        refresh = AsyncMock()
        monkeypatch.setattr(screen, "_refresh_persona_buddy_lifecycle", refresh)
        monkeypatch.setattr(screen, "_after_delete", AsyncMock())
        monkeypatch.setattr(
            screen.persona_handler,
            "refresh_persona_list",
            AsyncMock(return_value=[]),
        )

        screen.state.active_mode = "characters"
        await screen._after_profile_save(
            {**PROFILE, "id": "p-1", "version": 3}, source="local"
        )
        await screen._delete_entity("persona", "p-1", 3)

        assert refresh.await_args_list == [call("p-1"), call("p-1")]


async def test_server_profile_durable_changes_never_refresh_local_buddy(
    monkeypatch,
    mock_app_instance,
    stub_characters,
) -> None:
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        mock_app_instance.character_persona_scope_service.delete_persona_profile = (
            AsyncMock()
        )
        refresh = AsyncMock()
        monkeypatch.setattr(screen, "_refresh_persona_buddy_lifecycle", refresh)
        monkeypatch.setattr(screen, "_after_delete", AsyncMock())
        monkeypatch.setattr(
            screen.persona_handler,
            "refresh_persona_list",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(screen.persona_handler, "current_mode", lambda: "server")

        screen.state.active_mode = "characters"
        await screen._after_profile_save(
            {**PROFILE, "id": "p-1", "version": 3}, source="server"
        )
        await screen._delete_entity("persona", "p-1", 3)

        refresh.assert_not_awaited()


async def test_stale_buddy_action_does_not_refresh_replaced_workbench_selection(
    mock_app_instance,
    stub_characters,
) -> None:
    records = {
        "p-1": {**PROFILE, "version": 2, "is_active": True, "deleted": False},
        "p-2": {
            **PROFILE,
            "id": "p-2",
            "name": "Navigator",
            "version": 5,
            "is_active": True,
            "deleted": False,
        },
    }
    started = threading.Event()
    release = threading.Event()

    def blocked_writer(_preferences):
        started.set()
        release.wait(timeout=5)
        return True

    controller = _configure_persona_buddy(
        mock_app_instance,
        records,
        preferences=PersonaBuddyPreferences(
            enabled=True,
            selection=PersonaBuddySelection("local", "p-1"),
        ),
    )
    controller._preference_writer = blocked_writer
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("p-1", "Archivist")
        screen.post_message(
            PersonaBuddyActionRequested(
                action="close", source="local", persona_id="p-1", revision=2
            )
        )
        while not started.is_set():
            await asyncio.sleep(0)
        assert controller.current_preferences().open is True
        await screen._select_profile("p-2", "Navigator")
        await screen._select_profile("p-1", "Archivist")
        release.set()
        await pilot.pause()

        assert controller.current_preferences().open is True
        mock_app_instance.reconcile_persona_buddy_view.assert_not_awaited()


async def test_newer_buddy_action_wins_serialized_persistence_and_reconcile(
    mock_app_instance,
    stub_characters,
) -> None:
    records = {"p-1": {**PROFILE, "version": 2, "is_active": True, "deleted": False}}
    started = threading.Event()
    release = threading.Event()
    writes = []

    def blocked_first_writer(preferences):
        writes.append(preferences)
        if len(writes) == 1:
            started.set()
            release.wait(timeout=5)
        return True

    controller = _configure_persona_buddy(
        mock_app_instance,
        records,
        preferences=PersonaBuddyPreferences(
            enabled=True,
            selection=PersonaBuddySelection("local", "p-1"),
        ),
    )
    controller._preference_writer = blocked_first_writer
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("p-1", "Archivist")
        screen.post_message(
            PersonaBuddyActionRequested(
                action="close", source="local", persona_id="p-1", revision=2
            )
        )
        while not started.is_set():
            await asyncio.sleep(0)
        screen.post_message(
            PersonaBuddyActionRequested(
                action="show", source="local", persona_id="p-1", revision=2
            )
        )
        await pilot.pause()
        release.set()
        await pilot.pause()

        assert [preferences.open for preferences in writes] == [False, True]
        assert controller.current_preferences().open is True
        mock_app_instance.reconcile_persona_buddy_view.assert_awaited_once()


async def test_persona_json_export_excludes_buddy_preferences(
    mock_app_instance,
    stub_characters,
    tmp_path,
) -> None:
    record = {**PROFILE, "version": 2, "is_active": True, "deleted": False}
    _configure_persona_buddy(mock_app_instance, {"p-1": record})
    mock_app_instance.app_config = {
        "persona_buddy": {
            "enabled": True,
            "source": "local",
            "local_persona_id": "p-1",
            "open": False,
            "collapsed": True,
            "x": 17,
            "y": 9,
            "width": 42,
            "height": 14,
        }
    }
    app = PersonasTestApp(mock_app_instance)
    target = tmp_path / "persona.json"

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await screen._apply_mode("personas")
        await screen._select_profile("p-1", "Archivist")
        await screen._export_selected_character(str(target), fmt="json")

    exported = json.loads(target.read_text(encoding="utf-8"))
    assert exported == record
    assert "persona_buddy" not in exported
