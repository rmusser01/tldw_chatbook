"""TldwCli's command-palette providers and their key-display helpers.

Moved verbatim from ``app.py`` (TASK-33011, PR-H): ``ThemeProvider``, ``TabNavigationProvider``,
``QuickActionsProvider``, ``SettingsProvider``, ``CharacterProvider``, ``MediaProvider``,
``LibraryIngestProvider``, ``SetupWizardProvider``, ``PatternGalleryProvider`` and
``DeveloperProvider``, with ``_navigate_via_screen``, ``_display_key``, ``_bindings_to_shortcuts``
and the ``FOCUS_TOGGLE_PALETTE_ENTRY`` command. ``TldwCli.COMMANDS`` names the classes, so
``tldw_chatbook.app`` imports them eagerly and re-exports them.

Patch the names this code reads (``ALL_THEMES`` and so on) HERE: the classes resolve free names
through this module's globals, so a patch on ``tldw_chatbook.app`` alone no longer reaches them.
``Tests/Architecture/test_app_extracted_patch_targets.py`` fails on an app-module patch that can
only have been meant for code that moved out.
"""

from dataclasses import replace
from functools import partial
from typing import Any

from loguru import logger
from textual.app import App
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.content import Content
from textual.keys import KEY_DISPLAY_ALIASES

from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_INGEST,
    LIBRARY_NAV_CONTEXT_NOTES_CREATE,
    TAB_ACP,
    TAB_ARTIFACTS,
    TAB_CCP,
    TAB_CHAT,
    TAB_CHATBOOKS,
    TAB_EVALS,
    TAB_HOME,
    TAB_INGEST,
    TAB_LIBRARY,
    TAB_LLM,
    TAB_LOGS,
    TAB_MCP,
    TAB_MEDIA,
    TAB_MEETINGS,
    TAB_PERSONAS,
    TAB_RESEARCH,
    TAB_RESEARCH_WORKSPACE,
    TAB_SCHEDULES,
    TAB_SEARCH,
    TAB_SETTINGS,
    TAB_SKILLS,
    TAB_STATS,
    TAB_STTS,
    TAB_STUDY,
    TAB_TOOLS_SETTINGS,
    TAB_WATCHLISTS_COLLECTIONS,
    TAB_WORKFLOWS,
    TAB_WRITING,
    get_tab_display_label,
)
from tldw_chatbook.css.Themes.themes import ALL_THEMES, printable
from tldw_chatbook.Utils.input_validation import escape_markup

from .config import get_cli_config_path
from .UI.Navigation.main_navigation import NavigateToScreen
from .UI.Navigation.shell_destinations import (
    SHELL_DESTINATION_ORDER,
    get_shell_destination,
)
from .UI.Workbench.help import WorkbenchHelpPanel, WorkbenchHelpState


class ThemeProvider(Provider):
    """A command provider for theme switching."""

    def __init__(self, screen, *args, **kwargs):
        """Initialize the ThemeProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        """Search for theme commands."""
        matcher = self.matcher(query)

        # Always show the main "Change Theme" command
        main_command_score = matcher.match("Theme: Change Theme")
        if main_command_score > 0:
            yield Hit(
                main_command_score,
                matcher.highlight("Theme: Change Theme"),
                partial(self.show_theme_submenu),
                help="Open theme selection menu",
            )

        # The two Textual built-ins, the shipped ALL_THEMES catalog, then any
        # other registered theme -- the user's saved themes (TASK-31250).
        # custom_<name> is the editor's process-only Apply registration;
        # switching persists the default, and that name would not exist at
        # the next launch.
        available_themes = list(
            dict.fromkeys(
                [
                    "textual-dark",
                    "textual-light",
                    *(
                        theme.name if hasattr(theme, "name") else str(theme)
                        for theme in ALL_THEMES
                    ),
                    *(
                        name
                        for name in getattr(self.app, "available_themes", {})
                        if not str(name).startswith("custom_")
                    ),
                ]
            )
        )

        # Only show individual themes if user is specifically searching for
        # theme-related terms or (part of) a registered theme's name
        query_lower = query.lower().strip()
        keyword_match = any(
            term in query_lower
            for term in [
                "switch",
                "theme",
                "dark",
                "light",
                "color",
                "solarized",
                "gruvbox",
                "dracula",
            ]
        )
        name_match = bool(query_lower) and any(
            query_lower in name or query_lower in name.replace("_", " ").replace("-", " ")
            for name in available_themes
        )
        if keyword_match or name_match:
            for theme_name in available_themes:
                command_text = f"Theme: Switch to {theme_name.replace('_', ' ').replace('-', ' ').title()}"
                score = matcher.match(command_text)
                if score > 0:
                    # Both the display and the help parse markup; a saved
                    # theme's name is untrusted file text (R28). ponytail: a
                    # name with "[" shows unhighlighted -- highlight offsets
                    # would not line up with an escaped candidate.
                    display = (
                        Content(command_text)
                        if "[" in command_text
                        else matcher.highlight(command_text)
                    )
                    yield Hit(
                        score * 0.9,  # Slightly lower priority than main command
                        display,
                        partial(self.switch_theme, theme_name),
                        help=f"Change theme to {escape_markup(theme_name)}",
                    )

    async def discover(self) -> Hits:
        """Show only the main theme command when palette is first opened."""
        yield Hit(
            1.0,
            "Theme: Change Theme",
            partial(self.show_theme_submenu),
            help="Open theme selection menu",
        )

    def show_theme_submenu(self) -> None:
        """Show a notification with instruction to search for themes."""
        self.app.notify(
            "Type 'theme' in the command palette to see all available themes",
            severity="information",
        )

    def switch_theme(self, theme_name: str) -> None:
        """Switch to the specified theme and keep it as the launch default.

        TASK-33243: the switch shows now; the launch-default write (~140 ms)
        is queued and awaited off the UI thread in ``_persist_switch``,
        sharing the picker's numbered write queue (``ThemePicker._persist_
        use``) so ordering with a picker Use/Revert holds and quit waits
        for it (``wait_for_theme_quit_work``).

        Args:
            theme_name: The registered theme to show and keep, e.g. ``"nord"``.
        """
        from .css.Themes.theme_catalog import queue_launch_default, use_theme

        try:
            change = use_theme(self.app, theme_name, persist=False)
        except Exception as e:  # noqa: BLE001 - palette commands must not raise
            self.app.notify(f"Failed to apply theme: {escape_markup(e)}", severity="error")
            return
        change = replace(change, persisted=True)
        write = queue_launch_default(theme_name)
        self.app.run_worker(self._persist_switch(change, write), group="settings-theme-use", exit_on_error=False)

    async def _persist_switch(self, change: "ThemeChange", write: "QueuedLaunchDefault") -> None:
        """Await the palette's launch-default write, then show the toast.

        Mirrors ``ThemePicker._persist_use``: only the latest queued write
        (review M-1) reports -- a later picker Use or Revert supersedes it.

        Args:
            change: The switch, for the toast's "was:" name.
            write: The queued write of the theme the palette switched to.
        """
        from .css.Themes.theme_catalog import settle_launch_default, use_theme_toast

        theme_name = write.name
        try:
            persisted, caches_reloaded, latest = await settle_launch_default(self.app, write)
        except Exception as exc:  # noqa: BLE001 - reported as "not saved" below
            logger.warning(f"Saving the launch default {printable(theme_name)!r} failed: {type(exc).__name__}")
            persisted, caches_reloaded, latest = False, False, True
        if not latest:
            return
        outcome = replace(change, persisted=persisted, caches_reloaded=caches_reloaded)
        message, severity = use_theme_toast(theme_name, outcome)
        self.app.notify(message, severity=severity)


def _navigate_via_screen(
    app: App,
    route: str,
    success_message: str,
    screen_context: dict[str, object] | None = None,
) -> None:
    """Navigate through the screen router so palette commands work in shell mode."""
    app.post_message(NavigateToScreen(route, screen_context))
    app.notify(success_message, severity="information")


#: TASK-32887: spelled-out punctuation key names that appear in this
#: repo's BINDINGS (left/right square brackets, comma -- see
#: lab_frame.py, personas_screen.py, trajectory_timeline.py) rendered as
#: the glyphs users actually press. Textual's own KEY_DISPLAY_ALIASES
#: covers arrows/escape but not the spelled punctuation; anything not in
#: either map passes through unchanged.
_KEY_DISPLAY_SUPPLEMENT = {
    "left_square_bracket": "[",
    "right_square_bracket": "]",
    "comma": ",",
    "period": ".",
    "slash": "/",
    "minus": "-",
    "equal": "=",
}


def _display_key(key: str) -> str:
    """One binding key as the user-facing glyph for help surfaces."""
    display = KEY_DISPLAY_ALIASES.get(key) or _KEY_DISPLAY_SUPPLEMENT.get(key)
    return display if display is not None else key


def _bindings_to_shortcuts(bindings: Any) -> tuple[tuple[str, str], ...]:
    """Flatten BINDINGS entries into (key, description) pairs for help display.

    Accepts both Binding objects and the legacy tuple form so any screen's
    BINDINGS can be rendered as truthful shortcut help. Keys render as
    display glyphs (TASK-32887): the Evals help leaked the raw binding
    identifiers ``left_square_bracket``/``right_square_bracket`` as copy.
    """
    pairs: list[tuple[str, str]] = []
    for entry in bindings or ():
        if isinstance(entry, Binding):
            pairs.append((_display_key(entry.key), entry.description))
        elif isinstance(entry, (tuple, list)) and entry:
            key = _display_key(str(entry[0]))
            description = str(entry[2]) if len(entry) > 2 else ""
            pairs.append((key, description))
    return tuple(pairs)


class TabNavigationProvider(Provider):
    """Provider for tab navigation commands."""

    TAB_HELP_TEXT = {
        TAB_HOME: "Open Home for notifications, status, and next-best actions",
        TAB_CHAT: "Open Console for live agent work, approvals, tools, and RAG",
        TAB_LIBRARY: "Open Library for source material, imports, notes, media, conversations, and Search/RAG",
        TAB_ARTIFACTS: "Open Artifacts for generated outputs, reports, datasets, and Chatbooks",
        TAB_PERSONAS: "Open Roleplay for characters, personas, dictionaries, and behavior profiles",
        TAB_WATCHLISTS_COLLECTIONS: "Open Watchlists for monitored sources, runs, alerts, and recovery",
        TAB_SCHEDULES: "Open Schedules for run timing, triggers, pauses, retries, and recovery",
        TAB_WORKFLOWS: "Open Workflows for reusable procedures, dry-runs, and outputs",
        TAB_MCP: "Open MCP for servers, tools, permissions, auth, and audit",
        TAB_ACP: "Open ACP for agents, sessions, runtimes, diffs, and terminals",
        TAB_SKILLS: "Open Skills for Agent Skills discovery, validation, and attachments",
        TAB_SETTINGS: "Open global preferences, appearance, storage, and app behavior",
        TAB_MEETINGS: "Open Meetings to record a call or a room with a live transcript",
        TAB_CCP: "Switch to Roleplay for characters, personas, dictionaries, and world books",
        TAB_MEDIA: "Switch to media library",
        TAB_SEARCH: "Switch to Library search and RAG",
        TAB_INGEST: "Switch to content ingestion",
        TAB_EVALS: "Switch to evaluation tools",
        TAB_LLM: "Switch to model and provider management",
        TAB_STTS: "Switch to speech-to-text and text-to-speech tools",
        TAB_STUDY: "Switch to flashcards and quizzes",
        TAB_WRITING: "Switch to writing tools",
        TAB_RESEARCH: "Switch to research workflows",
        TAB_RESEARCH_WORKSPACE: "Open Research Workspace for grounded research",
        TAB_CHATBOOKS: "Switch to portable Chatbook context packs",
        TAB_TOOLS_SETTINGS: "Open MCP for legacy tools and settings",
        TAB_LOGS: "Switch to application logs",
        TAB_STATS: "Switch to statistics view",
    }

    NAVIGATION_TABS = tuple(
        destination.primary_route
        for destination in SHELL_DESTINATION_ORDER
    )

    POPULAR_TABS = (
        TAB_HOME,
        TAB_CHAT,
        TAB_LIBRARY,
        TAB_ARTIFACTS,
        TAB_MCP,
        TAB_SETTINGS,
    )

    # task-423: labeled deep-link commands into Library-folded content
    # types that would otherwise only fuzzy-match the generic Library
    # command (which lands on generic Library, not the content row). Each
    # entry is (legacy route, command text, help text); the route rides
    # ``_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT`` to land on its rail row.
    LIBRARY_SUBROUTE_COMMANDS: tuple[tuple[str, str, str], ...] = (
        (
            "artifacts",
            "Tab Navigation: Library — Artifacts",
            "Open Library All artifacts for reports and registered Chatbooks",
        ),
        (
            "skills",
            "Tab Navigation: Library — Skills",
            "Open Library's Skills row for Agent Skills packs, validation, and trust",
        ),
    )

    def __init__(self, screen, *args, **kwargs):
        """Initialize the TabNavigationProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    @classmethod
    def navigation_tab_ids(cls) -> tuple[str, ...]:
        return cls.NAVIGATION_TABS

    @classmethod
    def command_palette_tab_ids(cls) -> tuple[str, ...]:
        # One palette entry per shell destination. Legacy route ids (media,
        # search, ccp, tools_settings, llm_management, stts, evals, coding,
        # logs, stats, writing, research, ...) are no longer separate labeled
        # commands; they are alias terms on their owning destination's single
        # command (see search()).
        return cls.NAVIGATION_TABS

    @staticmethod
    def route_for_tab(tab_id: str) -> str:
        route_aliases = {
            "llm": TAB_LLM,
            TAB_TOOLS_SETTINGS: TAB_MCP,
            TAB_MCP: TAB_MCP,
            TAB_SETTINGS: TAB_SETTINGS,
        }
        return route_aliases.get(tab_id, tab_id)

    @classmethod
    def _shell_destination_for_tab(cls, tab_id: str):
        from .UI.Navigation.shell_destinations import (
            resolve_shell_route,
        )

        resolved = resolve_shell_route(cls.route_for_tab(tab_id))
        try:
            return get_shell_destination(resolved.destination_id)
        except KeyError:
            return None

    @classmethod
    def _destination_alias_terms(cls, destination) -> tuple[str, ...]:
        """Searchable legacy route names that resolve to ``destination``."""
        terms = {
            destination.destination_id,
            destination.label,
            destination.primary_route,
        }
        if destination.full_label:
            terms.add(destination.full_label)
        for related_route in destination.related_routes:
            terms.add(related_route)
            terms.add(get_tab_display_label(related_route))
        terms.update(destination.palette_aliases)
        for legacy_route in destination.legacy_routes:
            terms.add(legacy_route)
            terms.add(get_tab_display_label(legacy_route))
        return tuple(sorted(term for term in terms if term))

    @classmethod
    def _shell_help_text(cls, tab_id: str) -> str | None:
        destination = cls._shell_destination_for_tab(tab_id)
        if destination is None:
            return None
        return f"Open {destination.accessible_label} for {destination.purpose}"

    def _tab_command(self, tab_id: str) -> tuple[str, str, str]:
        for route, text, help_text in self.LIBRARY_SUBROUTE_COMMANDS:
            if route == tab_id:
                return text, tab_id, help_text
        destination = self._shell_destination_for_tab(tab_id)
        label = (
            destination.accessible_label
            if destination is not None
            else get_tab_display_label(tab_id)
        )
        help_text = self._shell_help_text(tab_id) or self.TAB_HELP_TEXT.get(
            tab_id, f"Switch to {label}"
        )
        return f"Tab Navigation: Switch to {label}", tab_id, help_text

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        tab_commands = [
            self._tab_command(tab_id) for tab_id in self.command_palette_tab_ids()
        ]

        for command_text, tab_id, help_text in tab_commands:
            destination = self._shell_destination_for_tab(tab_id)
            alias_terms = (
                self._destination_alias_terms(destination)
                if destination is not None
                else ()
            )
            score = max(
                matcher.match(command_text),
                matcher.match(help_text),
                *(matcher.match(term) for term in alias_terms),
            )
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.switch_tab, tab_id),
                    help=help_text,
                )

        # task-423: Library sub-route deep links (e.g. "skills").
        for route, command_text, help_text in self.LIBRARY_SUBROUTE_COMMANDS:
            score = max(
                matcher.match(command_text),
                matcher.match(help_text),
                matcher.match(route),
            )
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.switch_tab, route),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_tabs = [self._tab_command(tab_id) for tab_id in self.POPULAR_TABS]

        for command_text, tab_id, help_text in popular_tabs:
            yield Hit(
                1.0, command_text, partial(self.switch_tab, tab_id), help=help_text
            )

    def switch_tab(self, tab_id: str) -> None:
        """Switch to the specified tab."""
        try:
            route = self.route_for_tab(tab_id)
            self.app.post_message(NavigateToScreen(route))
            destination = self._shell_destination_for_tab(tab_id)
            label = (
                destination.accessible_label
                if destination is not None
                else get_tab_display_label(tab_id)
            )
            self.app.notify(f"Switched to {label}", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to switch tab: {e}", severity="error")


#: task-18812 / ADR-071: the command-palette entry for the Console focus
#: toggle -- one tuple reused by both QuickActionsProvider lists so the
#: command text, action id, and help string cannot drift apart.
FOCUS_TOGGLE_PALETTE_ENTRY = (
    "Quick Actions: Toggle Focus Mode",
    "toggle_focus_mode",
    "Hide or restore the Console's nav bar and header (Ctrl+Shift+F)",
)


class QuickActionsProvider(Provider):
    """Provider for quick action commands."""

    def __init__(self, screen, *args, **kwargs):
        """Initialize the QuickActionsProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        quick_actions = [
            (
                "Quick Actions: New Chat Conversation",
                "new_chat",
                "Start a new chat conversation",
            ),
            (
                "Quick Actions: New Character Chat",
                "new_character",
                "Start a new character-based conversation",
            ),
            ("Quick Actions: New Note", "new_note", "Create a new note"),
            (
                "Quick Actions: Import Media File",
                "import_media",
                "Import a new media file for processing",
            ),
            (
                "Quick Actions: Search All Content",
                "search_all",
                "Search across all content",
            ),
            FOCUS_TOGGLE_PALETTE_ENTRY,
        ]

        for command_text, action_id, help_text in quick_actions:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.execute_quick_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_actions = [
            (
                "Quick Actions: New Chat Conversation",
                "new_chat",
                "Start a new chat conversation",
            ),
            ("Quick Actions: New Note", "new_note", "Create a new note"),
            (
                "Quick Actions: Search All Content",
                "search_all",
                "Search across all content",
            ),
            (
                "Quick Actions: Import Media File",
                "import_media",
                "Import a new media file for processing",
            ),
            FOCUS_TOGGLE_PALETTE_ENTRY,
        ]

        for command_text, action_id, help_text in popular_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.execute_quick_action, action_id),
                help=help_text,
            )

    def execute_quick_action(self, action_id: str) -> None:
        """Execute the specified quick action."""
        try:
            if action_id == "new_chat":
                _navigate_via_screen(
                    self.app, TAB_CHAT, "Opened Console for a new conversation"
                )
            elif action_id == "new_character":
                _navigate_via_screen(
                    self.app,
                    TAB_PERSONAS,
                    "Opened Roleplay for character setup",
                )
            elif action_id == "new_note":
                _navigate_via_screen(
                    self.app,
                    TAB_LIBRARY,
                    "Opened Library for a new note",
                    {LIBRARY_NAV_CONTEXT_NOTES_CREATE: True},
                )
            elif action_id == "search_all":
                _navigate_via_screen(self.app, TAB_SEARCH, "Opened Library Search/RAG")
            elif action_id == "import_media":
                _navigate_via_screen(
                    self.app, TAB_INGEST, "Opened Import/Export for media import"
                )
            elif action_id == FOCUS_TOGGLE_PALETTE_ENTRY[1]:
                self.app.action_toggle_focus_mode()
        except Exception as e:
            self.app.notify(f"Failed to execute quick action: {e}", severity="error")


class SettingsProvider(Provider):
    """Provider for settings and preferences commands."""

    def __init__(self, screen, *args, **kwargs):
        """Initialize the SettingsProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        settings_commands = [
            (
                "Settings & Preferences: Backup & Restore",
                "backup_restore",
                "Create backups, inspect archives, and review recovery copies",
            ),
            (
                "Settings & Preferences: Open Config File",
                "open_config",
                "Open the configuration file for editing",
            ),
            (
                "Settings & Preferences: Show Database Stats",
                "db_stats",
                "Show database size and statistics",
            ),
            (
                "Settings & Preferences: Open Settings Tab",
                "open_settings",
                "Navigate to the Settings tab",
            ),
        ]

        for command_text, setting_id, help_text in settings_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_setting, setting_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_settings = [
            (
                "Settings & Preferences: Backup & Restore",
                "backup_restore",
                "Create backups, inspect archives, and review recovery copies",
            ),
            (
                "Settings & Preferences: Open Settings Tab",
                "open_settings",
                "Navigate to the Settings tab",
            ),
            (
                "Settings & Preferences: Open Config File",
                "open_config",
                "Open the configuration file for editing",
            ),
            (
                "Settings & Preferences: Show Database Stats",
                "db_stats",
                "Show database size and statistics",
            ),
        ]

        for command_text, setting_id, help_text in popular_settings:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_setting, setting_id),
                help=help_text,
            )

    def handle_setting(self, setting_id: str) -> None:
        """Handle settings commands."""
        try:
            if setting_id == "open_settings":
                _navigate_via_screen(self.app, TAB_SETTINGS, "Opened Settings")
            elif setting_id == "backup_restore":
                self.app.action_backup_restore()
            elif setting_id == "open_config":
                self.app.notify(
                    f"Config file location: {get_cli_config_path()}",
                    severity="information",
                )
            elif setting_id == "db_stats":
                _navigate_via_screen(self.app, TAB_STATS, "Opened Statistics")
        except Exception as e:
            self.app.notify(
                f"Failed to execute settings command: {e}", severity="error"
            )


class CharacterProvider(Provider):
    """Provider for character and persona management commands."""

    def __init__(self, screen, *args, **kwargs):
        """Initialize the CharacterProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        character_commands = [
            (
                "Character/Persona Management: Create New Character",
                "new_character",
                "Create a new character or persona",
            ),
            (
                "Character/Persona Management: Show All Characters",
                "list_characters",
                "Display all available characters",
            ),
            (
                "Character/Persona Management: Open Character Tab",
                "open_character_tab",
                "Navigate to Character Chat tab",
            ),
        ]

        for command_text, action_id, help_text in character_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_character_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_character_actions = [
            (
                "Character/Persona Management: Open Character Tab",
                "open_character_tab",
                "Navigate to Character Chat tab",
            ),
            (
                "Character/Persona Management: Create New Character",
                "new_character",
                "Create a new character or persona",
            ),
            (
                "Character/Persona Management: Show All Characters",
                "list_characters",
                "Display all available characters",
            ),
        ]

        for command_text, action_id, help_text in popular_character_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_character_action, action_id),
                help=help_text,
            )

    def handle_character_action(self, action_id: str) -> None:
        """Handle character management actions."""
        try:
            if action_id == "open_character_tab":
                _navigate_via_screen(
                    self.app,
                    TAB_PERSONAS,
                    "Opened Roleplay",
                )
            elif action_id == "new_character":
                _navigate_via_screen(
                    self.app,
                    TAB_PERSONAS,
                    "Opened Roleplay to create a character",
                )
            elif action_id == "list_characters":
                _navigate_via_screen(
                    self.app,
                    TAB_PERSONAS,
                    "Opened Roleplay to list characters",
                )
        except Exception as e:
            self.app.notify(
                f"Failed to execute character action: {e}", severity="error"
            )


class MediaProvider(Provider):
    """Provider for media and content management commands."""

    def __init__(self, screen, *args, **kwargs):
        """Initialize the MediaProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        media_commands = [
            (
                "Media & Content: Open Media Library",
                "open_media",
                "Navigate to media library",
            ),
            (
                "Media & Content: Search Transcripts",
                "search_transcripts",
                "Search through media transcripts",
            ),
            (
                "Media & Content: Import New Media",
                "import_new",
                "Import new media file",
            ),
        ]

        for command_text, action_id, help_text in media_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_media_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_media_actions = [
            (
                "Media & Content: Open Media Library",
                "open_media",
                "Navigate to media library",
            ),
            (
                "Media & Content: Import New Media",
                "import_new",
                "Import new media file",
            ),
            (
                "Media & Content: Search Transcripts",
                "search_transcripts",
                "Search through media transcripts",
            ),
        ]

        for command_text, action_id, help_text in popular_media_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_media_action, action_id),
                help=help_text,
            )

    def handle_media_action(self, action_id: str) -> None:
        """Handle media management actions."""
        try:
            if action_id == "open_media":
                # task-2851: "media" now aliases to Library's own Media row
                # (screen_registry._SCREEN_ALIASES) instead of the retired
                # standalone MediaScreen -- the toast says so, matching the
                # "Opened Library X" wording the search_transcripts branch
                # below already uses for its own Library-folded route.
                _navigate_via_screen(self.app, TAB_MEDIA, "Opened Library Media")
            elif action_id == "import_new":
                _navigate_via_screen(
                    self.app, TAB_INGEST, "Opened Import/Export for media import"
                )
            elif action_id == "search_transcripts":
                _navigate_via_screen(
                    self.app,
                    TAB_SEARCH,
                    "Opened Library Search/RAG for transcript search",
                )
        except Exception as e:
            self.app.notify(f"Failed to execute media action: {e}", severity="error")


class LibraryIngestProvider(Provider):
    """Provider for the Library ingest deep-link command."""

    COMMANDS = (
        (
            "Library: Chunking Lab",
            "open_chunking_lab",
            "Compare local chunking recipes and save reusable templates",
        ),
        (
            "Library: Import…",
            "open_library_ingest",
            "Open Library and import content",
        ),
    )

    def __init__(self, screen, *args, **kwargs):
        """Initialize the LibraryIngestProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        for command_text, action_id, help_text in self.COMMANDS:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_library_ingest_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        for command_text, action_id, help_text in self.COMMANDS:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_library_ingest_action, action_id),
                help=help_text,
            )

    def handle_library_ingest_action(self, action_id: str) -> None:
        """Handle Library ingest actions."""
        try:
            if action_id == "open_chunking_lab":
                _navigate_via_screen(
                    self.app,
                    "chunking_lab",
                    "Opened local Chunking Lab",
                    {"return_route": getattr(self.screen, "screen_name", "library")},
                )
                return
            if action_id == "open_library_ingest":
                _navigate_via_screen(
                    self.app,
                    TAB_LIBRARY,
                    "Opened Library to import content",
                    {LIBRARY_NAV_CONTEXT_INGEST: True},
                )
        except Exception as e:
            self.app.notify(f"Failed to open Library import: {e}", severity="error")


class SetupWizardProvider(Provider):
    """Provider for re-running the first-run setup wizard."""

    COMMANDS = (
        (
            "Setup: Run setup wizard…",
            "run_setup_wizard",
            "Walk through providers, models, and app configuration",
        ),
    )

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for command_text, action_id, help_text in self.COMMANDS:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_setup_wizard_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        for command_text, action_id, help_text in self.COMMANDS:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_setup_wizard_action, action_id),
                help=help_text,
            )

    def handle_setup_wizard_action(self, action_id: str) -> None:
        try:
            if action_id == "run_setup_wizard":
                from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
                    FirstRunSetupWizard,
                )

                self.app.push_screen(
                    FirstRunSetupWizard(self.app, rerun=True),
                    # TASK-31813: a re-run's cancellation must return to
                    # Settings, not route to the Console.
                    lambda result: self.app._handle_first_run_wizard_result(
                        result, cancel_to_console=False
                    ),
                )
        except Exception as e:
            self.app.notify(f"Failed to open setup wizard: {e}", severity="error")


class PatternGalleryProvider(Provider):
    """Command-palette entry that opens the pattern gallery."""

    COMMANDS = (
        (
            "Design System: Pattern Gallery",
            "open_pattern_gallery",
            "Browse every canonical component pattern live",
        ),
    )

    async def discover(self) -> Hits:
        for text, _id, help_text in self.COMMANDS:
            yield Hit(
                1.0,
                text,
                self.open_gallery,
                help=help_text,
            )

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for text, _id, help_text in self.COMMANDS:
            if (score := matcher.match(text)) > 0:
                yield Hit(
                    score,
                    matcher.highlight(text),
                    self.open_gallery,
                    help=help_text,
                )


    def open_gallery(self) -> None:
        """Load the gallery only when its palette command is invoked."""
        from .Widgets.pattern_gallery import PatternGalleryScreen

        self.app.push_screen(PatternGalleryScreen())


class DeveloperProvider(Provider):
    """Provider for developer and debug commands."""

    def __init__(self, screen, *args, **kwargs):
        """Initialize the DeveloperProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        dev_commands = [
            (
                "Developer/Debug Commands: Show App Info",
                "app_info",
                "Display application version and build info",
            ),
            (
                "Developer/Debug Commands: Open Log File",
                "open_logs",
                "Navigate to application logs",
            ),
            (
                "Developer/Debug Commands: Show Keybindings",
                "show_keys",
                "Display all keyboard shortcuts",
            ),
        ]

        for command_text, action_id, help_text in dev_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_dev_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_dev_actions = [
            (
                "Developer/Debug Commands: Open Log File",
                "open_logs",
                "Navigate to application logs",
            ),
            (
                "Developer/Debug Commands: Show App Info",
                "app_info",
                "Display application version and build info",
            ),
            (
                "Developer/Debug Commands: Show Keybindings",
                "show_keys",
                "Display all keyboard shortcuts",
            ),
        ]

        for command_text, action_id, help_text in popular_dev_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_dev_action, action_id),
                help=help_text,
            )

    def handle_dev_action(self, action_id: str) -> None:
        """Handle developer/debug actions."""
        try:
            if action_id == "open_logs":
                _navigate_via_screen(self.app, TAB_LOGS, "Opened Logs")
            elif action_id == "app_info":
                self.app.notify(
                    "tldw_chatbook - TUI for LLM interactions", severity="information"
                )
            elif action_id == "show_keys":
                self.show_keybindings()
        except Exception as e:
            self.app.notify(
                f"Failed to execute developer action: {e}", severity="error"
            )

    def show_keybindings(self) -> None:
        """Show a generated keybindings panel built from the app's BINDINGS."""
        try:
            state = WorkbenchHelpState(
                route_id="keybindings",
                title="App Keybindings",
                shortcuts=_bindings_to_shortcuts(getattr(self.app, "BINDINGS", ())),
            )
            self.app.push_screen(WorkbenchHelpPanel(state))
        except Exception as e:
            self.app.notify(f"Failed to show keybindings: {e}", severity="error")
