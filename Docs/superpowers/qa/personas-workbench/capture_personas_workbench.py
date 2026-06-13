"""Capture Personas workbench QA screenshots with the REAL app stylesheet + theme.

Unlike a bare-App harness, this loads `tldw_chatbook/css/tldw_cli_modular.tcss`
(the app's generated bundle) and activates the `agentic_terminal` theme, so the
SVG exports show what users actually see.

Run from the repo root:
    python3 Docs/superpowers/qa/personas-workbench/capture_personas_workbench.py
Outputs SVGs into Docs/superpowers/qa/personas-workbench/ux-polish/.
Convert to PNG with any SVG renderer (see svg2png snippet in the QA notes).
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

OUT = REPO_ROOT / "Docs/superpowers/qa/personas-workbench/ux-polish"
OUT.mkdir(parents=True, exist_ok=True)

CHARACTERS = [
    {
        "id": 1,
        "name": "Detective Sam",
        "description": "A hard-boiled noir detective with a soft spot for stray cats.",
        "personality": "Laconic, observant, dryly funny.",
        "scenario": "A rain-soaked city where every case goes deeper than it looks.",
        "system_prompt": "Stay in character as Detective Sam at all times.",
        "first_message": "The name's {{char}}. Who's asking?",
        "tags": ["noir", "detective"],
        "creator": "QA",
        "character_version": "1.0",
        "version": 1,
    },
    {"id": 2, "name": "Lab Assistant", "description": "A meticulous research assistant.", "version": 1},
]

PROFILES = [
    {
        "id": "p-1",
        "name": "Archivist",
        "description": "Preserve, organize, retrieve.",
        "system_prompt": "You are a meticulous archivist. Cite sources for every claim.",
        "version": 3,
    },
    {
        "id": "p-2",
        "name": "Researcher Agent",
        "description": "General research assistant.",
        "system_prompt": "Research carefully before answering.",
        "version": 1,
    },
]

CONVERSATIONS = [
    {"id": "conv-1", "title": "The Marlowe File"},
    {"id": "conv-2", "title": "Cold trail on 5th Street"},
]

MESSAGES = [
    ("Who hired you for the Marlowe case?", "A dame with expensive shoes and cheaper excuses."),
    ("Did you take the job?", "Rent doesn't pay itself. Of course I took it."),
]


def build_mock_app_instance():
    mock = MagicMock()
    service = Mock()
    service.list_persona_profiles = AsyncMock(
        return_value={"items": PROFILES, "total": len(PROFILES)}
    )
    service.get_persona_profile = AsyncMock(return_value=PROFILES[0])
    mock.character_persona_scope_service = service
    mock.app_config = {}
    mock.notify = Mock()
    return mock


async def main() -> None:
    import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as chm
    import tldw_chatbook.UI.Persona_Modules.personas_conversations_controller as pcc
    from textual.app import App
    from tldw_chatbook.css.Themes.themes import agentic_terminal_theme
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    chm.fetch_all_characters = lambda: [dict(c) for c in CHARACTERS]
    chm.fetch_character_by_id = lambda cid: next(
        dict(c) for c in CHARACTERS if str(c["id"]) == str(cid)
    )
    chm._default_character_db = lambda: object()
    pcc.list_character_conversations = (
        lambda db, character_id, limit=50, offset=0: [dict(c) for c in CONVERSATIONS]
    )
    pcc.retrieve_conversation_messages_for_ui = (
        lambda db, conversation_id, character_name, user_name, limit=200: list(MESSAGES)
    )

    mock = build_mock_app_instance()

    class QAApp(App):
        # The real generated bundle: gives the QA captures production styling.
        CSS_PATH = str(REPO_ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss")

        def __init__(self):
            super().__init__()
            self._mock = mock
            self.character_persona_scope_service = mock.character_persona_scope_service

        def __getattr__(self, name):
            if name.startswith(("_", "watch_", "compute_", "validate_", "action_", "key_", "on_")):
                raise AttributeError(name)
            return getattr(self._mock, name)

        def on_mount(self) -> None:
            self.register_theme(agentic_terminal_theme)
            self.theme = "agentic_terminal"
            self.push_screen(PersonasScreen(self))

    app = QAApp()
    async with app.run_test(size=(160, 50)) as pilot:
        screen = pilot.app.screen
        await pilot.pause()
        await pilot.pause()

        # 1. Characters mode with a selection: card + conversations + preview.
        await pilot.click("#personas-library-row-character-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        screen.query_one("#personas-preview-body").display = True
        await pilot.pause()
        pilot.app.save_screenshot(str(OUT / "ux-characters-selected.svg"))

        # 2. Conversation transcript open.
        await pilot.click("#personas-conversation-row-conv-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        pilot.app.save_screenshot(str(OUT / "ux-conversation-transcript.svg"))

        # 3. Editor with a validation error (Advanced section open).
        screen.query_one("#personas-preview-body").display = False
        await pilot.click("#personas-library-new")
        await pilot.pause()
        await pilot.click("#personas-char-editor-advanced-toggle")
        await pilot.pause()
        await pilot.click("#personas-char-editor-save")
        await pilot.pause()
        pilot.app.save_screenshot(str(OUT / "ux-editor-validation-error.svg"))

        # 4. Library search filter active.
        screen.state.has_unsaved_changes = False
        screen._edit_mode = "view"
        search = screen.query_one("#personas-library-search")
        search.value = "sam"
        await pilot.pause()
        await pilot.pause()
        pilot.app.save_screenshot(str(OUT / "ux-library-search.svg"))

        # 5. Personas mode with a selected profile.
        search.value = ""
        await pilot.pause()
        await pilot.click("#personas-mode-personas")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#personas-library-row-persona_profile-p-1")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        pilot.app.save_screenshot(str(OUT / "ux-personas-mode-selected.svg"))

    print("captured:", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    asyncio.run(main())
