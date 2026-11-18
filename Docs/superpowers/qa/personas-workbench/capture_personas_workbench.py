"""Capture Personas workbench QA screenshots with the REAL app stylesheet + theme.

Unlike a bare-App harness, this loads `tldw_chatbook/css/tldw_cli_modular.tcss`
(the app's generated bundle) and activates the `agentic_terminal` theme, so the
SVG exports show what users actually see.

Seeding goes through the same seam the Roleplay suites use
(`Tests/UI/roleplay_frame_harness.character_db_in_use`): a file-backed
ChaChaNotes DB seeded with the app's own `add_character_card` /
`add_conversation` / `add_message`, routed in through
`config.get_chachanotes_db_lazy`. (TASK-33910.33: the old script stubbed
`fetch_all_characters`, which the DB-paged loader no longer reads.)

Run from the repo root:
    python3 Docs/superpowers/qa/personas-workbench/capture_personas_workbench.py

Config isolation: unless TLDW_CONFIG_PATH is already set, the run points
TLDW_CONFIG_PATH, HOME and XDG_CONFIG_HOME/XDG_DATA_HOME at a scratch
directory and disables the keyring, so it never reads or writes
~/.config/tldw_cli or ~/.local/share/tldw_cli.

Outputs: SVGs land in a scratch directory (printed at the end), never in a
tracked path. Pass `--approval-set NAME` to write them to
Docs/superpowers/qa/personas-workbench/NAME/ instead, or `--out DIR` for an
explicit directory. Convert to PNG with any SVG renderer.
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[4]
QA_DIR = REPO_ROOT / "Docs/superpowers/qa/personas-workbench"

STEPS = (
    "ux-characters-selected.svg",
    "ux-conversation-transcript.svg",
    "ux-editor-validation-error.svg",
    "ux-library-search.svg",
    "ux-personas-mode-selected.svg",
)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    where = parser.add_mutually_exclusive_group()
    where.add_argument(
        "--approval-set",
        metavar="NAME",
        help="write into Docs/superpowers/qa/personas-workbench/NAME/ (tracked)",
    )
    where.add_argument("--out", metavar="DIR", help="write into DIR")
    return parser.parse_args(argv)


def _resolve_out(args: argparse.Namespace, scratch: Path) -> Path:
    if args.approval_set:
        name = Path(args.approval_set)
        if name.name != args.approval_set or args.approval_set in ("", ".", ".."):
            raise SystemExit("--approval-set takes a plain directory name")
        out = QA_DIR / args.approval_set
    elif args.out:
        out = Path(args.out).resolve()
    else:
        out = scratch / "captures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _isolate_config(scratch: Path) -> None:
    """Point every config/data path at ``scratch`` before tldw_chatbook imports."""
    if os.environ.get("TLDW_CONFIG_PATH"):
        return
    home = scratch / "home"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["TLDW_CONFIG_PATH"] = str(scratch / "config.toml")
    os.environ["HOME"] = str(home)
    os.environ["XDG_CONFIG_HOME"] = str(home / ".config")
    os.environ["XDG_DATA_HOME"] = str(home / ".local" / "share")
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"


CHARACTERS = [
    {
        "name": "Detective Sam",
        "description": "A hard-boiled noir detective with a soft spot for stray cats.",
        "personality": "Laconic, observant, dryly funny.",
        "scenario": "A rain-soaked city where every case goes deeper than it looks.",
        "system_prompt": "Stay in character as Detective Sam at all times.",
        "first_message": "The name's {{char}}. Who's asking?",
        "tags": ["noir", "detective"],
        "creator": "QA",
        "character_version": "1.0",
    },
    {
        "name": "Lab Assistant",
        "description": "A meticulous research assistant.",
    },
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
    (
        "Who hired you for the Marlowe case?",
        "A dame with expensive shoes and cheaper excuses.",
    ),
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


def seed_character_db(db) -> tuple[int, str]:
    """Seed ``db`` through the app's own writers.

    Characters, two conversations for Detective Sam and the same transcript in
    each, so whichever conversation the list shows first has messages to open.

    Returns:
        ``(sam_id, first_conversation_id)``: Detective Sam's id, and the id of
        the conversation the inspector lists first. The inspector's list is
        one row tall at the capture size, so only that row can be clicked.
    """
    sam_id = None
    for card in CHARACTERS:
        character_id = db.add_character_card(dict(card))
        if sam_id is None:
            sam_id = character_id
    for conversation in CONVERSATIONS:
        db.add_conversation(
            {
                "id": conversation["id"],
                "title": conversation["title"],
                "character_id": sam_id,
                "assistant_kind": "character",
                "assistant_id": str(sam_id),
                "assistant_authority_id": db.get_local_authority_id(),
            }
        )
        previous = None
        for index, (user_text, reply) in enumerate(MESSAGES):
            for role, sender, text in (
                ("user", "User", user_text),
                ("assistant", CHARACTERS[0]["name"], reply),
            ):
                message_id = f"{conversation['id']}-message-{index}-{role}"
                db.add_message(
                    {
                        "id": message_id,
                        "conversation_id": conversation["id"],
                        "parent_message_id": previous,
                        "sender": sender,
                        "role": role,
                        "content": text,
                    }
                )
                previous = message_id
        db.set_conversation_active_leaf(conversation["id"], previous)
    first = db.get_conversations_for_character(sam_id, limit=1)[0]["id"]
    return sam_id, first


async def main(out: Path, scratch: Path) -> None:
    import tldw_chatbook.config as config_module
    from textual.app import App
    from tldw_chatbook.css.Themes.themes import agentic_terminal_theme
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    # A file-backed DB: the screen's off-thread page reads would see an empty
    # :memory: database (one connection per thread).
    db = CharactersRAGDB(scratch / "chachanotes.db", "personas-qa-capture")
    sam_id, first_conversation_id = seed_character_db(db)
    mock = build_mock_app_instance()
    mock.chachanotes_db = db

    class QAApp(App):
        # The real generated bundle plus Roleplay's feature sheet, which the
        # real app parses on first navigation to Roleplay
        # (``TldwCli._SCREEN_OWNED_ROUTE_CSS``): together they give the QA
        # captures production styling. Without the feature sheet the items
        # toolbar (Roleplay frame slice B2b) lays out unstyled and + New
        # cannot be clicked.
        CSS_PATH = [
            str(REPO_ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"),
            str(REPO_ROOT / "tldw_chatbook" / "css" / "screen_feature_roleplay.tcss"),
        ]
        # Outside the real app's shell the nav bar resolves to `1fr` and takes
        # ~22 of 50 rows (measured 2026-10-24), squeezing the workbench to one
        # row per pane; pin it to the strip the user sees.
        CSS = "MainNavigationBar { height: 4 !important; }"

        def __init__(self):
            super().__init__()
            self._mock = mock
            self.character_persona_scope_service = mock.character_persona_scope_service

        def __getattr__(self, name):
            if name.startswith(
                ("_", "watch_", "compute_", "validate_", "action_", "key_", "on_")
            ):
                raise AttributeError(name)
            return getattr(self._mock, name)

        def on_mount(self) -> None:
            self.register_theme(agentic_terminal_theme)
            self.theme = "agentic_terminal"
            self.push_screen(PersonasScreen(self))

    app = QAApp()
    try:
        with patch.object(config_module, "get_chachanotes_db_lazy", lambda: db):
            await _capture(app, out, sam_id, first_conversation_id)
    finally:
        db.close_connection()
    print("captured:", sorted(p.name for p in out.glob("*.svg")))
    print("output dir:", out)


async def _capture(app, OUT: Path, sam_id: int, conversation_id: str) -> None:
    async with app.run_test(size=(160, 50)) as pilot:
        screen = pilot.app.screen
        await pilot.pause()
        await pilot.pause()

        # 1. Characters mode with a selection: card + conversations + preview.
        rows = screen.query_one("#personas-library-rows")
        rows.highlighted = rows.get_option_index(f"character:{sam_id}")
        rows.action_select()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        screen.query_one("#personas-preview-body").display = True
        await pilot.pause()
        pilot.app.save_screenshot(str(OUT / "ux-characters-selected.svg"))

        # 2. Conversation transcript open.
        await pilot.click(f"#personas-conversation-row-{conversation_id}")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        pilot.app.save_screenshot(str(OUT / "ux-conversation-transcript.svg"))

        # 3. Editor with a validation error (Advanced section open).
        screen.query_one("#personas-preview-body").display = False
        # + New opens the New chooser (Roleplay frame slice B2b); Enter picks
        # the current kind's Blank, dev's one-step create.
        await pilot.click("#personas-library-new")
        for _ in range(100):
            focused = pilot.app.focused
            if focused is not None and focused.id == "roleplay-items-chooser-options":
                break
            await pilot.pause(0.05)
        else:
            raise RuntimeError("the New chooser did not open")
        await pilot.press("enter")
        await pilot.pause()
        # The editor scrolls: bring the toggle on screen before clicking it.
        screen.query_one("#personas-char-editor-advanced-toggle").scroll_visible(
            animate=False
        )
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
        # The library search is debounced (0.2 s) before it filters the rows.
        await pilot.pause(0.6)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        pilot.app.save_screenshot(str(OUT / "ux-library-search.svg"))

        # 5. Personas mode with a selected profile.
        search.value = ""
        await pilot.pause(0.6)
        await pilot.app.workers.wait_for_complete()
        # At 160 columns the Personas chip sits past the right edge of the mode
        # strip (x=175 of 160 measured 2026-10-24), so a mouse click is out of
        # bounds: press the chip, which posts the same Button.Pressed.
        screen.query_one("#personas-mode-personas").press()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        rows.highlighted = rows.get_option_index("persona:p-1")
        rows.action_select()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        pilot.app.save_screenshot(str(OUT / "ux-personas-mode-selected.svg"))


def run(argv=None) -> None:
    args = _parse_args(argv)
    scratch = Path(tempfile.mkdtemp(prefix="personas-qa-capture-"))
    _isolate_config(scratch)
    # Before the chdir: a relative --out is relative to the caller's cwd.
    out = _resolve_out(args, scratch)
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    asyncio.run(main(out, scratch))


if __name__ == "__main__":
    run()
