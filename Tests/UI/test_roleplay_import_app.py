"""Roleplay's one Import picker, mounted (frame slice B2b, spec 1.4.6; TASK-33910.4).

On dev Roleplay had four import pickers: Characters' (cards and Buddy packs),
Lore's, Dictionaries' and Import Actor Pack's, each behind its own button, and
Personas had no Import at all. A lore book or a chat dictionary picked in the
Characters picker became a character (named "Book" or "Dict"). B2b puts one
picker behind ``#personas-library-import`` in every kind: a bounded sniff
(``roleplay_import_sniff``) routes each file to its own kind's importer, a
file that fits two kinds asks "Import as:", and the refusals name the file.

Every test here drives the screen like a user: a click on Import, then the
picker seam (``fake_file_picker``) answers with a fixture file, or the real
shared picker opens. The stores are real: a file-backed ChaChaNotes for
characters and lore books, the local chat dictionary service on the same
database, and the Actor Pack import service and controller.
"""

from __future__ import annotations

import asyncio
import builtins
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.screen import Screen
from textual.widgets import Button, Input

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.UI.Persona_Modules.roleplay_import_sniff as sniff_module
from Tests.Actor_Packs.test_actor_pack_attribution import services
from Tests.UI.roleplay_frame_harness import (
    CHOOSER_OPTIONS_ID,
    fake_file_picker,
    open_styled_roleplay,
    painted_rows,
    painted_text,
    select_roleplay_item,
    settle,
    start_new,
    wait_for_chooser,
    wait_until,
)
from Tests.UI.roleplay_import_fixtures import (
    EXPECTED_KINDS,
    EXPECTED_NAMES,
    write_import_fixtures,
    write_nested_json,
    write_sparse,
)
from tldw_chatbook.Actor_Packs.import_controller import ActorPackImportController
from tldw_chatbook.Character_Chat import Character_Chat_Lib
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
    ChatDictionaryScopeService,
)
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
    LocalChatDictionaryService,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.UI.Persona_Modules import buddy_conversion
from tldw_chatbook.UI.Screens import personas_screen as personas_screen_module
from tldw_chatbook.Widgets import enhanced_file_picker as efp
from tldw_chatbook.Widgets.Persona_Widgets.actor_pack_import_review import (
    ActorPackImportReviewDialog,
)
from tldw_chatbook.Widgets.Persona_Widgets.buddy_character_review import (
    BuddyCharacterReviewDialog,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

SIZE = (160, 45)
#: The one seeded character: the list needs a first row before a test starts.
SEED = "Abel Host"


@dataclass
class ImportEnv:
    """One test's stores and fixture files.

    Attributes:
        db: The file-backed ChaChaNotes (characters, lore books, dictionaries).
        files: Fixture name -> path (``write_import_fixtures``).
        dictionaries_service: The local chat dictionary service on ``db``.
        seeded: The character names before any import (the seed and the
            database's own default character).
    """

    db: object
    files: dict[str, Path]
    dictionaries_service: LocalChatDictionaryService
    seeded: list[str]

    def characters(self) -> list[str]:
        return [row["name"] for row in self.db.list_character_cards()]

    def lore_books(self) -> list[str]:
        return [
            book["name"] for book in WorldBookManager(self.db).list_world_books(True)
        ]

    def dictionaries(self) -> list[str]:
        listed = self.dictionaries_service.list_dictionaries(include_inactive=True)
        return [record["name"] for record in listed["dictionaries"]]

    def names(self, kind: str) -> list[str]:
        return {
            "character": self.characters,
            "lore": self.lore_books,
            "dictionary": self.dictionaries,
        }[kind]()


@pytest.fixture
def import_env(tmp_path, monkeypatch, mock_app_instance):
    """Real stores for every kind the one picker routes to."""
    root = tmp_path / "profile"
    db, importer, activation, _ = services(root)
    db.add_character_card({"name": SEED, "description": "seed"})
    dictionaries = LocalChatDictionaryService(
        db, history_store_path=tmp_path / "dictionary-history.json"
    )
    mock_app_instance.chat_dictionary_scope_service = ChatDictionaryScopeService(
        local_service=dictionaries, server_service=None
    )
    mock_app_instance.actor_pack_import_service = importer
    mock_app_instance.actor_pack_import_controller = ActorPackImportController(
        importer, activation
    )
    # The Buddy import guard pins the local character service and the
    # profile directory (as test_buddy_character_review does).
    scope = Mock()
    scope.local_service = importer._local_service
    scope.list_persona_profiles = AsyncMock(return_value={"items": [], "total": 0})
    mock_app_instance.character_persona_scope_service = scope
    monkeypatch.setattr(buddy_conversion, "get_user_data_dir", lambda: root)
    files = write_import_fixtures(tmp_path / "files")
    env = ImportEnv(db=db, files=files, dictionaries_service=dictionaries, seeded=[])
    env.seeded = env.characters()
    assert SEED in env.seeded
    yield env
    db.close_connection()


def _record_notices(monkeypatch, screen) -> list[str]:
    """Every message the screen notifies, in order (toasts are ``_notify``)."""
    seen: list[str] = []
    original = screen._notify

    def record(message, severity="warning", **kwargs):
        seen.append(message)
        return original(message, severity, **kwargs)

    monkeypatch.setattr(screen, "_notify", record)
    return seen


def _import_idle(screen) -> bool:
    """No picker worker, durable character import or Actor Pack step is running."""
    return (
        not screen._io_dialog_active
        and screen._active_character_import_worker() is None
        and screen._actor_pack_import_operation is None
    )


async def _press_import(pilot) -> None:
    """Click Import like a user (a Button ignores clicks for 0.2 s after one)."""
    button = pilot.app.screen.query_one("#personas-library-import", Button)
    await wait_until(
        pilot, lambda: not button.has_class("-active"), what="Import ready again"
    )
    assert await pilot.click("#personas-library-import")
    await pilot.pause()


# ---- AC#4: every format of today's four pickers, through one picker ----


@pytest.mark.parametrize("fixture", sorted(EXPECTED_KINDS))
async def test_every_format_imports_through_the_one_picker_into_its_own_kind(
    fixture, import_env, mock_app_instance, monkeypatch
):
    """From Characters, each file lands in its own kind's store (or opens its
    review). On dev a lore book or a chat dictionary picked here became a
    character, and an Actor Pack was refused as a card."""
    kind = EXPECTED_KINDS[fixture]
    opened = fake_file_picker(monkeypatch, import_env.files[fixture])
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        assert screen.state.active_mode == "characters"
        await _press_import(pilot)
        assert len(opened) == 1
        if kind in ("actor_pack", "buddy"):
            dialog = (
                ActorPackImportReviewDialog
                if kind == "actor_pack"
                else BuddyCharacterReviewDialog
            )
            await wait_until(
                pilot,
                lambda: isinstance(pilot.app.screen, dialog),
                what=f"the {dialog.__name__}",
            )
            await pilot.press("escape")
            await wait_until(
                pilot, lambda: _import_idle(screen), what="the import done"
            )
            assert import_env.characters() == import_env.seeded
            return
        name = EXPECTED_NAMES[fixture]
        await wait_until(
            pilot,
            lambda: name in import_env.names(kind),
            what=f"{name} in the {kind} store",
        )
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
        for other in ("character", "lore", "dictionary"):
            if other != kind:
                assert name not in import_env.names(other), other
        assert import_env.names(kind).count(name) == 1


async def test_import_is_offered_in_every_kind(
    import_env, mock_app_instance, monkeypatch
):
    """One Import, the same picker, in all four kinds. On dev Personas had
    no Import (task-443) and each other kind opened its own picker."""
    opened = fake_file_picker(monkeypatch, None)
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        for mode in ("characters", "personas", "lore", "dictionaries"):
            if mode != screen.state.active_mode:
                assert await pilot.click(f"#personas-mode-{mode}")
                await settle(pilot)
            button = screen.query_one("#personas-library-import", Button)
            assert button.display and not button.disabled, mode
            await _press_import(pilot)
            await wait_until(pilot, lambda: _import_idle(screen), what="the cancel")
        assert [kwargs["context"] for kwargs in opened] == ["character_import"] * 4
        assert len({kwargs["title"] for kwargs in opened}) == 1


# ---- The shared picker itself: one remembered folder, a literal title ----


@pytest.fixture
def picker_config(monkeypatch):
    """The picker's config reads and writes, in memory (test_file_picker_start_dir)."""
    store: dict[tuple[str, str], object] = {}

    def save_settings(section_values):
        for section, values in section_values.items():
            for key, value in values.items():
                store[(section, key)] = value

    monkeypatch.setattr(
        efp,
        "get_cli_setting",
        lambda section, key, default=None: store.get((section, key), default),
    )
    monkeypatch.setattr(
        efp,
        "save_setting_to_cli_config",
        lambda section, key, value: store.__setitem__((section, key), value),
    )
    monkeypatch.setattr(efp, "save_settings_to_cli_config", save_settings)
    return store


async def _open_real_picker(pilot):
    await _press_import(pilot)
    await wait_until(
        pilot,
        lambda: isinstance(pilot.app.screen, efp.EnhancedFileOpen),
        what="the shared file picker",
    )
    await pilot.pause()
    return pilot.app.screen


async def test_the_one_picker_starts_in_and_remembers_one_folder(
    import_env, mock_app_instance, picker_config, tmp_path
):
    """Characters' remembered folder serves every kind (OQ3 iv): the picker
    opens there, and a pick from another folder becomes the next start."""
    remembered = tmp_path / "remembered"
    remembered.mkdir()
    picker_config[("filepicker", "last_dir_character_import")] = str(remembered)
    lore = import_env.files["lore_tldw_export.json"]
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#personas-mode-lore")
        await settle(pilot)
        picker = await _open_real_picker(pilot)
        assert picker.context == "character_import"
        assert Path(picker._location).resolve() == remembered.resolve()
        picker.dismiss(lore)
        await wait_until(
            pilot,
            lambda: "Kepler Atlas" in import_env.lore_books(),
            what="the picked lore book",
        )
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
        key = ("filepicker", "last_dir_character_import")
        await wait_until(
            pilot,
            lambda: picker_config.get(key) == str(lore.parent),
            what="the remembered folder written",
        )
    assert ("filepicker", "last_dir_lore_import") not in picker_config


async def test_the_picker_title_lists_the_formats_literally(
    import_env, mock_app_instance, picker_config
):
    """The title names every kind it takes, in words: the shared picker's
    border title parses markup (TASK-34401, open), so it holds no "["."""
    from tldw_chatbook.UI.Screens.personas_screen import ROLEPLAY_IMPORT_TITLE

    assert "[" not in ROLEPLAY_IMPORT_TITLE
    for word in (
        "PNG",
        "WebP",
        "JSON",
        "Markdown",
        "lore book",
        "chat dictionary",
        "Actor Pack",
        "Buddy",
    ):
        assert word in ROLEPLAY_IMPORT_TITLE, word
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        picker = await _open_real_picker(pilot)
        dialog = picker.query_one("#enhanced-file-dialog")
        assert dialog.border_title == ROLEPLAY_IMPORT_TITLE
        top = painted_rows(picker)[dialog.region.y]
        assert ROLEPLAY_IMPORT_TITLE in top, top
        await pilot.press("escape")
        await wait_until(
            pilot, lambda: _import_idle(pilot.app.screen), what="the cancel"
        )


# ---- "Import as:" for a file that fits two kinds ----


async def test_an_ambiguous_markdown_file_asks_import_as_and_imports_the_choice(
    import_env, mock_app_instance, monkeypatch, tmp_path
):
    """Frontmatter (a card) and ``---@@@---`` rules (a dictionary) in one
    file: the screen asks, and the answer decides the kind. ``key: value``
    lines alone ask too, opening on Chat dictionary (owner decision D8), so
    one Enter imports a dictionary that has no ``---@@@---`` line."""
    both = tmp_path / "field notes.md"
    both.write_text(
        "---\nname: Field Notes\n---\nBP: blood pressure\n---@@@---\nHR: heart rate\n",
        encoding="utf-8",
    )
    rules = tmp_path / "vitals.md"
    rules.write_text("BP: blood pressure\nHR: heart rate\n", encoding="utf-8")
    fake_file_picker(monkeypatch, both, rules)
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        await _press_import(pilot)
        options = await wait_for_chooser(pilot)
        title = pilot.app.screen.query_one("#roleplay-items-chooser-title")
        assert str(title.render()) == "Import field notes.md as:"
        ids = [options.get_option_at_index(i).id for i in range(options.option_count)]
        assert ids == ["character", "dictionary"]
        assert options.highlighted_option.id == "character"
        await pilot.press("down", "enter")
        await wait_until(
            pilot,
            lambda: "field notes" in import_env.dictionaries(),
            what="the dictionary",
        )
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
        assert import_env.characters() == import_env.seeded
        await _press_import(pilot)
        options = await wait_for_chooser(pilot)
        ids = [options.get_option_at_index(i).id for i in range(options.option_count)]
        assert ids == ["character", "dictionary"]
        assert options.highlighted_option.id == "dictionary"
        await pilot.press("enter")
        await wait_until(
            pilot, lambda: "vitals" in import_env.dictionaries(), what="the rules"
        )
        await wait_until(pilot, lambda: _import_idle(screen), what="the second import")
        assert import_env.characters() == import_env.seeded


async def test_escape_in_import_as_imports_nothing(
    import_env, mock_app_instance, monkeypatch, tmp_path
):
    both = tmp_path / "both.md"
    both.write_text("---\nname: Both\n---\nBP: blood pressure\n---@@@---\n", "utf-8")
    fake_file_picker(monkeypatch, both)
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        await _press_import(pilot)
        await wait_for_chooser(pilot)
        await pilot.press("escape")
        await wait_until(pilot, lambda: _import_idle(screen), what="the cancel")
        await settle(pilot)
        assert import_env.characters() == import_env.seeded
        assert import_env.dictionaries() == []


# ---- AC#6: the bounded sniff refuses before reading, and never crashes ----


async def test_an_11_mb_file_is_refused_before_reading_and_the_toast_says_why(
    import_env, mock_app_instance, monkeypatch, tmp_path
):
    big = write_sparse(tmp_path / "huge card.json", 11 * 1024 * 1024)
    opened_paths: list[str] = []
    real_open = builtins.open

    def spy_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).name == big.name:
            opened_paths.append(str(file))
        return real_open(file, *args, **kwargs)

    reads: list[str] = []
    real_read_bytes, real_read_text = Path.read_bytes, Path.read_text

    def spy_read_bytes(self):
        if self.name == big.name:
            reads.append("read_bytes")
        return real_read_bytes(self)

    def spy_read_text(self, *args, **kwargs):
        if self.name == big.name:
            reads.append("read_text")
        return real_read_text(self, *args, **kwargs)

    fake_file_picker(monkeypatch, big)
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        notices = _record_notices(monkeypatch, screen)
        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr(Path, "read_bytes", spy_read_bytes)
        monkeypatch.setattr(Path, "read_text", spy_read_text)
        await _press_import(pilot)
        await wait_until(pilot, lambda: bool(notices), what="the refusal")
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
        monkeypatch.setattr(builtins, "open", real_open)
    assert notices == ["huge card.json is larger than 10 MB; nothing was imported."]
    assert opened_paths == [] and reads == []
    assert import_env.characters() == import_env.seeded


@pytest.mark.parametrize("opener", ["[", '{"a":'], ids=["arrays", "objects"])
async def test_a_nested_9_mb_json_says_not_a_roleplay_file_and_imports_nothing(
    opener, import_env, mock_app_instance, monkeypatch, tmp_path
):
    nested = write_nested_json(tmp_path / "deep.json", 9 * 1024 * 1024, opener=opener)
    fake_file_picker(monkeypatch, nested)
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        notices = _record_notices(monkeypatch, screen)
        await _press_import(pilot)
        await wait_until(pilot, lambda: bool(notices), what="the refusal")
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
    assert notices == ["deep.json is not a Roleplay file; nothing was imported."]
    assert import_env.characters() == import_env.seeded
    assert import_env.lore_books() == []
    assert import_env.dictionaries() == []


# ---- OQ14: the server runtime ----


async def test_a_card_under_the_server_runtime_is_refused_literally_and_a_lore_book_still_imports(
    import_env, mock_app_instance, monkeypatch
):
    """Import stays offered while browsing server characters (dev disabled
    it there). A card or a Buddy pack (it lands as a local character) cannot
    go to the read-only server list, so each is refused by name; a lore book
    goes to the local store, as dev's Lore picker already allowed."""
    records = [{"id": 7, "name": "Remote Elara"}]
    mock_app_instance.runtime_backend = "server"
    mock_app_instance.active_server_id = "server-a"
    mock_app_instance.character_persona_scope_service = SimpleNamespace(
        local_service=None,
        list_characters=AsyncMock(return_value={"items": records, "total": 1}),
        search_characters=AsyncMock(return_value={"items": records, "total": 1}),
        get_character=AsyncMock(
            return_value={"id": 7, "name": "Remote Elara", "description": "Remote"}
        ),
    )
    fake_file_picker(
        monkeypatch,
        import_env.files["card_v2.json"],
        import_env.files["buddy.tldw-persona-vpack"],
        import_env.files["lore_tldw_export.json"],
    )
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        assert screen.state.runtime_source == "server"
        notices = _record_notices(monkeypatch, screen)
        button = screen.query_one("#personas-library-import", Button)
        assert button.display and not button.disabled
        await _press_import(pilot)
        await wait_until(pilot, lambda: bool(notices), what="the refusal")
        await wait_until(pilot, lambda: _import_idle(screen), what="the refusal done")
        assert notices == [
            "Server characters are read-only here. card_v2.json was not imported."
        ]
        await _press_import(pilot)
        await wait_until(pilot, lambda: len(notices) == 2, what="the Buddy refusal")
        await wait_until(pilot, lambda: _import_idle(screen), what="the refusal done")
        assert notices[1] == (
            "Server characters are read-only here. "
            "buddy.tldw-persona-vpack was not imported."
        )
        await _press_import(pilot)
        await wait_until(
            pilot,
            lambda: "Kepler Atlas" in import_env.lore_books(),
            what="the lore book in the local store",
        )
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
    assert import_env.characters() == import_env.seeded


# ---- Reentrancy: a second Import while one runs ----


async def test_a_second_import_while_one_runs_says_so(
    import_env, mock_app_instance, monkeypatch
):
    """On dev a second press during a character import said so, and one
    during a lore or dictionary import was ignored without a word. The one
    picker asks one check for every kind; a lore import holds it here."""
    entered, release = threading.Event(), threading.Event()
    real_import = WorldBookManager.import_world_book

    def slow_import(self, *args, **kwargs):
        entered.set()
        release.wait(30)
        return real_import(self, *args, **kwargs)

    monkeypatch.setattr(WorldBookManager, "import_world_book", slow_import)
    fake_file_picker(monkeypatch, import_env.files["lore_tldw_export.json"])
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#personas-mode-lore")
        await settle(pilot)
        notices = _record_notices(monkeypatch, screen)
        try:
            await _press_import(pilot)
            await wait_until(pilot, entered.is_set, what="the import in its thread")
            await _press_import(pilot)
            await wait_until(pilot, lambda: bool(notices), what="the answer")
            assert notices == ["An import is already in progress."]
        finally:
            release.set()
        await wait_until(
            pilot, lambda: "Kepler Atlas" in import_env.lore_books(), what="the book"
        )
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
    assert import_env.lore_books() == ["Kepler Atlas"]


async def test_ctrl_n_does_nothing_while_an_import_runs(
    import_env, mock_app_instance, monkeypatch, tmp_path
):
    """Deferred minor T4b: Import and New held independent slots, so a Ctrl+N
    during the sniff opened the New chooser and "Import X as:" then stacked
    on it (new with B2b; nothing was lost). Now Ctrl+N does nothing while an
    import runs or its modal is open; a modal on top swallows the key itself,
    so the window is the sniff, held here."""
    entered, release = threading.Event(), threading.Event()
    real_sniff = sniff_module.sniff_import_file

    def slow_sniff(path, *, max_bytes):
        entered.set()
        release.wait(30)
        return real_sniff(path, max_bytes=max_bytes)

    monkeypatch.setattr(sniff_module, "sniff_import_file", slow_sniff)
    both = tmp_path / "both.md"
    both.write_text(
        "---\nname: Both\n---\nBP: blood pressure\n---@@@---\nHR: heart rate\n",
        "utf-8",
    )
    fake_file_picker(monkeypatch, both)
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        screen.query_one("#personas-library-rows").focus()
        try:
            await _press_import(pilot)
            await wait_until(pilot, entered.is_set, what="the held sniff")
            await pilot.press("ctrl+n")
            for _ in range(4):
                await pilot.pause()
            assert not screen._new_chooser_open
            assert pilot.app.screen is screen
        finally:
            release.set()
        options = await wait_for_chooser(pilot)
        assert "Import both.md as:" in _chooser_title(pilot)
        assert pilot.app.focused is options
        assert (
            sum(bool(s.query(f"#{CHOOSER_OPTIONS_ID}")) for s in pilot.app.screen_stack)
            == 1
        )
        await pilot.press("escape")
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
    assert import_env.characters() == import_env.seeded


async def test_import_does_nothing_while_the_new_chooser_is_open(
    import_env, mock_app_instance, monkeypatch
):
    """Deferred minor T4b, the other direction: an Import request that arrives
    while the New chooser is open opens no picker (new with B2b)."""
    opened = fake_file_picker(monkeypatch, import_env.files["lore_tldw_export.json"])
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        screen.query_one("#personas-library-new", Button).press()
        await wait_for_chooser(pilot)
        screen._open_roleplay_import()
        for _ in range(4):
            await pilot.pause()
        assert opened == [] and not screen._roleplay_import_running
        assert not screen._io_dialog_active
        await pilot.press("escape")
        await wait_until(
            pilot, lambda: not screen._new_chooser_open, what="the chooser closed"
        )
    assert import_env.lore_books() == []


def _chooser_title(pilot) -> str:
    """The open choice modal's painted title (the first stacked one from the top)."""
    for stacked in reversed(pilot.app.screen_stack):
        titles = stacked.query("#roleplay-items-chooser-title")
        if titles:
            return str(titles.first().render())
    return ""


# ---- The router's error branches (deferred minor T4a) ----


@pytest.mark.parametrize(
    "exc, toast",
    [
        (
            PermissionError(13, "denied"),
            "card_v2.json could not be imported: [Errno 13] denied",
        ),
        (RuntimeError("boom [b]x"), "Import failed: boom [b]x"),
    ],
    ids=["os-error", "unexpected"],
)
async def test_a_sniff_that_raises_toasts_literally_and_frees_the_picker(
    exc, toast, import_env, mock_app_instance, monkeypatch
):
    """Deferred minor T4a: the router's two error branches on the mounted
    screen. An OSError or ValueError from the sniff (a path the validator
    refuses, a file that cannot be stat'ed or read) is "{file} could not be
    imported: {exc}"; anything else reaches the worker's ``except Exception``
    and is "Import failed: {exc}". Both are painted literally, the app stays
    up, and the picker's slot is freed."""

    def raising(path, *, max_bytes):
        raise exc

    monkeypatch.setattr(sniff_module, "sniff_import_file", raising)
    fake_file_picker(monkeypatch, import_env.files["card_v2.json"])
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        notices = _record_app_notices(pilot.app)
        await _press_import(pilot)
        await wait_until(pilot, lambda: bool(notices), what="the toast", timeout=8.0)
        await wait_until(pilot, lambda: _import_idle(screen), what="the slot freed")
        assert [(m, kwargs["markup"]) for m, kwargs in notices] == [(toast, False)]
        assert not screen._roleplay_import_running
        assert pilot.app.is_running and pilot.app.return_code is None
        # The picker opens again.
        await _press_import(pilot)
        await wait_until(pilot, lambda: len(notices) == 2, what="the second toast")
        await wait_until(pilot, lambda: _import_idle(screen), what="the slot freed")
    assert import_env.characters() == import_env.seeded


# ---- AC#7: a hostile file name stays literal ----


@pytest.mark.parametrize(
    "stem", ["[b]x", "[@click=app.quit]N", "[TODO] y", "[x=y]z"], ids=range(4)
)
async def test_a_hostile_file_name_paints_literally_in_import_as_and_the_refusal(
    stem, import_env, mock_app_instance, monkeypatch, tmp_path
):
    both = tmp_path / f"{stem}.md"
    both.write_text("---\nname: Both\n---\nBP: blood pressure\n---@@@---\n", "utf-8")
    junk = tmp_path / f"{stem}.json"
    junk.write_text("[1, 2, 3]", "utf-8")
    fake_file_picker(monkeypatch, both, junk)
    async with open_styled_roleplay(
        "mock",
        mock_app_instance,
        size=SIZE,
        character_db=import_env.db,
        notifications=True,
    ) as pilot:
        screen = pilot.app.screen
        await _press_import(pilot)
        await wait_for_chooser(pilot)
        title = f"Import {stem}.md as:"
        assert any(title in row for row in painted_rows(pilot.app.screen))
        assert pilot.app.focused.id == CHOOSER_OPTIONS_ID
        await pilot.press("escape")
        await wait_until(pilot, lambda: _import_idle(screen), what="the cancel")
        await _press_import(pilot)
        refusal = f"{stem}.json is not a Roleplay file; nothing was imported."
        await wait_until(
            pilot,
            lambda: refusal in _painted_toasts(pilot.app.screen),
            what="the toast",
        )
        assert pilot.app.is_running


def _painted_toasts(screen) -> str:
    """What every toast paints, each toast's rows joined by spaces (toasts wrap)."""
    from textual.widgets._toast import Toast

    return "\n".join(
        " ".join(painted_text(screen, toast.content_region).split())
        for toast in screen.query(Toast)
    )


# ---- AC#5: where an import lands (frame slice B2b, spec 1.4.6) ----
#
# Auto-select runs only when every draft is clean, no newer intent arrived
# since the picker opened, and the screen still shows; otherwise the import
# still lands and a literal toast names where.


def _record_app_notices(app) -> list[tuple[str, dict]]:
    """Every app notification with its keyword arguments (``_notify`` calls it)."""
    seen: list[tuple[str, dict]] = []
    original = app.notify

    def record(message, **kwargs):
        seen.append((str(message), kwargs))
        return original(message, **kwargs)

    app.notify = record
    return seen


@contextmanager
def _slow_character_import(
    monkeypatch,
) -> Iterator[tuple[threading.Event, threading.Event]]:
    """Hold every character import in its thread until ``release`` is set."""
    entered, release = threading.Event(), threading.Event()
    real = Character_Chat_Lib.import_and_save_character_from_file_with_outcome

    def slow(*args, **kwargs):
        entered.set()
        release.wait(30)
        return real(*args, **kwargs)

    monkeypatch.setattr(
        Character_Chat_Lib, "import_and_save_character_from_file_with_outcome", slow
    )
    try:
        yield entered, release
    finally:
        release.set()


async def _dirty_character_draft(pilot) -> Input:
    """A new character draft with a typed name (the unsaved guard's state)."""
    screen = pilot.app.screen
    await start_new(pilot)
    await wait_until(pilot, lambda: screen._edit_mode == "create", what="the draft")
    await settle(pilot)
    name = screen.query_one("#personas-char-editor-name", Input)
    name.focus()
    await pilot.press(*"Draft Hero")
    await wait_until(
        pilot, lambda: screen.state.has_unsaved_changes, what="the dirty draft"
    )
    return name


def _draft_kept(screen, name: Input) -> bool:
    editor = screen.query_one("#ccp-character-editor-view")
    return bool(
        screen._edit_mode == "create"
        and editor.display
        and name.value == "Draft Hero"
        and screen.state.has_unsaved_changes
        and screen.state.selected_entity_id is None
    )


async def test_an_import_with_a_dirty_draft_lands_and_a_literal_toast_names_where(
    import_env, mock_app_instance, monkeypatch
):
    """D1: Import is unguarded. On dev the unsaved-changes dialog came before
    the picker (survey PROBE1), so the import never landed beside a draft."""
    fake_file_picker(monkeypatch, import_env.files["lore_tldw_export.json"])
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        name = await _dirty_character_draft(pilot)
        notices = _record_app_notices(pilot.app)
        await _press_import(pilot)
        toast = "Imported Kepler Atlas · in Lore books"
        await wait_until(
            pilot, lambda: any(m == toast for m, _ in notices), what="the toast"
        )
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
        await settle(pilot)
        assert import_env.lore_books() == ["Kepler Atlas"]
        assert screen.state.active_mode == "characters"
        # No unsaved-changes dialog either: the landing never asks.
        assert pilot.app.screen is screen
        assert _draft_kept(screen, name)
        assert [kwargs["markup"] for m, kwargs in notices if m == toast] == [False]


async def test_a_draft_started_during_a_slow_import_is_never_replaced(
    import_env, mock_app_instance, monkeypatch
):
    """Survey PROBE2: on dev the import that landed swapped its card in over
    a draft started meanwhile, with no prompt (the draft stayed dirty, out of
    view)."""
    fake_file_picker(monkeypatch, import_env.files["card_v2.json"])
    with _slow_character_import(monkeypatch) as (entered, release):
        async with open_styled_roleplay(
            "mock", mock_app_instance, size=SIZE, character_db=import_env.db
        ) as pilot:
            screen = pilot.app.screen
            notices = _record_notices(monkeypatch, screen)
            await _press_import(pilot)
            await wait_until(pilot, entered.is_set, what="the import in its thread")
            name = await _dirty_character_draft(pilot)
            release.set()
            await wait_until(
                pilot,
                lambda: "Wren Halloway" in import_env.characters(),
                what="the imported card",
            )
            await wait_until(
                pilot, lambda: _import_idle(screen), what="the import done"
            )
            await settle(pilot)
            assert pilot.app.screen is screen
            assert _draft_kept(screen, name)
            assert notices[-1:] == ["Imported Wren Halloway · in Characters"]


async def test_a_newer_selection_during_an_import_keeps_the_selection_and_toasts(
    import_env, mock_app_instance, monkeypatch
):
    seed_id = str(import_env.db.get_character_card_by_name(SEED)["id"])
    fake_file_picker(monkeypatch, import_env.files["card_v2.json"])
    with _slow_character_import(monkeypatch) as (entered, release):
        async with open_styled_roleplay(
            "mock", mock_app_instance, size=SIZE, character_db=import_env.db
        ) as pilot:
            screen = pilot.app.screen
            notices = _record_notices(monkeypatch, screen)
            await _press_import(pilot)
            await wait_until(pilot, entered.is_set, what="the import in its thread")
            await select_roleplay_item(pilot, screen, f"character:{seed_id}")
            await wait_until(
                pilot,
                lambda: screen.state.selected_entity_id == seed_id,
                what="the user's selection",
            )
            release.set()
            await wait_until(
                pilot, lambda: _import_idle(screen), what="the import done"
            )
            await settle(pilot)
            assert "Wren Halloway" in import_env.characters()
            assert screen.state.selected_entity_id == seed_id
            assert notices[-1:] == ["Imported Wren Halloway · in Characters"]


async def test_a_cursor_move_during_an_import_keeps_the_cursor_and_toasts(
    import_env, mock_app_instance, monkeypatch
):
    """A cursor move that loads nothing is still a newer intent (the list's
    user_moved): the import must not pull the cursor to its row."""
    fake_file_picker(monkeypatch, import_env.files["card_v2.json"])
    with _slow_character_import(monkeypatch) as (entered, release):
        async with open_styled_roleplay(
            "mock", mock_app_instance, size=SIZE, character_db=import_env.db
        ) as pilot:
            screen = pilot.app.screen
            notices = _record_notices(monkeypatch, screen)
            item_list = screen.query_one("#personas-library-rows")
            await _press_import(pilot)
            await wait_until(pilot, entered.is_set, what="the import in its thread")
            loaded = screen.state.selected_entity_id
            item_list.focus()
            await pilot.press("end")
            await settle(pilot)
            moved_to = item_list.highlighted
            release.set()
            await wait_until(
                pilot, lambda: _import_idle(screen), what="the import done"
            )
            await settle(pilot)
            assert "Wren Halloway" in import_env.characters()
            assert screen.state.selected_entity_id == loaded
            assert item_list.highlighted == moved_to
            assert notices[-1:] == ["Imported Wren Halloway · in Characters"]


async def test_a_clean_import_of_another_kind_switches_to_it_and_selects_it(
    import_env, mock_app_instance, monkeypatch
):
    """Quick import across kinds: on dev a lore book imported from Characters
    (only possible by the Lore picker) said "open Lore to see it"."""
    fake_file_picker(monkeypatch, import_env.files["lore_tldw_export.json"])
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        notices = _record_notices(monkeypatch, screen)
        await _press_import(pilot)
        await wait_until(
            pilot,
            lambda: screen.state.selected_entity_name == "Kepler Atlas",
            what="the lore book selected",
        )
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
        await settle(pilot)
        assert screen.state.active_mode == "lore"
        assert screen.state.selected_entity_kind == "lore"
        # Selected, so the plain imported copy (final review F3, spec 1.4.6):
        # "Imported X · in Kind" names where an import went only when it was
        # NOT selected, as the character path already did. The branch before
        # this fix said "Imported Kepler Atlas · in Lore books" here too.
        assert notices[-1:] == ["Kepler Atlas imported."]
        # The same file again: renamed, selected, and the rename note kept.
        await _press_import(pilot)
        await wait_until(
            pilot,
            lambda: screen.state.selected_entity_name == "Kepler Atlas 2",
            what="the renamed lore book selected",
        )
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
        await settle(pilot)
        assert notices[-1:] == [
            "Kepler Atlas 2 imported. Renamed to avoid a name clash."
        ]
        assert import_env.lore_books() == ["Kepler Atlas", "Kepler Atlas 2"]


async def test_an_import_that_lands_after_a_kind_switch_still_toasts(
    import_env, mock_app_instance, monkeypatch
):
    """Survey PROBE3: on dev a character import that landed after a switch
    to another kind said nothing at all (the character was created)."""
    fake_file_picker(monkeypatch, import_env.files["card_v2.json"])
    with _slow_character_import(monkeypatch) as (entered, release):
        async with open_styled_roleplay(
            "mock", mock_app_instance, size=SIZE, character_db=import_env.db
        ) as pilot:
            screen = pilot.app.screen
            notices = _record_notices(monkeypatch, screen)
            await _press_import(pilot)
            await wait_until(pilot, entered.is_set, what="the import in its thread")
            assert await pilot.click("#personas-mode-personas")
            await wait_until(
                pilot,
                lambda: screen.state.active_mode == "personas",
                what="Personas",
            )
            release.set()
            await wait_until(
                pilot, lambda: _import_idle(screen), what="the import done"
            )
            await settle(pilot)
            assert "Wren Halloway" in import_env.characters()
            assert screen.state.active_mode == "personas"
            assert notices[-1:] == ["Imported Wren Halloway · in Characters"]


async def test_a_dictionary_import_in_dictionaries_mode_toasts(
    import_env, mock_app_instance, monkeypatch
):
    """On dev a dictionary imported in Dictionaries was selected with no toast."""
    fake_file_picker(monkeypatch, import_env.files["dictionary_export.json"])
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#personas-mode-dictionaries")
        await settle(pilot)
        notices = _record_notices(monkeypatch, screen)
        await _press_import(pilot)
        await wait_until(
            pilot,
            lambda: screen.state.selected_entity_name == "Medical Abbrev",
            what="the dictionary selected",
        )
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
        # Selected: the plain imported copy (final review F3); before this fix
        # the branch said "Imported Medical Abbrev · in Chat dictionaries".
        assert notices[-1:] == ["Medical Abbrev imported."]


async def test_a_dictionary_import_beside_a_dirty_draft_toasts_where_it_went(
    import_env, mock_app_instance, monkeypatch
):
    """Final review F3, the other branch: a dictionary that lands beside a
    dirty draft is not selected, so the toast names where it went (R33), as
    a lore book's and a character's do. On dev a dictionary import was
    selected whenever Dictionaries showed, and said nothing."""
    fake_file_picker(monkeypatch, import_env.files["dictionary_export.json"])
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        name = await _dirty_character_draft(pilot)
        notices = _record_notices(monkeypatch, screen)
        await _press_import(pilot)
        toast = "Imported Medical Abbrev · in Chat dictionaries"
        await wait_until(pilot, lambda: toast in notices, what="the toast")
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
        await settle(pilot)
        assert import_env.dictionaries() == ["Medical Abbrev"]
        assert screen.state.active_mode == "characters"
        assert pilot.app.screen is screen and _draft_kept(screen, name)


async def test_an_import_while_a_save_is_in_flight_does_not_auto_select(
    import_env, mock_app_instance, monkeypatch
):
    """The landing reads R24's aggregate predicate, which counts a save still
    in flight; _run_guarded's narrower test (form edits and visual authoring
    only) would call this screen clean and switch kinds under the save."""
    fake_file_picker(monkeypatch, import_env.files["lore_tldw_export.json"])
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        notices = _record_notices(monkeypatch, screen)
        # As _handle_save_requested leaves it while the character save runs.
        screen._character_save_inflight = True
        assert not screen.state.has_unsaved_changes
        await _press_import(pilot)
        await wait_until(pilot, lambda: bool(notices), what="the toast")
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
        assert import_env.lore_books() == ["Kepler Atlas"]
        assert screen.state.active_mode == "characters"
        assert notices == ["Imported Kepler Atlas · in Lore books"]


def _hold_the_landing_refresh(
    monkeypatch, screen
) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold the landing's own character-list refresh until ``gate`` is set."""
    entered, gate = asyncio.Event(), asyncio.Event()
    landing = False
    real_land = screen._land_import
    real_refresh = screen.character_handler.refresh_character_list

    async def land(*args, **kwargs):
        nonlocal landing
        landing = True
        try:
            return await real_land(*args, **kwargs)
        finally:
            landing = False

    async def refresh(*args, **kwargs):
        if landing and not entered.is_set():
            entered.set()
            await gate.wait()
        return await real_refresh(*args, **kwargs)

    monkeypatch.setattr(screen, "_land_import", land)
    monkeypatch.setattr(screen.character_handler, "refresh_character_list", refresh)
    return entered, gate


@pytest.mark.parametrize("meanwhile", ["draft", "leave"])
async def test_what_happens_while_the_landing_refreshes_its_list_wins(
    meanwhile, import_env, mock_app_instance, monkeypatch
):
    """The landing decides, then awaits its own list refresh (a full-library
    read) or kind switch before it selects; it decides again after them, so
    a draft started in that window is kept, a screen left in it is not
    selected on, and the import toasts. On dev the character landing
    re-checked only that Characters still showed after its refresh, so a
    draft started during it was replaced (exists on dev, wider there; B2b
    fixes it); B2b's first cut dropped dev's re-check, so a screen left
    mid-refresh was selected on (new with B2b, fixed here)."""
    fake_file_picker(monkeypatch, import_env.files["card_v2.json"])
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        notices = _record_notices(monkeypatch, screen)
        loaded = screen.state.selected_entity_id
        entered, gate = _hold_the_landing_refresh(monkeypatch, screen)
        await _press_import(pilot)
        await wait_until(pilot, entered.is_set, what="the landing's refresh")
        if meanwhile == "draft":
            name = await _dirty_character_draft(pilot)
        else:
            await pilot.app.push_screen(Screen())
            await settle(pilot)
        gate.set()
        await wait_until(pilot, lambda: _import_idle(screen), what="the import done")
        await settle(pilot)
        assert "Wren Halloway" in import_env.characters()
        assert notices[-1:] == ["Imported Wren Halloway · in Characters"]
        if meanwhile == "draft":
            assert pilot.app.screen is screen
            assert _draft_kept(screen, name)
        else:
            assert pilot.app.screen is not screen
            assert screen.state.selected_entity_id == loaded


async def test_end_with_rows_still_unloaded_during_an_import_keeps_the_cursor(
    import_env, mock_app_instance, monkeypatch
):
    """End on a list whose rows are not all loaded owes a window before it
    moves; it is a newer move all the same, so the import must not pull the
    cursor back to its row (the cursor-move test's list is fully loaded).
    On dev every import selected (exists on dev; B2b fixes it)."""
    monkeypatch.setattr(personas_screen_module, "PERSONAS_LIBRARY_PAGE_SIZE", 3)
    for index in range(9):
        import_env.db.add_character_card(
            {"name": f"Zed {index:02d}", "description": "seed"}
        )
    fake_file_picker(monkeypatch, import_env.files["card_v2.json"])
    with _slow_character_import(monkeypatch) as (entered, release):
        async with open_styled_roleplay(
            "mock", mock_app_instance, size=SIZE, character_db=import_env.db
        ) as pilot:
            screen = pilot.app.screen
            notices = _record_notices(monkeypatch, screen)
            item_list = screen.query_one("#personas-library-rows")
            await settle(pilot)
            assert screen._window_state.more_available
            await _press_import(pilot)
            await wait_until(pilot, entered.is_set, what="the import in its thread")
            loaded = screen.state.selected_entity_id
            item_list.focus()
            await pilot.press("end")
            await wait_until(
                pilot,
                lambda: (
                    not screen._window_state.more_available
                    and item_list.highlighted == item_list.option_count - 1
                ),
                what="the cursor on the list's last row",
            )
            moved_to = item_list.highlighted
            release.set()
            await wait_until(
                pilot, lambda: _import_idle(screen), what="the import done"
            )
            await settle(pilot)
            assert "Wren Halloway" in import_env.characters()
            assert screen.state.selected_entity_id == loaded
            assert item_list.highlighted == moved_to
            assert notices[-1:] == ["Imported Wren Halloway · in Characters"]


async def test_a_search_typed_during_an_import_is_kept_and_the_import_toasts(
    import_env, mock_app_instance, monkeypatch
):
    """A search typed after the picker opened is a newer intent: the landing
    leaves it and toasts. On dev the character landing cleared it and
    selected (exists on dev); for lore and dictionaries dev kept the query,
    so clearing it there would have been new with B2b. B2b fixes both."""
    fake_file_picker(monkeypatch, import_env.files["card_v2.json"])
    with _slow_character_import(monkeypatch) as (entered, release):
        async with open_styled_roleplay(
            "mock", mock_app_instance, size=SIZE, character_db=import_env.db
        ) as pilot:
            screen = pilot.app.screen
            notices = _record_notices(monkeypatch, screen)
            await _press_import(pilot)
            await wait_until(pilot, entered.is_set, what="the import in its thread")
            loaded = screen.state.selected_entity_id
            search = screen.query_one("#personas-library-search", Input)
            search.focus()
            await pilot.press(*"Abel")
            await wait_until(
                pilot,
                lambda: screen.state.search_query == "Abel",
                what="the typed search",
            )
            release.set()
            await wait_until(
                pilot, lambda: _import_idle(screen), what="the import done"
            )
            await settle(pilot)
            assert "Wren Halloway" in import_env.characters()
            assert search.value == "Abel"
            assert screen.state.search_query == "Abel"
            assert screen.state.selected_entity_id == loaded
            assert notices[-1:] == ["Imported Wren Halloway · in Characters"]


async def test_a_sort_cycle_and_a_tag_pick_are_newer_intents(
    import_env, mock_app_instance
):
    """The user's sort cycle (``s`` on the list) and tag pick (the picker's
    only way to apply one) end an import's claim to the cursor, like a
    typed search. A programmatic search clear does not: the landing's own
    clear must still select (test_personas_workbench's
    ``test_import_success_refreshes_selects_and_clears_search``)."""
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, character_db=import_env.db
    ) as pilot:
        screen = pilot.app.screen
        item_list = screen.query_one("#personas-library-rows")
        item_list.focus()
        await settle(pilot)
        before = screen._import_intent_now()
        sort_before = screen.state.sort_key
        await pilot.press("s")
        await wait_until(
            pilot, lambda: screen.state.sort_key != sort_before, what="the sort"
        )
        await settle(pilot)
        after_sort = screen._import_intent_now()
        await screen._apply_tag_filter("hero")
        await settle(pilot)
        assert after_sort != before
        assert screen._import_intent_now() != after_sort
