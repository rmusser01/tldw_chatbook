"""Roleplay frame test harness: two styled tiers, the size matrix, painted geometry.

Spec section 5.7.1 (Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md):

- ``RoleplayMockApp`` is the delegating ``PersonasTestApp`` that used to live in
  ``test_personas_workbench.py`` (moved verbatim; both old modules re-export
  it under the old names). It is the UNSTYLED tier: ``ConsolidatedCSSApp``
  loads no app bundle, so unstyled tests never assert geometry.
- ``StyledRoleplayMockApp`` is styled tier 1: the boot bundle plus every lazy
  split sheet (``APP_STYLESHEETS``, derived from the build's own
  ``SCREEN_OWNED_SPLITS``), so it carries ``screen_feature_roleplay.tcss``.
- ``painted_rows``, ``click_meta_cells``, ``settle`` and ``wait_until`` (moved
  here from ``test_roleplay_hostile_names.py``, TASK-34400) and
  ``seed_mock_characters`` are the one copy every Roleplay test imports.
- ``roleplay_full_app()`` is styled tier 2: a real ``TldwCli`` that reaches
  Roleplay through a real route (the initial tab, or Ctrl+4 from Home), so
  ``TldwCli._ensure_screen_owned_css`` loads the Roleplay sheet exactly as it
  does for a user. Since frame slice B2a both styled tiers take
  ``character_db=``, a file-backed ChaChaNotes (``open_volume_db`` copies a
  session template of 348 or 5,000 seeded characters), so they prove
  persistence as spec 5.7.1 asks; tests without one still seed through the
  ``ccp_character_handler`` seams (``seed_mock_characters``).

A real ``TldwCli`` that pushes ``PersonasScreen`` itself skips
``_ensure_screen_owned_css`` and paints the inline header without its rules
(TASK-32187's Watchlists trap; ``Tests/UI/full_app_destination_context.py``).
Reach Roleplay by navigation, as ``roleplay_full_app`` does, or call
``app._ensure_screen_owned_css("personas")`` before the push;
``Tests/UI/test_roleplay_stylesheet.py`` scans the suite for the bare push.

Geometry is asserted relative to the MEASURED nav bar and header, never as
absolute rows (spec 5.7.2 item 1; ADR-210's compact nav moves rows below 35).
Not collected by pytest (the file name has no ``test_`` prefix).
"""

from __future__ import annotations

import asyncio
import inspect
import shutil
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from textual.content import Content
from textual.css.stylesheet import CssSource
from textual.geometry import Region
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widget import Widget

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.config as config_module
import tldw_chatbook.UI.Persona_Modules.roleplay_frame_state as frame_state
from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import APP_STYLESHEETS, CSS_DIR, ConsolidatedCSSApp
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import RoleplayItemList
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph, set_ascii_glyph_mode

#: The lazily loaded Roleplay sheet (build_css.SCREEN_OWNED_SPLITS).
ROLEPLAY_SHEET = CSS_DIR / "screen_feature_roleplay.tcss"

#: Character counts the B2a volume tests seed (spec 5.2 B2a): the volume
#: profile's 348 and the 5,000-row latency copy.
VOLUME_SIZES: tuple[int, ...] = (348, 5000)
#: The character spec 1.6's 4-key find opens (``ctrl+f isolde ⏎ ⏎``).
ISOLDE = "Captain Isolde Varga"
#: Session templates by (pytest basetemp, count); see ``volume_db_path``.
_VOLUME_TEMPLATES: dict[tuple[str, int], Path] = {}


class NotYetDelivered(Exception):
    """A journey step a later frame slice delivers (spec 5.7.3, RC-7).

    A full-journey test is ``xfail(strict=True, raises=NotYetDelivered)`` and
    raises this explicitly at each undelivered step, so a regression in a
    delivered step fails the test instead of hiding in a plain ``xfail``. The
    slice that delivers the last step removes the marker.
    """


def not_yet_delivered(step: str, owner: str) -> None:
    """Raise ``NotYetDelivered`` for one journey step.

    Args:
        step: What the step does (``"/ focuses the items search"``).
        owner: The frame slice that delivers it (``"B3"``).

    Raises:
        NotYetDelivered: Always.
    """
    raise NotYetDelivered(f"{step} (frame slice {owner})")


#: Spec 5.7.1 geometry matrix for B1: 80x24 is the degrade check, the other
#: three are the design centre. 60x24 joins with B7: B1's worst-case header
#: (every chip, a server, the longest kind) fits from 65 columns, and the
#: spec's below-64 layout is B7's and TASK-33910.24's.
ROLEPLAY_SIZES: tuple[tuple[int, int], ...] = (
    (80, 24),
    (120, 36),
    (160, 45),
    (220, 55),
)

#: The size matrix as a parametrize mark: ``@size_matrix`` runs a test once
#: per B1 size with a ``roleplay_size`` argument (ids ``80x24`` ...). A mark,
#: not a fixture, so test modules need no fixture import.
size_matrix = pytest.mark.parametrize(
    "roleplay_size", ROLEPLAY_SIZES, ids=lambda size: f"{size[0]}x{size[1]}"
)


class RoleplayMockApp(ConsolidatedCSSApp):
    """The unstyled tier: today's delegating ``PersonasTestApp``, moved verbatim."""

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


class StyledRoleplayMockApp(RoleplayMockApp):
    """Styled tier 1: the boot bundle plus every lazy split sheet (spec 5.7.1)."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]


#: The names the Roleplay test modules have always used (re-exported there).
PersonasTestApp = RoleplayMockApp
StyledPersonasTestApp = StyledRoleplayMockApp


def seed_mock_characters(
    monkeypatch: pytest.MonkeyPatch, records: list[dict[str, Any]]
) -> None:
    """Route the screen's character seams over ``records`` (both tiers).

    The same seams ``test_personas_workbench.stub_characters`` patches:
    ``fetch_all_characters``/``fetch_character_by_id`` plus the paged loader.

    Args:
        monkeypatch: The test's monkeypatch.
        records: Character dicts with at least ``id`` and ``name``.
    """
    import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as handler
    from Tests.UI.test_personas_dictionaries import patch_character_paging

    monkeypatch.setattr(
        handler, "fetch_all_characters", lambda: [dict(r) for r in records]
    )
    monkeypatch.setattr(
        handler,
        "fetch_character_by_id",
        lambda character_id: next(
            (dict(r) for r in records if str(r["id"]) == str(character_id)), None
        ),
    )
    patch_character_paging(monkeypatch)


def volume_character(index: int) -> dict[str, Any]:
    """One deterministic seeded card: a unique name, a tag and a description.

    Index 0 is ``ISOLDE`` (spec 1.6's 4-key find); the rest are
    ``Character 0001`` ... in name order, so row ``n`` of the name-sorted list
    is ``Character {n:04d}`` (row 0 is Isolde, the built-in Default Assistant
    sorts last).
    """
    name = ISOLDE if index == 0 else f"Character {index:04d}"
    return {
        "name": name,
        "description": f"Seeded volume character number {index}.",
        "tags": [("fantasy", "sci-fi", "noir")[index % 3]],
    }


def volume_db_path(tmp_path_factory: pytest.TempPathFactory, count: int) -> Path:
    """A closed ChaChaNotes file holding ``count`` seeded characters (plus the built-in one).

    Seeded once per test session through the app's own ``add_character_card``
    inside one transaction (measured: 0.4 s for 348, 6.3 s for 5,000), then
    copied per test by ``open_volume_db`` (about 0.13 s at 5,000). Closing
    checkpoints the WAL, so the template is one file.

    Args:
        tmp_path_factory: pytest's session-scoped factory.
        count: Characters to seed.

    Returns:
        The template's path.
    """
    key = (str(tmp_path_factory.getbasetemp()), count)
    if key not in _VOLUME_TEMPLATES:
        path = tmp_path_factory.mktemp(f"roleplay-volume-{count}") / "template.db"
        db = CharactersRAGDB(path, "roleplay-b2a-volume")
        try:
            with db.transaction():
                for index in range(count):
                    db.add_character_card(volume_character(index))
        finally:
            db.close_connection()
        _VOLUME_TEMPLATES[key] = path
    return _VOLUME_TEMPLATES[key]


def open_volume_db(tmp_path: Path, template: Path) -> CharactersRAGDB:
    """A per-test copy of a volume template, opened (the caller closes it)."""
    target = tmp_path / "chachanotes.db"
    shutil.copyfile(template, target)
    return CharactersRAGDB(target, "roleplay-b2a-test")


@contextmanager
def character_db_in_use(
    db: CharactersRAGDB, app_instance: Any = None
) -> Iterator[None]:
    """Route Roleplay's character and lore database reads and writes to ``db``.

    Roleplay resolves its character DB through ``config.get_chachanotes_db_lazy``
    at call time (``ccp_character_handler._default_character_db`` and
    ``Character_Chat_Lib._get_default_chachanotes_db``); lore reads
    ``app.chachanotes_db``. A ``:memory:`` DB cannot serve (one connection per
    thread: the screen's off-thread page reads would see an empty database).

    Args:
        db: A file-backed database (``open_volume_db`` or a test's own).
        app_instance: The mock tier's ``mock_app_instance``, whose
            ``chachanotes_db`` lore reads. Never a factory-built ``TldwCli``:
            its Actor Pack coordinator has no DB, and Roleplay's arrival
            would run recovery against it and fail the initial load.
    """
    if app_instance is not None:
        app_instance.chachanotes_db = db
    with patch.object(config_module, "get_chachanotes_db_lazy", lambda: db):
        yield


def _settings_for_full_app(ascii_glyphs: bool) -> Callable[..., Any]:
    """``get_cli_setting`` for the full-app tier: no splash, chosen glyph mode.

    The app reads ``splash_screen.enabled`` at compose and resets the glyph
    mode from ``appearance.ascii_glyphs`` at compose (app.py), so both must
    come from this patch, live for the whole ``run_test``. Measured while
    planning: arrival takes 7.7 s with the splash and 0.7 s without.
    """

    def settings(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        if section == "appearance" and key == "ascii_glyphs":
            return ascii_glyphs
        return default

    return settings


# wait_until, painted_rows, click_meta_cells and settle moved here unchanged
# from test_roleplay_hostile_names.py (TASK-34400) in B1: this harness is the
# one home of the Roleplay paint helpers.
async def wait_until(
    pilot: Pilot,
    predicate: Callable[[], bool],
    *,
    timeout: float = 20.0,
    what: str = "",
) -> None:
    """Poll ``predicate`` with a monotonic deadline.

    Args:
        pilot: The running test pilot.
        predicate: Returns True once the awaited state holds.
        timeout: Seconds to wait before failing.
        what: Names the awaited state in the failure message.

    Raises:
        AssertionError: ``predicate`` stayed False for ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"timed out after {timeout}s waiting for {what or predicate}")


def _first_list_item_present(app) -> bool:
    """Roleplay is up once its list shows a row, or its empty or recovery line."""
    screen = app.screen
    return isinstance(screen, PersonasScreen) and (
        bool(visible_item_ids(screen)) or list_state(screen).state != "rows"
    )


@dataclass(frozen=True)
class RoleplayItemView:
    """One loaded row of the items list, read back the way a test needs it.

    ``name``, ``state`` and ``meta`` are the row's data (``meta`` is line 2's
    scent, whether or not the density paints it); ``lines`` is what the prompt
    paints, gutter stripped, trailing blank lines dropped. ``marked`` is the
    list's mark state; ``painted_mark`` is what the row paints: the option's
    own prompt shows the resolved ``✓`` (``[x]`` in ASCII mode) in the
    gutter's mark column.
    """

    option_id: str
    kind: str
    item_id: str
    name: str
    state: str | None
    meta: str | None
    lines: tuple[str, ...]
    active: bool
    marked: bool
    painted_mark: bool
    disabled: bool


@dataclass(frozen=True)
class RoleplayListState:
    """``rows``, ``empty`` or ``recovery``, with the rows, the cursor and the status copy."""

    state: str
    items: tuple[RoleplayItemView, ...]
    highlighted: str | None
    text: str | None


def _item_list(node) -> RoleplayItemList:
    return node.query_one("#personas-library-rows", RoleplayItemList)


def item_at(node, index: int | str) -> RoleplayItemView:
    """The loaded row at ``index`` (0-based, list order), or by option id ``"<kind>:<id>"``.

    Args:
        node: The app, a screen or the pane (anything with ``query_one``).
        index: A row index or an option id.
    """
    item_list = _item_list(node)
    if isinstance(index, str):
        index = item_list.get_option_index(index)
    option = item_list.get_option_at_index(index)
    option_id = str(option.id)
    row = item_list.row(option_id)
    prompt = item_list.prompt_for(option_id)
    assert row is not None and prompt is not None, option_id
    gutter = len(prompt.lines[0][0][0])
    painted = ["".join(text for text, _style in line)[gutter:] for line in prompt.lines]
    while painted and not painted[-1].strip():
        painted.pop()
    # The prompt OptionList paints, not the list's state or its cached spans.
    shown = option.prompt
    shown_text = shown.plain if isinstance(shown, Content) else str(shown)
    mark_columns = frame_state.gutter_mark_columns()
    mark_cell = shown_text[mark_columns.start : mark_columns.stop]
    painted_mark = mark_cell == resolve_glyph("✓")
    kind, _, item_id = option_id.partition(":")
    return RoleplayItemView(
        option_id=option_id,
        kind=kind,
        item_id=item_id,
        name=row.name,
        state=row.state,
        meta=row.meta,
        lines=tuple(painted),
        active=item_list.active_id == option_id,
        marked=option_id in item_list.marked_ids(kind),
        painted_mark=painted_mark,
        disabled=option.disabled,
    )


def visible_item_ids(node) -> tuple[str, ...]:
    """Option ids of every loaded row, in list order (not only the viewport)."""
    return _item_list(node).loaded_ids()


def highlighted_item_id(node) -> str | None:
    """The cursor row's option id, or None."""
    return _item_list(node).highlighted_id()


def list_state(node) -> RoleplayListState:
    """The list's rows and cursor, and the empty or recovery copy shown beside it."""
    item_list = _item_list(node)
    items = tuple(item_at(node, index) for index in range(item_list.option_count))
    status = node.query_one("#personas-library-status")
    state, text = "rows", None
    if status.display and status.children:
        static = status.children[0]
        text = str(static.render())
        state = "empty" if static.id == "personas-library-empty" else "recovery"
    return RoleplayListState(state, items, highlighted_item_id(node), text)


def painted_option_rows(item_list: RoleplayItemList) -> dict[int, list[int]]:
    """The screen rows each option index is painted on (from the ``option`` style meta)."""
    region = item_list.content_region
    rows: dict[int, list[int]] = {}
    for y in range(region.y, region.bottom):
        style = item_list.screen.get_style_at(region.x, y)
        option = style.meta.get("option") if style is not None else None
        if option is not None:
            rows.setdefault(option, []).append(y)
    return rows


async def select_roleplay_item(
    pilot: Pilot,
    node,
    item_id: str | None = None,
    *,
    index: int | None = None,
    via: str = "enter",
    load: bool = True,
) -> None:
    """Move the cursor to a row and, by default, load it like a user (I6).

    Args:
        pilot: The running pilot.
        node: The app, screen or pane holding the list.
        item_id: Option id ``"<kind>:<id>"`` (or give ``index``).
        index: Row index when no id is given.
        via: ``"enter"`` (cursor + Enter's action) or ``"click"`` (a real mouse
            click on the row's name, which must be painted: an unpainted row
            fails loudly instead of clicking nothing, census finding F2).
        load: False only moves the cursor.
    """
    item_list = _item_list(node)
    if item_id is not None:
        index = item_list.get_option_index(item_id)
    if index is None:
        raise ValueError("select_roleplay_item needs item_id or index")
    item_list.highlighted = index
    await pilot.pause()
    if not load:
        return
    if via == "click":
        painted = painted_option_rows(item_list).get(index)
        assert painted, f"option {index} is not painted in {item_list.content_region}"
        name_column = len(frame_state.item_gutter(loaded=False, marked=False))
        content = item_list.content_region
        await pilot.click(
            item_list,
            offset=(
                content.x - item_list.region.x + name_column,
                painted[0] - item_list.region.y,
            ),
        )
    elif via == "enter":
        item_list.action_select()
    else:
        raise ValueError(f"unknown via {via!r}")
    await pilot.pause()


async def click_mark_column(pilot: Pilot, node, item_id: str) -> None:
    """A real mouse click on a painted row's mark column (gutter col 2, spec 4.10)."""
    item_list = _item_list(node)
    index = item_list.get_option_index(item_id)
    painted = painted_option_rows(item_list).get(index)
    assert painted, f"option {index} is not painted"
    column = frame_state.gutter_mark_columns().start
    await pilot.click(
        item_list,
        offset=(
            item_list.content_region.x - item_list.region.x + column,
            painted[0] - item_list.region.y,
        ),
    )
    await pilot.pause()


CHOOSER_OPTIONS_ID = "roleplay-items-chooser-options"


async def wait_for_chooser(pilot: Pilot):
    """Wait until Roleplay's choice modal (New, "Import as:") is open and focused.

    Found by its DOM id, not its class, so a test of dev's behaviour can run
    on a tree without the modal module (frame slice B2b).

    Returns:
        The modal's ``OptionList``.
    """

    def ready() -> bool:
        focused = pilot.app.focused
        return focused is not None and focused.id == CHOOSER_OPTIONS_ID

    await wait_until(pilot, ready, what="the choice modal")
    return pilot.app.focused


async def start_new(
    pilot: Pilot, choice_id: str | None = None, *, via: str = "click"
) -> None:
    """Open the New chooser and pick a variant like a user (frame slice B2b).

    On dev ``+ New`` and Ctrl+N created a blank item of the current kind at
    once; since B2b they open the New chooser on that kind's Blank, and Enter
    picks it.

    Args:
        pilot: The running pilot (Roleplay on screen).
        choice_id: The variant (``"character:portrait"``, ``"lore"``, ...);
            None picks the preselected Blank.
        via: ``"click"`` (a real click on ``+ New``), ``"press"``
            (``Button.press()``) or ``"ctrl+n"``.
    """
    if via == "click":
        await pilot.click("#personas-library-new")
    elif via == "press":
        pilot.app.screen.query_one("#personas-library-new").press()
    elif via == "ctrl+n":
        await pilot.press("ctrl+n")
    else:
        raise ValueError(f"unknown via {via!r}")
    options = await wait_for_chooser(pilot)
    if choice_id is not None:
        target = options.get_option_index(choice_id)
        for _ in range(options.option_count):
            if options.highlighted == target:
                break
            low = options.highlighted is None or options.highlighted < target
            await pilot.press("down" if low else "up")
        assert options.highlighted == target, f"{choice_id} cannot be highlighted"
    await pilot.press("enter")
    await pilot.pause()


def fake_file_picker(
    monkeypatch: pytest.MonkeyPatch, *picks: Path | str | None
) -> list[dict[str, Any]]:
    """Answer the shared file picker with ``picks``, like a user's choices (B2b).

    Roleplay imports ``EnhancedFileOpen`` inside its workers, so patching the
    module attribute replaces every picker it opens: a modal that dismisses
    with the next pick (the last one repeats) after its first refresh, as the
    Buddy review test's ``ChosenArchive`` does. The screen's code after the
    pick (the sniff, "Import as:", the review dialogs) runs for real.

    Args:
        monkeypatch: The test's monkeypatch.
        *picks: What each opened picker returns (None is Cancel).

    Returns:
        The keyword arguments of every picker constructed, in order (``title``,
        ``filters``, ``context``, ...).
    """
    from textual.screen import ModalScreen

    import tldw_chatbook.Widgets.enhanced_file_picker as picker_module

    opened: list[dict[str, Any]] = []
    queue = list(picks) or [None]

    class _PickedFile(ModalScreen[object]):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            opened.append(kwargs)
            self._pick = queue.pop(0) if len(queue) > 1 else queue[0]

        def on_mount(self) -> None:
            def pick() -> None:
                # Not awaited: awaiting dismiss() from this screen's own
                # message pump raises ScreenError.
                self.dismiss(self._pick)

            self.call_after_refresh(pick)

    monkeypatch.setattr(picker_module, "EnhancedFileOpen", _PickedFile)
    return opened


@asynccontextmanager
async def roleplay_full_app(
    *,
    size: tuple[int, int],
    entry: str = "initial_tab",
    ascii_glyphs: bool = False,
    notifications: bool = False,
    on_home: Callable[[Any], None] | None = None,
    character_db: CharactersRAGDB | None = None,
) -> AsyncIterator[Any]:
    """Styled tier 2: a real ``TldwCli`` arriving at Roleplay by a real route.

    With ``character_db`` the app reads and writes that database (frame slice
    B2a; spec 5.7.1); without it, seed through the character seams first
    (``seed_mock_characters``). Yields the pilot once the Roleplay screen shows
    its first list item.

    Args:
        size: Terminal ``(columns, rows)``.
        entry: ``"initial_tab"`` boots straight into Roleplay (the app's
            ``_push_initial_screen`` path); ``"ctrl+4"`` boots Home and presses
            Ctrl+4 (the in-app navigation path).
        ascii_glyphs: Run with ``appearance.ascii_glyphs`` on.
        notifications: Mount toasts (``run_test`` defaults to off).
        on_home: With ``entry="ctrl+4"``, called with the app while Home is
            showing, just before Ctrl+4 (to observe the pre-visit state).
        character_db: A file-backed ChaChaNotes to run against.
    """
    if entry not in ("initial_tab", "ctrl+4"):
        raise ValueError(f"unknown entry {entry!r}")
    app = _build_test_app(
        configured_default="personas" if entry == "initial_tab" else "home"
    )
    settings = _settings_for_full_app(ascii_glyphs)
    try:
        with (
            patch_app_global("get_cli_setting", side_effect=settings),
            (
                # Not ``app``: a factory-built app's Actor Pack coordinator was
                # wired without a DB, so setting ``app.chachanotes_db`` before
                # the visit makes recovery run against it and fail the initial
                # load (measured while planning B2a). Lore is mock-tier only.
                character_db_in_use(character_db)
                if character_db is not None
                else _no_patch()
            ),
        ):
            async with app.run_test(size=size, notifications=notifications) as pilot:
                await wait_until(
                    pilot,
                    lambda: getattr(app, "_initial_screen_pushed", False),
                    what="the initial screen",
                )
                if entry == "ctrl+4":
                    await wait_until(
                        pilot,
                        lambda: type(app.screen).__name__ == "HomeScreen",
                        what="Home",
                    )
                    if on_home is not None:
                        on_home(app)
                    await pilot.press("ctrl+4")
                await wait_until(
                    pilot,
                    lambda: _first_list_item_present(app),
                    what="Roleplay's list",
                )
                await settle(pilot)
                yield pilot
    finally:
        # The app set the process-wide glyph mode at compose; never leak it.
        set_ascii_glyph_mode(False)


@contextmanager
def _no_patch() -> Iterator[None]:
    yield


#: The two styled tiers (spec 5.7.1): ``"mock"`` (StyledRoleplayMockApp) and
#: ``"full"`` (roleplay_full_app). ``@styled_tiers`` gives a test a
#: ``styled_tier`` argument per tier.
STYLED_TIERS: tuple[str, ...] = ("mock", "full")
styled_tiers = pytest.mark.parametrize("styled_tier", STYLED_TIERS)


@asynccontextmanager
async def open_styled_roleplay(
    tier: str,
    mock_app_instance: Any,
    *,
    size: tuple[int, int],
    notifications: bool = False,
    ascii_glyphs: bool = False,
    app_class: type[StyledRoleplayMockApp] = StyledRoleplayMockApp,
    character_db: CharactersRAGDB | None = None,
) -> AsyncIterator[Any]:
    """Mount Roleplay under one styled tier and wait for its first list item.

    Args:
        tier: ``"mock"`` or ``"full"``.
        mock_app_instance: The ``Tests/UI/conftest.py`` fixture (mock tier only).
        size: Terminal ``(columns, rows)``.
        notifications: Mount toasts.
        ascii_glyphs: Run with the ASCII glyph mode on (restored afterwards).
        app_class: A ``StyledRoleplayMockApp`` subclass (mock tier only).
        character_db: A file-backed ChaChaNotes both tiers run against.
    """
    if tier == "full":
        async with roleplay_full_app(
            size=size,
            notifications=notifications,
            ascii_glyphs=ascii_glyphs,
            character_db=character_db,
        ) as pilot:
            yield pilot
        return
    if tier != "mock":
        raise ValueError(f"unknown tier {tier!r}")
    app = app_class(mock_app_instance)
    # The mock tier has no app compose that resets the process-wide mode.
    set_ascii_glyph_mode(ascii_glyphs)
    try:
        with (
            character_db_in_use(character_db, mock_app_instance)
            if character_db is not None
            else _no_patch()
        ):
            async with app.run_test(size=size, notifications=notifications) as pilot:
                await wait_until(
                    pilot, lambda: _first_list_item_present(app), what="Roleplay's list"
                )
                await settle(pilot)
                yield pilot
    finally:
        set_ascii_glyph_mode(False)


def chrome_bottoms(screen) -> tuple[int, int]:
    """``(nav bottom, header bottom)`` as measured, in screen rows (0-based)."""
    nav = screen.query_one(MainNavigationBar).region
    header = screen.query_one("#personas-header").region
    return nav.bottom, header.bottom


@dataclass(frozen=True)
class PaintedItem:
    """Where one option is painted (rows are option lines, not widgets)."""

    region: Region


def first_list_item(screen) -> PaintedItem:
    """The painted region of the first row of the items list."""
    item_list = _item_list(screen)
    rows = painted_option_rows(item_list).get(0)
    assert rows, "the first option is not painted"
    region = item_list.content_region
    return PaintedItem(Region(region.x, rows[0], region.width, len(rows)))


def painted_rows(screen: Screen) -> list[str]:
    """Every compositor row as plain text: what the terminal shows.

    Args:
        screen: The mounted screen whose compositor output is read.

    Returns:
        One string per terminal row, top to bottom, as currently painted.
    """
    return [strip.text for strip in screen._compositor.render_strips()]


def painted_text(screen, region: Region) -> str:
    """The plain text the compositor paints inside ``region`` (one line per row)."""
    rows = painted_rows(screen)
    return "\n".join(
        rows[y][region.x : region.right] for y in range(region.y, region.bottom)
    )


def assert_painted_inside(widget: Widget, pane: Widget) -> None:
    """Every cell of ``widget`` lies in ``pane``'s painted window and is not covered.

    The painted window is ``scrollable_content_region``, not the pane's outer
    region (lessons-testing-evidence, "painted window, not pane rectangle");
    and containment is not visibility, so the compositor's hit test must
    find ``widget`` (or a descendant) on its middle row (a docked sibling
    can cover a contained widget).
    """
    window = pane.scrollable_content_region
    region = widget.region
    assert region.width > 0 and region.height > 0, (
        f"{widget!r} paints nothing: {region}"
    )
    assert region.intersection(window) == region, (
        f"{widget!r} {region} escapes {pane!r}'s painted window {window}"
    )
    middle = region.y + (region.height - 1) // 2
    hit, _ = widget.screen.get_widget_at(region.x, middle)
    assert hit is widget or widget in hit.ancestors, (
        f"{widget!r} is covered by {hit!r} at ({region.x}, {middle})"
    )


def click_meta_cells(screen: Screen) -> list[tuple[int, int, str, str]]:
    """Every painted cell run that carries an ``@click`` action.

    Args:
        screen: The mounted screen whose compositor output is scanned.

    Returns:
        ``(x, y, text, action)`` for each painted segment whose style meta
        holds ``@click``; empty when no painted text is clickable markup.
    """
    hits = []
    for y, strip in enumerate(screen._compositor.render_strips()):
        x = 0
        for segment in strip:
            meta = segment.style.meta if segment.style is not None else {}
            if meta and "@click" in meta:
                hits.append((x, y, segment.text, meta["@click"]))
            x += segment.cell_length
    return hits


def drop_rule_from_loaded_sheet(app, sheet: Path, selector: str) -> None:
    """Delete one rule block from an already-loaded sheet and restyle the app.

    The executable form of the "delete one ``_roleplay.tcss`` header rule"
    discrimination check (spec B1): it edits the PARSED source the app holds,
    so it works the same whether the sheet arrived through a harness
    ``CSS_PATH`` or through the app's route loader.

    Args:
        app: A running app.
        sheet: The stylesheet file the app loaded.
        selector: The exact selector text heading the block to delete.

    Raises:
        KeyError: The app never loaded ``sheet`` (itself a finding).
    """
    key = (str(sheet), "")
    source = app.stylesheet.source[key]
    head = f"\n{selector} {{"
    start = source.content.index(head)
    end = source.content.index("}", start) + 1
    mutated = source.content[:start] + source.content[end:]
    app.stylesheet.source[key] = CssSource(
        mutated, source.is_defaults, source.tie_breaker, source.scope
    )
    app.stylesheet.reparse()
    app.stylesheet.update(app)


async def settle(pilot: Pilot) -> None:
    """Let the screen's own workers finish, then repaint twice.

    Only workers owned by the current screen: the full app runs app-wide
    workers that never finish, so ``app.workers.wait_for_complete()`` would
    wait forever there.

    Args:
        pilot: The running test pilot.
    """
    await pilot.pause()
    screen = pilot.app.screen
    unfinished = [
        worker
        for worker in pilot.app.workers
        if screen in worker.node.ancestors_with_self and not worker.is_finished
    ]
    if unfinished:
        await pilot.app.workers.wait_for_complete(unfinished)
    await pilot.pause()
    await asyncio.sleep(0)
    await pilot.pause()
