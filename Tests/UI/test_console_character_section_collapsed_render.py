"""TASK-35970: a collapsed Character section renders nothing while a send refreshes it.

Every send ends with a Character refresh: the controller publishes REFRESHING
(``loading=True``) and then the loaded result. ``ConsoleCharacterContext.
sync_state`` recomposed the whole browser, search Input included, for each
snapshot, whether or not the section was expanded. Textual answers the
recompose's layout requests with a whole-screen reflow: two per send window,
8.4-74 ms each, measured live on 2026-10-20 (160x45, a 36-turn chat).

In the mounted harness the reads cannot open the ``:memory:`` database, so
every Character check errors and publishes REFRESHING, then the error. Measured
on dev 0d7e418db1 with the section collapsed: two recomposes per check, eight
layout requests inside the section from Enter to the first chunk, sixteen over
a 2.5 s held stream and twenty-four after the reply ended.

Live on dev 0d7e418db1 (2026-10-26, 160x45, Anthropic claude-haiku-4-5, a
plain chat, 12 sends): two Character recomposes and eight layout requests
inside the collapsed section per send.

The first case drives a real send through the real Console with the section
collapsed. The second drives the check a send's end runs over a Character
controller that loads real groups, then opens the section and reads what it
shows -- the chats the send produced, with no Refreshing line, in every
painted frame -- and checks the open still reveals the section. The third
hides the browser under another ancestor (as the hidden rail does) and shows
it again. The last pins the one other request a collapsed send window made
live: the avatar painter's first write to the never-shown name caption.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static

# The UI fixture imports the app after selecting a per-test profile. Import it
# under the collection bootstrap first, before config freezes its source owner.
import tldw_chatbook.app  # noqa: F401
from Tests.private_profile import private_profile_test
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationGroup,
    CharacterConversationRow,
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
)
from tldw_chatbook.UI.Console_Modules.character_context import (
    ConsoleCharacterContextState,
    ConsoleCharacterOperationPhase,
    ConsoleCharacterScopeFingerprint,
    character_presentation_work,
)
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.Widgets.Console.console_character_context import (
    CharacterConversationButton,
    ConsoleCharacterContext,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

_SECTION_BODY = "#console-rail-section-body-character"


class _Renders:
    """Count the browser's recomposes and every layout request in the section."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.recomposes: list[ConsoleCharacterContextState] = []
        self.layouts: list[Widget] = []
        self.body: Widget | None = None
        self.active = False
        recompose = ConsoleCharacterContext.recompose
        refresh = Widget.refresh

        async def counted_recompose(widget):
            if self.active:
                self.recomposes.append(widget.state)
            return await recompose(widget)

        def counted_refresh(
            widget, *regions, repaint=True, layout=False, recompose=False
        ):
            body = self.body
            if (
                self.active
                and (layout or recompose)
                and body is not None
                and (widget is body or body in widget.ancestors)
            ):
                self.layouts.append(widget)
            return refresh(
                widget, *regions, repaint=repaint, layout=layout, recompose=recompose
            )

        monkeypatch.setattr(ConsoleCharacterContext, "recompose", counted_recompose)
        monkeypatch.setattr(Widget, "refresh", counted_refresh)

    def start(self, body: Widget) -> None:
        self.body = body
        self.recomposes.clear()
        self.layouts.clear()
        self.active = True

    def stop(self) -> None:
        self.active = False

    def names(self) -> list[str]:
        return [
            f"{type(w).__name__}#{w.id}" if w.id else type(w).__name__
            for w in self.layouts
        ]


def _observe_publications(
    monkeypatch, controller
) -> list[ConsoleCharacterContextState]:
    """Record each snapshot the controller hands the screen.

    Wraps the screen callback, not ``_publish``: the stock Character route
    checks its own methods by identity, and a wrapped ``_publish`` would move
    the check onto another route.
    """
    published: list[ConsoleCharacterContextState] = []
    forward = controller._state_changed

    def observe(state: ConsoleCharacterContextState) -> None:
        published.append(state)
        forward(state)

    monkeypatch.setattr(controller, "_state_changed", observe)
    return published


@private_profile_test
async def test_send_with_the_character_section_collapsed_renders_nothing_there(
    monkeypatch, request
) -> None:
    from Tests.UI.test_console_composer_run_controls import (
        _held_run_host,
        _mounted,
        _start_held_run,
        _wait_for,
    )
    from Tests.UI.test_console_left_rail import _click_rail_toggle

    renders = _Renders(monkeypatch)
    gateway, host = _held_run_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        body = console.query_one(_SECTION_BODY)
        browser = console.query_one(ConsoleCharacterContext)
        controller = console._character_context
        assert body.display is False, "precondition: Character starts collapsed"
        published = _observe_publications(monkeypatch, controller)
        try:
            renders.start(body)
            await _start_held_run(console, composer, pilot)
            gateway.release.set()
            await _wait_for(
                pilot,
                lambda: not console._console_run_active(),
                "the run never ended",
                timeout=10.0,
            )
            # The reply's end-of-run syncs check the Character scope again.
            await pilot.pause(1.0)
            renders.stop()
            assert any(
                state.phase is ConsoleCharacterOperationPhase.REFRESHING
                for state in published
            ), "the send published no Character refresh; nothing was measured"
            assert renders.recomposes == [], (
                f"{len(renders.recomposes)} Character recomposes while collapsed, "
                f"phases {[state.phase.value for state in renders.recomposes]}"
            )
            assert renders.names() == [], (
                f"{len(renders.layouts)} layout requests inside the collapsed "
                f"Character section: {renders.names()}"
            )

            # Opening it shows the newest snapshot, not a deferred Refreshing.
            await _click_rail_toggle(pilot, "character")

            def settled() -> bool:
                state = controller.state
                if state.phase is not ConsoleCharacterOperationPhase.IDLE:
                    return False
                if browser.query("#console-character-loading"):
                    return False
                errors = browser.query("#console-character-error")
                return bool(errors) == bool(state.error) and (
                    not errors or str(errors.first(Static).content) == state.error
                )

            await _wait_for(
                pilot,
                settled,
                lambda: (
                    f"the opened section shows {[type(w).__name__ + '#' + str(w.id) for w in browser.children]} "
                    f"for {controller.state.phase.value!r} {controller.state.error!r}"
                ),
            )
        finally:
            gateway.release.set()
            await pilot.pause(0.2)


def _row(revision: int, n: int) -> CharacterConversationRow:
    key = ResolvedLocalCharacterKey("authority", 1)
    return CharacterConversationRow.resolved(
        LocalCharacterConversationTarget(key, f"r{revision}-{n}"),
        character_label="Ada",
        title=f"Chat r{revision}-{n}",
        last_modified="2026-10-20T12:00:00Z",
        created_at="2026-10-01T00:00:00Z",
    )


def _install_loading_controller(monkeypatch, revision: list[int]) -> None:
    """Give the next mounted Console a Character controller that loads groups.

    The harness's own reads cannot open the ``:memory:`` database, so every
    check errors. This controller loads one group whose rows follow
    ``revision``: two chats at revision 7, three after it moves.
    """
    from Tests.UI.test_console_character_context import _controller, _Service
    from tldw_chatbook.UI.Console_Modules.wiring import (
        _sync_character_context_presentation,
    )

    database = SimpleNamespace(
        get_local_authority_id=lambda: "authority",
        get_character_conversation_search_revision=lambda: revision[0],
    )

    class Service(_Service):
        def recent_groups(self, *, group_limit: int, row_limit: int):
            # The send wrote to the current chat: one more, newest first.
            count = 2 if revision[0] == 7 else 3
            rows = tuple(_row(revision[0], n) for n in range(count))
            key = ResolvedLocalCharacterKey("authority", 1)
            return (CharacterConversationGroup(key, "Ada", rows, count, True),)

    original_build = chat_screen_module.build_console_controllers

    def build(screen, *args, **kwargs):
        original_build(screen, *args, **kwargs)
        screen._character_context = _controller(
            database_accessor=lambda: database,
            service_factory=Service,
            state_changed=lambda state: _sync_character_context_presentation(
                screen, state
            ),
        )

    monkeypatch.setattr(chat_screen_module, "build_console_controllers", build)


async def _collapse_loaded_character_section(pilot) -> Widget:
    """Wait for the groups to load, collapse the section and return its body."""
    from Tests.UI.test_console_composer_run_controls import _wait_for
    from Tests.UI.test_console_left_rail import _click_rail_toggle

    screen = pilot.app.screen
    controller = screen._character_context
    await _wait_for(
        pilot, lambda: controller.state.groups, "the Character groups never loaded"
    )
    body = screen.query_one(_SECTION_BODY)
    if body.display:
        await _click_rail_toggle(pilot, "character")
    assert body.display is False, "precondition: Character is collapsed"
    await pilot.pause(0.2)
    return body


async def test_opening_after_a_collapsed_refresh_shows_its_chats_without_refreshing(
    monkeypatch,
) -> None:
    from Tests.UI.test_console_composer_run_controls import _wait_for
    from Tests.UI.test_console_left_rail import make_console_pilot

    revision = [7]
    _install_loading_controller(monkeypatch, revision)
    renders = _Renders(monkeypatch)
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        controller = screen._character_context
        body = await _collapse_loaded_character_section(pilot)
        published = _observe_publications(monkeypatch, controller)

        renders.start(body)
        revision[0] = 8
        # The check every Console sync runs, including the one ending a send.
        assert await character_presentation_work(controller, screen) is True
        await pilot.pause(0.3)
        renders.stop()
        phases = [state.phase for state in published]
        # A mount-time check can still land in the window; the property is
        # about whatever is published while collapsed, Refreshing included.
        assert ConsoleCharacterOperationPhase.REFRESHING in phases, phases
        assert phases[-1] is ConsoleCharacterOperationPhase.IDLE, phases
        assert controller.state.data_revision == 8
        assert renders.recomposes == [], (
            f"{len(renders.recomposes)} Character recomposes while collapsed"
        )
        assert renders.names() == [], renders.names()

        # Every frame painted from here on: the rows the compositor laid out.
        painted: list[list[str]] = []
        paint = Screen._compositor_refresh

        def recorded_paint(painting: Screen) -> None:
            if painting is screen:
                painted.append(
                    [
                        str(widget.label)
                        for widget in painting._compositor.widgets
                        if isinstance(widget, CharacterConversationButton)
                    ]
                )
            return paint(painting)

        monkeypatch.setattr(Screen, "_compositor_refresh", recorded_paint)
        # A click on the disclosure, in the order the live app runs it: the
        # rail activates the section (arming its reveal), then the screen
        # opens it. Pressing through the pilot lets this harness run one
        # queued allocation pass in between, which drops the reveal on dev
        # too; live, the open comes first.
        rail = screen.query_one(ConsoleLeftRail)
        rail.activate_section("character")
        screen._toggle_console_rail_section("character", next_open=True)
        browser = screen.query_one(ConsoleCharacterContext)

        def titles() -> list[str]:
            return [
                str(button.label) for button in browser.query(".console-character-row")
            ]

        await _wait_for(
            pilot,
            lambda: "Chat r8-2" in titles(),
            lambda: f"the opened section shows {titles()}",
        )
        assert [title.split(" · ")[0] for title in titles()] == [
            "Chat r8-0",
            "Chat r8-1",
            "Chat r8-2",
        ]
        assert not browser.query("#console-character-loading")
        # Not even one frame painted the chats from before the send.
        stale = [frame for frame in painted if any("Chat r7-" in t for t in frame)]
        assert stale == [], f"{len(stale)} of {len(painted)} frames painted old chats"
        assert any("Chat r8-2" in title for frame in painted for title in frame)
        # And the open reveals the section as it did before the render was
        # deferred: no allocation pass measured the half-mounted body.
        outer = screen.query_one("#console-left-rail-body")
        await _wait_for(
            pilot,
            lambda: all(
                outer.can_view_entire(row)
                for row in browser.query(".console-character-row")
            ),
            lambda: (
                f"rows below the fold: scroll_y={outer.scroll_y} of "
                f"{outer.max_scroll_y}, active section {rail._active_section_id!r}"
            ),
        )
        assert rail._active_section_id == "character"


async def test_return_from_roleplay_keeps_focus_on_search_over_a_deferred_render(
    monkeypatch,
) -> None:
    """Coming back from Roleplay or Library lands on the Character search.

    The return opens a collapsed section and focuses its search Input at once.
    When a snapshot was waiting to be drawn, that open starts the deferred
    render, which pruned the focused Input and restored nothing: keyboard
    users came back with nothing focused.
    """
    from Tests.UI.test_console_composer_run_controls import _wait_for
    from Tests.UI.test_console_left_rail import make_console_pilot

    revision = [7]
    _install_loading_controller(monkeypatch, revision)
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        controller = screen._character_context
        await _collapse_loaded_character_section(pilot)
        browser = screen.query_one(ConsoleCharacterContext)
        revision[0] = 8
        assert await character_presentation_work(controller, screen) is True
        await pilot.pause(0.3)
        assert browser.state.data_revision == 8, "precondition: a newer snapshot"

        # The anchor a Roleplay/Library "Back to Console" return leaves.
        screen._pending_character_return_focus_id = "console-context-character"
        screen._workspace.restore_character_navigation_focus()
        assert screen._pending_character_return_focus_id is None

        def titles() -> list[str]:
            return [
                str(button.label) for button in browser.query(".console-character-row")
            ]

        await _wait_for(
            pilot,
            lambda: any("Chat r8-2" in title for title in titles()),
            lambda: f"the opened section shows {titles()}",
        )
        await pilot.pause(0.3)
        focused = pilot.app.focused
        assert getattr(focused, "id", None) == "console-character-search", (
            f"focus after the return: {focused!r}"
        )
        assert focused.is_attached and browser in focused.ancestors


class _HiddenRailApp(App):
    """The browser and its identity line under one hidable ancestor."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self.identity = Static("")
        self.holder = Vertical(id="holder")

    def compose(self) -> ComposeResult:
        with self.holder:
            yield self.identity
            browser = ConsoleCharacterContext(
                self.controller, identity_state=self.identity
            )
            self.controller._state_changed = browser.sync_state
            yield browser


async def test_a_hidden_ancestor_defers_the_render_until_the_browser_shows(
    monkeypatch,
) -> None:
    """Any hidden ancestor (the whole rail, say) defers; showing renders the newest."""
    from Tests.UI.test_console_character_context import _controller, _groups

    renders = _Renders(monkeypatch)
    controller = _controller()
    app = _HiddenRailApp(controller)
    async with app.run_test(size=(72, 35)) as pilot:
        browser = app.query_one(ConsoleCharacterContext)
        for _ in range(40):  # The mount's own refresh settles first.
            await pilot.pause(0.02)
            if controller.state.groups and browser.query(".console-character-group"):
                break
        identity_before = str(app.identity.content)
        assert identity_before == "Character 1 · Local · No open chat"
        app.holder.display = False
        await pilot.pause()
        renders.start(app.holder)
        # Two of the four groups the mount loaded: a visibly different list.
        loaded = ConsoleCharacterContextState(
            groups=_groups()[1:3],
            expanded_key=_groups()[1].key,
            data_revision=7,
            scope_fingerprint=ConsoleCharacterScopeFingerprint(
                database_identity=1,
                data_authority_id="authority",
                data_revision=7,
                current_character_id=1,
                current_character_label="Ada",
                open_conversation_id="1-0",
            ),
        )
        controller._publish(
            replace(
                loaded,
                phase=ConsoleCharacterOperationPhase.REFRESHING,
                loading=True,
            )
        )
        controller._publish(loaded)
        await pilot.pause()
        renders.stop()
        assert browser.state == loaded
        assert renders.recomposes == []
        assert renders.names() == []
        assert str(app.identity.content) == identity_before

        app.holder.display = True
        for _ in range(40):
            await pilot.pause(0.02)
            if len(browser.query(".console-character-group")) == 2:
                break
        assert len(browser.query(".console-character-group")) == 2
        assert len(browser.query(".console-character-row")) == 5
        assert not browser.query("#console-character-loading")
        assert str(app.identity.content) == "Ada · Local · Open"


class _CaptionApp(App):
    """The avatar painter's two caption Statics, as the rail composes them."""

    def compose(self) -> ComposeResult:
        name = Static("Roleplay character", id="console-character-name", markup=False)
        name.display = False  # A hidden seam, never shown (left_rail compose).
        yield name
        yield Static(
            "Reaction: Automatic", id="console-character-reaction-state", markup=False
        )


async def test_hidden_avatar_caption_takes_new_copy_without_a_layout(
    monkeypatch,
) -> None:
    """The painter's first write lands on a send; a hidden caption needs no layout.

    Live, the only Character-section layout request left in a collapsed send
    window was this one: the screen composes the hidden caption with
    "Roleplay character", and the avatar painter's first paint of a session
    (on its first send) writes "No character in this chat".
    """
    layouts: list[Widget] = []
    refresh = Widget.refresh

    def counted(widget, *regions, repaint=True, layout=False, recompose=False):
        if layout or recompose:
            layouts.append(widget)
        return refresh(
            widget, *regions, repaint=repaint, layout=layout, recompose=recompose
        )

    app = _CaptionApp()
    async with app.run_test(size=(40, 6)) as pilot:
        await pilot.pause()
        name = app.query_one("#console-character-name", Static)
        reaction = app.query_one("#console-character-reaction-state", Static)
        monkeypatch.setattr(Widget, "refresh", counted)
        ConsoleLeftRail.sync_character_caption(app.screen, None, None)
        assert str(name.content) == "No character in this chat"
        assert name not in layouts
        # Positive control: the shown Reaction line still lays out a change.
        ConsoleLeftRail.sync_character_caption(app.screen, None, "Happy")
        assert reaction in layouts
