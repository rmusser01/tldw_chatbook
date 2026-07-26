import pytest

from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.widgets import Button, Input, ListView, Static

from tldw_chatbook.UI.Chatbooks_Window_Improved import (
    ChatbookCard,
    ChatbooksWindowImproved,
    EmptyStateWidget,
)
from tldw_chatbook.UI.Screens.chatbooks_screen import ChatbooksScreen


@pytest.mark.asyncio
async def test_chatbooks_screen_uses_improved_window(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksScreenApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksScreen(self)

    app = ChatbooksScreenApp()
    async with app.run_test():
        assert app.screen.query_one(ChatbooksWindowImproved) is not None
        # The master shell's top nav lists owning destinations; chat lives
        # under Console and chatbooks is reachable via the palette, so the
        # shared escape is the Home/Console pair rather than per-route links.
        assert app.screen.query_one("#nav-home") is not None
        assert app.screen.query_one("#nav-console") is not None


@pytest.mark.asyncio
async def test_chatbooks_empty_state_explains_portable_context_and_escape(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksScreenApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksScreen(self)

    app = ChatbooksScreenApp()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()

        empty_state = app.screen.query_one(EmptyStateWidget)
        empty_text = "\n".join(
            str(widget.render()) for widget in empty_state.query(Static)
        )
        empty_buttons = [button.label.plain for button in empty_state.query(Button)]

        assert "portable knowledge packs" in empty_text
        assert "sessions" in empty_text
        assert "machines" in empty_text
        assert "teams" in empty_text
        assert "conversations" in empty_text
        assert "notes" in empty_text
        assert "characters/personas" in empty_text
        assert "prompts" in empty_text
        assert "media" in empty_text
        assert "seed Chat" in empty_text
        assert "shared navigation" in empty_text
        assert "return to Chat" in empty_text
        assert any("Create Local Pack" in label for label in empty_buttons)
        assert any("Import Local Pack" in label for label in empty_buttons)
        assert app.screen.query_one("#nav-home") is not None
        assert app.screen.query_one("#nav-console") is not None


@pytest.mark.asyncio
async def test_improved_window_exposes_server_action_cards(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksWindowApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksWindowImproved(self)

    app = ChatbooksWindowApp()
    async with app.run_test():
        window = app.query_one(ChatbooksWindowImproved)
        assert window.query_one("#create-server-action") is not None
        assert window.query_one("#import-server-action") is not None


@pytest.mark.asyncio
async def test_chatbooks_view_toggles_explain_grid_and_list_modes(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksWindowApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksWindowImproved(self)

    app = ChatbooksWindowApp()
    async with app.run_test():
        window = app.query_one(ChatbooksWindowImproved)
        grid_button = window.query_one("#view-grid", Button)
        list_button = window.query_one("#view-list", Button)

        assert str(grid_button.tooltip) == "Show chatbooks as visual cards."
        assert str(list_button.tooltip) == "Show chatbooks as a dense text list."


def _sample_chatbooks(count: int = 2):
    return [
        {
            "name": f"Chatbook {i}",
            "description": f"Sample chatbook number {i}",
            "size_mb": 1.5 * i,
            "created_at": "2026-01-0%d" % (i + 1),
            "statistics": {"conversations": i, "notes": i, "characters": i},
            "tags": ["sample"],
        }
        for i in range(1, count + 1)
    ]


# -- task-671: _update_content() used to mount a local Grid/ListView's
# children (`grid.mount(card)` / `list_view.mount(item)`) BEFORE the Grid/
# ListView itself was mounted into the already-attached `#chatbooks-
# container`. `Widget.mount()` raises `MountError` synchronously whenever
# `not self.is_attached`, so any non-empty `chatbooks` list crashed the
# `watch_chatbooks` -> `_update_content()` path; only the empty-state branch
# (which mounts a single `EmptyStateWidget` directly into the attached
# container) avoided it. Regression coverage for both the grid and list
# render paths.


@pytest.mark.asyncio
async def test_update_content_renders_multiple_chatbooks_grid_view_without_mount_error(
    monkeypatch,
):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksWindowApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksWindowImproved(self)

    app = ChatbooksWindowApp()
    async with app.run_test() as pilot:
        window = app.query_one(ChatbooksWindowImproved)
        assert window.view_mode == "grid"

        # This assignment used to raise MountError synchronously (inside the
        # `watch_chatbooks` -> `_update_content()` call chain) before the
        # fix.
        window.chatbooks = _sample_chatbooks(2)

        cards = window.query(ChatbookCard)
        assert len(cards) == 2

        # Cards are attached synchronously by `mount()`, but each card's own
        # `compose()` (its title/description children) only runs once the
        # message pump processes the queued mount -- pump once before
        # inspecting their content.
        await pilot.pause()
        rendered_names = {
            str(card.query_one(".chatbook-card-title", Static).render())
            for card in cards
        }
        assert rendered_names == {"Chatbook 1", "Chatbook 2"}

        # The `chatbooks` reactive is declared `recompose=True`; make sure
        # the cards survive the deferred recompose tick and aren't wiped
        # back out to an empty container.
        await pilot.pause()
        await pilot.pause()
        cards_after_recompose = window.query(ChatbookCard)
        assert len(cards_after_recompose) == 2, (
            "cards were wiped out by the deferred recompose() teardown -- "
            f"found {len(cards_after_recompose)} ChatbookCard widgets after "
            "pumping the message loop"
        )


@pytest.mark.asyncio
async def test_update_content_renders_multiple_chatbooks_list_view_without_mount_error(
    monkeypatch,
):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksWindowApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksWindowImproved(self)

    app = ChatbooksWindowApp()
    async with app.run_test() as pilot:
        window = app.query_one(ChatbooksWindowImproved)
        window.view_mode = "list"
        await pilot.pause()

        # This assignment used to raise MountError synchronously via
        # `list_view.mount(item)` before `list_view` was itself mounted.
        window.chatbooks = _sample_chatbooks(3)

        list_view = window.query_one(ListView)
        assert len(list_view.children) == 3


@pytest.mark.asyncio
async def test_server_create_action_uses_server_mode(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksWindowApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksWindowImproved(self)

    app = ChatbooksWindowApp()
    async with app.run_test():
        window = app.query_one(ChatbooksWindowImproved)
        recorded = {}

        async def fake_action_create_chatbook(execution_mode="local"):
            recorded["mode"] = execution_mode

        window.action_create_chatbook = fake_action_create_chatbook
        await window.action_create_chatbook_server()

        assert recorded["mode"] == "server"


@pytest.mark.asyncio
async def test_server_import_action_uses_server_mode(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksWindowApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksWindowImproved(self)

    app = ChatbooksWindowApp()
    async with app.run_test():
        window = app.query_one(ChatbooksWindowImproved)
        recorded = {}

        async def fake_action_import_chatbook(execution_mode="local"):
            recorded["mode"] = execution_mode

        window.action_import_chatbook = fake_action_import_chatbook
        await window.action_import_chatbook_server()

        assert recorded["mode"] == "server"


# -- task-637: mouse-capture guard for widget-level (non-screen) recompose --
# `chatbooks` was a `reactive([], recompose=True)` field when this guard
# landed; task-671 dropped the recompose flag (the deferred recompose wiped
# the watcher-built cards), so these tests drive `refresh(recompose=True)`
# directly. `ChatbooksWindowImproved` is a `Screen` subclass but is
# embedded as a plain child widget of `ChatbooksScreen` (a `BaseAppScreen`),
# not pushed via the screen stack -- it never inherited task-627's guard.
# The tests below drive `refresh(recompose=True)` directly rather than
# through `chatbooks =` -- a non-empty `chatbooks` list hits an unrelated,
# pre-existing bug in `_update_content()` (`grid.mount(card)`/
# `list_view.mount(item)` called before `grid`/`list_view` is itself
# mounted); calling `refresh(recompose=True)` exercises the exact same
# `Widget.recompose()` teardown/remount path task-637 guards, without
# tripping that unrelated bug.


@pytest.mark.asyncio
async def test_post_recompose_sweep_releases_a_capture_dispatched_during_the_teardown_drain(
    monkeypatch,
):
    """Residual-window regression (mirrors ``test_post_recompose_sweep_
    releases_a_capture_dispatched_during_the_teardown_drain`` in
    ``test_settings_rag_profile_region.py``, the task-627 code-review finding
    for ``BaseAppScreen``): a capture that lands on the VICTIM's own message
    pump -- queued before the recompose's pre-teardown release even ran, but
    processed DURING ``super().recompose()``'s own ``remove()`` drain -- must
    still be swept once the recompose fully completes.

    Reproduced deterministically with ``call_later`` on the victim's own
    pump, mechanism-equivalent to a forwarded ``MouseDown`` whose dispatch is
    still pending on the widget's pump when the enclosing recompose begins.
    """

    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksWindowApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksWindowImproved(self)

    app = ChatbooksWindowApp()
    async with app.run_test() as pilot:
        window = app.query_one(ChatbooksWindowImproved)
        victim = window.query_one("#chatbook-search", Input)

        # Schedule the recompose first (the widget's own next-callback),
        # then queue a capture-inducing message on the VICTIM's own pump --
        # modelling a MouseDown forwarded to the Input but not yet
        # dispatched when the teardown starts.
        window.refresh(recompose=True)
        victim.call_later(lambda: pilot.app.capture_mouse(victim))

        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        captured = pilot.app.mouse_captured
        assert captured is None, (
            f"stale capture survived the teardown drain: {captured!r} "
            f"(attached={getattr(captured, 'is_attached', None)}) -- clicks "
            "anywhere in the app are silently swallowed again (task-637)"
        )
