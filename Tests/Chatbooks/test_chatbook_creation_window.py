# TASK-32829: ``ChatbookCreationWindow.__init__`` used to assign
# ``self.app = app_instance`` on a Textual ``Widget``/``ModalScreen``, where
# ``app`` is a read-only property -- every construction raised
# ``AttributeError`` and the window could never be opened (documented in
# .superpowers/wave2-report.md). These tests pin the construct-and-mount
# contract the only production call site (UI/Tools_Settings_Window.py)
# depends on.

from textual.app import App
from textual.widgets import Input

import tldw_chatbook.UI.ChatbookCreationWindow as creation_module
from tldw_chatbook.UI.ChatbookCreationWindow import ChatbookCreationWindow


class _ScreenHost(App):
    def __init__(self, screen):
        super().__init__()
        self.screen_under_test = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)


def _stub_constructor_dependencies(monkeypatch, tmp_path):
    # Both constructor dependencies load the CLI config; pointing them at a
    # fresh temp dir keeps this test on the fresh-install empty-database
    # branch (same stubs as the wide-tier spot builder in
    # Tests/UI/test_modal_wide_tier.py).
    monkeypatch.setattr(
        creation_module,
        "get_chatbook_database_paths",
        lambda: {
            "ChaChaNotes": str(tmp_path / "chachanotes.db"),
            "Prompts": str(tmp_path / "prompts.db"),
            "Media": str(tmp_path / "media.db"),
        },
    )
    monkeypatch.setattr(creation_module, "ChatbookCreator", lambda *_a, **_k: None)


async def test_chatbook_creation_window_constructs_without_app_assignment(
    tmp_path, monkeypatch
):
    _stub_constructor_dependencies(monkeypatch, tmp_path)

    # Construction alone used to raise AttributeError (read-only Widget.app).
    window = ChatbookCreationWindow()

    assert window.selected_content is not None
    assert window.db_paths["chachanotes"] == tmp_path / "chachanotes.db"


async def test_chatbook_creation_window_mounts(tmp_path, monkeypatch):
    _stub_constructor_dependencies(monkeypatch, tmp_path)

    window = ChatbookCreationWindow()
    app = _ScreenHost(window)
    # ChatbookCreationWindow.on_mount reads ``self.app.config_data`` -- the
    # real TldwCli attribute -- so the stand-in host app carries it too.
    app.config_data = {}

    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.screen is window
        assert window.query_one("#chatbook-name", Input) is not None
        assert window.is_attached
