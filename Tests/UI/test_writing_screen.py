from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import ListView, Tree

from tldw_chatbook.UI.Screens.writing_screen import WritingScreen
from tldw_chatbook.UI.Writing_Window import WritingWindow


def test_writing_screen_composes_writing_window():
    app = SimpleNamespace(writing_scope_service=object())
    screen = WritingScreen(app)

    widgets = list(screen.compose_content())

    assert len(widgets) == 1
    assert isinstance(widgets[0], WritingWindow)


def test_writing_screen_round_trips_window_state():
    app = SimpleNamespace(writing_scope_service=object())
    screen = WritingScreen(app)
    window = WritingWindow(app)
    window.restore_state({"source": "server"})
    screen.query_one = lambda *_args, **_kwargs: window

    state = screen.save_state()
    screen.restore_state({"source": "local"})

    assert state == {"source": "server"}
    assert window.save_state() == {"source": "local"}


def test_writing_screen_applies_state_restored_before_compose():
    app = SimpleNamespace(writing_scope_service=object())
    screen = WritingScreen(app)

    screen.restore_state({"source": "server"})
    widgets = list(screen.compose_content())

    assert widgets[0].save_state() == {"source": "server"}


class FakeWritingScopeService:
    def __init__(self, *, server_available=True):
        self.server_available = server_available
        self.calls = []
        self.projects = {
            "local": [SimpleNamespace(id="local-project", title="Local Draft", version=1)],
            "server": [SimpleNamespace(id="server-project", title="Server Draft", version=2)],
        }
        self.structure = {
            "project": SimpleNamespace(id="local-project", title="Local Draft", version=1),
            "manuscripts": [
                {
                    "manuscript": SimpleNamespace(id="manuscript-1", title="Book One", version=1),
                    "chapters": [
                        {
                            "chapter": SimpleNamespace(id="chapter-1", title="Chapter One", version=1),
                            "scenes": [
                                SimpleNamespace(id="scene-1", title="Opening Scene", version=1),
                            ],
                        }
                    ],
                    "direct_scenes": [],
                }
            ],
            "unassigned_chapters": [
                {
                    "chapter": SimpleNamespace(id="chapter-loose", title="Loose Chapter", version=1),
                    "scenes": [
                        SimpleNamespace(id="scene-loose", title="Loose Scene", version=1),
                    ],
                }
            ],
        }

    async def list_projects(self, *, mode, **_kwargs):
        self.calls.append(("list_projects", mode))
        if mode == "server" and not self.server_available:
            raise ValueError("Server writing backend is unavailable")
        return list(self.projects[mode])

    async def get_project_structure(self, project_id, *, mode):
        self.calls.append(("get_project_structure", mode, project_id))
        if mode == "server" and not self.server_available:
            raise ValueError("Server writing backend is unavailable")
        return self.structure

    def get_capability(self, **kwargs):
        return SimpleNamespace(supported=True, reason=None, metadata=kwargs)


def _writing_window(scope=None):
    app = SimpleNamespace(writing_scope_service=scope or FakeWritingScopeService())
    return WritingWindow(app)


def test_writing_window_defaults_to_local_source():
    window = _writing_window()

    assert window.save_state() == {"source": "local"}
    assert window.current_source == "local"


@pytest.mark.asyncio
async def test_switching_source_reloads_projects_from_selected_source_only():
    scope = FakeWritingScopeService()
    window = _writing_window(scope)

    await window.load_projects("local")
    await window.switch_source("server")

    assert scope.calls == [("list_projects", "local"), ("list_projects", "server")]
    assert window.source_panel.project_titles == ["Server Draft"]
    assert window.source_panel.project_ids == ["server-project"]
    assert window.current_source == "server"


@pytest.mark.asyncio
async def test_missing_server_configuration_shows_unavailable_state_without_local_fallback():
    scope = FakeWritingScopeService(server_available=False)
    window = _writing_window(scope)

    await window.load_projects("local")
    await window.switch_source("server")

    assert scope.calls == [("list_projects", "local"), ("list_projects", "server")]
    assert window.source_panel.project_titles == []
    assert "Server writing backend is unavailable" in window.status_message


@pytest.mark.asyncio
async def test_outline_renders_project_hierarchy_and_unassigned_chapters():
    scope = FakeWritingScopeService()
    window = _writing_window(scope)

    await window.load_project_structure("local-project")

    assert window.outline_tree.labels == [
        "Local Draft",
        "Book One",
        "Chapter One",
        "Opening Scene",
        "Unassigned Chapters",
        "Loose Chapter",
        "Loose Scene",
    ]


def test_selecting_outline_node_loads_detail_panel():
    window = _writing_window()
    node_data = {
        "source": "local",
        "kind": "scene",
        "id": "scene-1",
        "project_id": "project-1",
        "title": "Opening Scene",
        "version": 3,
    }

    window.select_outline_node(node_data)

    assert window.detail_panel.selected_node == node_data
    assert window.detail_panel.title == "Opening Scene"
    assert "scene" in window.detail_panel.detail_text


class WritingWindowHarness(App):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def compose(self) -> ComposeResult:
        yield self.window


@pytest.mark.asyncio
async def test_mounted_project_list_renders_and_selects_project():
    scope = FakeWritingScopeService()
    window = _writing_window(scope)
    app = WritingWindowHarness(window)

    async with app.run_test():
        await window.load_projects("local")
        project_list = app.query_one("#writing-project-list", ListView)

        assert len(project_list.children) == 1
        assert getattr(project_list.children[0], "project_id") == "local-project"

        await window._handle_project_selected(SimpleNamespace(item=project_list.children[0]))

    assert ("get_project_structure", "local", "local-project") in scope.calls
    assert window.outline_tree.labels[0] == "Local Draft"


@pytest.mark.asyncio
async def test_mounted_outline_tree_renders_and_selects_nodes():
    window = _writing_window()
    app = WritingWindowHarness(window)

    async with app.run_test():
        await window.load_project_structure("local-project")
        tree = app.query_one("#writing-outline-tree", Tree)

        assert len(tree.root.children) == 2
        scene_node = tree.root.children[0].children[0].children[0]
        assert scene_node.data["kind"] == "scene"

        window._handle_outline_node_selected(SimpleNamespace(node=scene_node))

    assert window.detail_panel.title == "Opening Scene"
