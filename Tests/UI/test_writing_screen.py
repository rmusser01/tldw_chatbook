from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import ListView, Static, TextArea, Tree

from tldw_chatbook.Writing_Interop.server_writing_service import (
    REASON_DIRECT_MANUSCRIPT_SCENE,
    REASON_SCENE_REPARENT,
    REASON_TRASH_RESTORE,
    REASON_VERSION_HISTORY,
)
from tldw_chatbook.UI.Writing_Modules.writing_controller import WritingController
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
        self.entities = {
            "project": SimpleNamespace(
                id="local-project",
                title="Local Draft",
                subtitle="A draft",
                author="Ada",
                genre="Speculative",
                status="draft",
                synopsis="Project synopsis",
                target_word_count=90000,
                word_count=1200,
                version=1,
                metadata={"setting": "value"},
            ),
            "manuscript": SimpleNamespace(
                id="manuscript-1",
                project_id="local-project",
                title="Book One",
                status="draft",
                synopsis="Manuscript synopsis",
                word_count=1000,
                sort_order=1,
                version=1,
                metadata={},
            ),
            "chapter": SimpleNamespace(
                id="chapter-1",
                project_id="local-project",
                manuscript_id="manuscript-1",
                title="Chapter One",
                status="draft",
                synopsis="Chapter synopsis",
                word_count=500,
                sort_order=1,
                version=1,
                metadata={},
            ),
            "scene": SimpleNamespace(
                id="scene-1",
                project_id="local-project",
                chapter_id="chapter-1",
                manuscript_id=None,
                title="Opening Scene",
                status="draft",
                synopsis="Scene synopsis",
                body_markdown="Draft A",
                word_count=25,
                sort_order=1,
                version=1,
                metadata={},
            ),
        }
        self.projects = {
            "local": [self.entities["project"]],
            "server": [SimpleNamespace(id="server-project", title="Server Draft", version=2)],
        }
        self.structure = {
            "project": self.entities["project"],
            "manuscripts": [
                {
                    "manuscript": self.entities["manuscript"],
                    "chapters": [
                        {
                            "chapter": self.entities["chapter"],
                            "scenes": [
                                self.entities["scene"],
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
        self.versions = [
            SimpleNamespace(
                source="local",
                id="version-1",
                entity_kind="scene",
                entity_id="scene-1",
                project_id="local-project",
                version_number=1,
                metadata={"title": "Opening Scene"},
                body_markdown="Draft A",
            )
        ]
        self.trash_entries = [
            SimpleNamespace(source="local", entity_kind="project", entity_id="local-project", project_id="local-project", title="Local Draft"),
            SimpleNamespace(source="local", entity_kind="manuscript", entity_id="manuscript-1", project_id="local-project", title="Book One"),
            SimpleNamespace(source="local", entity_kind="chapter", entity_id="chapter-1", project_id="local-project", title="Chapter One"),
            SimpleNamespace(source="local", entity_kind="scene", entity_id="scene-1", project_id="local-project", title="Opening Scene"),
        ]

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

    async def create_project(self, *, mode, **payload):
        self.calls.append(("create_project", mode, dict(payload)))
        return SimpleNamespace(id="server-project-new", title=payload["title"], version=1)

    async def create_manuscript(self, project_id, *, mode, **payload):
        self.calls.append(("create_manuscript", mode, project_id, dict(payload)))
        return SimpleNamespace(id="part-new", project_id=project_id, title=payload["title"], version=1)

    async def create_chapter(self, project_id, *, mode, manuscript_id=None, **payload):
        self.calls.append(("create_chapter", mode, project_id, manuscript_id, dict(payload)))
        return SimpleNamespace(
            id="chapter-new",
            project_id=project_id,
            manuscript_id=manuscript_id,
            title=payload["title"],
            version=1,
        )

    async def create_scene(self, project_id, *, mode, chapter_id=None, manuscript_id=None, **payload):
        self.calls.append(("create_scene", mode, project_id, chapter_id, manuscript_id, dict(payload)))
        return SimpleNamespace(
            id="scene-new",
            project_id=project_id,
            chapter_id=chapter_id,
            manuscript_id=manuscript_id,
            title=payload["title"],
            body_markdown=payload.get("body_markdown", ""),
            version=1,
        )

    async def get_project(self, entity_id, *, mode, include_deleted=False):
        self.calls.append(("get_project", mode, entity_id, include_deleted))
        return self.entities["project"]

    async def get_manuscript(self, entity_id, *, mode, include_deleted=False):
        self.calls.append(("get_manuscript", mode, entity_id, include_deleted))
        return self.entities["manuscript"]

    async def get_chapter(self, entity_id, *, mode, include_deleted=False):
        self.calls.append(("get_chapter", mode, entity_id, include_deleted))
        return self.entities["chapter"]

    async def get_scene(self, entity_id, *, mode, include_deleted=False):
        self.calls.append(("get_scene", mode, entity_id, include_deleted))
        return self.entities["scene"]

    async def update_project(self, entity_id, payload, expected_version, *, mode):
        self.calls.append(("update_project", mode, entity_id, dict(payload), expected_version))
        return self.entities["project"]

    async def update_manuscript(self, entity_id, payload, expected_version, *, mode):
        self.calls.append(("update_manuscript", mode, entity_id, dict(payload), expected_version))
        return self.entities["manuscript"]

    async def update_chapter(self, entity_id, payload, expected_version, *, mode):
        self.calls.append(("update_chapter", mode, entity_id, dict(payload), expected_version))
        return self.entities["chapter"]

    async def update_scene(self, entity_id, payload, expected_version, *, mode):
        self.calls.append(("update_scene", mode, entity_id, dict(payload), expected_version))
        return self.entities["scene"]

    async def delete_project(self, entity_id, *, mode, expected_version=None):
        self.calls.append(("delete_project", mode, entity_id, expected_version))
        return SimpleNamespace(id=entity_id, deleted=True)

    async def delete_manuscript(self, entity_id, *, mode, expected_version=None):
        self.calls.append(("delete_manuscript", mode, entity_id, expected_version))
        return SimpleNamespace(id=entity_id, deleted=True)

    async def delete_chapter(self, entity_id, *, mode, expected_version=None):
        self.calls.append(("delete_chapter", mode, entity_id, expected_version))
        return SimpleNamespace(id=entity_id, deleted=True)

    async def delete_scene(self, entity_id, *, mode, expected_version=None):
        self.calls.append(("delete_scene", mode, entity_id, expected_version))
        return SimpleNamespace(id=entity_id, deleted=True)

    async def assign_chapter(self, entity_id, manuscript_id, *, mode, expected_version=None, sort_order=None):
        self.calls.append(("assign_chapter", mode, entity_id, manuscript_id, expected_version, sort_order))
        return self.entities["chapter"]

    async def autosave_scene(self, entity_id, *, mode, body_markdown, expected_version=None):
        self.calls.append(("autosave_scene", mode, entity_id, body_markdown, expected_version))
        self.entities["scene"].body_markdown = body_markdown
        return self.entities["scene"]

    async def create_version(self, entity_kind, entity_id, *, mode, snapshot=None, body_markdown=None, label=None):
        self.calls.append(("create_version", mode, entity_kind, entity_id, snapshot, body_markdown, label))
        return self.versions[0]

    async def list_versions(self, entity_kind, entity_id, *, mode, include_deleted=False, limit=100, offset=0):
        self.calls.append(("list_versions", mode, entity_kind, entity_id, include_deleted, limit, offset))
        return list(self.versions)

    async def restore_version_to_working_state(self, version_id, *, mode, entity_kind="scene", expected_version=None):
        self.calls.append(("restore_version_to_working_state", mode, version_id, entity_kind, expected_version))
        self.entities[entity_kind].body_markdown = "Draft A"
        return self.entities[entity_kind]

    async def list_trash(self, project_id=None, *, mode, limit=100, offset=0):
        self.calls.append(("list_trash", mode, project_id, limit, offset))
        return list(self.trash_entries)

    async def restore_project(self, entity_id, *, mode, expected_version=None):
        self.calls.append(("restore_project", mode, entity_id, expected_version))
        self.trash_entries = [entry for entry in self.trash_entries if entry.entity_id != entity_id]
        return self.entities["project"]

    async def restore_manuscript(self, entity_id, *, mode, expected_version=None):
        self.calls.append(("restore_manuscript", mode, entity_id, expected_version))
        self.trash_entries = [entry for entry in self.trash_entries if entry.entity_id != entity_id]
        return self.entities["manuscript"]

    async def restore_chapter(self, entity_id, *, mode, expected_version=None):
        self.calls.append(("restore_chapter", mode, entity_id, expected_version))
        self.trash_entries = [entry for entry in self.trash_entries if entry.entity_id != entity_id]
        return self.entities["chapter"]

    async def restore_scene(self, entity_id, *, mode, expected_version=None):
        self.calls.append(("restore_scene", mode, entity_id, expected_version))
        self.trash_entries = [entry for entry in self.trash_entries if entry.entity_id != entity_id]
        return self.entities["scene"]

    def get_capability(self, **kwargs):
        mode = kwargs.get("mode")
        action = kwargs.get("action")
        entity_kind = kwargs.get("entity_kind")
        parent_kind = kwargs.get("parent_kind")
        if mode == "server" and action == "create_version":
            return SimpleNamespace(supported=False, reason=REASON_VERSION_HISTORY, metadata=kwargs)
        if mode == "server" and action == "restore_deleted":
            return SimpleNamespace(supported=False, reason=REASON_TRASH_RESTORE, metadata=kwargs)
        if mode == "server" and action == "reparent" and entity_kind == "scene":
            return SimpleNamespace(supported=False, reason=REASON_SCENE_REPARENT, metadata=kwargs)
        if mode == "server" and action in {"create", "move"} and entity_kind == "scene" and parent_kind == "manuscript":
            return SimpleNamespace(supported=False, reason=REASON_DIRECT_MANUSCRIPT_SCENE, metadata=kwargs)
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
async def test_mounted_server_unavailable_state_updates_visible_status():
    scope = FakeWritingScopeService(server_available=False)
    window = _writing_window(scope)
    app = WritingWindowHarness(window)

    async with app.run_test():
        await window.load_projects("local")
        await window.switch_source("server")

        source_status = app.query_one("#writing-source-status", Static)
        window_status = app.query_one("#writing-status", Static)
        project_list = app.query_one("#writing-project-list", ListView)

        assert "Server writing backend is unavailable" in str(source_status.render())
        assert "Server writing backend is unavailable" in str(window_status.render())
        assert len(project_list.children) == 0


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


@pytest.mark.asyncio
async def test_controller_autosaves_scene_body_through_scene_autosave():
    scope = FakeWritingScopeService()
    controller = WritingController(scope)

    await controller.autosave_current(
        "local",
        "scene",
        "scene-1",
        {"body_markdown": "Draft B"},
        expected_version=1,
    )

    assert ("autosave_scene", "local", "scene-1", "Draft B", 1) in scope.calls


@pytest.mark.asyncio
async def test_controller_autosaves_container_metadata_without_body_fields():
    scope = FakeWritingScopeService()
    controller = WritingController(scope)

    await controller.autosave_current(
        "local",
        "project",
        "local-project",
        {"title": "Project Revised", "body_markdown": "not allowed"},
        expected_version=1,
    )
    await controller.autosave_current(
        "local",
        "manuscript",
        "manuscript-1",
        {"title": "Book Revised", "body_markdown": "not allowed"},
        expected_version=1,
    )
    await controller.autosave_current(
        "local",
        "chapter",
        "chapter-1",
        {"title": "Chapter Revised", "body_markdown": "not allowed"},
        expected_version=1,
    )

    assert ("update_project", "local", "local-project", {"title": "Project Revised"}, 1) in scope.calls
    assert ("update_manuscript", "local", "manuscript-1", {"title": "Book Revised"}, 1) in scope.calls
    assert ("update_chapter", "local", "chapter-1", {"title": "Chapter Revised"}, 1) in scope.calls


def test_detail_panel_enables_body_editor_only_for_scenes():
    scope = FakeWritingScopeService()
    panel = _writing_window(scope).detail_panel

    panel.load_entity(
        {"source": "local", "kind": "scene", "id": "scene-1", "project_id": "local-project"},
        scope.entities["scene"],
    )
    assert panel.body_editor_enabled is True
    assert panel.detail_text == "Draft A"

    panel.load_entity(
        {"source": "local", "kind": "chapter", "id": "chapter-1", "project_id": "local-project"},
        scope.entities["chapter"],
    )
    assert panel.body_editor_enabled is False
    assert "body_markdown" not in panel.detail_text
    assert "Chapter synopsis" in panel.detail_text

    panel.load_entity(
        {"source": "local", "kind": "manuscript", "id": "manuscript-1", "project_id": "local-project"},
        scope.entities["manuscript"],
    )
    assert panel.body_editor_enabled is False
    assert "body_markdown" not in panel.detail_text
    assert "Manuscript synopsis" in panel.detail_text


def test_detail_panel_local_versions_are_available_only_for_non_project_entities():
    scope = FakeWritingScopeService()
    panel = _writing_window(scope).detail_panel

    for kind in ("manuscript", "chapter", "scene"):
        panel.load_entity(
            {"source": "local", "kind": kind, "id": scope.entities[kind].id, "project_id": "local-project"},
            scope.entities[kind],
        )
        assert panel.create_version_enabled is True

    panel.load_entity(
        {"source": "local", "kind": "project", "id": "local-project", "project_id": "local-project"},
        scope.entities["project"],
    )
    assert panel.create_version_enabled is False


def test_detail_panel_version_list_is_read_only():
    scope = FakeWritingScopeService()
    panel = _writing_window(scope).detail_panel

    panel.set_versions(scope.versions)

    assert panel.version_list_read_only is True
    assert panel.version_labels == ["v1"]
    assert "Draft A" in panel.version_preview_text


def test_detail_panel_non_entity_selection_clears_editable_entity_state():
    scope = FakeWritingScopeService()
    panel = _writing_window(scope).detail_panel

    panel.load_entity(
        {"source": "local", "kind": "scene", "id": "scene-1", "project_id": "local-project"},
        scope.entities["scene"],
    )
    panel.set_versions(scope.versions)
    panel.load_node(
        {
            "source": "local",
            "kind": "unassigned_chapters",
            "id": None,
            "project_id": "local-project",
            "title": "Unassigned Chapters",
        }
    )

    assert panel.entity is None
    assert panel.selected_version_id is None
    assert panel.create_version_enabled is False
    assert panel.current_payload() == {}


@pytest.mark.asyncio
async def test_window_restore_local_version_updates_working_state_and_detail():
    scope = FakeWritingScopeService()
    window = _writing_window(scope)

    await window.load_entity_detail(
        {"source": "local", "kind": "scene", "id": "scene-1", "project_id": "local-project", "version": 2}
    )
    restored = await window.restore_selected_version("version-1")

    assert restored is scope.entities["scene"]
    assert (
        "restore_version_to_working_state",
        "local",
        "version-1",
        "scene",
        1,
    ) in scope.calls
    assert window.detail_panel.detail_text == "Draft A"


@pytest.mark.asyncio
async def test_window_local_trash_can_restore_deleted_entities():
    scope = FakeWritingScopeService()
    window = _writing_window(scope)

    entries = await window.load_trash("local-project")
    for entry in list(entries):
        await window.restore_trash_entry(entry)

    assert ("restore_project", "local", "local-project", None) in scope.calls
    assert ("restore_manuscript", "local", "manuscript-1", None) in scope.calls
    assert ("restore_chapter", "local", "chapter-1", None) in scope.calls
    assert ("restore_scene", "local", "scene-1", None) in scope.calls
    assert window.detail_panel.trash_entries == []


@pytest.mark.asyncio
async def test_server_project_create_update_delete_routes_scope_methods():
    scope = FakeWritingScopeService()
    window = _writing_window(scope)
    window.current_source = "server"

    await window.create_project({"title": "Server Project"})
    await window.load_entity_detail(
        {"source": "server", "kind": "project", "id": "local-project", "project_id": "local-project"}
    )
    update_payload = window.detail_panel.current_payload()
    await window.autosave_selected_entity()
    await window.delete_selected_entity()

    assert ("create_project", "server", {"title": "Server Project"}) in scope.calls
    assert ("update_project", "server", "local-project", update_payload, 1) in scope.calls
    assert ("delete_project", "server", "local-project", 1) in scope.calls


@pytest.mark.asyncio
async def test_server_child_create_and_chapter_assignment_route_through_scope_methods():
    scope = FakeWritingScopeService()
    controller = WritingController(scope)

    await controller.create_child(
        "server",
        {"kind": "project", "id": "local-project"},
        {"title": "Part Two"},
    )
    await controller.create_child(
        "server",
        {"kind": "manuscript", "id": "part-1", "project_id": "local-project"},
        {"title": "Server Chapter"},
    )
    await controller.create_child(
        "server",
        {"kind": "unassigned_chapters", "project_id": "local-project"},
        {"title": "Loose Server Chapter"},
    )
    await controller.assign_chapter(
        "server",
        "chapter-1",
        manuscript_id=None,
        expected_version=5,
        sort_order=10,
    )

    assert ("create_manuscript", "server", "local-project", {"title": "Part Two"}) in scope.calls
    assert (
        "create_chapter",
        "server",
        "local-project",
        "part-1",
        {"title": "Server Chapter"},
    ) in scope.calls
    assert (
        "create_chapter",
        "server",
        "local-project",
        None,
        {"title": "Loose Server Chapter"},
    ) in scope.calls
    assert ("assign_chapter", "server", "chapter-1", None, 5, 10) in scope.calls


def test_server_action_states_expose_unsupported_reason_codes():
    scope = FakeWritingScopeService()
    window = _writing_window(scope)
    window.current_source = "server"

    direct_scene = window.get_action_state("create", "scene", parent_kind="manuscript")
    version = window.get_action_state("create_version", "scene")
    trash = window.get_action_state("restore_deleted", "scene")
    reparent = window.get_action_state("reparent", "scene")

    assert direct_scene.supported is False
    assert direct_scene.reason == REASON_DIRECT_MANUSCRIPT_SCENE
    assert version.supported is False
    assert version.reason == REASON_VERSION_HISTORY
    assert trash.supported is False
    assert trash.reason == REASON_TRASH_RESTORE
    assert reparent.supported is False
    assert reparent.reason == REASON_SCENE_REPARENT


@pytest.mark.asyncio
async def test_server_scene_create_is_enabled_only_under_chapter():
    scope = FakeWritingScopeService()
    controller = WritingController(scope)

    await controller.create_child(
        "server",
        {"kind": "chapter", "id": "chapter-1", "project_id": "local-project"},
        {"title": "Server Scene", "body_markdown": "Draft"},
    )

    assert (
        "create_scene",
        "server",
        "local-project",
        "chapter-1",
        None,
        {"title": "Server Scene", "body_markdown": "Draft"},
    ) in scope.calls


@pytest.mark.asyncio
async def test_server_versions_and_trash_are_disabled_with_visible_reasons():
    scope = FakeWritingScopeService()
    window = _writing_window(scope)
    window.current_source = "server"

    await window.load_entity_detail(
        {"source": "server", "kind": "scene", "id": "scene-1", "project_id": "local-project"}
    )
    trash_entries = await window.load_trash("local-project")

    assert window.detail_panel.create_version_enabled is False
    assert window.detail_panel.unsupported_reasons["create_version"] == REASON_VERSION_HISTORY
    assert trash_entries == []
    assert window.detail_panel.unsupported_reasons["restore_deleted"] == REASON_TRASH_RESTORE
    assert REASON_TRASH_RESTORE in window.status_message


@pytest.mark.asyncio
async def test_server_version_reason_survives_autosave_refresh():
    scope = FakeWritingScopeService()
    window = _writing_window(scope)
    window.current_source = "server"

    await window.load_entity_detail(
        {"source": "server", "kind": "scene", "id": "scene-1", "project_id": "local-project"}
    )
    await window.autosave_selected_entity()

    assert window.detail_panel.unsupported_reasons["create_version"] == REASON_VERSION_HISTORY


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

        await window._handle_outline_node_selected(SimpleNamespace(node=scene_node))
        detail_title = app.query_one("#writing-detail-title", Static)
        detail_editor = app.query_one("#writing-detail-editor", TextArea)

    assert window.detail_panel.title == "Opening Scene"
    assert "Opening Scene" in str(detail_title.render())
    assert detail_editor.text == "Draft A"
    assert detail_editor.read_only is False
