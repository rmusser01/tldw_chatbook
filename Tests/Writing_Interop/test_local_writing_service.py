import sqlite3
import threading
import time
from contextlib import contextmanager

import pytest

from tldw_chatbook.Writing_Interop import local_writing_service
from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService


def test_local_writing_service_persists_project_hierarchy_and_unassigned_chapters(
    tmp_path,
):
    service = LocalWritingService(tmp_path / "writing.db")

    project = service.create_project(title="Novel", author="Ada", genre="sci-fi")
    manuscript = service.create_manuscript(
        project["id"], title="Book One", synopsis="Opening arc"
    )
    assigned_chapter = service.create_chapter(
        project["id"],
        title="Chapter 1",
        manuscript_id=manuscript["id"],
        synopsis="Arrival",
    )
    unassigned_chapter = service.create_chapter(project["id"], title="Loose Chapter")
    scene = service.create_scene(
        assigned_chapter["id"],
        title="Scene 1",
        content_markdown="Opening line.",
        synopsis="Meet Ada",
    )

    structure = service.get_structure(project["id"])

    assert project["source"] == "local"
    assert manuscript["project_id"] == project["id"]
    assert assigned_chapter["manuscript_id"] == manuscript["id"]
    assert unassigned_chapter["manuscript_id"] is None
    assert scene["content_markdown"] == "Opening line."
    assert structure["manuscripts"][0]["id"] == manuscript["id"]
    assert structure["manuscripts"][0]["chapters"][0]["id"] == assigned_chapter["id"]
    assert structure["manuscripts"][0]["chapters"][0]["scenes"][0]["id"] == scene["id"]
    assert structure["unassigned_chapters"][0]["id"] == unassigned_chapter["id"]


def test_local_writing_service_updates_versions_and_soft_deletes(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(
        project["id"], title="Chapter 1", manuscript_id=manuscript["id"]
    )
    scene = service.create_scene(
        chapter["id"], title="Scene 1", content_markdown="Draft"
    )

    updated_scene = service.update_scene(
        scene["id"],
        expected_version=1,
        title="Scene 1 revised",
        content_markdown="Revised draft",
    )
    deleted = service.delete_scene(scene["id"], expected_version=2)

    assert updated_scene["version"] == 2
    assert updated_scene["content_markdown"] == "Revised draft"
    assert deleted is True
    assert service.get_scene(scene["id"]) is None
    assert service.list_scenes(chapter["id"]) == []


def test_local_writing_service_updates_and_deletes_manuscripts_and_chapters(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(
        project["id"], title="Chapter 1", manuscript_id=manuscript["id"]
    )

    updated_manuscript = service.update_manuscript(
        manuscript["id"],
        expected_version=1,
        title="Book One Revised",
    )
    updated_chapter = service.update_chapter(
        chapter["id"],
        expected_version=1,
        title="Chapter 1 Revised",
        manuscript_id=None,
    )
    deleted_chapter = service.delete_chapter(chapter["id"], expected_version=2)
    deleted_manuscript = service.delete_manuscript(manuscript["id"], expected_version=2)

    assert updated_manuscript["version"] == 2
    assert updated_manuscript["title"] == "Book One Revised"
    assert updated_chapter["version"] == 2
    assert updated_chapter["manuscript_id"] is None
    assert deleted_chapter is True
    assert deleted_manuscript is True
    assert service.get_chapter(chapter["id"]) is None
    assert service.get_manuscript(manuscript["id"]) is None


def test_local_writing_service_preserves_chapter_manuscript_when_not_explicitly_changed(
    tmp_path,
):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(
        project["id"], title="Chapter 1", manuscript_id=manuscript["id"]
    )

    updated_chapter = service.update_chapter(
        chapter["id"],
        expected_version=1,
        title="Chapter 1 Revised",
    )

    assert updated_chapter["manuscript_id"] == manuscript["id"]


def test_local_writing_service_rejects_stale_expected_versions(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")

    with pytest.raises(ValueError, match="version conflict"):
        service.update_project(project["id"], expected_version=2, title="Stale")


def test_local_writing_service_creates_manual_scene_versions_and_restores_them(
    tmp_path,
):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(
        project["id"], title="Chapter 1", manuscript_id=manuscript["id"]
    )
    scene = service.create_scene(
        chapter["id"], title="Scene 1", content_markdown="# Draft\n\nOpening."
    )

    version = service.create_version("scene", scene["id"], label="First draft")
    service.update_scene(
        scene["id"],
        expected_version=1,
        title="Scene 1 revised",
        content_markdown="Changed.",
    )
    restored = service.restore_version(
        "scene", scene["id"], version["version_number"], expected_version=2
    )
    versions = service.list_versions("scene", scene["id"])

    assert version["source"] == "local"
    assert version["record_id"].startswith("local:writing_version:")
    assert version["entity_type"] == "scene"
    assert version["entity_id"] == scene["id"]
    assert version["version_number"] == 1
    assert version["label"] == "First draft"
    assert version["payload"]["content_markdown"] == "# Draft\n\nOpening."
    assert restored["title"] == "Scene 1"
    assert restored["content_markdown"] == "# Draft\n\nOpening."
    assert restored["version"] == 3
    assert versions[0]["version_number"] == 1
    assert versions[0]["payload"]["content_markdown"] == "# Draft\n\nOpening."


def test_local_writing_service_container_versions_snapshot_structure_not_body_drafts(
    tmp_path,
):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(
        project["id"], title="Chapter 1", manuscript_id=manuscript["id"]
    )
    scene = service.create_scene(
        chapter["id"], title="Scene 1", content_markdown="Opening line."
    )

    manuscript_version = service.create_version("manuscript", manuscript["id"])
    chapter_version = service.create_version("chapter", chapter["id"])

    assert manuscript_version["payload"]["chapter_ids"] == [chapter["id"]]
    assert manuscript_version["payload"]["rendered_markdown"] == "Opening line."
    assert "content_markdown" not in manuscript_version["payload"]
    assert chapter_version["payload"]["scene_ids"] == [scene["id"]]
    assert chapter_version["payload"]["rendered_markdown"] == "Opening line."
    assert "content_markdown" not in chapter_version["payload"]


def test_local_writing_service_lists_and_restores_soft_deleted_records(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(
        project["id"], title="Chapter 1", manuscript_id=manuscript["id"]
    )
    scene = service.create_scene(
        chapter["id"], title="Scene 1", content_markdown="Draft"
    )

    assert service.delete_scene(scene["id"], expected_version=1) is True
    trash = service.list_trash(entity_type="scene")
    restored = service.restore_trash("scene", scene["id"], expected_version=2)

    assert trash[0]["record_id"] == f"local:writing_scene:{scene['id']}"
    assert trash[0]["deleted"] == 1
    assert restored["id"] == scene["id"]
    assert restored["deleted"] == 0
    assert restored["version"] == 3
    assert service.list_trash(entity_type="scene") == []
    assert service.get_scene(scene["id"])["content_markdown"] == "Draft"


def test_local_writing_service_reorders_and_moves_chapters_and_scenes(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript_a = service.create_manuscript(project["id"], title="Book One")
    manuscript_b = service.create_manuscript(project["id"], title="Book Two")
    chapter_a = service.create_chapter(
        project["id"], title="Chapter 1", manuscript_id=manuscript_a["id"]
    )
    chapter_b = service.create_chapter(
        project["id"], title="Chapter 2", manuscript_id=manuscript_b["id"]
    )
    scene = service.create_scene(
        chapter_a["id"], title="Scene 1", content_markdown="Draft"
    )

    assert (
        service.reorder_entities(
            project["id"],
            "chapters",
            [
                {
                    "id": chapter_a["id"],
                    "sort_order": 10.0,
                    "version": 1,
                    "new_parent_id": manuscript_b["id"],
                }
            ],
        )
        is True
    )
    assert (
        service.reorder_entities(
            project["id"],
            "scenes",
            [
                {
                    "id": scene["id"],
                    "sort_order": 5.0,
                    "version": 1,
                    "new_parent_id": chapter_b["id"],
                }
            ],
        )
        is True
    )

    moved_chapter = service.get_chapter(chapter_a["id"])
    moved_scene = service.get_scene(scene["id"])

    assert moved_chapter["manuscript_id"] == manuscript_b["id"]
    assert moved_chapter["sort_order"] == 10.0
    assert moved_chapter["version"] == 2
    assert moved_scene["chapter_id"] == chapter_b["id"]
    assert moved_scene["sort_order"] == 5.0
    assert moved_scene["version"] == 2


def test_local_writing_service_supports_direct_manuscript_level_scenes(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")

    scene = service.create_scene(
        None,
        manuscript_id=manuscript["id"],
        title="Prologue",
        content_markdown="Direct scene.",
    )
    manuscript_scenes = service.list_scenes(manuscript_id=manuscript["id"])
    structure = service.get_structure(project["id"])

    assert scene["chapter_id"] is None
    assert scene["manuscript_id"] == manuscript["id"]
    assert manuscript_scenes[0]["id"] == scene["id"]
    assert structure["manuscripts"][0]["scenes"][0]["id"] == scene["id"]


def test_local_writing_service_persists_authoring_auxiliary_resources(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(
        project["id"], title="Chapter 1", manuscript_id=manuscript["id"]
    )
    scene = service.create_scene(
        chapter["id"], title="Scene 1", content_markdown="Draft"
    )

    character = service.create_character(
        project["id"],
        name="Ada",
        role="protagonist",
        cast_group="main",
        custom_fields={"voice": "dry"},
    )
    relationship = service.create_relationship(
        project["id"],
        from_character_id=character["id"],
        to_character_id=character["id"],
        relationship_type="self",
        bidirectional=False,
    )
    world = service.create_world_info(
        project["id"],
        kind="location",
        name="Capital",
        properties={"climate": "rain"},
        tags=["city"],
    )
    plot_line = service.create_plot_line(
        project["id"], title="Main Plot", color="#336699"
    )
    plot_event = service.create_plot_event(
        plot_line["id"],
        title="Inciting Incident",
        scene_id=scene["id"],
        event_type="plot",
    )
    plot_hole = service.create_plot_hole(
        project["id"],
        title="Continuity Issue",
        scene_id=scene["id"],
        plot_line_id=plot_line["id"],
        severity="high",
    )
    scene_characters = service.link_scene_character(
        scene["id"], character_id=character["id"], is_pov=True
    )
    scene_world = service.link_scene_world_info(scene["id"], world_info_id=world["id"])
    citation = service.create_citation(
        scene["id"],
        source_type="manual",
        source_title="Reference",
        excerpt="Quoted fact",
    )

    updated_character = service.update_character(
        character["id"],
        expected_version=1,
        notes="Revised notes",
    )
    updated_world = service.update_world_info(
        world["id"], expected_version=1, tags=["city", "capital"]
    )
    updated_plot_line = service.update_plot_line(
        plot_line["id"], expected_version=1, status="resolved"
    )
    updated_plot_event = service.update_plot_event(
        plot_event["id"], expected_version=1, title="New Incident"
    )
    updated_plot_hole = service.update_plot_hole(
        plot_hole["id"],
        expected_version=1,
        status="resolved",
        resolution="Fixed in scene.",
    )

    assert character["record_id"] == f"local:writing_character:{character['id']}"
    assert character["custom_fields"] == {"voice": "dry"}
    assert (
        service.list_characters(project["id"], role="protagonist")[0]["id"]
        == character["id"]
    )
    assert updated_character["version"] == 2
    assert updated_character["notes"] == "Revised notes"
    assert relationship["bidirectional"] is False
    assert (
        service.list_relationships(project["id"])[0]["record_id"]
        == f"local:writing_relationship:{relationship['id']}"
    )
    assert world["properties"] == {"climate": "rain"}
    assert updated_world["tags"] == ["city", "capital"]
    assert (
        service.list_world_info(project["id"], kind="location")[0]["id"] == world["id"]
    )
    assert updated_plot_line["status"] == "resolved"
    assert updated_plot_event["title"] == "New Incident"
    assert service.list_plot_events(plot_line["id"])[0]["id"] == plot_event["id"]
    assert updated_plot_hole["status"] == "resolved"
    assert (
        service.list_plot_holes(project["id"], status="resolved")[0]["id"]
        == plot_hole["id"]
    )
    assert (
        scene_characters[0]["record_id"]
        == f"local:writing_scene_character_link:{scene['id']}:{character['id']}"
    )
    assert scene_characters[0]["name"] == "Ada"
    assert scene_characters[0]["is_pov"] is True
    assert (
        scene_world[0]["record_id"]
        == f"local:writing_scene_world_info_link:{scene['id']}:{world['id']}"
    )
    assert scene_world[0]["name"] == "Capital"
    assert citation["record_id"] == f"local:writing_citation:{citation['id']}"
    assert service.list_citations(scene["id"])[0]["source_title"] == "Reference"

    assert service.unlink_scene_character(scene["id"], character["id"]) is True
    assert service.unlink_scene_world_info(scene["id"], world["id"]) is True
    assert service.delete_citation(citation["id"], expected_version=1) is True
    assert service.delete_plot_hole(plot_hole["id"], expected_version=2) is True
    assert service.delete_plot_event(plot_event["id"], expected_version=2) is True
    assert service.delete_plot_line(plot_line["id"], expected_version=2) is True
    assert service.delete_world_info(world["id"], expected_version=2) is True
    assert service.delete_relationship(relationship["id"], expected_version=1) is True
    assert service.delete_character(character["id"], expected_version=2) is True

    assert service.list_scene_characters(scene["id"]) == []
    assert service.list_scene_world_info(scene["id"]) == []
    assert service.list_citations(scene["id"]) == []


def test_local_writing_service_runs_local_research_and_persists_analyses(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    manuscript = service.create_manuscript(project["id"], title="Book One")
    chapter = service.create_chapter(
        project["id"], title="Chapter 1", manuscript_id=manuscript["id"]
    )
    scene = service.create_scene(
        chapter["id"],
        title="Market Chase",
        content_markdown="Ada follows the brass automaton through the market.",
    )
    service.create_character(
        project["id"], name="Ada", notes="Inventor hunting a brass automaton."
    )
    service.create_world_info(
        project["id"],
        kind="location",
        name="Clockwork Market",
        description="A busy market full of brass mechanisms.",
    )

    research = service.research_scene(scene["id"], query="brass market", top_k=3)
    scene_analyses = service.analyze_scene(scene["id"], analysis_types=["pacing"])
    chapter_analyses = service.analyze_chapter(
        chapter["id"], analysis_types=["continuity"]
    )
    plot_holes = service.analyze_project_plot_holes(project["id"])
    consistency = service.analyze_project_consistency(project["id"])
    listed = service.list_analyses(project["id"], scope_type="scene")

    assert research["source"] == "local"
    assert research["scene_id"] == scene["id"]
    assert len(research["results"]) == 3
    assert research["results"][0]["record_type"] == "writing_research_result"
    assert research["results"][0]["score"] > 0
    assert scene_analyses[0]["record_id"].startswith("local:writing_analysis:")
    assert scene_analyses[0]["scope_type"] == "scene"
    assert scene_analyses[0]["analysis_type"] == "pacing"
    assert scene_analyses[0]["findings"]
    assert chapter_analyses[0]["scope_type"] == "chapter"
    assert plot_holes[0]["analysis_type"] == "plot_holes"
    assert consistency[0]["analysis_type"] == "consistency"
    assert listed["total"] == 1
    assert listed["analyses"][0]["id"] == scene_analyses[0]["id"]


# --- TASK-21125: held thread-local connection + lifecycle ------------------


class _RollbackFailingConnection:
    """Proxy that makes ROLLBACK fail so masking can be observed."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, *args, **kwargs):
        if str(sql).strip().upper().startswith("ROLLBACK"):
            raise sqlite3.OperationalError("rollback failed")
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _RollbackFailingService(LocalWritingService):
    def _open_connection(self):
        return _RollbackFailingConnection(super()._open_connection())


class _GatedWritingService(LocalWritingService):
    """Parks the first transaction after BEGIN so close() must wait for it."""

    def __init__(self, db_path, entered, release):
        super().__init__(db_path)
        self._entered = entered
        self._release = release
        self._gate_armed = False

    def arm(self):
        self._gate_armed = True

    @contextmanager
    def _transaction(self):
        with LocalWritingService._transaction(self) as conn:
            if self._gate_armed:
                self._gate_armed = False
                self._entered.set()
                assert self._release.wait(10), "gate was never released"
            yield conn


def test_local_writing_service_reuses_one_held_connection_per_thread(
    tmp_path, monkeypatch
):
    opens = []
    real_connect = local_writing_service.connect_private_sqlite

    def _counting_connect(*args, **kwargs):
        opens.append(threading.current_thread().name)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(
        local_writing_service, "connect_private_sqlite", _counting_connect
    )
    service = LocalWritingService(tmp_path / "writing.db")
    try:
        project = service.create_project(title="Novel")
        opens_after_first_operation = len(opens)

        for _ in range(10):
            service.get_project(project["id"])
            service.list_projects()
            service.update_project(project["id"], expected_version=None, synopsis="s")

        # One schema connection plus one held connection for this thread; the
        # 30 follow-up operations open nothing at all.
        assert opens_after_first_operation <= 2
        assert len(opens) == opens_after_first_operation
    finally:
        service.close()


def test_local_writing_service_held_connection_pragmas_read_back(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    try:
        service.create_project(title="Novel")
        conn = service._connect()

        assert service._connect() is conn
        assert conn.isolation_level is None
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert int(conn.execute("PRAGMA synchronous").fetchone()[0]) == 1

        probe = sqlite3.connect(tmp_path / "writing.db")
        try:
            assert probe.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            probe.close()
    finally:
        service.close()


def test_local_writing_service_gives_each_thread_its_own_connection(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    try:
        main_conn = service._connect()
        seen = {}

        def _worker():
            seen["conn"] = service._connect()
            seen["project"] = service.create_project(title="From worker")

        thread = threading.Thread(target=_worker)
        thread.start()
        thread.join(10)

        assert seen["conn"] is not main_conn
        # Cross-thread visibility: the main thread's own connection sees it.
        assert service.get_project(seen["project"]["id"])["title"] == "From worker"
    finally:
        service.close()


def test_local_writing_service_close_releases_connections_and_rearms(tmp_path):
    service = LocalWritingService(tmp_path / "writing.db")
    project = service.create_project(title="Novel")
    conn = service._connect()

    service.close()

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
    # The store re-arms: the next operation transparently reopens.
    assert service.get_project(project["id"])["title"] == "Novel"
    assert service._connect() is not conn
    service.close()


def test_local_writing_service_close_waits_for_an_in_flight_operation(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    service = _GatedWritingService(tmp_path / "writing.db", entered, release)
    project = service.create_project(title="Novel")
    failures = []

    def _worker():
        try:
            service.update_project(
                project["id"], expected_version=None, synopsis="in flight"
            )
        except BaseException as exc:  # pragma: no cover - failure path
            failures.append(exc)

    service.arm()
    worker = threading.Thread(target=_worker)
    worker.start()
    assert entered.wait(10), "the gated transaction never started"

    releaser = threading.Timer(0.25, release.set)
    releaser.start()
    started = time.perf_counter()
    service.close()
    waited = time.perf_counter() - started
    releaser.join(10)
    worker.join(10)

    assert not failures, f"in-flight operation was broken by close(): {failures}"
    assert waited >= 0.2, (
        "close() did not wait for the in-flight operation "
        f"(returned after {waited:.3f}s)"
    )
    assert service.get_project(project["id"])["synopsis"] == "in flight"
    service.close()


def test_local_writing_service_transaction_error_is_not_masked_by_rollback(tmp_path):
    service = _RollbackFailingService(tmp_path / "writing.db")
    try:
        service.create_project(title="Novel")

        with pytest.raises(ValueError, match="original failure"):
            with service._transaction() as conn:
                conn.execute("UPDATE writing_projects SET title = 'x'")
                raise ValueError("original failure")
    finally:
        service.close()


def test_local_writing_service_memory_mode_serves_multiple_threads(tmp_path):
    service = LocalWritingService(":memory:")
    try:
        errors = []
        created = []

        def _worker(index):
            try:
                created.append(service.create_project(title=f"Novel {index}"))
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert not errors, f"memory-mode service broke under threads: {errors}"
        assert len(created) == 4
        assert len(service.list_projects()) == 4
    finally:
        service.close()
