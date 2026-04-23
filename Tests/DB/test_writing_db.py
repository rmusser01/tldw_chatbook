import json
import sqlite3

import pytest

from tldw_chatbook.DB.Writing_DB import WritingDatabase, WritingDBConflictError


@pytest.fixture()
def db(tmp_path):
    database = WritingDatabase(tmp_path / "writing.db", client_id="test_client")
    yield database
    database.close()


def test_schema_initializes_in_temp_db(tmp_path):
    db_path = tmp_path / "writing.db"
    database = WritingDatabase(db_path, client_id="test_client")
    database.close()

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {
        "schema_version",
        "writing_projects",
        "writing_manuscripts",
        "writing_chapters",
        "writing_scenes",
        "writing_versions",
    }.issubset(tables)


def test_project_create_list_get_update_soft_delete_restore(db):
    project = db.create_project(
        title="First Novel",
        subtitle="A draft",
        author="Ada",
        genre="Speculative",
        settings={"voice": "close-third"},
    )

    assert project["title"] == "First Novel"
    assert project["version"] == 1
    assert json.loads(project["settings_json"]) == {"voice": "close-third"}
    assert db.list_projects() == [project]
    assert db.get_project(project["id"]) == project

    updated = db.update_project(
        project["id"],
        {"title": "Second Novel", "target_word_count": 90000},
        expected_version=1,
    )

    assert updated["title"] == "Second Novel"
    assert updated["target_word_count"] == 90000
    assert updated["version"] == 2

    deleted = db.soft_delete_project(project["id"], expected_version=2)
    assert deleted["deleted"] == 1
    assert db.list_projects() == []
    assert db.get_project(project["id"]) is None

    restored = db.restore_project(project["id"], expected_version=3)
    assert restored["deleted"] == 0
    assert restored["version"] == 4
    assert db.list_projects() == [restored]


def test_manuscript_chapter_scene_hierarchy_and_assignment(db):
    project = db.create_project(title="Project")
    manuscript = db.create_manuscript(project["id"], title="Book One", sort_order=2)
    unassigned_chapter = db.create_chapter(
        project["id"],
        manuscript_id=None,
        title="Loose Chapter",
        sort_order=1,
    )

    assert db.list_manuscripts(project["id"]) == [manuscript]
    assert db.list_chapters(project["id"], manuscript_id=None) == [unassigned_chapter]

    assigned = db.assign_chapter(
        unassigned_chapter["id"],
        manuscript_id=manuscript["id"],
        expected_version=1,
    )
    assert assigned["manuscript_id"] == manuscript["id"]
    assert db.list_chapters(project["id"], manuscript_id=None) == []
    assert db.list_chapters(project["id"], manuscript_id=manuscript["id"]) == [assigned]

    unassigned = db.assign_chapter(
        unassigned_chapter["id"],
        manuscript_id=None,
        expected_version=2,
    )
    assert unassigned["manuscript_id"] is None

    scene_under_chapter = db.create_scene(
        project["id"],
        title="Chapter Scene",
        chapter_id=unassigned["id"],
        body_markdown="# Opening",
    )
    direct_scene = db.create_scene(
        project["id"],
        title="Direct Scene",
        manuscript_id=manuscript["id"],
        chapter_id=None,
        body_markdown="Direct prose",
    )

    assert scene_under_chapter["chapter_id"] == unassigned["id"]
    assert scene_under_chapter["manuscript_id"] is None
    assert direct_scene["chapter_id"] is None
    assert direct_scene["manuscript_id"] == manuscript["id"]
    assert db.list_scenes(project["id"], chapter_id=unassigned["id"]) == [
        scene_under_chapter
    ]
    assert db.list_scenes(
        project["id"], manuscript_id=manuscript["id"], chapter_id=None
    ) == [direct_scene]

    structure = db.get_project_structure(project["id"])
    assert structure["project"]["id"] == project["id"]
    assert structure["manuscripts"][0]["id"] == manuscript["id"]
    assert structure["unassigned_chapters"][0]["id"] == unassigned["id"]
    assert structure["unassigned_chapters"][0]["scenes"][0]["id"] == scene_under_chapter["id"]
    assert structure["manuscripts"][0]["direct_scenes"][0]["id"] == direct_scene["id"]


def test_soft_deleted_records_are_excluded_from_lists_and_present_in_trash(db):
    project = db.create_project(title="Project")
    manuscript = db.create_manuscript(project["id"], title="Book")
    chapter = db.create_chapter(project["id"], manuscript_id=manuscript["id"], title="Chapter")
    scene = db.create_scene(
        project["id"],
        manuscript_id=manuscript["id"],
        chapter_id=None,
        title="Scene",
    )

    db.soft_delete_manuscript(manuscript["id"], expected_version=1)
    db.soft_delete_chapter(chapter["id"], expected_version=1)
    db.soft_delete_scene(scene["id"], expected_version=1)

    assert db.list_manuscripts(project["id"]) == []
    assert db.list_chapters(project["id"], manuscript_id=manuscript["id"]) == []
    assert db.list_scenes(project["id"], manuscript_id=manuscript["id"], chapter_id=None) == []

    trash = db.list_trash(project["id"])
    assert [(item["entity_kind"], item["id"]) for item in trash] == [
        ("manuscript", manuscript["id"]),
        ("chapter", chapter["id"]),
        ("scene", scene["id"]),
    ]


def test_optimistic_version_mismatch_raises_deterministic_exception(db):
    project = db.create_project(title="Project")

    with pytest.raises(WritingDBConflictError, match="version mismatch"):
        db.update_project(project["id"], {"title": "Stale"}, expected_version=2)


def test_manual_version_snapshot_increments_without_changing_working_version(db):
    project = db.create_project(title="Project")
    manuscript = db.create_manuscript(project["id"], title="Book", synopsis="Working")

    version_one = db.create_version("manuscript", manuscript["id"])
    version_two = db.create_version(
        "manuscript",
        manuscript["id"],
        label="Milestone",
        snapshot={"title": "Book", "note": "custom"},
    )

    assert version_one["version_number"] == 1
    assert version_two["version_number"] == 2
    assert json.loads(version_two["snapshot_json"])["note"] == "custom"
    assert db.get_manuscript(manuscript["id"])["version"] == 1

    updated = db.update_manuscript(
        manuscript["id"],
        {"synopsis": "Changed"},
        expected_version=1,
    )
    assert updated["version"] == 2
    assert [item["version_number"] for item in db.list_versions("manuscript", manuscript["id"])] == [
        1,
        2,
    ]


def test_scene_parent_invariants_require_single_same_project_parent(db):
    project = db.create_project(title="Project")
    other_project = db.create_project(title="Other")
    manuscript = db.create_manuscript(project["id"], title="Book")
    chapter = db.create_chapter(project["id"], manuscript_id=manuscript["id"], title="Chapter")
    other_manuscript = db.create_manuscript(other_project["id"], title="Other Book")
    other_chapter = db.create_chapter(
        other_project["id"],
        manuscript_id=other_manuscript["id"],
        title="Other Chapter",
    )

    with pytest.raises(ValueError, match="exactly one"):
        db.create_scene(
            project["id"],
            title="Ambiguous",
            manuscript_id=manuscript["id"],
            chapter_id=chapter["id"],
        )

    with pytest.raises(WritingDBConflictError, match="does not belong to project"):
        db.create_scene(project["id"], title="Cross Manuscript", manuscript_id=other_manuscript["id"])

    with pytest.raises(WritingDBConflictError, match="does not belong to project"):
        db.create_scene(project["id"], title="Cross Chapter", chapter_id=other_chapter["id"])

    scene = db.create_scene(project["id"], title="Valid", chapter_id=chapter["id"])

    with pytest.raises(ValueError, match="exactly one"):
        db.move_scene_local(
            scene["id"],
            manuscript_id=manuscript["id"],
            chapter_id=chapter["id"],
            expected_version=1,
        )

    with pytest.raises(WritingDBConflictError, match="does not belong to project"):
        db.move_scene_local(
            scene["id"],
            manuscript_id=other_manuscript["id"],
            chapter_id=None,
            expected_version=1,
        )


def test_chapter_assignment_requires_same_project_manuscript(db):
    project = db.create_project(title="Project")
    other_project = db.create_project(title="Other")
    manuscript = db.create_manuscript(project["id"], title="Book")
    other_manuscript = db.create_manuscript(other_project["id"], title="Other Book")
    chapter = db.create_chapter(project["id"], manuscript_id=None, title="Loose")

    with pytest.raises(WritingDBConflictError, match="does not belong to project"):
        db.create_chapter(project["id"], manuscript_id=other_manuscript["id"], title="Cross")

    with pytest.raises(WritingDBConflictError, match="does not belong to project"):
        db.assign_chapter(
            chapter["id"],
            manuscript_id=other_manuscript["id"],
            expected_version=1,
        )

    with pytest.raises(WritingDBConflictError, match="does not belong to project"):
        db.update_chapter(
            chapter["id"],
            {"manuscript_id": other_manuscript["id"]},
            expected_version=1,
        )

    assigned = db.assign_chapter(
        chapter["id"],
        manuscript_id=manuscript["id"],
        expected_version=1,
    )
    assert assigned["manuscript_id"] == manuscript["id"]


def test_restore_version_preserves_structural_identity(db):
    project = db.create_project(title="Project")
    other_project = db.create_project(title="Other")
    manuscript = db.create_manuscript(project["id"], title="Book")
    version = db.create_version(
        "manuscript",
        manuscript["id"],
        snapshot={
            "project_id": other_project["id"],
            "title": "Restored Title",
            "synopsis": "Restored Synopsis",
        },
    )

    restored = db.restore_version_to_working_state(version["id"], expected_version=1)

    assert restored["project_id"] == project["id"]
    assert restored["title"] == "Restored Title"
    assert restored["synopsis"] == "Restored Synopsis"


def test_non_scene_versions_reject_markdown_body(db):
    project = db.create_project(title="Project")
    manuscript = db.create_manuscript(project["id"], title="Book")
    chapter = db.create_chapter(project["id"], manuscript_id=manuscript["id"], title="Chapter")

    with pytest.raises(ValueError, match="Only scene versions"):
        db.create_version("chapter", chapter["id"], body_markdown="illegal")

    version = db.create_version(
        "chapter",
        chapter["id"],
        snapshot={"title": "Chapter", "body_markdown": "illegal"},
    )

    assert version["body_markdown"] is None
    assert "body_markdown" not in json.loads(version["snapshot_json"])


def test_reorder_items_rolls_back_as_one_batch_on_version_conflict(db):
    project = db.create_project(title="Project")
    first = db.create_manuscript(project["id"], title="First", sort_order=1)
    second = db.create_manuscript(project["id"], title="Second", sort_order=2)

    with pytest.raises(WritingDBConflictError, match="version mismatch"):
        db.reorder_items(
            "manuscript",
            [first["id"], second["id"]],
            start=10,
            expected_versions={first["id"]: 1, second["id"]: 999},
        )

    assert db.get_manuscript(first["id"])["sort_order"] == 1
    assert db.get_manuscript(first["id"])["version"] == 1
    assert db.get_manuscript(second["id"])["sort_order"] == 2
    assert db.get_manuscript(second["id"])["version"] == 1
