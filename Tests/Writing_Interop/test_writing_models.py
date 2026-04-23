import pytest

from tldw_chatbook.Writing_Interop.writing_models import (
    WritingChapter,
    WritingDraft,
    WritingManuscript,
    WritingScene,
    WritingVersion,
)


def test_project_is_required_for_entities():
    with pytest.raises(ValueError, match="project_id is required"):
        WritingManuscript(source="local", id="m-1", project_id="", title="M")


def test_chapter_allows_project_level_unassigned_state():
    chapter = WritingChapter(
        source="local",
        id="c-1",
        project_id="p-1",
        manuscript_id=None,
        title="Chapter 1",
    )
    assert chapter.manuscript_id is None


def test_scene_direct_manuscript_requires_manuscript_id_and_no_chapter_id():
    with pytest.raises(ValueError, match="Direct manuscript scene requires manuscript_id"):
        WritingScene(
            source="local",
            id="s-1",
            project_id="p-1",
            chapter_id=None,
            manuscript_id=None,
            title="Scene 1",
        )


def test_scene_under_unassigned_chapter_allows_chapter_without_manuscript():
    scene = WritingScene(
        source="local",
        id="s-2",
        project_id="p-1",
        chapter_id="c-unassigned",
        manuscript_id=None,
        title="Scene 2",
    )
    assert scene.chapter_id == "c-unassigned"
    assert scene.manuscript_id is None


def test_container_drafts_do_not_accept_body_markdown():
    with pytest.raises(ValueError, match="Only scene drafts"):
        WritingDraft(
            source="local",
            entity_kind="chapter",
            entity_id="chapter-1",
            project_id="project-1",
            metadata={"title": "Chapter 1"},
            body_markdown="# illegal",
        )


def test_scene_draft_accepts_body_markdown():
    draft = WritingDraft(
        source="local",
        entity_kind="scene",
        entity_id="scene-1",
        project_id="project-1",
        metadata={"title": "Scene 1"},
        body_markdown="# legal",
    )
    assert draft.body_markdown == "# legal"


def test_non_scene_version_rejects_body_markdown():
    with pytest.raises(ValueError, match="Only scene versions"):
        WritingVersion(
            source="local",
            id="v-1",
            entity_kind="manuscript",
            entity_id="m-1",
            project_id="p-1",
            version_number=1,
            body_markdown="illegal",
        )
