"""Snippet import must preserve the live dataset-name edit (TASK-36037)."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_evals_snippet_editor import (
    EvalsHarness,
    _FakeAppInstance,
    _drive_import,
    _make_dataset,
    _snip,
)
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evaluations_Interop.evaluation_normalizers import (
    RESERVED_LOCAL_DATASET_SAMPLES_KEY,
)
from tldw_chatbook.UI.Evals.snippet_editor import SnippetEditor


async def _select_dataset(app, pilot, dataset_id):
    await pilot.pause()
    screen = app.screen
    screen.select(kind="dataset", id=dataset_id)
    await pilot.pause()
    for worker in list(screen.workers):
        if worker.node is screen and worker.group in {
            "evals-selection-regions",
            "evals-store-skills-probe",
        }:
            await worker.wait()
    await pilot.pause()
    return screen.query_one("#evals-snippet-editor", SnippetEditor)


async def _edit_name(editor, pilot, draft):
    field = editor.query_one("#evals-dataset-rename-input", Input)
    field.focus()
    await pilot.press("home", "shift+end", "backspace", *draft)
    await pilot.press("home", "right", "shift+right")
    await pilot.pause()
    assert field.value == draft
    assert field.has_focus
    assert field.cursor_position == 2
    assert field.selection.start == 1
    assert field.selection.end == 2
    return field


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix,content,draft,groups",
    [
        ("txt", "anchor \n", "   ", 1),
        ("csv", 'text,group\n"anchor ",imported\n', "  [/][name]  ", 2),
        ("json", '[{"text":"anchor ","group":"imported"}]', " taken ", 2),
    ],
    ids=["txt-whitespace", "csv-markup", "json-collision"],
)
@private_profile_test
async def test_import_preserves_live_name_edit(
    request, tmp_path, suffix, content, draft, groups
):
    """Whole-editor recomposition loses raw drafts, selection and focus."""
    db = EvalsDB(db_path=":memory:", client_id="snippet-drafts")
    try:
        _make_dataset(db, "taken", [])
        dataset_id = _make_dataset(db, "Stored name", [_snip("anchor", group="seed")])
        app = EvalsHarness(_FakeAppInstance(db))
        path = tmp_path / f"incoming.{suffix}"
        path.write_text(content, encoding="utf-8")
        async with app.run_test(size=(160, 45)) as pilot:
            editor = await _select_dataset(app, pilot, dataset_id)
            field = await _edit_name(editor, pilot, draft)
            selection = field.selection
            heading = editor.query_one("#evals-detail-dataset-name")
            rename_row = editor.query_one("#evals-dataset-rename-row")
            await _drive_import(editor, pilot, path)
            stored = db.get_dataset(dataset_id)
            assert stored["name"] == "Stored name"
            assert stored["metadata"]["sample_count"] == 2
            samples = stored["metadata"][RESERVED_LOCAL_DATASET_SAMPLES_KEY]
            assert [sample["text"] for sample in samples] == ["anchor", "anchor "]
            assert samples[-1]["group"] == (None if suffix == "txt" else "imported")
            assert str(editor.query_one("#evals-snippet-editor-summary").render()) == (
                f"inline · 2 snippets · {groups} {'group' if groups == 1 else 'groups'}"
            )
            assert (
                str(editor.query_one("#evals-snippet-warnings-summary").render())
                == "2 warnings"
            )
            assert "anchor␣" in str(editor.query_one("#evals-snippet-text-1").render())
            assert "exact dup of 1" in str(
                editor.query_one("#evals-snippet-meta-1").render()
            )
            assert app.app_instance.notifications[-1] == (
                "Imported 1 snippet(s).",
                "information",
            )
            current = editor.query_one("#evals-dataset-rename-input", Input)
            assert current.value == draft
            assert current is field
            assert editor.query_one("#evals-detail-dataset-name") is heading
            assert editor.query_one("#evals-dataset-rename-row") is rename_row
            assert app.focused is field
            assert field.has_focus
            assert field.cursor_position == 2
            assert field.selection == selection
    finally:
        db.close()


@pytest.mark.asyncio
@private_profile_test
async def test_cancelled_and_failed_imports_keep_name_edit(request, tmp_path):
    """Cancellation and worker parse/read failures must leave the draft alone."""
    db = EvalsDB(db_path=":memory:", client_id="snippet-draft-errors")
    try:
        dataset_id = _make_dataset(db, "Stored name", [_snip("anchor")])
        app = EvalsHarness(_FakeAppInstance(db))
        malformed = tmp_path / "malformed.json"
        malformed.write_text("{", encoding="utf-8")
        unreadable = tmp_path / "unreadable.txt"
        unreadable.write_bytes(b"\xff")
        async with app.run_test(size=(160, 45)) as pilot:
            editor = await _select_dataset(app, pilot, dataset_id)
            field = await _edit_name(editor, pilot, "  unfinished  ")
            selection = field.selection
            for path in (None, malformed, unreadable):
                before = len(app.app_instance.notifications)
                await _drive_import(editor, pilot, path)
                assert field is editor.query_one("#evals-dataset-rename-input")
                assert field.value == "  unfinished  "
                assert app.focused is field
                assert field.selection == selection
                if path is None:
                    assert len(app.app_instance.notifications) == before
                else:
                    assert app.app_instance.notifications[-1][1] == "error"
                stored = db.get_dataset(dataset_id)
                assert stored["name"] == "Stored name"
                assert stored["metadata"]["sample_count"] == 1
    finally:
        db.close()


@pytest.mark.asyncio
@private_profile_test
async def test_import_then_explicit_rename_validates_and_updates_catalog(
    request, tmp_path
):
    """Import cannot commit a name; Rename must still validate and publish it."""
    db = EvalsDB(db_path=":memory:", client_id="snippet-draft-rename")
    try:
        _make_dataset(db, "taken", [])
        dataset_id = _make_dataset(db, "Stored name", [])
        app = EvalsHarness(_FakeAppInstance(db))
        path = tmp_path / "incoming.json"
        path.write_text(
            '[{"text":"first","group":"new"},{"text":""}]', encoding="utf-8"
        )
        async with app.run_test(size=(160, 45)) as pilot:
            editor = await _select_dataset(app, pilot, dataset_id)
            assert editor.query_one("#evals-snippet-empty")
            field = await _edit_name(editor, pilot, "   ")
            await _drive_import(editor, pilot, path)
            assert field.value == "   "
            assert not editor.query("#evals-snippet-empty")
            assert (
                str(editor.query_one("#evals-snippet-editor-summary").render())
                == "inline · 1 snippet · 1 group"
            )
            assert (
                str(editor.query_one("#evals-snippet-warnings-summary").render())
                == "No warnings"
            )
            assert app.app_instance.notifications[-1] == (
                "Imported 1 snippet(s); skipped 1 invalid entry.",
                "information",
            )
            for draft, severity in (("   ", "warning"), ("taken", "error")):
                await _edit_name(editor, pilot, draft)
                assert await pilot.click("#evals-dataset-rename")
                await pilot.pause()
                assert db.get_dataset(dataset_id)["name"] == "Stored name"
                assert field.value == draft
                assert app.app_instance.notifications[-1][1] == severity
            await _edit_name(editor, pilot, "  Renamed dataset  ")
            assert await pilot.click("#evals-dataset-rename")
            await pilot.pause()
            assert db.get_dataset(dataset_id)["name"] == "Renamed dataset"
            assert (
                str(app.screen.query_one("#evals-detail-dataset-name").render())
                == "Renamed dataset"
            )
            labels = [
                str(button.label)
                for button in app.screen.query(Button)
                if (button.id or "").startswith("evals-rail-row-datasets-")
            ]
            assert any("Renamed dataset" in label for label in labels)
    finally:
        db.close()


@pytest.mark.asyncio
@private_profile_test
async def test_ordinary_import_keeps_name_and_table_scroll_owner(request, tmp_path):
    """Results nesting must keep Import pinned while the table can scroll."""
    db = EvalsDB(db_path=":memory:", client_id="snippet-layout")
    try:
        dataset_id = _make_dataset(db, "Stored name", [])
        app = EvalsHarness(_FakeAppInstance(db))
        path = tmp_path / "incoming.txt"
        path.write_text("\n".join(f"row {i}" for i in range(60)), encoding="utf-8")
        async with app.run_test(size=(160, 45)) as pilot:
            editor = await _select_dataset(app, pilot, dataset_id)
            await _drive_import(editor, pilot, path)
            assert (
                editor.query_one("#evals-dataset-rename-input", Input).value
                == "Stored name"
            )
            assert db.get_dataset(dataset_id)["name"] == "Stored name"
            assert db.get_dataset(dataset_id)["metadata"]["sample_count"] == 60
            table = editor.query_one("#evals-snippet-table")
            button = editor.query_one("#evals-import-snippets", Button)
            pane = app.screen.query_one("#evals-detail-pane")
            assert str(table.styles.overflow_y) == "auto"
            assert table.virtual_size.height > table.size.height
            assert pane.region.contains_region(button.region)
            button_region = button.region
            last_row = editor.query_one("#evals-snippet-text-59")
            assert not table.region.contains_region(last_row.region)
            table.scroll_end(animate=False)
            await pilot.pause()
            assert table.scroll_offset.y > 0
            assert table.region.contains_region(last_row.region)
            assert button.region == button_region
            assert await pilot.click("#evals-import-snippets")
            assert app.screen is not pane.screen
    finally:
        db.close()
