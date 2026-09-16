"""Real-store paging keeps every workflow, revision and draft reachable."""

import asyncio
import json
from threading import Event
from time import perf_counter
from uuid import UUID

import pytest
from textual.widgets import Button, Input, OptionList, TextArea

from Tests.UI.test_workflows_editor import WorkflowEditorHarness, painted_text
from Tests.UI.test_workflows_editor import choose_option as choose_existing
from Tests.Workflows.test_document_complexity import definition, nested_raw
from tldw_chatbook.UI.Workflows_Modules.editor import WorkflowEditor


@pytest.mark.parametrize("compact", [False, True])
async def test_library_names_do_not_project_full_definitions(
    tmp_path, monkeypatch, compact
):
    """Re-projecting every saved body for its name wastes page CPU and memory."""
    harness = WorkflowEditorHarness(tmp_path, seed=False)
    revisions = seed_library(harness)
    with harness.db.transaction() as cursor:
        for revision in revisions:
            raw = json.loads(revision.raw_json)
            raw["opaque"] = "x" * (512 * 1024)
            cursor.execute(
                "UPDATE workflow_revisions SET definition_json=? WHERE revision_id=?",
                (json.dumps(raw), revision.revision_id),
            )
    async with harness.run_test(size=(110 if compact else 160, 48)) as pilot:
        screen = harness.screen
        await settled(pilot, lambda: screen._loaded)
        await harness.workers.wait_for_complete()
        calls = []
        project = harness.workflow_documents.project

        def counted(raw):
            calls.append(len(raw))
            return project(raw)

        monkeypatch.setattr(harness.workflow_documents, "project", counted)
        started = perf_counter()
        if compact:
            screen.query_one("#workflow-library-selector", Button).press()
            await settled(
                pilot,
                lambda: (
                    harness.screen is not screen
                    and not harness.screen.query_one(
                        "#workflow-dialog-choices"
                    ).disabled
                ),
            )
            listing = harness.screen.query_one("#workflow-dialog-choices", OptionList)
        else:
            await screen.controller.load_library()
            await screen._show_library()
            listing = screen.query_one("#workflow-library-list", OptionList)
        labels = [str(listing.get_option_at_index(i).prompt) for i in range(20)]
        assert all(f"Workflow {index:02}" in labels[index] for index in range(20))
        print(
            {
                "compact": compact,
                "page_seconds": perf_counter() - started,
                "project_calls": len(calls),
                "projected_bytes": sum(calls),
            }
        )
        assert calls == [], "Library names must not decode/project entire revisions"


def seed_library(harness, count=23):
    return [
        harness.workflow_documents.create(
            json.dumps(
                {
                    "name": f"Workflow {index:02}",
                    "steps": [],
                    "metadata": {
                        "tldw_workflow": {
                            "format_version": 1,
                            "workflow_id": str(UUID(int=index + 1)),
                            "revision_id": str(UUID(int=index + 101)),
                            "parent_revision_ids": [],
                        }
                    },
                }
            )
        )
        for index in range(count)
    ]


@pytest.mark.parametrize("compact", [False, True])
async def test_admitted_surrogate_name_cannot_block_library_or_selection(
    tmp_path, compact
):
    harness = WorkflowEditorHarness(tmp_path, seed=False)
    ordinary = seed_library(harness, 1)[0]
    unusual = harness.workflow_documents.create(r'{"name":"\ud800","steps":[]}')
    async with harness.run_test(size=(110 if compact else 160, 48)) as pilot:
        screen = harness.screen
        await settled(pilot, lambda: screen._loaded)
        await screen.controller.load((ordinary.workflow_id, ordinary.revision_id))
        assert screen.controller.head == ordinary
        if compact:
            screen.query_one("#workflow-library-selector", Button).press()
            await settled(
                pilot,
                lambda: (
                    harness.screen is not screen
                    and not harness.screen.query_one(
                        "#workflow-dialog-choices"
                    ).disabled
                ),
            )
            listing = harness.screen.query_one("#workflow-dialog-choices", OptionList)
        else:
            await screen._show_library()
            listing = screen.query_one("#workflow-library-list", OptionList)
        assert listing.option_count == 2
        assert "Workflow 00" in str(listing.get_option_at_index(0).prompt)
        assert "\ud800" in str(listing.get_option_at_index(1).prompt)
        assert harness.workflow_documents.get_head(unusual.workflow_id) == unusual


async def settled(pilot, predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await pilot.pause()


async def choose_option(harness, pilot, option_id):
    def ready():
        choices = list(harness.screen.query("#workflow-dialog-choices"))
        return bool(
            choices
            and not choices[0].disabled
            and any(
                choices[0].get_option_at_index(index).id == option_id
                for index in range(choices[0].option_count)
            )
        )

    await settled(pilot, ready)
    await choose_existing(harness, pilot, option_id)


async def test_slow_old_search_cannot_replace_newer_library_results(
    tmp_path, monkeypatch
):
    harness = WorkflowEditorHarness(tmp_path, seed=False)
    revisions = seed_library(harness)
    async with harness.run_test(size=(160, 48)) as pilot:
        controller = harness.screen.controller
        await settled(pilot, lambda: harness.screen._loaded)
        read = harness.workflow_documents.list_workflow_summaries
        started, release = Event(), Event()

        def delayed(**kwargs):
            if kwargs.get("query") == "Workflow 00":
                started.set()
                assert release.wait(5)
            return read(**kwargs)

        monkeypatch.setattr(
            harness.workflow_documents, "list_workflow_summaries", delayed
        )
        old = asyncio.create_task(controller.load_library(query="Workflow 00"))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            await controller.load_library(query="Workflow 22")
        finally:
            release.set()
        await old
        assert controller.library_rows == (
            ("Workflow 22", revisions[-1].workflow_id, revisions[-1].revision_id),
        )
        assert controller.library_query == "Workflow 22"


@pytest.mark.parametrize("compact", [False, True])
async def test_oversized_search_recovers_without_changing_open_draft(tmp_path, compact):
    harness = WorkflowEditorHarness(tmp_path, seed=False)
    revisions = seed_library(harness)
    async with harness.run_test(size=(110 if compact else 160, 48)) as pilot:
        screen = harness.screen
        await settled(pilot, lambda: screen._loaded)
        draft = screen.controller.draft
        if compact:
            screen.query_one("#workflow-library-selector", Button).press()
            await settled(
                pilot,
                lambda: (
                    harness.screen is not screen
                    and not harness.screen.query_one(
                        "#workflow-dialog-choices"
                    ).disabled
                ),
            )
        search = harness.screen.query_one(
            "#workflow-page-search" if compact else "#workflow-library-search", Input
        )
        search.value = "x" * 513
        await settled(pilot, lambda: "Unable to load" in painted_text(harness.screen))
        assert screen.controller.draft == draft
        search.value = "Workflow 22"
        await settled(
            pilot,
            lambda: (
                "Unable to load" not in painted_text(harness.screen)
                and "Workflow 22" in painted_text(harness.screen)
            ),
        )
        listing = harness.screen.query_one(
            "#workflow-dialog-choices" if compact else "#workflow-library-list",
            OptionList,
        )
        assert not listing.disabled and listing.option_count == 1
        assert screen.controller.draft == draft
        assert (
            harness.workflow_documents.get_head(revisions[-1].workflow_id)
            == revisions[-1]
        )


async def test_legacy_over_limit_definition_remains_selectable_and_exportable(tmp_path):
    harness = WorkflowEditorHarness(tmp_path, seed=False)
    base = seed_library(harness, 1)[0]
    raw = (
        nested_raw(1000)
        .replace("11111111-1111-4111-8111-111111111111", base.workflow_id)
        .replace("22222222-2222-4222-8222-222222222222", base.revision_id)
    )
    with harness.db.transaction() as cursor:
        cursor.execute(
            "UPDATE workflow_revisions SET definition_json = ? WHERE revision_id = ?",
            (raw, base.revision_id),
        )
    async with harness.run_test(size=(110, 36)) as pilot:
        screen = harness.screen
        await settled(pilot, lambda: screen._loaded)
        editor = screen.query_one(WorkflowEditor)
        await settled(pilot, lambda: editor.read_only)
        assert editor.query_one("#workflow-raw-json", TextArea).text == raw
        assert screen.controller.head.raw_json == raw
        assert not screen.query_one("#workflow-export", Button).disabled
        assert screen.query_one("#workflow-save-revision", Button).disabled
        assert screen.query_one("#workflow-add-step", Button).disabled
        screen._more()
        await settled(
            pilot,
            lambda: (
                harness.screen is not screen
                and bool(harness.screen.query("#workflow-dialog-choices"))
            ),
        )
        choices = harness.screen.query_one("#workflow-dialog-choices", OptionList)
        assert not {"add", "discard", "recover", "edit-history"}.intersection(
            choices.get_option_at_index(index).id
            for index in range(choices.option_count)
        )
        await pilot.press("escape")


async def test_large_raw_edit_preserves_text_and_reuses_controls(tmp_path):
    harness = WorkflowEditorHarness(tmp_path, seed=False)
    saved = harness.workflow_documents.create(json.dumps(definition(500)))
    async with harness.run_test(size=(160, 48)) as pilot:
        screen = harness.screen
        editor = screen.query_one(WorkflowEditor)
        await settled(pilot, lambda: screen._loaded and editor.draft is not None)
        await pilot.pause()
        first_step = editor.query_one("#workflow-overview-0", Button)
        delays = []
        finished = asyncio.Event()

        async def heartbeat():
            previous = perf_counter()
            while not finished.is_set():
                await asyncio.sleep(0.005)
                current = perf_counter()
                delays.append(current - previous)
                previous = current

        beating = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        started = perf_counter()
        try:
            await screen.raw_changed(
                WorkflowEditor.RawEdited(
                    saved.raw_json + " ", (saved.workflow_id, saved.revision_id)
                )
            )
        finally:
            elapsed = perf_counter() - started
            finished.set()
            await beating
        assert harness.workflow_drafts.current.raw_text == saved.raw_json + " "
        assert len(editor.document["steps"]) == 500
        assert editor.query_one("#workflow-overview-0", Button) is first_step
        # A fast reconciliation may complete within one heartbeat interval.
        # Control identity is the deterministic guard; timings are diagnostic.
        assert delays
        print({"raw_edit_seconds": elapsed, "max_heartbeat_gap_seconds": max(delays)})


async def test_library_pages_search_and_off_page_head_keep_pending_draft(tmp_path):
    harness = WorkflowEditorHarness(tmp_path, seed=False)
    revisions = seed_library(harness)
    async with harness.run_test(size=(160, 48)) as pilot:
        screen = harness.screen
        await settled(pilot, lambda: screen._loaded)
        listing = screen.query_one("#workflow-library-list", OptionList)
        await settled(pilot, lambda: listing.option_count == 20)
        screen.query_one("#workflow-library-next", Button).press()
        await settled(pilot, lambda: listing.option_count == 3)
        listing.highlighted = 2
        listing.focus()
        await pilot.press("enter")
        await settled(
            pilot,
            lambda: (
                harness.workflow_drafts.current.workflow_id == revisions[-1].workflow_id
            ),
        )
        pending = harness.workflow_drafts.update('{"pending":')
        search = screen.query_one("#workflow-library-search", Input)
        search.value = "Workflow 00"
        await settled(pilot, lambda: listing.option_count == 1)
        assert listing.get_option_at_index(0).id == revisions[0].workflow_id
        assert harness.workflow_drafts.current == pending
        assert screen.controller.head == revisions[-1]
        assert not screen.query_one("#workflow-export", Button).disabled
        search.value = "no match"
        await settled(pilot, lambda: listing.option_count == 0)
        assert screen.query_one("#workflow-library-previous", Button).disabled
        assert screen.query_one("#workflow-library-next", Button).disabled
        assert harness.workflow_drafts.current == pending


@pytest.mark.parametrize("size", [(110, 36), (60, 20)])
async def test_compact_library_and_history_pages_reach_final_item(tmp_path, size):
    harness = WorkflowEditorHarness(tmp_path, seed=False)
    revisions = seed_library(harness)
    latest = revisions[-1]
    for index in range(22):
        documents = harness.workflow_documents
        documents.put_draft(
            latest.workflow_id,
            latest.revision_id,
            documents.edit_field(latest.raw_json, "/name", f"History {index}"),
            1,
        )
        latest = documents.save_revision(latest.workflow_id, latest.revision_id, 1)
    async with harness.run_test(size=size) as pilot:
        screen = harness.screen
        await settled(pilot, lambda: screen._loaded)
        screen.query_one("#workflow-library-selector", Button).press()
        await choose_option(harness, pilot, "__next_page__")
        await settled(
            pilot,
            lambda: (
                harness.screen.query_one(
                    "#workflow-dialog-choices", OptionList
                ).option_count
                == 4
            ),
        )
        await choose_option(
            harness, pilot, latest.workflow_id + ":" + latest.revision_id
        )
        await settled(
            pilot, lambda: harness.screen is screen and screen.controller.head == latest
        )
        screen._more_selected("versions")
        await choose_option(harness, pilot, "__next_page__")
        await settled(
            pilot,
            lambda: (
                harness.screen.query_one(
                    "#workflow-dialog-choices", OptionList
                ).option_count
                == 4
            ),
        )
        await choose_option(harness, pilot, latest.revision_id)
        await settled(
            pilot,
            lambda: harness.screen is screen and screen.controller.inspection == latest,
        )
        await settled(pilot, lambda: not screen._busy)
        screen._more_selected("return")
        await settled(pilot, lambda: not screen._busy)
        screen._more_selected("local-drafts")
        await choose_option(harness, pilot, "__next_page__")
        await settled(
            pilot,
            lambda: (
                getattr(harness.screen, "page_offset", 0) == 20
                and not harness.screen.query_one(
                    "#workflow-dialog-choices", OptionList
                ).disabled
            ),
        )
        choices = harness.screen.query_one("#workflow-dialog-choices", OptionList)
        target = choices.get_option_at_index(0).id
        await choose_option(harness, pilot, target)
        await settled(
            pilot,
            lambda: (
                harness.screen is screen
                and harness.workflow_drafts.current.base_revision_id == target
            ),
        )
        assert screen.controller.head == latest
