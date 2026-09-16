"""Real destination, real private draft owner, and production CSS at all widths."""

import asyncio
from pathlib import Path
from threading import Event
from typing import ClassVar

import pytest
from textual.binding import Binding
from textual.widgets import Button, Collapsible, Input, OptionList, TextArea

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen
from tldw_chatbook.UI.Workflows_Modules.controller import visible_pane_ids
from tldw_chatbook.UI.Workflows_Modules.editor import WorkflowEditor
from tldw_chatbook.UI.Workflows_Modules.library import ChoiceModal, StepChooser
from tldw_chatbook.UI.Workflows_Modules.reference_picker import (
    ReferencePicker,
    reference_choices,
)
from tldw_chatbook.Workflows.document_service import DocumentService
from tldw_chatbook.Workflows.draft_session import DraftSession
from tldw_chatbook.Workflows.models import DraftConflict, Issue


class WorkflowEditorHarness(ConsolidatedCSSApp):
    CSS_PATH: ClassVar = [
        BUNDLED_STYLESHEET,
        BUNDLED_STYLESHEET.parent / "screen_feature_workflows.tcss",
    ]
    BINDINGS: ClassVar = [Binding("f6", "focus_next_workbench_pane", "Next pane")]

    def __init__(self, tmp_path, *, seed=True):
        super().__init__()
        self.db = WorkflowsDB(tmp_path / "workflows.sqlite3")
        self.workflow_documents = DocumentService(self.db)
        if seed and not self.workflow_documents.list_workflows():
            fixture = Path(__file__).parents[1] / "fixtures/workflows/file_to_note.json"
            self.workflow_documents.create(fixture.read_text())
        self.workflow_drafts = DraftSession(self.workflow_documents)

    async def on_mount(self):
        await self.push_screen(WorkflowsScreen(self))

    async def on_unmount(self):
        await self.workflow_drafts.close()
        self.db.close()

    def action_focus_next_workbench_pane(self):
        self.screen.action_focus_next_workbench_pane()


def assert_hit(screen, widget):
    region = widget.region
    assert region.width and region.height
    x, y = region.x + region.width // 2, region.y + (region.height - 1) // 2
    painted, _ = screen.get_widget_at(x, y)
    assert painted is widget or widget in painted.ancestors, "\n".join(
        f"{item} {item.region} h={item.styles.height} min={item.styles.min_height} margin={item.styles.margin}"
        for item in screen.walk_children()
    )


def painted_text(screen):
    return "\n".join(strip.text for strip in screen._compositor.render_strips())


def svg_text_contrast(svg, label):
    """Measure exported effective ink over the rectangle actually behind it."""
    import re
    from xml.etree import ElementTree

    tree = ElementTree.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    node = next(
        node
        for node in tree.findall(".//s:text", ns)
        if " ".join((node.text or "").split()) == label
    )
    css = tree.find("s:style", ns).text
    rule = re.search(r"\." + re.escape(node.attrib["class"]) + r"\s*\{([^}]+)", css)[1]
    foreground = re.search(r"fill:\s*(#[a-fA-F0-9]{6})", rule)[1]
    x = float(node.attrib["x"]) + float(node.attrib["textLength"]) / 2
    y = float(node.attrib["y"]) - 5
    backgrounds = [
        rect.attrib["fill"]
        for rect in tree.findall(".//s:rect", ns)
        if rect.attrib.get("fill", "").startswith("#")
        and float(rect.attrib.get("x", 0))
        <= x
        < float(rect.attrib.get("x", 0)) + float(rect.attrib["width"])
        and float(rect.attrib.get("y", 0))
        <= y
        < float(rect.attrib.get("y", 0)) + float(rect.attrib["height"])
    ]
    background = backgrounds[-1]

    def luminance(color):
        channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [
            v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            for v in channels
        ]
        return sum(
            v * weight
            for v, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
        )

    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    return foreground, background, (light + 0.05) / (dark + 0.05)


@pytest.mark.parametrize(
    "size,label,floor",
    [
        ((160, 48), "Run", 3),
        ((110, 36), "Run", 3),
        ((60, 20), "Run", 3),
        ((160, 48), "Search workflows", 4.5),
    ],
)
async def test_effective_workflows_control_contrast(
    tmp_path, monkeypatch, size, label, floor
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=size) as pilot:
        await pilot.pause()
        await select_step(harness, pilot, "summarize")
        control = harness.screen.query_one(
            "#workflow-run" if label == "Run" else "#workflow-library-search"
        )
        assert_hit(harness.screen, control)
        foreground, background, ratio = svg_text_contrast(
            harness.export_screenshot(simplify=True), label
        )
        assert ratio >= floor, (label, foreground, background, ratio)


async def select_step(harness, pilot, step="prepare"):
    editor = harness.screen.query_one(WorkflowEditor)
    editor.show_step(step)
    await pilot.pause()
    await harness.workers.wait_for_complete()
    await pilot.pause()
    assert harness.screen.controller.section == "step:" + step, harness.screen._error
    assert not harness.screen._busy
    return editor


def field_for(editor, pointer):
    identifier = next(
        identifier
        for identifier, (bound, _) in editor.field_bindings.items()
        if bound == pointer
    )
    return editor.query_one("#" + identifier)


async def choose_option(harness, pilot, option_id):
    # A completed control worker may have only queued the modal's mount.
    async with asyncio.timeout(5):
        while not harness.screen.query("#workflow-dialog-choices"):
            await pilot.pause()
    options = harness.screen.query_one("#workflow-dialog-choices", OptionList)
    options.highlighted = options.get_option_index(option_id)
    options.focus()
    await pilot.pause()
    assert_hit(harness.screen, options)
    label = str(options.get_option(option_id).prompt).split(" · ")[0]
    assert " ".join(label.split()) in " ".join(painted_text(harness.screen).split())
    await pilot.press("enter")
    await pilot.pause()
    await harness.workers.wait_for_complete()
    await pilot.pause()


async def more_action(harness, pilot, action):
    assert await pilot.click("#workflow-more")
    await pilot.pause()
    await choose_option(harness, pilot, action)


async def test_raw_edits_reconcile_values_schema_and_summaries_without_synthetic_edits(
    tmp_path,
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot)
        owner, docs = harness.workflow_drafts, harness.workflow_documents
        raw = editor.query_one("#workflow-raw-json", TextArea)
        editor.show_issue(Issue("", "raw", ""))
        await pilot.pause()
        changed = docs.edit_field(
            owner.current.raw_text, "/steps/1/config/template", "Raw replacement"
        )
        generation = owner.current.generation
        raw.load_text(changed)
        raw.move_cursor((0, 12))
        selection = raw.selection
        await pilot.pause()
        assert field_for(editor, "/steps/1/config/template").text == "Raw replacement"
        assert harness.focused is raw and raw.selection == selection
        assert owner.current.generation == generation + 1
        assert_hit(harness.screen, raw)
        raw.load_text('{"steps":[')
        await pilot.pause()
        assert field_for(editor, "/steps/1/config/template").disabled
        changed = docs.edit_field(
            changed,
            "/steps/1/config",
            '{"action":"create","title":"Raw title","content":"Raw content"}',
            as_json=True,
        )
        changed = docs.edit_field(changed, "/steps/1/type", "notes")
        changed = docs.edit_field(changed, "/steps/1/name", "Renamed raw step")
        changed = docs.edit_field(changed, "/steps/1/retry", "2", as_json=True)
        raw.load_text(changed)
        raw.move_cursor((0, 15))
        selection = raw.selection
        generation = owner.current.generation
        await pilot.pause()
        assert editor.query_one("#workflow-raw-json") is raw
        assert harness.focused is raw and raw.selection == selection
        assert field_for(editor, "/steps/1/config/title").value == "Raw title"
        assert not any(
            pointer.endswith("/template")
            for pointer, _ in editor.field_bindings.values()
        )
        assert (
            "retry 2"
            in editor.query_one("#workflow-section-execution", Collapsible).title
        )
        nav = harness.screen.query_one("#workflow-navigation-list", OptionList)
        assert any(
            "Renamed raw step" in str(nav.get_option_at_index(i).prompt)
            for i in range(nav.option_count)
        )
        assert owner.current.generation == generation + 1
        title = field_for(editor, "/steps/1/config/title")
        title.value += " edited"
        await pilot.pause()
        assert (
            docs.field_text(
                owner.current.raw_text, "/steps/1/config/title", as_json=False
            )
            == "Raw title edited"
        )


async def test_field_refresh_keeps_unrepresented_value_source_disabled(tmp_path):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        owner, documents = harness.workflow_drafts, harness.workflow_documents
        owner.update(
            documents.edit_field(
                owner.current.raw_text, "/steps/2/config/provider", "123", as_json=True
            )
        )
        editor = await select_step(harness, pilot, "summarize")
        provider = field_for(editor, "/steps/2/config/provider")
        reference = editor.query_one("#" + provider.id + "-reference", Button)
        assert provider.disabled and reference.disabled
        field_for(editor, "/steps/2/config/max_tokens").value = "256"
        await pilot.pause()
        assert provider.disabled and reference.disabled
        assert (
            documents.field_text(owner.current.raw_text, "/steps/2/config/provider")
            == "123"
        )


@pytest.mark.parametrize(
    "section,pointer,partial,replacement",
    [
        ("step:summarize", "/steps/2/config/max_tokens", "", "256"),
        ("step:ingest", "/steps/0/config/extraction/extract_text", "t", "true"),
        ("inputs", "/inputs", '{"draft":', '{"draft":1}'),
    ],
)
async def test_incomplete_field_is_repairable_across_fresh_screens(
    tmp_path, section, pointer, partial, replacement
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        screen = harness.screen
        await screen._select_section(section)
        await pilot.pause()
        editor = screen.query_one(WorkflowEditor)
        field = field_for(editor, pointer)
        field.focus()
        field.scroll_visible(animate=False)
        await pilot.pause()
        if isinstance(field, TextArea):
            field.load_text(partial)
        else:
            field.value = partial
        await pilot.pause()
        assert harness.workflow_drafts.current.error
        assert not field.disabled
        assert_hit(screen, field)
        assert all(
            widget.disabled
            for key in editor.field_bindings
            if (widget := editor.query_one("#" + key)) is not field
        )
        await harness.workflow_drafts.flush()
        snapshot = screen.save_state()
        await harness.pop_screen()
        fresh = WorkflowsScreen(harness)
        fresh.restore_state(snapshot)
        await harness.push_screen(fresh)
        await pilot.pause()
        await harness.workers.wait_for_complete()
        editor = fresh.query_one(WorkflowEditor)
        field = field_for(editor, pointer)
        assert not field.disabled
        assert (field.text if isinstance(field, TextArea) else field.value) == partial
        field.focus()
        field.scroll_visible(animate=False)
        if isinstance(field, TextArea):
            field.load_text(replacement)
        else:
            field.value = ""
            await pilot.pause()
            await pilot.press(*replacement)
        await pilot.pause()
        assert harness.workflow_drafts.current.error is None, fresh._error
        assert (
            harness.workflow_documents.field_text(
                harness.workflow_drafts.current.raw_text, pointer
            )
            == replacement
        )
        assert_hit(fresh, field)


async def test_restart_protected_parseable_fragment_needs_explicit_advanced_acceptance(
    tmp_path,
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        owner = harness.workflow_drafts
        original = owner.current
        pending = owner.update_field("/steps/2/config/max_tokens", '0,"injected":true')
        await owner.flush()
    reopened = WorkflowEditorHarness(tmp_path)
    async with reopened.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        owner = reopened.workflow_drafts
        assert owner.current == pending and owner.field_edit is None
        assert reopened.screen.query_one("#workflow-save-revision", Button).disabled
        editor = await select_step(reopened, pilot, "summarize")
        assert field_for(editor, "/steps/2/config/max_tokens").disabled
        editor.show_issue(Issue("", "repair", ""))
        await pilot.pause()
        raw = editor.query_one("#workflow-raw-json", TextArea)
        raw.load_text(original.raw_text)
        await pilot.pause()
        assert (
            owner.current.error
            and owner.current.last_valid_json == original.last_valid_json
        )
        accept = editor.query_one("#workflow-repair-raw", Button)
        accept.scroll_visible(animate=False)
        await pilot.pause()
        assert_hit(reopened.screen, accept)
        assert await pilot.click(accept)
        await pilot.pause()
        assert isinstance(reopened.screen, ChoiceModal)
        await choose_option(reopened, pilot, "accept")
        assert owner.current.error is None
        assert owner.current.raw_text == original.raw_text
        assert owner.current.generation > pending.generation
        assert not reopened.screen.query_one("#workflow-save-revision", Button).disabled


async def test_history_copy_refuses_head_dirtied_at_owner_boundary_without_touching_source(
    tmp_path, monkeypatch
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        owner, docs = harness.workflow_drafts, harness.workflow_documents
        base = owner.base
        head = await owner.save_revision()
        await owner.select(base.workflow_id, base.revision_id)
        source_draft = owner.update('{"keep old draft":')
        await owner.flush()
        await harness.screen._inspect(base.revision_id)
        copy = docs.copy_revision_to_head
        dirty = []

        def interleave(source, target):
            dirty.append(
                docs.put_draft(
                    head.workflow_id, head.revision_id, '{"other writer":', 1
                )
            )
            return copy(source, target)

        monkeypatch.setattr(docs, "copy_revision_to_head", interleave)
        await more_action(harness, pilot, "edit-history")
        assert dirty
        assert docs.get_draft(head.workflow_id, head.revision_id) == dirty[0]
        assert docs.get_draft(base.workflow_id, base.revision_id) == source_draft
        assert owner.current == source_draft
        assert harness.screen.controller.inspection == base


@pytest.mark.parametrize(
    "pointer,want",
    [
        ("/inputs/repeat", '"decoy"'),
        ("/inputs/items/0/repeat", '"first"'),
        ("/inputs/items/1/a~1b~0c/repeat", '"target"'),
    ],
)
async def test_nested_json_issue_selects_exact_member_in_owning_field(
    tmp_path, pointer, want
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        owner, docs = harness.workflow_drafts, harness.workflow_documents
        owner.update(
            docs.edit_field(
                owner.current.raw_text,
                "/inputs",
                '{"repeat":"decoy","items":[{"repeat":"first"},{"a/b~c":{"repeat":"target"}}]}',
                as_json=True,
            )
        )
        await harness.screen._select_section("inputs")
        await pilot.pause()
        editor = harness.screen.query_one(WorkflowEditor)
        field = field_for(editor, "/inputs")
        field.move_cursor((0, 0))
        editor.show_issue(Issue(pointer, "nested", "Inspect nested value"))
        await pilot.pause()
        assert harness.focused is field
        assert field.selected_text == want
        assert_hit(harness.screen, field)


@pytest.mark.parametrize(
    "extra,size", [(0, (110, 36)), (12, (110, 36)), (12, (60, 20))]
)
async def test_admissible_reorder_confirmation_shows_dependencies_and_rejects_new_typing(
    tmp_path, extra, size
):
    import json

    from Tests.Workflows.helpers import prompt_definition

    harness = WorkflowEditorHarness(tmp_path, seed=False)
    definition = prompt_definition()
    definition["steps"] += [
        {"id": "side", "type": "prompt", "config": {"template": "independent"}},
        {"id": "end", "type": "prompt", "config": {"template": "done"}},
    ]
    definition["steps"][0]["on_success"] = "end"
    definition["steps"] += [
        {
            "id": f"later{i}",
            "type": "prompt",
            "config": {"template": "{{ prepare.text }}"},
        }
        for i in range(extra)
    ]
    harness.workflow_documents.create(json.dumps(definition))
    async with harness.run_test(size=size) as pilot:
        await pilot.pause()
        await select_step(harness, pilot, "finish")
        screen = harness.screen
        screen._structural_action("move", offset=1)
        await pilot.pause()
        painted = painted_text(harness.screen)
        if size[0] == 110:
            assert "/steps/2/config/template" in painted and "prepare.text" in painted
            assert "/steps/0/on_success" in painted and "end" in painted
        if extra:
            detail = harness.screen.query_one("#workflow-dialog-detail")
            detail.focus()
            detail.scroll_end(animate=False)
            await pilot.pause()
            assert "/steps/15/config/template" in painted_text(harness.screen)
            assert "prepare.text" in painted_text(harness.screen)
            assert_hit(harness.screen, detail)
        owner = harness.workflow_drafts
        owner.update(
            harness.workflow_documents.edit_field(
                owner.current.raw_text, "/name", "Typed during confirmation"
            )
        )
        unchanged = owner.current
        await choose_option(harness, pilot, "confirm")
        assert owner.current == unchanged
        assert "changed" in screen._error.lower()


async def test_reopen_ui_discovers_invalid_stale_draft_repairs_copies_and_saves(
    tmp_path,
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        owner = harness.workflow_drafts
        documents = harness.workflow_documents
        base = owner.base
        owner.update(documents.edit_field(base.raw_json, "/name", "First revision"))
        await owner.save_revision()
        head = owner.base
        await owner.select(base.workflow_id, base.revision_id)
        valid = owner.update(
            documents.edit_field(base.raw_json, "/name", "Recovered edits")
        )
        invalid = owner.update('{"unfinished":')
        await owner.flush()
    reopened = WorkflowEditorHarness(tmp_path)
    async with reopened.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert reopened.workflow_drafts.current.base_revision_id == head.revision_id
        # Find the stale base through the real menu, not by injecting its ID.
        await more_action(reopened, pilot, "local-drafts")
        options = reopened.screen.query_one("#workflow-dialog-choices", OptionList)
        stale = next(
            options.get_option_at_index(i)
            for i in range(options.option_count)
            if "Repair raw JSON" in str(options.get_option_at_index(i).prompt)
        )
        await choose_option(reopened, pilot, stale.id)
        assert reopened.workflow_drafts.current == invalid
        assert reopened.workflow_drafts.current.last_valid_json == valid.last_valid_json
        await more_action(reopened, pilot, "recover")
        await choose_option(reopened, pilot, "repair")
        raw = reopened.screen.query_one("#workflow-raw-json", TextArea)
        assert raw.text == '{"unfinished":'
        assert_hit(reopened.screen, raw)
        assert reopened.focused is raw
        raw.load_text(valid.raw_text)
        await pilot.pause()
        await more_action(reopened, pilot, "recover")
        await choose_option(reopened, pilot, "copy")
        copied = reopened.workflow_drafts.current
        assert copied.base_revision_id == head.revision_id
        assert (
            reopened.workflow_documents.field_text(
                copied.raw_text, "/name", as_json=False
            )
            == "Recovered edits"
        )
        old = reopened.workflow_documents.get_draft(base.workflow_id, base.revision_id)
        assert old.raw_text == valid.raw_text
        assert old.last_valid_json == valid.last_valid_json
        assert_hit(
            reopened.screen, reopened.screen.query_one("#workflow-save-revision")
        )
        assert await pilot.click("#workflow-save-revision")
        await pilot.pause()
        await reopened.workers.wait_for_complete()
        assert reopened.workflow_drafts.base.parent_revision_ids == (head.revision_id,)
    again = WorkflowEditorHarness(tmp_path)
    async with again.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        assert (
            again.workflow_documents.field_text(
                again.workflow_drafts.current.raw_text, "/name", as_json=False
            )
            == "Recovered edits"
        )


async def test_recovery_confirmation_typing_and_dirty_head_offer_do_not_overwrite(
    tmp_path,
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        screen = harness.screen
        owner, documents = harness.workflow_drafts, harness.workflow_documents
        base = owner.base
        await owner.save_revision()
        head = owner.base
        await owner.select(base.workflow_id, base.revision_id)
        owner.update(documents.edit_field(base.raw_json, "/name", "Source"))
        pristine = documents.get_draft(head.workflow_id, head.revision_id)
        await more_action(harness, pilot, "recover")
        newer = owner.update(
            documents.edit_field(owner.current.raw_text, "/name", "During confirmation")
        )
        await choose_option(harness, pilot, "copy")
        assert owner.current == newer
        assert "confirmation" in painted_text(screen).lower()
        assert documents.get_draft(head.workflow_id, head.revision_id) == pristine
        dirty = documents.put_draft(
            head.workflow_id, head.revision_id, '{"unrelated":', pristine.generation + 1
        )
        await more_action(harness, pilot, "recover")
        await choose_option(harness, pilot, "open-head")
        assert owner.current == dirty
        assert documents.get_draft(base.workflow_id, base.revision_id) == newer
        assert documents.get_draft(head.workflow_id, head.revision_id) == dirty


@pytest.mark.parametrize("change", ["selection", "head", "dirty-after-confirmation"])
async def test_recovery_confirmation_rechecks_selection_head_and_target(
    tmp_path, change
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        owner, documents = harness.workflow_drafts, harness.workflow_documents
        base = owner.base
        await owner.save_revision()
        head = owner.base
        await owner.select(base.workflow_id, base.revision_id)
        source = owner.update(
            documents.edit_field(base.raw_json, "/name", "Keep source")
        )
        await owner.flush()
        await more_action(harness, pilot, "recover")
        if change == "selection":
            await owner.select(head.workflow_id, head.revision_id)
            await owner.select(base.workflow_id, base.revision_id)
        else:
            target = documents.put_draft(
                head.workflow_id,
                head.revision_id,
                head.raw_json if change == "head" else '{"another_draft":',
                1,
            )
            if change == "head":
                documents.save_revision(
                    head.workflow_id, head.revision_id, target.generation
                )
        old_target = documents.get_draft(head.workflow_id, head.revision_id)
        await choose_option(harness, pilot, "copy")
        assert owner.current == source
        assert documents.get_draft(base.workflow_id, base.revision_id) == source
        assert documents.get_draft(head.workflow_id, head.revision_id) == old_target
        assert harness.screen._error
        if change == "dirty-after-confirmation":
            await more_action(harness, pilot, "recover")
            await choose_option(harness, pilot, "open-head")
            assert owner.current == old_target


async def test_real_ui_typing_during_save_recovery_and_second_save(
    tmp_path, monkeypatch
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        screen = harness.screen
        editor = await select_step(harness, pilot)
        field = field_for(editor, "/steps/1/config/template")
        field.load_text("First saved")
        await pilot.pause()
        owner, documents = harness.workflow_drafts, harness.workflow_documents
        base = owner.base
        save = documents.save_revision
        started, release = Event(), Event()

        def blocked(*args):
            revision = save(*args)
            started.set()
            assert release.wait(5)
            return revision

        monkeypatch.setattr(documents, "save_revision", blocked)
        try:
            assert await pilot.click("#workflow-save-revision")
            assert await asyncio.to_thread(started.wait, 3)
            assert not field.disabled
            field.focus()
            assert_hit(screen, field)
            field.load_text("Typed during save")
            await pilot.pause()
            newer = owner.current
            assert (
                documents.field_text(
                    newer.raw_text, "/steps/1/config/template", as_json=False
                )
                == "Typed during save"
            )
        finally:
            release.set()
        await harness.workers.wait_for_complete()
        await pilot.pause()
        head = documents.list_workflows()[0]
        assert owner.current == newer
        assert "Recover draft" in painted_text(screen), painted_text(
            screen
        ).splitlines()[5]
        assert await pilot.click("#workflow-save-revision")
        await pilot.pause()
        await harness.workers.wait_for_complete()
        await choose_option(harness, pilot, "copy")
        assert documents.get_draft(base.workflow_id, base.revision_id) == newer
        assert await pilot.click("#workflow-save-revision")
        await pilot.pause()
        await harness.workers.wait_for_complete()
        assert owner.base.parent_revision_ids == (head.revision_id,)
        assert (
            documents.field_text(
                owner.base.raw_json, "/steps/1/config/template", as_json=False
            )
            == "Typed during save"
        )


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("settled_before_mount", [False, True])
async def test_cancelled_screen_caller_cannot_unlock_recovery_owner(
    tmp_path,
    monkeypatch,
    fail,
    settled_before_mount,
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        screen = harness.screen
        owner, documents = harness.workflow_drafts, harness.workflow_documents
        base = owner.base
        await owner.save_revision()
        head = owner.base
        await owner.select(base.workflow_id, base.revision_id)
        source = owner.update(documents.edit_field(base.raw_json, "/name", "Copy me"))
        await more_action(harness, pilot, "recover")
        copy = documents.copy_draft_to_head
        started, release = Event(), Event()

        def blocked(*args):
            started.set()
            assert release.wait(5)
            if fail:
                raise OSError("private recovery path")
            return copy(*args)

        monkeypatch.setattr(documents, "copy_draft_to_head", blocked)
        options = harness.screen.query_one("#workflow-dialog-choices", OptionList)
        options.highlighted = options.get_option_index("copy")
        options.focus()
        try:
            await pilot.press("enter")
            assert await asyncio.to_thread(started.wait, 3)
            harness.workers.cancel_group(screen, "workflows-controls")
            await pilot.pause()
            assert owner.editing_locked
            assert screen.query_one(WorkflowEditor).disabled
            assert screen.query_one("#workflow-save-revision").disabled
            snapshot = screen.save_state()
            await harness.pop_screen()
            assert owner.editing_locked
        finally:
            release.set()
        # A fresh canonical screen must join the retained owner before choosing
        # a base; it must not reload a cached old selection over a successful copy.
        if settled_before_mount:
            await owner.flush()
        replacement = WorkflowsScreen(harness)
        replacement.restore_state(snapshot)
        await harness.push_screen(replacement)
        await pilot.pause()
        await harness.workers.wait_for_complete()
        assert not owner.editing_locked
        assert not harness.screen.query_one(WorkflowEditor).disabled
        assert owner.current.base_revision_id == (
            base.revision_id if fail else head.revision_id
        )
        assert documents.get_draft(base.workflow_id, base.revision_id) == source
        assert "private recovery path" not in painted_text(harness.screen)
        assert harness.screen.query_one(WorkflowEditor).draft == owner.current
        if fail:
            assert "Recovery failed" in painted_text(harness.screen)


async def test_load_failure_is_visible_and_retry_keeps_the_document_owner(
    tmp_path, monkeypatch
):
    harness = WorkflowEditorHarness(tmp_path)
    read = harness.workflow_documents.list_workflows

    def fail():
        raise OSError("private storage path")

    monkeypatch.setattr(harness.workflow_documents, "list_workflows", fail)
    async with harness.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert "Definitions could not be loaded" in painted_text(harness.screen)
        assert "private storage path" not in painted_text(harness.screen)
        monkeypatch.setattr(harness.workflow_documents, "list_workflows", read)
        assert await pilot.click("#workflow-retry-save")
        await pilot.pause()
        await harness.workers.wait_for_complete()
        assert harness.screen.controller.documents is harness.workflow_documents
        assert harness.workflow_drafts.current is not None


@pytest.mark.parametrize("size", [(160, 48), (110, 36), (60, 20)])
async def test_real_css_panes_f6_tab_and_precise_nested_issue_are_painted(
    tmp_path, size
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=size) as pilot:
        await pilot.pause()
        screen = harness.screen
        editor = await select_step(harness, pilot, "summarize")
        expected = visible_pane_ids(size[0])
        assert (
            tuple(
                pane
                for pane in (
                    "workflows-library",
                    "workflows-navigator",
                    "workflows-editor",
                )
                if screen.query_one("#" + pane).display
            )
            == expected
        )
        screen.query_one("#workflow-save-revision").focus()
        for _ in range(len(expected) * 2):
            await pilot.press("f6")
            assert_hit(screen, harness.focused)
            assert any(
                harness.focused.id == pane
                or any(parent.id == pane for parent in harness.focused.ancestors)
                for pane in expected
            )
        for _ in range(12):
            await pilot.press("tab")
            assert_hit(screen, harness.focused)
            assert all(parent.display for parent in harness.focused.ancestors)
        editor.show_issue(Issue("/steps/2/timeout_seconds", "timeout", "Check timeout"))
        await pilot.pause()
        field = field_for(editor, "/steps/2/timeout_seconds")
        assert harness.focused is field
        assert not screen.query_one(
            "#workflow-section-execution", Collapsible
        ).collapsed
        assert_hit(screen, field)
        assert "300" in painted_text(screen)
        assert screen.query_one("#workflow-form").scroll_y > 0
        assert_hit(screen, screen.query_one("#workflow-save-revision"))


@pytest.mark.parametrize("activation", ["enter", "space", "pointer"])
async def test_neighbors_are_real_selectable_controls(tmp_path, activation):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        await select_step(harness, pilot)
        neighbor = harness.screen.query_one("#workflow-next", Button)
        neighbor.focus()
        neighbor.scroll_visible(animate=False)
        await pilot.pause()
        assert_hit(harness.screen, neighbor)
        assert harness.focused is neighbor
        if activation == "pointer":
            assert await pilot.click("#workflow-next")
        else:
            await pilot.press(activation)
        await pilot.pause()
        assert harness.screen.controller.section == "step:summarize", (
            harness.screen._error
        )


async def test_typing_keeps_widget_cursor_and_lossless_durable_value(tmp_path):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot)
        template = field_for(editor, "/steps/1/config/template")
        template.focus()
        template.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.press("s", "d", "r", "f", "6")
        await pilot.pause()
        assert harness.focused is template
        assert template.text.startswith("sdrf6")
        assert_hit(harness.screen, template)
        await harness.workflow_drafts.flush()
        draft = harness.workflow_drafts.current
        assert (
            harness.workflow_documents.field_text(
                draft.raw_text, "/steps/1/config/template", as_json=False
            )
            == template.text
        )
        name = field_for(editor, "/steps/1/name")
        name.focus()
        await pilot.press("d", "r", "s")
        await pilot.pause()
        assert name.value == "drs"
        assert "drs" in str(editor.query_one("#workflow-editor-heading").renderable)
        nav = harness.screen.query_one("#workflow-navigation-list", OptionList)
        assert "drs" in str(nav.get_option("step:prepare").prompt)
        assert (
            harness.workflow_documents.project(
                harness.workflow_drafts.current.raw_text
            )["steps"][1]["id"]
            == "prepare"
        )


@pytest.mark.parametrize("size", [(160, 48), (110, 36), (60, 20)])
async def test_field_summaries_refresh_without_resetting_view_state(tmp_path, size):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=size) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot, "summarize")
        nav = harness.screen.query_one("#workflow-navigation-list", OptionList)
        nav.highlighted = 0  # Keyboard exploration is not the selected step.
        option = nav.get_option("step:summarize")
        inputs = editor.query_one("#workflow-section-inputs", Collapsible)
        inputs.collapsed = False
        provider = field_for(editor, "/steps/2/config/provider")
        provider.focus()
        provider.scroll_visible(animate=False)
        await pilot.pause()
        provider.value = ""
        await pilot.pause()
        assert "missing required value" in inputs.title
        assert str(nav.get_option("step:summarize").prompt).endswith(" !")
        await pilot.wait_for_scheduled_animations()
        scroll = editor.query_one("#workflow-form").scroll_offset
        generation = harness.workflow_drafts.current.generation
        await pilot.press("l")
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert "missing required value" not in inputs.title
        assert not str(nav.get_option("step:summarize").prompt).endswith(" !")
        assert field_for(editor, "/steps/2/config/provider") is provider
        assert harness.focused is provider and provider.cursor_position == 1
        assert not inputs.collapsed
        assert editor.query_one("#workflow-form").scroll_offset == scroll
        assert nav.highlighted == 0 and nav.get_option("step:summarize") is option
        assert harness.workflow_drafts.current.generation == generation + 1

        action = editor.query_one("#workflow-section-action", Collapsible)
        prompt = field_for(editor, "/steps/2/config/prompt")
        prompt.load_text("")
        await pilot.pause()
        assert "missing required value" in action.title
        prompt.load_text("Summarize this")
        await pilot.pause()
        assert "missing required value" not in action.title
        execution = editor.query_one("#workflow-section-execution", Collapsible)
        field_for(editor, "/steps/2/retry").value = "2"
        await pilot.pause()
        assert "retry 2" in execution.title and execution.collapsed


@pytest.mark.parametrize("transition", ["workflow", "inspect", "return", "create"])
async def test_operation_error_clears_after_successful_context_change(
    tmp_path, transition
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        screen = harness.screen
        base = harness.workflow_drafts.base
        if transition == "return":
            await screen._inspect(base.revision_id)

        async def malformed_import():
            harness.workflow_documents.create('{"name":\n')

        screen._start(malformed_import())
        await harness.workers.wait_for_complete()
        await pilot.pause()
        assert "Invalid JSON" in painted_text(screen)
        retry_was_visible = screen.query_one("#workflow-retry-save").display
        if transition == "workflow":
            other = harness.workflow_documents.create(
                '{"name":"Other workflow","version":1,"steps":[]}'
            )
            await screen._select_workflow(other.workflow_id, other.revision_id)
        elif transition == "inspect":
            await screen._inspect(base.revision_id)
        elif transition == "return":
            screen._more_selected("return")
            await harness.workers.wait_for_complete()
        else:
            await screen._create("Fresh context")
        await pilot.pause()
        assert "Invalid JSON" not in painted_text(screen)
        assert not screen.query_one("#workflow-retry-save").display
        status = str(screen.query_one("#workflow-draft-status").renderable)
        assert ("read-only" if transition == "inspect" else "Saved revision") in status
        assert not retry_was_visible


async def test_deleted_raw_step_returns_to_overview_and_can_edit_name(tmp_path):
    import json

    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot, "save")
        owner = harness.workflow_drafts
        document = harness.workflow_documents.project(owner.current.raw_text)
        document["steps"].pop()
        editor.query_one("#workflow-raw-json", TextArea).load_text(json.dumps(document))
        await pilot.pause()
        name = field_for(editor, "/name")
        name.value = "Name after deleting selected step"
        await pilot.pause()
        assert not harness.screen._error
        assert harness.screen.controller.section == editor.section == "overview"
        assert field_for(editor, "/name") is name
        nav = harness.screen.query_one("#workflow-navigation-list", OptionList)
        assert str(nav.get_option("overview").prompt).startswith("> ")
        await owner.flush()
        assert (
            harness.workflow_documents.project(owner.current.raw_text)["name"]
            == name.value
        )


async def test_validate_preserves_unrelated_operation_error(tmp_path):
    import json

    from Tests.Workflows.helpers import prompt_definition

    harness = WorkflowEditorHarness(tmp_path, seed=False)
    harness.workflow_documents.create(json.dumps(prompt_definition()))
    async with harness.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        screen = harness.screen

        async def malformed_import():
            harness.workflow_documents.create('{"name":\n')

        screen._start(malformed_import())
        await harness.workers.wait_for_complete()
        await pilot.pause()
        assert "Invalid JSON" in painted_text(screen)
        assert await pilot.click("#workflow-validate")
        await pilot.pause()
        await harness.workers.wait_for_complete()
        assert "Invalid JSON" in painted_text(screen)
        assert "Structure valid" in painted_text(screen)
        assert not screen.query_one("#workflow-retry-save").display


async def test_failed_flush_vetoes_real_screen_navigation_without_losing_text(
    tmp_path, monkeypatch
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot)
        field = field_for(editor, "/steps/1/config/template")
        field.load_text("pending authored text")
        await pilot.pause()
        write = harness.workflow_documents.put_draft

        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(harness.workflow_documents, "put_draft", fail)
        try:
            editor.show_step("summarize")
            await pilot.pause()
            assert harness.screen.controller.section == "step:prepare"
            assert not await harness.screen.flush_pending_work()
            assert (
                harness.workflow_drafts.current.raw_text.find("pending authored text")
                >= 0
            )
            assert "Not saved locally" in painted_text(harness.screen)
            assert_hit(harness.screen, harness.screen.query_one("#workflow-retry-save"))
        finally:
            monkeypatch.setattr(harness.workflow_documents, "put_draft", write)
        assert await pilot.click("#workflow-retry-save")
        await pilot.pause()
        assert await harness.screen.flush_pending_work()


async def test_historical_inspection_keeps_dirty_current_draft_and_edit_guard(tmp_path):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        base = harness.workflow_drafts.base
        harness.workflow_drafts.update(
            harness.workflow_documents.edit_field(
                base.raw_json, "/name", "Second version"
            )
        )
        second = await harness.workflow_drafts.save_revision()
        dirty = harness.workflow_drafts.update(
            harness.workflow_documents.edit_field(
                second.raw_json, "/name", "Unrelated dirty draft"
            )
        )
        await harness.screen._inspect(base.revision_id)
        await pilot.pause()
        assert harness.screen.query_one("#workflow-raw-json", TextArea).read_only
        assert "read-only" in painted_text(harness.screen)
        assert harness.workflow_drafts.current.raw_text == dirty.raw_text
        with pytest.raises(DraftConflict, match="Open it without overwriting"):
            await harness.screen.controller.edit_inspected_revision()
        assert harness.workflow_drafts.current.raw_text == dirty.raw_text
        harness.screen._more_selected("return")
        await pilot.pause()
        assert (
            harness.screen.query_one("#workflow-raw-json", TextArea).text
            == dirty.raw_text
        )
        assert harness.screen.save_state()["selection"] == (
            second.workflow_id,
            second.revision_id,
        )


async def test_overlapping_overview_refresh_keeps_one_complete_form(tmp_path):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        screen = harness.screen
        await asyncio.gather(screen._refresh_authoring(), screen._refresh_authoring())
        await pilot.pause()
        editor = screen.query_one(WorkflowEditor)
        steps = harness.workflow_documents.project(
            harness.workflow_drafts.current.raw_text
        )["steps"]
        buttons = list(editor.query(".workflow-linear-step"))
        assert len(buttons) == len(steps)
        assert len({button.id for button in buttons}) == len(steps)
        assert field_for(editor, "/name").value == "Local file to reviewed note"


async def test_resize_focus_and_selector_cancel_are_visible_and_keep_search_cursor(
    tmp_path,
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        search = harness.screen.query_one("#workflow-library-search", Input)
        search.focus()
        await pilot.press("f", "i", "l", "e", "left")
        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        assert harness.focused.id == "workflow-library-selector"
        assert_hit(harness.screen, harness.focused)
        assert await pilot.click("#workflow-library-selector")
        await pilot.pause()
        assert isinstance(harness.screen, ChoiceModal)
        await pilot.press("escape")
        assert harness.focused.id == "workflow-library-selector"
        await pilot.resize_terminal(160, 48)
        await pilot.pause()
        assert (
            harness.screen.query_one("#workflow-library-search", Input).value == "file"
        )
        assert search.cursor_position == 3
        assert harness.focused is not search


async def test_sections_scroll_focus_survive_step_navigation_without_documents_in_snapshot(
    tmp_path,
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot)
        editor.show_issue(Issue("/steps/1/timeout_seconds", "timeout", "Timeout"))
        await pilot.pause()
        scroll = editor.query_one("#workflow-form").scroll_y
        await select_step(harness, pilot, "review")
        await select_step(harness, pilot, "prepare")
        assert not editor.query_one(
            "#workflow-section-execution", Collapsible
        ).collapsed
        assert harness.focused is field_for(editor, "/steps/1/timeout_seconds")
        assert_hit(harness.screen, harness.focused)
        assert editor.query_one("#workflow-form").scroll_y == scroll
        state = harness.screen.save_state()
        assert "Summarize in three bullets" not in repr(state)
        assert "raw_text" not in repr(state)


async def test_reference_picker_filters_earlier_types_and_restores_exact_opener(
    tmp_path,
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot, "summarize")
        choices = reference_choices(editor.document, "summarize", "string", "steps")
        expressions = {value for _, value in choices}
        assert "{{ prepare.text }}" in expressions
        assert "{{ review.text }}" not in expressions
        assert "{{ ingest.media_ids }}" not in expressions
        prompt = field_for(editor, "/steps/2/config/prompt")
        prompt.focus()
        prompt.scroll_visible(animate=False)
        await pilot.pause()
        saved = prompt.selection
        screen = harness.screen
        screen.post_message(
            WorkflowEditor.ReferenceRequested(
                "/steps/2/config/prompt", "string", prompt
            )
        )
        await pilot.pause()
        assert isinstance(harness.screen, ReferencePicker)
        await pilot.press("escape")
        await pilot.pause()
        assert harness.focused is prompt
        assert prompt.selection == saved
        assert_hit(screen, prompt)


async def test_new_add_step_and_confirmed_discard_are_reachable_at_minimum(tmp_path):
    harness = WorkflowEditorHarness(tmp_path, seed=False)
    async with harness.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        screen = harness.screen
        assert await pilot.click("#workflow-more")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("M", "y", "space", "f", "l", "o", "w", "enter")
        await pilot.pause()
        await harness.workers.wait_for_complete()
        await pilot.pause()
        assert harness.workflow_drafts.current is not None, screen._error
        screen._step_chooser()
        await pilot.pause()
        assert isinstance(harness.screen, StepChooser)
        search = harness.screen.query_one(Input)
        search.value = "Render text"
        await pilot.pause()
        harness.screen.query_one(OptionList).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert_hit(harness.screen, harness.screen.query_one("#workflow-insert-confirm"))
        assert not harness.screen.query_one("#workflow-insert-confirm").disabled
        assert await pilot.click("#workflow-insert-confirm")
        await pilot.pause()
        await harness.workers.wait_for_complete()
        await pilot.pause()
        assert (
            len(
                harness.workflow_documents.project(
                    harness.workflow_drafts.current.raw_text
                )["steps"]
            )
            == 1
        ), (screen._error, type(harness.screen), screen._busy)
        screen._more_selected("discard")
        await pilot.pause()
        await pilot.press("escape")
        assert (
            len(
                harness.workflow_documents.project(
                    harness.workflow_drafts.current.raw_text
                )["steps"]
            )
            == 1
        )
        screen._more_selected("discard")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert (
            harness.workflow_documents.project(
                harness.workflow_drafts.current.raw_text
            )["steps"]
            == []
        ), screen._error


@pytest.mark.parametrize(
    "width, expected",
    [
        (160, ("workflows-library", "workflows-navigator", "workflows-editor")),
        (110, ("workflows-navigator", "workflows-editor")),
        (60, ("workflows-editor",)),
    ],
)
def test_only_visible_panes_participate_in_focus_cycle(width, expected):
    assert visible_pane_ids(width) == expected


async def test_invalid_json_remains_after_navigation_and_reopen(tmp_path):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        editor = harness.screen.query_one("#workflow-raw-json", TextArea)
        editor.load_text('{"steps": [')
        await pilot.pause()
        await harness.workflow_drafts.flush()
        assert harness.screen.query_one("#workflow-run", Button).disabled
        assert harness.workflow_drafts.update('{"steps": [').error is not None
    reopened = WorkflowEditorHarness(tmp_path)
    async with reopened.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert reopened.screen._loaded, reopened.screen._error
        assert reopened.screen._error == ""
        assert (
            reopened.screen.query_one("#workflow-raw-json", TextArea).text
            == '{"steps": ['
        )


async def test_expanded_editor_escape_restores_the_original_field(
    tmp_path,
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot)
        field = field_for(editor, "/steps/1/config/template")
        button = editor.query_one("#" + field.id + "-expand")
        original_height = field.styles.height
        button.focus()
        button.scroll_visible(animate=False)
        await pilot.pause()
        button.press()
        await pilot.pause()
        assert field.styles.height.value == 18
        await pilot.press("escape")
        assert field.styles.height == original_height
        assert harness.focused is field


async def test_console_refresh_and_stale_raw_edits_cannot_replace_editor(tmp_path):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        screen = harness.screen
        draft = harness.workflow_drafts.current
        from tldw_chatbook.UI.Workflows_Modules.console_context import (
            WorkflowConsoleContext,
        )

        editor = screen.query_one(WorkflowEditor)
        context = screen.query_one(WorkflowConsoleContext)
        context.apply_context(None)
        await pilot.pause()
        assert screen.query_one(WorkflowEditor) is editor
        assert screen.controller.section == "overview"
        assert harness.workflow_drafts.current == draft
        screen.post_message(WorkflowEditor.RawEdited("{", ("wrong", "wrong")))
        await pilot.pause()
        assert harness.workflow_drafts.current.raw_text == draft.raw_text


async def test_validate_uses_authoring_issues_and_cannot_enable_execution(tmp_path):
    import json
    import sys

    from Tests.Workflows.helpers import prompt_definition

    harness = WorkflowEditorHarness(tmp_path, seed=False)
    harness.workflow_documents.create(json.dumps(prompt_definition()))
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        await harness.screen._activate_issue()
        await pilot.pause()
        assert "Structure valid" in painted_text(harness.screen)
        assert not harness.screen.query_one("#workflow-retry-save").display
        assert "Saved revision" in str(
            harness.screen.query_one("#workflow-draft-status").renderable
        )
        run = harness.screen.query_one("#workflow-run", Button)
        assert run.disabled
        run.press()
        await pilot.pause()
        assert harness.screen.controller.section == "overview"
        assert not any(
            name in sys.modules
            for name in (
                "tldw_chatbook.Workflows.runtime",
                "tldw_chatbook.Workflows.runtime_lock",
                "tldw_chatbook.Workflows.run_service",
            )
        )
        editor = await select_step(harness, pilot)
        assert "Structure valid" not in painted_text(harness.screen)
        await harness.screen._activate_issue()
        field_for(editor, "/steps/0/config/template").load_text("")
        await pilot.pause()
        assert "Structure valid" not in painted_text(harness.screen)
        assert "authoring issue" in painted_text(harness.screen)
    assert not list(tmp_path.glob("*.lock"))


async def test_continuous_form_sections_expand_independently_and_restore(tmp_path):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot, "prepare")
        original = harness.workflow_drafts.current
        for name in ("outputs", "execution"):
            section = editor.query_one("#workflow-section-" + name, Collapsible)
            assert section.collapsed
            title = section.query_one("CollapsibleTitle")
            title.scroll_visible(animate=False)
            title.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert not section.collapsed
        assert not editor.query_one("#workflow-section-outputs", Collapsible).collapsed
        assert not editor.query_one("#workflow-section-action", Collapsible).collapsed
        await select_step(harness, pilot, "summarize")
        await select_step(harness, pilot, "prepare")
        for name in ("outputs", "execution"):
            assert not editor.query_one(
                "#workflow-section-" + name, Collapsible
            ).collapsed
        assert harness.workflow_drafts.current == original


@pytest.mark.parametrize("size", [(160, 48), (110, 36), (60, 20)])
async def test_pinned_status_distinguishes_stored_draft_from_saved_revision(
    tmp_path, size
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=size) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot, "summarize")
        status = harness.screen.query_one("#workflow-draft-status")
        assert "Saved revision · draft unchanged" in str(status.renderable)
        field = field_for(editor, "/steps/2/config/prompt")
        old_head = harness.screen.controller.head
        field.load_text("A locally stored draft, not exported yet")
        await pilot.pause()
        await harness.workflow_drafts.flush()
        await pilot.pause()
        assert "Draft stored · not a saved revision" in str(status.renderable)
        assert "Draft stored" in painted_text(harness.screen)
        assert harness.screen.controller.head == old_head
        assert_hit(harness.screen, status)
        assert await pilot.click("#workflow-save-revision")
        await pilot.pause()
        await harness.workers.wait_for_complete()
        await pilot.pause()
        assert "Saved revision · draft unchanged" in str(status.renderable)
        assert harness.screen.controller.head != old_head


async def test_short_viewport_keeps_prompt_label_with_focused_control(tmp_path):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot, "summarize")
        field = field_for(editor, "/steps/2/config/prompt")
        label = field.parent.query_one(".form-label")
        assert harness.focused is field
        assert "Prompt" in painted_text(harness.screen)
        assert "{{ prepare.text }}" in painted_text(harness.screen)
        assert_hit(harness.screen, label)
        assert_hit(harness.screen, field)
        editor.query_one("#" + field.id + "-reference").focus()
        await pilot.pause()
        field.focus()
        await pilot.pause()
        assert "Prompt" in painted_text(harness.screen)
        assert "{{ prepare.text }}" in painted_text(harness.screen)
        assert_hit(harness.screen, label)


async def test_short_viewport_real_app_paints_prompt_value_and_focus(
    tmp_path, monkeypatch
):
    from unittest.mock import AsyncMock

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_screen_navigation import _wait_for_initial_screen
    from tldw_chatbook.config import save_setting_to_cli_config
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    path = tmp_path / "real-app.sqlite3"
    db = WorkflowsDB(path)
    try:
        fixture = Path(__file__).parents[1] / "fixtures/workflows/file_to_note.json"
        DocumentService(db).create(fixture.read_text(encoding="utf-8"))
    finally:
        db.close()
    save_setting_to_cli_config("splash_screen", "enabled", False)
    monkeypatch.setattr("tldw_chatbook.config.get_workflows_db_path", lambda: path)
    app = _build_test_app(
        "home", config_overrides={"splash_screen": {"enabled": False}}
    )
    monkeypatch.setattr(app, "_refresh_model_catalogs", AsyncMock())
    async with app.run_test(size=(60, 20)) as pilot:
        await _wait_for_initial_screen(pilot)
        app.post_message(NavigateToScreen("workflows"))
        await pilot.pause()
        await asyncio.gather(
            *(w.wait() for w in app.workers if w.group.startswith("workflows-"))
        )
        await pilot.pause()
        assert isinstance(app.screen, WorkflowsScreen)
        editor = app.screen.query_one(WorkflowEditor)
        editor.show_step("summarize")
        await pilot.pause()
        await asyncio.gather(
            *(w.wait() for w in app.workers if w.group.startswith("workflows-"))
        )
        await pilot.pause()
        field = field_for(editor, "/steps/2/config/prompt")
        label = field.parent.query_one(".form-label")
        assert field.text == "{{ prepare.text }}"
        assert app.focused is field
        frame = painted_text(app.screen)
        assert "Prompt" in frame
        assert field.text in frame, (
            f"region={field.region}, content={field.content_region}, "
            f"padding={field.styles.padding}, border={field.styles.border}, "
            f"scroll={field.scroll_offset}\n{frame}"
        )
        assert_hit(app.screen, label)
        assert_hit(app.screen, field)
        assert field.content_region.height >= 1


async def test_reference_selection_inserts_at_consumer_selection_and_stays_visible(
    tmp_path,
):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        editor = await select_step(harness, pilot, "summarize")
        field = field_for(editor, "/steps/2/config/prompt")
        field.load_text("Before: ")
        field.focus()
        field.move_cursor((0, 8))
        await pilot.pause()
        harness.screen.post_message(
            WorkflowEditor.ReferenceRequested("/steps/2/config/prompt", "string", field)
        )
        await pilot.pause()
        harness.screen.query_one("#workflow-reference-step", Button).press()
        await pilot.pause()
        listing = harness.screen.query_one(OptionList)
        listing.highlighted = next(
            i
            for i in range(listing.option_count)
            if listing.get_option_at_index(i).id == "{{ prepare.text }}"
        )
        listing.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert field.text == "Before: {{ prepare.text }}"
        assert harness.focused is field
        assert_hit(harness.screen, field)
        assert (
            harness.workflow_documents.field_text(
                harness.workflow_drafts.current.raw_text,
                "/steps/2/config/prompt",
                as_json=False,
            )
            == field.text
        )
