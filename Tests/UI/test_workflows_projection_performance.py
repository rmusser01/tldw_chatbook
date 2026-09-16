"""Prepared display reads must retain semantics without reparsing each field."""

import json
from dataclasses import replace
from typing import ClassVar

import pytest
from textual.widgets import Button, Collapsible, Input, Static, TextArea

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.Workflows.test_document_complexity import IDENTITY, definition, nested_raw
from tldw_chatbook.UI.Workflows_Modules.controller import WorkflowsController
from tldw_chatbook.UI.Workflows_Modules.editor import WorkflowEditor
from tldw_chatbook.Workflows import document_service
from tldw_chatbook.Workflows.catalog import FieldSpec
from tldw_chatbook.Workflows.document_service import DocumentService, OpaqueNumber
from tldw_chatbook.Workflows.draft_session import DraftSession
from tldw_chatbook.Workflows.models import Draft


def draft_for(raw):
    return Draft(IDENTITY["workflow_id"], IDENTITY["revision_id"], 1, raw, raw, None)


def count_decodes(monkeypatch):
    calls = []
    original = document_service._decode

    def counted(raw):
        calls.append(len(raw))
        return original(raw)

    monkeypatch.setattr(document_service, "_decode", counted)
    return calls


@pytest.mark.parametrize("count", [100, 500])
def test_required_field_validation_uses_constant_document_parses(monkeypatch, count):
    documents = DocumentService(None)
    owner = DraftSession(documents)
    owner.current = draft_for(json.dumps(definition(count)))
    controller = WorkflowsController(documents, owner)
    calls = count_decodes(monkeypatch)
    assert controller.validate() == ()
    assert len(calls) <= 2, "Required fields must read the existing projection"


@pytest.mark.parametrize(
    "value,missing",
    [
        (None, True),
        ("", True),
        (" ", False),
        ("null", False),
        (0, False),
        (False, False),
        ([], False),
        ({}, False),
    ],
)
def test_required_field_projection_preserves_missing_semantics(value, missing):
    document = definition()
    document["steps"][0]["config"]["template"] = value
    documents = DocumentService(None)
    owner = DraftSession(documents)
    owner.current = draft_for(json.dumps(document))
    issues = WorkflowsController(documents, owner).validate()
    assert [(issue.pointer, issue.code) for issue in issues] == (
        [("/steps/0/config/template", "required")] if missing else []
    )


def test_absent_and_opaque_required_fields_keep_distinct_meanings():
    document = definition(2)
    del document["steps"][0]["config"]["template"]
    document["steps"][1]["config"]["template"] = "OPAQUE"
    raw = json.dumps(document).replace('"OPAQUE"', "1e9999")
    documents = DocumentService(None)
    owner = DraftSession(documents)
    owner.current = draft_for(raw)
    issues = WorkflowsController(documents, owner).validate()
    assert [(issue.pointer, issue.code) for issue in issues] == [
        ("/steps/0/config/template", "required")
    ]


def test_validation_reports_legacy_over_limit_content_without_projection_failure():
    documents = DocumentService(None)
    owner = DraftSession(documents)
    owner.current = draft_for(nested_raw(1000))
    issues = WorkflowsController(documents, owner).validate()
    assert len(issues) == 1
    assert issues[0].code == "invalid_json"
    assert "64" in issues[0].message


def test_editor_field_values_and_summaries_use_prepared_projection(monkeypatch):
    raw = json.dumps(definition())
    documents = DocumentService(None)
    editor = WorkflowEditor(documents)
    editor.draft = draft_for(raw)
    editor.document = documents.project(raw)
    calls = count_decodes(monkeypatch)
    assert (
        editor._field_value(
            "/steps/0/config/template", FieldSpec("config/template", "Template")
        )
        == "Text"
    )
    assert editor._step_summary(0, "action") == ""
    assert editor._step_summary(0, "execution") == " · retry unset / timeout unsets"
    assert calls == [], "Display reads must not parse the complete raw document"


def test_projected_field_helpers_preserve_opaque_tokens_and_shape_refusals():
    document = definition()
    document["inputs"] = {"number": "OPAQUE"}
    document["steps"][0]["retry"] = "NEGATIVE_ZERO"
    document["steps"][0]["timeout_seconds"] = True
    raw = (
        json.dumps(document)
        .replace('"OPAQUE"', "1.234567890123456789e9999")
        .replace('"NEGATIVE_ZERO"', "-0")
    )
    projected = DocumentService.project(raw)
    assert projected["inputs"]["number"] == OpaqueNumber("1.234567890123456789e9999")
    assert (
        DocumentService.projected_field_text(projected, "/inputs")
        == '{"number":1.234567890123456789e9999}'
    )
    assert DocumentService.projected_field_text(projected, "/steps/0/retry") == "-0"
    assert not DocumentService.projected_field_editable(
        projected, "/steps/0/retry", "integer"
    )
    assert not DocumentService.projected_field_editable(
        projected, "/steps/0/timeout_seconds", "integer"
    )
    assert DocumentService.projected_field_editable(
        projected, "/steps/0/config/template", "string"
    )
    projected["steps"][0]["config"]["unknown"] = "preserve"
    assert not DocumentService.projected_field_editable(
        projected, "/steps/0/config/template", "string"
    )


class EditorHarness(ConsolidatedCSSApp):
    CSS_PATH: ClassVar = [
        BUNDLED_STYLESHEET,
        BUNDLED_STYLESHEET.parent / "screen_feature_workflows.tcss",
    ]

    def compose(self):
        yield WorkflowEditor(DocumentService(None), id="workflows-editor")


@pytest.mark.parametrize("count", [0, 3, 20])
async def test_overview_mounts_ordered_steps_in_one_batch(monkeypatch, count):
    document = definition(count)
    document["steps"].reverse()
    async with EditorHarness().run_test(size=(110, 36)) as pilot:
        editor = pilot.app.query_one(WorkflowEditor)
        target = editor.query_one("#workflow-general")
        mount = target.mount
        batches = []

        def forwarded_mount(*widgets, **kwargs):
            steps = [
                widget for widget in widgets if widget.has_class("workflow-linear-step")
            ]
            if steps:
                batches.append(len(steps))
            return mount(*widgets, **kwargs)

        monkeypatch.setattr(target, "mount", forwarded_mount)
        await editor.render_draft(draft_for(json.dumps(document)), section="overview")
        await pilot.pause()
        buttons = list(target.query(".workflow-linear-step").results(Button))
        assert [button.id for button in buttons] == [
            f"workflow-overview-{index}" for index in range(count)
        ]
        assert [button.label.plain.split("\n")[1] for button in buttons] == [
            f"step_{index} · select to edit" for index in reversed(range(count))
        ]
        assert all(button.is_mounted for button in buttons)
        assert batches == ([count] if count else []), "Do not await one mount per step"
        if not count:
            assert target.query_one("#workflow-add-first", Button).is_mounted


@pytest.mark.parametrize("section", ["overview", "step:step_1"])
@pytest.mark.parametrize("rebuild", [True, False])
async def test_raw_value_refresh_keeps_controls_and_updates_step_labels(
    section, rebuild
):
    document = definition(3)
    async with EditorHarness().run_test(size=(110, 36)) as pilot:
        editor = pilot.app.query_one(WorkflowEditor)
        await editor.render_draft(draft_for(json.dumps(document)), section=section)
        buttons = list(
            editor.query(".workflow-linear-step, .workflow-neighbors Button")
        )
        field = editor.query_one("#workflow-field-0", Input)
        raw = editor.query_one("#workflow-raw-json", TextArea)
        editor.query_one("#workflow-section-advanced", Collapsible).collapsed = False
        await pilot.pause()
        raw.focus()
        await pilot.pause()
        raw.move_cursor((0, 5))
        selection = raw.selection
        document["name"] = "Updated workflow"
        for index, step in enumerate(document["steps"]):
            step["name"] = f"Updated {index}"
        changed = draft_for(json.dumps(document))
        await editor.render_draft(changed, rebuild=rebuild)
        assert (
            list(editor.query(".workflow-linear-step, .workflow-neighbors Button"))
            == buttons
        )
        assert editor.query_one("#workflow-field-0") is field
        assert field.value == (
            "Updated workflow" if section == "overview" else "Updated 1"
        )
        assert [button.label.plain for button in buttons] == (
            [
                "01  Updated 0 (prompt)\nstep_0 · select to edit",
                "02  Updated 1 (prompt)\nstep_1 · select to edit",
                "03  Updated 2 (prompt)\nstep_2 · select to edit",
            ]
            if section == "overview"
            else ["Previous: Updated 0", "Next: Updated 2"]
        )
        await editor.render_draft(replace(changed, raw_text=changed.raw_text + " "))
        assert editor.query_one("#workflow-field-0") is field
        invalid = replace(changed, raw_text='{"incomplete":', error="Invalid JSON")
        await editor.render_draft(invalid)
        assert editor.query_one("#workflow-field-0") is field and field.disabled
        assert raw.text == '{"incomplete":'
        await editor.render_draft(changed)
        assert editor.query_one("#workflow-field-0") is field and not field.disabled
        assert pilot.app.focused is raw and raw.selection == selection


@pytest.mark.parametrize(
    "change",
    ["order", "type", "shape", "workflow", "revision", "readonly", "focus", "section"],
)
async def test_raw_reuse_rebuilds_on_structure_shape_or_context_change(change):
    document = definition(3)
    async with EditorHarness().run_test(size=(110, 36)) as pilot:
        editor = pilot.app.query_one(WorkflowEditor)
        await editor.render_draft(
            draft_for(json.dumps(document)), section="step:step_1"
        )
        field = editor.query_one("#workflow-field-0")
        options = {}
        if change == "order":
            document["steps"].reverse()
        elif change == "type":
            document["steps"][1].update(type="notes", config={"title": "Notes"})
        elif change == "shape":
            document["steps"][1]["config"]["template"] = 123
        elif change in ("workflow", "revision"):
            document["metadata"]["tldw_workflow"][change + "_id"] = (
                "33333333-3333-4333-8333-333333333333"
            )
        elif change == "readonly":
            options["read_only"] = True
        elif change == "focus":
            options["focus"] = True
        elif change == "section":
            options["section"] = "overview"
        identity = document["metadata"]["tldw_workflow"]
        changed = replace(
            draft_for(json.dumps(document)),
            workflow_id=identity["workflow_id"],
            base_revision_id=identity["revision_id"],
        )
        await editor.render_draft(changed, **options)
        assert editor.query_one("#workflow-field-0") is not field
        if change == "type":
            assert "/steps/1/config/title" in {
                p for p, _ in editor.field_bindings.values()
            }
            assert "/steps/1/config/template" not in {
                p for p, _ in editor.field_bindings.values()
            }
        if change == "shape":
            template_id = next(
                identifier
                for identifier, (pointer, _) in editor.field_bindings.items()
                if pointer == "/steps/1/config/template"
            )
            template = editor.query_one("#" + template_id)
            assert template.disabled
            assert "Unrepresented field shape" in " ".join(
                str(item.render()) for item in template.parent.query(Static)
            )


async def test_editor_render_reuses_projection_and_keeps_raw_control(monkeypatch):
    raw = json.dumps(definition())
    draft = draft_for(raw)
    async with EditorHarness().run_test(size=(110, 36)) as pilot:
        editor = pilot.app.query_one(WorkflowEditor)
        calls = count_decodes(monkeypatch)
        await editor.render_draft(draft, section="step:step_0")
        await pilot.pause()
        assert len(calls) == 1
        calls.clear()
        raw_control = editor.query_one("#workflow-raw-json", TextArea)
        editor.query_one("#workflow-section-advanced", Collapsible).collapsed = False
        await pilot.pause()
        raw_control.focus()
        await pilot.pause()
        assert pilot.app.focused is raw_control
        raw_control.move_cursor((0, 5))
        selection = raw_control.selection
        await editor.render_draft(draft, rebuild=False)
        assert calls == []
        assert editor.query_one("#workflow-raw-json", TextArea) is raw_control
        assert raw_control.selection == selection
        assert pilot.app.focused is raw_control
        invalid = replace(draft, generation=2, raw_text="{", error="Invalid JSON")
        await editor.render_draft(invalid, rebuild=False)
        assert calls == []
        assert raw_control.text == "{"
        assert editor.document["steps"][0]["config"]["template"] == "Text"
        changed_raw = raw.replace('"Text"', '"Changed"')
        await editor.render_draft(draft_for(changed_raw), rebuild=False)
        assert len(calls) == 1
        assert editor.document["steps"][0]["config"]["template"] == "Changed"


async def test_legacy_raw_inspection_skips_projection_and_cannot_emit_edits(
    monkeypatch,
):
    async with EditorHarness().run_test(size=(110, 36)) as pilot:
        editor = pilot.app.query_one(WorkflowEditor)
        valid = draft_for(json.dumps(definition()))
        await editor.render_draft(valid, section="step:step_0")
        raw_control = editor.query_one("#workflow-raw-json", TextArea)
        calls = count_decodes(monkeypatch)
        raw = nested_raw(1000)
        await editor.render_draft(
            draft_for(raw), raw_only_reason="Workflow JSON exceeds 64 container levels"
        )
        assert calls == []
        assert raw_control.text == raw
        assert raw_control.read_only
        assert editor.read_only
        assert editor.field_bindings == {}
        assert not editor.query_one("#workflow-section-advanced", Collapsible).collapsed
        assert "64 container levels" in " ".join(
            str(item.render()) for item in editor.query(Static)
        )
        # Returning from raw inspection must not reuse its empty display tree.
        await editor.render_draft(valid, section="step:step_0")
        assert len(calls) == 1
        assert not raw_control.read_only
        assert editor.document["steps"][0]["id"] == "step_0"
