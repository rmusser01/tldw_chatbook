"""Mounted tests for the Personas workbench policy-rules editor (Task 11).

Covers: editor CRUD roundtrip + validation + the deny-by-default warning
(controller ruling from Task 7's review — non-negotiable), the inspector's
read-only policy summary, the workspace-switcher persona label suffix, and
the actor-pack import review policy-rule count + notice copy.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from textual.widgets import Button, Checkbox, Input, ListItem, ListView, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.Widgets.Console.console_workspace_switcher_modal import (
    workspace_persona_label_suffix,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import (
    PersonasInspectorPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_persona_visual_pack_widget import (
    PersonasPersonaVisualPackWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_policy_rules_editor import (
    PersonasPolicyRulesEditor,
    PersonaPolicyRulesChanged,
)

pytestmark = pytest.mark.asyncio


class _CaptureApp(ConsolidatedCSSApp):
    """Harness capturing PersonaPolicyRulesChanged messages at the app."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[PersonaPolicyRulesChanged] = []

    def on_persona_policy_rules_changed(self, message: PersonaPolicyRulesChanged) -> None:
        self.captured.append(message)


class EditorApp(_CaptureApp):
    def compose(self):
        yield PersonasPolicyRulesEditor(id="personas-policy-rules-editor")


async def _click(pilot, button_id: str) -> None:
    # press() rather than pilot.click(): the editor surface can scroll the
    # buttons offscreen in the narrow test viewport, and click() at stale
    # coordinates silently misses.
    pilot.app.query_one(button_id, Button).press()
    await pilot.pause()
    await pilot.pause()


async def _fill(pilot, input_id: str, value: str) -> None:
    widget = pilot.app.query_one(input_id, Input)
    widget.focus()
    widget.value = value
    await pilot.pause()


async def test_editor_add_rule_roundtrip_posts_validated_dict():
    app = EditorApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonasPolicyRulesEditor)
        editor.show_rules([])
        await pilot.pause()
        await _fill(pilot, "#personas-policy-kind", "mcp_tool")
        await _fill(pilot, "#personas-policy-name", "search_notes")
        pilot.app.query_one("#personas-policy-allowed", Checkbox).value = True
        pilot.app.query_one("#personas-policy-confirm", Checkbox).value = True
        await _fill(pilot, "#personas-policy-caps", "3")
        await _click(pilot, "#personas-policy-save")
        assert len(app.captured) == 1
        rules = app.captured[0].rules
        assert rules == [
            {
                "rule_kind": "mcp_tool",
                "rule_name": "search_notes",
                "allowed": True,
                "require_confirmation": True,
                "max_calls_per_turn": 3,
            }
        ]


async def test_editor_defaults_post_minimal_rule():
    app = EditorApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonasPolicyRulesEditor)
        editor.show_rules([])
        await pilot.pause()
        await _fill(pilot, "#personas-policy-kind", "skill")
        await _fill(pilot, "#personas-policy-name", "summarize")
        await _click(pilot, "#personas-policy-save")
        assert app.captured[-1].rules == [
            {
                "rule_kind": "skill",
                "rule_name": "summarize",
                "allowed": True,
                "require_confirmation": False,
                "max_calls_per_turn": None,
            }
        ]


async def test_editor_rejects_malformed_kind_with_status_message():
    app = EditorApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonasPolicyRulesEditor)
        editor.show_rules([])
        await pilot.pause()
        await _fill(pilot, "#personas-policy-kind", "bogus")
        await _fill(pilot, "#personas-policy-name", "search_notes")
        await _click(pilot, "#personas-policy-save")
        assert app.captured == []
        status = str(
            pilot.app.query_one("#personas-policy-status", Static).renderable
        )
        assert "kind" in status.lower()


async def test_editor_rejects_caps_below_one():
    app = EditorApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonasPolicyRulesEditor)
        editor.show_rules([])
        await pilot.pause()
        await _fill(pilot, "#personas-policy-kind", "mcp_tool")
        await _fill(pilot, "#personas-policy-name", "search_notes")
        await _fill(pilot, "#personas-policy-caps", "0")
        await _click(pilot, "#personas-policy-save")
        assert app.captured == []


async def test_editor_delete_removes_selected_rule_and_posts():
    app = EditorApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonasPolicyRulesEditor)
        editor.show_rules(
            [
                {
                    "rule_kind": "mcp_tool",
                    "rule_name": "search_notes",
                    "allowed": True,
                    "require_confirmation": False,
                    "max_calls_per_turn": None,
                }
            ]
        )
        await pilot.pause()
        list_view = pilot.app.query_one("#personas-policy-rules-list", ListView)
        list_view.index = 0
        list_view.action_select_cursor()
        await pilot.pause()
        await pilot.pause()
        await _click(pilot, "#personas-policy-delete")
        assert app.captured
        assert app.captured[-1].rules == []


async def test_deny_by_default_warning_appears_with_allow_rule_and_disappears():
    """HARD REQUIREMENT: visible deny-by-default warning whenever an ALLOW
    rule exists; recomputed on every rules change."""
    app = EditorApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonasPolicyRulesEditor)
        warning = pilot.app.query_one("#personas-policy-warning", Static)
        # No allow rule -> no warning.
        editor.show_rules(
            [
                {
                    "rule_kind": "mcp_tool",
                    "rule_name": "search_notes",
                    "allowed": False,
                    "require_confirmation": False,
                    "max_calls_per_turn": None,
                }
            ]
        )
        await pilot.pause()
        assert warning.display is False
        # An allow rule appears -> warning visible with the ruled copy.
        editor.show_rules(
            [
                {
                    "rule_kind": "mcp_tool",
                    "rule_name": "search_notes",
                    "allowed": True,
                    "require_confirmation": False,
                    "max_calls_per_turn": None,
                }
            ]
        )
        await pilot.pause()
        assert warning.display is True
        text = str(warning.renderable)
        assert "Allow rules active" in text
        assert "un-advertised" in text
        assert "spawn_subagent" in text
        # Removing the allow rule via a change hides it again.
        editor.show_rules([])
        await pilot.pause()
        assert warning.display is False


async def test_editor_warning_recomputed_after_save_toggle():
    app = EditorApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonasPolicyRulesEditor)
        editor.show_rules([])
        await pilot.pause()
        warning = pilot.app.query_one("#personas-policy-warning", Static)
        await _fill(pilot, "#personas-policy-kind", "skill")
        await _fill(pilot, "#personas-policy-name", "summarize")
        pilot.app.query_one("#personas-policy-allowed", Checkbox).value = True
        await _click(pilot, "#personas-policy-save")
        assert warning.display is True
        # Select the row, flip to deny, save: warning disappears.
        list_view = pilot.app.query_one("#personas-policy-rules-list", ListView)
        list_view.index = 0
        list_view.action_select_cursor()
        # Wait until the (deferred) Selected delivery has repopulated the
        # form, so the deny flip below cannot be overwritten by it.
        for _ in range(20):
            await pilot.pause()
            if "Editing rule" in str(
                pilot.app.query_one("#personas-policy-status", Static).renderable
            ):
                break
        pilot.app.query_one("#personas-policy-allowed", Checkbox).value = False
        await _click(pilot, "#personas-policy-save")
        assert warning.display is False


# ---- Inspector read-only summary -----------------------------------------


class InspectorApp(ConsolidatedCSSApp):
    def compose(self):
        yield PersonasInspectorPane(id="personas-inspector-pane")


async def test_inspector_policy_section_hidden_until_persona_selection():
    app = InspectorApp()
    async with app.run_test() as pilot:
        summary = pilot.app.query_one("#personas-policy-rules-summary", Static)
        assert summary.display is False
        inspector = pilot.app.query_one(PersonasInspectorPane)
        inspector.show_selection(name="Ops", kind="persona")
        inspector.show_policy_rules(
            [
                {
                    "rule_kind": "mcp_tool",
                    "rule_name": "search_notes",
                    "allowed": True,
                    "require_confirmation": True,
                    "max_calls_per_turn": 2,
                }
            ]
        )
        await pilot.pause()
        assert summary.display is True
        text = str(summary.renderable)
        assert "search_notes" in text
        assert "mcp_tool" in text
        await inspector.clear_selection()
        await pilot.pause()
        assert summary.display is False


async def test_inspector_policy_section_hidden_for_characters():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = pilot.app.query_one(PersonasInspectorPane)
        inspector.show_selection(name="Hero", kind="character")
        inspector.show_policy_rules([])
        await pilot.pause()
        assert (
            pilot.app.query_one("#personas-policy-rules-summary", Static).display
            is False
        )


# ---- Switcher persona label ----------------------------------------------


class _FakeRegistry:
    def __init__(self, defaults_by_id: dict[str, Any]) -> None:
        self._defaults = defaults_by_id

    def get_workspace(self, workspace_id: str):
        item = self._defaults.get(workspace_id)
        if item is None:
            return None
        return SimpleNamespace(assistant_defaults=item)


class _FakePersonaService:
    def __init__(self, records: dict[str, Any]) -> None:
        self._records = records

    def get_persona_profile(self, persona_id: str):
        record = self._records.get(persona_id)
        if record is None:
            raise KeyError(persona_id)
        return record


def _workspace(workspace_id: str = "ws-1") -> SimpleNamespace:
    return SimpleNamespace(workspace_id=workspace_id, name="Research")


def test_switcher_suffix_available_default_appends_label():
    app = SimpleNamespace(
        workspace_registry_service=_FakeRegistry(
            {
                "ws-1": SimpleNamespace(
                    assistant_kind="persona",
                    assistant_id="p1",
                    persona_memory_mode="read_only",
                )
            }
        ),
        local_character_persona_service=_FakePersonaService(
            {"p1": {"id": "p1", "name": "Ops persona", "deleted": False}}
        ),
    )
    assert workspace_persona_label_suffix(app, _workspace()) == " · Ops persona"


def test_switcher_suffix_absent_without_default_or_missing_persona():
    no_defaults = SimpleNamespace(
        workspace_registry_service=_FakeRegistry({}),
        local_character_persona_service=_FakePersonaService({}),
    )
    assert workspace_persona_label_suffix(no_defaults, _workspace()) == ""
    deleted_persona = SimpleNamespace(
        workspace_registry_service=_FakeRegistry(
            {
                "ws-1": SimpleNamespace(
                    assistant_kind="persona",
                    assistant_id="p1",
                    persona_memory_mode="read_only",
                )
            }
        ),
        local_character_persona_service=_FakePersonaService(
            {"p1": {"id": "p1", "name": "Ghost", "deleted": True}}
        ),
    )
    assert workspace_persona_label_suffix(deleted_persona, _workspace()) == ""


def test_switcher_suffix_degrades_silently_without_services():
    app = SimpleNamespace()
    assert workspace_persona_label_suffix(app, _workspace()) == ""


# ---- Import review policy-rule count -------------------------------------

def _import_review_with_carried_persona(tmp_path, carried_persona) -> Any:
    """Import a valid archive whose pack.json carries a persona record."""
    import json

    from Tests.Persona_Visual.test_persona_visual_importer import (
        _archive_payloads,
        _canonical,
        _identity,
        _replace_declared_payload,
        _write_archive,
    )
    from tldw_chatbook.Persona_Visual.importer import import_persona_visual_pack

    payloads = _archive_payloads()
    if carried_persona is not None:
        pack = json.loads(payloads["metadata/pack.json"])
        pack["pack"]["persona"] = carried_persona
        _replace_declared_payload(
            payloads, "metadata/pack.json", _canonical(pack)
        )
    archive = _write_archive(tmp_path / "pack.tldw-persona-vpack", payloads)
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700, parents=True, exist_ok=True)
    return import_persona_visual_pack(
        archive,
        staging_root=staging,
        persona_id="local-persona-1",
        persona_revision=7,
        expected_identity=_identity(),
    )


def test_import_review_counts_carried_policy_rules(tmp_path):
    rules = [
        {
            "rule_kind": "mcp_tool",
            "rule_name": "search_notes",
            "allowed": True,
            "require_confirmation": False,
            "max_calls_per_turn": 2,
        },
        {"rule_kind": "skill", "rule_name": "summarize", "allowed": False},
    ]
    review = _import_review_with_carried_persona(
        tmp_path, {"id": "p1", "policy_rules": rules}
    )
    assert review.policy_rule_count == 2


def test_import_review_policy_rule_count_zero_without_carried_record(tmp_path):
    review = _import_review_with_carried_persona(tmp_path, None)
    assert review.policy_rule_count == 0


async def test_pack_widget_notice_displays_narrowing_only_copy():
    class PackApp(ConsolidatedCSSApp):
        def compose(self):
            yield PersonasPersonaVisualPackWidget()

    app = PackApp()
    async with app.run_test() as pilot:
        browser = pilot.app.query_one(PersonasPersonaVisualPackWidget)
        browser.show_policy_rule_notice(2)
        await pilot.pause()
        notice = str(
            pilot.app.query_one("#personas-persona-visual-notice", Static).renderable
        )
        assert (
            "Carries 2 narrowing-only tool policy rule(s) — review before publishing."
            in notice
        )
        browser.show_policy_rule_notice(0)
        await pilot.pause()
        notice = str(
            pilot.app.query_one("#personas-persona-visual-notice", Static).renderable
        )
        assert "policy rule" not in notice
