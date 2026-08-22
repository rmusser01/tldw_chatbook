# test_chatbook_import_wizard_backup_honesty.py
# Description: Regression coverage for task-19550 (import wizard claimed a backup it never took)
"""
`ChatbookImportWizard` used to offer a default-ON "Create backup" checkbox
("Backup current database before importing"), read its value into the import
options, and then -- with the implementation still a bare
``# TODO: Implement actual backup functionality`` -- paint
``"✓ Created backup"`` into the progress list before running a
database-mutating import that has no rollback.

These tests pin the honest behaviour: no backup control, no backup status
row, no success claim for work that never ran, and an explicit statement of
the real risk (the import cannot be undone) on the last step the user can
still back out of.

The last test is the generic guard AC#4 asks for: any function in this module
that paints a ``"completed"`` status row must not carry a
TODO/"For now"/"not implemented" marker, and the set of status rows that can
be marked completed is pinned to the audited allowlist.
"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Checkbox, Static

from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ChatbookVersion
from tldw_chatbook.UI.Wizards import ChatbookImportWizard as wizard_module
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.ChatbookImportWizard import (
    ImportOptionsStep,
    ImportProgressStep,
)

# Every status row this flow is allowed to hard-code as "completed", and the
# work that earns the claim (task-19550 audit):
#   status-prepare       -- the importer/server request was actually built
#   status-indexes       -- FTS5 triggers maintain the indexes inside the writes
#   status-finalize      -- the import call returned
# "status-backup" is deliberately absent: no backup is taken, so no row may
# claim one.
#
# The four per-type rows (status-conversations / -notes / -characters /
# -media) used to be in this set. They left it in task-19734, which found
# that a literal "completed" was exactly the bug: the row was ticked off a
# MANIFEST count, so an all-skipped re-import showed four green
# "✓ Imported ..." rows over "Imported: 0". Their state is now derived from
# the import's per-type results, and
# Tests/Chatbooks/test_chatbook_import_result_honesty.py pins that no
# constant outcome state may be hard-coded for them again.
AUDITED_COMPLETED_STATUS_IDS = frozenset(
    {
        "status-prepare",
        "status-indexes",
        "status-finalize",
    }
)

_UNIMPLEMENTED_MARKERS = ("TODO", "FIXME", "For now", "not implemented", "placeholder")


class _StepHost(App):
    """Mount a single wizard step, the idiom used by Tests/Wizards."""

    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


def _fake_wizard(**wizard_data) -> SimpleNamespace:
    return SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data=dict(wizard_data),
        initial_execution_mode="local",
        can_go_back=True,
    )


def _options_step() -> ImportOptionsStep:
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Bundle",
        description="Bundle",
    )
    manifest.total_conversations = 1
    return ImportOptionsStep(
        wizard=_fake_wizard(**{"preview-validation": {"manifest": manifest}}),
        config=WizardStepConfig(
            id="import-options",
            title="Import Options",
            step_number=4,
        ),
    )


@pytest.mark.asyncio
async def test_import_options_step_offers_no_backup_control():
    """AC#1: the wizard must not offer to back the database up at all --
    a disabled or greyed-out box would still read as 'backup handled'."""
    step = _options_step()
    async with _StepHost(step).run_test(size=(120, 48)) as pilot:
        await pilot.pause()

        checkbox_labels = [
            str(checkbox.label).lower() for checkbox in step.query(Checkbox)
        ]
        checkbox_ids = [checkbox.id or "" for checkbox in step.query(Checkbox)]
        rendered = " ".join(
            str(static.renderable).lower() for static in step.query(Static)
        )

        assert checkbox_labels, "the options step should still offer its real options"
        assert not any("backup" in label for label in checkbox_labels), checkbox_labels
        assert not any("backup" in widget_id for widget_id in checkbox_ids), (
            checkbox_ids
        )
        assert "backup current database" not in rendered

        assert "create_backup" not in step.get_step_data()


@pytest.mark.asyncio
async def test_import_options_step_states_that_the_import_cannot_be_undone():
    """AC#5: the honest risk has to be visible on the last step the user can
    still cancel from -- the next step starts writing immediately."""
    step = _options_step()
    async with _StepHost(step).run_test(size=(120, 48)) as pilot:
        await pilot.pause()

        rendered = " ".join(
            str(static.renderable).lower() for static in step.query(Static)
        )

        assert "cannot be undone" in rendered
        assert "writes directly into your databases" in rendered


@pytest.mark.asyncio
async def test_import_run_never_claims_a_backup_it_did_not_take(monkeypatch):
    """AC#2, end to end: drive the real import path with a stale
    ``create_backup=True`` option still in the wizard data and prove no status
    update ever mentions a backup."""
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Bundle",
        description="Bundle",
    )
    manifest.total_conversations = 2

    step = ImportProgressStep(
        wizard=_fake_wizard(
            **{
                "file-selection": {"file_path": "/tmp/does-not-need-to-exist.zip"},
                "preview-validation": {"manifest": manifest},
                "conflict-resolution": {},
                # A stale option key from an older wizard version must not be
                # able to resurrect the claim.
                "import-options": {"execution_mode": "local", "create_backup": True},
            }
        ),
        config=WizardStepConfig(id="import-progress", title="Importing", step_number=5),
    )

    calls = {"importer": 0, "import_chatbook": 0}

    class _FakeImporter:
        def __init__(self, db_paths):
            calls["importer"] += 1
            self.db_paths = db_paths

        def import_chatbook(self, **kwargs):
            calls["import_chatbook"] += 1
            status = kwargs["import_status"]
            status.total_items = 2
            status.processed_items = 2
            status.successful_items = 2
            return True, "ok"

    # from-imports bind at import time: patch the CONSUMER namespace.
    monkeypatch.setattr(wizard_module, "ChatbookImporter", _FakeImporter)
    monkeypatch.setattr(
        wizard_module,
        "get_chatbook_database_paths",
        lambda: {"ChaChaNotes": "/tmp/cn.db"},
    )

    statuses: list[tuple[str, str, str]] = []
    completions: list[str] = []
    errors: list[str] = []

    async def _completion():
        completions.append("completed")

    async def _error(message):
        errors.append(message)

    step._update_status = lambda status_id, state, text: statuses.append(
        (status_id, state, text)
    )
    step._update_progress = lambda value: None
    step._show_completion = _completion
    step._show_error = _error

    await step._import_chatbook()

    # Prove the patched seams were the ones that ran, and that the run reached
    # the success path rather than dying early into the except branch.
    assert errors == []
    assert calls == {"importer": 1, "import_chatbook": 1}
    assert completions == ["completed"]
    assert statuses, "the import must still report its real progress"

    assert not any(status_id == "status-backup" for status_id, _, _ in statuses)
    assert not any("backup" in text.lower() for _, _, text in statuses)


def test_no_status_row_is_painted_from_an_unimplemented_code_path():
    """AC#4: a status row may not report progress or success from a TODO/no-op
    path, and the rows that hard-code "completed" are pinned to the audited
    allowlist.

    The marker scan covers every function that paints a status row, not only
    the ones with a literal "completed" -- task-19734 moved the four per-type
    rows onto result-derived states, and a TODO behind one of those would
    otherwise have slipped out of this guard's reach.
    """
    source = Path(inspect.getfile(wizard_module)).read_text(encoding="utf-8")
    tree = ast.parse(source)

    completed_status_ids: set[str] = set()
    status_painting_functions: list[str] = []
    offenders: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        status_calls = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_update_status"
        ]
        if not status_calls:
            continue

        status_painting_functions.append(node.name)
        completed_status_ids |= {
            call.args[0].value
            for call in status_calls
            if len(call.args) >= 2
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[1], ast.Constant)
            and call.args[1].value == "completed"
        }

        segment = ast.get_source_segment(source, node) or ""
        found = [marker for marker in _UNIMPLEMENTED_MARKERS if marker in segment]
        if found:
            offenders.append(f"{node.name}: {', '.join(found)}")

    assert completed_status_ids, "the parser found no completed status rows at all"
    assert "_paint_type_result_rows" in status_painting_functions, (
        "the result-driven row painter went missing; the per-type rows are "
        "back to being painted some other way"
    )
    assert offenders == [], (
        "these functions paint a status row while still carrying an "
        f"unimplemented marker: {offenders}"
    )
    assert completed_status_ids == set(AUDITED_COMPLETED_STATUS_IDS)
