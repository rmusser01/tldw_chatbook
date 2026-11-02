# test_chatbook_wizard_blocking_work_off_event_loop.py
# Description: Regression coverage for TASK-38595 (local import/export ran
#              synchronously inside the wizard workers, freezing the TUI)
"""
The chatbook wizards ran the full synchronous import/export pipelines
(``ChatbookImporter.preview_chatbook`` / ``scan_conflicts`` /
``import_chatbook``, ``ChatbookCreator.create_chatbook``) directly inside
their async workers. A coroutine that never awaits anything blocks the
event loop for its whole body, so the whole TUI froze until the ZIP
preflight, extraction, SQLite writes and media compression finished --
only the progress painted before and after the call was ever visible.

These tests pin the two halves of the fix (TASK-38595):

1. every blocking importer/creator call is dispatched through
   ``asyncio.to_thread`` and therefore executes on a worker thread, not
   the event-loop thread; and
2. each wizard worker dispatch carries ``exclusive=True`` and its own
   ``group=`` so a re-show of a step cancels the previous run instead of
   piling a second blocking import onto the first.

The importer/creator are stubbed with recording fakes; no real database
is constructed, so unlike the honesty suites these tests do not need the
bootstrap-profile config admission chain.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
)
from tldw_chatbook.UI.Wizards import ChatbookCreationWizard as creation_module
from tldw_chatbook.UI.Wizards import ChatbookImportWizard as import_module
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.ChatbookCreationWizard import ProgressStep
from tldw_chatbook.UI.Wizards.ChatbookImportWizard import (
    ConflictResolutionStep,
    ImportProgressStep,
    PreviewValidationStep,
)


class _StepHost(App):
    """Mount a single wizard step, the idiom used by Tests/Wizards."""

    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


class _FakeWizard:
    """The slice of ``WizardContainer`` a mounted step touches."""

    def __init__(self, **wizard_data):
        self.app_instance = MagicMock(app_config={})
        self.wizard_data = dict(wizard_data)
        self.initial_execution_mode = "local"
        self.can_go_back = True
        self.next_button = SimpleNamespace(label="Next", variant="primary")

    def query_one(self, selector, expect_type=None):
        assert selector == "#wizard-next", selector
        return self.next_button

    def refresh_current_step(self) -> None:
        return None


def _empty_manifest() -> ChatbookManifest:
    return ChatbookManifest(
        version=ChatbookVersion.V1, name="Bundle", description="d"
    )


@pytest.fixture
def to_thread_calls(monkeypatch):
    """Record every ``asyncio.to_thread`` dispatch, then delegate.

    The wizards call ``asyncio.to_thread(...)`` through the module
    attribute, so patching ``asyncio.to_thread`` itself is the recording
    seam; delegation keeps the real behaviour (a worker thread) intact.
    """
    calls = []
    real_to_thread = asyncio.to_thread

    async def _recording_to_thread(func, /, *args, **kwargs):
        calls.append((getattr(func, "__name__", str(func)), args, kwargs))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _recording_to_thread)
    return calls


def _dispatches_for(step) -> list:
    """Run ``on_show`` with ``run_worker`` replaced by a kwarg recorder."""
    dispatches: list = []

    def _record(work, **kwargs):
        dispatches.append(kwargs)

    step.run_worker = _record  # type: ignore[method-assign]
    step.on_show()
    return dispatches


def _assert_ran_off_loop(threads: dict, key: str) -> None:
    assert key in threads, "the stubbed blocking call never executed"
    assert threads[key] != threading.get_ident(), (
        "the blocking call ran on the caller's (event-loop) thread"
    )


# ---------------------------------------------------------------------------
# Worker dispatch: exclusive + distinct group
# ---------------------------------------------------------------------------


def test_preview_worker_is_exclusive_and_grouped():
    step = PreviewValidationStep(
        wizard=_FakeWizard(),
        config=WizardStepConfig(
            id="preview-validation", title="Preview", step_number=2
        ),
    )

    assert _dispatches_for(step) == [
        {"exclusive": True, "group": "chatbook-preview"}
    ]


def test_conflict_scan_worker_is_exclusive_and_grouped():
    step = ConflictResolutionStep(
        wizard=_FakeWizard(),
        config=WizardStepConfig(
            id="conflict-resolution", title="Conflicts", step_number=3
        ),
    )

    assert _dispatches_for(step) == [
        {"exclusive": True, "group": "chatbook-import-conflict-scan"}
    ]


def test_import_worker_is_exclusive_and_grouped():
    step = ImportProgressStep(
        wizard=_FakeWizard(),
        config=WizardStepConfig(
            id="import-progress", title="Importing", step_number=5
        ),
    )

    assert _dispatches_for(step) == [
        {"exclusive": True, "group": "chatbook-import"}
    ]


def test_export_worker_is_exclusive_and_grouped():
    step = ProgressStep(
        wizard=_FakeWizard(),
        config=WizardStepConfig(
            id="export-progress", title="Creating", step_number=5
        ),
    )

    assert _dispatches_for(step) == [
        {"exclusive": True, "group": "chatbook-export"}
    ]


# ---------------------------------------------------------------------------
# The blocking calls run via to_thread, off the event loop
# ---------------------------------------------------------------------------


async def test_preview_runs_off_the_event_loop(monkeypatch, to_thread_calls):
    threads: dict = {}

    class _StubImporter:
        def __init__(self, db_paths):
            pass

        def preview_chatbook(self, path):
            threads["preview"] = threading.get_ident()
            return _empty_manifest(), None

    monkeypatch.setattr(import_module, "ChatbookImporter", _StubImporter)
    monkeypatch.setattr(import_module, "get_chatbook_database_paths", lambda: {})
    step = PreviewValidationStep(
        wizard=_FakeWizard(**{"file-selection": {"file_path": "/tmp/b.zip"}}),
        config=WizardStepConfig(
            id="preview-validation", title="Preview", step_number=2
        ),
    )
    step.run_worker = lambda work, **kwargs: None  # type: ignore[method-assign]

    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await step._load_and_validate_chatbook()
        await pilot.pause()

    _assert_ran_off_loop(threads, "preview")
    assert any(name == "preview_chatbook" for name, _a, _k in to_thread_calls), (
        "preview_chatbook must be dispatched through asyncio.to_thread"
    )


async def test_conflict_scan_runs_off_the_event_loop(monkeypatch, to_thread_calls):
    threads: dict = {}

    class _StubImporter:
        def __init__(self, db_paths):
            pass

        def scan_conflicts(self, path):
            threads["scan"] = threading.get_ident()
            return [], None

    monkeypatch.setattr(import_module, "ChatbookImporter", _StubImporter)
    monkeypatch.setattr(import_module, "get_chatbook_database_paths", lambda: {})
    step = ConflictResolutionStep(
        wizard=_FakeWizard(
            **{
                "file-selection": {"file_path": "/tmp/b.zip"},
                "preview-validation": {"manifest": _empty_manifest()},
            }
        ),
        config=WizardStepConfig(
            id="conflict-resolution", title="Conflicts", step_number=3
        ),
    )
    step.run_worker = lambda work, **kwargs: None  # type: ignore[method-assign]

    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await step._scan_conflicts()
        await pilot.pause()

    _assert_ran_off_loop(threads, "scan")
    assert any(name == "scan_conflicts" for name, _a, _k in to_thread_calls), (
        "scan_conflicts must be dispatched through asyncio.to_thread"
    )


async def test_local_import_runs_off_the_event_loop(monkeypatch, to_thread_calls):
    from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution

    threads: dict = {}

    class _StubImporter:
        def __init__(self, db_paths):
            pass

        def import_chatbook(self, **kwargs):
            threads["import"] = threading.get_ident()
            return True, "ok"

    monkeypatch.setattr(import_module, "ChatbookImporter", _StubImporter)
    monkeypatch.setattr(import_module, "get_chatbook_database_paths", lambda: {})
    step = ImportProgressStep(
        wizard=_FakeWizard(
            **{
                "file-selection": {"file_path": "/tmp/b.zip"},
                "preview-validation": {"manifest": _empty_manifest()},
                "conflict-resolution": {
                    "resolution_strategy": ConflictResolution.SKIP
                },
                "import-options": {
                    "execution_mode": "local",
                    "import_media": True,
                    "prefix_imported": False,
                },
            }
        ),
        config=WizardStepConfig(
            id="import-progress", title="Importing", step_number=5
        ),
    )
    step.run_worker = lambda work, **kwargs: None  # type: ignore[method-assign]

    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await step._import_chatbook()
        await pilot.pause()

    _assert_ran_off_loop(threads, "import")
    assert any(name == "import_chatbook" for name, _a, _k in to_thread_calls), (
        "import_chatbook must be dispatched through asyncio.to_thread"
    )
    assert step.is_complete is True


async def test_local_export_runs_off_the_event_loop(
    tmp_path, monkeypatch, to_thread_calls
):
    threads: dict = {}

    class _StubCreator:
        def __init__(self, db_paths):
            pass

        def create_chatbook(self, **kwargs):
            threads["create"] = threading.get_ident()
            output_path = kwargs["output_path"]
            output_path.write_bytes(b"stub chatbook")
            return True, "created", {}

    monkeypatch.setattr(creation_module, "ChatbookCreator", _StubCreator)
    monkeypatch.setattr(
        creation_module, "get_chatbook_database_paths", lambda: {}
    )
    monkeypatch.setattr(
        creation_module, "secure_private_directory", lambda *args, **kwargs: None
    )
    export_path = tmp_path / "bundle.zip"
    step = ProgressStep(
        wizard=_FakeWizard(
            **{
                "basic-info": {"name": "Bundle"},
                "content-selection": {"selections": {}},
                "export-options": {"execution_mode": "local"},
                "export_path": str(export_path),
            }
        ),
        config=WizardStepConfig(
            id="export-progress", title="Creating", step_number=5
        ),
    )
    step.run_worker = lambda work, **kwargs: None  # type: ignore[method-assign]

    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await step._export_chatbook()
        await pilot.pause()

    _assert_ran_off_loop(threads, "create")
    assert any(name == "create_chatbook" for name, _a, _k in to_thread_calls), (
        "create_chatbook must be dispatched through asyncio.to_thread"
    )
    assert step.is_complete is True
    assert step.export_path == export_path
