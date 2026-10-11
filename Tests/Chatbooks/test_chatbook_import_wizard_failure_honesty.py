# test_chatbook_import_wizard_failure_honesty.py
# Description: Regression coverage for task-19869 (non-happy paths
#              understated what was written)
"""
Two paths on which the chatbook import wizard told the user less than it
knew (task-19869):

1. **A fatal error hid a completed write.** The exception handler updated
   only the finalize row and the panel title; the per-type rows froze on
   "⟳ Preparing ...", and no stats panel rendered at all. Meanwhile every
   item already written was permanently in the databases (there is no
   rollback, task-19550/279), so the screen read "nothing happened" about a
   run that had committed items.
2. **Server mode reported a 4-item chatbook as empty.** The job payload's
   ``successful_items`` defaulted to 0, ``skipped_items`` was hard-set to 0
   regardless of what the server did, and their sum drove the outcome
   vocabulary into EMPTY -- "this chatbook contained no items" -- about a
   chatbook whose manifest the wizard was holding in memory.

The pinned behaviour: the failure surface states that items may already
have been written and cannot be rolled back; per-type rows say what each
type's records show, or "unknown" -- never a frozen "Preparing"; a missing
server count is unknown, not zero; and "contained no items" is a claim only
a manifest with no items can earn.
"""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ContentType,
)
from tldw_chatbook.UI.Wizards import ChatbookImportWizard as wizard_module
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.ChatbookImportWizard import (
    PER_TYPE_STATUS_ROWS,
    ImportProgressStep,
)

pytestmark = pytest.mark.bootstrap_profile


class _StepHost(App):
    """Mount a single wizard step, the idiom used by Tests/Wizards."""

    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


class _FakeWizard:
    """The slice of ``WizardContainer`` a mounted progress step touches."""

    def __init__(self, **wizard_data):
        self.app_instance = MagicMock(app_config={})
        self.wizard_data = dict(wizard_data)
        self.initial_execution_mode = "local"
        self.can_go_back = True
        self.next_button = SimpleNamespace(label="Next →", variant="primary")

    def query_one(self, selector, expect_type=None):
        assert selector == "#wizard-next", selector
        return self.next_button


def _fake_wizard(**wizard_data) -> _FakeWizard:
    return _FakeWizard(**wizard_data)


def _manifest_with(items: list) -> ChatbookManifest:
    """A manifest holding the given (type, count) items, via from_dict."""
    content_items = []
    for content_type, count in items:
        for index in range(count):
            content_items.append(
                {
                    "id": f"{content_type.value}-{index}",
                    "type": content_type.value,
                    "title": f"Item {index}",
                    "description": None,
                    "created_at": None,
                    "updated_at": None,
                    "tags": [],
                    "metadata": {},
                    "file_path": None,
                }
            )
    now = datetime.now().isoformat()
    payload = {
        "version": ChatbookVersion.V1.value,
        "name": "Hand-built bundle",
        "description": "d",
        "author": "Tests",
        "created_at": now,
        "updated_at": now,
        "content_items": content_items,
        "relationships": [],
        "include_media": False,
        "include_embeddings": False,
        "media_quality": "thumbnail",
        "statistics": {
            "total_conversations": sum(
                1 for t, _ in items if t == ContentType.CONVERSATION
            ),
            "total_notes": sum(1 for t, _ in items if t == ContentType.NOTE),
            "total_characters": sum(
                1 for t, _ in items if t == ContentType.CHARACTER
            ),
            "total_media_items": sum(1 for t, _ in items if t == ContentType.MEDIA),
            "total_prompts": 0,
            "total_kept_briefings": 0,
            "total_size_bytes": 0,
        },
        "tags": [],
        "categories": [],
        "language": "en",
        "license": None,
    }
    return ChatbookManifest.from_dict(payload)


def _progress_step(manifest: ChatbookManifest, **option_overrides) -> ImportProgressStep:
    options = {
        "execution_mode": "local",
        "import_media": True,
        "prefix_imported": False,
    }
    options.update(option_overrides)
    return ImportProgressStep(
        wizard=_fake_wizard(
            **{
                "file-selection": {"file_path": "/tmp/bundle.zip"},
                "preview-validation": {"manifest": manifest},
                "conflict-resolution": {},
                "import-options": options,
            }
        ),
        config=WizardStepConfig(id="import-progress", title="Importing", step_number=5),
    )


async def _run_step_to_settlement(step, pilot):
    """Let the on_show worker finish (or drive the import directly)."""
    for _ in range(300):
        worker = getattr(step, "import_worker", None)
        if worker is not None and not worker.is_running:
            return
        await pilot.pause()
    await step._import_chatbook()


def _read_surface(step) -> dict:
    """Everything the completion/error surface renders."""
    rows = {
        row_id: str(step.query_one(f"#{row_id}", Static).renderable)
        for row_id, _content_type, _noun in PER_TYPE_STATUS_ROWS
    }
    return {
        "title": str(step.query_one("#progress-title", Static).renderable),
        "banner": str(step.query_one("#completion-message", Static).renderable),
        "stats": {
            stat: str(step.query_one(f"#{stat}", Static).renderable)
            for stat in ("stat-total", "stat-imported", "stat-skipped", "stat-failed")
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Path 1: a fatal error after at least one item is committed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fatal_error_states_what_was_already_written(monkeypatch):
    """AC#1/AC#2, born red at base.

    The importer commits one character and one conversation, plans two
    notes, and dies. At base the screen froze on "⟳ Preparing ..." with no
    stats panel; it must instead state that the two written items stay
    written, that the import cannot be rolled back, and that the unplanned
    media type is unknown.
    """
    manifest = _manifest_with(
        [
            (ContentType.CHARACTER, 1),
            (ContentType.CONVERSATION, 1),
            (ContentType.NOTE, 2),
            (ContentType.MEDIA, 1),
        ]
    )
    step = _progress_step(manifest)

    class _FakeImporter:
        def __init__(self, db_paths):
            self.db_paths = db_paths

        def import_chatbook(self, **kwargs):
            status = kwargs["import_status"]
            # Two types land...
            status.plan(ContentType.CHARACTER, 1)
            status.record_success(ContentType.CHARACTER)
            status.plan(ContentType.CONVERSATION, 1)
            status.record_success(ContentType.CONVERSATION)
            # ...one is mid-flight when the run dies...
            status.plan(ContentType.NOTE, 2)
            raise RuntimeError("disk exploded mid-import")

    monkeypatch.setattr(wizard_module, "ChatbookImporter", _FakeImporter)
    monkeypatch.setattr(
        wizard_module,
        "get_chatbook_database_paths",
        lambda: {"ChaChaNotes": "/tmp/cn.db"},
    )

    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        await _run_step_to_settlement(step, pilot)
        surface = _read_surface(step)

    assert surface["title"] == "❌ Import Failed"
    # The two committed items are named as written-and-staying.
    assert "2 item(s) were already written and stay written" in surface["banner"], (
        surface["banner"]
    )
    assert "cannot be rolled back" in surface["banner"], surface["banner"]
    # The mid-flight notes are named as unknown, not as nothing.
    assert "2 item(s) the import was working on may also have landed" in surface[
        "banner"
    ], surface["banner"]

    # Per-type rows: recorded results for the landed types, unknown for the
    # rest -- nothing frozen on "Preparing".
    assert surface["rows"]["status-characters"] == "✓ Imported 1 characters"
    assert surface["rows"]["status-conversations"] == "✓ Imported 1 conversations"
    assert "unaccounted for" in surface["rows"]["status-notes"], surface["rows"]
    assert "Unknown" in surface["rows"]["status-media"], surface["rows"]
    assert not any(
        "Preparing" in text or "Importing" in text for text in surface["rows"].values()
    ), surface["rows"]

    # The stats panel renders, with the counts recorded before the failure.
    assert surface["stats"] == {
        "stat-total": "4",
        "stat-imported": "2",
        "stat-skipped": "0",
        "stat-failed": "0",
    }, surface["stats"]


@pytest.mark.asyncio
async def test_fatal_error_with_no_recorded_result_says_may_have_written(monkeypatch):
    """The run died before recording anything: the surface still refuses to
    claim nothing happened."""
    manifest = _manifest_with([(ContentType.NOTE, 3)])
    step = _progress_step(manifest)

    class _FakeImporter:
        def __init__(self, db_paths):
            self.db_paths = db_paths

        def import_chatbook(self, **kwargs):
            raise RuntimeError("died before recording anything")

    monkeypatch.setattr(wizard_module, "ChatbookImporter", _FakeImporter)
    monkeypatch.setattr(
        wizard_module,
        "get_chatbook_database_paths",
        lambda: {"ChaChaNotes": "/tmp/cn.db"},
    )

    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        await _run_step_to_settlement(step, pilot)
        surface = _read_surface(step)

    assert "❌ Import failed" in surface["banner"], surface["banner"]
    assert "may still have been written" in surface["banner"], surface["banner"]
    assert "cannot be rolled back" in surface["banner"], surface["banner"]
    assert "Unknown" in surface["rows"]["status-notes"], surface["rows"]


@pytest.mark.asyncio
async def test_failed_import_return_reports_written_items_too(monkeypatch):
    """The importer can also return ``success=False`` with a partially
    populated status; that path gets the same surface (at base it rendered
    no stats panel either)."""
    manifest = _manifest_with([(ContentType.CHARACTER, 1), (ContentType.NOTE, 1)])
    step = _progress_step(manifest)

    class _FakeImporter:
        def __init__(self, db_paths):
            self.db_paths = db_paths

        def import_chatbook(self, **kwargs):
            status = kwargs["import_status"]
            status.plan(ContentType.CHARACTER, 1)
            status.record_success(ContentType.CHARACTER)
            return False, "fatal: could not finish"

    monkeypatch.setattr(wizard_module, "ChatbookImporter", _FakeImporter)
    monkeypatch.setattr(
        wizard_module,
        "get_chatbook_database_paths",
        lambda: {"ChaChaNotes": "/tmp/cn.db"},
    )

    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        await _run_step_to_settlement(step, pilot)
        surface = _read_surface(step)

    assert surface["stats"]["stat-imported"] == "1", surface["stats"]
    assert "1 item(s) were already written" in surface["banner"], surface["banner"]


# ---------------------------------------------------------------------------
# Path 2: server-mode payloads
# ---------------------------------------------------------------------------


class _FakeService:
    def __init__(self, payload):
        self.payload = payload
        self.import_calls: list[dict] = []

    def validate_server_import_selection(self, selections):
        return []

    async def import_chatbook_from_selection(self, path, **kwargs):
        self.import_calls.append(kwargs)
        return dict(self.payload)

    async def get_import_job(self, job_id):
        return dict(self.payload)


def _server_step(manifest: ChatbookManifest, payload: dict) -> ImportProgressStep:
    return _progress_step(
        manifest, execution_mode="server", import_media=False, import_embeddings=False
    )


async def _run_server_import(
    monkeypatch, manifest: ChatbookManifest, payload: dict
) -> dict:
    step = _server_step(manifest, payload)
    service = _FakeService(payload)
    lease = SimpleNamespace(service=service)

    def _fake_lease(*args, **kwargs):
        return lease

    async def _fake_close(_lease):
        return None

    monkeypatch.setattr(wizard_module, "server_chatbook_service_lease", _fake_lease)
    monkeypatch.setattr(
        wizard_module, "close_server_chatbook_service_lease", _fake_close
    )
    monkeypatch.setattr(wizard_module, "record_server_job", lambda *a, **k: None)

    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        await _run_step_to_settlement(step, pilot)
        return _read_surface(step)


@pytest.mark.asyncio
async def test_server_payload_without_counts_is_unknown_not_empty(monkeypatch):
    """AC#4/AC#5, born red at base.

    A completed job whose payload omits ``successful_items`` used to render
    "this chatbook contained no items" about a chatbook whose 4-item
    manifest the wizard was holding. It must instead say the server told us
    nothing, and present the counts as unknown.
    """
    manifest = _manifest_with([(ContentType.CONVERSATION, 4)])
    payload = {"job_id": "job-1", "status": "completed"}

    surface = await _run_server_import(monkeypatch, manifest, payload)

    assert "contained no items" not in surface["banner"], surface["banner"]
    assert "no item counts" in surface["banner"], surface["banner"]
    assert "unknown" in surface["banner"], surface["banner"]
    assert "4 item(s)" in surface["banner"], surface["banner"]
    assert "Unknown" in surface["title"], surface["title"]
    assert surface["stats"] == {
        "stat-total": "?",
        "stat-imported": "?",
        "stat-skipped": "?",
        "stat-failed": "?",
    }, surface["stats"]
    # The per-type rows stay the honest "no breakdown" line, not a claim
    # about the chatbook.
    assert all(
        "No per-type breakdown returned by the server" in text
        for text in surface["rows"].values()
    ), surface["rows"]


@pytest.mark.asyncio
async def test_server_reported_skips_are_not_invented_away(monkeypatch):
    """AC#4, born red at base: ``skipped_items = 0`` was hard-set, so a
    payload that reported 2 imported / 1 skipped displayed "2 of 2
    imported" with Skipped: 0."""
    manifest = _manifest_with([(ContentType.CONVERSATION, 3)])
    payload = {
        "job_id": "job-1",
        "status": "completed",
        "successful_items": 2,
        "failed_items": 0,
        "skipped_items": 1,
    }

    surface = await _run_server_import(monkeypatch, manifest, payload)

    assert surface["stats"]["stat-skipped"] == "1", surface["stats"]
    assert surface["stats"]["stat-imported"] == "2", surface["stats"]
    assert "2 of 3 item(s) imported" in surface["banner"], surface["banner"]
    assert "1 skipped" in surface["banner"], surface["banner"]


@pytest.mark.asyncio
async def test_server_all_skipped_is_not_an_empty_chatbook(monkeypatch):
    """AC#3: a payload of 0 imported / 4 skipped for a 4-item chatbook used
    to read EMPTY ("contained no items"); it is the SKIPPED outcome."""
    manifest = _manifest_with([(ContentType.CONVERSATION, 4)])
    payload = {
        "job_id": "job-1",
        "status": "completed",
        "successful_items": 0,
        "failed_items": 0,
        "skipped_items": 4,
    }

    surface = await _run_server_import(monkeypatch, manifest, payload)

    assert "contained no items" not in surface["banner"], surface["banner"]
    assert "4 of 4 item(s) were already present" in surface["banner"], surface["banner"]
    assert surface["stats"]["stat-skipped"] == "4", surface["stats"]


@pytest.mark.asyncio
async def test_server_zero_counts_for_a_listed_chatbook_is_not_an_empty_file(monkeypatch):
    """AC#3's remaining edge: the server reports explicit zeros for a
    chatbook whose manifest lists items. "This chatbook contained no items"
    is a claim about the FILE, and the manifest in hand contradicts it."""
    manifest = _manifest_with([(ContentType.CONVERSATION, 4)])
    payload = {
        "job_id": "job-1",
        "status": "completed",
        "successful_items": 0,
        "failed_items": 0,
        "skipped_items": 0,
    }

    surface = await _run_server_import(monkeypatch, manifest, payload)

    assert "contained no items" not in surface["banner"], surface["banner"]
    assert "Nothing was imported" in surface["banner"], surface["banner"]
    assert "4 item(s)" in surface["banner"], surface["banner"]


@pytest.mark.asyncio
async def test_server_failure_names_the_error_and_keeps_rows_unknown(monkeypatch):
    """A server job that ends 'failed' surfaces through the same honest
    error path: nothing claimed imported, present types unknown."""
    manifest = _manifest_with([(ContentType.CONVERSATION, 2)])
    payload = {"job_id": "job-1", "status": "failed", "message": "server exploded"}

    surface = await _run_server_import(monkeypatch, manifest, payload)

    assert surface["title"] == "❌ Import Failed"
    assert "❌ Import failed" in surface["banner"], surface["banner"]
    assert "cannot be rolled back" in surface["banner"], surface["banner"]
    assert "Unknown" in surface["rows"]["status-conversations"], surface["rows"]


# ---------------------------------------------------------------------------
# The vocabulary lock
# ---------------------------------------------------------------------------


def test_the_empty_claim_is_a_file_claim_and_stays_earned():
    """AC#7: the one-vocabulary rule (empty = a claim about the file,
    excluded = a claim about the run) is extended, not bypassed: the wizard
    only renders the EMPTY banner when nothing contradicts the manifest."""
    source = Path(wizard_module.__file__).read_text(encoding="utf-8")
    # The EMPTY banner still exists for the genuinely empty chatbook...
    assert "this chatbook contained no items" in source
    # ...and the manifest guard stands between it and a listed chatbook.
    assert "_manifest_item_count" in source
    # Missing server counts never resolve into a file claim.
    assert '"?" if self._server_counts_unknown' in source
