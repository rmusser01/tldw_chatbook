"""Dreams story detail modal (Dreams Phase 1, Task 7).

Two harnesses, mirroring the stream's own split: the modal's actions are
exercised on a bare ``App`` (no destination shell needed -- the modal owns
its DB handle via the ``dreams_db_getter`` seam and its handoff via the
``open_chat_with_handoff`` app attribute), while the Artifacts-screen
wiring (row click / Enter opens the modal, refresh after actions) reuses
the Task-6 ``DestinationHarness`` harness from
``test_artifacts_dreams_rows.py``.
"""

from __future__ import annotations

import pytest
from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text
from textual.app import App
from textual.widgets import Button, Static

import tldw_chatbook.UI.Screens.artifacts_dreams_modal as dsm_module
from Tests.Dreams.test_ingest_action import FakeCaptureBackend
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.Dreams.dreams_view import list_recent_dreams
from tldw_chatbook.Library.collections_capture_models import CollectionsCaptureError
from tldw_chatbook.UI.Screens.artifacts_dreams_modal import DreamsStoryModal
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen

pytestmark = pytest.mark.ui

_STORY_TITLE = "Cheap flights to Japan"
_STORY_BODY = "A story about fares."


# --- Shared helpers ---------------------------------------------------------


def _enable_dreams(monkeypatch) -> None:
    """Flip ``[dreams] enabled`` deterministically (Task 6's seam)."""
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default: (
            True if key == "enabled" else 3 if key == "queries_per_cycle" else default
        ),
    )


def _seed_db(tmp_path) -> DreamsDB:
    db = DreamsDB(tmp_path / "dreams.sqlite", "dreams-modal")
    collection = db.create_collection("2026-09-22", "scheduled", "digest")
    db.insert_story(
        collection,
        title=_STORY_TITLE,
        url="https://example.com/flights",
        snippet="Fares from $89",
        body=_STORY_BODY,
        status="complete",
        source="web",
        kind="deal",
        event_date=None,
        location="Japan",
        matched_topics=["visit japan"],
        query="cheap flights japan",
    )
    db.upsert_profile_entry(
        "topic", "visit japan", weight=0.9, searchable=1, source="seed"
    )
    return db


def _seed_failed_cycle(db: DreamsDB) -> None:
    collection = db.create_collection("2026-09-23", "scheduled", "digest")
    db.set_collection_status(collection, "failed")


def _story_row(db: DreamsDB) -> dict:
    rows = list_recent_dreams(db, limit=10)
    return next(row for row in rows if not row.get("synthetic"))


def _synthetic_row(db: DreamsDB) -> dict:
    rows = list_recent_dreams(db, limit=10)
    return next(row for row in rows if row.get("synthetic"))


def _feedback_kinds(db: DreamsDB, story_id: int) -> list[str]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT kind FROM dream_feedback WHERE story_id = ? ORDER BY id",
            (story_id,),
        ).fetchall()
    return [row[0] for row in rows]


def _story_kept(db: DreamsDB, story_id: int) -> int:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT kept FROM dream_stories WHERE id = ?", (story_id,)
        ).fetchone()
    assert row is not None
    return int(row[0])


def _renderable_text(renderable) -> str:
    if isinstance(renderable, Text):
        return renderable.plain
    if isinstance(renderable, Group):
        return "\n".join(_renderable_text(item) for item in renderable.renderables)
    if isinstance(renderable, Markdown):
        return str(renderable.markup)
    return str(renderable)


def _visible_text(widget) -> str:
    return "\n".join(
        _renderable_text(item.renderable)
        for item in widget.query(Static)
        if item.display and hasattr(item, "renderable")
    )


def _button_labels(widget) -> str:
    return " ".join(
        str(button.label)
        for button in widget.query(Button)
        if button.display and button.label is not None
    )


class _ModalApp(App):
    """Bare host app whose ``open_chat_with_handoff`` captures payloads."""

    def __init__(self) -> None:
        super().__init__()
        self.staged: list[object] = []

    def open_chat_with_handoff(self, payload) -> None:  # skills_screen idiom stub
        self.staged.append(payload)


# --- Modal actions ----------------------------------------------------------


@pytest.mark.asyncio
async def test_keep_toggle_writes_kept_feedback_and_refreshes(tmp_path):
    db = _seed_db(tmp_path)
    story = _story_row(db)
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: changed.append(1),
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("k")
        await pilot.pause()

        assert _story_kept(db, story["id"]) == 1, "k must set kept via the real DB"
        assert _feedback_kinds(db, story["id"]) == ["kept"]
        assert changed == [1], "on_changed must fire exactly once per action"
        assert app.screen is modal, "keep keeps the modal open"
        assert "Unkeep" in _button_labels(modal), "button label must flip"

        await pilot.press("k")  # toggle back off
        await pilot.pause()
        assert _story_kept(db, story["id"]) == 0
        assert _feedback_kinds(db, story["id"]) == ["kept", "kept"]
        assert len(changed) == 2, "each mutating action fires on_changed once"


@pytest.mark.asyncio
async def test_more_records_feedback_and_dismisses(tmp_path):
    db = _seed_db(tmp_path)
    story = _story_row(db)
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: changed.append(1),
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("m")
        await pilot.pause()

        assert _feedback_kinds(db, story["id"]) == ["more"]
        assert changed == [1]
        assert app.screen is not modal, "more/less dismiss the modal"


@pytest.mark.asyncio
async def test_less_records_feedback_and_dismisses(tmp_path):
    db = _seed_db(tmp_path)
    story = _story_row(db)
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: None,
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("l")
        await pilot.pause()

        assert _feedback_kinds(db, story["id"]) == ["less"]
        assert app.screen is not modal


@pytest.mark.asyncio
async def test_close_dismisses_modal(tmp_path):
    db = _seed_db(tmp_path)
    story = _story_row(db)
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: None,
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("q")
        await pilot.pause()

        assert app.screen is not modal
        assert _feedback_kinds(db, story["id"]) == [], "close records nothing"


@pytest.mark.asyncio
async def test_dive_stages_handoff_payload_and_records_feedback(tmp_path):
    db = _seed_db(tmp_path)
    story = _story_row(db)
    app = _ModalApp()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: None,
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("d")
        await pilot.pause()

        assert len(app.staged) == 1
        payload = app.staged[0]
        assert payload.source == "dreams"
        assert payload.item_type == "dream_story"
        assert payload.title == _STORY_TITLE
        assert payload.body == _STORY_BODY
        assert payload.suggested_prompt.strip(), "dive must carry a prompt"
        assert _feedback_kinds(db, story["id"]) == ["dived"]
        assert app.screen is not modal, "dive dismisses the modal"


@pytest.mark.asyncio
async def test_export_writes_markdown_stub_and_records_feedback(tmp_path, monkeypatch):
    db = _seed_db(tmp_path)
    story = _story_row(db)
    sink = tmp_path / "sink"
    monkeypatch.setattr(dsm_module, "_dreams_export_dir", lambda: sink)
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: None,
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("e")
        await pilot.pause()

        files = sorted(sink.glob("*.md"))
        assert len(files) == 1, f"exactly one export file, got {[f.name for f in files]}"
        assert files[0].name == "cheap-flights-to-japan-1.md"
        text = files[0].read_text(encoding="utf-8")
        assert _STORY_TITLE in text
        assert "https://example.com/flights" in text
        assert "visit japan" in text, "provenance must carry matched topics"
        assert "cheap flights japan" in text, "provenance must carry the query"
        assert _STORY_BODY in text
        assert _feedback_kinds(db, story["id"]) == ["exported"]
        assert app.screen is modal, "export keeps the modal open"


@pytest.mark.asyncio
async def test_ingest_submits_story_url_and_records_feedback(tmp_path):
    db = _seed_db(tmp_path)
    story = _story_row(db)
    backend = FakeCaptureBackend()
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: backend,
            on_changed=lambda: changed.append(1),
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("i")
        await pilot.pause()

        (request,) = backend.requests
        assert request.submitted_url == "https://example.com/flights"
        assert request.title == _STORY_TITLE
        assert request.freeform_note.startswith("via Dreams"), (
            "the capture note must attribute the item to Dreams"
        )
        assert _feedback_kinds(db, story["id"]) == ["ingested"]
        assert changed == [1], "on_changed fires exactly once after success"
        assert app.screen is modal, "ingest keeps the modal open"


@pytest.mark.asyncio
async def test_ingest_failure_is_a_notice_not_a_crash(tmp_path):
    db = _seed_db(tmp_path)
    story = _story_row(db)
    backend = FakeCaptureBackend(
        error=CollectionsCaptureError("capture_queue_full")
    )
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: backend,
            on_changed=lambda: changed.append(1),
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("i")
        await pilot.pause()

        assert backend.requests == []
        assert _feedback_kinds(db, story["id"]) == [], (
            "a failed capture records no ingested feedback"
        )
        assert changed == [], "on_changed must not fire for a failure"
        assert app.screen is modal, "a failed ingest must not dismiss or crash"


@pytest.mark.asyncio
async def test_ingest_without_capture_backend_degrades_to_notice(tmp_path):
    db = _seed_db(tmp_path)
    story = _story_row(db)
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: None,
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("i")
        await pilot.pause()

        assert _feedback_kinds(db, story["id"]) == []
        assert app.screen is modal, "a refused action must not dismiss or crash"


def _seed_llm_story_db(tmp_path) -> DreamsDB:
    """One llm-source story: its URL is the synthetic dreams:// scheme."""
    db = DreamsDB(tmp_path / "dreams-llm.sqlite", "dreams-modal")
    collection = db.create_collection("2026-09-22", "scheduled", "digest")
    db.insert_story(
        collection,
        title="LLM knowledge story",
        url="dreams://llm/0/rust%20tui",
        snippet="",
        body="A story straight from model knowledge.",
        status="complete",
        source="llm",
        kind="content",
        event_date=None,
        location=None,
        matched_topics=["rust tui"],
        query="rust tui",
    )
    db.upsert_profile_entry(
        "topic", "rust tui", weight=0.9, searchable=1, source="seed"
    )
    return db


@pytest.mark.asyncio
async def test_llm_story_does_not_offer_ingest_anywhere(tmp_path):
    db = _seed_llm_story_db(tmp_path)
    story = _story_row(db)
    backend = FakeCaptureBackend()
    changed: list[int] = []
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: backend,
            on_changed=lambda: changed.append(1),
        )
        await app.push_screen(modal)
        await pilot.pause()

        labels = _button_labels(modal)
        assert "Ingest" not in labels, "dreams:// rows must not offer Ingest"
        for word in ("Keep", "Dive deeper", "Export", "More like this",
                     "Less like this", "Close"):
            assert word in labels, f"http-only gate must not hide {word!r}"
        hints = _renderable_text(
            modal.query_one("#dsm-hints", Static).renderable)
        assert "Ingest" not in hints, "hints must not advertise a gated action"

        # The keyboard binding is guarded too: pressing i writes nothing.
        await pilot.press("i")
        await pilot.pause()
        assert backend.requests == []
        assert _feedback_kinds(db, story["id"]) == []
        assert changed == []
        assert app.screen is modal, "a refused ingest must not dismiss"


@pytest.mark.asyncio
async def test_http_story_still_offers_ingest(tmp_path):
    db = _seed_db(tmp_path)
    story = _story_row(db)
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: FakeCaptureBackend(),
            on_changed=lambda: None,
        )
        await app.push_screen(modal)
        await pilot.pause()

        assert "Ingest" in _button_labels(modal)
        hints = _renderable_text(
            modal.query_one("#dsm-hints", Static).renderable)
        assert "i Ingest" in hints


# --- Modal surface ----------------------------------------------------------


@pytest.mark.asyncio
async def test_footer_hints_advertise_exactly_the_seven_actions(tmp_path):
    db = _seed_db(tmp_path)
    story = _story_row(db)
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: None,
        )
        await app.push_screen(modal)
        await pilot.pause()

        hints = _renderable_text(modal.query_one("#dsm-hints", Static).renderable)
        for word in (
            "Keep",
            "Dive deeper",
            "Export",
            "Ingest",
            "More like this",
            "Less like this",
            "Close",
        ):
            assert word in hints, f"hint must advertise {word!r}"
        for word in ("Track", "Delete", "Cast", "Regenerate", "Share"):
            assert word not in hints, f"hint must not advertise {word!r}"


@pytest.mark.asyncio
async def test_story_detail_renders_provenance_and_query_preview(tmp_path, monkeypatch):
    _enable_dreams(monkeypatch)
    db = _seed_db(tmp_path)
    story = _story_row(db)
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: None,
            on_changed=lambda: None,
        )
        await app.push_screen(modal)
        await pilot.pause()

        text = _visible_text(modal)
        assert _STORY_TITLE in text
        assert "https://example.com/flights" in text
        assert "visit japan" in text
        assert "cheap flights japan" in text
        assert _STORY_BODY in text
        assert "preview (fallback queries until next cycle)" in text
        assert "visit japan recent developments" in text
        assert "surprising adjacent to visit japan" in text


@pytest.mark.asyncio
async def test_synthetic_failed_cycle_row_is_close_only_status_view(tmp_path):
    db = _seed_db(tmp_path)
    _seed_failed_cycle(db)
    synthetic = _synthetic_row(db)
    backend = FakeCaptureBackend()
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            synthetic,
            dreams_db_getter=lambda: db,
            capture_backend_getter=lambda: backend,
            on_changed=lambda: None,
        )
        await app.push_screen(modal)
        await pilot.pause()

        text = _visible_text(modal)
        assert "Cycle 2026-09-23" in text
        assert "failed" in text

        labels = _button_labels(modal)
        assert "Close" in labels
        for word in (
            "Keep",
            "Dive deeper",
            "Export",
            "Ingest",
            "More like this",
            "Less like this",
        ):
            assert word not in labels, f"synthetic row must not offer {word!r}"

        hints = _renderable_text(modal.query_one("#dsm-hints", Static).renderable)
        for word in (
            "Keep",
            "Dive deeper",
            "Export",
            "Ingest",
            "More like this",
            "Less like this",
        ):
            assert word not in hints

        await pilot.press("k")  # inert on a synthetic row
        await pilot.press("i")  # ingest is equally inert
        await pilot.pause()
        with db.connection() as conn:
            feedback_rows = conn.execute("SELECT COUNT(*) FROM dream_feedback").fetchone()
        assert int(feedback_rows[0]) == 0, "no action may write for a synthetic row"
        assert backend.requests == [], "no action may capture for a synthetic row"
        assert app.screen is modal

        await pilot.press("q")
        await pilot.pause()
        assert app.screen is not modal


@pytest.mark.asyncio
async def test_missing_dreams_db_degrades_to_notice_without_crash(tmp_path):
    story = _story_row(_seed_db(tmp_path))
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = DreamsStoryModal(
            story,
            dreams_db_getter=lambda: None,
            capture_backend_getter=lambda: None,
            on_changed=lambda: None,
        )
        await app.push_screen(modal)
        await pilot.pause()

        await pilot.press("k")
        await pilot.pause()
        assert app.screen is modal, "a refused action must not dismiss or crash"


# --- Artifacts-screen wiring ------------------------------------------------


async def _wait_for_dreams(screen, pilot, selector: str, *, attempts: int = 50):
    for _ in range(attempts):
        await pilot.pause(0.05)
        if screen._dreams and screen.query(selector):
            return
    raise AssertionError(f"dreams refresh never landed {selector!r}")


# The four wiring tests below build a real app via ``_build_test_app``,
# whose ``build_test_app_config`` calls the real ``load_settings()`` (a
# guarded config read). Under the default per-test TLDW_CONFIG_PATH redirect
# the config participant admitted at collection time no longer matches and
# admission fails closed (RecoveryRequired("raw_source_selection_changed")).
# ``bootstrap_profile`` (Tests/conftest.py, TASK-32873) keeps the
# collection-time profile so the bound selection still matches. The bare-App
# modal tests above never reach a guarded config read and need no marker.
@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_dream_row_click_opens_modal_and_actions_refresh_rows(
    tmp_path, monkeypatch
):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_db(tmp_path)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-row-1")

        await pilot.click("#artifacts-dream-row-1")
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, DreamsStoryModal)

        await pilot.press("k")
        await pilot.pause()
        assert _story_kept(app.dreams_db, 1) == 1

        await pilot.press("q")
        await pilot.pause()
        assert host.screen_stack[-1] is screen, "close returns to the Artifacts screen"

        # on_changed refreshed the screen's rows: the kept badge is painted.
        for _ in range(50):
            await pilot.pause(0.05)
            row = screen.query_one("#artifacts-dream-row-1", Static)
            if "kept" in _renderable_text(row.renderable):
                break
        assert "kept" in _renderable_text(
            screen.query_one("#artifacts-dream-row-1", Static).renderable
        )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_dream_row_enter_opens_modal(tmp_path, monkeypatch):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_db(tmp_path)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-row-1")

        screen.query_one("#artifacts-dream-row-1", Static).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(host.screen_stack[-1], DreamsStoryModal)


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_synthetic_row_click_opens_status_view(tmp_path, monkeypatch):
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_db(tmp_path)
    _seed_failed_cycle(app.dreams_db)
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-row-cycle-2026-09-23")

        await pilot.click("#artifacts-dream-row-cycle-2026-09-23")
        await pilot.pause()

        modal = host.screen_stack[-1]
        assert isinstance(modal, DreamsStoryModal)
        assert "Cycle 2026-09-23" in _visible_text(modal)


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_dream_row_ingest_uses_app_capture_service(tmp_path, monkeypatch):
    """The screen wires the modal's backend to the app's capture service."""
    _enable_dreams(monkeypatch)
    app = _build_test_app(configured_default="artifacts")
    app.dreams_db = _seed_db(tmp_path)
    backend = FakeCaptureBackend()
    app.local_collections_capture_service = backend
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)
        await _wait_for_dreams(screen, pilot, "#artifacts-dream-row-1")

        await pilot.click("#artifacts-dream-row-1")
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, DreamsStoryModal)

        await pilot.press("i")
        await pilot.pause()

        (request,) = backend.requests
        assert request.submitted_url == "https://example.com/flights"
        assert request.freeform_note.startswith("via Dreams")
        assert _feedback_kinds(app.dreams_db, 1) == ["ingested"]
