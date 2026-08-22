"""Task 13 (PR E, ACs 42/45 UI halves): the "Re-chunk older-engine items"
control on the Library Search/RAG panel.

Spec §10.0-§10.3: the control sits beside task-12's report line (the
reserved compose position), its worker runs in its OWN worker group behind
a mutual in-flight guard shared with the Settings backfill (NEVER
``exclusive=True`` -- Textual 8.2.8 CANCELS same-group workers, the
task-228 lesson), and the run summary surfaces as ``N re-chunked, M
skipped, K failed`` -- never a bare "done".

The scope service is stubbed at the app boundary but calls the REAL
re-chunk service against a REAL tmp Media DB, so the surfaced summary is
real DB work (§10.5: scratch data only).
"""

from __future__ import annotations

import asyncio
import re

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_rechunk_service import (
    BACKFILL_SLOT,
    RECHUNK_SLOT,
    acquire_bulk_rag_slot,
    bulk_rag_slot_in_flight,
    release_bulk_rag_slot,
    reset_bulk_rag_slots_for_tests,
)
from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
    LibrarySearchRagPanel,
)

REPORT_LINE_ID = "library-rag-legacy-chunk-line"
RECHUNK_BUTTON_ID = "library-rag-rechunk-legacy"
RECHUNK_SUMMARY_ID = "library-rag-rechunk-summary"
REPORT_COPY = "Chunked by an older engine: 1 items"


class _FakeDiagnosticsScopeService:
    """Stands in for the app's ``rag_admin_scope_service``.

    ``get_template_diagnostics`` feeds the report line; ``rechunk_legacy_media``
    is the launch seam (spec §10.4) -- it calls the REAL re-chunk service
    against the real media DB so the surfaced summary is real work.
    """

    def __init__(self, media_db: MediaDatabase | None, *, delay: float = 0.0) -> None:
        self._media_db = media_db
        self._delay = delay
        self.diagnostics_payload: dict = {}
        self.rechunk_calls: list[dict] = []

    async def get_template_diagnostics(self, **kwargs) -> dict:
        return self.diagnostics_payload

    async def rechunk_legacy_media(self, **kwargs) -> dict:
        self.rechunk_calls.append(dict(kwargs))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._media_db is None:
            return {"rechunked": 0, "skipped": 0, "failed": 0}
        from tldw_chatbook.Library.library_rechunk_service import rechunk_legacy_items

        return await rechunk_legacy_items(
            self._media_db,
            rag_service=kwargs.get("rag_service"),
            indexing_db=None,
        )


class _RechunkHost(ConsolidatedCSSApp):
    def __init__(
        self,
        state: LibraryRagPanelState,
        service: _FakeDiagnosticsScopeService | None = None,
        media_db: MediaDatabase | None = None,
    ) -> None:
        super().__init__()
        self._state = state
        self._service = service
        self._media_db = media_db
        self.notices: list[tuple[str, str]] = []

    @property
    def rag_admin_scope_service(self):
        return self._service

    @property
    def media_db(self):
        return self._media_db

    def notify(self, message, **kwargs) -> None:
        self.notices.append((str(message), str(kwargs.get("severity", "info"))))
        super().notify(message, **kwargs)

    def compose(self) -> ComposeResult:
        yield LibrarySearchRagPanel(self._state, id="library-search-rag-panel")


def _panel_state() -> LibraryRagPanelState:
    return LibraryRagPanelState.from_values(
        source_counts={"media": 1},
        query="",
        mode="search",
    )


def _clear_content(db: MediaDatabase, media_id: int) -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    with db.transaction() as conn:
        conn.execute(
            "UPDATE Media SET content = '', last_modified = ?, "
            "version = version + 1 WHERE id = ?",
            (now, media_id),
        )


def _seed_legacy_item(db: MediaDatabase, content: str) -> int:
    media_id, _, _ = db.add_media_with_keywords(
        title="legacy",
        media_type="plaintext",
        content=content,
        keywords=None,
        url=None,
        analysis_content=None,
        author=None,
        transcription_model=None,
        transcription_provenance=None,
        ingestion_date="2026-08-21",
        chunks=[{"text": "OLD legacy chunk one", "start_char": 0, "end_char": 19}],
        chunk_options={"size": 500, "max_size": 500, "overlap": 100},
    )
    return int(media_id)


async def _wait_for(
    predicate,
    *,
    attempts: int = 120,
    interval: float = 0.0,
    what: str = "condition",
):
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}")


async def _wait_for_run_summary(app, *, what: str = "the run's summary line"):
    """Wait for the run to LAND -- not merely for the row to display.

    The summary row is displayed from LAUNCH (it carries the live
    ``"Re-chunking…"`` interim line), so waiting on ``display`` alone
    returns while the worker -- and its guard slot -- is still in flight.
    Completion is the interim line being replaced by the formatted
    counts, which the worker only posts AFTER releasing the slot. Waits
    on real time (5 ms interval): the run happens on a WORKER thread, so
    zero-sleep polling can exhaust its attempts before the worker's
    own sleep/DB work finishes.
    (task-14: the AC-47 focus-outline CSS shifted pause timings enough to
    expose this race in the second-press test; waiting on the interim
    line's replacement makes completion deterministic.)
    """
    summary = app.query_one(f"#{RECHUNK_SUMMARY_ID}", Static)

    def _landed() -> bool:
        return summary.display and str(summary.renderable) != "Re-chunking…"

    await _wait_for(_landed, attempts=400, interval=0.005, what=what)
    return summary


@pytest.fixture(autouse=True)
def _clean_guard_slots():
    reset_bulk_rag_slots_for_tests()
    yield
    reset_bulk_rag_slots_for_tests()


@pytest.mark.asyncio
async def test_control_renders_beside_report_line_and_hides_with_it(tmp_path):
    service = _FakeDiagnosticsScopeService(None)
    app = _RechunkHost(_panel_state(), service, MediaDatabase(tmp_path / "m.db", client_id="t"))
    async with app.run_test() as pilot:
        # Report present -> the pair renders together.
        service.diagnostics_payload = {"legacy_chunk_report": REPORT_COPY}
        app.query_one(f"#{REPORT_LINE_ID}", Static)
        panel = app.query_one(LibrarySearchRagPanel)
        panel._apply_legacy_chunk_report(REPORT_COPY)
        await pilot.pause()
        button = app.query_one(f"#{RECHUNK_BUTTON_ID}", Button)
        assert button.display is True
        # Report empty -> both omit (no action offered on a clean library).
        panel._apply_legacy_chunk_report("")
        await pilot.pause()
        assert app.query_one(f"#{RECHUNK_BUTTON_ID}", Button).display is False


@pytest.mark.asyncio
async def test_press_runs_through_scope_service_and_surfaces_summary(tmp_path):
    db = MediaDatabase(tmp_path / "m.db", client_id="t")
    _seed_legacy_item(db, "alpha beta gamma. " * 30)
    second = _seed_legacy_item(db, "residual")
    _clear_content(db, second)
    service = _FakeDiagnosticsScopeService(db)
    app = _RechunkHost(_panel_state(), service, db)
    async with app.run_test() as pilot:
        service.diagnostics_payload = {"legacy_chunk_report": REPORT_COPY}
        panel = app.query_one(LibrarySearchRagPanel)
        panel._apply_legacy_chunk_report(REPORT_COPY)
        await pilot.pause()

        app.query_one(f"#{RECHUNK_BUTTON_ID}", Button).press()
        await _wait_for(
            lambda: service.rechunk_calls, what="the scope service launch"
        )
        await _wait_for_run_summary(app)
        await pilot.pause()

        # Launched through the scope service (the policy seam, §10.4).
        assert service.rechunk_calls[0].get("mode") == "local"
        # The summary is the real counts, never a bare "done".
        summary = app.query_one(f"#{RECHUNK_SUMMARY_ID}", Static)
        assert "1 re-chunked" in str(summary.renderable)
        assert "1 skipped" in str(summary.renderable)
        assert "0 failed" in str(summary.renderable)
        # The run ended: the guard slot is released and the button re-enabled.
        assert not bulk_rag_slot_in_flight(RECHUNK_SLOT)
        assert app.query_one(f"#{RECHUNK_BUTTON_ID}", Button).disabled is False


@pytest.mark.asyncio
async def test_press_degrades_with_notice_when_scope_service_missing():
    app = _RechunkHost(_panel_state(), service=None, media_db=None)
    async with app.run_test() as pilot:
        service_stub = _FakeDiagnosticsScopeService(None)
        service_stub.diagnostics_payload = {"legacy_chunk_report": REPORT_COPY}
        app._service = service_stub
        panel = app.query_one(LibrarySearchRagPanel)
        panel._apply_legacy_chunk_report(REPORT_COPY)
        await pilot.pause()
        # Now break the seam entirely.
        app._service = None

        app.query_one(f"#{RECHUNK_BUTTON_ID}", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert any("Re-chunk" in message for message, _ in app.notices), (
            f"a missing scope service must surface a notice, got {app.notices}"
        )
        assert not bulk_rag_slot_in_flight(RECHUNK_SLOT)


@pytest.mark.asyncio
async def test_rechunk_refuses_while_backfill_runs_and_kills_nothing():
    """AC 45, re-chunk direction: while a backfill is in flight the press is
    REFUSED with a notice -- no worker starts, and the in-flight backfill
    (represented by its slot + a live app worker) survives uncancelled."""
    service = _FakeDiagnosticsScopeService(None)
    app = _RechunkHost(_panel_state(), service, None)
    async with app.run_test() as pilot:
        service.diagnostics_payload = {"legacy_chunk_report": REPORT_COPY}
        panel = app.query_one(LibrarySearchRagPanel)
        panel._apply_legacy_chunk_report(REPORT_COPY)
        await pilot.pause()

        # Simulate the Settings backfill in flight: its slot is held and a
        # real Textual worker (its group's) is running.
        assert acquire_bulk_rag_slot(BACKFILL_SLOT) is None
        backfill_alive = []

        async def _backfill_body() -> None:
            for _ in range(20):
                backfill_alive.append(True)
                await asyncio.sleep(0)

        backfill_worker = app.run_worker(
            _backfill_body(), group="settings-rag-backfill", exclusive=False
        )

        app.query_one(f"#{RECHUNK_BUTTON_ID}", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert service.rechunk_calls == [], "the re-chunk must be refused, not run"
        assert any("backfill" in m.lower() for m, _ in app.notices), (
            f"the refusal must carry a notice, got {app.notices}"
        )
        assert not bulk_rag_slot_in_flight(RECHUNK_SLOT)
        # Neither cancels the other: the backfill worker ran to completion.
        await backfill_worker.wait()
        assert backfill_worker.state.name == "SUCCESS"
        assert backfill_alive
        release_bulk_rag_slot(BACKFILL_SLOT)


@pytest.mark.asyncio
async def test_second_rechunk_press_refused_first_run_survives():
    """AC 45, same-surface direction: a second press while the re-chunk runs
    is refused with a notice; the RUNNING re-chunk worker is never cancelled
    (separate group + guard, never ``exclusive=True``)."""
    service = _FakeDiagnosticsScopeService(None, delay=0.05)
    app = _RechunkHost(_panel_state(), service, None)
    async with app.run_test() as pilot:
        service.diagnostics_payload = {"legacy_chunk_report": REPORT_COPY}
        panel = app.query_one(LibrarySearchRagPanel)
        panel._apply_legacy_chunk_report(REPORT_COPY)
        await pilot.pause()

        button = app.query_one(f"#{RECHUNK_BUTTON_ID}", Button)
        button.press()
        await _wait_for(lambda: service.rechunk_calls, what="first launch")
        # The interactive refusal: the control is DISABLED while the run is
        # in flight, so a real second press cannot even happen...
        assert button.disabled is True
        # ...and the guard still refuses a programmatic/edge press with a
        # notice (force past the disabled state to prove the guard itself).
        button.disabled = False
        button.press()
        await pilot.pause()
        await pilot.pause()
        assert len(service.rechunk_calls) == 1, "the second press must not relaunch"
        assert any("already" in m for m, _ in app.notices), (
            f"expected an already-running notice, got {app.notices}"
        )

        # The first run SURVIVES the second press and completes.
        await _wait_for_run_summary(
            app, what="the surviving first run's summary"
        )
        assert not bulk_rag_slot_in_flight(RECHUNK_SLOT)


@pytest.mark.asyncio
async def test_backfill_trigger_refuses_while_rechunk_runs(monkeypatch):
    """AC 45, vice versa: the Settings backfill trigger refuses (with a
    notice) while a re-chunk is in flight -- and never cancels it."""
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    assert acquire_bulk_rag_slot(RECHUNK_SLOT) is None

    screen = SettingsScreen.__new__(SettingsScreen)
    worker_calls: list[bool] = []

    class _NotifyApp:
        def __init__(self) -> None:
            self.notices: list[tuple[str, str]] = []

        def notify(self, message, **kwargs) -> None:
            self.notices.append(
                (str(message), str(kwargs.get("severity", "information")))
            )

    fake_app = _NotifyApp()
    # ``SettingsScreen.app`` is a class-level property -- patch it the same
    # way Tests/UI/test_settings_rag_profile_region.py's ``fake_app`` does.
    monkeypatch.setattr(
        SettingsScreen, "app", property(lambda self: fake_app), raising=False
    )
    screen._rag_backfill_worker = lambda: worker_calls.append(True)  # type: ignore[method-assign]

    screen._trigger_library_rag_index_backfill()

    assert worker_calls == [], "the backfill must be refused, not started"
    assert any("re-chunk" in m for m, _ in fake_app.notices), (
        f"expected a mutual-refusal notice, got {fake_app.notices}"
    )
    assert bulk_rag_slot_in_flight(RECHUNK_SLOT), "the re-chunk must be untouched"

    # And with nothing in flight, the same trigger starts the worker.
    release_bulk_rag_slot(RECHUNK_SLOT)
    screen._trigger_library_rag_index_backfill()
    assert worker_calls == [True]
    release_bulk_rag_slot(BACKFILL_SLOT)


# --- task-14 / spec AC 47: the control's design-token state contract -----

def _rule_bodies(bundle_text: str, selector: str) -> list[str]:
    """Minimal CSS block reader (the guard-test idiom, e.g.
    test_non_obscuring_focus_contract.css_blocks)."""
    uncommented = re.sub(r"/\*.*?\*/", "", bundle_text, flags=re.DOTALL)
    bodies = []
    for match in re.finditer(r"\{(?P<body>[^{}]*)\}", uncommented):
        prefix = uncommented[: match.start()]
        start = max(prefix.rfind("}"), prefix.rfind(";")) + 1
        if selector in [item.strip() for item in prefix[start:].split(",")]:
            bodies.append(match.group("body"))
    return bodies


def test_rechunk_control_class_defines_all_states_with_ds_tokens():
    """AC 47: `.library-rag-recovery-action` (both the re-chunk button and
    the pre-existing Open-Import button) defines rest/hover/focus/disabled
    from ``$ds-*`` design tokens with no raw hex. The button is DISABLED
    for a whole re-chunk batch, so its disabled legibility matters (the
    Legible Disabled / TASK-1801 escape). Checked against the BUILT bundle
    so a missing ``build_css.py`` run fails here too."""
    from pathlib import Path

    bundle = Path(__file__).resolve().parents[2] / (
        "tldw_chatbook/css/tldw_cli_modular.tcss"
    )
    text = bundle.read_text(encoding="utf-8")

    rules = {
        "rest": _rule_bodies(text, ".library-rag-recovery-action"),
        "hover": _rule_bodies(text, ".library-rag-recovery-action:hover"),
        "focus": _rule_bodies(text, "Button.library-rag-recovery-action:focus"),
        "disabled": _rule_bodies(
            text, "Button.library-rag-recovery-action:disabled"
        ),
    }
    for state, bodies in rules.items():
        assert bodies, f"missing {state} rule for .library-rag-recovery-action"

    for state, bodies in rules.items():
        for body in bodies:
            assert "$ds-" in body, f"{state} rule must use $ds-* tokens: {body!r}"
            assert not re.search(r"#[0-9a-fA-F]{3,8}\b", body), (
                f"{state} rule must carry no raw hex: {body!r}"
            )

    # The disabled legibility escape must neutralise the generic
    # `Button:disabled { opacity: 50% }` dimmer (task-4023 RC-07 recipe).
    disabled = rules["disabled"][0]
    assert "opacity: 100%" in disabled
    assert "$ds-text-muted" in disabled or "$ds-text-primary" in disabled


def test_rechunk_summary_and_report_lines_use_the_styled_quiet_line_class():
    """AC 47 for the two Static controls: they compose the existing,
    bundle-styled `.library-rag-quiet-line` class (rest-only is the
    complete state set for a non-focusable Static) and introduce no new
    unstyled class token."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / (
        "tldw_chatbook/Widgets/Library/library_search_rag_panel.py"
    )
    text = source.read_text(encoding="utf-8")
    quiet = re.findall(r'classes="([^"{}]+)"', text)
    assert "library-rag-quiet-line" in quiet
    # Every class token composed by the panel is styled in the bundle.
    bundle = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/css/tldw_cli_modular.tcss"
    ).read_text(encoding="utf-8")
    for attr in quiet:
        for token in attr.split():
            assert f".{token}" in bundle or f"#{token}" in bundle, (
                f"panel composes unstyled class token {token!r}"
            )
