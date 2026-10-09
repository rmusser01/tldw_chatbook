"""The Library Reports empty-state CTA asks before the Daily Report demo
seeds anything (TASK-34000.23, review finding L-10).

Real app, production stylesheet (`_CssTrueDestinationHarness`), the test
app's own file-backed Subscriptions DB, one boot per test. Exactly one seam
is faked: `app.daily_report_demo_service`, a recording stub whose
`run_demo_detached` returns a done task the way the real service does
(`Tests/UI/test_artifacts_screen_reports.py` set the shape). The persisted
provider/model pair is layered over the REAL loaded config the same way
Task .21 did for RAG Answer, so the copy names what the brief would bill.

Data claims are asserted against the DB (row counts through
`app.subscriptions_db.transaction()`), never against widget state alone.

The 120x36 arms of the two size-parametrized tests live in the `_extended`
sibling, outside the PR-gate lane (this file stays under its 35 s budget).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _CssTrueDestinationHarness
from tldw_chatbook import config as app_config
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Subscriptions import briefing_service
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: Everything the demo's seed and first run write (the harness's
#: `subs_demo_state.py --report` prints the same tables).
_DEMO_TABLES = (
    "watchlists",
    "watchlist_sources",
    "subscriptions",
    "subscription_items",
    "local_watchlist_runs",
    "briefings",
    "briefing_presets",
)
_ZERO = {table: 0 for table in _DEMO_TABLES}


class _StubDemo:
    """Records starts; returns a done task like `run_demo_detached` does."""

    def __init__(self) -> None:
        self.started = 0

    def run_demo_detached(self):
        self.started += 1

        async def _noop():
            return {"status": "ok"}

        return asyncio.create_task(_noop())


def _persist_pair(monkeypatch, *, provider: str = "OpenAI", model: str = "gpt-4.1-mini"):
    """Layer `[chat_defaults]` over the real persisted config read
    `resolve_persisted_briefing_defaults` performs (a call-time disk read)."""
    real_load = app_config.load_cli_config_and_ensure_existence

    def _load_with_pair(*args, **kwargs):
        settings = dict(real_load(*args, **kwargs))
        chat_defaults = dict(settings.get("chat_defaults") or {})
        chat_defaults["provider"] = provider
        chat_defaults["model"] = model
        settings["chat_defaults"] = chat_defaults
        return settings

    monkeypatch.setattr(
        app_config, "load_cli_config_and_ensure_existence", _load_with_pair
    )


def _counts(app) -> dict[str, int]:
    with app.subscriptions_db.transaction() as conn:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in _DEMO_TABLES
        }


def _painted_flat(screen, region) -> str:
    """The painted text inside `region`, rows joined and whitespace
    collapsed, so a sentence that wraps at a space still matches."""
    strips = list(screen._compositor.render_strips())
    rows = [
        strips[y].crop(region.x, region.right).text.rstrip()
        for y in range(region.y, region.bottom)
    ]
    return " ".join(" ".join(rows).split())


@asynccontextmanager
async def _empty_reports(tmp_path, *, size=(160, 45), seed=None):
    """Boot Library ▸ Reports. `seed(app)` runs against the real
    Subscriptions DB before the screen mounts (a failed first run, a
    user-built preset); without it the page is empty."""
    app = _build_test_app(configured_default="library")
    db = CharactersRAGDB(tmp_path / "library.sqlite", client_id="demo-consent")
    app.chachanotes_db = db
    stub = _StubDemo()
    app.daily_report_demo_service = stub
    if seed is not None:
        seed(app)
    host = _CssTrueDestinationHarness(app, "library")
    try:
        async with host.run_test(size=size) as pilot:
            screen = host.screen
            await screen._select_library_rail_row("artifacts-reports")
            controller = None
            for _ in range(100):
                await pilot.pause(0.03)
                controller = getattr(screen, "_artifacts_controller", None)
                if controller and controller.page is not None and not controller.loading:
                    break
            assert controller is not None and controller.page is not None, (
                "Reports never finished loading"
            )
            if seed is None:
                assert not controller.page.items
            await pilot.pause()
            yield app, screen, pilot, stub
    finally:
        db.close_connection()


def _seed_daily_brief_schedule(app, *, failed_run: bool = False, preset=None) -> int:
    """A Daily Brief watchlist with a 24 h cadence -- what one demo run
    leaves behind -- optionally with a failed briefing row (the state the
    failure toast's "run the demo again" is read in) and/or a user-built
    default preset carrying its own provider/model."""
    db = app.subscriptions_db
    watchlist_id = int(WatchlistBundleService(db).create("Daily Brief")["id"])
    settings = {"briefing_cadence_seconds": 86_400}
    if preset is not None:
        settings["default_preset_id"] = db.insert_briefing_preset(
            "Daily Brief",
            roster_json='[{"name": "Host", "voice_profile_id": null}]',
            provider=preset[0],
            model=preset[1],
        )
    db.set_watchlist_briefing_settings(watchlist_id, **settings)
    if failed_run:
        briefing_id = db.insert_briefing(watchlist_id)
        db.update_briefing(briefing_id, status="failed", error="provider refused")
    return watchlist_id


async def _settle(pilot) -> None:
    for _ in range(3):
        await pilot.pause()


@pytest.mark.parametrize("size", [(160, 45)])
async def test_pressing_the_cta_without_confirming_writes_nothing(
    tmp_path, monkeypatch, size
):
    """AC#2, AC#5: one press shows the consequences and starts nothing; the
    Subscriptions DB stays empty; Cancel hides the block, still empty, and
    hands focus back to the CTA."""
    _persist_pair(monkeypatch)
    async with _empty_reports(tmp_path, size=size) as (app, screen, pilot, stub):
        assert _counts(app) == _ZERO
        screen.query_one("#library-artifacts-demo", Button).press()
        await _settle(pilot)
        assert stub.started == 0, "the Library CTA started the demo without asking"
        assert _counts(app) == _ZERO

        block = screen.query_one("#library-artifacts-demo-confirm")
        assert block.display and block.region.height > 0
        text = _painted_flat(
            screen, screen.query_one("#library-artifacts-demo-copy").region
        )
        for needle in (
            "Hacker News (front page)",
            "BBC World News",
            "Ars Technica",
            "hourly",
            "every 24 h",
            "openai · gpt-4.1-mini",
            "API quota",
            "Watchlists",
        ):
            assert needle in text, (needle, text)
        assert screen.focused is screen.query_one("#library-artifacts-demo-cancel")

        screen.query_one("#library-artifacts-demo-cancel", Button).press()
        await _settle(pilot)
        assert not screen.query_one("#library-artifacts-demo-confirm").display
        assert stub.started == 0
        assert _counts(app) == _ZERO
        assert screen.focused is screen.query_one("#library-artifacts-demo")


async def test_confirm_runs_the_detached_demo_once(tmp_path, monkeypatch):
    """AC#2's Confirm half: the run button is the one way in, and it goes
    through `run_demo_detached` exactly once; the block closes."""
    _persist_pair(monkeypatch)
    async with _empty_reports(tmp_path) as (app, screen, pilot, stub):
        screen.query_one("#library-artifacts-demo", Button).press()
        await _settle(pilot)
        assert stub.started == 0
        screen.query_one("#library-artifacts-demo-confirm-run", Button).press()
        await _settle(pilot)
        assert stub.started == 1
        assert not screen.query_one("#library-artifacts-demo-confirm").display


@pytest.mark.parametrize("size", [(160, 45)])
async def test_cta_sits_under_the_empty_state_sentence(tmp_path, monkeypatch, size):
    """AC#1, AC#3: the CTA is painted directly under the sentence that
    mentions it, inside the same (Items) pane, with the recurring label."""
    _persist_pair(monkeypatch)
    async with _empty_reports(tmp_path, size=size) as (app, screen, pilot, stub):
        count = screen.query_one("#library-artifacts-count", Static)
        cta = screen.query_one("#library-artifacts-demo", Button)
        items = screen.query_one("#library-canvas")
        assert count.region.height > 0 and cta.region.height > 0
        assert "No reports yet" in _painted_flat(screen, count.region)
        assert 1 <= cta.region.y - count.region.y <= 3, (cta.region, count.region)
        assert items.region.contains_region(count.region)
        assert items.region.contains_region(cta.region)
        assert "Set up a daily brief…" in _painted_flat(screen, cta.region)
        assert "Try report demo" not in _painted_flat(screen, items.region)


async def test_no_provider_disables_the_run_button_with_the_reason(
    tmp_path, monkeypatch
):
    """Review Focus 1: with no persisted provider the copy names Settings,
    the run button is disabled, and pressing it starts nothing."""

    def _no_provider():
        raise RuntimeError("Persisted briefing provider/model is unavailable.")

    monkeypatch.setattr(
        briefing_service, "resolve_persisted_briefing_defaults", _no_provider
    )
    async with _empty_reports(tmp_path) as (app, screen, pilot, stub):
        screen.query_one("#library-artifacts-demo", Button).press()
        await _settle(pilot)
        block = screen.query_one("#library-artifacts-demo-confirm")
        assert block.display
        text = _painted_flat(
            screen, screen.query_one("#library-artifacts-demo-copy").region
        )
        assert "No LLM provider is set up" in text
        assert "Settings (F4)" in text
        run = screen.query_one("#library-artifacts-demo-confirm-run", Button)
        assert run.disabled
        assert "Settings (F4)" in str(run.tooltip)
        run.press()
        await _settle(pilot)
        assert stub.started == 0
        assert _counts(app) == _ZERO


async def test_existing_schedule_discloses_reuse_not_seeding(tmp_path, monkeypatch):
    """A Daily Brief schedule already on disk: the copy says it will reuse
    it (one call now), and the run button says so."""
    _persist_pair(monkeypatch)
    async with _empty_reports(tmp_path) as (app, screen, pilot, stub):
        db = app.subscriptions_db
        watchlist_id = int(WatchlistBundleService(db).create("Daily Brief")["id"])
        db.set_watchlist_briefing_settings(
            watchlist_id, briefing_cadence_seconds=86_400
        )
        assert len(db.list_briefing_schedules()) == 1

        screen.query_one("#library-artifacts-demo", Button).press()
        await _settle(pilot)
        block = screen.query_one("#library-artifacts-demo-confirm")
        assert block.display
        text = _painted_flat(
            screen, screen.query_one("#library-artifacts-demo-copy").region
        )
        assert text.startswith("You already have a Daily Brief.")
        assert "openai · gpt-4.1-mini" in text
        assert "API quota" in text and "one call" not in text
        assert "cast script" in text and "TTS provider" in text
        run = screen.query_one("#library-artifacts-demo-confirm-run", Button)
        assert not run.disabled
        assert "Write today's brief" in _painted_flat(screen, run.region)
        assert stub.started == 0


async def test_failed_first_run_keeps_a_retry_through_the_same_consent(
    tmp_path, monkeypatch
):
    """Review 1 #2: a failed first run leaves a `failed` row (page not empty)
    and the failure toast says "run the demo again" -- the Library must still
    offer it, through the SAME consent, with the reuse copy; nothing starts
    until Confirm, which starts exactly one detached run."""
    _persist_pair(monkeypatch)
    async with _empty_reports(
        tmp_path, seed=lambda app: _seed_daily_brief_schedule(app, failed_run=True)
    ) as (app, screen, pilot, stub):
        assert [row.status for row in screen._artifacts_controller.page.items] == [
            "failed"
        ]
        actions = screen.query_one("#library-artifacts-empty-actions")
        assert actions.display and actions.region.height > 0
        cta = screen.query_one("#library-artifacts-demo", Button)
        assert "Set up a daily brief…" in _painted_flat(screen, cta.region)
        before = _counts(app)
        cta.press()
        await _settle(pilot)
        assert stub.started == 0
        assert _counts(app) == before
        text = _painted_flat(
            screen, screen.query_one("#library-artifacts-demo-copy").region
        )
        assert text.startswith("You already have a Daily Brief.")
        assert "openai · gpt-4.1-mini" in text
        run = screen.query_one("#library-artifacts-demo-confirm-run", Button)
        assert "Write today's brief" in _painted_flat(screen, run.region)
        run.press()
        await _settle(pilot)
        assert stub.started == 1
        assert not screen.query_one("#library-artifacts-demo-confirm").display


async def test_escape_and_leaving_the_view_dismiss_without_writing(
    tmp_path, monkeypatch
):
    """Review 1 minor 2: Escape closes the block; switching the rail with
    the block open leaves it closed on return. Both: DB untouched, nothing
    started."""
    _persist_pair(monkeypatch)
    async with _empty_reports(tmp_path) as (app, screen, pilot, stub):
        screen.query_one("#library-artifacts-demo", Button).press()
        await _settle(pilot)
        assert screen.query_one("#library-artifacts-demo-confirm").display
        assert screen.focused is screen.query_one("#library-artifacts-demo-cancel")
        await pilot.press("escape")
        await _settle(pilot)
        assert not screen.query_one("#library-artifacts-demo-confirm").display
        assert screen.focused is screen.query_one("#library-artifacts-demo")
        assert stub.started == 0 and _counts(app) == _ZERO

        screen.query_one("#library-artifacts-demo", Button).press()
        await _settle(pilot)
        assert screen.query_one("#library-artifacts-demo-confirm").display
        await screen._select_library_rail_row("artifacts-chatbooks")
        await _settle(pilot)
        assert screen._artifacts_controller.demo_consent is None
        await screen._select_library_rail_row("artifacts-reports")
        for _ in range(100):
            await pilot.pause(0.03)
            controller = screen._artifacts_controller
            if controller.page is not None and not controller.loading:
                break
        await _settle(pilot)
        assert not screen.query_one("#library-artifacts-demo-confirm").display
        assert stub.started == 0 and _counts(app) == _ZERO


async def test_reused_schedule_names_its_presets_own_pair(tmp_path, monkeypatch):
    """Review 1 minor 1: a reused schedule bills its default preset's own
    provider/model when the preset carries them, so the copy names THAT
    pair, not the persisted one."""
    _persist_pair(monkeypatch)
    async with _empty_reports(
        tmp_path,
        seed=lambda app: _seed_daily_brief_schedule(
            app, preset=("anthropic", "claude-haiku-4-5")
        ),
    ) as (app, screen, pilot, stub):
        screen.query_one("#library-artifacts-demo", Button).press()
        await _settle(pilot)
        text = _painted_flat(
            screen, screen.query_one("#library-artifacts-demo-copy").region
        )
        assert text.startswith("You already have a Daily Brief.")
        assert "anthropic · claude-haiku-4-5" in text
        assert "openai · gpt-4.1-mini" not in text
        assert stub.started == 0
