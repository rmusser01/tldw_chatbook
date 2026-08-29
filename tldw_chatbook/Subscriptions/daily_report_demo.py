"""One-click Daily Report demo: seed a real watchlist, run it live.

The demo IS the product (ADR-079): everything it creates -- watchlist, RSS
sources, briefing preset, 24h cadence -- is real, persistent, user-owned
setup, and the run-now path is the same claim-guarded machinery the scheduler
uses tomorrow. Idempotency keys on configured briefing schedules
(`list_briefing_schedules`), never on names.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from loguru import logger

from tldw_chatbook.Subscriptions.briefing_cast import dump_roster, validate_roster
from tldw_chatbook.Subscriptions.briefing_service import (
    STATUS_COMPLETE,
    STATUS_EMPTY,
    GenerationInFlightError,
    default_briefing_provider,
    generate_briefing,
)
from ..Chat.Chat_Functions import chat_api_call
from ..DB.Subscriptions_DB import SubscriptionsDB

DEMO_WATCHLIST_NAME = "Daily Brief"
DEMO_PRESET_NAME = "Daily Brief"
DEMO_CADENCE_SECONDS = 86_400
DEMO_STYLE_NOTES = (
    "A crisp daily news brief: top stories first, one short paragraph each, "
    "plain language, no filler."
)
#: Three stable RSS sources, all on the best-supported monitoring path.
#: Hacker News arrives via its official RSS feed (hnrss.org).
DEMO_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "name": "Hacker News (front page)",
        "source_type": "rss",
        "url": "https://hnrss.org/frontpage",
    },
    {
        "name": "BBC World News",
        "source_type": "rss",
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
    },
    {
        "name": "Ars Technica",
        "source_type": "rss",
        "url": "https://feeds.arstechnica.com/arstechnica/index",
    },
)
_AUDIO_SETTINGS_HINT = (
    "Audio skipped: add a TTS voice profile (Settings → Speech/TTS) and "
    "install the audio extra to hear tomorrow's brief."
)
_PROVIDER_GUIDANCE = (
    " Check your provider in Settings (F9) → API Keys, then run the demo again."
)


class DailyReportDemoService:
    """Seed-and-run orchestration for the Daily Report demo.

    All collaborators are injected: late-bound app services arrive as getters
    (the app wires this service before some of them exist), the chat callable
    and (Task 6) the synthesize callable are DI seams for tests.
    """

    def __init__(
        self,
        subscriptions_db: SubscriptionsDB,
        *,
        local_watchlists_getter: Callable[[], Any | None],
        dispatch_service: Any | None,
        app_getter: Callable[[], Any | None],
        tts_service_getter: Callable[[], Any | None] | None = None,
        tts_profile_service_getter: Callable[[], Any | None] | None = None,
        chat: Callable[..., Any] = chat_api_call,
        synthesize: Callable[..., Any] | None = None,
    ) -> None:
        self._db = subscriptions_db
        self._local_getter = local_watchlists_getter
        self._dispatch = dispatch_service
        self._app_getter = app_getter
        self._tts_getter = tts_service_getter or (lambda: None)
        self._tts_profiles_getter = tts_profile_service_getter or (lambda: None)
        self._chat = chat
        self._synthesize = synthesize

    async def run_demo(self) -> dict[str, Any]:
        """Run the whole demo; never raises (failures land in the outcome)."""
        outcome: dict[str, Any] = {
            "status": "error",
            "watchlist_id": None,
            "briefing_id": None,
            "audio": None,
            "reasons": [],
        }
        try:
            await self._run(outcome)
        except Exception as exc:  # noqa: BLE001 - the UI shows the outcome, not a traceback
            logger.warning(f"Daily report demo crashed: {type(exc).__name__}")
            outcome["reasons"].append(f"error:{type(exc).__name__}")
        return outcome

    async def _run(self, outcome: dict[str, Any]) -> None:
        local = self._local_getter()
        if local is None:
            outcome["status"] = "unavailable"
            outcome["reasons"].append("no-local-watchlists-service")
            return
        if not str(default_briefing_provider()).strip():
            outcome["status"] = "unavailable"
            outcome["reasons"].append("no-provider")
            return

        schedules = await asyncio.to_thread(self._db.list_briefing_schedules)
        if schedules:
            # Someone already has a daily report: reuse it, never re-seed.
            watchlist_id = int(schedules[0]["watchlist_id"])
            # Adaptation (disclosed in task-5-report): the plan's verbatim
            # `await self._default_preset_id(...)` awaited a sync method
            # (TypeError on the reuse path). Routed through `asyncio.to_thread`
            # instead -- fixes the await AND keeps this DB read off the event
            # loop, matching `_watchlist_source_ids`/`_count_items` above.
            preset_id = await asyncio.to_thread(
                self._default_preset_id, watchlist_id
            )
            outcome["watchlist_id"] = watchlist_id
            outcome["reasons"].append("existing-schedule")
        else:
            watchlist_id, preset_id = await self._seed(local)
            outcome["watchlist_id"] = watchlist_id
            outcome["reasons"].append("seeded")

        await self._notify(
            "Fetching today's stories",
            "Checking your Daily Brief sources…",
        )
        fetched = await self._check_sources(local, watchlist_id, outcome)
        if fetched == 0:
            outcome["status"] = "fetch_failed"
            await self._notify(
                "Daily brief could not fetch sources",
                "None of the Daily Brief sources could be reached. Check your "
                "network and try again -- your schedule is saved and will "
                "retry on its own.",
                severity="warning",
            )
            return

        await self._notify(
            "Writing your brief", "Your LLM provider is drafting today's brief…"
        )
        try:
            row = await generate_briefing(
                self._db, watchlist_id, preset_id=preset_id, chat=self._chat
            )
        except GenerationInFlightError:
            outcome["status"] = "in_flight"
            await self._notify(
                "A briefing is already being written",
                "Nothing else was started; watch the Watchlists artifacts pane.",
                severity="warning",
            )
            return
        outcome["briefing_id"] = row.get("id")
        # Adaptation (disclosed in task-5-report): `STATUS_EMPTY` is a
        # successful, schedule-advancing terminal state in briefing_service
        # (the `('complete', 'empty')` allowlist in `latest_completed_
        # watermark`/`list_briefing_schedules`), and a second same-day demo
        # run legitimately finds nothing new above the coverage watermark.
        # Only a `failed` row is a provider failure worth guidance for.
        if str(row.get("status")) not in (STATUS_COMPLETE, STATUS_EMPTY):
            outcome["status"] = "briefing_failed"
            await self._notify(
                "Daily brief failed to generate",
                "The LLM provider refused or failed"
                + (f": {row.get('error')}" if row.get("error") else "")
                + "." + _PROVIDER_GUIDANCE,
                severity="warning",
            )
            return

        # Task 6 fills this in (audio stage); until then the brief is the product.
        outcome["audio"] = "skipped"
        outcome["status"] = "complete"
        await self._notify(
            "Daily brief ready",
            "Your first daily report is ready -- see Artifacts → Reports. It "
            "refreshes daily from now on.",
        )

    # -- seeding ---------------------------------------------------------

    async def _seed(self, local: Any) -> tuple[int, int | None]:
        """Create the watchlist, sources, preset, and daily cadence."""
        watchlist, _created = await local.resolve_or_create_watchlist(
            DEMO_WATCHLIST_NAME
        )
        watchlist_id = int(watchlist["id"])
        for payload in DEMO_SOURCES:
            row = await local.create_source(dict(payload))
            # Adaptation (disclosed in task-5-report): `create_source`
            # returns a normalized row whose `id` is namespaced
            # (`"local:subscription:3"`); the bare integer lives under
            # `source_id` (`normalize_local_subscription_row`).
            await local.add_source_to_watchlist(
                watchlist_id=watchlist_id, source_id=int(row["source_id"])
            )
        roster, _audio_ready = await self._build_roster()
        preset_id = await asyncio.to_thread(
            self._db.insert_briefing_preset,
            DEMO_PRESET_NAME,
            roster_json=dump_roster(validate_roster(roster)),
            style_notes=DEMO_STYLE_NOTES,
        )
        await asyncio.to_thread(
            self._db.set_watchlist_briefing_settings,
            watchlist_id,
            selection_mode="auto_featured",
            default_preset_id=preset_id,
            briefing_cadence_seconds=DEMO_CADENCE_SECONDS,
        )
        return watchlist_id, preset_id

    async def _build_roster(self) -> tuple[list[dict[str, Any]], bool]:
        """Cast roster from the user's real voice profiles.

        `resolve_roster_voices` raises for speakers without a profile id --
        there is no default-voice fallback -- so zero profiles means a
        placeholder single-speaker roster and audio skipped (spec deviation 1).
        """
        profile_service = self._tts_profiles_getter()
        if profile_service is None:
            return [{"name": "Host", "voice_profile_id": None}], False
        try:
            result = await profile_service.list_profiles()
            profiles = list(result.value.profiles)
        except Exception:  # noqa: BLE001 - audio is optional; never fail the demo here
            return [{"name": "Host", "voice_profile_id": None}], False
        usable = [p for p in profiles if getattr(p, "profile_id", None) is not None]
        if not usable:
            return [{"name": "Host", "voice_profile_id": None}], False
        speakers = [
            {"name": "Host", "voice_profile_id": str(usable[0].profile_id)}
        ]
        if len(usable) > 1:
            speakers.append(
                {"name": "Analyst", "voice_profile_id": str(usable[1].profile_id)}
            )
        return speakers, True

    # -- run-now ---------------------------------------------------------

    async def _check_sources(
        self, local: Any, watchlist_id: int, outcome: dict[str, Any]
    ) -> int:
        """Run every source's check now; return how many produced items."""
        source_ids = await asyncio.to_thread(
            self._watchlist_source_ids, watchlist_id
        )
        for source_id in source_ids:
            launched = await local.launch_run(source_id=source_id)
            await local.execute_run(launched["run_id"])
        return await asyncio.to_thread(self._count_items, watchlist_id)

    def _watchlist_source_ids(self, watchlist_id: int) -> list[int]:
        with self._db.transaction() as conn:
            rows = conn.execute(
                "SELECT subscription_id FROM watchlist_sources "
                "WHERE watchlist_id = ? ORDER BY subscription_id",
                (watchlist_id,),
            ).fetchall()
        return [int(r["subscription_id"]) for r in rows]

    def _count_items(self, watchlist_id: int) -> int:
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM subscription_items AS i "
                "JOIN watchlist_sources AS ws ON ws.subscription_id = i.subscription_id "
                "WHERE ws.watchlist_id = ?",
                (watchlist_id,),
            ).fetchone()
        return int(row["n"])

    def _default_preset_id(self, watchlist_id: int) -> int | None:
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT default_briefing_preset_id FROM watchlists WHERE id = ?",
                (watchlist_id,),
            ).fetchone()
        return row["default_briefing_preset_id"] if row else None

    # -- notifications ----------------------------------------------------

    async def _notify(
        self, title: str, message: str, *, severity: str = "information"
    ) -> None:
        if self._dispatch is None:
            return
        try:
            self._dispatch.dispatch(
                app=self._app_getter(),
                category="briefing",
                title=title,
                message=message,
                severity=severity,
                source_entity_kind="daily_report_demo",
            )
        except Exception as exc:  # noqa: BLE001 - notifications must never fail the demo
            logger.warning(
                f"Demo stage notification failed: {type(exc).__name__}"
            )
