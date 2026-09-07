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

from tldw_chatbook.Subscriptions.briefing_audio import generate_script_audio
from tldw_chatbook.Subscriptions.briefing_cast import (
    dump_roster,
    generate_script,
    validate_roster,
)
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
_NO_PRESET_AUDIO_HINT = (
    "Audio skipped: this watchlist has no default briefing preset to cast with."
)
_EMPTY_WINDOW_AUDIO_HINT = "Audio skipped — nothing new to read today."
_PROVIDER_GUIDANCE = (
    " Check your provider in Settings (F9) → API Keys, then run the demo again."
)

#: Terminal run statuses `LocalWatchlistsService` records as a failure --
#: mirrors that module's own `_FAILED_RUN_STATUSES` (kept private there, so
#: restated here rather than imported across the boundary). The demo only
#: needs the yes/no question, never the distinctions between them.
_FAILED_RUN_STATUSES = frozenset({"failed", "error", "errored"})

# Qodo #11: one demo at a time may sit in the seed/check critical section.
# `asyncio.Lock` binds itself to the event loop it is first awaited on, and
# the module outlives test event loops (each pytest-asyncio case gets a
# fresh one), so the lock is held in a pair of module globals and rebound
# whenever the running loop changes -- in production there is exactly one
# loop for the process's whole life, so it behaves as a plain module-level
# lock; only the test harness ever triggers a rebind.
_DEMO_SECTION_LOCK: asyncio.Lock | None = None
_DEMO_SECTION_LOCK_LOOP: asyncio.AbstractEventLoop | None = None


def _demo_section_lock() -> asyncio.Lock:
    """The module-wide demo critical-section lock, loop-aware (Qodo #11)."""
    global _DEMO_SECTION_LOCK, _DEMO_SECTION_LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _DEMO_SECTION_LOCK is None or _DEMO_SECTION_LOCK_LOOP is not loop:
        _DEMO_SECTION_LOCK = asyncio.Lock()
        _DEMO_SECTION_LOCK_LOOP = loop
    return _DEMO_SECTION_LOCK


class DailyReportDemoService:
    """Seed-and-run orchestration for the Daily Report demo.

    All collaborators are injected: late-bound app services arrive as getters
    (the app wires this service before some of them exist), the chat callable
    and the synthesize callable are DI seams for tests.
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
        #: Strong references to in-flight demo tasks (Qodo #10) -- same
        #: discipline as `BriefingJobHandler._pending_generations`: a bare
        #: `asyncio.create_task` result with no other reference is only
        #: weakly held by the event loop, and this set is also the
        #: double-start guard's source of truth. Discarded on completion.
        self._pending_demos: set[asyncio.Task[dict[str, Any]]] = set()

    def demo_in_progress(self) -> bool:
        """Whether a demo started through this service is still running.

        The testable seam behind `run_demo_detached`'s refusal: the pending
        set is the same strong-ref registry production uses, so an assertion
        here observes exactly what the guard observes.
        """
        return any(not task.done() for task in self._pending_demos)

    def run_demo_detached(self) -> asyncio.Task[dict[str, Any]] | None:
        """Start one demo as an app-owned background task (Qodo #10).

        `run_demo` spans fetches and an LLM call -- minutes of wall time --
        and both CTAs (Artifacts empty state, Watchlists banner) used to run
        it INSIDE a screen-owned Textual worker. Textual cancels a widget's
        workers on unmount, so navigating away mid-demo cancelled the
        orchestration after some persistent state (watchlist, sources,
        preset) had already been committed, leaving partial seed state. The
        task this spawns is owned by the SERVICE, not any screen, so it
        survives navigation; completion and failure notifications already
        arrive through the dispatch service, so no screen needs the outcome.

        Refuses (and dispatches a calm "already running" notification) when
        a demo is still in flight: two near-simultaneous CTA presses must
        not both pass the empty-schedule check and double-seed (Qodo #11's
        other half -- the module-level lock in `_run` covers direct
        `run_demo` callers the same way).

        Returns:
            The spawned task -- already started, never awaited here; its
            result is the same outcome dict `run_demo` returns -- or `None`
            when a demo is already running.
        """
        if self.demo_in_progress():
            self._dispatch_notification(
                "A demo is already running",
                "Nothing else was started; watch the Watchlists artifacts pane.",
                severity="warning",
            )
            return None
        task = asyncio.create_task(self.run_demo(), name="daily_report_demo")
        self._pending_demos.add(task)
        task.add_done_callback(self._pending_demos.discard)
        return task

    async def run_demo(self) -> dict[str, Any]:
        """Run the whole demo; never raises (failures land in the outcome).

        Returns:
            The outcome dict. Keys: ``status`` (one of ``complete``,
            ``fetch_failed``, ``briefing_failed``, ``in_flight``,
            ``unavailable``, ``error``), ``watchlist_id`` (once resolved),
            ``briefing_id`` (once a briefing row exists), ``audio`` (once
            the audio stage was reached: ``complete``, ``skipped``, or
            ``failed``), and ``reasons`` (a list of machine-readable
            ``section:detail`` strings recording every skip/failure branch
            taken, e.g. ``existing-schedule``, ``empty-window``,
            ``audio:skipped:no-preset``).
        """
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

        # Qodo #11: schedule discovery, seeding, and the first fetch sit in
        # one module-wide critical section. Two concurrent callers that both
        # see "no schedule yet" would both seed (duplicate sources/presets
        # on one watchlist); the second caller now waits until the first has
        # committed, then legitimately reuses what it finds. Released before
        # generation: the LLM call already has its own claim machinery, and
        # holding a module lock across a multi-minute provider call would
        # serialize unrelated demos for no safety gain.
        async with _demo_section_lock():
            schedules = await asyncio.to_thread(self._db.list_briefing_schedules)
            if schedules:
                # Someone already has a daily report: reuse it, never re-seed.
                watchlist_id = int(schedules[0]["watchlist_id"])
                # Adaptation (disclosed in task-5-report): the plan's verbatim
                # `await self._default_preset_id(...)` awaited a sync method
                # (TypeError on the reuse path). Routed through `asyncio.to_thread`
                # instead -- fixes the await AND keeps this DB read off the event
                # loop, matching `_watchlist_source_ids` above.
                preset_id = await asyncio.to_thread(
                    self._default_preset_id, watchlist_id
                )
                audio_ready = await self._audio_ready_now()
                outcome["watchlist_id"] = watchlist_id
                outcome["reasons"].append("existing-schedule")
            else:
                watchlist_id, preset_id, audio_ready = await self._seed(local)
                outcome["watchlist_id"] = watchlist_id
                outcome["reasons"].append("seeded")

            await self._notify(
                "Fetching today's stories",
                "Checking your Daily Brief sources…",
            )
            fetched = await self._check_sources(local, watchlist_id)

        # Qodo #9: the verdict is the runs THIS invocation launched, never
        # the watchlist's lifetime item count -- on a reused schedule the
        # historical items would mask a total current-fetch failure.
        if not fetched:
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
        row_status = str(row.get("status"))
        if row_status not in (STATUS_COMPLETE, STATUS_EMPTY):
            outcome["status"] = "briefing_failed"
            await self._notify(
                "Daily brief failed to generate",
                "The LLM provider refused or failed"
                + (f": {row.get('error')}" if row.get("error") else "")
                + "." + _PROVIDER_GUIDANCE,
                severity="warning",
            )
            return

        if row_status == STATUS_EMPTY:
            outcome["reasons"].append("empty-window")

        # Qodo #13: branch on the row's status BEFORE the audio-ready check.
        # `generate_script` refuses a non-complete briefing by contract, so
        # an empty-window row reaching it could only come back as a spurious
        # "could not be synthesized" failure -- there is nothing to read,
        # which is a calm skip, not an audio failure.
        if row_status == STATUS_EMPTY:
            outcome["audio"] = "skipped"
            outcome["reasons"].append("audio:skipped:empty-window")
            await self._notify("Audio skipped", _EMPTY_WINDOW_AUDIO_HINT)
        elif audio_ready:
            await self._notify(
                "Recording audio", "Synthesizing your audio brief…"
            )
            outcome["audio"] = await self._generate_audio(
                row, preset_id, outcome
            )
        else:
            outcome["audio"] = "skipped"
            await self._notify("Audio skipped", _AUDIO_SETTINGS_HINT)

        outcome["status"] = "complete"
        await self._notify(
            "Daily brief ready",
            "Your first daily report is ready -- see Artifacts → Reports. It "
            "refreshes daily from now on.",
        )

    # -- seeding ---------------------------------------------------------

    async def _seed(self, local: Any) -> tuple[int, int | None, bool]:
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
        roster, audio_ready = await self._build_roster()
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
        return watchlist_id, preset_id, audio_ready

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

    async def _audio_ready_now(self) -> bool:
        _roster, ready = await self._build_roster()
        return ready

    # -- run-now ---------------------------------------------------------

    async def _check_sources(self, local: Any, watchlist_id: int) -> bool:
        """Run every source's check now; whether this invocation got through.

        Qodo #9: the answer comes from the runs launched HERE, each of which
        returns its terminal run row (`LocalWatchlistsService.execute_run`
        records `failed`/`error`/`errored` for every failure it contains,
        and never raises for a fetch failure). The watchlist's LIFETIME item
        count is deliberately not consulted: on a reused schedule the
        historical items would mask a total current-fetch failure and hand
        the briefing an empty window indistinguishable from "nothing new".
        Any non-failed run -- completed with zero new items included -- is a
        fetch that got through, and the legitimate `empty` window is handled
        downstream by the briefing's own status.

        Returns:
            `True` when at least one launched run did not fail. `False`
            when every launched run failed, and also when no sources exist
            to launch (nothing was fetched either way; the seed path always
            attaches three).
        """
        source_ids = await asyncio.to_thread(
            self._watchlist_source_ids, watchlist_id
        )
        any_success = False
        for source_id in source_ids:
            # Per-source isolation, matching the run pipeline's own
            # per-URL isolation: a source deleted between discovery and
            # launch raises `KeyError` from `launch_run`, and that must
            # cost one failed run, not the whole demo -- the other sources
            # still fetched.
            try:
                launched = await local.launch_run(source_id=source_id)
                run = await local.execute_run(launched["run_id"])
            except Exception as exc:  # noqa: BLE001 - one dead source is one failed run
                logger.warning(
                    f"Daily report demo: source {source_id} check failed: "
                    f"{type(exc).__name__}"
                )
                continue
            if str(run.get("status") or "").strip().lower() not in (
                _FAILED_RUN_STATUSES
            ):
                any_success = True
        return any_success

    async def _generate_audio(
        self,
        briefing_row: dict[str, Any],
        preset_id: int | None,
        outcome: dict[str, Any],
    ) -> str:
        """Cast + synthesize; any failure degrades to a text-only success."""
        # Qodo #12: a reused schedule whose default preset was cleared has
        # nothing to cast with. `generate_script` resolves a preset by id,
        # so a fabricated 0 could only produce a guaranteed failure dressed
        # up as an audio problem -- skip, accurately, instead.
        if preset_id is None:
            outcome["reasons"].append("audio:skipped:no-preset")
            await self._notify("Audio skipped", _NO_PRESET_AUDIO_HINT)
            return "skipped"
        briefing_id = int(briefing_row["id"])
        try:
            script = await generate_script(
                self._db,
                briefing_id,
                preset_id=int(preset_id),
                chat=self._chat,
            )
            if str(script.get("status")) != "complete":
                outcome["reasons"].append(
                    f"script:{script.get('status')}:{script.get('error')}"
                )
                await self._notify(
                    "Audio skipped", _AUDIO_SETTINGS_HINT, severity="warning"
                )
                return "skipped"
            kwargs: dict[str, Any] = {
                "tts_service": self._tts_getter(),
                "profile_service": self._tts_profiles_getter(),
            }
            if self._synthesize is not None:
                kwargs["synthesize"] = self._synthesize
            audio = await generate_script_audio(
                self._db, int(script["id"]), **kwargs
            )
            if str(audio.get("status")) != "complete":
                outcome["reasons"].append(
                    f"audio:{audio.get('status')}:{audio.get('error')}"
                )
                await self._notify(
                    "Audio could not be synthesized",
                    "Today's text brief is ready. "
                    + _AUDIO_SETTINGS_HINT,
                    severity="warning",
                )
                return "failed"
            return "complete"
        except Exception as exc:  # noqa: BLE001 - audio is optional, never fatal
            outcome["reasons"].append(f"audio-error:{type(exc).__name__}")
            await self._notify(
                "Audio could not be synthesized",
                "Today's text brief is ready. "
                + _AUDIO_SETTINGS_HINT,
                severity="warning",
            )
            return "failed"

    def _watchlist_source_ids(self, watchlist_id: int) -> list[int]:
        with self._db.transaction() as conn:
            rows = conn.execute(
                "SELECT subscription_id FROM watchlist_sources "
                "WHERE watchlist_id = ? ORDER BY subscription_id",
                (watchlist_id,),
            ).fetchall()
        return [int(r["subscription_id"]) for r in rows]

    def _default_preset_id(self, watchlist_id: int) -> int | None:
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT default_briefing_preset_id FROM watchlists WHERE id = ?",
                (watchlist_id,),
            ).fetchone()
        return row["default_briefing_preset_id"] if row else None

    # -- notifications ----------------------------------------------------

    def _dispatch_notification(
        self, title: str, message: str, *, severity: str = "information"
    ) -> None:
        """Dispatch one notification; never raises, no loop required.

        The sync core `_notify` wraps, so non-async callers (`run_demo_
        detached`'s refusal path, which must notify BEFORE returning) share
        the exact same containment and payload shape.
        """
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

    async def _notify(
        self, title: str, message: str, *, severity: str = "information"
    ) -> None:
        """Dispatch one stage notification through `_dispatch_notification`."""
        self._dispatch_notification(title, message, severity=severity)
