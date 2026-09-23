"""[dreams] Cycle orchestration: one dated collection per run.

``run_cycle`` is the whole discovery pipeline of spec §discovery staged
end-to-end — reclaim stale rows, refresh the interest profile from
notes/media/Personal-Context signals (ruling R18), apply the feedback loop
to topic weights (ruling R19), snapshot the profile, claim the local date,
synthesize queries, search, harvest the watchlist, rank unseen candidates,
generate one story per pick, and write every outcome as a row. Every
blocking call (web search, chat, SQLite) runs under ``asyncio.to_thread``
with DB writes grouped one hop per stage, copying
``briefing_service``'s discipline. Degradations — synthesis fallback, search
failure, budget trims, a failed signal source — never kill the cycle; they
append to the collection's ``degradation_notes``.

A date that already owns a collection row is not an error: later triggers
for the same date run in APPEND mode (spec §idempotency, ruling R11) — the
normal pipeline appends stories until the date's TOTAL reaches
``stories_per_cycle``, daily budgets still apply (an append may legitimately
produce zero stories, recorded as a degradation note), and the final status
is judged on the collection's total rows, so a filled collection can
upgrade.

``run_catchup_if_due`` implements the boot/surface catch-up predicate: a
laptop that slept through its slot gets today's cycle when the cadence says
one is due, or when the latest attempt failed (a failed day retries).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from loguru import logger

from tldw_chatbook.Dreams import (
    discovery,
    interest_profile,
    profile_sources,
    query_synthesis,
    story_service,
)
from tldw_chatbook.Dreams.discovery import Candidate
from tldw_chatbook.Dreams.settings import dreams_setting
from tldw_chatbook.Dreams.story_service import StoryResult
from tldw_chatbook.Utils.timestamps import to_utc_iso
from tldw_chatbook.Web_Scraping.WebSearch_APIs import perform_websearch

if TYPE_CHECKING:
    from tldw_chatbook.DB.Dreams_DB import DreamsDB

#: A ``generating`` row older than this is a crashed cycle, not a live one
#: (spec §date-bucket idempotency); the reclaim runs before each date claim.
_STALE_GENERATING_MINUTES = 15

#: URL namespace for LLM-knowledge stories: search failed or is disabled, so
#: there is no source URL — the synthetic scheme keeps the row honest about
#: having no link to dive into.
_LLM_URL_PREFIX = "dreams://llm/"

#: How far back the feedback loop reads (spec §feedback loop: a trailing
#: two-week window of story reactions). Feedback kinds count +1 (``more``/
#: ``kept``/``dived``/``ingested``) or −1 (``less``); ``exported``/
#: ``tracked`` are recorded but carry no weight signal. The per-reaction
#: step and clamp live with the snapshot that applies them
#: (``interest_profile.FEEDBACK_STEP``).
_FEEDBACK_WINDOW_DAYS = 14

_WORD_RE = re.compile(r"[a-z0-9]+")


# --- In-process cycle claims (copies _ACTIVE_BRIEFING_CLAIMS's discipline) --
#
# Scheduler fire, boot catch-up, and manual trigger share one process and one
# event loop; a module-level set of local dates currently being cycled stops
# every in-process collision. Mutated ONLY on the event loop: run_cycle never
# awaits between its membership check and its .add(), so two coroutines can
# never interleave between them — that is the whole reason this needs no
# lock. A claim cannot outlive the process, so a genuine crash still looks
# exactly like a crash to fail_stale_generating. Never mutate this set from
# a thread (e.g. from inside a function asyncio.to_thread runs).
_ACTIVE_CYCLE_DATES: set[str] = set()


@dataclass(slots=True)
class CycleDeps:
    """Injected collaborators for one cycle run.

    Attributes:
        dreams_db: The Dreams database (required; the only hard dependency).
        chachanotes_db_getter: Returns the notes DB or None; None means no
            notes keyword signals for the profile refresh (ruling R18).
        media_db_getter: Returns the media DB or None; None means no
            library-URL dedupe and no media keyword signals.
        subs_db_getter: Returns the subscriptions DB or None; None means no
            watchlist candidate pool.
        pc_service_getter: Returns the Personal Context service or None;
            None means no personal-context signals for the profile refresh.
        chat_getter: Returns the pre-bound chat seam
            (``story_service.resolve_dreams_chat``-shaped); may raise, the
            cycle degrades.
        perform_search: ``perform_websearch``-shaped search seam.
        now: Injected clock (UTC-aware); drives date bucketing, staleness,
            and catch-up windows.
    """
    dreams_db: DreamsDB
    chachanotes_db_getter: Callable[[], Any]
    media_db_getter: Callable[[], Any]
    subs_db_getter: Callable[[], Any]
    pc_service_getter: Callable[[], Any]
    chat_getter: Callable[[], Callable[..., Any]]
    perform_search: Callable[..., Any] = perform_websearch
    now: Callable[[], datetime] = field(
        default=lambda: datetime.now(UTC)
    )


@dataclass(slots=True)
class _BudgetPlan:
    """What the daily budgets still allow after this cycle's earlier spend.

    Attributes:
        searches: Search calls left today (caps how many queries run).
        stories: Story-generation calls left today (caps story count).
    """
    searches: int
    stories: int


def _local_date(now: datetime) -> str:
    """The user's local calendar date bucketing a cycle (spec §idempotency)."""
    return now.astimezone().strftime("%Y-%m-%d")


def _plan_budget(usage: dict) -> _BudgetPlan:
    """Trim remaining work to the day's global budgets (spec §budgets).

    Over-budget degrades by dropping the lowest-priority work first — the
    caller trims exploration queries (the trailing synthesized lines) before
    stories — and stamps why into the cycle's degradation notes.

    Args:
        usage: ``DreamsDB.usage_get`` result for the local date.

    Returns:
        The remaining search and story allowances (never negative).
    """
    searches = max(0, int(dreams_setting("max_searches_per_day"))
                   - int(usage["searches"]))
    stories = max(0, int(dreams_setting("max_llm_calls_per_day"))
                  - int(usage["llm_calls"]))
    return _BudgetPlan(searches=searches,
                       stories=min(int(dreams_setting("stories_per_cycle")),
                                   stories))


def _library_urls(deps: CycleDeps) -> set[str]:
    """URLs already in the media library; empty set when there is no DB.

    One parameterized read of the ``Media`` url column (UNIQUE-indexed),
    excluding soft-deleted and trashed rows — the library is dedupe signal,
    never resurfaced content (spec §non-goals). Runs on a worker thread (the
    caller wraps it in ``asyncio.to_thread``).
    """
    media_db = deps.media_db_getter()
    if media_db is None:
        return set()
    rows = media_db.execute_query(
        "SELECT url FROM Media"
        " WHERE deleted = ? AND is_trash = ? AND url IS NOT NULL",
        (0, 0),
    ).fetchall()
    return {str(row["url"]) for row in rows if row["url"]}


def _append_target(dreams_db: DreamsDB, local_date: str) -> dict | None:
    """Resolve the existing collection row a later trigger appends onto.

    Adds ``story_rows`` (rows already written for the date) and ``urls``
    (those rows' URLs, excluded from the append's candidate pool so the
    (collection_id, url) unique index never has to refuse a duplicate —
    covering also the crash window between an insert and its seen-ledger
    upsert). None when the date's row vanished between the INSERT conflict
    and this read; the caller treats that as ``date_taken``.
    """
    row = dreams_db.get_collection_by_date(local_date)
    if row is None:
        return None
    with dreams_db.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM dream_stories WHERE collection_id = ?",
            (row["id"],),
        ).fetchone()[0]
        urls = [
            str(r[0]) for r in conn.execute(
                "SELECT url FROM dream_stories WHERE collection_id = ?",
                (row["id"],),
            ).fetchall()
        ]
    row["story_rows"] = int(count)
    row["urls"] = urls
    return row


def _finish_cycle(
    dreams_db: DreamsDB,
    collection_id: int,
    *,
    degradation_notes: str | None,
    provider: str | None,
    model: str | None,
    completed_at: str,
) -> tuple[str, int]:
    """Stamp the collection's final status judged on its TOTAL story rows.

    Totals, not just this run's picks (ruling R11): in append mode the
    verdict must account for rows earlier triggers wrote. A 0-row ``failed``
    collection that appends cleanly upgrades to ``complete``; a collection
    still holding failed rows stays ``partial`` however much is appended —
    the rows are the truth, not the previous label.

    Returns:
        ``(final_status, total_story_rows)``.
    """
    with dreams_db.connection() as conn:
        counts = conn.execute(
            "SELECT COUNT(*) AS total,"
            " COALESCE(SUM(status = 'complete'), 0) AS complete"
            " FROM dream_stories WHERE collection_id = ?",
            (collection_id,),
        ).fetchone()
    total, complete = int(counts["total"]), int(counts["complete"])
    if complete == 0:
        final = "failed"
    elif complete == total:
        final = "complete"
    else:
        final = "partial"
    dreams_db.set_collection_status(
        collection_id, final, provider=provider, model=model,
        degradation_notes=degradation_notes, completed_at=completed_at,
        story_count=total)
    return final, total


def _matched_topics(topics: list[dict], candidate: Candidate) -> list[str]:
    """Snapshot topics whose text overlaps the candidate (title + snippet).

    Mirrors ``discovery._overlap_score``'s matching semantics (substring or
    shared word token, case-insensitive) so the stored matches agree with
    what ranked the story; not imported because that helper is private to
    its module.
    """
    text = f"{candidate.title} {candidate.snippet}".lower()
    words = set(_WORD_RE.findall(text))
    return [
        topic for topic in
        (str(t.get("text", "")).strip().lower() for t in topics)
        if topic and (topic in text or words & set(_WORD_RE.findall(topic)))
    ]


def _attributed_query(candidate: Candidate, queries: list[str]) -> str:
    """The search query that surfaced a candidate, for the story row.

    ``discovery.run_queries`` flattens the pool without provenance, so a web
    candidate is attributed the run's best word-overlapping query (ties keep
    the first). Watchlist items were not surfaced by any query (""), and an
    llm-knowledge story IS its query (carried as the candidate title).
    """
    if candidate.source == "llm":
        return candidate.title
    if candidate.source == "watchlist" or not queries:
        return ""
    words = set(_WORD_RE.findall(f"{candidate.title} {candidate.snippet}"
                                 .lower()))
    best, best_score = "", -1
    for query in queries:
        score = len(words & set(_WORD_RE.findall(query.lower())))
        if score > best_score:
            best, best_score = query, score
    return best


def _record_story(
    dreams_db: DreamsDB,
    local_date: str,
    collection_id: int,
    *,
    candidate: Candidate,
    result: StoryResult,
    matched_topics: list[str],
    query: str,
    count_call: bool,
) -> None:
    """Persist one story outcome row and its budget spend (one thread hop).

    Plain synchronous SQLite only — the cycle runs it under
    ``asyncio.to_thread``; keeping both writes together gives one hop per
    story instead of two. ``count_call`` skips the llm-budget bump for rows
    that never reached an LLM (no chat resolved — nothing was spent).
    """
    if count_call:
        dreams_db.usage_bump(local_date, llm_calls=1)
    dreams_db.insert_story(
        collection_id,
        title=result.title,
        url=candidate.url,
        snippet=candidate.snippet,
        body=result.body,
        status=result.status,
        source=candidate.source,
        kind=result.kind,
        event_date=result.event_date,
        location=result.location,
        matched_topics=matched_topics,
        query=query,
        error=result.error,
    )


def _pc_distillate_cache_path() -> Path:
    """The Personal-Context distillate cache under the user data dir.

    Same root helper ``get_dreams_db_path`` resolves through
    (``get_user_data_dir``), adapted to a sub-directory: the lock-tolerant
    PC reader treats this file as its cache-on-unlock distillate. The import
    is deferred so a config-import failure stays a cycle-time degradation,
    not a module-import crash. May raise when the secured directory cannot
    be resolved — the caller degrades to skipping the PC source.
    """
    from tldw_chatbook.config import get_user_data_dir

    return get_user_data_dir() / "dreams" / "pc_distillate.json"


def _preferred_sources(
    collected: list[tuple[str, list[dict]]],
) -> dict[tuple[str, str], str]:
    """Per merged ``(facet, text)`` key, the origin that weighed most.

    A topic the sources agree on keeps ONE row, so it needs one ``source``:
    the reader whose signal contributed the most weight wins, first-read
    order (notes, media, personal context) breaking ties. The reader's own
    ``source`` value is preferred when it sets one, with the origin name as
    the fallback.
    """
    winners: dict[tuple[str, str], tuple[float, str]] = {}
    for origin, rows in collected:
        for entry in rows:
            key = (str(entry.get("facet", "topic")),
                   str(entry.get("text", "")).strip().lower())
            if not key[1]:
                continue
            source = str(entry.get("source") or origin)
            weight = float(entry.get("weight", 0.0))
            best = winners.get(key)
            if best is None or weight > best[0]:
                winners[key] = (weight, source)
    return {key: source for key, (_, source) in winners.items()}


def _upsert_profile_signals(
    dreams_db: DreamsDB,
    merged: list[dict],
    source_by_key: dict[tuple[str, str], str],
    *,
    now_iso: str,
) -> None:
    """Persist merged signal rows; sync, the caller thread-offloads.

    Merge semantics (ruling R18): DO NOT clobber — existing rows whose
    ``source`` is ``user``/``seed`` keep their weight and their boost stamp
    entirely (only the feedback stage, R19, may stamp them later); derived
    rows take the new merged weight. Rows absent from the merge are NOT
    deleted — decay handles staleness. Protection matches the table's own
    uniqueness semantics (exact ``(facet, text)``), so a differently-spelled
    user row keeps its weight while the normalized derived row lands
    alongside it.

    Each row the refresh upserts also gets ``last_boosted_at = now``: fresh
    evidence restarts the decay clock, otherwise ``snapshot`` would floor
    every never-boosted row to ``0.05`` immediately and the personalization
    this stage builds would evaporate before a single query is synthesized.
    ``upsert_profile_entry`` deliberately never touches that column (it is
    owned by these flows), so the stamp is a separate batched UPDATE.
    """
    protected = {
        (str(row["facet"]), str(row["text"]))
        for row in dreams_db.list_profile()
        if str(row.get("source")) in ("user", "seed")
    }
    stamped: list[tuple[str, str]] = []
    for row in merged:
        key = (str(row["facet"]), str(row["text"]))
        if key in protected:
            continue
        dreams_db.upsert_profile_entry(
            key[0], key[1],
            weight=float(row["weight"]), searchable=1,
            source=source_by_key.get(key, "notes"),
        )
        stamped.append(key)
    if stamped:
        with dreams_db.transaction() as conn:
            conn.executemany(
                "UPDATE dream_interest_profile SET last_boosted_at = ?"
                " WHERE facet = ? AND text = ?",
                [(now_iso, facet, text) for facet, text in stamped],
            )


async def _refresh_profile_signals(
    deps: CycleDeps, now: datetime
) -> list[str]:
    """Stage 0a (ruling R18): rebuild derived profile rows from signals.

    Reads the three signal sources through the injected getters (a ``None``
    skips that source silently), merges them, and upserts the merged topics.
    Every source — and the write — degrades with a note on failure; the
    cycle always continues. Reads run one ``asyncio.to_thread`` hop per
    source DB, and the Dreams write is one more hop (stage discipline).
    """
    notes: list[str] = []
    collected: list[tuple[str, list[dict]]] = []
    notes_db = deps.chachanotes_db_getter()
    if notes_db is not None:
        try:
            collected.append(("notes", await asyncio.to_thread(
                profile_sources.read_note_topics, notes_db)))
        except Exception as exc:  # noqa: BLE001 - a dead source degrades
            notes.append(f"profile signals: notes failed: {exc}")
    media_db = deps.media_db_getter()
    if media_db is not None:
        try:
            collected.append(("media", await asyncio.to_thread(
                profile_sources.read_media_topics, media_db)))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"profile signals: media failed: {exc}")
    pc_service = deps.pc_service_getter()
    if pc_service is not None:
        try:
            cache_path = await asyncio.to_thread(_pc_distillate_cache_path)
        except Exception as exc:  # noqa: BLE001 - no temp fallback path
            notes.append(
                "profile signals: personal context cache path unresolvable"
                f" ({type(exc).__name__}); source skipped")
        else:
            # Never raises (lock-tolerant reader with the cache above).
            collected.append(("personal_context", await asyncio.to_thread(
                profile_sources.read_personal_context_topics, pc_service,
                cache_path=cache_path)))
    if not collected:
        return notes
    merged = interest_profile.merge_signals([rows for _, rows in collected])
    if not merged:
        return notes
    try:
        await asyncio.to_thread(
            _upsert_profile_signals, deps.dreams_db, merged,
            _preferred_sources(collected), now_iso=to_utc_iso(now))
    except Exception as exc:  # noqa: BLE001 - a failed write degrades
        notes.append(f"profile signals: profile write failed: {exc}")
    return notes


def _feedback_net(dreams_db: DreamsDB, *, now: datetime) -> dict[str, int]:
    """Net reactions per matched topic over the trailing window (sync).

    Joins ``dream_feedback`` → ``dream_stories.matched_topics`` and nets
    each normalized topic (+1 per ``more``/``kept``/``dived``/``ingested``,
    −1 per ``less``; ``exported``/``tracked`` are neutral). Nothing is
    written: ``interest_profile.snapshot`` applies the result as an offset,
    so a reaction counts exactly once per cycle for as long as it is inside
    the window -- never compounding into the stored weight -- and a derived
    topic's refresh cannot erase it. Goals are immune because the snapshot
    only offsets ``facet='topic'`` rows.

    Args:
        dreams_db: The Dreams database.
        now: Cycle clock; the window ends here.

    Returns:
        ``{normalized_topic: net}`` with zero nets omitted.
    """
    cutoff = to_utc_iso(now - timedelta(days=_FEEDBACK_WINDOW_DAYS))
    with dreams_db.connection() as conn:
        rows = conn.execute(
            "SELECT ds.matched_topics AS matched, fb.kind AS kind"
            " FROM dream_feedback AS fb"
            " JOIN dream_stories AS ds ON ds.id = fb.story_id"
            " WHERE fb.created_at >= ?",
            (cutoff,),
        ).fetchall()
    net: dict[str, int] = {}
    for row in rows:
        try:
            topics = json.loads(row["matched"] or "[]")
        except ValueError:
            continue
        kind = str(row["kind"])
        delta = (1 if kind in ("more", "kept", "dived", "ingested")
                 else -1 if kind == "less" else 0)
        if delta == 0 or not isinstance(topics, list):
            continue
        for topic in topics:
            topic = str(topic).strip().lower()
            if topic:
                net[topic] = net.get(topic, 0) + delta
    return {topic: value for topic, value in net.items() if value}


async def _apply_feedback(
    deps: CycleDeps, now: datetime
) -> tuple[dict[str, int], list[str]]:
    """Stage 0b (ruling R19): read recent story feedback for the snapshot.

    One ``asyncio.to_thread`` hop against the Dreams DB; any failure
    degrades with a note (and no offset) and never aborts the cycle.

    Returns:
        ``(net_by_topic, degradation_notes)``.
    """
    try:
        net = await asyncio.to_thread(_feedback_net, deps.dreams_db, now=now)
    except Exception as exc:  # noqa: BLE001 - the loop degrades, not the cycle
        return {}, [f"feedback loop failed: {exc}"]
    return net, []


async def run_cycle(deps: CycleDeps, *, trigger: str) -> dict:
    """Run one full discovery cycle for the current local date.

    Before the profile snapshot, two stages personalize it: the interest
    profile is refreshed from notes/media/Personal-Context signals (ruling
    R18) and recent story feedback is netted per topic and applied by the
    snapshot as a weight offset (ruling R19) — both degrade with notes and
    never abort the cycle.

    Never raises for provider/search failures: they degrade the collection
    (notes on the row) while every candidate still becomes a story row. A
    concurrent in-process run for the same date returns the in-flight marker
    instead of raising. A date that already owns a row runs in APPEND mode
    (ruling R11): the normal pipeline appends until the date's TOTAL story
    count reaches ``stories_per_cycle`` (the row keeps its creating
    ``trigger``), budgets still bind, and the final status is judged on the
    collection's totals — a filled collection may upgrade. Database errors
    propagate uncaught (briefing discipline: those are bugs, not
    degradations).

    Args:
        deps: Injected collaborators (see :class:`CycleDeps`).
        trigger: Which path fired (``scheduled``/``catchup``/``manual``/
            ``refresh``).

    Returns:
        ``{"collection_id": int | None, "status": str, "stories": int}`` —
        ``stories`` is the date's TOTAL story rows after this run.
    """
    now = deps.now()
    local_date = _local_date(now)
    dreams_db = deps.dreams_db

    # Event-loop-only claim: no await between the check and the add, so a
    # second dispatch for the same date can never interleave past the guard.
    if local_date in _ACTIVE_CYCLE_DATES:
        return {"collection_id": None, "status": "inflight", "stories": 0}
    _ACTIVE_CYCLE_DATES.add(local_date)
    try:
        cutoff = to_utc_iso(now - timedelta(minutes=_STALE_GENERATING_MINUTES))
        reclaimed = await asyncio.to_thread(
            dreams_db.fail_stale_generating, cutoff)
        if reclaimed:
            logger.info("Dreams cycle reclaimed {} stale generating row(s)",
                        reclaimed)
        # Seen-ledger retention (``seen_item_ttl_days``): an item not seen
        # for that long may be rediscovered.
        seen_cutoff = to_utc_iso(now - timedelta(
            days=int(dreams_setting("seen_item_ttl_days"))))
        await asyncio.to_thread(dreams_db.prune_seen, seen_cutoff)
        # Profile stages (rulings R18/R19) run BEFORE the snapshot so it
        # reflects refreshed signals and the feedback offsets; both degrade
        # with notes and never abort the cycle. Their notes are carried into
        # the collection row once it exists.
        stage_notes: list[str] = []
        stage_notes += await _refresh_profile_signals(deps, now)
        feedback_net, feedback_notes = await _apply_feedback(deps, now)
        stage_notes += feedback_notes
        snap = await asyncio.to_thread(
            interest_profile.snapshot, dreams_db, now_epoch=now.timestamp(),
            feedback=feedback_net)
        profile_digest = hashlib.sha256(
            json.dumps(snap, sort_keys=True).encode()).hexdigest()[:16]
        collection_id = await asyncio.to_thread(
            dreams_db.create_collection, local_date, trigger, profile_digest)
        notes: list[str] = list(stage_notes)
        configured_cap = int(dreams_setting("stories_per_cycle"))
        # Append mode (ruling R11): the date is taken, so this trigger
        # appends onto the existing row until the date's TOTAL reaches the
        # configured cap; the row keeps its creating trigger.
        append_mode = False
        prior_notes: str | None = None
        existing_urls: set[str] = set()
        story_cap = configured_cap
        if collection_id is None:
            target = await asyncio.to_thread(
                _append_target, dreams_db, local_date)
            if target is None:
                return {"collection_id": None, "status": "date_taken",
                        "stories": 0}
            collection_id = int(target["id"])
            append_mode = True
            prior_notes = target.get("degradation_notes")
            existing_urls = set(target["urls"])
            story_cap = max(0, configured_cap - int(target["story_rows"]))
            if story_cap <= 0:
                # Nothing can be added: stop before any provider, search, or
                # budget spend. The row is left exactly as it was.
                logger.info("Dreams cycle {} for {}: story cap already reached",
                            collection_id, local_date)
                return {"collection_id": collection_id,
                        "status": str(target["status"]),
                        "stories": int(target["story_rows"])}
        logger.info("Dreams cycle {} for {} (trigger={}, append={})",
                    collection_id, local_date, trigger, append_mode)
        queries: list[str] = []
        chat: Callable[..., Any] | None = None
        synthesis_called = False

        def counting_chat(**kwargs: Any) -> Any:
            """Chat seam wrapper recording whether synthesis invoked the LLM.

            The one synthesis call shares the daily ``llm_calls`` budget
            (spec §budgets / Task 4 review ruling), so the cycle must bump
            only when the LLM was actually invoked — the deterministic
            fallback after an unresolved provider makes no call at all.
            """
            nonlocal synthesis_called
            synthesis_called = True
            return chat(**kwargs)

        # The synthesis call spends the same daily llm budget as stories, so
        # it is checked BEFORE the call: an exhausted budget searches the
        # deterministic fallback queries and makes no model call at all.
        query_count = int(dreams_setting("queries_per_cycle"))
        pre_usage = await asyncio.to_thread(dreams_db.usage_get, local_date)
        llm_left = (int(dreams_setting("max_llm_calls_per_day"))
                    - int(pre_usage["llm_calls"]))
        try:
            # Thread-offloaded: the getter force-reloads config from disk
            # and may reach the keyring — neither belongs on the event loop.
            chat = await asyncio.to_thread(deps.chat_getter)
            if llm_left > 0:
                queries = await query_synthesis.synthesize_queries(
                    counting_chat, snapshot=snap, count=query_count,
                    exploration_slots=int(dreams_setting("exploration_slots")))
            else:
                queries = query_synthesis.preview_queries(
                    [str(t["text"]) for t in snap["topics"]], query_count)
                notes.append("llm budget exhausted: fallback queries, no "
                             "synthesis call")
        except Exception as exc:  # noqa: BLE001 - provider unavailable, degrade
            notes.append(f"query synthesis failed: {exc}")
        if synthesis_called:
            # Recorded BEFORE usage_get so the story budget below sees the
            # synthesis call as spent (controller ruling: the budget is
            # shared by the loop's LLM calls — synthesis is one).
            await asyncio.to_thread(
                dreams_db.usage_bump, local_date, llm_calls=1)

        usage = await asyncio.to_thread(dreams_db.usage_get, local_date)
        budget = _plan_budget(usage)
        if len(queries) > budget.searches:
            notes.append(
                f"search budget: dropped {len(queries) - budget.searches} "
                "lowest-priority queries (exploration first)")
            queries = queries[: budget.searches]
        if budget.stories < configured_cap:
            notes.append(f"llm budget: stories capped at {budget.stories}")
        # How many stories this run may add: the daily llm budget AND, in
        # append mode, the distance between the date's existing rows and the
        # configured cap (an exhausted budget legitimately appends zero).
        slots = min(budget.stories, story_cap)

        candidates: list[Candidate] = []
        if (dreams_setting("web_search_enabled") and budget.searches > 0
                and queries):
            found, used = await discovery.run_queries(
                deps.perform_search, engine=str(dreams_setting("search_engine")),
                queries=queries)
            await asyncio.to_thread(
                dreams_db.usage_bump, local_date, searches=used)
            candidates.extend(found)
        else:
            notes.append("web search disabled or over budget")

        # A local subscriptions read, not a web search: independent of the
        # search budget and the web-search kill switch.
        subs_db = deps.subs_db_getter()
        if subs_db is not None:
            candidates.extend(await asyncio.to_thread(
                discovery.fetch_watchlist_candidates, subs_db,
                freshness_hours=int(
                    dreams_setting("watchlist_freshness_hours")),
                now_epoch=now.timestamp()))

        # Seen-ledger filter BEFORE ranking (the clean form): keep only
        # unseen candidates, then rank with an empty seen set so the pure
        # call stays simple. In append mode the date's existing story URLs
        # are excluded too — the collection must gain NEW rows, not re-rank
        # ones an earlier trigger already wrote.
        # The ledger is keyed on ``discovery.normalize_url`` so a URL
        # variant (host casing, tracking parameters, trailing slash) of an
        # earlier find does not come back as new.
        unseen = await asyncio.to_thread(
            dreams_db.seen_filter_unseen,
            [discovery.normalize_url(c.url) for c in candidates])
        fresh = [c for c in candidates
                 if discovery.normalize_url(c.url) in unseen]
        library_urls = await asyncio.to_thread(_library_urls, deps)
        picked = discovery.dedupe_and_rank(
            fresh, seen_urls=set(),
            library_urls=library_urls | existing_urls,
            topics=snap["topics"], limit=slots)

        # LLM-knowledge mode (spec §error handling): total search failure or
        # the web-search kill switch still yields stories straight from the
        # model, visibly marked — a degraded cycle beats a silent one.
        if not candidates and queries and slots > 0 and chat is not None:
            llm_pool = [
                Candidate(title=query, snippet="", source="llm",
                          url=f"{_LLM_URL_PREFIX}{index}/"
                              f"{quote(query, safe='')}")
                for index, query in enumerate(queries)
            ]
            # Same cross-cycle dedupe as the web pool: a recurring fallback
            # query must not regenerate yesterday's story.
            llm_unseen = await asyncio.to_thread(
                dreams_db.seen_filter_unseen, [c.url for c in llm_pool])
            picked = [c for c in llm_pool
                      if c.url in llm_unseen and c.url not in existing_urls
                      ][:slots]
            notes.append(
                "web search unavailable; stories generated from llm knowledge")

        results: list[tuple[Candidate, StoryResult]] = []
        for candidate in picked:
            if chat is None:
                result = StoryResult(
                    title=candidate.title, body="", kind="content",
                    event_date=None, location=None, status="failed",
                    error="chat provider unavailable")
            else:
                result = await asyncio.to_thread(
                    story_service.generate_story, chat,
                    candidate=candidate, snapshot=snap)
            await asyncio.to_thread(
                _record_story, dreams_db, local_date, collection_id,
                candidate=candidate, result=result,
                matched_topics=_matched_topics(snap["topics"], candidate),
                query=_attributed_query(candidate, queries),
                count_call=chat is not None)
            results.append((candidate, result))

        # Every PICKED candidate's URL is upserted into the seen ledger,
        # whatever its outcome — surfaced-but-unpicked URLs are not recorded
        # here, and an empty or failed story still means "we already found
        # this".
        if results:
            await asyncio.to_thread(
                dreams_db.seen_upsert,
                [(discovery.normalize_url(c.url),
                  hashlib.sha256(c.title.encode()).hexdigest()[:16])
                 for c, _ in results])

        if append_mode:
            combined_notes = "; ".join(
                part for part in (prior_notes, "; ".join(notes) or None)
                if part) or None
        else:
            combined_notes = "; ".join(notes) or None
        final, total_rows = await asyncio.to_thread(
            _finish_cycle, dreams_db, collection_id,
            degradation_notes=combined_notes,
            provider=getattr(chat, "provider", None),
            model=getattr(chat, "model", None),
            completed_at=to_utc_iso(deps.now()))
        logger.info("Dreams cycle {} finished {}: {} story row(s) total",
                    collection_id, final, total_rows)
        return {"collection_id": collection_id, "status": final,
                "stories": total_rows}
    finally:
        _ACTIVE_CYCLE_DATES.discard(local_date)


def _catchup_due(
    dreams_db: DreamsDB, *, local_date: str, cadence_hours: int,
    now: datetime,
) -> bool:
    """The catch-up predicate's SQLite reads; runs in one ``to_thread`` hop.

    Due means: no collection for today's local date AND (no completed
    collection inside the cadence window, or the latest collection failed —
    including rows the stale-reclaim just converted). A ``complete`` or
    ``partial`` row counts as a cycle that happened.
    """
    with dreams_db.connection() as conn:
        today = conn.execute(
            "SELECT 1 FROM dreams_collections WHERE local_date = ?",
            (local_date,),
        ).fetchone()
        if today is not None:
            return False
        latest = conn.execute(
            "SELECT status FROM dreams_collections"
            " ORDER BY local_date DESC, id DESC LIMIT 1"
        ).fetchone()
        if latest is None:
            return True
        if str(latest["status"]) == "failed":
            return True
        cutoff = to_utc_iso(now - timedelta(hours=cadence_hours))
        recent = conn.execute(
            "SELECT 1 FROM dreams_collections"
            " WHERE status IN ('complete', 'partial') AND completed_at >= ?"
            " LIMIT 1",
            (cutoff,),
        ).fetchone()
        return recent is None


async def run_catchup_if_due(deps: CycleDeps) -> bool:
    """Run a catch-up cycle when today's collection is missing and due.

    The spec's boot/surface catch-up (spec §date-bucket idempotency):
    enabled + catch-up allowed + no collection for today's local date + the
    cadence says one is due (or the latest attempt failed after the stale
    reclaim) → ``run_cycle(trigger="catchup")``.

    Args:
        deps: Injected collaborators (see :class:`CycleDeps`).

    Returns:
        Whether a catch-up cycle was started.
    """
    if not dreams_setting("enabled") or not dreams_setting("catchup_enabled"):
        return False
    now = deps.now()
    local_date = _local_date(now)
    dreams_db = deps.dreams_db
    cutoff = to_utc_iso(now - timedelta(minutes=_STALE_GENERATING_MINUTES))
    await asyncio.to_thread(dreams_db.fail_stale_generating, cutoff)
    due = await asyncio.to_thread(
        _catchup_due, dreams_db, local_date=local_date,
        cadence_hours=int(dreams_setting("cadence_hours")), now=now)
    if not due:
        return False
    logger.info("Dreams catch-up due for {}", local_date)
    await run_cycle(deps, trigger="catchup")
    return True
