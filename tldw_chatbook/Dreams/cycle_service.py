"""[dreams] Cycle orchestration: one dated collection per run.

``run_cycle`` is the whole discovery pipeline of spec §discovery staged
end-to-end — reclaim stale rows, refresh the interest profile from
notes/media/Personal-Context signals (ruling R18), apply the feedback loop
to topic weights (ruling R19) and to goal query angles (task-33165:
goals never change weight, their feedback steers the angle), snapshot the
profile, claim the local date,
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
import sqlite3
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
    track_service,
)
from tldw_chatbook.Dreams.discovery import Candidate
from tldw_chatbook.Dreams.settings import dreams_execution_allowed, dreams_setting
from tldw_chatbook.Dreams.story_service import StoryResult
from tldw_chatbook.Utils.input_validation import validate_dreams_profile_text
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
#: ``kept``/``dived``/``ingested``/``tracked`` -- the Phase 2 Track flip:
#: tracking a story is a positive signal) or −1 (``less``); ``exported`` is
#: recorded but carries no weight signal. The per-reaction step and clamp
#: live with the snapshot that applies them
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
        dispatch_getter: Returns the ``NotificationDispatchService`` or
            ``None``; read only by the Phase 2 track check
            (``track_service.run_track_check``) to deliver a ``changed``
            verdict's notification. ``None`` (the default) means a run row
            without delivery -- existing cycle construction is unaffected.
        subs_service_getter: Returns the app's ``LocalWatchlistsService``
            or ``None``; read only by the Phase 2 lifecycle sweep
            (``track_service.sweep_track_lifecycle``) to disable the
            dream-created subscription of a wrapper it retires (Qodo #1,
            PR #2890). ``None`` (the default) keeps the legacy behavior --
            retired page items keep their subscription -- never an error.
        scheduling_db_getter: Returns the ``ScheduledTasksDB`` or
            ``None``; read only by the Phase 2 lifecycle sweep and untrack
            to disable the one-time reminder linked to a retired tracked
            item (Qodo #13, PR #2890). ``None`` (the default) leaves
            linked reminders enabled.
        guardian_db_getter: Returns the Guardian DB or ``None``; read only
            by the profile refresh, whose Guardian reader
            (``profile_sources.read_guardian_topics``) contributes
            AGGREGATE topic counts from feeds_discovery rules' alerts
            (ADR-204 contract 3). ``None`` (the default) means no
            Guardian signal -- a disabled Guardian never feeds Dreams.
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
    dispatch_getter: Callable[[], Any] | None = None
    subs_service_getter: Callable[[], Any] | None = None
    scheduling_db_getter: Callable[[], Any] | None = None
    guardian_db_getter: Callable[[], Any] | None = None


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
    claim: tuple[str, str] | None = None,
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
    if not dreams_db.set_collection_status(
        collection_id, final, provider=provider, model=model,
        degradation_notes=degradation_notes, completed_at=completed_at,
        story_count=total, claim=claim,
    ) and claim is not None:
        # The id names another collection now (round 4, I5): this cycle's
        # verdict is not that collection's to carry.
        raise sqlite3.IntegrityError(
            "dream collection is no longer the one this cycle claimed")
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

    ``discovery.run_queries`` stamps each web candidate with the query
    that surfaced it, and an llm-knowledge candidate is built from its
    query, so the answer is normally exact (``candidate.query``).
    Watchlist items were not surfaced by any query (""). Only a web
    candidate that arrives without its query (a caller building
    candidates by hand) falls back to the run's best word-overlapping
    query (ties keep the first).
    """
    if candidate.query:
        return candidate.query
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
    goal: str | None = None,
    claim: tuple[str, str] | None = None,
) -> None:
    """Persist one story outcome row (one thread hop).

    Plain synchronous SQLite only — the cycle runs it under
    ``asyncio.to_thread``. The story's model call is NOT counted here: its
    budget slot was reserved before the call was made (review of #2921,
    item 6).
    """
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
        goal=goal,
        claim=claim,
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
    order (notes, media, guardian, personal context) breaking ties. The
    reader's own ``source`` value is preferred when it sets one, with the
    origin name as the fallback.
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
            if (best is None
                    or (best[1] == "guardian" and source != "guardian")
                    or (source != "guardian" and weight > best[0])
                    or (best[1] == "guardian" and weight > best[0])):
                winners[key] = (weight, source)
    return {key: source for key, (_, source) in winners.items()}


def _stored_topic_text(text: str) -> str:
    """The text a derived topic is stored under, from its raw label.

    Exactly the path the signal refresh takes a label along:
    :func:`_stored_form_signals` puts it into the profile text rule's
    stored form, ``interest_profile.merge_signals`` strips and lowercases
    it, and :func:`_upsert_profile_signals` builds the key it stores the
    row under with THIS function. Anything that compares a raw label with
    a stored topic must go through here (follow-up to #2921: the Guardian
    consent check compared raw labels and halted every cycle whose
    Guardian topic the stored form had changed). Pinned by a test that
    runs the real refresh steps into a database and reads the row back.

    Raises:
        ValueError: The label fails the text rule; it is never stored.
    """
    return validate_dreams_profile_text(
        validate_dreams_profile_text(text).strip().lower())


def _stored_form_signals(
    collected: list[tuple[str, list[dict]]],
) -> tuple[list[tuple[str, list[dict]]], int]:
    """Every source's signal rows in the form they will be stored in.

    Review of #2921, round 4, I2. The text rule -- and with it the stored
    (NFC) spelling -- used to be applied after the sources were merged.
    One keyword arriving composed from notes and decomposed from media
    was then two merged rows with ONE stored key: upserted twice, the
    last write winning (1.0 from notes and 0.25 from media stored 0.25,
    source media). Each row is therefore put into its stored form before
    anything is merged or attributed, so the merge sums the two and the
    heavier source names the row.

    Returns:
        ``(collected, dropped)``: the same origins with the rows that
        pass the rule, their ``text`` replaced by the stored form; and
        how many rows failed it.
    """
    normalised: list[tuple[str, list[dict]]] = []
    dropped = 0
    for origin, rows in collected:
        kept: list[dict] = []
        for row in rows:
            try:
                kept.append({**row, "text": validate_dreams_profile_text(
                    str(row.get("text", "")))})
            except ValueError:
                dropped += 1
        normalised.append((origin, kept))
    return normalised, dropped


def _upsert_profile_signals(
    dreams_db: DreamsDB,
    merged: list[dict],
    source_by_key: dict[tuple[str, str], str],
    *,
    now_iso: str,
    dropped: int = 0,
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

    A merged row whose text fails ``validate_dreams_profile_text`` is not
    stored at all (counted in one debug line, never quoted). ``dropped``
    is how many rows the caller already left out for the same reason
    (:func:`_stored_form_signals`), so the one line reports them all.
    """
    existing = dreams_db.list_profile()
    protected = {
        (str(row["facet"]), str(row["text"]))
        for row in existing
        if str(row.get("source")) in ("user", "seed")
    }
    independent = {
        (str(row["facet"]), str(row["text"]))
        for row in existing if str(row.get("source")) != "guardian"
    }
    # Guardian evidence expires with consent; ordinary derived topics decay.
    with dreams_db.transaction() as conn:
        conn.execute("DELETE FROM dream_interest_profile WHERE source = 'guardian'")
    stamped: list[tuple[str, str]] = []
    for row in merged:
        merged_key = (str(row["facet"]), str(row["text"]))
        try:
            # The rule a typed topic must pass (review of #2921, round 3,
            # item 9): these keywords come from notes, media, Personal
            # Context and Guardian -- some from ingested third-party
            # content -- and enter the same synthesis prompt and fallback
            # search queries. The validated (NFC) form is what is stored --
            # through _stored_topic_text, the ONE function that also puts
            # Guardian's labels into this form for the consent check, so
            # the two cannot drift apart (review of #3097). On merged text
            # it gives exactly what the rule alone gave here before.
            key = (merged_key[0], _stored_topic_text(merged_key[1]))
        except ValueError:
            dropped += 1
            continue
        source = source_by_key.get(merged_key, "notes")
        if key in protected or (source == "guardian" and key in independent):
            continue
        dreams_db.upsert_profile_entry(
            key[0], key[1],
            weight=float(row["weight"]), searchable=1,
            source=source,
        )
        stamped.append(key)
    if dropped:
        # The count only: the text is exactly what must not be passed on.
        logger.debug(
            "Dreams signal refresh: dropped {} derived topic(s) that fail "
            "the profile text rule", dropped)
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

    Reads the signal sources through the injected getters (a ``None``
    skips that source silently), merges them, and upserts the merged
    topics. Every source — and the write — degrades with a note on
    failure; the cycle always continues. Reads run one ``asyncio.to_thread``
    hop per source DB, and the Dreams write is one more hop (stage
    discipline).

    The Guardian source (ADR-204 contract 3) rides the same discipline:
    ``guardian_db_getter`` contributes AGGREGATE topic counts from
    feeds_discovery rules' alerts only (the gate lives in
    ``profile_sources.read_guardian_topics``'s join), written under origin
    ``'guardian'`` so the profile keeps the provenance of every row.
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
    from tldw_chatbook.Guardian.settings import guardian_setting

    guardian_getter = getattr(deps, "guardian_db_getter", None)
    guardian_db = (
        guardian_getter()
        if guardian_getter is not None and guardian_setting("enabled") else None
    )
    if guardian_db is not None:
        try:
            collected.append(("guardian", await asyncio.to_thread(
                profile_sources.read_guardian_topics, guardian_db)))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"profile signals: guardian failed: {exc}")
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
    # Stored form FIRST, then merge (round 4, I2).
    collected, dropped = _stored_form_signals(collected)
    merged = interest_profile.merge_signals([rows for _, rows in collected])
    try:
        await asyncio.to_thread(
            _upsert_profile_signals, deps.dreams_db, merged,
            _preferred_sources(collected), now_iso=to_utc_iso(now),
            dropped=dropped)
    except Exception as exc:  # noqa: BLE001 - a failed write degrades
        notes.append(f"profile signals: profile write failed: {exc}")
    return notes


async def _guardian_snapshot_authorized(deps: CycleDeps, snap: dict) -> bool:
    return await asyncio.to_thread(_guardian_snapshot_authorized_sync, deps, snap)


def _guardian_snapshot_authorized_sync(deps: CycleDeps, snap: dict) -> bool:
    """Own lazy construction and reading together on the current worker."""
    if not any(row.get("source") == "guardian" for row in snap.get("topics", [])):
        return True
    from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker

    return run_finite_local_worker(_read_guardian_snapshot_authority, deps, snap)


def _read_guardian_snapshot_authority(deps: CycleDeps, snap: dict) -> bool:
    """Refuse an outbound snapshot whose Guardian consent was revoked.

    Both sides of the subset test are stored-form text
    (:func:`_stored_topic_text`): the snapshot holds Guardian topics as
    they were stored, so Guardian's current labels are put into the same
    form before they are compared. A label that fails the text rule is
    left out of the consented set -- that cannot widen consent: such a
    label is dropped by the refresh too, so it is never stored and never
    in a snapshot, and leaving it out only makes the set smaller. A
    snapshot topic that does not pass the rule itself (a row an earlier
    build stored) is uncertain consent, which never grants authority.
    """
    try:
        topics = {
            _stored_topic_text(str(row["text"]))
            for row in snap.get("topics", []) if row.get("source") == "guardian"
        }
    except ValueError:
        return False
    if not topics:
        return True
    from tldw_chatbook.Guardian.settings import guardian_setting

    try:
        if not guardian_setting("enabled") or deps.guardian_db_getter is None:
            return False
        db = deps.guardian_db_getter()
        if db is None:
            return False
        from tldw_chatbook.DB.base_db import operation_owned_connection

        with operation_owned_connection(db):
            current = profile_sources.read_guardian_topics(db)
        consented: set[str] = set()
        for row in current:
            try:
                consented.add(_stored_topic_text(str(row["text"])))
            except ValueError:
                continue  # never stored, so never in a snapshot (above)
        return bool(guardian_setting("enabled")) and topics <= consented
    except Exception:  # noqa: BLE001 - uncertain consent never grants authority
        return False


#: Feedback kinds carrying a POSITIVE signal (ruling R19): they net +1 on
#: the topic-weight path and steer a matched goal's angle toward "prefer".
_POSITIVE_FEEDBACK_KINDS = ("more", "kept", "dived", "ingested", "tracked")

#: Story kinds a query angle may name (task-33165 R1); ``unknown`` stories
#: steer as ``content``.
_ANGLE_STORY_KINDS = ("content", "event", "deal", "social_opportunity")


def _feedback_window_rows(dreams_db: DreamsDB, *, now: datetime) -> list:
    """The trailing window's feedback rows joined to stories, oldest first.

    One parameterized read shared by BOTH feedback halves (task-33165): the
    topic-weight net and the goal query-angle pass walk the same rows, so
    they can never disagree about what the window contains. Ordered by
    ``(created_at, id)`` so the angle pass's replacement semantics ("any
    positive after a negative replaces it") read chronologically even when
    reactions share a timestamp (insertion order breaks ties).
    """
    cutoff = to_utc_iso(now - timedelta(days=_FEEDBACK_WINDOW_DAYS))
    with dreams_db.connection() as conn:
        return conn.execute(
            "SELECT ds.matched_topics AS matched, ds.kind AS story_kind,"
            " ds.goal AS goal, fb.kind AS kind"
            " FROM dream_feedback AS fb"
            " JOIN dream_stories AS ds ON ds.id = fb.story_id"
            " WHERE fb.created_at >= ?"
            " ORDER BY fb.created_at, fb.id",
            (cutoff,),
        ).fetchall()


def _matched_texts(row: Any) -> list[str]:
    """Decode one window row's matched-topics JSON; junk decodes to []."""
    try:
        matched = json.loads(row["matched"] or "[]")
    except ValueError:
        return []
    return [str(topic) for topic in matched] if isinstance(matched, list) else []


def _net_from_rows(rows: list) -> dict[str, int]:
    """Net each row's reactions per normalized topic (pure; nothing written).

    +1 per ``more``/``kept``/``dived``/``ingested``/``tracked`` -- tracking
    a story is a positive signal, the Phase 2 Track flip -- −1 per ``less``;
    ``exported`` is neutral. ``interest_profile.snapshot`` applies the
    result as an offset, so a reaction counts exactly once per cycle for as
    long as it is inside the window -- never compounding into the stored
    weight -- and a derived topic's refresh cannot erase it.
    """
    net: dict[str, int] = {}
    for row in rows:
        kind = str(row["kind"])
        delta = (1 if kind in _POSITIVE_FEEDBACK_KINDS
                 else -1 if kind == "less" else 0)
        if delta == 0:
            continue
        for topic in _matched_texts(row):
            topic = topic.strip().lower()
            if topic:
                net[topic] = net.get(topic, 0) + delta
    return {topic: value for topic, value in net.items() if value}


def _feedback_net(dreams_db: DreamsDB, *, now: datetime) -> dict[str, int]:
    """Net reactions per matched topic over the trailing window (sync).

    Joins ``dream_feedback`` → ``dream_stories.matched_topics`` and nets
    each normalized topic. Nothing is written: the snapshot applies the
    result as an offset. Goals are immune because the snapshot only offsets
    ``facet='topic'`` rows.

    Args:
        dreams_db: The Dreams database.
        now: Cycle clock; the window ends here.

    Returns:
        ``{normalized_topic: net}`` with zero nets omitted.
    """
    return _net_from_rows(_feedback_window_rows(dreams_db, now=now))


def _write_goal_angles(dreams_db: DreamsDB, rows: list) -> None:
    """Steer goals' ``query_angle`` from window rows (task-33165).

    The OTHER feedback half: a goal never changes weight -- feedback on a
    story that a goal's own search query surfaced instead writes that
    goal's query angle, a short steering note derived from the reaction
    and the story's kind: ``less`` records ``"avoid: {kind}"`` and any
    positive kind records ``"prefer: {kind}"``; ``exported`` and
    unrecognized reactions are neutral. An ``unknown`` story steers as
    ``content``.

    Which goal a story belongs to is PROVENANCE, read from the story row's
    ``goal`` column (review of #2921, item 1): the cycle records the goal
    the generating query was written for and nothing else. It is never
    inferred from the story's text or from ``matched_topics`` -- a shared
    word ("to", "in") made every goal match every story, and a topic may
    carry the same text as a goal.

    Rows arrive oldest-first and each write REPLACES the previous note
    (plain UPDATE, no accumulation), so the newest non-neutral reaction in
    the window wins -- a positive after a negative replaces it and vice
    versa. The recorded goal joins the profile on its EXACT stored text --
    the key the profile itself uses, and what the cycle records -- so two
    goals that differ only by case stay two goals (round 3, N3: a
    lowercased key collapsed them and the angle landed on whichever was
    listed last). A goal that has since been removed matches no row.

    Args:
        dreams_db: The Dreams database.
        rows: ``_feedback_window_rows`` output, oldest first.
    """
    goals = {
        str(row["text"])
        for row in dreams_db.list_profile()
        if row.get("facet") == "goal"
    }
    if not goals:
        return
    updates: dict[str, str] = {}
    for row in rows:
        goal = str(row["goal"] or "")
        if goal not in goals:
            continue
        kind = str(row["kind"])
        if kind not in _POSITIVE_FEEDBACK_KINDS and kind != "less":
            continue
        story_kind = str(row["story_kind"] or "unknown")
        if story_kind not in _ANGLE_STORY_KINDS:
            story_kind = "content"
        angle = (f"prefer: {story_kind}" if kind in _POSITIVE_FEEDBACK_KINDS
                 else f"avoid: {story_kind}")
        updates[goal] = angle
    for text, angle in updates.items():
        dreams_db.set_goal_query_angle("goal", text, angle=angle)


def _apply_goal_angles(dreams_db: DreamsDB, *, now: datetime) -> None:
    """Read the window once, then steer matched goals' angles (sync)."""
    _write_goal_angles(dreams_db, _feedback_window_rows(dreams_db, now=now))


async def _apply_feedback(
    deps: CycleDeps, now: datetime
) -> tuple[dict[str, int], list[str]]:
    """Stage 0b (rulings R19 + task-33165): read recent story feedback.

    One ``asyncio.to_thread`` hop runs BOTH feedback halves against one
    window read: the topic net the snapshot applies as a weight offset, and
    the goal query-angle writes (goals never change weight; their feedback
    steers the angle instead). Any failure degrades with a note and never
    aborts the cycle; an angle-write failure does not take the topic
    offset down with it (independent mechanisms, separate notes).

    Returns:
        ``(net_by_topic, degradation_notes)``.
    """

    def stage() -> tuple[dict[str, int], list[str]]:
        rows = _feedback_window_rows(deps.dreams_db, now=now)
        notes: list[str] = []
        try:
            _write_goal_angles(deps.dreams_db, rows)
        except Exception as exc:  # noqa: BLE001 - angles degrade alone
            notes.append(f"goal angle feedback failed: {exc}")
        return _net_from_rows(rows), notes

    try:
        return await asyncio.to_thread(stage)
    except Exception as exc:  # noqa: BLE001 - the loop degrades, not the cycle
        return {}, [f"feedback loop failed: {exc}"]


async def run_cycle(deps: CycleDeps, *, trigger: str) -> dict:
    """Run one full discovery cycle for the current local date.

    Before the profile snapshot, two stages personalize it: the interest
    profile is refreshed from notes/media/Personal-Context signals (ruling
    R18) and recent story feedback is netted per topic and applied by the
    snapshot as a weight offset (ruling R19) — both degrade with notes and
    never abort the cycle.

    The master switch binds the run itself (Qodo #15, PR #2921): with
    ``[dreams] enabled`` off the cycle does not start (``"disabled"``,
    nothing claimed, nothing spent), and a cycle already running re-reads
    the flag before query synthesis, before EACH search, and before each
    story -- once it is off, no further search or model call starts, the
    stories already written are kept, and the collection row is closed
    with a degradation note rather than left ``generating``. A cycle
    switched off before it wrote ANY story reports ``"disabled"`` too and
    leaves no collection behind (review item 7): the date stays open, so
    switching Dreams back on runs the cycle without waiting out a cadence.

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
        ``status`` is the collection's final status, or one of the
        no-run markers ``"disabled"``/``"inflight"``/``"date_taken"``.
    """
    # The master switch is checked HERE, where every trigger arrives, and
    # again before each spend below (Qodo #15, PR #2921): callers gate on
    # ``[dreams] enabled`` before they start a cycle, but a cycle runs for
    # minutes and the Settings toggle only writes the flag, so one already
    # in flight kept searching and calling the model after Dreams was
    # switched off.
    if not dreams_setting("enabled") or not dreams_execution_allowed(deps.dreams_db):
        return {"collection_id": None, "status": "disabled", "stories": 0}
    now = deps.now()
    local_date = _local_date(now)
    dreams_db = deps.dreams_db

    # Event-loop-only claim: no await between the check and the add, so a
    # second dispatch for the same date can never interleave past the guard.
    if local_date in _ACTIVE_CYCLE_DATES:
        return {"collection_id": None, "status": "inflight", "stories": 0}
    _ACTIVE_CYCLE_DATES.add(local_date)
    halted = False

    def switched_off() -> bool:
        """Whether Dreams was switched off since this cycle started.

        Read live before each spend (synthesis, each search, each
        story), and sticky once true: a halted cycle starts no further
        search or model call and goes straight to closing its collection
        row. A call already in flight is allowed to finish -- a worker
        thread cannot be recalled -- and what it produced is kept.
        """
        nonlocal halted
        halted = halted or not dreams_setting("enabled") or not dreams_execution_allowed(deps.dreams_db)
        return halted

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
        # Track lifecycle sweep (Phase 2 Task 6, stage 0 with the other
        # maintenance passes): retire event-passed and quiet watches, pause
        # repeatedly failing ones. Degrade-never-abort -- a sweep exception
        # is a note on the collection, not a dead cycle -- and the sweep's
        # own notes ride the same degradation channel. The service getters
        # let a retirement also disable its dream-created subscription and
        # linked reminder (Qodo #1/#13, PR #2890); both are optional, and
        # the sweep degrades without them.
        stage_notes: list[str] = []
        try:
            stage_notes += await track_service.sweep_track_lifecycle(
                dreams_db, now=now,
                subs_service_getter=getattr(
                    deps, "subs_service_getter", None),
                scheduling_db_getter=getattr(
                    deps, "scheduling_db_getter", None),
            )
        except Exception as exc:  # noqa: BLE001 - the sweep degrades, never aborts
            stage_notes.append(f"track sweep failed: {exc}")
        # What a resume of the master switch could not finish -- above all
        # a tracked page left off because its name could not be looked up
        # then -- gets another try here, once per cycle (review round 4,
        # B2). Needs the watchlists service; degrades like the sweep.
        try:
            subs_service_getter = getattr(deps, "subs_service_getter", None)
            subs_service = (
                subs_service_getter() if subs_service_getter else None)
            if subs_service is not None:
                # ...and, on the same daily visit, every source Dreams
                # owns is checked for its egress mark from its real
                # options (round 4, B4).
                await track_service.backfill_egress_marks(
                    subs_service, dreams_db, verify=True)
                await track_service.retry_unfinished_resume(
                    subs_service, dreams_db,
                    scheduling_db_getter=getattr(
                        deps, "scheduling_db_getter", None),
                    now=now)
        except Exception as exc:  # noqa: BLE001 - degrades, never aborts
            stage_notes.append(f"track resume retry failed: {exc}")
        # Profile stages (rulings R18/R19) run BEFORE the snapshot so it
        # reflects refreshed signals and the feedback offsets; both degrade
        # with notes and never abort the cycle. Their notes are carried into
        # the collection row once it exists.
        stage_notes += await _refresh_profile_signals(deps, now)
        feedback_net, feedback_notes = await _apply_feedback(deps, now)
        stage_notes += feedback_notes
        snap = await asyncio.to_thread(
            interest_profile.snapshot, dreams_db, now_epoch=now.timestamp(),
            feedback=feedback_net)
        async def may_spend() -> bool:
            nonlocal halted
            if switched_off():
                return False
            authorized = await _guardian_snapshot_authorized(deps, snap)
            halted = halted or not authorized
            return not switched_off()

        def invocation_allowed() -> bool:
            nonlocal halted
            if switched_off():
                return False
            halted = halted or not _guardian_snapshot_authorized_sync(deps, snap)
            return not switched_off()

        profile_digest = hashlib.sha256(
            json.dumps(snap, sort_keys=True).encode()).hexdigest()[:16]
        collection_id = await asyncio.to_thread(
            dreams_db.create_collection, local_date, trigger, profile_digest)
        # What this cycle holds is not an id but a ROW: ids are reused
        # after a halted cycle's empty collection is deleted, so every
        # later write names the row by its date and creation stamp as
        # well (review of #2921, round 4, I5).
        claim: tuple[str, str] | None = None
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
            claim = (local_date, str(target["created_at"]))
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
        if claim is None:
            created = await asyncio.to_thread(
                dreams_db.get_collection_by_date, local_date)
            if created is not None and int(created["id"]) == collection_id:
                claim = (local_date, str(created["created_at"]))
        logger.info("Dreams cycle {} for {} (trigger={}, append={})",
                    collection_id, local_date, trigger, append_mode)
        queries: list[str] = []
        # Goal provenance (review of #2921, item 1): each synthesized or
        # fallback query row names the goal it was written for; this maps
        # the query string to that goal so a story can be attributed to
        # the goal whose query surfaced it -- exactly, never by overlap.
        # The rows are distinct queries (``query_synthesis._distinct``), so
        # a string names exactly one row.
        query_rows: list[dict] = []
        chat: Callable[..., Any] | None = None

        # Every spend is RESERVED before the work it pays for (review of
        # #2921, item 6), one slot at a time, right before the search or
        # model call it pays for: ``usage_reserve`` is one atomic
        # check-and-count, shared with the track checks, so a check
        # running while this cycle is mid-search or mid-story sees the
        # slot as taken. Reading the usage first and bumping it after the
        # work let the two of them spend the same last slot.
        query_count = int(dreams_setting("queries_per_cycle"))
        max_llm_calls = int(dreams_setting("max_llm_calls_per_day"))
        try:
            # Switched off already (Qodo #15, PR #2921): no provider is
            # resolved and nothing is synthesized -- with no chat and no
            # queries every later stage has nothing to spend on.
            if not switched_off():
                # Thread-offloaded: the getter force-reloads config from
                # disk and may reach the keyring — neither belongs on the
                # event loop.
                chat = await asyncio.to_thread(deps.chat_getter)
                if not await may_spend():
                    chat = None
                if chat is not None:
                    chat = discovery.guard_invocation(
                        chat, invocation_allowed, async_allowed=may_spend)
                # The synthesis call spends the same daily llm budget as
                # stories: its slot is taken BEFORE the call, and an
                # exhausted budget searches the deterministic fallback
                # queries and makes no model call at all.
                if chat is not None and not await asyncio.to_thread(
                        dreams_db.usage_reserve, local_date,
                        counter="llm_calls", limit=max_llm_calls):
                    # The labeled fallback rows (Phase 2 Task 1); their
                    # internal filter already dropped unsearchable goals.
                    query_rows = query_synthesis.preview_queries(
                        [str(t["text"]) for t in snap["topics"]],
                        snap.get("goals", []), count=query_count)
                    notes.append("llm budget exhausted: fallback queries, no "
                                 "synthesis call")
                elif await may_spend():
                    query_rows = await query_synthesis.synthesize_query_rows(
                        chat, snapshot=snap, count=query_count,
                        exploration_slots=int(
                            dreams_setting("exploration_slots")))
        except Exception as exc:  # noqa: BLE001 - provider unavailable, degrade
            notes.append(f"query synthesis failed: {exc}")
        queries = [str(row["query"]) for row in query_rows]
        goal_by_query = {
            str(row["query"]): str(row["goal"])
            for row in query_rows if row.get("goal")
        }

        # A PLAN, not a spend: how much of the day is left to aim for. The
        # slots themselves are reserved below, right before each use.
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
        if switched_off():
            # Switched off during synthesis: no search goes out.
            pass
        elif slots <= 0:
            # A story needs a model call. With no llm slot left for one,
            # a search would be made, counted and thrown away (review of
            # #2921, round 3, item 4); the track check has the same guard
            # (``_reserve_check_search``).
            notes.append(
                "llm budget exhausted: no searches made, no story could "
                "be written from them")
        elif (dreams_setting("web_search_enabled") and budget.searches > 0
                and queries):
            max_searches = int(dreams_setting("max_searches_per_day"))

            async def may_search() -> bool:
                """Whether the next search may go out; asked before each.

                One live read of the master switch, then ONE reserved
                slot -- exactly as each story reserves one llm slot
                (review of #2921, round 3, item 2). Reserving the whole
                batch up front left searches counted that a switch-off or
                a cancellation never sent, and the switch was read once
                for the batch, so the rest of it still went out.

                The bound that is left (round 4, I4): ONE slot can stay
                counted for a search that is never sent -- when the task
                is cancelled, or the app closed, while the reservation is
                on its thread hop or between it and the send. It is not
                handed back: a refund path is one more thing to get wrong
                for at most one unit per cycle. The same holds for the one
                model call each story reserves, and for a tracked check's
                one search and one judge call. The guide's budget
                paragraph says so.
                """
                if switched_off():
                    return False
                reserved = await asyncio.to_thread(
                    dreams_db.usage_reserve, local_date,
                    counter="searches", limit=max_searches)
                return bool(reserved) and await may_spend()

            found, sent = await discovery.run_queries(
                discovery.guard_invocation(deps.perform_search, invocation_allowed),
                engine=str(dreams_setting("search_engine")),
                queries=queries, before_each=may_search)
            if sent < len(queries) and not halted:
                # The plan above counted on these, but another Dreams
                # task took the slots meanwhile: stop at the cap.
                notes.append(
                    f"search budget: stopped after {sent} of "
                    f"{len(queries)} searches (spent by other Dreams work "
                    "meanwhile)")
            # Only what was searched is a query of this cycle from here on.
            queries = queries[:sent]
            for candidate in found:
                # The goal rides the candidate (round 3, N2) so the
                # duplicate-URL merge below can keep it.
                candidate.goal = goal_by_query.get(candidate.query)
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
        if (not candidates and queries and slots > 0 and chat is not None
                and not halted):
            llm_pool = [
                Candidate(title=query, snippet="", source="llm",
                          url=f"{_LLM_URL_PREFIX}{index}/"
                              f"{quote(query, safe='')}", query=query,
                          goal=goal_by_query.get(query))
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
            if switched_off():
                # One live read per story: the model call about to start
                # is the cycle's main spend. Stories already written stay.
                break
            if chat is None:
                result = StoryResult(
                    title=candidate.title, body="", kind="content",
                    event_date=None, location=None, status="failed",
                    error="chat provider unavailable")
            else:
                if not await asyncio.to_thread(
                        dreams_db.usage_reserve, local_date,
                        counter="llm_calls", limit=max_llm_calls):
                    # The plan above counted on this call, but another
                    # Dreams task took it meanwhile: stop at the cap.
                    notes.append(
                        "llm budget: spent by other Dreams work mid-cycle; "
                        "no further stories")
                    break
                if not await may_spend():
                    break
                result = await asyncio.to_thread(
                    story_service.generate_story, chat,
                    candidate=candidate, snapshot=snap)
                if halted:
                    break
            await asyncio.to_thread(
                _record_story, dreams_db, local_date, collection_id,
                candidate=candidate, result=result,
                matched_topics=_matched_topics(snap["topics"], candidate),
                query=_attributed_query(candidate, queries),
                # Only a query that found this page attributes a goal; a
                # watchlist item (no query) names none.
                goal=candidate.goal, claim=claim)
            results.append((candidate, result))

        if halted and not results:
            # Switched off before this run produced a story (review of
            # #2921, item 7). Nothing happened that a collection could
            # truthfully record, and closing one as ``failed`` would also
            # cost the user a cadence when they switch Dreams back on (a
            # failed row is an attempt the projection retries one cadence
            # later, and a date that owns a row is skipped by the boot
            # catch-up). So the run is reported as not run: an existing
            # collection it would have appended to is left exactly as it
            # was, and the row this run claimed is handed back. What was
            # already spent stays counted in the day's usage.
            logger.info(
                "Dreams cycle {} for {}: switched off before its first "
                "story; nothing recorded", collection_id, local_date)
            if append_mode:
                return {"collection_id": collection_id,
                        "status": str(target["status"]),
                        "stories": int(target["story_rows"])}
            await asyncio.to_thread(
                dreams_db.delete_empty_collection, collection_id)
            return {"collection_id": None, "status": "disabled", "stories": 0}
        if halted:
            notes.append(
                "dreams disabled mid-cycle: stopped before further searches "
                "or model calls")

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
            completed_at=to_utc_iso(deps.now()), claim=claim)
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
    if (not dreams_setting("enabled") or not dreams_setting("catchup_enabled")
            or not dreams_execution_allowed(deps.dreams_db)):
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
