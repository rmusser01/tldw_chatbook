"""Turn a briefing selection into a stored briefing (spec #2 phase 1).

This module is the pipeline's only writer. `briefing_selection` answers
"what does the next briefing cover?" and writes nothing; this module inserts
the `briefings` row, builds the prompt, makes exactly one chat call, and
records the outcome -- or records the failure. Three rules shape it:

**1. Every outcome is a row.** `generating` -> `complete`, `empty`, or
`failed` (+ error text). An empty window is a visible row saying "nothing
arrived", not an absent artifact the user has to interpret; a provider
outage is a row carrying the provider's own message. Silence is never a
state (spec §Error-handling ethos). There are no new `persist_event` names
here: statuses *are* the observability, deliberately, because the ADR-029
amendment admits exactly six events and this design must not widen a
privacy boundary the owner signs.

**2. Failure never advances the coverage window.** A `failed` row records no
`covers_through_item_id` and no junction rows, so `latest_completed_watermark`
still returns the last *delivered* line and the next attempt re-selects the
same items. This is the spec's named invariant, and it is why the junction
rows are written on the success path only -- a briefing that never reached
the user has covered nothing.

**3. The overflow note is the service's, not the model's.** The prompt asks
for it, and the service appends it to the body regardless. A model that
ignores the instruction would otherwise turn a stated truncation into a
silent one, which is precisely the failure the cap was designed not to have.

Zombie recovery (`fail_interrupted_briefings`) lives here but is *not*
called by `generate_briefing`: the caller runs it before invoking generation
(and on Artifacts load), because the guard it unwedges -- one generation per
watchlist at a time -- is the caller's guard. Folding it into generation
would make the service both the thing being guarded and the guard.

Egress, stated plainly (spec §Egress): building the prompt sends item
titles, excerpts and diffs to whichever provider is configured. That is the
user's choice of provider; local providers are the private option. Nothing
here is logged with content -- and that claim is pinned by a test rather
than merely asserted here, because the obvious way to log a provider failure
breaks it: this app's file sink runs with `diagnose=True`, so
`logger.opt(exception=True)` dumps the failing frame's locals, and the frame
at the failure site holds the prompt. See
`test_a_failed_generation_logs_no_item_content`.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from loguru import logger

from ..Chat.Chat_Functions import chat_api_call, extract_response_content
from .briefing_selection import (
    MODE_AUTO_FEATURED,
    VALID_MODES,
    BriefingSelection,
    select_briefing_items,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..DB.Subscriptions_DB import SubscriptionsDB

#: Statuses a `briefings` row can hold. `generating` is the insert default.
STATUS_GENERATING = "generating"
STATUS_COMPLETE = "complete"
STATUS_EMPTY = "empty"
STATUS_FAILED = "failed"

#: The error text a zombie row is failed with, so the UI can say *why*.
INTERRUPTED_ERROR = "interrupted"

#: Per-item body cap, in characters (spec §4: "per-item excerpt cap").
#: Keeps one scraped page from consuming the whole context window and
#: turning a bounded call into what looks like a provider outage.
EXCERPT_CHAR_CAP = 800

#: Completion budget for the single call.
BRIEFING_MAX_TOKENS = 2000

#: Low but non-zero: a briefing is summarization, not creative writing.
BRIEFING_TEMPERATURE = 0.3

#: Provider errors can carry a whole HTML error page. The row holds a
#: message a human reads in a status line, not a document.
ERROR_CHAR_CAP = 1000

_TRUNCATION_MARKER = "\n[... truncated at {cap} characters ...]"

_SYSTEM_PROMPT = """\
You are writing a briefing over items collected by a user's watchlist.

Write markdown. Group related items under short thematic headings rather \
than listing every item in turn. Cite every claim to the item it came from \
using its bracketed id exactly as given, e.g. [item 42]; never invent an id.

The items are of two kinds and must not be conflated:

- An **article** item is something a source published. Report what it says.
- A **page change** item is a diff of a monitored web page. Report what \
changed on that page -- the page did not publish an article, it was edited. \
Added lines begin with '+' and removed lines with '-'.

Items listed under "Queued by you" were picked out by the user and must be \
covered first and in more detail than the rest. Write only from the items \
given; if the material does not support a claim, leave it out.
"""


def _overflow_note(count: int) -> str:
    """The exact sentence the spec puts in the body for dropped items."""
    if count == 1:
        return "1 more item arrived in this window and is not covered."
    return f"{count} more items arrived in this window and are not covered."


def _excerpt(body: Any) -> str:
    """Item body, capped, with the truncation stated rather than hidden."""
    text = str(body or "").strip()
    if not text:
        return "(no body captured)"
    if len(text) <= EXCERPT_CHAR_CAP:
        return text
    return text[:EXCERPT_CHAR_CAP] + _TRUNCATION_MARKER.format(cap=EXCERPT_CHAR_CAP)


def _item_block(item: Mapping[str, Any]) -> str:
    """One item's section of the prompt, shaped by its `content_kind`.

    TASK-1343 made `content` the *diff* for change items, so formatting a
    change item like an article hands the model a wall of `+`/`-` lines with
    nothing to say they are a page's edits -- and the briefing then reads as
    if the page had published them. The kind is therefore stated in words,
    not merely implied by the shape of the text.
    """
    item_id = item.get("item_id")
    title = str(item.get("title") or "Untitled")
    source = str(item.get("source_name") or "unknown source")
    lines = [f"### [item {item_id}] {title}"]

    if str(item.get("content_kind") or "") == "change":
        lines.append(
            f'Kind: page change. This is a diff of the monitored page "{title}" '
            f"({source}) -- it is not an article. Report what changed on the page."
        )
        headline = []
        percentage = item.get("change_percentage")
        if percentage is not None:
            try:
                headline.append(f"{float(percentage):.0f}% of the page changed")
            except (TypeError, ValueError):
                # Degrade by omitting the number rather than failing the
                # whole briefing over one headline field.
                pass
        if item.get("change_type"):
            headline.append(str(item["change_type"]))
        if item.get("diff_summary"):
            headline.append(str(item["diff_summary"]))
        if headline:
            lines.append("Change: " + "; ".join(headline))
        if item.get("url"):
            lines.append(f"Page: {item['url']}")
        lines.append("Diff:")
    else:
        lines.append("Kind: article")
        lines.append(f"Source: {source}")
        if item.get("url"):
            lines.append(f"URL: {item['url']}")
        if item.get("published_date"):
            lines.append(f"Published: {item['published_date']}")
        lines.append(f"Excerpt (up to {EXCERPT_CHAR_CAP} characters):")

    lines.append(_excerpt(item.get("content")))
    return "\n".join(lines)


def build_briefing_prompt(
    items: Sequence[Mapping[str, Any]],
    featured_ids: set[int],
    overflow_count: int,
) -> tuple[str, str]:
    """Build the (system, user) prompt for one briefing. Pure.

    Ordering is performed here rather than inherited from the caller: the
    builder is tested directly, and "featured first" is a property of the
    prompt, so it must hold for any input order.

    Args:
        items: Normalized watchlist item dicts (`briefing_selection`'s
            `items`). `item_id`, `title`, `source_name`, `url`,
            `content`, `content_kind` and the three change columns are read.
        featured_ids: Raw item ids the user queued; these lead the prompt
            under a "Queued by you" heading.
        overflow_count: Items the cap dropped. Stated in the prompt so the
            model can close on it -- and stated again by the service in the
            body, so the note survives a model that ignores this.

    Returns:
        `(system_prompt, user_prompt)`.
    """
    featured = [item for item in items if item.get("item_id") in featured_ids]
    rest = [item for item in items if item.get("item_id") not in featured_ids]

    sections: list[str] = []
    if featured:
        sections.append(
            "## Queued by you\n\n"
            "The user picked these out. Cover them first, and in more detail.\n\n"
            + "\n\n".join(_item_block(item) for item in featured)
        )
    if rest:
        heading = "## Also in this window" if featured else "## In this window"
        sections.append(heading + "\n\n" + "\n\n".join(_item_block(item) for item in rest))
    if overflow_count > 0:
        sections.append(
            "## Coverage note\n\n"
            f"{_overflow_note(overflow_count)} "
            "State this at the end of the briefing."
        )

    return _SYSTEM_PROMPT, "\n\n".join(sections)


#: The exact citation convention `_SYSTEM_PROMPT` asks the model to use --
#: `[item 42]`, digits only, "never invent an id". `extract_citation_ids` is
#: this convention's own parser (spec #2 phase 2a, Task 6): turning the same
#: bracketed ids the prompt asked the model to write back into ids a reader
#: can navigate to.
_CITATION_ID_PATTERN = re.compile(r"\[item (\d+)\]")


def extract_citation_ids(body_markdown: str) -> list[int]:
    """Every `[item N]` id a briefing body cites, in first-seen order.

    Pure: no I/O, and no opinion on whether any of these ids still resolve
    to a live `subscription_items` row -- that is entirely the caller's
    question (Task 6's `WatchlistsCollectionsScreen._load_briefings`, via
    `SubscriptionsDB.get_subscription_items_by_ids`), since only the caller
    has a database to ask. This function only reads the text the model
    wrote.

    `[item x]`/`[item]` (no digits) are not a citation under this prompt's
    own convention and are silently ignored: the model was never asked to
    write anything but a digit (see `_SYSTEM_PROMPT`), so treating a
    non-numeric bracket as a malformed citation would be inventing a case
    the prompt never produces.

    Args:
        body_markdown: A briefing's `body_markdown`, or any text. Read as
            plain text -- markdown syntax and Rich markup in it are never
            interpreted, only the literal `[item N]` substring is matched.

    Returns:
        Deduplicated ids, in the order they first appear in the text.
    """
    seen: set[int] = set()
    ordered: list[int] = []
    for match in _CITATION_ID_PATTERN.finditer(body_markdown or ""):
        item_id = int(match.group(1))
        if item_id not in seen:
            seen.add(item_id)
            ordered.append(item_id)
    return ordered


def _append_overflow(body: str, overflow_count: int) -> str:
    """Append the overflow sentence to the model's body.

    Deliberately unconditional on what the model wrote: asking for the note
    in the prompt makes it likely, appending it here makes it certain. A
    truncation the user is not told about is the exact failure the item cap
    would otherwise introduce.
    """
    if overflow_count <= 0:
        return body
    return f"{body.rstrip()}\n\n---\n\n{_overflow_note(overflow_count)}\n"


def _selection_mode(db: "SubscriptionsDB", watchlist_id: int) -> str:
    """The watchlist's stored selection mode, defaulting honestly.

    A NULL (a row predating the column) or an unrecognized value falls back
    to the create-time default rather than raising: an unknown mode would
    otherwise escape from `select_briefing_items` *after* the `generating`
    row was inserted, leaving exactly the zombie row this design goes out of
    its way to avoid.
    """
    row = db.conn.execute(
        "SELECT briefing_selection_mode FROM watchlists WHERE id = ?",
        (watchlist_id,),
    ).fetchone()
    mode = row["briefing_selection_mode"] if row is not None else None
    if mode in VALID_MODES:
        return str(mode)
    if mode:
        logger.warning(
            f"watchlist {watchlist_id} has unknown briefing_selection_mode {mode!r}; "
            f"using {MODE_AUTO_FEATURED!r}"
        )
    return MODE_AUTO_FEATURED


def _default_provider() -> str:
    """The app's configured default chat endpoint.

    Read from `config.default_api_endpoint` (config.py:5324), the same value
    the rest of the app treats as "the default provider", and read at call
    time so a config reload is picked up. No provider name is hardcoded
    here; config.py owns the fallback.

    Shared with `briefing_cast.generate_script` (spec #2 phase 2a): a cast's
    provider resolution falls back through the same chain -- explicit args,
    then the preset's own provider, then this app default -- so both
    generation paths agree on what "the default" means without duplicating
    the config read.
    """
    from .. import config as app_config

    return str(app_config.default_api_endpoint)


def _error_text(exc: BaseException) -> str:
    """The exception's message, capped -- never a traceback.

    The row is rendered in a status line the user reads, so it holds what
    went wrong, not where. The stack goes to the log.
    """
    message = str(exc).strip() or exc.__class__.__name__
    if len(message) > ERROR_CHAR_CAP:
        message = message[:ERROR_CHAR_CAP] + " [...]"
    return message


async def _invoke_chat(
    chat: Callable[..., Any],
    *,
    endpoint: str,
    model: str | None,
    system: str,
    user: str,
) -> Any:
    """Make the one chat call, accepting a sync or async seam.

    The real `chat_api_call` is synchronous and does blocking network I/O,
    so it is offloaded to a thread rather than run on the event loop. The
    system prompt travels in `system_message`, not as a message role: that
    is the app's own division of labour (`Chat_Functions` "PHILOSOPHY"
    comment) -- each provider handler decides whether its API wants a system
    turn prepended or a separate top-level field.
    """
    kwargs: dict[str, Any] = {
        "api_endpoint": endpoint,
        "messages_payload": [{"role": "user", "content": user}],
        "system_message": system,
        "model": model,
        "streaming": False,
        "max_tokens": BRIEFING_MAX_TOKENS,
        "temp": BRIEFING_TEMPERATURE,
    }
    if inspect.iscoroutinefunction(chat):
        return await chat(**kwargs)
    result = await asyncio.to_thread(chat, **kwargs)
    if inspect.isawaitable(result):  # a sync callable returning an awaitable
        return await result
    return result


def _write_junction(
    db: "SubscriptionsDB",
    briefing_id: int,
    selection: BriefingSelection,
) -> None:
    """Record which items this briefing covered, and which were featured.

    Written before the status flips to `complete`, so a crash between the
    two leaves a `generating` row whose junction rows the selection
    exclusion already ignores (its allowlist is `('complete', 'empty')`) --
    and which recovery then fails honestly. The reverse order would briefly
    publish a complete briefing that covered nothing.
    """
    with db.transaction() as conn:
        for item in selection.items:
            conn.execute(
                "INSERT OR REPLACE INTO briefing_items "
                "(briefing_id, item_id, featured) VALUES (?, ?, ?)",
                (
                    briefing_id,
                    item["item_id"],
                    1 if item["item_id"] in selection.featured_ids else 0,
                ),
            )


# --- Sync DB work, grouped for `asyncio.to_thread` (whole-branch review ----
# fix 1) -----------------------------------------------------------------
#
# `generate_briefing` is `async`, but every one of these calls is a plain
# synchronous SQLite call. Before this fix they ran directly on the caller's
# event loop -- the screen dispatches `generate_briefing` from a Textual
# worker (`watchlists_collections_screen.py`'s `_generate_briefing`), so a
# contended write blocked the whole UI. Grouped into one `to_thread` hop per
# stage rather than one hop per statement: Task 4's `_sweep_and_guard`
# already proved this `db` object is safe to drive from a worker thread.
# Each helper below is plain -- no `db.transaction()` spanning a hop, no
# `await` inside one -- so it is safe to run on whichever thread
# `asyncio.to_thread` picks.


def _start_generation(
    db: "SubscriptionsDB", watchlist_id: int, preset_id: int | None, now: datetime | None
) -> tuple[int, str, int | None, BriefingSelection, dict[str, Any] | None]:
    """Everything before the chat call: insert the row, resolve the mode,
    read the prior watermark, select, and resolve the preset (if any).
    Returns `(briefing_id, mode, prior_watermark, selection, preset)`.

    The preset lookup is grouped into this same `to_thread` hop (spec #2
    phase 2a) rather than given its own -- one more plain SQLite read costs
    nothing extra added to a hop that already exists, and a second hop would
    only be pure overhead. `preset` is `None` both when `preset_id` is
    `None` (no preset requested) and when it no longer resolves (a deleted
    preset) -- `generate_briefing` cannot tell those two apart from this
    return value alone, but it doesn't need to: both mean "proceed on
    defaults."
    """
    briefing_id = db.insert_briefing(watchlist_id, status=STATUS_GENERATING)
    mode = _selection_mode(db, watchlist_id)
    prior_watermark = db.latest_completed_watermark(watchlist_id)
    selection = select_briefing_items(db, watchlist_id, mode=mode, now=now)
    preset = db.get_briefing_preset(preset_id) if preset_id is not None else None
    return briefing_id, mode, prior_watermark, selection, preset


def _finish_empty(
    db: "SubscriptionsDB",
    briefing_id: int,
    mode: str,
    preset_id: int | None,
    covers_through: int | None,
    selection: BriefingSelection,
) -> dict[str, Any]:
    """Record the empty-window outcome and read the finished row back."""
    db.update_briefing(
        briefing_id,
        status=STATUS_EMPTY,
        item_count=0,
        featured_count=0,
        overflow_count=selection.overflow_count,
        covers_through_item_id=covers_through,
        covers_from_ts=selection.covers_from_ts,
        selection_mode=mode,
        preset_id=preset_id,
    )
    return db.get_briefing(briefing_id)


def _finish_success(
    db: "SubscriptionsDB",
    briefing_id: int,
    mode: str,
    preset_id: int | None,
    model_used: str,
    covers_through: int | None,
    selection: BriefingSelection,
    body: str,
) -> dict[str, Any]:
    """Write the junction rows, flip the row to `complete`, and read it back.

    Junction rows first, status flip second -- see `_write_junction`'s own
    docstring for why the order is load-bearing.
    """
    _write_junction(db, briefing_id, selection)
    db.update_briefing(
        briefing_id,
        status=STATUS_COMPLETE,
        body_markdown=_append_overflow(body, selection.overflow_count),
        item_count=len(selection.items),
        featured_count=len(selection.featured_ids),
        overflow_count=selection.overflow_count,
        covers_through_item_id=covers_through,
        covers_from_ts=selection.covers_from_ts,
        selection_mode=mode,
        preset_id=preset_id,
        model_used=model_used,
    )
    return db.get_briefing(briefing_id)


def _finish_failure(
    db: "SubscriptionsDB",
    briefing_id: int,
    mode: str,
    preset_id: int | None,
    model_used: str,
    message: str,
) -> dict[str, Any]:
    """Record the failure outcome and read the finished row back.

    No `covers_through_item_id` and no junction rows: a briefing that never
    reached the user covered nothing, so the next attempt re-selects the
    same items. The spec's named invariant.
    """
    db.update_briefing(
        briefing_id,
        status=STATUS_FAILED,
        error=message,
        selection_mode=mode,
        preset_id=preset_id,
        model_used=model_used,
    )
    return db.get_briefing(briefing_id)


async def generate_briefing(
    db: "SubscriptionsDB",
    watchlist_id: int,
    *,
    chat: Callable[..., Any] = chat_api_call,
    provider: str | None = None,
    model: str | None = None,
    preset_id: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Generate one briefing for a watchlist and return the stored row.

    Never raises for a provider failure: the failure becomes the row's
    status and error, because a briefing the user can see failed is worth
    more than an exception they cannot.

    The caller is expected to have run `fail_interrupted_briefings` and to
    hold the one-generation-per-watchlist guard; this function does not
    check either (see the module docstring).

    Args:
        db: An open `SubscriptionsDB`.
        watchlist_id: The watchlist to brief.
        chat: The chat seam. Defaults to `Chat_Functions.chat_api_call`;
            may be sync or async. The only seam faked in tests.
        provider: Chat endpoint to use. Wins over the preset's own provider
            when given. Defaults to the preset's provider, then the app's
            configured default endpoint.
        model: Model name to pass through. Wins over the preset's own model
            when given. Defaults to the preset's model, then `None` (letting
            the provider handler choose its own default).
        preset_id: A `briefing_presets.id` (spec #2 phase 2a) to resolve
            provider/model defaults and style guidance from. A preset that
            no longer resolves (deleted between being chosen and this call)
            is treated exactly like no preset at all -- generation proceeds
            on `provider`/`model`/the app default, and the row records
            `preset_id=None` rather than the stale, dangling id: a deleted
            preset must not brick generation. `None` (the default) skips
            preset resolution entirely.
        now: Injected clock, forwarded to selection (the first briefing's
            7-day floor). Defaults to the current UTC time.

    Returns:
        The finished `briefings` row as a dict, whatever its status.

    Whole-branch review fix 1: every DB call here is plain synchronous
    SQLite, and the only genuinely async step is `_invoke_chat` (which
    itself offloads the real network call). Each stage's DB work is grouped
    into one small plain function and run in a single `asyncio.to_thread`
    hop -- not one hop per statement -- so the caller's event loop (a
    Textual worker thread, in the shipping caller) is never blocked by
    sqlite contention. The error boundary is unchanged: the `try/except`
    below still wraps `_invoke_chat` ONLY, so a database error from any of
    the `to_thread` calls still propagates to the caller uncaught, exactly
    as it did when these were plain statements.
    """
    briefing_id, mode, prior_watermark, selection, preset = await asyncio.to_thread(
        _start_generation, db, watchlist_id, preset_id, now
    )
    # The id actually recorded on the row: `None` for both "no preset was
    # requested" and "the requested preset no longer resolves" -- a stale
    # back-reference to a deleted preset is worse than no reference at all.
    recorded_preset_id = preset_id if preset is not None else None
    preset_provider = preset.get("provider") if preset else None
    preset_model = preset.get("model") if preset else None
    style_notes = preset.get("style_notes") if preset else None

    # `None` means "selection found no line to record" -- curated mode with
    # no prior briefing, or a genuinely empty window. Writing the prior
    # watermark back keeps the row self-describing (it states the line it
    # covers through) without moving the line: `latest_completed_watermark`
    # takes a MAX, so an echo is a no-op to it either way.
    covers_through = selection.covers_through_item_id
    if covers_through is None:
        covers_through = prior_watermark

    if not selection.items:
        row = await asyncio.to_thread(
            _finish_empty, db, briefing_id, mode, recorded_preset_id, covers_through, selection
        )
        logger.info(f"briefing {briefing_id}: empty window for watchlist {watchlist_id}")
        return row

    system, user = build_briefing_prompt(
        selection.items, selection.featured_ids, selection.overflow_count
    )
    if style_notes:
        # Appended rather than folded into `build_briefing_prompt` (whose
        # contract phase 1 owns and phase 2a does not touch): the preset's
        # guidance is a property of THIS call's cast, not of prompt assembly
        # itself.
        system = f"{system}\n\n## Style notes\n\n{style_notes}"
    endpoint = provider or preset_provider or _default_provider()
    resolved_model = model or preset_model
    model_used = f"{endpoint}/{resolved_model}" if resolved_model else endpoint

    try:
        raw = await _invoke_chat(
            chat, endpoint=endpoint, model=resolved_model, system=system, user=user
        )
    except Exception as exc:  # noqa: BLE001 - every provider failure is a row
        # No traceback: the log file sink runs with diagnose=True, which would
        # dump frame locals into the log file -- and the frame here is
        # `_invoke_chat`, whose locals are the prompt. That would put item
        # titles and excerpts in a file the user never chose to send anywhere,
        # falsifying this module's egress claim. The provider's own message
        # still reaches the user, on the row, where they are already looking.
        logger.warning(
            f"briefing {briefing_id}: generation failed against {endpoint}: "
            f"{type(exc).__name__}"
        )
        return await asyncio.to_thread(
            _finish_failure, db, briefing_id, mode, recorded_preset_id, model_used, _error_text(exc)
        )

    body = extract_response_content(raw).strip()
    if not body:
        # Recording this `complete` would show an empty artifact with no
        # error to explain it -- and would advance the window past items
        # nothing ever reported.
        logger.warning(f"briefing {briefing_id}: {endpoint} returned an empty response")
        return await asyncio.to_thread(
            _finish_failure,
            db,
            briefing_id,
            mode,
            recorded_preset_id,
            model_used,
            f"{endpoint} returned an empty response",
        )

    row = await asyncio.to_thread(
        _finish_success,
        db,
        briefing_id,
        mode,
        recorded_preset_id,
        model_used,
        covers_through,
        selection,
        body,
    )
    logger.info(
        f"briefing {briefing_id}: complete -- {len(selection.items)} items, "
        f"{selection.overflow_count} overflow, watermark {covers_through}"
    )
    return row


def fail_interrupted_briefings(
    db: "SubscriptionsDB", watchlist_id: int | None = None
) -> int:
    """Fail every `generating` briefing as `interrupted`; return the count.

    Zombie recovery, TASK-1090's shape: a worker that crashed mid-generation
    leaves a `generating` row that would wedge the one-generation-at-a-time
    guard shut forever. Only `generating` rows are touched -- finished
    history keeps its status, its body, its watermark and its own error text.

    Args:
        db: An open `SubscriptionsDB`.
        watchlist_id: Scope the sweep to one watchlist. `None` sweeps all,
            which is what a startup pass wants.

    Returns:
        How many rows were failed.
    """
    sql = (
        "UPDATE briefings SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE status = ?"
    )
    params: list[Any] = [STATUS_FAILED, INTERRUPTED_ERROR, STATUS_GENERATING]
    if watchlist_id is not None:
        sql += " AND watchlist_id = ?"
        params.append(watchlist_id)

    with db.transaction() as conn:
        count = conn.execute(sql, params).rowcount
    if count:
        logger.info(f"failed {count} interrupted briefing(s)")
    return count
