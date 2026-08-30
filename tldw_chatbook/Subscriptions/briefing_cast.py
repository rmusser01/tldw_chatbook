"""Cast a finished briefing into an N-speaker script (spec #2 phase 2a).

A **script** is a second, independent LLM pass over a briefing that has
already reached `complete`: given a roster (one or more named speakers, each
optionally bound to a character card and a TTS voice profile), it produces a
strict JSON array of `{"speaker", "text"}` turns -- the raw material a later
phase (2b) synthesizes into audio.

Two rules carry this module, both from the design spec's "Casting and audio"
and "Error handling ethos" sections:

**1. The briefing is never touched by a script's outcome.** Success or
failure, this module writes only to `briefing_scripts`. A script is a
*derivative* artifact; failing to cast one must not retroactively call the
briefing itself into question, and a briefing may be cast many times (a
roster edit, a retry) without any of those attempts leaving a mark on it.

**2. Validation is strict and failure is honest, by name.** An unknown
speaker in the model's reply fails naming the speaker; a malformed payload
fails naming the parse defect; a roster that references a character card
which no longer exists fails naming the card. None of these degrade
silently into a script that is technically `complete` but wrong.

A roster of one speaker produces narration through the identical path as a
roster of many -- there is no special "narration mode": the array contract,
the strict parser, and the failure rules are exactly the same either way.

Testing (spec #Testing): the only faked seam is `chat` (mirroring
`briefing_service`'s own rule) -- exactly like that module, everything else
here is exercised against a real `SubscriptionsDB`.

Nothing here logs prompt, roster, or turn content -- only exception types
-- for the same reason `briefing_service` avoids it: this app's log sink
runs with `diagnose=True`, which dumps a failing frame's locals, and the
frame at a cast failure holds the prompt.

task-1780 (spec: "Re-casting without the watchlist") adds a second entry
point, `generate_script_from_text`, casting directly from a ChaChaNotes
`kept_briefings` row (see `Subscriptions/briefing_keep.py`) instead of a
`Subscriptions_DB` `briefings` row, and writing into `kept_scripts` instead
of `briefing_scripts`. It reuses this module's pure pieces verbatim
(`build_cast_prompt`, `parse_script_turns`, `validate_roster`,
`_resolve_character_texts`, `_snapshot_roster`) but keeps its own claim set
and its own error-vs-row contract -- see that function's own section
comment, below `generate_script`, for the full reasoning.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from contextlib import contextmanager
from typing import Any, Callable, Collection, Iterator, Mapping, Optional

from loguru import logger

from ..Chat.Chat_Functions import chat_api_call, extract_response_content
from ..model_capabilities import deepseek_model_thinks_by_default
from .briefing_service import STATUS_COMPLETE as _BRIEFING_STATUS_COMPLETE
from .briefing_service import GenerationInFlightError, default_briefing_provider

#: Statuses a `briefing_scripts` row can hold. Unlike `briefings`, a script
#: has no `empty` status -- there is no "empty roster" outcome, since
#: `validate_roster` refuses an empty roster before any row is written.
VALID_STATUSES = ("generating", "complete", "failed")
STATUS_GENERATING = "generating"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"

#: The error text a zombie row is failed with (mirrors
#: `briefing_service.INTERRUPTED_ERROR` exactly).
INTERRUPTED_ERROR = "interrupted"

#: Provider errors and `ScriptCastError` messages alike are capped before
#: being written to the row: the row is rendered in a status line, not a
#: log file, and a provider can return a whole HTML error page as its
#: message.
ERROR_CHAR_CAP = 1000

#: Per-field cap on a bound character card's contribution (spec: "the
#: card's `personality` + `description` fields, truncated to 1000 chars
#: each") -- keeps one verbose card from dominating the cast prompt the
#: same way `briefing_service.EXCERPT_CHAR_CAP` bounds one item's excerpt.
CHARACTER_TEXT_CHAR_CAP = 1000

#: Completion budget for the cast call. Higher than a briefing's own
#: `BRIEFING_MAX_TOKENS` (2000): a script is dialogue turns covering the
#: same material the briefing already condensed once, and multi-speaker
#: back-and-forth reads longer than the equivalent prose.
CAST_MAX_TOKENS = 3000

#: Reasoning-typed models burn completion budget on thinking before any
#: visible text (TASK-21515 -- same defect the briefing call hit): give
#: them headroom so the same output length still fits.
CAST_REASONING_MAX_TOKENS = 12000

#: Dialogue benefits from a little more variation than a summarization
#: pass, but this is still a *scripted* adaptation of given material, not
#: free invention -- kept below `BRIEFING_TEMPERATURE`'s neighbourhood-plus
#: rather than a "creative writing" setting.
CAST_TEMPERATURE = 0.5

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

_OUTPUT_CONTRACT_TEMPLATE = (
    'Respond with ONLY a JSON array of {{"speaker", "text"}} turns; '
    "speaker must be one of: {names}"
)


class ScriptCastError(RuntimeError):
    """Raised when a roster, a cast prompt, or a model's reply is unusable.

    Every raise site names the specific defect (a duplicate speaker, an
    unknown speaker, a missing character card, a malformed payload) rather
    than a generic "cast failed" -- the message is what ends up on the
    `briefing_scripts` row a user reads.
    """


def validate_roster(roster: object) -> list[dict]:
    """Normalize and validate a preset's (or a script's) speaker roster.

    Args:
        roster: Untrusted roster data -- typically `load_roster`'s output,
            but accepted as `object` since a caller may hand this a value
            that was never valid JSON in the first place (e.g. a Python
            literal built directly in a test or a future UI form).

    Returns:
        A new list of normalized speaker dicts, each exactly
        `{"name": str, "role_prompt": str, "character_card_id": int | None,
        "voice_profile_id": str | None}`, in the input's original order.

    Raises:
        ScriptCastError: Naming the specific defect -- not a list, no
            speakers, a non-object entry, a blank or duplicate name, or a
            `character_card_id` that isn't an integer.
    """
    if not isinstance(roster, list):
        raise ScriptCastError("a roster must be a list of speakers")
    if not roster:
        # Spec: "a roster of one produces narration through the identical
        # path -- no special mode" -- but a roster of ZERO has no speaker
        # to narrate through, so this is the one shape that is never valid.
        raise ScriptCastError("a roster must have at least one speaker")

    seen_names: set[str] = set()
    normalized: list[dict] = []
    for index, entry in enumerate(roster):
        if not isinstance(entry, Mapping):
            raise ScriptCastError(f"roster entry {index} is not a speaker object")

        name = str(entry.get("name") or "").strip()
        if not name:
            raise ScriptCastError(f"roster entry {index} has no speaker name")
        if name in seen_names:
            raise ScriptCastError(f"duplicate speaker name {name!r} in roster")
        seen_names.add(name)

        role_prompt = str(entry.get("role_prompt") or "").strip()

        character_card_id = entry.get("character_card_id")
        if character_card_id is not None:
            try:
                character_card_id = int(character_card_id)
            except (TypeError, ValueError):
                raise ScriptCastError(
                    f"speaker {name!r} has a non-numeric character_card_id"
                ) from None

        voice_profile_id = entry.get("voice_profile_id")
        if voice_profile_id is not None:
            voice_profile_id = str(voice_profile_id)

        normalized.append(
            {
                "name": name,
                "role_prompt": role_prompt,
                "character_card_id": character_card_id,
                "voice_profile_id": voice_profile_id,
            }
        )
    return normalized


def dump_roster(roster: list[dict]) -> str:
    """Canonical JSON encoding of a roster, for storage.

    Used for both a preset's `roster_json` and a script's
    `roster_snapshot_json` -- the latter typically a roster enriched with a
    resolved `character_name` per speaker (see `_snapshot_roster`), which is
    why this function does not assume the fixed four-key shape
    `validate_roster` produces; it serializes whatever list of speaker dicts
    it is given.

    Args:
        roster: A list of speaker dicts.

    Returns:
        Deterministic JSON text (`sort_keys=True`), so two calls over
        equal data produce byte-identical text.
    """
    return json.dumps(roster, sort_keys=True)


def load_roster(text: str) -> list[dict]:
    """Inverse of `dump_roster`.

    Args:
        text: Stored roster JSON (a preset's `roster_json` or a script's
            `roster_snapshot_json`).

    Returns:
        The decoded list of speaker dicts, in stored order.

    Raises:
        ScriptCastError: If `text` is not valid JSON, or is not a JSON
            array of objects.
    """
    try:
        payload = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ScriptCastError("roster is not valid JSON") from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, Mapping) for item in payload
    ):
        raise ScriptCastError("roster JSON must be an array of speaker objects")
    return [dict(item) for item in payload]


def _snapshot_roster(
    roster: list[dict], load_character: Optional[Callable[[int], Optional[dict]]]
) -> list[dict]:
    """Enrich a validated roster with each bound speaker's card name.

    Deliberately tolerant: this runs before the `briefing_scripts` row is
    inserted, and `roster_snapshot_json` cannot be revised after insert (it
    is not in `update_briefing_script`'s allowlist), so a card that fails to
    resolve here must not block the row from ever being written -- it
    degrades to `character_name: None` instead. The STRICT check that
    actually fails a cast over a missing card is `_resolve_character_texts`,
    which runs later, inside `generate_script`'s try/except, because that
    is the failure the user must see as a `failed` script rather than a
    silently unwritten one.

    Args:
        roster: `validate_roster`'s output.
        load_character: Character card lookup, or `None` if unavailable.

    Returns:
        A new list of speaker dicts, each the input entry plus
        `character_name` (`str | None`).
    """
    enriched: list[dict] = []
    for speaker in roster:
        entry = dict(speaker)
        character_name = None
        card_id = speaker.get("character_card_id")
        if card_id is not None and load_character is not None:
            try:
                card = load_character(card_id)
            except Exception:  # noqa: BLE001 - best-effort only, never blocks the cast
                card = None
            if isinstance(card, Mapping):
                name = str(card.get("name") or "").strip()
                character_name = name or None
        entry["character_name"] = character_name
        enriched.append(entry)
    return enriched


async def _resolve_character_texts(
    roster: list[dict], load_character: Optional[Callable[[int], Optional[dict]]]
) -> dict[str, str]:
    """Resolve each bound speaker's character-card contribution, strictly.

    Spec: "A preset whose roster names a deleted character card fails the
    cast at that point, naming the card." Runs inside `generate_script`'s
    try/except, so a failure here becomes a `failed` script row rather than
    an unwritten one -- `_snapshot_roster`, above, already wrote the row
    with whatever it could resolve leniently; this is the strict half.

    `async`, and each `load_character` call is offloaded via `asyncio.
    to_thread` (Task 5 review, Important): `load_character` is a plain sync
    callable whose real implementation is a blocking DB read
    (`ChaChaNotesDB.get_character_card_by_id`), and unlike `_snapshot_
    roster`'s own call to it (already off the event loop, inside `_start_
    script`'s `asyncio.to_thread` wrapper), THIS function used to be called
    directly from `generate_script`'s own coroutine body -- i.e. on the
    event loop thread. The callback's own contract is unchanged (still a
    plain synchronous `Callable[[int], Optional[dict]]`); this function
    owns the threading on the caller's behalf, exactly like `_start_script`
    already does for its own `load_character` call.

    Args:
        roster: `validate_roster`'s output.
        load_character: Character card lookup by id. `None` if the caller
            has no lookup available at all (still an error for any speaker
            that names a card -- there is nothing to resolve it with).

    Returns:
        Speaker name -> character contribution (the card's `personality`
        and `description` fields, each capped at `CHARACTER_TEXT_CHAR_CAP`
        characters, joined). Speakers with no bound card are absent.

    Raises:
        ScriptCastError: Naming the character card id, if a speaker
            references one that does not resolve to a card.
    """
    texts: dict[str, str] = {}
    for speaker in roster:
        card_id = speaker.get("character_card_id")
        if card_id is None:
            continue
        card = (
            await asyncio.to_thread(load_character, card_id)
            if load_character is not None
            else None
        )
        if card is None:
            raise ScriptCastError(
                f"speaker {speaker['name']!r} references character card {card_id}, "
                "which could not be found"
            )
        personality = str(card.get("personality") or "").strip()[
            :CHARACTER_TEXT_CHAR_CAP
        ]
        description = str(card.get("description") or "").strip()[
            :CHARACTER_TEXT_CHAR_CAP
        ]
        contribution = "\n".join(part for part in (description, personality) if part)
        if contribution:
            texts[speaker["name"]] = contribution
    return texts


def build_cast_prompt(
    body_markdown: str,
    roster: list[dict],
    style_notes: str | None,
    character_texts: dict[str, str],
) -> tuple[str, str]:
    """Build the (system, user) prompt for one cast pass. Pure.

    Args:
        body_markdown: The finished briefing's body, sent verbatim as the
            user prompt -- the model is adapting existing material, not
            asked to invent one.
        roster: `validate_roster`'s output.
        style_notes: The preset's free-text style guidance, or `None`.
        character_texts: `_resolve_character_texts`'s output.

    Returns:
        `(system_prompt, user_prompt)`. `user_prompt` is `body_markdown`
        unchanged.
    """
    names = [speaker["name"] for speaker in roster]
    sections = [
        "You are adapting a written briefing into a spoken-style script for "
        "the speakers below. Cover the same material as the briefing -- do "
        "not invent claims it does not make -- and write it as natural "
        "back-and-forth dialogue appropriate to each speaker's role."
    ]

    speaker_lines = ["## Speakers"]
    for speaker in roster:
        speaker_lines.append(f"### {speaker['name']}")
        if speaker.get("role_prompt"):
            speaker_lines.append(speaker["role_prompt"])
        contribution = character_texts.get(speaker["name"])
        if contribution:
            speaker_lines.append(f"Character background: {contribution}")
    sections.append("\n\n".join(speaker_lines))

    if style_notes:
        sections.append("## Style notes\n\n" + style_notes)

    sections.append(_OUTPUT_CONTRACT_TEMPLATE.format(names=", ".join(names)))

    system = "\n\n".join(sections)
    return system, body_markdown


def parse_script_turns(text: str, roster_names: set[str]) -> list[dict]:
    """Parse a cast reply into strict `{"speaker", "text"}` turns.

    Tolerates the markdown fence local models routinely add and a
    prose-wrapped array (recovers the first `[`...`]` slice), then applies
    the strict output contract: anything short of a clean array of
    `{"speaker": str, "text": str}` objects, each speaker (after stripping
    incidental whitespace) a name in `roster_names`, fails naming the
    specific defect.

    Args:
        text: Raw provider reply.
        roster_names: Valid speaker names -- `validate_roster`'s `name`
            fields for this cast's roster (already stripped/canonical).

    Returns:
        `[{"speaker": str, "text": str}, ...]`, in reply order. Each
        `speaker` is the CANONICAL (stripped) name, not necessarily the raw
        string the model wrote -- matching `roster_names`, which is also
        canonical, so downstream rendering never sees a mismatch.

    Raises:
        ScriptCastError: Naming the defect -- unparsable JSON, JSON that is
            not an array, a non-object turn, a turn missing `speaker` or
            `text`, non-string `text` (naming the turn index), or a speaker
            not in `roster_names` (naming the speaker -- the named
            invariant: `test_an_unknown_speaker_fails_the_script_by_name`).
    """
    raw = (text or "").strip()
    fenced = _JSON_FENCE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    if not raw.startswith("["):
        # Some models prepend a sentence; recover the first array if present.
        start = raw.find("[")
        end = raw.rfind("]")
        raw = raw[start : end + 1] if start != -1 and end > start else raw

    try:
        payload = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ScriptCastError("the model did not return a JSON turn array") from exc
    if not isinstance(payload, list):
        raise ScriptCastError("the model returned JSON that is not a turn array")

    turns: list[dict] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ScriptCastError(f"turn {index} is not a turn object")
        if "speaker" not in item or "text" not in item:
            raise ScriptCastError(f"turn {index} is missing 'speaker' or 'text'")

        speaker = item["speaker"]
        turn_text = item["text"]
        if not isinstance(speaker, str) or not speaker.strip():
            raise ScriptCastError(f"turn {index} has a missing or non-string speaker")
        if not isinstance(turn_text, str):
            raise ScriptCastError(f"turn {index} has non-string text")
        # `roster_names` is `validate_roster`'s output -- already stripped --
        # but the model's raw reply carries no such guarantee. Canonicalize
        # the same way before the membership check, or a merely-padded name
        # (e.g. "Alice ") would fail the WHOLE cast as an unknown speaker.
        # The stored turn keeps the canonical name too, so downstream
        # rendering matches the roster exactly.
        canonical_speaker = speaker.strip()
        if canonical_speaker not in roster_names:
            raise ScriptCastError(f"unknown speaker {speaker!r} in turn {index}")

        turns.append({"speaker": canonical_speaker, "text": turn_text})

    if not turns:
        raise ScriptCastError("the model returned no turns")
    return turns


def _error_text(exc: BaseException) -> str:
    """The exception's message, capped -- never a traceback.

    Copies `briefing_service._error_text`'s exact shape (not imported: that
    function is private to its own module).
    """
    message = str(exc).strip() or exc.__class__.__name__
    if len(message) > ERROR_CHAR_CAP:
        message = message[:ERROR_CHAR_CAP] + " [...]"
    return message


def _effective_max_tokens(endpoint: str, model: str | None) -> int:
    """The completion budget for one cast call, reasoning-aware (TASK-21515).

    Copies `briefing_service._effective_max_tokens`'s exact shape (not
    imported: that function is private to its own module) against this
    module's own constants: the DeepSeek handler's reasoning-inclusive
    ``max_tokens`` means a reasoning-typed default model needs headroom, and
    only the native ``deepseek`` endpoint is widened.
    """
    endpoint_normalized = str(endpoint or "").strip().lower()
    if endpoint_normalized == "deepseek" and deepseek_model_thinks_by_default(model):
        return CAST_REASONING_MAX_TOKENS
    return CAST_MAX_TOKENS


async def _invoke_chat(
    chat: Callable[..., Any],
    *,
    endpoint: str,
    model: str | None,
    system: str,
    user: str,
) -> Any:
    """Make the one cast chat call, accepting a sync or async seam.

    Copies `briefing_service._invoke_chat`'s exact sync/async-seam shape
    (not imported: that function is private to its own module) -- the real
    `chat_api_call` is synchronous and does blocking network I/O, so it is
    offloaded to a thread rather than run on the event loop.
    """
    kwargs: dict[str, Any] = {
        "api_endpoint": endpoint,
        "messages_payload": [{"role": "user", "content": user}],
        "system_message": system,
        "model": model,
        "streaming": False,
        "max_tokens": _effective_max_tokens(endpoint, model),
        "temp": CAST_TEMPERATURE,
    }
    if inspect.iscoroutinefunction(chat):
        return await chat(**kwargs)
    result = await asyncio.to_thread(chat, **kwargs)
    if inspect.isawaitable(result):  # a sync callable returning an awaitable
        return await result
    return result


# --- Sync DB work, grouped for `asyncio.to_thread` (mirrors
# `briefing_service`'s "whole-branch review fix 1" shape exactly) ----------
#
# `generate_script` is `async`, dispatched from a Textual worker in the real
# caller (Task 5), so every plain synchronous SQLite call here is grouped
# into one `to_thread` hop per stage rather than run directly on the
# caller's event loop.


def _start_script(
    db: Any,
    briefing_id: int,
    preset_id: int,
    load_character: Optional[Callable[[int], Optional[dict]]],
) -> tuple[int, dict[str, Any], dict[str, Any], list[dict], set[str]]:
    """Everything before the chat call: validate, resolve, insert.

    Every check that must refuse WITHOUT ever creating a row runs here,
    before the terminal `insert_briefing_script` call -- a briefing that
    isn't `complete`, a preset that no longer exists, or a malformed roster
    all raise `ScriptCastError` with no row ever written.

    Returns:
        `(script_id, briefing, preset, roster, roster_names)`.

    Raises:
        ScriptCastError: If the briefing does not exist or is not
            `complete`, the preset does not exist, or the preset's roster
            fails `validate_roster`.
    """
    briefing = db.get_briefing(briefing_id)
    if briefing is None:
        raise ScriptCastError(f"briefing {briefing_id} does not exist")
    if briefing["status"] != _BRIEFING_STATUS_COMPLETE:
        raise ScriptCastError(
            f"briefing {briefing_id} is {briefing['status']!r}, not complete; "
            "a script can only be cast from a complete briefing"
        )

    preset = db.get_briefing_preset(preset_id)
    if preset is None:
        raise ScriptCastError(f"briefing preset {preset_id} does not exist")

    roster = validate_roster(load_roster(preset["roster_json"]))
    roster_names = {speaker["name"] for speaker in roster}
    snapshot_roster = _snapshot_roster(roster, load_character)

    script_id = db.insert_briefing_script(
        briefing_id,
        preset_id=preset_id,
        preset_name=preset["name"],
        roster_snapshot_json=dump_roster(snapshot_roster),
    )
    return script_id, briefing, preset, roster, roster_names


def _finish_script_success(
    db: Any, script_id: int, turns_json: str, model_used: str
) -> dict[str, Any]:
    """Record the completed cast and read the finished row back."""
    db.update_briefing_script(
        script_id, status=STATUS_COMPLETE, turns_json=turns_json, model_used=model_used
    )
    return db.get_briefing_script(script_id)


def _finish_script_failure(db: Any, script_id: int, message: str) -> dict[str, Any]:
    """Record the failed cast and read the finished row back.

    Touches only `briefing_scripts` -- the briefing this script was cast
    from is never written by this module, on any outcome (spec §Error
    handling ethos).
    """
    db.update_briefing_script(script_id, status=STATUS_FAILED, error=message)
    return db.get_briefing_script(script_id)


# --- In-process cast claims (spec #2 phase 4) ------------------------------
#
# Mirrors `briefing_service`'s own claim set exactly -- see that module's
# "In-process generation claims" section comment for the full reasoning (no
# lock needed because the check-then-add pair never awaits). Scoped to
# `briefing_id` here rather than a watchlist: a cast's collision unit is
# "this briefing is already being cast", not "this watchlist is already
# being briefed" -- `briefing_scripts` has no one-row-per-briefing
# invariant the way `briefings` has one per watchlist, but two casts of the
# SAME briefing running at once would still make two concurrent chat calls
# and write two concurrent `generating` rows for no reason.
#
# `GenerationInFlightError` is REUSED from `briefing_service`, not mirrored:
# unlike `_error_text`/`_invoke_chat` above (private helpers this module
# deliberately copies rather than imports, so each module's own contract
# reads independently), a caller that wants to catch "something is already
# generating" uniformly across briefings/scripts/audio needs ONE type to
# catch, not three near-identical ones with no shared base beyond
# `RuntimeError`.

_ACTIVE_CAST_CLAIMS: set[int] = set()

# Task-1890 (generalizing task-1812's briefings-side fix): `briefing_id ->
# briefing_scripts.id` for the SAME live claim above, but scoped to the
# actual ROW a live cast is writing rather than merely the briefing it
# belongs to. `_ACTIVE_CAST_CLAIMS` alone cannot tell a fresh live script row
# apart from a crash-zombie row left by a PRIOR process for the SAME
# briefing (the crash predates the claim) -- both read `generating` and
# share one `briefing_id`, so a briefing-scoped `exclude` incidentally
# shields the zombie too. An entry here is added only once the row actually
# exists (`generate_script` records it right after `_start_script`'s
# `INSERT` returns), and popped in `_claim_cast`'s `finally`, alongside the
# briefing-level claim itself, so it never outlives the claim it belongs to.
# `fail_interrupted_scripts`'s `exclude` (via `active_cast_claim_row_ids`
# below) reads from THIS registry, not `_ACTIVE_CAST_CLAIMS`.
_ACTIVE_CAST_CLAIM_ROW_IDS: dict[int, int] = {}


def active_cast_claims() -> frozenset[int]:
    """Snapshot of briefing ids a live `generate_script` call currently holds.

    See `briefing_service.active_briefing_claims` for the snapshot
    reasoning; this is its cast-scoped sibling.

    NOT what `fail_interrupted_scripts`'s `exclude` wants any more
    (task-1890, generalizing task-1812): a briefing-scoped snapshot cannot
    tell a fresh live script row apart from a crash-zombie row a PRIOR
    process left behind for the SAME briefing, so passing this here would
    incidentally shield the zombie too. Callers of the sweep want
    `active_cast_claim_row_ids()` instead.
    """
    return frozenset(_ACTIVE_CAST_CLAIMS)


def active_cast_claim_row_ids() -> frozenset[int]:
    """Snapshot of `briefing_scripts.id`s a live `generate_script` call has
    actually inserted (task-1890, mirroring `briefing_service.active_
    briefing_claim_row_ids` exactly, scoped to a briefing id instead of a
    watchlist id).

    A plain, already-copied `frozenset` -- safe to read from any thread,
    unlike `_ACTIVE_CAST_CLAIM_ROW_IDS` itself. Callers take this snapshot
    on the event loop, before handing control to a thread, and pass it as
    `fail_interrupted_scripts`'s `exclude`: that function is sync and runs
    under `asyncio.to_thread`, while the registry is mutated only on the
    event loop, so a cross-thread read of the live registry would be racy
    in a way a snapshot taken beforehand never is.

    Row-scoped, not briefing-scoped -- see `active_cast_claims`'s own
    docstring for why that distinction is the whole point: a genuine crash
    zombie and a fresh live claim can coexist for one briefing, and only
    naming the live row itself, rather than its whole briefing, lets a
    sweep tell them apart.

    A briefing whose claim has been taken but whose row has not been
    recorded yet -- the window inside `_start_script`'s own `asyncio.
    to_thread` hop, from the moment its `INSERT` runs until `generate_
    script`'s coroutine resumes on the event loop to record the id -- is
    not represented here: there is no id for it to report until `generate_
    script` records one. `pending_cast_claim_briefing_ids()` below names
    exactly those briefings, and `fail_interrupted_scripts`'s `exclude_
    briefings` closes the gap this function's row-scoping alone cannot.
    """
    return frozenset(_ACTIVE_CAST_CLAIM_ROW_IDS.values())


def pending_cast_claim_briefing_ids() -> frozenset[int]:
    """Snapshot of briefing ids with a live claim whose row id is not yet
    recorded (task-1890, mirroring `briefing_service.pending_briefing_
    claim_watchlist_ids` exactly, scoped to a briefing id).

    `_claim_cast` adds `briefing_id` to `_ACTIVE_CAST_CLAIMS` before
    `generate_script` ever calls `_start_script`, but `generate_script`
    only records the row id (`_ACTIVE_CAST_CLAIM_ROW_IDS[briefing_id] =
    script_id`) once that call's WHOLE `asyncio.to_thread` hop returns --
    and `_start_script`'s `INSERT` is its LAST statement, after the
    briefing/preset/roster validation and the roster snapshot. For that
    whole span the briefing is claimed and its row exists (once the
    `INSERT` itself has run) and reads `generating`, but is named by
    nothing: `active_cast_claim_row_ids()` is empty (no id recorded yet),
    and `active_cast_claims()` is briefing-scoped -- passing it as `fail_
    interrupted_scripts`'s row-scoped `exclude` does not even type-check,
    and passing it as `exclude_briefings` unconditionally would resurrect
    the exact over-protection this task removes (shielding a genuine
    same-briefing crash zombie alongside the live row).

    This is the set difference: claimed, but not yet recorded. Callers pass
    it as `fail_interrupted_scripts`'s `exclude_briefings`, ALONGSIDE
    `exclude=active_cast_claim_row_ids()`, never instead of it -- the
    moment a briefing's row id lands, it drops out of this set and the
    row-scoped `exclude` alone protects it, which is exactly what keeps the
    coexistence fix (a zombie and a live claim on the SAME briefing, swept
    and spared respectively) intact.

    A plain, already-copied `frozenset`, computed with no `await` between
    reading the two registries -- safe only when called from the event
    loop, same as the other two accessors in this section.
    """
    return frozenset(_ACTIVE_CAST_CLAIMS) - frozenset(_ACTIVE_CAST_CLAIM_ROW_IDS)


@contextmanager
def _claim_cast(briefing_id: int, *, script_id: int | None = None) -> Iterator[None]:
    """Claim `briefing_id` for the duration of one cast attempt.

    See `briefing_service._claim_briefing` for the full reasoning (no
    `await` between the membership check and the `.add()`, so no lock is
    needed); this is the identical shape, scoped to a briefing id instead
    of a watchlist id. Also usable directly by tests that need to simulate
    another in-process caller already holding a briefing.

    Task-1890: this context manager's `finally` is also where `_ACTIVE_
    CAST_CLAIM_ROW_IDS`' entry for `briefing_id` is cleared, on every exit
    path, alongside the briefing-level claim itself -- so the row-id
    registry can never outlive the claim it belongs to. `.pop(..., None)`
    is a safe no-op whenever no id was ever recorded.

    Args:
        briefing_id: The briefing about to be cast.
        script_id: The `briefing_scripts.id` this claim protects, if
            already known at entry. `generate_script` itself never passes
            this -- its own row does not exist until AFTER this context
            manager is entered, so it records the id into `_ACTIVE_CAST_
            CLAIM_ROW_IDS` itself, mid-block, once `_start_script` returns.
            This parameter exists for tests simulating another in-process
            claim against a row that already exists.

    Raises:
        GenerationInFlightError: If `briefing_id` is already claimed.
    """
    if briefing_id in _ACTIVE_CAST_CLAIMS:
        raise GenerationInFlightError(
            f"a script is already being cast for briefing {briefing_id}"
        )
    _ACTIVE_CAST_CLAIMS.add(briefing_id)
    if script_id is not None:
        _ACTIVE_CAST_CLAIM_ROW_IDS[briefing_id] = script_id
    try:
        yield
    finally:
        _ACTIVE_CAST_CLAIMS.discard(briefing_id)
        _ACTIVE_CAST_CLAIM_ROW_IDS.pop(briefing_id, None)


async def generate_script(
    db: Any,
    briefing_id: int,
    *,
    preset_id: int,
    chat: Callable[..., Any] = chat_api_call,
    load_character: Optional[Callable[[int], Optional[dict]]] = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Cast a complete briefing into a script and return the stored row.

    Never raises for a cast failure (a missing character card, a provider
    error, a malformed reply): the failure becomes the row's status and
    error, mirroring `generate_briefing`'s own contract. It DOES raise
    `ScriptCastError` for a request that never should have started a cast
    at all -- the briefing isn't `complete`, or the preset does not exist
    -- and in both of those cases no `briefing_scripts` row is ever
    written.

    Args:
        db: An open `SubscriptionsDB`.
        briefing_id: The `briefings.id` to cast. Must be `complete`.
        preset_id: The `briefing_presets.id` supplying the roster (and,
            absent explicit `provider`/`model`, the cast's own provider,
            model, and style notes).
        chat: The chat seam. Defaults to `Chat_Functions.chat_api_call`;
            may be sync or async. The only seam faked in tests.
        load_character: Character card lookup by id, or `None` if
            unavailable. A roster speaker bound to a `character_card_id`
            that this cannot resolve fails the cast, naming the card.
        provider: Chat endpoint to use. Wins over the preset's own
            provider when given.
        model: Model name to pass through. Wins over the preset's own
            model when given.

    Returns:
        The finished `briefing_scripts` row as a dict, whatever its
        status.

    Raises:
        ScriptCastError: If the briefing does not exist or is not
            `complete`, or the preset does not exist. No row is written in
            either case.
        GenerationInFlightError: If another in-process caller already
            holds `briefing_id`'s cast claim (phase 4's `_claim_cast`).
            Raised before `_start_script` ever runs, so no
            `briefing_scripts` row is written for the refused attempt
            either -- the identical no-orphan-row shape `ScriptCastError`
            already has.
    """
    with _claim_cast(briefing_id):
        script_id, briefing, preset, roster, roster_names = await asyncio.to_thread(
            _start_script, db, briefing_id, preset_id, load_character
        )
        # Task-1890: record the row THIS claim is now writing, as the very
        # next statement after the `to_thread` hop above returns (no
        # `await` in between) -- so nothing else on this event loop can
        # observe the claim without the row it now protects. From here
        # until `_claim_cast`'s `finally` pops it back out, `fail_
        # interrupted_scripts`'s row-scoped `exclude` (`active_cast_claim_
        # row_ids`) can tell this row apart from any OTHER `generating` row
        # on the same briefing -- in particular a crash-zombie left by a
        # prior process, which does not belong here and must still be
        # swept.
        #
        # For the ENTIRE `to_thread` hop above, this briefing's id sits in
        # `_ACTIVE_CAST_CLAIMS` with no corresponding entry here yet, so
        # `active_cast_claim_row_ids()` alone cannot protect the row this
        # exact call is about to insert (`_start_script`'s `INSERT` is its
        # LAST statement). That window is what `pending_cast_claim_
        # briefing_ids()` names and `fail_interrupted_scripts`'s `exclude_
        # briefings` closes -- every screen call site passes it alongside
        # `exclude=active_cast_claim_row_ids()`.
        _ACTIVE_CAST_CLAIM_ROW_IDS[briefing_id] = script_id

        endpoint = provider or preset.get("provider") or default_briefing_provider()
        resolved_model = model or preset.get("model")
        model_used = f"{endpoint}/{resolved_model}" if resolved_model else endpoint

        try:
            character_texts = await _resolve_character_texts(roster, load_character)
            system, user = build_cast_prompt(
                briefing.get("body_markdown") or "",
                roster,
                preset.get("style_notes"),
                character_texts,
            )
            raw = await _invoke_chat(
                chat, endpoint=endpoint, model=resolved_model, system=system, user=user
            )
            turns = parse_script_turns(extract_response_content(raw), roster_names)
        except Exception as exc:  # noqa: BLE001 - every cast failure is a row
            # No traceback: see the module docstring's egress note -- the frame
            # here holds the prompt, so only the exception's type is logged.
            logger.warning(f"script {script_id}: cast failed: {type(exc).__name__}")
            return await asyncio.to_thread(
                _finish_script_failure, db, script_id, _error_text(exc)
            )

        return await asyncio.to_thread(
            _finish_script_success, db, script_id, json.dumps(turns), model_used
        )


def fail_interrupted_scripts(
    db: Any,
    briefing_id: int | None = None,
    *,
    exclude: Collection[int] = (),
    exclude_briefings: Collection[int] = (),
    max_row_id: int | None = None,
) -> int:
    """Fail every `generating` script as `interrupted`; return the count.

    Mirrors `briefing_service.fail_interrupted_briefings` exactly: a worker
    that crashed mid-cast leaves a `generating` row that would otherwise
    wedge a one-cast-at-a-time guard shut forever. Only `generating` rows
    are touched -- finished history keeps its status, its turns, and its
    own error text.

    Args:
        db: An open `SubscriptionsDB`.
        briefing_id: Scope the sweep to one briefing's scripts. `None`
            sweeps every briefing's scripts, which is what a startup pass
            wants.
        exclude: `briefing_scripts.id`s to spare even though their row
            reads `generating` -- phase 4's claim-aware sweep, row-scoped
            since task-1890 (generalizing task-1812). A `generating` row
            whose OWN id is in this collection is a LIVE, in-process cast,
            not a crash zombie, and must survive unconditionally, in every
            scope. Callers snapshot `active_cast_claim_row_ids()` on the
            event loop and pass the result here; this function is sync and
            runs under `asyncio.to_thread`, so it never reads the live
            claim registry itself. Defaults to `()`, so every pre-task-1890
            caller is unchanged.

            Prior to task-1890 this was briefing-scoped (`active_cast_
            claims()`), which over-protected: a genuine crash-zombie row
            left by an earlier process can coexist with a freshly-claimed
            live row for the SAME briefing (the crash predates the claim),
            and a briefing-scoped `exclude` shielded both rows, not just
            the live one. Scoping to the row's own id fixes that: only the
            actual live row survives, and a same-briefing zombie is swept
            exactly as if no claim existed at all.
        exclude_briefings: Briefing ids to spare even though a row reads
            `generating`, regardless of that row's own id (task-1890,
            mirroring `fail_interrupted_briefings`'s `exclude_watchlists`).
            Closes a window row-scoped `exclude` alone cannot: between a
            claim being taken and its row's id being recorded, `generate_
            script`'s `to_thread` hop has not yet resumed on the event loop
            to record it -- for that whole span the row (once inserted)
            reads `generating` and is named by no row id at all. Callers
            pass `pending_cast_claim_briefing_ids()` here, snapshotted on
            the event loop exactly like `exclude` -- NEVER `active_cast_
            claims()` itself, which would resurrect the pre-task-1890
            over-protection `exclude` was narrowed away from. Defaults to
            `()`, so every caller that predates this fix is unchanged.

        max_row_id: The highest row id this sweep may touch. The startup
            reconcile passes the boundary it captured before this process
            could insert anything (``Subscriptions/startup_reconcile.py``),
            which is what stops it failing rows this process's own scheduler
            created moments earlier (Qodo, PR #1972). ``None`` -- every
            pre-existing, UI-gated caller -- is unbounded as before; those
            callers protect live rows with the claim-registry ``exclude``
            arguments above instead.

    Returns:
        How many rows were failed.
    """
    sql = (
        "UPDATE briefing_scripts SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE status = ?"
    )
    params: list[Any] = [STATUS_FAILED, INTERRUPTED_ERROR, STATUS_GENERATING]
    if briefing_id is not None:
        sql += " AND briefing_id = ?"
        params.append(briefing_id)
    if exclude:
        placeholders = ",".join("?" for _ in exclude)
        sql += f" AND id NOT IN ({placeholders})"
        params.extend(exclude)
    if exclude_briefings:
        placeholders = ",".join("?" for _ in exclude_briefings)
        sql += f" AND briefing_id NOT IN ({placeholders})"
        params.extend(exclude_briefings)
    if max_row_id is not None:
        sql += " AND id <= ?"
        params.append(int(max_row_id))

    with db.transaction() as conn:
        count = conn.execute(sql, params).rowcount
    if count:
        logger.info(f"failed {count} interrupted briefing script(s)")
    return count


# --- Casting directly from a kept briefing (task-1780, Task 4) -------------
#
# `generate_script_from_text` casts a NEW script from a ChaChaNotes
# `kept_briefings` row (`Subscriptions/briefing_keep.py` is the one writer
# of that table) rather than a `Subscriptions_DB` `briefings` row, and
# writes into `kept_scripts` instead of `briefing_scripts`. Everything
# above this section is untouched by it: `generate_script` still owns
# `briefing_scripts` exclusively, this owns `kept_scripts` exclusively, and
# the two never share a row, a claim set, or an assumption about which
# database's shape it is looking at.
#
# One structural difference matters enough to restate here as well as in
# the function's own docstring: `briefing_scripts` has a `status` column
# (`generating`/`complete`/`failed`), so `generate_script` can insert a
# `generating` placeholder BEFORE the chat call and update it either way
# afterward -- a `failed` row is still a row a user (or a zombie sweep) can
# see. `kept_scripts` (the v28->v29 migration, deliberately) has NO status
# column: there is no "this kept briefing has a script that failed" state
# to represent, and inventing one would be a schema change this task does
# not otherwise need. So a kept-briefing cast writes NOTHING until the
# chat call AND the parse both succeed; a chat failure or a parse failure
# raises straight out of `generate_script_from_text` instead of degrading
# into a row, and the caller (Task 5's modal) is expected to catch that and
# show it as a toast -- there is no `kept_scripts` place left to show it
# instead. There is also, consequently, no `fail_interrupted_kept_scripts`
# sibling to `fail_interrupted_scripts`: a crash mid-cast leaves NO partial
# row behind to need recovering, since nothing was ever written.


#: The roster a cast uses when `preset_id=None` -- Task 5's kept-briefings
#: modal offers "app default" as a first-class choice in its preset
#: `Select` (mirroring `ArtifactsPane`'s own `_APP_DEFAULT_PRESET_LABEL`
#: idiom for briefing GENERATION's own preset picker), but casting a
#: script -- unlike generating a briefing's text -- has no meaning without
#: SOME roster (`validate_roster` refuses an empty one, and this function
#: takes no separate `roster` parameter to supply one another way). A
#: single unbound speaker named "Narrator" is the one roster this module's
#: own principle already describes as needing no special handling: "a
#: roster of one speaker produces narration through the identical path as
#: a roster of many -- there is no special 'narration mode'" (module
#: docstring); `test_briefing_cast.py`'s own `ONE_SPEAKER_ROSTER` uses the
#: identical name for the identical reason.
_APP_DEFAULT_SPEAKER_NAME = "Narrator"
_APP_DEFAULT_ROSTER: list[dict] = [
    {
        "name": _APP_DEFAULT_SPEAKER_NAME,
        "role_prompt": "",
        "character_card_id": None,
        "voice_profile_id": None,
    }
]

#: `kept_scripts.preset_name` recorded for a `preset_id=None` cast -- the
#: literal string the design spec and implementation plan both quote for
#: this case.
APP_DEFAULT_PRESET_NAME = "(app default)"


# --- In-process kept-cast claims (task-1780, Task 4) -----------------------
#
# `_ACTIVE_KEPT_CAST_CLAIMS` is its OWN set, keyed by `kept_briefings.id`,
# deliberately NOT sharing `_ACTIVE_CAST_CLAIMS` above. `kept_briefings.id`
# (ChaChaNotes, this task's new table) and `briefings.id`
# (`Subscriptions_DB`, phase 1) are autoincrement primary keys in two
# entirely separate SQLite database FILES with no relationship to one
# another whatsoever -- id 5 in one table says nothing about id 5 in the
# other. Sharing one claim set would make casting kept briefing 5
# spuriously collide with (block, or be blocked by) a live cast of
# briefing 5: two completely unrelated rows that merely happen to share a
# small integer. `GenerationInFlightError` IS still reused (not mirrored),
# for the identical reason `_claim_cast`'s own section comment gives: a
# caller wants ONE exception type to catch across every kind of "something
# is already generating", not a third near-identical type.
#
# Mutated ONLY on the event loop, for the identical reason
# `_ACTIVE_CAST_CLAIMS`'s own section comment states: the check-then-add
# pair inside `_claim_kept_cast` never awaits, so no lock is needed, and
# mutating this set from a thread (e.g. from inside a function
# `asyncio.to_thread` runs) would race with that invariant.

_ACTIVE_KEPT_CAST_CLAIMS: set[int] = set()


def active_kept_cast_claims() -> frozenset[int]:
    """Snapshot of kept briefing ids a live `generate_script_from_text` call holds.

    Its own accessor, mirroring `active_cast_claims`'s shape exactly but
    reading `_ACTIVE_KEPT_CAST_CLAIMS` -- see that set's own section
    comment for why the two are never merged. No `fail_interrupted_*`
    sibling consumes this today (unlike `active_cast_claims`, whose whole
    purpose is excluding live claims from `fail_interrupted_scripts`'s
    sweep): a kept cast never leaves a partial `kept_scripts` row for a
    sweep to find in the first place (see this section's opening comment),
    so there is nothing here for a startup sweep to spare. Provided anyway,
    for the same "tests can simulate another in-process caller" reason
    `active_cast_claims` is.
    """
    return frozenset(_ACTIVE_KEPT_CAST_CLAIMS)


@contextmanager
def _claim_kept_cast(kept_briefing_id: int) -> Iterator[None]:
    """Claim `kept_briefing_id` for the duration of one kept-briefing cast.

    Identical shape to `_claim_cast`, scoped to `_ACTIVE_KEPT_CAST_CLAIMS`
    instead of `_ACTIVE_CAST_CLAIMS` -- see that set's own section comment
    for why a shared set would be wrong here. Also usable directly by
    tests that need to simulate another in-process caller already holding
    a kept briefing.

    Args:
        kept_briefing_id: The `kept_briefings.id` about to be cast from.

    Raises:
        GenerationInFlightError: If `kept_briefing_id` is already claimed.
    """
    if kept_briefing_id in _ACTIVE_KEPT_CAST_CLAIMS:
        raise GenerationInFlightError(
            f"a script is already being cast for kept briefing {kept_briefing_id}"
        )
    _ACTIVE_KEPT_CAST_CLAIMS.add(kept_briefing_id)
    try:
        yield
    finally:
        _ACTIVE_KEPT_CAST_CLAIMS.discard(kept_briefing_id)


def _start_cast_from_text(
    chacha_db: Any,
    subs_db: Any,
    kept_briefing_id: int,
    preset_id: Optional[int],
) -> tuple[str, list[dict], set[str], Optional[str], Optional[str], Optional[str], str]:
    """Everything before the chat call, for a cast-from-kept: validate, resolve.

    Every check that must refuse WITHOUT ever creating a row runs here --
    the same no-orphan-row contract `_start_script` upholds for a live
    cast, taken one step further: unlike `_start_script`, this function
    never inserts anything, even on success (see this module section's
    opening comment on the asymmetry) -- it only reads and validates.

    Args:
        chacha_db: An open `CharactersRAGDB` holding the kept briefing.
        subs_db: An open `SubscriptionsDB`, consulted ONLY for
            `get_briefing_preset` -- Task 4's whole point (AC #4) is that
            this succeeds after the watchlist AND the original preset used
            to write this kept briefing are both already gone; `subs_db`
            is used here purely to resolve whatever preset the CALLER
            names right now, which may be an entirely different preset.
        kept_briefing_id: The `kept_briefings.id` to cast from.
        preset_id: A `briefing_presets.id` to resolve the roster,
            provider, model, and style notes from, or `None` for the
            app-default cast (see `APP_DEFAULT_PRESET_NAME`'s own
            comment).

    Returns:
        `(body, roster, roster_names, provider, model, style_notes,
        preset_name)`.

    Raises:
        ScriptCastError: If the kept briefing does not exist, its body is
            empty or whitespace-only, `preset_id` is given but does not
            resolve to a preset, or the resolved roster fails
            `validate_roster`.
    """
    kept = chacha_db.get_kept_briefing(kept_briefing_id)
    if kept is None:
        raise ScriptCastError(f"kept briefing {kept_briefing_id} does not exist")

    body = (kept.get("body_markdown") or "").strip()
    if not body:
        raise ScriptCastError(
            f"kept briefing {kept_briefing_id} has an empty body; refusing to cast"
        )

    if preset_id is None:
        provider = model = style_notes = None
        preset_name = APP_DEFAULT_PRESET_NAME
        roster = validate_roster(_APP_DEFAULT_ROSTER)
    else:
        preset = subs_db.get_briefing_preset(preset_id)
        if preset is None:
            raise ScriptCastError(f"briefing preset {preset_id} does not exist")
        provider = preset.get("provider")
        model = preset.get("model")
        style_notes = preset.get("style_notes")
        preset_name = preset["name"]
        roster = validate_roster(load_roster(preset["roster_json"]))

    roster_names = {speaker["name"] for speaker in roster}
    return body, roster, roster_names, provider, model, style_notes, preset_name


def _get_kept_script(chacha_db: Any, script_id: int) -> dict[str, Any]:
    """Read a just-created `kept_scripts` row back by id.

    Task 1's CRUD has no single-row getter for `kept_scripts` (only
    `list_kept_scripts`, `create_kept_script`, `kept_script_source_ids`) --
    this is a direct, parameterized SELECT against the fixed table name,
    mirroring `briefing_keep._watchlist_name`'s own precedent for a read
    no existing CRUD method covers.

    Args:
        chacha_db: An open `CharactersRAGDB`.
        script_id: The `kept_scripts.id` just returned by
            `create_kept_script`, so this row is guaranteed to exist.

    Returns:
        The row as a dict.
    """
    cursor = chacha_db.execute_query(
        "SELECT * FROM kept_scripts WHERE id = ?", (script_id,)
    )
    return dict(cursor.fetchone())


def _finish_cast_from_text(
    chacha_db: Any,
    kept_briefing_id: int,
    preset_name: str,
    roster: list[dict],
    load_character: Optional[Callable[[int], Optional[dict]]],
    turns: list[dict],
    model_used: str,
) -> dict[str, Any]:
    """Snapshot the roster, write the newly cast script, and read it back.

    The ONLY write this whole cast path ever performs -- called after a
    successful chat call and a successful parse, never before (see this
    module section's opening comment on the asymmetry). Grouped into one
    `asyncio.to_thread` hop by its caller, exactly like `_start_script`
    groups its own `_snapshot_roster` call together with its DB insert:
    `load_character` is a plain, blocking, synchronous callable, so
    resolving it here -- off the event loop -- rather than in
    `generate_script_from_text`'s own coroutine body is what keeps this
    function's caller from ever invoking it directly on the loop thread.

    `source_script_id=NULL`: this script was cast directly from a kept
    briefing's text, not mirrored from a `Subscriptions_DB`
    `briefing_scripts` row.
    """
    snapshot_roster = _snapshot_roster(roster, load_character)
    script_id = chacha_db.create_kept_script(
        kept_briefing_id,
        source_script_id=None,
        preset_name=preset_name,
        roster_snapshot_json=dump_roster(snapshot_roster),
        turns_json=json.dumps(turns),
        model_used=model_used,
    )
    return _get_kept_script(chacha_db, script_id)


async def generate_script_from_text(
    chacha_db: Any,
    kept_briefing_id: int,
    *,
    preset_id: Optional[int],
    subs_db: Any,
    chat: Callable[..., Any] = chat_api_call,
    load_character: Optional[Callable[[int], Optional[dict]]] = None,
) -> dict[str, Any]:
    """Cast a NEW script from a kept briefing's body, into `kept_scripts`.

    The re-casting half of task-1780 (design spec: "Re-casting without the
    watchlist"): a kept briefing has already survived its source
    watchlist's deletion (`Subscriptions/briefing_keep.py`), and this is
    what lets it be cast into a script AGAIN, later, with whatever preset
    exists at that moment -- including a preset, or a watchlist, that did
    not even exist when the briefing was originally written. `subs_db` is
    consulted for exactly one thing, `get_briefing_preset(preset_id)`, to
    resolve the roster and provider/model/style notes the caller asks for
    right now; it is never read or written for anything else, which is
    what lets AC #4 hold (casting still works after the original watchlist
    AND the original preset are both deleted from `subs_db`).

    Unlike `generate_script`, this function DOES raise for a chat or parse
    failure, not only for a pre-flight refusal -- see this module
    section's opening comment (above `_APP_DEFAULT_ROSTER`) for why:
    `kept_scripts` has no `status` column, so there is no honest `failed`
    row it could write instead. Every raise -- pre-flight (a missing kept
    briefing, an empty body, a missing preset) or in-band (a provider
    error, an unknown speaker, a malformed reply, a missing character
    card) -- leaves `kept_scripts` completely untouched; the caller (Task
    5's modal) is expected to catch it and show it as a toast, the way it
    would any other failed action with no row of its own to carry the
    error.

    The kept briefing row itself (`kept_briefings`) is never written by
    this function on ANY outcome -- success, a pre-flight refusal, or an
    in-band failure -- mirroring `generate_script`'s "the briefing is
    never touched" rule one level up: a kept briefing may be cast from any
    number of times, successfully or not, without any of those attempts
    leaving a mark on the kept artifact itself.

    Args:
        chacha_db: An open `CharactersRAGDB` holding the kept briefing (and
            where the new `kept_scripts` row is written on success).
        kept_briefing_id: The `kept_briefings.id` to cast from.
        preset_id: A `briefing_presets.id` (resolved via `subs_db`)
            supplying the roster, and, absent an explicit override, this
            cast's own provider, model, and style notes. `None` casts a
            single-speaker "Narrator" narration using the app's default
            provider and no style notes -- see `APP_DEFAULT_PRESET_NAME`.
        subs_db: An open `SubscriptionsDB`, consulted only to resolve
            `preset_id`. See this docstring's own AC #4 note above.
        chat: The chat seam. Defaults to `Chat_Functions.chat_api_call`;
            may be sync or async. The only seam faked in tests.
        load_character: Character card lookup by id, or `None` if
            unavailable. A roster speaker bound to a `character_card_id`
            that this cannot resolve fails the cast, naming the card --
            identical semantics to `generate_script`'s own parameter of
            the same name; the roster here always comes from the PRESET
            being cast with, never from the kept briefing's own snapshot
            (a kept briefing carries no roster of its own at all).

    Returns:
        The newly created `kept_scripts` row as a dict.

    Raises:
        ScriptCastError: If the kept briefing does not exist, its body is
            empty, `preset_id` is given but does not resolve to a preset,
            the resolved roster is invalid, a bound character card cannot
            be resolved, or the model's reply fails to parse. No
            `kept_scripts` row is written in any of these cases.
        GenerationInFlightError: If another in-process caller already
            holds `kept_briefing_id`'s cast claim (`_claim_kept_cast`,
            this module's OWN claim set -- see its section comment for why
            it is never `_ACTIVE_CAST_CLAIMS`). Raised before `_start_
            cast_from_text` ever runs, so no `kept_scripts` row is written
            for the refused attempt either.
    """
    with _claim_kept_cast(kept_briefing_id):
        (
            body,
            roster,
            roster_names,
            provider,
            model,
            style_notes,
            preset_name,
        ) = await asyncio.to_thread(
            _start_cast_from_text, chacha_db, subs_db, kept_briefing_id, preset_id
        )

        endpoint = provider or default_briefing_provider()
        model_used = f"{endpoint}/{model}" if model else endpoint

        try:
            character_texts = await _resolve_character_texts(roster, load_character)
            system, user = build_cast_prompt(body, roster, style_notes, character_texts)
            raw = await _invoke_chat(
                chat, endpoint=endpoint, model=model, system=system, user=user
            )
            turns = parse_script_turns(extract_response_content(raw), roster_names)
        except Exception as exc:  # noqa: BLE001 - re-raised; no row exists to record it on
            # No traceback -- see the module docstring's egress note -- and
            # no row either: unlike `generate_script`, there is no `kept_
            # scripts` row already started for this to become a `failed`
            # status on, so the exception propagates to the caller as-is.
            logger.warning(
                f"kept briefing {kept_briefing_id}: cast from text failed: "
                f"{type(exc).__name__}"
            )
            raise

        return await asyncio.to_thread(
            _finish_cast_from_text,
            chacha_db,
            kept_briefing_id,
            preset_name,
            roster,
            load_character,
            turns,
            model_used,
        )
