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
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from typing import Any, Callable, Mapping, Optional

from loguru import logger

from ..Chat.Chat_Functions import chat_api_call, extract_response_content
from .briefing_service import STATUS_COMPLETE as _BRIEFING_STATUS_COMPLETE
from .briefing_service import _default_provider

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
        personality = str(card.get("personality") or "").strip()[:CHARACTER_TEXT_CHAR_CAP]
        description = str(card.get("description") or "").strip()[:CHARACTER_TEXT_CHAR_CAP]
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
    `{"speaker": str, "text": str}` objects, each speaker a name in
    `roster_names`, fails naming the specific defect.

    Args:
        text: Raw provider reply.
        roster_names: Valid speaker names -- `validate_roster`'s `name`
            fields for this cast's roster.

    Returns:
        `[{"speaker": str, "text": str}, ...]`, in reply order.

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
        if speaker not in roster_names:
            raise ScriptCastError(f"unknown speaker {speaker!r} in turn {index}")

        turns.append({"speaker": speaker, "text": turn_text})

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
        "max_tokens": CAST_MAX_TOKENS,
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
    """
    script_id, briefing, preset, roster, roster_names = await asyncio.to_thread(
        _start_script, db, briefing_id, preset_id, load_character
    )

    endpoint = provider or preset.get("provider") or _default_provider()
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
        logger.warning(f"script {script_id}: cast failed against {endpoint}: {type(exc).__name__}")
        return await asyncio.to_thread(_finish_script_failure, db, script_id, _error_text(exc))

    return await asyncio.to_thread(
        _finish_script_success, db, script_id, json.dumps(turns), model_used
    )


def fail_interrupted_scripts(db: Any, briefing_id: int | None = None) -> int:
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

    with db.transaction() as conn:
        count = conn.execute(sql, params).rowcount
    if count:
        logger.info(f"failed {count} interrupted briefing script(s)")
    return count
