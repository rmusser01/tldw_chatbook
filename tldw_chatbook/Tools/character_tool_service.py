"""Console character-card tools (TASK-32954, spec §3.1/§4).

``CharacterToolService`` is the impure orchestration seam between the
Agents runtime's ``LocalToolSpec`` handlers (``dict -> str``) and:

- Task 1's ``LocalCharacterPersonaService`` (local-only character CRUD).
- Task 2's UI-free ``resolve_avatar`` helper.

It owns argument bounding, the per-session truncation guard, and the
approval-card summary (never full field text/paths in logs -- env rule).
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from tldw_chatbook.Character_Chat.character_avatar import (
    image_backend_configured,
    resolve_avatar,
)
from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError

_LOGGER = logging.getLogger(__name__)

CHARACTER_FIELD_READ_BOUND = 8_000
_SEARCH_DESCRIPTION_CHARS = 160
EDITABLE_FIELDS: tuple[str, ...] = (
    "name", "description", "personality", "scenario", "first_message",
    "message_example", "system_prompt", "post_history_instructions",
    "creator_notes", "creator", "character_version", "alternate_greetings", "tags",
)
SERVER_REFUSAL = (
    "Character editing is local-only; switch this chat to local to create or "
    "edit characters."
)
_PUBLIC_EXECUTION_ERROR = "Character tool execution failed"


class _InvalidArgument(ValueError):
    pass


class CharacterReadGuard:
    """Fields read in full, per Console session (spec §3.1 truncation guard)."""

    def __init__(self) -> None:
        self._full: set[tuple[int, int, str]] = set()

    def record_full(self, character_id: int, version: int, field: str) -> None:
        self._full.add((character_id, version, field))

    def permits(self, character_id: int, version: int, field: str) -> bool:
        return (character_id, version, field) in self._full


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _outcome(status: str, message: str, *, retryable: bool = False, **extra: Any) -> str:
    return _json({"status": status, "retryable": retryable, "message": message, **extra})


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    return "" if value is None else str(value)


def _bounded_int(args: Mapping[str, Any], key: str, default: int) -> int:
    """Parse an integer argument, raising ``_InvalidArgument`` on any bad type.

    A model-supplied ``"abc"`` or ``3.5`` must surface as ``invalid_argument``,
    not crash into ``_run``'s generic exception path as a 500-style error.
    """
    raw = args.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise _InvalidArgument(f"'{key}' must be an integer")
    try:
        return int(raw)
    except ValueError:
        raise _InvalidArgument(f"'{key}' must be an integer") from None


def _character_id(arguments: Mapping[str, Any], key: str = "id") -> int:
    raw = arguments.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise _InvalidArgument(f"'{key}' must be a positive integer")
    if isinstance(raw, str) and not re.fullmatch(r"[1-9][0-9]{0,17}", raw):
        raise _InvalidArgument(f"'{key}' must be a positive integer")
    value = int(raw)
    if value < 1:
        raise _InvalidArgument(f"'{key}' must be a positive integer")
    return value


def save_approval_summary(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Approval-card view of a save: field names and sizes, never text."""
    args = arguments if isinstance(arguments, Mapping) else {}
    summary: dict[str, Any] = {
        "action": "update" if args.get("id") is not None else "create",
        "character": (f"#{args['id']}" if args.get("id") is not None
                      else str(args.get("name") or "")[:80]),
        "changed_fields": {
            f: f"{len(_text(args[f]))} chars" for f in EDITABLE_FIELDS if f in args
        },
    }
    avatar = args.get("avatar")
    if isinstance(avatar, Mapping):
        source = avatar.get("source")
        if source == "generate":
            backend = image_backend_configured() or "none configured"
            summary["avatar"] = f"generate with {backend} (paid backends may cost money)"
        elif source == "file":
            summary["avatar"] = f"file: {avatar.get('path')}"
        elif source == "remove":
            summary["avatar"] = "remove"
    return summary


class CharacterToolService:
    def __init__(
        self,
        *,
        service_loader: Callable[[], Any],
        runtime_source_loader: Callable[[], str],
        read_guard: CharacterReadGuard,
        on_changed: Callable[[int], None] | None = None,
        generate_avatar: Callable[[str], bytes] | None = None,
    ) -> None:
        self._service_loader = service_loader
        self._runtime_source_loader = runtime_source_loader
        self._guard = read_guard
        self._on_changed = on_changed
        self._generate_avatar = generate_avatar

    # -- public handlers (LocalToolSpec.handler: dict -> str) ---------------
    def search(self, arguments: object) -> str:
        return self._run(self._search, arguments)

    def get(self, arguments: object) -> str:
        return self._run(self._get, arguments)

    def save(self, arguments: object) -> str:
        return self._run(self._save, arguments)

    def _run(self, fn: Callable[[Mapping[str, Any]], str], arguments: object) -> str:
        if self._runtime_source_loader() == "server":
            return _outcome("unsupported", SERVER_REFUSAL)
        args = arguments if isinstance(arguments, Mapping) else {}
        try:
            return fn(args)
        except _InvalidArgument as exc:
            return _outcome("invalid_argument", str(exc))
        except ValidationError as exc:
            errors = exc.errors()
            field = str(errors[0]["loc"][0]) if errors and errors[0].get("loc") else "argument"
            detail = errors[0]["msg"] if errors else "invalid value"
            return _outcome("invalid_argument", f"'{field}': {detail}")
        except Exception as exc:  # noqa: BLE001 - boundary: never raise into the agent loop
            _LOGGER.error("Character tool execution failed category=%s",
                          re.sub(r"[^A-Za-z0-9_.-]", "_", type(exc).__name__)[:64])
            raise RuntimeError(_PUBLIC_EXECUTION_ERROR) from None

    # -- search -------------------------------------------------------------
    def _search(self, args: Mapping[str, Any]) -> str:
        limit = _bounded_int(args, "limit", 10)
        offset = _bounded_int(args, "offset", 0)
        if not 1 <= limit <= 25 or offset < 0:
            raise _InvalidArgument("limit must be 1-25 and offset >= 0")
        service = self._service_loader()
        query = str(args.get("query") or "").strip()
        rows = (service.search_characters(query, limit=limit + 1) if query
                else service.list_characters(limit=limit + 1, offset=offset))
        rows = list(rows or [])
        items = [
            {
                "id": r["id"], "name": r.get("name"),
                "description": _text(r.get("description"))[:_SEARCH_DESCRIPTION_CHARS],
                "tags": r.get("tags") or [], "version": r.get("version"),
                "has_avatar": bool(r.get("image")), "updated_at": str(r.get("last_modified") or ""),
            }
            for r in rows[:limit]
        ]
        extra = {"next_offset": offset + limit} if (len(rows) > limit and not query) else {}
        return _json({"status": "ok", "items": items, **extra})

    # -- get ----------------------------------------------------------------
    def _load(self, character_id: int) -> dict[str, Any] | None:
        try:
            return self._service_loader().get_character(character_id)
        except ValueError:
            return None

    def _field_page(self, card: Mapping[str, Any], field: str, offset: int) -> dict[str, Any]:
        text = _text(card.get(field))
        end = offset + CHARACTER_FIELD_READ_BOUND
        page: dict[str, Any] = {"text": text[offset:end]}
        if end < len(text):
            page["truncated"] = True
            page["next_offset"] = end
        else:
            self._guard.record_full(int(card["id"]), int(card["version"]), field)
        return page

    def _get(self, args: Mapping[str, Any]) -> str:
        character_id = _character_id(args)
        card = self._load(character_id)
        if card is None:
            return _outcome("not_found", f"No character with id {character_id}.")
        field = args.get("field")
        if field is not None:
            if field not in EDITABLE_FIELDS:
                raise _InvalidArgument("unknown field")
            offset = _bounded_int(args, "offset", 0)
            if offset < 0:
                raise _InvalidArgument("offset must be >= 0")
            return _json({"status": "ok", "id": character_id, "version": card["version"],
                          "field": field, **self._field_page(card, field, offset)})
        fields = {f: self._field_page(card, f, 0) for f in EDITABLE_FIELDS}
        return _json({"status": "ok", "id": character_id, "version": card["version"],
                      "has_avatar": bool(card.get("image")), "fields": fields})

    # -- save ---------------------------------------------------------------
    @staticmethod
    def _avatar_request_with_fallback_prompt(
        avatar_req: Mapping[str, Any], card_for_prompt: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Give ``generate`` a usable prompt when the card has no description.

        ``resolve_avatar`` composes a prompt from the card's description when
        no explicit ``prompt`` is given, and reports a failure when both are
        blank (spec §4.3's "add a description" case -- correct when a real
        card genuinely has neither). A brand-new character being created with
        only a name still deserves a best-effort avatar rather than an
        automatic failure, so this fills in a minimal name-based prompt only
        when neither an explicit prompt nor a description exists.
        """
        if avatar_req.get("source") != "generate":
            return avatar_req
        if str(avatar_req.get("prompt") or "").strip():
            return avatar_req
        if str(card_for_prompt.get("description") or "").strip():
            return avatar_req
        name = str(card_for_prompt.get("name") or "the character").strip() or "the character"
        return {**avatar_req, "prompt": f"a portrait of {name}"}

    def _save(self, args: Mapping[str, Any]) -> str:
        changes = {f: args[f] for f in EDITABLE_FIELDS if f in args}
        if "name" in changes and not str(changes["name"] or "").strip():
            raise _InvalidArgument("name must not be blank")
        avatar_req = args.get("avatar")
        service = self._service_loader()
        creating = args.get("id") is None
        character_id: int | None = None
        expected: int | None = None
        if creating:
            if not str(changes.get("name") or "").strip():
                raise _InvalidArgument("name is required to create a character")
            existing = service._require_db().get_character_card_by_name(str(changes["name"]).strip())
            if existing:
                return _outcome("duplicate_name",
                                f"A character named {changes['name']} already exists "
                                f"(id {existing['id']}); update it or choose another name.")
            card_for_prompt: Mapping[str, Any] = changes
        else:
            character_id = _character_id(args)
            expected = args.get("expected_version")
            if not isinstance(expected, int) or isinstance(expected, bool):
                raise _InvalidArgument("expected_version is required to update")
            current = self._load(character_id)
            if current is None:
                return _outcome("not_found", f"No character with id {character_id}.")
            if int(current["version"]) != expected:
                return _outcome("stale_version",
                                "The card changed since you read it; re-read with character_get.")
            for f in changes:
                if (len(_text(current.get(f))) > CHARACTER_FIELD_READ_BOUND
                        and not self._guard.permits(character_id, expected, f)):
                    return _outcome("read_full_field_first",
                                    f"Read the full '{f}' field with character_get before changing it.")
            card_for_prompt = {**current, **changes}

        avatar_status, payload, clear_image = "none", dict(changes), False
        if isinstance(avatar_req, Mapping):
            request = self._avatar_request_with_fallback_prompt(avatar_req, card_for_prompt)
            outcome = resolve_avatar(request, card_for_prompt, generate=self._generate_avatar)
            if outcome.kind == "image" and outcome.image:
                payload["image_base64"] = base64.b64encode(outcome.image).decode("ascii")
                avatar_status = "saved"
            elif outcome.kind == "remove":
                if creating:
                    avatar_status = "none"  # nothing to clear on a brand-new character
                else:
                    clear_image, avatar_status = True, "saved"
            else:
                avatar_status = f"failed: {outcome.reason}"

        if creating:
            try:
                record = service.create_character(payload)
            except ConflictError:
                return _outcome("duplicate_name",
                                f"A character named {changes['name']} already exists; "
                                f"update it or choose another name.")
        else:
            try:
                record = service.update_character(character_id, payload,
                                                  expected_version=expected,
                                                  clear_image=clear_image)
            except ConflictError as exc:
                if "already exists" in str(exc):
                    return _outcome("duplicate_name",
                                    f"A character named '{changes.get('name')}' already exists; "
                                    f"choose another name.")
                return _outcome("stale_version",
                                "The card changed since you read it; re-read with character_get.")
            except ValueError:
                return _outcome("stale_version",
                                "The card changed since you read it; re-read with character_get.")
        saved_id = int(record["id"])
        if self._on_changed is not None:
            self._on_changed(saved_id)
        return _json({"status": "saved", "retryable": False, "id": saved_id,
                      "version": record.get("version"), "changed_fields": sorted(changes),
                      "avatar": avatar_status})
