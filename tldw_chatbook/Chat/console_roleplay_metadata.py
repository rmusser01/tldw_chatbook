"""Guarded persistence helpers for Console roleplay conversation metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Chat.console_roleplay_identity import (
    ChatDisplayNameError,
    normalize_chat_display_name,
)


ROLEPLAY_CONTEXT_METADATA_KEY = "console_roleplay_context"
ROLEPLAY_CONTEXT_VERSION = 1


class RoleplayContextVersionError(ValueError):
    """A durable roleplay context belongs to a newer application version."""


@dataclass(frozen=True, slots=True)
class ConsoleRoleplayContext:
    """Trusted, optional roleplay context stored on a conversation."""

    user_name_override: str | None = None
    character_system_template: str | None = None


def parse_console_roleplay_context(raw_metadata: object) -> ConsoleRoleplayContext:
    """Safely project one version-one roleplay context from conversation metadata.

    Corrupt, incomplete, or future data is deliberately treated as absent so a
    conversation can still be restored by a build that does not understand it.
    """
    metadata = _metadata_object(raw_metadata)
    owned_context = metadata.get(ROLEPLAY_CONTEXT_METADATA_KEY)
    if not isinstance(owned_context, Mapping):
        return ConsoleRoleplayContext()
    version = owned_context.get("version")
    if not _is_integer(version) or version != ROLEPLAY_CONTEXT_VERSION:
        return ConsoleRoleplayContext()

    try:
        user_name_override = normalize_chat_display_name(
            owned_context.get("user_name_override"), blank_means_none=True
        )
    except ChatDisplayNameError:
        return ConsoleRoleplayContext()

    character_system_template = owned_context.get("character_system_template")
    if character_system_template is not None and (
        not isinstance(character_system_template, str)
        or not character_system_template.strip()
    ):
        return ConsoleRoleplayContext()

    return ConsoleRoleplayContext(
        user_name_override=user_name_override,
        character_system_template=character_system_template,
    )


def merge_console_roleplay_context(
    raw_metadata: object, context: ConsoleRoleplayContext
) -> str:
    """Merge trusted roleplay values without rewriting unrelated metadata.

    A build must not overwrite fields it cannot understand. Future versioned
    owned data therefore blocks this durable write, while safe read paths keep
    degrading to an empty context.
    """
    metadata = _metadata_object(raw_metadata)
    owned_context = metadata.get(ROLEPLAY_CONTEXT_METADATA_KEY)
    existing_version = (
        owned_context.get("version") if isinstance(owned_context, Mapping) else None
    )
    if (
        _is_integer(existing_version)
        and existing_version > ROLEPLAY_CONTEXT_VERSION
    ):
        raise RoleplayContextVersionError(
            "Cannot overwrite Console roleplay context at version "
            f"{existing_version}."
        )

    user_name_override = normalize_chat_display_name(
        context.user_name_override, blank_means_none=True
    )
    character_system_template = context.character_system_template
    if (
        not isinstance(character_system_template, str)
        or not character_system_template.strip()
    ):
        character_system_template = None

    if user_name_override is None and character_system_template is None:
        metadata.pop(ROLEPLAY_CONTEXT_METADATA_KEY, None)
    else:
        metadata[ROLEPLAY_CONTEXT_METADATA_KEY] = {
            "version": ROLEPLAY_CONTEXT_VERSION,
            **(
                {"user_name_override": user_name_override}
                if user_name_override is not None
                else {}
            ),
            **(
                {"character_system_template": character_system_template}
                if character_system_template is not None
                else {}
            ),
        }
    return json.dumps(metadata, sort_keys=True)


def _metadata_object(raw_metadata: object) -> dict[str, Any]:
    """Return a writable outer metadata object, or an empty safe fallback."""
    if isinstance(raw_metadata, Mapping):
        return dict(raw_metadata)
    if not isinstance(raw_metadata, str) or not raw_metadata:
        return {}
    try:
        decoded = json.loads(raw_metadata)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _is_integer(value: object) -> bool:
    """Return whether a JSON value is an integer but not boolean."""
    return isinstance(value, int) and not isinstance(value, bool)
