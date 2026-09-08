"""Creation-only workspace Persona resolution (ADR-079 and ADR-139)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ..Workspaces.models import DEFAULT_WORKSPACE_ID
from .console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID
from .console_session_settings import (
    ConsoleSessionSettings,
    blank_console_session_settings,
)


@dataclass(frozen=True)
class ConsoleAssistantStartup:
    """Resolved settings and identity for one new conversation."""

    settings: ConsoleSessionSettings
    assistant_kind: str = "generic"
    assistant_id: str = "console"
    persona_memory_mode: str | None = None
    notice: str = ""


def build_persona_agent_system_prompt(record: Mapping[str, Any]) -> str:
    """Compose a Persona into the same prompt used by the Console preview."""
    from ..Character_Chat.Character_Chat_Lib import compose_character_card_text

    return (
        compose_character_card_text(
            name=str(record.get("name") or "Workspace Agent"),
            system_prompt=str(record.get("system_prompt") or ""),
            personality=str(record.get("personality") or ""),
            description=str(record.get("description") or ""),
            user_name="User",
        )
        or "Stay in character."
    )


def resolve_new_console_assistant(
    app: Any, workspace_id: str, settings: ConsoleSessionSettings | None
) -> ConsoleAssistantStartup:
    """Resolve the explicit target workspace once; absence never blocks a chat.

    Explicit assistant identities bypass this helper at the store's creation
    boundary. Supplied custom prompts retain their existing precedence.
    """
    config = getattr(app, "app_config", {})
    settings = settings or blank_console_session_settings(
        config if isinstance(config, Mapping) else {}
    )
    plain = ConsoleAssistantStartup(settings)
    if (
        workspace_id in (CONSOLE_GLOBAL_WORKSPACE_ID, DEFAULT_WORKSPACE_ID, "")
        or settings.system_prompt is not None
        or settings.character_label
        or settings.persona_memory_mode is not None
    ):
        return plain
    try:
        registry = getattr(app, "workspace_registry_service", None)
        workspace = registry.get_workspace(workspace_id) if registry else None
        if (
            workspace is None
            or workspace.archived
            or workspace.assistant_defaults is None
        ):
            return plain
        personas = getattr(app, "local_character_persona_service", None)
        record = None

        def lookup(persona_id: str) -> Mapping | None:
            nonlocal record
            try:
                record = personas.get_persona_profile(persona_id) if personas else None
            except (KeyError, ValueError):
                record = None
            return record

        from ..Workspaces.assistant_defaults import resolve_effective_assistant_default

        effective = resolve_effective_assistant_default(
            workspace.assistant_defaults, lookup
        )
        if effective.status != "available" or not isinstance(record, Mapping):
            return replace(
                plain,
                notice="Workspace default Persona unavailable "
                f"({effective.degraded_reason or 'persona_unavailable'}). Started with None.",
            )
        mode = effective.persona_memory_mode or "read_only"
        return ConsoleAssistantStartup(
            settings=replace(
                settings,
                system_prompt=build_persona_agent_system_prompt(record),
                character_label=effective.label or "Workspace Agent",
                persona_memory_mode=mode,
            ),
            assistant_kind="persona",
            assistant_id=effective.assistant_id,
            persona_memory_mode=mode,
        )
    except Exception:  # noqa: BLE001 - a unavailable default must not block creation
        return replace(
            plain,
            notice="Workspace default Persona unavailable (persona_unavailable). Started with None.",
        )
