"""Prepared, exact-target Persona changes for Buddy management (ADR-139)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from ..Persona_Buddy.interaction import BuddyBinding
from ..Workspaces.models import DEFAULT_WORKSPACE_ID, WorkspaceAssistantDefaults
from .console_assistant_defaults import build_persona_agent_system_prompt
from .console_chat_store import ConsoleSettingsComponent
from .console_generation_settings_metadata import (
    ConsoleGenerationSettingsReadStatus,
    merge_console_generation_settings,
    snapshot_from_session_settings,
    strict_json_metadata_object,
)
from .console_turn_preparation import ConsoleTurnPreparationState


@dataclass(frozen=True)
class PreparedPersonaAssignment:
    """Read-only admission result; Apply owns the later domain mutation."""

    _apply: Callable[[], None]

    async def apply(self) -> None:
        """Commit without yielding between the final check and live publication.

        This bounded local SQLite update follows the existing synchronous store
        persistence seam. No provider, artwork, filesystem or network work occurs
        between admission and publication.
        """
        self._apply()


def _persona(app: Any, persona_id: str) -> dict[str, Any]:
    try:
        row = app.local_character_persona_service.get_persona_profile(persona_id)
    except (AttributeError, KeyError, ValueError) as exc:
        raise ValueError(
            "Selected Persona is unavailable. Reopen Buddy settings."
        ) from exc
    if (
        not isinstance(row, Mapping)
        or row.get("id") != persona_id
        or row.get("deleted")
        or not row.get("is_active", True)
        or row.get("backend", "local") != "local"
    ):
        raise ValueError(
            "Selected Persona is unavailable or inactive. Choose another Persona."
        )
    return deepcopy(dict(row))


def _session_identity(session: Any) -> tuple[Any, ...]:
    return (
        session.id,
        session.persisted_conversation_id,
        session.conversation_binding_revision,
        session.workspace_id,
        session.runtime_backend,
        session.ephemeral,
        session.assistant_kind,
        session.assistant_id,
        session.assistant_authority_id,
        session.character_id,
        session.character_name,
        session.persona_memory_mode,
        session.identity_revision,
        session.generation_settings_revision,
        session.settings_revision,
        session.settings,
        session.ephemeral_endpoint_policy,
    )


def _require_idle(store: Any, controller: Any, session: Any) -> None:
    # The direct queue projection is read-only; controller.activity_for can hydrate
    # recovery state and is deliberately avoided by preparation.
    activity = controller.prompt_queue_coordinator.activity(session.id)
    preparation = store.preparation_for_session(session.id)
    if (
        not controller.run_state_for(session.id).is_send_allowed
        or activity.occupies_slot
        or activity.preparing_before_acceptance
        or activity.accepted_live_turn
        or activity.has_queued_work
        or activity.needs_approval
        or controller.has_pending_approval_round(session.id)
        or store.dispatch_recovery_blocks_submission(session.id)
        or (
            preparation is not None
            and preparation.state
            not in {
                ConsoleTurnPreparationState.CANCELLED,
                ConsoleTurnPreparationState.SETTLED,
            }
        )
        or any(
            controller._interrupt_host.session_round_payloads(kind, session.id)
            for kind in (
                "approval",
                "skill_install",
                "skill_script",
                "question",
                "worktree_merge",
            )
        )
    ):
        raise ValueError(
            "Finish this conversation’s run, queued work or pending decision before changing its Persona."
        )
    lifecycle = store._settings_persistence_lifecycles.get(session.id)
    if lifecycle is not None and (
        lifecycle.lock.locked()
        or lifecycle.drain is not None
        or any(not task.done() for task in lifecycle.tasks)
    ):
        raise ValueError(
            "Conversation settings are still saving. Retry the Persona change afterwards."
        )


def _require_memory_confirmation(
    old_id: str | None, mode: str | None, choice: str | None
) -> None:
    if choice is not None and choice != old_id and mode == "read_write":
        raise ValueError(
            "This target permits memory writes. Review its Persona memory setting before assigning a different Persona."
        )


def prepare_buddy_persona_assignment(
    app: Any, binding: BuddyBinding | None, target: Any, persona_choice: str
) -> PreparedPersonaAssignment:
    """Validate and capture an assignment without changing the target or Persona.

    Args:
        app: Application owning local Persona and Console/workspace services.
        binding: Explicit conversation or workspace chosen by the user.
        target: Exact record resolved by the management coordinator.
        persona_choice: A local Persona id, ``#none``, or ``#unchanged``.

    Returns:
        An awaitable Apply command with its own stale-target guards.

    Raises:
        ValueError: The target/Persona is unavailable or changing it is unsafe.
    """
    if persona_choice == "#unchanged":
        return PreparedPersonaAssignment(lambda: None)
    if not isinstance(binding, BuddyBinding) or target is None:
        raise ValueError(
            "Choose a conversation or workspace before changing its Persona."
        )
    if not isinstance(persona_choice, str) or not persona_choice:
        raise ValueError("Choose a saved Persona or None.")
    choice = None if persona_choice == "#none" else persona_choice
    persona = _persona(app, choice) if choice is not None else None

    def validate_persona() -> None:
        if choice is not None and _persona(app, choice) != persona:
            raise ValueError(
                "Selected Persona changed. Reopen Buddy settings to review it."
            )

    if binding.kind == "workspace":
        registry = getattr(app, "workspace_registry_service", None)
        if registry is None:
            raise ValueError("Workspace storage is unavailable.")
        original = registry.get_workspace(binding.target_id)
        if original != target:
            raise ValueError("Workspace changed. Reopen Buddy settings.")

        def validate_workspace() -> Any:
            if getattr(app, "workspace_registry_service", None) is not registry:
                raise ValueError("Workspace profile changed. Reopen Buddy settings.")
            current = registry.get_workspace(binding.target_id)
            if (
                current is None
                or current.archived
                or current.workspace_id == DEFAULT_WORKSPACE_ID
                or str(getattr(current.authority, "value", current.authority))
                != "local-only"
                or current.created_at != original.created_at
                or current.assistant_defaults != original.assistant_defaults
                or current.assistant_defaults_explicit_none
                != original.assistant_defaults_explicit_none
            ):
                raise ValueError(
                    "Workspace default changed or is unavailable. Reopen Buddy settings."
                )
            return current

        validate_workspace()
        old = original.assistant_defaults
        _require_memory_confirmation(
            old.assistant_id if old else None,
            old.persona_memory_mode if old else None,
            choice,
        )
        defaults = (
            (
                replace(old, assistant_id=choice)
                if old is not None
                else WorkspaceAssistantDefaults(assistant_id=choice)
            )
            if choice is not None
            else None
        )

        def apply_workspace() -> None:
            validate_workspace()
            validate_persona()
            if defaults is None:
                if old is not None or not original.assistant_defaults_explicit_none:
                    registry.clear_assistant_defaults(
                        binding.target_id, expected_record=original
                    )
            elif defaults != old:
                registry.set_assistant_defaults(
                    binding.target_id, defaults, expected_record=original
                )

        return PreparedPersonaAssignment(apply_workspace)

    runtime = getattr(app, "console_runtime", None)
    store, controller = (
        getattr(runtime, "chat_store", None),
        getattr(runtime, "chat_controller", None),
    )
    if store is None or controller is None or controller.store is not store:
        raise ValueError(
            "Open the bound conversation in Console before changing its Persona."
        )

    def validate_session() -> None:
        if (
            getattr(app, "console_runtime", None) is not runtime
            or runtime.chat_store is not store
            or runtime.chat_controller is not controller
            or binding.resolve_session(store.sessions()) is not target
            or target.runtime_backend != "local"
            or target.assistant_kind not in (None, "generic", "persona")
            or target.character_id is not None
            or target.assistant_authority_id is not None
            or target.settings is None
        ):
            raise ValueError(
                "The bound local conversation is unavailable or uses a Character. Reopen Buddy settings."
            )
        _require_idle(store, controller, target)

    validate_session()
    original_identity = _session_identity(target)
    old_id = target.assistant_id if target.assistant_kind == "persona" else None
    _require_memory_confirmation(old_id, target.persona_memory_mode, choice)
    unchanged = choice == old_id
    settings = replace(
        target.settings,
        system_prompt=build_persona_agent_system_prompt(persona) if persona else None,
        character_label=str(persona.get("name") or choice) if persona else None,
        persona_memory_mode=(target.persona_memory_mode or "read_only")
        if persona
        else None,
    )
    snapshot = snapshot_from_session_settings(settings)
    persistence = store.persistence
    db = getattr(persistence, "db", None)
    conversation_id = target.persisted_conversation_id
    expected_row = (
        db.get_conversation_by_id(conversation_id)
        if db is not None and conversation_id
        else None
    )
    if conversation_id and expected_row is None:
        raise ValueError(
            "The saved conversation is unavailable. Reopen it before changing its Persona."
        )
    if expected_row is not None and (
        (expected_row.get("assistant_kind") or "generic")
        != (target.assistant_kind or "generic")
        or (expected_row.get("assistant_id") or "console")
        != (target.assistant_id or "console")
        or expected_row.get("assistant_authority_id") != target.assistant_authority_id
        or expected_row.get("character_id") != target.character_id
        or expected_row.get("persona_memory_mode") != target.persona_memory_mode
        or expected_row.get("runtime_backend", "local") != target.runtime_backend
        or (expected_row.get("system_prompt") or None)
        != (target.settings.system_prompt or None)
    ):
        raise ValueError(
            "Saved Persona identity changed. Reopen the conversation before changing its Persona."
        )

    def apply_session() -> None:
        with store.durable_preparation_lock, store.fork_source_transition(target.id):
            validate_session()
            validate_persona()
            if (
                _session_identity(target) != original_identity
                or store.persistence is not persistence
            ):
                raise ValueError(
                    "Conversation changed. Reopen Buddy settings to review it."
                )
            if unchanged:
                return
            if conversation_id is not None:
                with db.transaction():
                    row = db.get_conversation_by_id(conversation_id)
                    if row is None or row["version"] != expected_row["version"]:
                        raise ValueError(
                            "Saved conversation changed. Reopen it before changing its Persona."
                        )
                    metadata = strict_json_metadata_object(row.get("metadata") or {})
                    stored_settings = settings
                    if target.ephemeral_endpoint_policy is not None:
                        prior = metadata.get("console_session_settings", {})
                        stored_settings = replace(
                            settings,
                            base_url=prior.get("base_url")
                            if isinstance(prior, dict)
                            else None,
                        )
                    metadata = merge_console_generation_settings(metadata, snapshot)
                    metadata["console_session_settings"] = {
                        "version": 1,
                        **asdict(stored_settings),
                    }
                    if not db.update_conversation(
                        conversation_id,
                        {
                            "assistant_kind": "persona" if choice else None,
                            "assistant_id": choice,
                            "assistant_authority_id": None,
                            "character_id": None,
                            "persona_memory_mode": settings.persona_memory_mode,
                            "system_prompt": settings.system_prompt,
                            "metadata": json.dumps(metadata),
                        },
                        expected_version=expected_row["version"],
                    ):
                        raise ValueError("Could not save the Persona change.")
            # No await separates the committed local row and these plain dataclass
            # assignments. Existing settings writers use this same owner thread.
            target.settings = settings
            target.assistant_kind = "persona" if choice else "generic"
            target.assistant_id = choice or "console"
            target.persona_memory_mode = settings.persona_memory_mode
            target.assistant_default_notice = ""
            target.character_name = None
            target.canonical_settings_baseline = None
            target.has_user_work = True
            target.updated_at = datetime.now(UTC).isoformat()
            target.generation_settings_revision += 1
            target.settings_persistence_failures.pop(
                ConsoleSettingsComponent.GENERATION_SETTINGS, None
            )
            if conversation_id is not None:
                target.generation_durable_snapshot = snapshot
                target.generation_metadata_status = (
                    ConsoleGenerationSettingsReadStatus.VALID
                )
                store._seed_console_settings_owned_bases(target)
            store._bump_identity_revision(target.id)
            store._bump_settings_revision(target.id)
            store._bump_conversation_context_epoch(target.id)
            store._bump_speech_preference_epoch(target.id)

    return PreparedPersonaAssignment(apply_session)
