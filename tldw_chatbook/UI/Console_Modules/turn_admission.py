"""Read a Console send's service-owned turn authority off the UI pump (TASK-33620.15).

Measured before this change (mounted harness, one send, the main thread
sampled every 1 ms): the turn configuration builder
(``ConsoleSessionController._build_console_turn_execution_context``) took
531 ms on a cold first send -- RAG profile depth 288 ms, the MCP definition
maximum 114 ms, the scratch space 53 ms, the skill catalog 45 ms, project
authority 30 ms -- and 25-28 ms warm (the MCP maximum alone 22 ms). Live
(Anthropic haiku, 160x45), a first send's builder held the UI pump for
613 ms, mostly in the MCP store's storage-admission checks. Every one of
those reads services, files and databases; none reads the Console store.

So those reads -- the ones below -- run on a worker thread right before the
synchronous stretch that re-checks the send gate, builds the snapshot and
accepts the turn. That stretch is unchanged: the builder still reads the
store there, and it uses the precaptured values only when the session inputs
they were read for still match, otherwise it reads them inline as before. A
precapture that fails is dropped, so any error surfaces where it did before.

Imported on the first send only (ADR-097 boot census).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Chat.console_chat_controller import (
    capture_character_authority,
    capture_mcp_definition_maximum,
    capture_project_instruction_authority,
    capture_prompt_transform_inputs,
    capture_skill_context_maximum,
)
from tldw_chatbook.Chat.console_turn_context import (
    capture_change_review_admission,
    resolve_turn_persona_policy_rules,
    resolve_turn_tool_policy_profile_id,
)


@dataclass(frozen=True, slots=True)
class TurnAuthority:
    """The service-owned values one turn's snapshot freezes."""

    key: tuple[Any, ...]
    project_authority: Any
    workspace_roots: tuple[Any, ...]
    change_review_root_aliases: tuple[str, ...]
    change_review_skipped_roots: tuple[Any, ...]
    tool_policy_profile_id: str
    persona_policy_rules: tuple[Mapping[str, Any], ...]
    mcp_definition_maximum: dict[str, str]
    scratch_space: Any
    character_authority: Any
    prompt_transform_inputs: dict[str, Any]
    skill_context_maximum: dict[str, Any]
    rag_top_k: Any


@dataclass(frozen=True, slots=True)
class AuthorityInputs:
    """What the authority is read for, taken from the store on the UI pump."""

    session_id: str
    session: Any
    workspace_id: str | None
    include_bindings: bool
    character_repository: Any

    @property
    def key(self) -> tuple[Any, ...]:
        """Every session value a capture below reads, for the staleness check."""
        session = self.session
        return (
            self.session_id,
            self.workspace_id,
            self.include_bindings,
            session.workspace_id,
            session.project_instruction_state,
            session.assistant_kind,
            session.assistant_id,
            session.assistant_authority_id,
            session.runtime_backend,
            session.character_id,
            session.identity_revision,
            session.persisted_conversation_id,
            session.conversation_binding_revision,
        )


#: The authority precaptured for the turn being launched in this context.
_PRECAPTURED: ContextVar[TurnAuthority | None] = ContextVar(
    "console_turn_authority", default=None
)


def authority_inputs(controller: Any, session_id: str) -> AuthorityInputs:
    """Read the authority's inputs from the store, as the builder does.

    Args:
        controller: The Console session controller.
        session_id: The owning session.

    Returns:
        The inputs; the session is a shallow copy, so a worker never reads the
        live store record.
    """
    from .session import _console_live_runtime_enabled

    app_config = controller._provider_readiness_app_config()
    console_config = (
        app_config.get("console", {}) if isinstance(app_config, Mapping) else {}
    )
    if not isinstance(console_config, Mapping):
        console_config = {}
    store = controller._ensure_console_chat_store()
    session = next(item for item in store.sessions() if item.id == session_id)
    include_bindings = bool(
        _console_live_runtime_enabled(
            getattr(controller.app_instance, "app_config", None), console_config
        )
        and not store.session_one_shot_prefill(session_id)
        and session.assistant_kind != "character"
    )
    repository = getattr(
        (
            controller._ensure_console_chat_controller()
            if hasattr(controller, "_ensure_console_chat_controller_fn")
            else None
        ),
        "_visual_identity_repository",
        None,
    )
    return AuthorityInputs(
        session_id=session_id,
        session=copy.copy(session),
        workspace_id=store.session_workspace_id(session_id),
        include_bindings=include_bindings,
        character_repository=repository,
    )


def capture_authority(
    controller: Any,
    inputs: AuthorityInputs,
    *,
    mcp_definition_maximum: Mapping[str, str] | None = None,
) -> TurnAuthority:
    """Read every service-owned value the turn snapshot freezes.

    Touches no store and no widget, so it may run on a worker thread.
    """
    app = getattr(controller, "app_instance", None)
    session = inputs.session
    roots, aliases, skipped = capture_change_review_admission(app, inputs.workspace_id)
    if mcp_definition_maximum is None:
        mcp_definition_maximum = capture_mcp_definition_maximum(app)
    return TurnAuthority(
        key=inputs.key,
        project_authority=capture_project_instruction_authority(
            session,
            getattr(app, "workspace_registry_service", None),
            include_bindings=inputs.include_bindings,
        ),
        workspace_roots=roots,
        change_review_root_aliases=aliases,
        change_review_skipped_roots=skipped,
        tool_policy_profile_id=resolve_turn_tool_policy_profile_id(
            app, inputs.workspace_id
        ),
        persona_policy_rules=resolve_turn_persona_policy_rules(app, session),
        mcp_definition_maximum=dict(mcp_definition_maximum),
        scratch_space=controller._scratch_snapshot_provider(inputs.session_id),
        character_authority=capture_character_authority(
            session, inputs.character_repository
        ),
        prompt_transform_inputs=capture_prompt_transform_inputs(app, session),
        skill_context_maximum=capture_skill_context_maximum(app),
        rag_top_k=controller._rag_top_k_accessor(),
    )


def authority_for(
    controller: Any,
    inputs: AuthorityInputs,
    *,
    mcp_definition_maximum: Mapping[str, str] | None = None,
) -> TurnAuthority:
    """The precaptured authority when its inputs still match, else a fresh read."""
    precaptured = _PRECAPTURED.get()
    if precaptured is not None and precaptured.key == inputs.key:
        return precaptured
    return capture_authority(
        controller, inputs, mcp_definition_maximum=mcp_definition_maximum
    )


async def precapture(controller: Any, session_id: str) -> TurnAuthority | None:
    """Read the authority on a worker thread; ``None`` if anything failed.

    Args:
        controller: The Console session controller.
        session_id: The owning session.

    Returns:
        The authority, or ``None`` so the builder reads it inline and raises
        any error where it always did.
    """
    from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker

    try:
        inputs = authority_inputs(controller, session_id)
    except Exception:  # noqa: BLE001 -- the builder reports it at admission
        return None
    try:
        return await asyncio.to_thread(
            run_finite_local_worker, capture_authority, controller, inputs
        )
    except Exception:  # noqa: BLE001 -- the builder reports it at admission
        return None


@contextlib.contextmanager
def applied(authority: TurnAuthority | None) -> Iterator[None]:
    """Offer ``authority`` to the snapshot builds made inside this block."""
    reset = _PRECAPTURED.set(authority)
    try:
        yield
    finally:
        _PRECAPTURED.reset(reset)
