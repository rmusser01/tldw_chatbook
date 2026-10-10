"""One Chat-owned producer for detached Console turn configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence
from uuid import uuid4

from tldw_chatbook.Character_Chat.emote_directives import (
    CharacterEmoteAssetReference,
    CharacterEmoteRunSnapshot,
    project_character_emote_assets,
)
from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS
from tldw_chatbook.model_capabilities import is_vision_capable

from .attachment_core import max_history_images
from .console_dispatch_checkpoint import ConsoleLibraryItemScopeSnapshot
from .console_turn_context import (
    ConsoleCharacterAuthoritySnapshot,
    ConsoleTurnConfigurationSnapshot,
    capture_change_review_admission,
)

if TYPE_CHECKING:
    from .console_chat_models import ConsoleProviderSelection
    from .console_chat_store import ConsoleChatSession, ConsoleChatStore
    from .console_roleplay_identity import ConsolePresentationContext
    from .console_scratch_space import ConsoleScratchSnapshot
    from .console_turn_context import ConsoleProjectAuthoritySnapshot


# Console-composed builtin exclusions remain source-scoped; external/local
# profiles with these names keep their original permission-governed route.
CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS: frozenset = frozenset(
    tuple(LIBRARY_TOOL_DESCRIPTORS)
    + (
        "search_rag",
        "search_notes",
        "search_conversations",
        "get_conversation_history",
        "export_conversation",
    )
)


_UNSET_PLUGIN_SERVICE = object()


def _character_emote_snapshot_from_graph(
    actor_id: int | None,
    graph: Mapping[str, Any] | None,
    *,
    fallback_reason: str,
) -> CharacterEmoteRunSnapshot:
    """Project an active graph into a detached, bounded turn snapshot."""
    if graph is None:
        return CharacterEmoteRunSnapshot(
            actor_id=actor_id, fallback_reason=fallback_reason
        )
    try:
        pack_id = int(graph["pack"]["id"])
        pack_version_id = int(graph["version"]["id"])
        raw_assets = tuple(graph["assets"])
        if pack_id < 1 or pack_version_id < 1:
            raise ValueError
        # Preserve the newer single-pass projection while moving the result
        # into the detached turn snapshot used by custody and recovery.
        sources = project_character_emote_assets(raw_assets)
        assets: list[CharacterEmoteAssetReference] = []
        for state, source in sources.items():
            if not isinstance(source, Mapping):
                continue
            asset_id = source.get("id")
            expression_key = source.get("expression_key")
            if (
                isinstance(asset_id, bool)
                or not isinstance(asset_id, int)
                or asset_id < 1
                or not isinstance(expression_key, str)
            ):
                continue
            assets.append(
                CharacterEmoteAssetReference(
                    state=state,
                    expression_key=expression_key,
                    asset_id=asset_id,
                )
            )
        return CharacterEmoteRunSnapshot(
            actor_id=actor_id,
            pack_id=pack_id,
            pack_version_id=pack_version_id,
            states=tuple(asset.state for asset in assets),
            assets=tuple(assets),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return CharacterEmoteRunSnapshot(
            actor_id=actor_id, fallback_reason="resolver_error"
        )


def capture_character_authority(
    session: ConsoleChatSession, repository: Any | None = None
) -> ConsoleCharacterAuthoritySnapshot | None:
    """Freeze the identity fence that may authorize emotes for this turn."""
    if session.assistant_kind != "character":
        return None
    local_character_id = session.local_character_id()
    graph = None
    fallback_reason = "no_active_pack"
    if local_character_id is not None and repository is not None:
        try:
            graph = repository.get_active_actor_pack("character", local_character_id)
        except Exception:  # noqa: BLE001 -- uncertainty freezes no emote grant
            fallback_reason = "resolver_error"
    return ConsoleCharacterAuthoritySnapshot(
        identity_revision=session.identity_revision,
        runtime_backend=session.runtime_backend,
        assistant_id=session.assistant_id,
        assistant_authority_id=session.assistant_authority_id,
        local_character_id=local_character_id,
        emote_snapshot=_character_emote_snapshot_from_graph(
            local_character_id, graph, fallback_reason=fallback_reason
        ),
    )


def capture_prompt_transform_inputs(
    app: Any, session: ConsoleChatSession
) -> dict[str, Any]:
    """Capture bounded dictionary/world inputs without retaining a screen."""
    conversation_id = session.persisted_conversation_id
    db = getattr(app, "chachanotes_db", None)
    dictionary_entries: tuple[Any, ...] = ()
    world_books: tuple[Any, ...] = ()
    world_enabled = False
    if db is not None and conversation_id:
        try:
            from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as cdl

            dictionary_entries = tuple(
                cdl.collect_active_chatdict_entries(db, conversation_id, None)
            )
        except Exception:  # noqa: BLE001 -- optional prompt context fails closed
            dictionary_entries = ()
        try:
            from tldw_chatbook.Character_Chat.world_info_resolver import (
                _collect_active_world_books,
            )
            from tldw_chatbook.config import get_cli_setting

            books, _has_character_book = _collect_active_world_books(
                db, conversation_id, None
            )
            world_books = tuple(books)
            world_enabled = bool(
                get_cli_setting("character_chat", "enable_world_info", True)
            )
        except Exception:  # noqa: BLE001 -- optional prompt context fails closed
            world_books = ()
            world_enabled = False
    return {
        "conversation_id": conversation_id,
        "dictionary_entries": dictionary_entries,
        "world_books": world_books,
        "world_enabled": world_enabled,
    }


def _empty_local_skill_context() -> dict[str, Any]:
    """Represent a completed local capture that granted no skill authority."""
    return {
        "backend": "local",
        "available_skills": (),
        "blocked_skills": (),
        "context_text": "",
    }


def capture_skill_context_maximum(
    app: Any,
    workspace_id: str | None = None,
    *,
    _plugin_service: Any = _UNSET_PLUGIN_SERVICE,
) -> dict[str, Any]:
    """Capture the currently eligible local-skill catalog synchronously."""
    scope = getattr(app, "skills_scope_service", None)
    local = getattr(scope, "local_service", None) or getattr(
        app, "local_skills_service", None
    )
    if local is None:
        return _empty_local_skill_context()
    try:
        records = local._visible_records()  # noqa: SLF001 -- app-owned snapshot seam
        return _capture_skill_context_from_records(
            local, records, workspace_id, _plugin_service=_plugin_service
        )
    except Exception:  # noqa: BLE001 -- uncertainty freezes an empty maximum
        return _empty_local_skill_context()


def _capture_skill_context_from_records(
    local: Any,
    records: Mapping[str, Mapping[str, Any]],
    workspace_id: str | None,
    *,
    _plugin_service: Any = _UNSET_PLUGIN_SERVICE,
) -> dict[str, Any]:
    """Project one already-captured catalog without another native enumeration."""
    available: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for _, record in sorted(records.items()):
        summary = local._summary_for_record(record)  # noqa: SLF001
        # A built-in never reads trust; with no definition_digest the
        # later digest gates skip it, and execute re-verifies its pins.
        is_builtin = record.get("source") == "builtin"
        trust = None if is_builtin else getattr(local, "trust_service", None)
        if not summary.get("trust_blocked") and trust is not None:
            summary["definition_digest"] = trust.current_fingerprint_digest(
                str(summary.get("name", ""))
            )
        (blocked if summary.get("trust_blocked") else available).append(summary)
    plugin_service = (
        getattr(local, "plugin_service", None)
        if _plugin_service is _UNSET_PLUGIN_SERVICE
        else _plugin_service
    )
    if plugin_service is not None:
        plugin_context = plugin_service.capture_maximum(workspace_id)
        names = {item.get("name") for item in available + blocked}
        available.extend(
            row
            for row in plugin_context["available_skills"]
            if row["name"] not in names
        )
    return {
        "plugin_run_id": "pending:" + str(uuid4()),
        "available_skills": available,
        "blocked_skills": blocked,
        "context_text": "\n".join(
            f"- {item['name']}" for item in available if item.get("name")
        ),
        "backend": "local",
    }


def capture_mcp_definition_maximum(app: Any) -> dict[str, str]:
    """Capture exact eligible MCP identities and definition hashes."""
    service = getattr(app, "unified_mcp_service", None)
    if service is None:
        return {}
    try:
        if service.get_kill_switch():
            return {}
        from tldw_chatbook.MCP.hub_tool_catalog import (
            builtin_tools_from_inventory,
            local_tools_from_record,
        )

        local_service = getattr(service, "local_service", None)
        tools: list[Any] = []
        if local_service is not None:
            for record in local_service.get_external_servers() or ():
                tools.extend(local_tools_from_record(record))
            inventory = local_service.get_inventory()
            if isinstance(inventory, Mapping):
                tools.extend(builtin_tools_from_inventory(inventory))
        effective = service.effective_tool_states(tools)
        from tldw_chatbook.MCP.permission_store import definition_hash

        return {
            tool.tool_id: definition_hash(tool.description, tool.input_schema)
            for tool in tools
            if not (
                tool.server_key == "builtin:tldw_chatbook"
                and tool.name in CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS
            )
            and getattr(effective.get((tool.server_key, tool.name)), "state", "ask")
            != "deny"
        }
    except Exception:  # noqa: BLE001 -- uncertainty freezes an empty maximum
        return {}


_UNSET_SKILL_CONTEXT = object()
_UNSET_PREPARED_INPUT = object()


def capture_console_turn_configuration(
    app: Any,
    store: ConsoleChatStore,
    session_id: str,
    *,
    provider_selection: ConsoleProviderSelection,
    scratch_space: ConsoleScratchSnapshot | None,
    presentation_context: ConsolePresentationContext | None,
    rag_defaults: Mapping[str, Any],
    tool_configuration: Mapping[str, Any],
    project_authority: ConsoleProjectAuthoritySnapshot | None,
    skill_workspace_id: str | None,
    character_repository: Any | None,
    tool_policy_profile_id: str | None,
    persona_policy_rules: Sequence[Mapping[str, Any]] | None,
    mcp_definition_maximum: Mapping[str, str] | None = None,
    _require_current=None,
    _plugin_service: Any = _UNSET_PLUGIN_SERVICE,
    _skill_context_maximum: Any = _UNSET_SKILL_CONTEXT,
    _change_review_admission: Any = _UNSET_PREPARED_INPUT,
    _character_authority: Any = _UNSET_PREPARED_INPUT,
    _prompt_transform_inputs: Any = _UNSET_PREPARED_INPUT,
) -> ConsoleTurnConfigurationSnapshot:
    """Capture common domain state using the adapters' explicit selected values.

    Args:
        app: Owning application services, never a Console screen.
        store: Owning session store.
        session_id: Selected session, independent of the foreground session.
        provider_selection: Detached provider selection from the existing adapter.
        scratch_space: Existing scratch snapshot for that session.
        presentation_context: Existing adapter's presentation snapshot.
        rag_defaults: Existing route's selected retrieval defaults.
        tool_configuration: Existing route's tool configuration and budget.
        project_authority: Existing adapter's project authority capture.
        skill_workspace_id: Existing route's skill workspace, including None.
        character_repository: App-owned character/emote repository, if available.
        tool_policy_profile_id: Explicit profile value; None keeps default semantics.
        persona_policy_rules: Explicit rules; None keeps the empty posture.
        mcp_definition_maximum: Prepared definitions, or None for fresh sync capture.
        _change_review_admission: Prepared roots, aliases and skipped roots.
        _character_authority: Prepared character authority, including None.
        _prompt_transform_inputs: Prepared prompt-transform inputs.

    Returns:
        The complete detached snapshot. It carries no execution permission and
        retains no screen, controller or adapter callback.
    """

    def checked(callback, *args, **kwargs):
        value = callback(*args, **kwargs)
        if _require_current is not None:
            _require_current()
        return value

    session = next(item for item in store.sessions() if item.id == session_id)
    held_scope = session.rag_scope_holder.scope
    workspace_id = store.session_workspace_id(session_id)
    if _change_review_admission is _UNSET_PREPARED_INPUT:
        roots, aliases, skipped = checked(
            capture_change_review_admission, app, workspace_id
        )
    else:
        roots, aliases, skipped = _change_review_admission
    model = provider_selection.explicit_model or provider_selection.configured_model
    if mcp_definition_maximum is None:
        mcp_definition_maximum = capture_mcp_definition_maximum(app)
    return ConsoleTurnConfigurationSnapshot.capture(
        session_id=session_id,
        provider_selection=provider_selection,
        scratch_space=scratch_space,
        session_settings=store.effective_session_settings(session_id),
        workspace_roots=roots,
        change_review_root_aliases=aliases,
        change_review_skipped_roots=skipped,
        tool_policy_profile_id=tool_policy_profile_id,
        persona_policy_rules=persona_policy_rules,
        presentation_context=presentation_context,
        library_policy_maximum=session.library_policy_holder.snapshot,
        library_scope_maximum=ConsoleLibraryItemScopeSnapshot(
            note_ids=(
                tuple(
                    str(item.source_id)
                    for item in held_scope.items
                    if item.source_type == "note"
                )
                if held_scope is not None
                else ()
            ),
            media_ids=(
                tuple(
                    str(item.source_id)
                    for item in held_scope.items
                    if item.source_type == "media"
                )
                if held_scope is not None
                else ()
            ),
            conversations_allowed=held_scope is None,
        ),
        project_authority=project_authority,
        character_authority=(
            checked(capture_character_authority, session, character_repository)
            if _character_authority is _UNSET_PREPARED_INPUT
            else _character_authority
        ),
        prompt_transform_inputs=(
            checked(capture_prompt_transform_inputs, app, session)
            if _prompt_transform_inputs is _UNSET_PREPARED_INPUT
            else _prompt_transform_inputs
        ),
        skill_context_maximum=(
            checked(
                capture_skill_context_maximum,
                app,
                skill_workspace_id,
                **(
                    {}
                    if _plugin_service is _UNSET_PLUGIN_SERVICE
                    else {"_plugin_service": _plugin_service}
                ),
            )
            if _skill_context_maximum is _UNSET_SKILL_CONTEXT
            else _skill_context_maximum
        ),
        mcp_tool_maximum=mcp_definition_maximum,
        mcp_definition_maximum=mcp_definition_maximum,
        capabilities={
            "vision": bool(model)
            and is_vision_capable(provider_selection.provider, model or ""),
            "max_history_images": max_history_images(
                provider_selection.provider, model
            ),
        },
        rag_defaults=rag_defaults,
        tool_configuration=tool_configuration,
        provider_payload_settings={
            "streaming": provider_selection.streaming,
            "temperature": provider_selection.temperature,
            "top_p": provider_selection.top_p,
            "min_p": provider_selection.min_p,
            "top_k": provider_selection.top_k,
            "max_tokens": provider_selection.max_tokens,
            "seed": provider_selection.seed,
            "presence_penalty": provider_selection.presence_penalty,
            "frequency_penalty": provider_selection.frequency_penalty,
            "reasoning_effort": provider_selection.reasoning_effort,
            "reasoning_summary": provider_selection.reasoning_summary,
            "verbosity": provider_selection.verbosity,
            "thinking_effort": provider_selection.thinking_effort,
            "thinking_budget_tokens": provider_selection.thinking_budget_tokens,
        },
    )


# Same definition-time capsule format as the original stock skill source proof.
_CONSOLE_SKILL_CAPTURE_FUNCTIONS = {
    "_capture_skill_context_from_records": _capture_skill_context_from_records,
    "_empty_local_skill_context": _empty_local_skill_context,
}
_CONSOLE_SKILL_CAPTURE_SOURCE = (
    globals(),
    __file__,
    __spec__,
    getattr(__spec__, "origin", None),
    tuple(
        (globals(), name, function)
        for name, function in _CONSOLE_SKILL_CAPTURE_FUNCTIONS.items()
    )
    + (
        (
            globals(),
            "_CONSOLE_SKILL_CAPTURE_FUNCTIONS",
            _CONSOLE_SKILL_CAPTURE_FUNCTIONS,
        ),
        (globals(), "_UNSET_PLUGIN_SERVICE", _UNSET_PLUGIN_SERVICE),
        (globals(), "uuid4", uuid4),
    ),
    tuple(
        (
            _CONSOLE_SKILL_CAPTURE_FUNCTIONS,
            name,
            function,
            function.__code__,
            function.__globals__,
            function.__defaults__,
            function.__kwdefaults__,
            tuple((function.__kwdefaults__ or {}).items()),
            function.__closure__,
            tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
            vars(function).get("__wrapped__"),
        )
        for name, function in _CONSOLE_SKILL_CAPTURE_FUNCTIONS.items()
    ),
)
