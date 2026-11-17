"""Capture original visible-send values from the actual backend session."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from inspect import getattr_static
from types import MethodType

from . import console_chat_controller as controller_source
from .console_chat_models import ConsoleProviderSelection, ConsoleWorkspaceContext
from .console_chat_store import ConsoleChatStore
from .console_domain_services import _definition, _definition_current
from .console_execution_protocol import _SELECTION, _detach
from ..UI.Console_Modules import session as session_source
from ..UI.Console_Modules import turn_admission as admission_source
from ..UI.Screens import settings_library_rag_defaults as library_source


def capture_send_selection(backend, session, checked_config: Mapping) -> dict:
    """Use the same original builders and resident inputs as visible admission.

    The parent supplies its original screen-local RAG scope at the press. This
    projection is inert: the original received/configuration/native checks still
    govern the subsequent command.
    """
    from .console_execution_backend import ConsoleExecutionBackend

    if type(backend) is not ConsoleExecutionBackend:
        raise TypeError("Original Console backend is required")
    backend._require_owner_loop()
    backend.require_sources()
    store = backend.store
    if (
        type(store) is not ConsoleChatStore
        or store._sessions.get(session.id) is not session
        or store.active_session_id != session.id
        or session.settings is None
    ):
        raise RuntimeError("Console selected send session changed")
    for module, name, record in _FUNCTIONS:
        if getattr(module, name) is not record[0] or not _definition_current(record):
            raise RuntimeError("Console selected send builder changed")
    for name, record in _STORE_METHODS:
        method = getattr(store, name)
        if (
            type(method) is not MethodType
            or method.__self__ is not store
            or method.__func__ is not record[0]
            or getattr_static(type(store), name) is not record[0]
            or not _definition_current(record)
        ):
            raise RuntimeError("Console selected send reader changed")
    section = checked_config.get("console", {})
    section = section if isinstance(section, Mapping) else {}
    defaults = checked_config.get("chat_defaults", {})
    defaults = defaults if isinstance(defaults, Mapping) else {}
    settings = store.effective_session_settings(session.id)
    display_name = session_source._console_global_user_display_name(checked_config)
    enabled = admission_source.console_live_runtime_enabled(checked_config, section)
    selected = controller_source.build_console_provider_selection_from_settings(
        settings,
        app_config=checked_config,
        workspace_context=ConsoleWorkspaceContext(
            active_workspace_id=store.session_workspace_id(session.id)
        ),
        legacy_model=settings.model
        if settings.source == "user"
        else defaults.get("model"),
        endpoint_policy=store.session_ephemeral_endpoint_policy(session.id),
        identity_session=session,
        global_user_name=lambda: display_name,
    )
    if type(selected) is not ConsoleProviderSelection:
        raise TypeError("Original Console provider selection is required")
    provider = {
        item.name: getattr(selected, item.name)
        for item in fields(ConsoleProviderSelection)
    }
    provider["endpoint_provenance"] = selected.endpoint_provenance.value
    workspace = selected.workspace_context
    provider["workspace_context"] = {
        "active_workspace_id": workspace.active_workspace_id,
        "staged_sources": [
            {item.name: getattr(source, item.name) for item in fields(source)}
            for source in workspace.staged_sources
        ],
        "active_run_id": workspace.active_run_id,
        "handoff_id": workspace.handoff_id,
    }
    presentation = store.presentation_context(session.id, display_name)
    context = {
        item.name: getattr(presentation, item.name) for item in fields(presentation)
    }
    context["transcript_style"] = presentation.transcript_style.value
    boolean = controller_source.coerce_bool_setting
    tools = {
        "session_ephemeral": session.ephemeral,
        "agent_runtime_enabled": enabled,
        "native_tool_calls_enabled": boolean(
            section.get("native_tool_calls", True), True
        ),
        "local_tools_enabled": boolean(
            section.get("local_tools_enabled", False), False
        ),
        "direct_library_tools": library_source.load_direct_library_tools(
            checked_config
        ),
        "exchange_capture_enabled": boolean(
            section.get("exchange_capture", True), True
        ),
    }
    for name in (
        "project_instructions_startup_max_bytes",
        "project_instructions_nested_max_bytes",
    ):
        tools[name] = controller_source.coerce_int_setting(
            section.get(
                name, controller_source.DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES
            ),
            controller_source.DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
            minimum=controller_source.MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
            maximum=controller_source.MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
        )
    value = {
        "provider_selection": provider,
        "presentation_context": context,
        "rag_defaults": None,
        "tool_configuration": tools,
        "skill_workspace_id": None,
        "project_bindings_eligible": bool(
            enabled
            and not store.session_one_shot_prefill(session.id)
            and session.assistant_kind != "character"
        ),
        "agent_runtime_enabled": enabled,
    }
    _SELECTION(_detach(value, "selection", 0, set()), "selection")
    return value


_FUNCTIONS = tuple(
    (module, name, _definition(getattr(module, name)))
    for module, names in (
        (
            controller_source,
            (
                "build_console_provider_selection_from_settings",
                "coerce_bool_setting",
                "coerce_int_setting",
            ),
        ),
        (session_source, ("_console_global_user_display_name",)),
        (admission_source, ("console_live_runtime_enabled",)),
        (library_source, ("load_direct_library_tools",)),
    )
    for name in names
)
_STORE_METHODS = tuple(
    (name, _definition(getattr(ConsoleChatStore, name)))
    for name in (
        "effective_session_settings",
        "session_workspace_id",
        "session_ephemeral_endpoint_policy",
        "presentation_context",
        "session_one_shot_prefill",
    )
)
_CAPTURE = _definition(capture_send_selection)
