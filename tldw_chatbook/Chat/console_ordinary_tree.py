"""Private originating-turn custody for ordinary native Console execution."""

from __future__ import annotations

import asyncio
from inspect import getattr_static

from tldw_chatbook.app_services import read_mcp_service
from tldw_chatbook.Agents.activation import (
    callable_source_current as _function_unchanged,  # noqa: F401 - original source check import
    capture_callable_source as _function_record,  # noqa: F401 - original source record import
)
from tldw_chatbook.Agents.execution_capacity import (
    WorkOrigin,
    bind_ordinary_tree as bind_ordinary_tree,
    current_ordinary_tree as current_ordinary_tree,
    set_current_ordinary_tree,
)
from tldw_chatbook.Agents.ordinary_tree import (
    OrdinaryTurn as _OrdinaryTurn,
    capture_ordinary_personal_service as _capture_personal_service,
    encode_ordinary_receipt as _json,  # noqa: F401 - original receipt encoder import
)
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint, WorktreeScope

def _bootstrap_personal_service(app):
    from tldw_chatbook.Personal_Context.bootstrap import (
        bootstrap_personal_context_service,
    )

    service = bootstrap_personal_context_service()
    tree = current_ordinary_tree()
    if tree is not None:
        tree.publish_personal_service(app, service)
    return service


def finish_ordinary_submission(controller, task):
    tree = getattr(controller, "_ordinary_submissions", {}).pop(task, None)
    if tree is not None:
        try:
            tree.finish()
        finally:
            if current_ordinary_tree() is tree:
                set_current_ordinary_tree(None)


async def admit_ordinary_turn(
    controller,
    session,
    configuration,
    origin,
    text,
    candidates,
    *,
    agent_eligible,
    hook_read=None,
):
    """Select original sources before acquiring the complete ordinary footprint."""
    from tldw_chatbook.Agents.activation import worker_guard

    from .console_preparation_reads import current_agent_sources
    from .console_preparation_reads import run_preparation_read
    from tldw_chatbook.Utils.platform_files import supports_native_process_groups

    if controller._hooks_v2_runtime is None or not supports_native_process_groups():
        return None, configuration
    existing = current_ordinary_tree(controller)
    if existing is not None:
        existing.check()
        return existing, configuration
    source = current_agent_sources(controller)
    tree = _OrdinaryTurn(
        controller, session, configuration, origin, source, agent_eligible
    )
    recovery = getattr(controller, "_archive_recovery_runtime", None)
    claim = source.received_claim if source is not None else None
    if (
        agent_eligible
        and claim is not None
        and tree.store.received_turn_is_current(claim)
        and tree.origin is WorkOrigin.MANUAL
        and not session.ephemeral
        and session.runtime_backend == "local"
        and configuration.recovery_user_origin is not None
        and recovery is not None
        and recovery is tree.runtime._archive_recovery
    ):
        from .archive_recovery.runtime import ArchiveRecoveryRuntime

        if type(recovery) is ArchiveRecoveryRuntime:
            turn_id = recovery.accepted_user_turn(
                configuration.recovery_user_origin, session.id
            )
            if turn_id == claim.request_id:
                tree.archive_recovery = (recovery, turn_id)

    from .console_configuration_preparation import (
        original_mcp_capture_matches,
        original_skill_capture_preparation,
    )

    tree.mcp_metadata_witnessed = (
        source is not None
        and source.mcp is None
        and getattr_static(tree.app, "unified_mcp_service", None) is None
    ) or original_mcp_capture_matches(controller, session, configuration)
    pending_mcp = configuration.mcp_capture_pending
    pending_review = configuration.change_review_capture_pending
    pending_skill = configuration.skill_capture_pending
    original_configuration = configuration
    if (
        pending_mcp or pending_review or pending_skill
    ) and not original_mcp_capture_matches(
        controller, session, configuration, pending=pending_mcp
    ):
        raise RuntimeError("Original MCP capture source is unavailable; send again.")
    tree.skill_preparation = (
        original_skill_capture_preparation()
        if original_mcp_capture_matches(
            controller,
            session,
            configuration,
            pending=pending_mcp,
            require_complete=False,
        )
        else None
    )

    def capture():
        tree.check()
        if candidates:
            from .console_reference_preparation import _ReferenceRead

            tree.reference = _ReferenceRead(
                controller, session, origin, text, candidates
            )
            tree.reference.capture_authority()
            tree.reference.ordinary_tree = tree
        footprint = _capture_maximum(tree, hook_read=hook_read)
        tree.check_local_sources()
        tree.admit(footprint)
        return tree

    try:
        await run_preparation_read(
            worker_guard(controller._agent_bridge)(capture),
            creator=controller,
            session_id=session.id,
            reads=controller._preparation_reads,
            observers=(tree.runtime._preparation_reads,),
            require_current=tree.check,
        )
    finally:
        # Cancellation is delivered after the real capture returns. Its genuine
        # admission still belongs to this submission's original outer finally.
        with tree.runtime._execution_capacity_lock:
            admitted = tree in tree.runtime._ordinary_turns
        if admitted:
            if source is not None:
                source.ordinary_tree = tree
            pending = getattr(controller, "_ordinary_submissions", None)
            if pending is None:
                pending = controller._ordinary_submissions = {}
            pending[asyncio.current_task()] = tree
    set_current_ordinary_tree(tree)
    if pending_skill:
        from contextlib import nullcontext
        from dataclasses import replace

        from .console_configuration_preparation import (
            capture_console_skill_catalog_owned,
            finish_console_skill_catalog_owned,
            pending_skill_capture,
        )

        captured = pending_skill_capture(configuration)
        if captured is None:
            raise RuntimeError(
                "Original skill preparation source is unavailable; send again."
            )
        deferred, original_request_configuration = captured
        task = asyncio.current_task()
        records = [
            record
            for record in tree.runtime._turn_custody.values()
            if record.task is task and record.request is not None
        ]
        if (
            len(records) != 1
            or records[0].request.configuration is not original_request_configuration
        ):
            raise RuntimeError(
                "Original skill preparation custody changed; send again."
            )
        record = records[0]

        def current_skill_source():
            tree.check()
            if (
                tree.runtime._turn_custody.get(record.turn_id) is not record
                or record.task is not task
                or record.request is None
                or record.request.configuration is not original_request_configuration
                or not original_mcp_capture_matches(
                    controller,
                    session,
                    original_configuration,
                    pending=pending_mcp,
                    require_complete=False,
                )
            ):
                raise RuntimeError(
                    "Original skill preparation custody changed; send again."
                )

        operation = tree.borrow(tree.runtime).reserve_tool()
        try:
            with (
                tree.source.activate()
                if tree.source is not None
                else bind_ordinary_tree(tree)
            ):
                current_skill_source()
                catalog = await capture_console_skill_catalog_owned(
                    tree.app,
                    tree.store,
                    controller,
                    session_id=session.id,
                    turn_id=record.turn_id,
                    skill_workspace_id=deferred.selection.skill_workspace_id,
                    preparation=deferred.preparation,
                    reads=controller._preparation_reads,
                    observers=(tree.runtime._preparation_reads,),
                    require_current=current_skill_source,
                )
                prepared = await finish_console_skill_catalog_owned(
                    catalog,
                    reads=controller._preparation_reads,
                    observers=(tree.runtime._preparation_reads,),
                    require_current=current_skill_source,
                )
                current_skill_source()
                maximum = dict(prepared.maximum)
                if "plugin_turn_id" in configuration.skill_context_maximum:
                    maximum["plugin_turn_id"] = configuration.skill_context_maximum[
                        "plugin_turn_id"
                    ]
                configuration = replace(
                    configuration,
                    skill_context_maximum=maximum,
                    skill_capture_pending=False,
                )
                tree.configuration = configuration
                if tree.source is not None:
                    tree.source.configuration = configuration
        finally:
            with tree.source.activate() if tree.source is not None else nullcontext():
                operation.finish()
    if pending_mcp:
        from contextlib import nullcontext
        from dataclasses import replace

        from tldw_chatbook.MCP.console_snapshot import (
            _capture_console_definition_selection,
            standard_console_sources,
        )

        from .console_configuration_capture import (
            CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
            _capture_mcp_definition_selection,
        )

        operation = tree.borrow(tree.runtime).reserve_tool()
        try:
            with (
                tree.source.activate()
                if tree.source is not None
                else bind_ordinary_tree(tree)
            ):
                service = read_mcp_service(tree.app)
                if not original_mcp_capture_matches(
                    controller, session, original_configuration, pending=True
                ):
                    raise RuntimeError("Original MCP capture source changed.")
                if standard_console_sources(service):
                    selected = await _capture_console_definition_selection(
                        service,
                        CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
                        profile_id=configuration.tool_policy_profile_id,
                        persona_policy_rules=configuration.persona_policy_rules,
                        _admitted=True,
                    )
                else:
                    selected = _capture_mcp_definition_selection(
                        tree.app,
                        profile_id=configuration.tool_policy_profile_id,
                        persona_policy_rules=configuration.persona_policy_rules,
                        _admitted=True,
                    )
                from .console_configuration_preparation import (
                    consume_native_mcp_permission_publication,
                )

                consume_native_mcp_permission_publication(
                    tree.source if tree.source is not None else tree
                )
                if not original_mcp_capture_matches(
                    controller, session, original_configuration, pending=True
                ):
                    raise RuntimeError("Original MCP capture source changed.")
                tree.check_local_sources()
                configuration = replace(
                    configuration,
                    mcp_tool_maximum=frozenset(selected.definitions),
                    mcp_definition_maximum=selected.definitions,
                    mcp_definition_sources=selected.sources,
                    mcp_definition_complete=selected.complete,
                    mcp_capture_pending=False,
                    mcp_definition_capture="captured",
                )
                tree.configuration = configuration
                if tree.source is not None:
                    tree.source.configuration = configuration
                rows = {row[0]: row for row in selected.sources}
                if selected.complete and set(rows) == set(selected.definitions):
                    tree.mcp_source_maximum = rows
                await run_preparation_read(
                    worker_guard(controller._agent_bridge)(
                        lambda: _capture_maximum(tree)
                    ),
                    creator=controller,
                    session_id=session.id,
                    reads=controller._preparation_reads,
                    observers=(tree.runtime._preparation_reads,),
                    require_current=tree.check,
                )
        finally:
            with tree.source.activate() if tree.source is not None else nullcontext():
                operation.finish()
    from .console_turn_context import capture_change_review_admission

    review = await run_preparation_read(
        worker_guard(controller._agent_bridge)(
            lambda: capture_change_review_admission(tree.app, tree.workspace)
        ),
        creator=controller,
        session_id=session.id,
        reads=controller._preparation_reads,
        observers=(tree.runtime._preparation_reads,),
        require_current=tree.check,
    )
    if pending_review:
        from dataclasses import replace

        if not tree.footprint.opaque or not original_mcp_capture_matches(
            controller, session, original_configuration, pending=pending_mcp
        ):
            raise RuntimeError("Original Change Review capture source changed.")
        tree.check_local_sources()
        roots, aliases, skipped = review
        configuration = replace(
            configuration,
            workspace_roots=roots,
            change_review_root_aliases=aliases,
            change_review_skipped_roots=skipped,
            change_review_capture_pending=False,
        )
        tree.configuration = configuration
        if tree.source is not None:
            tree.source.configuration = configuration
    return tree, configuration


async def run_ordinary_preparation(controller, callback, *, ssh=None, opaque=False):
    """Retain one actual original preparation worker before its existing enqueue."""
    from threading import Event

    from tldw_chatbook.Agents.activation import worker_guard

    from .console_preparation_reads import run_preparation_read

    tree = current_ordinary_tree(controller)
    if tree is None:
        return await asyncio.to_thread(callback)
    tree.check_local_sources()
    operation = tree.borrow(tree.runtime, opaque=opaque).reserve_tool()
    entered = Event()

    def owned():
        entered.set()
        with tree.source.activate():
            if ssh is not None:
                with ssh.activate(operation):
                    return callback()
            try:
                return callback()
            finally:
                operation.finish()

    try:
        return await run_preparation_read(
            worker_guard(controller._agent_bridge)(owned),
            creator=controller,
            session_id=tree.session.id,
            reads=controller._preparation_reads,
            observers=(tree.runtime._preparation_reads,),
            require_current=tree.check,
        )
    finally:
        if not entered.is_set():
            with tree.source.activate():
                operation.finish()


def _permission_payload(tree):
    """Read only witnessed native metadata before admission; unknown waits."""
    permission = tree.source.permission if tree.source is not None else None
    if permission is None or tree.owner is None and not tree.mcp_metadata_witnessed:
        return None
    return permission.load()


def _capture_maximum(tree, *, hook_read=None):
    """Project stock declared effects; live composition may only narrow them."""
    from tldw_chatbook.Agents.persona_policy import (
        evaluate_tool_policy,
        parse_persona_policy_from_rules,
    )
    from tldw_chatbook.Workspaces.change_review_consent import ChangeReviewState

    configuration = tree.configuration
    policy = parse_persona_policy_from_rules(configuration.persona_policy_rules)
    advertised = lambda name: (
        evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name=name).advertised
    )
    reads, writes = [], []
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work

    automatic = current_automatic_work()
    goal = automatic.goal if automatic is not None else None
    goal_scope = goal.goal.request.tool_scope if goal is not None else None
    catalog_allowed = lambda source, name: (
        goal_scope is None or f"{source}:{name}" in goal_scope.catalog_tools
    )
    runtime_allowed = lambda name: goal is None or goal.permits_runtime(name)
    tree.catalog_permits = lambda source, name: (
        advertised(name) and catalog_allowed(source, name)
    )
    tree.goal_catalog_permits = catalog_allowed
    sources = {row[0]: row for row in configuration.mcp_definition_sources}
    if tree.mcp_metadata_witnessed and set(sources) == set(
        configuration.mcp_definition_maximum
    ):
        tree.mcp_source_maximum = sources
    opaque = bool(
        configuration.mcp_capture_pending
        or configuration.skill_capture_pending
        or (
            tree.agent_eligible
            and tree.source is not None
            and tree.source.permission is not None
            and not tree.mcp_metadata_witnessed
        )
        or (
            tree.agent_eligible
            and not tree.session.ephemeral
            and (
                (
                    goal_scope is None
                    or any(name.startswith("mcp:") for name in goal_scope.catalog_tools)
                )
                if tree.mcp_source_maximum is None
                else any(
                    tree.catalog_permits("mcp", row[3]) for row in sources.values()
                )
            )
        )
    )
    scratch = configuration.scratch_space
    if scratch is not None:
        scope = WorktreeScope(scratch.root, True, capture_directory_chain(scratch.root))
        reads.append(scope)
        writes.append(scope)
    if tree.source is not None and tree.source.root is not None:
        writes.append(WorktreeScope(tree.source.root, True, tree.source.root_identity))
    if tree.reference is not None:
        opaque |= tree.reference.opaque
        reads.extend(
            WorktreeScope(chain.canonical_root, True, chain)
            for chain in tree.reference.authority.chains
        )
    service = getattr_static(tree.app, "change_review_consent_service", None)
    tree.local_source_values += (
        (tree.app, "change_review_consent_service", service, None),
    )
    if service is not None:
        from tldw_chatbook.Workspaces.change_review_consent import (
            ChangeReviewConsentService,
        )

        if (
            type(service) is ChangeReviewConsentService
            and not service._custom_capability_reader
        ):
            status = service.status(tree.workspace)
            opaque |= (
                status.capability.state is ChangeReviewState.ENABLED
                and status.consent.state is ChangeReviewState.ENABLED
            )
        else:
            opaque = True
    engine = tree.runtime.get_hooks_v2(tree.session.id)
    opaque |= bool(engine is not None and engine.enabled and engine.definitions)
    permissions = tree.runtime._hook_permissions
    if permissions is not None:
        from tldw_chatbook.Agents.run_hooks import load_hooks_config

        from .console_hook_preparation import attempt_v2_configuration_for

        # This selects only a footprint. Hook preparation/launch still owns
        # its own checks, and a later recapture does not carry a Send read.
        shared = attempt_v2_configuration_for(permissions, tree.session.id, hook_read)
        review, _targets = (
            shared if shared is not None else permissions.v2_configuration()
        )
        hooks = load_hooks_config(
            {"hooks": review.config.section} if review.config.section_present else {}
        )
        opaque |= bool(hooks.enabled and (hooks.hooks or hooks.v2_handlers))
    opaque |= any(
        item.get("plugin_owned")
        for item in configuration.skill_context_maximum.get("available_skills", ())
    )
    if tree.agent_eligible:
        opaque |= _capture_personal_source(tree)
        opaque |= _capture_other_providers(tree)
        budget = configuration.tool_configuration.get("agent_run_budget_maximum")
        opaque |= bool(
            budget is not None
            and budget.max_subagents
            and advertised("spawn_subagent")
            and runtime_allowed("spawn_subagent")
        )
        bridge = tree.controller._agent_bridge
        if bridge._skills_service is not None:
            import os

            opaque |= bool(os.name != "nt" and runtime_allowed("run_skill_script"))
            opaque |= bool(runtime_allowed("install_skill"))
        if configuration.tool_configuration.get("local_tools_enabled"):
            from tldw_chatbook.Agents import local_tool_provider as local
            from tldw_chatbook.MCP.permission_store import (
                resolve_effective_state_by_key,
            )

            mcp = read_mcp_service(tree.app)
            if mcp is not None and not configuration.mcp_capture_pending:
                payload = _permission_payload(tree)
                if payload is None or not payload.get("kill_switch", False):
                    tree.local_maximum = {
                        name: effect
                        for name, effect in local.native_worktree_effects().items()
                        if advertised(name)
                        and catalog_allowed("local", name)
                        and (
                            name == "ask_user"
                            or payload is None
                            or resolve_effective_state_by_key(
                                payload,
                                local.LOCAL_SERVER_KEY,
                                name,
                                profile_id=configuration.tool_policy_profile_id,
                            ).state
                            != "deny"
                        )
                    }
                    opaque |= "opaque" in tree.local_maximum.values()
                    raw = getattr_static(tree.app, "raw_cli_runtime", None)
                    if (
                        raw is not None
                        and advertised("shell_exec")
                        and catalog_allowed("raw_shell", "shell_exec")
                    ):
                        from tldw_chatbook.Tools.raw_cli_executor import ordinary_raw_shell_metadata

                        original, raw_reachable, raw_values = ordinary_raw_shell_metadata(tree)
                        tree.raw_projection = (original, raw_reachable, raw_values)
                        opaque |= raw_values[:1] == ("unknown",)
                        if raw_reachable and (
                            payload is None
                            or resolve_effective_state_by_key(
                                payload,
                                local.LOCAL_SERVER_KEY,
                                "shell_exec",
                                profile_id=configuration.tool_policy_profile_id,
                            ).state
                            != "deny"
                        ):
                            opaque = True
    if tree.agent_eligible:
        opaque |= _capture_library_maximum(tree)
        builtin, builtin_unknown = _capture_builtin_maximum(
            tree, advertised, catalog_allowed
        )
        opaque |= builtin_unknown
        # Root enumeration is a preparation producer even without local tools.
        # Goal selects its finite bindings before any remote recovery scheduling.
        authority = configuration.project_authority
        goal_bindings = (
            None
            if goal is None
            else {
                reference.binding_id: (index == 0 and reference.access == "rw")
                for index, reference in enumerate(
                    (goal.goal.request.binding, *goal.goal.request.source_bindings)
                )
            }
        )
        writable = "write" in tree.local_maximum.values() or "write" in builtin.values()
        for binding in authority.options if authority is not None else ():
            if goal_bindings is not None and binding.binding_id not in goal_bindings:
                continue
            from tldw_chatbook.Tools.remote_root_types import is_remote

            from .console_chat_controller import _project_selection_from_snapshot

            root = _project_selection_from_snapshot(binding).root
            if is_remote(root):
                opaque = True
                continue
            chain = capture_directory_chain(root)
            scope = WorktreeScope(chain.canonical_root, True, chain)
            if scope not in reads:
                reads.append(scope)
            if (
                writable
                and binding.allow_write
                and (goal_bindings is None or goal_bindings[binding.binding_id])
            ):
                writes.append(scope)
    return WorktreeFootprint(tuple(reads), tuple(writes), opaque)


def _capture_builtin_maximum(tree, advertised, catalog_allowed):
    """Project declared effects and retain the actual tool classes for composition."""
    import importlib

    from tldw_chatbook import config
    from tldw_chatbook.Agents import tool_catalog as catalog
    from tldw_chatbook.MCP.permission_store import (
        BUILTIN_TOOL_SERVER_KEY,
        resolve_effective_state_by_key,
    )

    effects = {}
    for name, cls in zip(
        catalog.ALWAYS_ON_BUILTIN_NAMES,
        (catalog.CalculatorTool, catalog.DateTimeTool),
        strict=True,
    ):
        if advertised(name) and catalog_allowed("builtin", name):
            effects[name] = "none"
            tree.builtin_classes[name] = cls
    payload = _permission_payload(tree)
    for entry in catalog.gateable_builtin_tools():
        if not config.coerce_bool_setting(
            config.get_cli_setting("tools", entry.gate_key, False), False
        ):
            continue
        if (
            not advertised(entry.tool_name)
            or not catalog_allowed("builtin", entry.tool_name)
            or payload is not None
            and (
                payload.get("kill_switch", False)
                or resolve_effective_state_by_key(
                    payload,
                    BUILTIN_TOOL_SERVER_KEY,
                    entry.tool_name,
                    profile_id=tree.configuration.tool_policy_profile_id,
                ).state
                == "deny"
            )
            or tree.session.ephemeral
            and entry.tool_name in {"create_note", "update_note", "write_file"}
        ):
            continue
        module = importlib.import_module("tldw_chatbook.Tools." + entry.module_name)
        tree.builtin_classes[entry.tool_name] = getattr(module, entry.factory_name)
        effects[entry.tool_name] = entry.worktree_effect
    tree.builtin_maximum = effects
    return effects, "opaque" in effects.values()


def _capture_personal_source(tree):
    tree.personal_getter = getattr_static(
        tree.app, "get_personal_context_service", None
    )
    tree.personal_service = getattr_static(tree.app, "_personal_context_service", None)
    if tree.personal_getter is None:
        return False
    tree.personal_native = _capture_personal_service(tree, tree.personal_service)
    return not tree.personal_native


def _capture_library_maximum(tree):
    """Read original Library declarations and selected backends, never compose."""
    import sys

    from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS

    from .console_library_policy import ConsoleAssistantLibraryAccess
    from .console_runtime import _NativeLibraryProviderFactory

    factory = getattr_static(tree.controller, "_library_provider_factory", None)
    policy = tree.configuration.library_policy_maximum
    if (
        factory is None
        or policy is None
        or policy.assistant_access is not ConsoleAssistantLibraryAccess.ALLOWED
    ):
        return False
    direct = tree.configuration.tool_configuration.get("direct_library_tools", True)
    names = tuple(LIBRARY_TOOL_DESCRIPTORS) if direct else ("search_rag",)
    tree.library_maximum = frozenset(
        name for name in names if tree.catalog_permits("library", name)
    )
    if not (
        isinstance(factory, _NativeLibraryProviderFactory)
        and tree.runtime._library_provider_factory is factory
        and factory.runtime is tree.runtime
        and tree.runtime._app is tree.app
    ):
        return True
    tree.library_factory = factory
    # Native RAG can initialize the selected shared backend; it has no closed
    # original artifact/index destination contract at this boundary.
    if not direct:
        service = getattr_static(tree.app, "library_rag_search_service", None)
        tree.local_source_values += (
            (tree.app, "library_rag_search_service", service, None),
        )
        return service is not None
    values = list(tree.local_source_values)
    opaque = False
    services = (
        (
            "_media",
            "local_media_reading_service",
            "tldw_chatbook.Media.local_media_reading_service",
            "media",
        ),
        ("_notes", "notes_service", "tldw_chatbook.Notes.Notes_Library", "note"),
        (
            "_prompts",
            "local_prompt_service",
            "tldw_chatbook.Prompt_Management.local_prompt_service",
            "prompt",
        ),
        ("_skills", "local_skills_service", None, "skill"),
        (
            "_conversations",
            "local_chat_conversation_service",
            "tldw_chatbook.Chat.chat_conversation_service",
            "conversation",
        ),
    )
    # Only already loaded native source modules qualify. Classification never
    # imports a caller-selected backend or executes a service constructor.
    source_modules = [
        (sys.modules.get("tldw_chatbook.Agents.library_tool_provider"), None),
        (sys.modules.get("tldw_chatbook.Library.local_library_tool_service"), None),
    ]
    for service_field, attribute, module_name, item_type in services:
        service = getattr_static(tree.app, attribute, None)
        tree.library_services[service_field] = service
        values.append((tree.app, attribute, service, None))
        reachable = [
            LIBRARY_TOOL_DESCRIPTORS[name]
            for name in tree.library_maximum
            if LIBRARY_TOOL_DESCRIPTORS[name].item_type == item_type
        ]
        if service is None or not reachable:
            continue
        if any(descriptor.operation in {"save", "rechunk"} for descriptor in reachable):
            opaque = True
        module = sys.modules.get(module_name) if module_name is not None else None
        if module is None:
            opaque = True
            continue
        source_modules.append((module, service))
        # These Library methods use the concrete native DB interfaces. The
        # prompt facade's injected module and skill facades are not such proof.
        if service_field == "_prompts":
            opaque = True
        elif service_field in {"_media", "_notes", "_conversations"}:
            db_field = {
                "_media": "media_db",
                "_notes": "unified_db_template",
                "_conversations": "db",
            }[service_field]
            database = getattr_static(service, db_field, None)
            values.append((service, db_field, database, None))
            if database is not None:
                for path in ("db_path", "db_path_str"):
                    value = getattr_static(database, path, None)
                    values.append((database, path, value, None))
    media = tree.library_services["_media"]
    media_db = getattr_static(tree.app, "media_db", None)
    values.append((tree.app, "media_db", media_db, None))
    if media is not None or media_db is not None:
        if any(
            LIBRARY_TOOL_DESCRIPTORS[name].operation == "rechunk"
            for name in tree.library_maximum
        ):
            opaque = True
        else:
            # Its injected chunk/template/policy interfaces are only known at
            # actual composition; retain OPAQUE until their original source is closed.
            opaque |= any(
                LIBRARY_TOOL_DESCRIPTORS[name].operation
                in {"structure", "chunk", "spec_list", "spec_save"}
                for name in tree.library_maximum
            )
    policy_source = getattr_static(tree.app, "service_policy_enforcer", None)
    values.append((tree.app, "service_policy_enforcer", policy_source, None))
    for module, instance in source_modules:
        if module is None or not hasattr(module, "_NATIVE_ORDINARY_LIBRARY_CLASS"):
            opaque = True
            continue
        cls = module._NATIVE_ORDINARY_LIBRARY_CLASS
        if instance is not None and type(instance) is not cls:
            opaque = True
        values.append(
            (module, cls.__name__, getattr_static(module, cls.__name__), None)
        )
        opaque |= getattr_static(module, cls.__name__) is not cls
    tree.local_source_values = tuple(values)
    return opaque


def _capture_other_providers(tree):
    """Project actual configured provider capabilities without executing factories."""
    from tldw_chatbook import config
    from tldw_chatbook.Agents import tool_catalog

    opaque = False
    if tree.archive_recovery is not None:
        tree.other_maximum["archive_recovery"] = frozenset(
            name
            for name in tool_catalog.RECOVERY_RESERVED_TOOL_NAMES
            if tree.catalog_permits("archive_recovery", name)
        )
    canvas = getattr_static(tree.store, "canvas_turn_controller", None)
    tree.local_source_values += ((tree.store, "canvas_turn_controller", canvas, None),)
    if canvas is not None and config.get_canvas_execution_enabled():
        tree.other_maximum["canvas"] = frozenset(
            name
            for name in tool_catalog.CANVAS_RESERVED_TOOL_NAMES
            if tree.catalog_permits("canvas", name)
        )
        opaque = True
    if tree.personal_getter is not None and not tree.session.ephemeral:
        tree.other_maximum["personal-context"] = frozenset(
            name
            for name in tool_catalog.PROFILE_RESERVED_TOOL_NAMES
            if tree.catalog_permits("personal-context", name)
        )
    if (
        tree.configuration.tool_configuration.get("local_tools_enabled")
        and not tree.configuration.mcp_capture_pending
    ):
        payload = _permission_payload(tree)
        if payload is None or not payload.get("kill_switch", False):
            authority = tree.configuration.project_authority
            if (
                authority is not None
                and authority.options
                and tree.catalog_permits("virtual_cli", "virtual_cli")
            ):
                from tldw_chatbook.Agents.virtual_cli_provider import (
                    VIRTUAL_CLI_COMMANDS,
                    VIRTUAL_CLI_SERVER_KEY,
                )
                from tldw_chatbook.MCP.permission_store import (
                    resolve_effective_state_by_key,
                )

                tree.other_maximum["virtual_cli"] = frozenset({"virtual_cli"})
                tree.virtual_commands = frozenset(
                    command
                    for command in VIRTUAL_CLI_COMMANDS
                    if payload is None
                    or resolve_effective_state_by_key(
                        payload,
                        VIRTUAL_CLI_SERVER_KEY,
                        command,
                        profile_id=tree.configuration.tool_policy_profile_id,
                    ).state
                    != "deny"
                )
            if tree.catalog_permits("raw_shell", "shell_exec"):
                tree.other_maximum["raw_shell"] = frozenset({"shell_exec"})
    return opaque
