"""Executable inventory for every public fork-snapshot transition owner."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


STORE_PATH = Path("tldw_chatbook/Chat/console_chat_store.py")
IMAGE_RECOVERY_PATH = Path("tldw_chatbook/UI/Console_Modules/image.py")
RETRIEVAL_PATH = Path("tldw_chatbook/UI/Console_Modules/retrieval.py")
HYDRATION_PATH = Path("tldw_chatbook/Chat/console_conversation_hydration.py")
CONTROLLER_PATH = Path("tldw_chatbook/Chat/console_chat_controller.py")


# Bidirectional contract: every direct owner appears here and every entry must
# visibly enter one of the two canonical transition contexts. Delegated routes
# are listed separately so a future rename cannot silently remove the owner.
DIRECT_TRANSITION_ROUTES = frozenset(
    {
        "add_variant",
        "accept_roleplay_projection_persistence_result",
        "adopt_session_ephemeral_endpoint",
        "append_generation_message",
        "append_generation_variant",
        "append_message",
        "append_video_message",
        "begin_generation_attempt",
        "begin_variant_stream",
        "cancel_preparation",
        "close_session",
        "commit_console_settings_live",
        "create_sibling",
        "delete_message",
        "discard_provider_continuation",
        "finalize_variant_stream",
        "fork_source_transition",
        "hydrate_generation_metadata",
        "keep_generation_variant",
        "mark_message_complete",
        "mark_message_failed",
        "mark_message_send_blocked",
        "mark_message_stopped",
        "merge_persisted_generation_message",
        "merge_persisted_system_message",
        "persist_provider_continuation_event",
        "persist_message_if_needed",
        "persist_selected_generation",
        "persist_session_if_needed",
        "prepare_dispatch_recovery_message",
        "prepare_message_retry",
        "prepare_session_roleplay_projection_refresh",
        "promote_ephemeral_session",
        "publish_committed_identity",
        "publish_durable_recovery_owner",
        "publish_durable_turn_identity",
        "publish_durable_turn_owners",
        "refresh_session_roleplay_projections",
        "rename_session",
        "release_dispatch_recovery_action",
        "reload_quarantined_generation",
        "rollback_transient_send",
        "rollback_session_ephemeral_endpoint_adoption",
        "rollback_session_settings_replacement",
        "replace_message_thinking",
        "replace_session_settings",
        "retire_generation_attempt",
        "save_session_library_policy",
        "seed_character_roleplay",
        "select_variant",
        "set_active_leaf",
        "set_message_feedback",
        "set_message_metadata",
        "set_message_usage",
        "set_session_character_name",
        "set_session_context_policy_overrides",
        "set_session_pinned_prefill",
        "set_session_project_instruction_state",
        "set_session_rag_scope",
        "set_session_system_prompt",
        "set_session_thinking_history_policy",
        "set_session_user_display_name_override",
        "settle_dispatch_recovery",
        "settle_message_thinking",
        "stage_session_library_policy",
        "swap_session_character_roleplay",
        "transition_dispatch_recovery_for_retry",
        "update_message_content",
    }
)

DELEGATED_TRANSITION_ROUTES = {
    "confirm_auto_speak_destination": "_set_speech_preferences",
    "pause_auto_speak": "_set_speech_preferences",
    "resume_auto_speak": "_set_speech_preferences",
    "set_auto_speak": "_set_speech_preferences",
}

# These public methods assign a fork field but cannot race an eligible fork:
# pristine/restore methods operate before a forkable source exists; read-only
# projection writes only detached snapshots; streaming/deferred routes retain
# an ineligible pending/blank owner until a separately fenced terminal route.
SAFE_NON_TRANSITION_ASSIGNMENTS = frozenset(
    {
        "append_stream_chunk",
        "finalize_deferred_user_message_content",
        "read_only_messages_for_session",
        "replace_deferred_terminal_body",
        "repurpose_pristine_session",
        "reset_stream_content",
        "restore_persisted_session",
        "rollback_pristine_session_refresh",
        "refresh_pristine_session_settings",
    }
)

FORK_FIELD_ASSIGNMENTS = frozenset(
    {
        "assistant_authority_id",
        "assistant_id",
        "assistant_kind",
        "attachments",
        "character_id",
        "character_name",
        "character_system_template",
        "content",
        "context_policy_overrides",
        "ephemeral",
        "generation_metadata",
        "generation_projection_quarantined",
        "parent_message_id",
        "persisted_conversation_id",
        "persisted_message_id",
        "persona_memory_mode",
        "project_instruction_state",
        "rag_scope",
        "role",
        "runtime_backend",
        "settings",
        "speech_preferences",
        "status",
        "thinking_history_policy",
        "title",
        "user_display_name_override",
        "variants",
        "video_metadata",
        "workspace_id",
    }
)


def _store_methods() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(STORE_PATH.read_text(encoding="utf-8"))
    store = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ConsoleChatStore"
    )
    return {
        node.name: node
        for node in store.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_attributes(node: ast.AST) -> set[str]:
    return {
        child.func.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }


_TRANSITION_DECORATORS = frozenset(
    {"_fork_continuation_event_transition", "_fork_session_transition"}
)
_TRANSITION_CONTEXTS = frozenset({"_fork_source_transition", "_generation_owner_scope"})
_DETACHED_MUTATION_METHODS = frozenset(
    {
        "_canonical_generation_reload_candidate",
        "_generation_owner_candidate",
        "_hydrate_provider_continuations_from_persistence",
        "_snapshot",
        "_stage_fork_snapshot",
        "persist_roleplay_projection_plan",
    }
)
_NON_FORKABLE_MUTATION_METHODS = frozenset(
    {"_fold_stream_buffer_without_persistence", "_materialize_stream_buffer"}
)


def _expr_text(node: ast.expr | None) -> str | None:
    return ast.unparse(node) if node is not None else None


def _owner_text(node: ast.expr | None, bindings: dict[str, str]) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name) and node.id in bindings:
        return bindings[node.id]
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "id"
        and isinstance(node.value, ast.Name)
        and node.value.id in bindings
    ):
        return bindings[node.value.id]
    return _expr_text(node)


def _store_owned_callable(node: ast.expr, names: frozenset[str]) -> str | None:
    """Return a canonical boundary name without trusting arbitrary qualifiers."""

    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name) and target.id in names:
        return target.id
    if (
        isinstance(target, ast.Attribute)
        and target.attr in names
        and isinstance(target.value, ast.Name)
        and target.value.id in {"self", "console_chat_store"}
    ):
        return target.attr
    return None


def _executed_nodes(node: ast.AST) -> tuple[ast.AST, ...]:
    """Walk code executed by ``node`` without executing deferred closures."""

    found: list[ast.AST] = []

    def visit(current: ast.AST, *, root: bool = False) -> None:
        if not root and isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ):
            return
        found.append(current)
        for child in ast.iter_child_nodes(current):
            visit(child)

    visit(node, root=True)
    return tuple(found)


def _owner_bindings(node: ast.AST) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for child in _executed_nodes(node):
        if not isinstance(child, (ast.Assign, ast.AnnAssign)):
            continue
        targets = child.targets if isinstance(child, ast.Assign) else [child.target]
        value = child.value
        if not isinstance(value, ast.Call):
            continue
        if (
            isinstance(value.func, ast.Name)
            and value.func.id in {"deepcopy", "replace"}
            and value.args
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id in bindings
        ):
            owner = bindings[value.args[0].id]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = owner
            continue
        if not isinstance(value.func, ast.Attribute):
            continue
        if value.func.attr not in {"_session_or_raise", "_message_or_raise", "get"}:
            continue
        if not value.args:
            continue
        owner = _expr_text(value.args[0])
        if owner is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = owner
    return bindings


def _root_name(node: ast.expr) -> str | None:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _mutation_events(node: ast.AST) -> tuple[tuple[int, str | None], ...]:
    bindings = _owner_bindings(node)
    parameter_names = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parameter_names = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args)}
    default_owner = "session_id" if "session_id" in parameter_names else None
    events: list[tuple[int, str | None]] = []
    for child in _executed_nodes(node):
        targets: list[ast.expr] = []
        field: str | None = None
        if isinstance(child, ast.Assign):
            targets = child.targets
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
            targets = [child.target]
        elif (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "setattr"
            and len(child.args) >= 2
            and isinstance(child.args[1], ast.Constant)
            and isinstance(child.args[1].value, str)
        ):
            targets = [child.args[0]]
            field = child.args[1].value
        elif (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "set"
            and isinstance(child.func.value, ast.Attribute)
            and child.func.value.attr == "rag_scope_holder"
        ):
            targets = [child.func.value]
            field = "rag_scope"
        for target in targets:
            attrs = {
                nested.attr
                for nested in ast.walk(target)
                if isinstance(nested, ast.Attribute)
            }
            if field is not None:
                attrs.add(field)
            if not attrs & FORK_FIELD_ASSIGNMENTS:
                continue
            root = _root_name(target)
            owner = bindings.get(root or "")
            if owner is None and root == "self":
                owner = default_owner
            events.append((child.lineno, owner))
            break
    return tuple(events)


def _transition_ranges(node: ast.AST) -> tuple[tuple[int, int, str | None], ...]:
    ranges: list[tuple[int, int, str | None]] = []
    bindings = _owner_bindings(node)
    for child in _executed_nodes(node):
        if not isinstance(child, (ast.With, ast.AsyncWith)):
            continue
        for item in child.items:
            if _store_owned_callable(item.context_expr, _TRANSITION_CONTEXTS) is None:
                continue
            context = item.context_expr
            assert isinstance(context, ast.Call)
            owner = _owner_text(context.args[0], bindings) if context.args else None
            ranges.append((child.lineno, child.end_lineno or child.lineno, owner))
    return tuple(ranges)


def _decorated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _store_owned_callable(decorator, _TRANSITION_DECORATORS) is not None
        for decorator in node.decorator_list
    )


def _self_call_records(node: ast.AST) -> tuple[ast.Call, ...]:
    return tuple(
        child
        for child in _executed_nodes(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and isinstance(child.func.value, ast.Name)
        and child.func.value.id in {"self", "cls"}
    )


def _call_owner(
    call: ast.Call,
    callee: ast.AST | None = None,
    *,
    bindings: dict[str, str] | None = None,
) -> str | None:
    bindings = bindings or {}
    if callee is not None and isinstance(
        callee, (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        parameters = [*callee.args.posonlyargs, *callee.args.args][1:]
        direct_owners = {owner for _line, owner in _mutation_events(callee) if owner}
        preferred_owner = next(iter(direct_owners)) if len(direct_owners) == 1 else None
        if preferred_owner is not None:
            for keyword in call.keywords:
                if keyword.arg == preferred_owner:
                    return _owner_text(keyword.value, bindings)
        owner_index = next(
            (
                index
                for index, parameter in enumerate(parameters)
                if parameter.arg == preferred_owner
                or (
                    preferred_owner is None
                    and parameter.arg in {"session_id", "message_id"}
                )
            ),
            None,
        )
        if owner_index is not None:
            parameter = parameters[owner_index].arg
            for keyword in call.keywords:
                if keyword.arg == parameter:
                    return _owner_text(keyword.value, bindings)
            if owner_index < len(call.args):
                return _owner_text(call.args[owner_index], bindings)
    return _owner_text(call.args[0], bindings) if call.args else None


def _covered(
    line: int,
    owner: str | None,
    ranges: tuple[tuple[int, int, str | None], ...],
) -> bool:
    return any(
        start <= line <= end and owner is not None and boundary_owner == owner
        for start, end, boundary_owner in ranges
    )


def _roleplay_lease_transitioned(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Recognize the one deliberately detached, token-owned transition lease."""

    if node.name != "prepare_session_roleplay_projection_refresh":
        return False
    calls = _self_call_records(node)

    def is_call(statement: ast.stmt, name: str) -> bool:
        return (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and _store_owned_callable(statement.value, frozenset({name})) == name
        )

    def is_exact_release(statement: ast.stmt) -> bool:
        if not is_call(statement, "_release_roleplay_fork_transition"):
            return False
        assert isinstance(statement, ast.Expr)
        call = statement.value
        assert isinstance(call, ast.Call)
        return (
            bool(call.args)
            and _expr_text(call.args[0]) == "transition_token"
            and any(
                keyword.arg == "expected_session_id"
                and _expr_text(keyword.value) == "session_id"
                for keyword in call.keywords
            )
        )

    def is_token_return(statement: ast.stmt) -> bool:
        return (
            isinstance(statement, ast.Return)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "replace"
            and any(
                keyword.arg == "fork_transition_token"
                and _expr_text(keyword.value) == "transition_token"
                for keyword in statement.value.keywords
            )
        )

    direct_begin = [
        statement
        for statement in node.body
        if is_call(statement, "_begin_fork_source_transition")
    ]
    direct_lease = [
        statement
        for statement in node.body
        if isinstance(statement, (ast.With, ast.AsyncWith))
        and any(
            isinstance(child, ast.Assign)
            and _expr_text(child.value) == "session_id"
            and any(
                isinstance(target, ast.Subscript)
                and _expr_text(target.value) == "self._roleplay_fork_transition_leases"
                and _expr_text(target.slice) == "transition_token"
                for target in child.targets
            )
            for child in statement.body
        )
    ]
    direct_try = [
        statement for statement in node.body if isinstance(statement, ast.Try)
    ]
    begins = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "_begin_fork_source_transition"
        and _call_owner(call) == "session_id"
    ]
    releases = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "_release_roleplay_fork_transition"
        and call.args
        and _expr_text(call.args[0]) == "transition_token"
        and any(
            keyword.arg == "expected_session_id"
            and _expr_text(keyword.value) == "session_id"
            for keyword in call.keywords
        )
    ]
    lease_assignments = [
        child
        for child in _executed_nodes(node)
        if isinstance(child, ast.Assign)
        and _expr_text(child.value) == "session_id"
        and any(
            isinstance(target, ast.Subscript)
            and _expr_text(target.value) == "self._roleplay_fork_transition_leases"
            and _expr_text(target.slice) == "transition_token"
            for target in child.targets
        )
    ]
    materializations = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "_materialize_roleplay_projections_live"
        and _call_owner(call) == "session_id"
    ]
    returned_tokens = [
        child
        for child in _executed_nodes(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "replace"
        and any(
            keyword.arg == "fork_transition_token"
            and _expr_text(keyword.value) == "transition_token"
            for keyword in child.keywords
        )
    ]
    if (
        len(direct_begin) != 1
        or len(direct_lease) != 1
        or len(direct_try) != 1
        or len(begins) != 1
        or len(lease_assignments) != 1
        or not releases
        or not materializations
        or not returned_tokens
    ):
        return False
    lifecycle_try = direct_try[0]

    success_returns = 0

    def release_dominates_exit(block: list[ast.stmt]) -> bool:
        nonlocal success_returns
        for index, statement in enumerate(block):
            if is_token_return(statement):
                success_returns += 1
            elif isinstance(statement, ast.Return):
                if index == 0 or not is_exact_release(block[index - 1]):
                    return False
            elif isinstance(statement, (ast.Raise, ast.Break, ast.Continue)):
                return False
            if is_exact_release(statement):
                if index + 1 >= len(block) or not isinstance(
                    block[index + 1], (ast.Return, ast.Raise)
                ):
                    return False
            nested_blocks: list[list[ast.stmt]] = []
            if isinstance(statement, ast.If):
                nested_blocks.extend((statement.body, statement.orelse))
            elif isinstance(
                statement,
                (ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While),
            ):
                nested_blocks.extend((statement.body, statement.orelse))
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                nested_blocks.extend(
                    (statement.body, statement.orelse, statement.finalbody)
                )
                nested_blocks.extend(handler.body for handler in statement.handlers)
            elif isinstance(statement, ast.Match):
                nested_blocks.extend(case.body for case in statement.cases)
            if any(not release_dominates_exit(nested) for nested in nested_blocks):
                return False
        return True

    if not release_dominates_exit(lifecycle_try.body):
        return False
    if success_returns != 1:
        return False
    if not lifecycle_try.handlers or any(
        not handler.body
        or not is_exact_release(handler.body[0])
        or not isinstance(handler.body[-1], ast.Raise)
        for handler in lifecycle_try.handlers
    ):
        return False
    begin_line = begins[0].lineno
    return all(call.lineno > begin_line for call in materializations)


def _transitioned(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
) -> bool:
    if _decorated(node):
        return True
    if _roleplay_lease_transitioned(node):
        return True
    effects = list(_mutation_events(node))
    ranges = _transition_ranges(node)

    nested = {
        child.name: child
        for child in ast.walk(node)
        if child is not node
        and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for child in _executed_nodes(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            closure = nested.get(child.func.id)
            if closure is not None and _mutation_events(closure):
                closure_owner = _mutation_events(closure)[0][1]
                if closure_owner is None:
                    closure_owner = "session_id"
                effects.append((child.lineno, closure_owner))

    if methods is not None:
        mutating = _fork_mutating_routes(methods)
        bindings = _owner_bindings(node)
        for call in _self_call_records(node):
            assert isinstance(call.func, ast.Attribute)
            callee_name = call.func.attr
            if callee_name not in mutating:
                continue
            callee = methods.get(callee_name)
            if callee is not None and _transitioned(callee, methods=None):
                continue
            effects.append((call.lineno, _call_owner(call, callee, bindings=bindings)))

    return bool(ranges) and all(
        _covered(
            line,
            owner
            or (
                "message_id"
                if any(
                    boundary_owner == "message_id" for _, _, boundary_owner in ranges
                )
                else "session_id"
                if any(
                    boundary_owner == "session_id" for _, _, boundary_owner in ranges
                )
                else None
            ),
            ranges,
        )
        for line, owner in effects
    )


def _has_transition_boundary(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return bool(
        _decorated(node)
        or _transition_ranges(node)
        or _roleplay_lease_transitioned(node)
    )


def _assigned_attributes(node: ast.AST) -> set[str]:
    assigned: set[str] = set()
    for child in ast.walk(node):
        targets: list[ast.expr] = []
        if isinstance(child, ast.Assign):
            targets = child.targets
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
            targets = [child.target]
        for target in targets:
            assigned.update(
                nested.attr
                for nested in ast.walk(target)
                if isinstance(nested, ast.Attribute)
            )
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "setattr"
            and len(child.args) >= 2
            and isinstance(child.args[1], ast.Constant)
            and isinstance(child.args[1].value, str)
        ):
            assigned.add(child.args[1].value)
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "set"
            and isinstance(child.func.value, ast.Attribute)
            and child.func.value.attr == "rag_scope_holder"
        ):
            assigned.add("rag_scope")
    return assigned


def _fork_mutation_lines(node: ast.AST) -> tuple[int, ...]:
    return tuple(sorted({line for line, _owner in _mutation_events(node)}))


def _self_calls(node: ast.AST) -> set[str]:
    return {call.func.attr for call in _self_call_records(node)}


def _fork_mutating_routes(
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> set[str]:
    memo: dict[str, bool] = {}

    def mutates(name: str, active: frozenset[str] = frozenset()) -> bool:
        if name in memo:
            return memo[name]
        if name in active or name not in methods:
            return False
        if name in _DETACHED_MUTATION_METHODS | _NON_FORKABLE_MUTATION_METHODS:
            memo[name] = False
            return False
        node = methods[name]
        result = (bool(_mutation_events(node))) or any(
            mutates(callee, active | {name}) for callee in _self_calls(node)
        )
        memo[name] = result
        return result

    return {name for name in methods if mutates(name)}


def _synthetic_method(source: str) -> ast.FunctionDef:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _synthetic_callable(
    source: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    node = ast.parse(source).body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return node


def _external_live_mutations(tree: ast.Module) -> set[str]:
    """Find fork-field writes through live aliases and local delegates."""

    functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for child in ast.walk(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(child.name, []).append(child)

    def scan(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        live_seed: frozenset[str] = frozenset(),
        active: frozenset[tuple[int, frozenset[str]]] = frozenset(),
    ) -> set[str]:
        key = (node.lineno, live_seed)
        if key in active:
            return set()
        live = set(live_seed)
        detached: set[str] = set()
        parameters = (*node.args.posonlyargs, *node.args.args)
        live.update(
            parameter.arg
            for parameter in parameters
            if parameter.arg == "session" or parameter.arg.endswith("_session")
        )

        # Resolve simple aliases to a fixed point before inspecting writes so
        # assignment order and delegated helpers cannot hide ownership.
        changed = True
        while changed:
            changed = False
            for child in _executed_nodes(node):
                if not isinstance(child, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    child.targets if isinstance(child, ast.Assign) else [child.target]
                )
                value = child.value
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if (
                        isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Name)
                        and value.func.id in {"copy", "deepcopy", "replace"}
                    ):
                        detached.add(target.id)
                        live.discard(target.id)
                        continue
                    if isinstance(value, ast.Name) and value.id in live:
                        if target.id not in live:
                            live.add(target.id)
                            changed = True
                    elif (
                        target.id == "session" or target.id.endswith("_session")
                    ) and target.id not in detached:
                        if target.id not in live:
                            live.add(target.id)
                            changed = True

        found: set[str] = set()
        for child in _executed_nodes(node):
            if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    child.targets if isinstance(child, ast.Assign) else [child.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and _root_name(target) in live
                        and target.attr in FORK_FIELD_ASSIGNMENTS
                    ):
                        found.add(target.attr)
            if not isinstance(child, ast.Call):
                continue
            if (
                isinstance(child.func, ast.Name)
                and child.func.id == "setattr"
                and len(child.args) >= 2
                and isinstance(child.args[0], ast.Name)
                and child.args[0].id in live
                and isinstance(child.args[1], ast.Constant)
                and child.args[1].value in FORK_FIELD_ASSIGNMENTS
            ):
                found.add(str(child.args[1].value))
            if (
                isinstance(child.func, ast.Attribute)
                and child.func.attr == "set"
                and isinstance(child.func.value, ast.Attribute)
                and child.func.value.attr == "rag_scope_holder"
                and _root_name(child.func.value) in live
            ):
                found.add("rag_scope")

            callee_name = (
                child.func.id
                if isinstance(child.func, ast.Name)
                else child.func.attr
                if isinstance(child.func, ast.Attribute)
                else None
            )
            callees = functions.get(callee_name or "", [])
            if not callees:
                continue
            for callee in callees:
                callee_parameters = (
                    *callee.args.posonlyargs,
                    *callee.args.args,
                    *callee.args.kwonlyargs,
                )
                if callee_parameters and callee_parameters[0].arg in {"self", "cls"}:
                    callee_parameters = callee_parameters[1:]
                delegated_live = {
                    parameter.arg
                    for parameter, argument in zip(callee_parameters, child.args)
                    if isinstance(argument, ast.Name) and argument.id in live
                }
                parameter_names = {parameter.arg for parameter in callee_parameters}
                delegated_live.update(
                    keyword.arg
                    for keyword in child.keywords
                    if keyword.arg in parameter_names
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id in live
                )
                if delegated_live:
                    found.update(
                        scan(
                            callee,
                            frozenset(delegated_live),
                            active | {key},
                        )
                    )
        return found

    found: set[str] = set()
    for candidates in functions.values():
        for function in candidates:
            found.update(scan(function))
    return found


def test_transition_call_after_mutation_does_not_fence_the_mutation() -> None:
    node = _synthetic_method(
        """
def unsafe(self):
    self.session.title = "changed"
    self._fork_source_transition("session")
"""
    )

    assert _transitioned(node) is False


def test_arbitrary_qualified_transition_decorator_is_rejected() -> None:
    node = _synthetic_method(
        """
@evil._fork_session_transition
def unsafe(self, session_id):
    self.session.title = "changed"
"""
    )

    assert _transitioned(node) is False


def test_store_owned_qualified_transition_decorator_is_recognized() -> None:
    node = _synthetic_method(
        """
@console_chat_store._fork_session_transition
def safe(self, session_id):
    self.session.title = "changed"
"""
    )

    assert _transitioned(node) is True


def test_dead_begin_and_unrelated_release_do_not_fence_mutation() -> None:
    node = _synthetic_method(
        """
def unsafe(self, session_id):
    self._begin_fork_source_transition(session_id)
    self.session.title = "changed"
    self._release_roleplay_fork_transition("unrelated", expected_session_id=session_id)
"""
    )

    assert _transitioned(node) is False


def test_roleplay_lease_requires_the_exact_balanced_token_lifecycle() -> None:
    node = _synthetic_method(
        """
def prepare_session_roleplay_projection_refresh(self, session_id):
    transition_token = self._begin_fork_source_transition(session_id)
    self._materialize_roleplay_projections_live(session_id)
    self._release_roleplay_fork_transition("unrelated", expected_session_id=session_id)
"""
    )

    assert _transitioned(node) is False


def test_roleplay_lease_rejects_unreachable_lifecycle() -> None:
    node = _synthetic_method(
        """
def prepare_session_roleplay_projection_refresh(self, session_id):
    if False:
        transition_token = str(uuid4())
        self._begin_fork_source_transition(session_id)
        self._roleplay_fork_transition_leases[transition_token] = session_id
        self._materialize_roleplay_projections_live(session_id)
        self._release_roleplay_fork_transition(
            transition_token, expected_session_id=session_id
        )
        return replace(plan, fork_transition_token=transition_token)
    return None
"""
    )

    assert _transitioned(node) is False


def test_roleplay_lease_rejects_conditional_release_before_token_return() -> None:
    node = _synthetic_method(
        """
def prepare_session_roleplay_projection_refresh(self, session_id, fail=False):
    transition_token = str(uuid4())
    self._begin_fork_source_transition(session_id)
    self._roleplay_fork_transition_leases[transition_token] = session_id
    plan = self._materialize_roleplay_projections_live(session_id)
    if fail:
        self._release_roleplay_fork_transition(
            transition_token, expected_session_id=session_id
        )
    return replace(plan, fork_transition_token=transition_token)
"""
    )

    assert _transitioned(node) is False


def test_roleplay_lease_rejects_unreleased_early_return_before_materialization() -> (
    None
):
    node = _synthetic_method(
        """
def prepare_session_roleplay_projection_refresh(self, session_id, skip=False):
    transition_token = str(uuid4())
    self._begin_fork_source_transition(session_id)
    with self._fork_source_lock:
        self._roleplay_fork_transition_leases[transition_token] = session_id
    try:
        if skip:
            return None
        plan = self._materialize_roleplay_projections_live(session_id)
        return replace(plan, fork_transition_token=transition_token)
    except BaseException:
        self._release_roleplay_fork_transition(
            transition_token, expected_session_id=session_id
        )
        raise
"""
    )

    assert _transitioned(node) is False


@pytest.mark.parametrize(
    "unsafe_exit",
    (
        "raise RuntimeError('unreleased')",
        "for item in items:\n            break",
    ),
)
def test_roleplay_lease_conservatively_rejects_unreleased_exit_paths(
    unsafe_exit: str,
) -> None:
    node = _synthetic_method(
        f"""
def prepare_session_roleplay_projection_refresh(self, session_id, items=()):
    transition_token = str(uuid4())
    self._begin_fork_source_transition(session_id)
    with self._fork_source_lock:
        self._roleplay_fork_transition_leases[transition_token] = session_id
    try:
        {unsafe_exit}
        plan = self._materialize_roleplay_projections_live(session_id)
        return replace(plan, fork_transition_token=transition_token)
    except BaseException:
        self._release_roleplay_fork_transition(
            transition_token, expected_session_id=session_id
        )
        raise
"""
    )

    assert _transitioned(node) is False


@pytest.mark.parametrize(
    "structured_exit",
    (
        "match mode:\n            case 'skip':\n                return None",
        "async for item in items:\n            return None",
        "try:\n            return None\n        except* ValueError:\n            pass",
    ),
)
def test_roleplay_lease_rejects_unreleased_structured_exits(
    structured_exit: str,
) -> None:
    node = _synthetic_callable(
        f"""
async def prepare_session_roleplay_projection_refresh(
    self, session_id, mode="continue", items=()
):
    transition_token = str(uuid4())
    self._begin_fork_source_transition(session_id)
    with self._fork_source_lock:
        self._roleplay_fork_transition_leases[transition_token] = session_id
    try:
        {structured_exit}
        plan = self._materialize_roleplay_projections_live(session_id)
        return replace(plan, fork_transition_token=transition_token)
    except BaseException:
        self._release_roleplay_fork_transition(
            transition_token, expected_session_id=session_id
        )
        raise
"""
    )

    assert _transitioned(node) is False


def test_wrong_session_transition_does_not_fence_target_owner() -> None:
    node = _synthetic_method(
        """
def unsafe(self, session_id, other_session_id):
    session = self._session_or_raise(session_id)
    with self._fork_source_transition(other_session_id):
        session.title = "changed"
"""
    )

    assert _transitioned(node) is False


def test_private_mutation_before_later_transition_is_not_fenced() -> None:
    tree = ast.parse(
        """
class Store:
    def public(self, session_id):
        self._private(session_id)
        with self._fork_source_transition(session_id):
            self._unrelated()

    def _private(self, session_id):
        session = self._session_or_raise(session_id)
        session.title = "changed"

    def _unrelated(self):
        return None
"""
    )
    store = tree.body[0]
    assert isinstance(store, ast.ClassDef)
    methods = {
        child.name: child
        for child in store.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "public" in _fork_mutating_routes(methods)
    assert _transitioned(methods["public"], methods=methods) is False


def test_deferred_nested_closure_is_fenced_only_at_protected_invocation() -> None:
    unsafe = _synthetic_method(
        """
def unsafe(self, session_id):
    session = self._session_or_raise(session_id)
    with self._fork_source_transition(session_id):
        def mutate():
            session.title = "changed"
    mutate()
"""
    )
    safe = _synthetic_method(
        """
def safe(self, session_id):
    session = self._session_or_raise(session_id)
    def mutate():
        session.title = "changed"
    with self._fork_source_transition(session_id):
        mutate()
"""
    )

    assert _transitioned(unsafe) is False
    assert _transitioned(safe) is True


def test_holder_setter_is_counted_as_a_fork_field_mutation() -> None:
    node = _synthetic_method(
        """
def unsafe(self):
    self.session.rag_scope_holder.set(None)
"""
    )

    assert "rag_scope" in _assigned_attributes(node)


def test_setattr_and_private_delegate_are_counted_as_fork_mutations() -> None:
    tree = ast.parse(
        """
class Store:
    def public(self):
        self._private()

    def _private(self):
        setattr(self.session, "title", "changed")
"""
    )
    store = tree.body[0]
    assert isinstance(store, ast.ClassDef)
    methods = {
        child.name: child
        for child in store.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "public" in _fork_mutating_routes(methods)


def test_public_fork_transition_inventory_is_bidirectional() -> None:
    methods = _store_methods()
    actual_direct = {
        name
        for name, node in methods.items()
        if not name.startswith("_") and _has_transition_boundary(node)
    }

    assert actual_direct == DIRECT_TRANSITION_ROUTES
    for route in DIRECT_TRANSITION_ROUTES - {"fork_source_transition"}:
        assert _transitioned(methods[route], methods=methods), route
    for route, owner in DELEGATED_TRANSITION_ROUTES.items():
        assert owner in _called_attributes(methods[route])
        assert "_fork_source_transition" in _called_attributes(methods[owner])

    detached = methods["persist_roleplay_projection_plan"]
    assert "self" not in {
        node.id for node in ast.walk(detached) if isinstance(node, ast.Name)
    }
    assert "fork_transition_token" in {
        node.attr for node in ast.walk(detached) if isinstance(node, ast.Attribute)
    }


def test_every_public_direct_fork_field_assignment_is_fenced_or_classified() -> None:
    methods = _store_methods()
    mutation_routes = {
        name for name in _fork_mutating_routes(methods) if not name.startswith("_")
    }
    transitioned_assignments = mutation_routes & DIRECT_TRANSITION_ROUTES

    assert mutation_routes - transitioned_assignments == (
        SAFE_NON_TRANSITION_ASSIGNMENTS | frozenset(DELEGATED_TRANSITION_ROUTES)
    )


def test_external_console_modules_do_not_write_live_fork_fields_directly() -> None:
    for path in (
        HYDRATION_PATH,
        CONTROLLER_PATH,
        IMAGE_RECOVERY_PATH,
        RETRIEVAL_PATH,
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        direct = _external_live_mutations(tree)
        assert not direct, f"{path}: direct live fork writes: {sorted(direct)}"

    hydration = ast.parse(HYDRATION_PATH.read_text(encoding="utf-8"))
    restore = next(
        child
        for child in ast.walk(hydration)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "restore_persisted_session"
    )
    assert {keyword.arg for keyword in restore.keywords} >= {
        "user_display_name_override",
        "character_system_template",
    }

    controller = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
    rollback_calls = [
        child
        for child in ast.walk(controller)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "rollback_transient_send"
    ]
    # Shutdown/readiness, thinking preflight, and queued Capture On each roll
    # back the exact optimistic echo through the store's guarded owner.
    assert [
        (
            ast.unparse(call.func),
            tuple(ast.unparse(arg) for arg in call.args),
            {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords},
        )
        for call in rollback_calls
    ] == [
        (
            "self.store.rollback_transient_send",
            ("session.id", "echoed_user.id"),
            {
                "title": "pre_send_title",
                "persisted_conversation_id": "pre_send_conversation_id",
            },
        )
    ] * 3


def test_external_writer_scan_follows_alias_setattr_and_holder_mutations() -> None:
    source = """
def mutate(session):
    current = session
    alias = current
    alias.title = "changed"
    setattr(current, "content", "changed")
    alias.rag_scope_holder.set(None)
"""

    assert _external_live_mutations(ast.parse(source)) == {
        "content",
        "rag_scope",
        "title",
    }


def test_external_writer_scan_allows_explicit_detached_dto_aliases() -> None:
    source = """
def stage(session):
    detached = replace(session)
    alias = detached
    alias.title = "staged"
"""

    assert _external_live_mutations(ast.parse(source)) == set()


def test_external_writer_scan_follows_live_alias_into_local_delegate() -> None:
    source = """
def mutate(owner):
    owner.title = "changed"

def route(session):
    current = session
    mutate(current)
"""

    assert _external_live_mutations(ast.parse(source)) == {"title"}


def test_external_writer_scan_maps_keyword_only_and_deep_delegate_aliases() -> None:
    source = """
def mutate(*, owner):
    owner.title = "changed"

def middle(current, *, callback_owner):
    mutate(owner=callback_owner)

def route(session):
    alias = session
    middle(alias, callback_owner=alias)
"""

    assert _external_live_mutations(ast.parse(source)) == {"title"}


def test_ui_image_recovery_never_writes_store_owned_fork_fields() -> None:
    tree = ast.parse(IMAGE_RECOVERY_PATH.read_text(encoding="utf-8"))
    forbidden = {
        "_active_leaf_by_session",
        "_bump_payload_revision",
        "_nodes_by_session",
        "_recompute_active_path",
        "_register_tree_node",
    }
    accessed = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "store"
    }
    assert accessed.isdisjoint(forbidden)
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "persisted_conversation_id"
            for target in node.targets
        )
        for node in ast.walk(tree)
    )


def test_external_rag_writer_uses_one_store_transition_and_publication_seam() -> None:
    tree = ast.parse(RETRIEVAL_PATH.read_text(encoding="utf-8"))
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ConsoleRetrievalController"
    )
    methods = {
        node.name: node
        for node in controller.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    owner = methods["_apply_console_retrieval_scope_save"]
    delegated = methods["_apply_console_retrieval_scope_save_transition"]

    assert "fork_source_transition" in _called_attributes(owner)
    assert "_apply_console_retrieval_scope_save_transition" in _called_attributes(owner)
    assert "set_session_rag_scope" in _called_attributes(delegated)
