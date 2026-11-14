"""Read-only identity proof against the reviewed pre-extraction commit."""

import ast
import inspect
import subprocess
from pathlib import Path

BASE = "8f05765780ab79ece5a26e2681939e2eff420ae5"
ROOT = Path(__file__).resolve().parents[3]
P = "tldw_chatbook/"
MOVES = {
    "app_service_wiring.py": {
        "Library/collections_capture_models.py": (
            "_external_reference_payload_id",
            "_external_reference_failure_reason",
            "_collections_reference_authority_kind",
        ),
    },
    "UI/Screens/llm_screen.py": {
        "UI/LLM_Management/vllm_profiles.py": ("_classify_vllm_profile_validation",),
    },
    "UI/Screens/watchlists_collections_screen.py": {
        "UI/Watchlists_Modules/table_selection.py": ("_normalize_items_status_filter",),
    },
    "UI/MCP_Modules/mcp_workbench.py": {
        "MCP/unified_control_models.py": ("_target_id_from_server_key",),
    },
    "Chat/console_interrupt_rounds.py": {
        "Chat/console_chat_models.py": ("_DecisionClock",),
    },
    "tldw_api/client.py": {
        "tldw_api/utils.py": (
            "_workspace_source_path_id",
            "_raise_api_error_from",
            "_raise_if_redirected",
        ),
        "tldw_api/chat_loop_schemas.py": (
            "ChatQueueStatusResponse",
            "ChatQueueActivityResponse",
        ),
    },
    "Chat/console_chat_controller.py": {
        "Chat/console_prepared_request.py": (
            "_request_fingerprint",
            "_context_overflow_cause",
            "_flatten_preflight_messages",
            "_continuation_restore_target_for_resolution",
            "_normalize_world_info_history",
            "_is_empty_transcript_row",
        ),
    },
    "Chat/console_chat_store.py": {
        "Chat/console_chat_models.py": (
            "_VariantStreamBase",
            "_GenerationOwnerLockEntry",
            "_GenerationMutationBoundary",
            "_RoleplaySystemPromptWrite",
            "_RoleplaySyncWrite",
            "_RoleplayMessageProjectionWrite",
            "_RoleplayMessageProjectionPersistenceOutcome",
            "_RoleplayContextWrite",
            "ConsoleRoleplayProjectionPersistencePlan",
            "ConsoleRoleplayProjectionPersistenceResult",
            "_RoleplayPersistenceLockEntry",
            "ConsoleStagedConversationIdentity",
            "ConsoleDurableTurnCommit",
            "ConsoleDurablePostcommitEffects",
            "ConsoleDurableAcceptanceFingerprint",
            "_ConsoleStagedDurableOwnerIds",
            "_ConsoleDurableCommitReservation",
            "_ConsoleDurableTombstone",
            "ConsoleSettingsComponent",
            "ConsoleSettingsPolicyFailureLabel",
            "ConsoleSettingsPersistenceFailure",
            "ConsoleSettingsPersistenceOutcome",
            "_ConsoleSettingsPersistenceDrain",
            "_ConsoleSettingsPersistenceLifecycle",
            "ContinuationDurabilityResult",
        ),
    },
    "Widgets/Console/console_transcript.py": {
        "Widgets/Console/console_selection.py": (
            "_snap_to_line_bounds",
            "_markdown_cell_to_offset",
            "_tool_diff_display_text",
            "_diff_cell_to_offset",
            "_console_markdown_body_ends_in_open_fence",
        ),
    },
}


def nodes(path, *, original=False):
    source = (
        subprocess.check_output(
            ["git", "show", f"{BASE}:{P}{path}"], cwd=ROOT, text=True
        )
        if original
        else (ROOT / P / path).read_text()
    )
    return ast.parse(source).body


def normalize(node):
    """Normalize postponed annotations and indentation-only docstring changes."""
    for item in ast.walk(node):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(item, clean=False)
            if doc is not None:
                item.body[0].value.value = inspect.cleandoc(doc)
        if isinstance(item, (ast.AnnAssign, ast.arg)):
            targets = [(item, "annotation")]
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            targets = [(item, "returns")]
        else:
            targets = []
        for owner, attribute in targets:
            value = getattr(owner, attribute)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                setattr(owner, attribute, ast.parse(value.value, mode="eval").body)
    return ast.dump(node, include_attributes=False)


moved = retained = 0
for source, destinations in MOVES.items():
    old = nodes(source, original=True)
    current = {getattr(n, "name", None): n for n in nodes(source)}
    routes = {name: dest for dest, names in destinations.items() for name in names}
    for node in old:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name in routes:
            target = next(
                n
                for n in nodes(routes[node.name])
                if getattr(n, "name", None) == node.name
            )
            if node.name == "_context_overflow_cause":
                delayed = [
                    n
                    for n in target.body
                    if isinstance(n, ast.ImportFrom)
                    and n.module == "tldw_chatbook.Chat.console_context_compaction"
                ]
                assert len(delayed) == 1
                assert ast.unparse(delayed[0]) == (
                    "from tldw_chatbook.Chat.console_context_compaction import CompactionDecision"
                )
                target.body.remove(delayed[0])
            assert normalize(node) == normalize(target), (source, node.name)
            moved += 1
        else:
            if source == "app_service_wiring.py" and node.name in {
                "_resolve_collections_media_reference",
                "_resolve_collections_note_reference",
            }:
                reference_import = next(
                    n
                    for n in current[node.name].body
                    if isinstance(n, ast.ImportFrom)
                    and n.module == "tldw_chatbook.Library.collections_capture_models"
                )
                assert [n.name for n in reference_import.names] == [
                    "ExternalReferenceAvailability",
                    "_collections_reference_authority_kind",
                    "_external_reference_failure_reason",
                    "_external_reference_payload_id",
                ]
                reference_import.names = reference_import.names[:1]
            if source == "tldw_api/client.py" and node.name == "TLDWAPIClient":
                redirect = next(
                    n
                    for n in node.body
                    if getattr(n, "name", None) == "_raise_if_redirected"
                )
                assert len(redirect.decorator_list) == 1
                assert ast.unparse(redirect.decorator_list[0]) == "staticmethod"
                redirect.decorator_list = []
                target = next(
                    n
                    for n in nodes("tldw_api/utils.py")
                    if getattr(n, "name", None) == redirect.name
                )
                assert normalize(redirect) == normalize(target)
                assignment = next(
                    n
                    for n in current[node.name].body
                    if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == redirect.name
                )
                assert (
                    ast.unparse(assignment.value)
                    == "staticmethod(_raise_if_redirected)"
                )
                node.body.remove(redirect)
                current[node.name].body.remove(assignment)
                moved += 1
            assert normalize(node) == normalize(current[node.name]), (source, node.name)
            retained += 1
assert moved == 48, moved
for destination in {d for routes in MOVES.values() for d in routes}:
    if not (ROOT / P / destination).exists():
        raise AssertionError(destination)
    current = {getattr(n, "name", None): n for n in nodes(destination)}
    for node in nodes(destination, original=True):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert normalize(node) == normalize(current[node.name]), (
                destination,
                node.name,
            )
            retained += 1

original_capture = next(
    n
    for n in nodes("app_sqlite_retirement.py", original=True)
    if getattr(n, "name", None) == "capture_chat_borrower"
)
current_capture = next(
    n
    for n in nodes("Utils/app_shutdown.py")
    if getattr(n, "name", None) == "capture_chat_borrower"
)
assert ast.unparse(original_capture.body[1]) == "from . import config"
assert ast.unparse(current_capture.body[1]) == "from .. import config"
# Both relative imports select the same tldw_chatbook.config owner.
original_capture.body[1].level = 2
assert normalize(original_capture) == normalize(current_capture)
for source in ("Utils/app_shutdown.py", "app_sqlite_retirement.py"):
    current = {getattr(n, "name", None): n for n in nodes(source)}
    for node in nodes(source, original=True):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == "capture_chat_borrower":
                continue
            assert normalize(node) == normalize(current[node.name]), (source, node.name)
            retained += 1


def literal_assignment(path, name, *, original=False):
    assignment = next(
        n
        for n in nodes(path, original=original)
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (n.targets if isinstance(n, ast.Assign) else [n.target])
        )
    )
    return ast.literal_eval(assignment.value)


assert literal_assignment(
    "tldw_api/notes_workspace_schemas.py",
    "MAX_WORKSPACE_SOURCE_ID_CHARS",
    original=True,
) == literal_assignment(
    "tldw_api/notes_workspace_limits.py", "MAX_WORKSPACE_SOURCE_ID_CHARS"
)
for source in ("tldw_api/notes_workspace_schemas.py", "tldw_api/utils.py"):
    assert any(
        isinstance(n, ast.ImportFrom)
        and n.module.endswith("notes_workspace_limits")
        and any(a.name == "MAX_WORKSPACE_SOURCE_ID_CHARS" for a in n.names)
        for n in nodes(source)
    ), source
print(
    f"{moved} moved definitions and {retained} retained definitions match; "
    "borrower observation, source ID bound and redirect static binding preserved."
)
