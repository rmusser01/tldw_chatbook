"""Finite local permission payloads; every branch calls the original singleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .console_permission_operations import (
        HubTool,
        ServerMapping,
        ToolPackImportReview,
        ToolProfileBindingReview,
        WorkspaceAssistantDefaults,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadPermissionSnapshot:
    strict: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadPermissionInventory:
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadKillSwitch:
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadEffectiveStates:
    tools: list[HubTool]
    profile_id: str = "default"


@dataclass(frozen=True, slots=True, kw_only=True)
class CapturePermissionRender:
    tools: list[HubTool]
    profile_id: str = "default"


@dataclass(frozen=True, slots=True, kw_only=True)
class ListSessionApprovals:
    profile_id: str = "default"


@dataclass(frozen=True, slots=True, kw_only=True)
class RevokeSessionApproval:
    server_key: str
    tool_name: str
    profile_id: str = "default"


@dataclass(frozen=True, slots=True, kw_only=True)
class ClearSessionApprovals:
    profile_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SetToolState:
    server_key: str
    tool_name: str
    ui_state: str | None
    tool: HubTool | None = None
    profile_id: str = "default"
    expected_profile_digest: str | None = None
    expected_revision: int | None = None
    expected_policy_chain_digest: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SetServerDefault:
    server_key: str
    state: str | None
    profile_id: str = "default"
    expected_profile_digest: str | None = None
    expected_revision: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SetGlobalDefault:
    state: str
    profile_id: str = "default"
    expected_profile_digest: str | None = None
    expected_revision: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SetKillSwitch:
    value: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class AddArgRule:
    server_key: str
    tool_name: str
    args: Mapping[str, Any]
    tool: HubTool | None = None
    profile_id: str = "default"


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoveArgRule:
    server_key: str
    tool_name: str
    rule_id: str
    profile_id: str = "default"


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadArgRules:
    server_key: str
    tool_name: str
    profile_id: str = "default"


@dataclass(frozen=True, slots=True, kw_only=True)
class DismissSuggestion:
    server_key: str
    tool_name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RestoreSuggestion:
    server_key: str
    tool_name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ClearSuggestionDismissals:
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class InspectImport:
    archive_path: Path
    destination_id: str
    mappings: Sequence[ServerMapping] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ImportUnboundProfile:
    review: ToolPackImportReview


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoveImportedProfile:
    profile_id: str
    expected_revision: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ListToolProfiles:
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewFirstBind:
    workspace_id: str
    intended_defaults: WorkspaceAssistantDefaults
    action: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfirmFirstBind:
    review: ToolProfileBindingReview


PermissionOperation = (
    ReadPermissionSnapshot
    | ReadPermissionInventory
    | ReadKillSwitch
    | ReadEffectiveStates
    | CapturePermissionRender
    | ListSessionApprovals
    | RevokeSessionApproval
    | ClearSessionApprovals
    | SetToolState
    | SetServerDefault
    | SetGlobalDefault
    | SetKillSwitch
    | AddArgRule
    | RemoveArgRule
    | ReadArgRules
    | DismissSuggestion
    | RestoreSuggestion
    | ClearSuggestionDismissals
    | InspectImport
    | ImportUnboundProfile
    | RemoveImportedProfile
    | ListToolProfiles
    | ReviewFirstBind
    | ConfirmFirstBind
)

PERMISSION_OPERATION_TYPES = (
    ReadPermissionSnapshot,
    ReadPermissionInventory,
    ReadKillSwitch,
    ReadEffectiveStates,
    CapturePermissionRender,
    ListSessionApprovals,
    RevokeSessionApproval,
    ClearSessionApprovals,
    SetToolState,
    SetServerDefault,
    SetGlobalDefault,
    SetKillSwitch,
    AddArgRule,
    RemoveArgRule,
    ReadArgRules,
    DismissSuggestion,
    RestoreSuggestion,
    ClearSuggestionDismissals,
    InspectImport,
    ImportUnboundProfile,
    RemoveImportedProfile,
    ListToolProfiles,
    ReviewFirstBind,
    ConfirmFirstBind,
)


async def apply_permission_operation(operations, command):
    if type(command) is ReadPermissionSnapshot:
        return await operations.read_permission_snapshot(strict=command.strict)
    if type(command) is ReadPermissionInventory:
        return await operations.read_permission_inventory()
    if type(command) is ReadKillSwitch:
        return await operations.read_kill_switch()
    if type(command) is ReadEffectiveStates:
        return await operations.read_effective_states(
            tools=command.tools, profile_id=command.profile_id
        )
    if type(command) is CapturePermissionRender:
        return await operations.capture_permission_render(
            tools=command.tools, profile_id=command.profile_id
        )
    if type(command) is ListSessionApprovals:
        return await operations.list_session_approvals(profile_id=command.profile_id)
    if type(command) is RevokeSessionApproval:
        return await operations.revoke_session_approval(
            server_key=command.server_key,
            tool_name=command.tool_name,
            profile_id=command.profile_id,
        )
    if type(command) is ClearSessionApprovals:
        return await operations.clear_session_approvals(profile_id=command.profile_id)
    if type(command) is SetToolState:
        return await operations.set_tool_state(
            server_key=command.server_key,
            tool_name=command.tool_name,
            ui_state=command.ui_state,
            tool=command.tool,
            profile_id=command.profile_id,
            expected_profile_digest=command.expected_profile_digest,
            expected_revision=command.expected_revision,
            expected_policy_chain_digest=command.expected_policy_chain_digest,
        )
    if type(command) is SetServerDefault:
        return await operations.set_server_default(
            server_key=command.server_key,
            state=command.state,
            profile_id=command.profile_id,
            expected_profile_digest=command.expected_profile_digest,
            expected_revision=command.expected_revision,
        )
    if type(command) is SetGlobalDefault:
        return await operations.set_global_default(
            state=command.state,
            profile_id=command.profile_id,
            expected_profile_digest=command.expected_profile_digest,
            expected_revision=command.expected_revision,
        )
    if type(command) is SetKillSwitch:
        return await operations.set_kill_switch(value=command.value)
    if type(command) is AddArgRule:
        return await operations.add_arg_rule(
            server_key=command.server_key,
            tool_name=command.tool_name,
            args=command.args,
            tool=command.tool,
            profile_id=command.profile_id,
        )
    if type(command) is RemoveArgRule:
        return await operations.remove_arg_rule(
            server_key=command.server_key,
            tool_name=command.tool_name,
            rule_id=command.rule_id,
            profile_id=command.profile_id,
        )
    if type(command) is ReadArgRules:
        return await operations.read_arg_rules(
            server_key=command.server_key,
            tool_name=command.tool_name,
            profile_id=command.profile_id,
        )
    if type(command) is DismissSuggestion:
        return await operations.dismiss_suggestion(
            server_key=command.server_key, tool_name=command.tool_name
        )
    if type(command) is RestoreSuggestion:
        return await operations.restore_suggestion(
            server_key=command.server_key, tool_name=command.tool_name
        )
    if type(command) is ClearSuggestionDismissals:
        return await operations.clear_suggestion_dismissals()
    if type(command) is InspectImport:
        return await operations.inspect_import(
            archive_path=command.archive_path,
            destination_id=command.destination_id,
            mappings=command.mappings,
        )
    if type(command) is ImportUnboundProfile:
        return await operations.import_unbound_profile(review=command.review)
    if type(command) is RemoveImportedProfile:
        return await operations.remove_imported_profile(
            profile_id=command.profile_id, expected_revision=command.expected_revision
        )
    if type(command) is ListToolProfiles:
        return await operations.list_tool_profiles()
    if type(command) is ReviewFirstBind:
        return await operations.review_first_bind(
            workspace_id=command.workspace_id,
            intended_defaults=command.intended_defaults,
            action=command.action,
        )
    if type(command) is ConfirmFirstBind:
        return await operations.confirm_first_bind(review=command.review)
    raise TypeError("Unsupported local permission operation")
