"""Compatibility-preserving lazy public exports; recovery imports no services."""

from importlib import import_module

_EXPORTS = {
    "CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT": (
        ".conversation_browser_state",
        "CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT",
    ),
    "CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT": (
        ".conversation_browser_state",
        "CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT",
    ),
    "ConsoleConversationBrowserGroup": (
        ".conversation_browser_state",
        "ConsoleConversationBrowserGroup",
    ),
    "ConsoleConversationBrowserInputRow": (
        ".conversation_browser_state",
        "ConsoleConversationBrowserInputRow",
    ),
    "ConsoleConversationBrowserRow": (
        ".conversation_browser_state",
        "ConsoleConversationBrowserRow",
    ),
    "ConsoleConversationBrowserSection": (
        ".conversation_browser_state",
        "ConsoleConversationBrowserSection",
    ),
    "ConsoleConversationBrowserState": (
        ".conversation_browser_state",
        "ConsoleConversationBrowserState",
    ),
    "build_console_conversation_browser_state": (
        ".conversation_browser_state",
        "build_console_conversation_browser_state",
    ),
    "console_persisted_row_updated_sort": (
        ".conversation_browser_state",
        "console_persisted_row_updated_sort",
    ),
    "CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT": (
        ".display_state",
        "CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT",
    ),
    "ConsoleWorkspaceACPHandoffState": (
        ".display_state",
        "ConsoleWorkspaceACPHandoffState",
    ),
    "ConsoleWorkspaceContextState": (".display_state", "ConsoleWorkspaceContextState"),
    "ConsoleWorkspaceConversationRow": (
        ".display_state",
        "ConsoleWorkspaceConversationRow",
    ),
    "ConsoleWorkspaceConversationSectionState": (
        ".display_state",
        "ConsoleWorkspaceConversationSectionState",
    ),
    "ConsoleWorkspaceHandoffRow": (".display_state", "ConsoleWorkspaceHandoffRow"),
    "ConsoleWorkspaceServerAdapterState": (
        ".display_state",
        "ConsoleWorkspaceServerAdapterState",
    ),
    "LibraryWorkspaceDepthState": (".display_state", "LibraryWorkspaceDepthState"),
    "LibraryWorkspaceSourceRow": (".display_state", "LibraryWorkspaceSourceRow"),
    "build_library_workspace_depth_state": (
        ".display_state",
        "build_library_workspace_depth_state",
    ),
    "library_item_context_handoff": (".display_state", "library_item_context_handoff"),
    "build_console_workspace_state": (
        ".display_state",
        "build_console_workspace_state",
    ),
    "console_workspace_conversation_result_copy": (
        ".display_state",
        "console_workspace_conversation_result_copy",
    ),
    "console_workspace_conversation_visible_rows": (
        ".display_state",
        "console_workspace_conversation_visible_rows",
    ),
    "evaluate_workspace_eligibility": (
        ".eligibility",
        "evaluate_workspace_eligibility",
    ),
    "DEFAULT_WORKSPACE_ID": (".models", "DEFAULT_WORKSPACE_ID"),
    "DEFAULT_WORKSPACE_NAME": (".models", "DEFAULT_WORKSPACE_NAME"),
    "RuntimeBindingKind": (".models", "RuntimeBindingKind"),
    "RuntimeBindingStatus": (".models", "RuntimeBindingStatus"),
    "WorkspaceAuthority": (".models", "WorkspaceAuthority"),
    "WorkspaceEligibility": (".models", "WorkspaceEligibility"),
    "WorkspaceMembership": (".models", "WorkspaceMembership"),
    "WorkspaceOperation": (".models", "WorkspaceOperation"),
    "WorkspaceRecord": (".models", "WorkspaceRecord"),
    "WorkspaceRuntimeBinding": (".models", "WorkspaceRuntimeBinding"),
    "WorkspaceSyncStatus": (".models", "WorkspaceSyncStatus"),
    "WorkspaceTransferPolicy": (".models", "WorkspaceTransferPolicy"),
    "BindingNotFound": (".registry_service", "BindingNotFound"),
    "LocalWorkspaceRegistryService": (
        ".registry_service",
        "LocalWorkspaceRegistryService",
    ),
}

__all__ = [
    "CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT",
    "CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT",
    "CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT",
    "BindingNotFound",
    "ConsoleConversationBrowserGroup",
    "ConsoleConversationBrowserInputRow",
    "ConsoleConversationBrowserRow",
    "ConsoleConversationBrowserSection",
    "ConsoleConversationBrowserState",
    "ConsoleWorkspaceContextState",
    "ConsoleWorkspaceConversationRow",
    "ConsoleWorkspaceConversationSectionState",
    "ConsoleWorkspaceACPHandoffState",
    "ConsoleWorkspaceHandoffRow",
    "ConsoleWorkspaceServerAdapterState",
    "DEFAULT_WORKSPACE_ID",
    "DEFAULT_WORKSPACE_NAME",
    "LibraryWorkspaceDepthState",
    "LibraryWorkspaceSourceRow",
    "LocalWorkspaceRegistryService",
    "RuntimeBindingKind",
    "RuntimeBindingStatus",
    "WorkspaceAuthority",
    "WorkspaceEligibility",
    "WorkspaceMembership",
    "WorkspaceOperation",
    "WorkspaceRecord",
    "WorkspaceRuntimeBinding",
    "WorkspaceSyncStatus",
    "WorkspaceTransferPolicy",
    "build_console_conversation_browser_state",
    "build_library_workspace_depth_state",
    "library_item_context_handoff",
    "build_console_workspace_state",
    "console_persisted_row_updated_sort",
    "console_workspace_conversation_result_copy",
    "console_workspace_conversation_visible_rows",
    "evaluate_workspace_eligibility",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
