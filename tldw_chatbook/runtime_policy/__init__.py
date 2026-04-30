from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CAPABILITY_REGISTRY": (".registry", "CAPABILITY_REGISTRY"),
    "CapabilityEntry": (".types", "CapabilityEntry"),
    "DEFAULT_KEYRING_SERVICE_NAME": (".server_credentials", "DEFAULT_KEYRING_SERVICE_NAME"),
    "PHASE_ONE_REQUIRED_ACTION_IDS": (".registry", "PHASE_ONE_REQUIRED_ACTION_IDS"),
    "POLICY_FRESHNESS_WINDOW": (".source_state", "POLICY_FRESHNESS_WINDOW"),
    "PolicyDecision": (".types", "PolicyDecision"),
    "PolicyDeniedError": (".types", "PolicyDeniedError"),
    "PolicyEngine": (".engine", "PolicyEngine"),
    "RuntimeSourceState": (".types", "RuntimeSourceState"),
    "SERVER_CONTEXT_FAILURE_REASON_CODES": (".types", "SERVER_CONTEXT_FAILURE_REASON_CODES"),
    "SERVER_CREDENTIAL_ACCESS_TOKEN": (".server_credentials", "SERVER_CREDENTIAL_ACCESS_TOKEN"),
    "SERVER_CREDENTIAL_API_KEY": (".server_credentials", "SERVER_CREDENTIAL_API_KEY"),
    "SERVER_CREDENTIAL_BEARER_TOKEN": (".server_credentials", "SERVER_CREDENTIAL_BEARER_TOKEN"),
    "SERVER_CREDENTIAL_REFRESH_TOKEN": (".server_credentials", "SERVER_CREDENTIAL_REFRESH_TOKEN"),
    "ServicePolicyEnforcer": (".enforcement", "ServicePolicyEnforcer"),
    "InMemoryServerCredentialStore": (".server_credentials", "InMemoryServerCredentialStore"),
    "KeyringServerCredentialStore": (".server_credentials", "KeyringServerCredentialStore"),
    "ServerCredentialRef": (".server_credentials", "ServerCredentialRef"),
    "ServerCredentialScope": (".server_credentials", "ServerCredentialScope"),
    "ServerCredentialStore": (".server_credentials", "ServerCredentialStore"),
    "ServerContextError": (".server_context", "ServerContextError"),
    "ServerContextFailure": (".types", "ServerContextFailure"),
    "ServerContextFailureReason": (".types", "ServerContextFailureReason"),
    "UnsupportedCapabilityReportError": (".unsupported_capabilities", "UnsupportedCapabilityReportError"),
    "classify_backend_exception": (".enforcement", "classify_backend_exception"),
    "collect_unsupported_capability_reports": (
        ".unsupported_capabilities",
        "collect_unsupported_capability_reports",
    ),
    "get_capability_entry": (".registry", "get_capability_entry"),
    "normalize_runtime_source_state": (".source_state", "normalize_runtime_source_state"),
    "ActiveServerCapabilityService": (".server_capabilities", "ActiveServerCapabilityService"),
    "ActiveServerContext": (".server_context", "ActiveServerContext"),
    "DomainEdgeContract": (".domain_edge_contracts", "DomainEdgeContract"),
    "EventCursor": (".server_parity_models", "EventCursor"),
    "EventDedupeKey": (".server_parity_models", "EventDedupeKey"),
    "NormalizedEventRecord": (".server_parity_models", "NormalizedEventRecord"),
    "NotificationPresentationRecord": (".server_parity_models", "NotificationPresentationRecord"),
    "ProviderMigrationStatus": (".server_parity_models", "ProviderMigrationStatus"),
    "build_domain_capability_matrix": (".domain_edge_contracts", "build_domain_capability_matrix"),
    "build_unsupported_action_report": (".domain_edge_contracts", "build_unsupported_action_report"),
    "get_domain_edge_contract": (".domain_edge_contracts", "get_domain_edge_contract"),
    "list_domain_edge_contracts": (".domain_edge_contracts", "list_domain_edge_contracts"),
    "redact_secret": (".server_credentials", "redact_secret"),
    "runtime_source_state_from_dict": (".source_state", "runtime_source_state_from_dict"),
    "runtime_source_state_to_dict": (".source_state", "runtime_source_state_to_dict"),
    "RuntimeServerContextProvider": (".server_context", "RuntimeServerContextProvider"),
    "ServerContextUnavailable": (".server_context", "ServerContextUnavailable"),
    "ServerCredentialsUnavailable": (".server_context", "ServerCredentialsUnavailable"),
    "SyncIdentityMapEntry": (".server_parity_models", "SyncIdentityMapEntry"),
    "SyncReadinessReport": (".server_parity_models", "SyncReadinessReport"),
    "validate_unsupported_capability_report": (
        ".unsupported_capabilities",
        "validate_unsupported_capability_report",
    ),
    "validate_registry_completeness": (".registry", "validate_registry_completeness"),
}

__all__ = [
    "CAPABILITY_REGISTRY",
    "CapabilityEntry",
    "DEFAULT_KEYRING_SERVICE_NAME",
    "PHASE_ONE_REQUIRED_ACTION_IDS",
    "POLICY_FRESHNESS_WINDOW",
    "PolicyDecision",
    "PolicyDeniedError",
    "PolicyEngine",
    "RuntimeSourceState",
    "SERVER_CONTEXT_FAILURE_REASON_CODES",
    "SERVER_CREDENTIAL_ACCESS_TOKEN",
    "SERVER_CREDENTIAL_API_KEY",
    "SERVER_CREDENTIAL_BEARER_TOKEN",
    "SERVER_CREDENTIAL_REFRESH_TOKEN",
    "ServicePolicyEnforcer",
    "InMemoryServerCredentialStore",
    "KeyringServerCredentialStore",
    "ServerCredentialRef",
    "ServerCredentialScope",
    "ServerCredentialStore",
    "ServerContextError",
    "ServerContextFailure",
    "ServerContextFailureReason",
    "UnsupportedCapabilityReportError",
    "classify_backend_exception",
    "collect_unsupported_capability_reports",
    "get_capability_entry",
    "normalize_runtime_source_state",
    "ActiveServerCapabilityService",
    "ActiveServerContext",
    "DomainEdgeContract",
    "EventCursor",
    "EventDedupeKey",
    "NormalizedEventRecord",
    "NotificationPresentationRecord",
    "ProviderMigrationStatus",
    "build_domain_capability_matrix",
    "build_unsupported_action_report",
    "get_domain_edge_contract",
    "list_domain_edge_contracts",
    "redact_secret",
    "runtime_source_state_from_dict",
    "runtime_source_state_to_dict",
    "RuntimeServerContextProvider",
    "ServerContextUnavailable",
    "ServerCredentialsUnavailable",
    "SyncIdentityMapEntry",
    "SyncReadinessReport",
    "validate_unsupported_capability_report",
    "validate_registry_completeness",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
