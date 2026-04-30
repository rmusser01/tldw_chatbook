"""Machine-readable domain edge contracts for source-honest UX handoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DomainAuthority = Literal["local_and_server", "local_parity", "remote_only", "server_primary"]
SourceSelectorState = Literal["local", "server", "workspace"]

REQUIRED_UNSUPPORTED_REASON_CODES = (
    "server_required",
    "server_unavailable",
    "auth_required",
    "permission_denied",
    "capability_missing",
    "not_implemented_locally",
)

REMOTE_ONLY_DOMAIN_IDS = (
    "sharing",
    "web_clipper",
    "translation",
    "server_tools",
    "text2sql",
    "server_skills",
    "claims",
    "meetings",
    "outputs",
    "kanban",
    "prompt_studio",
)


@dataclass(frozen=True, slots=True)
class DomainEdgeContract:
    domain_id: str
    label: str
    authority: DomainAuthority
    source_selector_states: tuple[SourceSelectorState, ...]
    view_model_contract: str
    workspace_isolation: str = "none"
    uses_event_contract: bool = False
    uses_sync_contract: bool = False
    unsupported_local_reason_codes: tuple[str, ...] = ()

    def as_matrix_entry(self) -> dict[str, object]:
        return {
            "domain_id": self.domain_id,
            "label": self.label,
            "authority": self.authority,
            "source_selector_states": self.source_selector_states,
            "view_model_contract": self.view_model_contract,
            "workspace_isolation": self.workspace_isolation,
            "uses_event_contract": self.uses_event_contract,
            "uses_sync_contract": self.uses_sync_contract,
            "unsupported_local_reason_codes": self.unsupported_local_reason_codes,
        }


_DOMAIN_EDGE_CONTRACTS: tuple[DomainEdgeContract, ...] = (
    DomainEdgeContract(
        domain_id="chat",
        label="Chat",
        authority="local_and_server",
        source_selector_states=("local", "server", "workspace"),
        view_model_contract="chat_source_honest_view_v1",
        workspace_isolation="required",
        uses_event_contract=True,
        uses_sync_contract=True,
    ),
    DomainEdgeContract(
        domain_id="media_reading",
        label="Media And Reading",
        authority="local_and_server",
        source_selector_states=("local", "server"),
        view_model_contract="media_reading_source_honest_view_v1",
        uses_event_contract=True,
        uses_sync_contract=True,
    ),
    DomainEdgeContract(
        domain_id="notes_workspaces",
        label="Notes And Workspaces",
        authority="local_and_server",
        source_selector_states=("local", "server", "workspace"),
        view_model_contract="notes_workspace_source_honest_view_v1",
        workspace_isolation="required",
        uses_sync_contract=True,
    ),
    DomainEdgeContract(
        domain_id="writing",
        label="Writing",
        authority="local_and_server",
        source_selector_states=("local", "server", "workspace"),
        view_model_contract="writing_source_honest_view_v1",
        workspace_isolation="project_required",
        uses_sync_contract=True,
        unsupported_local_reason_codes=("not_implemented_locally",),
    ),
    DomainEdgeContract(
        domain_id="research",
        label="Research",
        authority="local_and_server",
        source_selector_states=("local", "server", "workspace"),
        view_model_contract="research_source_honest_view_v1",
        workspace_isolation="session_required",
        uses_event_contract=True,
        uses_sync_contract=True,
    ),
    DomainEdgeContract(
        domain_id="study_evaluations",
        label="Study And Evaluations",
        authority="local_and_server",
        source_selector_states=("local", "server", "workspace"),
        view_model_contract="study_evaluations_source_honest_view_v1",
        workspace_isolation="optional",
        uses_event_contract=True,
    ),
    DomainEdgeContract(
        domain_id="rag_embeddings",
        label="RAG And Embeddings",
        authority="server_primary",
        source_selector_states=("local", "server"),
        view_model_contract="rag_embeddings_source_honest_view_v1",
        uses_event_contract=True,
        unsupported_local_reason_codes=("not_implemented_locally",),
    ),
    DomainEdgeContract(
        domain_id="audio_voice",
        label="Audio And Voice",
        authority="local_and_server",
        source_selector_states=("local", "server"),
        view_model_contract="audio_voice_source_honest_view_v1",
        uses_event_contract=True,
    ),
)

_REMOTE_ONLY_CONTRACTS: tuple[DomainEdgeContract, ...] = tuple(
    DomainEdgeContract(
        domain_id=domain_id,
        label=domain_id.replace("_", " ").title(),
        authority="remote_only",
        source_selector_states=("server",),
        view_model_contract=f"{domain_id}_remote_only_view_v1",
        uses_event_contract=True,
        unsupported_local_reason_codes=("server_required",),
    )
    for domain_id in REMOTE_ONLY_DOMAIN_IDS
)

_CONTRACTS_BY_DOMAIN = {
    contract.domain_id: contract
    for contract in (*_DOMAIN_EDGE_CONTRACTS, *_REMOTE_ONLY_CONTRACTS)
}


def list_domain_edge_contracts() -> tuple[DomainEdgeContract, ...]:
    return tuple(_CONTRACTS_BY_DOMAIN.values())


def get_domain_edge_contract(domain_id: str) -> DomainEdgeContract:
    try:
        return _CONTRACTS_BY_DOMAIN[domain_id]
    except KeyError as exc:
        raise KeyError(f"Unknown domain edge contract: {domain_id}") from exc


def build_domain_capability_matrix() -> dict[str, dict[str, object]]:
    return {
        domain_id: contract.as_matrix_entry()
        for domain_id, contract in _CONTRACTS_BY_DOMAIN.items()
    }


def build_unsupported_action_report(
    *,
    domain_id: str,
    source: SourceSelectorState,
    reason_code: str | None = None,
) -> dict[str, object]:
    contract = get_domain_edge_contract(domain_id)
    resolved_reason = reason_code or _default_reason_code(contract=contract, source=source)
    if resolved_reason not in REQUIRED_UNSUPPORTED_REASON_CODES:
        raise ValueError(f"Unsupported reason code: {resolved_reason}")
    return {
        "operation_id": f"{domain_id}.unsupported.{source}",
        "source": source,
        "supported": False,
        "reason_code": resolved_reason,
        "user_message": _unsupported_message(contract=contract, source=source, reason_code=resolved_reason),
        "affected_action_ids": [],
        "domain_id": domain_id,
        "view_model_contract": contract.view_model_contract,
    }


def _default_reason_code(*, contract: DomainEdgeContract, source: SourceSelectorState) -> str:
    if contract.authority == "remote_only" and source != "server":
        return "server_required"
    if source == "local":
        return "not_implemented_locally"
    return "capability_missing"


def _unsupported_message(
    *,
    contract: DomainEdgeContract,
    source: SourceSelectorState,
    reason_code: str,
) -> str:
    if reason_code == "server_required":
        return f"{contract.label} is owned by the active server and is unavailable in {source} mode."
    if reason_code == "server_unavailable":
        return f"{contract.label} requires the active server, but the server is unavailable."
    if reason_code == "auth_required":
        return f"{contract.label} requires server authentication."
    if reason_code == "permission_denied":
        return f"The authenticated user is not allowed to use {contract.label}."
    if reason_code == "not_implemented_locally":
        return f"{contract.label} does not have local parity for this operation."
    return f"{contract.label} is missing the required server capability."
