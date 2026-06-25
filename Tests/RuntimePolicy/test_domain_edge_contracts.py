from __future__ import annotations

from tldw_chatbook.runtime_policy.domain_edge_contracts import (
    REQUIRED_UNSUPPORTED_REASON_CODES,
    REMOTE_ONLY_DOMAIN_IDS,
    build_domain_capability_matrix,
    build_remote_utility_local_parity_matrix,
    build_unsupported_action_report,
    get_domain_edge_contract,
    get_remote_utility_local_parity,
    list_domain_edge_contracts,
)
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
from tldw_chatbook.runtime_policy.unsupported_capabilities import validate_unsupported_capability_report


def test_domain_edge_matrix_covers_priority_server_parity_domains():
    expected_domains = {
        "chat",
        "media_reading",
        "notes_workspaces",
        "writing",
        "research",
        "study_evaluations",
        "rag_embeddings",
        "audio_voice",
        "sharing",
        "web_clipper",
        "translation",
        "server_tools",
        "text2sql",
        "skills",
        "claims",
        "meetings",
        "outputs",
        "kanban",
        "prompt_studio",
    }

    matrix = build_domain_capability_matrix()

    assert expected_domains.issubset(matrix)
    assert matrix["chat"]["source_selector_states"] == ("local", "server", "workspace")
    assert matrix["chat"]["uses_sync_contract"] is True
    assert matrix["sharing"]["authority"] == "remote_only"
    assert matrix["translation"]["authority"] == "local_parity"
    assert matrix["translation"]["source_selector_states"] == ("local", "server")
    assert matrix["skills"]["authority"] == "local_and_server"
    assert matrix["skills"]["source_selector_states"] == ("local", "server")
    assert matrix["kanban"]["authority"] == "local_and_server"
    assert matrix["kanban"]["source_selector_states"] == ("local", "server")
    assert "server_skills" not in matrix
    assert matrix["media_reading"]["uses_event_contract"] is True
    assert matrix["notes_workspaces"]["uses_sync_contract"] is True


def test_remote_only_local_reports_use_common_unsupported_shape():
    assert "translation" not in REMOTE_ONLY_DOMAIN_IDS

    for domain_id in REMOTE_ONLY_DOMAIN_IDS:
        report = build_unsupported_action_report(domain_id=domain_id, source="local")
        validated = validate_unsupported_capability_report([report], registry={})

        assert validated[0]["operation_id"] == f"{domain_id}.unsupported.local"
        assert validated[0]["source"] == "local"
        assert validated[0]["supported"] is False
        assert validated[0]["reason_code"] == "server_required"


def test_remote_utility_local_parity_registry_tracks_translation_pilot():
    matrix = build_remote_utility_local_parity_matrix()
    translation = get_remote_utility_local_parity("translation")

    assert translation.state == "pilot"
    assert translation.local_adapter == "TranslationScopeService.local_service"
    assert matrix["translation"]["state"] == "pilot"
    assert matrix["skills"]["state"] == "planned"
    assert matrix["kanban"]["state"] == "planned"
    assert "server_skills" not in matrix
    assert matrix["sharing"]["state"] == "remote_only"
    assert matrix["outputs"]["state"] == "remote_only"


def test_server_skills_alias_returns_canonical_skills_contract():
    assert get_domain_edge_contract("server_skills") is get_domain_edge_contract("skills")
    assert get_remote_utility_local_parity("server_skills") is get_remote_utility_local_parity("skills")


def test_skills_and_kanban_have_local_policy_actions():
    for action_id in [
        "skills.list.local",
        "skills.execute.launch.local",
        "skills.trust.unlock.local",
        "skills.trust.review.local",
        "skills.trust.approve.local",
        "skills.trust.reject.local",
        "skills.trust.rebootstrap.local",
        "skills.trust.rotate_key.local",
        "skills.trust.audit.local",
        "kanban.boards.list.local",
        "kanban.cards.create.local",
        "kanban.card_links.delete.local",
    ]:
        entry = CAPABILITY_REGISTRY[action_id]
        assert entry.required_source == "local"
        assert entry.authority_owner == "local"


def test_required_reason_codes_are_explicit_and_reportable():
    assert REQUIRED_UNSUPPORTED_REASON_CODES == (
        "server_required",
        "server_unavailable",
        "auth_required",
        "permission_denied",
        "capability_missing",
        "not_implemented_locally",
    )

    for reason_code in REQUIRED_UNSUPPORTED_REASON_CODES:
        report = build_unsupported_action_report(
            domain_id="sharing",
            source="server" if reason_code != "server_required" else "local",
            reason_code=reason_code,
        )
        validated = validate_unsupported_capability_report([report], registry={})

        assert validated[0]["reason_code"] == reason_code


def test_domain_contracts_expose_view_model_and_workspace_rules():
    writing = get_domain_edge_contract("writing")
    contracts = list_domain_edge_contracts()

    assert writing.view_model_contract == "writing_source_honest_view_v1"
    assert writing.unsupported_local_reason_codes == ("not_implemented_locally",)
    assert any(contract.domain_id == "research" for contract in contracts)
    assert get_domain_edge_contract("notes_workspaces").workspace_isolation == "required"
