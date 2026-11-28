from dataclasses import replace

from Tests.Research_Workspace.test_studio_models import (
    LOCAL,
    ROUTE,
    SECOND,
    SERVER,
    SOURCE,
)
from tldw_chatbook.Research_Workspace.contracts import ResearchCapability
from tldw_chatbook.Research_Workspace.studio_mapping import (
    PRIMARY_OUTPUT_MAPPINGS,
    project_studio_availability,
)
from tldw_chatbook.Research_Workspace.studio_models import PRIMARY_RESEARCH_OUTPUT_KINDS


def test_primary_output_mapping_keeps_all_ten_canonical_owner_paths_explicit():
    expected = {
        "local": (
            ("local_chatbook", "chatbook", None, "artifact_revision"),
            ("local_study_deck", "study_deck", None, "version"),
            ("local_quiz", "study_quiz", None, "version"),
            ("local_chatbook", "chatbook", None, "artifact_revision"),
            ("local_chatbook", "chatbook", None, "artifact_revision"),
        ),
        "server": (
            ("server_artifact", None, "summary", "version"),
            ("server_study_deck", None, None, "version"),
            ("server_quiz", None, None, "version"),
            ("server_artifact", None, "report", "version"),
            ("server_artifact", None, "compare_sources", "version"),
        ),
    }
    assert len(PRIMARY_OUTPUT_MAPPINGS) == 10
    for authority, rows in expected.items():
        for kind, expected_owner in zip(
            PRIMARY_RESEARCH_OUTPUT_KINDS, rows, strict=True
        ):
            owner = PRIMARY_OUTPUT_MAPPINGS[authority, kind]
            assert (
                owner.owner_kind,
                owner.membership_item_type,
                owner.artifact_type,
                owner.version_field,
            ) == expected_owner
            assert (
                owner.create_capability
                and owner.reopen_destination
                and owner.reopen_channel
            )
            assert owner.id_field and owner.association

    server_report = PRIMARY_OUTPUT_MAPPINGS["server", "report"]
    assert "replace" in server_report.supported_actions
    assert "delete" not in server_report.supported_actions


def test_studio_cards_require_explicit_capability_a_route_and_enough_ready_sources():
    summary = PRIMARY_OUTPUT_MAPPINGS["local", "summary"]
    compare = PRIMARY_OUTPUT_MAPPINGS["local", "compare_sources"]
    ready = ResearchCapability(
        True, "available", "Available", "owner", capability_revision="r4"
    )
    capabilities = {summary.create_capability: ready, compare.create_capability: ready}

    projected = project_studio_availability(LOCAL, (SOURCE,), ROUTE, capabilities)
    assert projected["summary"].available
    assert projected["summary"].owner_label == summary.owner_label
    assert projected["summary"].processing_route == ROUTE
    assert projected["summary"].capability_revision == "r4"
    assert projected["compare_sources"].reason_code == "insufficient_ready_sources"
    assert projected["flashcards"].reason_code == "studio_capability_unknown"
    assert not projected["flashcards"].available

    assert project_studio_availability(LOCAL, (SOURCE, SECOND), ROUTE, capabilities)[
        "compare_sources"
    ].available
    assert not project_studio_availability(LOCAL, (SOURCE,), None, capabilities)[
        "summary"
    ].available
    assert not project_studio_availability(
        SERVER, (replace(SOURCE, ref=SERVER),), ROUTE, capabilities
    )["summary"].available

    denied = replace(
        ready, available=False, reason_code="quota", recovery_action="Review quota."
    )
    unavailable = project_studio_availability(
        LOCAL, (SOURCE,), ROUTE, {summary.create_capability: denied}
    )["summary"]
    assert not unavailable.available
    assert unavailable.reason_code == "quota"
    assert unavailable.recovery == "Review quota."
