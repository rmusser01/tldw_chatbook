"""Only exact current capability and source evidence can enable an output."""

from dataclasses import replace

from tldw_chatbook.Research_Workspace.action_registry import (
    ResearchActionContext,
    ResearchActionProbes,
    project_research_actions,
)
from tldw_chatbook.Research_Workspace.chat_models import ResearchSourceVersionRef
from tldw_chatbook.Research_Workspace.contracts import (
    ProcessingRoute,
    QualifiedWorkspaceRef,
    ResearchCapability,
)


def _capture(authority="server"):
    ref = QualifiedWorkspaceRef(
        authority,
        "workspace",
        "profile" if authority == "server" else "",
        "account" if authority == "server" else "",
    )
    route = ProcessingRoute(
        authority, "research_fixture", provider="fixture", model="model"
    )
    source = ResearchSourceVersionRef(ref, "source", "41", "document", 1, 2, ("fts",))
    return ResearchActionContext(ref, "owner-revision", route, (source,))


def _ready(name):
    return ResearchCapability(
        True, "available", "Ready", name, capability_revision="owner-revision"
    )


def test_coarse_server_categories_cannot_enable_individual_actions():
    capture = _capture()
    probes = ResearchActionProbes(
        capture,
        {
            name: _ready(name)
            for name in (
                "text",
                "slides",
                "audio",
                "data_tables",
                "save_workspace_artifact",
            )
        },
    )
    availability = project_research_actions(capture, probes)
    assert all(
        not availability[action].available
        for action in ("mind_map", "timeline", "slides", "audio_summary", "data_table")
    )


def test_action_requires_real_generator_native_owner_and_exact_context():
    capture = _capture()
    ready = {
        name: _ready(name)
        for name in ("generate_data_table", "get_data_table", "save_workspace_artifact")
    }
    without_owner = {
        key: value for key, value in ready.items() if key != "get_data_table"
    }
    assert not project_research_actions(
        capture, ResearchActionProbes(capture, without_owner)
    )["data_table"].available
    probes = ResearchActionProbes(capture, ready)
    assert project_research_actions(capture, probes)["data_table"].available
    for changed in (
        replace(capture, capability_revision="next-revision"),
        replace(
            capture,
            processing_route=replace(capture.processing_route, model="another-model"),
        ),
        replace(
            capture,
            source_snapshot=(
                replace(capture.source_snapshot[0], association_version=3),
            ),
        ),
    ):
        projected = project_research_actions(changed, probes)["data_table"]
        assert (
            not projected.available
            and projected.reason_code == "studio_capability_stale"
        )


def test_unimplemented_local_owner_and_unready_selected_source_have_recovery():
    capture = _capture("local")
    projected = project_research_actions(capture, ResearchActionProbes(capture, {}))[
        "slides"
    ]
    assert not projected.available and projected.owner and projected.recovery_action
    server = _capture()
    missing = replace(
        server,
        source_snapshot=(
            replace(server.source_snapshot[0], catalog_item_id="unsaved"),
        ),
    )
    ready = {
        name: _ready(name)
        for name in ("generate_data_table", "get_data_table", "save_workspace_artifact")
    }
    action = project_research_actions(missing, ResearchActionProbes(missing, ready))[
        "data_table"
    ]
    assert (
        not action.available and action.reason_code == "studio_source_owner_unavailable"
    )
