from tldw_chatbook.UI.Workbench.route_inventory import (
    WORKBENCH_ROUTE_OWNERS,
    build_workbench_route_coverage,
)


def test_all_registered_screen_routes_have_workbench_migration_owner():
    coverage = build_workbench_route_coverage()

    assert coverage.missing_owner_routes == ()
    assert "chat" in coverage.screen_routes
    assert WORKBENCH_ROUTE_OWNERS["chat"] == "console"


def test_shell_legacy_aliases_have_workbench_migration_owner():
    coverage = build_workbench_route_coverage()

    for alias in (
        "conversations_characters_prompts",
        "characters",
        "prompts",
        "subscription",
    ):
        assert alias in coverage.all_known_routes
        assert alias not in coverage.missing_owner_routes


def test_route_coverage_exposes_future_destination_owners():
    coverage = build_workbench_route_coverage()

    assert coverage.owner_for_route["notes"] == "library"
    assert coverage.owner_for_route["console"] == "console"
    assert coverage.owner_for_route["tools_settings"] == "mcp"
    assert coverage.owner_for_route["llm_management"] == "settings"
    assert coverage.owner_for_route["writing"] == "artifacts_writing"
    assert coverage.owner_for_route["evals"] == "diagnostics_evals"
    assert coverage.owner_for_route["stats"] == "diagnostics_stats"
    assert coverage.owner_for_route["logs"] == "diagnostics_logs"
