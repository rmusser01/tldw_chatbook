import importlib
import sys

from tldw_chatbook.UI.Workbench.route_inventory import (
    WORKBENCH_ROUTE_OWNERS,
    build_workbench_route_coverage,
)


def test_workbench_package_import_stays_side_effect_light():
    module_names = (
        "tldw_chatbook.UI.Workbench.route_inventory",
        "tldw_chatbook.UI.Workbench",
        "tldw_chatbook.UI.Navigation.screen_registry",
        "tldw_chatbook.UI.Navigation.shell_destinations",
    )
    original_modules = {
        module_name: sys.modules.pop(module_name, None) for module_name in module_names
    }

    ui_package = sys.modules.get("tldw_chatbook.UI")
    had_workbench_attribute = ui_package is not None and hasattr(ui_package, "Workbench")
    original_workbench_attribute = (
        getattr(ui_package, "Workbench") if had_workbench_attribute else None
    )
    if had_workbench_attribute:
        delattr(ui_package, "Workbench")

    try:
        importlib.import_module("tldw_chatbook.UI.Workbench")

        assert "tldw_chatbook.UI.Workbench.route_inventory" not in sys.modules
        assert "tldw_chatbook.UI.Navigation.screen_registry" not in sys.modules
        assert "tldw_chatbook.UI.Navigation.shell_destinations" not in sys.modules
    finally:
        for module_name, module in original_modules.items():
            sys.modules.pop(module_name, None)
            if module is not None:
                sys.modules[module_name] = module
        if ui_package is not None:
            if had_workbench_attribute:
                setattr(ui_package, "Workbench", original_workbench_attribute)
            elif hasattr(ui_package, "Workbench"):
                delattr(ui_package, "Workbench")


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

    # Notes is retired as a standalone screen -- "notes" is now purely a
    # compatibility alias that resolves to Library (see screen_registry's
    # _SCREEN_ALIASES), so it keeps its "library" owner rather than
    # disappearing from the route inventory.
    assert coverage.owner_for_route["notes"] == "library"
    assert coverage.owner_for_route["console"] == "console"
    assert coverage.owner_for_route["tools_settings"] == "mcp"
    assert coverage.owner_for_route["llm_management"] == "settings"
    assert coverage.owner_for_route["writing"] == "artifacts_writing"
    assert coverage.owner_for_route["evals"] == "diagnostics_evals"
    assert coverage.owner_for_route["stats"] == "diagnostics_stats"
    assert coverage.owner_for_route["logs"] == "diagnostics_logs"
