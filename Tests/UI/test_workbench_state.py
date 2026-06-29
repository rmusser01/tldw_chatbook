import importlib
import sys

import pytest

from tldw_chatbook.UI.Workbench.workbench_state import (
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchMode,
    WorkbenchPaneState,
    WorkbenchState,
    normalize_workbench_id,
)


def test_workbench_package_import_does_not_load_widget_or_navigation_modules():
    module_names = (
        "tldw_chatbook.UI.Workbench",
        "tldw_chatbook.UI.Workbench.route_inventory",
        "tldw_chatbook.UI.Workbench.workbench_widgets",
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
        assert "tldw_chatbook.UI.Workbench.workbench_widgets" not in sys.modules
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


def test_workbench_state_rejects_duplicate_action_ids():
    with pytest.raises(ValueError, match="duplicate action id"):
        WorkbenchState(
            header=WorkbenchHeaderState(title="Console"),
            actions=(
                WorkbenchAction(id="run", label="Run"),
                WorkbenchAction(id="run", label="Run again"),
            ),
        )


def test_normalize_workbench_id_matches_widget_id_contract():
    assert normalize_workbench_id("run now") == "run-now"
    assert normalize_workbench_id(" run-now ") == "run-now"
    assert normalize_workbench_id("!!!") == "item"


def test_workbench_state_rejects_non_canonical_ids_before_widgets_mount():
    with pytest.raises(ValueError, match="non-canonical action id"):
        WorkbenchAction(id="run now", label="Run")

    with pytest.raises(ValueError, match="non-canonical mode id"):
        WorkbenchMode(id="rag mode", label="RAG")

    with pytest.raises(ValueError, match="non-canonical pane id"):
        WorkbenchPaneState(id="right rail", title="Inspector")


def test_workbench_state_rejects_duplicate_mode_and_pane_ids():
    with pytest.raises(ValueError, match="duplicate mode id"):
        WorkbenchState(
            header=WorkbenchHeaderState(title="Console"),
            modes=(
                WorkbenchMode(id="chat", label="Chat"),
                WorkbenchMode(id="chat", label="Chat again"),
            ),
        )

    with pytest.raises(ValueError, match="duplicate pane id"):
        WorkbenchState(
            header=WorkbenchHeaderState(title="Console"),
            panes=(
                WorkbenchPaneState(id="context", title="Context"),
                WorkbenchPaneState(id="context", title="Context again"),
            ),
        )


def test_workbench_mode_css_classes_include_active_status():
    mode = WorkbenchMode(id="rag", label="RAG", active=True, status="ready")

    assert mode.css_classes == "workbench-mode is-active status-ready"
