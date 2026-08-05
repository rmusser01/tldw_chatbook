from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_chatbook.UI.Workbench.workbench_state import (
    RecoveryState,
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchMode,
    WorkbenchState,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import (
    CommandStrip,
    WorkbenchActionRequested,
    WorkbenchFrame,
    _sort_state_children,
)


def _state(
    *,
    subtitle: str = "",
    action_label: str = "Settings",
    recovery_body: str = "Choose a provider to continue.",
) -> WorkbenchState:
    return WorkbenchState(
        header=WorkbenchHeaderState(
            title="Console",
            subtitle=subtitle,
            status="ready",
        ),
        modes=(
            WorkbenchMode(id="chat", label="Chat", active=True, status="ready"),
            WorkbenchMode(id="rag", label="RAG", status="empty"),
        ),
        actions=(
            WorkbenchAction(
                id="provider-recovery",
                label=action_label,
                tooltip="Open provider settings",
                primary=True,
            ),
        ),
        recovery=RecoveryState(
            title="Provider required",
            body=recovery_body,
            action=WorkbenchAction(
                id="provider-recovery",
                label=action_label,
                tooltip="Open provider settings",
                primary=True,
            ),
        ),
        route_id="console",
    )


def test_workbench_frame_sync_state_dispatches_one_snapshot_to_every_region():
    state = _state(
        subtitle="Provider setup needed",
        action_label="Choose model",
        recovery_body="Choose a model before running Search/RAG.",
    )
    collaborators = {
        "#workbench-header": SimpleNamespace(sync_state=Mock()),
        "#workbench-mode-strip": SimpleNamespace(sync_modes=Mock()),
        "#workbench-command-strip": SimpleNamespace(sync_actions=Mock()),
        "#workbench-recovery": SimpleNamespace(sync_state=Mock()),
        "#workbench-state-block": SimpleNamespace(sync_state=Mock()),
    }
    frame = SimpleNamespace(
        state=None,
        query_one=Mock(side_effect=lambda selector, *_args: collaborators[selector]),
        _sync_panes=Mock(),
        set_class=Mock(),
        add_class=Mock(),
        remove_class=Mock(),
        _route_class=None,
    )

    WorkbenchFrame.sync_state(frame, state)

    collaborators["#workbench-header"].sync_state.assert_called_once_with(state.header)
    collaborators["#workbench-mode-strip"].sync_modes.assert_called_once_with(
        state.modes
    )
    collaborators["#workbench-command-strip"].sync_actions.assert_called_once_with(
        state.actions
    )
    collaborators["#workbench-recovery"].sync_state.assert_called_once_with(
        state.recovery
    )
    collaborators["#workbench-state-block"].sync_state.assert_called_once_with(state)
    frame._sync_panes.assert_called_once_with(state.panes)
    frame.add_class.assert_called_once_with("route-console")
    assert frame._route_class == "route-console"


def test_recovery_action_posts_typed_workbench_request():
    strip = SimpleNamespace(post_message=Mock())
    button = SimpleNamespace(_workbench_action_id="provider-recovery")
    event = SimpleNamespace(button=button, stop=Mock())

    CommandStrip.on_workbench_button_pressed(strip, event)

    event.stop.assert_called_once_with()
    posted = strip.post_message.call_args.args[0]
    assert isinstance(posted, WorkbenchActionRequested)
    assert posted.action_id == "provider-recovery"


@pytest.mark.parametrize(
    ("attribute_name", "initial_ids", "desired_ids"),
    [
        ("_workbench_action_id", ["settings", "send"], ["send", "settings"]),
        ("_workbench_mode_id", ["chat", "rag"], ["rag", "chat"]),
        ("_workbench_pane_id", ["context", "transcript"], ["transcript", "context"]),
    ],
)
def test_state_child_sorting_matches_latest_snapshot(
    attribute_name,
    initial_ids,
    desired_ids,
):
    children = [
        SimpleNamespace(**{attribute_name: child_id}) for child_id in initial_ids
    ]
    widget = SimpleNamespace(children=children)

    def sort_children(*, key):
        widget.children.sort(key=key)

    widget.sort_children = sort_children

    _sort_state_children(
        widget,
        {child_id: index for index, child_id in enumerate(desired_ids)},
        attribute_name,
    )

    assert [
        getattr(child, attribute_name) for child in widget.children
    ] == desired_ids


def test_workbench_frame_direct_child_ids_are_stable_data_contract():
    frame = SimpleNamespace(
        children=[
            SimpleNamespace(id="workbench-header"),
            SimpleNamespace(id="workbench-mode-strip"),
            SimpleNamespace(id="workbench-command-strip"),
        ]
    )

    assert WorkbenchFrame.get_direct_child_ids(frame) == (
        "workbench-header",
        "workbench-mode-strip",
        "workbench-command-strip",
    )
