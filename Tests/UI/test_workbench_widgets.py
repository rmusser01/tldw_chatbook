from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.widgets import Button

from tldw_chatbook.UI.Workbench.workbench_state import (
    RecoveryState,
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchMode,
    WorkbenchState,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import (
    CommandStrip,
    DestinationHeader,
    ModeStrip,
    RecoveryCallout,
    WorkbenchActionRequested,
    WorkbenchFrame,
    _schedule_sort_state_children,
    _sort_state_children,
    _UNSYNCED,
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


# ---------------------------------------------------------------------------
# task-15452: sorting and pushing are both skipped when nothing moved
# ---------------------------------------------------------------------------


def _ordered_children(attribute_name, ids):
    return [SimpleNamespace(**{attribute_name: child_id}) for child_id in ids]


def test_state_child_sorting_is_skipped_when_the_order_already_matches():
    """`sort_children` is never free: it bumps the DOM version regardless.

    `NodeList._sort` calls `NodeList.updated`, which increments the update
    counter on this widget AND every ancestor up to the screen -- and that
    counter is part of the `query_one` LRU cache key. A no-op sort therefore
    evicts every cached `#id` lookup on the screen for nothing.
    """
    widget = SimpleNamespace(
        children=_ordered_children("_workbench_action_id", ["send", "settings"]),
        sort_children=Mock(),
    )

    _sort_state_children(
        widget,
        {"send": 0, "settings": 1},
        "_workbench_action_id",
    )

    widget.sort_children.assert_not_called()


def test_scheduling_a_state_child_sort_is_skipped_when_the_order_matches():
    """Not even the `call_next` message is worth posting for a no-op sort."""
    widget = SimpleNamespace(
        children=_ordered_children("_workbench_mode_id", ["chat", "rag"]),
        call_next=Mock(),
    )

    _schedule_sort_state_children(
        widget,
        {"chat": 0, "rag": 1},
        "_workbench_mode_id",
    )

    widget.call_next.assert_not_called()


def test_scheduling_a_state_child_sort_still_happens_when_the_order_differs():
    widget = SimpleNamespace(
        children=_ordered_children("_workbench_mode_id", ["rag", "chat"]),
        call_next=Mock(),
    )

    _schedule_sort_state_children(
        widget,
        {"chat": 0, "rag": 1},
        "_workbench_mode_id",
    )

    widget.call_next.assert_called_once()


def test_a_child_queued_for_removal_never_hides_a_real_reorder():
    """Children pending removal are still in `children` at schedule time.

    They key to `len(desired_order)` -- so they can only ever ADD an
    inversion, never mask one, and the conservative verdict is a schedule.
    """
    widget = SimpleNamespace(
        children=_ordered_children(
            "_workbench_action_id", ["stale", "settings", "send"]
        ),
        call_next=Mock(),
    )

    _schedule_sort_state_children(
        widget,
        {"send": 0, "settings": 1},
        "_workbench_action_id",
    )

    widget.call_next.assert_called_once()


def test_destination_header_skips_a_state_it_has_already_pushed():
    state = WorkbenchHeaderState(title="Console", subtitle="Ready", status="ready")
    header = SimpleNamespace(
        state=None,
        _synced_state=state,
        query_one=Mock(),
        set_class=Mock(),
    )

    DestinationHeader.sync_state(header, state)

    header.query_one.assert_not_called()
    header.set_class.assert_not_called()
    # `self.state` is still adopted, so identity semantics are unchanged.
    assert header.state is state


def test_destination_header_first_sync_runs_even_for_the_constructor_state():
    """The `on_mount` trap: `self.state` already equals the synced state.

    Comparing against `self.state` instead of a dedicated sentinel would
    turn every widget's mount-time self-sync into a no-op and leave the
    status/density classes -- which `compose` never sets -- unapplied.
    """
    state = WorkbenchHeaderState(title="Console", subtitle="Ready", status="running")
    header = SimpleNamespace(
        state=state,
        _synced_state=_UNSYNCED,
        query_one=Mock(return_value=Mock()),
        set_class=Mock(),
    )

    DestinationHeader.sync_state(header, state)

    assert header.query_one.call_count == 3
    assert header.set_class.call_count > 0
    assert header._synced_state == state


def test_mode_strip_skips_modes_it_has_already_pushed():
    modes = (WorkbenchMode(id="chat", label="Chat", active=True),)
    strip = SimpleNamespace(
        modes=(),
        _synced_modes=modes,
        children=[],
        mount=Mock(),
        call_next=Mock(),
    )

    ModeStrip.sync_modes(strip, modes)

    strip.mount.assert_not_called()
    strip.call_next.assert_not_called()
    assert strip.modes == modes


def test_command_strip_skips_actions_it_has_already_pushed():
    actions = (WorkbenchAction(id="send", label="Send", disabled=True),)
    strip = SimpleNamespace(
        actions=(),
        _synced_actions=actions,
        children=[],
        mount=Mock(),
        call_next=Mock(),
        _button_ids_by_action_id={"send": "workbench-action-send"},
    )

    CommandStrip.sync_actions(strip, actions)

    strip.mount.assert_not_called()
    strip.call_next.assert_not_called()
    assert strip.actions == actions
    assert strip._button_ids_by_action_id == {"send": "workbench-action-send"}


def test_command_strip_still_pushes_a_changed_action():
    """A one-attribute change (Send flipping enabled) must still land."""
    synced = (WorkbenchAction(id="send", label="Send", disabled=True),)
    changed = (WorkbenchAction(id="send", label="Send", disabled=False, primary=True),)
    button = Button("Send", id="workbench-action-send")
    setattr(button, "_workbench_action_id", "send")
    strip = SimpleNamespace(
        actions=synced,
        _synced_actions=synced,
        children=[button],
        mount=Mock(),
        call_next=Mock(),
        _button_ids_by_action_id={},
        _button_id=CommandStrip._button_id,
        _sync_button=Mock(),
    )

    CommandStrip.sync_actions(strip, changed)

    strip._sync_button.assert_called_once_with(button, changed[0])
    assert strip._synced_actions == changed


def test_recovery_callout_skips_a_state_it_has_already_pushed():
    state = RecoveryState(title="Provider required", body="Choose a provider.")
    callout = SimpleNamespace(
        state=None,
        _synced_state=state,
        query_one=Mock(),
        set_class=Mock(),
    )

    RecoveryCallout.sync_state(callout, state)

    callout.query_one.assert_not_called()
    callout.set_class.assert_not_called()
    assert callout.state is state


def test_recovery_callout_first_sync_of_none_still_hides_the_callout():
    """`None` is a real recovery state, hence the dedicated sentinel."""
    callout = SimpleNamespace(
        state=None,
        _synced_state=_UNSYNCED,
        _plain_text="stale",
        display=True,
        query_one=Mock(return_value=Mock()),
        set_class=Mock(),
    )

    RecoveryCallout.sync_state(callout, None)

    assert callout.display is False
    assert callout._plain_text == ""
    callout.set_class.assert_any_call(True, "is-hidden")
    assert callout._synced_state is None


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
