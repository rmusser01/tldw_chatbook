"""Finite [agents] workflow policy fields for canonical Console behavior drafts."""

from .settings_goal_policy import GOAL_SETTING_DEFAULTS, normalize_goal_setting

WORKFLOW_SETTING_FIELDS = (
    ("max_workflow_generations", "Planning and execution phase attempts", 8),
    ("max_workflow_child_launches", "Task launches", 12),
    ("max_workflow_model_calls", "Model calls including helpers", 32),
    ("max_workflow_budget_tokens", "Budget tokens", 500000),
    ("max_workflow_output_tokens", "Maximum output tokens per call", 8192),
    ("max_workflow_wall_seconds", "Cumulative active seconds", 900),
)
WORKFLOW_SETTING_DEFAULTS = {
    "workflow_runs_enabled": False,
    **{key: default for key, _, default in WORKFLOW_SETTING_FIELDS},
}
WORKFLOW_SETTING_KEYS = tuple(WORKFLOW_SETTING_DEFAULTS)
AGENT_SETTING_DEFAULTS = {**GOAL_SETTING_DEFAULTS, **WORKFLOW_SETTING_DEFAULTS}
AGENT_SETTING_KEYS = (*WORKFLOW_SETTING_KEYS, *GOAL_SETTING_DEFAULTS)


def normalize_agent_setting(key, value):
    if key in WORKFLOW_SETTING_DEFAULTS:
        return normalize_workflow_setting(key, value)
    if key in GOAL_SETTING_DEFAULTS:
        return normalize_goal_setting(key, value)
    raise ValueError("Unknown automatic-work setting.")


def compose_agent_policies(screen):
    from .settings_goal_policy import compose_goal_policy

    yield from compose_goal_policy(screen)
    yield from compose_workflow_policy(screen)


def sync_agent_policies(screen):
    from .settings_goal_policy import sync_goal_policy

    sync_goal_policy(screen)
    sync_workflow_policy(screen)


def normalize_workflow_setting(key, value):
    if key == "workflow_runs_enabled":
        if type(value) is not bool:
            raise ValueError("Workflow enablement must be on or off.")
        return value
    if (
        isinstance(value, bool)
        or not str(value).isdigit()
        or not 0 <= int(value) < 2**63
    ):
        raise ValueError(
            "Workflow limits must be finite nonnegative integers. Zero disables admission."
        )
    return int(value)


def compose_workflow_policy(screen):
    from textual.widgets import Input, Static

    from ...Widgets.state_checkbox import StateCheckbox

    yield Static("Workflow runs", classes="destination-section")
    yield StateCheckbox(
        "Enable workflow runs",
        value=bool(screen._console_behavior_value("workflow_runs_enabled")),
        id="settings-workflow-runs-enabled",
    )
    yield Static(
        "Finite limits; zero disables admission. Reductions apply to saved workflows. Drained review waits do not spend active time; Resume never refills the allowance.",
        markup=False,
    )
    for key, label, _default in WORKFLOW_SETTING_FIELDS:
        yield Static(label)
        yield Input(
            str(screen._console_behavior_value(key)),
            id="settings-" + key.replace("_", "-"),
            classes="settings-workflow-policy",
        )


def sync_workflow_policy(screen):
    from textual.css.query import NoMatches
    from textual.widgets import Checkbox, Input

    screen._syncing_workflow_policy = True
    try:
        for key in WORKFLOW_SETTING_KEYS:
            try:
                widget = screen.query_one("#settings-" + key.replace("_", "-"))
                with widget.prevent(Input.Changed, Checkbox.Changed):
                    value = screen._console_behavior_value(key)
                    widget.value = (
                        bool(value) if key == "workflow_runs_enabled" else str(value)
                    )
            except NoMatches:
                pass
    finally:
        screen._syncing_workflow_policy = False
