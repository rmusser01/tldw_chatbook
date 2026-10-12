"""Finite [agents] goal policy fields for canonical Console behavior drafts."""

GOAL_SETTING_FIELDS = (
    ("max_goal_generations", "Iterations", 3),
    ("max_goal_model_calls", "Model calls including helpers", 32),
    ("max_goal_budget_tokens", "Budget tokens", 500000),
    ("max_goal_output_tokens", "Maximum output tokens per call", 8192),
    ("max_goal_wall_seconds", "Elapsed seconds including waits", 900),
)
GOAL_SETTING_DEFAULTS = {
    "goal_runs_enabled": False,
    **{key: default for key, _, default in GOAL_SETTING_FIELDS},
}
GOAL_SETTING_KEYS = tuple(GOAL_SETTING_DEFAULTS)


def normalize_goal_setting(key, value):
    if key == "goal_runs_enabled":
        if type(value) is not bool:
            raise ValueError("Goal enablement must be on or off.")
        return value
    if (
        isinstance(value, bool)
        or not str(value).isdigit()
        or not 0 <= int(value) < 2**63
    ):
        raise ValueError(
            "Goal limits must be finite nonnegative integers. Zero disables admission."
        )
    return int(value)


def compose_goal_policy(screen):
    from textual.widgets import Checkbox, Input, Static

    yield Static("Goal runs", classes="destination-section")
    yield Checkbox(
        "Enable goal runs",
        value=bool(screen._console_behavior_value("goal_runs_enabled")),
        id="settings-goal-runs-enabled",
    )
    yield Static(
        "Finite limits; zero disables admission. Reductions apply to saved goals. Resume never refills the launch allowance.",
        markup=False,
    )
    for key, label, _default in GOAL_SETTING_FIELDS:
        yield Static(label)
        yield Input(
            str(screen._console_behavior_value(key)),
            id="settings-" + key.replace("_", "-"),
            classes="settings-goal-policy",
        )


def sync_goal_policy(screen):
    from textual.css.query import NoMatches
    from textual.widgets import Checkbox, Input

    screen._syncing_goal_policy = True
    try:
        for key in GOAL_SETTING_KEYS:
            try:
                widget = screen.query_one("#settings-" + key.replace("_", "-"))
                with widget.prevent(Input.Changed, Checkbox.Changed):
                    value = screen._console_behavior_value(key)
                    widget.value = (
                        bool(value) if key == "goal_runs_enabled" else str(value)
                    )
            except NoMatches:
                pass
    finally:
        screen._syncing_goal_policy = False
