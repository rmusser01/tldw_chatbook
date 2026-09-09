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
