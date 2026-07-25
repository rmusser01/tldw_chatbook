"""task-582: ScriptRunLimits must be overridable from the [skills] config.

The sandbox budget was a set of hardcoded defaults, so a legitimately
long-running or output-heavy skill script had no accommodation short of a code
change, and a user wanting a *tighter* budget could not impose one.

The trap these tests guard: `get_cli_setting("skills", {})` silently returns
`{}` for any section name without a dot (config.py), so the section-dict form
would make every knob permanently unreachable. Only the three-argument form
works.
"""

import pytest

from tldw_chatbook.Skills_Interop.skill_script_runner import ScriptRunLimits
from tldw_chatbook.Skills_Interop import local_skills_service as svc_module


@pytest.fixture
def config(monkeypatch):
    """Serve a fake [skills] table through the real 3-arg accessor shape."""
    values = {}

    def fake_get_cli_setting(section, key=None, default=None):
        # A caller using the section-dict form passes key as a dict/None; that
        # form cannot reach these values, so return the default and let the
        # assertion fail loudly rather than silently succeeding.
        if section != "skills" or not isinstance(key, str):
            return default
        return values.get(key, default)

    import tldw_chatbook.config as config_module

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    return values


def test_defaults_apply_when_nothing_is_configured(config):
    limits = svc_module.resolve_script_run_limits()
    assert limits == ScriptRunLimits()


def test_every_field_can_be_overridden(config):
    config.update(
        {
            "script_cpu_seconds": 3,
            "script_address_space_bytes": 128 * 1024 * 1024,
            "script_open_files": 32,
            "script_file_size_bytes": 1024 * 1024,
            "script_wall_clock_seconds": 12.5,
            "script_output_cap_bytes": 4096,
        }
    )
    limits = svc_module.resolve_script_run_limits()
    assert limits.cpu_seconds == 3
    assert limits.address_space_bytes == 128 * 1024 * 1024
    assert limits.open_files == 32
    assert limits.file_size_bytes == 1024 * 1024
    assert limits.wall_clock_seconds == 12.5
    assert limits.output_cap_bytes == 4096


def test_partial_override_keeps_other_defaults(config):
    config["script_cpu_seconds"] = 5
    limits = svc_module.resolve_script_run_limits()
    assert limits.cpu_seconds == 5
    assert limits.output_cap_bytes == ScriptRunLimits().output_cap_bytes


@pytest.mark.parametrize(
    "value",
    [0, -1, "banana", None, 3.5e400],
)
def test_invalid_values_fall_back_to_the_default(config, value):
    """AC#3: never let a bad value produce an unbounded or zero budget."""
    config["script_cpu_seconds"] = value
    limits = svc_module.resolve_script_run_limits()
    assert limits.cpu_seconds == ScriptRunLimits().cpu_seconds


def test_wall_clock_is_capped_to_the_run_budget_envelope(config):
    """AC#4: an over-large wall clock must not strand the agent run."""
    config["script_wall_clock_seconds"] = 100_000.0
    limits = svc_module.resolve_script_run_limits()
    assert limits.wall_clock_seconds <= svc_module.MAX_SCRIPT_WALL_CLOCK_SECONDS
    assert limits.wall_clock_seconds > 0


def test_section_dict_form_would_not_reach_these_values(monkeypatch):
    """AC#2: pin the reachability trap itself.

    If the implementation ever switches to `get_cli_setting("skills", {})`,
    this fake — which mirrors config.py's real behaviour of returning the
    default for a non-str key — makes it silently see nothing, so the
    override assertion below fails.
    """
    import tldw_chatbook.config as config_module

    calls = []

    def fake_get_cli_setting(section, key=None, default=None):
        calls.append((section, key))
        if section != "skills" or not isinstance(key, str):
            return default
        return 7 if key == "script_cpu_seconds" else default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)
    limits = svc_module.resolve_script_run_limits()
    assert limits.cpu_seconds == 7, (
        "the resolver must use the 3-arg get_cli_setting form; the "
        "section-dict form cannot reach [skills] values at all"
    )
    assert any(isinstance(key, str) for _section, key in calls)
