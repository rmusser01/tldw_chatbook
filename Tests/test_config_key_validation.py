"""TASK-26039: advisory unknown/deprecated config key validation."""

from __future__ import annotations

from tldw_chatbook import config as config_module
from tldw_chatbook.config import validate_config_keys, format_config_key_report


def _kinds(findings):
    return {(f.path, f.kind): f.suggestion for f in findings}


def test_unknown_top_level_key_is_reported():
    """AC#1."""
    findings = validate_config_keys({"totally_made_up_section": {"x": 1}})
    paths = {f.path for f in findings if f.kind == "unknown"}
    assert "totally_made_up_section" in paths


def test_unknown_nested_key_reported_with_dotted_path():
    """AC#5: nested tables are covered, not just top-level."""
    findings = validate_config_keys({"general": {"nonexistent_general_key": 1}})
    paths = {f.path for f in findings if f.kind == "unknown"}
    assert "general.nonexistent_general_key" in paths


def test_near_miss_suggests_intended_key():
    """AC#2."""
    findings = validate_config_keys({"general": {"focus_mdoe": True}})
    hit = [f for f in findings if f.path == "general.focus_mdoe"]
    assert hit and hit[0].suggestion == "general.focus_mode"


def test_deprecated_section_reports_replacement():
    """AC#3: a known-renamed section is deprecated, naming its replacement,
    not merely reported as unknown."""
    findings = validate_config_keys({"API": {"openai_api_key": "sk-x"}})
    dep = [f for f in findings if f.kind == "deprecated" and f.path.startswith("API")]
    assert dep, "legacy [API] must be flagged deprecated, not unknown"
    assert "api_settings" in (dep[0].suggestion or "")
    # and it must NOT also show up as an unknown finding
    assert not [f for f in findings if f.kind == "unknown" and f.path.startswith("API")]


def test_freeform_sections_are_exempt():
    """AC#6: user-extensible sections do not produce unknown-key noise."""
    cfg = {
        "api_settings": {"my_custom_local_provider": {"api_key": "x", "weird_key": 1}},
        "model_capabilities": {"models": {"some-brand-new-model": {"context": 1}}},
    }
    findings = validate_config_keys(cfg)
    assert findings == [], f"free-form sections must be silent, got {findings}"


def test_valid_config_has_no_findings():
    findings = validate_config_keys(
        {"general": {"focus_mode": True}, "chat": {}}
    )
    assert findings == []


def test_report_formatting_is_human_readable():
    findings = validate_config_keys(
        {"general": {"focus_mdoe": True}, "API": {"openai_api_key": "x"}}
    )
    report = format_config_key_report(findings)
    assert "focus_mode" in report  # the suggestion is shown
    assert "api_settings" in report  # the deprecation replacement is shown


def test_adapter_surfaces_findings_but_stays_valid(tmp_path):
    """AC#1 + AC#4: unknown/deprecated keys are reported to the user via the
    Settings diagnostics validator, but never make the config invalid."""
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

    target = tmp_path / "config.toml"
    target.write_text(
        "[general]\nfocus_mdoe = true\n"      # typo -> should suggest focus_mode
        "[API]\nopenai_api_key = \"sk-x\"\n"  # deprecated section
    )
    result = SettingsConfigAdapter().validate_config_file(target)
    assert result.valid is True, "advisory findings must never block startup"
    assert "focus_mode" in result.message
    assert "api_settings" in result.message


def test_qodo13_documented_agents_overrides_are_not_flagged():
    """Qodo #13 (PR #2301): [agents] keys are deliberately absent from the
    parsed default shape (their defaults live in Agents/agent_service.py to
    avoid two-homes drift), so documented overrides must not warn."""
    findings = validate_config_keys(
        {"agents": {"child_max_wall_seconds": 300.0, "max_live_subagents": 2}}
    )
    assert findings == [], findings
