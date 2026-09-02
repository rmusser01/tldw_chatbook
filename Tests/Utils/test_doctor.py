"""TASK-25906: aggregate health-check (doctor) surface."""

from __future__ import annotations

from tldw_chatbook.Utils.doctor import (
    DoctorCheck,
    check_config_load,
    check_optional_dependencies,
    check_database_integrity,
    check_provider_readiness,
    run_doctor,
    format_doctor_report,
)


def test_config_load_pass_and_fail():
    """AC#1/#4."""
    ok = check_config_load(load_failure=None)
    assert ok.status == "pass"

    class _F:
        path = "/x/config.toml"
        message = "line 3: bad"
    bad = check_config_load(load_failure=_F())
    assert bad.status == "fail"
    assert "config.toml" in bad.detail
    assert bad.remediation  # names a remediation


def test_provider_readiness_reports_names_never_keys():
    """AC#5: configured/not-configured, never the key value."""
    check = check_provider_readiness(providers=["openai", "anthropic"])
    assert check.status in ("pass", "warn")
    assert "openai" in check.detail and "anthropic" in check.detail
    # a secret-shaped value must never appear
    assert "sk-" not in check.detail

    none = check_provider_readiness(providers=[])
    assert none.status == "warn"


def test_optional_dependencies_summary():
    check = check_optional_dependencies(available={"pdf": True, "ebook": False})
    assert check.status in ("pass", "warn")
    assert "ebook" in check.detail  # the missing one is named


def test_database_integrity_pass_and_fail():
    ok = check_database_integrity(integrity_fn=lambda: True)
    assert ok.status == "pass"
    bad = check_database_integrity(integrity_fn=lambda: False)
    assert bad.status == "fail"
    assert bad.remediation
    # a raising integrity check is reported, not crashed
    def _boom():
        raise RuntimeError("db locked")
    err = check_database_integrity(integrity_fn=_boom)
    assert err.status == "fail"
    assert "db locked" in err.detail


def test_run_doctor_skips_network_by_default():
    """AC#3: network checks opt-in."""
    checks = run_doctor(include_network=False)
    assert all(isinstance(c, DoctorCheck) for c in checks)
    # at minimum the required check names are present
    names = {c.name for c in checks}
    assert "config" in names
    assert "optional-dependencies" in names
    assert "database" in names
    assert "providers" in names
    assert "private-paths" in names


def test_run_doctor_still_runs_when_config_failed(monkeypatch):
    """AC#6: reachable/usable without a working config."""
    from tldw_chatbook import config as cfg

    class _F:
        path = "/x/config.toml"
        message = "broken"
    monkeypatch.setattr(cfg, "get_config_load_failure", lambda: _F())

    checks = run_doctor(include_network=False)
    by_name = {c.name: c for c in checks}
    assert by_name["config"].status == "fail"
    # the other checks still ran despite the config failure
    assert len(checks) >= 5


def test_report_has_no_secrets_and_states_each_check():
    checks = [
        DoctorCheck(name="config", status="pass", detail="loaded"),
        DoctorCheck(name="providers", status="warn", detail="none configured", remediation="add a key"),
    ]
    report = format_doctor_report(checks)
    assert "config" in report and "providers" in report
    assert "pass" in report.lower() and "warn" in report.lower()
    assert "add a key" in report
