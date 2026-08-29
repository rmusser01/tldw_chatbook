"""Tests for the app-wide TLS trust policy (Utils/tls_trust.py) + config template."""
import tomllib

import tldw_chatbook.config as config_module


def test_default_config_template_has_network_ssl_verify():
    parsed = tomllib.loads(config_module.CONFIG_TOML_CONTENT)
    assert parsed["network"]["ssl_verify"] is True


import ssl as _ssl
from pathlib import Path

import pytest
from loguru import logger

import tldw_chatbook.Utils.tls_trust as tls_trust


@pytest.fixture(autouse=True)
def _clean_warn_state():
    tls_trust._warned_modes.clear()
    yield
    tls_trust._warned_modes.clear()


@pytest.fixture
def _set_ssl_config(monkeypatch):
    def _install(value):
        monkeypatch.setattr(
            tls_trust,
            "get_cli_setting",
            lambda section, key=None, default=None: (
                value if (section, key) == ("network", "ssl_verify") else default
            ),
        )

    return _install


@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("1", True),
        ("ON", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("OFF", False),
        ("", True),
        ("   ", True),
        (5, True),          # unsupported type -> fail safe
        (None, True),       # unsupported type -> fail safe
        (["x"], True),      # unsupported type -> fail safe
    ],
)
def test_tls_verify_setting_coercion(_set_ssl_config, raw, expected):
    _set_ssl_config(raw)
    assert tls_trust.tls_verify_setting() is expected


def test_tls_verify_setting_existing_path_string(tmp_path, _set_ssl_config):
    ca = tmp_path / "corp.pem"
    ca.write_text("# corp")
    _set_ssl_config(str(ca))
    assert tls_trust.tls_verify_setting() == str(ca)


def test_tls_verify_setting_missing_path_fails_safe(tmp_path, _set_ssl_config):
    _set_ssl_config(str(tmp_path / "missing.pem"))
    assert tls_trust.tls_verify_setting() is True


def test_tls_verify_setting_missing_path_logs_error(tmp_path, _set_ssl_config, capsys):
    _set_ssl_config(str(tmp_path / "missing.pem"))
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="ERROR")
    try:
        tls_trust.tls_verify_setting()
    finally:
        logger.remove(sink_id)
    assert any("ssl_verify" in m and "existing file" in m for m in messages)


def test_warn_tls_policy_once_per_mode(_set_ssl_config):
    _set_ssl_config(False)
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        tls_trust.warn_tls_policy()
        tls_trust.warn_tls_policy()
    finally:
        logger.remove(sink_id)
    warnings = [m for m in messages if "DISABLED" in m]
    assert len(warnings) == 1
    assert "API keys" in warnings[0]
