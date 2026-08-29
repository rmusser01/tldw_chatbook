"""Tests for the app-wide TLS trust policy (Utils/tls_trust.py) + config template."""
import tomllib

import tldw_chatbook.config as config_module


def test_default_config_template_has_network_ssl_verify():
    parsed = tomllib.loads(config_module.CONFIG_TOML_CONTENT)
    assert parsed["network"]["ssl_verify"] is True
