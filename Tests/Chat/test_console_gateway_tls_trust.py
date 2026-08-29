"""The Console gateway's owned HTTP client honors the app TLS trust policy."""
import ssl

import pytest

import tldw_chatbook.Utils.tls_trust as tls_trust
from tldw_chatbook.Chat import console_provider_gateway as gateway_mod


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
    ("config_value", "expected_mode"),
    [(False, ssl.CERT_NONE), (True, ssl.CERT_REQUIRED)],
)
@pytest.mark.owned_http_client
def test_gateway_client_honors_tls_policy(_set_ssl_config, config_value, expected_mode):
    _set_ssl_config(config_value)
    client = gateway_mod.ConsoleProviderGateway._new_owned_http_client()
    ctx = client._transport._pool._ssl_context
    assert ctx.verify_mode == expected_mode
