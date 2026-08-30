"""WsTransport passes the app TLS policy to websockets.connect for wss:// URLs."""
import ssl
import types

import pytest

import tldw_chatbook.Utils.tls_trust as tls_trust
from tldw_chatbook.LLM_Calls.realtime import transport as transport_mod


class _FakeWebsockets(types.SimpleNamespace):
    def __init__(self):
        captured = {}
        self.captured = captured

        async def connect(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return object()

        self.connect = connect


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
    ("url", "config_value", "ssl_expected"),
    [
        ("wss://example.invalid/rt", False, "unverified"),
        ("ws://example.invalid/rt", False, None),  # never passes ssl for ws://
        ("wss://example.invalid/rt", True, None),  # default policy -> no ssl kwarg
    ],
)
async def test_transport_passes_tls_policy(_set_ssl_config, url, config_value, ssl_expected):
    _set_ssl_config(config_value)
    fake = _FakeWebsockets()
    t = transport_mod.WsTransport()
    t._ws = None
    orig = transport_mod._websockets
    transport_mod._websockets = lambda: fake
    try:
        await t.connect(url, headers={})
    finally:
        transport_mod._websockets = orig
    kwargs = fake.captured["kwargs"]
    if ssl_expected == "unverified":
        ctx = kwargs["ssl"]
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.check_hostname is False
        assert ctx.verify_mode == ssl.CERT_NONE
    else:
        assert "ssl" not in kwargs
