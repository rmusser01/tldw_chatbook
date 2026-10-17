from __future__ import annotations

import base64
import hashlib
import json
import re
import struct
import sys
from contextlib import asynccontextmanager
from http.cookies import SimpleCookie
from urllib.parse import urlsplit

import pytest
from aiohttp.test_utils import TestClient, TestServer
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from fido2 import cbor

from tldw_chatbook.Canvas.web_auth import (
    CSRF_HEADER_NAME,
    SESSION_COOKIE_NAME,
    WEBSOCKET_PROTOCOL,
    build_web_auth_policy,
)
from tldw_chatbook.config import build_canvas_config_policy
from tldw_chatbook.Web_Server import serve
from tldw_chatbook.Web_Server.session_store import WebSessionStore
from tldw_chatbook.Web_Server.webauthn_service import WebAuthnService
from tldw_chatbook.Web_Server.webauthn_store import WebAuthnCredentialStore

pytestmark = [
    pytest.mark.skipif(
        not serve.check_web_server_available(),
        reason="web server optional dependencies are unavailable",
    ),
    pytest.mark.loopback_network,
]


class _FakeBaseServer:
    def __init__(
        self,
        command: str,
        host: str,
        port: int,
        title: str,
        *,
        public_url: str | None = None,
        statics_path: str,
        templates_path: str,
    ) -> None:
        self.command = command
        self.host = host
        self.port = port
        self.title = title
        self.public_url = public_url or f"http://{host}:{port}"
        self.statics_path = statics_path
        self.templates_path = templates_path
        self.download_manager = object()
        self.debug = False

    async def on_startup(self, _app) -> None:
        return None

    async def on_shutdown(self, _app) -> None:
        return None

    async def handle_download(self, _request):
        from aiohttp import web

        return web.Response(text="download")


def _server(tmp_path, policy):
    templates = tmp_path / "templates"
    statics = tmp_path / "static"
    (statics / "js").mkdir(parents=True, exist_ok=True)
    templates.mkdir(exist_ok=True)
    (templates / "app_index.html").write_text(
        (
            '<html><head><script src="{{ config.static.url }}js/textual.js">'
            "</script></head><body>{{ application.name }}</body></html>"
        ),
        encoding="utf-8",
    )
    (statics / "js" / "textual.js").write_text("fixed-runtime", encoding="utf-8")
    cls = serve.build_chatbook_web_server_class(_FakeBaseServer)
    return cls(
        command="python -m tldw_chatbook.app",
        host=policy.bind_host,
        port=policy.port,
        title="Test",
        public_url=(f"{policy.external_scheme}://{next(iter(policy.allowed_hosts))}"),
        statics_path=str(statics),
        templates_path=str(templates),
        web_auth_policy=policy,
    )


async def _app_without_control_broker(server):
    app = await server._make_app()
    app.on_startup.clear()
    app.on_shutdown.clear()
    return app


@pytest.mark.asyncio
async def test_loopback_root_auto_issues_host_only_session_and_csrf_boot_data(
    tmp_path,
) -> None:
    policy = build_web_auth_policy(host="127.0.0.1", port=8000, access_token=None)
    server = _server(tmp_path, policy)
    async with TestClient(
        TestServer(await _app_without_control_broker(server))
    ) as client:
        response = await client.get("/", headers={"Host": "127.0.0.1:8000"})
        body = await response.text()

    assert response.status == 200
    cookie = SimpleCookie(response.headers["Set-Cookie"])[SESSION_COOKIE_NAME]
    assert cookie["httponly"] is True
    assert cookie["samesite"] == "Strict"
    assert cookie["domain"] == ""
    assert cookie["secure"] == ""
    assert 'meta name="chatbook-csrf"' in body
    assert body.index("chatbook-auth.js") < body.index("textual.js")


@pytest.mark.asyncio
async def test_loopback_auto_login_still_rejects_a_host_header_mismatch(
    tmp_path,
) -> None:
    policy = build_web_auth_policy(host="127.0.0.1", port=8000, access_token=None)
    server = _server(tmp_path, policy)
    async with TestClient(
        TestServer(await _app_without_control_broker(server))
    ) as client:
        response = await client.get(
            "/", headers={"Host": "attacker.example"}, allow_redirects=False
        )

    assert response.status == 401
    assert "Set-Cookie" not in response.headers


@pytest.mark.asyncio
async def test_cross_site_loopback_get_does_not_mint_a_session(tmp_path) -> None:
    policy = build_web_auth_policy(host="127.0.0.1", port=8000, access_token=None)
    server = _server(tmp_path, policy)
    async with TestClient(
        TestServer(await _app_without_control_broker(server))
    ) as client:
        response = await client.get(
            "/",
            headers={
                "Host": "127.0.0.1:8000",
                "Sec-Fetch-Site": "cross-site",
            },
            allow_redirects=False,
        )

    assert response.status == 303
    assert "Set-Cookie" not in response.headers


@pytest.mark.asyncio
async def test_loopback_root_replaces_a_stale_cookie_after_server_restart(
    tmp_path,
) -> None:
    policy = build_web_auth_policy(host="127.0.0.1", port=8000, access_token=None)
    first_server = _server(tmp_path, policy)
    async with TestClient(
        TestServer(await _app_without_control_broker(first_server))
    ) as client:
        first_response = await client.get("/", headers={"Host": "127.0.0.1:8000"})
        stale_cookie = SimpleCookie(first_response.headers["Set-Cookie"])[
            SESSION_COOKIE_NAME
        ].value

    restarted_server = _server(tmp_path, policy)
    async with TestClient(
        TestServer(await _app_without_control_broker(restarted_server))
    ) as client:
        response = await client.get(
            "/",
            headers={
                "Host": "127.0.0.1:8000",
                "Cookie": f"{SESSION_COOKIE_NAME}={stale_cookie}",
            },
            allow_redirects=False,
        )

    assert response.status == 200
    replacement = SimpleCookie(response.headers["Set-Cookie"])[SESSION_COOKIE_NAME]
    assert replacement.value != stale_cookie


@pytest.mark.asyncio
async def test_remote_origin_redirects_to_manual_login_then_sets_secure_cookie(
    tmp_path,
) -> None:
    policy = build_web_auth_policy(
        host="127.0.0.1",
        port=8000,
        access_token="remote-only-secret",
        public_url="https://chatbook.example",
        trusted_proxy_addresses=["127.0.0.1"],
    )
    server = _server(tmp_path, policy)
    async with TestClient(
        TestServer(await _app_without_control_broker(server))
    ) as client:
        denied = await client.get(
            "/", headers={"Host": "chatbook.example"}, allow_redirects=False
        )
        login = await client.post(
            "/auth/login",
            headers={
                "Host": "chatbook.example",
                "Origin": "https://chatbook.example",
                "X-Forwarded-For": "203.0.113.9",
                "X-Forwarded-Proto": "https",
                "X-Forwarded-Host": "chatbook.example",
            },
            data={"access_token": "remote-only-secret"},
            allow_redirects=False,
        )
        body = await login.text()

    assert denied.status == 303
    assert denied.headers["Location"] == "/auth/login"
    assert login.status == 303
    assert login.headers["Location"] == "/"
    cookie = SimpleCookie(login.headers["Set-Cookie"])[SESSION_COOKIE_NAME]
    assert cookie["httponly"] is True
    assert cookie["secure"] is True
    assert cookie["samesite"] == "Strict"
    assert "remote-only-secret" not in body
    assert "remote-only-secret" not in repr(server)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/", "/download/thing", "/ws"])
async def test_every_authority_route_rejects_an_unauthenticated_remote_browser(
    tmp_path, path
) -> None:
    policy = build_web_auth_policy(
        host="192.168.1.20",
        port=8000,
        access_token="dedicated-secret",
        public_url="http://chatbook.example",
        allow_insecure_remote_http=True,
    )
    server = _server(tmp_path, policy)
    async with TestClient(
        TestServer(await _app_without_control_broker(server))
    ) as client:
        response = await client.get(
            path,
            headers={
                "Host": "chatbook.example",
                "Origin": "http://chatbook.example" if path == "/ws" else "",
                "Upgrade": "websocket" if path == "/ws" else "",
                "Connection": "Upgrade" if path == "/ws" else "",
            },
            allow_redirects=False,
        )

    assert response.status in {303, 401}


@pytest.mark.asyncio
async def test_state_free_textual_runtime_is_public_but_not_dynamic_boot_data(
    tmp_path,
) -> None:
    policy = build_web_auth_policy(
        host="192.168.1.20",
        port=8000,
        access_token="dedicated-secret",
        public_url="http://chatbook.example",
        allow_insecure_remote_http=True,
    )
    server = _server(tmp_path, policy)
    async with TestClient(
        TestServer(await _app_without_control_broker(server))
    ) as client:
        response = await client.get(
            "/static/js/textual.js", headers={"Host": "chatbook.example"}
        )
        body = await response.text()

    assert response.status == 200
    assert body == "fixed-runtime"


@pytest.mark.asyncio
async def test_owned_websocket_bootstrap_uses_csrf_subprotocol_without_url_credentials(
    tmp_path,
) -> None:
    policy = build_web_auth_policy(host="127.0.0.1", port=8000, access_token=None)
    server = _server(tmp_path, policy)
    async with TestClient(
        TestServer(await _app_without_control_broker(server))
    ) as client:
        response = await client.get(
            "/static/js/chatbook-auth.js", headers={"Host": "127.0.0.1:8000"}
        )
        source = await response.text()

    assert response.status == 200
    assert f'"{WEBSOCKET_PROTOCOL}"' in source
    assert 'meta[name="chatbook-csrf"]' in source
    assert "new NativeWebSocket" in source
    assert "csrf=" not in source


def test_create_server_resolves_only_dedicated_web_credential(monkeypatch) -> None:
    captured = {}

    class _Base:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(serve, "_load_textual_serve_server_class", lambda: _Base)
    monkeypatch.setattr(serve, "_load_textual_serve_app_service_class", lambda: None)
    values = {
        "access_token": "web-secret",
        "public_url": "http://chatbook.example",
        "allow_insecure_remote_http": True,
        "trusted_proxy_addresses": [],
        "tls_certificate": "",
        "tls_private_key": "",
    }
    monkeypatch.setattr(
        serve,
        "get_cli_setting",
        lambda section, key=None, default=None: values.get(key, default),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("TLDW_API_KEY", "legacy-secret")
    monkeypatch.delenv("TLDW_CHATBOOK_WEB_ACCESS_TOKEN", raising=False)

    server = serve.create_server(host="0.0.0.0", port=8000)

    assert server is not None
    policy = server._web_auth.policy
    assert policy.access_credential.reveal() == "web-secret"
    assert server._canvas_policy.remote_access_status == "insecure_development"
    assert "provider-secret" not in repr(policy)
    assert "legacy-secret" not in repr(policy)


def test_create_server_degrades_to_token_only_login_when_fido2_missing(
    monkeypatch, tmp_path
) -> None:
    """A missing python-fido2 must degrade passkeys, not fail served boot.

    ``webauthn_service.py`` imports fido2 at module top and fido2 rides the
    optional [web] extra, so a stale install that upgraded code without
    reinstalling hits an ImportError inside create_server. Parking ``None``
    under the module name in ``sys.modules`` makes that exact import raise
    ImportError; the server must still build with ``webauthn_service None``
    (token-only login — the 503 route half is covered by
    ``test_passkey_routes_answer_503_when_passkeys_unavailable``).
    """

    class _Base:
        def __init__(self, **kwargs):
            pass  # kwargs sink; the mixin holds the asserted state

    monkeypatch.setattr(serve, "_load_textual_serve_server_class", lambda: _Base)
    monkeypatch.setattr(serve, "_load_textual_serve_app_service_class", lambda: None)
    values = {
        "access_token": "web-secret",
        "public_url": "http://chatbook.example",
        "allow_insecure_remote_http": True,
        "trusted_proxy_addresses": [],
        "tls_certificate": "",
        "tls_private_key": "",
    }
    monkeypatch.setattr(
        serve,
        "get_cli_setting",
        lambda section, key=None, default=None: values.get(key, default),
    )
    # Keep the construction hermetic: both of these read the shared config
    # cache (and the backup-recovery bootstrap behind it) on a real machine;
    # the device-manager helpers below dodge the same seam on purpose.
    monkeypatch.setattr(
        serve,
        "get_canvas_config_policy",
        lambda **_: build_canvas_config_policy({}, web_auth_policy=None),
    )
    monkeypatch.setattr(serve, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("TLDW_API_KEY", "legacy-secret")
    monkeypatch.delenv("TLDW_CHATBOOK_WEB_ACCESS_TOKEN", raising=False)
    # A None entry under a module name is the documented CPython signal for
    # "import of this module halted": the from-import in create_server then
    # raises ImportError without needing to uninstall anything.
    monkeypatch.setitem(
        sys.modules, "tldw_chatbook.Web_Server.webauthn_service", None
    )

    server = serve.create_server(host="0.0.0.0", port=8000)

    assert server is not None
    # The mixin's __init__ keeps the service on the instance (the textual
    # base never sees it), so presence/None is asserted there.
    assert server._webauthn_service is None


def test_create_server_names_the_offending_session_day_key(monkeypatch, tmp_path) -> None:
    """#3127 review minor: a bad day-key config value raises a ValueError
    naming the offending key, mirroring the absolute<idle message family —
    not a bare ``invalid literal for int()`` that hides which setting to fix.
    """

    class _Base:
        def __init__(self, **kwargs):
            pass  # never reached: both bad values raise before construction

    monkeypatch.setattr(serve, "_load_textual_serve_server_class", lambda: _Base)
    monkeypatch.setattr(serve, "_load_textual_serve_app_service_class", lambda: None)
    values = {
        "access_token": "web-secret",
        "public_url": "http://chatbook.example",
        "allow_insecure_remote_http": True,
        "trusted_proxy_addresses": [],
        "tls_certificate": "",
        "tls_private_key": "",
    }
    monkeypatch.setattr(
        serve,
        "get_cli_setting",
        lambda section, key=None, default=None: values.get(key, default),
    )
    monkeypatch.setattr(
        serve,
        "get_canvas_config_policy",
        lambda **_: build_canvas_config_policy({}, web_auth_policy=None),
    )
    monkeypatch.setattr(serve, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.delenv("TLDW_CHATBOOK_WEB_ACCESS_TOKEN", raising=False)

    values["session_idle_timeout_days"] = "soon"
    with pytest.raises(ValueError, match="web_server.session_idle_timeout_days"):
        serve.create_server(host="0.0.0.0", port=8000)

    values["session_idle_timeout_days"] = 30
    values["session_absolute_timeout_days"] = "soon"
    with pytest.raises(ValueError, match="web_server.session_absolute_timeout_days"):
        serve.create_server(host="0.0.0.0", port=8000)


# --- Device manager pages (task 4, ADR-233) -------------------------------
#
# These tests deliberately construct the server without touching the shared
# config cache: `canvas_policy` is injected explicitly (latched off, so
# `_canvas_enabled()` never falls through to a global config read) and every
# exercised route (`/auth/login` POST, `/auth/devices`, `/auth/logout`,
# `/auth/devices/revoke`, `/auth/devices/rename`) is config-free.

_DEVICES_ACCESS_TOKEN = "device-manager-secret"
# A plaintext http public origin (like the insecure-remote tests above) keeps
# the session cookie non-Secure, which the plain-http aiohttp test jar will
# actually replay; the policy is still non-loopback, so GET "/" never
# auto-logins and an unauthenticated root visit 303s to the login page.
_DEVICES_PUBLIC_ORIGIN = "http://chatbook.example"

#: Every request through the trusted-proxy device-manager policy must carry
#: the forwarded routing trio; POSTs additionally need the same-origin Origin
#: header (session default) plus the per-request CSRF header.
_DEVICE_REQUEST_HEADERS = {
    "Origin": _DEVICES_PUBLIC_ORIGIN,
    "X-Forwarded-For": "203.0.113.9",
    "X-Forwarded-Proto": "http",
    "X-Forwarded-Host": "chatbook.example",
}


def _device_manager_server(tmp_path, policy, *, webauthn_service=None):
    """Build a config-free served server whose sessions persist to tmp_path."""

    templates = tmp_path / "templates"
    statics = tmp_path / "static"
    (statics / "js").mkdir(parents=True, exist_ok=True)
    templates.mkdir(exist_ok=True)
    (templates / "app_index.html").write_text(
        (
            '<html><head><script src="{{ config.static.url }}js/textual.js">'
            "</script></head><body>{{ application.name }}</body></html>"
        ),
        encoding="utf-8",
    )
    (statics / "js" / "textual.js").write_text("fixed-runtime", encoding="utf-8")
    cls = serve.build_chatbook_web_server_class(_FakeBaseServer)
    return cls(
        command="python -m tldw_chatbook.app",
        host=policy.bind_host,
        port=policy.port,
        title="Test",
        public_url=f"{policy.external_scheme}://{next(iter(policy.allowed_hosts))}",
        statics_path=str(statics),
        templates_path=str(templates),
        web_auth_policy=policy,
        canvas_policy=build_canvas_config_policy(
            {}, environ={"TLDW_CANVAS_ENABLED": "false"}
        ),
        session_store=WebSessionStore(tmp_path / "web_sessions.json"),
        webauthn_service=webauthn_service,
    )


@asynccontextmanager
async def _logged_in_client(
    tmp_path, *, remember: str | None = "1", webauthn_service=None
):
    """Yield one token-authenticated TestClient against a fresh server.

    The client carries the forwarded routing headers by default and has
    posted the login form with ``remember`` (default ``"1"``); ``None``
    omits the field entirely — the unchecked-checkbox submission — so tests
    can assert on ``web_sessions.json``.
    """

    policy = build_web_auth_policy(
        host="127.0.0.1",
        port=8000,
        access_token=_DEVICES_ACCESS_TOKEN,
        public_url=_DEVICES_PUBLIC_ORIGIN,
        allow_insecure_remote_http=True,
        trusted_proxy_addresses=["127.0.0.1"],
    )
    server = _device_manager_server(
        tmp_path, policy, webauthn_service=webauthn_service
    )
    client = TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_DEVICE_REQUEST_HEADERS),
    )
    client.chatbook_test_server = server
    async with client:
        form = {"access_token": _DEVICES_ACCESS_TOKEN}
        if remember is not None:
            form["remember"] = remember
        login = await client.post(
            "/auth/login",
            data=form,
            allow_redirects=False,
        )
        assert login.status == 303
        assert login.headers["Location"] == "/"
        client.chatbook_login_response = login
        yield client


async def _fresh_client(server) -> TestClient:
    """Return a new, not-yet-logged-in client sharing the same server."""

    return TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_DEVICE_REQUEST_HEADERS),
    )


async def _device_entries(client) -> list[tuple[str, bool]]:
    """Return ``(session_id, is_current)`` pairs listed on the devices page."""

    page = await client.get("/auth/devices")
    assert page.status == 200
    return [
        (session_id, current == "true")
        for session_id, current in re.findall(
            r'<section class="device" data-session-id="([^"]+)"'
            r'[^>]*data-current-device="([^"]+)"',
            await page.text(),
        )
    ]


async def _current_session_id(client) -> str:
    """Return the devices-page marker for the requesting device's session."""

    entries = await _device_entries(client)
    current = [session_id for session_id, is_current in entries if is_current]
    assert len(current) == 1
    return current[0]


async def _csrf(client) -> str:
    """Read the per-session CSRF token from the devices page's meta tag."""

    page = await client.get("/auth/devices")
    assert page.status == 200
    match = re.search(
        r'<meta name="chatbook-csrf" content="([^"]*)">', await page.text()
    )
    assert match is not None
    return match.group(1)


@pytest.mark.asyncio
async def test_devices_page_lists_and_revokes(tmp_path) -> None:
    async with _logged_in_client(tmp_path, remember="1") as client:
        page = await client.get("/auth/devices")
        body = await page.text()

        assert page.status == 200
        assert "Sign out this device" in body
        assert "Unknown device" in body  # aiohttp UA matches no marker
        assert "/static/chatbook/devices.js" in body
        script = await client.get("/static/chatbook/devices.js")
        assert script.status == 200

        resp = await client.post(
            "/auth/devices/revoke",
            json={"session_id": await _current_session_id(client)},
            headers={CSRF_HEADER_NAME: await _csrf(client)},
        )
        assert resp.status == 200
        assert (await resp.json())["ok"] is True

        # Current session revoked: the next request bounces to the login page.
        after = await client.get("/", allow_redirects=False)
        assert after.status == 303
        assert after.headers["Location"] == "/auth/login"


@pytest.mark.asyncio
async def test_logout_clears_cookie(tmp_path) -> None:
    async with _logged_in_client(tmp_path) as client:
        resp = await client.post(
            "/auth/logout",
            headers={CSRF_HEADER_NAME: await _csrf(client)},
            allow_redirects=False,
        )

        assert resp.status == 303
        assert resp.headers["Location"] == "/auth/login"
        deleted = SimpleCookie(resp.headers.get("Set-Cookie", "")).get(
            SESSION_COOKIE_NAME
        )
        assert deleted is not None
        assert deleted.value == ""

        denied = await client.get("/auth/devices")
        assert denied.status == 401


@pytest.mark.asyncio
async def test_revoking_another_device_ends_only_that_session(tmp_path) -> None:
    async with _logged_in_client(tmp_path) as client:
        other = await _fresh_client(client.chatbook_test_server)
        async with other:
            login = await other.post(
                "/auth/login",
                data={"access_token": _DEVICES_ACCESS_TOKEN, "remember": "1"},
                allow_redirects=False,
            )
            assert login.status == 303

            entries = await _device_entries(client)
            assert len(entries) == 2
            assert sum(1 for _sid, is_current in entries if is_current) == 1
            other_session_id = next(
                session_id for session_id, is_current in entries if not is_current
            )

            resp = await client.post(
                "/auth/devices/revoke",
                json={"session_id": other_session_id},
                headers={CSRF_HEADER_NAME: await _csrf(client)},
            )
            assert resp.status == 200
            assert (await resp.json())["ok"] is True

            assert (await client.get("/auth/devices")).status == 200
            assert (await other.get("/auth/devices")).status == 401


@pytest.mark.asyncio
async def test_revoke_without_the_csrf_header_is_denied_and_revokes_nothing(
    tmp_path,
) -> None:
    """CSRF-denial pin (#3127 review minor).

    A valid session POSTing the revoke route WITHOUT the per-request CSRF
    header must be refused with 401 before the service is touched: no
    session is revoked, both devices keep working. A pin, not a change —
    the middleware already enforces the header on every state-changing
    route; this locks the contract at the route boundary.
    """

    async with _logged_in_client(tmp_path) as client:
        other = await _fresh_client(client.chatbook_test_server)
        async with other:
            login = await other.post(
                "/auth/login",
                data={"access_token": _DEVICES_ACCESS_TOKEN, "remember": "1"},
                allow_redirects=False,
            )
            assert login.status == 303
            entries = await _device_entries(client)
            assert len(entries) == 2
            other_session_id = next(
                session_id for session_id, is_current in entries if not is_current
            )

            # Same request as the revoke test above, minus the CSRF header.
            denied = await client.post(
                "/auth/devices/revoke",
                json={"session_id": other_session_id},
            )

            assert denied.status == 401
            # Service untouched: both sessions survive and stay usable.
            assert len(await _device_entries(client)) == 2
            assert (await other.get("/auth/devices")).status == 200
            assert (await client.get("/auth/devices")).status == 200


@pytest.mark.asyncio
async def test_revoke_all_ends_every_device_session(tmp_path) -> None:
    async with _logged_in_client(tmp_path, remember="1") as client:
        second = await _fresh_client(client.chatbook_test_server)
        async with second:
            login = await second.post(
                "/auth/login",
                data={"access_token": _DEVICES_ACCESS_TOKEN, "remember": "1"},
                allow_redirects=False,
            )
            assert login.status == 303

            # The devices page offers the incident-response control at the
            # header level, distinct from any per-device revoke button.
            page = await client.get("/auth/devices")
            body = await page.text()
            assert page.status == 200
            assert "Sign out all devices" in body
            assert 'class="device-revoke-all"' in body

            resp = await client.post(
                "/auth/devices/revoke-all",
                headers={CSRF_HEADER_NAME: await _csrf(client)},
            )
            assert resp.status == 200
            assert (await resp.json())["ok"] is True
            assert resp.headers["Cache-Control"] == "no-store"

            # Every device is signed out, including both requesting browsers:
            # the root GET bounces to the login page and a device-manager
            # request is denied outright.
            after_first = await client.get("/", allow_redirects=False)
            assert after_first.status == 303
            assert after_first.headers["Location"] == "/auth/login"
            assert (await second.get("/auth/devices")).status == 401

    # Nothing survives on disk: a restart would come up session-free.
    assert WebSessionStore(tmp_path / "web_sessions.json").load() == {}


@pytest.mark.asyncio
async def test_rename_updates_and_escapes_the_devices_page(tmp_path) -> None:
    async with _logged_in_client(tmp_path) as client:
        resp = await client.post(
            "/auth/devices/rename",
            json={
                "session_id": await _current_session_id(client),
                "label": "Kitchen <tablet>",
            },
            headers={CSRF_HEADER_NAME: await _csrf(client)},
        )
        assert resp.status == 200
        assert (await resp.json())["ok"] is True

        body = await (await client.get("/auth/devices")).text()
        assert "Kitchen &lt;tablet&gt;" in body
        assert "Kitchen <tablet>" not in body


# --- QR pairing and remember checkbox (task 5, ADR-233) -------------------


@pytest.mark.asyncio
async def test_pair_issues_one_use_bootstrap_url(tmp_path) -> None:
    async with _logged_in_client(tmp_path) as client:
        resp = await client.post(
            "/auth/devices/pair",
            headers={CSRF_HEADER_NAME: await _csrf(client)},
        )
        assert resp.status == 200
        assert resp.headers["Cache-Control"] == "no-store"
        url = (await resp.json())["url"]
        assert "/auth/bootstrap?nonce=" in url

        # The devices page wires the vendored QR runtime before its driver.
        page = await client.get("/auth/devices")
        body = await page.text()
        assert page.status == 200
        assert body.index("qrcodegen.js") < body.index("devices.js")
        asset = await client.get("/static/chatbook/qrcodegen.js")
        assert asset.status == 200

        # A second device (fresh cookie jar, same server) exchanges the URL
        # once: 303 into the app with a session cookie; the nonce is
        # single-use, so a replay is denied.
        phone = await _fresh_client(client.chatbook_test_server)
        async with phone:
            bootstrap_path = f"{urlsplit(url).path}?{urlsplit(url).query}"
            first = await phone.get(bootstrap_path, allow_redirects=False)
            assert first.status == 303
            assert SESSION_COOKIE_NAME in SimpleCookie(first.headers["Set-Cookie"])
            second = await phone.get(bootstrap_path, allow_redirects=False)
            assert second.status == 401


@pytest.mark.asyncio
async def test_login_without_remember_stays_short_lived(tmp_path) -> None:
    async with _logged_in_client(tmp_path, remember=None) as client:
        assert (await client.get("/auth/devices")).status == 200
        cookie = SimpleCookie(
            client.chatbook_login_response.headers["Set-Cookie"]
        )[SESSION_COOKIE_NAME]
        # The unchecked box downgrades to the legacy short session (Spec
        # §3.3, ADR-233 decision 5): the cookie's lifetime is bounded by the
        # 8-hour absolute window, never the configured device window.
        assert int(cookie["max-age"]) <= 8 * 3600 + 1

    # The unchecked box leaves the session in memory only: nothing that
    # would survive a restart reaches the disk store.
    assert WebSessionStore(tmp_path / "web_sessions.json").load() == {}


@pytest.mark.asyncio
async def test_login_with_remember_keeps_the_configured_device_cookie_window(
    tmp_path,
) -> None:
    async with _logged_in_client(tmp_path, remember="1") as client:
        cookie = SimpleCookie(
            client.chatbook_login_response.headers["Set-Cookie"]
        )[SESSION_COOKIE_NAME]

    # "Checked" keeps the configured device-session windows (defaults here:
    # 30-day idle / 180-day absolute), so the cookie far outlives the
    # legacy 8-hour downgrade.
    assert abs(int(cookie["max-age"]) - 180 * 86400) <= 5


@pytest.mark.asyncio
async def test_login_form_offers_remember_checkbox_with_accurate_copy(
    tmp_path,
) -> None:
    policy = build_web_auth_policy(
        host="127.0.0.1",
        port=8000,
        access_token=_DEVICES_ACCESS_TOKEN,
        public_url=_DEVICES_PUBLIC_ORIGIN,
        allow_insecure_remote_http=True,
        trusted_proxy_addresses=["127.0.0.1"],
    )
    server = _device_manager_server(tmp_path, policy)
    async with TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_DEVICE_REQUEST_HEADERS),
    ) as client:
        page = await client.get("/auth/login")
        body = await page.text()

    assert page.status == 200
    assert '<input type="checkbox" name="remember" value="1" checked' in body
    # The copy must describe both states accurately (task-3 review ruling):
    # checked = persistent device session (the window is configurable, so no
    # hard-coded day count); unchecked = short-lived in-memory session that
    # ends when the server stops.
    assert "persistent device session" in body
    assert "30-day" not in body
    assert "short-lived in-memory session" in body
    assert "ends when the server stops" in body


def test_device_label_from_user_agent_platform_and_browser_markers() -> None:
    chrome_mac = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    assert serve._device_label_from_user_agent(chrome_mac) == "Macintosh · Chrome"

    edge_windows = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.2210.61"
    )
    assert serve._device_label_from_user_agent(edge_windows) == "Windows · Edge"

    android_chrome = (
        "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
    )
    assert serve._device_label_from_user_agent(android_chrome) == "Android · Chrome"

    iphone_safari = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 "
        "Safari/604.1"
    )
    assert serve._device_label_from_user_agent(iphone_safari) == "iPhone · Safari"

    firefox_windows = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 "
        "Firefox/121.0"
    )
    assert serve._device_label_from_user_agent(firefox_windows) == "Windows · Firefox"

    assert serve._device_label_from_user_agent("curl/8.4.0") == "Unknown device"
    assert serve._device_label_from_user_agent("") == "Unknown device"


# --- Served passkeys (task 3, ADR-235) --------------------------------------
#
# The ceremony builders mirror the in-process software authenticator from
# Tests/Web_Server/test_webauthn_service.py (none-format attestation, ES256
# assertion with UP|UV flags), but drive the SERVER's HTTP routes: begin
# POSTs hit the real middleware/handlers and completes carry the built
# responses, so the round trips exercise the full route contract.

_PASSKEY_RP_ID = "chatbook.example"
#: WebAuthn client data always carries the https RP origin — browsers only
#: sign for secure contexts — regardless of the served public scheme.
_PASSKEY_ORIGIN = f"https://{_PASSKEY_RP_ID}"
_PASSKEY_RP_ID_HASH = hashlib.sha256(_PASSKEY_RP_ID.encode()).digest()
_PASSKEY_FLAG_UP_UV = 0x05  # UP | UV
_PASSKEY_FLAG_UP_UV_AT = 0x45  # UP | UV | AT


def _passkey_service(tmp_path) -> WebAuthnService:
    """The real ceremony service over a tmp-path credential store."""

    return WebAuthnService(
        public_url=_PASSKEY_ORIGIN,
        store=WebAuthnCredentialStore(tmp_path / "webauthn_credentials.json"),
    )


def _passkey_cose_key(public_key: ec.EllipticCurvePublicKey) -> dict:
    numbers = public_key.public_numbers()
    return {
        1: 2,  # kty: EC2
        3: -7,  # alg: ES256
        -1: 1,  # crv: P-256
        -2: numbers.x.to_bytes(32, "big"),
        -3: numbers.y.to_bytes(32, "big"),
    }


def _passkey_auth_data(
    flags: int, counter: int, credential_id: bytes | None, cose_key: bytes | None
) -> bytes:
    data = _PASSKEY_RP_ID_HASH + struct.pack(">BI", flags, counter)
    if credential_id is not None:
        assert cose_key is not None
        data += (
            b"\x00" * 16
            + struct.pack(">H", len(credential_id))
            + credential_id
            + cose_key
        )
    return data


def _passkey_client_data(kind: str, challenge, origin: str = _PASSKEY_ORIGIN) -> str:
    return json.dumps({"type": kind, "challenge": challenge, "origin": origin})


async def _register_passkey_via_routes(
    client,
    *,
    label: str = "YubiKey 5",
    credential_id: bytes = b"\x02" * 32,
    extensions_padding: int = 0,
):
    """Drive register begin -> complete against the live authed routes.

    ``extensions_padding`` grows the attestation with a well-shaped
    authenticator-data extensions blob (ED flag set), standing in for the
    packed x5c chains that make real registration bodies large.
    """

    csrf = await _csrf(client)
    begin = await client.post(
        "/auth/passkeys/register/begin",
        json={"label": label},
        headers={CSRF_HEADER_NAME: csrf},
    )
    assert begin.status == 200
    payload = await begin.json()
    private_key = ec.generate_private_key(ec.SECP256R1())
    cose_key = cbor.encode(_passkey_cose_key(private_key.public_key()))
    flags = _PASSKEY_FLAG_UP_UV_AT | (0x80 if extensions_padding else 0)
    auth_data = _passkey_auth_data(flags, 0, credential_id, cose_key)
    if extensions_padding:
        auth_data += cbor.encode(
            {"org.example.padding": b"\x00" * extensions_padding}
        )
    attestation_object = cbor.encode(
        {"fmt": "none", "attStmt": {}, "authData": auth_data}
    )
    complete = await client.post(
        "/auth/passkeys/register/complete",
        json={
            "challenge_id": payload["challenge_id"],
            "client_data_json": _passkey_client_data(
                "webauthn.create", payload["publicKey"]["challenge"]
            ),
            "attestation_object_b64": base64.b64encode(attestation_object).decode(),
        },
        headers={CSRF_HEADER_NAME: csrf},
    )
    assert complete.status == 200
    return private_key, credential_id, await complete.json()


async def _login_passkey_via_routes(
    client, private_key: ec.EllipticCurvePrivateKey, credential_id: bytes, *, counter: int
):
    """Drive login begin -> complete against the live public routes."""

    begin = await client.post("/auth/passkeys/login/begin")
    assert begin.status == 200
    payload = await begin.json()
    assert payload["publicKey"]["allowCredentials"] == []
    client_data_json = _passkey_client_data(
        "webauthn.get", payload["publicKey"]["challenge"]
    )
    auth_data = _passkey_auth_data(_PASSKEY_FLAG_UP_UV, counter, None, None)
    signature = private_key.sign(
        auth_data + hashlib.sha256(client_data_json.encode()).digest(),
        ec.ECDSA(hashes.SHA256()),
    )
    return await client.post(
        "/auth/passkeys/login/complete",
        json={
            "challenge_id": payload["challenge_id"],
            "client_data_json": client_data_json,
            "authenticator_data_b64": base64.b64encode(auth_data).decode(),
            "signature_b64": base64.b64encode(signature).decode(),
        },
        allow_redirects=False,
    )


@pytest.mark.asyncio
async def test_passkey_register_begin_requires_session_and_csrf(tmp_path) -> None:
    service = _passkey_service(tmp_path)
    server = _device_manager_server(
        tmp_path,
        build_web_auth_policy(
            host="127.0.0.1",
            port=8000,
            access_token=_DEVICES_ACCESS_TOKEN,
            public_url=_DEVICES_PUBLIC_ORIGIN,
            allow_insecure_remote_http=True,
            trusted_proxy_addresses=["127.0.0.1"],
        ),
        webauthn_service=service,
    )
    async with TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_DEVICE_REQUEST_HEADERS),
    ) as visitor:
        anonymous = await visitor.post(
            "/auth/passkeys/register/begin", json={"label": "k"}
        )
        assert anonymous.status == 401

    async with _logged_in_client(tmp_path, webauthn_service=service) as client:
        no_csrf = await client.post("/auth/passkeys/register/begin", json={"label": "k"})
        assert no_csrf.status == 401


@pytest.mark.asyncio
async def test_passkey_register_begin_under_domain_service_returns_options(
    tmp_path,
) -> None:
    async with _logged_in_client(
        tmp_path, webauthn_service=_passkey_service(tmp_path)
    ) as client:
        response = await client.post(
            "/auth/passkeys/register/begin",
            json={"label": "YubiKey 5"},
            headers={CSRF_HEADER_NAME: await _csrf(client)},
        )

        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        payload = await response.json()
        assert isinstance(payload["challenge_id"], str) and payload["challenge_id"]
        assert payload["publicKey"]["rp"]["id"] == _PASSKEY_RP_ID
        assert payload["publicKey"]["challenge"]


@pytest.mark.asyncio
async def test_passkey_routes_answer_503_when_passkeys_unavailable(
    tmp_path,
) -> None:
    async with _logged_in_client(tmp_path, webauthn_service=None) as client:
        csrf = await _csrf(client)
        register = await client.post(
            "/auth/passkeys/register/begin",
            json={},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert register.status == 503
        assert (await register.json())["error"] == "passkeys_unavailable"

        login = await client.post("/auth/passkeys/login/begin")
        assert login.status == 503
        assert (await login.json())["error"] == "passkeys_unavailable"


@pytest.mark.asyncio
async def test_passkey_login_begin_is_public(tmp_path) -> None:
    server = _device_manager_server(
        tmp_path,
        build_web_auth_policy(
            host="127.0.0.1",
            port=8000,
            access_token=_DEVICES_ACCESS_TOKEN,
            public_url=_DEVICES_PUBLIC_ORIGIN,
            allow_insecure_remote_http=True,
            trusted_proxy_addresses=["127.0.0.1"],
        ),
        webauthn_service=_passkey_service(tmp_path),
    )
    async with TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_DEVICE_REQUEST_HEADERS),
    ) as visitor:
        response = await visitor.post("/auth/passkeys/login/begin")

        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        payload = await response.json()
        assert payload["publicKey"]["allowCredentials"] == []
        assert payload["publicKey"]["rpId"] == _PASSKEY_RP_ID
        assert isinstance(payload["challenge_id"], str) and payload["challenge_id"]

        # A garbage completion is denied without leaking ceremony detail.
        garbage = await visitor.post(
            "/auth/passkeys/login/complete",
            json={
                "challenge_id": "deadbeef",
                "client_data_json": "x",
                "authenticator_data_b64": "x",
                "signature_b64": "x",
            },
            allow_redirects=False,
        )
        assert garbage.status == 401


@pytest.mark.asyncio
async def test_passkey_register_and_login_round_trip_via_server_routes(
    tmp_path,
) -> None:
    service = _passkey_service(tmp_path)
    async with _logged_in_client(
        tmp_path, remember=None, webauthn_service=service
    ) as client:
        private_key, credential_id, result = await _register_passkey_via_routes(client)
        assert result["ok"] is True
        assert result["credential_id"] == credential_id.hex()[:8]
        assert result["label"] == "YubiKey 5"

        # The devices page lists the credential with its public prefix and
        # carries the full public hex id for the Remove control.
        page = await client.get("/auth/devices")
        body = await page.text()
        assert "Add a passkey" in body
        assert f'data-credential-id="{credential_id.hex()}"' in body
        assert credential_id.hex()[:8] in body

        # A fresh, not-yet-signed-in browser completes a full login.
        phone = await _fresh_client(client.chatbook_test_server)
        async with phone:
            complete = await _login_passkey_via_routes(
                phone, private_key, credential_id, counter=1
            )
            assert complete.status == 303
            assert complete.headers["Location"] == "/"
            cookie = SimpleCookie(complete.headers["Set-Cookie"])[SESSION_COOKIE_NAME]
            assert cookie["httponly"] is True
            assert cookie["samesite"] == "Strict"
            # The minted cookie authenticates: an authed route answers 200
            # (the index shell itself needs textual-serve machinery the fake
            # test server does not carry, so prove the session this way).
            authed = await phone.get("/auth/devices")
            assert authed.status == 200

    # Exactly one persistent record: the passkey-minted device session (the
    # authed registrar used remember=None, so nothing else hit the disk).
    records = WebSessionStore(tmp_path / "web_sessions.json").load()
    assert len(records) == 1
    [record] = records.values()
    assert record.device_label.endswith(" (passkey)")


@pytest.mark.asyncio
async def test_passkey_register_complete_accepts_attestation_up_to_32_kib(
    tmp_path,
) -> None:
    """Registration bodies may carry packed x5c chains past the 4 KiB default.

    The register/complete route alone admits bodies up to 32 KiB (#3130) —
    here a >4 KiB, well-shaped attestation (authenticator-data extensions
    blob) completes — while anything beyond the cap still answers 413.
    """
    async with _logged_in_client(
        tmp_path, webauthn_service=_passkey_service(tmp_path)
    ) as client:
        _private_key, _credential_id, result = await _register_passkey_via_routes(
            client, extensions_padding=8192
        )
        assert result["ok"] is True

        csrf = await _csrf(client)
        begin = await client.post(
            "/auth/passkeys/register/begin",
            json={"label": "too big"},
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert begin.status == 200
        payload = await begin.json()
        private_key = ec.generate_private_key(ec.SECP256R1())
        cose_key = cbor.encode(_passkey_cose_key(private_key.public_key()))
        credential_id = b"\x03" * 32
        auth_data = _passkey_auth_data(
            _PASSKEY_FLAG_UP_UV_AT | 0x80, 0, credential_id, cose_key
        ) + cbor.encode({"org.example.padding": b"\x00" * (48 * 1024)})
        attestation_object = cbor.encode(
            {"fmt": "none", "attStmt": {}, "authData": auth_data}
        )
        oversized = await client.post(
            "/auth/passkeys/register/complete",
            json={
                "challenge_id": payload["challenge_id"],
                "client_data_json": _passkey_client_data(
                    "webauthn.create", payload["publicKey"]["challenge"]
                ),
                "attestation_object_b64": base64.b64encode(attestation_object).decode(),
            },
            headers={CSRF_HEADER_NAME: csrf},
        )
        assert oversized.status == 413


@pytest.mark.asyncio
async def test_passkey_remove_via_devices_route(tmp_path) -> None:
    service = _passkey_service(tmp_path)
    async with _logged_in_client(tmp_path, webauthn_service=service) as client:
        private_key, credential_id, _result = await _register_passkey_via_routes(client)

        removed = await client.post(
            "/auth/passkeys/remove",
            json={"credential_id": credential_id.hex()},
            headers={CSRF_HEADER_NAME: await _csrf(client)},
        )
        assert removed.status == 200
        assert (await removed.json())["ok"] is True
        assert removed.headers["Cache-Control"] == "no-store"

        again = await client.post(
            "/auth/passkeys/remove",
            json={"credential_id": credential_id.hex()},
            headers={CSRF_HEADER_NAME: await _csrf(client)},
        )
        assert (again.status == 200) and (await again.json())["ok"] is False

        body = await (await client.get("/auth/devices")).text()
        assert credential_id.hex()[:8] not in body

        complete = await _login_passkey_via_routes(
            client, private_key, credential_id, counter=1
        )
        assert complete.status == 401


@pytest.mark.asyncio
async def test_passkey_pages_gate_on_service_availability(tmp_path) -> None:
    async with _logged_in_client(
        tmp_path, webauthn_service=_passkey_service(tmp_path)
    ) as client:
        devices = await (await client.get("/auth/devices")).text()
        assert "Add a passkey" in devices
        assert "/static/chatbook/passkeys.js" in devices
        login_page = await client.get("/auth/login")
        assert login_page.status == 200
        assert "Use passkey" in (await login_page.text())

    async with _logged_in_client(tmp_path, webauthn_service=None) as client:
        devices = await (await client.get("/auth/devices")).text()
        assert "Passkeys need a domain origin" in devices
        assert "Add a passkey" not in devices
        assert "/static/chatbook/passkeys.js" not in devices
        login_page = await client.get("/auth/login")
        assert login_page.status == 200
        body = await login_page.text()
        assert "Use passkey" not in body
        assert "/static/chatbook/passkeys.js" not in body
