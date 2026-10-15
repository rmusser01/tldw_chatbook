from __future__ import annotations

import re
from contextlib import asynccontextmanager
from http.cookies import SimpleCookie

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tldw_chatbook.Canvas.web_auth import (
    CSRF_HEADER_NAME,
    SESSION_COOKIE_NAME,
    WEBSOCKET_PROTOCOL,
    build_web_auth_policy,
)
from tldw_chatbook.config import build_canvas_config_policy
from tldw_chatbook.Web_Server import serve
from tldw_chatbook.Web_Server.session_store import WebSessionStore

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


def _device_manager_server(tmp_path, policy):
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
    )


@asynccontextmanager
async def _logged_in_client(tmp_path, *, remember: str = "1"):
    """Yield one token-authenticated TestClient against a fresh server.

    The client carries the forwarded routing headers by default and has
    posted the login form with ``remember`` (default ``"1"``), so Task 5 can
    flip the checkbox value and assert on ``web_sessions.json``.
    """

    policy = build_web_auth_policy(
        host="127.0.0.1",
        port=8000,
        access_token=_DEVICES_ACCESS_TOKEN,
        public_url=_DEVICES_PUBLIC_ORIGIN,
        allow_insecure_remote_http=True,
        trusted_proxy_addresses=["127.0.0.1"],
    )
    server = _device_manager_server(tmp_path, policy)
    client = TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_DEVICE_REQUEST_HEADERS),
    )
    client.chatbook_test_server = server
    async with client:
        login = await client.post(
            "/auth/login",
            data={"access_token": _DEVICES_ACCESS_TOKEN, "remember": remember},
            allow_redirects=False,
        )
        assert login.status == 303
        assert login.headers["Location"] == "/"
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
