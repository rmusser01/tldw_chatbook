from __future__ import annotations

from http.cookies import SimpleCookie

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tldw_chatbook.Canvas.web_auth import (
    SESSION_COOKIE_NAME,
    WEBSOCKET_PROTOCOL,
    build_web_auth_policy,
)
from tldw_chatbook.Web_Server import serve

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
