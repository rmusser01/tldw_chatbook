"""/m phone-first companion page + PWA assets — task 6, ADR-234 Phase 1b.

Route-level tests over the served HTML and the ``/static/mobile`` asset
table: the page's auth contract (401 without a session), its required
metadata (manifest link, CSRF meta, ``viewport-fit=cover``), the asset
content types / cache policies / plausible icon bytes, the narrow-viewport
"Phone view" discovery links on the login page and the served shell, and
the ``mobile_surface`` registration gate (a disabled surface 404s). The
client-side behavior of ``app.js`` is not under test here — the wire
contracts it consumes are owned by ``test_mobile_api.py``.
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tldw_chatbook.Canvas.web_auth import build_web_auth_policy
from tldw_chatbook.config import build_canvas_config_policy
from tldw_chatbook.Web_Server import mobile_api, serve
from tldw_chatbook.Web_Server.session_store import WebSessionStore

pytestmark = [
    pytest.mark.skipif(
        not serve.check_web_server_available(),
        reason="web server optional dependencies are unavailable",
    ),
    pytest.mark.loopback_network,
]


# The server/client scaffolding below (_FakeBaseServer, _page_server,
# _app_without_control_broker, _logged_in_client) is duplicated from
# Tests/Web_Server/test_web_auth_integration.py / test_mobile_api.py: no
# shared Tests/Web_Server/conftest.py exists (task-4 pre-flight ruling), so
# each file carries its own copy. Keep them in sync.


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


_PAGE_ACCESS_TOKEN = "mobile-page-secret"
_PAGE_PUBLIC_ORIGIN = "http://chatbook.example"
_PAGE_REQUEST_HEADERS = {
    "Origin": _PAGE_PUBLIC_ORIGIN,
    "X-Forwarded-For": "203.0.113.9",
    "X-Forwarded-Proto": "http",
    "X-Forwarded-Host": "chatbook.example",
}


def _page_server(tmp_path):
    """Build a config-free served server (no service: the page never needs one)."""

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
    policy = build_web_auth_policy(
        host="127.0.0.1",
        port=8000,
        access_token=_PAGE_ACCESS_TOKEN,
        public_url=_PAGE_PUBLIC_ORIGIN,
        allow_insecure_remote_http=True,
        trusted_proxy_addresses=["127.0.0.1"],
    )
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


async def _app_without_control_broker(server):
    app = await server._make_app()
    app.on_startup.clear()
    app.on_shutdown.clear()
    return app


@asynccontextmanager
async def _logged_in_client(tmp_path):
    """Yield one token-authenticated TestClient against a fresh server."""

    server = _page_server(tmp_path)
    client = TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_PAGE_REQUEST_HEADERS),
    )
    async with client:
        login = await client.post(
            "/auth/login",
            data={"access_token": _PAGE_ACCESS_TOKEN, "remember": "1"},
            allow_redirects=False,
        )
        assert login.status == 303
        assert login.headers["Location"] == "/"
        yield client


async def _devices_csrf(client) -> str:
    """Read the per-session CSRF token from the devices page's meta tag."""

    page = await client.get("/auth/devices")
    assert page.status == 200
    match = re.search(
        r'<meta name="chatbook-csrf" content="([^"]*)">', await page.text()
    )
    assert match is not None
    return match.group(1)


@pytest.fixture(autouse=True)
def mobile_surface_gate(monkeypatch):
    """Pin the ``[web_server] mobile_surface`` gate without loading config.

    Same discipline as ``test_mobile_api.py``: the guarded config loader
    must not run here, so the gate's ``get_cli_setting`` reference is
    patched; the returned installer flips the gate BEFORE the app's route
    table is built.
    """

    def install(enabled: bool) -> None:
        monkeypatch.setattr(
            mobile_api,
            "get_cli_setting",
            lambda section, key=None, default=None: (
                enabled
                if (section, key) == ("web_server", "mobile_surface")
                else default
            ),
        )

    install(True)
    return install


@pytest.mark.asyncio
async def test_mobile_page_requires_an_authenticated_session(
    tmp_path, mobile_surface_gate
) -> None:
    server = _page_server(tmp_path)
    async with TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_PAGE_REQUEST_HEADERS),
    ) as client:
        response = await client.get("/m")

    assert response.status == 401


@pytest.mark.asyncio
async def test_mobile_page_serves_required_metadata(tmp_path) -> None:
    async with _logged_in_client(tmp_path) as client:
        response = await client.get("/m")
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        body = await response.text()

    assert (
        '<meta name="viewport" content="width=device-width, initial-scale=1,'
        ' viewport-fit=cover">' in body
    )
    assert '<link rel="manifest" href="/static/mobile/manifest.webmanifest">' in body
    csrf_match = re.search(
        r'<meta name="chatbook-csrf" content="([^"]+)">', body
    )
    assert csrf_match is not None


@pytest.mark.asyncio
async def test_mobile_page_csrf_matches_the_session_token(tmp_path) -> None:
    async with _logged_in_client(tmp_path) as client:
        page = await client.get("/m")
        assert page.status == 200
        body = await page.text()
        csrf_match = re.search(
            r'<meta name="chatbook-csrf" content="([^"]+)">', body
        )
        assert csrf_match is not None
        assert csrf_match.group(1) == await _devices_csrf(client)


@pytest.mark.asyncio
async def test_mobile_page_references_its_assets(tmp_path) -> None:
    async with _logged_in_client(tmp_path) as client:
        body = await (await client.get("/m")).text()

    assert '<link rel="stylesheet" href="/static/mobile/app.css">' in body
    assert '<script src="/static/mobile/app.js" defer></script>' in body
    assert '<meta name="apple-mobile-web-app-capable" content="yes">' in body
    assert '<meta name="theme-color" content="' in body
    assert (
        '<link rel="apple-touch-icon" href="/static/mobile/apple-touch-icon.png">'
        in body
    )


@pytest.mark.asyncio
async def test_mobile_manifest_is_served_with_manifest_content_type(
    tmp_path,
) -> None:
    async with _logged_in_client(tmp_path) as client:
        response = await client.get("/static/mobile/manifest.webmanifest")
        assert response.status == 200
        assert response.headers["Content-Type"].startswith(
            "application/manifest+json"
        )
        assert response.headers["Cache-Control"] == "public, max-age=3600"
        manifest = json.loads(await response.text())

    assert manifest["start_url"] == "/m"
    assert manifest["display"] == "standalone"
    icon_srcs = {icon["src"] for icon in manifest["icons"]}
    assert icon_srcs == {
        "/static/mobile/icon-192.png",
        "/static/mobile/icon-512.png",
    }


@pytest.mark.parametrize(
    "name",
    ["icon-192.png", "icon-512.png", "apple-touch-icon.png"],
)
@pytest.mark.asyncio
async def test_mobile_icons_are_plausible_pngs(tmp_path, name) -> None:
    async with _logged_in_client(tmp_path) as client:
        response = await client.get(f"/static/mobile/{name}")
        assert response.status == 200
        assert response.headers["Content-Type"] == "image/png"
        assert response.headers["Cache-Control"] == "public, max-age=3600"
        body = await response.read()

    assert len(body) > 1024, name
    assert body.startswith(b"\x89PNG\r\n\x1a\n")
    # PNG icon sizes decode from the IHDR chunk (bytes 16-24, big-endian).
    width = int.from_bytes(body[16:20], "big")
    height = int.from_bytes(body[20:24], "big")
    expected = {"icon-192.png": 192, "icon-512.png": 512, "apple-touch-icon.png": 180}
    assert (width, height) == (expected[name], expected[name])


@pytest.mark.asyncio
async def test_mobile_index_asset_is_never_cached(tmp_path) -> None:
    async with _logged_in_client(tmp_path) as client:
        response = await client.get("/static/mobile/index.html")

    assert response.status == 200
    assert response.headers["Content-Type"].startswith("text/html")
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.asyncio
async def test_mobile_app_assets_are_served(tmp_path) -> None:
    async with _logged_in_client(tmp_path) as client:
        css = await client.get("/static/mobile/app.css")
        js = await client.get("/static/mobile/app.js")

    assert css.status == 200
    assert css.headers["Content-Type"].startswith("text/css")
    assert css.headers["Cache-Control"] == "public, max-age=3600"
    assert js.status == 200
    assert js.headers["Content-Type"].startswith("application/javascript")
    assert js.headers["Cache-Control"] == "public, max-age=3600"


@pytest.mark.asyncio
async def test_mobile_css_keeps_the_hidden_attribute_authoritative(
    tmp_path,
) -> None:
    """View switching relies on the ``hidden`` attribute (re-review r2).

    ``showView`` toggles only ``hidden``; author-origin ``display`` rules
    (``#view-chat { display: flex }``, ``.badge { display: inline-block }``)
    override the UA stylesheet's ``[hidden] { display: none }`` unless the
    stylesheet carries an explicit guard — the round-2 breakage stacked the
    chat view (and its composer) under the session list. No wire test can
    catch a CSS-vs-attribute interaction, so the served CSS is pinned.
    """
    async with _logged_in_client(tmp_path) as client:
        response = await client.get("/static/mobile/app.css")
        assert response.status == 200
        body = await response.text()

    # The guard itself…
    assert "[hidden]" in body
    assert "display: none !important" in body
    # …against the two author-origin display rules that defeated the
    # attribute in the wild.
    assert "#view-chat" in body and "display: flex" in body
    assert ".badge" in body and "display: inline-block" in body


@pytest.mark.asyncio
async def test_mobile_css_header_fits_narrow_viewports(tmp_path) -> None:
    """Header right-edge fit + visible divider (visual round 3).

    At 390px the actions cluster ("Approvals [n]" + "Open terminal" +
    "Sign out") measures ~305px at 13px link font. The pinned media query
    gives the cluster its own full-width row below 480px, so its available
    width is the whole content box — 390 − 2×10px padding = 370px — and
    "Sign out" can no longer be pushed past the right edge (the round-3
    clip: it shared a row with the brand and overflowed). The divider
    hairline is pinned to a translucent white that reads against the dark
    surface, unlike the old --border tone. The arithmetic below is the
    acceptance computation; the substring pins keep the rules from
    silently regressing.
    """
    async with _logged_in_client(tmp_path) as client:
        response = await client.get("/static/mobile/app.css")
        assert response.status == 200
        body = await response.text()

    viewport_px = 390
    header_padding_px = 10  # per side, in the narrow band
    cluster_measured_px = 305  # Approvals+badge, Open terminal, Sign out, gaps
    available_px = viewport_px - 2 * header_padding_px
    assert available_px == 370
    assert available_px > cluster_measured_px + 40  # ≥ 40px of slack

    # Own-row fit: the cluster spans the full row below 480px, right-anchored.
    assert "@media (max-width: 480px)" in body
    assert "flex-basis: 100%" in body
    assert "justify-content: flex-end" in body
    # Links never truncate mid-word.
    assert ".header-link" in body and "white-space: nowrap" in body
    # Perceivable divider hairline.
    assert "rgba(255, 255, 255, 0.12)" in body


@pytest.mark.asyncio
async def test_unknown_mobile_asset_is_404(tmp_path) -> None:
    async with _logged_in_client(tmp_path) as client:
        response = await client.get("/static/mobile/not-an-asset.js")

    assert response.status == 404


@pytest.mark.asyncio
async def test_mobile_page_offers_the_library_tab(tmp_path) -> None:
    """Phase 2b: the authed page carries the Chats/Library tab switch.

    The switch is a bottom tab bar (not header buttons — the header's
    narrow-viewport fit is measured and pinned by
    ``test_mobile_css_header_fits_narrow_viewports``), and the reading
    destinations (library list + note reader + media detail) are guarded
    sections in the same ``[hidden]`` view system.
    """
    async with _logged_in_client(tmp_path) as client:
        response = await client.get("/m")
        assert response.status == 200
        body = await response.text()

    assert 'id="tab-bar"' in body
    assert 'id="tab-chats"' in body and ">Chats</button>" in body
    assert 'id="tab-library"' in body and ">Library</button>" in body
    assert 'id="view-library"' in body
    assert 'id="notes-list"' in body
    assert 'id="media-list"' in body
    assert 'id="view-note"' in body
    assert 'id="view-item"' in body


@pytest.mark.asyncio
async def test_mobile_app_js_carries_the_library_surface(tmp_path) -> None:
    """The served client contains the library markers + the four fetch paths.

    Grep-style pinning over ``/static/mobile/app.js`` (the client's
    behavior is not executed here): the reading surface must ride the
    existing ``api()`` helper against exactly the four Task-2 GET routes.
    """
    async with _logged_in_client(tmp_path) as client:
        response = await client.get("/static/mobile/app.js")
        assert response.status == 200
        js = await response.text()

    for marker in (
        "tab-chats",
        "tab-library",
        "notes-list",
        "media-list",
        "notes-more-button",
        "media-more-button",
    ):
        assert marker in js, marker
    # The four reading fetch paths, as literal path fragments.
    assert '"/notes?page="' in js
    assert '"/notes/"' in js
    assert '"/library?page="' in js
    assert '"/library/"' in js


@pytest.mark.asyncio
async def test_mobile_app_js_tags_approval_surfaces_and_handles_run_status(
    tmp_path,
) -> None:
    """ADR-236 phone markers: each approval card carries a desktop/phone
    surface tag, and desktop ``run_status`` events resync the open chat
    plus the approvals inbox. Grep-style pinning over the served client.
    """
    async with _logged_in_client(tmp_path) as client:
        response = await client.get("/static/mobile/app.js")
        assert response.status == 200
        js = await response.text()

    # The surface tag: one badge class, rendered from the wire's surface
    # field, labelling desktop-relayed rounds "desktop" and phone ones
    # "phone" (wire text reaches only textContent, never a class name).
    assert "approval-surface" in js
    assert '=== "desktop"' in js
    assert '"desktop"' in js and '"phone"' in js

    # run_status handling: the desktop relay's completion pushes ride the
    # existing event envelope; the case block must resync the open chat
    # and refresh the approvals inbox.
    case_index = js.index('case "run_status":')
    block = js[case_index : js.index("break;", case_index)]
    assert "syncOpenSession()" in block
    assert "refreshApprovals()" in block


@pytest.mark.asyncio
async def test_mobile_css_styles_the_approval_surface_tag(tmp_path) -> None:
    """The surface tag reuses the reading surface's pill styling."""
    async with _logged_in_client(tmp_path) as client:
        response = await client.get("/static/mobile/app.css")
        assert response.status == 200
        body = await response.text()

    assert ".approval-surface" in body


@pytest.mark.asyncio
async def test_mobile_css_styles_the_reading_surface(tmp_path) -> None:
    """The served stylesheet carries the tab bar + pre-wrap reading rules.

    Note/transcript bodies are plain text rendered via ``textContent``;
    the plan's no-HTML-injection contract relies on ``white-space:
    pre-wrap`` (never innerHTML) for formatting.
    """
    async with _logged_in_client(tmp_path) as client:
        response = await client.get("/static/mobile/app.css")
        assert response.status == 200
        body = await response.text()

    assert ".tab-bar" in body
    assert ".load-more-button" in body
    assert ".reading-content" in body
    assert "white-space: pre-wrap" in body


@pytest.mark.asyncio
async def test_mobile_surface_false_leaves_the_page_unregistered(
    tmp_path, mobile_surface_gate
) -> None:
    mobile_surface_gate(False)
    async with _logged_in_client(tmp_path) as client:
        page = await client.get("/m")
        manifest = await client.get("/static/mobile/manifest.webmanifest")
        app_js = await client.get("/static/mobile/app.js")

    assert page.status == 404
    assert manifest.status == 404
    assert app_js.status == 404


@pytest.mark.asyncio
async def test_login_page_carries_the_phone_view_discovery_link(
    tmp_path, mobile_surface_gate
) -> None:
    server = _page_server(tmp_path)
    async with TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_PAGE_REQUEST_HEADERS),
    ) as client:
        response = await client.get("/auth/login")
        assert response.status == 200
        body = await response.text()

    assert 'href="/m"' in body
    assert "Phone view" in body
    # Narrow-viewport visibility is a CSS media query, never UA sniffing.
    assert "max-width" in body
    assert "@media" in body


def test_served_shell_carries_the_phone_view_discovery_link() -> None:
    # handle_index serves exactly ``serve._SERVED_SHELL_HTML`` with only the
    # CSRF/title/WS/font placeholders substituted, so asserting on the
    # constant covers every served copy. (A GET "/" through the test server
    # would additionally run ``resolve_web_font_size`` -> the guarded config
    # loader, which fails closed under the per-test sandbox — one of the 11
    # pre-existing environmental failures in test_web_auth_integration.py.)
    body = serve._SERVED_SHELL_HTML

    assert 'href="/m"' in body
    assert "Phone view" in body
    assert "@media" in body
    assert "max-width" in body
