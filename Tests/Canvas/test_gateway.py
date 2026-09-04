"""Native Canvas gateway route, security, and lifecycle tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlsplit

import aiohttp
import pytest
from aiohttp import web

from tldw_chatbook.Canvas.compiler import compile_canvas_document
from tldw_chatbook.Canvas.gateway import (
    BridgeConfirmationRequest,
    BridgeConfirmationResponse,
    CanvasGateway,
    CanvasGatewayEvent,
    CanvasGatewayScope,
    CanvasSourceResponse,
)
from tldw_chatbook.Canvas.limits import sha256_utf8


@dataclass
class _Authority:
    calls: list[tuple[str, CanvasGatewayScope]]

    async def resolve_render_plan(self, scope: CanvasGatewayScope):
        self.calls.append(("plan", scope))
        return compile_canvas_document(
            "<!doctype html><html><body><h1>Exact plan</h1></body></html>"
        )

    async def read_source(self, scope: CanvasGatewayScope) -> CanvasSourceResponse:
        self.calls.append(("source", scope))
        source = "<!doctype html><title>Exact source</title>"
        return CanvasSourceResponse(
            source=source,
            content_sha256=sha256_utf8(source),
        )

    async def read_events(
        self, scope: CanvasGatewayScope, *, after_event_id: str | None
    ) -> tuple[CanvasGatewayEvent, ...]:
        self.calls.append(("events", scope))
        return (
            CanvasGatewayEvent(
                event_id="event-a",
                kind="updated",
                canvas_id=scope.canvas_id,
                revision_id=scope.revision_id,
            ),
        )

    async def confirm_bridge(
        self, scope: CanvasGatewayScope, request: BridgeConfirmationRequest
    ) -> BridgeConfirmationResponse:
        self.calls.append(("bridge", scope))
        return BridgeConfirmationResponse(
            request_id=request.request.request_id, status="confirmed"
        )


def _scope(**changes: str) -> CanvasGatewayScope:
    values = {
        "browser_session_id": "browser-a",
        "conversation_session_id": "conversation-session-a",
        "canvas_id": "canvas-a",
        "revision_id": "revision-a",
    }
    values.update(changes)
    return CanvasGatewayScope(**values)


def test_gateway_event_metadata_is_closed_and_source_free() -> None:
    with pytest.raises(ValueError, match="metadata field"):
        CanvasGatewayEvent(
            event_id="event-a",
            kind="updated",
            canvas_id="canvas-a",
            revision_id="revision-a",
            metadata={"html": "<!doctype html><title>must not travel</title>"},
        )


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    value: object,
    *,
    origin: str,
    csrf: str | None = None,
    capability: str | None = None,
) -> aiohttp.ClientResponse:
    headers = {"Origin": origin, "Content-Type": "application/json"}
    if csrf is not None:
        headers["X-Canvas-CSRF"] = csrf
    if capability is not None:
        headers["Authorization"] = f"CanvasCapability {capability}"
    return await session.post(url, data=json.dumps(value), headers=headers)


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_gateway_starts_lazily_once_on_numeric_loopback_and_shuts_down() -> None:
    authority = _Authority([])
    gateway = CanvasGateway(authority=authority)
    assert gateway.started is False

    first = await gateway.start()
    second = await gateway.start()

    parsed = urlsplit(first)
    assert first == second
    assert parsed.hostname == "127.0.0.1"
    assert isinstance(parsed.port, int) and parsed.port > 0
    assert gateway.start_count == 1
    await gateway.aclose()
    assert gateway.started is False
    assert gateway.capabilities.closed is True


@pytest.mark.parametrize(
    "host", ["localhost", "0.0.0.0", "192.168.1.2", "example.test"]
)
def test_gateway_rejects_non_numeric_or_non_loopback_bind(host: str) -> None:
    with pytest.raises(ValueError, match="numeric loopback"):
        CanvasGateway(authority=_Authority([]), host=host)


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_startup_and_browser_open_failures_are_recoverable_without_state_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = CanvasGateway(authority=_Authority([]))
    real_site_start = web.TCPSite.start
    attempts = 0

    async def flaky_site_start(site: web.TCPSite) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("sensitive-bind-detail")
        await real_site_start(site)

    monkeypatch.setattr(web.TCPSite, "start", flaky_site_start)
    with pytest.raises(RuntimeError, match="could not start") as exc_info:
        await gateway.start()
    assert "sensitive-bind-detail" not in str(exc_info.value)

    opened: list[str] = []

    def unavailable_browser(url: str) -> None:
        opened.append(url)
        raise RuntimeError("private-browser-detail")

    launch = await gateway.open_shell(_scope(), opener=unavailable_browser)

    assert launch.opened is False
    assert launch.error_code == "browser_unavailable"
    assert "private-browser-detail" not in repr(launch)
    assert opened == [launch.browser_url]
    assert "?" not in launch.browser_url
    assert "#boot=" in launch.browser_url
    assert launch.clean_url == f"{gateway.origin}/canvas/"
    assert gateway.start_count == 1
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_boot_frame_plan_assets_events_source_and_bridge_are_exactly_scoped() -> (
    None
):
    authority = _Authority([])
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope())
    assert gateway.origin is not None
    origin = gateway.origin
    jar = aiohttp.CookieJar(unsafe=True)
    async with aiohttp.ClientSession(cookie_jar=jar) as session:
        shell = await session.get(launch.clean_url)
        assert shell.status == 200
        assert shell.content_type == "text/html"
        assert "no-store" in shell.headers["Cache-Control"]
        assert shell.headers["Referrer-Policy"] == "no-referrer"
        assert shell.headers["X-Content-Type-Options"] == "nosniff"
        assert shell.headers["X-Frame-Options"] == "DENY"
        assert "camera=()" in shell.headers["Permissions-Policy"]
        assert "frame-ancestors 'none'" in shell.headers["Content-Security-Policy"]

        bootstrap = launch.browser_url.split("#boot=", 1)[1]
        boot = await _post_json(
            session,
            f"{origin}/canvas/api/boot",
            {"bootstrap": bootstrap},
            origin=origin,
        )
        assert boot.status == 200
        boot_body = await boot.json()
        csrf = boot_body["csrf"]
        session_cookie = next(
            morsel.value for morsel in jar if morsel.key == "canvas_session"
        )
        assert session_cookie.encode() not in b"".join(gateway._sessions)
        assert session_cookie not in repr(gateway)
        assert bootstrap not in repr(gateway)
        assert boot_body["selection"] == {
            "canvas_id": "canvas-a",
            "revision_id": "revision-a",
        }

        frame = await _post_json(
            session,
            f"{origin}/canvas/api/frame",
            {},
            origin=origin,
            csrf=csrf,
        )
        assert frame.status == 200
        frame_body = await frame.json()

        top_level = await session.get(
            f"{origin}{frame_body['renderer_url']}",
            headers={"Sec-Fetch-Dest": "document", "Sec-Fetch-Site": "same-origin"},
        )
        assert top_level.status == 403

        # A refused top-level request does not consume the iframe capability.
        renderer = await session.get(
            f"{origin}{frame_body['renderer_url']}",
            headers={"Sec-Fetch-Dest": "iframe", "Sec-Fetch-Site": "same-origin"},
        )
        assert renderer.status == 200
        assert renderer.content_type == "text/html"
        assert renderer.headers["X-Frame-Options"] == "SAMEORIGIN"
        assert "connect-src 'none'" in renderer.headers["Content-Security-Policy"]
        assert "sandbox allow-scripts" in renderer.headers["Content-Security-Policy"]

        replay = await session.get(
            f"{origin}{frame_body['renderer_url']}",
            headers={"Sec-Fetch-Dest": "iframe", "Sec-Fetch-Site": "same-origin"},
        )
        assert replay.status == 401

        plan = await session.get(f"{origin}/canvas/api/plan")
        assert plan.status == 200
        plan_body = await plan.json()
        assert plan_body["runtime_profile"] == "canvas-v1"
        assert "source" not in plan_body
        assert (
            plan_body["root"]["children"][1]["children"][0]["children"][0]["text"]
            == "Exact plan"
        )

        asset = await session.get(f"{origin}/canvas/static/canvas_renderer.js")
        assert asset.status == 200
        assert asset.content_type == "text/javascript"
        assert asset.headers["Access-Control-Allow-Origin"] == "*"
        assert asset.headers["Cross-Origin-Resource-Policy"] == "cross-origin"
        assert 'crossorigin="anonymous"' in await renderer.text()
        missing = await session.get(f"{origin}/canvas/static/not-packaged.js")
        assert missing.status == 404

        exact_method = await session.head(launch.clean_url)
        assert exact_method.status == 405
        assert exact_method.headers["Cache-Control"] == "no-store"

        events = await session.get(f"{origin}/canvas/api/events")
        assert events.status == 200
        assert (await events.json())["events"][0]["revision_id"] == "revision-a"

        source_grant = await _post_json(
            session,
            f"{origin}/canvas/api/actions",
            {"action": "source_read"},
            origin=origin,
            csrf=csrf,
        )
        source_token = (await source_grant.json())["capability"]
        source = await session.get(
            f"{origin}/canvas/api/source",
            headers={"Authorization": f"CanvasCapability {source_token}"},
        )
        assert source.status == 200
        assert source.content_type == "text/plain"
        assert await source.text() == "<!doctype html><title>Exact source</title>"
        assert (
            await session.get(
                f"{origin}/canvas/api/source",
                headers={"Authorization": f"CanvasCapability {source_token}"},
            )
        ).status == 401

        bridge_grant = await _post_json(
            session,
            f"{origin}/canvas/api/actions",
            {"action": "bridge_confirm"},
            origin=origin,
            csrf=csrf,
        )
        bridge_token = (await bridge_grant.json())["capability"]
        bridge = await _post_json(
            session,
            f"{origin}/canvas/api/bridge",
            {
                "approved": True,
                "request": {
                    "version": "canvas-v1",
                    "request_id": "request-a",
                    "kind": "submit",
                    "value": {"answer": 42},
                },
            },
            origin=origin,
            csrf=csrf,
            capability=bridge_token,
        )
        assert bridge.status == 200
        assert await bridge.json() == {"request_id": "request-a", "status": "confirmed"}

        closed = await _post_json(
            session,
            f"{origin}/canvas/api/close",
            {},
            origin=origin,
            csrf=csrf,
        )
        assert closed.status == 200
        assert gateway.browser_session_count == 0
        assert (await session.get(f"{origin}/canvas/api/events")).status == 401

    assert authority.calls == [
        ("plan", _scope()),
        ("events", _scope()),
        ("source", _scope()),
        ("bridge", _scope()),
    ]
    assert not any(
        path.resource.canonical == "/canvas/api/conversations"
        for path in gateway.routes
    )
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_events_fail_closed_when_authority_returns_a_sibling_canvas() -> None:
    class SiblingEventAuthority(_Authority):
        async def read_events(
            self, scope: CanvasGatewayScope, *, after_event_id: str | None
        ) -> tuple[CanvasGatewayEvent, ...]:
            return (
                CanvasGatewayEvent(
                    event_id="event-sibling",
                    kind="updated",
                    canvas_id="canvas-sibling",
                    revision_id="revision-sibling",
                ),
            )

    gateway = CanvasGateway(authority=SiblingEventAuthority([]))
    launch = await gateway.open_shell(_scope())
    assert gateway.origin is not None
    origin = gateway.origin
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        bootstrap = launch.browser_url.split("#boot=", 1)[1]
        boot = await _post_json(
            session,
            f"{origin}/canvas/api/boot",
            {"bootstrap": bootstrap},
            origin=origin,
        )
        assert boot.status == 200
        events = await session.get(f"{origin}/canvas/api/events")
        assert events.status == 503
        assert "canvas-sibling" not in await events.text()
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_gateway_rejects_query_credentials_proxy_headers_origin_csrf_type_and_size() -> (
    None
):
    gateway = CanvasGateway(authority=_Authority([]), max_request_bytes=256)
    launch = await gateway.open_shell(_scope())
    assert gateway.origin is not None
    origin = gateway.origin
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        for query in ("bootstrap=secret", "capability=secret", "token=secret"):
            response = await session.get(f"{launch.clean_url}?{query}")
            assert response.status == 400
            assert query.split("=", 1)[1] not in await response.text()

        proxied = await session.get(
            launch.clean_url, headers={"X-Forwarded-Host": "attacker.test"}
        )
        assert proxied.status == 400
        wrong_host = await session.get(launch.clean_url, headers={"Host": "localhost"})
        assert wrong_host.status == 400

        bootstrap = launch.browser_url.split("#boot=", 1)[1]
        wrong_origin = await _post_json(
            session,
            f"{origin}/canvas/api/boot",
            {"bootstrap": bootstrap},
            origin="http://127.0.0.1:1",
        )
        assert wrong_origin.status == 403
        wrong_type = await session.post(
            f"{origin}/canvas/api/boot",
            data="{}",
            headers={"Origin": origin, "Content-Type": "text/plain"},
        )
        assert wrong_type.status == 415
        oversized = await session.post(
            f"{origin}/canvas/api/boot",
            data="x" * 257,
            headers={"Origin": origin, "Content-Type": "application/json"},
        )
        assert oversized.status == 413

        boot = await _post_json(
            session,
            f"{origin}/canvas/api/boot",
            {"bootstrap": bootstrap},
            origin=origin,
        )
        assert boot.status == 200
        no_csrf = await _post_json(
            session,
            f"{origin}/canvas/api/frame",
            {},
            origin=origin,
        )
        assert no_csrf.status == 403

        # Security headers are present on errors as well as successes.
        assert no_csrf.headers["Cache-Control"] == "no-store"
        assert no_csrf.headers["Referrer-Policy"] == "no-referrer"
        assert no_csrf.headers["X-Content-Type-Options"] == "nosniff"
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_selection_change_revokes_old_loads_and_shutdown_revokes_sessions() -> (
    None
):
    gateway = CanvasGateway(authority=_Authority([]))
    launch = await gateway.open_shell(_scope())
    replacement = await gateway.open_shell(_scope(revision_id="revision-replaced"))
    origin = gateway.origin
    assert origin is not None
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        bootstrap = launch.browser_url.split("#boot=", 1)[1]
        stale_boot = await _post_json(
            session,
            f"{origin}/canvas/api/boot",
            {"bootstrap": bootstrap},
            origin=origin,
        )
        assert stale_boot.status == 401
        bootstrap = replacement.browser_url.split("#boot=", 1)[1]
        boot = await _post_json(
            session,
            f"{origin}/canvas/api/boot",
            {"bootstrap": bootstrap},
            origin=origin,
        )
        body = await boot.json()
        frame = await _post_json(
            session,
            f"{origin}/canvas/api/frame",
            {},
            origin=origin,
            csrf=body["csrf"],
        )
        frame_url = (await frame.json())["renderer_url"]
        gateway.change_selection(
            browser_session_id=body["browser_session_id"],
            scope=_scope(revision_id="revision-b"),
        )
        old = await session.get(
            f"{origin}{frame_url}",
            headers={"Sec-Fetch-Dest": "iframe", "Sec-Fetch-Site": "same-origin"},
        )
        assert old.status == 401
        await gateway.aclose()
        assert gateway.browser_session_count == 0


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_console_runtime_owns_one_lazy_gateway_and_disposes_it() -> None:
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    runtime = ConsoleRuntime(object())
    authority = _Authority([])

    first = runtime.ensure_canvas_gateway(authority=authority)
    second = runtime.ensure_canvas_gateway(authority=authority)
    assert first is second
    assert first.started is False

    await first.start()
    await runtime.dispose()
    assert first.started is False
