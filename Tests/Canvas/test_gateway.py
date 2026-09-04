"""Native Canvas gateway route, security, and lifecycle tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from urllib.parse import urlsplit

import aiohttp
import pytest
from aiohttp import web

from tldw_chatbook.Canvas.capabilities import (
    CanvasCapabilityError,
    CanvasCapabilityScope,
    CanvasCapabilityStore,
)
from tldw_chatbook.Canvas.compiler import compile_canvas_document
from tldw_chatbook.Canvas.gateway import (
    BridgeConfirmationRequest,
    BridgeConfirmationResponse,
    CanvasGateway,
    CanvasGatewayEvent,
    CanvasGatewayLaunch,
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
        self,
        scope: CanvasGatewayScope,
        request: BridgeConfirmationRequest,
        *,
        settlement: object | None = None,
    ) -> BridgeConfirmationResponse:
        self.calls.append(("bridge", scope))
        if settlement is not None:
            assert settlement.try_settle(lambda: None)  # type: ignore[attr-defined]
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


def _launch_url(launch: CanvasGatewayLaunch, relative: str) -> str:
    return f"{launch.clean_url}{relative.lstrip('/')}"


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


async def _ready_bridge(
    session: aiohttp.ClientSession,
    gateway: CanvasGateway,
    launch: CanvasGatewayLaunch,
) -> tuple[str, str, str]:
    """Boot one shell, load one frame, and mint one bridge capability."""

    assert gateway.origin is not None
    origin = gateway.origin
    boot = await _post_json(
        session,
        _launch_url(launch, "api/boot"),
        {"bootstrap": launch.browser_url.split("#boot=", 1)[1]},
        origin=origin,
    )
    assert boot.status == 200
    csrf = (await boot.json())["csrf"]
    frame = await _post_json(
        session,
        _launch_url(launch, "api/frame"),
        {},
        origin=origin,
        csrf=csrf,
    )
    assert frame.status == 200
    grant = await _post_json(
        session,
        _launch_url(launch, "api/actions"),
        {"action": "bridge_confirm"},
        origin=origin,
        csrf=csrf,
    )
    assert grant.status == 200
    return origin, csrf, (await grant.json())["capability"]


async def _fresh_bridge_capability(
    session: aiohttp.ClientSession,
    launch: CanvasGatewayLaunch,
    *,
    origin: str,
    csrf: str,
) -> str:
    grant = await _post_json(
        session,
        _launch_url(launch, "api/actions"),
        {"action": "bridge_confirm"},
        origin=origin,
        csrf=csrf,
    )
    assert grant.status == 200
    return (await grant.json())["capability"]


def _bridge_request(
    *,
    request_id: str = "request-idempotent",
    kind: str = "submit",
    value: object = "bounded",
) -> dict[str, object]:
    return {
        "approved": True,
        "request": {
            "version": "canvas-v1",
            "request_id": request_id,
            "kind": kind,
            "value": value,
        },
    }


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


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_unreachable_browser_keeps_events_but_revokes_all_old_authority() -> None:
    class UnavailableAuthority(_Authority):
        async def read_events(self, scope, *, after_event_id):
            del after_event_id
            return (
                CanvasGatewayEvent(
                    "event-unavailable",
                    "disconnected",
                    scope.canvas_id,
                    scope.revision_id,
                    {"notice": "unavailable_on_branch"},
                ),
            )

    gateway = CanvasGateway(authority=UnavailableAuthority([]))
    launch = await gateway.open_shell(_scope())
    assert gateway.origin is not None
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, old_bridge = await _ready_bridge(session, gateway, launch)
        gateway.mark_browser_session_unavailable("browser-a")

        events = await session.get(_launch_url(launch, "api/events"))
        assert events.status == 200
        assert (await events.json())["events"][0]["kind"] == "disconnected"
        assert gateway.has_browser_session_for("conversation-session-a") is False
        assert (await session.get(_launch_url(launch, "api/state"))).status == 401
        actions = await _post_json(
            session,
            _launch_url(launch, "api/actions"),
            {"action": "source_read"},
            origin=origin,
            csrf=csrf,
        )
        assert actions.status == 403
        replay = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            _bridge_request(request_id="stale-after-branch"),
            origin=origin,
            csrf=csrf,
            capability=old_bridge,
        )
        assert replay.status == 403
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_unreachable_pending_shell_cannot_boot_after_branch_switch() -> None:
    gateway = CanvasGateway(authority=_Authority([]))
    launch = await gateway.open_shell(_scope())
    assert gateway.origin is not None

    gateway.mark_browser_session_unavailable("browser-a")

    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        boot = await _post_json(
            session,
            _launch_url(launch, "api/boot"),
            {"bootstrap": launch.browser_url.split("#boot=", 1)[1]},
            origin=gateway.origin,
        )
        assert boot.status == 401
    await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
async def test_gateway_cleanup_retains_runner_until_a_retry_settles(
    failure_type: type[BaseException],
) -> None:
    class FlakyRunner:
        def __init__(self) -> None:
            self.attempts = 0

        async def cleanup(self) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise failure_type()

    gateway = CanvasGateway(authority=_Authority([]))
    runner = FlakyRunner()
    gateway._runner = runner
    gateway._origin = "http://127.0.0.1:1"

    with pytest.raises(failure_type):
        await gateway.aclose()
    assert gateway._runner is runner
    assert gateway._origin == "http://127.0.0.1:1"

    await gateway.aclose()
    assert runner.attempts == 2
    assert gateway._runner is None
    assert gateway.origin is None


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_gateway_start_retries_unsettled_cleanup_before_rebinding() -> None:
    class FlakyRunner:
        def __init__(self) -> None:
            self.attempts = 0

        async def cleanup(self) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("private-cleanup-detail")

    gateway = CanvasGateway(authority=_Authority([]))
    runner = FlakyRunner()
    gateway._runner = runner

    with pytest.raises(RuntimeError, match="could not start") as exc_info:
        await gateway.start()
    assert "private-cleanup-detail" not in str(exc_info.value)
    assert gateway._runner is runner

    origin = await gateway.start()
    assert origin.startswith("http://127.0.0.1:")
    assert runner.attempts == 2
    await gateway.aclose()


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
    assert launch.clean_url.startswith(f"{gateway.origin}/canvas/gateway-")
    assert "#" not in launch.clean_url
    assert gateway.start_count == 1
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_two_shells_in_one_browser_keep_distinct_cookie_authority() -> None:
    authority = _Authority([])
    gateway = CanvasGateway(authority=authority)
    first = await gateway.open_shell(_scope())
    second = await gateway.open_shell(_scope(browser_session_id="browser-b"))
    assert first.clean_url != second.clean_url
    assert gateway.origin is not None
    origin = gateway.origin
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        first_bootstrap = first.browser_url.split("#boot=", 1)[1]
        crossed = await _post_json(
            session,
            _launch_url(second, "api/boot"),
            {"bootstrap": first_bootstrap},
            origin=origin,
        )
        assert crossed.status == 401
        first_boot = await _post_json(
            session,
            _launch_url(first, "api/boot"),
            {"bootstrap": first_bootstrap},
            origin=origin,
        )
        assert first_boot.status == 200
        second_boot = await _post_json(
            session,
            _launch_url(second, "api/boot"),
            {"bootstrap": second.browser_url.split("#boot=", 1)[1]},
            origin=origin,
        )
        assert second_boot.status == 200

        assert (await session.get(_launch_url(first, "api/events"))).status == 200
        assert (await session.get(_launch_url(second, "api/events"))).status == 200
    assert gateway.browser_session_count == 2
    assert gateway.has_browser_session_for("conversation-session-a") is True
    assert gateway.has_browser_session_for("conversation-session-b") is False
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_two_loopback_gateways_do_not_overwrite_browser_cookies() -> None:
    first_gateway = CanvasGateway(authority=_Authority([]))
    second_gateway = CanvasGateway(authority=_Authority([]))
    first = await first_gateway.open_shell(_scope())
    second = await second_gateway.open_shell(
        _scope(browser_session_id="browser-b", canvas_id="canvas-b")
    )
    assert first_gateway.origin is not None
    assert second_gateway.origin is not None
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        for gateway, launch in (
            (first_gateway, first),
            (second_gateway, second),
        ):
            response = await _post_json(
                session,
                _launch_url(launch, "api/boot"),
                {"bootstrap": launch.browser_url.split("#boot=", 1)[1]},
                origin=gateway.origin,
            )
            assert response.status == 200

        assert (await session.get(_launch_url(first, "api/events"))).status == 200
        assert (await session.get(_launch_url(second, "api/events"))).status == 200
    await first_gateway.aclose()
    await second_gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_expired_shell_binding_is_purged_before_shell_admission() -> None:
    now = [100.0]
    gateway = CanvasGateway(
        authority=_Authority([]),
        clock=lambda: now[0],
        max_shell_bindings=1,
    )
    first = await gateway.open_shell(_scope())
    now[0] += 31.0

    second = await gateway.open_shell(_scope(browser_session_id="browser-b"))

    assert first.clean_url != second.clean_url
    assert len(gateway._shell_bindings) == 1
    async with aiohttp.ClientSession() as session:
        assert (await session.get(first.clean_url)).status == 404
        assert (await session.get(second.clean_url)).status == 200
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_shell_admission_is_capped_without_revoking_unrelated_shell() -> None:
    gateway = CanvasGateway(authority=_Authority([]), max_shell_bindings=1)
    first = await gateway.open_shell(_scope())

    with pytest.raises(CanvasCapabilityError, match="shell capacity"):
        await gateway.open_shell(_scope(browser_session_id="browser-b"))

    assert len(gateway._shell_bindings) == 1
    async with aiohttp.ClientSession() as session:
        assert (await session.get(first.clean_url)).status == 200
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_full_capability_store_cannot_leak_pending_shell_bindings() -> None:
    gateway = CanvasGateway(authority=_Authority([]), max_shell_bindings=2)
    gateway.capabilities = CanvasCapabilityStore(max_active=1)
    gateway.capabilities.issue(
        CanvasCapabilityScope(
            browser_session_id="capacity-owner",
            load_id="capacity-load",
            conversation_session_id="capacity-conversation",
            canvas_id="capacity-canvas",
            revision_id="capacity-revision",
            action="shell_boot",
        ),
        ttl_seconds=30,
    )

    for attempt in range(8):
        with pytest.raises(CanvasCapabilityError, match="capacity"):
            await gateway.open_shell(
                _scope(browser_session_id=f"failed-browser-{attempt}")
            )

    assert gateway._shell_bindings == {}
    assert gateway.capabilities.active_count == 1
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
            _launch_url(launch, "api/boot"),
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
            _launch_url(launch, "api/frame"),
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

        forced_plan = await session.get(
            _launch_url(launch, "api/plan"),
            headers={"Sec-Fetch-Dest": "image", "Sec-Fetch-Site": "same-site"},
        )
        assert forced_plan.status == 403

        # A refused cross-context request does not consume the plan capability.
        plan = await session.get(
            _launch_url(launch, "api/plan"),
            headers={"Sec-Fetch-Dest": "empty", "Sec-Fetch-Site": "same-origin"},
        )
        assert plan.status == 200
        plan_body = await plan.json()
        assert plan_body["runtime_profile"] == "canvas-v1"
        assert "source" not in plan_body
        assert (
            plan_body["root"]["children"][1]["children"][0]["children"][0]["text"]
            == "Exact plan"
        )

        asset = await session.get(_launch_url(launch, "static/canvas_renderer.js"))
        assert asset.status == 200
        assert asset.content_type == "text/javascript"
        assert asset.headers["Access-Control-Allow-Origin"] == "*"
        assert asset.headers["Cross-Origin-Resource-Policy"] == "cross-origin"
        assert 'crossorigin="anonymous"' in await renderer.text()
        missing = await session.get(_launch_url(launch, "static/not-packaged.js"))
        assert missing.status == 404

        exact_method = await session.head(launch.clean_url)
        assert exact_method.status == 405
        assert exact_method.headers["Cache-Control"] == "no-store"

        events = await session.get(_launch_url(launch, "api/events"))
        assert events.status == 200
        assert (await events.json())["events"][0]["revision_id"] == "revision-a"

        source_grant = await _post_json(
            session,
            _launch_url(launch, "api/actions"),
            {"action": "source_read"},
            origin=origin,
            csrf=csrf,
        )
        source_token = (await source_grant.json())["capability"]
        source = await session.get(
            _launch_url(launch, "api/source"),
            headers={"Authorization": f"CanvasCapability {source_token}"},
        )
        assert source.status == 200
        assert source.content_type == "text/plain"
        assert await source.text() == "<!doctype html><title>Exact source</title>"
        assert (
            await session.get(
                _launch_url(launch, "api/source"),
                headers={"Authorization": f"CanvasCapability {source_token}"},
            )
        ).status == 401

        bridge_grant = await _post_json(
            session,
            _launch_url(launch, "api/actions"),
            {"action": "bridge_confirm"},
            origin=origin,
            csrf=csrf,
        )
        bridge_token = (await bridge_grant.json())["capability"]
        bridge = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
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
            _launch_url(launch, "api/close"),
            {},
            origin=origin,
            csrf=csrf,
        )
        assert closed.status == 200
        assert gateway.browser_session_count == 0
        assert (await session.get(_launch_url(launch, "api/events"))).status == 401

    assert authority.calls == [
        ("plan", _scope()),
        ("events", _scope()),
        ("source", _scope()),
        ("bridge", _scope()),
    ]
    assert not any(
        path.resource.canonical.endswith("/api/conversations")
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
            _launch_url(launch, "api/boot"),
            {"bootstrap": bootstrap},
            origin=origin,
        )
        assert boot.status == 200
        events = await session.get(_launch_url(launch, "api/events"))
        assert events.status == 503
        assert "canvas-sibling" not in await events.text()
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_boot_refuses_new_browser_session_at_capacity() -> None:
    gateway = CanvasGateway(authority=_Authority([]), max_browser_sessions=1)
    first = await gateway.open_shell(_scope())
    second = await gateway.open_shell(_scope(browser_session_id="browser-b"))
    assert gateway.origin is not None
    origin = gateway.origin
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        for launch, expected_status in ((first, 200), (second, 503)):
            bootstrap = launch.browser_url.split("#boot=", 1)[1]
            response = await _post_json(
                session,
                _launch_url(launch, "api/boot"),
                {"bootstrap": bootstrap},
                origin=origin,
            )
            assert response.status == expected_status
    assert gateway.browser_session_count == 1
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_boot_purges_expired_browser_session_before_capacity_check() -> None:
    now = 100.0
    gateway = CanvasGateway(
        authority=_Authority([]),
        max_browser_sessions=1,
        clock=lambda: now,
    )
    first = await gateway.open_shell(_scope())
    assert gateway.origin is not None
    origin = gateway.origin
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        first_bootstrap = first.browser_url.split("#boot=", 1)[1]
        first_boot = await _post_json(
            session,
            _launch_url(first, "api/boot"),
            {"bootstrap": first_bootstrap},
            origin=origin,
        )
        assert first_boot.status == 200

        now += 30 * 60
        second = await gateway.open_shell(_scope(browser_session_id="browser-b"))
        second_bootstrap = second.browser_url.split("#boot=", 1)[1]
        second_boot = await _post_json(
            session,
            _launch_url(second, "api/boot"),
            {"bootstrap": second_bootstrap},
            origin=origin,
        )
        assert second_boot.status == 200
    assert gateway.browser_session_count == 1
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
        for query in (
            "boot=secret",
            "bootstrap=secret",
            "capability=secret",
            "token=secret",
            "ACCESS_TOKEN=secret",
            "Secret=secret",
            "CsRf=secret",
            "CANVAS_SESSION=secret",
            "Canvas_Frame=secret",
            "canvas_PLAN=secret",
        ):
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
            _launch_url(launch, "api/boot"),
            {"bootstrap": bootstrap},
            origin="http://127.0.0.1:1",
        )
        assert wrong_origin.status == 403
        wrong_type = await session.post(
            _launch_url(launch, "api/boot"),
            data="{}",
            headers={"Origin": origin, "Content-Type": "text/plain"},
        )
        assert wrong_type.status == 415
        oversized = await session.post(
            _launch_url(launch, "api/boot"),
            data="x" * 257,
            headers={"Origin": origin, "Content-Type": "application/json"},
        )
        assert oversized.status == 413

        boot = await _post_json(
            session,
            _launch_url(launch, "api/boot"),
            {"bootstrap": bootstrap},
            origin=origin,
        )
        assert boot.status == 200
        no_csrf = await _post_json(
            session,
            _launch_url(launch, "api/frame"),
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
            _launch_url(launch, "api/boot"),
            {"bootstrap": bootstrap},
            origin=origin,
        )
        assert stale_boot.status == 401
        bootstrap = replacement.browser_url.split("#boot=", 1)[1]
        boot = await _post_json(
            session,
            _launch_url(replacement, "api/boot"),
            {"bootstrap": bootstrap},
            origin=origin,
        )
        body = await boot.json()
        frame = await _post_json(
            session,
            _launch_url(replacement, "api/frame"),
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
@pytest.mark.parametrize(
    "transition", ["reload", "close", "branch", "same-revision-selection"]
)
async def test_bridge_settlement_prevents_host_effect_after_scope_race(
    transition: str,
) -> None:
    class RacingAuthority(_Authority):
        def __init__(self) -> None:
            super().__init__([])
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.effects: list[str] = []

        async def confirm_bridge(
            self,
            scope: CanvasGatewayScope,
            request: BridgeConfirmationRequest,
            *,
            settlement: object | None = None,
        ) -> BridgeConfirmationResponse:
            self.entered.set()
            await self.release.wait()
            if settlement is None:
                self.effects.append(request.request.request_id)
                settled = True
            else:
                settled = settlement.try_settle(  # type: ignore[attr-defined]
                    lambda: self.effects.append(request.request.request_id)
                )
            return BridgeConfirmationResponse(
                request_id=request.request.request_id,
                status="confirmed" if settled else "refused",
            )

    authority = RacingAuthority()
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope())
    assert gateway.origin is not None
    origin = gateway.origin
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        boot = await _post_json(
            session,
            _launch_url(launch, "api/boot"),
            {"bootstrap": launch.browser_url.split("#boot=", 1)[1]},
            origin=origin,
        )
        boot_body = await boot.json()
        csrf = boot_body["csrf"]
        assert (
            await _post_json(
                session,
                _launch_url(launch, "api/frame"),
                {},
                origin=origin,
                csrf=csrf,
            )
        ).status == 200
        grant = await _post_json(
            session,
            _launch_url(launch, "api/actions"),
            {"action": "bridge_confirm"},
            origin=origin,
            csrf=csrf,
        )
        bridge_token = (await grant.json())["capability"]
        bridge_task = asyncio.create_task(
            _post_json(
                session,
                _launch_url(launch, "api/bridge"),
                {
                    "approved": True,
                    "request": {
                        "version": "canvas-v1",
                        "request_id": "request-race",
                        "kind": "submit",
                        "value": "bounded",
                    },
                },
                origin=origin,
                csrf=csrf,
                capability=bridge_token,
            )
        )
        await asyncio.wait_for(authority.entered.wait(), timeout=2)

        if transition == "reload":
            changed = await _post_json(
                session,
                _launch_url(launch, "api/frame"),
                {},
                origin=origin,
                csrf=csrf,
            )
            assert changed.status == 200
        elif transition == "close":
            changed = await _post_json(
                session,
                _launch_url(launch, "api/close"),
                {},
                origin=origin,
                csrf=csrf,
            )
            assert changed.status == 200
        elif transition == "branch":
            gateway.change_selection(
                browser_session_id="browser-a",
                scope=_scope(revision_id="revision-b"),
            )
        else:
            gateway.change_selection(
                browser_session_id="browser-a",
                scope=_scope(),
            )

        authority.release.set()
        bridge = await bridge_task
        assert bridge.status == 409
        assert authority.effects == []
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [("refused", 200), ("invalid", 503), ("exception", 503)],
)
async def test_bridge_callback_finalizes_an_unused_settlement_lease(
    outcome: str,
    expected_status: int,
) -> None:
    class RetainingAuthority(_Authority):
        lease: object | None = None

        async def confirm_bridge(
            self,
            scope: CanvasGatewayScope,
            request: BridgeConfirmationRequest,
            *,
            settlement: object | None = None,
        ) -> object:
            self.lease = settlement
            if outcome == "refused":
                return BridgeConfirmationResponse(
                    request_id=request.request.request_id,
                    status="refused",
                )
            if outcome == "invalid":
                return object()
            raise RuntimeError("private-authority-detail")

    authority = RetainingAuthority([])
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope())
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, capability = await _ready_bridge(session, gateway, launch)
        response = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            {
                "approved": True,
                "request": {
                    "version": "canvas-v1",
                    "request_id": "request-retained",
                    "kind": "submit",
                    "value": "bounded",
                },
            },
            origin=origin,
            csrf=csrf,
            capability=capability,
        )
        assert response.status == expected_status
        assert "private-authority-detail" not in await response.text()

    effects: list[str] = []
    assert authority.lease is not None
    assert not authority.lease.try_settle(  # type: ignore[attr-defined]
        lambda: effects.append("late-effect")
    )
    assert effects == []
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transition", ["reload", "close", "branch", "same-revision-selection", "error"]
)
async def test_committed_bridge_settlement_returns_exact_result_after_transition(
    transition: str,
) -> None:
    class CommittingAuthority(_Authority):
        def __init__(self) -> None:
            super().__init__([])
            self.settled = asyncio.Event()
            self.release = asyncio.Event()
            self.effects: list[str] = []

        async def confirm_bridge(
            self,
            scope: CanvasGatewayScope,
            request: BridgeConfirmationRequest,
            *,
            settlement: object | None = None,
        ) -> BridgeConfirmationResponse:
            assert settlement is not None
            assert settlement.try_settle(  # type: ignore[attr-defined]
                lambda: self.effects.append(request.request.request_id)
            )
            self.settled.set()
            await self.release.wait()
            if transition == "error":
                raise RuntimeError("post-commit-private-detail")
            return BridgeConfirmationResponse(
                request_id="wrong-result-id",
                status="refused",
            )

    authority = CommittingAuthority()
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope())
    request_body = {
        "approved": True,
        "request": {
            "version": "canvas-v1",
            "request_id": "request-committed",
            "kind": "submit",
            "value": "bounded",
        },
    }
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, capability = await _ready_bridge(session, gateway, launch)
        bridge_task = asyncio.create_task(
            _post_json(
                session,
                _launch_url(launch, "api/bridge"),
                request_body,
                origin=origin,
                csrf=csrf,
                capability=capability,
            )
        )
        await asyncio.wait_for(authority.settled.wait(), timeout=2)

        if transition == "reload":
            assert (
                await _post_json(
                    session,
                    _launch_url(launch, "api/frame"),
                    {},
                    origin=origin,
                    csrf=csrf,
                )
            ).status == 200
        elif transition == "close":
            assert (
                await _post_json(
                    session,
                    _launch_url(launch, "api/close"),
                    {},
                    origin=origin,
                    csrf=csrf,
                )
            ).status == 200
        elif transition == "branch":
            gateway.change_selection(
                browser_session_id="browser-a",
                scope=_scope(revision_id="revision-b"),
            )
        elif transition == "same-revision-selection":
            gateway.change_selection(
                browser_session_id="browser-a",
                scope=_scope(),
            )

        authority.release.set()
        response = await bridge_task
        assert response.status == 200
        assert await response.json() == {
            "request_id": "request-committed",
            "status": "confirmed",
        }

        # Losing the transport response cannot authorize a duplicate: replay
        # with the consumed grant never repeats the exact request ID's effect.
        replay = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            request_body,
            origin=origin,
            csrf=csrf,
            capability=capability,
        )
        assert replay.status in {401, 403, 409}
        assert authority.effects == ["request-committed"]
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_fresh_capability_replays_committed_bridge_without_authority() -> None:
    class CountingAuthority(_Authority):
        def __init__(self) -> None:
            super().__init__([])
            self.bridge_calls = 0
            self.effects: list[str] = []

        async def confirm_bridge(
            self,
            scope: CanvasGatewayScope,
            request: BridgeConfirmationRequest,
            *,
            settlement: object | None = None,
        ) -> BridgeConfirmationResponse:
            self.bridge_calls += 1
            assert settlement is not None
            assert settlement.try_settle(  # type: ignore[attr-defined]
                lambda: self.effects.append(request.request.request_id)
            )
            return BridgeConfirmationResponse(
                request_id=request.request.request_id,
                status="confirmed",
            )

    authority = CountingAuthority()
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope())
    body = _bridge_request(value={"alpha": "sensitive-payload", "beta": 2})
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, first_capability = await _ready_bridge(session, gateway, launch)
        first = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            body,
            origin=origin,
            csrf=csrf,
            capability=first_capability,
        )
        replay = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            _bridge_request(value={"beta": 2, "alpha": "sensitive-payload"}),
            origin=origin,
            csrf=csrf,
            capability=await _fresh_bridge_capability(
                session, launch, origin=origin, csrf=csrf
            ),
        )

        assert first.status == replay.status == 200
        assert await replay.json() == {
            "request_id": "request-idempotent",
            "status": "confirmed",
        }
        assert authority.bridge_calls == 1
        assert authority.effects == ["request-idempotent"]
        assert gateway.bridge_settlement_count == 1
        record = next(
            iter(next(iter(gateway._sessions.values())).bridge_settlements.values())
        )
        assert "sensitive-payload" not in repr(record)
        assert record.payload_digest.hex() not in repr(record)
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_cancelled_transport_after_settlement_replays_without_duplicate() -> None:
    class DelayedAuthority(_Authority):
        def __init__(self) -> None:
            super().__init__([])
            self.bridge_calls = 0
            self.settled = asyncio.Event()
            self.release = asyncio.Event()
            self.effects: list[str] = []

        async def confirm_bridge(
            self,
            scope: CanvasGatewayScope,
            request: BridgeConfirmationRequest,
            *,
            settlement: object | None = None,
        ) -> BridgeConfirmationResponse:
            self.bridge_calls += 1
            assert settlement is not None
            assert settlement.try_settle(  # type: ignore[attr-defined]
                lambda: self.effects.append(request.request.request_id)
            )
            self.settled.set()
            if self.bridge_calls == 1:
                await self.release.wait()
            return BridgeConfirmationResponse(
                request_id=request.request.request_id,
                status="confirmed",
            )

    authority = DelayedAuthority()
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope())
    body = _bridge_request(request_id="request-transport-loss")
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, capability = await _ready_bridge(session, gateway, launch)
        lost = asyncio.create_task(
            _post_json(
                session,
                _launch_url(launch, "api/bridge"),
                body,
                origin=origin,
                csrf=csrf,
                capability=capability,
            )
        )
        await asyncio.wait_for(authority.settled.wait(), timeout=2)
        lost.cancel()
        with pytest.raises(asyncio.CancelledError):
            await lost

        replay = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            body,
            origin=origin,
            csrf=csrf,
            capability=await _fresh_bridge_capability(
                session, launch, origin=origin, csrf=csrf
            ),
        )
        assert replay.status == 200
        assert await replay.json() == {
            "request_id": "request-transport-loss",
            "status": "confirmed",
        }
        assert authority.bridge_calls == 1
        assert authority.effects == ["request-transport-loss"]
        authority.release.set()
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changed_kind", "changed_value"),
    [("submit", "changed-payload"), ("download", "bounded")],
)
async def test_same_bridge_request_id_with_changed_content_is_refused(
    changed_kind: str,
    changed_value: object,
) -> None:
    authority = _Authority([])
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope())
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, capability = await _ready_bridge(session, gateway, launch)
        first = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            _bridge_request(),
            origin=origin,
            csrf=csrf,
            capability=capability,
        )
        collision = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            _bridge_request(kind=changed_kind, value=changed_value),
            origin=origin,
            csrf=csrf,
            capability=await _fresh_bridge_capability(
                session, launch, origin=origin, csrf=csrf
            ),
        )
        assert first.status == 200
        assert collision.status == 409
        assert authority.calls.count(("bridge", _scope())) == 1
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_bridge_settlement_capacity_expiry_and_load_revocation() -> None:
    now = [100.0]
    authority = _Authority([])
    gateway = CanvasGateway(
        authority=authority,
        clock=lambda: now[0],
        max_bridge_settlements=1,
    )
    launch = await gateway.open_shell(_scope())
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, capability = await _ready_bridge(session, gateway, launch)
        first = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            _bridge_request(request_id="request-capacity-a"),
            origin=origin,
            csrf=csrf,
            capability=capability,
        )
        assert first.status == 200
        assert gateway.bridge_settlement_count == 1

        full = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            _bridge_request(request_id="request-capacity-b"),
            origin=origin,
            csrf=csrf,
            capability=await _fresh_bridge_capability(
                session, launch, origin=origin, csrf=csrf
            ),
        )
        assert full.status == 503
        assert authority.calls.count(("bridge", _scope())) == 1

        now[0] += 301.0
        after_expiry = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            _bridge_request(request_id="request-capacity-b"),
            origin=origin,
            csrf=csrf,
            capability=await _fresh_bridge_capability(
                session, launch, origin=origin, csrf=csrf
            ),
        )
        assert after_expiry.status == 200
        assert gateway.bridge_settlement_count == 1

        reloaded = await _post_json(
            session,
            _launch_url(launch, "api/frame"),
            {},
            origin=origin,
            csrf=csrf,
        )
        assert reloaded.status == 200
        assert gateway.bridge_settlement_count == 0

        gateway.change_selection(browser_session_id="browser-a", scope=_scope())
        assert gateway.bridge_settlement_count == 0
        closed = await _post_json(
            session,
            _launch_url(launch, "api/close"),
            {},
            origin=origin,
            csrf=csrf,
        )
        assert closed.status == 200
        assert gateway.bridge_settlement_count == 0
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["reload", "selection", "close", "shutdown"])
async def test_bridge_settlements_revoke_with_authority_lifecycle(
    transition: str,
) -> None:
    gateway = CanvasGateway(authority=_Authority([]))
    launch = await gateway.open_shell(_scope())
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, capability = await _ready_bridge(session, gateway, launch)
        committed = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            _bridge_request(request_id="request-revoke"),
            origin=origin,
            csrf=csrf,
            capability=capability,
        )
        assert committed.status == 200
        assert gateway.bridge_settlement_count == 1

        if transition == "reload":
            changed = await _post_json(
                session,
                _launch_url(launch, "api/frame"),
                {},
                origin=origin,
                csrf=csrf,
            )
            assert changed.status == 200
        elif transition == "selection":
            gateway.change_selection(browser_session_id="browser-a", scope=_scope())
        elif transition == "close":
            changed = await _post_json(
                session,
                _launch_url(launch, "api/close"),
                {},
                origin=origin,
                csrf=csrf,
            )
            assert changed.status == 200
        else:
            await gateway.aclose()
        assert gateway.bridge_settlement_count == 0
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_concurrent_fresh_bridge_retries_share_one_authority_settlement() -> None:
    class CoordinatedAuthority(_Authority):
        def __init__(self) -> None:
            super().__init__([])
            self.bridge_calls = 0
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.effects: list[str] = []

        async def confirm_bridge(
            self,
            scope: CanvasGatewayScope,
            request: BridgeConfirmationRequest,
            *,
            settlement: object | None = None,
        ) -> BridgeConfirmationResponse:
            self.bridge_calls += 1
            self.entered.set()
            await self.release.wait()
            assert settlement is not None
            assert settlement.try_settle(  # type: ignore[attr-defined]
                lambda: self.effects.append(request.request.request_id)
            )
            return BridgeConfirmationResponse(
                request_id=request.request.request_id,
                status="confirmed",
            )

    authority = CoordinatedAuthority()
    gateway = CanvasGateway(authority=authority)
    launch = await gateway.open_shell(_scope())
    body = _bridge_request(request_id="request-concurrent")
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, first = await _ready_bridge(session, gateway, launch)
        capabilities = [first]
        for _ in range(5):
            capabilities.append(
                await _fresh_bridge_capability(
                    session, launch, origin=origin, csrf=csrf
                )
            )
        requests = [
            asyncio.create_task(
                _post_json(
                    session,
                    _launch_url(launch, "api/bridge"),
                    body,
                    origin=origin,
                    csrf=csrf,
                    capability=capability,
                )
            )
            for capability in capabilities
        ]
        await asyncio.wait_for(authority.entered.wait(), timeout=2)
        await asyncio.sleep(0)
        authority.release.set()
        responses = await asyncio.gather(*requests)

        assert [response.status for response in responses] == [200] * 6
        assert [await response.json() for response in responses] == [
            {"request_id": "request-concurrent", "status": "confirmed"}
        ] * 6
        assert authority.bridge_calls == 1
        assert authority.effects == ["request-concurrent"]
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_hung_bridge_owner_expires_pending_record_and_wakes_joiners() -> None:
    class HangingAuthority(_Authority):
        def __init__(self) -> None:
            super().__init__([])
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.record: object | None = None
            self.late_settled: bool | None = None
            self.effects: list[str] = []

        async def confirm_bridge(
            self,
            scope: CanvasGatewayScope,
            request: BridgeConfirmationRequest,
            *,
            settlement: object | None = None,
        ) -> BridgeConfirmationResponse:
            assert settlement is not None
            self.record = settlement._record  # type: ignore[attr-defined]
            self.entered.set()
            await self.release.wait()
            self.late_settled = settlement.try_settle(  # type: ignore[attr-defined]
                lambda: self.effects.append(request.request.request_id)
            )
            return BridgeConfirmationResponse(
                request_id=request.request.request_id,
                status="confirmed" if self.late_settled else "refused",
            )

    authority = HangingAuthority()
    gateway = CanvasGateway(
        authority=authority,
        bridge_settlement_ttl_seconds=0.15,
        max_bridge_waiters=3,
    )
    launch = await gateway.open_shell(_scope())
    body = _bridge_request(request_id="request-hung")
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, owner_capability = await _ready_bridge(session, gateway, launch)
        owner = asyncio.create_task(
            _post_json(
                session,
                _launch_url(launch, "api/bridge"),
                body,
                origin=origin,
                csrf=csrf,
                capability=owner_capability,
            )
        )
        await asyncio.wait_for(authority.entered.wait(), timeout=2)
        joiners = [
            asyncio.create_task(
                _post_json(
                    session,
                    _launch_url(launch, "api/bridge"),
                    body,
                    origin=origin,
                    csrf=csrf,
                    capability=await _fresh_bridge_capability(
                        session, launch, origin=origin, csrf=csrf
                    ),
                )
            )
            for _ in range(3)
        ]
        joined = await asyncio.wait_for(asyncio.gather(*joiners), timeout=2)
        assert [response.status for response in joined] == [503, 503, 503]

        live_session = next(iter(gateway._sessions.values()))
        assert live_session.bridge_settlements == {}
        assert authority.record is not None
        assert authority.record.expiry_handle is None  # type: ignore[attr-defined]

        authority.release.set()
        late = await asyncio.wait_for(owner, timeout=2)
        assert late.status == 409
        assert authority.late_settled is False
        assert authority.effects == []
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_bridge_pending_waiter_cap_refuses_excess_and_releases_on_completion() -> (
    None
):
    class CoordinatedAuthority(_Authority):
        def __init__(self) -> None:
            super().__init__([])
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.effects: list[str] = []

        async def confirm_bridge(
            self,
            scope: CanvasGatewayScope,
            request: BridgeConfirmationRequest,
            *,
            settlement: object | None = None,
        ) -> BridgeConfirmationResponse:
            self.entered.set()
            await self.release.wait()
            assert settlement is not None
            assert settlement.try_settle(  # type: ignore[attr-defined]
                lambda: self.effects.append(request.request.request_id)
            )
            return BridgeConfirmationResponse(
                request_id=request.request.request_id,
                status="confirmed",
            )

    authority = CoordinatedAuthority()
    gateway = CanvasGateway(
        authority=authority,
        bridge_settlement_ttl_seconds=5,
        max_bridge_waiters=2,
    )
    launch = await gateway.open_shell(_scope())
    body = _bridge_request(request_id="request-waiter-cap")
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, owner_capability = await _ready_bridge(session, gateway, launch)
        owner = asyncio.create_task(
            _post_json(
                session,
                _launch_url(launch, "api/bridge"),
                body,
                origin=origin,
                csrf=csrf,
                capability=owner_capability,
            )
        )
        await asyncio.wait_for(authority.entered.wait(), timeout=2)

        async def start_joiner() -> asyncio.Task[aiohttp.ClientResponse]:
            capability = await _fresh_bridge_capability(
                session, launch, origin=origin, csrf=csrf
            )
            return asyncio.create_task(
                _post_json(
                    session,
                    _launch_url(launch, "api/bridge"),
                    body,
                    origin=origin,
                    csrf=csrf,
                    capability=capability,
                )
            )

        first_joiner = await start_joiner()
        second_joiner = await start_joiner()
        record = next(
            iter(next(iter(gateway._sessions.values())).bridge_settlements.values())
        )
        for _ in range(20):
            if record.waiter_count == 2:
                break
            await asyncio.sleep(0)
        assert record.waiter_count == 2

        excess = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            body,
            origin=origin,
            csrf=csrf,
            capability=await _fresh_bridge_capability(
                session, launch, origin=origin, csrf=csrf
            ),
        )
        assert excess.status == 503
        assert (await excess.json())["error"] == "bridge_waiter_capacity"

        authority.release.set()
        finished = await asyncio.gather(owner, first_joiner, second_joiner)
        assert [response.status for response in finished] == [200, 200, 200]
        assert record.waiter_count == 0
        assert authority.effects == ["request-waiter-cap"]
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["reload", "selection", "close", "shutdown"])
async def test_pending_bridge_joiner_wakes_on_lifecycle_revocation(
    transition: str,
) -> None:
    class HangingAuthority(_Authority):
        def __init__(self) -> None:
            super().__init__([])
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def confirm_bridge(
            self,
            scope: CanvasGatewayScope,
            request: BridgeConfirmationRequest,
            *,
            settlement: object | None = None,
        ) -> BridgeConfirmationResponse:
            self.entered.set()
            await self.release.wait()
            assert settlement is not None
            settled = settlement.try_settle(lambda: None)  # type: ignore[attr-defined]
            return BridgeConfirmationResponse(
                request_id=request.request.request_id,
                status="confirmed" if settled else "refused",
            )

    authority = HangingAuthority()
    gateway = CanvasGateway(authority=authority, bridge_settlement_ttl_seconds=5)
    launch = await gateway.open_shell(_scope())
    shutdown: asyncio.Task[None] | None = None
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, owner_capability = await _ready_bridge(session, gateway, launch)
        body = _bridge_request(request_id=f"request-pending-{transition}")
        owner = asyncio.create_task(
            _post_json(
                session,
                _launch_url(launch, "api/bridge"),
                body,
                origin=origin,
                csrf=csrf,
                capability=owner_capability,
            )
        )
        await asyncio.wait_for(authority.entered.wait(), timeout=2)
        joiner = asyncio.create_task(
            _post_json(
                session,
                _launch_url(launch, "api/bridge"),
                body,
                origin=origin,
                csrf=csrf,
                capability=await _fresh_bridge_capability(
                    session, launch, origin=origin, csrf=csrf
                ),
            )
        )
        live_session = next(iter(gateway._sessions.values()))
        record = next(iter(live_session.bridge_settlements.values()))
        for _ in range(20):
            if record.waiter_count == 1:
                break
            await asyncio.sleep(0)
        assert record.waiter_count == 1

        if transition == "reload":
            assert (
                await _post_json(
                    session,
                    _launch_url(launch, "api/frame"),
                    {},
                    origin=origin,
                    csrf=csrf,
                )
            ).status == 200
        elif transition == "selection":
            gateway.change_selection(browser_session_id="browser-a", scope=_scope())
        elif transition == "close":
            assert (
                await _post_json(
                    session,
                    _launch_url(launch, "api/close"),
                    {},
                    origin=origin,
                    csrf=csrf,
                )
            ).status == 200
        else:
            shutdown = asyncio.create_task(gateway.aclose())

        joined = await asyncio.wait_for(joiner, timeout=2)
        assert joined.status == 503
        assert record.waiter_count == 0
        assert record.expiry_handle is None
        assert record.terminal_reason == "revoked"

        authority.release.set()
        completed_owner = await asyncio.wait_for(owner, timeout=2)
        assert completed_owner.status == 409
        if shutdown is not None:
            await asyncio.wait_for(shutdown, timeout=2)
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_committed_bridge_record_self_expires_without_later_request() -> None:
    gateway = CanvasGateway(
        authority=_Authority([]),
        bridge_settlement_ttl_seconds=0.05,
    )
    launch = await gateway.open_shell(_scope())
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, capability = await _ready_bridge(session, gateway, launch)
        committed = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            _bridge_request(request_id="request-self-expiry"),
            origin=origin,
            csrf=csrf,
            capability=capability,
        )
        assert committed.status == 200
        live_session = next(iter(gateway._sessions.values()))
        record = next(iter(live_session.bridge_settlements.values()))
        assert record.expiry_handle is not None

        for _ in range(40):
            if not live_session.bridge_settlements:
                break
            await asyncio.sleep(0.01)
        assert live_session.bridge_settlements == {}
        assert record.expiry_handle is None
    await gateway.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["reload", "selection", "close", "shutdown"])
async def test_bridge_expiry_handles_settle_on_completion_and_lifecycle(
    transition: str,
) -> None:
    class InspectingAuthority(_Authority):
        pending_handle: object | None = None
        record: object | None = None

        async def confirm_bridge(
            self,
            scope: CanvasGatewayScope,
            request: BridgeConfirmationRequest,
            *,
            settlement: object | None = None,
        ) -> BridgeConfirmationResponse:
            assert settlement is not None
            self.record = settlement._record  # type: ignore[attr-defined]
            self.pending_handle = self.record.expiry_handle  # type: ignore[attr-defined]
            assert settlement.try_settle(lambda: None)  # type: ignore[attr-defined]
            return BridgeConfirmationResponse(
                request_id=request.request.request_id,
                status="confirmed",
            )

    authority = InspectingAuthority([])
    gateway = CanvasGateway(
        authority=authority,
        bridge_settlement_ttl_seconds=5,
    )
    launch = await gateway.open_shell(_scope())
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as session:
        origin, csrf, capability = await _ready_bridge(session, gateway, launch)
        committed = await _post_json(
            session,
            _launch_url(launch, "api/bridge"),
            _bridge_request(request_id="request-handle-lifecycle"),
            origin=origin,
            csrf=csrf,
            capability=capability,
        )
        assert committed.status == 200
        assert authority.pending_handle is not None
        assert authority.pending_handle.cancelled()  # type: ignore[attr-defined]
        assert authority.record is not None
        assert authority.record.expiry_handle is not None  # type: ignore[attr-defined]

        if transition == "reload":
            changed = await _post_json(
                session,
                _launch_url(launch, "api/frame"),
                {},
                origin=origin,
                csrf=csrf,
            )
            assert changed.status == 200
        elif transition == "selection":
            gateway.change_selection(browser_session_id="browser-a", scope=_scope())
        elif transition == "close":
            changed = await _post_json(
                session,
                _launch_url(launch, "api/close"),
                {},
                origin=origin,
                csrf=csrf,
            )
            assert changed.status == 200
        else:
            await gateway.aclose()
        assert authority.record.expiry_handle is None  # type: ignore[attr-defined]
    await gateway.aclose()


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
    with pytest.raises(ValueError, match="different authority"):
        runtime.ensure_canvas_gateway(authority=_Authority([]))

    await first.start()
    await runtime.dispose()
    assert first.started is False


@pytest.mark.asyncio
async def test_console_runtime_stops_canvas_admission_before_authority_teardown() -> (
    None
):
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    order: list[str] = []

    class CanvasGatewayDouble:
        async def aclose(self) -> None:
            order.append("canvas-gateway")

    class ControllerDouble:
        async def shutdown(self) -> None:
            order.append("controller")

    runtime = ConsoleRuntime(object())
    runtime._canvas_gateway = CanvasGatewayDouble()
    runtime._chat_controller = ControllerDouble()

    await runtime.dispose()

    assert order[:2] == ["canvas-gateway", "controller"]
