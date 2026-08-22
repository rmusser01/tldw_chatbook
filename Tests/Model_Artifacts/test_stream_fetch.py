"""TASK-595 Task 3: streaming guarded fetch."""

import httpx
import pytest

from Tests.Model_Artifacts.acquisition_test_helpers import _trusted
from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer
from tldw_chatbook.Model_Artifacts.fetch import (
    FetchRestartRequired,
    FetchTooLargeError,
    FetchTransportError,
    FetchValidators,
    stream_fetch,
)

# Network opt-in (task-15111): this module fetches from
# `FixtureArtifactServer`, an in-process HTTP server on an ephemeral
# loopback port.
# The autouse guard in Tests/conftest.py denies egress by default; every address
# these tests reach is a port this process itself is listening on.
pytestmark = pytest.mark.allow_network

BODY = b"0123456789" * 1000  # 10 KB


@pytest.mark.asyncio
async def test_full_fetch_writes_and_reports(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY)
        dest = tmp_path / "f.bin"
        async with httpx.AsyncClient() as client:
            result = await stream_fetch(
                srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY) + 10,
                trusted_origins=_trusted(srv),
            )
    assert dest.read_bytes() == BODY
    assert result.bytes_written == len(BODY)
    assert result.resumed is False
    assert result.validators.etag == '"v1"'


@pytest.mark.asyncio
async def test_resume_uses_range_and_appends(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY)
        dest = tmp_path / "f.bin"
        dest.write_bytes(BODY[:4000])
        async with httpx.AsyncClient() as client:
            result = await stream_fetch(
                srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                resume_from=4000,
                validators=FetchValidators(etag='"v1"', last_modified=None),
                trusted_origins=_trusted(srv),
            )
    assert dest.read_bytes() == BODY
    assert result.resumed is True
    assert result.bytes_written == len(BODY) - 4000
    # The server actually saw a Range request.
    assert any("Range" in r for r in srv.requests["/f.bin"])


@pytest.mark.asyncio
async def test_changed_validator_raises_restart(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY, etag='"v2"')
        dest = tmp_path / "f.bin"
        dest.write_bytes(BODY[:100])
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchRestartRequired):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                    resume_from=100,
                    validators=FetchValidators(etag='"v1"', last_modified=None),
                    trusted_origins=_trusted(srv),
                )


@pytest.mark.asyncio
async def test_weak_etag_never_resumes(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY, weak_etag=True)
        dest = tmp_path / "f.bin"
        dest.write_bytes(BODY[:100])
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchRestartRequired):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                    resume_from=100,
                    validators=FetchValidators(etag='W/"v1"', last_modified=None),
                    trusted_origins=_trusted(srv),
                )


@pytest.mark.asyncio
async def test_no_range_support_raises_restart(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY, support_range=False)
        dest = tmp_path / "f.bin"
        dest.write_bytes(BODY[:100])
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchRestartRequired):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                    resume_from=100,
                    validators=FetchValidators(etag='"v1"', last_modified=None),
                    trusted_origins=_trusted(srv),
                )


@pytest.mark.asyncio
async def test_last_modified_only_resume_succeeds(tmp_path):
    """FetchValidators.strong allows etag=None + last_modified set; resume
    must actually use THAT validator in If-Range, not silently skip it."""
    lm = 'Wed, 21 Oct 2015 07:28:00 GMT'
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY, etag=None, last_modified=lm)
        dest = tmp_path / "f.bin"
        dest.write_bytes(BODY[:4000])
        async with httpx.AsyncClient() as client:
            result = await stream_fetch(
                srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                resume_from=4000,
                validators=FetchValidators(etag=None, last_modified=lm),
                trusted_origins=_trusted(srv),
            )
    assert dest.read_bytes() == BODY
    assert result.resumed is True
    assert result.bytes_written == len(BODY) - 4000
    assert any("If-Range" in r for r in srv.requests["/f.bin"])


@pytest.mark.asyncio
async def test_changed_last_modified_raises_restart_without_append(tmp_path):
    """Upstream changed (new Last-Modified) while resuming Last-Modified-only:
    a spec-compliant server answers the mismatched If-Range with a full 200,
    which must raise FetchRestartRequired -- and the destination must NOT
    have had any (mismatched) bytes appended."""
    with FixtureArtifactServer() as srv:
        srv.serve(
            "/f.bin", BODY, etag=None,
            last_modified="Thu, 22 Oct 2015 07:28:00 GMT",
        )
        dest = tmp_path / "f.bin"
        seed = BODY[:100]
        dest.write_bytes(seed)
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchRestartRequired):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                    resume_from=100,
                    validators=FetchValidators(
                        etag=None, last_modified="Wed, 21 Oct 2015 07:28:00 GMT"
                    ),
                    trusted_origins=_trusted(srv),
                )
    assert dest.read_bytes() == seed


@pytest.mark.asyncio
async def test_changed_last_modified_non_compliant_server_raises_restart(tmp_path):
    """A server that IGNORES If-Range entirely (still answers 206 even on a
    stale conditional) is a real, and for date-based conditionals especially
    under-implemented, failure mode -- distinct from the compliant-200 path
    covered by ``test_changed_last_modified_raises_restart_without_append``.
    The post-response Last-Modified comparison (not the status-code check)
    is what must catch this one."""
    with FixtureArtifactServer() as srv:
        srv.serve(
            "/f.bin", BODY, etag=None,
            last_modified="Thu, 22 Oct 2015 07:28:00 GMT",
            ignore_if_range=True,
        )
        dest = tmp_path / "f.bin"
        seed = BODY[:100]
        dest.write_bytes(seed)
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchRestartRequired):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                    resume_from=100,
                    validators=FetchValidators(
                        etag=None, last_modified="Wed, 21 Oct 2015 07:28:00 GMT"
                    ),
                    trusted_origins=_trusted(srv),
                )
    assert dest.read_bytes() == seed


@pytest.mark.asyncio
async def test_missing_etag_on_resume_raises_restart_without_append(tmp_path):
    """A 206 that OMITS ETag entirely must not be accepted as matching a
    saved strong ETag -- a missing validator is treated as a mismatch, not
    as "no information, assume it's fine". ``ignore_if_range=True`` models
    a non-compliant server that honors ``Range`` unconditionally (206)
    even though its route serves no ``ETag`` header at all (this is the
    only way to force the fixture to answer 206 while the response itself
    carries no ETag -- see FixtureArtifactServer.serve's ``ignore_if_range``
    docstring)."""
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY, etag=None, ignore_if_range=True)
        dest = tmp_path / "f.bin"
        seed = BODY[:100]
        dest.write_bytes(seed)
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchRestartRequired):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                    resume_from=100,
                    validators=FetchValidators(etag='"v1"', last_modified=None),
                    trusted_origins=_trusted(srv),
                )
    assert dest.read_bytes() == seed


@pytest.mark.asyncio
async def test_content_range_start_mismatch_raises_restart_without_append(tmp_path):
    """A 206 alone does not prove the body starts at ``resume_from`` --
    only ``Content-Range`` does. A server whose ``Content-Range`` header
    reports a DIFFERENT start than the ``Range`` request asked for must be
    rejected before the destination is ever opened for append."""
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY, bad_range_start=999)
        dest = tmp_path / "f.bin"
        seed = BODY[:100]
        dest.write_bytes(seed)
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchRestartRequired):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                    resume_from=100,
                    validators=FetchValidators(etag='"v1"', last_modified=None),
                    trusted_origins=_trusted(srv),
                )
    assert dest.read_bytes() == seed


@pytest.mark.asyncio
async def test_missing_content_range_on_resume_raises_restart_without_append(tmp_path):
    """A non-compliant server that answers 206 without ANY Content-Range
    header at all must be rejected the same way as a mismatched one --
    there is no header to trust, so resuming must not be assumed safe."""
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY, omit_content_range=True)
        dest = tmp_path / "f.bin"
        seed = BODY[:100]
        dest.write_bytes(seed)
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchRestartRequired):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=len(BODY),
                    resume_from=100,
                    validators=FetchValidators(etag='"v1"', last_modified=None),
                    trusted_origins=_trusted(srv),
                )
    assert dest.read_bytes() == seed


@pytest.mark.asyncio
async def test_max_bytes_bounds_final_size(tmp_path):
    with FixtureArtifactServer() as srv:
        srv.serve("/f.bin", BODY)
        dest = tmp_path / "f.bin"
        async with httpx.AsyncClient() as client:
            with pytest.raises(FetchTooLargeError):
                await stream_fetch(
                    srv.url("/f.bin"), dest, client=client, max_bytes=100,
                    trusted_origins=_trusted(srv),
                )


def test_cross_origin_header_policy_is_shared_not_mirrored():
    """Drift guard, task-19733: the mirror is gone; the policy is imported.

    This module used to keep its own copy of egress's strip tuple, and this
    test compared the two SETS -- which only detects drift after someone
    edits one side and runs the suite. It now asserts the stronger property:
    there is one object, so divergence is not expressible. The
    ``hasattr`` assertion is the anti-regression -- re-introducing a local
    ``_STRIP_HEADERS`` copy fails here.
    """
    from tldw_chatbook.Utils import egress
    from tldw_chatbook.Model_Artifacts import fetch

    assert fetch.CROSS_ORIGIN_SAFE_HEADERS is egress.CROSS_ORIGIN_SAFE_HEADERS
    assert not hasattr(fetch, "_STRIP_HEADERS")


@pytest.mark.asyncio
async def test_cross_origin_hop_keeps_range_but_drops_custom_credential(
    tmp_path, monkeypatch
):
    """Catalog -> CDN is the normal artifact download, and it is cross-origin.

    Two things must hold on that second hop at once (task-19733):
    ``Range``/``If-Range`` MUST survive or resume silently stops working,
    while a credential under a name nobody denylisted (``X-Artifact-Token``)
    must NOT. A rule that only drops four known names fails the second half;
    a rule that drops everything fails the first.

    Args:
        tmp_path: pytest fixture -- destination directory for the download.
        monkeypatch: pytest fixture -- neutralises the egress DNS check so
            this stays a transport-only test.
    """
    seen: list[httpx.Request] = []
    body = b"tail-bytes-after-resume"

    async def allow_egress(_url: str, *, trusted_origins: frozenset[str]) -> None:
        """Keep this transport-only test independent of DNS/egress policy.

        Args:
            _url: The URL ``stream_fetch`` is about to fetch; ignored.
            trusted_origins: Passed through by the caller; ignored.
        """

    def handler(request: httpx.Request) -> httpx.Response:
        """Record every request, then answer catalog with a redirect to the CDN.

        Args:
            request: The request the MockTransport was handed.

        Returns:
            A 302 for the catalog origin, a 206 partial response otherwise.
        """
        seen.append(request)
        if request.url.host == "catalog.example":
            return httpx.Response(
                302,
                headers={"Location": "https://cdn.example/model.gguf"},
                request=request,
            )
        return httpx.Response(
            206,
            headers={
                "content-range": f"bytes 100-{100 + len(body) - 1}/{100 + len(body)}",
                "etag": '"v1"',
            },
            content=body,
            request=request,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Model_Artifacts.fetch.check_url_or_raise_async", allow_egress
    )
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"x" * 100)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await stream_fetch(
            "https://catalog.example/model.gguf",
            dest,
            client=client,
            max_bytes=100 + len(body),
            resume_from=100,
            validators=FetchValidators(etag='"v1"', last_modified=None),
            headers={"X-Artifact-Token": "sentinel-not-a-real-key-19733"},
        )

    assert len(seen) == 2
    first, second = seen
    assert first.headers.get("x-artifact-token") == "sentinel-not-a-real-key-19733"
    assert "cdn.example" in str(second.url)
    assert "x-artifact-token" not in second.headers
    assert second.headers.get("range") == "bytes=100-"
    assert second.headers.get("if-range") == '"v1"'
    assert dest.read_bytes() == b"x" * 100 + body


@pytest.mark.asyncio
async def test_https_redirect_downgrade_stops_before_second_request(tmp_path, monkeypatch):
    """An HTTPS-to-HTTP redirect must fail before the HTTP hop is requested.

    This catches a missing downgrade guard: without it, the redirect loop
    makes a second request to the HTTP storage URL below.
    """
    requests: list[str] = []

    async def allow_egress(_url: str, *, trusted_origins: frozenset[str]) -> None:
        """Keep this transport-only test independent of DNS/egress policy."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url == httpx.URL("https://catalog.example/model.gguf"):
            return httpx.Response(
                302,
                headers={"Location": "http://storage.example/model.gguf"},
                request=request,
            )
        pytest.fail(f"downgraded redirect made a second request: {request.url}")

    monkeypatch.setattr(
        "tldw_chatbook.Model_Artifacts.fetch.check_url_or_raise_async", allow_egress
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FetchTransportError, match="HTTPS redirect downgrade"):
            await stream_fetch(
                "https://catalog.example/model.gguf",
                tmp_path / "model.gguf",
                client=client,
                max_bytes=1024,
            )

    assert requests == ["https://catalog.example/model.gguf"]


@pytest.mark.asyncio
async def test_client_level_auth_not_applied_on_cross_origin_hop(
    tmp_path, monkeypatch
):
    """A client-level ``auth=`` must not follow the catalog -> CDN hop.

    Independent review of task-19733: header stripping cannot reach this
    route because httpx applies a client-level ``auth`` inside ``send()``,
    after ``build_request`` produced the request the guard filtered. The
    same-origin first hop must still authenticate.

    Args:
        tmp_path: pytest fixture -- destination directory for the download.
        monkeypatch: pytest fixture -- neutralises the egress DNS check so
            this stays a transport-only test.
    """
    seen: list[httpx.Request] = []
    body = b"cdn-bytes"

    async def allow_egress(_url: str, *, trusted_origins: frozenset[str]) -> None:
        """Keep this transport-only test independent of DNS/egress policy.

        Args:
            _url: The URL ``stream_fetch`` is about to fetch; ignored.
            trusted_origins: Passed through by the caller; ignored.
        """

    def handler(request: httpx.Request) -> httpx.Response:
        """Record every request, then answer catalog with a redirect to the CDN.

        Args:
            request: The request the MockTransport was handed.

        Returns:
            A 302 for the catalog origin, the artifact bytes otherwise.
        """
        seen.append(request)
        if request.url.host == "catalog.example":
            return httpx.Response(
                302,
                headers={"Location": "https://cdn.example/model.gguf"},
                request=request,
            )
        return httpx.Response(200, content=body, request=request)

    monkeypatch.setattr(
        "tldw_chatbook.Model_Artifacts.fetch.check_url_or_raise_async", allow_egress
    )
    dest = tmp_path / "model.gguf"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=("alice", "sentinel-not-a-real-key-19733"),
    ) as client:
        await stream_fetch(
            "https://catalog.example/model.gguf",
            dest,
            client=client,
            max_bytes=len(body),
        )

    assert len(seen) == 2
    assert "authorization" in seen[0].headers
    assert "authorization" not in seen[1].headers
    assert dest.read_bytes() == body
