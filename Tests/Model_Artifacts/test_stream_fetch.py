"""TASK-595 Task 3: streaming guarded fetch."""

from urllib.parse import urlparse

import httpx
import pytest

from Tests.Model_Artifacts.fixture_http import FixtureArtifactServer
from tldw_chatbook.Model_Artifacts.fetch import (
    FetchRestartRequired,
    FetchTooLargeError,
    FetchValidators,
    stream_fetch,
)

BODY = b"0123456789" * 1000  # 10 KB


def _trusted(srv: FixtureArtifactServer) -> frozenset:
    """Trusted-origins set for a fixture server, in egress's real format.

    ``tldw_chatbook.Utils.egress._normalize_trusted`` / ``_post_resolution``
    key membership on the bare, lowercased HOSTNAME (e.g. ``"127.0.0.1"``),
    not a scheme+host+port URL string -- confirmed by reading
    ``egress._pre_resolution``/``_post_resolution`` and by the convention
    already used across ``Tests/Image_Generation/test_http_client.py`` and
    ``Tests/Utils/test_egress.py`` (``frozenset({"127.0.0.1"})``). The
    fixture server binds to the loopback IP literal, which classifies as
    "private" under ``_classify_ip`` and would otherwise be egress-blocked;
    listing it here is what lets policy allow it.
    """
    return frozenset({urlparse(srv.url("/")).hostname})


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


def test_strip_headers_mirror_matches_egress():
    """Drift guard: our local strip tuple must equal egress's."""
    from tldw_chatbook.Utils import egress
    from tldw_chatbook.Model_Artifacts import fetch

    assert set(fetch._STRIP_HEADERS) == set(egress._STRIP_HEADERS)
