from unittest.mock import patch, MagicMock
import httpx
import pytest

from tldw_chatbook.Local_Ingestion.web_article_ingestion import (
    extract_article_for_ingest,
)
from tldw_chatbook.Local_Ingestion.local_file_ingestion import PermanentIngestError
from tldw_chatbook.Utils import egress


def _allow_public_dns(monkeypatch, ip="93.184.216.34"):
    """Guarded fetch re-validates the target host via real DNS on every hop;
    pin resolution to a public IP so these tests exercise the mocked
    transport deterministically, without depending on the test
    environment's network access."""
    monkeypatch.setattr(egress, "_resolve", lambda host: [ip])


def _handler_for(body, *, status=200, ctype="text/html; charset=utf-8"):
    """MockTransport handler returning a fixed response for any request."""

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": ctype} if ctype else {}
        return httpx.Response(status, headers=headers, content=body, request=request)

    return handler


_RealClient = httpx.Client  # captured before `httpx.Client` gets patched below


def _client_returning(handler):
    # The extractor now fetches via guarded_fetch_httpx, which drives the
    # real httpx request/redirect machinery (client.build_request /
    # client.send) instead of `client.stream(...)`. Faking the transport
    # (rather than mocking `.stream`) lets that real machinery run against a
    # fake handler.
    def _make(**kwargs):
        return _RealClient(
            transport=httpx.MockTransport(handler),
            timeout=kwargs.get("timeout", 30.0),
            headers=kwargs.get("headers"),
        )

    return _make


def test_extract_success_shape(monkeypatch):
    _allow_public_dns(monkeypatch)
    html = (
        "<html><head><title>Hello</title></head><body><article><p>"
        + ("word " * 80)
        + "</p></article></body></html>"
    )
    with patch("httpx.Client", side_effect=_client_returning(_handler_for(html))):
        result = extract_article_for_ingest("https://example.com/post?utm_source=x", {})
    assert "word word" in result["content"]
    assert result["url"] == "https://example.com/post"  # canonical, tracking stripped
    assert result["title"]
    assert result["chunks"] == [] and result["analysis"] == ""


def test_non_html_content_type_is_permanent(monkeypatch):
    _allow_public_dns(monkeypatch)
    with patch(
        "httpx.Client",
        side_effect=_client_returning(_handler_for("%PDF-1.4", ctype="application/pdf")),
    ):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/x.pdf", {})


def test_empty_extraction_is_permanent(monkeypatch):
    _allow_public_dns(monkeypatch)
    html = "<html><body><script>window.__NEXT_DATA__={}</script></body></html>"
    # trafilatura returns None on a JS shell
    with (
        patch("httpx.Client", side_effect=_client_returning(_handler_for(html))),
        patch("trafilatura.extract", return_value=None),
    ):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/spa", {})


def test_permanent_on_4xx(monkeypatch):
    _allow_public_dns(monkeypatch)
    with patch(
        "httpx.Client", side_effect=_client_returning(_handler_for("nope", status=404))
    ):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/missing", {})


def test_retryable_on_503(monkeypatch):
    _allow_public_dns(monkeypatch)
    with patch(
        "httpx.Client", side_effect=_client_returning(_handler_for("busy", status=503))
    ):
        with pytest.raises(Exception) as ei:
            extract_article_for_ingest("https://example.com/busy", {})
        assert not isinstance(ei.value, PermanentIngestError)  # retryable


def test_missing_trafilatura_is_permanent(monkeypatch):
    _allow_public_dns(monkeypatch)
    import builtins

    real_import = builtins.__import__

    def _fail(name, *a, **k):
        if name == "trafilatura":
            raise ImportError("no trafilatura")
        return real_import(name, *a, **k)

    html = "<html><body><p>x</p></body></html>"
    with (
        patch("httpx.Client", side_effect=_client_returning(_handler_for(html))),
        patch.object(builtins, "__import__", _fail),
    ):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/a", {})


def test_oversized_streamed_body_is_permanent_before_full_buffer(monkeypatch):
    # Two 8 MB chunks: the guard must abort after the SECOND chunk crosses the
    # 10 MB cap without draining the (unbounded) remainder.
    _allow_public_dns(monkeypatch)
    big = b"x" * (8 * 1024 * 1024)
    drained = {"chunks": 0}

    def _gen():
        for _ in range(1000):  # far more than the cap would allow
            drained["chunks"] += 1
            yield big

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=_gen(),
            request=request,
        )

    with patch("httpx.Client", side_effect=_client_returning(handler)):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/huge", {})
    assert drained["chunks"] == 2  # aborted at the 2nd chunk, did not drain 1000


def test_invalid_url_is_permanent(monkeypatch):
    # Not an httpx.HTTPError subclass -- raised by the real client when it
    # builds the request for a syntactically malformed URL. Equivalent swap
    # for the old `client.stream.side_effect` mock: guarded_fetch_httpx
    # calls `client.build_request(...)` rather than `client.stream(...)`.
    _allow_public_dns(monkeypatch)
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.build_request.side_effect = httpx.InvalidURL("bad")
    with patch("httpx.Client", return_value=client):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://exa mple.com/x", {})


def test_unsupported_protocol_is_permanent():
    # The egress guard's scheme check rejects non-http(s) URLs before any
    # client/DNS interaction, so no client mock or DNS patch is needed here.
    with pytest.raises(PermanentIngestError):
        extract_article_for_ingest("ftp://example.com/x", {})


def test_dns_failure_is_permanent(monkeypatch):
    import socket

    def _boom(host):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(egress, "_resolve", _boom)
    with pytest.raises(PermanentIngestError):
        extract_article_for_ingest("https://no-such-host.invalid/x", {})


def test_connection_error_without_dns_cause_is_retryable(monkeypatch):
    _allow_public_dns(monkeypatch)
    err = httpx.ConnectError("connection refused")  # no gaierror cause
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.send.side_effect = err
    with patch("httpx.Client", return_value=client):
        with pytest.raises(Exception) as ei:
            extract_article_for_ingest("https://example.com/x", {})
    assert not isinstance(ei.value, PermanentIngestError)  # retryable transport error


def test_blocked_url_is_permanent(monkeypatch):
    monkeypatch.setattr(egress, "_resolve", lambda host: ["169.254.169.254"])
    with pytest.raises(PermanentIngestError, match="egress"):
        extract_article_for_ingest("http://metadata-ish.example/x", {})


def test_malformed_url_is_permanent():
    """Malformed URLs (e.g., invalid IPv6) raise PermanentIngestError, not ValueError.

    This regression test verifies that malformed URLs that urlparse.hostname
    would reject with ValueError are caught and converted to PermanentIngestError
    via the origin_set helper's safe handling.
    """
    # This URL has an invalid IPv6 literal that would raise ValueError.
    # Before the fix, this would crash with an unhandled ValueError.
    # After the fix, it raises PermanentIngestError (via egress guard).
    with pytest.raises(PermanentIngestError, match="egress"):
        extract_article_for_ingest("http://[::1/x", {})
