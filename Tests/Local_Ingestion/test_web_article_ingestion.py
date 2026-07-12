from unittest.mock import patch, MagicMock
import pytest

from tldw_chatbook.Local_Ingestion.web_article_ingestion import extract_article_for_ingest
from tldw_chatbook.Local_Ingestion.local_file_ingestion import PermanentIngestError


def _resp(html, *, status=200, ctype="text/html; charset=utf-8", final_url=None):
    r = MagicMock()
    r.status_code = status
    r.headers = {"content-type": ctype, "content-length": str(len(html))}
    r.url = final_url or "https://example.com/post"
    r.text = html
    r.iter_bytes = lambda chunk_size=65536: iter([html.encode("utf-8")])
    r.raise_for_status = MagicMock()
    return r


def _client_returning(resp):
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.get.return_value = resp
    return client


def test_extract_success_shape(monkeypatch):
    html = "<html><head><title>Hello</title></head><body><article><p>" + ("word " * 80) + "</p></article></body></html>"
    with patch("httpx.Client", return_value=_client_returning(_resp(html, final_url="https://example.com/post?utm_source=x"))):
        result = extract_article_for_ingest("https://example.com/post?utm_source=x", {})
    assert "word word" in result["content"]
    assert result["url"] == "https://example.com/post"            # canonical, tracking stripped
    assert result["title"]
    assert result["chunks"] == [] and result["analysis"] == ""


def test_non_html_content_type_is_permanent():
    with patch("httpx.Client", return_value=_client_returning(_resp("%PDF-1.4", ctype="application/pdf"))):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/x.pdf", {})


def test_empty_extraction_is_permanent(monkeypatch):
    html = "<html><body><script>window.__NEXT_DATA__={}</script></body></html>"
    # trafilatura returns None on a JS shell
    with patch("httpx.Client", return_value=_client_returning(_resp(html))), \
         patch("trafilatura.extract", return_value=None):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/spa", {})


def test_permanent_on_4xx():
    resp = _resp("nope", status=404)
    import httpx
    resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock(status_code=404)))
    with patch("httpx.Client", return_value=_client_returning(resp)):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/missing", {})


def test_retryable_on_503():
    resp = _resp("busy", status=503)
    import httpx
    resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock(status_code=503)))
    with patch("httpx.Client", return_value=_client_returning(resp)):
        with pytest.raises(Exception) as ei:
            extract_article_for_ingest("https://example.com/busy", {})
        assert not isinstance(ei.value, PermanentIngestError)     # retryable


def test_missing_trafilatura_is_permanent(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def _fail(name, *a, **k):
        if name == "trafilatura":
            raise ImportError("no trafilatura")
        return real_import(name, *a, **k)
    html = "<html><body><p>x</p></body></html>"
    with patch("httpx.Client", return_value=_client_returning(_resp(html))), \
         patch.object(builtins, "__import__", _fail):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/a", {})
