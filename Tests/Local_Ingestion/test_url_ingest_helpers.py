import pytest

from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    classify_ingest_source, canonicalize_url, PermanentIngestError,
)
from tldw_chatbook.Local_Ingestion.ingest_parse_worker import classify_parse_failure


@pytest.mark.parametrize("url,expected", [
    ("https://youtube.com/watch?v=abc", "video"),
    ("https://youtu.be/abc", "video"),
    ("https://vimeo.com/123", "video"),
    ("https://cdn.example.com/clip.mp4", "video"),
    ("https://cdn.example.com/talk.mp3", "audio"),
    ("https://example.com/blog/post", "article"),
    ("http://example.com/a", "article"),
])
def test_classify_urls(url, expected):
    assert classify_ingest_source(url) == expected


def test_classify_files_use_detect_file_type():
    assert classify_ingest_source("/tmp/a.pdf") == "pdf"
    assert classify_ingest_source("/tmp/a.mp3") == "audio"


def test_classify_non_http_scheme_is_not_url():
    # a file:// or bare path must NOT be treated as an http URL
    with pytest.raises(Exception):
        classify_ingest_source("file:///etc/passwd")   # no known extension -> detect_file_type raises


def test_canonicalize_url_strips_tracking_and_normalizes():
    got = canonicalize_url("HTTPS://Example.com:443/Path/?utm_source=x&b=2&a=1#frag")
    assert got == "https://example.com/Path?a=1&b=2"
    assert canonicalize_url("https://example.com/") == "https://example.com/"
    assert canonicalize_url("https://example.com/p/") == "https://example.com/p"


def test_classify_parse_failure_permanent_ingest_error():
    assert classify_parse_failure(PermanentIngestError("bad url")) is True
    assert classify_parse_failure(RuntimeError("transient")) is False
