import io
import pytest
from PIL import Image


def _png_bytes(size=(8, 8)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def ifu():
    from tldw_chatbook.Image_Generation.adapters import image_format_utils as m
    return m


@pytest.fixture(autouse=True)
def _policy_env(monkeypatch):
    """Deterministic egress policy for hostname-based test URLs (no real DNS)."""
    from tldw_chatbook.Utils import egress

    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])

    async def _resolve_async(host):
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", _resolve_async)
    monkeypatch.setattr(egress, "get_cli_setting", lambda s, k=None, d=None: d)


def test_format_from_bytes_detects_png(ifu):
    assert ifu.format_from_bytes(_png_bytes()) == "png"


def test_validate_and_convert_output_roundtrip(ifu):
    data, ctype = ifu.validate_and_convert_image_output(_png_bytes(), "image/png", "png", max_bytes=10_000_000)
    assert ctype == "image/png" and isinstance(data, (bytes, bytearray))


def test_validate_rejects_when_over_max_bytes(ifu):
    with pytest.raises(Exception):
        ifu.validate_and_convert_image_output(_png_bytes((256, 256)), "image/png", "png", max_bytes=10)


class _FakeStreamResponse:
    def __init__(self, status_code, headers, url, body=b""):
        self.status_code = status_code
        self.headers = headers
        self.url = url
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_bytes(self):
        yield self._body


def test_fetch_image_bytes_strips_credentials_on_cross_origin_redirect(monkeypatch, ifu):
    from tldw_chatbook.Image_Generation import http_client as hc

    seen = []

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stream(self, method, url, *, headers=None, cookies=None, timeout=None, follow_redirects=False):
            seen.append((url, dict(headers or {}), cookies))
            if url == "https://api.example.com/img":
                return _FakeStreamResponse(302, {"location": "https://attacker.example/img2"}, url)
            return _FakeStreamResponse(200, {"content-type": "image/png"}, url, body=_png_bytes())

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    content, ctype = ifu.fetch_image_bytes(
        "https://api.example.com/img",
        timeout=5,
        headers={"Authorization": "Bearer secret", "X-Other": "keep"},
        cookies={"session": "abc"},
        trusted_origins=frozenset({"api.example.com"}),
    )
    assert ctype == "image/png"
    assert content
    assert len(seen) == 2
    first_url, first_headers, first_cookies = seen[0]
    assert first_headers.get("Authorization") == "Bearer secret"
    assert first_cookies == {"session": "abc"}
    second_url, second_headers, second_cookies = seen[1]
    assert second_url == "https://attacker.example/img2"
    assert "Authorization" not in second_headers
    assert second_headers.get("X-Other") == "keep"
    assert second_cookies is None


def test_fetch_image_bytes_keeps_credentials_on_same_origin_redirect(monkeypatch, ifu):
    from tldw_chatbook.Image_Generation import http_client as hc

    seen = []

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stream(self, method, url, *, headers=None, cookies=None, timeout=None, follow_redirects=False):
            seen.append((url, dict(headers or {}), cookies))
            if url == "http://127.0.0.1:7801/img":
                return _FakeStreamResponse(302, {"location": "http://127.0.0.1:7801/img2"}, url)
            return _FakeStreamResponse(200, {"content-type": "image/png"}, url, body=_png_bytes())

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    content, ctype = ifu.fetch_image_bytes(
        "http://127.0.0.1:7801/img",
        timeout=5,
        headers={"Authorization": "Bearer local"},
        cookies={"swarm_token": "tok"},
        trusted_origins=frozenset({"127.0.0.1"}),
    )
    assert content
    assert len(seen) == 2
    assert all(h.get("Authorization") == "Bearer local" for _u, h, _c in seen)
    assert all(c == {"swarm_token": "tok"} for _u, _h, c in seen)


def test_fetch_image_bytes_strips_credentials_on_same_host_scheme_downgrade(monkeypatch, ifu):
    """fetch_image_bytes drops credentials on a same-host HTTPS->HTTP downgrade hop (task-568)."""
    from tldw_chatbook.Image_Generation import http_client as hc

    seen = []

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stream(self, method, url, *, headers=None, cookies=None, timeout=None, follow_redirects=False):
            seen.append((url, dict(headers or {}), cookies))
            if url == "https://127.0.0.1:7801/img":
                return _FakeStreamResponse(302, {"location": "http://127.0.0.1:7801/img2"}, url)
            return _FakeStreamResponse(200, {"content-type": "image/png"}, url, body=_png_bytes())

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    content, ctype = ifu.fetch_image_bytes(
        "https://127.0.0.1:7801/img",
        timeout=5,
        headers={"Authorization": "Bearer local"},
        cookies={"swarm_token": "tok"},
        trusted_origins=frozenset({"127.0.0.1"}),
    )
    assert content
    assert len(seen) == 2
    first_url, first_headers, first_cookies = seen[0]
    assert first_headers.get("Authorization") == "Bearer local"
    assert first_cookies == {"swarm_token": "tok"}
    second_url, second_headers, second_cookies = seen[1]
    assert second_url == "http://127.0.0.1:7801/img2"
    assert "Authorization" not in second_headers
    assert second_cookies is None


def test_fetch_image_bytes_strips_credentials_on_same_host_different_port(monkeypatch, ifu):
    """A same-host different-port hop crosses an origin boundary; credentials strip (task-568)."""
    from tldw_chatbook.Image_Generation import http_client as hc

    seen = []

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stream(self, method, url, *, headers=None, cookies=None, timeout=None, follow_redirects=False):
            seen.append((url, dict(headers or {}), cookies))
            if url == "http://127.0.0.1:7801/img":
                return _FakeStreamResponse(302, {"location": "http://127.0.0.1:9999/img2"}, url)
            return _FakeStreamResponse(200, {"content-type": "image/png"}, url, body=_png_bytes())

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    content, ctype = ifu.fetch_image_bytes(
        "http://127.0.0.1:7801/img",
        timeout=5,
        headers={"Authorization": "Bearer local"},
        cookies={"swarm_token": "tok"},
        trusted_origins=frozenset({"127.0.0.1"}),
    )
    assert content
    assert len(seen) == 2
    first_url, first_headers, first_cookies = seen[0]
    assert first_headers.get("Authorization") == "Bearer local"
    assert first_cookies == {"swarm_token": "tok"}
    second_url, second_headers, second_cookies = seen[1]
    assert second_url == "http://127.0.0.1:9999/img2"
    assert "Authorization" not in second_headers
    assert second_cookies is None
