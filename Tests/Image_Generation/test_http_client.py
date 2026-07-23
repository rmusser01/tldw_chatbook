import pytest

@pytest.fixture
def hc():
    from tldw_chatbook.Image_Generation import http_client as m
    return m

def test_rejects_non_http_scheme(hc):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    with pytest.raises(ImageGenerationError):
        hc._validate_egress_or_raise("file:///etc/passwd")

def test_allows_local_backend_url(hc):
    # user-configured local backends must pass the light guard
    hc._validate_egress_or_raise("http://127.0.0.1:7801/API/GetNewSession")  # no raise

def test_evaluate_url_policy_allowlist(hc):
    r = hc.evaluate_url_policy("https://x.aliyuncs.com/i.png", allowed_hosts={"aliyuncs.com"})
    assert r.allowed is True
    r2 = hc.evaluate_url_policy("https://evil.example/i.png", allowed_hosts={"aliyuncs.com"})
    assert r2.allowed is False

def test_fetch_json_parses(monkeypatch, hc):
    class FakeResp:
        status_code = 200
        is_redirect = False
        def json(self): return {"ok": True}
        def raise_for_status(self): pass
    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, *a, **k): return FakeResp()
    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    assert hc.fetch_json("POST", "http://127.0.0.1:7801/API/x", json={"a": 1}) == {"ok": True}


def test_fetch_json_revalidates_redirect_hop(monkeypatch, hc):
    # A redirect to a disallowed scheme must be re-validated and rejected,
    # not blindly followed (egress guard must run on every hop).
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    class RedirResp:
        is_redirect = True
        headers = {"location": "file:///etc/passwd"}
        url = "http://127.0.0.1:7801/x"
        def raise_for_status(self): pass
        def json(self): return {}
    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, *a, **k): return RedirResp()
    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    with pytest.raises(ImageGenerationError):
        hc.fetch_json("GET", "http://127.0.0.1:7801/x")


def test_fetch_json_defaults_no_autofollow(hc):
    # create_client must not auto-follow redirects by default (the manual
    # validated loop in fetch_json handles them instead).
    client = hc.create_client()
    try:
        assert client.follow_redirects is False
    finally:
        client.close()
