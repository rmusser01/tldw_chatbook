import httpx
import pytest

from tldw_chatbook.Utils import egress


@pytest.fixture
def hc():
    from tldw_chatbook.Image_Generation import http_client as m
    return m


@pytest.fixture(autouse=True)
def _policy_env(monkeypatch):
    """Deterministic egress policy: resolve every hostname to a fixed public IP,
    and force ``[web_security]`` to its enabled/no-extra-allowlist defaults so
    tests are not at the mercy of a developer's local config.toml."""
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])

    async def _resolve_async(host):
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", _resolve_async)
    monkeypatch.setattr(egress, "get_cli_setting", lambda s, k=None, d=None: d)


def test_rejects_non_http_scheme(hc):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    with pytest.raises(ImageGenerationError):
        hc._validate_egress_or_raise("file:///etc/passwd")


def test_rejects_gopher_scheme(hc):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    with pytest.raises(ImageGenerationError):
        hc._validate_egress_or_raise("gopher://127.0.0.1:70/x")


def test_blocks_local_backend_url_without_trust(hc):
    # Without an explicit trusted_origins grant, a private/loopback IP -- even
    # one that happens to be a locally-configured backend -- is blocked. This
    # is the real SSRF policy (task-498) superseding the old Phase-1 guard,
    # which was permissive for ANY http(s) URL including loopback.
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    with pytest.raises(ImageGenerationError):
        hc._validate_egress_or_raise("http://127.0.0.1:7801/API/GetNewSession")


def test_allows_local_backend_url_with_trusted_origin(hc):
    # A user-configured backend base_url's host, once threaded in as
    # trusted_origins by the caller, keeps working.
    hc._validate_egress_or_raise(
        "http://127.0.0.1:7801/API/GetNewSession",
        trusted_origins=frozenset({"127.0.0.1"}),
    )  # no raise


@pytest.mark.parametrize("private_url", [
    "http://10.0.0.5/x",
    "http://192.168.1.5/x",
    "http://172.16.0.5/x",
    "http://127.0.0.1/x",
])
def test_blocks_private_ip_ranges_when_untrusted(hc, private_url):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    with pytest.raises(ImageGenerationError):
        hc._validate_egress_or_raise(private_url)


def test_blocks_link_local(hc):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    with pytest.raises(ImageGenerationError):
        hc._validate_egress_or_raise("http://169.254.10.10/x")


def test_blocks_cloud_metadata_ip(hc):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    with pytest.raises(ImageGenerationError):
        hc._validate_egress_or_raise("http://169.254.169.254/latest/meta-data/")


def test_metadata_ip_blocked_even_when_trusted(hc):
    # trusted_origins grants private-range access, but cloud metadata is a
    # harder rule: it is blocked regardless (see Utils/egress.py docstring).
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    with pytest.raises(ImageGenerationError):
        hc._validate_egress_or_raise(
            "http://169.254.169.254/latest/meta-data/",
            trusted_origins=frozenset({"169.254.169.254"}),
        )


def test_evaluate_url_policy_allowlist(hc):
    r = hc.evaluate_url_policy("https://x.aliyuncs.com/i.png", allowed_hosts={"aliyuncs.com"})
    assert r.allowed is True
    r2 = hc.evaluate_url_policy("https://evil.example/i.png", allowed_hosts={"aliyuncs.com"})
    assert r2.allowed is False


def test_evaluate_url_policy_blocks_private_before_allowlist(hc):
    # A private IP is blocked by the egress policy even if it would otherwise
    # match the allowlist logic (no allowed_hosts given == allow-any-http(s)
    # under the old guard; must now still go through SSRF enforcement).
    r = hc.evaluate_url_policy("http://192.168.1.50/i.png")
    assert r.allowed is False
    assert r.reason == "private"


def test_evaluate_url_policy_allows_private_with_trusted_origins(hc):
    r = hc.evaluate_url_policy(
        "http://192.168.1.50/i.png",
        allowed_hosts={"192.168.1.50"},
        trusted_origins=frozenset({"192.168.1.50"}),
    )
    assert r.allowed is True


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
    assert hc.fetch_json(
        "POST", "http://127.0.0.1:7801/API/x", json={"a": 1},
        trusted_origins=frozenset({"127.0.0.1"}),
    ) == {"ok": True}


def test_fetch_json_blocks_untrusted_local_url(monkeypatch, hc):
    # Without trusted_origins, fetch_json must not even reach the transport.
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, *a, **k):
            raise AssertionError("must not reach the transport when the URL is blocked")
    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    with pytest.raises(ImageGenerationError):
        hc.fetch_json("POST", "http://127.0.0.1:7801/API/x", json={"a": 1})


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
        hc.fetch_json("GET", "http://127.0.0.1:7801/x", trusted_origins=frozenset({"127.0.0.1"}))


def test_fetch_json_revalidates_redirect_to_private_ip(monkeypatch, hc):
    # First hop is a trusted, public host; the redirect Location points at a
    # private IP that is NOT covered by trusted_origins -- the per-hop
    # revalidation must catch this even though the initial hop was fine.
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    class RedirResp:
        is_redirect = True
        headers = {"location": "http://192.168.1.77/steal"}
        url = "https://api.example.com/x"
        def raise_for_status(self): pass
        def json(self): return {}
    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, *a, **k): return RedirResp()
    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    with pytest.raises(ImageGenerationError):
        hc.fetch_json(
            "GET", "https://api.example.com/x",
            trusted_origins=frozenset({"api.example.com"}),
        )


def test_fetch_json_defaults_no_autofollow(hc):
    # create_client must not auto-follow redirects by default (the manual
    # validated loop in fetch_json handles them instead).
    client = hc.create_client()
    try:
        assert client.follow_redirects is False
    finally:
        client.close()


def test_create_client_respects_explicit_zero_timeout(hc):
    # An explicit timeout=0 is a real, meaningful value (fail-fast) and must
    # not be treated the same as "not given" -- `timeout or DEFAULT` would
    # silently replace it with the 120s default since 0 is falsy.
    client = hc.create_client(timeout=0)
    try:
        assert client.timeout == httpx.Timeout(timeout=0)
    finally:
        client.close()


def test_create_client_defaults_when_timeout_omitted(hc):
    client = hc.create_client()
    try:
        assert client.timeout == httpx.Timeout(timeout=hc._DEFAULT_TIMEOUT)
    finally:
        client.close()
