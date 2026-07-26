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


def test_fetch_json_strips_authorization_on_cross_origin_redirect(monkeypatch, hc):
    # A provider redirecting to an attacker-controlled but PUBLIC host is not
    # blocked by the SSRF policy (public is allowed) -- credentials must not
    # follow across that origin change (mirrors Utils.egress._hop_headers).
    seen = []

    class RedirResp:
        is_redirect = True
        headers = {"location": "https://attacker.example/steal"}
        url = "https://api.example.com/x"
        def raise_for_status(self): pass
        def json(self): return {}

    class FinalResp:
        is_redirect = False
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"ok": True}

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, method, url, *, headers=None, **k):
            seen.append((url, dict(headers or {})))
            return RedirResp() if url == "https://api.example.com/x" else FinalResp()

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    result = hc.fetch_json(
        "GET", "https://api.example.com/x",
        headers={"Authorization": "Bearer secret", "X-Other": "keep"},
        trusted_origins=frozenset({"api.example.com"}),
    )
    assert result == {"ok": True}
    assert len(seen) == 2
    first_url, first_headers = seen[0]
    assert first_headers.get("Authorization") == "Bearer secret"
    second_url, second_headers = seen[1]
    assert second_url == "https://attacker.example/steal"
    assert "Authorization" not in second_headers
    assert second_headers.get("X-Other") == "keep"


def test_fetch_json_strips_cookies_on_cross_origin_redirect(monkeypatch, hc):
    seen = []

    class RedirResp:
        is_redirect = True
        headers = {"location": "https://attacker.example/steal"}
        url = "https://api.example.com/x"
        def raise_for_status(self): pass
        def json(self): return {}

    class FinalResp:
        is_redirect = False
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"ok": True}

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, method, url, *, cookies=None, **k):
            seen.append((url, cookies))
            return RedirResp() if url == "https://api.example.com/x" else FinalResp()

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    hc.fetch_json(
        "GET", "https://api.example.com/x",
        cookies={"session": "abc"},
        trusted_origins=frozenset({"api.example.com"}),
    )
    assert seen[0][1] == {"session": "abc"}
    assert seen[1][1] is None


def test_fetch_json_keeps_authorization_on_same_origin_redirect(monkeypatch, hc):
    # Local backends (e.g. SwarmUI) may redirect within their own origin;
    # credentials must still be sent, or local-backend generation regresses.
    seen = []

    class RedirResp:
        is_redirect = True
        headers = {"location": "http://127.0.0.1:7801/y"}
        url = "http://127.0.0.1:7801/x"
        def raise_for_status(self): pass
        def json(self): return {}

    class FinalResp:
        is_redirect = False
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"ok": True}

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, method, url, *, headers=None, **k):
            seen.append((url, dict(headers or {})))
            return RedirResp() if url.endswith("/x") else FinalResp()

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    result = hc.fetch_json(
        "GET", "http://127.0.0.1:7801/x",
        headers={"Authorization": "Bearer local-token"},
        trusted_origins=frozenset({"127.0.0.1"}),
    )
    assert result == {"ok": True}
    assert len(seen) == 2
    assert all(h.get("Authorization") == "Bearer local-token" for _u, h in seen)


def test_fetch_json_strips_authorization_on_same_host_scheme_downgrade(monkeypatch, hc):
    """A same-host HTTPS->HTTP downgrade redirect is NOT same origin (task-568):
    a malicious backend could otherwise use a downgrade redirect to receive the
    token over plaintext."""
    seen = []

    class RedirResp:
        is_redirect = True
        headers = {"location": "http://127.0.0.1:7801/y"}
        url = "https://127.0.0.1:7801/x"
        def raise_for_status(self): pass
        def json(self): return {}

    class FinalResp:
        is_redirect = False
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"ok": True}

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, method, url, *, headers=None, **k):
            seen.append((url, dict(headers or {})))
            return RedirResp() if url.endswith("/x") else FinalResp()

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    result = hc.fetch_json(
        "GET", "https://127.0.0.1:7801/x",
        headers={"Authorization": "Bearer local-token"},
        trusted_origins=frozenset({"127.0.0.1"}),
    )
    assert result == {"ok": True}
    assert len(seen) == 2
    assert seen[0][1].get("Authorization") == "Bearer local-token"
    assert "Authorization" not in seen[1][1]


def test_fetch_json_strips_authorization_on_same_host_different_port(monkeypatch, hc):
    """A same-host different-port redirect crosses an origin boundary (task-568)."""
    seen = []

    class RedirResp:
        is_redirect = True
        headers = {"location": "http://127.0.0.1:9999/y"}
        url = "http://127.0.0.1:7801/x"
        def raise_for_status(self): pass
        def json(self): return {}

    class FinalResp:
        is_redirect = False
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"ok": True}

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, method, url, *, headers=None, **k):
            seen.append((url, dict(headers or {})))
            return RedirResp() if url.endswith("/x") else FinalResp()

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    result = hc.fetch_json(
        "GET", "http://127.0.0.1:7801/x",
        headers={"Authorization": "Bearer local-token"},
        trusted_origins=frozenset({"127.0.0.1"}),
    )
    assert result == {"ok": True}
    assert len(seen) == 2
    assert seen[0][1].get("Authorization") == "Bearer local-token"
    assert "Authorization" not in seen[1][1]


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


# --- fetch_bytes_via_post --------------------------------------------------


def test_fetch_bytes_via_post_returns_body_and_content_type(monkeypatch, hc):
    """Happy path: POST returns bytes + the content-type header."""
    class FakeResp:
        status_code = 200
        is_redirect = False
        content = b"\x89PNG\r\n\x1a\nrest-of-file"
        headers = {"content-type": "image/png"}
        def raise_for_status(self): pass
    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, method, url, *, json=None, **k):
            assert method == "POST"
            return FakeResp()
    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    body, ctype = hc.fetch_bytes_via_post(
        "https://api.example.com/gen", json={"prompt": "x"},
        trusted_origins=frozenset({"api.example.com"}),
    )
    assert body.startswith(b"\x89PNG") and ctype == "image/png"


def test_fetch_bytes_via_post_validates_egress_first(monkeypatch, hc):
    # Private IP without trusted_origins -> egress error, fake client never called.
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, *a, **k):
            raise AssertionError("must not reach the transport when the URL is blocked")
    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    with pytest.raises(ImageGenerationError):
        hc.fetch_bytes_via_post("http://127.0.0.1:7801/gen", json={"prompt": "x"})


def test_fetch_bytes_via_post_strips_credentials_on_cross_origin_redirect(monkeypatch, hc):
    # 307 to another host: Authorization absent on hop 2; same-origin keeps it
    # on hop 1 (mirrors test_fetch_json_strips_authorization_on_cross_origin_redirect).
    seen = []

    class RedirResp:
        is_redirect = True
        status_code = 307
        headers = {"location": "https://attacker.example/steal"}
        url = "https://api.example.com/x"
        def raise_for_status(self): pass

    class FinalResp:
        is_redirect = False
        status_code = 200
        content = b"bytes-payload"
        headers = {"content-type": "application/octet-stream"}
        def raise_for_status(self): pass

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, method, url, *, headers=None, **k):
            seen.append((url, dict(headers or {})))
            return RedirResp() if url == "https://api.example.com/x" else FinalResp()

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    body, ctype = hc.fetch_bytes_via_post(
        "https://api.example.com/x",
        headers={"Authorization": "Bearer secret", "X-Other": "keep"},
        json={"prompt": "x"},
        trusted_origins=frozenset({"api.example.com"}),
    )
    assert body == b"bytes-payload" and ctype == "application/octet-stream"
    assert len(seen) == 2
    first_url, first_headers = seen[0]
    assert first_headers.get("Authorization") == "Bearer secret"
    second_url, second_headers = seen[1]
    assert second_url == "https://attacker.example/steal"
    assert "Authorization" not in second_headers
    assert second_headers.get("X-Other") == "keep"


def test_fetch_bytes_via_post_redirect_to_private_ip_blocked(monkeypatch, hc):
    # 307 Location -> 10.0.0.1 raises an egress error, even though the first
    # hop was fine.
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    class RedirResp:
        is_redirect = True
        status_code = 307
        headers = {"location": "http://10.0.0.1/steal"}
        url = "https://api.example.com/x"
        def raise_for_status(self): pass

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, *a, **k): return RedirResp()

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    with pytest.raises(ImageGenerationError):
        hc.fetch_bytes_via_post(
            "https://api.example.com/x", json={"prompt": "x"},
            trusted_origins=frozenset({"api.example.com"}),
        )


def test_fetch_bytes_via_post_max_bytes_exceeded_raises(monkeypatch, hc):
    # Content longer than max_bytes -> clear error naming the cap, not a
    # silently truncated return.
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    class FakeResp:
        status_code = 200
        is_redirect = False
        content = b"x" * 100
        headers = {"content-type": "application/octet-stream"}
        def raise_for_status(self): pass
    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, *a, **k): return FakeResp()
    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    with pytest.raises(ImageGenerationError) as exc_info:
        hc.fetch_bytes_via_post(
            "https://api.example.com/gen", json={"prompt": "x"},
            trusted_origins=frozenset({"api.example.com"}),
            max_bytes=10,
        )
    assert "10" in str(exc_info.value)


def test_fetch_bytes_via_post_respects_explicit_zero_timeout(monkeypatch, hc):
    # timeout=0 is passed through to the client as-is (the task-497 lesson);
    # None falls back to _DEFAULT_TIMEOUT.
    captured = []

    class FakeResp:
        status_code = 200
        is_redirect = False
        content = b"bytes"
        headers = {"content-type": "application/octet-stream"}
        def raise_for_status(self): pass
    class FakeClient:
        def __init__(self, *a, **k):
            captured.append(k.get("timeout"))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def request(self, *a, **k): return FakeResp()

    monkeypatch.setattr(hc.httpx, "Client", FakeClient)
    hc.fetch_bytes_via_post(
        "https://api.example.com/gen", json={"prompt": "x"}, timeout=0,
        trusted_origins=frozenset({"api.example.com"}),
    )
    assert captured[-1] == 0

    captured.clear()
    hc.fetch_bytes_via_post(
        "https://api.example.com/gen", json={"prompt": "x"}, timeout=None,
        trusted_origins=frozenset({"api.example.com"}),
    )
    assert captured[-1] == hc._DEFAULT_TIMEOUT
