"""Egress-gated stream resolution (task-3401.11)."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Media_Playback import stream_resolve as sr


# -- fakes ----------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, url, status=200, headers=None):
        self.url = url
        self.status_code = status
        self.headers = headers or {}

    @property
    def is_redirect(self):
        return self.status_code in {301, 302, 303, 307, 308}


class _FakeClient:
    """Stands in for httpx.Client; responses scripted by URL prefix."""

    def __init__(self, routes):
        self._routes = routes  # dict[url -> _FakeResponse | list[_FakeResponse]]
        self.requests: list[str] = []

    def __call__(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _next(self, url):
        self.requests.append(url)
        value = self._routes[url]
        if isinstance(value, list):
            return value.pop(0)
        return value

    def head(self, url, **_kwargs):
        return self._next(url)

    def stream(self, _method, url, **_kwargs):
        response = self._next(url)

        class _StreamCtx:
            def __enter__(self):
                return response

            def __exit__(self, *exc):
                return False

        return _StreamCtx()


@pytest.fixture
def egress_calls(monkeypatch):
    calls: list[tuple[str, frozenset]] = []

    def fake_check(url, *, trusted_origins=frozenset()):
        calls.append((url, trusted_origins))

    monkeypatch.setattr(sr.egress, "check_url_or_raise", fake_check)
    monkeypatch.setattr(
        sr.egress, "origin_set", lambda url: frozenset({f"origin:{url}"})
    )
    return calls


def _mount_client(monkeypatch, routes):
    monkeypatch.setattr(sr.httpx, "Client", _FakeClient(routes))


TYPED = "https://videos.example.com/watch/abc"


# -- shape / scheme ---------------------------------------------------------------


def test_non_http_urls_refused(egress_calls):
    for bad in ("ftp://x.example.com/v.mp4", "notaurl", ""):
        with pytest.raises(sr.StreamResolutionError, match="not an http"):
            sr.resolve_stream_url(bad)


# -- direct media -----------------------------------------------------------------


def test_direct_media_url_with_ranges(egress_calls, monkeypatch):
    _mount_client(monkeypatch, {TYPED: _FakeResponse(TYPED, headers={"content-type": "video/mp4", "accept-ranges": "bytes", "content-length": "1048576"})})
    resolution = sr.resolve_stream_url(TYPED)
    assert resolution.final_url == TYPED
    assert resolution.seekable is True
    assert resolution.via_ytdlp is False
    assert resolution.content_length == 1048576
    # AC2: the typed hop carried the trust seed.
    assert egress_calls[0] == (TYPED, frozenset({f"origin:{TYPED}"}))


def test_redirect_chain_revalidates_every_hop(egress_calls, monkeypatch):
    cdn = "https://cdn.example.net/f/abc.mp4"
    _mount_client(monkeypatch, {
        TYPED: _FakeResponse(TYPED, status=302, headers={"location": cdn}),
        cdn: _FakeResponse(cdn, headers={"content-type": "video/mp4"}),
    })
    resolution = sr.resolve_stream_url(TYPED)
    assert resolution.final_url == cdn
    assert resolution.via_ytdlp is False
    # Hop 2 (the CDN) was validated WITHOUT the typed trust (AC2).
    assert egress_calls[0] == (TYPED, frozenset({f"origin:{TYPED}"}))
    assert (cdn, frozenset()) in egress_calls


def test_redirect_without_location_refused(egress_calls, monkeypatch):
    _mount_client(monkeypatch, {TYPED: _FakeResponse(TYPED, status=302, headers={})})
    with pytest.raises(sr.StreamResolutionError, match="without a Location"):
        sr.resolve_stream_url(TYPED)


def test_too_many_redirects(egress_calls, monkeypatch):
    hops = {TYPED: _FakeResponse(TYPED, status=302, headers={"location": TYPED})}
    _mount_client(monkeypatch, hops)
    with pytest.raises(sr.StreamResolutionError, match="too many redirects"):
        sr.resolve_stream_url(TYPED)


def test_egress_refusal_surfaces(egress_calls, monkeypatch):
    from tldw_chatbook.Utils.egress import EgressBlockedError

    def fake_check(url, *, trusted_origins=frozenset()):
        raise EgressBlockedError(url, "private-ip")

    monkeypatch.setattr(sr.egress, "check_url_or_raise", fake_check)
    monkeypatch.setattr(sr.httpx, "Client", _FakeClient({}))
    with pytest.raises(EgressBlockedError):
        sr.resolve_stream_url(TYPED)


# -- HLS / DASH ---------------------------------------------------------------------


def test_hls_url_refused_with_followup_note(egress_calls, monkeypatch):
    hls = "https://videos.example.com/master.m3u8"
    _mount_client(monkeypatch, {hls: _FakeResponse(hls, headers={"content-type": "application/vnd.apple.mpegurl"})})
    with pytest.raises(sr.StreamResolutionError, match="follow-up"):
        sr.resolve_stream_url(hls)


# -- yt-dlp fallback ------------------------------------------------------------------


def test_html_page_falls_to_ytdlp(egress_calls, monkeypatch):
    stream_url = "https://cdn.example.net/direct.mp4"
    _mount_client(monkeypatch, {TYPED: _FakeResponse(TYPED, headers={"content-type": "text/html"})})
    monkeypatch.setattr(sr.shutil, "which", lambda tool: f"/usr/bin/{tool}")

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "yt-dlp" and "-g" in cmd
        return SimpleNamespace(returncode=0, stdout=stream_url + "\n", stderr="")

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    # The ranges probe on the resolved URL is also mocked.
    monkeypatch.setattr(sr, "_probe_ranges", lambda url: (True, None))
    resolution = sr.resolve_stream_url(TYPED)
    assert resolution.final_url == stream_url
    assert resolution.via_ytdlp is True
    # The yt-dlp output URL was egress-validated WITHOUT trust (AC2).
    assert (stream_url, frozenset()) in egress_calls


def test_ytdlp_missing_binary_named(egress_calls, monkeypatch):
    _mount_client(monkeypatch, {TYPED: _FakeResponse(TYPED, headers={"content-type": "text/html"})})
    monkeypatch.setattr(sr.shutil, "which", lambda tool: None)
    with pytest.raises(sr.StreamResolutionError, match="yt-dlp"):
        sr.resolve_stream_url(TYPED)


def test_ytdlp_m3u8_output_refused(egress_calls, monkeypatch):
    _mount_client(monkeypatch, {TYPED: _FakeResponse(TYPED, headers={"content-type": "text/html"})})
    monkeypatch.setattr(sr.shutil, "which", lambda tool: "/usr/bin/yt-dlp")
    monkeypatch.setattr(
        sr.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(
            returncode=0, stdout="https://cdn.example.net/master.m3u8\n", stderr=""
        ),
    )
    with pytest.raises(sr.StreamResolutionError, match="follow-up"):
        sr.resolve_stream_url(TYPED)


def test_ytdlp_failure_surfaces_stderr(egress_calls, monkeypatch):
    _mount_client(monkeypatch, {TYPED: _FakeResponse(TYPED, headers={"content-type": "text/html"})})
    monkeypatch.setattr(sr.shutil, "which", lambda tool: "/usr/bin/yt-dlp")
    monkeypatch.setattr(
        sr.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="ERROR: Unsupported URL\n"
        ),
    )
    with pytest.raises(sr.StreamResolutionError, match="could not resolve"):
        sr.resolve_stream_url(TYPED)


# -- grammar --------------------------------------------------------------------------


def test_stream_video_registered_in_default_registry():
    from tldw_chatbook.Chat.console_command_grammar import (
        STREAM_VIDEO_COMMAND_NAME,
        default_console_registry,
    )

    registry = default_console_registry()
    assert STREAM_VIDEO_COMMAND_NAME in registry.available_names()
    parse = registry.parse("/stream-video https://example.com/v.mp4")
    assert parse.name == STREAM_VIDEO_COMMAND_NAME
    assert parse.args == "https://example.com/v.mp4"
