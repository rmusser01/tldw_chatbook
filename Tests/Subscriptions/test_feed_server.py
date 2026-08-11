"""Tests for `FeedDirectoryServer` (task-1760).

A real `http.server.ThreadingHTTPServer` on an ephemeral localhost port,
driven with a real `httpx` sync client -- never a mock of the stdlib
server -- so these tests actually exercise the socket, the thread, and the
handler's path resolution rather than a stand-in for them. Every server
started here is stopped in a `finally`/`try...finally` before the test
ends, and every port is ephemeral (`port=0`) or, for the double-start test,
irrelevant to any other test's socket -- nothing here binds a fixed port.

The load-bearing tests are the traversal matrix
(`test_a_symlink_inside_the_served_directory_pointing_outside_is_denied`
above all: `SimpleHTTPRequestHandler.translate_path` alone already
collapses literal `..`/absolute-path traversal in the URL -- see that
method's own docstring -- but does NOT stop a symlink planted inside the
served directory from resolving to a target outside it, since `open()`/
`os.stat()` follow symlinks by default) and the two mutation checks
described in the task's own instructions: removing the containment check
must fail the symlink test, and removing the GET/HEAD-only gate must fail
the 405 test. Both were verified by hand during development (see the
task's Implementation Notes) and are not re-run automatically here.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

import httpx
import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Subscriptions import feed_server as feed_server_module
from tldw_chatbook.Subscriptions.feed_server import (
    FeedDirectoryServer,
    FeedServerError,
    configured_bind_and_port,
    is_loopback_bind,
)

# Network opt-in (task-15111): this module binds a real briefings feed
# server on an ephemeral loopback port and fetches from it.
# The autouse guard in Tests/conftest.py denies egress by default; every address
# these tests reach is a port this process itself is listening on.
pytestmark = [pytest.mark.unit, pytest.mark.allow_network]


def _raw_http_request(port: int, request_bytes: bytes, timeout: float = 5.0) -> bytes:
    """Send raw bytes straight to the server's port and return whatever
    comes back over the wire, unmodified.

    Fix wave (task-1760 review, M1/L1 test-quality note): `httpx`
    normalizes URLs client-side before a request is ever sent -- a literal
    `..`, a bare `GET\\r\\n\\r\\n`, or an over-long request line can never
    reach the server through it. A raw socket is the only way to prove
    what the server itself does with bytes an `httpx`-based test can never
    put on the wire.
    """
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(request_bytes)
        chunks: list[bytes] = []
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        except OSError:
            pass
    return b"".join(chunks)


def _url_port(url: str) -> int:
    return int(url.rsplit(":", 1)[1].rstrip("/"))


@pytest.fixture
def served_dir(tmp_path: Path) -> Path:
    """A directory with one seeded 'feed' file, standing in for an
    exported feed directory's `feed.xml` + episode files."""
    directory = tmp_path / "feed"
    directory.mkdir()
    (directory / "feed.xml").write_text(
        "<rss><channel><title>Test Feed</title></channel></rss>", encoding="utf-8"
    )
    (directory / "episode-1.wav").write_bytes(b"RIFF....WAVEfmt ")
    return directory


@pytest.fixture
def server() -> FeedDirectoryServer:
    """A fresh, unstarted server -- stopped unconditionally after the test,
    even if the test itself never started it (`stop()` is a no-op then) or
    failed before reaching its own cleanup."""
    instance = FeedDirectoryServer()
    try:
        yield instance
    finally:
        instance.stop()


# --- start/stop basics -------------------------------------------------------


def test_is_running_and_url_are_none_before_start(server: FeedDirectoryServer) -> None:
    assert server.is_running is False
    assert server.url is None


def test_start_reports_the_real_bound_port_for_an_ephemeral_request(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """`port=0` asks the OS for any free port; the returned URL must name
    the ACTUAL bound port, not `0` -- a client cannot connect to port 0."""
    url = server.start(served_dir, port=0)
    assert url.startswith("http://127.0.0.1:")
    port_text = url.rsplit(":", 1)[1].rstrip("/")
    assert port_text.isdigit()
    assert int(port_text) != 0
    assert server.is_running is True
    assert server.url == url


def test_stop_is_a_no_op_when_never_started(server: FeedDirectoryServer) -> None:
    server.stop()  # must not raise
    assert server.is_running is False


def test_stop_closes_the_socket_so_a_later_request_is_refused(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    url = server.start(served_dir)
    httpx.get(url + "feed.xml", timeout=5.0).raise_for_status()

    server.stop()
    assert server.is_running is False
    assert server.url is None

    with pytest.raises(httpx.ConnectError):
        httpx.get(url + "feed.xml", timeout=5.0)


def test_double_start_refuses_and_leaves_the_first_server_serving(
    server: FeedDirectoryServer, served_dir: Path, tmp_path: Path
) -> None:
    """The task's own "one directory at a time" decision: a second
    `start()` while one is running REFUSES (naming the running URL)
    rather than restarting on the new directory -- see `FeedDirectoryServer.
    start`'s own docstring for why refusing was chosen as the simpler of
    the two options the plan allowed.
    """
    first_url = server.start(served_dir)

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "feed.xml").write_text("<rss></rss>", encoding="utf-8")

    with pytest.raises(FeedServerError, match=first_url):
        server.start(other_dir)

    # The FIRST server is undisturbed -- still serving its own directory.
    response = httpx.get(first_url + "feed.xml", timeout=5.0)
    assert response.status_code == 200
    assert "Test Feed" in response.text


def test_start_rejects_a_missing_directory(server: FeedDirectoryServer, tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FeedServerError):
        server.start(missing)
    assert server.is_running is False


def test_start_rejects_a_file_that_is_not_a_directory(
    server: FeedDirectoryServer, tmp_path: Path
) -> None:
    a_file = tmp_path / "not-a-dir.txt"
    a_file.write_text("hello", encoding="utf-8")
    with pytest.raises(FeedServerError):
        server.start(a_file)
    assert server.is_running is False


# --- round-trip GET (AC #1) --------------------------------------------------


def test_round_trip_get_of_a_seeded_feed_file(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    url = server.start(served_dir)

    response = httpx.get(url + "feed.xml", timeout=5.0)
    assert response.status_code == 200
    assert response.text == (served_dir / "feed.xml").read_text(encoding="utf-8")

    audio_response = httpx.get(url + "episode-1.wav", timeout=5.0)
    assert audio_response.status_code == 200
    assert audio_response.content == (served_dir / "episode-1.wav").read_bytes()


def test_head_request_succeeds_with_no_body(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    url = server.start(served_dir)
    response = httpx.head(url + "feed.xml", timeout=5.0)
    assert response.status_code == 200
    assert response.content == b""


def test_a_missing_file_inside_the_served_directory_is_a_plain_404(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    url = server.start(served_dir)
    response = httpx.get(url + "no-such-file.xml", timeout=5.0)
    assert response.status_code == 404


# --- method surface (AC #3 adjacent: read-only) -----------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_write_methods_are_rejected_with_405(
    server: FeedDirectoryServer, served_dir: Path, method: str
) -> None:
    url = server.start(served_dir)
    response = httpx.request(method, url + "feed.xml", timeout=5.0)
    assert response.status_code == 405


# --- malformed requests get a real error response (task-1760 review, M1) -----
#
# `log_message` used to read `self.path`, which `BaseHTTPRequestHandler.
# parse_request` does not set until AFTER the point where it calls
# `send_error` -> `log_error` -> `log_message` on these exact failure
# paths. Reading it raised `AttributeError`, which killed the intended
# 400/414 response (the client got zero bytes instead) and the per-
# connection handler thread with it. Both cases below are driven with a
# raw socket -- `httpx` builds well-formed requests and can never put a
# bare `GET\r\n\r\n` or a >64KiB request line on the wire.


def test_a_malformed_request_line_gets_a_400_not_a_dropped_connection(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """`GET / a HTTP/1.0` -- 4 words, not the 2-3 `parse_request` accepts
    -- is one of the reviewer's own confirmed repro lines. It is used
    (rather than a bare `GET\\r\\n\\r\\n`) specifically because its word
    count still lets `parse_request` validate and set `self.request_
    version` to a real `HTTP/1.0` *before* the word-count check fails --
    unlike a 1-word requestline, which leaves `request_version` at the
    stdlib's own `HTTP/0.9` default and (correctly, independent of this
    bug) sends a bare body with no status line at all, per that protocol.
    Both shapes exercise the same `self.path`-before-it-exists bug in
    `log_message`; this one just lets the assertion check a real status
    line too.
    """
    url = server.start(served_dir)
    port = _url_port(url)

    response = _raw_http_request(port, b"GET / a HTTP/1.0\r\n\r\n")
    assert response, "the client must receive a real response, not zero bytes"
    status_line = response.split(b"\r\n", 1)[0]
    assert status_line.startswith(b"HTTP/")
    assert b" 400 " in status_line

    # The bug killed one connection's handler thread, not the whole
    # server -- but a real regression here would show up as this
    # follow-up request failing too.
    follow_up = httpx.get(url + "feed.xml", timeout=5.0)
    assert follow_up.status_code == 200


def test_a_bare_malformed_request_line_does_not_crash_the_handler_thread(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """The reviewer's other confirmed repro: a 1-word requestline (`GET`
    alone). The stdlib itself, independent of this bug, answers a
    request it never upgraded past its own `HTTP/0.9` default with a bare
    body and no status line -- so this test's own bar is simply "the
    connection gets SOME response and the server survives", not a status
    line (covered by the sibling test above and the over-long-line test
    below instead).
    """
    url = server.start(served_dir)
    port = _url_port(url)

    response = _raw_http_request(port, b"GET\r\n\r\n")
    assert response, "the client must receive a real response, not zero bytes"

    follow_up = httpx.get(url + "feed.xml", timeout=5.0)
    assert follow_up.status_code == 200


def test_an_over_long_request_line_gets_a_414_not_a_dropped_connection(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """`handle_one_request`'s own >64KiB guard sets `self.command = ''`
    but never sets `self.path` at all before calling `send_error` -- the
    same missing-attribute trap as the malformed-request-line case above,
    on a different code path (`command` exists here; `path` still does
    not)."""
    url = server.start(served_dir)
    port = _url_port(url)

    oversized_target = b"A" * 70_000
    request = b"GET /" + oversized_target + b" HTTP/1.1\r\n\r\n"
    response = _raw_http_request(port, request)
    assert response, "the client must receive a real response, not zero bytes"
    status_line = response.split(b"\r\n", 1)[0]
    assert status_line.startswith(b"HTTP/")
    assert b" 414 " in status_line

    follow_up = httpx.get(url + "feed.xml", timeout=5.0)
    assert follow_up.status_code == 200


# --- stop() latency (task-1760 review, M2) -----------------------------------


def test_stop_returns_well_under_the_old_half_second_stall(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """`stop()` used to block for the stdlib's default 0.5s `serve_forever`
    poll interval (measured ~501ms) on the SAME thread the Stop button
    handler and `on_unmount` call it from -- a real, measured UI freeze.
    `start()` now runs the loop at a 50ms poll interval instead, bounding
    this to roughly a tenth of that. 150ms is a generous ceiling: well
    above the ~50ms this should take, comfortably below the ~500ms this
    reproduced before the fix, so this is not a flaky assertion on a
    loaded CI box.
    """
    server.start(served_dir)

    started_at = time.monotonic()
    server.stop()
    elapsed_seconds = time.monotonic() - started_at

    assert elapsed_seconds < 0.15, (
        f"stop() took {elapsed_seconds * 1000:.1f}ms, expected well under 150ms"
    )


# --- bind validation (task-1760 review, M3) -----------------------------------


def test_start_normalizes_a_blank_bind_to_the_safe_loopback_default(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """A blank `bind` (an operator's attempt to "unset" it, or a hand-
    edited config gone wrong) must resolve to loopback, never to an empty
    string -- `socket.bind(("", 0))` binds EVERY interface, silently
    turning the loopback-only default into a LAN-wide one."""
    url = server.start(served_dir, bind="")
    assert server.bind == "127.0.0.1"
    assert url.startswith("http://127.0.0.1:")
    assert is_loopback_bind(server.bind) is True


def test_start_normalizes_a_numeric_bind_to_the_safe_loopback_default(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """A `bind = 0` typo (meant for `port`) is a non-string, not merely a
    blank string -- `_normalize_bind` must catch that shape too, since
    `socket.bind(("0", 0))` and `socket.bind(("", 0))` both resolve to
    `0.0.0.0` just as readily."""
    url = server.start(served_dir, bind=0)  # type: ignore[arg-type]
    assert server.bind == "127.0.0.1"
    assert url.startswith("http://127.0.0.1:")


def test_configured_bind_and_port_falls_back_to_loopback_on_a_blank_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_cli_setting(section, key, default=None):
        if section == "briefings_feed_server" and key == "bind":
            return ""
        return default

    monkeypatch.setattr(feed_server_module, "get_cli_setting", _fake_get_cli_setting)
    bind, _port = configured_bind_and_port()
    assert bind == "127.0.0.1"


def test_a_non_loopback_bind_warns_once_and_is_loopback_bind_says_so(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """A deliberately widened bind (or a config mistake that still
    resolves to a real address, unlike a blank/typo'd one) must not be
    silently invisible -- `start()` logs a warning, and the same
    `is_loopback_bind` check lets the UI layer word its own toast around
    the identical fact rather than re-deriving it.

    `caplog` does not intercept loguru (this project's logger, per this
    repo's own established idiom -- see e.g. `Tests/Chat/
    test_attachment_policy.py`'s `TestSvgCapability`); a temporary loguru
    sink is used instead.
    """
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        server.start(served_dir, bind="0.0.0.0", port=0)
    finally:
        loguru_logger.remove(sink_id)

    assert any("not loopback-only" in message for message in messages), messages
    assert server.bind == "0.0.0.0"
    assert is_loopback_bind(server.bind) is False


def test_a_loopback_bind_never_warns(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        server.start(served_dir)  # default bind -- loopback
    finally:
        loguru_logger.remove(sink_id)

    assert messages == []
    assert is_loopback_bind(server.bind) is True


# --- port validation (task-1760 Qodo fix wave, F1) ----------------------------
#
# A port that survives `int(...)` in `configured_bind_and_port` can still be
# outside the valid 0-65535 TCP range (a config typo like `99999`) or, for a
# direct `FeedDirectoryServer.start` caller, not an `int` at all. Left
# unchecked, an out-of-range-but-parseable value reached `ThreadingHTTPServer`
# as a bare `OverflowError` -- not one of the types the UI's Serve handler
# (`watchlists_collections_screen.py`) catches, so it escaped as an unhandled
# exception. The two entry points are deliberately handled differently: a
# CONFIG-derived bad value degrades safely (falls back to ephemeral `0` with a
# warning, `configured_bind_and_port`'s existing bad-bind precedent); a bad
# value handed directly to `start()` is instead refused with `FeedServerError`
# (the type the UI already catches), the same way the directory-validation
# checks in that method already are.


def test_configured_bind_and_port_falls_back_to_ephemeral_on_an_out_of_range_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured port that parses as an int but is out of the valid
    0-65535 TCP port range (a typo, e.g. `99999`) degrades the same way a
    non-integer value already does, rather than being passed through to
    `start()` where it would reach the socket layer as a raw
    `OverflowError`."""

    def _fake_get_cli_setting(section, key, default=None):
        if section == "briefings_feed_server" and key == "port":
            return 99999
        return default

    monkeypatch.setattr(feed_server_module, "get_cli_setting", _fake_get_cli_setting)

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        _bind, port = configured_bind_and_port()
    finally:
        loguru_logger.remove(sink_id)

    assert port == 0
    assert any(
        "outside the valid 0-65535" in message for message in messages
    ), messages


def test_configured_bind_and_port_falls_back_to_ephemeral_on_a_negative_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_cli_setting(section, key, default=None):
        if section == "briefings_feed_server" and key == "port":
            return -1
        return default

    monkeypatch.setattr(feed_server_module, "get_cli_setting", _fake_get_cli_setting)
    _bind, port = configured_bind_and_port()
    assert port == 0


def test_configured_bind_and_port_still_uses_a_valid_configured_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new range check must not reject an ordinary, in-range value --
    only out-of-range/non-integer ones (regression guard alongside the
    pre-existing `test_configured_bind_and_port_reads_a_stored_section`)."""

    def _fake_get_cli_setting(section, key, default=None):
        if section == "briefings_feed_server" and key == "port":
            return 8123
        return default

    monkeypatch.setattr(feed_server_module, "get_cli_setting", _fake_get_cli_setting)
    _bind, port = configured_bind_and_port()
    assert port == 8123


def test_start_rejects_a_negative_port(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    with pytest.raises(FeedServerError):
        server.start(served_dir, port=-1)
    assert server.is_running is False
    assert server.url is None


def test_start_rejects_a_port_above_65535(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    with pytest.raises(FeedServerError):
        server.start(served_dir, port=70000)
    assert server.is_running is False
    assert server.url is None


def test_start_rejects_a_non_integer_port(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    with pytest.raises(FeedServerError):
        server.start(served_dir, port="443")  # type: ignore[arg-type]
    assert server.is_running is False
    assert server.url is None


def test_start_accepts_a_valid_nonzero_configured_port(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """A legitimate, in-range, nonzero port (not just `0` for ephemeral)
    must still work end to end -- the new range check must reject only
    genuinely invalid values, not ordinary ones a real config could set."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    url = server.start(served_dir, port=free_port)
    assert url == f"http://127.0.0.1:{free_port}/"
    response = httpx.get(url + "feed.xml", timeout=5.0)
    assert response.status_code == 200


# --- IPv6 URL bracketing (task-1760 Qodo fix wave, F2) ------------------------
#
# `is_loopback_bind` explicitly supports `::1` as a loopback bind target, but
# `start()` used to build its URL as `f"http://{bind}:{port}/"` unconditionally
# -- `http://::1:8080/` is not a URL any client can parse (an IPv6 literal in
# a URL authority component must be bracketed per RFC 3986 section 3.2.2).


def test_format_host_for_url_brackets_ipv6_literals() -> None:
    assert feed_server_module._format_host_for_url("::1") == "[::1]"
    assert feed_server_module._format_host_for_url("2001:db8::1") == "[2001:db8::1]"


def test_format_host_for_url_leaves_ipv4_and_hostnames_unbracketed() -> None:
    assert feed_server_module._format_host_for_url("127.0.0.1") == "127.0.0.1"
    assert feed_server_module._format_host_for_url("0.0.0.0") == "0.0.0.0"
    assert feed_server_module._format_host_for_url("localhost") == "localhost"


def _ipv6_loopback_bindable() -> bool:
    """True if this platform/runner can actually bind `::1` -- some CI
    sandboxes disable IPv6 entirely, which is a runner-environment fact,
    not a defect in the fix under test; the one test below that needs a
    real IPv6 socket is skipped (not failed) when this is False."""
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _ipv6_loopback_bindable(), reason="platform/runner cannot bind ::1"
)
def test_start_with_ipv6_loopback_bind_produces_a_bracketed_url_and_round_trips(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """The end-to-end case: binding `::1` must produce a URL a real client
    can parse and connect to, not just a correctly-formatted string in
    isolation (that half is covered by the two tests above, which do not
    need a live IPv6 socket at all)."""
    url = server.start(served_dir, bind="::1", port=0)
    assert url.startswith("http://[::1]:")
    assert url.endswith("/")

    response = httpx.get(url + "feed.xml", timeout=5.0)
    assert response.status_code == 200
    assert response.text == (served_dir / "feed.xml").read_text(encoding="utf-8")


# --- directory listings disabled, recursive FILE serving intact (M4) ---------


def test_root_index_is_refused_with_404_not_a_directory_listing(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """A podcast feed is fetched by URLs named in `feed.xml` -- nothing
    ever needs to browse the served directory, and an auto-generated
    index is pure exposure surface on top of the files a client already
    knows about."""
    url = server.start(served_dir)
    response = httpx.get(url, timeout=5.0)
    assert response.status_code == 404
    assert "<li>" not in response.text, "no directory listing HTML"


def test_a_subdirectory_index_is_also_refused_with_404(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    subdir = served_dir / "episodes"
    subdir.mkdir()
    (subdir / "bonus.wav").write_bytes(b"RIFF....WAVEfmt ")

    url = server.start(served_dir)
    response = httpx.get(url + "episodes/", timeout=5.0)
    assert response.status_code == 404
    assert "<li>" not in response.text


def test_a_known_file_inside_a_subdirectory_still_serves_200(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """M4's fix disables browsing, not recursive FILE serving -- an
    episode `feed.xml` names inside a subfolder must still resolve."""
    subdir = served_dir / "episodes"
    subdir.mkdir()
    (subdir / "bonus.wav").write_bytes(b"RIFF....WAVEfmt ")

    url = server.start(served_dir)
    response = httpx.get(url + "episodes/bonus.wav", timeout=5.0)
    assert response.status_code == 200
    assert response.content == b"RIFF....WAVEfmt "


# --- traversal matrix (AC #3) ------------------------------------------------


def test_dotdot_traversal_does_not_escape_the_served_directory(
    server: FeedDirectoryServer, served_dir: Path, tmp_path: Path
) -> None:
    """A sibling directory holds a 'secret' the served directory has no
    business exposing; `../`-shaped requests must never return it.

    Test-quality note (task-1760 review): this test alone is near-vacuous
    for its stated purpose -- `httpx` normalizes a literal `../` client-
    side before the request is ever sent (`httpx.URL("http://h/../x").
    raw_path == b"/x"`), so the literal `..` never reaches the wire here
    and the server only ever sees an ordinary missing-file request. Kept
    (it is still a real, if weaker, 404 assertion) alongside
    `test_raw_socket_dotdot_forms_httpx_would_never_put_on_the_wire`
    below, which puts each form directly on a raw socket instead.
    """
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "passwd").write_text("root:x:0:0", encoding="utf-8")

    url = server.start(served_dir)
    response = httpx.get(
        url + "../secret/passwd", timeout=5.0, follow_redirects=False
    )
    assert response.status_code in (404, 400)
    assert "root:x:0:0" not in response.text


@pytest.mark.parametrize(
    "wire_target",
    [
        b"/../secret/passwd",
        b"/../../secret/passwd",
        b"/..%2fsecret%2fpasswd",
        b"/..%5csecret%5cpasswd",
        b"//secret/passwd",
        b"/./../secret/passwd",
    ],
)
def test_raw_socket_dotdot_forms_httpx_would_never_put_on_the_wire(
    server: FeedDirectoryServer,
    served_dir: Path,
    tmp_path: Path,
    wire_target: bytes,
) -> None:
    """The real fix for the sibling test's near-vacuous coverage: each
    traversal form goes directly onto a raw socket, bypassing any client-
    side normalization, so the 404 this asserts is the SERVER's own
    answer -- not an artifact of a well-behaved HTTP client never sending
    the bad bytes in the first place.
    """
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "passwd").write_text("root:x:0:0", encoding="utf-8")

    url = server.start(served_dir)
    port = _url_port(url)

    request = (
        b"GET " + wire_target + b" HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )
    response = _raw_http_request(port, request)

    status_line = response.split(b"\r\n", 1)[0]
    assert b" 404 " in status_line or b" 400 " in status_line, response[:200]
    assert b"root:x:0:0" not in response


def test_percent_encoded_dotdot_traversal_does_not_escape(
    server: FeedDirectoryServer, served_dir: Path, tmp_path: Path
) -> None:
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "passwd").write_text("root:x:0:0", encoding="utf-8")

    url = server.start(served_dir)
    response = httpx.get(
        url + "..%2f secret%2fpasswd".replace(" ", ""),
        timeout=5.0,
        follow_redirects=False,
    )
    assert response.status_code in (404, 400)
    assert "root:x:0:0" not in response.text


def test_absolute_path_style_request_does_not_escape(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """A request path that LOOKS like an absolute filesystem path (e.g.
    `/etc/passwd`) is still just a URL path relative to the served
    directory -- it must never resolve to a real absolute path on disk."""
    url = server.start(served_dir)
    response = httpx.get(url.rstrip("/") + "/etc/passwd", timeout=5.0)
    assert response.status_code == 404


def test_a_symlink_inside_the_served_directory_pointing_outside_is_denied(
    server: FeedDirectoryServer, served_dir: Path, tmp_path: Path
) -> None:
    """The load-bearing traversal test: `translate_path` alone considers
    `served_dir/escape.xml` an in-bounds path (it never leaves the served
    directory syntactically), but the symlink's TARGET is outside it.
    Without the containment check in `_ContainedRequestHandler.
    translate_path`, `open()` would follow the link and serve the secret
    file's content; with it, this must 404.
    """
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    secret_file = secret_dir / "outside.txt"
    secret_file.write_text("do not serve me", encoding="utf-8")

    symlink_path = served_dir / "escape.xml"
    symlink_path.symlink_to(secret_file)
    assert symlink_path.resolve() == secret_file.resolve()

    url = server.start(served_dir)
    response = httpx.get(url + "escape.xml", timeout=5.0)
    assert response.status_code == 404
    assert "do not serve me" not in response.text


def test_a_symlink_inside_the_served_directory_pointing_inside_still_works(
    server: FeedDirectoryServer, served_dir: Path
) -> None:
    """The containment check must not be so broad it blocks a symlink that
    resolves INSIDE the served directory -- only escapes are denied."""
    target = served_dir / "feed.xml"
    link = served_dir / "alias.xml"
    link.symlink_to(target)

    url = server.start(served_dir)
    response = httpx.get(url + "alias.xml", timeout=5.0)
    assert response.status_code == 200
    assert response.text == target.read_text(encoding="utf-8")


# --- config defaults (never auto-start) --------------------------------------


def test_configured_bind_and_port_defaults_to_loopback_and_ephemeral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_cli_setting(section, key, default=None):
        assert section == "briefings_feed_server"
        return default

    monkeypatch.setattr(feed_server_module, "get_cli_setting", _fake_get_cli_setting)
    bind, port = configured_bind_and_port()
    assert bind == "127.0.0.1"
    assert port == 0


def test_configured_bind_and_port_reads_a_stored_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_cli_setting(section, key, default=None):
        if section == "briefings_feed_server" and key == "bind":
            return "0.0.0.0"
        if section == "briefings_feed_server" and key == "port":
            return 8123
        return default

    monkeypatch.setattr(feed_server_module, "get_cli_setting", _fake_get_cli_setting)
    bind, port = configured_bind_and_port()
    assert bind == "0.0.0.0"
    assert port == 8123


def test_configured_bind_and_port_falls_back_to_ephemeral_on_a_bad_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hand-edited config with a non-integer port must not raise --
    `configured_bind_and_port` degrades to the ephemeral default instead."""

    def _fake_get_cli_setting(section, key, default=None):
        if section == "briefings_feed_server" and key == "port":
            return "not-a-port"
        return default

    monkeypatch.setattr(feed_server_module, "get_cli_setting", _fake_get_cli_setting)
    _bind, port = configured_bind_and_port()
    assert port == 0


def test_nothing_in_this_module_starts_a_server_on_import_or_config_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #2, at the module's own boundary: reading configured defaults
    must never itself bind a socket. Proven by binding the one port
    `configured_bind_and_port`'s default would name (ephemeral, so this
    asserts the READ path never touches a socket at all, via a spy)."""
    calls: list[tuple] = []
    real_socket = socket.socket

    def _spy_socket(*args, **kwargs):
        calls.append((args, kwargs))
        return real_socket(*args, **kwargs)

    monkeypatch.setattr(socket, "socket", _spy_socket)
    configured_bind_and_port()
    assert calls == [], "reading config defaults must never open a socket"
