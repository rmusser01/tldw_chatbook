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
from pathlib import Path

import httpx
import pytest

from tldw_chatbook.Subscriptions import feed_server as feed_server_module
from tldw_chatbook.Subscriptions.feed_server import (
    FeedDirectoryServer,
    FeedServerError,
    configured_bind_and_port,
)

pytestmark = pytest.mark.unit


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


# --- traversal matrix (AC #3) ------------------------------------------------


def test_dotdot_traversal_does_not_escape_the_served_directory(
    server: FeedDirectoryServer, served_dir: Path, tmp_path: Path
) -> None:
    """A sibling directory holds a 'secret' the served directory has no
    business exposing; `../`-shaped requests must never return it."""
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "passwd").write_text("root:x:0:0", encoding="utf-8")

    url = server.start(served_dir)
    response = httpx.get(
        url + "../secret/passwd", timeout=5.0, follow_redirects=False
    )
    assert response.status_code in (404, 400)
    assert "root:x:0:0" not in response.text


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
