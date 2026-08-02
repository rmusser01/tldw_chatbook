"""Opt-in, localhost-by-default static file server for one exported
briefings podcast feed directory at a time (task-1760).

**Why this exists, and why it is not `[web_server]`.** The phase 3 design
(`Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md`,
"Exports and feed") assumed the app's existing `[web_server]` config could
serve an exported feed directory over localhost. Task-1760's own
investigation (see that task file's Description) found this false on every
count that matters: `[web_server]`'s `enabled` key is read by no code at
all, it is a *mutually exclusive* process mode (`app.py`'s `--serve` flag
runs `run_web_server(...)` and returns, instead of ever building the TUI),
and even engaged it only serves textual-serve's own bundled JS asset, never
a directory the caller names. This module is therefore standalone,
purpose-built, net-new server surface, with its own config section
(`[briefings_feed_server]`) -- never `[web_server]`'s.

**Security posture (Task-1760 AC #4) -- read before wiring this up
anywhere else.**

- **Bind scope.** Defaults to `127.0.0.1` (loopback only) -- nothing off
  the local machine can reach it unless an operator deliberately widens
  `bind` (e.g. to `0.0.0.0`) in `[briefings_feed_server]`. Widening it
  exposes the served directory to the whole LAN (or further, behind a
  router configured for it) with no authentication at all -- see the next
  point.
- **No authentication, by design, stated plainly rather than implied.**
  There is no login, token, or IP allow-list. Anyone who can open a TCP
  connection to the bound address and port can read every file under the
  served directory for as long as it is running. This is acceptable for
  its one intended use (pointing a podcast client at a feed you exported
  yourself, on a loopback address only you can reach) and unacceptable for
  anything wider -- the UI's own toast states this every time serving
  starts, not just this docstring.
- **Path containment.** `_ContainedRequestHandler` resolves every request
  path (following symlinks) and refuses (404) anything that resolves
  outside the served directory. `SimpleHTTPRequestHandler.translate_path`
  already collapses literal `..`/absolute-path traversal in the URL down to
  a path inside the served directory (it never joins a `..`-shaped or
  directory-shaped word onto the base at all -- see its own docstring), but
  it does **not** protect against a *symlink* planted inside the served
  directory whose target resolves elsewhere on disk: `open()`/`os.stat()`
  follow symlinks by default, and `translate_path` alone would hand back
  such a path as "inside" the directory (syntactically true) while its
  *resolved* target is not. The containment check below is exactly that
  second, independent check.
- **Method surface.** Only `GET`/`HEAD` are served; every other verb
  (`POST`, `PUT`, `DELETE`, `PATCH`) gets `405 Method Not Allowed`. This is
  a read-only static file server -- nothing it serves is meant to be
  written to.
- **Session-only, opt-in, off by default (AC #2).** Nothing in this
  module runs at import time, app startup, or config load --
  `FeedDirectoryServer.start()` must be called explicitly, and only the
  UI's Serve action calls it, only when a user presses it.
  `configured_bind_and_port()` below reads `[briefings_feed_server]` for
  *defaults*, never to auto-start anything.

See `Docs/User_Guide/watchlists.md`'s "Serving an exported feed" section
for the user-facing version of this same posture.
"""

from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

from loguru import logger

from ..config import get_cli_setting
from ..Utils.path_validation import validate_path_simple

#: `[briefings_feed_server]`'s own section name -- deliberately its own
#: config section, never `[web_server]`'s (module docstring). Only ever
#: read by `configured_bind_and_port()`, which supplies *defaults* for the
#: UI's Serve action; nothing here auto-starts from it.
_CONFIG_SECTION = "briefings_feed_server"

#: What `_ContainedRequestHandler.translate_path` hands back for a request
#: that resolves outside the served directory -- an ordinary, vanishingly
#: unlikely-to-exist filename (never a null byte or other invalid-path
#: sentinel; see that method's own comment for why the distinction
#: matters), so the caller's own `open()` raises a plain `FileNotFoundError`
#: and the stdlib's existing 404 handling takes over unchanged.
_DENIED_PATH_NAME = ".feed-server-path-denied-3f6c9d21"


class FeedServerError(RuntimeError):
    """Raised when the feed server cannot be started.

    Covers both "already running" (this class serves exactly one
    directory at a time -- see `FeedDirectoryServer.start`'s own docstring
    for why a second `start()` call refuses rather than restarting) and a
    destination that fails validation (missing, not a directory, or
    otherwise rejected by `Utils.path_validation.validate_path_simple`).
    """


def configured_bind_and_port() -> tuple[str, int]:
    """The `[briefings_feed_server]` `bind`/`port` defaults, from config.

    Uses the traditional three-argument `get_cli_setting(section, key,
    default)` form throughout -- never the two-argument dotted form, whose
    "a caller default in the second positional slot" heuristic this repo
    has already been bitten by once (TASK-1771). Reading this function
    never starts anything; it only supplies defaults for the explicit
    Serve action to pass to `FeedDirectoryServer.start`.

    Returns:
        `(bind, port)`. `bind` defaults to `"127.0.0.1"` (loopback-only --
        module docstring). `port` defaults to `0` (ephemeral: the OS picks
        any free port, reported back by `FeedDirectoryServer.start`'s
        return value) and falls back to `0` if the configured value is not
        a valid integer, rather than raising on a hand-edited config.
    """
    bind = str(get_cli_setting(_CONFIG_SECTION, "bind", "127.0.0.1"))
    raw_port = get_cli_setting(_CONFIG_SECTION, "port", 0)
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        logger.debug(
            "briefings_feed_server.port is not a valid integer ({}); "
            "using an ephemeral port instead.",
            type(raw_port).__name__,
        )
        port = 0
    return bind, port


class _ContainedRequestHandler(http.server.SimpleHTTPRequestHandler):
    """`SimpleHTTPRequestHandler(directory=...)`, hardened for AC #3.

    Two changes from the stdlib handler, both stated in the module
    docstring's Security posture: a resolved-path containment check on top
    of `translate_path` (closes the symlink-escape gap `translate_path`
    alone leaves open), and GET/HEAD only (every other verb gets 405).
    """

    def translate_path(self, path: str) -> str:  # noqa: D102 - stdlib override
        """`super().translate_path`, then refuse anything that resolves
        outside the served directory.

        `self.directory` (set by `SimpleHTTPRequestHandler.__init__` from
        the `directory=` keyword `FeedDirectoryServer.start` passes in via
        `functools.partial`) is resolved fresh on every call rather than
        cached -- this handler is re-instantiated per request by
        `socketserver`, so there is no meaningful cost to doing so, and it
        keeps this method self-contained.

        A candidate that fails resolution (`OSError` -- e.g. a path
        component that is not actually a directory) or that resolves
        outside the served root (`ValueError` from `Path.relative_to`) is
        answered with a path that cannot exist, so the caller's own
        `os.stat`/`open` naturally 404s -- there is no need to override
        `send_head` or duplicate its error handling here.
        """
        candidate = super().translate_path(path)
        served_root = Path(self.directory).resolve(strict=False)
        try:
            resolved_candidate = Path(candidate).resolve(strict=False)
            resolved_candidate.relative_to(served_root)
        except (OSError, ValueError):
            # A plain, almost-certainly-absent filename -- NOT a null byte
            # or other invalid-path sentinel: `send_head()`'s `open(path,
            # 'rb')` only converts `OSError` into a 404 response; a
            # `ValueError` (which an embedded null byte raises on some
            # platforms) propagates out of the request handler instead and
            # tears down the connection, which is not the answer AC #3
            # asks for. A `FileNotFoundError` (an `OSError` subclass) is
            # exactly what a plain missing name inside `served_root`
            # produces, and that IS handled -> a clean 404.
            return str(served_root / _DENIED_PATH_NAME)
        return candidate

    def _method_not_allowed(self) -> None:
        self.send_error(405, "Method Not Allowed")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming convention
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802 - stdlib naming convention
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib naming convention
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib naming convention
        self._method_not_allowed()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        """Route request logging to loguru debug instead of stderr.

        Request *paths* are logged at debug -- they are local requests
        against a directory the user themselves chose to export and serve,
        never file contents (module docstring). This overrides both the
        default per-request access log and `log_error` (which
        `BaseHTTPRequestHandler.log_error` implements by calling this same
        method), so a malformed request never reaches stderr either.
        """
        logger.debug(
            "Feed server request: {} {} ({})",
            self.command,
            self.path,
            (format % args) if args else format,
        )


class FeedDirectoryServer:
    """Starts/stops a `ThreadingHTTPServer` for one feed directory at a
    time, on a daemon thread.

    Not a module-level singleton -- each `WatchlistsCollectionsScreen`
    instance owns exactly one of these (mirroring how `ArtifactsPane`'s own
    module docstring describes the shared audio player: state that must
    survive a recompose lives off the recomposed widget). `start()` refuses
    a second call while already running (see its own docstring for why
    that is simpler than restarting), so this class enforces "one
    directory at a time" itself -- a caller does not have to.
    """

    def __init__(self) -> None:
        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None

    @property
    def is_running(self) -> bool:
        return self._httpd is not None

    @property
    def url(self) -> str | None:
        """The bound URL from the most recent successful `start()`, or
        `None` before the first `start()` or after `stop()`."""
        return self._url

    def start(self, directory: Path, *, bind: str = "127.0.0.1", port: int = 0) -> str:
        """Serve `directory` over HTTP, GET/HEAD only, until `stop()`.

        Refuses (raises) rather than restarting when already running --
        the simpler of the two choices task-1760's plan named as
        equally valid, and the one this whole class + the UI's Serve/Stop
        handlers are built around: a caller that wants to serve a
        different directory stops the current one first. Restarting
        in-place would need to reconcile "was the old directory's server
        still mid-request" with starting a new one; refusing sidesteps
        that entirely, at the cost of one extra Stop press.

        Args:
            directory: The directory to serve. Validated with
                `Utils.path_validation.validate_path_simple(...,
                require_exists=True)` -- the same validator `briefing_
                export.export_feed_directory` already applies to this
                exact directory when it was exported -- so a directory
                deleted or replaced with something hostile between export
                and serve is rejected here too, not merely trusted because
                it was validated once before.
            bind: The address to bind. Defaults to loopback-only (module
                docstring's Security posture) -- pass a wider address only
                when the caller has deliberately chosen to widen exposure.
            port: The port to bind, or `0` for an ephemeral port (the OS
                picks any free one). The actual bound port is always
                reported back in the returned URL, regardless of which was
                requested.

        Returns:
            The bound URL, e.g. `"http://127.0.0.1:54231/"`.

        Raises:
            FeedServerError: Already running, or `directory` fails
                validation.
            OSError: The bind itself fails (e.g. a fixed `port` already in
                use).
        """
        if self.is_running:
            raise FeedServerError(
                f"A feed is already being served at {self._url}. Stop it "
                "before serving a different directory."
            )
        try:
            resolved_directory = validate_path_simple(
                directory, require_exists=True
            ).resolve()
        except ValueError as exc:
            raise FeedServerError(f"Cannot serve this directory: {exc}") from exc
        if not resolved_directory.is_dir():
            raise FeedServerError(f"{resolved_directory} is not a directory.")

        handler_cls = functools.partial(
            _ContainedRequestHandler, directory=str(resolved_directory)
        )
        httpd = http.server.ThreadingHTTPServer((bind, port), handler_cls)
        # daemon_threads: a request handled by ThreadingHTTPServer's own
        # per-connection thread pool must never block interpreter exit --
        # the OUTER thread `serve_forever` runs on (below) is daemon=True
        # for the identical reason.
        httpd.daemon_threads = True
        actual_port = httpd.server_address[1]

        thread = threading.Thread(
            target=httpd.serve_forever,
            name="briefings-feed-server",
            daemon=True,
        )
        thread.start()

        self._httpd = httpd
        self._thread = thread
        self._url = f"http://{bind}:{actual_port}/"
        logger.debug(
            "Feed directory server started on {}:{}", bind, actual_port
        )
        return self._url

    def stop(self) -> None:
        """Stop serving, releasing the socket. A no-op if not running.

        `shutdown()` + `server_close()` is the documented stdlib pair for
        stopping a `serve_forever()` loop running on another thread; the
        `join()` afterward waits for that thread to actually exit before
        returning, so a caller that immediately calls `start()` again (or
        checks `is_running`) never races the old thread's teardown.
        """
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._httpd = None
        self._thread = None
        self._url = None
        logger.debug("Feed directory server stopped.")
