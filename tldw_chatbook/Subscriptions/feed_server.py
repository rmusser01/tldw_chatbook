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
  point. A blank, whitespace-only, or non-string `bind` value can never
  silently cause this widening: `_normalize_bind` falls back to the safe
  loopback default instead (task-1760 review, M3), and whenever the
  address actually bound is not loopback -- whether from a deliberate
  widening or a config mistake that still resolved to a real address --
  `start()` logs a warning once, so exposure is never purely implicit.
- **No authentication, by design, stated plainly rather than implied.**
  There is no login, token, or IP allow-list. Anyone who can open a TCP
  connection to the bound address and port can read every file under the
  served directory **and every subdirectory beneath it** (recursively --
  the handler is a plain `SimpleHTTPRequestHandler`, not scoped to one
  level) for as long as it is running. This is acceptable for its one
  intended use (pointing a podcast client at a feed you exported yourself,
  on a loopback address only you can reach) and unacceptable for anything
  wider -- the UI's own toast states this every time serving starts, not
  just this docstring. Directory listings are refused (404, task-1760
  review M4: `_ContainedRequestHandler.list_directory`) -- nothing is
  *browsable* -- but every file whose name a client already knows (from
  `feed.xml`, or guessed) is still fetchable anywhere in the tree. Point
  this at a dedicated export folder, never a general-purpose directory
  like your home folder, for exactly that reason.
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
import ipaddress
import socket
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

#: The safe fallback bind address -- loopback-only (module docstring).
#: `_normalize_bind` returns exactly this for anything that is not itself
#: a non-blank string, so a blank/typo'd config value can never silently
#: resolve to "every interface" (task-1760 review, M3).
_SAFE_DEFAULT_BIND = "127.0.0.1"


def _normalize_bind(raw_bind: object) -> str:
    """Coerce a config-supplied `bind` value to a safe, non-empty string.

    Fix wave (task-1760 review, M3): `bind` used to be passed through
    `str(...)` with no further check, so a blank config value (`bind =
    ""`, read back as `""`) or a numeric typo (`bind = 0`, meant for
    `port`) reached `socket.bind` unchanged -- and both of those name
    "every interface" to the stdlib (`socket.bind(("", 0))` and
    `socket.bind(("0", 0))` both resolve to `0.0.0.0`), silently turning a
    loopback-only default into a LAN-wide one. Anything that is not
    itself a non-blank string -- `None`, an int, a float, an empty or
    whitespace-only string -- falls back to the safe loopback default
    instead; a *non-empty* string (including `"0.0.0.0"`) is trusted as a
    deliberate choice and passed through unchanged (widening is not
    forbidden -- module docstring -- only an accidental blank/typo'd value
    silently causing it is).
    """
    if isinstance(raw_bind, str) and raw_bind.strip():
        return raw_bind.strip()
    return _SAFE_DEFAULT_BIND


def is_loopback_bind(bind: str) -> bool:
    """True if `bind` names a loopback-only address.

    Covers the hostname `localhost` and any address `ipaddress` reports as
    loopback (`127.0.0.0/8`, `::1`). Anything else -- a real interface
    address, `0.0.0.0`, `::`, a LAN hostname, an unparsable string -- is
    treated as reachable from somewhere other than this machine. Exposed
    (not `_`-prefixed) so the UI layer can word its own toast/warning
    around the same test `FeedDirectoryServer.start` uses (task-1760
    review, M3) rather than re-implementing it.

    Args:
        bind: The bind address to classify, e.g. `"127.0.0.1"`, `"::1"`,
            `"localhost"`, `"0.0.0.0"`, or a LAN hostname/address. Not
            normalized by the caller first -- this function does its own
            `.strip().lower()` -- so the raw, already-`_normalize_bind`-d
            value `FeedDirectoryServer.start`/`configured_bind_and_port`
            use is safe to pass in directly.

    Returns:
        `True` for `localhost` or any address `ipaddress.ip_address`
        reports as loopback (`127.0.0.0/8`, `::1`); `False` for every other
        value, including one `ipaddress` cannot parse at all (a LAN
        hostname is not an error here -- it is simply not loopback).
    """
    normalized = bind.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _is_ipv6_literal(bind: str) -> bool:
    """True if `bind` parses as an IPv6 address literal (e.g. `"::1"`).

    Shared by `_format_host_for_url` (URL bracketing) and
    `FeedDirectoryServer.start` (socket address-family selection --
    see `_IPv6ThreadingHTTPServer`) so the two never disagree about what
    counts as IPv6. A hostname such as `"localhost"` or an IPv4 address
    makes `ipaddress.ip_address` raise `ValueError`, which is treated as
    "not IPv6" here -- the same guard `is_loopback_bind` uses.
    """
    try:
        return ipaddress.ip_address(bind).version == 6
    except ValueError:
        return False


def _format_host_for_url(bind: str) -> str:
    """Render `bind` as it belongs in a URL authority component.

    Fix wave (task-1760 Qodo fix wave, F2): `FeedDirectoryServer.start`
    used to build `self._url` as `f"http://{bind}:{port}/"` unconditionally
    -- correct for an IPv4 address or a hostname, but not for an IPv6
    literal: `http://::1:8080/` is ambiguous (is `8080` another address
    group, or the port?) and no URL parser accepts it. A URL's authority
    component requires an IPv6 literal to be bracketed --
    `http://[::1]:8080/` -- per RFC 3986 section 3.2.2, which
    `is_loopback_bind` already implicitly relies on `::1` being a valid
    bind target for (its own docstring names `::1` as a loopback address
    this module supports).

    Args:
        bind: The bind address to format, already `_normalize_bind`-d --
            an IPv4 address, an IPv6 address, or a hostname such as
            `"localhost"`.

    Returns:
        `bind` unchanged for an IPv4 address or a hostname. Bracketed
        (`f"[{bind}]"`) when `bind` is an IPv6 literal (`_is_ipv6_literal`).
    """
    if _is_ipv6_literal(bind):
        return f"[{bind}]"
    return bind


class _IPv6ThreadingHTTPServer(http.server.ThreadingHTTPServer):
    """`ThreadingHTTPServer` with `address_family` forced to IPv6.

    Fix wave (task-1760 Qodo fix wave, F2, discovered while verifying the
    URL-bracketing fix): `http.server.HTTPServer` -- `ThreadingHTTPServer`'s
    own base class -- hard-codes `address_family = socket.AF_INET` as a
    class attribute; `socketserver.TCPServer.__init__` never inspects the
    bind *address* itself to pick a family. Binding an IPv6 literal (e.g.
    `"::1"`) through the base class therefore fails immediately with
    `socket.gaierror` (`getaddrinfo` cannot resolve an IPv6 literal for an
    `AF_INET` socket) on every platform -- independent of, and before,
    whatever `is_loopback_bind`/`_format_host_for_url` support for IPv6
    would otherwise mean in practice. `FeedDirectoryServer.start` selects
    this subclass instead of the base one whenever `bind` is an IPv6
    literal (`_is_ipv6_literal`); IPv4/hostname binds are unaffected.
    """

    address_family = socket.AF_INET6


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
        module docstring) and falls back to that same safe default (never
        an empty string) for a blank, whitespace-only, or non-string
        configured value -- see `_normalize_bind` (task-1760 review, M3).
        `port` defaults to `0` (ephemeral: the OS picks any free port,
        reported back by `FeedDirectoryServer.start`'s return value) and
        falls back to `0` if the configured value is not a valid integer,
        rather than raising on a hand-edited config. A value that DOES
        parse as an integer but falls outside the valid `0..65535` range
        (e.g. a typo like `99999`) falls back to `0` the same way -- see
        the Qodo fix-wave note below.
    """
    bind = _normalize_bind(get_cli_setting(_CONFIG_SECTION, "bind", _SAFE_DEFAULT_BIND))
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
    else:
        # task-1760 Qodo fix wave, F1: a value that DOES coerce to `int`
        # can still be out of the valid TCP port range (a typo like
        # `99999`, or a negative value) -- `int(...)` alone does not catch
        # that, and passing it straight through to `FeedDirectoryServer.
        # start` would reach `ThreadingHTTPServer`'s underlying
        # `socket.bind` as a bare `OverflowError`, which nothing in the UI
        # layer catches (`watchlists_collections_screen.py`'s Serve
        # handler only catches `FeedServerError`/`OSError`). A CONFIG-
        # derived bad value degrades safely here, the same way a blank/
        # typo'd `bind` already does above -- only a bad value handed
        # directly to `start()` by a caller that bypasses this function
        # is instead treated as a programming error and refused there
        # (see that method's own comment). The warning is type-only, not
        # value-only, matching this module's established convention for
        # logging a config value that turned out not to be trustworthy.
        if not (0 <= port <= 65535):
            logger.warning(
                "briefings_feed_server.port is outside the valid 0-65535 "
                "range (configured value was a {}); using an ephemeral "
                "port instead.",
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

    def list_directory(self, path: str):  # noqa: D102 - stdlib override
        """Refuse to render a directory listing; 404 instead.

        Fix wave (task-1760 review, M4): a podcast feed is fetched by URLs
        named in the served `feed.xml` -- nothing ever needs to browse the
        served directory itself, and an auto-generated index is pure
        exposure surface on top of that (every filename in the directory,
        and every subdirectory beneath it, becomes visible to anyone who
        can reach the bound address). Recursive *file* serving stays
        intact -- an episode named in `feed.xml` from a subfolder must
        still resolve -- only the browsable-index behaviour is removed.
        `SimpleHTTPRequestHandler.send_head` treats a `None` return from
        this method as "already handled, nothing further to do" (its own
        docstring), so a plain `send_error` here is the complete override.
        """
        self.send_error(404, "File not found")
        return None

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        """Route request logging to loguru debug instead of stderr.

        Request *paths* are logged at debug -- they are local requests
        against a directory the user themselves chose to export and serve,
        never file contents (module docstring). This overrides both the
        default per-request access log and `log_error` (which
        `BaseHTTPRequestHandler.log_error` implements by calling this same
        method), so a malformed request never reaches stderr either.

        Fix wave (task-1760 review, M1): `BaseHTTPRequestHandler.parse_
        request` sets `self.command = None` *before* `self.path` exists,
        and calls `send_error` -> `log_error` -> this method on every
        malformed-request-line path (an empty line, a bad HTTP version, an
        over-long request line) that runs before `self.command, self.path
        = words[:2]` ever executes. Reading `self.path` there raised
        `AttributeError`, which killed the response (the client got zero
        bytes instead of the intended 400/414) and the handler thread with
        it -- and produced no log line at all, the opposite of this
        method's own purpose. `getattr(..., "-")` never touches a missing
        attribute, so logging itself can never be the reason a malformed
        request goes unanswered.
        """
        logger.debug(
            "Feed server request: {} {} ({})",
            getattr(self, "command", "-"),
            getattr(self, "path", "-"),
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
        self._bind: str | None = None
        self._directory: Path | None = None

    @property
    def is_running(self) -> bool:
        return self._httpd is not None

    @property
    def url(self) -> str | None:
        """The bound URL from the most recent successful `start()`, or
        `None` before the first `start()` or after `stop()`."""
        return self._url

    @property
    def bind(self) -> str | None:
        """The normalized bind address actually used by the most recent
        successful `start()` (never the raw, possibly-blank value a caller
        passed in -- see `_normalize_bind`), or `None` before the first
        `start()` or after `stop()`. Exposed so a caller (the UI's Serve
        handler) can word its own exposure warning with `is_loopback_bind`
        without re-deriving the address from `url` (task-1760 review, M3).
        """
        return self._bind

    @property
    def directory(self) -> Path | None:
        """The resolved directory the most recent successful `start()` is
        serving, or `None` before the first `start()` or after `stop()`.
        Lets a caller (the export toast, task-1760 review, L1) tell
        whether a fresh export landed in the directory actually being
        served, without keeping its own duplicate bookkeeping.
        """
        return self._directory

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
                A blank/whitespace-only string (or any non-string value)
                is normalized to the safe loopback default instead of
                being passed through -- see `_normalize_bind` (task-1760
                review, M3): a caller does not have to pre-validate this
                itself, and cannot silently widen exposure by accident.
            port: The port to bind, or `0` for an ephemeral port (the OS
                picks any free one). Must be an `int` in `0..65535` --
                anything else (a negative value, a value above 65535, or a
                non-`int`) is refused with `FeedServerError` rather than
                reaching the socket layer (task-1760 Qodo fix wave, F1): an
                out-of-range-but-parseable port previously reached
                `ThreadingHTTPServer` as a bare `OverflowError`, which is
                not one of the types the UI's Serve handler catches. This
                is the boundary for a bad value handed directly to this
                method (a script, a test, a future caller); a bad value
                that came from *config* is instead normalized to a safe
                ephemeral fallback one layer up, in
                `configured_bind_and_port()` -- a hand-edited config
                degrading safely is a UX nicety, not the last line of
                defence, so the two paths are handled differently on
                purpose. The actual bound port is always reported back in
                the returned URL, regardless of which was requested.

        Returns:
            The bound URL, e.g. `"http://127.0.0.1:54231/"` -- or, for an
            IPv6 `bind` (e.g. `"::1"`), the bracketed form a URL requires,
            e.g. `"http://[::1]:54231/"` (task-1760 Qodo fix wave, F2).

        Raises:
            FeedServerError: Already running, `directory` fails
                validation, or `port` is not an `int` in `0..65535`.
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

        # task-1760 review, M3: normalize here too, not only in
        # `configured_bind_and_port` -- this method is the actual socket
        # boundary, and a future direct caller (a test, a script) must get
        # the same safe-by-default behaviour as the UI's Serve action.
        bind = _normalize_bind(bind)

        # task-1760 Qodo fix wave, F1: validate `port` here too, not only
        # in `configured_bind_and_port` -- this is the actual socket
        # boundary (the comment above, applied to `port` as well as
        # `bind`). `configured_bind_and_port` degrades a bad CONFIG value
        # safely (falls back to ephemeral `0` + a warning) because a
        # hand-edited config file is expected to sometimes be wrong; a bad
        # value reaching this method directly -- bypassing that
        # normalization entirely -- is instead a programming error and is
        # refused outright, the same way the directory-validation checks
        # above it already are. Left unchecked, an out-of-range-but-
        # parseable `int` (e.g. `99999`) would reach
        # `ThreadingHTTPServer.__init__` -> `socket.bind` as a bare
        # `OverflowError`, which is not one of the types the UI's Serve
        # handler (`watchlists_collections_screen.py`) catches -- it would
        # escape as an unhandled exception instead of the toast every
        # other rejection here produces.
        if not isinstance(port, int) or isinstance(port, bool) or not (0 <= port <= 65535):
            raise FeedServerError(
                f"Port {port!r} is not valid: it must be an integer between "
                "0 and 65535 (0 requests an OS-assigned ephemeral port)."
            )

        handler_cls = functools.partial(
            _ContainedRequestHandler, directory=str(resolved_directory)
        )
        # task-1760 Qodo fix wave, F2: `_IPv6ThreadingHTTPServer` for an
        # IPv6 `bind` -- see that class's own docstring for why the base
        # `ThreadingHTTPServer` cannot bind one at all (a hard-coded
        # `AF_INET` `address_family`, not a platform limitation). IPv4 and
        # hostname binds are unaffected -- same base class as before.
        server_cls = (
            _IPv6ThreadingHTTPServer if _is_ipv6_literal(bind) else http.server.ThreadingHTTPServer
        )
        httpd = server_cls((bind, port), handler_cls)
        # daemon_threads: a request handled by ThreadingHTTPServer's own
        # per-connection thread pool must never block interpreter exit --
        # the OUTER thread `serve_forever` runs on (below) is daemon=True
        # for the identical reason.
        httpd.daemon_threads = True
        actual_port = httpd.server_address[1]

        if not is_loopback_bind(bind):
            # task-1760 review, M3: nothing else here would ever tell an
            # operator that a blank/typo'd or deliberately widened `bind`
            # just made this directory reachable beyond this machine --
            # the module docstring's whole posture rests on that
            # distinction being visible, not merely documented. The bind
            # address itself is safe to log in full (an IP/hostname this
            # process itself resolved from config, never user or model
            # content), unlike the `type(exc).__name__`-only rule this
            # module follows for caught exceptions elsewhere.
            logger.warning(
                "Feed directory server binding to {} (not loopback-only) -- "
                "the served directory is reachable from beyond this "
                "machine while it is running. A wildcard IPv6 bind such as "
                "'::' is typically dual-stack (accepts plain IPv4 clients "
                "too), so the reachable surface can be wider than the "
                "address alone suggests.",
                bind,
            )

        thread = threading.Thread(
            target=lambda: httpd.serve_forever(poll_interval=0.05),
            name="briefings-feed-server",
            daemon=True,
        )
        thread.start()

        self._httpd = httpd
        self._thread = thread
        # task-1760 Qodo fix wave, F2: bracket an IPv6 `bind` literal here
        # -- `_format_host_for_url` -- since `is_loopback_bind` explicitly
        # supports `::1` as a bind target and an unbracketed IPv6 URL is
        # not merely stylistically off, it is not a URL a client can parse
        # at all.
        self._url = f"http://{_format_host_for_url(bind)}:{actual_port}/"
        self._bind = bind
        self._directory = resolved_directory
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

        Fix wave (task-1760 review, M2): `shutdown()` blocks until the
        `serve_forever()` loop's *next* `selector.select(poll_interval)`
        call returns -- at the stdlib's own 0.5s default `poll_interval`
        (unset before this fix wave), that measured ~501ms synchronous on
        the Textual event loop, since both callers of this method
        (`WatchlistsCollectionsScreen.handle_stop_feed_server_requested`
        and its `on_unmount`) call it directly rather than via a worker.
        `start()` now runs the loop at `poll_interval=0.05`, bounding this
        call to roughly a tenth of that -- no change needed here, since
        the bound comes entirely from how the loop it is joining was
        started.
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
        self._bind = None
        self._directory = None
        logger.debug("Feed directory server stopped.")
