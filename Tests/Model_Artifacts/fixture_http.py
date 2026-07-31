"""Localhost HTTP fixture for artifact-fetch tests. stdlib only."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class _Route:
    """One configured response for a fixture path."""

    body: bytes
    etag: str | None
    support_range: bool
    disconnect_after: int | None
    require_token: str | None
    last_modified: str | None = None


class FixtureArtifactServer:
    """Configurable localhost server: Range, ETag, faults, auth.

    Use as a context manager so the background thread and socket are always
    torn down, even on test failure::

        with FixtureArtifactServer() as srv:
            srv.serve("/f.bin", b"...")
            ... srv.url("/f.bin") ...
    """

    def __init__(self) -> None:
        self._routes: dict[str, _Route] = {}
        self.requests: dict[str, list[dict]] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _route(self):
                return outer._routes.get(self.path)

            def do_HEAD(self):
                self._respond(head_only=True)

            def do_GET(self):
                self._respond(head_only=False)

            def _respond(self, *, head_only: bool):
                route = self._route()
                outer.requests.setdefault(self.path, []).append(dict(self.headers))
                if route is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                if route.require_token and (
                    self.headers.get("Authorization") != f"Bearer {route.require_token}"
                ):
                    self.send_response(401)
                    self.end_headers()
                    return
                body = route.body
                start = 0
                status = 200
                range_header = self.headers.get("Range")
                if_range = self.headers.get("If-Range")
                honor_range = bool(range_header) and route.support_range
                if honor_range and if_range is not None:
                    # Real servers refuse a stale If-Range by falling back to
                    # a full 200 -- the behavior fetch.py's resume-mismatch
                    # detection depends on. Match against whichever validator
                    # this route currently has (an ETag comparison and a
                    # Last-Modified comparison are mutually exclusive in
                    # practice since the client sends only one).
                    current_validator = route.etag or route.last_modified
                    if if_range != current_validator:
                        honor_range = False
                if honor_range:
                    start = int(range_header.split("=")[1].split("-")[0])
                    body = body[start:]
                    status = 206
                self.send_response(status)
                if route.etag:
                    self.send_header("ETag", route.etag)
                if route.last_modified:
                    self.send_header("Last-Modified", route.last_modified)
                if route.support_range:
                    self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if head_only:
                    return
                cut = route.disconnect_after
                if cut is not None and cut < len(body):
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.connection.close()  # simulate mid-body drop
                    return
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def serve(
        self,
        path: str,
        body: bytes,
        *,
        etag: str | None = '"v1"',
        weak_etag: bool = False,
        last_modified: str | None = None,
        support_range: bool = True,
        disconnect_after: int | None = None,
        require_token: str | None = None,
    ) -> None:
        """Register (or replace) a route this server responds to.

        Args:
            path: Request path, e.g. ``"/f.bin"``.
            body: Full response body for a non-Range request.
            etag: ETag value to send (``None`` to omit the header).
            weak_etag: Wrap ``etag`` as a weak validator (``W/"..."``).
            last_modified: ``Last-Modified`` header value to send (``None``
                to omit). Also doubles as the conditional-range validator
                when ``etag`` is ``None``, mirroring real servers.
            support_range: Whether ``Range`` requests are honored (206) or
                ignored (always 200 with the full body).
            disconnect_after: If set, write only this many bytes then close
                the connection mid-body (simulates a dropped transfer).
            require_token: If set, require ``Authorization: Bearer <token>``.
        """
        if etag and weak_etag:
            etag = f"W/{etag}"
        self._routes[path] = _Route(
            body=body,
            etag=etag,
            support_range=support_range,
            disconnect_after=disconnect_after,
            require_token=require_token,
            last_modified=last_modified,
        )

    def url(self, path: str) -> str:
        """Absolute ``http://127.0.0.1:<port><path>`` URL for this server."""
        host, port = self._server.server_address
        return f"http://{host}:{port}{path}"

    def __enter__(self) -> "FixtureArtifactServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._server.shutdown()
        self._server.server_close()
        return False
