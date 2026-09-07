# artifact_share_server.py
"""Child-process HTTP server exposing one staged artifact share.

Run as ``python -m tldw_chatbook.Web_Server.artifact_share_server <manifest>``
by the app-side controller. Serves ONLY files inside the staging directory
(the manifest's parent). aiohttp is imported lazily so this module (and its
tests) import cleanly without the ``[web]`` extra.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import html
import os
import signal
import threading
import time
import urllib.parse
from pathlib import Path

from loguru import logger

from ..Utils.atomic_file_ops import atomic_write_json
from .artifact_share_manifest import (
    ArtifactShareAuth,
    ArtifactShareError,
    ArtifactShareManifest,
    load_manifest,
    verify_share_auth,
)

_AUTH_FAILURE_THRESHOLD = 10
_AUTH_LOCKOUT_SECONDS = 30.0
_VERIFIED_CACHE_LIMIT = 64

_BASE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
    "Cache-Control": "no-store",
}

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{share_name}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 46rem; padding: 0 1rem; }}
h1 {{ font-size: 1.4rem; }}
article {{ border: 1px solid #ccc; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
.desc {{ color: #444; }}
.meta {{ color: #777; font-size: .9rem; }}
.btn {{ display: inline-block; margin-top: .5rem; padding: .4rem .9rem; border-radius: 6px;
       background: #0b63a3; color: #fff; text-decoration: none; }}
footer {{ margin-top: 2rem; color: #777; font-size: .85rem; }}
</style>
</head>
<body>
<h1>{share_name}</h1>
<p>{count} artifact(s) available.</p>
{auth_note}
{rows}
<p><a class="btn" href="/bundle.zip" download>Download all</a></p>
<footer>Import these bundles into tldw_chatbook via Chatbooks &rarr; Import.</footer>
</body>
</html>
"""


class ArtifactShareServer:
    """Serve one staged share directory described by a manifest."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self.manifest: ArtifactShareManifest = load_manifest(self.manifest_path)
        self.staging_dir = self.manifest_path.parent.resolve()
        self._verified: set[tuple[str, str]] = set()
        self._failures: dict[str, tuple[int, float]] = {}

    # -- routes -------------------------------------------------------------

    def build_app(self):
        from aiohttp import web

        app = web.Application(client_max_size=1)
        app.middlewares.append(self._headers_middleware)
        if self.manifest.auth is not None:
            app.middlewares.append(self._auth_middleware)
        app.router.add_get("/", self.handle_index)
        app.router.add_get("/index.json", self.handle_index_json)
        app.router.add_get("/artifact/{key}", self.handle_artifact)
        app.router.add_get("/bundle.zip", self.handle_bundle)
        return app

    async def _headers_middleware(self, request, handler):
        from aiohttp import web

        try:
            response = await handler(request)
        except web.HTTPException as exc:
            for name, value in _BASE_HEADERS.items():
                exc.headers[name] = value
            raise
        for name, value in _BASE_HEADERS.items():
            response.headers[name] = value
        return response

    # aiohttp >=3.14 only honors new-style (request, handler) middleware when
    # marked; ``web.middleware`` just sets this attribute, which we do directly
    # to keep aiohttp lazily imported.
    _headers_middleware.__middleware_version__ = 1  # type: ignore[attr-defined]

    async def _auth_middleware(self, request, handler):
        from aiohttp import web

        auth = self.manifest.auth
        assert auth is not None  # middleware only installed when auth is set
        peer = request.remote or "unknown"
        now = time.monotonic()
        failures, locked_until = self._failures.get(peer, (0, 0.0))
        if locked_until > now:
            raise web.HTTPTooManyRequests(text="Too many failed attempts; retry shortly.")
        username, password = _parse_basic_auth(request.headers.get("Authorization", ""))
        if username is not None and (username, password) in self._verified:
            return await handler(request)
        if username is not None and verify_share_auth(auth, username, password):
            if len(self._verified) >= _VERIFIED_CACHE_LIMIT:
                self._verified.clear()
            self._verified.add((username, password))
            self._failures.pop(peer, None)
            return await handler(request)
        count = failures + 1
        locked_until = now + _AUTH_LOCKOUT_SECONDS if count >= _AUTH_FAILURE_THRESHOLD else 0.0
        self._failures[peer] = (count, locked_until)
        logger.warning(f"Artifact share auth failure from {peer} (attempt {count})")
        raise web.HTTPUnauthorized(
            headers={"WWW-Authenticate": 'Basic realm="tldw chatbook artifact share"'}
        )

    _auth_middleware.__middleware_version__ = 1  # type: ignore[attr-defined]

    async def handle_index(self, request):
        from aiohttp import web

        return web.Response(text=self._render_index(), content_type="text/html")

    async def handle_index_json(self, request):
        import json as _json

        from aiohttp import web

        payload = {
            "share_name": self.manifest.share_name,
            "created_at": self.manifest.created_at,
            "artifacts": [
                {
                    "key": item.key,
                    "name": item.display_name,
                    "description": item.description,
                    "kind": item.kind,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in self.manifest.artifacts
            ],
        }
        return web.Response(
            text=_json.dumps(payload, ensure_ascii=False),
            content_type="application/json",
        )

    async def handle_artifact(self, request):
        from aiohttp import web

        key = request.match_info["key"]
        try:
            path, display_name = self._resolve_staged(key)
        except _UnknownKeyError:
            raise web.HTTPNotFound(text="No such artifact.")
        except _GoneKeyError:
            raise web.HTTPGone(text="This artifact is no longer available.")
        return self._file_response(path, display_name)

    async def handle_bundle(self, request):
        bundle = self.staging_dir / "bundle.zip"
        if not bundle.is_file():
            from aiohttp import web

            raise web.HTTPGone(text="Bundle is no longer available.")
        return self._file_response(bundle, f"{self.manifest.share_name}-bundle.zip")

    def _file_response(self, path: Path, display_name: str):
        from aiohttp import web

        return web.FileResponse(path, headers={"Content-Disposition": _content_disposition(display_name)})

    # -- helpers ------------------------------------------------------------

    def _resolve_staged(self, key: str) -> tuple[Path, str]:
        for item in self.manifest.artifacts:
            # Compare encoded bytes: str inputs raise TypeError inside
            # compare_digest when non-ASCII (e.g. /artifact/%E2%82%AC), which
            # would surface as a 500 that bypasses the headers middleware.
            if not hmac.compare_digest(item.key.encode("utf-8"), key.encode("utf-8")):
                continue
            staged = item.staged_name
            if Path(staged).name != staged:  # separators/traversal never resolve
                break
            candidate = (self.staging_dir / staged).resolve()
            try:
                candidate.relative_to(self.staging_dir)
            except ValueError:
                break
            if candidate.is_file():
                return candidate, item.display_name
            raise _GoneKeyError(key)
        raise _UnknownKeyError(key)

    def _render_index(self) -> str:
        rows = []
        for item in self.manifest.artifacts:
            name = html.escape(item.display_name, quote=True)
            description = html.escape(item.description or "", quote=True)
            size = (
                f"{item.size_bytes / (1024 * 1024):.1f} MB"
                if item.size_bytes >= 1024 * 1024
                else f"{max(item.size_bytes, 0) / 1024:.1f} KB"
            )
            rows.append(
                f"<article><h3>{name}</h3>"
                f"<p class=\"desc\">{description}</p>"
                f"<p class=\"meta\">{html.escape(item.kind)} &middot; {size}</p>"
                f"<a class=\"btn\" href=\"/artifact/{item.key}\" download>Download</a></article>"
            )
        auth_note = (
            "<p>Protected sharing is active.</p>" if self.manifest.auth is not None else ""
        )
        return _PAGE_TEMPLATE.format(
            share_name=html.escape(self.manifest.share_name or "Shared artifacts", quote=True),
            count=len(self.manifest.artifacts),
            auth_note=auth_note,
            rows="\n".join(rows),
        )

    # -- lifecycle ----------------------------------------------------------

    def run(self, host: str, port: int) -> str:
        """Serve until SIGTERM or orphaned; return the bound URL (after start)."""
        from aiohttp import web

        parent_pid = os.getppid()

        async def _serve() -> str:
            loop = asyncio.get_running_loop()
            stop = asyncio.Event()
            runner = web.AppRunner(self.build_app(), access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            bound_host, bound_port = runner.addresses[0][:2]
            url = f"http://{_format_host(bound_host)}:{bound_port}"

            def _watch_parent() -> None:
                while True:
                    if os.getppid() != parent_pid:
                        loop.call_soon_threadsafe(stop.set)
                        return
                    time.sleep(2.0)

            threading.Thread(target=_watch_parent, name="share-ppid-watch", daemon=True).start()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, stop.set)
                except NotImplementedError:  # pragma: no cover - windows
                    pass
            atomic_write_json(
                self.manifest_path.parent / "status.json",
                {
                    "url": url,
                    "pid": os.getpid(),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                mode=0o600,
                privacy_safe_log=True,
            )
            print(f"ARTIFACT_SHARE_READY {url}", flush=True)
            logger.info(
                f"Artifact share serving {len(self.manifest.artifacts)} artifact(s) at {url}"
            )
            await stop.wait()
            await runner.cleanup()
            return url

        return asyncio.run(_serve())


class _UnknownKeyError(Exception):
    pass


class _GoneKeyError(Exception):
    pass


def _parse_basic_auth(header: str) -> tuple[str | None, str | None]:
    if not header.startswith("Basic "):
        return None, None
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None, None
    if ":" not in decoded:
        return None, None
    username, _, password = decoded.partition(":")
    return username, password


def _ascii_fallback(name: str) -> str:
    cleaned = name.encode("ascii", "replace").decode("ascii").replace('"', "'")
    return cleaned or "artifact"


def _content_disposition(filename: str) -> str:
    quoted = urllib.parse.quote(filename, safe="")
    return f"attachment; filename=\"{_ascii_fallback(filename)}\"; filename*=UTF-8''{quoted}"


def _format_host(host: str) -> str:
    if ":" in host:  # IPv6 literal needs brackets in URLs
        return f"[{host}]"
    return host


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="artifact-share-server",
        description="Serve one staged tldw_chatbook artifact share.",
    )
    parser.add_argument("manifest", type=Path, help="Path to the share manifest.json")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="Bind port (0 = ephemeral)")
    args = parser.parse_args(argv)
    server = ArtifactShareServer(args.manifest)
    server.run(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
