"""Route, auth, and containment tests for the artifact share server."""

import json
import subprocess
import sys
import threading
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tldw_chatbook.Web_Server.artifact_share_manifest import (
    build_share_auth,
    stage_share,
)
from tldw_chatbook.Web_Server.artifact_share_server import ArtifactShareServer

pytestmark = [pytest.mark.unit, pytest.mark.loopback_network]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _stage(tmp_path: Path, *, auth: bool, names: tuple[str, ...] = ("Alpha", "Beta")):
    records = []
    for index, name in enumerate(names, start=1):
        bundle = tmp_path / f"src-{index}.zip"
        bundle.write_bytes(f"bytes-of-{name}".encode())
        records.append(
            {
                "id": str(index),
                "chatbook_id": index,
                "name": name,
                "description": f"desc {name} <script>alert(1)</script>",
                "file_path": str(bundle),
            }
        )
    auth_model = build_share_auth("alice", "secret-pass") if auth else None
    manifest = stage_share(
        records, share_name="Test Library 'quotes'", auth=auth_model, share_root=tmp_path / "share"
    )
    return tmp_path / "share" / manifest.share_id / "manifest.json", manifest


@contextmanager
def _served(manifest_path: Path):
    """Run the share app on an ephemeral loopback port in a background thread."""
    import asyncio

    from aiohttp import web

    server = ArtifactShareServer(manifest_path)
    app = server.build_app()
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    holder: dict[str, object] = {}

    def _run() -> None:
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app, access_log=None)

        async def _start() -> None:
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            holder["url"] = f"http://{runner.addresses[0][0]}:{runner.addresses[0][1]}"
            ready.set()
            await holder["stop_event"].wait()
            await runner.cleanup()

        holder["stop_event"] = asyncio.Event()
        loop.run_until_complete(_start())
        loop.close()

    thread = threading.Thread(target=_run, name="test-share-server", daemon=True)
    thread.start()
    assert ready.wait(timeout=10), "share server did not start"
    yield holder["url"]
    loop.call_soon_threadsafe(holder["stop_event"].set)
    thread.join(timeout=10)


def _client(auth: tuple[str, str] | None = None) -> httpx.Client:
    return httpx.Client(
        base_url="ignored",
        timeout=10,
        auth=auth,
        follow_redirects=False,
    )


def test_index_page_lists_artifacts_escaped_no_auth(tmp_path):
    manifest_path, _manifest = _stage(tmp_path, auth=False)
    with _served(manifest_path) as url, _client() as client:
        response = client.get(f"{url}/")
        assert response.status_code == 200
        assert "text/html" in response.headers["Content-Type"]
        body = response.text
        assert "Alpha" in body and "Beta" in body
        assert "<script>" not in body  # descriptions are escaped
        assert "&lt;script&gt;" in body
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert "default-src" in response.headers["Content-Security-Policy"]


def test_index_json_and_artifact_download_match_staged_bytes(tmp_path):
    manifest_path, manifest = _stage(tmp_path, auth=False)
    share_dir = manifest_path.parent
    with _served(manifest_path) as url, _client() as client:
        listing = client.get(f"{url}/index.json")
        assert listing.status_code == 200
        payload = listing.json()
        assert payload["share_name"] == "Test Library 'quotes'"
        assert {entry["name"] for entry in payload["artifacts"]} == {
            item.display_name for item in manifest.artifacts
        }
        first = manifest.artifacts[0]
        entry = next(e for e in payload["artifacts"] if e["name"] == first.display_name)
        assert entry["sha256"] == first.sha256

        download = client.get(f"{url}/artifact/{first.key}")
        assert download.status_code == 200
        assert download.content == (share_dir / first.staged_name).read_bytes()
        assert download.headers["Content-Disposition"].startswith("attachment;")
        assert "application" in download.headers["Content-Type"]
        head = client.head(f"{url}/artifact/{first.key}")
        assert head.status_code == 200

        bundle = client.get(f"{url}/bundle.zip")
        assert bundle.status_code == 200
        with zipfile.ZipFile(__import__("io").BytesIO(bundle.content)) as opened:
            assert sorted(opened.namelist()) == sorted(
                item.staged_name for item in manifest.artifacts
            )


def test_unknown_key_is_404(tmp_path):
    manifest_path, _manifest = _stage(tmp_path, auth=False)
    with _served(manifest_path) as url, _client() as client:
        assert client.get(f"{url}/artifact/does-not-exist").status_code == 404


def test_non_ascii_unknown_key_is_404_with_security_headers(tmp_path):
    # %E2%82%AC is a euro sign: compare_digest on non-ASCII str raises
    # TypeError, which previously surfaced as a 500 that bypassed the
    # security-headers middleware. It must 404 like any unknown key and
    # still carry the middleware's header posture.
    manifest_path, _manifest = _stage(tmp_path, auth=False)
    with _served(manifest_path) as url, _client() as client:
        response = client.get(f"{url}/artifact/%E2%82%AC")
        assert response.status_code == 404
        assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_traversal_staged_name_is_never_served(tmp_path, monkeypatch):
    manifest_path, manifest = _stage(tmp_path, auth=False)
    # Tamper with the manifest the way a corrupted file would look: a
    # staged_name that escapes the staging directory must not resolve.
    data = json.loads(manifest_path.read_text())
    data["artifacts"][0]["staged_name"] = "../escape.zip"
    manifest_path.write_text(json.dumps(data))
    (tmp_path / "escape.zip").write_bytes(b"escaped")
    with _served(manifest_path) as url, _client() as client:
        key = json.loads(manifest_path.read_text())["artifacts"][0]["key"]
        assert client.get(f"{url}/artifact/{key}").status_code == 404


def test_auth_challenges_and_admits(tmp_path):
    manifest_path, _manifest = _stage(tmp_path, auth=True)
    with _served(manifest_path) as url:
        with _client() as client:
            denied = client.get(f"{url}/")
            assert denied.status_code == 401
            assert "WWW-Authenticate" in denied.headers
            assert client.get(f"{url}/index.json").status_code == 401
        with _client(auth=("alice", "secret-pass")) as client:
            assert client.get(f"{url}/").status_code == 200
        with _client(auth=("alice", "wrong")) as client:
            assert client.get(f"{url}/").status_code == 401


def test_auth_lockout_after_ten_failures(tmp_path):
    manifest_path, _manifest = _stage(tmp_path, auth=True)
    with _served(manifest_path) as url, _client(auth=("alice", "wrong")) as client:
        for _ in range(10):
            assert client.get(f"{url}/").status_code == 401
        assert client.get(f"{url}/").status_code == 429
        with _client(auth=("alice", "secret-pass")) as good:
            assert good.get(f"{url}/").status_code == 429  # same IP stays locked out


def test_content_disposition_strips_crlf(tmp_path):
    # Qodo #15: CR/LF (or any control byte) in a display name must never
    # reach the header, for per-artifact downloads and the bundle alike.
    payload = tmp_path / "crlf.zip"
    payload.write_bytes(b"crlf-bytes")
    records = [
        {
            "id": "1",
            "chatbook_id": 1,
            "name": "evil\r\nX-Injected: 1",
            "description": "",
            "file_path": str(payload),
        }
    ]
    manifest = stage_share(
        records,
        share_name="share\r\nSet-Cookie: pwn=1",
        auth=None,
        share_root=tmp_path / "share",
    )
    manifest_path = tmp_path / "share" / manifest.share_id / "manifest.json"
    with _served(manifest_path) as url, _client() as client:
        artifact = client.get(f"{url}/artifact/{manifest.artifacts[0].key}")
        assert artifact.status_code == 200
        disposition = artifact.headers["Content-Disposition"]
        assert "\r" not in disposition
        assert "\n" not in disposition

        bundle = client.get(f"{url}/bundle.zip")
        assert bundle.status_code == 200
        bundle_disposition = bundle.headers["Content-Disposition"]
        assert "\r" not in bundle_disposition
        assert "\n" not in bundle_disposition


def test_bundle_symlink_escape_is_gone_never_external_bytes(tmp_path):
    # Qodo #11: a bundle.zip swapped for a symlink to an outside file is
    # refused (410 Gone), and the external file's bytes are never served.
    manifest_path, _manifest = _stage(tmp_path, auth=False)
    share_dir = manifest_path.parent
    external = tmp_path / "external-secret.zip"
    external.write_bytes(b"external-secret-bytes")
    (share_dir / "bundle.zip").unlink()
    (share_dir / "bundle.zip").symlink_to(external)
    with _served(manifest_path) as url, _client() as client:
        response = client.get(f"{url}/bundle.zip")
        assert response.status_code == 410
        assert response.content != b"external-secret-bytes"


def test_build_app_gates_aiohttp_through_optional_deps(tmp_path, monkeypatch):
    # Qodo #3: build_app must fail with the standard guidance (via
    # require_dependency) when the [web] extra is missing, before any
    # lazy route-handler import could surface a bare ModuleNotFoundError.
    from tldw_chatbook.Web_Server import artifact_share_server as server_module

    manifest_path, _manifest = _stage(tmp_path, auth=False)
    server = ArtifactShareServer(manifest_path)

    def _missing(*_args, **_kwargs):
        raise ImportError(
            "Required dependency 'aiohttp' for feature 'web' is not available."
        )

    monkeypatch.setattr(
        "tldw_chatbook.Utils.optional_deps.require_dependency", _missing
    )
    with pytest.raises(ImportError, match=r"tldw_chatbook\[web\]|aiohttp"):
        server.build_app()

    def _pass_through(module_name: str, feature_name: str | None = None):
        assert (module_name, feature_name) == ("aiohttp", "web")
        import aiohttp

        return aiohttp

    monkeypatch.setattr(
        "tldw_chatbook.Utils.optional_deps.require_dependency", _pass_through
    )
    assert server.build_app() is not None


def test_verified_cache_stores_digests_not_plaintext(tmp_path):
    # Qodo #16: the replay cache holds SHA-256 digests of the credential
    # pair, never the plaintext pair; membership compares digests.
    from tldw_chatbook.Web_Server.artifact_share_server import _verified_digest

    manifest_path, _manifest = _stage(tmp_path, auth=True)
    server = ArtifactShareServer(manifest_path)
    digest = _verified_digest("alice", "secret-pass")
    server._verified.add(digest)

    assert len(digest) == 64  # sha256 hex
    assert server._verified == {digest}
    assert all(
        "alice" not in entry and "secret" not in entry for entry in server._verified
    )
    # a second pair yields a distinct digest (no cross-pair collisions here)
    assert _verified_digest("alice", "wrong") != digest


def test_escape_fragment_guarantees_no_raw_markup():
    # Qodo #6 (no-sanitizer branch): html.escape stays the defense; the
    # fragment helper guarantees no raw '<'/'>' survives into the page.
    from tldw_chatbook.Web_Server.artifact_share_server import _escape_fragment

    hostile = '<script>alert(1)</script>"onmouseover="x'
    fragment = _escape_fragment(hostile)
    assert "<" not in fragment
    assert ">" not in fragment
    assert "&lt;script&gt;" in fragment


@pytest.mark.integration
def test_subprocess_ready_line_and_download(tmp_path):
    manifest_path, _manifest = _stage(tmp_path, auth=False)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tldw_chatbook.Web_Server.artifact_share_server",
            str(manifest_path),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Read stdout lines until the ready marker (bounded wait).
        url = None
        for _ in range(100):
            line = process.stdout.readline()
            if not line:
                break
            if line.startswith("ARTIFACT_SHARE_READY "):
                url = line.split(" ", 1)[1].strip()
                break
        assert url, "child never reported ARTIFACT_SHARE_READY"
        status = json.loads((manifest_path.parent / "status.json").read_text())
        assert status["url"] == url
        assert status["pid"] == process.pid
        with _client() as client:
            assert client.get(f"{url}/").status_code == 200
    finally:
        process.terminate()
        process.wait(timeout=10)
