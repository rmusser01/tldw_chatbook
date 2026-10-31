"""Shared state for tests that download a model inside a running Textual app.

TASK-33621.21. Two pieces of process-global state decide whether the first
in-app download dies on ``fds_to_keep``: whether tqdm has built its write lock
yet, and whether CPython's resource tracker is already running. A test process
usually has both by the time a test runs, which lets a regression pass, so
`first_download_in_process` puts both back to a first launch and restores
them afterwards. `model_host` is a loopback stand-in for the Hugging Face hub.
"""

from __future__ import annotations

import http.server
import multiprocessing.resource_tracker as resource_tracker
import threading
from pathlib import Path

import pytest

tqdm_std = pytest.importorskip("tqdm.std")

_MISSING = object()


class _ModelHost(http.server.BaseHTTPRequestHandler):
    """Serves `/<repo>/resolve/main/<file>` the way the Hugging Face hub does."""

    files: dict[str, bytes] = {}
    requests: list[str] = []

    def log_message(self, *_args) -> None:
        return

    def _reply(self, include_body: bool) -> None:
        type(self).requests.append(f"{self.command} {self.path}")
        name = self.path.rsplit("/", 1)[-1]
        body = type(self).files.get(name)
        if body is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("X-Repo-Commit", "c" * 40)
        self.send_header("ETag", f'"{name}-{len(body)}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802 - http.server naming
        self._reply(include_body=False)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        self._reply(include_body=True)


@pytest.fixture
def model_host(tmp_path: Path):
    """A loopback model host, with huggingface_hub pointed at it and a fresh cache.

    Restored inside the fixture (not by the shared ``monkeypatch``): the
    suite's offline-latch fixture checks ``HF_HUB_OFFLINE`` at its own
    teardown, which runs before ``monkeypatch`` unwinds.
    """
    import huggingface_hub.constants as constants
    import huggingface_hub.utils._http as hub_http

    _ModelHost.files = {}
    _ModelHost.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ModelHost)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(constants, "HF_HUB_OFFLINE", False)
        patch.setattr(constants, "ENDPOINT", endpoint)
        patch.setattr(
            constants,
            "HUGGINGFACE_CO_URL_TEMPLATE",
            endpoint + "/{repo_id}/resolve/{revision}/{filename}",
        )
        patch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
        # A client another test closed would refuse every request.
        patch.setattr(hub_http, "_GLOBAL_CLIENT", None)
        try:
            yield _ModelHost
        finally:
            client = hub_http._GLOBAL_CLIENT
            if client is not None:
                client.close()
    server.shutdown()
    server.server_close()


@pytest.fixture
def first_download_in_process():
    """tqdm's lock unbuilt and no resource tracker running, as on a first launch.

    Restores tqdm's class state afterwards, so a wedged lock left by a failing
    run cannot reach another test.
    """
    lock_type = tqdm_std.TqdmDefaultWriteLock
    saved_class_lock = lock_type.th_lock
    saved_mp = lock_type.__dict__.get("mp_lock", _MISSING)
    saved_locks = {
        klass: klass.__dict__["_lock"]
        for klass in _tqdm_classes()
        if "_lock" in klass.__dict__
    }
    fresh_tracker = resource_tracker.ResourceTracker()
    with pytest.MonkeyPatch.context() as patch:
        for name in ("ensure_running", "register", "unregister", "getfd"):
            patch.setattr(resource_tracker, name, getattr(fresh_tracker, name))
        patch.setattr(resource_tracker, "_resource_tracker", fresh_tracker)
        lock_type.th_lock = tqdm_std.TRLock()
        if saved_mp is not _MISSING:
            del lock_type.mp_lock
        for klass in saved_locks:
            del klass._lock
        try:
            yield
        finally:
            lock_type.th_lock = saved_class_lock
            if saved_mp is _MISSING:
                if "mp_lock" in lock_type.__dict__:
                    del lock_type.mp_lock
            else:
                lock_type.mp_lock = saved_mp
            for klass in _tqdm_classes():
                if "_lock" in klass.__dict__ and klass not in saved_locks:
                    del klass._lock
            for klass, lock in saved_locks.items():
                klass._lock = lock
            stop = getattr(fresh_tracker, "_stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass


def _tqdm_classes() -> list[type]:
    seen: list[type] = []
    pending = [tqdm_std.tqdm]
    while pending:
        klass = pending.pop()
        if klass not in seen:
            seen.append(klass)
            pending.extend(klass.__subclasses__())
    return seen
