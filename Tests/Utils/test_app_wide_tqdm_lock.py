"""tqdm's write lock is buildable for every in-app download (TASK-33621.113).

TASK-33621.21 found that the first Hugging Face download inside the running
app built tqdm's process-wide write lock (even for a disabled bar), whose
``multiprocessing.RLock`` starts CPython's resource tracker with
``sys.stderr.fileno()``. Under Textual that is ``-1``, so the spawn raised
``ValueError: bad value(s) in fds_to_keep`` and left tqdm's class lock held.
Only Console dictation guarded against it, so a download started first by any
other feature (embeddings, TTS, transcription, the Models screen) still died
that way and wedged every later download in the process.

Each case runs a fresh interpreter: the test process has long since built
tqdm's lock and started the resource tracker, which would hide the defect.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("tqdm")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: A first launch with a download from a feature other than dictation: the
#: Hub call every such feature makes (sentence-transformers, transformers'
#: `from_pretrained`, faster-whisper, Kokoro all reach `hf_hub_download`),
#: inside a running Textual app, against a loopback stand-in for the Hub.
#: Spawn is forced so Linux (fork default) reproduces macOS's first launch.
_IN_APP_DOWNLOAD = textwrap.dedent(
    """
    import asyncio, http.server, json, multiprocessing, os, sys, threading, time

    multiprocessing.set_start_method("spawn", force=True)
    sys.path.insert(0, sys.argv[1])
    import tldw_chatbook  # noqa: F401 -- the package import under test

    result = {"tqdm_before_download": "tqdm" in sys.modules}
    BODY = b'{"model": "tiny"}'

    class Hub(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def _reply(self, body):
            self.send_response(200)
            self.send_header("X-Repo-Commit", "c" * 40)
            self.send_header("ETag", '"config-17"')
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            if body:
                self.wfile.write(BODY)

        def do_HEAD(self):
            self._reply(False)

        def do_GET(self):
            self._reply(True)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ.update(
        HF_ENDPOINT=f"http://127.0.0.1:{server.server_address[1]}",
        HF_HOME=sys.argv[2],
        HF_HUB_CACHE=os.path.join(sys.argv[2], "hub"),
        HF_HUB_OFFLINE="0",
        HF_HUB_DISABLE_XET="1",
    )
    from textual.app import App

    async def main():
        async with App().run_test():
            result["stderr_fileno"] = sys.stderr.fileno()

            def download():
                from huggingface_hub import hf_hub_download

                try:
                    path = hf_hub_download("tldw-test/feature-model", "config.json")
                    with open(path, "rb") as handle:
                        result["content_ok"] = handle.read() == BODY
                except BaseException as exc:  # reported, not raised
                    result["error"] = f"{type(exc).__name__}: {exc}"

            worker = threading.Thread(target=download, daemon=True)
            worker.start()
            deadline = time.monotonic() + 30
            while worker.is_alive() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            result["download_returned"] = not worker.is_alive()

    asyncio.run(main())
    from tqdm import std

    # From the main thread, which never touched the lock: a new worker thread
    # can reuse the dead download thread's id and re-enter a wedged RLock.
    lock = std.TqdmDefaultWriteLock.th_lock
    result["lock_free"] = lock.acquire(timeout=2)
    if result["lock_free"]:
        lock.release()
    result["dictation_guard_loaded"] = (
        "tldw_chatbook.Utils.model_download_guard" in sys.modules
    )
    print("RESULT " + json.dumps(result))
    """
)


def _run(script: str, *args: str, tmp_path: Path) -> dict:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("PYTHON", "HF_", "TQDM_"))
    }
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(PROJECT_ROOT), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = [
        line for line in completed.stdout.splitlines() if line.startswith("RESULT ")
    ]
    assert lines, completed.stdout[-2000:] + completed.stderr[-2000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows never starts the resource tracker for a lock",
)
def test_a_first_download_by_another_feature_completes_inside_the_app(tmp_path):
    """AC#1: no dictation guard ran, and the download still completes."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("textual")

    result = _run(_IN_APP_DOWNLOAD, str(tmp_path / "hf"), tmp_path=tmp_path)

    # A genuine first launch, inside Textual's stderr capture.
    assert result["tqdm_before_download"] is False
    assert result["stderr_fileno"] == -1
    assert result["dictation_guard_loaded"] is False
    assert result["download_returned"], "the download was still blocked after 30 s"
    assert "error" not in result, result["error"]
    assert result["content_ok"] is True
    assert result["lock_free"] is True, "the download left tqdm's class lock held"


_ARMING_PROBE = textwrap.dedent(
    """
    import json, sys

    sys.path.insert(0, sys.argv[1])
    if sys.argv[2] == "tqdm-first":
        import tqdm  # noqa: F401 -- another library imported tqdm before the app
    import tldw_chatbook  # noqa: F401

    result = {"tqdm_after_package": "tqdm" in sys.modules}
    before = {name for name in sys.modules if name.startswith("tldw_chatbook")}
    from tqdm import std

    result["mp_lock"] = repr(std.TqdmDefaultWriteLock.__dict__.get("mp_lock", "<unset>"))
    result["app_modules_added_by_tqdm"] = sorted(
        name
        for name in sys.modules
        if name.startswith("tldw_chatbook") and name not in before
    )
    print("RESULT " + json.dumps(result))
    """
)


@pytest.mark.parametrize("order", ["package-first", "tqdm-first"])
def test_the_package_decides_tqdms_lock_without_importing_tqdm(tmp_path, order):
    """AC#2: tqdm stays unimported at package import, and its lock is decided.

    Whichever imports first, tqdm ends up with its own thread-only fallback
    (``mp_lock = None``) before any bar can build a multiprocessing lock, and
    arming it imports no app module (the UI-ready census stays unchanged).
    """
    result = _run(_ARMING_PROBE, order, tmp_path=tmp_path)

    assert result["tqdm_after_package"] is (order == "tqdm-first")
    assert result["mp_lock"] == "None"
    assert result["app_modules_added_by_tqdm"] == []


_DOUBLE_ARMING_PROBE = textwrap.dedent(
    """
    import importlib, json, sys

    sys.path.insert(0, sys.argv[1])
    import tldw_chatbook

    # The package init runs again before tqdm loads (a reload, or a second
    # copy of the package): each run defines its own finder class.
    importlib.reload(tldw_chatbook)
    result = {
        "armers": sum(
            1 for finder in sys.meta_path if type(finder).__name__ == "_TqdmLockArmingFinder"
        )
    }
    try:
        from tqdm import std
    except RecursionError as exc:
        result["error"] = f"RecursionError: {exc}"
    else:
        result["mp_lock"] = repr(
            std.TqdmDefaultWriteLock.__dict__.get("mp_lock", "<unset>")
        )
    print("RESULT " + json.dumps(result))
    """
)


def test_running_the_package_init_twice_still_imports_tqdm(tmp_path):
    """Two executions of the init must not leave finders delegating to each other.

    Each execution defined a new finder class, so the isinstance guard missed
    the first finder; the two then asked each other for ``tqdm.std`` and every
    later ``import tqdm`` raised RecursionError.
    """
    result = _run(_DOUBLE_ARMING_PROBE, tmp_path=tmp_path)

    assert "error" not in result, result["error"][:200]
    assert result["armers"] == 1
    assert result["mp_lock"] == "None"
