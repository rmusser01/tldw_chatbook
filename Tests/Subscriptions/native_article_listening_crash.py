"""Explicitly selected Windows native crash evidence. Uses the guarded test profile."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.name != "nt", reason="qualified Windows playback")
@pytest.mark.parametrize("interruption", ["playing", "suspended", "startup-stop"])
def test_killed_app_restarts_manually_on_exact_retained_wav(tmp_path, interruption):
    import _winapi
    from tldw_chatbook.TTS._controlled_wav_windows import _creation_time

    assert Path(os.environ["TLDW_CONFIG_PATH"]).resolve().is_relative_to(tmp_path)
    assert Path(os.environ["HOME"]).resolve().is_relative_to(tmp_path)
    repo = Path(__file__).resolve().parents[2]

    def start(mode):
        return subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                "tools.evidence.watchlist_article_audio.session_crash_child",
                str(tmp_path),
                mode,
            ],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    def receipt(name, process):
        deadline = time.monotonic() + 60
        path = tmp_path / (name + ".json")
        while not path.exists() and time.monotonic() < deadline:
            failure = tmp_path / "failure.json"
            if failure.exists():
                pytest.fail(f"Private child refused {name}: {failure.read_text()}")
            assert process.poll() is None, f"child exited before {name}"
            time.sleep(0.01)
        assert path.exists(), f"timed out observing {name}"
        return json.loads(path.read_bytes())

    seed = start("seed")
    try:
        receipt("seed", seed)
        assert seed.wait(30) == 0
    finally:
        if seed.poll() is None:
            seed.kill()
            seed.wait(10)
    child = start(interruption)
    child_handle = resume = None
    try:
        initial = receipt(interruption + "-restore", child)
        born = receipt(interruption + "-birth", child)
        child_handle = _winapi.OpenProcess(0x00100000 | 0x1000, False, born["pid"])
        assert _creation_time(child_handle) == born["created"]
        assert _winapi.WaitForSingleObject(child_handle, 0) == _winapi.WAIT_TIMEOUT
        if interruption == "startup-stop":
            (tmp_path / "stop-now").touch()
            stopped = receipt("stopped", child)
            assert stopped == {"phase": "paused", "frame": 0}
            assert _winapi.WaitForSingleObject(child_handle, 0) == _winapi.WAIT_OBJECT_0
            assert child.wait(30) == 0
            print(
                json.dumps(
                    dict(
                        role=interruption,
                        frame=0,
                        predecessor_signaled=True,
                        automatic_return=False,
                    )
                )
            )
            return
        if interruption == "playing":
            observed = receipt("checkpoint", child)
            expected = observed["saved"]
            assert observed["observed"] > expected > 0
        else:
            expected = 0
        child.kill()
        child.wait(10)
        # Start the new app promptly, then observe the exact original handle.
        resume = start("resume")
        assert _winapi.WaitForSingleObject(child_handle, 5000) == _winapi.WAIT_OBJECT_0
        death_observed = time.perf_counter()
        restored = receipt("resume-restore", resume)
        assert restored == {**initial, "frame": expected}
        grant = receipt("resume-grant", resume)
        assert grant["clock"] >= death_observed + 0.250
        next_birth = receipt("resume-birth", resume)
        assert next_birth["clock"] >= death_observed + 0.250
        restarted = receipt("resumed", resume)
        assert restarted["observed"] > expected
        assert resume.wait(30) == 0
        print(
            json.dumps(
                dict(
                    role=interruption,
                    durable_frame=expected,
                    restarted_at=restored["frame"],
                    resumed_observation=restarted["observed"],
                    predecessor_signaled=True,
                    seconds_until_grant=round(grant["clock"] - death_observed, 3),
                    seconds_until_birth=round(next_birth["clock"] - death_observed, 3),
                )
            )
        )
    finally:
        for process in (child, resume):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(10)
        if child_handle is not None:
            _winapi.CloseHandle(child_handle)
