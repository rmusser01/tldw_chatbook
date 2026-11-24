"""Real Windows creation refusal, bounded handle cleanup, and a no-audio version probe."""

import os
import ctypes

import pytest

from tldw_chatbook.TTS import _controlled_wav_windows as native
from tldw_chatbook.TTS.controlled_wav_profile import (
    ControlledWavError,
    _pin_executable,
)

pytestmark = pytest.mark.skipif(os.name != "nt", reason="requires CreateProcessW")


def observe_native_creation(monkeypatch):
    calls = []
    kernel = native._kernel()

    class Boundary:
        def __getattr__(self, name):
            return getattr(kernel, name)

        def CreateProcessW(self, *args):
            created = kernel.CreateProcessW(*args)
            calls.append((bool(created), ctypes.get_last_error()))
            return created

    monkeypatch.setattr(native, "_kernel", lambda: Boundary())
    return calls


def test_creation_refusal_checks_every_setup_descriptor_before_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "refused.exe"
    path.write_bytes(b"not an executable")
    calls = observe_native_creation(monkeypatch)
    closed = []

    def checked_close(fd):
        os.close(fd)
        closed.append(fd)

    monkeypatch.setattr(native, "_close_fd", checked_close)
    with pytest.raises(native.SpawnRefused):
        native.spawn_player([str(path)], env=dict(os.environ))
    assert len(calls) == 1 and calls[0][0] is False and calls[0][1]
    assert len(set(closed)) == 3  # DEVNULL, stderr writer and reader
    for fd in closed:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_revocation_after_setup_skips_the_native_create(monkeypatch, tmp_path):
    calls = observe_native_creation(monkeypatch)
    closes = []

    def close(fd):
        os.close(fd)
        closes.append(fd)

    monkeypatch.setattr(native, "_close_fd", close)
    with pytest.raises(native.SpawnRefused):
        native.spawn_player(
            [str(tmp_path / "refused.exe")],
            env=dict(os.environ),
            start_allowed=lambda: False,
        )
    assert not calls and len(set(closes)) == 3


def test_refused_create_with_unconfirmed_setup_close_stays_uncertain(
    monkeypatch, tmp_path
):
    calls = observe_native_creation(monkeypatch)
    closes = []

    def refuse_one_close(fd):
        os.close(fd)
        closes.append(fd)
        if len(closes) == 1:
            raise OSError("unconfirmed setup close")

    monkeypatch.setattr(native, "_close_fd", refuse_one_close)
    with pytest.raises(native.UncertainBirth) as result:
        native.spawn_player([str(tmp_path / "missing.exe")], env=dict(os.environ))
    assert len(calls) == 1 and calls[0][0] is False
    assert not result.value.child.has_process
    assert len(set(closes)) == 3


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_qualified_version_uses_device_volume_name_and_checks_postbirth_cleanup(
    monkeypatch, cleanup_fails
):
    try:
        exe, pin = _pin_executable(None)
    except ControlledWavError as error:
        pytest.skip(str(error))
    child = None
    closes = []
    if cleanup_fails:

        def fail_first_close(fd):
            os.close(fd)
            closes.append(fd)
            if len(closes) == 1:
                raise OSError("simulated ambiguous postbirth close")

        monkeypatch.setattr(native, "_close_fd", fail_first_close)
    env = dict(os.environ)
    env.pop("FFREPORT", None)
    try:
        if cleanup_fails:
            with pytest.raises(native.UncertainBirth) as blocked:
                native.spawn_player([str(exe), "-version"], env=env)
            child = blocked.value.child
            assert child.has_process
        else:
            child = native.spawn_player([str(exe), "-version"], env=env)
        assert child.wait(timeout=10) == 0
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
            child.stderr.close()
            child.close_process_handle()
        assert pin.close()
    if cleanup_fails:
        assert len(set(closes)) == len(closes)
