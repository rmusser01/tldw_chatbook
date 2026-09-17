"""Exact profile caller and native evidence lifetime regressions."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


async def _sample_close_child(root, after):
    import os
    import time
    import types
    import wave

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_service import (
        _FakeTTSService,
        _selection,
        _successful_artifact,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import sample_audio_validation as validation
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_service import TTSProfileService

    sample = root / "sample.wav"
    with wave.open(str(sample), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x01\x00" * 32)
    payload = sample.read_bytes()
    repository = TTSProfileRepository(root / "profiles.sqlite")
    await repository.open()
    hold = next(iter(storage._holds.values()))
    service = TTSProfileService(repository, _FakeTTSService())
    artifact = _successful_artifact(_selection(), sample)
    native_close = os.close
    validation.os = types.SimpleNamespace(**vars(os))
    selected_inode = sample.stat().st_ino
    attempts = []

    def close(descriptor):
        if os.fstat(descriptor).st_ino == selected_inode:
            attempts.append(descriptor)
            if after:
                native_close(descriptor)
            raise OSError("injected selected sample close")
        native_close(descriptor)

    validation.os.close = close
    loaded = await service.create_from_artifact("Native sample", artifact)
    assert loaded.profile.display_name == "Native sample"
    assert loaded.profile.profile_id in service._sample_evidence
    assert len(attempts) == 1
    assert sample.read_bytes() == payload
    await repository.close()
    pause = storage._begin_local_pause()
    drained = pause.drain(time.monotonic() + 0.05)
    observed = _probe(hold.authority.control_root, hold.names)
    assert observed == "blocked", (after, drained, observed)
    assert not drained


@pytest.mark.parametrize("after", [False, True])
def test_sample_evidence_result_does_not_retire_uncertain_native_close(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_service_maintenance import _sample_close_child
asyncio.run(_sample_close_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


def test_concurrent_ordinary_evidence_keeps_all_valid_samples(tmp_path):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.TTS.test_sample_audio_validation import _complete_wav
from Tests.TTS.test_profile_service import test_sample_evidence_cache_concurrent_admission_retains_every_bounded_id
sample = Path(sys.argv[1]) / 'sample.wav'
sample.write_bytes(_complete_wav())
test_sample_evidence_cache_concurrent_admission_retains_every_bounded_id(sample)
""",
    )


async def _app_source_child(root, mode):
    import os
    import types
    from Tests.UI.app_factory import _build_test_app
    from Tests.TTS.test_sample_audio_validation import _complete_wav
    from Tests.TTS.test_profile_service import _selection, _successful_artifact
    from tldw_chatbook.TTS import sample_audio_validation as validation
    from tldw_chatbook.TTS.profile_types import TTSProfileDraft

    app = _build_test_app()
    service = await app._ensure_tts_profile_service()
    repository = service._repository
    sample = root / "sample.wav"
    sample.write_bytes(_complete_wav())
    selection = _selection(
        configuration_revision=service._current_configuration_revision("audio_cpp")
    )
    created = await repository.create_profile(
        TTSProfileDraft(
            display_name="Source",
            provider_id=selection.provider_id,
            model_id=selection.model_id,
            voice_id=selection.voice_id,
            response_format=selection.response_format,
            speed=selection.speed,
            options={},
        )
    )
    loaded = await service.get_profile(created.value.profile_id)
    original_open = os.open
    validation.os = types.SimpleNamespace(**vars(os))
    reads = []

    def opened(path, *args, **kwargs):
        if path == sample:
            reads.append(path)
        return original_open(path, *args, **kwargs)

    validation.os.open = opened
    if mode == "app_service":
        app._tts_profile_service = object()
    elif mode == "dependency":
        app.tts_service = object()
    elif mode == "coordinator":
        app._audio_cpp_artifact_lease_coordinator = object()
    elif mode == "repository":
        app._tts_profile_repository = object()
    elif mode == "selector":
        from tldw_chatbook import config

        config._CONFIG_CACHE_SOURCE = root / "foreign.toml"
    elif mode == "profile":
        from tldw_chatbook import config

        config._CONFIG_CACHE["general"]["users_name"] = "changed-profile"
    elif mode == "method":
        service._check_source = lambda: None
    elif mode == "call_shadow":
        from contextlib import nullcontext

        service._service_call = lambda *args: nullcontext()
    elif mode == "post_method":
        original_revision = service._tts_service.configuration_revision

        def revision(provider):
            result = original_revision(provider)
            service._check_source = lambda: None
            return result

        service._tts_service.configuration_revision = revision
    service.record_sample_evidence(loaded, _successful_artifact(selection, sample))
    if mode == "post_method":
        assert reads == [sample] and not service._sample_evidence
        assert not service._native_operations
    elif mode == "normal":
        assert reads == [sample] and service._sample_evidence
        import time

        assert await service._maintenance_drain(time.monotonic() + 0.2)
        before = tuple(reads)
        service.record_sample_evidence(loaded, _successful_artifact(selection, sample))
        assert tuple(reads) == before
        await service._maintenance_resume()
    else:
        assert not reads, "drifted original app service reached selected native sample"
        assert not service._sample_evidence
    await repository.close()


@pytest.mark.parametrize(
    "mode",
    [
        "app_service",
        "dependency",
        "coordinator",
        "repository",
        "selector",
        "profile",
        "method",
        "call_shadow",
        "normal",
        "post_method",
    ],
)
def test_original_app_profile_service_drift_refuses_before_sample_io(tmp_path, mode):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_service_maintenance import _app_source_child
asyncio.run(_app_source_child(Path(sys.argv[1]), sys.argv[2]))
""",
        mode,
    )


async def _sample_edge_child(root, mode):
    import os
    import time
    import types
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_sample_audio_validation import _complete_wav
    from Tests.TTS.test_profile_service import (
        _FakeTTSService,
        _selection,
        _successful_artifact,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import sample_audio_validation as validation
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_service import TTSProfileService

    sample = root / "sample.wav"
    payload = _complete_wav()
    sample.write_bytes(payload)
    repository = TTSProfileRepository(root / "profiles.sqlite")
    await repository.open()
    service = TTSProfileService(repository, _FakeTTSService())
    hold = next(iter(storage._holds.values()))
    validation.os = types.SimpleNamespace(**vars(os))
    reached = []
    open_sample = validation._open_sample_descriptor
    native_open = os.open
    leaked = []

    if mode == "late_open":

        def opened(*args, **kwargs):
            fd = open_sample(*args, **kwargs)
            reached.append(fd)
            raise OSError("after observed native open")

        validation._open_sample_descriptor = opened
    elif mode == "unreturned":

        def opened(path, flags):
            fd = native_open(path, flags)
            assert os.fstat(fd).st_ino == sample.stat().st_ino
            leaked.append(fd)
            reached.append(fd)
            raise OSError("unreturned native open")

        validation.os.open = opened
    elif mode == "lease_after":
        original_finish = validation._SampleNativeOperation.finish

        def finish(native):
            if native.leases:
                last = native.leases[-1]
                original_close = last.close

                def close():
                    reached.append(True)
                    original_close()
                    raise OSError("after actual lease release")

                last.close = close
            return original_finish(native)

        validation._SampleNativeOperation.finish = finish
    elif mode == "paused":
        pause = storage._begin_local_pause()

        def path_validation(*args, **kwargs):
            raise AssertionError("new path callback reached during ordinary pause")

        validation.validate_path = path_validation
    loaded = (
        await service.create_from_artifact(
            "Native sample", _successful_artifact(_selection(), sample)
        )
        if mode != "paused"
        else None
    )
    if mode == "paused":
        from Tests.TTS.test_profile_service import _profile
        from tldw_chatbook.TTS.profile_service import LoadedTTSProfile

        service.record_sample_evidence(
            LoadedTTSProfile(
                repository.generation,
                _profile(model_id="selected-model", voice_id="selected-voice"),
            ),
            _successful_artifact(_selection(), sample),
        )
        pause.resume()
    await repository.close()
    if mode not in {"normal", "paused"}:
        assert len(reached) == 1
    assert sample.read_bytes() == payload
    pause = storage._begin_local_pause()
    uncertain = mode in {"unreturned", "lease_after"}
    assert pause.drain(time.monotonic() + 0.03) is not uncertain
    assert _probe(hold.authority.control_root, hold.names) == (
        "blocked" if uncertain else "entered"
    )
    assert await service._maintenance_drain(time.monotonic() + 0.03) is not uncertain
    if mode in {"normal", "lease_after"}:
        assert loaded.profile.profile_id in service._sample_evidence
    else:
        assert not service._sample_evidence


@pytest.mark.parametrize(
    "mode", ["normal", "late_open", "unreturned", "lease_after", "paused"]
)
def test_exact_sample_operation_allocation_release_and_ordinary_refusal(tmp_path, mode):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_service_maintenance import _sample_edge_child
asyncio.run(_sample_edge_child(Path(sys.argv[1]), sys.argv[2]))
""",
        mode,
    )


@pytest.mark.asyncio
async def test_original_nested_snapshot_finishes_after_service_gate_closes(tmp_path):
    import time
    from Tests.TTS.test_profile_repository_lifecycle import _draft
    from Tests.TTS.test_profile_service import _FakeTTSService
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_service import TTSProfileService
    from tldw_chatbook.TTS.profile_errors import ProfileServiceError

    repository = TTSProfileRepository(tmp_path / "profiles.sqlite")
    await repository.open()
    await repository.create_profile(_draft("Snapshot"))
    service = TTSProfileService(repository, _FakeTTSService())
    original = repository.list_profiles

    async def page(*args, **kwargs):
        result = await original(*args, **kwargs)
        service._maintenance_close_admission()
        return result

    repository.list_profiles = page
    try:
        result = await service.bounded_profile_assignment_snapshot()
        assert result[0][0].display_name == "Snapshot" and result[0][1] == 0
        assert await service._maintenance_drain(time.monotonic() + 0.1)
        with pytest.raises(ProfileServiceError):
            await service.list_profiles()
        await service._maintenance_resume()
        assert (await service.list_profiles()).total == 1
    finally:
        await repository.close()


async def _codec_child(root, after):
    import io
    import time
    import types
    import av
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_service import (
        _FakeTTSService,
        _selection,
        _successful_artifact,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import sample_audio_validation as validation
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_service import TTSProfileService

    data = io.BytesIO()
    with av.open(data, "w", format="flac") as output:
        stream = output.add_stream("flac", rate=16000)
        frame = av.AudioFrame(format="s16", layout="mono", samples=32)
        frame.sample_rate = 16000
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        for packet in (*stream.encode(frame), *stream.encode(None)):
            output.mux(packet)
    sample = root / "sample.flac"
    sample.write_bytes(data.getvalue())
    repository = TTSProfileRepository(root / "profiles.sqlite")
    await repository.open()
    hold = next(iter(storage._holds.values()))
    dependency = _FakeTTSService()
    service = TTSProfileService(repository, dependency)
    selection = _selection(
        provider_id="openai", model_id="tts-1", voice_id="alloy", response_format="flac"
    )
    artifact = _successful_artifact(selection, sample)
    from dataclasses import replace

    artifact = replace(artifact, content_type="audio/flac")
    native_open = av.open
    opened, closed, frames = [], [], []

    class Container:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def demux(self, stream):
            for packet in self.native.demux(stream):

                class Packet:
                    def decode(self):
                        result = packet.decode()
                        frames.extend(result)
                        return result

                yield Packet()

        def close(self):
            closed.append(self.native)
            if after not in {False, "unreturned"}:
                self.native.close()
            if type(after) is bool:
                raise OSError("injected exact codec close")

    def open_container(*args, **kwargs):
        native = native_open(*args, **kwargs)
        opened.append(native)
        if after == "unreturned":
            raise OSError("unreturned actual container")
        return Container(native)

    validation.get_safe_import = lambda *_: (
        None if after == "unavailable" else types.SimpleNamespace(open=open_container)
    )
    if after == "invalid":
        sample.write_bytes(b"invalid compressed sample")
    if after == "late":
        original = validation._open_sample_container

        def late(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("late observed codec allocation")

        validation._open_sample_container = late
    loaded = await service.create_from_artifact("Codec", artifact)
    assert loaded.profile.display_name == "Codec"
    if after not in {"unavailable", "invalid"}:
        assert len(opened) == 1
    if type(after) is bool or after == "normal":
        assert closed == opened and frames
        assert 0 < frames[0].samples <= 16000
    elif after in {"late", "invalid"}:
        assert closed == opened and not frames
    else:
        assert not closed and not frames
    await repository.close()
    pause = storage._begin_local_pause()
    ready = pause.drain(time.monotonic() + 0.03)
    observed = _probe(hold.authority.control_root, hold.names)
    uncertain = type(after) is bool or after == "unreturned"
    assert observed == ("blocked" if uncertain else "entered"), (after, ready, observed)
    assert ready is not uncertain
    assert bool(service._sample_evidence) is (after == "normal")


@pytest.mark.parametrize("after", [False, True])
def test_codec_return_does_not_erase_native_container_cleanup_failure(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_service_maintenance import _codec_child
asyncio.run(_codec_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


@pytest.mark.parametrize(
    "mode", ["normal", "unavailable", "invalid", "late", "unreturned"]
)
def test_codec_positive_and_unavailable_or_unknown_native_outcomes(tmp_path, mode):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_service_maintenance import _codec_child
asyncio.run(_codec_child(Path(sys.argv[1]), sys.argv[2]))
""",
        mode,
    )


@pytest.mark.asyncio
async def test_cancelled_thread_awaiter_does_not_retire_post_read_evidence(tmp_path):
    import asyncio
    import threading
    import time
    from Tests.TTS.test_sample_audio_validation import _complete_wav
    from Tests.TTS.test_profile_service import (
        _FakeTTSService,
        _selection,
        _successful_artifact,
    )
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_service import TTSProfileService
    from tldw_chatbook.TTS.profile_types import TTSProfileDraft

    sample = tmp_path / "sample.wav"
    sample.write_bytes(_complete_wav())
    repository = TTSProfileRepository(tmp_path / "profiles.sqlite")
    await repository.open()
    selection = _selection()
    created = await repository.create_profile(
        TTSProfileDraft(
            display_name="Thread sample",
            provider_id=selection.provider_id,
            model_id=selection.model_id,
            voice_id=selection.voice_id,
            response_format="wav",
            speed=1.0,
            options={},
        )
    )
    dependency = _FakeTTSService()
    service = TTSProfileService(repository, dependency)
    loaded = await service.get_profile(created.value.profile_id)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = dependency.configuration_revision

    def revision(provider):
        entered.set()
        assert release.wait(4)
        return original(provider)

    dependency.configuration_revision = revision

    def record():
        try:
            service.record_sample_evidence(
                loaded, _successful_artifact(selection, sample)
            )
        finally:
            finished.set()

    task = asyncio.create_task(asyncio.to_thread(record))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not finished.is_set()
        assert not await service._maintenance_drain(time.monotonic() + 0.03)
        assert not service._sample_evidence
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        assert await service._maintenance_drain(time.monotonic() + 0.2)
        assert loaded.profile.profile_id in service._sample_evidence
        assert not service._native_operations
    finally:
        release.set()
        await repository.close()


@pytest.mark.asyncio
async def test_cancelled_original_create_retains_post_commit_repository_result(
    tmp_path,
):
    import asyncio
    import time
    from Tests.TTS.test_sample_audio_validation import _complete_wav
    from Tests.TTS.test_profile_service import (
        _FakeTTSService,
        _selection,
        _successful_artifact,
    )
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_service import TTSProfileService

    sample = tmp_path / "sample.wav"
    sample.write_bytes(_complete_wav())
    repository = TTSProfileRepository(tmp_path / "profiles.sqlite")
    await repository.open()
    service = TTSProfileService(repository, _FakeTTSService())
    committed, release = asyncio.Event(), asyncio.Event()
    original = repository.create_profile
    rows = []

    async def create(draft):
        result = await original(draft)
        rows.append(result)
        committed.set()
        await release.wait()
        return result

    repository.create_profile = create
    action = asyncio.create_task(
        service.create_from_artifact(
            "Committed", _successful_artifact(_selection(), sample)
        )
    )
    try:
        await asyncio.wait_for(committed.wait(), 2)
        action.cancel()
        await asyncio.sleep(0.01)
        action.cancel()
        assert not action.done()
        assert not await service._maintenance_drain(time.monotonic() + 0.02)
        assert (
            await repository.get_profile(rows[0].value.profile_id)
        ).value.display_name == "Committed"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await action
        assert await service._maintenance_drain(time.monotonic() + 0.2)
        assert not service._sample_evidence
        await service._maintenance_resume()
        assert (
            await service.get_profile(rows[0].value.profile_id)
        ).profile.display_name == "Committed"
    finally:
        release.set()
        await repository.close()
