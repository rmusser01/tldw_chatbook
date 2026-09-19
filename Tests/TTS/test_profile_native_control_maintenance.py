"""Exact primary and cleanup control identities on original private native work."""

import pytest
from Tests.TTS.test_profile_repository_maintenance import _run_private_child


async def _control_child(root, resource, phase, body_failure):
    import io
    import os
    import time
    import types
    from dataclasses import replace
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_sample_audio_validation import _complete_wav
    from Tests.TTS.test_profile_service import (
        _FakeTTSService,
        _selection,
        _successful_artifact,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import sample_audio_validation as validation
    from tldw_chatbook.TTS.profile_service import TTSProfileService
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.UI import stts_profile_library as ui

    class Control(BaseException):
        pass

    primary, cleanup = Control("original body"), Control("original cleanup")
    closed, body = [], []
    selected = root / ("selected.flac" if resource == "codec" else "selected.wav")
    payload = _complete_wav()
    if resource == "codec":
        import av

        data = io.BytesIO()
        with av.open(data, "w", format="flac") as output:
            stream = output.add_stream("flac", rate=16000)
            frame = av.AudioFrame(format="s16", layout="mono", samples=32)
            frame.sample_rate = 16000
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            for packet in (*stream.encode(frame), *stream.encode(None)):
                output.mux(packet)
        payload = data.getvalue()

        class Container:
            def __init__(self, native):
                self.native = native

            def __getattr__(self, name):
                return getattr(self.native, name)

            def demux(self, stream):
                for packet in self.native.demux(stream):

                    class Packet:
                        def decode(self):
                            result = packet.decode()
                            body.extend(result)
                            if body_failure:
                                raise primary
                            return result

                    yield Packet()

            def close(self):
                self.native.close()
                closed.append(self.native)
                if phase == "close":
                    raise cleanup

        def opened(*args, **kwargs):
            return Container(av.open(*args, **kwargs))

        validation.get_safe_import = lambda *args: types.SimpleNamespace(open=opened)
    selected.write_bytes(payload)
    repository = TTSProfileRepository(root / "profiles.sqlite")
    await repository.open()
    hold = next(iter(storage._holds.values()))
    service = TTSProfileService(repository, _FakeTTSService())
    if resource == "sample":
        validation.os = types.SimpleNamespace(**vars(os))

        def read(fd, count):
            result = os.read(fd, count)
            body.append(result)
            if body_failure:
                raise primary
            return result

        def close(fd):
            os.close(fd)
            closed.append(fd)
            if phase == "close":
                raise cleanup

        validation.os.read = read
        validation.os.close = close
    if resource == "export":

        class Stream:
            def __init__(self, native):
                self.native = native

            def write(self, value):
                result = self.native.write(value)
                body.append(result)
                if body_failure:
                    raise primary
                return result

            def close(self):
                self.native.close()
                closed.append(self.native)
                if phase == "close":
                    raise cleanup

        original = ui._open_profile_export_stream

        def opened(*args, **kwargs):
            stream = Stream(original(*args, **kwargs))
            kwargs["_outcome"].stream = stream
            return stream

        ui._open_profile_export_stream = opened
        library = ui.STTSProfileLibrary(lambda: None)
    if phase == "lease":
        original_close = storage.StorageLease.close

        def close(lease):
            original_close(lease)
            if type(getattr(lease, "native_owner", None)) in {
                validation._SampleNativeOperation,
                ui._ProfileExportOperation,
            }:
                closed.append(lease)
                raise cleanup

        storage.StorageLease.close = close
    try:
        if resource == "export":
            await library._run_profile_export(selected, "exact exported bytes")
        else:
            selection = (
                _selection(
                    provider_id="openai",
                    model_id="tts-1",
                    voice_id="alloy",
                    response_format="flac",
                )
                if resource == "codec"
                else _selection()
            )
            artifact = _successful_artifact(selection, selected)
            if resource == "codec":
                artifact = replace(artifact, content_type="audio/flac")
            await service.create_from_artifact("Committed before control", artifact)
    except Control as error:
        assert error is (primary if body_failure else cleanup), (
            "cleanup replaced original body control"
        )
    else:
        raise AssertionError("original control was swallowed")
    assert body and closed
    if resource != "export":
        assert (await repository.list_profiles()).value.total == 1
        assert selected.read_bytes() == payload
    else:
        assert selected.read_text() == "exact exported bytes"
    await repository.close()
    # Library import owns one known private startup; no runtime service was started.
    assert all(
        lease in storage._startups.values()
        or type(getattr(lease, "native_owner", None))
        in {validation._SampleNativeOperation, ui._ProfileExportOperation}
        for lease in storage._live_leases
    )
    storage._shutdown()
    pause = storage._begin_local_pause()
    assert not pause.drain(time.monotonic() + 0.02)
    assert _probe(hold.authority.control_root, hold.names) == "blocked"


@pytest.mark.parametrize("resource", ["sample", "codec", "export"])
@pytest.mark.parametrize("phase", ["close", "lease"])
@pytest.mark.parametrize("body_failure", [False, True])
def test_native_cleanup_preserves_exact_control_identity_and_primary_body(
    tmp_path, resource, phase, body_failure
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_native_control_maintenance import _control_child
asyncio.run(_control_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == 'True'))
""",
        resource,
        phase,
        str(body_failure),
    )
