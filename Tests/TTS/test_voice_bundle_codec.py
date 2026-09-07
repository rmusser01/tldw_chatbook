from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
from io import BytesIO
from uuid import UUID
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import pytest
import tldw_chatbook.TTS as tts

from tldw_chatbook.TTS.profile_portability import PortableTTSProfile
from tldw_chatbook.TTS.profile_reference_audio import validate_canonical_reference_wav
from tldw_chatbook.TTS.profile_reference_types import (
    REFERENCE_SAMPLE_ENCODING,
    CanonicalTTSCloneReference,
    TTSCloneRecipeRequirement,
)
from tldw_chatbook.TTS.profile_types import TTSProfileDraft
from tldw_chatbook.TTS.voice_bundle_codec import (
    EXPECTED_MEMBER_ORDER,
    TTSCloneVoiceBundle,
    TTSVoiceBundleError,
    TTSVoiceBundleSinks,
    encode_clone_voice_bundle,
    inspect_clone_voice_bundle,
)
import tldw_chatbook.TTS.voice_bundle_codec as codec


def canonical_wav() -> bytes:
    frames = b"\x00\x00" * 160
    fmt = struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16)
    body = b"WAVEfmt " + struct.pack("<I", 16) + fmt
    body += b"data" + struct.pack("<I", len(frames)) + frames
    return b"RIFF" + struct.pack("<I", len(body)) + body


def canonical_bundle() -> TTSCloneVoiceBundle:
    wav = canonical_wav()
    metadata = validate_canonical_reference_wav(wav)
    return TTSCloneVoiceBundle(
        profile=PortableTTSProfile(
            profile_id=UUID("01234567-89ab-4cde-8fab-0123456789ab"),
            draft=TTSProfileDraft(
                display_name="Narrator 界",
                provider_id="audio_cpp",
                model_id="model-界",
                voice_id=None,
                response_format="wav",
                speed=1.0,
                options={},
            ),
        ),
        reference=CanonicalTTSCloneReference(
            wav_bytes=wav,
            reference_text="Exact transcript.",
            sha256=sha256(wav).hexdigest(),
            byte_length=metadata.byte_length,
            duration_ms=metadata.duration_ms,
            sample_rate_hz=metadata.sample_rate_hz,
            channels=metadata.channels,
            sample_encoding=REFERENCE_SAMPLE_ENCODING,
        ),
        recipe_requirement=TTSCloneRecipeRequirement(
            recipe_id="pocket-tts",
            recipe_revision=7,
            model_id="model-界",
        ),
    )


def _codec_traceback_values(error: BaseException) -> list[object]:
    pending: list[object] = []
    current = error.__traceback__
    while current is not None:
        if current.tb_frame.f_globals.get("__name__") == codec.__name__:
            pending.extend(current.tb_frame.f_locals.values())
        current = current.tb_next
    values: list[object] = []
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        values.append(value)
        if type(value) is dict:
            pending.extend(value.keys())
            pending.extend(value.values())
        elif type(value) in (list, tuple, set, frozenset):
            pending.extend(value)
        elif isinstance(value, BaseException):
            pending.extend(value.args)
            pending.extend(
                item
                for item in (value.__cause__, value.__context__)
                if item is not None
            )
        elif not isinstance(value, type) and is_dataclass(value):
            pending.extend(getattr(value, field.name) for field in fields(value))
    return values


def test_writer_emits_exact_deterministic_four_member_bundle() -> None:
    source = canonical_bundle()

    first = encode_clone_voice_bundle(source)
    second = encode_clone_voice_bundle(source)

    assert first == second
    with ZipFile(BytesIO(first)) as archive:
        assert tuple(archive.namelist()) == EXPECTED_MEMBER_ORDER
        for member in archive.infolist():
            assert member.compress_type == ZIP_STORED
            assert member.flag_bits == 0
            assert member.extract_version == 20
            assert member.create_version == 20
            assert member.create_system == 3
            assert member.date_time == (1980, 1, 1, 0, 0, 0)
            assert member.external_attr == 0o100600 << 16
            assert member.internal_attr == 0
            assert member.extra == b""
            assert member.comment == b""
        assert archive.comment == b""


def test_writer_uses_exact_canonical_contents_and_round_trips() -> None:
    source = canonical_bundle()
    encoded = encode_clone_voice_bundle(source)

    with ZipFile(BytesIO(encoded)) as archive:
        manifest_bytes = archive.read("manifest.json")
        profile_bytes = archive.read("profile.json")
        transcript = archive.read("reference.txt")
        assert manifest_bytes.endswith(b"\n")
        assert profile_bytes.endswith(b"\n")
        assert not manifest_bytes.endswith(b"\n\n")
        assert not profile_bytes.endswith(b"\n\n")
        assert transcript == b"Exact transcript."
        for value in (manifest_bytes, profile_bytes):
            parsed = json.loads(value)
            assert value == (
                json.dumps(
                    parsed,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
        manifest = json.loads(manifest_bytes)
        assert manifest["entries"] == {
            name: {
                "sha256": sha256(archive.read(name)).hexdigest(),
                "size": len(archive.read(name)),
            }
            for name in ("profile.json", "reference.txt", "reference.wav")
        }

    assert inspect_clone_voice_bundle(encoded) == source


def test_bundle_values_and_errors_redact_private_contents() -> None:
    source = canonical_bundle()
    assert "Exact transcript" not in repr(source)
    assert source.reference.wav_bytes.hex() not in repr(source)

    with pytest.raises(TTSVoiceBundleError) as caught:
        inspect_clone_voice_bundle(b"PRIVATE malformed archive")

    error = caught.value
    assert error.code == "bundle_invalid"
    assert str(error) == "bundle_invalid"
    assert repr(error) == "TTSVoiceBundleError('bundle_invalid')"
    assert error.__context__ is None
    assert error.__cause__ is None
    assert "PRIVATE" not in repr(error)

    forged_code = TTSVoiceBundleError("PRIVATE code")  # type: ignore[arg-type]
    assert forged_code.code == "bundle_invalid"
    assert "PRIVATE" not in repr(forged_code)


def test_inspector_streams_exact_members_to_caller_owned_sinks() -> None:
    sinks = TTSVoiceBundleSinks(
        manifest_json=BytesIO(),
        profile_json=BytesIO(),
        reference_wav=BytesIO(),
        reference_txt=BytesIO(),
    )

    result = inspect_clone_voice_bundle(
        encode_clone_voice_bundle(canonical_bundle()),
        sinks=sinks,
    )

    assert result == canonical_bundle()
    with ZipFile(BytesIO(encode_clone_voice_bundle(canonical_bundle()))) as archive:
        assert sinks.manifest_json.getvalue() == archive.read("manifest.json")
        assert sinks.profile_json.getvalue() == archive.read("profile.json")
        assert sinks.reference_wav.getvalue() == archive.read("reference.wav")
        assert sinks.reference_txt.getvalue() == archive.read("reference.txt")
    assert "Exact transcript" not in repr(sinks)


def test_unsupported_bundle_version_has_bounded_distinct_code() -> None:
    encoded = encode_clone_voice_bundle(canonical_bundle())
    with ZipFile(BytesIO(encoded)) as archive:
        values = {name: archive.read(name) for name in EXPECTED_MEMBER_ORDER}
    manifest = json.loads(values["manifest.json"])
    manifest["schema_version"] = 2
    values["manifest.json"] = (
        json.dumps(
            manifest,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    target = BytesIO()
    with ZipFile(target, "w", compression=ZIP_STORED, allowZip64=False) as archive:
        for name in EXPECTED_MEMBER_ORDER:
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.external_attr = 0o100600 << 16
            archive.writestr(info, values[name])

    with pytest.raises(TTSVoiceBundleError) as caught:
        inspect_clone_voice_bundle(target.getvalue())

    assert caught.value.code == "unsupported_bundle"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_bundle_codec_is_available_from_the_tts_domain_boundary() -> None:
    assert tts.TTSCloneVoiceBundle is TTSCloneVoiceBundle
    assert tts.TTSVoiceBundleError is TTSVoiceBundleError
    assert tts.TTSVoiceBundleSinks is TTSVoiceBundleSinks
    assert tts.encode_clone_voice_bundle is encode_clone_voice_bundle
    assert tts.inspect_clone_voice_bundle is inspect_clone_voice_bundle


def test_public_failure_traceback_has_no_private_codec_frame_locals() -> None:
    private_archive = b"PK PRIVATE ARCHIVE CANARY"
    private_transcript = "PRIVATE TRANSCRIPT CANARY"
    private_path = "/private/staging/PRIVATE-PATH-CANARY"

    class PrivateSink(BytesIO):
        def __repr__(self) -> str:
            return private_path

    private_sink = PrivateSink(private_transcript.encode())
    sinks = TTSVoiceBundleSinks(
        manifest_json=private_sink,
        profile_json=PrivateSink(),
        reference_wav=PrivateSink(),
        reference_txt=PrivateSink(),
    )

    with pytest.raises(TTSVoiceBundleError) as caught:
        inspect_clone_voice_bundle(private_archive, sinks=sinks)

    traceback_locals: list[str] = []
    current = caught.value.__traceback__
    while current is not None:
        if current.tb_frame.f_globals.get("__name__") == (
            "tldw_chatbook.TTS.voice_bundle_codec"
        ):
            traceback_locals.append(repr(current.tb_frame.f_locals))
        current = current.tb_next
    rendered = "\n".join(traceback_locals)
    assert private_archive.hex() not in rendered
    assert repr(private_archive) not in rendered
    assert private_transcript not in rendered
    assert private_path not in rendered
    assert "PrivateSink" not in rendered


def test_sink_collaborator_failure_severs_private_traceback_locals_and_graph() -> None:
    private_transcript = "PRIVATE TRANSCRIPT IN VALID BUNDLE"
    private_path = "/private/staging/PRIVATE-SINK-PATH"
    source = canonical_bundle()
    source = replace(
        source,
        reference=replace(source.reference, reference_text=private_transcript),
    )
    private_archive = encode_clone_voice_bundle(source)

    class FailingSink(BytesIO):
        def __repr__(self) -> str:
            return private_path

        def seek(self, *_args, **_kwargs):
            raise RuntimeError(f"{private_path}: {private_transcript}")

    sinks = TTSVoiceBundleSinks(
        manifest_json=FailingSink(),
        profile_json=BytesIO(),
        reference_wav=BytesIO(),
        reference_txt=BytesIO(),
    )

    with pytest.raises(TTSVoiceBundleError) as caught:
        inspect_clone_voice_bundle(private_archive, sinks=sinks)

    codec_locals: list[str] = []
    current = caught.value.__traceback__
    while current is not None:
        if current.tb_frame.f_globals.get("__name__") == (
            "tldw_chatbook.TTS.voice_bundle_codec"
        ):
            codec_locals.append(repr(current.tb_frame.f_locals))
        current = current.tb_next
    rendered = "\n".join(codec_locals)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert repr(private_archive) not in rendered
    assert private_transcript not in rendered
    assert private_path not in rendered
    assert "FailingSink" not in rendered


@pytest.mark.parametrize(
    "raised_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_sink_control_flow_is_redelivered_fresh_without_private_locals(
    raised_type: type[BaseException],
) -> None:
    private_transcript = "PRIVATE CONTROL FLOW TRANSCRIPT"
    source = canonical_bundle()
    source = replace(
        source,
        reference=replace(source.reference, reference_text=private_transcript),
    )
    private_archive = encode_clone_voice_bundle(source)
    private_path = "/private/staging/PRIVATE-CONTROL-FLOW"

    class InterruptingSink(BytesIO):
        def seek(self, *_args, **_kwargs):
            raise raised_type(private_path)

        def __repr__(self) -> str:
            return private_path

    private_sink = InterruptingSink()
    sinks = TTSVoiceBundleSinks(
        manifest_json=private_sink,
        profile_json=BytesIO(),
        reference_wav=BytesIO(),
        reference_txt=BytesIO(),
    )

    with pytest.raises(raised_type) as caught:
        inspect_clone_voice_bundle(private_archive, sinks=sinks)

    rendered = ""
    current = caught.value.__traceback__
    while current is not None:
        if current.tb_frame.f_globals.get("__name__") == (
            "tldw_chatbook.TTS.voice_bundle_codec"
        ):
            rendered += repr(current.tb_frame.f_locals)
        current = current.tb_next
    assert caught.value.args == ()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert repr(private_archive) not in rendered
    assert private_transcript not in rendered
    assert private_path not in rendered
    assert "InterruptingSink" not in rendered
    values = _codec_traceback_values(caught.value)
    assert all(value is not private_archive for value in values)
    assert all(value is not private_sink for value in values)
    assert private_archive not in [value for value in values if type(value) is bytes]
    assert private_transcript not in [value for value in values if type(value) is str]
    assert private_path not in [value for value in values if type(value) is str]


def test_sink_non_control_base_exception_is_normalized_and_severed() -> None:
    private_archive = encode_clone_voice_bundle(canonical_bundle())
    private_path = "/private/staging/PRIVATE-GENERATOR-EXIT"

    class ExitingSink(BytesIO):
        def seek(self, *_args, **_kwargs):
            raise GeneratorExit(private_path)

    private_sink = ExitingSink()
    sinks = TTSVoiceBundleSinks(
        manifest_json=private_sink,
        profile_json=BytesIO(),
        reference_wav=BytesIO(),
        reference_txt=BytesIO(),
    )

    with pytest.raises(TTSVoiceBundleError) as caught:
        inspect_clone_voice_bundle(private_archive, sinks=sinks)

    values = _codec_traceback_values(caught.value)
    assert caught.value.code == "bundle_invalid"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert all(value is not private_archive for value in values)
    assert all(value is not private_sink for value in values)
    assert private_archive not in [value for value in values if type(value) is bytes]
    assert private_path not in [value for value in values if type(value) is str]


@pytest.mark.parametrize("alias_pair", [(0, 1), (1, 3), (2, 3)])
def test_duplicate_sink_identity_is_refused_before_any_sink_mutation(
    alias_pair: tuple[int, int],
) -> None:
    touched: list[str] = []

    class TrackingSink(BytesIO):
        def read(self, *_args, **_kwargs):
            touched.append("read")
            return super().read(*_args, **_kwargs)

        def seek(self, *_args, **_kwargs):
            touched.append("seek")
            return super().seek(*_args, **_kwargs)

        def truncate(self, *_args, **_kwargs):
            touched.append("truncate")
            return super().truncate(*_args, **_kwargs)

        def write(self, *_args, **_kwargs):
            touched.append("write")
            return super().write(*_args, **_kwargs)

    streams = [TrackingSink(f"original-{index}".encode()) for index in range(4)]
    streams[alias_pair[1]] = streams[alias_pair[0]]
    before = [stream.getvalue() for stream in streams]
    sinks = TTSVoiceBundleSinks(*streams)

    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(
            encode_clone_voice_bundle(canonical_bundle()), sinks=sinks
        )

    assert [stream.getvalue() for stream in streams] == before
    assert touched == []


def test_sink_capabilities_are_rejected_before_any_sink_mutation() -> None:
    class NotWritable(BytesIO):
        def writable(self) -> bool:
            return False

    streams = [BytesIO(f"original-{index}".encode()) for index in range(3)]
    invalid = NotWritable(b"original-invalid")
    before = [stream.getvalue() for stream in [invalid, *streams]]
    sinks = TTSVoiceBundleSinks(invalid, *streams)

    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(
            encode_clone_voice_bundle(canonical_bundle()), sinks=sinks
        )

    assert [stream.getvalue() for stream in [invalid, *streams]] == before


def test_encoder_collaborator_failure_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = canonical_bundle()
    transcript = source.reference.reference_text
    wav = source.reference.wav_bytes
    original = ZipFile.writestr

    def fail(self, info, data, *args, **kwargs):
        if getattr(info, "filename", "") == "profile.json":
            raise RuntimeError(f"{transcript}:{wav.hex()}")
        return original(self, info, data, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "writestr", fail)

    with pytest.raises(TTSVoiceBundleError) as caught:
        encode_clone_voice_bundle(source)

    rendered = ""
    current = caught.value.__traceback__
    while current is not None:
        if current.tb_frame.f_globals.get("__name__") == (
            "tldw_chatbook.TTS.voice_bundle_codec"
        ):
            rendered += repr(current.tb_frame.f_locals)
        current = current.tb_next
    assert caught.value.code == "bundle_invalid"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    values = _codec_traceback_values(caught.value)
    assert all(value is not source.profile.draft for value in values)
    assert transcript not in [value for value in values if type(value) is str]
    assert wav not in [value for value in values if type(value) is bytes]
    assert transcript not in rendered
    assert wav.hex() not in rendered


@pytest.mark.parametrize(
    "raised_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_encoder_control_flow_is_redelivered_fresh(
    monkeypatch: pytest.MonkeyPatch,
    raised_type: type[BaseException],
) -> None:
    source = canonical_bundle()
    transcript = source.reference.reference_text

    def interrupt(*_args, **_kwargs):
        raise raised_type(transcript)

    monkeypatch.setattr(ZipFile, "writestr", interrupt)

    with pytest.raises(raised_type) as caught:
        encode_clone_voice_bundle(source)

    assert caught.value.args == ()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    values = _codec_traceback_values(caught.value)
    assert all(value is not source for value in values)
    assert all(value is not source.profile.draft for value in values)
    assert transcript not in [value for value in values if type(value) is str]


@pytest.mark.parametrize("raised_type", [RuntimeError, GeneratorExit])
def test_encoder_failing_stream_is_normalized_without_private_graph(
    monkeypatch: pytest.MonkeyPatch,
    raised_type: type[BaseException],
) -> None:
    source = canonical_bundle()
    transcript = source.reference.reference_text

    class FailingStream(BytesIO):
        def write(self, _payload) -> int:
            raise raised_type(transcript)

    monkeypatch.setattr(codec, "BytesIO", FailingStream)

    with pytest.raises(TTSVoiceBundleError) as caught:
        encode_clone_voice_bundle(source)

    values = _codec_traceback_values(caught.value)
    assert caught.value.code == "bundle_invalid"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert all(value is not source for value in values)
    assert all(value is not source.profile.draft for value in values)
    assert transcript not in [value for value in values if type(value) is str]
