"""Typed Settings-to-Model-Library handoff contracts for audio.cpp."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from tldw_chatbook.UI.Navigation.audio_cpp_model_handoff import (
    AudioCppModelLibraryRequest,
    AudioCppModelLibraryResult,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    HandoffValueError,
    PendingHandoffStore,
)


def _request() -> AudioCppModelLibraryRequest:
    return AudioCppModelLibraryRequest(token="request-token-1", draft_revision=7)


def _result(root: Path) -> AudioCppModelLibraryResult:
    return AudioCppModelLibraryResult(
        token="request-token-1",
        draft_revision=7,
        artifact_id="audio-cpp-supertonic-3",
        revision="a" * 40,
        variant="f16",
        canonical_root=str(root),
    )


def test_handoff_values_are_frozen_slotted_and_root_redacted(tmp_path: Path) -> None:
    request = _request()
    result = _result(tmp_path.resolve())

    assert [item.name for item in fields(request)] == ["token", "draft_revision"]
    assert [item.name for item in fields(result)] == [
        "token",
        "draft_revision",
        "artifact_id",
        "revision",
        "variant",
        "canonical_root",
    ]
    assert not hasattr(request, "__dict__")
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        request.draft_revision = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.variant = "q8"  # type: ignore[misc]
    assert result.canonical_root not in repr(result)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (AudioCppModelLibraryRequest, {"token": "", "draft_revision": 1}),
        (AudioCppModelLibraryRequest, {"token": " request ", "draft_revision": 1}),
        (AudioCppModelLibraryRequest, {"token": "request", "draft_revision": True}),
        (AudioCppModelLibraryRequest, {"token": "request", "draft_revision": -1}),
        (
            AudioCppModelLibraryResult,
            {
                "token": "request",
                "draft_revision": 1,
                "artifact_id": "../private",
                "revision": "a" * 40,
                "variant": "f16",
                "canonical_root": "/managed/root",
            },
        ),
        (
            AudioCppModelLibraryResult,
            {
                "token": "request",
                "draft_revision": 1,
                "artifact_id": "audio-cpp-model",
                "revision": "a" * 40,
                "variant": "f16",
                "canonical_root": "relative/root",
            },
        ),
        (
            AudioCppModelLibraryResult,
            {
                "token": "request",
                "draft_revision": 1,
                "artifact_id": "audio-cpp-model",
                "revision": "a" * 40,
                "variant": "f16",
                "canonical_root": "/managed/../private",
            },
        ),
        (
            AudioCppModelLibraryResult,
            {
                "token": "request",
                "draft_revision": 1,
                "artifact_id": "audio-cpp-model",
                "revision": "a" * 40,
                "variant": "f16",
                "canonical_root": "/",
            },
        ),
        (
            AudioCppModelLibraryResult,
            {
                "token": "request",
                "draft_revision": 1,
                "artifact_id": "audio-cpp-model",
                "revision": "a" * 40,
                "variant": "f16",
                "canonical_root": "C:/managed/model",
            },
        ),
    ],
)
def test_handoff_values_reject_noncanonical_scalars(factory, kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory(**kwargs)


def test_audio_cpp_handoff_channels_are_explicit_and_independent(
    tmp_path: Path,
) -> None:
    assert HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST.value == (
        "audio_cpp_model_library_request"
    )
    assert HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT.value == (
        "audio_cpp_model_library_result"
    )
    store = PendingHandoffStore()

    assert store.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, _request()) == 1
    assert (
        store.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            _result(tmp_path.resolve()),
        )
        == 1
    )

    request_claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST)
    result_claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)
    assert request_claim is not None
    assert result_claim is not None
    assert request_claim.value == _request()
    assert result_claim.value == _result(tmp_path.resolve())


@pytest.mark.parametrize(
    ("channel", "value_factory"),
    [
        (HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, _request),
        (
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            lambda: _result(Path("/managed/root")),
        ),
    ],
)
def test_audio_cpp_claim_is_one_time_and_release_replays_exact_value(
    channel: HandoffChannel,
    value_factory,
) -> None:
    store = PendingHandoffStore()
    original = value_factory()
    revision = store.stage(channel, original)
    claim = store.claim(channel)

    assert claim is not None
    assert claim.revision == revision
    assert claim.value == original
    assert claim.value is not original
    assert store.claim(channel) is None
    assert store.release(claim) is True
    assert store.release(claim) is False

    replay = store.claim(channel)
    assert replay is not None
    assert replay.revision == revision
    assert replay.value == original
    assert replay.value is not claim.value
    assert store.acknowledge(claim) is False
    assert store.acknowledge(replay) is True
    assert store.acknowledge(replay) is False
    assert store.claim(channel) is None


def test_audio_cpp_store_rejects_hostile_subclasses_partial_and_wrong_values(
    tmp_path: Path,
) -> None:
    class HostileRequest(AudioCppModelLibraryRequest):
        pass

    class HostileResult(AudioCppModelLibraryResult):
        pass

    partial = object.__new__(AudioCppModelLibraryRequest)
    values = (
        (
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST,
            HostileRequest(token="request", draft_revision=1),
        ),
        (HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, partial),
        (HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, {"token": "request"}),
        (
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            HostileResult(
                token="request",
                draft_revision=1,
                artifact_id="audio-cpp-model",
                revision="a" * 40,
                variant="f16",
                canonical_root=str(tmp_path.resolve()),
            ),
        ),
        (HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT, _request()),
    )

    for channel, value in values:
        store = PendingHandoffStore()
        with pytest.raises(HandoffValueError):
            store.stage(channel, value)
        assert store.claim(channel) is None


def test_audio_cpp_detached_copy_reconstructs_every_scalar(tmp_path: Path) -> None:
    source = _result(tmp_path.resolve())
    store = PendingHandoffStore()
    store.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT, source)
    object.__setattr__(source, "artifact_id", "producer-mutated")
    object.__setattr__(source, "canonical_root", "/private/producer-mutated")

    claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)

    assert claim is not None
    assert claim.value.artifact_id == "audio-cpp-supertonic-3"
    assert claim.value.canonical_root == str(tmp_path.resolve())
    object.__setattr__(claim.value, "variant", "consumer-mutated")
    assert store.release(claim) is True
    replay = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)
    assert replay is not None
    assert replay.value.variant == "f16"
