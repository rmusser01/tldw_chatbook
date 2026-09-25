"""Real-model integration — runs only when OMNIVOICE_ONNX_ROOT points at the artifact."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OMNIVOICE_ONNX_ROOT"),
    reason="set OMNIVOICE_ONNX_ROOT to the installed omnivoice-onnx-int8hq tree",
)


@pytest.mark.asyncio
async def test_short_synthesis() -> None:
    from tldw_chatbook.TTS.backends.omnivoice import OmniVoiceOnnxTTSBackend

    backend = OmniVoiceOnnxTTSBackend(
        {
            "OMNIVOICE_MODEL_ROOT": os.environ["OMNIVOICE_ONNX_ROOT"],
            "OMNIVOICE_NUM_STEPS": 8,
        }
    )
    chunks = [chunk async for chunk in backend.generate_speech_stream(text="Testing omnivoice synthesis.")]
    assert len(chunks) == 1
    assert len(chunks[0]) > 44  # more than a bare WAV header
    await backend.close()


@pytest.mark.asyncio
async def test_clone_from_reference() -> None:
    reference = os.environ.get("OMNIVOICE_TEST_REFERENCE_WAV")
    if not reference:
        pytest.skip("set OMNIVOICE_TEST_REFERENCE_WAV to a ~5 s reference clip")
    from tldw_chatbook.TTS.backends.omnivoice import OmniVoiceOnnxTTSBackend

    backend = OmniVoiceOnnxTTSBackend(
        {
            "OMNIVOICE_MODEL_ROOT": os.environ["OMNIVOICE_ONNX_ROOT"],
            "OMNIVOICE_NUM_STEPS": 8,
        }
    )
    transcript = os.environ.get("OMNIVOICE_TEST_REFERENCE_TEXT", "the reference transcript")
    chunks = [
        chunk
        async for chunk in backend.generate_speech_stream(
            text="Cloned voice test.",
            reference_audio=reference,
            reference_text=transcript,
        )
    ]
    assert len(chunks) == 1
    assert len(chunks[0]) > 44
    await backend.close()
