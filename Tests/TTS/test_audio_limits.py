"""The retained-audio limit accepts its boundary and exposes safe recovery data."""

from uuid import UUID

import pytest

from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.TTS.audio_limits import (
    MAX_BUFFERED_AUDIO_BYTES,
    check_buffered_audio_size,
)


@pytest.mark.parametrize(
    "size", [0, MAX_BUFFERED_AUDIO_BYTES - 1, MAX_BUFFERED_AUDIO_BYTES]
)
def test_retained_audio_accepts_sizes_up_to_and_including_the_limit(size):
    check_buffered_audio_size(size)


@pytest.mark.parametrize(
    "size", [MAX_BUFFERED_AUDIO_BYTES + 1, MAX_BUFFERED_AUDIO_BYTES * 2]
)
def test_oversized_audio_exposes_the_complete_nonretryable_error_contract(size):
    with pytest.raises(TTSOperationError) as raised:
        check_buffered_audio_size(size)

    error = raised.value
    assert error.code == "request_invalid"
    assert str(error) == (
        "Generated audio is too long to buffer. Shorten the text and try again."
    )
    assert error.retryable is False
    assert error.recovery_action == "shorten_text"
    assert len(error.operation_id) == 32
    assert UUID(error.operation_id).version == 4
