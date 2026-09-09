"""Bound retained audio when a backend must assemble one complete container."""

from uuid import uuid4

from tldw_chatbook.TTS.adapter_types import TTSOperationError

MAX_BUFFERED_AUDIO_BYTES = 64 * 1024 * 1024


def check_buffered_audio_size(size: int) -> None:
    """Reject an oversized audio buffer before retaining the next chunk.

    Args:
        size: Combined byte size including the proposed next chunk.

    Raises:
        TTSOperationError: When complete-file audio exceeds the retained byte limit.
    """
    if size > MAX_BUFFERED_AUDIO_BYTES:
        raise TTSOperationError(
            code="request_invalid",
            message="Generated audio is too long to buffer. Shorten the text and try again.",
            operation_id=uuid4().hex,
            retryable=False,
            recovery_action="shorten_text",
        )
