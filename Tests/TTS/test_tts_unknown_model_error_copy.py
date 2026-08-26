"""TASK-15422: an unroutable TTS model is a configuration problem, not luck.

``UnknownLegacyModelError`` subclasses ``LookupError``, so it fell through
``_tts_outcome_code``/``_tts_error_copy`` to the generic bucket: metric
outcome ``generation_failed`` and the toast "Unexpected TTS generation
failure; retry". Retrying can never fix an id the compatibility bridge does
not route; a user who hit this (a custom model name before TASK-15420, an
unmapped ElevenLabs profile model still today) got no hint that the model
selection was what failed.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler
from tldw_chatbook.TTS.legacy_bridge import UnknownLegacyModelError

pytestmark = pytest.mark.unit


def test_unknown_legacy_model_error_is_a_model_configuration_outcome() -> None:
    """The metric outcome names the model, not a generic generation failure.

    Returns:
        None.
    """
    error = UnknownLegacyModelError("The selected TTS model is not available")

    assert TTSEventHandler._tts_outcome_code(error) == "model_invalid"


def test_unknown_legacy_model_error_copy_points_at_the_model_selection() -> None:
    """The toast names the model selection and where to fix it.

    Returns:
        None.
    """
    error = UnknownLegacyModelError("The selected TTS model is not available")

    assert TTSEventHandler._tts_error_copy(error) == (
        "The selected TTS model is not available for this provider; "
        "check the model in STTS Settings"
    )
