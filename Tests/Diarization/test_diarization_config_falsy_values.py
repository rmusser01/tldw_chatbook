"""A `[diarization]` setting the user sets to zero is honoured.

Tier-2 review S11, P2 [D1]: `_default_config_loader` did

    config[key] = get_cli_setting(f"diarization.{key}", default_value) or default_value

`get_cli_setting` already honours its own default when the key is absent,
so the trailing `or` could only ever fire on a **successfully resolved
falsy value** -- silently replacing it with the default.

`config.py`'s shipped `[diarization]` block documents seven keys as
tunable and legitimately zero (`vad_threshold`,
`vad_min_speech_duration`, `vad_min_silence_duration`,
`segment_overlap`, `min_segment_duration`, `merge_threshold`,
`min_speaker_duration`), each with a comment inviting the user to tune
it. Setting `segment_overlap = 0` got 0.5 s of overlap, with no warning
anywhere.

`False` is the sharper case: `memory_efficient` and
`detect_overlapping_speech` are booleans, so with a default of `True` a
user could never turn one off at all.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Local_Ingestion import diarization_service as ds


@pytest.fixture
def zeroed(monkeypatch: pytest.MonkeyPatch):
    """Every diarization setting resolves to a falsy value."""

    def fake(key: str, default=None):
        assert key.startswith("diarization."), key
        return 0 if isinstance(default, (int, float)) and not isinstance(
            default, bool
        ) else False

    monkeypatch.setattr(ds, "get_cli_setting", fake)


def test_a_zero_the_user_set_is_not_replaced_by_the_default(zeroed):
    config = ds.DiarizationService.__new__(
        ds.DiarizationService
    )._default_config_loader()

    for key in (
        "vad_threshold",
        "vad_min_speech_duration",
        "vad_min_silence_duration",
        "segment_overlap",
        "min_segment_duration",
        "merge_threshold",
        "min_speaker_duration",
    ):
        assert config[key] == 0, f"{key} was silently replaced by its default"


def test_a_false_the_user_set_is_not_replaced_by_the_default(zeroed):
    config = ds.DiarizationService.__new__(
        ds.DiarizationService
    )._default_config_loader()

    assert config["memory_efficient"] is False
    assert config["detect_overlapping_speech"] is False


def test_an_absent_setting_still_falls_back_to_the_shipped_default(monkeypatch):
    """Anti-vacuity: `get_cli_setting`'s own default still applies."""
    monkeypatch.setattr(ds, "get_cli_setting", lambda key, default=None: default)

    config = ds.DiarizationService.__new__(
        ds.DiarizationService
    )._default_config_loader()

    assert config["segment_overlap"] == 0.5
    assert config["merge_threshold"] == 0.5
