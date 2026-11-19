"""Settings session-summary pure-helper tests (issue #365)."""

import pytest

from tldw_chatbook.UI.Screens.settings_screen import _session_summary_section_values

pytestmark = pytest.mark.bootstrap_profile


def test_valid_values_pass_through():
    assert _session_summary_section_values(True, "3") == {
        "session_summary": {"enabled": True, "duration_seconds": 3}
    }


def test_duration_clamped():
    assert _session_summary_section_values(True, "0")["session_summary"]["duration_seconds"] == 1
    assert _session_summary_section_values(True, "99")["session_summary"]["duration_seconds"] == 30


def test_invalid_duration_falls_back_to_default():
    result = _session_summary_section_values(False, "abc")["session_summary"]
    assert result["duration_seconds"] == 3
    assert result["enabled"] is False


def test_empty_duration_falls_back():
    assert _session_summary_section_values(True, "")["session_summary"]["duration_seconds"] == 3


def test_duration_float_rounds_instead_of_truncating():
    assert (
        _session_summary_section_values(True, "2.9")["session_summary"][
            "duration_seconds"
        ]
        == 3
    )
