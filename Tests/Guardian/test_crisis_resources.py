# Tests/Guardian/test_crisis_resources.py
"""Crisis resource block extraction (P1 ruling): content, composition, markup safety."""
import pytest

from tldw_chatbook.Guardian.crisis_resources import (
    CRISIS_DISCLAIMER,
    CRISIS_RESOURCES_TEXT,
    render_crisis_block,
)


def test_resources_text_carries_all_four_resources():
    for token in (
        "988",
        "741741",
        "1-800-662-4357",
        "findahelpline.com",
    ):
        assert token in CRISIS_RESOURCES_TEXT, f"missing resource {token!r}"


def test_disclaimer_is_the_exact_sentence():
    assert CRISIS_DISCLAIMER == "tldw is not a mental health service."


def test_render_crisis_block_composes_resources_then_disclaimer():
    block = render_crisis_block()
    assert block == f"{CRISIS_RESOURCES_TEXT}\n\n{CRISIS_DISCLAIMER}"
    assert block.startswith("If you or someone you know is struggling")
    assert block.endswith(CRISIS_DISCLAIMER)


def test_rendered_block_is_markup_safe_plain_text():
    """No Textual square-bracket markup may hide in the crisis block."""
    block = render_crisis_block()
    assert "[" not in block and "]" not in block
    assert "═" not in block and "█" not in block


async def test_check_pipeline_uses_the_extracted_module(gate_on, tmp_path):
    """The checker's crisis notice path renders through crisis_resources.

    Byte-identical behavior after extraction: the surfaced notice text ends
    with exactly the extracted block (Task 1's inline tests keep passing;
    this pins the wiring to the new module).
    """
    from Tests.Guardian.test_check_pipeline import _checker, _fresh_db, _notes

    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "crisis awareness",
                "topic": "crisis_awareness",
                "pattern": r"suicid\w*",
                "action": "notify",
                "severity": "critical",
                "is_crisis": 1,
                "notification_frequency": "every_message",
                "display_mode": "inline_banner",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    result = await checker.check("I have been feeling suicidal")

    notice = result["notice"]
    assert notice is not None and notice["is_crisis"] is True
    assert notice["text"].endswith(render_crisis_block()), (
        "the crisis block must ride every surfaced crisis notice, verbatim"
    )
    assert notes["rows"] and notes["rows"][0].endswith(render_crisis_block())


@pytest.fixture()
def gate_on(monkeypatch):
    from tldw_chatbook.Guardian import settings as guardian_settings

    def fake_setting(key, default=None):
        if key == "enabled":
            return True
        return default

    monkeypatch.setattr(guardian_settings, "guardian_setting", fake_setting)
