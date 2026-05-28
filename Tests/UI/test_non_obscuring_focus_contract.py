import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VARIABLES = ROOT / "tldw_chatbook/css/core/_variables.tcss"
RESET = ROOT / "tldw_chatbook/css/core/_reset.tcss"
BUTTONS = ROOT / "tldw_chatbook/css/components/_buttons.tcss"
FORMS = ROOT / "tldw_chatbook/css/components/_forms.tcss"
AGENTIC = ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
NAV = ROOT / "tldw_chatbook/UI/Navigation/main_navigation.py"


def css_block(text: str, selector: str) -> str:
    """Return a CSS rule body whose selector list contains selector."""
    uncommented = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    for match in re.finditer(r"\{(?P<body>[^{}]*)\}", uncommented, flags=re.DOTALL):
        prefix = uncommented[: match.start()]
        selector_start = max(prefix.rfind("}"), prefix.rfind(";")) + 1
        selector_text = prefix[selector_start : match.start()]
        selectors = [item.strip() for item in selector_text.split(",")]
        if selector in selectors:
            return match.group("body")
    raise AssertionError(f"Missing CSS block for {selector}")


def assert_non_obscuring_focus(block: str) -> None:
    assert "outline: heavy" not in block
    assert "reverse" not in block
    assert "text-style: bold underline;" in block


def test_focus_tokens_are_defined_and_not_semantic_warning_or_error():
    text = VARIABLES.read_text(encoding="utf-8")
    for token in (
        "$ds-focus-fg",
        "$ds-focus-bg",
        "$ds-focus-accent",
        "$ds-input-focus-border",
        "$ds-input-focus-bg",
        "$ds-input-focus-accent",
    ):
        assert token in text
    assert "$ds-focus-accent: $ds-status-warning" not in text
    assert "$ds-focus-accent: $ds-status-error" not in text


def test_global_focus_fallback_is_visible_but_not_heavy():
    text = RESET.read_text(encoding="utf-8")
    block = css_block(text, "*:focus")
    assert "outline: heavy" not in block
    assert "outline: none" not in block
    assert any(cue in block for cue in ("outline: solid", "border:", "text-style:"))


def test_global_button_focus_uses_two_non_obscuring_cues():
    text = BUTTONS.read_text(encoding="utf-8")
    for selector in ("Button:focus", "Button:hover:focus"):
        block = css_block(text, selector)
        assert_non_obscuring_focus(block)
        assert "$ds-focus-bg" in block or "$ds-surface-raised" in block


def test_shared_form_and_native_inputs_use_thin_non_semantic_focus():
    text = FORMS.read_text(encoding="utf-8")
    for selector in (
        "Input:focus",
        "TextArea:focus",
        "Select:focus",
        ".form-input:focus",
        ".form-textarea:focus",
    ):
        block = css_block(text, selector)
        assert "outline: heavy" not in block
        assert "border: solid $ds-input-focus-border;" in block
        assert "border-bottom: solid $ds-input-focus-accent;" in block
        assert "$error" not in block
        assert "$warning" not in block


def test_console_and_library_visible_offenders_do_not_obscure_labels():
    text = AGENTIC.read_text(encoding="utf-8")
    for selector in (
        ".console-transcript-action-button:focus",
        ".library-source-action:focus",
    ):
        block = css_block(text, selector)
        assert_non_obscuring_focus(block)
        assert "$ds-status-warning" not in block
        assert "$ds-status-error" not in block


def test_console_composer_focus_uses_thin_input_treatment():
    text = AGENTIC.read_text(encoding="utf-8")
    block = css_block(text, "#console-native-composer.console-composer-focused")
    assert "border: heavy" not in block
    assert "border: solid $ds-input-focus-border;" in block
    assert "border-bottom: solid $ds-input-focus-accent;" in block


def test_top_navigation_inline_focus_uses_hybrid_contract():
    text = NAV.read_text(encoding="utf-8")
    focus = css_block(text, ".nav-button:focus")
    active = css_block(text, ".nav-button.is-active")
    active_focus = css_block(text, ".nav-button.is-active:focus")
    assert_non_obscuring_focus(focus)
    assert "outline: heavy" not in active
    assert_non_obscuring_focus(active_focus)
