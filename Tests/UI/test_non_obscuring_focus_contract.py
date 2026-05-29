import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
VARIABLES = ROOT / "tldw_chatbook/css/core/_variables.tcss"
RESET = ROOT / "tldw_chatbook/css/core/_reset.tcss"
BUTTONS = ROOT / "tldw_chatbook/css/components/_buttons.tcss"
FORMS = ROOT / "tldw_chatbook/css/components/_forms.tcss"
AGENTIC = ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
BASE_COMPONENTS = ROOT / "tldw_chatbook/Widgets/base_components.py"
WIDGETS = ROOT / "tldw_chatbook/css/components/_widgets.tcss"
MESSAGES = ROOT / "tldw_chatbook/css/components/_messages.tcss"
CHAT = ROOT / "tldw_chatbook/css/features/_chat.tcss"
CHAT_TABS = ROOT / "tldw_chatbook/css/features/_chat_tabs.tcss"
SIDEBARS = ROOT / "tldw_chatbook/css/layout/_sidebars.tcss"
BUNDLE = ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"
CODING = ROOT / "tldw_chatbook/css/features/_coding.tcss"
SEARCH_RAG = ROOT / "tldw_chatbook/css/features/_search-rag.tcss"
CONFIG_SEARCH = ROOT / "tldw_chatbook/css/features/config_search.tcss"
FEATURE_ALERTS = ROOT / "tldw_chatbook/css/features/feature_alerts.tcss"
RAG_SEARCH_WINDOW = ROOT / "tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py"


def css_blocks(text: str, selector: str) -> list[str]:
    """Return CSS rule bodies whose selector lists contain selector."""
    uncommented = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    blocks = []
    for match in re.finditer(r"\{(?P<body>[^{}]*)\}", uncommented, flags=re.DOTALL):
        prefix = uncommented[: match.start()]
        selector_start = max(prefix.rfind("}"), prefix.rfind(";")) + 1
        selector_text = prefix[selector_start : match.start()]
        selectors = [item.strip() for item in selector_text.split(",")]
        if selector in selectors:
            blocks.append(match.group("body"))
    return blocks


def css_block(text: str, selector: str) -> str:
    """Return a CSS rule body whose selector list contains selector."""
    blocks = css_blocks(text, selector)
    if blocks:
        return blocks[0]
    raise AssertionError(f"Missing CSS block for {selector}")


def assert_non_obscuring_focus(block: str) -> None:
    assert "outline: heavy" not in block
    assert "reverse" not in block
    assert "text-style: bold underline;" in block


def assert_thin_input_focus(block: str) -> None:
    assert "outline: heavy" not in block
    assert "border: thick" not in block
    assert "border: solid $ds-input-focus-border;" in block
    assert "border-bottom: solid $ds-input-focus-accent;" in block
    assert "$error" not in block
    assert "$warning" not in block


def assert_stable_solid_border_geometry(base: str, focus: str) -> None:
    for block in (base, focus):
        assert "border: thick" not in block
        assert "border: none" not in block
        assert "border: solid" in block
        assert "border-bottom: solid" in block


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


def test_settings_compact_input_focus_preserves_single_row_content():
    text = AGENTIC.read_text(encoding="utf-8")
    block = css_block(text, ".settings-compact-input:focus")
    assert "border:" not in block
    assert "border-bottom:" not in block
    assert "outline: none" not in block
    assert "outline: solid $ds-input-focus-accent;" in block
    assert "background: $ds-input-focus-bg;" in block


def test_top_navigation_inline_focus_uses_hybrid_contract():
    from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar

    text = MainNavigationBar.DEFAULT_CSS
    focus = css_block(text, ".nav-button:focus")
    active = css_block(text, ".nav-button.is-active")
    active_focus = css_block(text, ".nav-button.is-active:focus")
    assert_non_obscuring_focus(focus)
    assert "outline: heavy" not in active
    assert_non_obscuring_focus(active_focus)


def test_shared_navigation_button_uses_non_obscuring_active_and_focus_states():
    text = BASE_COMPONENTS.read_text(encoding="utf-8")
    focus = css_block(text, "NavigationButton:focus")
    active = css_block(text, "NavigationButton.active")
    active_focus = css_block(text, "NavigationButton.active:focus")
    assert_non_obscuring_focus(focus)
    assert "$ds-focus-bg" in focus or "$ds-surface-raised" in focus
    assert "border-left: thick" not in active
    assert "outline: heavy" not in active
    assert "text-style: bold;" in active
    assert_non_obscuring_focus(active_focus)
    assert "$ds-focus-bg" in active_focus or "$ds-surface-raised" in active_focus


def test_shared_collapsible_header_focus_is_underlined_and_non_heavy():
    text = WIDGETS.read_text(encoding="utf-8")
    block = css_block(text, "Collapsible > .collapsible--header:focus")
    collapsed_focus = css_block(text, "Collapsible.-collapsed > .collapsible--header:focus")
    assert_non_obscuring_focus(block)
    assert "outline: heavy" not in block
    assert "border-bottom: solid $ds-focus-accent;" in collapsed_focus


def test_message_action_buttons_focus_without_obscuring_labels():
    text = MESSAGES.read_text(encoding="utf-8")
    for selector in (".message-actions Button:focus", ".message-actions Button:focus:hover"):
        block = css_block(text, selector)
        assert_non_obscuring_focus(block)
        assert "$ds-focus-bg" in block or "$ds-surface-raised" in block


def test_chat_sidebar_toggle_focus_uses_two_non_obscuring_cues():
    text = CHAT.read_text(encoding="utf-8")
    block = css_block(text, ".chat-sidebar-toggle-button:focus")
    assert_non_obscuring_focus(block)
    assert "$ds-focus-bg" in block or "$ds-surface-raised" in block


def test_chat_rag_focus_within_uses_non_semantic_container_cue():
    text = CHAT.read_text(encoding="utf-8")
    blocks = css_blocks(text, ".rag-settings-panel:focus-within")
    assert len(blocks) == 1
    block = blocks[0]
    assert "$accent" not in block
    assert "$boost" not in block
    assert "border: round $ds-focus-accent;" in block
    assert "background: $panel;" in block


def test_chat_tab_active_state_is_readable_without_dominant_fill():
    text = CHAT_TABS.read_text(encoding="utf-8")
    active = css_block(text, ".chat-tab.active")
    active_focus = css_block(text, ".chat-tab.active:focus")
    assert "$primary" not in active
    assert "background: $ds-focus-bg;" in active
    assert "color: $ds-focus-fg;" in active
    assert "text-style: bold;" in active
    assert_non_obscuring_focus(active_focus)
    assert "$ds-focus-bg" in active_focus or "$ds-surface-raised" in active_focus


def test_feature_buttons_inherit_shared_button_focus_contract_without_duplicate_rules():
    button_text = BUTTONS.read_text(encoding="utf-8")
    for selector in ("Button:focus", "Button:hover:focus"):
        block = css_block(button_text, selector)
        assert_non_obscuring_focus(block)
        assert "$ds-focus-bg" in block or "$ds-surface-raised" in block

    assert css_blocks(CODING.read_text(encoding="utf-8"), ".coding-nav-button:focus") == []
    assert (
        css_blocks(
            FEATURE_ALERTS.read_text(encoding="utf-8"),
            "FeatureNotAvailableDialog Button:focus",
        )
        == []
    )


def test_search_rag_query_input_focus_targets_rendered_input_without_jitter():
    ui_text = RAG_SEARCH_WINDOW.read_text(encoding="utf-8")
    text = SEARCH_RAG.read_text(encoding="utf-8")
    assert 'classes="search-query-input-enhanced"' in ui_text
    base = css_block(text, ".search-query-input-enhanced")
    focus = css_block(text, ".search-query-input-enhanced:focus")
    assert_stable_solid_border_geometry(base, focus)
    assert_thin_input_focus(focus)
    assert "background: $ds-input-focus-bg;" in focus


def test_config_search_highlight_focus_uses_thin_non_semantic_focus():
    text = CONFIG_SEARCH.read_text(encoding="utf-8")
    for base_selector, focus_selector in (
        (".search-highlight", "Input.search-highlight:focus"),
        ("TextArea.search-highlight", "TextArea.search-highlight:focus"),
    ):
        base = css_block(text, base_selector)
        focus = css_block(text, focus_selector)
        assert_stable_solid_border_geometry(base, focus)
        assert_thin_input_focus(focus)
        assert "background: $ds-input-focus-bg;" in focus


def test_legacy_sidebar_focus_overrides_defer_to_shared_contracts():
    text = SIDEBARS.read_text(encoding="utf-8")
    for selector in (
        ".sidebar *:focus",
        ".sidebar Button:focus",
        ".sidebar Select:focus",
        ".sidebar Input:focus",
        ".setting-input:focus",
        ".sidebar-resize-button:focus",
    ):
        assert css_blocks(text, selector) == []


@pytest.mark.parametrize("selector", (".setting-input", ".sidebar-input", ".sidebar Select"))
def test_sidebar_inputs_use_stable_base_geometry_for_shared_focus(selector: str):
    text = SIDEBARS.read_text(encoding="utf-8")
    block = css_block(text, selector)
    assert "border: thick" not in block
    assert "border: round" not in block
    assert "border: solid" in block
    assert "border-bottom: solid" in block


@pytest.mark.parametrize("selector", (".setting-input", ".sidebar-input", ".sidebar Select"))
def test_bundled_sidebar_inputs_keep_stable_effective_geometry(selector: str):
    text = BUNDLE.read_text(encoding="utf-8")
    blocks = css_blocks(text, selector)
    assert blocks
    block = blocks[-1]
    assert "border: thick" not in block
    assert "border: round" not in block
    assert "border: solid" in block
    assert "border-bottom: solid" in block


def test_sidebar_preset_active_state_is_readable_without_dominant_fill():
    text = SIDEBARS.read_text(encoding="utf-8")
    active = css_block(text, ".preset-button.active")
    assert "outline: heavy" not in active
    assert "reverse" not in active
    assert "$primary" not in active
    assert "$warning" not in active
    assert "$error" not in active
    assert "background: $ds-focus-bg;" in active
    assert "color: $ds-focus-fg;" in active
    assert "text-style: bold underline;" in active
