import re
from pathlib import Path

import pytest

from tldw_chatbook.css.build_css import CSS_MODULES


ROOT = Path(__file__).resolve().parents[2]
CSS_DIR = ROOT / "tldw_chatbook/css"
VARIABLES = ROOT / "tldw_chatbook/css/core/_variables.tcss"
RESET = ROOT / "tldw_chatbook/css/core/_reset.tcss"
BUTTONS = ROOT / "tldw_chatbook/css/components/_buttons.tcss"
FORMS = ROOT / "tldw_chatbook/css/components/_forms.tcss"
LISTS = ROOT / "tldw_chatbook/css/components/_lists.tcss"
AGENTIC = ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
BASE_COMPONENTS = ROOT / "tldw_chatbook/Widgets/base_components.py"
WIDGETS = ROOT / "tldw_chatbook/css/components/_widgets.tcss"
MESSAGES = ROOT / "tldw_chatbook/css/components/_messages.tcss"
CHAT = ROOT / "tldw_chatbook/css/features/_chat.tcss"
CHAT_TABS = ROOT / "tldw_chatbook/css/features/_chat_tabs.tcss"
CONVERSATIONS = ROOT / "tldw_chatbook/css/features/_conversations.tcss"
SIDEBARS = ROOT / "tldw_chatbook/css/layout/_sidebars.tcss"
LAYOUT_TABS = ROOT / "tldw_chatbook/css/layout/_tabs.tcss"
BUNDLE = ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"
CODING = ROOT / "tldw_chatbook/css/features/_coding.tcss"
CODE_REPO = ROOT / "tldw_chatbook/css/features/_code_repo.tcss"
SEARCH_RAG = ROOT / "tldw_chatbook/css/features/_search-rag.tcss"
CONFIG_SEARCH = ROOT / "tldw_chatbook/css/features/config_search.tcss"
FEATURE_ALERTS = ROOT / "tldw_chatbook/css/features/feature_alerts.tcss"
INGESTION_REBUILT = ROOT / "tldw_chatbook/css/features/_ingestion_rebuilt.tcss"
NEW_INGEST = ROOT / "tldw_chatbook/css/features/_new_ingest.tcss"
UNIFIED_SIDEBAR = ROOT / "tldw_chatbook/css/components/_unified_sidebar.tcss"
WIZARDS = ROOT / "tldw_chatbook/css/features/_wizards.tcss"
EVALUATION_UNIFIED = ROOT / "tldw_chatbook/css/features/_evaluation_unified.tcss"
EVAL_NAV_SCREEN = ROOT / "tldw_chatbook/UI/Evals/navigation/eval_nav_screen.py"
EMBEDDINGS = ROOT / "tldw_chatbook/css/features/_embeddings.tcss"
INGEST = ROOT / "tldw_chatbook/css/features/_ingest.tcss"
TOOLS_SETTINGS = ROOT / "tldw_chatbook/css/features/_tools-settings.tcss"
TAB_DROPDOWN = ROOT / "tldw_chatbook/css/features/_tab_dropdown.tcss"
LLM_MANAGEMENT = ROOT / "tldw_chatbook/css/features/_llm-management.tcss"
MEDIA = ROOT / "tldw_chatbook/css/features/_media.tcss"
MEDIA_NAVIGATION_PANEL = ROOT / "tldw_chatbook/Widgets/Media/media_navigation_panel.py"
MEDIA_LIST_PANEL = ROOT / "tldw_chatbook/Widgets/Media/media_list_panel.py"
REPO_TREE_WIDGETS = ROOT / "tldw_chatbook/Widgets/Coding_Widgets/repo_tree_widgets.py"
CHATBOOKS_IMPROVED = ROOT / "tldw_chatbook/css/features/_chatbooks_improved.tcss"
CHATBOOKS_WINDOW_IMPROVED = ROOT / "tldw_chatbook/UI/Chatbooks_Window_Improved.py"
SAMPLE_BROWSER_DIALOG = ROOT / "tldw_chatbook/Widgets/Evals/sample_browser_dialog.py"
RAG_SEARCH_WINDOW = ROOT / "tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py"
EMOJI_PICKER = ROOT / "tldw_chatbook/Widgets/emoji_picker.py"
ENHANCED_FILE_PICKER = ROOT / "tldw_chatbook/Widgets/enhanced_file_picker.py"
MODEL_CARD_VIEWER = ROOT / "tldw_chatbook/Widgets/HuggingFace/model_card_viewer.py"
NOTES_TOOLBAR = ROOT / "tldw_chatbook/Widgets/Note_Widgets/notes_toolbar.py"
NOTES_EDITOR = ROOT / "tldw_chatbook/Widgets/Note_Widgets/notes_editor_widget.py"
NOTES_SYNC = ROOT / "tldw_chatbook/Widgets/Note_Widgets/notes_sync_widget.py"
NOTES_SYNC_IMPROVED = ROOT / "tldw_chatbook/Widgets/Note_Widgets/notes_sync_widget_improved.py"

BUNDLED_RESIDUAL_ACTIVE_SELECTED_CONTRACTS = (
    (LLM_MANAGEMENT, ".llm-nav-pane .llm-nav-button.-active"),
    (CODE_REPO, ".tree-node-selected"),
    (TAB_DROPDOWN, "#tab-dropdown-select SelectOverlay Option.-selected"),
)

SOURCE_ONLY_CSS_MODULES = (
    NEW_INGEST,
    UNIFIED_SIDEBAR,
)

NATIVE_CHOICE_SELECTED_MARKERS = (
    ".option-list--option-highlighted",
    ".selection-list--button-highlighted",
    ".selection-list--button-selected",
    ".selection-list--button-selected-highlighted",
    ".tree--cursor",
)

NATIVE_CHOICE_HOVER_MARKERS = (
    ".option-list--option-hover",
    ".tree--highlight-line",
)


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


def css_selectors(text: str) -> list[str]:
    """Return every selector from CSS rule selector lists."""
    uncommented = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    selectors = []
    for match in re.finditer(r"\{(?P<body>[^{}]*)\}", uncommented, flags=re.DOTALL):
        prefix = uncommented[: match.start()]
        selector_start = max(prefix.rfind("}"), prefix.rfind(";")) + 1
        selector_text = prefix[selector_start : match.start()]
        selectors.extend(item.strip() for item in selector_text.split(",") if item.strip())
    return selectors


def css_block(text: str, selector: str) -> str:
    """Return a CSS rule body whose selector list contains selector."""
    blocks = css_blocks(text, selector)
    if blocks:
        return blocks[0]
    raise AssertionError(f"Missing CSS block for {selector}")


def bundled_css_module_paths() -> set[Path]:
    """Return stylesheet module paths included by the generated app bundle."""
    return {
        (CSS_DIR / module).resolve()
        for module in CSS_MODULES
        if isinstance(module, str)
    }


def assert_non_obscuring_focus(block: str) -> None:
    assert "outline: heavy" not in block
    assert "reverse" not in block
    assert "text-style: bold underline;" in block


def assert_native_toggle_focus_contract(block: str) -> None:
    assert "outline: heavy" not in block
    assert "reverse" not in block
    assert "border: thick" not in block
    assert "$block-cursor" not in block
    assert "$block-hover" not in block
    assert "$primary" not in block
    assert "$accent" not in block
    assert "$warning" not in block
    assert "$error" not in block
    assert "background: $ds-focus-bg;" in block
    assert "color: $ds-focus-fg;" in block
    assert "text-style: bold underline;" in block


def assert_thin_input_focus(block: str) -> None:
    assert "outline: heavy" not in block
    assert "border: thick" not in block
    assert "border: solid $ds-input-focus-border;" in block
    assert "border-bottom: solid $ds-input-focus-accent;" in block
    assert "$error" not in block
    assert "$warning" not in block


def assert_thin_inline_input_focus(block: str) -> None:
    assert "outline: heavy" not in block
    assert "border: thick" not in block
    assert "border: round" not in block
    assert "border: solid $ds-input-focus-border;" in block
    assert "border-bottom: solid $ds-input-focus-accent;" in block
    assert "background: $ds-input-focus-bg;" in block
    assert "color: $ds-text-primary;" in block
    assert "$error" not in block
    assert "$warning" not in block


def assert_stable_solid_border_geometry(base: str, focus: str) -> None:
    for block in (base, focus):
        assert "border: thick" not in block
        assert "border: none" not in block
        assert "border: solid" in block
        assert "border-bottom: solid" in block


def assert_embeddings_focus_and_active_contracts(text: str) -> None:
    for selector in (
        ".embeddings-nav-button:focus",
        ".embeddings-toggle-button-enhanced:focus",
    ):
        block = css_block(text, selector)
        assert_non_obscuring_focus(block)
        assert "$ds-focus-bg" in block or "$ds-surface-raised" in block
        assert "$primary" not in block
        assert "$accent" not in block

    for selector in (".embeddings-nav-button.-active", ".filter-button.active"):
        block = css_block(text, selector)
        assert "outline: heavy" not in block
        assert "$primary" not in block
        assert "$accent" not in block
        assert "background: $ds-focus-bg;" in block
        assert "color: $ds-focus-fg;" in block
        assert "text-style: bold underline;" in block

    for selector in (
        "#embeddings-model-list ModelListItem.--highlight",
        "#embeddings-collection-list CollectionListItem.--highlight",
    ):
        block = css_block(text, selector)
        assert_readable_selected_state_contract(block)
        assert_no_dominant_selected_geometry(block)

    for selector in (
        ".embeddings-list-item.-selected",
        "ModelListItem.-selected",
        "CollectionListItem.-selected",
    ):
        assert css_blocks(text, selector) == []


def assert_feature_nav_active_contract(block: str) -> None:
    assert "outline: heavy" not in block
    assert "reverse" not in block
    assert "$accent" not in block
    assert "$primary" not in block
    assert "background: $ds-focus-bg;" in block
    assert "color: $ds-focus-fg;" in block
    assert "text-style: bold underline;" in block


def assert_readable_selected_state_contract(block: str) -> None:
    assert "outline: heavy" not in block
    assert "reverse" not in block
    assert "$block-cursor" not in block
    assert "$block-hover" not in block
    assert "$accent" not in block
    assert "$primary" not in block
    assert "background: $ds-focus-bg;" in block
    assert "color: $ds-focus-fg;" in block
    assert "text-style: bold underline;" in block


def assert_no_dominant_selected_geometry(block: str) -> None:
    assert "border-left" not in block
    assert "border-right" not in block
    assert "border: thick" not in block
    assert "border: heavy" not in block


def assert_readable_inline_selected_state_contract(block: str) -> None:
    assert "outline: heavy" not in block
    assert "reverse" not in block
    assert "$accent" not in block
    assert "$primary" not in block
    assert "background: $surface;" in block
    assert "color: $text;" in block
    assert "text-style: bold underline;" in block


def assert_native_row_selected_state_contract(block: str) -> None:
    assert_readable_inline_selected_state_contract(block)
    assert_no_dominant_selected_geometry(block)
    assert "$block-cursor" not in block
    assert "$primary-background" not in block
    assert "$primary" not in block
    assert "$accent" not in block
    assert "$warning" not in block
    assert "$error" not in block


def assert_native_row_hover_state_contract(block: str) -> None:
    assert "outline: heavy" not in block
    assert "reverse" not in block
    assert "$block-hover" not in block
    assert "$block-cursor" not in block
    assert "$primary-background" not in block
    assert "$primary" not in block
    assert "$accent" not in block
    assert "$warning" not in block
    assert "$error" not in block
    assert "background: $surface;" in block or "background: $surface-lighten-1;" in block
    assert "color: $text;" in block


def assert_all_native_choice_selectors_follow_contracts(text: str) -> None:
    selected_selectors = [
        selector
        for selector in css_selectors(text)
        if any(marker in selector for marker in NATIVE_CHOICE_SELECTED_MARKERS)
    ]
    assert selected_selectors
    for selector in selected_selectors:
        for block in css_blocks(text, selector):
            assert_native_row_selected_state_contract(block)

    hover_selectors = [
        selector
        for selector in css_selectors(text)
        if any(marker in selector for marker in NATIVE_CHOICE_HOVER_MARKERS)
    ]
    for selector in hover_selectors:
        for block in css_blocks(text, selector):
            assert_native_row_hover_state_contract(block)


def assert_all_native_tab_selectors_follow_contracts(text: str) -> None:
    active_selectors = [
        selector
        for selector in css_selectors(text)
        if ("Tab" in selector or "Tabs" in selector) and ".-active" in selector
    ]
    assert active_selectors
    for selector in active_selectors:
        for block in css_blocks(text, selector):
            assert_readable_selected_state_contract(block)


def assert_custom_widget_focus_contract(block: str) -> None:
    assert "outline: heavy" not in block
    assert "reverse" not in block
    assert "border: thick" not in block
    assert "$accent" not in block
    assert "$primary" not in block
    assert "$error" not in block
    assert "$warning" not in block
    assert "border: round $ds-focus-accent;" in block
    assert "background: $ds-focus-bg;" in block
    assert "color: $ds-focus-fg;" in block
    assert "text-style: bold underline;" in block


def assert_drop_zone_active_contract(text: str) -> None:
    block = css_block(text, ".drop-zone.active")
    assert "outline: heavy" not in block
    assert "reverse" not in block
    assert "border: thick" not in block
    assert "border-color:" not in block
    assert "transform:" not in block
    assert "box-shadow" not in block
    assert "$primary" not in block
    assert "$accent" not in block
    assert "background: $ds-focus-bg;" in block
    assert "border: round $ds-focus-accent;" in block
    assert "color: $ds-focus-fg;" in block
    assert "text-style: bold underline;" in block

    icon = css_block(text, ".drop-zone.active .drop-icon")
    assert "animation:" not in icon
    assert "$primary" not in icon
    assert "$accent" not in icon
    assert "color: $ds-focus-fg;" in icon
    assert "text-style: bold underline;" in icon

    message = css_block(text, ".drop-zone.active .drop-message")
    assert "$primary" not in message
    assert "$accent" not in message
    assert "color: $ds-focus-fg;" in message
    assert "text-style: bold underline;" in message


def assert_wizard_progress_active_contracts(text: str, scope: str = "") -> None:
    prefix = f"{scope} " if scope else ""
    number = css_block(text, f"{prefix}.step-number.active")
    assert "outline: heavy" not in number
    assert "reverse" not in number
    assert "border: thick" not in number
    assert "$primary" not in number
    assert "$accent" not in number
    assert "$warning" not in number
    assert "$error" not in number
    assert "background: $ds-focus-bg;" in number
    assert "border: round $ds-focus-accent;" in number
    assert "color: $ds-focus-fg;" in number
    assert "text-style: bold underline;" in number

    title = css_block(text, f"{prefix}.step-title.active")
    assert "outline: heavy" not in title
    assert "reverse" not in title
    assert "$primary" not in title
    assert "$accent" not in title
    assert "$warning" not in title
    assert "$error" not in title
    assert "color: $ds-focus-fg;" in title
    assert "text-style: bold underline;" in title

    if scope:
        return

    step = css_block(text, ".wizard-step.active")
    assert "visibility: visible;" in step
    assert "outline: heavy" not in step
    assert "reverse" not in step
    assert "border" not in step
    assert "background:" not in step
    assert "box-shadow" not in step
    assert "transform:" not in step
    assert "$primary" not in step
    assert "$accent" not in step


def assert_wizard_selection_active_contracts(text: str) -> None:
    for selector in (
        ".content-type-card.selected",
        ".preset-card.selected",
        "ProgressStep .status-item.active",
        "ImportProgressStep .status-item.active",
        "SmartContentTree Tree > .selected-node",
    ):
        assert_readable_selected_state_contract(css_block(text, selector))

    for selector in (
        ".content-type-card.selected .content-type-title",
        ".content-type-card.selected .content-type-description",
        ".preset-card.selected .preset-name",
        ".preset-card.selected .preset-description",
        ".preset-card.selected .preset-detail",
    ):
        block = css_block(text, selector)
        assert "$text-muted" not in block
        assert "$primary" not in block
        assert "$accent" not in block
        assert "color: $ds-focus-fg;" in block
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


def test_native_toggle_focus_states_use_non_obscuring_contracts():
    text = FORMS.read_text(encoding="utf-8")
    for selector in (
        "ToggleButton:focus > .toggle--label",
        "ToggleButton.-textual-compact:focus > .toggle--label",
        "ToggleButton:blur:hover > .toggle--label",
    ):
        assert_native_toggle_focus_contract(css_block(text, selector))

    focus = css_block(text, "Switch:focus")
    assert "border: thick" not in focus
    assert "$primary" not in focus
    assert "$accent" not in focus
    assert "$warning" not in focus
    assert "$error" not in focus
    assert "border: solid $ds-focus-accent;" in focus
    assert "background: $ds-focus-bg;" in focus


def test_bundled_native_toggle_focus_states_match_source_contracts():
    text = BUNDLE.read_text(encoding="utf-8")
    for selector in (
        "ToggleButton:focus > .toggle--label",
        "ToggleButton.-textual-compact:focus > .toggle--label",
        "ToggleButton:blur:hover > .toggle--label",
    ):
        assert_native_toggle_focus_contract(css_block(text, selector))

    focus = css_block(text, "Switch:focus")
    assert "border: thick" not in focus
    assert "$primary" not in focus
    assert "$accent" not in focus
    assert "$warning" not in focus
    assert "$error" not in focus
    assert "border: solid $ds-focus-accent;" in focus
    assert "background: $ds-focus-bg;" in focus


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


def test_console_session_tab_active_state_uses_selected_contract():
    for text in (
        AGENTIC.read_text(encoding="utf-8"),
        BUNDLE.read_text(encoding="utf-8"),
    ):
        for selector in (
            ".console-session-tab-active",
            ".console-session-tab-active:hover",
            ".console-session-tab-active:hover:focus",
        ):
            active = css_block(text, selector)
            assert_readable_selected_state_contract(active)
            assert_no_dominant_selected_geometry(active)
            assert "$ds-action-focus" not in active


def test_library_mode_chip_active_states_use_selected_focus_contracts():
    for text in (
        AGENTIC.read_text(encoding="utf-8"),
        BUNDLE.read_text(encoding="utf-8"),
    ):
        focus = css_block(text, ".library-mode-chip:focus")
        assert_non_obscuring_focus(focus)

        active = css_block(text, ".library-mode-chip.is-active")
        assert_readable_selected_state_contract(active)
        assert "border: solid $ds-focus-accent;" in active

        active_focus = css_block(text, ".library-mode-chip.is-active:focus")
        assert_non_obscuring_focus(active_focus)
        assert active_focus != active
        assert "border: round $ds-focus-accent;" in active_focus
        assert "$primary" not in active_focus
        assert "$accent" not in active_focus
        assert "background: $ds-focus-bg;" in active_focus
        assert "color: $ds-focus-fg;" in active_focus


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


def test_settings_category_active_states_use_selected_contract():
    for text in (
        AGENTIC.read_text(encoding="utf-8"),
        BUNDLE.read_text(encoding="utf-8"),
    ):
        for selector in (
            ".settings-active-section",
            "Button.settings-category-button.settings-active-section",
            "Button.settings-category-button.settings-active-section:focus",
            "Button.settings-category-button.settings-active-section:hover",
            "Button.settings-category-button.settings-active-section:hover:focus",
        ):
            active = css_block(text, selector)
            assert_readable_selected_state_contract(active)
            assert_no_dominant_selected_geometry(active)


def test_acp_selected_session_row_uses_selected_contract():
    for text in (
        AGENTIC.read_text(encoding="utf-8"),
        BUNDLE.read_text(encoding="utf-8"),
    ):
        blocks = css_blocks(text, ".acp-selected-session-row")
        assert len(blocks) == 1
        selected = blocks[0]
        assert_readable_selected_state_contract(selected)
        assert_no_dominant_selected_geometry(selected)


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


def test_conversations_collapsible_active_header_uses_selected_contract():
    for text in (
        CONVERSATIONS.read_text(encoding="utf-8"),
        BUNDLE.read_text(encoding="utf-8"),
    ):
        blocks = css_blocks(text, "Collapsible.-active > .collapsible--header")
        assert blocks, "Missing CSS block for Collapsible.-active > .collapsible--header"
        active = blocks[-1]
        assert_readable_selected_state_contract(active)
        assert_no_dominant_selected_geometry(active)
        assert "border-bottom: solid $ds-focus-accent;" in active


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
    for text in (
        CHAT_TABS.read_text(encoding="utf-8"),
        BUNDLE.read_text(encoding="utf-8"),
    ):
        for selector in (
            ".chat-tab.active",
            ".chat-tab.active:focus",
            ".chat-tab.active:hover",
            ".chat-tab.active:hover:focus",
        ):
            active = css_block(text, selector)
            assert_readable_selected_state_contract(active)
            assert_no_dominant_selected_geometry(active)


def test_layout_tab_active_states_use_underlined_selected_contracts():
    for path, selector in (
        (LAYOUT_TABS, "#tabs Button.-active"),
        (LAYOUT_TABS, "Tab.-active"),
        (LAYOUT_TABS, "Tabs:focus .-active"),
        (SIDEBARS, "TabbedContent Tab.-active"),
    ):
        blocks = css_blocks(path.read_text(encoding="utf-8"), selector)
        assert blocks
        for block in blocks:
            assert_readable_selected_state_contract(block)
            assert "border:" not in block

    for text in (
        LAYOUT_TABS.read_text(encoding="utf-8"),
        BUNDLE.read_text(encoding="utf-8"),
    ):
        active_link = css_block(text, ".tab-link.-active")
        assert "$accent" not in active_link
        assert "$primary" not in active_link
        assert "color: $ds-focus-fg;" in active_link
        assert "background: transparent;" in active_link
        assert "text-style: bold underline;" in active_link

    for selector in ("#tabs Button.-active", "TabbedContent Tab.-active"):
        blocks = css_blocks(BUNDLE.read_text(encoding="utf-8"), selector)
        assert blocks
        for block in blocks:
            assert_readable_selected_state_contract(block)

    assert_all_native_tab_selectors_follow_contracts(LAYOUT_TABS.read_text(encoding="utf-8"))
    assert_all_native_tab_selectors_follow_contracts(BUNDLE.read_text(encoding="utf-8"))


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


def test_ingestion_rebuilt_focus_overrides_defer_to_shared_contracts():
    text = INGESTION_REBUILT.read_text(encoding="utf-8")
    widget_focus_selector = re.compile(r"\b(Input|TextArea|Select|Button)\b.*:focus")
    offenders = [
        selector
        for selector in css_selectors(text)
        if widget_focus_selector.search(selector)
    ]
    assert offenders == []


def test_new_ingest_focus_overrides_defer_to_shared_contracts():
    text = NEW_INGEST.read_text(encoding="utf-8")
    card_base = css_block(text, ".media-card")
    card_focus = css_block(text, ".media-card:focus")
    assert "border: round $border;" in card_base
    assert "border: round $primary;" not in card_base
    assert "box-shadow" not in card_focus
    assert "border: round $ds-focus-accent;" in card_focus

    metadata_input_base = css_block(text, ".metadata-grid Input")
    assert "border: solid $border;" in metadata_input_base
    assert "border: solid $primary;" not in metadata_input_base

    assert css_blocks(text, ".mode-selector RadioButton") == []
    assert css_blocks(text, ".mode-selector RadioButton:hover") == []
    assert css_blocks(text, ".mode-selector RadioButton.-active") == []
    assert css_blocks(text, ".mode-description") == []
    assert css_block(text, "#mode-description")
    mode_button_base = css_block(text, "#mode-selector Button")
    mode_button_active = css_block(text, "#mode-selector Button.active")
    assert "border: round $border;" in mode_button_base
    assert_readable_selected_state_contract(mode_button_active)
    assert_no_dominant_selected_geometry(mode_button_active)

    widget_focus_selector = re.compile(r"\b(Button|Input|RadioButton)\b.*:focus")
    offenders = [
        selector
        for selector in css_selectors(text)
        if widget_focus_selector.search(selector)
    ]
    assert offenders == []


def test_new_ingest_drop_zone_active_state_is_readable():
    assert_drop_zone_active_contract(NEW_INGEST.read_text(encoding="utf-8"))


def test_wizard_progress_active_states_are_readable_without_dominant_fill():
    assert_wizard_progress_active_contracts(WIZARDS.read_text(encoding="utf-8"))


def test_bundled_wizard_progress_active_states_match_source_contracts():
    assert_wizard_progress_active_contracts(BUNDLE.read_text(encoding="utf-8"))


def test_wizard_progress_default_css_matches_active_state_contract():
    from tldw_chatbook.UI.Wizards.BaseWizard import WizardProgress

    assert_wizard_progress_active_contracts(WizardProgress.DEFAULT_CSS, scope="WizardProgress")


def test_wizard_selection_states_are_readable_without_dominant_fill():
    assert_wizard_selection_active_contracts(WIZARDS.read_text(encoding="utf-8"))


def test_bundled_wizard_selection_states_match_source_contracts():
    assert_wizard_selection_active_contracts(BUNDLE.read_text(encoding="utf-8"))


def test_evaluation_unified_focus_overrides_defer_to_shared_contracts():
    text = EVALUATION_UNIFIED.read_text(encoding="utf-8")
    widget_focus_selector = re.compile(r"\b(Input|TextArea|Select)\b.*:focus")
    offenders = [
        selector
        for selector in css_selectors(text)
        if widget_focus_selector.search(selector)
    ]
    assert offenders == []


def test_embeddings_focus_and_active_states_follow_shared_contracts():
    text = EMBEDDINGS.read_text(encoding="utf-8")
    assert_embeddings_focus_and_active_contracts(text)

    widget_focus_selector = re.compile(r"\b(Input|TextArea|Select|Button|Checkbox)\b.*:focus")
    offenders = [
        selector
        for selector in css_selectors(text)
        if widget_focus_selector.search(selector)
    ]
    assert offenders == []


def test_bundled_embeddings_focus_and_active_states_match_source_contracts():
    text = BUNDLE.read_text(encoding="utf-8")
    assert_embeddings_focus_and_active_contracts(text)


def test_feature_navigation_active_and_dropdown_focus_states_follow_contracts():
    tab_text = TAB_DROPDOWN.read_text(encoding="utf-8")
    tab_base = css_block(tab_text, "#tab-dropdown-select")
    tab_focus = css_block(tab_text, "#tab-dropdown-select:focus")
    assert_stable_solid_border_geometry(tab_base, tab_focus)
    assert_thin_input_focus(tab_focus)
    assert "background: $ds-input-focus-bg;" in tab_focus

    ingest_text = INGEST.read_text(encoding="utf-8")
    assert_feature_nav_active_contract(
        css_block(ingest_text, ".ingest-nav-pane .ingest-nav-button.active")
    )

    tools_text = TOOLS_SETTINGS.read_text(encoding="utf-8")
    assert_feature_nav_active_contract(
        css_block(tools_text, ".tools-nav-pane .ts-nav-button.active-nav")
    )

    search_text = SEARCH_RAG.read_text(encoding="utf-8")
    assert_feature_nav_active_contract(
        css_block(
            search_text,
            ".search-nav-pane .search-nav-button.-active-search-sub-view",
        )
    )


def test_bundled_feature_navigation_states_match_source_contracts():
    text = BUNDLE.read_text(encoding="utf-8")
    tab_base = css_block(text, "#tab-dropdown-select")
    tab_focus = css_block(text, "#tab-dropdown-select:focus")
    assert_stable_solid_border_geometry(tab_base, tab_focus)
    assert_thin_input_focus(tab_focus)
    assert_feature_nav_active_contract(
        css_block(text, ".ingest-nav-pane .ingest-nav-button.active")
    )
    assert_feature_nav_active_contract(
        css_block(text, ".tools-nav-pane .ts-nav-button.active-nav")
    )
    assert_feature_nav_active_contract(
        css_block(
            text,
            ".search-nav-pane .search-nav-button.-active-search-sub-view",
        )
    )


def test_llm_management_default_css_nav_states_follow_contracts():
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

    text = LLMManagementWindow.DEFAULT_CSS
    hover = css_block(text, ".llm-nav-button:hover")
    assert "outline: heavy" not in hover
    assert "border: thick" not in hover
    assert "$primary" not in hover
    assert "$accent" not in hover
    assert "$warning" not in hover
    assert "$error" not in hover
    assert "background: $ds-surface-raised;" in hover
    assert "color: $ds-text-primary;" in hover

    active = css_block(text, ".llm-nav-button.-active")
    assert_readable_selected_state_contract(active)
    assert_no_dominant_selected_geometry(active)


def test_ingest_selected_files_list_uses_non_obscuring_container_cue():
    for path in (INGEST, BUNDLE):
        blocks = css_blocks(path.read_text(encoding="utf-8"), ".ingest-selected-files-list")
        assert blocks, "Missing CSS block for .ingest-selected-files-list"
        for block in blocks:
            assert "outline: heavy" not in block
            assert "border: thick" not in block
            assert "border-left" not in block
            assert "border-right" not in block
            assert "$primary" not in block
            assert "$accent" not in block
            assert "$warning" not in block
            assert "$error" not in block
            assert "border: round $ds-focus-accent;" in block
            assert "background: $surface;" in block


def test_residual_active_selected_states_follow_shared_contracts():
    for path, selector in BUNDLED_RESIDUAL_ACTIVE_SELECTED_CONTRACTS:
        block = css_block(path.read_text(encoding="utf-8"), selector)
        assert_readable_selected_state_contract(block)
        assert_no_dominant_selected_geometry(block)


def test_bundled_residual_active_selected_source_files_are_shipped_modules():
    shipped_modules = bundled_css_module_paths()
    unshipped_contracts = [
        f"{path.relative_to(CSS_DIR).as_posix()} :: {selector}"
        for path, selector in BUNDLED_RESIDUAL_ACTIVE_SELECTED_CONTRACTS
        if path.resolve() not in shipped_modules
    ]
    assert unshipped_contracts == []


def test_source_only_css_modules_are_not_part_of_app_bundle():
    shipped_modules = bundled_css_module_paths()
    unexpectedly_shipped = [
        path.relative_to(CSS_DIR).as_posix()
        for path in SOURCE_ONLY_CSS_MODULES
        if path.resolve() in shipped_modules
    ]
    assert unexpectedly_shipped == []


def test_bundled_residual_active_selected_states_match_source_contracts():
    text = BUNDLE.read_text(encoding="utf-8")
    for _, selector in BUNDLED_RESIDUAL_ACTIVE_SELECTED_CONTRACTS:
        block = css_block(text, selector)
        assert_readable_selected_state_contract(block)
        assert_no_dominant_selected_geometry(block)


def test_native_listview_row_states_follow_shared_contracts():
    text = LISTS.read_text(encoding="utf-8")
    assert "height: auto;" in css_block(text, "ListView ListItem")
    assert_native_row_hover_state_contract(css_block(text, "ListView ListItem:hover"))
    assert_native_row_selected_state_contract(css_block(text, "ListView ListItem.--highlight"))


def test_bundled_native_listview_row_states_keep_effective_contracts():
    text = BUNDLE.read_text(encoding="utf-8")
    assert len(css_blocks(text, "ListView ListItem:hover")) == 1
    assert len(css_blocks(text, "ListView ListItem.--highlight")) == 1
    assert "height: auto;" in css_blocks(text, "ListView ListItem")[-1]
    assert_native_row_hover_state_contract(css_blocks(text, "ListView ListItem:hover")[-1])
    assert_native_row_selected_state_contract(css_blocks(text, "ListView ListItem.--highlight")[-1])

    assert_native_row_hover_state_contract(css_block(text, "#chatbooks-list ListItem:hover"))
    for selector in (
        "#chatbooks-list ListItem.--highlight",
        "ConfigSearchResult.--highlight",
    ):
        assert_native_row_selected_state_contract(css_block(text, selector))

    assert css_blocks(text, "ConfigSearchResult ListItem.--highlight") == []


def test_config_search_result_highlight_targets_rendered_list_item():
    text = CONFIG_SEARCH.read_text(encoding="utf-8")
    assert css_blocks(text, "ConfigSearchResult ListItem.--highlight") == []
    assert_native_row_selected_state_contract(css_block(text, "ConfigSearchResult.--highlight"))


def test_native_datatable_row_states_follow_shared_contracts():
    text = LISTS.read_text(encoding="utf-8")
    assert_native_row_selected_state_contract(css_block(text, "DataTable > .datatable--cursor"))
    assert_native_row_hover_state_contract(css_block(text, "DataTable > .datatable--hover"))
    assert_native_row_selected_state_contract(css_block(text, "DataTable > .datatable--selected"))


def test_bundled_native_datatable_row_states_keep_effective_contracts():
    text = BUNDLE.read_text(encoding="utf-8")
    for selector in (
        "DataTable > .datatable--cursor",
        "DataTable > .datatable--hover",
        "DataTable > .datatable--selected",
    ):
        assert len(css_blocks(text, selector)) == 1

    for selector in ("DataTable > .datatable--cursor", "DataTable > .datatable--selected"):
        assert_native_row_selected_state_contract(css_blocks(text, selector)[-1])
    assert_native_row_hover_state_contract(css_blocks(text, "DataTable > .datatable--hover")[-1])


def test_native_choice_and_tree_states_follow_shared_contracts():
    text = LISTS.read_text(encoding="utf-8")
    for selector in (
        "OptionList > .option-list--option-highlighted",
        "OptionList:focus > .option-list--option-highlighted",
        "SelectionList > .selection-list--button-highlighted",
        "SelectionList > .selection-list--button-selected",
        "SelectionList > .selection-list--button-selected-highlighted",
        "Tree > .tree--cursor",
        "Tree:focus > .tree--cursor",
    ):
        assert_native_row_selected_state_contract(css_block(text, selector))

    for selector in (
        "OptionList > .option-list--option-hover",
        "Tree > .tree--highlight-line",
    ):
        assert_native_row_hover_state_contract(css_block(text, selector))

    assert_all_native_choice_selectors_follow_contracts(WIZARDS.read_text(encoding="utf-8"))


def test_bundled_native_choice_and_tree_states_match_source_contracts():
    text = BUNDLE.read_text(encoding="utf-8")
    for selector in (
        "OptionList > .option-list--option-highlighted",
        "OptionList:focus > .option-list--option-highlighted",
        "SelectionList > .selection-list--button-highlighted",
        "SelectionList > .selection-list--button-selected",
        "SelectionList > .selection-list--button-selected-highlighted",
        "Tree > .tree--cursor",
        "Tree:focus > .tree--cursor",
    ):
        assert_native_row_selected_state_contract(css_block(text, selector))

    for selector in (
        "OptionList > .option-list--option-hover",
        "Tree > .tree--highlight-line",
    ):
        assert_native_row_hover_state_contract(css_block(text, selector))

    assert_all_native_choice_selectors_follow_contracts(text)


def test_media_selected_and_active_states_follow_shared_contracts():
    media_text = MEDIA.read_text(encoding="utf-8")
    for selector in (
        ".keyword-item.selected",
        ".keyword-item.selected:hover",
        ".review-item.selected",
        ".review-item.selected:hover",
    ):
        assert_readable_selected_state_contract(css_block(media_text, selector))

    nav_text = MEDIA_NAVIGATION_PANEL.read_text(encoding="utf-8")
    assert_readable_inline_selected_state_contract(
        css_block(nav_text, "MediaNavigationPanel .media-type-button.active")
    )

    list_text = MEDIA_LIST_PANEL.read_text(encoding="utf-8")
    assert_readable_inline_selected_state_contract(
        css_block(list_text, "MediaListPanel .media-item.selected")
    )


def test_bundled_media_selected_states_match_source_contracts():
    text = BUNDLE.read_text(encoding="utf-8")
    for selector in (
        ".keyword-item.selected",
        ".keyword-item.selected:hover",
        ".review-item.selected",
        ".review-item.selected:hover",
    ):
        assert_readable_selected_state_contract(css_block(text, selector))


def test_repo_tree_widget_selected_state_matches_code_repo_contract():
    text = REPO_TREE_WIDGETS.read_text(encoding="utf-8")
    selected = css_block(text, ".tree-node-selected")
    assert_readable_selected_state_contract(selected)
    assert_no_dominant_selected_geometry(selected)


def test_chatbooks_search_input_focus_uses_stable_thin_contracts():
    text = CHATBOOKS_IMPROVED.read_text(encoding="utf-8")
    base = css_block(text, "ChatbooksWindowImproved .search-input")
    focus = css_block(text, "ChatbooksWindowImproved .search-input:focus")
    assert_stable_solid_border_geometry(base, focus)
    assert_thin_input_focus(focus)
    assert "background: $ds-input-focus-bg;" in focus

    inline_text = CHATBOOKS_WINDOW_IMPROVED.read_text(encoding="utf-8")
    inline_base = css_block(inline_text, ".search-input")
    inline_focus = css_block(inline_text, ".search-input:focus")
    assert_stable_solid_border_geometry(inline_base, inline_focus)
    assert_thin_inline_input_focus(inline_focus)


def test_evals_sample_browser_selected_row_uses_readable_inline_contract():
    text = SAMPLE_BROWSER_DIALOG.read_text(encoding="utf-8")
    assert_readable_inline_selected_state_contract(css_block(text, ".sample-row.selected"))


def test_evals_sample_browser_selected_row_children_show_inline_selected_cue():
    text = SAMPLE_BROWSER_DIALOG.read_text(encoding="utf-8")
    for selector in (
        ".sample-row.selected .sample-id",
        ".sample-row.selected .sample-type",
        ".sample-row.selected .sample-preview",
    ):
        block = css_block(text, selector)
        assert "$accent" not in block
        assert "$primary" not in block
        assert "color: $text;" in block
        assert "text-style: bold underline;" in block


def test_evals_navigation_card_focus_is_non_obscuring_and_ordered_after_type_borders():
    text = EVAL_NAV_SCREEN.read_text(encoding="utf-8")
    focus = css_block(text, ".nav-card:focus")
    assert_custom_widget_focus_contract(focus)
    assert css_blocks(text, ".nav-card.quick-test") == []
    assert css_blocks(text, ".nav-card.batch") == []

    for selector in (".nav-card.quick_test", ".nav-card.batch_eval"):
        assert css_blocks(text, selector)
        assert text.index(selector) < text.index(".nav-card:focus")

    for selector in (
        ".nav-card:focus .card-icon",
        ".nav-card:focus .card-title",
        ".nav-card:focus .card-description",
        ".nav-card:focus .card-shortcut",
    ):
        assert css_blocks(text, selector) == []


def test_tamagotchi_focus_uses_non_obscuring_custom_widget_contract():
    from tldw_chatbook.Widgets.Tamagotchi.base_tamagotchi import BaseTamagotchi

    text = BaseTamagotchi.DEFAULT_CSS
    base = css_block(text, "BaseTamagotchi")
    focus = css_block(text, "BaseTamagotchi:focus")
    assert "border: round" in base
    assert "background: $panel;" in base
    assert "border: round $surface-lighten-1;" in base
    assert_custom_widget_focus_contract(focus)

    assert text.index("BaseTamagotchi.dead") < text.index("BaseTamagotchi:focus")
    assert text.index("BaseTamagotchi:focus") < text.index("BaseTamagotchi.compact")


def test_compact_custom_buttons_use_readable_focus_cues():
    for path, selector in (
        (EMOJI_PICKER, "EmojiButton.emoji_button:focus"),
        (ENHANCED_FILE_PICKER, "PathBreadcrumbs .breadcrumb-button:focus"),
    ):
        block = css_block(path.read_text(encoding="utf-8"), selector)
        assert_non_obscuring_focus(block)
        assert "$primary" not in block
        assert "$accent" not in block
        assert "background: $ds-focus-bg;" in block
        assert "color: $ds-focus-fg;" in block


def test_huggingface_model_card_selected_file_row_is_readable():
    text = MODEL_CARD_VIEWER.read_text(encoding="utf-8")
    for selector in (
        "ModelCardViewer .file-item.selected",
        "ModelCardViewer .file-item.selected:hover",
    ):
        assert_readable_inline_selected_state_contract(css_block(text, selector))


def test_notes_toolbar_active_toggle_uses_readable_active_contract():
    text = NOTES_TOOLBAR.read_text(encoding="utf-8")
    assert_readable_inline_selected_state_contract(
        css_block(text, "NotesToolbar Button.toggle.active")
    )


def test_notes_editor_focus_uses_stable_thin_input_contract():
    from tldw_chatbook.Widgets.Note_Widgets.notes_editor_widget import NotesEditorWidget

    text = NotesEditorWidget.DEFAULT_CSS
    base = css_block(text, "NotesEditorWidget")
    focus = css_block(text, "NotesEditorWidget:focus")
    assert_stable_solid_border_geometry(base, focus)
    assert "outline: heavy" not in focus
    assert "border: solid $surface-lighten-1;" in focus
    assert "border-bottom: solid $surface-lighten-1;" in focus
    assert "background: $surface;" in focus
    assert "color: $text;" in focus
    assert "$primary" not in focus
    assert "$accent" not in focus
    assert "$error" not in focus
    assert "$warning" not in focus


def test_notes_sync_progress_active_states_are_readable():
    for path, selector in (
        (NOTES_SYNC, "SyncProgressWidget.active"),
        (NOTES_SYNC_IMPROVED, "SyncProgressSection.active"),
    ):
        block = css_block(path.read_text(encoding="utf-8"), selector)
        assert "display: block;" in block
        assert "outline: heavy" not in block
        assert "$primary" not in block
        assert "$accent" not in block
        assert "$warning" not in block
        assert "$error" not in block
        assert "background: $surface;" in block
        assert "border: solid $surface-lighten-1;" in block
        assert "color: $text;" in block


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


def test_sidebar_section_focus_within_uses_non_semantic_container_cue():
    for path in (SIDEBARS, BUNDLE):
        blocks = css_blocks(
            path.read_text(encoding="utf-8"),
            ".sidebar-section-collapsible:focus-within",
        )
        assert blocks, "Missing CSS block for .sidebar-section-collapsible:focus-within"
        if path == SIDEBARS:
            assert len(blocks) == 1
        block = blocks[-1]
        assert "outline: heavy" not in block
        assert "border: thick" not in block
        assert "$primary" not in block
        assert "$accent" not in block
        assert "$warning" not in block
        assert "$error" not in block
        assert "border: round $ds-focus-accent;" in block
        assert "background: $ds-focus-bg;" in block


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
