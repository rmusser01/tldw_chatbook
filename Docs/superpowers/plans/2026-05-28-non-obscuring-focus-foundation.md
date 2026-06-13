# Non-Obscuring Focus Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build PR 1 from the accepted non-obscuring focus/selection design: shared tokens, global focus/button/input foundations, top-nav parity, screenshot-offender fixes, audit docs, tests, generated CSS, and visual evidence.

**Architecture:** Keep the change in the existing TCSS design system and the few inline Python `DEFAULT_CSS` surfaces that own app-shell state. The global fallback remains visible but non-obscuring, while control-specific rules add a second quiet cue. Full Console/Library and legacy feature-screen audits remain deferred to later PRs; this plan fixes only the visible PR 1 selectors and records the rest in audit inventory.

**Tech Stack:** Python 3.11+, Textual TCSS, inline `DEFAULT_CSS`, pytest, generated `tldw_cli_modular.tcss`, textual-web/CDP visual QA.

---

## Source Spec

- Design: `Docs/superpowers/specs/2026-05-28-non-obscuring-focus-selection-design.md`
- QA runbook: `Docs/superpowers/qa/product-maturity/screen-qa/textual-web-cdp-debugging.md`

## Scope Check

This plan intentionally covers PR 1 only:

- In scope: shared/global focus rules, shared inputs, top navigation, Library source-action screenshot case, Console transcript/action/composer screenshot cases, Settings reference preservation, generated CSS, audit inventory.
- Out of scope: full Console and Library screen review, Media/Evals inline widget migrations, Chat/Search/RAG/Embeddings/Coding legacy overrides, all data-table/tree migrations. These must be inventoried and assigned to later PRs.

## File Structure

- Modify `tldw_chatbook/css/core/_variables.tcss`: add neutral/primary focus tokens and input focus tokens.
- Modify `tldw_chatbook/css/core/_reset.tcss`: replace heavy global focus fallback with visible non-obscuring fallback.
- Modify `tldw_chatbook/css/components/_buttons.tcss`: update global `Button:focus` and `Button:hover:focus` to two-cue non-obscuring focus.
- Modify `tldw_chatbook/css/components/_forms.tcss`: update native `Input:focus`, `TextArea:focus`, `Select:focus`, `.form-input:focus`, and `.form-textarea:focus` to thin border plus subtle bottom emphasis.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss`: update Console/Library PR 1 selectors and preserve Settings reference styles.
- Modify `tldw_chatbook/UI/Navigation/main_navigation.py`: update `.nav-button:focus`, `.nav-button.is-active`, and focused active top-nav state.
- Modify `tldw_chatbook/css/tldw_cli_modular.tcss`: regenerate with `tldw_chatbook/css/build_css.py`.
- Modify `Tests/UI/test_focus_accessibility.py`: replace heavy-outline expectations with readable focus contract expectations.
- Modify `Tests/UI/test_master_shell_design_system_contract.py`: extend design-system contract for non-obscuring focus tokens and Library/Console selectors.
- Modify `Tests/UI/test_console_persistent_rails.py`: update stale composer focus border assertion that currently expects `border: heavy $ds-action-focus;`.
- Create `Tests/UI/test_non_obscuring_focus_contract.py`: source-level contract tests for TCSS and inline Python CSS.
- Create `Docs/superpowers/qa/non-obscuring-focus-selection/audit-inventory.md`: app-wide selector inventory with migration status.
- Create `Docs/superpowers/qa/non-obscuring-focus-selection/widget-exception-matrix.md`: native/semi-native widget focus fallback matrix.
- Create `Docs/superpowers/qa/non-obscuring-focus-selection/pr1-visual-qa.md`: final visual QA evidence paths and verification notes.
- Add final PNG screenshots under `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/` after rendered visual QA.

---

### Task 1: Create PR 1 Audit And Exception Docs

**Files:**
- Create: `Docs/superpowers/qa/non-obscuring-focus-selection/audit-inventory.md`
- Create: `Docs/superpowers/qa/non-obscuring-focus-selection/widget-exception-matrix.md`

- [ ] **Step 1: Create the audit inventory skeleton**

Use this table shape:

```markdown
# Non-Obscuring Focus Selection Audit Inventory

Date: 2026-05-28
Scope: PR 1 foundation plus app-wide inventory

| Selector | Owner | Screen/widget | Type | Current risk | Target state | PR/status | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `*:focus` | `tldw_chatbook/css/core/_reset.tcss` | Global | fallback | heavy outline can obscure dense labels | visible non-obscuring fallback | PR 1 | source contract |
```

- [ ] **Step 2: Add required PR 1 rows**

Include at least these rows:

```markdown
| `Button:focus` | `tldw_chatbook/css/components/_buttons.tcss` | Global buttons | button | saturated fill and heavy outline | readable text, underline, subtle background, no heavy outline | PR 1 | source contract |
| `Button:hover:focus` | `tldw_chatbook/css/components/_buttons.tcss` | Global buttons | button | stronger hover fill plus heavy outline | readable focused hover without heavy outline | PR 1 | source contract |
| `.nav-button:focus` | `tldw_chatbook/UI/Navigation/main_navigation.py` | Top navigation | nav tab | focus lacks underline contract | underline plus quiet secondary cue | PR 1 | Python source contract |
| `.nav-button.is-active` | `tldw_chatbook/UI/Navigation/main_navigation.py` | Top navigation | nav tab | selected fill can dominate | subtle selected fill, readable label | PR 1 | Python source contract |
| `.nav-button.is-active:focus` | `tldw_chatbook/UI/Navigation/main_navigation.py` | Top navigation | nav tab | combined state unspecified | selected remains readable, focus adds underline | PR 1 | Python source contract |
| `.library-source-action` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Library source browser | button | one-row action can be obscured by global focus | readable text, underline, subtle background | PR 1 visible offender | source contract and PNG |
| `.console-transcript-action-button:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Console transcript | button | action fill can obscure compact labels | readable text, underline, subtle background | PR 1 visible offender | source contract and PNG |
| `#console-native-composer.console-composer-focused` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Console composer | input container | heavy orange frame reads as warning/selection | thin border plus subtle emphasis | PR 1 visible offender | source contract and PNG |
| `#settings-shell Button:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Settings | button | reference pattern must stay readable | preserve underline plus raised background | PR 1 reference | existing Settings tests |
| `.settings-action-row Button:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Settings action rows | button | reference pattern must stay readable | preserve underline plus raised background | PR 1 reference | existing Settings tests |
| `#settings-impact-pane Button:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Settings impact pane | button | reference pattern must stay readable | preserve underline plus raised background | PR 1 reference | existing Settings tests |
| `Input:focus` | `tldw_chatbook/css/components/_forms.tcss` | Native inputs | input | full-field focus can read as warning or obscure compact content | thin border plus bottom emphasis | PR 1 | source contract and mounted check if feasible |
| `TextArea:focus` | `tldw_chatbook/css/components/_forms.tcss` | Native text areas | input | full-field focus can read as warning or obscure compact content | thin border plus bottom emphasis | PR 1 | source contract and mounted check if feasible |
| `Select:focus` | `tldw_chatbook/css/components/_forms.tcss` | Native selects | select | native focus treatment may fall back to heavy outline | thin border plus bottom emphasis where supported | PR 1 | source contract and visual/mounted exception note |
```

- [ ] **Step 3: Add deferred rows**

Add rows for known deferred surfaces with `PR/status` set to `Deferred`:

```markdown
| `NavigationButton:focus` | `tldw_chatbook/Widgets/base_components.py` | shared widget | button/nav | inline focus state needs review | two-cue non-obscuring focus | Deferred PR 3 | not yet migrated |
| `MediaNavigationPanel .media-type-button.active` | `tldw_chatbook/Widgets/Media/media_navigation_panel.py` | Media | active button | active fill needs review | active plus focus contract | Deferred PR 4+ | not yet migrated |
| `MediaListPanel .media-item.selected` | `tldw_chatbook/Widgets/Media/media_list_panel.py` | Media | selected row | selected row needs review | readable selected row | Deferred PR 4+ | not yet migrated |
| `.sample-row.selected` | `tldw_chatbook/Widgets/Evals/sample_browser_dialog.py` | Evals dialog | selected row | selected row needs review | readable selected row | Deferred PR 4+ | not yet migrated |
| `.chat-sidebar-toggle-button:focus` | `tldw_chatbook/css/features/_chat.tcss` | Chat | button | heavy outline feature override | two-cue non-obscuring focus | Deferred PR 4+ | not yet migrated |
| `.coding-nav-button:focus` | `tldw_chatbook/css/features/_coding.tcss` | Coding | nav button | local focus override needs review | two-cue non-obscuring focus | Deferred PR 4+ | not yet migrated |
| `.embeddings-nav-button:focus` | `tldw_chatbook/css/features/_embeddings.tcss` | Embeddings | nav button | local focus override needs review | two-cue non-obscuring focus | Deferred PR 4+ | not yet migrated |
| `Input:focus, Select:focus, TextArea:focus, Button:focus, Checkbox:focus` | `tldw_chatbook/css/features/_embeddings.tcss` | Embeddings | mixed controls | heavy outline feature override | widget-specific non-obscuring focus | Deferred PR 4+ | not yet migrated |
| `.sidebar *:focus` | `tldw_chatbook/css/layout/_sidebars.tcss` | Legacy sidebars | fallback | heavy outline sidebar override | visible non-obscuring fallback | Deferred PR 4+ | not yet migrated |
| `.sidebar Button:focus` | `tldw_chatbook/css/layout/_sidebars.tcss` | Legacy sidebars | button | heavy outline sidebar override | two-cue non-obscuring focus | Deferred PR 4+ | not yet migrated |
| `.sidebar Select:focus` | `tldw_chatbook/css/layout/_sidebars.tcss` | Legacy sidebars | select | heavy outline sidebar override | thin non-obscuring focus | Deferred PR 4+ | not yet migrated |
| `.sidebar Input:focus` | `tldw_chatbook/css/layout/_sidebars.tcss` | Legacy sidebars | input | heavy outline sidebar override | thin non-obscuring focus | Deferred PR 4+ | not yet migrated |
```

- [ ] **Step 4: Inventory every remaining local override match**

Run:

```bash
rg -n ":focus|\\.active|\\.selected|is-active|outline|reverse|background: \\$accent|background: \\$primary|background: \\$warning|background: \\$error" tldw_chatbook/css/features tldw_chatbook/css/layout tldw_chatbook/css/components tldw_chatbook/Widgets tldw_chatbook/UI/Navigation/main_navigation.py -g '*.tcss' -g '*.py'
```

For every match not migrated in PR 1, add or update an audit row with `PR/status` set to `Deferred PR 3` or `Deferred PR 4+`. This is required so the final regression scan in Task 7 has an owner for every remaining heavy outline, reverse text, strong selected fill, or semantic-color focus rule.

- [ ] **Step 5: Create the widget exception matrix**

Use this table shape:

```markdown
# Non-Obscuring Focus Widget Exception Matrix

Date: 2026-05-28

| Widget/control | Supported cues | PR 1 behavior | Exception/fallback | Test |
| --- | --- | --- | --- | --- |
| `Button` | background, foreground, text-style, outline | underline plus subtle background, no heavy outline | none | `Tests/UI/test_non_obscuring_focus_contract.py` |
| `Input` | border, background, foreground, outline | thin border plus subtle bottom emphasis | no full-field heavy frame | `Tests/UI/test_focus_accessibility.py` and `Tests/UI/test_non_obscuring_focus_contract.py` |
| `TextArea` | border, background, foreground, outline | thin border plus subtle bottom emphasis | no full-field heavy frame | `Tests/UI/test_focus_accessibility.py` and `Tests/UI/test_non_obscuring_focus_contract.py` |
| `Select` | border/background varies by Textual widget | thin border plus subtle bottom emphasis where supported | if Textual native styling resists bottom emphasis, document the verified fallback in this matrix and visual QA | `Tests/UI/test_focus_accessibility.py` and visual/mounted exception note |
| `DataTable` | cursor row and selected row styling | deferred | row highlight fallback allowed if readable | deferred |
| `Tree` | cursor/selected row styling | deferred | row highlight fallback allowed if readable | deferred |
| `ListView` | item selected/highlight classes | deferred | row highlight fallback allowed if readable | deferred |
| `SelectionList` | selected/highlight option styling | deferred | row highlight fallback allowed if readable | deferred |
| `Tabs` | tab active/focus classes | top nav only in PR 1 | native `Tabs` deferred | deferred |
| Custom button/list rows | background, marker text, classes, text-style | PR 1 for named Console/Library rows only | each custom row needs explicit selector audit | source contract or mounted test |
```

- [ ] **Step 6: Commit the docs if this task is implemented separately**

```bash
git add Docs/superpowers/qa/non-obscuring-focus-selection/audit-inventory.md Docs/superpowers/qa/non-obscuring-focus-selection/widget-exception-matrix.md
git commit -m "docs: add focus selection audit inventory"
```

---

### Task 2: Add Failing Contract Tests For The New Focus Rules

**Files:**
- Create: `Tests/UI/test_non_obscuring_focus_contract.py`
- Modify: `Tests/UI/test_focus_accessibility.py`
- Modify: `Tests/UI/test_master_shell_design_system_contract.py`
- Modify: `Tests/UI/test_console_persistent_rails.py`

- [ ] **Step 1: Create the new contract test module**

Add helpers like:

```python
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
    """Return a CSS rule body whose selector list contains selector.

    Do not use a prefix regex such as "selector {" here. Several repo
    styles use grouped selectors, and the target selector may appear before a
    comma rather than immediately before the opening brace.
    """
    rule_pattern = re.compile(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", re.DOTALL)
    for match in rule_pattern.finditer(text):
        selectors = [item.strip() for item in match.group("selectors").split(",")]
        if selector in selectors:
            return match.group("body")
    assert False, f"Missing CSS block for {selector}"


def assert_non_obscuring_focus(block: str) -> None:
    assert "outline: heavy" not in block
    assert "reverse" not in block
    assert "text-style: bold underline;" in block
```

- [ ] **Step 2: Add RED tests for tokens and global fallback**

```python
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
```

- [ ] **Step 3: Add RED tests for global button focus**

```python
def test_global_button_focus_uses_two_non_obscuring_cues():
    text = BUTTONS.read_text(encoding="utf-8")
    for selector in ("Button:focus", "Button:hover:focus"):
        block = css_block(text, selector)
        assert_non_obscuring_focus(block)
        assert "$ds-focus-bg" in block or "$ds-surface-raised" in block
```

- [ ] **Step 4: Add RED tests for shared inputs**

```python
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
```

- [ ] **Step 5: Add RED tests for Console and Library visible offenders**

```python
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
```

- [ ] **Step 6: Add RED tests for inline top navigation CSS**

```python
def test_top_navigation_inline_focus_uses_hybrid_contract():
    text = NAV.read_text(encoding="utf-8")
    focus = css_block(text, ".nav-button:focus")
    active = css_block(text, ".nav-button.is-active")
    active_focus = css_block(text, ".nav-button.is-active:focus")
    assert_non_obscuring_focus(focus)
    assert "outline: heavy" not in active
    assert_non_obscuring_focus(active_focus)
```

- [ ] **Step 7: Update existing heavy-outline accessibility expectations**

In `Tests/UI/test_focus_accessibility.py`, rename the heavy-outline tests to reflect visible non-obscuring focus. Replace assertions like:

```python
assert "outline: heavy $accent" in css_content
assert "outline: heavy" in css_content or "outline: solid" in css_content
```

with expectations that require visible cues but reject heavy outlines:

```python
assert "*:focus" in css_content
assert "text-style: bold underline" in css_content or "outline: solid" in css_content
assert "outline: none !important" not in css_content
```

Do not assert that the entire generated stylesheet lacks `outline: heavy`; PR 1 intentionally defers legacy feature/layout overrides. Instead, inspect the generated blocks for PR 1 migrated selectors only, or rely on `Tests/UI/test_non_obscuring_focus_contract.py` for source-level selector checks. Reuse the selector-list-aware `css_block()` helper for generated CSS too so grouped selectors are not missed.

- [ ] **Step 8: Update existing contract tests that conflict with the new design**

In `Tests/UI/test_master_shell_design_system_contract.py`, update `test_library_mode_chip_focus_keeps_active_label_readable` to expect `text-style: bold underline;` for focus and active focus.

In `Tests/UI/test_console_persistent_rails.py`, replace the broad expectation:

```python
assert "border: heavy $ds-action-focus;" in css
```

with a scoped assertion that the composer no longer uses a heavy focus border:

```python
assert "#console-native-composer.console-composer-focused" in css
assert "border: heavy $ds-action-focus;" not in css
```

- [ ] **Step 9: Run the new/updated tests and confirm RED**

Run:

```bash
PYTHONPATH=. pytest Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_focus_accessibility.py Tests/UI/test_master_shell_design_system_contract.py::test_library_mode_chip_focus_keeps_active_label_readable Tests/UI/test_console_persistent_rails.py -q
```

Expected: failures point to missing tokens, heavy outline rules, missing top-nav active-focus block, and stale Console/Library focus styles.

---

### Task 3: Implement Shared Tokens, Global Fallback, Buttons, And Inputs

**Files:**
- Modify: `tldw_chatbook/css/core/_variables.tcss`
- Modify: `tldw_chatbook/css/core/_reset.tcss`
- Modify: `tldw_chatbook/css/components/_buttons.tcss`
- Modify: `tldw_chatbook/css/components/_forms.tcss`

- [ ] **Step 1: Add focus tokens**

Add near the existing `$ds-*` tokens in `_variables.tcss`:

```css
$ds-focus-fg: $ds-text-primary;
$ds-focus-bg: $ds-surface-raised;
$ds-focus-accent: $primary;
$ds-input-focus-border: $primary;
$ds-input-focus-bg: $ds-surface-raised;
$ds-input-focus-accent: $primary;
```

- [ ] **Step 2: Replace global heavy focus fallback**

In `_reset.tcss`, replace the heavy rules with a visible conservative fallback:

```css
/* Visible non-obscuring keyboard focus fallback. */
*:focus {
    outline: solid $ds-focus-accent;
}

.dark *:focus {
    outline: solid $ds-focus-accent;
}

.light *:focus {
    outline: solid $ds-focus-accent;
}
```

Do not use `outline: none` globally.

- [ ] **Step 3: Update global button focus**

In `_buttons.tcss`, use two cues and remove heavy outline:

```css
Button:focus {
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold underline;
    outline: none;
}

Button:hover:focus {
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold underline;
    outline: none;
}
```

- [ ] **Step 4: Update shared form input focus**

In `_forms.tcss`, update native form widgets and form utility classes to use non-semantic focus tokens:

```css
Input:focus,
TextArea:focus,
Select:focus {
    border: solid $ds-input-focus-border;
    border-bottom: solid $ds-input-focus-accent;
    background: $ds-input-focus-bg;
    color: $ds-text-primary;
}

.form-input:focus {
    border: solid $ds-input-focus-border;
    border-bottom: solid $ds-input-focus-accent;
    background: $ds-input-focus-bg;
    color: $ds-text-primary;
}

.form-textarea:focus {
    border: solid $ds-input-focus-border;
    border-bottom: solid $ds-input-focus-accent;
    background: $ds-input-focus-bg;
    color: $ds-text-primary;
}
```

Run `python tldw_chatbook/css/build_css.py` after this edit. If Textual rejects or visually ignores `border-bottom` for a specific native widget, do not silently drop the cue: choose a verified non-obscuring alternative, update the tests for that widget only, and record the exception in `Docs/superpowers/qa/non-obscuring-focus-selection/widget-exception-matrix.md`.

- [ ] **Step 5: Run focused source tests**

Run:

```bash
PYTHONPATH=. pytest Tests/UI/test_non_obscuring_focus_contract.py::test_focus_tokens_are_defined_and_not_semantic_warning_or_error Tests/UI/test_non_obscuring_focus_contract.py::test_global_focus_fallback_is_visible_but_not_heavy Tests/UI/test_non_obscuring_focus_contract.py::test_global_button_focus_uses_two_non_obscuring_cues Tests/UI/test_non_obscuring_focus_contract.py::test_shared_form_and_native_inputs_use_thin_non_semantic_focus -q
```

Expected: these tests pass. Remaining tests for Console, Library, top nav, and generated CSS may still fail.

- [ ] **Step 6: Commit if implementing task-by-task**

```bash
git add tldw_chatbook/css/core/_variables.tcss tldw_chatbook/css/core/_reset.tcss tldw_chatbook/css/components/_buttons.tcss tldw_chatbook/css/components/_forms.tcss Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_focus_accessibility.py
git commit -m "style: add non-obscuring focus foundation"
```

---

### Task 4: Implement Console, Library, Settings Reference, And Top-Nav PR 1 Styles

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `Tests/UI/test_master_shell_design_system_contract.py`
- Modify: `Tests/UI/test_console_persistent_rails.py`

- [ ] **Step 1: Update Console composer focus**

Change `#console-native-composer.console-composer-focused` from a heavy action border to thin input treatment:

```css
#console-native-composer.console-composer-focused {
    border: solid $ds-input-focus-border;
    border-bottom: solid $ds-input-focus-accent;
    background: $ds-input-focus-bg;
}
```

Keep `#console-native-composer.console-composer-has-draft` as the semantic ready state.

- [ ] **Step 2: Update Console transcript focus and actions**

Update transcript focus and action-button focus:

```css
#console-native-transcript:focus {
    border: solid $ds-focus-accent;
}

.console-transcript-action-button:focus {
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold underline;
    outline: none;
}
```

- [ ] **Step 3: Update selected Console message combined state**

Keep selected message visible without escalating focus fill:

```css
.console-transcript-message-selected {
    background: $ds-surface-raised;
    color: $ds-text-primary;
    text-style: bold underline;
    border: solid $ds-focus-accent;
}
```

- [ ] **Step 4: Add Library source-action focus**

Add immediately after `.library-source-action`:

```css
.library-source-action:focus {
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold underline;
    outline: none;
}
```

- [ ] **Step 5: Update Library mode chip combined focus**

Change `.library-mode-chip:focus` and `.library-mode-chip.is-active:focus` to include underline:

```css
.library-mode-chip:focus {
    outline: none;
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold underline;
}

.library-mode-chip.is-active:focus {
    outline: none;
    border: solid $ds-focus-accent;
    background: $primary-darken-1;
    color: $ds-text-primary;
    text-style: bold underline;
}
```

- [ ] **Step 6: Update Settings compact input focus**

The Settings button focus already matches the reference pattern. Update the input focus at `.settings-compact-input:focus`:

```css
.settings-compact-input:focus {
    border: solid $ds-input-focus-border;
    border-bottom: solid $ds-input-focus-accent;
    background: $ds-input-focus-bg;
    color: $ds-text-primary;
    outline: none;
}
```

- [ ] **Step 7: Update inline top navigation CSS**

In `MainNavigationBar.DEFAULT_CSS`, update focus and active focus:

```css
.nav-button:focus {
    background: $surface;
    border: solid $primary;
    text-style: bold underline;
    color: $text;
    outline: none;
}

.nav-button.is-active {
    background: $primary-darken-1;
    border: solid $primary;
    text-style: bold;
    color: $text;
}

.nav-button.is-active:focus {
    background: $primary-darken-1;
    border: solid $primary;
    text-style: bold underline;
    color: $text;
    outline: none;
}
```

- [ ] **Step 8: Run focused contract tests**

Run:

```bash
PYTHONPATH=. pytest Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_master_shell_design_system_contract.py::test_library_mode_chip_focus_keeps_active_label_readable Tests/UI/test_console_persistent_rails.py -q
```

Expected: contract tests pass after the style updates.

- [ ] **Step 9: Commit if implementing task-by-task**

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/UI/Navigation/main_navigation.py Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_master_shell_design_system_contract.py Tests/UI/test_console_persistent_rails.py
git commit -m "style: align console library and nav focus states"
```

---

### Task 5: Regenerate The Modular Stylesheet And Verify Source/Generated Parity

**Files:**
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_master_shell_design_system_contract.py`
- Test: `Tests/UI/test_focus_accessibility.py`

- [ ] **Step 1: Regenerate CSS**

Run:

```bash
python tldw_chatbook/css/build_css.py
```

Expected: command completes. Existing missing-module warnings are acceptable only if they match the current baseline from `build_css.py`.

- [ ] **Step 2: Verify generated CSS contains the new contract**

Run:

```bash
rg -n "ds-focus|console-composer-focused|console-transcript-action-button:focus|library-source-action:focus|Button:focus|outline: heavy" tldw_chatbook/css/tldw_cli_modular.tcss
```

Expected:

- New `$ds-focus-*` tokens are present.
- PR 1 selectors are present.
- `Button:focus` in generated CSS no longer contains `outline: heavy`.
- Any remaining `outline: heavy` hits outside PR 1 migrated selectors are explicitly deferred with a tracked audit row. Do not stop at the raw `rg` output; reconcile each remaining hit against `Docs/superpowers/qa/non-obscuring-focus-selection/audit-inventory.md`.

- [ ] **Step 3: Run generated stylesheet tests**

Run:

```bash
PYTHONPATH=. pytest Tests/UI/test_focus_accessibility.py Tests/UI/test_master_shell_design_system_contract.py -q
```

Expected: pass. The accessibility tests must check the PR 1 migrated selectors or visible-focus fallback without requiring the entire generated stylesheet to be free of deferred heavy-outline overrides.

- [ ] **Step 4: Commit if implementing task-by-task**

```bash
git add tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_focus_accessibility.py Tests/UI/test_master_shell_design_system_contract.py
git commit -m "test: verify generated non-obscuring focus styles"
```

---

### Task 6: Add Mounted Checks And Rendered Visual QA Evidence

**Files:**
- Modify or create tests as needed under `Tests/UI/`
- Create: `Docs/superpowers/qa/non-obscuring-focus-selection/pr1-visual-qa.md`
- Add screenshots: `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/*.png`

- [ ] **Step 1: Add mounted checks for visible PR 1 targets**

Add mounted assertions in existing UI tests or the new contract module. Use existing harnesses from `Tests/UI/test_destination_visual_parity_correction.py` and `Tests/UI/test_console_native_transcript.py`.

Target behaviors:

```python
# Library
# Focus "#library-open-media" or another `.library-source-action` and assert it has focus.

# Console
# Focus "#console-native-composer" and assert it has class "console-composer-focused".
# Select a transcript message and focus a `.console-transcript-action-button`.
# Top nav
# Focus "#nav-console" and assert it remains readable and active/focus classes can coexist.
```

- [ ] **Step 2: Run the mounted checks**

Run the narrow tests that contain the new mounted assertions. Example:

```bash
PYTHONPATH=. pytest Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_console_native_transcript.py Tests/UI/test_destination_visual_parity_correction.py -q
```

Expected: pass or expose only unrelated baseline failures. If unrelated baseline failures appear, document exact failing tests in `pr1-visual-qa.md`.

- [ ] **Step 3: Capture rendered PNG evidence**

Use the runbook:

```markdown
Docs/superpowers/qa/product-maturity/screen-qa/textual-web-cdp-debugging.md
```

Capture at least:

- `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/pr1-library-focused-source-action.png`
- `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/pr1-console-focused-action-button.png`
- `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/pr1-console-focused-composer.png`
- `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/pr1-top-nav-focused-active.png`
- `Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/pr1-settings-selected-category-reference.png`

Use actual textual-web/CDP or an actual terminal screenshot. SVG export is not sufficient for visual approval.

- [ ] **Step 4: Verify PNG files**

Run:

```bash
file Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/*.png
```

Expected: every listed file reports PNG image data.

- [ ] **Step 5: Write visual QA notes**

Create `pr1-visual-qa.md` with:

```markdown
# Non-Obscuring Focus Selection PR 1 Visual QA

Date: 2026-05-28

## Evidence

| Case | Path | Method | Result |
| --- | --- | --- | --- |
| Library focused source action | `visual-captures/pr1-library-focused-source-action.png` | textual-web/CDP | Pending user approval |
| Console focused transcript action | `visual-captures/pr1-console-focused-action-button.png` | textual-web/CDP | Pending user approval |
| Console focused composer | `visual-captures/pr1-console-focused-composer.png` | textual-web/CDP | Pending user approval |
| Top nav focused active | `visual-captures/pr1-top-nav-focused-active.png` | textual-web/CDP | Pending user approval |
| Settings selected category reference | `visual-captures/pr1-settings-selected-category-reference.png` | textual-web/CDP | Pending user approval |

## Verification Commands

- `PYTHONPATH=. pytest ...`
- `file Docs/superpowers/qa/non-obscuring-focus-selection/visual-captures/*.png`
```

- [ ] **Step 6: Commit if implementing task-by-task**

```bash
git add Tests/UI Docs/superpowers/qa/non-obscuring-focus-selection
git commit -m "test: add non-obscuring focus visual evidence"
```

---

### Task 7: Final PR 1 Verification And Handoff

**Files:**
- All files touched in prior tasks.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
PYTHONPATH=. pytest Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_focus_accessibility.py Tests/UI/test_master_shell_design_system_contract.py Tests/UI/test_console_persistent_rails.py Tests/UI/test_console_native_transcript.py Tests/UI/test_destination_visual_parity_correction.py -q
```

Expected: pass, or document unrelated baseline failures with exact test names and failure messages.

- [ ] **Step 2: Run Settings reference tests**

Run:

```bash
PYTHONPATH=. pytest Tests/UI/test_settings_configuration_hub.py -q
```

Expected: pass, especially the tests asserting Settings focused buttons use readable underline focus and avoid reverse/heavy-obscuring behavior.

- [ ] **Step 3: Run CSS build one final time**

Run:

```bash
python tldw_chatbook/css/build_css.py
git diff -- tldw_chatbook/css/tldw_cli_modular.tcss
```

Expected: generated CSS is committed and no unexpected generated diff remains after the build.

- [ ] **Step 4: Check for accidentally broad changes**

Run:

```bash
git diff --name-status dev...HEAD
```

Expected: changed files are limited to the planned CSS, inline nav CSS, tests, generated CSS, QA docs, visual evidence, and the already-approved spec/plan docs on this branch. If this branch continues from the spec commits, compare the implementation slice against the implementation start commit as well:

```bash
git diff --name-status <implementation-start-commit>...HEAD
```

Use the second diff to catch accidental broad implementation changes without treating the approved design/plan documents as unexpected churn.

- [ ] **Step 5: Check for heavy focus regressions**

Run:

```bash
rg -n "outline: heavy|text-style: reverse|background: \\$ds-status-(warning|error|blocked|unsaved)|background: \\$warning|background: \\$error" tldw_chatbook/css tldw_chatbook/UI/Navigation/main_navigation.py Tests/UI
```

Expected: no PR 1 migrated focus selector uses heavy outline, reverse text, or semantic warning/error/blocked/unsaved background as its focus style. Deferred matches must be listed in `audit-inventory.md`.

For every remaining match, verify there is a matching row in `Docs/superpowers/qa/non-obscuring-focus-selection/audit-inventory.md` with a non-PR 1 status and owner. If a match has no row, either migrate it in PR 1 because it is in scope or add a deferred row before handoff.

- [ ] **Step 6: Update QA docs with final command output**

Record final verification commands and results in `Docs/superpowers/qa/non-obscuring-focus-selection/pr1-visual-qa.md`.

- [ ] **Step 7: Commit final updates**

```bash
git add tldw_chatbook/css tldw_chatbook/UI/Navigation/main_navigation.py Tests/UI Docs/superpowers/qa/non-obscuring-focus-selection
git commit -m "chore: verify non-obscuring focus foundation"
```

- [ ] **Step 8: Prepare PR summary**

Include:

- Foundation tokens and global fallback.
- Global button/input focus migration.
- Console/Library screenshot-offender fixes.
- Top-nav hybrid selected/focus behavior.
- Audit inventory and exception matrix.
- Tests and visual QA evidence paths.
- Deferred screen groups for later PRs.

---

## Review Checklist

- [ ] Focus has at least two readable cues where the widget supports it.
- [ ] No PR 1 focus selector uses heavy outline as the primary focus cue.
- [ ] No PR 1 focus selector uses warning/error/blocked/unsaved colors unless the selector is semantic.
- [ ] Active + focus and selected + focus states remain readable.
- [ ] Unknown focusable widgets still have a visible fallback.
- [ ] Settings reference behavior remains intact.
- [ ] Generated `tldw_cli_modular.tcss` is updated.
- [ ] Actual rendered PNG evidence exists for the screenshot cases.
- [ ] Deferred work is recorded instead of silently treated as done.
