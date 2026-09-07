---
id: TASK-16811
title: DEFAULT_CSS local variable fallbacks silently override bundle tokens
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
updated_date: '2026-08-16 17:08'
labels:
  - console
  - css
  - ui-polish
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Textual resolves `$variables` per CSS *source*, not globally-with-fallback: a
`$var: value;` declared inside a widget's `DEFAULT_CSS` unconditionally
governs every use of that name within that same source, even when the real
app bundle (loaded earlier in `CSS_PATH`) defines the token. The "local
fallback so DEFAULT_CSS parses standalone" pattern is therefore an
unconditional override, not a fallback.

Verified live during the turn-file-card final review (2026-08-16): a
selected `ConsoleTurnFileCard` renders background `$surface` (`#1e1e1e`)
instead of the bundle's `$ds-focus-bg` (`#51677e`), so a selected card is
visibly duller than every other selected transcript row despite the code's
own "parity with `.console-transcript-message-selected`" comment. The
pattern is inherited precedent, not unique to the card — `NavigationButton`
(`base_components.py`) and the `EmojiPickerScreen` rules carry the same
footgun.

Likely fix shape: move token-dependent rules for these widgets into the
scoped screen CSS sources (where bundle tokens resolve), keeping only
token-free structural rules in `DEFAULT_CSS` — then regenerate the bundle
(never hand-edit it). Assert the resolved background color in a
real-CSS-stack test, since a class-toggle assertion alone cannot catch
this (the class toggles correctly today; the color is what's wrong).

**Update (PR #1728, 2026-08-16):** the ConsoleTurnFileCard instance was
fixed on that PR after Qodo re-raised it — selection rules moved into
`css/components/_agentic_terminal.tcss`, bundle regenerated, and a
resolved-color parity test added
(`test_selected_card_uses_the_bundles_focus_background`). AC #1 is
therefore already satisfied; the remaining scope is the repo-wide
pattern audit (AC #2) and its bundle/test hygiene (AC #3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A selected ConsoleTurnFileCard renders the same focus background token as a selected plain transcript message under the real app CSS bundle, asserted on resolved color in a real-CSS-stack test
- [x] #2 Widgets audited for the same local-`$var`-in-DEFAULT_CSS override pattern (at minimum NavigationButton and EmojiPickerScreen) are either fixed the same way or explicitly recorded as intentional with a comment
- [x] #3 The CSS bundle is regenerated, never hand-edited, and existing geometry/CSS tests still pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Sweep all class-level CSS blocks for local $ds declarations\n2. Classify each: parse-fallback footgun vs deliberate palette\n3. Move token-dependent rules to bundle modules; drop fallbacks; regenerate\n4. Update the non-obscuring-focus contract tests to the new rule homes\n5. Resolved-color parity tests, verified red at the pre-fix tree
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Sweep found 8 files with local $ds declarations in class-level CSS. Seven carried the 'local fallback so it parses without the bundle' comment — a real footgun: written when $ds-focus-bg still equaled $surface (the stale MCPScreen comment in widget_defaults_self.tcss records that era), they silently became divergences when the token moved to #51677e. Fixed six by relocating the token-dependent rules into bundle modules (_navigation.tcss: NavigationButton active/focus; _widgets.tcss: EmojiButton focus, PathBreadcrumbs breadcrumb focus, BaseTamagotchi focus, ChatbooksWindowImproved search-input focus; _wizards.tcss: WizardProgress active steps) and dropping the local declarations; TreeNode's .tree-node-selected copy was dead weight (bundle's _code_repo.tcss copy always won) and was deleted. prompt_block_editor.py is a DELIBERATE widget palette (own $ds-surface-field token, $accent-based focus) — recorded intentional with a comment per AC2. Bundle + generated sheets regenerated via build_css (never hand-edited). Contract suite test_non_obscuring_focus_contract.py updated: the six affected contracts now read the new bundle homes AND assert the widget sources stay free of local $ds declarations. New Tests/UI/test_focus_token_parity.py pins resolved colors on the real CSS stack; the NavigationButton test was verified RED at the pre-fix tree (after fixing a mask: run_test auto-focus + the bundle's generic Button:focus painted canonical color over the shadowed .active rule — lesson filed in lessons-testing-evidence.md). Important nuance discovered: Button-subclass :focus states were already rescued at app tier by the generic Button:focus rule; the user-visible breakage was UNFOCUSED active/selected states (nav rail active item, wizard active step pre-_wizards-rules, tamagotchi focus as non-Button). Verified: 101 contract+parity, 124 file-picker/editor/repo suites, 363 modal suites, 49,419 collected. test_first_run_wizard_live_contract.py flakes order-dependently on BOTH dev and this branch (different tests per run, all pass in isolation) — pre-existing, not this change. Note: css/features/_chatbooks_improved.tcss is an ORPHANED module (absent from CSS_MODULES, diverged from the window's DEFAULT_CSS) — left untouched.
<!-- SECTION:NOTES:END -->
