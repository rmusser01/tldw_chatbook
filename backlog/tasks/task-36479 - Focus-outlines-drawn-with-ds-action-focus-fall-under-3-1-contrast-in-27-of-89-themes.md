---
id: TASK-36479
title: >-
  Focus outlines drawn with $ds-action-focus fall under 3:1 contrast in 27 of 89
  themes
status: Done
assignee:
  - '@claude'
created_date: '2026-10-23 11:02'
updated_date: '2026-10-24 16:32'
labels:
  - ui
  - accessibility
dependencies: []
references:
  - Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exists on dev today and is not fixed by Roleplay frame B2a (TASK-33910.3): the focus edge token $ds-action-focus measures under 3:1 against $surface in 27 of 89 themes, and dev draws focus outlines with it on 23 stylesheets today, among them Library's and Settings'. B2a's own items list already uses $ds-active-fg for its focus edge (owner decision 2026-10-23, "Both (Rec.)": file this follow-up and amend the Roleplay spec to name $ds-active-fg). Fix direction: $ds-active-fg (the Settings rail precedent) or a re-tuned token.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every focus edge dev draws with $ds-action-focus clears 3:1 against its surface on every measurable theme (scope per the title; focus edges dev draws with other tokens are listed in the notes, not fixed here)
- [x] #2 A contrast test pins it, so a token change that drops a theme under 3:1 fails
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory every $ds-action-focus use in the CSS sources (tcss modules and Python CSS blocks): focus edges (border*/outline* in a :focus, :focus-within or -focused rule) versus every other use; measure each against $surface, $panel and the focus rule's own fill on the 89 measurable themes.
2. Red first: a contrast test in Tests/UI/test_theme_contrast.py that discovers every focus edge in the app's CSS sources, resolves its token in its own source's variable scope, and requires 3:1 against $surface, $panel and its rule's own fill over each, on every measurable theme. Dev-debt tokens outside this task's scope (other $primary/$accent aliases) are listed with a stale-entry guard; a vacuity guard pins known edges. Run it against dev's tokens.
3. Fix with the smallest blast radius: switch the focus-edge declarations from $ds-action-focus to $ds-active-fg (the Settings rail and Roleplay list precedent); PromptBlockEditor's DEFAULT_CSS aliases the token to $accent, so its focus edges move to $text-accent. Every non-focus use keeps $ds-action-focus. Regenerate the generated sheets with css/build_css.py.
4. Update the tests that pin the old token literally, the DESIGN.md focus convention, the _variables.tcss comment and the Roleplay spec lines that cite the Library house rule.
5. Named mutations; gates: touched tests, paired run against fu-base, ruff, preflight; boot-CSS bytes and the broad-selector ratchet measured base vs head.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Focus edges that dev drew with $ds-action-focus now paint $ds-active-fg (PromptBlockEditor's paint $text-accent); a contrast test discovers and measures every focus edge on every measurable theme. Details below.

**Scope (AC#1 reworded on 2026-10-24 to match the title).** The AC said "every focus edge drawn on dev". A sweep of every CSS source found 210 focus edges; 43 paint $ds-action-focus (this task) and 152 paint other tokens that also fall under 3:1 (148 other $primary/$accent aliases: $ds-focus-accent 36 incl. the `*:focus` fallback, $ds-input-focus-accent 58, $ds-input-focus-border 10, raw $primary 7, raw $accent 37; plus $foreground 2, $tldw-active-edge 2). Those exist on dev today and are NOT fixed here; the test lists them one row per edge in Tests/UI/focus_edge_shortfall_on_dev.tsv (fix round 1; first a per-token list), the list can only shrink, and TASK-36663 tracks them.

**Decision: switch the edges, not retune the token.** $ds-action-focus has 67 other declaration lines in the sources (97 before this task; 13 non-focus edges such as `.is-active`, highlighted picker rows and modal frames; scrollbar-color-active; headings; solid and tinted backgrounds). Retuning it would recolour all of them. Switching the 26 focus-edge declarations in 9 tcss sources (33 edges) to $ds-active-fg, the Settings rail's and Roleplay list's token, changes only focus edges. PromptBlockEditor's DEFAULT_CSS aliases $ds-action-focus to $accent locally (its intentional palette), so its 4 focus declarations (10 edges) paint $text-accent, the accent's readable tint, and keep the hue.

**Measured (89 measurable themes).** tcss edges, dev: under 3:1 on $surface in 27 themes (sweet_sorbet 1.06:1), 29 adding $panel, 59 adding the fill each focus rule paints. Head: 0 themes; lowest 4.50:1 on $surface/$panel (sakura_viewing_picnic), 3.08:1 on a rule's own fill (catppuccin-frappe). PromptBlockEditor, dev: 36 / 37 / 42 themes; head: 0, lowest 3.95:1 (nord).

**AC#2.** Tests/UI/test_theme_contrast.py: `test_focus_edges_clear_non_text_contrast_on_every_theme` discovers every border/outline set by a `:focus`, `:focus-within` or `.*-focused` rule in the tcss modules and every class-level BUNDLED_CSS/BUNDLED_SCREEN_CSS/DEFAULT_CSS/CSS literal, resolves its token in its own source's scope, and requires 3:1 on $surface, $panel and its own rule's fill over each. An edge in its own fill's colour extends the fill (the theme editor's chips) and is measured on the surfaces only. The Workflows field is measured by its own reviewed test. `test_focus_edge_discovery_and_shortfall_list_are_current` pins one anchor per kind of source and keeps the dev-debt list honest. Red on dev's sources with the final logic: 64 of 89 themes, every failing edge a $ds-action-focus one. Nine named mutations go red; M6 survived its first run (the Workflows field kept the stale-list guard satisfied) and the guard was fixed. Both tests join the UI PR-gate census tail (floor 667 -> 669, dated comment).

**Limitations.** (1) Exists on dev, improved, not fully fixed: a Button whose focus rule sets no fill (Library rail rows and hub recents, Console rail carriers, the prompts modal's buttons) takes the generic Button:focus $tldw-focus-fill, and a painted probe reads $ds-active-fg at 2.33-2.88:1 against that fill in tokyo-night, atom-one-dark, catppuccin-frappe and earthy_nature (dev's $ds-action-focus: 1.26-2.25:1 there); against the surface beside it, 4.53:1 or more. The test does not model fills set by other rules. (2) Exists on dev, improved: $ds-active-fg reads 2.12:1 on litestep's $background, for any edge whose outside cell is the screen background (dev: 22 themes). (3) The commit message says "150" debt edges; the count is 152.

**Gates.** Paired run of 42 affected files against fu-base (-n 6): base 80 failed / head 75 failed; 5 head-only failures, none a regression (four pass alone 5/5 on both arms and did not recur in a traced head run; one also fails alone on base). The traced head run's 6 head-only failures: three fail alone on base, three replayed from the trace (402-568 tests) with the same result on both arms. Paired Tests/CI census checks: 220 passed both arms. Ruff check clean; ruff format: head 19 hunks = base 19 in test_theme_contrast.py, others equal. Preflight passes. Boot CSS 583,354 -> 583,312 bytes; no limit changed; broad-selector ratchet untouched; no pre-import change.

**Files.** css: core/_variables.tcss (comment), components/_agentic_terminal.tcss, components/_forms.tcss, features/_console.tcss, _console_panels.tcss, _lab.tcss, _library.tcss, _library_panels.tcss, _settings.tcss, _watchlists.tcss; generated: tldw_cli_modular.tcss, screen_agentic_library.tcss, screen_agentic_settings.tcss, screen_feature_watchlists.tcss; Widgets/Prompts/prompt_block_editor.py; tests: test_theme_contrast.py (new), six pins on the old token; scripts/ui_pr_gate_census.txt, scripts/check_ui_pr_gate_census.py; DESIGN.md; the Roleplay spec's rail-row lines. Evidence: .worktrees/.fu-evidence/36479/.

**Fix round 1 (pre-merge review, 2026-10-24).** (1) The debt list was keyed per token, so `$primary`, `$accent`, `$ds-focus-accent` and `$ds-input-focus-accent` were exempt wholesale (138 of the 152 edges): reverting `.library-rail-row:focus` to `$primary`, or adding a new `$accent` focus edge, stayed green (reproduced: `90 passed` under both). It is now a checked-in TSV keyed per edge (source, selector, property, painted value; 152 rows), and the guard fails on a row that no longer matches an edge or whose edge clears 3:1, or on a duplicate row. Both reproductions now go red (29 and 37 themes). (2) DESIGN.md and the Roleplay spec now scope the invariant to the edges moved off `$ds-action-focus` and point at the debt list and TASK-36663. (3) Literal-colour focus edges are measured (`solid white`/`solid black` note bodies, gated by `.-dark-mode`/`.-light-mode` polarity); an edge with no readable colour fails the sweep. (4) Class CSS written as `X: str = "..."` is discovered, as `widget_css.iter_blocks` lifts it. (5) BUNDLED_SCREEN_CSS parses without the token preamble (`build_css.build_screen_css`); the scope flag says so. (6) The PR table's "after" figures footnote the generic Button:focus fill. (7) The stale 3.77:1 figures now give the measured `$ds-active-fg` ratios in textual-dark: 5.12:1 on the Inspect rail's `$panel` ground and 6.26:1 on `$surface`. (8) Filed TASK-36663 for the 152 edges, the Button:focus fill, litestep's `$background` and the 13 non-focus `$ds-action-focus` edges.
<!-- SECTION:NOTES:END -->
