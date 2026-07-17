---
id: TASK-296
title: CSS dead-rule prune (+ optional deferred second bundle) per task-262 findings
status: Done
assignee: ['@claude']
created_date: '2026-07-17 23:40'
labels: [performance, startup, css]
dependencies: [task-262]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The task-262 investigation (Docs/Design/2026-07-17-css-split-investigation.md) rejected a per-screen CSS split (Textual's `_load_screen_css` does a full uncached `Stylesheet.reparse()` of every loaded source at each screen's first push, charging ~36-45ms warm per screen against a −38ms/−70ms startup win; `$ds-*` variables also don't cross file boundaries). It recommended instead: (1) a dead-CSS prune — 271 of 1,792 class/id tokens (15.1%) are referenced nowhere in the repo, concentrated in `_search-rag` and the largely-orphaned legacy `_embeddings` module, worth ~1.5-3k generated lines ≈ 15-30ms parse with zero mechanics risk (edit the `css/build_css.py` source modules, rebuild, verify byte-level rule removal + visual spot QA); and (2) ONLY if the startup budget still misses after the shipped diet (285/257/258): a two-phase load — boot bundle + one vars-prefixed deferred bundle applied via a single idle reparse after first paint (~70ms cold win, needs an early-navigation force-load guard + one-time full-app visual QA). Also fold in: remove the dead `CSS_PATH` on a `Container` at `Chat_Window_Enhanced.py:72` (no-op attribute, misleading).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dead selectors removed from the css source modules; rebuilt monolith drops the corresponding rules; parse-time delta measured
- [x] #2 No visual regressions (spot QA on the screens whose modules were pruned) — controller pixel A/B 2026-07-18: Console/Home/Library/Watchlists/Workflows/MCP/Settings captured on pruned vs base builds (fresh identical profile, textual-serve+playwright 2050x1240); 4 screens pixel-identical, 3 differ ONLY in dynamic content (composer caret blink 0.004%, footer memory counters 0.015%) — zero layout/style deltas
- [x] #3 Dead Chat_Window_Enhanced CSS_PATH attribute removed
- [x] #4 Two-phase-load decision recorded (implement only if startup budget still misses; otherwise explicitly declined in notes)
<!-- AC:END -->

## Implementation Plan

1. Verify `build_css.py` reproduces the committed monolith byte-identically (it does, modulo the `Generated:` timestamp line) so source-module edits are the only lever.
2. Script a per-token audit over every class/id token in the 43 bundled source modules using the strict verification protocol: bare-substring grep over every tracked non-CSS file (all .py incl. Tests/, md, toml, json); dynamic-construction grep for every hyphen/underscore-boundary prefix in construction context (`prefix{`, `prefix"`, `prefix'`, `prefix%`) across all .py; auto-keep `--` component classes and leading-`-` textual-internal classes; CSS references never confer liveness (whole selectors evaluated instead). When in doubt, keep.
3. Remove only whole rules where EVERY comma-separated selector contains at least one verified-dead token (such a selector can never match). Pure line deletions only (rule + adjacent single-line non-banner comment + redundant blank).
4. Delete unbundled orphan CSS files that nothing (build manifest, Python, tests) reads.
5. Remove the no-op `CSS_PATH` on the `Container` at `Chat_Window_Enhanced.py:72`.
6. Rebuild the monolith; verify the diff is pure removals; measure parse time with Textual's `Stylesheet.parse()` before/after; run CSS contract suites + a broad UI slice; reproduce any failure at the base commit before attributing.

## Implementation Notes

Executed on branch `worktree-css-prune-296` (base `origin/dev` @ 1fd7a099). Audit
script + full evidence artifact:
`/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/3c226b20-7367-4bd0-90e6-a316764632a0/scratchpad/css-prune-296/`
(`audit.py`, `audit_results.json` — per-token grep-absence evidence,
`VERIFIED_DEAD_TOKENS.txt`, `prune.py`, `parse_probe.py`, `monolith_before.tcss`).

**Token audit** (1,653 distinct class/id tokens across 2,136 top-level rules in
the bundled sources): **133 VERIFIED DEAD** (zero hits in any tracked non-CSS
file AND no dynamic-prefix construction in any .py); kept-in-doubt: **100**
flagged by the dynamic-prefix check (e.g. anything under a constructed
`chat-`/`action-`/`button-group-` prefix), **29** auto-kept
component/internal (`--`, leading `-`) classes, 1,391 plainly referenced. This
is deliberately narrower than the investigation doc's 271 "referenced nowhere"
count — the doc's caveat about runtime-composed names is honored by keeping
every prefix-constructible token.

**Prune**: 152 whole rules (every comma-part selector contained a dead token →
unmatchable) removed from 16 source modules, 846 source lines: `_embeddings` 26
rules, `_wizards` 24, `_search-rag` 23, `_evaluation_unified` 22, `_ingest` 20,
`_chat` 10, `_code_repo` 7, `splash_viewer.css` 5, `_splash` 4, others 1–3.
Also deleted 2 of the 7 unbundled orphan files — `components/loading_states.css`
and `features/_index.tcss` — after verifying build_css.py's manifest never reads
them and no Python/test references them. The other 5 orphans
(`_unified_sidebar.tcss`, `_chatbooks_improved.tcss`, `_ingestion_rebuilt.tcss`,
`_new_ingest.tcss`, `main.tcss`) are **kept**: they are read as fixtures by
`Tests/UI/test_non_obscuring_focus_contract.py` (incl. the intentional
`SOURCE_ONLY_CSS_MODULES` not-in-bundle contract) and
`Tests/UI/test_master_shell_design_system_contract.py:99`; deleting them means
contract-test surgery — follow-up material, not a free deletion. The
`_embeddings` "screen-unreachable chrome" block (~80 orphan-only tokens) is also
**kept**: it styles widgets that still exist and are instantiated by their tests
(`chunk_preview`, `toast_notification`, `detailed_progress`,
`embedding_template_selector`, `activity_log` …) — per the verification
protocol a test-instantiated widget is not dead; removing that CSS requires a
task-253-style legacy-widget deletion sweep first. 5 dead tokens
(`ccp-attachment-inspector-pane`, `ccp-behavior-detail-pane`,
`ccp-character-library-pane`, `ccp-column-title`, `dataset-management-form`)
survive only as comma-parts inside live shared rules
(`_agentic_terminal.tcss`/`_evaluation_unified.tcss`) — selector-list surgery
skipped to keep the diff pure-removal.

**Rebuild/measure**: monolith 15,146 → 14,291 lines (−855; git diff is pure
removals, the single `+` line is the regenerated `Generated:` timestamp);
Textual-parsed rules 2,136 → 1,984 (−152, exactly the audited rule count — the
pruned bundle parses cleanly). Parse time (`Stylesheet.parse()`, repo venv
Textual 8.2.7, warm, median of 10): **76.4 ms → 71.8 ms (−4.6 ms, −6.0 %)**,
proportional to the −7.1 % rule count; fresh-process first-parse runs are
order-biased by interpreter warmup, so the warm median is the honest metric
(scaled to the investigation's ~144 ms real-app cold parse this is ~−9 ms cold).

**AC#3**: removed the dead `CSS_PATH = "css/features/_chat.tcss"` on
`ChatWindowEnhanced(Container)` (Textual honors `CSS_PATH` only on
`App`/`Screen`; per the investigation doc §1 this was a no-op).

**AC#4 — two-phase deferred bundle: DECLINED.** The investigation doc
(`Docs/Design/2026-07-17-css-split-investigation.md` §7) gates option 2 on "only
if the first-paint budget still misses after tasks 285/257/258". The shipped
startup diet met the budget (cold import 1.5 s → 0.57 s), so the deferred
bundle's costs — an idle-callback reparse, an early-navigation force-load guard,
the `.disabled` equal-specificity tie audit, and a one-time full-app visual QA —
buy at most ~70 ms cold that is no longer the bottleneck. Doc §7 option 3
("re-measure before spending the QA") applies if the budget ever regresses.

**Verification**: `TLDW_TEST_MODE=1` app-module import OK. CSS contract suites
(`test_non_obscuring_focus_contract.py`,
`test_master_shell_design_system_contract.py`,
`Tests/QA/test_agentic_terminal_css_tokens.py`,
`Tests/QA/test_textual_highlight_selectors.py`, `test_focus_accessibility.py`):
126 passed, 2 failed — both reproduced unchanged at base 1fd7a099
(`test_library_mode_chip_selector_is_retired`,
`test_generated_bundle_uses_textual_highlight_selector`; pre-existing, the
`#embeddings-model-list …` selector is absent from the BASE bundle too). Broad
UI slice `test_console_native_chat_flow.py` + `test_library_shell.py` +
`test_home_screen.py` + `test_latest_dev_core_app_usability_smoke.py`:
**506 passed, 0 failed**. `test_chat_window_enhanced.py` (CSS_PATH removal):
29 passed. AC#2 (live screen spot-QA) is left for the controller.
