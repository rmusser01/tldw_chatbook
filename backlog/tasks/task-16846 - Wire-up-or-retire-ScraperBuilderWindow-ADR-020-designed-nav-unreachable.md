---
id: TASK-16846
title: 'Wire up or retire ScraperBuilderWindow (ADR-020 designed, nav-unreachable)'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
labels:
  - ui
  - dead-code
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15991 (PR #1701) made `UI/ScraperBuilderWindow.py` *openable* — it had never once
composed successfully (nonexistent `FormBuilder.create_switch`, a `Collapsible`
positional-string `MountError`, two backwards Selects), proof that nothing had ever
reached it. But it remains **nav-unreachable** at dev `ee741cf10`: repo-wide grep finds
`ScraperBuilderWindow` referenced only by its own file and its regression test
(`Tests/UI/test_scraper_builder_window.py`); zero matches in
`UI/Navigation/screen_registry.py` or any command-palette provider. User impact of the
15991 fix is nil until this decision is made.

The design record says it is a feature, not a leftover: "ADR-020: Visual Scraper Builder"
(`Docs/Development/Subscriptions/Subscriptions-Implementation-1.md:303`, Status:
Accepted, 2025-08-01 — note the number collides with an unrelated `backlog/decisions/`
ADR-020, a pre-existing doc-set collision) describes an interactive UI for testing
selectors and building extraction rules. Nothing anywhere marks it retired.

Decide (owner call): wire it into navigation/the Watchlists surface it was designed to
serve, or retire it and amend the ADR — the same fork its nav-unreachable sibling
`UI/SiteConfigSettings.py` sits on (whose own live Select bug is filed separately). If
wired, a live-drive check of the full build-test-export flow is due, since the window has
never been exercised by a user.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 An explicit decision is recorded against ADR-020 (wire or retire), with the ADR/doc updated to match (decision: RETIRE; ADR-020 amended in Docs/Development/Subscriptions/Subscriptions-Implementation-1.md)
- [x] #2 If wired: the window is reachable through real navigation, and its primary flow works in a live drive (not just the mount test) (N/A — decision is RETIRE)
- [x] #3 If retired: the window, its test, and its `SiteConfigSettings` sibling's disposition are handled together with reachability evidence (window + test deleted, retirement pin added; sibling's fork filed as TASK-16865 with its reachability evidence recorded there and in the ADR amendment)
<!-- AC:END -->

## Implementation Plan

1. Re-verify reachability at base `34ab21f7e`: repo-wide grep for `ScraperBuilderWindow`
   — expect only its own file, its test, the two generated screen-CSS sheets, the
   diagnostic-inventory row, and historical docs/task files.
2. Evidence for the wire side: hunt for a designed affordance (Watchlists surface,
   navigation/palette, docs); read `SiteConfigSettings` + `SiteConfigManager` to test
   the redundancy claim; map the consumers of the pipeline stack the builder's output
   targets (`web_scraping_pipelines.py`, `Subscriptions/scrapers/`).
3. Evidence for the retire side: enumerate the window's own stubs (unbound buttons,
   bindings to nonexistent actions, dead-end Downloads-JSON persistence) — a wire would
   be a feature build, not a nav entry.
4. Decide per the 16837/16195 playbook and the owner's standing rulings (stability over
   quick wins; no speculative UI). Decision = RETIRE (chain in Implementation Notes).
5. Execute: baseline the affected suites to files; delete
   `tldw_chatbook/UI/ScraperBuilderWindow.py` + `Tests/UI/test_scraper_builder_window.py`;
   regenerate the screen-CSS sheets via `build_css.py` (regenerate-never-hand-merge);
   surgical hand-edit of the diagnostic-inventory row + summary counts; add the module
   to `RETIRED_MODULES` in `Tests/Subscriptions/test_retired_modules_stay_retired.py`
   (mutation-verify by resurrecting the file from HEAD, Edit-based restore); amend
   ADR-020 in `Docs/Development/Subscriptions/Subscriptions-Implementation-1.md`.
6. SiteConfigSettings disposition: record it in the ADR amendment and file a follow-up
   wire-or-retire task for it (collision-safe ID against origin/dev + worktrees).
7. Re-run suites, `--collect-only` sweep over Tests/UI + Tests/Subscriptions, ruff on
   touched files; per-symbol dead-verdict table + notes; ACs; Done.

## Implementation Notes

**Decision: RETIRE** (redundant-by-supersession plus stub-beyond-facade; owner's
standing ruling — stability over quick wins, no speculative UI). Evidence chain, all
re-verified at branch base `34ab21f7e`:

1. **Nav-unreachable, and never was reachable.** Repo-wide grep finds
   `ScraperBuilderWindow` only in its own file, its regression test, the two generated
   screen-CSS sheets, one diagnostic-inventory row, and historical docs/task files. No
   screen-registry route, palette entry, button, or import — and TASK-15991 proved the
   window had never once composed (four stacked compose crashes), so zero users ever
   saw it in the year since ADR-020 (2025-08-01).
2. **Its persistence is a dead end.** "Save Configuration" writes
   `scraper_<domain>_<ts>.json` to `~/Downloads` — nothing anywhere reads such a file.
   "Load Configuration" (`#load-config-btn`) is composed but has NO handler. Four of
   its five BINDINGS name actions that do not exist (`action_test_selector`,
   `action_save_config`, `action_load_config`, `action_fetch_page` — only
   `action_close` is real). The per-rule Edit/Delete buttons (`edit-rule-*`,
   `delete-rule-*`) have no handlers. Wiring would be a feature build (re-plumb
   persistence, implement load/edit/delete/actions, live-drive a never-exercised
   fetch flow), not the "small honest completion" the wire branch required.
3. **The backend it integrates with is itself production-orphaned.** Its exported code
   imports `CustomScrapingPipeline`; repo-wide, `Subscriptions/web_scraping_pipelines.py`
   and the whole `Subscriptions/scrapers/` package have zero production consumers
   outside their own cluster (sole outside reference was the builder's generated-code
   string). The live watchlists path is `monitoring_engine.py`.
4. **The user need has a better-grained surviving surface.** `SiteConfigSettings` +
   `SiteConfigManager` cover per-site extraction selectors
   (content/title/date/author, exclude/ignore), JS options, custom headers, rate
   limits, auth, and presets — persisted to the `site_configs` table (the store the
   pipelines were built to read) instead of a Downloads JSON. It is itself
   nav-unreachable (same fork) — so the claim is "sole coherent candidate", not
   "already serves users"; its own wire-or-retire is now **TASK-16865** (filed with a
   collision-safe leapfrogged ID: full remote-ref + worktree sweep max was 16852, CLI
   probe answered 16853, leapfrogged to 16865, zero ghosts on any ref).
5. No designed affordance exists anywhere for opening the builder (grep across
   UI/Widgets/Navigation for "scraper": only the two sibling files + a docstring).

**Per-symbol dead-verdict table** (repo-wide grep at base; re-checked after deletion —
zero non-historical references remain outside the tombstone):

| Symbol | Was at | Production references | Verdict |
|---|---|---|---|
| `ScraperBuilderWindow` (Screen) | `UI/ScraperBuilderWindow.py:71` | none — no route/palette/import; only own file + own test + generated CSS + inventory row | dead |
| `_syntax_text_area` | `UI/ScraperBuilderWindow.py:56` | own file only | dead (transitively) |

(The module has exactly these two top-level symbols; nothing else imported it.)

**What shipped** (branch `task/16846-burn`, base `34ab21f7e`):

- Deleted `tldw_chatbook/UI/ScraperBuilderWindow.py` (921 lines) and
  `Tests/UI/test_scraper_builder_window.py` (15991's fixes are sunk groundwork).
- Regenerated the screen-CSS sheets via `python tldw_chatbook/css/build_css.py`
  (regenerate-never-hand-merge): `screen_css_scoped.tcss` −114 lines,
  `screen_css_self.tcss` −46, verified to contain ONLY ScraperBuilderWindow blocks
  (7→6 screen classes); the modular bundle's diff was timestamp-only and was
  dropped.
- `Docs/security/production-diagnostic-inventory.json`: surgical hand-edit per
  precedent — removed the ScraperBuilderWindow row (TASK-494, call_count 3),
  `owner_files` 499→498, `task_494_calls` 6952→6949. Proof of surgicality: after the
  edit, diff(committed, fresh `build_inventory()`) is byte-identical to the base's
  pre-existing drift (three foreign rows only: `chat_screen.py` 146→147,
  `SiteConfigSettings.py` 4→5, `console_transcript.py` 9→10). A `--write` would have
  absorbed that foreign drift into this commit.
- `Tests/Subscriptions/test_retired_modules_stay_retired.py`: added
  `tldw_chatbook.UI.ScraperBuilderWindow` to `RETIRED_MODULES` with the retirement
  rationale. Mutation-verified: red against the module resurrected from HEAD, green
  with it gone; resurrection was from committed HEAD content so no uncommitted work
  was at risk.
- ADR-020 amended in `Docs/Development/Subscriptions/Subscriptions-Implementation-1.md`
  (Status → Retired + amendment paragraph recording the chain and the sibling's
  disposition), following the ADR-019-amendment style.
- Filed follow-up `task-16865 - Wire-up-or-retire-SiteConfigSettings-…` covering the
  sibling AND the orphaned scrapers/pipelines cluster disposition.

**Evidence:** baseline (base, pre-change) for the six affected suites: 2 failed /
102 passed — both fails pre-existing in `test_persistent_diagnostic_inventory.py`
(checker exit 1 on the base's own drift; the 15743 `console_runtime.py`
metadata-only red known from 16837). After: the five non-inventory suites 41 passed /
0 failed; inventory suite 2 failed / 63 passed with byte-identical signatures and
zero ScraperBuilder mentions. `--collect-only` over Tests/UI + Tests/Subscriptions +
Tests/Architecture: 14,016 collected, zero errors. `ruff check` + `format --check`
clean on the touched test. Dangling-ref sweep post-deletion: remaining mentions are
the tombstone, historical docstrings/task files, the dated 2026-08-11 audit record,
and the amended ADR — no live references. Dev's 4 commits since base touch only TTS
files (no overlap with any touched path).
