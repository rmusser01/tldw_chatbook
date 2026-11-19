# Roleplay frame B1: one-row header and the lazy Roleplay stylesheet — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Roleplay's five header rows with one inline row (`Roleplay  <Kind> › <item>` · Unsaved chip · blocked-destination chip · `Local`/`Server: … · read-only`) driven by a new pure `roleplay_frame_state.py`, and give Roleplay its own lazily loaded stylesheet, with the two styled test tiers, the hostile-name test extended to the new header, and the glyph-map entries every later frame slice builds on.

**Architecture:** All header logic is pure (`UI/Persona_Modules/roleplay_frame_state.py`: inputs → view, the R24 predicate, the cell-measured fit, the degrade order, the escaping of untrusted text); `personas_screen.py` keeps only gather-and-paint glue: the header and purpose-line text move to the pure module (spec §5.11's B1 extraction row), and its size-ratchet row is set to whatever B1 measures (owner ruling 2026-10-04: expand the limit, never contort code). The header row is the existing `DestinationHeader` with the kind as its subtitle and a `before_status` tail of three `FittedText` widgets (a new, CSS-free shared widget in `UI/Workbench/workbench_widgets.py` that fits literal text to its own width on every render, so the item name ellipsises with the resolved glyph and never through markup). Every new rule lives in `css/features/_roleplay.tcss`, split whole into `screen_feature_roleplay.tcss` and parsed by `TldwCli._ensure_screen_owned_css` on the first visit, never through `PersonasScreen.CSS_PATH`; boot CSS falls.

**Tech Stack:** Python 3.12 (`.venv`, uv-managed), Textual 8.2.8 (`Content`, `textual.css.parse.parse_selectors`), Rich `cell_len`, pytest 8 + pytest-asyncio (`asyncio_mode = "auto"`) + pytest-xdist + pytest-timeout, the repo's `build_css.py` split-sheet build, Backlog.md CLI 1.44, the live harness under `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/harness/`.

**Spec:** `Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md` — slice card §5.2 "#### B1 · One-row header (D7 = A)"; also §1.1, §1.3, §2.12 items 1-4, §2.13, §3.12, §4.12, §5.3, §5.4, §5.7 (§5.7.1 the two styled tiers, §5.7.2 relative geometry), §5.8, §5.9 G4/G12/G17/G18/G20, §5.10, R12, R18, R24, R33, and Appendix B rows MF-01, MF-02, MF-03, MF-08, MF-14, MF-15, MF-18, FU-2, FU-3. The spec's line numbers are pinned to `84247cb843`; every anchor in this plan was re-taken by symbol on the B1 worktree at `f0a766a155` (B0's head then, on `origin/dev @ 3c439d606e`) and the whole plan was dry-run there, then on B0's rebased heads `bd41347b65` and `7495abd13f` (the review rounds). **B0 (PR #2977) merged to dev on 2026-10-04 06:24Z by rebase merge**, so its commits are on `origin/dev` under new SHAs and B1 no longer stacks on a branch in review; PR #2993 (merged the same morning) ruff-formatted `personas_screen.py` (16,397 → 16,525 lines, AST-identical) and its tests. The plan was then re-anchored on `origin/dev @ 83c264f286` and dry-run there. **2026-10-04 rework:** the owner sent the plan back (the three rulings at the top of Global Constraints), and TASK-34400 (PR #3015, merged that day) had already made every existing Roleplay surface paint untrusted text literally, a fix this plan's first Task 8 used to carry. The plan was re-anchored on `origin/dev @ 8c4dfe59a2` (TASK-34400, PR #2996 and PR #3006 included; #3006 touches no B1 file) and dry-run there. **2026-10-08 re-anchor:** dev moved 506 commits to `a793acbef5`. Under the plan's files only the PR-gate census, its checker and the workflow changed shape (the UI Fast Lane now runs four shards, and the admission-sensitive step lists fourteen files, so Task 11 Step 3's old anchor was gone), plus the module-size ratchet test (other rows), `app.py` (+3/−3, the same 5,710 lines), the regenerated bundle and one mark in `test_personas_workbench.py`; a rulings-and-coverage review's twelve findings were applied the same day. The plan was re-anchored on `origin/dev @ a793acbef5` and dry-run there. **2026-10-08, second re-anchor:** dev moved 32 more commits to `8d502ba250` (PRs #3031 and #3028; under the plan's files only the PR-gate census (+19 lines, 154 files), the workflow (+4), `_console.tcss` and two generated split sheets changed, and dev's boot CSS rose 134 B to 608,077, 13 B under its ceiling). An independent dry run there applied the whole plan in order (all 95 Old blocks matched exactly once, all 9 Creates applied) and passed every gate with B1's deltas unchanged; its six minor findings were applied the same day and each affected part re-run on `8d502ba250`. Every number below was measured on a paired base arm at `8d502ba250` unless the line names another SHA (Self-Review §5 lists what was re-run and what was not).

## Global Constraints

- **Owner rulings, 2026-10-04 (binding; they override anything below that disagrees):**
  1. **Pre-import census: "Expand it".** B1 keeps the header logic in its own new module, and the pre-import limits rise by exactly what B1 measures, with a dated note. Task 0 Step 6 records the sign-off; it does not ask again. The bound stays: exactly the constants B1's head exceeds, each raised to the paired-arm measurement, one ADR-097 ledger row, within +1 module and +1,000 lines (pass and largest route). Growth beyond that bound still STOPs for the owner.
  2. **Expand limits, never contort code.** When B1's straightforward edits grow a ratcheted file (`personas_screen.py` or any other), its row rises to the measured count in the same commit, with a dated "owner decision" comment naming TASK-33910.2. No step merges, moves, reflows or squeezes code the task would not otherwise change just to fit a ratchet, and no step exists only to save lines. Moving the header logic into `roleplay_frame_state.py` is the spec's design (§5.11's B1 extraction row), not a squeeze.
  3. **Wording.** Wherever this plan, its commit messages, the PR body, the task notes or any owner-facing text mentions a bug, it says whether the bug exists on dev today and whether this task fixes it. Never a bare "X crashes".
- Interpreter: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, run from the worktree root `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1`; never system `python`/`pytest`. Textual is 8.2.8.
- Module-size ratchet (`Tests/Architecture/test_module_size_ratchet.py`; ruling 2): `personas_screen.py` (PS) is 16,528 lines against its 16,528 row on `origin/dev @ 8d502ba250`, as on `a793acbef5` and `8c4dfe59a2` (TASK-34400 raised the row +3 with an owner-decision comment). B1 sets the row to the measured count in the commit that changes PS (Task 6 Step 9, by script, re-run after every rebase): lowered when B1 shrinks PS (§5.4 item 4; the dry run measured 16,528 → 16,525), raised with a dated owner-decision comment naming TASK-33910.2 if a rebase or a review fix makes B1's PS larger than the base row (the comment states B1's own delta against the base arm's line count; a row that is already red on the base arm is dev's, not B1's to raise, and the script STOPs). The same script adds a recipient-ceiling row for `roleplay_frame_state.py` at its measured count, and holds `tldw_chatbook/app.py`, the only other ratcheted file B1 edits: 5,710 lines against its 5,712 row on dev, and B1's two lines land exactly on the row (5,712), so dev growth before B1 merges raises that row the same way. PS keeps only gather-and-paint glue; the header text and the mode descriptors move to the pure module (spec §5.11), and no PS edit reshapes code B1 does not otherwise change.
- PS edits never touch the `personas_preview_coordinator` import block or the `_drain_*` helpers (PR #2862's two hunks; spec §5.11 "B never edits PS:360-375 or :820-960", re-anchored by symbol). New PS imports go after the `personas_state` import block.
- Boot parsed CSS: net ≤ 0 bytes in this PR (the B1 card's gate), measured with `_boot_parsed_css_census()` (608,077 B on `origin/dev @ 8d502ba250`, 13 B under the ceiling; 607,943 B on `a793acbef5`, 607,326 B on `8c4dfe59a2`, 607,951 B on B0's heads; the dry run on `8d502ba250` measured 607,958 after Task 5 and 607,862 after Task 6, −215 in all, exactly as on `a793acbef5` and `8c4dfe59a2`: −216 for the dead `_agentic_terminal.tcss` items, +52 for the new module banner, +45 for the `$ds-status-warning-readable` line, −96 for the task-523 rule and the blank line after its comment). The ceiling `MAX_BOOT_PARSED_CSS_BYTES = 608_090` never rises (the dead-item deletions are the B1 card's own scope, not a squeeze).
- Broad selectors: no increase. The ratchet constant `MAX_ANCESTOR_SCOPED_BARE_TYPE_RULES` is 274, and 274 was measured on BOTH arms in review at `7495abd13f` and again in the dry runs at `8c4dfe59a2`, `a793acbef5` and `8d502ba250` (B0's `test_ancestor_scoped_bare_type_rule_count_is_a_ratchet` waits for `_ui_ready`; the 273 measured earlier predates that), so there is zero headroom: the gate is head ≤ base. No bare-type subject anywhere: not in `BUNDLED_CSS` (counted) and not in `_roleplay.tcss` (uncounted, so it is pinned by its own test).
- CSS sources after the destination tour: +1 at most and below 56 (`test_destination_tour_css_sources_stay_below_parse_cache`; 41 on the base arm and 42 on head in the dry runs at `8c4dfe59a2`, `a793acbef5` and `8d502ba250`, as at B0's heads: the lazy sheet itself; Task 12 Step 2 re-measures both arms). B1 adds no Textual container class Roleplay did not already use: each one parses its own `DEFAULT_CSS` source on first use (a `HorizontalGroup` tail measured 43). No new widget declares `DEFAULT_CSS`, `CSS` or `BUNDLED_CSS`.
- The Roleplay sheet loads only through `TldwCli._SCREEN_OWNED_ROUTE_CSS[TAB_PERSONAS]`, never through `PersonasScreen.CSS_PATH`.
- `_roleplay.tcss`: class/id subjects only; zero numeric dimension literals (ADR-161's `_DIM` pattern); `$ds-*` tokens only. B1 needs no `$ds-roleplay-*` token (G17: "as needed"; every dimension has a scale token). It adds one design-system colour token, `$ds-status-warning-readable: $text-warning;` (Task 5), the readable sibling of `$ds-status-error-readable` (task-2230): the chips are words that must be read, and the decorative `$warning`/`$error` hues fall below AA on the panel in 29 and 40 of the 70 registered themes (measured while planning).
- The Roleplay split claims the R18 prefixes plus exactly four shared tokens (`workbench-header-title`, `workbench-header-subtitle`, `workbench-header-status`, `-active`); every Roleplay selector must also carry a Roleplay-only token (pinned in Task 5).
- Pre-import census (ruling 1): exactly +1 module (`tldw_chatbook.UI.Persona_Modules.roleplay_frame_state`; `MAX_PASS_ADDED_MODULES` is 557 on dev, B0's owner-signed raise), measured on paired arms; ADR-097 order (defer → shed → owner-signed ledger row in the SAME commit as the constant raise, spec Q10). The owner signed off on 2026-10-04 ("Expand it"); Task 0 Step 6 records that answer verbatim in `$EV/owner-signoff.txt`, and the raise lands in Task 6's commit (the first commit whose PS imports the module). Growth beyond the bound (a second added module, or more than +1,000 lines on the pass or the largest route) is not covered: `preimport_raise.py` STOPs, and only the owner can extend it (a subagent's or the controller's approval never can).
- UI-ready census and boot import weight: +0 against the paired base arm (`roleplay_frame_state` is never on the boot or `_ui_ready` path). The UI-ready census has been flaky by one module on BOTH arms (measured while planning at `af13839740`: `tldw_chatbook.DB.character_conversation_search` was resident at `_ui_ready` in about half the boots; on `8c4dfe59a2`, `a793acbef5` and `8d502ba250` the paired arms read 1033, the limit, in all three boots each, with identical module sets); Task 12 Step 2 therefore compares the module SETS of three boots per arm, never one pass/fail verdict.
- Never touch: `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/Library/library_shell_state.py`, `tldw_chatbook/UI/Library_Modules/screen_helpers.py`, `tldw_chatbook/Widgets/Library/library_emergency_return.py` (B0's never-touch list; PS is touched because the B1 card requires it). Roleplay production code never imports `Widgets/Library/`.
- `#personas-purpose` and `#personas-mode-strip` stay (until B6); the footer-hint builder stays (B3 replaces it).
- Untrusted text (R33). On dev today every existing Roleplay surface already paints untrusted text literally: TASK-34400 (PR #3015) made the Inspector's selected-name, validation and readiness lines `markup=False`, the Tag filter button's label literal `Content`, every Roleplay toast `notify(..., markup=False)`, and escaped the item name into the old header subtitle. B1 removes that subtitle copy and adds new sinks, which must keep the guarantee: the item label and the chips are literal `FittedText`s (`Content`, never markup); the shared markup-on `DestinationHeader` status gets `Utils.input_validation.escape_markup` (NOT `textual.markup.escape`, which mishandles `[/` and `[TODO] y` on Textual 8.2.8); text is measured and cut as plain text first, escaped last. The header's subtitle (markup-on, shared) carries only the app-authored kind; the untrusted item name never enters it; the server label reaches the markup-on status only through `escape_markup` inside `build_header_view`, the screen's one call site (FU-2, TASK-33910.20, later gives the shared header a literal mode). B1 removes the old subtitle, and with it TASK-34400's escaping there, because the name now paints in a literal label; every other TASK-34400 fix is unchanged, and its 22 + 81 hostile-text tests stay green (two of them asserted the retired subtitle and are re-pinned to the header that replaces it, Task 6 Step 7).
- Glyphs: every header glyph goes through `resolve_glyph`; user names never pass through `resolve_glyph_text`.
- DESIGN.md G12 amendment: the inline one-row variant for Lab and Roleplay, written as a sibling of the Console variant (ADR-210) in one "Destination header variants" list.
- No "Verified against" stamps in `Docs/User_Guide/` (the worktree CLAUDE.md forbids them); live-check dates and SHAs go in the task's Implementation Notes.
- Formatting: the formatter batches TASK-26983 (PS) and TASK-27000 (`ui-personas`) are Done on dev (PR #2993), so they ran before B1, the order spec §5.6/K19 allows. Every Python file B1 edits is `ruff format`-clean on dev (checked at `8c4dfe59a2`, and again at `a793acbef5` and `8d502ba250`, where Task 12 Step 5's check read `26 files already formatted` on head: PS, `personas_preview_controller.py`, `roleplay_draft_guard.py`, `workbench_widgets.py`, `glyph_fallback.py`, `adaptive_pane_shell.py`, `check_ui_pr_gate_census.py` and the edited tests, TASK-34400's two hostile-text modules included) except three that are not: `tldw_chatbook/app.py`, `tldw_chatbook/css/build_css.py` and `Tests/UI/test_css_build_integrity.py` (never run the formatter on those; it would reformat lines B1 does not own). B1 keeps the clean ones clean: every New block is written formatter-clean, `ruff format --check` covers every touched Python file but those three (Task 12 Step 5), and a check that fails is fixed with `ruff format <file>` (only B1's own lines can move, since the base is clean), then the PS line count is re-measured. New files are ruff-formatted.
- Every new `Tests/UI` file imports `tldw_chatbook.app` at module scope. Files that mount `PersonasScreen` carry `pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]` (TASK-32873: such files cannot share the UI Fast Lane invocation; they gate in the PR Fast Lane's admission-sensitive step, Task 11).
- Unstyled-tier tests (`PersonasTestApp`, a `ConsolidatedCSSApp`) never assert geometry; geometry runs under both styled tiers and is asserted relative to the measured nav and header (§5.7.2 item 1).
- Judging: "no new failures versus the paired base arm" (failure-set diff), never "passes unchanged". `$EV/paired.sh` prints `recovery=<n>` per arm; above single digits the comparison proves nothing: stop and fix the environment. One known exception, deterministic on dev and not the environment: the `shell` group's `Tests/UI/test_settings_configuration_hub.py` tests that build a fresh config raise `RecoveryRequired: raw_source_selection_changed` under the bootstrap plugin on both arms (Task 12 Step 4 says how to read that group's count).
- Every new geometry, containment or literal-text assertion is shown red by a named mutation that leaves the code importable; record it in `$EV/mutations.md`; restore with the Edit tool (never `git checkout --`, never `git stash`).
- A pytest run that reports "no tests ran" is a FAILED gate: read a nonzero count.
- Rebase protocol: B0 is on dev, so after Task 0 Step 1 B1 sits directly on `origin/dev`, and `$EV/stack-cut.txt` holds the dev SHA it was last moved onto (also the paired base arm, `$EV/base-sha.txt`). Every move is `$EV/restack.sh` (written in Task 0 Step 1): it refuses to run unless every commit in `CUT..HEAD` is B1's (its subject carries `(B1`, ` B1 ` or `TASK-33910.2`), runs `git rebase --onto origin/dev CUT`, records the new cut and base, and re-anchors the base arm. A conflicting rebase stops there: resolve it, `git -C $WT rebase --continue`, then `$EV/restack.sh --record`. Never hand-merge generated sheets, snapshots, the pre-import constants, the ADR-097 row, the PS ratchet row, the PR-gate census or the workflow's admission-sensitive step: take the upstream side of those files (Task 11 Steps 2-4 re-apply B1's lines by script). `css/components/_agentic_terminal.tcss` is a SOURCE file that PR #2953 re-indented on dev (merged 2026-10-04) and that open PRs #2862 and #2563 still edit: on a conflict there, take the upstream side and re-run Task 5 Step 4's deletion script (its count asserts re-verify the nine items). Then run `tldw_chatbook/css/build_css.py` and `tldw_chatbook/css/check_bundle_sync.py`, re-run `$EV/ratchet_rows.py` (Task 6 Step 9; idempotent) and Task 11 Steps 2-4, then `./scripts/preflight.sh`.
- Push and PR base: B1's PR targets `dev` (the required `Derived artifacts reproduce from their sources` workflow runs only on pull requests into `dev`, so Task 11's admission step only runs there). Push after every task's commit (the owner's standing rule): `git -C $WT push -u origin feat/roleplay-b1-one-row-header` the first time; after any `$EV/restack.sh`, only `git -C $WT push --force-with-lease=feat/roleplay-b1-one-row-header:$(cat $EV/pushed-sha.txt) origin HEAD`; record every pushed SHA in `$EV/pushed-sha.txt`. Never `git pull` the branch and never run `gh pr update-branch` on it (plain update-branch merges dev into it, and a server-side `--rebase` leaves this worktree and the evidence anchors behind the remote): re-sync only with `$EV/restack.sh` here and a force-with-lease push. The controller opens the PR (Task 12 Step 10) and merges it under CLAUDE.md's merge rules (strict: up to date with dev, the required check green on the current head, every thread resolved); Task 12 Step 9 confirms the required check ran on the PR's current head.
- Lint: new files are `ruff check` and `ruff format --check` clean. `tldw_chatbook/app.py` carries 96 pre-existing `ruff check` findings (94 `E402`, 2 `F401` on `8c4dfe59a2`; `Found 96 errors.` again on both arms at `a793acbef5` and at `8d502ba250`) on the base arm; its gate is "the same 96 on head" (Task 5 Step 7), never "All checks passed!".
- Worktrees live under `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/`, never `/tmp`. Never touch the main checkout's working tree. Never `git stash`.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Hostile names on B1's new header surfaces.** A character, persona, dictionary, lore book, entry key, tag, conversation title or server label named `[/]`, `[b]x`, `[@click=app.record('x')]N[/]`, `[/` or `[TODO] y` must paint as typed, raise nothing and act on no click. On dev today every existing Roleplay surface already does: TASK-34400 (PR #3015) fixed the pre-existing bug in which such a name exited the app (a render-time `MarkupError`) or became a live `@click` action. B1 changes one piece of that fix: it retires the old subtitle that carried the name, and with it TASK-34400's escaping there, because the name now paints in a literal label; every other TASK-34400 fix is unchanged. B1 must keep the guarantee on the surfaces it adds: the header's item label (a literal `FittedText`, viewing and editing) and the server label escaped into the shared markup-on status chip. Pinned in Task 8 by `test_the_header_item_label_and_server_label` (positive control, every painted copy clicked, `@click` scan; mutations `status-unescaped` and `item-label-markup`), in Task 4 by `test_server_status_is_escaped_for_the_markup_header_but_measured_plain`, in Task 6 by TASK-34400's two re-pinned tests (the item label while editing; `test_the_header_view_keeps_an_unsaved_item_out_of_markup`), and by the other 101 TASK-34400 hostile-text tests staying green. No PR lane ran either TASK-34400 module on dev; Task 11 gates both (`test_roleplay_hostile_names.py` in the admission-sensitive step, `test_roleplay_hostile_text_surfaces.py`, which carries no `bootstrap_profile` mark, in the UI Fast Lane census), so every pin here is checked on pull requests.
2. **The Unsaved chip agreeing with the aggregate, including in-flight saves.** A staged avatar, an open visual authoring or a save still running must show the chip although `has_unsaved_changes` is False; it must go when the domain is clean, and a successful persona Ctrl+S must clear it without waiting for the poll (the post-save sync inside `_after_profile_save` runs while the save is still flagged in flight, so B1 repaints once the flag clears; dev has no such chip, so this is a B1 design point, not a bug on dev). Pinned in Task 6 by `test_unsaved_chip_follows_the_aggregate_not_has_unsaved_changes` (seven cases) and `test_persona_save_clears_the_unsaved_chip_without_the_poll`, in Task 4 by `test_unsaved_predicate_is_the_aggregate_including_inflight_saves` and `test_the_leave_and_quit_guard_decides_on_the_same_predicate` (TASK-33622.14's guard asks the same predicate), and in Task 9 by the journey.
3. **The kind visible at 24 rows or fewer and never cut, with every part fitting.** The compact-height rule that hides a subtitle at ≤24 rows must lose to the inline rule, and with every chip, a server label, the longest kind and a 213-cell name, no header part may be clipped at 80x24. Pinned in Task 6 by `test_header_is_one_row_and_the_kind_stays_visible` (both tiers, four sizes), `test_every_header_part_fits_in_the_worst_case`, `test_header_chrome_cells_match_the_lazy_sheet` and `test_a_resize_refits_the_header_without_gathering_inputs` (a resize refits the chips from the cached inputs and never re-reads readiness, spec §2.13), and in Task 4 by `test_header_fits_from_65_columns_degrading_in_order`.
4. **A harness that never loads the lazy sheet, so geometry tests lie.** A bundle-only harness renders the old five-row header and would pass or fail for the wrong reason. A real `TldwCli` that pushes `PersonasScreen` itself skips `_ensure_screen_owned_css` and has the same blind spot (TASK-32187's Watchlists trap). Pinned in Task 1 (`StyledPersonasTestApp` re-pointed at `APP_STYLESHEETS`), Task 5 (`test_the_full_app_loads_the_sheet_on_the_first_visit_only`, `test_the_split_sheet_scan_reports_a_bundle_only_roleplay_harness`, the `_SPLIT_SHEET_OWNERS` entry with no owner, `test_no_full_app_test_pushes_roleplay_without_its_sheet`), and Task 7 (`test_deleting_one_header_rule_turns_the_one_row_assertion_red` under both styled tiers).
5. **Long, wide or zero-width names.** A 213-cell name must end in the resolved ellipsis (`…`, or `...` in ASCII mode, never Textual's CSS `…`), and a CJK, emoji or zero-width name must never paint past its width. Pinned in Task 4 by `test_item_fit_never_paints_past_its_width` and `test_item_fit_uses_ascii_markers_in_ascii_mode`, and in Task 6 by `test_a_long_name_ellipsises_and_the_kind_is_never_cut` and `test_header_row_paints_ascii_markers_in_ascii_mode` (both tiers, four sizes).
6. **The 0.25 s readiness poll now gathers the header's inputs on every tick.** It must not walk the DOM and must stay within 1 ms of the base arm's per-tick cost. On dev the draft aggregate finds the demand-mounted character editor with a `query_one` that walks the whole screen and raises `NoMatches` while browsing; that costs nothing today, because only the leave and quit guards read the aggregate. B1's per-tick gather would pay it on every tick (about 4 ms, measured in review on the first draft), so edit 16 reads the cached editor instead (about 0.1 ms). Pinned in Task 6 by `test_gathering_header_inputs_walks_no_dom` (mutation `aggregate-walks-dom`) and measured on both arms by Task 12 Step 3's poll probe.

---

## File Structure

**Create**

| Path | Single responsibility |
|---|---|
| `tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py` | Pure frame state (no Textual): `roleplay_has_unsaved_work`, `RoleplayHeaderInputs`/`RoleplayHeaderView`, `build_header_view`, `fit_header_item`, `ellipsize_cells`, `runtime_server_label`, the header chrome constants, `MODE_DESCRIPTORS`/`mode_descriptor`/`purpose_line` (moved from PS), `ROLEPLAY_PANE_CLASS_NAMES`. Later slices extend it (§5.11). |
| `tldw_chatbook/css/features/_roleplay.tcss` | Every Roleplay rule: the inline header, its tail and chips, and the copy of the Library's adaptive-shell rules keyed to Roleplay's classes. |
| `tldw_chatbook/css/screen_feature_roleplay.tcss` | Generated by `build_css.py` (never hand-edited). |
| `Tests/UI/roleplay_frame_harness.py` | The two styled tiers, the size matrix, the containment helper, the loaded-sheet mutation helper; home of the moved `PersonasTestApp` and the one home of the paint helpers (`painted_rows`, `click_meta_cells`, `settle`, `wait_until`, moved from TASK-34400's `test_roleplay_hostile_names.py`). Not collected. |
| `Tests/UI/test_roleplay_frame_harness.py` | Harness self-tests: tiers, sheet loading by both real routes (AC#8), discrimination (AC#6), the containment helper's refusals. Bootstrap-profile. |
| `Tests/UI/test_workbench_fitted_text.py` | `FittedText` contracts on a small host. UI Fast Lane. |
| `Tests/UI/test_roleplay_stylesheet.py` | Static sheet contracts: route-loaded, anchored, no bare type, boot bundle clean, shell-rule parity with the Library, the split-owners negative control, the chips' AA contrast on every theme, the dimension-literal floor. UI Fast Lane. |
| `Tests/UI/test_roleplay_frame_state.py` | Pure contracts of the frame state and the B1 Python-style floor. UI Fast Lane. |
| `Tests/UI/test_roleplay_header.py` | Mounted header contracts under both styled tiers at the four sizes, and the resize refit. Bootstrap-profile. |
| `Tests/UI/test_roleplay_journeys.py` | The B1 journey (Edit → type → chip → Ctrl+S → chip goes) under both styled tiers; later slices add J1-J4. Bootstrap-profile. |
| `Docs/superpowers/plans/2026-10-03-roleplay-b1-captures/*.txt` | The live check's approval set beside this plan (§5.4 item 6): four states × four sizes × two arms; every other capture, the `.ansi` colour captures and the PNGs stay in `$EV`. |

**Modify**

| Path | Change |
|---|---|
| `tldw_chatbook/UI/Screens/personas_screen.py` | One-row header compose (with a non-"Ready" initial state), `_update_title` → gather + paint, poll compares header inputs, `on_resize` repaints, persona-save ordering, blocked-chip click, the aggregate snapshot reads the cached editor instead of walking the DOM, header/purpose text and `_MODE_DESCRIPTORS` moved out, the task-523 comment dropped. 16,528 → 16,525 lines on dev; the ratchet row follows the measurement (ruling 2). |
| `tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py` | TASK-33622.14's leave and Ctrl+Q guard decides on `roleplay_has_unsaved_work()` (R24, G4: one predicate). |
| `tldw_chatbook/UI/Workbench/workbench_widgets.py` | New shared `FittedText` widget. |
| `tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py` | `open_provider_settings(defaults_key=...)` so the chip links to the chat_defaults provider. |
| `tldw_chatbook/Widgets/glyph_fallback.py` | Eleven §4.12 entries. |
| `tldw_chatbook/Widgets/adaptive_pane_shell.py`, `backlog/decisions/212-shared-adaptive-pane-shell.md` | The "ellipsis has no ASCII substitute" prose B1 makes false. |
| `tldw_chatbook/css/build_css.py` | `features/_roleplay.tcss` in `CSS_MODULES`; the Roleplay `ScreenOwnedSplit` (G18). |
| `tldw_chatbook/css/components/_agentic_terminal.tcss` | Nine never-composed `#personas-*` selector items deleted. |
| `tldw_chatbook/css/components/_workbench.tcss` | The dead task-523 `#personas-header.status-blocked` rule deleted (the header never takes `status-blocked` now). |
| `tldw_chatbook/css/core/_variables.tcss` | `$ds-status-warning-readable: $text-warning;`, the readable warning foreground for the Unsaved chip (sibling of `$ds-status-error-readable`). |
| `tldw_chatbook/css/tldw_cli_modular.tcss`, `tldw_chatbook/css/widget_defaults_scoped.tcss`, every `tldw_chatbook/css/screen_*.tcss` split sheet | Regenerated by `build_css.py` (the split sheets carry the variable preamble). |
| `tldw_chatbook/app.py` | `TAB_PERSONAS` import and route-map row. |
| `Tests/UI/test_personas_workbench.py`, `Tests/UI/test_personas_dictionaries.py` | The harness apps become re-exports; header re-pins; a persona-save chip test. |
| `Tests/UI/test_personas_subscription_readiness.py`, `Tests/Chat/test_console_glyphs.py`, `Tests/UI/test_unified_shell_phase6_first_time_replay.py` | Re-pins (the last one required the retired subtitle copy); the frame-glyph test. |
| `Tests/UI/test_roleplay_hostile_names.py` (TASK-34400, on dev) | Imports its paint helpers and the character seam from the harness (Task 1); its `Editing` assertion re-pinned to the item label (Task 6); extended with B1's header surfaces, the item label and the escaped server label, under styled tier 1 (Task 8). Every later slice extends it. Bootstrap-profile. |
| `Tests/UI/test_roleplay_hostile_text_surfaces.py` (TASK-34400, on dev) | Imports `click_meta_cells`/`painted_rows` from the harness (Task 1); its subtitle test re-pinned to the header view (Task 6). |
| `Tests/Architecture/test_builtin_theme_contrast.py` | Its readable-token guard also covers `ds-status-warning-readable`. |
| `Tests/UI/test_css_build_integrity.py`, `Tests/UI/test_consolidated_css_harness.py` | PersonasScreen in the CSS_PATH check (str hole fixed), the partition parameter, `_GENERATED_SHEETS`, the `_SPLIT_SHEET_OWNERS` entry. |
| `Tests/Architecture/test_module_size_ratchet.py` | PS row set to the measurement (and `app.py`'s raised only if dev grew it past B1's two lines; ruling 2); `roleplay_frame_state.py` row added. |
| `Tests/Performance/test_screen_preimport_payload_budget.py`, `backlog/decisions/097-boot-budget-ratchets.md`, `Tests/Performance/boot_budget_snapshots/*.json` | The owner-approved raise, its ledger row and the refreshed snapshots (by script). |
| `DESIGN.md`, `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`, `backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md`, `Docs/User_Guide/roleplay-chat-dictionaries.md` | G12, G4 (predicate; ADR-120 cross-reference), the guide's header delta and title. |
| `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `.github/workflows/derived-artifacts.yml` | Four fast files appended to the UI census (B1's three and TASK-34400's `test_roleplay_hostile_text_surfaces.py`, which no lane gated on dev; the shard count is read from the workflow); four bootstrap-profile files inserted into the admission-sensitive step by script (Task 11 Step 3). |
| `backlog/tasks/task-33910.2 - Roleplay-frame-B1-one-row-header-and-the-lazy-Roleplay-stylesheet.md` | Status, plan section, ACs, notes. |
| `backlog/tasks/task-33910.3 - …B2a…`, `task-33910.5 - …B3…`, `task-33910.18 - …B12…` | One dated note each (Task 13): B2a converts the full styled tier to DB seeding before its volume tests; B3 makes the blocked chip keyboard-reachable and adds it to `LEAVES_SCREEN_IDS`; B12 re-captures the guide's stale Roleplay screenshots. |

## Conventions for every task

Shell state does not persist between tool calls, so every command block re-declares what it needs. The Bash tool is zsh (no word splitting of unquoted variables), so multi-line blocks are written as `bash <<'EOF' … EOF`. These names mean:

```
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook
WT=$MAIN/.worktrees/roleplay-b1            # the slice worktree (head arm)
BASE=$MAIN/.worktrees/roleplay-b1-base     # detached paired base arm (Task 0)
PY=$MAIN/.venv/bin/python
EV=$MAIN/.worktrees/.b1-evidence           # evidence: git-ignored (.worktrees/ is ignored), outlives the session
```

Single test commands are written in full. Every `git` command uses `git -C`. "Old/New" blocks are applied with the Edit tool; each Old block matches exactly once (checked in the dry run).

Every pytest command whose count line is a gate reads it with `2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1`, never a bare `| tail -1`. When other sessions run pytest on this machine at the same time, pytest's teardown can print `PytestWarning: (rm_rf) error removing …/pytest-of-<user>/garbage-…` after the summary, and a bare `tail -1` then shows `  warnings.warn(` instead of the count (seen throughout the 2026-10-08 dry runs under load averages of 24-33, three times on gate lines). The failing-first runs keep `| tail -3`, because their Expected lines name the error text above the count; if teardown warnings push the count out of those three lines, re-read it with the grep. Use the same grep in any command you add, or add `--basetemp=$EV/bt-<name>` (as `paired.sh` does). A command whose count line is missing never passes a gate (pytest's `no tests ran` does not match the grep, so it prints nothing and fails the gate, as it should).

---

### Task 0: Preconditions, paired base arm, evidence helpers, the task start, the pre-import sign-off

**Files:**
- Create (evidence only, never committed): `$EV/restack.sh`, `$EV/b1-files.txt`, `$EV/pr_collisions.sh`, `$EV/stack-cut.txt`, `$EV/failed_ids.py`, `$EV/plug/b1_bootstrap_all.py`, `$EV/paired.sh`, `$EV/mutations.md`, `$EV/suites-roleplay.txt`, `$EV/suites-shell.txt`, `$EV/suites-css.txt`, `$EV/suites-perf.txt`, `$EV/preimport_measure.sh`, `$EV/preimport_raise.py`, `$EV/preimport-decision.md`, `$EV/owner-signoff.txt`, `$EV/base-sha.txt`, `$EV/b1-task-id.txt`, `$EV/preconditions.txt`, `$EV/pushed-sha.txt`
- Modify: `backlog/tasks/task-33910.2 - Roleplay-frame-B1-one-row-header-and-the-lazy-Roleplay-stylesheet.md`; commit this plan.

**Interfaces:**
- Consumes: nothing.
- Produces: B1 moved onto `origin/dev`; `$EV/restack.sh [--record]` (the only way B1 moves: Global Constraints, rebase protocol); `$EV/pr_collisions.sh` (every open PR touching a file in `$EV/b1-files.txt`); `$BASE` at the stack base; `$EV/stack-cut.txt` = `$EV/base-sha.txt` (the dev SHA B1 sits on); `$EV/b1-task-id.txt` (`33910.2`); `$EV/paired.sh <label> <suite-file> [pytest args…]` (failure-set diff, `recovery=` per arm); `$EV/preimport_measure.sh <base|head>` → `$EV/preimport-<arm>.json`; `$EV/preimport_raise.py` (idempotent owner-approved raise); `$EV/owner-signoff.txt` (verbatim).

This task has no production code, so it has no red/green cycle.

- [ ] **Step 1: Check the preconditions, then move B1 onto dev**

B1 was cut from B0's pre-merge head `f0a766a155` and has no commit of its own yet (the plan file is untracked). B0 (PR #2977) was rebased and amended by its review rounds, then **merged to dev on 2026-10-04 06:24Z by rebase merge**: its commits are on `origin/dev` under new SHAs (`dcffe4d36b` carries B0's notes; `83c264f286`, the PR's last review fix, was dev's head then), and the 22 pre-merge B0 commits under B1 are not ancestors of dev. B1 therefore no longer stacks on a branch in review. A plain `git rebase origin/dev` would replay those 22 commits and conflict; `git rebase --onto origin/dev f0a766a155` drops them and moves only B1's own commits (none yet). `$EV/restack.sh`, written here, is that move with its guards, and every later move reuses it. The paired base arm (Step 2) is the dev SHA B1 lands on: the new merge-base with `origin/dev`.

```bash
bash <<'OUTER'
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b1; EV=$MAIN/.worktrees/.b1-evidence
mkdir -p "$EV/tmp"
cat > "$EV/restack.sh" <<'EOF'
#!/usr/bin/env bash
# Usage: restack.sh [--record]
# Move B1's own commits onto origin/dev, record the dev SHA they now sit on
# (stack-cut.txt = base-sha.txt) and re-anchor the paired base arm. Refuses
# when CUT..HEAD holds a commit that is not B1's: a stale cut point would
# replay someone else's commits. --record skips the rebase and only records
# (after a conflicting rebase was resolved and continued by hand).
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b1
BASE=$MAIN/.worktrees/roleplay-b1-base; EV=$MAIN/.worktrees/.b1-evidence
if [ "${1:-}" != --record ]; then
  git -C "$MAIN" fetch origin --prune --quiet
  CUT=$(cat "$EV/stack-cut.txt" 2>/dev/null || echo f0a766a155)
  FOREIGN=$(git -C "$WT" log --format='%h %s' "$CUT..HEAD" | grep -vE '\(B1| B1 |TASK-33910\.2' || true)
  if [ -n "$FOREIGN" ]; then
    echo "STOP: commits in $CUT..HEAD that are not B1's (the cut point is stale):"
    echo "$FOREIGN"
    echo "B1's oldest commit: $(git -C "$WT" log --reverse --format='%h %s' -E --grep='\(B1| B1 |TASK-33910\.2' "$CUT..HEAD" | head -1)"
    echo "Write its parent's full SHA to $EV/stack-cut.txt, then re-run."
    exit 1
  fi
  git -C "$WT" rebase --onto origin/dev "$CUT"
fi
git -C "$WT" merge-base --is-ancestor origin/dev HEAD || { echo "STOP: HEAD does not contain origin/dev"; exit 1; }
NEW=$(git -C "$WT" rev-parse origin/dev)
echo "$NEW" > "$EV/stack-cut.txt"; echo "$NEW" > "$EV/base-sha.txt"
if [ -d "$BASE" ]; then git -C "$BASE" checkout --quiet --detach "$NEW"; fi
echo "B1 now on origin/dev @ $(git -C "$WT" rev-parse --short "$NEW"); own commits: $(git -C "$WT" rev-list --count "$NEW..HEAD")"
EOF
chmod +x "$EV/restack.sh"
cat > "$EV/b1-files.txt" <<'EOF'
tldw_chatbook/UI/Screens/personas_screen.py
tldw_chatbook/UI/Workbench/workbench_widgets.py
tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py
tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py
tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py
tldw_chatbook/Widgets/glyph_fallback.py
tldw_chatbook/Widgets/adaptive_pane_shell.py
tldw_chatbook/app.py
tldw_chatbook/css/build_css.py
tldw_chatbook/css/features/_roleplay.tcss
tldw_chatbook/css/components/_agentic_terminal.tcss
tldw_chatbook/css/components/_workbench.tcss
tldw_chatbook/css/core/_variables.tcss
tldw_chatbook/css/tldw_cli_modular.tcss
tldw_chatbook/css/widget_defaults_scoped.tcss
Tests/UI/test_personas_workbench.py
Tests/UI/test_personas_dictionaries.py
Tests/UI/test_personas_subscription_readiness.py
Tests/UI/test_unified_shell_phase6_first_time_replay.py
Tests/UI/test_roleplay_hostile_names.py
Tests/UI/test_roleplay_hostile_text_surfaces.py
Tests/UI/test_css_build_integrity.py
Tests/UI/test_consolidated_css_harness.py
Tests/Chat/test_console_glyphs.py
Tests/Architecture/test_module_size_ratchet.py
Tests/Architecture/test_builtin_theme_contrast.py
Tests/Performance/test_screen_preimport_payload_budget.py
Tests/Performance/boot_budget_snapshots/preimport_payload.json
Tests/Performance/test_boot_css_byte_budget.py
Tests/Performance/test_textual_css_fastpath.py
backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md
backlog/decisions/097-boot-budget-ratchets.md
backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md
backlog/decisions/212-shared-adaptive-pane-shell.md
DESIGN.md
Docs/User_Guide/roleplay-chat-dictionaries.md
scripts/ui_pr_gate_census.txt
scripts/check_ui_pr_gate_census.py
.github/workflows/derived-artifacts.yml
EOF
cat > "$EV/pr_collisions.sh" <<'EOF'
#!/usr/bin/env bash
# Usage: pr_collisions.sh -> one line per OPEN pull request that touches a file
# in b1-files.txt (B1's edits plus the budget files its gates read) or a
# generated split sheet. Files are read through the paginated API, so a PR with
# more than 100 files is read whole.
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; EV=$MAIN/.worktrees/.b1-evidence
cd "$MAIN/.worktrees/roleplay-b1"
mkdir -p "$EV/tmp"
gh pr list --state open --limit 300 --json number,title -q '.[] | "\(.number)\t\(.title)"' > "$EV/tmp/open-prs.tsv"
echo "open PRs scanned: $(wc -l < "$EV/tmp/open-prs.tsv" | tr -d ' ')"
while IFS=$'\t' read -r N TITLE; do
  gh api --paginate "repos/{owner}/{repo}/pulls/$N/files" --jq '.[].filename' > "$EV/tmp/pr-$N-files.txt"
  HITS=$(grep -Fxf "$EV/b1-files.txt" "$EV/tmp/pr-$N-files.txt" | tr '\n' ' ' || true)
  SPLITS=$(grep -c '^tldw_chatbook/css/screen_.*\.tcss$' "$EV/tmp/pr-$N-files.txt" || true)
  NOTE=""; [ "$SPLITS" = 0 ] || NOTE=" (+$SPLITS split sheets)"
  if [ -n "$HITS" ] || [ -n "$NOTE" ]; then echo "#$N $TITLE :: $HITS$NOTE"; fi
done < "$EV/tmp/open-prs.tsv"
EOF
chmod +x "$EV/pr_collisions.sh"
git -C "$MAIN" fetch origin --prune --quiet
cd "$WT"
echo "PR #2977 (B0): $(gh pr view 2977 --json state,mergedAt -q '.state+" "+.mergedAt')"
echo "B0 on dev: $(git -C "$WT" log --format='%h %s' --grep='TASK-33910\.1)' -1 origin/dev)"
CUT=$(cat "$EV/stack-cut.txt" 2>/dev/null || echo f0a766a155)
echo "cut point:       $(git -C "$WT" rev-parse --short "$CUT")"
echo "B1 head:         $(git -C "$WT" rev-parse --short HEAD)"
echo "B1-only commits: $(git -C "$WT" rev-list --count "$CUT..HEAD")"
echo "pre-merge B0 commits not on dev: $(git -C "$WT" rev-list --count "origin/dev..$CUT")"
for T in 34400 33622.14 26983 27000 33790; do
  F=$(git -C "$MAIN" ls-tree --name-only origin/dev backlog/tasks/ | grep -F "backlog/tasks/task-$T - ")
  echo "TASK-$T: $(git -C "$MAIN" show "origin/dev:$F" | grep -m1 '^status:')"
done
F=$(git -C "$MAIN" ls-tree --name-only origin/dev backlog/tasks/ | grep -F "backlog/tasks/task-33790 - ")
echo "TASK-33790 AC#4: $(git -C "$MAIN" show "origin/dev:$F" | grep -m1 -oE '^- \[.\] #4')"
git -C "$MAIN" cat-file -e origin/dev:tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py && echo "roleplay_draft_guard.py is on dev"
git -C "$WT" status --short | head
"$EV/pr_collisions.sh"
OUTER
```

Expected: `PR #2977 (B0): MERGED 2026-10-04T06:24:29Z`; `B0 on dev: dcffe4d36b chore(backlog): B0 notes after the final rebase and fix wave (TASK-33910.1)`; `cut point: f0a766a155` and `B1 head: f0a766a155` (first run); `B1-only commits: 0`; `pre-merge B0 commits not on dev: 22`; on `8d502ba250` (as on `a793acbef5` and `8c4dfe59a2`), `TASK-34400: status: Done`, `TASK-33622.14: status: Done`, `TASK-26983: status: Done`, `TASK-27000: status: Done`, `TASK-33790: status: To Do` and `TASK-33790 AC#4: - [x] #4`; `roleplay_draft_guard.py is on dev`; only `?? Docs/superpowers/plans/2026-10-03-roleplay-b1-one-row-header.md` in `status`; then the scanned-PR count and one line per open PR that touches a B1 file. Run on 2026-10-08 against `8d502ba250`, the scan listed 11 of 30 open PRs (against `a793acbef5` the same day, 13 of 32: #3028 and #3031 have merged since). The ones that move B1's own inputs: #2862 (personal context: PS outside B1's hunks, the pre-import constants and snapshot, the module-size ratchet, `build_css.py`, `_agentic_terminal.tcss`, `_variables.tcss`, `test_css_build_integrity.py`, `app.py`, the bundle, `widget_defaults_scoped.tcss` and six split sheets), #2563 (goal runs: PS, `glyph_fallback.py`, `app.py`, `_agentic_terminal.tcss`, `_variables.tcss`, the bundle), #3045 (non-Console efficiency: PS and `test_personas_workbench.py`), #3023 (DeepSeek trace and Console Send: PS, `app.py` and the boot-CSS budget test), #3036 (Console permission approvals: `build_css.py`, `_agentic_terminal.tcss`, `_variables.tcss`, the bundle and seven split sheets), #3022 (the PR-gate census and its checker), #3039 (the merge queue: the workflow) and #2868 and #3010 (`build_css.py` and the bundle; #2868 also the pre-import snapshot); the rest (#3029, #2921) touch only ADR-120 or `app.py`. (On 2026-10-04 against `8c4dfe59a2` it listed 11 of 28; #2995, #3021, #3011, #2930, #2918 and #2890 have left the list since, and #2953, which re-indented `_agentic_terminal.tcss`, merged before that run.) Then move B1 onto dev:

```bash
bash /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/restack.sh
```

Expected: `B1 now on origin/dev @ <dev's short SHA>; own commits: 0` (the untracked plan file is untouched by the move; the base arm does not exist yet, so nothing is re-anchored). From here on `$EV/stack-cut.txt` is the dev SHA every later move starts from.

Write what you saw to `$EV/preconditions.txt`. Consequences to record there:
- TASK-34400 `Done` → every existing Roleplay surface already paints untrusted text literally (the Inspector Statics, the Tag filter button, every toast, the old subtitle), `Tests/UI/test_roleplay_hostile_names.py` (22 tests) and `Tests/UI/test_roleplay_hostile_text_surfaces.py` (81) exist, and B1 extends them for its own new surfaces only (Tasks 1, 6 and 8). If it reads anything but `Done`, STOP and report: the plan assumes TASK-34400's fixes and test files.
- TASK-33790 `To Do` (the B1 task file lists it as a prerequisite: "should land first") → B1 proceeds anyway, and the PR body says why: B1 depended only on its AC#4 (Roleplay toasts paint square brackets literally), which TASK-34400 delivered on dev and ticked. Its other ACs (a visible 200-character persona-name limit before Save, a readable validation message, the same for a tool-rule name over 512 characters, and their tests) are still open on dev; they are not B1's, and B1 does not fix them. If AC#4 reads unticked, STOP and report.
- TASK-33622.14 `Done` → its leave and Ctrl+Q guard lives in `UI/Persona_Modules/roleplay_draft_guard.py`, and Task 6 routes its decisions through the one predicate (R24, G4). If the module is missing, STOP and report: the plan assumes it.
- TASK-26983 and TASK-27000 `Done` → the formatter batches ran before B1, which spec §5.6/K19 allows ("before B1 or after B12, never between"); B1 keeps the files format-clean (Global Constraints). If either reads anything but `Done`, STOP and ask the owner through the controller whether to run it first: once B1 starts it is blocked until B12 (TASK-33910.18) merges, and Task 13 then adds that dated note to it.
- Every PR the collision scan lists: if it merges before B1 does, follow the rebase protocol. #2862, #2563, #3045 or #3023 merging moves PS, its row, the glyph map, the boot snapshots or the pre-import limits (#2862's diff re-pins all three pre-import constants), and the raise and ratchet scripts decide what B1's head then exceeds; the owner's rulings cover that (ruling 1 within its bound, ruling 2 for ratchet rows).

- [ ] **Step 2: Create the paired base arm and the evidence directory**

```bash
bash <<'EOF'
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b1
BASE=$MAIN/.worktrees/roleplay-b1-base; EV=$MAIN/.worktrees/.b1-evidence
mkdir -p "$EV"
SB=$(cat "$EV/stack-cut.txt")   # the dev SHA Step 1 moved B1 onto (B1 has no commit of its own yet)
test "$SB" = "$(git -C "$WT" rev-parse HEAD)" || { echo "STOP: HEAD is not the stack base; redo Step 1"; exit 1; }
test "$SB" = "$(git -C "$WT" merge-base HEAD origin/dev)" || { echo "STOP: the stack base is not the merge-base with origin/dev"; exit 1; }
if [ -d "$BASE" ]; then git -C "$BASE" checkout --quiet --detach "$SB"; else git -C "$MAIN" worktree add --detach "$BASE" "$SB"; fi
echo "$SB" > "$EV/base-sha.txt"
git -C "$BASE" log --oneline -1
git -C "$MAIN" check-ignore -q "$EV" && echo "evidence dir is git-ignored"
PS=tldw_chatbook/UI/Screens/personas_screen.py
LINES=$(wc -l < "$BASE/$PS" | tr -d ' ')
ROW=$(grep -o '"'"$PS"'": [0-9]*' "$BASE/Tests/Architecture/test_module_size_ratchet.py" | grep -o '[0-9]*$')
echo "PS lines: $LINES ; ratchet row: $ROW"
if [ "$LINES" -le "$ROW" ] && [ $((ROW - LINES)) -le 50 ]; then echo "PS row holds (lines <= row <= lines + 50)"; else echo "STOP: the PS row is red at base"; exit 1; fi
EOF
```

Expected: the base worktree's head line (dev's head), `evidence dir is git-ignored`, `PS lines: 16528 ; ratchet row: 16528` on `8d502ba250` (as on `a793acbef5` and `8c4dfe59a2`), and `PS row holds (lines <= row <= lines + 50)`. The check is the ratchet's own invariant (`lines ≤ row ≤ lines + _SLACK_TOLERANCE_LINES`, 50), not equality: a legal state where another PR left slack must not stop B1. A `STOP` line means the PS row is red at base, and the spec's FU-3 precondition for B1 is a green PS row: report it.

- [ ] **Step 3: Write the evidence helpers and the suite lists**

```bash
bash <<'OUTER'
set -euo pipefail
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence
cat > "$EV/failed_ids.py" <<'EOF'
"""Print the node id of every failed or errored test case in a JUnit XML file."""
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
for case in root.iter("testcase"):
    if case.find("failure") is not None or case.find("error") is not None:
        print(f'{case.get("classname")}::{case.get("name")}')
EOF
mkdir -p "$EV/plug"
cat > "$EV/plug/b1_bootstrap_all.py" <<'EOF'
"""B1 evidence plugin (copy of B0's): import the app at collection time and keep
the bootstrap profile, so mounted Tests/UI suites do not fail at setup with
``RecoveryRequired: raw_source_selection_changed`` on BOTH arms
(backlog/docs/lessons-testing-evidence.md). It edits no test."""
import pytest


def pytest_collection_modifyitems(session, config, items):
    import tldw_chatbook.app  # noqa: F401

    for item in items:
        item.add_marker(pytest.mark.bootstrap_profile)
EOF
cat > "$EV/paired.sh" <<'EOF'
#!/usr/bin/env bash
# Usage: [PAIRED_PLUGIN=0] paired.sh <label> <suite-file> [extra pytest args...]
# Same pytest command on the base arm, then the head arm; prints the failures
# that are NEW on head. recovery=<n> must be single digits on both arms.
# No `set -u`: macOS /bin/bash 3.2 treats an empty "$@" as unbound.
set -o pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook
PY=$MAIN/.venv/bin/python
EV=$MAIN/.worktrees/.b1-evidence
LABEL=$1; SUITES=$2; shift 2
PLUGIN_ARGS="-p b1_bootstrap_all"
if [ "${PAIRED_PLUGIN:-1}" = 0 ]; then PLUGIN_ARGS=""; fi
for ARM in base head; do
  if [ "$ARM" = base ]; then T=$MAIN/.worktrees/roleplay-b1-base; else T=$MAIN/.worktrees/roleplay-b1; fi
  (cd "$T" && PYTHONPATH="$EV/plug" "$PY" -m pytest $(cat "$SUITES") $PLUGIN_ARGS "$@" -p no:cacheprovider -q -rf --timeout=300 \
      --basetemp="$EV/run-$LABEL-$ARM" --junitxml="$EV/$LABEL-$ARM.xml" > "$EV/$LABEL-$ARM.log" 2>&1)
  "$PY" "$EV/failed_ids.py" "$EV/$LABEL-$ARM.xml" | sort > "$EV/$LABEL-$ARM.failed"
  echo "$LABEL $ARM: $(grep -E '[0-9]+ (passed|failed|error)' "$EV/$LABEL-$ARM.log" | tail -1) recovery=$(grep -c 'RecoveryRequired:' "$EV/$LABEL-$ARM.log") plugin=${PLUGIN_ARGS:-off}"
done
echo "new failures on head [$LABEL] (must be empty):"
comm -13 "$EV/$LABEL-base.failed" "$EV/$LABEL-head.failed"
echo "(end of new failures [$LABEL])"
EOF
chmod +x "$EV/paired.sh"
printf '# Named mutations (B1)\n\n| Task | Mutation | Test that went red | Restored and green |\n|---|---|---|---|\n' > "$EV/mutations.md"
cat > "$EV/suites-roleplay.txt" <<'EOF'
Tests/UI/test_personas_workbench.py
Tests/UI/test_personas_dictionaries.py
Tests/UI/test_personas_lore.py
Tests/UI/test_personas_deferred_center_views.py
Tests/UI/test_personas_workbench_foundation.py
Tests/UI/test_personas_library_rail_focus_outline.py
Tests/UI/test_personas_workbench_state.py
Tests/UI/test_personas_subscription_readiness.py
Tests/UI/test_personas_center_canvas_layout.py
Tests/UI/test_personas_library_toolbar_layout.py
Tests/UI/test_personas_character_attach.py
Tests/UI/test_personas_character_editor_avatar.py
Tests/UI/test_personas_character_world_books_screen.py
Tests/UI/test_personas_editor_save_in_place.py
Tests/UI/test_personas_expression_slots.py
Tests/UI/test_personas_generation_wiring.py
Tests/UI/test_personas_library_scale.py
Tests/UI/test_personas_persona_visual_authoring.py
Tests/UI/test_personas_persona_visual_identity_pack.py
Tests/UI/test_persona_policy_rules_editor.py
Tests/UI/test_actor_pack_creation_workflow.py
Tests/UI/test_actor_pack_recovery_seam.py
Tests/UI/test_buddy_character_review.py
Tests/UI/test_petdex_import_review.py
Tests/UI/test_product_maturity_phase1_empty_setup_states.py
Tests/UI/test_roleplay_quit_guard.py
Tests/UI/test_unified_shell_phase6_first_time_replay.py
Tests/UI/test_personas_inspector_pane.py
Tests/UI/test_personas_preview.py
Tests/UI/test_roleplay_hostile_names.py
Tests/UI/test_roleplay_hostile_text_surfaces.py
Tests/Backup_Recovery/test_persona_visual_lifetimes.py
Tests/Backup_Recovery/test_visual_identity_lifetimes.py
Tests/Architecture/test_quit_flow_prompt_choke_point.py
EOF
cat > "$EV/suites-shell.txt" <<'EOF'
Tests/UI/test_destination_shells.py
Tests/UI/test_destination_visual_parity_correction.py
Tests/UI/test_destination_headers.py
Tests/UI/test_destination_header_compact_floor.py
Tests/UI/test_workbench_widgets.py
Tests/UI/test_screen_footer_hints.py
Tests/UI/test_master_shell_navigation.py
Tests/UI/test_master_shell_design_system_contract.py
Tests/UI/test_workbench_pane_focus.py
Tests/UI/test_theme_contrast.py
Tests/UI/test_adaptive_pane_shell.py
Tests/UI/test_destination_rail_row.py
Tests/UI/test_library_honesty_accessibility.py
Tests/Chat/test_console_glyphs.py
Tests/UI/test_console_context_rail_header.py
Tests/UI/test_console_environment_section.py
Tests/UI/test_console_inspector_section.py
Tests/UI/test_console_manual_unread.py
Tests/UI/test_console_rail_handle.py
Tests/UI/test_console_turn_file_card_notes.py
Tests/UI/test_console_workspace_context_rail.py
Tests/UI/test_console_workspace_tree.py
Tests/Workspaces/test_conversation_attention.py
Tests/UI/test_settings_configuration_hub.py
Tests/UI/test_settings_appearance_defaults.py
Tests/Chat/test_console_appearance.py
Tests/UI/test_change_review_screen.py
Tests/UI/test_console_composer_blink_wrap.py
Tests/UI/test_console_turn_file_card.py
Tests/UI/test_console_workspace_controller.py
EOF
cat > "$EV/suites-css.txt" <<'EOF'
Tests/UI/test_css_build_integrity.py
Tests/UI/test_consolidated_css_harness.py
Tests/UI/test_css_staleness_manifest.py
Tests/UI/test_component_pattern_governance.py
Tests/UI/test_design_token_governance.py
Tests/UI/test_widget_css_consolidation.py
Tests/Architecture/test_module_size_ratchet.py
Tests/Architecture/test_builtin_theme_contrast.py
Tests/Architecture/test_timer_path_static_update_inventory.py
EOF
cat > "$EV/suites-perf.txt" <<'EOF'
Tests/Performance/test_textual_css_fastpath.py
Tests/Performance/test_boot_css_byte_budget.py
Tests/Performance/test_ui_ready_module_census.py
Tests/Performance/test_app_import_weight.py
Tests/Performance/test_ui_latency_guardrails.py
Tests/Performance/test_screen_leaks.py
Tests/Performance/test_screen_preimport_payload_budget.py
EOF
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
for f in $(cat "$EV"/suites-*.txt); do [ -f "$f" ] || echo "MISSING $f"; done
echo "suite files checked"
OUTER
```

Expected: only `suite files checked`. The `roleplay` list is every module that imports the Roleplay test apps, the §5.7.4 Roleplay suites, every suite that asserts copy B1 changes (`test_unified_shell_phase6_first_time_replay.py` required the retired subtitle line), the suites of the Roleplay modules B1 edits (the preview controller, TASK-33622.14's guard and its quit-flow choke-point scan), the Inspector pane's suite, TASK-34400's two hostile-text suites (B1 moves their paint helpers, re-pins the two tests that asserted the retired subtitle, and extends the first) and the two lifetime suites that build `PersonasScreen` directly; `shell` holds the destination/header suites and every suite that turns ASCII glyph mode on or calls `resolve_glyph`/`resolve_glyph_text` (the glyph-map change, Task 2: Settings' appearance hub and defaults, Console appearance, Change Review, the composer blink, the turn file card, the workspace controller); `css` adds the theme-contrast guard on readable tokens and the timer-path census (red on dev, so Task 12 Step 4 also compares its unclassified sites as a B1 floor).

- [ ] **Step 4: Write the pre-import helpers (measure and raise)**

`preimport_measure.sh` is B0's, re-pointed at B1's arms. `preimport_raise.py` is B0's stateless, idempotent raise with B1's module, marker and comment prefix: it compares the head measurement with the limits in the BASE arm's copy of the test file, raises exactly the constants the head exceeds (keeping B0's `#: B0 …` comment above the constant and replacing only a `#: B1 …` line), leaves every other constant at the HEAD file's own value (so an upstream tightening that arrived with a rebase is never silently undone), and rewrites B1's one ledger row (unmerged, so rewriting it before merge is fine). It refuses to run when `base-sha.txt` is not `stack-cut.txt` or not an ancestor of `HEAD` (a stale base arm), and when a measurement exceeds the head file's limit without exceeding the base's (the head file was tightened after the base arm was taken). It dates its constant comment and its ledger row from the owner's sign-off (2026-10-04, the first field of `$EV/owner-signoff.txt`), never from the day it runs, as `ratchet_rows.py` uses the fixed ruling date: a re-run after a rebase on a later day rewrites nothing. The row's sign-off cell marks as verbatim only the owner's answer (the sign-off's last quoted string, `"Expand it (Rec.)"`); the question is paraphrased around it, never presented as the owner's words, and no quotes nest.

```bash
bash <<'OUTER'
set -euo pipefail
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence
mkdir -p "$EV/tmp"
cat > "$EV/preimport_measure.sh" <<'EOF'
#!/usr/bin/env bash
# Usage: preimport_measure.sh <base|head>  ->  $EV/preimport-<arm>.json + a summary line
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b1-evidence
ARM=$1
if [ "$ARM" = base ]; then T=$MAIN/.worktrees/roleplay-b1-base; else T=$MAIN/.worktrees/roleplay-b1; fi
cd "$T"
PYTHONPATH="$T" "$PY" - "$EV/tmp" > "$EV/preimport-$ARM.json" <<'PY'
import json, sys, tempfile
from pathlib import Path
from Tests.Performance.test_screen_preimport_payload_budget import _run_census
census = _run_census(Path(tempfile.mkdtemp(dir=sys.argv[1])))
routes = {row["route"]: [row["added_modules"], row["added_loc"]] for row in census["routes"]}
print(json.dumps({
    "modules": census["pass_added_modules"],
    "loc": census["pass_added_loc"],
    "fattest_route_loc": max(loc for _, loc in routes.values()),
    "routes": routes,
    "module_set": sorted({m for row in census["routes"] for m in row["modules"]}),
}))
PY
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], "| modules", d["modules"], "| LOC", d["loc"], "| roleplay (ccp)", d["routes"]["ccp"][0], "mods /", d["routes"]["ccp"][1], "LOC | fattest route", d["fattest_route_loc"], "LOC")' "$EV/preimport-$ARM.json" "$ARM"
EOF
chmod +x "$EV/preimport_measure.sh"
cat > "$EV/preimport_raise.py" <<'EOF'
"""Raise exactly the screen pre-import limits B1's head exceeds (owner-approved, spec Q10).

Stateless and idempotent: run it after `preimport_measure.sh head` before every
commit from Task 6 on. Inputs: $EV/preimport-base.json, $EV/preimport-head.json,
$EV/base-sha.txt, $EV/stack-cut.txt, $EV/b1-task-id.txt, $EV/owner-signoff.txt.
A constant it does not raise keeps the head file's current value. The comment and
the ledger row carry the sign-off's own date (the first field of
owner-signoff.txt), never the day the script runs, so a re-run after a rebase on
a later day rewrites nothing. The row quotes as verbatim only the owner's answer
(the sign-off's last quoted string); the question is paraphrased, not quoted.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

MAIN = Path("/Users/macbook-dev/Documents/GitHub/tldw_chatbook")
WT = MAIN / ".worktrees/roleplay-b1"
EV = MAIN / ".worktrees/.b1-evidence"
TEST_REL = "Tests/Performance/test_screen_preimport_payload_budget.py"
TEST = WT / TEST_REL
ADR = WT / "backlog/decisions/097-boot-budget-ratchets.md"
NEW_MODULE = "tldw_chatbook.UI.Persona_Modules.roleplay_frame_state"
NAMES = ("MAX_PASS_ADDED_MODULES", "MAX_PASS_ADDED_LOC", "MAX_SINGLE_ROUTE_ADDED_LOC")
#: The bounds the owner's Q10 answer approves (Task 0 Step 6). Beyond them: ask again.
CAP = {"modules": 1, "loc": 1000, "fattest_route_loc": 1000}


def limits(text: str) -> dict[str, int]:
    found = {}
    for name in NAMES:
        values = re.findall(rf"^{name} = ([\d_]+)$", text, re.M)
        if len(values) != 1:
            sys.exit(f"expected exactly one `{name} = <int>` line, found {values}")
        found[name] = int(values[0].replace("_", ""))
    return found


base = json.loads((EV / "preimport-base.json").read_text(encoding="utf-8"))
head = json.loads((EV / "preimport-head.json").read_text(encoding="utf-8"))
added = sorted(set(head["module_set"]) - set(base["module_set"]))
removed = sorted(set(base["module_set"]) - set(head["module_set"]))
if added not in ([], [NEW_MODULE]) or removed:
    sys.exit(f"STOP: unexpected module delta added={added} removed={removed}: something new became eager")
for key, cap in CAP.items():
    if head[key] - base[key] > cap:
        sys.exit(f"STOP: {key} grew {head[key] - base[key]} (> the approved {cap}); ask the owner again")

base_sha = (EV / "base-sha.txt").read_text(encoding="utf-8").strip()
cut_sha = (EV / "stack-cut.txt").read_text(encoding="utf-8").strip()
if base_sha != cut_sha:
    sys.exit(f"STOP: base-sha.txt {base_sha[:10]} is not stack-cut.txt {cut_sha[:10]}: run restack.sh --record")
is_ancestor = subprocess.run(["git", "-C", str(WT), "merge-base", "--is-ancestor", base_sha, "HEAD"])
if is_ancestor.returncode != 0:
    sys.exit(f"STOP: the base arm {base_sha[:10]} is not an ancestor of HEAD: run restack.sh")
base_text = subprocess.run(
    ["git", "-C", str(WT), "show", f"{base_sha}:{TEST_REL}"],
    capture_output=True, text=True, check=True,
).stdout
original = limits(base_text)
measured = {
    "MAX_PASS_ADDED_MODULES": head["modules"],
    "MAX_PASS_ADDED_LOC": head["loc"],
    "MAX_SINGLE_ROUTE_ADDED_LOC": head["fattest_route_loc"],
}
raised = {name: (original[name], measured[name]) for name in NAMES if measured[name] > original[name]}
task_id = (EV / "b1-task-id.txt").read_text(encoding="utf-8").strip()
signoff_path = EV / "owner-signoff.txt"
signoff = signoff_path.read_text(encoding="utf-8").strip() if signoff_path.exists() else ""
if raised and not signoff:
    sys.exit(f"STOP: {sorted(raised)} need raising and owner-signoff.txt is empty (Task 0 Step 6)")
signed = re.match(r"(\d{4}-\d{2}-\d{2}),", signoff)
if raised and not signed:
    sys.exit("STOP: owner-signoff.txt does not start with the sign-off's ISO date (Task 0 Step 6)")
signed_on = signed[1] if signed else ""
answer = re.search(r'"([^"]+)"$', signoff)
if raised and not answer:
    sys.exit("STOP: owner-signoff.txt does not end with the owner's quoted answer (Task 0 Step 6)")

text = TEST.read_text(encoding="utf-8")
current = limits(text)
tightened = [name for name in NAMES if name not in raised and measured[name] > current[name]]
if tightened:
    sys.exit(f"STOP: {tightened} exceed the head file's limit but not the base arm's: re-anchor, then re-measure")
for name in NAMES:
    value = raised[name][1] if name in raised else current[name]
    comment = (
        f"#: B1 TASK-{task_id}: {original[name]:_} -> {value:_} ({signed_on}), owner-approved; ADR-097 ledger.\n"
        if name in raised
        else ""
    )
    text = re.sub(
        rf"^(?:#: B1 TASK-[^\n]*\n)?{name} = [\d_]+$",
        lambda _m, c=comment, n=name, v=value: f"{c}{n} = {v:_}",
        text,
        count=1,
        flags=re.M,
    )
TEST.write_text(text, encoding="utf-8")

lines = ADR.read_text(encoding="utf-8").splitlines(keepends=True)
marker = f"Roleplay frame B1 (TASK-{task_id})"
lines = [line for line in lines if marker not in line]
if raised:
    start = next(i for i, line in enumerate(lines) if line.startswith("## Exception ledger"))
    header = next(i for i in range(start, len(lines)) if lines[i].startswith("| date | guard |"))
    end = header
    while end + 1 < len(lines) and lines[end + 1].startswith("|"):
        end += 1
    route_base, route_head = base["routes"]["ccp"], head["routes"]["ccp"]
    cause = (
        f"{marker}: new pure module `{NEW_MODULE}` (the header state, spec section 5.11), "
        "imported at module scope by `UI/Screens/personas_screen.py`, so the Roleplay (`ccp`) "
        f"route adds it. Same-session paired arms (base `{base_sha[:10]}`): pass {base['modules']} "
        f"modules / {base['loc']:,} LOC -> {head['modules']} / {head['loc']:,}; ccp route "
        f"{route_base[0]} / {route_base[1]:,} -> {route_head[0]} / {route_head[1]:,}. Defer was "
        "ruled out by the owner (spec Q10: a lazy import moves the cost onto the first Ctrl+4); "
        "nothing on the Roleplay route folds into it in B1."
    )
    said = answer[1].replace("|", "\\|")
    sign_off = (
        f"Owner, {signed_on}, asked whether the expanded limits also cover B1's one new module "
        f"(`roleplay_frame_state`); answer, verbatim: \"{said}\""
    )
    row = (
        f"| {signed_on} | screen pre-import payload | "
        + " / ".join(f"`{name}`" for name in raised)
        + " | "
        + " / ".join(f"{old:,} → {new:,}" for old, new in raised.values())
        + f" | {cause} | {sign_off} |\n"
    )
    lines.insert(end + 1, row)
ADR.write_text("".join(lines), encoding="utf-8")
print(
    "raised: " + (", ".join(f"{n} {o} -> {v}" for n, (o, v) in raised.items()) or "nothing (head within base limits)")
    + f" | ledger row: {'written' if raised else 'none'}"
)
EOF
echo "pre-import helpers written"
OUTER
```

Expected: `pre-import helpers written`. Before Task 6 nothing imports the new module: `preimport_raise.py` prints `raised: nothing (head within base limits) | ledger row: none`.

- [ ] **Step 5: Start the B1 task (already filed in PR #2960 — never re-file) and commit this plan**

Five-digit ids break `backlog task edit` (lessons-backlog-hygiene, TASK-15463), so edit the file directly.

```bash
bash <<'EOF'
set -euo pipefail
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
B1="backlog/tasks/task-33910.2 - Roleplay-frame-B1-one-row-header-and-the-lazy-Roleplay-stylesheet.md"
test -f "$B1" || { echo "MISSING B1 task"; exit 1; }
echo "33910.2" > "$EV/b1-task-id.txt"
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - "$B1" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
assert text.count("status: To Do\n") == 1, "B1 task is not To Do"
text = text.replace("status: To Do\n", "status: In Progress\n", 1)
text = text.replace("assignee: []\n", "assignee:\n  - '@claude'\n", 1)
if "## Implementation Plan" not in text:
    text = text.rstrip("\n") + (
        "\n\n## Implementation Plan\n\n<!-- SECTION:PLAN:BEGIN -->\n"
        "Docs/superpowers/plans/2026-10-03-roleplay-b1-one-row-header.md, Tasks 0 to 13.\n"
        "<!-- SECTION:PLAN:END -->\n"
    )
path.write_text(text, encoding="utf-8")
print("B1 task started")
PY
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_backlog_task_files.py | tail -1
git add "$B1" Docs/superpowers/plans/2026-10-03-roleplay-b1-one-row-header.md
git commit -m "chore(backlog): start TASK-33910.2 (Roleplay frame B1) with its implementation plan" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: `B1 task started`, the task-files check passing, and the commit line.

- [ ] **Step 6: Record the owner's pre-import sign-off (ruling 1; spec Q10), before Task 6 commits**

Task 6 is the first commit in which PS imports `roleplay_frame_state`, and from that commit the required perf guard `test_preimport_pass_payload_stays_within_budget` is red until a constant is raised. The owner answered the question on 2026-10-04: asked "does 'just expand the limits' also cover B1's one new module (roleplay_frame_state, +1 module / about +380 lines)?", the owner chose "Expand it (Rec.)". Nothing is asked again; this step writes the decision record and the sign-off whose date and answer `preimport_raise.py` puts in the ADR-097 ledger row (only the answer, the last quoted string, is quoted there as the owner's words):

```bash
bash <<'OUTER'
set -euo pipefail
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence
cat > "$EV/preimport-decision.md" <<'EOF'
# B1 pre-import growth: ADR-097 order applied

- Defer: a lazy import of roleplay_frame_state inside PS methods is the spec's Q10 option (c), which the owner rejected (it moves the cost onto the first Ctrl+4 and works against PERF-22). The module is the header's state, read at compose and on every header paint.
- Shed: nothing on the Roleplay (ccp) route folds into it in B1; the spec's planned shed (library_emergency_return -> StageReturnBar) is B0's payback and waits for PR #2862.
- Therefore: an owner-signed ledger row for the measured growth (spec Q10 option a), raised in the same commit as Task 6, re-checked before every later commit by $EV/preimport_raise.py.
- Owner sign-off: 2026-10-04, "Expand it (Rec.)" (ruling 1). Bound: exactly the constants B1's head exceeds, each raised to the paired-arm measurement, one ADR-097 ledger row, within +1 module and +1,000 lines (pass and largest route). Growth beyond the bound goes back to the owner.
EOF
cat > "$EV/owner-signoff.txt" <<'EOF'
2026-10-04, asked "does 'just expand the limits' also cover B1's one new module (roleplay_frame_state, +1 module / about +380 lines)?": "Expand it (Rec.)"
EOF
test -s "$EV/owner-signoff.txt" && echo "owner sign-off recorded: $(cat "$EV/owner-signoff.txt")"
OUTER
```

Expected: `owner sign-off recorded: 2026-10-04, asked … "Expand it (Rec.)"`. The bound is enforced by `preimport_raise.py`'s `CAP` and its module-delta check: a second added module, or more than +1,000 lines on the pass or the largest route, prints `STOP:` and the raise does not happen. That STOP is the one pre-import stop left: report it to the controller, which takes it to the owner (a subagent's or the controller's own approval is never sign-off). The dry runs on `8c4dfe59a2`, `a793acbef5` and `8d502ba250` measured the growth well inside the bound (+1 module, `roleplay_frame_state`; +379 lines on the pass and on the Roleplay route; the largest route unchanged; Task 6 Step 9).

B1 is a visible slice, so Task 12 Step 8 still sends the owner base/head PNG pairs at 80x24, 120x36, 160x45 and 220x55 for approval (ADR-007) and waits: that STOP stays.

---
### Task 1: The Roleplay frame harness — two styled tiers, the size matrix, painted-geometry helpers

Created first: every later geometry test runs on it. B0 stayed Library-only, so the harness is new (spec §5.7.1, MF-03). The delegating `PersonasTestApp` moves here verbatim from `test_personas_workbench.py` (the superset copy: the `test_personas_dictionaries.py` copy only lacked `_ensure_tts_profile_service`, whose absence and a `None` return behave the same in `PersonasScreen._character_tts_profile_service`), and both old modules re-export it. `StyledPersonasTestApp` becomes `StyledRoleplayMockApp`: the boot bundle **plus every split sheet** instead of the bundle alone. Measured while planning: re-pointing it is neutral at base (its users failed the same 2 tests on both arms, both pre-existing).

The harness is also the one home of the Roleplay paint helpers. TASK-34400 (on dev) defined `painted_rows`, `click_meta_cells`, `settle` and `wait_until` inside `Tests/UI/test_roleplay_hostile_names.py`, with Google-style docstrings and typed parameters, and `Tests/UI/test_roleplay_hostile_text_surfaces.py` imports two of them from there. They move here unchanged (byte-identical signatures, docstrings and bodies, checked by AST in the 2026-10-08 dry run). That module's `_seed_characters` patches the same three seams as this harness's `seed_mock_characters` (same patches; the harness copy has typed parameters and a longer docstring), so its six calls are renamed to `seed_mock_characters` and its copy is deleted; both hostile-text modules then import the helpers from here, so no helper exists twice. Their 22 + 81 tests must stay green (Step 5).

**Files:**
- Create: `Tests/UI/roleplay_frame_harness.py`, `Tests/UI/test_roleplay_frame_harness.py`
- Modify: `Tests/UI/test_personas_workbench.py` (the two harness classes become an import), `Tests/UI/test_personas_dictionaries.py` (same), `Tests/UI/test_roleplay_hostile_names.py` (its four paint helpers and `_seed_characters` become imports from the harness), `Tests/UI/test_roleplay_hostile_text_surfaces.py` (imports `click_meta_cells` and `painted_rows` from the harness)

**Interfaces:**
- Consumes: `Tests.UI.consolidated_css.APP_STYLESHEETS, CSS_DIR, ConsolidatedCSSApp`; `Tests.UI.app_factory._build_test_app(configured_default=...)`; `Tests.app_module_patches.patch_app_global`; `Tests.UI.test_personas_dictionaries.patch_character_paging`.
- Produces (later tasks import these by name):
  - `ROLEPLAY_SHEET: Path` (`CSS_DIR / "screen_feature_roleplay.tcss"`), `ROLEPLAY_SIZES: tuple[tuple[int, int], ...]`, `size_matrix` (parametrize mark → `roleplay_size`), `STYLED_TIERS = ("mock", "full")`, `styled_tiers` (parametrize mark → `styled_tier`);
  - `RoleplayMockApp(mock_app_instance)` (unstyled), `StyledRoleplayMockApp(RoleplayMockApp)` with `CSS_PATH = [str(path) for path in APP_STYLESHEETS]`, aliases `PersonasTestApp`, `StyledPersonasTestApp`;
  - `seed_mock_characters(monkeypatch, records: list[dict]) -> None`;
  - `async with roleplay_full_app(*, size, entry="initial_tab"|"ctrl+4", ascii_glyphs=False, notifications=False, on_home=None) as pilot`;
  - `async with open_styled_roleplay(tier, mock_app_instance, *, size, notifications=False, ascii_glyphs=False, app_class=StyledRoleplayMockApp) as pilot`;
  - `wait_until(pilot: Pilot, predicate, *, timeout=20.0, what="")`, `settle(pilot: Pilot)` (screen-owned workers only), `painted_rows(screen: Screen) -> list[str]`, `click_meta_cells(screen: Screen) -> list[tuple[int, int, str, str]]` (these four moved from `test_roleplay_hostile_names.py`, TASK-34400), `chrome_bottoms(screen) -> tuple[int, int]`, `first_list_item(screen)`, `painted_text(screen, region) -> str`, `assert_painted_inside(widget, pane)`, `drop_rule_from_loaded_sheet(app, sheet, selector)`.

- [ ] **Step 1: Write the self-tests (they fail: the harness does not exist)**

Create `Tests/UI/test_roleplay_frame_harness.py`:

```python
"""Self-tests for the Roleplay frame harness (spec 5.7.1; frame slice B1 AC#6, AC#8).

A geometry assertion is only evidence if the tier it runs under really loads
Roleplay's lazy sheet and if deleting a rule from that sheet turns it red.
These tests prove both, under both styled tiers, and that the containment
helper can fail.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from Tests.UI import test_personas_dictionaries, test_personas_workbench
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.roleplay_frame_harness import (
    ROLEPLAY_SHEET,
    ROLEPLAY_SIZES,
    RoleplayMockApp,
    StyledRoleplayMockApp,
    assert_painted_inside,
    drop_rule_from_loaded_sheet,
    roleplay_full_app,
    seed_mock_characters,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

CHARACTERS = [{"id": 1, "name": "Detective Sam", "version": 1}]


@pytest.fixture
def one_character(monkeypatch):
    seed_mock_characters(monkeypatch, CHARACTERS)


def test_the_size_matrix_is_the_b1_matrix():
    assert ROLEPLAY_SIZES == ((80, 24), (120, 36), (160, 45), (220, 55))
    assert ROLEPLAY_SHEET.name == "screen_feature_roleplay.tcss"


def test_the_moved_harness_is_the_one_object_under_every_old_name():
    """Moved verbatim and re-exported, so existing tests are untouched."""
    for module in (test_personas_workbench, test_personas_dictionaries):
        assert module.PersonasTestApp is RoleplayMockApp
        assert module.StyledPersonasTestApp is StyledRoleplayMockApp


def test_the_styled_mock_tier_loads_every_app_stylesheet():
    """The boot bundle AND every lazy split sheet, derived from the build's
    own SCREEN_OWNED_SPLITS (never named by hand)."""
    assert StyledRoleplayMockApp.CSS_PATH == [str(path) for path in APP_STYLESHEETS]
    assert "CSS_PATH" not in RoleplayMockApp.__dict__  # the unstyled tier


@pytest.mark.parametrize("entry", ["initial_tab", "ctrl+4"])
async def test_the_full_app_tier_reaches_roleplay_by_both_real_routes(
    entry, one_character
):
    async with roleplay_full_app(size=(120, 36), entry=entry) as pilot:
        assert type(pilot.app.screen).__name__ == "PersonasScreen"
        assert pilot.app.screen.query("#personas-library-rows > ListItem")


def test_dropping_a_rule_that_is_not_there_is_a_loud_error():
    class _NoSheetApp(App):
        pass

    with pytest.raises(KeyError):
        drop_rule_from_loaded_sheet(_NoSheetApp(), ROLEPLAY_SHEET, "#nothing")


class _ContainmentProbe(App):
    CSS = """
    Screen { layers: base overlay; }
    #other, #pane { width: 30; height: 3; }
    #inside, #covered, #outside { width: 10; height: 1; }
    #outside { offset: 40 0; }
    #cover { width: 20; height: 1; dock: top; layer: overlay; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="other"):
            yield Static("covered", id="covered")
        with Vertical(id="pane"):
            yield Static("inside", id="inside")
            yield Static("outside", id="outside")
        yield Static("cover", id="cover")


async def test_assert_painted_inside_fails_for_escape_and_cover():
    """The helper's two refusals each fire (it is not vacuous)."""
    app = _ContainmentProbe()
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        pane = app.query_one("#pane")
        assert_painted_inside(app.query_one("#inside"), pane)
        with pytest.raises(AssertionError, match="escapes"):
            assert_painted_inside(app.query_one("#outside"), pane)
        with pytest.raises(AssertionError, match="covered"):
            assert_painted_inside(app.query_one("#covered"), app.query_one("#other"))
```

- [ ] **Step 2: Run it to see it fail**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_frame_harness.py -q -p no:cacheprovider 2>&1 | tail -3`
Expected: a collection error, `ModuleNotFoundError: No module named 'Tests.UI.roleplay_frame_harness'`.

- [ ] **Step 3: Create `Tests/UI/roleplay_frame_harness.py`**

```python
"""Roleplay frame test harness: two styled tiers, the size matrix, painted geometry.

Spec section 5.7.1 (Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md):

- ``RoleplayMockApp`` is the delegating ``PersonasTestApp`` that used to live in
  ``test_personas_workbench.py`` (moved verbatim; both old modules re-export
  it under the old names). It is the UNSTYLED tier: ``ConsolidatedCSSApp``
  loads no app bundle, so unstyled tests never assert geometry.
- ``StyledRoleplayMockApp`` is styled tier 1: the boot bundle plus every lazy
  split sheet (``APP_STYLESHEETS``, derived from the build's own
  ``SCREEN_OWNED_SPLITS``), so it carries ``screen_feature_roleplay.tcss``.
- ``painted_rows``, ``click_meta_cells``, ``settle`` and ``wait_until`` (moved
  here from ``test_roleplay_hostile_names.py``, TASK-34400) and
  ``seed_mock_characters`` are the one copy every Roleplay test imports.
- ``roleplay_full_app()`` is styled tier 2: a real ``TldwCli`` that reaches
  Roleplay through a real route (the initial tab, or Ctrl+4 from Home), so
  ``TldwCli._ensure_screen_owned_css`` loads the Roleplay sheet exactly as it
  does for a user. It is seeded through the same ``ccp_character_handler``
  seams as the mock tier (``seed_mock_characters``), not through a temporary
  ChaChaNotes as spec 5.7.1 describes: a recorded B1 deviation, so this tier
  proves styling and routing, not persistence. Frame slice B2a converts it to
  DB seeding before its volume tests (TASK-33910.3 carries the note).

A real ``TldwCli`` that pushes ``PersonasScreen`` itself skips
``_ensure_screen_owned_css`` and paints the inline header without its rules
(TASK-32187's Watchlists trap; ``Tests/UI/full_app_destination_context.py``).
Reach Roleplay by navigation, as ``roleplay_full_app`` does, or call
``app._ensure_screen_owned_css("personas")`` before the push;
``Tests/UI/test_roleplay_stylesheet.py`` scans the suite for the bare push.

Geometry is asserted relative to the MEASURED nav bar and header, never as
absolute rows (spec 5.7.2 item 1; ADR-210's compact nav moves rows below 35).
Not collected by pytest (the file name has no ``test_`` prefix).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from textual.css.stylesheet import CssSource
from textual.geometry import Region
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widget import Widget

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import APP_STYLESHEETS, CSS_DIR, ConsolidatedCSSApp
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.glyph_fallback import set_ascii_glyph_mode

#: The lazily loaded Roleplay sheet (build_css.SCREEN_OWNED_SPLITS).
ROLEPLAY_SHEET = CSS_DIR / "screen_feature_roleplay.tcss"

#: Spec 5.7.1 geometry matrix for B1: 80x24 is the degrade check, the other
#: three are the design centre. 60x24 joins with B7: B1's worst-case header
#: (every chip, a server, the longest kind) fits from 65 columns, and the
#: spec's below-64 layout is B7's and TASK-33910.24's.
ROLEPLAY_SIZES: tuple[tuple[int, int], ...] = (
    (80, 24),
    (120, 36),
    (160, 45),
    (220, 55),
)

#: The size matrix as a parametrize mark: ``@size_matrix`` runs a test once
#: per B1 size with a ``roleplay_size`` argument (ids ``80x24`` ...). A mark,
#: not a fixture, so test modules need no fixture import.
size_matrix = pytest.mark.parametrize(
    "roleplay_size", ROLEPLAY_SIZES, ids=lambda size: f"{size[0]}x{size[1]}"
)


class RoleplayMockApp(ConsolidatedCSSApp):
    """The unstyled tier: today's delegating ``PersonasTestApp``, moved verbatim."""

    def __init__(self, mock_app_instance):
        super().__init__()
        self._mock = mock_app_instance
        self.character_persona_scope_service = (
            mock_app_instance.character_persona_scope_service
        )

    # Delegating these to a MagicMock would make Textual see phantom dynamic
    # hooks (``compute_*``/``watch_*``/...) on the App and crash at mount.
    _NON_DELEGATED_PREFIXES = (
        "_",
        "watch_",
        "compute_",
        "validate_",
        "action_",
        "key_",
        "on_",
    )

    def __getattr__(self, name):
        if name.startswith(self._NON_DELEGATED_PREFIXES):
            raise AttributeError(name)
        return getattr(self.__dict__["_mock"], name)

    def compose(self):
        # Mirrors the real app: an `AppFooterStatus` composed directly on
        # the app's own default screen (see app.py's `compose()`).
        # Task-264: `PersonasScreen` (via `BaseAppScreen.compose()`) now
        # mounts its OWN `AppFooterStatus` too, and
        # `PersonasScreen._register_footer_shortcuts()` resolves that
        # screen-owned instance via ``self.query_one("AppFooterStatus")`` --
        # so this default-screen widget is only kept around as a foil (the
        # tests below assert the registration does NOT land here).
        yield AppFooterStatus(id="app-footer-status")

    async def _ensure_tts_profile_service(self):
        """Delegate the real app's private lazy loader when a test provides it."""

        loader = self.__dict__["_mock"].__dict__.get("_ensure_tts_profile_service")
        if not callable(loader):
            return None
        result = loader()
        if inspect.isawaitable(result):
            result = await result
        return result

    def on_mount(self) -> None:
        self.push_screen(PersonasScreen(self))


class StyledRoleplayMockApp(RoleplayMockApp):
    """Styled tier 1: the boot bundle plus every lazy split sheet (spec 5.7.1)."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]


#: The names the Roleplay test modules have always used (re-exported there).
PersonasTestApp = RoleplayMockApp
StyledPersonasTestApp = StyledRoleplayMockApp


def seed_mock_characters(
    monkeypatch: pytest.MonkeyPatch, records: list[dict[str, Any]]
) -> None:
    """Route the screen's character seams over ``records`` (both tiers).

    The same seams ``test_personas_workbench.stub_characters`` patches:
    ``fetch_all_characters``/``fetch_character_by_id`` plus the paged loader.

    Args:
        monkeypatch: The test's monkeypatch.
        records: Character dicts with at least ``id`` and ``name``.
    """
    import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as handler
    from Tests.UI.test_personas_dictionaries import patch_character_paging

    monkeypatch.setattr(
        handler, "fetch_all_characters", lambda: [dict(r) for r in records]
    )
    monkeypatch.setattr(
        handler,
        "fetch_character_by_id",
        lambda character_id: next(
            (dict(r) for r in records if str(r["id"]) == str(character_id)), None
        ),
    )
    patch_character_paging(monkeypatch)


def _settings_for_full_app(ascii_glyphs: bool) -> Callable[..., Any]:
    """``get_cli_setting`` for the full-app tier: no splash, chosen glyph mode.

    The app reads ``splash_screen.enabled`` at compose and resets the glyph
    mode from ``appearance.ascii_glyphs`` at compose (app.py), so both must
    come from this patch, live for the whole ``run_test``. Measured while
    planning: arrival takes 7.7 s with the splash and 0.7 s without.
    """

    def settings(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        if section == "appearance" and key == "ascii_glyphs":
            return ascii_glyphs
        return default

    return settings


# wait_until, painted_rows, click_meta_cells and settle moved here unchanged
# from test_roleplay_hostile_names.py (TASK-34400) in B1: this harness is the
# one home of the Roleplay paint helpers.
async def wait_until(
    pilot: Pilot,
    predicate: Callable[[], bool],
    *,
    timeout: float = 20.0,
    what: str = "",
) -> None:
    """Poll ``predicate`` with a monotonic deadline.

    Args:
        pilot: The running test pilot.
        predicate: Returns True once the awaited state holds.
        timeout: Seconds to wait before failing.
        what: Names the awaited state in the failure message.

    Raises:
        AssertionError: ``predicate`` stayed False for ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"timed out after {timeout}s waiting for {what or predicate}")


def _first_list_item_present(app) -> bool:
    screen = app.screen
    return isinstance(screen, PersonasScreen) and bool(
        screen.query("#personas-library-rows > ListItem")
    )


@asynccontextmanager
async def roleplay_full_app(
    *,
    size: tuple[int, int],
    entry: str = "initial_tab",
    ascii_glyphs: bool = False,
    notifications: bool = False,
    on_home: Callable[[Any], None] | None = None,
) -> AsyncIterator[Any]:
    """Styled tier 2: a real ``TldwCli`` arriving at Roleplay by a real route.

    Seed data through the same seams first (``seed_mock_characters``; no
    ChaChaNotes DB yet, see the module docstring). Yields the pilot once the
    Roleplay screen shows its first list item.

    Args:
        size: Terminal ``(columns, rows)``.
        entry: ``"initial_tab"`` boots straight into Roleplay (the app's
            ``_push_initial_screen`` path); ``"ctrl+4"`` boots Home and presses
            Ctrl+4 (the in-app navigation path).
        ascii_glyphs: Run with ``appearance.ascii_glyphs`` on.
        notifications: Mount toasts (``run_test`` defaults to off).
        on_home: With ``entry="ctrl+4"``, called with the app while Home is
            showing, just before Ctrl+4 (to observe the pre-visit state).
    """
    if entry not in ("initial_tab", "ctrl+4"):
        raise ValueError(f"unknown entry {entry!r}")
    app = _build_test_app(
        configured_default="personas" if entry == "initial_tab" else "home"
    )
    settings = _settings_for_full_app(ascii_glyphs)
    try:
        with patch_app_global("get_cli_setting", side_effect=settings):
            async with app.run_test(size=size, notifications=notifications) as pilot:
                await wait_until(
                    pilot,
                    lambda: getattr(app, "_initial_screen_pushed", False),
                    what="the initial screen",
                )
                if entry == "ctrl+4":
                    await wait_until(
                        pilot,
                        lambda: type(app.screen).__name__ == "HomeScreen",
                        what="Home",
                    )
                    if on_home is not None:
                        on_home(app)
                    await pilot.press("ctrl+4")
                await wait_until(
                    pilot,
                    lambda: _first_list_item_present(app),
                    what="Roleplay's list",
                )
                await settle(pilot)
                yield pilot
    finally:
        # The app set the process-wide glyph mode at compose; never leak it.
        set_ascii_glyph_mode(False)


#: The two styled tiers (spec 5.7.1): ``"mock"`` (StyledRoleplayMockApp) and
#: ``"full"`` (roleplay_full_app). ``@styled_tiers`` gives a test a
#: ``styled_tier`` argument per tier.
STYLED_TIERS: tuple[str, ...] = ("mock", "full")
styled_tiers = pytest.mark.parametrize("styled_tier", STYLED_TIERS)


@asynccontextmanager
async def open_styled_roleplay(
    tier: str,
    mock_app_instance: Any,
    *,
    size: tuple[int, int],
    notifications: bool = False,
    ascii_glyphs: bool = False,
    app_class: type[StyledRoleplayMockApp] = StyledRoleplayMockApp,
) -> AsyncIterator[Any]:
    """Mount Roleplay under one styled tier and wait for its first list item.

    Args:
        tier: ``"mock"`` or ``"full"``.
        mock_app_instance: The ``Tests/UI/conftest.py`` fixture (mock tier only).
        size: Terminal ``(columns, rows)``.
        notifications: Mount toasts.
        ascii_glyphs: Run with the ASCII glyph mode on (restored afterwards).
        app_class: A ``StyledRoleplayMockApp`` subclass (mock tier only).
    """
    if tier == "full":
        async with roleplay_full_app(
            size=size, notifications=notifications, ascii_glyphs=ascii_glyphs
        ) as pilot:
            yield pilot
        return
    if tier != "mock":
        raise ValueError(f"unknown tier {tier!r}")
    app = app_class(mock_app_instance)
    # The mock tier has no app compose that resets the process-wide mode.
    set_ascii_glyph_mode(ascii_glyphs)
    try:
        async with app.run_test(size=size, notifications=notifications) as pilot:
            await wait_until(
                pilot, lambda: _first_list_item_present(app), what="Roleplay's list"
            )
            await settle(pilot)
            yield pilot
    finally:
        set_ascii_glyph_mode(False)


def chrome_bottoms(screen) -> tuple[int, int]:
    """``(nav bottom, header bottom)`` as measured, in screen rows (0-based)."""
    nav = screen.query_one(MainNavigationBar).region
    header = screen.query_one("#personas-header").region
    return nav.bottom, header.bottom


def first_list_item(screen) -> Widget:
    """The first row of the items list (B2a keeps the id, replaces the type)."""
    return screen.query("#personas-library-rows > ListItem").first()


def painted_rows(screen: Screen) -> list[str]:
    """Every compositor row as plain text: what the terminal shows.

    Args:
        screen: The mounted screen whose compositor output is read.

    Returns:
        One string per terminal row, top to bottom, as currently painted.
    """
    return [strip.text for strip in screen._compositor.render_strips()]


def painted_text(screen, region: Region) -> str:
    """The plain text the compositor paints inside ``region`` (one line per row)."""
    rows = painted_rows(screen)
    return "\n".join(
        rows[y][region.x : region.right] for y in range(region.y, region.bottom)
    )


def assert_painted_inside(widget: Widget, pane: Widget) -> None:
    """Every cell of ``widget`` lies in ``pane``'s painted window and is not covered.

    The painted window is ``scrollable_content_region``, not the pane's outer
    region (lessons-testing-evidence, "painted window, not pane rectangle");
    and containment is not visibility, so the compositor's hit test must
    find ``widget`` (or a descendant) on its middle row (a docked sibling
    can cover a contained widget).
    """
    window = pane.scrollable_content_region
    region = widget.region
    assert region.width > 0 and region.height > 0, (
        f"{widget!r} paints nothing: {region}"
    )
    assert region.intersection(window) == region, (
        f"{widget!r} {region} escapes {pane!r}'s painted window {window}"
    )
    middle = region.y + (region.height - 1) // 2
    hit, _ = widget.screen.get_widget_at(region.x, middle)
    assert hit is widget or widget in hit.ancestors, (
        f"{widget!r} is covered by {hit!r} at ({region.x}, {middle})"
    )


def click_meta_cells(screen: Screen) -> list[tuple[int, int, str, str]]:
    """Every painted cell run that carries an ``@click`` action.

    Args:
        screen: The mounted screen whose compositor output is scanned.

    Returns:
        ``(x, y, text, action)`` for each painted segment whose style meta
        holds ``@click``; empty when no painted text is clickable markup.
    """
    hits = []
    for y, strip in enumerate(screen._compositor.render_strips()):
        x = 0
        for segment in strip:
            meta = segment.style.meta if segment.style is not None else {}
            if meta and "@click" in meta:
                hits.append((x, y, segment.text, meta["@click"]))
            x += segment.cell_length
    return hits


def drop_rule_from_loaded_sheet(app, sheet: Path, selector: str) -> None:
    """Delete one rule block from an already-loaded sheet and restyle the app.

    The executable form of the "delete one ``_roleplay.tcss`` header rule"
    discrimination check (spec B1): it edits the PARSED source the app holds,
    so it works the same whether the sheet arrived through a harness
    ``CSS_PATH`` or through the app's route loader.

    Args:
        app: A running app.
        sheet: The stylesheet file the app loaded.
        selector: The exact selector text heading the block to delete.

    Raises:
        KeyError: The app never loaded ``sheet`` (itself a finding).
    """
    key = (str(sheet), "")
    source = app.stylesheet.source[key]
    head = f"\n{selector} {{"
    start = source.content.index(head)
    end = source.content.index("}", start) + 1
    mutated = source.content[:start] + source.content[end:]
    app.stylesheet.source[key] = CssSource(
        mutated, source.is_defaults, source.tie_breaker, source.scope
    )
    app.stylesheet.reparse()
    app.stylesheet.update(app)


async def settle(pilot: Pilot) -> None:
    """Let the screen's own workers finish, then repaint twice.

    Only workers owned by the current screen: the full app runs app-wide
    workers that never finish, so ``app.workers.wait_for_complete()`` would
    wait forever there.

    Args:
        pilot: The running test pilot.
    """
    await pilot.pause()
    screen = pilot.app.screen
    unfinished = [
        worker
        for worker in pilot.app.workers
        if screen in worker.node.ancestors_with_self and not worker.is_finished
    ]
    if unfinished:
        await pilot.app.workers.wait_for_complete(unfinished)
    await pilot.pause()
    await asyncio.sleep(0)
    await pilot.pause()
```

- [ ] **Step 4: Make the two old modules re-export the moved classes**

In `Tests/UI/test_personas_workbench.py`, Old:
```python
from Tests.UI.background_signals import wait_for_background_signal, wait_for_signal
```
New:
```python
from Tests.UI.background_signals import wait_for_background_signal, wait_for_signal

# Roleplay frame B1: the delegating harness apps moved to
# Tests/UI/roleplay_frame_harness.py (spec 5.7.1); these names stay importable
# from here for the modules that import them.
from Tests.UI.roleplay_frame_harness import PersonasTestApp, StyledPersonasTestApp
```

Then delete the two class definitions. Old (the whole block, from `class PersonasTestApp(ConsolidatedCSSApp):` through the end of `class StyledPersonasTestApp`, its two trailing blank lines, and the next class line as the anchor):
```python
class PersonasTestApp(ConsolidatedCSSApp):
    def __init__(self, mock_app_instance):
        super().__init__()
        self._mock = mock_app_instance
        self.character_persona_scope_service = (
            mock_app_instance.character_persona_scope_service
        )

    # Delegating these to a MagicMock would make Textual see phantom dynamic
    # hooks (``compute_*``/``watch_*``/...) on the App and crash at mount.
    _NON_DELEGATED_PREFIXES = (
        "_",
        "watch_",
        "compute_",
        "validate_",
        "action_",
        "key_",
        "on_",
    )

    def __getattr__(self, name):
        if name.startswith(self._NON_DELEGATED_PREFIXES):
            raise AttributeError(name)
        return getattr(self.__dict__["_mock"], name)

    def compose(self):
        # Mirrors the real app: an `AppFooterStatus` composed directly on
        # the app's own default screen (see app.py's `compose()`).
        # Task-264: `PersonasScreen` (via `BaseAppScreen.compose()`) now
        # mounts its OWN `AppFooterStatus` too, and
        # `PersonasScreen._register_footer_shortcuts()` resolves that
        # screen-owned instance via ``self.query_one("AppFooterStatus")`` --
        # so this default-screen widget is only kept around as a foil (the
        # tests below assert the registration does NOT land here).
        yield AppFooterStatus(id="app-footer-status")

    async def _ensure_tts_profile_service(self):
        """Delegate the real app's private lazy loader when a test provides it."""

        loader = self.__dict__["_mock"].__dict__.get("_ensure_tts_profile_service")
        if not callable(loader):
            return None
        result = loader()
        if inspect.isawaitable(result):
            result = await result
        return result

    def on_mount(self) -> None:
        self.push_screen(PersonasScreen(self))


class StyledPersonasTestApp(PersonasTestApp):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )


class PersonaBuddyWorkbenchApp(PersonasTestApp):
```
New (the anchor alone, so `class PersonaBuddyWorkbenchApp` follows the `stub_characters` fixture after its existing two blank lines):
```python
class PersonaBuddyWorkbenchApp(PersonasTestApp):
```

The module-level `from pathlib import Path` is now unused (every later `Path` use re-imports it locally). Old:
```python
import os
from pathlib import Path
import threading
```
New:
```python
import os
import threading
```

In `Tests/UI/test_personas_dictionaries.py`, Old:
```python
import copy
from pathlib import Path
from typing import Any
```
New:
```python
import copy
from typing import Any
```
Old:
```python
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
```
New:
```python
from Tests.UI.roleplay_frame_harness import PersonasTestApp, StyledPersonasTestApp
```
Old (its copy of the two classes, their two trailing blank lines, and the next function line as the anchor):
```python
class PersonasTestApp(ConsolidatedCSSApp):
    """Same harness as test_personas_workbench.py (delegating App)."""

    def __init__(self, mock_app_instance):
        super().__init__()
        self._mock = mock_app_instance
        self.character_persona_scope_service = (
            mock_app_instance.character_persona_scope_service
        )

    _NON_DELEGATED_PREFIXES = (
        "_",
        "watch_",
        "compute_",
        "validate_",
        "action_",
        "key_",
        "on_",
    )

    def __getattr__(self, name):
        if name.startswith(self._NON_DELEGATED_PREFIXES):
            raise AttributeError(name)
        return getattr(self.__dict__["_mock"], name)

    def compose(self):
        yield AppFooterStatus(id="app-footer-status")

    def on_mount(self) -> None:
        self.push_screen(PersonasScreen(self))


class StyledPersonasTestApp(PersonasTestApp):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )


async def _mounted(pilot):
```
New (the anchor alone):
```python
async def _mounted(pilot):
```

The re-export by direct import (not an assignment) matters: the CSS-ownership scan in `Tests/UI/test_consolidated_css_harness.py` resolves `CSS_PATH = StyledPersonasTestApp.CSS_PATH` (`_NavCaptureApp`) by importing the name from its source module, so it sees `APP_STYLESHEETS` and the Roleplay sheet.

Now make the harness the one home of the paint helpers. In `Tests/UI/test_roleplay_hostile_names.py`, the imports. Old:
```python
import asyncio
import time
from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import pytest
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widgets import ListView

import tldw_chatbook.app  # noqa: F401  -- collection-time import (bootstrap profile)
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_personas_dictionaries import (
    FakeDictScopeService,
    make_dict_record,
    patch_character_paging,
)
```
New:
```python
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import ListView

import tldw_chatbook.app  # noqa: F401  -- collection-time import (bootstrap profile)
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import _build_test_app

# Roleplay frame B1: the paint helpers and the character seam live in the
# frame harness, their one home (test_roleplay_hostile_text_surfaces.py
# imports them from there too).
from Tests.UI.roleplay_frame_harness import (
    click_meta_cells,
    painted_rows,
    seed_mock_characters,
    settle,
    wait_until,
)
from Tests.UI.test_personas_dictionaries import (
    FakeDictScopeService,
    make_dict_record,
)
```
Delete the five moved definitions (`_seed_characters`, `painted_rows`, `click_meta_cells`, `settle`, `wait_until`, contiguous on dev). Old (from `def _seed_characters` through the end of `wait_until`, its two trailing blank lines, and the next function line as the anchor):
```python
def _seed_characters(monkeypatch, records: list[dict]) -> None:
    """Route the screen's character seams over ``records``."""
    monkeypatch.setattr(
        character_handler_module,
        "fetch_all_characters",
        lambda: [dict(record) for record in records],
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: next(
            (dict(r) for r in records if str(r["id"]) == str(character_id)), None
        ),
    )
    patch_character_paging(monkeypatch)


def painted_rows(screen: Screen) -> list[str]:
    """Every compositor row as plain text: what the terminal shows.

    Args:
        screen: The mounted screen whose compositor output is read.

    Returns:
        One string per terminal row, top to bottom, as currently painted.
    """
    return [strip.text for strip in screen._compositor.render_strips()]


def click_meta_cells(screen: Screen) -> list[tuple[int, int, str, str]]:
    """Every painted cell run that carries an ``@click`` action.

    Args:
        screen: The mounted screen whose compositor output is scanned.

    Returns:
        ``(x, y, text, action)`` for each painted segment whose style meta
        holds ``@click``; empty when no painted text is clickable markup.
    """
    hits = []
    for y, strip in enumerate(screen._compositor.render_strips()):
        x = 0
        for segment in strip:
            meta = segment.style.meta if segment.style is not None else {}
            if meta and "@click" in meta:
                hits.append((x, y, segment.text, meta["@click"]))
            x += segment.cell_length
    return hits


async def settle(pilot: Pilot) -> None:
    """Let the screen's own workers finish, then repaint twice.

    Only workers owned by the current screen: the full app runs app-wide
    workers that never finish, so ``app.workers.wait_for_complete()`` would
    wait forever there.

    Args:
        pilot: The running test pilot.
    """
    await pilot.pause()
    screen = pilot.app.screen
    unfinished = [
        worker
        for worker in pilot.app.workers
        if screen in worker.node.ancestors_with_self and not worker.is_finished
    ]
    if unfinished:
        await pilot.app.workers.wait_for_complete(unfinished)
    await pilot.pause()
    await asyncio.sleep(0)
    await pilot.pause()


async def wait_until(
    pilot: Pilot,
    predicate: Callable[[], bool],
    *,
    timeout: float = 20.0,
    what: str = "",
) -> None:
    """Poll ``predicate`` with a monotonic deadline.

    Args:
        pilot: The running test pilot.
        predicate: Returns True once the awaited state holds.
        timeout: Seconds to wait before failing.
        what: Names the awaited state in the failure message.

    Raises:
        AssertionError: ``predicate`` stayed False for ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"timed out after {timeout}s waiting for {what or predicate}")


def _painted(screen) -> str:
```
New (the anchor alone):
```python
def _painted(screen) -> str:
```
The six `_seed_characters(` calls become `seed_mock_characters(` (four of them are the identical line `_seed_characters(monkeypatch, [])`, so this one rename is scripted, with its count asserted):

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
from pathlib import Path

path = Path("Tests/UI/test_roleplay_hostile_names.py")
text = path.read_text(encoding="utf-8")
assert text.count("_seed_characters(") == 6, text.count("_seed_characters(")
path.write_text(text.replace("_seed_characters(", "seed_mock_characters("), encoding="utf-8")
print("six calls now use seed_mock_characters")
PY
EOF
```

Expected: `six calls now use seed_mock_characters`.

In `Tests/UI/test_roleplay_hostile_text_surfaces.py`, Old:
```python
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_roleplay_hostile_names import (
    HOSTILE_NAMES,
    click_meta_cells,
    painted_rows,
)
```
New:
```python
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.roleplay_frame_harness import click_meta_cells, painted_rows
from Tests.UI.test_roleplay_hostile_names import HOSTILE_NAMES
```

- [ ] **Step 5: Run the self-tests, lint, and check the re-point is neutral at base**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check Tests/UI/roleplay_frame_harness.py Tests/UI/test_roleplay_frame_harness.py Tests/UI/test_personas_workbench.py Tests/UI/test_personas_dictionaries.py Tests/UI/test_roleplay_hostile_names.py Tests/UI/test_roleplay_hostile_text_surfaces.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check Tests/UI/roleplay_frame_harness.py Tests/UI/test_roleplay_frame_harness.py Tests/UI/test_personas_workbench.py Tests/UI/test_personas_dictionaries.py Tests/UI/test_roleplay_hostile_names.py Tests/UI/test_roleplay_hostile_text_surfaces.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_frame_harness.py -q -p no:cacheprovider --timeout=180 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/plug /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_hostile_names.py Tests/UI/test_roleplay_hostile_text_surfaces.py -p b1_bootstrap_all -q -p no:cacheprovider -n 6 --timeout=300 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1
EOF
```

Expected: `All checks passed!`, `6 files already formatted`, `7 passed`, then `103 passed` (TASK-34400's 22 + 81, unchanged: only where their helpers come from moved). Then the paired check over every module that imports the moved apps or helpers:

```bash
bash <<'EOF'
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence
"$EV/paired.sh" t1-roleplay "$EV/suites-roleplay.txt" -n 6 | tee -a "$EV/gates.txt"
EOF
```

Expected: `recovery=` single digits on both arms and nothing between `new failures on head [t1-roleplay] (must be empty):` and its end line. Pre-existing failures (the dry runs saw `test_resize_sync_skips_work_when_compact_state_is_unchanged`, `test_toolbar_single_row_at_wide_terminal`, the `TestPersonaHumanIdentityRemoval` set and others; `test_actor_pack_creation_workflow.py::test_navigation_signals_and_drains_pack_creation_before_continuing` failed 3/3 alone on BOTH arms in review and passes under `-n 6` only by chance, so it may show as "new" on either arm: re-run it alone on both) appear on both arms. Apply Task 12 Step 4's flake rule to anything new.

- [ ] **Step 6: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
git add Tests/UI/roleplay_frame_harness.py Tests/UI/test_roleplay_frame_harness.py Tests/UI/test_personas_workbench.py Tests/UI/test_personas_dictionaries.py Tests/UI/test_roleplay_hostile_names.py Tests/UI/test_roleplay_hostile_text_surfaces.py
git commit -m "test(roleplay): the frame harness with two styled tiers and the size matrix (B1)" -m "PersonasTestApp moves verbatim into Tests/UI/roleplay_frame_harness.py and is re-exported; StyledPersonasTestApp now loads the boot bundle plus every split sheet. roleplay_full_app() reaches Roleplay through the real initial-tab and Ctrl+4 routes. The harness is the one home of the paint helpers TASK-34400 added to test_roleplay_hostile_names.py (moved unchanged; both hostile-text modules import them from here)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

---

### Task 2: Glyph-map entries for the frame vocabulary (spec §4.12, G20)

**Files:**
- Modify: `tldw_chatbook/Widgets/glyph_fallback.py` (`ASCII_GLYPH_FALLBACKS`), `tldw_chatbook/Widgets/adaptive_pane_shell.py` (the `RAIL_ROW_ELLIPSIS` comment), `backlog/decisions/212-shared-adaptive-pane-shell.md` (the rail-row fitting bullet)
- Test: `Tests/Chat/test_console_glyphs.py`

**Interfaces:**
- Consumes: `resolve_glyph`, `resolve_glyph_text`, `ASCII_GLYPH_FALLBACKS`.
- Produces: ASCII substitutes `›`→`>`, `‹`→`<`, `…`→`...`, `·`→`-`, `—`→`-`, `⇥`→`>|`, `→`→`->`, `←`→`<-`, `↑`→`^`, `↓`→`v`, `×`→`x` (Task 5's fit and the header's chips rely on `›`, `…`, `·`).

Blast radius measured while planning (paired, every suite that turns ASCII mode on): exactly one existing test changes, `test_resolve_glyph_text_maps_embedded_markers`, whose label ends in `…`. Known side effect, new with B1 and not on dev, outside Roleplay and visible to users: in ASCII mode, Console surfaces that pass user text through `resolve_glyph_text` (staged file names in the composer, Inspect row text) now also rewrite `·`, `—`, `…`, `×` and arrows in that text, and `…`/arrows are wider in ASCII. Nobody has accepted it yet: the PR body lists it, and Task 12 Step 8 puts it to the owner for confirmation together with the screenshots (a "no" becomes a new step here before the PR merges).

- [ ] **Step 1: Write the failing test and re-pin the one that changes**

In `Tests/Chat/test_console_glyphs.py`, Old:
```python
        resolve_glyph_text(ConsoleComposerBar.VOICE_CHIP_TRANSCRIBING_LABEL)
        == "(~) Transcribing…"
    )
```
New:
```python
        resolve_glyph_text(ConsoleComposerBar.VOICE_CHIP_TRANSCRIBING_LABEL)
        == "(~) Transcribing..."  # "…" maps too since Roleplay frame B1
    )
```
Old:
```python
def test_tab_label_uses_ascii_markers_in_ascii_mode(ascii_mode):
```
New:
```python
#: Roleplay frame B1 (spec section 4.12, G20): the frame's own vocabulary.
_ROLEPLAY_FRAME_GLYPHS = {
    "›": ">",
    "‹": "<",
    "…": "...",
    "·": "-",
    "—": "-",
    "⇥": ">|",
    "→": "->",
    "←": "<-",
    "↑": "^",
    "↓": "v",
    "×": "x",
}


def test_roleplay_frame_glyphs_have_their_ascii_substitutes(ascii_mode):
    for glyph, substitute in _ROLEPLAY_FRAME_GLYPHS.items():
        assert ASCII_GLYPH_FALLBACKS[glyph] == substitute
        assert resolve_glyph(glyph) == substitute
    # "×" (multiplication) and "✕" (close) are different characters.
    assert ASCII_GLYPH_FALLBACKS["✕"] == "x"


def test_tab_label_uses_ascii_markers_in_ascii_mode(ascii_mode):
```

- [ ] **Step 2: Run to see it fail**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_glyphs.py -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `2 failed` — `test_roleplay_frame_glyphs_have_their_ascii_substitutes` (`KeyError: '›'`) and `test_resolve_glyph_text_maps_embedded_markers` (still `…`).

- [ ] **Step 3: Add the entries**

In `tldw_chatbook/Widgets/glyph_fallback.py`, Old:
```python
    "📎": "[+]",  # staged attachment indicator
}
```
New:
```python
    "📎": "[+]",  # staged attachment indicator
    # Roleplay frame vocabulary (spec section 4.12; frame slice B1): breadcrumb
    # and return arrows, the truncation ellipsis, separators, the indent key
    # and arrow-key names, and the multiplication sign ("✕" above is the
    # close glyph, a different character).
    "›": ">",  # breadcrumb / "go there" (Settings ›)
    "‹": "<",  # return crumb (‹ Library)
    "…": "...",  # truncation ellipsis
    "·": "-",  # inline separator
    "—": "-",  # em dash in copy
    "⇥": ">|",  # tab key
    "→": "->",  # right arrow key
    "←": "<-",  # left arrow key
    "↑": "^",  # up arrow key
    "↓": "v",  # down arrow key
    "×": "x",  # multiplication sign
}
```

Fix the B0 prose this makes false. In `tldw_chatbook/Widgets/adaptive_pane_shell.py`, Old:
```python
#: The ellipsis a squeezed title ends with (step 6), measured in cells. It goes
#: through ``resolve_glyph``, but the glyph map has no ASCII substitute for it,
#: so ASCII mode paints it unchanged.
```
New:
```python
#: The ellipsis a squeezed title ends with (step 6), measured in cells. It goes
#: through ``resolve_glyph``, so ASCII mode paints ``...`` (Roleplay frame B1
#: added the map entry); the fitter measures the resolved glyph.
```
In `backlog/decisions/212-shared-adaptive-pane-shell.md`, Old:
```markdown
the ellipsis is `…`, measured in cells, and has no ASCII substitute (the glyph map has none, so ASCII mode paints `…` too);
```
New:
```markdown
the ellipsis is `…`, measured in cells after `resolve_glyph` (Roleplay frame B1 added `…` → `...` to the glyph map, so ASCII mode paints `...`);
```

- [ ] **Step 4: Run to see it pass, then the glyph suites against the base arm**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_glyphs.py Tests/UI/test_destination_rail_row.py -q -p no:cacheprovider 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1`
Expected: all passed (the rail-row fitter already measures the resolved ellipsis, so its fit tests hold).

```bash
bash <<'EOF'
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence
"$EV/paired.sh" t2-shell "$EV/suites-shell.txt" -n 6 | tee -a "$EV/gates.txt"
EOF
```

Expected: no new failures on head. The dry run's base already fails `test_console_inspector_section.py` ×3 and `test_library_honesty_accessibility.py` ×6 (pre-existing).

- [ ] **Step 5: Named mutation `drop-go-glyph`**

Delete the `"›": ">",` line with the Edit tool; run `…/python -m pytest Tests/Chat/test_console_glyphs.py -q -p no:cacheprovider -k roleplay_frame`: `1 failed` (`KeyError: '›'`). Restore the line; `1 passed`. Record the row in `$EV/mutations.md`.

- [ ] **Step 6: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
git add tldw_chatbook/Widgets/glyph_fallback.py tldw_chatbook/Widgets/adaptive_pane_shell.py backlog/decisions/212-shared-adaptive-pane-shell.md Tests/Chat/test_console_glyphs.py
git commit -m "feat(glyphs): ASCII substitutes for the Roleplay frame vocabulary (B1, G20)" -m "Adds › ‹ … · — ⇥ → ← ↑ ↓ × to the glyph map; the voice-chip label pin now ends in '...'. The shared rail-row ellipsis now paints '...' in ASCII mode, and the prose that said it could not is corrected." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

---

### Task 3: `FittedText` — literal text refitted to its own width (shared workbench widget)

Why a widget: the header's item label must end a long name in the *resolved* ellipsis and keep `· editing`, at whatever width the row leaves it. CSS `text-overflow: ellipsis` paints Textual's own `…` even in ASCII mode (measured), and a fit computed at sync time is stale as soon as a chip appears beside it. A widget whose `render()` fits to its own `content_size.width` is correct on every repaint, needs no resize hook and parses no markup. It lives beside `DestinationHeader` in `UI/Workbench/workbench_widgets.py` (already imported by every destination, so census +0, and no new module: B1's one new module is the pure state, spec §5.11); FU-2 (a literal mode for the shared header) can build on it.

**Files:**
- Modify: `tldw_chatbook/UI/Workbench/workbench_widgets.py` (imports; a new class after `DestinationHeader`)
- Test: `Tests/UI/test_workbench_fitted_text.py` (create)

**Interfaces:**
- Consumes: `textual.content.Content`, `textual.widget.Widget`.
- Produces: `FittedText(value: Hashable = "", fit: Callable[[Any, int], str] | None = None, *, hide_when_empty: bool = False, **kwargs)` with `.value`, `.fitted_text: str` (the text at the current content width), `.set_value(value) -> None` (no-op when unchanged; else toggles `display` if `hide_when_empty` and `refresh(layout=True)`), `render() -> Content` (literal). The default fit returns `str(value)` whatever the width.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_workbench_fitted_text.py`:

```python
"""``FittedText``: literal text refitted to its own width (Roleplay frame B1).

The shared widget behind the Roleplay header's item label and chips. Small
host apps only (no destination screen), so this file runs in the UI fast lane.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from tldw_chatbook.UI.Workbench.workbench_widgets import FittedText

pytestmark = pytest.mark.asyncio


def _head(value: str, width: int) -> str:
    return value[:width]


class _Host(App):
    CSS = """
    #row { height: 1; }
    #label { width: 1fr; height: 1; }
    #chip { width: auto; height: 1; }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="row"):
            yield FittedText("[b]x[/] " + "y" * 60, _head, id="label")
            yield FittedText("", hide_when_empty=True, id="chip")


def _painted_row(app: App, y: int) -> str:
    return list(app.screen._compositor.render_strips())[y].text


async def test_text_is_literal_and_carries_no_action():
    """Markup-shaped text paints as typed (spec R33): no MarkupError, no meta."""
    app = _Host()
    async with app.run_test(size=(40, 3)) as pilot:
        await pilot.pause()
        row = _painted_row(app, 0)
        assert row.startswith("[b]x[/] yyy")
        for strip in app.screen._compositor.render_strips():
            for segment in strip:
                meta = segment.style.meta if segment.style is not None else {}
                assert "@click" not in meta


async def test_the_fit_follows_the_widget_width_on_every_resize():
    app = _Host()
    async with app.run_test(size=(40, 3)) as pilot:
        await pilot.pause()
        label = app.query_one("#label", FittedText)
        assert label.fitted_text == label.value[:40]
        await pilot.resize_terminal(25, 3)
        await pilot.pause()
        assert label.content_size.width == 25
        assert _painted_row(app, 0) == label.value[:25]


async def test_set_value_repaints_only_on_a_change(monkeypatch):
    app = _Host()
    async with app.run_test(size=(40, 3)) as pilot:
        await pilot.pause()
        label = app.query_one("#label", FittedText)
        calls = []
        original = label.refresh

        def counting_refresh(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(label, "refresh", counting_refresh)
        label.set_value(label.value)
        assert calls == []
        label.set_value("changed")
        assert calls == [{"layout": True}]


async def test_an_empty_chip_takes_no_space_until_it_has_text():
    app = _Host()
    async with app.run_test(size=(40, 3)) as pilot:
        await pilot.pause()
        chip = app.query_one("#chip", FittedText)
        assert chip.display is False
        chip.set_value("Unsaved")
        await pilot.pause()
        assert chip.display is True
        assert chip.region.width == len("Unsaved")
        chip.set_value("")
        await pilot.pause()
        assert chip.display is False


def test_the_widget_declares_no_css():
    """Rules belong to the owning destination's sheet (zero boot bytes)."""
    for name in ("DEFAULT_CSS", "CSS", "BUNDLED_CSS"):
        assert name not in FittedText.__dict__
```

- [ ] **Step 2: Run to see it fail**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workbench_fitted_text.py -q -p no:cacheprovider 2>&1 | tail -3`
Expected: a collection error, `ImportError: cannot import name 'FittedText'`.

- [ ] **Step 3: Implement**

In `tldw_chatbook/UI/Workbench/workbench_widgets.py`, Old:
```python
from collections.abc import Iterable
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
```
New:
```python
from collections.abc import Callable, Hashable, Iterable
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
```
Old:
```python
class CommandStrip(Horizontal):
```
New:
```python
def _whole_text(value: Hashable, width: int) -> str:
    """``FittedText``'s default fit: the value's text, whatever the width."""
    return str(value)


class FittedText(Widget):
    """One line of literal text, refitted to the widget's own width on every render.

    For a slot whose text must shorten in a controlled way as space changes:
    ``fit(value, width)`` decides what survives in ``width`` cells, so the
    caller keeps what matters and ends a cut with its own resolved ellipsis.
    CSS ``text-overflow: ellipsis`` cannot do either (it always paints a
    literal ``…``, even in ASCII glyph mode). The text is never parsed as
    markup, so an untrusted name renders literally, and ``set_value``
    repaints only on a change (task-15452: ``Static.update`` has no
    equality check of its own). Declares no CSS: every rule belongs to the
    owning destination's sheet.
    """

    def __init__(
        self,
        value: Hashable = "",
        fit: Callable[[Any, int], str] | None = None,
        *,
        hide_when_empty: bool = False,
        **kwargs: Any,
    ) -> None:
        """Build the label.

        Args:
            value: What to show; handed to ``fit`` with the current width.
            fit: ``(value, width) -> text``. Defaults to the whole value.
            hide_when_empty: Hide the widget while ``value`` is falsy (a chip
                that should take no space when it has nothing to say).
            **kwargs: Forwarded to ``Widget`` (``id``, ``classes``, ...).
        """
        super().__init__(**kwargs)
        self._value: Hashable = value
        self._fit: Callable[[Any, int], str] = fit or _whole_text
        self._hide_when_empty = hide_when_empty
        if hide_when_empty:
            self.display = bool(value)

    @property
    def value(self) -> Hashable:
        """The value currently shown."""
        return self._value

    @property
    def fitted_text(self) -> str:
        """The text exactly as it paints at the current content width."""
        return self._fit(self._value, self.content_size.width)

    def set_value(self, value: Hashable) -> None:
        """Show ``value``; does nothing when it is already shown.

        Args:
            value: The new value.
        """
        if value == self._value:
            return
        self._value = value
        if self._hide_when_empty:
            self.display = bool(value)
        self.refresh(layout=True)

    def render(self) -> Content:
        """Fit the value to the current content width, as literal text.

        Returns:
            Content: The fitted text with no markup parsing.
        """
        return Content(self.fitted_text)


class CommandStrip(Horizontal):
```

- [ ] **Step 4: Run to see it pass**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workbench_fitted_text.py Tests/UI/test_workbench_widgets.py -q -p no:cacheprovider 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1`
Expected: all passed (`5` new plus the existing workbench-widget tests).

- [ ] **Step 5: Named mutations `markup-render` and `no-equality-guard`**

1. `markup-render`: in `FittedText.render`, change `return Content(self.fitted_text)` to `return Content.from_markup(self.fitted_text)`. Run `…/python -m pytest Tests/UI/test_workbench_fitted_text.py -q -p no:cacheprovider -k literal`: `1 failed` (the row paints `x yyy`). Restore; `1 passed`.
2. `no-equality-guard`: delete the two lines `if value == self._value:` / `return` in `set_value`. Run `-k repaints_only`: `1 failed` (`calls == [{'layout': True}]` after the unchanged set). Restore; `1 passed`.

Record both in `$EV/mutations.md`.

- [ ] **Step 6: Lint and commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/UI/Workbench/workbench_widgets.py Tests/UI/test_workbench_fitted_text.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/UI/Workbench/workbench_widgets.py Tests/UI/test_workbench_fitted_text.py
git add tldw_chatbook/UI/Workbench/workbench_widgets.py Tests/UI/test_workbench_fitted_text.py
git commit -m "feat(workbench): FittedText, literal text refitted to its own width (B1)" -m "A CSS-free shared widget whose render() asks the caller's fit for the text at the current content width: no markup parsing (R33), the caller's resolved ellipsis instead of CSS's literal one, and set_value repaints only on a change." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: `All checks passed!`, `2 files already formatted`, the commit line.

---

### Task 4: `roleplay_frame_state.py` — the pure header state (spec §5.11, R24, R33)

The one new module B1 adds (spec §5.11's module map: "header and rail state …"). Pure: no Textual import (pinned), no widget query, no I/O. Nothing imports it yet, so this commit leaves the pre-import census unchanged; Task 6 wires it into PS.

Design decisions it encodes (each pinned below):
- **R24:** `roleplay_has_unsaved_work(snapshot)` is `not snapshot.is_clean`, and `is_clean` already counts in-flight saves. It takes the snapshot through a `Protocol`, never by importing `UI/Navigation/character_conversation_navigation.py` (PS imports that module lazily; importing it here would add modules to the route census).
- **Header copy:** title `Roleplay`; the kind (`MODE_LABELS`: Characters, Personas, Dictionaries, Lore — the same nouns as the mode strip that stays until B6) is the subtitle and is never cut; the interim item label (§5.3) is `› <name>` plus ` · editing` while `_edit_mode` is not `"view"` (B1 has no work sessions; `create` shows "New character"/"New persona"); the status is `Local` or `Server: <label> · read-only`, never "Ready"; the blocked chip is `No chat provider · Settings ›`.
- **Fit:** the item is cut first (resolved ellipsis), then dropped while ` · editing` survives. Chips and status use their longest forms that fit the header width, degrading in a fixed order; the unsaved chip is short below 100 columns (spec §1.3). The `HEADER_*_CELLS` constants mirror the sheet's spacing (Task 5) and are pinned against the mounted widgets in Task 6.
- **R33:** names and labels are measured and cut as plain text and escaped only for the markup-on status chip (`escape_markup`).
- **Moved from PS** (spec §5.11's B1 extraction row, "header compose and text"): `MODE_DESCRIPTORS` (verbatim; no test patches the old `_MODE_DESCRIPTORS`, so no re-export is needed — `grep -rn _MODE_DESCRIPTORS Tests/` is empty), `mode_descriptor`, `purpose_line` (the purpose line itself stays until B6).
- `ROLEPLAY_PANE_CLASS_NAMES`: Roleplay's destination classes for the shared shell, as plain strings (B7 builds `AdaptivePaneClasses` from them; Task 5's parity test uses them now).

**Files:**
- Create: `tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py`
- Test: `Tests/UI/test_roleplay_frame_state.py` (create)

**Interfaces:**
- Consumes: `WorkbenchHeaderState` (`UI/Workbench/workbench_state.py`), `escape_markup` (`Utils/input_validation.py`), `resolve_glyph` (Task 2's entries), `MODE_LABELS` (`Widgets/Persona_Widgets/personas_state.py`), `rich.cells.cell_len`. All already loaded on the Roleplay route.
- Produces: `HEADER_TITLE`, `HEADER_PADDING_CELLS = 2`, `KIND_GAP_CELLS = 2`, `ITEM_GAP_CELLS = 1`, `CHIP_CHROME_CELLS = 3`, `STATUS_CHROME_CELLS = 3`, `UNSAVED_SHORT_BELOW_COLUMNS = 100`, `UNSAVED_CHIP`, `UNSAVED_CHIP_SHORT`, `SERVER_LABEL_MAX_CELLS = 32`, `LAST_DEGRADE_STEP = 3`, `ROLEPLAY_PANE_CLASS_NAMES: tuple[str, str, str, str, str]`, `MODE_DESCRIPTORS: dict[str, str]`, `roleplay_has_unsaved_work(snapshot) -> bool`, `ellipsize_cells(text: str, budget: int) -> str`, `runtime_server_label(app_instance: object) -> str`, `RoleplayHeaderInputs(mode, edit_mode="view", item_name="", unsaved=False, provider_blocked=False, runtime_source="local", server_label="")` (frozen), `RoleplayHeaderView(state: WorkbenchHeaderState, item: tuple[str, bool], unsaved_chip: str, blocked_chip: str, status_plain: str)` (frozen), `mode_descriptor(mode) -> str`, `purpose_line(mode, count: int | None) -> str`, `header_kind(mode) -> str`, `header_item(inputs) -> str`, `initial_header_state(mode, runtime_source) -> WorkbenchHeaderState`, `fit_header_item(value: tuple[str, bool], width: int) -> str`, `build_header_view(inputs, width: int) -> RoleplayHeaderView`; private `_required_cells(kind, chips, status) -> int` (tests read it).

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_roleplay_frame_state.py`:

```python
"""Pure contracts of the Roleplay frame state module (frame slice B1: the header).

No Textual app is mounted here (spec 5.7.1, "pure unit"): every rule of the
one-row header -- the unsaved predicate, the item fit, the degrade order, the
escaping of untrusted text -- is a plain function of plain values.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.cells import cell_len

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from Tests.UI.python_style_inventory import inventory_styles
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    RoleplayDraftSnapshot,
)
from tldw_chatbook.UI.Persona_Modules import roleplay_frame_state as fs
from tldw_chatbook.Widgets.glyph_fallback import set_ascii_glyph_mode

MODULE = Path(fs.__file__)
ROOT = Path(__file__).resolve().parents[2]
MODES = ("characters", "personas", "dictionaries", "lore")

#: Every production module B1 changes. The repo-wide Python-style ratchet is
#: red on dev for other modules, so a failure-set diff could not see a NEW
#: offender here; this pin can (ADR-161's hard floor; B0's convention).
B1_PRODUCTION_MODULES = (
    "tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py",
    "tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py",
    "tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py",
    "tldw_chatbook/UI/Screens/personas_screen.py",
    "tldw_chatbook/UI/Workbench/workbench_widgets.py",
)


@pytest.fixture
def ascii_mode():
    """Run one test with ASCII mode on and ALWAYS restore the off default."""
    set_ascii_glyph_mode(True)
    try:
        yield
    finally:
        set_ascii_glyph_mode(False)


def _snapshot(**overrides) -> RoleplayDraftSnapshot:
    values = {
        "form_dirty": False,
        "character_visual_dirty": False,
        "persona_visual_dirty": False,
        "attachments_dirty": False,
        "inflight_save_domains": (),
    }
    values.update(overrides)
    return RoleplayDraftSnapshot(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"form_dirty": True},
        {"character_visual_dirty": True},
        {"persona_visual_dirty": True},
        {"attachments_dirty": True},
        {"inflight_save_domains": ("character form",)},
        {"inflight_save_domains": ("Persona form",)},
    ],
    ids=lambda o: ",".join(o) or "clean",
)
def test_unsaved_predicate_is_the_aggregate_including_inflight_saves(overrides):
    """R24: one predicate, the aggregate snapshot; an in-flight save counts."""
    snapshot = _snapshot(**overrides)
    assert fs.roleplay_has_unsaved_work(snapshot) is (not snapshot.is_clean)
    assert fs.roleplay_has_unsaved_work(snapshot) is bool(overrides)


def test_the_module_imports_no_textual():
    """Pure (spec 5.11): no ``textual`` import anywhere in the module."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "textual" not in imported


def test_kind_names_every_mode_and_the_new_item_while_creating():
    assert [fs.header_kind(mode) for mode in MODES] == [
        "Characters",
        "Personas",
        "Dictionaries",
        "Lore",
    ]
    creating = fs.RoleplayHeaderInputs(mode="personas", edit_mode="create")
    assert fs.header_item(creating) == "New persona"
    assert (
        fs.header_item(fs.RoleplayHeaderInputs(mode="characters", edit_mode="create"))
        == "New character"
    )
    viewing = fs.RoleplayHeaderInputs(mode="lore", item_name="Blackreach")
    assert fs.header_item(viewing) == "Blackreach"


def test_item_fit_keeps_the_whole_line_when_it_fits():
    assert fs.fit_header_item(("Detective Sam", False), 40) == "› Detective Sam"
    assert (
        fs.fit_header_item(("Detective Sam", True), 40) == "› Detective Sam · editing"
    )
    assert fs.fit_header_item(("", False), 40) == ""
    # ``FittedText``'s own default value, before the first paint pushes one.
    assert fs.fit_header_item("", 10) == ""


def test_item_fit_cuts_the_name_first_and_keeps_the_state_word():
    assert fs.fit_header_item(("Detective Sam", True), 20) == "› Detecti… · editing"
    # No room for even one character of the name: the state word survives,
    # so the header still reads "Characters · editing".
    assert fs.fit_header_item(("Detective Sam", True), 12) == "· editing"
    assert fs.fit_header_item(("Detective Sam", True), 5) == ""
    assert fs.fit_header_item(("Detective Sam", False), 8) == "› Detec…"
    assert fs.fit_header_item(("Detective Sam", False), 3) == ""


@pytest.mark.parametrize(
    "name",
    [
        "探偵サム・スペード",
        "Sam 🕵️ Spade",
        "Zero\u200bwidth\u200bname",
        "[/]",
        "x" * 300,
    ],
)
def test_item_fit_never_paints_past_its_width(name):
    for width, editing in itertools.product(range(0, 60), (False, True)):
        fitted = fs.fit_header_item((name, editing), width)
        assert cell_len(fitted) <= width, (name, width, fitted)


def test_item_fit_uses_ascii_markers_in_ascii_mode(ascii_mode):
    assert (
        fs.fit_header_item(("Detective Sam", True), 40) == "> Detective Sam - editing"
    )
    assert fs.fit_header_item(("Detective Sam", False), 10) == "> Detec..."


def test_ellipsize_measures_cells_and_resolves_the_glyph():
    assert fs.ellipsize_cells("abcdef", 6) == "abcdef"
    assert fs.ellipsize_cells("abcdef", 4) == "abc…"
    assert fs.ellipsize_cells("abcdef", 1) == ""
    assert fs.ellipsize_cells("探偵サム", 5) == "探偵…"


def _all_inputs():
    for mode, edit_mode, unsaved, blocked, source in itertools.product(
        MODES, ("view", "edit"), (False, True), (False, True), ("local", "server")
    ):
        yield fs.RoleplayHeaderInputs(
            mode=mode,
            edit_mode=edit_mode,
            item_name="A rather long character name for the header",
            unsaved=unsaved,
            provider_blocked=blocked,
            runtime_source=source,
            server_label="home-tldw.example.internal:8000"
            if source == "server"
            else "",
        )


def _required(view: fs.RoleplayHeaderView) -> int:
    return fs._required_cells(
        view.state.subtitle, (view.unsaved_chip, view.blocked_chip), view.status_plain
    )


def test_header_never_says_ready_and_always_names_the_kind():
    """RP-067: the false "Ready" badge is gone; the status is the data source."""
    for inputs in _all_inputs():
        for width in (40, 80, 120, 160, 220):
            view = fs.build_header_view(inputs, width)
            assert view.state.title == "Roleplay"
            assert view.state.subtitle == fs.header_kind(inputs.mode)
            assert view.status_plain != "Ready"
            assert view.state.status_label  # never empty: "Ready" never paints
            assert view.status_plain.startswith(
                "Local" if inputs.runtime_source == "local" else "Server"
            )
    # The composed state, before any input is gathered, already names both.
    for mode, source in itertools.product(MODES, ("local", "server")):
        state = fs.initial_header_state(mode, source)
        assert (state.title, state.subtitle) == ("Roleplay", fs.header_kind(mode))
        assert state.status_label == ("Local" if source == "local" else "Server")


def test_header_fits_from_65_columns_degrading_in_order():
    """Everything fits from 65 columns (the worst case's floor), and the
    degrade steps only ever shorten: status label, Settings, then "Server"."""
    for inputs in _all_inputs():
        previous = None
        for width in range(65, 261):
            view = fs.build_header_view(inputs, width)
            assert _required(view) <= width, (inputs, width, view)
            if previous is not None:
                # Wider never paints less.
                assert cell_len(view.status_plain) >= cell_len(previous.status_plain)
                assert cell_len(view.blocked_chip) >= cell_len(previous.blocked_chip)
            previous = view


def test_unsaved_chip_is_short_below_100_columns_and_absent_when_clean():
    dirty = fs.RoleplayHeaderInputs(mode="characters", unsaved=True)
    assert fs.build_header_view(dirty, 99).unsaved_chip == "Unsaved"
    assert fs.build_header_view(dirty, 100).unsaved_chip == "Unsaved changes"
    clean = fs.RoleplayHeaderInputs(mode="characters")
    assert fs.build_header_view(clean, 160).unsaved_chip == ""


def test_blocked_chip_shows_only_for_a_blocked_destination():
    blocked = fs.RoleplayHeaderInputs(mode="characters", provider_blocked=True)
    assert fs.build_header_view(blocked, 160).blocked_chip == (
        "No chat provider · Settings ›"
    )
    assert fs.build_header_view(blocked, 66).blocked_chip in {
        "No chat provider · Settings ›",
        "No chat provider ›",
    }
    ready = fs.RoleplayHeaderInputs(mode="characters")
    assert fs.build_header_view(ready, 160).blocked_chip == ""


def test_server_status_is_escaped_for_the_markup_header_but_measured_plain():
    """R33: the label is cut as plain text, then escaped for the markup-on chip."""
    inputs = fs.RoleplayHeaderInputs(
        mode="characters", runtime_source="server", server_label="[/]home"
    )
    view = fs.build_header_view(inputs, 160)
    assert view.status_plain == "Server: [/]home · read-only"
    assert view.state.status_label == "Server: \\[/]home · read-only"
    long_label = fs.RoleplayHeaderInputs(
        mode="characters", runtime_source="server", server_label="h" * 60
    )
    status = fs.build_header_view(long_label, 220).status_plain
    assert (
        cell_len(status) == cell_len("Server:  · read-only") + fs.SERVER_LABEL_MAX_CELLS
    )


def test_runtime_server_label_reads_the_label_then_the_id():
    def app(**state):
        return SimpleNamespace(
            runtime_policy=SimpleNamespace(state=SimpleNamespace(**state))
        )

    assert (
        fs.runtime_server_label(
            app(last_known_server_label=" home ", active_server_id="t-7")
        )
        == "home"
    )
    assert (
        fs.runtime_server_label(
            app(last_known_server_label=None, active_server_id="t-7")
        )
        == "t-7"
    )
    assert fs.runtime_server_label(app(last_known_server_label=object())) == ""
    assert fs.runtime_server_label(object()) == ""


def test_purpose_line_keeps_the_pre_move_copy():
    """Moved verbatim from the screen in B1 (F-033); B6 retires it."""
    assert fs.purpose_line("characters", 2) == "Characters — who the AI plays · 2"
    assert fs.purpose_line("personas", None) == "Personas — who you play in the chat."
    assert fs.mode_descriptor("unknown") == "unknown"


def test_the_pane_class_names_are_the_shell_parts_in_field_order():
    """(shell, nav, items, work, grip): B7 builds ``AdaptivePaneClasses`` from them."""
    assert fs.ROLEPLAY_PANE_CLASS_NAMES == (
        "roleplay-shell",
        "roleplay-nav",
        "roleplay-items",
        "roleplay-shell-work",
        "roleplay-shell-grip",
    )


def test_b1_production_modules_carry_no_python_style_violation():
    violations = {
        module: [
            (write.line, write.property, write.form)
            for write in inventory_styles((ROOT / module).read_text(encoding="utf-8"))
            if write.violation
        ]
        for module in B1_PRODUCTION_MODULES
    }
    assert {module: found for module, found in violations.items() if found} == {}
```

- [ ] **Step 2: Run to see it fail**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_frame_state.py -q -p no:cacheprovider 2>&1 | tail -3`
Expected: a collection error, `ImportError: cannot import name 'roleplay_frame_state'`.

- [ ] **Step 3: Create the module**

Create `tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py`:

```python
"""Pure state for the Roleplay destination frame (spec sections 1.3 and 5.11).

The Roleplay screen gathers inputs and pushes the views computed here into its
widgets. Everything in this module is a plain value or a pure function: no
Textual import, no widget query, no I/O, so every rule is unit-tested without
mounting a screen. Later frame slices extend it (rail state, the work-session
reducer, the keyboard projection); slice B1 brings the one-row header.

Untrusted text (spec R33). Item names and server labels arrive raw. They are
measured and cut as PLAIN text with resolved glyphs, then escaped only on the
way into a markup-on surface: the shared header's status chip gets
``escape_markup``. The fitted item label and the chips render literal
``Content`` and are never escaped (a backslash would paint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rich.cells import cell_len

from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchHeaderState
from tldw_chatbook.Utils.input_validation import escape_markup
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph
from tldw_chatbook.Widgets.Persona_Widgets.personas_state import MODE_LABELS

#: The destination's one public name (F-034: it matches the nav label).
HEADER_TITLE = "Roleplay"

#: Cells the inline header spends around its parts. They mirror
#: ``css/features/_roleplay.tcss``; ``Tests/UI/test_roleplay_header.py`` pins
#: them against the mounted widgets' computed styles, so a CSS edit that
#: forgets these fails there instead of clipping the status chip.
HEADER_PADDING_CELLS = 2  # #personas-header: padding 0 1
KIND_GAP_CELLS = 2  # the kind subtitle: margin-left $ds-space-2
ITEM_GAP_CELLS = 1  # #personas-header-item: margin-left $ds-space-1
CHIP_CHROME_CELLS = 3  # each chip: margin-left 1 + padding 0 1
STATUS_CHROME_CELLS = 3  # the status chip: margin-left 1 + padding 0 1

#: Below this many columns the unsaved chip takes its short form (spec 1.3).
UNSAVED_SHORT_BELOW_COLUMNS = 100
UNSAVED_CHIP = "Unsaved changes"
UNSAVED_CHIP_SHORT = "Unsaved"
#: A server label longer than this is cut (with the resolved ellipsis).
SERVER_LABEL_MAX_CELLS = 32
#: Degrade steps when the chips and the status do not fit (spec 1.3):
#: 0 full forms; 1 the status drops the server label; 2 the blocked chip
#: drops "Settings"; 3 the status reads just "Server".
LAST_DEGRADE_STEP = 3

#: Roleplay's destination classes for the shared adaptive pane shell, in
#: ``AdaptivePaneClasses`` field order (shell, nav, items, work, grip). Each
#: carries a Roleplay split prefix, so the shell rules in
#: ``css/features/_roleplay.tcss`` stay lazy (spec 2.12 item 2). Plain strings,
#: so this module stays free of Textual; frame slice B7 mounts the shell with
#: ``AdaptivePaneClasses(*ROLEPLAY_PANE_CLASS_NAMES)``.
ROLEPLAY_PANE_CLASS_NAMES: tuple[str, str, str, str, str] = (
    "roleplay-shell",
    "roleplay-nav",
    "roleplay-items",
    "roleplay-shell-work",
    "roleplay-shell-grip",
)

_ELLIPSIS = "…"
_SEPARATOR = "·"
_GO = "›"

#: One-line "what this kind is" copy: the purpose line under the header and
#: the mode chips' tooltips, until frame slice B6's rail glosses replace both
#: (spec G8). Moved verbatim from ``personas_screen.py`` in B1.
MODE_DESCRIPTORS: dict[str, str] = {
    "characters": "Characters — who the AI plays.",
    # F-034: the descriptor teaches the genre convention (characters = who
    # the AI plays, personas = who YOU play) instead of the vague "assistant
    # profiles" - without reviving the retired human-identity framing.
    "personas": "Personas — who you play in the chat.",
    "prompts": "Prompts — moving to the Library.",
    "dictionaries": "Dictionaries — text find/replace rules.",
    "lore": "Lore — world facts injected on keywords.",
}


class DraftSnapshotLike(Protocol):
    """The one property of ``RoleplayDraftSnapshot`` the predicate reads.

    A Protocol, not an import: the snapshot's module defines Roleplay's
    navigation dialogs, which the screen imports lazily, and this module must
    not drag them onto the route's pre-import census.
    """

    @property
    def is_clean(self) -> bool:
        """True when no domain is dirty and no save is in flight."""
        ...


def roleplay_has_unsaved_work(snapshot: DraftSnapshotLike) -> bool:
    """The one ADR-046 unsaved predicate (spec R24, section 3.12).

    True while any Roleplay draft domain is dirty OR a save is still in
    flight: ``is_clean`` already counts in-flight saves, so a chip driven by
    this stays on until the save completes, never just ``has_unsaved_changes``.

    Args:
        snapshot: ``PersonasScreen._aggregate_roleplay_draft_snapshot()``.

    Returns:
        Whether the destination holds work that is not yet safely saved.
    """
    return not snapshot.is_clean


def ellipsize_cells(text: str, budget: int) -> str:
    """Cut ``text`` to ``budget`` terminal cells, ending in the resolved ellipsis.

    Measures cells, not characters, so wide (CJK, emoji) and zero-width
    characters fit by what they paint. The same rule as the rail rows'
    fitter in ``Widgets/adaptive_pane_shell.py``, kept here so this module
    stays free of Textual.

    Args:
        text: Plain (unescaped) text.
        budget: Cells available.

    Returns:
        ``text`` when it fits; ``""`` when not even one character fits before
        the ellipsis; otherwise the longest fitting head plus the ellipsis.
    """
    if cell_len(text) <= budget:
        return text
    ellipsis = resolve_glyph(_ELLIPSIS)
    room = budget - cell_len(ellipsis)
    head = ""
    for character in text:
        if cell_len(head + character) > room:
            break
        head += character
    head = head.rstrip()
    return f"{head}{ellipsis}" if head else ""


def runtime_server_label(app_instance: object) -> str:
    """The active server's display label, read the way the Library reads it.

    Args:
        app_instance: The app (or a test double); only attributes are read.

    Returns:
        ``runtime_policy.state.last_known_server_label``, else its
        ``active_server_id``, else ``""``. Non-string values count as absent.
    """
    runtime_policy = getattr(app_instance, "runtime_policy", None)
    state = getattr(runtime_policy, "state", None)
    for name in ("last_known_server_label", "active_server_id"):
        value = getattr(state, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


@dataclass(frozen=True)
class RoleplayHeaderInputs:
    """Everything the one-row header shows, gathered by the screen.

    Attributes:
        mode: The active kind (``"characters"``, ``"personas"``,
            ``"dictionaries"`` or ``"lore"``).
        edit_mode: ``"view"``, ``"create"`` or ``"edit"``.
        item_name: The selected item's name, raw (untrusted), or ``""``.
        unsaved: ``roleplay_has_unsaved_work`` of the aggregate snapshot.
        provider_blocked: The destination-wide block: no ready chat provider
            for character chats (``console_handoff_readiness()``, which
            reads no selection).
        runtime_source: ``"local"`` or ``"server"``.
        server_label: ``runtime_server_label`` (raw), or ``""``.
    """

    mode: str
    edit_mode: str = "view"
    item_name: str = ""
    unsaved: bool = False
    provider_blocked: bool = False
    runtime_source: str = "local"
    server_label: str = ""


@dataclass(frozen=True)
class RoleplayHeaderView:
    """The header as it should paint at one width.

    Attributes:
        state: For ``DestinationHeader.sync_state``: the title, the kind as
            the subtitle and the status chip text (escaped: that chip is a
            markup-on ``Static``).
        item: The fitted item label's value, ``(text, editing)``, for
            ``fit_header_item`` (spec 5.3 interim: ``› <item>`` until B5b-2).
        unsaved_chip: ``"Unsaved changes"``, ``"Unsaved"`` or ``""`` (hidden).
        blocked_chip: The blocked-destination chip text, or ``""`` (hidden).
        status_plain: The status chip text before escaping, as it paints.
    """

    state: WorkbenchHeaderState
    item: tuple[str, bool]
    unsaved_chip: str
    blocked_chip: str
    status_plain: str


def mode_descriptor(mode: str) -> str:
    """The kind's one-line descriptor (falls back to its label, then its id)."""
    return MODE_DESCRIPTORS.get(mode, MODE_LABELS.get(mode, mode))


def purpose_line(mode: str, count: int | None) -> str:
    """The descriptor plus the live item count on one line (F-033).

    Args:
        mode: The active kind.
        count: Its item count, or ``None`` for a kind without one.

    Returns:
        ``"Characters — who the AI plays · 2"``, or the bare descriptor.
    """
    descriptor = mode_descriptor(mode)
    if count is None:
        return descriptor
    return f"{descriptor.rstrip('.')} · {count}"


def header_kind(mode: str) -> str:
    """The kind noun the header names (spec 1.3, RP-029); never cut."""
    return MODE_LABELS.get(mode, mode)


def header_item(inputs: RoleplayHeaderInputs) -> str:
    """The interim item text (spec 5.3): the new item's noun while creating."""
    if inputs.edit_mode == "create":
        return "New persona" if inputs.mode == "personas" else "New character"
    return inputs.item_name


def initial_header_state(mode: str, runtime_source: str) -> WorkbenchHeaderState:
    """The state the header is composed with, before any input is gathered.

    It already names the kind and the data source, so the shared header's
    default "Ready" chip never paints, whatever order the first mount and the
    first ``_update_title`` run in.

    Args:
        mode: The active kind.
        runtime_source: ``"local"`` or ``"server"``.

    Returns:
        The title, the kind as the subtitle and the bare data-source word.
    """
    return WorkbenchHeaderState(
        title=HEADER_TITLE,
        subtitle=header_kind(mode),
        status_label="Server" if runtime_source == "server" else "Local",
    )


def fit_header_item(value: tuple[str, bool], width: int) -> str:
    """Fit ``› <item>[ · editing]`` into ``width`` cells.

    The name is cut first (ending in the resolved ellipsis), then dropped
    with its marker; `` · editing`` survives while it fits, so the header
    still reads ``Characters · editing``. The kind is not in this text: it
    is the header subtitle and is never cut.

    Args:
        value: ``(item, editing)`` from ``RoleplayHeaderView.item``; an empty
            value (``FittedText``'s default, before the first paint) paints
            nothing.
        width: The label's content width in cells.

    Returns:
        The plain text to paint (never markup).
    """
    item, editing = value or ("", False)
    state = f"{resolve_glyph(_SEPARATOR)} editing" if editing else ""
    suffix = f" {state}" if state else ""
    if item:
        marker = f"{resolve_glyph(_GO)} "
        whole = f"{marker}{item}{suffix}"
        if cell_len(whole) <= width:
            return whole
        cut = ellipsize_cells(item, width - cell_len(marker) - cell_len(suffix))
        if cut:
            return f"{marker}{cut}{suffix}"
    return state if cell_len(state) <= width else ""


def _status_text(inputs: RoleplayHeaderInputs, step: int) -> str:
    """The status chip at one degrade step (spec 1.3; DESIGN.md:115)."""
    if inputs.runtime_source != "server":
        return "Local"
    read_only = f"{resolve_glyph(_SEPARATOR)} read-only"
    label = ellipsize_cells(inputs.server_label, SERVER_LABEL_MAX_CELLS)
    if step == 0 and label:
        return f"Server: {label} {read_only}"
    if step < LAST_DEGRADE_STEP:
        return f"Server {read_only}"
    return "Server"


def _blocked_text(step: int) -> str:
    """The blocked-destination chip at one degrade step (never "Ready")."""
    go = resolve_glyph(_GO)
    if step < 2:
        return f"No chat provider {resolve_glyph(_SEPARATOR)} Settings {go}"
    return f"No chat provider {go}"


def _required_cells(kind: str, chips: tuple[str, ...], status: str) -> int:
    """Cells the header needs with an empty item label."""
    return (
        HEADER_PADDING_CELLS
        + cell_len(HEADER_TITLE)
        + KIND_GAP_CELLS
        + cell_len(kind)
        + ITEM_GAP_CELLS
        + sum(cell_len(chip) + CHIP_CHROME_CELLS for chip in chips if chip)
        + cell_len(status)
        + STATUS_CHROME_CELLS
    )


def build_header_view(inputs: RoleplayHeaderInputs, width: int) -> RoleplayHeaderView:
    """The header view for ``inputs`` on a ``width``-column header.

    The unsaved chip is short below ``UNSAVED_SHORT_BELOW_COLUMNS``. Then the
    longest forms that fit win, degrading in this order until they fit:
    drop the server label, shorten the blocked chip, then show the status as
    ``Server``. The title and the kind never change, and the item label takes
    whatever is left (``fit_header_item``).

    Args:
        inputs: Gathered by the screen.
        width: The header's outer width in cells.

    Returns:
        The view to paint.
    """
    kind = header_kind(inputs.mode)
    unsaved = ""
    if inputs.unsaved:
        unsaved = (
            UNSAVED_CHIP_SHORT if width < UNSAVED_SHORT_BELOW_COLUMNS else UNSAVED_CHIP
        )
    for step in range(LAST_DEGRADE_STEP + 1):
        blocked = _blocked_text(step) if inputs.provider_blocked else ""
        status = _status_text(inputs, step)
        if _required_cells(kind, (unsaved, blocked), status) <= width:
            break
    return RoleplayHeaderView(
        state=WorkbenchHeaderState(
            title=HEADER_TITLE,
            subtitle=kind,
            status="ready",
            status_label=escape_markup(status),
        ),
        item=(header_item(inputs), inputs.edit_mode != "view"),
        unsaved_chip=unsaved,
        blocked_chip=blocked,
        status_plain=status,
    )
```

- [ ] **Step 4: Run to see it pass; lint**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_frame_state.py -q -p no:cacheprovider 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py Tests/UI/test_roleplay_frame_state.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py Tests/UI/test_roleplay_frame_state.py
EOF
```

Expected: `27 passed`, `All checks passed!`, `2 files already formatted`.

- [ ] **Step 5: Named mutations `has-unsaved-only` and `literal-ellipsis`**

1. `has-unsaved-only`: change `return not snapshot.is_clean` to `return bool(getattr(snapshot, "form_dirty", False))`. Run `…/python -m pytest Tests/UI/test_roleplay_frame_state.py -q -p no:cacheprovider -k unsaved_predicate`: `5 failed` (every non-form domain, including both in-flight cases). Restore; `7 passed`.
2. `literal-ellipsis`: in `ellipsize_cells` change `ellipsis = resolve_glyph(_ELLIPSIS)` to `ellipsis = _ELLIPSIS`. Run `-k ascii_markers`: `1 failed` (`> Detecti…` painted where `> Detec...` is required). Restore; `1 passed`.

Record both in `$EV/mutations.md`.

- [ ] **Step 6: Confirm the census is untouched, then commit**

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b1-evidence
"$EV/preimport_measure.sh" base && "$EV/preimport_measure.sh" head
"$PY" "$EV/preimport_raise.py"
cd $MAIN/.worktrees/roleplay-b1
git add tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py Tests/UI/test_roleplay_frame_state.py
git commit -m "feat(roleplay): pure frame state for the one-row header (B1)" -m "roleplay_frame_state.py: the R24 unsaved predicate over the aggregate snapshot, the header inputs and view, the cell-measured item fit with the resolved ellipsis, the fixed degrade order, plain-then-escaped untrusted text, the moved mode descriptors and purpose line, and Roleplay's pane class names. Nothing imports it yet." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: identical `base` and `head` census lines (`modules 557` on dev at `8c4dfe59a2` and on the base arm at `a793acbef5` and `8d502ba250`; the module is not imported yet), `raised: nothing (head within base limits) | ledger row: none`, the commit line.

---

### Task 5: The lazy Roleplay stylesheet (R18, G18, spec §2.12 items 2-4)

`css/features/_roleplay.tcss` holds every Roleplay rule; `build_css.py` splits it whole into `screen_feature_roleplay.tcss`, and `TldwCli._ensure_screen_owned_css` parses that on the first visit (initial tab or in-app navigation), never `PersonasScreen.CSS_PATH`. The header rules target `#personas-header.personas-header-inline`, which Task 6 composes, so this commit changes nothing on screen.

Measured while planning (simulated, then built for real in the dry run): with the R18 prefixes alone, the three header-child rules and the grip `:hover, .-active` rule keep tokens no Roleplay owner claims and stay in boot (+556 B, measured while planning, where B1's boot gate is net ≤ 0). Claiming exactly `workbench-header-title`, `workbench-header-subtitle`, `workbench-header-status` and `-active` moves everything: the module's boot remainder is its banner only. Claiming shared tokens is safe only while every Roleplay selector also carries a Roleplay token (a bare `.-active` rule here would restyle every button after the first Ctrl+4), so `test_every_roleplay_selector_carries_a_roleplay_token` enforces it. ADR-212 §3 ("Roleplay's copy must not add another `.-active` boot rule") holds. The broad-selector census cannot see lazy sheets, so `test_the_sheet_has_no_bare_type_subject` pins this one. No `$ds-roleplay-*` token is needed (G17: every dimension has a scale token).

The two chips are words that must be read (spec §4.12), so they take readable foregrounds, not the decorative status hues: measured while planning over the 70 registered themes, `$warning` (behind `$ds-status-unsaved` and `$ds-status-warning`) falls below AA on the panel in 29 themes and `$error` (behind `$ds-status-blocked`) in 40, while Textual's generated `$text-warning` and `$text-error` clear AA in all 70. `$ds-status-error-readable` (`$text-error`, task-2230) already exists; this task adds its sibling `$ds-status-warning-readable: $text-warning;` to `core/_variables.tcss` (+45 B of boot; every split sheet's variable preamble gains the line, so every generated sheet is rebuilt), and `test_header_chip_words_clear_aa_on_every_theme` pins both chips on every theme.

The nine never-composed `#personas-*` items in `components/_agentic_terminal.tcss` go too (−216 B; none is the last member of its group; no Python or test references). Boot after this task: −119 B (−216 for the dead items, +52 for the new module's banner, +45 for the token line).

**Files:**
- Create: `tldw_chatbook/css/features/_roleplay.tcss`; generated `tldw_chatbook/css/screen_feature_roleplay.tcss`; `Tests/UI/test_roleplay_stylesheet.py`
- Modify: `tldw_chatbook/css/build_css.py` (`CSS_MODULES`, `SCREEN_OWNED_SPLITS`), `tldw_chatbook/css/components/_agentic_terminal.tcss`, `tldw_chatbook/css/core/_variables.tcss` (`$ds-status-warning-readable`), `Tests/Architecture/test_builtin_theme_contrast.py` (its readable-token guard), `tldw_chatbook/app.py` (Constants import, `_SCREEN_OWNED_ROUTE_CSS`), regenerated `tldw_chatbook/css/tldw_cli_modular.tcss` and every `tldw_chatbook/css/screen_*.tcss` split sheet (their variable preamble), `Tests/UI/test_css_build_integrity.py`, `Tests/UI/test_consolidated_css_harness.py`, `Tests/UI/test_roleplay_frame_harness.py`

**Interfaces:**
- Consumes: `ROLEPLAY_PANE_CLASS_NAMES` (Task 4); `AdaptivePaneClasses` (B0, `Widgets/adaptive_pane_shell.py`); `LIBRARY_ADAPTIVE_READER_CLASSES` (B0); `ROLEPLAY_SHEET`, `open_styled_roleplay`, `roleplay_full_app(on_home=...)`, `styled_tiers`, `settle` (Task 1); `build_css.SCREEN_OWNED_SPLITS`, `ScreenOwnedSplit`; `scan_bundle_only_harnesses`, `_TESTS_ROOT` (`Tests/UI/test_consolidated_css_harness.py`); `_DIM` (`Tests/UI/test_component_pattern_governance.py`).
- Produces: selectors `#personas-header.personas-header-inline` (+ its `.workbench-header-title/-subtitle/-status` children), `#personas-header-tail`, `#personas-header-item`, `#personas-header-unsaved`, `#personas-header-blocked` (Task 6 composes them); `.roleplay-shell`, `.roleplay-shell > .roleplay-shell-work`, `.roleplay-shell > .roleplay-shell-grip` (+ `:hover`, `.-active`, `:focus`) for B7; `TldwCli._SCREEN_OWNED_ROUTE_CSS[TAB_PERSONAS] == ("screen_feature_roleplay.tcss",)`; the design-system token `$ds-status-warning-readable` (`$text-warning`).

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_roleplay_stylesheet.py` (Task 6 tightens one line once the task-523 rule leaves boot):

```python
"""Roleplay's lazy stylesheet: ownership, anchoring and parity (frame slice B1).

Spec 2.12 items 2-3, R18 and G18. ``features/_roleplay.tcss`` is split whole
into ``screen_feature_roleplay.tcss``, which the app parses on the first visit
to Roleplay. Static checks only (no app is mounted): the visit itself is pinned
by the harness self-tests (``test_roleplay_frame_harness.py``).
"""

from __future__ import annotations

import ast
import re
from dataclasses import astuple, fields
from pathlib import Path

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from textual.css.model import SelectorType
from textual.css.parse import parse_selectors

from Tests.UI import test_consolidated_css_harness as harness_scan
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, CSS_DIR
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import TAB_PERSONAS
from tldw_chatbook.css import build_css
from tldw_chatbook.UI.Persona_Modules.roleplay_frame_state import (
    ROLEPLAY_PANE_CLASS_NAMES,
)
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.adaptive_pane_shell import AdaptivePaneClasses
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LIBRARY_ADAPTIVE_READER_CLASSES,
)

SHEET_NAME = "screen_feature_roleplay.tcss"
SHEET = CSS_DIR / SHEET_NAME
SOURCE = CSS_DIR / "features" / "_roleplay.tcss"
LIBRARY_SOURCE = CSS_DIR / "features" / "_library.tcss"
#: Shared tokens the Roleplay split claims so its copies leave boot. They are
#: never an anchor: a selector made only of them would restyle other screens.
SHARED_CLAIMS = frozenset(
    {
        "workbench-header-title",
        "workbench-header-subtitle",
        "workbench-header-status",
        "-active",
    }
)
_TOKEN = re.compile(r"[#.]([A-Za-z0-9_-]+)")
_PREAMBLE_END = ".tldw-agentic-split-preamble-end { }"
ROLEPLAY_PANE_CLASSES = AdaptivePaneClasses(*ROLEPLAY_PANE_CLASS_NAMES)


def _roleplay_prefixes() -> tuple[str, ...]:
    for split in build_css.SCREEN_OWNED_SPLITS:
        for owner, sheet in split.sheets.items():
            if sheet == SHEET_NAME:
                return tuple(split.prefixes[owner])
    raise AssertionError(f"no screen-owned split writes {SHEET_NAME}")


def _anchored(member: str) -> bool:
    """True when a selector carries at least one Roleplay-only token."""
    own = [p for p in _roleplay_prefixes() if p not in SHARED_CLAIMS]
    return any(
        token not in SHARED_CLAIMS
        and any(token == p or token.startswith(f"{p}-") for p in own)
        for token in _TOKEN.findall(member)
    )


def _moved_rules(text: str) -> list[tuple[str, str]]:
    """``(selector, body)`` per rule after the sheet's variable preamble."""
    body = text.split(_PREAMBLE_END, 1)[1] if _PREAMBLE_END in text else text
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return [
        (" ".join(selector.split()), declarations)
        for selector, declarations in re.findall(r"([^{}]+)\{([^{}]*)\}", body)
    ]


def _members(selector: str) -> list[str]:
    return [" ".join(member.split()) for member in selector.split(",")]


def test_the_sheet_is_route_loaded_and_never_a_screen_css_path():
    """MF-02: Textual loads a screen's CSS_PATH under every app, so the sheet
    rides the route map, parsed by ``_ensure_screen_owned_css``."""
    assert TldwCli._SCREEN_OWNED_ROUTE_CSS[TAB_PERSONAS] == (SHEET_NAME,)
    assert not getattr(PersonasScreen, "CSS_PATH", None)


def test_every_roleplay_selector_carries_a_roleplay_token():
    """Claiming shared tokens is safe only while each selector is anchored."""
    rules = _moved_rules(SHEET.read_text(encoding="utf-8"))
    assert len(rules) >= 14
    for selector, _body in rules:
        for member in _members(selector):
            assert _anchored(member), f"{member!r} has no Roleplay token"
    # The guard itself is not vacuous:
    assert not _anchored(".workbench-header-title")
    assert not _anchored("Button.-active")
    assert _anchored("#personas-header .workbench-header-title")


def test_the_sheet_has_no_bare_type_subject():
    """The broad-selector census counts only boot sheets (spec 2.12 item 1),
    so this sheet pins itself: every subject is an id or a class."""
    for selector, _body in _moved_rules(SHEET.read_text(encoding="utf-8")):
        for selector_set in parse_selectors(selector):
            subject = selector_set.selectors[-1]
            assert subject.type in (SelectorType.ID, SelectorType.CLASS), selector
    bare = parse_selectors("#personas-header Static")[0].selectors[-1]
    assert bare.type is SelectorType.TYPE  # the check can fail


def test_the_boot_bundle_carries_no_roleplay_rule():
    """Every Roleplay rule moved: the bundle keeps only the module banner, and
    no header or shell token B1 adds is left in boot."""
    bundle = BUNDLED_STYLESHEET.read_text(encoding="utf-8")
    banner = "/* ===== MODULE: features/_roleplay.tcss ===== */"
    section = bundle.split(banner, 1)[1].split("/* ===== MODULE:", 1)[0]
    assert section.strip() == ""
    frame_tokens = {
        token
        for token in _TOKEN.findall(bundle)
        if token.startswith(("personas-header-", *ROLEPLAY_PANE_CLASS_NAMES))
    }
    assert frame_tokens == set()


def test_roleplay_pane_classes_carry_a_roleplay_split_prefix():
    """A shell class without the owner prefix would pin its rules to boot."""
    for css_class in astuple(ROLEPLAY_PANE_CLASSES):
        assert _anchored(f".{css_class}"), css_class


def _role_rules(text: str, classes: AdaptivePaneClasses) -> set[tuple[str, tuple]]:
    """The rules that use a shell class, with each class replaced by its role."""
    roles = {
        getattr(classes, field.name): "{" + field.name + "}"
        for field in fields(classes)
    }
    found = set()
    for selector, body in _moved_rules(text):
        if not any(token in roles for token in _TOKEN.findall(selector)):
            continue
        normalised = sorted(
            _TOKEN.sub(
                lambda m: m.group(0)[0] + roles.get(m.group(1), m.group(1)), member
            )
            for member in _members(selector)
        )
        declarations = tuple(
            sorted(" ".join(d.split()) for d in body.split(";") if d.strip())
        )
        found.add((", ".join(normalised), declarations))
    return found


def test_roleplay_shell_rules_match_the_library_block():
    """Spec 2.12 item 2: Roleplay's copy equals the Library's re-keyed block
    once class names are normalised to roles; six rules, nav/items have none."""
    library = _role_rules(
        LIBRARY_SOURCE.read_text(encoding="utf-8"), LIBRARY_ADAPTIVE_READER_CLASSES
    )
    roleplay = _role_rules(SOURCE.read_text(encoding="utf-8"), ROLEPLAY_PANE_CLASSES)
    assert len(library) == 6
    assert roleplay == library


def test_parity_sees_a_dropped_focus_reverse():
    """Negative control: the parity check is not vacuous."""
    mutated = SOURCE.read_text(encoding="utf-8").replace(
        "text-style: bold reverse;", "text-style: bold;"
    )
    library = _role_rules(
        LIBRARY_SOURCE.read_text(encoding="utf-8"), LIBRARY_ADAPTIVE_READER_CLASSES
    )
    assert _role_rules(mutated, ROLEPLAY_PANE_CLASSES) != library


def test_the_split_sheet_scan_reports_a_bundle_only_roleplay_harness(
    tmp_path, monkeypatch
):
    """``_SPLIT_SHEET_OWNERS`` lists no owner for the Roleplay sheet, so a
    harness pinned to the bundle alone that queries a Roleplay-only token is
    reported (spec B1: PersonasScreen must never exempt one)."""
    tests_root = tmp_path / "Tests"
    tests_root.mkdir()
    (tests_root / "test_synthetic_roleplay_harness.py").write_text(
        "from textual.app import App\n"
        "from Tests.UI.consolidated_css import BUNDLED_STYLESHEET\n"
        "from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen\n\n"
        "class BundleOnlyRoleplayHarness(App):\n"
        "    CSS_PATH = str(BUNDLED_STYLESHEET)\n\n"
        "    def on_mount(self):\n"
        "        self.push_screen(PersonasScreen(self))\n\n"
        "QUERY = '#personas-header-unsaved'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(harness_scan, "_TESTS_ROOT", tests_root)
    findings = harness_scan.scan_bundle_only_harnesses()
    assert [(cls, sheet) for _path, cls, sheet, _used in findings] == [
        ("BundleOnlyRoleplayHarness", SHEET_NAME)
    ]


#: Tests that build a real ``TldwCli`` and push ``PersonasScreen`` themselves
#: but assert no geometry (they check which center views are mounted), so the
#: sheet the push skips cannot mislead them.
_GEOMETRY_FREE_FULL_APP_PUSHES = frozenset(
    {"Tests/UI/test_personas_deferred_center_views.py"}
)


def _bare_roleplay_pushes(tests_root: Path) -> list[str]:
    """``path::function`` for every test function that builds the real app
    (``_build_test_app``) and constructs ``PersonasScreen`` itself without
    loading the route's sheet first (``_ensure_screen_owned_css``)."""
    found = []
    for path in sorted(tests_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "PersonasScreen" not in text or "_build_test_app" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            called = {
                getattr(call.func, "id", None) or getattr(call.func, "attr", None)
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            }
            if {"_build_test_app", "PersonasScreen"} <= called and (
                "_ensure_screen_owned_css" not in called
            ):
                relative = path.relative_to(tests_root.parent).as_posix()
                found.append(f"{relative}::{node.name}")
    return found


def test_no_full_app_test_pushes_roleplay_without_its_sheet(tmp_path):
    """A real app that pushes ``PersonasScreen`` itself skips the route loader,
    so the inline header paints without its rules (TASK-32187's Watchlists
    trap). Each such test loads the sheet first or is listed as geometry-free.
    Negative control: a synthetic offender is reported, a loaded one is not."""
    repo = Path(__file__).resolve().parents[2]
    offenders = [
        hit
        for hit in _bare_roleplay_pushes(repo / "Tests")
        if hit.split("::")[0] not in _GEOMETRY_FREE_FULL_APP_PUSHES
    ]
    assert offenders == []
    synthetic = tmp_path / "Tests"
    synthetic.mkdir()
    (synthetic / "test_synthetic_push.py").write_text(
        "async def test_bare():\n"
        "    app = _build_test_app()\n"
        "    await app.push_screen(PersonasScreen(app))\n\n\n"
        "async def test_loaded():\n"
        "    app = _build_test_app()\n"
        "    app._ensure_screen_owned_css('personas')\n"
        "    await app.push_screen(PersonasScreen(app))\n",
        encoding="utf-8",
    )
    assert _bare_roleplay_pushes(synthetic) == [
        "Tests/test_synthetic_push.py::test_bare"
    ]


def test_header_chip_words_clear_aa_on_every_theme():
    """Spec 4.12: the chips are words that must be READ, so their colour
    tokens resolve to AA (4.5:1) on the header's panel surface in every
    registered theme. Negative control: the decorative hue fails somewhere."""
    from textual.color import Color

    from Tests.UI.test_theme_contrast import (
        AA,
        _ratio,
        _resolve_color,
        _resolved_variables,
    )
    from tldw_chatbook.css.Themes.themes import ALL_THEMES

    variables = (CSS_DIR / "core" / "_variables.tcss").read_text(encoding="utf-8")
    alias = dict(re.findall(r"^\$(ds-[\w-]+):\s*\$([\w-]+);", variables, re.M))
    rules = dict(_moved_rules(SOURCE.read_text(encoding="utf-8")))

    def contrast(theme, textual_name: str) -> float:
        resolved = _resolved_variables(theme)
        panel = Color.parse(resolved["panel"])
        return _ratio(_resolve_color(resolved[textual_name], panel).hex, panel.hex)

    for selector in ("#personas-header-unsaved", "#personas-header-blocked"):
        token = re.search(r"color:\s*\$([\w-]+);", rules[selector])[1]
        for theme in ALL_THEMES:
            ratio = contrast(theme, alias[token])
            assert ratio >= AA, (selector, token, theme.name, round(ratio, 2))
    assert min(contrast(theme, "warning") for theme in ALL_THEMES) < AA


def test_the_roleplay_source_has_no_numeric_dimension_literal():
    """ADR-161's hard zero, pinned for this sheet: the repo-wide
    ``test_dimension_literal_ratchet`` is red on dev for other sheets, so a
    failure-set diff could not see a new literal here."""
    from Tests.UI.test_component_pattern_governance import _DIM

    text = re.sub(r"/\*.*?\*/", "", SOURCE.read_text(encoding="utf-8"), flags=re.S)
    assert _DIM.findall(text) == []
    assert _DIM.search("#probe { height: 1; }")  # the pattern can fire
```

Append the visit tests to `Tests/UI/test_roleplay_frame_harness.py`. Old:
```python
from Tests.UI.roleplay_frame_harness import (
    ROLEPLAY_SHEET,
    ROLEPLAY_SIZES,
    RoleplayMockApp,
    StyledRoleplayMockApp,
    assert_painted_inside,
    drop_rule_from_loaded_sheet,
    roleplay_full_app,
    seed_mock_characters,
)
```
New:
```python
from Tests.UI.roleplay_frame_harness import (
    ROLEPLAY_SHEET,
    ROLEPLAY_SIZES,
    RoleplayMockApp,
    StyledRoleplayMockApp,
    assert_painted_inside,
    drop_rule_from_loaded_sheet,
    open_styled_roleplay,
    roleplay_full_app,
    seed_mock_characters,
    settle,
    styled_tiers,
)
```
Old:
```python
def test_dropping_a_rule_that_is_not_there_is_a_loud_error():
```
New:
```python
def test_app_stylesheets_carry_the_roleplay_sheet():
    """APP_STYLESHEETS derives from the build's splits, so it gains the sheet."""
    assert ROLEPLAY_SHEET in APP_STYLESHEETS
    assert ROLEPLAY_SHEET.is_file()


@styled_tiers
async def test_only_the_styled_tiers_carry_the_roleplay_sheet(
    styled_tier, mock_app_instance, one_character
):
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=(120, 36)
    ) as pilot:
        assert pilot.app.stylesheet.has_source(str(ROLEPLAY_SHEET), "")
    unstyled = RoleplayMockApp(mock_app_instance)
    async with unstyled.run_test(size=(120, 36)) as pilot:
        await settle(pilot)
        assert not unstyled.stylesheet.has_source(str(ROLEPLAY_SHEET), "")


@pytest.mark.parametrize("entry", ["initial_tab", "ctrl+4"])
async def test_the_full_app_loads_the_sheet_on_the_first_visit_only(
    entry, one_character
):
    """AC#8: never parsed at boot (Home), parsed by either real route: the
    initial tab (``_push_initial_screen``) and Ctrl+4 (in-app navigation)."""
    at_home = []
    async with roleplay_full_app(
        size=(120, 36),
        entry=entry,
        on_home=lambda app: at_home.append(
            app.stylesheet.has_source(str(ROLEPLAY_SHEET), "")
        ),
    ) as pilot:
        assert type(pilot.app.screen).__name__ == "PersonasScreen"
        assert pilot.app.stylesheet.has_source(str(ROLEPLAY_SHEET), "")
    assert at_home == ([False] if entry == "ctrl+4" else [])


def test_dropping_a_rule_that_is_not_there_is_a_loud_error():
```

Edit the CSS-integrity suites. In `Tests/UI/test_css_build_integrity.py`, Old:
```python
    _CSS_ROOT / "screen_feature_watchlists.tcss",
)
```
New:
```python
    _CSS_ROOT / "screen_feature_watchlists.tcss",
    _CSS_ROOT / "screen_feature_roleplay.tcss",
)
```
Old:
```python
    ["features/_evals.tcss", "features/_scheduling.tcss", "features/_workflows.tcss"],
```
New:
```python
    [
        "features/_evals.tcss",
        "features/_scheduling.tcss",
        "features/_workflows.tcss",
        "features/_roleplay.tcss",
    ],
```
Old:
```python
        (
            "tldw_chatbook.UI.Screens.watchlists_collections_screen",
            "WatchlistsCollectionsScreen",
        ),
    ]:
        try:
            screen_cls = getattr(importlib.import_module(screen_module), screen_name)
        except ImportError as exc:  # optional-deps environments
            pytest.skip(f"{screen_module} unavailable here: {exc}")
        css_path = getattr(screen_cls, "CSS_PATH", None) or []
        for entry in css_path:
```
New:
```python
        (
            "tldw_chatbook.UI.Screens.watchlists_collections_screen",
            "WatchlistsCollectionsScreen",
        ),
        ("tldw_chatbook.UI.Screens.personas_screen", "PersonasScreen"),
    ]:
        try:
            screen_cls = getattr(importlib.import_module(screen_module), screen_name)
        except ImportError as exc:  # optional-deps environments
            pytest.skip(f"{screen_module} unavailable here: {exc}")
        css_path = getattr(screen_cls, "CSS_PATH", None) or []
        # A bare str would be walked one character at a time and never match
        # (Roleplay frame B1 review): normalise like the modal check above.
        entries = [css_path] if isinstance(css_path, (str, Path)) else list(css_path)
        for entry in entries:
```

In `Tests/UI/test_consolidated_css_harness.py`, Old:
```python
    "screen_feature_watchlists.tcss": (
        "WatchlistsCollectionsScreen",
        "DestinationHarness",
    ),
}
```
New:
```python
    "screen_feature_watchlists.tcss": (
        "WatchlistsCollectionsScreen",
        "DestinationHarness",
    ),
    # Roleplay frame B1: NO owner exempts a harness from this sheet.
    # PersonasScreen does not load it (TldwCli._SCREEN_OWNED_ROUTE_CSS does,
    # on navigation), so naming it would exempt exactly the bundle-only
    # harnesses that push it in on_mount. A styled Roleplay harness pins
    # APP_STYLESHEETS (Tests/UI/roleplay_frame_harness.py), which the
    # `sheet in pin` check already accepts.
    "screen_feature_roleplay.tcss": (),
}
```

- [ ] **Step 2: Run to see them fail**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_stylesheet.py Tests/UI/test_roleplay_frame_harness.py -q -p no:cacheprovider --timeout=180 2>&1 | tail -3`
Expected: failures and errors naming the missing sheet (`FileNotFoundError` on `screen_feature_roleplay.tcss`, `AssertionError: no screen-owned split writes screen_feature_roleplay.tcss`, `KeyError: 'personas'` on the route map, `assert ROLEPLAY_SHEET in APP_STYLESHEETS`).

- [ ] **Step 3: Write the source sheet**

Create `tldw_chatbook/css/features/_roleplay.tcss`:

```css
/* ========================================
 * FEATURES: Roleplay (Ctrl+4) on the Library frame
 * ========================================
 * Spec: Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md.
 * Every rule here is lazy: build_css.py moves it to
 * screen_feature_roleplay.tcss, which TldwCli._SCREEN_OWNED_ROUTE_CSS parses
 * on the first visit to Roleplay (never PersonasScreen.CSS_PATH: Textual
 * loads a screen's CSS_PATH under every app, including the unstyled test
 * harnesses). Every selector carries a Roleplay token (#personas-header...,
 * .roleplay-...). The split also claims the shared tokens
 * workbench-header-title/-subtitle/-status and -active so these rules can
 * leave the boot bundle; Tests/UI/test_roleplay_stylesheet.py refuses a
 * selector anchored only by one of them. $ds-* tokens only, no numbers.
 */

/* One-row inline DestinationHeader (spec 1.3; DESIGN.md "Destination header
   variants"), after #console-workbench-header.console-header-inline. The
   id+class rules (1,2,0) beat the compact-height and density rules that hide
   a subtitle at 24 rows or fewer (.shell-header-compact
   .workbench-header-subtitle is 0,2,0), so the kind stays visible. The cells
   these rules spend on padding and margins are mirrored by the HEADER_*_CELLS
   constants in UI/Persona_Modules/roleplay_frame_state.py. */
#personas-header.personas-header-inline {
    layout: horizontal;
    height: $ds-size-1;
    min-height: $ds-size-1;
    padding: $ds-space-0 $ds-space-1;
    margin: $ds-space-0;
    border: none;
}

#personas-header.personas-header-inline .workbench-header-title {
    width: auto;
    height: $ds-size-1;
    min-height: $ds-size-1;
}

/* The kind: auto width and no wrap, so it is never cut; the item label
   beside it is the part that gives way. */
#personas-header.personas-header-inline .workbench-header-subtitle {
    display: block;
    width: auto;
    height: $ds-size-1;
    min-height: $ds-size-1;
    margin: $ds-space-0 $ds-space-0 $ds-space-0 $ds-space-2;
    text-wrap: nowrap;
}

/* The authority chip (Local / Server: <label> · read-only) is a word, not a
   badge: neutral by id, since WorkbenchStatus has no neutral value. */
#personas-header.personas-header-inline .workbench-header-status {
    width: auto;
    height: $ds-size-1;
    min-height: $ds-size-1;
    margin: $ds-space-0 $ds-space-0 $ds-space-0 $ds-space-1;
    background: $ds-surface-panel;
    color: $ds-text-muted;
    text-style: none;
}

#personas-header-tail {
    layout: horizontal;
    width: $ds-width-fill;
    min-width: $ds-size-0;
    height: $ds-size-1;
}

/* Interim (spec 5.3): "› <item>" until the work-pane title row (B5b-2). */
#personas-header-item {
    width: $ds-width-fill;
    min-width: $ds-size-0;
    height: $ds-size-1;
    margin: $ds-space-0 $ds-space-0 $ds-space-0 $ds-space-1;
    color: $ds-text-muted;
}

#personas-header-unsaved,
#personas-header-blocked {
    width: auto;
    height: $ds-size-1;
    padding: $ds-space-0 $ds-space-1;
    margin: $ds-space-0 $ds-space-0 $ds-space-0 $ds-space-1;
    text-style: bold;
}

/* Words that must be READ: the readable status foregrounds, not the
   decorative $ds-status-unsaved/$ds-status-blocked hues, which fall below
   AA on the panel in many themes. */
#personas-header-unsaved {
    color: $ds-status-warning-readable;
}

#personas-header-blocked {
    color: $ds-status-error-readable;
}

#personas-header-blocked:hover {
    text-style: bold underline;
}

/* Adaptive pane shell (spec 2.12 item 2): the Library's shell and grip rules
   (features/_library.tcss) keyed to Roleplay's destination classes
   (ROLEPLAY_PANE_CLASS_NAMES in UI/Persona_Modules/roleplay_frame_state.py).
   Tests/UI/test_roleplay_stylesheet.py pins the two sets equal once the class
   names are normalised: change both or neither. No width rule: grip widths
   are inline styles from the resolver. Inert until frame slice B7 mounts the
   shell. */
.roleplay-shell {
    width: $ds-width-fill;
    min-width: $ds-size-0;
    height: $ds-height-full;
    min-height: $ds-size-0;
}

.roleplay-shell > .roleplay-shell-work {
    width: $ds-width-fill;
    min-width: $ds-size-0;
}

.roleplay-shell > .roleplay-shell-grip {
    height: $ds-height-full;
    min-height: $ds-size-0;
    margin: $ds-space-0;
    padding: $ds-space-0;
    border: none !important;
    content-align: center middle;
    background: $ds-surface-raised;
    color: $ds-text-muted;
    text-style: none;
    tint: transparent;
    outline: none;
}

.roleplay-shell > .roleplay-shell-grip:hover,
.roleplay-shell > .roleplay-shell-grip.-active {
    background: $ds-surface-raised;
    tint: transparent;
}

.roleplay-shell > .roleplay-shell-grip:hover {
    color: $ds-text-primary;
}

/* task-32053: focus inverts the whole grip (reverse), never an outline. */
.roleplay-shell > .roleplay-shell-grip:focus {
    background: $ds-surface-raised;
    color: $ds-action-focus;
    text-style: bold reverse;
    tint: transparent;
    outline: none;
}
```

- [ ] **Step 4: Register the module, the split and the route; delete the dead items; build**

In `tldw_chatbook/css/build_css.py`, Old:
```python
    "features/_workflows.tcss",
    "features/config_search.tcss",
```
New:
```python
    "features/_workflows.tcss",
    # Roleplay frame B1: Roleplay's own vocabulary, split whole into
    # screen_feature_roleplay.tcss (see SCREEN_OWNED_SPLITS).
    "features/_roleplay.tcss",
    "features/config_search.tcss",
```
Old:
```python
        prefixes={"workflows": ("workflow", "workflows")},
        pinned=frozenset(),
    ),
```
New:
```python
        prefixes={"workflows": ("workflow", "workflows")},
        pinned=frozenset(),
    ),
    # Roleplay frame B1 (spec R18, G18): Roleplay's rules load on the first
    # visit to Roleplay. Narrow prefixes, never bare `personas`. Exact-token
    # compose-site audit (2026-10-03, repo-relative paths, control-checked):
    # `personas-header*`, `personas-library*` and `personas-work*` are composed
    # only in UI/Screens/personas_screen.py and Widgets/Persona_Widgets/; the
    # `roleplay-*` and `personas-more*`/`personas-try*` prefixes have no compose
    # site yet. `personas-library-rows` stays pinned in boot (its `:focus` rule,
    # Tests/UI/test_personas_library_rail_focus_outline.py). The last four
    # entries claim SHARED tokens exactly (DestinationHeader's fixed child
    # classes and Textual's `-active`), so Roleplay's copies of the inline
    # header and grip rules leave boot; legal only because every Roleplay
    # selector also carries a Roleplay token, which
    # Tests/UI/test_roleplay_stylesheet.py enforces (a bare `.-active` rule
    # here would restyle every app button after the first Ctrl+4).
    ScreenOwnedSplit(
        modules=("features/_roleplay.tcss",),
        sheets={"roleplay": "screen_feature_roleplay.tcss"},
        prefixes={
            "roleplay": (
                "roleplay-shell",
                "roleplay-rail",
                "roleplay-nav",
                "roleplay-items",
                "personas-header",
                "personas-library",
                "personas-work",
                "personas-more",
                "personas-try",
                "workbench-header-title",
                "workbench-header-subtitle",
                "workbench-header-status",
                "-active",
            )
        },
        pinned=frozenset({"personas-library-rows"}),
    ),
```

Run the compose-site audit the comment cites (repo-relative paths, with a control prefix that must match), and keep its output in `$EV/prefix-audit.txt`:

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
for P in personas-header personas-library personas-work personas-more personas-try roleplay-shell roleplay-rail roleplay-nav roleplay-items; do
  echo "== $P: $(grep -rlE "[\"'#. ]$P(-[A-Za-z0-9_-]+)?[\"' ]" --include='*.py' --exclude=build_css.py --exclude=roleplay_frame_state.py tldw_chatbook | sort | tr '\n' ' ')"
done | tee /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/prefix-audit.txt
echo "== control (a prefix the Library composes): $(grep -rlE "[\"'#. ]library-adaptive-reader(-[A-Za-z0-9_-]+)?[\"' ]" --include='*.py' tldw_chatbook | wc -l | tr -d ' ') files" | tee -a /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/prefix-audit.txt
EOF
```

Expected: `personas-header`, `personas-library` and `personas-work` list only `tldw_chatbook/UI/Screens/personas_screen.py` and files under `tldw_chatbook/Widgets/Persona_Widgets/`; the other six list nothing; the control prints `5 files` (the filter finds real compose sites, so an empty row is evidence, not a broken grep). `build_css.py` (the split's own quoted prefix tuple) and `roleplay_frame_state.py` (`ROLEPLAY_PANE_CLASS_NAMES`, constants B7 composes) are excluded because neither composes a widget; the control line keeps no exclusion. Any other path: stop and report (a moved rule would leave a widget elsewhere unstyled before its first Roleplay visit).

Delete the never-composed `_agentic_terminal.tcss` items (each is a comma member whose group survives), then register the route and build:

```bash
bash <<'EOF'
set -euo pipefail
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
from pathlib import Path
path = Path("tldw_chatbook/css/components/_agentic_terminal.tcss")
text = path.read_text(encoding="utf-8")
for item, count in (
    ("#personas-title,\n", 1),
    ("#personas-list-pane,\n", 3),
    ("#personas-detail-pane,\n", 3),
    ("#personas-list-detail-divider,\n", 1),
    ("#personas-detail-inspector-divider,\n", 1),
):
    assert text.count(item) == count, (item, text.count(item))
    text = text.replace(item, "")
path.write_text(text, encoding="utf-8")
print("nine dead items removed")
PY
grep -rn "personas-title\b\|personas-list-pane\|personas-detail-pane\|personas-list-detail-divider\|personas-detail-inspector-divider" --include='*.py' tldw_chatbook Tests | head -3 || echo "no Python references (the ids are dead)"
EOF
```

Expected: `nine dead items removed` and `no Python references (the ids are dead)` (the ids have no producer and no test; under `pipefail` a grep that finds nothing exits 1, which used to end this block with exit 1 although it had succeeded).

In `tldw_chatbook/app.py`, Old:
```python
    TAB_LIBRARY,
    TAB_SCHEDULES,
```
New:
```python
    TAB_LIBRARY,
    TAB_PERSONAS,
    TAB_SCHEDULES,
```
Old:
```python
        TAB_WORKFLOWS: ("screen_feature_workflows.tcss",),
    }
```
New:
```python
        TAB_WORKFLOWS: ("screen_feature_workflows.tcss",),
        TAB_PERSONAS: ("screen_feature_roleplay.tcss",),
    }
```

(`TAB_PERSONAS = "personas"` is the canonical tab of the `personas`, `ccp`, `characters` and `roleplay` routes, so every alias loads the sheet. `app.py` is 5,710 lines against its 5,712 ratchet row on `8c4dfe59a2`, so these two lines land exactly on the row; if dev grows `app.py` first, Task 6 Step 9's `ratchet_rows.py` raises that row to the measurement with a dated owner-decision comment, ruling 2.)

The readable warning foreground the Unsaved chip uses. In `tldw_chatbook/css/core/_variables.tcss`, Old:
```css
$ds-status-error-readable: $text-error;
```
New:
```css
$ds-status-error-readable: $text-error;
/* Roleplay frame B1: the readable warning foreground, the sibling of
   $ds-status-error-readable for warning words that must be READ (the
   Roleplay header's Unsaved chip). $warning, the decorative status hue,
   measured below AA on the panel surface in 29 of the 70 registered themes
   (and $error in 40); Textual's generated $text-warning clears AA on all 70
   (Tests/UI/test_theme_contrast.py; Tests/UI/test_roleplay_stylesheet.py). */
$ds-status-warning-readable: $text-warning;
```

The guard against freezing a readable token to a literal colour names its tokens, so it must name this one too. In `Tests/Architecture/test_builtin_theme_contrast.py`, Old:
```python
        "ds-status-error-readable",
        "ds-text-placeholder",
```
New:
```python
        "ds-status-error-readable",
        # Roleplay frame B1: the readable warning foreground (the Unsaved chip).
        "ds-status-warning-readable",
        "ds-text-placeholder",
```

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py > /dev/null
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py 2>&1 | grep -i roleplay
grep -n "MODULE: features/_roleplay.tcss" -A2 tldw_chatbook/css/tldw_cli_modular.tcss
git status --short tldw_chatbook/css
EOF
```

Expected: `screen_feature_roleplay.tcss reproduces from its Python sources.`; the bundle's Roleplay section is the banner followed by blank lines only; `status` shows the new source and sheet (`??`) and ` M` for `build_css.py`, `_agentic_terminal.tcss`, `core/_variables.tcss`, `tldw_cli_modular.tcss` and the seven existing `screen_agentic_*`/`screen_feature_*`/`screen_modal_*` split sheets (each gained the token line in its variable preamble), and nothing else.

- [ ] **Step 5: Run to see them pass, then the CSS-integrity suites on both arms**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_stylesheet.py Tests/UI/test_roleplay_frame_harness.py -q -p no:cacheprovider --timeout=180 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1`
Expected: `23 passed` (11 static + 12 harness), then `Tests/Architecture/test_builtin_theme_contrast.py -k freeze_readable` → `1 passed` (the new token is a `$`-reference).

```bash
bash <<'EOF'
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence
"$EV/paired.sh" t5-css "$EV/suites-css.txt" -n 6 | tee -a "$EV/gates.txt"
EOF
```

Expected: no new failures on head; the paired failure-set diff is the verdict, and the names below are indicative only (dev moves: #2993 alone changed `tldw_api/client.py`, whose ratchet row is one of them). Pre-existing on both arms in the dry run: `test_css_staleness_manifest.py::TestBuilderIntegration::test_main_end_to_end_manifest_and_staleness` (a non-ASCII temp path), `test_component_pattern_governance.py::test_dimension_literal_ratchet` and `::test_python_style_ratchet` (red on dev for other sheets; B1's own floors are `test_the_roleplay_source_has_no_numeric_dimension_literal` and Task 4's `test_b1_production_modules_carry_no_python_style_violation`), and, on `bd41347b65`, `test_widget_css_consolidation.py::test_class_level_css_stays_within_the_allowlist` plus nine `test_module_size_ratchet.py` rows for other modules (`app_lifecycle.py`, `app_service_wiring.py`, `Chat/console_chat_controller.py`, `Chat/console_chat_store.py`, `tldw_api/client.py`, `UI/MCP_Modules/mcp_workbench.py`, `UI/Screens/llm_screen.py`, `UI/Screens/watchlists_collections_screen.py`, `Widgets/Console/console_transcript.py`): 13 on each arm, the same 13. Re-run on both arms at `8c4dfe59a2` (the module-size ratchet alone): the same nine rows fail on both (`9 failed, 39 passed` on base, `9 failed, 41 passed` on head: B1's new `roleplay_frame_state.py` row adds two passing checks). On `a793acbef5` a tenth row is red on dev (`Chat/console_interrupt_rounds.py`): `10 failed, 42 passed` on base, `10 failed, 44 passed` on head, the same ten rows; the same again on `8d502ba250`.

Boot CSS, measured the way the gate measures it (private profile):

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b1-evidence
for ARM in base head; do
  if [ "$ARM" = base ]; then T=$MAIN/.worktrees/roleplay-b1-base; else T=$MAIN/.worktrees/roleplay-b1; fi
  P="$EV/profile-boot-$ARM"; rm -rf "$P"; mkdir -p "$P/home" "$P/config" "$P/data"
  (cd "$T" && HOME="$P/home" XDG_CONFIG_HOME="$P/config" XDG_DATA_HOME="$P/data" TLDW_CONFIG_PATH="$P/config/config.toml" PYTHONPATH="$T" "$PY" -c 'from Tests.Performance.test_boot_css_byte_budget import _boot_parsed_css_census as c; print(sum(c().values()))' 2>/dev/null | tail -1 | sed "s/^/$ARM boot-css=/")
done | tee -a "$EV/gates.txt"
EOF
```

Expected: `head` = `base` − 119 (608,077 → 607,958 in the dry run on `8d502ba250`, 607,943 → 607,824 on `a793acbef5`, 607,326 → 607,207 on `8c4dfe59a2`; measured while planning: the token line adds exactly 45 B to the bundle).

- [ ] **Step 6: Named mutations**

Each is applied with the Edit tool, rebuilt with `build_css.py` when it touches a `.tcss` file, run, then restored and rebuilt:
1. `bare-active-rule`: append `.-active { tint: transparent; }` to `_roleplay.tcss`. `…/python -m pytest Tests/UI/test_roleplay_stylesheet.py -q -p no:cacheprovider -k carries_a_roleplay_token`: `1 failed` (`'.-active' has no Roleplay token`).
2. `bare-type-subject`: append `#personas-header-tail Static { color: $ds-text-muted; }`. `-k bare_type`: `1 failed`.
3. `unrouted-sheet`: delete the `TAB_PERSONAS: ("screen_feature_roleplay.tcss",),` row. `…/python -m pytest Tests/UI/test_roleplay_frame_harness.py -q -p no:cacheprovider -k first_visit`: `2 failed` (`has_source` False after both routes); `Tests/UI/test_css_build_integrity.py -k wired_to_app_routes`: `1 failed`.
4. `css-path-as-str`: add `CSS_PATH = str(Path(__file__).resolve().parents[2] / "css" / "screen_feature_roleplay.tcss")` as a class attribute of `PersonasScreen`. `…/python -m pytest Tests/UI/test_css_build_integrity.py -q -p no:cacheprovider -k "take_owned_sheets"`: `1 failed` (before the str fix in Step 1 this passed vacuously).
5. `decorative-chip-hue`: in `_roleplay.tcss` set the blocked chip back to `color: $ds-status-blocked;`. `…/python -m pytest Tests/UI/test_roleplay_stylesheet.py -q -p no:cacheprovider -k clear_aa`: `1 failed` (`('#personas-header-blocked', 'ds-status-blocked', 'apricot', 3.87)`; run while planning).

Record all five in `$EV/mutations.md`.

- [ ] **Step 7: Lint and commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/css/build_css.py Tests/UI/test_roleplay_stylesheet.py Tests/UI/test_roleplay_frame_harness.py Tests/UI/test_css_build_integrity.py Tests/UI/test_consolidated_css_harness.py Tests/Architecture/test_builtin_theme_contrast.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check Tests/UI/test_roleplay_stylesheet.py Tests/UI/test_roleplay_frame_harness.py Tests/Architecture/test_builtin_theme_contrast.py
for T in /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1-base /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1; do
  (cd "$T" && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check --output-format concise tldw_chatbook/app.py | grep '^Found')
done
git add tldw_chatbook/css/features/_roleplay.tcss tldw_chatbook/css/build_css.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/core/_variables.tcss tldw_chatbook/css/tldw_cli_modular.tcss tldw_chatbook/css/screen_*.tcss tldw_chatbook/app.py Tests/UI/test_roleplay_stylesheet.py Tests/UI/test_roleplay_frame_harness.py Tests/UI/test_css_build_integrity.py Tests/UI/test_consolidated_css_harness.py Tests/Architecture/test_builtin_theme_contrast.py
git commit -m "feat(css): Roleplay's lazy stylesheet, loaded on the first visit (B1)" -m "features/_roleplay.tcss splits whole into screen_feature_roleplay.tcss (R18 prefixes, personas-library-rows pinned, four exact shared claims under an anchor guard) and loads through TldwCli._SCREEN_OWNED_ROUTE_CSS, never PersonasScreen.CSS_PATH. It carries the inline-header rules, readable chip colours (new token \$ds-status-warning-readable, the sibling of \$ds-status-error-readable: AA on every theme) and a copy of the Library's adaptive-shell rules pinned equal by role. Nine never-composed #personas-* items leave _agentic_terminal.tcss: boot CSS -119 B." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: `All checks passed!`, `3 files already formatted`, then `Found 96 errors.` twice (`app.py`'s pre-existing findings, the same count on the base arm and on head: the two-line route-map edit adds none), and the commit line.

---

### Task 6: The one-row header in `personas_screen.py` (spec §1.3, R12, R24, R33)

PS keeps only glue: gather the inputs (`_gather_header_inputs`), compute the view in the pure module, paint it (`_paint_header`). The readiness poll and `on_resize` repaint from it; a persona save repaints after its in-flight flag clears. The blocked chip is a `FittedText`, so it is not a Tab stop: its click deep-links to Settings, and keyboard reach arrives with B3's Tab region (recorded as an interim in Task 13; today the header has no control at all, and the Inspector's readiness line keeps its own Settings link). The header never takes `status-blocked` any more, so task-523's red-badge rule is dead and leaves boot, and the token `personas-header` leaves the bundle with it.

What stays: `#personas-purpose`, `#personas-mode-strip`, the footer-hint builder, `_provider_send_block_reason` (the Inspector's gated readiness still uses it).

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (17 edits below), `tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py` (`open_provider_settings`), `tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py` (Step 4b), `Tests/UI/test_roleplay_frame_state.py` (the guard pin), `Tests/UI/test_unified_shell_phase6_first_time_replay.py` (a re-pin), `tldw_chatbook/css/components/_workbench.tcss` (the task-523 rule), regenerated `tldw_chatbook/css/tldw_cli_modular.tcss` and `tldw_chatbook/css/widget_defaults_scoped.tcss`, `Tests/UI/test_roleplay_stylesheet.py` (one tightened line), `Tests/UI/test_personas_workbench.py`, `Tests/UI/test_personas_dictionaries.py`, `Tests/UI/test_personas_subscription_readiness.py`, `Tests/UI/test_roleplay_hostile_names.py`, `Tests/UI/test_roleplay_hostile_text_surfaces.py` (re-pins), `Tests/Architecture/test_module_size_ratchet.py` (B1's rows, set to the measurement), and, through the raise script, `Tests/Performance/test_screen_preimport_payload_budget.py`, `backlog/decisions/097-boot-budget-ratchets.md`, `Tests/Performance/boot_budget_snapshots/preimport_payload.json`
- Test: `Tests/UI/test_roleplay_header.py` (create)

**Interfaces:**
- Consumes: Task 3's `FittedText`; Task 4's `frame_state.*`; Task 5's selectors; Task 1's harness (`open_styled_roleplay`, `size_matrix`, `styled_tiers`, `chrome_bottoms`, `first_list_item`, `painted_rows`, `painted_text`, `seed_mock_characters`, `settle`, `wait_until`); existing `PersonasScreen._aggregate_roleplay_draft_snapshot()`, `PersonasPreviewController.console_handoff_readiness() -> tuple[bool, str | None]`, `_selection_from_defaults(config, defaults_key)`; test helpers `_conversation_record`, `_install_conversation_db` (`Tests/UI/test_personas_workbench.py`).
- Produces: DOM ids `#personas-header` (class `personas-header-inline`), `#personas-header-tail`, `#personas-header-item`, `#personas-header-unsaved`, `#personas-header-blocked` (all `FittedText` but the tail); `PersonasScreen._header_inputs: RoleplayHeaderInputs | None`, `_gather_header_inputs() -> RoleplayHeaderInputs`, `_paint_header() -> None`, `_update_title() -> None` (unchanged name and call sites); `PersonasPreviewController.open_provider_settings(*, defaults_key: str | None = None)`.

- [ ] **Step 1: Write the failing header tests**

Create `Tests/UI/test_roleplay_header.py`:

```python
"""Mounted contracts of Roleplay's one-row header (frame slice B1, spec 1.3).

Geometry runs under BOTH styled tiers (``Tests/UI/roleplay_frame_harness.py``)
at every B1 size, measured relative to the nav bar and the header (spec 5.7.2
item 1), never as absolute rows. Behaviour checks that need no geometry run on
the styled mock tier only; the leave-guard check needs the real app's
navigation and runs on the full tier.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from rich.cells import cell_len
from textual.color import Color
from textual.widgets import Static

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.UI.roleplay_frame_harness import (
    assert_painted_inside,
    chrome_bottoms,
    first_list_item,
    open_styled_roleplay,
    painted_rows,
    painted_text,
    seed_mock_characters,
    settle,
    size_matrix,
    styled_tiers,
    wait_until,
)
from Tests.UI.test_personas_workbench import (
    _conversation_record,
    _install_conversation_db,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Persona_Modules import roleplay_frame_state as fs
from tldw_chatbook.UI.Screens.personas_screen import (
    PERSONAS_COMPACT_WORKBENCH_MAX_WIDTH,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import DestinationHeader, FittedText
from tldw_chatbook.Widgets.glyph_fallback import ASCII_GLYPH_FALLBACKS
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

#: Twelve characters with a dated meta line, so every row is two lines tall
#: like a real library (the date is the meta when a card has no description).
CHARACTERS = [
    {
        "id": index,
        "name": f"Character {index:02d}",
        "version": 1,
        "last_modified": f"2026-09-{index:02d}T12:00:00",
    }
    for index in range(1, 13)
]

#: 213 cells: longer than the item label is wide at every B1 size (220x55 included).
LONG_NAME = (
    "Ser Bartholomew Ignatius Fairweather-Montgomery of the Seventeen Silver "
    "Towers, Keeper of the Long Keys, Warden of the Quiet Marches and Sworn "
    "Shield of the Ninefold Orchard Gates beyond the Amber Hills of Lowmere"
)


@pytest.fixture
def roleplay_data(monkeypatch):
    """Twelve characters and one conversation through the screen's seams."""
    seed_mock_characters(monkeypatch, CHARACTERS)
    monkeypatch.setattr(
        character_handler_module, "_default_character_db", lambda: object()
    )
    _install_conversation_db(monkeypatch, [_conversation_record(1)])
    return CHARACTERS


def _parts(screen):
    header = screen.query_one("#personas-header")
    return {
        "header": header,
        "title": header.query_one("#workbench-header-title", Static),
        "subtitle": header.query_one("#workbench-header-subtitle", Static),
        "item": header.query_one("#personas-header-item", FittedText),
        "unsaved": header.query_one("#personas-header-unsaved", FittedText),
        "blocked": header.query_one("#personas-header-blocked", FittedText),
        "status": header.query_one("#workbench-header-status", Static),
    }


@styled_tiers
@size_matrix
async def test_header_is_one_row_and_the_kind_stays_visible(
    styled_tier, roleplay_size, mock_app_instance, roleplay_data
):
    """AC#1: one row at every size; the kind shows even at 24 rows or fewer."""
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size
    ) as pilot:
        screen = pilot.app.screen
        parts = _parts(screen)
        nav_bottom, _header_bottom = chrome_bottoms(screen)
        assert parts["header"].region.y == nav_bottom
        assert parts["header"].region.height == 1
        assert parts["subtitle"].display
        assert painted_text(screen, parts["subtitle"].region) == "Characters"
        if roleplay_size[1] <= 24:
            # The compact-height path that used to hide the subtitle is live.
            assert screen.has_class("shell-header-compact")


@styled_tiers
@size_matrix
async def test_first_list_item_sits_right_under_the_unchanged_band(
    styled_tier, roleplay_size, mock_app_instance, roleplay_data
):
    """AC#2: B1 moves only the header. The purpose line, the mode strip and
    the workbench chrome keep their rows, so the first item sits a fixed
    number of rows under the header: with today's 3-row nav that is row 19
    at the design centre (y 18) and row 18 at 80x24, whose compact workbench
    has one row less chrome."""
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size
    ) as pilot:
        screen = pilot.app.screen
        nav_bottom, header_bottom = chrome_bottoms(screen)
        assert header_bottom == nav_bottom + 1
        band = 13 if roleplay_size[0] <= PERSONAS_COMPACT_WORKBENCH_MAX_WIDTH else 14
        assert first_list_item(screen).region.y == header_bottom + band


@styled_tiers
async def test_seven_characters_show_at_120x36(
    styled_tier, mock_app_instance, roleplay_data
):
    """AC#2: at least 7 two-line rows show their name line at 120x36 (5 before)."""
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=(120, 36)
    ) as pilot:
        screen = pilot.app.screen
        window = screen.query_one("#personas-library-rows").scrollable_content_region
        items = list(screen.query("#personas-library-rows > ListItem"))
        assert items and all(item.region.height == 2 for item in items[:3])
        shown = [
            item for item in items if window.contains(item.region.x, item.region.y)
        ]
        assert len(shown) >= 7, [item.region for item in items]


@styled_tiers
@pytest.mark.parametrize(
    ("roleplay_size", "work_width", "inspector_width"),
    [((120, 36), 58, 30), ((160, 45), 78, 39), ((220, 55), 108, 54)],
    ids=["120x36", "160x45", "220x55"],
)
async def test_the_work_pane_and_inspector_keep_their_widths(
    styled_tier,
    roleplay_size,
    work_width,
    inspector_width,
    mock_app_instance,
    roleplay_data,
):
    """Spec 5.3's interim geometry: B1 moves only the header, so the work pane
    and the Inspector keep the base arm's widths (spec 5.3's "today" row,
    measured on both styled tiers on the base arm)."""
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size
    ) as pilot:
        screen = pilot.app.screen
        work = screen.query_one("#personas-work-area")
        inspector = screen.query_one("#personas-inspector-pane")
        assert (work.region.width, inspector.region.width) == (
            work_width,
            inspector_width,
        )


@styled_tiers
@size_matrix
async def test_every_header_part_fits_in_the_worst_case(
    styled_tier, roleplay_size, mock_app_instance, roleplay_data, monkeypatch
):
    """No part is clipped with every chip, a server label, a long name and the
    longest kind; the kind is painted whole (AC#4)."""
    inputs = fs.RoleplayHeaderInputs(
        mode="dictionaries",
        edit_mode="edit",
        item_name=LONG_NAME,
        unsaved=True,
        provider_blocked=True,
        runtime_source="server",
        server_label="home-tldw.example.internal",
    )
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size
    ) as pilot:
        screen = pilot.app.screen
        monkeypatch.setattr(screen, "_gather_header_inputs", lambda: inputs)
        screen._update_title()
        await settle(pilot)
        parts = _parts(screen)
        header = parts["header"]
        view = fs.build_header_view(inputs, header.region.width)
        expected = {
            "title": "Roleplay",
            "subtitle": "Dictionaries",
            "unsaved": view.unsaved_chip,
            "blocked": view.blocked_chip,
            "status": view.status_plain,
        }
        for name, text in expected.items():
            part = parts[name]
            assert part.display and part.region.width > 0, name
            # Inside the header's painted window and covered by no sibling.
            assert_painted_inside(part, header)
            assert painted_text(screen, part.region).strip() == text, name
        item = parts["item"]
        assert_painted_inside(item, header)
        painted = painted_text(screen, item.region).rstrip()
        assert painted == item.fitted_text
        assert cell_len(painted) <= item.region.width
        if painted.startswith("›"):
            assert painted.endswith("… · editing")
        else:
            assert painted in ("· editing", "")


@styled_tiers
async def test_header_chrome_cells_match_the_lazy_sheet(
    styled_tier, mock_app_instance, roleplay_data, monkeypatch
):
    """The fit's HEADER_*_CELLS constants are the sheet's real spacing, under
    both the harness CSS_PATH and the app's route loader."""
    inputs = fs.RoleplayHeaderInputs(
        mode="characters", unsaved=True, provider_blocked=True
    )
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=(160, 45)
    ) as pilot:
        screen = pilot.app.screen
        monkeypatch.setattr(screen, "_gather_header_inputs", lambda: inputs)
        screen._update_title()
        await settle(pilot)
        parts = _parts(screen)

        def chrome(widget, *, margin=True):
            styles = widget.styles
            left = styles.margin.left if margin else 0
            return left + styles.padding.left + styles.padding.right

        assert chrome(parts["header"], margin=False) == fs.HEADER_PADDING_CELLS
        assert parts["subtitle"].styles.margin.left == fs.KIND_GAP_CELLS
        assert parts["item"].styles.margin.left == fs.ITEM_GAP_CELLS
        assert chrome(parts["unsaved"]) == fs.CHIP_CHROME_CELLS
        assert chrome(parts["blocked"]) == fs.CHIP_CHROME_CELLS
        assert chrome(parts["status"]) == fs.STATUS_CHROME_CELLS


@pytest.mark.parametrize(
    "target", [(90, 45), (80, 24), (220, 55)], ids=lambda s: f"{s[0]}x{s[1]}"
)
async def test_a_resize_refits_the_header_without_gathering_inputs(
    target, mock_app_instance, roleplay_data, monkeypatch
):
    """Spec 2.13 ("resize does no data work"): a width-only change refits the
    chips and the status from the cached inputs (the short unsaved chip below
    100 columns, the degrade order below that) and never re-reads readiness
    or the draft aggregate."""
    inputs = fs.RoleplayHeaderInputs(
        mode="characters",
        unsaved=True,
        provider_blocked=True,
        runtime_source="server",
        server_label="home-tldw.example.internal",
    )
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        # The 0.25 s readiness poll re-gathers on its own: stop it, so only
        # the resize can repaint.
        screen._console_readiness_poll_timer.stop()
        screen._header_inputs = inputs
        screen._paint_header()
        await settle(pilot)
        parts = _parts(screen)
        gathered = []
        monkeypatch.setattr(
            screen, "_gather_header_inputs", lambda: gathered.append(1) or inputs
        )
        await pilot.resize_terminal(*target)
        await settle(pilot)
        view = fs.build_header_view(inputs, parts["header"].region.width)
        assert parts["unsaved"].value == view.unsaved_chip
        assert parts["blocked"].value == view.blocked_chip
        assert str(parts["status"].renderable) == view.status_plain
        assert gathered == []


async def test_gathering_header_inputs_walks_no_dom(
    mock_app_instance, roleplay_data, monkeypatch
):
    """The 0.25 s readiness poll gathers the header's inputs on every tick, so
    the gather must not query the DOM: a ``query_one`` for the demand-mounted
    character editor walks the whole screen and raises ``NoMatches`` while
    browsing (about 4 ms a tick, measured in review). The aggregate reads the
    cached editor instead."""
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        queried = []
        original = screen.query_one

        def counting(*args, **kwargs):
            queried.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(screen, "query_one", counting)
        screen._gather_header_inputs()
        assert queried == []


@styled_tiers
@size_matrix
async def test_a_long_name_ellipsises_and_the_kind_is_never_cut(
    styled_tier, roleplay_size, mock_app_instance, monkeypatch
):
    """AC#4, on a real first-paint auto-selection of a long-named character."""
    # Alone in the library, so the first-paint auto-selection (F-031) picks it.
    seed_mock_characters(monkeypatch, [{"id": 1, "name": LONG_NAME, "version": 1}])
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size
    ) as pilot:
        screen = pilot.app.screen
        parts = _parts(screen)
        await wait_until(
            pilot,
            lambda: parts["item"].value == (LONG_NAME, False),
            what="the long name in the header",
        )
        await settle(pilot)
        assert painted_text(screen, parts["subtitle"].region) == "Characters"
        painted = painted_text(screen, parts["item"].region).rstrip()
        assert painted.startswith("› Ser ") and painted.endswith("…"), painted
        assert painted == parts["item"].fitted_text


async def test_status_names_the_data_source_and_never_says_ready(
    mock_app_instance, roleplay_data
):
    """RP-067 and DESIGN.md:115: "Local", or "Server: <label> · read-only"."""
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        status = _parts(screen)["status"]
        assert painted_text(screen, status.region).strip() == "Local"
        pilot.app.runtime_policy = SimpleNamespace(
            state=SimpleNamespace(
                last_known_server_label="home-tldw", active_server_id=None
            )
        )
        screen._set_persona_editor_runtime_source("server")
        screen._update_title()
        await settle(pilot)
        assert (
            painted_text(screen, status.region).strip()
            == "Server: home-tldw · read-only"
        )
        assert "Ready" not in painted_rows(screen)[status.region.y]


async def test_the_header_composes_without_a_ready_chip(
    mock_app_instance, roleplay_data, monkeypatch
):
    """The composed state already names the kind and the data source, so the
    shared header's default "Ready" chip never paints, whatever the order of
    ``DestinationHeader.on_mount`` and the screen's first ``_update_title``."""
    composed = []
    original = DestinationHeader.__init__

    def recording_init(self, state, *args, **kwargs):
        if kwargs.get("id") == "personas-header":
            composed.append(state)
        original(self, state, *args, **kwargs)

    monkeypatch.setattr(DestinationHeader, "__init__", recording_init)
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)):
        pass
    assert [(state.subtitle, state.status_label) for state in composed] == [
        ("Characters", "Local")
    ]


async def test_blocked_chip_follows_destination_readiness_through_the_poll(
    mock_app_instance, roleplay_data
):
    """Shown only while no chat provider is ready for character chats; the
    0.25 s readiness poll clears it with no other refresh."""
    mock_app_instance.app_config = {
        "chat_defaults": {"provider": "openai", "model": "gpt-4o"}
    }
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        blocked = _parts(screen)["blocked"]
        assert blocked.display
        assert painted_text(screen, blocked.region).strip() == (
            "No chat provider · Settings ›"
        )
        mock_app_instance.app_config["api_settings"] = {
            "openai": {"api_key": "unit-test-placeholder-key"}
        }
        await wait_until(pilot, lambda: not blocked.display, what="the chip to go")


async def test_chip_words_take_the_readable_hues_and_the_status_is_neutral(
    mock_app_instance, roleplay_data, monkeypatch
):
    """The colours come from the lazy sheet: the readable status foregrounds
    ($text-warning, $text-error: AA on the panel in every theme, unlike the
    decorative $warning and $error) on the chips, and a neutral status word
    on the panel (no badge)."""
    inputs = fs.RoleplayHeaderInputs(
        mode="characters", unsaved=True, provider_blocked=True
    )
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        monkeypatch.setattr(screen, "_gather_header_inputs", lambda: inputs)
        screen._update_title()
        await settle(pilot)
        parts = _parts(screen)
        variables = pilot.app.get_css_variables()
        assert parts["unsaved"].display and parts["blocked"].display
        unsaved, blocked = parts["unsaved"].styles.color, parts["blocked"].styles.color
        assert unsaved.rgb == Color.parse(variables["text-warning"]).rgb
        assert blocked.rgb == Color.parse(variables["text-error"]).rgb
        assert parts["status"].styles.color not in (unsaved, blocked)
        assert parts["status"].styles.background == parts["header"].styles.background


async def test_clicking_the_blocked_chip_opens_settings_for_the_chat_provider(
    mock_app_instance, roleplay_data, monkeypatch
):
    """The chip deep-links to the provider ``console_handoff_readiness`` checks
    (chat_defaults), through app navigation and so through the leave guard."""
    mock_app_instance.app_config = {
        "character_defaults": {"provider": "anthropic", "model": "claude-3-haiku"},
        "chat_defaults": {"provider": "openai", "model": "gpt-4o"},
    }
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        posted = []
        original = screen.post_message

        def spy(message):
            posted.append(message)
            return original(message)

        monkeypatch.setattr(screen, "post_message", spy)
        assert await pilot.click("#personas-header-blocked")
        await pilot.pause()
        navigations = [m for m in posted if isinstance(m, NavigateToScreen)]
        assert [m.screen_name for m in navigations] == ["settings"]
        assert navigations[0].screen_context.get("provider") == "openai"


async def test_the_blocked_chip_passes_the_leave_guard(
    mock_app_instance, roleplay_data, monkeypatch
):
    """Spec 1.3: the chip "passes the leave guard". On the real app, a click
    with an unsaved draft asks Save / Discard / Stay before leaving, and Stay
    keeps Roleplay and the draft (the mock tier handles no navigation).

    The blocked state is set on the screen's own readiness seam, never taken
    from the process's provider config: in the PR Fast Lane's admission step
    this file runs after test_console_runtime_ownership.py in one pytest
    process, and there the real app read a ready provider and never showed
    the chip (2026-10-08 dry run). A chip that appears only then must be laid
    out before the click, or the click lands elsewhere.
    """
    from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
        RoleplayDraftNavigationDialog,
    )

    async with open_styled_roleplay("full", mock_app_instance, size=(160, 45)) as pilot:
        app = pilot.app
        screen = app.screen
        monkeypatch.setattr(
            screen.preview,
            "console_handoff_readiness",
            lambda: (False, "No chat provider is configured."),
        )
        screen._update_title()
        blocked = _parts(screen)["blocked"]
        await wait_until(
            pilot,
            lambda: blocked.display and blocked.region.width > 0,
            what="the blocked chip, laid out",
        )
        await settle(pilot)
        screen.state.has_unsaved_changes = True
        assert await pilot.click("#personas-header-blocked")
        await wait_until(
            pilot,
            lambda: isinstance(app.screen, RoleplayDraftNavigationDialog),
            what="the Save / Discard / Stay question",
        )
        await pilot.press("escape")  # Stay
        await wait_until(pilot, lambda: app.screen is screen, what="Stay")
        await settle(pilot)
        assert screen.state.has_unsaved_changes is True


@pytest.mark.parametrize(
    "case",
    [
        "attachment",
        "character_visual",
        "persona_visual",
        "persona_shared_visual",
        "visual_operation_inflight",
        "character_save_inflight",
        "persona_save_inflight",
    ],
)
async def test_unsaved_chip_follows_the_aggregate_not_has_unsaved_changes(
    case, mock_app_instance, roleplay_data
):
    """AC#3 (R24): every case here leaves ``has_unsaved_changes`` False, yet
    the aggregate is not clean, so the chip shows; it goes when the domain
    is clean again. The 0.25 s poll is the only refresh used."""
    async with open_styled_roleplay("mock", mock_app_instance, size=(160, 45)) as pilot:
        screen = pilot.app.screen
        chip = _parts(screen)["unsaved"]
        assert not chip.display

        async def dirty():
            if case == "attachment":
                await screen._ensure_center_view("character-editor")
                editor = screen.query_one(PersonasCharacterEditorWidget)
                editor.load_character({"id": 1, "name": "Ada", "image": b"old"})
                editor.set_avatar_image(b"new")
            elif case == "character_visual":
                screen._visual_identity_authoring = object()
            elif case == "persona_visual":
                screen._persona_visual_authoring = SimpleNamespace(dirty=True)
            elif case == "persona_shared_visual":
                screen._persona_shared_visual_identity_authoring = object()
            elif case == "visual_operation_inflight":
                loop = asyncio.get_running_loop()
                screen._visual_identity_operation_task = loop.create_future()
            elif case == "character_save_inflight":
                screen._character_save_inflight = True
            else:
                screen._profile_save_operation_inflight = True

        def clean():
            if case == "attachment":
                screen.query_one(
                    PersonasCharacterEditorWidget
                ).discard_unsaved_attachment()
            elif case == "character_visual":
                screen._visual_identity_authoring = None
            elif case == "persona_visual":
                screen._persona_visual_authoring = None
            elif case == "persona_shared_visual":
                screen._persona_shared_visual_identity_authoring = None
            elif case == "visual_operation_inflight":
                screen._visual_identity_operation_task.cancel()
                screen._visual_identity_operation_task = None
            elif case == "character_save_inflight":
                screen._character_save_inflight = False
            else:
                screen._profile_save_operation_inflight = False

        await dirty()
        assert screen.state.has_unsaved_changes is False
        assert fs.roleplay_has_unsaved_work(screen._aggregate_roleplay_draft_snapshot())
        await wait_until(pilot, lambda: chip.display, what="the unsaved chip")
        assert chip.value == "Unsaved changes"
        clean()
        assert not fs.roleplay_has_unsaved_work(
            screen._aggregate_roleplay_draft_snapshot()
        )
        await wait_until(pilot, lambda: not chip.display, what="the chip to go")


@styled_tiers
@size_matrix
async def test_header_row_paints_ascii_markers_in_ascii_mode(
    styled_tier, roleplay_size, mock_app_instance, monkeypatch
):
    """Spec 4.12: every header glyph goes through the glyph map (no CSS "…")."""
    # Alone in the library, so the first-paint auto-selection (F-031) picks it.
    seed_mock_characters(monkeypatch, [{"id": 1, "name": LONG_NAME, "version": 1}])
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=roleplay_size, ascii_glyphs=True
    ) as pilot:
        screen = pilot.app.screen
        parts = _parts(screen)
        await wait_until(
            pilot,
            lambda: parts["item"].value == (LONG_NAME, False),
            what="the long name in the header",
        )
        await settle(pilot)
        row = painted_rows(screen)[parts["header"].region.y]
        unmapped = set(row) & (set(ASCII_GLYPH_FALLBACKS) | {"…"})
        assert not unmapped, (unmapped, row)
        assert "> Ser " in row and "..." in row, row
```

TASK-33622.14 (Done on dev) moved the aggregate leave and Ctrl+Q guard into `UI/Persona_Modules/roleplay_draft_guard.py`, which decides with a bare `snapshot.is_clean`. R24/G4 want one predicate, so the guard asks `roleplay_has_unsaved_work()` too (Step 4b). Pin it in `Tests/UI/test_roleplay_frame_state.py` (it mounts nothing, so the file stays in the UI Fast Lane). Old:
```python
def test_b1_production_modules_carry_no_python_style_violation():
```
New:
```python
async def test_the_leave_and_quit_guard_decides_on_the_same_predicate(monkeypatch):
    """R24/G4: TASK-33622.14's guard (leaving Roleplay, Ctrl+Q) asks the one
    predicate, never ``is_clean`` itself: with the predicate answering
    "nothing unsaved", a dirty aggregate passes without a prompt. Positive
    control: unpatched, the same aggregate asks, and Stay keeps the screen."""
    from tldw_chatbook.UI.Persona_Modules import roleplay_draft_guard as guard

    dirty = _snapshot(form_dirty=True)
    screen = SimpleNamespace(
        _aggregate_roleplay_draft_snapshot=lambda: dirty,
        state=SimpleNamespace(active_mode="characters"),
    )
    asked = []

    async def ask(prompt):
        asked.append(type(prompt).__name__)
        return None  # Stay

    assert await guard.confirm_roleplay_drafts(screen, ask) is False
    assert asked == ["RoleplayDraftNavigationDialog"]
    asked.clear()
    monkeypatch.setattr(guard, "roleplay_has_unsaved_work", lambda snapshot: False)
    assert await guard.confirm_roleplay_drafts(screen, ask) is True
    assert asked == []


def test_b1_production_modules_carry_no_python_style_violation():
```

- [ ] **Step 2: Run to see it fail**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_header.py Tests/UI/test_roleplay_frame_state.py -q -p no:cacheprovider -n 6 --timeout=180 2>&1 | tail -3`
Expected: `62 failed, 33 passed` (exactly so in the dry run on `8c4dfe59a2`, as on `83c264f286`): the header file's 61 behaviour tests fail on assertions and missing widgets (`NoMatches` on `#personas-header-item`, header height 5 or 4, "Blocked" painted, an empty composed status label, `AttributeError` on `_paint_header` and `_gather_header_inputs`), and so does the guard test (`AttributeError`: the guard has no `roleplay_has_unsaved_work` yet); the six width cases pass already (B1 must leave them unchanged) and so do the frame-state file's 27 other tests. Nothing fails on collection.

- [ ] **Step 3: The chip's deep link names the provider the block is about**

In `tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py`, Old:
```python
    def open_provider_settings(self) -> None:
        """Deep-link to Settings > Providers & Models for the readout provider."""
        context: dict[str, Any] = {"category": SettingsCategoryId.PROVIDERS_MODELS}
        if self._readout_nav_provider:
            context["provider"] = self._readout_nav_provider
        self.screen.post_message(NavigateToScreen("settings", context))
```
New:
```python
    def open_provider_settings(self, *, defaults_key: str | None = None) -> None:
        """Deep-link to Settings > Providers & Models.

        Args:
            defaults_key: ``None`` names the preview readout's provider (the
                preview pane's link). ``"chat_defaults"`` names the provider
                ``console_handoff_readiness`` checks (the header's blocked
                chip, Roleplay frame B1), so the link lands on the provider
                the block is about.
        """
        provider = self._readout_nav_provider
        if defaults_key is not None:
            raw_config = getattr(self.screen.app_instance, "app_config", {}) or {}
            config = raw_config if isinstance(raw_config, Mapping) else {}
            provider = self._selection_from_defaults(config, defaults_key).provider
        context: dict[str, Any] = {"category": SettingsCategoryId.PROVIDERS_MODELS}
        if provider:
            context["provider"] = provider
        self.screen.post_message(NavigateToScreen("settings", context))
```

- [ ] **Step 4: Rewire `personas_screen.py` (16,528 → 16,525 lines on dev; Step 9 sets the ratchet row to whatever it measures)**

Apply each edit to `tldw_chatbook/UI/Screens/personas_screen.py` with the Edit tool, in order (edit 1's Old line also occurs in `personas_preview_controller.py`: it belongs in PS).

1. The `Click` event import (for the blocked chip). Old:
```python
from textual.css.query import QueryError
```
New:
```python
from textual.css.query import QueryError
from textual.events import Click
```

2. The frame-state and FittedText imports, placed after the `personas_state` block (clear of PR #2862's import hunk). Old:
```python
    MODE_LABELS,
    PersonasWorkbenchState,
)
```
New:
```python
    MODE_LABELS,
    PersonasWorkbenchState,
)
from ..Persona_Modules import roleplay_frame_state as frame_state
from ..Workbench.workbench_widgets import FittedText
```

3. `_MODE_DESCRIPTORS` moves to `frame_state.MODE_DESCRIPTORS`. Old (the table, its blank line, and the next comment line as the anchor):
```python
#: One-line "what this mode is" copy, shown under the title and as chip tooltips.
_MODE_DESCRIPTORS: dict[str, str] = {
    "characters": "Characters — who the AI plays.",
    # F-034: the descriptor teaches the genre convention (characters = who
    # the AI plays, personas = who YOU play) instead of the vague "assistant
    # profiles" - without reviving the retired human-identity framing.
    "personas": "Personas — who you play in the chat.",
    "prompts": "Prompts — moving to the Library.",
    "dictionaries": "Dictionaries — text find/replace rules.",
    "lore": "Lore — world facts injected on keywords.",
}

#: Modes genuinely coming to Roleplay — their chips carry the "· soon" marker.
```
New (the anchor alone):
```python
#: Modes genuinely coming to Roleplay — their chips carry the "· soon" marker.
```

4. The retired task-523 comment in `BUNDLED_CSS` (the badge it explains is gone; comments are stripped from the bundle, so only the blank line after it leaves the boot bytes). Old (the comment, its blank line, and the next rule line as the anchor):
```python
    /* Red cue when the staged Console handoff provider is unready (task-523):
       the "Blocked" badge word turns $ds-status-blocked. The rule CANNOT live
       here — app-bundle CSS (`.ds-status-badge { color: $ds-text-primary }`)
       outranks any widget DEFAULT_CSS regardless of specificity — so it lives
       in the app-tier source css/components/_workbench.tcss instead. */

    #personas-mode-strip {
```
New (the anchor alone):
```python
    #personas-mode-strip {
```

5. The cached header inputs replace the painted gated reason. Old:
```python
        self._console_header_block_reason: str | None = None
```
New:
```python
        self._header_inputs: frame_state.RoleplayHeaderInputs | None = None
```

6. The one-row header: the kind is the subtitle, the composed status already names the data source (`frame_state.initial_header_state`: `Local` or `Server`, so the shared header's default "Ready" chip has nothing to paint whatever the mount order), and the tail holds the item label and the two chips. The tail is a plain `Widget`: Textual's container classes each parse their own `DEFAULT_CSS` on first use (a `HorizontalGroup` measured +1 CSS source on the destination tour, over B1's +1 budget), and a plain `Widget` has no height rule of its own, so the bundle-less test tier keeps its base three-row header (a `Horizontal` is `height: 1fr` there and pushed the list off-screen in the dry run). Old:
```python
            yield DestinationHeader(
                WorkbenchHeaderState(
                    title="Roleplay",
                    subtitle=self._header_subtitle_text(),
                    status="ready",
                ),
                id="personas-header",
            )
```
New:
```python
            yield DestinationHeader(
                frame_state.initial_header_state(
                    self.state.active_mode, self.state.runtime_source
                ),
                before_status=Widget(
                    FittedText(
                        fit=frame_state.fit_header_item, id="personas-header-item"
                    ),
                    FittedText(hide_when_empty=True, id="personas-header-unsaved"),
                    FittedText(hide_when_empty=True, id="personas-header-blocked"),
                    id="personas-header-tail",
                ),
                id="personas-header",
                classes="personas-header-inline",
            )
```

7. Paint the whole header (item, chips, server label) as soon as the runtime source is known, before the first frame. Old:
```python
        self._set_persona_editor_runtime_source(self.persona_handler.current_mode())
        self.query_one(PersonasLibraryPane).set_mode(self.state.active_mode)
```
New:
```python
        self._set_persona_editor_runtime_source(self.persona_handler.current_mode())
        self._update_title()
        self.query_one(PersonasLibraryPane).set_mode(self.state.active_mode)
```

8. A resize refits the chips from the cached inputs (no readiness or data work on resize, spec §2.13). Old:
```python
        self._sync_responsive_workbench()

    def _sync_responsive_workbench(self) -> None:
```
New:
```python
        self._sync_responsive_workbench()
        self._paint_header()

    def _sync_responsive_workbench(self) -> None:
```

9. `_header_subtitle_text` is replaced by `frame_state.build_header_view`. TASK-34400 (on dev) escapes the name into this subtitle with `escape_markup`; B1 removes the name from the subtitle altogether (it paints in the literal item label), so the escape goes with the method. The module's `escape_markup` import stays: three `EnhancedFileOpen` titles still use it. Old (the method, its blank line, and the next method's first line as the anchor):
```python
    def _header_subtitle_text(self) -> str:
        """Live header subtitle: destination purpose plus the editing state."""
        suffix = " - unsaved" if self.state.has_unsaved_changes else ""
        if self._edit_mode == "create":
            noun = "persona" if self.state.active_mode == "personas" else "character"
            return f"New {noun}{suffix}"
        # TASK-34400: the shared header subtitle parses markup; names are untrusted.
        if self._edit_mode == "edit":
            name = self.state.selected_entity_name or "item"
            return f"Editing {escape_markup(name)}{suffix}"
        # Upstream improvement kept: surface the selected entity in view mode
        # when it has unsaved changes, instead of the bare purpose line.
        if self.state.has_unsaved_changes and self.state.selected_entity_name:
            return f"{escape_markup(self.state.selected_entity_name)}{suffix}"
        return "Author the pieces that shape a chat"

    def _mode_descriptor_text(self, mode: str) -> str:
```
New (the anchor alone):
```python
    def _mode_descriptor_text(self, mode: str) -> str:
```

10. The descriptor lookup reads the moved table. Old:
```python
        return _MODE_DESCRIPTORS.get(mode, MODE_LABELS.get(mode, mode))
```
New:
```python
        return frame_state.mode_descriptor(mode)
```

11. `_update_title` gathers, then paints; the logged statement and its indentation stay byte-identical, so the production diagnostic inventory does not drift. Old:
```python
    def _update_title(self) -> None:
        """Refresh the destination header; tolerate updates racing teardown."""
        try:
            header = self.query_one("#personas-header", DestinationHeader)
        except Exception:
            logger.opt(exception=True).debug("Could not update the personas header.")
            return
        # Same input as the inspector's readiness line (task-440): a staged
        # character/persona whose resolved provider would not answer must
        # not claim "Ready" - the existing degraded-state badge ("Blocked")
        # is the header's own established pattern (see stats_screen.py) for
        # this, so no new header UI is introduced. The fuller "what to do"
        # remedy text stays in the inspector's readiness line below it.
        provider_block_reason = self._provider_send_block_reason()
        status = "blocked" if provider_block_reason else "ready"
        header.sync_state(
            WorkbenchHeaderState(
                title="Roleplay",
                subtitle=self._header_subtitle_text(),
                status=status,
            )
        )
        self._console_header_block_reason = provider_block_reason
```
New:
```python
    def _update_title(self) -> None:
        """Gather the header's inputs and repaint it (spec 1.3; never "Ready")."""
        self._header_inputs = self._gather_header_inputs()
        self._paint_header()

    def _gather_header_inputs(self) -> frame_state.RoleplayHeaderInputs:
        """Read every header input; the chip follows the ADR-046 aggregate (R24)."""
        ready, _reason = self.preview.console_handoff_readiness()
        return frame_state.RoleplayHeaderInputs(
            mode=self.state.active_mode,
            edit_mode=self._edit_mode,
            item_name=self.state.selected_entity_name or "",
            unsaved=frame_state.roleplay_has_unsaved_work(
                self._aggregate_roleplay_draft_snapshot()
            ),
            provider_blocked=not ready,
            runtime_source=self.state.runtime_source,
            server_label=frame_state.runtime_server_label(self.app_instance),
        )

    def _paint_header(self) -> None:
        """Push the header view for the cached inputs; tolerate teardown races."""
        try:
            header = self.query_one("#personas-header", DestinationHeader)
        except Exception:
            logger.opt(exception=True).debug("Could not update the personas header.")
            return
        if self._header_inputs is None:
            return
        width = header.outer_size.width or self.size.width  # first paint: screen
        view = frame_state.build_header_view(self._header_inputs, width)
        header.sync_state(view.state)
        header.query_one("#personas-header-item", FittedText).set_value(view.item)
        chips = (("unsaved", view.unsaved_chip), ("blocked", view.blocked_chip))
        for chip, text in chips:
            header.query_one(f"#personas-header-{chip}", FittedText).set_value(text)
```

12. The purpose line formats through the moved helper (same copy; it stays until B6). Only the descriptor lookup and the final format move; the count gathering stays as it is (B2a owns the counts, spec §5.11). Old:
```python
        mode = self.state.active_mode
        descriptor = self._mode_descriptor_text(mode)
        count: int | None = None
```
New:
```python
        mode = self.state.active_mode
        count: int | None = None
```
Old:
```python
            count = len(self._lore_books_cache)
        if count is None:
            return descriptor
        return f"{descriptor.rstrip('.')} · {count}"
```
New:
```python
            count = len(self._lore_books_cache)
        return frame_state.purpose_line(mode, count)
```

13. The 0.25 s readiness poll also repaints when any header input changed: the aggregate (visual authoring, staged avatars and in-flight saves change it without any other refresh), the destination block, the kind after an unguarded mode switch (the voice-profile handoff), the runtime source. Its unmounted short-circuit still comes first. Old:
```python
        reason = self._provider_send_block_reason()
        # A background read can complete between header and inspector paints.
        if (
            reason == self._console_readiness_block_reason
            and reason == self._console_header_block_reason
        ):
            return
```
New:
```python
        reason = self._provider_send_block_reason()
        # A background read can complete between header and inspector paints;
        # the header also follows the draft aggregate and the destination block.
        if (
            reason == self._console_readiness_block_reason
            and self._gather_header_inputs() == self._header_inputs
        ):
            return
```

14. The blocked chip deep-links to Settings for the chat_defaults provider, through app navigation (and so through the leave guard), beside the preview pane's own provider link. Old:
```python
    @on(PreviewGreetingSelected)
```
New:
```python
    @on(Click, "#personas-header-blocked")
    def _open_chat_provider_settings(self, event: Click) -> None:
        """The blocked chip opens Settings for the provider a chat would use."""
        event.stop()
        self.preview.open_provider_settings(defaults_key="chat_defaults")

    @on(PreviewGreetingSelected)
```

15. A persona save repaints after clearing its in-flight flag (the post-save sync inside `_after_profile_save` runs while the flag is still set, so the chip would stay "Unsaved changes" after a successful Ctrl+S until the poll). Old:
```python
        try:
            await self._after_profile_save(saved, source=mode)
        finally:
            self._profile_save_operation_inflight = False
            if not profile_save_completion.done():
                profile_save_completion.set_result(None)
```
New (the completion resolves first: a repaint that raised must never leave a guard awaiting the save hanging, nor mask the save's own outcome):
```python
        try:
            await self._after_profile_save(saved, source=mode)
        finally:
            self._profile_save_operation_inflight = False
            if not profile_save_completion.done():
                profile_save_completion.set_result(None)
            self._update_title()
```

16. The aggregate snapshot reads the cached character editor instead of querying the DOM. On dev only the leave and quit guards read the aggregate, so its `query_one(PersonasCharacterEditorWidget)` costs nothing today; B1's 0.25 s poll gathers the header's inputs on every tick, and that query walks the whole screen and raises `NoMatches` while the editor is not mounted (the normal browsing state): measured in review at 3.5-4.2 ms a tick against 0.05-0.35 ms for the old poll, and 0.1 ms with this edit. The editor is only ever built by `_build_center_view` and cached by `_ensure_center_view` (`on_unmount` clears the cache), so `_ready_center_view` returns exactly the mounted editor or `None`. Old:
```python
        attachments_dirty = False
        try:
            editor = self.query_one(PersonasCharacterEditorWidget)
            attachments_dirty = editor.has_unsaved_attachment()
        except (AttributeError, QueryError):
            pass
```
New:
```python
        attachments_dirty = False
        # The cached demand-mounted editor, never a DOM query: the 0.25 s
        # readiness poll gathers this aggregate on every tick (Roleplay B1).
        try:
            editor = self._ready_center_view("character-editor")
            attachments_dirty = editor is not None and editor.has_unsaved_attachment()
        except (AttributeError, QueryError):
            pass
```

17. `WorkbenchHeaderState` has no use left in PS (edit 6 composes `frame_state.initial_header_state`, edit 11 paints `view.state`), so its import goes (`ruff` F401 otherwise). The line sits five lines above PR #2862's import hunk, outside its context. Old:
```python
from ..Navigation.shortcut_context import ShortcutAction, ShortcutContext
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Workbench.workbench_widgets import DestinationHeader
```
New:
```python
from ..Navigation.shortcut_context import ShortcutAction, ShortcutContext
from ..Workbench.workbench_widgets import DestinationHeader
```

Then: `wc -l < /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1/tldw_chatbook/UI/Screens/personas_screen.py` → `16525` on `8d502ba250`, `a793acbef5` and `8c4dfe59a2` (the base's count − 3 on whatever base Task 0 recorded; Step 9's script sets the ratchet row to this measurement, never the other way round: ruling 2), and `…/python -m ruff format --check tldw_chatbook/UI/Screens/personas_screen.py` → `1 file already formatted` (PS is format-clean on dev; Global Constraints).

- [ ] **Step 4b: The leave and Ctrl+Q guard decides on the same predicate (R24, G4; TASK-33622.14's module)**

`roleplay_draft_guard.py` is imported lazily by PS's `confirm_navigation`/`confirm_quit`, and `roleplay_frame_state` is already eager on the Roleplay route from edit 2, so this import adds nothing to any census. In `tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py`, Old:
```python
from ..Navigation.character_conversation_navigation import (
    RoleplayDraftNavigationDialog,
    RoleplayDraftRecoveryDialog,
    RoleplayDraftSnapshot,
)
```
New:
```python
from ..Navigation.character_conversation_navigation import (
    RoleplayDraftNavigationDialog,
    RoleplayDraftRecoveryDialog,
    RoleplayDraftSnapshot,
)
from .roleplay_frame_state import roleplay_has_unsaved_work
```
The first decision (no prompt when nothing is unsaved). Old:
```python
    if snapshot.is_clean:
        return True
```
New:
```python
    if not roleplay_has_unsaved_work(snapshot):
        return True
```
After Save and continue. Old:
```python
            if not failures and screen._aggregate_roleplay_draft_snapshot().is_clean:
                return True
```
New:
```python
            if not failures and not roleplay_has_unsaved_work(
                screen._aggregate_roleplay_draft_snapshot()
            ):
                return True
```
After Discard and continue. Old:
```python
        return screen._aggregate_roleplay_draft_snapshot().is_clean
```
New:
```python
        return not roleplay_has_unsaved_work(
            screen._aggregate_roleplay_draft_snapshot()
        )
```
Then `…/python -m ruff format --check tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py` → `1 file already formatted`, and `grep -c "is_clean" tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py` → `0` (every decision goes through the predicate; `inflight_save_domains` still decides only whether to wait).

- [ ] **Step 5: Retire the task-523 red-badge rule, rebuild, tighten the boot-bundle pin**

In `tldw_chatbook/css/components/_workbench.tcss`, Old:
```css
/* task-523: red cue when the Roleplay Console handoff provider is unready.
   `status-blocked` lands on the DestinationHeader (#personas-header); the
   badge word is the child #workbench-header-status (also .ds-status-badge,
   whose app-tier `color: $ds-text-primary` this id+class+descendant selector
   outranks so the "Blocked" word actually turns red). */
#personas-header.status-blocked .workbench-header-status {
    color: $ds-status-blocked;
}

/* Resume is briefly disabled while Roleplay dispatches exact typed navigation.
```
New (the anchor alone):
```css
/* Resume is briefly disabled while Roleplay dispatches exact typed navigation.
```

In `Tests/UI/test_roleplay_stylesheet.py`, Old:
```python
    """Every Roleplay rule moved: the bundle keeps only the module banner, and
    no header or shell token B1 adds is left in boot."""
```
New:
```python
    """Every Roleplay rule moved: the bundle keeps only the module banner, and
    no header or shell token B1 styles (nor the retired task-523
    ``#personas-header.status-blocked`` rule) is left in boot."""
```
Old:
```python
        if token.startswith(("personas-header-", *ROLEPLAY_PANE_CLASS_NAMES))
```
New:
```python
        if token.startswith(("personas-header", *ROLEPLAY_PANE_CLASS_NAMES))
```

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py > /dev/null
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py 2>&1 | tail -2
grep -c "personas-header" tldw_chatbook/css/tldw_cli_modular.tcss
git status --short tldw_chatbook/css
EOF
```

Expected: the sync check passes; `0` (no `personas-header` token left in the bundle; the Roleplay sheet owns it now, which also makes it a split-only token for the harness scan — every styled Roleplay harness pins `APP_STYLESHEETS`); ` M` for `_workbench.tcss`, `tldw_cli_modular.tcss` and `widget_defaults_scoped.tcss` (PS's `BUNDLED_CSS` lost a comment and a blank line).

- [ ] **Step 6: Run the header tests to see them pass**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_header.py Tests/UI/test_roleplay_stylesheet.py Tests/UI/test_roleplay_frame_harness.py Tests/UI/test_roleplay_frame_state.py -q -p no:cacheprovider -n 6 --timeout=180 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1`
Expected: `118 passed` (67 + 11 + 12 + 28).

What the dry run measured, for the record: the header is `(0, 3, W, 1)` at every size under both tiers; the subtitle paints `Characters` at 80x24 (the screen carries `shell-header-compact`); the first item sits at y 17 (row 18) at 80x24 and y 18 (row 19) at 120x36, 160x45 and 220x55 — the purpose line, the mode strip and the workbench chrome keep their rows, so the item is exactly 13 (compact workbench, ≤ 90 columns) or 14 rows under the header, as on the base arm; at 120x36 the list window is 13 rows, 7 two-line rows show their name line (5 before); the work pane and the Inspector keep 58/78/108 and 30/39/54 at 120, 160 and 220 columns on both tiers, as on the base arm; the bundle-less tier keeps a three-row header (title, subtitle, status; the auto-height tail is empty there).

- [ ] **Step 7: Re-pin the tests that pinned the old header**

The card's 22 `#personas-header` references: the ones that still hold (the `ds-destination-header` class, the title, the purpose line, the destination-shells and visual-parity contracts) stay as they are; these change.

In `Tests/UI/test_personas_workbench.py`, the header band: the workbench sits two rows (purpose, mode strip) under a one-row header. Old:
```python
            # ...so the workbench starts one row higher than the old layout.
            assert screen.query_one("#personas-workbench").region.y == 10
```
New:
```python
            # Roleplay frame B1: the header is one row, and the purpose line and
            # the mode strip sit right under it (y 6 at 170x50 with today's
            # 3-row nav; asserted relative to the measured header, spec 5.7.2).
            header = screen.query_one("#personas-header")
            assert header.region.height == 1
            assert (
                screen.query_one("#personas-workbench").region.y
                == header.region.bottom + 2
            )
```

Title, status word and the create-mode item. Old:
```python
            # F-031 auto-selects the first row on first paint, which makes
            # the provider gate operative (task-440): the mock config has no
            # ready provider, so the header honestly reads Blocked.
            assert str(status.renderable) == "Blocked"
            # dynamic suffix still appends in create mode
            screen._edit_mode = "create"
            screen._update_title()
            await pilot.pause()
            subtitle = str(
                screen.query_one(
                    "#personas-header #workbench-header-subtitle", Static
                ).renderable
            )
            assert "New character" in subtitle
```
New:
```python
            # Roleplay frame B1: the status word is the data source, never a
            # readiness badge; the provider block is the header's own chip.
            assert str(status.renderable) == "Local"
            assert screen.query_one("#personas-header-blocked").display is True
            # The kind is the subtitle; the item (interim) names a new draft.
            screen._edit_mode = "create"
            screen._update_title()
            await pilot.pause()
            subtitle = screen.query_one(
                "#personas-header #workbench-header-subtitle", Static
            )
            assert str(subtitle.renderable) == "Characters"
            item = screen.query_one("#personas-header-item", FittedText)
            assert item.value == ("New character", True)
```

An unready handoff provider: the status stays the data source and the block chip shows. Old:
```python
            header_status = str(
                screen.query_one(
                    "#personas-header #workbench-header-status", Static
                ).renderable
            )
            assert header_status != "Ready"

    async def test_readiness_surfaces_stay_ready_with_a_configured_provider(
```
New:
```python
            header_status = str(
                screen.query_one(
                    "#personas-header #workbench-header-status", Static
                ).renderable
            )
            assert header_status == "Local"  # never "Ready" (B1, RP-067)
            assert screen.query_one("#personas-header-blocked").display is True

    async def test_readiness_surfaces_stay_ready_with_a_configured_provider(
```

The configured-provider test's docstring. Old:
```python
        """Provider ready -> "Ready to chat in Console."/"Ready" copy shows."""
```
New:
```python
        """Provider ready -> "Ready to chat in Console."; the header shows no
        block chip and its status stays the data source (B1: never "Ready")."""
```

A ready provider: no block chip. Old:
```python
            assert "Ready to chat in Console." in str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert (
                str(
                    screen.query_one(
                        "#personas-header #workbench-header-status", Static
                    ).renderable
                )
                == "Ready"
            )
            assert (
                screen.query_one("#personas-attach-to-console", Button).disabled
                is False
            )
```
New:
```python
            assert "Ready to chat in Console." in str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert (
                str(
                    screen.query_one(
                        "#personas-header #workbench-header-status", Static
                    ).renderable
                )
                == "Local"
            )
            assert screen.query_one("#personas-header-blocked").display is False
            assert (
                screen.query_one("#personas-attach-to-console", Button).disabled
                is False
            )
```

Handoff provider unready while the character provider is ready. Old:
```python
            assert "openai" in readiness_text.lower()
            assert (
                str(
                    screen.query_one(
                        "#personas-header #workbench-header-status", Static
                    ).renderable
                )
                != "Ready"
            )
```
New:
```python
            assert "openai" in readiness_text.lower()
            assert (
                str(
                    screen.query_one(
                        "#personas-header #workbench-header-status", Static
                    ).renderable
                )
                == "Local"
            )
            assert screen.query_one("#personas-header-blocked").display is True
```

Handoff provider ready while the character provider is not. Old:
```python
            assert "Ready to chat in Console." in str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert (
                str(
                    screen.query_one(
                        "#personas-header #workbench-header-status", Static
                    ).renderable
                )
                == "Ready"
            )

    async def test_action_gate_precedes_provider_readiness_on_both_surfaces(
```
New:
```python
            assert "Ready to chat in Console." in str(
                screen.query_one("#personas-readiness-console", Static).renderable
            )
            assert screen.query_one("#personas-header-blocked").display is False

    async def test_action_gate_precedes_provider_readiness_on_both_surfaces(
```

The action-gate test's rationale, re-scoped (spec §1.3: the header block is destination-wide; the Inspector's gate is per staged item). Old:
```python
        """Qodo #824-2: with the action gate closed (unsaved edits), the
        provider axis is NOT operative -- the inspector shows the ACTION
        reason (never provider copy) and the header keeps its pre-task-440
        semantics rather than claiming a conflicting provider-"Blocked".
        One precedence rule on both surfaces: action gate first, provider
        readiness only once the gate opens."""
```
New:
```python
        """Qodo #824-2, re-scoped by Roleplay frame B1 (spec 1.3): the
        inspector speaks for the STAGED item, so with its action gate closed
        (unsaved edits) it shows the ACTION reason and never provider copy.
        The header now speaks for the DESTINATION: its block chip says no chat
        provider is ready whatever is selected, beside the unsaved chip. The
        two no longer conflict -- they answer different questions -- and the
        header never claims "Ready"."""
```

The action-gate test's header assertions. Old:
```python
            assert "openai" not in readiness_text.lower()  # no provider copy
            header_status = str(
                screen.query_one(
                    "#personas-header #workbench-header-status", Static
                ).renderable
            )
            assert header_status == "Ready"  # pre-task-440 header semantics
```
New:
```python
            assert "openai" not in readiness_text.lower()  # no provider copy
            header_status = str(
                screen.query_one(
                    "#personas-header #workbench-header-status", Static
                ).renderable
            )
            assert header_status == "Local"
            assert screen.query_one("#personas-header-blocked").display is True
            assert screen.query_one("#personas-header-unsaved").display is True
```

The blocked-class test becomes the block-chip test (renamed: its old name would now lie). Old:
```python
    async def test_header_carries_blocked_class_when_provider_unready(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Task-523: the header carries the ``status-blocked`` class (the red
        cue's CSS hook) while the staged handoff provider is unready, and drops
        it once the provider becomes ready."""
```
New:
```python
    async def test_header_blocked_chip_follows_the_handoff_provider(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Roleplay frame B1 (replaces task-523's ``status-blocked`` class): the
        header's block chip shows while the chat provider is unready and goes
        once it is ready; the header never takes the ``status-blocked`` class."""
```

Its body. Old:
```python
            screen = await self._select_first_character(pilot)
            header = screen.query_one("#personas-header")
            assert header.has_class("status-blocked") is True

            mock_app_instance.app_config["api_settings"] = {
                "anthropic": {"api_key": "unit-test-placeholder-key"}
            }
            screen._sync_title_and_console_actions()
            await pilot.pause()
            assert header.has_class("status-blocked") is False
```
New:
```python
            screen = await self._select_first_character(pilot)
            header = screen.query_one("#personas-header")
            chip = screen.query_one("#personas-header-blocked")
            assert chip.display is True
            assert header.has_class("status-blocked") is False

            mock_app_instance.app_config["api_settings"] = {
                "anthropic": {"api_key": "unit-test-placeholder-key"}
            }
            screen._sync_title_and_console_actions()
            await pilot.pause()
            assert chip.display is False
```

The red-badge test becomes the red-chip test under the lazy sheet (renamed). Old:
```python
    async def test_blocked_header_badge_renders_red_under_real_bundle(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Task-523 regression guard for the red cue's CSS cascade.

        The colour rule MUST live in app-tier CSS: a widget ``DEFAULT_CSS``
        rule is outranked by the bundle's ``.ds-status-badge`` (color:
        $ds-text-primary) regardless of selector specificity, so the badge
        would stay primary and the cue would never render. Uses
        ``StyledPersonasTestApp`` (loads the real bundle) and asserts the
        blocked-state badge colour DIFFERS from the ready-state colour - if the
        rule were outranked, both states would render the identical primary
        colour and this fails.
        """
```
New:
```python
    async def test_blocked_chip_renders_red_under_the_lazy_sheet(
        self, mock_app_instance, stub_characters, stub_conversations
    ):
        """Roleplay frame B1 (replaces the task-523 red badge): the block chip
        is coloured by the lazy Roleplay sheet, which only a styled tier loads
        (``StyledPersonasTestApp`` = the boot bundle plus every split sheet),
        in the readable error hue ($text-error via $ds-status-error-readable).
        The neutral status word must NOT turn red: it is the data source."""
```

Its body. Old:
```python
            screen = await self._select_first_character(pilot)
            badge = screen.query_one(
                "#personas-header #workbench-header-status", Static
            )
            assert screen.query_one("#personas-header").has_class("status-blocked")
            blocked_color = badge.styles.color

            mock_app_instance.app_config["api_settings"] = {
                "anthropic": {"api_key": "unit-test-placeholder-key"}
            }
            screen._sync_title_and_console_actions()
            await pilot.pause()
            assert not screen.query_one("#personas-header").has_class("status-blocked")
            ready_color = badge.styles.color

        assert blocked_color != ready_color
```
New:
```python
            screen = await self._select_first_character(pilot)
            status = screen.query_one(
                "#personas-header #workbench-header-status", Static
            )
            chip = screen.query_one("#personas-header-blocked")
            assert chip.display is True
            error = pilot.app.get_css_variables()["text-error"]
            assert chip.styles.color.rgb == Color.parse(error).rgb
            assert status.styles.color != chip.styles.color

            mock_app_instance.app_config["api_settings"] = {
                "anthropic": {"api_key": "unit-test-placeholder-key"}
            }
            screen._sync_title_and_console_actions()
            await pilot.pause()
            assert chip.display is False
```

The editing-state test: kind subtitle, item label and the aggregate-driven chip. Old:
```python
            screen = await self._edit_first_character(pilot)
            subtitle = screen.query_one(
                "#personas-header #workbench-header-subtitle", Static
            )
            text = str(subtitle.renderable)
            assert "Editing Detective Sam" in text
            assert "unsaved" not in text
            await self._type_in_description(pilot, screen)
            assert "Editing Detective Sam - unsaved" in str(subtitle.renderable)
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # Save-in-place: the editor stays open, so the header keeps
            # showing "Editing <name>" (just without the "- unsaved" suffix
            # now that the save cleared it).
            assert str(subtitle.renderable) == "Editing Detective Sam"
```
New:
```python
            screen = await self._edit_first_character(pilot)
            subtitle = screen.query_one(
                "#personas-header #workbench-header-subtitle", Static
            )
            item = screen.query_one("#personas-header-item", FittedText)
            chip = screen.query_one("#personas-header-unsaved", FittedText)
            assert str(subtitle.renderable) == "Characters"
            assert item.value == ("Detective Sam", True)
            assert chip.display is False
            await self._type_in_description(pilot, screen)
            assert chip.display is True
            assert chip.value == "Unsaved changes"
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # Save-in-place: the editor stays open, so the item still reads
            # "editing", and the save cleared the aggregate, so the chip goes.
            assert item.value == ("Detective Sam", True)
            assert chip.display is False
```

The `FittedText` import. Old:
```python
from tldw_chatbook.UI.tts_profile_recovery import dependency_recovery_actions
```
New:
```python
from tldw_chatbook.UI.tts_profile_recovery import dependency_recovery_actions
from tldw_chatbook.UI.Workbench.workbench_widgets import FittedText
```

The `Color` import. Old:
```python
from textual.app import App
from textual.screen import Screen
```
New:
```python
from textual.app import App
from textual.color import Color
from textual.screen import Screen
```

A new persona-save test in `TestPersonasMode`: with the poll stopped, the save path alone must clear the chip. Old:
```python
    async def test_local_profile_editor_roundtrip_builds_local_update_schema(
```
New:
```python
    async def test_persona_save_clears_the_unsaved_chip_without_the_poll(
        self, mock_app_instance, stub_characters, stub_scope_service
    ):
        """Roleplay frame B1: the post-save sync inside ``_after_profile_save``
        runs while the persona save is still flagged in flight, so a chip
        repainted only there would stay "Unsaved" after a successful save until
        some unrelated refresh. With the readiness poll stopped, the save path
        alone must clear it."""
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(160, 50)) as pilot:
            screen = await self._enter_personas_mode(pilot)
            screen._console_readiness_poll_timer.stop()
            await pilot.click("#personas-library-row-persona-p-1")
            await pilot.pause()
            screen.post_message(EditPersonaProfileRequested("p-1"))
            await pilot.pause()
            screen.state.has_unsaved_changes = True
            screen._update_title()
            chip = screen.query_one("#personas-header-unsaved", FittedText)
            assert chip.display is True
            screen.post_message(
                PersonaProfileSaveRequested({"id": "p-1", "name": "Archivist 2"})
            )
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            stub_scope_service.update_persona_profile.assert_awaited_once()
            assert screen._profile_save_operation_inflight is False
            assert chip.display is False

    async def test_local_profile_editor_roundtrip_builds_local_update_schema(
```

In `Tests/UI/test_personas_dictionaries.py`, the Options-dirty test reads the chip. Old:
```python
            await pilot.pause()
            assert screen.state.has_unsaved_changes is True
            subtitle = screen.query_one(
                "#personas-header #workbench-header-subtitle", Static
            )
            assert "- unsaved" in str(subtitle.renderable)

    async def test_reverting_edit_clears_dirty_flag(
```
New:
```python
            await pilot.pause()
            assert screen.state.has_unsaved_changes is True
            # Roleplay frame B1: the header's unsaved chip (spec R24).
            assert screen.query_one("#personas-header-unsaved").display is True

    async def test_reverting_edit_clears_dirty_flag(
```

The revert test reads the chip. Old:
```python
            assert screen.state.has_unsaved_changes is True
            subtitle = screen.query_one(
                "#personas-header #workbench-header-subtitle", Static
            )
            assert "- unsaved" in str(subtitle.renderable)
            name_input.value = original
            await pilot.pause()
            assert screen.state.has_unsaved_changes is False
            subtitle = screen.query_one(
                "#personas-header #workbench-header-subtitle", Static
            )
            assert "- unsaved" not in str(subtitle.renderable)
```
New:
```python
            assert screen.state.has_unsaved_changes is True
            chip = screen.query_one("#personas-header-unsaved")
            assert chip.display is True
            name_input.value = original
            await pilot.pause()
            assert screen.state.has_unsaved_changes is False
            assert chip.display is False
```

In `Tests/UI/test_personas_subscription_readiness.py`, the readiness-race test waits for the block chip instead of a "Ready" word. Old:
```python
        screen._sync_title_and_console_actions()
        header_status = screen.query_one(
            "#personas-header #workbench-header-status", Static
        )
        assert len(reads) >= 2
        async with asyncio.timeout(3):
            while str(header_status.renderable) != "Ready":
                await asyncio.sleep(0.01)
```
New:
```python
        screen._sync_title_and_console_actions()
        # Roleplay frame B1: the header's destination block chip (no longer a
        # "Ready" badge) is the header surface that must catch up.
        blocked = screen.query_one("#personas-header-blocked")
        assert len(reads) >= 2
        async with asyncio.timeout(3):
            while blocked.display:
                await asyncio.sleep(0.01)
```

In `Tests/UI/test_unified_shell_phase6_first_time_replay.py`, the first-time replay required the retired subtitle line (its only source was `_header_subtitle_text`, which edit 9 deletes); it now requires the purpose line's copy, which stays until B6. Old:
```python
                        "Author the pieces that shape a chat",
```
New:
```python
                        "who the AI plays",  # Roleplay frame B1: the purpose line
```

TASK-34400's two hostile-text modules each pinned the retired subtitle once; both now pin the surface that replaces it (the item label and the escaped status), and Task 8 adds the painted, clicked copies under a styled tier. In `Tests/UI/test_roleplay_hostile_names.py`, the character test's docstring. Old:
```python
    """Library row, card, Inspector, conversation row, the Tag filter button,
    a toast and the header's ``Editing`` subtitle."""
```
New:
```python
    """Library row, card, Inspector, conversation row, the Tag filter button,
    a toast and the header's item label while editing."""
```
Its last block. Old:
```python
        # Editing puts the name in the shared header's subtitle.
        screen.post_message(EditCharacterRequested("1"))
        await settle(pilot)
        assert screen._edit_mode == "edit"
        assert f"Editing {name}" in _painted(screen)
        assert click_meta_cells(screen) == []
```
New:
```python
        # Roleplay frame B1: editing names the item in the header's literal
        # item label, never in the shared markup-on subtitle (the kind). This
        # unstyled tier gives the label no row to paint on, so the painted
        # and clicked copy is pinned under styled tier 1 by
        # test_the_header_item_label_and_server_label.
        screen.post_message(EditCharacterRequested("1"))
        await settle(pilot)
        assert screen._edit_mode == "edit"
        item = screen.query_one("#personas-header-item", FittedText)
        assert item.value == (name, True)
        subtitle = screen.query_one("#workbench-header-subtitle", Static)
        assert str(subtitle.render()) == "Characters"
        assert click_meta_cells(screen) == []
```
Its imports. Old:
```python
from textual.widgets import ListView
```
New:
```python
from textual.widgets import ListView, Static
```
Old:
```python
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
```
New:
```python
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Workbench.workbench_widgets import FittedText
```

In `Tests/UI/test_roleplay_hostile_text_surfaces.py`, the subtitle test called `_header_subtitle_text`, which edit 9 deletes; the header view replaces it. Old:
```python
@pytest.mark.parametrize("text", HOSTILE_TEXT)
def test_the_header_subtitle_names_an_unsaved_item_literally(text):
    """View mode with unsaved edits puts the bare name in the shared header's
    markup-parsing subtitle; ``Editing`` mode is covered by the flow test."""
    from textual.content import Content

    screen = object.__new__(PersonasScreen)
    screen._edit_mode = "view"
    screen.state = SimpleNamespace(
        has_unsaved_changes=True,
        selected_entity_name=text,
        active_mode="characters",
    )
    assert Content.from_markup(screen._header_subtitle_text()).plain == (
        f"{text} - unsaved"
    )
```
New:
```python
@pytest.mark.parametrize("text", HOSTILE_TEXT)
def test_the_header_view_keeps_an_unsaved_item_out_of_markup(text):
    """Roleplay frame B1 replaced the subtitle that named an unsaved item: the
    shared header's markup-parsing subtitle names only the kind, the item goes
    to the literal item label as plain text, and a server label reaches the
    markup-parsing status chip only escaped (it parses back to what it shows)."""
    from textual.content import Content

    from tldw_chatbook.UI.Persona_Modules import roleplay_frame_state as fs

    view = fs.build_header_view(
        fs.RoleplayHeaderInputs(
            mode="characters",
            item_name=text,
            unsaved=True,
            runtime_source="server",
            server_label=text,
        ),
        220,
    )
    assert Content.from_markup(view.state.subtitle).plain == "Characters"
    assert view.item == (text, False)
    assert view.unsaved_chip == "Unsaved changes"
    assert Content.from_markup(view.state.status_label).plain == view.status_plain
    assert view.status_plain.startswith(f"Server: {text}")
```

Then run the re-pinned tests:

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/plug /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase6_first_time_replay.py -p b1_bootstrap_all -q -p no:cacheprovider --timeout=300 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1`
Expected: `1 passed` (about 18 s; it is in no PR lane, so only the paired `roleplay` group would otherwise see it).

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/plug /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_subscription_readiness.py -p b1_bootstrap_all -k "header or title or readiness or settings_edit_marks_dirty or reverting_edit_clears_dirty or route_renders_destination_workbench or action_gate_precedes or blocked or persona_save_clears or purpose" -q -p no:cacheprovider -n 6 --timeout=180 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1`
Expected: `35 passed` (every selected test; 35 in the dry runs on `8c4dfe59a2`, `a793acbef5` and `8d502ba250`).

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/plug /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_hostile_names.py Tests/UI/test_roleplay_hostile_text_surfaces.py -p b1_bootstrap_all -q -p no:cacheprovider -n 6 --timeout=300 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1`
Expected: `103 passed` (TASK-34400's 22 + 81: the two re-pinned tests keep their parametrisations).

- [ ] **Step 8: Paired arms for the Roleplay and shell groups**

```bash
bash <<'EOF'
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence
"$EV/paired.sh" t6-roleplay "$EV/suites-roleplay.txt" -n 6 | tee -a "$EV/gates.txt"
"$EV/paired.sh" t6-shell "$EV/suites-shell.txt" -n 6 | tee -a "$EV/gates.txt"
EOF
```

Expected: no new failures on head in either group (the first dry run: none beyond two load flakes, `test_persona_actor_pack_save_pins_selected_portrait_identity` and `test_settings_action_rows_keep_every_button_whole[size1]`, each 3/3 green in isolation on both arms). Step 4b changes TASK-33622.14's guard, so read `test_roleplay_quit_guard.py` and `test_quit_flow_prompt_choke_point.py` in the failure-set diff with care. On dev (`8d502ba250`) the whole quit-guard file is load-sensitive on BOTH arms: any of its nine tests can fail, alone or under `-n 6`, a different subset each run, and each failure read so far is in the setup helper, before any Ctrl+Q or guard step runs (`AssertionError: timed out waiting for the character editor to load the saved card`, beside an `app.__init__` log line `ChaChaNotesDB (CharactersRAGDB) instance not found`; Task 12 Step 4 lists the measured runs). So a "new" quit-guard failure proves nothing by itself: re-run the whole file alone on both arms, read each failure's message (one past that setup wait, or one that names the guard or a quit decision, is B1's to explain), and apply Task 12 Step 4's flake rule. `test_destination_visual_parity_correction.py`'s Roleplay rows (`SOURCE_PREP_WORKBENCHES["personas"]`, `COMPACT_DESTINATION_CONTRACTS["personas"]`) are the spec's "re-check": they pass on both arms (the workbench moves up and the panes grow, within both bounds). Apply Task 12 Step 4's flake rule to anything new.

- [ ] **Step 9: Ratchet rows set to the measurement (ruling 2), then the pre-import raise (ruling 1)**

Set B1's module-size ratchet rows from measurements (§5.4 item 4; #2862's convention; ruling 2), by script so the numbers are measured, not typed. For each ratcheted file B1 edits (PS and `app.py`), the row's "before" value comes from the BASE arm's copy of the ratchet file (never B1's own row): PS's row becomes its measured count, lowered or raised; `app.py`'s row changes only if `app.py` outgrew it (B1 does not shrink `app.py`). A raised row carries a dated owner-decision comment naming TASK-33910.2 (ruling 2) and B1's own delta, head lines minus the base arm's lines (never head minus the row, which would book dev's slack or overrun to B1); a lowered one says what moved out. A row that is already red on the base arm (its file over the row before B1 touches it; dev does not enforce this ratchet: ten rows were red there on `a793acbef5` and again on `8d502ba250`) is dev's: the script prints `STOP:` and raises nothing, because Task 0 Step 2 checks the PS row only once, before the first move. The script also adds the recipient-ceiling row for `roleplay_frame_state.py`. It is an upsert, because the rebase protocol re-runs it after every move: it first drops B1's earlier comment lines (each carries the marker `Roleplay frame B1`) and any `roleplay_frame_state.py` row, rewrites each row from the base value, and asserts each key occurs exactly once. It runs twice here to prove a re-run changes nothing.

```bash
bash <<'OUTER'
set -euo pipefail
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence
PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
cat > "$EV/ratchet_rows.py" <<'EOF'
"""Set B1's module-size ratchet rows to the measurement; idempotent (re-run after every move).

Run from the B1 worktree root. Owner ruling of 2026-10-04 (ruling 2): a
ratcheted file B1 grows gets its row raised to the measured count with a dated
owner-decision comment naming TASK-33910.2, never squeezed code; PS's row is
also lowered when B1 shrinks it (spec 5.4 item 4). Each row's "before" value is
read from the base arm's copy of the ratchet file ($EV/base-sha.txt), so a
re-run never reports B1's own row as the starting point, and B1's own delta is
measured against the base arm's line count, never against its row. A row that
is already red on the base arm is dev's: the script STOPs instead of raising it
in B1's name. Every comment line it writes carries MARKER, which is how a
re-run finds and replaces them.
"""
import re
import subprocess
from pathlib import Path

EV = Path("/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence")
REL = "Tests/Architecture/test_module_size_ratchet.py"
PS = "tldw_chatbook/UI/Screens/personas_screen.py"
APP = "tldw_chatbook/app.py"
FS = "tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py"
MARKER = "Roleplay frame B1"


def lines_of(rel: str) -> int:
    return len(Path(rel).read_text(encoding="utf-8").splitlines())


def row(rel: str) -> re.Pattern[str]:
    return re.compile(rf'^    "{re.escape(rel)}": (\d+),$', re.M)


def git_show(rev_path: str) -> str:
    return subprocess.run(
        ["git", "show", rev_path], capture_output=True, text=True, check=True
    ).stdout


base_sha = (EV / "base-sha.txt").read_text(encoding="utf-8").strip()
base_text = git_show(f"{base_sha}:{REL}")
path = Path(REL)
text = path.read_text(encoding="utf-8")
text = re.sub(rf"^    #[^\n]*{MARKER}[^\n]*\n", "", text, flags=re.M)
text = re.sub(rf'^    "{re.escape(FS)}": \d+,\n', "", text, flags=re.M)
report = []
for rel, may_lower in ((PS, True), (APP, False)):
    old = int(row(rel).search(base_text).group(1))
    before = len(git_show(f"{base_sha}:{rel}").splitlines())
    if before > old:
        raise SystemExit(
            f"STOP: {rel} is {before:,} lines on the base arm, over its {old:,} row: "
            "that red row is dev's, not B1's to raise; report it"
        )
    measured = lines_of(rel)
    current = row(rel).search(text)
    assert current, f"{rel} row not found"
    if measured > old:
        new = measured
        comment = (
            f"    # {MARKER} (TASK-33910.2), owner decision 2026-10-04: B1 adds\n"
            f"    # {measured - before:+,} lines ({before:,} -> {measured:,}); {MARKER} raises\n"
            f"    # the row {old:,} -> {new:,}, never squeezes code ({MARKER}).\n"
        )
    elif measured < old and may_lower:
        new = measured
        comment = (
            f"    # {MARKER} (TASK-33910.2): the header and purpose-line text moved\n"
            f"    # to roleplay_frame_state.py, {old:,} -> {new:,} lines ({MARKER}).\n"
        )
    else:
        new, comment = old, ""
    text = text.replace(current.group(0), f'{comment}    "{rel}": {new},', 1)
    report.append(f"{Path(rel).name} row {old} -> {new} (B1's own {measured - before:+d} lines)")
ps_row = row(PS).search(text)
text = text.replace(
    ps_row.group(0),
    ps_row.group(0)
    + f"\n    # {MARKER}: recipient ceiling for the moved header code (spec 5.4).\n"
    + f'    "{FS}": {lines_of(FS)},',
    1,
)
for key in (PS, APP, FS):
    assert text.count(f'"{key}":') == 1, f"{key} must have exactly one row"
path.write_text(text, encoding="utf-8")
print("; ".join(report) + f"; roleplay_frame_state row {lines_of(FS)}")
EOF
"$PY" "$EV/ratchet_rows.py"
FIRST=$(shasum Tests/Architecture/test_module_size_ratchet.py)
"$PY" "$EV/ratchet_rows.py"
[ "$FIRST" = "$(shasum Tests/Architecture/test_module_size_ratchet.py)" ] && echo "idempotent: the second run changed nothing"
"$PY" -m ruff format --check Tests/Architecture/test_module_size_ratchet.py
"$PY" -m pytest Tests/Architecture/test_module_size_ratchet.py -k "personas_screen or roleplay_frame_state or app.py" -q -p no:cacheprovider 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1
OUTER
```

Expected: the line `personas_screen.py row 16528 -> 16525 (B1's own -3 lines); app.py row 5712 -> 5712 (B1's own +2 lines); roleplay_frame_state row 369` twice (on `8d502ba250`, as on `a793acbef5` and `8c4dfe59a2`), `idempotent: the second run changed nothing`, `1 file already formatted`, and `6 passed` (each of the three rows' budget and slack checks). If a rebase or a review fix made PS or `app.py` larger than its base row, the line says so (`row N -> M` with M > N) and the row carries the owner-decision comment with B1's own delta: that is ruling 2 working, not a failure. A `STOP:` line means the base arm's own file is already over its row (dev left that row red): do not raise it in B1's name; report it to the controller. Other rows may be red on dev: compare the whole file's failure set in Task 12's `css` group, never here.

From this commit PS imports `roleplay_frame_state`, so the required guard `test_preimport_pass_payload_stays_within_budget` is red until a constant is raised. The raise lands in THIS commit, under the owner's 2026-10-04 sign-off (ruling 1) that Task 0 Step 6 recorded. The only stop here is the script's own: a `STOP:` line means the growth left the owner's bound (or the arms are stale), so do not commit; report it to the controller.

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b1-evidence
test -s "$EV/owner-signoff.txt" || { echo "STOP: owner-signoff.txt is empty: run Task 0 Step 6, which records the owner's 2026-10-04 answer"; exit 1; }
"$EV/preimport_measure.sh" base && "$EV/preimport_measure.sh" head
"$PY" "$EV/preimport_raise.py"
cd $MAIN/.worktrees/roleplay-b1
"$PY" scripts/update_boot_budget_snapshots.py --only preimport 2>&1 | tail -3
"$PY" -m pytest Tests/Performance/test_screen_preimport_payload_budget.py -p no:cacheprovider -q 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1
git diff --stat -- Tests/Performance backlog/decisions/097-boot-budget-ratchets.md
EOF
```

Expected (the dry run on `8d502ba250`): `base | modules 557 | LOC 416217 | roleplay (ccp) 62 mods / 54812 LOC | fattest route 128281 LOC`, `head | modules 558 | LOC 416596 | roleplay (ccp) 63 mods / 55191 LOC | fattest route 128281 LOC` (on `a793acbef5`: LOC 416115 → 416494, fattest route 128223; small LOC drift is fine; the LOC and largest-route limits, 425,347 and 135,111 on dev, keep their headroom); `raised: MAX_PASS_ADDED_MODULES 557 -> 558 | ledger row: written`; the snapshot refresh line; `1 passed`; a diff stat naming the test file, the ADR and `preimport_payload.json`. The new ledger row is dated `2026-10-04` and ends `` | Owner, 2026-10-04, asked whether the expanded limits also cover B1's one new module (`roleplay_frame_state`); answer, verbatim: "Expand it (Rec.)" | ``. If #2862 landed first, the `raised:` line may name other constants too, which is also correct (the script raises exactly what the head exceeds, within the owner's bound). Any `STOP:` line: stop and report.

- [ ] **Step 10: Named mutations**

Each with the Edit tool (and `build_css.py` for `.tcss`), run, restore, re-run green:
1. `subtitle-hidden-at-24`: in `_roleplay.tcss` delete `display: block;` from the `.workbench-header-subtitle` rule; rebuild. `…/python -m pytest Tests/UI/test_roleplay_header.py -q -p no:cacheprovider -k "kind_stays_visible and 80x24"`: `2 failed` (both tiers; the subtitle region is empty).
2. `header-not-inline`: in the `#personas-header.personas-header-inline` rule delete `layout: horizontal;`; rebuild. `-k kind_stays_visible`: `8 failed` (both tiers, four sizes: the parts stack vertically inside the one-row box, so the kind is never painted).
3. `chip-on-has-unsaved`: in `_gather_header_inputs` change `unsaved=frame_state.roleplay_has_unsaved_work(` … `),` to `unsaved=bool(self.state.has_unsaved_changes),`. `-k aggregate_not_has_unsaved`: `7 failed` (timed out waiting for the chip).
4. `persona-save-before-flag`: delete the `self._update_title()` line added at the end of the profile-save `finally`. `…/python -m pytest Tests/UI/test_personas_workbench.py -q -p no:cacheprovider -p b1_bootstrap_all -k persona_save_clears` (with `PYTHONPATH=$EV/plug`): `1 failed` (`assert True is False` on the chip).
5. `kind-gap-drift`: set `KIND_GAP_CELLS = 3` in `roleplay_frame_state.py`. `-k chrome_cells`: `2 failed` (both tiers).
6. `resize-no-repaint`: delete the `self._paint_header()` line edit 8 added to `on_resize`. `-k resize_refits`: `2 failed` (90x45 and 80x24 keep `Unsaved changes` and the 160-column status; 220x55 still passes, as it needs no change).
7. `resize-gathers`: change that line to `self._update_title()`. `-k resize_refits`: `3 failed` (`gathered == [1, 1]`: a resize re-read readiness and the drafts).
8. `item-not-flexible`: in `_roleplay.tcss` change `#personas-header-item`'s `width: $ds-width-fill;` to `width: auto;`; rebuild. `-k "worst_case or long_name"`: `16 failed` (both tests, both tiers, four sizes: the 213-cell name takes its whole natural width, so the chips and the status leave the header's painted window and no `…` ending paints).
9. `go-marker-unresolved`: in `fit_header_item` change `marker = f"{resolve_glyph(_GO)} "` to `marker = f"{_GO} "`. `…/python -m pytest Tests/UI/test_roleplay_frame_state.py Tests/UI/test_roleplay_header.py -q -p no:cacheprovider -n 6 -k ascii_markers`: `9 failed` (the pure ASCII fit and the mounted ASCII row under both tiers at four sizes: a `›` paints in ASCII mode).
10. `compose-ready-default`: in `roleplay_frame_state.initial_header_state` delete the `status_label=…` argument. `…/python -m pytest Tests/UI/test_roleplay_header.py Tests/UI/test_roleplay_frame_state.py -q -p no:cacheprovider -k "composes_without_a_ready_chip or never_says_ready"`: `2 failed` (the composed status label is empty, so the shared header would paint "Ready"; the pure pin sees it too).
11. `aggregate-walks-dom`: in edit 16 put back `editor = self.query_one(PersonasCharacterEditorWidget)` and `attachments_dirty = editor.has_unsaved_attachment()`. `-k walks_no_dom`: `1 failed` (`queried` holds the editor query).
12. `guard-on-is-clean`: in Step 4b's first decision put back `if snapshot.is_clean:`. `…/python -m pytest Tests/UI/test_roleplay_frame_state.py -q -p no:cacheprovider -k same_predicate`: `1 failed` (the patched predicate is ignored and the dialog is asked).
13. `inspector-wider`: in `css/components/_agentic_terminal.tcss` change `#personas-inspector-pane`'s `width: $ds-fr-2;` to `width: $ds-fr-3;`; rebuild. `-k keep_their_widths`: `6 failed` (both tiers, three sizes; spec 5.3's interim geometry: the pin sees a pane width move).

Record all thirteen in `$EV/mutations.md`. (Mutations 1, 3, 6 and 7 were run while planning on the `bd41347b65` dry run, 8-13 on the `83c264f286` dry run, and each failed exactly as stated; if one does not go red when you run it, replace it with one that does before ticking AC#11.)

- [ ] **Step 11: Lint and commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py Tests/UI/test_roleplay_header.py Tests/UI/test_roleplay_frame_state.py Tests/UI/test_roleplay_stylesheet.py Tests/UI/test_personas_workbench.py Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_subscription_readiness.py Tests/UI/test_unified_shell_phase6_first_time_replay.py Tests/UI/test_roleplay_hostile_names.py Tests/UI/test_roleplay_hostile_text_surfaces.py Tests/Architecture/test_module_size_ratchet.py Tests/Performance/test_screen_preimport_payload_budget.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py Tests/UI/test_roleplay_header.py Tests/UI/test_roleplay_frame_state.py Tests/UI/test_personas_workbench.py Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_subscription_readiness.py Tests/UI/test_unified_shell_phase6_first_time_replay.py Tests/UI/test_roleplay_hostile_names.py Tests/UI/test_roleplay_hostile_text_surfaces.py Tests/Architecture/test_module_size_ratchet.py Tests/Performance/test_screen_preimport_payload_budget.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_persistent_diagnostic_inventory.py 2>&1 | tail -1
git add tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py tldw_chatbook/UI/Persona_Modules/roleplay_draft_guard.py tldw_chatbook/css/components/_workbench.tcss tldw_chatbook/css/tldw_cli_modular.tcss tldw_chatbook/css/widget_defaults_scoped.tcss Tests/UI/test_roleplay_header.py Tests/UI/test_roleplay_frame_state.py Tests/UI/test_roleplay_stylesheet.py Tests/UI/test_personas_workbench.py Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_subscription_readiness.py Tests/UI/test_unified_shell_phase6_first_time_replay.py Tests/UI/test_roleplay_hostile_names.py Tests/UI/test_roleplay_hostile_text_surfaces.py Tests/Architecture/test_module_size_ratchet.py Tests/Performance/test_screen_preimport_payload_budget.py backlog/decisions/097-boot-budget-ratchets.md Tests/Performance/boot_budget_snapshots/preimport_payload.json
git commit -m "feat(roleplay): one-row header with unsaved and blocked chips (B1)" -m "Roleplay  <Kind> › <item> | Unsaved changes | No chat provider · Settings › | Local or Server: <label> · read-only, never Ready. The chip and TASK-33622.14's leave and Ctrl+Q guard follow one predicate over the ADR-046 aggregate (in-flight saves included); the readiness poll gathers the header inputs without walking the DOM; a persona save repaints after its flag clears and its completion resolves; the blocked chip deep-links to the chat_defaults provider. The header and purpose text and the mode descriptors moved to roleplay_frame_state; PS's size-ratchet row is set to the measurement (owner ruling 2026-10-04). The dead task-523 badge rule leaves boot. The two TASK-34400 hostile-text tests that asserted the retired subtitle now pin the item label and the escaped status. The owner-approved pre-import raise (2026-10-04, \"Expand it\") and its ADR-097 row land in this commit." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: `All checks passed!`, `13 files already formatted`, `diagnostic inventory verified: …`, the commit line.

---

### Task 7: Prove the styled tiers discriminate (AC#6)

A geometry assertion that cannot go red proves nothing. The card's self-test — "deleting one `_roleplay.tcss` header rule turns a header geometry assertion red under both styled tiers" — is written as an executable negative control: `drop_rule_from_loaded_sheet` edits the parsed source the running app holds, so the same test works whether the sheet arrived by a harness `CSS_PATH` (mock tier) or by the app's route loader (full tier; a `KeyError` there would itself mean the route never loaded it).

**Files:**
- Modify: `Tests/UI/test_roleplay_frame_harness.py`

**Interfaces:**
- Consumes: `drop_rule_from_loaded_sheet`, `open_styled_roleplay`, `styled_tiers`, `settle`, `wait_until`, `ROLEPLAY_SHEET` (Task 1); the `#personas-header.personas-header-inline` rule (Task 5); the inline header (Task 6).
- Produces: nothing for later tasks.

- [ ] **Step 1: Write the test**

In `Tests/UI/test_roleplay_frame_harness.py`, Old:
```python
    seed_mock_characters,
    settle,
    styled_tiers,
)
```
New:
```python
    seed_mock_characters,
    settle,
    styled_tiers,
    wait_until,
)
```
Old:
```python
def test_dropping_a_rule_that_is_not_there_is_a_loud_error():
```
New:
```python
@styled_tiers
async def test_deleting_one_header_rule_turns_the_one_row_assertion_red(
    styled_tier, mock_app_instance, one_character
):
    """AC#6: the discrimination check, run as an executable negative control."""
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=(120, 36)
    ) as pilot:
        screen = pilot.app.screen
        header = screen.query_one("#personas-header")
        assert header.region.height == 1
        drop_rule_from_loaded_sheet(
            pilot.app, ROLEPLAY_SHEET, "#personas-header.personas-header-inline"
        )
        await settle(pilot)
        await wait_until(
            pilot, lambda: header.region.height > 1, what="the header to regrow"
        )


def test_dropping_a_rule_that_is_not_there_is_a_loud_error():
```

- [ ] **Step 2: Run it (it passes now that the header is inline; it would fail before Task 6 on its first assertion)**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_frame_harness.py -q -p no:cacheprovider --timeout=180 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1`
Expected: `14 passed` (12 + the new test under both tiers).

- [ ] **Step 3: Show the drop really reaches the assertions (named mutation `drop-inline-header-rule`)**

Run `Tests/UI/test_roleplay_header.py -k "(kind_stays_visible or first_list_item or seven_characters) and 120x36"` while the rule is deleted from the source instead (delete the whole `#personas-header.personas-header-inline { … }` block from `_roleplay.tcss`, rebuild with `build_css.py`): `6 failed` (both tiers, three tests: the header is 6 rows, `assert 6 == 1`, because the tail widget adds a row to the base five; the first item no longer sits right under a one-row header; fewer than 7 characters show, the first now at y 23). This is the AC#2 geometry's red run too (§5.4 item 2(a)). Restore the block with the Edit tool, rebuild, `6 passed`. Record the row, together with the executable control above, in `$EV/mutations.md` (spec §5.4 item 2a).

- [ ] **Step 4: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check Tests/UI/test_roleplay_frame_harness.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check Tests/UI/test_roleplay_frame_harness.py
git add Tests/UI/test_roleplay_frame_harness.py
git commit -m "test(roleplay): the styled tiers discriminate a deleted header rule (B1 AC#6)" -m "Deleting the inline-header rule from the sheet the running app loaded regrows the header under both the mock tier and the real app." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

---

### Task 8: Hostile names on B1's new header surfaces (R33, AC#7)

On dev today no existing Roleplay surface parses an untrusted name as markup: TASK-34400 (PR #3015) fixed the pre-existing bug in which a character named `[/]` raised `MarkupError` while the screen was drawn and exited the whole app, and in which `[@click=…]` names became live click actions. That fix (the Inspector Statics, the Tag filter button, every toast, the old subtitle) is not B1's and this task does not redo it. B1 adds new places where a name paints: the header's item label (a literal `FittedText`, Tasks 3 and 6) and the status chip's server label (escaped by `build_header_view`, Task 4, into the shared markup-on header). This task extends TASK-34400's `Tests/UI/test_roleplay_hostile_names.py` to exactly those surfaces, under styled tier 1, where the header gives the item label its row (the unstyled tier does not, so TASK-34400's existing flow tests cannot see it paint). It changes no production code: Tasks 3, 4 and 6 already made both surfaces literal, so the new test passes when written, and Step 3's mutations show it can fail. The fixture is TASK-34400's `HOSTILE_NAMES`, which already adds `[/` and `[TODO] y` (both mishandled by `textual.markup.escape` on Textual 8.2.8), so the test also pins the escaper choice.

**Files:**
- Modify (test only): `Tests/UI/test_roleplay_hostile_names.py`

**Interfaces:**
- Consumes: `StyledRoleplayMockApp`, `open_styled_roleplay(app_class=...)`, `click_meta_cells`, `painted_rows`, `seed_mock_characters`, `settle`, `wait_until` (Task 1); `FittedText` (Task 3); the module's own `HOSTILE_NAMES`, `_RecordingApp`, `_assert_literal_and_inert`, `_painted` (TASK-34400); `EditCharacterRequested`.
- Produces: `_RecordingStyledApp` (the recording app under styled tier 1) and `test_the_header_item_label_and_server_label`, for later slices to extend.

- [ ] **Step 1: Write the test**

In `Tests/UI/test_roleplay_hostile_names.py`, the imports. Old:
```python
from unittest.mock import AsyncMock, Mock
```
New:
```python
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
```
Old:
```python
from Tests.UI.roleplay_frame_harness import (
    click_meta_cells,
    painted_rows,
    seed_mock_characters,
    settle,
    wait_until,
)
```
New:
```python
from Tests.UI.roleplay_frame_harness import (
    StyledRoleplayMockApp,
    click_meta_cells,
    open_styled_roleplay,
    painted_rows,
    seed_mock_characters,
    settle,
    wait_until,
)
```
The styled recording app, after the unstyled one. Old:
```python
    def action_record(self, value: str) -> None:
        self.recorded.append(value)
```
New:
```python
    def action_record(self, value: str) -> None:
        self.recorded.append(value)


class _RecordingStyledApp(_RecordingApp, StyledRoleplayMockApp):
    """``_RecordingApp`` under styled tier 1 (Roleplay frame B1): only the lazy
    Roleplay sheet gives the header's item label and chips their row."""
```
The B1 test, after the character flow test. Old:
```python
@pytest.mark.parametrize("name", HOSTILE_NAMES)
async def test_a_persona_name(name, mock_app_instance, monkeypatch):
```
New:
```python
@pytest.mark.parametrize("name", HOSTILE_NAMES)
async def test_the_header_item_label_and_server_label(
    name, mock_app_instance, monkeypatch
):
    """Roleplay frame B1's header surfaces: the item label (a literal
    ``FittedText``, viewing and editing) and the status chip's server label
    (escaped by ``build_header_view`` into the shared markup-on header)."""
    seed_mock_characters(
        monkeypatch, [{"id": 1, "name": name, "description": "d", "version": 1}]
    )
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, app_class=_RecordingStyledApp
    ) as pilot:
        screen = pilot.app.screen
        item = screen.query_one("#personas-header-item", FittedText)
        # Alone in the library, so the first-paint auto-selection (F-031) picks it.
        await wait_until(
            pilot,
            lambda: item.value == (name, False),
            what="the hostile name in the header",
        )
        pilot.app.runtime_policy = SimpleNamespace(
            state=SimpleNamespace(last_known_server_label=name, active_server_id=None)
        )
        screen._set_persona_editor_runtime_source("server")
        screen._update_title()
        await settle(pilot)
        assert item.fitted_text == f"› {name}"
        painted = _painted(screen)
        assert f"› {name}" in painted
        assert f"Server: {name} · read-only" in painted
        # Library row, card name, Inspector, header item label and status.
        await _assert_literal_and_inert(pilot, name, at_least=5)
        # Editing is a local-only action: back to the local source first.
        screen._set_persona_editor_runtime_source("local")
        screen._update_title()
        await settle(pilot)
        screen.post_message(EditCharacterRequested("1"))
        await settle(pilot)
        assert screen._edit_mode == "edit"
        assert f"› {name} · editing" in _painted(screen)
        assert click_meta_cells(screen) == []


@pytest.mark.parametrize("name", HOSTILE_NAMES)
async def test_a_persona_name(name, mock_app_instance, monkeypatch):
```

- [ ] **Step 2: Run it (it passes: Tasks 3, 4 and 6 already made both surfaces literal; Step 3 shows it can fail)**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/plug /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_hostile_names.py Tests/UI/test_roleplay_hostile_text_surfaces.py -p b1_bootstrap_all -q -p no:cacheprovider -n 6 --timeout=300 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1`
Expected: `108 passed` (TASK-34400's 22 + 81, and the new test's five names; `108 passed` again on `a793acbef5` and `8d502ba250`; the dry run on `8c4dfe59a2` painted each name exactly five times: library row, card name, Inspector, header item label and status).

- [ ] **Step 3: Named mutations (each verified in the dry run)**

Run `…/python -m pytest Tests/UI/test_roleplay_hostile_names.py -p b1_bootstrap_all -q -p no:cacheprovider -n 5 -k header_item_label` (with `PYTHONPATH=$EV/plug`) after each, then restore with the Edit tool and re-run green (`5 passed`):
1. `status-unescaped`: in `roleplay_frame_state.build_header_view` change `status_label=escape_markup(status),` to `status_label=status,`: `5 failed`.
2. `item-label-markup`: in `FittedText.render` change `return Content(self.fitted_text)` to `return Content.from_markup(self.fitted_text)`: `5 failed`.

Record both rows in `$EV/mutations.md`.

- [ ] **Step 4: Lint and commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check Tests/UI/test_roleplay_hostile_names.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check Tests/UI/test_roleplay_hostile_names.py
git add Tests/UI/test_roleplay_hostile_names.py
git commit -m "test(roleplay): hostile names on the new header item label and server label (B1, R33)" -m "Extends TASK-34400's hostile-name test to the surfaces B1 adds: the header's literal item label (viewing and editing) and the server label escaped into the shared markup-on status chip, under styled tier 1, with a positive control and a click on every painted copy. No production change: on dev these names already paint literally everywhere else (TASK-34400 fixed the pre-existing crash), and Tasks 3, 4 and 6 built the new surfaces literal." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: `All checks passed!`, `1 file already formatted`, the commit line.

---

### Task 9: The B1 journey — Edit → type → chip → Ctrl+S → chip goes (AC#5)

Spec §5.4 item 2 and §5.7.3: each journey a slice touches lives in `Tests/UI/test_roleplay_journeys.py` from that slice on. B1's is the unsaved-chip journey, under both styled tiers, driven like a user (a click on the card's Edit, a keystroke, Ctrl+S). Both tiers stub the character seams (Task 1: the full tier is not yet DB-seeded, a recorded deviation), so the journey proves that the save reached `update_character` with the typed text and that the chip followed it; save → reload persistence is B2a's to prove once the full tier seeds a real ChaChaNotes (TASK-33910.3 note, Task 13). Later slices add the §5.7.3 journeys J1-J4 here as prefix tests plus strict `xfail(raises=NotYetDelivered)` full journeys; B1 touches none of those rows (J2's slices are B4, B5b-2, B7, B11), so it adds no xfail.

**Files:**
- Test: `Tests/UI/test_roleplay_journeys.py` (create)

**Interfaces:**
- Consumes: `open_styled_roleplay`, `seed_mock_characters`, `settle`, `styled_tiers`, `wait_until` (Task 1); `FittedText` (Task 3); the card's `#personas-card-edit-character` button and `#personas-char-editor-description` (existing); `_conversation_record`, `_install_conversation_db`.
- Produces: the journeys module later slices extend.

- [ ] **Step 1: Write the test**

Create `Tests/UI/test_roleplay_journeys.py`:

```python
"""Roleplay journeys, one test per journey a frame slice touches (spec 5.7.3).

From frame slice B1: Edit -> type -> the header's unsaved chip appears ->
Ctrl+S -> the chip goes (B1 AC#5). Under both styled tiers, driven like a
user: a click on the card's Edit, keystrokes, Ctrl+S. Both tiers stub the
character seams (the full tier is not DB-seeded until B2a), so B1 checks the
payload that reached ``update_character``, not persistence. Later slices add
the spec's J1-J4 journeys here, each as a passing prefix test plus a strict
``xfail(raises=NotYetDelivered)`` full-journey test (RC-7).
"""

from __future__ import annotations

import pytest
from textual.widgets import TextArea

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.UI.roleplay_frame_harness import (
    open_styled_roleplay,
    seed_mock_characters,
    settle,
    styled_tiers,
    wait_until,
)
from Tests.UI.test_personas_workbench import (
    _conversation_record,
    _install_conversation_db,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import FittedText

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@styled_tiers
async def test_editing_shows_the_unsaved_chip_and_saving_clears_it(
    styled_tier, mock_app_instance, monkeypatch
):
    seed_mock_characters(
        monkeypatch,
        [{"id": 1, "name": "Detective Sam", "description": "Noir", "version": 1}],
    )
    monkeypatch.setattr(
        character_handler_module, "_default_character_db", lambda: object()
    )
    saved: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        character_handler_module,
        "update_character",
        lambda cid, data: saved.append((str(cid), dict(data))) or True,
    )
    _install_conversation_db(monkeypatch, [_conversation_record(1)])
    async with open_styled_roleplay(
        styled_tier, mock_app_instance, size=(120, 36)
    ) as pilot:
        screen = pilot.app.screen
        chip = screen.query_one("#personas-header-unsaved", FittedText)
        item = screen.query_one("#personas-header-item", FittedText)
        assert not chip.display

        assert await pilot.click("#personas-card-edit-character")
        await wait_until(pilot, lambda: screen._edit_mode == "edit", what="Edit")
        assert item.value == ("Detective Sam", True)
        assert not chip.display  # opening the editor is not an edit

        screen.query_one("#personas-char-editor-description", TextArea).focus()
        await pilot.pause()
        await pilot.press("x")
        await wait_until(pilot, lambda: chip.display, what="the unsaved chip")
        assert chip.value == "Unsaved changes"

        await pilot.press("ctrl+s")
        await settle(pilot)
        await wait_until(pilot, lambda: not chip.display, what="the chip to go")
        assert screen._edit_mode == "edit"  # save-in-place keeps the editor
        assert item.value == ("Detective Sam", True)
        # The save reached the character seam with the typed text.
        assert saved and saved[-1][0] == "1" and "x" in saved[-1][1]["description"]
```

- [ ] **Step 2: Run it**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_journeys.py -q -p no:cacheprovider --timeout=180 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1`
Expected: `2 passed` (it passes because Task 6 already delivered the behaviour; Step 3 shows it can fail).

- [ ] **Step 3: Named mutation `chip-never-shown`**

In `roleplay_frame_state.build_header_view`, change `if inputs.unsaved:` to `if False:`. Run the test: `2 failed` (timed out waiting for the unsaved chip). Restore; `2 passed`. Record the row.

- [ ] **Step 4: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check Tests/UI/test_roleplay_journeys.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check Tests/UI/test_roleplay_journeys.py
git add Tests/UI/test_roleplay_journeys.py
git commit -m "test(roleplay): journey - editing shows the unsaved chip, Ctrl+S clears it (B1 AC#5)" -m "Under the styled mock tier and the real app: click Edit, type, the chip appears; Ctrl+S, it goes while the editor stays open." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

---

### Task 10: Governance and the User Guide — DESIGN.md G12, ADR-046 G4 (predicate), the guide's header delta (§5.8)

**Files:**
- Modify: `DESIGN.md` (front matter `components:`; `### Destination Header`), `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md` (a dated amendment), `backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md` (a cross-reference, G4), `Docs/User_Guide/roleplay-chat-dictionaries.md` (the title and three passages)

**Interfaces:**
- Consumes: the shipped behaviour of Tasks 2-9.
- Produces: nothing for later tasks.

No "Verified against" stamp is added anywhere (the worktree CLAUDE.md forbids them; the page's existing legacy stamp at its end is B12's to delete, spec §5.8).

- [ ] **Step 1: DESIGN.md — the inline one-row variant (G12)**

In `DESIGN.md` front matter, Old:
```yaml
  destination-header:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.terminal-tall}"
    padding: "1 2 cells"
```
New:
```yaml
  destination-header:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.terminal-tall}"
    padding: "1 2 cells"
  destination-header-inline:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.none}"
    padding: "0 1 cell"
    height: "1 cell"
```
Old:
```markdown
The destination header is a product contract, not decoration. It carries title, one-line purpose, readiness, authority, primary action, and blocked recovery when needed. It uses `$ds-surface-panel`, `border: tall $ds-action-focus`, `padding: 1 2`, and bold text.
```
New:
```markdown
The destination header is a product contract, not decoration. It carries title, one-line purpose, readiness, authority, primary action, and blocked recovery when needed. It uses `$ds-surface-panel`, `border: tall $ds-action-focus`, `padding: 1 2`, and bold text.

**Destination header variants.** A destination may use one of these instead, keyed by its own id or class in its own sheet:

- **Inline one-row header (Lab, Roleplay).** One row with no border and `padding: 0 1` (`destination-header-inline` above): the title, a subtitle, optional chips, and the authority chip last. Chips are words, never a dot or a badge: an unsaved chip driven by the destination's one draft predicate (Roleplay: **Unsaved changes**), and a blocked chip that names a destination-wide block and its recovery (Roleplay: **No chat provider · Settings ›**). Chip words take the readable status foregrounds (`$ds-status-warning-readable`, `$ds-status-error-readable`), never the decorative status hues, which fall below AA on the panel in many themes. The authority chip says where the data lives (Roleplay: **Local**, or **Server: \<label\> · read-only**); it is never a "Ready" badge. Readiness may sit beside the destination's primary action in its work pane instead of in the header, and the one-line purpose may live in the destination's navigation. The subtitle stays visible at every terminal height. Lab: `.lab-header-inline` (`css/features/_lab.tcss`). Roleplay: `#personas-header.personas-header-inline` (`css/features/_roleplay.tcss`; the Roleplay frame spec of 2026-10-02, §1.3).
- **Console's one-row header ([ADR-210](backlog/decisions/210-console-region-ownership.md)).** One row carrying the workspace, authority and actions, with no title or purpose line: ADR-210's own exception, written into this section by its migration step 8 (TASK-33627). Whichever of that step and this list lands second rebases onto the other.
```

- [ ] **Step 2: ADR-046 — the one unsaved predicate (G4, B1's part)**

In `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`, Old:
```markdown
amendment is owned by
[TASK-31241](../tasks/task-31241%20-%20Align-character-conversation-navigation-decisions.md).

## Context
```
New:
```markdown
amendment is owned by
[TASK-31241](../tasks/task-31241%20-%20Align-character-conversation-navigation-decisions.md).

### 2026-10-03 amendment: one unsaved predicate (Roleplay frame B1)

The aggregate snapshot above is also the one answer to "does Roleplay hold
unsaved work?": `roleplay_has_unsaved_work()` in
`tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py` returns `not
snapshot.is_clean`, which counts every domain and in-flight save the aggregate
snapshot tracks (the character and Persona forms, character and Persona
visuals, staged attachments). From TASK-33910.2 it drives the Roleplay header's
Unsaved chip, and the leave and Ctrl+Q guards
(`roleplay_draft_guard.confirm_roleplay_drafts`, TASK-33622.14) decide on the
same predicate. The
[Roleplay frame spec](../../Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md)
(section 3.12, G4) routes the in-screen guard (B5b-1: selecting another item or
kind, which today checks only the form and visual authoring), the work-pane
title word and the commit-bar summary through the same predicate, and adds the
entry form and lore/dictionary Options as domains (B9a, with their own dated
amendment). The dialog's third choice stays "Stay".

## Context
```

ADR-120's aggregate-veto paragraph is the other half of G4's cross-reference. In `backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md`, Old:
```markdown
and continue, or Stay. Navigation proceeds only after every owned draft domain
is clean; failure keeps drafts mounted and recoverable.
```
New:
```markdown
and continue, or Stay. Navigation proceeds only after every owned draft domain
is clean; failure keeps drafts mounted and recoverable. (2026-10-03: the single
predicate over this aggregate is `roleplay_has_unsaved_work()`; see ADR-046's
2026-10-03 amendment.)
```

- [ ] **Step 3: The User Guide's header delta**

The page's title still carries the retired subtitle line. In `Docs/User_Guide/roleplay-chat-dictionaries.md`, Old:
```markdown
# Roleplay & Chat Dictionaries — Author the pieces that shape a chat
```
New:
```markdown
# Roleplay & Chat Dictionaries
```
Old:
```markdown
- **Header** — the title **Roleplay & Chat Dictionaries**, a subtitle that
  normally reads "Author the pieces that shape a chat" (it changes while
  you edit — "New character", "Editing \<name\>", and " - unsaved" when
  there are unsaved edits), and a status badge reading **Ready** or
  **Blocked**.
```
New:
```markdown
- **Header** — one row: the title **Roleplay**, the kind you are browsing
  (**Characters**, **Personas**, **Dictionaries** or **Lore**) and, for now,
  "› \<name\>" for the selected item, with " · editing" while an editor is
  open. A long name ends in "…" ("..." with ASCII glyphs on); the kind is
  never cut. On the right, in words: **Unsaved changes** (**Unsaved** below
  100 columns) while any Roleplay draft is unsaved or still saving — an
  edited form, a staged avatar, open visual authoring; **No chat provider ·
  Settings ›** while no chat provider is ready, whatever is selected (click
  it to open Settings › Providers & Models); and where your data lives:
  **Local**, or **Server: \<name\> · read-only**. The header never says
  "Ready".
```
Old:
```markdown
  provider the handoff would use isn't ready. Attach still works; Start
  Chat doesn't. The header badge reads **Blocked** in this state too.
```
New:
```markdown
  provider the handoff would use isn't ready. Attach still works; Start
  Chat doesn't. The header shows **No chat provider · Settings ›** whenever
  that provider isn't ready, whatever is selected.
```
Old:
```markdown
  it reflects is your chat provider default — what the Readiness line and
  the **Blocked** badge report on.
```
New:
```markdown
  it reflects is your chat provider default — what the Readiness line and
  the header's **No chat provider** chip report on.
```

- [ ] **Step 4: Claim strings as a read-list (spec §5.7.1 Docs row)**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
END=$(grep -n "Verified against" Docs/User_Guide/roleplay-chat-dictionaries.md | tail -1 | cut -d: -f1)
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_guide_claim_strings.py --end-line "$END" Docs/User_Guide/roleplay-chat-dictionaries.md | tee /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/claim-strings.txt
EOF
```

Read every candidate it prints. The header's strings in the title and the three passages B1 rewrote are of two kinds, checked two ways:
- **Literal fragments**, written as-is in `roleplay_frame_state.py`: `Unsaved changes`, `Unsaved`, `No chat provider`, `Local`, `read-only`, `editing`. The block below counts each there; a zero is a real gap: fix the module or the guide.
- **Composed strings**, built from `resolve_glyph` f-strings (`›` and `·` are glyph-map entries, Task 2): `› <name>` (and ` · editing` after it), `No chat provider · Settings ›`, `Server: <name> · read-only`, and their pieces `Settings ›` and `· read-only`. `roleplay_frame_state.py` never holds them as written, so a grep of the module finds nothing (a repo-wide grep finds `Settings ›` and `· read-only` in other modules, which proves nothing about the header), and the checker lists the full strings as not emitted (on `8d502ba250` it printed `:41 › <name>`, `:48 Server: <name> · read-only` and `:149 No chat provider · Settings ›`): that is expected, not a gap. The mounted painted-text assertions check them instead, and the block runs them: `test_blocked_chip_follows_destination_readiness_through_the_poll` (paints `No chat provider · Settings ›`), `test_status_names_the_data_source_and_never_says_ready` (`Local`, then `Server: home-tldw · read-only`), `test_a_long_name_ellipsises_and_the_kind_is_never_cut` (`› Ser…`, the kind uncut) and Task 8's `test_the_header_item_label_and_server_label` (`› <name>`, then `› <name> · editing`, for each hostile name).

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
FS=tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py
for S in "Unsaved changes" "Unsaved" "No chat provider" "Local" "read-only" "editing"; do
  echo "literal in the module: $S x$(grep -cF -- "$S" "$FS")"
done
echo "old title hits: $(grep -rn "Author the pieces" Docs/User_Guide tldw_chatbook Tests | wc -l | tr -d ' ')"
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/plug /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_roleplay_header.py Tests/UI/test_roleplay_hostile_names.py -p b1_bootstrap_all -k "blocked_chip_follows_destination_readiness_through_the_poll or status_names_the_data_source_and_never_says_ready or a_long_name_ellipsises_and_the_kind_is_never_cut or the_header_item_label_and_server_label" -q -p no:cacheprovider -n 6 --timeout=300 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1
EOF
```

Expected: every literal count at least 1 (`x2`, `x3`, `x2`, `x2`, `x1`, `x7` on `8d502ba250`), `old title hits: 0`, and `15 passed` (8 long-name cases, both tiers at four sizes; 1 blocked chip; 1 status; 5 hostile names). Candidates outside those passages are other slices' (B12 consolidates the page): list them in the PR body as "pre-existing, not B1's". The script exits 0 by design.

The page's screenshots predate B1: `Docs/User_Guide/images/roleplay/{overview,character-card,character-editor,dictionary-entries,lore-entries}.svg` (embedded in this page and its three sub-pages) still paint the old "Roleplay & Chat Dictionaries" header with a "Ready" badge, right above prose that now says the header never says "Ready". B1 does not re-capture them (B12, TASK-33910.18, consolidates the guide and its images): list them in the PR body as stale, and Task 13 adds a dated note to TASK-33910.18. Confirm the list is still exact: `grep -l '>Ready<' Docs/User_Guide/images/roleplay/*.svg` prints those five files.

- [ ] **Step 5: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
for T in /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1-base /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1; do
  (cd "$T" && PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/plug /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_master_shell_design_system_contract.py -p b1_bootstrap_all -q -p no:cacheprovider 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1)
done
git add DESIGN.md backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md Docs/User_Guide/roleplay-chat-dictionaries.md
git commit -m "docs(roleplay): inline header variant, the one unsaved predicate, the guide's header (B1)" -m "DESIGN.md gains a Destination header variants list (Lab and Roleplay inline; Console per ADR-210) and its token. ADR-046 records roleplay_has_unsaved_work() as the one predicate for the chip and the leave and Ctrl+Q guards; ADR-120's aggregate veto cross-references it. The Roleplay guide describes the one-row header and drops the retired subtitle from its title; no Verified-against stamp." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: `14 passed` on the base arm and `14 passed` on head (the suite reads DESIGN.md and `_workbench.tcss`; without the bootstrap plugin 14 of its 15 tests ERROR at setup on BOTH arms with `RecoveryRequired: raw_source_selection_changed`, which would compare two error sets and verify nothing), then the commit line.

---

### Task 11: Gate the new tests on pull requests (§5.4 item 3, MF-15, FU-3)

Two lanes, because two kinds of file:
- **UI Fast Lane** (`scripts/ui_pr_gate_census.txt`; since TASK-34353 the census runs as round-robin shards, one 20-minute job per entry of the workflow's `matrix.shard`, whose files run serially in census order; dev has added shards since: four on `a793acbef5` and `8d502ba250`, so every script here reads the count from the workflow, never from this paragraph): the four files that mount no destination screen and carry no `bootstrap_profile` mark — B1's `test_roleplay_frame_state.py`, `test_roleplay_stylesheet.py` and `test_workbench_fitted_text.py` (44 tests: 28 + 11 + 5), and TASK-34400's `test_roleplay_hostile_text_surfaces.py` (81 tests), which no PR lane ran on dev and which holds one of B1's re-pins (`test_the_header_view_keeps_an_unsaved_item_out_of_markup`, Task 6 Step 7): 125 tests, `125 passed` in 45 s to 47 s serial in the minimal venv on `a793acbef5` (Step 1's first `real`, three runs under a load average of 19 to 45) and 52.5 s on `8d502ba250`. They are appended at the census's end under one comment line, as TASK-33003.20's entry was, so no earlier file changes shard.
- **PR Fast Lane, "Run admission-sensitive suites"** (`.github/workflows/derived-artifacts.yml`): the four `bootstrap_profile` files that mount `PersonasScreen` — `test_roleplay_frame_harness.py`, `test_roleplay_header.py`, `test_roleplay_hostile_names.py` (TASK-34400's file, gated by no lane on dev; B1 extends it) and `test_roleplay_journeys.py`: 110 tests (14 + 67 + 27 + 2; the hostile-name file's 27 are TASK-34400's 22 plus B1's five), `110 passed` in 4 m 03 s to 4 m 40 s serial in the minimal venv on `a793acbef5` (four runs under a load average of 19 to 45) and 4 m 20 s on `8d502ba250`. They cannot join the census invocation (TASK-32873: enrolling the collection-time profile poisons sandboxed suites in the same run); this step exists for exactly such suites. Dev's own list in that step keeps growing (four files on `8c4dfe59a2`, fourteen on `a793acbef5` and `8d502ba250`), so Step 3 inserts B1's four by script before the step's `--timeout=300` line instead of matching dev's list, and Step 1 reads dev's list from the workflow.

The lanes' last five green PR runs, read twice on 2026-10-08 (dev `a793acbef5`) and again at `8d502ba250` (by the verifier around 21:00Z and on 2026-10-09 00:35Z; the same five runs, so the same maxima each time), took: UI Fast Lane shard 1 9 m 45 s to 11 m 47 s, shard 2 4 m 35 s to 5 m 29 s, shard 3 12 m 09 s to 15 m 02 s, shard 4 10 m 33 s to 13 m 55 s (of 20); PR Fast Lane 16 m 55 s to 22 m 12 s (of 30). With the slowest of B1's measured runs, 4 m 40 s, the PR Fast Lane's longest recent run comes to 26 m 52 s: under Step 1's 27-minute rule by only 8 s, so Step 1 measures again before committing and its rule decides (a STOP there is real: report it, do not trim tests to fit).

**Files:**
- Modify: `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `.github/workflows/derived-artifacts.yml`; then the final pre-import measurement (only if it moved): `Tests/Performance/test_screen_preimport_payload_budget.py`, `backlog/decisions/097-boot-budget-ratchets.md`, `Tests/Performance/boot_budget_snapshots/preimport_payload.json`

**Interfaces:**
- Consumes: the six new test files, the extended `test_roleplay_hostile_names.py` and the re-pinned `test_roleplay_hostile_text_surfaces.py`; `$EV/preimport_measure.sh`, `$EV/preimport_raise.py`.
- Produces: nothing for later tasks.

- [ ] **Step 1: Measure the lanes**

```bash
bash <<'EOF'
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b1; EV=$MAIN/.worktrees/.b1-evidence
FL=$EV/fastlane-venv
[ -x "$FL/bin/python" ] || uv venv --python 3.12 "$FL" > "$EV/fastlane-venv.log" 2>&1
(cd "$WT" && VIRTUAL_ENV="$FL" uv pip install -e . pytest pytest-asyncio pytest-timeout packaging) > "$EV/fastlane-install.log" 2>&1
cd "$WT"
WF=.github/workflows/derived-artifacts.yml
FAST="Tests/UI/test_roleplay_frame_state.py Tests/UI/test_roleplay_stylesheet.py Tests/UI/test_workbench_fitted_text.py Tests/UI/test_roleplay_hostile_text_surfaces.py"
MOUNTED="Tests/UI/test_roleplay_frame_harness.py Tests/UI/test_roleplay_header.py Tests/UI/test_roleplay_hostile_names.py Tests/UI/test_roleplay_journeys.py"
# Dev's own admission-sensitive files, read from the workflow (never typed), B1's four left out.
ADMISSION=$("$FL/bin/python" - "$WF" $MOUNTED <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
step = text.split("      - name: Run admission-sensitive suites\n", 1)[1].split("--timeout=300", 1)[0]
files = re.findall(r"^\s+(Tests/\S+\.py) \\$", step, re.M)
print(" ".join(path for path in files if path not in sys.argv[2:]))
PY
)
echo "UI Fast Lane shards in the workflow: $(grep -E '^ +shard: \[[0-9, ]+\]$' "$WF" | grep -oE '[0-9]+' | wc -l | tr -d ' ')" | tee -a "$EV/gates.txt"
echo "dev's admission-sensitive files: $(echo $ADMISSION | wc -w | tr -d ' ')" | tee -a "$EV/gates.txt"
# bash's own `time` times the whole pipeline and reports after it (`/usr/bin/time -p cmd 2>&1 | tail -1`
# would feed its report into the pipe and show only its last line, `sys`); grep, not `tail -1`, reads the
# count line and names every failed test.
TIMEFORMAT='real %R s'
echo "lane: UI Fast Lane, the four census files" | tee -a "$EV/gates.txt"
{ time "$FL/bin/python" -m pytest $FAST -p no:cacheprovider -q --timeout=180 2>&1 | grep -E '^(FAILED|ERROR) |[0-9]+ (passed|failed|error)'; } 2>&1 | tee -a "$EV/gates.txt"
echo "lane: PR Fast Lane, B1's four mounted files alone (their share)" | tee -a "$EV/gates.txt"
{ time "$FL/bin/python" -m pytest $MOUNTED -p no:cacheprovider -q --timeout=300 --tb=short 2>&1 | grep -E '^(FAILED|ERROR) |[0-9]+ (passed|failed|error)'; } 2>&1 | tee -a "$EV/gates.txt"
echo "lane: PR Fast Lane, the admission-sensitive step as CI runs it (dev's files, then B1's)" | tee -a "$EV/gates.txt"
{ time "$FL/bin/python" -m pytest $ADMISSION $MOUNTED -p no:cacheprovider -q --timeout=300 --tb=short 2>&1 | grep -E '^(FAILED|ERROR) |[0-9]+ (passed|failed|error)'; } 2>&1 | tee -a "$EV/gates.txt"
"$FL/bin/python" -m pytest $ADMISSION $MOUNTED --collect-only -q -p no:cacheprovider 2>&1 | grep -E '[0-9]+ tests? collected' | tail -1
for RUN in $(gh run list --workflow derived-artifacts.yml --event pull_request --status success --limit 5 --json databaseId -q '.[].databaseId'); do
  gh run view "$RUN" --json jobs -q '.jobs[] | select(.name | test("Fast Lane")) | "\(.name): \(.startedAt) -> \(.completedAt)"'
done | tee -a "$EV/gates.txt"
EOF
```

Expected: the shard count (`4` on `a793acbef5` and `8d502ba250`) and dev's admission-file count (`14` on both); `125 passed` and its `real` line (45 s to 47 s on `a793acbef5`, 52.5 s on `8d502ba250`); `110 passed` and its `real` line (243 s to 280 s on `a793acbef5`, four runs; 260.2 s on `8d502ba250`); the admission step all passed, its passed + xfailed count equal to the last line's `--collect-only` total (dev's files' count plus B1's 110: `445 passed, 2 xfailed` and `447 tests collected` on both SHAs, `real` 729 s and 764.6 s), with its `real` seconds; each lane's last five durations (`UI Fast Lane (1)` to `(4)`, `PR Fast Lane`). A failure in the combined run that does not occur in B1's run alone is the TASK-32873 interaction: stop and report it.

Decision:
- PR Fast Lane: its longest recent duration plus the second `real` (B1's four files alone) must stay under 27 minutes (its job limit is 30). Otherwise stop and report: this lane has no shard rule, so more room for it is the owner's decision (with the slowest B1 run measured, 4 m 40 s on `a793acbef5`: 22 m 12 s + 4 m 40 s = 26 m 52 s, 8 s inside the rule; on `8d502ba250`: 22 m 12 s + 4 m 20 s = 26 m 32 s).
- UI Fast Lane: Step 2 appends the four files and prints the shard each lands in (round-robin by position, the shard count read from the workflow). For each shard that gains files, its longest recent duration plus the first `real` (a conservative bound: the whole four-file run) must stay under 19 m 30 s (on `a793acbef5` and `8d502ba250` each of the four shards gains one file, and the longest is 15 m 02 s + 47 s, or + 52.5 s on `8d502ba250`: 15 m 55 s). If one would not, follow the lane's own rule (the workflow comment, TASK-34353: "Add a shard (not minutes) when one nears the cap"): add the next number to `matrix.shard` in `.github/workflows/derived-artifacts.yml` in this task's commit, re-run Step 2's shard lines, and say so in the PR body. Never raise `timeout-minutes`.

- [ ] **Step 2: Append the four files to the census (after the final rebase; by script, never by hand)**

```bash
bash <<'OUTER'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'EOF'
import re
from pathlib import Path

B1_FILES = [
    "Tests/UI/test_roleplay_frame_state.py",
    "Tests/UI/test_roleplay_stylesheet.py",
    "Tests/UI/test_workbench_fitted_text.py",
    "Tests/UI/test_roleplay_hostile_text_surfaces.py",
]
B1_HEADER = (
    "# Roleplay frame B1 (TASK-33910.2): pure header state, the lazy sheet, FittedText,"
    " and TASK-34400's widget-level hostile-text sinks."
)
census = Path("scripts/ui_pr_gate_census.txt")
lines = census.read_text(encoding="utf-8").splitlines()
# An earlier run's block goes first, wherever a rebase left it; census order
# is run order, so B1 appends and never re-sorts other files.
lines = [line for line in lines if line.strip() not in B1_FILES and line != B1_HEADER]
while lines and not lines[-1].strip():
    lines.pop()
lines += ["", B1_HEADER, *B1_FILES]
census.write_text("\n".join(lines) + "\n", encoding="utf-8")
n = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))

checker = Path("scripts/check_ui_pr_gate_census.py")
text = checker.read_text(encoding="utf-8")
COMMENT_TAIL = (
    "# test_roleplay_stylesheet.py and test_workbench_fitted_text.py gate the pure\n"
    "# header state, the lazy Roleplay sheet's ownership and FittedText; TASK-34400's\n"
    "# test_roleplay_hostile_text_surfaces.py (no lane ran it; B1 re-pins one of\n"
    "# its tests) gates the widget-level hostile-text sinks. B1's mounted Roleplay\n"
    "# files are bootstrap-profile and run in the PR Fast Lane's\n"
    "# admission-sensitive step instead (TASK-32873).\n"
)
# Strip exactly B1's own comment lines from an earlier run, wherever another PR's
# comment lines landed around them (never a lazy run up to MINIMUM_FILES).
text = re.sub(
    r"^# Roleplay frame B1 raised it to \d+: test_roleplay_frame_state\.py,\n"
    + re.escape(COMMENT_TAIL),
    "",
    text,
    flags=re.M,
)
comment = f"# Roleplay frame B1 raised it to {n}: test_roleplay_frame_state.py,\n" + COMMENT_TAIL
text, count = re.subn(r"^MINIMUM_FILES = \d+$", comment + f"MINIMUM_FILES = {n}", text, count=1, flags=re.M)
if count != 1:
    raise SystemExit("no `MINIMUM_FILES = <int>` line found")
checker.write_text(text, encoding="utf-8")
print("census entries:", n)
EOF
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_ui_pr_gate_census.py; echo "exit=$?"
NS=$(grep -E '^ +shard: \[[0-9, ]+\]$' .github/workflows/derived-artifacts.yml | grep -oE '[0-9]+' | wc -l | tr -d ' ')
echo "UI Fast Lane shards: $NS"
for I in $(seq 0 $((NS - 1))); do echo "shard $((I + 1)): $(/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_ui_pr_gate_census.py --shard "$I" "$NS" | grep -cE 'test_roleplay_frame_state|test_roleplay_stylesheet|test_workbench_fitted_text|test_roleplay_hostile_text_surfaces') B1 files"; done
git diff --stat scripts/
OUTER
```

Expected: `census entries: <dev's count + 4>` (158 on `8d502ba250`, whose census lists 154 files under `MINIMUM_FILES = 151`; 156 on `a793acbef5`, 152 files there), the checker's line (`OK: 158 Tests/UI files in the PR gate (floor 158); every listed path exists.` on `8d502ba250`), `exit=0`, `UI Fast Lane shards: <the workflow's count>` (4 on both), one line per shard with its share of the four files (`shard 1: 1 B1 files` through `shard 4: 1 B1 files` on both), and a diff touching only the two `scripts/` files. A second run leaves `git diff --stat scripts/` unchanged. On a rebase conflict in either, take dev's side and re-run.

- [ ] **Step 3: Add the four bootstrap-profile files to the admission-sensitive step (by script, never by hand)**

Dev edits this step's file list often (fourteen files on `a793acbef5` and `8d502ba250`, four on `8c4dfe59a2`), so an Old block on it goes stale between rebases. The script inserts B1's four files just before the step's `--timeout=300 \` line, in the order they run, with a YAML comment above the step (a comment line inside the backslash-continued `pytest` command would end it). It is an upsert: it first removes B1's own lines from an earlier run, then asserts the step and its anchor occur exactly once, and it re-runs after every rebase with Steps 2 and 4.

```bash
bash <<'OUTER'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'EOF'
from pathlib import Path

import yaml

B1_FILES = [
    "Tests/UI/test_roleplay_frame_harness.py",
    "Tests/UI/test_roleplay_header.py",
    "Tests/UI/test_roleplay_hostile_names.py",
    "Tests/UI/test_roleplay_journeys.py",
]
STEP = "      - name: Run admission-sensitive suites\n"
ANCHOR = "            --timeout=300 \\\n"
COMMENT = (
    "      # Roleplay frame B1 (TASK-33910.2): the mounted Roleplay frame suites (both\n"
    "      # styled tiers, hostile names, the journeys) are bootstrap_profile files too.\n"
)
workflow = Path(".github/workflows/derived-artifacts.yml")
text = workflow.read_text(encoding="utf-8").replace(COMMENT, "")
own = {f"            {path} \\" for path in B1_FILES}
text = "\n".join(line for line in text.split("\n") if line not in own)
assert text.count(STEP) == 1, f"expected one admission-sensitive step, found {text.count(STEP)}"
before, after = text.split(STEP)
body, rest = after.split("\n\n", 1)
body += "\n"
assert body.count(ANCHOR) == 1, f"expected one --timeout=300 line in the step, found {body.count(ANCHOR)}"
body = body.replace(ANCHOR, "".join(f"            {path} \\\n" for path in B1_FILES) + ANCHOR)
workflow.write_text(before + COMMENT + STEP + body + "\n" + rest, encoding="utf-8")

steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["pr-fast-lane"]["steps"]
run = next(step["run"] for step in steps if step.get("name") == "Run admission-sensitive suites")
files = [line.strip().rstrip(" \\") for line in run.splitlines() if line.strip().startswith("Tests/")]
assert files[-4:] == B1_FILES, files[-4:]
print(f"admission-sensitive step: {len(files)} files, B1's four last")
EOF
git diff --stat .github/workflows/derived-artifacts.yml
OUTER
```

Expected: `admission-sensitive step: <dev's count + 4> files, B1's four last` (18 on `a793acbef5` and `8d502ba250`) and a diff of `6 +` lines (the two comment lines and the four files) in the workflow. A second run changes nothing. On a rebase conflict in the workflow, take dev's side and re-run this step.

- [ ] **Step 4: Final paired pre-import measurement, raise check and snapshot refresh**

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b1-evidence
"$EV/preimport_measure.sh" base && "$EV/preimport_measure.sh" head
"$PY" "$EV/preimport_raise.py"
cd $MAIN/.worktrees/roleplay-b1
"$PY" scripts/update_boot_budget_snapshots.py --only preimport 2>&1 | tail -3
"$PY" -m pytest Tests/Performance/test_screen_preimport_payload_budget.py -p no:cacheprovider -q 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tail -1
"$PY" -c 'import json,sys; b,h=(json.load(open(f"{sys.argv[1]}/preimport-{a}.json")) for a in ("base","head")); print("added modules:", sorted(set(h["module_set"])-set(b["module_set"])), "| removed:", sorted(set(b["module_set"])-set(h["module_set"])))' "$EV"
git status --short Tests/Performance backlog/decisions/097-boot-budget-ratchets.md
EOF
```

Expected: `added modules: ['tldw_chatbook.UI.Persona_Modules.roleplay_frame_state'] | removed: []`, `raised: MAX_PASS_ADDED_MODULES 557 -> 558 | ledger row: written` (idempotent), `1 passed`, and ` M` only where the final LOC moved the row's text or the snapshot.

- [ ] **Step 5: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
git status --short
git add scripts/ui_pr_gate_census.txt scripts/check_ui_pr_gate_census.py .github/workflows/derived-artifacts.yml Tests/Performance/test_screen_preimport_payload_budget.py backlog/decisions/097-boot-budget-ratchets.md Tests/Performance/boot_budget_snapshots/preimport_payload.json
git commit -m "ci: gate the Roleplay frame B1 tests on pull requests" -m "Four fast files are appended to the UI Fast Lane census (floor = entry count; census order and every earlier file's shard unchanged): B1's three and TASK-34400's test_roleplay_hostile_text_surfaces.py, which no lane ran before. The four bootstrap-profile Roleplay suites, TASK-34400's hostile-name file among them, join the PR Fast Lane's admission-sensitive step. Final paired pre-import measurement re-checked." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: only those paths modified before the commit (files unchanged by Step 4 are simply not staged), then the commit line.

---

### Task 12: Final gates, paired arms, the arrival cost, the live check, owner screenshots, and the PR body

**Files:**
- Create (evidence only, never committed): `$EV/plug/b1_tour_count.py`, `$EV/uiready-{base,head}-{1,2,3}.txt`, `$EV/gates-preimport-{base,head}.out`, `$EV/arrival_probe.py`, `$EV/arrival.txt`, `$EV/poll_probe.py`, `$EV/poll.txt`, `$EV/timer-sites-{base,head}.txt`, `$EV/live_b1.sh`, `$EV/check_b1_captures.py`, `$EV/live.log`, `$EV/harness-state/` (B1-private harness masters, runs and captures), `$EV/screens/*.png`, `$EV/owner-screenshots.txt`, `$EV/outcomes.txt` (what the owner's rulings changed, read from both arms' files), `$EV/pr-body.md`
- Create (committed in Task 13): `Docs/superpowers/plans/2026-10-03-roleplay-b1-captures/*.txt` (32 files)

**Interfaces:**
- Consumes: everything above; Task 0's `$EV/paired.sh`, `$EV/preimport_measure.sh`, `$EV/preimport-{base,head}.json`, `$EV/suites-*.txt`, `$EV/mutations.md`, `$EV/stack-cut.txt`, `$EV/base-sha.txt`, `$EV/b1-task-id.txt`.
- Produces: `$EV/pr-body.md` for the controller; the capture set and `$EV/owner-screenshots.txt` for Task 13.

This task records evidence; every gate below must hold before Task 13 ticks an acceptance criterion. "No new failures" is judged by `paired.sh` (failure-set diff, `recovery=` in single digits on both arms), never by a raw pass count.

- [ ] **Step 1: Confirm the base arm is still the stack base**

B0 is on dev, so the stack base is the dev SHA B1 was last moved onto (`$EV/stack-cut.txt` = `$EV/base-sha.txt` = the base arm's `HEAD` = `git merge-base HEAD origin/dev`), and every commit after it is B1's.

```bash
bash <<'EOF'
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b1; BASE=$MAIN/.worktrees/roleplay-b1-base; EV=$MAIN/.worktrees/.b1-evidence
git -C "$MAIN" fetch origin --prune --quiet
CUT=$(cat "$EV/stack-cut.txt"); B=$(cat "$EV/base-sha.txt")
echo "recorded base: $(git -C "$WT" rev-parse --short "$CUT") ; dev: $(git -C "$WT" rev-parse --short origin/dev)"
[ "$CUT" = "$B" ] && echo "stack-cut and base-sha agree" || echo "RE-ANCHOR: stack-cut.txt and base-sha.txt differ"
[ "$CUT" = "$(git -C "$BASE" rev-parse HEAD)" ] && echo "base arm matches" || echo "RE-ANCHOR the base arm"
git -C "$WT" merge-base --is-ancestor "$CUT" HEAD && echo "B1 sits on the recorded base" || echo "B1 is NOT on the recorded base"
[ "$CUT" = "$(git -C "$WT" merge-base HEAD origin/dev)" ] && echo "base is dev's merge-base: ok" || echo "REBASE: dev moved, or the recorded base is stale"
echo "commits after the base: $(git -C "$WT" rev-list --count "$CUT..HEAD"), not B1's: $(git -C "$WT" log --format=%s "$CUT..HEAD" | grep -cvE '\(B1| B1 |TASK-33910\.2' || true)"
EOF
```

Expected: `stack-cut and base-sha agree`, `base arm matches`, `B1 sits on the recorded base`, `base is dev's merge-base: ok` and `not B1's: 0`. What each other line means:
- `REBASE: dev moved` (normal once dev advances; required before the PR merges, since dev is strict): run the block below.
- `RE-ANCHOR` (the files or the base arm disagree, e.g. a hand-resolved move that never recorded): `"$EV/restack.sh" --record`, then re-run this step.
- `B1 is NOT on the recorded base`, or `not B1's` above 0: the recorded cut is stale. If every commit in `$(git -C $WT merge-base HEAD origin/dev)..HEAD` is B1's (the same `grep -cvE` prints 0 for that range), run `"$EV/restack.sh" --record` and re-run this step; otherwise STOP and report (B1 carries someone else's commits).

```bash
bash <<'EOF'
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b1; EV=$MAIN/.worktrees/.b1-evidence
"$EV/restack.sh"
cd "$WT"
"$MAIN/.venv/bin/python" tldw_chatbook/css/build_css.py > /dev/null
"$MAIN/.venv/bin/python" tldw_chatbook/css/check_bundle_sync.py 2>&1 | tail -2
git status --short | head
EOF
```

A B1 cut point older than `8d502ba250` will rebase over #3028 (merged 2026-10-08: the census, the workflow's admission step and `_console.tcss`), so the re-run of Task 11 Steps 2-4 below is expected there, not optional. A conflicting rebase stops at the conflict (`set -e`): resolve it by the rebase protocol (Global Constraints: the upstream side of every generated or measured file; `_agentic_terminal.tcss` by re-running Task 5 Step 4's deletion script), `git -C … rebase --continue`, `"$EV/restack.sh" --record`, and run the block's remaining lines by hand. Then re-run `$EV/ratchet_rows.py` (Task 6 Step 9; idempotent) and Task 11 Steps 2-4, commit what they rewrote (message `chore: re-measure B1 after moving onto the new stack base`, with the `Co-Authored-By` line), push with `--force-with-lease` (Global Constraints, "Push and PR base"), and continue with Step 2 here. Never hand-merge a generated sheet, a snapshot, the pre-import constants, the ADR-097 row, the PS ratchet row, the census or the workflow's admission-sensitive step.

- [ ] **Step 2: Census and byte gates on both arms, same session**

The broad-selector and destination-tour tests are private-profile tests: run as their own child process (the `TLDW_TEST_PRIVATE_PROFILE_NODE` form, as B0 did) so a plugin can reach them; `b1_tour_count.py` makes the tour test print its count (it fails by design: read the number). The UI-ready census is flaky by one module on both arms (Global Constraints), so it is measured as module sets over three boots per arm.

```bash
bash <<'OUTER'
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b1-evidence
mkdir -p "$EV/plug" "$EV/tmp"
cat > "$EV/plug/b1_tour_count.py" <<'EOF'
"""B1 evidence plugin: make the destination-tour CSS test print its source count.

It sets the test module's CSS_SOURCE_SOFT_LIMIT to 0, so the assertion
message carries the measured count and the test FAILS BY DESIGN: read the
number, never the verdict. It edits no test file.
"""


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        module = getattr(item, "module", None)
        if module is not None and hasattr(module, "CSS_SOURCE_SOFT_LIMIT"):
            module.CSS_SOURCE_SOFT_LIMIT = 0
EOF
NODE_BROAD="Tests/Performance/test_textual_css_fastpath.py::test_ancestor_scoped_bare_type_rule_count_is_a_ratchet"
NODE_TOUR="Tests/Performance/test_ui_latency_guardrails.py::test_destination_tour_css_sources_stay_below_parse_cache"
for ARM in base head; do
  if [ "$ARM" = base ]; then T=$MAIN/.worktrees/roleplay-b1-base; else T=$MAIN/.worktrees/roleplay-b1; fi
  P="$EV/profile-gates-$ARM"; rm -rf "$P"; mkdir -p "$P/home" "$P/config" "$P/data"
  cd "$T"
  BOOT=$(HOME="$P/home" XDG_CONFIG_HOME="$P/config" XDG_DATA_HOME="$P/data" TLDW_CONFIG_PATH="$P/config/config.toml" PYTHONPATH="$T" "$PY" -c 'from Tests.Performance.test_boot_css_byte_budget import _boot_parsed_css_census as c; print(sum(c().values()))' 2>/dev/null | tail -1)
  BROAD=$(env TLDW_TEST_PRIVATE_PROFILE_NODE="$NODE_BROAD" TLDW_TEST_CONFIG_ROOT="$P" HOME="$P/home" XDG_CONFIG_HOME="$P/config" XDG_DATA_HOME="$P/data" TLDW_CONFIG_PATH="$P/config/config.toml" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest "$NODE_BROAD" -p pytest_asyncio.plugin -p pytest_timeout -p no:cacheprovider -s -q 2>&1 | grep -o '\[census\] total=[0-9]*' | tail -1)
  TOUR=$(env TLDW_TEST_PRIVATE_PROFILE_NODE="$NODE_TOUR" TLDW_TEST_CONFIG_ROOT="$P" HOME="$P/home" XDG_CONFIG_HOME="$P/config" XDG_DATA_HOME="$P/data" TLDW_CONFIG_PATH="$P/config/config.toml" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$T:$EV/plug" "$PY" -m pytest "$NODE_TOUR" -p pytest_asyncio.plugin -p pytest_timeout -p b1_tour_count -p no:cacheprovider -q 2>&1 | grep -oE 'CSS sources after a full tour: [0-9]+' | tail -1)
  WEIGHT=$("$PY" -m pytest Tests/Performance/test_app_import_weight.py::test_app_import_own_module_count_stays_at_the_post_diet_size -p no:cacheprovider -q -s 2>&1 | grep -o 'boot-import-weight: [0-9]*/[0-9]*' | tail -1)
  "$PY" -m pytest Tests/Performance/test_screen_preimport_payload_budget.py -p no:cacheprovider -q -s > "$EV/gates-preimport-$ARM.out" 2>&1
  PRE=$(grep -o 'PREIMPORT PAYLOAD CENSUS: [0-9]* modules / [0-9]* LOC' "$EV/gates-preimport-$ARM.out" | tail -1)
  ROUTE=$(grep -E '^\s+ccp\s+[0-9]+ mods' "$EV/gates-preimport-$ARM.out" | tail -1 | tr -s ' ' | sed 's/^ //')
  PREGUARD=$(grep -oE '[0-9]+ (passed|failed)' "$EV/gates-preimport-$ARM.out" | tail -1)
  CLOSURE=$("$PY" -m pytest Tests/Packaging/test_config_import_closure.py -p no:cacheprovider -q 2>&1 | grep -oE '[0-9]+ (passed|failed)' | tail -1)
  for N in 1 2 3; do
    HOME="$P/home" XDG_CONFIG_HOME="$P/config" XDG_DATA_HOME="$P/data" TLDW_CONFIG_PATH="$P/config/config.toml" PYTHONPATH="$T" "$PY" - "$EV/tmp" > "$EV/uiready-$ARM-$N.txt" 2>/dev/null <<'PY'
import sys
import tempfile
from pathlib import Path

from Tests.Performance.test_ui_ready_module_census import _boot_and_census

print("\n".join(sorted(_boot_and_census(Path(tempfile.mkdtemp(dir=sys.argv[1]))))))
PY
  done
  UIREADY="ui-ready boots: $(wc -l < "$EV/uiready-$ARM-1.txt" | tr -d ' ')/$(wc -l < "$EV/uiready-$ARM-2.txt" | tr -d ' ')/$(wc -l < "$EV/uiready-$ARM-3.txt" | tr -d ' ')"
  echo "$ARM | boot-css=$BOOT | $BROAD | $TOUR | $UIREADY | $WEIGHT | $PRE | route: $ROUTE | preimport guard: $PREGUARD | config-closure: $CLOSURE" | tee -a "$EV/gates.txt"
done
"$PY" - "$EV" <<'PY' | tee -a "$EV/gates.txt"
import sys
from pathlib import Path

ev = Path(sys.argv[1])


def resident(arm: str) -> set[str]:
    boots = [set((ev / f"uiready-{arm}-{n}.txt").read_text().split()) for n in (1, 2, 3)]
    return set().union(*boots)


base, head = resident("base"), resident("head")
print("ui-ready: resident on head, never on base:", sorted(head - base))
print("ui-ready: resident on base, never on head:", sorted(base - head))
PY
OUTER
```

Expected shape (values are indicative: every field was measured in the dry run at `8d502ba250` with these same commands, and again in the fix round there; the gate is the base-to-head delta, never these absolute values):

```
base | boot-css=608077 | [census] total=274 | CSS sources after a full tour: 41 | ui-ready boots: 1033/1033/1033 | boot-import-weight: 681/686 | PREIMPORT PAYLOAD CENSUS: 557 modules / 416217 LOC | route: ccp 62 mods 54812 LOC | preimport guard: 1 passed | config-closure: 1 passed
head | boot-css=607862 | [census] total=274 | CSS sources after a full tour: 42 | ui-ready boots: 1033/1033/1033 | boot-import-weight: 681/686 | PREIMPORT PAYLOAD CENSUS: 558 modules / 416596 LOC | route: ccp 63 mods 55191 LOC | preimport guard: 1 passed | config-closure: 1 passed
ui-ready: resident on head, never on base: []
```

(Each UI-ready boot read 1033 or 1034 on either arm while planning at `af13839740`: `tldw_chatbook.DB.character_conversation_search` came and went; at `8c4dfe59a2`, `a793acbef5` and `8d502ba250` all six boots read 1033.) The gate, line by line:
- boot CSS: head ≤ base (−215 B expected: 608,077 → 607,862 in the dry run on `8d502ba250`, 607,943 → 607,728 on `a793acbef5`), and head < 608,090 (dev's base sits 13 B under it at `8d502ba250`; B1 only widens that);
- broad selectors: head ≤ base (both arms read 274 = `MAX_ANCESTOR_SCOPED_BARE_TYPE_RULES` in review: zero headroom, so any counted rule B1 added would turn the required guard red);
- CSS sources after the tour: head ≤ base + 1, and head < 56;
- UI-ready: `resident on head, never on base` is `[]`; boot import weight equal;
- pre-import: only `tldw_chatbook.UI.Persona_Modules.roleplay_frame_state` added (Task 11 Step 4 printed the set), its LOC within the owner's bound, `preimport guard: 1 passed` on both arms;
- config closure passes on both arms.

Any other delta is a FAIL: stop and find the cause. A missing field (an empty `$BROAD`, `$TOUR` or `$PRE`) means that command errored: re-run it alone without the `grep` and read why.

- [ ] **Step 3: Module-size ratchet rows, the first-visit arrival cost and the poll's per-tick cost (spec §2.13)**

```bash
bash <<'EOF'
set -u
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
PS=tldw_chatbook/UI/Screens/personas_screen.py; FS=tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py
echo "PS lines $(wc -l < "$PS" | tr -d ' ') ; row $(grep -o "\"$PS\": [0-9]*" Tests/Architecture/test_module_size_ratchet.py)"
echo "FS lines $(wc -l < "$FS" | tr -d ' ') ; row $(grep -o "\"$FS\": [0-9]*" Tests/Architecture/test_module_size_ratchet.py)"
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Architecture/test_module_size_ratchet.py -k "personas_screen or roleplay_frame_state" -p no:cacheprovider -q 2>&1 | grep -E '[0-9]+ (passed|failed|error)' | tee -a /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/gates.txt
EOF
```

Expected: `PS lines 16525 ; row "tldw_chatbook/UI/Screens/personas_screen.py": 16525` on `8d502ba250`, `a793acbef5` and `8c4dfe59a2` (other values if #2862 or another PS change merged first; lines and row always equal, the row having been set to the measurement, up or down, by `ratchet_rows.py`), `FS lines 369 ; row …: 369`, and `4 passed` (each row's budget and slack checks).

The lazy sheet's parse lands on the first Ctrl+4, so its cost is measured (spec §2.13 "Lazy CSS" row; evidence, not a CI threshold): a cold process per sample, the real `TldwCli` on Home with twelve stub characters, the seconds from the key press to Roleplay's first list item and the time spent in `_ensure_screen_owned_css`, ten samples per arm interleaved ABBA.

```bash
bash <<'OUTER'
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b1-evidence
cat > "$EV/arrival_probe.py" <<'EOF'
"""B1 evidence: the first Ctrl+4 visit on the real app (spec 2.13, "Lazy CSS" row).

Run from an arm's worktree root with PYTHONPATH set to that root. It boots a
real TldwCli on Home with the splash off and twelve stub characters, presses
Ctrl+4 once, and prints the seconds from the key press to Roleplay's first list
item, plus the time spent inside _ensure_screen_owned_css (about 0 on the base
arm, which has no Roleplay sheet), and whether the visit loaded
UI.Navigation.character_conversation_navigation (B1's first header paint reads
the draft aggregate, whose snapshot type lives there and is imported lazily, so
the pre-import census cannot see it). One cold process per sample.
"""

import asyncio
import sys
import time

import pytest

import tldw_chatbook.app  # noqa: F401  -- the app module first, as in the test suites
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as handler
from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_personas_dictionaries import patch_character_paging

RECORDS = [{"id": i, "name": f"Character {i:02d}", "version": 1} for i in range(1, 13)]


def _settings(section, key=None, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


async def _visit() -> tuple[float, float, bool]:
    app = _build_test_app(configured_default="home")
    spent: list[float] = []
    original = app._ensure_screen_owned_css

    def timed(route):
        start = time.perf_counter()
        try:
            return original(route)
        finally:
            spent.append(time.perf_counter() - start)

    app._ensure_screen_owned_css = timed
    with patch_app_global("get_cli_setting", side_effect=_settings):
        async with app.run_test(size=(120, 36)) as pilot:
            deadline = time.monotonic() + 120
            while not (
                getattr(app, "_initial_screen_pushed", False)
                and type(app.screen).__name__ == "HomeScreen"
            ):
                if time.monotonic() > deadline:
                    sys.exit("Home never arrived")
                await pilot.pause(0.02)
            await pilot.pause(0.5)
            spent.clear()
            start = time.perf_counter()
            await pilot.press("ctrl+4")
            while not (
                type(app.screen).__name__ == "PersonasScreen"
                and app.screen.query("#personas-library-rows > ListItem")
            ):
                if time.monotonic() > deadline:
                    sys.exit("Roleplay never showed its list")
                await asyncio.sleep(0.005)
            arrival = time.perf_counter() - start
    nav = "tldw_chatbook.UI.Navigation.character_conversation_navigation"
    return arrival, sum(spent), nav in sys.modules


with pytest.MonkeyPatch.context() as mp:
    mp.setattr(handler, "fetch_all_characters", lambda: [dict(r) for r in RECORDS])
    mp.setattr(
        handler,
        "fetch_character_by_id",
        lambda cid: next((dict(r) for r in RECORDS if str(r["id"]) == str(cid)), None),
    )
    patch_character_paging(mp)
    arrival_s, css_s, nav_loaded = asyncio.run(_visit())
print(f"arrival_s={arrival_s:.4f} css_load_s={css_s:.4f} nav_module={nav_loaded}")
EOF
: > "$EV/arrival.txt"
for CYCLE in 1 2 3 4 5; do
  for ARM in base head head base; do
    if [ "$ARM" = base ]; then T=$MAIN/.worktrees/roleplay-b1-base; else T=$MAIN/.worktrees/roleplay-b1; fi
    P="$EV/profile-arrival"; rm -rf "$P"; mkdir -p "$P/home" "$P/config" "$P/data"
    LINE=$(cd "$T" && HOME="$P/home" XDG_CONFIG_HOME="$P/config" XDG_DATA_HOME="$P/data" TLDW_CONFIG_PATH="$P/config/config.toml" PYTHONPATH="$T" "$PY" "$EV/arrival_probe.py" 2>/dev/null | tail -1)
    echo "$ARM $LINE" >> "$EV/arrival.txt"
  done
done
"$PY" - "$EV/arrival.txt" <<'EOF' | tee -a "$EV/gates.txt"
import statistics
import sys

rows = [line.split() for line in open(sys.argv[1], encoding="utf-8") if "arrival_s=" in line]
for arm in ("base", "head"):
    arrival = [float(row[1].split("=")[1]) for row in rows if row[0] == arm]
    css = [float(row[2].split("=")[1]) for row in rows if row[0] == arm]
    nav = {row[3] for row in rows if row[0] == arm}
    print(
        f"arrival {arm}: n={len(arrival)} median {statistics.median(arrival):.3f} s"
        f" | css load median {statistics.median(css) * 1000:.1f} ms | {', '.join(sorted(nav))}"
    )
EOF
OUTER
```

Expected: `n=10` on both arms (a smaller `n` means a sample died: run the probe alone, without `2>/dev/null`, and read why). Measured while planning (two samples per arm on `bd41347b65`): base 2.46-2.60 s with a css load of 0.0 ms, head 2.19-2.39 s with a css load of about 26 ms (Pilot arrival includes the screen's first data load; the absolute value is not the live app's). The gate: head's median arrival ≤ base's + 0.25 s, and head's css-load median < 250 ms (the spec's per-transition stall budget, RC-6). Otherwise STOP and report both medians to the controller: the owner decides (ADR-097 order: defer, shed, exception). The `nav_module=` column records whether the first visit loaded `UI.Navigation.character_conversation_navigation` (measured in review: about 5 ms and one module, loaded on head by the header's first paint through the draft aggregate; the pre-import census cannot see it because the import is lazy): copy both arms' values into the PR body's budget ledger.

The readiness poll's per-tick cost (spec §2.13; review: the first draft's poll walked the DOM on every tick). B1's poll gathers the header inputs on every 0.25 s tick, in every mode and with nothing selected, where the old poll returned early; this measures it on both arms: a real `TldwCli` straight into Roleplay with 12 and with 300 stub characters (the first is auto-selected), the poll's own timer stopped, 200 direct calls timed in Characters and again in Lore, two cold processes per arm and size.

```bash
bash <<'OUTER'
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b1-evidence
cat > "$EV/poll_probe.py" <<'EOF'
"""B1 evidence: the 0.25 s readiness poll's per-tick cost (Task 12 Step 3).

Usage (from an arm's worktree root, PYTHONPATH set to it): poll_probe.py <rows>.
Boots a real TldwCli straight into Roleplay with <rows> stub characters, stops
the poll's own timer, and times 200 direct calls of
_poll_console_handoff_readiness in Characters (first row auto-selected), then
in Lore. Prints one line per mode with the median and p90 in microseconds.
"""

import asyncio
import statistics
import sys
import time

import pytest

import tldw_chatbook.app  # noqa: F401  -- the app module first, as in the test suites
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as handler
from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_personas_dictionaries import patch_character_paging

ROWS = int(sys.argv[1])
RECORDS = [{"id": i, "name": f"Character {i:03d}", "version": 1} for i in range(1, ROWS + 1)]


def _settings(section, key=None, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


def _ticks(screen, count: int = 200) -> tuple[float, float]:
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        screen._poll_console_handoff_readiness()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    return statistics.median(samples), samples[int(len(samples) * 0.9)]


async def _run() -> dict[str, tuple[float, float]]:
    app = _build_test_app(configured_default="personas")
    with patch_app_global("get_cli_setting", side_effect=_settings):
        async with app.run_test(size=(160, 45)) as pilot:
            deadline = time.monotonic() + 120
            while not (
                type(app.screen).__name__ == "PersonasScreen"
                and app.screen.query("#personas-library-rows > ListItem")
            ):
                if time.monotonic() > deadline:
                    sys.exit("Roleplay never showed its list")
                await pilot.pause(0.02)
            await pilot.pause(1.0)
            screen = app.screen
            screen._console_readiness_poll_timer.stop()
            results = {"characters": _ticks(screen)}
            await pilot.click("#personas-mode-lore")
            await pilot.pause(1.0)
            results["lore"] = _ticks(screen)
    return results


with pytest.MonkeyPatch.context() as mp:
    mp.setattr(handler, "fetch_all_characters", lambda: [dict(r) for r in RECORDS])
    mp.setattr(
        handler,
        "fetch_character_by_id",
        lambda cid: next((dict(r) for r in RECORDS if str(r["id"]) == str(cid)), None),
    )
    patch_character_paging(mp)
    measured = asyncio.run(_run())
for mode, (median, p90) in measured.items():
    print(f"rows={ROWS} mode={mode} median_us={median:.0f} p90_us={p90:.0f}")
EOF
: > "$EV/poll.txt"
for ROWS in 12 300; do
  for ARM in base head head base; do
    if [ "$ARM" = base ]; then T=$MAIN/.worktrees/roleplay-b1-base; else T=$MAIN/.worktrees/roleplay-b1; fi
    P="$EV/profile-poll"; rm -rf "$P"; mkdir -p "$P/home" "$P/config" "$P/data"
    (cd "$T" && HOME="$P/home" XDG_CONFIG_HOME="$P/config" XDG_DATA_HOME="$P/data" TLDW_CONFIG_PATH="$P/config/config.toml" PYTHONPATH="$T" "$PY" "$EV/poll_probe.py" "$ROWS" 2>/dev/null | grep '^rows=' | sed "s/^/$ARM /") >> "$EV/poll.txt"
  done
done
"$PY" - "$EV/poll.txt" <<'EOF' | tee -a "$EV/gates.txt"
import statistics
import sys

cells: dict[tuple[str, str, str], list[float]] = {}
for line in open(sys.argv[1], encoding="utf-8"):
    arm, rows, mode, median, _p90 = line.split()
    cells.setdefault((rows, mode, arm), []).append(float(median.split("=")[1]))
for rows, mode in sorted({(r, m) for r, m, _a in cells}):
    base = statistics.median(cells.get((rows, mode, "base"), [float("nan")]))
    head = statistics.median(cells.get((rows, mode, "head"), [float("nan")]))
    verdict = "ok" if head <= base + 1000 else "OVER"
    print(f"poll {rows} {mode}: base {base:.0f} us | head {head:.0f} us | {verdict} (head <= base + 1000 us)")
EOF
OUTER
```

Expected: four `poll …` lines (`rows=12` and `rows=300`, each in `mode=characters` and `mode=lore`), each `ok`, from `$EV/poll.txt` holding 16 sample lines (a missing cell prints `nan`: run the probe alone without `2>/dev/null` and read why). Measured in review with the aggregate fix applied: about 115 µs a tick on head against 50-344 µs for the base arm's poll (the unfixed draft took 3.5-4.2 ms). An `OVER` line: STOP and report both medians to the controller. The PR body quotes these lines instead of the first draft's "compares one more value".

- [ ] **Step 4: Paired arms — every suite group**

```bash
bash <<'EOF'
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence
"$EV/paired.sh" roleplay "$EV/suites-roleplay.txt" -n 6 | tee -a "$EV/gates.txt"
"$EV/paired.sh" shell "$EV/suites-shell.txt" -n 6 | tee -a "$EV/gates.txt"
"$EV/paired.sh" css "$EV/suites-css.txt" -n 6 | tee -a "$EV/gates.txt"
PAIRED_PLUGIN=0 "$EV/paired.sh" perf "$EV/suites-perf.txt" | tee -a "$EV/gates.txt"
EOF
```

Expected: for each of the four groups, `recovery=` in single digits on both arms (the perf group runs with `plugin=off`, as CI runs it; for `shell`, see its note below) and nothing between `new failures on head [<group>] (must be empty):` and `(end of new failures [<group>])`. Pre-existing failures appear on both arms and are not B1's; the failure-set diff decides, and these names are indicative only (dev moves under them): in the `css` group the ones Task 5 Step 5 lists plus `test_timer_path_static_update_inventory.py`'s three (red on dev; its floor is the block below; 17 on each arm at `a793acbef5` and again at `8d502ba250`, the same 17, a tenth ratchet row `Chat/console_interrupt_rounds.py` among them); in `roleplay`, `test_actor_pack_creation_workflow.py::test_navigation_signals_and_drains_pack_creation_before_continuing` (failed 3/3 alone on BOTH arms in review; under `-n 6` it passes by chance on either arm, so it can show as "new": re-run it alone on both arms before calling it), one `test_personas_inspector_pane.py` test (both arms in the `83c264f286` dry run), eight `test_personas_workbench.py` tests (the six `TestPersonaHumanIdentityRemoval` cases ERROR at setup with `RecoveryRequired: raw_participant_not_installed`, plus `test_floating_buddy_close_refreshes_active_personas_inspector` and `test_resize_sync_skips_work_when_compact_state_is_unchanged`; both arms in the `8c4dfe59a2` dry run, plugin on), and the quit-guard file: all nine of its tests are load-sensitive on dev, on BOTH arms. Run alone with the plugin at `8d502ba250` it failed 9/9 on base and 9/9 on head in one session (load about 33), 4/9 on base and 1/9 on head with the arms run side by side (load about 29), and 3/9 on base and 2/9 on head run one arm after the other (load 18-22), a different subset each time. Each of the ten failures in the last two sessions is the setup helper's `AssertionError: timed out waiting for the character editor to load the saved card`, before any Ctrl+Q or guard step; in the first two sessions every failing test also logged one `app.__init__` line `ChaChaNotesDB (CharactersRAGDB) instance not found`. Under `-n 6` five of them showed as "new" on head there by chance. So any quit-guard test can show as new: re-run the whole file alone on both arms, read each failure's message (one past that setup wait is B1's to explain: Step 4b changes the guard), and apply the flake rule below; in `shell`, `test_destination_header_compact_floor.py::test_destination_header_is_compact_at_the_80x24_floor` is intermittent on dev (a Schedules worker `NoMatches`; alone at `8d502ba250` it failed 2 of 3 runs on base and passed 3/3 on head in one session, and passed 13/13 on both arms in another). Two `test_console_workspace_context_rail.py` tests are not settled: `test_expanded_chats_section_capped_row_marker_surfaces_on_header` and `test_console_workspace_selector_is_compact_plain_status_row` showed as "new" on head under `-n 6` at `8d502ba250` and passed 3/3 alone on both arms there; in a second session, run alone 13 times per arm (interleaved), they failed only on head, the first twice (`assert 'Chats' == 'Chats ●'`: the header was read one `pilot.pause()` after `sync_state`, before its recompose landed) and the second once (`NoMatches` on `#console-active-workspace`, queried while the section recomposed), against no failure on base; the first then ran alone 20 more times per arm with 0 failures on base and 0 on head. Both are races in the test, B1 changes no Console code, and the one thing a Console harness sees differently is that `APP_STYLESHEETS` now carries B1's split sheet (5,604 stylesheet rules against 5,588); a paired probe of the first race (20 runs per arm, interleaved) saw the marker after the first pause in every run on both arms, in a median 199 ms on base and 168 ms on head. By the flake rule below, a head-only failure still counts: if either shows as new, run the rule to its end on both arms, and if head fails while base stays clean, STOP and report it with these numbers. The `shell` group's `recovery=` is not single digits on dev and that is not the environment: `Tests/UI/test_settings_configuration_hub.py`'s advanced-config, diagnostics, schedules-gate and fresh-config appearance tests build their own fresh config and raise `RecoveryRequired: raw_source_selection_changed` under the bootstrap plugin on both arms (`recovery=14` on both arms at `8d502ba250`, every line from that file; run alone on base, those tests gave `13 failed, 8 passed` with 11 recovery lines). Accept a `shell` count above 9 only when it is equal on both arms and every `RecoveryRequired` line traces to that file (`grep -n 'RecoveryRequired:' "$EV/shell-<arm>.log"` and read the test above each); anything else is the environment: stop and fix it; in `perf`, `test_ui_ready_module_census.py` can fail on either arm (its one-module flake; Step 2's set comparison is that gate). Expect a long run: both arms run every group.

Flaky-test rule: re-run each new failure three times on EACH arm (from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1-base` and from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1`: `PYTHONPATH=$EV/plug /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <file> -p b1_bootstrap_all -k <test name> -p no:cacheprovider -q`). If it passes all three on head, it is a load flake under `-n`: record its name and error text in `$EV/gates.txt` and in the PR body (the dry runs saw such: `test_persona_actor_pack_save_pins_selected_portrait_identity` and `test_settings_action_rows_keep_every_button_whole[size1]` in the first, `test_character_tts_preview_create_and_edit_reuse_existing_speech_surfaces` and two `shell` tests in review, each 3/3 green alone on both arms). If it fails on head in a re-run while base passes all three, three runs are not enough to call it: run ten more on each arm. It is a regression only if base passes all ten and head fails again: fix it before continuing. If base fails at least once in those ten, it is a pre-existing flake: record both arms' failure counts (B0's plan review measured a Watchlists test failing 5/10 on base and 6/10 on head after a 3/3 base run that looked clean).

The timer-path census (`css` group) is red on dev, so a failure-set diff cannot see B1 add a clock-reachable `.update(` that defaults to `layout=True` (B1's poll now reaches the header paint on every tick). Its floor compares the unclassified sites themselves, line numbers dropped (B1 moves PS lines):

```bash
bash <<'EOF'
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b1-evidence
NODE=Tests/Architecture/test_timer_path_static_update_inventory.py::test_no_timer_path_update_defaults_to_layout_true
for ARM in base head; do
  if [ "$ARM" = base ]; then T=$MAIN/.worktrees/roleplay-b1-base; else T=$MAIN/.worktrees/roleplay-b1; fi
  (cd "$T" && "$PY" -m pytest "$NODE" -p no:cacheprovider -q --tb=long 2>&1) \
    | grep -oE "[A-Za-z0-9_/.]+\.py:\[[0-9, ]+\] in [A-Za-z0-9_.<>]+ recv=[^ ]+" \
    | sed -E 's/:\[[0-9, ]+\] in / in /' | sort -u > "$EV/timer-sites-$ARM.txt"
  echo "$ARM: $(wc -l < "$EV/timer-sites-$ARM.txt" | tr -d ' ') unclassified timer-path sites"
done | tee -a "$EV/gates.txt"
echo "timer-path sites on head, never on base (must be empty):"
comm -13 "$EV/timer-sites-base.txt" "$EV/timer-sites-head.txt" | tee -a "$EV/gates.txt"
echo "(end of new timer-path sites)"
EOF
```

Expected: the same nonzero count on both arms (71 at `a793acbef5` and `8d502ba250`; 67 at `67fc531047`) and nothing between the two marker lines. A zero count on base means the grep missed the census's message format: read `--tb=long` output by hand before trusting the comparison. A new site is B1's: pass `layout=` explicitly at it (only where the rendered size cannot change) or classify it in `CLASSIFIED_SITES`, then re-run.

Then the head-only suites, in the two invocations CI uses (TASK-32873: the bootstrap-profile files never share a run with the sandboxed fast files; the fast set is the UI Fast Lane's four census files, TASK-34400's widget-level hostile-text file among them):

```bash
bash <<'EOF'
set -u
PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python; EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
FAST="Tests/UI/test_roleplay_frame_state.py Tests/UI/test_workbench_fitted_text.py Tests/UI/test_roleplay_stylesheet.py Tests/UI/test_roleplay_hostile_text_surfaces.py"
MOUNTED="Tests/UI/test_roleplay_frame_harness.py Tests/UI/test_roleplay_header.py Tests/UI/test_roleplay_hostile_names.py Tests/UI/test_roleplay_journeys.py"
"$PY" -m pytest $FAST -p no:cacheprovider -q 2>&1 | grep -E '^(FAILED|ERROR) |[0-9]+ (passed|failed|error)' | tee -a "$EV/gates.txt"
"$PY" -m pytest $FAST --collect-only -q -p no:cacheprovider 2>&1 | grep -E '[0-9]+ tests? collected'
"$PY" -m pytest $MOUNTED -p no:cacheprovider -q -n 6 --timeout=300 2>&1 | grep -E '^(FAILED|ERROR) |[0-9]+ (passed|failed|error)' | tee -a "$EV/gates.txt"
"$PY" -m pytest $MOUNTED --collect-only -q -p no:cacheprovider 2>&1 | grep -E '[0-9]+ tests? collected'
EOF
```

Expected: `125 passed` and `125 tests collected` (28 + 5 + 11 + 81: the UI Fast Lane's four files, Task 11), then `110 passed` and `110 tests collected` (14 + 67 + 27 + 2; the hostile-name file holds TASK-34400's 22 and B1's five), as in the dry runs on `a793acbef5` and `8d502ba250`.

- [ ] **Step 5: Preflight and lint**

```bash
bash <<'EOF'
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b1; EV=$MAIN/.worktrees/.b1-evidence; PY=$MAIN/.venv/bin/python
cd "$WT"
PYTHON=$PY ./scripts/preflight.sh 2>&1 | tail -15 | tee -a "$EV/gates.txt"
CHANGED=$(git diff --name-only "$(cat "$EV/stack-cut.txt")"...HEAD -- '*.py' | grep -v '^tldw_chatbook/app.py$')
"$PY" -m ruff check $CHANGED
# Not format-clean at base (Global Constraints): never formatted by B1.
FORMATTED=$(echo "$CHANGED" | grep -vE '^(tldw_chatbook/css/build_css\.py|Tests/UI/test_css_build_integrity\.py)$')
echo "format-checked files: $(echo "$FORMATTED" | wc -l | tr -d ' ')"
"$PY" -m ruff format --check $FORMATTED
for T in "$MAIN/.worktrees/roleplay-b1-base" "$WT"; do (cd "$T" && "$PY" -m ruff check --output-format concise tldw_chatbook/app.py | grep '^Found'); done
EOF
```

Expected: every preflight check passes (bundle sync, profile-owned path census, production diagnostic inventory, backlog task ids, readable task files, the gated Tests/UI census and the rest; B0's plan lists them). If the Canvas Mermaid asset step cannot download, set `TLDW_CANVAS_MERMAID_INPUT_DIR` to an existing offline cache and re-run. Ruff: `All checks passed!`; `format-checked files: N` and `N files already formatted` with the same N (26 in the dry runs on `8c4dfe59a2`, `a793acbef5` and `8d502ba250`: every changed Python file but `app.py`, `build_css.py` and `test_css_build_integrity.py`, which were not format-clean at base, and whose `ruff format --diff` B1 leaves the same size on both arms); and `Found 96 errors.` on both arms for `app.py` (pre-existing: 94 `E402` and 2 `F401` at `8c4dfe59a2`, the same on both; `Found 96 errors.` on both arms again at `a793acbef5` and `8d502ba250`, and every preflight check passed on head at both).

- [ ] **Step 6: Live check — the harness self-test, B1-private masters, the driver and the checker**

B1 is a visible slice: base and head captures differ by design, so the comparison is mechanical (the checker) plus the owner's eyes (Step 8). The harness lives in this worktree (it reached dev with PR #2957); its state is B1-private (`$EV/harness-state`), so no other session can rebuild or overwrite the masters mid-capture (harness README, "Traps"). The driver covers the three lifecycle entries of spec §5.7.5 (route activation, resize restoration, untouched arrival on the initial tab), the B1 journey (Edit, type, the Unsaved chip, Ctrl+S) and a run with no chat provider (the blocked chip). It types only into the editor's Name input and only once that input shows its focused `┌` border (README hazard: a stray letter elsewhere is a Roleplay hotkey).

```bash
bash <<'OUTER'
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; EV=$MAIN/.worktrees/.b1-evidence
H=$MAIN/.worktrees/roleplay-b1/Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/harness
export HARNESS_STATE=$EV/harness-state
APP_WT=$MAIN/.worktrees/roleplay-b1 "$H/launch.sh" --self-test 2>&1 | tail -3
APP_WT=$MAIN/.worktrees/roleplay-b1-base "$H/make_profiles.sh" 2>&1 | tail -3
cat > "$EV/live_b1.sh" <<'EOF'
#!/usr/bin/env bash
# Usage: live_b1.sh <base|head> <COLSxROWS>
# Roleplay frame B1 live check on the rp-review harness (B1-private HARNESS_STATE):
# route activation, the B1 journey, resize restoration, untouched arrival on the
# initial tab, and a run with no chat provider. Captures land in
# $HARNESS_STATE/captures/b1-<arm>/b1-<state>-<size>.{txt,ansi,png}.
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook
H=$MAIN/.worktrees/roleplay-b1/Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/harness
ARM=$1; SIZE=$2; COLS=${SIZE%x*}; ROWS=${SIZE#*x}
case "$ARM" in
  base) export APP_WT=$MAIN/.worktrees/roleplay-b1-base ;;
  head) export APP_WT=$MAIN/.worktrees/roleplay-b1 ;;
  *) echo "usage: live_b1.sh <base|head> <COLSxROWS>" >&2; exit 2 ;;
esac
export HARNESS_STATE=$MAIN/.worktrees/.b1-evidence/harness-state PNG=1
export CAPTURES=$HARNESS_STATE/captures/b1-$ARM
OTHER=80x24; [ "$SIZE" = 80x24 ] && OTHER=120x36
S="b1${ARM}${COLS}"
# drive_lib.sh sources env.sh (PY; R = the harness dir) and defines shot, clk, has, key, typ, pal.
. "$H/drive_lib.sh" "$S" "$SIZE"
booted() { "$PY" "$R/waitfor.py" "$S" "4 Roleplay" 240 && "$PY" "$R/waitfor.py" "$S" "Ctrl+Q" 30; }
quit() { tmux -L "$S" send-keys C-q; sleep 3; tmux -L "$S" kill-server 2>/dev/null || true; }
to_roleplay() { clk "⌃4 Roleplay"; sleep 4; has "Modes:" || { pal "Switch to Roleplay"; sleep 4; }; }

# 1. Route activation: a fresh run (mock provider ready) on its default tab, then Roleplay.
"$R/launch.sh" "$S" "$COLS" "$ROWS" golden
booted || { echo "  !! boot timed out ($ARM $SIZE)"; quit; exit 1; }
to_roleplay; shot b1-route-arrival
clk "Captain Isolde Varga" L; sleep 3; shot b1-selected
clk "  Edit  " B; sleep 3; shot b1-editing
# 2. The journey: type into the Name input only once it shows the focused border.
POS=$(tmux -L "$S" capture-pane -p | "$PY" -B -c '
import sys
rows = sys.stdin.read().split("\n")
top = next((i for i, line in enumerate(rows) if "Character Editor" in line), None)
if top is not None:
    col = rows[top].index("Character Editor")
    for i in range(top + 1, len(rows)):
        j = rows[i].find("Name", col)
        if j >= 0 and not rows[i][col:j].strip():
            print(i + 1, j + 1)
            break
')
DIRTY=0
if [ -n "$POS" ]; then
  NR=${POS% *}; NC=${POS#* }
  "$R/click.sh" "$S" $((NC + 4)) $((NR + 2)); sleep 0.8
  if tmux -L "$S" capture-pane -p | sed -n "$((NR + 1))p" | grep -q "┌"; then
    key End; typ " II"; sleep 1.5; shot b1-dirty; DIRTY=1
  fi
fi
if [ "$DIRTY" = 1 ]; then
  # 3. Resize restoration: the header refits from its cached inputs.
  tmux -L "$S" resize-window -x "${OTHER%x*}" -y "${OTHER#*x}"; sleep 3; shot b1-dirty-resized
  tmux -L "$S" resize-window -x "$COLS" -y "$ROWS"; sleep 3; shot b1-dirty-restored
  key C-s; sleep 3; shot b1-saved
else
  echo "  !! Name input not focused ($ARM $SIZE): the dirty, resize and saved states were skipped"
fi
clk "Personas  "; sleep 3; shot b1-personas
quit
# 4. Untouched arrival: the same run restarted with Roleplay as its initial tab.
"$PY" "$R/profile_config.py" set "$HARNESS_STATE/runs/$S/config.toml" general default_tab personas > /dev/null
REUSE=1 "$R/launch.sh" "$S" "$COLS" "$ROWS" golden
if booted; then sleep 3; shot b1-initial-arrival; else echo "  !! restart timed out ($ARM $SIZE)"; fi
quit
# 5. No chat provider: a fresh run without the mock fragment.
EXTRA_TOML="" "$R/launch.sh" "$S" "$COLS" "$ROWS" golden
if booted; then to_roleplay; shot b1-noprovider; else echo "  !! boot timed out ($ARM $SIZE)"; fi
quit
EOF
chmod +x "$EV/live_b1.sh"
cat > "$EV/check_b1_captures.py" <<'EOF'
"""Roleplay frame B1: mechanical checks of the live captures (plain .txt).

Usage: check_b1_captures.py <captures dir holding b1-base/ and b1-head/>

The header row is the row just above the purpose line, and the row above it
must be the nav bar's bottom rule, so the header is one row on head whatever
the nav height (ADR-210's compact nav changes it below 35 rows). Prints, per
capture, the purpose line's row on each arm (the shift is the rows B1 gave
back) and the head's header row, and FAILs a capture whose head header breaks
a B1 rule.
"""

import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
SIZES = ("80x24", "120x36", "160x45", "220x55")
STATES = (
    "route-arrival",
    "selected",
    "editing",
    "dirty",
    "dirty-resized",
    "dirty-restored",
    "saved",
    "personas",
    "initial-arrival",
    "noprovider",
)
KINDS = ("Characters", "Personas", "Dictionaries", "Lore")
PURPOSE = (
    "— who the AI plays",
    "— who you play in the chat",
    "— text find/replace rules",
    "— world facts injected on keywords",
)
RULE = set("─━═ ")


def purpose_row(lines: list[str]) -> int | None:
    return next(
        (i for i, line in enumerate(lines) if any(p in line for p in PURPOSE)), None
    )


failed = missing = 0
for size in SIZES:
    for state in STATES:
        name = f"b1-{state}-{size}.txt"
        paths = [ROOT / f"b1-{arm}" / name for arm in ("base", "head")]
        if not all(path.exists() for path in paths):
            missing += 1
            print(f"MISSING {name}: base={paths[0].exists()} head={paths[1].exists()}")
            continue
        base, head = (path.read_text(encoding="utf-8").split("\n") for path in paths)
        pb, ph = purpose_row(base), purpose_row(head)
        problems, header = [], ""
        if ph is None or ph < 2:
            problems.append("no purpose line on head")
        else:
            header = head[ph - 1]
            if not head[ph - 2].strip() or set(head[ph - 2]) - RULE:
                problems.append("the row above the header is not the nav's bottom rule")
            if "Roleplay" not in header:
                problems.append("no title")
            if "Ready" in header or "Blocked" in header:
                problems.append("a readiness badge word")
            # The restarted run may restore its last kind; every other state's is known.
            kinds = {"personas": ("Personas",), "initial-arrival": KINDS}.get(
                state, ("Characters",)
            )
            if not any(kind in header for kind in kinds):
                problems.append(f"the kind ({' or '.join(kinds)}) is missing")
            if (state == "noprovider") != ("No chat provider" in header):
                problems.append("the blocked chip is wrong for this run's provider")
            if not header.rstrip().endswith("Local"):
                problems.append("the status is not Local")
            dirty = state in ("dirty", "dirty-resized", "dirty-restored")
            if dirty != ("Unsaved" in header):
                problems.append(
                    "the unsaved chip is " + ("missing" if dirty else "stray")
                )
            if (
                state in ("editing", "dirty", "dirty-restored")
                and "editing" not in header
            ):
                problems.append("the editing word is missing")
        shift = None if pb is None or ph is None else pb - ph
        failed += bool(problems)
        print(
            f"{'FAIL' if problems else 'ok  '} {name}: purpose row "
            f"{None if pb is None else pb + 1} -> {None if ph is None else ph + 1} "
            f"(shift {shift}) | {header.strip()!r}"
            + (f" | {'; '.join(problems)}" if problems else "")
        )
total = len(SIZES) * len(STATES)
print(
    f"captures: {total - missing} compared, {missing} missing, {failed} failing a head check"
)
EOF
echo "live helpers written"
OUTER
```

Expected: `SELF-TEST: PASS`; `make_profiles.sh` finishing with the `golden` master built under `$EV/harness-state` (masters that already exist are B1's own from an earlier run: keep them); `live helpers written`. A self-test FAIL caused by another session using the real profile is spurious (README "Traps"): read the diff before blaming the harness.

- [ ] **Step 7: Live captures — both arms at 80x24, 120x36, 160x45 and 220x55, then the checks**

```bash
bash <<'EOF'
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; EV=$MAIN/.worktrees/.b1-evidence
: > "$EV/live.log"
for ARM in base head; do
  for SIZE in 80x24 120x36 160x45 220x55; do
    echo "== $ARM $SIZE" >> "$EV/live.log"
    "$EV/live_b1.sh" "$ARM" "$SIZE" >> "$EV/live.log" 2>&1
  done
done
grep -E '^==|!!' "$EV/live.log"
"$MAIN/.venv/bin/python" "$EV/check_b1_captures.py" "$EV/harness-state/captures" | tee -a "$EV/gates.txt"
EOF
```

Expected: eight `==` lines with no `!!` line under them, then one `ok` line per capture and `captures: 80 compared, 0 missing, 0 failing a head check`. The `shift` column is 4 at 36, 45 and 55 rows and 3 at 24 rows (the five-row and four-row headers became one row; the `dirty-resized` captures shift by the other size's amount). A `!! no match:` or `!! Name input not focused` line is a driver miss, not a finding: relaunch that arm and size (`"$EV/live_b1.sh" <arm> <size>`, runs are cheap) until it is gone (at 80x24 the rail shows about one row, so `Captain Isolde Varga` may be off the list: the run then edits the auto-selected first character, which every check accepts). A `FAIL` line on head is a defect: stop and debug, except a FAIL whose only problem is the nav-rule check, which means the nav draws its bottom row differently at that size: read the capture, and record it in `$EV/gates.txt` if the header itself is one row.

Copy the approval set (spec §5.4 item 6: the `.txt` captures the slice selects, beside its plan; PNGs go to the owner, not into the tree):

```bash
bash <<'EOF'
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; EV=$MAIN/.worktrees/.b1-evidence; C=$EV/harness-state/captures
D=$MAIN/.worktrees/roleplay-b1/Docs/superpowers/plans/2026-10-03-roleplay-b1-captures
mkdir -p "$D" "$EV/screens"
for ARM in base head; do
  for SIZE in 80x24 120x36 160x45 220x55; do
    for STATE in initial-arrival editing dirty noprovider; do
      N=b1-$STATE-$SIZE
      cp "$C/b1-$ARM/$N.txt" "$D/$ARM-$N.txt"
      cp "$C/b1-$ARM/$N.png" "$EV/screens/$ARM-$N.png"
    done
  done
done
echo "captures: $(ls "$D" | wc -l | tr -d ' ') pngs: $(ls "$EV/screens" | wc -l | tr -d ' ')"
EOF
```

Expected: `captures: 32 pngs: 32`. A `cp: … No such file` line means a capture was skipped (a driver miss, or `png failed` from playwright: `$MAIN/.venv/bin/python -m playwright install chromium`): re-run that arm and size first.

- [ ] **Step 8: STOP — owner screenshot approval (spec §5.4 item 6, ADR-007)**

B1 changes what every Roleplay visit looks like. Report to the controller with the sixteen base/head PNG pairs in `$EV/screens/` (`base-b1-<state>-<size>.png` beside `head-b1-<state>-<size>.png`: untouched arrival, editing, the Unsaved chip, the no-provider chip, at 80x24, 120x36, 160x45 and 220x55) and Step 7's checker lines, and wait. Put one more question to the owner in the same report, because nobody has accepted it yet: "New with B1, not on dev: with ASCII glyphs on, Console's staged file names and Inspect rows (text that goes through `resolve_glyph_text`) now also show `· — … × →` and the other frame glyphs as their ASCII forms, and `…` and the arrows get wider. Keep it, or should B1 keep user text out of the new glyph entries?" Write the owner's reply to both, verbatim, to `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/owner-screenshots.txt`. A subagent's or the controller's own approval is not owner approval, and nothing is waived. If the owner asks for changes (to the screenshots or to the ASCII side effect), they become new steps in the task that owns the behaviour (Task 2 for the glyph map, with its tests), then Steps 2-8 here run again. AC #11 is ticked only once this file holds an approval of both.

- [ ] **Step 9: Merge-time re-verification**

Run just before the controller opens the PR, and again before it merges:

```bash
bash <<'EOF'
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b1; EV=$MAIN/.worktrees/.b1-evidence
git -C "$MAIN" fetch origin --prune --quiet
(cd "$WT" && "$MAIN/.venv/bin/python" scripts/check_backlog_task_ids.py | tail -1)
echo "dev DESIGN.md header-variant lists: $(git -C "$MAIN" show origin/dev:DESIGN.md | grep -c 'Destination header variants')"
"$EV/pr_collisions.sh"
echo "dev commits since the stack base touching B1's files:"
git -C "$WT" log --oneline "$(cat "$EV/stack-cut.txt")..origin/dev" -- $(cat "$EV/b1-files.txt") | head -20
PR=$(gh pr list --head feat/roleplay-b1-one-row-header --state open --json number -q '.[0].number // empty')
if [ -n "$PR" ]; then
  echo "PR #$PR: base $(gh pr view "$PR" --json baseRefName -q .baseRefName), head $(gh pr view "$PR" --json headRefOid -q .headRefOid | cut -c1-10), local $(git -C "$WT" rev-parse --short=10 HEAD), pushed $(cut -c1-10 "$EV/pushed-sha.txt")"
  gh pr checks "$PR" 2>&1 | grep -F "Derived artifacts reproduce from their sources" || echo "MISSING: the required check has not run on this head"
fi
EOF
```

Expected: `No duplicate task IDs across …` (B1 files no task; a duplicate means dev moved under the branch: rebase first, per `backlog/docs/lessons-backlog-hygiene.md`); `dev DESIGN.md header-variant lists: 0` (if it reads 1, ADR-210 step 8, TASK-33627, landed first: rebase and fold B1's Lab-and-Roleplay bullet into dev's list so there is one list, per G12 and FU-4); the collision lines (Task 0 Step 1's form); an empty list of dev commits touching B1's files since the stack base; and, once the PR exists, `base dev`, the same SHA three times (PR head, local, pushed) and the required check's line with `pass` on that head. Any dev commit in that list (for example #2862, #2563, #3045, #3023, #3036, #3022, #3039, #2868 or #3010 merging: they move PS, the ratchet file, the glyph map, the boot bytes, `_agentic_terminal.tcss`, the census, the workflow or the pre-import limits; #3028, which moved the census, the workflow and `_console.tcss`, has already merged at `8d502ba250`, so a B1 cut before that SHA lists it here) means: run Step 1's rebase block, push with `--force-with-lease`, and re-run Steps 2-5 and this step. A `MISSING` line or a red check: do not merge. The controller merges under CLAUDE.md's merge rules: re-sync only this PR and only by rebase (`$EV/restack.sh` plus a force-with-lease push here, never `gh pr update-branch`), then merge the moment the required check is green on the current head.

- [ ] **Step 10: Write the PR body**

Every number in the ledger is generated from the evidence (the gate script's `base |`/`head |` lines, the pre-import JSON, the two arms' files), never typed: a typed value is a placeholder that can ship stale.

```bash
bash <<'OUTER'
set -u
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b1-evidence; WT=$MAIN/.worktrees/roleplay-b1
ID=$(cat "$EV/b1-task-id.txt")
BASE_SHA=$(cut -c1-10 "$EV/base-sha.txt")
PS_LINES=$(wc -l < "$WT/tldw_chatbook/UI/Screens/personas_screen.py" | tr -d ' ')
PS_BASE=$(wc -l < "$MAIN/.worktrees/roleplay-b1-base/tldw_chatbook/UI/Screens/personas_screen.py" | tr -d ' ')
LEDGER_ROWS=$("$PY" - "$EV" "$MAIN/.worktrees/roleplay-b1-base" "$WT" <<'PY'
import json
import re
import sys
from pathlib import Path

ev, base_wt, head_wt = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
lines = {}
for line in (ev / "gates.txt").read_text(encoding="utf-8").splitlines():
    match = re.match(r"^(base|head) \| (.*)$", line)
    if match:
        lines[match[1]] = match[2]  # the last measurement per arm wins


def field(arm: str, pattern: str) -> str:
    found = re.search(pattern, lines.get(arm, ""))
    return found[1] if found else "MISSING"


def number(text: str) -> str:
    return f"{int(text):,}" if text.isdigit() else text


boot = {arm: field(arm, r"boot-css=(\d+)") for arm in ("base", "head")}
broad = {arm: field(arm, r"\[census\] total=(\d+)") for arm in ("base", "head")}
tour = {arm: field(arm, r"full tour: (\d+)") for arm in ("base", "head")}
ready = {arm: field(arm, r"ui-ready boots: ([\d/]+)") for arm in ("base", "head")}
weight = {arm: field(arm, r"boot-import-weight: (\d+)/") for arm in ("base", "head")}
scoped = {
    arm: (root / "tldw_chatbook/css/widget_defaults_scoped.tcss").stat().st_size
    for arm, root in (("base", base_wt), ("head", head_wt))
}
delta = int(boot["head"]) - int(boot["base"]) if all(v.isdigit() for v in boot.values()) else "?"
print(f"| Boot parsed CSS (`_boot_parsed_css_census()`) | {number(boot['base'])} B | {number(boot['head'])} B | {delta:+} B (expected −215: −216 dead `_agentic_terminal` items, +52 module banner, +45 the readable-warning token, −96 the task-523 rule); ceiling 608,090 unchanged |" if delta != "?" else "| Boot parsed CSS | MISSING | MISSING | re-run Step 2 |")
print(f"| `widget_defaults_scoped.tcss` (holds the `PersonasScreen` segment) | {scoped['base']:,} B | {scoped['head']:,} B | {scoped['head'] - scoped['base']:+} B (expected −2: the blank line after the retired task-523 comment) |")
print(f"| Broad selectors | {broad['base']} | {broad['head']} | gate head ≤ base (the constant is 274: no headroom); the lazy sheet pins its own subjects |")
print(f"| UI-ready modules (three boots) | {ready['base']} | {ready['head']} | gate: no module resident on head that is never resident on base (the census block) |")
print(f"| Boot import weight | {weight['base']} | {weight['head']} | gate: equal |")
print(f"| CSS sources after the destination tour | {tour['base']} | {tour['head']} | gate: head ≤ base + 1 (the Roleplay sheet) and below 56 |")
base, head = (json.loads((ev / f"preimport-{arm}.json").read_text(encoding="utf-8")) for arm in ("base", "head"))
rb, rh = base["routes"]["ccp"], head["routes"]["ccp"]
print(f"| Pre-import pass modules | {base['modules']} | {head['modules']} | {head['modules'] - base['modules']:+d} `UI.Persona_Modules.roleplay_frame_state`; owner-signed ADR-097 row, raised in the commit whose PS first imports it (Task 6) |")
print(f"| Pre-import pass LOC | {base['loc']:,} | {head['loc']:,} | {head['loc'] - base['loc']:+,} (raised only if it exceeded its limit; see the ledger row) |")
print(f"| Roleplay route (`ccp`) | {rb[0]} mods / {rb[1]:,} LOC | {rh[0]} mods / {rh[1]:,} LOC | {rh[1] - rb[1]:+,} LOC |")
print(f"| Largest route | {base['fattest_route_loc']:,} LOC | {head['fattest_route_loc']:,} LOC | {head['fattest_route_loc'] - base['fattest_route_loc']:+,} LOC |")
PY
)
# What the owner's two rulings changed, read from the committed files on both arms (never typed).
OUTCOMES=$("$PY" - "$MAIN/.worktrees/roleplay-b1-base" "$WT" <<'PY'
import re
import sys
from pathlib import Path

base_wt, head_wt = Path(sys.argv[1]), Path(sys.argv[2])
PRE = "Tests/Performance/test_screen_preimport_payload_budget.py"
RATCHET = "Tests/Architecture/test_module_size_ratchet.py"


def constant(root: Path, name: str) -> int:
    text = (root / PRE).read_text(encoding="utf-8")
    return int(re.search(rf"^{name} = ([\d_]+)$", text, re.M)[1].replace("_", ""))


def ratchet_row(root: Path, rel: str) -> int:
    text = (root / RATCHET).read_text(encoding="utf-8")
    return int(re.search(rf'^    "{re.escape(rel)}": (\d+),$', text, re.M)[1])


limits = []
for name in ("MAX_PASS_ADDED_MODULES", "MAX_PASS_ADDED_LOC", "MAX_SINGLE_ROUTE_ADDED_LOC"):
    old, new = constant(base_wt, name), constant(head_wt, name)
    limits.append(f"`{name}` {old:,} → {new:,} (raised)" if new > old else f"`{name}` unchanged at {old:,}")
print("; ".join(limits))
rows = []
for rel in ("tldw_chatbook/UI/Screens/personas_screen.py", "tldw_chatbook/app.py"):
    old, new = ratchet_row(base_wt, rel), ratchet_row(head_wt, rel)
    word = "raised" if new > old else "lowered" if new < old else "unchanged"
    rows.append(f"`{Path(rel).name}` row {old:,} → {new:,} ({word})")
frame_state = ratchet_row(head_wt, "tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py")
print("; ".join(rows) + f"; new `roleplay_frame_state.py` row {frame_state:,}")
PY
)
printf '%s\n' "$OUTCOMES" > "$EV/outcomes.txt"   # Task 13's notes quote the same two lines
PRE_OUTCOME=$(echo "$OUTCOMES" | sed -n 1p); ROW_OUTCOME=$(echo "$OUTCOMES" | sed -n 2p)
{
cat <<EOF
## Roleplay frame B1: one-row header and the lazy Roleplay stylesheet

TASK-${ID} · spec \`Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md\` §5.2 B1 · plan \`Docs/superpowers/plans/2026-10-03-roleplay-b1-one-row-header.md\` · on dev ${BASE_SHA} (B0, #2977, merged 2026-10-04)

### What changed
- Header: one inline row, \`Roleplay  <Kind> › <item>[ · editing]\`, then **Unsaved changes** (the ADR-046 aggregate, in-flight saves included; **Unsaved** below 100 columns), **No chat provider · Settings ›** (destination-wide; a click opens Settings › Providers & Models for the chat_defaults provider, through the leave guard), and **Local** or **Server: <label> · read-only**. On dev today the header carries a "Ready"/"Blocked" badge that reads provider settings only and can say "Ready" before a send that is then refused (RP-067, from the 2026-10-01 review); B1 fixes the header's part: the badge is gone (the header is even composed with the data-source word). The Inspector's own readiness line is unchanged by B1. The kind is never cut, and a long name ends in the resolved ellipsis.
- New pure \`UI/Persona_Modules/roleplay_frame_state.py\` (inputs to view, the R24 predicate, the cell-measured fit and degrade order, plain-then-escaped untrusted text, the composed initial state, the moved mode descriptors and purpose line) and a shared, CSS-free \`FittedText\` in \`UI/Workbench/workbench_widgets.py\`; \`personas_screen.py\` keeps the gather-and-paint glue (${PS_BASE} → ${PS_LINES} lines). TASK-33622.14's leave and Ctrl+Q guard now decides on the same predicate (R24, G4). The draft aggregate reads the cached character editor instead of walking the DOM on every poll tick.
- Lazy \`css/features/_roleplay.tcss\` → \`screen_feature_roleplay.tcss\`, loaded by \`TldwCli._SCREEN_OWNED_ROUTE_CSS[TAB_PERSONAS]\` on the first visit (initial tab or Ctrl+4), never \`PersonasScreen.CSS_PATH\`; Roleplay's copy of the Library's adaptive-shell rules, pinned equal by role. Readable chip colours through a new \`\$ds-status-warning-readable\` token (the sibling of \`\$ds-status-error-readable\`), because \`\$warning\`/\`\$error\` text measured below AA on the panel in 29/40 of 70 themes. Nine never-composed \`#personas-*\` selector items and the dead task-523 badge rule leave boot.
- R33 (hostile names), no bug fix in this PR: on dev before B1, TASK-34400 (#3015) had already fixed the pre-existing bug in which a Roleplay name shaped like markup (\`[/]\`) exited the app. B1 removes the old header subtitle, and with it TASK-34400's escaping there, because the name now paints in a literal label; every other TASK-34400 fix is unchanged. B1's own new surfaces stay literal: the header's item label (a \`FittedText\`, literal \`Content\`) and the server label (escaped by \`build_header_view\` into the shared markup-on status). TASK-34400's hostile-name test is extended to both (\`test_the_header_item_label_and_server_label\`), its paint helpers moved to the frame harness (one home), and its two subtitle tests re-pinned to the header that replaced the subtitle.
- Glyph map: \`› ‹ … · — ⇥ → ← ↑ ↓ ×\` (G20). DESIGN.md "Destination header variants" (G12); ADR-046 dated amendment and the ADR-120 cross-reference (G4, the one predicate); the Roleplay guide's header and title.
- Tests: the frame harness with two styled tiers (\`Tests/UI/roleplay_frame_harness.py\`) and the header, stylesheet, frame-state, FittedText and journey suites, plus the hostile-name extension; four fast files appended to the UI Fast Lane census (B1's three, and TASK-34400's widget-level \`test_roleplay_hostile_text_surfaces.py\`, which no PR lane ran on dev and which holds one of B1's re-pins), four bootstrap-profile files in the PR Fast Lane's admission-sensitive step (TASK-34400's \`test_roleplay_hostile_names.py\` among them, also ungated on dev).

### Owner decisions (2026-10-04) and what they changed
- Pre-import census: "Expand it" — the one new module (\`roleplay_frame_state\`) stays its own module and the limits B1's head exceeds rise to the paired-arm measurement, with one ADR-097 ledger row quoting the answer (bound: +1 module, +1,000 lines). Outcome: ${PRE_OUTCOME}.
- Size ratchets: expand the limit, never contort code — B1's \`personas_screen.py\` row is set to the measured count (and any ratcheted file B1 grows is raised with a dated owner-decision comment). Outcome: ${ROW_OUTCOME}. The B1 card's literal gate for \`personas_screen.py\` ("≤16,436 today") predates this ruling and is superseded by it.

### Prerequisites
- TASK-33790, which the B1 task lists as a prerequisite, is still To Do on dev. B1 depended only on its AC#4 (Roleplay toasts paint square brackets literally), which TASK-34400 delivered on dev before B1. Its other ACs (a visible 200-character persona-name limit before Save, a readable validation message, the same for a tool-rule name over 512 characters, and their tests) are still open on dev; they are not B1's, and B1 does not fix them.

### Census measurements (paired arms, same session; raw output of the gate script)
\`\`\`
EOF
grep -E '^(base|head) \||^ui-ready:' "$EV/gates.txt" | tail -4
grep -E '^arrival (base|head):' "$EV/gates.txt" | tail -2
grep -E '^poll ' "$EV/gates.txt" | tail -4
cat <<EOF
\`\`\`

### Budget ledger (spec §5.10)
| Guard | Base | Head | Note |
|---|---|---|---|
EOF
echo "$LEDGER_ROWS"
cat <<EOF
| PS lines (module-size ratchet) | ${PS_BASE} | ${PS_LINES} | row set to the measurement (owner ruling 2026-10-04: expand the limit, never squeeze code); recipient-ceiling row for \`roleplay_frame_state.py\` |
| Readiness poll, per 0.25 s tick | see the \`poll\` lines above | | B1 gathers the header inputs on every tick (readiness and the draft aggregate), where the old poll returned early with nothing staged; gate head ≤ base + 1 ms per tick |
| First Roleplay visit, lazy imports | see the \`arrival\` lines' \`nav_module\` | | the header's first paint reads the draft aggregate, whose snapshot type loads \`UI.Navigation.character_conversation_navigation\` lazily (about 5 ms, one module; invisible to the pre-import census) |
| Destination tour switch time, Ctrl+4 retention, ADR-115 | — | — | perf and roleplay groups: no new failures (paired arms below) |
| First Ctrl+4 (evidence, ABBA, 10 per arm) | see the arrival lines above | | the Roleplay sheet's parse is the css-load median |

### Paired base arm (failure-set diff, "no new failures")
Each arm ran with the collection-time plugin \`b1_bootstrap_all\` (imports \`tldw_chatbook.app\` and keeps the bootstrap profile on every item; lessons-testing-evidence), except the perf group, which runs as CI runs it. \`recovery=\` is each arm's \`RecoveryRequired\` count. The timer-path census is red on dev, so its unclassified sites are compared directly (last lines).
EOF
grep -E '^[a-z0-9-]+ (base|head): |^new failures|^\(end' "$EV/gates.txt" | grep -v '^arrival '
grep -E 'unclassified timer-path sites' "$EV/gates.txt" | tail -2
cat <<EOF

### Named mutations (each red, then restored green)
EOF
cat "$EV/mutations.md"
cat <<EOF

### Live check (harness from PR #2957, \`launch.sh --self-test\` PASS, B1-private HARNESS_STATE)
Both arms at 80x24, 120x36, 160x45 and 220x55, fresh launches: route activation, the Edit → type → Unsaved → Ctrl+S journey, a resize and its restoration, the untouched initial-tab arrival, and a run with no chat provider. Checker:
\`\`\`
EOF
grep -E '^captures: ' "$EV/gates.txt" | tail -1
cat <<EOF
\`\`\`
Approval captures: \`Docs/superpowers/plans/2026-10-03-roleplay-b1-captures/\`. Owner screenshot approval (ADR-007), verbatim:
> $(tr '\n' ' ' < "$EV/owner-screenshots.txt")

### Interims (spec §5.3), deviations and known side effects
- The item name sits in the header as \`› <item>\`, in its own literal label beside the kind, until B5b-2's work-pane title row; \`#personas-purpose\` and the mode strip stay until B6.
- The blocked chip is a click target only (it passes the leave guard): keyboard reach and its \`LEAVES_SCREEN_IDS\` entry arrive with B3's Tab region (TASK-33910.5 carries the note); the Inspector's readiness line keeps its own Settings link meanwhile.
- Exists on dev today, not fixed by B1 (B5b-1 fixes it): selecting another item checks only the form and visual authoring (\`_confirm_discard_unsaved\`), so with a staged avatar or a save in flight a selection change discards or waits without asking. B1 adds no such path; its new chip now shows **Unsaved changes** in those states, so the chip and the selection guard can disagree until B5b-1 (the leave and Ctrl+Q guards already ask, and decide on the chip's predicate). The list rows' unsaved badge still follows \`has_unsaved_changes\`, as on dev.
- The styled full tier is seeded through the same character seams as the mock tier, not a temporary ChaChaNotes (spec §5.7.1): it proves styling and routing, not save → reload persistence. The journey checks the payload that reached \`update_character\`; B2a converts the tier before its volume tests (TASK-33910.3 carries the note).
- New with B1, not on dev (dev's five-row header has no single row to clip): below 65 columns the worst-case one-row header (every chip, a server label, the longest kind) does not fit and its right end clips. B1 does not fix this; its matrix starts at 80x24, and the spec's below-64 layout is B7's (60x24) and TASK-33910.24's.
- R33's escaper is \`Utils.input_validation.escape_markup\`, not the \`textual.markup.escape(text)\` the spec names: on Textual 8.2.8 \`escape('[/')\` still raises \`MarkupError\` and \`'[TODO] y'\` renders as \`' y'\` (pinned by the hostile-name test); the spec's R33 wording should say so when it is next touched.
- The Roleplay split claims four SHARED tokens beyond R18's prefix list (\`workbench-header-title\`, \`-subtitle\`, \`-status\`, \`-active\`), so its copies of the header and grip rules leave boot; legal only because every Roleplay selector also carries a Roleplay token (\`test_every_roleplay_selector_carries_a_roleplay_token\`).
- The B1 card's "size-matrix fixture" is a parametrize mark (\`size_matrix\`, ids \`80x24\` …), so test modules need no fixture import.
- New with B1, not on dev, outside Roleplay: with ASCII glyphs on, Console surfaces that pass user text through \`resolve_glyph_text\` (staged file names, Inspect rows) now also rewrite \`· — … × →\` and friends; \`…\` and the arrows are wider in ASCII. Put to the owner with the screenshots (Task 12 Step 8); the reply is quoted above.
- The guide's screenshots predate B1 (\`Docs/User_Guide/images/roleplay/{overview,character-card,character-editor,dictionary-entries,lore-entries}.svg\` still show the "Roleplay & Chat Dictionaries" header with a "Ready" badge): stale until B12 re-captures them (TASK-33910.18 carries the note).

### Not in B1
No ADR-011 capture: B1 retires no legacy path and adds no worker or timer, but the existing 0.25 s readiness poll now gathers the header inputs on every tick (measured above: the \`poll\` lines and the ledger row). \`-residual\` (spec §2.12 item 2) waits for B7, where the residual work strip exists. PERF-22's deterministic counts are B2a's and B6's. No "Verified against" stamp in the guide.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
} > "$EV/pr-body.md"
wc -l "$EV/pr-body.md"
grep -n "MISSING" "$EV/pr-body.md" || echo "no MISSING field"
OUTER
```

Open `$EV/pr-body.md` and check every ledger row against the census block above it (both come from the same `gates.txt` lines; a `MISSING` field means Step 2's line lacks that field: re-run Step 2). A delta other than the expected ones in the Note column means a gate failed: go back to Step 2. Add any load flake Step 4 recorded, and Task 10 Step 4's "pre-existing, not B1's" claim-string candidates. Hand the file to the controller; do not open the PR from this task.

---

### Task 13: Close out TASK-33910.2 (and note the follow-up slices)

**Files:**
- Modify: `backlog/tasks/task-33910.2 - Roleplay-frame-B1-one-row-header-and-the-lazy-Roleplay-stylesheet.md` (ACs, Implementation Notes, status), and one dated note each on `task-33910.3` (B2a), `task-33910.5` (B3) and `task-33910.18` (B12) (Step 2). TASK-33790 gets no note: its AC#4 (literal toasts) was delivered and ticked by TASK-34400 before B1, and its other ACs are not B1's (the B1 notes and the PR body say so).
- Commit: `Docs/superpowers/plans/2026-10-03-roleplay-b1-captures/` (Task 12 Step 7)

**Interfaces:**
- Consumes: `$EV/gates.txt`, `$EV/mutations.md`, `$EV/owner-screenshots.txt`, `$EV/outcomes.txt`, `$EV/preimport-{base,head}.json`, `$EV/base-sha.txt`, `$EV/b1-task-id.txt` (Tasks 0, 11 and 12).
- Produces: nothing for later tasks.

Five-digit ids break `backlog task edit` (lessons-backlog-hygiene, TASK-15463), so every task file is edited directly. Tick an AC only for what the evidence proves:

| AC | Evidence |
|---|---|
| #1 one row at the four sizes, subtitle visible at ≤24 rows | `test_header_is_one_row_and_the_kind_stays_visible` (both tiers × four sizes); Task 12 Step 7 checker |
| #2 first item on row 19, ≥7 characters at 120x36, relative to the measured nav and header | `test_first_list_item_sits_right_under_the_unchanged_band`, `test_seven_characters_show_at_120x36` (row 18 at 80x24, whose compact workbench has one row less chrome: recorded in the notes), `test_the_work_pane_and_inspector_keep_their_widths` (spec §5.3's "unchanged"); `$EV/mutations.md` (`drop-inline-header-rule`, `inspector-wider`) |
| #3 chip iff the aggregate predicate | `test_unsaved_predicate_is_the_aggregate_including_inflight_saves`, `test_unsaved_chip_follows_the_aggregate_not_has_unsaved_changes` (seven cases), `test_persona_save_clears_the_unsaved_chip_without_the_poll`, `test_the_leave_and_quit_guard_decides_on_the_same_predicate` |
| #4 long name ellipsises, kind never cut | `test_a_long_name_ellipsises_and_the_kind_is_never_cut`, `test_item_fit_never_paints_past_its_width`, `test_every_header_part_fits_in_the_worst_case` (both tiers, every part painted inside the header and uncovered); `$EV/mutations.md` (`item-not-flexible`, `go-marker-unresolved`) |
| #5 the journey | `test_editing_shows_the_unsaved_chip_and_saving_clears_it` (both tiers); the live `dirty`/`saved` captures |
| #6 discrimination under both styled tiers | `test_deleting_one_header_rule_turns_the_one_row_assertion_red`; `$EV/mutations.md` (`drop-inline-header-rule`) |
| #7 hostile names | `test_the_header_item_label_and_server_label` (B1's new surfaces, five names, positive control, every painted copy clicked; mutations `status-unescaped`, `item-label-markup`), and TASK-34400's 22 + 81 hostile-text tests still green with B1's re-pins (`test_roleplay_hostile_names.py`, `test_roleplay_hostile_text_surfaces.py`: every other Roleplay surface, fixed on dev before B1) |
| #8 loaded on the first visit, never `CSS_PATH`, scans cover it | `test_the_full_app_loads_the_sheet_on_the_first_visit_only`, `test_the_sheet_is_route_loaded_and_never_a_screen_css_path`, the `_SPLIT_SHEET_OWNERS` entry and its negative control, `test_no_full_app_test_pushes_roleplay_without_its_sheet`, `test_screens_do_not_take_owned_sheets_onto_css_path` |
| #9 DESIGN.md G12 | Task 10 Step 1 |
| #10 the plan | this file (Task 0 Step 5's commit) |
| #11 spec §5.4 gates | Task 12 Steps 2-7 and `$EV/owner-screenshots.txt` (approval) |

- [ ] **Step 1: Write the Implementation Notes and tick the proven criteria**

Set `TICK` to the AC numbers the table above proves (all eleven once Task 12 is green and the owner approved; drop `11` while `$EV/owner-screenshots.txt` holds no approval). Status becomes Done only when all eleven are ticked.

```bash
bash <<'OUTER'
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b1; EV=$MAIN/.worktrees/.b1-evidence
TICK="1 2 3 4 5 6 7 8 9 10 11"
cd "$WT"
B1="backlog/tasks/task-33910.2 - Roleplay-frame-B1-one-row-header-and-the-lazy-Roleplay-stylesheet.md"
TODAY=$(date +%F); HEAD_SHA=$(git rev-parse --short HEAD); BASE_SHA=$(cut -c1-10 "$EV/base-sha.txt")
# Every measured figure below comes from the evidence (never typed), as in the PR body.
FIGURES=$("$MAIN/.venv/bin/python" - "$EV" <<'PY'
import json
import re
import sys
from pathlib import Path

ev = Path(sys.argv[1])
gates = (ev / "gates.txt").read_text(encoding="utf-8").splitlines()
arms = {}
for line in gates:
    match = re.match(r"^(base|head) \| (.*)$", line)
    if match:
        arms[match[1]] = match[2]  # the last measurement per arm wins


def delta(pattern: str) -> int:
    return int(re.search(pattern, arms["head"])[1]) - int(re.search(pattern, arms["base"])[1])


base, head = (json.loads((ev / f"preimport-{arm}.json").read_text(encoding="utf-8")) for arm in ("base", "head"))
polls = [line for line in gates if line.startswith("poll ")][-4:]
print(f"{delta(r'boot-css=(\d+)'):+,} B")
print(f"{delta(r'full tour: (\d+)'):+d}")
print(f"{head['modules'] - base['modules']:+d} module")
print(f"{sum(' ok ' in line for line in polls)} of {len(polls)} poll lines within 1 ms of the base arm")
PY
)
BOOT_DELTA=$(echo "$FIGURES" | sed -n 1p); TOUR_DELTA=$(echo "$FIGURES" | sed -n 2p)
PRE_DELTA=$(echo "$FIGURES" | sed -n 3p); POLL_VERDICT=$(echo "$FIGURES" | sed -n 4p)
PRE_OUTCOME=$(sed -n 1p "$EV/outcomes.txt"); ROW_OUTCOME=$(sed -n 2p "$EV/outcomes.txt")
cat > "$EV/notes.md" <<EOF
## Summary
Roleplay's five header rows are one inline row driven by a new pure module, and Roleplay has its own lazily loaded stylesheet.

## Changes
- New tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py: the R24 unsaved predicate over the ADR-046 aggregate, header inputs and view, the cell-measured item fit with the resolved ellipsis, the fixed degrade order, plain-then-escaped untrusted text, the moved mode descriptors and purpose line, Roleplay's pane class names.
- personas_screen.py: compose with a non-"Ready" initial state, gather (_gather_header_inputs) and paint (_paint_header); the readiness poll compares header inputs and the draft aggregate reads the cached editor instead of walking the DOM; on_resize refits from the cached inputs; a persona save resolves its completion, then repaints; the blocked chip deep-links to Settings for the chat_defaults provider. $(wc -l < "$MAIN/.worktrees/roleplay-b1-base/tldw_chatbook/UI/Screens/personas_screen.py" | tr -d ' ') -> $(wc -l < tldw_chatbook/UI/Screens/personas_screen.py | tr -d ' ') lines.
- roleplay_draft_guard.py (TASK-33622.14's leave and Ctrl+Q guard) decides on roleplay_has_unsaved_work().
- Shared FittedText (UI/Workbench/workbench_widgets.py): literal text refitted to its own width, no CSS.
- css/features/_roleplay.tcss -> screen_feature_roleplay.tcss through the route map; the readable \$ds-status-warning-readable token; nine dead _agentic_terminal items and the task-523 rule removed (boot ${BOOT_DELTA}).
- Glyph map (G20), DESIGN.md header variants (G12), ADR-046 amendment and ADR-120 cross-reference (G4), the Roleplay guide's header and title.
- Tests: the Roleplay frame harness (two styled tiers; the one home of the paint helpers TASK-34400 added), header, stylesheet, frame-state, FittedText and journey suites; TASK-34400's hostile-name test extended to the new item label and server label, and its two subtitle tests re-pinned to the header that replaced the subtitle; PR-gate census and the admission-sensitive step.
- R33 (hostile names): on dev before B1, TASK-34400 had already fixed the pre-existing bug in which a name shaped like markup exited the app. B1 removes the old header subtitle, and with it TASK-34400's escaping there, because the name now paints in a literal label; every other TASK-34400 fix is unchanged. B1 keeps its new surfaces (item label, server label) literal, pinned by test_the_header_item_label_and_server_label.

## Evidence
- Live check verified $TODAY against $HEAD_SHA (base arm $BASE_SHA): both arms at 80x24, 120x36, 160x45 and 220x55 on the B1-private harness; checker lines and the owner's approval in the PR body; approval captures in Docs/superpowers/plans/2026-10-03-roleplay-b1-captures/.
- Paired base arm, every group (roleplay, shell, css, perf): no new failures; RecoveryRequired counts in single digits.
- Census: boot CSS ${BOOT_DELTA}, broad selectors and UI-ready and boot import weight unchanged, CSS sources after the tour ${TOUR_DELTA}, pre-import ${PRE_DELTA} (owner-signed ADR-097 row in the Task 6 commit, owner sign-off 2026-10-04 "Expand it": ${PRE_OUTCOME}); the size-ratchet rows set to the measurement (owner ruling 2026-10-04: ${ROW_OUTCOME}); the readiness poll's per-tick cost: ${POLL_VERDICT} (the poll lines in the PR body); the first visit loads UI.Navigation.character_conversation_navigation lazily (recorded in the ledger).
- Named mutations: the table in the PR body.

## Decisions
- The kind is the DestinationHeader subtitle (app-authored, never cut); the interim item paints in its own literal FittedText, so no untrusted name enters the shared markup-on header; only the server label does, through escape_markup at build_header_view.
- First list item: row 19 at 36 rows and more, row 18 at 80x24 (the compact workbench is one row shorter there); asserted relative to the measured nav and header.
- Chips use readable foregrounds (\$text-warning, \$text-error), not the spec's \$ds-status-warning hue, which fails AA on the panel in 29 of 70 themes.
- Interims: \`› <item>\` until B5b-2; #personas-purpose and the mode strip until B6; the blocked chip is mouse-only until B3's Tab region (TASK-33910.5 noted); the in-screen selection guard ignores staged attachments and in-flight saves (exists on dev today; B1 does not fix it, B5b-1 does), so B1's new chip and that guard can disagree until then, and the list rows' unsaved badge still follows has_unsaved_changes, as on dev. -residual waits for B7.
- The B1 card's literal personas_screen.py gate ("≤16,436 today") predates the owner's 2026-10-04 ruling and is superseded by it: the row follows the measurement (${ROW_OUTCOME}).
- TASK-33790 (listed as B1's prerequisite) is still To Do: B1 depended only on its AC#4, which TASK-34400 delivered on dev; its other ACs are still open on dev, not B1's, and B1 does not fix them.
- Deviations (each in the PR body): the styled full tier is seeded through the character seams, not a temporary ChaChaNotes, so it proves styling and routing, not persistence (B2a converts it: TASK-33910.3 noted); below 65 columns the worst-case one-row header clips (new with B1, not on dev; B1 does not fix it, B7's 60x24 and TASK-33910.24 do); with ASCII glyphs on, Console's staged file names and Inspect rows now also rewrite the new frame glyphs (new with B1, not on dev; the owner's answer is in owner-screenshots.txt); R33 uses Utils.input_validation.escape_markup, not textual.markup.escape (which raises on "[/" and swallows "[TODO]" on Textual 8.2.8); the Roleplay split claims four shared tokens beyond R18's list under an anchor guard; the card's size-matrix "fixture" is a parametrize mark.
EOF
"$MAIN/.venv/bin/python" - "$B1" "$EV/notes.md" $TICK <<'PY'
import re
import sys
from pathlib import Path

path, notes, ticks = Path(sys.argv[1]), Path(sys.argv[2]).read_text(encoding="utf-8"), sys.argv[3:]
text = path.read_text(encoding="utf-8")
for number in ticks:
    text, count = re.subn(rf"^- \[ \] #{number} ", f"- [x] #{number} ", text, count=1, flags=re.M)
    assert count == 1 or f"- [x] #{number} " in text, f"AC #{number} not found"
if "<!-- SECTION:NOTES:BEGIN -->" in text:
    text = re.sub(
        r"<!-- SECTION:NOTES:BEGIN -->.*?<!-- SECTION:NOTES:END -->",
        lambda _m: f"<!-- SECTION:NOTES:BEGIN -->\n{notes}<!-- SECTION:NOTES:END -->",
        text,
        count=1,
        flags=re.S,
    )
else:
    text = text.rstrip("\n") + f"\n\n## Implementation Notes\n\n<!-- SECTION:NOTES:BEGIN -->\n{notes}<!-- SECTION:NOTES:END -->\n"
unticked = re.findall(r"^- \[ \] #(\d+)", text, flags=re.M)
if not unticked:
    text = text.replace("status: In Progress\n", "status: Done\n", 1)
path.write_text(text, encoding="utf-8")
print("unticked:", unticked or "none", "| status:", re.search(r"^status: (.*)$", text, re.M)[1])
PY
"$MAIN/.venv/bin/python" scripts/check_backlog_task_files.py | tail -1
OUTER
```

Expected: `unticked: none | status: Done` (or the numbers you left out and `status: In Progress`), then the task-files check passing.

- [ ] **Step 2: Hand the recorded interims and deviations to the slices that remove them**

Each note is dated, names B1, and is written only once (a re-run prints `already noted`). The note goes into the task's Implementation Notes section, creating it if absent.

```bash
bash <<'OUTER'
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; cd "$MAIN/.worktrees/roleplay-b1"
"$MAIN/.venv/bin/python" - "$(date +%F)" <<'PY'
import sys
from pathlib import Path

today = sys.argv[1]
NOTES = {
    "33910.3": (
        "B1 deviation: the styled full tier (Tests/UI/roleplay_frame_harness.py, "
        "roleplay_full_app) is seeded through the ccp_character_handler seams, not a "
        "temporary ChaChaNotes (spec 5.7.1). B2a converts it to DB seeding "
        "(attach_chachanotes_db or a tmp-file CharactersRAGDB, add_character_card) before "
        "its volume tests, and the B1 journey then asserts save -> reload."
    ),
    "33910.5": (
        "B1 interim: #personas-header-blocked is click-only. B3 makes it keyboard-reachable "
        "and adds it to LEAVES_SCREEN_IDS (spec I2), so it is never an arrival or F6 stop."
    ),
    "33910.18": (
        "B1 left the guide's Roleplay screenshots stale: Docs/User_Guide/images/roleplay/"
        "{overview,character-card,character-editor,dictionary-entries,lore-entries}.svg "
        "still show the 'Roleplay & Chat Dictionaries' header with a 'Ready' badge. B12 "
        "re-captures them."
    ),
}
for task_id, body in NOTES.items():
    matches = sorted(Path("backlog/tasks").glob(f"task-{task_id} - *.md"))
    assert len(matches) == 1, (task_id, matches)
    path = matches[0]
    text = path.read_text(encoding="utf-8")
    if "(Roleplay frame B1, TASK-33910.2)" in text:
        print(f"TASK-{task_id}: already noted")
        continue
    line = f"- {today} (Roleplay frame B1, TASK-33910.2): {body}\n"
    if "<!-- SECTION:NOTES:END -->" in text:
        text = text.replace("<!-- SECTION:NOTES:END -->", line + "<!-- SECTION:NOTES:END -->", 1)
    else:
        text = text.rstrip("\n") + f"\n\n## Implementation Notes\n\n<!-- SECTION:NOTES:BEGIN -->\n{line}<!-- SECTION:NOTES:END -->\n"
    path.write_text(text, encoding="utf-8")
    print(f"TASK-{task_id}: noted")
PY
"$MAIN/.venv/bin/python" scripts/check_backlog_task_files.py | tail -1
OUTER
```

Expected: `TASK-33910.3: noted`, `TASK-33910.5: noted`, `TASK-33910.18: noted` (or `already noted`), then the task-files check passing. If Task 0 Step 1 found TASK-26983 or TASK-27000 not Done and the owner chose to keep B1 going, add the same kind of note to each: `Blocked until Roleplay frame B12 (TASK-33910.18) merges: spec §5.6/K19; B1 started on <date>`.

- [ ] **Step 3: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b1
ID=$(cat /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b1-evidence/b1-task-id.txt)
backlog task list --plain 2>/dev/null | grep -F "TASK-${ID} "
git status --short
git add "backlog/tasks/task-${ID} - Roleplay-frame-B1-one-row-header-and-the-lazy-Roleplay-stylesheet.md" backlog/tasks/task-33910.3\ -\ *.md backlog/tasks/task-33910.5\ -\ *.md backlog/tasks/task-33910.18\ -\ *.md Docs/superpowers/plans/2026-10-03-roleplay-b1-captures
git commit -m "chore(backlog): B1 evidence, live-check captures and implementation notes (TASK-33910.2)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: the B1 task listed with its new status; `git status --short` showing only the four task files (` M`) and the captures directory (`??`) before the commit (delete a stray `backlog/tasks/task-task- - .md` if one appears; the plan was committed in Task 0), then the commit line.

No lessons entry unless the work produced a real incident (spec §5.4 item 8). Two candidates measured while planning, for the controller to judge: the decorative status hues (`$warning`, `$error`) fail AA as text on the panel in many themes, and the UI-ready census flakes by one module on dev.

---

## Self-Review

Performed after finishing the plan, against the B1 card, the sections and rulings the orchestrator named, and the writing-plans conventions; re-run on 2026-10-04 after the second review round's findings were applied and B0 (PR #2977) merged to dev, and again on 2026-10-04 after the owner sent the plan back (the three rulings in Global Constraints) and TASK-34400 merged: the plan was re-anchored on `origin/dev @ 8c4dfe59a2` and dry-run there; and again on 2026-10-08, after dev moved to `a793acbef5` and a rulings-and-coverage review's twelve findings were applied: re-anchored and dry-run on `a793acbef5` (§5); and again later on 2026-10-08, after dev moved to `8d502ba250` (PR #3028 merged) and an independent dry run there passed with six minor findings: all six were applied (one in part, with evidence), each affected part re-run on `8d502ba250`, and the stated figures re-anchored there (§5, §6). The first draft (5,412 lines, Tasks 0-11) was completed with Tasks 12-13; the review rounds then reworked Tasks 0, 4, 5, 6, 8, 10, 11, 12 and 13, the 2026-10-04 rework reworked Tasks 0, 1, 6, 8, 11, 12 and 13, the 2026-10-08 rework reworked Tasks 0, 1, 2, 5 (one block's exit status), 6 (one test and the ratchet script), 11, 12 and 13, and the second 2026-10-08 round changed the Conventions, Task 0 Step 4's ledger-row wording, seventeen count reads in Tasks 1-11, Task 10 Step 4, the known-failure notes of Task 6 Step 8 and Task 12 Step 4, and figures throughout (§6 lists what changed and why).

**1. Spec coverage — every B1 card bullet and every required item:**

| B1 card / required item | Task |
|---|---|
| Task 0 setup: B0 merged (#2977, rebase merge, 2026-10-04), so B1 moves off the pre-merge B0 head `f0a766a155` onto `origin/dev` with `--onto` (`$EV/restack.sh`, which refuses foreign commits); the paired base arm is the new merge-base with dev; evidence dir, mutation log, pre-import measure/raise helpers adapted from B0 (the raise keeps the head file's other limits and refuses a stale base arm), the open-PR collision scan; the owner's pre-import sign-off (2026-10-04, "Expand it", ruling 1) recorded before the commit whose PS imports the module, with a STOP kept only for growth beyond its bound | 0 (Steps 1-6); 6 Step 9 (the raise script's `STOP:` beyond the bound); Global Constraints (rulings, rebase protocol, push and PR base); 12 Steps 1 and 9 |
| §5.6/K19: the formatter batches TASK-26983/27000 run before B1 or after B12 | 0 Step 1 (both Done on dev via #2993: before B1, so PS is format-clean and stays so; Global Constraints) |
| The §1.3 header, subtitle `› <item>` interim (§5.3) | 4 (state), 6 (compose and paint); the item is a literal label beside the kind (see "deviations" below) |
| `PM/roleplay_frame_state.py` (pure; header state first) | 4 |
| `roleplay_has_unsaved_work()` (R24) drives the chip, following `_aggregate_roleplay_draft_snapshot()` | 4 (predicate), 6 (chip, poll, persona-save ordering) |
| §3.12's quit row and G4: TASK-33622.14's leave and Ctrl+Q guard (`roleplay_draft_guard.py`, Done on dev) decides on the same predicate | 6 Step 4b (+ its pin in `test_roleplay_frame_state.py` and mutation `guard-on-is-clean`); 10 Step 2 (the ADR-046 amendment names it; ADR-120 cross-references the amendment) |
| The blocked-destination chip, passing the leave guard | 4 (text, degrade), 6 (gather, click deep link, `test_the_blocked_chip_passes_the_leave_guard` on the full tier), 3 (`FittedText`) |
| Never "Ready", including before the first paint | 4 (`initial_header_state`), 6 (compose; `test_the_header_composes_without_a_ready_chip`) |
| Lazy `_roleplay.tcss` + `ScreenOwnedSplit` (R18, G18), route-map registration, first-visit load, never `CSS_PATH` | 5 |
| `PersonasScreen` in `test_screens_do_not_take_owned_sheets_onto_css_path` (str hole fixed); `_SPLIT_SHEET_OWNERS` entry with no owner; a real app that pushes `PersonasScreen` itself must load the sheet first | 5 (`test_no_full_app_test_pushes_roleplay_without_its_sheet`, with its negative control) |
| Roleplay's copy of the shell/grip rules + parity test (B0 is on dev, so not deferred to B2a) | 4 (class names), 5 |
| Two styled harness tiers, size matrix, `assert_painted_inside`, created first | 1; 6 uses `assert_painted_inside` on every header part (§5.7.2 item 3) |
| Discrimination self-test under both tiers | 7 (plus the source-deletion mutation, now also red for the AC#2 geometry) |
| Hostile-name test (card, lore book, dictionary, entry key, tag, persona, conversation title, server label; `[/`, `[TODO] y`) and the header's call-site escaping (FU-2). On dev TASK-34400 created the test (22 + 81 tests over every existing Roleplay surface, fixing the pre-existing exit-on-markup bug); B1 extends it to its own new surfaces and keeps the rest green | 1 (the paint helpers' one home is the harness), 6 Step 7 (the two tests that asserted the retired subtitle, re-pinned), 8 (`test_the_header_item_label_and_server_label`: item label and server label, styled tier 1); 4 (`escape_markup` in `build_header_view`, the screen's one call site; the subtitle carries only the kind) |
| Glyph-map entries + an ASCII render test | 2; 6 (`test_header_row_paints_ascii_markers_in_ascii_mode`, both tiers, four sizes; mutation `go-marker-unresolved`) |
| Delete the dead `AT` selector items | 5 |
| `#personas-purpose`/mode strip stay; the footer-hint builder stays | Global Constraints; 6 ("what stays") |
| DESIGN.md G12 amendment (sibling of ADR-210's Console entry) | 10 Step 1; 12 Step 9 re-checks dev |
| G4 (B1's part: the predicate), G17 (no `$ds-roleplay-*` needed), G18, G20 | 10 Step 2; Global Constraints; 5; 2 |
| Acceptance: one row at the four sizes, subtitle at ≤24 rows; row 19 / ≥7 characters relative to nav and header; chip parity; ellipsis and kind; journey | 6, 9 (Task 13's table maps each AC to its tests) |
| §5.3's interim geometry: the work pane and the Inspector unchanged (58/78/108, 30/39/54) | 6 (`test_the_work_pane_and_inspector_keep_their_widths`, both tiers, three sizes; mutation `inspector-wider`) |
| Re-pins: `:548` (y 10 → 6), `:871/8131/8154`, the 22 `#personas-header` refs, the visual-parity re-check, and the first-time replay that required the retired subtitle | 6 Steps 7-8 |
| Gates: CSS sources +1 < 56, boot net ≤ 0, PS's ratchet row set to the measurement (ruling 2; the spec's "≤16,436" gate predates the owner's ruling and #2993's re-measured row), pre-import +1 measured, broad selectors (zero headroom), UI-ready, preflight, paired arms per B0, the timer-path census floor | 5, 6 (Step 9's `ratchet_rows.py`), 11, 12 Steps 2-5 |
| §2.13 guards that B1 touches: resize does no data work; the lazy sheet inside the visit budget; the readiness poll's per-tick cost | 6 (`test_a_resize_refits_the_header_without_gathering_inputs`; edit 16 and `test_gathering_header_inputs_walks_no_dom`), 12 Step 3 (ABBA arrival with the lazy-import column; the poll probe) |
| Live check at 80x24/120x36/160x45/220x55 + owner screenshot STOP | 12 Steps 6-8 |
| User Guide delta (§5.8), no "Verified against" stamps | 10 Step 3 (the title too; claim strings and the stale screenshots in Step 4) |
| §5.4: backlog task, mutations, paired arms, preflight + PR-gate census, ledger (generated), ratchet rows + recipient ceiling, screenshots, docs, lessons | 0, every task, 11, 12, 13 |
| Task close-out, and the interims handed to the slices that remove them (B2a, B3, B12) | 13 (Steps 1-3) |

**2. Placeholder scan.** Searched the plan for "TBD", "TODO" (outside the hostile name `[TODO] y`), "implement later", "similar to Task", "add appropriate", "write tests for" and for any unfilled `@@…@@` marker left from the re-measurement: none in any step (re-run after the 2026-10-04 rework, and again after the second 2026-10-08 round, whose own fill-in markers are all replaced). The values an executor supplies are measured facts with their capture commands, or external replies: the owner's answers (Task 0 Step 6 records the 2026-10-04 pre-import answer verbatim; Task 12 Step 8 waits for the screenshot approval; a `STOP:` from `preimport_raise.py` beyond the bound goes back to the owner) and `TICK` in Task 13 (driven by its evidence table). Task 11 asks no owner question: a full UI Fast Lane shard gets another shard, the workflow's own rule. The PR body's ledger and its owner-decision outcomes, and the task notes' measured figures (boot bytes, tour sources, pre-import modules, the poll verdict, what the rulings changed), are generated from `gates.txt`, the pre-import JSON, `outcomes.txt` and the two arms' files, so no typed measurement can ship stale (re-checked in the 2026-10-08 rework, after a review found the notes still typed `−215 B`, `+1` and "within 1 ms"). Dates and SHAs in the notes are generated by the script.

**3. Type and name consistency.** The four paint helpers (`painted_rows`, `click_meta_cells`, `settle`, `wait_until`) and `seed_mock_characters` are defined once, in the harness, and imported by the harness self-tests, the header, journey and both TASK-34400 hostile-text modules (Task 1); no module defines its own copy. `RoleplayHeaderInputs(mode, edit_mode, item_name, unsaved, provider_blocked, runtime_source, server_label)` and `RoleplayHeaderView(state, item, unsaved_chip, blocked_chip, status_plain)` match between Task 4's module, its tests, Task 6's `_gather_header_inputs`/`_paint_header` and every test that monkeypatches `_gather_header_inputs`. `initial_header_state(mode, runtime_source)` matches Task 4's module and test, Task 6's compose and mutation 10. `fit_header_item((item, editing), width)` is the `FittedText` fit in Task 6's compose and the value tests compare (`("Detective Sam", True)`); it also takes `FittedText`'s empty default. The DOM ids `#personas-header`, `.personas-header-inline`, `#personas-header-tail`, `-item`, `-unsaved`, `-blocked`, `#personas-work-area`, `#personas-inspector-pane` match across the sheet (Task 5), the compose (Task 6), `_parts()`, the width pin and the hostile-name, journey and checker code. `HEADER_*_CELLS` constants match the sheet's spacing and are pinned by `test_header_chrome_cells_match_the_lazy_sheet` under both tiers. The harness names (`open_styled_roleplay`, `roleplay_full_app`, `styled_tiers`, `size_matrix`, `settle`, `wait_until`, `chrome_bottoms`, `painted_text`, `assert_painted_inside`, `click_meta_cells`, `drop_rule_from_loaded_sheet`) are imported exactly as Task 1 defines them. The evidence scripts agree on file names: `stack-cut.txt` = `base-sha.txt` (written only by `restack.sh`; read by Task 0 Step 2, the raise and ratchet scripts, Task 12 Steps 1, 5 and 9), `b1-files.txt` (the collision scan and Task 12 Step 9's dev log), `pushed-sha.txt` (Global Constraints, Task 12 Step 9), `preimport-{base,head}.json` (the measure script → the raise script, Task 11 Step 4, the PR body), `ratchet_rows.py` (Task 6 Step 9 → the rebase protocol, Task 12 Step 1), `b1_bootstrap_all` and `b1_tour_count` under `$EV/plug`, and the `gates.txt` lines the PR body reads (`^(base|head) \|`, `^ui-ready:`, `^arrival`, `^poll `, `unclassified timer-path sites`, `^captures:`).

**4. Review Focus placement.** Each of the six lines names its pinning tests, each written in full in its owning task: hostile names on B1's new surfaces (Tasks 8, 6 and 4, with TASK-34400's tests kept green), the chip and the aggregate including in-flight saves and the guard (Tasks 6, 4, 9), the kind at ≤24 rows and every part fitting, including the resize refit (Tasks 6 and 4), a harness or a direct push that never loads the lazy sheet (Tasks 1, 5, 7), long, wide and zero-width names (Tasks 4 and 6), and the poll's per-tick cost (Tasks 6 and 12).

**5. Verification done while finishing the plan.**

**The second 2026-10-08 re-anchor, on `origin/dev @ 8d502ba250`** (32 commits after `a793acbef5`: PRs #3031 and #3028). An independent verifier dry-ran the whole plan there on fresh detached worktrees under `.worktrees/`, in plan order, and reported PASS with six minor findings and no blocker or major:
- all 95 Old blocks matched exactly once as Edit-tool strings (one, ADR-212's long line, matches as a substring, which is what the Edit tool does) and all 9 Creates applied; a second fresh application, scripts included, gave 35 files byte-identical to the step-by-step tree; the four moved helpers have dev's source (AST compared); every scripted step gave its Expected output (the rename, the nine-item deletion, both CSS builds, `ratchet_rows.py` idempotent with `6 passed`, `preimport_raise.py` 557 → 558, the snapshot refresh, Task 11 Steps 2-4 idempotent);
- every failing-first run as stated (T1 `ModuleNotFoundError`, T2 `2 failed`, T3 and T4 `ImportError`, T5 `15 failed, 8 passed`, T6 `62 failed, 33 passed`) and every pass run as stated (T1 `7`/`103`, T2 `356`, T3 `22`, T4 `27`, T5 `23` and boot CSS −119, T6 `118`, then `1`/`35`/`103`, T7 `14`, T8 `108`, T9 `2`, T10 `14` on both arms);
- all thirteen of Task 6 Step 10's mutations red with the plan's counts and restored green, Task 8's two (`5 failed`, then `5 passed`), and those of Tasks 2, 3, 4, 5, 7 and 9;
- the module-size ratchet `10 failed, 42 passed` on base and `10 failed, 44 passed` on head (the same ten rows); PS 16,525, `roleplay_frame_state.py` 369 and `app.py` 5,712, each equal to its row; the pre-import guard `1 passed` on both arms; lint and format as Task 12 Step 5 states (`Found 96 errors.` on both arms, `26 files already formatted`);
- Task 12 Step 2's gates on both arms: base `608077 | 274 | 41 | 1033×3 | 681/686 | 557 / 416,217 | ccp 62 / 54,812`, head `607862 | 274 | 42 | 1033×3 | 681/686 | 558 / 416,596 | ccp 63 / 55,191` (UI-ready module sets equal both ways); the timer-path floor (71 on both arms, none new); the head-only suites (`125`/`125` and `110`/`110`); `preflight` all passed (`OK: 158`); the PR body generator with no `MISSING` field;
- Task 11 Step 1 in a minimal venv: `125 passed` (52.5 s), `110 passed` (260.2 s) and the step as CI runs it `445 passed, 2 xfailed` against `447 tests collected` (764.6 s); the PR Fast Lane bound 22 m 12 s + 4 m 20 s = 26 m 32 s;
- the poll probe (head 116/56 µs a tick against base 56/1 µs, all `ok`), the arrival probe (median 1.419 s base, 1.537 s head, the lazy sheet 17.2 ms, `nav_module` False/True);
- the paired groups: the changed existing test files (no new failure, `recovery=6` on both arms), `css` (`17 failed` on both, none new), `perf` (`27 passed` on both); `roleplay` and `shell` each showed "new" failures that the known-failure notes now explain (§6).

This round then applied the six findings and re-ran, on the same SHA and on its own detached worktrees `b1rw-fix3-{base,head}` (the head tree is the verifier's applied tree, with the pre-import files left at base so the revised raise script could run):
- Task 6 Step 9's pre-import block, extracted from this plan with its paths re-pointed: `base | modules 557 | LOC 416217 | … | fattest route 128281 LOC`, `head | modules 558 | LOC 416596 | … | fattest route 128281 LOC`, `raised: MAX_PASS_ADDED_MODULES 557 -> 558 | ledger row: written`, the snapshot line and `1 passed`; the test file and snapshot came out identical to the verifier's, and the ledger row identical up to its sign-off cell, which now reads `` Owner, 2026-10-04, asked whether the expanded limits also cover B1's one new module (`roleplay_frame_state`); answer, verbatim: "Expand it (Rec.)" ``; the whole head tree then differed from the verifier's applied tree in that one cell only (a full diff of both); a re-run changed nothing, and a sign-off with no quoted answer printed `STOP: owner-signoff.txt does not end with the owner's quoted answer` and left both files untouched;
- Task 12 Step 2's gate block, extracted and re-pointed: the same two lines as the verifier's, field for field, and `ui-ready: resident on head, never on base: []` both ways;
- Task 11 Steps 2 and 3 on the applied head: `census entries: 158`, `OK: 158 Tests/UI files in the PR gate (floor 158)`, `UI Fast Lane shards: 4`, one B1 file per shard, `admission-sensitive step: 18 files, B1's four last`, and the re-run changed nothing; the collision scan: 11 of 30 open PRs, the same eleven;
- Task 10 Step 4's new block: the six literal fragments counted `x2 x3 x2 x2 x1 x7`, `old title hits: 0`, and the four painted-text tests `15 passed`; the claim checker printed `139 quoted strings checked, 46 not emitted`, the composed strings among them, as the step now says;
- the eighteen grep count reads (the seventeen rewritten lines and Task 10 Step 4's new one), extracted from this plan with their paths re-pointed and run on the final head tree (Task 10 Step 5's on both arms): all nineteen runs printed their count line (`14`, `108`, `356`, `22`, `28`, `25`, `120`, `1`, `35`, `108`, `6`, `1`, `14`, `108`, `2`, `15` passed, `14` on each arm, then `1`: the final tree's counts, larger than a step's Expected where a later task adds tests), while a bare `tail -1` of the same output printed `warnings.warn(` in all nineteen (this round's other pytest sessions were running at the time);
- the lanes' last five green PR runs, re-read 2026-10-09 00:35Z: the same five runs (shard maxima 11 m 47 s, 5 m 29 s, 15 m 02 s, 13 m 55 s; PR Fast Lane 22 m 12 s);
- the known-failure isolation runs (load 17-29), each with the plugin: `test_roleplay_quit_guard.py` alone failed 4/9 on base and 1/9 on head with the arms side by side, then 3/9 and 2/9 one arm after the other, every failure the setup helper's editor-load timeout; the Settings-hub fresh-config subset alone on base gave `13 failed, 8 passed` with 11 `RecoveryRequired: raw_source_selection_changed` lines (and the verifier's 14 `shell` recovery lines per arm all trace to that file); the three `shell` tests alone, 3 runs and then 10 more per arm interleaved: the compact-floor test passed 13/13 on both arms, while the two context-rail tests failed only on head (3 of 26 test runs, none on base), which the flake rule would call a regression, so the first ran alone 20 more times per arm (0 failures on base, 0 on head) and a paired timing probe of its race ran 20 times per arm (marker after the first pause every time; median 199 ms base, 168 ms head, 5,588 and 5,604 stylesheet rules); Task 12 Step 4 records them as not settled.

Not run at `8d502ba250` by either pass: Task 12 Steps 6-8 (the live harness, its captures and PNGs, the owner's answers), Step 9 against an open PR, Task 0 Step 1's `restack.sh` move and Step 5's task start, and anything that needs an owner answer. This round did not re-run the intermediate-state counts of the seventeen re-read commands (it changed how they read the count, not what they run; the verifier's counts above are those steps' Expected values), nor the full paired groups, the mutations, the arrival and poll probes, Task 11 Step 1's minimal venv or `preflight.sh`: none of its changes touches what they run, and the verifier's results above stand for them.

**The 2026-10-08 re-anchor's dry run, on `origin/dev @ a793acbef5`** (dev moved on to `83ecb08c99` while it ran: PR #3031, whose only change under the plan's files is the generated `screen_modal_console_delete_receipt.tcss`, which B1 regenerates and no Old block anchors on; nothing was re-run there). Three detached scratch worktrees under `.worktrees/` (a paired base arm, a head arm and a Task-1-only tree); every file assembled mechanically from this plan's own Create and Old/New blocks, in plan order, by a script that requires each Old block to match exactly once in its file, and the plan's own scripted steps extracted and run with their paths re-pointed (the `_seed_characters` rename, the dead-item deletion, both CSS builds, `ratchet_rows.py`, `preimport_measure.sh`, `preimport_raise.py` with the recorded sign-off, the snapshot refresh, Task 11's census and admission scripts). The final plan was applied twice to fresh head trees; the second build's `test_roleplay_header.py` is byte-identical to the one the fixes were verified on.
- all 95 Old blocks matched exactly once in plan order and the 9 Creates applied (Task 11 Step 3's old Old block matched 0 times on this dev and is now a script); PS 16,528 → 16,525, `roleplay_frame_state.py` 369, `app.py` 5,710 → 5,712 (its row);
- Task 0 Step 1's checks (`TASK-34400`, `TASK-33622.14`, `TASK-26983`, `TASK-27000` Done; `TASK-33790: status: To Do`; `TASK-33790 AC#4: - [x] #4`) and its collision scan (13 of 32 open PRs); Task 0 Step 2's PS check (`PS lines: 16528 ; ratchet row: 16528`, `PS row holds`);
- Task 1 Step 5 on the Task-1-only tree: `All checks passed!`, `6 files already formatted`, `7 passed`, `103 passed`; the four moved helpers' source byte-identical to dev's (AST segments compared);
- boot CSS 607,943 → 607,824 after Task 5 → 607,728 after Task 6 (−119, then −96: −215, as on `8c4dfe59a2`); `widget_defaults_scoped.tcss` 80,168 → 80,166;
- `ratchet_rows.py` twice (`personas_screen.py row 16528 -> 16525 (B1's own -3 lines); app.py row 5712 -> 5712 (B1's own +2 lines); roleplay_frame_state row 369`, the second run changed nothing, format-clean, `6 passed` for the three rows); its raise path with PS padded by eight lines (`row 16528 -> 16533 (B1's own +5 lines)` and the three-line owner-decision comment), restored to the identical file; its STOP pointed at a row red on dev (`STOP: tldw_chatbook/app_lifecycle.py is 2,224 lines on the base arm, over its 2,134 row …`, the ratchet file untouched); the module-size ratchet alone `10 failed, 42 passed` on base and `10 failed, 44 passed` on head, the same ten rows;
- `preimport_raise.py` (`raised: MAX_PASS_ADDED_MODULES 557 -> 558 | ledger row: written`, comment and ledger row dated 2026-10-04 from the sign-off; re-runs changed nothing); pre-import base 557 modules / 416,115 LOC, head 558 / 416,494 (`ccp` 62 → 63 modules, 54,812 → 55,191; the largest route 128,223 on both), the only added module `roleplay_frame_state`;
- Task 11: Step 2 (`census entries: 156`, `OK: 156 Tests/UI files in the PR gate (floor 156)`, `UI Fast Lane shards: 4`, one B1 file in each shard) and Step 3 (`admission-sensitive step: 18 files, B1's four last`), both re-run with nothing changed; Step 1 in the minimal venv: the four census files `125 passed` (`real` 45 s to 47 s, three runs), B1's mounted files alone `110 passed` (243 s, 259 s, 280 s; 260 s hand-timed), and the step as CI runs it `445 passed, 2 xfailed` against `447 tests collected` (`real` 729 s) once the blocked-chip test was fixed (§6; `1 failed, 444 passed, 2 xfailed` twice before it); the lanes' last five green runs (UI Fast Lane shards up to 11 m 47 s, 5 m 29 s, 15 m 02 s and 13 m 55 s; PR Fast Lane up to 22 m 12 s);
- Task 6 Step 7's selections: `35 passed` and the first-time replay `1 passed`;
- Task 12: Step 1's checks (all five agreeing lines; `not B1's: 0`); Step 2's gate lines on both arms (quoted in Step 2's Expected shape: boot CSS 607,943 → 607,728, broad 274 on both, tour sources 41 → 42, UI-ready 1033 in all six boots with identical module sets, boot-import weight 681 on both, pre-import guard and config closure `1 passed` on both); Step 3's ratchet lines (`PS lines 16525 ; row … 16525`, `FS lines 369 ; row … 369`, `4 passed`); Step 4's `css` group paired (`17 failed` on both arms, the same 17: ten ratchet rows, the three timer-path tests, the two governance ratchets, the staleness manifest and the class-level CSS allowlist; nothing new on head; `recovery=0` on both), the timer-path floor (71 unclassified sites on both arms, none new) and the head-only suites (`125 passed` / `125 tests collected`, `110 passed` / `110 tests collected`); Step 5 (`preflight: all derived-artifact checks passed`, `All checks passed!`, `format-checked files: 26`, `26 files already formatted`, `Found 96 errors.` on both arms); Step 10's PR body generated from that evidence (`no MISSING field`; the outcomes lines read `` `MAX_PASS_ADDED_MODULES` 557 → 558 (raised) `` with both LOC limits unchanged, and `` `personas_screen.py` row 16,528 → 16,525 (lowered); `app.py` row 5,712 → 5,712 (unchanged) ``) and Task 13 Step 1's notes generated from the same files (`boot -215 B`, tour `+1`, `+1 module`);
- TASK-34400's two modules: `108 passed` on head and `103 passed` on base (`-n 6`, the bootstrap plugin, `recovery=0` on both).

Not run in the 2026-10-08 re-anchor: Task 12 Step 4's `roleplay`, `shell` and `perf` paired groups, Step 3's arrival and poll probes, Steps 6-8 (the live harness, its captures and PNGs, the owner's answers), Step 9 against an open PR, Task 6 Step 2's failing-first run and its thirteen mutations, the paired checks of Tasks 1, 2, 5 and 6, the mutations of Tasks 2-9, Task 0 Step 1's `restack.sh` move and Step 5's task start, and anything that needs an owner answer. Their Expected lines keep the SHA they were last measured on.

**The rework's dry run (2026-10-04), on `origin/dev @ 8c4dfe59a2`** (TASK-34400's merge `d531297e4f`, PR #2996 and PR #3006; #3006 merged mid-run and touches no B1 file, and the base arm measured identically at `a78a9a900b` and at `8c4dfe59a2`). Two detached scratch worktrees under `.worktrees/` (a paired base arm and a head arm); every file assembled mechanically from this plan's own Create and Old/New blocks, in plan order, by a script that requires each Old block to match exactly once in its file; the plan's own scripted steps extracted from the plan and run with their paths re-pointed (the `_seed_characters` rename, the `_agentic_terminal.tcss` deletion, both CSS builds, `ratchet_rows.py`, `preimport_measure.sh`, `preimport_raise.py` with the recorded sign-off, the snapshot refresh, the census script):
- all 96 Old blocks matched exactly once and the 9 Creates applied, then the whole plan was applied a second time to a fresh scratch worktree and produced byte-identical files. One Old block (Task 6 edit 1, the `Click` import) also occurs in `personas_preview_controller.py`, where a first pass put it; Task 6 Step 4 now names PS as every edit's file. PS 16,528 → 16,525, `roleplay_frame_state.py` 369, `app.py` 5,710 → 5,712 (exactly its row);
- `ruff check` on every touched Python file but `app.py`: `All checks passed!`; `ruff format --check` on the 26 that were format-clean at base: `26 files already formatted`; `app.py`: `Found 96 errors.` on both arms; `build_css.py` and `test_css_build_integrity.py`: the same `ruff format --diff` size on both arms;
- Task 6 Step 2's failing-first run, on a third scratch worktree holding only Tasks 1-5 and Task 6 Step 1: `62 failed, 33 passed`;
- the fast files: `44 passed` (44 collected; 12.5 s serial in the dev venv); the mounted files with the bootstrap plugin (harness 14, header 67, journeys 2, hostile names 27, hostile text surfaces 81): `191 passed` in one `-n 6` run, after Task 8's test was fixed (its first version switched the source to "server" before editing, where editing is a local-only action, and failed 5/5 on `_edit_mode`); Task 6 Step 7's selection: `35 passed`; the console-glyph and rail-row suites: `356 passed`;
- TASK-34400's two modules on head: `108 passed` (22 + 5 + 81: Task 1's helper move and Task 6's two re-pins kept the existing 103 green); Task 8's mutations `status-unescaped` and `item-label-markup`: `5 failed` each, restored `5 passed`; the new test finds each name painted exactly five times;
- the changed existing test files on both arms (failure-set diff): the eight UI files with the plugin, `2 failed, 635 passed, 6 errors` on base and `2 failed, 642 passed, 6 errors` on head, the same eight (the six `TestPersonaHumanIdentityRemoval` setup errors, `test_floating_buddy_close_refreshes_active_personas_inspector`, `test_resize_sync_skips_work_when_compact_state_is_unchanged`), no new one, `recovery=6` on both; the glyph, theme-contrast, module-size-ratchet and pre-import files without it: `9 failed` on both arms (the nine pre-existing ratchet rows), no new one; `Tests/Architecture/test_module_size_ratchet.py` alone: `9 failed, 39 passed` on base, `9 failed, 41 passed` on head; the design-system contract (Task 10 Step 5): `14 passed` on both arms;
- the evidence scripts: `ratchet_rows.py` twice (`personas_screen.py row 16528 -> 16525; app.py row 5712 -> 5712; roleplay_frame_state row 369`, the second run changed nothing, the file format-clean, `6 passed` for the three rows); the raise (`raised: MAX_PASS_ADDED_MODULES 557 -> 558 | ledger row: written`, the snapshot refreshed, the guard `1 passed`, and Task 11 Step 4's re-run the same); Task 11's census script twice (`census entries: 141`, the checker `OK: 141 Tests/UI files in the PR gate (floor 141)`, idempotent; shard 1 takes two of B1's files and shard 2 one);
- the gate lines (Task 12 Step 2's commands, both arms, same session): base `boot-css=607326 | [census] total=274 | CSS sources after a full tour: 41 | ui-ready boots: 1033/1033/1033 | boot-import-weight: 681/686 | 557 modules / 415369 LOC | ccp 62 mods 54812 LOC | preimport guard: 1 passed | config-closure: 1 passed`; head `boot-css=607111 | 274 | 42 | 1033/1033/1033 | 681/686 | 558 / 415748 | ccp 63 / 55191 | 1 passed | 1 passed`; UI-ready module sets identical on both arms; boot CSS 607,326 → 607,207 after Task 5 → 607,111 after Task 6 (−215); `widget_defaults_scoped.tcss` 80,168 → 80,166 B; the only added pre-import module `roleplay_frame_state` (+379 lines on the pass and the `ccp` route, the largest route unchanged at 127,526), so only `MAX_PASS_ADDED_MODULES` (557) is exceeded;
- the open-PR collision scan (Task 0 Step 1's script, the revised file list): 11 of 28 open PRs touch a B1 file.

Not run in the rework (outside its brief): the full paired Roleplay, shell, css and perf groups, `preflight.sh`, the live harness and its PNGs, the arrival and poll probes, the minimal fast-lane venv, Task 6 Step 10's thirteen mutations and Tasks 2, 3, 4, 5 and 9's (their code is unchanged since the `83c264f286` and `bd41347b65` dry runs below), and anything that needs an owner answer.

**The previous dry run, kept for its history** (superseded wherever the rework measured again): on `origin/dev @ 67fc531047` (B0's merge `83c264f286` plus one docs-only PR, #2982, which touches no B1 file), extracted with `git archive` into a scratch tree outside the repository (no `Docs/superpowers`, `Docs/Design` or `qa`), with every file assembled mechanically from this plan's own Create and Old/New blocks, in plan order, by a script that requires each Old block to match exactly once (target chosen by the block's own text, ties broken by the path the step names):
- all 88 Old blocks (the first draft's 75 plus 13 from the review round) matched exactly once, beside the 10 created files and the scripted `_agentic_terminal.tcss` deletion; every one of them also matched on `83c264f286` before the dry run (PR #2993's formatter left the anchors byte-identical). PS measured 16,525 → 16,517 (−8: edit 17 drops the `WorkbenchHeaderState` import that edit 6 leaves unused, found by `ruff` in this run), `roleplay_frame_state.py` 369;
- `ruff check` on every touched Python file: clean, and `app.py`'s 96 pre-existing findings (94 `E402`, 2 `F401`) unchanged; `ruff format --check` on the 24 touched files that were clean at base: clean (one header-test line was reflowed into the plan from the formatter's diff);
- the fast files (frame state 28, FittedText 5, stylesheet 11): 44 passed, 44 collected; the mounted files (harness 14, header 67, hostile names 20, journeys 2): 103 passed in one `-n 6` run (2 m 42 s); Task 6 Step 2's failing-first run on a tree holding only Tasks 1-5 and Task 6 Step 1: `62 failed, 33 passed`; Task 6 Step 7's selection: 35 passed; the first-time replay re-pin: 1 passed;
- boot CSS 607,951 → 607,832 after Task 5 → 607,736 after Task 6 (−215); `widget_defaults_scoped.tcss` 80,168 → 80,166; pre-import 557 → 558 modules, 415,329 → 415,708 lines (`ccp` 62 → 63, 54,752 → 55,131; the largest route unchanged at 127,522), the only added module `roleplay_frame_state`, so only `MAX_PASS_ADDED_MODULES` (557) is exceeded;
- base-arm widths on both styled tiers: work pane 58/78/108, Inspector 30/39/54 (spec §5.3's "today" row), pinned by the new width test;
- the CSS group (eight suites, with the theme-contrast guard): the same 13 failures on both arms (the 13 Task 5 Step 5 names), no new one; the timer-path census: 67 unclassified sites on both arms, none new (the floor's grep works on dev's message format);
- the quit guard, its choke-point scan, the Inspector pane and the preview suites on both arms: the same failures (8 in `test_roleplay_quit_guard.py`, 1 in the Inspector pane) except `test_a_vanished_roleplay_quit_prompt_means_stay[question]`, which fails alone on BOTH arms (3/3 on base) and passed on base under `-n 6` by chance: pre-existing;
- named mutations run and red as stated: `chip-on-has-unsaved` (7 failed), `kind-gap-drift` (2), `item-not-flexible` (16), `go-marker-unresolved` (9), `compose-ready-default` (2, with the third selected test, `test_status_names_the_data_source_and_never_says_ready`, still passing), `aggregate-walks-dom` (1), `guard-on-is-clean` (1), `inspector-wider` (6), `tag-label-markup` (5), and Task 7's source deletion (6: `assert 6 == 1`, `assert 9 == 3 + 1`, the first character at y 23); each restored green;
- the evidence scripts, extracted from this plan: `ratchet_rows.py` run twice (`PS row 16525 -> 16517; roleplay_frame_state row 369`, the second run changed nothing, the two ratchet rows `4 passed`, the file format-clean); Task 11's census script run twice (`census entries: 129`, the checker `OK … floor 129`, idempotent); the poll probe on both arms (Characters: base 53-57 µs, head 113-115 µs a tick; Lore: base 1 µs, head about 60 µs; 12 and 300 rows); the arrival probe's new column (`nav_module=False` on base, `True` on head); the PR body's ledger generator on the dry-run numbers; `pr_collisions.sh` live (12 of 26 open PRs touch a B1 file); every helper `bash -n`, `py_compile` and `ruff` clean.

Earlier dry runs, kept for their history: on `f0a766a155` (the draft's Expected lines: the remaining mutations, the paired Roleplay and shell groups, the minimal fast-lane timing), on `bd41347b65` (75 Old blocks, PS 16,397 → 16,388 before #2993's reflow, the arrival probe, the capture checker on a synthetic capture) and, in review, on `7495abd13f` (the broad census 274 on both arms, the design-system contract `14 passed` with the plugin, the paired groups' known failures and load flakes).

Not run in that dry run: the full paired Roleplay, shell and perf groups, the broad-selector, tour and UI-ready censuses, the live harness captures and PNGs, `preimport_raise.py` (it needs git arms and an owner sign-off), the minimal fast-lane venv, `preflight.sh`, and anything that needs an owner answer. The design-system contract errored identically on both arms of the scratch tree only because it lacked `Docs/Design` and `Docs/superpowers` (review measured `14 passed` on both arms of a real worktree).

**6. Defects found and fixed.**

In the first draft:
- The plan was truncated after Task 11: Tasks 12 and 13 and this review were missing, while earlier tasks already referred to Task 12 Steps 1, 4 and 8 and to Task 13.
- B0 was rebased after B1 was cut, so a plain rebase would have replayed B0's 22 old commits; the move became `--onto` from a recorded cut point.
- The chips used decorative hues that fall below AA on the panel in 29 and 40 of the 70 themes; they now use `$ds-status-warning-readable` (new) and `$ds-status-error-readable`, pinned on every theme.
- Three PS deletions and the `_workbench.tcss` deletion left an extra blank line, and two test-harness deletions relied on invisible trailing blank lines: all six are anchored on the following line.
- Nothing pinned the resize path: `test_a_resize_refits_the_header_without_gathering_inputs` now does, with two mutations.
- Task 5's lint included `app.py`, whose 96 pre-existing findings made "All checks passed!" impossible; `app.py` is compared across arms.

In the second review round (applied 2026-10-04, with B0's merge):
- Performance: the poll gathered the header inputs on every tick through a `query_one` that walked the whole DOM while the editor was unmounted (about 4 ms a tick). Edit 16 reads the cached editor; `test_gathering_header_inputs_walks_no_dom` pins it; Task 12 Step 3 measures the per-tick cost on both arms and the PR body quotes it.
- TASK-33622.14's guard decided on a bare `is_clean`: Step 4b routes its three decisions through the predicate, with a pin, a mutation and the ADR-046/ADR-120 wording.
- A hostile tag still parsed as markup on the Tag filter button (a bug on dev then; TASK-34400 has since fixed it on dev, and the rework dropped this plan's copy of the fix).
- The first-time replay test outside every suite required the retired subtitle: re-pinned and added to the `roleplay` group; the guide's title drops the same copy.
- The persona-save `finally` repainted before resolving the save's completion (a repaint error would have hung a guard awaiting it); the Old block now carries the `set_result` line, so the New block cannot leave a stray copy behind (the review's suggested New block would have).
- `fit_header_item` raised on `FittedText`'s empty default value if anything painted before the first push (a defect in this plan's own draft code, never on dev); the fit now takes it.
- The header composed with the shared "Ready" status and relied on mount order to replace it; it composes with `initial_header_state` (and drops the now-unused import).
- New mounted geometry had no red run: Task 6 Step 10 gained six mutations and Task 7's deletion run covers the AC#2 tests; the worst-case and chrome-cell tests run under both tiers and use `assert_painted_inside`; the parity test gained the shared-visual and visual-operation cases; a width pin covers spec §5.3's "unchanged" row.
- The full styled tier's spec deviation, the 65-column floor, the escaper, the shared-token claims and the size-matrix mark were written down only in passing: §7 and the PR body now carry them, and Task 13 hands the interims to B2a, B3 and B12.
- Evidence tooling: the ratchet-row script and the census comment strip were not idempotent; the raise script wrote the base arm's value back over an upstream tightening; the cut point had no guard against foreign commits; the PS precondition was stricter than the ratchet; the prefix audit matched its own non-compose sources; the design-system check ran without the plugin and compared two error sets; the PR body's static ledger rows could ship stale; the open-PR watch list was hard-coded and stale; several suites that exercise B1's files were in no paired group, and the red-on-dev timer census had no floor. Each is fixed where it lives.
- Stale facts: the broad census is 274 on both arms (zero headroom), B0's head moved twice and then merged, #2993 reformatted PS (row 16,525) and made TASK-26983/27000 Done, and `test_navigation_signals_and_drains_pack_creation_before_continuing` and `test_a_vanished_roleplay_quit_prompt_means_stay[question]` are isolation-dependent pre-existing failures.

In the 2026-10-04 rework (the owner's rulings, TASK-34400 on dev, dev at `8c4dfe59a2`):
- Ruling 1: Task 0 Step 6 records the owner's pre-import sign-off ("Expand it", 2026-10-04) in `$EV/owner-signoff.txt` instead of stopping to ask; the only pre-import STOP left is `preimport_raise.py`'s, for growth beyond the bound. The screenshot STOP (Task 12 Step 8) stays.
- Ruling 2: the Global Constraint "PS net ≤ 0 lines" is gone. `ratchet_rows.py` sets the PS row to the measurement in either direction (a raised row carries a dated owner-decision comment naming TASK-33910.2) and holds `app.py` the same way (B1's two lines now land exactly on its 5,712 row). Edit 12 no longer folds the purpose line's `if`/`elif` count chain into a dict: that reshape existed only to save lines; the chain stays and only the descriptor and the format move. Edit 16's comment sits on its own line instead of trailing the `try:`.
- Ruling 3: Review Focus 1, 2 and 6, the R33 constraint, edit 16's note, Task 8, the commit messages, the PR body and the notes say which bugs exist on dev (none of the hostile-name ones: TASK-34400 fixed them before B1) and that B1 fixes none of them; the per-tick DOM walk and the post-save repaint are B1 design points, not dev bugs.
- TASK-34400: Task 8 no longer edits the Inspector pane, the library pane or `_notify`, and no longer creates `test_roleplay_hostile_names.py`: it extends it with `test_the_header_item_label_and_server_label` (B1's item label and server label, styled tier 1). Task 1 moves the four paint helpers into the harness as their one home, unchanged (byte-identical, docstrings included, since the 2026-10-08 rework), retires `_seed_characters` in favour of the harness's `seed_mock_characters` (the same patches), and both hostile-text modules import them from there. Task 6 Step 7 re-pins the two TASK-34400 tests that asserted the retired subtitle (the `Editing <name>` paint and `_header_subtitle_text`), and edit 9's Old block carries TASK-34400's `escape_markup` lines. Task 13 no longer notes TASK-33790 (TASK-34400 ticked its AC#4). Task 4's style floor drops the two modules B1 no longer edits; the collision scan's file list follows.
- Dev changes under the plan: the UI Fast Lane runs the census as round-robin shards (TASK-34353) and the census holds comment lines in run order, so Task 11 appends B1's three files under one comment line (the old script re-sorted the census and refused comment lines) and a shard nearing its cap gets one more shard, the workflow's own rule, instead of an owner question about minutes; #2953 (the `_agentic_terminal.tcss` re-indent), #2992 and #3001 merged, and the nine dead items were re-verified on dev (same counts, still unreferenced); `_variables.tcss` and the generated sheets moved under Task 5 without moving its anchors.
- Every number was re-measured on paired arms at `8c4dfe59a2` (§5).

In the 2026-10-08 rework (dev at `a793acbef5`; a rulings-and-coverage review's twelve findings, all applied, and what the re-anchor's dry run found):
- Ruling 3, still unmet in three owner-facing places: the selection-guard interim now says the gap exists on dev today (`_confirm_discard_unsaved` checks only the form and visual authoring) and that B5b-1, not B1, fixes it; the below-65-column clipping says it is new with B1's one-row header and not fixed by B1 (B7, TASK-33910.24); the PR body says the "Ready"/"Blocked" badge is on dev today (RP-067) and that B1 fixes the header's part. Applied in the PR body, the task notes and §7.
- "B1 changes none of TASK-34400's production edits" was wrong: edit 9 deletes `_header_subtitle_text` with TASK-34400's two `escape_markup` calls. Global Constraints, Review Focus 1, the PR body and the notes now say B1 removes the old subtitle and that escaping with it, and that every other TASK-34400 fix is unchanged.
- Task 1 claimed the four paint helpers move "unchanged" while three docstrings had been edited: they are now dev's byte for byte (checked by AST), the move note is a comment, and the `_seed_characters` claim says what is true (the same patches, not the same text).
- The PR body named the owner's rulings but not their outcomes: it now prints, from both arms' files, which pre-import constant rose and that PS's row was lowered, and says the card's "≤16,436" gate is superseded; the notes repeat it from `$EV/outcomes.txt`.
- TASK-33790 (the B1 task's listed prerequisite, still To Do) is checked in Task 0 Step 1 (its status and AC#4) and explained in the PR body and the notes.
- `ratchet_rows.py` booked "+N lines" as head minus the base ROW, so a rebase onto a dev whose row was red would have raised it in B1's name: it now reports B1's own delta against the base arm's line count and STOPs on a row that is red at base (both paths exercised).
- `preimport_raise.py` dated its comment and ledger row with the day it ran: it now takes the sign-off's date, so a later re-run rewrites nothing.
- The notes typed `−215 B`, `+1` and "within 1 ms": they are generated from `gates.txt` and the pre-import JSON.
- The Console ASCII-glyph side effect was called "accepted" with nobody's acceptance: it is now "new with B1, not on dev", and Task 12 Step 8 puts it to the owner with the screenshots.
- Review Focus 1's re-pinned `test_the_header_view_keeps_an_unsaved_item_out_of_markup` lived in a file no PR lane ran: Task 11 adds `test_roleplay_hostile_text_surfaces.py` (81 tests, no `bootstrap_profile` mark, about 37 s serial in the minimal venv) to the UI Fast Lane census.
- §1's close-out row pointed at a Step 2b that does not exist.
- Task 11 no longer matched dev: the UI Fast Lane has four shards, not two (the scripts now read the count from the workflow), the census holds 152 files under a floor of 151, and the admission-sensitive step lists fourteen files, so Step 3's Old block matched 0 times. Step 3 is now an idempotent script that inserts B1's four files before the step's `--timeout=300` line; Step 1 reads dev's list from the workflow and times B1's files alone and the step as CI runs it.
- Found by the re-anchor's dry run: (a) Step 1's `{ /usr/bin/time -p … 2>&1 | tail -1; }` fed the timing report into the pipe, so it printed only `sys …`, never `real`; it uses bash's `time` keyword now, and greps the count line and every failed test. (b) Run as CI runs it, after dev's fourteen admission files in one pytest process, the step read `1 failed, 444 passed, 2 xfailed` in both of two runs, and the second run named `test_the_blocked_chip_passes_the_leave_guard` (`timed out after 20.0s waiting for the blocked chip`), while B1's four files alone passed (`110 passed`, four times) and dev's fourteen alone passed on the base arm (`335 passed, 2 xfailed`). One file per run against that test isolated the cause: only after `Tests/UI/test_console_runtime_ownership.py` did it fail. The full app there read a ready provider, so the chip never showed; with the blocked state set on the screen's readiness seam, the chip showed but `pilot.click` returned False, because a chip that appears only then is not laid out yet. The test now sets the seam, waits for the chip's painted width and settles before the click: `84 passed, 1 xfailed` for that file plus the test, and Task 11 Step 1's combined run passes (§5). Whether the cause is state that test file leaves behind or the order alone was not investigated further; it is a dev file B1 does not touch. (c) Task 5 Step 4's deletion block ended with exit 1 on success (a no-match `grep` under `pipefail`). (d) Under concurrent pytest sessions a `| tail -1` can show `warnings.warn(` instead of the count (Conventions now says how to read it).
- Dev changes under the plan: a tenth module-size ratchet row is red on dev (`Chat/console_interrupt_rounds.py`); the collision scan lists 13 of 32 open PRs (#3045, #3023, #3036, #3028, #3022 and #3039 are new).

In the second 2026-10-08 round (dev at `8d502ba250`; the independent dry run's six minor findings, each applied, one with corrections, and re-run there, §5):
- Figures anchored on `a793acbef5` had gone stale when PR #3028 merged: dev's boot CSS rose 134 B to 608,077 (13 B under the never-raised 608,090 ceiling), the pre-import pass to 416,217 lines and its largest route to 128,281, the PR-gate census to 154 files, and the open-PR list to 11 of 30. B1's deltas and every gate verdict were unchanged, and every script reads live values, so no step changed behaviour; every stated figure is now anchored on `8d502ba250` (the header, Global Constraints, Tasks 0, 4, 5, 6, 8, 11 and 12, and §5), the older SHAs' values kept as history. Task 0 Step 1's collision text drops #3028 and #3031, Task 12 Step 1 says a cut older than `8d502ba250` rebases over #3028 (so Task 11 Steps 2-4 re-run there), and Step 9's list of PRs to watch no longer names #3028.
- Seventeen pytest commands (Tasks 1-11) read their gate count with a bare `| tail -1`; under load three printed `warnings.warn(` instead (pytest's `rm_rf` teardown warning). All seventeen now pipe through `grep -E '[0-9]+ (passed|failed|error)' | tail -1`, as Task 12 already did, and Conventions states the rule (the failing-first runs keep `| tail -3`, which shows the error text their Expected lines name).
- Task 10 Step 4 said `Settings ›` and the other header strings "are emitted by `roleplay_frame_state.py`" and told the executor to `grep -rF` them, but the composed strings exist only as `resolve_glyph` f-strings, so the grep finds nothing and the checker lists them as not emitted. The step now counts the six literal fragments in the module and runs the four mounted tests that assert the painted composed strings (`15 passed`).
- The known-failure notes undercounted, and two of the finding's own claims did not hold on re-run, so it was applied with corrections. (a) The quit-guard file does not simply fail 9/9 on dev: it is load-sensitive on both arms (9/9 and 9/9 in the verifier's session; 4/9 and 1/9, then 3/9 and 2/9 in this round's), every failure read so far being the setup helper's editor-load timeout, so Task 6 Step 8 and Task 12 Step 4 now say any of its nine tests can show as new and how to tell a B1 failure from it. (b) The `shell` group's `recovery=14` is not load: the same Settings-hub tests raise `RecoveryRequired` on both arms (11 lines with the subset alone on base), so the suggested "re-run at lower load" was rejected and Global Constraints and Task 12 Step 4 now say when that count is acceptable. (c) The Schedules compact-floor test is listed as intermittent on dev, as suggested. (d) The two Console context-rail tests the finding called load flakes failed only on head in this round's isolation runs; more paired runs and a timing probe found no B1 effect, but Task 12 Step 4 records them as not settled rather than as dev flakes.
- §5 gave UI Fast Lane shard 2's maximum as 5 m 42 s; Task 11's own text and the GitHub run data say 5 m 29 s (fixed).
- `preimport_raise.py` wrote the whole of `owner-signoff.txt`, its date and the question included, into the ADR-097 row as "Owner, verbatim", with nested double quotes. It now quotes only the owner's answer (the sign-off's last quoted string) as verbatim, paraphrases the question around it, and STOPs if the sign-off has no quoted answer.

**7. Spec items not mapped to code in B1, by design or as deviations (each recorded in the PR body and the task notes):**
- `-residual` (spec §2.12 item 2's list for Roleplay's copy): no element carries it until B7's residual work strip, and the Library has no such rule to keep parity with, so a B1 rule would be dead and untestable. B7 adds it (its card names it), under a Roleplay prefix rather than a bare shared claim.
- The interim `› <item>` is not inside the `DestinationHeader` subtitle string, as §1.3 and §5.3 word it, but in a literal `FittedText` right after the subtitle: it paints the same `Roleplay  <Kind> › <item>`, keeps the kind uncut, ends long names in the resolved ellipsis (CSS's ellipsis ignores ASCII mode), and keeps untrusted names out of the shared markup-on header until FU-2 (TASK-33910.20).
- The Unsaved chip's colour is `$ds-status-warning-readable`, not the `$ds-status-warning` §1.3 names: same hue family, readable (above).
- AC #2's "row 19" holds at 36 rows and more; at 80x24 the first item is on row 18 because the compact workbench is one row shorter there (measured on both arms); the test asserts the band relative to the measured header.
- The blocked chip has no keyboard path until B3's Tab region (an interim; TASK-33910.5 gets the note, including its `LEAVES_SCREEN_IDS` entry).
- Exists on dev today, not fixed by B1 (B5b-1 fixes it): selecting another item checks only the form and visual authoring (`_confirm_discard_unsaved`), so with a staged avatar or a save in flight a selection change does not ask. B1's new chip shows in those states, so the chip and that guard can disagree until B5b-1; the list rows' unsaved badge still follows `has_unsaved_changes`, as on dev. The ADR-046 amendment says exactly which domains the aggregate tracks.
- The styled full tier is seeded through the same character seams as the mock tier, not "a temporary ChaChaNotes seeded through the app's own APIs" (§5.7.1). Converting it means re-verifying every full-tier test against a real DB, which B1's ACs (styling, routing, the chip) do not need; the journey instead checks the payload that reached `update_character`. B2a, the first slice with volume tests, converts it (TASK-33910.3 gets the note), so save → reload persistence is first proved there.
- New with B1, not on dev (dev's five-row header has no single row to clip): below 65 columns the worst-case one-row header (every chip, a server label, the longest kind) does not fit and its right end clips. B1 does not fix this; its matrix starts at 80x24; §5.7.1's 60x24 degrade check and the spec's below-64 layout belong to B7 and TASK-33910.24. No new degrade step (such as a shorter blocked chip) was invented, because the spec defines none.
- R33's escaper is `Utils.input_validation.escape_markup`, not the `textual.markup.escape(text)` the spec names: on Textual 8.2.8 `escape('[/')` still raises `MarkupError` and `'[TODO] y'` renders as `' y'`. The hostile-name fixture pins the choice; the spec's R33 wording should be corrected when it is next touched.
- The Roleplay split claims four shared tokens beyond R18's prefix list (`workbench-header-title`, `-subtitle`, `-status`, `-active`), so its copies of the header and grip rules leave boot; legal only under the anchor guard (`test_every_roleplay_selector_carries_a_roleplay_token`).
- The card's "size-matrix fixture" is a parametrize mark (`size_matrix`), so test modules need no fixture import.
- The guide's screenshots predate B1 and stay stale until B12 (TASK-33910.18 gets the note).
- New with B1, not on dev, outside Roleplay: with ASCII glyphs on, Console text that goes through `resolve_glyph_text` (staged file names, Inspect rows) also rewrites the new frame glyphs. Nobody has accepted it on the owner's behalf: Task 12 Step 8 asks the owner with the screenshots.
- Spec §5.10's PERF-22 deterministic counts are B2a's and B6's; B1 records the first-visit arrival evidence its lazy sheet affects and the poll's per-tick cost. No ADR-011 capture: §5.4 item 5 does not list B1, which retires no legacy path and adds no worker or timer (the existing poll does more per tick, measured).
