# PR3334 backlog integration plan (2026-11-01)

## Binding execution contract

Task 1 is one coupled integration: rebase the selected B2b PR first, preserve both histories, implement the bounded-sniff and marks-hint repairs below, qualify the affected behavior/artifacts, then commit and hand off one complete branch for independent review. The numbered tasks in the historical preparation below are substeps for this one owner, not parallel writers. The current task/spec/ADRs and this contract override stale source-era queue, base, ratchet and publication clauses.

Selected source:30c14c310d10fa2bb95ff11f09f3f9bb82a719e7; original base804dc00231fb95eb7fe952d121e126d51b32d782. Actual integration base:62a169f906508ed02bd7546ab278d916638d1607, confirmed merge of PR3333. Worktree:C:/Users/GDesktop-1/.codex/worktrees/pr3298-resync/tldw_tui; branch codex/pr3334-resync. Live selection:OPEN, non-draft, unarmed,0 threads. Rebase onto this exact origin/dev before repairs; never merge dev into the branch or copy an old source snapshot over current owners.

The user explicitly authorized hosted-CI and numeric-size waivers, focused local verification and superpowers:requesting-code-review instead of Qodo for every eligible PR. Root owns all remote actions and exact-head publication, including disabling auto-merge if needed, exact force-with-lease and protection restoration. The implementer performs no remote mutation or subagent dispatch. Preserve product/security/source/custody guards and numeric thresholds; an authorized numeric exception is WAIVED, not PASS. No full suite or required-workflow dispatch.

Never act on PR3267/3297, promote or merge drafts, perform releases/main work, or modify the main checkout. Release PR3245/3246 remain outside this session even if their status changes. No stashing/shared-work destruction. Reuse the existing main .venv only as interpreter; private tests use native fixture ownership. This isolated worktree is reused; do not create another worktree for routine execution.

Read relevant task/spec/ADRs and UI design-language/component-patterns before modification. Follow .superpowers/preparations/pr3334-integration-preflight.md and backlog-targeted-verification-guidance.md. Select the affected verification pool; do not run every neighboring Roleplay/UI module. Preserve current CSS inputs/plugins/tokens/dialog owners and regenerate bundles only through the canonical generator when required by changed inputs. Await and record every writer/generator/formatter exit before frozen tests; inspect session_id and exit_code, never only output.

ADR required:no new ADR. Existing011/031/097/150/161 govern this existing advisory import and truthful hint contract. Do not introduce a new storage/provider/security/UX boundary or broaden B3 Escape behavior. Keep task completion limited to demonstrated ACs; broader existing gaps remain open.

## Historical detailed preparation, subordinate to the contract above

# PR #3334 B2b Repair and Rebase Plan

> Preparation only. Execute in an isolated PR #3334 checkout after the current workspace is free. No implementation, tests, or ref mutation were performed while preparing this plan. Use systematic debugging, test-driven development, verification before completion, and requesting-code-review at execution.

**Goal:** Preserve B2b's bounded import routing for valid files whose deciding evidence occurs after the 64 KiB sniff, make the marks hint truthful under an active filter, then rebase the reviewed PR onto current `dev` without losing either side's independent work.

**Source:** `../pr3334-initial-review.md`, `../pr3334-source-review/` at PR head `30c14c310d10fa2bb95ff11f09f3f9bb82a719e7`, base `804dc00231fb95eb7fe952d121e126d51b32d782`. The source review reports C0/I1/M1; it did not execute tests or render the UI. The current live base and queue mode must be checked anew during execution.

**Architecture:** Keep the current `sniff_import_file` stat-before-read cap and one read of at most `SNIFF_MAX_BYTES`. Thread an EOF/truncation fact into `classify` so readable text with no marker in an incomplete prefix asks for a kind, while complete negative evidence and malformed/deeply nested JSON retain their refusal. Reuse the pane's existing `_filtered` state to suppress the marks bar's `esc` suffix while Escape clears search/tag first; the Clear button remains a marks action.

**Tech stack:** Python 3.12+, Textual 8.x, pytest, existing pydantic-core parser; no new dependencies.

**Spec:** `Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md` §1.4.6, §1.5.5, §4.5, §5.2; task `backlog/tasks/task-33910.4 - Roleplay-frame-B2b-one-row-toolbar-New-chooser-Import-picker-search-aware-empty-states.md` AC#4/#6 and existing `task-33910.47` AC#1. Existing B2b plan: `Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md`.

**ADR required:** no. **ADR path:** N/A. **Reason:** The repair restores the existing bounded advisory sniff and truthful Escape copy within established import/UI boundaries. Relevant existing decisions: ADR-011 (workers), ADR-031 (truthful hints), ADR-097 (boot ratchets), ADR-150/161 (visual values and patterns). No schema, contract, dependency, or durable UX structure changes.

## Global constraints

- Preserve the 10 MiB card/lore/dictionary size refusal before any read, regular-file guard, pack extension routing, and one read of at most 65,536 bytes. Preserve no archive open or pixel decode.
- Incomplete evidence means an `ask` verdict with `TEXT_KINDS`; complete text with no Roleplay markers may remain `not_roleplay`. A 9 MiB deeply nested JSON must still return `not_roleplay` without crashing.
- Keep the first Escape under active search/tag clearing that filter and retaining marks. Keep the Clear button clearing marks. At rest without a filter, `Clear esc` remains accurate.
- Follow the existing B2b owner decisions and task AC; no broad source refactor, new CSS value, or full-suite run. Rebuild generated CSS only if the rebase changes its source inputs.
- Read `backlog/docs/lessons-testing-evidence.md`, `lessons-live-verification.md`, and `lessons-backlog-hygiene.md` during execution. Use a private app profile before any app imports or mounted tests.

## Review focus

1. A `.json` file with more than 64 KiB of leading whitespace, below 10 MiB, must ask rather than refuse.
2. A Markdown or other card-text file with its valid fenced JSON card after byte 65,536 must ask rather than refuse.
3. The same prose-only file that reaches EOF within the read must still refuse; exactly 65,536 bytes at EOF must not be treated as truncated.
4. Existing malformed/deep JSON and binary text controls must still refuse without parser crash or excessive read.
5. With marks and an active search or tag, the bar must omit the false `esc` hint; after clearing the filter, it may show `Clear esc` and the next Escape clears marks.

## Task 1: Repair the bounded sniff

**Files:** Modify `tldw_chatbook/UI/Persona_Modules/roleplay_import_sniff.py`; test `Tests/UI/test_roleplay_import_sniff.py`.

**Interface:** Keep `sniff_import_file(path, *, max_bytes) -> SniffResult`. Extend `classify(suffix, prefix, *, truncated=False) -> SniffResult` (or equivalent explicit EOF flag) without changing existing direct callers. Determine `truncated` from the already obtained stat size and actual prefix length; never read an extra byte. A concurrent short read is inconclusive.

- [ ] Add failing file-level tests for a late fenced JSON card in `.md` and one non-Markdown card-text suffix, and leading whitespace before a valid JSON card in `.json`; each file is over 65,536 bytes but under 10 MiB and yields `("ask", TEXT_KINDS)`. Include a 65,536-byte complete prose control and a 65,537-byte prose control to pin the EOF distinction.
- [ ] Run only these nodes and confirm the late-fence/whitespace cases fail for `not_roleplay` on the pre-fix branch. Record the source hash and exact failure lines.
- [ ] Pass the truncation fact into the classifier. At the no-marker Markdown/card-text returns, ask only if the prefix is readable UTF-8 and truncated. For JSON leading whitespace, ask if the truncated prefix contains only whitespace; do not convert pydantic recursion/deeply nested errors into `ask`. Preserve prior JSON partial-object `ask` behavior.
- [ ] Run the new nodes, then the whole `Tests/UI/test_roleplay_import_sniff.py` file on the frozen repaired tree. Confirm the existing 11 MiB no-read, 64 KiB read limit, pack, binary, malformed/deep JSON, and UTF-8-cut cases still pass.

## Task 2: Correct marks copy under filter precedence

**Files:** Modify `tldw_chatbook/UI/Persona_Modules/roleplay_frame_state.py` and `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py`; test `Tests/UI/test_roleplay_item_list.py` and one focused mounted Escape case in `Tests/UI/test_roleplay_item_list_app.py`.

**Interface:** Add a keyword-only `filtered: bool = False` to `marks_bar_view(...)` (or an equivalent explicit state argument); pass the pane's existing `_filtered` value. Keep its action tuple and layout degradation unchanged. The wide `clear` label is `Clear esc` only when no filter is active, otherwise `Clear`.

- [ ] Add a pure copy check for both active and inactive filter at wide width, including `hidden == 0` while a filter is active, and the existing narrow degradation/width assertions. Add or extend one mounted test to show a filtered marked list paints `Clear`, Escape clears search/tag while marks persist, then the bar paints `Clear esc` and the next Escape clears marks.
- [ ] Run the new nodes on the pre-fix branch and confirm the false-hint failure.
- [ ] Implement the label selection using the passed filter state; avoid any new key binding or CSS.
- [ ] Run the new nodes plus existing marks-bar and Escape focused nodes. Update `task-33910.47` AC/notes/status only after the combined behavior is verified; leave B3's wider Escape work intact.

## Task 3: Rebase and preserve both histories

**Files:** Resolve only actual conflicts in the 19 overlap files listed in `../pr3334-source-review/current-dev-overlap-stat.txt`; see report for ownership map. Do not merge `dev` into the feature branch.

- [ ] Check exact checkout and clean state, PR #3334 current head/state, live `dev`, queue mode, and auto-merge state. If auto-merge became armed, disable it before any push. Capture pre-rebase commit list and file/blob inventory. Do not touch drafts #3267/#3297, the main checkout, or release.
- [ ] Rebase the PR branch onto current `origin/dev` (or use the documented `gh pr update-branch --rebase` path when queue is off/dry). Resolve conflicts by preserving current `dev` workflow/census/diagnostic owners and all independent B2b product/test behavior. Preserve each branch's unrelated ratchet changes and ADR-097 ledger entries. Never re-pin a ratchet to a stale count.
- [ ] Regenerate CSS with `python tldw_chatbook/css/build_css.py` from source modules. Check generated Roleplay rules plus current `dev` Plugins split, tokens, `ConsoleCodeReviewModal`, and Library folder dialog rules all remain. Do not hand-edit generated bundles.
- [ ] Compare pre/post B2b commits, changed-path inventory and reviewed source blobs; explain any commit Git drops as already in `dev` or explicitly restore its behavior. Validate the 19 overlap files and post-rebase task/ADR/census records line by line where both arms edited the same region.
- [ ] Push only the rebased PR #3334 branch with `--force-with-lease` after local qualification. The merge/auto-merge decision remains with the parent workflow and must obey current Qodo/head/thread rules.

## Task 4: Focused qualification and review

- [ ] Run targeted PR tests: `Tests/UI/test_roleplay_import_sniff.py`, `test_roleplay_import_app.py` (dirty draft, draft during slow import, newer selection/cursor/search, clean cross-kind landing, refresh race), New chooser availability/failure nodes, `test_roleplay_item_list.py` and focused mounted marks/Escape nodes, no-match/Also-in/loading/Retry nodes in `test_roleplay_item_list_app.py`, tag picker, toolbar geometry, theme contrast, Roleplay stylesheet, design-token/component governance, CSS bundle sync, and the relevant combined census/ratchet/CI workflow contracts. Use exact nodes where a whole file is large.
- [ ] Run targeted lint/format/static checks on changed Python and YAML/JSON/CSS contract files; verify clean diff and `git diff --check`. No full suite: repository instructions require an explicit user opt-in, and none is in scope. Hosted CI and size sweep were waived for this preparation; do not present source-era passes as combined-head evidence.
- [ ] Perform source self-review and invoke `superpowers:requesting-code-review` on the final combined head. Address findings, re-run affected focused checks, record start/end head hashes and exact outputs, then update B2b task implementation notes and the PR description with the repair and honest evidence limits.

## Execution handoff

This file is a repair preparation, not implementation authorization or merge approval. The parent may execute the already authorized oldest non-draft PR rebase and targeted local checks when its workspace is available. The remaining follow-up ownership and preservation risks are in `preparation-report.md`.



## Execution qualification (2026-11-01)

All 31 selected source commits were rebased onto exact dev
62a169f906508ed02bd7546ab278d916638d1607 without dropping a commit.
The bounded EOF repair and truthful filtered marks hint are implemented.
Integration qualification also repaired the Roleplay-only resting placeholder
contrast, applied dev's preflight dependency to the added admission lane,
and normalized the unchanged-dev CSS ownership helper's Windows path keys.
These preserve the existing contracts; no new ADR is required.

Targeted receipts deduplicate 310 latest PASS identities, one unavailable native
FIFO test, and three explicitly WAIVED numeric module-ceiling failures.
The retained dev ceilings measure 17,505/16,859, 1,322/855 and 1,579/1,297.
No threshold is increased; the CSS tightening to 268 remains.
Canonical CSS, 1,326-entry census (1,324-entry dev prefix intact; floor 1,319),
and diagnostic inventory checks passed. Source-era captures and task checks
remain historical. Full suite, hosted CI, full size/pre-import sweeps, native
POSIX FIFO and live external service qualification were not performed.

TASK-33910.47 was checked and set Done through the installed Backlog compatibility
CLI after focused behavioral/static verification; TASK-33910.4 carries these
qualified implementation notes. Other named follow-ups remain open. Root owns
the remaining independent whole-branch review and publication. Exact commands,
failed/setup attempts, source chronology and final binding live in this plan's
`.superpowers/sdd/2026-11-01-pr3334-backlog-integration/` evidence workspace.
