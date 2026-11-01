# Roleplay frame B0: shared frame primitives and the ADR — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the Library's adaptive reader shell, grip, messages, a rail-row fitter and the compact search input into one lazily imported shared module, add neutral resolver aliases and an opt-in `BaseAppScreen` Tab region, and record it all in a new ADR, with no visible change anywhere.

**Architecture:** The shell code moves (not rewritten) to `tldw_chatbook/Widgets/adaptive_pane_shell.py`, where every widget carries a *destination* class set instead of Library class names; the Library keeps its names as thin subclasses and same-object aliases, so every `@on` handler, type query and CSS rule keeps matching. The pure resolver stays in `Utils/adaptive_reader_state.py` and only gains same-object aliases, pinned byte-identical by a golden digest captured before the edit. `BaseAppScreen` gains a `TAB_REGION` binding that hands straight back to Textual's own walk while `TAB_REGION is None`, which is every route in B0.

**Tech Stack:** Python 3.12 (`.venv`, uv-managed), Textual 8.2.8, Rich `cell_len`, pytest 8 + pytest-asyncio (`asyncio_mode = "auto"`) + pytest-xdist, the repo's `build_css.py` split-sheet build, Backlog.md CLI 1.44.

**Spec:** `Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md` — slice card §5.2 "#### B0 · Shared frame primitives and the ADR (no visible change)"; also §2.1, R17, §2.12 items 1-5, §2.13, §1.4.3, §4.3, §5.4, §5.7, §5.9 (G1, G1a, G1b, G2, G3, G15, G16), §5.10, §5.11 and Appendix B (MF-01, MF-14, MF-19, MF-21, MF-23, RC-5). The spec's line numbers are pinned to `84247cb843`; every anchor in this plan was re-taken on `origin/dev @ fccf70d3b0` by symbol (spec MF-20 rule).

## Global Constraints

- Python and Textual: run everything with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` from the worktree root `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0`; never system `python`/`pytest`. Textual is 8.2.8.
- Boot parsed CSS: net ≤ 0 bytes in this PR, measured with `_boot_parsed_css_census()` (608,040 / 608,090 B, 50 B headroom, at `fccf70d3b0`). The ceiling `MAX_BOOT_PARSED_CSS_BYTES = 608_090` never rises.
- Broad selectors: stay 273 (`MAX_ANCESTOR_SCOPED_BARE_TYPE_RULES = 274`). Every new rule subject is an id or a class; no bare-type subject anywhere (lazy sheet or `BUNDLED_CSS`). B0 adds no CSS rule at all and edits no `.tcss` file (Task 6 is test-only), so no generated sheet is touched.
- UI-ready census: +0 against the paired base arm. Measured 1,033 / 1,033 at `fccf70d3b0`; the spec's "stays 1,032" is stale, the gate is "+0 versus base".
- Boot import weight: +0 (681 / 686). The new module is never on the `import tldw_chatbook.app` closure.
- Pre-import payload: exactly +1 module (`tldw_chatbook.Widgets.adaptive_pane_shell`), measured on both arms; defer → shed → an owner-signed ADR-097 ledger row in the same commit as the constant raise (spec Q10). No blanket re-pin. A subagent can never grant the sign-off. The question is asked in Task 0 Step 6, before any Library import lands, and the raise lands in the SAME commit as Task 3 (the first commit whose Library modules import the shared module), so no commit on the branch leaves the required perf guard red. `$EV/preimport_raise.py` raises only the constants the measured head exceeds (merge-order dependent: `MAX_PASS_ADDED_MODULES` on today's dev; `MAX_PASS_ADDED_LOC` and possibly `MAX_SINGLE_ROUTE_ADDED_LOC` if #2862 lands first) and is re-run, idempotently, before every later commit.
- Config import closure (`Tests/Packaging/test_config_import_closure.py`) passes: `Utils/adaptive_reader_state.py` stays a stdlib-only leaf; add no import to it.
- Never touch: `tldw_chatbook/UI/Screens/personas_screen.py`, `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/Library/library_shell_state.py`, `tldw_chatbook/UI/Library_Modules/screen_helpers.py`, `tldw_chatbook/Widgets/Library/library_emergency_return.py`.
- Never place code in `tldw_chatbook/Widgets/destination_rail.py` (UI-ready resident); no UI-ready-resident module may import the new module.
- No `StageReturnBar` in B0 (R17: it arrives at the post-#2862 shed).
- The shared module imports nothing from `Widgets/Library/`, `Library/` or `UI/Library_Modules/`, and has no `library-`-prefixed class token in any string constant (the grip id suffix `-library-grip` names the resolver pane id `library`, not a class). Roleplay never imports `Widgets/Library/`.
- New widget classes declare no `DEFAULT_CSS`, `CSS` or `BUNDLED_CSS`.
- CSS: class/id subjects only; no new broad selectors; no numeric dimension literals in any `.tcss`; generated sheets are never hand-edited — rebuild with `tldw_chatbook/css/build_css.py` and confirm with `tldw_chatbook/css/check_bundle_sync.py`.
- Python visual style writes: a non-literal `width`/`height`/`padding`/`margin`/`background`/`color`/`border`/`opacity` write needs `# ds-runtime: <reason>` on the same or the immediately preceding line; moved code keeps its markers exactly where they are.
- ADR-097 ratchets never rise without a ledger row; snapshots change only through `scripts/update_boot_budget_snapshots.py`.
- Add no `logger` call to `tldw_chatbook/UI/Navigation/base_app_screen.py` (production diagnostic inventory pins it).
- No `query_one` after an `await` without a `NoMatches` guard (preflight W002).
- Do not rename these test functions (node ids pinned by the Live closeout catalogue): `test_sync_layout_retains_every_mounted_child_identity`, `test_all_five_regions_remain_inside_representative_media_widths`, `test_hiding_focused_pane_moves_focus_to_truthful_restore_grip` (in `Tests/UI/test_library_adaptive_reader_shell.py`), `test_shared_resolution_uses_adaptive_width_classes`, `test_resolution_never_mutates_saved_preferences` (in `Tests/Library/test_library_adaptive_reader_state.py`). Keep `_ProbeApp` in `Tests/UI/test_library_adaptive_reader_shell.py` (imported by `test_library_layout_repair.py`).
- The route marker classes `library-media-route`, `library-notes-route`, `library-media-pane-grip` stay rule-free.
- Judging: "no new failures versus the paired base arm" (diff of failure sets), never "passes unchanged"; parts of these suites are already red on dev.
- Local Tests/UI environment (`backlog/docs/lessons-testing-evidence.md`, "Tests/UI `RecoveryRequired` at setup is a profile-selection trip" and "A local red wall of `RecoveryRequired` hides the guard you meant to run"): every new `Tests/UI` file imports `tldw_chatbook.app` at module scope (collection time), and `$EV/paired.sh` runs each arm with the `b0_bootstrap_all` collection plugin (Task 0 Step 3). Each paired run prints `recovery=<n>` per arm; a count above single digits means that comparison is two red walls and proves nothing: stop and fix the environment before judging it.
- Rebase protocol (spec §2.12 item 6, §5.11): never hand-merge generated sheets or snapshots. On any rebase onto dev: take dev's side of a conflicted generated file or snapshot; take dev's side of the pre-import constants in `Tests/Performance/test_screen_preimport_payload_budget.py` and of B0's ledger row (the raise script rewrites both); then run `build_css.py` and `check_bundle_sync.py`, re-anchor the base arm (Task 11 Step 1), redo Task 10 Steps 1-2 (re-measure, re-raise, snapshot refresh, PR-gate census script) and `./scripts/preflight.sh`.
- Every new geometry or containment assertion is shown red by a named mutation that leaves the code importable; record it in `$EV/mutations.md` and then in the task's Implementation Notes. Restore every mutation with the Edit tool (never `git checkout --`, never `git stash`).
- A pytest run that reports "no tests ran" is a FAILED gate: read a nonzero count.
- No "Verified against" stamps in `Docs/User_Guide/` (B0 changes no behaviour, so it has no guide delta).
- PR-gate census: every new fast `Tests/UI` file goes into `scripts/ui_pr_gate_census.txt` and `MINIMUM_FILES` rises to the new entry count, computed by script after the final rebase (Task 10 Step 2), never typed by hand.
- The ADR number 211 is provisional: re-run the all-remotes sweep at merge time. Keep the number out of code and CSS (docstrings, comments, test text, `patterns.json`, commit messages): code cites "the shared adaptive-pane-shell ADR (`backlog/decisions/`)". The number appears only in docs (the ADRs, the decisions README, `design-language.md`, `component-patterns.md`, the backlog task, this plan) and the ADR-097 ledger row, so a renumber is one scripted pass (Task 11 Step 8).
- Worktrees live under `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/`, never `/tmp`. Never touch the main checkout's working tree.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Library message identity after aliasing.** When the shared shell posts `PaneVisibilityChanged` or `AdaptivePaneShellResized`, every Library handler bound with `@on(LibraryPaneVisibilityChanged)`, `@on(AdaptiveReaderShellResized)` or `@on(MediaShellResized)` must still fire, and both handlers bound to the aliased resize class must run. A subclass alias would silently stop the footer's narrow-stage chip from refreshing. Pinned in Task 3 by `test_library_handlers_bound_to_aliases_still_fire_for_the_shared_shell`, `test_library_message_names_are_the_shared_message_objects` and `test_no_naming_convention_handler_exists_for_the_aliased_messages` (both directions: no handler keyed on a retired Library name, and none keyed on a shared name that would suddenly start receiving Library posts).
2. **The CSS re-key leaving Library grips unstyled.** On a fresh launch into Library Notes the grips must paint and focus exactly as before (`reverse` on focus, raised surface, no outline). A renamed destination class would fall back to Button's default look. Pinned in Task 6 by `test_a_focused_library_grip_resolves_the_lazy_sheet_focus_rule`, `test_the_boot_bundle_carries_only_the_grip_state_pair_of_the_shell_rules` and the re-keyed ownership pins, and in Task 11 by the live 120x36 / 160x45 (plus 220x55) `.txt` and `.ansi` comparison, which includes a capture with the Nav grip focused. B0 makes no visit-order claim (spec §2.12 item 2).
3. **`TAB_REGION is None` routes changing Tab order.** From any start (nothing focused, nav bar, first content control, a `TextArea`, last content control, footer), Tab and Shift+Tab must land exactly where Textual's stock binding landed, and F1's generic help must still list "Copy selected text". Pinned in Task 7 by `test_tab_region_none_walks_exactly_like_textuals_stock_binding`, `test_generic_f1_help_lists_the_same_rows_as_textuals_screen`, the static subclass audit and the every-route mounted test (arm-agnostic, run on both paired arms in Task 11).
4. **`fit_rail_row_label` with wide, emoji or zero-width titles.** A CJK, emoji or zero-width-space title must never paint past the rail in terminal cells, and the count must never be clipped at any width. Pinned in Task 4 by `test_wide_and_zero_width_titles_fit_by_terminal_cells` and `test_the_count_is_never_clipped`.
5. **Resolver alias drift.** The neutral names must stay the same objects and produce byte-identical layouts; `nav_open`/`nav_width` must never become dataclass fields (that would change equality, `astuple`, `replace` and positional construction). Pinned in Task 1 by `test_resolver_golden_grid_is_byte_identical`, `test_neutral_pane_aliases_are_the_reader_objects` and `test_nav_properties_read_the_library_fields_without_becoming_fields`.

---

## File Structure

**Create**

| Path | Single responsibility |
|---|---|
| `tldw_chatbook/Widgets/adaptive_pane_shell.py` | The shared primitive: `AdaptivePaneClasses`, the three messages, `AdaptivePaneGrip`, `AdaptivePaneShell`, `DestinationRailRow` + `FittedRailRowLabel` + `fit_rail_row_label` + `rail_row_content` + `DestinationRailRowButton`, and the promoted `SelectAllOnFocusingClickInput`. |
| `Tests/UI/test_adaptive_pane_shell.py` | Shared shell contracts with neutral probe classes, the Library compatibility seam, the CSS re-key pins, the promoted input and the `panes` registration. Fast; PR-gate censused. |
| `Tests/UI/test_destination_rail_row.py` | `fit_rail_row_label` at 24/31/35 cells, the §1.4.3 fallback order, the never-clipped count, wide glyphs, and the row button. Fast; censused. |
| `Tests/UI/test_base_app_screen_tab_region.py` | Tier-1 Tab-region mechanism against an in-test "before" arm, F1 rows, the opt-in contract, and the static audit of every production screen. Fast; censused. |
| `Tests/UI/test_base_app_screen_tab_region_routes.py` | Tier-2: every registered route with `TAB_REGION is None` follows the stock Tab walk on real mounted screens (~65 s), against Textual's own walk taken in the same mount. Arm-agnostic; not censused; run on both paired arms in Task 11. |
| `backlog/decisions/211-shared-adaptive-pane-shell.md` | ADR-211 (G1, G1a, G1b, G15). |
| `backlog/tasks/task-33910.1 - Roleplay-frame-B0-shared-pane-shell-primitives-and-the-frame-ADR-no-visible-change.md` | The B0 task (already filed in PR #2960 under parent TASK-33910): status, assignee and Implementation Plan section in Task 0 Step 5; close-out in Task 11 Step 10. |
| `backlog/tasks/task-33910*.md` (all other files; read-only here) | The parent and subtasks B1-B12 (`.2`-`.18`) and follow-ups (`.19`-`.28`), filed with the spec in PR #2960; this plan only cites their ids. |
| `Docs/superpowers/plans/2026-10-02-roleplay-b0-shared-frame-primitives.md` | This plan (already committed on the branch, because the ADR and the B0 task link to it). |
| `Docs/superpowers/plans/2026-10-02-roleplay-b0-captures/*.txt`, `*.ansi` | The live-check captures (base and head; Notes list, Notes editor and the focused Nav grip; 120x36, 160x45 and 220x55): 36 files. PNGs go to the owner, not the tree. |

**Modify**

| Path | Change |
|---|---|
| `tldw_chatbook/Utils/adaptive_reader_state.py` | Read-only `nav_open`/`nav_width` properties on preferences and effective layout; four same-object aliases at the end of the module. |
| `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py` | Becomes `LIBRARY_ADAPTIVE_READER_CLASSES`, the thin `LibraryAdaptiveReaderPaneGrip`/`LibraryAdaptiveReaderShell` subclasses and the message aliases. |
| `tldw_chatbook/Widgets/Library/library_rail.py` | Re-exports the shared `SelectAllOnFocusingClickInput`; `LibraryRailSearchInput` becomes a thin subclass that keeps "/" swallowing on. |
| `tldw_chatbook/UI/Navigation/base_app_screen.py` | `TAB_REGION`, the region `BINDINGS` (with Screen's copy binding re-spread), `arrival_focus_target()`, the two region actions. |
| `tldw_chatbook/Widgets/Library/library_browse_reader_shell.py`, `Tests/UI/test_on_mount_mro_convention.py`, `Tests/UI/test_watchlists_content_pane.py` | Comment/docstring-only edits whose prose B0 makes false (Tasks 3 and 7). |
| `tldw_chatbook/css/patterns.json`, `backlog/docs/component-patterns.md`, `tldw_chatbook/Widgets/pattern_gallery.py` | The `panes` widget-contract family (G16). |
| `Tests/Library/test_library_adaptive_reader_state.py` | Golden grid digest, alias identity, `nav_*` property tests. |
| `Tests/UI/test_library_adaptive_reader_shell.py` | The three CSS-ownership pins re-keyed to `LIBRARY_ADAPTIVE_READER_CLASSES` (names unchanged). |
| `backlog/decisions/086-library-adaptive-reader-shell.md`, `backlog/decisions/084-library-media-reader-ia.md`, `backlog/decisions/README.md` | "Amended by: ADR-211" lines (G2, G3) and the README rows. |
| `backlog/docs/design-language.md` | §2.8 pointers for G1a, G1b and G15. |
| `Tests/Performance/test_screen_preimport_payload_budget.py`, `backlog/decisions/097-boot-budget-ratchets.md`, `Tests/Performance/boot_budget_snapshots/preimport_payload.json` | The owner-signed raise of exactly the constants the head exceeds, its ledger row and the refreshed snapshot: first in Task 3's commit, re-checked before every later commit, finalised in Task 10. |
| `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py` | Three new censused files; `MINIMUM_FILES` set by script to the entry count after the final rebase (dev count + 3: 124 at `fccf70d3b0`, 125 at `185c845fe8`). |

## Conventions for every task

Shell state does not persist between tool calls, so every command block below re-declares what it needs. These names mean:

```
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook
WT=$MAIN/.worktrees/roleplay-b0            # the slice worktree (head arm)
BASE=$MAIN/.worktrees/roleplay-b0-base     # detached paired base arm (Task 0)
PY=$MAIN/.venv/bin/python
EV=$MAIN/.worktrees/.b0-evidence           # evidence dir: git-ignored (.worktrees/ is ignored), outlives the session
```

Multi-line command blocks are written as `bash <<'EOF' … EOF` so they behave the same under zsh (zsh does not word-split `$var`). Single test commands are written in full. Every `git` command uses `git -C "$WT"`.

---

### Task 0: Preconditions, paired base arm, evidence helpers, the programme task ids, the pre-import question

**Files:**
- Create: `$EV/failed_ids.py`, `$EV/paired.sh`, `$EV/plug/b0_bootstrap_all.py`, `$EV/mutations.md`, `$EV/suites-library.txt`, `$EV/suites-focus.txt`, `$EV/suites-css.txt`, `$EV/suites-perf.txt`, `$EV/preimport_measure.sh`, `$EV/preimport_raise.py`, `$EV/preimport-decision.md`, `$EV/owner-signoff.txt` (evidence only, never committed)
- Uses (already filed with the spec in PR #2960 — never re-file them): the parent `backlog/tasks/task-33910 - Roleplay-on-the-Library-frame-sub-project-B.md`, the B0 task `backlog/tasks/task-33910.1 - Roleplay-frame-B0-shared-pane-shell-primitives-and-the-frame-ADR-no-visible-change.md`, the slice subtasks `task-33910.2` … `.18` (B1-B12, spec §5.1 order) and the follow-ups `task-33910.19` … `.28` (FU-1..FU-5, then single-stage, library-slash, nav-fragment, content-batch, persona-duplicate)
- Modify: the B0 task file (status, assignee and its Implementation Plan section, Step 5). This plan is already committed on the branch (the ADR's `Plan:` link and the B0 task's Implementation Plan section cite it).

**Interfaces:**
- Consumes: nothing.
- Produces: the base worktree `$BASE` at the merge-base; `$EV/base-sha.txt`; `$EV/b0-task-id.txt` (`33910.1`); `$EV/slice-ids.txt` and `$EV/followup-ids.txt` (`<key> <id>` lines, e.g. `B7 33910.12`, `FU-1 33910.19`); `$EV/paired.sh <label> <suite-file> [pytest args…]`, which runs one pytest command on both arms (with the collection-time bootstrap plugin unless `PAIRED_PLUGIN=0`) and prints the failures that are new on head plus each arm's `RecoveryRequired` count; `$EV/preimport_measure.sh <base|head>` → `$EV/preimport-<arm>.json`; `$EV/preimport_raise.py` (idempotent owner-approved raise); the owner's verbatim Q10 answer in `$EV/owner-signoff.txt`.

This task has no production code, so it has no red/green cycle; its steps are setup that every later task depends on.

- [ ] **Step 1: Check the spec's precondition (§5.13 item 3) and rebase if it is met**

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b0
git -C "$MAIN" fetch origin --prune --quiet
for PR in 2957 2960; do echo "PR #$PR: $(gh pr view $PR --json state -q .state)"; done
git -C "$WT" status --short | head
git -C "$WT" log --oneline -1
EOF
```

Expected: two `PR #…: OPEN|MERGED` lines, an empty `status` (this plan is already committed on the branch), and the branch head. If both PRs say `MERGED`, run `git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 rebase origin/dev` (the branch holds only the plan commit, so this replays one docs commit). If either is still `OPEN`, stop and report it to the controller: the spec says B0 starts only once both are on dev. Continue only when the controller says so, and note its answer in `$EV/preconditions.txt` after Step 2 creates `$EV`. While #2960 is open, read the spec from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-library-ux/Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md`; while #2957 is open, Task 11 takes the harness from that PR's branch.

- [ ] **Step 2: Create the paired base arm and the evidence directory**

```bash
bash <<'EOF'
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b0
BASE=$MAIN/.worktrees/roleplay-b0-base; EV=$MAIN/.worktrees/.b0-evidence
mkdir -p "$EV"
MB=$(git -C "$WT" merge-base HEAD origin/dev)
if [ -d "$BASE" ]; then git -C "$BASE" checkout --quiet --detach "$MB"; else git -C "$MAIN" worktree add --detach "$BASE" "$MB"; fi
echo "$MB" > "$EV/base-sha.txt"
git -C "$BASE" log --oneline -1
git -C "$MAIN" check-ignore -q "$EV" && echo "evidence dir is git-ignored"
EOF
```

Expected: the base worktree's head line (the merge-base, `fccf70d3b0` unless Step 1 rebased) and `evidence dir is git-ignored`.

- [ ] **Step 3: Write the evidence helpers and the suite lists**

```bash
bash <<'OUTER'
set -euo pipefail
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence
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
cat > "$EV/plug/b0_bootstrap_all.py" <<'EOF'
"""B0 evidence plugin: import the app at collection time and keep the bootstrap profile.

Without it most mounted Tests/UI tests fail at setup with
``RecoveryRequired: raw_source_selection_changed`` on BOTH arms, and a
failure-set diff of two red walls cannot see a regression
(backlog/docs/lessons-testing-evidence.md, "A local red wall of
RecoveryRequired hides the guard you meant to run"). It changes when the
import happens and which profile a test keeps; it edits no test.
"""
import pytest


def pytest_collection_modifyitems(session, config, items):
    import tldw_chatbook.app  # noqa: F401

    for item in items:
        item.add_marker(pytest.mark.bootstrap_profile)
EOF
cat > "$EV/paired.sh" <<'EOF'
#!/usr/bin/env bash
# Usage: [PAIRED_PLUGIN=0] paired.sh <label> <suite-file> [extra pytest args...]
# Runs the same pytest command on the base arm and the head arm (sequentially,
# same interpreter) and prints the failures that are NEW on head. Each arm runs
# with the b0_bootstrap_all collection plugin unless PAIRED_PLUGIN=0 (the perf
# group: those guards run plain pytest in CI and measure in subprocesses).
# recovery=<n> must be single digits on both arms, or the diff is vacuous.
# No `set -u`: macOS /bin/bash 3.2 treats an empty "$@" as unbound.
set -o pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook
PY=$MAIN/.venv/bin/python
EV=$MAIN/.worktrees/.b0-evidence
LABEL=$1; SUITES=$2; shift 2
PLUGIN_ARGS="-p b0_bootstrap_all"
if [ "${PAIRED_PLUGIN:-1}" = 0 ]; then PLUGIN_ARGS=""; fi
for ARM in base head; do
  if [ "$ARM" = base ]; then T=$MAIN/.worktrees/roleplay-b0-base; else T=$MAIN/.worktrees/roleplay-b0; fi
  (cd "$T" && PYTHONPATH="$EV/plug" "$PY" -m pytest $(cat "$SUITES") $PLUGIN_ARGS "$@" -p no:cacheprovider -q -rf \
      --basetemp="$EV/run-$LABEL-$ARM" --junitxml="$EV/$LABEL-$ARM.xml" > "$EV/$LABEL-$ARM.log" 2>&1)
  "$PY" "$EV/failed_ids.py" "$EV/$LABEL-$ARM.xml" | sort > "$EV/$LABEL-$ARM.failed"
  echo "$LABEL $ARM: $(grep -E '[0-9]+ (passed|failed|error)' "$EV/$LABEL-$ARM.log" | tail -1) recovery=$(grep -c 'RecoveryRequired:' "$EV/$LABEL-$ARM.log") plugin=${PLUGIN_ARGS:-off}"
done
echo "new failures on head [$LABEL] (must be empty):"
comm -13 "$EV/$LABEL-base.failed" "$EV/$LABEL-head.failed"
echo "(end of new failures [$LABEL])"
EOF
chmod +x "$EV/paired.sh"
printf '# Named mutations (B0)\n\n| Task | Mutation | Test that went red | Restored and green |\n|---|---|---|---|\n' > "$EV/mutations.md"
cat > "$EV/suites-library.txt" <<'EOF'
Tests/Library/test_library_adaptive_reader_state.py
Tests/Library/test_library_media_reader_state.py
Tests/Library/test_library_rail_width.py
Tests/Library/test_library_rail_state.py
Tests/Library/test_library_conversation_reader_state.py
Tests/Library/test_library_skills_reader_state.py
Tests/Widgets/Library/test_library_rail.py
Tests/UI/test_library_adaptive_reader_shell.py
Tests/UI/test_library_adaptive_reader_closeout.py
Tests/UI/test_library_artifacts_focus.py
Tests/UI/test_library_collections_capture_reader.py
Tests/UI/test_library_collections_reader_geometry.py
Tests/UI/test_library_conversation_reader.py
Tests/UI/test_library_crit10_layout.py
Tests/UI/test_library_crit10_notes_details.py
Tests/UI/test_library_crit8_keyboard.py
Tests/UI/test_library_crit8_notes_loader.py
Tests/UI/test_library_crit8_polish_media.py
Tests/UI/test_library_crit8_polish_shell.py
Tests/UI/test_library_crit9_notes.py
Tests/UI/test_library_crit9_rail.py
Tests/UI/test_library_crit9_shell.py
Tests/UI/test_library_file_notes_workspace.py
Tests/UI/test_library_grip_capture.py
Tests/UI/test_library_honesty_accessibility.py
Tests/UI/test_library_ingest_clear_focus.py
Tests/UI/test_library_ingest_keyboard.py
Tests/UI/test_library_ingest_resize_focus.py
Tests/UI/test_library_layout_repair.py
Tests/UI/test_library_media_characterization.py
Tests/UI/test_library_media_reader_flow.py
Tests/UI/test_library_media_reader_shell.py
Tests/UI/test_library_media_render_fixes.py
Tests/UI/test_library_media_return_settlement.py
Tests/UI/test_library_multiselect_media.py
Tests/UI/test_library_notes_authority_layout.py
Tests/UI/test_library_notes_empty_toolbar.py
Tests/UI/test_library_notes_folder_navigator.py
Tests/UI/test_library_notes_reader.py
Tests/UI/test_library_notes_w3_layout.py
Tests/UI/test_library_notes_w4_editor.py
Tests/UI/test_library_notes_w4_layout.py
Tests/UI/test_library_notes_w5_kbd_focus.py
Tests/UI/test_library_notes_wave_editor_keys.py
Tests/UI/test_library_notes_wave_import_ux.py
Tests/UI/test_library_notes_wave_list.py
Tests/UI/test_library_phase_c_region_ownership.py
Tests/UI/test_library_phase_c_switch_storm.py
Tests/UI/test_library_prompt_action_journeys.py
Tests/UI/test_library_prompt_resize_focus.py
Tests/UI/test_library_prompts_canvas.py
Tests/UI/test_library_prompts_reader.py
Tests/UI/test_library_rag_history_keyboard.py
Tests/UI/test_library_rag_mode_scope_keyboard.py
Tests/UI/test_library_rag_query_return.py
Tests/UI/test_library_rag_recovery_navigation.py
Tests/UI/test_library_rag_result_focus.py
Tests/UI/test_library_rail_focus_visibility.py
Tests/UI/test_library_rail_profile_admission.py
Tests/UI/test_library_reader_press_scope_t22228.py
Tests/UI/test_library_resize_focus_gates_t23025.py
Tests/UI/test_library_shell.py
Tests/UI/test_library_skills_reader.py
Tests/UI/test_non_obscuring_focus_contract.py
Tests/UI/test_on_mount_mro_convention.py
Tests/UI/test_product_maturity_gate16_library_search_rag.py
Tests/UI/test_library_crit9_grammar.py
Tests/UI/test_library_choice_strips.py
Tests/UI/test_library_rag_query_gate_race.py
Tests/UI/test_library_rag_rechunk_action.py
Tests/UI/test_library_rag_legacy_chunk_report.py
Tests/Library/test_library_rag_state.py
Tests/Live/test_library_adaptive_reader_closeout.py
Tests/Packaging/test_config_import_closure.py
EOF
cat > "$EV/suites-focus.txt" <<'EOF'
Tests/Architecture/test_base_app_screen_recompose_focus_seam.py
Tests/Architecture/test_on_mount_super_guard.py
Tests/Architecture/test_persistent_diagnostic_inventory.py
Tests/UI/test_product_maturity_phase1_keyboard_focus.py
Tests/UI/test_master_shell_navigation.py
Tests/UI/test_nav_overflow_layout.py
Tests/UI/test_screen_navigation.py
Tests/UI/test_workbench_focus_help.py
Tests/UI/test_screen_footer_hints.py
Tests/UI/test_shell_chrome_contract.py
Tests/UI/test_workbench_pane_focus.py
Tests/UI/test_destination_visual_parity_correction.py
Tests/UI/test_console_tab_scope.py
Tests/UI/test_destination_rail.py
Tests/UI/test_settings_interface_keyboard_journeys.py
Tests/UI/test_settings_provider_keyboard_journeys.py
Tests/UI/test_settings_speech_keyboard_journeys.py
Tests/UI/test_llm_screen_lab_adoption.py
Tests/UI/test_llm_gguf_source_modes.py
Tests/UI/test_vllm_lab_geometry.py
Tests/UI/test_workflows_run.py
Tests/UI/test_workflows_editor.py
Tests/UI/test_speech_profile_navigation.py
Tests/UI/test_speech_playground_pane_lifecycle.py
Tests/UI/test_focus_keeps_committed_value.py
Tests/UI/test_product_maturity_phase6_focus_visual_sweep.py
Tests/UI/test_first_run_wizard_live_contract.py
Tests/UI/test_personas_workbench.py
Tests/UI/test_personas_dictionaries.py
Tests/UI/test_personas_lore.py
Tests/UI/test_personas_deferred_center_views.py
Tests/UI/test_personas_workbench_foundation.py
Tests/UI/test_personas_library_rail_focus_outline.py
Tests/UI/test_theme_contrast.py
EOF
cat > "$EV/suites-css.txt" <<'EOF'
Tests/UI/test_css_build_integrity.py
Tests/UI/test_consolidated_css_harness.py
Tests/UI/test_css_staleness_manifest.py
Tests/UI/test_component_pattern_governance.py
Tests/UI/test_widget_css_consolidation.py
Tests/UI/test_pattern_gallery_snapshots.py
Tests/UI/test_pattern_gallery_layout.py
Tests/UI/test_pattern_gallery_command.py
Tests/Architecture/test_module_size_ratchet.py
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
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
for f in $(cat "$EV"/suites-*.txt); do [ -f "$f" ] || echo "MISSING $f"; done
echo "suite files checked"
OUTER
```

Expected: only `suite files checked` (no `MISSING` line). The six Search/RAG-panel suites at the end of the library list exercise `library_search_rag_panel.py`, whose box (`SelectAllOnFocusingClickInput`) gains the shared `_on_key` in Task 5. `test_screen_preimport_payload_budget.py` is in the perf list because spec §5.7.4 runs it on every PR; it passes on both arms only once Task 3's raise is in.

- [ ] **Step 4: Write the pre-import helpers (measure and raise)**

`preimport_measure.sh` takes one census per arm (the same subprocess the guard runs, through its own `_run_census`) and keeps the module SET, so the raise can check that the shared module is the only addition whether the guard passes or fails. `preimport_raise.py` is stateless and idempotent: it compares the head measurement with the limits in the BASE arm's copy of the test file (`git show <base-sha>:…`), sets every constant the head exceeds to the measured head value, puts any other constant back to its base value, and writes B0's one ledger row naming exactly the raised constants (old → new), or removes it when nothing is raised. B0's ledger row is unmerged, so rewriting it in place before merge does not break the ledger's append-only rule.

```bash
bash <<'OUTER'
set -euo pipefail
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence
mkdir -p "$EV/tmp"
cat > "$EV/preimport_measure.sh" <<'EOF'
#!/usr/bin/env bash
# Usage: preimport_measure.sh <base|head>  ->  $EV/preimport-<arm>.json + a summary line
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b0-evidence
ARM=$1
if [ "$ARM" = base ]; then T=$MAIN/.worktrees/roleplay-b0-base; else T=$MAIN/.worktrees/roleplay-b0; fi
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
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], "| modules", d["modules"], "| LOC", d["loc"], "| library", d["routes"]["library"][0], "mods /", d["routes"]["library"][1], "LOC | fattest route", d["fattest_route_loc"], "LOC")' "$EV/preimport-$ARM.json" "$ARM"
EOF
chmod +x "$EV/preimport_measure.sh"
cat > "$EV/preimport_raise.py" <<'EOF'
"""Raise exactly the screen pre-import limits B0's head exceeds (owner-approved, spec Q10).

Stateless and idempotent: run it after `preimport_measure.sh head` before every
commit from Task 3 on. Inputs: $EV/preimport-base.json, $EV/preimport-head.json,
$EV/base-sha.txt, $EV/b0-task-id.txt, $EV/owner-signoff.txt.
"""
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

MAIN = Path("/Users/macbook-dev/Documents/GitHub/tldw_chatbook")
WT = MAIN / ".worktrees/roleplay-b0"
EV = MAIN / ".worktrees/.b0-evidence"
TEST_REL = "Tests/Performance/test_screen_preimport_payload_budget.py"
TEST = WT / TEST_REL
ADR = WT / "backlog/decisions/097-boot-budget-ratchets.md"
SHARED = "tldw_chatbook.Widgets.adaptive_pane_shell"
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
if added not in ([], [SHARED]) or removed:
    sys.exit(f"STOP: unexpected module delta added={added} removed={removed}: something new became eager")
for key, cap in CAP.items():
    if head[key] - base[key] > cap:
        sys.exit(f"STOP: {key} grew {head[key] - base[key]} (> the approved {cap}); ask the owner again")

base_sha = (EV / "base-sha.txt").read_text(encoding="utf-8").strip()
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
task_id = (EV / "b0-task-id.txt").read_text(encoding="utf-8").strip()
signoff_path = EV / "owner-signoff.txt"
signoff = signoff_path.read_text(encoding="utf-8").strip() if signoff_path.exists() else ""
if raised and not signoff:
    sys.exit(f"STOP: {sorted(raised)} need raising and owner-signoff.txt is empty (Task 0 Step 6)")
today = datetime.date.today().isoformat()

text = TEST.read_text(encoding="utf-8")
limits(text)  # exactly one line per constant before editing
for name in NAMES:
    value = raised[name][1] if name in raised else original[name]
    comment = (
        f"#: B0 TASK-{task_id}: {original[name]:_} -> {value:_} ({today}), owner-approved; ADR-097 ledger.\n"
        if name in raised
        else ""
    )
    text = re.sub(
        rf"^(?:#: B0 TASK-[^\n]*\n)?{name} = [\d_]+$",
        lambda _m, c=comment, n=name, v=value: f"{c}{n} = {v:_}",
        text,
        count=1,
        flags=re.M,
    )
TEST.write_text(text, encoding="utf-8")

lines = ADR.read_text(encoding="utf-8").splitlines(keepends=True)
marker = f"Roleplay frame B0 (TASK-{task_id})"
lines = [line for line in lines if marker not in line]
if raised:
    start = next(i for i, line in enumerate(lines) if line.startswith("## Exception ledger"))
    header = next(i for i in range(start, len(lines)) if lines[i].startswith("| date | guard |"))
    end = header
    while end + 1 < len(lines) and lines[end + 1].startswith("|"):
        end += 1
    lib_base, lib_head = base["routes"]["library"], head["routes"]["library"]
    # The ADR number is read from the file name (Task 9 may renumber it).
    adr_files = sorted((WT / "backlog/decisions").glob("*-shared-adaptive-pane-shell.md"))
    adr_ref = f"ADR-{adr_files[0].name[:3]}" if adr_files else "ADR number pending, Task 9"
    cause = (
        f"{marker}, the shared adaptive-pane-shell ADR ({adr_ref}): new shared module `{SHARED}`, "
        "imported at module scope by `Widgets/Library/library_adaptive_reader_shell.py` (thin "
        "subclasses) and `Widgets/Library/library_rail.py` (re-exported input), so the Library "
        f"route adds it. Same-session paired arms (base `{base_sha[:10]}`): pass {base['modules']} "
        f"modules / {base['loc']:,} LOC -> {head['modules']} / {head['loc']:,}; library route "
        f"{lib_base[0]} / {lib_base[1]:,} -> {lib_head[0]} / {lib_head[1]:,}. Defer is impossible "
        "(a base class and a re-export cannot be lazy); nothing can be shed in B0 "
        "(`library_emergency_return.py` is reserved for the post-#2862 shed, which pays this back, R17)."
    )
    quote = signoff.replace("|", "\\|").replace("\n", " ")
    row = (
        f"| {today} | screen pre-import payload | "
        + " / ".join(f"`{name}`" for name in raised)
        + " | "
        + " / ".join(f"{old:,} → {new:,}" for old, new in raised.values())
        + f" | {cause} | Owner, verbatim: \"{quote}\" |\n"
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

Expected: `pre-import helpers written`. Both scripts are exercised for real in Task 3's checkpoint; before Task 3 the head measures the same as the base and `preimport_raise.py` prints `raised: nothing (head within base limits) | ledger row: none`.

- [ ] **Step 5: Record the programme's task ids and start the B0 task (already filed in PR #2960 — never re-file)**

The parent TASK-33910 and its 28 subtasks were filed with the spec in PR #2960: B0 = TASK-33910.1, B1-B12 = `.2`-`.18` in spec §5.1 order (B1, B2a, B2b, B3, B4, B5a, B5b-1, B5b-2, B5c, B6, B7, B8, B9a, B9b, B10, B11, B12), FU-1..FU-5 = `.19`-`.23`, and the five other §5.13 item 2 follow-ups = `.24`-`.28`. After Step 1's rebase they are on this branch. Record their ids for the later steps (the ADR tokens in Task 9 Step 5, the merge-time check in Task 11 Step 8, the close-out in Task 11 Step 10), then mark B0 In Progress and give it its Implementation Plan section (CLAUDE.md §5). Five-digit ids break `backlog task edit` (lessons-backlog-hygiene, TASK-15463), so edit the file directly.

```bash
bash <<'EOF'
set -euo pipefail
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
P=33910
B0="backlog/tasks/task-33910.1 - Roleplay-frame-B0-shared-pane-shell-primitives-and-the-frame-ADR-no-visible-change.md"
test -f "backlog/tasks/task-${P} - Roleplay-on-the-Library-frame-sub-project-B.md" || { echo "MISSING parent: PR #2960 is not on this branch; go back to Step 1"; exit 1; }
test -f "$B0" || { echo "MISSING B0 task: go back to Step 1"; exit 1; }
echo "${P}.1" > "$EV/b0-task-id.txt"
printf '%s\n' "B1 ${P}.2" "B2a ${P}.3" "B2b ${P}.4" "B3 ${P}.5" "B4 ${P}.6" "B5a ${P}.7" "B5b-1 ${P}.8" "B5b-2 ${P}.9" "B5c ${P}.10" "B6 ${P}.11" "B7 ${P}.12" "B8 ${P}.13" "B9a ${P}.14" "B9b ${P}.15" "B10 ${P}.16" "B11 ${P}.17" "B12 ${P}.18" > "$EV/slice-ids.txt"
printf '%s\n' "FU-1 ${P}.19" "FU-2 ${P}.20" "FU-3 ${P}.21" "FU-4 ${P}.22" "FU-5 ${P}.23" "single-stage ${P}.24" "library-slash ${P}.25" "nav-fragment ${P}.26" "content-batch ${P}.27" "persona-duplicate ${P}.28" > "$EV/followup-ids.txt"
for ID in $(awk '{print $2}' "$EV/slice-ids.txt" "$EV/followup-ids.txt"); do
  ls backlog/tasks/task-${ID}\ -\ *.md >/dev/null 2>&1 || echo "MISSING TASK-${ID}"
done
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - "$B0" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
assert text.count("status: To Do\n") == 1, "B0 task is not To Do"
text = text.replace("status: To Do\n", "status: In Progress\n", 1)
text = text.replace("assignee: []\n", "assignee:\n  - '@claude'\n", 1)
if "## Implementation Plan" not in text:
    text = text.rstrip("\n") + (
        "\n\n## Implementation Plan\n\n<!-- SECTION:PLAN:BEGIN -->\n"
        "Docs/superpowers/plans/2026-10-02-roleplay-b0-shared-frame-primitives.md, Tasks 0 to 11.\n"
        "<!-- SECTION:PLAN:END -->\n"
    )
path.write_text(text, encoding="utf-8")
print("B0 task started")
PY
backlog task list --plain 2>/dev/null | grep -c "TASK-${P}"
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_backlog_task_files.py | tail -1
git add "$B0"
git commit -m "chore(backlog): start TASK-33910.1 (Roleplay frame B0) with its implementation plan" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: no `MISSING` line, `B0 task started`, a count of 29 (the parent plus 28 subtasks), the task-files check passing, and the commit line. A `MISSING` line means PR #2960 is not on this branch yet: go back to Step 1.

- [ ] **Step 6: STOP — ask the pre-import question now (spec Q10), before any Library import lands**

Task 3 is the first commit in which the Library's modules import the shared module, and from that commit the required perf guard `test_preimport_pass_payload_stays_within_budget` is red until a constant is raised. So the owner's answer is needed before Task 3 commits, not at the end. Write the decision record:

```bash
cat > /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence/preimport-decision.md <<'EOF'
# B0 pre-import growth: ADR-097 order applied

- Defer: not possible. `Widgets/Library/library_adaptive_reader_shell.py` subclasses the shared shell and grip at module scope, and `Widgets/Library/library_rail.py` re-exports the shared input; a base class and a re-export cannot be lazy. Lazy-importing in Roleplay instead (spec Q10 option c) was rejected by the owner: it moves the cost onto the first Ctrl+4.
- Shed: nothing on the Library route can fold into the shared module in B0 without touching a never-touch file. The planned shed is `library_emergency_return.py` -> `StageReturnBar` after PR #2862 (R17), which pays this back.
- Therefore: an owner-signed ledger row for the measured growth (spec Q10 option a), raised in the same commit as Task 3, re-checked before every later commit by `$EV/preimport_raise.py`.
EOF
```

Report to the controller and wait for the answer before committing Task 3. The controller asks the owner, verbatim:

> "Roleplay B0 adds one module to the screen pre-import census: `tldw_chatbook.Widgets.adaptive_pane_shell`, which the Library's compatibility modules import at module scope, plus about +540 lines on the pass and on the Library route (spec §5.10 forecast about +0.5k; measured +539 on `fccf70d3b0` from this plan's code while planning). Defer is impossible (the Library subclasses it at module scope) and there is nothing to shed in B0 (the planned shed, `library_emergency_return.py`, waits for PR #2862 and pays this back). Which limit it breaches depends on merge order: on today's dev, `MAX_PASS_ADDED_MODULES` (557 → 558); if #2862 lands first, `MAX_PASS_ADDED_LOC` instead (59 lines of headroom there), possibly with `MAX_SINGLE_ROUTE_ADDED_LOC`. May I raise exactly the constants B0's head exceeds, each to the value measured on paired arms in one session, with one ADR-097 ledger row quoting those numbers, as long as the growth stays within +1 module and +1,000 lines (pass and largest route)? Anything beyond that comes back to you."

If the controller also wants a screenshot waiver for this no-visible-change slice (spec §5.4 item 6 has none for B0), it asks in the same message; the answer is recorded verbatim in `$EV/owner-screenshots.txt` (Task 11 Step 7). Never waive silently.

Write the owner's reply, verbatim, to `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence/owner-signoff.txt`. A subagent's or the controller's own approval is not sign-off. While the answer is pending, Tasks 1 and 2 may proceed (neither makes the Library import the shared module), and so may Task 7 out of order (it touches neither the shared module nor the Library); Task 3's checkpoint (Step 6) stops without it, so Task 3 does not commit. If the owner declines, stop at Task 3 without committing it: B0 cannot merge with a red pre-import guard (spec Q10), and the controller decides the next step.

---

### Task 1: Neutral resolver aliases and the golden grid

**Files:**
- Modify: `tldw_chatbook/Utils/adaptive_reader_state.py` (the fields of `AdaptiveReaderLayoutPreferences` and `AdaptiveReaderEffectiveLayout`; end of file)
- Test: `Tests/Library/test_library_adaptive_reader_state.py` (import block; end of file)

**Interfaces:**
- Consumes: `resolve_adaptive_reader_layout`, `normalize_adaptive_reader_preferences`, `AdaptiveReaderLayoutProfile`, `AdaptiveReaderLayoutPreferences`, `AdaptiveReaderEffectiveLayout` (all existing in `Utils/adaptive_reader_state.py`).
- Produces (same objects, never subclasses): `AdaptivePaneProfile = AdaptiveReaderLayoutProfile`, `AdaptivePanePreferences = AdaptiveReaderLayoutPreferences`, `AdaptivePaneLayout = AdaptiveReaderEffectiveLayout`, `resolve_adaptive_pane_layout = resolve_adaptive_reader_layout`; read-only properties `nav_open: bool` (reads `library_open`) and `nav_width: int` (reads `library_width`) on `AdaptiveReaderLayoutPreferences` and `AdaptiveReaderEffectiveLayout`.

- [ ] **Step 1: Add the golden grid test, pinned to a digest captured on unmodified code**

The digest `7234b56ac02aa1491b3e96fe02288ce0229ce5dd07dfe8ed68e80326d362bd85` was computed with exactly this code on `origin/dev @ fccf70d3b0` before any B0 edit (0.76 s). Edit the import block at the top of `Tests/Library/test_library_adaptive_reader_state.py`:

Old:
```python
from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Utils.adaptive_reader_state import (
```

New:
```python
from __future__ import annotations

import hashlib
import itertools
from dataclasses import astuple, replace

import pytest

from tldw_chatbook.Utils import adaptive_reader_state
from tldw_chatbook.Utils.adaptive_reader_state import (
```

Then append to the end of the file:

```bash
cat >> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0/Tests/Library/test_library_adaptive_reader_state.py <<'EOF'


# ---------------------------------------------------------------------------
# Roleplay frame B0 (the shared adaptive-pane-shell ADR): the golden resolver grid.
#
# The neutral aliases are the SAME objects as the reader names, so comparing
# their outputs with each other would prove nothing. ``_GOLDEN_DIGEST`` was
# captured on UNMODIFIED ``adaptive_reader_state.py`` (origin/dev fccf70d3b0,
# 2026-10-02) before B0 touched the module: a change to it is a resolver
# behaviour change, never an alias side effect. The profiles are literals,
# not ``screen_constants`` reads, so a Library profile retune elsewhere cannot
# move this digest -- it pins the resolver and nothing else. They cover every
# behaviour switch a declared profile uses today.
# ---------------------------------------------------------------------------

_GOLDEN_PROFILES = (
    ("default", AdaptiveReaderLayoutProfile()),
    (
        "media",
        AdaptiveReaderLayoutProfile(
            work_min_width=46, list_grows=True, list_first_when_empty=True
        ),
    ),
    (
        "notes",
        AdaptiveReaderLayoutProfile(
            work_min_width=48,
            list_comfort_width=64,
            list_grows=True,
            list_first_when_empty=True,
        ),
    ),
    (
        "collections",
        AdaptiveReaderLayoutProfile(work_min_width=48, work_comfort_width=56),
    ),
    ("prompts_skills", AdaptiveReaderLayoutProfile(work_min_width=48)),
    ("artifacts", AdaptiveReaderLayoutProfile(list_first_when_empty=True)),
    ("one_cell_grips", AdaptiveReaderLayoutProfile(grip_width=1)),
)
_GOLDEN_WIDTHS = tuple(sorted(set(range(0, 301, 3)) | set(range(56, 73))))
_GOLDEN_WALK = tuple(range(40, 241, 4)) + tuple(range(240, 39, -4))
_GOLDEN_DIGEST = "7234b56ac02aa1491b3e96fe02288ce0229ce5dd07dfe8ed68e80326d362bd85"


def _golden_preferences() -> tuple[AdaptiveReaderLayoutPreferences, ...]:
    """Every preference the grid feeds the resolver, built by the normaliser.

    Open flags x {automatic, custom x library {1, 36, 999} x items {1, 50,
    999}}: 40 distinct normalised preferences, the out-of-range widths
    exercising the clamps.
    """
    raws: list[dict[str, object]] = []
    for library_open, items_open in itertools.product((True, False), repeat=2):
        raws.append({"library_open": library_open, "items_open": items_open})
        for library_width, items_width in itertools.product(
            (1, 36, 999), (1, 50, 999)
        ):
            raws.append(
                {
                    "library_open": library_open,
                    "items_open": items_open,
                    "custom_widths_enabled": True,
                    "library_width": library_width,
                    "items_width": items_width,
                }
            )
    return tuple(normalize_adaptive_reader_preferences(raw) for raw in raws)


def _golden_digest(resolve) -> str:
    """SHA-256 over every resolved layout in the grid, plus hysteresis walks.

    Args:
        resolve: The resolver under test (a reader name or a neutral alias).

    Returns:
        The hex digest of the ``repr`` of every input/output tuple, in order.
    """
    digest = hashlib.sha256()
    for (name, profile), prefs in itertools.product(
        _GOLDEN_PROFILES, _golden_preferences()
    ):
        for width, priority, reader_has_item in itertools.product(
            _GOLDEN_WIDTHS, (None, "library", "items"), (True, False)
        ):
            out = resolve(
                width,
                prefs,
                profile,
                priority=priority,
                reader_has_item=reader_has_item,
            )
            digest.update(
                repr(
                    (
                        name,
                        astuple(prefs),
                        width,
                        priority,
                        reader_has_item,
                        astuple(out),
                    )
                ).encode()
            )
        previous = None
        for width in _GOLDEN_WALK:
            previous = resolve(width, prefs, profile, previous=previous)
            digest.update(
                repr((name, astuple(prefs), width, "walk", astuple(previous))).encode()
            )
    return digest.hexdigest()


@pytest.mark.parametrize("resolver_name", ["resolve_adaptive_reader_layout"])
def test_resolver_golden_grid_is_byte_identical(resolver_name: str) -> None:
    resolve = getattr(adaptive_reader_state, resolver_name)
    assert len(set(_golden_preferences())) == 40
    assert _golden_digest(resolve) == _GOLDEN_DIGEST
EOF
```

- [ ] **Step 2: Run it while `adaptive_reader_state.py` is still unmodified (this run is the capture check)**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest "Tests/Library/test_library_adaptive_reader_state.py::test_resolver_golden_grid_is_byte_identical" -q -p no:cacheprovider 2>&1 | tail -1`

Expected: `1 passed`. If it fails, `origin/dev` changed the resolver after `fccf70d3b0`: the assertion message prints the digest of the unmodified module on the left of `==`; copy that value into `_GOLDEN_DIGEST`, put the base SHA from `$EV/base-sha.txt` and today's date in the comment above it, re-run (`1 passed`), and only then continue. Never recompute the digest after Step 6 has edited the module.

- [ ] **Step 3: Show the golden test discriminates (named mutation `hysteresis-4-to-5`)**

With the Edit tool, in `tldw_chatbook/Utils/adaptive_reader_state.py` change `LAYOUT_HYSTERESIS_WIDTH = 4` to `LAYOUT_HYSTERESIS_WIDTH = 5`. Run:

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest "Tests/Library/test_library_adaptive_reader_state.py::test_resolver_golden_grid_is_byte_identical" -q -p no:cacheprovider 2>&1 | tail -3`

Expected: `1 failed`, with `AssertionError` comparing a different digest (`91df5229…` on `fccf70d3b0`). Restore `LAYOUT_HYSTERESIS_WIDTH = 4` with the Edit tool, re-run, expect `1 passed`, and append a row to `$EV/mutations.md`: `| 1 | LAYOUT_HYSTERESIS_WIDTH 4 -> 5 | test_resolver_golden_grid_is_byte_identical | yes |`.

- [ ] **Step 4: Write the failing alias and property tests**

Edit the imports in `Tests/Library/test_library_adaptive_reader_state.py`:

Old:
```python
from dataclasses import astuple, replace
```
New:
```python
from dataclasses import astuple, fields, replace
```

Old:
```python
from tldw_chatbook.Utils.adaptive_reader_state import (
    PANE_GRIP_WIDTH,
    READER_COMFORT_WIDTH,
    AdaptiveReaderEffectiveLayout,
```
New:
```python
from tldw_chatbook.Utils.adaptive_reader_state import (
    PANE_GRIP_WIDTH,
    READER_COMFORT_WIDTH,
    AdaptivePaneLayout,
    AdaptivePanePreferences,
    AdaptivePaneProfile,
    AdaptiveReaderEffectiveLayout,
```

Old:
```python
    resolve_adaptive_reader_layout,
)
from tldw_chatbook.Library.library_media_reader_state import (
```
New:
```python
    resolve_adaptive_pane_layout,
    resolve_adaptive_reader_layout,
)
from tldw_chatbook.Library.library_media_reader_state import (
```

Old:
```python
@pytest.mark.parametrize("resolver_name", ["resolve_adaptive_reader_layout"])
```
New:
```python
@pytest.mark.parametrize(
    "resolver_name", ["resolve_adaptive_reader_layout", "resolve_adaptive_pane_layout"]
)
```

Append:

```bash
cat >> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0/Tests/Library/test_library_adaptive_reader_state.py <<'EOF'


def test_neutral_pane_aliases_are_the_reader_objects() -> None:
    """Same objects, never subclasses: behaviour is identical by construction."""
    assert AdaptivePaneProfile is AdaptiveReaderLayoutProfile
    assert AdaptivePanePreferences is AdaptiveReaderLayoutPreferences
    assert AdaptivePaneLayout is AdaptiveReaderEffectiveLayout
    assert resolve_adaptive_pane_layout is resolve_adaptive_reader_layout


def test_nav_properties_read_the_library_fields_without_becoming_fields() -> None:
    """``nav_*`` are read-only views; a FIELD would change equality and astuple.

    It would also break positional construction of the layout (six required
    positional fields, then ``grip_width``).
    """
    layout = AdaptiveReaderEffectiveLayout(True, False, 31, 0, 60, None)
    assert (layout.nav_open, layout.nav_width) == (True, 31)
    prefs = AdaptiveReaderLayoutPreferences(library_open=False, library_width=40)
    assert (prefs.nav_open, prefs.nav_width) == (False, 40)
    for dataclass_type in (
        AdaptiveReaderEffectiveLayout,
        AdaptiveReaderLayoutPreferences,
    ):
        names = {field.name for field in fields(dataclass_type)}
        assert not names & {"nav_open", "nav_width"}, dataclass_type
    with pytest.raises(AttributeError):
        layout.nav_open = False  # type: ignore[misc]
EOF
```

- [ ] **Step 5: Run the file to see it fail**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_adaptive_reader_state.py -q -p no:cacheprovider 2>&1 | tail -5`

Expected: `ERROR collecting Tests/Library/test_library_adaptive_reader_state.py` with `ImportError: cannot import name 'AdaptivePaneLayout' from 'tldw_chatbook.Utils.adaptive_reader_state'`.

- [ ] **Step 6: Add the properties and the aliases**

In `tldw_chatbook/Utils/adaptive_reader_state.py`:

Old:
```python
    library_open: bool = True
    items_open: bool = True
    custom_widths_enabled: bool = False
    library_width: int = LIBRARY_TARGET_WIDTH
    items_width: int = ITEMS_TARGET_WIDTH
```
New:
```python
    library_open: bool = True
    items_open: bool = True
    custom_widths_enabled: bool = False
    library_width: int = LIBRARY_TARGET_WIDTH
    items_width: int = ITEMS_TARGET_WIDTH

    @property
    def nav_open(self) -> bool:
        """Neutral read-only name for ``library_open`` (Roleplay frame B0).

        A property, never a field: fields drive equality, ``astuple`` and
        ``replace``, which the golden grid pins.
        """
        return self.library_open

    @property
    def nav_width(self) -> int:
        """Neutral read-only name for ``library_width`` (Roleplay frame B0)."""
        return self.library_width
```

Old:
```python
    reader_width: int
    priority_pane: PaneName | None
    grip_width: int = PANE_GRIP_WIDTH
```
New:
```python
    reader_width: int
    priority_pane: PaneName | None
    grip_width: int = PANE_GRIP_WIDTH

    @property
    def nav_open(self) -> bool:
        """Neutral read-only name for ``library_open`` (Roleplay frame B0)."""
        return self.library_open

    @property
    def nav_width(self) -> int:
        """Neutral read-only name for ``library_width`` (Roleplay frame B0)."""
        return self.library_width
```

Append the aliases:

```bash
cat >> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0/tldw_chatbook/Utils/adaptive_reader_state.py <<'EOF'


# Neutral names for destinations that are not the Library (Roleplay frame B0;
# the shared adaptive-pane-shell ADR). The SAME objects, not subclasses, so behaviour is byte-identical
# by construction; the golden grid in
# Tests/Library/test_library_adaptive_reader_state.py pins it against a digest
# captured before these lines existed. No new import: this module stays a
# config-safe stdlib leaf (TASK-22223). The pane ids stay "library"/"items"
# (PaneName) for every destination; Roleplay's navigation rail is the
# "library" pane.
AdaptivePaneProfile = AdaptiveReaderLayoutProfile
AdaptivePanePreferences = AdaptiveReaderLayoutPreferences
AdaptivePaneLayout = AdaptiveReaderEffectiveLayout
resolve_adaptive_pane_layout = resolve_adaptive_reader_layout
EOF
```

- [ ] **Step 7: Run the file and the config closure**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_adaptive_reader_state.py Tests/Packaging/test_config_import_closure.py -q -p no:cacheprovider 2>&1 | tail -2`

Expected: `379 passed` — 378 in the state file (374 existing on `fccf70d3b0` + 2 golden parameters + 2 new tests) plus the closure test. If dev moved, confirm the state file's count with `--collect-only -q Tests/Library/test_library_adaptive_reader_state.py | tail -1` and expect that number plus 1, with no failures.

- [ ] **Step 8: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
git add tldw_chatbook/Utils/adaptive_reader_state.py Tests/Library/test_library_adaptive_reader_state.py
git commit -m "feat(layout): neutral adaptive-pane aliases pinned by a golden resolver grid (B0)" -m "Same-object aliases and read-only nav_open/nav_width; the digest was captured on unmodified code (shared pane-shell ADR)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

---

### Task 2: The shared shell, grip and messages (`Widgets/adaptive_pane_shell.py`)

**Files:**
- Create: `tldw_chatbook/Widgets/adaptive_pane_shell.py`
- Test: `Tests/UI/test_adaptive_pane_shell.py` (create)

**Interfaces:**
- Consumes: `AdaptivePaneLayout`, `PANE_GRIP_WIDTH`, `PaneName` from `tldw_chatbook.Utils.adaptive_reader_state` (Task 1).
- Produces:
  - `ARROW_UPPER_POSITION_RATIO: float = 0.35`
  - `@dataclass(frozen=True) class AdaptivePaneClasses(shell: str, nav: str, items: str, work: str, grip: str)`
  - `class PaneToggleRequested(Message)` with `.pane: PaneName`; `class AdaptivePaneShellResized(Message)`; `class PaneVisibilityChanged(Message)` with `.pane: PaneName`, `.open: bool`
  - `class AdaptivePaneGrip(Button)`: `__init__(pane: PaneName, *, open: bool, pane_label: str, destination_class: str, painted_names: Mapping[str, str] | None = None, extra_classes: str = "", width: int = PANE_GRIP_WIDTH, **kwargs)`; attributes `pane`, `pane_label`, `painted_names: dict[str, str]`, `grip_width`, `pane_open`; methods `sync_width(width: int)`, `sync_open(open: bool)`, `sync_label(pane_label: str)`, `painted_name() -> str`, `render() -> Content`.
  - `class AdaptivePaneShell(Horizontal)`: class attribute `grip_type: ClassVar[type[AdaptivePaneGrip]] = AdaptivePaneGrip`; `__init__(library: Widget, items: Widget, work: Widget, layout: AdaptivePaneLayout, *, id_prefix: str, library_label: str, items_label: str, destination: AdaptivePaneClasses, painted_names: Mapping[str, str] | None = None, grip_classes: str = "", **kwargs)`; attributes `destination`, `library`, `items`, `work`, `library_grip`, `items_grip`, `effective_layout`, `_applied_layout`, `_last_focused_descendant`; method `sync_layout(layout: AdaptivePaneLayout, *, manual_reopen: PaneName | None = None)`. Grip ids are `f"{id_prefix}-library-grip"` and `f"{id_prefix}-items-grip"`.

The grip and shell bodies are moved from `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py` (dev `fccf70d3b0`) with four changes only: the destination class replaces the hard-wired Library class; `painted_names` replaces the module constant; the grip records `pane_open` and gains `sync_label()`; the messages have neutral names. Everything else, including both `# ds-runtime:` comment lines and the MRO-dispatched `on_mount`/`on_resize`/`on_descendant_focus`, is byte-for-byte the Library code.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_adaptive_pane_shell.py`:

```python
"""Contracts for the shared adaptive pane shell (Roleplay frame B0).

The shell, grip and messages moved to ``Widgets/adaptive_pane_shell.py``
unchanged in behaviour; the Library keeps thin subclasses and same-object
aliases. These tests pin the shared contract with NEUTRAL probe classes and,
in later sections, the Library compatibility seam, the CSS re-key, the
promoted search input and the ``panes`` pattern family.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Button, Static

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Utils import adaptive_reader_state as ars
from tldw_chatbook.Widgets import adaptive_pane_shell as shared

ROOT = Path(__file__).resolve().parents[2]

#: Neutral probe destination: no stylesheet names these classes.
PROBE_CLASSES = shared.AdaptivePaneClasses(
    shell="probe-pane-shell",
    nav="probe-pane-nav",
    items="probe-pane-items",
    work="probe-pane-work",
    grip="probe-pane-grip",
)


def _layout(*, nav_open: bool = True, items_open: bool = True) -> ars.AdaptivePaneLayout:
    return ars.AdaptivePaneLayout(
        library_open=nav_open,
        items_open=items_open,
        library_width=28 if nav_open else 0,
        items_width=40 if items_open else 0,
        reader_width=82,
        priority_pane=None,
    )


class _StyledShellApp(ConsolidatedCSSApp):
    """The app's real CSS (bundle + every split sheet).

    The probe classes carry no rules, so the shell's height is pinned inline,
    as a destination sheet would do; the grips get ``h-full``/``p-0``/
    ``border-none`` from the boot utilities.
    """

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, painted_names: dict[str, str] | None = None) -> None:
        super().__init__()
        self.painted_names = painted_names
        self.visibility: list[tuple[str, bool]] = []
        self.toggles: list[str] = []
        self.resizes = 0

    def compose(self) -> ComposeResult:
        shell = shared.AdaptivePaneShell(
            Static("Nav"),
            Static("Items"),
            Static("Work"),
            _layout(),
            id_prefix="probe",
            library_label="Characters",
            items_label="Lore books",
            destination=PROBE_CLASSES,
            painted_names=self.painted_names,
            id="probe-shell",
        )
        shell.styles.height = 30
        yield shell

    @on(shared.PaneVisibilityChanged)
    def _visibility(self, event: shared.PaneVisibilityChanged) -> None:
        self.visibility.append((event.pane, event.open))

    @on(shared.PaneToggleRequested)
    def _toggle(self, event: shared.PaneToggleRequested) -> None:
        self.toggles.append(event.pane)

    @on(shared.AdaptivePaneShellResized)
    def _resized(self, event: shared.AdaptivePaneShellResized) -> None:
        self.resizes += 1


def _painted_column(app: App, widget: Widget) -> str:
    """The first painted character of each of ``widget``'s rows, top to bottom."""
    strips = list(app.screen._compositor.render_strips())
    column = []
    for y in range(widget.region.y, widget.region.bottom):
        text = strips[y].crop(widget.region.x, widget.region.right).text.strip()
        column.append(text[:1] or " ")
    return "".join(column)


async def test_shared_shell_puts_only_its_destination_classes_on_its_parts() -> None:
    app = _StyledShellApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", shared.AdaptivePaneShell)
        assert list(shell.children) == [
            shell.library,
            shell.library_grip,
            shell.items,
            shell.items_grip,
            shell.work,
        ]
        assert shell.has_class("probe-pane-shell")
        assert shell.library.has_class("probe-pane-nav")
        assert shell.items.has_class("probe-pane-items")
        assert shell.work.has_class("probe-pane-work")
        assert shell.library_grip.has_class("probe-pane-grip")
        assert shell.items_grip.has_class("probe-pane-grip")
        leaked = sorted(
            {
                css_class
                for node in [shell, *shell.walk_children()]
                for css_class in node.classes
                if css_class.startswith("library-")
            }
        )
        assert leaked == []


async def test_shared_shell_posts_the_neutral_messages() -> None:
    app = _StyledShellApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", shared.AdaptivePaneShell)
        assert app.resizes >= 1
        # The first sync installs a layout from nothing, so both panes report.
        assert app.visibility == [("library", True), ("items", True)]
        shell.sync_layout(_layout(nav_open=False))
        await pilot.pause()
        assert app.visibility[-1] == ("library", False)
        assert shell.library.display is False and shell.library.disabled is True
        shell.items_grip.press()
        await pilot.pause()
        assert app.toggles == ["items"]


async def test_grip_paints_the_destination_painted_name() -> None:
    app = _StyledShellApp(painted_names={"Characters": "Kinds"})
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", shared.AdaptivePaneShell)
        assert shell.library_grip.painted_name() == "Kinds"
        assert shell.library_grip.tooltip == "Collapse Characters pane"
        assert _painted_column(app, shell.items_grip).startswith("Lorebooks")


async def test_sync_label_repaints_the_painted_name_once_and_only_on_a_change() -> None:
    """``sync_open`` alone never repaints a renamed grip: nothing reactive changed."""
    app = _StyledShellApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grip = app.query_one("#probe-shell", shared.AdaptivePaneShell).items_grip
        grip.sync_label("Chat dictionaries")
        await pilot.pause()
        assert _painted_column(app, grip).startswith("Chatdictionari")
        assert grip.tooltip == "Collapse Chat dictionaries pane"
        refreshes: list[tuple] = []
        original = grip.refresh

        def counting_refresh(*args, **kwargs):
            refreshes.append(args)
            return original(*args, **kwargs)

        grip.refresh = counting_refresh
        grip.sync_label("Chat dictionaries")
        assert refreshes == []


class _FocusShellApp(App):
    def compose(self) -> ComposeResult:
        yield shared.AdaptivePaneShell(
            Vertical(Button("Nav action", id="probe-nav-action")),
            Vertical(Button("Items action", id="probe-items-action")),
            Static("Work"),
            _layout(),
            id_prefix="probe",
            library_label="Characters",
            items_label="Lore books",
            destination=PROBE_CLASSES,
            id="probe-shell",
        )


async def test_closing_a_focused_pane_moves_focus_to_its_grip() -> None:
    app = _FocusShellApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        app.query_one("#probe-nav-action", Button).focus()
        await pilot.pause()
        shell = app.query_one("#probe-shell", shared.AdaptivePaneShell)
        shell.sync_layout(_layout(nav_open=False))
        await pilot.pause()
        assert app.focused is shell.library_grip


def test_the_shared_module_carries_no_library_class_and_imports_no_library_code() -> None:
    """Roleplay imports this module: it must never pull the Library in.

    Class tokens are checked in string constants, not as a substring of the
    source: the grip ids are ``f"{id_prefix}-library-grip"``, whose constant
    piece ``-library-grip`` names the resolver's ``"library"`` pane, not a
    class, and does not start with ``library-``.
    """
    source = Path(shared.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_literals = sorted(
        token
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        for token in node.value.split()
        if token.startswith("library-")
    )
    assert class_literals == []
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden = (
        "tldw_chatbook.Widgets.Library",
        "tldw_chatbook.Library",
        "tldw_chatbook.UI.Library_Modules",
        ".Library",
    )
    assert sorted(name for name in imported if name.startswith(forbidden)) == []


def test_no_ui_ready_resident_module_imports_the_shared_shell(tmp_path: Path) -> None:
    """The UI-ready census has no headroom: residents must not import it."""
    for name in ("data", "config", "home"):
        (tmp_path / name).mkdir()
    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "HOME": str(tmp_path / "home"),
        "PYTHONPATH": str(ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)
    code = (
        "import sys\n"
        "import tldw_chatbook.Utils.adaptive_reader_state\n"
        "import tldw_chatbook.Widgets.destination_rail\n"
        "import tldw_chatbook.UI.Navigation.base_app_screen\n"
        "print('tldw_chatbook.Widgets.adaptive_pane_shell' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "False"


SHARED_WIDGET_NAMES = ["AdaptivePaneShell", "AdaptivePaneGrip"]


def test_shared_widgets_declare_no_class_level_css() -> None:
    """Every shell rule is destination-keyed in a lazy sheet: zero boot bytes."""
    for name in SHARED_WIDGET_NAMES:
        widget_class = getattr(shared, name)
        for attribute in ("DEFAULT_CSS", "CSS", "BUNDLED_CSS"):
            assert attribute not in vars(widget_class), (name, attribute)
```

- [ ] **Step 2: Run to see it fail**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider 2>&1 | tail -3`

Expected: `ERROR collecting Tests/UI/test_adaptive_pane_shell.py` — `ImportError: cannot import name 'adaptive_pane_shell' from 'tldw_chatbook.Widgets'`.

- [ ] **Step 3: Create the module**

Create `tldw_chatbook/Widgets/adaptive_pane_shell.py`:

```python
"""Shared adaptive pane shell for destination frames (Roleplay frame B0).

One three-pane structure -- a navigation rail, an items list and a work pane,
with a full-height grip after each optional pane -- shared by the Library's
adaptive readers and (from Roleplay frame B1) the Roleplay destination. The
grip, shell and messages moved here from the Library's
``Widgets/Library/library_adaptive_reader_shell.py`` unchanged in behaviour;
that module keeps the Library's names as thin subclasses and same-object
aliases.

Placement rules (the shared adaptive-pane-shell ADR, ``backlog/decisions/``):

- Never import this module from a UI-ready-resident module (for example
  ``Widgets/destination_rail.py``): the UI-ready census has no headroom.
  Destination modules import it, and they load with their own route.
- Import nothing from ``Widgets/Library/``, ``Library/`` or
  ``UI/Library_Modules/``: Roleplay imports this module and must never pull
  the Library package in with it.
- Declare no ``DEFAULT_CSS``, ``CSS`` or ``BUNDLED_CSS``. Every shell, grip
  and row rule is keyed to a destination's own classes
  (``AdaptivePaneClasses``) and lives in that destination's lazy split
  sheet, so this module costs zero boot CSS bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.content import Content
from textual.containers import Horizontal
from textual.events import DescendantFocus
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button

from tldw_chatbook.Utils.adaptive_reader_state import (
    PANE_GRIP_WIDTH,
    AdaptivePaneLayout,
    PaneName,
)

#: Where the navigation pane's upper arrow sits, as a fraction of the grip's
#: height; the lower arrow mirrors it (task-32355). The items grip paints one
#: centred arrow.
ARROW_UPPER_POSITION_RATIO = 0.35


@dataclass(frozen=True)
class AdaptivePaneClasses:
    """The destination classes one adaptive pane shell puts on its parts.

    Every rule that styles a shell is keyed to these classes and lives in the
    destination's lazy split sheet (the shared pane-shell ADR). The CSS build moves a rule to a
    lazy sheet only when every class token in its selector carries that
    sheet's owner prefix; a neutral token has no owner, so it would pin the
    rule to the boot bundle. Each value therefore starts with the
    destination's split prefix.

    Attributes:
        shell: Class on the shell container itself.
        nav: Class on the navigation rail (the resolver's ``"library"`` pane).
        items: Class on the items list (the ``"items"`` pane).
        work: Class on the work pane.
        grip: Class on both pane grips. Focus code also reads it to recognise
            a grip without a magic string.
    """

    shell: str
    nav: str
    items: str
    work: str
    grip: str


class PaneToggleRequested(Message):
    """Request a manual toggle of one optional pane.

    Attributes:
        pane: ``"library"`` (the navigation pane) or ``"items"``: the
            resolver's ``PaneName`` values, the same for every destination.
    """

    def __init__(self, pane: PaneName) -> None:
        super().__init__()
        self.pane = pane


class AdaptivePaneShellResized(Message):
    """Report that the settled shell allocation may need resolving."""


class PaneVisibilityChanged(Message):
    """Report that an optional pane's APPLIED visibility changed.

    task-32225: distinct from ``AdaptivePaneShellResized``, which the shell
    posts from ``on_resize`` -- its OWN size. A pane toggle only changes child
    widths, so nothing announced "the navigation pane is now closed" and a
    footer's narrow-stage return chip went stale in both directions. Posted
    from ``sync_layout``, the one place an applied layout is installed, so no
    destination can forget to announce it.

    Attributes:
        pane: Which optional pane changed.
        open: Its applied visibility after the change.
    """

    def __init__(self, pane: PaneName, open: bool) -> None:
        super().__init__()
        self.pane = pane
        self.open = open


class AdaptivePaneGrip(Button):
    """Narrow keyboard and pointer control for one optional pane.

    ``width`` is the destination profile's ``grip_width``: the grip paints
    exactly the columns the resolver held back for it (task-31633 AC#2). The
    grip carries its destination's grip class and paints its pane's name down
    its own column, through ``painted_names`` where the painted noun differs
    from the spoken one.
    """

    BINDINGS = [Binding("enter,space", "press", "Press button", show=False)]

    def __init__(
        self,
        pane: PaneName,
        *,
        open: bool,
        pane_label: str,
        destination_class: str,
        painted_names: Mapping[str, str] | None = None,
        extra_classes: str = "",
        width: int = PANE_GRIP_WIDTH,
        **kwargs: Any,
    ) -> None:
        """Build one pane grip sized to its destination's profile.

        Args:
            pane: Which optional pane this grip toggles -- ``"library"`` or
                ``"items"``. Carried on the ``PaneToggleRequested`` message
                the press posts.
            open: Whether that pane is open right now. Decides the arrow and
                the action copy only; geometry never changes with it.
            pane_label: Human name of the pane, used verbatim in the tooltip
                and accessible name ("Collapse Items pane").
            destination_class: The destination's grip class
                (``AdaptivePaneClasses.grip``). Always present: focus code
                reads it to know it must never restore focus onto a grip.
            painted_names: ``pane_label`` -> the noun painted down the column
                where it differs from the spoken label. ``None`` paints the
                label itself.
            extra_classes: Space-separated CSS classes appended to the
                destination class.
            width: The destination profile's ``grip_width``, in cells. Below
                four cells the arrow becomes a one-cell guillemet.
            **kwargs: Forwarded to ``Button`` (``id``, ``disabled``, ...).
        """
        self.pane = pane
        self.pane_label = pane_label
        self.painted_names: dict[str, str] = dict(painted_names or {})
        self.grip_width = width
        self.pane_open = open
        classes = destination_class
        if extra_classes:
            classes = f"{classes} {extra_classes}"
        super().__init__(compact=True, flat=True, classes=classes, **kwargs)
        self.sync_width(width)
        self.add_class("h-full")
        self.add_class("p-0")
        self.styles.line_pad = 0
        self.add_class("border-none")
        self.styles.content_align = ("center", "middle")
        self.sync_open(open)

    def sync_width(self, width: int) -> None:
        """Size the grip to the layout's reservation, in place.

        Args:
            width: The resolved layout's ``grip_width`` in cells.

        Returns:
            None.
        """
        self.grip_width = width
        # ds-runtime: profile-supplied grip columns reserved by the layout
        self.set_styles(width=width)
        self.styles.min_width = width
        self.styles.max_width = width

    def sync_open(self, open: bool) -> None:
        """Patch arrow and action copy without changing geometry.

        The in-place alternative to recomposing the grip: label, accessible
        name and tooltip are assigned only when they actually differ, so a
        shell re-sync that changes nothing costs no refresh -- and the grip
        cannot be the widget a recompose detaches while it holds focus.

        Args:
            open: Whether the pane this grip toggles is now open.

        Returns:
            None.
        """
        self.pane_open = open
        action = "Collapse" if open else "Expand"
        copy = f"{action} {self.pane_label} pane"
        # task-31633 AC#2: the arrow is as wide as the grip. The
        # "<---"/"--->" run is four cells, so a grip narrower than that would
        # paint a truncated "<" -- it takes the one-cell guillemet instead.
        if self.grip_width < len("<---"):
            label = "‹" if open else "›"
        else:
            label = "<---" if open else "--->"
        if self.label != label:
            self.label = label
        if self._name != copy:
            self._name = copy
        if self.tooltip != copy:
            self.tooltip = copy

    def sync_label(self, pane_label: str) -> None:
        """Rename the pane this grip controls, repainting only on a change.

        ``sync_open`` patches only the reactive ``label`` (the arrow), so a new
        ``pane_label`` with an unchanged open state would leave the old painted
        name on screen: nothing reactive changed. This re-derives the
        accessible copy and repaints once, and does nothing at all when the
        name is unchanged (Roleplay's list grip renames with the kind).

        Args:
            pane_label: The pane's new human name.

        Returns:
            None.
        """
        if pane_label == self.pane_label:
            return
        self.pane_label = pane_label
        self.sync_open(self.pane_open)
        self.refresh()

    def painted_name(self) -> str:
        """Return the name this grip paints down its own column.

        (task-32355) The handle used to carry its name only in ``_name`` and
        ``tooltip`` -- neither of which a terminal paints -- so every collapsed
        pane was an unexplained ``--->``. The column is a few cells wide and
        twenty to forty-five rows tall, so the name goes DOWN it.

        Returns:
            The letters to paint, one per row, already trimmed to the rows
            above the first arrow. Empty when there is no room at all.
        """
        name = self.painted_names.get(self.pane_label, self.pane_label)
        return "".join(name.split())[: max(self._first_arrow_row(), 0)]

    def _first_arrow_row(self) -> int:
        """Return the topmost row ``render`` paints an arrow on."""
        return min(self._arrow_rows())

    def _arrow_rows(self) -> set[int]:
        """Return the rows the collapse arrow is painted on."""
        height = max(self.content_region.height, 1)
        last_row = height - 1
        if self.pane == "library" and height > 1:
            upper_row = round(last_row * ARROW_UPPER_POSITION_RATIO)
            return {upper_row, last_row - upper_row}
        return {last_row // 2}

    def render(self) -> Content:
        """Paint the pane's name above the arrows it already carries.

        Returns:
            Content: Full-height grip content -- the name one letter per row
            from the top, then the arrows at the approved rows.
        """
        height = max(self.content_region.height, 1)
        arrow_rows = self._arrow_rows()
        arrow = self.label.plain
        name = self.painted_name()
        lines = [
            name[row]
            if row < len(name)
            else arrow
            if row in arrow_rows
            else " "
            for row in range(height)
        ]
        return Content.from_text("\n".join(lines))

    @on(Button.Pressed)
    def request_toggle(self, event: Button.Pressed) -> None:
        """Translate native Button activation into the shell message."""
        if event.button is not self:
            return
        event.stop()
        self.post_message(PaneToggleRequested(self.pane))


class AdaptivePaneShell(Horizontal):
    """Own adaptive pane structure while callers own state and behavior.

    A destination supplies its own ``AdaptivePaneClasses``; a destination that
    needs its own grip TYPE (the Library keeps ``LibraryAdaptiveReaderPaneGrip``
    so type queries by that name still match) overrides ``grip_type``. A
    subclass must not define ``on_mount``, ``on_resize`` or
    ``on_descendant_focus``: Textual dispatches each once per class in the MRO
    that defines one, so a redefinition would run the shared body twice.
    """

    #: The grip class this shell builds.
    grip_type: ClassVar[type[AdaptivePaneGrip]] = AdaptivePaneGrip

    def __init__(
        self,
        library: Widget,
        items: Widget,
        work: Widget,
        layout: AdaptivePaneLayout,
        *,
        id_prefix: str,
        library_label: str,
        items_label: str,
        destination: AdaptivePaneClasses,
        painted_names: Mapping[str, str] | None = None,
        grip_classes: str = "",
        **kwargs: Any,
    ) -> None:
        """Assemble the three-pane shell around caller-owned pane widgets.

        Args:
            library: Widget for the leftmost (navigation rail) pane.
            items: Widget for the middle (list) pane.
            work: Widget for the work pane.
            layout: The resolved layout to mount with: which optional panes
                are open and how wide each is.
            id_prefix: Per-destination id stem for the composed grips, giving
                each destination its own stable selectors.
            library_label: Human name of the navigation pane, for grip copy.
            items_label: Human name of the items pane, for grip copy.
            destination: The destination's classes for every part.
            painted_names: Passed to both grips (see ``AdaptivePaneGrip``).
            grip_classes: Extra CSS classes for both grips.
            **kwargs: Forwarded to ``Horizontal`` (``id``, ``classes``, ...).

        Both grips are sized from ``layout.grip_width`` -- the width the
        resolver held back for them (task-31952 AC#3), so a caller cannot
        paint a grip the resolver never reserved.
        """
        super().__init__(**kwargs)
        self.destination = destination
        self.add_class(destination.shell)
        self.library = library
        self.items = items
        self.work = work
        self.library.add_class(destination.nav)
        self.items.add_class(destination.items)
        self.work.add_class(destination.work)
        self.library_grip = self.grip_type(
            "library",
            open=layout.library_open,
            pane_label=library_label,
            destination_class=destination.grip,
            painted_names=painted_names,
            extra_classes=grip_classes,
            width=layout.grip_width,
            id=f"{id_prefix}-library-grip",
        )
        self.items_grip = self.grip_type(
            "items",
            open=layout.items_open,
            pane_label=items_label,
            destination_class=destination.grip,
            painted_names=painted_names,
            extra_classes=grip_classes,
            width=layout.grip_width,
            id=f"{id_prefix}-items-grip",
        )
        self._last_focused_descendant: dict[PaneName, Widget | None] = {
            "library": None,
            "items": None,
        }
        self.effective_layout = layout
        self._applied_layout: AdaptivePaneLayout | None = None

    def compose(self) -> ComposeResult:
        """Compose retained navigation, items, grips, and work widgets."""
        yield self.library
        yield self.library_grip
        yield self.items
        yield self.items_grip
        yield self.work

    def on_mount(self) -> None:
        """Apply initial geometry and request a settled resize projection."""
        self.sync_layout(self.effective_layout)
        self.call_after_refresh(self.post_message, AdaptivePaneShellResized())

    def on_resize(self) -> None:
        """Request layout resolution after the shell allocation changes."""
        self.post_message(AdaptivePaneShellResized())

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Remember optional-pane focus before a grip activation moves it."""
        target = event.widget
        for pane_name, pane in (
            ("library", self.library),
            ("items", self.items),
        ):
            if self._is_valid_focus_target(pane, target):
                self._last_focused_descendant[pane_name] = target
                return

    def _pane_focus_chain(self, pane: Widget) -> list[Widget]:
        """Return currently reachable pane targets in Textual focus order."""
        if not self.is_mounted:
            return []
        return [
            target
            for target in self.app.screen.focus_chain
            if target is pane or pane in target.ancestors
        ]

    def _is_valid_focus_target(self, pane: Widget, target: Widget | None) -> bool:
        """Return whether ``target`` is currently reachable within ``pane``."""
        return target is not None and target in self._pane_focus_chain(pane)

    def sync_layout(
        self,
        layout: AdaptivePaneLayout,
        *,
        manual_reopen: PaneName | None = None,
    ) -> None:
        """Patch pane display and exact cell widths in place."""
        previous_layout = self._applied_layout
        self.effective_layout = layout
        focused = self.app.focused if self.is_mounted else None
        evacuation_target: Widget | None = None
        manual_reopen_pane: Widget | None = None
        manual_reopen_name: PaneName | None = None
        automatic_reopen_target: Widget | None = None
        for pane_name, pane, grip, open, width, was_open in (
            (
                "library",
                self.library,
                self.library_grip,
                layout.library_open,
                layout.library_width,
                (
                    previous_layout.library_open
                    if previous_layout is not None
                    else layout.library_open
                ),
            ),
            (
                "items",
                self.items,
                self.items_grip,
                layout.items_open,
                layout.items_width,
                (
                    previous_layout.items_open
                    if previous_layout is not None
                    else layout.items_open
                ),
            ),
        ):
            if (
                not open
                and focused is not None
                and (focused is pane or pane in focused.ancestors)
            ):
                if focused is not pane and self._is_valid_focus_target(pane, focused):
                    self._last_focused_descendant[pane_name] = focused
                evacuation_target = grip
            if pane.display != open:
                pane.display = open
            if pane.disabled != (not open):
                pane.disabled = not open
            if pane.styles.width is None or pane.styles.width.value != width:
                pane.remove_class("w-fill", "w-3fr")
                # ds-runtime: pane columns resolved from the measured shell width
                pane.set_styles(width=width)
            if pane.styles.min_width is None or pane.styles.min_width.value != width:
                pane.styles.min_width = width
            if pane.styles.max_width is None or pane.styles.max_width.value != width:
                pane.styles.max_width = width
            if previous_layout is None:
                pane.add_class("h-full")
            if open and not was_open and focused is grip:
                automatic_reopen_target = next(
                    (
                        candidate
                        for candidate in self.screen.focus_chain
                        if (candidate is pane or pane in candidate.ancestors)
                        and candidate.display
                        and not candidate.disabled
                    ),
                    grip,
                )
            if grip.grip_width != layout.grip_width:
                # Reserve-and-paint holds for every layout the shell is given,
                # not only the one it was built with (task-31952 AC#3).
                grip.sync_width(layout.grip_width)
            grip.sync_open(open)
            if open and not was_open and manual_reopen == pane_name:
                manual_reopen_pane = pane
                manual_reopen_name = pane_name
        if previous_layout is None:
            self.work.display = True
            self.work.add_class("w-fill")
            self.work.styles.min_width = 0
            self.work.add_class("h-full")
        for pane_name, was_open, now_open in (
            (
                "library",
                None if previous_layout is None else previous_layout.library_open,
                layout.library_open,
            ),
            (
                "items",
                None if previous_layout is None else previous_layout.items_open,
                layout.items_open,
            ),
        ):
            if was_open != now_open:
                self.post_message(PaneVisibilityChanged(pane_name, now_open))
        self._applied_layout = layout
        if evacuation_target is not None:
            self.screen.set_focus(evacuation_target, scroll_visible=False)
        elif manual_reopen_pane is not None and manual_reopen_name is not None:
            focus_chain = self._pane_focus_chain(manual_reopen_pane)
            target = self._last_focused_descendant[manual_reopen_name]
            if target not in focus_chain:
                target = next(iter(focus_chain), None)
            if target is not None:
                self.screen.set_focus(target, scroll_visible=False)
        elif automatic_reopen_target is not None:
            # Keep focus recovery synchronous so it cannot overwrite a newer
            # explicit focus change queued before the next refresh.
            self.screen.set_focus(automatic_reopen_target, scroll_visible=False)
```

- [ ] **Step 4: Run to see it pass**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider 2>&1 | tail -2`

Expected: `8 passed`. (Run on its own, without the module-scope `import tldw_chatbook.app`, every test here would error at setup with `RecoveryRequired: raw_source_selection_changed`; with it, the file passes alone and in any company.)

- [ ] **Step 5: Named mutations (record each in `$EV/mutations.md`)**

1. `sync-label-no-refresh`: delete the line `        self.refresh()` at the end of `AdaptivePaneGrip.sync_label`. Run `…/python -m pytest "Tests/UI/test_adaptive_pane_shell.py::test_sync_label_repaints_the_painted_name_once_and_only_on_a_change" -q -p no:cacheprovider`. Expected `1 failed` (the painted column still starts with `Lorebooks`). Restore the line with the Edit tool; re-run: `1 passed`. (Verified in a scratch prototype on `fccf70d3b0`.)
2. `no-evacuation`: change `                evacuation_target = grip` to `                evacuation_target = None`. Run `…::test_closing_a_focused_pane_moves_focus_to_its_grip`. Expected `1 failed`. Restore; `1 passed`.
3. `resident-imports-shell`: add the line `import tldw_chatbook.Widgets.adaptive_pane_shell  # noqa: F401` after the existing imports of `tldw_chatbook/Widgets/destination_rail.py`. Run `…::test_no_ui_ready_resident_module_imports_the_shared_shell`. Expected `1 failed` (`'True' == 'False'`). Remove the line; `1 passed`. (`destination_rail.py` is a never-place file: confirm `git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 diff --stat tldw_chatbook/Widgets/destination_rail.py` prints nothing afterwards.)
4. `ignore-painted-names`: in `AdaptivePaneGrip.painted_name`, change `        name = self.painted_names.get(self.pane_label, self.pane_label)` to `        name = self.pane_label`. Run `…::test_grip_paints_the_destination_painted_name` (or `-k destination_painted_name`). Expected `1 failed` (`assert 'Characters' == 'Kinds'`: the grip paints its spoken label). Restore; `1 passed`.
5. `library-class-literal`: add the line `_PROBE = "library-adaptive-reader-shell"` after `ARROW_UPPER_POSITION_RATIO = 0.35`. Run `…::test_the_shared_module_carries_no_library_class_and_imports_no_library_code`. Expected `1 failed` naming `library-adaptive-reader-shell` (and not the grip id's `-library-grip`). Remove the line; `1 passed`.

- [ ] **Step 6: Lint**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Widgets/adaptive_pane_shell.py Tests/UI/test_adaptive_pane_shell.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
git add tldw_chatbook/Widgets/adaptive_pane_shell.py Tests/UI/test_adaptive_pane_shell.py
git commit -m "feat(widgets): shared adaptive pane shell, grip and messages (B0)" -m "Moved from the Library shell with destination classes, painted_names and an equality-guarded sync_label (shared pane-shell ADR)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

---

### Task 3: Library compatibility — thin subclasses and same-object aliases

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py` (whole file replaced)
- Modify (docstrings only): `tldw_chatbook/Widgets/Library/library_browse_reader_shell.py` (`LibraryBrowseReaderShell.on_mount`), `Tests/UI/test_on_mount_mro_convention.py` (module docstring)
- Modify (the pre-import raise, Step 6): `Tests/Performance/test_screen_preimport_payload_budget.py`, `backlog/decisions/097-boot-budget-ratchets.md`, `Tests/Performance/boot_budget_snapshots/preimport_payload.json`
- Test: `Tests/UI/test_adaptive_pane_shell.py` (import block; append)

**Interfaces:**
- Consumes: everything Task 2 produced; `$EV/preimport_measure.sh`, `$EV/preimport_raise.py` and `$EV/owner-signoff.txt` (Task 0).
- Produces (in `tldw_chatbook.Widgets.Library.library_adaptive_reader_shell`):
  - `LIBRARY_ADAPTIVE_READER_CLASSES: AdaptivePaneClasses` = `("library-adaptive-reader-shell", "library-adaptive-reader-library", "library-adaptive-reader-items", "library-adaptive-reader-work", "library-adaptive-reader-pane-grip")` for `(shell, nav, items, work, grip)`
  - `LIBRARY_ADAPTIVE_READER_GRIP_CLASS: str` (unchanged value), `LIBRARY_PANE_GRIP_NAMES = {"Library": "Nav"}`, `LIBRARY_ARROW_UPPER_POSITION_RATIO`
  - `AdaptiveReaderShellResized is AdaptivePaneShellResized`, `LibraryPaneVisibilityChanged is PaneVisibilityChanged`, `PaneToggleRequested` re-exported
  - `class LibraryAdaptiveReaderPaneGrip(AdaptivePaneGrip)`: `__init__(pane, *, open, pane_label, extra_classes="", width=PANE_GRIP_WIDTH, destination_class=LIBRARY_ADAPTIVE_READER_GRIP_CLASS, painted_names=LIBRARY_PANE_GRIP_NAMES, **kwargs)`
  - `class LibraryAdaptiveReaderShell(AdaptivePaneShell)`: `grip_type = LibraryAdaptiveReaderPaneGrip`; `__init__(library, items, work, layout, *, id_prefix, library_label, items_label, grip_classes="", **kwargs)` (the old signature).

The Library class names stay exactly as they were, so `_library.tcss`, every literal-class test, the crit8 bare grip host and the boot bytes are unchanged (B0 research: zero CSS diff).

- [ ] **Step 1: Write the failing tests**

Edit the import block of `Tests/UI/test_adaptive_pane_shell.py`:

Old:
```python
import os
import subprocess
```
New:
```python
import os
import re
import subprocess
```

Old:
```python
from tldw_chatbook.Utils import adaptive_reader_state as ars
from tldw_chatbook.Widgets import adaptive_pane_shell as shared
```
New:
```python
from tldw_chatbook.Utils import adaptive_reader_state as ars
from tldw_chatbook.Widgets import adaptive_pane_shell as shared
from tldw_chatbook.Widgets.Library import library_adaptive_reader_shell as library_shell
from tldw_chatbook.Widgets.Library.library_browse_reader_shell import MediaShellResized
```

Append:

```bash
cat >> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0/Tests/UI/test_adaptive_pane_shell.py <<'EOF'


# ---------------------------------------------------------------------------
# Library compatibility (B0): thin subclasses and same-object aliases.
# ---------------------------------------------------------------------------


def test_library_message_names_are_the_shared_message_objects() -> None:
    """``@on`` matches class identity; a subclass alias would stop matching."""
    assert library_shell.LibraryPaneVisibilityChanged is shared.PaneVisibilityChanged
    assert library_shell.AdaptiveReaderShellResized is shared.AdaptivePaneShellResized
    assert library_shell.PaneToggleRequested is shared.PaneToggleRequested
    assert MediaShellResized is shared.AdaptivePaneShellResized


def test_library_widgets_are_thin_subclasses_of_the_shared_widgets() -> None:
    assert issubclass(library_shell.LibraryAdaptiveReaderShell, shared.AdaptivePaneShell)
    assert issubclass(library_shell.LibraryAdaptiveReaderPaneGrip, shared.AdaptivePaneGrip)
    assert (
        library_shell.LibraryAdaptiveReaderShell.grip_type
        is library_shell.LibraryAdaptiveReaderPaneGrip
    )
    assert library_shell.LIBRARY_ADAPTIVE_READER_GRIP_CLASS == (
        library_shell.LIBRARY_ADAPTIVE_READER_CLASSES.grip
    ) == "library-adaptive-reader-pane-grip"
    # Textual dispatches these once per MRO class that defines one.
    for handler in ("on_mount", "on_resize", "on_descendant_focus"):
        assert handler not in vars(library_shell.LibraryAdaptiveReaderShell), handler


async def test_library_grip_builds_its_own_destination_class_and_nav_name() -> None:
    """A bare Library grip (the crit8 host shape) still carries its class."""

    class _GripHost(App):
        def compose(self) -> ComposeResult:
            yield library_shell.LibraryAdaptiveReaderPaneGrip(
                "library", open=True, pane_label="Library", width=1, id="bare-grip"
            )

    app = _GripHost()
    async with app.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        grip = app.query_one("#bare-grip", library_shell.LibraryAdaptiveReaderPaneGrip)
        assert grip.has_class("library-adaptive-reader-pane-grip")
        assert grip.painted_names == {"Library": "Nav"}


#: Naming-convention handler names for the aliased messages, in both
#: directions, in Textual's public and private (``_on_``) forms.
_ALIASED_MESSAGE_HANDLER = re.compile(
    r"def _?on_(adaptive_reader_shell_resized|library_pane_visibility_changed"
    r"|pane_visibility_changed|adaptive_pane_shell_resized)\b"
)


def test_no_naming_convention_handler_exists_for_the_aliased_messages() -> None:
    """``@on`` matches class identity; naming-convention handlers key on ``__name__``.

    After the aliasing, a handler named for a retired Library name never
    fires, and one named for a shared name would START receiving every
    Library post on any ancestor (App, a screen, a test host) without anyone
    binding it. Textual's dispatcher also looks up the ``_on_`` form. Both
    directions, both the package and the tests.
    """
    hits = sorted(
        f"{path.relative_to(ROOT)}: {match.group(0)}"
        for tree in ("tldw_chatbook", "Tests")
        for path in (ROOT / tree).rglob("*.py")
        for match in _ALIASED_MESSAGE_HANDLER.finditer(
            path.read_text(encoding="utf-8", errors="ignore")
        )
    )
    assert hits == []


class _LibraryAliasHost(App):
    """Handlers bound exactly the way ``LibraryScreen`` binds them."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def compose(self) -> ComposeResult:
        yield library_shell.LibraryAdaptiveReaderShell(
            Static("Library"),
            Static("Items"),
            Static("Work"),
            _layout(),
            id_prefix="alias",
            library_label="Library",
            items_label="Items",
            id="alias-shell",
        )

    @on(library_shell.LibraryPaneVisibilityChanged)
    def _visibility(self, event) -> None:
        self.events.append(f"visibility:{event.pane}:{event.open}")

    @on(library_shell.AdaptiveReaderShellResized)
    def _resized_reader(self, event) -> None:
        self.events.append("resized:reader")

    @on(MediaShellResized)
    def _resized_media(self, event) -> None:
        self.events.append("resized:media")


async def test_library_handlers_bound_to_aliases_still_fire_for_the_shared_shell() -> None:
    app = _LibraryAliasHost()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert "visibility:library:True" in app.events
        assert "visibility:items:True" in app.events
        # Two handlers on one (aliased) class both run, as on LibraryScreen.
        assert "resized:reader" in app.events and "resized:media" in app.events
        assert len(app.query(library_shell.LibraryAdaptiveReaderPaneGrip)) == 2
        shell = app.query_one("#alias-shell", library_shell.LibraryAdaptiveReaderShell)
        shell.sync_layout(_layout(nav_open=False))
        await pilot.pause()
        assert "visibility:library:False" in app.events
EOF
```

- [ ] **Step 2: Run to see the new tests fail**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider 2>&1 | tail -8`

Expected: `3 failed, 10 passed`. FAILED: `test_library_message_names_are_the_shared_message_objects` (`assert <class '…LibraryPaneVisibilityChanged'> is <class '…PaneVisibilityChanged'>`), `test_library_widgets_are_thin_subclasses_of_the_shared_widgets` (`issubclass` is False), `test_library_grip_builds_its_own_destination_class_and_nav_name` (`AttributeError: … has no attribute 'painted_names'`). `test_no_naming_convention_handler_exists_for_the_aliased_messages` and `test_library_handlers_bound_to_aliases_still_fire_for_the_shared_shell` already pass: they pin behaviour that must survive the move (Step 5's mutations show both discriminate).

- [ ] **Step 3: Replace `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py`**

Write the whole file:

```python
"""Library names for the shared adaptive pane shell (Roleplay frame B0).

The retained three-role structure that lived here -- grip, shell and their
messages -- moved unchanged to ``tldw_chatbook.Widgets.adaptive_pane_shell``,
so a second destination (Roleplay) can share it without importing this
package. This module keeps every Library name working:

- The messages are the SAME objects (``LibraryPaneVisibilityChanged is
  PaneVisibilityChanged``). Every Library handler binds with ``@on(...)``,
  which matches on class identity, so nothing re-routes; a subclass alias
  would silently stop matching the shared shell's posts.
- The widgets are thin subclasses that supply only the Library's destination
  classes (``LIBRARY_ADAPTIVE_READER_CLASSES``) and painted names. They stay
  real classes because ``query(LibraryAdaptiveReaderPaneGrip)`` matches on the
  class NAME, and because Library code and tests construct them directly.
- Neither subclass defines ``on_mount``, ``on_resize`` or
  ``on_descendant_focus``: Textual dispatches those once per class in the MRO
  that defines one, so a redefinition here would run the shared body twice.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from textual.widget import Widget

from tldw_chatbook.Utils.adaptive_reader_state import (
    PANE_GRIP_WIDTH,
    AdaptiveReaderEffectiveLayout,
    PaneName,
)
from tldw_chatbook.Widgets.adaptive_pane_shell import (
    ARROW_UPPER_POSITION_RATIO,
    AdaptivePaneClasses,
    AdaptivePaneGrip,
    AdaptivePaneShell,
    AdaptivePaneShellResized,
    PaneToggleRequested,
    PaneVisibilityChanged,
)

__all__ = [
    "LIBRARY_ADAPTIVE_READER_CLASSES",
    "LIBRARY_ADAPTIVE_READER_GRIP_CLASS",
    "LIBRARY_ARROW_UPPER_POSITION_RATIO",
    "LIBRARY_PANE_GRIP_NAMES",
    "AdaptiveReaderShellResized",
    "LibraryAdaptiveReaderPaneGrip",
    "LibraryAdaptiveReaderShell",
    "LibraryPaneVisibilityChanged",
    "PaneToggleRequested",
]

LIBRARY_ARROW_UPPER_POSITION_RATIO = ARROW_UPPER_POSITION_RATIO

#: The Library's destination classes (Roleplay frame B0). These are the class names the
#: Library always used, so ``css/features/_library.tcss`` matches them
#: unchanged, every token keeps the Library split prefix (the rules stay in the
#: lazy ``screen_agentic_library.tcss``), and boot CSS does not move.
LIBRARY_ADAPTIVE_READER_CLASSES = AdaptivePaneClasses(
    shell="library-adaptive-reader-shell",
    nav="library-adaptive-reader-library",
    items="library-adaptive-reader-items",
    work="library-adaptive-reader-work",
    grip="library-adaptive-reader-pane-grip",
)

#: Shared class every Library adaptive reader shell puts on BOTH of its pane
#: grips. Named here so focus code can recognise a grip without a magic string
#: (task-31567: the grips are the shell's first focusable widgets, so a
#: recompose hands them focus unless someone puts it back).
LIBRARY_ADAPTIVE_READER_GRIP_CLASS = LIBRARY_ADAPTIVE_READER_CLASSES.grip

#: (task-32355) What a grip PAINTS where its pane's spoken name would not fit
#: or would not be the name the guide uses. The Library pane's tooltip says
#: "Expand Library pane"; the handle itself is the "Nav" handle
#: (``Docs/User_Guide/library.md``), which is also the only form that reads in
#: a five-cell column.
LIBRARY_PANE_GRIP_NAMES = {"Library": "Nav"}

#: Same objects as the shared messages, never subclasses (module docstring).
AdaptiveReaderShellResized = AdaptivePaneShellResized
LibraryPaneVisibilityChanged = PaneVisibilityChanged


class LibraryAdaptiveReaderPaneGrip(AdaptivePaneGrip):
    """The Library's pane grip: the shared grip with the Library's class and names."""

    def __init__(
        self,
        pane: PaneName,
        *,
        open: bool,
        pane_label: str,
        extra_classes: str = "",
        width: int = PANE_GRIP_WIDTH,
        destination_class: str = LIBRARY_ADAPTIVE_READER_GRIP_CLASS,
        painted_names: Mapping[str, str] | None = LIBRARY_PANE_GRIP_NAMES,
        **kwargs: Any,
    ) -> None:
        """Build one Library grip; see ``AdaptivePaneGrip.__init__``.

        The two Library defaults are the only additions: the grip always
        carries ``LIBRARY_ADAPTIVE_READER_GRIP_CLASS`` (a bare grip built
        outside any shell still gets its ``:focus`` rule) and paints "Nav" for
        the Library pane.
        """
        super().__init__(
            pane,
            open=open,
            pane_label=pane_label,
            destination_class=destination_class,
            painted_names=painted_names,
            extra_classes=extra_classes,
            width=width,
            **kwargs,
        )


class LibraryAdaptiveReaderShell(AdaptivePaneShell):
    """The Library's adaptive reader shell: the shared shell, Library-keyed."""

    grip_type: ClassVar[type[AdaptivePaneGrip]] = LibraryAdaptiveReaderPaneGrip

    def __init__(
        self,
        library: Widget,
        items: Widget,
        work: Widget,
        layout: AdaptiveReaderEffectiveLayout,
        *,
        id_prefix: str,
        library_label: str,
        items_label: str,
        grip_classes: str = "",
        **kwargs: Any,
    ) -> None:
        """Assemble a Library shell; see ``AdaptivePaneShell.__init__``.

        Keeps the pre-B0 signature: every Library caller and subclass
        (``LibraryBrowseReaderShell``, ``LibraryArtifactsReaderShell``) builds
        it exactly as before.
        """
        super().__init__(
            library,
            items,
            work,
            layout,
            id_prefix=id_prefix,
            library_label=library_label,
            items_label=items_label,
            destination=LIBRARY_ADAPTIVE_READER_CLASSES,
            painted_names=LIBRARY_PANE_GRIP_NAMES,
            grip_classes=grip_classes,
            **kwargs,
        )
```

- [ ] **Step 4: Run the new tests and the closest Library suites**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider 2>&1 | tail -1`

Expected: `13 passed`.

Then compare the closest Library suites against the base arm:

```bash
bash <<'EOF'
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence
cat > "$EV/suites-shell-quick.txt" <<'X'
Tests/UI/test_library_adaptive_reader_shell.py
Tests/UI/test_library_crit8_keyboard.py
Tests/UI/test_library_crit10_layout.py
Tests/UI/test_library_notes_w3_layout.py
Tests/UI/test_library_media_reader_shell.py
Tests/UI/test_library_collections_reader_geometry.py
Tests/UI/test_library_layout_repair.py
Tests/UI/test_library_media_render_fixes.py
Tests/UI/test_on_mount_mro_convention.py
X
"$EV/paired.sh" shellquick "$EV/suites-shell-quick.txt" -n 4
EOF
```

Expected: a `shellquick base:` and a `shellquick head:` summary line, each with `recovery=` in single digits (the collection plugin removes the `RecoveryRequired: raw_source_selection_changed` setup wall that plain pytest shows on BOTH arms), and nothing between `new failures on head [shellquick] (must be empty):` and `(end of new failures [shellquick])`. Failures present on both arms are fine.

- [ ] **Step 5: Named mutations `subclass-alias` and `naming-convention-handler`**

1. `subclass-alias`: in `library_adaptive_reader_shell.py` replace `LibraryPaneVisibilityChanged = PaneVisibilityChanged` with:

```python
class LibraryPaneVisibilityChanged(PaneVisibilityChanged):
    """Mutation only."""
```

Run `…/python -m pytest "Tests/UI/test_adaptive_pane_shell.py::test_library_handlers_bound_to_aliases_still_fire_for_the_shared_shell" "Tests/UI/test_adaptive_pane_shell.py::test_library_message_names_are_the_shared_message_objects" -q -p no:cacheprovider`. Expected: `2 failed` (no `visibility:` events at all; identity false). Restore the alias line with the Edit tool; `2 passed`.

2. `naming-convention-handler`: in `Tests/UI/test_adaptive_pane_shell.py`, add to `_LibraryAliasHost` a method `def on_pane_visibility_changed(self, event) -> None:` whose body is `pass`. Run `…/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider -k naming_convention`. Expected `1 failed` naming `Tests/UI/test_adaptive_pane_shell.py: def on_pane_visibility_changed`. Remove the method; `1 passed`.

Record both.

- [ ] **Step 6: Fix the prose B0 makes false, then the pre-import checkpoint (spec Q10)**

Two docstrings name `LibraryAdaptiveReaderShell` as the class whose `on_mount` is defined in its own `__dict__`; after this task that `on_mount` lives on `AdaptivePaneShell`.

In `tldw_chatbook/Widgets/Library/library_browse_reader_shell.py`:

Old:
```python
        No ``super().on_mount()``: Textual's dispatcher already invokes
        ``LibraryAdaptiveReaderShell.on_mount`` separately for this Mount
        event (TASK-31822).
```
New:
```python
        No ``super().on_mount()``: Textual's dispatcher already invokes
        ``AdaptivePaneShell.on_mount`` (the shared shell the Library shell
        subclasses) separately for this Mount event (TASK-31822).
```

In `Tests/UI/test_on_mount_mro_convention.py`:

Old:
```python
defined in its own class ``__dict__`` (SafeModalDismissMixin or
LibraryAdaptiveReaderShell), so all 19 were redundant and removed. The one
```
New:
```python
defined in its own class ``__dict__`` (SafeModalDismissMixin or
LibraryAdaptiveReaderShell, whose ``on_mount`` now lives on the shared
``AdaptivePaneShell``), so all 19 were redundant and removed. The one
```

Then the pre-import checkpoint. From this commit on, the Library route imports `tldw_chatbook.Widgets.adaptive_pane_shell`, so the required guard `test_preimport_pass_payload_stays_within_budget` (perf-guard.yml) is red until a constant is raised, and every later commit would carry that red onto dev. The raise therefore lands in THIS commit. `$EV/owner-signoff.txt` must hold the owner's verbatim answer from Task 0 Step 6; if it is empty or the owner declined, STOP here without committing and report to the controller.

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b0-evidence
test -s "$EV/owner-signoff.txt" || { echo "STOP: no owner sign-off yet (Task 0 Step 6)"; exit 1; }
"$EV/preimport_measure.sh" base && "$EV/preimport_measure.sh" head
"$PY" "$EV/preimport_raise.py"
cd $MAIN/.worktrees/roleplay-b0
"$PY" scripts/update_boot_budget_snapshots.py --only preimport 2>&1 | tail -3
"$PY" -m pytest Tests/Performance/test_screen_preimport_payload_budget.py -p no:cacheprovider -q 2>&1 | tail -1
git diff --stat -- Tests/Performance backlog/decisions/097-boot-budget-ratchets.md
EOF
```

Expected (on `fccf70d3b0`): `base | modules 557 | … | library 176 mods / …`, `head | modules 558 | … | library 177 mods / …`; `raised: MAX_PASS_ADDED_MODULES 557 -> 558 | ledger row: written`; the snapshot script's refresh line; `1 passed`; a diff stat naming the test file, `097-boot-budget-ratchets.md` and `preimport_payload.json`. If #2862 landed first, the `raised:` line names the LOC constant(s) instead, which is also correct; any `STOP:` line means stop and report (an unexpected module, growth beyond the approved bound, or no sign-off).

- [ ] **Step 7: Lint and commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py tldw_chatbook/Widgets/Library/library_browse_reader_shell.py Tests/UI/test_adaptive_pane_shell.py Tests/UI/test_on_mount_mro_convention.py Tests/Performance/test_screen_preimport_payload_budget.py
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence/plug /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_on_mount_mro_convention.py -p b0_bootstrap_all -q -p no:cacheprovider 2>&1 | tail -1
git add tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py tldw_chatbook/Widgets/Library/library_browse_reader_shell.py Tests/UI/test_adaptive_pane_shell.py Tests/UI/test_on_mount_mro_convention.py Tests/Performance/test_screen_preimport_payload_budget.py backlog/decisions/097-boot-budget-ratchets.md Tests/Performance/boot_budget_snapshots/preimport_payload.json
git commit -m "refactor(library): adaptive reader shell becomes thin subclasses of the shared shell (B0)" -m "Same-object message aliases keep every @on handler bound; Library class names are the Library destination classes (shared pane-shell ADR). The owner-approved pre-import raise lands in this same commit (ADR-097 ledger), so the perf guard is never red on the branch." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: `All checks passed!`; the MRO-convention file at `1 failed, 1 passed`, where the failure is the pre-existing `test_no_screen_reintroduces_super_on_mount` (red on dev and on the base arm with the same four offending sites: three `Persona_Widgets` buddy modals and `console_endpoint_template_modal.py`; B0 adds none); then the commit line.

---

### Task 4: Rail-row primitives — `DestinationRailRow`, `fit_rail_row_label`, `DestinationRailRowButton`

**Files:**
- Modify: `tldw_chatbook/Widgets/adaptive_pane_shell.py` (imports; append)
- Test: `Tests/UI/test_destination_rail_row.py` (create); `Tests/UI/test_adaptive_pane_shell.py` (`SHARED_WIDGET_NAMES`)

**Interfaces:**
- Consumes: `resolve_glyph` from `tldw_chatbook.Widgets.glyph_fallback` (existing, UI-ready resident, so importing it adds no module); `rich.cells.cell_len`.
- Produces:
  - `RAIL_ROW_CURRENT_MARKER = "▸"`, `RAIL_ROW_ELLIPSIS = "…"`
  - `@dataclass(frozen=True) class DestinationRailRow(row_id: str, title: str, short_title: str = "", count: str = "", short_count: str = "", key: str = "", count_loading: bool = False, disabled: bool = False)`
  - `@dataclass(frozen=True) class FittedRailRowLabel(prefix: str, title: str, count: str, key: str)` with property `plain -> str`
  - `fit_rail_row_label(row: DestinationRailRow, width: int, *, current: bool = False) -> FittedRailRowLabel`
  - `rail_row_content(row: DestinationRailRow, width: int, *, current: bool = False) -> Content` (literal text, never markup; `.plain == fit_rail_row_label(...).plain`)
  - `class DestinationRailRowButton(Button)`: `__init__(row: DestinationRailRow, *, current: bool = False, **kwargs)`; attributes `rail_row`, `is_current`; `sync_row(row: DestinationRailRow, *, current: bool) -> None`; refits on `Resize`.

Decisions this task pins (spec §1.4.3 leaves them open; recorded in the ADR): the prefix (`"▸ "` current, `"  "` otherwise, via `resolve_glyph`) counts toward the width — that is the only reading under which the spec's RC-11 sentence holds; the key hint is emitted as two spaces + the letter (the minimal form the spec's examples print), right alignment being the row button's styling job in B6; counts are caller-formatted strings (B6 owns the count states); the ellipsis is `…`, measured in cells, and has no ASCII substitute (`glyph_fallback.ASCII_GLYPH_FALLBACKS` has no entry for it, so `resolve_glyph` returns it unchanged and ASCII mode paints `…` too); the row button carries the canonical `w-full` and `h-1` sizing utilities and `line_pad = 0` so its text width is its rail width; a style-only change (a count that starts or stops loading with unchanged text) still repaints, because `Content.__eq__` and the `label` reactive compare plain text only.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_destination_rail_row.py`:

```python
"""``fit_rail_row_label`` and ``DestinationRailRowButton`` (Roleplay frame B0).

Rail text width is pane width minus 4 (round border + row padding): 24 / 31 /
35 cells at 120 / 160 / >=180 columns (spec section 1.4.1). The fallback order
is spec section 1.4.3: title + count + key -> drop the key -> short count ->
short title -> short title + short count -> ellipsis. The canonical noun
outlives the key hint, and the count is never clipped. The two-cell row
prefix counts toward the width.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from rich.cells import cell_len
from textual.app import ComposeResult
from textual.containers import Vertical

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Widgets import glyph_fallback
from tldw_chatbook.Widgets.adaptive_pane_shell import (
    DestinationRailRow,
    DestinationRailRowButton,
    fit_rail_row_label,
    rail_row_content,
)

CHARACTERS = DestinationRailRow(row_id="characters", title="Characters", count="(28)", key="c")
PERSONAS = DestinationRailRow(row_id="personas", title="Personas", count="(4)", key="p")
LORE = DestinationRailRow(
    row_id="lore",
    title="Lore books",
    short_title="Lore",
    count="(3 · 2 on)",
    short_count="(3)",
    key="l",
)
DICTIONARIES = DestinationRailRow(
    row_id="dictionaries",
    title="Chat dictionaries",
    short_title="Dictionaries",
    count="(3 · 2 on)",
    short_count="(3)",
    key="d",
)
ROWS = (CHARACTERS, PERSONAS, LORE, DICTIONARIES)

#: A row whose short title is much shorter than its title, so every one of
#: the six steps is reachable at some width.
FALLBACK = DestinationRailRow(
    row_id="fallback",
    title="Chat dictionaries",
    short_title="Dicts",
    count="(3 · 2 on)",
    short_count="(3)",
    key="d",
)

WIDE = (
    DestinationRailRow(row_id="cjk", title="漢字のキャラクター名", count="(12)", key="c"),
    DestinationRailRow(row_id="emoji", title="🎭 Masks 🎭 Theatre", count="(3)", key="m"),
    DestinationRailRow(row_id="zero-width", title="Zoë\u200b Zero\u200bwidth", count="(1)", key="z"),
)


@pytest.mark.parametrize(
    ("row", "width", "current", "expected"),
    [
        (CHARACTERS, 24, True, "▸ Characters (28)  c"),
        (PERSONAS, 24, False, "  Personas (4)  p"),
        (LORE, 24, False, "  Lore books (3 · 2 on)"),
        (DICTIONARIES, 24, False, "  Chat dictionaries (3)"),
        (LORE, 31, False, "  Lore books (3 · 2 on)  l"),
        (DICTIONARIES, 31, False, "  Chat dictionaries (3 · 2 on)"),
        (CHARACTERS, 35, False, "  Characters (28)  c"),
        (PERSONAS, 35, False, "  Personas (4)  p"),
        (LORE, 35, False, "  Lore books (3 · 2 on)  l"),
        (DICTIONARIES, 35, False, "  Chat dictionaries (3 · 2 on)  d"),
    ],
)
def test_spec_rows_fit_the_bordered_rail_at_24_31_35(row, width, current, expected) -> None:
    label = fit_rail_row_label(row, width, current=current)
    assert label.plain == expected
    assert cell_len(label.plain) <= width


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (33, "  Chat dictionaries (3 · 2 on)  d"),  # 1. title + count + key
        (32, "  Chat dictionaries (3 · 2 on)"),  # 2. the key drops first
        (29, "  Chat dictionaries (3)"),  # 3. short count
        (22, "  Dicts (3 · 2 on)"),  # 4. short title
        (17, "  Dicts (3)"),  # 5. short title + short count
        (10, "  Dic… (3)"),  # 6. ellipsis, count whole
    ],
)
def test_fallback_order_drops_the_key_first_and_never_the_count(width, expected) -> None:
    assert fit_rail_row_label(FALLBACK, width).plain == expected


@pytest.mark.parametrize(
    ("row", "width", "current", "expected"),
    [
        (LORE, 20, False, "  Lore books (3)"),
        (DICTIONARIES, 20, False, "  Dictionaries (3)"),
        (CHARACTERS, 16, True, "▸ Characte… (28)"),
        (DICTIONARIES, 16, False, "  Dictionar… (3)"),
    ],
)
def test_narrow_rails_keep_the_noun_before_the_detail(row, width, current, expected) -> None:
    assert fit_rail_row_label(row, width, current=current).plain == expected


@pytest.mark.parametrize("width", range(1, 41))
@pytest.mark.parametrize("row", (*ROWS, FALLBACK), ids=lambda row: row.row_id)
def test_the_count_is_never_clipped(row, width) -> None:
    label = fit_rail_row_label(row, width)
    assert label.count in {row.count, row.short_count or row.count}
    tail = f"{label.count}  {label.key}" if label.key else label.count
    assert label.plain.endswith(tail)
    for current in (True, False):
        assert (
            rail_row_content(row, width, current=current).plain
            == fit_rail_row_label(row, width, current=current).plain
        )


def test_a_loading_count_never_paints_the_key_hint() -> None:
    """RC-11: a hint painted while loading would vanish on every arrival."""
    loading = DestinationRailRow(
        row_id="lore",
        title="Lore books",
        short_title="Lore",
        count="(…)",
        key="l",
        count_loading=True,
    )
    for width in (0, 16, 24, 31, 35, 60):
        assert fit_rail_row_label(loading, width).key == ""
    assert fit_rail_row_label(loading, 24).plain == "  Lore books (…)"


def test_an_unknown_width_renders_the_full_label() -> None:
    assert fit_rail_row_label(DICTIONARIES, 0).plain == "  Chat dictionaries (3 · 2 on)  d"


def test_the_current_marker_follows_ascii_glyph_mode() -> None:
    glyph_fallback.set_ascii_glyph_mode(True)
    try:
        assert fit_rail_row_label(CHARACTERS, 24, current=True).plain == "> Characters (28)  c"
    finally:
        glyph_fallback.set_ascii_glyph_mode(False)


@pytest.mark.parametrize("width", range(1, 41))
@pytest.mark.parametrize("row", WIDE, ids=lambda row: row.row_id)
def test_wide_and_zero_width_titles_fit_by_terminal_cells(row, width) -> None:
    label = fit_rail_row_label(row, width)
    if width >= 2 + cell_len(row.short_count or row.count):
        assert cell_len(label.plain) <= width, (label.plain, width)
    assert label.count in {row.count, row.short_count or row.count}


class _RailApp(ConsolidatedCSSApp):
    """Real CSS so the row's ``w-full``/``h-1`` utilities apply."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, rows, current_id: str | None = None) -> None:
        super().__init__()
        self.rows = rows
        self.current_id = current_id

    def compose(self) -> ComposeResult:
        with Vertical(id="rail"):
            for row in self.rows:
                yield DestinationRailRowButton(
                    row,
                    current=row.row_id == self.current_id,
                    id=f"rail-row-{row.row_id}",
                )


async def test_the_row_button_refits_its_label_when_its_width_changes() -> None:
    app = _RailApp(ROWS, current_id="characters")
    async with app.run_test(size=(60, 12)) as pilot:
        rail = app.query_one("#rail")
        painted = {}
        for width in (35, 31, 24):
            rail.styles.width = width
            await pilot.pause()
            for button in app.query(DestinationRailRowButton):
                assert button.content_region.width == width
                expected = fit_rail_row_label(
                    button.rail_row, width, current=button.is_current
                )
                assert button.label.plain == expected.plain
            painted[width] = app.query_one(
                "#rail-row-dictionaries", DestinationRailRowButton
            ).label.plain
        assert painted == {
            35: "  Chat dictionaries (3 · 2 on)  d",
            31: "  Chat dictionaries (3 · 2 on)",
            24: "  Chat dictionaries (3)",
        }


async def test_the_row_button_renders_untrusted_text_literally() -> None:
    """R33: a ``[/]`` name would raise MarkupError; ``[@click=…]`` would act.

    The ``sync_row`` and the second refit force real label assignments: at
    width 40 the first refit has the same plain text as the label built at
    construction, so it assigns nothing and could not catch a markup parse.
    """
    row = DestinationRailRow(row_id="hostile", title="[/] [@click=app.quit]Boom", count="(1)")
    app = _RailApp((row,))
    async with app.run_test(size=(60, 8)) as pilot:
        app.query_one("#rail").styles.width = 40
        await pilot.pause()
        button = app.query_one(DestinationRailRowButton)
        assert "[/] [@click=app.quit]Boom" in button.label.plain
        button.sync_row(
            DestinationRailRow(row_id="hostile", title="[b]x [@click=app.quit]Boom", count="(2)"),
            current=True,
        )
        await pilot.pause()
        assert button.label.plain == "▸ [b]x [@click=app.quit]Boom (2)"
        app.query_one("#rail").styles.width = 20
        await pilot.pause()
        assert "[b]x" in button.label.plain
        assert not any("@click" in str(span.style) for span in button.label.spans)


async def test_a_style_only_change_repaints_the_row() -> None:
    """``Content.__eq__`` and the ``label`` reactive compare plain text only.

    A count that starts loading with unchanged text changes only the count's
    style (dim); a plain-text comparison alone would keep the stale style.
    """
    known = DestinationRailRow(row_id="chats", title="Chats", count="(3)")
    app = _RailApp((known,))
    async with app.run_test(size=(60, 8)) as pilot:
        app.query_one("#rail").styles.width = 24
        await pilot.pause()
        button = app.query_one(DestinationRailRowButton)
        assert not any("dim" in str(span.style) for span in button.label.spans)
        button.sync_row(replace(known, count_loading=True), current=False)
        await pilot.pause()
        assert button.label.plain == "  Chats (3)"
        assert any("dim" in str(span.style) for span in button.label.spans)


async def test_sync_row_patches_in_place_and_skips_an_unchanged_row() -> None:
    app = _RailApp((LORE,))
    async with app.run_test(size=(60, 8)) as pilot:
        app.query_one("#rail").styles.width = 24
        await pilot.pause()
        button = app.query_one(DestinationRailRowButton)
        refreshes: list[tuple] = []
        original = button.refresh

        def counting_refresh(*args, **kwargs):
            refreshes.append(args)
            return original(*args, **kwargs)

        button.refresh = counting_refresh
        button.sync_row(LORE, current=False)
        assert refreshes == []
        button.refresh = original
        loading = DestinationRailRow(
            row_id="lore",
            title="Lore books",
            short_title="Lore",
            count="(…)",
            key="l",
            count_loading=True,
        )
        button.sync_row(loading, current=False)
        await pilot.pause()
        assert app.query_one(DestinationRailRowButton) is button
        assert button.label.plain == "  Lore books (…)"
```

In `Tests/UI/test_adaptive_pane_shell.py`:

Old:
```python
SHARED_WIDGET_NAMES = ["AdaptivePaneShell", "AdaptivePaneGrip"]
```
New:
```python
SHARED_WIDGET_NAMES = ["AdaptivePaneShell", "AdaptivePaneGrip", "DestinationRailRowButton"]
```

- [ ] **Step 2: Run to see it fail**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_destination_rail_row.py -q -p no:cacheprovider 2>&1 | tail -3`

Expected: `ERROR collecting Tests/UI/test_destination_rail_row.py` — `ImportError: cannot import name 'DestinationRailRow' from 'tldw_chatbook.Widgets.adaptive_pane_shell'`.

- [ ] **Step 3: Implement**

Edit the imports of `tldw_chatbook/Widgets/adaptive_pane_shell.py`:

Old:
```python
from textual import on
```
New:
```python
from rich.cells import cell_len
from textual import on
```

Old:
```python
from textual.events import DescendantFocus
```
New:
```python
from textual.events import DescendantFocus, Resize
```

Old:
```python
    PaneName,
)
```
New:
```python
    PaneName,
)
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph
```

Append to the end of the module:

```python


# ---------------------------------------------------------------------------
# Rail rows (spec section 1.4.3). Modelled on the Library rail's own row
# label builder, but deliberately neither shared with it nor golden-tested
# against it: the fit steps differ by design.
# ---------------------------------------------------------------------------

#: The current-row marker (spec section 1.4.3), resolved through the glyph map
#: so ASCII mode paints ``>``. Two cells with its trailing space -- the same as
#: a non-current row's two-space prefix -- so rows never shift when the current
#: row moves.
RAIL_ROW_CURRENT_MARKER = "▸"

#: The ellipsis a squeezed title ends with (step 6), measured in cells. It goes
#: through ``resolve_glyph``, but the glyph map has no ASCII substitute for it,
#: so ASCII mode paints it unchanged.
RAIL_ROW_ELLIPSIS = "…"

#: Gap between the count and the key hint (the spec's examples print two cells).
_RAIL_ROW_KEY_GAP = "  "


@dataclass(frozen=True)
class DestinationRailRow:
    """One row of a destination's navigation rail.

    A sibling of the Library's own row record, not a subclass of it: Roleplay
    never imports ``Widgets/Library/``. Counts arrive pre-formatted because
    their states (loading, known, ``3 of 28``, ``100+``, failed) belong to the
    destination (spec section 1.4.2); this record only says how to shorten
    them.

    Attributes:
        row_id: Stable id; callers build the button's DOM id from it.
        title: The canonical noun, for example ``"Lore books"``.
        short_title: The fallback noun (``"Lore"``); ``""`` means ``title``.
        count: The count text including its parentheses (``"(3 · 2 on)"``),
            or ``""`` for a row with no count.
        short_count: The shorter count (``"(3)"``); ``""`` means ``count``.
        key: The kind key hint (``"l"``), or ``""``.
        count_loading: ``True`` while the count is in flight. The key hint is
            never painted then (RC-11): it would vanish when the count lands.
        disabled: Whether the row's button is disabled.
    """

    row_id: str
    title: str
    short_title: str = ""
    count: str = ""
    short_count: str = ""
    key: str = ""
    count_loading: bool = False
    disabled: bool = False


@dataclass(frozen=True)
class FittedRailRowLabel:
    """A rail row label after fitting, kept in parts so a button can style them.

    Attributes:
        prefix: ``"▸ "`` (resolved) on the current row, ``"  "`` otherwise.
        title: The title as painted -- full, short or ellipsized; ``""`` only
            when not even one character fits beside the count.
        count: The count as painted (full or short). Never truncated.
        key: The key hint, or ``""`` once it has been dropped.
    """

    prefix: str
    title: str
    count: str
    key: str

    @property
    def plain(self) -> str:
        """The label as one string, exactly as it paints."""
        body = " ".join(part for part in (self.title, self.count) if part)
        tail = f"{_RAIL_ROW_KEY_GAP}{self.key}" if self.key else ""
        return f"{self.prefix}{body}{tail}"


def _ellipsize(text: str, budget: int) -> str:
    """Cut ``text`` to ``budget`` terminal cells, ending in the ellipsis.

    Measures cells, not characters, so wide (CJK, emoji) and zero-width
    characters fit by what they paint.

    Args:
        text: The title to shorten.
        budget: Cells available for the title.

    Returns:
        ``text`` when it already fits; ``""`` when not even one character
        fits before the ellipsis; otherwise the longest fitting head plus the
        ellipsis.
    """
    if cell_len(text) <= budget:
        return text
    ellipsis = resolve_glyph(RAIL_ROW_ELLIPSIS)
    room = budget - cell_len(ellipsis)
    head = ""
    for character in text:
        if cell_len(head + character) > room:
            break
        head += character
    head = head.rstrip()
    return f"{head}{ellipsis}" if head else ""


def fit_rail_row_label(
    row: DestinationRailRow, width: int, *, current: bool = False
) -> FittedRailRowLabel:
    """Fit one rail row's label into ``width`` terminal cells (spec section 1.4.3).

    Tries, in order, and returns the first that fits:

    1. title + count + key hint (the key is skipped while the count loads);
    2. title + count -- the key drops first: keys are always in the footer
       and F1;
    3. title + short count;
    4. short title + count;
    5. short title + short count;
    6. the short title ellipsized, with the short count whole.

    The canonical noun outlives the key hint and the count is never clipped:
    when even step 6 cannot fit, the title shrinks to nothing and the label
    may exceed ``width`` by the count alone.

    Args:
        row: The row to fit.
        width: Available text cells. ``0`` or less (compose time, before
            layout) returns the full label.
        current: Whether the row is the destination's current kind.

    Returns:
        The fitted label, in parts.
    """
    prefix = f"{resolve_glyph(RAIL_ROW_CURRENT_MARKER)} " if current else "  "
    short_title = row.short_title or row.title
    short_count = row.short_count or row.count
    key = "" if row.count_loading else row.key
    candidates = (
        FittedRailRowLabel(prefix, row.title, row.count, key),
        FittedRailRowLabel(prefix, row.title, row.count, ""),
        FittedRailRowLabel(prefix, row.title, short_count, ""),
        FittedRailRowLabel(prefix, short_title, row.count, ""),
        FittedRailRowLabel(prefix, short_title, short_count, ""),
    )
    if width <= 0:
        return candidates[0]
    for candidate in candidates:
        if cell_len(candidate.plain) <= width:
            return candidate
    count_cells = cell_len(f" {short_count}") if short_count else 0
    title = _ellipsize(short_title, width - cell_len(prefix) - count_cells)
    return FittedRailRowLabel(prefix, title, short_count, "")


def rail_row_content(
    row: DestinationRailRow, width: int, *, current: bool = False
) -> Content:
    """The fitted label as literal ``Content``.

    The key hint and a loading count are dim and the current row is bold.
    Built from ``Content`` parts and never parsed as markup, so a title such
    as ``"[/]"`` paints as typed (spec R33).

    Args:
        row: The row to render.
        width: Available text cells (see ``fit_rail_row_label``).
        current: Whether the row is the destination's current kind.

    Returns:
        Content whose ``plain`` equals ``fit_rail_row_label(...).plain``.
    """
    label = fit_rail_row_label(row, width, current=current)
    parts: list[str | tuple[str, str]] = [label.prefix, label.title]
    if label.count:
        if label.title:
            parts.append(" ")
        parts.append((label.count, "dim") if row.count_loading else label.count)
    if label.key:
        parts.append((f"{_RAIL_ROW_KEY_GAP}{label.key}", "dim"))
    content = Content.assemble(*parts)
    return content.stylize("bold") if current else content


class DestinationRailRowButton(Button):
    """A rail row button that refits its own label whenever its width changes.

    Per-button ``on_resize``, not one rail-level pass, so the fit also follows
    a vertical scrollbar's gutter: the rail's own size does not change when
    its scrollbar appears, but each row's content width does. Rows are
    patched through ``sync_row``, never recomposed. Width and height come from
    the canonical ``w-full`` and ``h-1`` utilities; every other row rule
    belongs to the destination's lazy sheet.
    """

    def __init__(
        self, row: DestinationRailRow, *, current: bool = False, **kwargs: Any
    ) -> None:
        """Build a row button.

        Args:
            row: The row this button renders.
            current: Whether the row is the destination's current kind.
            **kwargs: Forwarded to ``Button`` (``id``, ``classes``, ...).
        """
        self.rail_row = row
        self.is_current = current
        super().__init__(
            rail_row_content(row, 0, current=current),
            compact=True,
            disabled=row.disabled,
            **kwargs,
        )
        self.add_class("w-full")
        self.add_class("h-1")
        self.styles.line_pad = 0

    def sync_row(self, row: DestinationRailRow, *, current: bool) -> None:
        """Patch the row in place; does nothing when nothing changed.

        Args:
            row: The row's new state.
            current: Whether the row is now the destination's current kind.
        """
        if row == self.rail_row and current == self.is_current:
            return
        self.rail_row = row
        self.is_current = current
        if self.disabled != row.disabled:
            self.disabled = row.disabled
        self._refit()

    def on_resize(self, event: Resize) -> None:
        """Refit the label to the new content width."""
        self._refit()

    def _refit(self) -> None:
        """Assign a refitted label only when it differs from the current one.

        ``Content.__eq__`` compares plain text only, and so does the ``label``
        reactive, so a style-only change (a count that starts or stops
        loading with unchanged text) is set past that equality check and
        repainted explicitly.
        """
        label = rail_row_content(
            self.rail_row, self.content_region.width, current=self.is_current
        )
        painted = self.label
        if label.plain != painted.plain:
            self.label = label
        elif label.spans != painted.spans:
            self.set_reactive(Button.label, label)
            self.refresh(layout=True)
```

- [ ] **Step 4: Run to see it pass**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_destination_rail_row.py Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider 2>&1 | tail -1`

Expected: `360 passed` — 347 rail-row cases (10 spec + 6 order + 4 narrow + 200 never-clipped + 3 single + 120 wide + 4 mounted) plus 13 shell tests; it must equal `--collect-only -q … | tail -1` for the two files. (The expected strings were verified against a scratch prototype of this code on `fccf70d3b0`; the markup and style-only tests as revised in plan review.)

- [ ] **Step 5: Named mutations**

1. `key-not-dropped-first`: in `fit_rail_row_label`, swap the second and third candidates (title+count and title+short-count). Run `…/python -m pytest Tests/UI/test_destination_rail_row.py -q -p no:cacheprovider -k "fallback_order or spec_rows"`. Expected: failures at widths 32 and 24/31 (`"  Chat dictionaries (3)"` appears where `(3 · 2 on)` was expected). Restore; all pass.
2. `measure-chars-not-cells`: in `_ellipsize`, replace `if cell_len(head + character) > room:` with `if len(head + character) > room:`. Run `-k wide_and_zero_width`. Expected: failures on the `cjk` and `emoji` rows. Restore; all pass.
3. `markup-label`: edit ONLY the occurrence in `DestinationRailRowButton._refit` (the module has another `self.label = label`, in `AdaptivePaneGrip.sync_open`): replace `        if label.plain != painted.plain:\n            self.label = label` with `        if label.plain != painted.plain:\n            self.label = label.plain`. Run `-k untrusted_text`. Expected: `1 failed` (`- ▸ [b]x [@click=app.quit]Boom (2)` / `+ ▸ x Boom (2)`: the string was parsed as markup). Restore; `1 passed`.
4. `no-refit-on-resize`: in `DestinationRailRowButton.on_resize`, replace the body line `        self._refit()` that follows `"""Refit the label to the new content width."""` with `        pass`. Run `-k refits_its_label`. Expected: `1 failed` (at 31 the dictionaries row keeps its width-0 full label `…(3 · 2 on)  d`). Restore; `1 passed`.
5. `drop-w-full`: in `DestinationRailRowButton.__init__`, comment out `        self.add_class("w-full")`. Run `-k refits_its_label`. Expected: `1 failed` (`content_region.width` is the label's own width, not 35). Restore; `1 passed`.
6. `plain-only-compare`: in `_refit`, change `        elif label.spans != painted.spans:` to `        elif False:`. Run `-k style_only`. Expected: `1 failed` (no `dim` span after the count starts loading). Restore; `1 passed`.

Record all six in `$EV/mutations.md`.

- [ ] **Step 6: Lint, re-check the pre-import raise, and commit**

The shared module grew, and from Task 3 on it is on the Library route: re-measure the head and re-run the (idempotent) raise before committing. On today's dev the module count is unchanged (558) and nothing moves; if #2862 landed first, the LOC limit is raised to the new measurement under the same sign-off, or the script stops at the approved bound.

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b0-evidence
cd $MAIN/.worktrees/roleplay-b0
"$PY" -m ruff check tldw_chatbook/Widgets/adaptive_pane_shell.py Tests/UI/test_destination_rail_row.py Tests/UI/test_adaptive_pane_shell.py
"$EV/preimport_measure.sh" head && "$PY" "$EV/preimport_raise.py"
"$PY" -m pytest Tests/Performance/test_screen_preimport_payload_budget.py -p no:cacheprovider -q 2>&1 | tail -1
git add tldw_chatbook/Widgets/adaptive_pane_shell.py Tests/UI/test_destination_rail_row.py Tests/UI/test_adaptive_pane_shell.py Tests/Performance/test_screen_preimport_payload_budget.py backlog/decisions/097-boot-budget-ratchets.md
git commit -m "feat(widgets): destination rail rows with fit_rail_row_label (B0)" -m "Spec 1.4.3 fallback order measured in terminal cells; the count is never clipped and a loading count never paints its key (shared pane-shell ADR)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: `All checks passed!`; the `head |` summary (558 modules on today's dev); `raised: MAX_PASS_ADDED_MODULES 557 -> 558 | ledger row: written` (the row's LOC figures follow the new measurement); `1 passed`; the commit line. Any `STOP:` line: do not commit; report.

---

### Task 5: Promote `SelectAllOnFocusingClickInput` with its "/" re-arm

**Files:**
- Modify: `tldw_chatbook/Widgets/adaptive_pane_shell.py` (imports; append)
- Modify: `tldw_chatbook/Widgets/Library/library_rail.py` (imports; the two input classes)
- Test: `Tests/UI/test_adaptive_pane_shell.py` (import block; append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `class SelectAllOnFocusingClickInput(Input)` in the shared module with `__init__(*args, swallow_slash_on_focus: bool = False, **kwargs)`. `tldw_chatbook.Widgets.Library.library_rail.SelectAllOnFocusingClickInput` is the same object (re-export, so `library_search_rag_panel.py` and `Tests/Widgets/Library/test_library_rail.py`'s identity test keep working). `LibraryRailSearchInput(SelectAllOnFocusingClickInput)` keeps `swallow_slash_on_focus=True` as its default and defines no `_on_key`.

The class body moves verbatim from `library_rail.py`; the `/` interception moves verbatim from `LibraryRailSearchInput._on_key` into the shared class behind the keyword. The handler sequence Textual dispatches for `LibraryRailSearchInput` is unchanged (one `_on_key` override, then `Input._on_key`). For the Search/RAG panel's box the only difference is that `Input._on_key` restarts the cursor blink a second time on non-printable keys — the same path the Notes filter already runs with `swallow_slash_on_focus=False`.

- [ ] **Step 1: Write the failing tests**

Edit the import block of `Tests/UI/test_adaptive_pane_shell.py`:

Old:
```python
from pathlib import Path

from textual import on
```
New:
```python
from pathlib import Path

import pytest
from textual import on
```

Old:
```python
from textual.widgets import Button, Static
```
New:
```python
from textual.widgets import Button, Input, Static
```

Old:
```python
from tldw_chatbook.Widgets.Library import library_adaptive_reader_shell as library_shell
```
New:
```python
from tldw_chatbook.Widgets.Library import library_adaptive_reader_shell as library_shell
from tldw_chatbook.Widgets.Library import library_rail
```

Append:

```bash
cat >> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0/Tests/UI/test_adaptive_pane_shell.py <<'EOF'


# ---------------------------------------------------------------------------
# The promoted compact search input (B0).
# ---------------------------------------------------------------------------


def test_library_rail_re_exports_the_shared_search_input() -> None:
    assert library_rail.SelectAllOnFocusingClickInput is shared.SelectAllOnFocusingClickInput
    assert issubclass(
        library_rail.LibraryRailSearchInput, shared.SelectAllOnFocusingClickInput
    )
    # The re-arm lives once, in the shared class.
    assert "_on_key" not in vars(library_rail.LibraryRailSearchInput)


class _InputApp(App):
    def __init__(self, factory) -> None:
        super().__init__()
        self.factory = factory

    def compose(self) -> ComposeResult:
        yield self.factory()


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: shared.SelectAllOnFocusingClickInput(value="Work", id="box"), "Work/"),
        (
            lambda: shared.SelectAllOnFocusingClickInput(
                value="Work", id="box", swallow_slash_on_focus=True
            ),
            "Work",
        ),
        (lambda: library_rail.LibraryRailSearchInput(value="Work", id="box"), "Work"),
        (
            lambda: library_rail.LibraryRailSearchInput(
                value="Work", id="box", swallow_slash_on_focus=False
            ),
            "Work/",
        ),
    ],
    ids=[
        "shared-default-types-slash",
        "shared-opt-in-swallows",
        "library-rail-default-swallows",
        "library-notes-filter-types",
    ],
)
async def test_slash_in_a_focused_box_follows_swallow_slash_on_focus(factory, expected) -> None:
    app = _InputApp(factory)
    async with app.run_test(size=(40, 5)) as pilot:
        box = app.query_one("#box", Input)
        box.focus()
        await pilot.pause()
        box.action_end()
        await pilot.press("/")
        await pilot.pause()
        assert box.value == expected
EOF
```

- [ ] **Step 2: Run to see it fail**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider -k "search_input or slash" 2>&1 | tail -4`

Expected: `3 failed, 2 passed`. FAILED: `test_library_rail_re_exports_the_shared_search_input` and the `shared-default-types-slash` and `shared-opt-in-swallows` cases (`AttributeError: module 'tldw_chatbook.Widgets.adaptive_pane_shell' has no attribute 'SelectAllOnFocusingClickInput'`). The two `library-rail-*` cases already pass on dev (`LibraryRailSearchInput` exists with that behaviour): they pin behaviour the move must keep.

- [ ] **Step 3: Add the class to the shared module**

Imports in `tldw_chatbook/Widgets/adaptive_pane_shell.py`:

Old:
```python
from textual.events import DescendantFocus, Resize
```
New:
```python
from textual.events import DescendantFocus, Focus, Key, MouseDown, Resize
```

Old:
```python
from textual.widgets import Button
```
New:
```python
from textual.widgets import Button, Input
```

Append to the end of the module:

```python


# ---------------------------------------------------------------------------
# Compact search input, promoted from the Library rail (B0).
# ---------------------------------------------------------------------------


class SelectAllOnFocusingClickInput(Input):
    """An ``Input`` whose FIRST click (the one that also focuses it) selects
    all text instead of just positioning the cursor there (LIB-17).

    Textual's ``Input`` already defaults ``select_on_focus=True`` (its own
    ``_on_focus`` sets ``Selection(0, len(value))``), but ``Input.
    _on_mouse_down`` ALWAYS repositions the cursor to the click offset --
    and ``Screen._forward_event`` calls ``set_focus()`` *synchronously*,
    before the click is even forwarded to this widget, so ``self.has_focus``
    already reads ``True`` by the time ``_on_mouse_down`` runs regardless of
    whether THIS click is the one that focused the box. Checking
    ``has_focus`` there cannot distinguish the two cases; the ``Focus``
    event (posted by ``set_focus``) and this ``MouseDown`` (posted right
    after, by the same click) are instead queued back-to-back on THIS
    widget's own message pump and processed in that order, so ``_on_focus``
    marks a one-shot "a focusing click may still be inbound" flag that
    ``_on_mouse_down`` consumes if it is the very next thing processed --
    the same-gesture window this whole mechanism depends on.

    A second Textual quirk this override must also account for: message
    dispatch (``MessagePump._get_dispatch_methods``) walks the class's
    entire MRO and invokes EVERY class's own ``_on_mouse_down`` in turn
    (most-derived first) -- calling ``super()`` is not what wires this up,
    and NOT calling it does not skip it either. Without
    ``event.prevent_default()`` in the select-all branch below, ``Input.
    _on_mouse_down`` (the base class, further up the MRO) still runs
    immediately afterward and silently overwrites the select-all with its
    own ``Selection.cursor(click_offset)`` -- ``prevent_default()`` is the
    documented mechanism (checked at the top of that MRO walk) that stops
    it, the exact same seam ``_on_key`` below leans on for the "/" re-arm.

    Without this fix, a plain mouse click on a not-yet-focused Input
    silently wins the race and undoes ``select_on_focus``'s "replace me"
    framing entirely, regardless of where in the box the click lands. For a
    prefilled query box this means the box's stale text survives the very
    interaction (click, then type) a user relies on to replace it:
    live-reproduced by a click landing near the start of "quokka" and
    typing a character, which PREPENDED instead of replacing ("Zquokka").

    Scoped to the focusing click only: once the box already has focus (no
    new ``Focus`` event, so the flag is never armed), a click positions the
    cursor precisely as normal -- expected mid-text editing is unaffected.
    The flag also self-clears shortly after arming (``call_after_refresh``)
    so a LATER, unrelated click on an already-focused box (e.g. after a
    Tab-focus with no immediately-following click) never inherits a stale
    "select all" from an earlier, unconsumed focus event.

    Promoted from the Library rail for Roleplay frame B0, together
    with the rail search box's "/" re-arm, which is an opt-in keyword here:
    ``swallow_slash_on_focus=True`` makes a FOCUSED box intercept "/" and
    select all (so the next keystroke replaces a stale query) instead of
    typing it. It defaults to ``False`` because a shared box may hold text
    that contains "/" (Roleplay item names; the Library Notes filter's
    folder paths). The Library rail search box keeps ``True`` through
    ``LibraryRailSearchInput``. "/" only ever acts as a focus accelerator
    while the box is NOT focused; the screen-level handler gates that.
    """

    def __init__(
        self, *args: Any, swallow_slash_on_focus: bool = False, **kwargs: Any
    ) -> None:
        """Build the box.

        Args:
            *args: Positional arguments forwarded to ``Input.__init__``.
            swallow_slash_on_focus: When ``True``, a focused box intercepts
                "/" and selects all text instead of typing it.
            **kwargs: Keyword arguments forwarded to ``Input.__init__``.
        """
        super().__init__(*args, **kwargs)
        self._select_all_pending_click = False
        self._swallow_slash_on_focus = swallow_slash_on_focus

    def _on_focus(self, event: Focus) -> None:
        self._select_all_pending_click = True
        self.call_after_refresh(self._clear_select_all_pending_click)

    def _clear_select_all_pending_click(self) -> None:
        self._select_all_pending_click = False

    async def _on_mouse_down(self, event: MouseDown) -> None:
        if self._select_all_pending_click:
            self._select_all_pending_click = False
            self._pause_blink(visible=True)
            self.select_all()
            self._selecting = True
            self.capture_mouse()
            # See the class docstring: stops Textual's own MRO walk from
            # ALSO invoking Input._on_mouse_down for this event, which
            # would otherwise overwrite the select-all above.
            event.prevent_default()
            return
        await super()._on_mouse_down(event)

    async def _on_key(self, event: Key) -> None:
        # Same slash representations the screen-level handler accepts --
        # some platforms/layouts emit key="slash" without character="/".
        if self._swallow_slash_on_focus and (
            event.key in {"/", "slash"} or event.character == "/"
        ):
            self.select_all()
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)
```

- [ ] **Step 4: Make `library_rail.py` re-export it and keep the Library default**

Imports in `tldw_chatbook/Widgets/Library/library_rail.py` (only the two input classes used `Focus`, `Key`, `MouseDown` and `Input`; nothing else in the module does):

Old:
```python
from textual.events import DescendantFocus, Focus, Key, MouseDown, Resize
```
New:
```python
from textual.events import DescendantFocus, Resize
```

Old:
```python
from textual.widgets import Button, Input, Static
```
New:
```python
from textual.widgets import Button, Static
```

Old:
```python
from tldw_chatbook.Widgets.destination_rail import (
    RAIL_SECTION_TOGGLE_PREFIX,
```
New:
```python
# Re-exported: library_search_rag_panel.py imports it from here, and
# Tests/Widgets/Library/test_library_rail.py pins that both names are one object.
from tldw_chatbook.Widgets.adaptive_pane_shell import SelectAllOnFocusingClickInput
from tldw_chatbook.Widgets.destination_rail import (
    RAIL_SECTION_TOGGLE_PREFIX,
```

Replace the two class definitions (everything from the line `class SelectAllOnFocusingClickInput(Input):` up to, not including, the line `class LibraryRailRowButton(Button):`) with the thin subclass. Run this exact script:

```bash
bash <<'OUTER'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'EOF'
from pathlib import Path

path = Path("tldw_chatbook/Widgets/Library/library_rail.py")
text = path.read_text(encoding="utf-8")
start = text.index("class SelectAllOnFocusingClickInput(Input):")
end = text.index("class LibraryRailRowButton(Button):")
replacement = '''class LibraryRailSearchInput(SelectAllOnFocusingClickInput):
    """Rail search box where a second "/" re-arms the query instead of typing.

    "/" is the Library screen's focus-the-search key (F-012); once the box
    itself has focus the screen's on_key never sees printable keys, so a
    second "/" would insert a literal slash into the query -- the settings
    screen's task-1584 live trap, solved the same way here: intercept it
    and select-all so the next keystroke replaces the stale text. LIB-17:
    a stale query surviving a screen re-entry (the rail rebuilds a fresh
    box seeded from the screen's persisted query, unfocused) is covered by
    the SAME "select-all, don't clear" promise this seam already makes for
    "/" -- ``SelectAllOnFocusingClickInput`` extends it to the box's first
    click too, so whichever way the user re-enters the box (click or "/"),
    typing replaces rather than appends.

    Opt-in only: pass ``swallow_slash_on_focus=False`` for a box whose
    CONTENT legitimately contains "/" (fix round 1 Important 4 -- the
    Notes filter matches folder-style paths like "Work/Q3", so swallowing
    every "/" it receives while focused made that unable to be typed at
    all; the rail search box has no such content and keeps the default).
    Ruling: "/" only ever acts as the focus-accelerator while the box is
    NOT focused (the screen-level handler already gates that); once
    focused, a box with the swallow disabled treats "/" like any other
    character.

    Roleplay frame B0: the "/" re-arm itself moved into the shared
    ``SelectAllOnFocusingClickInput`` (``Widgets/adaptive_pane_shell.py``),
    where it is off by default; this subclass only keeps the rail search
    box's default ON, so every Library construction behaves as before.
    """

    def __init__(
        self, *args: Any, swallow_slash_on_focus: bool = True, **kwargs: Any
    ) -> None:
        """Build the search box, optionally opting out of "/" swallowing.

        Args:
            *args: Positional arguments forwarded to ``Input.__init__``.
            swallow_slash_on_focus: When ``True`` (the default rail-search
                behavior), a focused box intercepts "/" and selects all
                text instead of typing it, per the class docstring. Pass
                ``False`` for a box whose content legitimately contains
                "/" (e.g. the Notes filter), where "/" must type normally
                once the box already has focus.
            **kwargs: Keyword arguments forwarded to ``Input.__init__``.
        """
        super().__init__(
            *args, swallow_slash_on_focus=swallow_slash_on_focus, **kwargs
        )


'''
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
print("replaced", end - start, "chars")
EOF
grep -n "class SelectAllOnFocusingClickInput\|class LibraryRailSearchInput\|class LibraryRailRowButton\|_on_key\|_on_mouse_down" tldw_chatbook/Widgets/Library/library_rail.py
OUTER
```

Expected: `replaced <n> chars`, then exactly two grep lines: `class LibraryRailSearchInput(SelectAllOnFocusingClickInput):` and `class LibraryRailRowButton(Button):` (no `_on_key`, no `_on_mouse_down`, no `class SelectAllOnFocusingClickInput`).

In `Tests/UI/test_adaptive_pane_shell.py`:

Old:
```python
SHARED_WIDGET_NAMES = ["AdaptivePaneShell", "AdaptivePaneGrip", "DestinationRailRowButton"]
```
New:
```python
SHARED_WIDGET_NAMES = [
    "AdaptivePaneShell",
    "AdaptivePaneGrip",
    "DestinationRailRowButton",
    "SelectAllOnFocusingClickInput",
]
```

- [ ] **Step 5: Run the new tests and the Library input suites against the base arm**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider 2>&1 | tail -1`

Expected: `18 passed`.

```bash
bash <<'EOF'
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence
printf '%s\n' Tests/Widgets/Library/test_library_rail.py Tests/UI/test_library_notes_w4_editor.py Tests/UI/test_library_notes_w5_kbd_focus.py Tests/UI/test_library_notes_folder_navigator.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_library_crit9_grammar.py Tests/UI/test_library_choice_strips.py Tests/UI/test_library_rag_query_gate_race.py Tests/UI/test_library_rag_rechunk_action.py Tests/UI/test_library_rag_legacy_chunk_report.py Tests/Library/test_library_rag_state.py > "$EV/suites-input-quick.txt"
"$EV/paired.sh" inputquick "$EV/suites-input-quick.txt" -n 4
EOF
```

Expected: `recovery=` in single digits on both arms, and no line between `new failures on head [inputquick] (must be empty):` and `(end of new failures [inputquick])`. The last six files exercise the Search/RAG panel, whose box (`SelectAllOnFocusingClickInput`) now inherits the shared `_on_key`.

- [ ] **Step 6: Named mutation `shared-default-swallows`**

Change the shared default to `swallow_slash_on_focus: bool = True`. Run `…/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider -k slash`. Expected: `1 failed` (`shared-default-types-slash`: `'Work' == 'Work/'`). Restore `False`; `4 passed`. Record it.

- [ ] **Step 7: Lint, re-check the pre-import raise, and commit**

`library_rail.py` now imports the shared module too (no new module) and the shared module grew while `library_rail.py` shrank: re-measure and re-run the idempotent raise, exactly as in Task 4 Step 6.

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b0-evidence
cd $MAIN/.worktrees/roleplay-b0
"$PY" -m ruff check tldw_chatbook/Widgets/adaptive_pane_shell.py tldw_chatbook/Widgets/Library/library_rail.py Tests/UI/test_adaptive_pane_shell.py
"$EV/preimport_measure.sh" head && "$PY" "$EV/preimport_raise.py"
"$PY" -m pytest Tests/Performance/test_screen_preimport_payload_budget.py -p no:cacheprovider -q 2>&1 | tail -1
git add tldw_chatbook/Widgets/adaptive_pane_shell.py tldw_chatbook/Widgets/Library/library_rail.py Tests/UI/test_adaptive_pane_shell.py Tests/Performance/test_screen_preimport_payload_budget.py backlog/decisions/097-boot-budget-ratchets.md
git commit -m "refactor(widgets): promote the select-all search input with an opt-in slash re-arm (B0)" -m "Shared default swallow_slash_on_focus=False; the Library rail search box keeps True; library_rail re-exports the same class (shared pane-shell ADR)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: `All checks passed!`; the `head |` summary (still 558 modules); a `raised:` line naming the same constant(s) as Task 4; `1 passed`; the commit line. Any `STOP:` line: do not commit; report.

---

### Task 6: CSS — the Library block keyed to the Library destination classes (zero boot bytes, no `.tcss` change)

**Files:**
- Test: `Tests/UI/test_library_adaptive_reader_shell.py` (the three CSS-ownership pins re-keyed; names unchanged); `Tests/UI/test_adaptive_pane_shell.py` (import block; append)
- No `.tcss` file changes: the explanation of the destination classes lives in the `LIBRARY_ADAPTIVE_READER_CLASSES` comment (Task 3) and ADR §3. A stylesheet comment would do nothing for behaviour, travel into the generated `screen_agentic_library.tcss` (which #2862 also regenerates, so a rebase would hit a generated-file conflict), and carry the provisional ADR number into source and generated output.

**Interfaces:**
- Consumes: `LIBRARY_ADAPTIVE_READER_CLASSES` (Task 3); `build_css.SCREEN_OWNED_SPLITS` (each `ScreenOwnedSplit` has `.sheets: dict[owner, sheet]` and `.prefixes: dict[owner, tuple[str, ...]]`); `BUNDLED_STYLESHEET` from `Tests/UI/consolidated_css.py`.
- Produces: nothing new for later tasks.

Why no selector changes: the Library's destination classes ARE its existing class names, so every rule keeps matching, every token keeps the `library` split prefix (the rules stay in the lazy `screen_agentic_library.tcss`), and boot bytes cannot move. The one boot-resident part of the block is the 214 B `:hover, .-active` grouped grip rule: it is in the bundle on dev because the `.-active` token has no split owner. B0 leaves it exactly where it is (net 0); the ADR flags it as a B1 risk for Roleplay's copy.

These tests are pins on behaviour that already holds, so Step 2 expects PASS; the named mutations in Step 4 are what show they discriminate.

- [ ] **Step 1: Re-key the three ownership pins and write the new pins**

In `Tests/UI/test_library_adaptive_reader_shell.py`:

Old:
```python
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    AdaptiveReaderShellResized,
    LibraryAdaptiveReaderShell,
    PaneToggleRequested,
)
```
New:
```python
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LIBRARY_ADAPTIVE_READER_CLASSES,
    AdaptiveReaderShellResized,
    LibraryAdaptiveReaderShell,
    PaneToggleRequested,
)
```

Old:
```python
def _library_sources_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in LIBRARY_SOURCES)
```
New:
```python
def _library_sources_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in LIBRARY_SOURCES)


#: Roleplay frame B0: the CSS-ownership pins key on the Library
#: DESTINATION classes the shared shell is built with, not on literals, so a
#: rename on either side fails here instead of leaving the grips unstyled.
_SHELL = f".{LIBRARY_ADAPTIVE_READER_CLASSES.shell}"
_SHELL_RULE = f"{_SHELL} {{"
_WORK_RULE = f"{_SHELL} > .{LIBRARY_ADAPTIVE_READER_CLASSES.work} {{"
_GRIP_RULE = f"{_SHELL} > .{LIBRARY_ADAPTIVE_READER_CLASSES.grip} {{"
_GRIP_FOCUS_RULE = f"{_SHELL} > .{LIBRARY_ADAPTIVE_READER_CLASSES.grip}:focus {{"
```

Old:
```python
    assert ".library-adaptive-reader-shell {" in source
    assert ".library-adaptive-reader-shell > .library-adaptive-reader-work {" in source
    assert (
        ".library-adaptive-reader-shell > .library-adaptive-reader-pane-grip {"
        in source
    )
```
New:
```python
    assert _SHELL_RULE in source
    assert _WORK_RULE in source
    assert _GRIP_RULE in source
```

Old:
```python
    grip_block = sheet.text.split(
        ".library-adaptive-reader-shell > .library-adaptive-reader-pane-grip {",
        1,
    )[1].split("}", 1)[0]
```
New:
```python
    grip_block = sheet.text.split(_GRIP_RULE, 1)[1].split("}", 1)[0]
```

Old:
```python
    shared_grip = source.split(
        ".library-adaptive-reader-shell > .library-adaptive-reader-pane-grip {",
        1,
    )[1].split("}", 1)[0]
```
New:
```python
    shared_grip = source.split(_GRIP_RULE, 1)[1].split("}", 1)[0]
```

Old:
```python
    assert (
        ".library-adaptive-reader-shell > .library-adaptive-reader-pane-grip:focus {"
        in source
    )
```
New:
```python
    assert _GRIP_FOCUS_RULE in source
```

In `Tests/UI/test_adaptive_pane_shell.py`:

Old:
```python
import ast
import os
```
New:
```python
import ast
import os
from dataclasses import astuple
```

Old:
```python
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
```
New:
```python
from Tests.UI.consolidated_css import (
    APP_STYLESHEETS,
    BUNDLED_STYLESHEET,
    ConsolidatedCSSApp,
)
```

Old:
```python
from tldw_chatbook.Utils import adaptive_reader_state as ars
```
New:
```python
from tldw_chatbook.css import build_css
from tldw_chatbook.Utils import adaptive_reader_state as ars
```

Append:

```bash
cat >> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0/Tests/UI/test_adaptive_pane_shell.py <<'EOF'


# ---------------------------------------------------------------------------
# CSS (B0): the Library block keyed to the Library destination classes.
# ---------------------------------------------------------------------------


def _library_split_prefixes() -> tuple[str, ...]:
    """The prefixes of the ONE owner whose sheet is the Library's lazy sheet.

    Not every prefix of the split that writes it: a class carrying another
    owner's prefix would send its rule to that owner's sheet (or pin it to
    boot), and must fail here.
    """
    for split in build_css.SCREEN_OWNED_SPLITS:
        for owner, sheet in split.sheets.items():
            if sheet == "screen_agentic_library.tcss":
                return tuple(split.prefixes[owner])
    raise AssertionError("no screen-owned split writes screen_agentic_library.tcss")


def test_library_destination_classes_carry_the_library_split_prefix() -> None:
    """A token without the owner prefix would pin its rule to the boot bundle."""
    prefixes = _library_split_prefixes()
    for css_class in astuple(library_shell.LIBRARY_ADAPTIVE_READER_CLASSES):
        assert any(
            css_class == prefix or css_class.startswith(f"{prefix}-")
            for prefix in prefixes
        ), css_class


def test_the_boot_bundle_carries_only_the_grip_state_pair_of_the_shell_rules() -> None:
    """Only the ``:hover, .-active`` grip rule is boot-resident (``-active`` has
    no split owner); every other shell rule is lazy. A re-key that pulls more
    into boot fails here before the byte census does."""
    bundle = BUNDLED_STYLESHEET.read_text(encoding="utf-8")
    lines = [
        line.strip() for line in bundle.splitlines() if ".library-adaptive-reader-" in line
    ]
    assert lines == [
        ".library-adaptive-reader-shell > .library-adaptive-reader-pane-grip:hover,",
        ".library-adaptive-reader-shell > .library-adaptive-reader-pane-grip.-active {",
    ]


class _LibraryStyledHost(ConsolidatedCSSApp):
    """The app's real CSS: the boot bundle plus every lazy split sheet."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        yield library_shell.LibraryAdaptiveReaderShell(
            Static("Library"),
            Static("Items"),
            Static("Work"),
            _layout(),
            id_prefix="styled",
            library_label="Library",
            items_label="Items",
            id="styled-shell",
        )


async def test_a_focused_library_grip_resolves_the_lazy_sheet_focus_rule() -> None:
    """The grip's ``:focus`` rule (``bold reverse``, task-32053) lives in the lazy
    Library sheet; Button's default focus look is ``bold underline``."""
    app = _LibraryStyledHost()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grip = app.query_one("#styled-library-grip")
        grip.focus()
        await pilot.pause()
        assert "reverse" in str(grip.styles.text_style)
        assert "reverse" not in str(app.query_one("#styled-items-grip").styles.text_style)
EOF
```

- [ ] **Step 2: Run the pins (they hold already)**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py Tests/UI/test_library_adaptive_reader_shell.py -q -p no:cacheprovider -k "split_prefix or boot_bundle or lazy_sheet_focus or shared_shell_structure or grip_width_beside or calm_visual" 2>&1 | tail -1`

Expected: `7 passed` — the 3 new tests, `test_shared_shell_structure_is_owned_by_shared_tcss_selectors`, both parametrized arms of `test_no_sheet_declares_a_grip_width_beside_the_inline_one`, and `test_shared_tcss_owns_the_calm_visual_contract_for_every_reader`.

- [ ] **Step 3: Run the touched files against the base arm, and the new file alone**

```bash
bash <<'EOF'
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence
printf '%s\n' Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_css_build_integrity.py > "$EV/suites-css-quick.txt"
"$EV/paired.sh" cssquick "$EV/suites-css-quick.txt" -n 4
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider 2>&1 | tail -1
git status --short tldw_chatbook/css
EOF
```

Expected: `recovery=` in single digits on both arms and nothing between `new failures on head [cssquick] (must be empty):` and its end marker (the re-keyed pins keep their node ids); `21 passed` for the shared-shell file; and NO output from `git status --short tldw_chatbook/css` (this task changes no stylesheet, so boot bytes cannot move; Task 11 Step 2 measures them on both arms).

- [ ] **Step 4: Named mutations**

1. `rename-grip-class`: in `library_adaptive_reader_shell.py` change `grip="library-adaptive-reader-pane-grip",` to `grip="library-adaptive-reader-grip",`. Run `…/python -m pytest Tests/UI/test_adaptive_pane_shell.py Tests/UI/test_library_adaptive_reader_shell.py -q -p no:cacheprovider -k "lazy_sheet_focus or shared_shell_structure or grip_width_beside or calm_visual"`. Expected: the focus-rule test fails (`bold underline`), and the re-keyed pins fail (`IndexError` / missing substring). Restore; all pass.
2. `neutral-token`: in `library_adaptive_reader_shell.py` change `work="library-adaptive-reader-work",` to `work="adaptive-pane-work",`. Run `-k split_prefix`. Expected `1 failed` naming `adaptive-pane-work`. Restore; `1 passed`.
3. `other-owner-prefix`: change the same line to `work="settings-work",` (`settings` is the Settings split's owner prefix). Run `-k split_prefix`. Expected `1 failed` naming `settings-work`. Restore; `1 passed`.

Record all three.

- [ ] **Step 5: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_adaptive_pane_shell.py
git add Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_adaptive_pane_shell.py
git commit -m "test(css): key the Library shell rule pins to its destination classes (B0)" -m "No stylesheet change and zero boot bytes; ownership pins read LIBRARY_ADAPTIVE_READER_CLASSES, the split-prefix pin checks the Library owner only, and a styled test pins the lazy focus rule (shared pane-shell ADR)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

---

### Task 7: `BaseAppScreen.TAB_REGION` and `arrival_focus_target()`

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/base_app_screen.py` (imports; class attributes after `VERTICAL_BREAKPOINTS`; methods after `_first_focusable_in_content`)
- Modify (comment only): `Tests/UI/test_watchlists_content_pane.py` (the "`BaseAppScreen` defines no `BINDINGS` of its own" comment this task makes false)
- Test: `Tests/UI/test_base_app_screen_tab_region.py` (create, censused); `Tests/UI/test_base_app_screen_tab_region_routes.py` (create, not censused)

This task touches neither the shared module nor the Library, so it may run before Task 3 while the owner's Task 0 Step 6 answer is pending.

**Interfaces:**
- Consumes: `textual.css.match.match(selector_sets, node) -> bool`, `textual.css.parse.parse_selectors(css: str) -> tuple[SelectorSet, ...]`, `App.action_focus_next/previous`, `Screen.focus_next/focus_previous(selector)`, `BaseAppScreen._first_focusable_in_content()` (existing).
- Produces: `BaseAppScreen.TAB_REGION: ClassVar[str | None] = None`; `BaseAppScreen.BINDINGS` = `[tab → region_focus_next, shift+tab → region_focus_previous, *Screen's other bindings]` (non-priority); `arrival_focus_target() -> Widget | None`; `action_region_focus_next()`, `action_region_focus_previous()`; `_move_region_focus(direction: int) -> Widget | None`. B3 opts Roleplay in by setting `TAB_REGION = "#screen-content, #screen-content *"` and overriding `arrival_focus_target()`.

Mechanism (B0 research, focus.md): `DOMNode._merge_bindings` lets a subclass's entry replace Screen's per key, so the new binding replaces Screen's `app.focus_next` on every screen that does not re-declare `tab`. With `TAB_REGION is None` the action calls `App.action_focus_next/previous` — exactly what Screen's binding ran — so Tab is unchanged (E1/E2/E4: 10/10, 45/46 with the one difference reproduced base-vs-base, 12/12). `check_action` gating is wrong here: once the merge replaces Screen's binding there is no fall-through, and a `False` gate kills Tab. Screen's copy binding must be re-spread, or F1's generic help loses "Copy selected text" on ten screens. `arrival_focus_target()` is consulted only on the opted-in, nothing-focused Tab path; it is never wired into first paint in B0 (the nav bar's mount settling depends on `App.AUTO_FOCUS`).

- [ ] **Step 1: Write the failing tier-1 tests**

Create `Tests/UI/test_base_app_screen_tab_region.py`:

```python
"""``BaseAppScreen.TAB_REGION`` and ``arrival_focus_target()`` (Roleplay frame B0).

A bare ``textual.app.App`` hosts a real ``BaseAppScreen`` subclass whose
``compose`` is overridden (the ``test_base_app_screen_recompose_focus_seam``
pattern), so no ``TldwCli`` is mounted. The module still imports
``tldw_chatbook.app`` at collection time: ``Tests/UI/conftest.py``'s autouse
fixture would otherwise import it for the first time inside the per-test
sandbox and every test would error at setup with ``RecoveryRequired``
(lessons-testing-evidence).

The behaviour-neutral claim is checked against an IN-TEST "before" arm: a
subclass that re-declares Textual's own ``("tab", "app.focus_next")`` pair,
which is exactly what every route ran before B0. Never against recorded
sequences: first-mount content timing varies (B0 research E3).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, TextArea

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from tldw_chatbook.app_command_providers import _bindings_to_shortcuts
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.screen_registry import registered_screen_routes

_REGION = "#screen-content, #screen-content *"


class _ProbeScreen(BaseAppScreen):
    """Nav chrome, a content region with four controls, and a footer button."""

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(), "probe")

    def compose(self) -> ComposeResult:
        with Horizontal(id="probe-nav"):
            yield Button("Home", id="probe-nav-home")
            yield Button("Console", id="probe-nav-console")
        with Container(id="screen-content"):
            yield Input(id="probe-a")
            yield Button("B", id="probe-b")
            yield TextArea(id="probe-c")
            yield Button("D", id="probe-d")
        yield Button("Footer", id="probe-footer")


class _StockProbeScreen(_ProbeScreen):
    """The pre-B0 arm: Textual's own app-namespaced Tab bindings."""

    BINDINGS = [
        Binding("tab", "app.focus_next", "Focus Next", show=False),
        Binding("shift+tab", "app.focus_previous", "Focus Previous", show=False),
    ]


class _OptInProbeScreen(_ProbeScreen):
    TAB_REGION = _REGION

    def arrival_focus_target(self):
        return self.query_one("#probe-b")


class _EmptyOptInProbeScreen(_ProbeScreen):
    """Opted in, but the content region holds nothing focusable yet."""

    TAB_REGION = _REGION

    def compose(self) -> ComposeResult:
        with Horizontal(id="probe-nav"):
            yield Button("Home", id="probe-nav-home")
            yield Button("Console", id="probe-nav-console")
        yield Container(id="screen-content")
        yield Button("Footer", id="probe-footer")


class _OwnTabProbeScreen(_ProbeScreen):
    """The ChatScreen/LibraryScreen shape: a screen that re-declares tab."""

    BINDINGS = [Binding("tab", "probe_tab", "Probe tab", show=False)]

    def __init__(self) -> None:
        super().__init__()
        self.probe_tabs = 0

    def action_probe_tab(self) -> None:
        self.probe_tabs += 1


class _Host(App):
    def __init__(self, screen_type: type[_ProbeScreen]) -> None:
        super().__init__()
        self._screen_type = screen_type

    def on_mount(self) -> None:
        self.push_screen(self._screen_type())


async def _walk(screen_type, key: str, start: str | None, presses: int = 10) -> list:
    app = _Host(screen_type)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = app.screen
        if start is None:
            screen.set_focus(None)
        else:
            screen.query_one(f"#{start}").focus()
        await pilot.pause()
        landed = []
        for _ in range(presses):
            await pilot.press(key)
            landed.append(getattr(screen.focused, "id", None))
        return landed


_STARTS = [None, "probe-nav-home", "probe-a", "probe-c", "probe-d", "probe-footer"]


@pytest.mark.parametrize("key", ["tab", "shift+tab"])
@pytest.mark.parametrize("start", _STARTS)
async def test_tab_region_none_walks_exactly_like_textuals_stock_binding(start, key) -> None:
    assert await _walk(_ProbeScreen, key, start) == await _walk(_StockProbeScreen, key, start)


def test_generic_f1_help_lists_the_same_rows_as_textuals_screen() -> None:
    """``App._show_generic_screen_help`` renders ``getattr(screen, "BINDINGS")``."""
    assert _bindings_to_shortcuts(BaseAppScreen.BINDINGS) == _bindings_to_shortcuts(
        Screen.BINDINGS
    )


def test_merged_tab_bindings_are_screen_scoped_and_not_priority() -> None:
    """A priority Tab would preempt every screen's own on_key Tab trap."""
    merged = BaseAppScreen._merged_bindings.key_to_bindings
    assert [(b.action, b.priority) for b in merged["tab"]] == [("region_focus_next", False)]
    assert [(b.action, b.priority) for b in merged["shift+tab"]] == [
        ("region_focus_previous", False)
    ]
    assert set(merged) == set(Screen._merged_bindings.key_to_bindings)
    # The merge would still inherit Screen's copy binding without the re-spread;
    # F1's generic help reads BaseAppScreen.BINDINGS itself, so pin the list.
    assert [b.action for b in BaseAppScreen.BINDINGS if b.key == "ctrl+c,super+c"] == [
        "screen.copy_text"
    ]


async def test_region_actions_are_allowed_by_default() -> None:
    app = _Host(_ProbeScreen)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app.screen.TAB_REGION is None
        assert app.screen.check_action("region_focus_next", ()) is True
        assert app.screen.check_action("region_focus_previous", ()) is True


async def test_default_arrival_focus_target_is_the_first_content_control() -> None:
    app = _Host(_ProbeScreen)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app.screen.arrival_focus_target() is app.screen.query_one("#probe-a")


async def test_opt_in_region_wraps_inside_content_and_never_reaches_chrome() -> None:
    landed = await _walk(_OptInProbeScreen, "tab", "probe-d", presses=8)
    assert landed[0] == "probe-a"
    assert not any(
        name and name.startswith(("probe-nav", "probe-footer")) for name in landed
    ), landed


async def test_opt_in_from_chrome_keeps_the_app_wide_walk() -> None:
    assert await _walk(_OptInProbeScreen, "tab", "probe-nav-home", presses=2) == [
        "probe-nav-console",
        "probe-a",
    ]


async def test_opt_in_with_nothing_focused_lands_on_arrival_focus_target() -> None:
    assert await _walk(_OptInProbeScreen, "tab", None, presses=1) == ["probe-b"]


async def test_opt_in_with_an_empty_content_region_never_lands_tab_on_chrome() -> None:
    """The default hook falls back to the screen's first focusable widget (the
    nav bar) when the content holds nothing focusable; that target is outside
    the region and must be ignored, not focused."""
    assert await _walk(_EmptyOptInProbeScreen, "tab", None, presses=1) == [None]


async def test_a_screen_that_redeclares_tab_keeps_its_own_binding() -> None:
    app = _Host(_OwnTabProbeScreen)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.screen.query_one("#probe-a").focus()
        await pilot.pause()
        await pilot.press("tab")
        assert app.screen.probe_tabs == 1
        assert app.screen.focused is app.screen.query_one("#probe-a")


#: Screens that keep their own Tab binding by design, with the action name.
_OWN_TAB = {
    # TASK-2154.11: Console's region-scoped Tab.
    "tldw_chatbook.UI.Screens.chat_screen.ChatScreen": ("focus_next", "focus_previous"),
    # task-32052: Library's Tab confinement; TAB_REGION adoption is FU-1.
    "tldw_chatbook.UI.Screens.library_screen.LibraryScreen": ("focus_next", "focus_previous"),
}


def _production_screen_classes() -> list[type[BaseAppScreen]]:
    for route in registered_screen_routes():
        try:
            route.load_screen_class()
        except Exception:  # noqa: BLE001 - the PR fast lane installs minimal deps
            continue
    seen: set[type] = set()
    stack = list(BaseAppScreen.__subclasses__())
    while stack:
        screen_class = stack.pop()
        if screen_class in seen:
            continue
        seen.add(screen_class)
        stack.extend(screen_class.__subclasses__())
    return sorted(
        (cls for cls in seen if cls.__module__.startswith("tldw_chatbook.")),
        key=lambda cls: f"{cls.__module__}.{cls.__qualname__}",
    )


def test_every_production_screen_inherits_the_region_binding_or_is_allowlisted() -> None:
    classes = _production_screen_classes()
    names = {f"{cls.__module__}.{cls.__qualname__}" for cls in classes}
    assert len(classes) >= 15, sorted(names)
    assert set(_OWN_TAB) <= names
    assert "tldw_chatbook.UI.Screens.personas_screen.PersonasScreen" in names
    wrong: dict[str, object] = {}
    for cls in classes:
        name = f"{cls.__module__}.{cls.__qualname__}"
        merged = cls._merged_bindings.key_to_bindings
        actions = (
            tuple(binding.action for binding in merged.get("tab", [])),
            tuple(binding.action for binding in merged.get("shift+tab", [])),
        )
        expected = _OWN_TAB.get(name, ("region_focus_next", "region_focus_previous"))
        if actions != ((expected[0],), (expected[1],)):
            wrong[name] = actions
        if cls.TAB_REGION is not None:
            wrong[name] = ("TAB_REGION", cls.TAB_REGION)
    assert wrong == {}
```

- [ ] **Step 2: Run to see it fail**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_base_app_screen_tab_region.py -q -p no:cacheprovider 2>&1 | tail -12`

Expected: `7 failed, 15 passed`. The 12 `test_tab_region_none_walks…` cases, `test_generic_f1_help…`, `test_opt_in_from_chrome…` (the stock walk already goes nav → nav → content) and `test_a_screen_that_redeclares_tab…` PASS: nothing changed yet. FAILED: `test_merged_tab_bindings…` (`('app.focus_next', False)`), `test_region_actions_are_allowed_by_default` and `test_default_arrival_focus_target…` (`AttributeError: … 'TAB_REGION'` / `'arrival_focus_target'`), `test_opt_in_region_wraps_inside_content…`, `test_opt_in_with_nothing_focused…` and `test_opt_in_with_an_empty_content_region…` (Tab walks app-wide; the last lands on `probe-nav-home`), and `test_every_production_screen…` (every non-allowlisted screen reports `('app.focus_next',)`).

- [ ] **Step 3: Implement**

In `tldw_chatbook/UI/Navigation/base_app_screen.py`:

Old:
```python
from typing import TYPE_CHECKING, Optional, Dict, Any
```
New:
```python
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional
```

Old:
```python
from textual.message import Message
```
New:
```python
from textual.binding import Binding
from textual.message import Message
```

Old:
```python
from textual.css.query import QueryError
```
New:
```python
from textual.css.match import match
from textual.css.parse import parse_selectors
from textual.css.query import QueryError
```

Old:
```python
    VERTICAL_BREAKPOINTS = [
        (0, "shell-header-compact"),
        (_DESTINATION_HEADER_COMPACT_FLOOR_HEIGHT + 1, "shell-header-normal"),
    ]
```
New:
```python
    VERTICAL_BREAKPOINTS = [
        (0, "shell-header-compact"),
        (_DESTINATION_HEADER_COMPACT_FLOOR_HEIGHT + 1, "shell-header-normal"),
    ]

    #: Roleplay frame B0 (the shared adaptive-pane-shell ADR): the opt-in Tab region. ``None`` -- every
    #: route in B0 -- keeps Textual's stock app-wide Tab walk exactly. A
    #: selector string (Roleplay sets ``"#screen-content, #screen-content *"``
    #: in B3) confines Tab/Shift+Tab to it while focus is inside; from chrome
    #: (nav bar, footer) the walk stays app-wide so the bar remains
    #: traversable -- the Library task-32052 / Console TASK-2154.11 rule as one
    #: shared seam.
    TAB_REGION: ClassVar[str | None] = None

    BINDINGS = [
        # Replaces Screen's ("tab", "app.focus_next") and ("shift+tab",
        # "app.focus_previous") through DOMNode._merge_bindings (a later class
        # wins per key). Non-priority like Screen's: a priority Tab here would
        # preempt every screen's own on_key Tab trap (lessons-textual).
        # ChatScreen and LibraryScreen re-declare tab and keep theirs. Never
        # gate region_focus_* off in check_action: once this replaces
        # Screen's binding there is no fall-through, so False kills Tab.
        Binding("tab", "region_focus_next", "Focus Next", show=False),
        Binding("shift+tab", "region_focus_previous", "Focus Previous", show=False),
        # Re-spread the rest of Screen.BINDINGS: App._show_generic_screen_help
        # renders F1 from getattr(screen, "BINDINGS") on ten screens, and
        # PersonasScreen and ChatScreen spread this list; without it the
        # "Copy selected text" row would vanish from their help.
        *(binding for binding in Screen.BINDINGS if binding.key not in ("tab", "shift+tab")),
    ]
```

Old:
```python
            if inside is not None:
                return inside
        return next(iter(chain), None)
```
New:
```python
            if inside is not None:
                return inside
        return next(iter(chain), None)

    def arrival_focus_target(self) -> "Widget | None":
        """Where Tab lands when this opted-in screen has nothing focused.

        Consulted only by ``_move_region_focus`` while ``TAB_REGION`` is set
        and nothing holds focus -- never at first paint in B0, where
        ``App.AUTO_FOCUS`` still decides (the nav bar's mount settling
        depends on it). Roleplay (B3) overrides it to return its items list.
        An override returns a focusable widget inside ``TAB_REGION``, or
        ``None`` to fall back to the region walk. A target outside the region
        is ignored the same way: the default can return the nav bar when the
        content holds nothing focusable yet.

        Returns:
            ``_first_focusable_in_content()`` by default.
        """
        return self._first_focusable_in_content()

    def action_region_focus_next(self) -> None:
        """Tab: confined to ``TAB_REGION`` when set, else Textual's own walk."""
        self._move_region_focus(1)

    def action_region_focus_previous(self) -> None:
        """Shift+Tab: the reverse of ``action_region_focus_next``."""
        self._move_region_focus(-1)

    def _move_region_focus(self, direction: int) -> "Widget | None":
        """Move focus one step, inside ``TAB_REGION`` while focus is in it.

        Args:
            direction: ``1`` for Tab, ``-1`` for Shift+Tab.

        Returns:
            The newly focused widget, or ``None``.
        """
        region = self.TAB_REGION
        if region is None:
            # Exactly what Screen's binding ran: App.action_focus_next is
            # ``self.screen.focus_next()``, and this binding only fires while
            # this screen is ``app.screen``. Kept as the App call so a future
            # TldwCli override still applies on every non-opted route.
            if direction > 0:
                self.app.action_focus_next()
            else:
                self.app.action_focus_previous()
            return self.focused
        focused = self.focused
        selector_set = parse_selectors(region)
        if focused is None:
            target = self.arrival_focus_target()
            # Only a target inside the region: the default hook falls back to
            # the screen's first focusable widget (often the nav bar) when the
            # content holds nothing focusable yet.
            if (
                target is not None
                and target.focusable
                and match(selector_set, target)
            ):
                self.set_focus(target)
                return self.focused
            selector = region
        elif match(selector_set, focused):
            # The same predicate Screen._move_focus filters with, so "inside"
            # and "walk" can never disagree.
            selector = region
        else:
            selector = "*"
        if direction > 0:
            return self.focus_next(selector)
        return self.focus_previous(selector)
```

- [ ] **Step 4: Run to see it pass**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_base_app_screen_tab_region.py Tests/Architecture/test_base_app_screen_recompose_focus_seam.py Tests/Architecture/test_on_mount_super_guard.py -q -p no:cacheprovider 2>&1 | tail -1`

Expected: `29 passed` (22 + 3 + 4). The tier-1 file also passes run on its own (its module-scope `import tldw_chatbook.app`).

- [ ] **Step 5: Write the tier-2 every-route test**

Create `Tests/UI/test_base_app_screen_tab_region_routes.py`:

```python
"""Every route with ``TAB_REGION is None`` keeps Textual's stock Tab walk (Roleplay frame B0).

Mounted real screens: slow (about 3 s per screen class), so NOT in the PR-gate
census. The test is ARM-AGNOSTIC: the B0 gate run executes it on both paired
arms (Task 11 copies this file into the base arm), so a red on head only is a
B0 change, and a red on both arms is a limit of the probe, not of B0.

The oracle is Textual's own stock walk taken in the SAME mount: from a saved
focus S, ``app.action_focus_next()`` -- exactly what Screen's
``("tab", "app.focus_next")`` binding ran before B0, and what
``region_focus_next`` calls while ``TAB_REGION is None`` -- records its
target; focus goes back to S; then the real key press must land on the same
widget. Never a recorded sequence and never a model of the chain: first-mount
content timing varies (B0 research E3: one cold Meetings mount lacked a
late-mounted button).

Parametrised over ``module.Class`` strings read from the registry, so
collecting this file imports no screen module; each test loads its own class.
"""

from __future__ import annotations

import time

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.screen_registry import registered_screen_routes

#: Re-declare Tab themselves (allowlisted in test_base_app_screen_tab_region.py);
#: their own suites pin their Tab.
_OWN_TAB_CLASS_NAMES = frozenset({"ChatScreen", "LibraryScreen"})


def _routes_by_class_path() -> dict[str, object]:
    """``"module.Class"`` -> its first registered route, importing nothing."""
    routes: dict[str, object] = {}
    for route in registered_screen_routes():
        if route.class_name in _OWN_TAB_CLASS_NAMES:
            continue
        routes.setdefault(f"{route.module_path}.{route.class_name}", route)
    return routes


_ROUTES = _routes_by_class_path()


class _Host(ConsolidatedCSSApp):
    def __init__(self, app_instance, screen_class) -> None:
        super().__init__()
        self.app_instance = app_instance
        self._screen_class = screen_class

    async def on_mount(self) -> None:
        await self.push_screen(self._screen_class(self.app_instance))

    def on_navigate_to_screen(self, message) -> None:
        """Swallow navigation requests: this host mounts one screen only."""


def _inside_content(widget) -> bool:
    """Whether ``widget`` sits in ``#screen-content``.

    Steps that START in the chrome are not compared: the nav bar's overflow
    re-lays itself out as focus moves through it, so the oracle's walk and
    the key press that follows can see different chains (measured red,
    intermittently, on BOTH arms while planning: Shift+Tab from
    ``nav-meetings``). The first step of each direction is always compared,
    so Shift+Tab still crosses from the content into the chrome once.
    """
    return any(ancestor.id == "screen-content" for ancestor in widget.ancestors)


def _ident(widget) -> str | None:
    if widget is None:
        return None
    return widget.id or f"<{type(widget).__name__}>"


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    "class_path", sorted(_ROUTES), ids=lambda path: path.rsplit(".", 1)[-1]
)
async def test_tab_from_the_first_content_control_follows_the_stock_walk(class_path) -> None:
    screen_class = _ROUTES[class_path].load_screen_class()
    if not (isinstance(screen_class, type) and issubclass(screen_class, BaseAppScreen)):
        pytest.skip(f"{class_path}: not a loadable BaseAppScreen in this environment")
    host = _Host(_build_test_app(), screen_class)
    async with host.run_test(size=(140, 42)) as pilot:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if isinstance(host.screen, screen_class) and host.screen.region.width > 0:
                break
            await pilot.pause(0.02)
        screen = host.screen
        assert isinstance(screen, screen_class), type(screen)
        await pilot.pause(0.8)
        # Arm-agnostic: the base arm has neither the attribute nor the actions.
        assert getattr(screen, "TAB_REGION", None) is None
        if hasattr(screen, "action_region_focus_next"):
            assert screen.check_action("region_focus_next", ()) is True
            assert screen.check_action("region_focus_previous", ()) is True
        mismatches = []
        compared = 0
        for key, stock_walk in (
            ("tab", host.action_focus_next),
            ("shift+tab", host.action_focus_previous),
        ):
            screen.set_focus(screen._first_focusable_in_content())
            await pilot.pause(0.1)
            for step in range(3):
                start = screen.focused
                if start is None or (step and not _inside_content(start)):
                    break
                compared += 1
                stock_walk()
                expected = screen.focused
                screen.set_focus(start)
                await pilot.pause(0.05)
                await pilot.press(key)
                await pilot.pause(0.05)
                if screen.focused is not expected:
                    mismatches.append(
                        (key, _ident(start), _ident(expected), _ident(screen.focused))
                    )
        assert compared >= 2, compared
        assert mismatches == []
```

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_base_app_screen_tab_region_routes.py -q -p no:cacheprovider -n 4 -rs 2>&1 | tail -4`

Expected: one pass per routable non-override `BaseAppScreen` class (about 21 on `fccf70d3b0`; `passed` must be ≥ 15) and a skip for any route that is not one; `passed + skipped` equals `--collect-only -q | tail -1`. If a case fails, run the same file on the base arm before debugging (copy it in as Task 11 Step 4 does, run, then delete it): red on head only is a B0 Tab change, so stop and fix it; red on both arms is a limit of the probe (for example a widget that consumes Tab itself), so record the route and the reason in `$EV/gates.txt` and in the PR body.

- [ ] **Step 6: Named mutations**

1. `confine-none`: in `_move_region_focus`, change `region = self.TAB_REGION` to `region = self.TAB_REGION or "#screen-content, #screen-content *"`. Run `…/python -m pytest Tests/UI/test_base_app_screen_tab_region.py -q -p no:cacheprovider -k stock_binding`. Expected: most of the 12 cases fail (Tab no longer reaches the nav or footer). Restore; `12 passed`.
2. `drop-copy`: delete the line `        *(binding for binding in Screen.BINDINGS if binding.key not in ("tab", "shift+tab")),`. Run `-k "f1_help or merged_tab"`. Expected `2 failed`: the F1 test loses "Copy selected text", and the merged test's explicit `BaseAppScreen.BINDINGS` check fails (the merge alone still inherits Screen's copy binding, so without that check this mutation would leave the merged test green). Restore; `2 passed`.
3. `priority-tab`: add `priority=True` to the `tab` binding. Run `-k merged_tab`. Expected `1 failed`. Restore; `1 passed`.
4. `arrival-outside-region`: in `_move_region_focus`, delete the line `                and match(selector_set, target)`. Run `-k empty_content_region`. Expected `1 failed` (`['probe-nav-home'] == [None]`). Restore; `1 passed`.
5. `dead-tab` (tier 2): in `_move_region_focus`, replace the line `                self.app.action_focus_next()` with `                pass`. Run `…/python -m pytest Tests/UI/test_base_app_screen_tab_region_routes.py -q -p no:cacheprovider -n 4`. Expected: nearly every case fails (20 of 21 while planning; a screen whose first content control is its only Tab stop cannot see it). Restore; all pass.

(Variants of 1 and 2 were run in the B0 research prototype: 12/12 and the F1 test red; all five were re-run against this plan's code in plan review.) Record all five.

- [ ] **Step 7: Fix the comment B0 makes false, lint and commit**

In `Tests/UI/test_watchlists_content_pane.py`:

Old:
```python
    # `BaseAppScreen` defines no `BINDINGS` of its own, so this resolves
    # through the MRO to Textual's `Screen.BINDINGS` (tab/shift+tab/copy at
    # the time of writing) -- checking the resolved attribute, not assuming
    # BaseAppScreen is empty, is the point of the audit.
```
New:
```python
    # `BaseAppScreen` re-declares Screen's tab/shift+tab/copy keys (the shared
    # adaptive-pane-shell ADR: tab/shift+tab become the opt-in region actions,
    # copy is re-spread from `Screen.BINDINGS`) -- checking the resolved
    # attribute, not assuming its contents, is the point of the audit.
```

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/UI/Navigation/base_app_screen.py Tests/UI/test_base_app_screen_tab_region.py Tests/UI/test_base_app_screen_tab_region_routes.py Tests/UI/test_watchlists_content_pane.py
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence/plug /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_watchlists_content_pane.py -p b0_bootstrap_all -q -p no:cacheprovider -n 4 2>&1 | tail -1
git diff --stat -- tldw_chatbook/
git add tldw_chatbook/UI/Navigation/base_app_screen.py Tests/UI/test_base_app_screen_tab_region.py Tests/UI/test_base_app_screen_tab_region_routes.py Tests/UI/test_watchlists_content_pane.py
git commit -m "feat(navigation): opt-in BaseAppScreen TAB_REGION and arrival_focus_target (B0)" -m "None on every route keeps Textual's stock Tab walk; Screen's copy binding is re-spread so F1 help is unchanged; an arrival target outside the region is ignored (shared pane-shell ADR)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: `All checks passed!`; the watchlists content-pane summary line, in which the audit test holding this comment passes (it reads the resolved `BaseAppScreen.BINDINGS`, whose keys are unchanged: tab, shift+tab, ctrl+c, super+c) and any other failure also fails on the base arm (check with `$EV/paired.sh` on that one file) or is a pre-existing flake under the Task 11 Step 4 rule (plan review saw three intermittent failures in this file on BOTH arms); the diff stat shows only `base_app_screen.py` under `tldw_chatbook/` and no `logger` line was added (`git diff HEAD~1 -- tldw_chatbook/UI/Navigation/base_app_screen.py | grep -c "logger"` prints `0` after the commit).

---

### Task 8: The `panes` pattern family (G16)

**Files:**
- Modify: `tldw_chatbook/css/patterns.json` (new family after `sizing`)
- Modify: `backlog/docs/component-patterns.md` (quick-reference row; new section at the end)
- Modify: `tldw_chatbook/Widgets/pattern_gallery.py` (module docstring only)
- Test: `Tests/UI/test_adaptive_pane_shell.py` (import block; append: the `panes` registration and a permanent Python-style pin on B0's production modules)

**Interfaces:**
- Consumes: `build_css.CSS_MODULES` (existing); `inventory_styles(source) -> list[StyleWrite]` with `.violation`, `.line`, `.property`, `.form` from `Tests/UI/python_style_inventory.py` (existing; 0 violations in today's versions of the four existing modules).
- Produces: `patterns.json` → `families["panes"]` with `owning_sheet = "features/_library.tcss"`, a `widget_contract` naming `AdaptivePaneShell`, `classes = {}`, and a `note`.

Decision: `panes` is a **widget-contract family** with no public classes, like `lists`. A canonical neutral class would either sit in a rule (a neutral token pins the rule to boot, against 50 B of headroom) or be a nominal class with no rule; canonical Library classes would have to be "rendered" in the gallery, which renders against the boot bundle only, so a sample would paint unstyled and misdocument the family (and both gallery SVGs are already stale on dev). The gallery registration is therefore a docstring entry that changes no rendering. B1 adds Roleplay's prefixed classes beside it if it needs per-class governance.

- [ ] **Step 1: Write the failing test**

Edit the import block of `Tests/UI/test_adaptive_pane_shell.py`:

Old:
```python
import ast
import os
```
New:
```python
import ast
import json
import os
```

Old:
```python
from Tests.UI.consolidated_css import (
    APP_STYLESHEETS,
    BUNDLED_STYLESHEET,
    ConsolidatedCSSApp,
)
```
New:
```python
from Tests.UI.consolidated_css import (
    APP_STYLESHEETS,
    BUNDLED_STYLESHEET,
    ConsolidatedCSSApp,
)
from Tests.UI.python_style_inventory import inventory_styles
```

Append:

```bash
cat >> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0/Tests/UI/test_adaptive_pane_shell.py <<'EOF'


# ---------------------------------------------------------------------------
# The ``panes`` pattern family (B0, G16) and B0's Python-style floor.
# ---------------------------------------------------------------------------

#: Every production module B0 changes. ``test_python_style_ratchet`` is red on
#: dev for other modules, so a failure-set diff cannot see a NEW offender here;
#: this pin can (ADR-161's hard floor, the same inventory the ratchet uses).
B0_PRODUCTION_MODULES = (
    "tldw_chatbook/Widgets/adaptive_pane_shell.py",
    "tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py",
    "tldw_chatbook/Widgets/Library/library_rail.py",
    "tldw_chatbook/UI/Navigation/base_app_screen.py",
    "tldw_chatbook/Utils/adaptive_reader_state.py",
)


def test_b0_production_modules_carry_no_python_style_violation() -> None:
    violations = {
        module: [
            (write.line, write.property, write.form)
            for write in inventory_styles((ROOT / module).read_text(encoding="utf-8"))
            if write.violation
        ]
        for module in B0_PRODUCTION_MODULES
    }
    assert {module: found for module, found in violations.items() if found} == {}


def test_panes_family_is_registered_in_the_registry_the_catalog_and_the_gallery() -> None:
    registry = json.loads(
        (ROOT / "tldw_chatbook/css/patterns.json").read_text(encoding="utf-8")
    )
    panes = registry["families"]["panes"]
    assert panes["classes"] == {}
    assert "AdaptivePaneShell" in panes["widget_contract"]
    assert panes["owning_sheet"] in build_css.CSS_MODULES
    catalog = (ROOT / "backlog/docs/component-patterns.md").read_text(encoding="utf-8")
    assert "## Panes (adaptive pane shell)" in catalog
    assert "| panes |" in catalog
    gallery = (ROOT / "tldw_chatbook/Widgets/pattern_gallery.py").read_text(
        encoding="utf-8"
    )
    assert "``panes``" in gallery and "AdaptivePaneShell" in gallery
EOF
```

- [ ] **Step 2: Run to see it fail**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider -k panes_family 2>&1 | tail -2`

Expected: `1 failed` with `KeyError: 'panes'`.

- [ ] **Step 3: Register the family**

In `tldw_chatbook/css/patterns.json`:

Old:
```json
    }
  },
  "deprecated": {
```
New:
```json
    },
    "panes": {
      "owning_sheet": "features/_library.tcss",
      "widget_contract": "AdaptivePaneShell + AdaptivePaneGrip + DestinationRailRowButton (Widgets/adaptive_pane_shell.py): every rule is keyed to a destination's own AdaptivePaneClasses and lives in that destination's lazy split sheet; no neutral public classes (the shared adaptive-pane-shell ADR)",
      "classes": {},
      "note": "Roleplay frame B0 (the shared adaptive-pane-shell ADR in backlog/decisions/). A neutral class in a selector has no split owner, so build_css would pin its rule to the boot bundle; destination classes keep every rule lazy. Library: LIBRARY_ADAPTIVE_READER_CLASSES, rules in features/_library.tcss. Roleplay: its own prefixed copy in features/_roleplay.tcss (B1), pinned equal to the Library set by B1's parity test."
    }
  },
  "deprecated": {
```

Validate: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import json; d = json.load(open('tldw_chatbook/css/patterns.json')); print(list(d['families'])[-1])"` → `panes`.

In `backlog/docs/component-patterns.md`, the quick-reference table:

Old:
```markdown
literals, task 12) |

## Entry template
```
New:
```markdown
literals, task 12) |
| panes | `features/_library.tcss` | none — widget contract (`AdaptivePaneShell`/`AdaptivePaneGrip`/`DestinationRailRowButton` keyed to destination classes; ADR-211) |

## Entry template
```

Append the section:

````bash
cat >> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0/backlog/docs/component-patterns.md <<'EOF'

---

## Panes (adaptive pane shell)

**Purpose.** One three-pane destination frame — navigation rail, items list,
work pane — with a full-height grip after each optional pane, shared by the
Library's adaptive readers and Roleplay (ADR-211). The widgets live in
`tldw_chatbook/Widgets/adaptive_pane_shell.py`; the pure geometry lives in
`Utils/adaptive_reader_state.py` (`resolve_adaptive_pane_layout`). This family
is a **widget contract**, not a class vocabulary: every rule is keyed to a
destination's own classes and lives in that destination's lazy split sheet.

**When-not-to-use.** Not for a destination that is not on the adaptive-shell
grammar (ADR-211 does not authorise Watchlists convergence, an app-wide
workbench or a third shell pane). Never add a neutral class to a shell rule:
a token with no split owner pins the whole rule to the boot bundle.

**Structure.**

```python
from tldw_chatbook.Widgets.adaptive_pane_shell import (
    AdaptivePaneClasses,
    AdaptivePaneShell,
)

ROLEPLAY_CLASSES = AdaptivePaneClasses(
    shell="roleplay-shell",
    nav="roleplay-nav",
    items="roleplay-items",
    work="roleplay-shell-work",
    grip="roleplay-shell-grip",
)

def compose_frame(self):
    yield AdaptivePaneShell(
        rail, items, work, layout,
        id_prefix="roleplay",
        library_label="Navigation",
        items_label="Characters",
        destination=ROLEPLAY_CLASSES,
        id="roleplay-shell",
    )
```

B1 chooses Roleplay's final names; every value must equal or start with one
of the destination's split prefixes (for Roleplay, spec R18's
`roleplay-shell`, `roleplay-rail`, `roleplay-nav`, `roleplay-items`) or its
rule lands in the boot bundle.

**Class inventory.** None public. Each destination supplies an
`AdaptivePaneClasses` set: `shell`, `nav`, `items`, `work`, `grip`. The
Library's set is `LIBRARY_ADAPTIVE_READER_CLASSES`
(`library-adaptive-reader-shell`, `-library`, `-items`, `-work`,
`-pane-grip`); Roleplay's arrives in B1.

**States.** Grip rest: `$ds-surface-raised` background, `$ds-text-muted`
text, no outline. Hover: `$ds-text-primary`. Focus: `$ds-action-focus`
with `bold reverse` and no outline (task-31276: an outline on a shell-tall
grip paints over the work pane's first row). Disabled: grips are never
disabled; a closed pane is `display: none` + `disabled`. Rail rows
(`DestinationRailRowButton`): the current row is `▸` + bold, the focused
row is the destination's left focus bar, never the same token as current.

**Tokens consumed.** `$ds-surface-raised`, `$ds-text-muted`,
`$ds-text-primary`, `$ds-action-focus`, `$ds-width-fill`, `$ds-height-full`,
`$ds-size-0`, `$ds-space-0`. Widths are never tokens: the resolver sets them
inline (`# ds-runtime:`), and the grip reserves exactly the profile's
`grip_width`.

**Python idiom.** Resolve with `resolve_adaptive_pane_layout`, then
`shell.sync_layout(layout)` (equality-guarded; never recompose). Rename a
grip with `grip.sync_label(name)`. Fit rail rows with
`fit_rail_row_label(row, width, current=...)` and patch them with
`DestinationRailRowButton.sync_row`. Handle `PaneToggleRequested`,
`AdaptivePaneShellResized` and `PaneVisibilityChanged` with `@on(...)`.

**Lifecycle.** Canonical as a contract (registry `"classes"` empty by
design; owning sheet `features/_library.tcss` for the Library's set).
EOF
````

In `tldw_chatbook/Widgets/pattern_gallery.py`:

Old:
```python
``Tests/UI/snapshots/pattern_gallery/`` pin this rendering (spec 3.4).
"""
```
New:
```python
``Tests/UI/snapshots/pattern_gallery/`` pin this rendering (spec 3.4).

Widget-contract families have no public classes, so nothing of theirs is
composed here by class name: ``lists`` is exercised through the stock widgets
below, and ``panes`` (``AdaptivePaneShell``/``AdaptivePaneGrip``/
``DestinationRailRowButton``; the shared adaptive-pane-shell ADR) is deliberately NOT rendered -- its
rules are keyed to destination classes in lazy split sheets, and this gallery
renders against the boot bundle only, so a sample would paint unstyled and
misdocument the family. See ``backlog/docs/component-patterns.md``.
"""
```

- [ ] **Step 4: Run to see it pass, then the governance suites against the base arm**

`cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider 2>&1 | tail -1`

Expected: `23 passed`.

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b0-evidence
"$EV/paired.sh" css "$EV/suites-css.txt" -n 4
# The two ratchets are red on BOTH arms on dev, so the failure-set diff cannot
# see a new offender inside them: diff their offender lines instead.
for ARM in base head; do
  if [ "$ARM" = base ]; then T=$MAIN/.worktrees/roleplay-b0-base; else T=$MAIN/.worktrees/roleplay-b0; fi
  (cd "$T" && PYTHONPATH="$EV/plug" "$PY" -m pytest Tests/UI/test_component_pattern_governance.py -p b0_bootstrap_all -k "python_style_ratchet or dimension_literal_ratchet" -p no:cacheprovider -q --tb=short 2>&1) \
    | sed -nE 's/^E +([A-Za-z_][^ ]*\.(py|tcss)(:[0-9]+)?: .*)$/\1/p' | grep -vE '(Error|Warning)' | sort -u > "$EV/ratchet-offenders-$ARM.txt"
  echo "$ARM offenders: $(wc -l < "$EV/ratchet-offenders-$ARM.txt")"
done
diff "$EV/ratchet-offenders-base.txt" "$EV/ratchet-offenders-head.txt" && echo "ratchet offenders identical"
EOF
```

Expected: no new failures on head, with `recovery=` in single digits on both arms. (On `fccf70d3b0` both arms carry 17 known failures: 11 module-size rows, the dimension-literal and Python-style ratchets, one class-level-CSS allowlist test, both gallery snapshots and one staleness-manifest fixture test.) Then the same nonzero offender count on both arms (4 on `fccf70d3b0` in plan review: three `.tcss` sheets and `Widgets/Library/library_skill_work_pane.py:170`; both tests are private-profile child runs, so the filter drops the child's warning and traceback lines) and `ratchet offenders identical`: B0 adds no `.tcss` file and no Python-style write without its `# ds-runtime:` marker. (`test_b0_production_modules_carry_no_python_style_violation` above is the permanent form of the Python half.)

- [ ] **Step 5: Named mutation `drop-ds-runtime`**

In `tldw_chatbook/Widgets/adaptive_pane_shell.py`, delete the line `        # ds-runtime: profile-supplied grip columns reserved by the layout` (above `self.set_styles(width=width)` in `AdaptivePaneGrip.sync_width`). Run `…/python -m pytest Tests/UI/test_adaptive_pane_shell.py -q -p no:cacheprovider -k python_style`. Expected: `1 failed` naming `tldw_chatbook/Widgets/adaptive_pane_shell.py` with one `(<line>, 'width', 'set_styles')` entry (verified against the inventory while planning). Restore the line with the Edit tool; `1 passed`. Record it.

- [ ] **Step 6: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
git add tldw_chatbook/css/patterns.json backlog/docs/component-patterns.md tldw_chatbook/Widgets/pattern_gallery.py Tests/UI/test_adaptive_pane_shell.py
git commit -m "docs(patterns): register the panes widget-contract family (B0, G16)" -m "Registry, catalog section and gallery docstring; no rendering change and no class to define (shared pane-shell ADR)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

---

### Task 9: ADR-211 and the governance amendments (G1, G1a, G1b, G2, G3, G15)

**Files:**
- Create: `backlog/decisions/211-shared-adaptive-pane-shell.md`
- Modify: `backlog/decisions/086-library-adaptive-reader-shell.md` (header), `backlog/decisions/084-library-media-reader-ia.md` (header), `backlog/decisions/README.md` (three rows), `backlog/docs/design-language.md` (§2.8)

**Interfaces:**
- Consumes: the names produced by Tasks 1-8 (the ADR cites them).
- Produces: ADR-211 (provisional number).

Why the ADR comes now, not first: it records decisions the code made concrete (the class names, the `AdaptivePaneClasses` shape, the Tab mechanism, the `panes` registration) and cites the census rule the next task applies. Written first, it would be a second copy of the spec that drifts during implementation; written last-but-two, it is reviewed against working code and still lands in the same PR.

There is no executable behaviour here, so the "test" steps are exact verification commands.

- [ ] **Step 1: Re-run the ADR number sweep**

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook
git -C "$MAIN" fetch origin --prune --quiet
{
  git -C "$MAIN" for-each-ref --format='%(refname)' refs/remotes/origin refs/heads | while read -r ref; do
    git -C "$MAIN" ls-tree --name-only "${ref}" backlog/decisions/ 2>/dev/null
  done
  git -C "$MAIN" worktree list --porcelain | sed -n 's/^worktree //p' | while read -r wt; do ls "${wt}/backlog/decisions" 2>/dev/null; done
} | sed -nE 's#^(.*/)?([0-9]{3})-.*#\2#p' | sort -un | tail -5
EOF
```

Expected: the five highest ADR numbers anywhere; `210` the highest on 2026-10-02. If anything ≥ 211 appears, finish this task with `211`, commit it, and then run the renumber script of Task 11 Step 8 with the next free number (it rewrites every `ADR-211` and `211-shared-adaptive-pane-shell` in the files this branch changed, this plan excluded, and renames the file). Code and CSS carry no ADR number, so nothing else changes.

- [ ] **Step 2: Write the ADR**

Create `backlog/decisions/211-shared-adaptive-pane-shell.md` (the `@@B0@@`, `@@FU1@@` and `@@B7@@` tokens are replaced in Step 5 from `$EV/b0-task-id.txt`, `$EV/followup-ids.txt` and `$EV/slice-ids.txt`):

````markdown
# ADR-211: Share the adaptive pane shell across destinations (Library and Roleplay)

Status: Accepted 2026-10-02 (owner-approved design spec, PR #2960). The number is provisional until merge: re-run the all-remotes sweep and renumber if 211 is taken (`backlog/docs/lessons-backlog-hygiene.md`, "ADR numbers collide across concurrent branches").
Date: 2026-10-02
Task: [TASK-@@B0@@](../tasks/task-@@B0@@%20-%20Roleplay-frame-B0-shared-pane-shell-primitives-and-the-frame-ADR-no-visible-change.md)
Spec: [Roleplay on the Library frame (sub-project B)](../../Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md), sections 2.1, 2.12, 4.3, 5.9 (G1, G1a, G1b, G2, G3, G15, G16) and 5.10
Plan: [B0 implementation plan](../../Docs/superpowers/plans/2026-10-02-roleplay-b0-shared-frame-primitives.md)
Amends: [ADR-086](086-library-adaptive-reader-shell.md) (the shell is no longer Library-local); [ADR-084](084-library-media-reader-ia.md) (its five-column grips become the shared grip grammar)
Supersedes: N/A

## Decision

The adaptive pane shell — a navigation rail, an items list and a work pane, with a full-height grip after each optional pane — is a **shared destination primitive** for the Library's adaptive readers and for Roleplay. Its widgets live in one module, `tldw_chatbook/Widgets/adaptive_pane_shell.py`; its pure layout policy stays in `tldw_chatbook/Utils/adaptive_reader_state.py`.

### 1. What is shared, and where it lives

| Responsibility | Home | Rule |
|---|---|---|
| Resolver, profile, preferences, effective layout, normaliser | `Utils/adaptive_reader_state.py` | Behaviour byte-identical. `AdaptivePaneProfile`, `AdaptivePanePreferences`, `AdaptivePaneLayout` and `resolve_adaptive_pane_layout` are the SAME objects as the reader names; `nav_open`/`nav_width` are read-only properties, never dataclass fields. A golden digest captured before the aliases existed pins the resolver. The normaliser is unchanged: Roleplay maps its persisted `nav_open` onto `library_open` in its own adapter. The pane ids stay `"library"`/`"items"` for every destination. |
| Shell, grip, messages | `Widgets/adaptive_pane_shell.py` | `AdaptivePaneShell`, `AdaptivePaneGrip` (destination class, `painted_names`, equality-guarded `sync_label`), `AdaptivePaneClasses`, `PaneToggleRequested`, `AdaptivePaneShellResized`, `PaneVisibilityChanged`. |
| Rail-row primitives | same | `DestinationRailRow`, `fit_rail_row_label` (spec section 1.4.3 order, measured in terminal cells; the two-cell prefix counts; the count is never clipped; a loading count never paints its key), `rail_row_content` (literal `Content`, never markup), `DestinationRailRowButton` (patched through `sync_row`, never recomposed). |
| Compact search input | same | `SelectAllOnFocusingClickInput` with its "/" re-arm behind `swallow_slash_on_focus`, default `False`; the Library rail search box keeps `True`. |
| Stage return bar | not yet | `StageReturnBar` joins the module only at the post-#2862 shed, when `library_emergency_return.py` folds into it (paying back one pre-import module). |

### 2. The Library keeps every name

- The Library's messages are the same objects as the shared ones (`LibraryPaneVisibilityChanged is PaneVisibilityChanged`, `AdaptiveReaderShellResized is AdaptivePaneShellResized`). Every Library handler binds with `@on(...)`, which matches class identity; a subclass alias would silently stop matching.
- `LibraryAdaptiveReaderShell` and `LibraryAdaptiveReaderPaneGrip` are thin subclasses that only supply `LIBRARY_ADAPTIVE_READER_CLASSES` and the "Nav" painted name; they define no MRO-dispatched lifecycle handler.
- `Widgets/Library/library_rail.py` re-exports the shared input.
- Stays Library-local: the ordinary-route rail contract and its 3-cell handle; `LibraryRail` and `LibraryShellState`; all `library_screen.py` glue (emergency receipt and restore, the pane-toggle if-ladder, the static F6 tuples, the Escape chain); `[library.reader]` and its Settings rows. Copy the contracts, not the code.

### 3. CSS: destination classes, never neutral rules

- Every shell, grip and row rule is keyed to a destination's own `AdaptivePaneClasses` and lives in that destination's lazy split sheet. The build moves a rule to a lazy sheet only when every class token carries the sheet's owner prefix, so a neutral token would pin the rule to the boot bundle (50 B of headroom on 2026-10-02).
- The Library's destination classes are its existing class names, so the Library block in `css/features/_library.tcss` is unchanged and boot bytes did not move. Its one boot-resident member, the 214 B `:hover, .-active` grouped grip rule, is in the bundle because `.-active` has no split owner; it stays. Roleplay's copy (B1) must not add another `.-active` boot rule without an equal boot deletion in the same PR.
- No width rules: widths stay inline from the resolver (`# ds-runtime:`). The shared widgets declare no `DEFAULT_CSS`, `CSS` or `BUNDLED_CSS`.
- The `panes` pattern family is a widget contract with no public classes (`css/patterns.json`, `backlog/docs/component-patterns.md`, `Widgets/pattern_gallery.py`).

### 4. Shared focus contracts

- `BaseAppScreen.TAB_REGION` (default `None`) with `region_focus_next`/`region_focus_previous` and an `arrival_focus_target()` hook (default: the first focusable widget in `#screen-content`). `None` keeps Textual's stock Tab walk on every route; a destination opts in by setting a selector. Library and Console keep their own Tab bindings; Library adoption is FU-1 (TASK-@@FU1@@).
- Grip evacuation: when an open pane closes while it holds focus, focus moves to that pane's grip (`AdaptivePaneShell.sync_layout`).
- Dynamic F6 arrives with the Roleplay frame (B7, TASK-@@B7@@).

### 5. One collapse grammar, one round border per pane

On adaptive-shell routes a collapsed pane is a full-height grip and each pane has one round border (owner ruling Q1). Recorded exceptions until the Library-adoption follow-up (FU-1): the Library's ordinary routes (a text "Collapse" button, a 3-cell handle with unpersisted state — `LibraryNavigationRailHandle` in `Widgets/Library/library_rail.py` — and three solid border layers in `css/features/_library.tcss`) and the Library's outer frame.

### 6. Boundaries

- Roleplay never imports `Widgets/Library/`: that package's `__init__` eagerly imports every Library widget, and the Library's pre-import route is 176 modules / 125k LOC.
- The shared module imports nothing from `Widgets/Library/`, `Library/` or `UI/Library_Modules/`, contains no Library class literal, and is never imported by a UI-ready-resident module (never `Widgets/destination_rail.py`). Tests pin all three.
- Not authorised: Watchlists convergence, an app-wide workbench, a third shell pane.

### 7. The census rule

Each slice measures the UI-ready census, the pre-import payload and the boot CSS bytes on its paired base arm and its head in the same session, and answers any growth in ADR-097's order: defer, then shed, then an owner-signed exception row in the same commit as the constant change. No blanket re-pin. Boot CSS is net zero or lower in every PR. After each re-key of a counted broad selector, the broad-selector ratchet is lowered to the measured count. B0's own row: +1 pre-import module (`Widgets.adaptive_pane_shell`), ledgered in ADR-097.

### 8. Slice order and the fold-before-rail arithmetic

B0 and B1 may proceed in parallel, except B1's shell-rule copy and parity test, which need B0's Library destination classes. The Inspector folds into the work pane (B5b-2) before the Roleplay rail arrives (B6): adding a 28/35-column rail while the Inspector stays would leave the work pane at 34 columns at 120 (below its 40-column minimum) and 60 at 160; with the fold first the rail costs nothing (interim floors ≥58 / ≥78 / ≥108).

## Exceptions to the design language

### Grips are not slivers (G1a)

`backlog/docs/design-language.md` §2.8 says a shell sidebar collapses "to zero width (never a sliver)". On adaptive-shell routes a collapsed pane keeps a full-height **five-cell grip** at every width. The grip is a labelled, reachable control — it paints the pane's name and its arrow, takes focus, and answers Enter and Space — not a sliver of the pane. Authority: ADR-084's accepted grips ("retains a five-column grip for each pane", ADR-084 lines 15-16; "Both collapsed panes continue to consume five columns each for reachable grips", lines 73-74) and the owner's 2026-09-09 change (PR #2550, `7fc19997e3`), which amended ADR-086 to keep "the intentionally five-cell controls" while widening the columns. Not ADR-148 (Proposed). Below 64 columns a zero-width form arrives only with the single-stage follow-up (spec R38); until then the grips stay at every width.

### Adaptive rails follow the Library policy (G1b)

§2.8 sets `$ds-sidebar-min-width: 35`. An adaptive-shell rail instead follows the resolver's rail policy: `clamp((3W + 8) // 16 + 5, 29, 39)` cells with a 24-cell floor (`Utils/library_rail_width.py`, ADR-086). Roleplay **deliberately shares** the Library's policy — one house rail width, and "Nav" means the same in both destinations. This is a recorded coupling: B7's golden test pins Roleplay's resolved table, so a Library rail-policy change fails a Roleplay test and is reviewed for both destinations.

### Runtime geometry lives in the Python leaf (G15)

§2.8 promotes shared feature geometry to `_variables.tcss`. Geometry a resolver computes from the measured terminal (rail, list and work widths, grip reservations) promotes instead to the config-safe Python leaf `Utils/adaptive_reader_state.py`: a stylesheet token cannot depend on the measured shell, and the widths reach widgets as inline styles marked `# ds-runtime:`.

## Context

ADR-084 kept the first adaptive reader inside Media; ADR-086 shared it across five Library destinations but kept it Library-local and declined an application-wide implementation. The Roleplay redesign (spec, 2026-10-02) adopts the same frame. Copying the shell would duplicate the geometry, focus evacuation and persistence behaviour ADR-086 refused to clone; importing `Widgets/Library/` would pull the Library package onto Roleplay's route. Budgets at the base (`origin/dev @ fccf70d3b0`): boot CSS 608,040 / 608,090 B; broad selectors 273 / 274; UI-ready 1,033 / 1,033 modules; pre-import 557 / 557 modules.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Copy the shell into Roleplay | Two implementations of hard geometry and focus code, ADR-086's own reason to share; they would drift. |
| Roleplay imports `Widgets/Library/` | Pulls the Library package (eager `__init__`) onto Roleplay's route. |
| Put the primitives in `Widgets/destination_rail.py` | UI-ready resident: the module cost lands on every boot, with zero census headroom. |
| A new `Utils` module for the aliases | One more module in both the boot and UI-ready censuses. |
| A neutral boot CSS component for the shell (~1.1 KB) | 50 B of boot headroom; fails the byte budget (spec fresh-lens MF-01). |
| Subclass message aliases | `@on` matches class identity; a subclass would stop matching the shared shell's posts. |
| Gate Tab with `check_action` and fall through to Screen's binding | The binding merge replaces Screen's binding, so a `False` gate kills Tab. |
| Canonical neutral classes for the `panes` family | A neutral class in a rule pins it to boot; without a rule it is nominal; the gallery renders boot CSS only. |

## Consequences

- The Library's behaviour, classes, CSS bytes and Tab order are unchanged; its compatibility names become thin subclasses and aliases.
- Roleplay (B1 onward) builds its frame from the shared module with its own destination classes, its own lazy sheet (`screen_feature_roleplay.tcss`), and `TAB_REGION` (B3).
- The pre-import census carries +1 module until the post-#2862 shed pays it back.
- Rail-row fitting decisions the spec left open are fixed here: the prefix counts toward the width; the key hint is two spaces plus the letter (right alignment is the destination's styling); counts are caller-formatted; the ellipsis is `…`, measured in cells, and has no ASCII substitute (the glyph map has none, so ASCII mode paints `…` too); a style-only row change (a count that starts or stops loading with unchanged text) repaints, because `Content` equality and the `label` reactive compare plain text only.
- `LibraryBrowseReaderShell.apply_route` still relabels a grip by assigning `pane_label` and calling `sync_open` with an unchanged open state, which does not repaint the painted name; `AdaptivePaneGrip.sync_label` is the fix, adopted by the Library in FU-1 (TASK-@@FU1@@).

## Links

- [ADR-084: Library Media reader information architecture](084-library-media-reader-ia.md)
- [ADR-086: Share an adaptive reader shell within Library destinations](086-library-adaptive-reader-shell.md)
- [ADR-097: Boot budget ratchets](097-boot-budget-ratchets.md)
- [Spec: Roleplay on the Library frame (sub-project B)](../../Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md)
- [Design language](../docs/design-language.md) §2.8
- [Component patterns: Panes](../docs/component-patterns.md)
````

- [ ] **Step 3: Amend ADR-086, ADR-084 and the README (G2, G3)**

In `backlog/decisions/086-library-adaptive-reader-shell.md`:

Old:
```markdown
Status: Accepted
Date: 2026-08-24
```
New:
```markdown
Status: Accepted
Amended by: [ADR-211](211-shared-adaptive-pane-shell.md) (accepted 2026-10-02)
Date: 2026-08-24
```

In `backlog/decisions/084-library-media-reader-ia.md`:

Old:
```markdown
Status: Accepted
Date: 2026-08-23
```
New (G3: one line pointing to the new ADR plus G1c; the `@@B7@@` token is replaced in Step 5):
```markdown
Status: Accepted
Amended by: [ADR-211](211-shared-adaptive-pane-shell.md) (accepted 2026-10-02; shared grip grammar. Its G1c carve-out from lines 79-80, destination-owned non-geometry preferences outside the shared normaliser, is added by Roleplay B7, TASK-@@B7@@)
Date: 2026-08-23
```

In `backlog/decisions/README.md` (the 084 Media row was missing from the index; it is added beside the 086 row it shares an amendment with):

Old:
```markdown
| [ADR-086](086-library-adaptive-reader-shell.md) | Accepted | Share one structural adaptive reader shell inside Library while keeping Media, Conversations, Notes, Prompts, and Skills behavior destination-owned. |
```
New:
```markdown
| [ADR-084](084-library-media-reader-ia.md) | Accepted; amended by ADR-211 | Make Library Media an adaptive reader with a permanent Reader: an independently collapsible Library rail and Items list, five-column grips, and preferred-versus-responsive pane state. |
| [ADR-086](086-library-adaptive-reader-shell.md) | Accepted; amended by ADR-211 | Share one structural adaptive reader shell inside Library while keeping Media, Conversations, Notes, Prompts, and Skills behavior destination-owned. |
```

Old:
```markdown
| [ADR-210](210-console-region-ownership.md) | Accepted | Give each Console region one job — authority header, identity tab strip, one Chats browser, three-group Inspect, four-slot status strip, run-owning composer — and re-home every duplicated control. |
```
New:
```markdown
| [ADR-210](210-console-region-ownership.md) | Accepted | Give each Console region one job — authority header, identity tab strip, one Chats browser, three-group Inspect, four-slot status strip, run-owning composer — and re-home every duplicated control. |
| [ADR-211](211-shared-adaptive-pane-shell.md) | Accepted | Share the adaptive pane shell (grips, messages, rail rows, compact search input) between the Library and Roleplay through destination classes, neutral resolver aliases and an opt-in Tab region. |
```

- [ ] **Step 4: Design-language pointers (G1a, G1b, G15)**

In `backlog/docs/design-language.md`:

Old:
```markdown
- **Standard shell sidebar:** `$ds-sidebar-width: 25%`,
  `$ds-sidebar-min-width: 35`, `$ds-sidebar-max-width: 80`, docked left,
  collapsible to zero width (never a sliver).
- **Feature geometry stays local** until a second consumer appears; then it
  promotes to `_variables.tcss` (e.g. `$ds-console-composer-height`).
```
New:
```markdown
- **Standard shell sidebar:** `$ds-sidebar-width: 25%`,
  `$ds-sidebar-min-width: 35`, `$ds-sidebar-max-width: 80`, docked left,
  collapsible to zero width (never a sliver).
  **Exception (ADR-211):** on adaptive-pane-shell routes (Library adaptive
  readers, Roleplay) the navigation rail follows the resolver's 29-39 rail
  policy with a 24-cell floor, and a collapsed pane keeps a full-height
  five-cell grip — a labelled control, not a sliver — at every width until
  the single-stage follow-up lands.
- **Feature geometry stays local** until a second consumer appears; then it
  promotes to `_variables.tcss` (e.g. `$ds-console-composer-height`).
  **Runtime geometry** — widths a resolver computes from the measured
  terminal — promotes instead to the config-safe Python leaf
  `Utils/adaptive_reader_state.py` (ADR-211): a stylesheet token cannot
  depend on the measured shell.
```

- [ ] **Step 5: Substitute the task ids and verify**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
EV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - "$EV" <<'PY'
import sys
from pathlib import Path
ev = Path(sys.argv[1])
ids = dict(line.split() for line in (ev / "slice-ids.txt").read_text().splitlines() if line.strip())
ids.update(line.split() for line in (ev / "followup-ids.txt").read_text().splitlines() if line.strip())
tokens = {"@@B0@@": (ev / "b0-task-id.txt").read_text().strip(), "@@FU1@@": ids["FU-1"], "@@B7@@": ids["B7"]}
for name in ("backlog/decisions/211-shared-adaptive-pane-shell.md", "backlog/decisions/084-library-media-reader-ia.md"):
    path = Path(name)
    text = path.read_text(encoding="utf-8")
    for token, value in tokens.items():
        text = text.replace(token, value)
    path.write_text(text, encoding="utf-8")
print(tokens)
PY
grep -c "@@" backlog/decisions/211-shared-adaptive-pane-shell.md backlog/decisions/084-library-media-reader-ia.md
for KEY in B0 B7; do ls backlog/tasks | grep -F "task-$(grep "^$KEY " "$EV/slice-ids.txt" | cut -d' ' -f2) - "; done
ls backlog/tasks | grep -F "task-$(grep '^FU-1 ' "$EV/followup-ids.txt" | cut -d' ' -f2) - "
EOF
```

Expected: the token map (`{'@@B0@@': '33910.1', '@@FU1@@': '33910.19', '@@B7@@': '33910.12'}`), `…211-shared-adaptive-pane-shell.md:0` and `…084-library-media-reader-ia.md:0`, then the B0, B7 and FU-1 task file names (every cited id exists). If Task 0 Step 8 was ruled out of scope, there is no `slice-ids.txt` or `followup-ids.txt`: replace `(TASK-@@FU1@@)` and `, TASK-@@B7@@` with nothing instead, so the ADR cites the spec ids FU-1 and B7 only.

- [ ] **Step 6: Verify and commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
grep -n "Amended by: \[ADR-211\]" backlog/decisions/086-library-adaptive-reader-shell.md backlog/decisions/084-library-media-reader-ia.md
grep -c "ADR-211" backlog/decisions/README.md backlog/docs/design-language.md
grep -n "## Exceptions to the design language\|### Grips are not slivers (G1a)\|### Adaptive rails follow the Library policy (G1b)\|### Runtime geometry lives in the Python leaf (G15)\|### 7. The census rule" backlog/decisions/211-shared-adaptive-pane-shell.md
grep -rn "ADR-211\|211-shared" tldw_chatbook Tests scripts --include='*.py' --include='*.tcss' --include='*.json' | head -5
git add backlog/decisions/211-shared-adaptive-pane-shell.md backlog/decisions/086-library-adaptive-reader-shell.md backlog/decisions/084-library-media-reader-ia.md backlog/decisions/README.md backlog/docs/design-language.md
git commit -m "docs(adr): share the adaptive pane shell across destinations (B0)" -m "Amends ADR-086 and ADR-084; records the grip, rail-width and runtime-geometry exceptions and the per-slice census rule." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: two `Amended by` lines (the 084 one carrying the G1c pointer and B7's task id); `README.md:3` and `design-language.md:2`; five heading lines; NO output from the code/CSS/JSON grep (the number lives only in docs); then the commit.

---

### Task 10: Final pre-import measurement, the PR-gate census and the fast lane

**Files:**
- Modify (only if the final measurement moved): `Tests/Performance/test_screen_preimport_payload_budget.py`, `backlog/decisions/097-boot-budget-ratchets.md`; refreshed by script: `Tests/Performance/boot_budget_snapshots/preimport_payload.json`
- Modify: `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`

**Interfaces:**
- Consumes: `$EV/preimport_measure.sh`, `$EV/preimport_raise.py`, `$EV/owner-signoff.txt` (Task 0); the raise already committed with Task 3; the three censused test files from Tasks 2, 4 and 7; `scripts/update_boot_budget_snapshots.py --only preimport` (existing; it refuses an over-limit measurement unless the constant was raised first).
- Produces: nothing for later tasks.

The owner's Q10 answer was taken in Task 0 Step 6 and the raise landed with Task 3, so no commit on the branch leaves the guard red. This task re-measures both arms in one session at the final code, re-runs the idempotent raise (it rewrites the ledger row with the final numbers and the ADR's real number), refreshes the snapshot, and grows the PR-gate census by script. Run Steps 1 and 2 again after every rebase (Global Constraints, rebase protocol).

- [ ] **Step 1: Final paired measurement, raise check and snapshot refresh**

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b0-evidence
"$EV/preimport_measure.sh" base && "$EV/preimport_measure.sh" head
"$PY" "$EV/preimport_raise.py"
cd $MAIN/.worktrees/roleplay-b0
"$PY" scripts/update_boot_budget_snapshots.py --only preimport 2>&1 | tail -3
"$PY" -m pytest Tests/Performance/test_screen_preimport_payload_budget.py -p no:cacheprovider -q -s 2>&1 | grep -E "PREIMPORT PAYLOAD CENSUS|^\s+library\s+[0-9]+ mods|[0-9]+ (passed|failed)"
"$PY" -c 'import json,sys; b,h=(json.load(open(f"{sys.argv[1]}/preimport-{a}.json")) for a in ("base","head")); print("added modules:", sorted(set(h["module_set"])-set(b["module_set"])), "| removed:", sorted(set(b["module_set"])-set(h["module_set"])))' "$EV"
git status --short Tests/Performance backlog/decisions/097-boot-budget-ratchets.md
EOF
```

Expected (on `fccf70d3b0`): `base | modules 557 | LOC 412048 | library 176 mods / 125112 LOC | fattest route 125112 LOC`, `head | modules 558 | LOC 412587 | library 177 mods / 125651 LOC | fattest route 125651 LOC` (the plan's code measured in scratch; small LOC drift is expected if anything was reformatted); `raised: MAX_PASS_ADDED_MODULES 557 -> 558 | ledger row: written`; the snapshot refresh line; the census line, the `library` route row and `1 passed`; `added modules: ['tldw_chatbook.Widgets.adaptive_pane_shell'] | removed: []`; and ` M` only for `preimport_payload.json` and, if the final LOC or the ADR number changed the row's text, `097-boot-budget-ratchets.md`. If #2862 landed first, the `raised:` line names the LOC constant(s) instead. Any `STOP:` line, any other added module, or a red guard: stop and report (something new became eager, or the growth passed the bound the owner approved).

- [ ] **Step 2: Add the three fast test files to the PR-gate census (after the final rebase)**

Do this after the last rebase onto dev: dev grows the census often (TASK-33661 took it from 121 to 122 entries and raised `MINIMUM_FILES` to 121 after `fccf70d3b0`; open PR #2953 adds two more). The script reads the entries the way `read_census` and the workflow do (skipping blank and `#` lines), adds B0's three files, sorts (the file is kept sorted; census order is the CI run order), and sets `MINIMUM_FILES` to the resulting count with a B0 comment line above it. On a rebase conflict in either file, take dev's side and re-run the script; never hand-merge.

```bash
bash <<'OUTER'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'EOF'
import re
from pathlib import Path

B0_FILES = [
    "Tests/UI/test_adaptive_pane_shell.py",
    "Tests/UI/test_base_app_screen_tab_region.py",
    "Tests/UI/test_destination_rail_row.py",
]
census = Path("scripts/ui_pr_gate_census.txt")
lines = census.read_text(encoding="utf-8").splitlines()
if any(line.strip().startswith("#") for line in lines):
    raise SystemExit("census has comment lines now: insert the three paths by hand at their sorted places")
entries = sorted({line.strip() for line in lines if line.strip()} | set(B0_FILES))
census.write_text("\n".join(entries) + "\n", encoding="utf-8")
n = len(entries)

checker = Path("scripts/check_ui_pr_gate_census.py")
text = checker.read_text(encoding="utf-8")
text = re.sub(r"^# Roleplay frame B0 raised it to \d+:.*\n(?:# .*\n)*?(?=MINIMUM_FILES = )", "", text, flags=re.M)
comment = (
    f"# Roleplay frame B0 raised it to {n}: test_adaptive_pane_shell.py,\n"
    "# test_destination_rail_row.py and test_base_app_screen_tab_region.py gate the\n"
    "# shared pane shell, rail-row fitting and the behaviour-neutral Tab region.\n"
    "# The mounted every-route Tab test (test_base_app_screen_tab_region_routes.py,\n"
    "# ~65 s) stays out of the fast lane; the B0 gate run executes it on both arms.\n"
)
text, count = re.subn(r"^MINIMUM_FILES = \d+$", comment + f"MINIMUM_FILES = {n}", text, count=1, flags=re.M)
if count != 1:
    raise SystemExit("no `MINIMUM_FILES = <int>` line found")
checker.write_text(text, encoding="utf-8")
print("census entries:", n)
EOF
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_ui_pr_gate_census.py; echo "exit=$?"
git diff --stat scripts/
OUTER
```

Expected: `census entries: <dev's count + 3>` (124 on `fccf70d3b0`, 125 on `185c845fe8`), the checker's success line, `exit=0`, and a diff stat touching only the two `scripts/` files (3 insertions in the census, the comment and the constant in the checker).

- [ ] **Step 3: Run the censused files the way the UI Fast Lane does, and time them**

The fast lane installs only `-e . pytest pytest-asyncio pytest-timeout packaging` (`.github/workflows/derived-artifacts.yml`, "the census is verified against this dependency set") and runs the census serially within a 20-minute job (spec §5.4 item 3). The static audit in `test_base_app_screen_tab_region.py` skips a route whose module fails to import and requires at least 15 screen classes; check that number under the minimal dependency set, not only under the full dev `.venv`.

```bash
bash <<'EOF'
set -euo pipefail
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; WT=$MAIN/.worktrees/roleplay-b0; EV=$MAIN/.worktrees/.b0-evidence
FL=$EV/fastlane-venv
[ -x "$FL/bin/python" ] || uv venv --python 3.12 "$FL" > "$EV/fastlane-venv.log" 2>&1
(cd "$WT" && VIRTUAL_ENV="$FL" uv pip install -e . pytest pytest-asyncio pytest-timeout packaging) > "$EV/fastlane-install.log" 2>&1
cd "$WT"
"$FL/bin/python" -c 'import sys; sys.path.insert(0, "."); from Tests.UI.test_base_app_screen_tab_region import _production_screen_classes as f; print("fast-lane screen classes:", len(f()))' 2>/dev/null | tail -1 | tee -a "$EV/gates.txt"
{ /usr/bin/time -p "$FL/bin/python" -m pytest Tests/UI/test_adaptive_pane_shell.py Tests/UI/test_base_app_screen_tab_region.py Tests/UI/test_destination_rail_row.py -p no:cacheprovider -q --timeout=180 2>&1 | tail -1; } 2>&1 | tee -a "$EV/gates.txt"
RUN=$(gh run list --workflow derived-artifacts.yml --event pull_request --status success --limit 1 --json databaseId -q '.[0].databaseId')
gh run view "$RUN" --json jobs -q '.jobs[] | select(.name == "UI Fast Lane") | "last green UI Fast Lane: \(.startedAt) -> \(.completedAt)"' | tee -a "$EV/gates.txt"
EOF
```

Expected: `fast-lane screen classes: <n>` with n ≥ 15 (if it is lower, lower the test's floor to the measured minimal-environment count with a comment naming this measurement, and re-run); `392 passed` (23 + 22 + 347; read the nonzero count, it must equal the three files' `--collect-only` total) followed by `real <seconds>`; and the start and end of the last green UI Fast Lane run (the job runs only on pull requests). The lane's last duration plus these seconds must stay under 20 minutes; if it does not, stop and report (spec §5.4 item 3). The editable install writes only git-ignored build metadata (`*.egg-info/`, `build/`).

- [ ] **Step 4: Commit**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
git status --short
git add Tests/Performance/test_screen_preimport_payload_budget.py backlog/decisions/097-boot-budget-ratchets.md Tests/Performance/boot_budget_snapshots/preimport_payload.json scripts/ui_pr_gate_census.txt scripts/check_ui_pr_gate_census.py
git commit -m "perf: final pre-import measurement and PR-gate census for the shared pane shell (B0)" -m "Snapshot refreshed at the final code; the owner-approved raise and its ADR-097 row landed with the Library adoption commit and are re-measured here. Adds B0's three fast UI tests to the PR-gate census, floor = entry count." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: `git status --short` lists only those paths as modified (no untracked file); then the commit line.

---

### Task 11: Final gates, paired arms, the live check, owner screenshots, and the PR body

**Files:**
- Create: `Docs/superpowers/plans/2026-10-02-roleplay-b0-captures/` (36 captures: `.txt` and `.ansi`, base and head, Notes list / Notes editor / focused Nav grip, 120x36 / 160x45 / 220x55)
- Modify: `backlog/tasks/task-33910.1 - Roleplay-frame-B0-shared-pane-shell-primitives-and-the-frame-ADR-no-visible-change.md` (ACs, Implementation Notes, status)
- Create (evidence only): `$EV/gates.txt`, `$EV/pr-body.md`, `$EV/screens/*.png` (18), `$EV/owner-screenshots.txt`, `$EV/harness-state/` (B0-private harness masters and runs)

**Interfaces:**
- Consumes: everything above.
- Produces: the PR body at `$EV/pr-body.md` for the controller.

This task records evidence; every gate below must hold before the B0 task is marked Done. "No new failures" is judged by `paired.sh` (failure-set diff, with `recovery=` in single digits on both arms), never by a raw pass count.

- [ ] **Step 1: Confirm the base arm is still the merge-base**

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook
git -C "$MAIN" fetch origin --prune --quiet
echo "merge-base: $(git -C $MAIN/.worktrees/roleplay-b0 merge-base HEAD origin/dev)"
echo "base arm:   $(git -C $MAIN/.worktrees/roleplay-b0-base rev-parse HEAD)"
EOF
```

Expected: the two SHAs are equal. If the branch was rebased, follow the rebase protocol (Global Constraints): re-anchor the base arm with `git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0-base checkout --quiet --detach "$(git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0 merge-base HEAD origin/dev)"`, write the new SHA to `$EV/base-sha.txt`, run `build_css.py` and `check_bundle_sync.py` (nothing of B0's should change), redo Task 10 Steps 1-2 (re-measure both arms, re-raise, refresh the snapshot, re-run the census script) and commit their output, then continue with Step 2 here. Never hand-merge a generated sheet, a snapshot, the pre-import constants or B0's ledger row: take dev's side and let the scripts rewrite them.

- [ ] **Step 2: Census and byte gates on both arms, same session**

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b0-evidence
NODE="Tests/Performance/test_textual_css_fastpath.py::test_ancestor_scoped_bare_type_rule_count_is_a_ratchet"
for ARM in base head; do
  if [ "$ARM" = base ]; then T=$MAIN/.worktrees/roleplay-b0-base; else T=$MAIN/.worktrees/roleplay-b0; fi
  P="$EV/profile-gates-$ARM"; rm -rf "$P"; mkdir -p "$P/home" "$P/config" "$P/data"
  cd "$T"
  BOOT=$(HOME="$P/home" XDG_CONFIG_HOME="$P/config" XDG_DATA_HOME="$P/data" TLDW_CONFIG_PATH="$P/config/config.toml" PYTHONPATH="$T" "$PY" -c 'from Tests.Performance.test_boot_css_byte_budget import _boot_parsed_css_census as c; print(sum(c().values()))' 2>/dev/null | tail -1)
  BROAD=$(env TLDW_TEST_PRIVATE_PROFILE_NODE="$NODE" TLDW_TEST_CONFIG_ROOT="$P" HOME="$P/home" XDG_CONFIG_HOME="$P/config" XDG_DATA_HOME="$P/data" TLDW_CONFIG_PATH="$P/config/config.toml" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest "$NODE" -p pytest_asyncio.plugin -p pytest_timeout -p no:cacheprovider -s -q 2>&1 | grep -o '\[census\] total=[0-9]*' | tail -1)
  UIREADY=$("$PY" -m pytest Tests/Performance/test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size -p no:cacheprovider -q -s 2>&1 | grep -o 'ui-ready-census: [0-9]*/[0-9]*' | tail -1)
  WEIGHT=$("$PY" -m pytest Tests/Performance/test_app_import_weight.py::test_app_import_own_module_count_stays_at_the_post_diet_size -p no:cacheprovider -q -s 2>&1 | grep -o 'boot-import-weight: [0-9]*/[0-9]*' | tail -1)
  "$PY" -m pytest Tests/Performance/test_screen_preimport_payload_budget.py -p no:cacheprovider -q -s > "$EV/gates-preimport-$ARM.out" 2>&1
  PRE=$(grep -o 'PREIMPORT PAYLOAD CENSUS: [0-9]* modules / [0-9]* LOC' "$EV/gates-preimport-$ARM.out" | tail -1)
  LIBROUTE=$(grep -E '^\s+library\s+[0-9]+ mods' "$EV/gates-preimport-$ARM.out" | tail -1 | tr -s ' ' | sed 's/^ //')
  PREGUARD=$(grep -oE '[0-9]+ (passed|failed)' "$EV/gates-preimport-$ARM.out" | tail -1)
  CLOSURE=$("$PY" -m pytest Tests/Packaging/test_config_import_closure.py -p no:cacheprovider -q 2>&1 | grep -oE '[0-9]+ (passed|failed)' | tail -1)
  echo "$ARM | boot-css=$BOOT | $BROAD | $UIREADY | $WEIGHT | $PRE | route: $LIBROUTE | preimport guard: $PREGUARD | config-closure: $CLOSURE" | tee -a "$EV/gates.txt"
done
EOF
```

Expected (numbers at `fccf70d3b0`; the gate is the base-to-head delta):

```
base | boot-css=608040 | [census] total=273 | ui-ready-census: 1033/1033 | boot-import-weight: 681/686 | PREIMPORT PAYLOAD CENSUS: 557 modules / 412048 LOC | route: library 176 mods 125112 LOC | preimport guard: 1 passed | config-closure: 1 passed
head | boot-css=608040 | [census] total=273 | ui-ready-census: 1033/1033 | boot-import-weight: 681/686 | PREIMPORT PAYLOAD CENSUS: 558 modules / 412587 LOC | route: library 177 mods 125651 LOC | preimport guard: 1 passed | config-closure: 1 passed
```

Any delta other than the pre-import +1 module (and its LOC, within the +1,000 bound the owner approved) is a FAIL: stop and find the cause. The `route:` field is the Library route's own LOC, the "largest route" row of the spec §5.10 ledger.

- [ ] **Step 3: Module-size ratchet exemption**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
PYTHONPATH=$PWD /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
import subprocess
from Tests.Architecture.test_module_size_ratchet import _BUDGETS
changed = set(subprocess.run(["git", "diff", "--name-only", "origin/dev...HEAD"], capture_output=True, text=True, check=True).stdout.split())
print(sorted(changed & set(_BUDGETS)) or "B0 touches no governed module")
PY
EOF
```

Expected: `B0 touches no governed module` (spec section 5.7.4: B0 is exempt). The ratchet's failure set is compared in Step 4's `css` run.

- [ ] **Step 4: Paired arms — every suite group**

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; EV=$MAIN/.worktrees/.b0-evidence
"$EV/paired.sh" library "$EV/suites-library.txt" -n 4 | tee -a "$EV/gates.txt"
"$EV/paired.sh" focus "$EV/suites-focus.txt" -n 4 | tee -a "$EV/gates.txt"
"$EV/paired.sh" css "$EV/suites-css.txt" -n 4 | tee -a "$EV/gates.txt"
PAIRED_PLUGIN=0 "$EV/paired.sh" perf "$EV/suites-perf.txt" | tee -a "$EV/gates.txt"
# The tier-2 every-route Tab test is arm-agnostic: run the head's file on both
# arms (copied into the base arm as an untracked file, then removed).
cp "$MAIN/.worktrees/roleplay-b0/Tests/UI/test_base_app_screen_tab_region_routes.py" "$MAIN/.worktrees/roleplay-b0-base/Tests/UI/"
printf '%s\n' Tests/UI/test_base_app_screen_tab_region_routes.py > "$EV/suites-routes.txt"
"$EV/paired.sh" routes "$EV/suites-routes.txt" -n 4 | tee -a "$EV/gates.txt"
rm "$MAIN/.worktrees/roleplay-b0-base/Tests/UI/test_base_app_screen_tab_region_routes.py"
git -C "$MAIN/.worktrees/roleplay-b0-base" status --short | head -3
EOF
```

Expected: for each of the five groups (`library`, `focus`, `css`, `perf`, `routes`), `recovery=` in single digits on both arms (the perf group runs with `plugin=off` and has none either) and nothing between `new failures on head [<group>] (must be empty):` and `(end of new failures [<group>])`; the `routes` group passes on both arms; and an empty `status` for the base arm afterwards. (Expect a long run: `test_library_shell.py` alone is 843 tests; both arms run.)

Flaky-test rule: re-run each new failure three times on EACH arm (from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0-base` and from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0`: `PYTHONPATH=$EV/plug /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <file> -p b0_bootstrap_all -k <test name> -p no:cacheprovider -q`). If it passes all three on head, it is a load flake under `-n`: record its name and error text in `$EV/gates.txt` and in the PR body (plan review measured two such in the focus group, each 3/3 green in isolation on both arms). If it fails on head in a re-run while base passes all three, three runs are not enough to call it: run ten more on each arm. It is a regression only if base passes all ten and head fails again: fix it before continuing. If base fails at least once in those ten, it is a pre-existing flake: record both arms' failure counts. (Plan review measured `test_watchlists_content_pane.py::test_snapshot_modal_renders_remote_markup_as_literal_text` failing 5/10 on base and 6/10 on head in isolation, after a 3/3 base run that looked clean.)

Then the head-only suites (the tier-2 routes file already ran on both arms above):

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py Tests/UI/test_destination_rail_row.py Tests/UI/test_base_app_screen_tab_region.py Tests/Library/test_library_adaptive_reader_state.py -p no:cacheprovider -q -n 4 2>&1 | tail -1 | tee -a /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence/gates.txt
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_adaptive_pane_shell.py Tests/UI/test_destination_rail_row.py Tests/UI/test_base_app_screen_tab_region.py Tests/Library/test_library_adaptive_reader_state.py --collect-only -q 2>&1 | tail -1
EOF
```

Expected: `N passed` with N equal to the collected count on the second line (770 on `fccf70d3b0`: 23 + 347 + 22 + 378), and no failures.

- [ ] **Step 5: Preflight and lint**

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python ./scripts/preflight.sh 2>&1 | tail -15 | tee -a /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence/gates.txt
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check $(git diff --name-only origin/dev...HEAD -- '*.py')
EOF
```

Expected: every preflight check passes (bundle sync, profile-owned path census, production diagnostic inventory, backlog task ids, readable task files, chachanotes allowlist, index plan pins, textual worker contract, timestamp writers, the gated Tests/UI census). If the Canvas Mermaid asset step cannot download, set `TLDW_CANVAS_MERMAID_INPUT_DIR` to an existing offline cache and re-run. Ruff: `All checks passed!`

- [ ] **Step 6: Live check — Library Notes at 120x36, 160x45 and 220x55, base vs head, fresh launches, text and colour**

The `.txt` captures hold plain text only. The grips get `h-full`, `p-0` and `border-none` from Python, so a CSS re-key that left them unstyled would not move a single character: only the `.ansi` captures (SGR colour, bold and reverse codes) can show it. Both are compared, a third capture per size holds the Nav grip focused (the lazy `:focus` rule, `bold reverse`), and 220x55 is added for spec §5.4 item 6's screenshot round. The harness state is B0-private (`$EV/harness-state`), so no other session (B1 may run in parallel) can rebuild or overwrite the masters mid-capture (harness README, "Traps").

Get the harness (it lives on PR #2957's branch until that merges; if #2957 is on dev and this branch is rebased, use `H=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0/Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/harness` instead):

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook
[ -d $MAIN/.worktrees/rp-harness-pr2957 ] || git -C $MAIN worktree add --detach $MAIN/.worktrees/rp-harness-pr2957 origin/backlog/roleplay-fix-first-tasks
ls $MAIN/.worktrees/rp-harness-pr2957/Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/harness/launch.sh
EOF
```

Self-test, then build B0-private masters with the base arm's code:

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; EV=$MAIN/.worktrees/.b0-evidence
H=$MAIN/.worktrees/rp-harness-pr2957/Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/harness
export HARNESS_STATE=$EV/harness-state
APP_WT=$MAIN/.worktrees/roleplay-b0 "$H/launch.sh" --self-test 2>&1 | tail -3
APP_WT=$MAIN/.worktrees/roleplay-b0-base "$H/make_profiles.sh" 2>&1 | tail -3
EOF
```

Expected: `SELF-TEST: PASS`, then `make_profiles.sh` finishing with the `golden` master built under `$EV/harness-state` (if it reports the masters already exist, they are B0's own from an earlier run: keep them). A self-test FAIL caused by another session using the real profile is spurious (README "Traps"): read the diff before blaming the harness.

Capture both arms at three sizes, each on a fresh launch, with PNGs:

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b0-evidence
H=$MAIN/.worktrees/rp-harness-pr2957/Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/harness
export HARNESS_STATE=$EV/harness-state PNG=1
for ARM in base head; do
  if [ "$ARM" = base ]; then export APP_WT=$MAIN/.worktrees/roleplay-b0-base; else export APP_WT=$MAIN/.worktrees/roleplay-b0; fi
  export CAPTURES="$HARNESS_STATE/captures/b0-$ARM"
  for SIZE in 120x36 160x45 220x55; do
    COLS=${SIZE%x*}; ROWS=${SIZE#*x}; S="b0${ARM}${COLS}"
    "$H/launch.sh" "$S" "$COLS" "$ROWS" golden
    "$PY" "$H/waitfor.py" "$S" "4 Roleplay" 240 && "$PY" "$H/waitfor.py" "$S" "Ctrl+Q" 30
    "$H/drive_library.sh" "$S" "$SIZE" full
    # Back to Notes, then click the Nav grip's arrow (leftmost match): the click
    # focuses the grip, so its lazy :focus rule paints. Same gesture on both arms.
    ( . "$H/drive_lib.sh" "$S" "$SIZE"
      has "Notes (5)" || { clk "--->" L; sleep 2; }
      clk "Notes (5)" L; sleep 3
      if has "<---"; then clk "<---" L; else clk "--->" L; fi; sleep 2
      shot library-notes-navgrip-focus )
    tmux -L "$S" send-keys C-q; sleep 3; tmux -L "$S" kill-server 2>/dev/null || true
  done
done
ls "$HARNESS_STATE/captures/b0-base" "$HARNESS_STATE/captures/b0-head" | grep -cE '^library-notes-(list|editor|navgrip-focus)-[0-9]+x[0-9]+\.(txt|ansi|png)$'
EOF
```

Expected: `54` (2 arms x 3 sizes x 3 captures x `.txt`/`.ansi`/`.png`). A `!! no match:` line from the driver means a step missed its anchor, and a `png failed` line means playwright could not render (`$MAIN/.venv/bin/python -m playwright install chromium`): relaunch and re-run that arm and size (runs are cheap).

Compare text AND colour:

```bash
bash <<'EOF'
C=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence/harness-state/captures
for SIZE in 120x36 160x45 220x55; do
  for NAME in library-notes-list library-notes-editor library-notes-navgrip-focus; do
    for EXT in txt ansi; do
      if cmp -s "$C/b0-base/$NAME-$SIZE.$EXT" "$C/b0-head/$NAME-$SIZE.$EXT"; then
        echo "identical $NAME-$SIZE.$EXT"
      else
        echo "DIFFERS $NAME-$SIZE.$EXT"; diff "$C/b0-base/$NAME-$SIZE.$EXT" "$C/b0-head/$NAME-$SIZE.$EXT" | head -20
      fi
    done
  done
done
EOF
```

Expected: 18 `identical` lines. If a pair differs, every differing line must be explained in the PR body. Any difference in a grip column is a FAIL, in the `.txt` (the painted `Nav`/`Notes`/`Items` letters, `<---`, `--->`, `‹`, `›`) or in the `.ansi` (any SGR code on those cells: colour, bold, reverse); stop and debug. Only clock-dependent text (relative ages) and the cursor-blink state may differ.

Commit the text and colour captures beside this plan (spec section 5.4 item 6); the PNGs go to the owner (Step 7), not into the tree:

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; EV=$MAIN/.worktrees/.b0-evidence; C=$EV/harness-state/captures
D=$MAIN/.worktrees/roleplay-b0/Docs/superpowers/plans/2026-10-02-roleplay-b0-captures
mkdir -p "$D" "$EV/screens"
for ARM in base head; do for SIZE in 120x36 160x45 220x55; do for NAME in library-notes-list library-notes-editor library-notes-navgrip-focus; do
  cp "$C/b0-$ARM/$NAME-$SIZE.txt" "$D/$ARM-$NAME-$SIZE.txt"
  cp "$C/b0-$ARM/$NAME-$SIZE.ansi" "$D/$ARM-$NAME-$SIZE.ansi"
  cp "$C/b0-$ARM/$NAME-$SIZE.png" "$EV/screens/$ARM-$NAME-$SIZE.png"
done; done; done
echo "captures: $(ls "$D" | wc -l | tr -d ' ') pngs: $(ls "$EV/screens" | wc -l | tr -d ' ')"
EOF
```

Expected: `captures: 36 pngs: 18`.

- [ ] **Step 7: STOP — owner screenshot approval (spec §5.4 item 6, ADR-007)**

B0 is not one of the slices without a screenshot round (only B5b-1 and B9a are). Report to the controller with the nine base/head PNG pairs in `$EV/screens/` (`base-<name>-<size>.png` beside `head-<name>-<size>.png`) and Step 6's 18 `identical` lines, and wait. Write the owner's reply, verbatim, to `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence/owner-screenshots.txt`. If the owner granted a waiver for this no-visible-change slice in answer to Task 0 Step 6's message, that reply is already the verbatim record: still send the PNGs for information, and append the date they were sent. A subagent's or the controller's own approval is not owner approval, and nothing is waived silently. AC #5 is ticked only once this file holds an approval (or the recorded waiver).

- [ ] **Step 8: Re-verify the ids at merge time (tasks and ADR), and renumber if needed**

```bash
bash <<'EOF'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; EV=$MAIN/.worktrees/.b0-evidence
git -C "$MAIN" fetch origin --prune --quiet
OURS=$( { awk '{print $2}' "$EV/slice-ids.txt" "$EV/followup-ids.txt" 2>/dev/null; cut -d. -f1 "$EV/b0-task-id.txt"; } | sort -u)
OTHER_REFS=$(git -C "$MAIN" for-each-ref --format='%(refname)' refs/remotes/origin refs/heads | grep -v 'roleplay-b0-shared-frame-primitives$')
OTHER_WTS=$(git -C "$MAIN" worktree list --porcelain | sed -n 's/^worktree //p' | grep -vE '/roleplay-b0(-base)?$')
{
  for ref in $OTHER_REFS; do git -C "$MAIN" ls-tree -r --name-only "$ref" -- backlog/tasks backlog/drafts backlog/completed backlog/archive 2>/dev/null || true; done
  echo "$OTHER_WTS" | while read -r wt; do [ -n "$wt" ] && ls "$wt/backlog/tasks" "$wt/backlog/drafts" "$wt/backlog/completed" 2>/dev/null; done
} | sed -nE 's#^"?(.*/)?task-([0-9]+(\.[0-9]+)*) .*#\2#p' | sort -u > "$EV/other-task-ids.txt"
{
  for ref in $OTHER_REFS; do git -C "$MAIN" ls-tree --name-only "$ref" backlog/decisions/ 2>/dev/null; done
  echo "$OTHER_WTS" | while read -r wt; do [ -n "$wt" ] && ls "$wt/backlog/decisions" 2>/dev/null; done
} | sed -nE 's#^(.*/)?([0-9]{3})-.*#\2#p' | sort -un > "$EV/other-adr-numbers.txt"
(cd "$MAIN/.worktrees/roleplay-b0" && "$MAIN/.venv/bin/python" scripts/check_backlog_task_ids.py | tail -1)  # the 33910 ids are on dev via PR #2960; a duplicate here is a rebase/renumber issue per lessons-backlog-hygiene
grep -qx 211 "$EV/other-adr-numbers.txt" && echo "ADR COLLISION: 211 (highest elsewhere: $(tail -1 "$EV/other-adr-numbers.txt"))"
echo "id check done"
EOF
```

Expected: `No duplicate task IDs across …` and `id check done`. The B0 programme's ids (TASK-33910 and `.1`-`.28`) are already on dev via PR #2960, so a reported duplicate comes from dev moving underneath this branch: rebase first and re-run; renumber only the side that has not merged, per `backlog/docs/lessons-backlog-hygiene.md`. An `ADR COLLISION` means renumber the ADR with this script (NEW = the next free number above the highest elsewhere). It is scoped to the files this branch changed (never a repo-wide grep: dev's own files may cite a different ADR with that number), and it excludes this plan, whose scripts quote the old number; code and CSS carry no ADR number, so only the branch's docs, the B0 task and the ADR-097 ledger row change:

```bash
bash <<'EOF'
NEW=212   # replace with the next free three-digit number
PLAN=Docs/superpowers/plans/2026-10-02-roleplay-b0-shared-frame-primitives.md
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
git diff --name-only -z origin/dev...HEAD -- . ":(exclude)$PLAN" | xargs -0 grep -lE --null 'ADR[- ]211|211-shared-adaptive-pane-shell' | xargs -0 sed -i '' -E "s/ADR([- ])211/ADR\1${NEW}/g; s/211-shared-adaptive-pane-shell/${NEW}-shared-adaptive-pane-shell/g"
git mv backlog/decisions/211-shared-adaptive-pane-shell.md "backlog/decisions/${NEW}-shared-adaptive-pane-shell.md"
printf '\n> ADR-211 in this plan was renumbered to ADR-%s at merge (Task 11 Step 8).\n' "$NEW" >> "$PLAN"
git diff --name-only -z origin/dev...HEAD -- . ":(exclude)$PLAN" | xargs -0 grep -nE 'ADR[- ]211|211-shared' || echo "no stale number left"
EOF
```

Then move the ADR's row in `backlog/decisions/README.md` to its numeric position, run `./scripts/preflight.sh`, `git add -u` and commit (`docs: renumber the shared pane-shell ADR to <NEW> (B0)`, with the `Co-Authored-By` line). B0 touches no `.tcss`, so nothing needs a CSS rebuild; the next `preimport_raise.py` run reads the number from the renamed file.

- [ ] **Step 9: Write the PR body**

```bash
bash <<'OUTER'
MAIN=/Users/macbook-dev/Documents/GitHub/tldw_chatbook; PY=$MAIN/.venv/bin/python; EV=$MAIN/.worktrees/.b0-evidence
ID=$(cat "$EV/b0-task-id.txt")
ADR=$(ls "$MAIN/.worktrees/roleplay-b0/backlog/decisions" | sed -nE 's/^([0-9]{3})-shared-adaptive-pane-shell\.md$/ADR-\1/p')
PRE_ROWS=$("$PY" - "$EV" <<'PY'
import json
import sys

ev = sys.argv[1]
base, head = (json.load(open(f"{ev}/preimport-{arm}.json")) for arm in ("base", "head"))
lib_b, lib_h = base["routes"]["library"], head["routes"]["library"]
print(f"| Pre-import pass modules | {base['modules']} | {head['modules']} | {head['modules'] - base['modules']:+d} `Widgets.adaptive_pane_shell`; owner-signed ADR-097 row, raised in the same commit as the Library adoption (Task 3); paid back by the post-#2862 shed |")
print(f"| Pre-import pass LOC | {base['loc']:,} | {head['loc']:,} | {head['loc'] - base['loc']:+,} (raised only if it exceeded its limit; see the ledger row) |")
print(f"| Largest route (library) | {lib_b[0]} mods / {lib_b[1]:,} LOC | {lib_h[0]} mods / {lib_h[1]:,} LOC | {lib_h[1] - lib_b[1]:+,} LOC (spec forecast about +0.5k) |")
PY
)
{
cat <<EOF
## Roleplay frame B0: shared frame primitives and the ADR (no visible change)

TASK-${ID} · ${ADR} · spec \`Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md\` §5.2 B0 · plan \`Docs/superpowers/plans/2026-10-02-roleplay-b0-shared-frame-primitives.md\`

### What changed
- \`Widgets/adaptive_pane_shell.py\` (new): the Library's adaptive shell, grip and messages moved here with destination classes; \`fit_rail_row_label\` + \`DestinationRailRow\` + \`DestinationRailRowButton\`; the promoted \`SelectAllOnFocusingClickInput\` (\`swallow_slash_on_focus=False\` by default).
- Library compatibility: \`library_adaptive_reader_shell.py\` is thin subclasses + same-object aliases; \`library_rail.py\` re-exports the input. Library class names, CSS rules and boot bytes are unchanged; no \`.tcss\` file changes.
- \`Utils/adaptive_reader_state.py\`: same-object neutral aliases and read-only \`nav_open\`/\`nav_width\`, pinned by a golden digest captured before the edit.
- \`BaseAppScreen\`: opt-in \`TAB_REGION\` (None everywhere) and \`arrival_focus_target()\`; Tab and F1 unchanged.
- \`panes\` widget-contract family; ${ADR} amending ADR-086 and ADR-084; design-language §2.8 pointers.
- Backlog: the Roleplay frame parent, subtasks B0-B12 and the spec §5.13 follow-ups (FU-1..FU-5 and five named ones).

### Census measurements (paired arms, same session; raw output of the gate script)
\`\`\`
EOF
grep -E '^(base|head) \|' "$EV/gates.txt" | tail -2
cat <<EOF
\`\`\`

### Budget ledger (spec §5.10)
| Guard | Base | Head | Note |
|---|---|---|---|
| Boot parsed CSS (\`_boot_parsed_css_census()\`) | 608,040 B | 608,040 B | net 0 B: no CSS rule added; ceiling 608,090 unchanged |
| Broad selectors | 273 | 273 | ratchet 274 unchanged |
| UI-ready modules | 1,033 | 1,033 | +0 (the spec's "1,032" is stale) |
| Boot import weight | 681 | 681 | +0 |
EOF
echo "$PRE_ROWS"
cat <<EOF
| PS lines (module-size ratchet) | n/a | n/a | B0 touches no governed module (exempt, §5.7.4) |
| CSS sources after the tour | +0 | +0 | no new sheet |

### Paired base arm (failure-set diff, "no new failures")
Each arm ran with the collection-time plugin \`b0_bootstrap_all\` (imports \`tldw_chatbook.app\` in \`pytest_collection_modifyitems\` and adds \`bootstrap_profile\` to every item; lessons-testing-evidence, "A local red wall of RecoveryRequired hides the guard you meant to run"), except the perf group, which runs as CI runs it. \`recovery=\` is each arm's \`RecoveryRequired\` count.
EOF
grep -E '^[a-z]+ (base|head):|^new failures|^\(end' "$EV/gates.txt"
cat <<EOF

### Named mutations (each red, then restored green)
EOF
cat "$EV/mutations.md"
cat <<EOF

### Live check (harness from PR #2957, \`launch.sh --self-test\` PASS, B0-private HARNESS_STATE)
Library Notes list, Notes editor and the focused Nav grip at 120x36, 160x45 and 220x55, fresh launch per arm: base and head \`.txt\` (text) and \`.ansi\` (SGR colour) captures identical (\`Docs/superpowers/plans/2026-10-02-roleplay-b0-captures/\`). No visit-order claim (§2.12 item 2). Owner screenshot approval (ADR-007), verbatim:
> $(tr '\n' ' ' < "$EV/owner-screenshots.txt")

### Not in B0
No guide delta (no behaviour change; no "Verified against" stamps). No ADR-011 capture (B0 retires no legacy path and adds no worker or timer). \`StageReturnBar\` waits for the post-#2862 shed. The browse shell's relabel-without-repaint is recorded in ${ADR} for FU-1.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
} > "$EV/pr-body.md"
wc -l "$EV/pr-body.md"
OUTER
```

Open `$EV/pr-body.md` and check every number in the ledger table against the census block above it (the block is the measurement). The pre-import rows are generated from the measured JSON; where a static row (boot CSS, broad selectors, UI-ready, boot import weight) differs from the census block — for example because dev moved after `fccf70d3b0` — replace the table value with the measured one; if a delta other than the pre-import growth appears, the gate failed and Step 2 must be revisited. Add any load flake recorded in Step 4 and any route the tier-2 test could not probe. Hand the file to the controller; do not open the PR from this task.

- [ ] **Step 10: Close out the B0 task file and commit**

Edit `backlog/tasks/task-33910.1 - Roleplay-frame-B0-shared-pane-shell-primitives-and-the-frame-ADR-no-visible-change.md` directly (five-digit ids break `backlog task edit`): tick `#1`–`#10` (`- [ ]` → `- [x]`) only for the criteria the evidence above proves (`#5` needs `$EV/owner-screenshots.txt`; `#9` is this plan; `#10` is the Step 2 gate block); set `status: Done` only if all ten are ticked; and add, after the plan section (use the ADR's final number):

```markdown
## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Summary
The Library's adaptive reader shell is a shared destination primitive (ADR-211) with no visible change.

## Changes
- New tldw_chatbook/Widgets/adaptive_pane_shell.py: shell, grip (destination class, painted_names, sync_label), messages, rail-row fitting and button, promoted select-all input.
- Library compatibility: thin subclasses and same-object aliases; library_rail.py re-exports the input. No stylesheet changes.
- Neutral resolver aliases pinned by a golden digest captured before the edit.
- BaseAppScreen TAB_REGION (None on every route) and arrival_focus_target() (a target outside the region is ignored); Screen's copy binding re-spread.
- panes widget-contract family; ADR-211 amending ADR-086 and ADR-084; design-language section 2.8 pointers.
- Pre-import census: the owner-signed raise (only the constants the head exceeded) and its ADR-097 row landed in the same commit as the Library adoption, so no commit left the guard red; three fast UI files added to the PR-gate census (floor = entry count).
- Backlog: parent, subtasks B0-B12 and the ten spec 5.13 follow-ups filed (ACs copied from the slice cards).

## Evidence
- Paired base arm (merge-base recorded in the PR body), each arm with the collection-time bootstrap plugin: no new failures in the library, focus, css, perf and routes groups; RecoveryRequired counts in single digits.
- Census: boot CSS, broad selectors, UI-ready and boot import weight unchanged; pre-import +1 module only, LOC within the approved bound.
- Live check (verified <date> against <head sha> and base <base sha>): Library Notes list, editor and focused Nav grip at 120x36, 160x45 and 220x55, fresh launches, B0-private harness state; base and head .txt and .ansi captures identical (Docs/superpowers/plans/2026-10-02-roleplay-b0-captures/). Owner approved the PNG pairs on <date>.
- Fast lane: the three censused files pass under the minimal fast-lane dependency set; serial time and screen-class count in the PR body.
- Named mutations: see the table in the PR body (golden hysteresis, sync_label refresh, evacuation, resident import, ignored painted names, library class literal, subclass alias, naming-convention handler, fallback order, cell measurement, markup label, refit on resize, w-full, style-only compare, grip-class rename, neutral token, other-owner prefix, shared slash default, confine-none, drop-copy, priority Tab, arrival outside region, ds-runtime marker).

## Decisions
- Library destination classes are the existing Library class names (zero CSS diff, zero boot bytes).
- panes is a widget-contract family (no public classes); the gallery registration is a docstring entry because the gallery renders boot CSS only.
- Rail-row fitting: the prefix counts toward the width; the key hint is two spaces plus the letter; counts are caller-formatted; the ellipsis is measured in cells and has no ASCII substitute; style-only changes repaint.
- The ADR number lives only in docs and the ledger row, so a renumber at merge is one scripted pass.
<!-- SECTION:NOTES:END -->
```

```bash
bash <<'EOF'
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-b0
ID=$(cat /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/.b0-evidence/b0-task-id.txt)
backlog task list --plain 2>/dev/null | grep -F "TASK-${ID} "
git status --short
git add "backlog/tasks/task-${ID} - Roleplay-frame-B0-shared-pane-shell-primitives-and-the-frame-ADR-no-visible-change.md" Docs/superpowers/plans/2026-10-02-roleplay-b0-captures
git commit -m "chore(backlog): B0 evidence, live-check captures and implementation notes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
EOF
```

Expected: the B0 task listed with its new status; `git status --short` shows only the B0 task file (` M`) and the captures directory (`??`) before the commit (delete a stray `backlog/tasks/task-task- - .md` if one appears; the plan was committed in Task 0).

---

## Self-Review

Performed after writing, against the spec's B0 card and the sections the orchestrator named, and re-run after the plan-review round of 2026-10-02 (35 findings applied; see "Plan review" at the end).

**1. Spec coverage — every B0 card bullet:**

| B0 card item | Task |
|---|---|
| Scope: the new ADR (G1) | 9 (G1a, G1b, G15 inside it; G2/G3 header + README, the ADR-084 line carrying the G1c pointer; G16 in 8) |
| Scope: `Widgets/adaptive_pane_shell.py` (R17) | 2 (shell, grip, messages), 4 (rail rows), 5 (input) |
| Scope: Library compatibility, subclass + same-object aliases, `library_rail.py` re-export, no `StageReturnBar` | 3, 5 (no `StageReturnBar` anywhere) |
| Scope: neutral aliases in ARS | 1 |
| Scope: `TAB_REGION` + `arrival_focus_target()` (opt-in, default `None`) | 7 (an arrival target outside the region is ignored) |
| Scope: no boot CSS; destination class; Library block stays lazy, re-keyed | 6 (re-key = Library names declared as destination classes; pins re-keyed to the constant; no `.tcss` file changes) |
| Scope: `panes` family in registry + catalog + gallery | 8 |
| Never touches PS/LS/library_shell_state/screen_helpers/library_emergency_return | Global Constraints; no task edits them; Task 11 Step 3 lists changed governed files |
| Acceptance: golden outputs identical over a normalised-preference grid | 1 (digest captured on unmodified code; both names) |
| Acceptance: `LibraryPaneVisibilityChanged is PaneVisibilityChanged` | 3 |
| Acceptance: Library pure and shell suites, no new failures vs paired base arm | 3 (quick), 5 (input + Search/RAG panel suites), 6 (css quick), 11 Step 4 (full `library` group), all with the bootstrap plugin and `recovery=` gate |
| Acceptance: Tab unchanged on every `TAB_REGION is None` route | 7 (tier 1 vs in-test before arm, static audit, tier 2 every route against the stock walk in the same mount), 11 Step 4 (tier 2 on both arms) |
| Acceptance: live Library Notes 120x36 / 160x45 fresh launch = base | 11 Step 6 (`.txt` and `.ansi`, plus a focused Nav grip capture and 220x55), Step 7 (owner screenshots) |
| Tests: `test_adaptive_pane_shell.py`, `test_destination_rail_row.py` (24/31/35, count never clipped, §1.4.3 order), alias identity + golden in `test_library_adaptive_reader_state.py`, `test_base_app_screen_tab_region.py` | 2-8, 4, 1, 7 |
| Gates: boot CSS ≤ 0, broad 273, module-size exempt, UI-ready, pre-import +1 measured → ledger, config closure | 0 Step 6 (question), 3 Step 6 (raise in the same commit), 4-5 (re-check), 10 (final), 11 Steps 2-3 |
| DoD §5.4: backlog task (the §5.13 item 2 tasks were filed with the spec, PR #2960, TASK-33910.1-.28), mutations + paired arms, preflight + PR-gate census (+ fast-lane time), budget ledger, captures + owner screenshots, docs, lessons | 0, every task, 10-11; ADR-011 and guide delta not applicable (stated in the PR body); no lesson unless an incident occurs |

**2. Placeholder scan.** Searched the plan for "TBD", "TODO", "implement later", "similar to Task", "add appropriate", "write tests for": none in any step. The only values an executor supplies are measured or external facts with exact capture commands: the owner's verbatim replies (Task 0 Step 6, Task 11 Step 7), `NEW` in the renumber script (only on an ADR collision, Task 11 Step 8), the live-check date and SHAs in the Implementation Notes, and any static PR-body ledger value that the census block in Task 11 Step 9 shows differently (the pre-import rows are generated from the measured JSON). The `@@B0@@`, `@@FU1@@` and `@@B7@@` tokens are substituted from the id files and checked by `grep -c "@@"` in Task 9 Step 5.

**3. Type and name consistency.** `AdaptivePaneClasses(shell, nav, items, work, grip)` is used identically in Tasks 2, 3, 6 and 8 (the catalog example now uses R18 split prefixes); `grip_type`, `destination`, `painted_names`, `pane_open`, `sync_label` match between the shared module (2), the Library subclasses (3) and the tests; `fit_rail_row_label(row, width, *, current)`, `FittedRailRowLabel.plain`, `rail_row_content`, `DestinationRailRowButton.rail_row/is_current/sync_row/_refit` match between Task 4's code and tests; `swallow_slash_on_focus` keeps the Library default `True` and the shared default `False` in Task 5's code, tests and ADR; `TAB_REGION`, `region_focus_next/previous`, `arrival_focus_target`, `_move_region_focus` (with `selector_set`) match between Task 7's code, both test files, the mutations and the ADR; the renamed `test_no_naming_convention_handler_exists_for_the_aliased_messages` is the name used in Review Focus 1 and Task 3. `SHARED_WIDGET_NAMES` grows in Tasks 4 and 5 only after the classes exist. The helper scripts agree on file names: `preimport_measure.sh` writes `preimport-{base,head}.json`, which `preimport_raise.py`, Task 10 Step 1 and the PR body read; Task 0 Step 5 writes `slice-ids.txt` and `followup-ids.txt` from the filed TASK-33910 ids, which Task 9 Step 5 and Task 11 Step 8 read.

**4. Review Focus placement.** Each of the five lines names its pinning tests, and each test is written in full in its owning task: identity, handler binding and the naming-convention guard (Task 3), lazy focus rule and boot-bundle pin (Task 6) plus the live `.txt`/`.ansi` check (Task 11), Tab equivalence and F1 rows (Task 7), wide/zero-width fitting and the never-clipped count (Task 4), golden digest and alias identity (Task 1).

**Verification done while planning (on `origin/dev @ fccf70d3b0`, no repo edits).** First round: the golden test code passes in 0.76 s with digest `7234b56a…` and a hysteresis mutation changes it; a compact `w-full` button's content width equals its rail width (35/31/24); the resident-module import probe prints `False`. (That round's claim that a scratch prototype passed every shell test was wrong for one test: its `"library-" not in source` check could never pass against the planned `-library-grip` ids. The check is now an AST scan of string constants.) Plan-review round, on a `git archive` export of `fccf70d3b0` in the session scratchpad with every file assembled mechanically from this plan's own Old/New, create and append blocks (all 44 Old strings matched exactly once):
- ruff passes on all 14 touched files;
- each new Tests/UI file passes run alone: 23 (shared shell), 347 (rail rows), 22 (Tab region); the state file plus the config closure give 379; Task 6 Step 2's subset gives 7 and Task 5's slash subset 5;
- all named mutations listed in Tasks 2-8 that touch the new code went red as stated and green on restore (14 of them run as a batch, including `markup-label`, `plain-only-compare`, `no-refit-on-resize`, `drop-w-full`, `arrival-outside-region`, `drop-copy` = 2 failed, `library-class-literal`, `ignore-painted-names`, `naming-convention-handler`, `other-owner-prefix`, `drop-ds-runtime`, `subclass-alias` = 2 failed);
- the tier-2 route test gave 21 passed on both arms, three runs each, after restricting compared steps to those starting in `#screen-content` (its first version was red intermittently on BOTH arms at the nav bar's overflow); `dead-tab` turns 20 of 21 red;
- the measure and raise scripts: base 557 modules / 412,048 LOC, head 558 / 412,587 (library 176 → 177 modules, +539 LOC); `raised: MAX_PASS_ADDED_MODULES 557 -> 558`, idempotent on a second run, guard green; a simulated #2862-first base (limits 500 / 378,740) raised only `MAX_PASS_ADDED_LOC`; the over-bound and unexpected-module cases print `STOP:` and exit 1;
- the PR-gate census script gives 124 entries and a green checker, idempotent;
- the programme tasks are filed separately (TASK-33910 and `.1`-`.28`, PR #2960); Task 0 Step 5 only records their ids and starts B0;
- a paired run of six touched suites with the bootstrap plugin: `recovery=0` on both arms; the one head-only failure was a pre-existing flake (5/10 failing on base in isolation), which is why Task 11 Step 4's flake rule extends to ten runs; the MRO-convention file's one failure is pre-existing on both arms;
- the ratchet offender extraction gives the same four offenders on both arms.
Not run while planning: the paired full suite groups, the live harness captures and PNGs, the minimal fast-lane venv (needs a network install).

**Spec items not mapped to code in B0 (by design):** each later slice's own plan (§5.13 item 1). Dynamic F6 is recorded in the ADR only (B7). The browse shell's relabel-without-repaint is recorded, not fixed (it would change Library behaviour; FU-1 adopts `sync_label`). B0's screenshot round now covers 120x36, 160x45 and 220x55 with owner approval (Task 11 Steps 6-7), although nothing visible changes.

**Plan review (2026-10-02), applied.** The `library-` substring blocker; collection-time app imports in the three new Tests/UI files; the bootstrap plugin and `recovery=` gate for every paired run; the markup test that now forces real assignments; the `drop-copy` expectation (kept at 2 by pinning `BaseAppScreen.BINDINGS`); the Task 2/Task 5 expected outputs; committing this plan; the strengthened flake rule; the style-only repaint and the ellipsis wording; `.ansi` comparison, a focused-grip capture, a private harness state, PNGs at three sizes and the owner's screenshot stop; filing B1-B12 and the follow-ups; the merge-order-aware, stateless raise script; the raise moved into Task 3's commit, with the question asked in Task 0; the scripted PR-gate census after the final rebase; the arm-agnostic tier-2 test; dropping the stylesheet comment; keeping the ADR number out of code with a scoped renumber script; the two-direction naming-convention guard; the owner-specific split-prefix pin; the region-checked arrival target; the minimal fast-lane run with timing; the three stale comments; the G3 line with G1c; the catalog example's R18 prefixes; the extra named mutations; the six Search/RAG suites; the permanent Python-style pin; the Library-route LOC in the ledger and the pre-import guard in the perf group. None was skipped.

> ADR-211 in this plan was renumbered to ADR-212 at merge (Task 11 Step 8).
