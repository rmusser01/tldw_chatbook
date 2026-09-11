# Handoff Formatter Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Remove only the pure Handoff formatter body from LibraryScreen without changing its output or callers.

**Architecture:** The existing screen_helpers module owns the unchanged body. The Screen retains a normal instance-method wrapper; existing canonical constants and eligibility policy remain their owners.

**Tech Stack:** Python 3.12, pytest, Textual, stdlib AST/source comparison.

ADR required: no. ADR path: N/A. Reason: mechanical support-layer move under DESIGN.md section7 and the existing Library decomposition doctrine.

Spec: `Docs/superpowers/specs/2026-09-10-pr2427-handoff-formatter-extraction-design.md`.
Worktree: `.worktrees/pr2427-review-recovery`; baseline production/test bytes at `02597164ff` (local docs head `318acc9ad3`). Preserve the unrelated September8 reader-paydown plan.

## Task 1: guarded pure move

Files: modify only `Tests/Architecture/test_library_support_layer_surface.py`,
`tldw_chatbook/UI/Library_Modules/screen_helpers.py`, and
`tldw_chatbook/UI/Screens/library_screen.py`.

- [x] Add a location/delegation test to the existing support-surface file, not a new framework. Import modules inside the test; use getattr(..., None) so the baseline fails an explicit callable assertion. Verify the helper's `__module__` and `__globals__` point to screen_helpers and the Screen re-export is the same callable. Patch only the Screen module's helper alias with a one-call recorder. Invoke `LibraryScreen._workspace_handoff_summary_label(None, state=sentinel_state)`; assert the exact returned sentinel and `calls == [sentinel_state]`. Parameterize positional and keyword state invocation if needed to pin both existing signatures.
- [x] Run the new test before modifying production; record the expected missing-helper assertion failure:
  `.venv/bin/python -I -m pytest Tests/Architecture/test_library_support_layer_surface.py -k handoff --basetemp <fresh-TMPDIR>/pytest -q`.
- [x] Add `_workspace_handoff_summary_label(state: LibraryWorkspaceDepthState) -> str` to screen_helpers. Copy the entire existing method body, including comments/docstring, dedenting exactly four spaces. Add stdlib `re`, TYPE_CHECKING, the type-only canonical state import, and canonical prefix/eligibility imports. Do not add a new module or dependencies.
- [x] Import that helper into library_screen's existing support import list. Replace only the original method body with `return _workspace_handoff_summary_label(state)`. Keep its original signature, four production call sites, constant re-exports and other eligibility imports unchanged. Remove the now-unused Screen `import re`; make the prefix constant's existing compatibility re-export explicit with a same-name alias. This avoids new F401 findings without removing the promised constant surface.
- [x] Run the new guard and complete support file. Compare dedented body bytes and AST to `git show 02597164ff:tldw_chatbook/UI/Screens/library_screen.py`; assert zero other Screen AST changes after replacing the wrapper node and removing the one new import alias from the comparison. Verify original tests are unchanged apart from the added guard. Record exact line reduction and full production diff.

## Task 2: qualification and checkpoint

Files: record evidence in the spec/plan, TASK31932 notes and `backlog/docs/pr-2427-rebase-reconciliation.md`. Do not change any existing budget or fixture.

- [x] Freeze production/test sources for verification. Use a fresh per-user TMPDIR basetemp on each pytest command and `.venv/bin/python -I`; no full sweep.
- [x] Run full files under the existing observation-only native runner (`/private/tmp/pr2427-fd-identity.OTL9up/native_fd_identity.py`, inspect before use):
  `Tests/UI/test_library_crit9_rail.py`,
  `Tests/Widgets/Library/test_library_rail_workspace_action_sync.py`,
  `Tests/Widgets/Library/test_library_rail.py`,
  `Tests/UI/test_post_release_workspaces_library_depth.py`,
  `Tests/UI/test_library_entry_compose_once.py`.
  Use `env -u NO_COLOR TERM=xterm-256color COLORTERM=truecolor`; record terminal result and final unique native SQLite/lock attribution, not cumulative additions. If the temporary observer is unavailable, restore an equivalent observation-only instrument before making cleanup claims.
- [x] Run complete support-layer, Library packaging preimport and screen-preimport payload tests. Prior review baseline:9 support/packaging passed; preload failed504/500 twice. Compare current results without masking/raising the inherited budget.
- [x] Run complete Screen and Library-module size/private-owner guards. Baseline126 passed/13 size failures; preserve all caps and report remaining failures explicitly. Expected LibraryScreen reduction approximately60 lines, not a green size gate.
- [x] Run scoped Ruff and whitespace checks. Do not autoformat unrelated code or change the moved body's bytes. Run `PYTHON=.venv/bin/python bash scripts/preflight.sh` using an existing verified Mermaid input cache if needed; all seven derived checks must pass.
- [x] Obtain sequential independent spec-compliance and code-quality reviews. Address actual regressions within this move; separately record unrelated baseline failures without fixing them in this commit.
- [ ] Commit exact owned paths and save to existing PR using a normal push if remote still equals02597164ff. If concurrent remote changes exist, stop rather than overwrite. Do not rebase during frozen verification; any later dev update requires its own qualification.
- [ ] Update existing weekly automation with the new checkpoint and blockers. Do not merge while size, resource, preload or final-head review/check gates remain open.

## Evidence

Implementation eca8c083e7 is reviewed and qualified:153 native UI passed with zero SQLite/lock retention; support10, selectedHandoff6 and private87 passed. Guard group50 passed/14 existing failures (13 size, preload504/500); all7preflight pass. Screen32724->32664 (-60),975 above unchanged cap. No new lint findings. Exact terminal logs and parity evidence are in the reconciliation report. Publication/automation checkpoint pending.

Known baseline logs:
`/private/tmp/pr2427-formatter-review-tests.log` (9 passed,1 preload failure),
`/private/tmp/pr2427-formatter-preload-repeat.log` (same504/500),
`/private/tmp/pr2427-formatter-review-output.log` (6 selected Handoff tests passed).
