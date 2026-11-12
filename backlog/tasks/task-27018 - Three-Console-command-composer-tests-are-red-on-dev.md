---
id: TASK-27018
title: Three Console command-composer tests are red on dev
status: Done
assignee:
  - '[rmusser01]'
created_date: '2026-09-01 19:04'
labels:
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deterministic on dev at 3f30fb686 (whole file, -p no:randomly, 3 failed / 100 passed, identical on pristine dev and feature branches): test_raw_cli_collapsed_state_retains_danger_label_and_one_row_geometry, test_console_unknown_command_second_unmodified_enter_sends_as_text, test_console_collapsed_paste_starting_with_slash_sends_normally. Recorded on TASK-25715's ledger as finding 6; this task gives the trio an owner. Bisect not yet run -- start there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All three tests pass, or each is individually re-decided against current composer behaviour
- [x] #2 The cause is bisected to a commit before any fix is written
<!-- AC:END -->

## Implementation Plan

1. Verify the premise at branch base `ecc0a531c8`: locate all three tests
   (the danger-label test's def line wraps, so name-grep misses it), run each,
   and classify per the session taxonomy (behavioral vs config-admission mask
   vs hook-consent gate).
2. Bisect each test's ORIGINAL failure (as filed at `3f30fb686`) from its
   birth commit: the two Enter/paste tests were added by `d0cc91fad3`
   (2026-07-12), the raw-CLI danger-label test by `0f67f3b952` (2026-08-27).
3. Under the mask bypass (scratch bootstrap_profile plugin), determine whether
   each behavioral regression is still live or was fixed upstream; probe the
   real submit_draft call shape and gateway delivery for the two send tests.
4. Fix per classification; targeted runs; A/B baseline via
   `git checkout HEAD --` swaps; close out with bisect commits, commands,
   results, ADR check.

## Implementation Notes

**All three named tests now pass. Each was individually re-decided against
current composer behaviour; all three failures were test-side drift, not live
product defects. Bisects were completed before any edit was written (AC#2).**

- **Bisects (fresh 3.12 venv, `-p no:randomly`, per-test `git bisect run`):**
  - `test_console_unknown_command_second_unmodified_enter_sends_as_text` and
    `test_console_collapsed_paste_starting_with_slash_sends_normally`: born
    green at `d0cc91fad3` (2026-07-12); **first bad `a26cdafd80` (2026-08-22
    22:28, "fix(console): resume Library-gated sends")** for BOTH. The filing-
    time signature (no "accepted" text, "Failed Response failed") was the
    durable-turn persistence refusal era, later repaired for these harnesses
    by TASK-21590's `_build_console_send_test_app` attaching an in-memory DB.
  - `test_raw_cli_collapsed_state_retains_danger_label_and_one_row_geometry`:
    born green at `0f67f3b952` (2026-08-27); **first bad `b62407e258`
    (2026-08-31 23:14, PR #2281 / TASK-25812 CSS split)** — the danger color
    rendered unstyled (white). The token was then redefined by task-31264
    (`1a1b5c19e0`, 2026-09-04): `$ds-status-error-readable` went from the
    dark-canvas-only literal `#ff8fa3` to theme-generated `$text-error`
    (resolves `#D17E92` under textual-dark). A theme-aware fix existed on
    `codex/dev-test-review-20260904` (`d407ef77b5`, 2026-09-05) but that
    branch never merged to dev.
- **Current-red classification and fixes:**
  - Danger-label test: stale color literal. Fix: derive the pin from the
    harness theme (`host.get_css_variables()["text-error"]`), parametrized
    over `textual-dark` and `textual-light` — a port of the unmerged
    `d407ef77b5` approach. Distinctness-from-ordinary, collapsed-retention,
    class and one-row-geometry assertions all unchanged.
  - Second-Enter and collapsed-paste tests: behavior verified INTACT by probe
    (submit_draft awaited exactly once with the literal/pasted text and the
    active session id; the capturing gateway receives the row ~1 s after the
    "accepted" marker under load). Three stacked test-side faults fixed:
    (1) config-admission mask — the mounted factory app reloads real config
    through the guarded loader and fails closed with
    `RecoveryRequired("raw_source_selection_changed")`; enrolled per-node with
    `@pytest.mark.bootstrap_profile` (TASK-32873 pattern, ADR-126 machinery);
    (2) stale exact-signature pin — the Library-gated seam (`a26cdafd80`)
    now calls submit_draft with custody kwargs (origin, configuration,
    staged-evidence, custody hooks), so `assert_awaited_once_with(draft,
    session_id=...)` can never match; now asserts await_count == 1, the
    positional draft arg, and the session_id kwarg;
    (3) an async race — reading `gateway.sent_messages` immediately after the
    "accepted" transcript marker; a bounded `_wait_for_gateway_row` poll
    helper (~5 s, same shape as the file's other poll helpers) replaces it.
- **Evidence (targeted runs, worktree `.worktrees/test-console-reds`):**
  - Base trio: 3 failed (two at the mask, one `Color(209,126,146) !=
    Color(255,143,163)`).
  - Head trio: `4 passed` (danger-label x 2 themes + the two send tests),
    also `4 passed` under default (random) ordering.
  - A/B via `git checkout HEAD -- Tests/UI/test_console_command_composer.py`:
    base file state reproduces `3 failed`.
  - Whole-file name-diff: base `52 failed / 51 passed` -> head
    `49 failed / 55 passed`; the FAILED-list diff shows exactly the three
    named tests leaving and nothing else moving (no new failures, no flapping
    membership).
  - Ruff findings identical base vs head on both files touched by this task
    (28 pre-existing findings each side; zero introduced).
- **Left for the mass owner (out of scope here):** the file's remaining 49
  reds are the config-admission mask on the mounted tests (same enrollment
  class as the two enrolled here) — one owner decision could enroll the
  mounted subset wholesale. The never-merged
  `codex/dev-test-review-20260904` branch (d407ef77b5, 7a6fff605f,
  b7490d4aa9, 4011d596d8) is a broader composer-suite repair worth mining
  before anyone re-does that work.
- **Modified files:** `Tests/UI/test_console_command_composer.py`, this task
  file.
- ADR required: no — test-only changes; enrollment via the existing
  bootstrap_profile mechanism (TASK-32873 pattern); no architectural
  decision made.
