---
id: TASK-14920
title: Repair 20 stale-contract failures in the console and personas UI suites
status: Done
assignee: []
created_date: '2026-08-11 02:00'
updated_date: '2026-08-11 06:18'
labels:
  - tests
  - console
  - personas
  - dev-baseline
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by task-14912's sweep, which ran every affected `Tests/UI` file WHOLE for the first time (its AC#4 exists because a file that has ever contained a hang has an unknown pass count). Two files carry failures nobody had counted:

- `Tests/UI/test_console_native_chat_flow.py` — **291 passed / 18 failed**
- `Tests/UI/test_personas_workbench.py` — **310 passed / 2 failed**

**These are NOT caused by the bounding work and were NOT hidden by a hang.** Every one was reproduced against the pristine `eb9708cc4` copy of each file (`git show HEAD:<path>` into a temp file, run, delete), producing identical failure sets. They are **stale-contract breakage from the screen-decomposition programme**: the tests still call seams that moved.

**CORRECTED INVENTORY (coordinator, measured per-test on `b4c5105ed` — the original description in this file repeated an unverified claim and was wrong).** The dominant shape is NOT the moved seam. Measured:

- **11 x `textual.pilot.OutOfBounds: Target offset is outside of currently-visible screen region`** — every `test_console_browser_selecting_*` and `test_console_workspace_conversation_*`. This is the click-addresses-screen-coordinates trap this repo has hit before (a Start button below the fold at 170x48 needed `scroll_visible()` first): `pilot.click` takes SCREEN coordinates, so a target pushed below the fold by a layout change fails here rather than reporting the layout change. Whether the layout growth is legitimate or a regression is the thing to determine.
- **3 x `assert None is not None` on `_CharacterHandoffStore.identity_at_append`** — the character-handoff seam no longer records an identity at append time.
- **1 x `assert [] == ['Hello User, I am Elara.']`** — the unscoped character session seeds no greeting.
- **2 x moved seam** (the only ones of this shape): `'ChatScreen' object has no attribute '_save_console_message_as_media'` and `'…_ensure_active_console_session_settings'`.

Plus `test_personas_workbench.py` 310 passed / 2 failed.

This matters beyond the count. The screen-decomposition programme's own lesson is that *extraction cannot outrun growth* and that a one-way ratchet is what makes a gain stick — but a ratchet measures size, not whether the suites that pin the extracted behaviour still run. Twenty failing tests in the console's main flow suite are twenty behaviours nobody is actually checking, and the longer they sit the more they read as background noise (the "a suite that no gate runs can rot invisibly" lesson, one directory over).

Each failure needs triage before repair: a moved seam is a test fix, but `identity_at_append is None` and an empty greeting list may be real product regressions. Do not mass-rewrite to green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the 20 failures is classified as stale-contract (test fix) or real regression (product fix), with the evidence that decided it — not repaired wholesale to green
- [x] #2 Any classified as a real regression is fixed in the product, or filed separately with its reproduction if it needs owner judgement
- [x] #3 Both files run WHOLE with a READ nonzero pass count and zero failures (one test is `xfail(strict=True)` -> task-15120: repairing its click exposed a store-vs-service workspace divergence needing a product ruling; strict means a fix flips it loudly)
- [x] #4 If the moved-seam shape (`_ensure_active_console_session_settings` and friends) recurs across other suites, the sweep that finds them is checkable rather than asserted
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both files WHOLE on b4c5105ed and measure the exact failure set per test.
2. Classify each failure by bisecting with git archive into a scratch tree outside the repo (run the failing test at the suspected commit and its parent).
3. Cluster A (12 x pilot OutOfBounds): decide product-vs-test by proving whether the row is reachable at all (scroll_visible + real pilot.click) before touching any test.
4. Cluster B (4 console + 2 personas character/greeting): find the commit that changed the seam, read its intent, decide product-vs-test.
5. Cluster C (2 moved seams): confirm the behaviour still exists at its new home; repoint the call, never delete the assertion.
6. Repair, preserving each test's original claim; mutation-check every repaired test.
7. Add a checkable AST sweep for the moved-seam shape across Tests/UI (AC#4).
8. Re-run both files WHOLE plus the keep-green suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired all 20 dev-baseline failures after classifying each one by bisecting with `git archive <sha> | tar -x` into a scratch tree outside the repo (run against the same venv; the extracted tree's own `tldw_chatbook` is what imports, verified). Baseline re-measured twice as **18 failed / 291 passed** (console) and **2 failed / 310 passed** (personas), byte-identical failure sets on a pristine `git archive HEAD` tree and on the worktree — the corrected inventory in the description was right.

**Cluster A — 12 x `pilot.OutOfBounds` (test fix).** Every `test_console_browser_selecting_*` (6) and `test_console_workspace_conversation_*` (6) died in `_click_console_workspace_conversation_for_id` / `_for_row_key`. Bisected to `41d36c04c` "feat(console): split context rail sections" (TASK-14810, 2026-08-10): all 12 pass at its parent `d2d303d69` and fail at it. The change is legitimate and the rows are still reachable — probed at 160x48, `#console-left-rail-body` is a `VerticalScroll` with virtual height 99 against a 29-row viewport, the Conversations section body starts at y=45, and the target row sits at y=70; `row.scroll_visible(animate=False, force=True)` moves it to y=37, after which a real `pilot.click` returns True, activates the session and switches the workspace. So the helpers now scroll before clicking (the same thing a user does), keeping the hit-tested click rather than degrading to `Button.press()`. Mutation: disabling the row press handler in `chat_screen.py` fails 14 of these tests, so the repaired click still detects a dead row.

**Cluster B — 6 x character handoff (test fix; stale double).** 4 console (`identity_at_append is None`, `store.messages == []`) + 2 personas (`assert [] == ['Hello User, I am Elara.']`). Bisected to `a6cc05d8b` "feat: seed dynamic character chat templates" (2026-08-08), which moved the greeting seam from `store.append_message(...)` to `store.seed_character_roleplay(...)` and did not touch either suite. Both suites drove the handoff through hand-rolled store stubs, and the handoff wraps its seed call in `except Exception`, so the stubs' missing method surfaced as a swallowed `AttributeError` — the tests quietly started asserting the ABSENCE of the greeting. Both doubles now subclass the real `ConsoleChatStore` (persistence `None`) and override `create_session`/`append_message` to observe and delegate, so the greeting text comes from production's own template expansion. Mutations: `greeting_template=""` fails all 6; `global_default="Zed"` turns the content assertions red.

**Cluster C — 2 x moved seam (product fix + test fix).** `ChatScreen._save_console_message_as_media` and `._ensure_active_console_session_settings` moved to `ConsoleMessageController` (wave 3, `391b7bf69`) and `ConsoleSessionController` (wave 2, `4de93c10d`). Both tests were ADDED LATER, by `7dbbc401b` (TASK-2154 FB-07, 2026-08-07) — they were **born red and merged**, verified by running them at that very commit in an archived tree. Repointing them at the controllers made `test_console_save_as_savers_confirm_at_success_severity` fail on exactly the thing it was written to pin: FB-07 moved Note/Media/Prompt to `severity="success"` and **missed Chatbook**, which was still confirming at `"information"`. Fixed in the product (`Console_Modules/message.py`); the never-green test had been masking it for four days.

**AC#4 sweep.** `Tests/UI/test_console_moved_seam_guard.py` derives the moved-seam set from the live classes (private callables defined on a Console controller that `ChatScreen` no longer exposes — 83 of them today) and AST-scans every `Tests/**/test_*.py` for `<name>.<seam>(...)` calls off a non-controller base. It is proven to discriminate: it reports the exact `console._save_console_message_as_media()` line when that fix is reverted, stays silent on `controller.<seam>()`, `ConsoleSessionController.<seam>`, and `screen._session.<seam>()`, and refuses to run vacuously (asserts a non-empty seam inventory and >100 modules parsed). Zero hits repo-wide after the repairs.

**Two intermittents surfaced, not caused here.** `test_console_conversation_browser_starred_section_updates_from_row_action` and `test_console_workspace_conversation_search_shows_local_rows_before_slow_persisted_search` failed once in a whole-file run under concurrent load while passing 8/8 in isolation; neither touches any code this task changed, and neither appears in either 18-failure baseline. Both were the family this file already documents at `_wait_for_browser_conversation_row` (TASK-1900: "each pause overruns AND the render takes longer, so the budget shrinks exactly when it needs to grow"): a single fixed `pilot.pause` before a render assertion. They now use the same wall-clock deadline via a new `_wait_for_browser_render`, with the original assertions left verbatim after the wait, plus an explicit `not release.is_set()` so the "before the slow persisted search" window is still asserted rather than assumed. Discrimination-checked (the waits fail, with context, on a condition that never holds).

**Also worth knowing:** one whole-file run early in the session reported 49 failures instead of 18. It was an environment glitch, not the suite — the same window in which the shell lost `getcwd()` and every repo read returned `EPERM`. Two subsequent whole runs (worktree and pristine archive) both produced exactly the same 18. Separately, these UI tests DO reach a live `127.0.0.1:8080` `/v1/models` endpoint when one is listening on the machine.

Modified: `Tests/UI/test_console_native_chat_flow.py`, `Tests/UI/test_personas_workbench.py`, `tldw_chatbook/UI/Console_Modules/message.py`. Added: `Tests/UI/test_console_moved_seam_guard.py`, two entries in `backlog/docs/lessons-testing-evidence.md`.

A third pre-existing intermittent surfaced the same way: `test_save_image_button_reflects_the_real_screen_ephemeral_accessor` raised `OutOfBounds` on `pilot.click(f"#console-message-{id}")` in roughly two whole-file runs in five (it is in the failure list of an early unmodified run too). It already called `scroll_visible()` + a fixed `pilot.pause(0.2)`; a later transcript recompose moved the target again. It now goes through a new `_click_after_scrolling_into_view`, which re-scrolls and re-clicks until the click is delivered. That helper deliberately does NOT require `pilot.click` to return `True` — a container whose top-left cell belongs to an inline image child returns `False` while still delivering the event, and requiring `True` made the test fail 3/3 (caught before shipping).

**Final counts (read, solo).** `Tests/UI/test_console_native_chat_flow.py`: **309 passed, 0 failed** twice consecutively. `Tests/UI/test_personas_workbench.py`: **312 passed, 0 failed**. Keep-green: `Tests/UI/test_screen_navigation.py` 126 passed; `Tests/UI/test_background_signal_bounds.py` + the new guard 13 passed. Ruff clean on every touched file.

**Not decided / for the owner.** TASK-14810's rail split is correct and its rows are reachable, but with all three sections open by default the Conversations section starts ~20 rows BELOW the fold at 160x48 (rail virtual height 99 vs a 29-row viewport) — the shipped rail tests assert order and independent collapse, never on-screen reachability. That is a discoverability question, not a regression, and is left for the owner rather than filed under a guessed task id.
<!-- SECTION:NOTES:END -->
