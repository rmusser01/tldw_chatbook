---
id: TASK-15512
title: Six Settings, Console and Library tests are red on dev
status: To Do
assignee: []
labels:
  - test-health
  - settings
priority: medium
---

## Description

Six tests fail on `origin/dev` with no local changes. They were found while
baselining task-15270 (running the same modules against the branch and against
dev to tell genuinely-new failures from inherited ones), and they are not caused
by that branch. Filed so they are not silently re-attributed to whatever change
happens to run next to them.

Measured on `origin/dev` at `d85e6cff1`:

- `test_console_workbench_contract.py::test_console_registers_footer_workbench_shortcuts`
  -- footer hint text drift: expected `... nter send | Ctrl+K ...`, actual
  `... nter send / queue | Ctrl+K ...`. A `/ queue` affordance was added to the
  composer hint without updating the contract test.
- `test_settings_configuration_hub.py::test_settings_ownership_records_cover_categories_and_runtime_boundaries`
  -- ownership records differ by one entry:
  `model_capabilities.models.<model>.context_window` is present where the
  expected tuple does not carry it.
- `test_settings_configuration_hub.py::test_settings_console_behavior_saves_display_name_exactly`
  -- times out waiting for the toast `Console behavior settings saved.`
- `test_settings_configuration_hub.py::test_settings_provider_category_saves_provider_defaults_without_sampling`
  -- saved-settings list comes back empty where
  `('chat_defaults', 'provider', 'llama_cpp')` and `('chat_defaults', 'model', 'qwen')`
  were expected.
- `test_settings_configuration_hub.py::test_settings_provider_switch_does_not_save_stale_endpoint`
  -- same shape: empty saved list where a `chat_defaults` provider entry was expected.

- `test_library_shell.py::test_library_shell_rail_search_submit_runs_search_canvas_query`
  -- **corrected**: an earlier note here said the search scope had widened.
  That was misread from a truncated pytest diff. The actual delta is `top_k`:
  the test pinned `5`, which is `LIBRARY_RAG_FALLBACK_TOP_K`, the value used
  only when NO RAG profile resolves. The Library canvas has no depth control,
  so production resolves the active profile's `default_top_k` (TASK-15020/B3)
  and falls back to 5 only when unresolvable. The literal was pinning the
  fallback and broke the moment a profile became resolvable. Stale contract.

## Triage outcome (2026-08-11)

The save cluster was two defects stacked, and the top one hid the bottom one.

**Correction to this task's original framing.** It said the save cluster was
"consistent with the Settings save path not completing, which would be
user-visible". The first half is right; the severity was wrong. The crash below
is fatal only under pytest -- in production stdlib logging swallows it -- so no
user ever lost a save to it.

**1. A malformed log call crashed the save worker (fixed here).**
`settings_screen.py:18095` wrote a stdlib-logging call in loguru's style --
`logger.warning("...(screen_type={}, ...)", a, b, c)`. stdlib formats with `%`,
so the three arguments are never consumed and `record.getMessage()` raises
`TypeError: not all arguments converted during string formatting`. Production
swallows this (prints "--- Logging error ---", carries on, message lost), but
`_pytest.logging.LogCaptureHandler.handleError` deliberately re-raises so bad
log calls fail tests -- which killed the Textual worker mid-save and made three
tests report a timeout instead of their real assertion. 19 such calls existed
across `app.py` and `settings_screen.py`; all are converted to `%s` and
`Tests/test_stdlib_logging_format_style.py` guards the class.

That alone fixes `test_settings_console_behavior_saves_display_name_exactly`.

**2. Underneath it, a real product bug — filed as task-15740 (high).**
With the crash gone, two tests fail on their actual assertion: the save persists
nothing. Measured cause: pressing Save adds `credential_env_var`, `endpoint` and
`model_context_window` to the draft's `dirty_keys` as empty edits, for fields the
user never touched (`['model','provider']` before the click, five keys after).
The empty `model_context_window` then trips its positive-integer guard and
`return`s before anything is written. Same shape as task-15673 -- the app's own
repopulation is indistinguishable from a user edit.

So those two are **not** stale contracts; they assert the save persists and it
genuinely does not. They stay red until task-15740 is fixed.

**Stale contracts, fixed here.** Three tests pinned values that a feature
legitimately changed and nobody updated:

* `test_settings_ownership_records_cover_categories_and_runtime_boundaries` --
  `2d88425ba` (conversation memory and compaction controls) gave Providers &
  Models and Console Behaviour new owned config sections
  (`model_capabilities.models.<model>.context_window`,
  `console.conversation_budget_*`, `console.compaction_*`) and left two
  exhaustive tuples behind.
* `test_console_registers_footer_workbench_shortcuts` -- `14cc326e4` (visible
  prompt queue) made Enter genuinely "send / queue".
* `test_library_shell_rail_search_submit_runs_search_canvas_query` -- see above;
  now asserts against `library_rag_profile_top_k()` rather than a literal that
  encoded "no profile exists".

## Acceptance Criteria

- [x] Each of the six failures is attributed to its causing change, with the commit identified
- [x] It is established whether the three save-related failures are stale contracts or a genuine break in the Settings save path, with evidence either way
- [x] Any genuine product break found is fixed rather than absorbed into the tests' expectations (the log-call crash is fixed; the save refusal is filed as task-15740 rather than absorbed)
- [ ] All six pass on dev

## Loose end handed off

A sixth-adjacent failure appeared during verification and is NOT one of this
task's six: `test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`
fails only in a multi-module run (`ConflictError`, note version mismatch). It
passes alone, passes with its own module alone on base dev, and passes behind a
settings-module prefix. The decisive comparison — the same 5-module set on base
dev — was killed by the environment three times. Filed as **task-15741** with
everything ruled out so far, rather than left as an unexplained red.

## New arrival, attributed but not fixed here

`test_settings_footer_hints.py::test_narrow_footer_collapses_but_f1_help_stays_truthful`
went red on dev between `537451cb8` (green in this branch's runs there) and
`61f6ae575` (fails on an untouched worktree at that commit) -- "expected
collapsed footer at 70 cols". Same class as this task's six: a dev merge in
that range changed footer behaviour without its contract test. Left for the
usual triage rather than absorbed here.
