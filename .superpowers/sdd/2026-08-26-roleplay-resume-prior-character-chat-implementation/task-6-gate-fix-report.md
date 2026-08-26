# Task 6 targeted-gate fixture correction report

Date: 2026-08-26

## Result

The two evidence-backed test-fixture defects are corrected without changing
production code. The exact nine nodes reported by the Task 6 pre-close gate now
pass, and focused selections covering both altered fixture paths pass. The parent
task still owns the complete nine-file targeted-gate rerun and final closeout.

## Baseline causality

Before any correction, the exact nine pre-close node IDs reproduced as
`9 failed`. This matched the pre-close comparison against the starting commit:

- The two partial-projection cases stored
  `character_name_snapshot=None` while the durable character owner was
  `Alraune`. The exact v2 ownership guard therefore rejected the fixture by
  design.
- The seven native-flow cases prepared provider state only in
  `app.app_config`. Real mount-time scoped-rail persistence invalidated that
  cache, after which readiness reloaded the isolated per-test disk config and
  discarded those in-memory-only values. The same seven failures were present
  at the starting commit.

These were fixture/harness defects rather than regressions in the Task 6
production changes. No production behavior was weakened.

## Corrections

### Durable roleplay ownership fixture

`Tests/Chat/test_console_chat_store.py` now persists the owned historical name
`Alraune` in the two partial-projection cases. The test parses the saved durable
metadata and explicitly asserts the exact roleplay ownership context before
exercising projection repair. Separate legacy fail-closed coverage is unchanged.

### Native provider configuration harness

`Tests/UI/test_console_native_chat_flow.py` now persists provider/default/key/
endpoint/model fixtures into the hermetic per-test config before mount and then
force-reloads the app snapshot. Post-mount provider changes follow the
production-faithful save path: persist config, replace the active
`ConsoleSessionSettings`, and synchronize the console control bar.

A focused regression,
`test_native_ready_console_config_survives_cache_invalidating_reload`, proves
that prepared llama.cpp provider state survives a real scoped-rail config save
and force reload.

## RED evidence

1. Exact pre-close nodes, before correction:

   - Command: the nine explicit node IDs listed in
     `task-6-preclose-report.md`, run together with `pytest -q --tb=short`.
   - Result: `9 failed` (exit 1).

2. New cache-invalidation regression, before the persistent helper correction:

   - Command:
     `../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py::test_native_ready_console_config_survives_cache_invalidating_reload -q --tb=short`
   - Result: `1 failed, 2 warnings`; the force reload recovered the disk defaults
     (`OpenAI`, `gpt-5.6-terra`) instead of the prepared values
     (`llama_cpp`, `prepared-model`).

## GREEN evidence

Corrections were applied and checked one fixture path at a time:

1. Durable ownership fixture:

   - Both parameterized partial-projection nodes: `2 passed, 1 warning in 1.03s`.

2. Persistent native-ready helper and real cache invalidation:

   - New focused regression: `1 passed, 2 warnings in 2.66s`.
   - The two improvement-disclosure nodes plus collapsed-layout node:
     `3 passed, 2 warnings in 7.52s`.

3. Remaining corrected native nodes:

   - Generic provider send: `1 passed, 2 warnings`.
   - Send-button click: `1 passed, 2 warnings`.
   - Successful-send tooltip: `1 passed, 2 warnings`.
   - Configured-model gateway: `1 passed, 2 warnings`.

4. Exact pre-close nine-node rerun:

   - Result: `9 passed, 2 warnings in 13.89s`.

5. Focused helper-regression selections:

   - `../../.venv/bin/python -m pytest Tests/Chat/test_console_chat_store.py -k 'roleplay_context or roleplay_projection' -q --tb=short --disable-warnings`
   - Result: `3 passed, 295 deselected, 1 warning in 0.31s`.
   - `../../.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -k 'native_ready_console_config or provider_selection or generic_provider_send or send_button_click or successful_send or configured_model_reaches or improvement_disclosure or collapsed_layout' -q --tb=short --disable-warnings`
   - Result: `12 passed, 288 deselected, 2 warnings in 15.48s`.

6. Static verification:

   - `../../.venv/bin/python -m ruff check Tests/Chat/test_console_chat_store.py Tests/UI/test_console_native_chat_flow.py`
   - Result: `All checks passed!`
   - `git diff --check`
   - Result: exit 0 with no output.

## Files and boundaries

Changed test files:

- `Tests/Chat/test_console_chat_store.py`
- `Tests/UI/test_console_native_chat_flow.py`

Evidence artifact:

- `.superpowers/sdd/2026-08-26-roleplay-resume-prior-character-chat-implementation/task-6-gate-fix-report.md`

No production, CSS, backlog, ADR, shared test-helper, or branch-history changes
were made. No full repository suite was run. ADR required: no; these are local
test-fixture corrections that preserve existing runtime boundaries and policy.

## Warnings and concerns

The pytest summaries above retain the observed warning counts: the existing
`requests` dependency-version and `pydub` `audioop` warnings. Pytest also emitted
the known late temporary-directory cleanup warnings (`PermissionError` and
`OSError: Directory not empty`) after its summaries. None was an assertion,
runtime, or collection failure.

The remaining concern is intentionally outside this correction: the parent must
rerun the complete nine-file targeted gate to establish joined-suite evidence
before Task 6 can be closed.
