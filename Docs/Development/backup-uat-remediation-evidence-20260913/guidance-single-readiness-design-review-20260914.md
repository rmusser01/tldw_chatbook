# Single readiness result for one guidance projection

**Design is suitable for a focused TDD experiment.** Unlike a widened derivation memo, this proposal obtains the original full readiness result first, then passes it only through the existing synchronous presentation legs. No implementation was reviewed or approved; no tests/apps/edits were run. The earlier broad-scope rejection remains valid.

## Reviewed source boundary

`_active_console_settings_readiness:9846` remains the sole acquisition API, including its current existing per-pass-memo behavior. Its uncached implementation calls native session ensuring/convergence, provider selection, fresh config readiness, active-run state and model warnings. Keep all of that unchanged and call it at the start of `_sync_console_transcript_guidance:15127`, exactly where the first blocker helper currently triggers it.

Both returned types are frozen dataclasses (`Chat/console_session_settings.py:366,498`). The fields consumed by these projections are scalar values. `_console_provider_blocker_copy:14952` after acquisition only checks readiness and calls pure `build_console_readiness_presentation`; `_console_provider_recovery_action:15053` similarly builds fixed display/action strings. The intervening empty-action helper only combines strings. There is no product session/config write, await, native operation, UI update or scheduled callback between those returned-readiness projections.

`_build_console_setup_card_state:15064` first acquires readiness, then checks model selection, first-send state and current transcript messages in that order. The first-send getter only initializes an existing screen-local flag from app_config; it does not invoke its separate persistence writer. The message getter reads the active store's transcript. Preserve both predicates at their current location, after the reused result; do not move them to the initial acquisition or suppress them.

Session creation and task177 eligible blocked-default convergence can perform real state/native writes **inside the first unchanged readiness acquisition**. They finish before its returned pair is used. The proposed pair sharing does not keep a config memo or operation open across those writes. This is the key difference from the rejected broad `_console_derivation_scope` around builders.

## Smallest API shape

Use an optional keyword-only `settings_readiness` pair on the three existing helpers; when it is None, call the original acquisition exactly as today. Guidance alone passes its one local pair to all three. A single pair avoids independently missing/mismatched settings and readiness parameters. No screen attribute, cache, decorator, new operation lifetime or fallback TypeError handler is needed.

Retain surface query/update and `_sync_console_setup_modal` after all projections, outside any new memo. Modal synchronization can apply blocking state, start discovery and schedule focus; it must not inherit cached native/config authority. Independent inspector/workbench/action callers retain their default fresh acquisition. Actual send/effect admission remains unchanged.

A concurrent external config update may now be reflected on the next guidance pass rather than cause internally mixed first/second/third display results in one pass. That is the explicit display snapshot boundary, not a promise to preserve every removed read's transient exception opportunity. It must not be extended to a later callback, save, await or effect.

## Focused proof obligations

1. On a real fixed private profile, wrap/delegate the original uncached readiness acquisition and native/config entry counters; call the actual guidance method with its real helpers. Baseline should derive three times; candidate once outside an already-active derivation memo. Compare actual blocker/action/card outputs. Do not replace the readiness result or only assert a mocked call count. Remaining internal config checks still run; do not claim all613 historical provider-config calls shrink threefold.
2. Include cold/eligible default convergence, not only a prewarmed session: assert the original real session creation/task177 save path completes before projection, the resulting pair matches stored effective settings, and the expected native persistence/readback occurs. Separately prove explicit user settings are not changed.
3. Save real provider settings between two guidance passes and assert the second pass reflects fresh values; independently invoke each helper without the keyword and verify its fresh default. Preserve task177 disk-refresh/fallback tests and task15452 existing scoped tests.
4. Assert first-send/message predicates and both UI consumers retain their order; no memo remains during surface/modal calls. Projection failures retain exact identity and do not leave a reusable result on the screen; initial native refusal/error uses its original path.

Existing zero-argument helper lambdas occur in `test_console_fleet_wake_ui_freshness.py:285`, `test_console_workbench_contract.py:703`, `test_console_right_rail.py:711–715`, and prompt-controller fixtures. Only fixtures actually receiving guidance's new keyword need a narrow signature update. Do not weaken behavior assertions or add production compatibility exception swallowing for test doubles.

Existing useful coverage: `Tests/Backup_Recovery/test_console_provider_derivation.py` (native once/freshness/session/exception), UI task15452 draft equality tests, and task177 session-settings convergence/fresh save/setup-card tests. New native proof is still required: source reading alone does not establish output equality or startup improvement.

Reviewed HEAD: `f311792948ffc252657eb208cb63ceb57fface3d`.
ChatScreen SHA256: `701f3ac336d2c103c344cb9f0c5e171b10e4cc64edf2821b7e5d14b6592bb654`.
