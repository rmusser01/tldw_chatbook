# Exact f311 Windows startup review

Run34930406527/job104257246915, revision `f311792948ffc252657eb208cb63ceb57fface3d`. Read-only local log/source review; no tests, app boots, edits, or remote reads. Git comparison confirms production source is unchanged from6b201; f311 changes diagnostic/tests and records only.

All23 extracted profile records match selected source and report no observer/config-observer errors. cProfile is disabled. Extraction: `/private/tmp/uat-f3117-startup-profiles.json`; numeric/hash receipt: `/private/tmp/uat-f3117-startup-independent-review-summary.json`. Source-match flags are selected diagnostic receipts, not independent installed-package verification.

## Outcomes

31 non-UI cases PASS in9.76s. All four primary UI cases time out at the original60s limit: claim/recompose, external-copy, keyboard llama-cpp, keyboard llamafile. No completed primary passing call/session durations exist.

| Matched diagnostic | Dev call / pytest session | Candidate |
|---|---|---|
|Keyboard llama-cpp|PASS42.59s /51.89s; two warnings|60s timeout|
|Keyboard llamafile|PASS35.49s /39.39s; two warnings|60s timeout|

Dev completed mount durations are12.876769s and10.058438s. The new focus observer actually works on dev: llama-cpp returns after42 completed press+pause steps in15.541099s; llamafile returns after39 in10.847275s. Dev then executes the remaining case and closes normally. The steps are latest event-observed completed appends, not attempted key counts.

## Candidate progress and exact pending step

Neither candidate enters `test_focus`, any `test_settle`, or `test_close`. Neither returns `_mount_models`. Both remain in the enclosing test's mount await at `test_llm_gguf_source_modes.py:1367`.

- Llama-cpp: context entry at `_mount_models:112` in20–30s samples, then `await app.push_screen(screen)` at114 in40–60s samples.
- Llamafile: context entry at112 in30–40s samples, then push-screen mount at114 in50–60s samples. Earlier10–20s observations have no suspended coordinate yet; that is not a recorded focus/modal wait.

The last measured gate is initial Models-screen mounting. Unlike6b201, these attempts never reach the Refresh→Start focus traversal. Therefore this run cannot resolve the prior focus/timer hypothesis: it supplies no candidate focus count or inner Pilot await. Dev's successful39/42-step traversal is useful baseline behavior but cannot stand in for the unexecuted candidate body.

## Native phases and terminal threads

Candidate final config entry counts are561/514, with25/23 user-directory entries. These include nested operations and represent less completed work than6b201's913/32 after mounting; they are not a cost reduction or unique-admission count.

At20–50s, observed main-thread native configuration paths include compose/inspector projection, character-context publication, and attach/transcript/setup readiness. Final llama-cpp profile records a running config operation aged0.043582s under `_load_recovered_images → _sync_native_console_chat_ui → _sync_console_chat_core_state → provider selection → _provider_readiness_app_config → native admission`. The timeout text later catches main in native descriptor security. This identifies additional real startup synchronization, not a proven redundant callback or infinite repetition.

Final llamafile profile records main waiting in the actual initializer condition (`storage_admission.initializing:244`) beneath readiness/transcript/attach reconciliation; current operation age0.133353s. The subsequent timeout stack has advanced into `_records/startup_permission` and Windows parent-handle validation. Worker6108 is concurrently executing ordinary Notes builtin-Samira resource reading/retirement through `_backfill_chachanotes_messages_fts → get_chachanotes_db_lazy → seed_builtin_content → ensure_builtin_samira → resource_stream/native_open`. This is actual native work and progress, not proof of a permanently retained initializer or an identified wait owner. No long-lived active write transaction is established.

Diagnostic llama-cpp pool: four idle workers and four Textual timer waits. Diagnostic llamafile: four idle, three timer waits, one native Samira-resource worker. All four primary terminal main stacks also end in Windows native handle work; their default pools have idle capacity. These differ from6b201's all-eight timer waits after completed mounting. The old executor-saturation theory does not describe the captured f311 state.

Live thread coordinates and active rows are non-atomic. Elapsed operation age is not continuous time in its sampled native function. The final profile and later text stack may legitimately show different paths, as they do for the initializer. No product permission, fingerprint, or SQLite refusal is shown by these timeout diagnostics.

## Interpretation and disposition

Observed timing is variable: same product6b201 previously passed two primary cases and completed both candidate mounts in43.22/44.52s; this attempt passes neither primary and completes neither candidate mount by60s. The test-only observer change and uncontrolled CI conditions preclude attributing the difference to any single factor. It does establish that the previous progress was not stable startup acceptance. It does not disprove native decoder correctness or measured microcost improvements, nor prove those improvements sufficient for end-to-end startup.

The current supported lead remains synchronous, fresh native configuration/admission work during initial Console/Models mounting, sometimes alongside Notes builtin initialization. There is no new safe patch, cache, relaxed check, deadline change, or focus correction justified by this run alone. Preserve both the6b201 post-mount focus failure and the present pre-mount failures; do not replace one with a single universal cause. Separate current native59+1/plaintext receipts are outside this review and must be verified independently.

Bounded code-only terminal-thread extraction: `/private/tmp/uat-f3117-startup-terminal-frames.json`.
