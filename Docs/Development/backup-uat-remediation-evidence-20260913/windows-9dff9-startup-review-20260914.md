# Exact 9dff Windows startup review

Run 34926584886 / job 104245831443, revision `9dff9dd1b2a73795eece24970ac43f1dc571c2b6`. Read-only review of the saved job log, all 23 extracted profile records, and current matching source; no new tests or app runs. The extracted records exactly match the JSON embedded in the log. All 23 report matching selected source, no observer/config-observer errors, and cProfile disabled. These selected-source receipts are not a new installed-package receipt.

## Outcomes

31 non-UI cases pass in 10.20s. All four primary UI cases time out at their unchanged 60s limit: claim/recompose, external copy, keyboard llama-cpp, keyboard llamafile. There are no primary UI passes in this run.

| Matched diagnostic | Pinned dev test call / pytest session | Candidate |
|---|---|---|
| Keyboard llama-cpp | PASS 42.41s / 53.19s, two warnings | 60s timeout |
| Keyboard llamafile | PASS 37.40s / 40.86s, two warnings | 60s timeout |

The dev calls and sessions are different quantities. Final observation elapsed times (54.109/42.875 dev; 60.204/60.016 candidate) also include diagnostic lifetime and must not replace pytest call times. Both comparisons use separate processes and the unchanged keyboard assertions; finite monitoring remains enabled despite cProfile being off. This is repeatable comparative regression evidence, not an equal-completed-work throughput measurement.

## Where the candidates stop

Both candidates have exactly one active test and mount helper. At 20–40s `_mount_models` is suspended at `test_llm_gguf_source_modes.py:112`, awaiting `context.__aenter__()`. At 50–60s it is suspended at line 114, awaiting `app.push_screen(screen)`. Neither returns the helper, enters its subsequent settling loop, or begins the keyboard actions. Dev finishes mounting, executes the keyboard body, enters `_close_context`, and exits. Thus this failure is initial app/Models-screen mounting, not a modal/keyboard assertion that became slow after successful setup.

`ChatScreen.on_mount` schedules `_reconcile_console_after_attach` at `chat_screen.py:16565`; the callback calls `_sync_native_console_chat_ui` at 16579. Late samples show this actual attach callback still progressing through its original UI synchronization. Textual `Screen._invoke_and_clear_callbacks` awaits callbacks, and `push_screen` returns `AwaitMount`, which waits for mounted events. Samples do not identify the exact unfulfilled widget event or prove a circular wait; they do locate substantial synchronous native work on the event-loop thread while the test awaits mount.

## Native work and actual contention

* Llama-cpp at 30.016s: thread 1736 is saving Console rail preferences through the real config transaction/native admission; main thread 9988 is waiting in `config_participants.operation:343` beneath `CompactModelBar.compose:46`. This identifies a concrete concurrent writer and config-lock wait in the same non-atomic observation. The transaction subsequently returns in 1.838075s; it is not evidence of a permanently retained writer.
* At 50.016s, llama-cpp main is waiting on config acquisition during `_sync_console_mode_bar`, while worker 7040 is in Notes-path/user-directory native acquisition. At 60.204s main is actively acquiring native authority through `_provider_readiness_app_config:7615 → session._maybe_refresh_stale_default_console_settings:3602 → transcript guidance → _sync_native_console_transcript:18234 → _sync_native_console_chat_ui:18515 → attach reconciliation`. Its recorded current config-operation age is 0.452666s.
* Llamafile at 40.016s samples actual prompt-history loading and default-path native config access. At 60.016s main is in a nested `raw._check` / Windows descriptor security check through the same readiness source, this time from roleplay/identity/control-bar synchronization at `chat_screen.py:18512`. The existing outer `_run_console_config_sync` is already active (0.411265s); nested entry age is 0.003430s. Grouping that already-guarded control-bar body again would not remove these required nested checks.
* Final counts are 512 and 470 config-operation entries and 23 user-directory entries per candidate; these counts include nested operations and are not unique native admissions. Each records one completed transaction/publication/rebuild. Llamafile's completed transaction is 3.379229s. No active transaction remains at either final observation.
* Each terminal default pool has four idle workers, three Textual timer waits, and one active Notes/builtin-content worker. Llama-cpp's active worker is entering visual-identity native authority; llamafile's is inserting the built-in Samira character through the real Notes transaction. This is materially different from d2f's all-eight timer-wait samples. The old timer-pool hypothesis is not established for this run. Neither terminal main stack is the periodic Canvas watcher.

The nearest supported causal lead is slow initial Console reconciliation/configuration work, sometimes contending with ordinary background configuration/Notes initialization. Native config is demonstrably on the late main-thread path, rather than an inference from aggregate historical profiles. There is no proof of one stuck lock, an infinite reconcile loop, or a specific removable security check. Active-row age and sampled coordinates are non-atomic; they do not measure continuous time in SID conversion or identify a mutex owner. Do not add inclusive timings or attribute all pre-mount delay to one sampled helper.

## Smallest next proof

Use the actual observed attach/readiness path as the finite measurement boundary, rather than changing keyboard waits or suppressing required mount work. A disposable native fixture can execute the existing `session._maybe_refresh_stale_default_console_settings → _provider_readiness_app_config` and subsequent readiness projections with their real store/config provenance, and measure the unchanged sequence's outer acquisitions versus nested checks. Include the observed rail-preference writer or Notes-path contender only for a separately controlled contention case. Any proposed removal/combination must first prove identical fresh config, source, authority and UI results; this artifact alone does not authorize a cache, wider scope, omitted check, or callback deferral. A whole-screen counter alone would again obscure which actual boundary costs time.

The separate native backup 42+43 success remains separate evidence. This startup failure is preserved. VoiceAEC checkout/filename failures are unrelated and not assessed here.

Hashes and bounded per-comparison timing/coordinate data: `/private/tmp/uat-9dff9-startup-independent-review-summary.json`.
