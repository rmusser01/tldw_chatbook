# Exact 6b201 Windows startup evidence

Read-only review of run34928712931/job104252216498 at `6b20175799c47f44cf287065207630ab1ad86e93`. No app/test runs, edits, or remote requests. Extracted all 21 profile records to `/private/tmp/uat-6b201-startup-profiles.json`; all selected-source checks match, all observer/config-observer error lists are empty, and cProfile is disabled. This checks the diagnostic source receipts, not the separate pending installed/native acceptance run.

## Exact outcomes

31 non-UI cases PASS in 11.91s.

| Primary UI case | Outcome | Test call | Pytest session |
|---|---|---:|---:|
| `test_claim_authority_survives_screen_recompose_and_not_window_selection` | PASS |53.59s|59.85s|
| `test_external_copy_keyboard_geometry_and_unrelated_views_stay_stable` | PASS |56.55s|60.50s|
| `test_supported_width_keyboard_reaches_each_provider_source_and_actions[llamacpp-llama-cpp]` |60s timeout|No completed call duration|No completed passing session|
| Same keyboard case `[llamafile-llamafile]` |60s timeout|No completed call duration|No completed passing session|

The 60.50s passing session includes setup/teardown and does not violate the unchanged 60s test-call limit. The overall workflow remains failed.

| Diagnostic keyboard case | Dev call/session | Candidate | Completed `_mount_models` duration, dev / candidate |
|---|---|---|---|
|llama-cpp|PASS41.37s /48.47s, two warnings|60s timeout|8.149784s /43.215541s|
|llamafile|PASS35.48s /37.94s, two warnings|60s timeout|10.248790s /44.521641s|

## Actual progress versus 9dff

This is materially more completed startup work. Both candidates now return the real mount helper; neither did in 9dff. Both have eight completed settle-helper entries and actually execute the keyboard body. The primary claim and external-copy cases pass; both timed out at9dff. These are observed outcome/progress improvements, not proof that one source change alone accounts for every timing difference between CI runs.

Candidate samples at20s still await app context entry (`_mount_models:112`); at30–40s they await `push_screen` (`:114`). By50s both helpers have returned. Llama-cpp's test is then atline1479 and llamafile's at1446. At60.031s both are at exactly:

`test_llm_gguf_source_modes.py:1532: traversed = await _press_until_focus(pilot, start)`.

All preceding assertions on that linear path have executed: external-source geometry/screenshot checks, return to managed mode, and Refresh focus at1531. The helper has not returned, so Start focus/paint, Start activation, claim/worker assertions, later cleanup, and final success remain unproven. Neither has entered `test_close`.

`_press_until_focus` at242–259 is a bounded80-iteration real Tab traversal. Each iteration awaits `pilot.press(key)`, then `pilot.pause()`, then records/checks focus. The selected observer currently identifies the enclosing test await, not the helper's inner await or iteration. Therefore the precise demonstrated pending step is Refresh→Start keyboard traversal; the records cannot distinguish a single outstanding Pilot/message wait from continued traversal through multiple controls or focus never reaching Start.

## Late threads and configuration

At30–40s the event-loop thread still performs real native config work during Console attach/transcript/rail readiness synchronization. At50 and60s neither candidate has an active selected config operation, transaction, publication, or user-directory call. Both have913 config-operation entries and32 user-directory entries, versus9dff's512/470 and23 while mount was still unfinished. Counts include nested operations and differing completed work; they are not unique admissions or a cost/speed ratio.

All eight default-executor threads in **both primary and both diagnostic** timeout dumps are in Textual `_win_sleep.wait_inner:100`. Storage-admission and UI-stall observer threads are waiting normally. Primary main threads and diagnostic llama-cpp main are in the Windows event-loop selector. Diagnostic llamafile's final text dump catches Textual callback dispatch/signature inspection; its preceding profile catches the selector. Neither terminal snapshot identifies a config-lock holder or ongoing native configuration operation.

This differs from9dff, where the candidates never returned mount, main was still performing native config checks, and four executor workers were idle with one active Notes/builtin worker. It resembles earlier timer-wait snapshots, but **does not prove executor deadlock**: timer due times, cancellation state, queued work, and the exact inner Pilot await are not recorded. Textual `_win_sleep` does submit both blocking timer waits and cancellation signalling to the default executor, which is a concrete contention route; these stacks alone do not show that cancellation work is actually queued behind long waits.

The unchanged `Pilot.press` awaits key dispatch and `_wait_for_screen`; `Pilot.pause` awaits message processing and idle. Thus a selector main thread is compatible with waiting for normal progress as well as contention. Native config has also consumed much more of the initial mount window than dev:44s versus8–10s for comparable completed mount helpers. The remaining keyboard failure may still reflect the overall cost of that startup plus otherwise normal subsequent traversal. It is not evidence that the focus code itself is broken or that its limit should change.

## Bounded disposition

Preserve both failures and existing guards/deadlines. The artifact supports further examination of the exact `_press_until_focus → Pilot.press/pause` wait, not another speculative native-config correction at the terminal point. The smallest missing causal evidence is its current inner await and whether Tab iterations continue; a timer-pool correction additionally requires demonstrating the actual queued cancellation/due-timer condition. No safe product correction follows solely from these samples. Do not remove traversed-focus assertions, expand the60s limit, suppress Canvas polling, or change executor sizing based on the all-eight-wait snapshot.

The separate34928730294 native-correctness/microcost/installed-backup run remains pending and is not accepted or otherwise classified here.

Hashes and bounded per-comparison progress/terminal-frame metadata: `/private/tmp/uat-6b201-startup-independent-review-summary.json`.
