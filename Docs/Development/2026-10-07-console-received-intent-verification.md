# Console received intent verification

Task: TASK-34563.15. Base: d76319fafd26b7ce13d8b80a49665a22e1bbc984.
ADR: [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md).
Plan: [received intent](../superpowers/plans/2026-10-07-console-received-intent.md).

## Result and remaining latency

Eligible Send now enters existing runtime custody and its single received admission slot before checked hook/configuration preparation. A detached intent carries exact authored draft and staged-input revisions. The same custody record promotes once into the existing request; edits, Stop, Close and replaced owners cannot clear a newer draft or dispatch stale input. Queue admission remains with the original queue coordinator. Custom queue and launch callbacks retain their supported call shape.

Initial review and checked capture survive navigation without retaining a screen continuation. Required finite provider, Library, reference and workspace-availability reads use the existing physical preparation-read owner. Runtime disposal drains the original availability reader and its outer owner, including repeated cancellation, while borrowed connections remain borrowed. Saved acceptance retains the original physical result; Stop or replacement after a successful native save suppresses composer clearing and dispatch. WAL/NORMAL, saved-failure-keeps-draft, temporary chats, permission/source checks and required history/consent/trace/effect order remain unchanged.

This is a responsiveness and ownership checkpoint, not completion of the speed target. The matched full-app diagnostic with real storage and trace settlement measured:

| Sample | Before | After |
| --- | ---: | ---: |
| First Send | 10.844108 s | 10.646876 s |
| Second Send | 10.094182 s | 6.821233 s |
| Third Send (streaming) | 8.902770 s | 6.049486 s |

One three-turn run per source state is insufficient to attribute all variance. The cold first Send barely changed. Save-success to trace-reservation-entry still occupies 6.430, 3.501 and 2.894 seconds after the change. This is a broad interval, not attribution to trace storage itself. Send heartbeat maxima remain 277/128/159 ms; typing in the full-app diagnostic also retains a 714 ms maximum. The under-one-second whole-Send and universal sub-100ms responsiveness goals remain open.

## Targeted integrated checks

Both implementation lanes finished before root ran integrated checks; native runs were sequential with isolated profiles and original deadlines. No full sweep or external provider request was run.

| Label | Result | Evidence |
| --- | --- | --- |
| received-intent-final-domain | 68 passed | Exact inputs/revisions, custody, promotion, queue, custom callbacks, native preparation, saved outcomes and selected original regressions. |
| received-workspace-owner-green | 9 passed | Held original availability SQL, repeated disposal cancellation, borrowed/owned handles, original availability cancellation controls and actual Enter cleanup. |
| received-intent-final-ui | 9 passed, 2 failed | Actual Enter/button feedback, identical retype, initial review and original configuration-worker controls passed with clean teardown; the two failures were investigated below. |
| received-intent-navigation-custom-green | Original custom-launch control passed; navigation failed | Corrected async launch adapter to honor replacement of its paired sync callback. |
| received-navigation-saved-adapter | Navigation passed | Original held configuration, actual uninstalled-screen removal, same runtime custody, real file-backed saved turn and unchanged successor draft. |
| received-intent-after-1 | Three turns passed | 63.688 s driver; three complete traces, three replies and links, no leftover dispatch checkpoints, unchanged source during the run. |

The actual visible eager-navigation harness with production CSS measured natural Preparing frames at 19.135 ms (Enter) and 23.084 ms (button); driver input mutations at 1.421/1.532 ms and natural changed-input frames at 5.699/5.760 ms. All occurred while the original native configuration operation was held, and exact native resources retired before teardown. These are headless supplied compositor frames, not physical terminal flush measurements.

Final successful runs retired their bounded native trees/profile normally with zero forced retirement, identity overflow or lookup races. Separate in-test assertions, rather than process emptiness, prove original SQL/connection/lease retirement. The earlier ownership RED had one diagnostic lookup race; it is not reported as zero.

The whole-Send harness preserves the original action clock, heartbeat, real writes and trace barriers. Its pre-existing provider-resolution wrapper selects the streaming third turn and therefore declines the private stock-only provider worker injection. It is a diagnostic, not provider-budget acceptance. Native-work audit/stack sampling is omitted.

## Review corrections and evidence limits

Causal controls first failed for receipt after configuration, missing exact revision APIs, late custom queue replacement, draft clearing after stopped/replaced native save, and runtime disposal finishing before original availability resources retired. Their corrected targeted controls pass. Independent source review found no remaining actionable issue in these changes.

Navigation fixtures required three production prerequisites: Textual attachment/stack membership rather than the historical is_mounted latch; file-backed persistence rather than the db-less or thread-affine in-memory app fixture; and an adapter-backed configured provider rather than a llama.cpp fixture that probes a live local server. Original failure logs remain preserved. No production refusal, source gate, deadline or cleanup assertion was weakened.

AST and whitespace checks pass. All nine new Python files pass formatter checks. Baseline-relative Ruff reports zero introduced findings and 307 inherited findings. Existing module-size ratchets and inherited static debt remain red; caps were not raised. TASK-34563.15 remains In Progress under the repository's full DoD rather than claiming all static/performance acceptance.

Product changes span received models/driver, store/controller/runtime, configuration/Library/provider preparation, Console wiring/session/queue/workspace and composer revision publication. Local evidence is under .superpowers/sdd/2026-10-07-console-received-intent (final-source.json, final-static-completed.json, planned-selection.json and regression receipts), plus the shared checks and native-pairs directories using the labels above. No primary checkout edit, push or merge.
