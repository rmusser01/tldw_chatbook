# Console received admission verification

Task: TASK-34563.11. Decision: [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md).
Plan: [Received admission](../superpowers/plans/2026-10-07-console-received-admission.md).
Base: `661da42c2688e078235f42461907ee39f441b562`.

## Scope

Existing complete runtime requests reserve the same per-session store slot used by full preparations before input transfer or task scheduling. The exact claim promotes atomically, and stale cleanup cannot release its successor. Runtime keeps original-source references and uses a private lazy task constructor. Controller activity, origin gates, Stop and Close see received work. Nonpromoting submissions recheck their claim before acceptance; initial claim release precedes the next queued turn. Recovery checks its original store/session witness.

This is the admission foundation. UI intake still constructs the complete request before custody. Unconfigured input, synchronous draft/staging revisions and early screen-free capture remain to be enabled. No 100 ms rendered/input feedback or subsecond action-to-adapter result is claimed here.

## Baseline and regression evidence

- On unchanged base, 13 selected existing cases passed in two sequential batches.
- Meaningful runtime RED: a second actual synchronous `accept_turn` call was accepted before either task stepped and took the losing request's staged input. The new store API test also failed as expected because that API did not exist.
- The runtime RED fixture left a pending Canvas watcher. That is an explicit fixture limitation; contained process exit did not establish proper runtime cleanup. Final fixtures now dispose the actual runtime.
- The baseline scheduling-failure control patched global task creation and emitted unrelated unawaited Canvas coroutine warnings. It now injects the same failure at the narrow private custody constructor.

Final integrated results on unchanged product source:

| Run | Result |
| --- | --- |
| New admission/custody/saved-turn controls, first pass | 44 passed, 4 fixture failures |
| Original selected compatibility controls | 13 passed |
| Mounted Send/queue/ownership and saved-turn cancellation controls | 8 passed |
| New controls after valid fixture corrections | 48 passed |

The final qualified selection is **69 passing cases, zero failures/errors/skips**. The initial four failures were invalid preparation context values and an incorrectly shaped forged queue token; only those test fixtures changed before rerun. The duplicate actual runtime acceptance RED now passes, including exact loser-input preservation. The real failed SQLite checkpoint control keeps the original input and recovery, commits no messages/conversation and makes no provider call.

All four integrated runs exited with normal contained-Job emptiness, native identity release, pipe/monitor retirement and private profile deletion. No forced retirement, diagnostic overflow or identity lookup races occurred. This proves runner process retirement, not universal application native cleanup. The original compatibility run still emitted one legacy Canvas cleanup coroutine warning; the final new and mounted/save runs did not emit unawaited/destroyed-task warnings. Deprecation warnings remain.

Scoped syntax and diff checks pass. New files have clean Ruff/format results; existing files add no lint or formatter diagnostics. Baseline lint remains 169 store / 60 controller. Existing formatter deltas remain unchanged (controller 15, runtime 38, queue coordinator 5, lifetime tests 4, ownership tests 9). Module budgets remain exceeded: controller 30,212 → 30,286 against 29,299; store 22,789 → 22,814 against 22,344; interrupt host unchanged 6,520 against 6,471. No cap was raised. These are qualifications, not a full static-green claim.

Local source hashes and complete per-run receipts are in `.superpowers/sdd/2026-10-07-console-received-admission/{final-static,final-audit}.json`; immutable run logs/XML/result/custody receipts use the four `received-integrated-*` labels under the shared-preparation checks directory. All product hashes stayed unchanged across final behavioral execution; only the new test fixture changed.

## Review qualifications

Independent review identified missing revalidation after awaited work on nonpromoting routes, queue capacity exemptions hiding foreign claims, and recovery into a same-ID replacement session. Focused fixes and regression controls address those boundaries in the existing owners.

Existing bare native reads remain a separate lifetime qualification: hook admission, reference expansion, Library policy reads and hook runtime configuration/context reads may continue after cancellation of their awaiting Task. Draining an entire received-submit Task would neither prove native retirement nor preserve bounded provider shutdown. This slice introduces no new native producer and preserves the exact existing ordinary-save and resident hook-review retirement drains. AC #3's physical-retirement requirement stays open; early UI capture cannot be declared safe from this foundation alone.

Saved acceptance failure continues to stop dispatch and keep the draft. WAL/NORMAL, required consent/trace/checkpoint/effect ordering and awaited history are unchanged. The broader latency target remains unmet until enabled-path timing and real rendered/input feedback are measured.
