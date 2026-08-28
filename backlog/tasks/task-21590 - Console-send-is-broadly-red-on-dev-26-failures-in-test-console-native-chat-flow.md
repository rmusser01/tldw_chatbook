---
id: TASK-21590
title: >-
  Console send is broadly red on dev   26 failures in test console native chat
  flow
status: Done
assignee:
  - '@codex'
created_date: '2026-08-23'
updated_date: '2026-08-28 21:55'
labels:
  - testing
  - dev-red
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_console_native_chat_flow.py` fails 26 tests on pristine dev. The failures span
generic-provider send, the retry/regenerate/continue family, and the first-send-flag pair — the
core Console send path. This is either a stale harness or a real break in sending, and until
someone determines which, every Console branch inherits a red baseline that hides regressions in
exactly the app's most-used flow.

A scratch probe on pristine dev shows the draft is **neither sent nor cleared after Enter**, and
a single-line control fails the same way — so it is not shift+enter specific.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A determination is recorded, with evidence, of whether Console send is genuinely broken in the shipped app or only in the test harness
- [x] #2 If the app is broken, the send path is fixed and a test pins the behaviour that regressed
- [x] #3 If the harness is stale, the harness is repaired so the tests exercise the real send path again — not deleted, and not relaxed until they pass
- [x] #4 `Tests/UI/test_console_native_chat_flow.py` is green on dev — **304 passed on current dev; TASK-22000 repaired the final 2 queue-contract failures**
- [x] #5 The fix is verified by mutation: breaking send makes these tests fail again

## Evidence (verified first-hand on dev 33ff5b754, 2026-08-23)

```
pytest Tests/UI/test_console_native_chat_flow.py -q -p no:randomly
  -> 26 failed, 271 passed  (7m 33s)
```

Surfaced by the TASK-21501/21123 implementer, which classified rather than waved through the
composer-suite reds it inherited: 9 of its 12 composer-suite failures trace to this same root
cause. Independently reproduced here before filing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: `backlog/decisions/046-visible-bounded-console-prompt-queue.md`

Reason: this is a verification and backlog-closeout pass for the already accepted
ADR-046 behavior and TASK-22000 repair; it introduces no storage, ownership,
runtime-boundary, or long-lived UX decision.

1. Trace one representative failure through the real send path (button press -> screen
   handler -> prompt-queue dispatch -> controller) instead of reading the tests first, and
   record the exact refusal.
2. Identify the commit that introduced the refusal (`git log -S` on the refusal copy, then an
   A/B of that commit's production tree against the unchanged test file).
3. Answer the only question that matters live: boot the REAL `TldwCli` headless under an
   isolated HOME/XDG/`TLDW_CONFIG_PATH` sandbox with only the network boundary stubbed, press
   Enter in the real composer, and record whether the provider is called and the draft clears.
4. Repair whichever side is wrong. If the harness: restore the production precondition the
   factory app is missing, do not delete/xfail/relax any test.
5. Mutation-check every repaired test: break the production gate and confirm each goes red.
6. Re-run the file and the related composer suite; run `./scripts/preflight.sh`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Console send is not broken for real users.** A headless Pilot run of the real `TldwCli`
(no `_build_test_app` stubs) under an isolated `HOME`/`XDG_*`/`TLDW_CONFIG_PATH` sandbox, with
only `chat_api_call` stubbed, sends normally: `store.persistence = ChatPersistenceService`,
`commit_durable_turn` callable, one provider call, draft cleared, transcript
`user 'hello' [complete] / assistant 'LIVE-PROBE-REPLY-OK' [complete]`, run state COMPLETED. A
second live probe confirmed a second send in the same session also works. The 26 failures were
**stale test doubles**, in two layers, plus one genuine product conflict.

### Root cause 1 — the factory app has no ChaChaNotes DB, and a durable turn now fails closed

`56db75386` ("fix(console): harden durable turn ownership", 2026-08-23) rewrote TASK-19900.3's
durable-acceptance gate. Before it, the gate required `persistence is not None and
persistence.db is not None` and refused through `_block()`. After it, the gate is
`durable_turn and not callable(durable_commit)` and returns a **bare `ConsoleSubmitResult`**.
Two consequences: a `persistence=None` store (which `_build_test_app` always produces, because
it patches `get_chachanotes_db_lazy` to `None`) is now treated as a durable turn that cannot
commit; and the refusal writes no system row and raises no toast, so the symptom is literally
"pressing Send does nothing". A/B with the test file unchanged: `56db75386^` → 1 passed,
`56db75386` → 1 failed.

Repair: `Tests/UI/app_factory.py` gains `attach_chachanotes_db(app)`, and the 26 send tests
build through a local `_build_console_send_test_app()`. The DB is **`:memory:` on purpose** —
`ConsoleRuntime.ensure_agent_bridge` refuses to build an agent bridge for a `:memory:` DB, so
this restores exactly the precondition the send path lost without also flipping 26 tests onto
the agent loop (which a file-backed DB does, since `[console] agent_runtime` defaults on).

### Root cause 2 — two stale doubles the first refusal was hiding

* `_ReadyResolutionGateway` returned a ready resolution with no `resolved_destination`.
  `a26cdafd8` made that typed destination mandatory, so `_finalize_turn_execution_context`
  raised and the send was blocked with "Provider destination is incomplete." The stub now
  derives it through the production classifier (`resolve_console_destination`) so it cannot
  drift again.
* `WorkspaceLinkingPersistence`, a three-method hand-rolled persistence double three tests
  installed over the store's real adapter, has no `commit_durable_turn` and hit the same
  silent refusal. Its only job — linking a persisted conversation into the workspace registry
  — is what the real `ChatPersistenceService` already does through the same registry, so the
  double is deleted and those tests now assert against production's own linking.

### The 2 tests left red are correct, and pin a real regression (TASK-22000)

`test_console_composer_stop_is_subdued_when_idle` and
`test_console_duplicate_send_during_stream_does_not_break_stop_control` assert ADR-046 /
TASK-14808 / TASK-15121: an accepted live turn re-labels Send to "Queue" and admits a FIFO
follow-up rather than blocking. Verified live on the real app mid-run: `send.label = 'Queue'`,
`send.disabled = True`, `console-send-blocked` set, tooltip "Wait for the active Console run to
finish before sending", `dispatch_recovery_blocks_submission = True` for a
`DISPATCH_STARTED` owner whose own state says `runtime_active=True, recovery_needed=False`.
Introduced by `2c7fcd200`. This is **not** harness staleness — but neither is it mine to flip:
`Tests/Chat/test_console_dispatch_recovery_fix_round1.py::test_healthy_durable_owner_is_not_
recovery_before_checkpoint_transition` asserts `blocks_submission is True` for exactly that
healthy live owner, so the durable-turn programme pinned the opposite contract on purpose.
Two shipped contracts disagree and an owner must choose; filed as TASK-22000. Relaxing either
side here is the "green suite that proves nothing" this task exists to prevent.

### Counts

| | before | after |
|---|---|---|
| `Tests/UI/test_console_native_chat_flow.py` | 26 failed, 271 passed | 2 failed, 295 passed |
| `Tests/UI/test_console_composer_cursor.py` | 1 failed, 28 passed | 0 failed, 29 passed |

### Mutation evidence (every repaired test killed by at least one production mutation)

* **M1** — delete `self._launch_chain(...)` in `ConsolePromptQueueUIController._stage_normal_chain`: **21 of 25 killed**. Survivors: the three retry/continue/regenerate action tests (different entry point) and one block-path test.
* **M2** — force `durable_commit = None` (restores this task's regression): **22 of 25 killed**. Survivors: the same three action tests, which use non-manual origins the durable gate does not cover.
* **M3** — make `_resolved_destination_for_context` always raise: **23 of 25 killed**. Survivors: the two block-path tests, which refuse before the provider is resolved — both killed by M2.

Union of M2 and M3 = **25 of 25**. No repaired test passes vacuously. All mutations were applied
and reverted by hand; `git diff -- tldw_chatbook/` is empty.

### Files

* `Tests/UI/app_factory.py` — new `attach_chachanotes_db`
* `Tests/UI/test_console_native_chat_flow.py` — `_build_console_send_test_app`, 26 call sites, `_ReadyResolutionGateway` destination, `WorkspaceLinkingPersistence` removed
* `Tests/UI/test_console_composer_cursor.py` — the one send-driving test in that module
* `backlog/tasks/task-22000 - ...` — the ADR-046 queue conflict
* `backlog/docs/lessons-testing-evidence.md` — the fail-closed-double lesson

No production code changed.

## Follow-up: the repair unblocked a real egress seam (2026-08-24)

Review caught that the repaired tests were attempting live network egress: 16 teardown
errors from `Tests/conftest.py`'s `_no_network_io` guard (task-15111), each recording
`socket.connect -> 57.150.97.129:443` (an `openaipublic.blob.core.windows.net` address).
The tests themselves passed; only the guard saw it, and because the errors attach to
*passing* tests that still fail early on dev, a naive A/B baseline shows nothing.

**Seam.** Not `chat_api_call`. Tracing the guard's `_deny` with a stack dump gave one
distinct stack (the six connects are tiktoken's own retries):

```
_submit_console_native_draft -> run_prompt_chain -> submit_draft -> _accept_durable_turn
  -> resume_durable_postcommit -> _run_durable_postcommit_effect
  -> _stream_assistant_response -> _apply_conversation_memory_preflight
  -> ConsoleProviderGateway.prepare_chat_request -> prepare_provider_request
  -> _account_categories -> _count_wire -> count_console_messages_tokens
  -> token_counter.estimate_tokens -> count_tokens_tiktoken -> get_tiktoken_encoding
```

`tiktoken.get_encoding` downloads its BPE blobs on a cold cache. The old harness never
got past the durable-acceptance gate, so it never reached `prepare_chat_request` at all —
the repair is what exposed the seam.

**Fix.** A `_no_tiktoken_bpe_download` autouse fixture in `Tests/UI/conftest.py`, sibling
to the existing `_disable_model_catalog_refresh` (task-16198, added for the same guard,
same shape). It patches the single chokepoint `token_counter.get_tiktoken_encoding` to
return `None`, which drives the already-tested no-tokenizer branch — the branch a default
install takes anyway, since tiktoken is not a base dependency (task-2526). No
`allow_network`, no `loopback_network`, no socket patching, and no narrowing of the
dispatch under test: the send still runs the whole real path, it just counts tokens by
character estimate. `Tests/UI` scope keeps `Tests/Chunking` (which legitimately needs a
real tokenizer and skips when it is uncached) untouched.

**Counts after the fix:** `test_console_native_chat_flow.py` + `test_console_composer_cursor.py`
= **2 failed, 326 passed, 0 errors** (was 2 failed, 326 passed, **16 errors**). The 2
failures are the unchanged TASK-22000 pair.

**Mutation kills are unchanged:** M2 (force `durable_commit = None`) still kills 22/25,
M3 (make `_resolved_destination_for_context` always raise) still kills 23/25, union
survivors = none, so 25/25. Stubbing the tokenizer did not weaken what the tests exercise.

**Wider cluster:** the 20 adjacent Console UI modules re-run at 26 failed / 600 passed
with **0 blocked-egress attempts and 0 teardown errors** — the same 26 failures in the
same 13 modules as before this change. Those tests still stop at the durable gate, so
they never reached the tokenizer; the fixture pre-empts the seam for whenever they are
repaired.

## Closeout verification (2026-08-28)

Current `origin/dev` at `473e7c9298` includes TASK-22000's queue-contract repair. The exact
acceptance file now passes **304 tests** in 380.03 seconds. The adjacent composer and durable
queue/recovery scope passes **54 tests** in 45.68 seconds. No production or test changes were
required in this closeout; this branch only reconciles TASK-21590's stale acceptance and status
metadata with the merged behavior. The repository derived-artifact preflight also passed all six
guards (CSS bundles, profile-owned paths, production diagnostics, backlog IDs, schema allowlist,
and index-plan pins).

ADR required: no. ADR-046 remains authoritative, and this verification introduced no new
architecture decision. The closeout surfaced no new incident-backed lesson beyond the testing
evidence already recorded above.
<!-- SECTION:NOTES:END -->
