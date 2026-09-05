# TASK-31737: Tool-chain ownership and snapshot approval validation

Date: 2026-09-05. Base: `33a14b11c`, following the ordinary-send fix on
`codex/chatbook-snapshot-trace-fix`. The shared dirty checkout was untouched.
Architecture: [ADR-097](../decisions/097-console-reference-backed-semantic-trace-ledger.md),
especially ordered call boundaries and conversation/turn/run/call ownership.
No new schema, authority, capture policy or production build admission.

## Failure and fix

Mounted calculator approval previously failed with `trace_turn_unavailable`,
both after restoration and without any snapshot operations. The factory assumed
the final payload descriptor owned the saved user turn. Tool results are
provider-only artifacts, so that assumption does not hold for tool loops.

Tool continuations now validate their supplied saved-turn candidate against the
unique recorded first agent call in the same chain. Owner, segment, turn and
frozen policy must agree. The previous logical call must be response-bearing,
and its surface must remain current. `response_started` is allowed because the
agent intentionally queues response settlement; requiring `complete` races that
existing handoff. Fresh-send unsaved-turn rejection is unchanged.

Independent review reproduced an additional stale-run case: a newer run could
use the identical surface. A new regression failed before the correction.
Reservations now atomically append the already-defined `call_boundary` event,
and continuation must follow the latest such event in its segment. This checks
call order, not prompt equality or wall-clock timestamps. No history body is
copied into the event. Missing, foreign, policy-mismatched, ambiguous and stale
chains fail before a new reservation or provider dispatch.

## Automated verification

- The controller/agent/gateway regression uses real SQLite and the production
  trace factory, with only inference replaced. It initially failed while the
  older hand-built boundary passed. Both now pass, including distinct artifact
  and saved-response links and completed call sequences 0 and 1.
- Seven runtime scenarios cover valid continuation (including a new factory
  recovering from SQLite), missing chain, foreign turn, changed policy,
  changed surface, ambiguous origin and a newer run with the same surface.
- Final whole-module run across controller, runtime, repository, service and
  native reader: **374 passed, 1 baseline failure, 1 warning, 45.70s**.
  The failure is the previously independently reproduced unchanged-dev catalog
  count assertion (expects29, actual26), outside this fix. No full suite run.
- Independent re-review approved the corrected ordering check; its targeted
  verification passed20 tests with the existing requests-version warning.
- Ruff 0.16.6 introduces no diagnostics: runtime2, repository4, runtime-test1
  and controller-test16 inherited findings remain unchanged. Security rules
  are included. Changed-range formatting and whitespace checks pass. Whole-file
  lint debt is not reported as clean.
- Production-factory storage release gates passed both append and replacement
  scenarios (2 passed, 84.43s), five fresh databases each with200 turns.
  Median trace-owned bytes were293,280 and316,591, below the2MiB cap.
  Second-half/first-half byte ratios were0.9913 and0.9849; row ratios0.9929
  and0.9920. The ordered event preserves the existing linear-growth contract.

Evidence: `/private/tmp/chatbook-tool-{trace-red,chain-red,stale-run-red,order-green,chain-final-targeted}.log`.

## Measured live approval handoff

Fixture: `/private/tmp/llamacpp-chatbook-validation.3vIa7q`. Actual mounted
`TldwCli`, visible composer action and approval card, real calculator, production
Admin HTTP snapshot routes, supervisor/store and native llama-server. This is
mounted-app UAT, not physical-terminal input or a new browser walkthrough.

Official b10816 ARM64, supplied Gemma4 26B/4A Q4_K_M, CPU, context16384,
one slot, full-SWA enabled, reasoning disabled, output cap128. The scratch config
and all data paths were isolated before import. An OS fence denied real-home
writes and all egress except the two fixture localhost ports. `HOME` was unchanged.

1. The model requested calculator `17 * 19`; the actual card required approval.
2. Save/restart/restore completed with **6085 tokens**, while the card remained
   pending. Messages, selected durable records and provider settings stayed
   unchanged; the lifecycle caused no inference request.
3. The harness checked the sole builtin tool and exact arithmetic arguments,
   then pressed **Approve once**. The real tool returned323 and the model
   responded `The result of 17 * 19 is 323.`
4. The continuation reused **6054 tokens / processed73**. An identical captured
   payload sent directly to the cold native runtime reused0 / processed6127.
   Native prompt processing was1193.579ms warm versus55812.482ms cold.
5. Pause/Resume produced no snapshot operations or automatic send and preserved
   messages and selected durable records. App teardown reported no exception.
6. A separate actual calculator approval with **no lifecycle operations** passed.

SQLite inspection verified the two UAT calls have the same recorded turn and
run, sequences0/1, statescomplete, routesagent_first/tool_loop and two ordered
call-boundary events. The final diagnostic exception list is empty.

Evidence: `chatbook-tool-chain-final.log`, `chatbook-tool-chain-result.json`,
`chatbook-tool-chain-wire.json`, `chatbook-tool-chain-control-final.log`, separate
control wire/exception captures and pending/after-restore mounted SVGs.

## Limits and cleanup

- Production `TESTED_TEXT_BUILD_SHA256` remains empty. Only the disposable
  fixture admitted this exact candidate executable. These results do not establish
  general model, streaming, multimodal, concurrent-slot or fleet support.
  Chatbook payloads still omit `id_slot`; one-slot observation is not affinity.
- Pre-fix in-flight traces lack ordered call-boundary events. Their continuation
  fails closed; this change does not backfill chronology or silently disable capture.
- Origin uniqueness is a global scan; `LIMIT 2` bounds returned rows, not work.
  A query-only in-memory clone of the actual table/index schema with100,000
  synthetic calls measured100 samples: median2.199ms, p952.610ms, max3.516ms.
  This is not disk or end-to-end reservation latency. Indexing remains a scaling
  consideration; no schema expansion was made for this correction.
- The harness permanently deleted its two synthetic saved copies across the
  two successful lifecycle runs. Zero copies and14 historical receipts remain.
  Logs and synthetic databases are retained; user models, binaries and profiles
  were not changed.
  The owned native/API processes exited and both fixture ports have no listeners.

## PR2433 rebase and Qodo review follow-up

Rebased onto dev `66a1cbf8f`. The sole conflict was competing testing-lesson
appendices; both were retained. Range-diff confirmed unchanged product patches.

Qodo identified a missing actor binding, an unscoped test cursor, and a missing
validation-exception docstring. A new changed-actor regression failed with
`DID NOT RAISE ValueError`: the original chain accepted another actor. Durable
agent `run_id` now encodes the canonical UUIDv4 pair `actor_uuid:chain_uuid`.
The same pair recovers from SQLite across factory reconstruction; a changed
actor finds no authoritative origin and fails before reservation. Existing
opaque-string consumers and legacy shadow binding accept this identity. No
new schema or cache was introduced; ADR097 decision4 records the clarification.
Historical chain-only traces remain readable but cannot authorize continuations.

Both integration reads now use separate shared transaction contexts, with no
raw cursor retained across the awaited second send. `get_run_origin` documents
its ValueError contract. Independent read-only review approved the follow-up
with no findings. Fresh whole targeted modules: **375 passed, 1 known baseline
catalog failure, 1 warning, 52.85s**. No introduced Ruff/security findings;
changed-range formatting and diff checks pass. Earlier live native UAT above
was not rerun for this review follow-up; the real controller/agent/factory
regressions were rerun with inference replaced.

Fresh post-review storage release gates: **2 passed, 4 deselected, 1 warning,
89.42s**, preserving both append and replacement growth requirements.
