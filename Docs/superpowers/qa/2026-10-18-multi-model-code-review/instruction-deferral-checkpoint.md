# Native instruction-deferral checkpoint repair, 2026-10-29

The runtime emitted tool-result stubs when loading project instructions, but left
the corresponding canonical calls pending. The next valid hosted response then
failed continuation validation. The defect is reproduced and repaired; this is
not a successful AC24 qualification.

## Evidence and attribution

The failed run at `b35bb489ea`, primary run
`2804f4b41a9e4ff8aa10a55ce75fbdf2`, retained an active revision24 checkpoint with
eight rounds and four pending `code_review_read` calls in its last round. Request25
completed at07:15:21.993Z; no tool dispatch steps intervened before request26
began at07:15:22.137Z and ended with the generic error at07:15:25.847Z. The saved
checkpoint was read through an immutable SQLite connection.

One local diagnostic used the production Console/provider builder, the same42
tool schemas, model `unsloth/Qwen3.8-27B-GGUF:Q8_0`, and max_tokens4096. Splash
returned a complete native read call, a clean terminal stream, and usable usage:
11,975 input +80 output =12,055 tokens. No tools or children executed.

Replaying that captured valid response entirely offline gives:

| Previous state | Result |
| --- | --- |
| No checkpoint | Accepted |
| Exact saved eight-round checkpoint | `ContinuationValidationError` |
| Same checkpoint after normal failed-call transitions for four deferrals | Accepted ninth round |

The historical stream bytes and original exception are absent. This proves a
reachable defect with the exact retained state, not the contents of the missing
historical response. Timing/state strongly support this boundary; the supplied
thermal transition and later optional `GET /props`404 do not identify its cause.

## Minimal repair

Before requesting another response, the deferral branch now uses the existing
`ToolCallFinished` persistence and acknowledgment path for every original call,
with state `failed` and the same fixed deferral result already sent on the wire.
Deferred calls never execute or enter permission review. A persistence or
acknowledgment refusal stops the run before another request. A newly issued call
still undergoes normal review and dispatch.

Canonical validation, accepted review decisions, sealing, accounting, permissions
and child limits are unchanged. No server/provider workaround is introduced.
Existing ADR069/179/281 contracts apply; no new ADR is required.

## Verification

- Eight new tests failed before repair at continuation validation. All eight pass
  afterward: one/four deferred calls, final/tool responses, and first/later
  persistence/acknowledgment refusals. Responses use the real hosted SSE parser
  and continuation candidate builder.
- New tests plus existing continuation-runtime and Console request-grammar files:
  158 passed, one failed. The private-progress DB test fails identically with the
  original HEAD runtime function; it is not claimed fixed.
- Seven relevant preparation tests pass. The unrelated profile-guidance test was
  excluded from this focused selection.
- Nine offline controls using installed Splash1.3.2's stream emitter pass,
  including normal text/tools and explicit truncated/malformed-tool failures.
  These are scripted controls, not additional live runs.
- New test lint/format and whitespace pass. Runtime lint has the same21 findings
  as HEAD, zero additions. Its one pre-existing formatter difference is unchanged;
  the modified block is formatted. Self-review confirms persistence and
  acknowledgment precede local state advancement and the next request.

The earlier request-grammar regression supplied plain `ModelTurn` responses with
no canonical checkpoint. It could pass while this defect remained. The new
regression carries the checkpoint through every response.

## Retention and scope

Raw local response, request receipt, emitter controls, immutable-checkpoint replay,
RED/GREEN/baseline logs and static comparisons remain separately under
`/private/tmp/task35582-splash-reproduction/`. The original failed run remains at
`/private/tmp/task35582-v1-final60/attempt-wu1p3l55`, unmodified. No private payload
archive is added to the PR.

Additional actual usage: one local generation, zero cloud requests/children.
Cumulative actual usage:413 generation requests/28 children. Preparation attempts
blocked by the test network guard made no server request; their receipts remain
separate. No qualification, automatic retry, budget extension or merge occurred.
AC24 stays open, the task In Progress, and PR3286 draft. The separate
[dev integration report](dev-sync-2026-10-29.md) discloses existing size-test
failures; this repair does not relabel them green.
