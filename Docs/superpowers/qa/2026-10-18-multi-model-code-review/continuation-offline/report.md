# Offline continuation request repair

The two reproduced request-shape defects are fixed. PR #3286 remains draft,
TASK-35582 remains In Progress and AC24 remains unchecked. This follow-up made
**zero live provider requests and zero product child launches**. Cumulative live
spending remains 338 requests / 23 children. No budget request or live retry.

## Reproduction and fix

The new regression executes the real agent loop, native Console serializer,
immutable gateway request preparation, dispatcher parameter projection and strict
custom-hosted payload builder. Model responses and the context-preparation decision
are scripted; the HTTP transport rejects all network attempts. This is offline
protocol evidence, not a live qualification.

The original producer emits a `name` field on native deferral results. The strict
builder rejects the next request as `tool result is malformed or orphaned`.
Native calls now omit that fenced-only field; ID-less fenced calls keep their
names. Call IDs, result order and the fixed deferral text are unchanged.

That correction exposed a second rejection on the same path: the internal
`PROJECT_INSTRUCTION_ROW_KEY` remained on the context row and the strict builder
reported `message fields are unsupported`. The existing prepared serializer now
removes that key alongside other internal metadata. Original runtime rows and
delivery receipts keep it; serialized counting and dispatch use the same copy.
No provider validator, permission, accounting or child budget was relaxed.

The complete control now reaches `done`: both original calls are deferred without
review or execution, separate instruction context reaches the next request, the
reissued read executes once, and its tool result survives the following request.
Preparing the payload does not mutate the original messages. Existing fenced
coverage preserves tool names and result/context order.

## Checks and limits

- Full-path pre-fix reproduction: one failure at strict tool-result validation.
- Name-only correction: one failure at strict context-field validation.
- Both corrections: the same full-path regression passes.
- Chat grammar and immutable request tests: **103 passed**.
- Agent preparation and instruction runtime: **76 passed**, one existing profile
  selection failure. The same guidance test fails with both original HEAD source
  files restored (`raw_source_selection_changed`).
- Strict hosted payload and instruction-persistence controls: **12 passed**, two
  existing profile-selection failures. Those same two failures reproduce with
  the original HEAD producer and serializer functions restored in a private test
  process. No baseline repair was made.
- Total final targeted selection: **191 passed, three classified baseline failures**.
  The single regression and diagnostic reruns overlap these totals.
- All four changed Python files parse and pass formatter checks; whitespace passes.
  Lint comparison: zero new findings, seven pre-existing, one import-order finding
  removed. The prepared-request formatter also wraps one existing long type alias.
- [Scoped review](review.md): no actionable findings. Earlier feature reviews and
  live evidence were not repeated.

The failed live run retained only metadata for the rejected request. These tests
prove both defects and the repaired offline path; they cannot retroactively prove
which exact bytes caused that live failure. They also do not establish live V1
completion, unified mapping, or clean live usage settlement. See the unchanged
[final-run report](../final60/report.md). The separate optional Windows GGUF CI
failure remains outside this fix.

Before publication, the prior documentation head `8f1ad19509` was rechecked:
its [normal CI](https://github.com/rmusser01/tldw_chatbook/actions/runs/40798969952)
failed UI shard 17 because
`test_applying_on_anthropic_submits_none_of_the_hidden_fields` timed out at
180 seconds. The required aggregate consequently failed. This test was not
rerun or changed; the new source commit requires its own normal PR CI result.

ADR required: no new ADR. Existing ADR-069 and ADR-281 apply: this restores the
existing protocol-safe project-context and review continuation contracts.

Raw red/green, baseline and static receipts remain in
`/private/tmp/task35582-continuation-offline/`. A separately preserved archive and
its hash are listed in `retention.json`; raw logs are not added to the PR.
