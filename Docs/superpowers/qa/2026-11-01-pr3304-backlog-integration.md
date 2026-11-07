# PR3304 backlog integration — 2026-11-01

Source head: `683fd05b866855820c9075810ed2cb93aa2db93e`.
Integrated base: `f34510953e5d174e74625fef6fbc33f9ba87f197`.

This record qualifies the streaming-default PR and its integration repairs. Hosted CI and size limits are waived by the user for backlog clearance; independent `superpowers:requesting-code-review` replaces Qodo. Draft PRs and PR3267/3297 are excluded. No release is part of this work.

## Changes and review

Preserved current dev's credential/readiness, reasoning ownership, workflow and census changes while rebasing all functional commits. Four source commits containing only obsolete size-pin adjustments were omitted; changed owners are pinned at their exact measured sizes under the user waiver.

The independent review found and the integration fixed:

- Separate admission and settlement for both physical generations; an unknown first call retains its automatic-budget reservation and prevents an unbudgeted retry.
- Successful application/json responses to streamed requests normalize content, native tools and usage through the shared adapter with one HTTP request and the normal response-close owner.
- Immutable physical usage snapshots plus a bounded two-leg reporting view. Validate each raw payload before summing; preserve known first-call usage/cache counts, explicit zero and live-output presence semantics. Unknown or malformed counts do not become fabricated zero usage.
- Capture-on owned Retry preserves the accepted preparation and frozen request. Generic agent fallbacks retain actor/chain identity and ordered call sequences so tool work before and after a retry can continue.
- The failed-call trace guard permits only the verified fallback overlay's literal streaming True-to-False change. Provider, endpoint, model, all other generation settings, reasoning, system composition and tools must still match.

Existing ADR097, ADR131, ADR134 and ADR156 govern these corrections; no new ADR or authority boundary is introduced. Initial and final independent reports are retained under the isolated worktree parent (`pr3304-initial-review.md` in the pr3298 parent; `pr3304-accounting-design-review.md` and `pr3304-integration-review.md` in the pr3304 parent). The final external report/handoff records the exact committed head.

## Verification

Receipts are under the isolated `pr3304-resync` worktree parent. Every cited run contains its command, JUnit XML, logs and before/after source hashes; all reported unchanged sources. Counts overlap and must not be added as unique coverage.

- `r-aa9ba238`: causal failures before the budget, usage and JSON repairs.
- `r-63863f6e`: all 51 existing automatic-provider-budget tests passed. Four other failures in that intermediate run were capture/response-close assertion mistakes subsequently corrected and covered below.
- `r-f1e91764`: 695 passed, four failed in an intermediate 699-test run. The two capture-sequence assertions were corrected and subsequently exercised with the real trace factory. The two DeepSeek base-URL expectation failures reproduce on exact base f345 in `pr3298-resync/r-395cec11`; no product guard or assertion was changed to hide them. This run includes 34 custom-credential, 13 stranded-key readiness, 89 CI queue contracts and 71 streaming-default config tests, all passing.
- `r-8bdd9164`: 21 passed, including real Custom/vLLM capture, tool calls before/after fallback, owned recovery, foreign-owner rejection and strict transport-only transition controls.
- `r-66c563d6`: 200 passed, zero failed/skipped. Full affected accounting, trace runtime, surface recovery, provenance, refused-stream and local-adapter modules. Includes the legacy direct llama fallback control, calculator result and same-run ordered traces.
- `r-8fe21639`: 15 passed, one failed. The progressive Console reply, persisted Custom retry, explicit Off-to-new-chat journey, saved-Off and legacy-Off overrides, current census, six exact size pins and three fallback controls passed. The unset Settings child's behavioral assertions passed but its native-executor retirement failed at the unchanged three-second deadline.
- `r-966d53c7` and `r-d772a011`: the unset cleanup failure reproduced in isolation. Content-free native stacks identified initial ChaChaNotes FTS backfill seeding built-in cards, with a spend read waiting on the same storage initialization.
- `r-cf4e1867`: the corrected unset Settings case passed, including teardown. It now observes the existing open boot gate until that specific finite backfill leaves pending/in-flight before its early return. No runtime guard, deadline, fixture ownership rule or behavior assertion was weakened. Temporary instrumentation was removed. UI qualification is the passing cases across these receipts, not a claimed all-green 16-test run.

Ruff comparison across all 19 changed Python files found no new diagnostics versus f345 (existing duplicate-import messages normalized only for moved line references). The final test-only cleanup change also passes Ruff. Whitespace checks pass. No full suite was run.

The old source head's hosted derived-artifact job failed because prerequisite lanes were cancelled; its artifact checks are not evidence for this integrated tree. No required workflow was manually dispatched.

## Limits retained

Named `custom-hosted` fallback remains excluded under TASK37762. OpenAI's existing streamed-error status loss/retries remain TASK37700; the shared first-token deadline remains TASK37760; direct llama usage remains TASK37181. Unknown failed manual work is not claimed as fully measured. No paid-provider or release qualification is claimed.
