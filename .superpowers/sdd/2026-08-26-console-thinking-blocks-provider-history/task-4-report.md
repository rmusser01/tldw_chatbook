# Task 4 Report — Provider/history projection

## Status

Complete. Task 4 resolves optional displayable-thinking replay and mandatory
ADR-063 continuation in the same owner-atomic prepared request. No Backlog/UI or
exchange child work was performed.

ADR required: no new ADR. ADR-090 and ADR-063 already govern the thinking and
continuation boundaries implemented here.

## Carried prerequisite

The assigned base covered helper-level token behavior but three production
settlement paths still lost the replacement generation token. RED production-path
tests reproduced the fault for normal accepted-turn rollback, durable postcommit
rollback, and Stop settlement rollback. `ConsoleChatStore` now retains the first
token issued while an exact recovery owner is in flight; generic release/restore
consumes that retained token. Pre-token refusal preserves the prior token, rollback
invalidates only a still-current matching replacement, and a newer token survives.

Committed separately as `5f31efc6d6` (`fix: retain recovery generation token across
settlement`). Focused GREEN: 10 passed, including all three production paths with
and without a newer token plus the prior helper/preflight cases.

## Implementation

- Added canonical provider-history types and a pure resolver for normalized
  Auto/Include/Exclude policy, effective Required overlay, exact target
  compatibility, and content-free strict Include refusal.
- Added the initial exact llama.cpp/vLLM `start_anchored_think` serializer. It
  reconstructs `<think>…</think>` only in the frozen provider wire artifact;
  semantic assistant content remains the visible answer.
- Extended prepared conversation units with thinking owner groups beside
  continuation groups. Exact owner markers bind both sidecars, and trimming or
  compaction removes a whole visible/thinking/continuation owner unit.
- Provider preparation now resolves both histories before counting, stores one
  immutable wire payload, and uses that same payload for final token accounting and
  dispatch. Strict Include fails before provider contact when a compatible source
  claims an unsafe/unknown serializer.
- `ConsoleChatSession` now owns normalized `thinking_history_policy`. Missing/null/
  invalid values resolve to Auto; temporary sessions, first persistence, durable
  hydration, durable-turn creation, temporary promotion, and screen-state restore
  round-trip the live value without a schema change. Effective Required never writes
  back over the saved preference.
- Direct and agent paths construct supported sidecars only from active-path assistant
  owners and read the live session policy. Compaction preserves the private owner
  marker and policy/groups. Auxiliary summary and citation-repair streams pass no
  sidecars and explicitly discard typed displayable/proprietary thinking events.
- Proprietary, stopped, failed, opaque, incompatible, and non-canonical application
  copies never become replay groups. No generic cross-provider assistant-text
  translation was added.

## Verification

- RED: new history suite initially failed collection on the missing thinking-owner
  interface; session lifecycle had 3 expected failures; direct/agent integration had
  2 expected failures; the carried production rollback slice failed the no-newer-token
  cases in all three outer paths.
- Task 4 history/prepared/provider suite: **389 passed, 2 skipped** (expected loopback
  permission skips).
- Controller/agent/capture/session regression suite: **509 passed**.
- Dispatch recovery adjacent regression suite: **65 passed**.
- Focused history + session lifecycle: **23 passed**.
- Focused direct/agent/auxiliary/opaque slice: **4 passed**.
- Carried rollback production/helper slice: **10 passed**.
- Ruff format check, Ruff lint, and `git diff --check`: passed for the planned files
  plus the narrow session/persistence expansion.

No full repository suite was run, per repository and task instructions. No known
functional concern remains; the only expected skips were sandbox loopback-listener
permissions.
