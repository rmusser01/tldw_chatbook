# Task 8 implementation report

## Outcome

Task 8 splits the pre-gateway Console configuration snapshot from the final
execution context and wires the normal immediate/queued submit path to fresh
per-execution Library authority. The final context cannot be constructed without
an immutable authority and resolved destination. Task 9 destination classification
and Task 10 provider gating are deliberately not implemented.

ADR check: no new ADR was required. This task directly implements
[ADR-079](../../../backlog/decisions/079-console-library-conversation-authority.md),
status Accepted.

## Backlog and baseline evidence

- `backlog task edit 19900.2 -a @codex -s "In Progress" --plan "..."` completed,
  and `backlog task 19900.2 --plain` showed `In Progress`, assignee `@codex`, the
  Task 8 delivery link, and ADR-079 path/reason.
- Exact pre-feature baseline using `../../.venv/bin/python`:
  `Tests/Chat/test_console_chat_controller.py`,
  `Tests/Chat/test_console_turn_execution_context.py`, and
  `Tests/Chat/test_console_agent_bridge.py` — **441 passed, 2 inherited warnings**
  in 29.82 seconds.

## RED-first evidence

- All planned Task 8 tests were added or converted before production edits.
- Exact Step 3 RED stopped with **3 expected collection errors**: the authority,
  turn-context, and auto-RAG modules could not import the missing
  `ConsoleTurnConfigurationSnapshot`. Composer/harness collection did not proceed
  after those import failures. This was the intended missing-split failure.
- No production file had been modified when RED was recorded; only the required
  Backlog transition and tests differed from the base.

## Implementation and order evidence

- Moved the old fields, recursive detachment, constructor guard, and `capture`
  factory to frozen `ConsoleTurnConfigurationSnapshot`.
- Added frozen `ConsoleTurnExecutionContext(configuration, library_authority,
  resolved_destination)` with strict non-`None` type checks, boundary copies, and
  read-only compatibility properties for existing consumers.
- Immediate submit validates/admit-checks the draft before configuration capture.
  Queued entries capture only when the queue coordinator has claimed/dequeued and
  calls the ordinary submit path.
- The controller awaits fresh
  `library_policy_coordinator.capture_for_execution(session_id)` before building
  `ConsoleTurnLibraryAuthority`; a raised read becomes exact Never/Blocked,
  `source="unavailable"`, `error_code="policy_read_error"` rather than reusing the
  holder.
- Authority freezes `AUTOMATIC_LIBRARY_SOURCE_TYPES`, item-scope note/media IDs and
  conversation exclusion, Direct/RAG selector, provider/model/endpoint intent,
  and a fresh attempt ID. Later live selector, scope, provider, and policy changes
  do not mutate the accepted authority.
- Gateway resolution happens next. An injected immutable
  `resolution.resolved_destination` is used when supplied; the transitional
  existing resolution seam otherwise yields an empty credential-free identity
  and `ConsoleEgressClass.UNKNOWN`. No endpoint classification, provider switching,
  or provider composition was added.

The ordering regression observes the literal sequence
`configuration -> policy -> authority -> gateway -> rag`, where RAG receives the
complete final context. The queue regression holds the first turn open, admits a
second prompt, and observes no second config/policy capture until the claimed turn
executes.

## Mutation evidence

Each mutation was applied independently, its named protection was run, and the
production implementation was restored before the next probe:

- Moving immediate configuration capture before admission failed
  `test_immediate_capture_follows_admission_and_precedes_gateway` because the
  rejected blank draft produced a `configuration` event.
- Capturing configuration in `queue_prompt` before dequeue failed
  `test_queued_configuration_and_policy_capture_only_after_dequeue` with two
  captures where only the running turn's one capture was allowed.
- Reusing the cached holder instead of calling the coordinator failed
  `test_unavailable_fresh_read_defeats_cached_allowed_holder`: cached
  Automatic/Allowed survived instead of unavailable Never/Blocked.
- Allowing `None` authority/destination failed
  `test_final_context_requires_complete_authority_and_destination` with
  `DID NOT RAISE`.

After restoration, the exact four named tests passed **4/4**.

## Final verification

- Exact Task 8 Step 3 battery: **83 passed, 1 inherited Requests dependency
  warning** in 16.86 seconds.
- Scoped controller/turn-context/agent-bridge compatibility battery:
  **443 passed, 1 inherited Requests dependency warning** in 21.43 seconds.
- Scoped Ruff over every changed Python source/test: **all checks passed**.
- `git diff --check`: passed.
- Per repository and task instructions, no full suite, push, live profile, or user
  database was used.

## Self-review

- Confirmed configuration and policy capture occur after admission, while queue
  admission itself performs neither capture.
- Confirmed fresh policy capture precedes authority construction and no exception
  path falls back to cached Allowed authority.
- Confirmed the fixed automatic source tuple, item IDs, provider intent, selector,
  and final configuration are copied rather than aliased.
- Confirmed final construction occurs only after a ready gateway result and rejects
  incomplete authority/destination values.
- Confirmed compatibility accessors are properties only; the final type has no
  legacy `capture` constructor or optional authority/destination defaults.
- Confirmed the fallback resolved destination is conservative UNKNOWN and stores no
  endpoint or credential, leaving all Task 9 classification work untouched.
- Confirmed no built-in Library provider selection/reservation, automatic
  preparation state, schema, migration, sync/export, deprecated Settings, or
  Task 9+ behavior was added.
- No generalizable new incident beyond the existing testing/backlog lessons arose,
  so no lessons document was changed.

## Fix round 1 — complete contexts on every execution attempt

Review base: `76bcdc413`. The review finding was reproducible: normal submit
already constructed a complete final context, but retry, queued retry,
continue, regenerate, edit/resend, and provider-continuation recovery passed a
pre-gateway `ConsoleTurnConfigurationSnapshot` into provider execution. Both
provider-boundary methods also synthesized that incomplete snapshot when their
caller omitted the final context.

### RED evidence

- Before fix production edits, the exact Step 3 command finished with **9
  failed, 79 passed, 1 inherited Requests dependency warning**.
- Four failures were the parameterized retry/continue/regenerate/edit-resend
  real paths reaching the streaming boundary with configuration only.
- Queued retry failed after the queue had visibly reacquired a `HELD`
  reservation; continuation recovery failed at the agent boundary.
- The remaining three failures proved direct streaming, `_run_agent_reply`,
  and Library-provider composition accepted an incomplete snapshot.
- A second RED/green cycle proved that an absent policy coordinator reused the
  staged Allowed holder; the named regression now requires exact unavailable
  Never/Blocked fail-closed authority.

### Fix and ordering

- Added one async attempt finalizer used by all reviewed action/recovery paths:
  capture configuration after action admission or queue recovery claim, await a
  fresh policy capture, resolve the gateway destination, then construct the
  complete immutable `ConsoleTurnExecutionContext` only for a ready execution.
- Queue admission still captures nothing. A queued retry captures only from the
  recovery callback after `recover_and_drain` has reacquired its slot.
- Retry, continue, regenerate, and edit/resend now pass that complete context
  through payload construction and streaming. Continuation recovery does the
  same through history assembly and `_run_agent_reply`.
- Until Tasks 15/16 durably persist the original continuation attempt's frozen
  authority, continuation recovery performs a fresh fail-closed capture. It
  never invents the unavailable original snapshot or reuses a cached Allowed
  holder. A missing coordinator and a coordinator exception both produce exact
  Never/Blocked, `source="unavailable"`, `error_code="policy_read_error"`.
- `_stream_assistant_response_inner`, `_run_agent_reply`, and
  `_library_provider_for_context` now reject anything that is not an actual
  complete `ConsoleTurnExecutionContext` before provider or Library composition.
  The compatibility configuration alias remains only for proven read-only or
  pre-gateway consumers.
- Gateway resolution still uses the existing injected destination seam and
  conservative UNKNOWN fallback. No Task 9 destination classification, Task 10
  Library gating/provider selection, endpoint inference, or schema work was
  added.

### Mutation and compatibility evidence

- Mutating only the retry handoff back to
  `turn_context.configuration` failed
  `test_message_actions_thread_one_captured_context[retry]` at the real provider
  boundary. Restoring the complete handoff made the named test pass.
- Exact Step 3 GREEN: **89 passed, 1 inherited Requests dependency warning**.
- Controller/turn-context/agent-bridge compatibility trio: **443 passed, 1
  inherited warning**.
- Queue, citation-boundary, and provider-gateway compatibility excluding two
  sandbox-only localhost tests: **381 passed, 2 deselected, 1 inherited
  warning**.
- The excluded gateway tests both fail at `socket.bind` with `PermissionError`
  under this sandbox. A detached worktree at the pre-fix base reproduced both
  identical errors.
- The broader provider-continuation file has 27 setup/durability failures at
  `Local continuation intent is stale or unavailable; save and retry.` A
  representative failing test reproduced identically in the detached pre-fix
  worktree. The task-local real continuation recovery test supplies the durable
  boundary explicitly and passes.
- Scoped Ruff over all changed Python files and `git diff --check` pass.

### Fix-round self-review

- Re-enumerated every reviewed provider-executing path and confirmed each ready
  call reaches streaming/agent execution with the final type; no configuration-
  only fallback remains at either provider boundary.
- Confirmed action validation precedes capture, queued retry capture observes a
  held claim, fresh policy precedes gateway resolution, and final construction
  follows the ready gateway result.
- Confirmed fresh unavailable policy defeats a cached Allowed holder for both
  raised reads and missing coordinator wiring.
- Confirmed incomplete contexts fail before direct provider streaming, agent
  composition, and Library factory invocation.
- Confirmed missing-owner close races still return the existing session-closed
  result before context validation, while any live provider execution is strict.
- Confirmed the fix does not persist continuation authority early, classify
  destinations, reserve Library names, or gate Library providers; those remain
  owned by later frozen tasks.
