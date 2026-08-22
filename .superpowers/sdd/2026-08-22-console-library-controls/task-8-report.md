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
