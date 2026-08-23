# Task 9 implementation report

## Outcome

Task 9 classifies the gateway-resolved effective endpoint conservatively and
records a separate, non-durable disclosure state on each live Console session.
Classification uses only the normalized endpoint: loopback/local transports are
on-device; literal private/link-local/ULA addresses are private network; literal
global addresses and exact canonical cloud origins are public network; everything
unresolved, custom, missing, or malformed is unknown. Provider names and API-key
presence never influence the result.

ADR check: no new ADR was required. This task directly implements
[ADR-079](../../../backlog/decisions/079-console-library-conversation-authority.md),
status Accepted. `TASK-19900.2` remains In Progress with its acceptance criteria
unchecked for Task 10 and Task 11.

## RED-first evidence

- All intended Task 9 destination, gateway, policy-axis, and settlement tests
  were authored before the original production edits.
- The initial combined RED stopped with exit 4 and one collection error because
  `console_library_destination` did not exist. Split causation runs showed the
  same missing-module collection error for the destination/store tests, **4/4
  gateway failures** because `resolved_destination` was absent, and **4/4
  authority failures** because the session runtime record was absent.
- A settlement-publication refinement failed **1/1** before settlement moved
  ahead of completion subscriber publication.
- A malformed-userinfo refinement failed **2 tests** while 24 table cases passed:
  malformed userinfo was being stripped to an otherwise canonical cloud origin.
  It now fails closed to the bounded unknown identity.

## Implementation

- Added standard-library-only parsing with `urllib.parse` and `ipaddress` after
  gateway endpoint normalization. The endpoint identity contains only normalized
  scheme, host, and non-default port. Default `:80`/`:443` ports collapse;
  userinfo, path, query, and fragment are discarded.
- Invalid scheme/host/port/userinfo/control input, absent endpoints, overlong
  input, and overlong hostnames produce the bounded `external/unknown` identity.
  No DNS lookup or provider/API-key heuristic is used.
- Wrapped the public gateway resolution seam so every ready-return branch,
  including llama.cpp and configured/built-in cloud paths, receives one immutable
  `ConsoleResolvedDestination` after its effective endpoint is finalized.
- Added a per-session runtime record with current destination, last resolved
  identity, and optional disclosure. A first-ever external resolution does not
  invent an on-device transition. When either frozen policy axis can place
  Library data in the request, an on-device-to-non-device change raises the
  disclosure; another identity change replaces or clears it, and terminal
  settlement clears it while retaining the last destination.
- The Task 8 controller records this state only after constructing the complete
  immutable execution context. The store settles it on complete, regenerated
  complete, failed, and stopped assistant paths before completion publication.
  Durable Library policy is never modified.

## Files and contract scope

The planned destination module and gateway/test files were changed. The file
list expanded minimally to `console_chat_store.py`, `console_chat_controller.py`,
`test_console_chat_store.py`, and `test_console_turn_library_authority.py` because
Task 9 explicitly requires persistent live-session state, final-context wiring,
policy-combination proof, and centralized settlement semantics. The report and
shared progress ledger were also updated as required. There is no frozen-contract
deviation and no Task 10 provider gating, provider construction, tool reservation,
schema, sync, import, or export work.

## Mutation and negative evidence

Each production mutation was restored before the next probe:

- Treating an arbitrary custom hostname as public failed its fail-closed case
  (**1 failed, 34 passed**).
- Disabling the on-device transition guard failed the pure transition case and
  all three eligible policy-axis combinations (**4 failed, 1 passed**).
- Removing centralized completion settlement failed complete,
  regenerated-complete, and subscriber-order cases (**3 failed, 2 passed**).
- Building identity from the raw configured URL failed both credential-stripping
  probes (**2 failed**).
- Settling the active session instead of the message-owning session failed the
  navigation/isolation regression (**1 failed**); the restored test passes.

The privacy probes use URL userinfo, path, query/fragment secrets, and a separate
configured API-key canary. None appears in the destination repr, identity key, or
runtime state. Stable identity remains unchanged when only those discarded URL
parts change.

## Verification

- Destination plus full gateway coverage excluding two sandbox-only socket tests:
  **301 passed, 2 deselected, 1 inherited warning**.
- Authority, turn-context, and store coverage: **320 passed, 1 inherited
  warning**.
- Controller and agent-bridge compatibility: **427 passed, 1 inherited warning**.
- Task 8 UI authority consumers: **63 passed, 1 inherited warning**.
- Dispatch checkpoint repository/codecs: **52 passed, 1 inherited warning**.
- Scoped Ruff over all changed Python source/tests: all checks passed.
- `git diff --check`: passed.

The two deselected gateway tests bind localhost sockets and are unavailable in
this sandbox (`PermissionError` at `socket.bind`); their exclusion and identical
Task 8 baseline limitation are already documented. Every executed battery emits
only the pre-existing Requests/urllib3/charset dependency-version warning. Per
repository and task instructions, no full suite, push, live profile, or user
database was used.

## Self-review

- Confirmed canonical-cloud classification comes from the sanitized effective
  origin, not provider identity, model, API key, or DNS.
- Confirmed malformed and adversarial inputs cannot create unbounded state or
  leak userinfo, path, query, fragment, or API keys.
- Confirmed the public gateway wrapper attaches classification to all return
  branches after normalization and readiness resolution.
- Confirmed first external resolution is not treated as a transition; disclosure
  replacement, return-to-device clearing, settlement, completion subscriber
  ordering, session navigation, and cross-session isolation are covered.
- Confirmed runtime state is session-owned, non-synced, and independent of the
  durable policy holder; the complete Task 8 authority/context boundary remains
  intact.
- No generalizable new incident beyond the existing testing/backlog lessons arose,
  so no lessons document was changed.
