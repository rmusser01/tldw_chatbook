# ADR-174: Consume durable sharing operations and versioned Notes link deletion

Status: Accepted
Date: 2026-09-20
Related task: TASK-32881
Amends: [ADR-073](073-notes-sync-round-trip-and-interoperability-constraints.md)

## Context

The server release replaces ephemeral clone jobs with recipient-owned operation
receipts, returns a source page instead of a list, and requires an observed link
version for deletion when Notes Sync is active. Chatbook's existing HTTP methods
and both Sharing service families cannot express these contracts.

## Decision

- Serialize canonical clone `name`, accepting `new_name` as a Python input alias.
  A `CloneWorkspaceRequest` owns one excluded `idempotency_key`, generated once
  on construction or supplied by its caller. Reusing the request replays the same
  admission; service callers can retain and supply a key explicitly. Never create
  keys inside transport dispatch or polling. An explicit new logical clone uses
  a new key; an uncertain response must reuse the existing request/key.
  The Sharing panel retains keys on the app, keyed by configured server ID and
  base URL, stable authenticated-user authority, share, and server-normalized name.
  The existing authority resolver validates its captured context and resolves user
  identity independently of credential rotation. Retries, temporary input changes,
  and panel remounts replay the same request. At 100 retained intents, new intents
  fail closed without evicting uncertain requests; existing retries still work.
  `Start another clone` explicitly retires only the active account's selected
  identity before another admission;
  `Clone / retry` otherwise replays even a terminal receipt. Keys are not credentials
  and are independently scoped to each recipient by each server.
- Preserve the full canonical operation receipt (identity, progress, result,
  readiness, warnings, error, retryability, poll URL). Add a scoped receipt read
  method constructed from share and operation IDs, never dispatching an arbitrary
  server-provided URL. Legacy `job_id` responses remain parseable. Do not conflate
  queued with copied or automatically retry a failed clone as a new admission.
  Per the server's approved 2026-08-25 Durable Shared Workspace Clone Jobs Design,
  receipt reads and matching admission replay remain valid after share revocation.
- Add an explicit paginated source method with offset, limit, query and state.
  Preserve pagination, summary and partial errors. Accept legacy list/source field
  names; expose canonical `source_id`/`origin_url` plus compatibility accessors.
  Existing list methods traverse pages so old callers do not silently see only
  the first page. Each next page is separately server-authorized.
- Forward optional `dataset_id`, selected `expected_version`, `idempotency_key`,
  and `reason` through connected Notes scope/service/HTTP boundaries. Never read
  a newer version to bypass a conflict. Legacy non-Sync deletion can omit them;
  Sync's 428 and stale-version errors remain visible to callers.

## Alternatives and consequences

Weakening the server's auth or optimistic-concurrency checks would sacrifice the
new contracts; it is rejected. Returning just page items from every method would
hide completeness/partial failures; explicit page APIs preserve that information.
Breaking all list consumers is unnecessary, so list convenience methods remain.
No local database/schema change, new clone worker, automatic polling timer or UI
redesign is introduced. Callers needing reload-persistent replay must retain the
request key and operation receipt in their own existing operation lifecycle.

## Verification

Exercise real httpx MockTransport through TLDWAPIClient for canonical headers/body,
replay after a lost response, operation polling, pages/aliases and exact selected
Notes deletion preconditions; cover both Sharing families and Notes scope policy.
Run only focused suites, scoped lint/format and Bandit; record results in TASK-32881.
