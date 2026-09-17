# Console model settings repair implementation plan

> Execute independent repairs in parallel, preserving exclusive file ownership;
> integrate and review the complete result in the current working checkout.

**Goal:** Fix every confirmed defect in the Console model-settings investigation
and the directly adjacent failures found during implementation and review.

**Approved design:** The user's investigation findings and follow-up requirement
are recorded in `Docs/Development/console-model-modal-investigation-2026-09-16.md`.
The user authorized implementation and a nearby-bug sweep. Preserve conversation
values on same-target Apply, retain exact endpoint identity during discovery,
keep credential readiness off the UI thread, and resolve context capacity from
server metadata, then model/API defaults, then exactly 32,000 tokens.

**Architecture:** Keep the current conversation/default ownership and endpoint
registry. Introduce a shared context-limit resolver and a bounded gateway-owned
metadata cache; both UI estimates and immutable provider request snapshots use
that resolution. Credential readiness reads a background-populated snapshot.

**Constraints:** Python 3.12, Textual 8; no new dependencies. No real credentials
in diagnostics; isolate profiles and provider I/O. Preserve unrelated working
tree changes. Run targeted tests only. The user subsequently authorized a PR
against dev; publish the isolated repair branch without merging or a full-suite run.

ADR required: yes (context capacity); no new ADR for the three corrective fixes.
ADR path: `backlog/decisions/052-console-conversation-memory-and-compaction-policy.md`
(amendment), with existing ADR-095, ADR-012, and ADR-146 governing the repairs.
Reason: server-first capacity and the estimated system fallback change the
unknown-window behavior; the remaining fixes restore accepted contracts.

## 1. TASK-32707: retain conversation settings

- [x] Promote the four red mounted reopen/apply/default reproductions to regular
  tests. Retain checks of live values and config reload.
- [x] Correct `ConsoleChatController.rebase_console_settings_draft` same-target
  handling; do not discard fields hidden by the quick modal. Preserve explicit
  Inherit edits and established target-change behavior.
- [x] Verify repeated Apply, default profiles, provider/model A-B-A, hidden
  fields, and source endpoint retention; fix nearby defects with regressions.
- [x] Run the relevant store/controller/modal flow tests and scoped Ruff.

## 2. TASK-32566: endpoint discovery

- [x] Extend the existing task's AC and plan before implementation.
- [x] Promote the red New endpoint/Create parent-modal reproduction; add exact
  identity, stale result, and credential-scoping cases.
- [x] Fix `console_settings_modal.py` discovery and evidence ownership, routing
  endpoint interpretation through the entry family without dropping entry ID.
- [x] Fix directly related button/probe failures, preserving saved entries on
  probe failure. Coordinate any `chat_screen.py` edits with the primary agent.
- [x] Run mounted parent-child tests and existing discovery/entry tests; lint.

## 3. TASK-32708: responsive credential readiness

- [x] Turn the Keychain heartbeat/cache reproductions into behavior tests using
  controlled I/O gates; do not contact the real Keychain.
- [x] Resolve subscription credential presence in a bounded background path
  used by `provider_readiness.py`; preserve synchronous send authentication.
- [x] Cache success/failure from completion, single-flight concurrent work, and
  cover invalid files, timeout, TTL expiry, and refreshed readiness.
- [x] Run subscription/readiness/UI tests; inspect adjacent blocking callers.

## 4. TASK-32709: consistent context capacity

- [x] Amend ADR-052 before implementation for the approved estimated fallback.
- [x] Add priority/provenance tests for remote/model/API/system limits, invalid
  integers, exact models vs generic prefixes, and display/runtime consistency.
- [x] Implement shared pure resolution; keep 32,000 fallback estimated. Use
  existing model capabilities and defaults instead of stale Console tables.
- [x] Add bounded server discovery using provider-supported metadata endpoints,
  authorized credentials, response-size limits, total deadlines, single-flight
  and bounded identity-scoped cache. Do not persist endpoints or credentials
  into conversation metadata.
- [x] Carry capacity/provenance in provider resolution and use it in request
  preparation/compaction. Refresh the Console estimate off-thread/off-loop and
  reject stale endpoint/model results.
- [x] Verify priority, no-metadata/error fallback, cancellation, cache isolation,
  output reservation, explicit user budget, modal and next-send behavior.

## 5. Integration and nearby-bug review

- [x] Review changes across file boundaries and inspect nearby state transitions:
  reopens, switching provider/model/endpoint, save failure, delayed completion,
  stale cache and default-to-new-chat publication.
- [x] Add a failing behavior test before fixing each additional confirmed defect;
  record any limitation with exact evidence rather than broadening speculatively.
- [x] Run targeted integrated tests and mounted UI flows, then scoped lint and
  format checks. Avoid rewriting the generated CSS or unrelated files.
- [x] Update investigation notes, user docs, task AC/implementation notes, and
  any incident-backed lesson; mark Done through the CLI only after verification.
