# Backend Parity Phase Tracker Design

## Purpose

Create a live tracker for completing the remaining backend server-parity phase and handing stable contracts to the UX developer. The tracker must show what is safe to build against, what backend work remains, which verification commands prove readiness, and which items are intentionally deferred.

The tracker is not a replacement for the roadmap in `2026-04-29-backend-server-parity-handoff-roadmap-design.md`. It is the operational checklist that keeps the roadmap honest while Tranche 3, Tranche 4, and the UX handoff packet finish.

## Current Context

The stale event/notification review findings are closed in the current `dev` branch:

- Event state models include `authenticated_principal_id` in `EventCursor`, `NormalizedEventRecord`, and `EventDedupeKey`.
- Production server event scope derives principal identity from active server context before observers start.
- `EventStateRepository.record_event_and_advance_processed_cursor()` atomically inserts event rows, dedupe rows, and processed cursor state.
- `EventCursorStore` is documented and tested as a process-local test/compatibility implementation, not the production event authority.
- `ClientNotificationsDB` remains the authoritative local notification inbox; no parallel local notification database should be introduced.

Recent committed foundation checkpoints:

- `35b9b19e` Add server parity event and sync foundations
- `a68307df` Wire server parity state repositories
- `4391e93a` Wire durable server notification events
- `604c4b98` Scope server event state by credential principal
- `84b28848` Cover atomic event cursor rollback
- `af503adb` Route research mirror reports through sync scope

## Tracker Location

Create the live tracker as:

`Docs/superpowers/trackers/backend-parity-phase-tracker.md`

Reasoning:

- A tracker is updated often, so it should not churn the design roadmap.
- It needs to be readable by backend and UX contributors without interpreting implementation files.
- It should survive current UI churn and remain backend-contract focused.

If `Docs/superpowers/trackers/` does not exist, create it when implementing the tracker.

## Tracker Status Values

Use a small fixed status vocabulary:

- `done`: code, contract, and tests are committed.
- `ready-for-ux`: backend contract is stable enough for UX work; implementation may still have non-blocking hardening.
- `in-progress`: active backend work exists but the contract is not ready.
- `security-blocked`: non-security foundation code exists, but a security-affecting credential, auth, redaction, logout, or server-switch row is not complete; dependent production work must not consume the seam as complete.
- `blocked`: cannot progress without another tranche, server API, or ownership decision.
- `deferred`: explicitly out of scope for this phase.
- `not-started`: no committed implementation or contract yet.

Each row must include enough evidence to justify its status. A row cannot be `done` or `ready-for-ux` without a committed test or contract reference.
Security-affecting auth or credential rows are gate blockers: if any such row is not `done`, the Connection/Auth gate is `security-blocked`, not `done`.

## Tracker Schema

The live tracker should use Markdown tables with these columns:

| Field | Meaning |
| --- | --- |
| `Area` | Tranche, shared foundation, or domain. |
| `Item` | Specific deliverable. |
| `Status` | One of the fixed status values. |
| `UX readiness` | `safe`, `partial`, `blocked`, or `not-applicable`. |
| `Source authority` | `local`, `server`, `workspace`, `mixed`, `remote-only`, or `not-applicable`. |
| `Backend owner` | Person, lane, or `unassigned`. |
| `Contract path` | Spec, schema, or service contract file. |
| `Contract version` | Version string from the contract, or `none` if the row is not contract-bearing. |
| `Stability` | `stable`, `provisional`, `experimental`, `blocked`, or `not-applicable`. |
| `Implementation evidence` | Commit hash, module, or test proving the row. |
| `Remaining work` | Concrete next action, not a vague theme. |
| `Blocker` | Missing API, dependency, owner conflict, or `none`. |

Rows should be small enough that a contributor can claim or finish one row without editing unrelated code.
Rows marked `ready-for-ux` must have `Source authority`, `Contract version`, and `Stability` populated with non-placeholder values.

## Required Sections

### Phase Gate Summary

Track Tranche 1 through Tranche 5 at the gate level:

- Connection/auth gate.
- Event/notification gate.
- Sync/mirror gate.
- Domain edge closure gate.
- UX handoff packet gate.

Each gate row should state whether dependent work can consume the seam and list the focused test command that proves the gate.

### Shared Foundations

Track shared seams separately from domains:

- Active server and capability authority.
- Credential storage and global sign-out.
- Event identity, observer lifecycle, cursor/dedupe, retention, and presentation projection.
- `ClientNotificationsDB` local inbox authority.
- `SyncStateRepository` identity, cursor, mirror, conflict, and outbox-disabled state.
- Provider migration audit and raw-client-builder guard.

This section prevents domain work from creating duplicate authorities.

### Domain Work Matrix

Track each domain from the roadmap:

- Chat.
- Media/Reading.
- Notes/Workspaces.
- Writing.
- Research.
- Study/Evaluations.
- RAG/Embeddings.
- Audio/Voice.
- Remote-only utility rollup.
- Sharing.
- Web clipper.
- Translation.
- Server tools.
- Text2SQL.
- Server skills.
- Claims.
- Meetings.
- Outputs.
- Kanban.
- Prompt Studio.

Each domain row must answer:

- Which source is authoritative: local, server, workspace, or mixed.
- Which operations are supported.
- Which operations are unsupported and through which reason-code report.
- Whether dry-run mirror reporting exists.
- Whether UX can render a source selector without reading service internals.

The remote-only utility rollup is only a summary row. Handoff readiness must be tracked on the individual remote-only utility rows because each utility can have different server APIs, auth requirements, capability reason codes, and blockers.

### UX Handoff Readiness

Track the exact UX-facing artifacts:

- Source authority and source selector contract.
- Unsupported-action report shape.
- Notification/event feed record contract.
- Sync dry-run mirror and conflict report contract.
- Domain capability matrix.
- Workspace isolation rules.
- Server unavailable/auth-expired/error presentation rules.

The UX developer should be able to scan this section and decide what to build now versus what must wait.

### Verification Ledger

Track the latest focused verification commands and results:

- Command.
- Date.
- Commit.
- Result.
- Known non-blocking warnings.

Do not make UI tests blocking while the UI layer is being rewritten, except for service wiring or contract behavior tests.

### Deferred Work

Keep deferred work explicit so it does not re-enter the phase accidentally:

- Write sync and persisted local outbox.
- Background mutation replay.
- Conflict resolution policies beyond report-only dry-run.
- UI rebuild implementation details.
- Workflow orchestration and scheduler behavior.
- Local parity for remote-only utilities unless separately approved.

## Update Rules

- Update the tracker in the same commit as meaningful backend contract work, or in a follow-up tracker-only commit before handoff.
- Do not mark a row `done` without committed tests or a committed contract path.
- Do not mark a row `ready-for-ux` if the UX developer would need to inspect implementation internals to know source authority, capability status, or unsupported behavior.
- If a row depends on a server API that is not present, mark it `blocked` and name the API.
- If a row depends on a design decision, mark it `blocked` and name the owner needed for the decision.
- If a row depends on unresolved security-affecting credential/auth/logout/redaction work, mark the row `security-blocked` and make the Connection/Auth gate `security-blocked`.
- Keep blocker text concrete and actionable.

## Initial Seed Status

The first live tracker should seed these statuses from current `dev`:

- Connection/auth: `done` only if secure credential storage, credential redaction, logout/global sign-out cleanup, and server-switch invalidation rows are also `done`. If any of those security-affecting rows remain open, seed the gate as `security-blocked` and list the exact open row as the blocker.
- Event/notifications: `ready-for-ux` for event identity, durable repository, server notification observer, and feed projection; durable replay beyond the repository retention policy remains hardening.
- Sync/mirror foundation: `in-progress`; repository is present, dry-run mirror reports are wired for notes, media, and research, but remaining domains need explicit coverage.
- Domain edge closure: `in-progress`; remote-only unsupported reports exist broadly, but each remote-only utility must be seeded as its own row and UX readiness must be confirmed row by row.
- UX handoff packet: `not-started` or `in-progress` depending on whether the tracker implementation creates the first packet section.

## Acceptance Criteria

- The live tracker exists at `Docs/superpowers/trackers/backend-parity-phase-tracker.md`.
- Every remaining roadmap item maps to one tracker row or an explicit deferred row.
- Every `done` or `ready-for-ux` row names committed evidence.
- Backend blockers and UX readiness are visible in the same row.
- The tracker identifies the next actionable backend row without rereading the full roadmap.
- The tracker does not claim UI implementation completion.

## Review Notes

This spec intentionally designs the tracker only. Creating and maintaining the live tracker is the next implementation step after user review.
