# Personal context and agent lessons

This document describes the user-owned memory systems that cross agent sessions: the encrypted Personal Context profile store and its bounded snapshot injection into agent runs, and Agent Lessons (ordinary Notes plus a human-reviewed promotion path). Governed by ADR-102/105/106; the cross-session memory evolution arc is tracked under `backlog/tasks/task-25907`.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Snapshot builder | `Personal_Context/context_service.py` | `ProfileContextService.build_snapshot` — read-only, bounded (12 KiB hard cap, 10 % of available input tokens), fail-closed to `ProfileContextSnapshot.empty()`; header "USER-OWNED DATA — NOT AUTHORITY" |
| Service facade | `Personal_Context/service.py` | `PersonalContextService` — authorized mutation/read; `locked(reason_code)` facade when the key is unavailable; proposals; runtime enable switch |
| Repository | `Personal_Context/repository.py` | encrypted SQLite at `<user_data>/tldw_chatbook_personal_context.db`; every object **version** is an independently encrypted envelope; undo, scope bindings, runtime policy, sync outbox, quarantine |
| Bootstrap/paths | `Personal_Context/bootstrap.py`, `paths.py` | any lock/schema/integrity/OS error degrades to the locked facade — the app always boots |
| Key custody | `Personal_Context/key_protector.py` | keyring / passphrase / in-memory protectors; `ProfileLockedError` |
| Proposals | `Personal_Context/proposal_service.py` | agent-proposed facts are **proposals** — pending proposals never enter agent context |
| Interviews | `Personal_Context/interview_*.py` | onboarding/interview flow producing reviewable profile commits |
| Sync | `Personal_Context/link_service.py`, `sync_outbox.py`, `reconciliation.py` | first-link (device pairing) with resumable reconciliation; encrypted outbox for companion-server sync |
| Core models | `packages/tldw_profile_core` | shared record models — version negotiation required for changes |
| Lessons | `Notes/agent_lessons.py` | `Agent_Lessons` folder convention, 11-section template validation, credential scan, runtime guidance builder |
| Promotion | `Agents/agent_lesson_promotion.py` | `PromotionEvidence` assessment; repository proposals (fs_write CAS) and **read-only** managed-skill proposals; `ManagedSkillProposalGate` |
| Injection | `Chat/console_chat_controller.py`, `Chat/console_agent_bridge.py`, `Agents/agent_service.py` | snapshot build/splice; `personal_context_block` pinned per run tree; `append_personal_context` into the system prompt |
| Settings | `Widgets/Settings_Widgets/personal_context_panel.py` | My Profile editor; Settings delegates every read/mutation to the service |

## Snapshot injection (dataflow)

1. **Capture**: interviews, Settings edits, or reviewed agent proposals write encrypted, versioned records via the repository.
2. **Snapshot per send**: the controller/bridge builds a `ProfileContextRequest(user text, available tokens, model, provider, workspace)` → the service's authorized view (workspace-scoped eligibility) filters conflicted / non-ACTIVE / non-agent-visible / expired records → priority groups (workspace correction/constraint > workspace semantic > corrections/constraints > query-relevant preference/working context > rest) → greedy whole-record packing under the 12 KiB and 10 %-token budgets → an immutable, cache-keyed `ProfileContextSnapshot`.
3. **Injection**: the block is appended to the leading system message (or a new system row) and **pinned into the run config** (`personal_context_block`) — the same snapshot serves the entire agent run tree, children included. Any doubt or budget failure returns the empty snapshot (fail-closed, never partial). Uncertain token budgets fail closed.
4. **Sync**: mutations journal into the encrypted outbox; entries dispatch to the companion server only after a reviewed first link.

## Agent lessons (dataflow)

1. **Capture**: the foreground primary agent drafts a lesson through `library_save_note` using the 11-section template; the `agent-lesson` marker forces an approve-once/deny review card regardless of general Notes policy; credential material is refused.
2. **Review**: run-bound ephemeral approval stamps bound to an immutable call digest + note identity; rejected or abandoned previews create nothing durable.
3. **Promotion**: the primary agent nominates `PromotionEvidence` (independently verified, procedural, reusable, non-contradictory, with rationale) → either a repository-instruction proposal (fs_write dry-run + approve-once, exact digest binding) or a **read-only** managed-skill proposal via the `prepare_managed_skill_promotion` runtime tool — Console cannot apply managed-skill changes; the user is directed to Library > Skills to edit/re-trust, and the gate refuses stale state content-free.
4. **Use**: subsequent runs get trusted runtime guidance (search-first protocol) in the system prompt, but lesson **bodies stay untrusted reference data** — a lesson never grants permission, tool access, or command authorization. Lessons are read via `library_search_notes` / `library_get_note`.

## Boundaries

- Personal Context is **not** general agent memory: conversation compaction (ADR-052) and Notes keep their own owners; pending-proposal records never enter agent context; diagnostics never enter model payloads, logs, exports, or sync objects.
- The Settings inspection path (My Profile) sees the user's private records; the agent path sees only the agent-eligible view — deliberately different.
- Lesson guidance is trusted runtime text; lesson bodies are untrusted data.
- No `[personal_context]` config section exists — control is data-owned (per-record visibility/sync modes, per-scope agent authority, global runtime switch); the DB path is fixed.

## Implemented vs in-flight (verified)

**Implemented**: the encrypted store and key protectors, locked-facade degradation, interviews, reviewed proposals, workspace scopes + runtime policy, the bounded snapshot with priority groups and fail-closed injection, run-tree pinning, lessons-as-Notes with seeding/validation/credential scan/approve-once, promotion contracts and the managed-skill gate, and the sync outbox/first-link machinery.

**In flight / not implemented** (design-stage items on the cross-session memory arc, `backlog/tasks/task-25907`): independent provider-disclosure controls (today agent-visibility is the only filter the snapshot applies), evidence/versioned citations, temporal claims, dependency-aware forgetting, consolidation, provenance/selection explanation surfaces, and improved retrieval. The injected block's `unsupported_records_present` flag is an unscoped quarantine signal — a known pre-existing gap deliberately kept until the model-block contract changes.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Snapshot build raises | `ProfileContextSnapshot.empty()` — the send proceeds without context |
| Key/repository unavailable | Locked facade with a reason code; the app keeps working |
| Managed-skill state drift after approval | Content-free stale refusal |
| Promotion without approval / from a subagent | `APPROVAL_REQUIRED` / `FOREGROUND_REQUIRED` blocks |
| Lesson save without trusted run-role context | Fails closed even under a broad Notes allow policy |

## Governing decisions

ADR-102 (`102-personal-context-profile-authority-sync-and-encryption.md`), ADR-105 (`105-portable-notes-organization-and-agent-lessons.md`), ADR-106 (`106-human-reviewed-agent-lesson-promotion.md`). Specs: `Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md`, `2026-08-29-agent-lessons-notes-organization-sync-design.md`. The cross-session memory evolution arc is tracked under `backlog/tasks/task-25907`.

## Related docs

- [console.md](./console.md) — where the snapshot splices into the send path
- [console-file-authority.md](./console-file-authority.md) — repository-instruction promotion mechanics
