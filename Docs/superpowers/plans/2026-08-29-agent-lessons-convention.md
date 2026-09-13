# Agent Lessons Notes Convention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a user-owned `Agent_Lessons` Notes convention that foreground primary agents can search, preview, and explicitly save as verified reusable evidence while subagents remain draft-only and every retrieved lesson stays untrusted.

**Architecture:** Ordinary Notes remain the only durable lesson store. `Notes/agent_lessons.py` owns pure convention/template/classification helpers; the Notes transaction from TASK-24308 remains the mutation authority and atomically revalidates exact-marker plus pending-receipt state. The existing Console approval round issues a single-use call-bound stamp, while trusted run identity is carried through the existing `run_context` binding into the Library provider/service. No new permission store, approval UI, memory database, or model dependency is added.

**Tech Stack:** Python 3.11, SQLite/FTS5, existing Library tools and Console agent runtime, Textual approval card, `pytest` fake-model harnesses.

---

## Scope and prerequisites

- Implements `TASK-24309` after TASK-24307 and TASK-24308.
- Before code changes, read `backlog/docs/lessons-testing-evidence.md`, verify both dependencies are Done, set TASK-24309 to `In Progress`, and add its Implementation Plan section linking this document, ADR-105, and ADR-106.
- Read the approved spec, ADR-105, ADR-106, ADR-030, ADR-032, and the TASK-24308 transaction plan.
- Do not add embeddings, semantic ranking, automatic conversation capture, a durable draft, a promotion workflow, background generation, or a general DLP framework.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/105-portable-notes-organization-and-agent-lessons.md` and `backlog/decisions/106-human-reviewed-agent-lesson-promotion.md`

Reason: ADR-105 owns ordinary-Notes storage, organization, and the untrusted-data boundary. ADR-106 extends it with forced foreground approval, role-aware enforcement, receipt classification, and single-use stamp binding. This plan implements both without changing their boundaries.

## File responsibility map

- `tldw_chatbook/Notes/agent_lessons.py`: constants, template rendering/validation, high-confidence credential checks, pure capability guidance, canonical call digest, and immutable lesson-classification value objects.
- `tldw_chatbook/Agents/run_context.py`: trusted `(run_id, agent_kind)` context binding for review and provider invocation threads.
- `tldw_chatbook/Agents/library_tool_provider.py`: privately issued ephemeral per-call Agent Lesson approval authorities and trusted invocation context handoff; it never consumes them before the Notes transaction.
- `tldw_chatbook/Chat/console_chat_controller.py`: preflight classified Library save calls into the existing per-call approval round; no policy persistence.
- `tldw_chatbook/Library/local_library_tool_service.py`: fail-closed routing of trusted lesson mutation context into the Notes transaction; ordinary Notes remain compatible.
- TASK-24308's `NotesInteropService` implementation: transaction-time classification and stamp consumption before any note/organization mutation.
- `tldw_chatbook/Agents/agent_service.py`: capability- and role-aware send-time guidance only.

### Task 1: Add monotonic Agent Lessons seed ownership (schema v61)

**Files:**

- Create: `tldw_chatbook/DB/migrations/chachanotes_v60_to_v61_agent_lessons_seed.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Create: `Tests/DB/test_chachanotes_agent_lessons_seed_migration.py`

- [x] **Step 1: Write failing real-v60 reopen tests.** Start from the historical v60 fixture, migrate, reopen, and assert the migration creates no folder, keyword, or note. Cover rollback and fresh-schema parity while proving the shipped v59→v60 publication-intent migration remains unchanged.
- [x] **Step 2: Run the focused test.** Run `pytest -q Tests/DB/test_chachanotes_agent_lessons_seed_migration.py`; expect failure because v59 is absent.
- [x] **Step 3: Add the minimal monotonic state table.** Use `(profile_id, dataset_id)` as the key, `scope_mode in ('local_only','synchronized')`, `state in ('unknown','not_seeded','seeded')`, optional `folder_sync_id`, and a seed fingerprint. Wire a genuine v60→v61 migration plus fresh schema and set `_CURRENT_SCHEMA_VERSION = 61`. Synchronized state advances `unknown → not_seeded|seeded → seeded`; deletion never resets it.
- [x] **Step 4: Run green.** Run the focused test; expect all tests to pass.
- [x] **Step 5: Commit.** Stage the migration, DB runner, focused test, and the schema-bump guard/version-pin updates required by those files; commit `feat(notes): add Agent Lessons seed state`.

### Task 2: Define the lesson template, marker, and safe content boundary

**Files:**

- Create: `tldw_chatbook/Notes/agent_lessons.py`
- Create: `Tests/Notes/test_agent_lessons.py`
- Create: `Tests/Notes/test_agent_lesson_secret_validation.py`

- [x] **Step 1: Write failing pure tests.** Pin the exact folder `Agent_Lessons`, exact marker `agent-lesson`, one-note-per-lesson rules, public related note IDs, folder-independent discovery, case-variant non-matches, and honest `Unknown`/empty failed-attempt behavior.
- [x] **Step 2: Pin the approved template.** Use these required headings; `Promotion candidate` is optional:

```python
REQUIRED_SECTIONS = (
    "Applicability", "Symptoms", "Feedback or trigger", "Provenance",
    "Root cause", "Verified solution", "Failed attempts and why",
    "Verification evidence", "Generalizable principle and rationale",
    "Caveats", "Related lessons",
)
```

Unknown provenance or validation limits are written as `Unknown`; the renderer never invents feedback or failed attempts.

- [x] **Step 3: Write failing credential-boundary tests.** Reject unambiguous private-key blocks, credible live-key prefixes, and explicit credential assignments with non-placeholder material. Accept hashes, UUIDs, stack traces, error IDs, redacted values, and explicit fake examples. Assert rejected content is absent from logs, errors, and durable tables.
- [x] **Step 4: Run red.** Run `pytest -q Tests/Notes/test_agent_lessons.py Tests/Notes/test_agent_lesson_secret_validation.py`; expect import/test failures.
- [x] **Step 5: Implement the pure module.** Use short anchored regular expressions only. Return stable generic refusal codes; do not log matches, score entropy, or add dependencies. Expose pure render/validate/classify/digest helpers without copying Notes SQL.
- [x] **Step 6: Run green and commit.** Run the red command; expect pass. Commit `feat(notes): define Agent Lessons evidence format`.

### Task 3: Seed only at the correct readiness boundary

**Files:**

- Modify: `tldw_chatbook/Notes/agent_lessons.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/Sync_Interop/notes_organization_sync_service.py`
- Modify: `tldw_chatbook/Notes/notes_organization_repository.py`
- Create: `Tests/Notes/test_agent_lessons_seed.py`
- Modify: `Tests/Sync_Interop/test_notes_organization_enrollment.py`
- Modify: `Tests/Sync_Interop/test_notes_organization_adapters.py`
- Modify: `Tests/Sync_Interop/test_notes_organization_two_device.py`

- [x] **Step 1: Write failing seed/race tests.** Cover local-only schema readiness, synchronized organization readiness, restart idempotency, rename/delete non-recreation, exact-name collision review, untouched empty seed convergence, and edited/copied/acknowledged/used/different-spelling review. A copied general-outbox candidate is not safe for automatic retirement even if it is not yet acknowledged.
- [x] **Step 2: Add historical-bootstrap and upgrade coverage.** Observe an exact-root remote upsert inside `NotesOrganizationRepository.apply_envelope()` after payload validation but before duplicate/stale returns, so an upsert followed by rename or tombstone records monotonic seed evidence in the same transaction. For an upgraded already-ready synchronized profile whose v59 state is still `unknown`, replay/pull history once instead of taking the ready short circuit; current heads alone cannot prove that a formerly exact root was renamed or deleted.
- [x] **Step 3: Run red.** Run `pytest -q Tests/Notes/test_agent_lessons_seed.py Tests/Sync_Interop/test_notes_organization_enrollment.py Tests/Sync_Interop/test_notes_organization_adapters.py Tests/Sync_Interop/test_notes_organization_two_device.py`; expect seed behavior failures.
- [x] **Step 4: Implement one idempotent initializer.** Local-only profiles run after the app has a schema-ready Notes service; synchronized profiles run only after the full six-domain group passes its complete readiness gate. Revalidate state and exact/case-fold roots inside one Notes transaction, reuse an active exact-spelling root or create it with the existing cursor-aware folder repository, record any synchronized intent plus the seed state atomically, and run this before pending-receipt finalization. Do not create the marker until a lesson exists.
- [x] **Step 5: Converge only provably untouched races.** Automatic adoption requires the coordinator-created fingerprint, exact root spelling, active version-1 empty/unused state, no unrelated receipt/suppression/head, and an intent that was never copied or acknowledged. Retire that unpublished intent and adopt the winning remote identity atomically; every edited, used, copied, acknowledged, or differently spelled candidate enters the existing adoption-review path.
- [x] **Step 6: Run green and commit.** Run the red command; expect pass. Commit `feat(notes): seed Agent Lessons safely`.

### Task 4: Pin the existing trusted run actor across review and provider threads

**Files:**

- Create: `Tests/Agents/test_agent_run_context.py`
- Modify: `Tests/Agents/test_agent_service.py`

- [x] **Step 1: Write context characterization tests.** Primary and subagent runs expose the exact existing immutable `CurrentRunActor` inside the loop-wide review hook and the fresh per-tool daemon thread. A threaded fleet child is intentionally represented by the existing subagent actor and gains no third privileged role. Nested bindings restore LIFO; unbound/direct calls return no actor and never default to primary.
- [x] **Step 2: Run the focused characterization.** Run `pytest -q Tests/Agents/test_agent_run_context.py Tests/Agents/test_agent_service.py`; expect the existing loop-wide and per-invocation bindings to pass. Mutation-check by removing each production binding in turn and confirm its corresponding test fails.
- [x] **Step 3: Reuse the established authority seam.** Downstream Agent Lesson preflight and invocation code reads `current_run_actor()`; do not add a second identity value/ContextVar, widen the generic `ToolProvider.invoke` protocol, or map an unbound/direct caller to primary. Change `run_context.py` or `agent_service.py` only if the characterization exposes a real gap.
- [x] **Step 4: Commit the evidence.** Commit only the characterization tests and any minimal proven fix as `test(agents): pin trusted run role for lesson calls`.

### Task 5: Force exact per-call approval before Agent Lesson dispatch

**Files:**

- Modify: `tldw_chatbook/Agents/library_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`
- Modify: `Tests/Agents/test_library_tool_provider.py`
- Create: `Tests/Chat/test_console_agent_lesson_approval.py`
- Modify: `Tests/UI/test_chat_approval_card.py`

- [x] **Step 1: Write failing classification and role tests.** A foreground-primary `library_save_note` call enters review when its request adds the exact marker, its current note is marked, or its note owns `pending_organization`/`placement_review`. The same classified call from a subagent or fleet run returns per-call `foreground_required`; an unbound/direct call returns `approval_required`. Those refusals create no card, approval round, stamp, or authority. Case variants and unrelated Notes do not trigger this policy. Classification failures fail closed without exposing note content.
- [x] **Step 2: Write failing per-call card tests.** Two same-name saves with different `call_id` values render separately and can receive different decisions. Agent Lesson rows offer only `approve_once` and `deny`; ordinary MCP/builtin option behavior is unchanged. Rejection returns a per-call refusal and does not invoke the provider.
- [x] **Step 3: Run red.** Run `pytest -q Tests/Agents/test_library_tool_provider.py Tests/Chat/test_console_agent_lesson_approval.py Tests/UI/test_chat_approval_card.py`; expect the Library provider not to participate in review.
- [x] **Step 4: Add a role-first Library preflight/stamp seam.** Compose the already-built `LibraryToolProvider` into `build_tool_review_hook`. Preflight reads the loop-wide trusted run identity before creating any pending row: classified subagent/fleet/unbound calls return their structured per-call refusal immediately and never call `request_approvals`. Only a foreground primary returns an `MCPPendingCall`-compatible row with safe compact arguments, exact `call_id`, and `options=("approve_once", "deny")`. On approval, record an ephemeral single-use stamp keyed by `(run_id, call_id, canonical_call_digest)` and carrying note/create identity, classification, content/organization preconditions, and receipt state/version. Clear this run's stamps at hook entry and on cancellation.
- [x] **Step 5: Keep the existing UI protocol.** Reuse per-call `call_id` decisions already supported by `ChatApprovalCard`; make only the smallest formatting change needed to show create/update, title, classification, and digest without showing full note content.
- [x] **Step 6: Run green and commit.** Run the red command; expect pass. Commit `feat(notes): require per-call lesson approval`.

### Task 6: Enforce role, stamp, and classification in the Notes transaction

**Files:**

- Modify: `tldw_chatbook/Agents/library_tool_provider.py`
- Modify: `tldw_chatbook/Library/local_library_tool_service.py`
- Modify: TASK-24308's `tldw_chatbook/Notes/Notes_Library.py`
- Modify: `Tests/Library/test_local_library_tool_service.py`
- Modify: `Tests/Library/test_cross_runtime_parity.py`
- Create: `Tests/Notes/test_agent_lesson_mutation_authority.py`

- [x] **Step 1: Write the fail-closed matrix.** Test marked, newly marked, pending-organization, and placement-review creates/updates for primary-approved, primary-unapproved, subagent, fleet, direct provider, and MCP/direct-service calls. Ordinary Notes retain existing behavior.
- [x] **Step 2: Write transaction-race tests.** Between review and transaction, add/remove the marker; create/delete/transition the receipt; change content version; change organization version; reuse a stamp; change call arguments; or swap note identity. Every case returns `approval_required`, `content_changed`, `organization_changed`, or `foreground_required` without any note, folder, keyword, receipt, or intent mutation.
- [x] **Step 3: Run red.** Run `pytest -q Tests/Notes/test_agent_lesson_mutation_authority.py Tests/Library/test_local_library_tool_service.py Tests/Library/test_cross_runtime_parity.py`; expect unauthorized paths to write or lack structured refusal.
- [x] **Step 4: Pass opaque private authority, not caller booleans.** On approve-once, `LibraryToolProvider` privately issues an immutable authority object plus a registry-backed single-use token bound to the complete reviewed snapshot. `invoke` reads the trusted run identity and passes that opaque object/token through an internal typed context without consuming it. Only the exact issuer instance can authenticate it; direct service/MCP callers cannot construct a valid authority. Do not expose the context, token, or issuer in the public JSON schema, result, logs, or MCP wire.
- [x] **Step 5: Authenticate and consume under the TASK-24308 transaction.** After opening the Notes transaction and before any mutation, load the current marker and unresolved receipt, derive classification, require `agent_kind == "primary"`, and call the private issuer's `consume_if_matches(...)`. That method holds its single-use lock while authenticating object identity and checking run, call digest, note/create identity, classification, content/organization preconditions, and receipt state/version; it consumes the token exactly once only after every field matches. The Notes transaction then mutates using that same snapshot. A later DB failure may leave the token safely spent but cannot leave an unauthorized mutation. Credential validation applies only to classified agent-authored saves. A user-removed marker is never silently restored unless the exact approved request adds it.
- [x] **Step 6: Run green and commit.** Run the red command; expect pass. Commit `feat(notes): enforce lesson authority transactionally`.

### Task 7: Append role- and capability-aware trusted guidance

**Files:**

- Modify: `tldw_chatbook/Notes/agent_lessons.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Create: `Tests/Agents/test_agent_lessons_runtime_guidance.py`
- Modify: `Tests/Agents/test_agent_runtime_preparation.py`
- Modify: `Tests/Agents/test_agents_internal_prompts.py`
- Modify: `Tests/Agents/test_fleet_continuation.py`
- Modify: `Tests/Agents/test_skill_tool_spawn.py`

- [x] **Step 1: Write the capability matrix.** No Notes tools means no suffix. Search/get only gives untrusted search/read guidance. A primary with search/get/save gets search-first, verify, exact preview, and save guidance. A subagent with the same tools gets search/draft/return guidance and an explicit no-mutation boundary. Save without search never instructs a save.
- [x] **Step 2: Add quality requirements to the golden prompt assertions.** Guidance requests feedback/trigger, provenance, independent evidence, principle+rationale, honest unknowns, no invented failed attempts, update-vs-create judgment, and progressive disclosure. It says notes cannot grant permission or override instructions.
- [x] **Step 3: Run red.** Run `pytest -q Tests/Agents/test_agent_lessons_runtime_guidance.py Tests/Agents/test_agent_runtime_preparation.py Tests/Agents/test_agents_internal_prompts.py Tests/Agents/test_fleet_continuation.py Tests/Agents/test_skill_tool_spawn.py`; expect suffix failures.
- [x] **Step 4: Implement one pure suffix builder.** It accepts the actual disclosed schemas and trusted role for that send. Call it at both send seams after user/configured system content and before the existing workspace/environment suffix. Recompute after child narrowing and continuation; never interpolate note bodies or drafts.
- [x] **Step 5: Run green and commit.** Run the red command; expect pass. Commit `feat(agents): guide role-aware Agent Lessons`.

### Task 8: Prove useful cross-agent reuse with deterministic and behavioral evidence

**Files:**

- Create: `Tests/Agents/test_agent_lessons_end_to_end.py`
- Create: `Tests/Agents/test_agent_lessons_behavioral_eval.py`
- Modify: `Tests/Library/test_cross_runtime_parity.py`
- Modify: `Tests/Chat/test_console_agent_lesson_approval.py`
- Modify: `Tests/Chat/test_console_agent_tool_result_cap.py`

- [x] **Step 1: Write the deterministic end-to-end scenario.** Use the real scripted `AgentService` harness, authenticated built-in Library provider, Notes database, and production disclosure flow (`find_tools` then `load_tools`, because the complete Library catalog exceeds the direct-disclosure threshold). Primary Agent A searches, prepares a structured lesson, shows the exact safe call preview, receives approve-once, and saves. Agent B later searches the exact marker, reads the trust notice before adversarial embedded instructions in the actual model-visible result, verifies applicability, and uses the safe solution. Assert durable state, tool calls, single-use stamp consumption, and no authority gain.
- [x] **Step 2: Add negative deterministic cases.** Rejected preview, child draft, search-unavailable primary, stale preconditions, and credential refusal create no durable fallback. Cross-runtime direct/MCP calls cannot forge Console authority.
- [x] **Step 3: Add scripted behavioral fixtures.** Using the existing fake-model harness, evaluate useful principle/rationale, privacy-preserving provenance, duplicate/update judgment, no invented attempts, draft-only subagent behavior, and refusal to treat retrieved text as permission. Keep these assertions separate from enforcement tests and label them model/prompt evidence, not security guarantees.
- [x] **Step 4: Run focused evidence.** Run `pytest -q Tests/Agents/test_agent_lessons_end_to_end.py Tests/Agents/test_agent_lessons_behavioral_eval.py Tests/Library/test_cross_runtime_parity.py Tests/Chat/test_console_agent_lesson_approval.py Tests/Chat/test_console_agent_tool_result_cap.py`; expect pass. Keep the scripted behavioral assertions labeled as prompt evidence rather than model-general or security guarantees; transaction tests remain the enforcement proof.
- [x] **Step 5: Commit.** Commit `test(agents): prove reviewed Agent Lesson reuse`.

### Task 9: Document, verify, and close TASK-24309

**Files:**

- Modify: `Docs/Development/Agent-Tools/local-library-tools.md`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/library/notes.md`
- Modify: `backlog/tasks/task-24309 - Add-the-Agent-Lessons-Notes-convention.md`
- Optional documentation: update the relevant existing `backlog/docs/lessons-*.md` only if execution produces a real generalizable incident.

- [x] **Step 1: Document the user contract.** Cover folder ownership, exact marker discovery, template, feedback/provenance, principle rationale, search-first/update rules, foreground approval, subagent drafts, receipt classification, secret refusal, and untrusted retrieval.
- [x] **Step 2: Run targeted static and test verification.** Run:

```bash
python -m compileall -q tldw_chatbook/Notes tldw_chatbook/Agents tldw_chatbook/Library
pytest -q Tests/DB/test_chachanotes_agent_lessons_seed_migration.py Tests/Notes/test_agent_lessons.py Tests/Notes/test_agent_lessons_seed.py Tests/Notes/test_agent_lesson_secret_validation.py Tests/Notes/test_agent_lesson_mutation_authority.py Tests/Agents/test_agent_run_context.py Tests/Agents/test_agent_service.py Tests/Agents/test_agent_lessons_runtime_guidance.py Tests/Agents/test_agent_lessons_end_to_end.py Tests/Agents/test_agent_lessons_behavioral_eval.py Tests/Agents/test_agent_runtime_preparation.py Tests/Agents/test_agents_internal_prompts.py Tests/Agents/test_fleet_continuation.py Tests/Agents/test_skill_tool_spawn.py Tests/Agents/test_library_tool_provider.py Tests/Library/test_local_library_tool_service.py Tests/Library/test_cross_runtime_parity.py Tests/Chat/test_console_agent_lesson_approval.py Tests/UI/test_chat_approval_card.py Tests/Widgets/test_tool_message_widgets.py Tests/Chat/test_console_agent_tool_result_cap.py
git diff --check
```

Expected: compile succeeds, targeted tests pass, and `git diff --check` prints nothing. Do not run the full suite without user opt-in.

- [x] **Step 3: Live-verify through the TASK-24307 schema-safe gate.** Coordinate v61 compatibility first; isolate `HOME`, XDG paths, config, and data directory under two disposable roots. Against the current server verify seed/rename/move, exact marker search, primary preview/approval/save, Agent B discovery, subagent refusal, stale stamp refusal, pending/offline finalization, and no sensitive test content.

  Live evidence (2026-08-30): a shallow disposable checkout of protected server
  `dev` at `54448ef08970e4a348478bdf47be5715c875241c` ran on localhost with an
  isolated auth database, fresh per-user databases, XDG/HOME roots, and a
  disposable test key. The production FastAPI app and production Sync router,
  authentication, adapters, and SQLite materializers were used; unrelated app
  lifespan/background startup was disabled because its optional RAG stack was
  not part of this gate. Capabilities advertised the complete six-domain Notes
  organization group with ready server-trusted storage.

  Devices A and B converged on one folder sync ID. The first run exposed a real
  dataset-wide idempotency conflict when B's legacy inventory republished A's
  pulled seed; `205179140f` now skips exact applied remote heads, with a focused
  regression, and a fresh-server rerun had no retained or rejected envelope. A
  rename to `Reviewed_Agent_Lessons`, move under `Shared_Knowledge`, and repeated
  enrollment propagated without recreating the conventional spelling. A safe
  Note was published before its organization dependencies; B then found exactly
  one spelling-exact `agent-lesson` result in the renamed folder. Device C kept
  an offline placement receipt pending through the not-ready gate, finalized it
  after complete enrollment, and published its marker/link without rejection.

  In a separately isolated client root, the production `AgentService`, Console
  approval hook, Library provider, and real Notes database passed the selected
  primary approve-once save/Agent B untrusted reuse, subagent refusal, stale
  stamp refusal, credential refusal, and receipt-finalization cases (6 passed).
  A durable database/log scan found no credential or private-key patterns. No
  live probe emitted Note bodies.
- [x] **Step 4: Self-review and close.** Inspect prompt order, every role/tool combination, approval clearing/consumption, every transaction race, and all durable/log owners. Complete ACs, add concise Implementation Notes referencing ADR-105/106, set TASK-24309 Done, and repeat task/ADR collision checks before merge.
- [x] **Step 5: Commit.** Commit `docs(notes): complete Agent Lessons convention`.
