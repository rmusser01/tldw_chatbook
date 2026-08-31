# Human-Reviewed Agent Lesson Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a foreground primary turn verified Agent Lesson evidence into one exact human-reviewed proposal for an eligible repository instruction or Chatbook-managed local skill without allowing lesson text, child agents, or stale previews to gain write authority.

**Architecture:** `Agents/agent_lesson_promotion.py` owns small immutable proposal/snapshot types and pure eligibility rules. Repository promotion has two existing-card reviews: a primary-only approve-once gate to prepare an exact read-only proposal, followed by a separate approve-once gate to apply that exact run-bound prepared record. Application reuses `fs_write` with expected-digest/expected-absent compare-and-swap under a canonical-target lock and descriptor-anchored atomic write. Managed skills have no Console mutation path: the primary emits exact replacement text and the user applies it manually through the existing Library editor and `LocalSkillsService.update_skill` version/re-trust flow.

**Tech Stack:** Python 3.11, existing workspace-local tool provider, project-instruction resolver/runtime, Textual approval card, managed local-skills service, `pytest` fake-model harnesses.

---

## Scope and prerequisites

- Implements `TASK-24613` after TASK-24309.
- Before code changes, read `backlog/docs/lessons-testing-evidence.md`, verify TASK-24309 is Done, set TASK-24613 to `In Progress`, and add its Implementation Plan section linking this document and ADR-106.
- Read the approved spec, ADR-106, ADR-105, ADR-009, ADR-032, ADR-069, and current project-instruction/file-tool tests.
- Do not add a promotion database, proposal queue, dedicated apply tool, Git/PR monitor, automatic mutation, managed-skill filesystem editing, or scheduled improver.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/106-human-reviewed-agent-lesson-promotion.md`

Reason: promotion crosses untrusted Notes, file authority, project-instruction activation, approval, and managed-skill trust. ADR-106 fixes those boundaries; this task implements it without creating a new contract.

## File responsibility map

- `tldw_chatbook/Agents/agent_lesson_promotion.py`: immutable proposal records, target eligibility, canonical digests, preview equivalence, and static capability/role guidance.
- `tldw_chatbook/Agents/project_instruction_resolver.py`: securely recompute the effective applicable instruction chain for one eligible target.
- `tldw_chatbook/Agents/project_instruction_runtime.py`: expose a content-bounded, ephemeral snapshot/revalidation seam tied to the current binding and activation ledger.
- `tldw_chatbook/Tools/local_tool_impls.py`: generic atomic full-file dry-run and compare-and-swap write; no Agent Lessons knowledge.
- `tldw_chatbook/Agents/local_tool_provider.py`: `fs_write` promotion-preparation inputs, two forced per-call reviews, bounded run-bound prepared-proposal records, exact application stamp binding, and invocation-time context revalidation.
- `tldw_chatbook/Chat/console_chat_controller.py`: pass the selected binding identity/fingerprint and current instruction context into the per-run local provider.
- `tldw_chatbook/Agents/agent_service.py`: capability- and role-aware promotion guidance; no durable proposal state.
- Existing Library skill editor and `LocalSkillsService`: remain the sole managed-skill mutation/version/re-trust owners.

### Task 1: Define pure evidence and target eligibility

**Files:**

- Create: `tldw_chatbook/Agents/agent_lesson_promotion.py`
- Create: `Tests/Agents/test_agent_lesson_promotion.py`

- [ ] **Step 1: Write failing evidence tests.** Eligible evidence is independently verified, procedural, reusable, narrowly scoped, and carries provenance plus principle rationale. A fixed incident count is neither required nor sufficient. Unverified, contradictory, interaction-specific, credential-bearing, or permission-seeking content is ineligible.
- [ ] **Step 2: Write failing target tests.** Repository eligibility is only `AGENTS.md` or `AGENTS.override.md` inside the selected writable binding. Managed local skills are `proposal_only`. Built-in/runtime/server/read-only skills, arbitrary files, absent authority, and paths outside the binding are ineligible with stable non-sensitive reason codes.
- [ ] **Step 3: Run red.** Run `pytest -q Tests/Agents/test_agent_lesson_promotion.py`; expect missing module failures.
- [ ] **Step 4: Implement small immutable types.** Include `PromotionEvidence`, `RepositoryInstructionTarget`, `RepositoryInstructionProposal`, `ManagedSkillProposal`, and `PromotionEligibility`. Proposal equality binds target, binding ID, locator fingerprint, effective-chain digest, expected file state, full replacement digest/content, and verification command/text.
- [ ] **Step 5: Run green and commit.** Run the focused test; expect pass. Commit `feat(agents): define lesson promotion proposals`.

### Task 2: Snapshot and revalidate the effective instruction target

**Files:**

- Modify: `tldw_chatbook/Agents/project_instruction_resolver.py`
- Modify: `tldw_chatbook/Agents/project_instruction_runtime.py`
- Modify: `Tests/Agents/test_project_instruction_resolver.py`
- Modify: `Tests/Agents/test_project_instruction_runtime.py`
- Modify: `Tests/Agents/test_project_instruction_concurrency.py`
- Modify: `Tests/Agents/test_project_instruction_path_targets.py`

- [ ] **Step 1: Write failing snapshot tests.** For one eligible target, capture selected binding ID, locator fingerprint, root identity, target-relative path, target digest or absent state, and the ordered effective instruction-chain `(relative_path, kind, digest)` tuples applicable to that target directory. Do not expose unrelated instruction bodies.
- [ ] **Step 2: Write failing race tests.** Retarget the binding; change ancestor identity; add/remove/change an applicable `AGENTS.md`/override; replace the target through a symlink race; or advance activation context. Revalidation must report stale/ineligible and never bless the old proposal.
- [ ] **Step 3: Run red.** Run `pytest -q Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_runtime.py Tests/Agents/test_project_instruction_concurrency.py Tests/Agents/test_project_instruction_path_targets.py`; expect missing snapshot APIs.
- [ ] **Step 4: Add one read-only resolver/runtime seam.** Reuse `_read_candidate`, canonical root identity, and broad-to-specific ordering. Hash canonical JSON metadata for `effective_chain_digest`; do not persist snapshots or duplicate instruction discovery. A missing target is represented as `expected_absent`, not a zero digest.
- [ ] **Step 5: Run green and commit.** Run the red command; expect pass. Commit `feat(agents): snapshot instruction promotion targets`.

### Task 3: Add generic dry-run and atomic compare-and-swap to `fs_write`

**Files:**

- Modify: `tldw_chatbook/Tools/local_tool_impls.py`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `Tests/Tools/test_local_tool_impls.py`
- Modify: `Tests/Tools/test_local_tool_sensitive_paths.py`
- Modify: `Tests/Agents/test_local_tool_provider.py`

- [ ] **Step 1: Write failing generic write tests.** Existing behavior remains compatible when new fields are omitted. `dry_run=true` writes nothing and returns bounded JSON containing `target_state`, current SHA-256 or `absent`, replacement SHA-256, byte count, and a bounded unified diff. `expected_sha256` and `expected_absent` are mutually exclusive.
- [ ] **Step 2: Write failure/race tests.** Wrong digest, file-created-after-absent-preview, file-deleted-after-digest-preview, symlink/parent swap, unencodable content, denied path, and failed final replace all leave the prior target unchanged. Use deterministic barriers to start two same-expectation writers together and prove exactly one succeeds. Swap a parent/symlink at the same barrier and prove the write cannot be redirected.
- [ ] **Step 3: Run red.** Run `pytest -q Tests/Tools/test_local_tool_impls.py Tests/Tools/test_local_tool_sensitive_paths.py Tests/Agents/test_local_tool_provider.py`; expect schema and write failures.
- [ ] **Step 4: Implement the minimal generic API.** Extend `write_file` with keyword-only `dry_run=False`, `expected_sha256=None`, and `expected_absent=False`. Encode before touching disk. For mutation, acquire one process-wide lock keyed by the canonical target; while it is held, open the already-validated parent directory with directory/no-follow flags, revalidate root and parent identities, read/recheck the target through descriptor-relative no-follow operations, and compare the expectation. Write and fsync a private temporary file in that same opened directory, then use descriptor-relative atomic replacement for an existing target or exclusive descriptor-relative creation for expected-absent. Clean up only the task-owned temp entry before releasing the lock. The lock makes competing Chatbook writers serialize; descriptor-relative no-follow operations prevent a parent/symlink swap from redirecting the write.
- [ ] **Step 5: Keep the schema narrow.** Add the three optional fields to `fs_write`; describe that preconditions are required for instruction promotion. Do not modify `fs_edit`/`fs_patch` or add a new apply tool.
- [ ] **Step 6: Run green and commit.** Run the red command; expect pass. Commit `feat(tools): add atomic fs_write preconditions`.

### Task 4: Gate and build an exact ephemeral repository proposal

**Files:**

- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `Tests/Agents/test_local_tool_provider.py`
- Create: `Tests/Chat/test_console_agent_lesson_promotion_context.py`
- Modify: `Tests/UI/test_chat_approval_card.py`

- [ ] **Step 1: Write failing composition/role tests.** Scratch/no-binding, read-only binding, missing locator fingerprint, changed root identity, subagent, and fleet calls cannot prepare a repository promotion. A selected writable binding passes its exact ID/fingerprint/root guard and instruction snapshot callback to the provider for this run only.
- [ ] **Step 2: Write the proposal-preparation review tests.** The primary first calls `fs_write(dry_run=true, promotion={...})`, where the bounded promotion object carries verified lesson public IDs, rationale, and verification text/command. This call forces an existing approval-card round with only approve-once/deny even when `fs_write` is broadly allowed. Denial constructs no proposal or record. Generic `fs_write(dry_run=true)` without `promotion` retains ordinary filesystem policy and is not treated as lesson promotion.
- [ ] **Step 3: Write failing exact-preview lifecycle tests.** After preparation approval, the eligible call returns one exact `RepositoryInstructionProposal`: target, binding metadata, effective-chain summary/digest, current expectation, exact replacement digest/content, bounded diff, evidence IDs, rationale, verification, and `proposal_digest`. The provider retains the identical immutable record only in a bounded run-keyed in-memory map. It clears records on run cancellation/end, binding invalidation, deny, superseding preparation, or provider disposal; no Note or database row is created.
- [ ] **Step 4: Run red.** Run `pytest -q Tests/Agents/test_local_tool_provider.py Tests/Chat/test_console_agent_lesson_promotion_context.py`; expect missing context, preparation gate, and proposal output.
- [ ] **Step 5: Pass trusted context at composition.** Extend `_compose_local_provider` from its existing `project_selection` inputs rather than reading whichever workspace is active later. `LocalToolProvider` accepts an immutable proposal-context callback; the model cannot supply or override binding/fingerprint/chain values. The model-supplied rationale/verification remain untrusted descriptive fields, never authority.
- [ ] **Step 6: Wire the preparation gate per call.** Extend `build_local_review_hook` to pass `call_id` into `pending_gate_for`. Before normal allow/session resolution, a primary promotion-preparation call always yields a per-call row with only approve-once/deny; subagent/fleet/unbound callers get a structured refusal. Store the approve-once preparation stamp by `(run_id, call_id, canonical_call_digest)` and require/consume it before running the dry-read. Ordinary local-tool stamps and permissions remain unchanged.
- [ ] **Step 7: Build and retain only after approval.** Combine the resolver snapshot and generic file preview, revalidating the root guard before and after reads. Hash canonical JSON of the complete proposal and retain the exact object under `(run_id, proposal_digest)` with a strict per-run bound. Return the same complete proposal to the transcript so the later application card can reproduce it exactly. Never log lesson text or replacement content.
- [ ] **Step 8: Run green and commit.** Run the red command; expect pass. Commit `feat(agents): preview exact instruction promotions`.

### Task 5: Force one exact approval and revalidate at application

**Files:**

- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`
- Create: `Tests/Chat/test_console_agent_lesson_promotion_approval.py`
- Modify: `Tests/UI/test_chat_approval_card.py`
- Modify: `Tests/Agents/test_local_tool_provider.py`

- [ ] **Step 1: Write the approval matrix.** An application call is a foreground primary `fs_write` to an eligible instruction target with full content, exactly one expected-state precondition, and `proposal_digest` resolving to this run's prepared record. It forces a second `approve_once|deny` round despite local-tool allow/session state. The card renders the stored exact proposal/diff, not a newly summarized payload. Subagent/fleet calls return `foreground_required`; unbound/direct calls return `approval_required`.
- [ ] **Step 2: Write binding/stale races.** After preview but before review/invoke, change target bytes, expected state, replacement, binding fingerprint, root identity, effective chain, role, run, or call ID. Every mismatch fails without mutation. Reusing a consumed stamp fails.
- [ ] **Step 3: Run red.** Run `pytest -q Tests/Chat/test_console_agent_lesson_promotion_approval.py Tests/UI/test_chat_approval_card.py Tests/Agents/test_local_tool_provider.py`; expect broad allows or stale calls to proceed.
- [ ] **Step 4: Extend the per-call local review hook for application.** Resolve the run-bound `proposal_digest`, require the application arguments to match the stored record byte-for-byte, and create a row with only approve-once/deny that renders that same record. Stamp `(run_id, call_id, canonical_call_digest, proposal_digest)` with the complete snapshot; do not persist or broaden it. Denial invalidates the prepared record.
- [ ] **Step 5: Enforce immediately before `fs_write`.** Authenticate primary role, consume both the application stamp and prepared proposal once under the provider lock, recompute binding/root/effective-chain state, compare complete proposal equivalence, then call generic `write_file` with the reviewed expectation/content. The generic write performs the final locked, descriptor-anchored atomic file-state CAS. Any failure invalidates this prepared application and never resets user edits.
- [ ] **Step 6: Run green and commit.** Run the red command; expect pass. Commit `feat(agents): guard instruction promotion application`.

### Task 6: Keep managed local skills proposal-only

**Files:**

- Modify: `tldw_chatbook/Agents/agent_lesson_promotion.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py`
- Modify: `tldw_chatbook/Agents/agent_runtime.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Docs/User_Guide/library/skills.md`
- Create: `Tests/Agents/test_agent_lesson_skill_promotion.py`
- Modify: `Tests/Agents/test_tool_catalog.py`
- Modify: `Tests/Agents/test_agent_service.py`
- Modify: `Tests/Agents/test_agent_runtime.py`
- Create: `Tests/Chat/test_console_agent_lesson_skill_promotion_approval.py`
- Modify: `Tests/Skills/test_local_skills_service.py`
- Modify: `Tests/UI/test_library_skills_canvas.py`

- [ ] **Step 1: Write failing capability and preparation-review tests.** The Console catalog may list/search/get managed skills but exposes no create/update/delete skill tool. A foreground primary with the effective skill-read capability receives one native read-only `prepare_managed_skill_promotion` action. Its call contains skill public ID/name, expected version/trust state, current/replacement digests, exact replacement, evidence note IDs, rationale, and verification, and always enters an existing approve-once/deny card before exact proposal construction. Subagent/fleet schemas omit the action and return candidate text only; direct/unbound invocation fails closed.
- [ ] **Step 2: Write failing lifecycle tests.** Denial, cancellation, stale version/trust/content, changed payload, or wrong run/call ID produces no proposal state or output. Approval issues a private single-use preparation stamp bound to the complete call; the runtime action re-reads the skill through `LocalSkillsService`, consumes the stamp, verifies current version/trust/content digest, and returns the exact proposal. Any in-memory record is bounded to that run/call and discarded after return or failure because application is manual.
- [ ] **Step 3: Write existing-owner regression tests.** Manual Library application calls `LocalSkillsService.update_skill(expected_version=..., trust_approved=False)`. A stale version refuses; a successful edit increments version and remains inactive/untrusted until the existing reviewed re-trust completes. Raw `fs_*` cannot reach managed skill storage.
- [ ] **Step 4: Run red.** Run `pytest -q Tests/Agents/test_agent_lesson_skill_promotion.py Tests/Chat/test_console_agent_lesson_skill_promotion_approval.py Tests/Skills/test_local_skills_service.py Tests/UI/test_library_skills_canvas.py`; expect missing preparation review, proposal guidance, and ownership coverage.
- [ ] **Step 5: Register and dispatch the primary-only runtime action.** Define the bounded `prepare_managed_skill_promotion` schema/name in `Agents/tool_catalog.py`. `AgentService` discloses it only for a primary with effective Library skill-read capability and injects the run-owned callback into `LoopDeps`. Add the explicit dispatch branch in `agent_runtime.py` before generic provider invocation, preserving `call_id` and structured refusal/result handling. Pin catalog visibility, primary/subagent narrowing, callback dispatch, missing-callback refusal, and no generic-provider fallthrough in `test_tool_catalog.py`, `test_agent_service.py`, and `test_agent_runtime.py`.
- [ ] **Step 6: Reuse the existing approval surface.** Add a small private `ManagedSkillProposalGate` in `agent_lesson_promotion.py`; compose it into the controller's existing combined review hook and inject its stamp-consuming callback into the primary-only runtime action. The gate authenticates trusted run role before producing a pending row and offers only approve-once/deny. It persists nothing and cannot mutate a skill.
- [ ] **Step 7: Add proposal-only guidance and docs.** Tell the primary to invoke the reviewed preparation action, show its exact returned replacement, and direct the user to the Library editor. Do not add an agent-controlled service mutation, hidden application handoff, or filesystem exception. Permit a later read-only verification after the user manually applies/re-trusts.
- [ ] **Step 8: Run green and commit.** Run the red command plus `pytest -q Tests/Agents/test_tool_catalog.py Tests/Agents/test_agent_service.py Tests/Agents/test_agent_runtime.py`; expect pass. Commit `feat(agents): propose managed skill improvements safely`.

### Task 7: Add role- and capability-aware promotion behavior

**Files:**

- Modify: `tldw_chatbook/Agents/agent_lesson_promotion.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Create: `Tests/Agents/test_agent_lesson_promotion_guidance.py`
- Create: `Tests/Agents/test_agent_lesson_promotion_behavioral_eval.py`
- Modify: `Tests/Agents/test_fleet_continuation.py`

- [ ] **Step 1: Write the guidance matrix.** No lesson read capability means no promotion guidance. A primary with verified evidence plus eligible read-only context may suggest one smallest focused proposal. Repository application guidance appears only with a writable selected binding and `fs_write`; managed skills remain manual. Subagents never ask for promotion approval or apply changes.
- [ ] **Step 2: Add scripted behavioral cases.** Observe one strong verified signal being eligible without a fixed count; rejection on weak/contradictory evidence; principle+rationale over accumulated rules; smallest edit; explicit unknowns; and child evidence handoff. Label these as model/prompt evidence, not authorization.
- [ ] **Step 3: Run red.** Run `pytest -q Tests/Agents/test_agent_lesson_promotion_guidance.py Tests/Agents/test_agent_lesson_promotion_behavioral_eval.py Tests/Agents/test_fleet_continuation.py`; expect missing behavior.
- [ ] **Step 4: Reuse one send-time pure helper.** Build from actual disclosed schemas, trusted run role, and immutable target capability. Recompute on continuation/narrowing. Never place lesson bodies, proposal bodies, or outcomes in system/project instruction ownership.
- [ ] **Step 5: Run green and commit.** Run the red command; expect pass. Commit `feat(agents): guide reviewed lesson promotion`.

### Task 8: Prove outcome handling and close TASK-24613

**Files:**

- Create: `Tests/Agents/test_agent_lesson_promotion_end_to_end.py`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/library/notes.md`
- Modify: `backlog/tasks/task-24613 - Add-human-reviewed-Agent-Lesson-promotion-proposals.md`
- Optional documentation: update the relevant existing `backlog/docs/lessons-*.md` only for a real execution incident.

- [ ] **Step 1: Write repository end-to-end coverage.** Primary reads verified lesson evidence, prepares one exact dry-run, user approves the matching application, write CAS succeeds, verification runs, and the changed instruction becomes effective only in a later activation/run. Denied, stale, failed, reverted, and superseded cases preserve current user state.
- [ ] **Step 2: Write outcome-note coverage.** Applied/rejected/failed outcome text is offered as a separate ordinary Agent Lesson update and requires its own TASK-24309 approval. Only a persisted approved rejection suppresses an identical later suggestion; synchronized outcomes never authorize a write on another device.
- [ ] **Step 3: Run targeted verification.** Run:

```bash
python -m compileall -q tldw_chatbook/Agents tldw_chatbook/Tools
pytest -q Tests/Agents/test_agent_lesson_promotion.py Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_runtime.py Tests/Agents/test_project_instruction_concurrency.py Tests/Agents/test_project_instruction_path_targets.py Tests/Tools/test_local_tool_impls.py Tests/Tools/test_local_tool_sensitive_paths.py Tests/Agents/test_local_tool_provider.py Tests/Chat/test_console_agent_lesson_promotion_context.py Tests/Chat/test_console_agent_lesson_promotion_approval.py Tests/UI/test_chat_approval_card.py Tests/Agents/test_agent_lesson_skill_promotion.py Tests/Agents/test_tool_catalog.py Tests/Agents/test_agent_service.py Tests/Agents/test_agent_runtime.py Tests/Chat/test_console_agent_lesson_skill_promotion_approval.py Tests/Skills/test_local_skills_service.py Tests/UI/test_library_skills_canvas.py Tests/Agents/test_agent_lesson_promotion_guidance.py Tests/Agents/test_agent_lesson_promotion_behavioral_eval.py Tests/Agents/test_agent_lesson_promotion_end_to_end.py Tests/Agents/test_fleet_continuation.py
git diff --check
```

Expected: compile succeeds, targeted tests pass, and `git diff --check` prints nothing. Do not run the full suite without user opt-in.

- [ ] **Step 4: Live-verify in a disposable writable binding.** Confirm exact preview, changed-file refusal, expected-absent race refusal, binding retarget refusal, effective-chain change refusal, successful later-run activation, child refusal, manual skill edit/version refusal/re-trust, and separate outcome-note approval. Preserve pre-existing dirty edits and never use reset/checkout to recover.
- [ ] **Step 5: Self-review and close.** Inspect every eligibility/role/capability combination, approval clearing/consumption, target/path races, trust transitions, ACs, and ADR-106. Add Implementation Notes, set TASK-24613 Done, and repeat provisional task/ADR collision checks before merge.
- [ ] **Step 6: Commit.** Commit `docs(agents): complete lesson promotion`.
