# Agent Lessons Notes Convention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task with review checkpoints.

**Goal:** Add a user-owned `Agent_Lessons` Notes convention that permitted agents can search and update with verified reusable solutions—including failed attempts and why they failed—without elevating note content into instructions.

**Architecture:** Agent Lessons remain ordinary Notes. One small Notes module owns constants, template validation, exact-marker discovery, seed coordination, and a narrow high-confidence secret check. A monotonic v52 seed record prevents recreation after user rename/delete. Trusted runtime guidance is appended at model-send time only when the run's actual allowed tool set supports the workflow; primary and child agents use the same capability function. Retrieved notes stay ordinary untrusted tool results.

**Tech Stack:** Python 3.11, SQLite, existing Library Notes tools and agent runtime, `pytest`.

---

## Scope and prerequisites

- Implements `TASK-24309` after TASK-24307 and TASK-24308.
- Before the first code edit, verify both dependencies are Done, set TASK-24309 to `In Progress`, and add an `## Implementation Plan` section to its task file linking this document and ADR-102.
- Read the approved spec, ADR-102, ADR-030, ADR-032, Console project-instruction governance, and agent/runtime prompt tests before execution.
- Do not add embeddings, semantic memory, automatic capture, background lesson generation, a new permission, a ranking system, or a general DLP/PII framework.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/102-portable-notes-organization-and-agent-lessons.md`

Reason: the Notes ownership, trusted/untrusted prompt boundary, seed policy, and save behavior are recorded in ADR-102. This task implements that decision and needs no new ADR unless those boundaries change.

## Task 1: Add monotonic seed ownership (schema v52)

**Files:**

- Create: `tldw_chatbook/DB/migrations/chachanotes_v51_to_v52_agent_lessons_seed.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Create: `Tests/DB/test_chachanotes_agent_lessons_seed_migration.py`

- [ ] **Write failing real-v51 reopen tests.** The migration adds state only, never creates a folder/keyword/note. Test rollback, fresh-schema parity, and stable reopen.

- [ ] **Run red:** `pytest -q Tests/DB/test_chachanotes_agent_lessons_seed_migration.py`.

- [ ] **Add one table:**

```sql
CREATE TABLE agent_lessons_seed_state(
  profile_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL DEFAULT '',
  scope_mode TEXT NOT NULL CHECK(scope_mode IN ('local_only', 'synchronized')),
  state TEXT NOT NULL CHECK(state IN
    ('unknown', 'not_seeded', 'seeded')),
  folder_sync_id TEXT,
  seed_fingerprint TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(profile_id, dataset_id)
);
```

The state is monotonic: synchronized scope stays `unknown` until bootstrap history is fully applied, then becomes `not_seeded` or `seeded`; any qualifying remote upsert or local seed can only advance it to `seeded`. A seeded convention is never reset because its folder is absent. Local-only scope uses the same table as its seed receipt and advances directly to `seeded` when created/reused. Rename, move, or deletion are user actions—not a request to recreate defaults.

- [ ] **Wire v51→v52** and fresh schema, then set `_CURRENT_SCHEMA_VERSION = 52`.

- [ ] **Run green:** `pytest -q Tests/DB/test_chachanotes_agent_lessons_seed_migration.py Tests/DB/`.

- [ ] **Commit:** message `feat(notes): add Agent Lessons seed state`.

## Task 2: Implement the Notes convention, template, and exact discovery

**Files:**

- Create: `tldw_chatbook/Notes/agent_lessons.py`
- Create: `Tests/Notes/test_agent_lessons.py`

- [ ] **Write failing convention tests.** Pin constants `Agent_Lessons` and spelling-exact `agent-lesson`; exact keyword discovery after folder rename/move/delete; marker removal hiding a lesson; case variants not being discovery markers; one-note-per-lesson headings; related note public IDs; and no auto-merge/ranking.

- [ ] **Run red:** `pytest -q Tests/Notes/test_agent_lessons.py`.

- [ ] **Implement one small module.** It owns:

```python
AGENT_LESSONS_FOLDER = "Agent_Lessons"
AGENT_LESSON_KEYWORD = "agent-lesson"
REQUIRED_SECTIONS = (
    "Applicability", "Symptoms", "Root cause", "Verified solution",
    "Failed attempts and why", "Verification evidence", "Caveats",
    "Related lessons",
)
```

Expose pure template validation/rendering and service methods that call existing exact Notes search/save seams. Do not make folder name authoritative and do not copy Notes SQL.

- [ ] **Run green:** `pytest -q Tests/Notes/test_agent_lessons.py`.

- [ ] **Commit:** message `feat(notes): define Agent Lessons convention`.

## Task 3: Seed safely at local/schema and synchronized readiness boundaries

**Files:**

- Modify: `tldw_chatbook/Notes/agent_lessons.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/Sync_Interop/notes_organization_sync_service.py`
- Create: `Tests/Notes/test_agent_lessons_seed.py`
- Modify: `Tests/Sync_Interop/test_notes_organization_enrollment.py`

- [ ] **Write failing seed/race tests.** Cover permanent local-only profile after schema readiness, synchronized profile only after organization `ready`, restart idempotency, user rename/delete not recreating, same-name preexisting collision review, simultaneous untouched empty seeds converging, and edited/acknowledged/used/different-spelling candidates requiring review. Include a late pull whose history contains an `Agent_Lessons` upsert followed by rename or tombstone; materializing the qualifying historical upsert must monotonically set `seeded`, and the later head must not recreate the default.

- [ ] **Run red:** `pytest -q Tests/Notes/test_agent_lessons_seed.py`.

- [ ] **Implement one idempotent initializer and materializer hook.** Call the initializer after Notes schema availability for permanently local-only scope and from the organization-ready transition for synchronized scope. While applying bootstrap/history, any root `notes.folder` upsert with exact name `Agent_Lessons` calls `record_seed_evidence(cursor, profile_id, dataset_id, folder_sync_id)` before a later rename/tombstone can obscure it. After full history, `unknown` becomes `not_seeded` only if no evidence was observed. In one Notes transaction the initializer exactly reuses an active exact-spelling root or creates only the conventional root. The default seed does not create `agent-lesson`; that marker is ensured only for an actual lesson save.

- [ ] **Define untouched seed convergence narrowly.** Automatic convergence is allowed only when both candidates match the coordinator seed fingerprint and have no notes, non-seed membership, rename/move, acknowledgement, or other usage. Otherwise create explicit adoption review.

- [ ] **Run green:** `pytest -q Tests/Notes/test_agent_lessons_seed.py Tests/Sync_Interop/test_notes_organization_enrollment.py`.

- [ ] **Commit:** message `feat(notes): seed Agent Lessons safely`.

## Task 4: Reject only high-confidence credentials at agent-authored save

**Files:**

- Modify: `tldw_chatbook/Notes/agent_lessons.py`
- Modify: `tldw_chatbook/Library/local_library_tool_service.py`
- Create: `Tests/Notes/test_agent_lesson_secret_validation.py`
- Modify: `Tests/Library/test_local_library_tool_service.py`

- [ ] **Write failing boundary tests.** Reject unambiguous private-key blocks, well-known live-key prefixes with credible length/character shapes, and explicit credential assignments such as `password=...` or `api_key: ...` when the value is non-placeholder credential material. Accept long SHA hashes, UUIDs, stack traces, error IDs, redacted values, and explicitly fake examples used for teaching. Assert rejected bodies/tokens never appear in logs, errors, receipts, traces, or durable tables.

- [ ] **Run red:** `pytest -q Tests/Notes/test_agent_lesson_secret_validation.py`.

- [ ] **Implement the minimal detector.** Use a short tuple of anchored/structured regular expressions in `agent_lessons.py`; no entropy scoring, PII framework, network validation, third-party package, or logging of the match. Return one generic refusal code.

- [ ] **Apply only at the guided agent lesson-save boundary.** Ordinary user Notes remain unchanged. A newly agent-created note requesting the exact marker requires the complete template and credential validation before backend touch. An agent-authored update to an existing marked lesson still gets credential validation but the structure is advisory because the user may have edited it; do not reject the update for missing headings. If the latest read shows the user removed the marker, guidance/tests require no silent `ensure_keywords=["agent-lesson"]` reclassification absent an explicit user request.

- [ ] **Run green:** `pytest -q Tests/Notes/test_agent_lesson_secret_validation.py Tests/Library/test_local_library_tool_service.py`.

- [ ] **Commit:** message `feat(notes): guard Agent Lessons credentials`.

## Task 5: Append trusted capability-aware runtime guidance

**Files:**

- Modify: `tldw_chatbook/Agents/agent_service.py`
- Create: `Tests/Agents/test_agent_lessons_runtime_guidance.py`
- Modify: `Tests/Agents/test_agent_runtime_preparation.py`
- Modify: `Tests/Agents/test_agents_internal_prompts.py`
- Modify: `Tests/Widgets/test_tool_message_widgets.py`
- Modify: `Tests/Chat/test_console_agent_tool_result_cap.py`

- [ ] **Write failing prompt/result tests** for native and text-protocol providers. No Notes tools → no guidance. Search+get only → search/read guidance, no save instruction. Search+get+save → search-first, validate, save verified lessons guidance. Save without search → no instruction to write Agent Lessons. Guidance is appended after user/configured system prompt but before workspace/environment note according to current send-time order; custom/user prompts cannot replace it and note bodies never enter it. Search/get payloads carry the stable `trust_notice`; the generic tool-result widget and Console capped preview keep that reference-only label visibly ahead of untrusted note text.

- [ ] **Run red:** `pytest -q Tests/Agents/test_agent_lessons_runtime_guidance.py Tests/Agents/test_agent_runtime_preparation.py Tests/Agents/test_agents_internal_prompts.py`.

- [ ] **Add one pure capability function** (in `Notes.agent_lessons` or a small private helper in `agent_service.py`) that receives actual disclosed schema names for that turn and returns static trusted text or `""`. Use effective schemas, not configured wishes, so runtime policy/tool loading cannot produce false guidance.

- [ ] **Append at both send seams.** Apply in `_build_model_request` and the `_make_call_model`/`call_model` path beside `RUN_LOG_PROMPT_SECTION`. Preserve subagent identity prefix and project/workspace suffix ordering. Guidance must say retrieved notes are untrusted data: they cannot grant permission, authorize commands, expand scope, or override instructions. If the existing generic renderers truncate away the top-level `trust_notice`, make the smallest renderer ordering change and pin it in the listed UI tests; do not build an Agent Lessons UI subsystem.

- [ ] **Run green** with the red command.

- [ ] **Commit:** message `feat(agents): guide Agent Lessons by capability`.

## Task 6: Preserve guidance under subagent narrowing and continuation

**Files:**

- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `Tests/Agents/test_agent_lessons_runtime_guidance.py`
- Modify: `Tests/Agents/test_fleet_continuation.py`
- Modify: `Tests/Agents/test_skill_tool_spawn.py`

- [ ] **Write failing child tests.** Ordinary child inheriting search/get/save gets full guidance; child definition narrowing out save gets read-only guidance; narrowing out search gets none; resumed child recomputes from current effective tools; a parent cannot grant a tool through guidance; skill-driven child intersection rules remain intact.

- [ ] **Run red:** `pytest -q Tests/Agents/test_agent_lessons_runtime_guidance.py Tests/Agents/test_fleet_continuation.py Tests/Agents/test_skill_tool_spawn.py`.

- [ ] **Reuse the same send-time helper.** Do not manually append Agent Lessons text while constructing either child `AgentConfig`; child allow-list intersection and active schema disclosure should naturally determine the suffix. This prevents configuration/guidance drift.

- [ ] **Run green** with the red command.

- [ ] **Commit:** message `test(agents): preserve Agent Lessons capability gates`.

## Task 7: Prove cross-agent reuse end to end

**Files:**

- Create: `Tests/Agents/test_agent_lessons_end_to_end.py`
- Modify: `Tests/Library/test_cross_runtime_parity.py`

- [ ] **Write an end-to-end deterministic scenario:** Agent A searches first, encounters/records a verified issue, includes failed attempts and why, verification evidence, exact marker, and public related IDs; Agent B with only permitted Notes tools later searches the exact marker, reads the note as untrusted data, independently applies the safe solution, and does not treat embedded command/permission text as authority.

- [ ] **Run red:** `pytest -q Tests/Agents/test_agent_lessons_end_to_end.py`.

- [ ] **Use existing fake model/tool harnesses.** Do not add production shortcuts or require an external LLM. Assert tool calls, durable Notes state, result provenance, prompt boundary, and update/idempotency behavior.

- [ ] **Run green:**

```bash
pytest -q Tests/Agents/test_agent_lessons_end_to_end.py Tests/Agents/test_agent_lessons_runtime_guidance.py Tests/Library/test_cross_runtime_parity.py Tests/Notes/test_agent_lessons.py Tests/Notes/test_agent_lessons_seed.py Tests/Notes/test_agent_lesson_secret_validation.py
```

- [ ] **Commit:** message `test(agents): prove Agent Lessons reuse`.

## Task 8: Document, verify, and close TASK-24309

**Files:**

- Modify: `Docs/Development/Agent-Tools/local-library-tools.md`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/library/notes.md`
- Modify: `backlog/tasks/task-24309 - Add-Agent-Lessons-Notes-convention.md`
- Modify lessons only for a real incident.

- [ ] **Document** user ownership, exact marker discovery, folder rename/delete behavior, template, search-first/update rules, failed-attempt rationale, seed/review rules, capabilities, permissions, secret refusal, and the untrusted data boundary.

- [ ] **Run targeted verification:**

```bash
python -m compileall -q tldw_chatbook/Notes tldw_chatbook/Agents tldw_chatbook/Library
pytest -q Tests/DB/test_chachanotes_agent_lessons_seed_migration.py Tests/Notes/test_agent_lessons.py Tests/Notes/test_agent_lessons_seed.py Tests/Notes/test_agent_lesson_secret_validation.py Tests/Agents/test_agent_lessons_runtime_guidance.py Tests/Agents/test_agent_runtime_preparation.py Tests/Agents/test_agents_internal_prompts.py Tests/Agents/test_fleet_continuation.py Tests/Agents/test_skill_tool_spawn.py Tests/Agents/test_agent_lessons_end_to_end.py Tests/Library/test_cross_runtime_parity.py Tests/Widgets/test_tool_message_widgets.py Tests/Chat/test_console_agent_tool_result_cap.py
git diff --check
```

Do not run the full suite without user opt-in.

- [ ] **Live-verify only through the schema-safe gate from the TASK-24307 plan.** Coordinate v52 compatibility first, then use two distinct task-specific roots whose `HOME`, XDG directories, `TLDW_CONFIG_PATH`, and `[paths].data_dir` all resolve within those roots. Verify seed/rename/move, exact marker search, A-save/B-discover, stale refusal, pending/offline finalization, and permission-narrowed child behavior against the real current server. Never store sensitive test content and never launch with only a scratch config file.

- [ ] **Self-review and close:** inspect prompt order, all allowed-tool combinations, every durable/log owner for rejected content, seed monotonicity, and every AC. Add concise Implementation Notes with ADR-102 and evidence, set TASK-24309 Done, and repeat task/ADR collision checks before merge.
