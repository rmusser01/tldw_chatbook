# Agent Lessons and Notes Organization Sync Design

**Date:** 2026-08-29

**Status:** Owner-approved and independently reviewed; awaiting final written-spec review

**Server baseline reviewed:** `tldw_server` `origin/dev` at `1ad2f1e5b30c49ea75396e4b713496b73e875fec`

## Summary

Chatbook will add a default, user-manageable Notes folder named `Agent_Lessons`. Agents with Notes permission will use it to record verified, reusable solutions, including approaches that failed and why. Other agents will discover those lessons through the existing Notes search capability when they encounter similar symptoms. Human corrections and observed outcomes are evidence rather than truth: the agent verifies them, explains the generalizable principle and its rationale, and shows a concise preview before the user approves a save.

This feature will not introduce a new server sync domain or a separate agent-memory store. The server already defines a complete, indivisible six-domain Notes organization group. Chatbook will first consume that existing contract, then extend its Notes tools, and finally add the Agent Lessons convention and agent guidance on top.

The authoritative discovery marker is the exact keyword `agent-lesson`. The folder is the default visible organization, but it cannot be the sole identity mechanism because users may rename or delete it. A verified lesson may later support a small human-reviewed proposal against an authorized user-owned skill or repository instruction file, but the lesson itself remains untrusted evidence and never changes instructions automatically.

## Problem

Agents repeatedly spend time rediscovering solutions to problems already solved in earlier interactions. Ordinary Notes can hold the knowledge, but the application currently lacks:

- a recognizable default location for reusable agent lessons;
- a consistent lesson format that captures verification and failed approaches;
- a low-friction, user-approved way to capture corrective feedback and explain why it generalizes;
- agent-facing guidance to search before retrying work and save only proven solutions;
- a safe boundary for proposing, reviewing, verifying, and recording an instruction improvement without treating memory as instruction authority;
- exact folder and keyword scopes in the Notes tools;
- Chatbook support for the server's portable Notes organization model.

Without portable folder and keyword organization, lessons cannot be reliably found across synchronized Chatbook devices. Folder-name-only discovery would also break when a user exercises normal ownership by renaming or deleting the folder.

## Goals

1. Make verified lessons discoverable by later primary agents and subagents through normal Notes search.
2. Preserve failed attempts and explanations so later agents do not repeat them.
3. Keep lessons in the user's ordinary Notes system, under existing Notes permissions and synchronization.
4. Consume the server's complete six-domain Notes organization group without inventing a parallel protocol.
5. Preserve user ownership: the folder and notes remain visible, editable, movable, renamable, and deletable.
6. Make enrollment, migration, and interrupted synchronization lossless and reviewable.
7. Treat retrieved lessons as untrusted reference material, never as authority.
8. Capture high-quality feedback or observed signals with concise, privacy-preserving provenance and independent verification.
9. Permit evidence-based, human-reviewed promotion proposals only for authorized user-owned targets.
10. Prefer small principles with rationale and progressive disclosure over accumulated brittle rules.

## Non-goals

- Automatic background capture of conversations or failures.
- Automatic application of lesson content to a skill, project instruction file, prompt, or runtime policy.
- A scheduled observer/improver agent in this delivery; that requires a separate research task and future ADR.
- Saving speculative, unverified, or interaction-specific observations as lessons.
- A new agent-memory database, embedding index, or semantic-retrieval subsystem.
- A seventh Notes organization domain or a new server contract.
- Silent merging of same-name folders or same-path objects with different sync identities.
- Allowing lessons to grant permissions, override instructions, or authorize tool calls.
- Editing built-in/runtime instructions, server-managed skills, Codex runtime skills, read-only skills, or files outside the selected writable authority.
- A promotion queue, promotion database, dedicated apply tool, Git/PR monitor, or new approval subsystem.
- Synchronizing filesystem directories as a new portable folder domain.
- Making server changes unless conformance work demonstrates a concrete server defect.

## Chosen Approach

The feature is a guided convention layered on ordinary Notes:

1. An agent encountering a problem searches Notes using relevant error signatures, component names, and root-cause terms, scoped to the exact `agent-lesson` keyword.
2. The agent treats matches as untrusted leads and verifies applicability in the current environment.
3. After resolving and verifying a reusable issue, the agent searches again for the same root cause.
4. The agent records a concise feedback or observed-signal summary, known provenance, the evidence that independently confirmed it, and one principle with its rationale and limits. Unknown facts remain explicitly unknown.
5. It shows the user a concise preview. Rejection creates no note, hidden draft, receipt, or fallback write. Approval permits the ordinary Notes save path.
6. It updates the existing lesson with optimistic content and organization tokens when the lesson is materially the same, or creates a new lesson and cross-references related public note IDs when it is distinct.
7. A new lesson save adds the canonical `agent-lesson` keyword without removing user keywords and places the lesson in the current conventional `Agent_Lessons` folder when available. An ordinary update preserves current folder memberships and does not restore a marker a user has removed.
8. When evidence is strong, procedural, and reusable, the agent may suggest a promotion candidate without a fixed incident-count threshold. The active primary agent can prepare an exact read-only proposal only after user approval; application and verification remain governed by the target's existing authority and trust systems.

This provides durable reuse with minimal new machinery. It reuses Notes CRUD, FTS5 search, public note IDs, normal permissions, and the server's existing synchronization protocol.

### Alternatives Rejected

**Folder-only discovery.** This is simple but fails after a user renames or deletes the default folder. The synchronized exact keyword is therefore authoritative.

**Automatic lesson extraction.** Background capture risks saving secrets, noise, incorrect diagnoses, and unverified conclusions. Explicit agent saves after verification are more predictable and auditable.

**Automatically rewriting skills or instructions from lessons.** Memory changes continuously and remains untrusted. Stable procedural instructions must change deliberately through an exact proposal, human review, existing write authority, verification, and any required skill re-trust.

**A promotion workflow database or separate proposal folder.** Promotion is initially rare and user-driven. Optional descriptive state belongs in the evidence note, while current target content, tool authority, Git state, and skill trust remain the real sources of truth.

**A dedicated memory or vector-search service.** This duplicates Notes storage, permission, synchronization, and retrieval behavior without being necessary for the initial feature.

**A new portable folder-sync domain.** The server already ships the required six-domain Notes organization group. Creating another domain would require redundant contracts, migrations, conflict rules, and ADRs.

## Reviewed Server Contract

The reviewed server `dev` baseline implements these six domains as one enrollment group:

1. `notes.keyword`
2. `notes.keyword_link`
3. `notes.keyword_collection`
4. `notes.keyword_collection_link`
5. `notes.folder`
6. `notes.folder_link`

All use schema version 1, server-trusted validation, and upsert/tombstone operations. A client must enroll all six; it must not advertise or attempt partial group support. The group complements `notes.note`; conversation keyword links additionally depend on the applicable conversation domain.

Normative server sources include:

- `tldw_Server_API/app/core/Sync/v2/notes_organization.py`
- `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_organization.py`
- `tldw_Server_API/app/core/Sync/v2/materializers/notes_organization.py`
- `tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py`
- `Docs/superpowers/specs/2026-08-08-notes-organization-sync-design.md`
- `Docs/API/Sync_V2_M1.md`
- server ADR-031, ADR-034, and ADR-035

The implementation plan must revalidate these sources against the then-current server `dev` head before coding. Chatbook will duplicate only the small pure contract functions and normative fixtures needed for interoperability; it will not add a shared package or dependency solely for this feature.

### Contract Constraints

- Portable resource sync IDs are canonical lowercase UUIDv4 strings. Existing local database IDs remain local and receive separate sync-ID columns.
- Link IDs are deterministic SHA-256 hashes of the server-defined canonical JSON identity payload. Chatbook must pass the server's normative vectors exactly.
- Folder upserts carry a stripped, non-empty name of at most 500 characters and a nullable parent sync ID.
- Portable derived folder paths are relative and at most 500 characters. Folder names must be one relative segment: they reject `.`, `..`, `/`, and `\`.
- Server collision identity uses case-folding, not NFKC normalization. Chatbook's portable validation and collision checks must match that behavior.
- Resource tombstones have empty payloads. Link tombstones retain their identity payloads.
- Resource tombstones do not cascade to child resources or links. Dormant relationships become effective again if their resource is restored.
- Effective folder membership is `(manual UNION source-managed) MINUS suppressions`.
- A canonical folder-link upsert clears suppression and ensures manual membership. A canonical tombstone removes manual membership and adds suppression without deleting source provenance.

## Chatbook Data Model and Migration

Chatbook will preserve local primary keys and add stable portable sync identities to the existing organization records. It will add the minimal suppression, synchronization-state, and durable-intent data required by the server contract. It will not repurpose local folder IDs as UUIDs.

Migration durably assigns a UUIDv4 sync ID to every legacy organization resource, including unpublished and soft-deleted resources, before adoption or publication decisions. Identity allocation is independent from publication state, so IDs remain stable across restarts and unpublished hierarchies can reference one another. Deterministic link IDs are derived only after their referenced resource IDs are assigned. Publication dependencies are processed in order: resources before links, parents before children, and referenced Notes or conversations before their organization links. Existing soft-deleted history is represented as an upsert followed by a tombstone so another device can reconstruct identity and deletion.

The current local folder model uses absolute display paths with a leading slash, NFKC plus case-fold collision keys, and cascading subtree deletion. Portable behavior must instead follow the server contract:

- the local leading slash is excluded from the portable 500-character path limit;
- portable collision checks use server-compatible case-folding;
- explicit canonical deletion is distinct from a descendant becoming effectively hidden beneath a deleted ancestor;
- deleting a folder must not emit unintended child tombstones;
- legacy deleted cohorts may be explicitly backfilled where necessary to preserve their already-recorded state.

Relaxing the portable comparison from NFKC-plus-case-fold to case-fold-only must be guarded by migration analysis and conformance tests. Existing collisions are preserved for explicit review rather than silently renamed or merged.

Logical Notes folder paths and filesystem paths remain separate boundaries. A logical folder valid under the sync contract remains storable and searchable even if a filesystem adapter cannot represent it. Filesystem sync must report or mark that boundary failure and must not silently truncate or transform the portable identity.

## Enrollment and Adoption

The six-domain group has a durable client enrollment state:

```text
initializing -> pulling -> adoption review -> ready
      \           \             \
       +-----------+-------------+-> failed -> retry from durable checkpoint
```

1. **Initializing:** register all six domains and wait for the server group bootstrap to become ready.
2. **Pulling:** apply the complete remote snapshot and retained history before publishing local organization state.
3. **Adoption review:** compare legacy local objects with portable server objects without guessing identity from equal names or paths.
4. **Ready:** enable ordinary local organization mutations and durable outbound synchronization.

The `failed` state retains the last durable checkpoint and a non-sensitive error reason. A retry resumes the safe phase or returns to initialization when the server requires a new bootstrap; it does not discard pulled heads, local candidates, or adoption decisions.

All six organization domains are registered together, but their referenced subject domains remain explicit dependencies. Note folder and note keyword links are publishable only when the corresponding `notes.note` identity/head is available. Conversation keyword links are publishable only when the corresponding `chat.conversation` domain and identity/head are available. A link whose subject dependency is not enrolled remains local and reviewable rather than being sent as an invalid envelope; this does not permit advertising partial support for the six-domain organization group.

During enrollment, ordinary note content edits continue. General organization writes are blocked until the group is ready, except for the explicitly defined pending Agent Lessons flow below.

If a local and server object have the same visible path but different sync IDs, the user receives reviewable choices:

- merge the local content and memberships into the server object;
- rename and publish the local object as distinct; or
- keep the local object unpublished/local.

No choice drops data. Server canonical heads remain authoritative for their sync identities, while conflicting local candidates remain recoverable until resolved.

Enrollment and retry are resumable after interruption. Chatbook must not claim the group ready until bootstrap, pull, materialization, and required adoption decisions are complete.

## Durable Synchronization Intent

The existing note outbox path can log and swallow an enqueue failure after a local mutation. Organization support must close that loss window.

Each ready-state Notes operation records its local mutation and immutable, version-bound synchronization intents in one transaction in the Notes database. An intent contains the fully normalized canonical envelope draft needed for later publication: domain and schema version, operation, object or link ID, payload, dependency references, source local revision, required base/head token, and a stable idempotency or mutation ID. Dispatchers must not reconstruct an older intent from whatever mutable row state happens to exist later.

A dispatcher may copy an intent into the general sync outbox, recording the outbox identity on the local intent. The intent remains retryable until the matching server acknowledgement is durably recorded; acknowledged intents can then be compacted under an explicit retention policy. A crash at any point before acknowledgement replays the same idempotent mutation rather than creating a new logical operation. Cross-database atomicity is not assumed.

Outbound ordering publishes referenced notes and organization resources before their links. The server group is eventually consistent across envelopes; Chatbook does not assume a multi-envelope server transaction.

For source-managed folder memberships, local provenance is retained and canonical suppression is projected exactly as server ADR-035 requires.

## Notes Tool Contract

### `library_search_notes`

Add optional exact filters:

- `folder_id`: preferred stable opaque public folder identity returned by search and folder APIs;
- `folder`: exact relative folder path for callers that do not yet have a public identity;
- `keyword`: spelling-exact whole-keyword match after trimming, never a substring or case-folded name match.

`folder_id` and `folder` are alternative forms of the same filter and cannot disagree. Relative path resolution uses server-compatible segment validation and case-fold matching. An ambiguous path returns an explicit conflict rather than selecting a folder. When a folder filter and keyword filter are both present, both apply. The existing lexical query and pagination behavior remain; the feature does not add semantic retrieval. Results include bounded folder metadata—stable public folder ID, display name, and relative path—plus an opaque `organization_version` derived from the returned note's locally known folder and keyword link heads. Results retain the existing stable public note ID.

Agent Lessons discovery normally supplies `keyword="agent-lesson"`. Folder metadata is for organization and display, not a new relevance-ranking algorithm; the current folder is not a required scope because users can rename or move it. The existing response-size fitter continues to cap oversized results.

### `library_get_note`

Return the same bounded folder/keyword metadata and current opaque `organization_version` with each note read. This is required because search results are match-oriented and a version-locked update normally re-reads the complete note through `library_get_note`. Continuation reads may return the current organization token on every page; organization changes do not invalidate the existing content cursor, but an update must use the most recently returned token.

### `library_save_note`

Add an optional additive `ensure_keywords` field. The operation ensures the named whole keywords are attached while preserving every existing user keyword. Creation and attachment still obey the server's trim and case-fold uniqueness/conflict rules, but name-based search remains spelling-exact so a differently cased resource cannot become the Agent Lessons marker implicitly. The operation never treats absence from `ensure_keywords` as a request to remove a keyword.

The tool coordinates note content, requested folder membership, additive keywords, and durable synchronization intent through the Notes transaction boundary. Updates retain optimistic `expected_version` checks. Any update that requests an organization change also accepts the opaque `expected_organization_version` returned by search/read and fails if locally known folder or keyword link heads changed. On either stale token, an agent must re-read and merge rather than overwrite.

Agent Lessons uses folder placement only when creating a new lesson or finalizing a pending lesson. An ordinary lesson update does not supply a folder change. It supplies `ensure_keywords=["agent-lesson"]` only when the latest read still contains the canonical marker; if a user already removed the marker, the agent does not silently reclassify the note without an explicit user request. A keyword removal or folder move racing the save is detected by `expected_organization_version` and becomes a reviewable conflict.

High-confidence credential material is rejected at the agent-authored lesson save boundary with a structured validation error. Rejected content is not logged. The first version uses a small dependency-free detector limited to unambiguous formats such as private-key blocks and recognized live-token prefixes or credential assignments; it does not invent a general entropy scanner or PII classifier. Hashes, error IDs, and clearly fake/example values must not be rejected merely for being long. Avoiding personal data and large raw logs is also explicit agent guidance, but it is not represented as a claim of perfect PII detection.

Permission denial creates no note, folder, keyword, pending receipt, or hidden fallback write.

The preview-before-save behavior reuses the existing tool-review path rather than adding a Notes API, modal, or durable approval system. The agent first narrates whether the operation creates or updates a lesson and summarizes the title, applicability, root cause, verified solution, failed attempts, verification, provenance, generalizable principle, and any promotion nomination.

The immutable `library_save_note` call enters an explicit existing approval round whenever it:

- requests exact `agent-lesson` keyword assurance;
- targets a note currently carrying that exact marker; or
- targets a note owned by an Agent Lessons `pending-organization` or `placement-review` receipt.

This classification overrides a broader ordinary-Notes allow setting. The approval card shows a compact summary derived from the exact call plus a digest. Its ephemeral approval stamp is bound to the run, immutable call digest, note identity or create operation, observed Agent Lesson classification, content/organization preconditions, and receipt state/version where applicable. Rejection/cancellation prevents executor entry and creates no state; an edited payload is a new call requiring a new preview and approval. This is a narrow policy on the existing approval infrastructure, not a parallel subsystem.

The Notes transaction boundary receives the trusted run role and approval stamp. In the same transaction that would mutate content or organization, it recomputes the exact marker plus pending/placement-receipt classification and validates that it matches the reviewed classification. Any addition, removal, or transition of marker/receipt state after review fails without mutation as stale/`approval_required` and requires a fresh preview. A subagent request for any classified Agent Lesson state fails before mutation with structured `foreground_required`, even if the note is still pending and has no marker, and returns its draft/evidence to the primary. If search is unavailable, the primary may show an unsaved draft but must not issue the final lesson-save call because duplicate/root-cause discovery did not run.

## Agent Lesson Format

One note records one reusable lesson. Agent-created lessons use a stable Markdown structure:

```markdown
# <concise problem and resolution>

## Applicability
Repository/project, component, platform, relevant versions, and other scope limits.

## Symptoms
Observable errors or behavior, including useful exact signatures.

## Root cause
The verified explanation.

## Verified solution
The smallest reproducible resolution.

## Failed attempts and why
What was tried, what happened, and why it did not solve the problem.

## Verification evidence
Concise commands, tests, observations, or other evidence and results that confirmed the resolution. Do not include large raw logs.

## Feedback or observed signal and provenance
Source type, known date and environment, a concise signal summary, and safe stable evidence references. Use `Unknown` instead of guessing. Do not copy raw conversations or personal identity.

## Generalizable principle
The small reusable principle, why it generalizes, and when it should not be applied.

## Caveats
Known limits, risks, or conditions where the solution should not be applied.

## Related lessons
Stable public note IDs for distinct but related lessons.
```

If no failed approach occurred, the section says `None; the first tested approach succeeded` rather than inventing work. The structure is advisory for user-edited notes but required for newly generated agent lessons. Agents should keep exact error signatures only when useful and sanitized, keep evidence concise, and omit secrets, personal data, and large raw logs.

An optional promotion section is added only when the evidence supports a proposal:

```markdown
## Promotion candidate
Status: Proposed | Applied | Rejected | Reverted or superseded
Target hint: <logical skill name or repository-relative instruction scope>
Principle: <one small procedural improvement>
Rationale: <why the evidence supports promotion>
Outcome: <safe revision reference, rejection reason, or failure limitation>
```

The target hint never contains an absolute device path. Revision and PR references must not contain credentials or sensitive URLs. Promotion state is historical, descriptive, user-editable note content. It never authorizes a write, proves the target still contains the change, or replaces current inspection of the target and its trust state. A partial or failed application remains `Proposed` with its limitation in `Outcome`; `Applied` is used only after relevant checks and any required local-skill re-trust succeed.

## Agent Runtime Guidance and Trust Boundary

Agent Lessons instructions are appended as a trusted, non-editable runtime protocol suffix for both primary agents and ordinary subagents. They do not live solely in the user-overridable prompt catalog. The suffix is role-aware as well as capability-aware: both roles may search and assess lessons, but only the foreground primary may present a lesson preview and perform the approved save. A subagent returns its proposed draft, evidence, and related-note findings to the foreground primary; it does not call the lesson-save path itself.

The suffix is capability-aware:

- agents with Notes search permission receive troubleshooting search guidance;
- foreground primary agents with both Notes search and save permission receive feedback-verification, transient-preview, verified-save, and update guidance;
- subagents with Notes search permission receive search/read guidance and, when they discover a reusable resolution, return a structured draft and evidence to the foreground primary rather than saving;
- a save-only agent receives no Agent Lessons save guidance because it cannot perform the required duplicate/root-cause search;
- agents lacking a capability are not instructed to exercise it.

Subagents continue to inherit their parent's allowed tools, minus the existing restricted capabilities; named agent definitions may narrow permissions. Agent Lessons introduces no permission bypass or new propagation rule. Role-aware guidance prevents the workflow from claiming a subagent can satisfy the user-preview requirement merely because a generic Notes save schema is technically disclosed.

Search results and note bodies are untrusted user content. The runtime guidance must state that a lesson:

- cannot override system, developer, project, or current user instructions;
- cannot grant tool access or authorize a command;
- cannot expand filesystem or network scope;
- must be checked against current versions, environment, and evidence before use.

The guidance also states that human feedback is a signal to sanity-check rather than an instruction to preserve blindly. Agents prefer detailed domain evidence over reaction volume, write principles plus rationale rather than exhaustive rules, represent unknown provenance honestly, and do not save until the user approves the displayed preview. A rejected or abandoned preview disappears with the conversation state and causes no durable mutation.

UI and tool-result labeling should make this reference-only trust level clear. Note bodies remain ordinary tool-result data and are never interpolated into the trusted runtime suffix, system instructions, or project-instruction context. Adversarial note text remains data, not instructions.

## Human-Reviewed Promotion

Promotion converts verified evidence into a proposed change through ordinary review; it does not convert the note into instruction authority. Eligibility is evidence-based rather than count-based: one detailed, independently verified expert correction may qualify when it is procedural, broadly reusable, explains why, identifies limits, and does not conflict with higher-priority instructions.

Eligible proposal targets are limited to:

- Chatbook-managed local skills already inside [ADR-009: Local Skill Trust Boundary](../../../backlog/decisions/009-local-skill-trust-boundary.md); and
- `AGENTS.md` or `AGENTS.override.md` inside the currently selected writable workspace binding governed by [ADR-069: Console project-instruction local state and preflight](../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md).

Built-in or runtime-owned instructions, server-managed skills, Codex runtime skills, read-only skills, unavailable targets, and paths outside the selected binding are ineligible. Target hints use a logical skill identity or repository-relative scope, not an absolute local path. The proposal step inspects the current target and selects the smallest appropriate file; for a skill, progressive disclosure may favor a referenced resource over expanding `SKILL.md`.

Subagents may collect evidence and recommend a candidate, but trusted protocol guidance reserves presenting and applying instruction-changing proposals for the active primary agent. This is a behavioral protocol layered on existing security—not a new restriction on generic filesystem tools. Existing workspace/tool review and local-skill trust checks remain the enforcement boundaries.

Application paths are target-specific. Repository instruction proposals may use approved workspace file tools only after those mutation seams support the compare-and-swap rule below. Chatbook-managed local skills are protected application data outside the selected workspace binding and must never be edited through raw filesystem tools. The current Console exposes no managed-skill mutation tool, so the initial promotion feature is proposal-only for local skills: the user applies an accepted proposal through the existing Library editor/service, which calls `LocalSkillsService.update_skill(expected_version=..., trust_approved=False)`. The primary agent may re-read and verify the result afterward. A later agent-controlled application path would require the promotion ADR to identify an existing application-controlled action with the same version and trust transitions; it must not bypass the service.

The primary-agent flow is:

1. Re-read the lesson as untrusted evidence and inspect the current authorized target, surrounding context, existing user edits, and trust/revision state. For repository instructions, capture the selected binding ID, locator fingerprint, effective applicable instruction chain, and full target-content digest (or an explicit absent-file state). For a managed skill, capture the service version and trust state.
2. Ask the user whether to create a proposal. Approval permits a read-only exact diff preview, not a write.
3. Present one focused diff with the captured target state, rationale, expected effect, verification plan, application path, and any activation limitation.
4. Immediately before repository-instruction application, revalidate the selected binding ID and locator fingerprint, recompute the effective applicable instruction chain, and compare the exact preview target/diff. A new or changed applicable instruction, retargeted binding, target change, or proposal change invalidates the prior approval and requires a new preview. For a managed skill, the Library service must still hold the captured expected version and compatible trust state.
5. After explicit approval of that exact preview, apply a repository-instruction mutation through the existing filesystem/tool review. The reviewed mutation carries the same target, expected full-content SHA-256 digest (or expected-absent state), and replacement content represented by the preview. The write boundary checks the expectation and performs a path-safe atomic same-directory replace/create; mismatch fails without writing. Preserve unrelated and pre-existing user edits; never reset them automatically after a failure. Managed-skill proposals instead wait for manual application through the Library editor/service.
6. Run the smallest relevant existing deterministic check or golden scenario. When behavioral verification is unavailable, require explicit domain review and label effectiveness unverified rather than manufacturing proof.
7. A changed Chatbook-managed local skill remains inactive until ADR-009's reviewed re-trust succeeds. A repository instruction change can affect only a later run or lazy activation; the current agent must not claim to have inherited it.
8. Report the outcome and offer a separate concise Notes update. That update also requires approval.

The preview state is ephemeral interaction state, not a durable approval receipt or promotion database; the actual repository mutation still enforces the approved expectation atomically at its existing write boundary. `Applied` means the write, relevant checks, and required re-trust completed. A synchronized outcome remains historical: another device must independently locate and inspect the current target. Only a discoverable lesson whose user-approved update records `Rejected` suppresses a later identical suggestion; otherwise rejection is ephemeral and recurrence cannot be prevented. Materially new evidence may justify reconsideration.

## Default Folder Lifecycle

For a synchronized profile, `Agent_Lessons` is seeded as a root Notes folder only after the six-domain organization group is ready and its snapshot/history has been applied. For a permanently local-only profile, it is seeded after the local organization schema migration is ready; no server enrollment state is required. In both modes, an idempotent Notes-service initializer runs when the profile's Notes service becomes available and again when a synchronized profile transitions to ready, so the default folder is visible without waiting for an agent save or application restart. The seed is not a blind schema-migration insert.

Seeding rules:

1. Reuse an active root folder with the exact conventional spelling. A case-fold-equivalent but differently spelled root is an adoption conflict, not an automatic match.
2. Only notes carrying the exact `agent-lesson` keyword participate in Agent Lessons discovery; reusing the folder does not reclassify unrelated notes.
3. Determine whether the conventional folder was seeded before by inspecting synchronized `notes.folder` upsert history for a former root named `Agent_Lessons`, even if its current head is renamed or tombstoned.
4. Store a dataset/profile-scoped monotonic seed state: `unknown` until bootstrap history is fully applied, then `not-seeded` or `seeded`. Materializing any qualifying remote upsert changes it to `seeded`; local seed or exact reuse persists `seeded` in the same transaction. A cached false value can never suppress evidence received by a later pull.
5. For a local-only database, persist an equivalent local seed receipt.
6. Do not recreate a folder merely because the user renamed or deleted the seeded folder.
7. If a later agent explicitly saves a new lesson and no current conventional folder exists, recreate the conventional folder for that save without restoring or modifying the former folder.

Two devices can race after bootstrapping the same empty dataset. If both create the exact initial default folder, the losing device may automatically adopt the winning remote folder, cancel its candidate's unacknowledged publication intent, and retire that unpublished candidate only when it is still an untouched, empty, coordinator-created seed. Any edit, membership, acknowledgement, different spelling, or unrelated use requires adoption review.

The reviewed server currently retains envelope history non-destructively and bootstraps deleted resources as upsert then tombstone. If future retention becomes destructive, the server/client contract must preserve an equivalent durable seeded-before marker before relying on compaction.

## Pending Agent Lesson Flow

A verified lesson must not be lost merely because organization enrollment is still initializing or the canonical keyword is awaiting collision review. If the foreground primary has permission, has shown the preview, receives approval, and requests a lesson save before organization can be safely finalized:

1. Save the ordinary note and a content-free pending-organization receipt in one local Notes transaction.
2. Keep the pending note local-only; do not attach folder/keyword organization or publish it as a completed lesson yet. Every normal note dispatcher excludes note IDs whose receipt is in the blocking `pending-organization` state.
3. Include it in originating-device Agent Lessons searches through the pending receipt, clearly labeled pending.
4. Once the group is ready and the canonical keyword is resolved, atomically attach the canonical keyword, create the immutable note/resource/link sync intents, and attach the current conventional folder when unambiguous. If placement succeeds, clear the receipt. If placement is ambiguous, transition it to the non-blocking `placement-review` state with the desired conventional placement and collision IDs before allowing publication. The blocking state cannot become invisible to dispatch until all required publication intents and any required review record exist in that same transaction.

The receipt stores only stable local identity and desired organization intent, not lesson content. A `placement-review` receipt survives restart, does not block note/keyword publication or discovery, and remains until the user resolves or dismisses the placement conflict. Deleting the note cancels either receipt state. A permission denial creates nothing. Crash recovery tests must cover each boundary before, during, and after finalization, outbox copy, server acknowledgement, and local acknowledgement cleanup.

Both receipt states continue to classify their note as an Agent Lesson for role-aware preview/approval even when the exact keyword is not yet attached. Content updates to those notes therefore pass through the same foreground-primary approval and transaction-time classification check as marked lessons. Receipt finalization, transition, dismissal, marker attachment/removal, and concurrent note updates cannot downgrade that requirement between review and execution.

The server's uniqueness rules are case-insensitive. A differently spelled case-fold-equivalent folder such as `agent_lessons` is never silently adopted as `Agent_Lessons`; it produces a durable placement review. Once the canonical keyword is safely established, that folder conflict does not block keyword-based lesson completion or discovery—the note is saved with the keyword and without conventional placement until reviewed. A differently spelled case-fold-equivalent keyword such as `Agent-Lesson` requires adoption/rename review because automatically treating its existing memberships as Agent Lessons could reclassify unrelated user notes. Name-based `keyword="agent-lesson"` search is spelling-exact and returns none of those variant memberships. Until the canonical marker conflict is resolved, the new lesson remains locally pending.

If a coordinator-created folder or keyword loses a cross-device race, automatic repair is permitted only when the losing object was created for that exact pending save, or was the untouched empty initial default seed described above, and remains unedited and unrelated. Repair adopts the winning canonical resource and transfers only the losing pending save's intended membership before retiring the unpublished candidate. User edits, non-conventional spelling, or unrelated memberships require explicit review. The lesson content remains intact even when placement requires review.

For a permanently local-only profile, the Notes service can complete the organization locally without waiting for a server enrollment state.

## Discovery, Updates, and Deletion

Lexical retrieval remains the discovery mechanism. Agent guidance should search using combinations of:

- exact error signatures;
- component or subsystem names;
- affected platform and versions;
- suspected or confirmed root-cause terms.

Duplicates are tolerated because silently coalescing diagnoses is riskier than retaining distinct evidence. Agents cross-reference related public note IDs and update only when the root cause and applicability are materially the same. Updates preserve the latest user-controlled organization state and require both content and organization concurrency tokens when organization is involved.

Feedback quantity is not a ranking signal. Conflicting evidence is retained and explained rather than resolved by vote count. A lesson marked `Applied`, `Rejected`, or `Reverted or superseded` does not cause automatic Git, PR, target-file, or cross-device monitoring.

Removing the exact `agent-lesson` keyword removes a note from Agent Lessons discovery even if it remains in the folder. Deleting the note also removes it. Renaming or moving the folder does not remove discovery while the keyword remains.

## Failure Handling and Observability

- Search failure does not block troubleshooting; the agent reports the unavailable lookup and continues within its existing authority. It may show an unsaved lesson draft but does not finalize a new lesson until duplicate/root-cause search succeeds.
- Rejected or abandoned lesson previews and promotion proposals create no durable object, receipt, or hidden fallback write.
- A lesson changing after preview triggers normal optimistic-concurrency refusal and a new preview rather than overwrite.
- Unknown provenance, versions, and validation limits remain explicit; the agent does not synthesize missing facts or failed attempts.
- Save validation errors return actionable structured reasons without logging rejected content.
- Enrollment and durable-intent dispatch expose initializing, pending, review-required, retrying, and ready states rather than silently dropping work.
- Offline operation after ready records durable intent and synchronizes later.
- Contract-invalid remote envelopes fail closed into the existing conflict/review path.
- A pending lesson remains locally searchable and visibly pending until finalization or deletion.
- A stale promotion digest, changed binding fingerprint/effective instruction chain, unavailable/ineligible target, denied write, failed validation, or incomplete local-skill re-trust leaves the proposal unapplied and reports the precise non-sensitive limitation. Existing user edits are never automatically reset.
- Metrics and logs use IDs, domain names, state transitions, and error classes, not note bodies or credentials.

## Delivery Sequence

### Stage 1: Notes Organization Parity

Implement all six server domains together: contract validation, stable IDs, suppressions, durable local intent, adapters, materializers, enrollment, legacy migration/adoption, and server-vector conformance.

### Stage 2: Notes Tool Support

Add exact folder/keyword filters and organization metadata to search/get; add additive keyword assurance and coordinated transactional behavior to save; implement pending-organization receipts.

### Stage 3: Agent Lessons Convention

Add folder lifecycle behavior, the feedback/provenance/principle lesson template, capability-aware runtime guidance, transient preview-before-save behavior, untrusted-result labeling, credential checks, search-before-save/update behavior, and primary/subagent permission coverage.

### Stage 4: Human-Reviewed Promotion

In a separate atomic task and ADR, add capability-aware promotion guidance, exact foreground proposal previews, target eligibility, primary/subagent role behavior, and relevant verification. Extend the existing repository mutation seam with an expected full-content digest/expected-absent precondition checked at the write boundary plus a path-safe atomic same-directory replace/create; the approved preview and mutation carry the same target, expectation, and replacement. Revalidate selected binding identity and the effective applicable instruction chain before application. Managed-skill promotion remains proposal-only in Console and is applied manually through the existing Library editor/`LocalSkillsService.update_skill` version/trust path. Do not add a promotion database, dedicated promotion apply tool, background watcher, or automatic instruction mutation.

### Future: Scheduled Improver Research

Create a design/research backlog task rather than an implementation-ready task. It must study authorized feedback sources, domain-expert filtering, privacy and retention, schedule/idempotency, reusable improver templates, domain-specific weighting, golden/reference evaluation, rollback and regression detection, cost and outcome metrics, and crawl-walk-run deployment. Any implementation requires a separate future ADR and explicit owner approval; automatic application remains excluded unless that later decision authorizes it.

Each stage must be independently testable. Chatbook must not advertise the six-domain group until Stage 1 is complete as a group.

## Verification Strategy

Targeted deterministic structural and integration coverage will include:

- migration of active and soft-deleted legacy folders, keywords, collections, and links;
- canonical UUID, hash, Unicode, path-length, `.`, `..`, hierarchy, operation, and payload vectors;
- folder-link suppression and source-provenance behavior;
- non-cascading tombstone and restore behavior;
- interrupted enrollment, restart, adoption review, failed bootstrap/retry, and missing `notes.note` or `chat.conversation` dependencies;
- offline durable-intent retry without lost organization changes;
- pending lesson creation, dispatcher exclusion, local discovery, finalization, cancellation, every finalization/dispatch/acknowledgement crash boundary, race repair, and placement-review restart/resolution;
- two-device synchronization across all six domains, including simultaneous untouched default-folder seeds converging on one canonical folder;
- rename, move, delete, keyword-removal discovery behavior, and user folder/keyword edits racing agent updates;
- exact and case-fold-equivalent conventional folder/keyword collision review, including verifying that `agent_lessons` yields durable placement review and `Agent-Lesson` memberships do not match `keyword="agent-lesson"`;
- exact keyword and folder search filtering plus bounded folder metadata;
- `library_get_note` returning current organization metadata/token across content continuations, with stale organization updates refused;
- additive keywords and optimistic-version conflicts;
- capability- and role-specific trusted suffix construction for custom prompt catalogs, primary agents, subagents, and save-only agents, with note bodies never entering trusted context;
- permission refusal, subagent `foreground_required` refusal, and rejected lesson-save approval producing no note, folder, keyword, pending receipt, or fallback write;
- transaction-time approval-stamp binding for marked, pending-organization, and placement-review lessons, including marker/receipt addition, removal, and transition races between review and execution plus subagent updates to still-unmarked pending lessons;
- exact new-lesson template validation and logical target-hint validation rejecting absolute device paths;
- adversarial lesson content remaining ordinary tool-result data and failing to grant authority;
- credential-like content rejection without sensitive logging;
- acceptance of long hashes, error IDs, and clearly fake/example credentials that do not meet the high-confidence rejection boundary;
- synchronized and permanently local-only profile initialization showing the default folder before the first lesson save;
- eligible and ineligible promotion targets, including read-only proposal versus write application capability;
- repository mutation compare-and-swap refusal for stale content, expected-absent races, changed binding identity/effective instruction chains, and approved-preview/mutation mismatches;
- preservation of pre-existing target edits and no automatic reset after failed application;
- managed-skill proposal-only Console behavior, Library service version refusal, modification remaining inactive until reviewed re-trust, and repository instructions taking effect only in a later run/activation;
- historical applied, rejected, reverted/superseded, failed, and cross-device outcome handling without automatic monitoring.

Scripted/golden behavioral evaluations will observe, without claiming a security invariant:

- unverified, contradictory, or search-unavailable feedback remaining an unsaved draft;
- concise privacy-preserving provenance, honest `Unknown` values, and no invented failed attempts;
- one strong verified signal qualifying for a promotion suggestion without a fixed incident count;
- subagents returning a draft/evidence to the foreground primary instead of saving or applying;
- an end-to-end case where Agent A records a verified resolution with failed attempts and Agent B finds and safely applies it;
- preference for a small principle with rationale and progressive disclosure over brittle rule accumulation; and
- explicit reporting that behavioral effectiveness remains unverified when only domain review is possible.

Behavioral evaluation results are evidence about the selected model and prompt configuration, not authorization, deterministic correctness, or a security guarantee.

Repository policy requires targeted tests for changed functionality by default. A full suite is run only when the user explicitly opts in.

## ADR and Governance

**ADR required:** yes

**Existing ADR path:** `backlog/decisions/102-portable-notes-organization-and-agent-lessons.md`

**Promotion ADR path:** allocate the next genuinely unclaimed number during revised implementation planning.

**Reason:** ADR-102 governs portable organization, Notes ownership, verified saves, and the rule that lesson bodies remain outside instruction authority. Its current accepted text does not yet govern forced approval overriding ordinary Notes allow state, trusted run-role enforcement, subagent refusal, pending/placement classification, or execution-time approval-stamp binding. Before Stage 3 implementation, amend ADR-102 to record that fail-closed boundary and preview non-persistence, explicitly relating [ADR-030: Local Library Agent Tool Boundary](../../../backlog/decisions/030-local-library-agent-tool-boundary.md). Human-reviewed promotion additionally crosses Notes, local-skill trust, repository instruction context, filesystem authority, and approval boundaries, so it requires a distinct ADR extending ADR-102, [ADR-009: Local Skill Trust Boundary](../../../backlog/decisions/009-local-skill-trust-boundary.md), [ADR-032: Local Agent Tool Permission Boundary](../../../backlog/decisions/032-local-agent-tool-permission-boundary.md), and [ADR-069: Console Project-Instruction Local State and Preflight](../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md).

ADR-102 consumes the existing server contract and relates the resulting tool behavior to the existing Notes interoperability decisions. Its Stage 3 amendment must define Agent Lesson classification as exact-marker, `pending-organization`, or `placement-review` ownership; bind existing approval stamps to immutable calls and reviewed classification; revalidate classification atomically at the Notes transaction boundary; reject subagent mutation; and keep rejected previews non-persistent. The promotion ADR must preserve ADR-102's untrusted-note boundary and use existing target authority/trust enforcement rather than claiming lesson text can authorize a change. A new server ADR is not required unless conformance testing reveals a server-side architectural defect.

The existing implementation Backlog tasks and Superpowers plans must link this revision and ADR-102. Promotion receives its own atomic task, ADR, and implementation plan. The scheduled-improver item is research/design only and must require a separate future ADR before implementation. Every new identifier remains provisional until checked against the latest remote branch and open PRs.

## Source-Informed Principles

The feedback-quality and human-reviewed promotion additions adapt the inner-skill/outer-improver lessons described in Anthropic's article [How Warp builds self-improving agents on Claude](https://claude.com/blog/how-warp-builds-self-improving-agents-on-claude) (2026-08-26): preserve detailed feedback, explain why, prefer principles to exhaustive rules, keep stable procedural instructions distinct from mutable memory, use progressive disclosure, propose the smallest focused edit, validate against deterministic evidence where possible, and retain human control over instruction changes. Chatbook deliberately stops short of Warp's scheduled improver in this delivery.

## Acceptance Summary

The design is satisfied when Chatbook can enroll in and faithfully consume the complete server Notes organization group; a permitted agent can search exact Agent Lessons, preview and record one verified structured lesson with additive keyword and folder organization, preserve failed attempts, safe provenance, evidence, principle, and rationale, and another permitted agent can discover it across devices without the lesson gaining instructional authority. A foreground primary agent can turn strong evidence into one exact human-reviewed proposal against an eligible target while existing workspace approval, Git state, and skill trust remain authoritative. User rejection, rename/delete choices, offline work, migration conflicts, interrupted enrollment, stale proposals, and failed/reverted promotion remain lossless and explicit. The scheduled improver is represented only by a future research/design task.
