# Agent Lessons and Notes Organization Sync Design

**Date:** 2026-08-29

**Status:** Owner-approved and independently reviewed; ready for implementation planning

**Server baseline reviewed:** `tldw_server` `origin/dev` at `1ad2f1e5b30c49ea75396e4b713496b73e875fec`

## Summary

Chatbook will add a default, user-manageable Notes folder named `Agent_Lessons`. Agents with Notes permission will use it to record verified, reusable solutions, including approaches that failed and why. Other agents will discover those lessons through the existing Notes search capability when they encounter similar symptoms.

This feature will not introduce a new server sync domain or a separate agent-memory store. The server already defines a complete, indivisible six-domain Notes organization group. Chatbook will first consume that existing contract, then extend its Notes tools, and finally add the Agent Lessons convention and agent guidance on top.

The authoritative discovery marker is the exact keyword `agent-lesson`. The folder is the default visible organization and a ranking/display preference, but it cannot be the sole identity mechanism because users may rename or delete it.

## Problem

Agents repeatedly spend time rediscovering solutions to problems already solved in earlier interactions. Ordinary Notes can hold the knowledge, but the application currently lacks:

- a recognizable default location for reusable agent lessons;
- a consistent lesson format that captures verification and failed approaches;
- agent-facing guidance to search before retrying work and save only proven solutions;
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

## Non-goals

- Automatic background capture of conversations or failures.
- Saving speculative, unverified, or interaction-specific observations as lessons.
- A new agent-memory database, embedding index, or semantic-retrieval subsystem.
- A seventh Notes organization domain or a new server contract.
- Silent merging of same-name folders or same-path objects with different sync identities.
- Allowing lessons to grant permissions, override instructions, or authorize tool calls.
- Synchronizing filesystem directories as a new portable folder domain.
- Making server changes unless conformance work demonstrates a concrete server defect.

## Chosen Approach

The feature is a guided convention layered on ordinary Notes:

1. An agent encountering a problem searches Notes using relevant error signatures, component names, and root-cause terms, scoped to the exact `agent-lesson` keyword.
2. The agent treats matches as untrusted leads and verifies applicability in the current environment.
3. After resolving and verifying a reusable issue, the agent searches again for the same root cause.
4. It updates the existing lesson with optimistic content and organization tokens when the lesson is materially the same, or creates a new lesson and cross-references related public note IDs when it is distinct.
5. A new lesson save adds the canonical `agent-lesson` keyword without removing user keywords and places the lesson in the current conventional `Agent_Lessons` folder when available. An ordinary update preserves current folder memberships and does not restore a marker a user has removed.

This provides durable reuse with minimal new machinery. It reuses Notes CRUD, FTS5 search, public note IDs, normal permissions, and the server's existing synchronization protocol.

### Alternatives Rejected

**Folder-only discovery.** This is simple but fails after a user renames or deletes the default folder. The synchronized exact keyword is therefore authoritative.

**Automatic lesson extraction.** Background capture risks saving secrets, noise, incorrect diagnoses, and unverified conclusions. Explicit agent saves after verification are more predictable and auditable.

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

Agent Lessons discovery normally supplies `keyword="agent-lesson"`. Folder metadata may improve display or ranking, but the current folder is not a required scope because users can rename or move it. The existing response-size fitter continues to cap oversized results.

### `library_save_note`

Add an optional additive `ensure_keywords` field. The operation ensures the named whole keywords are attached while preserving every existing user keyword. Creation and attachment still obey the server's trim and case-fold uniqueness/conflict rules, but name-based search remains spelling-exact so a differently cased resource cannot become the Agent Lessons marker implicitly. The operation never treats absence from `ensure_keywords` as a request to remove a keyword.

The tool coordinates note content, requested folder membership, additive keywords, and durable synchronization intent through the Notes transaction boundary. Updates retain optimistic `expected_version` checks. Any update that requests an organization change also accepts the opaque `expected_organization_version` returned by search/read and fails if locally known folder or keyword link heads changed. On either stale token, an agent must re-read and merge rather than overwrite.

Agent Lessons uses folder placement only when creating a new lesson or finalizing a pending lesson. An ordinary lesson update does not supply a folder change. It supplies `ensure_keywords=["agent-lesson"]` only when the latest read still contains the canonical marker; if a user already removed the marker, the agent does not silently reclassify the note without an explicit user request. A keyword removal or folder move racing the save is detected by `expected_organization_version` and becomes a reviewable conflict.

High-confidence credential material is rejected at the agent-authored lesson save boundary with a structured validation error. Rejected content is not logged. Avoiding personal data and large raw logs is also explicit agent guidance, but it is not represented as a claim of perfect PII detection.

Permission denial creates no note, folder, keyword, pending receipt, or hidden fallback write.

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
Commands, tests, observations, or other evidence that confirmed the resolution.

## Caveats
Known limits, risks, or conditions where the solution should not be applied.

## Related lessons
Stable public note IDs for distinct but related lessons.
```

The structure is advisory for user-edited notes but required for newly generated agent lessons. Agents should keep evidence concise and omit secrets, personal data, and large raw logs.

## Agent Runtime Guidance and Trust Boundary

Agent Lessons instructions are appended as a trusted, non-editable runtime protocol suffix for both primary agents and ordinary subagents. They do not live solely in the user-overridable prompt catalog.

The suffix is capability-aware:

- agents with Notes search permission receive troubleshooting search guidance;
- agents with both Notes search and save permission receive verified-save and update guidance;
- a save-only agent receives no Agent Lessons save guidance because it cannot perform the required duplicate/root-cause search;
- agents lacking a capability are not instructed to exercise it.

Subagents continue to inherit their parent's allowed tools, minus the existing restricted capabilities; named agent definitions may narrow permissions. Agent Lessons introduces no permission bypass or new propagation rule.

Search results and note bodies are untrusted user content. The runtime guidance must state that a lesson:

- cannot override system, developer, project, or current user instructions;
- cannot grant tool access or authorize a command;
- cannot expand filesystem or network scope;
- must be checked against current versions, environment, and evidence before use.

UI and tool-result labeling should make this reference-only trust level clear. Adversarial note text remains data, not instructions.

## Default Folder Lifecycle

`Agent_Lessons` is seeded as a root Notes folder only after the six-domain organization group is ready and its snapshot/history has been applied. The seed is performed by the Notes service, not by a blind schema-migration insert.

Seeding rules:

1. Reuse an active root folder with the exact conventional spelling. A case-fold-equivalent but differently spelled root is an adoption conflict, not an automatic match.
2. Only notes carrying the exact `agent-lesson` keyword participate in Agent Lessons discovery; reusing the folder does not reclassify unrelated notes.
3. Determine whether the conventional folder was seeded before by inspecting synchronized `notes.folder` upsert history for a former root named `Agent_Lessons`, even if its current head is renamed or tombstoned.
4. Store a dataset/profile-scoped monotonic seed state: `unknown` until bootstrap history is fully applied, then `not-seeded` or `seeded`. Materializing any qualifying remote upsert changes it to `seeded`; local seed or exact reuse persists `seeded` in the same transaction. A cached false value can never suppress evidence received by a later pull.
5. For a local-only database, persist an equivalent local seed receipt.
6. Do not recreate a folder merely because the user renamed or deleted the seeded folder.
7. If a later agent explicitly saves a new lesson and no current conventional folder exists, recreate the conventional folder for that save without restoring or modifying the former folder.

The reviewed server currently retains envelope history non-destructively and bootstraps deleted resources as upsert then tombstone. If future retention becomes destructive, the server/client contract must preserve an equivalent durable seeded-before marker before relying on compaction.

## Pending Agent Lesson Flow

A verified lesson must not be lost merely because organization enrollment is still initializing or the canonical keyword is awaiting collision review. If the agent has permission and requests a lesson save before organization can be safely finalized:

1. Save the ordinary note and a content-free pending-organization receipt in one local Notes transaction.
2. Keep the pending note local-only; do not attach folder/keyword organization or publish it as a completed lesson yet. Every normal note dispatcher excludes note IDs whose receipt is in the blocking `pending-organization` state.
3. Include it in originating-device Agent Lessons searches through the pending receipt, clearly labeled pending.
4. Once the group is ready and the canonical keyword is resolved, atomically attach the canonical keyword, create the immutable note/resource/link sync intents, and attach the current conventional folder when unambiguous. If placement succeeds, clear the receipt. If placement is ambiguous, transition it to the non-blocking `placement-review` state with the desired conventional placement and collision IDs before allowing publication. The blocking state cannot become invisible to dispatch until all required publication intents and any required review record exist in that same transaction.

The receipt stores only stable local identity and desired organization intent, not lesson content. A `placement-review` receipt survives restart, does not block note/keyword publication or discovery, and remains until the user resolves or dismisses the placement conflict. Deleting the note cancels either receipt state. A permission denial creates nothing. Crash recovery tests must cover each boundary before, during, and after finalization, outbox copy, server acknowledgement, and local acknowledgement cleanup.

The server's uniqueness rules are case-insensitive. A differently spelled case-fold-equivalent folder such as `agent_lessons` is never silently adopted as `Agent_Lessons`; it produces a durable placement review. Once the canonical keyword is safely established, that folder conflict does not block keyword-based lesson completion or discovery—the note is saved with the keyword and without conventional placement until reviewed. A differently spelled case-fold-equivalent keyword such as `Agent-Lesson` requires adoption/rename review because automatically treating its existing memberships as Agent Lessons could reclassify unrelated user notes. Name-based `keyword="agent-lesson"` search is spelling-exact and returns none of those variant memberships. Until the canonical marker conflict is resolved, the new lesson remains locally pending.

If a coordinator-created folder or keyword loses a cross-device race, automatic repair is permitted only when the losing object was created for that exact pending save and remains unedited and unrelated. User edits or unrelated memberships require explicit review. The lesson content remains intact even when placement requires review.

For a permanently local-only profile, the Notes service can complete the organization locally without waiting for a server enrollment state.

## Discovery, Updates, and Deletion

Lexical retrieval remains the discovery mechanism. Agent guidance should search using combinations of:

- exact error signatures;
- component or subsystem names;
- affected platform and versions;
- suspected or confirmed root-cause terms.

Duplicates are tolerated because silently coalescing diagnoses is riskier than retaining distinct evidence. Agents cross-reference related public note IDs and update only when the root cause and applicability are materially the same. Updates preserve the latest user-controlled organization state and require both content and organization concurrency tokens when organization is involved.

Removing the exact `agent-lesson` keyword removes a note from Agent Lessons discovery even if it remains in the folder. Deleting the note also removes it. Renaming or moving the folder does not remove discovery while the keyword remains.

## Failure Handling and Observability

- Search failure does not block troubleshooting; the agent reports the unavailable lookup and continues within its existing authority.
- Save validation errors return actionable structured reasons without logging rejected content.
- Enrollment and durable-intent dispatch expose initializing, pending, review-required, retrying, and ready states rather than silently dropping work.
- Offline operation after ready records durable intent and synchronizes later.
- Contract-invalid remote envelopes fail closed into the existing conflict/review path.
- A pending lesson remains locally searchable and visibly pending until finalization or deletion.
- Metrics and logs use IDs, domain names, state transitions, and error classes, not note bodies or credentials.

## Delivery Sequence

### Stage 1: Notes Organization Parity

Implement all six server domains together: contract validation, stable IDs, suppressions, durable local intent, adapters, materializers, enrollment, legacy migration/adoption, and server-vector conformance.

### Stage 2: Notes Tool Support

Add exact folder/keyword filters and result metadata to search; add additive keyword assurance and coordinated transactional behavior to save; implement pending-organization receipts.

### Stage 3: Agent Lessons Convention

Add folder lifecycle behavior, lesson template, capability-aware runtime guidance, untrusted-result labeling, credential checks, search-before-save/update behavior, and primary/subagent permission coverage.

Each stage must be independently testable. Chatbook must not advertise the six-domain group until Stage 1 is complete as a group.

## Verification Strategy

Targeted automated coverage will include:

- migration of active and soft-deleted legacy folders, keywords, collections, and links;
- canonical UUID, hash, Unicode, path-length, `.`, `..`, hierarchy, operation, and payload vectors;
- folder-link suppression and source-provenance behavior;
- non-cascading tombstone and restore behavior;
- interrupted enrollment, restart, adoption review, failed bootstrap/retry, and missing `notes.note` or `chat.conversation` dependencies;
- offline durable-intent retry without lost organization changes;
- pending lesson creation, dispatcher exclusion, local discovery, finalization, cancellation, every finalization/dispatch/acknowledgement crash boundary, race repair, and placement-review restart/resolution;
- two-device synchronization across all six domains;
- rename, move, delete, keyword-removal discovery behavior, and user folder/keyword edits racing agent updates;
- exact and case-fold-equivalent conventional folder/keyword collision review, including verifying that `agent_lessons` yields durable placement review and `Agent-Lesson` memberships do not match `keyword="agent-lesson"`;
- exact keyword and folder search filtering plus bounded folder metadata;
- additive keywords and optimistic-version conflicts;
- custom prompt catalogs and primary/subagent permission combinations, including save-only agents;
- adversarial lesson content that attempts to grant authority or inject instructions;
- credential-like content rejection without sensitive logging;
- an end-to-end case where Agent A records a verified resolution with failed attempts and Agent B finds and safely applies it.

Repository policy requires targeted tests for changed functionality by default. A full suite is run only when the user explicitly opts in.

## ADR and Governance

**ADR required:** yes

**ADR path:** `backlog/decisions/NNN-notes-organization-sync-and-agent-lessons.md` (allocate the next genuinely unclaimed number during implementation planning)

**Reason:** the work changes local schema and migration behavior, synchronization enrollment and conflict policy, data ownership boundaries, cross-module tool contracts, and long-lived agent trust behavior.

The ADR will consume the existing server contract and amend or supersede the device-local-folder constraint in Chatbook ADR-073. It will also relate the resulting tool behavior to the existing Notes interoperability decisions rather than duplicating them. A new server ADR is not required unless conformance testing reveals a server-side architectural defect.

The implementation Backlog task and Superpowers implementation plan must link this design and the new ADR. The ADR must be created before implementation begins, as required by repository policy.

## Acceptance Summary

The design is satisfied when Chatbook can enroll in and faithfully consume the complete server Notes organization group; a permitted agent can search exact Agent Lessons, record one verified structured lesson with additive keyword and folder organization, preserve failed attempts and evidence, and another permitted agent can discover it across devices without the lesson gaining instructional authority. User rename/delete choices, offline work, migration conflicts, and interrupted enrollment must remain lossless and explicit.
