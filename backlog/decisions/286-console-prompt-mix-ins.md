# ADR-286: Console prompt mix-ins

Status: Accepted 2026-11-01 (user-approved design; implementation status below)
Date: 2026-11-01
Related Task: [TASK-38534](../tasks/task-38534%20-%20Design-prompt-mix-ins-for-Console-sends.md)
Spec: [Console prompt mix-ins](../../Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md)
Plan: [Prompt mix-in implementation](../../Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md)
Implementation: Complete in slices 1–9 and reviewed final integration corrections (TASK-38544 reopened for current-dev integration). Evidence: Docs/superpowers/qa/2026-11-01-prompt-mix-ins/verification.md
Amends: [ADR-040](040-versioned-prompt-artifacts-and-safe-improvement-transactions.md), adding a separately discriminated mix-in artifact; [ADR-057](057-portable-chatbook-prompt-records.md), adding an explicitly versioned portable mix-in record
Preserves: [ADR-097](097-console-reference-backed-semantic-trace-ledger.md), [ADR-098](098-visible-bounded-console-prompt-queue.md), [ADR-095](095-conversation-owned-console-generation-settings.md), [ADR-210](210-console-region-ownership.md)

## Context

Users want reusable modifiers such as style guidance, output structure and scene
visualization requests that can be combined with an ordinary prompt and toggled
before each send. These modifiers must remain distinct from Prompt insertion,
System replacement, Recipes, Skills and executable workflows.

The library already owns Prompt/Recipe artifacts, source capabilities, versions,
search and portability. Console owns per-chat state, immutable send inputs,
request transforms, queues and provider execution. A second catalog or independent
send path would duplicate those owners. Mutating the System prompt would also
make per-message modifiers persistent and disrupt stable request prefixes.

Current-send-only instructions introduce an unavoidable cache trade-off: the next
request's history omits the previous injection, so a cache entry covering that
injection may not match. Existing Anthropic caching marks the final message as
well as tools/system; it needs a clean-history boundary to preserve eligible
older-history reuse. A stable routing key alone cannot change prefix identity.

Retry has a different requirement from regeneration. Library edits should affect
future work, while Retry must reproduce the original mix-in instructions after
edits, deletion or restart. Optional sanitized diagnostic traces cannot guarantee
exact operational retry content, especially when capture is off.

## Decision

1. **Add a dedicated Mix-in type to the existing prompt library.** Reuse library
   CRUD, source authority, search, versions, collections and portability. Each
   record has one literal instruction body and Before/After placement, plus normal
   library metadata. A strict kind/version definition is canonical. Compiled
   compatibility text is never a second editable authority or an execution escape
   hatch. Existing server-v1 and Console-v2 Prompt/Recipe definitions retain their
   contracts. No parameter forms, automatic activation or executable actions are
   introduced.

2. **Extend compatibility boundaries explicitly.** Brief/detail/search/write,
   capability negotiation, retained versions, import/export and normal execution
   pickers must recognize the new type. Unsupported sources refuse mix-in
   operations instead of flattening records to Prompt or switching authorities.
   A new portable record version carries mix-ins while existing record readers
   retain their prior meaning. Imported records are not auto-enabled. Source
   references are authority-qualified stable identities, not ambient numeric IDs.

3. **Keep chat selection separate from artifact content.** Saved chats retain
   selected references, enabled flags, group order and last-used revision facts.
   Temporary chats follow their existing lifetime and promotion path. New chats,
   including copy/fork destinations, start with none selected; inherited historical
   receipts never activate selections. Reopening restores that chat's references
   and uses current definitions for its next new send. This preference is local
   chat state, not a new global/workspace/persona default or sync protocol.

4. **Use the existing composer Prompts button and Prompt Workbench.** Add a Mix-ins
   section and enabled count on the existing entry point, not a separate manager
   or persistent button. Users add library entries, toggle explicit checkboxes,
   inspect/edit through the library owner, reorder within Before/After groups and
   remove chat selections without deleting library records. Disabled entries keep
   their positions. Placement edits move an entry to the end of its new group;
   library browsing order never determines injection order. Preserve design-token,
   keyboard, focus, exact-target and narrow-layout contracts.

5. **Freeze immutable mix-in inputs during admission.** Capture the originating
   chat's selection before asynchronous preparation, resolve enabled source
   definitions and revisions, and validate a coherent snapshot before acceptance.
   Queue entries capture at enqueue, not drain. Later edits and toggles affect
   future captures, not accepted work. The existing queue stays bounded and
   process-memory-only. Tool authority remains independently checked at dispatch.
   Machine follow-ups and automatic new turns do not newly resolve human-selected
   modifiers in v1.

6. **Compose only the current user-message envelope.** Existing command, skill,
   dictionary, world-info and retrieval processing occurs before literal mix-in
   composition. Apply enabled Before bodies in visible order, the current
   prepared user message, then enabled After bodies in visible order. String
   components are separated by two newlines without trimming body text; multipart
   content preserves original part order and nontext content with added text
   blocks around it. Apply once per attempt and reuse through tool continuations.
   Account for bodies before final budgeting, capture binding and dispatch.

7. **Preserve current-send-only history semantics.** The composer and stored user
   message remain original. Used mix-in bodies and receipts do not become future
   history messages, compaction summaries or retrieval queries. Turning off stops
   explicit inclusion for future new sends. Prior assistant output remains
   ordinary context and can still influence style. Mix-ins grant no tool/file/
   rendering authority and cannot make an empty draft sendable.

8. **Separate Retry from regeneration.** Retry uses the exact recorded body,
   placement and order of the chosen original attempt, including a recorded-empty
   set. Label nonempty cases `Retry with original mix-ins`. Regeneration,
   edit-and-resend and explicit new-turn Continue capture current choices and
   current definitions. Missing/corrupt new-format retry records refuse exact
   Retry and offer explicit regeneration; they never substitute current content.
   Predating legacy attempts have no mix-ins.

9. **Persist operational snapshots independently of diagnostic capture.** Use the
   existing conversation database and persistence owner for immutable used
   definitions and ordered attempt references. Reuse each distinct used revision
   within the conversation; do not retain whole growing provider requests.
   Retain exact bodies in content storage, not generic metadata or logs. Durable
   attempts commit references before dispatch, with draft-preserving refusal on
   failure. Temporary attempts keep memory snapshots; promotion includes them.
   Retained bodies survive library edits/deletion, remain available while a
   conversation/fork owns a receipt, and follow existing backup/restore and
   conversation deletion lifetimes. Trace filtering/purge does not purport to
   erase operational retry data. Trace projection/export retains its existing
   credential and PII protections.

10. **Preserve caching without changing instruction semantics.** Before/After is
    relative to the current message, not the entire request. Keep system/tool
    content unchanged and separators/order deterministic. Where supported, cache
    the clean-history boundary before the volatile mix-in/user tail. Preserve
    existing cache switches, TTLs, degrade behavior, endpoint/model capabilities
    and breakpoint budgets. Do not promise full previous-request reuse or force
    writes merely to improve a metric. Older stable history can remain eligible;
    the newest exchange may need processing again. No cache behavior can justify
    replaying disabled historical instructions.

11. **Fail visibly instead of changing intent.** Missing enabled definitions,
    invalid/oversized content, stale source/session targets, unsupported source
    capabilities and snapshot persistence failure block preparation while
    preserving the draft. Unavailable disabled entries remain visible without
    blocking. Selection-save failure is visibly unsaved with exact-revision retry.
    Context limits count mix-ins; never silently truncate or omit them. Existing
    output handling remains authoritative; there is no automatic repair call,
    renderer, image generator or follow-up workflow.

## Alternatives considered

| Alternative | Reason rejected |
| --- | --- |
| Mark ordinary Prompts with tags | Tags become behavioral contracts and ordinary insertion can accidentally execute modifiers as full prompts. |
| Separate catalog/database/manager | Duplicates library identity, CRUD, search, versions, backup and portability. |
| Editable per-chat copies or pinned versions requiring Update | Adds a second editing/update workflow; the approved behavior is library edits affecting future sends. |
| Paired Before/After sections per mix-in | More authoring structure than needed; one body and one placement cover v1. |
| Variable forms and automatic activation | Adds parsing and interaction semantics, including ambiguity with literal HTML/CSS/SVG, beyond approved scope. |
| Inject into System or retained history | Violates message-scoped off behavior and can disturb a larger cache prefix. |
| Resolve queued modifiers only at dispatch | Edits or toggles change a message after the user queued it. |
| Retry against current library/toggles | Cannot repeat the original attempt and hides the difference from regeneration. |
| Recover exact Retry from diagnostic trace alone | Capture can be disabled, sanitized, purged or incomplete; operational behavior cannot depend on it. |
| Duplicate complete provider requests per attempt | Reintroduces the growing-history storage problem addressed by ADR-097. |
| Maximize cache hits by retaining old instructions | Makes off unreliable; correctness takes precedence over a cache metric. |
| Mix-in-specific rendering or automatic generation | The approved first version supplies instructions and uses existing response/tool behavior. |

## Consequences

Users get reusable, independently toggled response guidance without modifying
their typed messages or System prompt. One library remains authoritative; one
Console runtime controls dispatch. Original retry instructions remain inspectable
after library changes and without diagnostic capture.

The change reaches storage and artifact/source contracts, not only UI. Required
migrations must increment the actual schema versions at implementation time.
Older server/client formats may not support mix-ins and must report that boundary.
Conversation preferences and exact operational snapshots need explicit backup,
deletion, fork and privacy coverage.

Current-send-only injection reduces some full-tail cache reuse even with an
unchanged selection. Eligible clean history can still be reused; actual savings
require provider/model-specific measurements. Models can ignore instructions or
produce output the existing surface cannot render.

## Verification and rollout

The linked spec defines targeted scenarios for artifact round trips and source
refusal; literal composition and multipart preservation; chat isolation and
ordering; edit/queue/reopen behavior; exact Retry versus current regeneration;
capture-off persistence and privacy; backup/fork/purge; and cache-prefix and
breakpoint behavior. Live checks must exercise the real composer Prompts control,
narrow layouts and keyboard operations in an isolated profile. Cache-hit claims
require real usage evidence, not only payload equality or mocks.

At design acceptance this ADR accompanied no product code or migrations. Slices
1–9 now implement it with targeted native and historical live qualification; final
integration corrections and scoped re-review are complete under TASK-38544.


### Operational replacement proof (Task 38540 review refinement)

The unreleased v82 attempt record additionally carries `replaces_attempt_id`,
`result_revision_id`, and `trace_call_id`. The first is an explicit accepted
Retry/Regenerate/accepted-original-Resend relationship, separate from exact-Retry lineage. The latter two
are body-free, set-once identities: the exact complete replacement result's
semantic revision and the originating trace call bound through the exact native
dispatch checkpoint at first-call reservation. Ordinary initial attempts cannot
use this replacement exception. A bounded, cycle-free predecessor chain must
reach the exact traced origin, and the selected latest revision must equal the
sealed result. Subsequent user edits cannot inherit result authority from retained
metadata. Existing trace origin/terminal/policy/range/descriptor checks remain.

Result sealing shares the existing complete-generation settlement transaction;
rollback includes the seal. Capture Off retains the operational result identity
without creating diagnostic bodies. Fork copies retry snapshots but clears all
three proof fields. Temporary promotion preserves accepted replacement lineage
but does not invent a durable result or trace seal for process-local historical
attempts; unavailable/unsealed evidence never authorizes trace restoration. No new
table or provider-body copy is introduced. The exact recovery catalog tracks the
unreleased v82 table/guard amendments; no released v82 compatibility is claimed.

### Historical fork receipt handoff (TASK-38544 final review)

The existing immutable fork snapshot carries a conversation-owned historical
receipt projection, including exact markers/classification and retained Retry
predecessors. Historical predecessors outside the copied prefix have nullable
message anchors; they never fabricate a message or grant trace/replacement proof.
Fresh attempt admission retains its strict two-live-anchor invariant. The existing
caller-cursor copier shares its historical read/write seam with live publication
and the narrowly scoped promotion contribution. Forks clear all three replacement
proof fields, preserve unavailable/partial/future marker classification, and start
with empty active selections. Temporary forks own this projection independently
of source closure; durable publication agrees with reopening without render-time
DB reads. Source fences and parent transaction rollback remain authoritative.

## Current-dev integration allocation

The focused publication port targets dev `2acbcd42f3e7b07fd122f5f6dc6b22e7385f10c5`. Its existing ChaChaNotes schemas 82–85 retain their historical meaning; prompt mix-in tables install at schema 86, after 85. Prompts advances from 5 to 6. The former experimental feature schema 82 is not a released predecessor and is not accepted as a historical recovery catalog. Prior captures and hashes remain historical to the reviewed mixed-ancestry commits; current source and qualification manifests describe the port separately. Historical Git objects can be unavailable in fresh clones and are reported unavailable, never treated as verified. Current published source checks remain strict.

### Accepted usage facts (PR3346 review)

Source capture and Preview only build provisional immutable snapshots. Accepted manual input publishes last-used revision at native acceptance (or ephemeral acceptance); queue input publishes these facts when custody admission succeeds. Current-choice Regenerate, Edit and Continue publish after their operational receipt is accepted. Original Retry and queue drain never refresh current-choice facts. Publication reads no source, skips unavailable preferences and never re-adds removed selections. Current-capture witnesses authorize accepted placement/order observations only while their captured preference revision remains current; a changed layout advances that revision and rejects stale Workbench intents. Workbench project/save retain name presentation and explicit layout CAS. Used-version-only facts do not advance the intent revision. Current received input commits its draft before publishing accepted layout; accepted execution no longer applies pre-acceptance input-currentness checks. Frozen Retry, queued drain and externally supplied configurations have no current-layout observation authority; queue current observations publish once at admission. A newer user intent wins over delayed layout publication. A placement already used by the accepted revision is not labeled updated since last send. Refused hooks, provider resolution and preparation preserve all choice facts.
