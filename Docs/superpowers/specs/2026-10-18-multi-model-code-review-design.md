# Primary-agent multi-model code review — design

- Date: 2026-10-18
- Status: Implemented contracts; qualification and final review pending under TASK-35582 (2026-10-26)
- Implementation task: [TASK-35582](../../../backlog/tasks/task-35582%20-%20Implement-primary-agent-multi-model-code-review-recipe.md)
- Detailed implementation plan: [Multi-model code review implementation](../plans/2026-10-18-multi-model-code-review-implementation.md) (implementation in progress; evidence tracked per slice)
- ADR required: yes
- ADR path: [backlog/decisions/281-primary-agent-multi-model-code-review.md](../../../backlog/decisions/281-primary-agent-multi-model-code-review.md)
- Reason: Defines a reusable agent entry point, preset references, review-source and report ownership, role capabilities, and a cross-run verification contract.

## Purpose and approved choices

Let a user say “do multi-model code review” for a PR, commit range, patch, or
working-tree change. The primary agent coordinates independent reviewers using
the user's chosen APIs/models, presents each report, and offers consolidation
with evidence-based verification of every finding.

The approved defaults are:

- Every reviewer receives the same general review brief. An optional specialist
  focus supplements it; it does not remove the general review obligation.
- One saved default team references existing agent presets. Per-run overrides do
  not change saved preferences.
- Individual reports appear before a unified report is generated.
- Verification is a separate, optional stage using a fresh subagent and a saved
  verifier provider/model preset.
- The primary coordinates the stages through existing fleet execution. The
  feature does not introduce a general workflow executor or nested delegation.

## Current code and governing boundaries

The original design was checked against checkout `b07fc04252`. The implemented
owners below are qualified by [recorded evidence](../qa/2026-10-18-multi-model-code-review/README.md);
live-provider acceptance remains unverified. The governing boundaries are:

| Owner | Existing behavior to reuse |
| --- | --- |
| `Agents/agent_models.py::AgentDefinition` | Preset instructions, provider/model, sampling parameters, tool narrowing, and configured fallbacks |
| `Agents/agent_routing.py` and ADR-147 | Preset routing, gated ad-hoc overrides, readiness, and child-owned parameters |
| `Agents/agent_service.py` and `fleet_coordinator.py` | Spawn/wait/cancel, budgets, run lineage, settlement, and result capture |
| `Agents/tool_catalog.py` | Provider registration and discovery; combined fleet results can be shortened |
| `Chat/console_agent_bridge.py` | Per-run composition; model-invoked skill tools currently execute as child runs |
| `Chat/console_scratch_space.py` | Per-tab scratch ownership and leases; scratch is temporary |
| `Agents/local_tool_provider.py`, `Tools/git_tool_impls.py` | Admitted source reads, path validation, exclusions, and read-only Git operations |
| `Widgets/settings_agents_panel.py` | Canonical Settings Agents panel and existing preset authoring |
| `Agents/canvas_tool_provider.py`, `Canvas/guide.py`, ADR-149 | Discoverable, bounded, packaged guidance loaded on demand |

Children have zero further-subagent budget. An orchestration skill invoked as an
ordinary model tool therefore cannot coordinate the review team. The recipe must
load into the primary run.

Console currently fixes `RunBudget.max_subagents` at two total children per primary
turn; `max_live_subagents` is a separate concurrency limit. Reusing fleet execution
does not automatically admit a third reviewer or a verifier in that same turn.
The bounded continuation policy below extends the accepted-work integration in
ADR-134/135 through the existing wake owner and AgentRuns schema 25 amendment below.
With `max_live_subagents=1`, children run synchronously inside spawn and no fleet
wait tool is offered. Preserve that kill switch. Source-based verification remains
available; executable-check feedback is unavailable for an inline verifier.

`AgentConfig.allowed_tools` governs catalog tools, not every runtime capability.
Role isolation must close the relevant runtime gates as well as narrow catalog
tools. A prompt saying “read only” is not an authority boundary.

Existing workspace binding admission, excluded paths, project-instruction
activation, provider-destination consent, permission floors, and cancellation
continue to apply. A recipe or preset grants no new authority.

## User flow

1. The user requests multi-model review and supplies or identifies a target.
2. The primary loads the review guide, resolves the default team plus explicit
   overrides, and resolves any genuinely ambiguous target.
3. Preparation validates destinations and captures a versioned review source
   view. The primary briefly states the target, reviewer identities, and focuses.
   An explicit review request already authorizes this stage; no redundant
   confirmation is added beyond existing permission or destination-consent gates.
4. The primary launches independent reviewer children and waits for their
   terminal results. Host-admitted primary continuations handle any remaining
   slots under the budget policy below; the per-turn child ceiling stays intact.
5. The primary presents a compact comparison and separately readable individual
   reports, including incomplete or failed participants.
6. The primary offers consolidation and verification. Silence is not acceptance.
   A request that already explicitly includes this second stage is sufficient
   authorization, but the individual reports still appear first.
7. After acceptance, the primary loads verification guidance and launches a fresh
   verifier. It reads provenance-free claim batches from the host ledger and
   evidence from the captured source view.
8. The primary presents a new unified report and a complete disposition mapping.

Examples: “Review this patch with my default team”; “Use reviewers A and C and
give C a performance focus”; “Consolidate those reports and verify every finding
with my verifier preset.” Only an explicit save request updates team preferences.

## Recipe and host responsibilities

### Guide and entry points

Expose primary-only `code_review_guide(stage)` through the existing tool-provider
seam. Its closed stages are `review` and `verify`; unknown stages and extra arguments are
rejected. It returns a versioned, packaged guide and the applicable default-team
metadata from admitted configuration, bounded to 12 KiB UTF-8 including its JSON
envelope. It performs no model call, spawn, source capture, or arbitrary file read.

Tool discovery briefly describes the natural-language use case; detailed
instructions load only when needed. The `[tools] code_review_enabled` gate
defaults to true because guide retrieval itself starts no paid work. Ephemeral
and policy restrictions can still remove the tool.

Guide metadata contains preset references, display names, configured identities,
focuses, and missing/disabled-reference diagnostics. Endpoint URLs, credentials,
and unrequested private conversation context are not included. Spawn-time
readiness and routing remain authoritative; a guide result is not a grant or a
promise that a later spawn will succeed.

Validate configuration and tool inputs through the existing input/path validation
boundaries. Non-model guide projections retain bounded stage/version/status/size
metadata rather than guide bodies or user-authored focus text.

Provide an optional importable inline `$multi-model-review` skill pointing to the
same guide. Like the existing Canvas example, it uses `context: inline`,
`user_invocable: true`, and `disable_model_invocation: true`, and follows normal
skill trust checks. Natural-language use works without installing that skill.

### Execution and support

The primary owns target clarification, task composition, spawn/wait decisions,
presentation, the second-stage offer, and final assembly. Existing `AgentService`
and fleet ownership continue to own execution, budgets, hooks, cancellation, and
settlement.

Review-specific support belongs to the existing Console/agent service boundary.
The following contracts are part of this design; implementation planning maps
them onto provider registration and the existing dispatch/capture hooks.

| Interface | Contract |
| --- | --- |
| `code_review_prepare(target, reviewers?, verifier?)` | Primary-only. Validate team, source authority, storage and child/continuation admission; seal the source manifest and participant snapshots; return an opaque review ID, context-control revision, slot references and a small summary. Explicit run choices may include provider/model destination overrides admitted through ADR-147; saved choices remain preset references/focuses. No model or child launches. |
| `spawn_subagent(review_id, review_slot_id)` | A review-scoped alternative to generic `task` spawning. The host resolves the slot's captured preset/target, assembles the common task plus focus, and binds its source view, role and result sink. Generic `task`, `agent`, `provider`, `model` and `isolation` arguments cannot be mixed into this form. Ordinary spawning stays unchanged. |
| `code_review_read(review_id, view, document_ref?, cursor?, source_query?, source_range?)` | Bounded, role-checked access to `summary`, `source`, `report`, `finding_index`, `verification_batch` or verifier-only `checkpoint`. The checkpoint exposes current submission revision, batch cursor, pending delivered units, complete-unit disposition status and pending check references without original provenance; it accepts no document/cursor/source selection. For `source`, a strict list/literal-search/relative-path-resolution query may replace a document reference; a document read may instead select a strict 1-based `source_range`, confined to captured bytes. The host derives caller scope. Reviewers get captured source only; the verifier gets source, blinded batches and its own proposal-specific bounded result wait. Only the primary gets original provenance and proposal/claim details through `summary`. |
| `code_review_submit(review_id, kind, records, expected_revision, submission_id)` | Run-bound structured submission of reviewer finding chunks/completion, verifier check proposals/dispositions/completion, or primary-only `check_actions` (claim/decline). `expected_revision` is the bound participant attempt's revision, or the primary's separate check-action stream revision. Another actor or host raw capture cannot invalidate it. Validate schemas, references, role, live authority and quota before committing. Rejected records must be corrected with the same submission ID; a different ID leaves the rejection outstanding. Completion refuses outstanding rejections from other IDs, and the same completion ID can be retried after repair. Bounded field/type feedback reaches the model without raw invalid input values. Model-supplied identities or completion claims confer no authority; claiming never executes a command or bypasses permission. |
| `code_review_release(review_id, expected_revision)` | Primary-only release after explicit user discard/keep/export intent. Check the separate context-control revision; refuse while execution or read leases are active. Tombstone before cleanup; repeated release cannot recreate a context. |

These tools share the single `code_review_enabled` gate and normal progressive
discovery. Context IDs are lookup references, not transferable bearer authority.
Every operation also checks the owning session, conversation/source path,
workspace authority and current lifecycle generation. Revocation, branch/context
change, close or cancellation invalidates pending execution/read grants.

A repeated accepted slot launch returns its existing receipt without dispatching
again. A proven pre-dispatch refusal leaves the slot unlaunched; accepted failures
or uncertain execution do not retry automatically. Submission IDs are scoped to
the bound slot attempt (or primary-run/review/verifier-attempt check-action stream)
and make identical replays idempotent; conflicting replays
or stale submission revisions refuse without changing accepted records. Readback
and acknowledgments distinguish the caller's submission revision from context-control
revision and publication cursors. Existing child launch, provider and permission owners still
perform the actual execution and admission.

The host owns a versioned `ReviewContext` record containing the review ID/context-control revision,
owning session and source-path epoch, binding/source manifest, guide version,
participant definition/target snapshots, child-run receipts, trusted stage grants,
per-attempt submission revisions, primary check-action streams, proposal/claim/result
state, actual verifier feedback mode and publication sequence, quota reservations,
original-finding index, private verification-ID map, batch
cursor and disposition ledger. It stores no credentials. Its states are
`prepared`, `reviewing`, `awaiting_verification`, `verifying`, `complete`,
`partial`, `blocked`, `cancelled` and `released`. A successful model reply alone
cannot set `complete`: terminal run receipts and ledger validation are required.

This is bounded feature state, not another execution engine. The primary chooses
the next permitted operation; the host validates the record and delegates accepted
work to the existing runtime. Trusted stage grants come from the owning accepted
user request or existing acceptance/approval boundary. A model-supplied boolean
cannot authorize verification, continuation, export or release.

The guide alone cannot establish source immutability, actual model identity, or
complete report retention. Those facts must come from preparation, final provider
dispatch, and report-capture receipts. Stage sequencing and review judgment remain
agent behavior; mechanical identity, scope, and coverage validation are host checks.

## Child admission and bounded primary continuations

Preserve the existing per-turn child count of two and all narrower effective
limits. Do not raise global defaults, erase previous spawns, refund accepted
children, or reinterpret concurrency as a total-launch allowance.

Preparation records the number of requested reviewers, whether verification is
already authorized, the current turn's remaining child slots, and the remaining
ADR-134/135 automatic-chain allowance. Reserve the next launch batch through the
existing owners before dispatch. If the current turn has no remaining slot,
preparation returns `child_budget_unavailable` before source materialization; it
does not manufacture another manual turn. If later batches cannot be admitted
under known finite chain limits, return the capacity reason before paid work and
let the user choose a supported team or explicitly change those existing limits.

For the default two remaining slots, the admitted schedule is:

| Accepted request | Initial primary turn | Bounded automatic continuation |
| --- | --- | --- |
| Two reviewers, review only | Launch both reviewers; present originals | None |
| Two reviewers and pre-authorized verification | Launch both reviewers; present originals | Launch one fresh verifier |
| Three reviewers, review only | Launch two reviewers; show progress | Launch the third; present all originals |
| Three reviewers and pre-authorized verification | Launch two reviewers; show progress | Launch the third, present originals, then launch the verifier using the second slot |

Larger teams use further batches only while both the per-turn ceiling and the
original causal chain's finite generation/child/call/token/time budgets admit
them. For example, with two slots per turn, `ceil(max(N + V - R, 0) / 2)` is the
minimum number of additional turns, where `N` is reviewers, `V` is one only when
verification is pre-authorized, and `R` is current remaining slots. This is a
lower-bound admission check, not a guarantee about future provider calls or
resource availability. Recheck actual remaining capacity at each boundary.

A continuation needs a host-sealed review-boundary receipt: review/context-control
revision, exact next slot/slot-attempt/stage, settled preceding slot receipts, trusted stage
grant, source-path epoch, original chain and owner-runtime identity. Integrate
this typed eligibility with the existing Console wake/accepted-attempt boundary
(ADR-134/135); never fake a survivor result or parse authority from a prompt.
It is the only added continuation source and cannot submit arbitrary new tasks.
Duplicate callbacks/claims cannot mint another dispatch or a new causal allowance.

Extend the existing claim, read, accept, abort, completion and recovery owners as
one lifecycle. A proven pre-acceptance abort removes that attempt's active review
claim atomically with the existing source claims and may permit a new attempt only
while its live receipt remains valid. Keep aborted attempt audit rows; retain
accepted/completed/uncertain links and charges so cancellation or duplicate
callbacks cannot reclaim consumed work. Stop/close/epoch fences revoke live
eligibility before any retry; restart never reconstructs the ephemeral bundle.

Automatic continuations remain automatic: reserve/accept through the existing
ledger, honor manual drafts/queues and primary-capacity reserve, the autowake kill
switch, finite budgets and source/context fences. No review path can force a
wake when automatic work is disabled, exhausted or review-required. Show
`continuation_blocked` with saved progress and a manual resume action that is new
explicit user work; do not silently retry, replenish a chain or ask again for
already-granted stage permission. Stop cancels pending review boundaries before
child cancellation. Closing the tab or an uncertain accepted attempt forbids
automatic replay.

If verification was not originally authorized, the stage-one grant cannot launch
it. The user's later explicit acceptance starts ordinary accepted user work with
its normal fresh turn budget, referencing the same captured review context.
Continuation authorization never changes source identity or participant presets.
Verifier claim batches run inside that verifier's one child execution; a new
child is not spawned for every batch. Retries of failed roles require explicit
user work and new attempt receipts, preserving the prior reports and accounting.

## Default team and overrides

Store one device-local default under `[agents.code_review]` in the existing config
owner. `reviewers` is an ordered list of records with `preset_id` and optional
`focus`; `verifier_preset_id` references one existing preset. References use the
stable agent-definition IDs, while Settings displays their current names.

Defaults are an empty reviewer list and an unset verifier. The application never
selects paid provider/model destinations on the user's behalf. Users configure
the team in the existing canonical Settings Agents panel, beside preset editing.
Keep this to an ordered preset selector, optional focus per row, and a verifier
selector; no new destination, team editor framework, or workspace override layer.

The team needs at least two reviewers to be represented as multi-model review.
The suggested setup is two or three; the continuation admission above determines
admissible size separately from concurrency. A missing verifier does not prevent independent review; it
blocks stage two until the user selects a valid verifier. A missing/disabled
reviewer blocks launching the configured team until the user repairs the choice
or explicitly requests a smaller team.

Resolve explicit per-run choices over the saved team. Provider/model overrides
use existing ADR-147 admission and allowlists. Recipe loading never enables
overrides or rewrites a preset. Preparation accepts reviewer/verifier run choices
with preset ID, optional reviewer focus and optional destination override containing
provider and/or model. Freeze the requested override and resolved target before
scoped spawning; no fresh routing arguments can be supplied at launch.
Every effective reviewer and verifier needs an
explicit resolved destination; unresolved choices do not quietly inherit the
primary. Equal resolved provider/model destinations are shown as duplicates so
the user can knowingly retain them or change the team.

Capture the selected preset definitions, focuses, guide version, and resolved
targets for this review. A later preset/config edit cannot change an already
captured participant. Changed or revoked authority still refuses execution.
Configured fallbacks retain existing behavior; the report identifies requested
and actual destinations and every switch. Unknown upstream model identity stays
unknown; a model's self-description is not dispatch evidence.

## Target capture and source consistency

Each review has an opaque review ID and a manifest containing target kind,
repository/binding identity, captured revisions or content hashes, scope,
comparison semantics, capture time, guide version, and participant references.

| Target | Capture rule |
| --- | --- |
| PR | Resolve the intended repository and exact base/head commits; record merge-base versus direct comparison explicitly |
| Commit or range | Resolve immutable commit IDs and the chosen comparison; a single commit compares with its selected parent |
| Patch | Capture exact patch bytes and identify matching preimage/postimage context; reject an incompatible preimage rather than guess |
| Working tree | Capture staged/unstaged changes and any explicitly included untracked files, plus the source set used for review |

Acquire PR information through an already available, authorized integration or
local Git data. A URL alone does not grant network access or prove history is
available. If acquisition is unavailable, accept a user-supplied patch/context or
report the blocker. SSH-bound inputs retain their existing filesystem-only tool
limitations; Git operations are not silently moved to the local host.

Review children read a frozen, read-only view of admitted source. Committed
content can be materialized from recorded revisions. Dirty or patch context is
captured before spawning; partial reads or a changing capture fail consistency
checks. Source hashes describe the actual bytes delivered, not only the diff.
Additional context must come from the same captured revision/view. A request for
uncaptured mutable context requires a new capture or an explicit coverage gap.

The bounded source-read surface includes listing, literal search and relative-path
resolution within that view, so reviewers can discover unchanged callers/helpers
and tests without ordinary filesystem/Git tools. Queries select only a captured
comparison side; they cannot choose another revision. Committed context can be
materialized lazily under the same authority/quotas. Patch/working-tree targets
accept explicit `context_paths` captured before launch; discovery of mutable
content stays within that frozen inventory. Uncaptured context returns
`context_not_captured`, never current checkout bytes. Excluded names stay hidden,
and bounded/partial searches expose coverage gaps. New inventory publications do
not change the immutable target digest or another participant's submission revision.

Source-content reads accept `source_range={start_line, end_line}` with strict
1-based integer bounds within the captured document. This allows direct access to
a deep search hit without paging preceding content into the model. A range cannot
accompany a source query or a different view. Actual delivered spans/digests and
bounded continuation pages preserve the original selection, target/epoch and role;
conflicting or widened cursor selections refuse. Omitting a range starts at line 1.

Preparation preserves path exclusions, sensitive-path checks, symlink rules,
read-only bindings, and project-instruction handling. It must not bypass them
through a raw Git archive, whole-repository copy, raw shell, or direct host paths.
Project-instruction bodies retain their existing ephemeral delivery contract;
they are not copied into report metadata or logs.

The review provider implements the existing `PathAwareToolProvider.path_targets`
seam. Under current actor/source authority it maps captured refs and cursors to
their original admitted project paths, never physical capture paths. Literal
resolve queries map their validated file; broad list/search queries map their
admitted directory/root and return references/ranges, with source bodies delivered
only by a content read after exact-target preparation. The existing activation
ledger owns nested discovery, per-chain deferral/delivery, staleness and consent;
the mapping does not load guidance, capture source or advance cursors. Forged,
foreign, excluded or stale references cannot trigger unauthorized instruction
reads. Guidance remains ephemeral context and cannot grant permission.

Snapshot size/read limits use bounded source operations and the aggregate
reservations below. Missing, excluded, binary, unsupported, or truncated material is recorded as coverage, never
interpreted as absent code. If the full requested scope cannot be captured, state
the limitation before launching a narrowed review and obtain the user's scope
choice. No source checkout, commit, reset, or worktree creation is implied.

Later source edits do not change the review's meaning: verification uses the
original view. A review of newer code is a new target. If the original view is
unavailable, verification is blocked rather than silently using current files.

## Storage admission and release

Use these finite v1 limits, counted in actual UTF-8/file bytes rather than model
tokens. They are feature constants, not new user settings. Existing narrower
tool/file limits still apply.

| Scope | Limit |
| --- | ---: |
| Captured source and disposable verification copies/outputs per review | 64 MiB |
| Regular files in all review-owned areas, including staging and generated output | 2,048 per review |
| Original reports, raw role output, manifests, indexes and disposition ledger combined per review | 16 MiB |
| Raw output plus submitted report/dispositions for one participant | 1 MiB, within the combined 16 MiB |
| Total review-owned disk bytes, including staging and temporary writes, per review | 80 MiB |
| Retained plus reserved review-owned disk bytes per Console tab | 160 MiB |
| Retained review contexts per Console tab | Four, including active/partial contexts |

These are initial bounded-product limits, not measured throughput optima. Capture
the changed files and the admitted context set; do not materialize the whole
repository merely because a Git revision is available. This checkout has about
1 GiB of tracked content, which is intentionally beyond one source-view allowance.
Committed additional context may be materialized lazily from the captured revision
under the same file/byte reservation; dirty uncaptured context requires a new
consistent capture. Missing captured context remains an explicit coverage limit.

The session owner atomically reserves a context slot and byte allowances before
materialization. Source accounting includes paths, copied content and verification
working copies; metadata/report accounting includes raw and structured forms,
indexes and journal/staging bytes. Count physical copies separately; deduplication
is not required. Recheck available reservation before every append/chunk/write.
Concurrent preparations cannot spend the same remaining capacity. Release unused
source reservation after sealing and unused report reservation after settlement;
retained bytes continue to count until actual cleanup.

Oversized input returns `source_quota_exceeded` before launching reviewers. If a
changed capture, report stream, verification output or disk error exhausts an
admitted allowance, stop further affected-role work, retain accepted records and
mark the result partial with a content-free reason and retained/omitted counts.
Never truncate into an apparently complete report. Snapshot or report overflow
does not automatically broaden a limit or rerun a model.

When tab capacity/context count is exhausted, return `review_storage_full` and
list retained review IDs and sizes. Require explicit user choice to keep/export
and release a prior context, or narrow the new scope. Do not silently evict an
older report, an awaiting-verification review, or active work. Successful export
alone does not release the source bundle unless the user also requested release.

Explicit release and tab close tombstone access first; native execution/read
leases drain before filesystem cleanup and quota release. A terminal/cancelled
row is not proof that a writer or lease stopped. Late callbacks cannot publish
after tombstoning, and a cleanup failure retains its occupied quota until actual
deletion succeeds. These are per-tab limits, not a machine-wide disk guarantee.

## Reviewer task and capabilities

Every task includes the same shared brief, target manifest, access to the captured
source view, reporting contract, and relevant authorized project guidance. Add the
selected preset instructions and optional specialist focus without replacing the
general brief. Do not include other reviews or the primary's suspected defects.

The general brief prioritizes concrete correctness, security, compatibility, and
regression risks introduced by the change. It asks reviewers to inspect relevant
callers, tests, and contracts, distinguish pre-existing issues from introduced
ones, state uncertainty, and avoid style preferences or a findings quota.

Reviewers receive admitted source-inspection capabilities only. Close mutating,
process/script, peer-report, and unrelated history/runtime capabilities at their
actual gates; narrowing the preset's catalog allowlist alone is insufficient.
They cannot inspect other reviewer reports through shared scratch paths. This is
independence of supplied context and tool access, not a claim that model training
or underlying provider infrastructure is independent.

## Structured reports, identity, and retention

The host assigns participant slots such as `R1` and qualifies original findings as
`R1-F1`. These IDs are for the primary/user view, never the verifier view. Records
use `schema_version = 1`; unknown versions, unknown fields, duplicate local keys,
invalid enums and out-of-view source/evidence references refuse submission without
mutating the accepted ledger. Do not ask another model to reconstruct an index
from prose. The authoritative original report is the validated finding/coverage
record set rendered for the user; capture raw role output separately for inspection.

| Record | Required data |
| --- | --- |
| Reviewer finding | Stable per-role `local_key`; title; nonempty locations with captured `source_ref` and 1-based line spans; trigger; faulty behavior/impact; evidence references; severity (`critical`, `high`, `medium`, `low`); uncertainty text |
| Coverage | Inspected source references; excluded/unavailable/truncated areas and reasons; explicit empty-findings declaration when applicable |
| Host report envelope | Review/schema/guide/preset identity; slot and run receipt; target manifest digest; requested/actual destinations and fallback switches; capture/submission status and completeness |
| Verification disposition | Opaque `verification_unit_id`; `confirmed`, `refuted` or `unresolved`; reason; valid captured-source/check evidence references; optional proposed duplicate-unit reference; nonempty `next_check` required for unresolved, optional for confirmed/refuted |
| Check proposal/action/result | Verifier intent with opaque unit references; primary-only claim/decline; host result bound to target/attempt, with a native execution receipt only when execution occurred. Proposed, declined, unavailable, cancelled, expired or uncertain work cannot be presented as an executed success. |

Finding submissions may append bounded chunks under revision/submission-ID checks;
the host assigns an original ID once and acknowledges it durably within the live
scratch owner's atomic publication boundary. Replaying the same local key with
different bytes refuses. The participant submits coverage and requests completion;
the host seals its report only after the actual terminal run receipt and a complete
validated record set agree. Missing submission, failed capture, malformed data or
an unsuccessful run cannot produce a successful empty report. Terminal free text
is a summary, not an alternate authoritative findings channel; unstructured output
is retained/labeled as such rather than silently parsed as a complete report.

One finding record is at most 8 KiB serialized; a submission contains at most
eight records and also fits the caller's current input/output/tool envelope. Larger
records refuse with their local key and size, preserving earlier accepted chunks.
The per-participant and aggregate storage quotas bound record count/bytes. Any
unsubmitted or refused claim makes that role incomplete and visible in coverage;
there is no unlimited report-format repair loop.

Run status, requested/actual destination, fallback switches, guide/preset
fingerprints, and receipt completeness are host-derived envelope fields. A
failed/cancelled/budget-exhausted run or malformed report remains incomplete,
even if some findings are usable. It never becomes an empty successful review.

Use a review-owned bundle under the existing per-tab scratch manager for captured
source and complete, policy-permitted report bodies. Capture child output before
combined fleet rendering truncates it. Keep original reports separate from the
unified assessment; apply existing secret handling and disclose any withheld or
unretained content. A bounded run-row summary is not a full report.

Report access is bounded/paged and scoped to the owning chat and role. The primary
coordinates from the complete host finding index and capture receipts; it does
not load every raw report page into its prompt. Users retain separate access to
the full stored reports. Truncation, capture failure, storage caps, or missing
pages remain explicit. Never increase every tool-result budget or rely on
conversation compaction to make a verification input fit.

Every model-bound review response must fit its complete serialized envelope,
including records/text, cursor, digest and revision fields, within the smaller
positive run `max_tool_result_chars` and the existing 16,000-character native
continuation ceiling. Apply the native ceiling conservatively to both protocols,
including when the ordinary cap is unlimited. Keep byte/token checks independently.
Pack fewer whole items or source spans before generic runtime truncation; cursor
progress reflects only delivered content. An indivisible item yields an explicit
transport gap, preserving its stored original. If the minimum control envelope
cannot fit, refuse before capture/paid work or mutation and retain a host-visible
reason. Preflight mutating acknowledgments; never clip their IDs/revisions or
return an endlessly repeated empty page. Guide envelopes also obey their 12 KiB
byte cap and this smaller character limit, without semantic truncation.
Cursor replay preserves the pinned page or refuses if current capacity cannot
carry it; it never silently repacks that cursor. These limits govern model-bound
tool delivery, while native report/export readers retain their own bounded
authority and can show originals too large for that model transport.

The bundle is session-local and survives follow-up turns while its Console tab
and scratch lease exist. Tab closure/retirement may remove it; no crash recovery,
sync, or indefinite retention is promised. Individual reports are readable in the
chat before that boundary. Normal conversation persistence retains what was
actually presented; a transient report reference is not a durable full copy.
Users can explicitly export/keep reports through existing authorized destinations.
Reopening a chat without the bundle cannot claim to recover omitted raw reports.

## Fresh verification and unified assessment

Stage two launches one new child run using the captured verifier choice (or a new
explicitly accepted override), not a resumed reviewer. The initial task contains
the target digest and bounded verification-read interface, not every raw report.
The host, not the primary model, constructs the verifier view as follows:

1. Project each accepted original finding onto its substantive claim: target,
   locations, trigger, behavior/impact, evidence references and uncertainty. Omit
   original/slot IDs, author/model names, original severity votes, report headings,
   run receipts, report file references and agreement/group counts.
   Normalize evidence to common captured-source references, ranges and digests;
   per-reviewer read-receipt identities remain private.
2. Coalesce only byte-identical projected claims after deterministic JSON
   serialization; source identity is part of the key. This removes exact-repeat
   multiplicity without an LLM merge. Claims with different triggers, evidence or
   uncertainty stay distinct, even when they might describe the same defect.
3. Assign each projected claim an unrelated random opaque `verification_unit_id`.
   Keep the reversible original-ID-to-unit-ID map and multiplicity private to the
   host/primary. Order units by a stable opaque key, not by reviewer/report order.
4. Supply unit records in bounded batches without duplicate group proposals.
   Substantive source evidence is preserved; code can itself identify a vendor or
   author, and paraphrases can suggest related claims. The guarantee is removal
   of attribution and exact-repeat counts, not perfect statistical anonymity.

The verifier may propose duplicates only after independently assessing their
distinct units, and still owes a disposition for each unit. A duplicate proposal
cannot copy a verdict between nonidentical claims. Expand one accepted disposition
through the host map only for exact-repeat originals, preserving every original
ID and its independently captured provenance in the user report.

### Context admission and incremental verification

Use the selected verifier's existing effective context/capacity resolution and
token estimator. Before each provider call, account for the actual serialized
system/task/history, tools, project guidance, source evidence and configured
response reserve. Unknown capacity or insufficient space returns an explicit
capacity reason; it does not infer unlimited context or silently switch models.

Use explicit checkpoint-backed request-history reduction for the verifier child;
existing primary-only run-log eviction cannot provide it. Preserve the original
task, admitted guidance, active claim/evidence references, unacknowledged tool
rounds and required provider continuation/private state. Replace only completed
rounds whose results are recoverable through the accepted verifier checkpoint.
Do this before batch admission, independent of run-log settings, and validate the
final prepared provider request again. Reduction never grants history access,
replays a tool, resets a provider conversation or creates another verifier. An
indivisible protocol group that cannot fit produces an explicit request-level
`review_context_reduction_blocked` gap with retained progress.

Cached batch pages replay their original bytes, including historical revisions.
After reduction or cached replay, the verifier reads `view="checkpoint"` for its
current revision and next action. It never resubmits already accepted decisions.
An empty pending-delivered list is distinct from all units having dispositions:
undelivered units still require the next batch. Completion uses the current
checkpoint revision and remains subject to normal sealing and coverage checks.

A verification batch contains at most eight units and at most 4,096 estimated
claim-input tokens, further limited to half the remaining admissible input space.
The other half remains available for evidence reads; each source-evidence page is
at most 1,024 estimated tokens and is further narrowed to actual remaining space.
The ordinary final request-capacity check remains authoritative. A single claim
that cannot fit an otherwise empty bounded batch after recoverable-history
reduction is marked unresolved with `claim_too_large`,
without truncating away its trigger/evidence or silently changing its meaning.
A claim that fits model input but cannot fit alone with its complete tool-result
envelope is unresolved `review_record_too_large_for_tool_result`. Whole-envelope
packing and explicit gap accounting precede cursor advancement.

The verifier submits dispositions incrementally under its own attempt's submission
revision. Commit each accepted chunk and evidence receipt to the host ledger before
acknowledging it or reducing its completed history rounds. Return a stable
cursor and unverified-unit references; keep committed results outside prompt
history so compaction cannot erase progress. The next batch includes bounded
references to prior decisions when relevant, not all prior prose. A repeated
cursor/submission cannot double-count a finding, and a stale/conflicting update
cannot replace committed evidence without a new explicit verification attempt.

All batches remain inside the fresh verifier's one run and normal finite budgets.
If its run ends before finishing, the host finalizes remaining units as unresolved
with `verification_incomplete`. The primary can present the honest partial report
without ingesting all original bodies. Additional verifier attempts require new
explicit user work; batching itself never grants fresh calls, tokens or children.

The verifier checks claims against source and, when available and permitted,
targeted reproductions. Source-inspection authority stays read-only. Executable
checks require the existing process/permission path and a disposable verification
location; they must not mutate the user's checkout or execute automatically merely
because a reviewer suggested a command. Bind feedback availability from the actual
launch branch before the verifier's first tool call; preparation/settings previews
cannot replace that fact. An inline verifier's proposal is accepted together with
an immediate host `unavailable` result, reason `check_feedback_unavailable_inline`.
Its result read never waits or notifies the blocked synchronous parent. It may
complete source-based decisions; execution-dependent claims remain unresolved.
Do not force a fleet, retry on a fresh child or execute after that verifier returns.

A threaded verifier submits at most one outstanding proposal and awaits its
proposal-specific result through `verification_batch`. A pending request makes
the primary's existing `wait_agents`
return `check_requested`, including when it was already waiting; the verifier stays
live and uncollected. The primary reads the proposal, claims or declines it through
`check_actions`, then uses an existing authorized process tool with the host's
captured-target/disposable-executor specification. A claim binds one actual accepted
execution; replay cannot dispatch again. No extra child, automatic turn or permission
is granted. Host settlement returns bounded evidence to that same live verifier.

The result read waits on host state within the verifier's existing deadline and
cancellation fences, without model polling. Use the existing definitive-after-start
tool policy and finite owner bounds so a generic per-call timeout cannot abandon
the waiter. Actual tool-admission refusal resolves/signals unstarted checks without
bypassing capacity, including when a waiting verifier occupies the only tool worker.
Primary finalization resolves/signals pending work before fleet draining.
Denial, unavailable quota-enforcing execution, expiry, close or cancellation releases
the wait with the actual reason; dependent claims remain unresolved. Native execution
leases/charges persist until actual settlement/cleanup. A proposal alone is never
execution evidence. Once the verifier has ended, a late result cannot silently
rewrite its decision; it needs an explicit new verification attempt.

Every original claim receives one disposition:

| Disposition | Evidence standard |
| --- | --- |
| Confirmed | Source trace or reproduction establishes the trigger and faulty behavior; identify which kind of evidence was obtained |
| Refuted | Specific counter-evidence defeats the claim, with the relevant source/version or observed result |
| Unresolved | Context, capability, time, or evidence is insufficient; state what would settle it |

Failure to reproduce alone is not refutation. Agreement, model reputation, and
confidence wording do not prove a finding. Findings reported by only one model
remain eligible. Recheck duplicate groupings for distinct triggers or impacts.

Mechanical validation first requires one disposition for every verification unit,
no invented units and valid duplicate references. It then expands the private map
and requires every original finding ID exactly once in the final disposition
mapping. Missing, malformed or unprocessed units stay unresolved and make
verification incomplete. This validation establishes coverage, not semantic truth.

The primary creates a new unified report containing confirmed findings ordered by
impact, unresolved findings and next checks, refutations with reasons, the complete
original-to-canonical/disposition mapping, target identity, actual participants,
executed checks, and coverage gaps. Preserve disagreements and the original
reports. A newly discovered issue is separately labeled with its own evidence;
it must not be passed off as a reviewer's original claim.

## Failure and cancellation behavior

| Condition | Required behavior |
| --- | --- |
| No configured team | Explain setup in the existing Agents settings; do not guess destinations |
| Routing/consent/authority refusal | Preserve the refusing owner's reason; no recipe-specific bypass |
| Insufficient current child budget or known continuation allowance | Refuse preparation before source capture/paid work; report remaining total slots separately from concurrency |
| Automatic continuation disabled, exhausted or fenced | Preserve review progress as `continuation_blocked`; no new allowance, fabricated survivor, repeated acceptance question or automatic replay |
| Reviewer fails after team launch | Present available reports and the failed slot; offer retry or explicitly partial consolidation |
| Source capture drifts or lacks required history | Block a coherent-target claim; request a supported target/scope |
| Combined result is shortened | Retrieve complete captured reports; missing originals remain visible gaps |
| A report is too large for a model prompt | Read the structured index and bounded claim batches; do not concatenate raw report pages into the prompt |
| Invalid, stale or conflicting structured submission | Preserve accepted records; return the schema/reference/revision reason; never reconstruct findings from prose |
| Claim cannot fit selected verifier capacity | Preserve the original and mark its unit unresolved as `claim_too_large`; no silent model substitution |
| Source/tab/participant quota exhausted or disk write fails | Stop affected publication/work, retain accepted records, and report the relevant quota plus omitted coverage; no silent eviction |
| Verifier fails or exhausts budget | Preserve individual reports and completed checks; remaining claims stay unresolved |
| User declines verification | Stop after individual reports without launching a verifier |
| User cancels | Follow existing child cancellation and terminal-settlement rules; partial work stays labeled |
| Original source/report bundle has expired | Block complete follow-up verification and explain what must be re-supplied or reviewed again |

Retries and fallback retain existing runtime limits; this feature adds no unbounded
retry loop. Operations exceeding 100 ms run on existing worker paths. Every primary
turn retains its child cap; automatic continuations share their originating chain's
finite allowance, and verification batches share the verifier run's finite budgets.

## Verification required for implementation

Use targeted tests and the repository's testing/live-verification lessons:

1. Guide discovery, fixed packaged resources, bounded results, argument
   validation, tool gates, inline-skill trust, and installed-wheel behavior.
2. Team ordering, stable references, missing/disabled presets, optional focuses,
   duplicate destinations, per-run overrides, and canonical Settings persistence.
3. Concurrent children through the real Console adapter to the final provider
   boundary: requested/actual identities, sampling ownership, fallback attribution,
   continuation isolation, and unchanged primary selection.
4. Pinned PR/range/patch/dirty inputs, changing captures, excluded paths, symlinks,
   unavailable/binary/truncated source, role isolation, and no implicit source edits.
5. Full-report capture/readback versus fleet and run-row truncation, malformed
   reports, cross-chat refusal, session expiry, cancellation, and failure statuses.
6. Acceptance/decline/pre-authorized stage two, a fresh verifier, identity-blinded
   initial claims, complete finding coverage, invalid duplicate references, and
   confirmed/refuted/unresolved evidence semantics.
7. A small recorded behavioral qualification through the actual Console/provider
   path with two configured reviewer destinations and a fresh verifier. Include
   a seeded defect, false positive, duplicate, failed reviewer, truncated-result
   control, and source-drift control. Attribute scripted/fault-injection results
   separately from live-model observations; schema-valid output alone is not proof
   of useful review. Record actual model calls and checks without credentials.
8. Actual default-two-child admission: two/three reviewers, with and without
   pre-authorized verification, prior spawns in the initial turn, capacity
   refusal, duplicate boundary/acceptance callbacks, one-chain accounting,
   autowake disabled/exhausted, manual priority, stop/close/context changes and
   uncertain accepted continuation. Count actual launches, not planned slots.
   Also exercise actual inline spawning without an injected fleet, immediate
   unavailable checks and feedback-mode binding after settings change across turns.
9. Versioned finding/disposition schemas, source-reference validation, role-bound
   submission, idempotent retries, stale revisions, malformed/unstructured output,
   sealing only after terminal receipts, and complete unit-to-original expansion.
10. Provenance-free verifier projection: no participant prefixes, report ordering,
    models, private receipts, group counts or original severity votes; exact-repeat
    coalescing preserves all original mappings, while nonidentical triggers remain
    distinct and require individual dispositions.
11. Small-context verifier admission, oversized individual claims, evidence-page
    bounds, multi-batch progress across compaction, interrupted final batches and
    post-return evidence. Assert exact request-size admission and durable-within-
    session ledger coverage rather than requiring all raw reports in one prompt.
    Independently test small result-character limits, complete JSON/cursor/revision
    delivery through native/fenced append, and indivisible transport-size gaps.
12. Source/report/context quotas at the exact boundary and one byte/file/context
    above it; concurrent reservations; staging/verification-output accounting;
    actual disk errors; explicit release; cleanup failure; leased/cancelled writers;
    and late callbacks. Tab close must not refund capacity before cleanup settles.
13. Captured-reference/query mapping through the real instruction ledger, nested
    guidance delivered once per chain before source evidence, no cursor progress
    on deferral, and unchanged exclusion/consent/permission boundaries.

Update `Docs/Architecture/agent-runtime.md`, `tool-catalog.md`, and
`console-file-authority.md` where their implemented contracts change, plus the
relevant user guide and example-skill documentation. Update the canonical Settings
surface using ADR-150/161 tokens and patterns. Any changed existing DB owner needs
its normal versioned migration; this design does not require a new review database.

## Scope exclusions and implementation handoff

The first version has one global default team plus per-run overrides. It does not
add named-team management, per-workspace teams, a model benchmark, new provider
adapters, a general workflow engine, automatic fixes, automatic GitHub comments,
or merge actions. Durable review-job recovery and a universal artifact store are
outside this feature.

## Design-review resolutions

| Review issue | Resolution |
| --- | --- |
| Two-child default cannot launch reviewers plus verifier | Explicit total-slot preflight and host-sealed, same-chain primary continuations; preserve each turn's cap and automatic-work limits |
| Participant-qualified IDs expose reviewer grouping | Private many-to-one original map and unrelated verifier IDs; remove exact-repeat multiplicity, attribution and report grouping from initial batches |
| Preparation/report support lacks a firm contract | Define prepare, scoped spawn, role-aware read, structured submission and release contracts; versioned record schemas, idempotency, terminal sealing and host validation |
| Paging does not bound accumulated model context | Primary coordinates from the finding index; verifier uses capacity-admitted batches/evidence pages and an incremental host disposition ledger |
| Per-operation bounds do not cap total retained storage | Fixed per-review/per-participant/per-tab byte and file/context caps, atomic reservations, explicit release and cleanup-aware accounting |

TASK-35582 is In Progress following user authorization on 2026-10-18. Its
Implementation Plan links the approved slices and ADR check. Implementation Notes
and completed acceptance criteria require recorded implementation/verification
evidence. This specification is not an implementation-completion claim.


### Native acceptance and continuation implementation amendment (Task35582, slice 7)

The native Console owns a session-local `ConsoleCodeReviewCoordinator`. An actual
accepted USER message (owning active path, no automatic origin) registers stage
intent before tool parameters exist. Direct review authorizes reviewers; explicit
review-and-verify authorizes both. A saved verifier authorizes neither a later stage
nor a short affirmative. Quoted, conditional and negated controls fail closed.
A native, exact-review verification offer may accept a later USER “yes”; a decline,
source fence or session fence invalidates it. Model prose cannot install an offer.

A host prepare-admission callback narrows that intent to exact target, team and
run-override fingerprints. Tool JSON proposes scope inside existing intent; it
cannot create a grant, select an unadmitted binding, waive destination consent or
add verification. The native guide exposes a default working-tree target only for
the already selected admitted local binding. Missing/ambiguous source or review
selection returns an explicit needs-attention outcome. No blanket second stage
question is added to an already explicit review-and-verify instruction.

Native source authorization reuses the composed `ToolCatalogRegistry` run-policy
counter and `LocalToolProvider.authorize_review_source` / `_verdict_for` owner,
including current permission state, persona floors, native approval, admitted
binding guard and revocation. Capture and later source access require this gate;
review intent and project-instruction consent are not file permissions. Destination
admission reuses `confirm_project_dispatch` and
`commit_project_instruction_dispatch_decision` for the actual resolved target,
current selected binding and instruction state. The existing consent owner retains
its provider/endpoint scope; no parallel destination-consent store is introduced.

Each accepted automatic continuation or later explicit manual stage binds the
actual new primary actor, exact live grants and existing source authority to the
retained context. Context metadata consumes its existing quota. Original
preparation, old streams and terminal receipts remain immutable; retries receive a
new slot attempt. Later manual acceptance uses fresh current turn capacity and
consent. Its default verifier retains the admitted definition/route; an explicit
new choice is routed and frozen through current admission. Serialized metadata or
same-session membership cannot authorize takeover. Retired contexts and controller
shutdown discard their ephemeral authority registries.

The existing `ConsoleFleetWakeCoordinator` remains the only delivery owner. AgentRuns
schema 24→25 adds the body-free `automatic_review_boundary_claims` link and nullable
boundary identity on existing wake attempts. Ordinary cause/attempt-kind constraints
remain unchanged. Claim validates the live host receipt and actual terminal SQLite
predecessor steps, then reserves one normal fleet-wake generation atomically.
Acceptance revalidates live authority before provider preparation. Proven prepared
abort deletes only its exact active link and releases that reservation; its aborted
attempt ID remains audit-only. Accepted, completed and uncertain links/charges remain
consumed. Restart enters existing review-required recovery; it reconstructs no bundle
or executable grant. Notifications occur outside review-store/SQLite transaction
locks, avoiding a store→DB / DB→store lock inversion.

Native settlement mounts or reuses `ConsoleCodeReviewModal` and displays each
reviewer’s original page before acknowledging presentation. Its paged reader
retains full originals, raw output and captured evidence; `u` reopens it in the
owning Console chat. The transcript separately records the session-local reference.
The nonzero Console journeys exercise this actual handoff before fresh verification.
Normal persistence of content actually presented
is permitted; automatic source/raw/report bundle bodies remain outside DB/log metadata
and later model history. Disabled, exhausted, fenced or uncertain continuation
projects `continuation_blocked` and retains available progress without promising an
automatic retry. Explicit new work never resets the old automatic chain.


Task 7 review corrections: verification authority accepts only a complete,
unqualified native stage clause; conditional, negated or unknown qualifications
leave verification ungranted until a later accepted instruction or exact native
offer acceptance. `Retry reviewer R1` binds R1 from the accepted USER message;
caller/tool slot proposals cannot substitute another slot. `Resume review` binds
only never-started reviewer slots in the single retained review to fresh current
manual admission, preserving the original capture, frozen routes, earlier streams
and terminal receipts. It revalidates destination consent and current child source
permission and assigns fresh attempts; it does not carry old verification intent
into a reviewer-only request. Ambiguous reviewer/review selection fails explicitly.
Old live wake eligibility is superseded and removed from the existing wake owner
outside review locks; stale callbacks cannot overwrite the new revision's status.
Prepared ownership is released only by the existing proven-abort lifecycle;
accepted/uncertain evidence is never refunded or replayed. A still-accepted missing
slot blocks manual resume until its outcome is known. The old automatic chain and
its used/reserved budget remain unchanged.

Later manual-stage grants record accepted stage intent against the retained context;
they are composed before the new primary's local permission callback is installed.
They never substitute for file permission: native child claims and source
read/materialization fail closed if that current callback is missing or denies.
The initial preparation continues to authorize source access before capture and
exact preparation-grant issuance. Fixture-only direct issuer/preparation helpers
remain distinct from the actual Console admission evidence.


Task 7 acceptance-race correction: the registered native boundary authority also
supplies an independent acceptance scope. Existing ledger acceptance acquires that
scope before its SQLite admission transaction and retains it through the
post-commit native accepted transition. Final manual-stage publication acquires
the same scope before the review-store lock, then revalidates exact selected-slot
claims and accepted/uncertain boundary state before issuing or retiring authority.
Destination confirmation holds neither scope nor publication locks. This gives
old acceptance versus manual supersession one winner, including the interval
after validation and the interval after commit but before native notification.
Ordinary claim-time validation never acquires the acceptance scope: its existing
progress-lock→DB→store/registry order cannot wait on a scope held by acceptance
while acceptance waits on DB. Manual publication makes no AgentRuns ledger calls. Accepted
links/charges remain consumed; manual wins leave the old preparation available
only to the existing proven-abort cleanup. No schema or durable-body changes.


### Qualification integration: native selected-destination continuity

Real native custom-endpoint qualification exposed a generic provider-continuation
identity mismatch. Existing ADR-063/147 amendments preserve raw selected endpoint
and canonical execution identity through the existing assistant continuation owner,
using nested destination JSON V2 and continuation JSON V3. This does not persist
review jobs, extend expired source leases or restore review authority. The existing
read/submit role checks, ephemeral source/store retention, native budgets and manual
Resume rules remain. Exact repair scope and compatibility requirements are recorded
in the implementation-plan addendum and QA/ac24/continuation-identity-design.


Native CaptureOn compatibility follows the ADR097 live-native amendment and
QA/ac24/native-trace-delta-design contract. The repair preserves source/role/provider
authority and current checkpoint format while supporting exact captured native
rounds; no generic workflow or durable review-job recovery is introduced.


## Accepted review-wake provider/catalog repair (adopted 2026-10-27)

Follow QA/ac24/review-wake-repair-design and amended ADR281/230 plus ADR063/179
carrier clarification. Preserve SYSTEM/AGENT_WAKE order and strict engine ownership
using existing distinct_roles where a hosted request has a later SYSTEM row. For
exact accepted review wakes only, obtain bounded current pinned capacity before
freeze, revalidate all lifecycle/source/grant fences, and require actual stage tool
reachability after normal planning. Ordinary nonblocking/4096bootstrap/R57 behavior
is unchanged. No authority, schema or budget increase; failed accepted preparation
retains charges/reports and creates no replay. Actual configured handler-backed
native [2,1] completion with cold metadata is the integration oracle.


### Source correction to accepted-wake carrier diagnosis (Ruling63)

Exact BASE7952422 controller source deliberately projects automatic SYSTEM
transcript notices as payload-only trailing USER while preserving AGENT_WAKE
authority and order. The prior reconstructed SYSTEM-carrier diagnosis was not
the actual wire path. Its carrier conversion/Safe-capture clauses are superseded;
ordinary serialization and the strict engine stay unchanged. Accepted-review-only
pinned capacity/revalidation and normal schema reachability requirements remain.
The full handler-backed fixture instead exposes completed checkpoint forwarding
without native replay rows. A focused replay contract is pending before source
changes at that extra boundary. Original diagnostic evidence is retained with
QA/ac24/review-wake-repair-design/diagnosis-controller-correction.md. No authority,
schema or additional live allowance follows from this correction.


### Canonical completed-history replay amendment (Ruling64, before code)

Adopt QA/ac24/review-wake-repair-design/completed-replay-correction.md and
completed-replay-controller-assessment.md. Preserve the corrected existing
transcript SYSTEM/payload USER policy; carrier-only work is removed. The real
accepted-wake failure is retained complete checkpoint forwarding without its
canonical native owner rows, alongside the independently proved capacity/tool loss.

After existing host owner/target admission, derive canonical recorded assistant/
tool history at final preparation from the exact complete persisted checkpoint.
Reuse the existing translator; keep one collapsed semantic owner plus attachment
as the atomic budget/eviction unit. Preserve exact IDs/args/results and one final
content, private reasoning sidecar, all retained complete groups alongside current
active groups, child isolation and explicit active Resume gates. No re-execution,
fuzzy owner match, missing group tolerance or engine relaxation. Count the same
projection before fit and avoid duplicate completed-expanded body charge.

A narrowly deterministic one-to-many trace derivation may reuse existing completed
owner/revision, private attachment and terminal lineage/range machinery. Recompute
all output rows under frozen policy; derived rows are not fabricated saved ordinary
revisions. Preserve count/span/source/actor/turn/CAS and final shadow gates. No
arbitrary tool span acceptance, active-custody resurrection, growing per-call
history arrays, codec/GC change, registry or new schema. Any needed durable shape
requires concrete assessment before adding it. Existing provider/checkpointV3 and
legacy readers remain. Actual handler-backed [2,1] plus CaptureOff/On, replay/live
coexistence, counting/eviction and strict negatives are required before closure.


### Implemented accepted-wake and replay boundary

The corrected contract is implemented and independently reviewed through c3f99ec.
Accepted review continuations use bounded pinned capacity/stage reachability;
complete engine checkpoints replay canonical recorded rows and the saved final
owner once, with atomic semantic grouping/counting and strict incoming ownership.
Existing body-free review-tool persistence and frozen disclosure projections stay;
transient known credentials reach final trace binding without new secret storage.
Product automatic wakes remain CaptureOff; Safe/Full evidence uses eligible later
MANUAL turns. QA/ac24/review-wake-repair-fix records passing native fake-HTTP [2,1],
masking/reconstruction/rejection controls and the one deferred Minor fixture gap.
AC23 is supported; live AC24 remains unqualified. Pending-dispatch refusal recovery
keeps its visible shutdown limitation; one equivalent BASE probe narrows historical
attribution without proving all new predicates or application exit.
