# Chunking Lab: template authoring and recoverable A/B experiments

Date: 2026-09-04
Status: Implemented and reviewed under ADR-118; final targeted acceptance passed473 tests. Qualifications and historical failures are documented in Docs/Chunking_Lab_Verification.md.
Scope: v1. Task-level implementation/review is recorded in completed TASK-31421–TASK-31428; no merge or push is implied.

## 1. Purpose and confirmed decisions

Chunking Lab helps a user understand how a chunking recipe treats real text,
refine it, and save a reusable custom template. It is a keyboard-first Textual
workbench within Chatbook's existing visual system (Operate mode).

Confirmed with the user:

- v1: one sample with one configuration or lightweight comparison of two.
- v2: three or more configurations against the same sample.
- v3: corpus evaluation, with an Evals module/screen connection to be designed.
- v1 editing: method-specific controls plus JSON; advanced settings survive
  round trips, and preview executes the full configuration.
- v1 execution and template saving are local. Results record their backend so
  future server support does not require guessing where a run happened.
- Reopening automatically restores the previous sample, A/B configurations,
  and completed results to recover from bugs and crashes. Manual experiment
  saving is not a prerequisite for recovery.

The user accepted the design defaults in sections 3–10, then requested a review
before planning and authorized incorporating its seven findings and two improvements.
The limits remain initial engineering defaults to verify during implementation,
not measured performance claims.

## 2. Existing code and architectural implications

The first review inspected this older checkout. A fresh all-ref planning sweep
found completed template convergence on `origin/dev`, inspected at commit
`91757b61e9c7e9f920d80a0ce282261b4161ffff` on 2026-09-04. That is the implementation
baseline; do not execute this plan against the retired implementation here.

- ADR-078 and completed TASK-19801–19806 establish one flat body:
  `{preprocessing, chunking, postprocessing, classifier, metadata}`. Name,
  description, and tags are record fields. `chunking.config` contains method options.
- The Media DB is the only template catalog. It already has UUIDs, integer versions,
  builtin protection, soft deletion, live-name uniqueness, and validate-on-write.
  Add atomic optimistic concurrency using those versions, not another migration.
- `Chunking/template_runtime.py` already owns mapping, name resolution, and full
  pre/chunk/post execution. Extend this seam; do not recreate a pipeline engine.
  `chunking_templates.py` and its file store were deleted. Inheritance and multiple
  chunk-stage authoring are therefore not v1 features.
- `RAG_Admin/template_validation.py` deliberately matches server validation,
  including unknown-operation acceptance. Keep it intact. A separate local-preview
  capability preflight must reject unsupported/no-op executable settings; its
  verdict is not a claim of server validation parity.
- The current runtime synthesizes offsets and can discard preprocessing metadata.
  Lab results need trustworthy structured output and explicit unavailable mapping,
  implemented at the non-vendored seam or refused by capability preflight.
- The former template editor and template list widgets were deliberately removed
  as unreachable legacy UI in TASK-253. They are reference material, not new UI
  foundations.
- TASK-24404 proposes a Settings-hosted creation form and is still To Do on dev.
  This design replaces that placement with Library ownership under ADR-003.
  Reconcile that task before landing the Lab UI; do not ship two authoring surfaces.
- ADR-073 requires reuse of the vendored chunk engine/processor, with Chatbook changes
  outside the vendored tree. No second splitting implementation is introduced.

ADR required: yes

ADR path: `backlog/decisions/118-chunking-lab-local-execution-and-recovery.md`.

Reason: the new workflow defines durable recovery storage, full-template execution
semantics, result provenance, and long-lived UI ownership. Existing ADR-003 and
ADR-073 and ADR-078 constrain it but do not settle those new decisions.

Applicable decisions:

- [ADR-003: Settings/Library boundary](../../../backlog/decisions/003-settings-library-rag-defaults.md)
- [ADR-031: keybindings and truthful footer hints](../../../backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-073: one vendored chunking engine](../../../backlog/decisions/073-vendored-chunking-engine-parity.md)
- [ADR-076: Library discovery and deep links](../../../backlog/decisions/076-library-lifecycle-progressive-disclosure.md)
- ADR-078: `backlog/decisions/078-chunking-template-convergence.md` (on dev;
  absent from this older checkout).
- [ADR-118: Lab execution and recovery](../../../backlog/decisions/118-chunking-lab-local-execution-and-recovery.md)
- [Screen decomposition design](2026-08-02-screen-decomposition-design.md)

## 3. Placement and screen structure

Owner: Library. Provide a Library tool entry and command-palette action
named **Chunking Lab**, plus an **Experiment with chunking** action on a local
Library item's extracted text. Use a dedicated screen reached from Library,
with a return destination, so the experiment can use the available terminal area
without growing the Library screen's internal implementation. Its local backend
stays explicit even if the surrounding app is using a server runtime.

The header carries the sample label, Local execution, persistence status, and
the active candidate's primary Run preview action. Existing Library starter
filtering must not hide a direct command or deep link.

Three functional regions share the workbench:

1. **Sample:** source details, Paste / Load text file / Choose Library text,
   editable sampled text, and an explicit full-text/excerpt indication.
2. **Configuration:** active candidate, template picker, method-specific controls,
   Controls / JSON views, validation, Pin as A, and Save as template.
3. **Results:** A / B / Compare views, chunk list, full selected chunk, statistics,
   execution details, and source/transformed-text inspection.

At wide sizes, comparison gives A and B equal space and can collapse the sample
and editor to make room. At narrow sizes, show one primary region at a time with
explicit Sample / Configure / Results navigation; A/B switching retains the
selected source location or independent chunk selection. Never squeeze two
unreadable text columns together. Proposal: verify at 80x24, 120x40, and 160x50.

Use the existing theme tokens, dense fields, readable text states, and pane focus
conventions. No new visual identity or global navigation strip is needed. F6
uses the app's pane traversal; F1 describes the active actions. Proposed local
shortcuts are r run, p pin baseline, s save template, and Escape back, active only
outside text entry and where the action works. Visible buttons remain available
while typing. Copy/select/undo within text editors retain native widget behavior.

## 4. Sample and A/B lifecycle

Start with one editable candidate B; the display can call it Configuration until
A is pinned. Users may start from a basic method or duplicate a saved template.

Supported sample sources in v1 are pasted text, a UTF-8 text file, and already
extracted text from an eligible local Library item. PDF parsing, transcription,
remote fetches, and ingestion belong to the existing import workflows. Selecting
a source copies its text into the session; the Lab does not edit the original.

Both candidates run against the same immutable sample revision. Explicit edits
or reloads create a new revision. Existing results remain inspectable against
their original text and become stale relative to the current sample. The Lab
never silently clips a source to fit its sample limit.

**Pin as A** requires a completed result matching B's current valid draft and
sample. It copies the exact recipe, resolved dependencies, and result into A,
then leaves B editable. Replacing an existing A requires a deliberate Replace
baseline action. An existing template can be loaded into B before pinning it.

A's configuration is frozen, but can be rerun against the current sample. B's
configuration can change independently. A sample change stales both outputs;
Run both runs the two recipes against the new sample under the same engine and
pipeline-execution versions. Configuration edits stale only that candidate. Pinning, template
loading, and source replacement are undoable within the working session.

Only one preview executes at a time in v1. Run both captures one batch before
either candidate executes: exact sample, both full configurations, loaded record
identities, and required runtime identities. Validate both candidates and persist
the batch manifest before starting A, then schedule A followed by B using only
those captured inputs. A save failure blocks launching the batch with Retry/Export
recovery available. Execution must not reread a mutable saved template when
B starts. An unavailable captured runtime produces an error, not a substitution.

Editing while a run proceeds is allowed: it affects the next batch, and completed
outputs are marked stale relative to changed drafts. A failed A may be followed
by B to expose both outcomes; cancellation stops the active run and the remaining
queue. A batch is comparable only when both captured runs complete successfully.
Any older retained output is labeled Previous result and must not be substituted
for a failed batch member. New execution waits until the previous worker has
stopped. Navigating away requests cancellation and checkpoints state; reopening
marks unfinished batch members Interrupted and never automatically reruns them.

## 5. Lossless controls and JSON

Keep the raw JSON text and last successfully parsed document separately. JSON
parsing is not a persistence prerequisite. A typed execution model must not be
used to rewrite the stored authoring document if it would discard unknown keys.

- Controls patch only their known paths in the complete document.
- Valid JSON refreshes the controls. Additional fields/stages remain in JSON.
- Invalid JSON stays editable and recoverable with line/column diagnostics;
  controls and preview are suspended until it parses. Save as template is also
  disabled, while draft recovery and export remain possible.
- Invalid intermediate control text is recovered too. It must not replace a
  valid numeric value with zero/default or accidentally run an older valid draft.
- Controls edit the flat body's one `chunking` block. Ordered preprocessing and
  postprocessing operations and advanced settings use JSON in v1.
- Method changes preserve unrelated fields. Incompatible existing options remain
  visible with actionable errors; removing them is an explicit edit.
- Switching views preserves all values and ordering of operation arrays. JSON
  whitespace/indentation can change after a control edit; semantic values cannot.
- Units are method-specific: words, sentences, paragraphs, tokens, or the actual
  structural unit used by that method. Tokenizer identity is explicit when relevant.

An incomplete control edit owns a pending patch against the last valid document;
that document is not treated as the current executable draft. Switching to JSON
may show it read-only with a Pending control edits notice naming the fields.
The user must correct those fields or explicitly Discard pending control edits
before JSON editing resumes. Run, Pin, and Save remain disabled while a pending
invalid patch exists. Returning to Controls restores the exact entered strings.
Conversely, invalid raw JSON owns the draft and blocks control edits until corrected
or explicitly discarded. Persist the owning view, raw input, pending patches, and
base document together so restart cannot resolve competing edits differently.

Preserve unknown metadata and extension fields. An unknown field with potentially
executable meaning is preserved but blocks execution until supported or removed.
The UI must distinguish preservation from support. Unsupported templates can be
retained in session recovery and exported, even when they cannot be saved as a
validated runnable template.

Use ADR-078's flat body as the v1 authoring format, with name/description/tags
edited separately. Preserve imported legacy/stage-shaped documents as recoverable
raw drafts with an unsupported-format error; do not silently convert, introduce
inheritance, or revive the retired file store. Existing migration/quarantine tools
remain the route for legacy records. Classifier rules are preserved as selection
metadata and clearly labeled not evaluated by a direct single-sample preview;
they must never select a different recipe behind the editor.

## 6. Full-template execution and provenance

Extend the existing headless `Chunking/template_runtime.py` seam to produce a
structured execution report for sampled text and an unsaved flat body. Lab preview
and saved-template apply must share execution, with the existing list return shape
kept as an adapter. No new splitting or template operation algorithms are needed.

Preflight runs server-shape validation plus a separately named Lab capability check:
all requested operations/options must be understood and supported locally, required
assets must already exist, and execution defaults must be captured explicitly.
No hidden config/provider values may change a captured run. The supported set is
test-backed; reject known lossy/ignored combinations until the seam can preserve
their meaning. This stricter Lab gate does not globally redefine ADR-078 validation.

### Template loading and snapshots

Load templates only through the existing DB-backed service. Capture UUID/version,
record fields, and the complete body when loaded; editing creates a detached draft.
Runs consume immutable body/options snapshots, not DB handles, live caches, or file
references. Pinning freezes those inputs. A later saved-template edit cannot change
A or a queued B. Refreshing B from the catalog is an explicit undoable replacement.
Import is a detached draft, never another template namespace.

### Chunk and operation contract

Preserve engine and operation output as internal chunk records at the runtime seam:
`text`, engine metadata, source-span information with its coordinate space, and
transformation provenance. Engine strings become records with empty metadata and
unavailable spans unless authoritative mapping is provided. Dictionary outputs
retain their non-text fields; no stringify-and-discard conversion is allowed.

Reuse the vendored operations without changing their accepted input algorithms.
Capture their structured effects outside the vendored engine, and do not hand
dictionaries to string-only operations or coerce them into meaningless strings.
Declare each operation's accepted input/output shape and validate stage placement.
For v1:

- Filtering preserves surviving records and their spans; recalculate list indices
  and totals after the pipeline completes.
- Metadata-only operations preserve text and spans and keep user metadata separate
  from authoritative provenance/count fields so key collisions cannot overwrite them.
- Context/overlap insertion preserves contributing records' metadata and identifies
  inserted text. It invalidates the output's exact source map unless every segment
  has a trustworthy mapping; inherited offsets must not masquerade as a full map.
- Merging records retains each contributor's metadata/provenance separately, without
  last-writer-wins field loss. Produce an exact combined map only when all segments
  and inserted separators are accounted for; otherwise label mapping unavailable.
- Preprocessing that changes text invalidates original-source mappings unless the
  operation supplies a verified transformation map. A later chunk stage's offsets
  then refer to processed text and must not be relabeled as original-source offsets.

Reject unsupported operation/shape combinations with a stage-specific error rather
than dropping metadata to make them run. Golden fixtures must exercise dict-producing
methods through filtering, merging, and context insertion, including metadata key
collisions and unavailable mappings. Persist structured records; legacy callers
that require a flat response flatten only at their outer return adapter.

The flat recipe executes preprocessing, its one chunk stage, then postprocessing;
operation arrays preserve their declared order. An empty result after filtering is
a valid zero-chunk result, not permission to fall back to unconfigured chunking.
If the vendored processor cannot meet a combination's semantics, refuse that
combination rather than fork its algorithms. Unknown operations, conditions, and
unsupported options produce specific preflight errors, not successful partial runs.

v1 execution scope: local non-LLM methods and operations supported by
the existing engine. Local-only preview must never activate a configured cloud
provider or download missing models/tokenizers implicitly. Templates requiring
LLM operations remain recoverable/editable with an unavailable explanation.
Adding local LLM-assisted operations is a separate scope decision, not inferred
from the word "local".

Each completed run stores:

- Run ID, optional comparison batch ID, candidate ID, input revision, sample identity
  and exact sampled text.
- Original full configuration and resolved effective configuration/dependencies.
- Backend (`local` in v1), engine and pipeline-execution versions, relevant local
  tokenizer/dependency identities, start/end times, and terminal status.
- Structured chunks, available metadata, diagnostics, and metric definitions/units.
- Trustworthy source spans or an explicit mapping-unavailable state.

Compute content identities from the exact sample and complete executable inputs;
do not identify a run by mutable template name. Completed records are immutable.
The save/edit association is separate from the identity of a preview result.
This complete run-input identity distinguishes results; it is not the criterion
for allowing comparison between two intentionally different configurations.

Heavy execution runs off the Textual event loop. Use a bounded worker process
for non-cooperative CPU/regex work so cancellation and execution limits have a
real termination boundary. Give each request its session epoch and revision;
late completion cannot revive a cleared session or overwrite a newer result.

Proposed v1 limits for review: 2 MiB UTF-8 sample text, 10,000 output chunks,
32 MiB serialized output per result, and 60 seconds per preview. Enforce them
before or during execution, with a failed/limited state and intact previous
results. Offer choosing a smaller excerpt explicitly. These are engineering
defaults, not performance claims. A serialized-output check alone is insufficient
to bound intermediate memory; process resource behavior needs targeted verification.

## 7. Comparison and interpretation

Separate run identity from comparison compatibility. Direct v1 comparison requires
successful results with identical sample content identities, backend, engine version,
and pipeline-execution version. Other runtime/dependency differences are shown, not
automatically rejected: different methods, options, and chunking tokenizers are
legitimate experimental variables. Historical outputs remain readable, but a
sample/backend/version mismatch receives a reason and Run both recovery action
instead of a misleading comparative delta. Restored results from the same older
engine can still be compared to each other; the warning concerns their age, not
their compatibility with each other.

Show chunk count; minimum/median/p95/maximum size; total emitted size; elapsed
time; and oversized chunks against an explicit method-specific budget where one
exists. Characters are universally countable; words and tokens have named counting
semantics. Character counts and chunk counts can be compared across methods.
Method-budget units remain labeled separately: a paragraph budget is not subtracted
from a word budget. Token-count deltas require the same measurement tokenizer;
if candidates' chunking tokenizers differ, show labeled separate counts without a
token delta. Recounting both outputs with one explicitly selected local measurement
tokenizer can enable that delta without rerunning chunking. Measurement identity
is stored separately from run identity. A single elapsed time is a runtime
observation, not a benchmark ranking. Do not invent a quality score.

Compare includes a compact configuration diff tied to the selected result snapshots,
not the newer live draft: added, removed, and changed JSON paths with A/B values.
Default to effective executable settings, including captured defaults;
offer the complete authored-document diff to inspect metadata and classifier rules.
Pipeline arrays are order-sensitive and compared by position; do not imply that
same-numbered stages are semantically equivalent. Long values open in an inspector
rather than truncating the only available evidence. Runtime/dependency differences
are shown alongside the diff, and a stale badge identifies any newer live edits.

Selecting a chunk links to the original source span and overlapping chunks in the
other output only when the execution supplies reliable spans. Repeated text must
not be aligned using a guessed string match. Preprocessing or generated context
may prevent an exact map; then inspect transformed/output text and show why source
alignment is unavailable. Span coordinate systems must be identified explicitly.

Overlap statistics are measured from valid mappings, or labeled unavailable.
Total emitted/source size ratio is expansion, not measured overlap: inserted
metadata/context and transformations can also increase it. Never use the old
preview widget's character approximation as evidence.

Statistics describe output shape. Corpus retrieval quality requires queries and
expected evidence or relevance judgments; this belongs to the v3 evaluation design.

## 8. Automatic recovery

Persistence boundary: one active experiment per local application profile,
in a dedicated versioned SQLite recovery database under that profile's data root.
This keeps autosave independent of template mutations and avoids a main conversation
database migration for scratch work. It is local application data, not config TOML,
a temporary directory, repository content, or an Evals run.

Store a session document plus immutable sample/result snapshots referenced by it.
Draft checkpoints update small state without rewriting every chunk. In a transaction,
save a completed output and publish its session reference together. Retain the
current and previous valid checkpoint and all snapshots they reference; garbage
collect only unreferenced snapshots after the new checkpoint commits.
Snapshots needed by an available in-session undo action also count as referenced.
While Undo restore is available, its displaced checkpoint is an explicit retained
reference separate from the rolling previous checkpoint; view-only autosaves
cannot advance it out of retention. Clear session removes all of these references.

Recover sample text and provenance, raw/invalid JSON, incomplete control input,
A/B recipes, completed outputs, stale state, active views, and selected chunks.
Text is restored from the stored snapshot even when its original source disappears.

Autosave proposal: 300 ms trailing debounce, with a maximum one-second checkpoint
interval during continuous editing under normal storage conditions. Sample
replacement, pinning A, run start/completion, and navigation request immediate
checkpoints. Flush on orderly exit. Show Saving / Saved locally / Save failed;
only a committed transaction earns Saved locally. A sudden crash can lose changes
after the last committed checkpoint; do not promise zero-keystroke loss.

Every recovery-relevant mutation increments the session revision, including raw
invalid input. Autosave requests and acknowledgments carry profile, session epoch,
and revision. Serialize writes through one session writer; coalescing may skip an
intermediate draft only when a newer full checkpoint includes its retained state.
Show Saved locally only when the acknowledged committed revision equals the latest
in-memory revision in the same profile/epoch. Acknowledging revision 12 while the
user is editing revision 13 leaves Saving visible. A delayed acknowledgment cannot
clear a newer Save failed state, regress the displayed saved time, or mark a restored
or cleared session saved. Retry writes the latest in-memory checkpoint, not the
failed request's obsolete snapshot.

If a run was active at shutdown, recover it as Interrupted and retain the previous
completed output. A changed engine leaves old results readable with a version
label. Two old results from the same execution versions remain comparable; mixing
old and newly produced results requires rerunning both. Recovery never starts execution.

On write failure, retain the working session in memory, show the last saved time,
and offer Retry and Export recovery snapshot. A result that failed persistence
can be inspected but is explicitly Unsaved. A restore read failure or unsupported
newer schema must not initialize an empty database over the existing one. Recover
the previous valid checkpoint where possible and explain any rollback; otherwise
preserve the file and expose recovery/export guidance.

Two app instances must not silently overwrite the same active session. Use revision
checks for saves; on conflict preserve the losing in-memory draft and offer reload
or export. A separate multi-experiment manager is outside v1.

### Export and restore recovery snapshots

Pair Export recovery snapshot with **Restore recovery snapshot**, available even
when the normal session store cannot be read. Export a versioned JSON envelope of
the current in-memory checkpoint, its referenced samples/results, raw editor input,
pending patches, and dependency snapshots. Export does not require the database
to be writable or a template to be executable. Include integrity digests for
sample/result references; digests detect inconsistency, not trusted authorship.
Do not export unrelated profiles, the undo history, or the previous checkpoint.

Restore first validates the envelope version, structural types, reference integrity,
candidate count, and bounded file/payload sizes without altering the active session.
Initial transfer limits: 256 MiB total envelope, 2 MiB per raw draft, 8 MiB current
checkpoint excluding blobs, at most 16 referenced sample/result blobs combined,
and JSON nesting depth 64, alongside the existing per-sample/result/chunk limits.
Apply matching bounds to active state: reject an over-limit edit explicitly and
retain its previous value rather than accepting an experiment that cannot export.
Previous checkpoints and undo history are not part of the export envelope.
Raw invalid template JSON and incomplete controls are opaque draft content and
must survive this validation exactly; executable-template validation is deferred
to Run/Save. Imported active runs become Interrupted. No embedded file path or
template reference triggers a filesystem read, network call, or automatic execution.
Do not silently truncate an oversized import or drop an incompatible result.

Show a summary of the candidate/sample/results to restore and require an explicit
Replace current session action. Before replacement, persist the current in-memory
checkpoint as the previous checkpoint and install the imported session atomically,
with a new session epoch. Invalidate pending old writes, cancel old runs, and wait
for their termination before publishing the replacement. If preserving the old
session or committing the replacement fails, leave it active and report the error.
Replacement is a guarded transition: suspend content edits while cancellation and
the serialized writer settle, then commit. The persisted epoch changes only on
successful replacement; failure retains the old session's save/retry authority.
Read-only inspection of the validated import remains possible until persistence
is repaired. A malformed/newer envelope never overwrites the active session.

Retain the displaced checkpoint as a one-level recovery undo until the first
subsequent content mutation, not merely until a view-selection autosave. Expose
Undo restore during that interval. This is a bounded replacement safeguard, not
a named experiment/history feature; Clear session removes it as well.

Recovery data includes full sample text and outputs and is stored under the existing
local profile protections; this feature adds no new encryption claim. Show a small
"Session and sample saved locally" explanation. Clear session requires confirmation,
increments the session epoch, cancels work, and removes its current/previous
recovery snapshots. Pending writes cannot resurrect it. Ordinary SQLite deletion
is not a secure-erasure promise.

## 9. Saving reusable templates

Use the existing local template storage service and its validate-on-write boundary.
Lab-originated saves additionally require the same capability preflight as preview,
enforced in a headless service entry, not only by disabling a UI button. Validate
the final body and record fields after save-dialog edits. Keep the existing global
server-parity validator unchanged. Preserve name/description/tags as record fields,
reject the case-insensitive whole-word reserved name `auto`, and surface listing
decorations for stored-invalid and reserved-name records. Tags normalize to their
existing column; do not treat that documented record/body boundary as data loss.

Default to Save as new for built-ins and externally loaded configurations. Existing
custom templates can be updated explicitly using an expected UUID/version captured
when loaded. Compare and update atomically within the media database transaction;
an earlier UI-side check followed by an unconditional update is forbidden. Use a
compare-and-swap predicate over record ID, UUID, version, live state, and builtin
protection, incrementing the existing version on success. No Media DB migration is
needed. A timestamp alone is insufficient. Zero matching rows
means a conflict or deletion: retain the draft and offer Reload or Save as new,
never an automatic overwrite. Built-in protection must be rechecked in that same
transaction. Creation relies on the database's uniqueness constraint to arbitrate
concurrent same-name saves. Successful saves refresh the existing ingest picker's
cached template list. Do not add a second Settings editor under TASK-24404.

Saving validates the whole document but does not require a successful preview.
Show "Not previewed with current settings" where applicable. Save A uses its
pinned effective recipe; Save B uses its current validated body and record fields.
Neither operation may drop metadata or other advanced fields. A method change
never silently removes an incompatible option; explain the required explicit edit.

Autosave is automatic session recovery; Save as template is explicit creation or
update of a reusable recipe. The interface must distinguish these actions. Saving
does not re-chunk the sample's Library source, change global RAG defaults, rebuild
embeddings, or automatically enroll content in evaluation.

Include JSON template import/export and the paired recovery export/restore workflow
in section 8. A searchable saved-template picker is sufficient for v1; a general
template management destination and named experiment library are deferred.

## 10. Boundaries, evolution, and alternatives

Keep the screen thin. Its region widgets/controller belong in a dedicated
`UI/Chunking_Lab_Modules/` package; headless execution belongs under `Chunking/`,
recovery storage follows `DB/` conventions, and template CRUD stays in `RAG_Admin`.
The app/profile owns recovery lifetime; ephemeral widgets do not own durability.

Use stable candidate IDs in the session model, with an explicit two-candidate v1
limit, rather than designing storage around permanent `result_a`/`result_b` columns.
Do not build arbitrary comparison scheduling, server adapters, or Evals plugins
before those versions are scoped.

v2 adds a candidate collection, configuration/result summary table, and inspection
of a selected pair on one sample. Existing v1 sessions migrate without discarding A/B.

v3 should let Evals own corpus selection, evaluation definitions, run scheduling,
aggregate metrics, and history. "Evaluate in Evals" passes immutable configurations
and dependency identities. "Inspect in Chunking Lab" opens one evaluated sample
and its results while preserving the user's existing active session before replacing
it. The existing Evals screen/bench contracts need dedicated design and tests; a
navigation link alone does not deliver chunking evaluation. No inert Evals action
ships in v1.

Alternatives considered:

- **Settings-hosted editor:** easy discovery beside defaults, but mixes an active
  experiment and its sample data into a configuration owner; Library is preferred.
- **JSON-only editor:** smaller form surface, but obscures method units and makes
  common tuning slow. Controls plus lossless JSON fits both common and advanced use.
- **Full visual pipeline builder:** powerful, but duplicates stage composition UI
  before the lossless execution contract is established. Defer beyond v1.
- **Manual-only experiment saving:** rejected by the user in favor of automatic
  recovery of samples and completed results.
- **Single JSON recovery file:** simple for small drafts, but repeatedly rewrites
  sample/results and complicates atomic publication of larger outputs. SQLite is
  proposed to separate small checkpoints from immutable result payloads.
- **Preview via method/options extraction:** contradicted by the full-configuration
  requirement; preserve pipeline execution and structured results instead.

## 11. Acceptance and evidence

Implementation tasks should cover these outcomes independently:

1. The Lab is reachable through Library and the palette and returns to its opener.
2. Paste, text-file load, and local Library text create recoverable sample snapshots.
3. A user previews B, pins A, edits B, compares, and saves either recipe locally.
4. Controls/JSON/save/reopen preserve nested unknown metadata, additional options,
   ordered operations, and complete advanced configurations without silent field loss.
5. Invalid JSON and incomplete controls survive a process restart exactly as entered.
6. Full pre/chunk/post execution is identical between draft preview and applying the
   same saved template under the same inputs; meaningful fixtures prove both stages
   alter the result. Legitimate zero-output pipelines do not trigger a fallback.
7. Unknown executable settings, invalid conditions, missing dependencies, and
   unsupported methods block clearly; local-only runs make no implicit network calls.
8. Loading captures the full saved body and UUID/version; later catalog changes
   cannot mutate pinned A or an already queued B.
9. Every result identifies its sample, full effective recipe, backend, and engine;
   late or canceled work cannot overwrite a newer/cleared session.
10. Changed inputs stale the correct outputs; mismatched samples/engines cannot
    produce an apparently valid comparison.
11. Restart after a committed checkpoint restores sample, A/B drafts/results, and
    view state; an active run becomes Interrupted without rerunning automatically.
12. Crash injection around result publication restores either the old complete
    checkpoint or the new complete checkpoint, never mixed configuration/result data.
13. Failed writes, incompatible schema, and concurrent-instance conflicts preserve
    recoverable data and do not falsely show Saved locally.
14. Exact source mapping is verified on repeated text; transformations without
    mappings expose unavailable alignment/overlap instead of guessed highlights.
15. Cancellation, sample/output limits, and large-result browsing keep the UI usable.
16. Built-in templates remain protected; explicit saving affects no Library content
    or global defaults. Clear session removes its recovery references and survives
    queued autosaves and late workers.
17. Keyboard-only flows and readable persistence/error states work at the proposed
    terminal sizes, including typing r/p/s inside editors and actual footer actions.
18. The Lab writes ADR-078 flat bodies through the sole DB catalog. Legacy pipeline
    or parent-reference documents remain recoverable but block Run/Save; the retired
    file store and inheritance resolver are not reintroduced.
19. Dict-producing methods pass through supported postprocessing without string-type
    crashes or lost contributor metadata; modified text never keeps invalid exact spans.
20. Concurrent template updates cannot overwrite an intervening record change;
    invalid but syntactically correct templates fail create/update at the save boundary.
21. Run both freezes both recipes and the sample before A starts. Editing any input
    or updating a saved template while A runs cannot change B's captured execution inputs;
    a failed member cannot borrow a previous result to complete the comparison.
22. Different methods/tokenizers can be compared on common measurements; full run
    identities differ as expected, and incompatible measurement units have no delta.
23. Switching views with invalid control input exposes the pending edit and blocks
    JSON mutation, Run, Pin, and Save until correction or explicit discard; restart
    restores the same editing authority and values.
24. Export from an unwritable session followed by restore into a writable profile
    reproduces the sample, invalid drafts, pending controls, and completed A/B results.
    Malformed imports, newer envelope versions, and failed replacement transactions
    preserve the current session. Old workers/writes cannot resurrect replaced state.
25. Configuration diffs compare the selected results' captured documents, preserve
    ordered operation differences, captured defaults, and newer-draft staleness.
26. An older autosave acknowledgment cannot mark a newer draft Saved locally or
    clear a newer failure. View-only saves do not remove the one-level Undo restore.

Use focused unit/property tests for lossless document edits and revision rules,
real temporary SQLite for transactions/migrations and concurrent writers, subprocess
crash tests for durable recovery, and Textual Pilot plus isolated live terminal
verification for the user flow. Use real chunk execution fixtures, not only mocks
of a successful response. Do not launch against the user's normal profile or run
the full test suite without their opt-in. This design phase runs no app tests.

## 12. Review resolution and next step

The source-backed review identified seven implementation-contract gaps. This revision
resolves them in the relevant sections rather than leaving an independent issue list:

| Review finding/improvement | Resolution |
| --- | --- |
| Lossy inheritance and competing parent namespaces | Sections 2/6: reuse completed ADR-078 convergence; no new inheritance system |
| String-only postprocessors versus structured chunks | Section 6: chunk records and operation-specific metadata/mapping rules |
| Missing full validation and atomic save conflict checks | Section 9: strengthened transactional save boundary |
| Mutable inputs between A and B in Run both | Section 4: captured and persisted batch before execution |
| Run identity conflated with comparison compatibility | Section 7: separate compatibility and metric-unit rules |
| Pending invalid controls versus JSON editing authority | Section 5: explicit correction/discard handoff, persisted pending edits |
| Export without a restore path | Section 8: validated export/restore with atomic replacement and recovery undo |
| Inspectable configuration differences | Section 7: result-snapshot configuration diff |
| Older save acknowledgment falsely reporting latest draft saved | Section 8: profile/epoch/revision-aware status |

Acceptance outcomes 18–26 make these resolutions testable. This is a design review,
not proof that the implementation already satisfies them.

Planning reconciliation: the initial review's legacy implementation findings are
historical, not a request to rebuild deleted infrastructure. ADR-118 adopts ADR-078
and adds a Lab-only capability gate and recovery boundary. TASK-24404 must be
reconciled to this Library-owned authoring workflow before the UI lands.

Historical planning handoff: the [implementation plan](../plans/2026-09-04-chunking-lab.md)
was subsequently executed in the isolated `codex/chunking-lab` worktree.
TASK-24404 was archived as superseded. Final review found seven integration gaps
and five bounded refinements; the single correction wave is tracked in TASK-31428
and `.superpowers/sdd/2026-09-04-chunking-lab/final-fix-report.md`.
Final acceptance awaits the controller's scoped re-review. The original integration
gate remains non-green; see [verification and limitations](../../Chunking_Lab_Verification.md).
