# Portrait compositing design and plan self-review

Date: 2026-10-24
Status: Revised-plan second review complete after resolving all three follow-up findings; implementation not started
Scope: Updated design, ADR-243, follow-up plan and TASK-36551–36555
Source inspected: local `60b7694af9`, original generation/draft/repository/editor code and prior artwork qualification evidence.

## Initial planning review (historical)

The initial conclusion below was superseded by three findings in the requested
follow-up review. Their fixes and the second review are recorded at the end.

### Initial result

The plan covers the requested one-time manual selection, three edit requests and
exact locally assembled eye/mouth pairs. It keeps the existing six-frame playback
and shared provider/admission boundaries. No unresolved planning blocker remains.
The five tasks are To Do with unchecked outcome AC; this review is not product,
migration, provider, UI or audio qualification.

## Corrections incorporated during review

1. **Coordinate reuse needs durable ownership.** Session-only masks would require
   selecting again after reopen. Added explicit Save regions with full source/
   prepared-base identity and an independent CAS revision. Saving changes no active
   frame/performance revision; unchanged current-revision Save is a no-op.
2. **A retry cannot replace only one blink pose.** Defined component-dependent groups
   and atomic staging. An incomplete/failed candidate cannot publish, and unaffected
   frames stay byte-identical. Paid request counts describe components rather than
   six output roles.
3. **Reopen must not feather twice.** Store a bounded optional recipe snapshot on
   the applied parent, recover finished planes from existing frames only after exact
   pixel checks, and use binary support copying during assembly. No extra persisted
   component images or raw provider outputs are required.
4. **Arbitrary bases are not necessarily a resting face.** Require an explicit
   open-eyes/closed-mouth base confirmation, with manual six-frame imports remaining
   available. No implicit fourth request, source retouch or semantic face detector.
5. **Manual edit failure must preserve work.** A successful Import/Clear switches to
   manual lineage; cancelled/invalid imports preserve both frames and components.
   Changed saved regions retire only the current draft sample and its disposable
   generated work, leaving the applied set active.
6. **Persistence needs an actual next-version migration and real restore/edit test.**
   Corrected the SQL allowlist path to `tldw_chatbook/DB/sql_validation.py`, require
   executed/packaged migration SQL, and defer the version filename until implementation.
   Add fresh restored application open followed by another component retry, plus
   unchanged actual portable export/sync controls.
7. **Small previews hid the original failure.** Require a full-resolution labeled
   2×3 matrix, detail-mask review and Console-sized playback. Pixel invariants prove
   pairing/preserved surroundings, not plausible eyelids, alignment or seamless tone.
8. **Historical evidence must stay honest.** Original six-request tasks/reviews stay
   historical. The updated spec links the follow-up plan; ADR-242 carries partial
   supersession metadata and ADR-243 owns new decisions. Real configured in-app
   adapter evidence is distinct from external ChatGPT patches; unavailable coverage
   stays open rather than being counted from mocks.

## Coverage map

| Requirement | Atomic owner |
| --- | --- |
| Strict region/recipe geometry, version and pixel/alpha/pair invariants | TASK-36551 |
| Budget accounting, explicit image closure and no repeated feathering | TASK-36551 |
| Dependent frame-group atomicity and unaffected-byte preservation | TASK-36551, TASK-36554 |
| Separate settings/performance revisions, lineage and next-version migration | TASK-36552 |
| Ordinary import/playback compatibility, purge, restore and export boundaries | TASK-36552, TASK-36555 |
| Source-space mapping, keyboard/numeric adjustment and detailed masks | TASK-36553 |
| Neutral-base confirmation, explicit Save/Cancel, unchanged-save no-op | TASK-36552, TASK-36553 |
| Three-call disclosure, one-call component retry, partial/uncertain failure | TASK-36554 |
| Source/profile/editor/config/recipe races and late completion rejection | TASK-36552–36554 |
| Installed editor → Console → reopen, restore → another edit | TASK-36555 |
| Full-resolution review, real adapter proof and truthful qualification limits | TASK-36555 |
| Targeted static/token/CSS/SQL/packaging/Backlog checks and final docs | Each owning task; combined TASK-36555 |

## Planning verification

- Refreshed origin; all reachable-object task/ADR IDs and 35 registered worktrees
  were swept before creation (max 36550 / 242). Explicit new IDs are 36551–36555
  and ADR-243. This is a snapshot, not a future reservation.
- Backlog ID/path guard: all 5511 task files unique by filename and frontmatter,
  with Windows-compatible paths.
- Backlog readability guard: all 5511 task files pass frontmatter/owned-marker checks.
- Five new task files: To Do, sequential outcome AC, only existing lower-numbered
  dependencies, valid ADR/spec/plan references, no checked AC or premature task
  Implementation Plan/Notes. The separate plan contains the execution steps.
- Local document links, Markdown fence balance, API/field consistency, requirement
  coverage and placeholder scan reviewed; `git diff --check` passes.

No application tests, provider requests or image/audio edits ran for this planning
update. No product code, schema file, dependency, user profile or default changed.

## Revised plan review after three findings

Reviewed after the user's request to address the three source-anchored findings.
The prior planning commit is `9e143ea951`; current product source remains unchanged.

### Resolved findings

1. **Fresh batches could inherit complete old slots.** Kept manual `stage_group`
   separate from authoritative `replace_composition`. A single immutable candidate
   owns components, recipe/revision and all current frames. After consent, install
   rest/zero components before any request. `can_apply`, Apply/sample buttons and
   the publication accessor all require three current components plus six current
   roles in composed mode. Retry failure preserves the previous valid snapshot;
   fresh-batch failure cannot use old slots. TASK-36551/36554/36555 now require the
   previously-complete-set → failed-fresh-batch regression and unchanged active DB.
2. **PNG encoding changes could invalidate otherwise identical regions.** Replaced
   the planned encoded-base hash with `base_pixel_sha256`, using the exact tagged
   RGBA8/dimension serialization in the spec. Original source and artifact byte
   digests remain separate. TASK-36551/36552/36553/36555 now cover equivalent encodings,
   stable revision/reopen/retry, and true source/dimension/RGBA rejection controls.
3. **Full-resolution pair review had no product implementation owner.** TASK-36554
   now owns `PortraitCompositionReview`, its Review pairs action, original-PNG input,
   labeled overview/common native-detail crops and keyboard pan. TASK-36555 must open
   that actual modal; an external QA sheet cannot count. The pixel witness test fails
   if the viewer consumes a fitted thumbnail. Sequential source decode, a bounded
   raster, explicit close/resize/stale disposal and shared budgets prevent a new
   six-canvas memory spike. Static/Reduce motion needs no new clock or audio owner.

### Second-pass checks

- Traced complete, partial, retry-failed, fresh-failed, manual-imported, region-changed
  and reopened states. Manual merging cannot satisfy composed readiness. Snapshot
  validation failure preserves frames/components/provenance/readiness atomically;
  cancelled consent sends no calls and changes no draft. Changed regions reset to
  an empty composed snapshot rather than implicitly becoming manual.
- API/field names match across design, plan and all five task AC. Component ordering
  and available roles are explicit; candidate bytes are hidden from repr and released
  on discard. Existing source authority and both DB revisions still fence publication.
- Canonical pixel identity is shared by saved settings and applied provenance. PNG
  byte hashing remains solely artifact/source integrity; equivalent encodings do
  not weaken original-source identity or admit actual pixel/geometry changes.
- The second pass also traced existing Apply re-normalization: unchanged stored
  roles are content-validated and retained byte-for-byte under CAS, so encoder
  differences do not defeat retry preservation. New/changed inputs still normalize;
  TASK-36552 owns a focused repository control.
- Viewer detail reads original normalized frames, uses one common source rectangle,
  preserves crop RGBA without pre-downsampling and pans instead of allocating an
  unbounded montage. The controller actively retires the displayed modal on owner/
  candidate changes; worker predicates alone are insufficient. Final terminal
  rendering limits are stated; no physical pixel equivalence or semantic art-quality
  detector is claimed. Review is available before Apply without a new checkbox gate.
- Existing ADR-243 already chooses fresh candidates, source-bound base identity and
  pair review. Its accepted decision statements are unchanged; only metadata links
  these implementation clarifications. No new architectural boundary or ADR needed.
- Backlog parse/ID guards pass across 5511 task files. Five tasks remain To Do with
  34 unchecked AC, prior-only dependencies, valid ADR/spec/plan links and no premature
  task Implementation Plan/Notes. Markdown links/fences, placeholder/API scans and
  `git diff --check` pass. Planning controls also confirm the encoded-hash counterexample
  and canonical-hash equivalence on synthetic in-memory PNGs.

**Conclusion:** all three findings are addressed and no further blocking planning
issue remains in the revised design, plan or task criteria. Implementation and all
runtime/provider/viewer assertions remain To Do; the checks above review contracts
and documentation, not shipped behavior. No application tests, migrations, provider
requests, actual portrait edits, full sweep, push, PR or merge ran in this revision.
