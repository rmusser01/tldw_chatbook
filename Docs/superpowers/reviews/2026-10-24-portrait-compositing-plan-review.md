# Portrait compositing design and plan self-review

Date: 2026-10-24
Status: Planning review complete; implementation not started
Scope: Updated design, ADR-243, follow-up plan and TASK-36551–36555
Source inspected: local `60b7694af9`, original generation/draft/repository/editor code and prior artwork qualification evidence.

## Result

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
