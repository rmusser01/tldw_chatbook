---
id: TASK-31391
title: Add current-server snapshots and reusable vLLM launch profiles
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 22:34'
updated_date: '2026-09-04 18:11'
labels:
  - vllm
  - lab
  - profiles
dependencies:
  - TASK-31389
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make repeated vLLM operation efficient and honest by separating the immutable running configuration from editable restart intent and retaining reusable non-secret launch profiles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The UI displays an immutable current-server snapshot separately from the next-launch draft.
- [x] #2 Edits made while running are labeled as next-restart changes and can be applied with one Restart with draft action.
- [x] #3 Users can create, select, rename, duplicate, and delete named vLLM profiles containing only approved non-secret launch fields.
- [x] #4 The last selected vLLM view and profile restore across screen recomposition and application restart.
- [x] #5 Storage, migration if required, privacy, and profile lifecycle tests cover invalid, stale, and recovery states.
- [x] #6 Profile-loaded structured expert values are always visible and editable under Advanced, invalid/repairable profile fields receive adjacent recovery, and active-runtime Current versus Next restart context stays accurate through selection, lifecycle, and draft changes.
- [x] #7 Create/save/rename/duplicate validation failures map to the correct visible adjacent profile, source, model, environment, network, or Advanced control with bounded actionable copy; duplicate-name and invalid-rename outcomes do not fall back to generic reload messaging.
- [x] #8 Initial saved-profile hydration is distinguished from deliberate profile selection: exact current launch evidence survives only for the bound profile/runtime fingerprint, and all profile mutation/select handlers are inert in existing-server mode even when invoked programmatically.
- [x] #9 Initial profile-store hydration is a fail-closed readiness gate: success reconciles the exact selected profile before exposing READY, while corrupt, future, unavailable, or failed loads invalidate stale evidence and show persistent adjacent recovery without hiding truthful Stop.
- [x] #10 A newly constructed or lazily mounted child view begins unreconciled and cannot project any saved-profile READY evidence until its parent explicitly supplies a reconciled profile document.
- [x] #11 Before initial profile reconciliation, local profile selection/mutation and every draft field or mode control are disabled and forged child/screen events cannot mutate the placeholder draft, repository, or inherited connection generation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add strict launch-profile schema validation and a versioned device-local repository with optimistic revisions, safe canonical names, a 32-profile cap, selected-profile restoration, and atomic writes.
2. Extend vLLM launch ownership with immutable current-server snapshots and restart sequencing that preserves generation fencing, claim ownership, and exact-process termination guarantees.
3. Add the profile/current-server/restart controls to the vLLM Lab, keeping editable next-restart configuration distinct from immutable current state and using thread workers for repository I/O.
4. Cover schema, migration/future-version behavior, atomicity, optimistic conflicts, name collisions, profile UX, snapshot privacy, and restart safety with focused tests.
5. Run focused verification, update the ADR-linked task notes and acceptance criteria, and record exact evidence in the implementation report.

Fix Round 1:
6. Add regression tests and strict source-specific model validation at construction, decode, and pre-write boundaries without disclosing rejected values.
7. Harden document and adjacent lock handling against symlinks using descriptor-based no-follow checks, and add a deterministic two-process same-revision race test.
8. Amend accepted ADR-117 to the approved Task 4 exact V1 schema and rerun all scoped verification gates.

Fix Round 2:
9. Add deterministic regressions for unavailable ownership verification and lock-path replacement after acquisition, proving bytes and lock targets remain untouched.
10. Fail closed when platform ownership cannot be verified and revalidate the held lock descriptor/path identity immediately before protected document access and atomic replacement.

Fix Round 3:
11. Add fresh-repository regressions for missing, non-callable, raising, and invalid-result effective-UID capabilities, asserting generic repository errors and zero filesystem mutation.
12. Preflight and normalize ownership capability before directory/lock creation, reuse the validated effective UID for descriptor checks, then rerun all scoped gates.

Fix Round 4:
13. Add RED assertions for a raising ownership probe that require both `__cause__` and `__context__` to be `None`, no canary anywhere in the reachable exception graph, a generic safe message, and zero filesystem mutation for absent and existing parents.
14. Normalize ownership-probe failures outside the active exception handler without changing UID validation, descriptor checks, symlink defenses, locking, or CAS behavior.
15. Run isolated GREEN, the focused Task 4 suites, whole vLLM workflow, and exact static gates; append evidence, restore Done, and commit the narrow fix.

Task 6 Fix Round 2:
16. Add sequential RED tests for byte-capped profile reads, duplicate keys at every JSON object level, byte preservation, and real success/failure log-sink privacy canaries.
17. Route profile writes through a privacy-safe atomic wrapper that preserves lock/CAS/no-follow/inode guarantees, and reject oversized or duplicate-key documents before schema construction.
18. Run focused GREEN, the complete profile/core matrix, and exact static/diff/privacy gates; append evidence before restoring Done.

ADR required: no new ADR
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 remains the accepted device-local profile/CAS boundary; this round enforces bounded strict reads and privacy-safe atomic outcomes without changing the architecture.

Final UX fix round:
19. Add RED mounted profile tests proving every persisted structured expert value is projected into a visible editable control and every invalid/repairable field receives bounded adjacent recovery.
20. Add RED current-versus-next projection tests across profile selection, draft edits, checks, lifecycle changes, and presentation recomposition.
21. Implement only the view/controller projection needed for those outcomes; retain the exact V1 schema, CAS, atomic-write, privacy, and launch-only raw-argument boundaries.
22. Run focused RED/GREEN nodes sequentially and the complete primary, compatibility, responsive, profile/privacy, static, inventory, CSS, and diff gates before checking the new AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 already defines these structured profile fields and immutable Current versus editable Next ownership; this round repairs their presentation without changing storage.

UX Fix Round 2/5:
23. Add RED mounted tests for duplicate-name, invalid rename, source-specific model-value recovery, profile-cap/schema-field mapping, and existing-server action availability.
24. Classify only allowlisted profile validation messages at the controller boundary, then project bounded adjacent copy to the exact visible control while preserving generic reload recovery for corrupt/future/conflict storage outcomes.
25. Run focused RED/GREEN profile/view nodes and the complete requested gates before checking the new AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 already fixes the exact profile schema, local-only ownership, corrupt-store recovery, and adjacent repair outcome; this round corrects presentation routing only.

UX Fix Round 3/5:
26. Add RED profile-hydration and forged-event regressions covering exact bound-profile preservation, mismatch invalidation, and byte/revision stability in existing-server mode.
27. Reuse the app-scoped connection owner and repository revision as the authoritative reconciliation seams; do not introduce persistence or handoff changes.
28. Run focused repository/workflow GREEN checks and the requested complete gates before checking the AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This is a narrow correction within ADR-117's existing profile, ownership, and readiness contract.

UX Fix Round 4/5:
29. Add sequential RED delayed-profile-load success and failure regressions around a newly mounted screen carrying app-scoped READY evidence.
30. Project hydration as an explicit readiness prerequisite, retain bounded adjacent profile-store recovery across recomposition, and reconcile exact saved-profile identity before restoring actions.
31. Run focused profile/workflow GREEN checks and the requested full gates before checking the new AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This tightens ADR-117's existing selected-profile restoration and fail-closed corrupt-store behavior; no profile schema, storage, or ownership boundary changes.

UX Fix Round 5/5:
32. Extend the delayed-load RED test to cover the child view's constructor default, initial mount, direct lifecycle projection, and all saved-readiness surfaces.
33. Make the profile-ready flag fail closed by default and require explicit parent hydration before projecting any inherited profile readiness; retain the current exact reconciliation path.
34. Run the complete requested lifecycle, profile, responsive, privacy, static, inventory, CSS, and diff qualification before checking the new AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 already makes selected profile restoration a prerequisite for reusable readiness; this change closes a default-state projection hole without changing storage.
Final closure:
35. Add a RED delayed-repository race that exercises mounted controls and forged messages while the profile worker is blocked.
36. Gate child draft/profile emitters and LLMScreen mutation handlers on completed hydration, leaving Stop independent.
37. Run the complete requested qualification, record evidence, check the new AC, and restore Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This completes ADR-117's existing selected-profile hydration prerequisite without changing profile storage, schema, or ownership.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-117’s device-local vLLM profile and restart boundaries. Added exact strict V1 JSON profiles at get_user_data_dir()/vllm_launch_profiles.json with a 32-profile cap, canonical Unicode/casefold name uniqueness, deterministic duplicate suffixes, restrictive atomic writes, and cross-process portalocker CAS; future/corrupt documents and write failures remain byte-preserving and fail closed. Added selected-profile restoration through thread-worker I/O, launch-only raw-argument isolation, exact claim-bound immutable Current server snapshots, safe Next restart dirty labels, and two-generation restart sequencing that proves the old process dead and claim released before a new reservation. Termination failure retains the old snapshot and creates no second process. No DB/schema migration.

Fix Round 1 tightened the same ADR-117 boundary: model values now receive source-specific non-secret validation during construction, decode, and immediately before write; nonexistent safe absolute local paths remain repairable while repository-ID, option, traversal, control, and credential-URL violations fail generically. Profile document and adjacent lock leaves now use no-follow descriptor opens, regular/current-owner/private-mode and inode-identity checks, with no chmod of existing pathnames. A simultaneous two-process revision-0 barrier proves one successful CAS and one conflict. ADR-117 remains Accepted and is amended to the approved exact V1 key names, 120-code-point names, document selected_profile_id/revision, and intentional omission of updated_at. Focused profile/setup/connection/UI tests, the complete vLLM workflow, full Ruff on new files, scoped Ruff on all touched Python files, focused mypy, formatting of new files, py_compile, and diff checks pass. ADR: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md. Evidence: .superpowers/sdd/2026-09-03-vllm-lab-console-complete-redesign/task-4-report.md.

Fix Round 2 makes file ownership validation mandatory: platforms without a valid effective-UID API now fail closed before profile bytes are read or written. Each mutation revalidates the exact held lock descriptor against the named private regular lock immediately before document load/revision validation and again immediately before the shared atomic writer. Deterministic inode-swap tests cover both boundaries, preserve the document and both lock targets, and prove the moved held inode is unlocked/closed after failure. This relies on ADR-117’s existing private, user-owned data-directory boundary; if an untrusted principal can rename within that parent, userspace leaf checks cannot eliminate the last instruction-level rename window, while atomic replace still does not follow a destination symlink. No ADR, schema, database, UI, or process-lifecycle change. Exact RED/GREEN and static evidence is appended to .superpowers/sdd/2026-09-03-vllm-lab-console-complete-redesign/task-4-report.md.

Fix Round 3 moves ownership capability validation ahead of every filesystem access or mutation. Missing, non-callable, raising, and invalid-result effective-UID hooks now become cause-free generic VllmProfileCorrupt errors; no raw exception or sensitive payload reaches callers. One validated UID is reused transaction-locally for document/lock descriptor ownership and both held-lock identity guards. The 8-case absent/existing-directory matrix proves no document, lock, or data directory is created on failure and a pre-existing directory/sentinel remains unchanged. No ADR/schema/database/UI/lifecycle change. Exact evidence: .superpowers/sdd/2026-09-03-vllm-lab-console-complete-redesign/task-4-report.md.

Fix Round 4 corrects the exception-object contract within ADR-117’s fail-closed storage boundary. Ownership-probe exceptions are converted to an invalid sentinel inside the handler, then the generic VllmProfileCorrupt is raised after the handler exits, so callers receive neither `__cause__` nor `__context__` and no sensitive probe payload is reachable. The absent/existing-parent matrix now walks the complete exception graph while retaining its zero-filesystem-mutation assertions. Ownership validation, descriptor checks, symlink and lock-swap protections, CAS, schema, UI, and process lifecycle are unchanged. Exact RED/GREEN and static evidence: .superpowers/sdd/2026-09-03-vllm-lab-console-complete-redesign/task-4-report.md.

Task 6 Fix Round 2 caps profile documents at 2 MiB before JSON decoding and
rejects duplicate JSON keys recursively at every object level. Oversized,
duplicate-version, duplicate-profile-field, and duplicate-nested documents
fail closed and mutation attempts preserve their exact bytes. Those four tests
were RED before the reader guard and GREEN as `4 passed, 61 deselected` (the
three duplicate mutation canaries also pass after final strengthening). Profile
commits now opt into the shared atomic writer's privacy-safe log mode; real
Loguru success/failure sinks prove no path, raw exception, credential, raw
command, URL, or model value is emitted. The privacy tests were RED as `2
failed, 59 deselected` and GREEN with the preservation control as `3 passed, 58
deselected`. Atomic replace, restrictive mode, held-lock identity, no-follow,
inode revalidation, and CAS ordering are unchanged. No new ADR: ADR-117 already
owns this exact private profile repository boundary.

The final UX fix round projects every profile-backed structured expert value
into its editable Advanced control, so loading a profile can no longer hide
dtype, tensor parallelism, maximum model length, GPU utilization, or remote
code trust behind launch-only state. Profile-name validation and repair
failures now appear beside the profile field and move focus there. Outer Lab
context and the vLLM body refresh Current versus Next after profile selection,
draft mutation, lifecycle settlement, and presentation recomposition without
stealing focus; active-runtime edits retain the immutable current snapshot and
make the checked next-launch draft available to Restart. The new profile-value,
field-adjacent repair, outer-context, recomposition, and mounted restart nodes
were RED before projection changes and GREEN afterward. The V1 device-local
schema, CAS/atomic storage, secret exclusion, and launch-only raw-argument
boundary are unchanged, so ADR-117 remains sufficient.

Final shared qualification for this round: the setup/connection/profile and
mounted workflow/geometry primary passed `308` tests in `428.25s` with no
descriptor-growth warning; the production CSS build/sync/staleness gate passed
`39`; format, critical Ruff, `py_compile`, both profile/diagnostic inventories,
and `git diff --check` passed. No profile schema/storage file or Console/Settings
persistence boundary was changed.

UX Fix Round 2/5 classifies every profile schema/editable-field validation and
projects bounded recovery beside the actual visible source-specific or Advanced
control. Rename duplicate-name and Duplicate capacity failures now travel the
real mounted button -> event -> worker path and focus the profile-name or
profile selector recovery respectively, rather than generic reload copy.
Mounted Hugging Face/local-source focus, existing-mode action, and asynchronous
mutation tests cover the action matrix. The source/mode slice moved from six
RED cases to seven GREEN cases; the later real mounted action regression also
passes. No profile schema, storage, raw-argument, or persistence boundary
changed, so ADR-117 remains sufficient.

UX Fix Round 3/5 reconciles an initial selected-profile load against the exact
bound launch snapshot instead of comparing it only with the new screen's
placeholder draft. Exact live evidence is retained only for the launch-bound
profile/fingerprint; a different restored profile invalidates safely. In
Existing server mode, view and controller guards make selection, create, save,
rename, duplicate, delete, and delayed delete confirmation inert. A mounted
repository regression proves exact profile bytes and revision remain unchanged.
The V1 schema, CAS/atomic storage, and launch-only argument policy are untouched,
so ADR-117 remains sufficient and no storage migration is introduced.
Final Round 3 qualification passed the `60`-case workflow, `71`-case geometry,
and `329`-case five-file primary gates under the normal descriptor limit.

UX Fix Round 4/5 makes selected-profile hydration a persistent, fail-closed
readiness prerequisite. A new screen masks inherited target evidence while the
repository worker is pending, disables the selector and all profile mutations,
and reconciles the loaded profile against the exact launch-bound identity before
revealing READY. Corrupt, future, unavailable, or failed loads clear stale
external candidates, invalidate inherited readiness, preserve independently
truthful Stop, and keep repair/reload guidance adjacent to the disabled profile
selector without focus theft. The delayed success/lifecycle and corrupt failure
paths were RED at their owning seams before the gate and are GREEN in
`test_navigation_to_fresh_models_screen_preserves_exact_ready_handoff` and
`test_fresh_screen_profile_load_failure_invalidates_ready_with_recovery`.

Shared final evidence is `65` workflow, `71` geometry/Tab, and `334` complete
primary tests passing under the normal descriptor limit; compatibility remains
`350/352` with only the two documented untouched Console/Settings baseline
failures. CSS `39`, credential/retention privacy `7`, critical Ruff, scoped
format, `compileall`, both direct inventories, scope review, and diff checks
passed. No profile schema, storage, CSS source, Console/Settings persistence, or
handoff-consumer file changed. ADR-117 remains the governing profile/readiness
boundary; no new ADR or generalized lesson was needed.
UX Fix Round 5/5 changes the child profile prerequisite default from implicitly
ready to explicitly unreconciled. Standalone and lazily mounted children cannot
project inherited READY evidence until the parent completes exact selected-profile
hydration; direct component and geometry fixtures now opt into reconciled state
explicitly. The production-shaped delayed worker test proves the complete
projection remains non-verified before hydration and returns exactly after the
selected profile reconciles. No profile schema, repository, persistence, or
migration changed.

Final evidence: focused post-format regressions `9 passed`; workflow `70 passed`;
geometry/Tab `71 passed`; five-file primary `339 passed` in `447.91s` under the
normal FD limit. Compatibility retained the same two untouched baselines
(`350 passed, 2 failed`). Privacy `7`, CSS `39`, deterministic double build,
bundle sync, critical Ruff, scoped format, `compileall`, both inventories,
scope/diff review, and `git diff --check` passed. ADR-117 remains the governing
profile/readiness boundary; no new ADR or generalized lesson was needed.

Final closure completes the hydration prerequisite at the interaction boundary.
Every child control capable of changing the selected profile, launch draft,
mode, check generation, or external model is disabled before reconciliation;
child emitters and the corresponding `LLMScreen` message handlers reject
programmatic attempts as well. The delayed repository regression exercises a
real mount plus click, press, direct Input/TextArea edits, and forged screen
messages while preserving the exact inherited READY generation and bound
target; successful profile reconciliation restores readiness and Use. The
pre-fix run exposed `19` enabled controls and attempted preflight, and the final
focused closure set passed `5`.

Final qualification passed workflow `74`, geometry/Tab `71`, and the complete
five-file primary `343` in `446.64s`. Compatibility remains the exact unchanged
`350/352` baseline; privacy `7`, CSS `39`, deterministic double build and bundle
sync, critical Ruff, scoped format, `compileall`, both direct inventories,
scope review, and `git diff --check` passed. No profile schema, repository,
persistence, CSS, Chat, Console, Settings, or handoff-consumer file changed.
ADR-117 remains the governing profile/readiness boundary; no new ADR or lesson
was required.
Final combined-review hardening validates the shared bind-address rule at every
profile mutation while deliberately allowing bounded legacy V1 documents to
load into the existing source-aware repair projection. Invalid saved binds
cannot launch or trigger runtime/network probes, and the next attempted write
revalidates every profile fail-closed. Persistence tests cover empty, URL,
host:port, path, whitespace/invalid host, oversized values, plus IPv4, IPv6,
and localhost success. This is direct ADR-117 enforcement; no new ADR or schema
change is required.

Final repair-selection closure keeps an invalid non-selected legacy profile as
an ephemeral UI target instead of attempting to persist its selection. The
durable selected ID, revision, and bytes remain unchanged while the invalid
profile is projected beside its field-level repair copy; only correcting and
saving that target or deleting it can cross the repository boundary. Every
other mutation and lifecycle check remains blocked by the existing all-profile
validation rule. A corrected save atomically validates the full document and
may then select the repaired profile, while deletion preserves the prior valid
selection. Reopening restores durable selection truth because repair selection
is never stored. This is direct hardening under ADR-117; no new ADR, schema, or
generalized lesson is required.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-31219. The Task 6 collision correction moved
the full dependent vLLM sequence into one collision-free monotonic block so no
task depends on a future/higher task id. It therefore moved to TASK-31267 after
TASK-31264 preflight, TASK-31265 readiness, and TASK-31266 Console handoff. The
record was originally added by
`ffc4f9d8f8343169097dcac40d3ba4ed0a2177c0`.

A second merge-time sweep found that `origin/dev` had advanced to
`1a1b5c19e0bb3243effb1ae9671158b6670ad6da` and now canonically claimed the
intermediate TASK-31263 and TASK-31264 IDs for unrelated theme follow-up work.
The complete vLLM sequence therefore moved together from TASK-31263..31268 to
the next contiguous block proven free across every fetched non-vLLM ref,
TASK-31282..31287. This profiles task maps TASK-31267 -> TASK-31286; ADR-117
remained collision-free.

A third merge-time sweep found that `origin/dev`
`24d931d0a4f6beec3e0fd7e94d24850ca196e86c` had made the unrelated theme
TASK-31282..31284 claims canonical. Across every fetched non-vLLM local and
remote ref, TASK-31386 was the numeric maximum and TASK-31387..31392 were the
first six contiguous IDs strictly above it. The complete vLLM chain therefore
moved together from TASK-31282..31287 to TASK-31387..31392; this profiles task
maps TASK-31286 -> TASK-31391. ADR-117 remained collision-free across the same
refs.
