# PR 2504 Rebase and Review Follow-up

**Goal:** Rebase the complete reviewed audio integration onto current dev, address every PR finding and failing automatic check, update the same PR, and leave it open for the user's later merge decision.

**Architecture:** Preserve ADR-094 Console custody, ADR-097 trace privacy/source ownership and ADR-098 isolated speculative voice. This is maintenance of approved behavior, not renewed audio qualification.

ADR required: no new ADR
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md` (with existing ADR-094 and ADR-097)
Reason: Behavior-preserving integration, targeted admission/boot defects, and declared portable vendor build corrections within existing contracts.

## Final relevant dev refresh — 2026-09-09

Dev advanced from `5655c4820733d24754f872c21cc6196d83bf1504` to
`2e3389e694e93592a1c66e5c3416bf29a1057d6c` with PR #2534's official Kokoro
runtime and exact-owner speech cancellation fixes. This is directly relevant to
the user's complete-audio/latest-dev integration request. Preserve those incoming
changes rather than pinning publication before them. The earlier dev checkpoints
below remain historical evidence, not a claim about this final base.

ADR required: no new ADR
ADR path: existing ADR-098, ADR-139 and incoming
`backlog/decisions/140-official-kokoro-pytorch-runtime.md`
Reason: mechanically compose accepted upstream runtime and lifecycle contracts;
do not redesign them or repeat their native/model/playback qualification.

1. Review the immutable incoming delta and its shared paths against the completed
   local fixes. Root reads the new ADR/tasks/testing lessons; an independent
   source review selects fake-only TTS integration tests before any execution.
2. Preserve a named backup of the clean reviewed local checkpoint. Rebase onto
   exact dev `2e3389e6`, retaining every upstream and feature code change. Resolve
   documentation composition explicitly. For the generated diagnostic inventory,
   inspect actual statement changes before using the official generator.
3. Prove incoming/replayed source equivalence, including every shared path, and
   retain schema 69/70 bytes, voice 71, native pins and hard-off resources. Run the
   selected fake TTS/route checks and final startup guards without changing any
   budget. Exclude the new Kokoro runtime test module: collection imports Torch
   and evaluates native MPS availability.
4. Complete separate spec and quality reviews of the frozen startup correction,
   then final source/version/inventory checks. Update the existing PR using an
   explicit lease against observed published head `ab4bd317470efef88478e297c432759ee6efe9e9`.
   Preserve the fresh generated PR-description block; observe normal automatic
   PR CI only. Leave the PR OPEN, UNMERGED, with auto-merge disabled.

## Global Constraints

- Work only in `.worktrees/speculative-duplex-voice-dev`, branch `codex/speculative-duplex-voice-dev`. Preserve main and the original feature worktree, including all unrelated edits.
- Published PR head is `47de10b893f4be42cdef7931e3ddecef33c6c539`; prior integrated dev is `37bf45fb6232a1d4fb50fdba8f3c19c856ae7664`; fetched dev is `7e81ed55db66ace04cb3dd1f8feb6d40a21f6f48`.
- Dev subsequently advanced to `565dc499210491def757bed325c4014e324dc470`. The reviewed commits were cleanly rebased onto it at `8b928e2c40ea74fbe53995969f3dd503bfb05ed5`; the resulting tree exactly matches the reviewed work plus the eight upstream TTS/setup paths. Four focused fake-adapter/version tests passed. Both pre-rebase checkpoints remain backed up locally.
- Keep a local backup of the published head. Rebase the exact reviewed net tree, including load-bearing merge resolutions; do not drop features by replaying only non-merge commits. Preserve every newer dev feature, schema migration, trace guard, privacy rule and TTS contract.
- Keep 700 ms default silence, incremental STT, same-turn interruption, distinct post-playback turns, sequential cancellable TTS, and fail-closed AEC. Do not relax any timing, capacity, custody, safety or resource limit.
- Packaged platform qualification and build identity bytes remain unchanged and unqualified. No release bypass, release publishing, or companion version change.
- Use existing main `.venv` and read-only cached dependencies only. No installs or user configuration changes.
- Targeted software tests only. No full suite, live app/audio, physical qualification, route checks, providers/models, repetitions, soaks, local native compilation/loading/callback/GIL tests or installed-wheel tests. File-only native provenance checks are allowed. Only normal automatic PR CI may build/test hosted wheels.
- Never merge the PR. Push only its existing branch with an explicit lease against the observed remote head. Address review threads with evidence; do not implement incorrect architectural suggestions.

## Task 1: Rebase the reviewed net tree and compose schema 70

**Owner/files:** One implementer owns branch/rebase operations and only conflict resolutions, schema renumbering/tests/census, generated diagnostic inventory, this plan, and relevant task notes. No other code writer runs concurrently.

1. Verify clean tracked state, exact published head, dev head, linked-worktree identity, and absent/non-conflicting backup ref. Create `codex/speculative-duplex-voice-pr2504-pre-rebase-47de10b893` pointing to the published head.
2. Consolidate the exact published tree into one commit parented by prior integrated dev using `git commit-tree`; prove its tree equals the published tree. CAS-update only the integration branch from the exact published head, then perform a real `git rebase --onto` fetched dev from the old base. No reset/checkout cleanup. The untracked new plan is controller-owned until this task explicitly commits it.
3. Expected conflicts are DB module, Canvas migration test and diagnostic inventory. Preserve dev's schema-69 source-pin migration byte-for-byte; move the voice provenance extension from 68-to-69 to 69-to-70, including method/map/version/tests/census references. Preserve source-pin and provenance guards together. Resolve shared code structurally, never replace it with an older whole-file snapshot.
4. Record a meaningful failing migration composition test before the fix when possible; verify a real schema-69 predecessor upgrades to 70 and retains dev source-pin behavior plus voice provenance/index guards. Use temporary SQLite only. Run focused migration/trace tests and the official diagnostic generator/check, not the full suite.
5. Compare rebased tree against read-only combined-tree preview `595b87974a5275e709ce1787f89944803843db28`, explicitly accounting for conflict/schema/generated changes. Verify dev ancestor, original feature head/diff hash, packaged authority hashes, and no unrelated paths staged.
6. Commit the resolution/plan/brief task notes as appropriate; report exact commit graph, files, commands/results/warnings and limitations to this plan's ignored `task-1-report.md`. Do not push or answer PR threads. Existing TASK-23175 AC19 covers preserving complete integration onto dev; retain In Progress.

## Task 2: Correct validated review findings

**Owner/files:** `UI/Console_Modules/wiring.py`, targeted UI admission tests, `Chat/console_speculative_voice_session.py`, `Audio/rolling_transcript.py`, optional explanatory `Audio/aec_backend.py` docstring, TASK-23175 follow-up notes, and the stale schema label at `scripts/index_plan_pin_census.tsv:15` found by Task 1 review.

1. Before implementation add a follow-up acceptance criterion/plan to TASK-23175 for rebased PR fixes and open-PR handoff, without marking broader qualification complete.
2. Add a regression for a deleted session during synchronous send admission: missing owner must give the established closed-session refusal while preserving the captured draft and attachments. Add a control proving unrelated internal `KeyError` for a valid owner is not hidden. Demonstrate RED before narrowly normalizing only a proven missing owner in the shared admission boundary; do not blanket-catch `KeyError` in the UI.
3. Move `FrozenTracePolicy` into the local import group. Add Google-style Args/Returns for material transcript comparison without changing normalization behavior. Optionally explain why the native child's lazy AEC import intentionally avoids the app-side optional-dependency registry under ADR-098.
4. Do not add a second UI-thread hop: existing process delivery runs on the UI loop. Do not import the app optional-dependency graph into the isolated child. Do not replace the bounded stdlib wire validator with Pydantic. Cite the independent report for these three rejected suggestions.
5. Run the new regression/control plus exact adjacent draft refusal, rolling comparison, lazy AEC and UI-thread identity tests, excluding all real-native nodes. Apply scoped lint/diff checks, commit explicit paths, write `task-2-report.md` with RED/GREEN evidence.
6. Correct the census's comment-only schema label from 69 to 70; the index/test entries already reference v70 and must remain unchanged.

## Task 3: Restore lazy boot without weakening performance guards

**Owner/files:** Voice-specific hot imports in Console UI/runtime/store/persistence/provider/trace modules, lazy voice-preview composition if needed, boot import/worker tests, and relevant documentation.

1. Reproduce the startup import regression using the existing software-only Perf Guard command/tests. Latest dev already uses 973/973 allowed modules. Preserve that budget and required worker identity; never refresh or raise the ratchet to accommodate eager voice imports.
2. Defer native stream, voice controls/settings, dispatch/promotion and trace voice contracts until first voice use; type-only imports belong under TYPE_CHECKING. Preserve synchronous winning claims and constructor/disposal semantics. Confirm every duplicate hot edge is removed.
3. Prefer lazy creation of the initially empty voice preview on first relevant use, preserving projection ordering and cleanup. Do not move definitions into a hot module solely to evade the module counter. Investigate actual widget lifecycle before choosing the smallest correct seam.
4. Replace a blind fixed 1-second wait in the startup worker anti-vacuity test with bounded condition waiting for the same required worker if its timing defect remains after rebase. Do not delete/weaken its assertion or lengthen a blind sleep.
5. Verify boot guard plus focused first-voice activation, actual factory/fake-child, preview and synchronous custody behavior with no hardware/providers. Commit only explicit changed paths; report RED/GREEN, exact startup module count and warnings in `task-3-report.md`.

The constructor audit also required lazy creation behind the existing gateway registry property and an existing-owner-only read in `app.py::_confirm_and_quit`. First creation during pending session close inherits the exact existing close fence before returning; unused close/quit never constructs an owner, and existing permits remain mandatory. Task 3 review approved these preservation details. Final reported census is 972/973 with all nine voice modules absent; all 20 distinct Perf Guard nodes passed across the documented failure-resuming runs, with focused first-use and close/quit coverage.

## Task 4: Correct hosted native wheel portability and paths

**Owner/files:** `.github/workflows/voice-aec-wheels.yml` (resolve exact current filename), `native/voice_aec/.gitattributes`, vendor patch tooling/metadata and the single affected upstream header, related file-only packaging regressions.

1. Read existing declared patch and pristine-verification mechanism before editing. Add `<stddef.h>` for global `size_t` in the affected WebRTC clock-drift header, matching adjacent AEC3 headers, as an explicit second pinned patch; update patch series, patch hashes, PATCHES metadata and current closure manifest consistently. Preserve pristine original bytes/hashes and reverse-patch verification. No hidden source rewrite or relaxed integrity check.
2. Add a file-only regression proving the patch/manifest closure is valid and the new include exists for this compile error. Do not compile, load or test native code locally.
3. Enforce LF checkout for the native pinned source/manifest/patch scope through existing nested `.gitattributes`, not repository-wide line-ending changes. Verify a temporary Git checkout with `core.autocrlf=true` preserves closure bytes using file-only checks.
4. Fix cibuildwheel `{project}` test paths to repository-root-relative `native/voice_aec/tests/test_binding.py` and `Tests/Packaging/test_voice_aec_installed_wheel.py`, including any duplicate configuration. Test the command/path contract without executing installed-wheel test scripts locally.
5. Run only file-level verifier/packaging regressions and static checks. Preserve exact companion/app version 0.2.0, qualification/build-identity JSON and release controls. Commit explicit files; report all local limits and required automatic hosted CI evidence in `task-4-report.md`.
6. Final combined review identified that the preserved schema-69 source-pin SQL
   dependency was missing from the exact source inventory. Add that one existing
   migration path and a required-scope regression after demonstrating RED; do not
   change its SQL or broaden inventory roots. Recheck only the affected file-level
   inventory guard, then obtain the same reviewer's focused fix verification.
7. Automatic Windows CI at `e44bad71fb` passes checkout hashes but rejects
   `api/audio/echo_control.h` during reverse-pristine verification. Diagnose the
   temporary patch stage with file-only Git operations, including inherited
   line-ending configuration. Demonstrate a focused failure before correcting
   only the patch-operation boundary and its regression if proven. Preserve all
   pinned bytes and strict raw-byte hashing; never normalize hashes or accept
   changed pristine content. Hosted CI, not local native execution, proves the fix.
8. The same automatic run reaches a later GCC error in
   `reverb_model_estimator.h`: `std::unique_ptr` is used without `<memory>`.
   Add the direct include as a third independently pinned patch, retaining the
   first two patch hashes and the unchanged pristine closure. Update only its
   declared patch metadata, affected current hashes, exact inventory, and
   file-only regressions/documentation. Reuse the verified reverse-to-pristine
   staging procedure; no local compilation, integrity exception or runtime
   behavior change. Review this correction with the Windows residual before
   another normal PR CI run from the pushed fix.
9. Hosted macOS ARM builds and installed tests pass for all three Python versions;
   the subsequent archive checker rejects legitimate trailing-slash directory
   records produced by wheel repair. Add synthetic-archive RED/GREEN coverage and
   accept only benign normalized directory metadata while retaining malformed
   path, duplicate-member, license/provenance and shared-library rejection. No
   local wheel build or extension load is needed. Diagnose the separate Intel
   unresolved CPU-feature symbol from pinned upstream sources before planning
   any source-closure correction.
10. The Intel diagnosis confirms the exact pinned upstream
    `system_wrappers/source/cpu_features.cc` is absent; its selected API target
    supplies only declarations. Add this one implementation to the existing
    explicit compile-source allowlist, not the broad GN umbrella target. Verify
    its official Git blob `ebcb48c15fb20ddeda6c5844e097d7b2835cbd81` and SHA-256
    `e4bac0600ca4a36436431db0e1377886c98a5362eb2a403a40c83dc53b85f643` before import.
    Preserve every old 315-entry pristine line and all three patches; update the
    exact closure to 316 and independently reviewed pristine-manifest anchor to
    `596ddbb3291fc5fd432376ef4bdfee6fed67bd999709638d68436f2b1ef041a4`. Regenerate
    only the affected metadata/current manifest through the verified staged
    transition, add exact source inventory/regression coverage and update current
    source-closure documentation. Existing ADR-098 records this omitted-support-
    source repair before implementation; no new runtime boundary or dependency
    version is chosen. No local native build/load or release qualification.
11. Automatic run `34300791333` at rebased `d21e930658` passes the old failure
    boundaries but exposes later missing resampler/Abseil symbols and Windows
    socket-header collisions. Complete the read-only pinned-source and Windows
    include-order diagnoses before implementation. Add only the exact omitted
    x86 resampler source plus the minimal Abseil optional-access/logging include
    closure; preserve both upstream exception branches rather than requiring a
    compiler exception mode. Keep x86-only resampler compilation excluded on
    ARM64 through the existing architecture filter. Independently verify all new
    source blobs and hashes, preserve all 316 old pristine entries and three
    patch pins, and record the reviewed new closure/subset anchors in ADR-098
    before import. Address the proven Windows include collision with one
    Windows-only `WIN32_LEAN_AND_MEAN` definition in the existing provenance
    mapping, not an upstream patch or global compiler setting. Use meaningful
    file-only RED/GREEN guards for source identity, architecture selection and
    exact compiler-definition agreement; strict regenerated metadata, legal
    closure, inventory and tamper checks must remain enforced. One worker owns
    the combined native changes, followed by focused independent review and
    another normal automatic PR run. No local CMake/configure/compiler/native
    load, manual workflow, release qualification or merge is authorized.
    The completed diagnosis identifies exactly six added paths: Sinc SSE,
    `bad_optional_access.cc`, `raw_logging.cc`, `raw_logging.h`, `atomic_hook.h`
    and `log_severity.h`. The independently checked 322-entry pristine anchor is
    `fc832ca362423a49a79752e139be919c05456356529b8edb56c87d5993e6d156`; generated
    Abseil subset tree becomes `305085097eb6e5f3fe48baa59519a7faaa62eedd` while
    the complete upstream subtree pin stays unchanged. Existing old-315 and
    CPU-addition controls remain, and removing precisely these six additions
    must recover the old-316 ledger byte-for-byte. No whole GN target import.
12. Run `34302675575` at `4effec1421` compiles Windows sources successfully but
    fails linking the existing `time_utils.cc` reference to `timeGetTime`.
    The pinned upstream timeutils target declares `winmm.lib`; restore this
    existing platform dependency with a WIN32-only
    `target_link_libraries(webrtc_aec3 PUBLIC winmm)` so the static library's
    consumer receives it. Add a focused file-only RED/GREEN guard for exact
    Windows-only transitive target linkage, not compiler execution. Keep all
    source, patch, definition, legal, qualification and version bytes unchanged.
    The existing repair tool treats Winmm as a Windows system library; do not
    bundle a DLL or relax archive checks. Existing ADR-098 covers the platform
    boundary; this restores the upstream system link, not a new runtime choice.
    Obtain a focused review, then use only normal automatic PR CI for native
    link/import proof. No local build/load, manual workflow or merge.
13. Run `34303873079` at `f063035853` passes all three Windows wheel builds,
    bindings and installed corpus tests, then correctly rejects a private
    `msvcp140` DLL bundled by repair. CMake's default dynamic CRT choice did not
    implement the existing single-static-extension packaging policy. After the
    read-only runtime and allocation-boundary audits, set MSVC_RUNTIME_LIBRARY
    consistently on both `webrtc_aec3` and `_native`, inside an MSVC-only block
    after both targets exist: `MultiThreaded$<$<CONFIG:Debug>:Debug>` (Release
    /MT, Debug /MTd). Use the existing CMake property, not global flags, a DLL
    allowlist/exclusion or a redistributable prerequisite. Keep all source,
    patch, definition, ownership, exception, SIMD and archive-policy bytes.
    ADR required: no new ADR; amend existing ADR-098 to make its static packaging
    policy's runtime and servicing consequences explicit. Separate CRT instances
    must not exchange allocation ownership; the current bytes/dict/C callback
    boundary preserves that rule. Debug CRT builds are not release payloads;
    statically linked runtime security updates require rebuilding/requalification.
    Add one file-only RED/GREEN guard for MSVC scope, both target names, property
    value and placement, with only adjacent linkage/platform controls. Root also
    corrects the release guide's stale empty-patch wording. Focused independent
    review and normal automatic CI must prove native imports and unchanged
    archive validation. No local native/CMake/full-suite or manual CI/merge.

## Final root-owned handoff

### Round 6 idle scheduler follow-up

**Superseded before publication:** dev `733f7a628c` merged its independently
reviewed scheduler-start-after-readiness fix (`a7ef7addb8`) while this candidate
was being verified. Preserve that upstream lifecycle correction. The redundant
four-line idle optimization and its tests were removed using their exact local
diff; the patch, RED/GREEN and approved review remain in current-plan scratch.
The steps below record the attempted repair, not code intended for this PR.

Automatic Perf Guard at `ab4bd31747` measures 974/973 modules at the exact
UI-ready flag: the immediate scheduler's first tick can load heartbeat and
emergency-stop support before that flag. The same tree measured 972 locally;
the first-use imports are genuinely ordering-sensitive, not a census error.

ADR required: no new ADR
ADR path: `backlog/decisions/097-boot-budget-ratchets.md`
Reason: Remove an unnecessary idle-queue disk read within existing scheduling
safety/liveness contracts (TASK-26004, TASK-26025, TASK-31507); do not defer
time-sensitive scheduler startup or its first heartbeat.

1. Root is the sole writer for `Scheduling/scheduler/loop.py`, focused scheduler
   regression tests and these follow-up notes. The independent reviewer is
   read-only. Existing TASK-23175 PR-check/boot acceptance covers this repair.
2. Demonstrate RED with an actual empty `PriorityQueue`: a tick must not call
   the emergency-stop reader when no dispatch exists, but must persist its
   normal heartbeat. Add empty-to-nonempty controls proving stopped/unreadable
   state holds the original queue, then clearing it permits normal dispatch.
3. Add only a synchronous empty-queue return before the existing stop read.
   Preserve stop-before-pop on every nonempty queue, off-loop file access,
   immediate scheduler start, heartbeat/cancellation behavior and all ratchets.
4. Run exact new/adjacent software tests and the existing UI-ready/worker guards,
   with temporary files/fake handlers only. No snapshot refresh, full sweep,
   live audio, native execution or manual CI. Obtain focused review before
   committing/pushing; leave the same PR open and unmerged.
5. Keep this scheduler-only maintenance outside the voice qualification source
   and lint inventories: it changes neither voice runtime composition nor its
   evidence/qualification tooling. Verify both changed Python files explicitly
   and record their exact PR commit; do not claim the unchanged voice digest
   alone identifies this boot fix. No packaged authority is regenerated.

### Final integration onto Buddy-enabled dev 733f7a628c

ADR required: no new ADR
ADR paths: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`,
`backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md`,
`backlog/decisions/097-boot-budget-ratchets.md`
Reason: Preserve both already-approved ownership contracts and chronological
schema migrations while integrating current dev; no new runtime or authority.

1. Keep backup `codex/voice-pr2504-before-dev733-rebase` of the exact local
   pre-rebase head. One implementer owns rebase/conflict resolution, voice
   migration renumbering and regression/inventory updates. Root/reviewer are
   read-only except separate ignored reports. No stash/reset/checkout cleanup.
2. Rebase the complete reviewed series from dev c4a onto exact fetched
   `733f7a628c064005d27bed8bbcf53aae01b5dd45`. Preserve Buddy-owned schema70
   SQL and source-pin69 bytes unchanged; move voice provenance70 to71 (70→71),
   including method/map/test/census/source-inventory references. Keep migration
   guards, rollback, both indexes and every trace/custody invariant. Do not
   replay pre-Buddy versions of shared Console/runtime/DB code wholesale.
3. Preserve newer Buddy conversation/defaults/decision/speech ownership, dev's
   scheduler lifecycle fix and SQL allowlist/index changes alongside lazy voice
   startup and shutdown custody. Review exact overlapping hunks; no unrelated
   Buddy redesign or repeat of already-reviewed native repairs.
4. Demonstrate a real schema70 Buddy predecessor upgrades to71 retaining Buddy
   and source-pin tables/indexes plus voice provenance; run focused migration,
   ownership/lazy-voice and incoming scheduler tests only. Keep protected dirty
   feature/main worktrees and all evidence untouched.
5. Regenerate shared diagnostics only through its official generator after
   reviewing changed statement ownership; reconcile exact index/source/lint
   inventories. No authority regeneration, full sweep, live/native/local build,
   models/providers, soak, manual workflow, release or merge.
6. Freeze the integration for independent shared-hunk/schema/boot review, then
   publish the same PR with observed-head lease and monitor its normal CI. This
   exact dev is the final publication target; later independent dev commits are
   reported as drift instead of silently restarting another integration cycle.

Integration review refinement (before implementation): the incoming Buddy's
per-kind cards and host clocks do not yet consume the existing mixed-type
approval/install/script ledger. Compose them through the current typed projection
and exact successfully rendered Buddy owner/head claim, preserving the ordinary
Console active-session guard and the union of independently visible views. Keep
one allowance/expiry authority, finishing-head ordering, strict original resolvers,
and question/worktree host behavior. Call controller projection/clock methods only
outside the aliased non-reentrant host lock. Scope includes the Buddy coordinator,
modal, shared controller/host, focused real-controller mounted/fake-clock tests and
their exact source/lint closure. Existing ADR-139 records this implementation
clarification; no new permission, ledger, lifetime service or ADR is introduced.
Also retain the synchronous voice shutdown fence before the incoming receipt-owner
await, with one software regression. Correct only scheduler timing documentation
in `Utils/boot_worker_policy.py`: IMMEDIATE bypasses the stagger gate but the
scheduler starts after UI readiness; enum, identities, limits and runtime stay put.
The public direct-announcer seam must also route the three typed kinds through
their existing live stable-ID sanitized announcer, preserving exact-once delivery
and failed-delivery retry. Legacy host scanning must not duplicate those notices;
question/worktree retain the incoming notice path. Cover the direct and scanning
entries with the existing real-round privacy regression before changing the seam.
Mechanical lint completion removes only the now-unused `use_human_input_wait`
import and the unused result binding of the unchanged `host.run_round` call;
existing delayed-import E402 debt remains outside this integration scope.

Post-freeze spec review correction (before implementation): fence a worker's
rendered-claim snapshot against completed Console/Buddy claim changes before
either pausing or starting the existing typed allowance. Prove the exact
expiry/suspend interleaving with a deterministic barrier, keeping callbacks
outside the shared non-reentrant lock and introducing no second ledger. Restore
host-scan first delivery and retry for a typed decision initially rendered then
hidden: dispatch exact live stable IDs through the existing sanitized announcer
outside the lock; retain question/worktree behavior. Scope is controller/host,
focused Buddy/headless tests and these notes. Existing ADR ownership is unchanged.
The deterministic expiry/suspend RED observed `active_since=101.0` after release;
the visible-to-hidden notice RED observed zero delivery attempts for all three
kinds, including retry cases. Shared claim revision fencing and exact typed scan
dispatch pass the final 21-case focused gate (3 existing dependency warnings,
9.03s), with scoped lint/format/diff checks. Startup diagnostics remain root-owned;
this gate is not a boot census, native run or full-suite result.

Conditional startup-admission diagnosis scope (before tests): use a hand-built
screen to invoke the real post-reconciliation starter, captured transcript timer
callback and sync/coalescer through an awaited dependency barrier. Prove idle
timer admission and its actual overlap-to-worker mechanism without setting the
coalesced-request flag by hand. Preserve independently active viewed work,
other-session in-flight work, wake delivery and pending review publication, plus
timer idempotence. A fix, if separately authorized after evidence, is limited to
initial polling admission; no budget/allowlist/coalescer or lifecycle exception
change. Existing boot/lifecycle ADR boundaries apply; no new ADR is required.
The uncaptured early-quit MountError remains a separate root-owned diagnosis.
Root subsequently captured the early-quit/shutdown failure at the deferred legacy
workspace-alias worker's mount on an unattached tray (mounted/running flags still
true). Before that repair, add exact detached-tray/sibling and post-await owner
change regressions plus live error/success controls. Guard the captured tray's
attachment and screen lifecycle immediately before mount and revalidate current
tray identity after its await before allocator admission. No broad MountError
suppression or global recomposition change. Both startup repairs are authorized;
all focused new cases remain in the existing runtime-ownership test file.
Both source-seam repairs passed 15 selected cases (zero skips, 3 existing warnings,
7.87s), including real detached-mount RED and successful mounted alias controls.
Root then ran the unchanged no-plugin six-case actual startup gate against frozen
source: 6 passed, zero skips, 3 warnings, 54.34s. UI-ready census is 972/973;
slow-mount scheduler, quit/shutdown/setup-failure and worker census pass without
budget or allowlist changes. Protected original feature worktree remains unchanged.

Policy review disposition before fixture adaptation: ADR-094 already retains
approval/install/script decisions through no-view runtime domain routers; the
incoming blanket None-setter/immediate-refusal assumption is obsolete for those
three kinds. Preserve ADR-139's question-specific no-view refusal and exact Buddy
retention exception. Adapt only the incoming skill fixture to a worker-owned
request, exact pending identity, unchanged finite allowance while hidden, and
explicit bounded cancellation before the existing target close/reopen assertions.
Do not add a wake-origin discriminator or remove stable admission proxies. The
independent immutable-source review is recorded in current-plan scratch as
`dev733-headless-policy-review.md`; ADR-139 makes the per-kind policy explicit.

Final dev733 local integration checkpoint: the real 19-commit rebase completed at
`fb503c9feddf7964d33d2e16629c3b0c8ee13710`, preserving the requested pre-rebase
backup. Genuine schema-70 predecessor RED/GREEN establishes schema 71 after the
unchanged source-pin 69 and Buddy 70 migrations. The composed Buddy typed ledger
and quit-fence frozen-source gate passed 20 cases; schema/inventory checks passed
9; direct-announcer/privacy/retry checks passed 5. Root's separate pre-inspected
TTS/receipt/Canvas joined gate passed 19 cases. Known dependency warnings remain;
these are targeted software results, not new native or hosted-CI evidence.
The official diagnostic inventory has no drift (594 owners, 1384 TASK-492 calls,
55 TASK-31551 calls, 7684 TASK-494 calls, 12 sink files). Exact evidence and the
final immutable commit belong in current-plan `dev733-rebase-report.md`.

After scoped reviews, run one proportionate joined software gate covering changed seams and obtain a final read-only combined review. Refresh dev/PR/comments and account for any changes. Push the same branch with explicit force-with-lease, reply to each inline finding with the fix or grounded explanation, and resolve only addressed threads. Observe normal automatic checks; diagnose failures before more tests or changes. Record exact final head/check status and leave PR 2504 OPEN and UNMERGED. Preserve worktrees and recoverable evidence.

### Publication-blocking dev5655 refresh

ADR required: no new ADR.
ADR path: existing ADR-098 and ADR-139; boot ADR-097 remains applicable.
Reason: mechanical integration of reviewed Library changes and regeneration of
derived diagnostics; no new schema, permission, runtime or voice behavior.

After bridge checkpoint `efdb8e605cf2fabba84233b59a01caa0868baad8`, the
read-only three-tree comparison against newly fetched dev
`5655c4820733d24754f872c21cc6196d83bf1504` proves two conflicts in generated
diagnostic metadata (app entry and aggregate), which block ordinary PR CI.
This concrete blocker narrowly supersedes the earlier dev733 cutoff; it is not
permission to repeat feature qualification or chase non-blocking later drift.

1. Preserve a named backup of this clean reviewed series before rebasing all
   commits from dev733 onto exact dev5655. Read-only review of all five shared
   paths found no voice, schema or runtime-ownership conflict.
2. Resolve only generated diagnostic conflicts from actual combined source:
   inspect the app statement delta, regenerate with the official checker, and
   require zero drift. Do not copy either branch's aggregate blindly.
3. Prove every other incoming byte equals the replay delta and that the frozen
   Buddy bridge, native tree, source-pin/Buddy/voice migrations and hard-off
   authority are unchanged. Keep the independent bridge review on its immutable
   efdb checkpoint while performing this mechanical replay.
4. Run five exact safe Library/DB/source checks recorded in the current-plan
   drift review, plus six actual startup/census cases (slow mount, three early
   exits, UI-ready and worker census). The eight prior pure policy cases are
   not a substitute for these real startup guards. No budget/snapshot increase.
5. Complete immutable bridge specification and quality review, fresh clean
   source/version/static checks, then update the same PR with exact lease
   protection. Observe only normal automatic CI and leave OPEN/UNMERGED.

## Local verification checkpoint — 2026-09-08

The final documentation-only dev update `03d1b253c8de9d22408ae93845937ae89c893fcb`
was integrated by a clean four-commit rebase. The resulting implementation head
is `e1f02fc98de3838d8d6a52db13936d8719a1f5fb`; its only delta from the preceding
reviewed tree is the 27 upstream Library documentation/Backlog additions.
Both earlier refresh checkpoints and the original published head remain backed up.

All four scoped reviews approved their changes. Task 4 adds a second independently
hash-pinned `<stddef.h>` patch and native-only LF checkout attributes; its full
verifier reverses both patches against all 315 immutable pristine source hashes.
The Windows-style checkout test includes an actual CRLF control and loads the
verifier from that checkout. Both cibuildwheel test paths now resolve from the
repository root. Source/lint inventories retain the schema-70 paths and lazy-owner
test; no inventory roots, runtime limits, versions or release controls were relaxed.

Fresh root verification at that implementation head:

- Joined changed-seam gate: **41 passed**, zero skips, 26.22 seconds. Covers
  schema-69 source pins plus schema-70 voice provenance, deleted-owner draft and
  attachment refusal, lazy owners/preview, actual visible factory with a fake
  child and all four exit modes, hard-off legacy selection, sealed trace import,
  unused quit, file-only native checkout/workflow/inventory checks, and fake TTS.
- Boot gate: **17 passed**, zero skips, 37.37 seconds. Measured own imports
  **646/660**, UI-ready **972/973**, and CSS **803655/804000**. All nine voice
  modules remain cold; budgets, snapshots and worker identities are unchanged.
- Ruff passed 23 follow-up Python paths (excluding unchanged boot-import debt in
  `app.py`); eight scoped formatting checks passed. Index census remains
  **293 declarations / 293 rows / 78 pinned**. Strict source digest is
  `867029fee6a25da2d3eeceb73d382089307a4c7421d70c60cd7a3640b1d78f70`.

The joined run has three existing dependency warnings; the boot run has seven
warnings including three intentional budget reports and an existing datetime
deprecation. Neither result is a full-suite or pristine-warning claim. Exact
selectors, intermediate RED/GREEN results and review reports remain under
`.superpowers/sdd/2026-09-08-voice-pr-2504-followup/`. Historical counts above
retain their original checkpoints and are not summed with these final runs.

The original feature HEAD, its binary diff and all 19 unrelated dirty paths are
unchanged. Packaged build identity and all-platform hard-off qualification bytes
retain their original hashes. No local native compilation/loading, live audio,
provider/model request, hardware qualification, repetition, soak, installation or
full sweep occurred. Hosted native build/installed-wheel proof remains for normal
automatic PR CI. Final combined review, push and remote handoff follow this local
checkpoint; merging remains explicitly prohibited.

The final combined review found one source-identity gap and no other blocking
issues. Commit `4eca1a38a6e2674a14ddc1bcbff195fd2d9f3e75` adds dev's preserved
schema-69 source-pin SQL to the exact inventory and its required-scope regression
(two files, two lines). The guard failed for that missing path before the fix and
then passed in 0.95 seconds; scoped lint/format passed. This does not modify SQL,
runtime behavior or packaged authority. The earlier source digest belongs only to
its stated implementation checkpoint; the inventory and task-note changes require
a new clean committed digest at publication.

The final reviewer's focused re-review approved `4eca1a38a6` and closed the sole
P2 with no remaining findings. No tests were repeated for review. The publication
step now updates the existing PR and observes automatic CI; no merge is authorized.

## First automatic CI follow-up — 2026-09-08

Published `e44bad71fb` left all six original review threads resolved and auto-merge
disabled. All non-native automatic checks passed. Native run `34297893189`
exposed four later boundaries: temporary patch reversal inherited Windows global
autocrlf; GCC required a direct reverb-model `<memory>` include; repaired macOS
ARM wheels carried legitimate directory records; Intel imports lacked the pinned
CPU feature implementation. Hosted ARM builds and installed tests passed for all
three Python versions before archive validation failed; this is not whole-job or
release qualification evidence.

The four minimal corrections are committed through `3d0f1878c9` with focused
RED/GREEN evidence: seven inherited-line-ending cases, 19 three-patch provenance
cases, 34 synthetic-wheel directory/security cases and 22 final CPU-closure,
tamper, inventory and synthetic-sdist cases. These overlapping cohorts are not
summed. Scoped lint/format, full file-only vendor validation and exact version lock
pass. The CPU addition preserves every original 315 pristine entry and all three
patch hashes; ADR-098 records the reviewed 316-file anchor. Build identity and
all-platform hard-off qualification bytes remain unchanged.

No local native build/load, installed-wheel execution, app/audio, provider/model,
physical test, soak or full suite ran. A focused combined review, latest-dev
refresh and another normal automatic PR run must validate the resulting update;
hosted linkage/platform success is not yet claimed. The PR stays unmerged.

## Second automatic CI follow-up — 2026-09-08

Rebased all 11 reviewed commits onto dev `8184ba8f6b`, retaining backup
`codex/voice-pr2504-before-final-ci-rebase`. Only the generated diagnostic-summary
count conflicted; the official generator confirmed exactly 7686 to 7687 with no
owner/sink drift, and the rebuilt artifact passes. The focused shared-file
composition review approved both retained voice and incoming Library changes.
Eight selected Library cases and two boot guards passed; UI-ready stays 972/973
and boot CSS stays 803655/804000. Published `d21e930658` to the same PR with
exact lease protection; every original thread remains resolved and auto-merge off.

Automatic native run `34300791333` passed both ARM platform jobs completely,
including installed tests and repaired-wheel validation. All non-native automatic
checks passed, including PR Fast Lane and derived artifacts. The three x86 jobs
reached later issues: missing Sinc SSE and Abseil optional-access implementations,
and Windows legacy Winsock headers colliding with explicit Winsock 2. The
read-only diagnoses established the exact step-11 repair before implementation.
Windows commit `69145b6d8f` adds one platform-only definition; two file guards
first failed, then all four selected consistency/scope guards passed. Its staged
and final full vendor verifiers pass with all 316 existing source entries intact.

ADR-098 records the independently checked six-source/header addition and ARM64
selection requirement before that repair. Hosted native CI must still verify
the resulting x86 builds/imports. No physical or release qualification is inferred
from the successful ARM jobs; signed/unsigned qualification jobs were skipped.
No local native execution, audio, providers, full sweep or merge was performed.

The six-file repair then reproduced the exact reviewed 322-entry pristine anchor
and Abseil subset identity. Nine expected source/count/architecture/inventory
guards failed before implementation; 39 focused provenance, include closure,
old-ledger preservation, patch roundtrip/tamper, line-ending, platform-mapping,
inventory and synthetic archive cases passed afterward. Three additional exact
Abseil/legal guards passed. All 316 old entries, the original 315/CPU control,
all three patches and the Windows definition remain intact; scoped lint/format
and the actual full offline verifier pass. These are file-level results, not
local native execution or final hosted-platform evidence. The complete repair
requires its focused combined review and automatic CI before handoff.

## Third automatic CI follow-up — 2026-09-08

The focused x86 repair review approved with no findings. A clean, no-overlap
rebase onto dev `80f29a9a1dcd9307662714233c65605e4c517b11` produced published
`4effec1421`; the entire incoming binary diff equals the resulting tree delta.
Eight selected new-dev/boot cases passed with UI-ready unchanged at 972/973.
Automatic run `34302675575` passes Linux x86, Linux ARM and macOS ARM completely;
Intel macOS is still running at this checkpoint. Every non-native check passes.
All six original review threads remain addressed, with auto-merge disabled.

Windows compiled all sources but exposed the missing upstream Winmm system link
at final linking. Step 12 restores only that target-owned transitive dependency.
Commit `e57c28b58a` contains CMake and one static regression: the new guard failed
before the fix, then four exact file-level guards passed, with scoped lint/format
and diff checks passing. No source, patch, definition, inventory, legal, version,
qualification or archive-policy bytes changed. These are file-level proofs;
normal automatic Windows CI must establish native link/import success. The
focused review and next automatic run remain required before final handoff.
No local native execution, full suite, live audio, manual workflow or merge ran.

## Fourth automatic CI follow-up — 2026-09-08

Published `f063035853` passes all four non-Windows native jobs and every
non-native check. Windows builds, repairs and tests all three Python wheels,
including binding and installed corpus checks, then correctly fails the final
archive check for a private MSVCP runtime DLL. The Winmm repair is proven.
The read-only runtime diagnosis confirms CMake's default dynamic runtime and
the exact two-target static property; a separate source audit checks allocation
ownership. These justify step 13 without relaxing the archive policy.

Commit `d3966073ae` implements only the property and its regression. One expected
RED precedes four GREEN file guards; scoped lint/format and diff checks pass.
The existing source/patch/definition/qualification bytes remain untouched.
Static checks are not native evidence; the next automatic run must pass real
Windows build/import and the unchanged archive checker. A focused review is
required before push. New dev `8aa2211f2b` adds Library wait cancellation/timeout
handling; inspect its incoming delta, retain a backup, cleanly rebase and verify
the narrow software interaction before publishing. No local native execution,
full sweep, live audio, manual CI or merge is authorized.
