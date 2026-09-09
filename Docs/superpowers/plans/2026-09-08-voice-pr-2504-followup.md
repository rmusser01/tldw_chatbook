# PR 2504 Rebase and Review Follow-up

**Goal:** Rebase the complete reviewed audio integration onto current dev, address every PR finding and failing automatic check, update the same PR, and leave it open for the user's later merge decision.

**Architecture:** Preserve ADR-094 Console custody, ADR-097 trace privacy/source ownership and ADR-098 isolated speculative voice. This is maintenance of approved behavior, not renewed audio qualification.

ADR required: no new ADR
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md` (with existing ADR-094 and ADR-097)
Reason: Behavior-preserving integration, targeted admission/boot defects, and declared portable vendor build corrections within existing contracts.

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

## Final root-owned handoff

After scoped reviews, run one proportionate joined software gate covering changed seams and obtain a final read-only combined review. Refresh dev/PR/comments and account for any changes. Push the same branch with explicit force-with-lease, reply to each inline finding with the fix or grounded explanation, and resolve only addressed threads. Observe normal automatic checks; diagnose failures before more tests or changes. Record exact final head/check status and leave PR 2504 OPEN and UNMERGED. Preserve worktrees and recoverable evidence.

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
