# PR 2504 Rebase and Review Follow-up

**Goal:** Rebase the complete reviewed audio integration onto current dev, address every PR finding and failing automatic check, update the same PR, and leave it open for the user's later merge decision.

**Architecture:** Preserve ADR-094 Console custody, ADR-097 trace privacy/source ownership and ADR-098 isolated speculative voice. This is maintenance of approved behavior, not renewed audio qualification.

ADR required: no new ADR
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md` (with existing ADR-094 and ADR-097)
Reason: Behavior-preserving integration, targeted admission/boot defects, and declared portable vendor build corrections within existing contracts.

## Global Constraints

- Work only in `.worktrees/speculative-duplex-voice-dev`, branch `codex/speculative-duplex-voice-dev`. Preserve main and the original feature worktree, including all unrelated edits.
- Published PR head is `47de10b893f4be42cdef7931e3ddecef33c6c539`; prior integrated dev is `37bf45fb6232a1d4fb50fdba8f3c19c856ae7664`; fetched dev is `7e81ed55db66ace04cb3dd1f8feb6d40a21f6f48`.
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

**Owner/files:** `UI/Console_Modules/wiring.py`, targeted UI admission tests, `Chat/console_speculative_voice_session.py`, `Audio/rolling_transcript.py`, optional explanatory `Audio/aec_backend.py` docstring, and TASK-23175 follow-up notes.

1. Before implementation add a follow-up acceptance criterion/plan to TASK-23175 for rebased PR fixes and open-PR handoff, without marking broader qualification complete.
2. Add a regression for a deleted session during synchronous send admission: missing owner must give the established closed-session refusal while preserving the captured draft and attachments. Add a control proving unrelated internal `KeyError` for a valid owner is not hidden. Demonstrate RED before narrowly normalizing only a proven missing owner in the shared admission boundary; do not blanket-catch `KeyError` in the UI.
3. Move `FrozenTracePolicy` into the local import group. Add Google-style Args/Returns for material transcript comparison without changing normalization behavior. Optionally explain why the native child's lazy AEC import intentionally avoids the app-side optional-dependency registry under ADR-098.
4. Do not add a second UI-thread hop: existing process delivery runs on the UI loop. Do not import the app optional-dependency graph into the isolated child. Do not replace the bounded stdlib wire validator with Pydantic. Cite the independent report for these three rejected suggestions.
5. Run the new regression/control plus exact adjacent draft refusal, rolling comparison, lazy AEC and UI-thread identity tests, excluding all real-native nodes. Apply scoped lint/diff checks, commit explicit paths, write `task-2-report.md` with RED/GREEN evidence.

## Task 3: Restore lazy boot without weakening performance guards

**Owner/files:** Voice-specific hot imports in Console UI/runtime/store/persistence/provider/trace modules, lazy voice-preview composition if needed, boot import/worker tests, and relevant documentation.

1. Reproduce the startup import regression using the existing software-only Perf Guard command/tests. Latest dev already uses 973/973 allowed modules. Preserve that budget and required worker identity; never refresh or raise the ratchet to accommodate eager voice imports.
2. Defer native stream, voice controls/settings, dispatch/promotion and trace voice contracts until first voice use; type-only imports belong under TYPE_CHECKING. Preserve synchronous winning claims and constructor/disposal semantics. Confirm every duplicate hot edge is removed.
3. Prefer lazy creation of the initially empty voice preview on first relevant use, preserving projection ordering and cleanup. Do not move definitions into a hot module solely to evade the module counter. Investigate actual widget lifecycle before choosing the smallest correct seam.
4. Replace a blind fixed 1-second wait in the startup worker anti-vacuity test with bounded condition waiting for the same required worker if its timing defect remains after rebase. Do not delete/weaken its assertion or lengthen a blind sleep.
5. Verify boot guard plus focused first-voice activation, actual factory/fake-child, preview and synchronous custody behavior with no hardware/providers. Commit only explicit changed paths; report RED/GREEN, exact startup module count and warnings in `task-3-report.md`.

## Task 4: Correct hosted native wheel portability and paths

**Owner/files:** `.github/workflows/voice-aec-wheels.yml` (resolve exact current filename), `native/voice_aec/.gitattributes`, vendor patch tooling/metadata and the single affected upstream header, related file-only packaging regressions.

1. Read existing declared patch and pristine-verification mechanism before editing. Add `<stddef.h>` for global `size_t` in the affected WebRTC clock-drift header, matching adjacent AEC3 headers, as an explicit second pinned patch; update patch series, patch hashes, PATCHES metadata and current closure manifest consistently. Preserve pristine original bytes/hashes and reverse-patch verification. No hidden source rewrite or relaxed integrity check.
2. Add a file-only regression proving the patch/manifest closure is valid and the new include exists for this compile error. Do not compile, load or test native code locally.
3. Enforce LF checkout for the native pinned source/manifest/patch scope through existing nested `.gitattributes`, not repository-wide line-ending changes. Verify a temporary Git checkout with `core.autocrlf=true` preserves closure bytes using file-only checks.
4. Fix cibuildwheel `{project}` test paths to repository-root-relative `native/voice_aec/tests/test_binding.py` and `Tests/Packaging/test_voice_aec_installed_wheel.py`, including any duplicate configuration. Test the command/path contract without executing installed-wheel test scripts locally.
5. Run only file-level verifier/packaging regressions and static checks. Preserve exact companion/app version 0.2.0, qualification/build-identity JSON and release controls. Commit explicit files; report all local limits and required automatic hosted CI evidence in `task-4-report.md`.

## Final root-owned handoff

After scoped reviews, run one proportionate joined software gate covering changed seams and obtain a final read-only combined review. Refresh dev/PR/comments and account for any changes. Push the same branch with explicit force-with-lease, reply to each inline finding with the fix or grounded explanation, and resolve only addressed threads. Observe normal automatic checks; diagnose failures before more tests or changes. Record exact final head/check status and leave PR 2504 OPEN and UNMERGED. Preserve worktrees and recoverable evidence.
