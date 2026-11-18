---
id: TASK-33079
title: Verify concurrent Parakeet and OmniVoice model installs
status: Done
assignee:
  - '@claude'
created_date: '2026-09-27 18:30'
labels:
  - wizard
  - model-artifacts
  - tts
  - stt
dependencies:
  - TASK-32958
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first-run wizard can now install two curated models into the same managed artifact store: Parakeet in the Speech step and OmniVoice (1.1 GB) in the Voice step. Both installs run on background workers and neither blocks Next, so a Full-track user can start one while the other is still downloading. The design relies on the store's per-artifact locks to keep the two installs apart, but TASK-32958's live UAT never ran them at the same time (it would have needed a second 1.1 GB download). This task closes that gap before a user finds it.

**Premise corrected (TASK-34100.14, 2026-10-19).** The two installs never run in parallel. `provision()` holds one global exclusive acquisition session lease (`ACQUISITION_SESSION_LEASE_KEY`) for the whole download, so a second install started while the first downloads used to fail fast as busy -- with copy that named the caller's own model. TASK-34100.14 makes the second install wait in line ("Waiting for the OmniVoice voice model download to finish…") and start on its own, names the model actually downloading when a host fails fast instead, and lets either be cancelled. The acceptance criteria below were rewritten to that premise; the originals are kept under Implementation Notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With both installs started together on a fresh profile, the second waits for the first, both finish, both models verify, and each step then reports its model as ready
- [x] #2 Cancelling one install while the other runs or waits leaves the other install unaffected and leaves no partial files for the cancelled one (an install that fails rather than being cancelled keeps its partial so a retry resumes it -- TASK-34100.14 AC#4)
- [x] #3 An automated test pins concurrent provisioning of two different artifacts in one store, using fake sources (no real download)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed on TASK-34100.14's evidence (branch `fix/task-34100.14-download-coordinator`, 2026-10-19).

Original acceptance criteria, replaced because they assumed the installs run in parallel and that a failed install leaves nothing behind:
1. With both installs running at once on a fresh profile, both finish, both models verify, and each step then reports its model as ready
2. Cancelling or failing one install while the other runs leaves the other install unaffected and leaves no partial files for the failed one
3. An automated test pins concurrent provisioning of two different artifacts in one store, using fake sources (no real download)

Evidence:
- #3 -- `Tests/Model_Artifacts/test_download_control.py`: two `ArtifactAcquisitionService`s over one store root, each on its own thread with `asyncio.run` (the wizard's shape), against loopback fixture sources. `test_second_install_waits_for_the_first_and_then_both_install` (queued event names the holder, the second touches no network until the first finishes, both install), `test_second_install_fails_fast_naming_the_model_actually_downloading`, `test_cancel_while_queued_leaves_the_running_download_alone`. Mutation-checked: forcing the fail-fast branch, or dropping the holder label, turns them red.
- #1 -- live, real upstreams, fresh profile `g14v3-both` (evidence `setup-wizard-ux-qa/evidence/g14v3-both/`): OmniVoice (1,081.9 MiB) started in the Voice step; Parakeet v2 started in the Speech step showed "Waiting for the OmniVoice voice model download to finish…" (04), started on its own at 18:17:05 when OmniVoice finished (05), and reached "Installed and active." (06); both artifacts on disk, staging empty (07); the Voice step then read "OmniVoice is installed — runs offline on this computer." (08).
- #2 -- live, fresh profile `g14v3-final` (evidence `.../g14v3-final/`): with OmniVoice downloading, the queued Parakeet install was cancelled -- "Not installed. Download cancelled — nothing was saved." -- while OmniVoice's stage kept growing (199 → 232 MB in 4 s; only its stage present) (03, 04); cancelling OmniVoice then removed its stage 0.55 s after the click, leaving artifacts and staging empty (06-08). The other direction, cancelling the running download while the second waits: `test_cancelling_the_running_download_lets_the_queued_one_start` (mutation-checked), and live on fresh profile `g14v3-recheck` (evidence `.../g14v3-recheck/`): with Parakeet queued behind OmniVoice (05), cancelling OmniVoice removed its 290 MB stage in 0.40 s, Parakeet's stage appeared within 0.95 s (07), and Parakeet went on to "Installed and active." with staging empty and no OmniVoice artifact (09, 10).
<!-- SECTION:NOTES:END -->
