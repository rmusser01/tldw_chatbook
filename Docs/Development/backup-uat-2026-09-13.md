# Backup and recovery UAT — 2026-09-13

**Result: FAILED. Neither the first-time nor the experienced-user journey reached a usable backup and restored profile.** This is an agent-operated acceptance walkthrough, not a human-participant usability study. No product fixes were made during the test.

Tested PR: [#2642](https://github.com/rmusser01/tldw_chatbook/pull/2642), revision `49923b824dbc6a180a10eeeb55e938de9efe907c`. Tracking: TASK-32562, which remains In Progress.

## Method and limits

The exact revision was built into a wheel and installed into a disposable directory. The application ran through its normal `python -P -m tldw_chatbook` entry point, with ordinary keyboard events in real tmux PTYs. There was no Textual Pilot, widget-value assignment, patched application, test-mode environment, or pre-completed onboarding for the first-time user.

The host was macOS 26.5.2 / Darwin 25F84, arm64, Python 3.12.11. The novice and fresh restore sessions used 120×40 terminals; the power-user session used 160×50. Each had a separate disposable home, config, data, cache, and temporary directory. No real provider credentials were used. Optional provider/model/voice setup was skipped using the displayed onboarding controls.

Native desktop Terminal automation was unavailable. Evidence consists of terminal text and ANSI frames, keyboard-action records, filesystem checks, and separately labeled diagnostics. These are not screenshots, and window rendering/accessibility were not validated. Linux and Windows were **not** exercised in this persona UAT; earlier native automated tests remain separate evidence.

[Final verification](backup-uat-2026-09-13/final-verification.json) confirms all 2,849 recorded installed files remained unchanged. The wheel SHA-256 was `2329f9240560ee4e3a7f769dc57abd364934816a7029cf343ed0c5f06777e9f3`.

## Journey results

“Blocked” means the dependent user outcome could not be tested after a preceding failure. It is not a pass.

| Journey / action | Result | Observed outcome |
| --- | --- | --- |
| First-time: complete quick onboarding without a provider | Pass | Traversed the actual six-step wizard and entered the application. |
| First-time: create and save a note | Pass | Saved “Novice backup acceptance note” with a unique reference in its body. |
| First-time: find backup using the guide's F9 shortcut | Fail | F9 did not open the feature. The current UI exposes Settings on F4; Ctrl+P → backup worked. |
| First-time: review a default Complete backup | Fail | Normal setup left `config.toml.lock`; review classified it as an unsupported owner and the archive as Partial. |
| First-time: acknowledge Partial and create an unencrypted backup | Fail | Create became available, but the operation ended `Failed: capturing / backup_operation_failed`. No output archive appeared. |
| Power user: add a second profile | Pass | Selected its config through the file picker; both configurations appeared in the form. |
| Power user: add a folder of 1,800 synthetic files | Pass for selection only | Selected the folder through the directory picker. Capture/restore of those files did not complete. |
| Power user: include credentials without supplying encryption passwords | Pass for validation only | Review refused with “Enter and confirm an encryption password.” No credential export occurred. |
| Power user: enter mismatched passwords | Pass | Review refused with “Passwords must match.” |
| Power user: create encrypted multi-profile backup | Fail | With matching passwords and Partial acknowledgement, capture ended `backup_operation_failed`; no archive appeared. |
| Inspect damaged archive | Pass for rejection | Inspection failed and restore controls remained unavailable. The message was only `backup_operation_failed`. |
| Inspect valid plaintext fixture | Pass | UI verified 134 files / 26,716,920 payload bytes; correctly identified Partial and credentials excluded. |
| Fresh installation: choose “Restore a backup” during onboarding | Pass for entry | Opened Backup & Restore, then selected Inspect / restore. |
| Fresh installation: inspect encrypted fixture with wrong password | Pass | Refused with `archive_unlock_failed`; password field cleared. |
| Fresh installation: retry with correct password | Pass | Archive verified and restore controls became available. |
| Isolated restore using the displayed destination fields | Fail | After 23 directory entries and a profile name, review refused with `backup_operation_failed`. |
| Diagnostically assisted isolated restore | Fail | Correcting shared database destinations allowed review, but Confirm ended `Failed: restoring / backup_operation_failed`. No destination directories were published. |
| Manual inert extraction of one selected group | Limited pass | Review and confirmation succeeded for an empty directory group, with 0 payload bytes. This does **not** prove nonempty file recovery. |
| Open restored profile and find the saved note | Blocked | No isolated restore completed. |
| Replace existing application data; later rollback | Blocked | The journeys produced no Complete archive or successfully restored profile. Earlier automated replacement tests do not satisfy this UAT. |
| Credential round trip; 1,800-file restore; cancellation; interruption recovery | Not completed | No passing claim for these cases. |

## Findings requiring resolution before acceptance

### 1. Running-app backup creation failed in both journeys

The novice had one newly saved note and the application's normal initial stores. The power user had two selected profiles and an external folder. Both reviews reported available storage/capability, then creation failed with the same generic message. Neither output archive existed afterward.

Evidence: [novice failure](backup-uat-2026-09-13/novice-90-result-after.txt), [power-user failure](backup-uat-2026-09-13/power-47-cancel-focus-after.txt), and [filesystem verification](backup-uat-2026-09-13/final-verification.json). The observations bound failure to within 129 seconds of novice confirmation and 142 seconds of power-user confirmation; these are observation bounds, not exact operation durations.

After closing the novice app normally, an **offline fixture-preparation call** to the installed backup API captured the same synthetic profile successfully. This narrows the observed difference to the running-app path, but does not establish its root cause or count as UI backup success. Process samples and application logs were retained for follow-up.

### 2. Normal setup prevents Complete classification

Review lists `unknown: unsupported` for the ordinary `config.toml.lock` file created during setup/configuration. The independently produced archive also records `shared_config:unknown:config.toml.lock` as excluded for reason `unsupported`.

Evidence: [coverage detail](backup-uat-2026-09-13/novice-65-coverage-detail-after.txt), [fixture manifest verification](backup-uat-2026-09-13/offline-fixture-verification.json).

An early observation that acknowledgement permanently disabled Create was **retracted**: after input events settled, Create was focusable and both captures started. The confirmed problems are Complete classification and subsequent capture failure. The coverage summary continued to say acknowledgement was required even after it had been supplied.

### 3. Destination review exposes internal constraints without usable guidance

The UI required **23 absolute directory inputs** for the novice fixture. Many were labeled only `root:<hash> (archive_root)`. Independent directories satisfy the wording shown in the form, but five hidden entries represent the same ChaChaNotes database. Review refused without identifying that relationship or the fields to correct.

A separate read-only reproduction using the recorded choices returned `shared_target_split`. After four fields were changed **through the UI** to share the fifth field's directory, review succeeded. Confirm then failed before publication. A private staging-only diagnostic returned `owner_relocation_unverified:db.agent_runs`: further owner-layout constraints were not communicated by the accepted review.

This establishes a failed user workflow for the submitted destinations, not that every possible destination layout fails. The assisted continuation is explicitly not an unassisted acceptance pass.

Evidence: [all submitted destination labels and values](backup-uat-2026-09-13/restore-ui-destinations.json), [initial refusal](backup-uat-2026-09-13/power-61-review-restore-after.txt), [shared-path diagnostic](backup-uat-2026-09-13/restore-review-diagnostic.log), [assisted review](backup-uat-2026-09-13/power-64-assisted-review-after.txt), [restore failure](backup-uat-2026-09-13/power-67-restore-result-after.txt), [staging diagnostic](backup-uat-2026-09-13/restore-staging-diagnostic.log).

### 4. Navigation and recovery feedback need correction

`Docs/Backup-and-Recovery.md` directs users to F9, while the current navigation shows F4 Settings. Ctrl+P → backup is usable. Coverage, restore, and extraction present long internal owner/group identifiers and metadata lists. Most failures collapse to `backup_operation_failed`, which gives the user no corrective action. Manual extraction group labels did not identify their contents; the selected group turned out to contain only an empty directory.

## Fixture provenance and retained evidence

The restore fixture was prepared only after UI creation had failed and the novice process had exited. It was created by the installed `capture_service` and `archive_writer` APIs from the actual synthetic novice profile, without deleting its lock or changing product code. An encrypted variant was produced by the installed crypto implementation. Neither fixture is credited as UI backup creation or credential recovery.

Its plaintext SHA-256 is `c6f97ef4a00a597bdd5374bdf8ae523677463f6a319aef80ea5ad5a8c7f10673`. Read-only SQLite verification confirms the archive contains the exact note saved through the UI. The original note also remained intact after testing.

Selected frames, diagnostics, and verification records are retained beside this report, with an [evidence hash index](backup-uat-2026-09-13/evidence-sha256.json). Full local evidence, fixture scripts, keyboard actions, ANSI frames, process samples, and disposable data remain at `/private/tmp/chatbook-backup-uat-20260913-xtbqee8d`. All three owned tmux application sessions exited through normal Quit actions; the private tmux server was no longer running. A capture attempted immediately after the final Quit found that the session had already exited; this was a test-recorder cleanup condition, not an application failure.

Only this report, its evidence, and TASK-32562 tracking were changed in the repository. Product behavior, dependencies, and the PR's requester-written Change summary were not changed. Documentation/evidence validation replaces code tests and Bandit for this reporting-only change; no claim is made that existing failing tests or CI checks were resolved.

Acceptance remains withheld pending fixes and a repeat of the failed and blocked journeys.
