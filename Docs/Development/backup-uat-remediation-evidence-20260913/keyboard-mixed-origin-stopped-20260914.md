# Actual keyboard UAT — stopped after mixed-origin recovery handoff

Requested immutable revision: `9e5914fca140e762b66ec4a7beb77cf2af450ca0`.
Fixture: `/private/tmp/chatbook-backup-final-uat-20260914-p_he73bh`.
Product tree: `39ae597ee119a72a0b1b3d12a1b69d4ba2dbcad6`.
Wheel SHA256: `bd660f042e357e2d5fab77e5dfa4a7f3e44e3a57d08f91225bee11d6acc3f77f`.
All 5,349 installed artifact files remain identical to the initial receipt (`installed-final-verification.json`).

## Completed observations

- Ordinary `python -P -m tldw_chatbook` launched in a new private tmux session; no Pilot, test mode, widget assignment, onboarding preseed or product patch.
- Actual six-step onboarding completed; blank Library note created and saved by keyboard. Original title: `Final keyboard original 9e5914`. Body: `Restore this original after later rollback. FINAL-9E5914-20260914.`
- F4 Settings then the ordinary command palette opened Backup & Restore. Review showed Complete coverage; plain backup creation ended Archive verified (frame027).
- Archive SHA256 `e54210cb14d3c699b7f04f51b1e543cdb46eaa16160693ffe5edfb54f893a340`; consistency coherent. Copied sealed Notes payload verified its manifest hash and contained the exact original note (`original-archive-readback.json`). No live SQLite read was used.
- Incoming real novice archive verified again, 136 files, excluded credentials. Replacement recovery handoff occurred by its visible button.
- Selected existing local config and separate private Files needing setup directory. Explicitly selected the 55 required safety-copy files and re-reviewed. Actual UI plan pages053–114 and subsequent sealed plan agree: 150 restore, 90 retire, 149 preserve. Every restore/retire target lies inside this disposable fixture; shared checkout resources are preserved only (`recorded-plan-destination-check.json`).
- Credential review stopped operation `023a50bf2c7540b49cfede4827c40fc4`. Abort untouched succeeded, recovery exited status0. Journal has only candidate_staged → prepared → prepublication_aborted (`aborted-journal-events.json`). No publication occurred.
- Fresh ordinary installed app reopened. Initial Library landing timed out at 5 seconds, with a recorded 5825ms event-loop stall and cancellation in `_study_count_or_none` via `_run_library_service_call`. Failure preserved before a single explicit Retry. Notes initialized successfully; Retry loaded the original note.
- Actual onscreen exact title/body readback is frame `final-abort-reopen-042`. Ordinary Ctrl+Q completed status0 (frame043). A DB/WAL/SHM copy made only after Quit independently confirms the one exact original note (`after-abort-readback.json`).
- All 135 shared Evals/config and assets file hashes remained unchanged after Abort and Quit.

## Failure and limits

The initial tmux process inherited cwd `/private/tmp/chatbook-backup-pr-20260912`. Initial ordinary launch used `-P` and installed-only PYTHONPATH. The controller identified `recovery_restart.restart` subsequently invokes `python -c` without `-P`; inherited cwd precedes PYTHONPATH and can import the checkout during recovery. Actual recovery review showed checkout Evals/config and assets paths. The fresh installed ordinary restart additionally logged Evals `storage_scope_not_enrolled`. These observations are preserved, but there was no intrusive module-origin inspection of the already-exited recovery process; the exact causal import trace is being independently regression-tested by the controller.

This invalidates an immutable-package-throughout-handoff claim. At the controller's instruction, further replacement, incoming-note/new-note, later rollback and next Complete backup were NOT attempted on this fixture. No cwd, binding, registry or fixture repair was made. The initial plain backup and observed Abort/readback are factual outcomes; they do not establish full replacement acceptance or a fixed Library timeout.

Secrets were generated separately with private permissions and entered into masked fields via stdin-fed tmux paste; their values were not placed in new action logs, environment or command arguments. The private-secrets directory must remain private and should not be included in shared evidence bundles.

All terminal frames/ANSI files and action logs remain in this evidence directory. Initial failure log: `post-abort-library-timeout.log`; final app log: `final-stopped-app.log`. Both tmux application sessions are exited, status0. Historical UAT fixture was only read as reference and not modified.
