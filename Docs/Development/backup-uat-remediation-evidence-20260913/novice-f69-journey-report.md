# Actual novice/newcomer keyboard UAT

Artifact tree: `f69a11ecb0720db043d9e89616ff46dc789f8bad`; wheel SHA256 `2cb2fff7cf0612c6e8f29593d288dc93db445a093d14969abe0f185fe037b1ec`. All5347 installed-file hashes remain unchanged after the child run. Ordinary prepared `python -P -m tldw_chatbook` entry points ran in separate fresh disposable HOME directories. Actions used tmux keyboard events only; no Pilot, test mode, widget assignment, or product patch.

## Passed

- Novice six-step Welcome→Provider→Model→Voice→Protect→Summary; selected Write your first note, then Blank note. Typed title/body and used Escape autosave. Evidence novice02–14.
- F4 Settings→CtrlP Backup→Create→new plaintext filename→Complete coverage review→Capture→Archive verified. Evidence novice15–26. The first actual capture attempt on this artifact succeeded.
- Archive contains the exact saved note, id `f8df7474-0275-4422-8d9d-d00268b4edee`, title `First saved note merged backup`, body `My first saved note survives backup and restore. MERGED-NOVICE-20260913-F69A.`. Manifest consistency coherent. Archive: `../novice/first-time-complete.tldw-backup.zip`, SHA256 `cf386124bb7c0daf46ffe0a127454713bde3d83d677a88dc6cf289def7406114`. Evidence `novice-archive-verification.json`.
- Original live note reopened through the UI; appended ` Live editing resumed after verified backup.` and used Escape autosave. Read-only source SQLite verification confirms the edit. Evidence novice41–44 and `newcomer-restored-note-and-live-write.json`.
- Fresh newcomer Welcome→Restore backup→Inspect same archive→one new base/display name→Review→Confirm→Restoration validated. Profile `26c4691935a647c38237112490d51540`; base `../newcomer/restores/first-note`; display name `Recovered first note`. Restored note exactly matches the archived note, excluding the later source edit. Database SHA256 equals archived payload `445cf3feb90c6bdf94955acfdc61979486670cb6c500f5c87174194025abdfa1`. Evidence newcomer01–22 and the readback JSON.
- Review correctly refused the initial base beneath an existing parent containing recovery control storage (`isolated_destination_parent_overlaps_control`) and displayed guidance to use a separate private restore parent. Created only a private `newcomer/restores` parent, chose its new child base, and review succeeded. No partial restore was published by the refusal.

## Failed: native open receipt

Restored profiles→Open in new process launched the actual child; its Console UI mounted (newcomer27–28). F4 was intercepted by a Get started overlay (29); ordinary CtrlQ exited the child (30). The parent reported Failed opening profile / backup_operation_failed, profile recovery_required (31). This failed baseline is preserved and was not retried.

`newcomer-open-failure-config-delta.json` records the exact cause: the staged/restored config initially matched bootstrap fingerprint `51ef9dfd4c921d15b735754046fba01513aa9b46448a622372cdb0cda7989876`; after child mount its hash was `5ae1763b7892104e525ee0be729d4fc463bbc118124b20d95c3cd1daca8f704b`. The only semantic change was the automatic addition of default `console.rail_state` values. No opened receipt was present. Source investigation identifies `_build_console_rail_state → _ensure_console_rail_scope_seed → _save_console_rail_preferences`. The incidental Evals warning is not treated as the terminal cause.

## Environment interruptions

Host ENOSPC twice prevented the controller from writing a before-snapshot: before novice title actions10/11 and before newcomer Open action27. No corresponding key was sent in either incident. Work paused, duplicate build inputs were reclaimed by the controller, and separate resumed action labels preserve the distinction. All entered note content subsequently saved and verified; no backup/restore retry was counted as a pass. The child Open was sent only after rechecking1.8GiB free. The final open failure occurred after startup's automatic configuration write, as shown by the exact semantic diff above.

Overall journey remains failed until the native Open/ordinaryQuit acceptance passes on the corrected integrated artifact. Automated regressions are recorded separately from this actual keyboard evidence.
