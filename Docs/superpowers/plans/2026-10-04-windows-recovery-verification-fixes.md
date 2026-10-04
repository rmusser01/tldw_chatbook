# Windows Recovery Verification Fixes

TASK-34408. User authorized both demonstrated Windows corrections in this chat.

ADR required: yes, amendment to the existing durability contract
ADR path: `backlog/decisions/126-complete-local-backup-and-recovery.md`
Reason: record the bootstrap creation intent and identity-bound receipt used to
preserve native durability while limiting subsequent publication barriers.

## Findings and Scope

Diagnostic wrapping of the real native reopen call showed that the failing
directory is `C:\Users`, requested access `0x6`. Private directory flushes succeed.
`publication._pending(durable=True)` redundantly flushes every ancestor after
flushing the pending record and bootstrap directory. Successful registration
already flushes changed parents, but simply deleting the ancestor loop is unsafe:
a creator can die after mkdir and before its parent barrier, leaving a directory
that a later attempt would otherwise adopt. Changed permissions cannot establish
whether a previous creation completed. No native access check or required
persistence barrier may be weakened.

The test helper selects anonymous subprocess pipes with `select`, which supports
only sockets on Windows. Its negative readiness assertions have the same defect.

## Implementation and Verification

1. Add RED regressions using actual registration/publication records: flush only
   the changed bootstrap directory during pending verification, retain required
   barrier failure propagation, and prove newly created ancestors are durably
   published by registration.
2. Persist an exclusive creation intent before mkdir, perform creation, its parent
   barrier and intent retirement through the same pinned parent, and refuse retry
   while an intent survives. Establish a private identity/path-bound ancestry
   receipt before pending publication. Certified publication flushes its pending
   record and bootstrap directory; uncertified legacy pending operations retain
   their original ancestor barriers. Validate the receipt in every control reader.
3. In parallel, correct the test-only pipe readiness helper using non-consuming
   native Windows readiness and existing POSIX readiness. Keep one bounded
   deadline for complete lines and preserve EOF/error and absence assertions.
4. Run targeted native Windows primitive, admission/lifetime, Eval retained-source
   replacement/selective/rollback and schema regressions. Include installed-product
   evidence when the changed publication path is exercised there. No full sweep.
5. Review the combined fix, run touched-scope static checks, and update both this
   task and pending TASK-34407 verification with measured evidence.

## Execution Record

- Initial native reproduction: recovery setup fails at the unconditional ancestor
  loop, with `OpenFileById(C:\Users, access=0x6)` returning WinError 5.
- Native namespace flushes themselves succeed; an older primitive test's final
  empty-directory assertion sees conftest's `test_data` sibling. Use an owned child
  directory for that fixture's namespace rather than weakening its assertion.
- The first publication regression failed against the unchanged ancestor loop.
  Review of a simple loop removal then reproduced the interrupted-mkdir retry
  gap. The final protocol preserves a durable intent and refuses retry, including
  after a real child is killed, permissions change, or a different case is used.
- The creation/receipt regressions also cover required file/directory barrier
  failures, changed root identity/path, held-parent rename, and unresolved legacy
  pending records. Fresh review found no remaining substantive defects.
- Real pipe regressions cover text/binary responses, non-consuming readiness,
  partial-line deadlines, EOF/errors, independent children, and 120 KB of stderr
  before a response. Logging-heavy fixtures preserve stderr in owned files:
  unread Windows stderr pipes had blocked children before their first marker.
- Native contention fixtures use the actual facade for fault injection and
  capability replacement. Registry observations take the existing native lock;
  killed-process release retries only the documented incompatible-client reason
  within two seconds. Windows directory alias refusal uses a real junction.
- The later-snapshot test now follows the existing directed owner dependency:
  restoring config alone preserves Eval source authority. Missing-file checks use
  ENOENT rather than platform-specific exception text; the Windows alias case uses
  a real hard link. No production owner dependency or provenance gate changed.
- Two file-symlink negatives require a Windows privilege unavailable to this
  account; only actual WinError 1314 skips those cases. Other creation errors fail.
- The installed-wheel two-profile flow installed and entered restore with
  `PIP_NO_COMPILE=1`, but exceeded the existing 300-second test timeout. It remains
  unverified. A separate complete-rebackup fixture fails before publication with
  unavailable/overlapping owner inventory on this host and is excluded from the
  final targeted recovery run. No full sweep or complete-release claim is made.

## Final Targeted Evidence

All native runs use an owned disposable directory under the real user's home as
TEMP/TMP, retaining conftest's profile isolation and the real native facade.

- Recovery run: `Tests/Utils/test_windows_files.py`, the new pending-durability
  module, bootstrap registry reader, Eval private overrides, retained definitions,
  selective retention, rollback retention, later snapshots, and the complete
  `test_default_config_files_fresh_process_recovery_then_owner_reopen` matrix.
  **124 passed, 2 skipped, 1 deselected in 765.36 seconds.** The two skips are the
  actual WinError 1314 file-symlink cases. Only the host-blocked complete-rebackup
  case described above was deselected. Present/absent settings, prepared/published/
  activation-staged/activation phases, and finish/rollback owner reopening pass.
- Final contention run: complete pipe helper and admission modules, all 13
  pending-durability cases, migrated raw native uncertainty and ambiguous
  publication cases, runtime uncertainty, actual settings writer exclusion,
  Eval pause/draft/drain/failed-native-close/default-selector cases, and shared
  Eval selector/YAML schema validation. **100 passed in 152.00 seconds.**
- These runs contain **211 distinct passing cases**; the 13 pending-durability
  cases occur in both. No POSIX execution or full-suite execution was performed.
- New helper/regression files pass Ruff lint and formatting. Changed production
  ranges pass formatting. All 21 touched Python files parse; comparison with
  HEAD finds **zero added lint diagnostics**, with 134 inherited diagnostics
  retained. Strict JSON integer checks explicitly exclude bool. Diff whitespace
  check passes. Unrelated inherited formatting is preserved.
- Fresh whole-change review and subsequent protocol review found no remaining
  substantive defects. The earlier interrupted-creation, unstable-rights, stale
  receipt observation and activation recovery-reader findings were corrected
  and exercised by native regressions before this result.
- The owner subsequently requested a PR against dev. Installed-package and
  complete-release gates remain unverified as
  recorded above; these targeted results do not certify the entire release.
