# Cross-platform backup verification — 2026-09-12

TASK-32496; PR2642 targets dev. This record supersedes the platform limitation in
the earlier Python encryption verification. Work remains in progress until the
Windows installed product cases pass.

Focused Windows run
[34714728661](https://github.com/rmusser01/tldw_chatbook/actions/runs/34714728661)
at `d0054781362b2a0a70301ef2587162a2f0aba908` passed all 42 native tests and
created the actual plaintext F9 backup, then failed restore validation with
`OSError(errno=22, winerror=123)`. All 14 artifact hashes matched. The safe Python
stack sampler completed and the separate fatal log was empty. The original error
identified logical root IDs containing colons used as temporary directory names
in `_validate_installed`. That one physical path now uses the same SHA256 root key
as staging; logical topology and archive identities are unchanged. A regression
reproduced the colon path before correction. Actual macOS plain F9 creation,
restore and open passed after correction (36.09 seconds). Windows verification
of this correction reached backup and restore success in focused run
[34715486325](https://github.com/rmusser01/tldw_chatbook/actions/runs/34715486325)
at `acb134a73fa36462ec26e7074f900321aa2ca795`. Native tests passed 42/42 and all 14
artifact hashes matched, but the actual UI case failed waiting for an open-profile
operation after activating the enabled profile button. No child process started.
Test-only observations now distinguish queued keyboard focus, handler dispatch,
and entry into the real terminal suspension; the original error observer also
covers the service's static error mapper used before launch.

Full Linux product verification of application revision `acb134a73` passed all
three F9 modes (127.25 seconds), the two-profile roundtrip (66.75 seconds), and
replacement/later rollback (281.96 seconds), with no failures or skips. Its public
codeload SHA256 is
`163cb81a15b0e836553dca5d9bab10dcd7f1c50fa5088c668776beb029c4aac7`.
The same application passed all five macOS product cases in 334.08 seconds.
Subsequent test-observer and actual macOS plaintext F9 regression passed four
cases in 35.27 seconds. Windows-only multi-profile fixture deadlines now cover
observed native capture duration and sequential opens; production limits are
unchanged. Full Windows product success remains required.

## Source and boundaries

Integrated source: `7d101919099780c6f4236c05845e8d86513a87d9`.
Public codeload archive SHA256:
`276ebc0917636f421489bdf35b5445f41ce448d21ec3a22fc1bd3a3c9509f9bd`.
Python worker SHA256 remains:
`451a3dc158c3c1e8ee599f384a23046eb9d446ca94e8c5e04b8ffe9c6e068c05`.

Latest directory-correction source:
`ebe86139753b56b7af9b363ea8e3b274723ed868`.
Public codeload archive SHA256:
`3d4fad9d513510ddba9f81cc8577fc0aa9f4d02dc8ff06841bba8cdc6dd9e161`.
Its full macOS product sequence passed all five cases in 335.28 seconds: three
F9 create/restore/open modes, two-profile capture/restore/open, and combined
replacement/later rollback. Linux passed all 49 native cases (1.97 seconds),
three F9 modes (129.70 seconds), the two-profile roundtrip (66.65 seconds), and
combined replacement/later rollback (284.56 seconds), with no failures or skips.
Windows full run
[34713157016](https://github.com/rmusser01/tldw_chatbook/actions/runs/34713157016)
passed all 42 native tests, including nested directory publication and installed
metadata, but passed only 15 of 22 product cases (seven failures, no skips).
All 22 artifact hashes matched. Three console cases exceeded their 65-second
backup observers before restore; mounted backup exceeded its outer observer.
Replacement reported a terminal capture failure. Two seed processes exited with
Windows access violation while their timed stack dumps were printing, consistent
with the [CPython timed-dump race](https://github.com/python/cpython/issues/140815).
That is a diagnostic hypothesis, not proof of the crash cause. The next test
correction uses a bounded Python stack observer and retains genuine crash detection
and original operation errors. Application deadlines remain unchanged.
Full Windows recovery is still unverified. Earlier results below remain distinct
historical evidence.

The local filesystem interface leaves POSIX's standard-library os unchanged.
Linux uses renameat2 no-replace, fsync and flock; macOS retains native exclusive
rename and full-sync; Windows uses Python ctypes native handles, ACL checks,
relative no-replace rename, actual namespace flush and native byte locks. Native
identity and every operation still refuse unsupported storage or failed barriers.
Archive paths use POSIX separators on every host. Existing encrypted age-v1 format,
credential review, safety copies and recovery journals are preserved.

Windows verified SQLite descriptor reads use a bounded main-file snapshot from the
held object, never a pathname reopen. Its source offset and identity/timestamps are
checked, SQL remains query-only, and the existing 576 MiB profile artifact bound
applies (transient memory can approach twice that). POSIX descriptor reads remain
unchanged. Independent review verified native close proof and failed-connection
retirement, including overridden close methods.

## Actual product runs

All runs use private synthetic profiles, blocked application network access and
NullKeyring. No personal profile or real credential store is used. Installed-wheel
fixtures retain installed-file hashes. Native primitive counts alone do not prove
complete backup/recovery.

- Linux: Debian kernel6.12.107+deb13-amd64, x86_64, Python3.12.8, local ext4.
  At5b7e60d96, all3 installed F9 create/restore/open modes passed; full replacement
  and later rollback passed; two captured profiles restored and opened with native
  contents. At integrated7d1019190,47native tests passed and3F9 modes passed126.96s;
  two-profile roundtrip passed66.04s; combined replacement/later rollback passed281.09s.
  All integrated Linux product cases completed with no failures or skips.
- macOS: the integrated interface regression passed3installed F9 modes104.11s and
  the combined replacement/later-rollback case190.94s. Exact7d1019190 full product
  sequence then passed5cases373.46s:3F9modes, two-profile capture/restore/open and
  combined replacement/later rollback.
- Windows2022Server, AMD64, Python3.12.10: actual baseline
  [34706507112](https://github.com/rmusser01/tldw_chatbook/actions/runs/34706507112)
  collected25cases:20passed5failed. Those failures exposed directory rights-reopen,
  relative rename and missing Windows child-runtime environment handling. Native
  fixes and full integration in
  [34707982570](https://github.com/rmusser01/tldw_chatbook/actions/runs/34707982570)
  could not reach test bodies:all39cases hit shared app-fixture setup errors from
  an untrusted ancestor owner in runner temporary storage. No native or product
  success is claimed from that run.
- Windows run [34708741479](https://github.com/rmusser01/tldw_chatbook/actions/runs/34708741479)
  at `065c5344eb8b34337ea68fe1e623456be633ba61` separated native and product phases.
  Native: 25 passed, one failed because a test omitted required keyword arguments.
  Actual exclusive rename, namespace flush, rights reopening and locking passed.
  Product collection still refused the runner's D-drive ancestor owner; no product
  body ran. All nine receipt hashes were verified.
  Anonymous security diagnostics showed a trusted C-drive user temporary ancestry.
  They also exposed CPython's OWNER RIGHTS ACL principal, which the adapter had
  incorrectly interpreted as public access. The correction resolves that principal
  only against the owner measured from the same security descriptor.
- Windows run [34709688754](https://github.com/rmusser01/tldw_chatbook/actions/runs/34709688754)
  at `df281c240` passed all 32 native cases and 15 of 19 product-phase cases,
  with no skips. All ten SQLite descriptor cases and five dependency lock cases
  passed. Four product cases failed: two-profile seeding could not overwrite its
  config, replacement and later rollback refused incomplete inventory, and mounted
  backup timed out awaiting review. Installed-wheel receipts were produced for
  three fixtures. Eleven of twelve manifest-listed artifact files were present
  and hash-matched; Actions omitted one hidden sanitized log, so the artifact set
  was incomplete. Its harness runs both phases from a private C-drive copy
  of the committed tracked source, verifying exact Unicode path sets and recording
  archive and copied-file hashes. It does not alter D-drive permissions or broaden
  production owner trust. OWNER RIGHTS regressions passed locally: 22 passed,
  10 native-only cases skipped on macOS; these skips are not Windows evidence.
  The next correction uses explicit atomic replacement for ordinary config writes
  and aligns each child fixture's HOME and USERPROFILE. Safe inventory diagnostics,
  visible sanitized log names and escaped-path redaction improve failure evidence.
- Windows [34710456481](https://github.com/rmusser01/tldw_chatbook/actions/runs/34710456481)
  at `ea917deb027af20150e665c192c264e4db596fc8`: all 34 native cases passed,
  including ordinary config replacement; product remained 15/19 with no skips.
  All 11 artifact hashes verified. The inventory diagnostic identified unavailable
  dictionary and built-in artwork roots, with 31 required artwork members missing
  from the refused tree. Windows directory stream enumeration had treated the
  documented no-stream result (`ERROR_HANDLE_EOF`) as an error. The next native
  correction handles only that result; other query failures still propagate and
  alternate streams remain detected. The two-profile seed also exceeded its
  60-second observer while creating records. Windows seeding now permits 120
  seconds and records a thread dump at 55 seconds. The pytest observer permits
  the existing replacement/later-rollback child deadlines to run in sequence.
- Windows [34711217652](https://github.com/rmusser01/tldw_chatbook/actions/runs/34711217652)
  at `b3843f718054ccbe88e7d1661fe9c8079b452baa`: all 40 native cases passed,
  including actual directory stream checks. Product cases: 16 passed, six failed,
  no skips; all 21 artifact hashes verified. Mounted F9 backup creation and archive
  verification passed. Plaintext and encrypted archives were created through the
  real console UI, but isolated restore failed with a permission error. All three
  console preflights passed; terminal suspension remains a real operation.
  The credential case exposed a test path-separator mismatch, and the two-profile
  seed completed before its log reader failed on Windows text encoding. Replacement
  seed observers expired before the later recovery operations could run.
  Test corrections use POSIX archive paths, native ACL mode observations, explicit
  UTF-8 child output and longer Windows seed observer bounds. Application deadlines
  are unchanged. macOS plaintext UI, encrypted credential UI and two-profile
  regressions passed in 34.45, 44.08 and 50.99 seconds respectively.
- Windows [34712601107](https://github.com/rmusser01/tldw_chatbook/actions/runs/34712601107)
  at `3180c688bf8354087b55a7461987483c045be334` passed all 40 native cases and
  failed its one selected plaintext F9 restore case. All 13 artifact hashes matched.
  This was a focused diagnostic, not full product qualification. Its bounded error
  metadata identifies `PermissionError`, errno 13, Windows error 5, in
  `journal.observe_artifact` through `WindowsOS.open` and `NtCreateFile`.
  Read-only opens incorrectly required a regular file, preventing journal inspection
  of directories. The correction permits an existing file or directory for a
  read-only open while preserving required-directory and mutating-file constraints.
  A native nested-tree publication and metadata test covers this operation sequence.
  Restored-directory metadata now uses the directory persistence barrier; its
  failure-propagation regression failed before the correction. No exception
  messages, local variables or source contents enter the diagnostic receipt.
  Full Windows product verification remains required.

## Regression and retained failures

- New archive-path and supported-platform regressions were observed failing before
  their fixes; capture/release/native-platform scope then passed66tests4.17s.
- Windows descriptor snapshot and existing migration scope:21passed1.38s. These
  local tests exercise real SQLite and filesystem operations while selecting the
  Windows snapshot route; the native Actions run is separate evidence.
- Private paths:66passed2existing macOS fixture skips. Private SQLite:324passed,
  3failures reproduced on clean ce115673e, plus1Windows-only skip. Inventory and
  release first pass:98passed,3old inventory failures reproduced on clean ce115673e;
  3obsolete exact-platform assertions were updated for the requested native contract
  and pass in the66test run above.
- Scoped compile and diff checks passed; CI workflow shape15passed. No new Ruff or
  Bandit findings relative to the clean baseline. Parent crypto retains2existing
  low-severity subprocess findings; private_sqlite retains7existing findings outside
  new code; the native Windows module has no Bandit findings.
- Linux private-file regressions at `b3843f718`: 66 passed, two existing platform
  skips, no failures (4.67 seconds). Public source archive SHA256:
  `c7885b030153c4fadc9c0ffe7352a492b572d7870b3646692ddb54dd627ffb76`.
  The preceding run exposed a fixture assumption about umask; the final-symlink
  test now explicitly creates its intended 0644 outside target. The actual refusal
  behavior was unchanged. This verifies the shared atomic config-write correction
  on the supplied Linux host as well as macOS.

Linux failures and synthetic fixtures remain on the authorized host under the
private test directory. Earlier source distributions had group-writable extracted
or installed parents; only disposable fixture permissions and driver umask were
corrected. Source bytes and production containment checks were preserved. Bulk raw
artifact transfer was rejected by automatic review because fixture data might be
credential-related; only validated numeric/identity summaries are recorded here.

No physical power-loss, device-detachment or application-version upgrade guarantee
is inferred from these process-level and product tests.

The necessary RAG generation dependency writer now keeps POSIX directory locking
and uses a stable private regular lock file on Windows. Six focused real-filesystem
and process-contention cases plus the existing independent-indexer case pass locally;
five selected native Windows product lock cases now pass. The same six focused cases and independent
indexer case passed on the actual Linux host at `065c5344e` (7 passed, 8.89 seconds),
after installing their existing optional NumPy and ChromaDB test dependencies in
the disposable virtual environment. No generation record format changes.
