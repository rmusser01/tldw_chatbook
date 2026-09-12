# Cross-platform backup verification — 2026-09-12

TASK-32496; PR2642 targets dev. This record supersedes the platform limitation in
the earlier Python encryption verification. Work remains in progress until the
Windows installed product cases pass.

## Source and boundaries

Integrated source: `7d101919099780c6f4236c05845e8d86513a87d9`.
Public codeload archive SHA256:
`276ebc0917636f421489bdf35b5445f41ce448d21ec3a22fc1bd3a3c9509f9bd`.
Python worker SHA256 remains:
`451a3dc158c3c1e8ee599f384a23046eb9d446ca94e8c5e04b8ffe9c6e068c05`.

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
  success is claimed from that run. The next run separates native cases from shared
  app startup and records anonymous ancestor ownership/permission diagnostics.

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
native Windows verification is pending. No generation record format changes.
