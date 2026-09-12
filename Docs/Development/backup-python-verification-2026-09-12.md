# Python backup encryption verification, 2026-09-12

TASK-32495; Linux testing TASK-32494;
[PR 2642](https://github.com/rmusser01/tldw_chatbook/pull/2642).
This record covers the user-requested removal of Go. Earlier Go qualification
receipts are historical and do not qualify the Python implementation.

The Python worker uses the existing `pycryptodomex` dependency and preserves age v1
single-scrypt archives. The parent retains a fixed worker SHA-256 integrity check,
isolated subprocess execution, cancellation, streaming budgets and private output
publication. Worker edits must update the parent checksum in the same reviewed
change. No filesystem qualification has been widened.
The unchanged `native_qualification.json` SHA-256 is
`0aa458794127116a63dda7ccb61bcb26e41fe42534f83824b128a732d58c763d`.

## Verified implementation and component cases

| Check | Result | Receipt |
| --- | --- | --- |
| Worker framing, authentication, bounds, legacy archives and closed pipes | 63 passed, 16.39s | `/private/tmp/task32495-worker-fix1-module-green.log` |
| Parent transport, cancellation, memory, integrity and compatibility | 42 passed, 12.02s | `/private/tmp/task32495-transport-module-final2.log` |
| Full backup availability module | 24 passed, 3.19s | `/private/tmp/task32495-release-gates.log` |
| Source/editable/sdist/wheel encryption and artifact contracts | 11 passed, 27.59s | `/private/tmp/task32495-python-packaging-report.md` |
| Archive reader/writer and credential modules | 132 passed, one pre-existing failure, 14.18s | `/private/tmp/task32495-archive-credentials.log` |
| Installed F9 create → restore → open: plain, encrypted, included credentials | 3 passed, 106.47s | `/private/tmp/task32495-f9-endtoend.log` |
| Installed F9 replacement followed by retained-copy rollback | 1 passed, 207.23s | `/private/tmp/task32495-f9-later-rollback-bounded.log` |

Worker commit `0ffb1fdc3`; parent integration commit `76a0f1657`.
Worker SHA-256: `451a3dc158c3c1e8ee599f384a23046eb9d446ca94e8c5e04b8ffe9c6e068c05`.
Independent reviews: `/private/tmp/task32495-python-worker-review.md` and
`/private/tmp/task32495-python-transport-review.md`.

Tests ran on Darwin 25.5.0/arm64, Python 3.12.11, in the private current checkout
with private HOME/XDG/config/temp state, NullKeyring and the Python network guard.
No real profile, keychain, model or network endpoint was used. No Go was executed.
Existing Requests dependency warnings remain disclosed; the archive adversarial
test also deliberately emits a duplicate-ZIP-member warning. No cases were skipped.

The plaintext `test_explicit_directory_metadata_and_media_storage_roundtrip`
compares unnormalized expected directory records to reader output containing the
default `synthetic: false` field. The exact failure reproduces on clean pre-Python
commit `2220f3bdc` in 1.17s: `/private/tmp/task32495-archive-baseline.log`.
The test, archive reader and archive writer are byte-identical between that baseline
and the Python checkout. Neither their behavior nor their assertions were changed.
The current batch's JUnit is `/private/tmp/task32495-product-58g82scb/results.xml`;
the baseline JUnit is `/private/tmp/task32495-product-2y9gij34/results.xml`.

## Installed package evidence

The installed F9 fixture used the ordinary `py3-none-any` wheel SHA-256
`52fc0c226b2d0bcd95f107df85417b7f44903cad4de191dd930fcae2ee8daa73`.
All 2,027 installed files were preserved by the shared fixture's teardown checks.
Its receipt is `/private/tmp/task32495-product-dk2y2ont/pytest/f9-native-package0/native-package.json`;
JUnit is `/private/tmp/task32495-product-dk2y2ont/results.xml`. All three cases
recorded successful opening and preserved source data.

The combined replacement/later-rollback case used wheel SHA-256
`7e408f89dc4b154a3646fc69437cf28311bd19580b30ea811e012c8bae0f5dab` and
preserved all 2,027 installed files. Its final checkpoint reports successful,
validated restoration. Receipt and JUnit are under
`/private/tmp/task32495-product-pvzjhf_4`.

Its first attempt was interrupted by the disposable runner's 230s aggregate limit
while the later child had consumed about 122s of its 180s allowance. That failure is
preserved at `/private/tmp/task32495-f9-later-rollback.log`. Only the private
aggregate guard was changed to 480s: existing serial child limits 110 + 150 + 180s plus 40s
setup. All repository, UI and operation deadlines stayed unchanged. The complete
rerun passed in 207.23s, also below the former aggregate limit.

All three implementation reviews and the final cross-task code review approved
the correction; `/private/tmp/task32495-final-review.md` records no Critical or
Important code findings. Production worker/checker Bandit reports have no findings;
the parent retains its two pre-existing LOW fixed-subprocess findings. Test-only
assertion/subprocess findings and the existing warning are disclosed in the reports.

## Linux verification

Public code revision `98b5d0b24c44def024f87ab54af61cb6b01b1511` was downloaded
directly from GitHub on the supplied host. Archive SHA-256:
`7f7e9a98a3e05b1dd649f1db9ce6f8d42e3c2b6be3d3b4300551998ef9a74c0a`.
The worker and parent hashes matched the reviewed macOS source. Host:
Linux 6.12.107+deb13-amd64/x86_64, Python 3.12.8, Cryptodome 3.23.0, ext4.

| Targeted batch | Result | Outer duration |
| --- | --- | ---: |
| Python worker | 63 passed | 18.20s |
| Crypto transport | 42 passed | 13.34s |
| Source/editable/sdist/wheel packaging | 11 passed | 21.10s |

All 116 tests passed with no skips. The driver asserts that Go is absent from PATH,
uses private HOME/XDG/config/temp locations and NullKeyring, blocks parent Python
network access, and installs isolated dependencies from the explicitly populated
offline cache. It makes no OS-level worker-network-containment claim.

The first attempt failed entirely during fixture setup: two new harness parent
directories inherited 0775 permissions from the remote umask. Direct stat confirmed
those parents, while case/temp directories were 0700. The driver now explicitly
creates private 0700 parents; application checks were preserved. The failed driver
and artifacts remain under `artifacts-python-20260912-091533` on the host. Successful
receipts are `artifacts-python-20260912-091812` and `artifacts-python-20260912-091853`,
under `/home/ml-user/Working/chatbook-backup-linux-20260912`, with local copies in
`/private/tmp/task32495-linux-success`.

The prior [Linux baseline](backup-linux-baseline-2026-09-12.md) records missing
Linux publication/full-sync primitives and non-Darwin native qualification refusal.
Replacing encryption with Python does not itself resolve those limitations.

The final reviewer approved the code before the combined rollback and Linux runs
finished. The controller subsequently checked the successful logs and JSON/JUnit
receipts above, closing those two pending evidence items. This does not represent
a new independent review or additional tests. No full-suite or CI workflow execution
is claimed. PR merge conflicts and the human-authored Change summary requirement
remain separate merge gates.
