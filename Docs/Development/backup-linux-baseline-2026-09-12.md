# Linux backup baseline, 2026-09-12

TASK-32494; [PR2642](https://github.com/rmusser01/tldw_chatbook/pull/2642).
Source: `2220f3bdcd5c8e9f43bd4acac10a07a12ef049d0`, fetched directly from the public
GitHub repository. This is the baseline before the user's Python-only correction
(TASK-32495), not qualification of the replacement implementation.

Environment: Debian Linux6.12.107+deb13-amd64, x86_64, ext4 home filesystem,
Python3.12.8 in a private venv, Go1.26.2 provisioned privately for the initial
baseline. The system Python is3.13.5; that interpreter was not the test interpreter.
Test root: `/home/ml-user/Working/chatbook-backup-linux-20260912`.
No real application profile, keychain, database or model was used.

| Targeted batch | Result | Outer duration |
| --- | --- | ---: |
| Existing helper Go unit tests | Passed | 2.68s |
| Existing helper Go vet | Passed | 0.51s |
| Native helper protocol and source availability checks | 2 passed | 3.98s |
| Crypto behavior/interoperability/adversarial module | 37 passed | 13.19s |
| Native filesystem module | 5 passed,28 failed | 3.07s |
| Archive reader and writer modules | 8 passed,65 failed | 4.87s |

No skips replaced required evidence. Python batches ran serially with private
HOME/XDG/config/temp locations, NullKeyring, the Python network guard and offline
Go caches. The normal-UID runs do not claim inherited native network isolation.
Logs, JUnit and environment records are retained locally under
`/private/tmp/task32494-linux-results`, and remotely under `artifacts` and
`artifacts-normaluid` in the test root. Exact driver:
`/private/tmp/chatbook-linux-tests-20260912.py`; remote original and corrected drivers
are `run-tests.py` and `run-tests-normaluid.py`.

The initial user-namespace attempt stopped all Python cases during fixture setup:
the namespace changed ownership of host root directories from UID0 to unmapped
UID65534. Chatbook refused those parents with `untrusted_directory_owner`. An
explicit inside/outside `stat` check confirmed this. A two-case prerequisite rerun
under normal IDs passed; the table shows subsequent normal-UID outcomes. The failed
namespace logs remain separate and are not counted as product test failures.

Concrete Linux limitations in the tested source:

- `native_files._rename_new` directly calls macOS `renameatx_np`, absent on Linux.
- Native barriers use macOS `fcntl.F_FULLFSYNC`, also absent. This accounts for59
  archive assertions failing before their intended validation boundary, and other
  dependent process-exit/native cases.
- `qualification.native_identity` explicitly refuses non-Darwin platforms, so
  publication/capture/replacement availability cannot become true on this host.
- Several native tests expect Darwin-specific successful qualification or refusal
  codes. Their failures do not prove a Linux operation ran incorrectly; that
  operation was unavailable before the tested point.
- Four archive cases stop with `unsafe_directory`; their fixture permissions still
  need separate assessment. They are not silently attributed to the native ABI.

The Python-only encryption correction does not itself resolve these filesystem
limitations. No Linux release capability was enabled. Further Go tests/builds were
stopped when the user rejected that dependency. Three synthetic ciphertext files
from the completed baseline were retained as fixed independent compatibility vectors
under `Tests/Backup_Recovery/fixtures/age_v1`; their README records hashes and inputs.
