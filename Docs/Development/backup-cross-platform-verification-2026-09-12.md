# Cross-platform backup verification — 2026-09-12

TASK-32496; PR2642 targets dev. This record supersedes the platform limitation in
the earlier Python encryption verification. Work remains in progress until the
Windows installed product cases pass.

Revision `ae901141df253ba8705589b4bb85ec1d3790031d` confirms the Windows
restart correction in [34727849451](https://github.com/rmusser01/tldw_chatbook/actions/runs/34727849451).
Native294/294 and product75/77 pass, with no skips. Support71, plain,
encrypted, encrypted-credentials and two-profile groups pass. Replacement and
later rollback both start the fresh process and inspect the archive, then fail
first restore preview with `backup_operation_failed`. Neither begins replacement
publication or later rollback. The existing bounded exception observer is now
also installed in the fresh test driver to identify the original preview error.
All123 artifact hashes verify, with clean9,853-file source and installed2,030-file
receipts; source archive SHA256 is
`8d4043ed62225ffe0bda94ea5deeb1332e0877b238b98b64f932225344d0d667`.
Actual macOS combined replacement/later rollback passes192.22 seconds. Linux
passes87 reader checks17.66 seconds and combined282.21 seconds; public source
SHA256 is `ae44214a6c9fb3ba4ab865f6f40733ab92320e6eea24db2648f876f475bda28f`.

Revision `95f22d9126708499a52c648599fbd2c498713a22` passes every actual
installed macOS product flow: F9 three modes (90.46 seconds), two-profile restore
and open (48.97 seconds), combined replacement/later rollback (181.76 seconds).
The supplied Linux host passes 81 admission/reader checks (17.11 seconds), F9
three (111.75 seconds), two-profile (68.07 seconds), and combined (281.20 seconds).
Linux public source SHA256 is
`85f93f09f407eff93ec53d9dac7edf61c7ee86e780ae2b644184777404d41b92`.

Windows [34726981082](https://github.com/rmusser01/tldw_chatbook/actions/runs/34726981082)
at that exact revision passes native294/294 and product69/71: support65, all three
F9 modes and installed two-profile pass. Both shared seeds now pass and reach
normal F9 restart, but replacement exits with `0xC0000005` and rollback's fresh
interpreter receives a truncated script (`SyntaxError` at standalone `import`).
Neither reaches the fresh replacement driver. All121 artifact hashes and seven
2,030-file installed receipts verify against clean source, 9,850 tracked files.

The Windows restart now uses `subprocess.Popen` with fixed argv, filtered
environment, explicit standard handles and `close_fds=True`, followed by `_exit`
only after successful creation. This preserves old-process native-lease release
while bypassing the UCRT exec environment construction and argument splitting.
The failure matches [CPython143327](https://github.com/python/cpython/issues/143327)
and the underlying [UCRT report](https://github.com/python/cpython/issues/137934).
POSIX still uses `execve`. Two regressions fail before the Windows branch and
pass after it; the existing restart suite also passes (14 cases, 48.69 seconds).
Independent review requires and confirms explicit forwarding of handles0/1/2.
The Windows test observes a private PID/creation-time receipt and actual fresh
process completion within the original total deadline; it still requires the
unchanged restored-content/result assertions. Four observer regressions cover
delayed completion, timeout cleanup, identity mismatch and missing receipt.
Production and helper Bandit report zero findings; Ruff is clean. The subsequent
Windows run above confirms restart and exposes the remaining preview failure.

Current correction replaces startup readmission's 10ms timer polling with a
completion future posted by the same native worker after all cleanup. Its owning
task shields that future through cancellation, then performs the original pause,
lease and retirement checks before reopening producers. Two real held-gate tests
fail with the old polling wait and pass with notification; startup/cancellation
scope passes 16 tests (29.92 seconds). Actual macOS installed seed passes (16.41
seconds). Independent review found no issues; production Bandit reports none and
Ruff adds none to its existing findings. The seed's count-based release observer
now uses its intended elapsed-time budget (60 seconds Windows, five elsewhere).

This follows Windows [34726369956](https://github.com/rmusser01/tldw_chatbook/actions/runs/34726369956),
which passes native42 and reaches archive packaging but fails the count-based
readmission observer (one product failure, 197.42 seconds). All13 artifact hashes
and the installed receipt verify. The valid main-thread profile records 8,949
event-loop iterations and 17,548 sleep calls in 5.006 seconds, despite requested
10ms waits. Main-thread CPU is 2.781 seconds including diagnostic overhead;
29,913 calls exceed the bounded aggregation. Exact reproduction of the timing
mechanism remains Windows-specific: emulating coarse clock/resolution alone on
macOS does not spin. The observed excessive polling is removed directly.
The same corrected diagnostic revision passes77 Linux checks (16.40 seconds),
public source SHA256 `1c53b2f499ab86009809509f468c9c6759384dbcc7b43028aa19d5423ab3cc83`.

Windows [34724659597](https://github.com/rmusser01/tldw_chatbook/actions/runs/34724659597)
at `3b4a348889c808d821b9a183bc60ea9725ddc56d` passes native 294/294,
all three F9 modes, installed two-profile and support 53. Replacement and rollback
still fail with real `AdmissionTimeout` after final root validation. Product totals
are 57 passed, two failed, no skips; all 116 artifact hashes and seven installed
receipts verified. The blocking wait did not resolve the admission delay.
The same production revision passes the complete actual macOS suite (F9 three:
88.32 seconds; two-profile: 53.44 seconds; combined replacement/rollback: 191.70
seconds) and Linux SSH suite (69 reader/admission checks: 16.01 seconds; F9 three:
112.15 seconds; two-profile: 68.24 seconds; combined: 276.60 seconds). Linux public
source SHA256: `9206be22c7aa1a410958c048746d637661277c3cc05cbb8a820aa01a804e3035`.

The test-only replacement diagnostic samples current-process/native-thread
CPU every five seconds using the existing psutil dependency. Only numeric IDs,
CPU times and Python-thread association are retained: top 32 threads and last 12
samples. Unavailable thread data is explicit, never reported as zero. Group/token
timing remains, with native-call wrappers disabled. Three regression failures
preceded the passing ten-case diagnostic suite; Ruff/Bandit and embedded-child
compilation pass. A fixed workflow choice runs the one shared failing replacement
case. No production code, assertions or deadlines change in this diagnostic.

Windows [34725309476](https://github.com/rmusser01/tldw_chatbook/actions/runs/34725309476)
at `135fd148186443e535ae5c724540b148857bd440` passes native 42/42 but
expires at the seed's 300-second observer before packaging; no recovery failure
was recorded. All 12 artifact hashes and the 2,030-file installed receipt verify.
The 52.007-second admission group consumes 7.859 seconds on its own thread and
56.750 process CPU seconds. The final 54.687-second sampling window records
50.109 main-thread CPU seconds versus 7.844 backup-thread seconds. All four sparse
main-thread samples are inside Windows completion-port polling. The last pause
checks occur during cleanup, so they do not explain the active capture delay.
The next diagnostic profiles only five seconds on the main event-loop thread,
retaining bounded code metadata and timings to distinguish callback churn from
completion-port churn. No production or deadline changes are inferred yet.
Actual Linux diagnostic/admission scope at this revision passes 72 cases in
16.11 seconds; public source SHA256 is
`999173a1c296247e41cee1557ab379fe00e4b092bdc5601b5d45beb3499171a5`.

The first bounded profile, `b8cbf6268`, is invalid for thread attribution:
actual macOS seed output includes backup-worker calls and impossible thread-clock
totals (497 seconds inside a 5.124-second window). Its product combined
replacement/rollback case still passes (186.05 seconds). The current Python 3.12
cProfile measurement must not justify any production change. A concurrent-worker
regression and explicitly current-thread observation are required before reuse.
Windows [34726047676](https://github.com/rmusser01/tldw_chatbook/actions/runs/34726047676)
at that revision passes native 42 and reproduces real `AdmissionTimeout` during
seed capture (one product failure, 177.60 seconds). All 14 artifact hashes and the
installed receipt verify; its mixed-thread profiler data is not accepted evidence
of a CPU cause. The concurrent-worker regression reproduces contamination before
replacing cProfile with `sys.setprofile` on the loop thread plus an explicit thread
guard. Call state is bounded to 128 stack frames and 256 code keys, exporting only
32 metadata/timing rows. Fifteen diagnostic tests pass (0.88 seconds), including
exclusion of a busy worker and plausible main-thread CPU totals; Ruff/Bandit pass.

Observer-free Windows [34723928412](https://github.com/rmusser01/tldw_chatbook/actions/runs/34723928412)
at `4810bfddaedaf7e29f68ee9f5d707418f6bf3ec6` passes native 294/294 and
product 55/57, with no skips. The same five groups pass, but replacement/rollback
now expose real `AdmissionTimeout` after final root validation, before capture.
All 114 artifact hashes and seven installed-package receipts match clean source.
Removing the observer did not resolve Windows maintenance. Its macOS combined
replacement/later rollback case passes (189.30 seconds).

The remaining indefinite startup waiter still retries the native lock every 10 ms.
For waits with neither deadline nor cancellation, `_lock` now uses one blocking
native acquisition, preserving its held descriptor and all subsequent validation.
Timed/cancellable waits keep their original polling and limits. Existing Windows
handles are synchronous, matching [LockFileEx's documented wait semantics](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex).
Two held-gate regressions cover the blocking wait and deleted-root refusal after
release. Admission/monitor scope passes 46 tests (14.11 seconds), and startup,
readmission and cancellation scope passes 14 (32.85 seconds). Independent review
accepted the operation semantics; Windows performance benefit remains unverified.
Production Bandit has zero findings and Ruff adds none to its existing seven.

Revision `a5843f36e87faf10a42d19953ad02336498b39ea` passes all actual installed
macOS product cases: three F9 modes (90.90 seconds), two-profile restore/open
(55.16 seconds), and combined replacement/later rollback (187.14 seconds, tested
before commit with the same production change). Windows full matrix
[34723256876](https://github.com/rmusser01/tldw_chatbook/actions/runs/34723256876)
completed with native294/294 and product55 passes/two shared-seed failures.
All three F9 modes, two-profile and support51 passed; replacement and rollback
still expired during seed capture before their target workflows started. All114
artifact hashes match clean source and installed-package receipts. The shared
seed's diagnostic wrapped roughly two million native calls; an otherwise unchanged
run without this all-call timing observer will isolate its overhead. Lightweight
stack/failure observers remain, with production code and deadlines unchanged.
The same revision passes actual Linux reader/admission checks
(67 cases, 15.51 seconds), three F9 modes (112.66 seconds), installed-wheel
two-profile restore/open (68.45 seconds), and combined replacement/later rollback
(278.16 seconds). Linux public codeload SHA256:
`ddf7f210480e0e74a015df89764a8636dac087fa6646b3c563edece838095c21`.

Windows matrix [34722216765](https://github.com/rmusser01/tldw_chatbook/actions/runs/34722216765)
at `85d614822d4e6a7dbcebe1533cb2e4a1b3ce9206` passed all three F9
create/restore/open modes, installed-wheel two-profile restore/open, support43,
and native42 in each of seven jobs. All 117 artifact hashes verified.
Replacement and rollback still failed their
300-second seed observers, after successful maintenance admission. Stacks show
repeated native containment walks over 21 roots during discovery and capture.
Capture now tries lexical owner candidates first, then retains every other root
for physical-alias fallback. The original native containment function remains
the authority; ordinary admission and deadlines are unchanged. Eight regressions
cover the reduced walks, alias fallback, path boundaries and native refusals.
Root-order/materialization tests pass 52 cases (17.84 seconds). Capture/core tests
pass 77, with one pure-import assertion reproduced using unchanged HEAD code in
the child. Independent review found no issues; production Bandit has zero findings
and Ruff adds none to its 12 existing findings. Actual installed-wheel macOS
combined replacement/later rollback passes with this change (187.14 seconds).

Linux `85d614822` passed 59 reader/admission cases (15.46 seconds), three F9 modes
(113.31 seconds), installed-wheel two-profile (68.24 seconds), and combined
replacement/later rollback (277.04 seconds). Public source SHA256:
`b7e71443beb46f6d443a0d4b22c6b065301d17830a962015495ea4ebf12ff7c2`.
macOS on that revision passed three F9 modes (169.28 seconds) and installed-wheel
two-profile (110.82 seconds), but its combined test expired during the second
replacement operation; the root-order result above is a subsequent changed-code
run. The Linux `35dfcab31` fresh-fixture reproduction also passed combined
replacement/rollback (284.60 seconds); its original observer failure remains
recorded below.

Windows matrix [34721454240](https://github.com/rmusser01/tldw_chatbook/actions/runs/34721454240)
at `35dfcab316edf945be3add4db1525d2725a462d3` passed installed-wheel two-profile
restore/open, support36, and all native42 suites. Other five product jobs failed.
The retained-gate waiter is sleeping as intended, but final root validation still
took 61.734 seconds and exceeded maintenance's unchanged deadline. Lifetime native
call totals cannot attribute that cost; the next test observer records bounded
per-group native counts and wall/thread/process CPU times to distinguish excess
work from waiting. No new production admission change is made from those totals.

Restored child stacks identify the UI stall precisely: Settings' internal-prompt
count repeatedly re-entered the guarded configuration source while composing.
The count now holds one existing configuration operation across its unchanged
resolver reads. Every nested source check remains active; the scope retires on
errors, and the next count admits again. Two regressions failed before the fix;
32 configuration/admission tests pass (14.99 seconds), independent review found
no issues, and actual installed-wheel macOS plaintext F9 roundtrip passes (35.16
seconds). Production Bandit has zero findings. Seven existing authoring fixture
errors reproduced with the unchanged module before test bodies; its existing
Ruff import warning is also unchanged. The credential F9 test additionally waits
for mounted, settled checkbox focus before sending exactly one space event; its
actual macOS credential-inclusive flow passes (50.78 seconds).

Linux `35dfcab31` passed 54 reader/admission checks (14.80 seconds), three F9 modes
(124.41 seconds), and installed-wheel two-profile (75.36 seconds). Combined
replacement reached successful replacement and later explicit credential review,
then timed out waiting 65 seconds for the final rollback operation. Its same-
revision fresh-fixture reproduction subsequently passed (284.60 seconds), without
changing production code or deadlines. Public source SHA256:
`01f48698290d8f3b3e9b58ce2e2d648c040e5138afafda82cea049e316db3204`.

Windows matrix [34720655217](https://github.com/rmusser01/tldw_chatbook/actions/runs/34720655217)
at `bbc83d6c56155eac13cef174b6bb9633a322b203` passed the installed-wheel
two-profile restore/open case, encrypted F9 flow, all 33 support checks and
42 native checks in every job. Plain and credential-inclusive F9 flows created
and restored their archives, then hit Textual readiness timeouts opening the
restored app. Replacement expired during its seed backup; rollback seed capture
reported AdmissionTimeout. No replacement attempt is counted as passed on Windows.

The ordinary gate preflight eliminated competing root scans, but the wait still
reopened registry/gate files and checked their ACLs repeatedly. One final root
scan took 69.235 seconds; the other 518 group calls totaled 4.483 seconds. The
wait now uses one native gate descriptor outside registry authority, preserves
its original deadline/cancellation, observes gate identity, and releases the
temporary gate before all fresh registry/root checks. Three regressions first
failed for repeated registry reads, then passed with the fix. Full admission
scope passed 37 tests; lifecycle/recovery scope passed 33. Independent review
found no issues and separately passed eight gate/cancellation/remap tests.
Touched production Bandit found zero issues. The current diff also passes actual
installed-wheel macOS plaintext F9 create/restore/open (54.05 seconds). A bounded
Python-owned stack observer in the restored test child will identify its startup
stall; it records no locals or values and changes no application behavior. All
109 artifact hashes from the completed Windows matrix match. Next Windows
verification is pending.

Actual `bbc83d6c5` product tests pass on macOS: three F9 modes 125.68 seconds,
installed-wheel two-profile 66.19 seconds, combined replacement/later rollback
257.78 seconds. The same revision passes on the supplied Linux SSH host:
51 reader/admission checks 12.50 seconds; F9 three modes 123.78 seconds;
installed-wheel two-profile 71.26 seconds; combined replacement/later rollback
283.14 seconds. Linux public codeload SHA256:
`45e5e99dc7caee6e22a88384bb3602851a6934c53b7a6df589607f039bd45e6d`.

Windows matrix [34719523728](https://github.com/rmusser01/tldw_chatbook/actions/runs/34719523728)
at `ec2c668f731b3b830549c2f423f99b3072cc73a9` passed the complete encrypted-
credential F9 create/restore/open flow and all 24 support cases. All seven jobs
passed 42 native cases. Plain/encrypted archives were created and restored, but
fresh app opens hit Textual screen readiness failures after 48-second UI stalls.
The backup monitor's synchronous native filesystem poll now runs off-loop; its
single probe is retained through cancellation, and runtime coordination remains
on the original task. New regressions first reproduced the blocking behavior.

The rollback seed recorded an actual maintenance timeout after a 62.2-second
root scan while startup readmission was scanning the same registry. There were
21 distinct roots, so duplicate-root caching would not address it. Ordinary
admission now probes requested gates before scanning roots: a closed gate can
only defer admission. Temporary locks are released before unchanged fresh group,
identity, permission and alias validation. Pending remaps retain their original
immediate alias-aware refusal. The existing remap test caught a missing exception
to the fast path before correction. Application deadlines remain unchanged.
Gate/monitor/native-recovery scope passed 59 cases (88.06 seconds); independent
review accepted both production changes. Seven old failure injectors were repaired
to wrap the actual file/directory barriers after their old fcntl hook moved.
All their authority, journaling and cleanup assertions remain intact. Touched
production Bandit has zero findings; Ruff adds no findings to its prior baseline.

The Windows two-profile case actually passed both fresh opens (653.44 seconds),
but its job correctly failed for a missing installed-package receipt. Historical
two-profile runs used the verified checkout, unlike the wheel-based F9 and
replacement flows. It now uses the existing wheel fixture and verifies the loaded
code path in every child. The installed-wheel macOS two-profile case passes
(89.06 seconds). Windows verification of this fixture change is pending.

Linux `ec2c668f7`: ten new reader/observer cases passed (1.47 seconds), all three
F9 modes passed (130.05 seconds), two-profile passed (66.70 seconds), and combined
replacement/later rollback passed (283.85 seconds). Its public codeload SHA256 is
`69d783c3a2e66dfff306c538020089b56b0c0bcc57138fbba11ea272a96382c6`.
The macOS full run passed its first four cases, then its aggregate 480-second test
driver expired during the combined case; it is not counted as a full pass.

Latest full matrix [34718268264](https://github.com/rmusser01/tldw_chatbook/actions/runs/34718268264)
at `9b643f744e5bfdcca1c54083b848c12a9e048a77` passed all 42 native tests in
all seven jobs and 16 support cases. Six product cases failed; none skipped.
All 105 artifact hashes match, and every source receipt identifies the same clean
9,839-file source. The three F9 modes and two-profile case completed archive
creation and restore validation, then failed opening the restored app because
MCP recovery startup still used stdlib `os.geteuid` on Windows. That module now
uses the existing native adapter for ownership and consistent descriptor/path
identity checks. The projected-identity regression failed before correction;
four targeted checks pass. Actual macOS plaintext create/restore/open passes
(45.87 seconds). Broader MCP activation checks passed 65 cases and failed six;
all six failures reproduced with the unchanged HEAD module loaded in their child
processes, separately from the new native regressions.

The same missed native interface in restored skill handling and provider
reconnection review is corrected. Both native-parent regressions failed before
the import changes. Provider scope passed 51 cases (58.04 seconds); skills and
timing scope passed 27 (35.63 seconds). Independent review accepted all three
module integrations. Ruff is clean and touched production/timing Bandit has zero
findings. The Windows support group now includes eight focused reader regressions
in addition to all original product cases.

Replacement failed with an actual `AdmissionTimeout` after its final maintenance
root revalidation. Rollback's seed exceeded its packaging observer while still
performing root checks; later rollback did not start. A test-only observer now
records bounded aggregate call counts and inclusive timings for admission and
native metadata operations, plus numeric root/ancestor counts. It retains no
paths, SIDs or local values. Application deadlines and checks remain unchanged.

Full product verification of `9b643f744` passed on macOS (all five cases,
340.12 seconds) and on the SSH Linux host (three F9 modes 130.52 seconds,
two-profile 66.90 seconds, replacement/later rollback 285.86 seconds).
Linux public codeload SHA256:
`f7419760bc681b81f3b9509f472b01254b086f3f6bde9e5142cc72928bac2d04`.
Windows completion remains required.

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

Full Windows run
[34716196686](https://github.com/rmusser01/tldw_chatbook/actions/runs/34716196686)
at `4d667bd7adcb51bada00e796bdd07a0c66f06e02` passed 42 native and 16 of 22
product cases; six failed, none skipped. All 39 artifact hashes matched. All
three actual F9 modes created and restored archives, including encrypted
credential review; mounted backup also passed. Plain and encrypted UI cases
held a button instance replaced by a profile-list refresh. The credential case
entered the real Windows terminal suspension and launched its child, which then
failed because the filtered launch environment lacked Windows home selectors.
The launcher now retains `USERPROFILE`, `HOMEDRIVE`, and `HOMEPATH` while still
excluding provider and app overrides. Both native-home selection regressions
failed before this change and pass afterward; actual macOS plaintext roundtrip
also passed (three selected cases, 36.83 seconds).

The two-profile test failed on a backslash receipt key after successful capture;
fixture manifest/marker keys now match their POSIX file-member keys. Replacement
and rollback fixtures timed out while capture was still progressing. Windows
test observers are calibrated to the measured native runtime, with production
deadlines unchanged. Next verification splits the existing full selection into
seven independent Windows jobs, preserving all 22 collected pytest node IDs,
native tests, and failure receipts. Each flow must still pass.

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
