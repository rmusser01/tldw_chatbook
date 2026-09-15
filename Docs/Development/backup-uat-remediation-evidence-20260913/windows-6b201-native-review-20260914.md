# Independent 6b201 Windows native-close result

Run34928730294 / job104252275027, exact revision `6b20175799c47f44cf287065207630ab1ad86e93`. **The run is failed solely by one new invalid-handle fixture; all43 product cases passed.** No app/test rerun or repository edit was performed.

| Scope | Verified result |
|---|---|
| Artifact manifest |27/27 SHA256 matches |
| Native phase |59 complete:58 passed,1 failed;0 skipped/errors/unfinished;1.326s JUnit suite time |
| Product phase |43/43 passed;0 skipped/errors/unfinished;508.088s JUnit suite time |
| Installed receipt |2867 files,2475 comparable Python files,0 mismatches |
| Exact Git Python |2475 blobs verified:2443 Windows-CRLF,32 exact,0 mismatches |
| Wheel SHA256 |`691cf8af93833dcd98ce0f1a8a7af58e3e12c96853e5708281cdd0fa91e3d609` |

Clean source receipt matches the revision. Additionally verified both native test modules, mounted driver, subprocess helper, native-package fixture, runner and thread observer against exact Git with CRLF accounting. Detailed hashes/accounting are in `/private/tmp/uat-6b201-native-independent-summary.json` and the retained verification files under the artifact root.

## Sole failure and minimal correction

`test_real_warm_decode_cannot_hide_invalid_handle` warms a genuine file handle successfully, then expects `native.security(-1, False)` to raise. It did not raise (`DID NOT RAISE OSError`), as retained in `junit-failures/c4cf6b45a193c669.txt`. The assumption that -1 is unconditionally invalid is incorrect: Microsoft documents the current-process pseudo-handle as `(HANDLE)-1`. [GetCurrentProcess documentation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentprocess).

Recommend the single test-only replacement with **`native.security(0, False)`**, keeping the warm valid-file call and existing OSError assertion. A null HANDLE avoids pseudo-handle semantics and the numeric handle-reuse race of closing and reusing an old handle value. Do not change product checks or classify this as a cached-permission bypass. A fresh native run must verify the corrected premise. My earlier implementation review missed the -1 fixture assumption; the native result supersedes that part of the earlier assessment.

## What the passing cases prove

The17 new native-module cases finished16 passed/1 fixture failure; all42 original native cases passed. Actual ACL grant→hardening and path replacement identity tests passed. The count test observed **11 fresh GetSecurityInfo calls and11 descriptor frees**, with only3 GetAce and4 SID conversions in the initial decode across the repeated identical observations. Three200-read samples gave medians0.0016168s warm cached versus0.0039557s original decoding. This is a concrete native microcost result, not a startup speed or overall security-cost attribution claim.

Product coverage remains4 TTS close,8 DB-status,12 Canvas maintenance,6 Canvas policy worker,11 Canvas binding lifetime and2 actual installed mounted routes. Settings passed156.024s; Library passed113.166s (JUnit case durations include setup). Settings child construct completed20.406s, mount yielded39.203s, service close completed120.016s. Library construct completed20.296s, mount yielded38.828s, close completed109.984s. These phase values are child-relative instrumented observations and should not be added to JUnit durations.

Both exact drivers assert Complete inventory and successful Complete archive, coherent acquired-archive readback, before-capture note in archived DB and after-capture note absent, successful resumed ordinary writes/readbacks, and Canvas enabled before backup and after maintenance. The driver asserts installed package origin and the outer helper requires child exit0 plus terminal marker. Each route retained five unchanged/available inventories; runtime leases transition8→1→0, with pending/operations/raw operations0 at resume. Thus the passing route results cover those real assertions, not only UI navigation.

This does not mark the entire PR accepted, explain earlier historical failures, or resolve the separate remaining startup keyboard timeouts. Source and installed receipt checks qualify only this immutable6b201 run.
