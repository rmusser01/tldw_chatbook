# Independent Windows d2f89 native-close verification

**Verified successful:** run **34924835239**, exact source **d2f89fbc01d49b326c14c7ef2f345d230032d0d4**, existing `native-close-diagnostic` selection. Read-only local artifact/Git inspection; no repository edits or app/test runs.

All **27** entries in the artifact SHA256 manifest were rehashed successfully. Clean source receipt and both complete JUnit reports agree with the run summary: **42/42 native PASS** (1.736s), **43/43 product PASS** (785.716s), **zero failed, errored, skipped or unfinished** cases. Effective return code is zero.

Product coverage is exactly 4 TTS, 8 DB-status, 12 Canvas policy maintenance, 6 policy-worker, 11 Canvas view-binding, and the two installed mounted routes. The six new policy-worker cases remain included and pass. Settings and Library outcomes:

| Actual installed route | JUnit case time | Child service close completed |
|---|---:|---:|
| Settings | 240.558s | 191.547s |
| Library | 215.317s | 209.922s |

JUnit case time includes fixture work; child times are instrumented elapsed observations, not startup-budget qualification. Selected order and retained fixtures map Settings to `test_mounted_console_complete_0` and Library to `_1`.

The exact driver is unchanged from reviewed 8db. Both passing routes require: Complete inventory preview; Canvas enabled before capture; successful Complete backup; native/runtime pause fully resumed; Canvas still enabled; actual post-backup note write plus old/new note readback; acquired archive with coherent manifest; original note bytes present and post-capture note bytes absent from the archived primary database. The child asserts the installed package origin, uses the existing network guard and completes ordinary app and service cleanup. These are actual executed assertions supported by passing JUnit, not inferred solely from phase markers.

Each route retained five inventory observations with no scope change or unavailable dependency. Each runtime log records five producer stages, cache retirement from eight leases to one, then zero at resume; pending operations, operations and raw operations are zero at that final observation. No failed runtime hook is recorded.

The installed receipt covers **2,867 files**. All **2,475 installed Python hashes** match source receipt and exact Git blobs using documented Windows checkout conversion: **2,443 CRLF + 32 byte-exact, zero mismatches**. Wheel SHA256: `c46cbb7e5e7a76d04bd91d249dae5e09d24752ac08ee8ac3674cacc6f12740e3`. This verifies retained installation evidence, not a new remote filesystem read. Driver, native-package/subprocess helpers, policy-worker tests, runner and diagnostic helper hashes were also compared against the exact source revision.

Git comparison confirms no changes under `tldw_chatbook` between 8db and d2f; this run changes test selection/observations. It establishes current Settings and Library acceptance for this bounded selection. It does not explain previous Windows failures, qualify all support cases, or establish whole-PR/startup success. The parent's separate GGUF keyboard timeout investigation remains outside this verification; security-descriptor decoding memoization is not present in this product.

Evidence: `/private/tmp/uat-d2f89-native-independent-summary.json`; raw receipts under `/private/tmp/uat-windows-d2f89-native-close/{verification.json,git-blob-verification.json,case-accounting.json}`. Artifact root: `/private/tmp/uat-windows-d2f89-native-close/backup-platform-windows-2022-py3.12-native-close-diagnostic-d2f89fbc01d49b326c14c7ef2f345d230032d0d4`.
