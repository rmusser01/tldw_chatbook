# Defer private SQLite atomic-write dependencies

TASK-34406.29 profiles the first send on dev `8e1b0c7e72db18e00e460d7fab2d0a909ceb7f2f`, after the Quit correctness repair in #3168. The first combined diagnostic hit the original 32,768-event limit and was rejected without timing or I/O attribution. A distinct scalar-only diagnostic passed every original gate and recorded 974.036 ms of instrumented post-save time. Its overlapping helper-start spans led to a separate, finite isolated-child import diagnostic. These profiling numbers are not an unprofiled speed result or an additive savings estimate.

Each isolated SQLite read helper imports `private_paths`. That module eagerly imported `secrets` and `tempfile`, although only `atomic_private_write_bytes` uses them. Move the two imports to the existing atomic writer, before any resource acquisition. Every helper process, permission, path, source/freshness, admission, atomic-write, custody, cancellation, deadline and native-retirement check remains unchanged. There is no cache, pooling, TTL, schema, dependency or API change. Existing ADR-126 and ADR-230 apply; this routine import relocation needs no new ADR. Earlier losing candidates remain separate.

Four POSIX causal controls execute the real fixed isolated helper for `prepare` and `pin_source`, refusing each write-only import in turn. All four fail on unchanged product with the explicit dependency-refusal marker, then pass with the import relocation. They require the original successful protocol, inode, 0600 mode, intact content and child exit. The connected eight-module run records **649 passed, 24 failed, 2 skipped**. All 24 failure nodes and terminal exceptions reproduce against exact unchanged dev code. Retired-owner and inventory expectations, and an incomplete copied-helper fixture, remain disclosed failures. An initial baseline selector truncated a filename containing a space and ran no tests; that receipt is excluded. Existing atomic-write and helper-custody tests were not changed. Scoped Ruff/format checks add no findings; inherited findings remain. Independent product and benchmark-chain reviews passed.

## Actual first-send comparison

The measured baseline is `5e3fb719600cc84a31961426aaed947deb9138f9`; candidate is `c03e965e5a898983116ec528f5b5ea4208d74419`. Both have identical tests and metadata, and baseline runtime equals dev. All eight original unprofiled fresh-app/private-profile first sends qualified in B/C/C/B/B/C/C/B order before statistics. Full settings, 27 ordered schemas, headers, exact CEDAR input/reply and HTTP 200, complete verified traces, zero checkpoints, strict logs, source/helper maps, normal Quit/unmount, driver/server exits and observed PID/port closure passed. No sends were excluded, retried or replaced.

| Median | Baseline | Candidate |
| --- | ---: | ---: |
| Enter-to-HTTP | 2.368540 s | 1.951788 s |
| Post-save preparation | 1032.5 ms | 999.5 ms |

Enter-to-HTTP median is 416.8 ms lower (17.6%). Every matched pair improves: baseline-minus-candidate differences are 449.2, 64.4, 474.8 and 285.1 ms. This passes the predeclared lower-median/all-four-pairs release gate. Post-save results are mixed: differences are -105, -99, +197 and +49 ms; two pairs are slower despite the 33 ms lower median. This does not establish uniform post-save improvement, a per-helper saving or an open-count reduction.

Raw evidence remains private under `paired-postsave-current29-private-import-firstsend-r1/`, alongside its qualified result, protocol, source/helper seals and independent reviews. Publication preserves every measured Python byte and excludes only unrelated prior Quit-task closure metadata. Required current-base/current-head CI and resolution of every PR thread still precede merge; inherited failures cannot waive a required check. Four pairs on one host establish no population p95, OS-cold-cache, Windows/native Terminal, later-send, large-history or broad-provider result. No full local suite was run.
