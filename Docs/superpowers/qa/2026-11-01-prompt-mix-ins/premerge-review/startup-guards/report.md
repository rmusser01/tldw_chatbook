# PR3346 frozen-source final five guards

Result: **5 passed, 3 existing headroom warnings in 29.20s**. The exact five unchanged guards ran serially after root released the isolated slot and confirmed all other native/app/performance and heavy commands stopped. All owned processes are now stopped. No source, task, threshold, allowlist, census, assertion, snapshot or exemption was changed during this performance phase.

| Guard | Actual result |
| --- | --- |
| UI-ready module census | 1031 / 1031 modules; snapshot drift +31/-22 |
| Boot worker/thread census | Original allowlist passes; diagnostic 38 worker and 22 thread starts |
| Textual CSS fastpath | Original ancestor-rule ratchet passes at limit 268; exact count not separately captured |
| Screen preimport payload | 554 / 558 modules; 426108 / 426199 LOC |
| Largest preimport route | Library 132417 / 135111 LOC |
| First-send warmup | 93 / 94 modules; diagnostic 57485 LOC |

The preimport payload has 91 LOC headroom and the largest route has 2694. Relative to the historical final-five report, the current frozen bytes add 12 measured preimport LOC; this report records the new result rather than inheriting the older 426096 LOC. All original limits remain unchanged. The three warnings are the original UI-ready, preimport and warmup headroom reports. These are five bounded structural/census checks, not a destination-tour latency test or a full test sweep.

The run used HEAD `b1108874cab28d23f660aa22c11c863466a84a8e` plus the frozen, uncommitted PR fixes. source-before.json and source-after.json contain matching SHA256 manifests for 3450 tracked/untracked product, Performance-test and relevant bootstrap/config paths. source-proof.json confirms identical file sets, hashes, HEAD, product diff hash and observer-driver hash across the run. The complete manifest SHA256 is `dda775a8ba7079a1fe74e5260a7ddee8ac3cf5bfa91d59be96cacba3b3dd86ee`.

Command, from `/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook`:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /private/tmp/mixins-pr3346-frozen-final-five/probe.py pr3346-frozen > /private/tmp/mixins-perf-pr3346-frozen.log 2>&1
```

The observer driver is byte-identical to the previous five-guard probe (SHA256 `0e671fdc3e15b81d7d06a185610de510d45640aedfa8c195131656e4c0d4dd1c`). It invokes only the five exact original pytest nodes, in the original order, and observes return values from census helpers without changing them. The four guard-file hashes are identical to those in the previous five-run artifact. commands.json records exact nodes, original pytest arguments, unchanged-file proof and output paths.

The probe returned 0. Each original subprocess census uses synchronous subprocess.run, and the mounted CSS test completed teardown. The post-run process audit found no matching remaining probe/app-census Python process; process-proof.json records the bounded metadata-only check. Its default sandbox process query was unavailable, so the successful audit used approved read-only process access.

artifact-files.json lists every report, command/probe, source/process proof, test log and four observed census JSONs, with exact absolute paths and hashes. The generated private pytest profile `/private/tmp/mixins-perf-pr3346-frozen-tmp` is excluded from delivery evidence. No new app or performance run remains active, and this agent will make no further source/task changes.
