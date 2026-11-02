# Final five bounded guards after UI24/UI26 fixes

Result: **5 passed, 3 existing headroom warnings in 31.98s**. Run on HEAD `7ac214599d4969a4d1a57db8bef41ff02eb433c9` plus the approved uncommitted fixes, after the core owner explicitly confirmed source frozen and all runtime tests stopped. No concurrent app tests were announced during this run. All owned processes are now stopped.

| Guard | Result |
| --- | --- |
| UI-ready module census | 1031 / 1031 modules |
| Boot worker/thread census | Original allowlist passes |
| Textual CSS fastpath | Original ancestor-rule ratchet passes (limit 268) |
| Screen preimport payload | 554 / 558 modules; 426096 / 426199 LOC |
| First-send warmup | 93 / 94 modules |

Largest route: Library 132405 / 135111 LOC. Warmup diagnostic LOC: 57485. No limits, exemptions, source snapshots, or assertions changed. The exact CSS count was not separately captured; the original test's bound passed. These are five bounded structural/census checks, not a repeated destination-tour latency test or a full test suite.

Command:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /private/tmp/mixins-perf-probe.py 7ac-final > /private/tmp/mixins-perf-7ac-final.log 2>&1
```

The previously used probe runs the original five exact pytest nodes serially and captures return values of their census helpers without modifying them. The original tests themselves enforce all limits. `source-proof.json` records the product diff, relevant source/test hashes, and confirms they stayed identical from the start of this run to completion. This agent changed no repository files.

`artifact-files.json` enumerates the exact report, source proof, driver, test log, and four observed census JSONs. Ephemeral pytest profiles are not claimed as report artifacts.
