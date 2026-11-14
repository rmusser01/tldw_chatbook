# PR3315 integration evidence

Source98cd750928af894f5fe8c1044d07ed96dab20211 was rebased onto dev3e55f1f3aa1396ab0ae2fe29a19f397cda88c239. The production delta remains fourteen lines of Tools inspector guidance. Current-dev ownership, exact search ranking, design tokens, recovery guards, egress checks and native-worker retirement remain intact.

## Targeted results

The frozen234-case selection covered endpoint probing, local discovery, discovery journeys, Tools hub and changed Settings hub paths:229 passed,5 failed (r-acf94357). Two stale discovery expectations also failed on the exact base. They now assert all twelve ordered IP/localhost candidates and both llama.cpp probe URLs while retaining the no-Ollama-fallback assertion. The whole discovery module then passed59 cases; the appearance save passed with its real persistence assertion and a bounded30-second completion wait (r-0210995b,60 passed with2 remaining schedule failures).

The schedule test retains180x50 and100x30 viewports, painted text, enabled button labels, app-owned scheduler projection and persisted configuration. Earlier mouse attempts sometimes missed or failed to activate after a reported initial hit. A logging-only run passed both; no failing event trace established a production cause. Pilot.click documents an initial-hit result and snapshots coordinates before yielding. The final test scrolls/requeries and focuses the current enabled visible button, then drives Textual's real Enter binding through Button.Pressed and the real persistence worker. Both cases pass (r-6d778999,2 passed,0 errors/skips,source hashes unchanged). Neighboring failure-path mouse tests are unchanged and passed in the original run. This is keyboard-recovery evidence, not proof that no production mouse race exists.

All234 originally selected nodes have passing evidence across the frozen run and affected reruns; this is not one clean full-suite run. AST comparison confirms only the four repaired test functions changed after the original run. All other passing definitions and production code are unchanged.

Census checker passed with the exact1268-entry dev prefix plus8 source nodes, floor1276;83 focused census contracts passed. Settings measured35740 lines; other ratchet rows are unchanged. Eight changed Python files parse. Scoped Ruff comparison found14 existing base/current findings with0 introduced; final fatal/error checks and git diff --check passed. Receipts and review are in sibling pr3315-resync (pr3315-census-isolated.xml,pr3315-scoped-static-final.json,pr3315-repair-scope.json and the r-* directories).

Independent initial integration review found no Critical/Important findings at5d7cad1c8598495e398cf25cd976352c8689b23a. An additional read-only event-path review confirmed focused Enter exercises the same Button.Pressed/persistence path. Final immutable-head review remains required before publication.

ADR126 andADR240 cover the existing isolation/Settings ownership boundaries; no new ADR is required. Hosted CI and size limits are waived by the user. No full sweep, hosted/performance/native/live-service qualification, release or main promotion is claimed. Drafts and PR3267/3297 remain excluded.
