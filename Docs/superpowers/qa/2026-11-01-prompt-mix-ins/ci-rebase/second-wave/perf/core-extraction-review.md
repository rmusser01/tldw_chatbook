# Independent scoped review: core five extractions

Base: ba1cafc55dd5ac9b6b1f066c936206e652ed6b65. Candidate: current uncommitted shared tree, reviewed 2026-11-02 after implementer reported these product files settled. Scope: controller -> console_agent_reply; store -> console_settings_persistence; persistence -> console_chat_fork; trace service -> trace models; display state -> console_memory_banner. Read-only checkout review, no helper agents and no tests repeated.

## Strengths

- Independently compared 37 moved definitions against the base, normalizing only live-host attribute qualification and the helper's local host import. Thirty-five match exactly. The remaining two differences are the approved required work_origin helper argument (the public method keeps and explicitly passes its original default), and docstring indentation in the settings drain. All executable moved bodies match.
- Every retained class wrapper keeps its original parameter list/defaults and decorators, including the controller's checked_generation, remote-session close and generation retirement ordering. Every wrapper forwards each input exactly once under the same name. Await/return behavior remains appropriate to its original async/sync method.
- Agent and settings helpers look up original owner globals at use time, including callbacks and code after awaits. The fork helpers use a local live original persistence-module reference. This retains the documented monkeypatch seams without copied namespaces or snapshot globals.
- Settings drain task creation, asyncio.shield, lifecycle retention/retirement, incarnation and durable CAS checks move unchanged. Mix-in acceptance/usage state and failed-value retry conditions remain unchanged. Fork configuration and input validation remain before existing persistence/transaction owners; receipt authority is not moved into a value helper.
- Trace tree algorithms/classes move without executable-body changes into the existing pure model owner. Service aliases retain the same class/function objects; pure sequence state does not acquire DB/repository responsibilities. Memory banner validation and exact-anchor behavior also match their original implementations.
- Inspected pending/preparing/controller source witness tables and references. The moved orchestration/validation functions are not pre-existing stock callback entries that could now hide an extracted callee. The live _run_agent_reply test observer still sees the retained decorated public wrapper entry; auxiliary routing test explicitly follows the extracted implementation as well.

## Findings

Critical: none.
Important: none.
Minor: none.

No speculative redesign or unrelated baseline cleanup recommended.

## Evidence and limits

Independent proof artifacts: /private/tmp/mixins-core-review-ast.py, /private/tmp/mixins-core-review-ast.json, /private/tmp/mixins-core-review-ast-differences.txt. Wrapper forwarding was separately checked against original parameter names; no omissions/renames/duplicates found. Source diffs and relevant callers/tests were read, including post-acceptance continuation overrides and current-source tables.

Read implementer evidence under /var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/mixins-ci3346-core-extract-i4xaa7gn: final-controls.json/log reports 11 passed, including original agent request/remote-session controls, real preferences flush/reopen/cancelled waiter, store size and fork late monkeypatch seam. Existing broader cohort failures are qualified by the implementer; this review does not relabel the earlier aggregate all-green. owned-agent.json references an ephemeral test source no longer present in the checkout. The implementer supplied a labeled reconstruction in owned-agent-test.py; this review relies on the final original test nodes instead of treating reconstructed bytes as original execution evidence. The supplied source-fence-audit.md and final move proof agree with this independent source inspection.

## Assessment

These five mechanical extractions are ready for the parent's combined validation stage. No source fix requested. This is not a merge approval for the entire feature: final combined performance/census checks, finalized evidence and other parallel extractions remain parent-owned.
