### Spec Compliance

- ✅ Spec compliant for Task 1. Reviewed immutable `f843ca811f01da6c39d903b6cd7328d68d50416f..07e7c23cb58b39928af3393d3e447e172d804e86`: ten source/test files. `session.py:4835-4850` now ignores only cursor/selection navigation while retaining session, incarnation, exact consumed revision, widget generation, authored edit serial and segment identity. No unconditional clearing or text-only ownership comparison was added.
- ✅ The mounted held-readiness matrix adds actual Left and Ctrl+A navigation and proves that only navigation changed (`Tests/UI/test_console_runtime_ownership.py:3305-3322`). Its existing assertions verify durable consumed metadata, one literal USER message/provider call, source composer/workspace preservation, focus, drained work and switch-away/back persistence (`:3372-3422`). The same-text authored replacement remains protected (`:3325-3340`).
- ✅ Startup repair defers a genuine failure-only module at its two use sites (`console_chat_controller.py:27523`, `:27719`) and prevents its eager return (`Tests/Performance/test_ui_ready_module_census.py:179`). Saved census evidence is 1034 RED → 1033 GREEN at the unchanged 1033 limit (`census-red.log:62`, `startup-green.log:114`). Budget constants, snapshots, historical QA, skip/xfail markers and production hook semantics have no changes in this package.
- ✅ The initially missing primary-to-child shared bridge confirmation run is now supplied separately: `shared-child-closure-green.json:1` records the exact node, head `bc7c2dde9febfd93cbc3bc115c8cbc61c3ee752b` and exit 0; its log records **1 passed in 2.65s**. This is distinct from the 181-test child/wake allowance coverage.
- ⚠️ The complete original freshchat inheritance, authority, persistent handoff and startup feature contracts live primarily in unchanged code and are not independently established by this task diff. This approval is the Task 1 gate; controller-owned whole-branch review remains required. The package changes neither those implementations nor their authority boundaries.

### Strengths

- The production fix is small and uses the existing composer commit path. Focused contract inspection confirms a random per-widget generation epoch (`console_composer_bar.py:692-702`) and a generation/revision-fenced captured commit (`:4124-4209`), so navigation tolerance does not permit a replacement widget or later same-text authored draft to be consumed.
- The fixture repairs retain behavioral assertions: actual provider delivery precedes cursor send assertions (`test_console_composer_cursor.py:492-506`); blocked custody must contain the exact original and restore it before checking composer text (`test_console_native_chat_flow.py:5136-5146`); the closure test checks distinct local/network groups and scanner identity rather than a stale queue count (`test_console_interaction_boot_closure.py:82-99`).
- The isolated skill snapshot case keeps the real button handler, prompt dispatcher, captured runtime request and exact later suffix assertions (`test_console_send_draft_snapshot.py:353-396`). The separate new empty-hook pair checks real unchanged/stale admission (`Tests/Chat/test_console_hook_admission.py:142-181`). The unchanged hook implementation applies the stash fence even to ready empty inventories (`UI/Console_Modules/hooks.py:252-266`), supporting the stated seam boundary.

### Issues

#### Critical (Must Fix)

- None found in Task 1.

#### Important (Should Fix)

- None found in Task 1.

#### Minor (Nice to Have)

- Inherited test cleanup remains noisy: `ownership-green.log:36-44` reports four invalid-escape SyntaxWarnings and FD growth **489**, and `composer-repair-green.log:1652` reports FD growth **490**, both above the existing 200 guard. The prior scoped review already records this resource debt. Retain this qualification; these logs do not locate a production leak, and this task adds no suppression.
- Combined skill-await plus actual hook refusal remains a coverage limitation: the forwarding seam at `test_console_send_draft_snapshot.py:356-365` intentionally bypasses the later unchanged-stash gate. Separate real hook unchanged/stale controls are appropriate for the scoped snapshot test, but neither establishes the entire combined mounted flow. The report correctly states this boundary; a future focused integration regression could check refusal and preserved draft through both owners.

### Evidence and Scope Checks

- Read the supplied diff once, recovering portions truncated by tool output. Two changed-file functions were completed because their hunks ended mid-function: mounted matrix `test_console_runtime_ownership.py:3205-3425` and session consumer/draft save ordering `session.py:4819-4947`.
- Named outside-diff risks checked: composer widget/revision commit contract (`console_composer_bar.py`); the claimed real empty-hook stash fence (`UI/Console_Modules/hooks.py:157-266`); collection-time configuration potentially escaping profile isolation (`Tests/conftest.py:14-53`, `:1181-1295`). Existing bootstrap redirection/profile protection supports the added fixture markers and collection import.
- Independently rehashed all ten current source/test files against `task-1-source-hashes.json`; every SHA-256 matched immutable head `07e7c23cb58b39928af3393d3e447e172d804e86`. No Git command, test, application launch, index/HEAD/branch change or source edit was performed by this reviewer.
- Checked saved final logs: ownership **78 passed, 1 inherited XFAIL, 5 warnings**; affected start/compaction/RAG/trace **181 passed**; hook admission **22 passed**; snapshot/readiness **4 passed**; closure/environment **50 passed**; gateway recovery **2 passed**; persisted-ready qualification **7 passed**. Counts overlap and are not summed. The composer 62-pass run and startup 18-pass run retain their original failures; subsequent focused receipts qualify the repaired nodes.
- Checked baseline failure logs named by the report: closure queue-count failure; environment/config selection errors; captured-send missing `cached_context_window`; mounted fixture failures. The saved archive receipt selects base `f843ca811f01da6c39d903b6cd7328d68d50416f`, with tests run from that archive. No baseline test was rerun.
- Fatal Ruff log says `All checks passed!`; working and exact committed formatter receipts report exit 0 with empty logs; whitespace receipt reports exit 0. No full-suite, blanket lint/security, production leak cleanup or complete live-user qualification is inferred.

### Assessment

**Task quality: Approved.**

The navigation resurrection defect is closed with the requested ownership fences, and equivalent failure-only import shedding restores the unchanged boot ratchet. The expanded mounted controls and retained RED/GREEN evidence support the fix; inherited cleanup and explicitly isolated combined-flow coverage remain qualified above.
