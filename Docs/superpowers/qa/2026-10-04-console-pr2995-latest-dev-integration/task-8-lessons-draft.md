## An already-locked helper can still hide storage work

During PR #2995's integration with committed Close, an exact-record helper removed a recursive acquisition of the registry lock. Sixteen phase controls and the final58 startup/navigation checks passed, but independent review found that the helper still called AgentRunsDB.get_run under that lock. A delayed SQLite read could therefore hold up the same registry lock that Close and cancellation needed. A lock-depth check alone would have missed it.

Trace calls through storage getters and park a real read while the actual owner-thread Close runs. Keep storage observations outside the registry section, then atomically recheck exact record and current runtime ownership before deciding. Copying a row does not make independent database writers transactional with that decision; characterize that limit and separately prove the final pre-save check refuses a terminal source. Evidence: task-8-review.md and task-8-fix1-report.md in the final PR #2995 integration QA directory.

## A Close refusal needs a live source witness before Close

PR #2995 initially tried to restore a no-owning-turn fixture by stripping the primary cancellation and assistant registrations after arming a creation card. The production primary-source predicate requires those registrations, so the source was already stale before Close. A later denial could have passed without exercising the Close fence.

The corrected two no-parent phases use actual surviving-child rows and trusted actor context, with a positive exact-record/source-live witness at the original parked-card or enrichment barrier. Owning-parent and primary grant cases retain their original primary setup. Preserve those positive witnesses alongside the denial, outcome and cleanup assertions; reaching a card alone does not prove that its execution authority is still live. Evidence: task-8-surviving-child-fixture-proposal.json, task-8-surviving-child.json and task-8-review.md in the final integration QA directory.
