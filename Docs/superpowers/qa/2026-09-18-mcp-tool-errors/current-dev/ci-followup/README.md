# Approved PR2716 CI follow-up

Owner approved the gallery at `19ceab4dc3`. Production, native-runner and visual
source hashes still match that approval. This follow-up changes only test startup,
the bounded CI job budget, and evidence/docs. No new visual approval is required.

[Run 35545242345](https://github.com/rmusser01/tldw_chatbook/actions/runs/35545242345)
passed all 1,152 main cases in 14m45s. Its admission invocation finished with
122 passes, one existing expected failure and one failure in 4m59s; the combined
job also exceeded its 20-minute budget.

The synchronous-construction watcher test could finish while its seven-second
splash was still active. CI recorded splash closure, runtime disposal, then a
late Console mount whose store was already gone. An explicit mounted-screen
precondition [fails against that setup](ci-startup-red.txt). Simply waiting for
Console is wrong for this test: Console mount creates the Canvas controller,
whereas the original assertions require Canvas to remain unwarmed.

The correction selects real Home startup, disables splash through a scoped
delegated getter, joins the actual startup task, and waits for the Home header.
All original watcher, uncreated gateway/controller, disable-latching and disposal
assertions remain. No shared-profile setting is written. Independent read-only
review verified the final correction and the scope of the CI budget change.

The serial job now has a 30-minute maximum, accounting for both measured pytest
steps plus setup. Test targets, per-test deadlines, runner count, dependencies,
required context and prerequisite relationship are unchanged. ADR-103 applies;
no new architectural decision. The [old timeout pin fails](ci-budget-red.txt)
before the workflow change; all [26 CI contracts pass](ci-contracts.xml) after it.

The [124-case affected admission group](admission.xml) reports **123 passed,
1 existing expected failure**. The [isolated lifecycle regression](isolated.xml)
also passes. Seven [artifact checks](preflight.txt) pass in the sandbox; the
Mermaid check initially cannot fetch its pinned inputs, then
[passes with network access](mermaid.txt). [Static comparison](static.json)
adds no diagnostics, and changed ranges pass formatting. No full local suite.

[Receipt](receipt.json) and [export manifest](export-manifest.json) preserve
counts, source hashes, review disposition and evidence provenance. Current-head
remote CI remains required before merge; owner approval is already recorded.
