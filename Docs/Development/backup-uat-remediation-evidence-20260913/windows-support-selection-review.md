# Windows support selection independent review

Disposition: approved; no actionable findings in the two-file runner/workflow delta.

The existing support-diagnostic dispatch now reaches exactly the remaining nine native cases: three mounted screens, two Console profile-open cases, the 1800-file case and three unsafe-parent ACL cases. Independent collection returned exit 0 with exactly those nine nodes (/private/tmp/uat-support-selection-independent-collect.log). It no longer silently runs the entire support selection. Full dispatch still schedules the complete support selection; no existing support tests were removed. Full/replacement include the new Persona/Evals regressions, and full/support include the Console lifetime cases.

Only the complete support selection receives a 120-minute product subprocess ceiling and 135-minute job ceiling. The 15-minute difference accommodates the unchanged native phase (ceiling 10 minutes) plus setup/receipt/upload overhead; this is a bounded test-run observation allowance, not product performance acceptance. Other selections retain 80/90 minutes, including support-diagnostic. Individual product/child deadlines and assertions are untouched.

The actual matrix selection is passed directly to the runner and used in artifact names, so support-diagnostic receipts are distinguishable from complete support receipts. Upload remains always-run and uses the existing failure-preserving artifact directory. No new workflow input, product mode, permissions or runtime guard is introduced.

Independent existing CI queue-contract suite: 17 passed in 0.74 seconds, exit 0 (/private/tmp/uat-support-selection-independent-contract.log). Existing dependency/pytest cleanup warnings did not fail checks. No source edits by reviewer.
