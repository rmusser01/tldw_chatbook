# SDD ledger — plan: Docs/superpowers/plans/2026-10-03-console-pr2995-review-and-merge.md

User authorized rebase, all review repairs and merge on 2026-10-03. Old draft final-review cap is superseded by this explicit new request.
Rebase complete: ce918c467898eb24feff216bced45b779bf310c6 → f843ca811f01da6c39d903b6cd7328d68d50416f; pinned dev 01a2020981c6197e5cd9945e5287567ad977edfe. Original commits retained in private recovery bundle.

## Preflight consistency scan

| Tasks | Producer/consumer | Finding |
|---|---|---|
| Task1 vs controller rebase | Rebased UI/runtime contracts feed ownership comparison | One documentation append conflict resolved; production changes merged automatically and need qualification. |
| Task1 vs external review | Committed repairs feed current-head bot review | Future feedback handled only when posted and verified. |
| Task1 internal | Mounted navigation regressions vs ownership fence | Authored edits require identity/revision checks; cursor navigation must not count as editing. |
| Task1 internal | Import deferral vs boot ratchets | Genuine deferred imports preserve behavior; limits and guards remain unchanged. |

Ruling: Reuse the existing integration task and approved ADRs for this follow-up — defects restore existing contracts — if wrong, task scope or an ADR will need revision.
Ruling: Repair the startup breach by deferring demonstrated boot cost — ADR-097 forbids silently raising the limit — if wrong, a feature path could need an additional lazy import correction.

Task1: pending; BASE f843ca811f01da6c39d903b6cd7328d68d50416f.

Heartbeat: finish-console-pr-2995-review-and-merge created ACTIVE for quiet delayed Qodo/CI follow-up; disable after confirmed merge and bookkeeping.
Rebase receipt confirms pinned dev ancestry and zero changed historical QA paths.

Task1 scope clarification: verified failures in additional covering boot/composer tests belong to the existing all-issues acceptance criterion; qualify immutable base before repairing stale expectations. All eleven derived checks completed with exit0 on the working repair tree.

Controller baseline replay: exact original21 nodes passed, exit0, 65.73s pytest / 72.26s process; production fingerprints unchanged during run. Evidence baseline21.json/log.
Task1 preliminary results: complete ownership78pass/1inherited strictXFAIL/5warnings; start/compaction/evidence/trace181pass; boot census1033/1033, unchanged limits. Additional covering fixture repairs remain pending.

Ruling: Isolate the older skill-await snapshot test at the hook-review seam while retaining the real prompt dispatcher, runtime request and original draft assertions — current hook admission deliberately refuses a changed captured stash and has separate unchanged-owner controls — if wrong, combined skill and hook interaction coverage could be incomplete and need another regression.

Task1 implementer DONE_WITH_CONCERNS: /root/pr2995_ownership_perf_fix, commit07e7c23cb58b39928af3393d3e447e172d804e86. Full report retains47 command receipts; source-only ten paths. Task review pending /root/pr2995_task1_review, immutable package review-f843ca811f..07e7c23cb5.diff.
Latest dev0001eba40419859ce39ed4952f0f8df7b40639bd differs pinned01a only by three Backlog tasks; no application/test/script delta.

Task 1: complete — spec compliant and quality approved by /root/pr2995_task1_review; no new Critical/Important finding. Minor inherited warnings and explicit hook seam qualification retained. Source07e maps byte-identically to rebased bc7c2dde9f. Missing genuine-child bridge regression separately qualified at bc7c2dde9f: one pass, no warnings. Initial setup-directory error retained separately.
Whole-branch review dispatched /root/pr2995_whole_branch_review on gpt-6-astra, immutable range0001eba404..bc7c2dde9f. Source/docs package95paths; historicalQA1524paths byte-unchanged versus ce918c4. Latest-dev Backlog IDs/readability guards bothexit0.

Final whole-branch review complete by /root/pr2995_whole_branch_review on bc7c2dde9f. No Critical/Important finding; two actionable Minor issues: prevalidation duplicate grant clearing and inaccurate child-tool guide. Task2 final fix wave BASEbc7c2dde9febfd93cbc3bc115c8cbc61c3ee752b, pending. User all-issues scope includes both.

Task2 implementer DONE: /root/pr2995_final_review_fix commit0d2e581ba96ef00adb02e2c800cd449582f8a629, exactparentbc7c2dde9f. Four genuine RED failures; complete shutdown owner36pass/no pytest warnings; fatal Ruff/formatter/diffcheckexit0. Two owned source/doc paths only. Scoped independent review pending /root/pr2995_final_scoped_review; package review-bc7c2dde9f..0d2e581ba9.diff.

Task 2: complete — scoped spec PASS/quality APPROVED by /root/pr2995_final_scoped_review, both findings addressed, no new defect. Latest dev81c7c94f48 rebase yields872efa5bdc; all78owned feature source/test blobs identical. Upstream backup/startup seam qualification running.

Latest-dev qualification at872efa5bdc: affected backup42pass/no warnings, startup/import13pass/3unchanged headroom warnings. Feature78blobidentity retained after upstream PR2994. All local source/review gates passed; external Qodo/current-head checks pending.
