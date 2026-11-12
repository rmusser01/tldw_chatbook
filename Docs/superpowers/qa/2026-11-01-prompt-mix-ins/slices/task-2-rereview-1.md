### Finding Verdicts

- **I1 — BOM-prefixed Mix-in Markdown falls through to raw snippet logging — ADDRESSED.** `tldw_chatbook/Prompt_Management/Prompts_Interop.py:806-810` removes exactly one optional leading U+FEFF into an envelope-only view and uses that same view for both discrimination and full framing validation. Valid BOM envelopes reach the strict portable decoder (`:815-821`); malformed framing/JSON raises the bounded record error (`:812-817`) before legacy parsing. The unchanged UTF-8 reader invokes this actual parser (`:1097-1099`), and its ValueError branch returns/logs the bounded error then continues before mutation (`:1108-1117`). `PromptChatbookRecordError` supplies fixed text in `prompt_chatbook_record.py:50-52`.
- **Literal preservation — verified by code and covering regression assertions.** Prefix removal acts only on the transport string; it neither strips the JSON line nor edits decoded body/definition strings. Legacy parsing still receives original `content`. `Tests/Library/test_prompt_export_roundtrip.py:554-569` constructs both a transport BOM and a separate literal body BOM; `:587` compares the complete parser result, and `:619-623` compares stored type, exact body and exact definition after real file import.
- **Malformed refusal and privacy coverage — verified.** The two four-case parametrizations (`Tests/Library/test_prompt_export_roundtrip.py:572-597,600-636`) cover valid, bad JSON, missing closing fence and appended legacy sections. File bytes are checked for EF BB BF; malformed imports require a bounded failure and zero active destination records; both paths assert absence of body sentinel, raw-snippet and legacy-parser logs. No parser, file reader, storage or mutation owner is mocked in these new cases.

### New Breakage in the Fix Diff

- None found. The product change is limited to consistent envelope-prefix handling; the existing strict framing, codec and legacy-parser inputs retain their behavior. Backlog edits document the fix and its scoped evidence without claiming task completion.

### Out-of-Scope Observations

- None newly identified. The previous report's five inherited Chatbook digit-sentinel failures, inherited static debt and intentionally unavailable server snapshot capture remain outside this fix and are not cleared by this review.

### Checks and Evidence

- Read the supplied fix diff once, plus task brief, previous I1 review, appended implementer fix report and binding review context. Inspected only focused parser, actual UTF-8 file-import/error-handler continuations, bounded error class and test fixtures needed to verify I1.
- The appended report names the two new regression nodes, records eight RED failures, and supplies the covering two-file command and GREEN result: **114 passed in 2.14s**. The diff's assertions substantiate its preservation/refusal/privacy claims. Formatting, diff-check and exact-base inherited Ruff comparison are reported separately; these execution claims were not independently regenerated.
- No additional test was warranted: the changed statements and new real-import cases answer the scoped BOM defect directly. No suite, git command, subagent, product/index/HEAD/branch mutation, push or merge was performed. Only this ignored review report was written.

### Verdict

**Fix round: All findings addressed, no new Critical/Important breakage.** I1 is closed.

Skipped rerunning the 114 tests, full-suite/live-server/UI checks and independent static-evidence regeneration. No new material risk found in this fix; earlier task-level qualifications remain in force.
