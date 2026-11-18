- **I1 — Fence completion to the current Workbench view. — ADDRESSED.** `tldw_chatbook/Widgets/Console/console_prompts_modal.py:372` now requires mounted Mix-ins mode, captured generation and exact rendered-view identity. `_refresh_mixins` captures editor identity/raw draft and checks both before either success or error replacement (`:388`, `:402`, `:416`); it checks generation again after awaited removal and after mount (`:421`). Mode mounting, including Back, and unmount invalidate older generations (`:731`, `:633`). Inspect, action, Save and Preview completions check their captured generation/view before publication (`:462`, `:517`, `:564`); editor mounting checks currentness before and after awaited mutations and before focus (`console_prompt_mixins_view.py:256`). The held projection regression tests both delayed success and error after Browse navigation (`Tests/UI/test_console_prompt_mixins.py:586`); six additional held inspect/Save/Preview success/error cases preserve the selected Browse widget (`:700`).
- **I2 — Preserve edits made while Save is pending. — ADDRESSED.** Save captures the submitted raw tuple and editor identity (`tldw_chatbook/Widgets/Console/console_prompts_modal.py:525`). After the exact Library write, it rejects stale view/editor ownership, advances the editor record to the returned saved version and its baseline to the submitted tuple, then leaves later raw typing mounted and dirty (`:532`–`:547`). Only an unchanged submitted draft refreshes away the editor (`:548`). `_ConsolePromptMixinsFlow.save` returns the actual saved record after existing source/session/incarnation/preference checks and reconciliation (`tldw_chatbook/UI/Console_Modules/prompts.py:1182`, `:1225`); the retained editor receives the resulting preference revision for a second conditional Save (`console_prompts_modal.py:539`). Rendered row projections remain immutable. The regression changes name and placement, holds Save, types newer Instructions, verifies the first body at version 2 and the surviving dirty editor, then saves the newer body at version 3 (`Tests/UI/test_console_prompt_mixins.py:632`).

## New Breakage in the Fix Diff

- None found. The fix preserves existing exact source/session/incarnation/revision authority and expected-version writes; it adds UI publication ownership and a returned saved record rather than replacing those checks. Original-versus-current mix-in capture/retry behavior is untouched. No style/layout changes require a new visual review.

## Out-of-Scope Observations

- None newly identified. The prior generation-variant count-copy follow-up remains with Task9; the previously qualified baseline failures and broader runtime/lifetime verification limits are unchanged and are not reclassified as passing here.

## Checks and Evidence

- Reviewed task-8-brief.md, task-8-review.md I1/I2, appended task-8-report.md fix evidence, review-context.md and supplied `review-3c77bbbdd7..0dcf71c2e5.diff` (commit/stat/full diff). Fix base: `3c77bbbdd701052bb778eaf1bc09fea7b838c8ae`; head: `0dcf71c2e5a3e93aecf1618a62f3c423a7b35f20`. Supplementary source reads were limited to the changed completion code, its mode/Back caller, shared editor raw-draft contract and existing source/currentness checks needed to verify the findings. No git commands were run.
- Inspected covering test source and existing logs, rather than rerunning reported tests. `task-8-evidence/fix-round-1/covering-final.txt` reports `19 passed in 27.50s`; final `held-completions-final.txt` reports `9 passed, 10 deselected in 6.85s`. These cases cover the reported races and the added success/error controls; their assertions match the fix paths. The final nine-case run follows the reported immutable-rendered-projection adjustment.
- Inspected `fix-round-1/lint-delta.json` (`[]`), `commit.json` and the fix source manifest. Independently recomputed SHA-256 of all five manifest files: every file matches `fix-round-1/source-hashes.json`. The existing covering evidence is therefore qualified to the inspected fix bytes. No claim of independently rerunning its tests or static checks is made.
- Read-only product/index/HEAD/branch review; only this report was written. No subagents, suites, live cloud calls or additional layout review.

## Verdict

**Fix round: All findings addressed, no new Critical/Important breakage.** I1 and I2 are closed for this scoped round.

Skipped repeat tests, suites, live-provider and additional layout checks; existing Task9 follow-ups and baseline failures remain outside this approval.
