# Search settings PR integration

The search-settings work is based directly on `dev` commit `86a8054edb0853fb10997eec0cec2a7db5f99f1c`. Only the task commit was transplanted; the original branch's 37 unrelated documentation commits are excluded.

## Integration decisions

- Preserve `dev`'s Network save action, availability-aware Personal Profile footer, privacy/Canvas controls, snapshot preferences, and write-time config publication.
- Preserve shared TLS verification and Tavily bearer-header credentials while resolving saved keys on each request. Keep request failures at the shared credential-safe dispatch boundary.
- Replace obsolete raw-editor helper tests with mounted retained-session coverage. Backup operations serialize while controls are busy; repeated confirmed loads read the current backup, and typing during a read wins over its completion.
- Recognize all 30 current Settings categories. The new Web Search field/action containers use the Settings prefix, so their styles load with the existing lazy Settings stylesheet. Boot-parsed CSS is unchanged from `dev`: 805,669 bytes on each tree.
- Renumber the shared-default task from TASK-32187 to TASK-32193. A fresh sweep found 32187 claimed by `origin/fix/boot-css-ratchet-paydown`; 224 remote refs and 32 worktrees were checked, with maximum 32192. Tasks 32188–32191 retain their IDs.

The independent integration review found and then verified the repair of a duplicated/unavailable Personal Profile footer action. No remaining integration findings were reported.

## Baseline limitations

A broader integration run also exercised unrelated filesystem-tool tests. Eighteen failed with “Private scratch space is unavailable; the tool was not run.” All eighteen reproduced when the tested provider module was replaced, through a temporary pytest plugin, with the exact unmodified source from `origin/dev`. That probe passed one neighboring case. This PR changes only search schema/default wiring in that provider; it does not repair the existing filesystem executor test setup.

The boot-CSS ratchet also fails on the existing `dev` total of 805,669 bytes versus its 804,000-byte limit. This PR adds zero boot bytes and does not change the limit or pinned snapshot. The CSS bundle and lazy screen stylesheets reproduce from their sources.

The existing Requests dependency warning remains. No full repository sweep, real-provider request, or participant usability study was performed. Screenshot evidence uses mounted Textual with production stylesheets and local Menlo rasterization.

## Governance

Existing ADR-012 (provider credentials), ADR-032 (local tool permissions and shared search default), and ADR-033 (Settings commit models) govern the implementation. No new runtime, dependency, storage, or permission boundary was added during PR integration. Task records contain the earlier feature-specific review and regression evidence.

## Final verification

- Final search/settings gate: **505 passed, 3 skipped** in 214.85 seconds. Covers basic/deep tools, Console launch, backend request/credential contracts, config snapshot/persistence/delete/encryption, raw/guided Settings lifecycle, footer/profile actions, navigation/category sweeps, screen-state storage, and import provenance. The skipped cases require explicit live-provider opt-in.
- Search-specific LocalToolProvider checks: **20 passed**, 215 unrelated cases deselected.
- New modules/tests pass full Ruff and formatting; modified legacy Python files pass scoped syntax/undefined-name checks. All 34 changed Settings formatting ranges passed. Backlog filename/frontmatter IDs are unique, CSS generation matches, and diff hygiene passed.
- Diagnostic inventory matches: 594 owners, 1,351 TASK-492 calls, 55 TASK-31551 calls, 7,650 TASK-494 calls, and 12 sink files. Reviewed the 22 removed and nine added search diagnostics; raw response/URL/error content was replaced with closed classifications. The removed SearX URL log also removes one legacy path-privacy candidate.
- Mounted 80×24 and 120×35 captures were refreshed on the rebased tree and rasterized; compact guided test controls and raw-draft recovery were inspected.

## Qodo review follow-up

All three findings are addressed: Google-style documentation covers the three
backend metadata classes and every public guided-settings API, and raw saves
distinguish a committed file from subsequent refresh failure. A credential-safe
post-commit exception carries the exact available snapshot and backup. Settings
adopts that baseline, preserves newer edits, and reports Saved to disk with
restart/copy guidance. If the snapshot read itself failed, Save remains blocked
across navigation and validation until explicit Revert establishes a baseline.

Six real-file fault cases reproduced snapshot/runtime/view refresh failures;
two mounted navigation cases reproduced unsafe adoption of an external edit
after snapshot loss. Each failed before its corresponding fix. The independent
reviewer confirmed the recovery-fence repair and found no remaining issues.
The related search/settings/config gate passed **201 tests**. After the final
recovery refinement, all **43 raw-draft/snapshot tests** passed. Ruff, formatting,
diff hygiene and the complete derived-artifact preflight passed. The existing
Requests warning and `dev` boot-CSS breach remain unchanged; the latter is
tracked by the separate PR #2560. No additional ADR, full sweep, or live provider
request was needed.
