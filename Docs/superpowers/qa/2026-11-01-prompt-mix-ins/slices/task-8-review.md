## Spec Compliance

- ❌ Issues found. The normal Workbench/runtime integration meets the requested shape, but asynchronous UI completion is not fenced to its rendered mode/editor revision. Two mounted probes confirm violations of stale-callback and dirty-draft preservation requirements (findings I1–I2).
- Reviewed supplied package `review-a1fed4545b..3c77bbbdd7.diff`, BASE `a1fed4545bfc3b438d356b27503afdd6f5f611bd`, HEAD `3c77bbbdd701052bb778eaf1bc09fea7b838c8ae`, against task-8-brief.md, current rulings, review-context.md and task-8-report.md. The first combined documentation output was truncated; the affected leading diff portion was recovered from the same package, not regenerated.
- ✅ Existing `ACTION_PROMPTS` and Browse landing survive; Mix-ins is an additional Workbench section with enabled count, not a permanent composer manager (`console_composer_menu_modal.py:133`, `console_prompts_browse.py:102`, `UI/Console_Modules/prompts.py:1837`). Ordinary modes remain registered (`console_prompts_modal.py:192`).
- ✅ Exact session/incarnation/binding/source-object and revision checks precede preference mutations; Add delegates to `enable_reference`, toggles/moves/removals use runtime CAS. Placement observations reconcile once after reads, preserving disabled flags and last-used facts (`UI/Console_Modules/prompts.py:972`, `:996`, `:1112`, `:1182`).
- ✅ Shared literal editor validates Name/Instructions at Save; version is passed to source save; ordinary dirty close and failed-save paths retain the editor (`console_prompt_mixins_view.py:248`, `console_prompts_modal.py:456`, `:1962`). The pending-save exception is I2.
- ✅ Preview uses the injected exact builder once, with read-only capture owned by the prior slice; fallback explicitly passes `publish_facts=False` (`UI/Console_Modules/prompts.py:1226`, `:1793`). Historical inspection invokes the operational snapshot loader only on explicit assistant-turn detail and does not substitute current Library bodies (`inspector_data.py:171`, `console_conversation_inspector.py:789`).
- ✅ Usage count is repr-hidden and added without schema/generic metadata fields; publication derives from the confirmed batch, hydration uses the raw operational reader, and lookup failure invalidation retains matching ownership/metadata fences (`console_chat_models.py:1322`, `console_chat_store.py:3260`, `:8879`, `console_conversation_hydration.py:597`, `console_turn_resend.py:285`). Retry still consults operational authority; failed marked/unqualified originals expose current-choice regeneration (`console_message_actions.py:817`).
- ⚠️ Task9 actual-app/live-provider proof and original-baseline attribution of the ten Canvas/tool-label failures are outside this task. No whole-branch green claim is warranted.

## Strengths

- Real SQLite/runtime-backed mounted coverage exercises ordering, disabled placement reconciliation, exact-origin refusal, save recovery and unchanged draft/System content (`Tests/UI/test_console_prompt_mixins.py:127`, `:443`, `:486`, `:536`). The production-host test exercises the actual action route at all three required sizes (`:324`).
- Original receipt/count tests include capture-off reopen, raw-corrupt markers, missing original recovery, forged display counts and preservation after transport/new-attempt-write failure (`Tests/Chat/test_console_prompt_mixin_retry.py:1195`, `:1234`, `:1276`, `:1331`).
- Token-backed feature styles reuse the existing shell and shared controls, and the generated bundle mirrors the source (`css/features/_console_panels.tcss:2600`). No dependency, schema, provider endpoint or extra preference store was added.

## Issues

### Critical

- None found in this task-scoped review.

### Important

- **I1 — Fence completion to the current Workbench view.** `tldw_chatbook/Widgets/Console/console_prompts_modal.py:371` awaits projection and then removes/mounts body children without verifying that Mix-ins still owns the body. `enter_mode`/Back (`:598`, `:614`) do not cancel or invalidate the `console-mixins` worker. Mounted reproduction: hold `flow.project` after producing its result; start `_refresh_mixins` in the normal worker group; enter Browse; release the projection. Observed `RESULT browse 0 1`: navigation state was Browse, with zero Browse widgets and one Mix-ins widget. A delayed refresh silently replaces the destination the user chose and breaks mode/dirty/focus bookkeeping. Invalidate/cancel on mode change and check a captured view generation before both success and error publication. Apply the same ownership rule to inspect/save/preview completions and add a held-worker navigation regression.

- **I2 — Preserve edits made while Save is pending.** `tldw_chatbook/Widgets/Console/console_prompts_modal.py:456` reads the editor fields, awaits the Library write, then unconditionally refreshes/removes the editor. The editor remains writable during that await, and no draft revision comparison protects later typing. Mounted reproduction: edit Instructions to `saved first draft`; hold `flow.save` after Save captured its arguments; change Instructions to `newer unsaved draft`; release. Observed `RESULT editor_count=0 stored_body=saved first draft`. The newer unsaved text is gone without a discard decision. Retain/compare the submitted raw draft and current editor identity/revision before replacing it; keep newer text dirty against the successfully saved version, or prevent editing for the bounded pending operation. Add a held-save regression that asserts the newer draft survives.

### Minor / follow-up

- **Projection-copy follow-up:** `ConsoleVariant` and the generation copy/apply seams do not carry `mixin_usage_count` (`console_chat_models.py:1361`, `console_chat_store.py:17631`, `:18132`). Generic message copies preserve it via dataclass fields (`console_chat_models.py:1329`), but selecting a generation changes metadata while leaving the previous count. An end-to-end failed-generation scenario was not established here; include this seam in Task9's required copied-projection audit rather than treating the current audit as exhaustive. Counts must remain display-only and match the selected receipt or become unqualified.
- Existing exact-BASE menu evidence contains `RuntimeWarning: coroutine 'ChatScreen._sync_native_console_chat_ui' was never awaited` (`task-8-evidence/exact-base-menu.txt:201`). This is inherited evidence noise, not a newly attributed Task8 failure; retain it in the final baseline accounting.

## Bounded native UI finish

- Inspected the supplied six-view `captures-final/inspection-sheet.png`, plus full-size `workbench-80x24.png` and `editor-80x24.png`. These are existing production-style Textual exports; no new visual iteration, browser substitute, identity pass or font detector was run.
- At 160×45 and 100×30, the selected rows and named actions are legible and grouped Before/After; compact actions preserve hierarchy. At 80×24 the two-column action grids keep Inspect/Edit, Move and Remove readable. Back/Close stay visible outside the scrolling work area at every size. The editor shows readable Name/Instructions and a visible focused Name/caret state. Scrollbars indicate content below the viewport; Placement and Save require scrolling in the shorter captures.
- The packet supports static layout/readability and one editor focus state. Dynamic keyboard traversal/scroll reachability comes from the supplied mounted interaction results, not screenshots alone. No fresh native OS-terminal glyph or contrast measurement is claimed. Exported shared box/checkbox glyph tofu also affects shell chrome; it limits glyph evidence, not an attribution to the new TCSS.
- Governing context: pinned PRODUCT.md/DESIGN.md terminal-native Operate direction, backlog design-language/component-patterns, and Impeccable craft-floor. No web-specific font/depth/motion rule was imposed on the TUI.
- Recomputed supplied source manifest hashes: all product/test files match `source-hashes-at-final-capture.json`; only the later Backlog note differs. `source-hashes-final.json` matches every listed file. This qualifies the captures to the reviewed implementation.

## Checks and review boundary

- Read existing final log summaries: 94 risk cases, 9 recovery cases, 25 integration/factory cases and 3 source-qualified capture cases pass. `lint-delta.json` is `[]`. Did not rerun those tests or regenerate captures.
- Required three-file run remains 158 passed / 6 failed; four are documented Task1 baseline nodes, two have exact Task8 BASE evidence. Ten Canvas/tool-label nodes also fail exact Task8 BASE and await Task9 earlier-origin audit. These are not green results.
- Ran only two new bounded, mounted, real-runtime probes for the uncovered pending-refresh and pending-save races. Both reproduced the reported defects. The first standalone probe import exited with `Recovery required: recovery_scope_uncertain` before exercising UI; the successful runs explicitly imported the repository test bootstrap into fresh temporary profiles. Ordinary optional-dependency warnings appeared in the standalone bootstrap; no provider/network run was attempted.
- Changed-file supplementary reads were limited to cut-off mode/back/close, hydration, generic-copy and generation-copy functions needed for named async-ownership/projection risks. Focused outside-diff checks: raw-marker classification in `chat_persistence_service.py:1298`; retry/source-currentness in `console_prompt_mixin_service.py:273`/`:509`; source Save/version behavior in `prompt_scope_service.py:1994`; explicit fork projection construction in `console_chat_fork.py:1471`. No count field appears in that explicit fork DTO; restore requalifies counts. The voice/fork lifetime audit is not an end-to-end requalification of all earlier-task paths.
- No product, index, HEAD, branch or task-status changes; only this review report is written.

## Assessment

**Spec compliance:** Needs fixes (I1, I2).

**Task quality:** Needs fixes. The runtime/source ownership work is careful and the existing evidence is substantial, but pending UI work can overwrite navigation or lose user edits. Fix and narrowly verify those lifecycle cases before accepting Task8.

Skipped full suites, live provider/actual-terminal Task9 verification and exhaustive generation/fork/promotion requalification. Remaining risk: pending callbacks can currently discard newer UI state; baseline failures remain unresolved evidence, not a green gate.
