# Textual pending draft follow-ups implementation plan

> **For agentic workers:** Use the existing task files as the requirements and record RED/GREEN evidence per task. Independent Dictionary, Theme and Parakeet owners run in parallel; both Prompt tasks stay together because they share a canvas.

**Goal:** Complete TASK-36454 through TASK-36458, retain accepted input across asynchronous publication, and open a reviewed, green PR against dev.

**Architecture:** Keep each draft with its existing destination owner. Save completion adopts the actual persisted identity/version and acknowledges only submitted input. Structural reconciliation re-reads live child input after awaits; installation completion checks the original request before selecting the current form.

**Tech stack:** Python >=3.12, Textual >=8,<9, existing SQLite/TOML services.

**Spec:** The five linked task files and [source-bound audit](../qa/2026-10-23-textual-book-after-pr3219.md).

ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md; backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md; backlog/decisions/050-external-parakeet-roots-with-managed-vad.md; backlog/decisions/097-boot-budget-ratchets.md
Reason: bounded asynchronous acknowledgement/publication repairs; each task links its remaining applicable decisions, and no storage, service, consent or ownership boundary changes.

## Global constraints

- Preserve existing service, storage, artifact activation, history, navigation and consent boundaries; no new dependencies or global draft owner.
- Read applicable canonical ADRs and the design language; existing token-backed controls and stable widget identities remain.
- Targeted local tests only. Preserve private evidence; never delete .test-tmp or use git stash.
- Public regression tests must run in required PR CI; retain every inherited selector and the existing shard/cap policy. Initial insertion preserves all inherited slots; the measured CI follow-up below permits a two-slot swap after repeated job-cap cancellations.
- Use Superpowers requesting-code-review, never Qodo. Create PR to dev; merging this new PR is not requested.

## Review focus

- Raw live input with its Changed message queued must survive acknowledgement.
- A second Save must use the committed identity/version rather than duplicate the row.
- Both Basic lanes, a later clear and cross-lane input must converge after structural awaits.
- A later Theme name must remain separate from the actually saved file identity.
- Parakeet success may finish consented activation while declining obsolete form selection, including ABA choices.

## Task 1: Basic Prompt first mount (TASK-36456)

Files: library_prompts_canvas.py; Tests/UI/test_library_prompt_pending_drafts.py.
Interfaces: existing PromptBlockEditor.replace_block_state and canvas semantic messages; stable block IDs, metadata/order unchanged.
- [x] Add mounted first-mount controls for both lanes, later key, clear and cross-lane delivery.
- [x] Run unchanged production and witness retention assertions fail.
- [x] Reconcile latest raw Basic values after structural awaits, preserving incremental composition.
- [x] Run targeted controls; record counts and evidence in task notes.
ADR required: no. Existing ADR-033 application state, ADR-040, ADR-076 govern this bounded projection repair.

## Task 2: Library Prompt Save (TASK-36454)

Files: library_prompts_controller.py; new prompt_save_ack.py (existing completion extraction to respect controller ceiling); library_prompts_canvas.py; prompt_block_editor.py only if raw Advanced capture is needed; same regression file.
Interfaces: existing save_prompt result identity/version and canvas incremental state projection. Consumes Task 1's safe Basic reconciliation.
- [x] Add real SQLite create/update success, later Basic/Advanced/metadata, queued Changed, failure, next Save and guarded-exit regressions.
- [x] Witness retention failures before changing acknowledgement.
- [x] Capture raw current draft, retain newer blocks, adopt actual identity/version and project coherent saved/draft state without replacing children.
- [x] Verify all controls and existing Prompt reader/editor journeys, document behavior.
ADR required: no. Existing ADR-033 application state, ADR-040/049/076/086 apply; no persistence boundary changes.

## Task 3: Dictionary Settings Save (TASK-36455)

Files: personas_screen.py dictionary settings handler; personas_dictionary_detail.py; dedicated mounted regression file.
- [x] Claim task and record its exact implementation plan before code.
- [x] RED: ordinary/later raw input, queued Changed, real SQLite version and retry/guards.
- [x] GREEN: acknowledge submitted settings while preserving later raw controls and aggregate dirty.
- [x] Verify targeted service/UI controls and document evidence.
ADR required: no. Existing decisions linked in TASK-36455 apply; entry/transition ownership stays unchanged.

## Task 4: Theme Save (TASK-36457)

Files: settings_theme_editor.py; settings_theme_picker.py; dedicated mounted regression file; theme-system.md.
- [x] Claim and plan; RED real TOML ordinary/later palette/name/dark, Save-as/failure/next Save controls.
- [x] Preserve raw current draft, saved-file identity and existing overwrite rules; auto-return only when no later draft remains.
- [x] GREEN mounted controls and existing explicit leave protections; record evidence.
ADR required: no. Existing decisions linked in TASK-36457 apply; no new commit model.

## Task 5: Parakeet install selection (TASK-36458)

Files: library_screen.py install handlers; library_ingest_canvas.py if necessary; dedicated regression file and existing fixture adjustment.
- [x] Claim and plan; RED actual button/consent/worker/UI publication with bounded external acquisition seams.
- [x] Retain original request context and compare current live form at completion; obsolete selection stays untouched and status is honest.
- [x] GREEN ordinary, provider/folder/precision/source changes, ABA and queued delivery; consented activation unchanged.
ADR required: no. Existing ADR-014/025/050/033 application state and ADR-011 apply.

## Integration and delivery

- [x] Inspect all diffs, update relevant architecture docs and task notes, append exact regression selectors to required PR lanes, then qualify the measured two-slot CI rebalance below.
- [x] Run targeted UI/service/static/governance verification; inspect diagnostic inventory statement delta before regeneration if necessary.
- [x] Commit explicit files, request fresh-context Superpowers code review, verify and fix actionable feedback.
- [x] Mark tasks Done through CLI only when AC/evidence/review checks hold; push and create/attach PR against dev.
Delivery gate: check all CI and paginated review feedback on the current head, address failures/feedback, and record final exact-head results in [PR #3228](https://github.com/rmusser01/tldw_chatbook/pull/3228). The task/documentation closure is bundled before that run to avoid a second CI cycle solely for status metadata. CI completion is tracked by the PR check rollup and final description, rather than a subsequent source commit.


## CI qualification follow-up: measured census rebalance

ADR required: no
ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
Reason: apply the existing workflow's log-replay/reorder policy; no new gate, dependency, shard, test or timeout policy. Existing TASK-35834 and TASK-33664.4 own the established capacity rules.

After `a7129eabea` UI shard 3 hit its 20-minute cap twice, stop unchanged retries. Attempt 1 passed all 169 tests in 1150.31s before shutdown cancellation; attempt 2 reached 98% before cancellation. Eighteen other prerequisite jobs and Perf Guard passed. Preserve both failures separately from the next source qualification.

1. Replay all thirteen complete attempt-1 shard logs with the existing `ui_lane_shard_replay.py` cost attribution. Rule out test weakening: read-only native MCP profiling finds no measured robust saving, and all test controls remain intact.
2. Swap only inherited census slots 77 and 432: `test_destination_rail_row.py` (shard 12, 94.06s) and `test_mcp_permissions_handoff.py` (shard 3, 221.69s). Keep all 720 unique selectors, all 56 appended regressions and 662 of 664 inherited positions fixed. Preserve annotation ownership.
3. Check exact selector multiset, changed slots, all thirteen ordered disjoint checker outputs and existing census/replay/workflow contracts. Record calibrated before/after estimates as estimates, not hosted passes. Keep 13 shards, serial execution, minimal dependencies, 20-minute job and 180-second test caps unchanged.
4. Obtain a scoped fresh-context Superpowers review of the source/evidence delta, fix actionable feedback, and publish one necessary source update. Follow normal pull-request CI on that exact head and refresh paginated feedback before delivery. No merge is authorized.
