# Native phased workflows: current-dev integration audit

What this change does: the local feature adds approved planning and phases while sharing native Console, AgentService, accounting, file authority, physical ownership and recovery. Current dev has since added review-boundary automatic work and extracted live execution paths. Integration must preserve both owners.

Read-only source audit on 2026-11-05. Local source: `e08627ac38392fb890984169376b1c242c3a2f3e`. Upstream: `3c39215749a96d5c1b8c05972fdd2aeb7588b002`. References below say **local** or **dev**; they are not candidate-tree line numbers. No tests, profiles, Git index/refs or source files changed. This report is the sole write. Scope is release integration, not a second review of the whole implementation.

## Must fix

1. **Reserve workflow schema 26–30 after shipped review schema 25.**

Local `tldw_chatbook/DB/AgentRuns_DB.py:320,865-939` uses schema 29 and numeric stage checks 26–29. Dev `tldw_chatbook/DB/AgentRuns_DB.py:866-875` adds `review_boundary_id`; its shipped `DB/migrations/agent_runs_v24_to_v25_review_boundaries.sql` adds the review-claim table. Dev `DB/recovery_operations.py:325-341` already owns 24→25. Reusing 25 for workflows gives two meanings to one version and breaks exact restore migration.

Keep dev's shipped review migration and `DB/recovery_review_boundary_schema.py` unchanged. Allocate the unpublished workflow stages:

| Current-dev route | Meaning | Old local stage |
| --- | --- | --- |
| 24→25 | Review-boundary claims, shipped | none |
| 25→26 | Workflow core tables | 25 |
| 26→27 | Workflow admission / active intervals | 26 |
| 27→28 | Checkpoints / resume commands | 27 |
| 28→29 | Change windows | 28 |
| 29→30 | Repair-input digest | 29 |

Reach all five workflow SQL files, `_CURRENT_SCHEMA_VERSION`, constructor checks/stamps, migration tuple stamps, catalog composition and supported versions, row-validation thresholds, tests and current architecture prose. Preserve dev's review table, column, authorizer entries and both physical v25 column-order variants. Keep dev's stronger type check in `DB/recovery_goal_schema.py:90-99`; the feature's `<24` behavior already exists there. Do not mass-replace every 25–29: legacy declarations and historical QA retain their original meaning.

Workflow admission renames/drops `automatic_wake_attempts.attempt_kind` (local `DB/recovery_workflow_schema.py:364-379`). Verify it preserves `review_boundary_id`, review rows and foreign keys with populated attempts, not empty tables. Retain the additive `workflow_phase` check alongside fleet/Goal values.

2. **Recognize old local workflow profiles by complete catalog before any upgrade.**

Numeric renumbering alone is unsafe. Old v26 already has the admission table replacement; renumbered guards skip stages on misleading stamps. Old v29 already has the repair column; new v30 would repeat its ALTER. Shipped dev v25 has no workflow tables, but today's `validate_workflow_rows` queries them for `MAX(version)>=25` (local `DB/recovery_workflow_schema.py:94-180`).

The smallest safe compatibility bridge fits existing DB/recovery owners:

- Freeze the actual old local v25–v29 **complete** catalogs as legacy installed declarations, separate from canonical 26–30 catalogs. Retain every previously recognized physical predecessor variant. Do not derive legacy catalogs from current dev's v24 strings: persisted SQL comments changed ADR-147 to ADR-344 in `agent_runs` / `agent_definitions`, and exact equality includes comments. Snapshot old declarations once; no Git access at runtime.
- Match complete catalog **and** old stamp, then validate rows under that matched legacy workflow stage. Version or presence of `workflow_runs` alone is insufficient. Unknown/hybrid layouts refuse; do not stamp them as current or repair by best effort.
- For each recognized old stage, one fixed installed bridge in `synchronous=FULL`, `BEGIN IMMEDIATE`: add the review column/table from existing `REVIEW_BOUNDARY_COLUMN` / `REVIEW_BOUNDARY_TABLE`; append stamp `old_stage + 1`; validate resulting exact catalog/domain rows; commit. Keep earlier history rows. Do not run `REVIEW_BOUNDARY_MIGRATION` wholesale: its v25 INSERT collides. Then run canonical successors through v30.
- Register the bridge's exact resulting catalogs as canonical variants. Old admission-or-later profiles receive `review_boundary_id` **after** existing `attempt_kind`; a fresh canonical database may have opposite order. Do not rebuild unrelated tables to force order equality.
- Run recognition/bridge at the start of `AgentRunsDB._initialize_schema`, before the first `executescript` (local `AgentRuns_DB.py:411-420`). General initialization would otherwise add review artifacts and destroy the original recognizable shape. Existing private storage admission remains the owner. No global sweep, runtime or profile scan.
- Restore staging reuses the same declarations through `_AgentRunsAdapter` and `Backup_Recovery/sqlite_validation.py`. Existing code proves complete installed catalogs (`1084-1107`) but selects migrations by version only (`1194-1203`). Add a narrow `db.agent_runs` exact-legacy-catalog bridge selection inside the existing transaction, before that numeric loop. Do not register two outgoing normal steps from one version; rejection is intentional. Recheck exact catalog, rows and version, then continue existing migration. Preserve cancellation/rollback and temporary authorizer scope (`1263-1286`).
- Row validation must use catalog-matched stage for both lineages: legacy v25 needs workflow validation; shipped review-only v25 must not query missing tables. An internal stage argument or narrow classifier shared with the installed declarations suffices. No generic migration framework.

Preserve IDs, launches, plan/control revisions, results, reservations, charges, uncertain intervals and checkpoints. The bridge adds compatibility only; activation and explicit Resume still gate restored execution. This audit does not authorize opening or changing the user's live profile.

3. **Port workflow checks into upstream's live request/reply paths.**

Dev `Agents/agent_service.py:3063-3083,3278,3336` delegates model/tool execution to `Agents/agent_request_execution.py`. Local planning seed validation is still in the old service body (`agent_service.py:3469-3471`). Dev's live receiver owns review-history and request admission (`agent_request_execution.py:264-269,385-456`), and denies non-review tools for review children (`608-615`). Keeping wrappers without moving checks leaves native planning unguarded; restoring local bodies discards shipped confinement.

Preserve dev extraction. Port the planning seed check after actual final payload bounding, plus task/planning source scopes and worker custody into live `make_call_model` / `make_invoke_tool`. Preserve review-child denial, check-process settlement, first-request disclosure and cancellation. Keep equivalent checks on remaining non-native request paths. Each method must have one live definition.

Dev `Chat/console_chat_controller.py:27117-27168` delegates `_run_agent_reply` to `Chat/console_agent_reply.py:12`. That body owns pre-provider Stop (`362-371,732-733`) and bridge dispatch (`773`). Port original Console source capture and ordinary-tree scope to actual dispatch/return there. Do not resurrect the old controller body. Preserve upstream `console_streaming_request.py`, `console_pre_provider_stop.py`, received-dispatch, continuation and settings owners. Local qualification does not qualify newly combined receivers.

4. **Use current storage/model owners and retain both automatic-work authorities.**

Workflow-only local `Chat/console_workflow_runs.py:1036,1047` imports `_core_operation`, absent on dev. Dev `Backup_Recovery/participants.py:432` calls it `core_operation`. This is the unused/cancelled phase settlement path; an old import strands recovery exactly on failure. Use the current owner. Preserve dev AgentRuns factory slots for `store_kernel.core_access`, `core_cached_connection`, `register_core_connection`, `core_closing`; no obsolete aliases needed.

Dev `Agents/agent_models.py:63-78` owns `ToolPathTarget` / `PathAwareToolProvider`, re-exported from `tool_catalog.py:46-48`; local `tool_catalog.py:1036-1049` still defines them. Keep current canonical types/re-export and compose workflow scope gates with current registry. Avoid duplicate nominal models.

Preserve Provider_Contract imports, `AgentConfig.review_binding`, `SpawnAdmissionReplay`, review child registry, review-boundary acceptance/abort and original locks. Dev `DB/automatic_work.py` registers native review authorities and retains claims. Add workflow authority beside that owner, not a wholesale old ledger replacement. Review-driven automatic turns/children are newly relevant native actors for worktree/opaque admission. Check their actual physical admission/retirement paths; neither review terminal rows nor a completed future alone proves physical drain or grants workflow authority.

## Governance

`git ls-tree` prefix inspection found **no** `backlog/decisions/285-*` on pinned dev. ADR-285 is available. Amend existing `backlog/decisions/285-native-phased-agent-workflows.md` rather than minting another ADR. Add: shipped v25 keeps review meaning; workflows own 26–30; exact local legacy 25–29 receive a fixed transactional bridge; unknown hybrids refuse; existing catalog/authorizer/publication owners retain restore authority; accounting/approval state survives and migration never resumes execution. Link from implementation plan and task notes. Record before implementation because this defines persistent compatibility. Update current architecture schema prose; preserve historical QA source/version labels and append integrated evidence.

ADR required: existing ADR amendment.
ADR path: `backlog/decisions/285-native-phased-agent-workflows.md`.
Reason: collision handling and accepted legacy shapes define a persistent compatibility contract.

## Targeted evidence

Run against the assembled candidate, after source-boundary resolution. Local-HEAD results are not candidate evidence.

- Extend `Tests/DB/test_workflow_runs_migration.py`: shipped v25 both physical lineages, fresh creation, all five frozen local stages; constructor/reopen, standalone canonical SQL and staged restore; max 30, exact catalog, FK/quick checks and unchanged populated review/fleet/Goal/workflow rows. Include old v29 with results/checkpoints/charges and old v25 where the collision differs most.
- Extend `Tests/Backup_Recovery/test_workflow_recovery_schema.py`: legacy read-only validation leaves bytes unchanged; bridge failure/cancellation rolls back; wrong-stamp/hybrids refuse; migration authority is temporary/exact; backup→restore→rebackup retains explicit-resume state. Retain upstream `test_agent_runs_recovery_schema.py` and `test_goal_recovery_schema.py`. Existing agent recovery rollback/authority tests are the model.
- Run `Tests/Agents/test_code_review_boundary_ledger.py` (v24 upgrade, replay, acceptance revalidation, uncertainty), `test_agent_service_review_containment.py`, `test_agent_service_review_state_scope.py`, and `Tests/Chat/test_console_code_review_authority.py` alongside workflow admission/policy/child-launch tests.
- Run workflow native planning, wave, settlement, recovery and restart cases: `Tests/Chat/test_workflow_planning_native.py`, `test_workflow_phase_wave_native.py`, `test_workflow_settlement_delivery.py`, `test_workflow_recovery.py`, `Tests/DB/test_workflow_restart_process.py`. Run `Tests/Chat/test_console_ordinary_tree.py` and worktree-activity tests against extracted receivers. Add a focused native case where a held review child/automatic review continuation blocks conflicting workflow launch until physical return.
- Run focused hook-admission and original coroutine-warning regressions after adapting worker entry. Preserve deferred coroutine creation from `98e8035315`; historical QA is evidence of earlier source, not integrated source.
- Static checks: duplicate class methods including private names; imports at changed native paths; migration-source guard; scoped lint and whitespace. Preserve foreign staged paths from upstream bytes under root's scope policy. This audit did not import local foreign bytes.

Verdict: fix 1–4 and qualify combined native receivers before presenting a ready-for-review PR. A draft may accurately disclose pending integration; old local qualification cannot support a passing-candidate claim.

Not checked: no candidate merge tree, executed tests, live GUI/provider run or user profile. Main risk: silently retaining old bodies or using version-only migration across the two v25 meanings.
