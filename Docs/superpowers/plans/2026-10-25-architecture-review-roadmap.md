# Architecture review remediation — roadmap

> **For agentic workers:** this is a **program roadmap**, not a per-task TDD plan. The repo rule (CLAUDE.md §5) is that each task gets its own implementation plan when it moves to In Progress. Execute one task per PR. Use superpowers:subagent-driven-development per task (recommended) or superpowers:executing-plans, and read the task file plus the cited report section first.

**Goal:** Remove the structural risks found by the 2026-10-25 architecture review in an order that adds guards first, so every later change is measurable. Each step ships as a small, revertible PR.

**Architecture:** Four phases:
1. Defects and guardrails.
2. Pure moves that fix dependency direction.
3. Real seams (services bundle, registries, repositories). All three owner decisions are made (2026-10-25); the services-bundle ADR (TASK-36947.1) must be accepted before TASK-36947.2 starts.
4. Decomposition of the god functions and god screens.

Every structural PR lowers the ratchet it affects in the same commit.

**Tech Stack:** Python ≥3.12, Textual 8.x, SQLite; stdlib AST architecture tests in `Tests/Architecture/`.

**Spec:** `Docs/QA/architecture-review-2026-10-25/report.md` (findings F1–F17), with claim-level evidence in `Docs/QA/architecture-review-2026-10-25/verification.md`. Base: `origin/dev` @ `7a5e2513c5`.

**Backlog:** 13 work-stream parents TASK-36941…TASK-36953 (subtasks `.1`–`.9`), plus TASK-1378.1–.2 under the existing Settings split task. Labels: `architecture-review-2026-10-25`.

## Global Constraints

- **One task, one PR.** It targets `dev`. Merge-queue rules from CLAUDE.md apply: arm only after the local code review on the current head, rebase and never merge-in, `--force-with-lease` only.
- **Pure moves are byte-for-byte** with re-export shims at the old path. No behaviour change. Existing tests pass unmodified, except in the tasks whose job is retargeting tests (TASK-36952.2).
- **Ratchets move in the same commit.** A PR that shrinks a measured thing (module size, lambda count, string lookups, comparisons, the layer allowlist) lowers the pinned value in that commit. Rows rise only under the owner's 2026-10-04 ruling.
- **Tests:**
  - Run every pytest through the scratch wrapper (scratch HOME, XDG and `TLDW_CONFIG_PATH`, plus `--basetemp`), with xdist.
  - Run `Tests/Architecture` serially.
  - No test or probe may write the real profile; a sandboxed launch denies writes to `~/.config/tldw_cli` and `~/.local/share/tldw_cli`.
- **Preflight:** run `./scripts/preflight.sh` before opening a PR, and again after any rebase that touches `backlog/`.
- **Owner decisions (all made 2026-10-25) are not reopened by implementers:**
  - TASK-36941.3: keep ADR-012 and align the hosted engine to it (stored key before the environment variable when no credential source is recorded). Re-confirmed with ADR-012 in view. Custom endpoints are an owner choice recorded inside that task.
  - TASK-36943.2/.3: on an ADR-number collision the oldest file keeps the number and each younger file is renumbered with provenance.
  - TASK-36947.1: adopt a frozen, typed, per-feature services bundle built once in `app_service_wiring`, with a shared factory for the other entry points and no DI container. TASK-36947.1 writes the ADR; TASK-36947.2 waits for the owner to accept it.
- **Defer to existing owners:** TASK-32807.5 (dead Chat modules and `_chat_api_kwargs`), TASK-32898 (Interop helpers), TASK-34905 and TASK-33132 (transaction consolidation), TASK-34906 (migration registry), TASK-32463 (over-budget screens). Do not re-solve their scope inside these tasks.

## Review Focus

The failure modes most likely to bite users that no single task's tests cover by default:

1. **A move silently detaches a test patch.** Patching the old module path stops reaching code that moved; this has happened before (`test_app_extracted_patch_targets.py`). Every move task greps `Tests/` for patches on the moved names and either retargets them or proves the shim keeps them effective. A patch that no longer reaches its code is the bug class that once wrote the real config.
2. **A DB split or injected coordinator moves work outside the write transaction** (TASK-36949.2–.4). The semantic-mutation authorizer fails closed, so the symptom is a refused write, not corruption. Each task asserts the moved writes and the coordination run inside one `transaction()`, including a rollback case.
3. **Boot import cost regresses** because a new package or shim imports eagerly (TASK-36944.x, 36945.1, 36947.2). The boot-import snapshot and pre-import ratchets (ADR-097 `097-boot-budget-ratchets.md`, TASK-33276) must stay green; new neutral packages must not import Chat, DB or Textual at module level.
4. **Settings save order changes** when a category moves into the panel registry (TASK-1378.x). Pin ADR-033 (`backlog/decisions/033-settings-commit-models-three-honestly-labeled.md`) commit behaviour per category, including unsaved-change prompts on navigation, before moving it.
5. **The credential alignment silently switches a user's key** (TASK-36941.3). Users of engine presets who have both an env var and a stored key, and no recorded credential source, will now see the stored key used. The task pins the order with both present and adds a CHANGELOG entry.

---

## Phase 0 — Land the record (this PR)

- The report, verification ledger, measurement scripts, 69 task files and this roadmap.
- **Gate:**
  - `./scripts/preflight.sh` green;
  - `scripts/check_backlog_task_ids.py` re-run against a freshly fetched `origin/dev` immediately before merge (IDs 36941–36953 and 1378.1–.2 are provisional until they land);
  - `backlog-py doctor` clean.

## Phase 1 — Defects and guardrails (parallel; no task depends on another)

| Order | Task | Why first | Lowers / adds |
|---|---|---|---|
| 1 | TASK-36941.1 Subscriptions rollback | Data integrity: abandoned writes can be committed later | — |
| 2 | TASK-36942.2 agent_runtime import-footprint ratchet | Freezes today's footprint so it cannot grow; full purity is TASK-36944.10 after the provider-contract move | Adds the pinned module set |
| 3 | TASK-36942.1 size rows for all 4,000+ line modules | Later streams lower these rows | Adds 40 rows and the auto-guard |
| 4 | TASK-36942.3 layer-direction test | Every phase-2 move shrinks its allowlist | Adds the allowlist |
| 5 | TASK-36942.4 app string-lookup ratchet | Phase-3 services work lowers it | Adds the census |
| 6 | TASK-36941.2 Research public methods | Independent defect | — |
| 7 | TASK-36941.4 `chat_models.py` removal | Check TASK-19571's rule first | — |
| 8 | TASK-36943.1 docs match code | Cheap; prevents re-review churn | — |
| 9 | TASK-36943.2 ADR numbering rule and guard | Stops new collisions immediately | Adds the frozen list of shared numbers |

**Start early in phase 1:** write the services-bundle ADR (TASK-36947.1) so it can be accepted before phase 3.

**Exit gate:** TASK-36942.1–.4 merged, so every later PR has a number to lower.

## Phase 2 — Pure moves that fix dependency direction (four parallel lanes)

| Lane | Sequence | Shared-file hotspots to serialize |
|---|---|---|
| A: dependency direction | 36944.1 → 36944.2, and 36944.1 → 36944.10 (agent runtime purity; lowers 36942.2's pinned set). Then 36944.3, .4, .6, .7, .8 in any order. 36944.5 and 36944.9 last (Console files) | `Chat/console_*`, `Widgets/Console/*` |
| B: config | 36945.1, 36945.2, 36945.3, 36945.4 in any order (TASK-36947.6 retires config's DB singletons in phase 3) | `config.py`: one config PR in flight at a time |
| C: providers | 36946.1 → 36946.2; 36941.3 (decided: keep ADR-012, stored credential first) | `Chat/Chat_Functions.py`, `provider_registry.py` |
| D: small independents | 36949.1, 36949.5, 36948.2, 36953.2, 36953.3; 36943.3 (ADR renumbering) after 36943.2 | `DB/ChaChaNotes_DB.py` (36949.x and 36948.2 both touch its imports) |

**Cross-task interfaces fixed here, which later tasks consume:**

- **TASK-36944.1:** the neutral package is `tldw_chatbook/Provider_Contract/`. Its modules hold the provider exceptions (from `Chat/Chat_Deps.py`), continuation, sampling-parameter and thinking-block types, plus `provider_endpoint_contract`, which continuation imports. Importing it loads none of the Chat, DB, Agents, LLM_Calls, UI, Widgets or Event_Handlers packages, nor config, textual, httpx or sqlite3. TASK-36944.2's boundary test and TASK-36944.10 import from it.
- **TASK-36949.1:** `CharactersRAGDB.managed_transaction_depth() -> int` (0 outside a managed transaction). It serves both kinds of caller, those asking whether a transaction is open and the Persona_Visual guards asking whether they own the outermost one. TASK-36949.2–.4 and the existing repositories call it instead of reading `_local.transaction_depth`.
- **TASK-36942.3:** the allowlist is a TSV beside the test, one `src_module<TAB>dst_module<TAB>kind` row per allowed upward edge. Phase-2 PRs delete rows.
- **TASK-36942.4:** the census is a TSV of `package<TAB>count`, plus a frozen list of allowed private names. Phase-3 PRs lower counts.
- **TASK-36945.4:** the tools receive the database from the provider that registers them (`Agents/tool_catalog.py`'s builtin construction path). TASK-36947.6 later deletes the config DB globals, once the services factory exists.

**Exit gate:** the layer allowlist has shrunk by at least the edges lanes A and D remove, and `config.py` is at least 2,584 lines shorter (TASK-36945.1).

## Phase 3 — Real seams

| Task | Depends on | Notes |
|---|---|---|
| TASK-36947.2 services bundle | 36947.1 ADR accepted | Shape, ownership, teardown and alias period come from the ADR; aliases keep the old attributes working |
| TASK-36947.3 RAG service move | 36947.2, 36942.4 | First consumer cluster; removes RAG_Search → Event_Handlers |
| TASK-36947.4 Console controller without the app | 36947.2 | Land before TASK-36951.3 (same file) |
| TASK-36947.5 ConsoleRuntime hook-review state | — | Independent; can start in phase 2 |
| TASK-36947.6 config exposes no DB instances | 36947.2, 36945.4 | Every entry point gets its DB from the bundle or the shared factory |
| TASK-1378.1 → TASK-1378.2 Settings panel registry | — | Network proves the registry, then Providers & Models; pin the settings commit-model behaviour (ADR-033 `033-settings-commit-models-three-honestly-labeled.md`) first |
| TASK-36948.1 participant declarations, TASK-36948.3 unsaved-editor state | — | Both touch `Backup_Recovery/`; serialize |
| TASK-36949.2 learning repository | 36949.1 | ~45 call sites in two Study_Interop services |
| TASK-36949.3 semantic coordinator injected; TASK-36949.4 chat-sync adapter | — | Each must keep its work inside the write transaction (Review Focus 2) |
| TASK-36953.1 narrow client protocols | — | Typing-only |

**Cross-task interface:** TASK-1378.1 defines the panel contract in `tldw_chatbook/UI/Settings_Modules/category_panels.py`. It is a `SettingsCategoryPanel` Protocol (render, has_unsaved_changes, request_save, revert, scope_text, commit_model) and a `CATEGORY_PANELS: dict[SettingsCategoryId, SettingsCategoryPanel]` registry. TASK-1378.2 registers the Providers & Models panel there.

**Exit gate:**
- the app string-lookup census is lower than at filing;
- ChaChaNotes_DB imports neither Chat nor Sync_Interop;
- participants has no hard-coded store list.

## Phase 4 — God functions and god screens

| Task | Depends on | Notes |
|---|---|---|
| TASK-36950.1 → TASK-36950.2 agent loop table, then phases | — | 54 test files pin `run_agent_loop`; pure moves only |
| TASK-36950.3 → TASK-36950.4 fleet class, then run-log tools | — | `_run_one` shrinks by ≥1,800 lines in .3 |
| TASK-36951.1, TASK-36951.2 Console hosts own state | — | Serialize: both edit `console_chat_controller.__init__` |
| TASK-36951.3 reply request object | after 36947.4 | Same file as 36947.4 |
| TASK-36952.1 → .2 → .3 Library media | — | Follow recipe §3 and §24; measure click-to-paint before and after .3 |

**Cross-task interfaces:**
- **TASK-36950.1:** the special-tool handlers live in one module-level dict keyed by tool name, each handler taking `(call, state)`. TASK-36950.2's phase functions take and return the same `LoopState`.
- **TASK-36950.3:** the per-run object is a class in a new module beside `agent_service.py`. TASK-36950.4 adds the run-log tools to it as methods.

**Follow-ups to file when phase 4's media series lands** (not filed now, so no task references future work):
- the same Phase A2 series for Notes;
- the remaining Settings categories after TASK-1378.2;
- the remaining screens after TASK-36948.3.

## Progress measures (record at each phase exit)

| Measure | At filing (7a5e2513c5) | Source |
|---|---|---|
| Executable SCC size (top-level + function-level imports) | 1,217 modules / 67 packages (1,496 incl. TYPE_CHECKING) | `scripts/analyze.py` |
| Executable upward imports: domain→presentation / DB→feature / kernel→higher | 80 / 335 / 89 (83 / 358 / 99 incl. TYPE_CHECKING) | `scripts/layers.py` |
| App string lookups (UI / non-UI) | 1,176 / 222 | TASK-36942.4 census |
| Unbudgeted 4,000+ line modules | 40 | TASK-36942.1 guard |
| `library_screen.py` file lines (class lines) / methods / `__init__` lambdas | 37,542 (36,411) / 1,354 / 413 | screen ratchet (file lines, methods), `scripts/deleg.py` (class), TASK-36952.1 (lambdas) |
| `SettingsScreen` category comparisons | 239 | TASK-1378.1 ratchet |
| `_run_one` / `run_agent_loop` lines | 4,163 / 2,451 | `scripts/classes.py graph.json funcs` |
| `config.py` lines | 12,258 | TASK-36942.1 row |
