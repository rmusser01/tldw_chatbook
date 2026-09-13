# Console AGENTS.md Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe, Codex-compatible `AGENTS.md` project guidance to Console agent runs, with root context first, lazy nested activation second, and interoperability/rollout evidence third.

**Architecture:** One selected workspace folder binding is both the v1 working directory and instruction authority root. Root guidance is resolved into an ephemeral, untrusted provider-context rider; a later delivery adds registry-owned path mapping plus an atomic tool-batch preparation hook for nested guidance. Only versioned control metadata persists locally, while automatically loaded instruction bodies remain run-local and never enter transcripts, agent logs, or database rows.

**Tech Stack:** Python 3.11+, Textual 8, SQLite/FTS5 migrations, dataclasses and structural protocols, existing Console provider gateway/agent runtime/tool catalog, pytest + pytest-asyncio + Hypothesis, Backlog.md.

---

## Source of truth and execution constraints

- Approved design: `Docs/superpowers/specs/2026-08-20-agents-md-support-design.md`
- Accepted decision: `backlog/decisions/069-console-project-instruction-local-state-and-preflight.md`
- Delivery tasks: `TASK-19634`, `TASK-19635` (depends on 19634), and `TASK-19636` (depends on 19635).
- ADR required: **yes**.
- ADR path: `backlog/decisions/069-console-project-instruction-local-state-and-preflight.md`.
- Reason: this changes provider/runtime trust boundaries, cross-module tool contracts, and local-only durable session state.
- Planning worktree: `.worktrees/agents-md-support`, branch `codex/agents-md-support`.
- Command precondition for this worktree: run `source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate` in each execution shell; all `python` commands below refer to that environment.
- Baseline evidence on 2026-08-20: the following command passed 71 tests (two warnings) across tool ownership, review ordering, local-provider composition, schema-local state, workspace bindings, and Context UI:

```bash
python -m pytest Tests/Agents/test_tool_catalog_owner_cache.py Tests/Agents/test_agent_runtime_review_hook.py Tests/Chat/test_console_local_review_hook.py Tests/DB/test_chachanotes_context_summary_migration.py Tests/Workspaces/test_workspace_folder_bindings.py Tests/UI/test_console_context_modal.py -q
```
- Do not launch Chatbook against the real profile during the schema delivery. Use `tmp_path`/in-memory databases until the schema-changing branch is integrated everywhere that shares the real data directory.
- Before each delivery, repeat the open-PR/branch collision check from `backlog/docs/lessons-backlog-hygiene.md` and rebase onto the intended integration branch. If the ChaChaNotes schema head is no longer v32, allocate the next migration from the actual head and update every v32/v33 path below consistently.
- Do not start TASK-19635 until TASK-19634 is integrated or rebased as its exact base; do not start TASK-19636 until TASK-19635 is integrated or rebased as its exact base.

## File responsibility map

| File | Responsibility |
| --- | --- |
| `tldw_chatbook/Chat/console_project_instructions.py` | Versioned session control state, binding/destination fingerprints, eligible-binding validation, notice state, and immutable provider rider rows. No filesystem reads or UI-state derivation. |
| `tldw_chatbook/Agents/project_instruction_resolver.py` | Immutable source/candidate/snapshot/outcome contracts; secure bounded root and root-to-target resolution; raw admission and rendering. No Textual, database, or provider imports. |
| `tldw_chatbook/Agents/project_instruction_runtime.py` | Delivery-2 run-local activation ledger, per-chain cursors, target-union preparation, and ephemeral warning/context rows. |
| `tldw_chatbook/Agents/tool_catalog.py` | Structural `PathAwareToolProvider` contract and the single cached first-wins owner lookup shared by preflight and dispatch. |
| `tldw_chatbook/Agents/local_tool_provider.py` | Selected-root capability filtering and structural path-target mapping for all supported `fs_*`/`git_*` tools. |
| `tldw_chatbook/Tools/patch_tool_impls.py` | Reusable bounded parse entry point used by both `fs_patch` execution and preflight; one grammar only. |
| `tldw_chatbook/Agents/agent_runtime.py` | Typed preparation result, preparation-before-review ordering, canonical deferral stubs, and code-only warning callback. |
| `tldw_chatbook/Agents/agent_models.py` | Carry the already-resolved response-reserve token count in `AgentConfig` so every parent/child request budgets against the value actually dispatched. |
| `tldw_chatbook/Agents/agent_service.py` | One exact request builder for send/headroom, per-chain initial-context injection, and shared preparation/warning state across parent/subagent loop dependencies. |
| `tldw_chatbook/Chat/console_agent_bridge.py` | Build one run context/ledger, thread the captured startup candidate and consent callback into `AgentService`, and avoid persistence of automatic bodies. |
| `tldw_chatbook/Chat/console_provider_gateway.py` | Preserve tagged canonical rows through provider routing; consume the marker only for llama.cpp requests serialized directly here. |
| `tldw_chatbook/Chat/Chat_Functions.py` | Shared all-handler metadata sanitizer: preserve the marker only for explicitly marker-aware native grouping adapters and strip it from copied rows before every other registered handler. |
| `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` | Consume the preserved marker during final Anthropic/Gemini native conversion and emit provider-valid result/context grouping. |
| `tldw_chatbook/Chat/console_chat_store.py` | Own per-session control state and its temporary-to-durable/write-through lifecycle. |
| `tldw_chatbook/Chat/chat_persistence_service.py` | Narrow project-context persistence adapter methods over `CharactersRAGDB`. |
| `tldw_chatbook/DB/ChaChaNotes_DB.py` + migration SQL | Local-only JSON column and version-neutral accessors; exclude it from sync behavior. |
| `tldw_chatbook/Chat/console_chat_controller.py` | Dispatch-time binding revalidation, secure startup-candidate capture, selected-root provider composition, and first-use consent/UI callbacks. |
| `tldw_chatbook/UI/Console_Modules/session.py` | Explicit screen-state serialization/restoration of the four control fields. |
| `tldw_chatbook/Chat/console_chat_models.py` | Extend `ConsoleContextSnapshot` with metadata-only project-instruction state. |
| `tldw_chatbook/Chat/console_display_state.py` | Pure rail/Context display-state derivation. |
| `tldw_chatbook/Widgets/Console/console_project_instructions.py` | Compact rail row plus choose-folder/first-use/recovery modal views. |
| `tldw_chatbook/UI/Console_Modules/right_rail.py` | Mount the compact row above staged Sources and route its action to the existing Context surface. |
| `tldw_chatbook/Widgets/Console/console_context_modal.py` | Render the metadata-only Project Instructions section and explicit next-send payload view. |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Wire controller callbacks, modal results, snapshot construction, and mounted-state refresh. |

Keep these boundaries. In particular, do not put Textual calls into the resolver/runtime modules, do not put instruction bodies into `ConsoleChatSession`, and do not introduce a second tool-argument parser.

## Delivery 1 — Startup project context (`TASK-19634`)

### Task 1: Claim the delivery and pin the current schema/base

**Files:**
- Modify: `backlog/tasks/task-19634 - Add-startup-AGENTS.md-project-context-to-Console.md`
- Read: `backlog/docs/lessons-testing-evidence.md`
- Read: `backlog/docs/lessons-live-verification.md`
- Read: `backlog/docs/lessons-backlog-hygiene.md`

- [ ] **Step 1: Recheck in-flight work and the branch base.** Run `git branch -a --format='%(refname:short)' | rg -i 'agents-md|project-instruction'` and `gh api -X GET /search/issues -f q='repo:rmusser01/tldw_chatbook is:pr is:open AGENTS.md'`. Expected: no competing implementation; otherwise stop and reconcile.
- [ ] **Step 2: Confirm the schema head.** Run `rg -n '_CURRENT_SCHEMA_VERSION' tldw_chatbook/DB/ChaChaNotes_DB.py`. Expected on this plan's base: `32`. If different, rename and renumber the planned migration/tests before coding.
- [ ] **Step 3: Put the task in progress.** Run `backlog task edit 19634 -s "In Progress"` and add an Implementation Plan that links this file, TASK-19634, the accepted spec, and ADR-069. Re-read with `backlog task 19634 --plain`.
- [ ] **Step 4: Record the scoped baseline.** Re-run the six-file 71-test command from the plan preamble. Expected: all pass before feature edits.
- [ ] **Step 5: Commit task metadata only.** Commit message: `docs: start TASK-19634 AGENTS.md startup context`.

### Task 2: Add versioned control state, fingerprints, and config limits

**Files:**
- Create: `tldw_chatbook/Chat/console_project_instructions.py`
- Create: `Tests/Chat/test_console_project_instructions.py`
- Modify: `tldw_chatbook/config.py:1189-1223,2584-2589`
- Modify: `Tests/test_config_console_defaults.py:160-190`

- [ ] **Step 1: Write failing state-codec tests.** Cover explicit new-session enabled defaults; null/missing/malformed/forward-versioned legacy state disabled; exact four-field JSON round trip; no raw locator, source path, digest, endpoint, or body in the JSON.
- [ ] **Step 2: Write failing fingerprint tests.** Pin domain-separated SHA-256 outputs for canonical locator identity and provider destination identity. Changing provider/custom endpoint must change the notice key; changing only the model must not. URL credentials and paths must not appear in the visible destination label.
- [ ] **Step 3: Run the new tests.** Run `pytest Tests/Chat/test_console_project_instructions.py Tests/test_config_console_defaults.py -q`. Expected: failures because the module/config keys do not exist.
- [ ] **Step 4: Implement the minimal frozen contracts.** Use explicit parsing, never `asdict()` over an open-ended model:

```python
PROJECT_CONTEXT_VERSION = 1
EPHEMERAL_ORIGIN_KEY = "_chatbook_ephemeral_origin"

@dataclass(frozen=True, slots=True)
class ProjectInstructionControlState:
    project_instructions_enabled: bool
    working_folder_binding_id: str | None = None
    working_folder_locator_fingerprint: str | None = None
    project_instruction_notice_key: str | None = None

    @classmethod
    def new_session(cls) -> "ProjectInstructionControlState":
        return cls(project_instructions_enabled=True)

    @classmethod
    def legacy_disabled(cls) -> "ProjectInstructionControlState":
        return cls(project_instructions_enabled=False)
```

`encode_project_context_json()` must emit `version` plus only those four keys. `decode_project_context_json()` returns `legacy_disabled()` for any untrusted shape it cannot prove is v1.
- [ ] **Step 5: Add the two bounded config keys.** Normalize `project_instructions_startup_max_bytes` and `project_instructions_nested_max_bytes` with `coerce_int_setting`, default 32768, minimum 1, and a conservative implementation ceiling (1 MiB is enough; do not add more knobs). Update the config template comments.
- [ ] **Step 6: Re-run `python -m pytest Tests/Chat/test_console_project_instructions.py Tests/test_config_console_defaults.py -q`, then run `python -m ruff check tldw_chatbook/Chat/console_project_instructions.py tldw_chatbook/config.py Tests/Chat/test_console_project_instructions.py Tests/test_config_console_defaults.py`.** Expected: pass.
- [ ] **Step 7: Commit.** `git commit -m "feat(console): add project instruction control state"`.

### Task 3: Implement secure root resolver and admission

**Files:**
- Create: `tldw_chatbook/Agents/project_instruction_resolver.py`
- Create: `Tests/Agents/test_project_instruction_resolver.py`
- Create: `Tests/Agents/test_project_instruction_resolver_properties.py`

- [ ] **Step 1: Write precedence and boundary tests.** Cover root-only startup, `AGENTS.override.md` before `AGENTS.md`, whitespace-only override fallback, invalid/unreadable/oversized override suppressing standard fallback, BOM, strict UTF-8, sibling isolation, no global discovery, and no ascent above the binding root.
- [ ] **Step 2: Write secure-read tests.** Use `tmp_path` and monkeypatches to cover file symlinks, ancestor symlinks, POSIX `O_NOFOLLOW`/lstat-fstat identity changes, capped `limit + 1` growth, Windows `FILE_ATTRIBUTE_REPARSE_POINT`, and missing platform metadata. Missing security metadata must omit only that source with a content-free code.
- [ ] **Step 3: Write startup budget/property tests.** Pin whole-source admission, byte counts including BOM, canonical confinement, and O(1) discovery (startup visits exactly the binding-root directory). Use Hypothesis only for path/budget invariants, not filesystem race timing. Defer root-to-target traversal and deepest-first nested selection to Task 9.
- [ ] **Step 4: Run tests and confirm failure.** `pytest Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_resolver_properties.py -q`. Expected: import/behavior failures.
- [ ] **Step 5: Implement focused value objects and resolver.** The public seam should remain small:

```python
@dataclass(frozen=True, slots=True)
class InstructionSource:
    canonical_path: Path
    relative_path: str
    scope: str
    kind: Literal["override", "standard"]
    body: str
    byte_count: int
    digest: str

@dataclass(frozen=True, slots=True)
class InstructionOutcome:
    relative_path: str
    scope: str
    code: Literal[
        "omitted_byte_budget", "omitted_token_budget", "stale",
        "invalid", "resolution_failed",
    ]

@dataclass(frozen=True, slots=True)
class StartupInstructionCandidate:
    binding_id: str
    binding_root: Path
    locator_fingerprint: str
    dispatch_started_wall_ns: int
    source: InstructionSource | None
    outcomes: tuple[InstructionOutcome, ...]

@dataclass(frozen=True, slots=True)
class InstructionChainDelivery:
    source_digests: tuple[str, ...]
    outcomes: tuple[InstructionOutcome, ...]

@dataclass(frozen=True, slots=True)
class InstructionSnapshot:
    binding_id: str
    binding_root: Path
    locator_fingerprint: str
    dispatch_started_wall_ns: int
    startup_source: InstructionSource | None
    global_outcomes: tuple[InstructionOutcome, ...]
    primary_delivery: InstructionChainDelivery
    warning_codes: tuple[str, ...]

class ProjectInstructionResolver:
    def resolve_startup(self, *, binding_id: str, binding_root: Path,
                        locator_fingerprint: str, max_bytes: int,
                        dispatch_started_wall_ns: int) -> StartupInstructionCandidate: ...
```

The canonical path and digest are memory-only and must have no serializer/display field; `digest` is SHA-256 over the exact raw file bytes (including an optional BOM). The candidate is the byte-admitted, securely pinned controller-to-service handoff; `AgentService` freezes the final `InstructionSnapshot` after the primary chain's token admission. `startup_source` remains the selected pinned source even if one chain token-omits it; `primary_delivery` records exactly what the primary request received, while the run context derives a separate `InstructionChainDelivery` for each child and Delivery 2 reuses that contract for nested revisions. Use `time.time_ns()` only for the filesystem-mtime cutoff and a separate monotonic clock for performance. Use standard-library descriptor reads; read at most `max_bytes + 1`; compare file and every ancestor identity before/after. Never log `body`, `canonical_path`, `digest`, or raw exception text.
- [ ] **Step 6: Add token admission as an injected function.** The resolver owns raw bytes; a pure `admit_sources(sources, safe_input_tokens, count_tokens)` helper owns whole-source token admission. Reuse `Utils.token_counter.count_tokens_messages`/`get_model_token_limit` at the bridge boundary rather than importing provider state here.
- [ ] **Step 7: Re-run `python -m pytest Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_resolver_properties.py -q`, then run `python -m ruff check tldw_chatbook/Agents/project_instruction_resolver.py Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_resolver_properties.py`.** Expected: pass.
- [ ] **Step 8: Commit.** `git commit -m "feat(agents): resolve AGENTS.md guidance safely"`.

### Task 4: Add local-only conversation persistence and migration

**Files:**
- Create: `tldw_chatbook/DB/migrations/chachanotes_v41_to_v42_console_project_context.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:165,284-305,4684-4742,4836-4910,7890-7965`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py:19-380`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:74-315,315-470,519-690,2532-2790`
- Create: `Tests/DB/test_chachanotes_console_project_context_migration.py`
- Create: `Tests/Chat/test_console_chat_store_project_instructions.py`
- Modify: `Tests/Chatbooks/test_chatbook_importer.py`

- [ ] **Step 1: Seed failing migration tests from a real v41 database.** Add named cases `test_v41_to_v42_adds_nullable_local_column`, `test_v41_to_v42_recovers_column_present_version_still_41`, `test_v41_to_v42_rejects_wrong_start_version`, and `test_fresh_schema_contains_console_project_context_column`. Assert v42, nullable `console_project_context_json`, rollback/version guards, v41-to-v42 data preservation, and fresh database availability. Use only `tmp_path`/in-memory DBs. The original delivery used v32→v33; the PR replay renumbered it after `dev` acquired v33→v41.
- [ ] **Step 2: Write failing local-write and preservation tests.** A set/clear round trip must not change conversation `version`/`last_modified`, create `sync_log`, or alter synchronized payloads. Ordinary `update_conversation`, soft delete, restore, and restart must preserve the value. A forced adapter write failure keeps the updated in-memory control state, emits only `project_instruction_state_write_failed`, and does not touch synchronized metadata. Add a contract assertion/docstring at the DB accessor boundary stating that any future inbound conversation apply/sync path must update an explicit synchronized-column allowlist and preserve this column through create/update/delete/undelete/replay/conflict handling.
- [ ] **Step 3: Write import conflict tests.** `SKIP` must leave an existing local value untouched. Current non-skip paths, including `REPLACE`, must create a separate row with null project context and leave the existing row unchanged.
- [ ] **Step 4: Run `python -m pytest Tests/DB/test_chachanotes_console_project_context_migration.py Tests/Chat/test_console_chat_store_project_instructions.py Tests/Chatbooks/test_chatbook_importer.py -q`.** Expected: named failures for the missing fresh-schema column, v41→v42 migration/accessors, state lifecycle, and importer preservation.
- [ ] **Step 5: Implement the fresh schema and rollback-safe additive migration.** Add nullable `console_project_context_json TEXT` to the canonical `CREATE TABLE conversations`; set `_CURRENT_SCHEMA_VERSION = 42`; add `_MIGRATE_V41_TO_V42_SQL`, `_migrate_from_v41_to_v42`, and migration-map entry `41`. The method requires start version 41, checks `PRAGMA table_info(conversations)`, executes the `ALTER TABLE` only when the column is absent, then performs the guarded `41 -> 42` version update in the surrounding rollback-safe transaction. A database left with the column present and version 41 completes by updating only the version; any other partial/error state raises `SchemaError` and leaves version 41.
- [ ] **Step 6: Add explicit local-only accessors and the future-sync extension contract.** Add `get_conversation_console_project_context()` and `set_conversation_console_project_context()` as bare, parameterized queries; the setter must not touch watched columns. Its Google-style docstring names the local-only preservation invariant for future inbound apply/sync code. Tests inspect current `conversations_sync_*` trigger SQL and importer/mutation paths to prove the column is excluded or preserved; do not invent an inbound service.
- [ ] **Step 7: Extend the persistence protocol/service and session lifecycle.** Add one `project_instruction_state` field to `ConsoleChatSession` whose dataclass default factory is `ProjectInstructionControlState.legacy_disabled`. Only `ConsoleChatStore.create_session()` explicitly supplies `new_session()`; every direct construction and every restore therefore fails closed unless new-session creation opted in. `restore_persisted_session()` decodes the database value and defaults legacy-disabled. Update in-memory state before attempting its optional local-only write; on failure retain it, post only `project_instruction_state_write_failed` with “may not survive restart” copy, and never fall back to synchronized metadata. Temporary sessions keep state in memory; `promote_ephemeral_session()` first completes ordinary durable conversation promotion, then best-effort writes project context. A forced project-context write failure leaves the conversation/session durably promoted and usable while retaining the in-memory choice and warning; it must not roll back or fail ordinary promotion.
- [ ] **Step 8: Re-run the exact Step-4 command.** Expected: pass, including partial-migration recovery, fail-open optional write behavior, durable promotion, and no sync/version churn.
- [ ] **Step 9: Commit.** `git commit -m "feat(db): persist Console project context locally"`.

### Task 5: Assemble startup rider, binding consent, and selected-root tools

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:3244-3375,6380-6515,7163-7395`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py:744-908,1412-2050`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py:1499-1615,1747-1835`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py:106-166,744-930`
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:1137-1405,3140-3378` (Anthropic/Gemini marker-aware native conversion)
- Modify: `tldw_chatbook/Agents/local_tool_provider.py:93-220,680-940`
- Modify: `tldw_chatbook/Agents/agent_models.py:239-247`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Create: `Tests/Chat/test_console_agent_project_instructions.py`
- Modify: `Tests/Chat/test_console_local_review_hook.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `Tests/Chat/test_chat_functions.py`
- Modify: `Tests/Chat/test_anthropic_native_tools.py`
- Modify: `Tests/Chat/test_google_native_tools.py`
- Modify: `Tests/Chat/test_console_rewind_summarize.py`

- [ ] **Step 1: Write dispatch gating tests.** New agent sessions with one eligible binding auto-select it; zero/multiple bindings hold the send for recovery; removed/unauthorized/missing/retargeted bindings never silently retarget. Direct/plain and character-forced-plain sends never discover or transmit project instructions.
- [ ] **Step 2: Write consent-key and captured-snapshot tests at the controller/service seam.** Consent occurs before any provider request even when no root file exists. The notice names the final snapshot's relative root source, scope, byte count, and omission/warning state without a body. Proceed stores the destination-scoped notice key and performs no second filesystem read; cancel aborts the send and discards the candidate/snapshot; disable does the same and turns the feature off. Provider/custom endpoint changes re-prompt; model-only changes do not.
- [ ] **Step 3: Write startup transport/leak tests with a sentinel body.** Text, multimodal, retry, regenerate, and continue agent sends include the labeled rider exactly once. Parent and newly spawned subagent model chains each receive the same immutable root snapshot exactly once on their initial provider context, without Delivery-2 nested ledger/cursor machinery. The sentinel is absent from store messages, AgentRunsDB steps, run log, exception/log capture, `/rewind` input, and any automatic tool result. An explicit `fs_read` or assistant quotation remains normally persisted.
- [ ] **Step 4: Write selected-root/read-only tests.** Enabled sessions compose `LocalToolProvider` at the validated binding root; disabled/legacy sessions retain `[console] workspace_root`/cwd behavior. Read-only bindings omit `fs_write`, `fs_edit`, and `fs_patch` from the catalog while retaining read/git tools and instruction loading.
- [ ] **Step 5: Run `python -m pytest Tests/Agents/test_agent_service.py Tests/Chat/test_console_agent_project_instructions.py Tests/Chat/test_console_local_review_hook.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_chat_functions.py Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_google_native_tools.py Tests/Chat/test_console_rewind_summarize.py -q`.** Expected: failures for the absent candidate/snapshot, exact request/payload budget, consent, provider-matrix sanitization, native grouping, rider, and selected-root behavior.
- [ ] **Step 6: Capture the startup candidate before notice handling.** After provider resolution and before `_run_agent_reply` performs a provider call, re-resolve the binding, compare its locator fingerprint, securely resolve the byte-bounded `StartupInstructionCandidate`, and compose the selected-root provider. Compute whether the destination notice key is already acknowledged, then pass the candidate plus a controller-owned `confirm_project_instruction_dispatch(snapshot) -> Literal["proceed", "cancel", "disable"]` callback into `ConsoleAgentBridge.run_reply()`. Do not display a notice or write its key yet, and never reread the root body after this point.
- [ ] **Step 7: Reuse one exact model-request builder for payload admission and sending.** Add backward-compatible `AgentConfig.response_reserve_tokens: int = 2048`; `ConsoleAgentBridge` explicitly sets it to `resolution.max_tokens or DEFAULT_RESPONSE_RESERVATION`, and children inherit it unchanged. Refactor `AgentService._make_call_model()` so a pure `_build_model_request(messages, active_schemas)` returns the exact bounded `messages` plus native `tools` payload that `chat_call()` will send, including the ordinary system prompt, fence protocol or native schemas (not both), staged/compacted history, and run-log prompt. A pure `safe_project_instruction_tokens(request, candidate_rows)` subtracts `response_reserve_tokens` from `get_model_token_limit`, `count_tokens_messages`, and `estimate_tokens(json.dumps(native_tools, sort_keys=True, separators=(",", ":")))`; wrapper labels and candidate rows are counted, unknown/raising/nonpositive limits return zero, and native schemas are not double-counted in fenced mode.
- [ ] **Step 8: Freeze the exact snapshot, then request consent.** The primary `_run_one` builds the exact first request, whole-source admits or token-omits the candidate, freezes `InstructionSnapshot`, and only then invokes `confirm_project_instruction_dispatch`. Proceed acknowledges the key and reuses that snapshot without a read or rebuild; cancel/disable discard it and return before `chat_call`. Append an admitted source only to the run-local message copy as a tagged user row wrapped by `[Project instructions — untrusted repository context]`. The same `AgentService` path injects the active startup snapshot before every child's first provider call.
- [ ] **Step 9: Preserve and consume the marker through one provider-complete boundary.** Attach `EPHEMERAL_ORIGIN_KEY` internally and keep it through `ConsoleProviderGateway._chat_api_kwargs()`. In `Chat_Functions.py`, add explicit `EPHEMERAL_GROUPING_ENDPOINTS = frozenset({"anthropic", "google"})` plus a copy-only sanitizer: for those two endpoints preserve the marker for native grouping; for every other key in `API_CALL_HANDLERS` strip it before constructing handler kwargs. Anthropic/Gemini consume it in `LLM_API_Calls.py`; the gateway consumes it only for direct llama.cpp requests that bypass `chat_api_call`. Anthropic coalesces all `tool_result` blocks and the following project-context text block into one user turn; Gemini emits all function responses before a separate user text part/turn. Every other OpenAI-compatible/cloud/local/custom/fenced handler receives ordinary rows with no internal key; fenced/local content closes the complete results fence before a separately labeled context section. Add a parametrized parity test over `sorted(API_CALL_HANDLERS)` so adding a provider without sanitizer coverage fails. There is no current durable exchange-capture subsystem to modify; the origin tag defines the omission boundary any such subsystem must honor later.
- [ ] **Step 10: Keep `/rewind` ordering intact.** Pin with a regression test that `_apply_context_summary_compaction()` runs before the startup row is appended. Do not add compaction logic.
- [ ] **Step 11: Filter write specs when composing the selected-root local provider.** Add one explicit `allow_write` constructor option or a small filtered-spec helper; do not mutate the global default spec table.
- [ ] **Step 12: Re-run the exact Step-5 pytest command, then run `python -m ruff check tldw_chatbook/Chat/console_project_instructions.py tldw_chatbook/Agents/project_instruction_resolver.py tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/Chat_Functions.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_console_agent_project_instructions.py Tests/Chat/test_console_local_review_hook.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_chat_functions.py Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_google_native_tools.py Tests/Chat/test_console_rewind_summarize.py --ignore F821`, followed by `python -m ruff check tldw_chatbook/Agents/agent_service.py --select F821 --output-format=concise`.** Expected: the first command passes and the second reports exactly the single pre-existing `RunLogWriter` finding; no new Ruff finding is allowed.
- [ ] **Step 13: Commit.** `git commit -m "feat(console): send startup AGENTS.md context"`.

### Task 6: Add basic rail, Context, chooser, and first-use UI

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py:529-545`
- Modify: `tldw_chatbook/Chat/console_display_state.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Create: `tldw_chatbook/Widgets/Console/console_project_instructions.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/UI/Console_Modules/right_rail.py:150-245`
- Modify: `tldw_chatbook/Widgets/Console/console_context_modal.py:40-230`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py:1717-1795`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:2200-2310,9860-10330,12890-13065`
- Create: `Tests/UI/test_console_project_instructions.py`
- Modify: `Tests/UI/test_console_context_modal.py`
- Modify: `Tests/UI/test_console_right_rail.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write pure display-state tests.** Pin `Off`, `Choose folder`, `None`, `<N> loaded`, and `Warning`, with metadata rows for binding label, locator-match state, relative source, scope, byte count, outcome, and warning code. No body field exists in the display state.
- [ ] **Step 2: Write modal behavior tests.** The setup mode selects exactly one eligible binding or disables/cancels; the notice mode has Proceed/Cancel/Disable and sanitized destination copy; recovery never enables stale entries. Use implemented single-letter bindings only and ensure footer hints match actions.
- [ ] **Step 3: Write screen-state compatibility tests.** The four fields round-trip explicitly through `UI/Console_Modules/session.py`; absent/malformed/forward-versioned legacy state restores disabled. No raw path/body is serialized.
- [ ] **Step 4: Write rail/Context and side-effect-free preview tests.** The row mounts above Sources, opens the existing Context modal, and refreshes in place. Add pure `build_project_instruction_preview(base_messages, startup_candidate, request_builder) -> ProjectInstructionPreview` that clones the candidate/control state into a disposable preview context, calls the same rider and exact request builders against copied messages, and returns metadata plus the exact next-send payload. Opening/closing preview must not mutate the session transcript or control state, acknowledge consent, pin/activate a source in the live dispatch context, consume live byte/token budgets, create/advance a ledger cursor/receipt, or change subsequent dispatch results. The preview may securely reread the root into its disposable candidate, but actual dispatch always captures a fresh candidate.
- [ ] **Step 5: Run `python -m pytest Tests/UI/test_console_project_instructions.py Tests/UI/test_console_context_modal.py Tests/UI/test_console_right_rail.py Tests/UI/test_console_native_chat_flow.py -q`.** Expected: failures for missing widget/display state, modal decisions, screen-state codec, and session-scoped callback wiring.
- [ ] **Step 6: Implement the smallest widget and preview surface.** One feature module may contain the compact row and its two modal modes; do not create a file browser/editor. Keep untrusted labels `markup=False`. Controller preview construction must instantiate only the disposable preview context from Step 4 and discard it when the modal closes; it must never call the live consent callback or reuse a live `InstructionActivationLedger`.
- [ ] **Step 7: Wire async controller callbacks from `ChatScreen`.** The worker/controller requests a decision; only the Textual main loop mounts/dismisses modals. Background/parked sessions must scope the decision and disposable preview to their own session ID, not whichever session is visible.
- [ ] **Step 8: Re-run the exact Step-5 command, then run `python -m ruff check tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_display_state.py tldw_chatbook/Widgets/Console/console_project_instructions.py tldw_chatbook/UI/Console_Modules/right_rail.py tldw_chatbook/Widgets/Console/console_context_modal.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_project_instructions.py Tests/UI/test_console_context_modal.py Tests/UI/test_console_right_rail.py Tests/UI/test_console_native_chat_flow.py`.** Expected: pass at 80x24, 100x30, and 140x40 pilot sizes.
- [ ] **Step 9: Commit.** `git commit -m "feat(console): show project instruction status"`.

### Task 7: Verify and close Delivery 1

**Files:**
- Modify: `backlog/tasks/task-19634 - Add-startup-AGENTS.md-project-context-to-Console.md`

- [ ] **Step 1: Run `python -m pytest Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_resolver_properties.py Tests/Chat/test_console_project_instructions.py Tests/Chat/test_console_chat_store_project_instructions.py Tests/Chat/test_console_agent_project_instructions.py Tests/Chat/test_console_local_review_hook.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_google_native_tools.py Tests/Chat/test_console_rewind_summarize.py Tests/DB/test_chachanotes_console_project_context_migration.py Tests/Chatbooks/test_chatbook_importer.py Tests/Workspaces/test_workspace_folder_bindings.py Tests/UI/test_console_project_instructions.py Tests/UI/test_console_context_modal.py Tests/UI/test_console_right_rail.py Tests/UI/test_console_native_chat_flow.py Tests/test_config_console_defaults.py -q`.** Expected: all pass.
- [ ] **Step 2: Apply the repository's verified formatter policy.** This repository has no whole-repo formatter gate, and the existing large seams already fail `ruff format --check`; do not normalize them in this feature. Set `CHATBOOK_AGENTS_NEW_PY=(tldw_chatbook/Chat/console_project_instructions.py tldw_chatbook/Agents/project_instruction_resolver.py tldw_chatbook/Widgets/Console/console_project_instructions.py Tests/Chat/test_console_project_instructions.py Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_resolver_properties.py Tests/DB/test_chachanotes_console_project_context_migration.py Tests/Chat/test_console_chat_store_project_instructions.py Tests/Chat/test_console_agent_project_instructions.py Tests/UI/test_console_project_instructions.py)` and run `python -m ruff check ${CHATBOOK_AGENTS_NEW_PY[@]}` plus `python -m ruff format --check ${CHATBOOK_AGENTS_NEW_PY[@]}`. Set `CHATBOOK_AGENTS_EXISTING_PY=(tldw_chatbook/config.py tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/Chat_Functions.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_display_state.py tldw_chatbook/UI/Console_Modules/right_rail.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_context_modal.py Tests/test_config_console_defaults.py Tests/Chatbooks/test_chatbook_importer.py Tests/Agents/test_agent_service.py Tests/Chat/test_console_local_review_hook.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_chat_functions.py Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_google_native_tools.py Tests/Chat/test_console_rewind_summarize.py Tests/UI/test_console_context_modal.py Tests/UI/test_console_right_rail.py Tests/UI/test_console_native_chat_flow.py)`; run `python -m ruff check ${CHATBOOK_AGENTS_EXISTING_PY[@]} --ignore F821`, then `python -m ruff check ${CHATBOOK_AGENTS_EXISTING_PY[@]} --select F821 --output-format=concise`. The latter must contain exactly the pre-existing `agent_service.py: RunLogWriter` finding and no addition. Run `git diff --check 5047b6962...HEAD`. Expected: new files pass both gates, existing files add no lint diagnostic, and whitespace is clean.
- [ ] **Step 3: Set `CHATBOOK_AGENTS_QA_DIR=/tmp/chatbook-agents-md-delivery1`, create its `pytest`, `sqlite`, and `runlog` children, and run the sentinel cases from Step 1 with fixture value `CHATBOOK_AGENTS_SENTINEL_7d1e9c` and that artifact root. Dump each temporary SQLite database through the test helper, then run `rg -n 'CHATBOOK_AGENTS_SENTINEL_7d1e9c' /tmp/chatbook-agents-md-delivery1`.** Expected: the test's allowlisted provider-spy artifact is the only body-bearing match; SQLite dumps, ordinary pytest capture, and run logs contain none.
- [ ] **Step 4: Review the diff against TASK-19634 only.** Delivery 1 must not contain `prepare_tool_calls`, nested path mapping, or subagent ledger code.
- [ ] **Step 5: Complete the Backlog task.** Check every AC, add concise Implementation Notes including ADR-069 and exact verification, then `backlog task edit 19634 -s Done`. Re-read the task after the CLI mutation.
- [ ] **Step 6: Commit closeout.** `git commit -m "docs: close TASK-19634 startup project context"`.

## Delivery 2 — Nested path activation (`TASK-19635`)

### Task 8: Add registry-owned path-target mapping

**Files:**
- Modify: `backlog/tasks/task-19635 - Add-nested-AGENTS.md-activation-before-Console-tools.md`
- Modify: `tldw_chatbook/Agents/tool_catalog.py:343-350,821-1015`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `tldw_chatbook/Tools/patch_tool_impls.py:105-178,370-450`
- Modify: `Tests/Agents/test_tool_catalog.py`
- Modify: `Tests/Agents/test_tool_catalog_owner_cache.py`
- Create: `Tests/Agents/test_project_instruction_path_targets.py`
- Modify: `Tests/Tools/test_patch_tool_impls.py`

- [ ] **Step 1: Recheck base/in-flight work, mark TASK-19635 In Progress, and add its task Implementation Plan.** Re-read with `--plain`.
- [ ] **Step 2: Write first-wins owner tests.** Register colliding builtin/local/skill/MCP fakes in different orders. `resolve_owner_for_name(name)` must return the exact `(tool_id, provider)` used by `invoke_by_name`; preflight must never call shadowed providers.
- [ ] **Step 3: Write the complete local mapping matrix.** Assert: `fs_read`/`fs_write`/`fs_edit` use the exact target's parent chain; `fs_list` uses the listed-directory chain; `fs_glob`/`fs_grep` use only the binding root regardless of pattern prefixes; `fs_patch` uses the union of parsed create/modify parent chains for real and `dry_run` calls while invalid/delete/rename forms preserve existing parser errors; `git_branches`, unfiltered `git_diff`, unfiltered `git_log`, and every `git_status` call (including `path`) use only the discovered repository-root chain; path-filtered `git_diff(path)`/`git_log(path)` use repository root through the directory target or lexical parent for file/deleted targets, while `commit_range`, `staged`, and `stat` do not alter scope; `git_blame(path)` uses repository root through the file parent; web/todo/spawn/process/skill-script/MCP/opaque tools return no targets.
- [ ] **Step 4: Write built-in mapping tests.** `read_file`/`write_file` exact parent and `list_directory` directory scope inside the selected binding; another authorized binding returns an outside-instruction-scope target, not a second hierarchy. Disabled built-ins report no targets.
- [ ] **Step 5: Run `python -m pytest Tests/Agents/test_tool_catalog.py Tests/Agents/test_tool_catalog_owner_cache.py Tests/Agents/test_project_instruction_path_targets.py Tests/Tools/test_patch_tool_impls.py -q`.** Expected: failures for the absent path-aware protocol, atomic owner resolver, and shared patch-target parser.
- [ ] **Step 6: Add the structural contracts.** Keep them immutable and provider-neutral:

```python
@dataclass(frozen=True, slots=True)
class ToolPathTarget:
    path: Path | None
    kind: Literal["exact", "directory", "repository", "outside"]

@runtime_checkable
class PathAwareToolProvider(Protocol):
    def path_targets(self, tool_id: str, args: Mapping[str, Any]) -> tuple[ToolPathTarget, ...]: ...
```

Expose a single registry method that atomically returns the cached first-wins `(tool_id, owner)`; make `invoke_by_name()` use that same result rather than resolving twice across a possible cache race.
- [ ] **Step 7: Reuse the patch parser.** Add a pure `parse_patch_targets(diff_text)` wrapper around `parse_unified_diff()` and make both execution and preflight call it. Deletes/renames/invalid forms keep existing errors; no copied regex/parser.
- [ ] **Step 8: Re-run the exact Step-5 command and `python -m ruff check tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Tools/patch_tool_impls.py Tests/Agents/test_tool_catalog.py Tests/Agents/test_tool_catalog_owner_cache.py Tests/Agents/test_project_instruction_path_targets.py Tests/Tools/test_patch_tool_impls.py`.** Expected: pass.
- [ ] **Step 9: Commit.** `git commit -m "feat(agents): map tool calls to instruction scopes"`.

### Task 9: Add shared activation ledger and lazy nested resolution

**Files:**
- Create: `tldw_chatbook/Agents/project_instruction_runtime.py`
- Modify: `tldw_chatbook/Agents/project_instruction_resolver.py`
- Create: `Tests/Agents/test_project_instruction_runtime.py`
- Create: `Tests/Agents/test_project_instruction_concurrency.py`

- [ ] **Step 1: Write ledger, child-initial-context, and nested-receipt tests.** Root snapshot is active at dispatch; nested sources pin once; parent/subagents share source outcomes and byte budget; each chain has its own delivery revision/cursor. `initial_context_for_chain(chain_id, payload_state)` returns every currently active root/nested row plus an `InstructionDeliveryReceipt`; the cursor advances only when `mark_payload_sent(receipt)` verifies those receipt-tagged rows are included in that chain's outgoing payload. A nested `prepare` result follows the same receipt path on the next model call. Cover a child spawned after nested activation, a child whose first request races a later activation, successful nested delivery, provider failure after request construction, and identical retry without repeated deferral.
- [ ] **Step 2: Write outcome and token-headroom tests.** Byte/stale/invalid/read failures are global terminal no-content outcomes; token omission is per chain. Cover small and unknown/raising model windows, large native schemas, long histories, wrapper/deferral overhead, nonpositive allowance, and two chains with different headroom. Each unseen outcome defers/warns that chain exactly once; identical retry proceeds.
- [ ] **Step 3: Write deterministic concurrency tests.** Use barriers, never sleeps. First lock wins the remaining nested budget; deepest-first admission inside a batch is deterministic; later chains receive explicit omissions.
- [ ] **Step 4: Write lazy discovery tests and the exact batch contract.** Add immutable `NestedResolutionBatch(sources: tuple[InstructionSource, ...], outcomes: tuple[InstructionOutcome, ...])` and `resolve_targets(binding_root, targets, *, max_bytes, dispatch_started_wall_ns, pinned_by_canonical_path) -> NestedResolutionBatch`. Resolution walks only root-to-target chains (O(depth)), applies deterministic deepest-first admission with broad-to-specific rendering, skips created/changed-after-dispatch candidates, retains already pinned content after edit/delete, and never walks sibling subtrees. `pinned_by_canonical_path` is memory-only and values are reused by identity without rereading.
- [ ] **Step 5: Run `python -m pytest Tests/Agents/test_project_instruction_runtime.py Tests/Agents/test_project_instruction_concurrency.py Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_resolver_properties.py -q`.** Expected: import/contract failures for the absent runtime ledger and nested resolver.
- [ ] **Step 6: Implement one lock-owned ledger, content-free receipt, and per-chain payload state.** Add immutable `InstructionDeliveryReceipt(receipt_id, chain_id, through_revision, source_digests, outcome_keys, row_keys)`; it contains no body/path beyond already-approved relative outcome keys and is never serialized. The ledger stores exact `InstructionSource`/`InstructionOutcome` values, remaining raw budget, dispatch wall-clock cutoff, activation revision, and warning keys. Per-chain state stores delivered revision/outcome keys plus a run-local `InstructionChainPayloadState`; it never stores bodies outside the shared source objects. `InstructionChainPayloadState.capture(messages, active_schemas, calls)` is refreshed by `AgentRuntime` immediately before preparation and uses AgentService's same `_build_model_request` plus the runtime's canonical `build_project_instruction_deferral_rows(calls)` to calculate headroom for prospective context rows. Neither type has persistence methods.
- [ ] **Step 7: Implement `prepare(calls, chain_id, registry, payload_state)` as orchestration around injected resolver/owner lookups.** Union targets before locking; recheck/admit deepest-first under the lock; call `payload_state.safe_input_tokens(candidate_rows)` so the exact current system prompt, compacted/current messages, disclosed fence/native schemas, staged context, prospective deferral stubs, wrapper labels, and response reserve are counted. `retry_with_context` returns broad-to-specific tagged rows plus one receipt whose `row_keys` are copied into internal-only row metadata; the ledger does not advance yet. On the next `_build_model_request`, AgentService verifies every receipt row survived request bounding, then calls `mark_payload_sent(receipt)` immediately before `chat_call`; a provider exception after that point leaves the cursor advanced because that exact payload was attempted. If the rows are absent, do not send or mark them—return the existing terminal overflow/error path. A missing, raising, or nonpositive allowance records per-chain `omitted_token_budget`. Outside-binding targets add a content-free warning only.
- [ ] **Step 8: Re-run the exact Step-5 command, then run `python -m ruff check tldw_chatbook/Agents/project_instruction_resolver.py tldw_chatbook/Agents/project_instruction_runtime.py Tests/Agents/test_project_instruction_runtime.py Tests/Agents/test_project_instruction_concurrency.py Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_resolver_properties.py`.** Expected: pass.
- [ ] **Step 9: Commit.** `git commit -m "feat(agents): track nested project instruction activation"`.

### Task 10: Add preparation-before-review and canonical deferral

**Files:**
- Modify: `tldw_chatbook/Agents/agent_runtime.py:208-275,455-650`
- Modify: `tldw_chatbook/Agents/agent_service.py:268-335,1280-1335`
- Modify: `Tests/Agents/test_agent_runtime_review_hook.py`
- Create: `Tests/Agents/test_agent_runtime_preparation.py`
- Modify: `Tests/Agents/test_agent_service.py`
- Modify: `Tests/Agents/test_agent_service_review_state_scope.py`

- [ ] **Step 1: Write typed-result and receipt tests.** Only `proceed` and `retry_with_context` construct successfully; proceed carries neither rows nor receipt; retry requires tagged ephemeral rows and exactly one content-free `InstructionDeliveryReceipt`. Receipt row keys must match the returned rows and reject bodies/absolute paths.
- [ ] **Step 2: Write ordering/atomicity/payload-state tests.** Runtime captures the exact current `messages`, active schemas, and complete call batch in the chain payload state immediately before preparation. Preparation receives the entire call batch once. Retry creates one fixed tool-result stub per original call, preserves ID/name/order/cardinality, skips review and execution, appends the separate receipt-tagged context row, stages the receipt for the next request, then loops back to the model. Cursor advancement is absent at preparation time and occurs only when AgentService verifies the staged rows in the built payload.
- [ ] **Step 3: Write exception tests.** A preparation exception emits only `project_instruction_preparation_failed` plus tool names/count through `on_ephemeral_runtime_warning`, logs no exception/traceback/body, and proceeds into unchanged `review_tool_calls`. Warning-callback failure is swallowed with code-only logging.
- [ ] **Step 4: Write no-hook and existing-review regression tests.** Absent preparation remains byte-identical; existing review fail-open/fail-closed ownership and verdict strings do not change.
- [ ] **Step 5: Run `python -m pytest Tests/Agents/test_agent_runtime_review_hook.py Tests/Agents/test_agent_runtime_preparation.py Tests/Agents/test_agent_service.py Tests/Agents/test_agent_service_review_state_scope.py -q`.** Expected: missing preparation result/hooks, payload-state capture, and child-initial-context wiring.
- [ ] **Step 6: Implement frozen result types and LoopDeps hooks.** `AgentRuntime` owns stubs; the hook never returns tool results or review verdicts:

```python
@dataclass(frozen=True, slots=True)
class ToolBatchPreparation:
    status: Literal["proceed", "retry_with_context"]
    ephemeral_rows: tuple[Mapping[str, Any], ...] = ()
    delivery_receipt: InstructionDeliveryReceipt | None = None
```

- [ ] **Step 7: Thread exact chain context and staged receipts through `AgentService`.** Add optional `project_instruction_context` and stable `chain_id` inputs to `_run_one`; primary and inline spawned children share the context but get distinct IDs. Before each chain's first call, `_build_model_request` asks `initial_context_for_chain` for the currently active root/nested rows. For both initial and later nested receipts, it includes the tagged rows in the exact outgoing request, verifies every `row_key` survived request bounding, and calls `mark_payload_sent(receipt)` immediately before `chat_call`; provider failure after request construction cannot re-defer the same revision on retry. Immediately before later preparation, `AgentRuntime` refreshes the same chain's `InstructionChainPayloadState`; the unchanged one-argument `prepare_tool_calls(calls)` closure reads that state and returns the staged receipt. Extend `review_state_scope` tests so a child cannot clobber the parent's staged receipt/cursor.
- [ ] **Step 8: Re-run the exact Step-5 command and `python -m ruff check tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Agents/agent_service.py Tests/Agents/test_agent_runtime_review_hook.py Tests/Agents/test_agent_runtime_preparation.py Tests/Agents/test_agent_service.py Tests/Agents/test_agent_service_review_state_scope.py --ignore F821`, then run `python -m ruff check tldw_chatbook/Agents/agent_service.py --select F821 --output-format=concise`.** Expected: tests and the first Ruff command pass; the second contains exactly the pre-existing `RunLogWriter` finding.
- [ ] **Step 9: Commit.** `git commit -m "feat(agents): prepare tool batches before review"`.

### Task 11: Wire nested preparation and provider grammar without persistence leaks

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py`
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `Tests/Chat/test_chat_functions.py`
- Modify: `Tests/Chat/test_anthropic_native_tools.py`
- Modify: `Tests/Chat/test_google_native_tools.py`
- Create: `Tests/Chat/test_console_project_instruction_provider_grammar.py`
- Create: `Tests/Chat/test_console_project_instruction_persistence_boundary.py`
- Modify: `Tests/Chat/test_console_agent_bridge_local.py`

- [ ] **Step 1: Write end-to-end fake-provider tests.** A multi-call batch targeting a new nested scope defers before approval/execution, sends stubs then ephemeral context, and allows the reconsidered call through normal approval. Non-path-aware/opaque calls do not activate nested guidance.
- [ ] **Step 2: Write exact outgoing-payload grammar and provider-parity tests.** Mock final handler/HTTP sends, not only gateway input. OpenAI-compatible: every tool response row precedes a separate user context row. Gemini: every `functionResponse` precedes a separate user text part/turn. Anthropic: one user turn contains all `tool_result` blocks first and one distinct context text block last. Fenced/local: the complete results fence closes before a separately labeled context section. Parametrize over every `API_CALL_HANDLERS` key: Anthropic/Google may see the marker only inside their marker-aware converter and must consume it before HTTP; all other handlers must receive copied rows with no marker. Context text never appears in a tool result.
- [ ] **Step 3: Write parent/subagent delivery-receipt tests.** Both share admission but each receives unseen revisions before execution. A concurrently activated source cannot execute in another chain until that chain receives it. Assert preparation alone leaves the cursor unchanged, the next provider payload contains all receipt rows and advances it once, a provider failure after that payload does not re-defer the revision, and an identical reconsidered call proceeds without a loop.
- [ ] **Step 4: Write persistence-boundary tests with a sentinel.** Automatic bodies are absent from `AgentStep`, AgentRunsDB, run log, transcript/context event, review verdict, exception, and application log. Source-relative metadata/warning codes may appear. Explicit reads/model quotations remain untouched.
- [ ] **Step 5: Run `python -m pytest Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_chat_functions.py Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_google_native_tools.py Tests/Chat/test_console_project_instruction_provider_grammar.py Tests/Chat/test_console_project_instruction_persistence_boundary.py Tests/Chat/test_console_agent_bridge_local.py -q`.** Expected: failures for absent ledger/preparation wiring, all-handler sanitization, final-adapter grouping, and persistence omission.
- [ ] **Step 6: Upgrade the Delivery-1 run context to one dispatch ledger in `ConsoleAgentBridge.run_reply()`.** Construct it from the captured `StartupInstructionCandidate`; when primary `AgentService._run_one` freezes the exact token-admitted `InstructionSnapshot`, call `accept_primary_snapshot(snapshot)` to seed root source/outcomes and the primary cursor. Capture the per-run registry's exact owner cache and pass the same context plus preparation/warning callbacks into parent and children. Never serialize either object.
- [ ] **Step 7: Add transport serialization at the provider-complete final boundary.** Canonical runtime order stays transport-independent. The gateway preserves tags while routing to `chat_api_call`; `Chat_Functions.py` strips them for every handler except the explicit Anthropic/Google marker-aware set; those two converters consume them during native grouping; the direct llama.cpp builder consumes them when it bypasses the dispatcher. Native coalescing follows Step 2 exactly. Fenced/local rendering uses the runtime's canonical result-stub helper and closes the full results section before its context section. No provider request body or generic handler kwargs retain the marker.
- [ ] **Step 8: Post metadata-only activation events.** Use relative sources/scopes and outcome codes. Do not route them through `on_step` or transcript messages.
- [ ] **Step 9: Re-run the exact Step-5 command, then run `python -m ruff check tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/Chat_Functions.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_models.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_chat_functions.py Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_google_native_tools.py Tests/Chat/test_console_project_instruction_provider_grammar.py Tests/Chat/test_console_project_instruction_persistence_boundary.py Tests/Chat/test_console_agent_bridge_local.py`.** Expected: no new Ruff finding beyond the recorded base diagnostic set.
- [ ] **Step 10: Commit.** `git commit -m "feat(console): activate nested AGENTS.md guidance"`.

### Task 12: Verify and close Delivery 2

**Files:**
- Modify: `backlog/tasks/task-19635 - Add-nested-AGENTS.md-activation-before-Console-tools.md`

- [ ] **Step 1: Run `python -m pytest Tests/Agents/test_tool_catalog.py Tests/Agents/test_tool_catalog_owner_cache.py Tests/Agents/test_project_instruction_path_targets.py Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_resolver_properties.py Tests/Agents/test_project_instruction_runtime.py Tests/Agents/test_project_instruction_concurrency.py Tests/Agents/test_agent_runtime_review_hook.py Tests/Agents/test_agent_runtime_preparation.py Tests/Agents/test_agent_service.py Tests/Agents/test_agent_service_review_state_scope.py Tests/Tools/test_patch_tool_impls.py Tests/Chat/test_console_agent_project_instructions.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_bridge_local.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_google_native_tools.py Tests/Chat/test_console_project_instruction_provider_grammar.py Tests/Chat/test_console_project_instruction_persistence_boundary.py -q`.** Expected: pass.
- [ ] **Step 2: Run property and concurrency tests repeatedly.** If `pytest-repeat` is installed, run `pytest Tests/Agents/test_project_instruction_concurrency.py Tests/Agents/test_project_instruction_runtime.py -q -x --count=20`. Otherwise run `for i in {1..20}; do pytest Tests/Agents/test_project_instruction_concurrency.py Tests/Agents/test_project_instruction_runtime.py -q -x || break; done`. Expected: twenty clean runs with no flakes.
- [ ] **Step 3: Repeat Task 7 Step 2's exact lint/format policy, appending Delivery-2-created files `tldw_chatbook/Agents/project_instruction_runtime.py Tests/Agents/test_project_instruction_path_targets.py Tests/Agents/test_project_instruction_runtime.py Tests/Agents/test_project_instruction_concurrency.py Tests/Agents/test_agent_runtime_preparation.py Tests/Chat/test_console_project_instruction_provider_grammar.py Tests/Chat/test_console_project_instruction_persistence_boundary.py` to `CHATBOOK_AGENTS_NEW_PY`, and appending modified existing files `tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Tools/patch_tool_impls.py tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_tool_catalog.py Tests/Agents/test_tool_catalog_owner_cache.py Tests/Tools/test_patch_tool_impls.py Tests/Agents/test_agent_runtime_review_hook.py Tests/Agents/test_agent_service_review_state_scope.py Tests/Chat/test_console_agent_bridge_local.py` to `CHATBOOK_AGENTS_EXISTING_PY`. Run the same no-new-diagnostic assertion and `git diff --check 5047b6962...HEAD`. Set `CHATBOOK_AGENTS_QA_DIR=/tmp/chatbook-agents-md-delivery2`, direct the sentinel tests' pytest capture/SQLite dumps/run logs there, then run `rg -n 'CHATBOOK_AGENTS_SENTINEL_7d1e9c' /tmp/chatbook-agents-md-delivery2`.** Expected: new files pass Ruff check/format, existing files add no lint diagnostic, and only the allowlisted fake provider-request spy contains the body.
- [ ] **Step 4: Review scope.** No complete UX/docs/performance/UAT work belongs in this delivery beyond metadata needed for correctness.
- [ ] **Step 5: Complete TASK-19635 AC/notes/status and re-read the file.** Link ADR-069 and both delivery commits.
- [ ] **Step 6: Commit closeout.** `git commit -m "docs: close TASK-19635 nested project context"`.

## Delivery 3 — Interoperability and rollout (`TASK-19636`)

### Task 13: Complete UX states and documentation

**Files:**
- Modify: `backlog/tasks/task-19636 - Verify-and-roll-out-Console-AGENTS.md-support.md`
- Modify: `tldw_chatbook/Widgets/Console/console_project_instructions.py`
- Modify: `tldw_chatbook/Widgets/Console/console_context_modal.py`
- Modify: `tldw_chatbook/Chat/console_display_state.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_project_instructions.py`
- Modify: `Tests/UI/test_console_context_modal.py`
- Modify: `Tests/UI/test_console_right_rail.py`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/console/context-and-rag.md`
- Modify: `Docs/User_Guide/console/sessions-tabs-workspaces.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Recheck base/in-flight work, mark TASK-19636 In Progress, and add its task Implementation Plan.** Re-read with `--plain`.
- [ ] **Step 2: Extend UI tests for every final state.** Cover Off, Choose folder, None, loaded count, Warning, removed/retargeted recovery, override precedence, scope/outcome rows, and a nested activation event. Verify modal focus, Escape/cancel behavior, implemented key hints, and untrusted markup handling at 80x24/100x30/140x40.
- [ ] **Step 3: Implement the tested final-state mapping only.** `Off` exposes Enable; `Choose folder` exposes the chooser; `None` reports an eligible selected binding with no effective source; `<N> loaded` opens source/scope/outcome metadata; `Warning` opens deduplicated content-free codes and recovery actions. Aggregate identical warning category/source pairs once per run, keep the rail row to one line at 80 columns, and expose bodies only through the explicit nonpersistent exact Next Send view. Do not add an editor or another settings surface.
- [ ] **Step 4: Update user docs.** Explain discovery, precedence, selected binding/cwd, no global files, lazy nested scope, untrusted status, first-use destination consent, read-only behavior, budgets/config, warnings, legacy defaults, and explicit-read persistence.
- [ ] **Step 5: Document ecosystem differences accurately.** Codex supplies override/standard hierarchy and broad-to-specific composition; Claude Code supplies the lazy path-sensitive inspiration but uses `CLAUDE.md`, not native `AGENTS.md`. Chatbook deliberately uses binding authority and ephemeral user context.
- [ ] **Step 6: Update `AGENTS.md` Special Systems.** Add a concise project-instruction entry pointing to ADR-069/spec and revise `[console] workspace_root` guidance so selected project-instruction bindings take precedence only for enabled sessions.
- [ ] **Step 7: Run `python -m pytest Tests/UI/test_console_project_instructions.py Tests/UI/test_console_context_modal.py Tests/UI/test_console_right_rail.py -q`, `python -m ruff check tldw_chatbook/Widgets/Console/console_project_instructions.py tldw_chatbook/Widgets/Console/console_context_modal.py tldw_chatbook/Chat/console_display_state.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_project_instructions.py Tests/UI/test_console_context_modal.py Tests/UI/test_console_right_rail.py`, and `git diff --check`.** Expected: pass and clean Markdown whitespace.
- [ ] **Step 8: Commit.** `git commit -m "docs(console): complete AGENTS.md rollout UX and guidance"`.

### Task 14: Record performance, provider UAT, full verification, and closeout

**Files:**
- Create: `Tests/Agents/test_project_instruction_performance.py`
- Create: `Docs/superpowers/qa/agents-md-support-2026-08/README.md`
- Modify: `backlog/tasks/task-19636 - Verify-and-roll-out-Console-AGENTS.md-support.md`
- Modify if a real reusable incident occurs: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Add deterministic performance tests.** Instrument directory visits in a deep synthetic tree. Startup must inspect one directory (O(1)); first nested activation must visit only root-to-target depth (O(depth)); no recursive walk. Record timings as evidence, but assert operation counts rather than fragile wall-clock thresholds.
- [ ] **Step 2: Run `python -m pytest Tests/Agents Tests/Chat Tests/DB Tests/Chatbooks Tests/Workspaces Tests/UI Tests/Tools Tests/test_config_console_defaults.py -q`.** Expected: all affected subsystem suites pass; record exact count/duration in the QA README.
- [ ] **Step 3: Run the broader and static gates exactly.** Run `python -m pytest -q`. Repeat Task 12 Step 3's no-new-diagnostic/new-files-only formatter policy, adding `Tests/Agents/test_project_instruction_performance.py` to `CHATBOOK_AGENTS_NEW_PY`; do not run `ruff format` on existing seams. Then run `python -m mypy tldw_chatbook/Agents/project_instruction_resolver.py tldw_chatbook/Agents/project_instruction_runtime.py tldw_chatbook/Chat/console_project_instructions.py`; `python -m bandit -q -r tldw_chatbook/Agents/project_instruction_resolver.py tldw_chatbook/Agents/project_instruction_runtime.py tldw_chatbook/Chat/console_project_instructions.py`; `python -m pip check`; `python -c 'import pathlib,tomllib; d=tomllib.loads(pathlib.Path("pyproject.toml").read_text()); assert d["project"]["license"] == "AGPL-3.0-or-later"'`; and `git diff --check 5047b6962...HEAD`. Record exact counts and the one verified `RunLogWriter` Ruff baseline separately; do not call any added diagnostic or test failure green.
- [ ] **Step 4: Prepare an isolated live profile.** Set `TLDW_TEST_MODE=1`, `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TLDW_CONFIG_PATH`, and `[paths].data_dir` to a scratch directory before importing or launching Chatbook. Never point a schema-changing branch at the real data directory.
- [ ] **Step 5: Run native cloud-provider UAT.** With a user-supplied credential in environment only: consent, root rider, nested tool deferral, reconsidered tool success, warning/recovery, and no saved body in the scratch DB/logs. Exercise multimodal input if supported.
- [ ] **Step 6: Run fenced/local-model UAT.** Repeat root + nested activation and successful retry against a local/fenced transport. Verify the tool-results fence closes before the labeled context section.
- [ ] **Step 7: Inspect the user-visible TUI.** Capture 80x24/100x30/140x40 evidence for compact row, chooser/notice, Context metadata, warning/recovery, and activation event. Confirm top-to-bottom reading order and actual actions, not just rendering.
- [ ] **Step 8: Perform final sentinel audit.** Search scratch database tables, AgentRunsDB, run logs, Textual logs, captured requests, and exported Context JSON. The sentinel may exist only in explicit next-send inspection and the actual provider request spy; explicit user/tool/model echoes are documented exceptions.
- [ ] **Step 9: Complete documentation/task hygiene.** Check every TASK-19636 AC, add Implementation Notes with exact evidence and ADR-069, decide whether a real incident merits a lessons entry, and set Done via CLI. Audit TASK-19634/16322/16323 statuses from the board.
- [ ] **Step 10: Commit closeout.** `git commit -m "docs: close Console AGENTS.md rollout"`.

## Final acceptance matrix

Before calling the feature complete, verify these cross-delivery invariants explicitly:

- Automatic instruction text is user-level ephemeral context, never system policy.
- One selected eligible binding is both authority root and working directory; binding ID retargets are detected by locator fingerprint.
- Override precedence, empty fallback, invalid no-fallback, bounded stable reads, and symlink/reparse refusal match the approved spec.
- Startup cost is O(1); nested discovery is O(depth); no recursive scan exists.
- Root and nested budgets are separate 32 KiB defaults and whole-source token admission respects the actual provider/model/tool payload.
- Existing permission/change review is unchanged and runs only after successful preparation.
- Deferred batches preserve every tool call's protocol identity and cannot partially execute.
- Registry ownership is first-wins and shared by preflight/dispatch; shadowed providers are never inspected.
- Parent/subagent ledgers share admission but track delivery per chain without retry loops.
- Disabled/legacy sessions retain current local-provider behavior; read-only selected bindings cannot advertise write/edit/patch.
- The local-only JSON column never generates sync/version churn and is preserved by real mutation/import paths.
- First-use consent is scoped to binding locator + provider destination, not model, and occurs even with no root file.
- UI/log/persistence surfaces contain metadata and content-free codes only; explicit reads and model quotations keep normal behavior.
