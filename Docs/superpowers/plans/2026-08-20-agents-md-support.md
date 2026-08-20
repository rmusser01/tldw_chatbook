# Console AGENTS.md Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe, Codex-compatible `AGENTS.md` project guidance to Console agent runs, with root context first, lazy nested activation second, and interoperability/rollout evidence third.

**Architecture:** One selected workspace folder binding is both the v1 working directory and instruction authority root. Root guidance is resolved into an ephemeral, untrusted provider-context rider; a later delivery adds registry-owned path mapping plus an atomic tool-batch preparation hook for nested guidance. Only versioned control metadata persists locally, while automatically loaded instruction bodies remain run-local and never enter transcripts, agent logs, or database rows.

**Tech Stack:** Python 3.11+, Textual 8, SQLite/FTS5 migrations, dataclasses and structural protocols, existing Console provider gateway/agent runtime/tool catalog, pytest + pytest-asyncio + Hypothesis, Backlog.md.

---

## Source of truth and execution constraints

- Approved design: `Docs/superpowers/specs/2026-08-20-agents-md-support-design.md`
- Accepted decision: `backlog/decisions/069-console-project-instruction-local-state-and-preflight.md`
- Delivery tasks: `TASK-16320`, `TASK-16322` (depends on 16320), and `TASK-16323` (depends on 16322).
- ADR required: **yes**.
- ADR path: `backlog/decisions/069-console-project-instruction-local-state-and-preflight.md`.
- Reason: this changes provider/runtime trust boundaries, cross-module tool contracts, and local-only durable session state.
- Planning worktree: `.worktrees/agents-md-support`, branch `codex/agents-md-support`.
- Baseline evidence on 2026-08-20: the following command passed 71 tests (two warnings) across tool ownership, review ordering, local-provider composition, schema-local state, workspace bindings, and Context UI:

```bash
python -m pytest Tests/Agents/test_tool_catalog_owner_cache.py Tests/Agents/test_agent_runtime_review_hook.py Tests/Chat/test_console_local_review_hook.py Tests/DB/test_chachanotes_context_summary_migration.py Tests/Workspaces/test_workspace_folder_bindings.py Tests/UI/test_console_context_modal.py -q
```
- Do not launch Chatbook against the real profile during the schema delivery. Use `tmp_path`/in-memory databases until the schema-changing branch is integrated everywhere that shares the real data directory.
- Before each delivery, repeat the open-PR/branch collision check from `backlog/docs/lessons-backlog-hygiene.md` and rebase onto the intended integration branch. If the ChaChaNotes schema head is no longer v32, allocate the next migration from the actual head and update every v32/v33 path below consistently.
- Do not start TASK-16322 until TASK-16320 is integrated or rebased as its exact base; do not start TASK-16323 until TASK-16322 is integrated or rebased as its exact base.

## File responsibility map

| File | Responsibility |
| --- | --- |
| `tldw_chatbook/Chat/console_project_instructions.py` | Versioned session control state, binding/destination fingerprints, eligible-binding validation, notice state, and immutable provider rider rows. No filesystem reads or UI-state derivation. |
| `tldw_chatbook/Agents/project_instruction_resolver.py` | Secure bounded file discovery/read, root and root-to-target resolution, precedence, budgets, pinning, outcomes, and rendering. No Textual or database imports. |
| `tldw_chatbook/Agents/project_instruction_runtime.py` | Delivery-2 run-local activation ledger, per-chain cursors, target-union preparation, and ephemeral warning/context rows. |
| `tldw_chatbook/Agents/tool_catalog.py` | Structural `PathAwareToolProvider` contract and the single cached first-wins owner lookup shared by preflight and dispatch. |
| `tldw_chatbook/Agents/local_tool_provider.py` | Selected-root capability filtering and structural path-target mapping for all supported `fs_*`/`git_*` tools. |
| `tldw_chatbook/Tools/patch_tool_impls.py` | Reusable bounded parse entry point used by both `fs_patch` execution and preflight; one grammar only. |
| `tldw_chatbook/Agents/agent_runtime.py` | Typed preparation result, preparation-before-review ordering, canonical deferral stubs, and code-only warning callback. |
| `tldw_chatbook/Agents/agent_service.py` | Thread preparation/warning hooks and shared activation state into parent/subagent loop dependencies. |
| `tldw_chatbook/Chat/console_agent_bridge.py` | Build one run ledger, attach startup/nested ephemeral rows, and preserve canonical provider grammar without persisting automatic bodies. |
| `tldw_chatbook/Chat/console_provider_gateway.py` | Strip internal origin tags at the final wire boundary while serializing context separately from tool results for all transport families. |
| `tldw_chatbook/Chat/console_chat_store.py` | Own per-session control state and its temporary-to-durable/write-through lifecycle. |
| `tldw_chatbook/Chat/chat_persistence_service.py` | Narrow project-context persistence adapter methods over `CharactersRAGDB`. |
| `tldw_chatbook/DB/ChaChaNotes_DB.py` + migration SQL | Local-only JSON column and version-neutral accessors; exclude it from sync behavior. |
| `tldw_chatbook/Chat/console_chat_controller.py` | Dispatch-time binding revalidation, first-use consent, startup admission, selected-root provider composition, and UI callbacks. |
| `tldw_chatbook/UI/Console_Modules/session.py` | Explicit screen-state serialization/restoration of the four control fields. |
| `tldw_chatbook/Chat/console_chat_models.py` | Extend `ConsoleContextSnapshot` with metadata-only project-instruction state. |
| `tldw_chatbook/Chat/console_display_state.py` | Pure rail/Context display-state derivation. |
| `tldw_chatbook/Widgets/Console/console_project_instructions.py` | Compact rail row plus choose-folder/first-use/recovery modal views. |
| `tldw_chatbook/UI/Console_Modules/right_rail.py` | Mount the compact row above staged Sources and route its action to the existing Context surface. |
| `tldw_chatbook/Widgets/Console/console_context_modal.py` | Render the metadata-only Project Instructions section and explicit next-send payload view. |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Wire controller callbacks, modal results, snapshot construction, and mounted-state refresh. |

Keep these boundaries. In particular, do not put Textual calls into the resolver/runtime modules, do not put instruction bodies into `ConsoleChatSession`, and do not introduce a second tool-argument parser.

## Delivery 1 — Startup project context (`TASK-16320`)

### Task 1: Claim the delivery and pin the current schema/base

**Files:**
- Modify: `backlog/tasks/task-16320 - Add-startup-AGENTS.md-project-context-to-Console.md`
- Read: `backlog/docs/lessons-testing-evidence.md`
- Read: `backlog/docs/lessons-live-verification.md`
- Read: `backlog/docs/lessons-backlog-hygiene.md`

- [ ] **Step 1: Recheck in-flight work and the branch base.** Run `git branch -a --format='%(refname:short)' | rg -i 'agents-md|project-instruction'` and `gh api -X GET /search/issues -f q='repo:rmusser01/tldw_chatbook is:pr is:open AGENTS.md'`. Expected: no competing implementation; otherwise stop and reconcile.
- [ ] **Step 2: Confirm the schema head.** Run `rg -n '_CURRENT_SCHEMA_VERSION' tldw_chatbook/DB/ChaChaNotes_DB.py`. Expected on this plan's base: `32`. If different, rename and renumber the planned migration/tests before coding.
- [ ] **Step 3: Put the task in progress.** Run `backlog task edit 16320 -s "In Progress"` and add an Implementation Plan that links this file, TASK-16320, the accepted spec, and ADR-069. Re-read with `backlog task 16320 --plain`.
- [ ] **Step 4: Record the scoped baseline.** Re-run the six-file 71-test command from the plan preamble. Expected: all pass before feature edits.
- [ ] **Step 5: Commit task metadata only.** Commit message: `docs: start TASK-16320 AGENTS.md startup context`.

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
- [ ] **Step 6: Run focused tests.** Expected: pass.
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
    relative_path: str
    scope: str
    body: str | None
    byte_count: int
    outcome: str  # loaded / omitted_byte_budget / invalid_utf8 / ...

class ProjectInstructionResolver:
    def resolve_startup(self, binding_root: Path, *, max_bytes: int,
                        dispatch_started_ns: int) -> tuple[InstructionSource, ...]: ...
```

Use standard-library descriptor reads; read at most `max_bytes + 1`; compare file and every ancestor identity before/after. Never log `body` or raw exception text.
- [ ] **Step 6: Add token admission as an injected function.** The resolver owns raw bytes; a pure `admit_sources(sources, safe_input_tokens, count_tokens)` helper owns whole-source token admission. Reuse `Utils.token_counter.count_tokens_messages`/`get_model_token_limit` at the bridge boundary rather than importing provider state here.
- [ ] **Step 7: Run focused tests and `ruff check` on the new module/tests.** Expected: pass.
- [ ] **Step 8: Commit.** `git commit -m "feat(agents): resolve AGENTS.md guidance safely"`.

### Task 4: Add local-only conversation persistence and migration

**Files:**
- Create: `tldw_chatbook/DB/migrations/chachanotes_v32_to_v33_console_project_context.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:165,4684-4742,4836-4910,7890-7965`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py:19-380`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:74-315,315-470,519-690,2532-2790`
- Create: `Tests/DB/test_chachanotes_console_project_context_migration.py`
- Create: `Tests/Chat/test_console_chat_store_project_instructions.py`
- Modify: `Tests/Chatbooks/test_chatbook_importer.py`

- [ ] **Step 1: Seed failing migration tests from a real v32 database.** Assert v33, nullable `console_project_context_json`, idempotent version guard, v32-to-v33 upgrade preservation, and fresh database availability. Use only `tmp_path`/in-memory DBs.
- [ ] **Step 2: Write failing local-write tests.** A set/clear round trip must not change conversation `version`/`last_modified`, create `sync_log`, or alter synchronized payloads. Ordinary `update_conversation`, soft delete, restore, and restart must preserve the value.
- [ ] **Step 3: Write import conflict tests.** `SKIP` must leave an existing local value untouched. Current non-skip paths, including `REPLACE`, must create a separate row with null project context and leave the existing row unchanged.
- [ ] **Step 4: Run the three focused files.** Expected: failures for the missing column/accessors.
- [ ] **Step 5: Implement the additive migration and explicit accessors.** The SQL is only `ALTER TABLE ... ADD COLUMN ...` plus a guarded schema-version update. Add `get_conversation_console_project_context()` and `set_conversation_console_project_context()` as bare, parameterized queries; the setter must not touch watched columns.
- [ ] **Step 6: Extend the persistence protocol/service and session lifecycle.** Add one `project_instruction_state` field to `ConsoleChatSession` whose dataclass default factory is `ProjectInstructionControlState.legacy_disabled`. Only `ConsoleChatStore.create_session()` explicitly supplies `new_session()`; every direct construction and every restore therefore fails closed unless new-session creation opted in. `restore_persisted_session()` decodes the database value and defaults legacy-disabled. Temporary sessions keep state in memory; `promote_ephemeral_session()` flushes it only after the durable conversation exists and includes it in rollback behavior.
- [ ] **Step 7: Run focused DB/store/import tests.** Expected: pass, including no sync/version churn.
- [ ] **Step 8: Commit.** `git commit -m "feat(db): persist Console project context locally"`.

### Task 5: Assemble startup rider, binding consent, and selected-root tools

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:3244-3375,6380-6515,7163-7395`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py:744-908,1412-2050`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py:1499-1615,1747-1835`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py:93-220,680-940`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Create: `Tests/Chat/test_console_agent_project_instructions.py`
- Modify: `Tests/Chat/test_console_local_review_hook.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `Tests/Chat/test_console_rewind_summarize.py`

- [ ] **Step 1: Write dispatch gating tests.** New agent sessions with one eligible binding auto-select it; zero/multiple bindings hold the send for recovery; removed/unauthorized/missing/retargeted bindings never silently retarget. Direct/plain and character-forced-plain sends never discover or transmit project instructions.
- [ ] **Step 2: Write consent-key tests at the controller seam.** Consent occurs before any provider request even when no root file exists. Proceed stores the destination-scoped notice key; cancel aborts the send; disable turns the feature off. Provider/custom endpoint changes re-prompt; model-only changes do not.
- [ ] **Step 3: Write startup transport/leak tests with a sentinel body.** Text, multimodal, retry, regenerate, and continue agent sends include the labeled rider exactly once. Parent and newly spawned subagent model chains each receive the same immutable root snapshot exactly once on their initial provider context, without Delivery-2 nested ledger/cursor machinery. The sentinel is absent from store messages, AgentRunsDB steps, run log, exception/log capture, `/rewind` input, and any automatic tool result. An explicit `fs_read` or assistant quotation remains normally persisted.
- [ ] **Step 4: Write selected-root/read-only tests.** Enabled sessions compose `LocalToolProvider` at the validated binding root; disabled/legacy sessions retain `[console] workspace_root`/cwd behavior. Read-only bindings omit `fs_write`, `fs_edit`, and `fs_patch` from the catalog while retaining read/git tools and instruction loading.
- [ ] **Step 5: Run focused tests and observe failures.** Expected: missing startup snapshot, rider, consent, and selected-root behavior.
- [ ] **Step 6: Capture the immutable startup candidate at the controller boundary.** After provider resolution and before `_run_agent_reply` performs a provider call, re-resolve the binding, compare its locator fingerprint, create/verify the destination notice key, securely resolve the byte-bounded root candidate, and compose the selected-root provider. Pass that immutable candidate into `ConsoleAgentBridge.run_reply()`; never reread the root body inside the bridge or gateway.
- [ ] **Step 7: Perform exact token admission where the run registry exists.** In `ConsoleAgentBridge.run_reply()`, after `_compose_run_registry_and_allowed()` has produced the collision-resolved tool schemas, calculate safe input headroom from the actual provider/model limit, compacted history, response reserve, and exact run schemas. Admit the candidate as a whole source, render it as a tagged user-level row with a conspicuous wrapper such as `[Project instructions — untrusted repository context]`, and append it only to the run-local message copy. Thread the same admitted root snapshot into each new parent/subagent model chain once. Do not rebuild the registry in the controller.
- [ ] **Step 8: Strip only internal transport metadata.** Attach `EPHEMERAL_ORIGIN_KEY` internally. In `_chat_api_kwargs()` and llama.cpp builders, copy each row and strip only internal metadata immediately before wire serialization; preserve the content. Mark sensitive logging for any call carrying the internal tag. There is no current durable exchange-capture subsystem to modify; the origin tag defines the omission boundary any such subsystem must honor later.
- [ ] **Step 9: Keep `/rewind` ordering intact.** Pin with a regression test that `_apply_context_summary_compaction()` runs before the startup row is appended. Do not add compaction logic.
- [ ] **Step 10: Filter write specs when composing the selected-root local provider.** Add one explicit `allow_write` constructor option or a small filtered-spec helper; do not mutate the global default spec table.
- [ ] **Step 11: Run focused controller/bridge/gateway/rewind/local-provider suites.** Expected: pass.
- [ ] **Step 12: Commit.** `git commit -m "feat(console): send startup AGENTS.md context"`.

### Task 6: Add basic rail, Context, chooser, and first-use UI

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py:529-545`
- Modify: `tldw_chatbook/Chat/console_display_state.py`
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
- [ ] **Step 4: Write rail/Context integration tests.** The row mounts above Sources, opens the existing Context modal, and refreshes in place. `ConsoleContextSnapshot` carries metadata plus an explicit next-send payload that may show the rider body only when the user opens that payload view.
- [ ] **Step 5: Run focused UI tests and observe failures.** Expected: missing widget/state.
- [ ] **Step 6: Implement the smallest widget surface.** One feature module may contain the compact row and its two modal modes; do not create a file browser/editor. Keep untrusted labels `markup=False`.
- [ ] **Step 7: Wire async controller callbacks from `ChatScreen`.** The worker/controller requests a decision; only the Textual main loop mounts/dismisses modals. Background/parked sessions must scope the decision to their own session ID, not whichever session is visible.
- [ ] **Step 8: Run focused UI/session tests.** Expected: pass at 80x24, 100x30, and 140x40 pilot sizes.
- [ ] **Step 9: Commit.** `git commit -m "feat(console): show project instruction status"`.

### Task 7: Verify and close Delivery 1

**Files:**
- Modify: `backlog/tasks/task-16320 - Add-startup-AGENTS.md-project-context-to-Console.md`

- [ ] **Step 1: Run focused suites.** Run all new Delivery-1 files plus affected existing suites under `Tests/Agents`, `Tests/Chat`, `Tests/DB`, `Tests/Chatbooks`, `Tests/Workspaces`, and the named Console UI files. Expected: all pass.
- [ ] **Step 2: Run static checks.** `ruff check` on changed Python files, the repository's formatter check, and `git diff --check`. Expected: clean. Do not bulk-format unrelated code.
- [ ] **Step 3: Run sentinel leakage search.** Use a unique test sentinel and assert it appears only in the resolver's in-memory result and fake provider request spy. Search test logs/artifacts and database dumps; expected: no other occurrence.
- [ ] **Step 4: Review the diff against TASK-16320 only.** Delivery 1 must not contain `prepare_tool_calls`, nested path mapping, or subagent ledger code.
- [ ] **Step 5: Complete the Backlog task.** Check every AC, add concise Implementation Notes including ADR-069 and exact verification, then `backlog task edit 16320 -s Done`. Re-read the task after the CLI mutation.
- [ ] **Step 6: Commit closeout.** `git commit -m "docs: close TASK-16320 startup project context"`.

## Delivery 2 — Nested path activation (`TASK-16322`)

### Task 8: Add registry-owned path-target mapping

**Files:**
- Modify: `backlog/tasks/task-16322 - Add-nested-AGENTS.md-activation-before-Console-tools.md`
- Modify: `tldw_chatbook/Agents/tool_catalog.py:343-350,821-1015`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `tldw_chatbook/Tools/patch_tool_impls.py:105-178,370-450`
- Modify: `Tests/Agents/test_tool_catalog.py`
- Modify: `Tests/Agents/test_tool_catalog_owner_cache.py`
- Create: `Tests/Agents/test_project_instruction_path_targets.py`
- Modify: `Tests/Tools/test_patch_tool_impls.py`

- [ ] **Step 1: Recheck base/in-flight work, mark TASK-16322 In Progress, and add its task Implementation Plan.** Re-read with `--plain`.
- [ ] **Step 2: Write first-wins owner tests.** Register colliding builtin/local/skill/MCP fakes in different orders. `resolve_owner_for_name(name)` must return the exact `(tool_id, provider)` used by `invoke_by_name`; preflight must never call shadowed providers.
- [ ] **Step 3: Write the complete local mapping matrix.** Exact parent scopes for `fs_read/write/edit`; listed-directory scope for `fs_list`; binding root only for `fs_glob/grep`; parsed create/modify targets for `fs_patch` including `dry_run`; every approved filtered/unfiltered git rule; no targets for web/todo/opaque tools.
- [ ] **Step 4: Write built-in mapping tests.** `read_file`/`write_file` exact parent and `list_directory` directory scope inside the selected binding; another authorized binding returns an outside-instruction-scope target, not a second hierarchy. Disabled built-ins report no targets.
- [ ] **Step 5: Run focused tests and observe failures.** Expected: protocol/owner resolver absent.
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
- [ ] **Step 8: Run mapping/catalog/patch suites.** Expected: pass.
- [ ] **Step 9: Commit.** `git commit -m "feat(agents): map tool calls to instruction scopes"`.

### Task 9: Add shared activation ledger and lazy nested resolution

**Files:**
- Create: `tldw_chatbook/Agents/project_instruction_runtime.py`
- Modify: `tldw_chatbook/Agents/project_instruction_resolver.py`
- Create: `Tests/Agents/test_project_instruction_runtime.py`
- Create: `Tests/Agents/test_project_instruction_concurrency.py`

- [ ] **Step 1: Write ledger tests.** Root snapshot is active at dispatch; nested sources pin once; parent/subagents share source outcomes and byte budget; each chain has its own delivery revision/cursor; a new child begins at the active snapshot revision.
- [ ] **Step 2: Write outcome-loop tests.** Byte/stale/invalid/read failures are global terminal no-content outcomes; token omission is per chain. Each unseen outcome defers/warns that chain exactly once; identical retry proceeds.
- [ ] **Step 3: Write deterministic concurrency tests.** Use barriers, never sleeps. First lock wins the remaining nested budget; deepest-first admission inside a batch is deterministic; later chains receive explicit omissions.
- [ ] **Step 4: Write lazy discovery tests.** Extend `ProjectInstructionResolver` with `resolve_targets(binding_root, targets, *, max_bytes, dispatch_started_ns, pinned)`. Resolution walks only root-to-target chains (O(depth)), applies deterministic deepest-first admission with broad-to-specific rendering, skips created/changed-after-dispatch candidates, retains already pinned content after edit/delete, and never walks sibling subtrees.
- [ ] **Step 5: Run and observe failures.** Expected: runtime module absent.
- [ ] **Step 6: Implement one lock-owned ledger.** The ledger stores sources/outcomes, remaining raw budget, dispatch timestamp, and a revision counter. Per-chain state stores only delivered revision/outcome keys. Neither type has serialization methods.
- [ ] **Step 7: Implement `prepare(calls, chain_id, registry, token_allowance)` as pure orchestration around injected resolver/owner lookups.** Union targets before locking; recheck/admit under the lock; return tagged ephemeral rows or proceed. Outside-binding targets add a content-free warning only.
- [ ] **Step 8: Run runtime/concurrency/resolver suites and ruff.** Expected: pass.
- [ ] **Step 9: Commit.** `git commit -m "feat(agents): track nested project instruction activation"`.

### Task 10: Add preparation-before-review and canonical deferral

**Files:**
- Modify: `tldw_chatbook/Agents/agent_runtime.py:208-275,455-650`
- Modify: `tldw_chatbook/Agents/agent_service.py:268-335,1280-1335`
- Modify: `Tests/Agents/test_agent_runtime_review_hook.py`
- Create: `Tests/Agents/test_agent_runtime_preparation.py`
- Modify: `Tests/Agents/test_agent_service.py`
- Modify: `Tests/Agents/test_agent_service_review_state_scope.py`

- [ ] **Step 1: Write typed-result tests.** Only `proceed` and `retry_with_context` construct successfully; retry requires tagged ephemeral rows; proceed carries none.
- [ ] **Step 2: Write ordering/atomicity tests.** Preparation receives the entire call batch once. Retry creates one fixed tool-result stub per original call, preserves ID/name/order/cardinality, skips review and execution, appends the separate context row, then loops back to the model.
- [ ] **Step 3: Write exception tests.** A preparation exception emits only `project_instruction_preparation_failed` plus tool names/count through `on_ephemeral_runtime_warning`, logs no exception/traceback/body, and proceeds into unchanged `review_tool_calls`. Warning-callback failure is swallowed with code-only logging.
- [ ] **Step 4: Write no-hook and existing-review regression tests.** Absent preparation remains byte-identical; existing review fail-open/fail-closed ownership and verdict strings do not change.
- [ ] **Step 5: Run focused tests and observe failures.** Expected: missing fields/result.
- [ ] **Step 6: Implement frozen result types and LoopDeps hooks.** `AgentRuntime` owns stubs; the hook never returns tool results or review verdicts:

```python
@dataclass(frozen=True, slots=True)
class ToolBatchPreparation:
    status: Literal["proceed", "retry_with_context"]
    ephemeral_rows: tuple[Mapping[str, Any], ...] = ()
```

- [ ] **Step 7: Thread hooks through `AgentService`.** Parent and inline spawned subagents receive the same ledger/preparation owner but distinct stable chain IDs. Extend existing `review_state_scope` tests so the new state cannot be clobbered by an inline child.
- [ ] **Step 8: Run runtime/service suites.** Expected: pass.
- [ ] **Step 9: Commit.** `git commit -m "feat(agents): prepare tool batches before review"`.

### Task 11: Wire nested preparation and provider grammar without persistence leaks

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Create: `Tests/Chat/test_console_project_instruction_provider_grammar.py`
- Create: `Tests/Chat/test_console_project_instruction_persistence_boundary.py`
- Modify: `Tests/Chat/test_console_agent_bridge_local.py`

- [ ] **Step 1: Write end-to-end fake-provider tests.** A multi-call batch targeting a new nested scope defers before approval/execution, sends stubs then ephemeral context, and allows the reconsidered call through normal approval. Non-path-aware/opaque calls do not activate nested guidance.
- [ ] **Step 2: Write exact grammar tests for four transport families.** OpenAI/Gemini: all tool responses then separate user context row. Anthropic: one user turn with all `tool_result` blocks before a distinct context text block. Fenced/local: close the complete results fence before a separately labeled context section.
- [ ] **Step 3: Write parent/subagent delivery tests.** Both share admission but each receives unseen revisions before execution. A concurrently activated source cannot execute in another chain until that chain receives it.
- [ ] **Step 4: Write persistence-boundary tests with a sentinel.** Automatic bodies are absent from `AgentStep`, AgentRunsDB, run log, transcript/context event, review verdict, exception, and application log. Source-relative metadata/warning codes may appear. Explicit reads/model quotations remain untouched.
- [ ] **Step 5: Run focused tests and observe failures.** Expected: bridge does not yet construct/wire preparation.
- [ ] **Step 6: Build one dispatch ledger in `ConsoleAgentBridge.run_reply()`.** Seed it from the immutable startup snapshot, capture the per-run registry's exact owner cache, and pass preparation/warning callbacks into `AgentService`. Never serialize the ledger.
- [ ] **Step 7: Add transport serialization at the gateway boundary.** Canonical runtime order stays transport-independent; provider-specific grouping happens only in one helper adjacent to `_chat_api_kwargs()`. Internal origin tags survive until serialization/capture omission, then are stripped.
- [ ] **Step 8: Post metadata-only activation events.** Use relative sources/scopes and outcome codes. Do not route them through `on_step` or transcript messages.
- [ ] **Step 9: Run bridge/gateway/grammar/persistence suites.** Expected: pass.
- [ ] **Step 10: Commit.** `git commit -m "feat(console): activate nested AGENTS.md guidance"`.

### Task 12: Verify and close Delivery 2

**Files:**
- Modify: `backlog/tasks/task-16322 - Add-nested-AGENTS.md-activation-before-Console-tools.md`

- [ ] **Step 1: Run all Delivery-2 focused suites plus Delivery-1 regression files.** Expected: pass.
- [ ] **Step 2: Run property and concurrency tests repeatedly.** If `pytest-repeat` is installed, run `pytest Tests/Agents/test_project_instruction_concurrency.py Tests/Agents/test_project_instruction_runtime.py -q -x --count=20`. Otherwise run `for i in {1..20}; do pytest Tests/Agents/test_project_instruction_concurrency.py Tests/Agents/test_project_instruction_runtime.py -q -x || break; done`. Expected: twenty clean runs with no flakes.
- [ ] **Step 3: Run static/diff/security checks.** Include a sentinel search across pytest capture, temporary DB exports, and run logs. Expected: no automatic body leakage.
- [ ] **Step 4: Review scope.** No complete UX/docs/performance/UAT work belongs in this delivery beyond metadata needed for correctness.
- [ ] **Step 5: Complete TASK-16322 AC/notes/status and re-read the file.** Link ADR-069 and both delivery commits.
- [ ] **Step 6: Commit closeout.** `git commit -m "docs: close TASK-16322 nested project context"`.

## Delivery 3 — Interoperability and rollout (`TASK-16323`)

### Task 13: Complete UX states and documentation

**Files:**
- Modify: `backlog/tasks/task-16323 - Verify-and-roll-out-Console-AGENTS.md-support.md`
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

- [ ] **Step 1: Recheck base/in-flight work, mark TASK-16323 In Progress, and add its task Implementation Plan.** Re-read with `--plain`.
- [ ] **Step 2: Extend UI tests for every final state.** Cover Off, Choose folder, None, loaded count, Warning, removed/retargeted recovery, override precedence, scope/outcome rows, and a nested activation event. Verify modal focus, Escape/cancel behavior, implemented key hints, and untrusted markup handling at 80x24/100x30/140x40.
- [ ] **Step 3: Implement only missing polish.** Aggregate warnings by category/source once per run; keep the row compact; show bodies only in the explicit exact next-send payload view. Do not add a new editor or settings duplicate.
- [ ] **Step 4: Update user docs.** Explain discovery, precedence, selected binding/cwd, no global files, lazy nested scope, untrusted status, first-use destination consent, read-only behavior, budgets/config, warnings, legacy defaults, and explicit-read persistence.
- [ ] **Step 5: Document ecosystem differences accurately.** Codex supplies override/standard hierarchy and broad-to-specific composition; Claude Code supplies the lazy path-sensitive inspiration but uses `CLAUDE.md`, not native `AGENTS.md`. Chatbook deliberately uses binding authority and ephemeral user context.
- [ ] **Step 6: Update `AGENTS.md` Special Systems.** Add a concise project-instruction entry pointing to ADR-069/spec and revise `[console] workspace_root` guidance so selected project-instruction bindings take precedence only for enabled sessions.
- [ ] **Step 7: Run focused UI and docs checks.** Expected: pass and `git diff --check` clean.
- [ ] **Step 8: Commit.** `git commit -m "docs(console): complete AGENTS.md rollout UX and guidance"`.

### Task 14: Record performance, provider UAT, full verification, and closeout

**Files:**
- Create: `Tests/Agents/test_project_instruction_performance.py`
- Create: `Docs/superpowers/qa/agents-md-support-2026-08/README.md`
- Modify: `backlog/tasks/task-16323 - Verify-and-roll-out-Console-AGENTS.md-support.md`
- Modify if a real reusable incident occurs: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Add deterministic performance tests.** Instrument directory visits in a deep synthetic tree. Startup must inspect one directory (O(1)); first nested activation must visit only root-to-target depth (O(depth)); no recursive walk. Record timings as evidence, but assert operation counts rather than fragile wall-clock thresholds.
- [ ] **Step 2: Run the complete focused suite.** Include every new project-instruction test and affected agent/controller/provider/database/workspace/UI/Chatbook-import suite. Expected: pass.
- [ ] **Step 3: Run broader regression and static gates.** Run the repository's full unit/integration suite, `ruff check`, formatter check, license/security checks, and `git diff --check`. Record exact counts and any established unrelated baseline separately.
- [ ] **Step 4: Prepare an isolated live profile.** Set `TLDW_TEST_MODE=1`, `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TLDW_CONFIG_PATH`, and `[paths].data_dir` to a scratch directory before importing or launching Chatbook. Never point a schema-changing branch at the real data directory.
- [ ] **Step 5: Run native cloud-provider UAT.** With a user-supplied credential in environment only: consent, root rider, nested tool deferral, reconsidered tool success, warning/recovery, and no saved body in the scratch DB/logs. Exercise multimodal input if supported.
- [ ] **Step 6: Run fenced/local-model UAT.** Repeat root + nested activation and successful retry against a local/fenced transport. Verify the tool-results fence closes before the labeled context section.
- [ ] **Step 7: Inspect the user-visible TUI.** Capture 80x24/100x30/140x40 evidence for compact row, chooser/notice, Context metadata, warning/recovery, and activation event. Confirm top-to-bottom reading order and actual actions, not just rendering.
- [ ] **Step 8: Perform final sentinel audit.** Search scratch database tables, AgentRunsDB, run logs, Textual logs, captured requests, and exported Context JSON. The sentinel may exist only in explicit next-send inspection and the actual provider request spy; explicit user/tool/model echoes are documented exceptions.
- [ ] **Step 9: Complete documentation/task hygiene.** Check every TASK-16323 AC, add Implementation Notes with exact evidence and ADR-069, decide whether a real incident merits a lessons entry, and set Done via CLI. Audit TASK-16320/16322/16323 statuses from the board.
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
