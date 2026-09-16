# Managed Plugins Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the complete approved managed-plugin and marketplace system, including every shared-hook addition and qualified Cursor/Codex interoperability.

**Architecture:** Four subsystem plans share existing Console, Skills, MCP, trust and permission owners. Begin with a usable native local-skill path; qualify hooks and direct MCP before exposing richer package adapters and the browser. Each task has its own acceptance evidence and can be reviewed before the next dependent task begins.

**Tech Stack:** Python >=3.12, Textual 8.2.8, Pydantic >=2.4,<3, SQLite, httpx, portalocker 3.2.0 and existing host crypto/credential services.

**Spec:** [Plugin spec](../specs/2026-09-15-managed-plugins-design.md) and [hook spec](../specs/2026-09-15-expanded-hook-runtime-design.md).

## Global Constraints

- Python >=3.12; current checkout pins Textual 8.2.8, Pydantic >=2.4,<3 and portalocker 3.2.0. Preserve these pins; use the existing SQLite/httpx/crypto/keyring seams.
- One installation and selected revision exist per user-data directory.
- Global default activation starts disabled. Importing a catalog does not install or enable its entries.
- All approved hook additions are in scope. Delivery stages are ordering, not deferral.
- No parallel agent/permission runtime, package build/install execution, vendor grants or per-workspace package versions.
- Required constraints never disappear because parsing, configuration, hooks or persistence fail. Native/foreign instructions remain attributed untrusted context.
- Package activation grants no filesystem binding, tool permission, network credential or trusted project status. Console local tools retain scratch/explicit-binding authority.
- Apply current authority before injection/launch/dispatch/result acceptance. Workspace disable preserves other authorized scopes; namespace marker changes alone do not cancel all work.
- Full-suite runs require explicit user opt-in. Every task runs its exact feature/regression files and a successful control on the same production entry.
- Implementation uses an isolated execution worktree and profile; do not repoint a shared editable environment. Verify child interpreter package provenance as well as pytest cwd.
- Every To Do task must move In Progress and receive its Implementation Plan via Backlog CLI before code changes. Add Implementation Notes and mark Done only after its acceptance criteria, review, targeted tests and static checks pass.

ADR required: yes
ADR paths: [ADR-162](../../../backlog/decisions/162-managed-agent-plugins.md); [ADR-163](../../../backlog/decisions/163-expanded-console-hook-runtime.md)
Reason: Existing accepted ADRs govern this implementation; no additional architecture decision is introduced by task decomposition.

---

## Plans and milestones

| Plan | Deliverable | Execution order |
| --- | --- | --- |
| [Native package foundation](2026-09-15-plugin-foundation.md) | Local install/trust/enable/native-skill use/disable; update, recovery and owned data cleanup. | F1-F8 |
| [Shared hooks](2026-09-15-expanded-hooks.md) | Standalone v2 command hooks and every new lifecycle/effect; MCP handlers after direct transport. | H1-H5, then H6 after M4 |
| [Direct MCP](2026-09-15-plugin-mcp.md) | Typed results, three protocol profiles, credential authority and scoped owned tools. | M1-M4 |
| [Adapters, marketplaces and UI](2026-09-15-plugin-marketplaces-and-ui.md) | All native capabilities, qualified vendor imports/catalogs, Git acquisition and Plugins UI. | I1-I7 |

Recommended serial order is the table below. Tasks on separate branches may run
independently only when their declared dependencies and file ownership permit it;
parallel task execution is an execution choice, not part of this planning run.
All implementation tasks remain **To Do**. This commit implements no runtime feature.

## Atomic Backlog tasks

| Key | Task | Prerequisites |
| --- | --- | --- |
| F1 | [TASK-32668](../../../backlog/tasks/task-32668%20-%20Inspect-immutable-native-plugin-packages.md) — Inspect immutable native plugin packages | Approved design |
| F2 | [TASK-32669](../../../backlog/tasks/task-32669%20-%20Store-plugin-registry-state-under-one-runtime-owner.md) — Store plugin registry state under one runtime owner | F1 |
| F3 | [TASK-32670](../../../backlog/tasks/task-32670%20-%20Authenticate-complete-plugin-authority-snapshots.md) — Authenticate complete plugin authority snapshots | F2 |
| F4 | [TASK-32671](../../../backlog/tasks/task-32671%20-%20Commit-and-recover-reviewed-plugin-installation-changes.md) — Commit and recover reviewed plugin installation changes | F3 |
| F5 | [TASK-32672](../../../backlog/tasks/task-32672%20-%20Admit-native-plugin-skills-through-existing-Console-authority.md) — Admit native plugin skills through existing Console authority | F4 |
| F6 | [TASK-32673](../../../backlog/tasks/task-32673%20-%20Stop-and-revoke-plugin-work-within-the-requested-scope.md) — Stop and revoke plugin work within the requested scope | F5 |
| F7 | [TASK-32674](../../../backlog/tasks/task-32674%20-%20Drain-plugin-work-before-applying-updates-and-rollback.md) — Drain plugin work before applying updates and rollback | F6 |
| F8 | [TASK-32675](../../../backlog/tasks/task-32675%20-%20Delete-plugin-data-only-after-exact-root-users-drain.md) — Delete plugin data only after exact-root users drain | F6, F7 |
| H1 | [TASK-32676](../../../backlog/tasks/task-32676%20-%20Validate-explicit-v2-hook-definitions-and-effects.md) — Validate explicit v2 hook definitions and effects | Approved design |
| H2 | [TASK-32677](../../../backlog/tasks/task-32677%20-%20Execute-v2-command-hooks-with-bounded-resource-ownership.md) — Execute v2 command hooks with bounded resource ownership | H1 |
| H3 | [TASK-32678](../../../backlog/tasks/task-32678%20-%20Integrate-hook-input-transformations-and-post-event-barriers.md) — Integrate hook input transformations and post-event barriers | H2 |
| H4 | [TASK-32679](../../../backlog/tasks/task-32679%20-%20Wire-session-child-and-compaction-hook-boundaries.md) — Wire session child and compaction hook boundaries | H3 |
| H5 | [TASK-32680](../../../backlog/tasks/task-32680%20-%20Schedule-bounded-Stop-continuations-and-teardown.md) — Schedule bounded Stop continuations and teardown | H4 |
| M1 | [TASK-32681](../../../backlog/tasks/task-32681%20-%20Preserve-typed-MCP-tool-results-through-client-services.md) — Preserve typed MCP tool results through client services | Approved design |
| M2 | [TASK-32682](../../../backlog/tasks/task-32682%20-%20Add-qualified-direct-Streamable-HTTP-MCP-transport.md) — Add qualified direct Streamable HTTP MCP transport | M1 |
| M3 | [TASK-32683](../../../backlog/tasks/task-32683%20-%20Bind-MCP-credentials-to-stable-reviewed-authority.md) — Bind MCP credentials to stable reviewed authority | M2, F3 |
| M4 | [TASK-32684](../../../backlog/tasks/task-32684%20-%20Expose-owned-plugin-MCP-tools-with-scoped-connection-leases.md) — Expose owned plugin MCP tools with scoped connection leases | M3, F5, F6, F8, H3 |
| H6 | [TASK-32685](../../../backlog/tasks/task-32685%20-%20Invoke-MCP-backed-hooks-through-normal-tool-authority.md) — Invoke MCP-backed hooks through normal tool authority | H5, M4 |
| I1 | [TASK-32686](../../../backlog/tasks/task-32686%20-%20Register-all-native-plugin-capability-types.md) — Register all native plugin capability types | F5, H6, M4 |
| I2 | [TASK-32687](../../../backlog/tasks/task-32687%20-%20Qualify-Cursor-and-Codex-package-and-hook-adapters.md) — Qualify Cursor and Codex package and hook adapters | I1 |
| I3 | [TASK-32688](../../../backlog/tasks/task-32688%20-%20Acquire-bounded-Git-and-local-plugin-snapshots.md) — Acquire bounded Git and local plugin snapshots | F1, F7 |
| I4 | [TASK-32689](../../../backlog/tasks/task-32689%20-%20Browse-Git-catalogs-and-import-selected-marketplace-sources.md) — Browse Git catalogs and import selected marketplace sources | I2, I3 |
| I5 | [TASK-32693](../../../backlog/tasks/task-32693%20-%20Add-Plugins-navigation-and-source-qualified-browsing.md) — Add Plugins navigation and source-qualified browsing | I4 |
| I6 | [TASK-32694](../../../backlog/tasks/task-32694%20-%20Review-and-manage-plugin-operations-across-UI-surfaces.md) — Review and manage plugin operations across UI surfaces | I5, F8 |
| I7 | [TASK-32695](../../../backlog/tasks/task-32695%20-%20Qualify-plugin-lifecycle-interoperability-and-authoring-examples.md) — Qualify plugin lifecycle interoperability and authoring examples | I6 |

## Verified existing seams

The plan was grounded against the current checkout, not an assumed plugin API:

- `Skills_Interop/skill_trust_service.py` and `skill_trust_store.py` own passphrase trust, encrypted snapshots and marker posture. Existing standalone marker identity is a pair; plugin authority adds its own tuple/namespace.
- `DB/private_sqlite.py:connect_private_sqlite` requires a registered owner; `Tests/DB/test_private_sqlite_inventory.py` must admit the new plugin store.
- `Agents/tool_catalog.py:ToolProvider` and `ToolCatalogRegistry.register_provider` are the shared tool seams; plugin packages do not get a second catalog.
- `Agents/run_hooks.py:RunHooksEngine` has separate sync/async/notification/close entries. `agent_runtime.py`, `agent_service.py` and `console_agent_bridge.py` own guard/post-tool entry, including durable and preauthorized paths.
- `Chat/console_runtime.py:ensure_run_hooks`, `console_chat_controller.py`, `console_context_compaction.py` and prompt-queue/interrupt owners are the actual admission/settlement boundaries.
- `MCP/client.py:MCPClient.call_tool` currently drops structured/error fields into content-only presentation. M1 introduces typed results before H6 relies on them.
- `MCP/local_store.py:LocalExternalMCPProfile` currently stores command/args/env; `LocalMCPControlService.connect_profile` launches through stdio. M2 adds the explicit generic HTTP profile/transport, not a tldw_server wrapper.
- No generic MCP OAuth binding service was established by this inspection. M3 qualifies existing host auth support and reports unsupported authentication explicitly; this plan does not promise vendor-account access.
- `UI/Navigation/screen_registry.py`, `main_navigation.py`, `UI/stable_command_palette.py` and canonical `settings_screen.py` own destination discovery. Legacy settings screens are excluded.

## Spec coverage and review amendments

| Contract | Owning tasks |
| --- | --- |
| Plugin sections 1-3: scope, portable core, closed native extension, dialect selection | F1, I1, I2 |
| Sections 4.1-4.2: immutable identity, private storage, one runtime owner | F1, F2 |
| Section 4.3: complete authenticated authority, trust posture and credential continuity | F3, F4, M3 |
| Sections 5.1/5.3: all component types, context authority, partial support and evidence axes | F5, I1, I2, I5, I7 |
| Section 5.2: direct transport, typed results, credentials and scoped tools | M1-M4, H6 |
| Section 6: safe Git/local acquisition, catalogs, refresh and selected app imports | I3, I4 |
| Section 7: named/default workspace activation and immutable run revisions | F5-F7, M4 |
| Section 8.1-8.2: exact review and all durable recovery states | F3, F4 |
| Section 8.3: update drain, current-policy rollback and retention | F7 |
| Section 8.4: immediate scoped stop versus persistence, shared requests, data drain | F6, F8, M4, H5, H6 |
| Section 8.5: surviving children and honest process/data reuse | F2, F6, F8, H2 |
| Section 9: token UI, review identity, navigation and recovery | I5, I6 |
| Section 10: bounded acquisition/cache/context/receipt behavior | F1, F7, H2, I3, I4 |
| Sections 11-13: qualification and every approved review amendment | Each task's matrix; integrated I7 |
| Hook definitions/results/matchers/required-context contracts | H1, H2, H6 |
| SessionStart/End and UserPromptSubmit | H4, H5 |
| PreToolUse/final guards and PostToolUse/Failure pending checkpoints | H3 |
| SubagentStart/Stop and PreCompact/PostCompact | H4 |
| Stop continuations and Interrupt | H5 |
| Provisional MCP initialization, normal authority and causal cycles | M4, H4, H6 |
| Application/runtime budgets, fair queue, notification/reap separation | H2, H5 |
| Vendor event/argv/regex/cwd/output qualification | I2, H6 |
| Lost constraints / explicit authority coverage / recovery proof | F1, F3, F4, I2 |
| Pending post-event race / final-argument guarantees / error-first MCP results | H3, M1, H6 |
| Credential renewal / scope-specific cancellation / data-writer drain | M3, F6, M4, F8 |

No hook additions are dropped or silently moved outside the approved release.
Unsupported vendor semantics remain visible per the design; they are not a claim
that the source feature is implemented. Cross-platform/original-host comparison
claims need their own actual evidence rather than a plan or parser test.

## Verification and task closeout

- [ ] Before each task, read its Backlog file and both specs, start it via CLI and copy its scoped checklist into its Implementation Plan. Do not add implementation notes to untouched To Do tasks.
- [ ] Verify the selected Python environment imports this exact checkout, including isolated child subprocesses; install dependencies in an isolated verification environment if missing. Do not rewrite the shared environment's editable binding.
- [ ] Run the RED/GREEN commands listed in the subsystem task, then its exact neighboring regression files. Record actual executed counts, skips, platform and production entry; do not infer behavior from a source-string assertion.
- [ ] Run syntax/format/static checks on changed Python files and whitespace on the exact diff. The repository currently declares no formatter/linter entry point; use Black and Ruff only in the isolated verification environment, without adding runtime dependencies or reformatting unrelated files. Apply formatting before the final tests if it changes source.
- [ ] For UI, read the design constitution, use existing tokens, rebuild CSS and run the listed route/token/bundle guards. Verify real keyboard/async behavior at 80x24 and 120x35 using an isolated app profile.
- [ ] Record each task's evidence and remaining unsupported conditions; satisfy ACs before marking Done. Commit only its reviewed files. A full test suite remains opt-in.

Concrete local static checks (substitute only the task's exact changed Python paths):

```bash
python -m compileall -q tldw_chatbook/Plugins
python -m ruff check --select E9,F63,F7,F82 tldw_chatbook/Plugins
python -m black --check tldw_chatbook/Plugins
git diff --check
```

For hooks/MCP/UI tasks, use the exact new/changed files from their file lists
instead of the Plugins directory. Do not autoformat the large legacy controllers
wholesale: isolate the new modules and keep legacy integration edits minimal.
The local static tools are verification aids; their absence is not a reason to
claim a check passed. No full-suite run is authorized by this plan.

## Planning provenance and handoff

The user accepted the amended written design by asking to continue. ADR-162 and
ADR-163 now record accepted implementation direction; implementation still needs
the evidence above. The design task closes when the plans/task graph and document
checks pass; the 25 implementation tasks stay open.

Task IDs were allocated through Backlog CLI after fresh origin refs plus a scan
of reachable task paths and all available worktrees. The CLI's lower local offer
was renumbered only for this new uncommitted batch; no existing task was renamed.
Allocation is a snapshot, so recheck before merge and distinguish the current
session's Codex turn snapshots from competing task owners.
The final scan found a competing TASK-32690 in the workflows authoring worktree.
This batch moved I5-I7 to TASK-32693 through TASK-32695, preserved backward-only
dependencies and recorded the original numbers in those task files.

Execution may use a fresh subagent per task with review gates or inline task-by-task
execution. Both start with F1, preserve dependency order and use isolated worktrees
at execution time. No extra feature-design approval is needed for these accepted
contracts; material departures require updating the affected specification/ADR.
