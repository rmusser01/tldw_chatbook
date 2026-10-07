# Console file authority and project instructions

This document describes what files a Console agent run may touch: the private per-chat scratch space, Named Workspaces with explicit folder bindings, the admitted-roots contract, the AGENTS.md / AGENTS.override.md project-instruction system with its activation ledger, and workspace assistant defaults (personas + permission profiles). Governed by ADR-069 and ADR-079.

## Authoritative files

| File | Role |
| --- | --- |
| `Chat/console_scratch_space.py` | `ConsoleScratchSpaceManager`, immutable `ConsoleScratchSnapshot` (root + token + `(dev, inode)` identity), `lease()`, tombstone/dispose |
| `Chat/console_chat_controller.py` | `capture_run_admitted_workspace_roots()` — builds the immutable `RunAdmittedWorkspaceRoot` tuple for a run |
| `Workspaces/registry_service.py` | `LocalWorkspaceRegistryService` — workspace CRUD, folder bindings, assistant defaults provisioning |
| `Workspaces/models.py` | `WorkspaceRecord`, `WorkspaceRuntimeBinding`, `WorkspaceAssistantDefaults` |
| `Workspaces/assistant_defaults.py` | `resolve_effective_assistant_default()`, posture preview |
| `Tools/workspace_file_roots.py` | ContextVars for run bindings; `run_workspace()`, `allowed_file_roots()`, `frozen_workspace_roots()`, `workspace_context_note()` |
| `Chat/console_project_instructions.py` | `ProjectInstructionControlState`, domain-separated fingerprints, strict JSON, persistence column |
| `Agents/project_instruction_resolver.py` | `ProjectInstructionResolver.resolve_startup()` / `resolve_targets()` — pinned discovery + lazy nested activation |
| `Agents/project_instruction_runtime.py` | `InstructionActivationLedger` — shared budget, per-chain delivery, `prepare()` |
| `Agents/persona_policy.py` | Persona policy parsing, `evaluate_tool_policy()`, `persona_floor_state()` |
| `Agents/run_tool_policy.py` | `RunToolPolicy.check()` — run-id-keyed call caps |

Governance: `backlog/decisions/069-console-project-instruction-local-state-and-preflight.md`, `079-workspace-assistant-defaults.md`; spec `Docs/superpowers/specs/2026-08-20-agents-md-support-design.md`.

## File authority model

Every live Console Chat gets **private temporary scratch** (ADR-082). Named Workspaces may add **explicit folder bindings**. Local `fs_*`/git tools use scratch unless project instructions explicitly select one binding. `[console] workspace_root` is compatibility-only outside this Console path and **never grants a Console Chat access** (it still configures the standalone MCP server and legacy non-Console callers).

### Scratch space

Per-**tab**, not per-conversation: each open Console tab gets its own opaque root with a `(dev, inode)` identity snapshot. A `lease()` context manager keeps a generation alive across a complete filesystem access; closing the tab cleans up best-effort; hard-crash residue is never re-attached by a later process. Scratch also hosts tool-result spill (see [tool-catalog.md](./tool-catalog.md)).

### Admitted roots

At run start, `capture_run_admitted_workspace_roots` freezes the run's authority as an immutable tuple of `RunAdmittedWorkspaceRoot` records — workspace id, binding id/alias, root path, locator fingerprint, root identity (name/dev/ino/mode), `allow_write`, and a guard closure that re-checks registry membership, locator, identity, and access on use. The Default/global workspace admits `()` — path tools are simply not advertised. With a project-instruction binding selected, that binding is the run's **only** admitted root. Frozen roots are revalidated at use: existing, non-symlink, non-drifted, identity-matching.

### Read-only bindings

`allow_write=False` (or no writable root) filters every mutating spec — `fs_write`, `fs_edit`, `fs_patch`, todo tools are not advertised at all ("read-only binding omits patching entirely").

## Project instructions (AGENTS.md / AGENTS.override.md)

Project instructions are **untrusted, ephemeral user context** bounded by one selected local-filesystem binding. They never grant tool permission; guidance delivery is separate from and runs before the unchanged security review.

### Control state and consent

`ProjectInstructionControlState` persists locally (the `conversations.console_project_context_json` column — local-only, absent from every sync trigger and payload). It stores the enabled flag, the selected binding id, a **locator fingerprint** (canonical path identity — the registry may retarget a locator under an existing id, so the id alone is insufficient), and a **notice key** = locator fingerprint × provider-destination fingerprint (credential-free endpoint identity). A provider/endpoint change therefore requires renewed consent; a model-only change does not. A retargeted/missing/unauthorized binding demands explicit re-selection.

### Discovery and lazy nested activation

`resolve_startup` pins the binding root canonically (no-follow ancestor identity capture) and reads `AGENTS.override.md` first, falling back to `AGENTS.md` only when the override is absent or oversized (an oversized override also suppresses the same-directory fallback). `resolve_targets` implements lazy nested activation: when a tool call targets a path inside the binding, the resolver walks root→target chains, `lstat`-ing every component (directories, non-symlink, non-reparse), resolving broad-to-specific scopes under a byte budget with a staleness cutoff, and snapshotting promotions with pre/post read plus ancestor re-verification.

### Activation ledger

`InstructionActivationLedger` is the single shared registry-ownership path feeding tool review:

- ONE lock-owned budget shared by **all** dispatch model chains (parent and sub-agents), with a monotonic `activation_revision`.
- Per-chain state tracks delivered sources and exact payload/token state; `initial_context_for_chain` stages everything active-but-unseen for a chain, so a spawned child gets its own delivery.
- `prepare(calls, chain_id, registry, payload_state)` maps each call through `ToolCatalogRegistry.resolve_owner_for_name` — the same first-registrant-wins owner mapping dispatch uses — then asks only `PathAwareToolProvider.path_targets(tool_id, args)` for validated scopes. Provider mapping exceptions mark the call `outside` with no content.
- When anything is newly required, the runtime appends protocol-safe deferral rows plus ephemeral rows and re-asks the model **without** reviewing or dispatching; the runtime (not the hook) synthesizes the deferral results, so call ids, ordering, and cardinality have one owner. `mark_payload_sent` advances a chain only for the exact issued receipt — forged or stale rows raise.
- Instruction bodies never enter tool results, steps, or logs; ephemeral rows carry an origin key and are stripped from final messages. Preparation failures emit a content-free UI warning and proceed to the unchanged security review.

Built-in `read_file`/`list_directory`/`write_file` participate via `BuiltinToolProvider.path_targets` (exact/directory kinds, `outside` when beyond the instruction root); `fs_glob`/`fs_grep` use only the binding root; opaque process/skill-script tools get startup guidance only.

### Lesson promotion

A managed-skill promotion path lets the agent propose an instruction/lesson edit through the local tools: foreground-primary-only, tool-call-id-keyed `approve_once` stamp consumed exactly once, snapshot revalidation (binding/activation/target/chain change → stale refusal), CAS write via `fs_write`'s `expected_sha256`/`expected_absent`, max 8 proposals per run. Lesson text remains untrusted evidence and grants no filesystem, Notes, or managed-skill authority by itself.

## Workspace assistant defaults (ADR-079)

Explicit workspaces carry reference-backed `assistant_defaults` (persona + permission profile); Default/global stays unset. Resolution mirrors the server (`resolve_effective_assistant_default`, with degraded reasons like `persona_deleted`, `persona_feature_disabled`). Persona policy rules **narrow only**:

- deny-by-default advertising activates only when the tool kind has at least one allow rule; bounded `prefix*` wildcards;
- `require_confirmation` floors allow→ask (origin `persona_policy`), never widens;
- `max_calls_per_turn` takes the minimum across matched rules, enforced per run id by `RunToolPolicy` so concurrent children keep independent budgets;
- profiles inherit unset keys from `default`; every existing gate/floor applies first.

The fixed narrowing order: config gates → ephemeral restrictions → ADR-069 binding access → kill switch → profile grant resolution → persona policy floor → call caps. Read-only bindings still strip mutating fs tools even when persona rules would allow them.

## Dataflow: discovery → ledger → review

1. The session enables project instructions and selects one workspace folder binding; control state persists with fingerprints.
2. `resolve_startup` pins the root instruction file into the ledger's startup source; the first-use notice shows before the first provider request.
3. Per model turn, `LoopDeps.prepare_tool_calls` fires with the complete tool batch; the ledger resolves owners, pulls `path_targets`, lazily activates nested chains under the shared budget, and returns retry-with-context plus ephemeral rows and a delivery receipt when anything new is needed.
4. The runtime re-asks the model with the delivered guidance; once nothing is new, the batch proceeds to the unchanged `review_tool_calls` permission gate and normal dispatch.

## Verified gotchas

1. A registry locator can be retargeted under an existing binding id — consent must fingerprint the locator, not trust the id.
2. An oversized `AGENTS.override.md` suppresses the same-directory `AGENTS.md` fallback too.
3. `workspace_context_note` renders roots relative to the launch cwd (never `../..` chains) and JSON-escapes workspace names against prompt injection.
4. The ledger's owner mapping must be the registry's cached mapping — a parallel lookup could disagree with dispatch about who owns a tool.
5. Persona `require_confirmation` floors can only narrow; a persona can never lift a permission-store ask or deny.
