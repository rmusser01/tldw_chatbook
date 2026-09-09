# ADR-138: Portable workflow definitions and local execution

- Status: Accepted by the user on 2026-09-08
- Date: 2026-09-08
- Task: [TASK-32077](../tasks/task-32077%20-%20Design-portable-Workflows-editor-and-local-execution-parity.md)
- Design: [Workflows local execution and portable authoring](../../Docs/superpowers/specs/2026-09-08-workflows-local-first-parity-design.md)
- Revision: the user approved the revised Chatbook contracts and safety defaults on 2026-09-08. The paired-server synchronization protocol still requires counterpart agreement and implementation; this acceptance does not assert server support.
- Supersedes: N/A

## Context

The user approved a Workflows redesign with a workflow library, step navigator, and overview/focused-card canvas. Forms are continuous and collapsible. V1 must execute workflows locally without tldw_server; branching follows in v2 and parallelism in v3. Definitions should be shareable and synchronized across Chatbook and tldw_server where reasonably possible.

The server dev baseline `6cd2745f696af04668a61c20b84ab8a9e69ca5e4` provides an ordered-step API definition, an extensible step catalog, and immutable saved versions. Its web canvas export is a different nodes/edges representation. Workflow visibility is private and workflows are absent from the current Sync v2 domains. Chatbook has reusable local domain services but no corresponding general workflow editor/engine.

## Decision

1. Use the server API workflow definition as the common execution/exchange document. Preserve stable step IDs, config, and metadata. UI projections and local view preferences do not become an alternative execution format.
2. Provide a Chatbook-owned sequential engine and a single run-state service over private workflow persistence. Reuse local services through explicit adapters. Do not embed the server application or create a second implementation of each underlying domain service.
3. Select Local or a configured server for a complete run. Unsupported local capabilities fail preflight; there is no automatic remote fallback. Report local execution separately from external-network requirements.
4. Execute immutable saved snapshots. Draft changes cannot modify running work. Interrupted or uncertain side effects require recovery review rather than automatic replay.
5. Propose a versioned `metadata.tldw_workflow` namespace for global workflow/revision IDs, revision parents, and portable requirements. Bind actual resources and credentials per installation. Adopt this namespace through a paired server/client contract before claiming cross-client sync support.
6. V1 supports reviewed canonical JSON import/export and explicit fetch/publish against existing servers. A separate v1 sync workstream adds an advertised `workflow.definition` domain, durable revision checks, recoverable server version projection, and explicit conflicts. It must not overload an existing notes/chat domain or imply current server support.
7. Sync saved definition revisions only in v1. Local drafts, credentials, grants, active runs, results, and view state retain their local owners. File sharing creates an explicit reusable copy/provenance relationship and does not grant execution authority.
8. Preserve unsupported branching/parallel definitions. Editing/execution capability follows the v1/v2/v3 roadmap, not whether a parser can retain the JSON.
9. Materialize reviewed ordinary defaults/run values and requirement-owned destination bindings into explicit run inputs. Capture non-secret selections, target identity, contract revisions, limits, and launch-operation identity in a private manifest. Resolve credentials only through the bound credential owner or a proven adapter-specific runtime-secret path. The server is not assumed to interpret the proposed metadata namespace.
10. Evaluate the whole definition's effects before admission/export, including completion callbacks and opaque metadata. Non-empty completion hooks block local v1 execution; remote execution requires destination/output disclosure and normal egress authorization. Preservation never grants execution authority or certifies opaque content secret-free.
11. Retain physical attempt ownership until all owned work exits. Separate engine timeout, adapter request timeout, and durable human-response deadline. Repeated launch delivery uses the same operation identity; an uncertain effect or unconfirmed remote launch cannot automatically create a replacement attempt/run. Run budgets persist through retry/restart, and policy increases are explicit recorded amendments.
12. Recover drafts independently of saved revisions, including invalid editor buffers. Navigation waits for durable draft persistence or explicit loss acknowledgment. Results identify their original run/revision/target/attempt even while a different draft is edited. Define keyboard focus transitions for collapsed/hidden regions and typed input references as part of the application structure.
13. Deliver a local file-to-note end-to-end slice first, retain the complete 21-step v1 target, and retain paired-server workflow sync as an explicit v1 milestone. This sequencing does not move branching from v2 or parallelism from v3.

## Alternatives considered

| Alternative | Reason not selected |
| --- | --- |
| Adopt the web editor's nodes/edges export as the common format | It differs from the execution API and currently loses metadata/version information. |
| Require a server for every run | Does not meet the approved local execution requirement. |
| Import the server's complete runtime | Couples the TUI to FastAPI, server databases, and server scheduler dependencies. |
| Reimplement LLM/RAG/media services inside workflows | Duplicates established behavior and creates avoidable parity drift. |
| Offload unsupported steps automatically | Changes execution authority and data movement during a supposedly local run. |
| Sync by name/version or last-write-wins | Names and integer IDs are not portable identity; concurrent edits can be lost. |
| Add workflow content to an existing sync domain | Changes a negotiated contract without corresponding server support. |
| Flatten imported control flow for v1 | Changes the meaning of a shared workflow. |
| Auto-merge concurrent step edits | Array order, references, and prompts can conflict semantically even when fields differ. |
| Save only valid revisions; retain no invalid editor buffer | Ordinary navigation or a restart could lose unfinished authoring work, even though run history is durable. |
| Generate execution controls directly from discovered config schemas | The inspected server distinguishes step-level controls from adapter configuration and human-response deadlines; schema presence alone does not establish runtime semantics. |
| Free capacity as soon as a run is marked timed out/cancelled | An owned tool worker may still be running and producing effects, allowing unsafe overlapping retries. |
| Block the first usable workflow on all baseline adapters and server sync | A smaller end-to-end slice can validate the shared contracts while preserving the eventual v1 scope. |

## Consequences and constraints

- Adapter availability is separate from definition readability and form coverage; a type or schema entry is not a working local adapter.
- Existing servers remain usable for explicit exchange, while sync readiness depends on negotiated support and metadata-preserving clients.
- The new store must register with private-path, backup, migration, and lifecycle ownership under ADR-029 and the current backup design.
- Runtime/source authority remains with ADR-033 owners; no new root AppState is introduced.
- Workflows owns authoring and run summaries; Console retains detailed live activity, consistent with ADR-011.
- Existing tool permission, path, and credential boundaries continue to apply to imported workflows.
- The approved local profile is workflow-runtime scoped, not a machine-wide quota; domain-owned notes/media are not workflow artifact cleanup targets. Remote limits are advertised as enforced only when the server actually supports them.
- Safety/privacy differences from the server, including payload-bearing log handling and unresolved-expression rejection, are named conformance restrictions rather than hidden format changes.
- This ADR records accepted Chatbook architecture, not approval to implement a new server protocol or a claim of completed parity.
- Identifier allocation was checked against local remote refs and worktrees and must be rechecked before integration.

## Related decisions

- [ADR-008: Sync v2 client contract alignment](008-sync-v2-client-m1-contract-alignment.md)
- [ADR-011: Workbench UI system](011-chatbook-workbench-ui-system.md)
- [ADR-029: Local private data boundary](029-local-private-data-boundary.md)
- [ADR-031: TUI keybindings](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-033: Application session state ownership](033-application-session-state-ownership.md)
- [ADR-060: Notes interoperability constraints](060-notes-sync-round-trip-and-interoperability-constraints.md)
- [ADR-068: Local research execution precedent](068-local-research-execution-engine.md)
- [ADR-126: Local backup and recovery](126-complete-local-backup-and-recovery.md)
