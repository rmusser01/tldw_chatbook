# Console Watchlists workflow round-trip UAT

Date: 2026-08-29
Task: TASK-22868
Status: automated, visual, redaction, and latest-dev reconciliation checks green

## Outcome

The deterministic UAT exercises a new-user Console request through real Watchlists tools and durable local services. It creates three local RSS sources, groups them into one Watchlist, follows each check receipt, generates and follows one briefing receipt, saves an every-24-hours schedule, lists and opens the completed briefing, and proves that the agent can consume information that exists only in that briefing.

The run uses a temporary profile, temporary SQLite databases, and a loopback-only RSS fixture server. It does not read or modify a live user profile, contact the public internet, install ATHF, create a hunt document, or test a briefing-to-hunt handoff.

## Tested boundary

The automated loop uses the public `ConsoleAgentBridge.run_reply` entry point, `ConsoleChatStore`, the production `LocalToolProvider`, the real Watchlists command/query services, the real operation coordinator, and the real briefing persistence/projection paths. The scripted gateway supplies model planning and deterministic briefing prose only; it does not simulate tool results or database effects.

One limitation remains explicit: this harness does not mount Textual's `ConsoleChatController`. The controller's app-owned provider composition has no public, non-mounted injection seam; calling its private composition helper would weaken the public-seam proof. Existing targeted controller-composition regressions remain part of the verification gate, while the production-shaped Textual captures separately mount the real Console, Watchlists, and Library screens. This report therefore does not claim a literal mounted-controller end-to-end run.

## Durable results

- Sources: `local:subscription:1`, `local:subscription:2`, `local:subscription:3`
- Collection: `local:watchlist:1`
- Source-check receipts: `local:watchlist_run:1`, `local:watchlist_run:2`, `local:watchlist_run:3`
- Briefing receipt: `local:briefing:1`
- Recurrence: `86,400` seconds, with scheduler reload acknowledged
- Briefing: complete, with ordered selected-item and cited-item provenance
- Cross-surface projections: Watchlists membership and Settings scheduled-job projection agree with the durable rows
- Agent consumption: the final answer contains the private briefing-only sentinel; the sentinel itself is intentionally omitted from committed evidence
- Permission audit: the explicitly allowed local Watchlists tool set exactly equals the Watchlists tools invoked by the run
- Source-check concurrency: observed peak stayed within the four-worker cap

The schedule's “existing model” does not mean the model in the currently open Console conversation. It means the persisted briefing model/provider selection for that collection when one exists, falling back to the application's persisted briefing/provider defaults. Recurring runs resolve from those saved settings; changing the chat model alone does not silently change the scheduled briefing model.

## External MCP privacy proof

The external MCP registration exposes only the shared metadata/receipt tools:

- `watchlists_list_sources`
- `watchlists_list_collections`
- `watchlists_list_briefings`
- `watchlists_get_operations_status`
- `watchlists_get_operation_status`

Console-only source mutation, collection mutation, source checking, briefing generation, schedule mutation, item/body retrieval, search, and full briefing retrieval are absent from discovery. Direct dispatch of full briefing retrieval is refused. Serialized discovery, receipt results, and permission state contain neither the private briefing sentinel nor the fixture article body. Warmed read-only file hashes and an exact SQLite schema-and-row dump are identical before and after the external calls.

## Skill and framework regression

Local fixtures cover a root skill, a multi-skill repository, and a generic framework repository. Classification remains `root_skill`, two ordered candidates, and `framework_repository` respectively. Import remains untrusted until explicit review (`trust_approved=False`), a second submit is refused while an import owns the single-flight coordinator, and the reported result reflects the completed import rather than the refused submit. No remote repository is cloned or installed.

## First Run regression status

All 140 selected First Run tests have passing evidence across an explicit environment split. A sandboxed broad run produced 137 passes, deselected the known order-sensitive geometry node, and failed only the two tests that require a temporary `127.0.0.1` listener. Those two loopback nodes pass with local-bind permission, and the geometry node passes in a fresh isolated process. Three representative final-tip checks also pass for fresh-profile offer, persisted provider/model selection, and returning to Console without losing the user's work. Exact commands and node IDs are recorded in `evidence.json`.

## HCI review

For a first-time user, the Console path is strongest when the agent states what it will create, asks only for consequential approvals, names returned receipts, and ends with a compact summary containing source count, Watchlist name, next run, and a direct instruction for opening the briefing. “Existing model” must be expanded the first time it appears because users otherwise reasonably infer the current chat model.

For a power user, canonical IDs, exact receipt states, deterministic cadence, provider/model provenance, and an auditable permission list are the useful density. Repeated approval prose and generic success messages become noise; the Console should preserve terse tool-state disclosure while keeping detail expandable.

Two bounded craft passes inspected production-shaped 180×50 and 160×42 captures. The first pass caught a capture-timing defect that showed `Cadence Off`; the final capture waits for the exact selected briefing and persisted `86,400`-second cadence, and visibly states every-24-hours, next eligibility, last attempt/success, and app-open scope. Receipt cards and generic framework recovery remained readable without clipping at both sizes.

Latest-dev reconciliation exposed one real adjacent interaction defect: a completed skill import refreshed the Library rail by replacing its mounted navigation controls, so a user's already-visible Media control could be detached before the press landed. `LibraryRail.sync_state` now patches stable expanded-shell rows and Details content in place when the structure is unchanged, with mounted regression coverage proving the same Media control survives import completion and remains usable. The reconciliation also aligned the stale Watchlists failure-copy assertion with the canonical safe classifier projection; the durable-operation/briefing/scheduler gate is now 119/119.

The Console/provider/MCP gate is 413 passed with four optional `mcp_unified` skips. Two cancellation tests still emit non-failing owner-thread shutdown warnings from the latest Console prompt-queue code. They do not affect the UAT outcome, but they are retained in the evidence rather than suppressed.

## Reproducibility and branch state

- Worktree label: `.worktrees/uat-threat-intel`
- Original TASK-22868 pre-task HEAD: `a43ddfee49d81cdd7d7f082b54c0e83307523598`
- Refreshed and tested `origin/dev`: `18384c80d1e2ff1a9b5748ac6bba3aea737cf6a5`
- Merge base after reconciliation: `18384c80d1e2ff1a9b5748ac6bba3aea737cf6a5`
- Reconciled tested HEAD before this evidence update: `911a0131194b03c6c14ab61373d626512a8cefad`
- Latest-dev reconciliation: complete; the 50-commit branch was rebased onto the refreshed dev tip and every TASK-22868 gate named above was rerun

Machine-readable evidence and the redacted transcript live in `Docs/superpowers/qa/console-watchlists-workflow-2026-08/`.
