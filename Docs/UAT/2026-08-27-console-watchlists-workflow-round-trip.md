# Console Watchlists workflow round-trip UAT

Date: 2026-08-29
Task: TASK-22868
Status: rebased onto latest `origin/dev`, Qodo findings addressed, targeted
gates green, and independent pre-PR round-three review approved

## Outcome

The disposable-profile UAT exercises a new-user Console request through real
Watchlists tools and durable local services. It creates three local RSS sources,
groups them into one Watchlist, follows each check receipt, generates and follows
one briefing receipt, saves an every-24-hours schedule, lists and opens the
completed briefing, and proves that the agent can consume information that
exists only in that briefing.

The run uses a temporary profile, temporary SQLite databases, and scripted local
feed/model fixtures. It does not read or modify a live user profile, contact the
public internet, install ATHF, create a hunt document, or test a
briefing-to-hunt handoff.

## Evidence boundaries

The evidence deliberately separates three claims:

1. **Service round trip.** `Tests/QA/test_console_watchlists_workflow_uat.py`
   drives the public `ConsoleAgentBridge.run_reply` seam, production local-tool
   provider, durable Watchlists services, operation coordinator, briefing
   persistence, external-MCP publication boundary, and skill/framework fixtures.
2. **Mounted composition and navigation.** `Tests/UI/test_console_watchlists_mounted_uat.py`
   mounts the real app with `ConsoleChatController`, real app-owned provider
   composition, visible approval controls, durable receipt following, and public
   navigation into Watchlists, Settings, and Library. Its model and feed executor
   are deterministic local fixtures; no socket is opened. The test exports actual
   mounted Console states at 180x50 and 160x42.
3. **Seeded rendering fixtures.** `capture_uat.py` mounts real Textual screens with
   deterministic pre-seeded state for focused responsive/HCI review. Its six
   Console, Watchlists, and Library SVGs are layout fixtures, not proof that a
   tool loop ran.

## Durable results

- Sources: `local:subscription:1`, `local:subscription:2`,
  `local:subscription:3`
- Collection: `local:watchlist:1`
- Source-check receipts: `local:watchlist_run:1`,
  `local:watchlist_run:2`, `local:watchlist_run:3`
- Briefing receipt: `local:briefing:1`
- Recurrence: `86,400` seconds, with scheduler reload acknowledged
- Post-commit honesty: an unavailable persisted provider/model route cannot
  turn a confirmed schedule write into a false storage failure; the receipt
  remains `ok`, requests reload, marks `briefing_route_ready: false`, and points
  to Settings before any future model egress
- Briefing: complete, with ordered selected-item and cited-item provenance
- Cross-surface projections: Watchlists membership and Settings schedule state
  agree with the durable rows
- Agent consumption: the final answer contains the deterministic briefing-only
  marker, while external MCP serialization does not
- Permission audit: the explicitly allowed local Watchlists tool set equals the
  Watchlists tools invoked by the run

“Existing model” never means the model in the active Console conversation. At
run time, manual and recurring no-preset briefings resolve one persisted pair:
the collection/preset provider and model first; otherwise persisted
`chat_defaults.provider` and `.model`; otherwise the configured model saved for
that same persisted provider. An unavailable pair fails closed before egress.
The accepted receipt, provider call, and durable `model_used` provenance all use
the same resolved pair.

## External MCP privacy proof

External MCP publishes only the bounded source, collection, briefing-list, and
operation-receipt tools. Console-only mutation, source checking, briefing
generation, scheduling, item/body retrieval, search, and full briefing retrieval
are absent from discovery. Direct dispatch of full briefing retrieval is
refused. Serialized discovery, receipt results, and permission state contain
neither the briefing-only marker nor the fixture article body. The warmed SQLite
file, schema, and rows are unchanged by external calls.

## Skill and framework regression

Local fixtures cover a root skill, a multi-skill repository, and a generic
framework repository. Classification remains `root_skill`, two ordered
candidates, and `framework_repository`. Import remains untrusted until explicit
review (`trust_approved=False`), a second submit is refused while an import owns
the single-flight coordinator, and the reported result reflects the completed
import rather than the refused submit. No remote repository is cloned or
installed. The three changed Library skill/import files pass 204 targeted tests.

## First Run regression status

First Run remains a regression prerequisite, not new implementation. On the
reconciled latest-dev tree, the three plan-target files passed 138 sandbox-safe
tests. Their only two sandbox failures were disposable `127.0.0.1` peer binds;
both exact nodes passed with loopback permission. No full repository sweep was
run.

## HCI review

For a first-time user, the Console path is clearest when the agent previews what
it will create, asks only for consequential approvals, names returned receipts,
and ends with the source count, Watchlist name, next eligibility, and a direct
way to open the briefing. The first mention of “existing model” needs the exact
persisted-default explanation above; “current model” is a reasonable but unsafe
first-time interpretation.

For a power user, canonical IDs, exact terminal receipt states, deterministic
cadence, provider/model provenance, and an auditable permission list are the
useful density. Repeated approval prose and generic success messages become
noise; terse tool state with expandable detail scales better.

The mounted UAT found an interaction defect that service tests could not expose:
resume-state synchronization repeatedly called `set_batch` with an identical
approval round, remounting Select controls and eventually producing duplicate
IDs. The shared approval-card boundary now treats identical
`(round_id, phase, calls)` updates as idempotent, while changed calls, phase, or
round still render normally. A focused regression preserves the mounted row,
Select, and button identity across repeated syncs without weakening approval or
loop-detection semantics.

The six seeded fixtures retain the earlier responsive craft review for Console,
Watchlists, and Library at 180x50 and 160x42. The two separately named
`mounted-console-*` SVGs come from the real mounted composition/navigation UAT
and contain the briefing-only rendered assertion; they are not seeded by
`capture_uat.py`.

## Reproducibility and branch state

- Worktree: `.worktrees/uat-threat-intel`
- Pre-task HEAD: `a43ddfee49d81cdd7d7f082b54c0e83307523598`
- Review-fix base HEAD: `e9fe184a05ec901e691a4dd592dcbf6f4b31a1eb`
- Reconciled code HEAD tested: `25be18705ec897596e61c0cebfe20814157b6530`
- Current observed `origin/dev`: `b1ada0fba2cafe4aee34441926ee96e036ccef55`
- Current merge base: `b1ada0fba2cafe4aee34441926ee96e036ccef55`
- Reconciliation: complete for the recorded targeted gates; no push or merge was
  performed before the PR was opened
- Independent review: approved at
  `2274046883ac513aca0c3960504b945cbdef1110`; no findings remain in scope
- PR review: all eleven actionable Qodo findings are addressed. In addition to bounded
  briefing following and serialized setting writes, the subscriptions
  migration is packaged and transaction-owned, readiness tests use real
  SQLite variants, both user-controlled evidence paths are centrally confined,
  Library policy denial is fixed/non-retryable, and the seeded capture sandbox
  is removed on both render and cleanup failure. Follow-up corrections validate
  dynamic test identifiers, context-own the in-memory SQLite connection from
  acquisition, and align every evidence surface on this tested revision.

Machine-readable evidence, exact bounded commands, capture hashes, and the
redacted transcript live in
`Docs/superpowers/qa/console-watchlists-workflow-2026-08/`.
