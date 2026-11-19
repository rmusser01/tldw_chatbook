# Task8 — pinned dev integration and final qualification

## Frozen source

Final source commit: `40b9aacaa21f7e8c6f2b1989efbb8e8021a1fe77`. Worktree/index were clean at freeze. Root checkpoint was `edaeececfb3733885bb5fec3ea7b5823266dff23`; the earlier checkpoint `8bd3c9f5e529efc298bd6ccd63e0265878b07cf2` and both verified recovery bundles/refs remain available.

One fetch exposed dev `5e0341d1ec701865e019eb2fd8a5e2028ab2d474` beyond scoped `a1571e86…`. Integration stopped until root recorded expanded scope. One rebase replayed all 42 commits from integrated `df2ba424…` onto that pin, producing union `bc184be782`. No second fetch, push, merge, PR message, dependency install, broad suite, cap increase, new skip, suppression, or hook bypass occurred. The dirty shared checkout was untouched.

[Freeze map](task-8-final-freeze-map.json) records full SHAs, parent chains, 42 old/new commit pairs, all final blobs, corrective functions/assertions, and exact Task7 carry. [Range diff](task-8-rebase-range-diff.log) and the conflict receipts distinguish rebase reconciliation from the separate corrective commit.

## Reconciliation and behavior

The three incoming production collisions were `ConsoleChatController.__init__`, `request_chat_create_confirm`, and `execute_agent_chat_create`. Other incoming methods and nonoverlap production blobs remain exact. Wizard extraction, liveness/worker rules, first-run recovery, module caps and six census owner moves are retained. The only changed incoming nonoverlap files are the two explicitly approved mounted test fixtures.

The existing plain lock now has one named locked validator and a public wrapper that acquires it once. Final decisions and remembered grants validate exact preparation under the existing lock; token retirement stays outside it. No lock spans SQLite I/O.

Prepared creation checks committed Close before dispatch, source liveness, and exact approval immediately before save. A successful approved save retains its conversation and v2 draft. At the owner-loop restore boundary a closed source prevents placement/start and records `source_unavailable` with `draft` or `not_started`. Uncertain metadata writes retain `review_required/outcome_unconfirmed`. Native preacceptance withdrawal and the AgentRuns acceptance cutoff are unchanged. Source Close after acceptance retains target receipt, answer, consumed draft, and charge. Completion resolves the current view callback only when executed and owns no durable artifact. Legacy fork orphan cleanup remains intact.

Raw new-chat fixtures now use actual trusted preparation and exact approval. Their saved-draft differences are explicit in the assertion map. Rulings39–40 add verified primary identity to the grant race and real owned databases to prepared mounted cases. The two cases without a primary turn use actual parent/child AgentRuns rows and a current surviving-child actor, `same_workspace`/`draft`, with no primary cancel/message registrations. Before Close they prove exact record/source liveness; the standalone case also proves its unresolved parked card, request/token identity, and kind. Existing cleanup targets the exact child run ID. All original no-owner, Close, sibling, fleet, and termination assertions remain.

## Qualification

| Receipt | Result |
|---|---|
| [Prepared phases and original native neighbors](task-8-green-phases.json) | 16 passed, including bounded real worker decision/Close and both acceptance sides |
| [Adapted executor integration](task-8-integration.json) | 11 passed; fork parameters and orphan assertions retained |
| [Affected existing owners](task-8-affected.json) | 68 passed / 2 fixture failures retained and resolved below; passing cases not replayed |
| [Verified-primary follow-up](task-8-fixture-followup.json) | Primary grant-race passes; same grouped UI fixture then exposed its third unconfigured persistence caller |
| [Faithful surviving-child mounted group](task-8-surviving-child.json) | 1 public group passed, including its five original scenarios |
| [Final combined startup/navigation](task-8-final-startup-navigation.json) | 58 passed: five wizard AST guards, affected caps, startup/import/CSS owners, real Console→Library→Console navigation, final six saved-draft controls and stronger postacceptance receipt/consumption assertions |
| [Fatal Ruff](task-8-final-fatal.json), [touched formatter ratchet](task-8-final-formatter-ratchet.json), [source whitespace](task-8-source-whitespace.json) | Pass |

Final ChatScreen: **25,199 lines / 759 methods**, within unchanged **25,218 / 759**. Ready census **1033/1033**. Retained final warnings: import weight **681/686** with snapshot drift +25/−13; ready census +5/−2 with zero headroom; preimport payload **557/557**, 415,298/425,347 LOC and library route 127,527/135,111 LOC. Task7's original FD-growth223 warning remains historical evidence, not a claimed fix.

Formatter-only changes after qualification carry through [full-module AST proofs](task-8-final-format-proof.json), chained to [the initial formatting proof](task-8-format-proof.json). The final six phase cases and postacceptance receipt case reran after the last behavioral edit. Earlier integration cases all exercise either normal restore or an exception before `target` assignment; adding the explicit `target is not None` condition confines source-close error reporting to restore and leaves those tested branches unchanged. Lock/decision and native coordinator bodies did not change after their passing controls.

## Preservation and derived outputs

[Final preservation](task-8-final-preservation.json) contains immutable before/after SHA256 and blob maps for every selected test, recursively imported test helper/conftest, production source, and incoming/shared method. All **34 Task7 changed functions** are exact AST matches; its unaffected adoption/hooks/schema/provider qualification carries without replay.

All **11,572 required historical QA blobs**, all **11,750 QA blobs in the root checkpoint**, and all **42 incoming QA additions** are exact. Recovery and earlier receipts remain immutable.

- [Diagnostic reconstruction](task-8-diagnostics-before.json): exact committed rebuild, 650 owners and16 sink files. [Owner/statement union](task-8-diagnostic-union-proof.json) proves the controller's90 feature +2 incoming calls. [Three-way safe-pin proof](task-8-diagnostic-safe-pins-three-way.json) preserves feature removals and incoming pins; no inventory rewrite or waiver.
- [Worker contract](task-8-worker-contract.json):324 DOM lookups /151 functions and68 wait pushes /26 non-worker entry points, no new violations. Freeze map proves exact await/wait/UI owner unions.
- [UI census](task-8-ui-census.json):135 files, unchanged incoming floor133.
- [CSS reproduction](task-8-css-reproduction.json): every generated stylesheet rebuilds byte-for-byte; all source/output hashes recorded.

## Retained failures and limits

The initial phase run failed8 cases because prior rigs fabricated `source-message`; actual Close correctly required a real store row. Corrected phase RED then failed6 intended saved-draft assertions and passed2 native Close controls. The real prepared decision worker failed its bounded5-second join on the naive recursive-lock union; the isolated process exited, with no retained non-daemon worker.

The original grant-race fixture passes1/1 on the immutable incoming controller/test and fails with the feature's existing primary-only remember rule because it had no primary identity. Only its trusted identity/context setup changed. Mounted failures progressed from missing persistence in two callers to the same third caller. The superseded stripped-primary attempt timed out at302.85s; its child retained the original failure and the300-second executor-join warning. It is failed setup evidence, not Close qualification. Ruling40 replaced it with genuinely live surviving-child controls. No test process remains running.

The first formatter ratchet rejected the helper's dedented formatting and adjacent inherited signature; final context-correct formatting passes the unchanged ratchet. Full-diff whitespace exits2 with3,211 findings entirely in preserved historical QA and is byte-for-byte identical to the immutable rebase union output. [Attribution](task-8-whitespace-attribution-final.json) and the passing all-source check retain that distinction. Initial report-helper import and naive safe-pin/whitespace parsing diagnostics also remain; their corrected proofs are separate files.

Self-review covered lock acquisition, exact record fields, Close/save/acceptance phases, current observer lookup, original fork behavior, faithful fixture phase witnesses, and source/QA unions. Root owns independent scoped review, any resulting fixes, fresh remote ancestry, current-head external checks, publication and normal merge. This report does not claim those gates complete.

[Safe evidence manifest](task-8-safe-evidence-manifest.json) lists only explicit JSON/log/XML copies with absolute paths, SHA256 and size. No profiles, configs, database or cache files are copied.
