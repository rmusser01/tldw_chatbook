# K3 fix round 1 re-review

## Finding verdicts

- **I1 - New scanner units remain admitted after another session revokes or pauses the job: ADDRESSED.** `tldw_chatbook/Workspaces/project_index/service.py:506` now enters the existing registry/store admission fence and compares live authority revision/exclusions, local stop, and the durable job predicate before returning authority for a new unit. `tldw_chatbook/DB/Workspace_Project_Index_DB.py:458` checks exact scope, enabled Building state, control job revision, and active job ID/token/revision/authority revision in one transaction. The registry/store contexts exit before worker I/O. Thus a unit whose admission follows a completed remote Pause/Disable/Clear cannot use the revoked claim; already-admitted bounded work remains correctly distinguished. `Tests/Workspaces/test_project_index_service.py:1095` places the barrier before the second admission, after the first page commits, and exercises all three management actions through a second service with real stores/scanner. It asserts denied authority, no second Directory request, zero indexed files and native retirement. `Tests/DB/test_workspace_project_index_db.py:428` independently rejects all seven tested exact-claim/state mismatches.

- **I2 - A delayed management operation can stop a newer accepted job for the same binding: ADDRESSED.** `tldw_chatbook/Workspaces/project_index/service.py:208` and `:324` hold the existing service lock across the validated transition and capture of the affected run objects. This serializes capture with _start's dispatch interval. Pause selects its exact pre-transition claim; revocation snapshots existing affected runs while its transition is fenced. `service.py:199` now accepts the captured tuple and performs no later key-based lookup. Signalling and joining remain outside transition locks, so a later accepted run cannot become an old operation's cancellation target. `Tests/Workspaces/test_project_index_service.py:1140` delays the stop after the durable transition, physically retires the old writer, starts a replacement, and checks that the replacement is not stopped and completes Ready. Pause, Disable and Clear are covered; Clear's existing lifetime/scope refusal remains intact.

## New breakage in the fix diff

**None identified.** The four-path fix retains registry-before-store ordering and mandatory store activation admission, adds no worker I/O or joins inside the transition fences, and does not change native inheritance, cleanup ownership, runtime identity or the K4/K5 interfaces. The new service lock acquisition follows _start's existing lock order; the worker/supervisor publication paths do not acquire that service lock while holding their registry/store fence.

## Out-of-scope observations

No new observations. Original nonblocking M1 (historical _Run retention) remains deferred, and inherited test/lint noise remains explicitly recorded. No untouched source was re-reviewed.

## Evidence and scope

- Reviewed the re-review template, original task brief, I1/I2 findings and appended fix report. Source interval: `7f5255e2706ac1d6a83803bd4574f78d293cf6a7` -> `485c23cf7e3c63c1f42a684db12ff5fadfdf75ca`. Read the supplied commit/stat/-U10 package once; a reference-only pass supplied line anchors. No git diff was rederived.
- The fix report gives the exact focused command selecting `remote_management_denies_next`, `delayed_local_stop` and `scanner_admission_requires` across the service/store tests: 13 intended failures, then 13 passes. The described failures match the new assertions and the missing predicate in the fix diff, rather than fixture setup errors.
- Reported complete amended coverage is `python -m pytest Tests/Workspaces/test_project_index_service.py Tests/DB/test_workspace_project_index_db.py -q`: 54 passed, 1 inherited warning in 764.23s. The subsequent equivalent test-only lambda-to-def lint correction was followed by the affected delayed-stop selection: 3 passed in 68.24s. The report records scoped Ruff, four-file format and diff checks passing, and discloses the inherited pytest/Pydantic output. These are retained implementer receipts; no independent rerun or raw-log regeneration is claimed.
- The appended initial-RED clarification supplies original transcript event IDs, commands and assertion outcomes. It also identifies actual K2 Windows ACL list-denial evidence (1 passed in 2.60s). These correct the corresponding evidence caveats in the original review; the separate POSIX chmod skip and absence of POSIX execution qualification remain limits. No recovery of those old receipts required another test run.
- No test suite or focused probe was run during this re-review: the changed predicates, captured identities and supplied targeted tests resolve the two concrete doubts. No repository crawl, untouched-source inspection, subagent, source/index/HEAD/branch mutation or root metadata closure occurred. Only this authorized review artifact was written with explicit UTF-8 via escalated shell.

## Verdict

**Fix round: All findings addressed, no new Critical/Important breakage.**

**Scoped spec compliance: Approved. Scoped code quality: Approved.** Both original blocking findings are closed at this fix head. Original nonblocking observations and K4/K5/platform integration limits remain recorded; this is the K3 task gate, not whole-branch or merge approval.