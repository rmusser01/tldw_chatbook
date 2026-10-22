# K4 task review

## Spec compliance

**Issues found.** The implementation covers the approved K4 file/interface scope, including the narrowly authorized executor accessor and lexical-filter optimization, but does not yet satisfy conditional one-call approval provenance across permission-floor changes, the combined simultaneous-read limit, or the strict serialized-result ceiling in all stale-result cases. These are task-gate findings against BASE `9e6d127a0db2bb4841e219fccc65b58a865fffc6` -> HEAD `8ea34f516fd5e8a9f1cca3d3863cc463f8cc14e2`.

## Strengths

- Run authority is code-issued and checked against the captured workspace/binding/root/locator identities; optional specs are composed per alias and unioned when the first alias is Off (`tldw_chatbook/Agents/local_tool_provider.py:1106`, `:914`). The production preparation is offloaded after run-root capture (`tldw_chatbook/Chat/console_chat_controller.py:15630`, `:15656`), and the capability note follows final tool filtering (`tldw_chatbook/Chat/console_agent_bridge.py:5133`). SSH roots are explicitly excluded.
- Candidate ranking receives the union of the requesting floor and durable live exclusions before top-k. Live captures must match path, identity and hash, and final disclosure rechecks the registry witness, scope, state and manifest revision under the fence (`tldw_chatbook/Workspaces/project_index/service.py:251`, `:265`, `:294`, `:340`). Coverage remains None, preventing broader Settings counters from leaking to narrower runs. The old-scope refusal and separate retained-floor integration test accurately distinguish those contracts (`Tests/Agents/test_project_index_tool.py:205`).
- Normal conditional read reviews ignore name/session stamps and record exact run/call/arguments decisions (`tldw_chatbook/Agents/local_tool_provider.py:1256`, `:1276`). Native and synthesized runtime call-ID tests exercise the actual review/runtime flow; real worker/SQLite tests exercise stale files, late cross-process exclusions/detach, optional aliases and bounded output. Instruction-like content remains tool evidence (`Tests/Agents/test_project_index_tool.py:63`, `:611`).
- Query retirement failures retain operation ownership and make shutdown report unresolved native retirement (`tldw_chatbook/Workspaces/project_index/service.py:194`, `:806`). The report correctly distinguishes synthetic failure injection from real native-worker proof.
- The optimization preserves pre-ranking filtering and query-local normalization without an authority cache (`tldw_chatbook/DB/Workspace_Project_Index_DB.py:807`). Its runtime-oracle test covers hidden/dotted names, Unicode/case, prefix boundaries and exclusions (`Tests/DB/test_workspace_project_index_db.py:441`). On the executed Python 3.12 runtime, the implementation and oracle agree; no unrelated query/schema changes appear in the diff.

## Consolidated findings

### Critical

None identified in this task scope.

### Important

**I1. Conditional approval provenance is lost if fs_read changes from Ask to Allowed before dispatch.**

Anchors: `tldw_chatbook/Chat/console_interrupt_rounds.py:5497`, `:5501`, `:5504`; `tldw_chatbook/Agents/local_tool_provider.py:1256`, `:3648`, `:3658`.

The hook still writes every conditional row's raw decision into the ordinary name-keyed stamp map before separately writing the normalized exact-call stamp. `_verdict_for` only consults the exact-call path when the *current* fs_read state is Ask. Reproduction by control flow: fs_search_index is Ask and fs_read is Ask when the row is reviewed; the approve-once/deny row receives a forged `always_allow` (or `approve_session`); fs_read becomes Allowed before invocation while search stays Ask. Dispatch now takes the ordinary name-stamp branch and persists the broad search grant at line 3658. Even an ordinary `approve_once` from the conditional row becomes a reusable name stamp in this branch. Thus narrowing only the exact-call copy does not meet the explicit no-standing-grant and one-call requirements. The existing floor-change test exercises the opposite direction (Allowed -> Ask), and the forged-decision test holds the floor at Ask.

Keep conditional-row provenance out of ordinary name-based approval stamps, and preserve/consume its exact-call semantics even if the floor changes before dispatch. Add focused Ask -> Allowed tests with search still Ask, forged broad responses, replay and sibling IDs; assert no persistent grant. Also cover a floor change between the generic fallback's initial resolution and its review row, so a newly conditional row cannot flow into generic persistence.

**I2. The writer and query readers do not share the two-read admission limit.**

Anchors: `tldw_chatbook/Workspaces/project_index/service.py:105`, `:202`, `:563`, `:729`; `tldw_chatbook/Agents/local_tool_provider.py:1190`; unchanged `tldw_chatbook/Workspaces/project_index/scanner.py:56`, `:80`, `:373`; unchanged `tldw_chatbook/Tools/project_index_scan_worker.py:242`, `:466`.

The semaphore admits two query operations, each creating its own scanner and worker. The writer uses the separate service scanner and `_start_pinned` checks only other writer runs, not active searches. Scanner locks are per instance and the worker performs the actual file read without any common permit. Consequently, two query captures can overlap one writer capture: three simultaneous reads. This is possible even within one binding when Refresh starts after both queries pass their pre-capture fences; changing the stored state to Building does not stop reads already in progress. Queries against Ready bindings can also coexist with a writer for another binding. The reported per-scanner max_active_reads=1 therefore cannot establish the combined <=2 requirement.

Apply a common read-admission budget to writer and query captures, retained until actual reader retirement; keep the existing independent cache/writer lifetime rules. Add a deterministic held-read overlap test that starts two searches and Refresh and observes the aggregate active-read peak.

**I3. A later stale candidate can push an already accepted result over 16 KiB.**

Anchors: `tldw_chatbook/Workspaces/project_index/service.py:303`, `:320`, `:332`, `:349`; `Tests/Agents/test_project_index_tool.py:330`.

The byte check applies to each proposed hit using the stale flag at that moment. After hits fit, a subsequent stale candidate changes the final wrapper from `status=ready, refresh_required=false` to `status=partial, refresh_required=true`, adding one serialized byte. The final return is not checked again. A ready payload of exactly 16,384 bytes therefore returns 16,385 bytes if a later candidate is stale. This is attainable with two one-line excerpts of 12,000 and 4,130 characters at a.txt/b.txt, both below the per-excerpt cap, followed by a stale third candidate and default limit 10. A read-only stdlib JSON calculation using the exact DTO field layout confirmed 16,384 -> 16,385. Existing tests check stale-only results and long live results separately.

Reserve the largest possible final wrapper size when admitting hits, or enforce the budget on the final serialized result without corrupting JSON. Add the mixed valid-then-stale boundary regression.

### Minor

**M1. Verification output contains inherited warning noise.** `.superpowers/sdd/2026-10-20-workspace-project-keyword-index/k4-final-covering.log:8` records the existing Pydantic Field deprecation; the final profile also emits optional-audio dependency warnings. These are not new K4 defects and do not justify suppressing warnings or expanding this task. Preserve the qualification instead of describing verification output as pristine.

## Evidence assessed and review checks

- Read the brief first, then report, then the supplied 97,635-byte commit/stat/-U10 diff once in consecutive portions. Did not rederive the diff, read the whole plan, inspect excluded root-owned metadata, mutate product/index/HEAD, use network, or spawn agents.
- All 13 changed paths match the approved task scope. No added dependency, schema migration, automatic scan, Library mutation, or semantic index was introduced by this diff.
- Read the retained final logs: required covering **59 passed/1 warning**; focused DB **22 passed/1 warning**; final affected provider **9 passed/17 deselected/1 warning**. The hook/root-pin selection is correctly reported as **1 failed/59 passed/15 deselected/1 warning**, with WinError1314 at symlink fixture creation, before the intended authority assertion.
- Read final query-profile output: 20,000-file logical cache, 50 samples/5 warmups, common/rare/miss p95 **308.5711/242.7471/221.7858 ms**. This meets the permitted cache-only final timing qualification; it is not a repeated final live-Capture benchmark. Earlier failures remain acknowledged in the report.
- Named dependency check for I2: inspected scanner ownership/locks, capture_file, actual worker file-read loop, and the writer-start/scan path. This established that query permits do not constrain writer reads.
- Named integration check for I1: diff hunks cut off the surrounding review-hook aggregation and generic dispatch/fallback branches. Read those bounded source continuations to follow the raw and normalized stamps through dispatch and persistence. Also checked the handler continuation because the diff omitted how service output reaches final tool serialization.
- Ran only a pure-stdlib, read-only serialization calculation for the concrete I3 doubt; no imports of the application, native resources, fixtures, or test-suite reruns. No concurrency or permission integration test was rerun; I1/I2 are deterministic source-path findings requiring new focused regressions.

## Cannot-verify gates

- POSIX execution, GUI approval interaction, and actual recovery/restore during an active K4 query remain unqualified, as reported. Prior K1-K3 owner/retirement and storage-limit guarantees were not re-reviewed; final whole-branch verification owns those cross-task gates.
- The inherited selected-root symlink-swap test did not reach its assertion on this Windows account. A privilege-capable environment is needed for that specific security check; its setup failure is not passing evidence.
- Final performance evidence is the expressly approved canonical cache-only reconstruction/query path. Final live-file end-to-end timing and other Python/platform suffix runtimes were not executed; the source/oracle review does not claim they were.

## Task quality

**Needs fixes.** The implementation has strong run-bound authority checks, meaningful real integration coverage and candid evidence qualifications. I1-I3 are concrete missed contracts in the new K4 paths; approve this task only after narrow fixes and focused regressions for those paths.