# PR #3162 local backlog rebase qualification — 2026-10-31

The six original PR commits were replayed onto `57e330e735fff25515778e839f31ed85602c34f2`, producing `52269c3dd4b330a5cf6ccd938e7ea1f467a05cf0` before this local evidence/pin commit. Original remote head: `366bd3dbfa620b466c6fecb38c363e3093130841`. This record qualifies that integration; it does not claim a fresh final-dev rebase, current-head independent review, publication, CI, or merge.

ADR required: no. Existing [ADR166](../../../backlog/decisions/166-agent-assisted-archive-recovery.md) and [ADR228](../../../backlog/decisions/228-sqlite-store-kernel.md) govern the archive and SQLite custody boundaries. The relevant current portrait and binding-authority decisions remain in force.

## Integration decisions

- ChaChaNotes archive chronology now follows the existing portrait83 and region84 migrations as **84→85**. Workspace archive chronology follows binding-authority9 as **9→10**. Both SQL files are real runtime migration inputs. Existing data, index, transaction rollback and version fences are retained.
- All frozen ChaChaNotes predecessor catalogs through84 and Workspace8/9 catalogs were preserved. The exact85/10 successors and explicit recovery steps were added; native authorizers and validation allowlists were not broadened. Hybrid Subscriptions predecessor support follows the same explicit portrait/region/archive sequence.
- Workspace8 backups still pass frozen-catalog validation and still refuse automatic migration with `unsupported_schema_migration`, as current dev already required. The9→10 archive recovery step does not invent an8→9 authority migration. Normal live Workspace8→10 opening remains tested.
- BaseDB keeps current dev's native private connector, nested operation ownership, admission and retiring-handle behavior. The incoming finite deadline is forwarded into the private connection owner and translated through the existing helper timeout path. Workspace constructor failure closes its owned native connection.
- Conversation archive/restore advances the local optimistic version. Archive-only writes do not emit sync payloads and do not prune the latest shared payload. Shared edits, including mixed archive/shared writes, still emit updates. Existing cycle/stale-receipt tests cover this contract; Workspace archive_revision remains the clock-independent restoration fence.
- Root's source conflict resolutions retained both the current Console presenter/view-hook ownership and the archive hooks, current environment_scope_stamp imports, typed thinking ownership, and the current lazy-import census members. The registry test conflict retains current dev's explicit database fixture ownership and applies the new archive tests through that fixture.

## Evidence

All new runs use the shared Python 3.12 environment with native 0700 private config/temp directories created through `platform_files.os`. Ordinary Windows temp permissions failed admission before collection on the first attempt; no privacy guard was bypassed. Commands, JUnit receipts and byte-preserving compressed logs are in [the companion directory](2026-10-31-pr3162-backlog-rebase/).

Earlier rebase stages:

| Scope | Result | Qualification |
| --- | --- | --- |
| Fresh/migration/archive/recovery DB batch | 67 passed,253.46s | Before later replay commits; original tool output retained in the session. |
| Remaining predecessor/portrait/kernel/archive-query batch | 29 passed,3 failed,299.67s | [JUnit](2026-10-31-pr3162-backlog-rebase/schema-followup.xml.gz). All behavior passed. Failures were the current-base decorative SQL list and two Evals version-guard expectations. |
| Native connection retirement, failed Workspace construction, registry cleanup/archive snapshot | 24 passed,132.53s | [JUnit](2026-10-31-pr3162-backlog-rebase/cleanup.xml.gz). After the cleanup commit conflict resolution. |

The earlier intermediate run had 20 passes before 3 failures: one helper timeout under concurrent work and two old portrait-head expectations already being corrected. The timed-out 83 predecessor aggregate passed its focused rerun in 23.32s. These intermediate outcomes are not presented as final-head successes. The subsequent replay changed the Evals guard expectations, so those two historical failures require a current scoped check. The nine decorative SQL files are unchanged base debt; neither new archive SQL is in that set.

Final integration-head outcomes (the failing receipts are retained):

| Scope | Result | Receipt |
| --- | --- | --- |
| Original tools, search, coordinator, operations, result store, review-hook and dispatch selection | 173 passed; 4 dispatch confirmation waits timed out | [feature.xml](2026-10-31-pr3162-backlog-rebase/feature.xml), [log](2026-10-31-pr3162-backlog-rebase/feature.log.gz) |
| Original cards plus first mounted variants | 14 card tests passed; 4 mounted flows failed before provider entry | [mounted.xml](2026-10-31-pr3162-backlog-rebase/mounted.xml) |
| Fresh85 catalog, shared sync retention/stale cycle/mixed writes, corrected migration guards, CSS and all affected size rows | 32 passed in 32.41s | [guards.xml](2026-10-31-pr3162-backlog-rebase/guards.xml) |
| Bounded mounted retry after fixture retirement/wait correction | Button/100×30 journey passed all assertions; agent/160×44 failed before provider entry | [integration.xml](2026-10-31-pr3162-backlog-rebase/integration.xml) |
| Stock received-send proof | Failed in the imported current-dev helper's 10s probe-entry wait, before archive-proof assertions | [remaining-feature.xml](2026-10-31-pr3162-backlog-rebase/remaining-feature.xml) |
| Isolated mounted agent owner diagnostic | Failed again; exact main-thread stack retained | [agent-owner.log.gz](2026-10-31-pr3162-backlog-rebase/agent-owner.log.gz) |
| Final focused ownership, exact thinking-owner, chronology, manual/received dispatch and captured structured follow-up | 21 passed; mounted agent remained failed (197.07s total) | [final-feature.xml](2026-10-31-pr3162-backlog-rebase/final-feature.xml), [stack/log](2026-10-31-pr3162-backlog-rebase/final-feature.log.gz) |

The original 15s dispatch wait failed serially once and passed unchanged on the next observation-only run (18.55s whole call). The 10s thread sample was in native Windows storage admission; hook preparation completed normally in 1.84s. The mounted agent's 30s sample instead found the main Textual loop in unrelated skill candidate discovery, building local skill trust through configuration admission and native private-path/handle validation. All 16 initially observed admission and skill-discovery files are byte-identical to the rebase base: [source comparison](2026-10-31-pr3162-backlog-rebase/native-admission-comparison.json). These are precise sampled locations, not a claim that every failed flow is proved baseline-only.

The test correction retires the actual ConsoleRuntime, consumes turn/readiness waiters, waits for a search-only turn before using its result card, and uses bounded observed-state/turn waits for the slower Windows admission path. Production search/operation deadlines and authority checks are unchanged. The mounted archive fixture now injects the empty skill inventory already represented by its captured execution configuration, avoiding unrelated profile-backed UI skill discovery. Real archive stores, runtime, tools, permissions, confirmation and provider transport double remain in use. The final focused run passed all 21 non-mounted cases, including both manual and received dispatch, exact thinking-owner positive/negative controls, real structured captured follow-up restoration, view-hook ownership and chronology. With empty skill discovery, the mounted agent progressed through provider tool loading but still did not publish results within 60s. Its sampled worker was in lifecycle owner-sequence allocation → AgentRunsDB.get_run → native storage admission; the two additional full files and four allocator/lifecycle methods match the base. The mounted-agent and stock received-send failures remain unresolved review blockers; neither is claimed baseline-only. No further broad retry was run.


## Static and generated evidence

[static.json](2026-10-31-pr3162-backlog-rebase/static.json) records the initial integration comparison; [final-static.json](2026-10-31-pr3162-backlog-rebase/final-static.json) parses all 90 changed Python files, including 49 product files, with no errors or duplicate test definitions and no introduced local Ruff diagnostics. Available Ruff is 0.4.5. It reports existing PR/base debt, so this is not a claim of a clean global lint run: 373 current diagnostics versus 305 on the rebase base and 368 on the original PR across the checked paths. Every apparent product addition was inspected against both parents; none is absent from both. New-versus-dev E721 findings are intentional exact-type custody validation. Test F811 findings are imported pytest fixtures shadowed by fixture parameters, not overwritten test bodies; E402/E731 test style warnings already exist on the original PR.

The diagnostic inventory verifies without regeneration: 708 owners, 1058 TASK-492 calls, 56 TASK-31551 calls, 7772 TASK-494 calls and 16 sink files. Its only changed entry is agent_runtime.py: 12 diagnostic calls remain 12; the review-hook warning now describes refusing recovery calls while preserving legacy behavior for other tools. It retains the existing tool-name/exception fields and adds no content, secret, path or URL field. No sink topology changes. [Review receipt](2026-10-31-pr3162-backlog-rebase/diagnostic-review.txt).

`python -X utf8 tldw_chatbook/css/build_css.py` succeeded and produced no working-tree change. All source modules and generated sheets were already synchronized after conflict resolution. Product/test/script diffs and the local working diff pass `git diff --check`.

## Size waiver and census

On 2026-10-31 the user explicitly instructed: “Ignore size limits and use requesting-code-review in place of qodo,” followed by skipping CI and merging after rebase and requesting-code-review for each PR. This local preparation applies that waiver only to exact measured affected size rows: [size-pins.json](2026-10-31-pr3162-backlog-rebase/size-pins.json). Twelve rows were measured, including the ChatScreen line/method pair; the duplicate controller dictionary key was removed. No production code was rearranged to satisfy size limits and no performance budget was raised.

The UI fast-lane census appends only `Tests/UI/test_console_archive_recovery_cards.py`, whose 14 cases passed. The original 1173-entry prefix is byte-for-byte unchanged and the floor is now 1174; the checker passes. The planned second archive file was deliberately deferred because its mounted/received qualification remains incomplete. No existing entry was removed or reordered. [Prefix receipt](2026-10-31-pr3162-backlog-rebase/census-prefix.json).

## Remaining qualification

No full suite or live endpoint request was run. The original endpoint allowance remains exhausted at 30/30; historical native/live evidence is preserved under the original PR's dated QA records and is not a new Windows live verification. The broad mounted run stopped before the exact OS-descriptor plateau test; Windows also lacks its required lsof tool. No new OS-descriptor plateau result is claimed. Direct native handle ownership tests remain separately reported. Failures and unexecuted selections remain visible in the receipts.

TASK-35585 remains In Progress. Root owns the final current-dev rebase, fresh independent requesting-code-review under the user's Qodo replacement, remote publication and merge. CI was explicitly waived by the user; this local record does not report hosted CI as passed.

## Current-dev integration onto0d05d27fa1 (2026-10-31)

The eight local commits through64e796aef1 were replayed onto fixed current dev `0d05d27fa1d569f1edebbc2346d43f8c03da9911`, producing provisional replay head `e1c7648729`. This section supersedes the earlier base for integration provenance; earlier test receipts and57e330 source comparisons remain historical evidence. [Range-diff](2026-10-31-pr3162-backlog-rebase/current-dev/range-diff.txt.gz) and the [current source manifest](2026-10-31-pr3162-backlog-rebase/current-dev/source-sha256.json) accompany the final compatibility corrections.

Conflict resolutions retain current Pressed inputs/stash and asynchronous snapshot ownership. Archive proof is attached only after the current producer/fences; both prepared and synchronous user admission retain every other configuration field. The deadline-aware ChaCha setup also consumes its PRAGMA cursor, preserving current dev's retained-frame WAL fix. Schema85/10, predecessor catalogs and native admission remain intact. The existing selection helpers stay in console_selection.py. All previous view-hook keys and values survive; five archive projections are added. Landed approval-suggestion and Kanban production files are byte-identical to current dev. Notification/plugin-named methods in intersecting owners are unchanged. Canonical CSS now changes only the four archive/Library source/bundle files relative to dev; all existing voice token definitions are retained.

Current scoped outcomes:

| Scope | Outcome |
| --- | --- |
| Dedicated normal/queued user-entry origin, comparing all non-proof fields to the real configuration producer |2 passed,61.42s; [receipt](2026-10-31-pr3162-backlog-rebase/current-dev/entry-proof-final.xml.gz) |
| Sync callback replacement, async replacement refusal, retained WAL setup frames |3 passed,8.91s; [receipt](2026-10-31-pr3162-backlog-rebase/current-dev/small-controls.xml.gz) |
| Exact12 affected size rows |Both original ceiling/slack functions pass for every row; dated waiver retained. |
| Census |1174 current entries preserved byte-for-byte; only verified archive cards appended; checker passes1175. |
| Static/inventory |90 changed Python files/48 product files parse; no duplicate tests. Conflict F821/F811 clean. The unchanged scanner recomputed all48 changed product sources against the committed base inventory: exact match,711 owners,16 sink files. Only the previously reviewed recovery-refusal warning differs; no regeneration needed. |
| Single mounted archive representative on current dev |Failed106.60s total, before provider entry, with original60s state wait and completed cleanup; [receipt](2026-10-31-pr3162-backlog-rebase/current-dev/mounted-representative.xml.gz), [trace](2026-10-31-pr3162-backlog-rebase/current-dev/mounted-representative.log.gz). |

The mounted trace finds activation and spend-projection workers waiting in storage_admission.initializing. Active workers are reading qualification evidence through native Windows stat and opening AgentRuns storage for count_subagents_by_conversation through verified private SQLite parents. The state is STREAMING/Agent running with no gateway/tool calls. This is precise current-base pre-provider evidence, not a passed archive journey or proof that archive-specific behavior is sound. The prior received-send failure remains unqualified; its rerun was conditional on the mounted case passing.

The original native wired MCP payload-thread assertions also failed for normal and queued entry. The setup-only probe confirms that source objects, selected config module/profile/paths and original methods remain valid before and after app construction. The fixture's imported test store forces db-less sessions ephemeral; current stock capture may keep those maxima empty and defer persisted maxima to composition. These eager-read expectations are therefore not evidence of lost archive proof. They remain unchanged for independent review. Early added fixed nonempty/empty-maximum assumptions also failed and were removed; their receipts are retained. The final dedicated proof test observes the actual producer instead of assuming a particular capture route or maximum.

The single authorized non-archive native-turn control also failed before provider entry:87.93s total/78.10s call, with the same profile/DB/view fixture, original60s wait,180s outer bound and full cleanup. Its activation worker was in bootstrap._registry → native Windows stat/open_handle; another database worker waited in storage_admission.initializing. Its frozen source changes only the provider response/user text and the completion assertion. This reproduces shared pre-provider work without archive tool execution; it does not qualify the archive journey or establish a platform-wide cause. [Control receipt](2026-10-31-pr3162-backlog-rebase/current-dev/nonarchive-control.xml.gz), [exact stack](2026-10-31-pr3162-backlog-rebase/current-dev/nonarchive-control-stack.txt), [full log](2026-10-31-pr3162-backlog-rebase/current-dev/nonarchive-control.log.gz). No retry or received-send rerun followed. Root owns independent review of the clean code handoff. Task35585 remains In Progress/AC2 unchecked. No broad test battery, live endpoint request or remote operation was performed.
