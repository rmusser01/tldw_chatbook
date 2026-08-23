# SDD ledger — plan: Docs/superpowers/plans/2026-08-22-console-library-controls.md

## Setup

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/console-library-controls`
- Branch: `docs/console-rag-ux-design`
- Execution base after reconciliation: `925ea32ba` (`origin/dev`)
- Reviewed design commits after rebase: `719f5c232..a514ff870`
- Spec authority: `Docs/superpowers/specs/2026-08-22-console-library-controls-design.md`
- ADR: `backlog/decisions/079-console-library-conversation-authority.md` (Accepted)
- Schema head rechecked at 44; v45 remains available.
- Collision scan: PR #1933 is the target; no separate Console Library implementation was found. PR #1659 is workspace creation work and does not overlap this policy/recovery scope.
- Ruling: Treat the approved 2026-08-22 spec/ADR/plan as the replacement for PR #1933's original one-commit 2026-08-21 proposal, rebased onto current `origin/dev` — the user approved the replacement design and selected execution — cost if wrong: updating the PR will rewrite its feature-branch history, so no push occurs without the owner's explicit approval.
- Ruling: At the schema-delivery boundary, supplement the plan's named foundation battery with the complete `Tests/DB/` and `Tests/ChaChaNotesDB/` migration subtrees — the incident-backed migration lesson says schema bumps reach fixtures outside the changed modules, while this remains a targeted database sweep rather than a full repository sweep — cost if wrong: added verification time, with no product-state mutation.

## Preflight task self-consistency scan

| Task | Internal check | Result |
| --- | --- | --- |
| T1 | Branch/schema/baseline gates precede task metadata and production edits. | Consistent. |
| T2 | RED tests cover every produced pure/config contract before implementation. | Consistent; spec §3.2 resolves that fresh databases do not require a legacy seed. |
| T3 | Opener audit consumes T2's single sanitized helper; only legacy v44 opens require a non-None seed. | Consistent. |
| T4 | Historical fixture, pre-read IMMEDIATE lock, atomic v45 DDL/seed, and all four Sync-v1 triggers share one migration boundary. | Consistent. |
| T5 | Sync-v2/source proof and export/import compatibility land before runtime can emit new state. | Consistent. |
| T6 | Repositories own parameterized persistence/CAS; coordinator owns off-loop holder publication; contribution protocol stays generic. | Consistent. |
| T7 | Store lifecycle consumes T6 seams and delays identity publication until commit. | Consistent. |
| T8 | Task-local start gate precedes the configuration/final-context split and fresh authority capture. | Consistent. |
| T9 | Destination classification occurs after gateway normalization and stores no credentials. | Consistent. |
| T10 | Permanent reservation is independent of optional provider composition; policy cannot be widened by children. | Consistent. |
| T11 | Integrated runtime tests and Backlog closure consume T8–T10 only. | Consistent. |
| T12 | Pure preparation/action matrix and bounded sidecar precede controller integration. | Consistent. |
| T13 | Automatic retrieval moves out of screen ownership and cannot dispatch on failure. | Consistent. |
| T14 | Conversation/policy/USER/assistant/checkpoint/contributions share one commit; postcommit effects are idempotent. | Consistent. |
| T15 | Loader hydrates ownership before queue advancement; recovery actions use CAS and preserve one owner. | Consistent. |
| T16 | ADR-063 handoff precedes tool execution and one history predicate serves all projections. | Consistent. |
| T17 | Delivery-3 fault injection, lint, and task hygiene follow T12–T16. | Consistent. |
| T18 | One fixed two-axis chip opens a policy-only modal; screen remains thin. | Consistent. |
| T19 | Search-only modal and future-session Settings are distinct from live Direct/RAG selection. | Consistent. |
| T20 | Source rows, citation terminology, and Selected-turn focus share a pure display boundary. | Consistent. |
| T21 | Production hierarchy/full CSS render tests precede responsive styling and generated bundle refresh. | Consistent. |
| T22 | Minimized activity is captured at the trusted provider-result boundary before model delivery. | Consistent. |
| T23 | Store buffer, sidecar projection, atomic promotion, and redacted export consume bounded T22 events. | Consistent. |
| T24 | Inspector projection remains separate from Sources and closes only after the Delivery-5 battery. | Consistent. |
| T25 | Documentation follows verified product behavior and has a link/content contract test. | Consistent. |
| T26 | Recording provider drives production composition and the integrated battery remains targeted to this feature. | Consistent. |
| T27 | Live QA is explicitly isolated; repository-only purge proof and owner approval gate the full suite. | Consistent. |

## Shared-file pair scan

Each row represents one task pair that names at least one common file. Later tasks consume or extend the earlier task's owned seam; no contradictory edit ordering was found.

| Pair | Shared file(s) | Finding |
| --- | --- | --- |
| T2 ↔ T3 | `tldw_chatbook/config.py` | Ordered helper production/consumption. |
| T2 ↔ T19 | `tldw_chatbook/config.py` | Foundation defaults precede Settings UI. |
| T3 ↔ T4 | `tldw_chatbook/DB/ChaChaNotes_DB.py` | Constructor propagation precedes migration runner. |
| T3 ↔ T5 | `tldw_chatbook/DB/ChaChaNotes_DB.py` | Opener contract precedes message projection. |
| T3 ↔ T19 | `tldw_chatbook/config.py` | Migration helper and Settings defaults are separate sections. |
| T4 ↔ T5 | `tldw_chatbook/DB/ChaChaNotes_DB.py` | Schema/Sync-v1 precedes Sync-v2/export work. |
| T5 ↔ T12 | `tldw_chatbook/Chat/trajectory_export.py` | Message-state export precedes preparation sidecar exclusion. |
| T5 ↔ T16 | message sync/export/import projection files | T16 reuses and integration-tests T5's closed state seam. |
| T5 ↔ T23 | `tldw_chatbook/Chat/trajectory_export.py` | Message state precedes activity redaction. |
| T6 ↔ T7 | `tldw_chatbook/Chat/chat_persistence_service.py` | Repository adapter precedes lifecycle integration. |
| T6 ↔ T14 | `tldw_chatbook/Chat/chat_persistence_service.py` | Foundation atomic adapter precedes durable turn acceptance. |
| T6 ↔ T23 | `tldw_chatbook/Chat/chat_persistence_service.py` | Generic contribution seam precedes activity use. |
| T7 ↔ T13 | `tldw_chatbook/Chat/console_chat_store.py` | Store holders precede preparation ownership. |
| T7 ↔ T14 | persistence service and store | Staged identity precedes atomic acceptance. |
| T7 ↔ T15 | `tldw_chatbook/Chat/console_chat_store.py` | Lifecycle base precedes recovery hydration. |
| T7 ↔ T16 | `tldw_chatbook/Chat/console_chat_store.py` | Store base precedes continuation integration. |
| T7 ↔ T23 | persistence service and store | Generic promotion seam precedes activity contribution. |
| T8 ↔ T10 | `tldw_chatbook/Chat/console_chat_controller.py` | Final context precedes provider composition. |
| T8 ↔ T11 | TASK-19900.2 file | Start/implementation precedes closeout. |
| T8 ↔ T13 | controller | Frozen context precedes preparation integration. |
| T8 ↔ T14 | controller | Frozen context precedes atomic acceptance. |
| T8 ↔ T15 | controller | Frozen context precedes recovery action integration. |
| T8 ↔ T16 | controller | Frozen context precedes continuation/history integration. |
| T10 ↔ T13 | controller | Runtime provider policy precedes automatic preparation dispatch. |
| T10 ↔ T14 | controller | Provider gate precedes accepted-turn dispatch. |
| T10 ↔ T15 | controller | Provider gate precedes recovery retry. |
| T10 ↔ T16 | controller | Provider gate precedes continuation handoff. |
| T10 ↔ T18 | `chat_screen.py` | Runtime event seam precedes thin policy routing. |
| T10 ↔ T19 | `chat_screen.py` | Runtime selector precedes search/settings split. |
| T10 ↔ T20 | `chat_screen.py` | Runtime seam precedes source/selection routing. |
| T10 ↔ T22 | Direct/RAG provider files | Authorization precedes activity capture. |
| T10 ↔ T24 | `chat_screen.py` | Runtime seam precedes activity projection routing. |
| T12 ↔ T16 | `trajectory_export.py` | Preparation sidecar semantics precede all projection integration. |
| T12 ↔ T17 | TASK-19900.3 file | Start/implementation precedes closeout. |
| T12 ↔ T23 | trajectory model/export/import files | Preparation exclusion precedes activity sidecar addition. |
| T13 ↔ T14 | controller and store | Preparation ready state feeds atomic acceptance. |
| T13 ↔ T15 | controller and store | Preparation ownership feeds recovery. |
| T13 ↔ T16 | controller and store | Preparation/dispatch lifecycle feeds handoff. |
| T13 ↔ T18 | `wiring.py` | Store/controller wiring precedes policy controller wiring. |
| T13 ↔ T19 | `retrieval.py` | Automatic ownership removal precedes manual-search-only cleanup. |
| T13 ↔ T23 | store | Preparation state precedes activity buffer lifetime. |
| T13 ↔ T26 | controller tests | Unit behavior precedes production workflow qualification. |
| T14 ↔ T15 | controller and store | Durable owner commit precedes recovery. |
| T14 ↔ T16 | controller and store | Durable owner commit precedes continuation handoff. |
| T14 ↔ T23 | persistence service and store | Atomic acceptance precedes contribution-based activity persistence. |
| T15 ↔ T16 | controller and store | Dispatch recovery precedes ADR-063 transfer. |
| T15 ↔ T23 | store | Recovery lifetime precedes buffer integration. |
| T16 ↔ T22 | `agent_service.py` | Run/provider history base precedes actor attribution. |
| T16 ↔ T23 | store and trajectory export | Recovery projection precedes activity projection/redaction. |
| T18 ↔ T19 | `chat_screen.py` | Policy modal routing precedes search/settings split. |
| T18 ↔ T20 | display state and screen | Policy display precedes source/selection display additions. |
| T18 ↔ T24 | `chat_screen.py` | Policy routing precedes activity routing. |
| T18 ↔ T26 | status-chip tests | Unit copy/interaction precedes workflow qualification. |
| T19 ↔ T20 | `chat_screen.py` | Search/settings routing precedes source routing. |
| T19 ↔ T24 | `chat_screen.py` | Search/settings routing precedes activity routing. |
| T20 ↔ T24 | right rail, transcript, screen, and tests | Selected-turn/citation base precedes activity subsection. |
| T20 ↔ T26 | right-rail tests | Unit projection precedes workflow qualification. |
| T21 ↔ T26 | render tests | Responsive render contract precedes integrated scenarios. |
| T22 ↔ T24 | TASK-19900.5 file | Start/implementation precedes closeout. |
| T24 ↔ T26 | right-rail tests | Activity UI precedes integrated scenarios. |
| T25 ↔ T27 | TASK-19900.6 file | Documentation start precedes final qualification/closeout. |

## Shared-interface pair scan

| Producer ↔ consumer | Shared interface | Finding |
| --- | --- | --- |
| T2 ↔ T3 | `ConsoleLibraryMigrationSeed`, config seed helper | Exact producer/consumer names agree. |
| T2 ↔ T4 | migration seed and closed assistant state vocabulary | Exact values agree. |
| T2 ↔ T5 | `AssistantGenerationState` normalization | T5 owns boundary propagation only. |
| T2 ↔ T6 | policy/state value contracts | Repository types agree with frozen ledger. |
| T2 ↔ T7 | holder/default/read-result contracts | Lifecycle ownership is explicit. |
| T3 ↔ T4 | `CharactersRAGDB(..., console_library_migration_seed=...)` | Fresh/current vs legacy-v44 behavior agrees with spec §3.2. |
| T4 ↔ T5 | migrated Sync-v1 payload with explicit state | Sync-v2 compatibility lands immediately after schema. |
| T4 ↔ T6 | v45 policy/checkpoint/message schema | Repository SQL depends on completed migration. |
| T6 ↔ T7 | policy coordinator, repositories, contribution protocol | Store consumes without duplicating ownership. |
| T6 ↔ T12 | checkpoint states and generic contribution | Pure preparation reuses, does not redefine. |
| T6 ↔ T14 | `insert_with_messages` and contribution seam | One caller-owned transaction. |
| T6 ↔ T15 | checkpoint read/CAS/settlement/handoff primitives | Recovery uses expected versions/revisions. |
| T7 ↔ T8 | fresh `capture_for_execution` and session holder | Runtime never treats cache as authority. |
| T7 ↔ T14 | staged identity/publication seam | Publication remains postcommit. |
| T7 ↔ T23 | contribution-aware promotion | Activity lands through the existing generic seam. |
| T8 ↔ T10 | final execution context | Provider presence/selection is frozen. |
| T8 ↔ T12 | final execution context | Preparation consumes immutable authority/destination. |
| T8 ↔ T22 | attempt/run authority identity | Activity attribution follows the accepted turn. |
| T9 ↔ T11 | `ConsoleResolvedDestination` | Integrated disclosure tests consume exact identity. |
| T9 ↔ T14 | resolved destination in acceptance/checkpoint | Persisted form is credential-free. |
| T10 ↔ T11 | reserved names and policy-aware provider construction | Integrated gate consumes exact set. |
| T10 ↔ T22 | trusted Direct/RAG provider result seam | Capture cannot be spoofed by catalog source text. |
| T12 ↔ T13 | preparation state/outcome and preparation contribution | Controller integration follows pure state. |
| T12 ↔ T14 | ready preparation and bounded disclosure contribution | Atomic acceptance owns persistence. |
| T12 ↔ T15 | pause/action/state matrix | Recovery labels/actions remain UI-neutral. |
| T12 ↔ T23 | sidecar-only trajectory semantics | Activity adds a separate projection. |
| T13 ↔ T14 | prepared exact draft/evidence/authority | Acceptance cannot re-resolve inputs. |
| T14 ↔ T15 | durable USER/assistant/checkpoint owner | Recovery never creates a second row. |
| T15 ↔ T16 | exclusive dispatch-to-continuation ownership | ADR-063 wins before tools execute. |
| T16 ↔ T22 | run actor/provider invocation context | Attribution is bound at the authoritative call. |
| T18 ↔ T21 | policy chip/modal production composition | Responsive tests use the real hierarchy. |
| T19 ↔ T21 | search modal production composition | Responsive tests use the real hierarchy. |
| T20 ↔ T21 | compact source/display composition | Expanded-content matrix covers it. |
| T20 ↔ T24 | selected-turn focus and rail subsection ordering | Activity follows cited sources. |
| T22 ↔ T23 | bounded `LibraryActivityEvent` | Buffer persists already-minimized values only. |
| T23 ↔ T24 | `LibraryActivityView` and save state | Widget remains a thin projection. |
| T24 ↔ T26 | completed activity UI contract | Workflow fixture verifies production reachability. |
| T25 ↔ T26 | documented truth table vs deterministic production scenarios | Docs are checked against implemented claims. |
| T26 ↔ T27 | integrated evidence and recording provider | Live qualification builds on deterministic coverage. |

## Execution log

- Task 1 pre-feature baseline: 316 passed, 1 third-party RequestsDependencyWarning in 17.61s using `../../.venv/bin/python -m pytest Tests/ChaChaNotesDB/test_migration_atomicity.py Tests/Sync_Interop/test_chat_outbox_producer.py Tests/Chat/test_console_chat_store.py -q`.
- Remote Backlog filename sweep found no competing `task-19900` claim.
- Task 1 controller verification of reviewer ⚠️ items: the exact schema command targeted `tldw_chatbook/DB/ChaChaNotes_DB.py` and returned line 450 with version 44; collision/branch/PR checks and the 316-test baseline all ran before task-file mutation; the spec, ADR-079, and named lesson guidance were read before execution.
- Task 1: Ruling: retain the Backlog CLI's section markers and AC numbering — Task 1 explicitly mandates `backlog task edit ... --plan`, and Backlog 1.44.0 deterministically normalizes the task body while performing that required metadata transition; reverting the normalization would make the recorded file diverge from the CLI-owned format and would recur on the next required edit — cost if wrong: the Task-1 commit contains mechanical task-body formatting noise, but no semantic AC change or production change.
- Task 1: minor (deferred): the clean pre-feature baseline emits one environment-level `RequestsDependencyWarning` from installed `requests`/`urllib3`/charset packages; it predates the feature diff and should be triaged by the final review rather than hidden or fixed in this task.
- Task 1: complete (commits a514ff8..232881d, review passed after plan-mandated ruling; 1 deferred minor).
- Task 2: dispatched implementer `/root/task2_implementer` at base `232881d07` with strict RED-first TDD and report `task-2-report.md`.
- Task 2: initial implementation commit `815d3aec9`; review found 3 Important gaps (assistant default, strict quoted-bool fallback, malformed durable policy validation). Fix round 1 dispatched to the original implementer from fix base `815d3aec9`.
- Task 2: minor (deferred): the same environment-level RequestsDependencyWarning appears in the focused Task-2 run; final review should triage it with the baseline warning.
- Task 2: fix round 1/5 (3 addressed, 0 open — assistant default, strict malformed-bool fallback, durable policy validation; commits 815d3ae..cf97f83).
- Task 2: complete (commits 232881d..cf97f83, review clean; 73 focused tests passed, scoped Ruff and diff-check passed).
- Task 3: dispatched fresh implementer `/root/task3_implementer` at base `cf97f835d` with RED-first opener audit and report `task-3-report.md`.
- Task 3: minor (deferred): the focused opener audit exposes the baseline RequestsDependencyWarning plus three existing SyntaxWarnings; keep visible for final warning triage.
- Task 3: complete (commits cf97f83..c57e636, review clean; 5 focused tests passed, scoped Ruff and diff-check passed).
- Task 4: dispatched fresh implementer `/root/task4_implementer` at base `c57e63675` with memory/tmp-only migration TDD and report `task-4-report.md`.
- Task 4: initial implementation commit `74ab7873c`; review found 1 Important rollback-proof gap (seed insert and guarded version update were outside the failure-injection matrix). Fix round 1 dispatched to the original implementer from fix base `74ab7873c`.
- Task 4: minor (deferred): required migration runs expose the same baseline RequestsDependencyWarning; final warning triage owns it.
- Task 4: fix round 1/5 (1 addressed, 0 open — seed/version rollback injection; commits 74ab787..8329542).
- Task 4: complete (commits c57e636..8329542, review clean; 47 migration tests passed three consecutive times, adjacent opener tests passed, scoped Ruff and diff-check passed).
- Task 5: dispatched fresh implementer `/root/task5_implementer` at base `832954291` with strict sync/export RED-first compatibility tests and report `task-5-report.md`.
- Task 5: initial implementation commit `923fcd69e`; review found 6 Important gaps (legacy compatibility hash, persisted/delete proof validation, builder/apply continuation symmetry, Chatbook active-vs-complete reconciliation, active JSON/trajectory malformed-state rejection, Markdown evidence). Fix round 1 dispatched to the original implementer from fix base `923fcd69e`.
- Task 5 controller verification of reviewer ⚠️: Task 4's installed-trigger behavior test and report prove Sync-v1 create emits explicit NULL and all four final triggers carry the field; the missing-key compatibility normalizer does not stand alone as that proof.
- Task 5: minor (deferred): focused runs expose the baseline RequestsDependencyWarning; final warning triage owns it.
- Task 5: fix round 1 re-review left 2 Important items open: Chatbook import still downgraded `continuation_active` when no active checkpoint survived instead of rejecting the incompatible pair, and trajectory validation rejected every portable `continuation_active` projection even though trajectory export intentionally omits the private continuation. Fix round 2 dispatched to the original implementer from fix base `8e7e9a72b`.
- Task 5: fix round 2/5 (2 addressed, 0 open — strict Chatbook continuation ownership and portable trajectory `continuation_active` round-trip; commits 8e7e9a7..170d845).
- Task 5: complete (commits 8329542..170d845, review clean; 138 focused tests and 8 companion state-contract tests passed, scoped Ruff/source proof/diff-check passed).
- Task 6: dispatched fresh implementer `/root/task6_implementer` at base `170d8453d` with strict RED-first repository/coordinator/checkpoint/contribution coverage and report `task-6-report.md`.
- Task 6: initial implementation commit `1b57edc07`; review found 9 Important gaps (corrupt-row policy CAS, soft-deleted policy authority, active-lineage recovery selection, deleted-conversation recovery/mutation, dropped durable attachments, permissive checkpoint free-text/authority invariants, incomplete attempt/identifier validation, contribution early-commit escape, and capture rebind TOCTOU). Fix round 1 dispatched to the original implementer from fix base `1b57edc07`.
- Task 6: minor (deferred): focused runs expose the baseline RequestsDependencyWarning; final warning triage owns it.
- Task 6: fix round 1 re-review addressed 8/9 Important findings; the supplied raw `sqlite3.Cursor` remained an unavoidable transaction escape because a contribution could call `cursor.connection.set_authorizer(None)` before committing. Fix round 2 must amend the frozen spec/plan/ADR interface to a restricted transaction-writer capability before changing code, then prove the supplied capability exposes no connection or transaction-control path.
- Task 6: fix round 2 re-review confirmed the raw connection/commit escape was addressed, but found 1 new Important writer-validation gap: a `?` inside a literal/comment falsely proved parameterization and `INSERT ... ON CONFLICT DO UPDATE` violated the insert-only/no-upsert contract. Fix round 3 dispatched to the original implementer from fix base `0fdc008d1`.
- Task 6: fix round 3 re-review confirmed exact SQL validation, but found 1 new Important downstream-compatibility gap: planned preparation/activity contributions cannot atomically allocate mandatory unique `message_trajectory_metadata.seq` values through the frozen write-only capability. Per the fix-loop policy, fix round 4 is assigned to a fresh implementer from base `e75ad8344` and must add a narrow transaction-owned sequence allocator without reopening raw cursor/read authority.
- Task 6: fix round 1/5 (8 addressed, 1 open — all repository/coordinator/checkpoint findings fixed; raw cursor transaction escape remained; commits 1b57edc..3a849d2).
- Task 6: fix round 2/5 (raw cursor escape addressed by restricted writer; 1 new SQL-validation gap open; commits 3a849d2..0fdc008).
- Task 6: fix round 3/5 (exact parameterized insert-only grammar addressed; 1 new trajectory-sequence compatibility gap open; commits 0fdc008..e75ad83).
- Task 6: fix round 4/5 (trajectory sequence allocator addressed, 0 open; commits e75ad83..444e9b6).
- Task 6: complete (commits 170d845..444e9b6, review clean; 119 focused Task-6 tests and 118 adjacent tests passed, targeted mutation/static/source/diff checks passed; authoritative contribution contract amended consistently in spec/plan/ADR/brief).
- Task 7 controller ruling: the expanded Delivery-1 database gate may close with the six `test_core_sqlite_owner_privacy.py` failures documented as unrelated baseline because their exact node IDs reproduce identically at pre-Delivery1 commit `815d3aec9` and neither their tests nor media implementation changed through `444e9b624`; all 48 residual Delivery-1-induced failures were corrected and the final sweep was 1468 passed/1 skipped/6 verified-baseline failures. Cost if wrong: TASK-19900.1 closes while an independently existing media privacy defect remains visible and explicitly deferred, but no feature regression is hidden.
- Task 7: initial implementation commit `51ace1b3b`; review found 5 Important gaps (cross-database workspace membership written before the atomic Chat bundle, configured defaults bypassed by common production creation paths, incomplete holder unregister/register lifecycle, synchronous policy read during event-loop restore, and incomplete per-write-boundary/commit-timing evidence) plus 1 Minor stale v44 test comment. TASK-19900.1 must return to In Progress and affected ACs remain unchecked until the fix/re-review gate passes.
- Task 7: fix round 1 re-review addressed defaults, holder lifecycle, off-loop hydration, and promotion write-boundary coverage, but left 1 Important workspace atomicity escape through the instance-shadow legacy promotion branch, 1 required first-persistence post-exit observer gap, and stale v44/persistence API documentation. Fix round 2 dispatched to the original implementer from base `375c915a3`; TASK-19900.1 must reopen again until clean re-review.
- Task 7: fix round 1/5 (4 Important addressed, 1 Important atomicity escape plus evidence/docs open; commits 51ace1b..375c915).
- Task 7: fix round 2/5 (remaining atomicity/evidence/docs addressed, 0 Important open; commits 375c915..f7817be).
- Task 7: minor (deferred): `ConsoleChatPersistence` does not yet statically declare the now-mandatory `promote_console_conversation_bundle` method; runtime fails closed and the independent review found no behavioral/data-integrity escape. Final cross-task review should either add the protocol member or confirm a later persistence-interface task owns it.
- Task 7: complete (commits 444e9b6..f7817be, review approved with one deferred Minor; TASK-19900.1 Done with 13/13 ACs; 318 Store, 227 foundation, and 150 focused compatibility tests passed, scoped Ruff/diff passed; DB and UI baseline failures remain explicitly qualified).
- Task 8: initial implementation commit `76bcdc413`; review found 1 Important authority-boundary leak: retry/continue/regenerate/edit-resend/queue-retry/continuation recovery and provider fallbacks still passed a configuration-only snapshot into real execution. Fix round 1 dispatched to the original implementer from base `76bcdc413`.
- Task 8: minor (deferred): focused batteries continue to expose the pre-existing Requests dependency-version warning; final warning triage owns it.
- Task 8: fix round 1/5 (1 Important authority-boundary leak addressed, 0 open; commits 76bcdc4..87fac76).
- Task 8: complete (commits f7817be..87fac76, review clean; 89 exact, 443 compatibility, and 381 supplemental non-socket tests passed; scoped Ruff/diff and mutation probe passed; TASK-19900.2 remains In Progress).
- Task 9: conservative endpoint-only egress classification and isolated live-session disclosure implemented RED-first; 301 destination/gateway, 320 authority/store, 427 controller/bridge, 63 UI, and 52 checkpoint tests pass, with two sandbox socket tests deselected and only the inherited Requests dependency warning; scoped Ruff/diff and five mutation/negative probes pass; TASK-19900.2 remains In Progress.
- Task 9 fix round 1/5: all five review findings addressed at the real dispatch and exact-owner settlement seams; expanded RED 37 failed/27 passed, final focused 64 passed, full affected 553 passed, gateway 318 passed/2 sandbox socket deselections, Task-8 compatibility 94 passed, agent bridge 247 passed; five mutation probes plus scoped Ruff/diff pass; TASK-19900.2 remains In Progress.
- Task 9 fix round 2/5: decoded `http+unix` targets now reject whitespace, Cc/Cf controls, and malformed UTF-8; RED 6 failed/1 valid passed, focused GREEN 7 passed, destination/gateway 325 passed/2 sandbox socket deselections, authority/store 311 passed, decoded-validation mutation failed all 6 named invalid cases; scoped Ruff/diff pass; TASK-19900.2 remains In Progress.
- Task 10: one derived 19-name reservation now covers every Skill/MCP and skill-runner collision path; final-context construction admits only exact Direct/RAG built-in provider instances with their exact live issued authority. RED was 33 failed/102 passed; final scoped GREEN was 136 passed and affected compatibility was 616 passed, each with the inherited Requests warning. Blocked-gate, permanent-reservation, and third-party-spoof mutations failed as intended; scoped Ruff/diff pass. TASK-19900.2 remains In Progress with ACs unchecked.
- Task 10 fix round 1/5: replaced the provider-global latest authority with identity-keyed weak live capabilities owned strongly by each run registry; overlapping Direct/RAG registries remain independently usable through cache reset and peer cleanup, while copied markers still fail and 32 released registries leave empty issuer state. RED was 6 failed/113 deselected; exact GREEN 141 passed and affected aggregate 621 passed, each with the inherited Requests warning. Singleton and equality mutations failed 4 tests each; scoped Ruff/diff pass. TASK-19900.2 remains In Progress with ACs unchecked.
- Task 11: Delivery 2 integrated qualification added real two-handle/two-session durable freshness and freeze coverage, post-claim queued capture, bridge-level Allowed-to-Blocked cache invalidation, parent/child shared authority with narrowing, and the full four-policy by private/public/Unknown dispatch matrix. Baseline was 411 passed; test-authoring RED was 2 failed/247 deselected due a fixture bypassing required progressive disclosure and was corrected without production changes; final fresh battery was 424 passed/1 inherited warning in 16.73s. Stale-holder, fresh-composition/cache, reservation-removal, and Unknown-to-on-device mutations all failed their named ratchets; scoped Ruff/diff pass. No Task 12 work or production change was needed; TASK-19900.2 closed Done with 8/8 ACs.
- Task 11 fix round 1/5: reopened TASK-19900.2 and replaced the overclaimed durable seam with two real persistence/store/coordinator stacks plus submitted controller/agent execution; added first-external, exact-owner, and concurrent second-session production dispatch ratchets; narrowed Unknown evidence to the existing runtime projection because user-facing copy belongs to Task 20; amended the exact gate with the full store and tool-owner-cache suites. Final amended gate was 776 passed/1 inherited warning in 34.45s; six fresh mutations plus scoped Ruff/diff passed; no production defect/change or new lesson, and the Task-7 `ConsoleChatPersistence` static protocol gap remains explicitly deferred.
- Task 12: pure immutable preparation transitions/action data and the USER-owned bounded `library_preparation` sidecar contribution/projection are implemented; generic trajectory skips the sidecar, default/full export are equally canonical and bounded, and import is inert. RED was 2 missing-module collection errors; final focused was 92 passed, broad affected was 301 passed, and the unchanged controller/settings/queue gate was 252 passed, each with the inherited Requests warning. Nine RED/counterfactual probes, scoped Ruff, and diff-check passed. TASK-19900.3 remains In Progress with all 22 delivery ACs unchecked; controller/store/retrieval/recovery/UI integration remains Task 13+.
- Task 12 fix round 1/5: strict bounded construction and post-corruption guards now reject malformed/mutable preparation and transition shapes; real Task-8 context session/attempt authority is enforced, retries immutably rebind that authority, and one 792-shape Cartesian matrix owns the exact 21 legal transitions. RED was 1 missing-validation-API collection error; final state was 105 passed and affected adjacent was 350 passed, each with the inherited Requests warning. Constructor-disable, two illegal-edge, retry-rebind, and all original Task-12 mutations failed their named ratchets; scoped Ruff/diff pass. TASK-19900.3 remains In Progress with all 22 ACs unchecked.
- Task 13: eligible immediate/queued plain user text now uses a controller/store-owned fixed-category automatic Library gate with frozen Task-8 authority, exact scope, sealed evidence, bounded zero/bypass contributions, fail-closed pause, Retry/Bypass/Cancel, exact CAS, navigation lifetime, and queue-safe cancellation. Mounted retrieval retains manual search but no longer owns automatic send retrieval or fail-open notices. Baseline was 464 passed; RED was missing `ConsolePreparationOutcome`; final affected verification was 486 passed/1 inherited warning. All eight required mutations failed their ratchets; scoped Ruff/diff pass. TASK-19900.3 remains In Progress with all 22 ACs unchecked because Task14+ durability/recovery and recovery UI remain incomplete.
- Task 13 fix round 1/5: Retry/Bypass now continue the exact frozen manual or queued send through its real provider path once; queue recovery retains the exact claim and blocks later work without spin. COMMITTING/ACCEPTED/DISPATCH_STARTED align with real volatile acceptance and provider-attempt boundaries, all pre/post-preparation exits settle exact ownership, text plus attachments remains eligible, evidence-probe errors skip duplicate spend, and missing/changed typed destinations fail closed. The obsolete standing automatic toggle was removed outside its one-time migration input. A final boundary RED caught and fixed stranded USER-persistence failure. Final affected verification was 578 passed/1 inherited warning; runtime companions were 24 passed and 10 passed/1 independently defective fixture deselected. Nine mutations plus scoped Ruff/source/diff checks passed. TASK-19900.3 remains In Progress with all 22 ACs unchecked; no Task-14 durability/checkpoint or recovery UI is claimed.
- Task 13 fix round 2/5: recovered queue acceptance now defers only registry settlement to one exact-entry finalizer, so Retry/Bypass acknowledge accepted work once or return refusal/exception to the pending head with later work paused. A bounded live continuation freezes exact attachment objects, resolved prefill/one-shot identity, and the production evidence launch; recovery never rereads or clears newer staged state. Destination/reclaim/CAS/action races return stable submit results, dispatch markers sit immediately before the actual direct/agent call, and close leaves accepted ownership until the live task finalizes. Baseline was 578 passed; RED was 22 failures/35 passes; final affected was 601 passed and runtime/queue/UI companions were 100 passed, each with one inherited warning. Eight round-2 mutations, scoped Ruff/format, source scans, and diff checks passed. TASK-19900.3 remains In Progress with all 22 ACs unchecked; Task14+ persistence/reconstruction and recovery UI remain out of scope.
- Task 13 fix round 3/5: postaccept direct/agent recovery exceptions now preserve accepted USER/failed-assistant truth, acknowledge the exact reclaimed queue entry once, and never return it pending or duplicate it. A volatile full-submit task fence makes shutdown cancel/await PREPARING through accepted preflight work, rechecks immediately before both external calls, and lets close preserve COMMITTING/ACCEPTED owners until finalization. Monotonic prefill tokens protect same-text re-arms, and a real EvidenceBundle launch lease releases only the captured identity after acceptance. Baseline was 601 affected plus 100 runtime/queue/UI tests; bounded RED exposed 11 failures; final affected was 613 passed and companions were 100 passed, each with one inherited warning. Seven restored mutations plus scoped Ruff/format/source/privacy/diff checks passed. TASK-19900.3 remains In Progress with all 22 ACs unchecked; no Task14+ durable checkpoint/reconstruction or recovery UI is claimed.
- Task 13 fix round 4/5: exact Task→session ownership under a lock prevents same-session submit replacement; synchronous off-thread shutdown marshals teardown/cancellation to the owner loop; postaccept direct/agent cancellation returns exact committed accepted truth; READY close tolerates only its already-removed transient echo while retaining the evidence lease; and prefill revisions reject bool/non-int/negative values. The reviewer-expanded command reproduced 613 baseline tests and finished at 623 passed after 10 new probes; companions remained 100 passed, each with one inherited warning. Five restored mutation families were killed. Scoped Ruff lint and changed-range formatting pass; the prior whole-file format claim is withdrawn because clean HEAD already fails `console_chat_store.py`, while changed test/controller files and the changed store range are formatter-clean. Source/privacy/diff checks pass. TASK-19900.3 remains In Progress with all 22 ACs unchecked; no Task14+ durable checkpoint/reconstruction or recovery UI is claimed.
- Task 13 fix round 5/5: closed-loop submit tasks are synchronously detached under the task-key RLock and clean only their exclusively owned volatile preparation/outcome/continuation identity without any cancel/schedule/await; live same-session peers retain ownership and weakref/GC proves the controller cycle breaks. Queue/presentation callback failure now cannot skip preparation, submit/stream, or headless teardown; same-thread callers receive the original exception after cleanup, while off-thread scheduled teardown suppresses raw loop logging after cleanup. RED was 3 intended failures plus 1 off-thread privacy failure; final debug focus was 4 passed, exact affected gate 627 passed, and companions 100 passed, each with one inherited Requests warning. Both required mutations, scoped Ruff/format/source/privacy/diff checks passed. TASK-19900.3 remains In Progress with all 22 ACs unchecked; no Task14+ durable checkpoint/reconstruction or recovery UI is claimed.
- Task 13 lifecycle evidence correction: all 3 private asyncio warning suppressions/manual coroutine closes are removed. Supported awaited shutdown on a live owner loop now proves terminal task/controller collection and no lifecycle diagnostic; emergency already-closed-loop `begin_shutdown()` proves exact fail-closed volatile detachment/no provider/no closed-loop API call and publicly captures the expected destroyed-pending Task diagnostic without an uncaptured never-awaited warning. The mixed closed/live peer still preserves and cleanly shuts down the live owner. Baselines were 627 affected and 100 companions; final debug focus was 5 passed, affected was 628 passed, and companions remained 100 passed, each with only the inherited Requests warning. Round-5 blanket no-warning evidence is retracted for emergency detachment. Controller lifecycle docs, Task-13 report/notes, and an incident-backed testing lesson record the boundary. TASK-19900.3 remains In Progress with all 22 ACs unchecked; no clean emergency shutdown, Task14+ durability, or recovery claim is made.
- Task 14: durable manual/queued acceptance now commits conversation/policy, exact USER attachments, empty accepted assistant, v45 checkpoint, sync/hash/version owners, and Task-12 contributions in one `BEGIN IMMEDIATE`; preparation-keyed postcommit effects publish/clear/project/ack/hook/history/CAS/provider at most once and provider entry follows the accepted checkpoint CAS. RED was 24 failures; final Task14 was 25 passed, exact Task13 was 628 passed, companions were 100 passed, and DB/state companions were 62 passed, each with the inherited Requests warning. Nine mutations plus Ruff/privacy/diff checks passed. TASK-19900.3 remains In Progress with 22 unchecked; Task15 recovery/terminal/queue restart and Task16 continuation/projections remain incomplete.
- Task 14 fix round 1/5: durable sends now require the atomic adapter capability independently of `.db`; one locked global preparation/fingerprint owner rejects cross-session collisions and forged reuse; exact queued acknowledgement survives chain teardown; all used evidence/prefill makes reconstruction conservative; ten real postcommit seams and actual SQLite COMMIT failure are injected; and content-bearing recovery caches clean into a bounded 128-entry body-free tombstone set. RED was 9 failed/9 passed; final Task14 was 46 passed, exact Task13 was 628 passed, companions were 100 passed, and DB/state companions were 126 passed, each with the inherited Requests warning. Seven restored mutations plus scoped Ruff/format/privacy/diff checks passed. TASK-19900.3 remains In Progress with all 22 unchecked; the stale `dispatch_started` terminal checkpoint remains explicitly Task15-owned.
