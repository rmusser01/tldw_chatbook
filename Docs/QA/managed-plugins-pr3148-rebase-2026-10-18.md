# PR #3148 dev rebase verification — 2026-10-18

> Follow-up: the retained P3, Settings qualification and named static/CI blockers are now covered by the [blocker follow-up report](managed-plugins-pr3148-blockers-2026-10-18.md). The original runs, limits and 38 decisions below remain historical evidence.

## Outcome

[PR #3148](https://github.com/rmusser01/tldw_chatbook/pull/3148) is rebased onto recorded `dev` revision `b07fc0425288fb51d17cde5f745051720a7c9db6`. Conflicts were resolved, relevant targeted checks ran, and one fresh independent review completed. The reviewed source, `141369945f5ee4ae0e5d7b5fbb00a3fbce61728e`, was published and its remote head verified. The review supports the **draft update**; **merge readiness is not established**.

The original published head, `7b4d5c501ba8500267d8f9f84f524742c33ec882`, remains recoverable in protected refs. Publication used that exact force-with-lease. The subsequent task/documentation closeout changes no product or test object and receives its own lease-protected push. Draft status, auto-merge off, worktree and original evidence are preserved.

## Integration and source custody

The 33-commit integration reconciles overlapping plugin/hook implementations with current Console, MCP, workspace, consent, compaction, migration and lazy navigation owners. Main DB version 81 and receipt migration 75 remain canonical; the plugin registry migration from 4 to 5 remains. Authenticated historical interpretations stay distinct from fresh inspections. Installation or activation grants no tool or filesystem authority.

The final upstream fold preserved all 33,388 paths by mode/type/blob comparison: 210 branch-changed paths and 17 upstream-only changes, with no overlap in that fold. Earlier shared Settings, ChatScreen and CSS resolutions have separate literal three-way records. The two final correction commits change only tests and task/plan records; production bytes remain identical to `76acb63b8b826e07876e0ba2342e9945d648c174`.

Local evidence is preserved under `.superpowers/sdd/2026-10-17-managed-plugins-dev-rebase/`, including `root-new-dev-fold02/verified.json`, `root-replay-range-diff03.{json,txt}`, stop-by-stop resolutions, source snapshots and protected-ref records. Range comparison 03 ends at the penultimate test commit; the review package includes the final event-serialization correction.

## Actual targeted checks

Checks used fixed CPython 3.12.11, clean Git archives, a private pipless environment, complete private HOME/XDG configuration, null keyring, actual pre-import profile/network guards, a pre-DNS refusal control and frozen product-origin verification. Mapped Darwin checks used the actual boot primitive and test-owned numeric-loopback peers. No real user profile, external service, dependency installation or full test suite was used.

| Cohort | Source | Actual result |
| --- | --- | --- |
| Plugins failures and controls, phase 24 | `76acb63b` | 75 passed, 6 failed, 1 setup skip; 153.62s |
| Runtime/scratch and normal MCP, phase 25 | `76acb63b` | 29 passed; 15.95s |
| UI failures and controls, phase 26 | `76acb63b` | 10 passed, 2 failed; 93.88s |
| New-dev profile/session controls, phase 27 | `76acb63b` | 5 passed; 69.37s |
| Corrected fixtures, resume and MCP lanes, phase 28 | `f8bbc1a1` | 14 passed, 1 failed, no skips; 38.64s |
| Corrected keyboard cases and focus controls, phase 29 | `f8bbc1a1` | 6 passed; 88.48s |
| Remaining row-removal refusal case, phase 30 | `14136994` | 1 passed; 2.83s |

These are separate cohorts, not an additive suite-pass claim. The affected renewals reached their original assertions: all three MCP worker/effect authority lanes; automatic compaction; resumed frozen context and fresh attempt identity; vendor transform and final refusal; occupied admission; local operations; keyboard masking/reference/revocation; and matching refusal, nonmatching allowance and reservation cleanup after availability rows are removed.

The setup skip came from an owned-peer parameter named `live` colliding with the existing live-test keyword policy. Renaming that parameter allowed all three lane cases to run normally. The final failure came from shallow copying nested immutable event data in the test; the existing event serializer supplied strict JSON containers without relaxing the parser.

Every original input in phases 24–30 remained unchanged, with zero unexpected outputs. Known derived caches are recorded separately. Earlier failed phases remain intact. They include Plugins 947/96/1/3 passed/failed/error/skipped, runtime 1,394/10/1 passed/failed/error, UI 352/317/41 passed/failed/error, and the first renewal's 221 passed/32 call failures/9 setup errors. The named repair renewals supersede specific outcomes; they do not relabel those whole runs as passing. Unchanged passing evidence carries only within the recorded source/owner or unchanged-function applicability.

### Static checks and retained limits

R31–R33 preserved 1,398 existing production signatures; R36 preserved 761, adding only a private scratch publication query. Bounded Ruff comparisons introduced no diagnostics; R36 retained 194 exact existing diagnostics and 145 inherited formatter edits. The final five test files and three-line envelope correction compile and pass their targeted Ruff/Black checks. This is not repository-wide clean lint.

The runtime scratch publication query does no filesystem I/O and cannot itself detect external physical replacement. Inspected worker launch/acceptance paths retain physical checks; published effect currentness remains a separate check.

## Independent requesting-code-review

The sole fresh reviewer inspected substantive production boundaries in passes: admission/refusal, hook process/effect authority, Console/runtime/compaction, retained interpretations, acquisition/catalog lifecycle, credentials/variables, migration and UI/review owners. It used the requesting-code-review template without helper agents or repeated test execution.

**Findings: 0 Critical, 0 Important, 1 Minor (P3) in inspected source. Draft update: supported. Ready to merge: No.** Most test bodies, historical documentation, generated CSS and every conflict/golden artifact were not exhaustively reviewed. Zero Important findings does not establish that all failures are inherited or that all 210 changed files are fully proved.

Review package: `root-final-review-source04.diff`, SHA256 `bc5f7eda6890ee1678c5c4cd7e008e8ecc5940c8e39e0ccfc8c379e9167bffea`. Full local review: `root-final-review04.md`, SHA256 `42fba4380fe0701f20d7ad311f04762a2bf02956e136bfb69c23462b65f1c1da`.

### P3 retained: confirmation after candidate list reload

At `tldw_chatbook/Widgets/plugin_review_modal.py:563`, `PluginStatusModal._load()` replaces candidate rows/status on paging or refresh but keeps the prior review and enabled Confirm discard action. After reviewing A and moving away, confirmation may release A's reference while A is no longer visible. The service still validates that original exact review; this is not physical deletion or a demonstrated authority bypass.

This source-level finding remains open and is nonblocking for draft publication. Invalidate the review/confirmation on reload, or keep its exact target visibly pinned, and add a focused reload/page regression before calling it resolved. No product fix wave or scoped re-review was required for this retained Minor finding.

## Remaining qualification limits

- **281 Settings outcomes remain unqualified.** Five moved-helper consumers were repaired and renewed. Unchanged defining ASTs do not prove the other dependencies unaffected; the remaining cases are neither proven inherited failures nor proven new regressions.
- **The ChatScreen size gate fails:** 24,382 lines against a 24,369 cap, versus dev's 24,380. R29 retains 11 inherited lines plus two lazy-import lines without raising the cap. The dimension gate retains 18 matches in four exact-dev sheets; Plugins has none. Other inherited module/lint/format debt and the Library grammar/set failure remain recorded.
- Hook backup passed with the canonical private-HOME data root. Custom data-base configuration projection remains unqualified/unfixed.
- Full suite, native Linux/Windows, actual OS keychain/reboot, original vendor hosts and external executable replacement races remain unqualified. The independent reviewer did not perform live UI validation.
- Remote CI is separate from these local checks. It was not awaited as a merge gate. No merge or auto-merge was performed.

## Rulings I made

The following 38 decisions are reproduced chronologically with their reasons and costs. Earlier pending/checkpoint language records the state at that time; the completion and limits above supersede it for this integration outcome.

### R1

Ruling: R1 — treat the explicit rebase/conflict/targeted-check/review request as one new integration task, with one fresh implementer and one independent requesting-code-review gate covering the rebased PR — this authorizes adapting current owners and updating the existing PR, not redoing the26 completed implementation tasks or inheriting an exhausted prior final-wave cap — cost if wrong: reopen the specific integration outcome; preserve the original branch and evidence, keep root Git/Backlog control, no helpers/full suite/merge/auto-merge or automatic platform-support expansion.

### R2

Ruling: R2 — reopen existing TASK32695 with one new integration criterion and this plan rather than minting or renumbering another task — the task already owns lifecycle qualification and exists under the same identity on both trees, while the new work remains one atomic PR update — cost if wrong: split only a genuinely new architectural decision later; preserve its14 completed historical criteria and exact notes, and do not call old-base evidence new-base qualification.

### R3

Ruling: R3 — accept the StageA facade consolidation and exact signatures in section8.1, keeping current NativeHooks/host_references/MCPProviderGroup and actual consent/native/response-rule owners — both branches implement complementary security behavior and whole-file preference would lose live safeguards — cost if wrong: correct only the reached seam and rerun its actual entry/refusal controls; one private process adapter views existing custody, never double-reserves, loses required matchers, replaces current guards or restores removed infrastructure.

### R4

Ruling: R4 — preserve the bounded historical reference/adapter forms and issue new inspections under the exact /2 identities in section8.2, with complete authenticated matching and identical-semantic coalescing from section8.3 — existing protected records bind immutable interpretations and cannot be relabelled or silently upgraded — cost if wrong: refuse that retained record and correct its precise recipe/comparison using independent historical evidence; no fallback, automatic migration/re-signing, foreign grant, public normalizer registry or executable support upgrade from recovery.

### R5

Ruling: R5 — keep current ChaChaNotes81 and shipped receipt migration75, omit the obsolete branch receipt74 artifact, and retain the accepted plugin-registry4to5 marketplace migration — root verified identical919byte receipt DDL and current81 declaration; duplicate/downgrade would corrupt shipped migration ownership — cost if wrong: repair the exact migration/inventory boundary with real chain/current recovery controls; no new main-DB schema step or relaxed predecessor catalogue.

### R6

Ruling: R6 — accept the frozen-source/private-venv targeted-check plan and exact initial scope in root-stageB-path-scope01, allowing independent historical source captures to produce literal compatibility expectations — moving editable paths cannot qualify rebased consumers or immutable old interpretations — cost if wrong: retain the failed phase and renew only affected checks on proven source; use minimal child environment/private profiles/null keyring/external-network refusal, no blanket bootstrap or host/loopback grant, installs, cap changes, full suite or inherited-pass relabelling.

### R7

Ruling: R7 — resolve the initial c148 design-only stop to the explicit pinned-dev versions of its7 metadata paths — current dev retains the accepted later design/task/ADR index and root verified its exact original Codex snapshot lesson is present, with no product path in this commit — cost if wrong: restore the specific missing original fact from the protected branch; record these original/stage2/stage3 bytes, drop only if the resulting index is entirely empty, and keep later branch amendments/new compatibility decisions to their recorded replay stops rather than replacing newer current contracts.

### R8

Ruling: R8 — resolve the initial16bf planning-only stop to the explicit pinned-dev versions of its28 conflicted metadata paths — the original25 task identities/master plan already exist on dev and its hook/spec/ADR planning includes newer accepted amendments, with no product code in this commit — cost if wrong: restore a precise absent planning requirement from the protected source; preserve original task histories through their later replay/final root integration metadata, record the empty-index proof and retain current consent/migration/architecture planning instead of replaying obsolete plan text over it.

### R9

Ruling: R9 — retain current dev TASK32668 at the first F1 stop — direct comparison shows dev already Done with independently qualified strict Pydantic manifest validation and newer implementation notes, while the replayed initial task is an older InProgress state with the same4criteria — cost if wrong: preserve a precise missing historical note from the original branch; do not regress status, replace current validation with old counts or call either historical proof rebased-source qualification. Product resolution remains solely with the accepted implementer.

### R10

R10 — observed dev advance before qualification
Fresh remote read keeps PR lease7b4/automerge off but dev is now0c89dc4d56f65da8b79403dcb118eeb0727d8e57, a descendant of pinned1fac. Incremental product delta is only console_preparation_reads.py; twelve paths include its targeted tests/docs and a testing lesson. Complete the current reviewed fixed-basis replay, then rebase its resulting short chain onto observed dev before any stable-source checks/review. No new product contract, owner or broad diagnosis is authorized; maps retain original basis and exact provenance. Final review and qualification bind updated base; retain original pin for resolution/range evidence. Evidence: root-remote-ref-check01.json, root-dev-advance01.json.

### R11

R11 — preserve both closed native selector contracts
Current $alias:command:id / $alias:rule:id and strict named JSON remain. Original /alias:id / @alias:id route only exact typed captured rows at original spans, never generated reference text or fallback names/kinds. Original slash-only one-argument shorthand is opaque remainder; zero/multiple slash arguments use strict JSON. Keep current render_command(snapshot,component_id,arguments), ComponentRecord+explicit snapshot, existing AgentDefinition and unified available_skills. Only declared argument-name metadata is needed for route-local distinction; no public argument_form mode. Existing ADR162 and managed spec updated before product edits; root-I1-metadata-resolution02.json records latest hashes. This corrects earlier source-incomplete root inference, preserves both originals and current architecture, and grants no authority.

### R12

R12 — preserve current builtin-alias review capability
Root inspected exact fixed-pin host_references.capture_reference/validate_reference: current tool_references supports declared alias to builtin:* via actual BuiltinToolProvider schema/digest owner, not only MCP. Proposal58025c groupB would refuse new reviews it cannot represent canonically, unnecessarily removing this current capability. Accept I1 with narrow amendment: fresh explicit builtin-alias reviews keep existing closed dev_tool form and actual owner validation; canonical MCP references retain original strict target/dependency rules. Preserve old rows, competing-form refusal, actual parent/EMPTY and permission ceilings. No third representation or new owner/authority. Existing ADR162/spec amended before code; later a1 general canonical-new-review wording must explicitly carry this exception. Include a positive fresh builtin-alias review/admission/recovery/use witness and negative schema/parent controls.

### R13

R13 — I2 bounded catalog and foreign-agent reconciliation
Accept proposal7eb045b, full classificationf97835 and exact36-path scopef56c99. Four retained expected JSON files are qualification-only, generated from literal old sources after stable-source isolation; no fake hashes now. Fresh /2 preserves captured catalog absent external overlay, bounded canonical carrier for explicit external overlay, refuses conflicting simultaneous inputs; manifest wholesale replacement remains. Fresh /2 foreign agents require explicit supported tools: omission visibly blocks only affected agent, explicit empty remains EMPTY, no implicit inherit. Historical dev/branch remain exact. Existing ADR162/spec amended before product edits, no new ADR/authority/schema/dependency. Root-I2-handoff01 authorizes source-only implementation on captured067 inputs, no runtime tests/imports/Git or outside-path changes.

### R14

Ruling: R14 — I5 Workbench inventory coverage follows current Navigation/screen_registry.py plus shell_destinations.py; removed UI/Workbench/route_inventory.py stays absent. AC5 owner names change, full registered route/Meetings/Research/Plugins coverage/shortcuts remain. Current dev removed old owner. Cost if wrong: repair precise current coverage consumer and rerun route/palette/navigation controls; no resurrected infrastructure/weakened checks. Counts historical, integration AC15 pending.

### R15

Ruling: R15 — accept I5 existing SCREEN_OWNED_SPLITS/plugins-only prefix plus current app route CSS map. Defers owned browser selectors; shared/singular/type-only modal rules remain boot-loaded for standalone first use. Fixed-source estimate fits current cap without a new loader/modal owner/cap raise; exact later builder/runtime checks decide. Cost if wrong: correct exact selector ownership/cascade and rerun source-sync, actual cap, first-route and standalone modal controls; no cap increase or unrelated styling trim. Existing ADR162/spec/plan/task amended with ADR097 policy, no new exception/ADR.

### R16

Ruling: R16 — integrate I6 through actual async Console capture/current received-turn capacity gates, extracting only private nonawaiting reservation publication and allowing current private MCP composer to retain actual empty owned group solely for explicit discovery; normal run composition and all authority/kill-switch/ceilings remain unchanged — old248 inline/resolve-only capture and missing old composer do not match current host ownership, while a reviewed connection before discovery has no reviewed tools — cost if wrong: revise these bounded private seams and exact real-session/pending-turn/empty-discovery/cancellation/normal-run controls before qualification; no fake context, acquired tool grant, caller impersonation, lost received claim or unrelated cleanup. Accepted complete40dispositions ac4ce0d8/6212c98c; allbasis/read-only/inventory/meta hashes verified,3newcurrentinputs captured; original5metadata amended before code.

### R17

Ruling: R17 — adapt I7 to actual moved turn authority and ADR220 compaction owner using captured workspace and one named live release accessor; retain current exact typed skill/command/manual-rule execution/current_definition and narrow only skill supporting-file read by kind — historical session/controller bodies moved, and imposing original skill-only execute lookup would regress accepted R11 routes while current typednames already remove the old ambiguous fallback — cost if wrong: repair only these current owners/callers and renew actual owning-workspace/compaction cancellation+reservation+retry and typed manual/read/MCP/retired controls before qualification; no broad receiver, pending-turn theft, lost retry/failure gates, name-derived grant or unrelated scope release. Two new actual source paths captured exactbeforecode; existing ADR162/163/220 linked, five latest metadata02 hashes protected.

### R18

Ruling: R18 — preserve required occurrences through a private non-executable declaration projection and minimal atomic scoped catalog failure state; propagate only its fixed verification failure as existing RecoveryRequired through actual Console capture while keeping legitimate inactive/unselected/untrusted/revoked scopes empty — current strict cwd parsing and optional catalog exception wrappers could otherwise erase accepted required constraints during replay — cost if wrong: repair only these reached owners and renew exact matcher/nonmatch, missing cwd/body, scoped/global disable, healthy neighbor, synchronous/prepared capture, already-captured admission/final-publication and management no-read controls; no parser relaxation, stale authority, executable fallback, new store or permission grant.

### R19

Ruling: R19 — complete independent historical goldens with expected/provenance beside each package subdirectory and a valid existing literal working cwd, keeping workspace placeholders in argv/env and templated cwd as refusal — package capture hashes every member and both literal old parsers reject templated cwd, so the initial fixture sketch would self-hash or fail before its intended behavior — cost if wrong: recapture only the affected literal fixtures and renew their selector/actual-owner recovery comparisons; no production/parser relaxation, integrated-derived expected value, authority bypass or automatic migration.

### R20

Ruling: R20 — prove every frozen original input hash unchanged and separately retain only derived bytecode/known pytest cache outputs; launch with explicit-B and external pytest cache — first actual phase left all33354 input bytes unchanged but -I ignored the bytecode environment flag, causing476 generated caches and a whole-tree parity refusal — cost if wrong: discard affected qualification claims and regenerate the source/child provenance; unexpected outputs or any original-file change still fail, actual guards/profile isolation remain, and the initial result/cache artifacts and dependency-cache uncertainty are preserved.

### R21

Ruling: R21 — rerun only the sixteen source-bound retained cases with the actual Darwin boot primitive available, retaining the same frozen source, private complete profile, null keyring and pre-import network/profile guards — phase02 failed at real sysctlbyname kern.bootsessionuuid and the exact stdlib-only sandbox probe returned errno1 while two native reads succeeded with matching hashes — cost if wrong: withdraw only these qualification claims and retain the failed phase; no synthesized identity, changed production fallback, blanket host/network permission, or inherited-platform qualification. Native read evidence: root-boot-probe-{sandbox,native}01.json.

### R22

Ruling: R22 — accept the exact46path combined cleanup a9730867 and proposal b50fbfe8: restore original strict MCP string environment validation with bounded public-inspection regressions, place four historical vendor package fixtures outside their installation storage, and remove only introduced lint/format defects — actual phase02 reproduced the MCP NameError/containment refusals and full-tree dev comparisons distinguish84actionable Ruff items and494introduced formatter edits from7inherited import warnings/631inherited formatter edits — cost if wrong: restore the precise frozen basis and rerun affected owner controls; preserve lazy exports, ValueError refusal API, all historical observations/crypto/custody and inherited debt, no new authority, schema, dependency, global autofix or cap increase. Newly reached phase04 module/CSS growth requires a separately bound minimal helper/token addendum before those additional edits.

### R23

Ruling: R23 — accept budget/CSS addendum77c81976 with exact13extra paths03a6c288, retaining existing method signatures and actual owners while moving only the enumerated invocation/compaction, lazy export/draft projection, profile-action and route-CSS implementations; replace18Pluginsdimension literals with identical existing tokens and regenerate CSS via the current isolated guarded builder — actual phase04 and exact-base comparisons prove introduced size/style growth, and measured formatting adds160controller/151Settings lines — cost if wrong: restore affected frozen bodies and renew owning entry/refusal/lazy/cancellation tests; no cap increase, blank-line manipulation, new store/facade/public service, authority/grant or native-identity change. Controller≤29184,compaction≤4185,Settings≤35818,app≤5710; MCP≤literal-dev6878 retains its original6760cap failure explicitly. Seven inherited style sources and six other overbudget modules remain exactdev; source-only projections are not runtime passes.

### R24

Ruling: R24 — correct only R23 route-table placement: restore app_navigation.py to its exact1110line frozen body and place the unchanged private sixentry CSSmetadata table in existing screen_registry.py, preserving an explicit app class alias to the same dictionary — source-bound budget inspection caught navigation itself pinned at1110, so moving the literal there merely displaced the new failure; registry already owns route metadata and is already imported — cost if wrong: restore exact captured registry/app bodies and renew route/first-use/lazy/startup controls; no cap raise, unrelated simplification, new route fields/helper/public contract, loader/cache or authority change. Exact extra defining input is scope c94f8809/proposal af4fcc6d; existing ADR097/162 apply.

### R25

Ruling: R25 — use the actual Darwin qualification context only for the named repaired inspection/retention cohort and remaining Plugins cohort, with unchanged frozen3a7888 source, complete private profiles, null keyring, pre-import actual profile/network guards and normal per-test numeric-loopback markers — phase02 real boot reads failed errno1 and phase03 same16passed native; the new stdlib-only127.0.0.1:0 bind probe also fails errno1, while source shows only two explicitly marked local Git/catalog loopback cases and no allow_network/live declarations — cost if wrong: withdraw affected native/transport claims and retain failure logs; no synthetic identity, production fallback, source/fixture exemption, external-network grant, inherited Linux/Windows support or unrelated host execution.

### R26

Ruling: R26 — accept exact fivepath phase05 repair8e5d7a76/proposalad6a2363: correct used/unused review bindings at their respective test sites, align one fresh /2 test with the already accepted consumed-variable/dependency-closure contract, and fix only precise lint export spellings/annotation anchors — phase05 actual157cases148pass/9fail proves eight wrong-binding failures plus one stale global-variable assertion; ADR1621046-1053/spec2080-2083 explicitly preserve independent siblings while consumed references still fail closed; static comparison leaves33precise diagnostics and11source-proven inherited I001 blocks, all632remaining Black edits inherited — cost if wrong: restore frozen3a7888 inputs and renew exact nine failures plus existing consuming-variable controls; no production variable/adapter/auth semantics, exception/cancellation/refusal behavior, public wildcard exports, golden values, broad file ignore or cap change.

### R27

— Configure the qualification runner data root under its actual private XDG_DATA_HOME, as the existing child-launch invariant requires; preserve private mode/config/guards and rerun only the six setup errors. No product/test change. Cost if wrong: child-profile renewal remains invalid and must be repeated.

### R28

Ruling: R28 — Extend the same actual Darwin/private guarded context to the exact mapped runtime and UI cohorts: real managed-runtime owners require the already proven boot primitive; MCP owned wire peers bind numeric127.0.0.1:0 and forward only their exact selected port, whose actual sandbox bind refused errno1. Preserve all pre-import profile/network guards and negative control, complete private configuration, null keyring and per-test authority. No allow_network/live marker is selected. Cost if wrong: withdraw only affected cohort claims and rerun with corrected isolation; no external grants, synthetic boot identity or platform expansion.

### R29

Ruling: R29 — preserve the accepted two net lines of ChatScreen lazy import moves after the latest upstream status strip adds44 lines: actual24382/737methods versus literal dev24380/737 and cap24369/740. Report inherited11 plus branch2 as an open size-gate limit, with no cap change or unrelated abstraction/spacing compression. Both helpers remain absent until first use as the actual import-closure checks prove. Cost if wrong: reopen only this bounded size/lazy-owner tradeoff and its startup/first-use controls; do not call the failed size gate passing.

### R30

Ruling: R30 — qualify the unchanged hook-backup control with the literal complete default config and canonical data directory inside private HOME, removing only the root runner custom [paths] injection. Actual isolated phase17 and source projection show guarded config uses the injected custom base while normalized load_settings omits it and recovery inventory therefore observes a missing canonical parent. Preserve the restrictive test, private0700 layout/null keyring/actual guards/pre-DNS negative and source hashes; no product/path-authority change. Cost if wrong: withdraw the renewal and repair its exact environment/projection; custom data-base projection is not qualified or claimed fixed by this task.

### R31

Ruling: R31 — accept combined proposal1c816c11/scope2b3ce429 at ae5a7b: seven existing production owners and27 exact test paths repair the closed actual failures. Preserve required occurrences in one private immutable service/session/workspace/capture-bound refusal ceiling outside removable rows, issuing explicit known-empty captures; missing, foreign, stale or replaced proof refuses. Only original unavailable-hook metadata may return, never executable rows, absent archive producers or permissions. Validate consumed shared data, refuse only wholly missing/empty host lookup PATH, bind direct provider custody to checked typed origins, check actual input dependencies and issue fresh pending IDs per resumed attempt without recapturing the frozen maximum. Correct only named stale/private-lifetime/cross-thread test fixtures and five R23 Settings consumers. Reason: closed actual traces prove these six product gaps and stale test interfaces; all34 source bases and23 supports match. Cost if wrong: restore the exact captured inputs and renew affected positive/refusal/currentness controls; no new module/schema/owner/grant, historical golden/parser/cap change, broad profile exemption or unrelated baseline cleanup. Keep281 unrelated Settings outcomes, inherited Library/CSS/static and R29 size limits visible; runtime remains root-owned after source/static handoff.

### R32

Ruling: R32 — extend the accepted combined repair by only console_runtime.py and console_agent_bridge.py: their existing row-only predicates skip hook configuration/admission after all native rows are removed, before the service can restore required non-executable metadata. Bind their exact captured inputs and include the same private immutable captured-requirements indicator in these predicates and the already-scoped controller gates; explicit known-empty/local-only captures retain the cold-empty path. The indicator only routes to existing validation and cannot execute or admit anything itself. Use the private carrier boolean derived only from original unavailable_hooks; original ready-only rows removed stay removed. Cost if wrong: restore these two exact predicate bodies and renew actual engine/bridge/child plus cold-empty/refusal controls; no new owner/public API/grant, widened continuation, runtime bypass or unrelated refactor. Existing ADR162/163 R31 amendment already governs this reached custody handoff; no new ADR.

### R33

Ruling: R33 — preserve the existing legacy schema-v1 zero-authority resume solely at resume_maximum: validate any present captured proof, or require no native rows when proof is absent, apply the existing zero intersection, then issue an owned immutable empty refusal record preserving known workspace. Never reconstruct catalog authority or archive producers. Normal fresh/managed admit still refuses missing/foreign/stale proof. With an actual valid zero/empty record, the existing zero intersection may discard subsequently substituted native rows before membership validation, preserving the established no-reactivation control in test_continuation.py rather than accepting execution. Missing proof with native rows at the legacy owner still refuses. Cost if wrong: restore this exact legacy-only adaptation and renew parsed-v1 local/required-zero, substitution, managed-v2 archive and ordinary missing-proof controls; no generic zero/missing-proof bypass, owner/signature/schema/parser change or current-catalog fallback. Existing ADR162/163 zero-continuation contracts apply.

### R34

Ruling: R34 — accept only the one-case frozen MCP diagnostic fe4c11e5/inputa7420fe9, not any source repair: structured-stdio normalization on unchanged ce7b74c42f with the existing private runner/actual guards/null keyring/native context. One ignored evidence plugin temporarily observes exact MCPHookExecutor._invoke_worker and ToolCatalogRegistry.invoke_by_name calls, preserving each call/result/exception and all assertions, restored at teardown. Record at most32events/128KiB of bounded exception codes or hashes/24relative traceback frames and result type/ok/dispatch metadata, never arguments/payloads/credentials/environment/outputs. No nested pytest child is selected and the stdio peer remains uninstrumented. Reason: the ordinary dispatch passes but the generic worker catch hides the later cause; source inspection cannot prove a fix. Cost if wrong: discard diagnostic inference and retain raw failure/source provenance; no renewed qualification claim, production catch/permission/audit weakening, source copy edit, external network or expanded case. Root alone runs it after plugin source/self-check verification; existing source fixes remain unapproved. ADR required: no, test-only diagnostic implements existing ADR162/163 without contract or storage change.

### R35

Ruling: R35 — accept one additional SAME structured-stdio frozen diagnostic 0bae4ef3/inputc6af89ea. The actual first observation matches provider mcp_execution_failed at MCPToolProvider._execute catch1973/2000, which still hides its original exception; no product cause is inferred. Within only the hook catalog wrapper, a current-thread temporary trace observes exception events on that exact _execute code object, disables line/opcode events, records existing bounded safe codes/hashes/24frames and restores the exact prior trace in finally. Leave preexisting tracing untouched with a fixed unavailable marker; observe no other frame/thread/peer, locals/args/payloads/outputs or raw dynamic text. Retain32events/4096bytes each/128KiB total and original calls/results/exceptions/assertions/guards/source. Cost if wrong: discard only this diagnostic inference and retain both actual failed observations; no production catch/permission/audit weakening, source change or qualification claim. Root alone executes after artifact/stdlib helper proof. ADR required: no, a bounded test-only observation under existing contracts. Non-MCP combined11path proposal remains unapproved until actual MCP cause and final combined scope are complete.

### R36

Ruling: R36 — accept combined renewal proposalf0576f8b/scope0e8aca47 at ce7b74c42f: exact12existing paths (4production/8tests),35verified supports account for41failed outcomes. _MCPDelivery.current uses only the existing published effect callback on exact engine.loop, retaining full worker authority elsewhere and all context/dependency/cancel/deadline/inherited/permission/definition/audit/custody gates. NativeHooks retains physical scratch is_live on worker use/accept and adds only an exact-issued-record memory query under the existing manager lock for effects; no filesystem IO/lease/grant there. The memory query cannot detect external physical replacement itself. Resumed attempt-local frozen context retains the already-issued fresh pending ID and identical remaining maximum; input-hook failure abandons only exact preaccept preparation through the existing protected owner. Named fixture/expectation corrections preserve real guards/clicks/audits/terminal cleanup, including disconnected preapproval refusal. Reason: closed41outcomes, actual two bounded observations and exact source/caller proof identify these gaps. Cost if wrong: restore captured12inputs and renew only affected positive/refusal/currentness/custody paths; no new public API/owner/schema, body/parser/golden/CSS/cap change, permission bypass or unrelated cleanup. After repair commit, root folds observed devb07 (17upstream-only paths, zero overlap with entire branch) before any new stable qualification/review, proving all exact objects and affected private-profile behavior. Preserve original source/results; later newly reachable failures return to root rather than weaken assertions. Existing ADR162/163 and design amended before code; no new ADR.

### R37

Ruling: R37 — accept test-only proposal1e31110b/scopeb5465758 at76ac: five existing test files correct all eight closed failures and the accidental live-keyword setup skip. Assert both real automatic compaction decisions and four history units; compare the frozen maximum at first admission while retaining fresh IDs and exact private carrier identity at later narrowed admissions; remove stale file.read from the calculator event; enter the existing guarded plugin_test_admission owner and inspect lifecycle.engine; settle and scroll the real keyboard target before existing visibility/Enter assertions; rename only the owned positive lane parameter to current. Reason: exact current traces and reached source show fixture/expectation errors, not additional product changes. Cost if wrong: restore these five captured inputs and renew affected actual positive/refusal assertions; no production/guard/skip-policy/stylesheet/golden/cap change, --run-live, direct button invocation or synthetic permission. ADR required: no new ADR; direct test correction under existing ADR162/163/R31–R36. Root commits/freezes and renews affected failures plus listed controls, then dispatches the fresh independent review; no full suite or merge.

### R38

Ruling: R38 — correct only the row-removal test envelope with the existing HookEvent.model_dump data projection before lifecycle.event, preserving matching refusal, nonmatch allowance, required-only engine, no provider/live work and reservation cleanup. Reason: closed phase28 reaches the real engine but shallow source.data copying retains nested immutable mappings that the strict JSON parser correctly refuses; the existing tool pipeline uses this same serializer. Cost if wrong: restore the one captured test file and rerun its actual positive/refusal case; no production/parser/authority change, manual reservation or assertion loss. ADR required: no new ADR, test-only use of existing ADR163 event contract. Phase28 retains14passes/1failure/no skips and phase29sixpasses88.48s, all33388inputs unchanged/zero unexpected outputs. Root will commit/freeze and run only the remaining node before the independent review; full-suite/publication limits remain.
