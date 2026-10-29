# SDD ledger — plan: Docs/superpowers/plans/2026-10-28-textual-37266-37268.md

Identity: branch `codex/textual-task-37266-37268`, isolated worktree
`C:/Users/GDesktop-1/.codex/worktrees/textual-settings-patterns/tldw_tui`.
Starting HEAD `a5f952da225e4dc3e5ffcf116c1947969bea1ed4`; fetched dev
`7884e11a496a677c8180cc5d67234a57b7d659b9`. The one ahead commit is the approved
audit/tasks packet. No product delta is in that commit.

Spec authority: complete TASK-37266/37267/37268 records. All three requested by
the user for implementation, PR against dev, CI and review feedback; no merge.

## Rulings

Ruling: Execute the existing atomic task specs through serial subagent-driven implementation and independent task/final reviews — the user explicitly authorized burning down these reviewed bug-fix tasks — cost if wrong: workflow overhead or review-driven rework, with no unapproved product scope.

Ruling: Run targeted local tests only — the user's AGENTS.md requires opt-in for full sweeps and overrides generic skill full-suite instructions — cost if wrong: unrelated defects may first appear in hosted gates and must then be diagnosed.

Ruling: Reuse existing ADRs for the bounded repairs — current tasks and source preserve screen/service/storage owners — cost if wrong: any implementation boundary change requires a new ADR before that change proceeds.

Ruling: Use fresh native-private external profile/TEMP roots through the plan launcher — workspace/default TEMP native766 are unsafe for profile tests — cost if wrong: process/fixture overhead and failed qualification, never relaxed privacy guards.

Ruling: Preserve ResearchController's fetched-result cache contract absent a concrete current consumer requiring change — ResearchWindow selection/actions do not consume controller.selected_run and own their own current target — cost if wrong: a newly discovered consumer needs a bounded reviewed adaptation; no cache-mirror consistency is claimed without evidence.

## Preflight scan

| Task/interface pair | Checked requirements / produced-consumed state | Result |
| --- | --- | --- |
| Task1 internal | Raw submitted form, post-rail applicability, persisted metadata and newer selection | Consistent; no is_dirty-only shortcut or namespace merge is mandated. |
| Task2 internal | Window target vs controller fetch cache; timer/source state and empty detail | Cache wording narrowed to actual consumers under the ruling above; current window publication remains required. |
| Task3 internal | Synchronous writer off-loop vs UI callbacks, captured operation and no-clobber | Consistent; preserve admission and recoverable late-collision boundary. |
| Task4 internal | Existing census/limits, task closure and all-current-head CI | Consistent; PR open after green, no merge authorized. |
| Task1 / Task4 | New Evals cases consumed by shared census and reviews | Root exclusively owns census and integration metadata. |
| Task2 / Task4 | New Research cases consumed by shared census and reviews | Root exclusively owns census and integration metadata. |
| Task3 / Task4 | New Notes cases consumed by shared census and reviews | Root exclusively owns census and integration metadata. |
| Tasks1/2/3 shared native testing seam | Existing privacy/timeout fixtures consumed, not rewritten | One implementer at a time; unique run labels and immutable receipts. |

## Progress

- Plan: authored/self-reviewed against all task ACs and five review-focus risks.
- Workspace: native SDD script initialized plan-path marker; existing linked worktree reused.
- Task1: pending; baseline and implementer identity to be recorded before dispatch.
- Task2: pending.
- Task3: pending.
- Task4: pending integrated checks/review/closure/PR/current-head CI.

## Delivery gate

Do not stop at PR creation or partial CI. All three task reviews, whole-branch
review, actual current-head application/required/Perf CI and fresh feedback
check must finish. Use Superpowers instead of Qodo. Keep PR open.

Setup commit: `cc18b42ee8a8bea3d43c77e602eabdb245932f49`. All three tasks In Progress with CLI-written plans.
Task1 BASE: `cc18b42ee8a8bea3d43c77e602eabdb245932f49`; implementer pending dispatch.

Task1 implementer: `/root/implement37266`, fresh isolated context,
`gpt-6.1-sol` high. Status: running; root owns commits and later review.

Task1 checkpoint: 19 new native cases passed in `UNIQUE-task1-green-03`
(445.69s parent run); initial invalid harness runs and nine intended RED
assertions retained. Root identified older Save A / newer Save B / return-to-A
applicability risk; implementer confirmed and is adding a focused regression
and saved-baseline comparison before report/review. Existing controls pending.

Task4 read-only preparation: `inherited-capacity.json` pins all 1140 existing
selectors and their source hashes, retaining the larger prior/full-phase or
latest-complete-run weight per entry (17734.94s). No shared CI source changed.
New case charges, capacity choice and actual final-head hosted CI remain pending.

Task1 implementation HEAD: `fe84f97f1dc5db73e532b6bdcad2c5b8379d55f3`.
Root independently parsed actual XML/hashes: 20 final-source new cases passed,
24 existing behavior controls passed through the documented bootstrap marker,
and 4/5 governance checks passed. The canonical-class Windows separator failure
and initial legacy-control source-binding setup error remain explicit in report.
Scoped lint/format/whitespace and frozen source hashes verified. Review pending.

Ruling: Treat the canonical-class failure as inherited Windows path-separator debt and enroll that unchanged guard for an actual Linux hosted outcome — executed guard/CSS/registry inputs match BASE and the guard compares native backslash paths with registry slash paths — cost if wrong: a remaining class ownership defect must be repaired when the new hosted check runs; adds one bounded CI case rather than relaxing the guard.

Task1 reviewer: `/root/review37266`, fresh isolated `gpt-6.1-sol` high,
package `review-cc18b42ee8..fe84f97f1d.diff`. Waiting dual verdict.

Task1 review: spec not compliant / Needs fixes. Important1: value-equality
applicability misses normalized-equal newer Saves and same-Save raw roundtrips
with later editing context. Fix round1/5 starts from `fe84f97f1d`; original
implementer resumes with complete finding. Root read fix-loop.md before dispatch.
Task1: minor (deferred): inherited pytest/Pydantic/optional-audio warning output
is disclosed, not warning-free; final whole-branch review must triage it.
Task1 cannot-verify items: census/capacity/whole-branch review/current-head hosted
CI/green open PR are root-owned Task4 gates; no task-level completion claim yet.

Task4 preparation: `ci-static-baseline.json` pins seven candidate CI Python
sources: lint clean, only existing queue-pressure/derived-workflow formatter
debt. Two Ruff cache-access errors retained and resolved with `--no-cache`;
future scoped checks avoid shared cache writes. No CI source modified.

Task1 fix1 checkpoint: six new native intended REDs retained from FIX_BASE;
Character-only editor-local synchronous context/publication revision implemented.
`UNIQUE-task1-fix1-green-01` running 31 cases (all26 new, five selected existing).
Canonical Save control includes native retired-watcher focus transition check.
Do not redispatch Task1 or start Task2 until fix commit/scoped re-review approves.

Latest fetched `origin/dev`: `3efe8476cebf70eceafa605d1abed76a4aad4e0e`.
Its incoming changes do not touch the three owned product areas, profile fixtures,
census or derived workflow. Branch has NOT been rebased yet; do it during final
integration after all three task implementations/reviews.

Task1 fix1 implementation HEAD: `8fbc9d2e1d91c31654742736f652d173b07b3ea9`.
Root verified exact final source hashes/static outputs and actual receipts:
31 parent passes, all26 child passes; six intended RED children before fix.
Package `review-fe84f97f1d..8fbc9d2e1d.diff` ready. Waiting implementer to finish
report final result/doc hash (BA70482...) before resuming scoped reviewer.

Task1 fix1 report now complete (31 passed/770.80s, all26 actual child passes,
735.915s inclusive new-case parent work; final Docsha BA70482...). Resumed
`/root/review37266` with scoped fix package and complete Important1. Await verdict
at `task-1-fix1-review.md`; do not rerun passing cases or reopen initial diff.

Task1: fix round 1/5 (1 addressed, 0 open — value-equality applicability fixed
for equal-valued Saves and returned editing context; commits fe84f97f1d..8fbc9d2e1d).
Scoped re-review approved; no new Critical/Important or out-of-scope findings.
Task1 implementation/review complete. Backlog stays In Progress pending root
integrated checks/whole-branch review/closure. Deferred warning Minor and the
root-owned hosted Windows-guard qualification remain explicit.

Task2 BASE: `8fbc9d2e1d91c31654742736f652d173b07b3ea9`; pending fresh dispatch.

Task2 implementer: `/root/implement37267`, fresh isolated `gpt-6.1-sol` high;
brief includes complete authority/constraints and source-return/native-timer
runtime pointers. Running; root owns commits and later task review.

Task2 setup diagnostics: initial request-signature harness failure and then
pre-existing artifact API TypeError before intended timer assertion are retained,
not RED. Root confirmed controller `get_artifact(run_id, artifact_name, mode=...)`
versus real keyword-only scope signature; blame dates April/July, not this branch.
Local close owners TASK-16483 (presentation/fake auto-load tests) and TASK-21127
(service offload) are Done; broader follow-up ownership check remains root-owned.

Ruling: Keep the pre-existing Research artifact-loading signature defect outside TASK-37267 and arrange payload fixture state from real keyword scope reads — the task fixes native timer/source publication and does not require changing that separate controller-service call contract — cost if wrong: the existing artifact action remains broken and needs a separate tracked correction; no artifact-action success is claimed by these payload-retention controls.

Separate follow-up created via CLI: TASK-37330, medium/To Do, four unchecked ACs,
no implementation plan. Links TASK-16483/21127/19300/37267 and ADR033/068/078.
Fresh duplicate/allocation scan:412 refs,366 trees,8490 bodies,55 registered
worktrees,183 uncommitted task records,no errors; ceiling37300 +30 buffer.
Controller calls still positional in all412 refs and54 worktree heads; no pending
controller edits. Public filtered proof `artifact-followup-ownership.json`, private
creation receipt `artifact-followup-creation.json`. No extra implementation.

Task2 checkpoint: seven native cases GREEN (186.11s, all seven child XMLs pass),
all13 legacy behavior controls passed so far. Static guard failures under exact
baseline attribution; do not call whole batch green. Controller remains untouched.

Task2 implementation HEAD: `41e530c9f6a8936f85f5c09107907e22d27cd55d`.
Root independently parsed hashes/outcomes:3 intended RED children,7 final GREEN
children;13 legacy parent cases pass. Controls total46pass/8fail/0E/S,3 actual
private governance children pass. Root captured fresh scoped lint/format and
whitespace receipts (all0) because original static outputs lacked a saved receipt.
Exact owned source hashes and baseline-comparison booleans verified.

Ruling: Accept the unchanged timer/worker static failures as baseline limitations for TASK-37267's scoped review — immutable-input comparisons show identical120site/75root semantics and worker findings, with no new Research obligation — cost if wrong: existing platform and inventory gaps remain and any required hosted failure still needs repair; no all-green local governance or whole-repository clearance is claimed.

Task2 review: `/root/review37267`, fresh isolated `gpt-6.1-sol` high, approved
spec+quality with0Critical/Important; `task-2-review.md`. Baseline attribution,
payload fixture qualification and current action/source consumers independently
checked. Root-only completion gates remain pending. Task2 implementation/review
complete; Backlog stays In Progress for root integrated closure.
Task2: minor (deferred): inherited pytest/Pydantic/optional-dependency and splash
invalid-escape warnings remain disclosed for final review triage.

Task3 BASE: `41e530c9f6a8936f85f5c09107907e22d27cd55d`; pending fresh dispatch.

Task3 implementer `/root/implement37268`, fresh isolated `gpt-6.1-sol` high,
running from complete task brief and runtime pointers. Root owns Git and any
shared diagnostic-inventory refresh after the owned source is frozen/reviewed.

Task3 ownership expanded narrowly for two necessary existing synchronous test
assumptions: UI/test_library_shell ordinary note-write function+docstring and
Library/test_library_file_export Notes hard-link refusal function/native helper.
Preserve all assertions and no fake-screen product fallback; plan/brief updated.

Task3 checkpoint: intended native dispatch RED retained at
`task3-red-notes-dispatch` (1parent/1actual child assertion failure; main/writer
same thread, watchdog before App ACK). Earlier stale-property/newline/readiness/
focus harness diagnostics remain invalid evidence. Ten-case GREEN running.
Only Notes exporter product changed; local _fail reuses same sanitized warning
AST but moves owner from _write to _fail, so root inventory refresh is expected.

Task4 timing input correction before any CI mutation: the retained latest run
folder held36 logs, including8 non-UI jobs. The provisional model mixed those
with UI costs and double-counted24 entries. `inherited-ui-only-comparison.json`
proves the difference; old snapshot retained as
`inherited-capacity-all-lanes-provisional.json`. Corrected helper/model uses
exactly28 successful UI shard logs: inherited weight17138.64777s (was17734.94156).
All1140 selectors and prior worst costs retained; no CI configuration changed.

Task3 implementation HEAD: `6a122e0e946cdff5bdb0e1974e009969d7ffb159`.
Root independently parsed actual XML/hashes:1intended RED child,10 final GREEN
children (10parentpasses,1066.489s session),18/19 old controls pass. The one old
Escape/Replace case fails before export in source loading; exact retry repeats
that pre-export failure. Root verified five final hashes, all4 runtime/test pins
match executed GREEN; fresh scoped3-module lint/format0; shell3E721 exactly equal
baseline; owned-function formatting/other AST conservation verified.
Package `review-41e530c9f6..6a122e0e94.diff` ready; task review pending.

Ruling: Keep the older Notes Escape/Replace control locally unqualified and require its actual enrolled CI case to pass — both local failures occur in unchanged source loading before the modified exporter runs, while ten new native outcome cases pass — cost if wrong: a source-startup or escape/export defect must be repaired in final CI; neither failed local attempt is relabeled green or used to waive the existing gate.

Task3 shared diagnostic maintenance remains required: four calls stay four,
single Notes warning owner `_write` to `_fail`, inventory digest
`b4a748817136c4c4f8de` to `c352738411d1464094ba`; no sink topology delta. Initial
stdout-only drift log is empty and separately qualified, not a passing receipt.
Root must capture actual stdout/stderr/returncode before/after regeneration.

Task3 review /root/review37268 approved spec+quality,0Critical/Important.
Root read the complete report; local Escape/Replace, diagnostic/census/CI gates remain.
Minor(deferred): inherited warning noise and exactly3 baseline shell E721 findings.
All three task implementations/reviews complete; no redispatch.

Ruling: Enroll all44 new selections with32 serial UI shards and retain max-parallel17 under amended ADR-103 — measured full-phase model makes31 infeasible even at ideal balance, while32 forecasts999.24s against1020s — cost if wrong: extra checkout/later-wave work and another measured capacity adjustment; no timeout, test-selection or concurrency relaxation is allowed, and actual final-head CI remains required.

Task4 census contract RED: five intended assertions fail against unchanged28-shard/census source,0E/S. Capacity computation retains all1140 old +44 new =1184; 19142.87577s model. ADR amendment written before CI mutation. Three new split-file contracts added. Root portable Task3 archive99members/23XML, all hashes verified,10 actual child passes.

Task4 integration:262 distinct CI contract cases have passing latest observations across successive runs;9 backlog/diagnostic-engine/size cases pass. Initial integration34F isolated whitespaceIDs/header. Corrective32P/2F mixed live census during root rebase; frozen exact22/23 retry2P, no assertion/checker changes. Added incident lesson. Two description IDs only now explicit whitespace-free; final44-ID collection matches enrolled measured set exactly.

Rebase onto3efe8476ce succeeded as473fd01e6e; all owned blobs conserved except merged testing lesson file preserving both sides. Root task implementation notes recorded via CLI; statuses remain In Progress pending hosted closure. Inventory before1/statements0/write0 conserved all latest-dev data except expected4-call owner digest; real postcheck running.

Final diagnostic check0; before/statements/write/post1/0/0/0, only expected owner digest changed. Public rawreceipt ready. Whole-branch reviewer /root/final_review37266_37268 (fresh most-capable gpt-6-astra high) reviewing3efe8476ce..16b74ab4e2 with full128-path index and focused text. Sent completed postcheck proof; no product changes since package. Root opening PR/CI while review proceeds.

PR3290 created/attached: https://github.com/rmusser01/tldw_chatbook/pull/3290. Hosted HEAD7c2b8f5aab066035ed9d06f9719a93d184f8e5c3. Derived40810284866,Perf40810284730 queued/dispatched; no unresolved threads. No auto-merge armed; keep open. Final reviewer informed of tiny post-package metadata/evidence supplement. Critical hosted targets: oldNotesEscape file shard8, canonical guard shard22; all44 added selectors require actual duration rows and passing jobs.

Final whole-branch review approved for hosted qualification:0Critical/Important/newMinor; two inherited minor categories retained. Report final-review.md independently verifies raw archives/children/source pins/capacity. No source fix requested; no fix wave needed. Hosted/CLI closure remain real gates.

Ruling: Accept the final reviewers bounded scope deferrals while retaining every hosted completion gate — these fixes preserve existing remote, source-storage, catalog and atomic-writer contracts and their evidence makes no broader capability, rollback, terminal-frequency or whole-repo claim — cost if wrong: an overlooked supported journey requires a separately evidenced repair, and any actual current-head required failure still must be fixed before delivery.

Final review declined-to-judge dispositions (all explicitly accepted, none silently waived):
1. Live remote Research server/mutations: unchanged remote contract; async-double controls qualify source fence only; no live/rollback claim.
2. Artifact-button success: pre-existing contract defect has TASK37330; action still needs its separate repair, not a payload-retention claim.
3. Real Notes SQLite mutation: source read/capture owner unchanged; seeded source preservation plus real destination bytes only.
4. Arbitrary concurrent programmatic catalog restoration: existing owner paths, outside supported timer/view transitions repaired; any concrete supported regression remains actionable.
5. Rollback of a started retired-screen write: existing accepted atomic-write behavior intentionally continues; only retired UI publication suppressed.
6. Natural race frequency/storage latency/full terminal/SLA: native controlled ordering only, no measured frequency or SLA inferred.
7. Whole-repo warnings/static/format debt: conserved scoped limitations retained; any required hosted failure is still a repair gate.
8. Re-audit conserved81 historical audit blobs: source hashes prove unchanged background; new128-path branch claims/evidence reviewed.
9. Current hosted/platform-wide/queue/finalcompletion: no success assumed; root continues all actual checks, two named control passes, feedback and CLI closure.

Source HEAD7c2b8f5aab fully green: Derived40810284866 all40jobs, Perf40810284730.32 UI pass,max679s;44 added selectors,oldNotesEscape8,canonical22 actual rows+successful jobs. All41 retained job logs hash-verified.0unresolved threads. Qualified-proof.json/public hosted-source-head.json.

Backlog CLI closed37266/37267/37268, exit0 each,13/13AC checked and notes/final summaries populated.37330 staysToDo. No redispatch or rerun closure script. Root verified all tracked product/tests/scripts/workflows/security inputs identical to qualified7c2b8f5a. Final metadata commit/push follows; watch its actual current-head CI and feedback before delivery. Final review approved/no fixes; all11 Ruling lines must reach final response before deleting only this plan workspace.

FINAL METADATA HEAD3e414ff824f88bf8e13a5aef1003be3525481a87 pushed onPR3290. Source remains exactly7c2b8f5aab testedinputs (gitdiff exit0). ALL THREE TASKS DONE,13ACchecked,close-tasks.py finished0 for3; do NOT rerun closure. Public report/ledger/hosted-source-head/backlog-closure records committed.68 documentation links valid. Final reviewapproved0C/I/newMinor, no morecodechanges needed.

Current FINAL CI: Derived40815454377 (40expectedjobs),Perf40815454364; two one-off platform workflows intentionallyskipped40815454374/40815454444. Latest snapshot finalhead1applicationSuccess/10inprogress/27queued,Perfprogress,0threads. Continue watch-current.py3290; collect-current-logs.py3290 oncompletedjobs; qualify-current.py3290 ONLYafterallrunscomplete. They store proof under .test-tmp/textual-37266-37268/pr-3290, headed3e414ff8. First qualifiedsource7c2b proofpublickept, private qualified-proof.json currently7cwillbeoverwrittenbyqualifyfinal. Do not pushmoremetadata/triggerthirdcycle.

After finalheadallgreen, update PRbody frombodyfile with exact finalDerived/Perf links and observed44cases/oldNotes+canonical/currentheadmaxUItimes; no commit forfinalCIreceipt (publicQAexplicitlydelegatesfinalmetadatasnapshottoPRdescription). Inspect actualreviews/threads0,PRstateOPEN,Gitownedtrackedcleanexcept.testtmp. Extractall11Rulinglines from ledger into finalresponse (reason+costeach), preservepublicledgeralreadyinGit, then verifyabsolute thisplanSDDpathanddeleteONLYit withnativePowerShell. Keepprivate.testtmpandopenPRworktree. Finaluseranswertasks37266–68Done,PR3290opengreen,Superpowersapproved,37330separateToDo; no merge.

Metadata3e414ff824 CI finished:37Derivedjobs success/3failure (UI12,UI19,anddependentaggregate); Perfpass;2intentionaloneoffskips;0threads. UI12Media filterroduct→t and UI19Note reopen line245 are independently native-reproduced, not failures in the3 requested features. Native Media pacedkeys1F,original+Paste1P each; Note held-row delivery1F after originalpicker+autosave, titleprep1P; initialtitleNoMatches invalid forreopen. Reportsandrawsources/receipts retained.

Fresh ownercatalog417refs/368trees/8493bodies/55WT/182pendingrecords,0errors,max37330. TASK37360 createdonceMedium/ToDo+4uncheckedAC foractualMediaordinarytyping/prefix-loss bug; links priorowners/ADRs. ExistingNoteDroppedGestureownerTASK34000.85 foundinrefcatalog; no duplicatecreated.

Ruling: Repair the two existing Reader/link controls through verified native setup under TASK-34000.18 and TASK-34000.11 while retaining their full outcome assertions, and track broader product gestures separately — identical source passed firstCI and native probes prove setup publication/selection races outside these3 product fixes; Paste is a valid Reader setup, tree-plus-DOM readiness preserves the originalsingleNoteRowpress — cost if wrong: test preparation could hide a supported user-gesture defect, so retain negative oracles/native witnesses, explicitly leave Media typing open asTASK37360 and Notes action loss asTASK34000.85, and require actual final-head CI; neither product defect is claimed repaired.

Task5 combinedCI fixBASE3e414ff824; nativeAppslotfree; fixbriefci-fix-wave-brief.md. One fixsubagent andone scopedre-review underfinal-review protocol; no product changespermitted. ALL3 requestedtasksremainDone; do NOT rerunclose-tasks.py. Rulingcountnow12 (supersedes old11 cleanup instructions); everyRulingmustreachfinalreason/costlist. PublicQA/ledger mustretain failed3e run and newfollowup; no third-cycle avoidance claim after a real source/test fix.

CIwave1 mandatory ownedsource3Reader/fullLink/Preview sibling pass5; optional extendedconsumer failedline87. Oldhelper initialSourceBootstrapinvalid; readinessqualifiedoldhelperfirstEnterfollowedLisbonthenfailedline105TwoLinkPicker. They are not exactbaselineequivalent. CurrentreadinessTrace decisive: nativeEnterfocusedPreview→LibraryScreen→Work/Itemsrecompose→AppfocusItemsGrip, oldPreviewpruning, nofollowaction; currentIndex snapshot persists. NewhelperheldfullLink1P/noWatchdog; wrongReadernegativefails[2]!=[1]; heldalreadyList1P actualrailfreshEpoch4 confirmed.

Ruling: Extend the local note-link Preview setup barrier within the same CI fix wave while retaining every keyboard/link/picker oracle — actual native dispatch proves the helper can return on painted/focused controls that are about to be retired before Enter binds — cost if wrong: a helper-timing regression could be concealed, so require held-publication negative/positive plus the actual full keyboard consumer and scoped review; production input delivery during render remains an explicit existingTASK34000.85 obligation, not a claimed repair.

Rulingcountnow13. Root amendedbrief/plan before _preview edits;3ownedfilesunchangedscope, no product/timer/census changes. Final cleanup mustextractall13Rulinglines, supersedingearlier11/12notes.

CURRENTRESUME: PR3290 remoteHEADstill3e414ff824 FAILEDtwoUIcontrols+dependentaggregate, allotherjobs/Perfpassed,2intentionaloneoffskips,0threads. Task37266–68allDone13ACsource7cqualified; doNOTredispatchfeaturetasks orrerunclose-tasks.py. NewTASK37360MediumToDocreatedonce(followupCLI0),TASK37330ToDoexisting; broaderNotesgestureowner34000.85inrefcatalog(existingnotcurrentfile). Freshcatalog417refs/368trees/8493bodies/55WT/182pending/max37330; raw52MBsinglelinePRIVATE, NEVERGetContent/dump.

Native investigations completed/archivedci-repairs/native-investigations.tar.gz(37members6directcases allhasheschecked). ReportsMedia/Note showmatchingmechanisms; historicalCIorderingonlyinferred. Twoinvestigatoragentsdone. Soleactiveimplementer /root/fix3290_ci_wave(gpt6.1solhigh), owns3testfilesONLY, noGit/task/productwrites. NativeAppslotownedbyfixer; rootmustnotApp/rebase. CombinedwaveBASE3e; readci-fix-wave-brief incl Root-approvedPreview amendment. RootDocTask5+Ruling12/13 recordedbeforeedits.

Fixcurrentevidence: owned-source-01required3Reader/fullLink/Preview sibling5P; optionalExtendedkeyboard1Fline87. Heldfullnew1P81sNoWatchdog realRootFoldersheldhelperpending then1press/1load; wrongReadernegativeintended[2]!=[1]; alreadyListheld1PactualrailrefreshEpoch4 retainsrailaction. Static3fileslint0 andexactbaselineformatdebt/noNew, ASToutsideauthorizedowners/tailsidentical. InvalidscratchcollectionreadconfigbeforeUIbootstrapNEVERApp/privateonlysummary; SourceBootstrapsfailuresnotRED. OldhelperoptionalreadinessbaselinepassedfirstsingleLinkEnterthenFline105TwoLinkPicker; NOTexactequivalencefornewline87. NewreadinessTrace provesEntertofocusedPreview→pendingWorkrecompose→focusGripbeforeAppbinding,noLisbonload. Rootamendedscopebefore_preview editsinSAMEsupportfile, keepsalloracles/bounds, no productfix. Fixer nowqualifying_previewpublicationbarrier+heldold/new+extendedkeyboard/sibling. Waitcompleteci-fix-wave-report.md thenverifyactualraw/source pins, doNOTrerunpassedReaderunlesssourcechanged.

ROOTMETADATApending: DocQAintrocorrected; backlog-closure.jsonqualificationhistorical7c→3e only; ci-repairs/failed-metadata-head.json retainsactual2fail+aggregate andsourceconservation0. Addedactualincidentlessons. Needfinaltask5checklist/QA/ledger+archivefixproofs afterreport. Taskowners34000.18/.11 alreadyread; CLI--append-notesSUPPORTED (useappendpreservehistory,nohigherTASKidlinksfromoldtasks;QA linkinstead), doafterimplementationreview. NewMediaTypingToDo37360doesNOTrepairproduct; nativePasteprepsReaderonly.

NEXT: One scopedindependentSuperpowersre-review ofcombinedfixwave(standardsolhighfreshnochildren), usingdiff3e..newfixcommit andcomplete2CI+Previewfindinglist/proof; reportsourceTask5. NoQodo/nomerge. Afterreview0addressmetadata, pushownbranchONEtimewithfix+docs, thenwatchactualnewheadall40Derived+Perf(2oneoffskip),collectcurrentlogshelperandqualify-current.py. Currenthelperwatch/collect/qualify scriptsdynamichead; doNOTuseoldclose/finishmetadata scripts. FinalcurrentproofonlyinPRbodyavoidsanothermetadataCIcycle. Afterfinalgreen, updatePRbodyaccuratefinalrunURLs/cases/head,maxUI,0threads; extractALL13Rulinglines(currentfinal-rulings.jsonold11mustrefresh),archivepublicledgerbeforefinalcommit thenverifyabsanddeleteONLYthisplanSDDworkspaceviaPowerShell;keep.testtmp/openWT. FinaluserPR3290opengreen,3Done,Superpowers0newfindings,37330+37360separateToDo; includeexhaustiveRulingsreason/costperSkill.

CI wave1 committed asa27f09d930f515405f28c63e3521697a835aa331:exact3testfiles; root independently verified74artifacts/sources+16XMLs, staticconservation, heldfolder/Previewoldnew andwrongReadernegative. ScopedSuperpowers /root/rereview3290_ci_wave approvedall3findings,0C/I/Minor/newbreakage. Portablearchive129members24cases(allhashesverified), invalidoutsideUIconfigcaptureexcluded. NewbacklogIDchecks3pass; taskowner34000.11/.18notesappendedCLI0/historypreserved. All3featuretasksstillDone13AC;37330+37360stillToDo. RootmetadatafinalpushandactualheadCIremain. No furtherlocalAppbatteryneeded; no merge. ALL13Rulinglinesmustreachfinalanswer,publicledgerpreservedbeforeonlythisplanSDDcleanup.
