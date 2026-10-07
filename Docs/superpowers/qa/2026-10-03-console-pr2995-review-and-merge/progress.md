# SDD ledger — plan: Docs/superpowers/plans/2026-10-03-console-pr2995-review-and-merge.md

User authorized rebase, all review repairs and merge on 2026-10-03. Old draft final-review cap is superseded by this explicit new request.
Rebase complete: ce918c467898eb24feff216bced45b779bf310c6 → f843ca811f01da6c39d903b6cd7328d68d50416f; pinned dev 01a2020981c6197e5cd9945e5287567ad977edfe. Original commits retained in private recovery bundle.

## Preflight consistency scan

| Tasks | Producer/consumer | Finding |
|---|---|---|
| Task1 vs controller rebase | Rebased UI/runtime contracts feed ownership comparison | One documentation append conflict resolved; production changes merged automatically and need qualification. |
| Task1 vs external review | Committed repairs feed current-head bot review | Future feedback handled only when posted and verified. |
| Task1 internal | Mounted navigation regressions vs ownership fence | Authored edits require identity/revision checks; cursor navigation must not count as editing. |
| Task1 internal | Import deferral vs boot ratchets | Genuine deferred imports preserve behavior; limits and guards remain unchanged. |

Ruling: Reuse the existing integration task and approved ADRs for this follow-up — defects restore existing contracts — if wrong, task scope or an ADR will need revision.
Ruling: Repair the startup breach by deferring demonstrated boot cost — ADR-097 forbids silently raising the limit — if wrong, a feature path could need an additional lazy import correction.

Task1: pending; BASE f843ca811f01da6c39d903b6cd7328d68d50416f.

Heartbeat: finish-console-pr-2995-review-and-merge created ACTIVE for quiet delayed Qodo/CI follow-up; disable after confirmed merge and bookkeeping.
Rebase receipt confirms pinned dev ancestry and zero changed historical QA paths.

Task1 scope clarification: verified failures in additional covering boot/composer tests belong to the existing all-issues acceptance criterion; qualify immutable base before repairing stale expectations. All eleven derived checks completed with exit0 on the working repair tree.

Controller baseline replay: exact original21 nodes passed, exit0, 65.73s pytest / 72.26s process; production fingerprints unchanged during run. Evidence baseline21.json/log.
Task1 preliminary results: complete ownership78pass/1inherited strictXFAIL/5warnings; start/compaction/evidence/trace181pass; boot census1033/1033, unchanged limits. Additional covering fixture repairs remain pending.

Ruling: Isolate the older skill-await snapshot test at the hook-review seam while retaining the real prompt dispatcher, runtime request and original draft assertions — current hook admission deliberately refuses a changed captured stash and has separate unchanged-owner controls — if wrong, combined skill and hook interaction coverage could be incomplete and need another regression.

Task1 implementer DONE_WITH_CONCERNS: /root/pr2995_ownership_perf_fix, commit07e7c23cb58b39928af3393d3e447e172d804e86. Full report retains47 command receipts; source-only ten paths. Task review pending /root/pr2995_task1_review, immutable package review-f843ca811f..07e7c23cb5.diff.
Latest dev0001eba40419859ce39ed4952f0f8df7b40639bd differs pinned01a only by three Backlog tasks; no application/test/script delta.

Task 1: complete — spec compliant and quality approved by /root/pr2995_task1_review; no new Critical/Important finding. Minor inherited warnings and explicit hook seam qualification retained. Source07e maps byte-identically to rebased bc7c2dde9f. Missing genuine-child bridge regression separately qualified at bc7c2dde9f: one pass, no warnings. Initial setup-directory error retained separately.
Whole-branch review dispatched /root/pr2995_whole_branch_review on gpt-6-astra, immutable range0001eba404..bc7c2dde9f. Source/docs package95paths; historicalQA1524paths byte-unchanged versus ce918c4. Latest-dev Backlog IDs/readability guards bothexit0.

Final whole-branch review complete by /root/pr2995_whole_branch_review on bc7c2dde9f. No Critical/Important finding; two actionable Minor issues: prevalidation duplicate grant clearing and inaccurate child-tool guide. Task2 final fix wave BASEbc7c2dde9febfd93cbc3bc115c8cbc61c3ee752b, pending. User all-issues scope includes both.

Task2 implementer DONE: /root/pr2995_final_review_fix commit0d2e581ba96ef00adb02e2c800cd449582f8a629, exactparentbc7c2dde9f. Four genuine RED failures; complete shutdown owner36pass/no pytest warnings; fatal Ruff/formatter/diffcheckexit0. Two owned source/doc paths only. Scoped independent review pending /root/pr2995_final_scoped_review; package review-bc7c2dde9f..0d2e581ba9.diff.

Task 2: complete — scoped spec PASS/quality APPROVED by /root/pr2995_final_scoped_review, both findings addressed, no new defect. Latest dev81c7c94f48 rebase yields872efa5bdc; all78owned feature source/test blobs identical. Upstream backup/startup seam qualification running.

Latest-dev qualification at872efa5bdc: affected backup42pass/no warnings, startup/import13pass/3unchanged headroom warnings. Feature78blobidentity retained after upstream PR2994. All local source/review gates passed; external Qodo/current-head checks pending.

Publication successful: force-with-leasece918c4→7a15e3e9b4998dba8d8726bbdb722ff60a3df83d, verified latest dev81c7c94f48 before push. PR2995 ready, updated description. Initial current-head CI queued; Qodo pending. External snapshot files pr-*-initial-ready.json. No bypass; normal exact-head merge remains authorized after review/check gates.

Qodo manual current-head review requested via documented /agentic_review: https://github.com/rmusser01/tldw_chatbook/pull/2995#issuecomment-5976066978. No Qodo review posted at initialready snapshot; do not interpret absence as approval. CodeRabbit is informationally skipped on non-defaultdev base, not a failure. Await Qodo lifecycle output plus current-head CI.

Qodo acknowledged manual request5976066978 with eyes reaction by qodo-code-review[bot]. Current-head PR/UI FastLane + PerfGuard jobs remain queued; no Qodo findings/review yet. Quiet heartbeat ACTIVE verified directly in automation.toml after native view transient error. Do not disable until actual merge + task bookkeeping confirmed.

Continuation checkpoint: ready head7a15e3e9b4998dba8d8726bbdb722ff60a3df83d; dev81c7c94f48. Source/QA working tree clean. Task1 and Task2 COMPLETE, broad review and scoped re-review COMPLETE; do not redispatch them. Qodo summary5976093598 posted, review-status5976096132 is still Qodo is busy working; review comments/reviews remain empty. Resume by retrieving updated status5976096132 and all paginated comments/reviews, then verify/fix each actionable finding using fresh source-fix task and scoped review. Do not count summary/busy/eyes as completion. Current required legs/perf queued. Re-fetch latest dev before actual normal exact-head merge, qualify only any new affected seams; retain backup source identity receipts. Heartbeat ACTIVE and verified; no further user approval required for authorized review fixes/push/rebase/merge. Keep task34215.2 InProgress until actual merge gates/AC8 complete.

Heartbeat2026-10-04T03:22:15: Qodo current-head reviewCOMMENTED5404068729 on7a15e3e9b4. Eight findings posted in summary5976126336, seven inline4175971051/55/59/62/66/69/74; no inline for#8. Qodo initial snapshot retained. PerfGuard now running; PR/UIFastLane queued. Runtime-only read-only triage /root/pr2995_qodo_runtime_triage active; Task3 source fixes2/3/4/6 BASE7a15e3e9b4998dba8d8726bbdb722ff60a3df83d pending.
Ruling: Accepted or uncertain attempts stay charged and require root review settlement — ADR-211 explicitly permits refunds only before acceptance, so Qodo5 proposed accepted rollback refund is rejected — if wrong, conservative charging could require a separate approved reconciliation policy.

Runtime triage DONE /root/pr2995_qodo_runtime_triage:1/5/8 confirmed; bare admission offload violates loop-owned source/target cutoff. Owned conversation-commit task must shield/drain before settlement and capacity release.
Ruling: Refuse SQLite writer contention with a scoped zero-timeout acceptance connection while retaining the no-await cutoff — bare offloading loses the ADR-211 source/manual ownership fence — if wrong, cold connection or disk I/O latency may still need a separately reviewed admission architecture.
Ruling: Log only fixed phase and exception type for unexpected start failures — raw exception/traceback can contain prompt or credentials despite type-only message formatting — if wrong, diagnostics may need additional explicitly safe fields.
Task4 planned, pending Task3 sourcecommit; do not dispatch concurrent source workers.

Current-head7a15e3e9b4 PerfGuard SUCCESS in both observed runs; PR/UI FastLane running. Task4 brief clarified triage obligation: failed durable review settlement must keep the affected runtime allowance fail-closed, including siblings, without poisoning replacement-owner authority. No new charging/retry policy.

Latest dev advanced to225fbeba0ac1832921df95d20970a6d2b826a515 viaPR2986 (six changed paths; prompt_queue UI removes unreachable dispatch-recovery shelf branch, tests/docs). Read-only compare retained; defer Git fetch/rebase until sequential source fixes/reviews complete. Then preserve owned source identities and qualify affected queue/dispatch UI owners before publication.

Initial-head7a PRFastLane failed: four Tests/UI/test_console_runtime_ownership.py nodes (revisit callbacks, superseded screen detach, headless delivery poll, unchanged native acceptance timeout);142pass/1inheritedXFAIL/5warnings. Direct completed-job log retainedpr-initial-fastlane-job.log; gh run log-failed was unavailable until aggregate completion. Read-only triage /root/pr2995_ci_ownership_triage active; no concurrent source worker/tests. Targeted source/test repair to plan after verified diagnosis; no passing-current-CI claim.

Task3 implementer DONE /root/pr2995_qodo_surface_fix commitfc1627e99b8722a0a69812530d950410e8356ba0, exactparent7a15e3e9b4. Eightownedpaths only; complete affected owners194pass+composer36pass, no pytest warning summary; correctedRED6fail/5pass and finalmountedBASEcomposer3expectedfails retained. FatalRuff/newfileformat/BASEratchets/whitespace/blobfingerprints exit0. Task3 independent scope review pending /root/pr2995_task3_review; package review-7a15e3e9b4..fc1627e99b.diff (39600bytes). Task4 remains undispatched until this gate passes.

Task 3: complete (commits7a15e3e9b4..fc1627e99b, scoped spec compliant/qualityapproved by /root/pr2995_task3_review; no findings). Reviewer cross-task qualification limits retained; runtime/CI/startup gates remain open. Task4 BASEfc1627e99b8722a0a69812530d950410e8356ba0. Combined startup ratchets moved to controller final integration after sequential CI repair to qualify the actual final tree once.

CI triage DONE /root/pr2995_ci_ownership_triage: exactCImerge112298tree equals7a; three lifecycle calls faster than original local pass, slow-runner explanation unproved. Leading startup/harness competing-screen claim hypothesis remains unproved, native63.5s durable diagnostic lacks attempt provenance. Task5 planned after Task4 review: four-node diagnosis/deterministic cause before correction, preserve original assertions then original four-file CI batch; no concurrent tests/source.
Task4 source worker /root/pr2995_qodo_runtime_fix running BASEfc1627e99b; next independent concurrency review required.

Task4 initial RED12fail/1pass retained; firstfocusedGREEN12pass/1failure corrected empty-assistant-placeholder oracle. Complete coveringrun221pass/18fail under diagnosis: healthy receipted work must not be transiently restricted; symlink-file tests violate native privacy boundary; final source getters need owned native connection scope. Covering manual commit tests fail at hook/profile admission before protected path; immutable-base attribution in progress.
Ruling: Repair covering profile fixtures only after immutable-base reproduction, using the existing private-profile helper and preserving original test bodies — the authorized baseline repairs and actual commit controls require reaching the protected path — if wrong, profile and hook lifecycle interaction may require another regression.

Task4 source-only commitd3b2ade44b49afc96f78c592ac271e3231472b27, parentfc1627e99b. Worker reports final260distinct affected cases pass/no warnings; childscopeBASE15fail/6pass/1teardownerror repaired using existing private helper, original sync bodies retained; finalchild21+DB29pass. Same-owner recovery must retain unresolved fallback; manual independent-root controls passed. Exact final report/fingerprints pending before independent review dispatch; do not markTask4complete or dispatchTask5yet. Root plan/Backlog metadata remain trackeddirty.

Task4 independent review /root/pr2995_task4_review: spec issues/qualityNeedsFixes. ImportantI1 reproduced on real file: older recovery postcommit foreign-owner sweep deletes newer accepted uncertainty restriction; check_active changes settlement_unconfirmed to active while charge1/attemptaccepted. Review task-4-review.md and task-4-review-recovery-race.json retained. Task4 fix round1/5 pending originalworker; FIX_BASEd3b2ade44b49afc96f78c592ac271e3231472b27. Task5 remains gated. Git/PR rechecked: locald3b2ade, remote7a15e3e9b4 OPEN/dev, rootplan+Backlogonly dirty/indexempty.

Task4 fix round1 sourcec0b251a71422536a3da2be421dd60f79e2cb230a, exactparentd3b2ade44b/twoDBsource-testpaths/indexempty. Recovery snapshots foreign entries inside durable transaction and conditionally retires same entry tokens after commit; later owner/new or refreshed entries survive, no registry lock over SQLite I/O. RED2 expected bypasses; completeDB32+focused8=40pass/no warnings; finaloverlap3pass/no warnings; FIX_BASE and originalBASE committedformat/fatalRuff/whitespace pass. Report appendix/commit fingerprints read. Scoped re-review /root/pr2995_task4_fix1_review running, package review-d3b2ade44b..c0b251a714.diff12674bytes. Task5 remains gated. Fresh paginated PR snapshots show same8Qodo findings/7threads/onecompletedreview; no new actionable feedback.

Task 4: fix round 1/5 (1 addressed, 0 open — I1 late recovery cleanup fenced by captured entry identity; commitsd3b2ade44b..c0b251a714). Independent scoped /root/pr2995_task4_fix1_review specPASS/qualityAPPROVED/no new breakage.
Task 4: complete (commitsfc1627e99b..c0b251a714, initial task review plus scoped fix1 clean). Review current40+final3/hashes/static verified without suite reruns; combined startup/external gates remain controller-owned.
Task5 BASEc0b251a71422536a3da2be421dd60f79e2cb230a pending fresh source worker, original4nodes→causaldiagnosis→single originalfourfilebatch qualifies completeownership. No parallel source/tests.

Task5 source dispatched /root/pr2995_ci_ownership_fix on gpt-6.1-sol/high/forknone, BASEc0b251a714. Complete CI batch once qualifies ownership; original four exact failures precede correction, no silent slow-CI assumption/timeout increase. Worker owns only diagnosed source/tests, rootmetadata excluded.

Task5 first failure reproduced atBASEc0b251a714: startupalreadyfinished withConsoleA, then harnessmanuallypushesConsoleB yieldingdefault/A/B. Leave dismissesB/gen2; retainedA/gen1cannotreattach andnotifyhookNone. This differs from earlier unproved late-startup theft hypothesis. Worker still diagnosing othernodes/nativeprovenance and deterministiccanonicalcontrols before correction. No production defect conclusion yet.

Task5 originalfour diagnostic:3 lifecyclefailures/nativeunchangedpass76.40s. Nativetrace reached actualreadybarrier and bothfences; its earlierCI timeoutcause remains separately unproved. Worker proposed ownership-file-local factory settinginitialpushedbeforemanualmount; realstartup factory retained for canonicalcontrols. Scope authorized; no production change warranted by reproducedcause.
Ruling: Repair the three reproduced ownership failures through canonical manual-screen test setup and qualify real startup/public navigation separately — duplicate content Consoles violate the existing mounted harness contract and explain the stale claims — if wrong, a legitimate startup/navigation interaction may need another regression.

Task5 sourcee7cc5337617781e9a4b08b9e324297b537fcad07, parentc0b251a714, oneTests/UI ownershippath/indexempty. Canonicalmanualstartupfixture+twoactualcontrols; three lifecycleRED reproduced and corrected, nativeinitialdiagnosticpassed/separatehistoricaltimeoutcauseunproved. Singleoriginalfourfilebatch155pass/1inheritedstrictXFAIL/5inheritedwarnings343.84s, allsixnativeconsumptioncasespass; FDgrowth523 disclosed, no suppression/newmarkers. FinalsourceSHA25691c7698d342e412bf8796575c7a72b4a2fe42985bc8ff1169a98018694619521; fatalRuff/BASEcommittedformat/whitespacepass. Task5reportread, independent /root/pr2995_task5_review pending, package25308bytes. Metadata lesson/plan/task excluded; latestdev rebase gated on review.

Task 5: complete (commitsc0b251a714..e7cc533761, independent speccompliant/qualityApproved by /root/pr2995_task5_review). No Critical/Important finding. Nativehistoricaltimeout remains root current-head CI qualification; inheritedfivewarnings+misnameddisposablediagnosticfield null are recordedMinor observations, no production correction or generalleakclaim. Explicit task5-format-proof.json closes emptyformatterlog provenancegap exit0/stablesource.
Latestdev nowca2992cb10b24307fbae050643472ffb0a4388e7 (GitHubverified; sharedorigin/dev updated by externalfetch). Deltafrom81c:1863paths;25featureintersections. Source/fixture overlap19ASTformat-only plus5canonical realchanges: Chacha4CRUDwriters nowimmediate, queue-recoverydeadbranch removal, newegresstests, conftestprofileenrollment. Anthropicsettingsauth additions and library/security/module extractions outside feature overlap need concrete seam qualification. Rootstructuralcompare+canonicaldiffs retained. Finalrebase now separateTask6 to preserve allreviewedfeatureandlatestdev contracts; source implementation remains delegated.
Ruling: Preserve upstream formatting and qualify owned Python ASTs alongside exact non-overlap and historical QA blobs — byte identity cannot survive the formatter-heavy dev merge — if wrong, a source-text-dependent guard may need additional focused repair.

Task6 preflight: Task5 reviewed ownership/source feeds rebase; no new runtime boundary. Task1/3/4 startup and source-text guards consume reformatted dev; exact AST and static/boot qualification are required. Task6 retention proof covers non-overlapupstream/historicalQA; real5overlap changes must be retained and checked. No fullsuite/guardbypass/retrypolicy allowed. Root QA exporter now includes top-levelJUnit and onlyexplicit inspected Task5 evidence regularfiles, no recursiveprofileexport. Initialaudit stopped beforecopy; all396 sk- matcheswere task-pathsuffixes; properboundary corrected and knowncredentialcomparisonretained. Final351files9258114bytes audit0/privateprofileFalse. Metadata archival commit before Task6hand-off pending.
