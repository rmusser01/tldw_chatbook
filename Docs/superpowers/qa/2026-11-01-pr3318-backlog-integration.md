# PR3318 backlog integration evidence

Source00272b4d525fd14d77326ace78d20bd2be7a02d8 (29 commits) rebased onto confirmed dev945d89184bd0824e97a79c7243fa7a07c0fdc8c6 as7662706edf687fb7b1524427caa8986a748a7ab8, followed by the independently reviewed ownership repairs below. Non-draft PR only; hosted CI and size limits explicitly waived by the user. No release/main work.

## Repairs

Temporary chats default to Capture Off; explicit Capture On still pauses for Save & Send or Send without capture. Current-dev trace Cancel, physical-call refusal evidence, received-input snapshots and native retirement remain preserved.

Card settlement now compares the original session incarnation/object and draft revision, not text equality. Successful sends clear only their original draft; later identical or empty authored input survives. Cancel uses the store's exact restoration receipt and returns original attachments only while that draft owner remains current; otherwise explicit recovery remains available. Voice handoff records the original empty draft revision without granting composer-send authority. The controller outcome is retained synchronously before awaited UI cleanup, so interruption after an accepted send cannot resurrect it on the unsent shelf. Existing ADR097 and ADR230 govern these changes; no new architecture decision.

## Focused execution

Receipts are in the sibling pr3315-resync/r-* directories, each with command, XML, logs and before/after hashes. Every listed run confirmed unchanged source hashes during execution. Only focused tests ran, with mounted UI processes serialized.

- r-b4c5a7a5: Cancel ownership regressions RED (later empty text overwritten / identical revision replaced); two initial send waits did not reach the provider and are not causal evidence.
- r-1052ccc8: held send reached the real provider, then erased a newer identical draft (RED); unchanged send control passed.
- r-7840a730:50 passed/1 failed across ownership, capture admission, unsent, voice and all9 trace-recovery cases. The sole unchanged default-temporary-send control hit its20-second provider wait.
- r-dcab56ea: that unchanged default-temporary-send control passed; both new answered-send/UI-interruption regressions were RED because the completed turn returned to recovery.
- r-3cdee70d:7 passed/1 failed. Production settled accepted recovery and stored draft, but the new test cancelled every refresh, including the independent recovery refresh. The test was corrected to interrupt once and delegate subsequent refreshes to the real callback; all assertions retained.
- r-dd12995c:5 passed,0 failures/errors/skips: both interruption cases, both newer-identical send actions, and restored native not-saved status assertions.
- r-d4b4b099:151 passed/2 failed across controller exchanges, ephemeral persistence, staged-evidence dispatch and census contracts. The two exchange tests lacked bootstrap_profile, so real config admission refused the redirected source before the asserted behavior. No production guard was changed.
- r-7c8cd9aa:55 passed,0 failures/errors/skips after adding the two existing private-profile markers: entire exchanges file plus mounted Save Chat/status copy and menu contracts. Assertion strength is unchanged; the fixture restores profile bytes between tests.

Earlier first-fix receipts include a bounded provider-entry wait failure and two duplicate-workspace fixture failures; these were diagnosed and repaired before the final runs. This evidence is not a claim of one clean aggregate sweep.

## Static checks and review

Diagnostic inventory verified716 owners,1065 TASK492 calls,56 TASK31551 calls,7791 TASK494 calls,16 sink files; later changes add no diagnostic sink/call. Census checker passes1285 entries, preserving the exact1276-entry dev prefix plus6 source and3 repair selections. Scoped Ruff on31 changed Python files:281 base findings,281 current,0 introduced. git diff --check passes. No broad size-policy changes were made under the waiver.

Independent reviewer /root/pr3318_integration_review, using superpowers:requesting-code-review, approved the integrated source and final repairs with no Critical/Important defects. The only remaining review note was a fixture docstring typo, corrected after testing; no executable statement changed. Separate read-only ownership investigation confirmed the promotion/native lifetime limitations below already exist on dev.

## Qualification limits

Hosted CI, full-suite/platform qualification and size limits are waived or out of scope. Existing promotion-thread lifetime/postcommit failures, saved-identity rollback, general recovery claim filtering, and accepted-trace/context-compaction draft ownership are unchanged baseline behavior. Compaction attachment loss and post-save Archive refusal remain separately tracked (TASK37810/TASK37741); capture-copy/guidance/first-frame/narrow-strip follow-ups remain open. No claim is made that this PR repairs those paths.
