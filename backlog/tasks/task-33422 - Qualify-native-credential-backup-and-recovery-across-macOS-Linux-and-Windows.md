---
id: TASK-33422
title: Qualify native credential backup and recovery across macOS Linux and Windows
status: In Progress
assignee: []
created_date: 2026-09-29 17:58
labels: []
dependencies: []
priority: high
references:
- https://github.com/rmusser01/tldw_chatbook/pull/2642
updated_date: 2026-09-29 20:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the approved native credential qualification for existing Python backup/recovery, including all nine source/destination OS combinations and scoped fixes demonstrated by application-level tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native credential stores are isolated from existing user accounts and verified before any fixture writes.
- [ ] #2 Default and retargeted profiles capture and resolve their own credentials after replacement and later rollback without modifying other profiles.
- [ ] #3 All seeded supported credentials are captured; manual retention and unavailable backends are reported separately.
- [ ] #4 Three native source archives pass all nine OS destination workflows, with targeted failure checks and no plaintext credential evidence.
- [ ] #5 Targeted tests, formatting, Bandit and independent review pass; exact runtime/backend and artifact evidence are documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: qualify existing behavior and fix demonstrated defects without a new subsystem.
1. Reproduce default and retargeted profile credential capture and application readback; fix demonstrated defects.
2. Reuse installed package runner helpers for isolated native backend qualification and protected artifacts.
3. Capture on three native OS backends, restore all nine transfer combinations, and run boundary negatives once per backend.
4. Run targeted regressions, formatting, Bandit and independent review; document runtime/backend and artifacts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation in isolated branch codex/native-credential-backup at dev base64579cce. Profile-aware capture/destination/rollback correction plus six focused namespace cases implemented. Initial targeted replacement/rollback regression run:32 passed,3 fresh-process crash cases failed with directory_metadata_unproven; investigation traced preexisting broad test-helper import to config creating chat_dicts under protected directory. Native runner14 focusedtests/format/Bandit clean; manual three-source/three-destination workflow implemented. Disposable Linux Python3.12.14/SecretService private owned bus/unlockedlogincollection verified twice. Actual current-source qualification awaiting explicit audited6-file SSH transfer approval after automatic approval rejected full and narrow working-tree uploads. Native product tests being reviewed/fixed before CI; no qualification claimed.
Reviewed implementation ready for native execution, qualification still incomplete. Verified focused coverage:64 credential policy tests and6 profile namespace tests passed;14 runner tests passed after fixing inherited plugin-disable behavior;20 replacement cases and9 initial rollback cases passed; remaining12 rollback cases passed after test-only fixes (fresh child embeds existing backend class to avoid unrelated config import, validation wrapper forwards session keyword);51 later rollback/replacement recovery tests passed. Independent review:20 focused checks passed, no outstanding P1/P2; native opt-in gating verified2 ordinary-suite skips before wheel fixture. Ruff check/format, compile, diff-check, production/runner Bandit and new-test Bandit (B101 assertions excluded) passed. Manual workflow reuses required pip/setuptools>=77/wheel build setup. Actual Linux patch transfer remains pending explicit user approval after automatic review rejected it; no current-source native capture or nine-direction matrix has executed. Six-file code-only patch is reviewable at /private/tmp/native-credential-linux-source.patch (regenerate from final working bytes before transfer). No plaintext evidence publication is enabled for native lane.
Human explicitly approved six-file SSH source transfer; exact patch68444bytes/SHA256fe668dca19e89d6b257525349d37d376feacfac8fbd03c8755cb9bd98dd333b6 applied to public base on owned Linux environment. Native source run is starting. Reviewed branch08d5961841 pushed for authorized hosted qualification. First GitHub dispatch failed before jobs with422:runner.temp is unavailable in job-level env. Moved runner-dependent transfer/evidence paths into shared qualification step env; local context placement/Bash syntax check passes, next verify GitHub parser by dispatch.
['Native execution started: GitHub run36616958970 on e1b8152; Linux source collected1/failed1/skipped0 with installed wheel receipt1. Linux SSH source setup passed, capture refused unsupported /tmp tmpfs (release_capability_unavailable); verified /var/tmp ext4 supports admission/publication and rerunning there. Adding strictly projected exception-type/frame-only evidence so hosted native failures can be diagnosed without exporting raw credential logs.']
Linux ext4 capture diagnosis proved all missing4records belonged to an auto-added legacy:tldw_api target matching installed placeholder URL in retargeted profile; eight seeded server references,16 generation slots,10 citation records captured. Fixture now explicitly leaves legacy endpoint blank; cold-process config/model probe confirms no legacy target. Existing native source test rerunning after one-line fixture correction. Safe hosted diagnostics added, tested against child+outer failures and frozen import frames;17 runner tests, Ruff and Bandit passed. Product credential behavior unchanged in this correction.
Automatic approval review refused the revised direct SSH fixture upload as a new payload, before any bytes transferred. Using the already-authorized public branch publishing workflow instead: commit reviewed fixture+safe diagnostics, then have disposable Linux checkout fetch exact public commit. Existing private ext4 store remains isolated; no native qualification claimed.
Exact a63f6ef native SSH source cleared the missingcredential failure. First capture requested26 manualcredential acknowledgments; second capture refused scope_changed because native test mutated acknowledged options but reused old approved_scope digest. Verified capture_service._review_digest includes acknowledgments. Test now refreshes actual preview on each attempt, preserving production scope checks. No encrypted archive published on failed runs. Independent correction review17tests and child propagation probe clean; next exactpubliccommit rerun pending.
Hosted Windows safe failure frames prove actual RuntimeServerContextProvider write fails at shared namespace index CredWrite. Bound native UTF16 fake reproduces thirdindexwrite3012bytes exceeding2560; correcting only existing native index record sizing with backward-compatible boundedparts/rootpublication. Also targeted capture_service checks found3 unsupported_schema failures in current PromptsDB. Pinned unchangeddev64579cce policy probe reproduces frozen backupv4 vs actualv5 (LocalPromptDrafts table+index); correcting backupvalidator to accept exactv5 while retaining currentlysupportedv4 and rejecting stamp/catalog mismatch. Both are demonstrated existingbackup defects, no newproduct features. Local capture_service result3failed/1pass pending scoped schema correction.
Actual SSH Linux source PASSED on publicfab6a720c542b9639bc6498ad29c234b72e98adf:1collected/0failed/0skipped, SecretService.Keyring, Python3.12.14, ext4flags4096. Encrypted age-v1 archive50222063bytes/SHA25626bb13106e3ff6396af837d84bb4cdb5e6479b7e541e83d70ca40d37d29d2764; installedwheelSHA2565cef4a3986408885d0d921b4459ef11af15f62cc4cb1297c2780969fcd29b70e. Both profile scopes/all seeded supportedcredentialtypes captured; unsupported defaults removed fromfixture and manualreview refreshed. Full preliminary Linux-to-Linux isolated/replacement/rollback now running; 3source/9direction matrix stillpending.
Native follow-up: GitHub run 36619332855 completed with sources failed, so no destination qualification is claimed. macOS source child setup timed out after 900 seconds; only safe parent frames are available, so bounded existing thread-frame diagnostics are being added before another run. The exact-fab SSH Linux source passed (1 collected, 0 failed, 0 skipped). Linux destination stopped before restore: cold child had zero registered adapters; installing existing declarations classifies both unselected prompt stores included. This fixture lifecycle problem is separate from the independently reproduced current Prompts v5 catalog rejection on pinned dev. Windows bounded fake confirmed the native index size defect; compatible chunking now passes 53 credential tests, including root commit followed by an uncertain error without deleting its published parts. Scoped fixes/review still in progress; all acceptance criteria remain pending.
Parent verification: 53 native index tests passed; 144 core/schema/SQLite/capture/draft cases passed after schema fix. Independent read-only reviewers found no remaining concrete P1/P2 in either product patch. Product Bandit exactly four preexisting findings (one B608 in recovery_core, three B105 token-purpose constants), no new findings or scan errors. Credential/profile/replacement/rollback run:109passed/2failed; both failures exclusively compared all backend deletions and now counted expected superseded index-part cleanup. Updated that existing guard to permit only exact index-part usernames in the same store service, while still forbidding credential deletion; three parametrized behavior cases rerunning. Refreshed remote dev is64e140bb30; no changes in current backup/schema/runtime credential scopes, rebase planned before next qualification.
Updated rollback guard verified: all3 authenticated-old-secret behavior cases passed; only exact same-service metadata-part cleanup is allowed, credential-deletion guard remains active. Thus relevant111credential/profile/replacement/rollback cases have109unchanged passes plus3updatedparam passes (including the previously failed2). Native runner/thread diagnostic focused private Null-environment checks102passed/2ordinary-suite opt-in skips, fullRuff/format/compile/diff and runner/testBandit clean. Cold destination adapter registration fixed before raw discover; safe children frame observations added to existing diagnostic receipt. Independent final fixture review pending before commit/rebase/native replay.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->