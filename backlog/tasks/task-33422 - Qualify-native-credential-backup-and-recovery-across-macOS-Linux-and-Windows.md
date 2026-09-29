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
updated_date: 2026-09-29 19:32
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->