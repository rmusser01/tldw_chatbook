# Backup UAT remediation and repeat — 2026-09-13

**Acceptance is still in progress.** This report follows the [failed original walkthrough](backup-uat-2026-09-13.md). It does not replace or erase those results. Work remains confined to local backup and recovery on [PR #2642](https://github.com/rmusser01/tldw_chatbook/pull/2642), TASK-32562.

## Method

Persona acceptance uses an immutable installed wheel, the ordinary `python -P -m tldw_chatbook` entry point, and keyboard events in real tmux terminals. Fresh first-time profiles traverse the actual six-step wizard. No Textual Pilot, widget-value assignment, test-mode environment, pre-completed onboarding, or product patch is used for a claimed keyboard pass. Synthetic homes and credentials keep these tests separate from personal data.

Terminal text and ANSI frames, keyboard records, archive hashes, and read-only database/file checks are the evidence. Desktop pixels and screen-reader accessibility have not been evaluated. Native automated macOS, SSH Linux, and GitHub Actions Windows tests are separate from the keyboard persona walkthroughs. Diagnostic and fixture-assisted results are identified explicitly.

## Corrections made

| Original or repeated finding | Correction and verification |
| --- | --- |
| Ordinary `config.toml.lock` prevented Complete classification | Recognize only the exact installed private empty lock; unsafe or unrelated files still refuse. Native inventory and actual default-profile review checks cover this. |
| Live Console, Library, and first-launch backup/restore could not settle | Correct the installed retained Library state contract and retire newly acquired native file-backed caches at the measured finite Console, Library, and Home worker boundaries. Borrowed, custom, in-memory, foreign-thread, active-operation, and transaction lifetimes remain protected. Installed capture verifies archived note content and subsequent live writes; installed first-launch restore verifies recovered note content. |
| Ordinary polling could crash during live registry publication | Read the existing native registry under its shared lock. A live publisher completes before inspection; abandoned durable intent still refuses. Bootstrap does not create or repair files. Deterministic native concurrency and unsafe-lock tests cover both paths. |
| Restore required internal hashed directory entries | Request one local base and display name per profile, plus independent external locations, and derive installed owner paths. Preserve shared stores and unused optional selectors; reject invalid private parents before confirmation. |
| F9 and generic recovery feedback were misleading | Document F4 Settings and the command palette; show bounded corrective failure guidance, reviewed Partial coverage, and the contents/counts of nonempty extraction groups. |
| Larger recovery records could not be reopened | Bound recovery descriptors/journals separately from the unchanged bootstrap limit, read in chunks of at most 64 KiB, and refuse excessive records before publication. Tests cover short reads, exact limits, growth, reopening, and actual 1,800-file publication. |
| Opening a restored app reported failure after ordinary Quit | Avoid defaults-only writes during shutdown and first Console mount for the exact unchanged process-bound recovered configuration. The fresh keyboard restore/Open/normal Quit repeat now passes on tree `580c657bba5e05e185f8f62243c17f4c16eb28e6`, with verified receipt, unchanged config/note and all installed hashes. A separate legitimate-settings-edit continuity issue remains under correction. |
| Replacement of the default profile rejects internal inspection storage | Recognize only the untouched synthetic config container around exact excluded default recovery stores. Bind the actual application namespaces without granting authority over the control parent. Native replacement, untouched abort, later rollback, new-copy inspection, and namespace checks passed in 328.26 seconds; 31 focused regressions passed independently. Keyboard repeat remains pending. |
| Saved, closed Library note falsely prevents recovery handoff | Recognize the installed Library note work pane and the current prompt mutation field. Dirty, saving, conflicting and unknown editor states still refuse. Independent tests: 83 passed; installed first-note capture and guarded handoff: two passed. The latter checks shutdown/request creation and does not claim an external CLI restart. |
| Required safety-file selection needs excessive keyboard navigation | An unavailable dependency review focuses the explicit selection button; selecting the required files focuses Review restore. Selection still invalidates the plan and requires another review and confirmation. Two regression cases failed before the correction and passed afterward; both also passed independently. |
| Bound profiles refuse installed preferences writers; duplicate sidebar methods bypass the guarded writer | Exact installed emoji, sidebar and runtime policy operations use the existing configuration lease and hold their registry guard until native IO retires. Custom paths and invalid-parent refusals remain intact. Nine later duplicate sidebar definitions were removed. Independent checks: 85 passed; an installed Console capture, archive readback and resumed-write regression passed in 47.96 seconds. Repeated capture of a newly created sibling still needs the separate correction described below. |

## Power-user keyboard results

These results use product `c5c05181d834d4b7596107a154216a330af8261d`, installed at `/private/tmp/chatbook-backup-uat-repeat-20260913-07b3zfn3`. The later quit correction is being repeated separately on a fresh build.

| Action | Result and evidence |
| --- | --- |
| Missing or mismatched encryption passwords | Pass: explicit refusal before capture. |
| Two-profile encrypted backup with credentials and external research | Pass after correcting a fixture that had never initialized its second profile through normal startup. The original fixture refusal is retained. Null keyring scopes were explicitly reviewed and omitted. |
| Wrong archive password, then correct password | Pass: wrong password refused; correct inspection verified 3,868 files and 54,226,805 payload bytes. |
| Restore two profiles and the shared research folder | Pass: restoration validated; all 1,800 research files match their source SHA-256 values, and source files remain unchanged. |
| Recover a nonempty Notes/chats group | Pass: five extracted files, 21,667,840 bytes, each matching the verified manifest. |
| Optional credential inclusion | Pass for the synthetic provider credential retained inside the authenticated encrypted archive. Restored configurations are sanitized, the encrypted recovery archive is retained, and 42 unavailable keyring scopes were explicitly acknowledged. This does not claim recovery of omitted keyring credentials. |
| Open restored profile and quit normally | Failed on this revision: the child mounted Settings, but shutdown rewrote the config and invalidated its fingerprint. The correction has independent review and regression coverage; fresh keyboard repeat is pending. |
| Cancel an active capture | Pass: Running → Cancellation requested → Cancelled. No output archive; 1,802 source/config file hashes unchanged. |

Encrypted archive SHA-256: `86324ee519464336a064867bd37d6bc499fcb88d20e8290df31b5a947cfeee9e`.

## First-time keyboard repeats

The `53e3c8d0359f547b3ce1e5b4d5fc827fe67df308` and `04029dd08beb82a56dde576332cc8a57da7f63d3` repeats completed real onboarding, saved a note, found backup through F4/command palette, and obtained Complete review. Capture still failed. Those failures exposed the retained Library and Home worker cases; they are not counted as acceptance passes.

A fresh immutable build of `a743fd473818481797e5133f05daa9e85c749cd2` repeated onboarding → saved note → Complete review, but capture still failed. A separate metadata-only diagnostic identified the remaining lease: Home active-work refresh → unread notification count → the installed notification database. The exact finite callback correction subsequently passed the installed full first-note regression. Historical failure evidence remains at `/private/tmp/chatbook-backup-uat-repeat-20260913-ydouf8av`.

The fresh merged repeat uses Git tree `f69a11ecb0720db043d9e89616ff46dc789f8bad`, wheel SHA-256 `2cb2fff7cf0612c6e8f29593d288dc93db445a093d14969abe0f185fe037b1ec`, and evidence root `/private/tmp/chatbook-backup-uat-repeat-20260913-kk6_dews`. This is a staged tree, not a final commit. All 5,347 installed file hashes were checked against its artifact receipt.

| Fresh keyboard journey | Result |
| --- | --- |
| Six-step onboarding → first saved note → Complete review → capture | Pass: archive verified, with the exact saved note checked read-only in both source and archive. Archive SHA-256 `cf386124bb7c0daf46ffe0a127454713bde3d83d677a88dc6cf289def7406114`. |
| Resume ordinary note editing after capture | Pass: keyboard edit and Escape autosave verified in the live database. |
| New installation → Welcome Restore → inspect → one base/name → review → confirm | Pass: restoration validated; archived note payload matches the restored database. An overlapping parent was refused first, then an independent private parent succeeded. |
| Open restored app → ordinary Quit | Failed: child mounted Console, but the parent reported `backup_operation_failed` and marked the profile `recovery_required`. The exact cause is under investigation; an Evals warning alone is not treated as the cause. |

Disk exhaustion interrupted the terminal recorder before several keyboard actions. Those actions were not sent; affected steps resumed after reclaiming reproducible, task-owned build inputs. These interruptions are not counted as product failures or acceptance passes.

The subsequent Console correction passed a new independent restore and actual keyboard Open/Quit journey on tree `580c657bba5e05e185f8f62243c17f4c16eb28e6`. The parent returned a successful Needs setup result and kept the profile restoration_validated; the native receipt, exact note/config bytes and 5,347 installed hashes verified after exit. See [the curated acceptance report](backup-uat-remediation-evidence-20260913/newcomer-c8-open-quit-report.md). The f69 failure above remains historical evidence.

## Additional acceptance findings under correction

- Default-profile replacement's credential review and actual Abort passed, with both original notes unchanged. After acknowledging all 21 unavailable synthetic keyring scopes, replacement failed before safety-archive creation. Read-only reconstruction proved `unknown_producer_dependency`: the safety copy contains the notes/chat database but omits a preserved persona-artwork dependency. A second actual Abort succeeded; database and configuration hashes remained identical to the first Abort. The correction will require explicit safety-file selection during review, before staging or pending recovery state.
- Actual mounted rail and theme edits persist, but then invalidate the restored profile's config fingerprint and recovery-catalog open result. Two focused real-app regressions reproduce this separately from the now-passing no-edit Quit. The repair must preserve namespace roots and activation restrictions while recognizing successful installed configuration writes; external modifications must not gain enrollment.

The safety-copy dependency correction now requires an explicit **Select required safety-copy files** action and another review. It passed independent review (nine focused tests) and an installed macOS default-profile replacement and later rollback in 328.90 seconds. The keyboard repeat on immutable tree `fe62a5aa319969ae2f21403ab3ca61059a8e3707` verified that explicit selection makes the plan available. Confirmation then failed with `backup_operation_failed` after staging and before journal or safety-copy creation. Read-only diagnosis proved the absent incoming UI preferences file cannot acquire publication scope without covering the protected config parent. The original saved note/database and all installed artifact hashes remain unchanged. See the [preserved failure and provenance](backup-uat-remediation-evidence-20260913/replacement-fe62-report.md). This attempt is not a replacement pass.

That keyboard repeat exposed a further problem before it could start: after the earlier successful untouched Abort, ordinary startup of the original default profile fails with `storage_scope_not_enrolled`. Its saved file-only binding does not cover the configuration writer's companion files. Original database/configuration bytes still match the post-Abort receipt. The native replacement/rollback test stays in recovery mode and did not exercise this normal restart, so its pass does not resolve the new failure.

A repair for exact configuration companion operations and verification of the existing bound user-data directory passed independent review and 24 tests. It holds the existing selector lease and registry lock for the native operation without enrolling the config parent or changing persisted fingerprints. Full startup then exposed the first binding's omission of ordinary temporary files in the user-data folder.

The first-binding correction proves and binds the existing installed data directory, with no control-parent coverage or automatic expansion of older bindings. It passed 75 focused tests and 76 independent checks, including shared-profile compatibility. Ordinary CLI construction, app mounting, saved-note readback and clean return now have explicit native checkpoints after Abort, replacement and later rollback. All three passed in the source-only workflow, but its first child used the actor-scaffold correction before the final shared-profile adjustment; a final immutable-artifact repeat remains required.

The separate runtime writer and duplicate-method corrections now have independent approval. The installed Console regression passed on wheel SHA-256 `a724c742380d09f82365acec09db0e5aef61e062e9fd4cf9b4d70e17fa355eac`; all four changed product files matched their reviewed hashes and the package fixture verified its 2,865 installed files after the test. This automated test is distinct from keyboard acceptance. A further native check reproduced a bound-profile failure: saving a previously absent runtime-policy file succeeds, ordinary capture registers it in a new namespace outside that profile's binding, and the next save refuses. Repeated-capture continuity and publication of absent config files remain unfinished.

The proposed fingerprint-only settings-continuity repair remains unapplied pending the specific approval requested after automatic review rejected that persistent-boundary change.

## Actual interruption and restart

A fresh independent profile on tree `580c657bba5e05e185f8f62243c17f4c16eb28e6` passed actual capture interruption → ordinary restart → keyboard retry. The verified retry archive contains all 1,800 research files and the exact saved note; original files and note rows were unchanged, and no incomplete archive was published. All 1,936 payloads and 5,347 installed file hashes were checked. The sole config change on restart was the proven ordinary first-Console default layout initialization. See the [acceptance report](backup-uat-remediation-evidence-20260913/power-interruption-fresh-report.md) for hashes and the earlier inconclusive attempt preserved separately.

## Native platform evidence

| Platform / product | Result |
| --- | --- |
| macOS, Library correction `04029dd08b` | Installed Library → F4 → Complete capture → archive readback → resumed write passed. Independent retained-state/lifecycle check: 102 passed. |
| macOS, Home/quit/registry corrections through `a743fd4738` | Installed first-launch restore passed; 23 profile-open/config-persistence checks passed; installed Console, Settings, and Library capture routes all passed. Independent registry review ran 38 passing tests. |
| Linux SSH, `04029dd08beb82a56dde576332cc8a57da7f63d3` | 113 tests passed, no skips, 227.28 seconds. Includes installed Console/Settings/Library capture and actual 1,800-file publication. Exact source patch receipt retained on the host. |
| Linux SSH, `a743fd473818481797e5133f05daa9e85c749cd2` | 114 tests passed, no skips, 338.79 seconds, including new Home, registry, first-launch restore, quit, worker lifetime, and native capture checks. These do not substitute for the failing full keyboard onboarding-to-note path. |
| Windows, `d0882f84b0` | Plain, encrypted, encrypted-credential, two-profile roundtrip, replacement, and later rollback selections passed. Support: 191 passed and two failed (live registry refusal and the 1,800-file test deadline). [Run evidence](https://github.com/rmusser01/tldw_chatbook/actions/runs/34785915153). |
| Windows, `ad88df3415049134bac230314ca244f48c6510d0` | Failed: 42 native filesystem checks passed; product checks had 258 passes and 11 failures. Failures comprise three mounted capture routes, one public-lock fixture, six profile-open cases and the actual 1,800-file publication deadline. Bounded diagnostics and a Windows ACL fixture correction have been reviewed; the new native Windows run remains pending. No current-tree Windows pass is claimed. [Run evidence](https://github.com/rmusser01/tldw_chatbook/actions/runs/34788128200). |
| Linux SSH, merged tree `f69a11ecb0720db043d9e89616ff46dc789f8bad` | Exact 16,248 source blobs verified. Lifetime/capture/open checks: 104 passed, no skips, 428.85 seconds. Schema/replacement-admission/crash checks: 139 passed, no skips, 143.95 seconds. Default-profile replacement/later rollback: failed at the fresh replacement subprocess's 150-second deadline; investigation remains open. |

Earlier Linux native capture, encrypted controls, replacement, and later rollback also passed on `c5c05181…`; the later rollback repeat used the committed test-only measured first-preview wait budget. These older results do not substitute for checking the new corrections.

## Remaining acceptance work

- Correct restored-profile continuity after ordinary settings writes.
- Correct and repeat default-profile replacement and later rollback, including normal startup after untouched Abort, replacement and rollback.
- Complete affected native Linux/Windows checks and publish final evidence with the PR.

The requester subsequently approved the two exact path-inventory comparison entries, the two-file Linux diagnostic transfer, and configuration-write binding continuity for both default and restored profiles. Their execution and verification are recorded below.

No merge has been performed. The requester's PR Change summary remains unchanged.

### Bound preferences capture correction

The three installed config siblings now retain their existing config exclusion scope during capture. Independent native testing passed all 15 cases in 83.28 seconds, including actual Complete backup followed by another owner write/readback for sidebar state, recent emoji and runtime policy. The shared-parent case drains a real writer under the other bound selector, then captures only the requested profile; registry and binding contents remain unchanged. The final strengthened scope-change assertion passed in 24.73 seconds. Existing capture/SQLite regressions passed 48 cases in 60.73 seconds.

The replacement/rollback correction now has independent approval. The committed originating plan, manifest and publication evidence establish which exact installed preferences file was originally absent, including when imported and local profile IDs differ. This permits later rollback to capture the current file in its safety copy and then remove it. Default-layout native lifecycle coverage passed 22 cases in 112.38 seconds, including real process interruption before/after activation-pair publication, both fresh-process recovery actions, foreign-intent refusal without changes, and ordinary writes by all three installed owners afterward. Independent review passed 51 checks across scope validation, explicit bootstrap roots, default interruption recovery, retirement and the corrected recovered-media fixture. No config-parent namespace was enrolled and no placeholder file was created.

A separate config-only isolated restore completed recovery but failed ordinary Open because its user-data child did not yet exist. The failure reproduced on the preceding source baseline. The correction returns that missing child to the existing admitted directory-creation path only when an existing bound directory already contains it and its parent is private. No root is added. Companion and actual isolated recovery/Open checks passed 29 cases in 17.54 seconds, including a fresh app writing and reading a native note; independent checks passed six cases. Ruff and Bandit reported no findings for this correction.

These source-native checks do not replace final installed keyboard or Windows/Linux repeats. A fresh installed keyboard walkthrough on immutable tree `336c272dab58d2116055d3f335437e5f46af60c7` is in progress; that artifact includes publication/recovery fixes but predates the config-only missing-child correction.

### Filename review correction

The fresh keyboard repeat entered `first-complete.zip`. Review showed Complete coverage, then Create failed with `invalid_backup_suffix` and generic guidance. The unchanged capture suffix predicate is now also checked before service preview and operation startup. The form and bounded error message name `.tldw-backup.zip` and the encrypted `.tldw-backup.zip.age` suffix. Eight regressions passed after reproducing the service/UI failures; independent review passed 13 checks, including valid plain/encrypted roundtrips and credential protection. Ruff and Bandit were clean. This adds no automatic filename rewriting or archive format change.

The immutable `336c272d` keyboard repeat succeeded after correcting the filename: all 135 payloads (26,860,280 bytes) and the saved note were verified in archive SHA-256 `e11915a78c0d0b6029de6300851d3bd5e49c21e28dd25374336fe62d2f5bd13c`. It then completed the actual Settings handoff to a fresh recovery process. All 5,347 installed file hashes remained unchanged. Replacement review is in progress.

The first immutable `336c272d` Abort attempt is **inconclusive**: UAT verification opened the live SQLite database in read-only mode after replacement reached its prepared state. SQLite can change its shared-memory sidecar even through that connection. The preserved artifact proof identified exactly that sidecar as changed and refused recovery; main database bytes remained unchanged. This must not be counted as a product Abort failure or used to relax recovery checks. A clean repeat will perform no SQLite opens while replacement is pending.

Fresh installed `336c272d` probes passed all three credential-review→Abort cases in 2.56 seconds: absent/present default config siblings with the actual null keyring, and a WAL-backed credential refusal preserving exact database/WAL/SHM/config bytes and native identities. No SQLite connection occurred while prepared. The contaminated fixture remains preserved; no product change was made for that refusal.

The latest immutable `ab24f6b0c2690f7705bdcddf46a941104541fe07` package separately passed installed checks for config-only missing-user-directory creation and both plain/credential-encrypted filename review refusals. All 5,347 installed file hashes remained unchanged; wheel SHA-256 `198a7494d194b553e1bba81d01adf401ae5e6590c93f9f310fc0659e489ad8bb`. These three checks use native child processes and the Textual test harness, not actual keyboard automation.

The clean immutable `ab24f6b0` keyboard repeat passed credential-review → Abort → normal startup → original-note readback → normal Quit. Abort reported success and removed the pending recovery entry. The original note appeared in Library and its exact title/body matched a byte-copied database snapshot; verification never opened the live SQLite database while prepared. See the [clean Abort receipt](backup-uat-remediation-evidence-20260913/replacement-ab24-clean-abort-reopen.json). Acknowledged replacement and later rollback remain in progress.

After that normal restart and Quit, the same keyboard session acknowledged all 21 synthetic keyring omissions and repeated replacement review. Review refused with `target_unverified`; no new replacement started. The existing default binding now has a stale configuration fingerprint and `activation: null`. This is a separate default-binding case from the pending restored-profile settings-continuity proposal, which currently handles only activated bindings. Independent read-only causal review is underway; the original saved note remains verified. See the [refusal and binding receipt](backup-uat-remediation-evidence-20260913/replacement-ab24-post-restart-binding.json).

Independent read-only review confirmed the retry cause: automatic normal Console startup adds only `console.rail_state` to the TOML, invalidating the unactivated binding fingerprint. The installed MCP recovery check then raises `mcp_recovery_binding_changed`; inventory cannot publish an included target config, so destination review refuses `target_unverified`. This is a configuration-write continuity defect, not an Abort failure. Two new native tests reproduce it for explicit and implicit configuration selectors (two expected failures in 2.11 seconds). Those RED tests remain with the unapplied continuity proposal; no product repair or authority record modification has been made.


### Approved configuration-write continuity repair

The installed configuration writer now refreshes only the fingerprint of the same verified existing profile after successful private publication. Both unactivated default profiles and restored profiles retain their exact storage roots, namespace membership and activation association. External preimage changes, substituted publication/authority identities, failed writes and interrupted writes cannot gain a refreshed binding. Native reader closure now retires the same tracked descriptor it opens.

Independent review found and reproduced a cross-profile race in the first repair: an unrelated successful profile write invalidated the current transaction. The corrected check verifies the selected profile and association while retaining global pending-recovery and registry checks. A real two-process regression confirms both disjoint writes and both bindings remain valid. The final focused run passed 39 checks in 56.81 seconds; the two duplicated mounted-edit cases were then consolidated into the existing profile-opening suite, whose 27 checks passed in 103.05 seconds. Independent review approved all eight frozen source/test hashes and separately passed 32 companion/bootstrap checks, five persistence checks and the native concurrency reproduction.

The approved two literal-comparison inventory entries are applied. The path census passes with 54 occurrences matched by 51 approved exceptions; the persistent diagnostic inventory also passes. Ruff is clean on the new helper/bootstrap and new tests. Bandit reports no findings in the helper/bootstrap/raw/private-path scope and exactly the same three existing config findings as the staged baseline.

The two approved Linux diagnostic files were transferred and their exact hashes verified on the native host. The bounded default replacement diagnostic passed against the preserved `f69a11ec` installed package with all 2,865 installed files unchanged. Its earlier finalization failure did not reproduce; no deadline or product rule was changed. The full default replacement/later-rollback repeat also passed on the preserved Linux build, with all 2,865 installed files unchanged. The bootstrap suite passed 31 checks in 4.01 seconds. Current-build platform/keyboard acceptance remains in progress.
