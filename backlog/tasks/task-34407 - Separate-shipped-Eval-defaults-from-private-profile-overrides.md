---
id: TASK-34407
title: Separate shipped Eval defaults from private profile overrides
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 17:58'
updated_date: '2026-10-10 02:43'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow startup from a writable checkout while evaluation settings inherit current shipped defaults and preserve explicit private customizations. Reuse the current YAML lifecycle and recovery ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default loading uses all shipped definitions without admitting the package directory as private user storage.
- [x] #2 Sparse private overrides inherit updated defaults and preserve explicit values, mutable drafts, failed writes, and YAML null values.
- [x] #3 Runtime and recovery select the same effective-config-owned override file; missing overrides are normal and legacy retained definitions keep existing provenance and activation gates.
- [x] #4 Targeted native checks, integration regressions, static analysis, and configuration documentation verify the change.
- [x] #5 Recovery treats only genuinely absent canonical overrides as unused; nonregular or unreadable override paths block complete backup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/220-evaluation-defaults-and-private-overrides.md
Reason: clarify evaluation configuration ownership while preserving ADR-029, ADR-032, ADR-040 and ADR-126.
1. Verify existing merge, YAML persistence, selectors and recovery gates.
2. Add regressions for shipped defaults and private sparse overlays.
3. Split resource and override selectors; adapt the existing loader and recovery binding.
4. Verify native lifecycle and legacy retention behavior; update docs and implementation notes.

2026-10-09 PR review follow-up:
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Default Eval loading reads the shipped YAML as an immutable resource and merges private sparse eval_overrides.yaml beside the effective config.toml. Existing deep merge, raw participant, atomic YAML writer, mutable drafts and recovery owner are reused. Explicit custom files remain full configurations; exports do not mark primary drafts saved. Profile switching preserves held drafts. Missing canonical overrides are normal and legacy retained definitions keep existing provenance/activation gates. Initial private-read failure retains the shipped baseline; full rollback omits only the exact canonical unused declaration.
ADR: backlog/decisions/220-evaluation-defaults-and-private-overrides.md (extends ADR-029/032/040/126). Design/plan: Docs/superpowers/specs/2026-10-04-evaluation-private-overrides-design.md and Docs/superpowers/plans/2026-10-04-evaluation-private-overrides.md.
Verification: reproduced the exact writable-checkout error and missing eight task types before implementation (5 failed, 4 passed). Initial focused native run: 29 passed. Subsequent user-authorized TASK-34408 fixes unlocked deeper native evidence: final recovery run 124 passed, 2 skipped, 1 deselected, including all override, retained/selective/rollback/later-snapshot cases and present/absent fresh-process recovery plus owner reopening for finish and rollback. Final pipe/admission/lifetime/selector/schema run: 100 passed; 211 distinct passing cases across both runs. New files/changed production ranges pass Ruff formatting; the combined 21-file lint comparison adds zero diagnostics (134 inherited), all files parse and whitespace checks pass. Fresh whole-change and protocol reviews have no remaining substantive findings.
Limits: two file-symlink cases skip only actual WinError 1314 because this Windows account lacks the privilege. One broader complete-rebackup fixture fails owner inventory before publication and was deselected. The installed-wheel two-profile flow installed and entered restore but exceeded its existing 300-second timeout, so the complete installed-product release gate remains unverified. No full sweep or POSIX execution was performed. Targeted feature and recovery evidence is complete; these results do not certify every release gate. No safety checks were weakened, and no commit/push/PR was requested.
Updated files: Evals selectors/loader/recovery/README; raw/settings participant integration; override, owner/lifetime/retention regressions; design, plan, ADR and incident-backed testing lessons. TASK-34408 carries the independent Windows protocol/helper corrections.
PR preparation: feature branch fast-forwarded to dev a7d9bca5da; no upstream file overlaps. Fresh focused PR check: 38 passed in 9.11s (pipe helper, pending durability, private Eval overrides). Backlog ID/frontmatter guards pass after the documented voluntary renumbering.
Diagnostic artifact review: the existing loaded-selected-path info message became debug and gained the word evaluation; the interpolation remains selected, with no config contents or secrets added. The five-call count and persistent sinks are unchanged. Regenerated Docs/security/production-diagnostic-inventory.json from clean tracked current-dev sources plus the PR; the only owner delta is Evals/config_loader.py, and all metadata/topology/candidate projections are unchanged. The clean tracked profile-owned path census also passed (54 occurrences, 22 files, 51 exceptions).

2026-10-09 PR #3018 review follow-up: the single delegated read-only reviewer found an existing nonregular canonical override was silently classified unused. Discovery now probes link metadata, keeps nonregular objects missing_required and metadata/read errors unavailable, and reserves unused for genuine absence. Added directory/dangling-link/FIFO/unreadable regressions; meaningful native RED was 2 failed/2 capability skips, then the focused rebased override/durability/pipe run was 40 passed/2 skips. The Windows account cannot create file symlinks and has no FIFO primitive; a focused Windows/Linux/macOS CI job also covers retained/selective/rollback/later-snapshot paths. Changed Python files pass formatting and lint with the two existing exact-type E721 checks retained. The published PR was rebased onto dev a190654d4c; append-only documentation conflicts preserve both versions. Three unrelated unpublished Console documentation commits stay in the original checkout. No new ADR: this restores ADR-220/126 absence and blocking-inventory semantics. Final CI is pending; the task remains In Progress until those results arrive.

Expanded verification follow-up: the first platform run exposed 32 missing-fixture setup errors; affected modules now explicitly reuse the existing helper_resource_root fixture. The shared rebackup oracle now checks canonical private overrides rather than the immutable package resource, and the Eval-specific full-App scenario seeds an actual private override before capture so retention is not vacuous. Native finite recovery coverage: 71 passed, 4 capability skips, with one full-App rebackup exceeding its existing 45-second child deadline. Preserved phase/stack diagnostics locate that delay inside unchanged dev live-capture/startup-reacquisition code; no deadline was increased and the broader result remains unqualified. Removed all temporary diagnostics. The shared child helper now records UTF-8 output to an owned file, avoiding inherited-pipe EOF and decoding failures without changing its deadline. Windows CI initially created fixtures under the elevated default Administrators owner, correctly refused as selector_unverified; CI now reuses the genuine user-default-owner launcher from the existing Console test tooling, changing only test-process TokenOwner and restoring it afterwards, never production guards or ACLs. A new exact-head platform run will verify these corrections before merge.

Linux follow-up reached 75 passing cases with one remaining stale full-App oracle: restored private overrides retain their actual filename instead of eval_config.yaml. The fixture now records exact trusted restore-plan destinations and source digests in its own receipt, verifies each retained file after original/candidate removal, and compares the complete rebackup Eval payload multiset with those exact sources. The Eval-specific scenario requires one actual incoming private source; no retention assertion is removed or made vacuous. Generic media scenarios no longer demand a nonexistent packaged-default payload. All shared and Eval-specific embedded scripts compile with unique replacement anchors.

Precise Windows child evidence identifies the remaining full-App inventory refusal in bundled chat.prompt templates and persona artwork under the Actions checkout, whose native owner is Administrators rather than TokenUser. Linux/macOS are genuinely runner-owned and pass all 76 cases. The Windows workflow now assigns the disposable checkout to the actual authenticated runner using the native ownership operation, verified to be the resolved Git workspace. This changes real fixture ownership, not metadata projections, production policy, guards, file contents, or test deadlines. The original refusal evidence is retained; one more exact-head run will confirm the fixture correction.

Windows packaged-resource roots remain unavailable after leaf/checkout ownership correction because the Actions source is beneath the shared D:\a workspace ancestry. Stage the same Git commit using a no-hardlinks clone beneath the runner profile private ancestor, verify the exact commit, assign real runner ownership there, and install/run from that source so isolated subprocess imports match it. POSIX remains on its original checkout. This preserves every native parent/privacy check and file identity qualification rather than authorizing a shared ancestor. The full refusal records are retained in the owned child-log artifact.

The private Windows clone failed before Python setup because long Backlog filenames exceed Git for Windows default path handling. The clone now sets core.longpaths=true before checkout. The completed job log is preserved; no test deadline, test selection, safety gate or production code changed. Rebased cleanly onto latest dev ec83a5565a (merge-queue tooling only); final platform verification remains pending.

Git long-path checkout succeeds. The native icacls recursion then fails on long Backlog documentation paths, which are not recovery resources. Ownership setup now covers the actual private checkout root plus tldw_chatbook and packages runtime resource trees; all runtime paths retain real runner ownership and all production qualification checks remain unchanged. Explicit shared-fixture registration also passes lint without import-shadow suppressions.

Final pre-budget CI on aca49d97e1: Linux 76 passed, macOS 76 passed; Windows 74 passed, 1 FIFO skip, 1 full-App outer timeout. Genuine private source ownership fixes complete inventory. The same read-only reviewer traced the remaining failure: at least 30.476s of startup/media setup occur before capture, leaving less than 15s of the inherited 45s process deadline for a capture with its own 30s watchdog. Admission/readmission/maintenance/capture/Windows facade are unchanged from dev; the waiting readmission is expected while capture holds its exclusive gate, and no holder cycle is demonstrated. Only this full-App Windows capture/reopen case now has a 90s overall cold-start budget. Capture still has its original 30s watchdog, pytest retains its 180s case bound, POSIX and all other child defaults remain 45s, and every functional assertion is retained. Qualification under the corrected budget is pending.

The 90s Windows run completes original capture and isolated restore, confirming the outer-budget remedy, then exposes a later fixture mismatch: the direct reopen subprocess sets HOME but inherits the pytest session USERPROFILE, so Windows Path.home() looks for restored.json in the wrong profile. Reopen now sets USERPROFILE to the same actual restored home. The two sibling direct reopens in the shared temporary-media fixture receive the same one-line correction. No product selector, custody, capture limit or retention assertion changes. Final pipeline verification is pending.
<!-- SECTION:NOTES:END -->

## Renumbering Provenance

Originally TASK-34363. The 2026-10-04 pre-push sweep found a published claim on `origin/fix/model-metadata-never-rejects-listing`: One model's oversized metadata rejects a provider's whole model list, created at 18:27. This record was created at 17:58, but voluntarily moved to TASK-34407 before publication to keep the peer branch's published references stable. Implementation and acceptance criteria are unchanged; local live references were updated.
