---
id: TASK-31992
title: Add recovery inventory for configuration durable files and optional content
status: Done
assignee:
- codex
created_date: 2026-09-07 23:52
updated_date: 2026-09-11 19:59
labels:
- backup-recovery
dependencies:
- task-31978
- task-31986
- task-31987
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: All app-owned durable file categories and directory topology are classified with explicit dependency and metadata rules.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All app-owned durable file categories and directory topology are classified with explicit dependency and metadata rules.
- [x] #2 External folders/models/temporary media/diagnostics remain opt-in without weakening custom app-owned storage coverage.
- [x] #3 Aliases, unknown durable entries, missing required files, and unsupported metadata produce truthful coverage results.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the existing tree inventory and establish a new behavioral metadata RED; retain empty-directory evidence.
2. Add immutable versioned preview metadata and locally supplied optional selections, keeping explicit topology and aggregate owner overlap checks.
3. Trace registered producers and canonical selectors for current/history config, templates/prompts, local definitions, persona assets, TTS stores/references, generated assets and model stores; map unresolved rows explicitly and keep completeness blocked.
4. Add import-light installed owner factories, known managed-secret locations and pure relocation policy without runtime bootstrap or secret decryption.
5. Classify external/model/diagnostic/temporary selections and output/control exclusions; refuse links, aliases, mounts, unsupported metadata and missing selected/required sources.
6. Run targeted file inventory cases and affected architecture/import/admission guards, scoped static checks and self-review.
7. Update owner documentation and exact implementation evidence; commit task-owned files and complete independent controller review before closing the task.

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of the approved recovery inventory and metadata contract, reusing ADR-029/030/036/059/060.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented versioned directory/file metadata and pure selection-aware recovery
inventory; config/history/definition owners, TTS schema4/BLOB capture, persona
manual/builtin references, Skills and installed model closure adapters. Config
remapping uses exact installed section.key names and never edits arbitrary prose.
Checked SHA256 streaming retains native retirement/admission; ordinary raw writers
remain explicitly participant_pending for coordinated task10 drain. Existing
outputs, unknown owners, unsafe pending state and unsupported metadata still block
Complete; no archive/activation/credential qualification is claimed.

Validation: final focused inventory51 passed; combined core/inventory/SQLite and
owner census181 passed; shared-reader/admission guards62 passed; runtime/path/Skills/
TTS seams78 passed; new SQLite behavioral copy/policy23 passed. Diagnostic inventory
guard remains a proven clean-BASE failure in unchanged earlier-slice files, retained
as task26 branch verification debt. Scoped27-file fatal lint,13-file format check,
exact lazy-export AST comparison and git diff --check passed. Full command history,
intermediate failures and owner handoffs are in the task9 execution report.

ADR: backlog/decisions/126-complete-local-backup-and-recovery.md, reused with the
approved ownership/private-path/sync ADRs. Design references preserved. Five-digit
Backlog direct-file fallback was used during implementation; the completion CLI
updated the correct task, and its rewritten design references were verified. No unrelated work, full suite, network/download, push or merge.

Scoped follow-up after provisional05844a86f maps six installed durable defaults:
chat.prompt_history, ui.state, ui.emoji_recents, ui.themes, chatbooks.registry and
chatbooks.archives. Baseline includes retained default content exports; registry
selection follows the installed Prompts DB sibling even outside the default data
root. Arbitrary registry file_path remains inert. Exact process instance lock and
Creator chatbooks/Importer imports scratch have source-backed exclusions, while
data/temp topology is retained and unknown siblings or wrong-kind paths refuse.
Each new ordinary writer stays participant_pending for task10; no writer bypass.
Focused/aggregate/source-census follow-up97 passed; six checked byte-for-byte file
captures and real producer cleanup fixtures included. Scoped three-file fatal lint,
two-file format check and git diff --check passed. The report records actual RED,
intermediate private helper/fixture authority corrections and final evidence.
Initial commit is preserved; reviewer covers recorded BASE through follow-up HEAD.
ADR-126 and existing design references retained.

Independent review fix round1 addresses only I1/I2: checked root-only optional
exclusions retain unsafe/unavailable evidence, and all three persona semantic
roots declare and enforce the exact profile core edge before peer reads. Existing
full-tree/builtin empty selection and public APIs remain unchanged. Tasks15/17
must validate semantic included_directory dependencies even though directory
creation bypasses raw-file copy. Behavioral RED22 failures reproduced the review;
final affected file/core/aggregate/census checks184 passed. Four-file scoped fatal
lint/format and git diff --check passed. Baseline diagnostics/warnings remain
recorded debt; no environment or unrelated inventory change. ADR-126 reused.
Independent spec/quality review and scoped re-review approved I1/I2 at
84dfad68b5a053fd96cf3b9620dfc140e7918f81 with no open Critical/Important findings.
All three criteria are complete. Maintenance enrollment remains assigned to
TASK-31993; complete-backup exposure still depends on later qualification tasks.
2026-09-11: Read-only preflight /private/tmp/two-profile-skills-chatbook-cohort-preflight.md identifies potential existing shared registry/cross-profile owned ZIP reference loss. Scope remains original multi-profile/shared durable stores. Authorize separate focused test module using actual app/service selected registry path derived from shared Prompts DB and native note-containing Chatbook export/create, both profile-owned ZIP roots, actual Complete capture and explicit isolated relocation. Establish behavioral result before any product edits; do not split shared registry, override roots, manufacture typed references, drop records or change A-owned combined fixture. Agent P owns focused test and diagnostic; root reviews minimal correction if observed.
2026-09-11 actual shared Chatbook regression failed1/21.14s at held capture shared_payload_mismatch after successful native two-profile note ZIP export/shared registry/Complete preview. Report /private/tmp/shared-chatbook-registry-red-and-proposal.md. Release bounded correction in config_adapter._ChatbookRegistry, concrete inventory closure call after existing physical aliases, staging passes existing authenticated selected StorageItems to Chatbook checks, plus new focused test. Union only archive dependencies already declared by actual same-path/shared-group registry peers; deterministic reference IDs; cross-profile refs require selected exact peer+archive owner/dependency and explicit destination. Preserve reader shared_payload_mismatch, ordinary no-context/same-profile policy, other owners/external refs. Agent P implements actual capture→isolated→fresh shared ZIP readers + finite negatives; no new framework/authority/owner or blanket profile relaxation. Root handles review, relevant census and scoped commit.
Shared Chatbook registry correction independently reviewed and approved: exact same actual shared registry derives deterministic union only of included peer archive dependencies. Selected typed owner/context/mapping checks retained. Native original red at shared_payload_mismatch; corrected full capture/public ZIP/isolated/twofreshappreaders1/42.51s,16 policy tests and30 existing relocation/census tests green. Two fixture-only failures preserved accurately. Production Bandit0/Ruff4→4. Frozen4 source/patch/30 evidence hashes verified. Report /private/tmp/chatbook-shared-registry-correction-report.md; review /private/tmp/shared-chatbook-independent-review.md. Root committing exact3product+newtest.
Original Task26 populated-owner preflight identifies subscriptions.assets native isolated mapping path reaches owner_relocation_unverified. Release bounded expected-success regression in new Tests/Backup_Recovery/test_briefing_audio_restore.py using actual watchlist/briefing/script/TTS stored profile and deterministic external synthesis seam. Actual public capture+isolated restore/fresh passive script/audio resolution; source-only finding must be reproduced. No product changes or guard exemption before evidence/source review. Root owns tracking/commit; no external synthesis/model/network.
Actual native briefing/profile seed with declared private pydub succeeded, but Complete preview refuses only the genuine ProfileStoreLease tldw_chatbook_tts_profiles.db.lock (0-byte regular singlelink native file): read-only retained diagnostic0.40s confirms. Release separate bounded TTS/recovery.py + new Tests/Backup_Recovery/test_tts_profile_lock_inventory.py, recognize only exact selected TTS DB+'.lock' with validated empty regular/nonlink/singlelink semantics as intentionally excluded; absent excluded, unsafe/nonempty still refuse. No blanket suffix policy/new owner or native lease change. Tests first actual native lock and wrong kinds/contents/unknown-neighbor, then guarded A briefing can reach original asset relocation. Root handles owner-ledger updates/commit.
Exact TTS profile store lock correction approved for commit: selected DB+.lock only, primary owner once; absent or empty regular singlelink intentionally excluded, linked/nonempty/dir refused, references aliases unchanged. Actual native red2/1.71 ->8new+11census19pass12.54s; Ruff/Bandit0, no census exemptions. Root independent full2file/source/snapshot/patch/5receipt hash review approved /private/tmp/tts-lock-inventory-independent-review.md. /private/tmp/chatbook-tts-lock-inventory-report.md preserves initial fixture failures separately. Separate actual briefing expected-success rerun remains necessary.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-9)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.
