# Portable profile relocation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Relocate an already used Windows portable bundle with its original unavailable, retaining the copied source and all recovery checks.

**Architecture:** Acquire the copied source as inert, bounded input. Installed owner declarations produce a credential-excluded partial archive, then existing isolated restore publishes a fresh profile under independent recovery authority. The portable launcher selects only a verified committed profile; modified bundled WezTerm confines its files and socket endpoints to Data.

**Tech stack:** Python 3.12 standard library, existing recovery adapters/Pydantic/SQLite, pinned WezTerm Rust release.

**Spec:** [Approved design](../specs/2026-10-28-portable-profile-relocation-design.md)

ADR required: yes
ADR paths: backlog/decisions/255-bundled-wezterm-portable-runtime-roots.md and backlog/decisions/256-detached-portable-profile-import.md
Reason: terminal runtime paths and a distinct detached recovery input boundary.

## Global constraints

- Destination-only; no source-side preparation and no original-path IO.
- Complete supported base profile; unknown durable content and unsupported owners block, never silently omit.
- Copied source/control evidence remains unchanged and inert.
- No manufactured maintenance authority, ancestry reset, imported approval activation, or keyring lookup.
- Archive consistency is partial and managed credentials are excluded.
- SQLite main/WAL/SHM is acquired together; normalize disposable copies only.
- Fresh publication uses existing isolated restore, installed validation and activation review.
- Local NTFS, same account; no full test sweep without user opt-in.

## Review focus

- Config or database locators pointing outside copied Data never cause original-path reads (Tasks 1/3/4).
- Nonempty WAL retains committed rows; rollback or damaged sidecars refuse unchanged (Task 3).
- A changed member list or byte before sealing or executor admission prevents publication; approval binds the sealed snapshot during destination staging (Tasks 1/4).
- Crash after publication or concurrent confirmation never selects an uncommitted profile or republishes blindly (Task 4).
- Moving a selected restored generation again imports the selected payload, without recursively treating retained generations as live owners (Task 4/5).

## File responsibilities

- `Backup_Recovery/detached_source.py`: pinned source census/acquisition and immutable review identity.
- `Backup_Recovery/detached_evidence.py`: pathless strict copied control/journal completion checks.
- `Backup_Recovery/detached_layout.py`: installed confined owner declarations, staging validation, normalization and partial archive producer.
- `Backup_Recovery/detached_worker.py`: one-shot validation child, terminated before native publication; private bounded result protocol.
- `Backup_Recovery/portable_relocation.py`: independent host, durable outer transaction, review/confirmation and verified selection.
- `Chat/__init__.py`: lazy existing service exports so pure goal models do not initialize ordinary config inside recovery.
- `Packaging/windows/portable/portable.py` / `app_entry.py`: entry dispatch and selected-profile launch.
- Matching focused tests under `Tests/Backup_Recovery/` and `Tests/Packaging/`.
- `Packaging/windows/portable/wezterm/`: reproducible modified binaries, license/provenance and runtime qualification.

### Task 1: Stable detached source acquisition

**Interfaces:** `SourceReview` immutable records; `review_source(root: Path) -> SourceReview`; `acquire_source(review: SourceReview, destination: Path) -> Path`; `recheck_source(review: SourceReview) -> None`.

- [x] Write tests for real files, source digest/member mutation, hard links/reparse points, unsafe/colliding members, private acquisition and no source writes.
- [x] Run the tests and observe missing behavior fail.
- [x] Implement using installed pinned no-follow traversal/private-file primitives; hash through held descriptors and compare complete census before/after acquisition.
- [x] Run focused source tests. Expected: all pass and source hashes unchanged.

### Task 2: Pathless copied recovery evidence

**Interfaces:** `validate_copied_evidence(root: Path) -> tuple[str, ...]` over acquired private source; reason codes only, no paths followed from records.

- [x] Test duplicate keys/unknown versions/corrupt chains, pending intent/registry state and valid terminal journal sequences with path fields referencing nonexistent roots.
- [x] Observe RED, factor existing pure journal parsing, implement conservative completion checks without constructing Journal.
- [x] Run new evidence tests and relevant existing journal parser tests. Expected: valid terminal evidence accepted, every ambiguous publication refused.

### Task 3: Complete confined installed owner import

**Interfaces:** `produce_detached_archive(source: Path, work: Path, cancel: Event) -> DetachedImportResult`, containing `archive: Path` and inert review lines. Consumes acquired input and Task 2 validator; does not capture live owners.

- [x] Test base config, main Notes/conversations DB, other installed base SQLite/owned files and representative assets against exact installed policies; test custom external/unknown content refusal and inert external workspace references.
- [x] Test WAL committed row survival using real SQLite and damaged/rollback sidecar refusal; observe RED.
- [x] Add explicit installed layout mapping that resolves only acquired members, validates dependencies, normalizes staged SQLite sets, excludes credentials and constructs a strict partial manifest.
- [x] Feed existing archive writer/acquire/verify and test the archive against installed isolated destination planning. Expected: complete supported payload with no copied control/approval authority and no old-target reads.

### Task 4: Journaled destination-only relocation

**Interfaces:** `relocation_review(data: Path) -> review`; `confirm_relocation(data: Path, digest: str) -> selected profile`; `selected_profile(data: Path) -> verified selection or None`. Fresh subprocess receives selected config/control/profile only after `_launch_state` validation.

- [x] Test review/cancel, source unavailable, changed review, two contenders, interruption at acquisition/archive/publication/selection and repeated launch/move.
- [x] Observe RED; implement independent private host/generation roots, an exclusive outer transaction and durable records that bind the approved source and one existing restore operation.
- [x] Reuse archive reader, `resolve_destinations(mode="isolated")`, `restore_isolated` and `select_profile`; ambiguity refuses for existing recovery resolution instead of republishing under a new ID.
- [x] Verify all installed activation-required owners still require review and credentials need setup again. Expected: retained bytes unchanged, committed profile selected only after normal validation.

### Task 5: Portable launcher, terminal and packaged journeys

**Interfaces:** bundled launcher dispatches review inside WezTerm before normal app imports; terminal GUI/CLI/mux share explicit roots and bounded socket addressing.

- [x] Test launcher relocation dispatch, filtered environment and verified selection. Preserve normal first-use startup and refusal semantics.
- [x] Finish pinned WezTerm build; verify provenance, modified notice, all bundled license digests and matching executables.
- [x] Rebuild wheel and pristine ZIP, extract into deep space/Unicode paths, create real Notes/config/assets, move first with original unavailable, confirm local review and reopen twice.
- [x] Capture real TUI via matching CLI from Data; verify actual IPC/log paths under Data and unchanged host WezTerm namespace.
- [x] Run targeted new and existing archive/planner/isolated/activation/admission tests, Ruff, manifest checks and diff whitespace checks. Update docs/task with exact evidence and artifact hashes.

## Execution ledger

- Design approved by user on 2026-10-28. Existing managed worktree retained.
- Ruling: continue implementation directly within approved scope; no redundant execution-method approval. Focused independent evidence/layout work may use the already authorized parallel-agents workflow.
- Ruling: targeted verification only, as AGENTS.md explicitly requires opt-in for a full test sweep.
- Pre-flight: Tasks 1 and 2 feed Task 3 through an acquired local Path; Task 3 feeds Task 4 via strict archive Path; Task 4 feeds Task 5 via verified selected profile. No source record is launch authority.
- Task 1: 10 focused native source checks pass, including real NTFS hard links and a junction, byte/member mutations, copy-time mutation and disjoint private acquisition.
- Task 4: focused selection/locking/repeat-source/environment checks and real native publication pass; exact rebuilt-package GUI qualification is tracked separately below.
- Ruling: the exact installed immutable built-in asset owner may supply declared built-in dependencies, validated against imported IDs/hashes/references. Normal startup seeds these references; refusing every normal base profile would defeat approved coverage. Custom assets still require acquired bytes. Cost if wrong: reject the dependency; never substitute unverified content.
- Targeted regression audit: 95 passed before five unrelated existing test/environment failures stopped the run: Windows builtin Path.stat POSIX-mode assertion, unavailable encrypted helper, two symlink fixtures lacking Windows privilege, and stale schema expectation 77 versus installed 84. Evidence: output/recovery-regressions.log. No full sweep was run and these failures are not reported as green.
- Review fixes: growth is bounded before EOF/write (11 source checks pass); old-location publication markers require pathless terminal and outer-selection proof before a later move; confirmation now includes owned content, bytes/space, setup notices and concrete destination.
- Ruling: approval binds the sealed acquired snapshot, as the approved source-boundary contract specifies. Recheck retained source before executor admission; external edits during destination staging remain retained and are never silently merged. The plan's earlier unlimited post-review freeze claim exceeded that boundary. Cost if wrong: a later external edit remains only in retained source; the review explicitly explains this.
- Task 4 integration: real default profile reached committed isolated authority; a fresh process reopened the fixed real note and verified all 80 installed activation owners pending. Moving that selected generation again with its prior folder absent produced another committed fresh profile. Evidence: output/relocation-{resume-trace,fresh-start,second-move}.log.
- Ruling: isolated owner validation runs in an exiting child; ordinary startup leases remain enforced. Pure Chat exports are lazy to fix a demonstrated goal-validator dependency importing ordinary config during native recovery. No manual lease retirement or startup bypass is used. Cost if wrong: validation/publication refuses with retained evidence.
- Independent final review found two base-coverage refusals: installed notes-sync/video/log artifacts were missing from initial census; evidence grammar reserved recovery-like names globally in skill payload. Narrow installed census declarations and canonical control-root parsing now have dedicated passing controls.
- Final review declined native execution/packaged journeys while they were running. Ruling: verify those with actual modified executables and rebuilt ZIP rather than infer them from code. Cost if unverified: do not claim portable qualification.
- Ruling: detached partial input can list proven absent default core stores as unused, with no custom selector or sidecar/lock residue, explicit review disclosure and all included-owner dependency checks. The real note probe and normal lazy DB creation otherwise produce valid raw profiles the capture-era required-core census rejects. No empty source payload is fabricated. Cost if wrong: historical absence cannot be distinguished from earlier deletion; the partial snapshot and listed absences make that limit explicit.
- Packaged GUI diagnosis: identical short ASCII and Unicode executable locations succeed, while a 168-character ASCII location fast-fails 0xC0000409. Removing the pinned optional conpty.dll/OpenConsole.exe pair makes the same long-location marker succeed. Ruling: use the existing kernel32 ConPTY fallback (Windows 10 1809+) and omit that pair, retaining the other upstream helpers/notices. No executable relocation, short-name dependency or native-guard bypass. Focused packaging regression observed RED, then all 15 packaging checks passed.
- Actual GUI source acquisition refused a driver-held Data/home/AppData/Local/NVIDIA/GLCache file. The installed layout already classifies Local, home/.cache and home/.local/share/wezterm as transient. Acquisition now applies those same exact relative subtree exclusions without opening their contents; ordinary siblings remain visible and a junction at an excluded root still refuses. Both new checks observed RED, then all 13 source guards passed. No durable-owner or native permission policy changed.

- Final qualification: output/Chatbook-Windows-Portable-Final.zip, 171,084,982 bytes, SHA-256 781e6a6df0e4c18e1859396d4a0e6fd70dd73d5e409a0d1662cb7e35b9efb718. All 8,815 members exclude Data and the optional ConPTY pair; current recovery/launcher source bytes match. The fresh exact-ZIP probe found 93 migrations and persisted a real note. Its independently used restored Data was then moved into the final extraction and the complete extracted folder moved again with both prior locations unavailable. Actual Chatbook.cmd review/confirmation and TUI passed; normal Ctrl+Q exit, two fresh note reopens, all 80 activation requirements pending, unchanged retained source/generation hashes and matching host WezTerm snapshots before/during/after passed. The live GUI socket was beneath Data/wezterm/runtime. Evidence: output/portable-final-verification.json, output/final-{review,tui}-pane.txt and output/portable-final-post-tui.log.
- Verification: initial combined new recovery cohort 131 passed; subsequent source guards 13 passed, evidence 79 passed, layout 27 plus four added residue checks passed, packaging 15 passed, native builder three checks/five subtests and five exact-source Rust helper tests passed. Current 22-file Ruff check/format and whitespace checks passed; matching final wheel/sdist manifest checks passed. All seven shipped native files match their x64 dependency audit. Wider legacy failures remain documented in the task; no full sweep, clean-machine or all-optional-feature qualification is claimed.
- Implementation interface adjustment: one `relocate(data, confirm)` operation owns review/confirmation and returns the selection, with `verified_selection` for startup, instead of adding three separate coordinator APIs. It retains the planned authority boundaries and crash behavior.
