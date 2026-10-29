---
id: TASK-37205
title: Spike portable Windows Chatbook with bundled WezTerm and local data
status: Done
assignee:
  - '@codex'
created_date: '2026-10-28T18:27:00Z'
updated_date: '2026-10-29T00:18:00Z'
labels:
  - packaging
  - windows
  - spike
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove that an extracted Windows bundle launches Chatbook without installed Python or a terminal and preserves an independent data profile when the bundle is relocated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A build command produces a Windows x64 ZIP containing a pinned verified WezTerm release, a private Python runtime, Chatbook and its runtime dependencies.
- [x] #2 The launcher keeps default config, databases, logs, caches and temporary files in the bundle Data directory without importing the existing user profile.
- [x] #3 An installed-runtime smoke check writes and reopens real SQLite data after relocating the bundle, and a real WezTerm window reaches Chatbook.
- [x] #4 Targeted checks and documentation explain portable-data behavior, build provenance and remaining proof-of-concept limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/254-windows-portable-wezterm-spike.md; backlog/decisions/255-bundled-wezterm-portable-runtime-roots.md; backlog/decisions/256-detached-portable-profile-import.md
Reason: packaged runtime/data ownership, contained terminal IPC and an independent detached recovery input boundary.

1. Assemble embedded Python, an installed Chatbook wheel and checksum-pinned WezTerm source/assets.
2. Filter inherited overrides and keep application and terminal state inside portable Data.
3. Build matching GUI/CLI/mux with explicit roots and scoped Windows discovery; use existing system ConPTY for long executable locations.
4. Implement the user-approved destination-only reviewed import, retaining copied source/control unchanged, excluding managed credentials and preserving native publication/activation checks.
5. Verify source acquisition, installed owner/schema/dependency coverage, pending/crash/tamper/concurrency guards and repeat relocation with targeted tests.
6. Rebuild a pristine ZIP; verify real terminal review/TUI, normal exit, note persistence, repeat move with original unavailable and unchanged host WezTerm namespace.
7. Record provenance, exact evidence, limits and implementation notes.

Approved design: Docs/superpowers/specs/2026-10-28-portable-profile-relocation-design.md
Completed implementation plan and rulings: Docs/superpowers/plans/2026-10-28-portable-profile-relocation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Windows x64 portable proof of concept with a private CPython 3.12.10 runtime, installed Chatbook 0.2.3 wheel and matching modified WezTerm GUI/CLI/mux from pinned 20240203-110809-5046fc22 source. Explicit terminal home/data/runtime selectors, scoped Windows discovery and bounded Data-relative socket calls keep terminal-owned state inside Data. The pinned optional conpty.dll/OpenConsole.exe pair fast-fails at long executable locations; omitting it selects upstream's existing system ConPTY fallback (Windows 10 1809+). Other helpers, source patch, build provenance and hashed license notices remain included.

Destination-only relocation needs no preparation or original location. Stable no-follow acquisition and pathless copied-control completion checks feed installed owner/schema/dependency validation and a credential-excluded partial archive. Existing isolated recovery publishes fresh native authority; committed associations are independently checked at startup. Copied source and earlier generations remain unchanged/inert, approvals remain historical, all activation reviews remain required and managed credentials need setup again. Ambiguous pending operations, unsafe aliases, unknown durable content and unsupported layouts refuse unchanged. Interrupted local publication resumes the same receipt-bound operation. Validation exits its own worker before publication; lazy existing Chat exports prevent pure goal validation from initializing ordinary config. No ancestry reset, manual lease retirement or imported execution authority was added. Exact declared transient subtrees are skipped before reading GPU-held cache files; sibling content and excluded-root junction checks remain enforced.

Governance: ADR-254, ADR-255 and user-approved ADR-256, preserving ADR-029/125/126. Spec/plan: Docs/superpowers/{specs,plans}/2026-10-28-portable-profile-relocation*. Main changes: Packaging/windows/portable, detached source/evidence/layout/worker/coordinator modules, pure journal parsing, terminal-selector preservation in isolated_restore, lazy Chat exports, focused tests and lessons-live-verification.md. The coordinator uses relocate(data, confirm) plus verified_selection instead of three separate proposed APIs; the approved authority/crash contract is unchanged.

Final artifact: output/Chatbook-Windows-Portable-Final.zip, 171,084,982 bytes, SHA-256 781e6a6df0e4c18e1859396d4a0e6fd70dd73d5e409a0d1662cb7e35b9efb718. Its 8,815 entries exclude Data and the optional ConPTY pair; current launcher/recovery source bytes match. Matching final wheel/sdist pass Packaging/check_manifest.py. All seven shipped native files match the x64 direct/delay dependency audit; modified binary/source/patch/toolchain provenance and 734 notices for 477 package declarations are preserved. No test Data or credentials are shipped. Base source revision: 798f8fca75a7acba56da5f523329efa52c4a78f0 plus this worktree's reviewed changes.

Live qualification: the first reviewed import reached real Chatbook, exited normally and reopened the fixed real note with all 80 activation requirements pending. Retained source hashes and the host terminal namespace matched. The exact final ZIP also passed its own fresh-runtime probe from System32 (93 migrations, package beneath runtime/Lib/site-packages). That independently used restored Data generation was then moved into the final extraction and the complete extracted folder moved again, with previous Data and extraction paths unavailable. The exact Chatbook.cmd showed the second review, accepted relocate and reached the actual Home/Console UI. Its live gui-sock endpoint was inside Data/wezterm/runtime. Normal Ctrl+Q exit and two fresh installed-runtime probes reopened the same note; all 80 activation requirements stayed pending, every retained source/generation file hash matched, and host WezTerm snapshots before/during/after were identical. Evidence: output/portable-final-verification.json, output/final-review-pane.txt, output/final-tui-pane.txt, output/portable-final-post-tui.log; earlier full-profile native repeat-move proof is in output/relocation-{fresh-start,second-move}.log.

Targeted verification: combined new recovery run 131 passed; subsequent source guards 13 passed, evidence 79 passed, layout 27 plus four added residual-state checks passed, packaging 15 passed, native builder three checks/five subtests and five exact-source Rust helper tests passed. Current 22-file Ruff check/format, git diff --check and final wheel/sdist manifest validation pass. Independent final review findings were addressed with narrow census/evidence fixes and regressions; actual native/ZIP journeys cover the areas it could not qualify.

Wider legacy runs are not green: recovery-regressions.log records 95 passed/five failed (Windows POSIX-mode expectation, encrypted helper unavailable, two symlink fixtures needing privilege, stale schema 77 vs installed 84). recovery-isolated-regressions.log records seven passed/five failed (three encrypted-helper cases, symlink privilege, fresh-launch fixture isolated_activation_required). A later single fixture rerun stopped at its shared-parent posture. These harness/unsupported-encrypted-path results are retained; equivalent new native unencrypted publication/reopen journeys pass. No full test sweep was run, per repository opt-in policy.

Qualification is local NTFS/same Windows account, not a signed production release or a clean-machine/all-features claim. Unsupported optional layouts refuse; ML models, optional feature extras and external programs are not included. Copied raw state cannot prove historical multi-store coherence, so import remains explicitly partial. Retained source may still contain old credentials; protect all Data. Clipboard/images/graphics variants and cross-account/filesystem moves are not qualified. No commit, push or PR was requested; the managed worktree is preserved.

Follow-up TASK-37269 resolves the earlier recorded Windows/helper failures: fully installed isolated-worker dependencies, native ACL/junction/named-pipe fixtures, installed-schema expectations, private test temporary storage and correct fresh-child home/profile selection. Frozen combined verification:361 passed; packaging18+5 subtests; full post-rebase isolated/model cohort20 passed; final real CLI selection/data case passed after an observer negative-control RED/GREEN proof. Independent review found no actionable issues. Existing runtime guard/crypto bytes are unchanged; the ZIP qualification above remains historical to its recorded source base and hash. Portable decisions now use canonical ADR-254/255/256 after resolving dev numbering collisions. Details and remaining prototype limits are in TASK-37269 and Docs/superpowers/plans/2026-10-28-windows-recovery-test-fixes.md.
<!-- SECTION:NOTES:END -->
