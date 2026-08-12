---
target: the merged File Notes page in the production Library shell
total_score: 26
p0_count: 1
p1_count: 2
timestamp: 2026-08-11T20-58-28Z
slug: ok-widgets-library-library-file-notes-workspace-py
---
# File Notes final acceptance critique

## Design Health Score

| Heuristic | Score (0-4) | Assessment |
|---|---:|---|
| Visibility of system status | 3 | Save, root, conflict, and Git states are present, but high-stakes states do not gain enough semantic emphasis. |
| Match with the real world | 3 | Files, folders, drafts, and session changes map to familiar concepts; the push review still leads with implementation vocabulary. |
| User control and freedom | 2 | Navigation and recovery are generally strong, but conflict Reload replaces the preserved draft without a distinct confirmation. |
| Consistency and standards | 3 | The page largely follows the Neon Workbench system, with one local fixed focus color and inconsistent disabled treatment. |
| Error prevention | 2 | Git safeguards are excellent, but destructive conflict Reload and unreadable disabled actions weaken prevention. |
| Recognition rather than recall | 3 | Actions and states are mostly labeled in place; placeholder-only fields and the vague Maintenance label still require inference. |
| Flexibility and efficiency | 3 | Keyboard flows and direct actions are strong at working widths, but the production shell makes Files unreachable at 40x20. |
| Aesthetic and minimalist design | 2 | Content-safe focus and restrained borders are materially better; push review and maintenance paths remain dense. |
| Error recognition and recovery | 3 | Recovery copy is unusually complete, but conflict and offline states need stronger visual signaling and safer Reload semantics. |
| Help and documentation | 2 | Guidance is accurate, but persistent labels and plain-language outcome summaries are incomplete in advanced flows. |
| **Total** | **26/40** | **Acceptable at working widths, but not release-ready across the declared viewport and data-safety contract.** |

## Anti-pattern verdict

The page passes the AI-slop check. It is a specific, terminal-native local-file workbench with strong authority, provenance, and recovery semantics. It does not look like a generic dashboard, decorative AI shell, or template-driven card grid.

The deterministic detector was attempted against this target and exited with `Error: bundled detector not found.` It produced no JSON, rule counts, or findings. That result is unavailable evidence, not a clean detector pass. Browser overlay inspection was not applicable because this surface is a native Python Textual interface, not an HTML DOM.

## Overall impression

The completed sequence solved the original visual-fidelity problem. Focus no longer adds content-obscuring borders to note rows and controls, animation is absent, structural borders have a job, local-folder authority is clearer, routine actions are distilled, and Git recovery language is substantially more honest. At wide and moderate widths, File Notes now feels like a trustworthy local workbench.

The final production-shell pass found one release blocker and two high-priority trust gaps. At 40x20 the Library rail remains 120 columns wide while the File Notes canvas begins at x=120, so the entire file-backed UI is outside the viewport. In conflict states, Reload silently replaces a draft that the page says is preserved. Disabled File Notes actions also inherit the known Textual dimming stack without the app-tier contrast repair used elsewhere. These are narrow, actionable defects rather than a reason to redesign the page again.

## What is working

- Focus styling is content-safe. Button, tree-row, and workflow focus cues no longer consume the cell rows that contain labels or entries.
- The visual system fits the product. Semantic state, source authority, and recovery drive color and structure; decorative animation and borders are absent.
- Normal editing has low to moderate cognitive load. Navigator, editor, persistence state, and primary actions form a legible hierarchy at working widths.
- Action disclosure improved. Less-frequent operations live under Maintenance, while recovery actions appear when relevant.
- Local authority is explicit. Exact linked-root and warning details remain available without crowding the main surface.
- Session Git has unusually strong operational honesty. It scopes mutations to session paths, exposes uncertainty and recovery, and keeps complete result copy keyboard reachable.
- Focus repair and compact component behavior are heavily tested. The selected acceptance matrix passed 32 focused tests, including repeated focus transitions and 40x20 component flows.

## Cognitive load

Routine note editing passes single-focus, grouping, one-at-a-time, and primary-task checks. Load rises in Git and recovery states.

- **Chunking: fail.** The push review can render roughly 17 information blocks with object IDs, refs, lease, transport, hooks, authentication, and provenance at similar weight.
- **Visual hierarchy: fail.** Conflict, error, offline, and ordinary save states share muted status treatment instead of a stronger semantic ladder.
- **Minimal choices: fail.** A conflict editor can expose four actions, or eight when Maintenance is open, before the user has resolved the central problem.
- **Working memory: fail.** Placeholder-only search and path fields lose their meaning after input, and the path field changes meaning by action.
- **Progressive disclosure: fail in the push review.** Technical provenance is valuable, but it appears before a concise statement of what will happen, where, and with what side effects.

## Technical audit

| Category | Score (0-4) | Assessment |
|---|---:|---|
| Accessibility | 2 | Keyboard and focus behavior are strong, but disabled labels and persistent field labeling remain incomplete. |
| Performance | 3 | Disk scanning and search are offloaded, but complete Tree rebuilds and sorting still occur synchronously on the UI loop. |
| Responsive behavior | 1 | Component harnesses pass at 40x20, but the production Library shell places the entire Files canvas outside the viewport. |
| Theming | 3 | Most styles use semantic tokens; the Git panel locally fixes the focus background to `#51677e`. |
| Anti-patterns | 4 | No decorative motion, generic AI decoration, content-obscuring focus border, or arbitrary ornamental structure remains. |
| **Total** | **13/20** | **Acceptable implementation quality with a release-blocking shell integration defect.** |

## Priority issues

### P0 - File Notes is outside the viewport at 40x20

**Evidence:** `library_screen.py:3577-3586` defines the active Notes workflow as Database-only. `_library_notes_compact_stage_applies` at `library_screen.py:4067-4071` therefore returns false for Files after entry. In `library_screen.py:7643-7675`, both the rail and canvas stay displayed; the canvas has a 40-column minimum. A production-shell geometry probe measured `#library-rail` at x=0, width=120 and `#library-file-notes-workspace` at x=120, width=520 in a 40-column screen.

**Impact:** The product advertises a supported compact viewport but provides no visible or keyboard-reachable local file-backed notes UI there. Direct component tests pass because they omit the owning Library shell.

**Recommendation:** Extend compact single-stage routing to the Files workflow, preserve the active stage across resize, and add a full `LibraryScreen` regression that asserts actual viewport intersection and keyboard reachability at 40x20.

**Tracking:** TASK-15502.

### P1 - Conflict Reload silently discards the preserved draft

**Evidence:** The conflict surface says the draft is preserved, but `_reload_file` in `library_file_notes_workspace.py` only flushes ordinary dirty state. In conflict, it immediately reopens disk and overwrites the editor. The existing test explicitly expects the draft to be replaced.

**Impact:** The highest-risk recovery state presents a destructive action as routine. This violates the local-first trust contract and makes the reassurance about draft preservation misleading.

**Recommendation:** Rename the action to `Discard draft and reload disk`, make the first activation open a distinct confirmation with Cancel focused, and revalidate file/session freshness before intentional replacement. Keep the complete base/draft/disk experience in TASK-399.8.2.

**Tracking:** TASK-15503, bounded immediate guard; TASK-399.8.2, full conflict experience.

### P1 - Disabled File Notes actions may be unreadable

**Evidence:** File Notes workspace and Git actions use ordinary disabled Buttons. `_buttons.tcss:37-45` applies `$text-disabled` and 50 percent color, while Textual also contributes dim styling. File Notes has no app-tier override comparable to the repaired Console and Library surfaces. DESIGN.md records that the inherited stack puts all shipped themes below the 3:1 product minimum.

**Impact:** Users cannot reliably read which action is unavailable or why, especially in trust-sensitive Git and recovery states. The state may appear broken rather than intentionally unavailable.

**Recommendation:** Add an app-tier File Notes disabled treatment that measures at least 3:1, remains non-actionable through a stable non-color cue, and retains visible reason copy.

**Tracking:** TASK-15504.

### P2 - Push review leads with technical provenance instead of the outcome

**Evidence:** `library_file_notes_git_panel.py:1470-1531` and `1841-1887` can present roughly 17 peer-weight blocks covering object identity, refs, lease, transport, hooks, authentication, and session provenance.

**Impact:** Even experienced users must reconstruct the core operation from details before deciding whether it is safe.

**Recommendation:** Lead with four blocks: what changes, where they go, exact scope, and side effects. Put OIDs, lease, transport, hooks, and authentication under `Technical details` without removing them.

### P2 - High-stakes states are visually under-signaled

**Evidence:** The save-state line remains `$text-muted` through ordinary, conflict, and error states. Offline linked-root state also depends on whether a separate runtime warning happens to be present.

**Impact:** The page communicates danger through copy alone, so conflict and offline states can scan like routine status.

**Recommendation:** Add semantic warning, conflict, and error classes while retaining complete text. Do not introduce decorative animation or content-consuming borders.

### P2 - Search and path inputs rely on disappearing placeholders

**Evidence:** The search and path Inputs around `library_file_notes_workspace.py:828-833` and `864-868` use placeholders without persistent labels. The path field's meaning varies with the selected action.

**Impact:** After typing, users must remember the field's purpose and current operation.

**Recommendation:** Add compact persistent labels or an adjacent state-specific instruction that survives input.

### P2 - Git focus styling duplicates a fixed palette value

**Evidence:** `library_file_notes_git_panel.py:610-612` defines `$ds-focus-bg: #51677e`, which is consumed by row, button, and workflow focus rules. A live theme probe returned the same RGB value in dark and light themes.

**Impact:** Focus contrast and brand behavior can diverge from the selected theme even though the central design system already exposes semantic focus tokens.

**Recommendation:** Remove the local literal and use the central semantic focus surface. Test behavior through token resolution rather than pinning the hex value.

### P2 - Large result sets rebuild complete Trees on the UI loop

**Evidence:** Scanning and search are offloaded, but `library_file_notes_workspace.py:1269-1284` and `1476-1539` synchronously sort and add all nodes or results to Textual Trees.

**Impact:** Large linked roots can pause interaction even though disk work itself is asynchronous.

**Recommendation:** Use bounded batches, incremental diffing, or pagination while preserving focus and selection. This is already part of TASK-399.4 acceptance criterion 8 and should not be duplicated.

## Persona red flags

- **Alex, frequent keyboard user:** the 40x20 shell failure makes the entire workflow unreachable, while a ready Git list can still expose up to six actions.
- **Jordan, local-first non-Git user:** `draft preserved` followed by a routine-looking Reload implies safety that does not exist; the push review front-loads specialist language.
- **Sam, accessibility-focused user:** production-shell focus cannot reach the offscreen canvas at 40x20, disabled labels can be illegible, and dynamic screen-reader announcements remain unverified.
- **Morgan, local-first operator:** silent conflict draft loss directly contradicts the page's authority and recovery promises.

## Minor observations

- Replace `Click Delete again` with `Activate Delete again` or `Press Enter again` so keyboard users are not treated as an exception.
- `Maintenance` is accurate but vague. `More file actions` is more immediately recognizable.
- The heavy workflow-scroll focus outline is a monitoring risk, not a confirmed defect; no current evidence shows it obscuring content.
- Earlier findings around content-obscuring focus borders, friendly folder authority, outcome-led Session Git entry, local Save Copy, Back copy, key guidance, commit focus, and compact component fit are resolved.

## Questions to consider

- Should the immediate implementation sequence be TASK-15502, TASK-15503, then TASK-15504, keeping all P2 polish behind those blockers?
- After the bounded Reload confirmation ships, should TASK-399.8.2 proceed directly into the full Base, Draft, and Disk comparison flow?
- Should push review distillation and semantic high-stakes styling be one follow-up polish task, or stay separate so each can be verified independently?
