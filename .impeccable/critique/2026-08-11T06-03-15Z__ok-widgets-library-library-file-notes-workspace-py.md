---
target: the post-merge local File Notes page and its elements
total_score: 28
p0_count: 0
p1_count: 2
timestamp: 2026-08-11T06-03-15Z
slug: ok-widgets-library-library-file-notes-workspace-py
---
# File Notes UI critique

## Design Health Score

| Heuristic | Score (0–4) | Assessment |
|---|---:|---|
| Visibility of system status | 3 | Save and Git states are visible, but the labels do not explicitly say where normal edits persist. |
| Match with the real world | 3 | Folder and note concepts are familiar; Git terminology remains expert-oriented. |
| User control and freedom | 3 | Recovery, navigation, and staged Git actions provide control without hiding consequences. |
| Consistency and standards | 3 | Most labels and control treatments are consistent, with a few navigation-copy mismatches. |
| Error prevention | 4 | Scoped Git operations, confirmation behavior, and local draft preservation are unusually strong. |
| Recognition rather than recall | 2 | The page exposes too many peer actions and assumes knowledge of Session Git semantics. |
| Flexibility and efficiency | 3 | Keyboard flow and compact layouts are thoughtfully supported, though frequent file actions require extra traversal. |
| Aesthetic and minimalist design | 2 | The workbench is purposeful, but operational copy and equal-weight controls crowd the main task. |
| Error recognition and recovery | 3 | Recovery paths exist and are honest, but critical instructions can be compressed in small terminals. |
| Help and documentation | 2 | Contextual guidance exists, yet autosave authority and Git scope need plainer explanations. |
| **Total** | **28/40** | **Good, at the lower edge: structurally trustworthy but still cognitively dense.** |

## Anti-pattern verdict

The page passes the AI-slop check. It reads as a bespoke terminal workbench, not a generic dashboard or decorative AI interface. Its risk is the opposite: too much operational language and too many equally prominent controls.

The deterministic detector could not run because its bundled detector was missing. It returned no JSON, so this is unavailable evidence—not a zero-finding result. A browser overlay was not applicable: this is a Python Textual widget rendered into terminal cells, with no equivalent HTML DOM to inspect reliably.

## Overall impression

The redesign has materially improved the File Notes experience. Local authority is visible, focus behavior is content-safe, compact layouts are deliberately engineered, and Session Git uses honest scope and recovery semantics. The remaining opportunity is to make those underlying guarantees legible at a glance: normal editing should unmistakably say “saved to the local folder,” recovery instructions must remain complete at small sizes, and secondary file operations should stop competing with the primary writing task.

## What is working

- **Authority is honest.** The UI distinguishes linked-folder state, session-scoped Git behavior, and local recovery instead of implying cloud-style magic.
- **Progressive disclosure is substantive.** Advanced Git actions and recovery paths are available without permanently occupying the editing surface.
- **Compact behavior is intentionally designed.** Navigator/editor alternation and focus safeguards preserve the primary task in narrow terminals.
- **Risky actions are restrained.** Scoped paths, review-before-commit behavior, and confirmation states reduce accidental destructive work.
- **The visual character fits Chatbook.** It is terminal-native, efficient, and restrained rather than ornamental.

## Cognitive load

The page passes five of eight cognitive-load checks and fails chunking, visual hierarchy, and minimal choices. Load is moderate rather than severe, but it rises exactly where users most need confidence.

- A saved-note editor exposes six adjacent actions: New, Move, Delete, Protect, Reload, and Refresh.
- A dirty, conflict, or error state can expose seven by adding Save Copy.
- A ready Session Git flow asks the user to parse roughly five or six decisions across navigation, refresh, selection, bulk disclosure, commit, and push.
- The core workflow is recognizable, but maintenance, recovery, and destructive actions are not sufficiently separated by frequency or consequence.

## Emotional journey

Entry is reassuring: the linked local folder and note context establish ownership. The confidence valley appears during ordinary editing because `Idle`, `Dirty`, `Saving`, and `Saved` do not say that the file-backed folder is the persistence authority; the later appearance of `Save Copy` can make users wonder whether normal edits were ever saved. Session Git then raises complexity through specialist language, but its scoping, review, and recovery behavior rebuild trust once understood.

## Priority issues

### P1 — The local persistence contract is ambiguous

**Evidence:** `library_file_notes_workspace.py` composes the editor status and action row around lines 831–863. `_set_save_state` around lines 3163–3171 converts internal state to generic labels such as `Idle`, `Dirty`, `Saving`, and `Saved`. Recovery controls around lines 3460–3477 introduce `Save Copy` without distinguishing it from normal automatic persistence.

**Impact:** A user can see that something was saved without knowing whether it was saved directly to the linked local file, a database, or an internal draft. This weakens the page’s central promise and makes the recovery action harder to interpret.

**Recommendation:** Make authority part of every persistence state: `Auto-save: idle`, `Saving to local folder…`, `Saved to local folder`, and `Conflict: local draft preserved`. Rename `Save Copy` to `Save draft as copy` so it is unmistakably a recovery path.

**Impeccable command:** clarify.

### P1 — Critical Git recovery copy can be elided in compact terminals

**Evidence:** `library_file_notes_git_panel.py` constrains status, repository, selected-path, and action copy to two lines with hidden overflow and no wrapping around lines 630–649. `_fit_fixed_regions` around lines 1672–1682 applies two-line fitting. This is appropriate for telemetry but risky for warning, error, and recovery instructions.

**Impact:** At 40×20, the UI can preserve its geometry while withholding the end of the exact instruction needed to recover. The interface looks stable but may be operationally incomplete.

**Recommendation:** Continue fitting routine telemetry, but allow warning, error, and recovery content to wrap fully inside the scrollable surface. Never truncate the action or command required to recover.

**Impeccable command:** harden.

### P2 — File actions lack a clear frequency and consequence hierarchy

**Evidence:** The editor action row around lines 843–863, styled around lines 486–496, presents maintenance, creation, protection, and destructive actions at near-equal weight.

**Impact:** Frequent writing and navigation compete with occasional maintenance. Keyboard users must traverse up to seven controls, while less experienced users must evaluate all of them before acting.

**Recommendation:** Keep the primary row focused on the next likely action and place Reload/Refresh—and possibly Move/Protect—behind a `Maintenance` disclosure. Reserve semantic danger styling for Delete only when confirmation is armed.

**Impeccable command:** distill.

### P2 — Session Git describes its mechanism before its outcome

**Evidence:** The workspace entry is `Session Git (0)`. The panel later says `Session paths only · stages complete file state`, which is accurate but requires Git vocabulary before the benefit is clear.

**Impact:** New or occasional Git users must decode session paths, staging, and complete-file state before knowing why the tool is useful.

**Recommendation:** Lead with the outcome: `Review and commit only notes changed during this Chatbook session.` Keep the precise Git scope as secondary supporting copy.

**Impeccable command:** clarify.

## Persona red flags

- **Alex, the frequent keyboard user:** opening a note can lead to six or seven peer actions with no visible accelerators, increasing traversal for routine work.
- **Jordan, the local-first non-Git user:** generic save states and terms such as `stage`, `worktree`, and `complete file state` obscure otherwise strong local guarantees.
- **Sam, the accessibility-focused user:** focus behavior and complete labels are strong, but recovery copy fitting can hide essential information in the smallest supported viewport.

## Minor observations

- Use warning semantic color for the linked-root warning, not copy alone.
- Standardize `Back to navigator` and `‹ Navigator` into one navigation phrase.
- Replace pipe-separated key-guide copy with spacing or grouping that reads as guidance rather than telemetry.
- The exact-path Details dialog is justified; it protects the main hierarchy while keeping verifiable authority available.
- Wide Session Git appropriately hides unrelated toolbars and keeps the draft editor mounted.
- One independent mounted-test sequence observed a medium-width focus assertion failure in the commit-review footer, but the exact 17-test sequence passed on immediate parent rerun. Treat this as a P3 monitoring signal, not a confirmed defect; stress-repeat the transition before changing behavior.

## Questions to consider

- Should the persistence line describe the linked local folder as the authority in every state, including idle?
- Do Reload and Refresh earn permanent primary-row space, or do they belong under Maintenance?
- Should the entry point lead with `Review session changes` and retain `Session Git` as explanatory secondary text?
- At 40 columns, should routine telemetry yield vertical space whenever a complete recovery instruction needs it?
