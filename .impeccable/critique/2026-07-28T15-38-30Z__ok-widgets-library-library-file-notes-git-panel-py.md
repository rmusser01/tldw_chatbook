---
score: 25
p0: 0
p1: 4
p2: 1
verdict: functional-pass-ux-conditional
timestamp: 2026-07-28T15-38-30Z
slug: ok-widgets-library-library-file-notes-git-panel-py
---
Method: dual-agent (A: Wegener the 2nd · B: Harvey the 2nd)

# File Notes Session Git — senior UX/HCI acceptance critique

## Verdict

**Functional acceptance: PASS. UX acceptance: CONDITIONAL.**

The feature safely completes its promised core loop: Chatbook edits are written to disk, only current-process File Notes paths are offered, selected and bulk staging work, and unrelated index state survives both Stage all and Unstage all. It is safe enough to use now by a Git-literate user.

It is not yet polished enough to feel like the primary notes-management surface for a mixed-audience user base. The largest gaps are authority-state presentation, focused-control legibility, status visibility at small terminal heights, and translation from Git internals into note-centered language.

Design-specificity verdict: **behaviorally bespoke; visually generic.** The safety policy is distinctly Chatbook. The presentation still reads like a raw Git sidebar.

## Acceptance journeys exercised

| Journey | Result | Evidence |
|---|---|---|
| Open a tracked Markdown note and edit only its body | Pass | Body edit autosaved; `Saved` returned without a manual save action. |
| Preserve YAML frontmatter byte-for-byte while editing the body | Pass | Git diff contained one appended body line; frontmatter had no diff. |
| Reflect session changes near real time | Pass | `Session Git (0)` advanced to `(1)` and `(2)` after autosaves. |
| Cancel repository trust safely | Pass | Cancel received initial focus; no status/staging ran. |
| Accept process-scoped trust | Pass | Canonical repository and branch appeared; status populated. |
| Stage and unstage a selected path | Pass | Git index changed only for `notes/inbox.md`, then returned to unstaged. |
| Stage all and unstage all session paths | Pass | Both edited notes moved into and out of the index as a group. |
| Preserve unrelated external index state | Pass | Externally staged `unrelated.md` remained staged after Chatbook Unstage all. |
| Exclude repository dirt from a prior/fresh Chatbook process | Pass | A new process over the dirty repo showed `Session Git (0)` and “No current-session Git changes.” |
| Retain search state across Session Git | Pass | Query `recognition` and its single result returned intact after Back. |
| Explain a non-Git notes root | Pass with UX issue | “Selected File Notes root is not in a Git worktree” appeared, but the focused Back control lost its visible label. |
| Adapt across terminal sizes | Mixed | 150, 70, and 48 columns remained usable; at 40×20 the final status line clipped and the editor toolbar clipped `Protect` to `P`. |
| Keyboard Back from the change list | Inconclusive | Real tmux ignored Escape twice; an exact mounted Textual probe returned to Files and restored focus. Treat as a physical-terminal regression-test gap, not a confirmed defect. |

## Nielsen heuristic score

| # | Heuristic | Score (0–4) | Evidence |
|---|---|---:|---|
| 1 | Visibility of system status | 2 | Many states exist, but the sole status line is muted, can retain an obsolete action summary, and clips at 40×20. |
| 2 | Match between system and real world | 2 | Repository authority is explicit; `HEAD`, index ownership, lineage, and “Stage update” are not translated for note-centric users. |
| 3 | User control and freedom | 3 | Cancel-first trust, exact Unstage, Back, retained context, and safe mutation gates are strong; focused Back legibility is not. |
| 4 | Consistency and standards | 2 | Behavior is internally consistent, but File Notes bypasses adjacent Library section, toolbar, badge, and status conventions. |
| 5 | Error prevention | 4 | Trust, repository revalidation, stale gates, autosave flush, path closure, and exact eligibility are excellent. |
| 6 | Recognition rather than recall | 2 | State is textual, but rows are compound strings, actions appear/disappear, shortcuts are hidden, and change verbs are omitted. |
| 7 | Flexibility and efficiency | 3 | Selected and bulk actions, stable row identity, keyboard navigation, and retained context are strong; accelerators/filtering are absent. |
| 8 | Aesthetic and minimalist design | 2 | Clean, but flat hierarchy, large dead zones, wrapped authority copy, and competing editor controls hurt scanability. |
| 9 | Help users recognize, diagnose, and recover from errors | 3 | Problems are usually named and Refresh/Back exist; recovery copy is sometimes vague or technical, and stale rows can cross authority states. |
| 10 | Help and documentation | 2 | Scope and trust explanations exist; there is no visible keyboard legend, glossary, or plain-language explanation of staging states. |
|  | **Total** | **25/40** | **Core-safe, presentation-incomplete.** |

## Cognitive load

**High: 6 of 8 load checks fail.**

- Single focus fails: wide mode keeps an editable note and its toolbar active while Session Git owns the navigator.
- Chunking fails: path, state, and disabled reason are one wrapping sentence.
- Hierarchy fails: there is no visible `Session Git` title, and repository, scope, rows, actions, and result feedback have similar weight.
- One-thing-at-a-time fails: note editing and Git decisions remain equally available.
- Minimal choices fails: a complex row can expose Back, Refresh, selected Stage/Unstage, and two bulk actions alongside the editor’s actions.
- Working memory fails: users must remember the meaning of session ownership, external staging, lineage, complete-file staging, and Stage update.
- Grouping passes structurally: authority, list, selected actions, bulk actions, and status are separate regions.
- Progressive disclosure passes: Session Git is a separate navigator mode and trust is gated.

## Emotional journey

1. **Entry — confidence:** `Session Git (N)` gives the user a bounded set and makes the session model feel manageable.
2. **Trust — appropriate anxiety:** The warning correctly explains configured Git-filter risk. Process-only trust, canonical path, safe initial focus, and Escape recovery rebuild confidence.
3. **Status — comprehension valley:** The user lands in a technically accurate but flat list. A researcher may feel they have left Chatbook and entered a lower-level Git tool.
4. **Mutation — reassurance:** “in progress,” disabled actions, and exact result counts reduce fear of accidental index changes.
5. **Completion — mechanically safe:** The actual preservation behavior is excellent, but success copy does not restate the most reassuring promise: unrelated repository state was left untouched.

## What is already strong

- **The safety model is excellent.** Trust is process-scoped; the canonical repository and branch are visible; complete-file staging is disclosed; stale and mutation states gate action; unrelated index state is preserved.
- **The data loop is trustworthy.** Autosave, exact frontmatter preservation, disk state, session ownership, and Git index behavior aligned during live use.
- **Context retention is excellent.** Search query/results, expanded folders, open editor, and session rows survive navigator transitions and resize changes.
- **State coverage is deep.** Ready, empty, checking, stale, unavailable, error, conflict, externally staged, clean, moved, and owned states all have policy support.

## Priority findings

### P1 — Focus can erase the only visible recovery label

**Observed:** In both a fresh-session empty state and a non-Git unavailable state at 70 columns, focus moved to Back as intended, but the control rendered as only `┏━━━━━━━━━━━━━━━┓`; “Back to Files” was not visible. This is the exact state where Back is the primary or only recovery action.

**Cause:** Buttons are forced to one terminal row while the focus treatment uses a heavy outline (`library_file_notes_git_panel.py:160–171`). The workspace intentionally focuses Back when there are no visible rows (`library_file_notes_workspace.py:1955–1967`).

**Improve:** Do not combine a one-cell button with a box-like outline. Use reverse video/background/text style on the same row, or allow three rows where a full outline is required. Add visual assertions for focused Back, Refresh, Stage, and Unstage at 40/70 columns.

### P1 — Repository authority can change while old rows remain visible

**Observed in a mounted state probe:** Ready, untrusted, and unavailable states each retained the same old row in both the model and visible list.

**Cause:** `render_untrusted()` and `render_unavailable()` update authority/status but do not clear or hide `_rows`; only `render_status()` replaces rows (`library_file_notes_git_panel.py:319–401`).

**Why it matters:** Even with actions disabled, paths from repository A can appear under authority text for repository B or “unavailable.” That undermines the feature’s strongest safety concept.

**Improve:** Clear rows and selection whenever repository identity is untrusted, changed, or unavailable. Retain rows only for checking/stale/error states that are proven to belong to the same repository identity.

### P1 — Result and recovery feedback is too easy to miss or misread

**Observed:** After one path was unstaged, a second path was edited. Reopening Session Git correctly showed two rows, but the footer still said `Unstaged 1 · clean 0 · blocked 0`. At 40×20, the final result wrapped below the viewport, leaving `Unstaged 2 · … ·` without its blocked count.

**Cause:** The retained `_git_action_detail` is reapplied after status rehydration (`library_file_notes_workspace.py:1119–1121`, `1234–1243`), and all routine/success/error feedback shares `$text-muted` at the bottom (`library_file_notes_git_panel.py:126–130`).

**Improve:** Give feedback a stable, always-visible strip immediately above actions. Reset or timestamp an action summary when the session/status generation changes. Use explicit text tokens—`READY`, `STAGED`, `STALE`, `BLOCKED`, `FAILED`—plus restrained semantic styling. Replace “settle the draft” with an exact next step.

### P1 — Rows show Git state but omit the note change users care about

**Observed/source evidence:** Each row flattens `path · Git state — disabled reason` into one Static. The model already knows whether the note was created, modified, moved, or deleted, but that change verb is not shown. At narrow widths, technical state/reason strings wrap heavily.

**Improve:** Lead with note intent, then expose Git detail:

`Edited  study/hci.md  [Ready to stage]`

Use a second line or selected-row detail for disabled reasons, middle-elide long paths, and preserve full text on focus. Pair plain language with precise Git language rather than removing the latter.

### P2 — The Git task competes with editor controls and lacks keyboard signposting

**Observed:** Wide mode can display up to 15 actions across Session Git, the editor toolbar, and root selection. At 40 columns the editor’s first toolbar clips `Protect` to `P`. The Git panel exposes no visible keyboard contract. `Back to Files` also returns to search results when search was the prior mode.

**Improve:** Add group labels (`Selected note`, `All eligible (N)`), visually quiet the editor toolbar while Session Git owns focus, stack/wrap the editor toolbar at the same narrow breakpoint, and add `↑↓ Select · Tab Actions · Enter Run · Esc Navigator`. Rename Back dynamically to `Back to search results` or use neutral `Back to navigator`.

## Persona red flags

### Alex — terminal/Git power user

- Likes the exact index preservation, bulk actions, stable selection, and fast retained state.
- Will miss direct accelerators for refresh/stage/unstage and an actionable-only filter.
- Will find the variable Tab order and wrapping compound rows slow in a large session.

### Sam — keyboard, low-vision, or screen-reader user

- Benefits from text labels, non-color-only state, safe trust focus, and strong mutation prevention.
- Is blocked when a focused one-line button loses its label.
- May not learn that status changed because `Static.update()` has no explicit announcement contract and feedback sits at the bottom.
- Needs physical-terminal/screen-reader validation; browser ARIA tests do not apply to Textual.

### Priya — researcher/student with modest Git knowledge

- Understands “edited/moved/deleted note” better than `HEAD`, index ownership, or lineage.
- May overestimate the trust warning’s risk because the dialog does not say that Cancel leaves normal note editing available.
- Needs success feedback that says the thing she cares about: “2 session notes staged; unrelated repository changes were not touched.”

## Minor observations and upgrades

- `Session Git (N)` counts coalesced session lineages, not necessarily actionable unstaged changes. Prefer `Session paths (N)` or clarify the count.
- Use sentence case consistently: `Stage all`, `Unstage all`.
- Split repository and branch across purposeful lines instead of allowing an arbitrary wrap.
- Empty file trees and zero-result searches need explicit empty copy; search exceptions currently look like zero matches.
- The trust action is a warning/approval, not a destructive/error action.
- Detached, unborn, and unavailable HEAD labels lack mounted visual coverage.
- For large sessions, add an `Actionable only` toggle or filter only after real use shows scanning friction; do not make it part of the immediate fix wave.

## Recommended sequence

1. Fix focused one-line control legibility and add visual assertions.
2. Clear rows across repository-authority transitions.
3. Make the status/recovery strip stable, current, visible, and semantic.
4. Redesign rows around note-change verbs plus state badges.
5. Reduce action competition and add keyboard signposting/narrow toolbar wrapping.

## Product questions

1. **How should the surface be framed?**
   - `Prepare this session for commit` — note-centered and outcome-oriented. **Recommended**
   - `Session Git` — concise and appropriate for expert users.
   - Dual label: `Prepare session` with `Git staging` as secondary copy.

2. **What should happen to the editor toolbar while Session Git is active in wide mode?**
   - Keep the note visible but visually quiet or collapse its actions. **Recommended**
   - Keep the current fully active editor and toolbar.
   - Hide the editor entirely until the user returns to the navigator.

3. **What should successful bulk feedback emphasize?**
   - Promise plus count: `2 session notes staged; unrelated repository changes untouched.` **Recommended**
   - Compact counts only.
   - A detailed per-path activity log.

## Run notes

- Resolved target: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/file-notes-move-tombstone/tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`
- Target slug: `ok-widgets-library-library-file-notes-git-panel-py`
- Ignore list: `.impeccable/critique/ignore.md` absent; no ignored findings.
- Assessment independence: Assessment A remained isolated from detector/Assessment B. Synthesis waited for A to complete.
- Detector: exact output `[]` (0 findings). This is non-probative because Impeccable’s detector does not scan Python/Textual semantics.
- Browser applicability: not applicable. This is a native Textual TUI with no DOM, browser route, or overlay target. No live web server, visibility mutation, or overlay injection was used.
- Runtime method: real Textual surface in an isolated tmux server, disposable real Git repository, 150×42, 70×28/24, 48×24, and 40×20 viewports; one exact mounted keyboard probe.
- Automated fallback evidence: Assessment B ran `Tests/UI/test_library_file_notes_git.py` — 49 passed in 10.78 seconds.
- Cleanup: all three TUI processes exited, the dedicated tmux server closed, and both disposable `/private/tmp` fixture roots were removed.
- Accessibility boundary: keyboard completion and visible text/focus were exercised; terminal screen-reader announcement behavior and theme-specific contrast remain unverified.
