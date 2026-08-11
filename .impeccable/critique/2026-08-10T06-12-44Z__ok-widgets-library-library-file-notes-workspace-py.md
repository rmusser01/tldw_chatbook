---
target: the local File Notes page and its elements
total_score: 25
p0_count: 0
p1_count: 2
timestamp: 2026-08-10T06-12-44Z
slug: ok-widgets-library-library-file-notes-workspace-py
---
## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of System Status | 3 | Root, save, conflict, action, and Git states are explicit; empty and disabled states are less explanatory. |
| 2 | Match System / Real World | 3 | Folder and file language is natural, but Protect and Session Git assume prior product or Git knowledge. |
| 3 | User Control and Freedom | 3 | Escape, Back, Restore, Reload, Save Copy, cancel paths, and guarded transitions are strong; general undo is absent. |
| 4 | Consistency and Standards | 2 | Main workspace buttons and trees inherit destructive focus outlines while the adjacent Git panel uses content-safe focus styling. |
| 5 | Error Prevention | 3 | Autosave, conflict guards, read-only leases, two-step deletion, protected checkpoints, and commit reviews prevent common losses. |
| 6 | Recognition Rather Than Recall | 2 | Eight editor actions and a path field shared by New, Move, and Save Copy require interpretation and memory. |
| 7 | Flexibility and Efficiency | 3 | Search, keyboard traversal, responsive panes, retained state, and batch Git actions serve power users. |
| 8 | Aesthetic and Minimalist Design | 1 | Neon focus chrome obscures content, disabled ghosts remain visible, and eight equal actions flatten hierarchy. |
| 9 | Error Recovery | 3 | Conflict, interrupted saves, tombstone restore, uncertain Git recovery, and action-specific status copy preserve work. |
| 10 | Help and Documentation | 2 | The guide is strong and contextual purpose copy exists, but in-surface help is uneven and often loses to jargon or clipping. |
| **Total** | | **25/40** | **Acceptable: significant improvements needed** |

## Anti-Patterns Verdict

**Does this look AI-generated?** No. It is unusually bespoke and serious about local authority, retained drafts, Git safety, and keyboard workflows. The failure is narrower: the cyberpunk styling collapses into bright rectangular focus chrome. The neon frame becomes louder than the file or document state and makes a trusted workbench feel like a decorative terminal skin.

**Deterministic scan:** The required detector was attempted against `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`. The sandboxed run failed with an EPERM realpath error; the elevated retry reached the entrypoint but failed with `bundled detector not found`. It emitted no JSON, counts, rule names, or file findings, so there are no detector false positives to classify.

The source evidence is nevertheless deterministic. `css/core/_reset.tcss` documents that Textual outlines paint over perimeter cells, then applies `*:focus { outline: solid $ds-focus-accent; }`. File Notes gives its root and toolbar buttons a one-row, borderless geometry without a local focus override. The adjacent Session Git panel explicitly replaces that outline with background, foreground, bold, and underline. The inconsistency identifies the mechanism and the repository's established safe replacement.

**Visual overlays:** No browser overlay is available. File Notes is a terminal-native Textual widget with no HTML counterpart, so browser injection would be misleading. Assessment A instead mounted the production Library screen and stylesheet through a Textual harness at 120x40 and 160x45, then inspected rasterized screenshots. Both sizes showed the focused Tree's cyan frame dominating the pane and replacing edge cells. At 120x40, purpose copy and file labels also clipped.

## Overall Impression

File Notes has excellent recovery engineering inside an under-resolved visual shell. Its single biggest opportunity is to replace perimeter focus chrome with content-safe focus cues, then let the current task, selected file, and save state carry the hierarchy.

## What's Working

- **Local authority is honest.** The Database and Files source distinction, direct-folder purpose copy, linked root status, and retained Library frame make it clear that files on disk are authoritative.
- **Recovery is unusually strong.** Autosave states, conflict handling, retained drafts, two-step delete, Restore, Reload, Save Copy, protected checkpoints, and guarded navigation reduce real data-loss risk.
- **The responsive model is intentional.** The workspace switches between navigator and editor below 80 mounted columns and measures toolbar labels before stacking them. It does not rely on fluid text or accidental wrapping.

## Cognitive Load

Five of eight checks fail, which places the current surface in the high-load band.

- **Single focus fails:** global navigation, Library rail, source selector, root authority row, navigator, editor, two toolbars, and Git entry all compete.
- **Chunking fails:** the first toolbar has five actions and the editor exposes eight in total.
- **Grouping passes:** Library, navigator, editor, and Git phases are structurally recognizable.
- **Visual hierarchy fails:** focus chrome outranks selection and document state.
- **One thing at a time fails:** irrelevant disabled actions remain visible before a file is selected.
- **Minimal choices fails:** the editor exposes eight actions; Git preparation can expose up to nine.
- **Working memory passes narrowly:** selected path, root, save status, and conflict context remain visible, although the shared path field still requires users to remember which action will consume it.
- **Progressive disclosure passes at the safety layer:** commit and push are phased, but list-level Git and editor actions need more distillation.

## Emotional Journey

Entry is reassuring. Files is explicitly selected, direct-folder ownership is stated, and the linked root is visible. The first emotional valley arrives as soon as the user tabs into the task: a bright frame can replace button labels or edge rows while eight mostly unavailable actions remain on screen. Editing then becomes reassuring again because Dirty, Saving, Saved, Conflict, and recovery actions are explicit. Git is a safety peak and a comprehension valley: its policy is rigorous, but the volume of repository, staging, transport, and recovery language makes it feel procedural rather than cozy. Clear save and Git results provide a strong ending.

## Priority Issues

### [P1] Focus outlines overwrite the content they are meant to clarify

**Why it matters:** Keyboard focus is clearest exactly when labels and rows become least readable. One-row buttons can turn into empty outlined boxes; focused Trees can lose perimeter cells. This is an accessibility and task-completion defect, not visual preference.

**Evidence:** `css/core/_reset.tcss:6-18`, `css/components/_lists.tcss:108-118`, `library_file_notes_workspace.py:407-416`, `library_file_notes_workspace.py:480-490`, and the safe precedent at `library_file_notes_git_panel.py:688-720`.

**Fix:** Give the File Notes workspace a single content-safe focus vocabulary. Suppress the global outline on its one-row buttons, Trees, editor, and meaningful scroll surfaces. For buttons, use the Git panel's background plus bold underline treatment. For Trees, recolor the cursor row with the sanctioned focus foreground/background and preserve the row text and click metadata. Inputs and TextArea already have focused borders, so clear the redundant outline. Replace heavy Git scroll outlines with a quiet surface change and scrollbar cue. Add render tests proving first/last Tree rows and button labels survive focus.

**Suggested command:** `impeccable quieter`

### [P1] Eight editor actions form a wall of disabled ghosts

**Why it matters:** Before a file is selected, users must scan New, Move, Delete, Restore, Protect, Reload, Save Copy, and Refresh to discover that only a small subset applies. Low-contrast disabled labels look broken rather than intentionally unavailable.

**Evidence:** All actions compose at `library_file_notes_workspace.py:836-850`; applicability is expressed only through `disabled` at `library_file_notes_workspace.py:3348-3397`.

**Fix:** Project actions by state. The empty editor should emphasize New and keep Refresh with the navigator. An active file can reveal Move, Delete, and Protect. Dirty or conflicted state can reveal Reload and Save Copy. A tombstone can replace the normal group with Restore. If an unavailable action must remain, append a short text reason rather than relying on dimming or a tooltip. Give destructive and recovery actions separate semantic groups.

**Suggested command:** `impeccable distill`

### [P2] Authority copy clips at the width where reassurance matters most

**Why it matters:** At 120x40 the direct-folder sentence truncates and the linked root becomes middle-elided telemetry. The interface technically exposes authority but weakens the human-readable answer to "where am I editing?"

**Evidence:** One-row nowrap clipping at `library_file_notes_workspace.py:375-405`, composed copy at `library_file_notes_workspace.py:765-795`, and root fitting at `library_file_notes_workspace.py:1327-1334`.

**Fix:** Use shorter semantic copy at moderate widths, such as "Files edits this folder directly. Sync mirrors files into Library." Lead with a friendly folder label, for example "Local folder: Research Notes", and reserve the full path for Details. Permit two lines before clipping at narrow supported widths.

**Suggested command:** `impeccable adapt`

### [P2] Session Git is safe but insufficiently progressive

**Why it matters:** The guarded commit and push mechanics are excellent, but the list surface can present up to nine actions plus repository, status, scope, guide, rows, and selection telemetry. Operators can learn it; researchers and students may read it as another application embedded in the navigator.

**Evidence:** `library_file_notes_git_panel.py:1138-1237`.

**Fix:** Keep Back and Refresh persistent. Show Trust only before trust exists. Show one contextual row action at a time. Put Stage all and Unstage all behind a compact bulk disclosure. Reveal Commit only after staged changes exist and Push only after a qualifying local commit. Move recurring key guidance into the footer after first use.

**Suggested command:** `impeccable distill`

## Persona Red Flags

**Alex, power user:** Search, Escape, pane traversal, retained state, and bulk Git staging are strong. Tabbing through the global shell, rail, source strip, root actions, inputs, eight editor actions, and Git is long. The focus box identifies a pane perimeter instead of the exact row or action Alex will execute, and main editor actions lack direct visible accelerators.

**Sam, keyboard-only or low-vision user:** The global outline can overwrite Tree and one-row button content precisely when focus must be readable. Focus vocabulary is inconsistent: buttons in Git use background plus bold underline; main buttons inherit a frame; Trees combine an outer frame with a cursor style; inputs can receive both a border recolor and an outline. Disabled action contrast is weak. The positive is that primary flows are keyboard-reachable and most critical states are text-labeled.

**Jordan, first-timer:** Session Git, staging, repository trust, Protect, and the shared relative-path field assume prior knowledge. The empty editor does not name the recommended next action, while eight toolbar labels compete beneath it. The Files-versus-Sync explanation helps, but clips at 120 columns.

## Minor Observations

- No target-local border animation or transition was found. The reported animation is most plausibly focus movement and repainting, not authored motion. Polling and autosave timers do not animate borders.
- The navigator's neutral right divider is structurally defensible, but it amplifies the "boxes everywhere" impression once a second bright focus frame appears.
- "Idle" beneath "No file selected" is technically correct but not useful. "Select or create a file" would better answer the next question.
- The current tests are strong on retention, routing, fitting, and responsive geometry. They lack visual assertions that focused perimeter content remains intact and disabled labels remain readable.
- Product documentation is comprehensive, but the surface should not require the guide to explain why Save Copy is disabled or how the path field relates to three actions.

## Questions to Consider

- If bright accent color is reserved for state, authority, and recovery, why does merely focusing the file tree receive the strongest signal on the screen?
- When no file is selected, are New and Refresh the only decisions that deserve pixels?
- Should Session Git remain a permanent navigator action, or appear only after the current session has file changes?
- At 120 columns, does the full Library rail provide enough orientation to justify a second nested navigator, or should the rail compact while preserving a breadcrumb and Escape path?
