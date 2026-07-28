# File Notes Prepare Session UX Design

Date: 2026-07-28
Task: [TASK-1235](../../../backlog/tasks/task-1235%20-%20Polish-File-Notes-prepare-session-for-commit-UX.md)
Source critique: [File Notes Session Git acceptance critique](../../../.impeccable/critique/2026-07-28T15-38-30Z__ok-widgets-library-library-file-notes-git-panel-py.md)
Primary decision: [ADR-035](../../../backlog/decisions/035-file-notes-session-git-index-controls.md)
Conforms to: [ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md), [ADR-011](../../../backlog/decisions/011-chatbook-workbench-ui-system.md), and [File Notes disk authority](../../../backlog/decisions/029-file-notes-disk-authority.md)

## Summary

Refine the shipped File Notes Session Git view into a note-centered
`Prepare session for commit` workflow. The change fixes five observed usability
problems: focused controls losing their labels, rows surviving an authority
change, stale or clipped result feedback, Git-first row language, and competing
editor actions.

This is presentation and interaction-state repair. It does not change which
paths Chatbook records, trusts, stages, or unstages; how Git ownership is
proved; or where note and recovery data live.

## User outcome

A user can finish a work or study session, open one bounded preparation view,
understand which notes were created, edited, moved, deleted, or restored, and
stage only eligible session notes. The view clearly reports what happened and
reassures the user that Chatbook targeted only eligible session paths. The same
workflow remains legible and keyboard-operable in a 40-column terminal.

## Approved product choices

- The visible workflow title is `Prepare session for commit`.
- In a wide two-pane layout, the open note remains visible and editable while
  its action toolbars are collapsed or visually quiet.
- Successful feedback keeps the approved promise-plus-count form while
  limiting the promise to what Chatbook proves:
  `2 session notes staged; Chatbook targeted only eligible session paths.`
- The fix remains staging-only. Commit, push, remote, credential, branch,
  hunk, and full-repository controls remain outside Chatbook.

## Scope

The work covers the existing File Notes Git panel, its workspace presentation
state, the incumbent Library/editor styling needed for responsive action
layout, and focused tests. It preserves the existing process-session owner and
Git service contracts.

The work does not add direct Git accelerators, an actionable-only filter, a
general activity log, new shared UI infrastructure, polling, persistence, or
support for additional repository states. Those are not needed to correct the
observed acceptance failures.

## Approach

Repair the current stable panel and workspace composition using existing
Chatbook focus, two-line list-row, semantic-status, and responsive-layout
patterns.

Two broader alternatives are rejected:

- Replacing the File Notes navigator with a new workbench framework would
  broaden a focused usability repair and duplicate incumbent primitives.
- Hiding the editor pane entirely would remove useful session context and make
  switching back to edit feel like a route change. The editor stays mounted;
  only competing structural actions become quiet.

## Information hierarchy

The preparation view is ordered as:

1. a one-line return control and the title `Prepare session for commit`;
2. repository authority and branch;
3. the persistent scope statement `Session paths only · stages complete file
   state`;
4. a compact keyboard guide;
5. the scrollable session-note list;
6. an always-visible current-status and last-action strip;
7. selected-note and bulk actions.

Only the session-note list scrolls. The title, scope, feedback, and action
regions remain outside that scroll area so the latest result and recovery
control cannot fall below the viewport.

The existing compact navigator entry may remain `Session Git (N)`. `N`
continues to mean coalesced current-process session groups, not all repository
changes and not necessarily all actionable notes.

## Return and keyboard behavior

Use the neutral label `Back to navigator` because the preserved destination may
be the file tree or search results. It retains the existing behavior: return to
the prior navigator mode, restore focus to the Session Git entry, and leave the
editor mounted.

The visible guide reads:

`Up/Down Select | Tab Actions | Enter Run | Esc Back`

The existing key contract remains unchanged. `Up` and `Down` change the
selected row, `Tab` and `Shift+Tab` traverse visible actions, `Enter` activates
only the focused action, and `Escape` returns to the prior navigator.

Every one-line control uses a same-row focus treatment: contrasting background
and foreground plus bold or underline. It uses no outline or border that can
replace the label glyphs. Focused Back, Refresh, Stage, Stage update, Unstage,
Stage all, and Unstage all labels must remain readable at 40 and 70 columns.

## Session-note rows

Rows lead with the note change rather than the Git implementation state:

```text
EDITED   study/hci.md
READY TO STAGE · Git: unstaged
```

```text
MOVED    inbox/topic.md -> study/topic.md
UPDATE REQUIRED · stage the moved note before Chatbook can unstage it
```

The primary line contains an uppercase text verb and the display path or move.
The verb comes from the coalesced session group:

| Session change | Verb |
| --- | --- |
| Create | `CREATED` |
| Modify | `EDITED` |
| Move or chained move | `MOVED` |
| Delete | `DELETED` |
| Restore | `RESTORED` |

Compound histories project the existing coalesced group's latest note intent:

- create then delete renders `DELETED`;
- delete then restore renders `RESTORED`;
- edit then revert to `HEAD` renders `EDITED` with
  `NO ACTION · matches HEAD`;
- chained moves render `MOVED` from the original source to the final
  destination.

These labels derive from existing group history and do not alter coalescing,
stable identity, topology, eligibility, or ownership.

The second line contains a plain-language semantic token followed, when useful,
by the precise Git state. Color may reinforce but never replace the token.

| Existing row state | User-facing token and explanation |
| --- | --- |
| Unstaged | `READY TO STAGE · Git: unstaged` |
| Staged by Chatbook | `STAGED · by Chatbook` |
| Staged with newer unstaged edits | `UPDATE AVAILABLE · newer note edits are not staged` |
| Staged with changed path lineage | `UPDATE REQUIRED · stage the moved note before unstaging` |
| Staged or partially staged externally | `BLOCKED · already staged outside Chatbook; manage this path in Git, then Refresh` |
| Clean | `NO ACTION · matches HEAD` |
| Ignored | `BLOCKED · ignored by Git; change the ignore rule or stage outside Chatbook, then Refresh` |
| Conflict | `BLOCKED · resolve the Git conflict outside Chatbook, then Refresh` |
| Unsupported index or repository state | `BLOCKED · <specific reason>; resolve it outside Chatbook, then Refresh` |
| Row-specific Git failure | `FAILED · <specific reason>; <exact retry or external recovery>, then Refresh` |

Long primary labels middle-elide within the row rather than forcing the Git
state and actions out of view. At widths that fit, the selected-note label
exposes the complete path or move; at the bounded `40×20` layout it retains
enough leading and trailing context to identify the note. The selected detail
also gives a concise blocking reason and recovery. This view does not add a
pointer-only disclosure.

Rows retain their existing stable identity and selection rules. This design
changes formatting only; it does not change grouping, eligibility, ownership,
or action mapping.

## Repository-authority transitions

Rows are displayable only under the repository authority that produced them.

- Rendering untrusted, changed-repository, unavailable, or non-repository
  states immediately clears visible rows, selected-row presentation, and
  action eligibility.
- Checking, stale, or failed refresh may retain prior rows only when the
  process owner still binds them to the same selected root and repository
  identity.
- A later successful status replaces the list and restores selection only when
  the same stable session group remains.

The process-session owner remains the sole authority. Its existing immutable
snapshot supplies the current selected-root binding, trusted repository
identity, exact session-change tuple, and Git status generation. The workspace
only compares and render-projects those owner-supplied values; it keeps no
independent repository identity, authority generation, or validity state. The
panel receives and renders the resulting state. The existing snapshot contract
already supplies every token needed by this repair, so no owner API extension
is required.

## Current status and last action

One fixed strip immediately above the actions separates current repository
freshness from the most recent action result. The first line is always current:

- `Status: CURRENT · READY — 2 can be staged · 1 can be unstaged.`
- `Status: CHECKING — Checking current session notes...`
- `Status: STALE — Session notes changed; refresh status before staging.`
- `Status: CURRENT · BLOCKED — Save conflict must be resolved before staging. Return to the editor.`
- `Status: STALE · ERROR — Git status failed. Retry Refresh.`
- `Status: TRUST REQUIRED`, `Status: UNAVAILABLE`, or
  `Status: UPDATING INDEX` for the existing authority and mutation states.

When an action has completed under the same authority and exact session-change
snapshot, a separate second line reports it:

- `Last action: STAGED — 2 session notes staged; Chatbook targeted only eligible session paths.`
- `Last action: UNSTAGED — 2 session notes unstaged; Chatbook restored only its owned session entries.`

Singular and plural forms are correct. A count means successfully affected
coalesced session-note rows, not endpoint paths or all repository files.
Existing clean, already-staged, skipped, and blocked counts remain available
after the promise when they materially explain a bulk result.

The last-action line is associated with the selected-root binding, repository
authority, and exact session-change snapshot already known by the workspace. It
survives the action's automatic postflight status refresh so users can read the
result. A later session mutation, root change, repository identity change, or
newer action invalidates it before the next render. A manual or automatic
refresh updates the independent current-status line and never overwrites it
with an old action summary.

The promise appears only after the existing service reports a certain,
successful checked postflight under the unchanged selected-root binding,
repository and `HEAD` identity, ownership contract, and action snapshot, with
at least one confirmed affected coalesced group. Nonzero, uncertain, mismatched,
or zero-effect outcomes show counts plus failure/recovery text and no promise.
The promise describes Chatbook's proven Git target/entry boundary; it does not
supersede the existing configured-filter and concurrent-external-index
disclosures.

Repository freshness and blocking/error outcome are orthogonal. A
same-authority refresh failure retains the prior row semantics as stale,
disables mutation, and puts `STALE · ERROR` plus recovery in this strip.
`FAILED` inside a row remains reserved for an existing row-specific failure.

Routine, success, stale, blocked, and error states use restrained semantic
styling, but the text token is authoritative. Recovery text names the next
action; vague instructions such as `settle the draft` are not used.

## Actions

Actions remain the exact ADR-035 set and action mapping. Presentation groups
them as:

- `Selected note: <path or move>`: Stage, Stage update, or Unstage when
  eligible;
- bulk controls with independent counts: `Stage all (S)` and `Unstage all (U)`.

`S` and `U` are separately derived from the existing stage-eligible and
unstage-eligible row sets; one shared eligible count is never shown. Labels use
sentence case. Hidden or disabled actions keep the existing eligibility rules.
Checking and mutation continue to disable index mutations without disabling
Back or editor input.

## Editor quieting and responsive layout

When Prepare session for commit is visible beside the editor in a wide layout:

- the path breadcrumb, save state, note body, cursor, and selection remain
  visible;
- editor typing and debounced autosave remain active;
- competing structural/action toolbars collapse;
- a compact muted note may explain that editor actions return with the
  navigator, without presenting a new action.

Returning to the file tree or search results restores the toolbar immediately.
The editor widget is never remounted solely for this mode change.

In a narrow one-pane layout, Navigator and Editor continue to alternate. The
preparation view does not compete with an off-screen editor. When Editor is the
visible pane, its normal actions use the existing responsive grouping and wrap
to additional rows when needed; labels such as `Protect` are never shortened or
clipped to one character.

At `40×20`, fixed-region text has a strict height budget: repository authority,
selected-note detail, current status, and last action each use at most two
lines and middle-elide bounded path/detail text. Recovery instructions are
written to fit that bound; arbitrary diagnostic detail remains sanitized and
bounded. Action rows stay horizontal while their full labels fit and stack only
when the actual panel width requires it. If vertical space tightens, only the
scrollable session list loses height. The current-status line, any last-action
line, Back, and currently eligible actions remain visible.

The workspace uses a presentation class on the existing stable composition. It
does not add a second mode authority, a generic collapse helper, or editor
recomposition.

## Empty, trust, and failure states

Each non-list state owns the list region and shows no stale rows:

- no session changes: `No notes changed in this Chatbook session.`;
- trust required: the existing process-only warning and `Trust and check
  status`, with Cancel initially focused;
- not a repository: `This notes folder is not in a Git worktree. Notes remain
  fully usable.`;
- Git unavailable or unsupported: the specific reason plus an exact next step;
  when Chatbook cannot recover it, name the required external action and the
  subsequent Refresh;
- error under the same authority: retained rows are visibly stale and all
  mutation actions remain disabled.

Back remains readable and available in every state. Trust wording continues to
disclose configured-filter execution and makes clear that declining does not
disable normal note editing.

## State and data boundaries

No database, filesystem, Git command, staging signature, trust, session
coalescing, service lifecycle, or mutation-gate behavior changes.

The only additional presentation information is:

- a note-change verb derived from the existing coalesced group;
- a semantic display token derived from the existing row state;
- whether an action result still matches the current root binding, repository,
  and exact session-change snapshot;
- whether the existing editor toolbar should render quietly for the current
  responsive mode.

These are view projections. They are not persisted and do not become new
authorities.

The production change stays local to
`library_file_notes_git_panel.py` and
`library_file_notes_workspace.py`. The Git panel reuses Chatbook's existing
height-one background/foreground plus bold-underline focus contract, two-line
Library row structure, semantic status badges, and its incumbent
`-stack-actions` resize class. Workspace toolbar quieting and narrow stacking
use local presentation classes in the widget's existing `DEFAULT_CSS`. No Git
service, owner, schema, global design-system CSS, or ADR changes are needed.

## Focused verification

Per the approved task boundary, verification for this repair is focused. It
adds no broad CI, network, long-soak, or performance gate and does not run the
repository's multi-hour full suite. Targeted lint/format checks still cover
every changed production and test file.

Focused model/widget tests cover:

- every coalesced change kind renders the correct note verb;
- every existing Git state renders a non-color semantic token and the existing
  action eligibility;
- untrusted, repository-changed, unavailable, and non-repository transitions
  clear rows and selection;
- stale/error retention is allowed only for the same authority;
- a later session-change snapshot replaces an obsolete action result, while
  the action's postflight refresh preserves it beside an independently current
  status line;
- success messages use correct singular/plural counts and the checked
  Chatbook-target/ownership promise;
- focused one-line controls retain their complete labels.

Mounted Textual tests at `150×42`, `70×28`, `70×24`, and specifically `40×20`
cover:

- title, hierarchy, fixed feedback, selected/bulk groups, and keyboard guide;
- keyboard selection, action traversal, activation, Escape, and focus restore;
- editor actions quieting and restoration without editor remount or draft loss;
- narrow editor action wrapping with no clipped labels;
- checking, trust, empty, stale, blocked, failed, and non-repository states;
- wrapped authority, selected-path, recovery, and postflight copy leaving the
  current-status/action regions visible while only the list loses height.

One compact disposable-repository acceptance flow edits at least two session
notes, stages them in bulk, and confirms both the promise-plus-count feedback
and preservation of an unrelated externally staged path. It then unstages the
session notes and confirms the unrelated path remains staged.

The live terminal acceptance pass uses the same `150×42`, `70×28/24`, and
`40×20` sizes. At `40×20` it explicitly verifies visible postflight feedback,
the complete `Protect` label in Editor, and Escape returning to the retained
navigator entry with focus restored.

## ADR check

ADR required: no

ADR path: N/A; this task conforms to
`backlog/decisions/035-file-notes-session-git-index-controls.md`,
`backlog/decisions/033-application-session-state-ownership.md`,
`backlog/decisions/011-chatbook-workbench-ui-system.md`, and
`backlog/decisions/029-file-notes-disk-authority.md`.

Reason: this is a focused correction to presentation, focus, responsive
layout, and display-state invalidation under already accepted ownership and Git
safety contracts. It introduces no storage/schema, sync, data ownership,
service boundary, security, dependency, or long-lived application-structure
decision.
