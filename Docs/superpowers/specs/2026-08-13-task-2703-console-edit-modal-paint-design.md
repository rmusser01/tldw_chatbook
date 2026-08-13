# TASK-2703 Console Edit Message Action Paint Design

**Status:** Approved for planning

**Task:** TASK-2703

## Problem

The Console Edit Message modal reserves space for **Cancel**, **Save**, and
**Edit & resend**, and the buttons remain focusable and actionable, but the
three-button USER shape does not paint them in real terminals. Existing
headless checks reported mounted widgets with non-zero regions, but did not ask
whether the compositor actually included the buttons.

A real-bundle rendered-frame probe at both reported terminal sizes established
the root cause. The non-USER Cancel/Save shape paints normally. In the USER
shape, the longer explanation wraps while the editor remains fixed at 16 rows;
the three-row action region plus its top margin then extends beyond the fixed
modal's content region. Hit-testing each button center returns the opaque modal,
not the button, and no button cells reach the compositor. Adding scoped button
colors, borders, and focus styling does not repair that overflow.

The application-wide `Button { border: none; }` rule may still affect the live
button face or focus cue after containment is fixed, but that is a separate
hypothesis. It will be changed only if a separate rendered-frame or live-driver
RED proves it necessary.

## Goals

- Keep every Edit Message action fully contained and painted in the final frame.
- Preserve a visible, non-color-only focus cue for keyboard navigation.
- Cover both modal shapes: USER messages with three actions and non-USER
  messages with Cancel and Save.
- Verify the result through the rendered frame and real terminal drivers.
- Remove the obsolete User Guide workaround after live verification.

## Non-goals

- No outer-modal resizing, copy changes, DOM reordering, or handler changes.
- No global Button redesign and no changes to other Console modals.
- No custom widget, new state, worker, dependency, or configuration.

## Chosen Design

Keep the modal's existing fixed outer dimensions, context copy, action row,
button variants, DOM order, and event behavior. Replace only the editor's fixed
16-row height with flexible remaining-space sizing plus an eight-row minimum. The
context and error rows retain content-driven height; the fixed action row stays
the modal's final child. This lets the editor yield the rows consumed by wrapped
USER context while preserving the larger editor in the shorter non-USER shape.

The minimal production correction belongs in
`ConsoleEditMessageModal.DEFAULT_CSS`, because the fault is this widget's inner
layout rather than a shared application rule. If the now-contained buttons or
focus cue still fail a separate rendered-frame/live RED under the final bundle,
add the smallest app-tier selector scoped to `#console-edit-message-actions` in
`tldw_chatbook/css/components/_agentic_terminal.tcss`. The required fallback
focus cue is `outline: heavy $ds-action-focus` together with the established
focus foreground/background; the focused label must remain painted. Rebuild
`tldw_cli_modular.tcss` with the repository CSS builder after any source-module
change; never hand-edit the bundle.

## Rendering and Accessibility Contract

For both modal shapes:

- every full action region must remain inside the action row, modal content
  region, and viewport;
- compositor cells clipped to each button's own region must contain that
  button's label/style, rather than matching the same words in context prose;
- `app.get_widget_at(*button.region.center)` must return that button;
- Tab order must remain Cancel, Save, then Edit & resend when present;
- focusing each action must produce a compositor-visible difference in that
  button's cells and a non-color cue; if incumbent focus styling does not meet
  that contract, use the scoped heavy focus outline defined above;
- action/background and focused-action/background contrast must meet the
  repository's established 3:1 action-label floor;
- mouse and Enter activation behavior remains covered by incumbent tests.

Representative checks will use the reported 200x50 and 235x52 sizes. A smaller
supported viewport may be included as a stress case only if it does not expand
the production scope.

## Verification

1. Add a test harness that loads the real generated application stylesheet,
   mounts `ConsoleEditMessageModal` in USER and non-USER forms, and asserts on
   button-region compositor cells, hit-testing, and containment rather than
   computed geometry or whole-frame word presence alone.
2. Capture the USER RED against the current fixed editor: action regions extend
   outside the modal content region, hit-testing returns the modal, and button
   cells are absent. The non-USER control must remain green.
3. Add only the flexible editor sizing correction and rerun the same assertions
   to GREEN.
4. After containment is green, independently test ordinary and focused button
   cells under the real bundle. Add scoped app-tier paint/focus rules only for
   any remaining RED.
5. Mutation-check layout containment separately from any paint/focus rule: the
   fixed 16-row editor must fail USER containment, while removing a required
   focus cue must fail the focused-cell oracle.
6. Run incumbent modal behavior, Console wiring, CSS bundle-sync, Ruff, and
   compilation checks.
7. Run a temporary, untracked harness through two real drivers: tmux and an
   actual non-tmux terminal/TTY. Preserve SGR styling (`tmux capture-pane -e`
   or the driver's equivalent), Tab through Cancel → Save → Edit & resend, and
   compare the targeted button cells for the focus palette and non-color cue.
   Verify the USER and non-USER labels, containment, and mouse-visible hit
   targets. A generic whole-frame diff is not sufficient. Remove the harness
   afterward.
8. Remove the TASK-2703 workaround from
   `Docs/User_Guide/console/branching-and-rewind.md` only after live evidence is
   green.

## Failure Handling and Privacy

This change adds no runtime failure path, persistence, logging, or user-data
flow. A missing or stale bundle is caught by the existing bundle-sync gate.
Temporary live verification uses synthetic message bodies only. Before import
or launch it sets scratch `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
`XDG_CACHE_HOME`, `TLDW_CONFIG_PATH`, and a scratch `[paths].data_dir`; model
catalog and network refresh are disabled. The relevant real config/profile/data
paths are fingerprinted before and after both live-driver runs to prove that
isolation held.

## Alternatives Rejected

- **Paint-only app-tier rule:** cannot composite an action row that lies outside
  its opaque modal content region.
- **Global Button repair:** broader than the defect and risks changing every
  application surface.
- **Custom action widgets:** duplicates native Button semantics and adds
  unnecessary accessibility and event surface.

## ADR Check

ADR required: no

ADR path: N/A

Reason: this is a localized visual bug fix using the existing stylesheet and
design-system boundary. It changes no storage, ownership, service contract,
security boundary, dependency, runtime policy, or long-lived UX structure.
