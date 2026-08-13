# TASK-2703 Console Edit Message Action Paint Design

**Status:** Approved for planning

**Task:** TASK-2703

## Problem

The Console Edit Message modal reserves space for **Cancel**, **Save**, and
**Edit & resend**, and the buttons remain focusable and actionable, but their
paint can disappear in real terminals. Existing headless checks prove only that
the widgets are mounted, sized, and clickable; they do not prove that the final
application stylesheet paints button glyphs or a visible focus state.

The application-wide `Button { border: none; }` rule is loaded at app tier and
therefore outranks widget `DEFAULT_CSS`. The Edit Message modal currently has no
app-tier escape for its fixed action row, so relying on Textual's default button
face is not a stable rendering contract.

## Goals

- Paint every enabled Edit Message action in the real application stylesheet.
- Preserve a visible, non-color-only focus cue for keyboard navigation.
- Cover both modal shapes: USER messages with three actions and non-USER
  messages with Cancel and Save.
- Verify the result through the rendered frame and real terminal drivers.
- Remove the obsolete User Guide workaround after live verification.

## Non-goals

- No modal resizing, copy changes, DOM reordering, or handler changes.
- No global Button redesign and no changes to other Console modals.
- No custom widget, new state, worker, dependency, or configuration.

## Chosen Design

Add a narrowly scoped app-tier rule under
`tldw_chatbook/css/components/_agentic_terminal.tcss` for Buttons descended
from `#console-edit-message-actions`. The rule will explicitly provide a
painted surface, readable foreground, and visible edge using existing design
system tokens. The primary action will remain visually distinct, and a scoped
`:focus` rule will use the established focus foreground/background plus a
non-color cue such as underline or outline.

The selector will not change width, height, margin, content, order, variants,
or event behavior. The generated `tldw_cli_modular.tcss` bundle will be rebuilt
with the repository's existing CSS builder; it will not be hand-edited.

## Rendering and Accessibility Contract

For both modal shapes:

- every enabled action label must appear in the exported rendered frame;
- every action region must remain inside the action row, modal, and viewport;
- Tab order must remain Cancel, Save, then Edit & resend when present;
- focusing each action must produce a compositor-visible style difference and
  a non-color cue;
- action/background and focused-action/background contrast must meet the
  repository's established 3:1 action-label floor;
- mouse and Enter activation behavior remains covered by incumbent tests.

Representative checks will use the reported 200x50 and 235x52 sizes. A smaller
supported viewport may be included as a stress case only if it does not expand
the production scope.

## Verification

1. Add a test harness that loads the real generated application stylesheet,
   mounts `ConsoleEditMessageModal` in USER and non-USER forms, and asserts on
   exported SVG/compositor paint rather than computed geometry alone.
2. Capture RED before the app-tier rule: at least one action or required edge /
   focus paint assertion must fail for the existing stylesheet.
3. Add the minimal scoped source rule, rebuild the bundle, and rerun the same
   test to GREEN.
4. Mutation-check the paint and focus declarations independently.
5. Run incumbent modal behavior, Console wiring, CSS bundle-sync, Ruff, and
   compilation checks.
6. Run a temporary, untracked harness under tmux and a separate PTY with an
   isolated scratch profile. Capture both modal shapes and verify the exact
   labels and focused action paint. Remove the harness afterward.
7. Remove the TASK-2703 workaround from
   `Docs/User_Guide/console/branching-and-rewind.md` only after live evidence is
   green.

## Failure Handling and Privacy

This change adds no runtime failure path, persistence, logging, or user-data
flow. A missing or stale bundle is caught by the existing bundle-sync gate.
Temporary live verification uses scratch HOME/XDG/config paths and contains no
message bodies beyond fixed synthetic text.

## Alternatives Rejected

- **Global Button repair:** broader than the defect and risks changing every
  application surface.
- **Custom action widgets:** duplicates native Button semantics and adds
  unnecessary accessibility and event surface.
- **Widget-only `DEFAULT_CSS`:** cannot reliably outrank the app-loaded global
  Button rule that causes the live cascade.

## ADR Check

ADR required: no

ADR path: N/A

Reason: this is a localized visual bug fix using the existing stylesheet and
design-system boundary. It changes no storage, ownership, service contract,
security boundary, dependency, runtime policy, or long-lived UX structure.
