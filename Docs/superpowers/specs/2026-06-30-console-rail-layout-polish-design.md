# Console Rail Layout Polish Design

## Problem

The Console setup-blocked screen has three visible UX defects:

- The focused conversation browser can draw a large high-saturation blue boundary box that reads as a selected panel rather than a focused control.
- The left rail and transcript empty-state surfaces leave avoidable unused vertical space, making the screen feel unfinished.
- The setup blocker/control stack at the top must remain visually solid when content scrolls behind or beneath it.

## Approved Direction

Use the existing Console workbench framing, but make the left rail use one scroll owner instead of nested scroll boxes.

```text
Before
+------------------------------------------------+
| Setup blocked                                  |
+------------------------------------------------+
| Provider / model / actions                     |
+------------------------------------------------+
| + Left rail ----------------+  + Transcript --+|
| | Context header             |  |              ||
| | + scroll body -----------+ |  |              ||
| | | Staged context         | |  |              ||
| | | + nested scroll -----+ | |  |              ||
| | | | blue focus box     | | |  |              ||
| | | +--------------------+ | |  |              ||
| | +------------------------+ |  |              ||
| +----------------------------+  +--------------+|
+------------------------------------------------+
| Composer                                       |
+------------------------------------------------+

After
+------------------------------------------------+
| Setup blocked                                  |  solid, opaque row
+------------------------------------------------+
| Provider / model / actions                     |  solid, opaque row
+------------------------------------------------+
| + Left rail ----------------+  + Transcript --+|
| | Context header             |  |              ||
| | + single scroll body ----+ |  |              ||
| | | Staged context         | |  |              ||
| | | Conversation browser   | |  |              ||
| | | Workspace status       | |  |              ||
| | +------------------------+ |  |              ||
| +----------------------------+  +--------------+|
+------------------------------------------------+
| Composer                                       |
+------------------------------------------------+
```

## Requirements

- Keep the left rail keyboard reachable through the existing F6 focus registry.
- Remove nested conversation-browser scrolling so the rail body is the scroll owner.
- Preserve row/button focus affordances inside the rail.
- Keep solid opaque backgrounds on setup-blocked and control rows.
- Let the transcript empty panel fill its transcript region.
- Preserve existing selectors used by tests and compatibility code.

## ADR Check

ADR required: no
ADR path: N/A
Reason: this is scoped UI layout and focus styling polish within existing Console/Textual contracts. It does not change storage, provider boundaries, runtime contracts, security, sync policy, or long-lived architecture.
