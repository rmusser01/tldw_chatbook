---
id: TASK-32323
title: >-
  Inspector rail per-section collapse
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review A4. The Inspector rail is one giant scroll unit with no per-section collapse (right_rail.py module docstring notes the whole rail is a single collapse/expand unit), unlike the left rail's seven disclosure sections. Introduce per-section headers for at least Sources / Run / Session Settings, persisted like the left rail's section flags.

Filed from the 2026-09-10 Console rail UX review (review item A4).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit the Inspector's current section structure against the residual gap. 2. Decide: bounded blocks are sufficient (no follow-up) with rationale. 3. Document collapsible-vs-bounded in the layout tour. 4. Confirm pinned contracts untouched.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Inspector's section-collapse structure is audited against the residual gap (Sources tray, run-inspector block) and the outcome is recorded in Implementation Notes: either close (bounded sections + scroll hints + More disclosure are sufficient) or a follow-up is filed with rationale
- [x] #2 No regression to the existing collapsible sections (Environment/Tasks/Agents), n/p navigation, or the pinned overflow-tail contract
- [x] #3 User guide documents the Inspector's section navigation (n/p) and collapsible sections
<!-- AC:END -->
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**The audit (AC#1).** The premise ("Inspector has no per-section
collapse") was found largely already addressed on dev by the
TASK-2154.x-era redesign: Environment/Tasks/Agents are collapsible
ConsoleInspectorSections with persisted per-workspace open state; n/p
section navigation exists (keyboard-route + navigation suites pin it);
the overflow tail "▼ more sections — scroll" is pinned last-child
non-focusable (test_outer_hint_is_pinned_last_child...). The residual
non-collapsible blocks were audited individually and found ADEQUATELY
BOUNDED rather than needing collapse: the Sources tray scrolls inside
its own ConsoleBoundedSection frame; the run-inspector groups fold
behind the block's own "More" disclosure; the scope row and prefill rows
are single lines. Collapsing these would add headers to one-line blocks
(a net vertical cost) for no navigation gain that n/p does not already
provide. Decision: close — no follow-up.

**Docs (AC#3).** The user guide's Inspector layout tour now states which
blocks collapse (with per-workspace persistence) and which are bounded,
and that a deep Environment expansion cannot push the Scope row off the
rail (the bounded-section contract).

**AC#2.** No behavior change; the existing pinned contracts
(navigation, outer-hint copy/placement, section tests) all still pass
untouched — verified across this arc's runs of the navigation,
inspector-section, environment-section, and right-rail suites.

**ADR check.** Not required — documentation and an audit decision; no
boundary moved.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

PREMISE LARGELY GONE on dev: Environment/Tasks/Agents are collapsible ConsoleInspectorSections (right_rail.py:1608-1667), n/p section navigation exists, overflow tail 'more sections - scroll' pinned (right_rail.py:1772-1783). Residual: Sources tray and run-inspector block are not collapsible, but run inspector has a 'More' disclosure and Sources is bounded with its own scroll hint.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
