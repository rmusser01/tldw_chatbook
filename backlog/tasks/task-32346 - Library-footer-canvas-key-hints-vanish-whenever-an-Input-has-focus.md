---
id: TASK-32346
title: 'Library footer: canvas key hints vanish whenever an Input has focus'
status: Done
assignee: []
created_date: '2026-09-11 06:14'
updated_date: '2026-09-11 07:34'
labels:
  - library
  - ux
  - critique-10
  - keyboard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After submitting a search (or typing in any Library Input) the footer reads 'typing in field | F6 next pane' while the canvas keys u (use Library context in Console), o (open evidence) and / still work and are named nowhere; keyboard-only users lose their key map at the moment of use (A caps 14/15; B marks the footer-hint claim contradicted). Not pinned by name. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The canvas verbs stay visible while an Input has focus, prefixed by the field state (e.g. 'typing in field · esc leaves field | u … | o …')
- [x] #2 When width is short, 'F6 next pane' is dropped before any canvas verb
- [x] #3 Pinned on Search/RAG and the Media list
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: footer chips while an Input has focus (Search/RAG + Media list)
2. Rewrite the isinstance(focused, (Input, TextArea)) block in _library_footer_shortcuts_for_current_state: lead with 'typing in field', name 'esc leave field' (gated on check_action), keep the swallowed verbs behind 'after esc: ', push 'F6 next pane' last
3. Green + footer suites, docs stamp
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The snapshot's premise is wrong and the fix keeps the correction: u, o and / are non-priority Bindings, so a focused Input consumes the printable key before any Screen binding sees it -- re-advertising them as live would be the dead-key lie task-31272 removed, and task-31223's suppression stays. What was missing was the way back.

_library_footer_shortcuts_for_current_state's isinstance(focused, (Input, TextArea)) block now splits the route set into swallowed single printable keys and kept multi-char ones, and rebuilds it as:
  typing in field | esc <label> | after esc: <swallowed verbs> | <other kept> | F6 next pane
'esc leave field' is synthesised only where the route set carries no Escape chip of its own, gated through check_action('library_blur_text_field') so it can never advertise a refusal; where the surface already owns Escape (the Media list's 'esc focus rail', the Reader's 'esc close') that chip speaks for itself. '/' is dropped from the swallowed list by the same _library_slash_would_land predicate that already drops the live chip. 'F6 next pane' moves LAST so AppFooterStatus's retain-the-prefix degradation drops it before any canvas verb (AC#2).

Live (tmux, seeded profile): Search/RAG query box at 235 cols reads 'typing in field | esc leave field | after esc: u use Library context in Console · o open evidence · / focus search | enter run search | F6 next pane'; the Media filter box reads '... | esc focus rail | after esc: / focus search · s select | F6 next pane'; the Reader Find bar at 160 cols keeps the verbs and drops F6 to the reserved cluster.

Files: tldw_chatbook/UI/Screens/library_screen.py (that one block only), Tests/UI/test_library_crit10_viewer.py (new), Docs/User_Guide/library.md (footer paragraph + the retired 'enter select evidence' claim, stamped).
<!-- SECTION:NOTES:END -->
