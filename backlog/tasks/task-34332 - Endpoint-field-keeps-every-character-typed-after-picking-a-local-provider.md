---
id: TASK-34332
title: 'Endpoint field keeps every character typed after picking a local provider'
status: Done
assignee:
  - '@claude'
created_date: '2026-10-03 16:30'
labels:
  - first-run-wizard
  - provider-step
  - ux-review-2026-10-02
dependencies:
  - TASK-34100.1
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Characters typed into the Provider step's Endpoint field soon after picking llama.cpp are dropped. The URL ends up truncated, and Next is then refused. This is the "users lose keystrokes" symptom that review finding cross-cutting-17 describes. TASK-34100.1's ACs did not cover it, and it is not a regression: base loses more characters.

Evidence (TASK-34100.1 review round 2, live, real app, files under .worktrees/setup-wizard-ux-qa/evidence/):
- Base 3c439d606e, Tab then End, 30 Backspaces, then the URL typed as one burst: the field holds "http://127.0" with "Endpoint URL must include a valid host and port." (g1-v2-base80/02-base-endpoint-truncated.txt).
- Branch under a main-thread sampler, same sequence: "http://127.0.0.1:" with "Endpoint URL must not include an empty port." Five Ctrl+N presses were then refused with "The provider settings are invalid." (g1-v2-prof/00-branch-endpoint-truncated.txt).
- The branch without the sampler kept the whole value (g1-v2-fk120/04, g1-v2-fm80/06), so the loss depends on how busy the UI loop is while the user types.

The likely cause is something that rewrites `#setup-provider-endpoint` while the user types: a selection-change or validation handler that sets `.value`, or a re-render. TASK-34100.6 owns the endpoint fields; coordinate there if it lands first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The code path that drops or overwrites typed Endpoint characters is identified and recorded
- [x] #2 A Pilot test that picks llama.cpp and types a full URL with no pauses ends with exactly that URL in the field; it fails on the pre-fix code
- [x] #3 Verified live on a fresh isolated profile with the UI loop under load (for example during the localhost scan): the typed URL survives and Next accepts it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed inside TASK-34100.6 (which owns the endpoint fields), commit 96ea86bb28.

**The code path (AC#1).** Not a handler rewriting `.value`: message dispatch.
Textual's `Input` inserts a printable key in its own handler, but resolves
Backspace, End, Delete and Ctrl+U through non-priority bindings only after the
key has bubbled to the App, one message hop per ancestor. In a burst the
characters typed after those keys were inserted first and the late Backspaces
then deleted them (`http://127.0.0.1:9099` became `http://127.0`). Separately,
the Screen's Tab binding is not `priority`, so keys typed right after Tab still
reached the provider list (End jumped it to the last row, a legacy alias, and
replaced the endpoint). Same mechanism as lessons-textual "Screen's Tab binding
is not priority".

**Fix.** `first_run_provider_key_entry.BurstSafeInput` runs the field's own
editing bindings in arrival order (the endpoint and key fields use it), and
`ProviderChoiceList` has priority Tab/Shift+Tab bindings.

**Test (AC#2).** `Tests/Wizards/test_first_run_provider_endpoints.py::
test_a_url_typed_in_one_burst_after_a_pick_lands_whole` posts the burst the way
the driver does (no awaits between keys). RED on base a1571e86db (the field
ended `''`), green after.

**Live (AC#3), 2026-10-04,** fresh isolated profile `g6-burst` at 80x24, while
the localhost scan ran: pick llama.cpp, then Tab, End, 30 Backspaces and
`http://127.0.0.1:8181` sent as one tmux burst -- the field kept the whole URL,
Next accepted it and Model listed the real llama-server's model
(`setup-wizard-ux-qa/evidence/g6-final/90-…`, `92-…`). Note: a burst that sends
the pick's Enter and the Tab in the same instant can still Tab before the
endpoint panel appears (focus then lands past it); that is faster than the
filed repro and than a person.
<!-- SECTION:NOTES:END -->

