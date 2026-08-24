---
id: TASK-21596
title: >-
  Video generation resolves the MiniMax secret eagerly whenever the config is read
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - performance
  - video-generation
  - keyring
priority: low
---
## Description

TASK-21111 removed the *boot* consumer that forced a Keychain query during `TldwCli.__init__`,
but the underlying shape remains: `Video_Generation` resolves the MiniMax secret whenever anything
asks for the full config. Opening Settings or Console video still pays the Keychain query up
front rather than at send time.

Keyring cost is easy to under-count because `keyring.get_keyring()` memoizes the backend — the
first caller pays backend discovery for everyone, so the expensive site is whichever runs first,
not whichever looks heaviest.

## Acceptance Criteria

- [ ] The MiniMax secret is resolved when it is used, not when the config is read
- [ ] Opening Settings and opening Console video are measured to perform zero keyring calls
- [ ] The measurement counts keyring calls across the whole interaction, not just the site being changed, so a relocation cannot pass as a removal
- [ ] A missing or rejected credential still surfaces the same error to the user, at send time

## Evidence

From TASK-21111: the first keyring touch of every boot was `Video_Generation/config._keyring_get`
at **18.2 ms** (11.3 ms backend discovery plus the query), while the three sites the original
finding named cost 0.33, 0.41 and 0.04 ms. That task took the boot path to zero keyring calls;
this is the remaining on-demand path.
