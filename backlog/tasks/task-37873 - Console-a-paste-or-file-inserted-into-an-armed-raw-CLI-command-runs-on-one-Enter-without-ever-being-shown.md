---
id: TASK-37873
title: >-
  Console: a paste or file inserted into an armed raw-CLI command runs on one
  Enter without ever being shown
status: Done
assignee:
  - '@claude'
created_date: '2026-10-30 04:51'
updated_date: '2026-10-30 05:15'
labels:
  - console
  - composer
  - raw-cli
  - ux-review-2026-09-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review of the TASK-37800 change (owner ruling 2026-10-29: paste then Enter sends) found that an armed raw-CLI draft made of a physically typed '! ' plus a large paste now runs on ONE Enter while the composer shows only '! Pasted text | 125 characters | Expand'. A review probe on the real ChatScreen with the real RawCliRuntime saw start_user_command receive the hidden 'echo hidden-host-command-...' text. On dev the same Enter only showed '! Expand?' and nothing started (the Run button already ran the hidden text there). A host command with the OS user's full authority must be on screen in full before it runs: a collapsed chip in a raw command is the pastejacking shape, and Enter is the main gesture. An inserted text file in a raw command collapses the same way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Text pasted or a text file inserted after a physically typed '! ' prefix shows in full in the composer; no collapsed chip stands for any part of a raw-CLI command
- [x] #2 Outside a raw-CLI command, large pastes and inserted files still collapse as before, and the TASK-37800 one-Enter tests stay green
- [x] #3 A Pilot test on the real ChatScreen and composer, with the real RawCliRuntime armed, pastes and inserts a file after '! ' and asserts the full command shows before one Enter runs exactly that command; it fails on the pre-change code
- [x] #4 Docs/User_Guide/console/chat-basics.md says nothing collapses inside a raw CLI command
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Test first on the real ChatScreen: armed runtime, typed '! ', paste / file insert over the threshold, assert the full display, then one Enter runs it. 2. Composer: an insert that keeps the trusted '! ' prefix is inserted literal (one predicate shared with the trust rule). 3. User Guide raw CLI and Large pastes. 4. Ratchet row under the owner rule.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix.** The composer's trust rule for a typed '! ' prefix (an edit that starts at index 2 or later and replaces no selection keeps it trusted) is now one shared predicate, `ConsoleComposerBar._keeps_raw_cli_prefix`. `insert_pasted_text` and `insert_file_segment` ask it first: inside a raw command the paste or file goes in literal, so its full text is on screen before Enter or Run; everywhere else collapse is unchanged. Trust exists only after '!' and space are typed into an empty composer, and undo, snapshot replacement and /prompt clear it, so these two inserts are the only way a collapsed segment could enter a raw command.

**Tests.** `test_text_inserted_into_an_armed_raw_command_shows_in_full_before_it_runs` (Tests/UI/test_console_raw_cli_arm_in_place.py; real ChatScreen, composer, controller and an armed RawCliRuntime; real keys for '! ', a real Paste event, then `insert_file_segment`): the full command shows, the status reads RAW CLI · HOST ACCESS with Run, and one Enter starts exactly that command and empties the composer. Red on the pre-change code ('! Pasted text | 125 characters | Expand') and on origin/dev; with only the file guard removed it fails at the file ('📄 run.sh · 1 KB | Expand'). One app run covers both inserts, so the gated file stays near the lane's 30 s budget (5 cases, about 27 s at load 25).

**Live** (real app, isolated profile, null keyring, raw CLI unlocked, 160x45). Typed '! pwd', Enter opened the arm confirmation, Arm host access armed it; replaced 'pwd' with a bracketed paste of a long echo command. The composer showed the whole command over two rows with RAW CLI · HOST ACCESS; one Enter ran it (Tool row 'Exited 0', stdout the same text) and emptied the composer. A chat paste after a question still collapsed ('Pasted text | 102 characters | Expand') and sent on one Enter; claude-haiku-4-5 replied 'Worker crashed and recovered.'

**Docs and ratchet.** User Guide console/chat-basics.md: 'Large pastes' and 'Raw CLI user commands' say nothing collapses inside a raw command. console_composer_bar.py size row 7142 -> 7155 with a dated owner-rule comment. No new modules or CSS.
<!-- SECTION:NOTES:END -->
