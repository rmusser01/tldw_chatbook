---
id: TASK-41101
title: "398 legacy 'Verified against' stamp lines remain on 28 User Guide pages, which CLAUDE.md and the page template forbid"
status: Done
assignee: ['@claude']
updated_date: 2026-11-16 21:18
created_date: 2026-11-15 02:20
labels:
- follow-up
- docs
- user-guide
priority: low
references:
- Docs/User_Guide/_template.md
- Docs/User_Guide/library/notes.md
- Docs/User_Guide/library/media-and-conversations.md
- Docs/User_Guide/index.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's. Counts rechecked after the restack onto dev ecad404bd4 (review r14cl-rev1).

**Why.** CLAUDE.md ("UI changes") and `Docs/User_Guide/_template.md:49-51` say a page carries no "Verified against" lines: verification belongs in the task's Implementation Notes, and parallel PRs appending stamps at the same spot caused most User Guide sync conflicts. Old stamps keep inviting new ones, and they tell readers a stale branch and date.

**What is there.** `grep -c "Verified against"` over `Docs/User_Guide` counts 408 lines on 34 files. One is the template's own rule, and nine are on the five pages other tasks own (below), which leaves 398 lines on 28 pages for this task. The largest: `library/notes.md` 73, `library/media-and-conversations.md` 68, `library/import-and-export.md` 49, `library.md` 48, `library/file-notes.md` 26, `settings.md` 25, `library/search-and-rag.md` 18, `console.md` 14; `index.md` has one at `:184-186`.

**Not this task.**
- The four Roleplay pages (`roleplay-chat-dictionaries.md` and its three sub-pages, one stamp each) belong to TASK-33910.18 AC#4.
- `First_Run_Setup.md` (5 lines) belongs to TASK-34100.15 AC#7, under its own evidence rule: its stamps' evidence moves into the named tasks' notes, and TASK-33008 AC#8 drops its stamp requirement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No page under Docs/User_Guide other than the four Roleplay pages and First_Run_Setup.md contains a 'Verified against' stamp; the template's rule line stays
- [x] #2 Any fact a removed stamp carried that the page body does not (for example a retired screen named only in the stamp) is kept in the body or dropped on purpose, and the notes say which
- [x] #3 A Tests/Docs check fails when a 'Verified against' stamp is added to a User Guide page, shown red with one planted stamp; the five pages other tasks own are exempt by name only until those tasks land
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory every verification stamp on the in-scope pages: the exact "Verified against" lines plus their variants ("Re-verified against", "Re-stamped against", "verified on <branch>", "checked against <sha>"), including whole "## Verified against" sections and the "—" separators before footer stamps. Excluded: First_Run_Setup.md, the four Roleplay pages, the template's rule line.
2. For each stamp, compare its product facts with the page body: a current user-facing fact the body lacks goes into the body as one plain present-tense sentence (checked against current code first); verification provenance (branch, sha, date, size, test counts) and change history are dropped on purpose. Evidence worth keeping (QA evidence links, stated verification limits) moves into the named task's notes when that task does not already hold it.
3. Fix body sentences that point at a removed stamp ("below", "this stamp").
4. Add a Tests/Docs/test_user_guide_contract.py check that fails on a stamp-shaped paragraph in any User Guide page except the five other-task pages (by name); show it red with one planted stamp.
5. Record kept/dropped facts per page in the notes; run the Docs tests and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed every verification stamp from the User Guide except on the five pages other tasks own, kept the few product facts that lived only in a stamp, moved the verification limits into the tasks the stamps named, and added a contract check that fails on a new stamp.

**Scope.** All 28 pages in the description, plus two the exact-phrase count missed: `console/branching-and-rewind.md` ("…rewind verified against dev @ ff435772c — 2026-07-31") and `buddy.md` ("checked against Chatbook dev 307df6c79 on 2026-09-07"; its heading "Verification scope and related guides" is now "Related guides", with no inbound links). Variants went too: "Re-verified against", "Re-stamped against", "Verified on <branch>", "Verified for TASK-…", "Reviewed on …", "Docs pass <date>", "Amended — <date>", "*Not verified live — task, date, branch*", "Copy synced with code — task, date". The `## Verified against` sections of notes.md and file-notes.md, the footer stamp runs at the end of most pages, every lone `—` separator in front of them, and stamp commentary in the body ("Fifteen 'Verified against' stamps were added to this page…", "This carries no 'Verified against' stamp on purpose", "this page's own task-32555 stamp") were also removed. 30 files, +150 -5041. `First_Run_Setup.md`, the four Roleplay pages and the template's rule line are untouched. The pages were split across six helper agents, one group of pages each. Each agent checked every stamp's facts against the page body and the current source, and the coordinator reviewed every added line.

**AC#2: what was kept.** These facts lived only in a stamp, were confirmed against current source, and went into the body as one plain sentence each (stamp task in brackets):
- console.md: the typing-locked notice "Typing is locked until setup finishes — press Enter to use the highlighted action." [32533; `console_setup_modal._TYPING_LOCKED_HINT`]; "No models reported for this provider. Use Custom ID if needed." [33001.1].
- console/context-and-rag.md: the "Environment unavailable — Refresh to retry" row when git status cannot be read and nothing earlier exists [31450]; both Changes actions open the working tree while Review changes / Review open the recorded turn [31665]; the Refresh tooltip "Refresh — Environment" [31665].
- library.md: "Use it in Console" unlocks once a result is selected; entering Conversations focuses its first row [32228].
- library/search-and-rag.md: resizing or results arriving keeps the focused control in view [32715/32716].
- library/media-and-conversations.md (10): reopening Library closes Items where both cannot fit [32065]; the Analyze offer is withdrawn on a filter/type/page/Select change [28007]; the scope line's bare count when the total is inexact [32350]; a type-list-only failure keeps row actions [31982]; the Find reason on Highlights/Info [32348]; Move to trash drops the rail count at once [3020/14901]; "Type: markdown (stored as plaintext)" [4023]; Review-these set naming [31274]; the "Conversations (N)" heading and "5 messages · 16m" rows [2859/32364]; "/" and first-row focus in Conversations [32228/32302].
- library/file-notes.md (5): the picker tries `[notes] sync_directory` before home [32251]; the 500-entry count ceiling ("48+ notes") [32611/32643]; the "Enter Open · Select folder to use this folder" hint [32611]; one-row breadcrumb and path with middle ellipsis [32614]; where the Session Git commit receipt appears [32615/32265].
- library/import-and-export.md (11): Clear keeps the caret; the link check obeys `[web_security]` and follows no redirects [19556]; option-group titles lead with a value to fix or a blocked package [14822-14826]; one "Import finished — …" notification per batch [2857/32054]; Skills and Collections are never in a bundle [197]; the pre-check ⚠ line, Copy install command and "What's missing" fold [14822]; the cookies file is only read [3306]; PaddleOCR/docext companion packages [3306/3307]; Parakeet Browse… fills only its field [32664]; the egress receipt names the refused address [3311/3312]; Export's Escape destination and quality-strip footer [197/14902].
- library/notes.md (13): select mode ends on Escape or a filter change [32185]; the "Folders & placement" heading and unselected reason [32617]; deleted linkers drop out of Linked from [32186]; Preview/Info focus border [32613]; a blank new note is discarded on Escape/rail/tab [4021]; an empty title saves as "Untitled" [3315]; "○ Keep a folder synced" with its not-ready line [32534/32545]; runs collapse at eight files [32625]; folder-picker sort, counts, vault marker and Ctrl+R with a link to file-picker.md [32611/32643]; "Resolution history unavailable — …" [32549]; "Ⅱ Paused · Next: Resume" [32519]; Tab stays inside the sync pane [32607-32613]; an unclosed fence counts as code [PR #2549 review].
- library/collections.md: resizing keeps annotation drafts [32662]. library/skills.md: Escape closes the sort strip; the "User can invoke: ✓ yes ⇄ no" toggle label [14902]; import success updates the list and rail count [32058].
- settings.md: a theme file copied into the folder is not listed until Import… or restart [32948]. schedules.md (5): the pushed detail closes with "'Title' is no longer in the queue."; the "Local only — nothing to sync…" refusals; liveness refreshes every 5 s; the Duplicate pause-failure warning [31823]; "Solve: eligible / not eligible — <reason>" [18940]. mcp.md: the **Unknown** permission state [2740/2270/2870]; the selected server lists first and Open tool catalog filters to it [32283]. openai-compatible-tts.md: Speech Lab seeds the saved Model/Voice names, with "(no catalog check)" [15421].

**AC#2: what was dropped on purpose.** (1) Provenance, in every stamp: branch, sha, date, terminal size, live/tmux/headless method, captures, test names and counts, review names. (2) History: "regressed and restored", "used to read", "before this it never did", fixed-defect narratives and the long "Contradicted and corrected here" passages, whose corrections the body already carries. (3) Facts the body already states (several hundred; each agent named where the body says it). (4) Stale against current code, so not restored. Examples: library.md's staging-row second line, the trailing ⇄ on cyclers, the graduation toast and the ● count marker; search-and-rag's "Media excluded — semantic only." note and "plain Search requires every word" (TASK-17755 added the prefix fallback); media's one-cell grips, "analysed", "· ValueError" and "esc cancel" on the type strip; notes' "Draft — not saved yet", "60 applied · durable receipt recorded" and the callout run-on (now fixed); settings' "(Textual)/(saved)" Appearance labels, "Hidden for Anthropic", "API key source: not required"; console's Alt+M Apply/Defaults… states (that popover is gone) and the snow backdrop; import-export's retired "Some files may fail to import" dialog and "esc back to hub"; prompts' "cannot be undone" delete copy (the modal now offers Undo); context-and-rag's "Costs / Exchange / Next Send" tab names. (5) Unverifiable at reasonable cost, so not restated as current: e.g. media Retry keeping focus, "No preview in Select mode." reachability, bulk-toolbar clipping below 110 columns (task-15140), Tool Pack exports carrying character-tool rules (no Tool Packs page exists), the keyless `credential_source = "none"` write, two fleet PR3a-2 wake follow-ups. (6) Internal mechanics: function and seam names, CSS selectors, performance figures, log fields. The full stamp text is at the base commit: `git show cc6006fcf5:Docs/User_Guide/<page>`.

**Evidence moved.** Where a stamp recorded a verification limit or evidence pointer its named task did not hold (not driven live, code-level only, pinned by a named test, capture names), it is appended to that task's Implementation Notes under "User Guide verification record (moved here by TASK-41101 …)". That covers 77 task files: 2374, 15455, 15777, 15860, 16071, 16174, 16688, 16851, 18060, 18940, 19701, 23024, 23026, 28007, 31220, 31635, 31660, 31662, 31664, 31665, 31710, 31745, 31823, 31826, 31827, 31968, 31979, 31982, 32102, 32140, 32146, 32277-32281, 32284, 32286, 32289-32291, 32345, 32533, 32536, 32538-32540, 32544, 32549, 32553-32555, 32557, 32558, 32604-32606, 32621, 32940-32946, 32948, 32949, 32955, 32956, 33001, 33001.1-33001.3, 33001.7, 33061, 33064, 33065. Notable: task-33001.1's notes cited "console.md stamp" for its live popover walk, and that walk is now in the task. Stamps that named no task kept their limits here:
- schedules redesign PR-1 to PR-4, mcp/console/context-and-rag "PR-T2/PR-T3": "docs pass against shipped code/tests, live check pending" (Task 9 / §14 gate).
- voice-profiles slice 4: the Speech & TTS pointer note and the default-profile precedence bullet were not re-driven live.
- meetings PR #2471, e060f8d27, PR #2456 (no System Audio Recording grant), e26193495, meeting-program tasks 7/8: pilot or unit tests only.
- fleet PR2b Task 6 (Delete-to-cancel and the finished-row token figure not confirmed live), fleet PR3a-2 Task 7 (wake-turn UI staleness and a deferred notice labelling a done child "running", found and filed then), the turn file card (docs-only, 457 passed), the live turn-activity line (test only), chunking-agent-tools Task 6 and student-workflow Task 2 (not driven live).
- chunking-engine-parity on settings/rag.md (no live TUI walkthrough).
- buddy.md: setup and voice guidance were checked against dev 307df6c79 with mounted coverage and the 09-05 Migu UAT, not re-executed in a fresh physical session; native-terminal UAT does not certify every terminal transport.

**AC#3.** `Tests/Docs/test_user_guide_contract.py::test_user_guide_pages_carry_no_verified_against_stamps` (in the lane that already runs Tests/Docs) flags "verified/stamped/checked (live) against/on" followed within 200 characters by a date or an `@sha`, on newline-joined page text so wrapped stamps match. Body prose such as "is verified against a live call" and the template's rule line do not match. `STAMP_EXEMPT_PAGES` names exactly the five other-task pages, and a second assertion fails once one of them is stamp-free, so its exemption must go when its task lands. Evidence:
- Red, planted stamp: one `*Verified against dev @ 0123456789 — 2026-11-15 (planted).*` appended to home.md gave `{'home.md': [198]}`.
- Red, stale exemption: temporarily adding home.md to the exempt set gave "Stamp-free now; remove from STAMP_EXEMPT_PAGES: ['home.md']".
- Both reverted. Green: `Tests/Docs` 33 passed (`-n 4`, load 63).
- Before the cleanup, the same pattern counted 438 matches on 36 pages at the base; it now matches only the five exempt pages.

**Left for follow-ups (body problems found while reading, not stamps):** listed in the round report for the coordinator to file. Examples: library.md's stale "You will probably never see this" callout (contradicted by task-32349); import-and-export step 7 " · attempt 2" vs the code's " · retry N"; skills.md's bold "You can invoke" vs the label "User can invoke"; dated "(Was … superseded by task-N)" history clauses left in several pages; meetings.md sentences about "this page's verification host"; no User Guide home for Change Review's commit/push dialog or for Tool Packs. Inline size notes "(verified at 235x52)" in media-and-conversations and "(verified on tmux 3.6a…)" in console.md were left: they scope a claim to a terminal size or tmux version, not a branch and date.

Files: 30 `Docs/User_Guide` pages, `Tests/Docs/test_user_guide_contract.py`, the 77 named task files above.

**Fix round 1 (review r15doc-rev1).**
- The test now names pages with `relative_to(GUIDE).as_posix()`. On Windows, `str()` gave `roleplay-chat-dictionaries\lore-books.md`, so the three Roleplay sub-pages were not exempt and `pages[name]` raised KeyError on the nightly windows-latest run. `test_stamp_exemptions_match_windows_page_names` pins this with a `PureWindowsPath`.
- The stamp pattern now also catches a bare 7-40 character hex sha (`*Verified against dev 0123456789 …*`), "Re-verified live <date>", and any line opening `*Verified` / `*Re-verified` (a stamp with no date, such as ``*Verified against `fix/foo` (task-123: …)*``). Its 200-character window no longer crosses a ". " or ";" sentence break, so "checked against the allowlist; a reminder due 2026-01-05" is prose. `test_stamp_pattern_tells_stamps_from_prose` pins five stamp shapes and three prose sentences.
- Red evidence:
  - With the old regex and `str()` names: 5 failed — the Windows name, the three new shapes, and the allowlist prose.
  - Green: `Tests/Docs` 42 passed.
- Coverage:
  - On the base tree cc6006fcf5, the widened pattern catches the same 374 of 390 stamp-ish paragraphs as before. The 16 it does not catch are legitimate prose or mid-paragraph fragments with no date or sha.
  - On the current pages it finds nothing outside the five exempt pages, so no further stamp needed removing.
- Copy fixes:
  - context-and-rag.md names **Review in Change Review…** and **Review & commit…** as the working-tree openers, in the Environment intro and under the rows table (`chat_screen._open_change_review_current_mode`'s ruling).
  - file-notes.md places the Session Git receipt "above the Edit or Manage pane", matching `#file-notes-action-status` in the compose.

**Fix round 2 (review r15doc-rev2).** The bare-sha alternative of `_STAMP_REF` now also requires a hex letter (`(?=[0-9a-f]*[a-f])`). Before, plain decimal numbers such as "verified on 1000000 notes" or "checked against the 10485760-byte cap" matched as shas. Red: two new prose cases in `test_stamp_pattern_tells_stamps_from_prose` failed (2 failed, 8 passed). Green: `Tests/Docs` 44 passed. Base-tree paragraph coverage is unchanged at 374 of 390.
<!-- SECTION:NOTES:END -->
