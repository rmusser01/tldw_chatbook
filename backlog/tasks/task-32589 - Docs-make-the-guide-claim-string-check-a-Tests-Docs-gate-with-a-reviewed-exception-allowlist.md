---
id: TASK-32589
title: >-
  Docs: make the guide-claim string check a Tests/Docs gate with a
  reviewed-exception allowlist
status: To Do
assignee: []
created_date: '2026-09-14 23:17'
labels:
  - docs
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A guide sentence that quotes what the app prints is falsifiable in one grep, and nothing in this repo was running that grep. task-32558 ran it by hand over five User_Guide pages and found NINE false claims in roughly three hundred — on pages already carrying dozens of 'Verified against' stamps. The stamp convention cannot catch this class: a stamp verifies the claim it names, not the chapter it sits in. The sharpest instance: commit 67fec3f350 (task-15122, 2026-08-11) rewrote the Session Git panel's heading, scope line AND keyboard guide; file-notes.md kept all three false spellings for a month while FIFTEEN stamps were added to that page across six dates, three of them naming that panel, two of those three live walks that drove it end to end through trust, staging and a real commit, and a third that captured the panel's own header control — with all three true strings inside that same captured frame (wave4-caps/import-kbd/import-21-back-cue-files.txt paints '‹ Files' at :16, 'Review session changes' at :17, the scope line at :19 and the keyboard guide at :20). grep -rF 'Prepare session for commit' tldw_chatbook/ returned nothing on any of those days.

scripts/check_guide_claim_strings.py is now in the tree and is the finding half: it extracts every quoted and bolded run from a page, normalises the guide's markdown escaping, and greps each against tldw_chatbook/, splitting composed f-string lines into their literal runs. Verified against the pre-sweep file it flags all four of file-notes.md's false claims by line number. It deliberately exits 0 and prints a read-list rather than a verdict, because a guide legitimately quotes strings no source emits: historical '(Was ...)' clauses, composed examples, and the reader's own input. It already carries --fail-on-miss for the day the judgement exists.

What is missing is the judgement half — the reviewed-exception allowlist, in the idiom EXPECTED_CHACHANOTES_INDEXES and REVIEWED_METADATA_ONLY_DIAGNOSTICS already use here, so that every quoted string a guide carries is either resolvable against the code or a recorded decision. Same argument that made the Select.BLANK sweep (task-32568) a task rather than a lesson: enforcement, or it decays into folklore. Nine incidents in one sweep is the evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Tests/Docs test fails when a User_Guide page quotes a UI string no source file emits and that string is not in the allowlist
- [ ] #2 The allowlist records each exception with its reason — historical (Was ...) clause, composed example, user input, or not-yet-shipped — so an entry is a decision a reviewer made rather than a silence
- [ ] #3 The nine claims task-32558 corrected are covered: running the check against each pre-sweep page flags them, and against the corrected pages does not
- [ ] #4 The check runs over every page under Docs/User_Guide/, not only the five this sweep walked
- [ ] #5 A page can be onboarded incrementally: adding a page does not require clearing the whole repo's backlog of exceptions in one commit
- [ ] #6 Wall time is stated, and the check is in preflight only if it fits its budget
- [ ] #7 A claim that names a surface resolves against THAT SURFACE's own modules AND against string LITERALS only — both conditions, in one probe. Neither half alone catches the incident that produced this task. Measured on it ("scoped" = the 12 library_notes* modules; "AST" = non-docstring ast.Constant values; the guide said "press Export… in Notes" while the Notes toolbar ships a bare "Export" at library_notes_canvas.py:1842): repo-wide+raw 35 / repo-wide+AST 9 / scoped+raw 1 / scoped+AST 0 — and for the TRUE spelling "Export", scoped+AST 13. repo-wide+AST misses because Export… is a live Button label on six surfaces the claim did not name (meetings_screen:316, artifacts_pane:1392, library_media_canvas:1178, library_prompts_canvas:890 and :1681, library_conversations_canvas:151, console_conversation_inspector:1518). scoped+raw misses because of ONE hit — library_notes_controller.py:5406, a stale docstring on handle_library_notes_export, the handler for the very button that ships bare — so prose about the Notes export action, inside a Notes module, satisfies it. WARNING TO WHOEVER TAKES THIS: this criterion has now been wrong twice, each time by prescribing one half. Before implementing, re-run all four probes and check the fix catches the false spelling (0) AND passes the true one (13). Every number here is labelled with the probe that produced it, because an unlabelled scoped+AST figure is what let the scoping-only prescription ship.
- [ ] #8 The page-to-module binding is explicit and reviewable — a page or section declares the surface it documents, and an undeclared page is reported rather than silently passing on a tree-wide probe
<!-- AC:END -->
