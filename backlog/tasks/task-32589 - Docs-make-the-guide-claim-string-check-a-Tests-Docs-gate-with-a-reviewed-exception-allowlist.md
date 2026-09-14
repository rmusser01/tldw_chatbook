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
A guide sentence that quotes what the app prints is falsifiable in one grep, and nothing in this repo was running that grep. task-32558 ran it by hand over five User_Guide pages and found NINE false claims in roughly three hundred — on pages already carrying dozens of 'Verified against' stamps. The stamp convention cannot catch this class: a stamp verifies the claim it names, not the chapter it sits in. The sharpest instance: commit 67fec3f350 (task-15122, 2026-08-11) rewrote the Session Git panel's heading, scope line AND keyboard guide; file-notes.md kept all three false spellings for a month while FIFTEEN stamps were added to that page across six dates, three of them naming that panel, two of those three live walks that drove it end to end through trust, staging and a real commit, and a third that captured the panel's own header control with the false heading three lines above it. grep -rF 'Prepare session for commit' tldw_chatbook/ returned nothing on any of those days.

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
<!-- AC:END -->
