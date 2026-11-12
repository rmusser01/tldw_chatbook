---
id: TASK-40331
title: "The User Guide template still says captures are 200x50, while the capture driver and 20 of 22 images use 235x52"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- docs
priority: low
references:
- Docs/User_Guide/_template.md
- scripts/capture_user_guide.py
dependencies:
- TASK-38970
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (docs stream, TASK-38969/38970). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** The next person who follows the template's hand recipe makes 200x50 captures that do not match the other images, relying on a nav-bar clipping claim that is no longer true.

**Evidence.** `Docs/User_Guide/_template.md:100` uses `run_test(size=(200, 50))`, and `:138-147` says "Standard size: 200x50 ... 13-item nav bar ... about one column of slack" (the bar now has 15 destinations). The template's own driver pointer (`:65-69`, 814be96ead) contradicts it: `scripts/capture_user_guide.py:44` sets `SIZE = (235, 52)`, and 20 of the 22 User Guide SVGs are 235x52 (the other two are the approval-card harness at 120x20). The driver has no size option (`main` takes one target, `:373-374`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The template names the driver's 235x52 as the standard size and drops the 13-item, 200-column passage
- [ ] #2 The hand recipe in the template uses the same size as the driver
<!-- AC:END -->
