---
id: TASK-21148
title: Wizard layout and density pass
status: To Do
assignee: []
created_date: '2026-08-25 06:15'
labels:
  - ux
  - wizard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings P-1, V-1, V-2, Z-1, Z-2, Z-3, F-2, F-3, G-2, N-6, S-4 (findings.md): steps overflow at 140x40 while carrying rows of decorative dead space; step titles scroll away first; the Voice step leads with plumbing and hides Test and Hear below the fold; the full-track tracker drops all step titles and truncates step 10 to '1'; 80x24 shows one provider row with no guidance; tool switches take 4 rows each; the step total changes mid-flight when Protect joins; the summary config path wraps mid-character.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 140x40, each quick-track step shows its title, primary content, and primary action without scrolling
- [ ] #2 Voice step leads with a one-line purpose and Test and Hear; endpoint/model/format/speed live under an Advanced disclosure
- [ ] #3 Full-track tracker keeps step titles at 140 cols and renders two-digit step numbers
- [ ] #4 Below a minimum size the wizard shows an enlarge-terminal hint; at 80x24 every step remains operable
- [ ] #5 Protect appears in the quick track from the start (marked skipped when keyless); the step total never changes mid-flight
- [ ] #6 Summary config path never wraps mid-character
<!-- AC:END -->
