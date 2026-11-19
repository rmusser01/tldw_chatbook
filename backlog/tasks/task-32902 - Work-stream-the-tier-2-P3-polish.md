---
id: TASK-32902
title: "Work stream: the tier-2 P3 polish"
status: Done
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 108 P3 findings. Lowest priority in the review and explicitly the last thing to burn down; several
are one-line corrections (`domain.replace("www.", "")` strips `www.` anywhere in the netloc, so
`newww.example.com` becomes `neexample.com`), several are `re.compile` inside a per-item loop, and a
large share are diagnostics that reach no sink.

Do not let this stream block the others, and do not open it before the P0/P1 streams are closed.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every P3 is either fixed or closed with a recorded reason
- [ ] #2 No P3 change alters behaviour beyond the finding it closes
<!-- AC:END -->

## Closeout audit 2026-10-11

Board audit (branch `chore/board-prune-prune-d`, dev at a3c47c6c72): this task's work is
landed on `origin/dev` but the status field was never flipped. Verdict: **Done**.

The tier-2 P3 wave completed on dev: PR #2814 (p3a: Library, UI screens and small
packages), PR #2815 (p3b: API client, Web, Evals, Models and Speech) and PR #2816 (p3c: "the
final tier-2 P3 findings - Notes, Backup_Recovery, Canvas, Media"), all merged 2026-09-23..27,
every batch commit citing TASK-32902. Branch commits carry the AC#1 disposition discipline
(fixed / closed-with-reason / retracted, e.g. "is RETRACTED, not fixed" in the p3c range).
Note: unrelated Dreams commits also cite "task-32902" from before that task was renumbered to
32917 (81409116c4); they are not this task.

AC checkboxes are left as found; this section is the audit's evidence record.
