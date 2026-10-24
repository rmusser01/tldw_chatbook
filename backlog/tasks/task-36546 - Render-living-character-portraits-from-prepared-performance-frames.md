---
id: TASK-36546
title: Render living character portraits from prepared performance frames
status: Done
created_date: 2026-10-23 22:56
dependencies:
- TASK-36542
- TASK-36543
- TASK-36545
labels:
- console
- character-chat
- ui
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
- Docs/superpowers/reviews/2026-10-23-task-36546-review.md
assignee:
- '@codex'
updated_date: 2026-10-24 02:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Animate the selected character expression with audio-reactive mouth poses, independent blinking and subtle idle motion while preserving Console ownership, geometry and motion preferences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prepared sets display the matching mood and source; mouth changes follow owned output snapshots rather than text streaming, with independent blinking and fixed-canvas idle motion.
- [x] #2 Automatic, manual and valid historical expression precedence is preserved; unprepared, stale or unavailable sets fall back without substituting an unrelated neutral talking face.
- [x] #3 Same-source Apply/Remove and revision changes refresh the visible portrait after commit; late predecessor preparation or paint cannot restore a removed/replaced set.
- [x] #4 Dynamic/Static, Reduce motion, animations-off, reactions-off and avatar visibility retain their governing behavior; hide/show, source/profile/session changes and resize fence and release visual work.
- [x] #5 Frame preparation stays off-thread within existing active/in-flight/shared budgets, paints at most 30 times per second and performs no per-frame decode, SQLite read or rail remount.
- [x] #6 Mounted tests prove changed pixels, silence closure, mood retention, historical replay restoration, geometry stability, viewer snapshots, bounded resource release and stale-worker rejection.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Implement approved disposable visual playback and owned speech integration within existing avatar/source, resource and motion boundaries.

1. Read existing expression selection, avatar preparation/rendering, source/revision storage, viewer and UI token rules.
2. Add failing pure clock/token/preparation and mounted changed-pixel/ownership/revision tests using existing harnesses.
3. Define motion tokens and pure blink/idle clock; prepare six fitted frames off-thread under shared image budgets with complete cleanup.
4. Extend the existing avatar with optional performance/speech input, fixed-canvas idle motion and <=30 fps playback, honoring visibility and effective motion.
5. Resolve exact local source/performance revision off-thread on activation/invalidation; fence stale profile/session/source/revision/renderer/motion work and preserve manual/historical precedence.
6. Verify mounted pixels, geometry, viewer copy, same-source replacement/removal, resource cleanup and fallback; rebuild CSS, run targeted guards/regressions, self-review, document and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented six-role off-thread preparation within existing shared image reservations and living portrait playback in the existing avatar. Added packaged motion tokens, independent blink/idle clocks, fixed-canvas shifts, fresh owned-mouth publication checks, current/next buffer cleanup, source/revision/tombstone-aware lookup and context fencing. Manual and historical mood precedence, viewer copies and all governing motion preferences remain intact. Serialized only live-emote consumption to retain every beat; unchanged forced refresh keeps its mounted animation. ADR-242 applies. Verification: 68 combined focused cases passed, 32 final affected cases passed after monochrome optimization, 20 relevant legacy controls passed, and final token/CSS/screen guards passed 17 cases. New/previously clean files lint clean; zero introduced legacy lint findings; AST/format/diff checks pass. Atomic review corrected delayed loud paints, same-source removal races, optional DB-read failures, missing-source neutral talking fallback and small-art offset scaling. Full evidence and exact HEAD baseline caption/resize findings: Docs/superpowers/reviews/2026-10-23-task-36546-review.md. No full suite or physical audio qualification claimed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
