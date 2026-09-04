---
id: TASK-31253
title: Theme editor - Generate from Primary ignores the primary hue (HSL 0-1 vs degrees)
status: Done
created_date: 2026-09-04 05:23
assignee:
- '@claude'
labels:
- ui
- settings
- theme-editor
- ux-review-2026-09
priority: high
updated_date: 2026-09-04 06:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
textual.color.Color.hsl returns hue in the 0-1 range; _generate_theme_from_primary passes it to _adjust_color, which treats hue as degrees. Every primary therefore yields a red secondary, a cyan accent and a reddish background. Live: primary #9966FF -> secondary #e83735, accent #65fdff, background #161212. Generated hex is also lowercase while every other path is uppercase. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generate from Primary produces a secondary and background whose hue is within 30 degrees of the primary's hue, and an accent roughly complementary to it
- [x] #2 Generated colours are uppercase #RRGGBB like the rest of the editor
- [x] #3 A unit test checks the generated palette for a purple, a green and an orange primary
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
hsl.h * 360 before the degree-based HSL helper; generated hex uppercased. Parametrised test over purple/green/orange primaries checks hue distance and complementary accent.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
