---
id: TASK-36699
title: Keep codec portraits stationary during mouth and blink playback
status: Done
created_date: 2026-10-25 19:52
assignee:
- '@codex'
updated_date: 2026-10-25 20:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The seven-character Pocket TTS recording exposed whole-image bobbing. At the native 24px fallback size, a one-pixel idle translation is conspicuous, and the enlarged comparison exaggerated it further. Keep portrait registration fixed while mouth and eyelid overlays animate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default live Console and editor performance playback retain a fixed portrait position throughout speech, pauses and blinks.
- [x] #2 Audio-reactive rest/small/open mouth selection and independent blinking continue to work with existing motion/accessibility settings.
- [x] #3 A fresh Pocket TTS recording of the seven selected characters confirms zero runtime portrait translation, and the enlarged comparison adds no translation.
- [x] #4 Targeted regression, CSS/token checks and review evidence are recorded; remaining timing/provider/hardware limitations stay documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR; existing ADR-242 and ADR-243 apply. This is a user-requested motion correction within existing playback/token boundaries. Amend their source-linked design description to stationary portraits.
1. Add a failing packaged-motion regression proving zero translation while mouth poses and blinks still change.
2. Set the existing portrait idle-offset token to zero and rebuild the generated CSS bundle; update motion documentation.
3. Run targeted playback/UI/CSS checks and make a fresh seven-character live Pocket TTS capture. The comparison must not add translation.
4. Review the change and footage, record limits, then update task notes/status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Set the existing portrait idle-offset token to zero and rebuilt generated CSS copies. Console/editor retain registration while mouth poses and blinks continue. Updated the design, ADR-242 note, architecture, design language and incident-based live-verification lesson. The packaged-clock regression failed on offset1 and passes on0, exercising all six roles. 67 targeted playback/UI/token/CSS checks passed; Ruff/format/diff checks and independent review passed. Fresh Pocket TTS/alba calls for all seven characters verified 2,993 zero-offset native observations, each byte-equal to its stationary prepared role. Corrected comparison adds no translation; MP4 decode/timestamp checks pass. Evidence: Docs/superpowers/qa/portrait-compositing/2026-10-25-stationary/; local recording in output/codec-stationary-recording-2026-10-25. ADR required: no new ADR; ADR-242/243 apply. Existing timing hitches, short reversals, low fallback resolution, physical graphics/audio-loopback qualification and live image-provider TASK-36555 AC5 remain open. No full sweep, user-profile edits, permanent voice assignments or publishing.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
