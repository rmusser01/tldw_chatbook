---
id: TASK-38612
title: Audio capture hardening batch from the 2026-11-02 audit
status: Done
created_date: 2026-11-02 14:30
labels:
- hardening
- audio
priority: low
updated_date: 2026-11-03 02:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two capture-path items: the VAD loop drops sub-frame tail bytes of every chunk for non-default buffer durations causing continuous small audio loss, and stop_recording can raise out of terminate when the join times out. See bug-sweep-2026-11-02-dev.md Audio findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VAD path carries partial frames across chunks instead of dropping them
- [x] #2 stop_recording wraps terminate so it cannot raise over callers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Pure _assemble_vad_frames helper carries sub-frame tails across chunks with zero sample loss; pyaudio terminate wrapped like the class __del__ guard preserving the success path. Commit 903a7305d3.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
