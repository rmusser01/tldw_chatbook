---
id: TASK-38612
title: Audio capture hardening batch from the 2026-11-02 audit
status: To Do
created_date: 2026-11-02 14:30
labels:
- hardening
- audio
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two capture-path items: the VAD loop drops sub-frame tail bytes of every chunk for non-default buffer durations causing continuous small audio loss, and stop_recording can raise out of terminate when the join times out. See bug-sweep-2026-11-02-dev.md Audio findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 VAD path carries partial frames across chunks instead of dropping them
- [ ] #2 stop_recording wraps terminate so it cannot raise over callers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
