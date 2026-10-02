---
id: TASK-33530
title: Move log file I/O off the event loop without losing crash-time lines
status: To Do
created_date: 2026-09-29 19:45
labels:
- performance
- logging
- perf-audit-2026-09
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split out of PERF-03 (TASK-33262). Its AC #3 (file and Logs-buffer handlers do no file I/O on the event-loop thread) was first met with a daemon log-writer thread, which review removed. It saved about 80 us per INFO record, but brought six hazards. (1) The last records before os._exit, os.execve (recovery restart) or a fatal signal were lost. (2) A timed-out drain let close() and unmount close handlers still in use. (3) stop() switched concurrent callers to direct writes before the queue drained, so records could land out of order. (4) A record queued between a maintenance drain and the handler's closed flag was skipped. (5) Unmount joined the thread on the loop for up to 5 s. (6) Logs-buffer entries were lost when loop callbacks were still pending at shutdown. Sinks now write synchronously again, as before PERF-03, with one redaction per record. Any new attempt must close all six, and must first measure whether loop-side log I/O still matters after PERF-03's volume demotions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A measurement shows how much event-loop time file logging costs per Console interaction after PERF-03, and the task proceeds only if it matters
- [ ] #2 Records logged before os._exit, os.execve or a fatal signal are on disk (test with a forced exit)
- [ ] #3 close(), a maintenance pause and unmount never close or skip a handler with accepted-but-unwritten records, and never block the event loop
- [ ] #4 Record order is preserved across writer start and stop
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
