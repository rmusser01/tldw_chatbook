---
id: TASK-548
title: >-
  Console /rewind: inspector next-send preview should reflect boundary
  compaction
status: Done
assignee: []
created_date: '2026-07-24'
labels:
  - console
  - ux
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With `/rewind`'s "Summarize up to here" (PR #844), the actual send path compacts provider payloads at the dispatch choke point (pre-boundary turns replaced by the stored summary folded into the leading system prefix). The Run Inspector's read-only "next send payload" preview (`ConsoleChatController.build_context_snapshot`) builds its payload without `annotate_ids` and never routes through `_apply_context_summary_compaction`, so while a boundary summary is active the preview shows the full pre-compaction history — diverging from what is actually dispatched. The preview already deviates in documented ways (no skill substitution / world-info), but this gap defeats a user auditing "am I actually saving context?". Fix: apply the same compaction to the snapshot (annotate → compact → strip within the snapshot build), or clearly label the preview as pre-compaction and surface the compacted token delta separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With an active boundary summary, the inspector's next-send preview matches the actually-dispatched payload shape (summary in system prefix, pre-boundary turns absent) or is explicitly labeled pre-compaction with the compaction effect shown
- [x] #2 No `NATIVE_MESSAGE_ID_KEY` (or other private keys) appear in the previewed rows
- [x] #3 Preview behavior unchanged when no summary is active
<!-- AC:END -->

## Implementation Notes

`build_context_snapshot` now mirrors the dispatch choke point: the payload is built with `annotate_ids=True`, `_apply_context_summary_compaction` runs after the dictionary transform (exactly like the send path), and the private `NATIVE_MESSAGE_ID_KEY` is stripped immediately after — so with an active boundary summary the preview shows the compacted payload (summary folded into the leading system row) and the duplicated `system` field is now derived from the payload's own leading system row (falling back to the bare session prompt). Tests: compacted preview (pre-boundary rows gone, summary in messages[0] + system field, no private keys), no-summary unchanged, dangling-boundary un-compacted (leak-rule parity).
