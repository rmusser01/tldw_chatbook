---
id: TASK-23026
title: >-
  Exchange capture stores the whole conversation on every send, forever
status: To Do
assignee: []
created_date: '2026-08-27'
labels:
  - storage
  - database
  - privacy
priority: high
---

## Description

`messages_payload` is on `CAPTURE_REQUEST_ALLOWLIST`, so every send persists a blob containing the
entire conversation so far. The blob grows **2.8 KB at turn 1 -> 145.4 KB at turn 200**, totalling
**15.40 MB for a single 200-turn conversation**. Capture is **on by default**.

There is **no retention path**. The only purge is user-invoked and filtered to
`capture_detail = 'full'`; nothing hard-deletes conversations or messages in production, so the
`ON DELETE CASCADE` never fires and soft-deleted conversations keep their blobs indefinitely.

This is a storage finding, not a latency one - write cost is 0.05-0.20 ms. It matters because the
database it bloats is the one boot migrations and backups walk.

## Acceptance Criteria

- [ ] A long conversation does not accumulate a full copy of itself per turn - store a reference, a delta, or a bounded excerpt
- [ ] Existing oversized captures are reclaimable without the user knowing to run a manual purge
- [ ] Whatever capture retains is still sufficient for the debugging the feature exists to support - say what that is
- [ ] Growth re-measured over 200 turns after the change
- [ ] `omitted_keys` behaviour is reviewed: it currently omits only `api_key`, while the payload carries the user's entire conversation

## Evidence

Measured with the exact production kwargs shape (`_chat_api_kwargs_from_prepared`), reproduced
independently at 15.35 MB. An earlier probe reported this feature **clean** at 0.2 KB per turn - it
had been built from a hand-made input rather than the real caller's kwargs, and was refuted.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.
