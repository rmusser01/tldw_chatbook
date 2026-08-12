---
id: task-15513
title: Surface the high-value server-only ingest options in server mode
status: To Do
assignee: []
labels:
  - library
  - ingest
  - parity
  - server
priority: medium
---

## Description

task-3309 closed the request-layer half of the server ingest parity gap: the
fields the Library sends are now ones the server actually declares. This is the
other half — options the server accepts that the Library has no control for.

Measured against a live server (see
`Docs/Design/2026-08-11-server-ingest-field-contract.md`): 48 declared form
fields cannot be set from the ingest canvas. Most are server-side machinery
with no local counterpart and are not worth a control. The 2026-08-07 parity
audit named the ones that are:

- `overwrite_existing` — re-import over an existing item instead of duplicating
- `keep_original_file` — retain the uploaded original server-side
- `custom_prompt` / `system_prompt` — per-import analysis prompts
- `generate_embeddings` — already sent by the service, but not user-settable

The design question this needs answered first: the Library ingest canvas serves
both backends, and none of these exist locally. So either the controls appear
only in server mode (a mode-dependent form, which the canvas does not currently
do) or they appear always and are gated with a reason in local mode (the
existing `— needs X` idiom). That choice belongs to whoever owns the canvas's
information architecture, not to the request layer.

## Acceptance Criteria

- [ ] A decision is recorded on whether server-only controls are mode-dependent or always-present-and-gated, with the reasoning
- [ ] The named options are settable in server mode and reach the server (asserted against the declared-field fixture)
- [ ] In local mode the same options are either absent or carry a reason saying they are server-only, never silently inert
- [ ] The remaining server-only fields stay recorded as deliberately unexposed
