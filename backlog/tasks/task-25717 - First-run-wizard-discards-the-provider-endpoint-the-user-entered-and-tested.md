---
id: TASK-25717
title: First-run wizard discards the provider endpoint the user entered and tested
status: Done
assignee: []
created_date: '2026-08-31 05:08'
updated_date: '2026-08-31 06:35'
labels:
  - console
  - ux-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The provider step accepts an endpoint, confirms it with a successful connection test, and then does not persist it. The saved draft records only the provider key while the sibling voice step persists its own endpoint correctly. The one value that determines whether the product can reach a model is dropped after the interface confirmed it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An endpoint entered on the provider step is persisted with the rest of that step's draft
- [ ] #2 A successful connection test does not report success for a value that will not be retained
- [ ] #3 Completing first-run setup with a local provider leaves a configuration that can reach that provider
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
INVALID AS FILED -- mis-framed. Needs re-diagnosis before anyone codes it.

The observation was right (no `endpoint` under [first_run.draft_values.provider])
but the conclusion was wrong on two counts:

1. That block is a RESUME CHECKPOINT, not the applied configuration. The
   endpoint's real home is `api_settings.<provider>.api_url`, written by
   `persist_provider_setup` at commit time (Chat/provider_setup_persistence.py).
   Its absence from the draft means "resume will not repopulate the field",
   not "setup discarded your endpoint".

2. Retaining it there is forbidden on purpose. Adding `endpoint` to
   `_SETUP_DRAFT_FIELD_TYPES[STEP_PROVIDER]` broke
   test_dismissal_waits_for_irreversible_provider_executor_write, which
   asserts `endpoint_secret not in repr(provider_step.__dict__)` -- the
   endpoint is deliberately scrubbed from memory after commit.

THE REAL QUESTION, still open: in the live run the finished config had no
`api_settings.llama_cpp` block at all and the summary reported "no credentials
or saved endpoint" for a provider whose connection test had just succeeded.
That points at the provider COMMIT path, not the draft schema. Reproduce by
completing the provider step for a local provider and inspecting the written
config. Reverted.
<!-- SECTION:NOTES:END -->
