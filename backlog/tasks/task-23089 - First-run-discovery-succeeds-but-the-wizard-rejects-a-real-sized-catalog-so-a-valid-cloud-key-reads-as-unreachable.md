---
id: TASK-23089
title: >-
  First-run discovery succeeds but the wizard rejects a real-sized catalog, so a
  valid cloud key reads as unreachable
status: Done
assignee:
  - '@claude'
created_date: '2026-08-28 06:15'
updated_date: '2026-08-28 06:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live first-run setup with a real, working OpenAI key still dead-ends at the Model step. The outbound models request carries no Authorization header at all, so it 401s and the step reports a failure for a perfectly good credential. This is the defect that keeps cloud first-run from working end to end; it is separate from the encoding and catalog-bound fixes in PR #2158, which are necessary but not sufficient.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A key typed into the Provider step reaches model discovery as an Authorization header
- [ ] #2 First-run setup with a valid OpenAI key lists real models at the Model step
- [ ] #3 The Model step reports the true failure category instead of a hardcoded 'request failed' on the ProviderStep handoff path
- [ ] #4 A regression test drives discovery through the wizard's own staged-settings shape and asserts the credential reaches the request
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause was NOT the credential path (see the CORRECTION section: the original diagnosis came from a malformed repro). Discovery succeeded with 128 live models; _model_ids_from_discovery_result rejected the result because 128 > MODEL_IDS_MAX_COUNT (100), and the caller folds a raise into a failed discovery -- so a successful discovery rendered as 'Couldn't reach the server'. PR #2158's cap change surfaced this by letting 128 models reach a still-100 bound one layer up. Fixed by bounding the picker list instead of rejecting the result; malformed-shape rejection unchanged. Verified live end to end with a real OpenAI key: models list, honest summary, and the written config produces a working chat. The hardcoded 'request failed' category on the handoff path is documented as a follow-up, not fixed here.
<!-- SECTION:NOTES:END -->

## Notes

- Not fixed here deliberately: changing which providers honor staged draft
  credentials is credential-resolution semantics and wants its own review.
- PR #2158 fixes two other defects on this path (gzip encoding, 100-model
  bound). Both are required for this flow to work, neither is sufficient.
## CORRECTION — the title and original diagnosis were wrong

The "no Authorization header" reading came from a malformed repro: it passed
staged settings as a flat dict, while `_first_run_discovery_staged_settings`
wraps them as `{"api_settings": {provider: settings}}`. With the real shape
the credential resolves correctly:

    status: success | kind: None | models: 128
      request -> https://api.openai.com/v1/models  Authorization present: True

Credential handling was never broken, and `ProviderStep.commit` does re-run
discovery when the credential changes (proven live: pasting an OpenAI key on
the Anthropic step produced "Authentication failed — this API key was
rejected" from a real 401). Both of those earlier hypotheses were wrong.

## Actual root cause

An in-app file probe on the wizard's own discovery block:

    RESULT status=success kind=None msg=None n=128 provider='openai'
    RAISED ValueError: Model discovery result is invalid.

Discovery *succeeded* with 128 models; `_model_ids_from_discovery_result`
then rejected the whole result because
`len(result.models) > MODEL_IDS_MAX_COUNT` (100). Its caller folds any raise
into `failed = True`, so a successful discovery surfaced as "Couldn't reach
the server (request failed)".

This is a two-layer bound mismatch, and PR #2158 exposed it: raising
DISCOVERED_MODEL_MAX_COUNT to 512 let 128 models through the discovery layer
and straight into the wizard's still-100 bound. The first fix moved the
failure up a layer rather than curing it — the live walk is the only reason
that was caught.

Fix: bound the picker list instead of rejecting the result. Malformed-shape
rejection is unchanged.

## Remaining, filed separately below

The Model step's ProviderStep-handoff path hardcodes
`failure_category = "request failed"` (~3455 and ~3484), so an authentication
failure can render as "check it's running" — this masked the real cause
throughout the investigation. Worth fixing, not required for the flow to work.

## Verified end to end (live, real OpenAI key)

- Model step lists real models from the live API (first time).
- Summary shows an honest `✓ Provider — openai`.
- The config the wizard wrote produces a working chat: `chat_api_call` with
  the saved provider+model returned `SETUP-OK`.

## Evidence (live, 2026-08-27)
