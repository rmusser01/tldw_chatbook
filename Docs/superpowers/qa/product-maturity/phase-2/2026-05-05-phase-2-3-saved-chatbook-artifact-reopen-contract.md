# Phase 2.3 Saved Chatbook Artifact Reopen Contract

Date: 2026-05-05
Task: TASK-9.3
Branch: codex/product-maturity-phase2-3-artifact-reopen-contract
Workflow: Console-saved Chatbook artifact -> Artifacts provenance -> Console reopen payload

## What Was Verified

Phase 2.3 verifies the next narrow Phase 2 gate: a local Chatbook artifact record created from Console metadata can be recognized in Artifacts and reopened into Console with enough saved-response provenance to review the artifact without manual reconstruction.

Verified contract:

- Artifacts still selects the latest local Chatbook record from the local Chatbook service.
- Console-saved Chatbook metadata is identified by `artifact_source=console` and `artifact_kind=assistant-response`.
- Artifacts shows saved-response provenance before launch, including provider/model authority when present.
- Artifacts shows a bounded saved response preview before launch.
- Launching the artifact into Console preserves Chatbook id, record id, file path, tags, categories, source metadata, conversation id, message id, provider, model, bounded content preview, and truncation status.
- Empty local Chatbook state and local Chatbook service failure recovery remain unchanged.
- Newly exposed provenance strings are sanitized with the same no-script/no-event-handler boundary as existing Chatbook fields.

## Automated Evidence

Initial red run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_reopens_console_saved_chatbook_with_provenance -q
```

Result:

- `1 failed, 3 warnings in 10.91s`.
- Failure was expected: Artifacts displayed only the generic Chatbook description and did not show `OpenAI / gpt-4.1` or carry Console metadata into the launch payload.

Focused green verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_reopens_console_saved_chatbook_with_provenance -q
```

Result:

- `1 passed, 1 warning in 7.87s`.

Focused sanitization verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_reopens_console_saved_chatbook_with_provenance Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_sanitizes_chatbook_metadata_before_console_launch -q
```

Result:

- `2 passed, 1 warning in 6.94s`.

## Defects Fixed

- `workflow-degradation`: Artifacts could launch a generic latest Chatbook but discarded Console-saved artifact provenance, so the user could not tell whether the launched artifact was the saved response they intended to reopen.
- `recognition`: Artifacts did not expose provider/model authority or saved response preview for Console-created artifacts.

## UX Notes

- The design stays text-first and terminal-native: Artifacts adds concise provenance rows rather than a new selector or full artifact browser.
- The saved response content is bounded as `content_preview` so Console status cards do not render unbounded message bodies.
- The existing latest-artifact behavior remains intact; multi-artifact selection is intentionally outside this gate.

## Residual Risk

- This gate verifies the latest Console-saved artifact visibility and Console reopen payload, not a full multi-artifact picker.
- Home resume/open controls for newly saved artifacts remain unverified.
- Full `.chatbook` export packaging remains outside this gate; the local registry record is the authority used here.

## Exit Decision

Pass for Phase 2.3. Artifacts can now recognize and reopen a Console-saved Chatbook artifact with visible provenance. Phase 2 should continue with Home resume controls for saved artifacts.
