# Evaluation case sensitivity follow-up — TASK-33645

Server PR2979 has landed. Chatbook's typed EvaluationSpec omits case_sensitive, so explicit settings disappear from create/update JSON.

ADR required: no
ADR path: N/A
Reason: routine missing-field parity fix within the existing API contract.

## Stage 1: Reproduce
**Goal:** Verify current main and failing flag round trips.
**Success Criteria:** Original schema/client tests pass; new true/false/default controls fail for the missing field.
**Tests:** Existing two evaluation modules plus extended serialization controls.
**Status:** Complete — fresh main baseline16 passes; new controls5 failures/7 passes for the missing flag.

## Stage 2: Repair
**Goal:** Preserve the flag in the shared typed schema.
**Success Criteria:** Explicit true/false survive create and update; defaults and omitted partial updates remain compatible.
**Tests:** Targeted schema/client tests, Ruff and Bandit.
**Status:** Complete —21 targeted passes, zero skips, including public-client create and response-to-update payloads. The added boundary controls fail twice against the exact original schema. Compile/format clean; Ruff155 production UP045 and one client I001 are unchanged. Bandit137LOW B101 assertions across the two test files, zero production findings/errors; six added assertions and all131 originals preserved.

## Stage 3: Publish
**Goal:** Publish a reviewed companion PR and link server UAT523.
**Success Criteria:** Independent review clear, normal commit and PR linked; no false hosted acceptance.
**Tests:** Diff check and fresh-source review.
**Status:** Complete — companion [PR2950](https://github.com/rmusser01/tldw_chatbook/pull/2950) published and independently reviewed CLEAR, including the public-client coverage follow-up. The claimed missing server field already exists in landed server commit88f8b81, with schema and runner controls; no duplicate server edit is needed. Implementation/publication criteria are complete; normal merge remains pending at this recorded checkpoint.

Final evidence: `/private/tmp/pr2979-chatbook-2027-boundary-verification.json` and `-independent-review.json`. No full-suite, hosted, PostgreSQL or native acceptance is claimed.
