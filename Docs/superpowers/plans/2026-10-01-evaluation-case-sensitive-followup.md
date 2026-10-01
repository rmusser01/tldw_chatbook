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
**Status:** Complete —20 targeted passes, zero skips; compile/format clean. Ruff155 inherited UP045 notices unchanged, no new findings. Bandit21LOW test assertions, zero production findings/errors.

## Stage 3: Publish
**Goal:** Publish a reviewed companion PR and link server UAT523.
**Success Criteria:** Independent review clear, normal commit and PR linked; no false hosted acceptance.
**Tests:** Diff check and fresh-source review.
**Status:** In Progress — independent scoped review CLEAR, no actionable P1/P2; normal commit and companion publication pending. Task criteria remain open until the PR is linked.
