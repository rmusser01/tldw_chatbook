# RAG Real-Embedding Capability Gate Design

## Goal

Keep the TASK-19642.3 RAG verification nodes offline by default without turning
the real-model performance benchmark into a mocked or vacuous test.

## Root Cause

The benchmark passes as an isolated control because `Tests/RAG_Search/conftest.py`
establishes offline environment defaults before Hugging Face evaluates its
module-level constants. The thirteen-node inventory run exposes the defect:
earlier RAG collection/imports evaluate those constants first, so the later
environment defaults cannot change the already-frozen client state.

Twelve inventory nodes then complete normally. The remaining node,
`TestEmbeddingPerformance.test_real_model_performance`, requests the
session-scoped `real_transformers_session` fixture before it discovers that the
real MiniLM model is unavailable. That fixture tries to initialize
`hf-internal-testing/tiny-bert`, and the benchmark then tries to initialize
`sentence-transformers/all-MiniLM-L6-v2`. Hugging Face catches and retries the
guard's `OSError`, so the test body skips while teardown correctly reports the
swallowed egress attempts. The capability gate fixes this order-dependent path
without relying on mutable Hugging Face import state.

## Design

Use the existing `TLDW_RUN_REAL_EMBEDDINGS` capability switch to gate the real
performance benchmark before its body requests `real_transformers_session`.
The gate is a combined collection/setup-time `requires_real_embeddings`
`pytest.mark.skipif` marker in `Tests/RAG_Search/conftest.py`. Its condition
covers both the existing embedding dependency check and the explicit capability
switch; its reason remains `Embeddings dependencies not available` when the
dependency is missing and otherwise identifies the disabled
`TLDW_RUN_REAL_EMBEDDINGS` capability. The benchmark replaces its existing
`requires_embeddings` marker with this combined marker, avoiding stacked and
potentially ambiguous skip reasons.

The benchmark also receives `pytest.mark.integration`, matching the existing
real-embedding integration suite and satisfying the task's explicit
integration-classification requirement. The integration marker labels the
test; `requires_real_embeddings` is the behavior that prevents initialization.

The default run therefore performs no transformer warm-up and no real model
construction. It reports the benchmark as an explicit capability skip. When
the capability is enabled, the benchmark remains unchanged and continues to
exercise the real embedding implementation rather than a test double.

This change does not weaken or bypass the repository-wide network guard. It
does not add `allow_network`, modify production embedding code, or alter the
other twelve inventory nodes.

## Error and Capability Behavior

- Missing embedding dependencies continue to skip with the existing
  `Embeddings dependencies not available` reason.
- Available dependencies without `TLDW_RUN_REAL_EMBEDDINGS=1` skip before any
  real-model fixture or background initialization begins.
- An explicitly enabled benchmark can run from an already-populated local model
  cache. On a cold cache, the unchanged repository network guard still blocks
  Hugging Face egress and reports the attempt at teardown even if the benchmark
  catches the model-load exception and skips. Download authorization and a
  capable integration environment are separate from this offline-default task;
  this change deliberately does not add `allow_network` or weaken that guard.

## Verification

Use the current failing node as the TDD regression:

1. With `TLDW_RUN_REAL_EMBEDDINGS` and `TLDW_TEST_ALLOW_HF_DOWNLOADS`
   explicitly absent from the command environment, run the benchmark alone as
   the order-dependence control. Expected: one ordinary model-unavailable skip
   and no guard error.
2. Run all thirteen exact TASK-19520 inventory nodes together under the same
   environment. Expected RED: twelve passed, the benchmark body skipped, and
   teardown errored with recorded `huggingface.co` attempts (eight in the
   captured baseline; any nonzero count proves the defect).
3. Add the minimal gate and rerun the exact thirteen-node command. Expected
   GREEN: all thirteen complete without errors—twelve passed, one explicit
   capability skip—and no guard-reported network attempts.
4. Run Ruff and whitespace checks only for the modified test files and revision
   range.

No broad RAG or repository test suite is required because the user limited
verification to modified functionality.

## ADR Check

ADR required: no

ADR path: N/A

Reason: this is a test-only correction applying an existing capability switch
and network-guard policy. It introduces no production, security, storage,
dependency, or cross-module contract decision.
