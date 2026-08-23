# RAG Real-Embedding Capability Gate Design

## Goal

Keep the TASK-19642.3 RAG verification nodes offline by default without turning
the real-model performance benchmark into a mocked or vacuous test.

## Root Cause

Twelve of the thirteen inventory nodes already complete under the repository's
blocked-network guard. The remaining node,
`TestEmbeddingPerformance.test_real_model_performance`, requests the
session-scoped `real_transformers_session` fixture before it discovers that the
real MiniLM model is unavailable. That fixture tries to initialize
`hf-internal-testing/tiny-bert`, and the benchmark then tries to initialize
`sentence-transformers/all-MiniLM-L6-v2`. Hugging Face catches and retries the
guard's `OSError`, so the test body skips while teardown correctly reports the
swallowed egress attempts.

## Design

Use the existing `TLDW_RUN_REAL_EMBEDDINGS` capability switch to gate the real
performance benchmark before its body requests `real_transformers_session`.
The gate belongs in `Tests/RAG_Search/conftest.py` beside the existing embedding
dependency markers and is consumed by
`Tests/RAG_Search/test_embeddings_performance.py`.

The default run therefore performs no transformer warm-up and no real model
construction. It reports the benchmark as an explicit capability skip. When
the capability is enabled, the benchmark remains unchanged and continues to
exercise the real embedding implementation rather than a test double.

This change does not weaken or bypass the repository-wide network guard. It
does not add `allow_network`, modify production embedding code, or alter the
other twelve inventory nodes.

## Error and Capability Behavior

- Missing embedding dependencies continue to skip with the existing dependency
  reason.
- Available dependencies without `TLDW_RUN_REAL_EMBEDDINGS=1` skip before any
  real-model fixture or background initialization begins.
- An explicitly enabled benchmark retains its existing model-unavailable skip
  behavior after attempting the real initialization path.

## Verification

Use the current failing node as the TDD regression:

1. Confirm it reaches teardown with recorded `huggingface.co` attempts before
   the capability gate.
2. Add the minimal gate and confirm the node becomes an explicit capability
   skip with no guard error.
3. Run all thirteen exact TASK-19520 inventory nodes together under the default
   blocked-network guard. Expected outcome: twelve passed, one explicit
   capability skip, and no guard-reported network attempts.
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
