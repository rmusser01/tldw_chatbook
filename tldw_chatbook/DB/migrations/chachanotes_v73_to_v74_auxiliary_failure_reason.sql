-- TASK-33621.3: a failed Console compaction attempt was billed, marked
-- status='failed', and its reason code was thrown away -- the ledger could
-- say THAT a summary call failed but never WHY, so a user (or support) could
-- not tell a lineage fault from a provider error from an unusable summary.
-- Record the content-free reason code the compaction transaction already
-- computes. NULL for every row written before this step and for successes.
-- The CHECK mirrors ConsoleContextRepository.finish_auxiliary_attempt: a
-- reason only on a non-successful terminal status, shaped [a-z][a-z_]{0,63}.

ALTER TABLE console_auxiliary_attempts
  ADD COLUMN failure_reason TEXT
  CHECK(
    failure_reason IS NULL
    OR (
      status IN ('failed', 'cancelled', 'stale', 'timed_out')
      AND length(failure_reason) BETWEEN 1 AND 64
      AND failure_reason GLOB '[a-z]*'
      AND failure_reason NOT GLOB '*[^a-z_]*'
    )
  );
