## A watcher test must prove selection and await worker completion

**PR #2709 / TASK-32819, 2026-09-18.** The MCP stale-panel test intermittently
retained a `fetch` preview in CI while passing 20 isolated repetitions. Replaying
the inspector's deferred mount focus showed that Enter could land in its JSON
editor, leaving `fetch` selected instead of `search`. The subsequent preview was
then valid, not a stale-panel update. This watcher test now uses the table's
selection action to emit the real selection event, asserts the new owner before
releasing the simulated active run, and joins workers before checking cleanup.
Counted UI pauses do not join off-loop nonce cleanup. When keyboard routing is
not the subject, avoid making it an unverified prerequisite for an ownership test.

