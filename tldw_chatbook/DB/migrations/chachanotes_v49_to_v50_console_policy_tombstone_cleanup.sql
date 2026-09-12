-- ChaChaNotes v49 -> v50: retire Console Library policy rows that have no live
-- conversation behind them (task-22225).
--
-- The v47 -> v48 step seeded `console_conversation_library_policy` from
-- `SELECT id FROM conversations` with no `deleted` predicate, so every
-- conversation a profile had ever held -- tombstones included -- received a
-- permanent row, written inside the boot version-bump transaction. That seed
-- now excludes soft-deleted conversations; this step is what brings a database
-- that ALREADY ran the shipped seed to the same state, so the two populations
-- cannot diverge.
--
-- The predicate is the read path's, not the seed's inverse: the repository's
-- `_POLICY_SELECT` joins `conversations` and `_result_from_row` fail-closes
-- unless `conversations.deleted = 0`, and both writers refuse a conversation
-- that is missing or deleted. Every row this removes is therefore one the
-- application already treats as absent -- including a row whose conversation
-- was hard-deleted with foreign keys off, which the same join drops. `NOT
-- EXISTS` rather than `NOT IN` so a NULL can never turn the whole predicate
-- unknown and silently delete nothing.
--
-- Removing dead policy is not a policy change: a conversation cannot be
-- undeleted through any path in the application, and a conversation that has
-- no policy row is an ordinary, supported state -- `add_conversation` has
-- never written one, and `ConsoleLibraryPolicyCoordinator.save` inserts
-- revision one on demand for a live conversation that lacks it.
--
-- DML only, no DDL: no new table, index, or trigger, so no allowlist or index
-- census entry is required. Idempotent -- a second application deletes nothing.

DELETE FROM console_conversation_library_policy
 WHERE NOT EXISTS (
     SELECT 1
       FROM conversations
      WHERE conversations.id
            = console_conversation_library_policy.conversation_id
        AND conversations.deleted = 0
 );
