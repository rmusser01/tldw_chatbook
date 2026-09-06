# Pre-import budget repair

Task: TASK-31552, PR #2419. Owner approved paying down the inherited breach.

ADR required: no new ADR.
ADR path: backlog/decisions/097-boot-budget-ratchets.md.
Reason: defer imports within existing ownership and runtime contracts.

## Evidence and constraints

On dev 2c9c144181b9 plus the snapshot PR, the unchanged whole-registry census
reports 547 modules / 422,544 LOC (limits 500 / 380,000). The previous pristine
dev census also breached. Do not raise limits, remove routes, change accounting,
or shift these imports onto boot. Preserve Textual event classes and existing
test/consumer patch seams. No new dependency, storage or service boundary.

## Implementation and verification

1. Add cold-process regression guards before production edits and observe RED.
   Existing complete census failure also remains the aggregate acceptance test.
2. In `UI/Screens/library_screen.py`, move the measured Collections, Conversation
   Reader, Note Import, Notes Sync, RAG Search and Skills controller imports and
   note-import helpers to construction/use sites. Preserve static helper calls
   and annotation resolution. Do not change eager event classes or unrelated
   Conversations/Export compatibility bindings. Probe savings: 20 modules /
   33,989 LOC.
3. Independently defer `settings_rag_profile_adapter` through explicit forwarding
   functions in `UI/Screens/settings_screen.py`, retaining existing patch seams.
   Defer Tool Pack runtime services/modals to actual consumers, with type-only
   service imports in `Widgets/Settings_Widgets/tool_profiles_panel.py`.
   Probe combined Settings savings: 37 modules / 25,023 LOC. Keep event types eager.
4. Run new closure guards and complete unchanged census. Verify affected full
   Settings RAG/Tool Packs test files, Library controller/import/sync first-use
   tests and mounted screen lifecycle. Compare unrelated failures to pristine
   dev; do not broaden the repair based on inherited fixture failures.
5. Run all four boot-budget guards, scoped lint/format and derived-artifact
   reproduction. Apply ADR-097 tightening only if measured headroom exceeds its
   standard slack; never increase a constant or pin a breached snapshot.
6. Obtain independent code review, address findings with targeted regressions,
   update task/UAT evidence, push with exact lease and settle current-head Qodo
   review and required CI before merging the exact verified head.

Only targeted verification is authorized; no full repository test sweep.
