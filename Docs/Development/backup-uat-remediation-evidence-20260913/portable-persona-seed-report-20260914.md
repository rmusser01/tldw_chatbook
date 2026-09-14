# Portable Persona seed — TASK-32562

Only Tests/Backup_Recovery/test_created_persona_subtree_rollback.py changes: 43 additions,2 deletions in the synthetic seed string. FROZEN and independently APPROVED. Full20 native cases passed482.20s; no failures or skips. Independent seed-only native privacy/marker/graph probe passed1case5.31s; /private/tmp/uat-portable-persona-seed-independent-review.md.

Exact Windows9e5914 RED is preserved in /private/tmp/uat-windows-9e5914-replacement: publisher.py179 explicitly refuses non-POSIX. One direct failure plus19 shared fixture setup errors occurred before backup. The unrelated publisher remains unchanged.

The corrected fixture represents legitimate already-existing Persona artwork. It uses the existing _snapshot PNG/manifest, publisher pure validators/asset-row/cleanup-marker builders, UUID nested pack/version layout, native private creators, and the real repository.activate_new_pack. It checks each payload byte-for-byte, PNG metadata hash/length, manifest hash and active repository graph. There is no new fixture format, direct SQL, mocked backup owner/guard/generation, or platform-specific skipped assertion. All20 actual capture/replacement/reopen/rollback and negative bodies remain unchanged.

Validation log /private/tmp/uat-portable-persona-seed-green.log:20PASSED482.20s. Scope is real native filesystem/database/UI calls against the current source checkout, not a newly installed immutable wheel. Full lifecycle capture/replacement/credential-review Abort/ordinary reopen+write/later rollback/restored reopen passed, plus all19 graph and retirement refusal cases. Independent AST comparison confirms no changes outside the seed string. Native Windows acceptance remains required after integration.

Ruff0 (/private/tmp/uat-portable-persona-seed-ruff.json). Bandit baseline/current both5B101 assertions; no new findings (/private/tmp/uat-portable-persona-seed-bandit{,-baseline}.json). git diff --check clean. Frozen source hash receipt /private/tmp/uat-portable-persona-seed-hashes.json. No product edits, commits, pushes or workflow dispatches.
