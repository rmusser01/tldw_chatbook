- **Binding-wide semantic admission** — ADDRESSED. `tldw_chatbook/DB/Workspace_Project_Index_DB.py:389-446` probes every existing namespace under the group gate through must-exist read-only SQLite, validates the catalog, detects enabled consent, a ready reference, or retained generation/member rows, then positively closes and inventories each peer. `sql_unit` carries that result only through the finite write unit (`:475-523`), and `transaction` selects source-qualified admission for a semantic peer even when the target is keyword-only (`:535-548`). `allocation.py:156-161` refuses a second active store before preflight. The two-namespace, retained-generation, disabled-settings, absent-gate, and overlap tests are in `Tests/Workspaces/test_project_index_allocation.py:593-936`.
- **Status creates a missing lifetime file** — ADDRESSED. `tldw_chatbook/Workspaces/project_index/service.py:227-234,361-371` requests an existing-only lease and projects a missing one as Interrupted; Busy continues to project Building. `Tests/Workspaces/test_project_index_service.py:1885-1915` calls status twice and asserts the named file remains absent.
- **Credential-shaped persisted semantic settings** — ADDRESSED for the agreed Task1 syntax boundary. `tldw_chatbook/Workspaces/project_index/models.py:199-232` restricts provider/model keys, requires exact versioned references tied to those keys, and requires lowercase SHA-256 v1 fingerprints before `configure_semantic` serializes settings. `Tests/Workspaces/test_project_semantic_settings.py:207-266` rejects assignment, URL, mismatched references, and malformed digests, and verifies valid references round-trip. Actual catalog membership and fingerprint construction remain the explicitly deferred Task2 contract.

### New Breakage in the Fix Diff

None.

### Out-of-Scope Observations

- The reported raw connection census still fails on an extra `app_service_wiring._notes_sync_state_db_has_configured_roots` site. The implementer's report records that production source as byte-identical at FIX_BASE and HEAD; the fix only extends the expected C99 inventory range in `Tests/DB/test_private_sqlite_inventory.py:817`. This inherited census failure is non-blocking for this scoped fix review.

### Checks

- Inspected the provided `42a75186..439f905f` fix diff and current changed call paths; no git command or test was run.
- Verified the named receipt commands and outputs: `green-review-round1-direct-b` reports 17 passed and the inherited census failure; `green-review-round1-uncertainty-a` reports 3 passed including the real child status test; `green-review-round1-service-compat` reports 2 passed with child receipts; `green-review-round1-overlap` and `green-review-round1-disabled` report 1 passed each; `green-review-round1-peer-retention-b` reports 2 passed on the final DB source. `fix-round1-static-final-c` reports scoped Ruff check, format check, and diff check return code 0. Earlier green receipts used intermediate source hashes; the final retention correction has its own focused passing receipt.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage. Open findings: none.
