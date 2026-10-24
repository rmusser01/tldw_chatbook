### Spec Compliance

- ❌ Issues found: A keyword transaction in one namespace uses the legacy allowance even when another namespace in the same binding has semantic consent; semantic-qualified checks are selected only from the current database (tldw_chatbook/DB/Workspace_Project_Index_DB.py:436-445, tldw_chatbook/Workspaces/project_index/allocation.py:341). A status read can create a lifetime control file through the ordinary acquiring helper (tldw_chatbook/Workspaces/project_index/service.py:231,360). The settings boundary accepts arbitrary opaque strings such as an API token and serializes them to settings_json, so the credentials-never-stored rule is unenforced (tldw_chatbook/Workspaces/project_index/models.py:200-211, tldw_chatbook/DB/Workspace_Project_Index_DB.py:700-705).
- ⚠️ Cannot verify from diff: The pinned SQLite pager/VFS argument and actual native peak receipts live in the named private proof/receipt artifacts, not the product diff. The controller should assess those alongside the reported test results. The unchanged live disclosure path and scanner text queue bounds also span earlier code and later semantic tasks; this diff only shows affected call boundaries.

### Strengths

- The v4 schema is additive and the migration runs inside a transaction; new real-SQLite tests exercise preserved keyword rows and failed-DDL rollback (tldw_chatbook/DB/migrations/workspace_project_index_004.sql:1-7, Tests/DB/test_project_semantic_schema.py:40-130).
- The allocation owner keeps the native gate through close and inventory, checks a hand-calculated page bound, and rejects a negative allowance. Tests use actual SQLite and held native capabilities (tldw_chatbook/DB/Workspace_Project_Index_DB.py:362-411, tldw_chatbook/Workspaces/project_index/allocation.py:232-338, Tests/Workspaces/test_project_index_allocation.py:154-208).
- Read, scanner and capture boundaries were split into finite SQL units, with a real scanner-wait lookup test and targeted native contention controls (tldw_chatbook/Workspaces/project_index/service.py:242-260,464-515,1180-1270, Tests/Workspaces/test_project_index_allocation.py:278-565). The implementer reports targeted tests and scoped Ruff/format checks; I did not rerun them.

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- tldw_chatbook/DB/Workspace_Project_Index_DB.py:436-445: The choice between source-qualified semantic admission and the old keyword allowance reads only this namespace's semantic_control. A new keyword namespace can therefore write while an older namespace in the same binding retains semantic vectors, without the qualified SQLite/mode/older-writer checks required for competing writes. Decide semantic-qualified admission from the complete binding-group state under the gate and add a two-namespace test with semantic consent in one namespace and keyword mutation in the other.
- tldw_chatbook/Workspaces/project_index/service.py:345-365: A Building status probe calls _lifetime, whose acquire at line 231 defaults to creating the lock file. If the named file is absent, an ordinary status read allocates durable control outside a mutation, contradicting the read-only status contract and potentially changing quota inventory. Use an existing-only observation and project the absent lifetime as interrupted; test that status leaves the file absent.
- tldw_chatbook/Workspaces/project_index/models.py:200-211: provider_ref, config_ref and other persisted identifiers reject URLs and @ but accept credential values such as api_key=secret or a bare token. configure_semantic writes these values verbatim into settings_json at tldw_chatbook/DB/Workspace_Project_Index_DB.py:700-705. Define and validate reference syntax against secret-free configuration identifiers at this persistence boundary, and test credential-shaped inputs are refused.

#### Minor (Nice to Have)

- tldw_chatbook/DB/Workspace_Project_Index_DB.py:716: configure_semantic commits its CAS, releases its gate, and then reloads state. Another admitted writer can change the semantic revision or keyword scope before this reload, causing the first call to return the second writer's state or raise stale_scope after its own successful commit. Construct and return the immutable state from the transaction's known values; a synchronized two-writer test would pin the contract.

### Assessment

**Task quality:** Needs fixes

**Reasoning:** The schema and finite-unit gate are substantial, well-tested foundations, but group-wide admission, read-only status, and secret-free persistence are binding requirements. The return-value race is smaller but worth correcting while this API is new.

**Focused checks:** Inspected the changed diff plus the existing SQLiteStoreKernel._held_connection/close and BindingLifetime.acquire/close contracts to judge native close ordering and whether a status acquisition can create a file. No tests were rerun.


### Clarification for implementation (supersedes the Minor item above)

- Retraction — tldw_chatbook/DB/Workspace_Project_Index_DB.py:220-235,661-716: The @_activated wrapper owns an outer sql_unit around the entire configure_semantic call, including its post-transaction load_semantic_state. The nested units borrow that same live gate, so a cooperating second writer cannot intervene between the CAS commit and returned state. The Minor two-writer return-race finding is withdrawn. Updated counts: 0 Critical, 3 Important, 0 Minor.
- Persistence boundary for the third Important item — tldw_chatbook/Workspaces/project_index/models.py:189-220 and tldw_chatbook/DB/Workspace_Project_Index_DB.py:700-705: Task 1 need not construct a provider catalog or resolve references in the factory/runtime. A documented, versioned, closed grammar for provider_ref and config_ref is an adequate Task 1 boundary if it only represents code-resolvable configuration keys and excludes endpoints, userinfo, query strings, key-value material and freeform secrets. Apply bounded identifier grammar to provider_id/model_id appropriate to their intended IDs, and explicit digest/version validation to destination_fingerprint, config_fingerprint and approved_destination_fingerprint. Prove invalid credential-shaped/reference-shaped values are rejected before persistence and that valid references round-trip without raw configuration values. Task 2 must then resolve these references from code-issued catalog/config entries and keep actual credentials outside identity and stored payloads. A bare opaque ID cannot be proved non-secret by syntax alone; this clarification does not claim that it can.

