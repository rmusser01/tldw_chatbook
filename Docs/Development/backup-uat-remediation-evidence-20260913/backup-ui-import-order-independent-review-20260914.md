# Independent shared-fixture import-order review

Source/lifecycle review approved; one trivial Ruff formatting correction remains at review time: add the requested blank line before the new module-level import comment. No reviewer edits.

Reviewed `Tests/UI/test_backup_restore_screen.py` against 4773628967. The only semantic change is moving the existing replacement_case import from two test bodies into module collection. An independent AST normalization removed only imports of that exact helper and proved the rest of the module identical, including every test, fixture, assertion and cleanup block.

The import chain is concrete: test_held_sqlite_rollback imports state from test_capture_sqlite_materialization, which imports application_authority from test_core_owners. That module explicitly imports Chunking._template_conversion before shared per-test HOME/config retargeting because Chunking module defaults are bound on first import. Tests/conftest.py establishes a private collection profile before tests, then its autouse isolation fixture changes HOME/config per case. This move follows the already-documented fixture lifecycle and removes dependence on another test module happening to import core_owners first.

replacement_case remains the same contextmanager and is still entered only inside the two original tests. Its directory, database, native admission, publication and cleanup actions remain inside that body. Importing its definition does not enter a replacement operation. No environment override, guard change, global cache clearing, test filtering or assertion weakening was added. Collection imports more of the already-used fixture graph up front; that is the intended correction and is scoped to this UI test module.

Author's fresh standalone outcome verified from `/private/tmp/uat-backup-ui-import-order-green.log`: 23 passed in 39.88s. Earlier combined ordering pass and standalone failures are author evidence, not independently rerun here. Independent AST/source checks suffice for this small move; no duplicate expensive native run was launched.

Pre-format file SHA256: 38514eb41c9fc4623a62db37758fd367ef8780a0d3fdb6c6eef0d35d0ff75aaa. Ruff reported only I001 requiring a blank line before the newly added comment at line8. Await final whitespace-only hash/check for unconditional staging approval.

Final approval: APPROVED with no remaining findings. Independently verified final SHA256 `14478147fe79498431431b3357b0a89603a0c85da2a36f3d04f4fa225a7f7932`; the only follow-up is the requested blank line. Ruff and diff check now pass. The source/lifecycle assessment and author 23-case native evidence above remain applicable without another runtime rerun.
