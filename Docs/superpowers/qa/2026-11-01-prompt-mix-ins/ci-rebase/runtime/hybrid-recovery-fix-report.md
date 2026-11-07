# PR3346 P0 hybrid recovery fix

Parent approved exact schema86 hybrid integration after the causal finding in `hybrid-recovery-investigation.md`, and updated TASK38544 plan before implementation. No manifests, commits, push, schema SQL or unrelated files were changed.

## Diff and authority

`DB/recovery_operations.py` adds six lines: import frozen CHACHANOTES_V86_SCHEMAS, compose those four complete lineages with each of the two existing Subscription index variants, register them alongside every retained predecessor, and map only those exact composed catalogs to embedded stamp86. Existing Subscription2/3 registration applies unchanged.

`Tests/Backup_Recovery/test_canvas_recovery_validation.py` adds86 to the existing exact embedded-stamp parameter/mapping. Existing tests still check both Subscription versions, reject mismatched Primary stamps85/87 for86, and prove validation/rejection leaves candidate bytes unchanged.

No SQL authorizer, trusted-schema scope, payload validator, catalog normalization, version-range rule, schema migration, or resource limit was changed. Current custom catalogs remain unsupported; corrupt payloads now pass exact catalog admission and reach the original integrity refusal. Actual native81–84 migrations validate their resultant86 shared file.

## Evidence

- Original current hybrid-valid RED:1 failed2.21s, unsupported_schema.
- Read-only catalog observer:1 failed2.82s; actual embedded stamp86,0 current policy matches, exactly1 proposed frozen86 composite match, no current embedded-stamp admission.
- Added86 stamp regression RED before product edit:1 failed,16 deselected in0.94s (`-x` intentionally stops first new failure).
- Final covering run:32 passed57.00s. All seven CI failures (current valid/corrupt/rewrite; genuine81/82/83/84 hybrid migration), current custom-schema denial, and24 exact-catalog/stamp matrix cases (83/84/86 ×4 lineages ×2 index variants). Each stamp case internally checks Subscription2 and3 plus wrong-stamp denials and unchanged candidate bytes.
- Scoped git diff --check clean. Test Ruff0 before/after and formatted. Source Ruff2 before/after,0 additions; source retains7 identical pre-existing formatter transformations,0 additions. No broad formatting cleanup.

The passing current valid/rewrite cases retain direct SQL denial, trusted_schema=0 outside the bounded pure-check scope, nested restriction ownership, resource deadline preservation and payload-row preservation. No full suite or unrelated recovery sweep was run.

## Owned artifacts

Additional repository ownership:
- `tldw_chatbook/DB/recovery_operations.py` (+6)
- `Tests/Backup_Recovery/test_canvas_recovery_validation.py` (+7/-4)

Private task reports:
- `hybrid-recovery-investigation.md`
- `hybrid-recovery-fix-report.md` (this file)

Private evidence:
- `/private/tmp/mixins-hybrid-red.log`
- `/private/tmp/mixins-hybrid-catalog-probe.log`
- `/private/tmp/mixins-hybrid-stamp-red.log`
- `/private/tmp/mixins-hybrid-green.log`
- `/private/tmp/mixins-hybrid-lint-comparison.json`
- `/private/tmp/mixins-hybrid-format-comparison.json`

Owned pytest basetemp directories:
- `/private/tmp/mixins-hybrid-red`
- `/private/tmp/mixins-hybrid-catalog-probe`
- `/private/tmp/mixins-hybrid-stamp-red`
- `/private/tmp/mixins-hybrid-green`

Input P0 log belongs to parent. The earlier UI6/UI16 reports identify prior four owned repository files and their artifacts; combined total ownership is six repository files. All other working changes are untouched by this agent.
