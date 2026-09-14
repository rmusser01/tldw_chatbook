# ACL restoration failure metadata

Frozen two-file test-only delta: the existing exact restored/native tuple inequality still raises. Only its error text now reports owner equality, original/restored control integers and XOR, revision integers, ACL lengths and equality. No SID, ACL content, hashes, paths, native API changes, or restoration policy changes. Native descriptor lifetimes and error chaining are unchanged.

New byte-valued and null-DACL mismatch cases first failed (2 failed, 8 deselected in 0.75s), then all helper tests passed (10 passed in 0.69s). They require exact bounded error text, verify excluded synthetic sensitive values and both descriptor releases. Existing four individual-field refusal cases and body/restore error tests remain passing.

Evidence: `/private/tmp/uat-acl-metadata-red.log`, `/private/tmp/uat-acl-metadata-green.log`, `/private/tmp/uat-acl-metadata-hashes.json`. Ruff check/format and diff check pass. Bandit helper has zero findings; test assertions are B101 only (11 baseline, 17 current; six added test assertions), with no other security findings. Native Windows descriptor mismatch cause remains pending the authorized diagnostic rerun; this is observability, not a restoration fix.
