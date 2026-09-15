# Canvas verified-byte checkout preservation

TASK-32562. Frozen scope: `.gitattributes` and `Tests/Canvas/test_profiles.py`; hashes `/private/tmp/uat-canvas-attributes-hashes.json`. No Canvas product code, generated resource bytes, profile capability, fixture override, or deadline changed. Existing Persona path fixture remains untouched by this scope.

The exact Windows e683 source/installed receipts demonstrate Git LF→CRLF conversion of byte-verified Canvas resources. Native strict loading of a disposable receipt-equivalent resource copy reproduces manifest-integrity rejection, empty application snapshot, then compiler runtime-profile refusal before the stale-policy race. Evidence `/private/tmp/uat-canvas-e683-profile-proof.json` and `/private/tmp/uat-canvas-e683-stale-policy-review.md`.

Ten exact `-text` rules preserve the committed bytes of the two catalog-selected manifests plus the union of their outputs and library files. There is no wildcard. The existing age_worker LF rule stays unchanged. Catalog/shell/authoring files remain outside these rules. Git documents that unsetting text prevents checkin/checkout EOL conversion: https://git-scm.com/docs/gitattributes#_text . No binary diff/merge changes or renormalization are introduced.

The new test derives its resource set from the actual catalog/manifests, copies actual repository attributes into a disposable repository, stages with autocrlf=false, removes the resources, and performs native checkout-index under autocrlf=true. Both profiles must then be executable under the unchanged strict native loader with retained runtime assets. The test does not mirror a ten-path rule list or change executable flags. Local Git environment is isolated from inherited Git variables/global/system configuration; commands use an absolute resolved Git executable, no network, and bounded waits.

- RED: exact manifest integrity failure, 1 failed in 0.79s, `/private/tmp/uat-canvas-attributes-red.log`.
- GREEN: all profile tests plus actual native stale-true policy test, 44 passed in 2.10s, `/private/tmp/uat-canvas-attributes-green.log`.
- Native git check-attr verifies all ten derived closure paths unset text, five excluded paths unchanged, and age_worker still text/eol=lf: `/private/tmp/uat-canvas-attributes-check.txt`.
- Ruff clean. Bandit adds only the new test's assertions; no new nonassert finding. Exact counts and source hashes in the receipt. Diff check clean.

No native Windows pass claim; the existing Windows selection must execute this exact byte-preserving checkout and stale-policy case. No commit/push performed.
