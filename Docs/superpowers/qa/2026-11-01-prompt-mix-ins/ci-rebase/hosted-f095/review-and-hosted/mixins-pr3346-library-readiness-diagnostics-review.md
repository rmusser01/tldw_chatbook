# Independent UI19 Library failure-diagnostics review

Verdict: no actionable Critical, Important, or Minor findings in the settled test-only change against HEAD `f09597cdd5a2b5af1b07dea7eabe2eb6d33d9c80`.

Scope: `_library` in `Tests/UI/library_note_links_support.py`, the new present/missing-row diagnostic controls, and retained targeted evidence. No product or test edits and no test runs were performed by this reviewer.

The healthy-source DOM wait is wrapped in an AssertionError handler without changing its callback, arguments, implicit timeout, short-circuit predicate order, or prior source Retry flow. On failure only, the helper snapshots readiness booleans, render/state generations, row count/parent-active state, and current-screen type/id. It adds an exception note and re-raises the same original error. No source body, note text, configuration secrets, or automatic instruction contents are included. It performs `query_one` only when the row query is nonempty and introduces no await between these diagnostic reads.

The new two-case test validates both row-present and row-absent paths, a single unchanged readiness-predicate invocation, the exact original AssertionError identity, expected metadata, and absence of query_one for a missing row. It stubs the failing wait solely in this negative control; the existing native link journey is unchanged and remains separately exercised.

Evidence inspected under `/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/mixins-ci-f095-ui19-l34gz_tj`: `diagnostics-covering.log` records **3 passed in 30.54 seconds** (the real keyboard picker/link insertion journey plus both diagnostic cases). `static-proof.json` records exact wait/predicate AST equality and no added lint diagnostics; the old support-file I001/B023 findings remain explicitly inherited. Two existing splash-screen escape warnings were visible in the native run.

This is a diagnostics improvement, not a claimed fix for or root-cause attribution of the unreproduced hosted UI19 failure. The original timeout remains a failure and will now carry additional state for diagnosis. No readiness/authority assertion was relaxed.

## Reviewed working-tree SHA-256

- `Tests/UI/library_note_links_support.py`: `27fc9262e89732e2fc93bf4564d2159a8f7da6502875ff1418e7a72d3a9a532c`
- `Tests/UI/test_library_note_links_readiness_diagnostics.py`: `546fe8e103474859902128bccca92e0d040d55b089a51385492a455b51a558b9`
