# Independent UI29 optional-warmup fixture review

Verdict: no actionable Critical, Important, or Minor findings in the settled test-only patch against HEAD `f09597cdd5a2b5af1b07dea7eabe2eb6d33d9c80`.

Reviewed `Tests/UI/test_console_first_send_import_cycles.py`, the pyproject `web` extra, and `Utils/optional_deps.py` web availability classification. No tests or product edits were performed by this reviewer.

The fixture catches only `ModuleNotFoundError` where the requested name is exactly `textual_serve` and the exception names that same root. Missing required modules, missing dependencies within an installed optional package, child-module absence, ordinary ImportError, and import-lock RuntimeError remain fatal. Added controls assert preservation of the original exception object, not merely any failure.

The loop continues across the full production FIRST_SEND_MODULES sequence and explicitly records the absent optional root. A control asserts that later required imports still execute. The fresh-child missing-root branch installs its import finder before app/Console boot and asserts actual absence; it does not fabricate an imported substitute or remove the entry from the list. The environment branch retains the actual installed environment. The import helper used by positive controls is embedded from its actual source in the child, avoiding divergent duplicate implementations.

Both original cycle and dynamic-import checks remain unchanged and now execute against both closure variants. The graph walker, closure accounting, positive cycle-finder control, subprocess timeout, and production warmup remain unchanged. This repairs a test fixture for a documented optional extra; it adds no production exception waiver.

Evidence read: `/private/tmp/mixins-f095-cycles/tests.log` and settled diff. The implementer reports the local environment includes textual_serve; both environment and root-blocked branches passed. The log records **11 passed in 6.38 seconds**. A subsequent formatter-only parenthesization/wrapping change was inspected in final source. These are retained implementer results, not reviewer reruns or a claim that broader CI is green.

Reviewed working-tree SHA-256: `51e0fe80324f61303d40b7f9d37999ce216acb0eb6fc02688f14647a6f2a0a78`.
