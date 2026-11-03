# f095 UI19 Notes readiness: investigation and failure diagnostics

No product fix or identified root cause is claimed. Hosted UI19 remains an intermittent failure not reproduced in the two meaningful exact-node runs. Two test-only files now make any future recurrence actionable without changing the readiness contract:

- Tests/UI/library_note_links_support.py catches only AssertionError around the existing healthy-DOM wait. It appends an exception note containing loaded/error-presence flags, state/rendered generations, reconcile-dirty flag, canvas-error presence, Notes row count/parent-active state, and current screen identity/type. It re-raises the exact same exception object. Query-one executes only when the row exists. No note bodies, paths, exception payloads or user text are captured.
- Tests/UI/test_library_note_links_readiness_diagnostics.py adds two no-startup negative controls (row present/missing). They verify the original false predicate is evaluated once, the original assertion survives by identity, the exact metadata is present and absent rows never reach query-one.

Final diagnostics-covering:3passed30.54s, including both unit controls and the unchanged actual Notes keyboard picker/autosave/backlink journey. No probe plugin is installed in this final run. static-proof.json proves the entire original healthy `_until` call—including predicate expression/order and omitted/default timeout—is AST-identical to f095. No retries, timeout allowances, expected booleans or source gates changed. Ruff delta support2→2 inherited (I001/B023), new test0; scoped diff-check clean. The new module is formatted. Parent owns publication and fresh hosted follow-up; source and test processes are stopped.

The following retains the earlier observational phase verbatim; its no-source-change statement describes that phase before the approved failure-diagnostics amendment.

---

# f095 UI19 Notes link readiness investigation

No repository source or test changes. The exact hosted failure is the healthy Library source DOM wait in library_note_links_support.py, after the healthy snapshot assertion. The hosted log does not identify which conjunct failed.

Ran one corrected observational exact node on current f095 production plus disjoint parent CI fixes: Tests/UI/test_library_note_links_picker_and_preview.py::test_typing_two_brackets_opens_a_keyboard_picker_that_inserts_a_real_link. Result1passed31.67s, preserving all note-link picker, keyboard, autosave/backlink and URL assertions. No timeout increase or source/authority substitution.

The private pytest plugin wraps only the test utility's `_until` for this named phase. It invokes the original predicate, records metadata when it changes, then returns its unchanged result to the original bounded wait. It does not change the app/controller/reconciliation callbacks, readiness fields, timeout or return value. Exact snapshot in predicates.jsonl: first observation true; loaded=true, lookup_error=None, rendered_generation=state_generation=1, entry_dirty=false, canvas_error=false, one Notes row mounted/displayed/enabled, parent_active=true, screen_current=true. All listed ancestors were displayed/mounted. Therefore no failed predicate was reproduced, and no product or fixture cure is claimed.

The first attempt (`observed`) failed in the probe JSON writer because Textual App.is_mounted is a method rather than Widget's boolean. That probe-only failure reached the predicate but cannot qualify application behavior. Serialization was corrected without application mutation; `observed-fixed` is the meaningful run. Both exact commands/logs/XML are retained. The actual private plugin is ui19_probe.py; its final hash is recorded below.

No further cohort or baseline run was started because no causal failure exists yet to attribute. A hosted rerun or a specifically approved predecessor cohort with the same metadata probe can narrow an intermittent failure; changing readiness/authority or adding time now would be unsupported. All processes stopped and parent/performance owner notified.
