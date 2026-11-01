# PR #3166 review and rebase verification

Reviewed base: `4de437a5c5c586bb1afea7c727cc2998752754d3` (dev).
Reviewed head: `f152dfc4a6c4fbf2fcd277543757a46a0d2874e2`.

The complete twelve-commit PR was rebased onto dev. Three conflicts were
independent additions to the two lessons documents; both sides were retained.
Range-diff confirmed that these were the only commit-patch differences.

An independent reviewer received the exact Git range and requirements, with no
session history, and reviewed production code and meaningful tests in passes.
They found no critical issues and one important issue: failed tokenizer
preparation hid recorded Current spend as well as optional estimates.

## Finding and correction

The cost display now builds measured-only history while preparation is unavailable.
Rows needing estimates still defer the entire total; no partial realized amount
is presented as the full Current spend. Provider usage, auxiliary usage and fleet
tokens retain their original aggregation. Historical payload projection,
fingerprinting, cache-break estimates and stale context/Next-send presentation
remain gated. Readiness joins the settled-cache key, so readiness transitions
cannot reuse an estimated or incompletely projected snapshot.

The new regression failed on the reviewed code (one failure, one passing mixed-history
control). After correction, 68 focused cost-tracker, spend-projection, tokenizer
and new-regression checks passed. Both original formatting and storage checks
remain in place. The reviewer checked the correction and reported no remaining
blockers in the change.

## Rebased validation

Native Windows verification selected the changed storage/worker/CSS/UI functionality,
using genuine private fixture ownership. The broader run completed with 126 passes
and one skill-affinity assertion failure. An isolated recheck passed in 40.21 seconds;
the only test change added the actual thread IDs to a failure diagnostic. This is
a successful recheck, not proof that the initial failure's cause was fixed.
The broader run and focused cost run overlap and must not be added as distinct cases.
No full-suite sweep was run.

Commands use `.venv/Scripts/python.exe` and the native private-fixture runner:

```text
Docs/Development/evidence/windows-performance-2026-10-18/matching_tests.py <changed storage, performance and UI files> -q --timeout=180
Docs/Development/evidence/windows-performance-2026-10-18/matching_tests.py Tests/Performance/test_console_skill_context_owned.py -q --timeout=180
Docs/Development/evidence/windows-performance-2026-10-18/matching_tests.py Tests/UI/test_console_measured_cost_without_tokenizer.py Tests/Chat/test_console_cost_tracker.py Tests/UI/test_console_spend_projection.py Tests/UI/test_console_tokenizer_preparation.py -q --timeout=60
```

The reviewer did not rerun performance probes or independently reproduce historical
baseline failures. Existing raw measurements remain the timing evidence; no new
speedup claim is made by this review fix. The earlier abnormal diagnostic exit
remains unclassified. Withholding optional rough estimates on failed backend
preparation remains the documented conservative policy; recorded usage is distinct.
No storage authority, permissions, source checks, admission or lifetime policy changed.
ADR required: no; this is a correction within existing ADR-230/231 display boundaries.
