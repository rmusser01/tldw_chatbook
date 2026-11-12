# Native Console refresh coroutine warning

Task: TASK-39435. Code:5f03c716fe plus followup98e8035315, in the existing shared dev worktree. Independent whole-fix review plus scoped correction approved spec compliance and quality, with no remaining product findings.

## Cause and correction

The mounted diagnostic completed its journey, then reported the exact unawaited ChatScreen._sync_native_console_chat_ui coroutine. CPython coroutine-origin tracking and tracemalloc both locate the original finally replay at chat_screen.py:17791, called from attachment reconciliation:15393. The printed textual/cache.py warning location is the receiver, not the allocation site.

An eagerly created coroutine can remain inside Textual Worker._work when its outer task is replaced or cancelled before entry. Native view-refresh workers now receive async callables, so creation occurs when the worker executes. Controller proxies use one local async late-binding forwarder, preserving the original return value and direct-await behavior. The followup fixes the same native callback stored as ConsoleContextReadSnapshot.refresh. Existing groups, nonexclusive/exclusive policy, currentness, error and teardown guards remain. No runtime owner, schema, dependency or visual styling changed.

## Verification scopes

- Origin-traced actual mounted RED: child fails on the warning after the journey completed,43.35s. The same diagnostic GREEN completes without target warning/journey error,38.10s, on its stated source epoch.
- Real Textual refusal, pre-entry cancellation and exclusive replacement regression: original3 native coroutine warnings; corrected path passes and publishes an actual session tab. A separate actual-awaited-dependency control verifies live sync errors propagate.
- Stored callback alias: RED1F/1P8.26s, then GREEN2P7.37s through actual for_screen binding and Textual scheduling, including cancellation before entry and admitted publication.
- Final source-pinned executions:4 focused worker/alias/mounted checks pass22.32s, plus public workflow refusal1P6.84s. Before/after13-file hashes match each other and committed source. These are separate invocations on the same final bytes, with overlapping earlier controls.

## Evidence limits

The earlier14P44.07 result has no contemporaneous launch-time source pins; its precise formatting epoch remains unproven. New final-source receipts provide current proof without rewriting that history. The interrupted controller selection's55F/157P are partial observed counts. Original native7F/12P, separate supported-bootstrap1F/6P and round1 covering1F/11P remain adverse receipts. Failure inventories distinguish the corrected readiness fake, profile refusal before screen construction, incomplete DB/mode-bar/doc fakes and an obsolete error-injection point. Source-order classification does not establish historical baseline passes. The later amended worker control checks the actual awaited dependency; the original stale assertion remains unchanged.

Scoped lint/format comparison shows no introduced diagnostics or changed-range formatter overlap, while inherited whole-file debt remains. No full suite, provider replay, cross-platform qualification or global pristine-output claim. Fixture retirement markers do not prove literal application restart. Earlier setup/trace-overhead attempts remain separately qualified.


## Review and retained decisions

[Implementation report](implementation.md), [initial review](review-initial.md), [approved scoped review](review-final.md), [complete rulings and costs](rulings.md), and [raw evidence manifest](evidence-manifest.json) retain the exact scope. The initial review found the stored-callback sibling; the followup corrected it and added its real-worker regression. Historical CV1/CV2 remain qualifications, not retroactively passing results.
