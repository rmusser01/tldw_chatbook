# Independent Buddy management qualification

Date: 2026-09-10 · Creator: tldw-project

The production Textual application was run headlessly in a disposable profile
against the downloaded Petdex Homelander archive. This is application integration
evidence, not a native terminal or physical voice acceptance test.

## Observed journey

1. Open Console's composer Menu, then Buddy. Import the prepared native archive
   through Import pack & size and Apply. Exactly one independent Buddy appears;
   no Persona is created and its visual graph has no Persona owner.
2. Reopen management and use Create character. Prepare the real expression preview,
   scroll it into view, and observe six distinct rendered image hashes during
   Dynamic playback. Static retains one frame across 1.2 seconds.
3. Explicitly create the character with animation retained. Fourteen expression
   assets are saved; their file hashes match the recorded hashes. Cancel the
   management form afterward: the created character remains and the staged motion
   preference is not applied.
4. Remove the disposable import file, stop the app, and reopen the same profile.
   The saved Buddy, character, original sheet bytes, expression files and selection
   remain available without fetching the pet again.

The original artwork creator remains **Serhat**, source
`https://petdex.dev/pets/homelander`, with an unspecified license and empty notices.
The native archive SHA-256 is
`f68165117c54d274ad9a3b438891ce56e2b5d42ba08fb839831963fc43cd573d`;
its sheet SHA-256 is
`df5d026fcbb9587c11fee9aac52ef247c1496fbfc34e00a9511c4c98b6ae6ce2`.
No downloaded pet art or user profile is committed.

## Checks and limits

- Focused management/snapshot checks: **69 passed**, one optional external
  collection probe skipped.
- Adjacent Persona Petdex review, character conversion and animation checks:
  **56 passed**.
- The final Petdex source/conversion/publication, management and coordinator run:
  **113 passed**, 64.50 seconds.
- The separate real-archive application/restart journey: **1 passed**, 37.04 seconds
  on the source tree committed as `8aaadb56f1f8826cadfae743bbfdf1508b95cf49`,
  including dev `5a9371c4f3` and the final review fixes `503437d371`.
- Current application-shell, CSS-integrity and boot-budget checks: **30 passed**,
  42.96 seconds, on that same combined source tree.
- Eight touched Python files pass Ruff check and format; all eleven current CSS
  outputs reproduce from source, including the newly integrated Watchlists sheet.

The earlier scratch-harness failures were corrected in the harness: an offscreen
avatar intentionally pauses, the visual repository returns an `assets` collection,
and immutable artwork metadata needs a plain-dict copy for a JSON receipt. None
required a production behavior change. Existing environment warnings concerned
Requests dependencies, Pillow/datetime deprecations and unrelated pytest temporary
directory cleanup.

The committed workflow regressions are
`Tests/UI/test_buddy_management_petdex.py`,
`Tests/UI/test_buddy_management_modal.py` and
`Tests/Persona_Buddy/test_buddy_management_import.py`. They use local fixture art
to exercise cancellation, changed source/profile/selection and publication failure.
The live-source run additionally checks the actual downloaded archive's size,
animation and persistence through the production app. Physical microphone and
native terminal qualification remain in their existing dedicated tasks.

The final review fixes cover local imports when POSIX descriptor APIs are absent,
actionable path-free publication errors, and recovery when artwork publication
succeeds before settings persistence fails. Capability-forced tests cover the local
folder, JSON and ZIP fallback; an actual Windows host was not available. Cancel
before Apply publishes nothing. After partial Apply, installed artwork remains
durable with the prior settings, and retry reuses it.

Accepted ADR-139 is restored exactly. Supplemental ADR-146 records independent
publication and character creation, partially superseding only ADR-145's
saved-Persona-only destination restriction. The merged diagnostic inventory keeps
current dev's owners and adds the reviewed constant debug message for a failed
disposable avatar geometry remount; it interpolates no user content or paths.

All seven required derived-artifact preflight checks passed after that integration.
The scoped re-review confirmed all six original findings were addressed and found
one residual Windows junction gap in the new fallback. Root accepted that finding
and reused the shared filesystem identity check at initial and repeated directory
and leaf admission. Eight distinct modeled reparse cases now reject both new reads
and stale snapshots; the source/conversion/publication run passed **47 tests** in
2.42 seconds. Those cases failed before the fix (an initial ninth duplicate case
was removed). Ruff check/format and whitespace checks pass. The POSIX descriptor
route exercised by the application journey is unchanged by this residual fix.

Implementation rulings: integrate the approved unshipped feature history before
qualification because dev lacked Petdex; finish the quota-interrupted Task 2
locally with independent review; replace the plan's Accepted ADR amendment with
supplemental ADR-146; and close the remaining junction finding locally after the
single scoped final re-review. These decisions retain the approved behavior and
bounded review process; their costs are integration/review rework and an additional
decision record. No finding is deferred by those rulings.

## PR feedback qualification

Qodo's eight confirmed findings are fixed; its retry-key concern is contradicted
by a regression using real generated archives and publication. The complete
[feedback evidence](2026-09-10-buddy-qodo-verification.md) records each verdict,
the importer/playback test runs and a deterministic correction to a frame-assertion
timer race. Root reran the downloaded-source application journey on production
commit `84c567f4405dc9999082a5e242b41b9e219df585`: **1 passed in 29.98 seconds**.
The installer checks passed all **12** cases and all seven required local preflight
gates passed. The initial PR CI run additionally passed **784** fast-contract tests
and its required derived-artifact gate; the updated PR must pass its own gates.
