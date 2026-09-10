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
- The separate real-archive application/restart journey: **1 passed**, 25.82 seconds.
- Eight touched Python files pass Ruff check and format; all ten CSS outputs
  reproduce from source.

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
