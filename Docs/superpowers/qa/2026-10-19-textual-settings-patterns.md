# Settings Textual patterns qualification

Scope: TASK-35676, TASK-35677 and TASK-35678. Initial base:
`98c7eb308d46d25f8297fb40b1d23dc377e2c8e0`. Python 3.12 and Textual 8.2.8.
Only targeted tests ran, using the repository's Windows private-profile runner.
No real user configuration was modified.

## Reproduced event-origin defect

A value-changing programmatic paste-control refresh initially passed on a
clean category. With an unrelated `max_parallel_runs` draft edit, each control
failed the same mounted regression: the queued Changed event added an
untouched `collapse_large_pastes` or `paste_collapse_threshold` value to the
draft. The existing dirty key remained, masking the extra staged value.
Both cases failed on the original source, then passed with Textual's
`prevent(...)` around the two assignments, and passed after widget extraction.
This qualifies the concern from the book; it does not establish a defect in
the other temporary synchronization flags elsewhere in Settings.

## Ownership and compatibility census

- The moved region contains only the paste section heading, explanation,
  collapse checkbox, threshold row and help. Existing field IDs, labels and
  token-backed classes remain. The direct-child Static inset now also targets
  the panel's Statics, preserving its original geometry.
- `ComposerPasteSettingsPanel` owns composition, displayed values and local
  Changed handling. Its `Edited` message reaches the screen's existing staging
  helpers. Validation, category drafts and the configuration writer remain
  screen-owned (ADR-033); the panel has no config/app proxy.
- Field search still targets the same Input ID. The screen's sync method calls
  the panel's targeted sync; equal values leave the Input selection untouched.
  The threshold handler remains as a thin direct-call compatibility entry for
  existing tests, without a second decorated event subscription.
- The widget import occurs only when rendering Console Behavior, matching the
  approved screen-decomposition design. TASK-35017's page-order work is not
  implemented or displaced. No other section moved.

## Action and focus matrix

The before/after category rules are unchanged:

| Action | Supported categories | Other categories |
| --- | --- | --- |
| Save | Existing guided categories, plus Network's own writer | Existing guidance |
| Revert | Existing guided categories, plus Advanced Config's raw draft | No changes/guidance path |
| Test | Existing `TESTABLE_SETTINGS_CATEGORIES` | Existing no-test warning |
| Set active / clone / backfill | Library & RAG | Inert |
| Add / edit / delete / export record | Personal Context | Inert |

The pure support helper feeds dispatch, guided buttons, footer and F1 help.
It performs no config or database reads. Category-owned Theme and Personal
Context widgets retain their context-sensitive presentation. Dirty, busy,
validation and confirmation checks remain at execution time. Printable keys
continue typing in text fields; button/direct calls can explicitly bypass that
focus guard. ThemeOptionList retains its own `t` Try action.

One existing footer test expected the screen-level Theme action while leaving
focus in ThemeOptionList. The failure reproduced on the original source.
The test now explicitly releases widget focus before testing the screen binding;
its assertions are unchanged. Affected real-app tests also gained the required
private-profile marker instead of reading the developer profile.

## Behavioral evidence

- Original affected Settings run: 11 passed; the one Theme focus setup failure
  described above was then reproduced on the original source and corrected.
- Final contract run: 8 passed. This includes both programmatic-origin cases,
  compact/wide widget identity/selection tests, compact/wide field-search,
  actual typing and category-return journeys, the category capability matrix,
  and the corrected screen-binding test.
- Existing affected tests cover clean initial state, toggle focus, threshold
  save, staged Save/Revert, active-category help, printable keys, confirmation,
  clean Revert and narrow-footer/F1 discoverability.
- A full private-profile TldwCli served locally was inspected in the browser:
  field search found the paste controls; Enter changed the checkbox; `s` saved
  it. Typing threshold 1500 staged one edit. Revert opened its confirmation;
  Tab then Enter confirmed, restoring 1000 and leaving no unsaved changes.
  The private TOML contained the saved checkbox value and threshold 1000.
  The application log contained zero `unhandled_exception` records.
- Independent review inspected final event ownership, action support, CSS
  geometry and compatibility paths and reported no remaining defects.

## Validation boundaries

New files pass scoped Ruff lint and formatting. SettingsScreen has the same
13 pre-existing Ruff findings as its original source; none were introduced.
Generated stylesheet sync, worker-contract and Backlog guards pass. The new
regressions are appended to the existing PR UI census without moving earlier
entries. Full-suite verification was not requested and was not run.

The broader governance selection finished with 81 passes and 16 failures:
seven size violations in unchanged modules; existing dimension/Python-style/
Library-sheet violations; Windows path-separator and text-decoding assumptions
in guard self-tests; and the inherited boot-CSS byte ceiling breach
(624,809 / 608,090 bytes in this Windows checkout). Eighteen relevant guard
sources/subjects, including all five boot-parsed CSS sources, were compared
with dev `8136af6f6f` and are unchanged by this PR. The added selector is in the
lazy Settings sheet. No existing ceiling or allowance was raised. The screen
pre-import payload guard passed; Settings' own size ratchet passed both checks
and was tightened to the measured 35,805 lines.

Eleven derived-artifact preflight guards passed: generated stylesheets,
profile paths, persistent diagnostics, Backlog IDs/files, schema allowlist,
index pins, worker contract, timestamp writers, markup interpolation and UI
census. Canvas Mermaid reproduction requires the exact pinned Python 3.12.11;
the installed interpreters do not match that version, so this unrelated check
could not qualify locally. It was not bypassed or regenerated.

## Updated-dev integration

Rebased onto `8136af6f6f`, preserving its new Tools & MCP category. Its Test,
Revert and Add action branches remain before generic Settings guards, and its
live mode-specific footer/F1 projection stays instance-owned (ADR-240).
Independent review confirmed these branches are AST-identical to dev and
all 327 pre-existing UI census entries remain in order, followed by six new
entries (floor 333).

The first post-rebase rerun passed six cases before later child processes
refused the test root's newly broadened ACL. Writing a diagnostic helper with
the sandbox file tool had made that disposable parent group-writable. The
failure occurred during config import, before the test body. Restored the
exact owned `.test-tmp` directory to private mode using the application's
native Windows filesystem API, leaving all product guards enabled. Subsequent
diagnostic writes use the native command runner and do not widen that ACL.
The rejected runs are retained; they are not counted as behavioral passes.
