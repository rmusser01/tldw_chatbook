# Prompt History journey review — TASK-32630

More actions → History now closes the action menu, reveals Info when entered
from Basic, opens the disclosure, and focuses its visible title. Advanced and
Info remain on their current projection. Mounted draft fields survive History
selection and paging.

History owns its focus handoff across local child replacement. Selection keeps
the version row focused; loading older versions returns to the first appended
row; page failure offers a keyboard Retry. Restore begins after modal return,
so disabling its opener cannot send focus to the global Home navigation.
Cancel and retryable outcomes return to Restore; successful adoption returns
to the current History title. Newer focus and replacement Prompt scopes take
precedence over pending handoffs.

Focused multiline version rows require four rows: at three, the outline painted
over the version line and left only the change summary. The minimum now uses
`$ds-size-4`; no token values or retained-history service contracts changed.

## Verification

**226 targeted checks passed**, with no full repository sweep:

| Command (using `.venv/bin/python -m pytest`) | Result |
| --- | --- |
| `Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_prompt_history_journeys.py -k history -q --tb=short --show-capture=no` | 76 passed, 277 deselected |
| `Tests/UI/test_library_prompt_history_controller.py Tests/Library/test_library_prompts_state.py Tests/Prompts_DB/test_prompts_db_retained_history.py Tests/Prompt_Management/test_prompt_history_normalizers.py -k history -q --tb=short --show-capture=no` | 124 passed, 188 deselected |
| `Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py Tests/Architecture/test_library_prompts_wiring.py -q --tb=short --show-capture=no` | 26 passed |

The 11 new production-CSS/real-SQLite journeys cover both 170×48 and 80×24 in
both themes, keyboard entry/selection/paging, literal read-only previews,
retained editor widgets, actual confirmation Cancel/Restore, identical restore,
page and DB failure/retry, dirty gating, and newer user focus during a held
page request. Existing checks retain snapshot-unavailable recovery, immutable
scope/request rejection, same-ID ABA protection, collapse during adoption,
compatibility gates, restore type changes and database atomicity.

The collapse-during-fetch check now waits for the outcome's mounted controls
before collapsing; service completion alone could leave it holding outgoing
busy controls. Running the controller checks in isolation also caught an eager
package import cycle from importing the canvas mixin into History. The region
uses the same post-recompose ordering locally, without importing the widget
facade. [Lint comparison](lint-results.json) reports no new diagnostics; the
new journey file and History region pass Ruff and format checks. The large
existing files retain their baseline diagnostics. Each pytest invocation
reported two pre-existing temporary-directory cleanup warnings.

## Native app and persistence

The final run used the real `TldwCli`/`LinuxDriver` in an owned tmux session,
with isolated configuration and all 12 configured storage roots contained in
the private profile. Profile ownership was asserted. The runner created 12
versions through Save at each measured size, opened History from Basic, loaded
the ten-row first page and two older rows, selected the oldest snapshot,
visited its read-only body, canceled confirmation, then restored it.

[Native results](native-results.json) record 170×48 dark and 80×24 light, normal
Ctrl+Q, `app.run` returning, exit 0, and the observed zsh shell at 80×24. The
owned session was closed after verification. [Read-only SQLite evidence](persistence.json)
confirms both final Prompts are active at version 13 with the version-1 body,
all 13 retained versions present, and the oldest snapshot still intact.
The [runner](native_check.py) takes a prepared private profile, tmux socket and
session as arguments; it is deliberately not pointed at personal configuration.

All six SVGs were rendered and visually inspected:

| State | Wide dark | Compact light |
| --- | --- | --- |
| Oldest selected row | [Selection](selection-170.svg) | [Selection](selection-80.svg) |
| Cancel returns to Restore | [Cancel](cancel-170.svg) | [Cancel](cancel-80.svg) |
| New current version and focused History title | [Restored](restored-170.svg) | [Restored](restored-80.svg) |

The existing success toast remains transient and can cover lower history rows;
the restored History title is readable above it. Native service failures and
unsupported snapshots were not injected in this run; their evidence is the
targeted tests. No external provider, full-suite, push, merge or PR verification
is claimed. Integration into `dev` remains pending.

ADR required: no. This applies existing ADR-049, ADR-086, ADR-150 and ADR-161;
storage, authority, service contracts and restore semantics are unchanged.
