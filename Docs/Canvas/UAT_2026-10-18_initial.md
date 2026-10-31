# Initial local Canvas UAT — 2026-10-18 (superseded checkpoint)

## Verdict: not signed off

The local storage/archive foundation passes its fresh acceptance checks, and
the isolated preview supports the tested interactions. The complete Console
journey is blocked by a reproduced startup crash. Additional served-terminal
assertion failures and one v1 timer-readiness failure remain unresolved.
These are not evidence that server synchronization works: sync remains disabled.

Across 41 unique selected candidate checks, 33 passed and eight failed or were
blocked after the narrowly scoped profile-setup requalification below. This is
a deduplicated ledger, not the output of one all-green suite. No case was skipped.

## Scope and environment

- Source: `03a043fc474660ff9cd9376af38cab7f9b8f73e6` on
  `codex/canvas-followups-20261017`. No product or permanent test code changed
  during UAT.
- macOS, CPython 3.12.11, SQLite 3.49.1, Textual 8.2.8, textual-serve 1.1.3,
  Python Playwright 1.58.0. Browser checks used the existing headless-shell 1234
  binary, reporting Chrome for Testing 151.0.7922.34; no browser was downloaded.
- Application launches ran under `Tests/conftest.py`'s disposable profile and
  real-profile write guard. SQLite and archives were synthetic, test-owned data.
  Provider responses were deterministic local doubles, not live model calls.
- The browser walkthrough used the existing owned TLS parent/real TldwCli child
  driver. Its test-only candidate policy was explicit. The browser connector
  accepted the owned self-signed certificate for its disposable target; the tab
  was closed afterward. This is not public-certificate qualification.
- The CLI wrapper was unavailable offline (`ENOTCACHED`), so the installed
  browser connector and existing Python real-browser journeys were used.

## Acceptance results

| Area | Result and boundary |
| --- | --- |
| New sibling-history foundation | 16 passed: genuine predecessor upgrade/rollback/retry, fresh-catalog parity, tied imports and serial authored allocation, huge-sequence refusal, real ZIP export/import/re-export and idempotent restore. |
| Preview/history/Temporary lifecycle | Browser/authority checks passed for promotion and destruction, exact archive reopen after source purge, multiple-canvas selection/rename/hot reload, and branch reachability. These do not replace a working full Console launch. |
| Bridge and downloads | Native gateway/authority checks passed for explicit payload confirmation, exact draft delivery, passive downloads and source-export warnings. Full Console composer integration remains blocked at startup. |
| Mermaid | All eight useful-example fixtures completed real v2 startup, including flow/sequence, mixed/four diagrams, Unicode and near-limit cases. |
| Sandbox | Benign counter/form/timer/SVG/data-image interactions and the complete v2 adversarial case passed. The v1 adversarial case failed its intermediate timer-readiness expectation; v1 does not receive a clean corpus verdict. |
| Browser-session isolation | The mounted URL non-transferability control passed after retaining its admitted private config source. |
| Full native Console create/auto-open | Failed: missing left rail in the resize handler during startup; no successful auto-open claim. |
| Full served Console create/update | All four plain/Mermaid and normal/read-publication variants failed to reach the initial Composer. Original child stderr identifies the same resize crash. |
| Served direct TLS / trusted proxy | Both lower-level two-child flows reached previews, copied-URL denial, confirmed submit/download, hot reload and branching, then failed the terminal text assertion for `CANVAS_LIVE_REOPENED`. The remaining reconnect assertions were not reached; neither whole case passed. |

## Findings and attribution

### Console startup crash — existing owner, no duplicate fix

`UI/Screens/chat_screen.py::_adapt_console_workspace_to_width` guards the grid
lookup, then queries the rails/handles outside that guard. During partial
composition the grid exists while `#console-left-rail` does not. The original
served child's stderr reports `NoMatches` at line 21523; the browser displays
"Terminal session ended" before any Canvas is opened.

The same native auto-open node fails with the same resize stack on the retained
pre-foundation `2b37746902a8` source. The affected screen, harness and browser test
files are unchanged across the foundation commits. Freshly fetched `origin/dev`
`b07fc0425288` still has the unguarded queries; no claim was made that a rebase
already contains a fix. The lifecycle defect and proposed qualification already
have an owner in TASK-25907.23; TASK-33621.36 also records related Console drift.
No change to those other workstreams was made here.

### Served terminal text/reconnect qualification

Both TLS modes fail at `test_canvas_served_flow.py:1283`: the terminal DOM does
not contain `CANVAS_LIVE_REOPENED`. The direct-TLS failure reproduces on the
pre-foundation source with the same explicit Chromium binary. Earlier Canvas
assertions passed, but the terminal-rendering/input/observation cause is not
established and the reconnect portion is unqualified. Do not delete the
assertion, substitute a direct action call or claim the whole journey passed.

### V1 timer readiness

The `timer_fire_rate_limit` corpus member reached `failed` before the test
observed the expected intermediate `ready` state. The v2 corpus passed. A single
same-browser baseline v1 run passed; that does not explain or fix the candidate
failure. No timeout, quota, expected status or security guard was changed, and
there is no new v1 security-clearance claim.

## Receipts and setup controls

Raw logs, JUnit, the disposable profile-marker plugin and control state are in
`/private/tmp/chatbook-canvas-uat-20261018.TXp1eb`.

| Receipt | Observed outcome |
| --- | --- |
| `native.xml` / `native.log` | 13 passed, two refused by config-source admission before the intended behavior. |
| `native-admitted.xml` / `.log` | One archive reopen passed; the native full-Console node exposed the resize crash. |
| `sibling-acceptance.xml` / `.log` | 16 passed. |
| `served.xml` / `.log` | Two passed, eight failed; includes one profile-setup refusal subsequently qualified below. |
| `session-isolation.xml` / `.log` | One passed with admitted profile retained. |
| `served-startup-diagnostic.xml` / `.log` | One failed; original child stderr captures the actual resize exception. |
| `base-native.xml` / `.log` | Same native resize failure on the pre-foundation source. |
| `base-browser.xml` / `.log` | Initial baseline attempt could not resolve Chromium under the temporary source path; not behavioral attribution. |
| `base-browser-explicit.xml` / `.log` | Same direct-TLS terminal assertion failed; the v1 corpus passed once. |
| `walkthrough.xml` / `.log`, `state.json` | Owned parent driver cleaned up successfully; its passing harness assertion is **not** a passing functional walkthrough. |

The temporary plugin applies the repository's existing `bootstrap_profile`
marker only to named live consumers. It retains the profile admitted at
collection rather than redirecting it mid-test; it does not bypass admission,
disable raw participants or change production code. Original failed receipts
remain intact. Comparisons use the same existing interpreter; the baseline
Chromium rerun explicitly names the installed binary because a source tree
under `/private/tmp` has no user-cache ancestor for the fallback resolver.

The browser snapshot and screenshot recorded the ended terminal. The screenshot
is local at `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.playwright-mcp/canvas-uat-20261018-TXp1eb-ended.png`.
Cleanup confirmed that the exact owned parent/child PIDs were absent and port
55419 was closed; no other browser tab or server was stopped. The connector's
certificate-error override was explicitly restored to false and detached.

## Workstream disposition

- User execution hold: server sync and the related client sync tasks
  TASK-35377–35383 remain To Do; no implementation is resumed implicitly.
- TASK-35490 is the requested To Do spike for alignment with `tldw_server`
  shared/project spaces. It requires a pinned contract/ownership/permissions
  inventory, a practical recommendation and a multi-user conformance matrix;
  filing it does not approve implementation or establish server support.
- TASK-35376's local data-foundation evidence remains valid, but this report
  withholds overall user-journey acceptance until the recorded blockers and
  remaining browser failures are resolved and the affected journeys re-run.
- No full suite, live provider, non-Chromium platform, companion-server test,
  normal-profile migration, dependency install, push, merge or sync activation.
  Existing baseline census/static debt remains outside this UAT. Schema/archive
  allocations still require revalidation before integration.
