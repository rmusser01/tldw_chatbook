# Console approval integration and qualification

The approved interaction is implemented in the isolated managed checkout. It shows
captured action/targets/profile/location, immediate **Allow once** and **Deny**,
**More options**, counted one-time batch consent, complete lazy redacted
**Details**, neutral keyboard focus and owner-attributed feedback. Permission
lifetimes and gate/stamp/storage contracts remain unchanged. ADR-221 governs it.

The implementation is **not fully qualified for release**. Native/browser visual
inspection and the reported pause have no matched presented-frame measurements.
Tasks remain In Progress where their acceptance or definition-of-done gates are
open. This report does not claim either 100 ms feedback or 200 ms actionable-card
p95, a speed improvement, or a repaired freeze.

## Evidence

Only targeted checks ran, using the existing real-profile guard before HOME
redirection, fresh private roots, pinned module origins, offline models and null
keyring. Startup admission was checked before pytest collection. See the
content-free result summaries and hashes in receipts.json; private logs are
retained in this plan’s ignored workspace. No full application suite ran.

The final root checks passed: captured presentation 18, feedback reduction 29,
Details 18, interaction 21, artwork generator 5, recorder 16 and CSS bundle 5.
Task5 independently reviewed fix evidence passed owners 34, feedback 29, MCP 90,
trace 26, interrupt host 29, UI 9 and marker audit 2. Earlier provider and activity
receipts remain under ../task-5/ rather than being counted twice.

The regenerated guide SVG was rendered to PNG and visually inspected. It shows
Read file, notes/release.md, Writer, Chat scratch, one-time scope, Details,
Allow once, Deny and More options, with no timeout countdown. Post-fix generator verification passed all five checks. The generator now
uses the same theme-variable fallback as production, and its bare-theme test
prevents reliance on test-runner theme pins. Exported Textual frames are headless
visual evidence, not native terminal presentation timestamps.

## Preserved failures and limits

- Token governance: 7 passed and the existing Windows diagnostic-path assertion
  failed. It was reproduced on the unchanged baseline in Task3.
- Startup: lazy app/Tool Pack import cases passed; the UI-ready optional-service
  flag wait failed at line 769 on both HEAD and exact baseline 3ba1706c5f.
  The app had reached UI-ready; the subsequent optional-work flags did not all
  appear within the unchanged three-second wait.
- Component governance: 13 passed and the same five failures as exact Task3
  BASE. Their failure messages match; normalized Python-style diagnostics add
  and remove no entries. New central tokens remain legal; the existing Windows
  path-exemption error falsely counts token definitions. Existing Ruff/screen-size
  ratchets are retained, with earlier exact baseline attribution. No ratchet or assertion
  was lowered. Changed Task6 Python passes owned Ruff and formatter checks.
- Actual Windows virtual/local dispatch remains unqualified at the unchanged
  root-pin boundary. The native handle’s volume serial and Python 3.12 st_dev
  disagree despite matching inode/reparse checks. Task5’s local-provider run
  retained the same 46 failing IDs on its exact owner-source baseline.
- Details preparation has separate component characterization in ../task-4/:
  historical Task4 source: 40 warm first-page preparations, p95 about 153.7 ms, cold about 142.6 ms and
  peak about 1.07 MiB. That measures historical background preparation only, before the final target-metadata addition. It is not an
  actionable paint measurement or a before/after speed comparison. Complete
  scalar redaction still encodes the scalar; the two-page cache bounds retained
  display pages, not all input memory. No speculative performance repair followed.

## Native and browser qualification

[capture-preflight.md](capture-preflight.md) records the independent capability
check: no available browser, disabled native computer APIs and no supported
matched-state input/presentation observer. Existing ConPTY bytes, terminal cells,
compositor strips and server frames cannot qualify perceived paint. Separate
final-native-status.json and final-browser-status.json retain zero samples,
null distributions and qualified=false, with source hashes and missing stages.

Qualification still requires comparable private BASE/HEAD real-app runs on the
same hardware/load, all three sizes (80x24, 120x40, 170x48), Inspect open/closed,
dark/light and single/batch/raw-deny/large scenarios. Each warm distribution needs
at least 40 samples, nearest-rank p95 and separately reported cold observations.
Input arrival and the matching feedback/actionable presentation must share a
monotonic clock or a calibrated offset with bounded error. Last-output-to-card
cost and input queueing also remain unmeasured.

Primary browser research identifies possible measurement primitives, not a
working capture here: [W3C Event Timing](https://www.w3.org/TR/event-timing/) is
thresholded/coarsened next-render timing and may precede asynchronous feedback;
[W3C Paint Timing](https://www.w3.org/TR/paint-timing/) distinguishes paint from
implementation-specific presentation; [CDP frame-swap metadata](https://chromedevtools.github.io/devtools-protocol/tot/Page/#type-ScreencastFrameMetadata)
would need exposure, matching state and calibrated clocks.

## Decisions and review

[execution-decisions.md](execution-decisions.md) preserves all 14 controller
rulings and their costs. The temporary author-only gap was later independently
reviewed. Task5 fix1 review found all five findings addressed and no new material
breakage. The new integration files pass 9 scope cases and 4 painted grouped-card cases
in dark/light at 80x24. They connect controller settlement to actual permission
store/cache owners and private SQLite, with a harmless external transport double.
Existing FIFO/timeout/queued-navigation/late-page/replacement/filled-composer
Alt+A coverage is credited separately rather than duplicated. This does not
qualify native dispatch or the full native/browser size/theme/Inspect matrix.
[The final integrated review](final-review.md) identified two Important product gaps and one Minor clarity gap; the bundled fix wave and scoped re-review addressed all three with no new material breakage. [Final-fix evidence](final-fix/README.md) records the bounded preview, disclosure and compact action tests. Qualification blockers remain independently open. There is no PR, merge, push or release in this task.
