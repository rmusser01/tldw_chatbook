# Portrait compositing qualification — 2026-10-24

TASK-36551–36554 are implemented and reviewed. TASK-36555 has completed the
installed/restore/deterministic playback checks below; **real configured image-adapter
qualification remains open**. Existing ADR-243 applies. No new provider, dependency,
speech clock or recovery bypass was introduced.

## Installed application and restore

`installed-journey.json` records an actual wheel-installed TldwCli in a disposable,
normally admitted private profile, using production navigation and visible controls:

- Select/save all three regions, confirm a neutral base, then three component requests.
- Retry small mouth and eyelids through actual one-request confirmations; unaffected
  encoded roles stay exact and all six pairs pass component recovery/invariant checks.
- Apply a complete set, then start a fresh batch with a failed middle component.
  The partial candidate cannot Apply/sample; the previous active DB set stays exact.
  Retry the missing component, fail another retry, and retain the complete candidate.
- Open the actual Review pairs modal and compare every eye/mouth tile against the
  same source-coordinate crop of its original frame PNG. No provider, speech or write
  occurs during review. Observe all six roles through the shipped offline-cycle timer.
- Play the guarded sample and normal Console message through actual TTS service,
  WAV parsing, submitted PCM analysis, presentation ownership and avatar timers.
  Reopen exact settings/provenance/frame bytes and Remove without changing source
  artwork or saved regions. Context replacement rejects the retired speech owner.

There are ten deterministic image attempts: three initial edits, two component
retries, three fresh edits including one uncertain failure, one missing-component
retry, and one failed retry. Eight responses are recorded. Only image transport,
TTS synthesis transport and external device callbacks are fake; no test publishes
speech snapshots or invokes avatar ticks. The audio fixture honors the app's selected
WAV format and proves its decoded PCM was submitted to the real sink. This is **not**
new live TTS, acoustic output or hardware DAC evidence.

`native-restored-edit.json` records a separate actual F9 full-profile backup, isolated
restore and fresh wheel-installed app. Regions/provenance/frame hashes survive normal
admission. A differently encoded prepared PNG reuses the same region revision and
allows an actual confirmed one-request small-mouth retry. Old frames remain applied
until Apply; one changed RGBA byte and changed original source bytes each reject reuse.
Restoring original source bytes restores eligibility. No raw component sidecar is needed.

`package-evidence.json` identifies both wheels and checks all 18 changed production
files against their installed copies, including the actual v82→v83 packaged SQL.
`capture-hashes.json` hashes all 15 retained PNG/SVG captures. The tests assert exact
native crop pixels, source preservation and frame hashes rather than role counts alone.

## Artwork provenance and live limitation

`pair-review-observations.json` and `region-editor-observations.json` record the existing
Samira source, canonical prepared pixel identity, historical external ChatGPT edits,
actual production-CSS viewer/editor geometry and native raster hashes. The retained
wide/narrow captures show the real in-app viewer and region controls. These probes make
**zero provider calls**. Existing generated patches qualify local compositing and visual
inspection; they do not qualify an application's configured image adapter.

`live-adapter/availability.json` reports the actual default configuration: SwarmUI is
both default and the only enabled backend. This application's reference-image path
does not support that destination. **Zero live image requests were sent; no fallback
was selected.** Real alignment/lighting/semantic quality from three actual configured
in-app edits and selected retries remains unproved, so TASK-36555 AC5 stays unchecked.

For a future supported configured destination, inventory is safe by default:

```sh
PYTHONPATH=. .venv/bin/python Docs/superpowers/qa/portrait-compositing/2026-10-24/live_adapter_probe.py --output /private/tmp/portrait-adapter-inventory
```

An explicitly authorized run adds `--accept-five-image-requests` with a fresh output
directory. It uses the actual app/editor/pair-viewer/Console journey from this source
checkout and the configured image adapter for at most five requests: three components,
small-mouth retry and eyelid retry. It retains request/result hashes, raw responses,
frames/recipe and actual viewer captures. Credentials are held in a disposable mode-0600
config file, never embedded in subprocess arguments or retained evidence; keyring-only
credentials are not copied into the isolated null-keyring profile. Transport uncertainty
is recorded without automatic retries. Interruptions preserve safe artifacts and the
availability report before private cleanup, then re-raise the interruption. If an
interrupted manifest is incomplete, it is retained and `image_requests` is `null`
(unknown), rather than an invented zero. Existing evidence is rejected before dispatch.
Even a completed run reports `executed_review_required`, with `live_qualified=false`,
until a human/agent inspects the actual artwork and captures. Synthetic speech/device
boundaries remain in that probe and must not be described as live audio qualification.

## Verification

Commands use `.venv/bin/python -m pytest`, targeted paths and private basetemps;
`-q -o addopts='' -p no:cacheprovider --tb=short --show-capture=no`:

- Installed journey: **1 passed in 42.88s**, basetemp
  `/private/tmp/radio-installed-composition8`.
- Native F9 restore/fresh open/component retry: **1 passed in 52.81s**, basetemp
  `/private/tmp/radio-native-restored-component`.
- Probe inventory/success/uncertain failure/no-overwrite controls: **6 passed in 0.32s**,
  basetemp `/private/tmp/radio-live-probe-final3`; all transports fake or absent.
- Final affected core/draft, region repository/migration, character/Actor Pack/Chatbook
  exports, SQL inventory, token and CSS-sync gate: **99 passed in 40.85s**, basetemp
  `/private/tmp/radio-composition-final-affected2`, `--import-mode=importlib` required
  by the Actor Pack relative-import fixture. The initial default-import collection
  error was corrected without changing product code.
- Units 1–4 reviews retain their focused generation, race, widget, native-renderer,
  memory/cleanup and baseline-governance evidence. Unrelated component-governance
  offenders were reproduced byte-for-byte on pre-change source; broad legacy DB lint
  has no added diagnostics. These are not claimed globally green.
- Changed qualification Python files pass Ruff and formatting; `git diff --check` passes.

The installed harness initially exposed its own stale-avatar/input-retention assumption:
the widget intentionally releases its input record after preparation. The final check
uses the active saved record and hydrated `PreparedPerformance`, and observes real
speech-driven non-rest frames. No production playback workaround was added.

No full suite, new live image call, new hardware audio qualification, push or merge.


## Full-review corrections and refreshed evidence

The three follow-up findings are fixed and regression-tested; see
[the corrections review](../../../reviews/2026-10-24-portrait-compositing-corrections-review.md).
Manual authoring survives an unavailable generation reference, open dialogs retire
on persisted source/region changes, and interrupted qualification retains uncertainty
evidence before cleanup. A queued-validation race was also caught and fixed before
completion. The final installed and native F9 restore/edit journeys both pass with
real worker/render readiness barriers and the original timeouts.

`corrections/` preserves fresh installed/native results, both wheel/source/SQL identities,
zero-request backend inventory and verification counts. Earlier captures/results above
remain historical evidence rather than being relabeled as the corrected build. There
is still no live reference-image adapter execution or new hardware audio evidence.
