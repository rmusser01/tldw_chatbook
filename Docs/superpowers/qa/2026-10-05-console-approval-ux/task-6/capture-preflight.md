# Independent Task6 qualification preflight

Read-only reviewer approval_baseline_review examined current supported tooling.
CUA inventory returned apps=[] and browsers=[]; documented iab selection and
createBrowserTab('iab','about:blank',{visible:false}) both returned
Browser is not available: iab. No tab/app was created or workaround attempted.
Native computer APIs are disabled. Available screenshot geometry has no presented
frame timestamp/error contract. Existing LinuxDriver/tmux captures contain SVG
and terminal cells; Windows ConPTY qualification observes bytes, parsing and
process lifetime. Character-switcher measurement observes compositor rendering.
None establishes when the identified approval state reached the native display.

No matching presented-frame/calibrated-input observer was found in existing
benchmark, script, browser-test or served UI sources. Current served UI rAF code
schedules repaint; it is not a presentation receipt.

Primary browser research:
- W3C Event Timing: https://www.w3.org/TR/event-timing/ — next-render latency is
  coarsened and thresholded, and may precede asynchronous server feedback.
- W3C Paint Timing: https://www.w3.org/TR/paint-timing/ — paintTime differs from
  nullable implementation-specific presentationTime; verify actual support.
- CDP frame-swap metadata: https://chromedevtools.github.io/devtools-protocol/tot/Page/#type-ScreencastFrameMetadata
  — not exposed by current CUA APIs; would still need matched-state attribution
  and a bounded measurement/error model.

Exact missing prerequisite: supported native terminal and served-browser observers
that identify actual input arrival and presentation of the matching actionable
and feedback frames. Use one monotonic clock or calibrated offset with bounded
error. Existing native/browser status receipts retain zero samples and
qualified=false. Task34569 AC3 cannot close under current capabilities.

After Task5: run actual owner/gate and mounted journeys, all modified targeted
checks, changed-file static analysis, token/component/bundle/startup guards; retain
BASE-attributed failures. Update the three guides, captured-view SVG generator
and its destination-validation tests; inspect generated artwork. Qualify all
three sizes, Inspect open/closed and dark/light per transport when supported.
Collect40warm samples per single/batch/raw-deny/large distribution with identical
BASE/HEAD conditions and boundary definitions; retain cold, missing and failed
samples, input queueing and full last-output-to-card gap. No qualified speed
claim, unrequested full sweep, guard changes or release action.
