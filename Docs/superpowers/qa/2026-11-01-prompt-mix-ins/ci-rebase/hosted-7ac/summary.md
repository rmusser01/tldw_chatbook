# Hosted7ac CI follow-up

The follow-up fixes two reproduced lifecycle defects. The shared settings test
drain could repeatedly gather an already-finished task without advancing the
event loop, preventing its registered retirement callback from running. Yielding
after gather lets the real owner become quiescent without closing admission or
changing errors. The real-owner regression fails under the old helper and passes
with the fix; the exact Anthropic hidden-field/provider-wire journey also passes.

A queued readiness worker could look up a session after it closed, before its
existing try/finally. KeyError then left pending waiters unsettled. Hosted and
local runs reproduced the end-to-end hang. Moving entry setup into the existing
cold/refusal boundary preserves source, publication, cancellation and physical
read-retirement checks. A real-store race proves no native read or stale
publication, settled waiters, and a successful current-owner retry. The original
journey passes. No timeout, permission, cache lifetime or threshold changes.

## Targeted evidence

- Settings drain: bounded real-owner RED at5 seconds; pure regression and exact
  Anthropic UI/wire test GREEN2 passed in10.86s.
- Readiness close: unchanged original journey RED45.38s; deterministic race RED;
  six existing cancellation, physical-read, source and runtime controls pass;
  final corrected canonical-settings race and original journey2 passed14.31s.
  The initial retry-fixture mismatch is retained and not called green.
- Independent settings-drain and readiness-close reviews have no findings.
- Final five original startup guards pass31.98s: UI-ready1031/1031, preimport
  554/558 modules and426096/426199LOC, warmup93/94, original CSS and boot-worker
  guards. No snapshots, limits or allowlists change.
- Final size/layer/lookup/footprint checks216 pass; UI gate checker83 pass.
  Worker, safe-markup and persistent-diagnostic preflights pass without repins.
- Two pure/race regressions append to the existing1331 gate entries; floor1333
  records exact coverage. Existing shard positions and workflow timeouts stay.

These are phase-qualified, overlapping results; no full-suite pass is claimed.
Exact commands, source hashes and all failed/corrected iterations are retained
through explicit inventories. Large artifacts use deterministic losslessgzip;
raw log/patch and whitespace-sensitive Python-source wrappers preserve text exactly.

## Other results on the original7ac head

Perf Guard first failed only Settings arrival at10.1s against unchanged10s. Its
same-head rerun passed every step. Navigation, Settings, CSS and tour source were
byte-identical to the previous passing head. An exact local tour also passed;
its timing is diagnostic because a backup constructor test may have overlapped.
No performance code change or root-cause claim follows from that observation.

Windows eval recovery first refused the initial capture with scope_changed,
before maintenance, restore or rebackup. The full retained-definition journey
passed locally with identical193-row scopes. The complete hosted child log did
not identify a changed owner or field. Its same-head Windows rerun passed; Linux
and macOS had passed initially. Capture refusal and seals remain unchanged.

The7ac required run finished with39 successful jobs, the two confirmed UI failures
above, and its failing aggregate. Earlier UI3/16/19/28/29 follow-ups passed on7ac.
Fresh hosted confirmation of the lifecycle fixes remains separate from these
historical results. The PR stays draft/unarmed, with Qodo bypassed.
