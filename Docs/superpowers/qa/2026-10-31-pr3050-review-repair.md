# PR3050 independent-review repair

Reviewed integration: `b43981e7bf4623fb8821aa994a91a8ed12f83716`, based on
`57e330e735fff25515778e839f31ed85602c34f2`.

ADR required: no new ADR. Existing ADR:
[230 — Console Send preparation and I/O ownership](../../../backlog/decisions/230-console-send-preparation-and-io-ownership.md).
These are corrections to existing owner-settlement and callback compatibility
contracts, with no new storage or authority policy.

## Findings and repair

1. Successful stock-receipt draft CAS used the direct composer projection while
   the active session could already be B and the visible composer still A.
   Removing A's accepted prefix before session settlement discarded B's typed
   suffix. Successful received and fallback CAS now use the same existing owner
   settlement helper. Unpressed receipts supply their actual input snapshot.
   The queued-receipt caller also forwards its successful CAS result.
2. Async capture was chosen even after the wired synchronous capture callback
   was replaced. Dispatch now honors that synchronous override inside its
   existing precapture context. A replacement during an async capture refuses
   the stale result before queue mutation or draft cleanup.

## Evidence

All receipts below are local sibling paths under `../test-runtime/`, use separate
private TEMP/TMP/config roots, and retain the repository's importlib mode and
original deadlines. Product code was unchanged during all RED runs.

- `review-fixes-red/result.xml`: 3 failed, 1 passed in 60.44 s. Typed received
  intent plus real runtime custody/publication lost the B-only suffix; fallback
  preserved it. Entry replacement selected async instead of sync, and replacement
  during an await queued the draft instead of refusing it.
- `review-queue-red/result.xml`: 1 failed in 32.44 s. Original received admission,
  configuration capture, queue mutation and successful draft CAS selected the
  unsettled projection instead of the settlement helper.
- `review-unpressed-red/result.xml`: 1 failed in 32.03 s. The unpressed receipt
  variant also lost the B-only suffix.

- `review-fixes-green/result.xml`: 8 passed, 1 failed in 210.13 s. Pressed
  received and fallback publication, both callback replacements, original queued
  receipt settlement, and the three direct queue/source compatibility controls
  passed. The unpressed variant failed at the existing initial Console
  reconciliation gate, before snapshot/custody/publication; the combined run is
  not reported as fully green.
- `review-unpressed-green/result.xml`: the isolated unpressed variant passed in
  65.79 s with unchanged product/test code, original deadlines and a fresh private
  profile. All nine selected cases therefore have passing evidence; this does
  not erase the combined-run startup failure or establish its underlying cause.
- Syntax compilation and Ruff F821/F811 checks passed for all five changed Python
  files; the final diff whitespace check passed.

## Limits

The mounted switch test deliberately isolates post-acceptance publication; it
calls the real publication/CAS owner but does not establish physical save or
provider execution. The queue test uses original admission/capture/mutation and a
projection observer; the mounted test establishes the helper's actual composer
behavior. Callback replacement tests use the existing dispatcher fixture and real
queue registry, not a native performance measurement.

CI and full-suite execution are explicitly skipped. This repair does not
requalify 100 ms/one-second latency, physical terminal feedback, the earlier
four-second native-read timing probe, or complete Close/platform behavior.
TASK-34601 and TASK-34601.1 remain In Progress.
