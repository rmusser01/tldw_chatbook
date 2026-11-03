# PR3346: archive confirmation lifecycle repair

Diagnostic published head: `97a53c67aefeb64fab42eca1592dc2cf7d6bff46`.
Verified dev base: `2dcf038cb8be0af991f92c43b0c262f02471c51d`.
The enclosing manifests identify the final corrected source and evidence bytes;
older captures keep their original phase attribution.

## Demonstrated defect and repair

Required admission-sensitive CI failed before the original resize journey:
FULL Console sync cleared a queryable archive confirmation card whose fixed
controls were absent. Native Compose/Mount/Unmount probes demonstrated
both pending publication and retirement. Textual8.2.8 sets the parent's mounted
flag after its mount handler, and the flag stays latched during retirement.
A subsequent real Select-Unmount probe found all six outer controls still
queryable and mounted while the Select's internal overlay was already gone.
The reviewer reopened the initial clear verdict for that concrete Important gap.

The existing setter now adopts the newest payload first, then requires each of
its six typed controls to be mounted and outside native pruning/closing before
rendering. Its existing mount handler replays the latest payload. A boolean
reports actual completed projection; the sole view caller refuses deferred
acknowledgment. Exact session/request, allowed scope, expiry, one-shot decisions
and coordinator/source authority remain intact. No polling, callback, remount,
broad exception catch or permission change was added.

Two source-remount integration wait predicates previously called query_one
during a valid remove/mount gap. They now check presence before the exact original
display/request conditions. All original assertions, authority and deadlines
are preserved; the five-variant resize file is entirely unchanged. Existing
ADR166 applies; this lifecycle repair introduces no new architectural decision.

[Implementation and frozen qualification](repair/final-report.json) ·
[Independent final source review](review/review.md) ·
[Root gates and documentation review](review/root-gates-addendum.md)

## Verification on frozen final bytes

Separate overlapping cohorts are not added together.

| Cohort | Result and qualification |
| --- | --- |
| Original behavior, permanent lifecycle controls |8 expected failures in1.91s |
| Actual Select-Unmount regression before the final guard |1 expected failure in1.44s, SelectOverlay NoMatches |
| Final ordinary native cohort |34 passed in50.80s: complete26-case card module, all5 original resize variants and3 original recovery journeys |
| Final static/design/size guards |209 passed in19.39s |
| Repository preflights |Worker, markup537, diagnostics728 owners/16 sinks and UI census1337 pass |
| Scoped Ruff and formatting |All4 touched source/test files have zero diagnostics against zero97a baseline; new/final blocks formatted |
| Original five startup guards |5 passed,3 inherited headroom warnings in29.23s |

The9 new permanent cases use native Compose/Mount/Unmount and include latest
replacement/clear, truthful incomplete-projection refusal, stale stamped event
refusal and actual Select retirement. Final source hashes match before/after
the34-case run. The reviewer independently checked JUnit34/0failures/errors/skips,
all37 explicitly owned producer artifacts and the preservation proofs.

Early observations remain explicit: the first draft used unsupported
DOMQuery.filter(type), corrected before final qualification; an isolated original
resize observation failed its final band assertion for an unconfirmed reason;
the extra flow run exposed the test's absent-card predicate. These are retained
as failed/superseded observations, not final successes. No dictionary fixture,
dictionary authority or recovery guard was bypassed.

Startup metrics are unchanged: UI-ready1031/1031 modules; preimport554/558 modules
and426171/426199 LOC (**28 lines headroom**); largest Library route132480/135111
LOC; warmup93/94. Original CSS≤268 and worker/thread allowlists pass. All3456
captured runtime/bootstrap/Performance/affected native-test paths and the original
observer bytes match before/after. These are count/payload guards, not a fresh
latency measurement. Size/performance pins, workflows and the UI census are
unchanged; both affected UI files were already enrolled whole.

## Evidence and publication

`artifact-map.json` records68 explicit producer identities, exact source/retained
hashes, collision checks, reversible whitespace-sensitive wrappers and deterministic
gzip. Scratch profiles, databases and caches are excluded. Existing earlier
redactions and phase qualifications remain untouched.

All scoped source review findings are closed. Auto-merge was disabled when the
hosted failure appeared. Fresh protected CI on the corrected published head and
the actual merge are tracked on PR3346; no future hosted success is claimed here.
Qodo remains bypassed under the owner's explicit instruction.

Skipped: full repository sweep, live provider/cache-billing checks and a new
native terminal/font tour. Risk: startup payload and module budgets remain tight.
