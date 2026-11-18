# G72p Pilot census

This preserves the completed Windows diagnostic on product commit
`5fd0c8cd5ef01789aa5ba935a242e27a204da618`. Both unchanged original journeys
passed. The original UI and backend guards completed with no blocked
destinations. The optional census also completed: 505 Pilot barriers, 60 header
calls, no failed/truncated observation and no unfinished calls.

| Send → original reply seen | README | Approved edit |
|---|---:|---:|
| Original adjusted time | 25.0793 s | 28.9114 s |
| Raw window | 27.8421783 s | 33.0375923 s |
| Pilot waits started | 211 | 203 |
| Successful queued callbacks | 109,533 | 124,249 |
| `ConsoleActivityHeader.sync_header` calls | 21 | 39 |
| Header call wall union | 9.4001 ms | 11.9194 ms |

No `reply_completed` mark was emitted, so the endpoint is the original
`reply_seen` fallback, followed by successful original verification. Provider
boundaries came from the same UI guard's original event clocks: five README
requests and six edit requests. No mirror-arrival clock or cross-process time
subtraction was used.

Twenty README Chat samples had 360–401 hidden nodes among 502–575 children;
24 edit samples had 393–425 among 572–641. The fixed named groups include the
90-node hidden right rail and 85 hidden nodes in the left rail. These are sampled
`display:none` ancestry counts, not clipping/occlusion measurements. Groups
overlap and cannot be added indiscriminately. Existing initial selector contracts
prevent simply deleting the hidden Model children. The already-deferred
RunInspector had only its owner node.

This was an **independent diagnostic arm**, not a qualifying pair and not evidence
of equality with earlier runs. Pilot wait overlap (11.5934 / 16.2164 seconds)
includes app work serviced during waits; it is not exclusive Pilot CPU or a
possible saving. The observer's reported thread-CPU counter total was 0.71875 s.
Windows counter deltas quantized at 15.625 ms, so tiny per-call CPU totals are not
reliable; header wall time and counts are the useful measurements here.

## Reproduce the readout

`readout.py` is byte-identical to the completed arm's original readout.
`audit.py` exports only allowlisted aggregates and hashes, using these supplied
original numeric files:

- `run-metadata.json` and `timings.json`
- `lab-observer/ui-*.json` and `lab-observer/backend-*.json`
- `lab-observer/pilot-census-ui-*.json`

From this directory, using Python 3.12 or newer:

```sh
python -I -B audit.py /path/to/completed/run --output /path/to/new-result.json
python -I -B test_readout.py
```

The output file must be new. Inputs are read-only. This runs no application,
profile, network connection or benchmark. `g72p-result.json` contains the verified
result; `source-pins.json` preserves the exact product, original QA/guard and
diagnostic source hashes. The result also records all input receipt sizes/hashes,
the original detailed numeric readout hash and the portable readout source hashes.
The original Windows JSON has CRLF line endings; its byte hash is kept separately
from the canonical LF JSON hash so cross-platform reproduction remains exact.
No machine paths, raw events, prompts, dynamic widget IDs or private profile data
are included here. The supplied metadata may contain host paths; the reader never
exports them.

The live probe used code-local monitoring of the unchanged Pilot and named
header method. A one-shot original `Trace.install` return discovered the existing
objects without early product imports. Probe cleanup preceded original guard
finish, with the complete original finish in `finally`. Optional evidence failure
could not replace the original test result. This directory reproduces/audits the
numeric readout; it does not install instrumentation or launch another arm.
