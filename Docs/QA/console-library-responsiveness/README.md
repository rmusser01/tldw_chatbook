# Console Stop and Library opening

## Console Stop

TASK-33621.79 follows one exact accepted send through preparation and
durable postcommit work. The mounted checks cover the visible Stop button
and Ctrl+G, preservation of a newer draft, cancellation of the requested
session, and another send reaching the provider. The authenticated mobile
checks cover both before-reply and native-postcommit windows with a different
chat selected. The cancel response reports success, the requested response
persists as Stopped, and the sibling chat stays idle.

Two private macOS PTY runs stopped preparation before adapter entry and
later stopped a streaming reply. Preparation Stop acknowledgment painted
in 157 ms and 452 ms. Both processes quit normally and left zero private
SQLite descriptors open. The runs still recorded 50 and 100 ms misses.

## Library PTY results

The paired runs used the same 120×50 private macOS PTY and sealed profile
with 10,000 conversations, 10 notes and 11 media items. Each arm retained
the complete 13-second cold and warm observation windows. The populated
measurement ends at the terminal write containing the exact source counts.
The comparisons were collected before rebasing onto dev. Each row uses its
own recorded baseline.

| Change | Cold before → after | Individual cold savings | Warm before → after |
| --- | ---: | --- | ---: |
| Read visible workspace memberships together, TASK-40969 | 2,496 → 1,823 ms median | 525, 788, 1,066 ms | 155 → 151 ms median |
| Skip repeat evidence for saved graduation, TASK-40970 | 1,359 → 1,214 ms final pair | 145 ms in the final pair, 237 and 287 ms in the two corrected prototype pairs | 59 → 61 ms final pair |
| Keep the populated graduated rail mounted, TASK-40971 | 1,520 → 1,330 ms median | 191 ms faster, 294 ms slower, 2,041 ms faster | 61 → 56 ms median |

The membership change reduced 41 separate workspace lookups before paint
to one in the fixture. Cold gaps above 50 ms moved from 23 to 26 across
those pairs. The onboarding change removed seven repeat evidence calls
and retained the attention card after return. An earlier prototype skipped
that card refresh and was discarded. All paired rail candidates skipped
the rail rebuild. The longest rail baseline also had slower unrelated
phases as host load changed. The worst rail candidate cold gap was 604 ms.
All Library processes used ordinary quit and left zero owned private
SQLite descriptors. Full raw measurements remain local.

## Integration with dev

Dev already preserves the full graduated rail on stable updates. The new
regression reaches the correct count and keeps Search focused on dev, then
fails waiting on a queued focus callback. The guard in this PR retains the
broad in-place path and lets the pending native rebuild publish its
replacement controls before that callback runs. The rail timing above is
historical evidence for retaining the controls.

The PR gate retains the existing ordered census and appends the exact
desktop accepted Stop, saved-graduation, and rail callback cases. The
Console size ceiling is unchanged. The Explore check waits for the Back
control as well as settled evidence, using its existing wait bound.

## Cold Notes Sync and ordinary Quit

TASK-41605 and TASK-41606 use the same sealed 120×50 macOS PTY, 10,000
conversations, 10 notes and 11 media items. Each process kept the full
13-second observation window after both the cold and warm Library entries.
The baseline uses the exact product source of `592a16211844`. The
baseline/candidate order alternated across the three pairs. The host
reported 18 CPUs and load varied throughout.

| Run order | One-minute load, start → end | Cold populated paint | Warm populated paint | Notes Sync constructor import CPU | Gaps above 50 ms, both windows | Worst gap | SQLite descriptors at app return |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 baseline | 109.9 → 92.0 | 1,555 ms | 264 ms | 66.55 ms | 7 | 551 ms | 7 |
| 1 candidate | 88.8 → 75.9 | 1,648 ms | 148 ms | 9.25 ms | 8 | 745 ms | 0 |
| 2 candidate | 74.1 → 64.2 | 1,186 ms | 162 ms | 8.11 ms | 7 | 418 ms | 0 |
| 2 baseline | 62.5 → 58.5 | 1,401 ms | 129 ms | 67.85 ms | 13 | 541 ms | 7 |
| 3 baseline | 58.5 → 56.5 | 1,142 ms | 152 ms | 67.97 ms | 6 | 560 ms | 7 |
| 3 candidate | 55.5 → 59.6 | 1,227 ms | 88 ms | 8.76 ms | 8 | 427 ms | 0 |

The cold median moved from 1,401 ms to 1,227 ms. One of the three cold
pairs improved. The warm median moved from 152 ms to 148 ms. The Notes
Sync import used about 59 ms less UI-thread CPU at the median and dropped
in every pair. The full windows had 26 then 23 event-loop gaps above
50 ms, and the worst gap rose from 560 ms to 745 ms. These are Library
navigation measurements. The 100 ms Send busy-paint threshold was
outside this journey.

The shutdown trace identified a MainThread Evals connection from startup
schema inspection and a MainThread Subscriptions connection from service
wiring. At ordinary app return the baseline retained both slots, both
storage leases, and all seven descriptors. In each candidate,
`_close_main_databases` retired the same native handles on their owner
thread and both leases disappeared. Every process exited normally with
zero profile or network guard violations. Median Ctrl+Q to app return was
667 ms on the baseline and 665 ms on the candidate. The candidate medians
were 206 ms before the shutdown handler began, 21 ms in main-store
retirement, and 434 ms in the rest of the shutdown handler.

A fourth candidate traced the existing Subscriptions interpreter-exit
hook. It observed zero private SQLite descriptors at app return and saw
the exit hook close its own subsequent native connection. It also exited
normally. These runs used Library with no active evaluation writer.
The active-transaction unmount tests verify each original connection
remains live.

The first seed invocation failed to import the harness package before it
created a fixture. The raw log and all subsequent sealed runs are retained
locally. The SHA-256 of the paired evidence manifest is
`0e24269119f624ed1dc2bcbe389836be6c980be5f6c1658ed1a441b2135bb9d9`.
The additional exit-hook trace has SHA-256
`d4707ec8b4b9f1cb6702dfe00f45ecf675bc7a54c8731edeb90881c7f55a5709`.

The fresh-process constructor test caught both eager imports on the
baseline. The native unmount test caught the retained startup handles
and covers an active write transaction in each store. The chooser test
now checks the existing “Preparing folder sync…” copy and still tests
the subsequent transition to readiness.

The wider local checks reproduced the same existing first-Send warm-up
failure on both sources. The warm-up counted 95 modules against 94 after
`RAG_Search.scope_evidence` was added. The legacy Notes production-app
suite also reproduced the same pre-mount profile refusals on both
sources. The existing bootstrap-profile mode reached more of the suite
and reproduced the same six startup failures on both sources. The
original failure logs and warnings are retained.

The matched 477-case Notes batches both passed and produced the same four
warnings. Descriptor growth was 210 against the unchanged limit of 200.
Each process ended with 224 descriptors and 38 SQLite leases from the
`sync.sqlite3` and `notes-sync.sqlite3` test stores. The two structured
receipts have identical SHA-256
`8ee453970f3c1c2076a867673281ad93ca6c6f6cf9fd6e694d56b6d5ba6a5626`.
The database and shutdown suite passed 79 cases with one platform skip.
The derived-artifact checks and Ruff gates passed. Final source formatting
was verified against the measured source hashes and has identical syntax
trees.

## Startup saved-agent discovery

TASK-41607 used the same sealed macOS PTY and the exact `b2b6163032`
product baseline. Two diagnostic traces and two follow-up stock traces
identified Console browser construction and launch recovery before the
first usable composer. A disposable projection-sharing candidate removed
only a 0.40–0.82 ms rebuild. Its three launch pairs had medians of 6.27 s
on the baseline and 8.16 s on the candidate. The experiment lives in the
private evidence directory.

The bounded change queues launch recovery through the app's existing
deferred startup task. It issues a single read-only lookup through the
finite worker. The baseline ran that lookup on MainThread. All three
candidates ran it on a worker and finished the lookup before the
composer's first usable frame.

| Run order | One-minute load, start → end | Launch to usable composer | First key to echo | Time in launch scheduling on UI thread | Worst loop gap through first-input window |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 baseline | 57.5 → 47.0 | 14.34 s | 38 ms | 114.69 ms | 2,103 ms |
| 1 candidate | 46.5 → 37.7 | 10.83 s | 37 ms | 8.81 ms | 2,275 ms |
| 2 candidate | 36.5 → 40.0 | 10.74 s | 38 ms | 3.83 ms | 1,664 ms |
| 2 baseline | 41.9 → 56.8 | 9.86 s | 32 ms | 76.63 ms | 1,694 ms |
| 3 baseline | 57.2 → 59.5 | 10.64 s | 28 ms | 74.35 ms | 1,672 ms |
| 3 candidate | 59.6 → 59.3 | 9.57 s | 78 ms | 7.32 ms | 1,629 ms |

The known UI-thread stall fell from 76.63 ms to 7.32 ms at the median.
Launch-to-usable moved from 10.64 s to 10.74 s at the median, and two
of the three pairs improved. Across both startup and 13 seconds after
typing, there were 56 then 61 gaps above 50 ms, and the worst gap rose
from 2,103 ms to 2,275 ms. First-keystroke and whole-startup results
varied across the pairs.

Every run retained the full first-input window. All three candidates
and two baselines completed both 13-second Library windows. The first
baseline hit the existing five-second Library source deadline while
initializing a second Notes database instance. Its cold failure and
startup measurements are retained. Its warm visit was skipped by the
driver following that failure. All six child processes quit ordinarily
with zero scoped SQLite descriptors and zero profile or network guard
violations.

The baseline regression opened native SQLite synchronously in the
production scheduling entry point. The final tests hold the real
connection opener and check worker affinity, current-source and
automatic-wake revalidation, physical cancellation retirement, and
the unchanged durable recovery checks. The 37 targeted tests passed.
Eight boot and module-size checks passed with two census warnings
(1,040 of 1,040 UI-ready modules and 687 of 688 boot imports).
Derived-artifact preflight, Backlog guards, and Ruff gates passed.
The SHA-256 of the private evidence manifest is
`1a2904a16eebc4715f1e5a99eebeb57a4a529019d2f920354ef63a9b0d8911a2`.

## Local Notes page and total

TASK-41637 uses the exact `75994498f7` baseline with the same sealed
120×50 macOS PTY profile. In two stock traces, the second same-file
Notes owner took 243 ms and 324 ms to initialize. The count then used
its own worker connection. A disposable joined-read prototype removed
the extra owner in all three runs. Its cold median moved from 974 ms
to 887 ms, measured from app key handling. All six prototype results
remain in the private receipts.

The final comparison measures from writing Ctrl+3 to the terminal flush
that paints 10 Notes, 11 Media and 10,000 Conversations. Notes timings
span owner selection, page reading and counting on the same worker or
across the original two workers.

| Run order | One-minute load, start → end | Cold populated paint | Warm populated paint | Cold Notes work | Warm Notes work |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 baseline | 35.5 → 34.1 | 988 ms | 127 ms | 601 ms | 285 ms |
| 1 candidate | 34.5 → 31.6 | 1,087 ms | 87 ms | 406 ms | 233 ms |
| 2 candidate | 30.6 → 28.1 | 993 ms | 77 ms | 393 ms | 236 ms |
| 2 baseline | 28.1 → 32.7 | 995 ms | 211 ms | 367 ms | 326 ms |
| 3 baseline | 33.0 → 25.5 | 941 ms | 34 ms | 597 ms | 215 ms |
| 3 candidate | 25.0 → 17.9 | 893 ms | 78 ms | 428 ms | 212 ms |

Each candidate used the existing profile owner and one Notes SQLite
preparation per cold and warm visit. Each baseline used two preparations
per visit and constructed a second owner on the cold visit. Median Notes
work fell from 597 ms to 406 ms cold and from 285 ms to 233 ms warm.
Median warm paint improved from 127 ms to 78 ms, with two faster pairs.
Cold paint moved from 988 ms to 993 ms. Its two faster pairs saved 2 ms
and 48 ms, and the first pair was 99 ms slower. Warm navigation can
paint its retained snapshot while the fresh read runs. Median launch to
usable input was 8.14 s on the baseline and 8.69 s on the candidate.

The six processes kept all three 13-second windows, quit ordinarily and
left zero scoped SQLite descriptors or profile and network guard
violations. Gaps above 50 ms moved from six to nine in the cold windows
and four to five in the warm windows. Worst gaps moved from 417 ms to
575 ms cold and 156 ms to 81 ms warm. Send busy paint was outside this
journey.

The real source regression found two worker opens on the baseline and
one on the candidate. The four final checks passed with exact paging,
unknown total on count failure, denied access without a database open,
and the original route for a different profile owner. The broader run
recorded 166 passes and 24 failures. The frozen baseline reproduced all
14 Notes profile-admission failures, and all 16 Notes cases passed in the
existing bootstrap profile. The ten native worker tests blocked by the
macOS sandbox passed with process inspection for their private children.
Final derived-artifact, Backlog and Ruff checks passed. The SHA-256 of
the 180-entry private receipt manifest is
`3014644680e34feff930338a608b513afe6509cbe43ff8ed7d5d61bb5133e1a7`.

## Study and Quiz snapshot experiment

On the `3d909d3b4a` baseline, two identity-preserving cold traces put
source loading at 548–582 ms. Conversations finished last in both.
The Study decks, due flashcards and quiz counts used three separate
SQLite preparations for the same profile database during each visit.

The candidate ran the three original stock scope reads on one finite
worker. It kept separate policy checks, kept the in-memory and custom
routes, and retried the existing independent reads if the shared worker
failed. The baseline regression reproduced the extra opens. An initial
candidate collided with the screen's existing Study count field and
failed 12 tests. After the rename, the same 12 cases passed, including
the real file-backed counts, denied due cards and failed-worker fallback.

The same sealed 120×50 macOS PTY measured key write to populated terminal
paint, in alternating order. Source time is the wall time of the cold
source snapshot call.

| Run order | One-minute load, start → end | Cold populated paint | Warm populated paint | Cold source time |
| --- | --- | ---: | ---: | ---: |
| 1 baseline | 22.8 → 19.8 | 1,419 ms | 105 ms | 866 ms |
| 1 candidate | 19.7 → 36.4 | 1,772 ms | 205 ms | 1,153 ms |
| 2 candidate | 37.8 → 39.9 | 1,845 ms | 350 ms | 664 ms |
| 2 baseline | 39.9 → 47.3 | 1,626 ms | 390 ms | 782 ms |
| 3 baseline | 47.7 → 47.8 | 2,399 ms | 347 ms | 1,129 ms |
| 3 candidate | 47.7 → 55.1 | 1,785 ms | 421 ms | 551 ms |

Every measured Library visit used three Study SQLite preparations on the
baseline and one on the candidate. Cold source time improved from 866 ms
to 664 ms at the median. End-to-end cold paint rose from 1,626 ms to
1,785 ms, with two slower pairs. Warm paint went from 347 ms to 350 ms.
Cold windows had 13 then 11 gaps over 50 ms and worst gaps of 820 then
832 ms. Warm windows had 11 then eight gaps and worst gaps of 402 then
266 ms. All six processes kept all three 13-second windows, exited
ordinarily, and left zero scoped SQLite descriptors.

Two candidate runs spent 339 ms and 222 ms between source completion and
the populated frame. The Notes Sync preparation worker overlapped those
intervals for 192 ms and 136 ms, respectively. Library mount requests
preparation when the app has yet to construct the runtime, including
profiles where the boot evidence gate deferred construction.

The candidate was removed after the paired run. The patch, initial
failures, final controls and full recordings remain in the private
experiment directory. The 145-entry receipt manifest has SHA-256
`0bd591d76f817e7becf9d1f3444737909e9c46bef1301bf8f1498417ac75e32c`.

## Notes Sync preparation timing control

Both arms used the exact `505dbf69e1` product source, the same sealed
120×50 macOS PTY fixture, and alternating order. A diagnostic wrapper
on an unprepared runtime's first Library visit called through immediately
on the baseline. The deferred arm released the same native preparation
after the terminal flushed the frame with the exact populated counts.
The runtime was constructed in all six runs. Native preparation occupied
169–229 ms before that frame on the baseline and finished 134–353 ms
after the frame in the deferred arm, inside the same observation window.

| Run order | One-minute load, start → end | Cold populated paint | Warm populated paint | Cold source snapshot | Worst cold loop gap |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 baseline | 72.3 → 89.9 | 3,097 ms | 537 ms | 1,353 ms | 1,028 ms |
| 1 deferred | 91.2 → 99.8 | 2,811 ms | 343 ms | 1,082 ms | 1,279 ms |
| 2 deferred | 96.3 → 80.3 | 3,030 ms | 254 ms | 1,329 ms | 1,143 ms |
| 2 baseline | 80.0 → 86.7 | 3,312 ms | 484 ms | 1,921 ms | 1,051 ms |
| 3 baseline | 88.9 → 103.5 | 2,201 ms | 761 ms | 1,274 ms | 963 ms |
| 3 deferred | 106.6 → 123.8 | 2,075 ms | 396 ms | 1,150 ms | 872 ms |

Median cold paint moved from 3,097 ms to 2,811 ms. The three cold
pairs were 286 ms, 282 ms, and 126 ms faster. In the third pair the
deferred arm reached the preparation request 150 ms earlier and took
24 ms longer from that request to paint. Cold source reads also varied
substantially and the host reported 18 CPUs. Median warm paint moved
from 537 ms to 343 ms, with native preparation already complete before
each warm visit. Median launch to usable input stayed near 20.6 seconds
on both arms. The variation in phases outside the hold limits how much
of the cold gain can be attributed to Notes Sync timing.

Across the full cold windows, gaps over 50 ms moved from 25 to 24 and
the worst gap grew from 1,051 ms to 1,279 ms. Across the warm windows,
gaps moved from 14 to 15 and the worst gap fell from 340 ms to 274 ms.
All six runs kept the 13-second first-input, cold and warm windows.
They quit normally and left zero scoped SQLite descriptors and zero
profile or network guard violations. These journeys cover Library
navigation and initial typing.

The timing hold stayed in the diagnostic harness. In the cold traces,
0.84–1.65 s elapsed before source loading began and 1.08–1.92 s during
the source snapshot. The Conversations scope read returned within
38 ms of snapshot completion in all six runs. The 104-entry private
receipt manifest has SHA-256
`4e4ddffecab1546dd355be2ef48b0175b6f7404d22f935ecea71d4c2ae2d741c`.

## Cold Library conversation message counts

TASK-41668 traced the Library's 20-row conversation page on the
`7c562db08e` source. The first two cold diagnostic visits painted in
1,337 ms and 1,057 ms. On the first trace, the page read spent 317 ms
acquiring its native connection and the message count used 80 ms of
worker CPU. The two traces each recorded seven denied socket attempts
in the fixture.

The query now filters the conversation side of its existing join. With
the same 20 IDs and real sealed SQLite corpus, the median count CPU
fell from 59.10 ms to 2.14 ms. The read-only comparisons matched each
count across 1-, 20-, and 75-ID pages under all four deletion-option
combinations. The native regression checks counts for deleted and
missing IDs, a duplicate ID, and soft-deleted messages. Its SQLite
progress counter saw 620 groups of 100 instructions on the baseline.
The candidate passed the test's 200-group bound.

Both sets of alternating 120×50 macOS PTY pairs used the sealed
10,000-conversation fixture, the frozen baseline, and a candidate
with the conversation filter change. The CPU column measures the exact
message count for the cold conversation page. First echo is measured
from process launch to the terminal flush containing the typed input.

### Empty tokenizer cache override

The existing fixture's explicit empty cache selected the upstream
tokenizer download path. The network guard rejected those attempts in
both arms and the caller stack identified `tiktoken.load`.

| Run order | One-minute load, start → end | Cold paint | Warm paint | Count CPU | Launch to first echo | Denied network attempts |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 baseline | 17.3 → 12.8 | 758 ms | 114 ms | 42.89 ms | 6.16 s | 5 |
| 1 candidate | 12.7 → 14.2 | 812 ms | 71 ms | 2.12 ms | 6.87 s | 5 |
| 2 candidate | 17.9 → 24.8 | 1,384 ms | 316 ms | 2.64 ms | 10.80 s | 6 |
| 2 baseline | 23.8 → 17.9 | 810 ms | 48 ms | 50.86 ms | 7.75 s | 5 |
| 3 baseline | 17.4 → 29.6 | 923 ms | 63 ms | 49.26 ms | 6.88 s | 5 |
| 3 candidate | 30.3 → 40.9 | 1,106 ms | 298 ms | 2.88 ms | 9.10 s | 6 |

Median count CPU moved from 49.26 ms to 2.64 ms. Median cold paint
rose from 810 ms to 1,106 ms, and every cold pair was slower. Across
the full cold windows the number of gaps over 50 ms rose from eight
to eleven, with worst gaps of 254 ms and 574 ms.

### Shipped offline tokenizer

The follow-up used the harness's existing bundled-tokenizer option,
which removes the fixture override and lets the app select its shipped
assets. All six processes reported zero denied network attempts.

| Run order | One-minute load, start → end | Cold paint | Warm paint | Count CPU | Launch to first echo | Denied network attempts |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 candidate | 42.7 → 29.7 | 736 ms | 50 ms | 2.03 ms | 8.85 s | 0 |
| 1 baseline | 29.7 → 21.2 | 806 ms | 72 ms | 41.05 ms | 6.66 s | 0 |
| 2 baseline | 14.6 → 12.6 | 821 ms | 39 ms | 48.06 ms | 6.67 s | 0 |
| 2 candidate | 12.6 → 11.9 | 781 ms | 50 ms | 2.13 ms | 5.52 s | 0 |
| 3 candidate | 12.0 → 11.6 | 709 ms | 74 ms | 1.77 ms | 5.64 s | 0 |
| 3 baseline | 12.7 → 11.7 | 774 ms | 41 ms | 39.95 ms | 6.32 s | 0 |

Median cold paint fell from 806 ms to 736 ms. The individual savings
were 70 ms, 40 ms, and 65 ms. Median cold source time fell from 535 ms
to 464 ms and count CPU from 41.05 ms to 2.03 ms. Median warm paint rose
from 41 ms to 50 ms. Cold gaps over 50 ms fell from nine to six, with a
worst gap of approximately 228 ms on both arms. The warm windows each
recorded one gap over 50 ms and their worst gaps were 52 ms and 51 ms.
Both sets of pairs and both diagnostic runs kept the full 13-second
first-input, cold and warm windows. All fourteen quit normally with zero
scoped SQLite descriptors and no profile boundary violations.

The native database and service selection reported 119 passes. Its two
legacy migration fixture errors reproduced on the frozen database source.
Two of the four Library checks passed in the ordinary profile. The other
two refused the config source before mount in both versions. In the
existing bootstrap profile, the in-place reconciliation check passed on
both versions and the focus assertion reproduced on both. The edited
ranges passed formatting and Ruff added zero findings. The 254-entry
private receipt manifest has SHA-256
`745e491310327ae5f270f471e351d2e3876d346e296761e27fdd64001153c80a`.

## App startup through first input

With the shipped tokenizer, launch to a physically echoed first key took
5.52–8.85 seconds across the six runs above. The message count began
after the first usable composer in each run.

| Startup stage | Observed wall time |
| --- | ---: |
| App import | 1.06–1.64 s |
| Additional initial Console import | 0.35–0.53 s |
| App construction | 0.85–1.20 s |
| Initial Console push through first usable composer | 2.36–4.28 s |

Notes initialization was the longest parallel constructor worker at
0.42–0.57 s. The Console push through first paint was the largest
measured stage. Sampled stacks included stylesheet matching, native
layout and SQLite preparation. The traces recorded 45–48 guarded
SQLite helper starts before the first usable composer, including
Chat/Notes, scheduled tasks and Workspace storage. The startup samples
identify the work present in those intervals and leave its necessity
and individual contribution open.

## Guarded SQLite helpers before the first usable composer

Two caller traces on `ae511df60c` used the sealed 120×50 macOS PTY fixture
and the shipped offline tokenizer. They measured 6.03 s and 8.37 s from
launch to the first echoed key, with one-minute load moving from
5.3 to 18.4 and from 18.4 to 29.7. Every observed helper used the transient
SQLite preparation operation for an admitted native connection.

| Store before the composer | Main-thread prepares per trace | Worker prepares per trace |
| --- | ---: | ---: |
| Chat/Notes | 1 | 14 / 13 |
| Scheduled tasks | 4 | 4 |
| Workspaces | 1 | 6 |
| Agent runs | 1 | 5 |
| Library collections | 1 | 2 |
| Library ingest jobs | 1 | 1 |
| Subscriptions | 1 | 1 |
| Evals | 1 | 0 |
| Prompts | 0 | 1 |
| Media | 0 | 1 |
| Total | 11 | 35 / 34 |

Main-thread preparation occupied 385–567 ms, including 253–379 ms in
the helper request/reply exchange. Parent-thread CPU inside preparation
was 63–80 ms. Worker prepares overlapped with one another and with the
main thread.

Scheduling took 142–208 ms across its four main-thread prepares. Those
calls checked the schema, reconciled stale automation runs, and read
the automation-definition and reminder transfer recovery candidates.
The worker side separately loaded the reminder and automation queues,
recovered interrupted task runs, and pruned task history. Most Chat/Notes
prepares came from finite workers doing Browser, attention, character
scope or recovery reads. The three Quick Note reads were for delete
receipts, create receipts, and blocked conflicts in that order.

### Empty Quick Note recovery control

The disposable candidate used one guarded owner-filtered read that
includes blocked receipts. A zero total and empty page established
the fast path on this fixture. A nonempty result would proceed to the
original ordered recovery. The candidate used the same finite worker,
native admission and connection retirement as the original reads.

The six alternating PTY runs confirmed that both arms found zero
receipts. All three candidates saved two real helper starts. The
receipt-to-composer column starts at the first Quick Note receipt
read on each arm. The startup helper count ends at the first usable
composer.

| Run order | One-minute load, start → end | Launch to first echo | Receipt read to composer | Main / total startup helpers | Quick Note helpers and wait | Cold Library | Warm Library |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 baseline | 50.8 → 43.0 | 11.53 s | 5.23 s | 11 / 48 | 3 (226 ms) | 2044 ms | 383 ms |
| 1 candidate | 44.4 → 53.1 | 11.07 s | 5.29 s | 11 / 44 | 1 (107 ms) | 1455 ms | 63 ms |
| 2 candidate | 53.9 → 53.3 | 18.72 s | 9.44 s | 11 / 44 | 1 (184 ms) | 1511 ms | 560 ms |
| 2 baseline | 50.8 → 46.3 | 21.41 s | 9.19 s | 11 / 50 | 3 (562 ms) | 2058 ms | 340 ms |
| 3 baseline | 46.2 → 45.9 | 13.54 s | 6.24 s | 11 / 48 | 3 (275 ms) | 1355 ms | 455 ms |
| 3 candidate | 45.1 → 36.8 | 12.50 s | 5.93 s | 11 / 46 | 1 (72 ms) | 1421 ms | 197 ms |

The candidate reached its first receipt read 345 ms, 2,928 ms, and
597 ms earlier. From that read through the usable composer it took
53 ms and 253 ms longer in the first two pairs and 312 ms less in the
third. This established a reduction in worker activity on the empty
profile and left the launch-to-input benefit unresolved. The candidate
stayed in the disposable harness.

Across the full first-input windows, gaps over 50 ms moved from 19
to 16 and the worst gap moved from 330 to 307 ms. Cold Library gaps
moved from 13 to 14, with worst gaps of 841 and 677 ms. Warm Library
gaps moved from seven to five, with worst gaps of 399 and 477 ms. All
eight traced processes kept all three 13-second windows, quit normally
and left zero scoped SQLite descriptors. The profile and network guards
recorded zero violations. The 148-entry private receipt manifest has
SHA-256 `8d8d95de67aeefc78fc9cb125f96d8cfe8629db4b7f1f5350b80eb649e5b9019`.
