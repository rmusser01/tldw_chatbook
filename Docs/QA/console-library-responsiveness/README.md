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
