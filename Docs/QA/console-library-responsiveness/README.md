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
