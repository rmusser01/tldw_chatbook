# ADR-132: Fleet history navigation

Status: Accepted
Date: 2026-09-07
Related task: TASK-15201
Related decisions: ADR-017, ADR-043, ADR-131

The fleet rail keeps a four-row preview and counts the full current fleet in
its summary. Its existing View all tail opens a read-only sub-agent run picker
for the selected conversation, including earlier and superseded runs. Selecting
a saved run returns to the existing Console drill-in; it grants no execution,
steering, or continuation authority. The callback rechecks the conversation and
record identity so switching sessions while the picker is open cannot expose a
foreign run in the new session.

The picker loads 50 rows at a time off the UI thread, plus one lookahead row.
AgentRunsDB provides a metadata-only, conversation-scoped query ordered by
created_at/id descending, using a cursor over that pair. New runs do not shift
the next page as they would with OFFSET. Previous pages use the saved cursor
stack; Refresh returns to the newest page. No steps, results, or transcripts
are fetched until a run is selected. Limits and cursor types are validated.
There is no schema migration or new index requirement; the existing
conversation index narrows each query. An unattached live handle has no saved
run to inspect yet and appears once its run is created.

The generic Inspector section gains an optional display-row limit. Counts and
the underlying rows remain complete; only mounted preview rows are bounded.
The fleet opts in to that limit and to the already-supported View all tail.
Expanding reveals its View all action after layout (or the section itself when
there is no action); the programmatic refresh path does not scroll. TASK-15110
caps each section at 20% of the outer rail, so scrolling only the header leaves
the tail clipped by the inner viewport. The existing cap remains intact.
Navigation remains mouse- and keyboard-accessible.

Alternatives: showing every run in the rail would mount unbounded content in a
shared scroll; loading complete steps for the picker would repeatedly hydrate
large payloads; routing to the message Trajectory screen would omit historical
child runs not represented by primary-message trace events. The picker is a
temporary selection surface, not a second run-detail implementation.
