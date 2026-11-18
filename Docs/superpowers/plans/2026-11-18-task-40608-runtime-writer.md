# Windows Console runtime and profile writer

User decisions on November 18 put the Console runtime outside the UI and
select a single writer for cooperating app instances. Textual submits Send,
draft, selection and approval intents. A poll reads detached display state.
The existing process runtime already runs provider calls, tools and decision
checks. Its current full projection is the compatibility snapshot.

ADR required: yes. ADR path: [ADR-126](../../../backlog/decisions/126-complete-local-backup-and-recovery.md#task-40608-external-console-runtime-and-profile-writer).
The writer coordinates changes from multiple processes to the same stores.

1. Extend the existing polled projection. Bind deltas to the last adopted
   request in the same runtime, session incarnation, attachment and visit.
   Preserve snapshot validation, original receipt ordering, readiness time and
   the five-second action deadline. Exercise a changed section, an empty delta,
   an invalid or missed base, source replacement and uncertain retirement.
2. Add a profile-scoped writer transport with an exclusive Windows owner and
   local, same-user admission. Use bounded data-only messages for explicitly
   named domain operations. Only the elected owner constructs the original
   repositories for a submitted write. An unacknowledged mutation is never
   retried automatically. Verify two real client processes, draining, abrupt
   disconnect and restart against private disposable storage.
3. Route stock Settings changes to agent definitions, Console run/step/trace
   updates, ChaChaNotes mutations and Chatbook imports to this owner. Preserve
   transactional groups, post-commit notifications, import conflicts, receipt
   order, path/ACL and maintenance checks. A client can read a commit and
   another client can write between steps in a paused turn. Retain separate
   profile identities, explicit database paths and legacy schema support.
4. Update Textual's display consumers to apply changed sections and preserve
   local edit, focus and mount state. Keep action checks on the runtime. Prove
   normal Windows startup uses the verified process route. Reconcile external
   changes at the original authority boundaries.
5. Freeze product code and run the original Windows README and approved-edit
   journeys as B1/C1, C2/B2, B3/C3 with fresh private profiles. Preserve the
   provider script, mounted assertions, approval order, filesystem/semantic
   checks and retirement. Report six candidate times, matched baselines and
   medians. Each candidate turn must finish under five seconds of application
   time, with at least a 30% median improvement for each journey. Checkpoint
   verified work and receipts on PR 3575.

An idle turn whose runtime remains alive during writer restart retains its
approval or provider wait. An outstanding mutation with an unknown outcome
is surfaced to its caller. The first database open after restart must
distinguish live run owners before orphan recovery publishes failure receipts.
Two profiles selecting the same database file retain the original access and
identity checks and must coordinate writes to that physical store. A profile
path alone cannot establish exclusive custody of an aliased database. Chatbook
imports retain their per-store success and failure boundaries. Ordinary edits
to independent Media and Prompts stores remain with those stores.
