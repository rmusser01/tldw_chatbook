# ADR-225: App-scoped SQLite retirement

Status: Accepted
Date: 2026-10-07
Related Task: TASK-34661

## Context

Actual private-profile app shutdown leaves the constructor Chat/config,
Evaluations, Library Collections, Subscriptions and Workspace native handles
queryable. Test-owned cleanup subsequently closes them, but is not production
retirement evidence. Four stores are app-composed; Chat selects a reusable
shared config/Notes owner. Blanket closure would revoke an existing borrower's
handle and contradict existing native ownership and failed-close contracts.

## Decision

After existing producer owners settle, explicitly retire eligible current-thread
handles of the four app-composed stores through their existing close boundaries.
Drain Evaluations through its existing async close. Track Chat's pre-existing
current-thread handle without opening one; preserve an unchanged live borrower,
and retire only a new app-opened handle or eligible app-opened replacement.
Keep the shared owner reusable.

Preserve active borrowers, foreign-thread ownership, uncertain settlement and
failed-close cache/lease evidence. A cancelled Worker wrapper, an unlinked file,
GC or a cleared Python attribute does not establish native retirement. Keep
primary shutdown failures and expose cleanup failures without sensitive bodies
or paths. Do not introduce a generic lifecycle manager or global quiescence.

## Alternatives

- Process-wide/global close: rejected because this app does not own every
  borrowed handle or worker in the process.
- Keep all five caches until process death: rejected for app-owned handles;
  it leaves repeated app sessions retaining resources without an owner boundary.
- GC or increased FD limits: rejected; neither establishes native retirement.

## Consequences

Retirement and eligibility remain explicit at the one composition/shutdown root.
Uncertain or busy handles remain visible and must not be counted as closed.
Tests must measure real native state before their own cleanup and distinguish
app shutdown from physical executor join. This decision does not qualify the
whole application, native/Windows viewport matrix or aggregate FD budget.

## Links

- [Written implementation contract](../../Docs/superpowers/specs/2026-10-07-task34661-sqlite-app-retirement-design.md)
- [Current RED and native inventory](../../Docs/QA/task-34661/README.md)
- [ADR036](036-application-service-composition-lifecycle.md)
- [ADR094](094-console-turn-lifetime-and-navigation-boundary.md)
- [ADR126](126-complete-local-backup-and-recovery.md)

## Approved scope extension — 2026-10-07

The user approved applying this same contract to app-created AgentRuns receipt
and worker handles after the five-store controls exposed remaining descriptor
growth. Use the existing Console lifetime and operation-owned source boundaries;
preserve shared borrowers, owning-thread closure and active/failed-close evidence.
The earlier five-store implementation is not a whole-app retirement certificate.
No process-wide registry, sweep or quiescence policy is added by this extension.

Retain the disposed Console runtime on its app: dropping this existing lifetime
fence admits a second owner through late access. Keep AgentRuns' existing Console
close ordering, but verify eligible settled-source retirement and expose failure.
Late single/batched badge reads use the existing operation-owned guard rather
than retaining a replacement handle beyond that finite read. Warm borrowed
native handles and their live transactions remain unchanged.

Reply workers own newly opened handles through their existing attempt wrapper;
the shared roster read gives both preview callers the same finite ownership.
At exit, include the controller's existing physical-worker task census in the
unchanged bounded drain. Budget expiry does not cancel that physical owner's
task or certify retirement: active native ownership still reports incomplete.
The stricter incomplete result applies only to exact file-backed AgentRuns;
in-memory/custom owners retain their original close-error compatibility.
