# ADR-208: DB migration .sql files are the executed source of truth

Status: Accepted
Date: 2026-10-01
Related Task: [task-19565 - Schema artifacts nobody verifies](../tasks/task-19565%20-%20Schema%20artifacts%20nobody%20verifies%20-%2052%20of%2075%20triggers%20unpinned%20no%20pinned%20bodies%20and%2012%20decorative%20migration%20files.md)

## Context

Historically the migration step executed embedded Python constants; 12 of 26
`DB/migrations/*.sql` files were decorative twins kept aligned only by a
comment, while the packaging test pinned them as shipped wheel content. A
maintainer editing the `.sql` file changed nothing, and nothing compared any
file to its constant. Discovered by the 2026-08-21 holistic review (Lane 3,
F2/F10) and recorded as TASK-19565.

## Decision

The on-disk `.sql` file is the single source a migration executes.

- Migrations that can be expressed as SQL execute their `DB/migrations/*.sql`
  file at runtime; embedded constants no longer shadow them.
- Purely decorative twins (migrations whose steps are Python logic a `.sql`
  file cannot express) are deleted rather than kept as misdirection; the set
  of file-backed migrations is now explicit.
- `Tests/DB/test_migration_file_sources.py` enforces the contract: every
  file-backed migration actually opens its file, and no migration silently
  regresses to an unopened twin.
- The packaging release checker parametrizes over the file-backed migration
  set, so a wheel that drops one fails by name.

## Consequences

- Editing a `.sql` file now changes runtime migration behavior — the file is
  authoritative, reviewable, and diffable.
- Packaging evidence grows with the file-backed set (41 → 55 migrations
  parametrized at the time of landing); that is intended visibility, not cost.
- Migrations requiring Python orchestration keep their code path and simply
  have no `.sql` twin.

## Alternatives considered

- Keep embedded constants and add a file↔constant comparator test: rejected —
  it preserves two sources and a class of drift the task was filed to close.
- Move all migrations to Python-only and delete every `.sql`: rejected —
  loses reviewable SQL artifacts and breaks the packaging migration-shipping
  contract that already exists.
