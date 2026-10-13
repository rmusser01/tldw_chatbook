# ADR-225: beautifulsoup4 is a declared core dependency

- Status: Accepted
- Date: 2026-10-11
- Deciders: owner delegate (2026-10-11 perf-review burn-down residue, TASK-21231)
- Governance: TASK-21231; supersedes the packaging posture TASK-21104 found

## Context

TASK-21104 (2026-08) made `Subscriptions/monitoring_engine.py`'s
`from bs4 import BeautifulSoup` lazy because `beautifulsoup4` was declared only
in extras, so a base install could not import the app. The guard was correct,
but the declaration it guarded was dishonest in the other direction:
`markdownify` — a `[project]` core dependency since before the review —
transitively pins `beautifulsoup4<5,>=4.9`. Every healthy base install
therefore already has bs4, by accident rather than by declaration
(TASK-21231, verified on dev `b2b1e2e0d`; re-verified on `a3c47c6c72`:
`pyproject.toml` line 75 core `markdownify`, bs4 only at lines 124/193/352/396
under `websearch`/`ebook`/`subscriptions`/`web`).

Two consequences of leaving it undeclared:

1. The TASK-21104 guard's "bs4 is not installed" branch and the extras'
   "install this to get bs4" promise are untestable against a real base
   install — a real base install always has bs4 via markdownify.
2. A future `markdownify` release that drops the pin silently changes what a
   base install of this app can do, with nothing in the repo recording that
   we relied on it.

## Decision

`beautifulsoup4` is a **core dependency**, declared in `[project]
dependencies` next to `markdownify`, and the `websearch`, `ebook`,
`subscriptions`, and `web` extras stop re-declaring it. The TASK-21104 lazy
guard stays: it costs nothing when bs4 is present and still degrades
honestly on a broken/partial install.

## Alternatives considered

- **Keep bs4 optional and accept the markdownify accident.** Rejected: the
  guard's not-installed branch stays unreachable, and the pin we depend on is
  recorded nowhere.
- **Make `markdownify` optional instead** (the only way bs4 is genuinely
  absent from a base install). Rejected: `markdownify` is imported by core
  modules on the boot path; moving it behind guards is a large, risky change
  to win a testability property nobody needs — the guard remains as the
  safety net for damaged environments.

## Consequences

- `Tests/Packaging/test_extras_import_closure.py` keeps passing: extras that
  lose the redundant declaration still resolve bs4 from core.
- The TASK-21104 guard comment in `monitoring_engine.py` is updated to
  describe the new reality (guard-as-safety-net, not guard-as-feature-gate).
- A future `markdownify` that drops its bs4 pin no longer changes the base
  install's capabilities; the direct declaration holds.
