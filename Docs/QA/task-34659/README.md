# Independent cursor lifetime correction

Base: `61e28695bc4fa9a62e8f14771a529591576d96ce`. Branch:
`codex/sqlite-interrupt-ownership`. Production code is unchanged.

The original CharactersRAGDB case failed with `Cannot operate on a closed
cursor` (1 failed, 1.17s). Its transaction manager correctly closes its own
cursor. The test now escapes a separate cursor from the already-admitted native
connection and independently asserts the managed cursor is closed at scope exit.
All original commit, native borrower, maintenance exclusion and explicit
retirement checks remain.

The complete affected module passed **60 tests in 19.95s**, with no pytest
warnings. Changed-range formatting passes. Full-file Ruff retains exactly the
same 26 code/message diagnostics as the base; this is not a whole-file lint-clean
claim. No imports or fixture names were removed to hide inherited diagnostics.
Self-review found no production or admission changes. Both Backlog guards pass.

Commands used the existing interpreter with `PYTHONDONTWRITEBYTECODE=1`,
`pytest -q -p no:cacheprovider` and separate private `--basetemp` directories.
Only targeted tests ran. Earlier Ruff cache-permission failures remain local;
the successful static checks used `--no-cache`.

Raw logs remain local; no aggregate environment dumps are published. SHA-256:

| Receipt | SHA-256 |
| --- | --- |
| `/private/tmp/sqlite-owner-baseline-5kh1ad/cursor-baseline.log` | `3f2ade2999140be987218e2bf6b9c42dddecf6901ee28676edc02bbfe90d4454` |
| `/private/tmp/sqlite-owner-green-df4AtC/cursor-module.log` | `85c5335f3a0d55b6226f39ebfd55a3791a40bea03e4914806ec262b9363867ec` |
| `Tests/Backup_Recovery/test_core_participant_lifetimes.py` | `9e8e5fc9a0ddf16d29216e9c0ea2d3be400aec5938092e2e790000af07e0786f` |

This corrects the closed-cursor baseline test. It does not establish application
SQLite shutdown retirement, aggregate FD qualification or the broader switcher
qualification matrix. No other sessions were contacted or interrupted.

Final follow-up retains the same test source hash and exact 26 inherited lint
diagnostics; changed-range formatting passes. The affected lifetime module also
passed in the later integration selection. Approved application/AgentRuns
production ownership changes are tracked separately in TASK34661, not attributed
to this test correction. Incoming static-gate failures and independent review/
integration remain outstanding; this task stays In Progress. No source commit,
push or merge is claimed.
