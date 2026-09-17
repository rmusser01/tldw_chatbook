# Windows startup follow-up at e683efb864

The Windows job in [run 34835129237](https://github.com/rmusser01/tldw_chatbook/actions/runs/34835129237) finished unsuccessfully. All 31 non-UI checks passed in 9.93 seconds; all four original full-app cases exceeded their unchanged 60-second deadline. macOS and Linux jobs passed. These are startup checks, separate from the native backup workflow qualification.

The same diagnostic lifecycle passed on dev 4631b60 in 47.99 seconds (observer 51.375 seconds). Candidate e683 exceeded 60 seconds, with its last profile at 59.531 seconds. All eight profile records matched the expected source and reported no observer error. The [receipt](windows-startup-e683-receipt-20260914.json) retains the job identity and downloaded-log hash; [profiles](windows-startup-e683-profiles-20260914.json) retain the observations.

The candidate completed construction but was still composing its initial Console at the last observation. Native configuration/storage admission remains a substantial part of startup: 215 acquisitions, 1,107 registry reads and 117,935 native handle opens were observed. Their inclusive durations overlap and must not be added. This evidence does not establish a deadlock or a completed startup time, and later Console refresh improvements do not establish a cold-start pass.

No timing budget or native admission guard was changed. Windows startup acceptance remains unresolved. The independently running e683 57-case backup support repeat and d308 replacement run retain their original revisions and outcomes.
