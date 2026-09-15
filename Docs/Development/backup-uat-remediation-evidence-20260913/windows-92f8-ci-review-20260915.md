# Exact92f8 Windows CI audit

Read-only local log/source review; no apps/tests, network operations, repository edits or PR merge.

**Voice Windows job104289676625/run34941051470 succeeds.** The Windows `git config --global core.longpaths true` step executes before checkout (log36–37). Checkout fetches and checks out genuine refs/remotes/pull/2642/merge, exact SHA `cab693520f7be6b24b3292111a1f69618f65da72` (115/124/229). Its logged merge message names feature92f8f9224bc3cadfc22ba9fe997597b1ce89a05a into base4e4558bff21a6c158432df4e08dad73b832eea79. This is the full synthetic merge checkout, not feature HEAD or a fabricated ancestry workaround. The synthetic commit is not locally present, so parent identities here are qualified as CI checkout/message evidence rather than independently inspected local commit objects.

The unchanged pristine WebRTC/vendor-tree verifier executes successfully; cibuildwheel then builds, repairs, installs in isolated venvs and tests all three WindowsAMD64 wheels:

| CPython | Binding selection | Installed native/corpus smoke | Wheel qualification |
|---|---|---|---|
|3.11|4passed,31deselected,2warnings,.11s|Passed|Passed|
|3.12|4passed,31deselected,2warnings,.11s|Passed|Passed|
|3.13|4passed,31deselected,2warnings,.10s|Passed|Passed|

The logged command chains binding pytest with `&& python Tests/Packaging/test_voice_aec_installed_wheel.py`; each whole command and wheel finishes successfully. Thus the silent installed native/corpus script completes, not merely the pytest half. The31deselections are the existing explicit smoke selection, not new skipped failures. The2warnings per interpreter concern unknown pytest asyncio_mode/timeout options. Repaired wheel metadata/notices/shared-library verification explicitly qualifies all3wheels (1821–1823), and artifact10385627514 uploads1,287,646bytes. Wheel SHA256 values are recorded in the summary **as reported CI build hashes**; no wheel artifacts were downloaded/rehashed in this audit. This closes the observed inherited Windows checkout blocker and confirms this job's actual downstream qualification.

**Startup remains failed** in job104289594769/run34941051473:31non-UI pass9.70s; all4original UI cases time out at unchanged60s. Against pinned dev4631b60f8dd9623fc55bf16f4a37e29fcb1240c7, both keyboard comparisons pass with2warnings: cpp58.51session/49.92call, file43.63session/37.85call. Both candidate comparisons time out. The failure-only diagnostic exits1 and is not a passing acceptance result. All23 monitoring records are source-matched with0observer/configerrors; cProfile is disabled. No further performance attribution was attempted.

Independent Git comparison confirms tldw_chatbook,Tests,Packaging,native andpyproject.toml are byte-identical between2031d and92f8. Commit92f8 changes only the Windows checkout step and tracking/evidence documentation. Voice uses the combined PR merge source noted above; startup deliberately checks the exact92f8 feature source. Do not conflate those source selections, or Voice success with whole-PR/startup acceptance.

Compact summary and both input-log SHA256 digests: `/private/tmp/uat-92f8-windows-ci-summary.json`.
