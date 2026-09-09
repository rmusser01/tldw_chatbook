# Final integration evidence

This package records targeted integration checks after rebasing onto `a36fc6133c69f77b8b14596a59918de34261f8ef`, including the original failed expectation and its narrower correction. Test selections overlap and must not be summed into a unique total. No full-suite result is claimed.

| Recorded check | Outcome |
|---|---|
| [Post-rebase targeted selection](tests/post-rebase-targeted.log) | 197 passed, 2 skipped; 1 inherited warning |
| [Broader final targeted selection](tests/final-targeted-regressions.log) | 658 passed, 1 stale fresh-AllTalk expectation failed, 2 skipped; 3 warnings |
| [Targeted corrections](tests/final-corrections-targeted.log) | 107 passed, 1 skipped, 140 deselected; 1 inherited warning |
| [Fresh-default UI correction](tests/final-alltalk-ui-correction.log) | 6 passed |
| [Independent provider review](reviews/provider-code-review.md) | One stale test expectation identified; no additional product finding in its stated scope; 36 selected fake tests passed |
| [Ruff comparison](ruff/README.md) | 0 new findings; 160 origin/dev diagnostics versus 152 inherited current diagnostics in 12 existing changed files |

The skipped tests require real MPS or local ONNX artifacts in those fake/test selections. Their skips remain skips; separate live evidence does not turn those test invocations into passes. The broader 658-pass run retains its actual failure, followed by a 6-case correction run. [Scope/count receipts](tests/summary.json) retain the exact terminal summaries. [Exact command receipts](tests/test-command-receipts.json) preserve the selectors separately from quiet output. The 107-pass run uses `-k 'not test_speech'`, which deselects the entire Speech UI module; the subsequent exact-node run qualifies all six fresh-provider defaults.

The Ruff audit includes the two Python 3.12 missing-runtime messages and the corrected AllTalk fresh-default expectation. All current findings map to baseline debt, including four manually checked changed-line cases. Complete raw diagnostics, line classifications, settings and hashes are packaged; the duplicate baseline/current source snapshots remain under the original task-local audit path recorded by the receipt.

The [SQLite capability probe](sqlite/post-rebase-sqlite-capability.json) reports Python 3.12.11 / SQLite 3.49.1 with NO_CKPT_ON_CLOSE support. The [static compatibility audit](sqlite/post-rebase-sqlite-harness-audit.md) found no required harness change and explains the new pure-Python helper packaging closure. The current mounted harness does not instantiate the durable profile owner, so its cleanup does not establish the incoming profile-store terminal policy.

The [final wheel receipt](wheel/final-wheel-verify.json) verifies 2,274 Python files byte-for-byte across source, wheel and installation for source commit `1a92c851839f239b0b0794c7024cb96f8ba95ad6`, wheel SHA256 `df325ea04b2af82d392de3e2f38ce3aa6820c1137beada42726fe2036ad7a2d9`. Its complete map is represented losslessly through the [shared source manifest](../kokoro-english/provenance/source-hashes.json), with explicit key prefixes, overrides and removals.

The final installed-wheel matrix completed all three tuples. Each has runtime/content exit 0, 3/3 complete successful clips exact under local Whisper-small, joined cleanup, and unchanged installed package source. CPU/MPS use WAV and ONNX CPU uses actual MP3. The [matrix](wheel/installed-wheel-matrix.json), [per-tuple summary](wheel/live-summary.json) and [owned-process release](wheel/ownership-release.json) retain the complete outcome. The [controller recipe](wheel/recipes/run_installed_wheel.py.txt) uses the installed interpreter with -I and an explicit installed-package gate. All 14 recorded worker/parent/player/ASR PIDs are absent, and the [user config hash remains unchanged](wheel/user-config-after.json). The [root CLI closeout receipt](closeout/task-closeout-receipt.json) records updates to eight completed tasks. These mounted Lab and trusted Console paths exercise real playback, active-generation Stop and successor. They do not qualify full-app profile-store shutdown or acoustic loopback.

[Package provenance](package-provenance.json) retains original paths and hashes. [Verification](verification.json) checks byte-preserving copies and exact JSON round trips. No wheel binary, models, audio, private profile data, credentials or duplicate large source maps are copied here.

[Independent cross-receipt verification](cross-receipt-verification.json) reconstructs all four factored JSON records from their serialized references and confirms that every live tuple's 2,265 application Python hashes match the final wheel, both before and after the run. The wheel receipt additionally covers nine profile-core Python files.
