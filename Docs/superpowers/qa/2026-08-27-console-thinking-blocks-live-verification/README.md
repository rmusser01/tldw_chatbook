# Console thinking blocks live verification

This evidence set exercises the production Console controller, persistence service,
request preparation, transcript projection, and Textual widgets with deterministic
adapter-edge fixtures. The harness creates its own profile and databases beneath this
directory; it does not use or modify the operator's normal configuration.

## Result

The final isolated run passed on 2026-08-27. The machine-readable observations are in
[`observations.json`](observations.json), with seven painted Textual frames under
[`captures/`](captures/):

1. displayable thinking expanded while the turn is live;
2. the same block collapsed once the visible answer begins;
3. the presentation setting hiding the block immediately;
4. persisted history restored collapsed and lazily mounted after restart;
5. the exact proprietary-unavailable notice, backed by turn evidence;
6. a capable model with no event producing no thinking row; and
7. the saved history policy overridden by a visible, disabled `Required` state.

The functional portion also verifies Auto, Include, Exclude, and Required replay;
pre-provider refusal for incompatible durable displayable and proprietary history; a
plain model on the same local backend dispatching through the non-thinking adapter;
and exact-owner recovery through Retry and Bypass after a resumed refusal.

The generated profile's config hash was identical before and after the run. A blocked
tiktoken cache download fell back locally and was not part of the deterministic
provider fixture. The environment also emitted the repository's existing Requests
dependency warning.

## Reproduce

From the isolated feature worktree root:

```bash
qa_root="$PWD/Docs/superpowers/qa/2026-08-27-console-thinking-blocks-live-verification"
env TLDW_TEST_MODE=1 \
  HOME="$qa_root/profile/home" \
  XDG_CONFIG_HOME="$qa_root/profile/xdg-config" \
  XDG_DATA_HOME="$qa_root/profile/data" \
  XDG_CACHE_HOME="$qa_root/profile/cache" \
  TLDW_CONFIG_PATH="$qa_root/profile/config.toml" \
  TMPDIR="$qa_root/profile/tmp" \
  QA_EVIDENCE_ROOT="$qa_root" \
  PYTHONPATH=. \
  ../../.venv/bin/python "$qa_root/live_verify.py"
```

Move the ignored `profile/`, `scratch/`, and `captures/` runtime directories aside
before a fully fresh rerun. The script recreates the profile and config itself.

## Verification notes

The live harness was corrected without product changes after four harness-only
failures: SVG text split across spans, a stale control-state field assumption, an
omitted required modal constructor argument, and a mounted settings section below the
painted viewport. Preserving those incidents made the final assertions follow the
real UI contracts rather than weakening them.

Three final frames (live displayable, proprietary unavailable, and Required replay)
were rasterized with macOS Quick Look and inspected visually. The committed frames
normalize only exporter trailing whitespace before hashing. This live gate supplements
the earlier 20/20 responsive Textual confirmation matrix; it does not replace the
targeted automated suite.
