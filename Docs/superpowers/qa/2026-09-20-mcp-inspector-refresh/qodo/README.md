# PR2757 Qodo closeout

The owner approved the original eight-capture gallery and merge. This follow-up
addresses Qodo review without changing layouts, copy, service admission or styles.

- The alleged orphan mount is unsupported on Textual 8.2.8: registration and
  attachment happen synchronously before the await. Two regressions use the real
  mount/remove boundary, retire while the mount event is still pending, verify
  no stale panel/preview, then reopen exactly one panel.
- Public refresh and token docstrings now explain arguments and nullable ownership.
- Three isolated reconciliation cases and three component cases cover policy-only
  changes. The latter reproduced stale policy facts; effective state, exact-input
  rules and displayed session grants now participate in refresh equality. Changed
  policy retires the existing form/preview, including profile-wide session-list
  changes. Equal definition **and policy** preserve draft/cursor/focus/preview.

[316 current-source targeted passes](qualified-cases.json): 21 refresh/ownership
cases and 295 adjacent inspector/prepared-test cases. No full sweep. The initial
475-case evidence remains historical; the unchanged Workflows dimension failure
is still present on current dev and outside this PR. Independent review found no
blocker. [Finding dispositions and failed attempts](review.json).

[Fresh native result](native/result.json) and [lifecycle](lifecycle.json) qualify
four theme/size cells, eight captures, zero execution/network, normal shutdown,
released lock, ten healthy private DBs and unchanged user defaults. The previous
approved gallery remains representative: [six pixel-identical images and two
caret-only differences](visual-comparison.json), confirmed by manual inspection
and [SVG element deltas](svg-delta.json). Current captures are in `native/`.

Current-head CI, accumulated review and final live-dev/merge-tree verification
remain before the authorized merge.
