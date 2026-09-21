# Tools header visual review

Only the header row changes. Tool identities, values, focused selection and the
surrounding layout are retained. These are native captures, using synthetic
catalog labels and tags. The final runner checks the composed screen.

## Wide — 170×48

| Theme | Before | After |
|---|---|---|
| Dark | ![Dark before](before/textual-dark-170x48-tools.svg) | ![Dark after](after/textual-dark-170x48-tools.svg) |
| Light | ![Light before](before/textual-light-170x48-tools.svg) | ![Light after](after/textual-light-170x48-tools.svg) |

## Compact — 120×40

The Schema column retains its existing horizontal-scroll behavior.

| Theme | Before | After |
|---|---|---|
| Dark | ![Dark compact before](before/textual-dark-120x40-tools.svg) | ![Dark compact after](after/textual-dark-120x40-tools.svg) |
| Light | ![Light compact before](before/textual-light-120x40-tools.svg) | ![Light compact after](after/textual-light-120x40-tools.svg) |

[Verification and scope](README.md) · [Measured terminal/pixel differences](visual-comparison.json)

The owner approved these captures for PR2749. [Final integration](CLOSEOUT.md)
confirms the same four views remain pixel-identical on dev `b91340a5db`.

[Rebased validation](rebased/verification.json) confirms all four views remain
pixel-identical on dev `495b2c4522`; the rebase required no conflict choices.
