# PR2713 current-dev visual review

Twelve actual native captures from the integrated cancellation fix.

| Theme / size | Working | Cancelling | Retry |
| --- | --- | --- | --- |
| textual-dark / 80x24 | [Working / Cancel](textual-dark-80x24-working.svg) | [Cleanup pending](textual-dark-80x24-cancelling.svg) | [Retry available](textual-dark-80x24-retry-available.svg) |
| textual-dark / 170x48 | [Working / Cancel](textual-dark-170x48-working.svg) | [Cleanup pending](textual-dark-170x48-cancelling.svg) | [Retry available](textual-dark-170x48-retry-available.svg) |
| textual-light / 80x24 | [Working / Cancel](textual-light-80x24-working.svg) | [Cleanup pending](textual-light-80x24-cancelling.svg) | [Retry available](textual-light-80x24-retry-available.svg) |
| textual-light / 170x48 | [Working / Cancel](textual-light-170x48-working.svg) | [Cleanup pending](textual-light-170x48-cancelling.svg) | [Retry available](textual-light-170x48-retry-available.svg) |

The conflict keeps both restored-profile guards and the displayed-operation identity.
No stylesheet values changed. Compact detail-toolbar clipping remains in PR2712;
external transport, connected catalog refresh and execution remain separate.
