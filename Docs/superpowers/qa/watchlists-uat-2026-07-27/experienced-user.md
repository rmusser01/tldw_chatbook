# Watchlists UAT — experienced-user journey, 2026-07-28

Follow-up to `notes.md`, which stopped at a blocked new-user path. That block
(task-995) is fixed, so this run picks up from there. `origin/dev` `13551ae8a`,
clean profile (`uat_wl_exp`, deleted afterwards), 235x52.

**Click columns were computed by character this time**, via a helper, after the
first UAT mis-filed task-996 off byte offsets. See
`backlog/docs/lessons-live-verification.md`.

## Confirmed fixed, live

- **task-995** — the Sources toolbar renders its controls: search box, all three
  filters, `New Source`, `Filters`. `New Source` opens the create form.
- **task-997** — the tree draws `▸  Morning AI Brief  0` on one row.
- **task-996** — clicking `Notifications` switched to Notifications, and every
  other click landed on the control under the pointer. The strip routes
  correctly; the original report was a harness error.

## Reached

Created two watchlists through the UI (`Morning AI Brief`, `Security Watch`).
Both appear in the tree, and the centre heading follows the newly created one —
scope-follows-creation works.

## Blocked again, one step further along

**The create-source form cannot be filled in.** Typing after opening it, clicking
into the `Name` input's own row, and tabbing five times then typing all left the
fields empty. A control click on `Notifications` immediately afterwards switched
the section correctly, so the app was responsive throughout.

Filed as **task-1035 (critical)**.

## Still not covered

Everything downstream of having sources: tag filtering, scope switching across
watchlists with real data, Console staging of a scope, rename, delete, and the
Feeds/Items/Content reading path. The experienced-user journey needs a third run
once 1035 is resolved.
