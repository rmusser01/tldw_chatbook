# ADR-160: Progressive file picker listings

Status: Accepted (user-approved progressive loading and sorting, 2026-09-13)
Task: TASK-32565

## Decision

Shared file pickers start in filesystem discovery order and publish usable
entries before enumeration finishes. Progress reports discovered counts rather
than an invented percentage; completion is explicit. Sorting controls remain
available during and after scanning. Choosing a sort applies it to discovered
entries and subsequent batches, preserving the highlighted path and selection.
No automatic sort occurs when scanning completes.

Offer discovery order, name, modified time, last accessed time, creation time,
and size, with ascending/descending directions. The parent entry stays first.
Timestamp and size sorts require metadata for all discovered entries and run
off the UI thread; ordinary discovery/name browsing fetches metadata only near
the viewport. Missing metadata sorts last in either direction. Creation means
filesystem birth time where available, never POSIX inode-change time. Access
time is the filesystem-reported value (which may be maintained lazily).

Scanning, metadata, and user-supplied file filters run in workers. Directory
generations and projection revisions reject stale publication after navigation,
refresh, filter/sort changes, or dismissal. Bounded delivery prevents a fast
producer flooding the UI. Filesystem calls cannot be forcibly interrupted;
cancelled operations stop cooperatively and may never publish into a new view.
Disk contents are read again only on navigation/refresh; cached records are
ephemeral and do not introduce persistent state.

Keep the existing OptionList navigation/caller contracts. Rows have a known
single-line height, avoiding eager Rich measurement of offscreen rows. Append
and rebuild in short batches, yielding between them. User sorting can move
rows, but must preserve highlighted/selected paths and keyboard navigation.

## Alternatives

- Debounce alone already reduced search rebuild frequency but left eager work.
- Sorting before first display delays every usable result until enumeration ends.
- A new virtual-list widget would duplicate selection and keyboard contracts;
  bounded publication and constant-height measurement use the existing widget.
- Using ctime as creation time is incorrect on POSIX; unavailable birth time
  remains unknown and is explained in the UI.

## Verification

Mounted checks gate filesystem enumeration and metadata to prove partial
results, responsive interaction, stale-result rejection and lazy access.
Sorting tests include missing birth times, both directions and mid-scan changes.
Large real temporary directories measure UI heartbeat gaps, first-result latency
and scroll/resize behavior. Existing shared/enhanced picker tests cover consumers.
