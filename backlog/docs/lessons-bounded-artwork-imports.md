# Lessons: bounded artwork imports

## Validate ZIP names before the library normalizes them

**TASK-32031, Petdex imports.** A byte-patched ZIP stored `pet.json` followed by a
NUL and extra characters in both ZIP headers. Python zipfile exposed `filename`
as `pet.json`, retained the original text in `orig_filename`, and read it normally.
A traversal check over filename alone admitted the ambiguous manifest. A failing
regression reproduced this; rejecting any original/normalized filename mismatch
before member selection fixed it. Validate the raw member identity as well as the
path that the archive library exposes for extraction.

## Pillow context exit does not necessarily release decoded image memory

**TASK-32031, Petdex preview review.** The first ExitStack implementation entered
Pillow images as context managers. An encoding-failure regression found their
pixel cores still usable after context exit: file handles had closed, but decoded
images remained live. Registering explicit image.close callbacks released every
sheet, crop and converted frame on both success and failure. Verify released
resources directly when claiming bounded preview memory; do not infer it from a
with statement alone.
