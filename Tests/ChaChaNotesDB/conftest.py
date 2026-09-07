# conftest.py for Tests/ChaChaNotesDB (task-1460).
#
# The session-scoped `chachanotes_template_db` fixture these tests copy from
# lives in the ROOT Tests/conftest.py (hoisted in task-1462 so Tests/Chatbooks
# can share it). Fixture resolution walks up the conftest chain, so the
# per-file fixtures in this directory keep working unchanged.
