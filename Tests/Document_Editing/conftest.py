import itertools

import pytest

from tldw_chatbook.Document_Editing.models import DocumentRequest, RequestKey


@pytest.fixture
def request_for():
    ids = itertools.count(1)

    def make(text, language="json", **overrides):
        key = dict(  # noqa: C408 - prescribed programme factory
            authority_id="test",
            document_id="n1",
            session_generation=1,
            draft_revision=1,
            language_revision=1,
            request_id=next(ids),
        )
        for field in tuple(key):
            if field in overrides:
                key[field] = overrides.pop(field)
        return DocumentRequest(
            key=RequestKey(**key), text=text, language=language, **overrides
        )

    return make
