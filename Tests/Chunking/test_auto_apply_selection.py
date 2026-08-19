import pytest

# --- Ported (chunking-engine-parity Task 4) ---------------------------------
# Upstream file: tldw_Server_API/tests/Chunking/test_auto_apply_selection.py
# Skipped: templates (TemplateClassifier) is deferred to sub-project #2; not in the Phase-A vendored set. Remove this block when the module is vendored in
# its own sub-project and re-sync the test from upstream.
pytest.importorskip("tldw_chatbook.NoSuchDeferredModule",
                    reason="skipped: templates (TemplateClassifier) is deferred to sub-project #2; not in the Phase-A vendored set")

from tldw_chatbook.Chunking.engine.templates import TemplateClassifier


def test_auto_apply_selects_higher_score():

    t1 = {"classifier": {"media_types": ["document"], "title_regex": r"^Doc", "min_score": 0.0, "priority": 0}}
    t2 = {"classifier": {"media_types": ["document"], "filename_regex": r"\.pdf$", "min_score": 0.0, "priority": 0}}
    s1 = TemplateClassifier.score(t1, media_type="document", title="Doc Title", url=None, filename=None)
    s2 = TemplateClassifier.score(t2, media_type="document", title="Other", url=None, filename="file.pdf")
    # Both non-zero; ensure at least one scores
    assert max(s1, s2) > 0
