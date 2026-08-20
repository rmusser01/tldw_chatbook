import pytest

# --- Ported (chunking-engine-parity Task 4) ---------------------------------
# Upstream file: tldw_Server_API/tests/Chunking/test_template_learner.py
# Skipped: templates is deferred to sub-project #2; not in the Phase-A vendored set. Remove this block when the module is vendored in
# its own sub-project and re-sync the test from upstream.
pytest.importorskip("tldw_chatbook.NoSuchDeferredModule",
                    reason="skipped: templates is deferred to sub-project #2; not in the Phase-A vendored set")

from tldw_chatbook.Chunking.engine.templates import TemplateLearner


def test_template_learner_produces_boundaries():

    example = """Chapter 1

    Introduction

    # Section
    Content here
    """
    tpl = TemplateLearner.learn_boundaries(example)
    assert isinstance(tpl, dict)
    assert "boundaries" in tpl
    assert len(tpl["boundaries"]) > 0
