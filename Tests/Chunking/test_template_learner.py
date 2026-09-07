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
