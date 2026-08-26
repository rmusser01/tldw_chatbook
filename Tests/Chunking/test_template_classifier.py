import pytest

from tldw_chatbook.Chunking.engine.templates import (
    TemplateClassifier,
    TemplateProcessor,
    ChunkingTemplate,
    TemplateStage,
)


def test_template_classifier_scores_basic_matches():

    cfg = {
        "classifier": {
            "media_types": ["document"],
            "title_regex": r"^My Report",
            "min_score": 0.1,
            "priority": 1,
        }
    }
    s = TemplateClassifier.score(cfg, media_type="document", title="My Report 2024", url=None, filename=None)
    assert s >= 0.1


def test_template_processor_hierarchical_returns_dict_chunks():

    text = "# Heading\n\nPara one.\n\nPara two."
    template = ChunkingTemplate(
        name="demo",
        base_method="sentences",
        stages=[
            TemplateStage(
                "chunk",
                [
                    {
                        "method": "sentences",
                        "config": {
                            "hierarchical": True,
                            "hierarchical_template": {
                                "boundaries": [{"kind": "header_atx", "pattern": r"^#\\s+", "flags": "m"}]
                            },
                        },
                    }
                ],
            )
        ],
    )
    proc = TemplateProcessor()
    chunks = proc.process_template(text, template)
    assert isinstance(chunks, list)
    assert isinstance(chunks[0], dict)
    assert "text" in chunks[0]
    assert "metadata" in chunks[0]
