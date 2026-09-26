"""`BaseEvalRunner._is_valid_xml` refuses an entity-expansion payload.

Tier-2 review S11, P2 [D1]: this was the sharpest entry on the repo's
open-XXE register (`Tests/Subscriptions/
test_watchlist_opml_entity_expansion.py::_KNOWN_UNHARDENED`) because it
parses **model output** -- a poisoned document in the corpus chooses the
XML the model emits, so the payload is prompt-injection reachable.

`except Exception: return False` is no help: a billion-laughs payload
exhausts memory *during* the parse, before any exception. With defusedxml
the DTD is refused up front and `EntitiesForbidden` (a `ValueError`) is
caught by the existing handler, so the answer becomes the correct
"not valid XML" instead of "valid, and here are your gigabytes".

The legacy eval run stack this method lives in is currently unreachable
from the shipped app (the slice's headline finding), so this is latent
rather than live -- but the register entry is real either way, and the
fix is one import.
"""

from __future__ import annotations

from tldw_chatbook.Evals.eval_runner import BaseEvalRunner

BILLION_LAUGHS_ANSWER = """<?xml version="1.0"?>
<!DOCTYPE answer [
  <!ENTITY lol "lol">
  <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
  <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
]>
<answer>&lol5;</answer>
"""


def test_an_entity_expansion_answer_is_not_valid_xml():
    assert BaseEvalRunner._is_valid_xml(None, BILLION_LAUGHS_ANSWER) is False


def test_ordinary_xml_is_still_valid():
    assert BaseEvalRunner._is_valid_xml(None, "<answer>42</answer>") is True


def test_malformed_xml_is_still_invalid():
    assert BaseEvalRunner._is_valid_xml(None, "<answer>42") is False
