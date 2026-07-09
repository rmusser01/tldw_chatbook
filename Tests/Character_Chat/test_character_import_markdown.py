"""Regression tests for Markdown-wrapped character card imports."""

from __future__ import annotations

import json

from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    load_character_card_from_string_content,
)


def _v2_card(name: str = "Markdown Hero") -> dict:
    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": name,
            "description": "Imported from Markdown.",
            "personality": "Careful and direct.",
            "scenario": "A Markdown import test.",
            "first_mes": "Hello from Markdown.",
            "mes_example": "User: hi\nCharacter: hello",
            "creator_notes": "Keep the wrapper format structured.",
            "system_prompt": "Stay in character.",
            "post_history_instructions": "Keep replies brief.",
            "alternate_greetings": ["Second hello."],
            "tags": ["markdown", "test"],
            "creator": "Test Suite",
            "character_version": "1.0",
        },
    }


def test_load_character_card_from_markdown_json_code_block() -> None:
    card = _v2_card("Fenced JSON Hero")
    markdown = f"""# Character Card

```json
{json.dumps(card)}
```
"""

    parsed = load_character_card_from_string_content(markdown)

    assert parsed is not None
    assert parsed["name"] == "Fenced JSON Hero"
    assert parsed["first_message"] == "Hello from Markdown."
    assert parsed["message_example"] == "User: hi\nCharacter: hello"


def test_load_character_card_from_markdown_yaml_frontmatter() -> None:
    markdown = """---
spec: chara_card_v2
spec_version: '2.0'
data:
  name: YAML Hero
  description: Imported from Markdown frontmatter.
  personality: Careful and direct.
  scenario: A Markdown import test.
  first_mes: Hello from YAML frontmatter.
  mes_example: "User: hi\\nCharacter: hello"
  creator_notes: Keep the wrapper format structured.
  system_prompt: Stay in character.
  post_history_instructions: Keep replies brief.
  alternate_greetings:
    - Second hello.
  tags:
    - markdown
    - yaml
  creator: Test Suite
  character_version: '1.0'
---

# YAML Hero
"""

    parsed = load_character_card_from_string_content(markdown)

    assert parsed is not None
    assert parsed["name"] == "YAML Hero"
    assert parsed["first_message"] == "Hello from YAML frontmatter."
    assert parsed["message_example"] == "User: hi\nCharacter: hello"


def test_load_character_card_from_invalid_markdown_returns_none() -> None:
    markdown = """# Not a Character Card

This prose does not contain YAML frontmatter or a fenced JSON character card.
"""

    assert load_character_card_from_string_content(markdown) is None
