import json
from pathlib import Path
from typing import Union

from pydantic import TypeAdapter

from .models import ProfileManifest, ProfileProposal, ProfileRecord, ProfileScope


CanonicalObject = Union[ProfileManifest, ProfileScope, ProfileRecord, ProfileProposal]


def export_json_schema(path: Path) -> None:
    schema = TypeAdapter(CanonicalObject).json_schema(ref_template="#/$defs/{model}")
    schema.update(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "tldw Personal Context Profile v1",
            "version": 1,
        }
    )
    path.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
