import json
from pathlib import Path

from .models import ProfileManifest, ProfileProposal, ProfileRecord, ProfileScope


def export_json_schema(path: Path) -> None:
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "tldw Personal Context Profile v1", "version": 1,
              "oneOf": [ProfileManifest.model_json_schema(), ProfileScope.model_json_schema(), ProfileRecord.model_json_schema(), ProfileProposal.model_json_schema()]}
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
