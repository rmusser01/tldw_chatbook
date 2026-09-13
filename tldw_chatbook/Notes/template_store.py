"""The shared user note-template file used by UI import and the headless writer."""

import json
import sys

from ..Backup_Recovery import raw_participants as raw


def read_templates():
    """Read selected user templates; shipped fallback remains a separate source."""
    with raw._scope(sys.modules[__name__], "note_templates") as operation:
        selected = raw._selected(operation)
        with raw._file(operation, selected, "r") as stream:
            data = json.load(stream)
        return data.get("templates", data)


def merge_templates(entries, *, unique=False, selected=None):
    """Reread and publish one complete merge; return only committed keys/count."""
    with raw._scope(
        sys.modules[__name__], "note_templates", writing=True, selected_read=selected
    ) as operation:
        destination = raw._selected(operation)
        raw._mkdirs(operation)
        try:
            with raw._file(operation, destination, "r") as stream:
                data = json.load(stream)
                templates = data.get("templates", data)
        except FileNotFoundError:
            templates = {}
        # Corrupt existing files refuse instead of erasing unrelated records.
        if not isinstance(templates, dict):
            raise ValueError("invalid_note_templates")
        keys = []
        for base_key, value in entries:
            key, counter = base_key, 1
            while unique and key in templates:
                key = f"{base_key}_{counter}"
                counter += 1
            templates[key] = value
            keys.append(key)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with raw._file(operation, temporary, "w") as stream:
                json.dump(
                    {"templates": templates}, stream, indent=2, ensure_ascii=False
                )
            raw._replace(operation, temporary, destination)
        finally:
            raw._remove_temporary(operation, temporary)
        return keys, len(templates)
