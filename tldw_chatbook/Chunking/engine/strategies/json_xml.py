# strategies/json_xml.py
"""
JSON and XML chunking strategies.
Handles structured data formats while preserving structure and relationships.
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Optional
from xml.etree.ElementTree import ParseError

from loguru import logger

from ..base import BaseChunkingStrategy, ChunkMetadata, ChunkResult


# Local helpers to read optional configuration from config.txt / env
def _get_chunking_bool(key: str, default: bool) -> bool:
    try:
        import os
        from tldw_chatbook.Chunking._shims.testing import is_truthy
        v = os.getenv(key.upper())
        if v is None:
            from tldw_chatbook.Chunking._shims.config import load_comprehensive_config
            cp = load_comprehensive_config()
            if hasattr(cp, 'has_section') and cp.has_section('Chunking'):
                v = cp.get('Chunking', key, fallback=str(default))
        s = str(v).strip().lower() if v is not None else str(default).lower()
        return is_truthy(s)
    except (ImportError, AttributeError, KeyError, ValueError) as e:
        logger.debug(f"_get_chunking_bool: config lookup failed for '{key}', using default={default}: {e}")
        return default

def _get_chunking_str(key: str, default: str) -> str:
    try:
        import os
        v = os.getenv(key.upper())
        if v is None:
            from tldw_chatbook.Chunking._shims.config import load_comprehensive_config
            cp = load_comprehensive_config()
            if hasattr(cp, 'has_section') and cp.has_section('Chunking'):
                v = cp.get('Chunking', key, fallback=default)
        return str(v) if v is not None else default
    except (ImportError, AttributeError, KeyError, ValueError) as e:
        logger.debug(f"_get_chunking_str: config lookup failed for '{key}', using default='{default}': {e}")
        return default
import contextlib

from ..exceptions import InvalidInputError
from ..security_logger import get_security_logger

# Precompiled XML security patterns (ASCII-only)
_DOCTYPE_SYSTEM_RE = re.compile(r'<!DOCTYPE\s+[^>]*\bSYSTEM\b', re.IGNORECASE)
_ENTITY_DECL_RE = re.compile(r'<!ENTITY\b', re.IGNORECASE)


def _contains_unsafe_xml_decls(text: str) -> bool:
    """Return True if XML text contains unsafe DOCTYPE SYSTEM or ENTITY declarations."""
    if not text:
        return False
    return bool(_DOCTYPE_SYSTEM_RE.search(text) or _ENTITY_DECL_RE.search(text))


class JSONChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks JSON data while preserving structure.
    Supports both arrays and objects with configurable chunking keys.
    """

    MAX_JSON_SIZE = 50_000_000  # 50MB limit for security

    def __init__(self, language: str = 'en'):
        """
        Initialize JSON chunking strategy.

        Args:
            language: Language code (not used for JSON but kept for consistency)
        """
        super().__init__(language)
        logger.debug("JSONChunkingStrategy initialized")

    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> list[str]:
        """
        Chunk JSON data.

        Args:
            text: JSON string to chunk
            max_size: Maximum items/keys per chunk
            overlap: Number of items/keys to overlap
            **options: Additional options:
                - chunkable_key: Key to chunk in dict (default: 'data')
                - preserve_metadata: Keep non-chunked keys in each chunk
                - output_format: 'json' or 'text'

        Returns:
            List of JSON strings or text representations
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []

        # Security check
        if len(text) > self.MAX_JSON_SIZE:
            raise InvalidInputError(
                f"JSON text size ({len(text)} bytes) exceeds maximum "
                f"allowed size ({self.MAX_JSON_SIZE} bytes)"
            )

        # Quick, non-parsing nesting depth guard to avoid recursion bombs
        # Only counts braces/brackets outside of string literals
        def _estimate_nesting_depth(s: str, limit: int = 2000) -> None:
            in_str = False
            esc = False
            depth = 0
            for ch in s:
                if in_str:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch in '{[':
                        depth += 1
                        if depth > limit:
                            raise InvalidInputError(f"JSON nesting depth exceeds safe limit ({limit})")
                    elif ch in '}]' and depth > 0:
                        depth -= 1

        try:
            _estimate_nesting_depth(text, limit=2000)
        except InvalidInputError:
            # Propagate as intended security validation error
            raise
        except (MemoryError, RecursionError) as e:
            # If our estimator runs out of memory or stack, the JSON is likely too deep
            logger.warning(f"JSON nesting depth check failed with {type(e).__name__}, continuing to parser")
        except (TypeError, ValueError) as e:
            # If our estimator fails for other reasons, continue to json.loads which will raise appropriately
            logger.debug(f"JSON nesting depth estimation failed: {e}")

        # Parse JSON
        try:
            json_data = json.loads(text)
        except (json.JSONDecodeError, RecursionError) as e:
            # RecursionError can occur with extremely deep nesting; surface as InvalidInputError
            logger.error(f"Invalid or excessively nested JSON data: {e}")
            raise InvalidInputError(f"Invalid or excessively nested JSON data: {e}") from e

        # Get options
        output_format = options.get('output_format', 'json')

        # Chunk based on JSON type
        if isinstance(json_data, list):
            chunks = self._chunk_json_list(json_data, max_size, overlap, **options)
        elif isinstance(json_data, dict):
            chunks = self._chunk_json_dict(json_data, max_size, overlap, **options)
        else:
            raise InvalidInputError(
                "JSON must be a top-level array or object. "
                f"Got: {type(json_data).__name__}"
            )

        # Convert to requested format
        if output_format == 'json':
            return [json.dumps(chunk, indent=2) for chunk in chunks]
        else:
            return [self._json_to_text(chunk) for chunk in chunks]

    def chunk_with_metadata(self,
                            text: str,
                            max_size: int,
                            overlap: int = 0,
                            **options) -> list[ChunkResult]:
        """Chunk JSON and return metadata with source offsets aligned to the original text.

        Note: `output_format` affects metadata only. Chunk text is always sourced
        from the original JSON string slice for reliable offsets.
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []

        if len(text) > self.MAX_JSON_SIZE:
            raise InvalidInputError(
                f"JSON text size ({len(text)} bytes) exceeds maximum "
                f"allowed size ({self.MAX_JSON_SIZE} bytes)"
            )

        def _estimate_nesting_depth(s: str, limit: int = 2000) -> None:
            in_str = False
            esc = False
            depth = 0
            for ch in s:
                if in_str:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch in '{[':
                        depth += 1
                        if depth > limit:
                            raise InvalidInputError(f"JSON nesting depth exceeds safe limit ({limit})")
                    elif ch in '}]' and depth > 0:
                        depth -= 1

        _estimate_nesting_depth(text, limit=2000)

        try:
            json_data = json.loads(text)
        except (json.JSONDecodeError, RecursionError) as e:
            logger.error(f"Invalid or excessively nested JSON data: {e}")
            raise InvalidInputError(f"Invalid or excessively nested JSON data: {e}") from e

        output_format = options.get('output_format', 'json')
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if overlap >= max_size:
            logger.warning(f"Overlap {overlap} >= max_size {max_size}. Setting to 0.")
            overlap = 0

        results: list[ChunkResult] = []

        def _emit_chunks(spans: list[tuple[int, int]], method_opts: dict[str, Any]) -> None:
            step = max(1, max_size - overlap)
            idx = 0
            for i in range(0, len(spans), step):
                window = spans[i:i + max_size]
                if not window:
                    continue
                start_char = window[0][0]
                end_char = window[-1][1]
                with contextlib.suppress(Exception):
                    end_char = self._expand_end_to_grapheme_boundary(text, end_char, options=options)
                chunk_text = text[start_char:end_char]
                md = ChunkMetadata(
                    index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    word_count=len(chunk_text.split()) if chunk_text else 0,
                    language=self.language,
                    overlap_with_previous=overlap if i > 0 else 0,
                    overlap_with_next=overlap if (i + step) < len(spans) else 0,
                    method='json',
                    options=method_opts,
                )
                results.append(ChunkResult(text=chunk_text, metadata=md))
                idx += 1

        if isinstance(json_data, list):
            spans = self._scan_top_level_array_spans(text)
            if not spans or len(spans) != len(json_data):
                logger.debug("Array span scan mismatch; falling back to serialized element search")
                spans = []
                cursor = 0
                for item in json_data:
                    try:
                        item_str = json.dumps(item, ensure_ascii=False)
                    except Exception:
                        item_str = str(item)
                    idx = text.find(item_str, cursor)
                    if idx == -1:
                        idx = cursor
                    end = min(len(text), idx + len(item_str))
                    spans.append((idx, end))
                    cursor = end
            _emit_chunks(spans, {'output_format': output_format})
            return results

        if isinstance(json_data, dict):
            chunkable_key = options.get('chunkable_key', 'data')
            preserve_metadata = options.get('preserve_metadata', True)
            pairs = self._scan_top_level_object_pairs(text)
            key_map = {p.get('key'): p for p in pairs if p.get('key') is not None}

            def _emit_pair_chunks(pair_spans: list[tuple[int, int]], note: dict[str, Any]) -> None:
                _emit_chunks(pair_spans, {**note, 'chunkable_key': chunkable_key, 'preserve_metadata': preserve_metadata})

            if chunkable_key in json_data:
                target = json_data[chunkable_key]
                pair = key_map.get(chunkable_key)
                if pair and isinstance(target, list):
                    vstart, vend = pair.get('value_span') or (None, None)
                    if isinstance(vstart, int) and isinstance(vend, int) and vstart < vend:
                        sub = text[vstart:vend]
                        sub_spans = self._scan_top_level_array_spans(sub)
                        if sub_spans:
                            spans = [(vstart + s, vstart + e) for (s, e) in sub_spans]
                            _emit_chunks(spans, {'output_format': output_format, 'chunkable_key': chunkable_key})
                            return results
                elif pair and isinstance(target, dict):
                    vstart, vend = pair.get('value_span') or (None, None)
                    if isinstance(vstart, int) and isinstance(vend, int) and vstart < vend:
                        sub = text[vstart:vend]
                        sub_pairs = self._scan_top_level_object_pairs(sub)
                        if sub_pairs:
                            spans = [(vstart + p['pair_span'][0], vstart + p['pair_span'][1]) for p in sub_pairs]
                            _emit_chunks(spans, {'output_format': output_format, 'chunkable_key': chunkable_key})
                            return results

            # Default: chunk by top-level keys
            pair_spans = [p['pair_span'] for p in pairs]
            _emit_pair_chunks(pair_spans, {'output_format': output_format, 'chunkable_key': chunkable_key})
            return results

        raise InvalidInputError(
            "JSON must be a top-level array or object. "
            f"Got: {type(json_data).__name__}"
        )

    def _scan_top_level_array_spans(self, text: str) -> list[tuple[int, int]]:
        """Return spans of top-level array elements in the original JSON text."""
        spans: list[tuple[int, int]] = []
        n = len(text)
        i = 0
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != '[':
            return spans
        i += 1
        depth = 1
        in_str = False
        esc = False
        elem_start: Optional[int] = None
        last_non_ws: Optional[int] = None
        while i < n:
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
                if depth == 1 and elem_start is None:
                    elem_start = i
                    last_non_ws = i
                i += 1
                continue
            if ch in '[{':
                if depth == 1 and elem_start is None:
                    elem_start = i
                    last_non_ws = i
                depth += 1
                i += 1
                continue
            if ch in ']}':
                if depth == 1:
                    if elem_start is not None:
                        end = (last_non_ws + 1) if last_non_ws is not None else i
                        spans.append((elem_start, end))
                        elem_start = None
                        last_non_ws = None
                    depth -= 1
                    i += 1
                    break
                depth -= 1
                if elem_start is not None and not ch.isspace():
                    last_non_ws = i
                i += 1
                continue
            if depth == 1:
                if ch == ',':
                    if elem_start is not None:
                        end = (last_non_ws + 1) if last_non_ws is not None else i
                        spans.append((elem_start, end))
                        elem_start = None
                        last_non_ws = None
                elif not ch.isspace():
                    if elem_start is None:
                        elem_start = i
                    last_non_ws = i
            i += 1
        return spans

    def _scan_top_level_object_pairs(self, text: str) -> list[dict[str, Any]]:
        """Return spans for top-level key/value pairs in the original JSON text."""
        pairs: list[dict[str, Any]] = []
        n = len(text)
        i = 0
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != '{':
            return pairs
        i += 1
        depth = 1
        in_str = False
        esc = False
        pair_start: Optional[int] = None
        last_non_ws: Optional[int] = None
        key_buf: Optional[list[str]] = None
        reading_key = False
        current_key: Optional[str] = None
        expecting_key = True
        expecting_value = False
        value_start: Optional[int] = None
        while i < n:
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                    if reading_key and key_buf is not None:
                        key_raw = '"' + ''.join(key_buf) + '"'
                        try:
                            current_key = json.loads(key_raw)
                        except Exception:
                            current_key = ''.join(key_buf)
                        reading_key = False
                        expecting_key = False
                    if pair_start is not None:
                        last_non_ws = i
                else:
                    if reading_key and key_buf is not None:
                        key_buf.append(ch)
                i += 1
                continue
            if ch == '"':
                in_str = True
                if depth == 1 and expecting_key:
                    reading_key = True
                    key_buf = []
                    if pair_start is None:
                        pair_start = i
                if depth == 1 and expecting_value and value_start is None:
                    value_start = i
                    expecting_value = False
                if pair_start is not None:
                    last_non_ws = i
                i += 1
                continue
            if ch in '[{':
                if depth == 1 and expecting_value and value_start is None:
                    value_start = i
                    expecting_value = False
                if pair_start is not None and not ch.isspace():
                    last_non_ws = i
                depth += 1
                i += 1
                continue
            if ch in ']}':
                if depth == 1:
                    if pair_start is not None:
                        end = (last_non_ws + 1) if last_non_ws is not None else i
                        pairs.append({
                            'key': current_key,
                            'pair_span': (pair_start, end),
                            'value_span': (value_start if value_start is not None else pair_start, end),
                        })
                        pair_start = None
                        last_non_ws = None
                        current_key = None
                        value_start = None
                        expecting_key = True
                        expecting_value = False
                    depth -= 1
                    i += 1
                    break
                depth -= 1
                if pair_start is not None and not ch.isspace():
                    last_non_ws = i
                i += 1
                continue
            if depth == 1:
                if ch == ':':
                    expecting_value = True
                    i += 1
                    continue
                if ch == ',':
                    if pair_start is not None:
                        end = (last_non_ws + 1) if last_non_ws is not None else i
                        pairs.append({
                            'key': current_key,
                            'pair_span': (pair_start, end),
                            'value_span': (value_start if value_start is not None else pair_start, end),
                        })
                    pair_start = None
                    last_non_ws = None
                    current_key = None
                    value_start = None
                    expecting_key = True
                    expecting_value = False
                    i += 1
                    continue
                if not ch.isspace():
                    if pair_start is None:
                        pair_start = i
                    last_non_ws = i
                    if expecting_value and value_start is None:
                        value_start = i
                        expecting_value = False
            else:
                if pair_start is not None and not ch.isspace():
                    last_non_ws = i
            i += 1
        return pairs

    def _chunk_json_list(self,
                        json_list: list[Any],
                        max_size: int,
                        overlap: int,
                        **options) -> list[list[Any]]:
        """
        Chunk a JSON array.

        Args:
            json_list: List to chunk
            max_size: Maximum items per chunk
            overlap: Number of items to overlap
            **options: Additional options

        Returns:
            List of chunked arrays
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        if overlap >= max_size:
            logger.warning(f"Overlap {overlap} >= max_size {max_size}. Setting to 0.")
            overlap = 0

        chunks = []
        step = max(1, max_size - overlap)

        for i in range(0, len(json_list), step):
            chunk = json_list[i:i + max_size]
            chunks.append(chunk)

        return chunks

    def _chunk_json_dict(self,
                        json_dict: dict[str, Any],
                        max_size: int,
                        overlap: int,
                        **options) -> list[dict[str, Any]]:
        """
        Chunk a JSON object.

        Args:
            json_dict: Dictionary to chunk
            max_size: Maximum keys per chunk
            overlap: Number of keys to overlap
            **options: Additional options including 'chunkable_key'

        Returns:
            List of chunked dictionaries
        """
        chunkable_key = options.get('chunkable_key', 'data')
        preserve_metadata = options.get('preserve_metadata', True)
        # Global toggle to emit a single metadata reference instead of repeating
        single_ref = bool(options.get('single_metadata_reference', _get_chunking_bool('json_single_metadata_reference', False)))
        ref_key = str(options.get('metadata_reference_key', _get_chunking_str('json_metadata_reference_key', '__meta_ref__')))

        # Check if chunkable key exists and is dict or list
        if chunkable_key not in json_dict:
            # If no chunkable key, chunk the entire dict by keys
            return self._chunk_dict_by_keys(json_dict, max_size, overlap)

        chunkable_data = json_dict[chunkable_key]

        if isinstance(chunkable_data, list):
            # Chunk the list within the dict
            chunked_lists = self._chunk_json_list(chunkable_data, max_size, overlap)

            chunks = []
            meta_payload = {k: v for k, v in json_dict.items() if k != chunkable_key}
            meta_ref_id = None
            if preserve_metadata and single_ref and meta_payload:
                # Stable reference id based on metadata payload
                try:
                    import hashlib as _hash
                    meta_ref_id = _hash.sha1(
                        json.dumps(meta_payload, sort_keys=True).encode('utf-8'),
                        usedforsecurity=False,
                    ).hexdigest()[:12]
                except (TypeError, ValueError, json.JSONDecodeError) as e:
                    logger.debug(f"Failed to generate metadata reference hash: {e}")
                    meta_ref_id = 'meta'
                # Emit a leading metadata chunk
                chunks.append({ref_key: meta_ref_id, 'metadata': meta_payload})
            for chunk_list in chunked_lists:
                if preserve_metadata:
                    if single_ref and meta_ref_id is not None:
                        new_chunk = {chunkable_key: chunk_list, ref_key: meta_ref_id}
                    else:
                        # Keep other keys
                        new_chunk = json_dict.copy()
                        new_chunk[chunkable_key] = chunk_list
                else:
                    new_chunk = {chunkable_key: chunk_list}
                chunks.append(new_chunk)

            return chunks

        elif isinstance(chunkable_data, dict):
            # Chunk the nested dict
            chunked_dicts = self._chunk_dict_by_keys(chunkable_data, max_size, overlap)

            chunks = []
            meta_payload = {k: v for k, v in json_dict.items() if k != chunkable_key}
            meta_ref_id = None
            if preserve_metadata and single_ref and meta_payload:
                try:
                    import hashlib as _hash
                    meta_ref_id = _hash.sha1(
                        json.dumps(meta_payload, sort_keys=True).encode('utf-8'),
                        usedforsecurity=False,
                    ).hexdigest()[:12]
                except (TypeError, ValueError, json.JSONDecodeError) as e:
                    logger.debug(f"Failed to generate metadata reference hash: {e}")
                    meta_ref_id = 'meta'
                chunks.append({ref_key: meta_ref_id, 'metadata': meta_payload})
            for chunk_dict in chunked_dicts:
                if preserve_metadata:
                    if single_ref and meta_ref_id is not None:
                        new_chunk = {chunkable_key: chunk_dict, ref_key: meta_ref_id}
                    else:
                        new_chunk = json_dict.copy()
                        new_chunk[chunkable_key] = chunk_dict
                else:
                    new_chunk = {chunkable_key: chunk_dict}
                chunks.append(new_chunk)

            return chunks
        else:
            # Can't chunk this key's value
            logger.warning(f"Chunkable key '{chunkable_key}' is not a list or dict")
            return [json_dict]


    def _chunk_dict_by_keys(self,
                           data_dict: dict[str, Any],
                           max_size: int,
                           overlap: int) -> list[dict[str, Any]]:
        """
        Chunk a dictionary by its keys.

        Args:
            data_dict: Dictionary to chunk
            max_size: Maximum keys per chunk
            overlap: Number of keys to overlap

        Returns:
            List of chunked dictionaries
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        if overlap >= max_size:
            logger.warning(f"Overlap {overlap} >= max_size {max_size}. Setting to 0.")
            overlap = 0

        all_keys = list(data_dict.keys())
        chunks = []
        step = max(1, max_size - overlap)

        for i in range(0, len(all_keys), step):
            chunk_keys = all_keys[i:i + max_size]
            chunk_dict = {k: data_dict[k] for k in chunk_keys}
            chunks.append(chunk_dict)

        return chunks

    def _json_to_text(self, json_obj: Any) -> str:
        """
        Convert JSON object to human-readable text.

        Args:
            json_obj: JSON object to convert

        Returns:
            Text representation
        """
        if isinstance(json_obj, dict):
            lines = []
            for key, value in json_obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{key}: {json.dumps(value, indent=2)}")
                else:
                    lines.append(f"{key}: {value}")
            return '\n'.join(lines)
        elif isinstance(json_obj, list):
            return '\n'.join([str(item) for item in json_obj])
        else:
            return str(json_obj)


class XMLChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks XML data while preserving structure and hierarchy.
    """

    MAX_XML_SIZE = 50_000_000  # 50MB limit for security

    def __init__(self, language: str = 'en'):
        """
        Initialize XML chunking strategy.

        Args:
            language: Language code for text content
        """
        super().__init__(language)
        self._security_logger = get_security_logger()
        logger.debug("XMLChunkingStrategy initialized")

    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> list[str]:
        """
        Chunk XML data.

        Args:
            text: XML string to chunk
            max_size: Maximum words in content per chunk
            overlap: Number of XML elements to overlap
            **options: Additional options:
                - preserve_structure: Keep parent elements
                - output_format: 'xml' or 'text'
                - include_paths: Include element paths in output

        Returns:
            List of XML strings or text representations
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []

        # Security check
        if len(text) > self.MAX_XML_SIZE:
            raise InvalidInputError(
                f"XML text size ({len(text)} bytes) exceeds maximum "
                f"allowed size ({self.MAX_XML_SIZE} bytes)"
            )

        # Parse XML with security hardening against XXE attacks
        try:
            # Try to use defusedxml if available (most secure)
            try:
                import defusedxml.ElementTree as DefusedET

                # Import the common submodule explicitly to reference exception types
                from defusedxml import common as defused_common

                # Additional pre-check for unsafe DOCTYPE SYSTEM / ENTITY declarations
                if _contains_unsafe_xml_decls(text):
                    logger.error("XML contains unsafe DOCTYPE SYSTEM or ENTITY declaration")
                    self._security_logger.log_xxe_attempt(text[:500], source="xml_chunk")
                    raise InvalidInputError("XML contains unsafe DOCTYPE SYSTEM or ENTITY declaration which is not allowed for security reasons")

                try:
                    root = DefusedET.fromstring(text)
                    logger.debug("Using defusedxml for secure XML parsing")
                except (defused_common.EntitiesForbidden,
                        defused_common.ExternalReferenceForbidden,
                        defused_common.DTDForbidden,
                        defused_common.NotSupportedError) as e:
                    logger.error(f"Blocked potential XXE attack: {e}")
                    raise InvalidInputError(f"XML contains forbidden constructs (potential XXE attack): {e}") from None
            except ImportError:
                raise InvalidInputError(
                    "Secure XML parsing requires the 'defusedxml' package."
                ) from None
        except ParseError as e:
            logger.error(f"Invalid XML data: {e}")
            raise InvalidInputError(f"Invalid XML data: {e}") from e

        # Get options
        output_format = options.get('output_format', 'text')
        include_paths = options.get('include_paths', True)

        # Extract XML structure
        elements = self._extract_xml_elements(root)

        if not elements:
            return []

        # Chunk elements
        chunks = self._chunk_xml_elements(elements, max_size, overlap, **options)

        # Convert to requested format
        result = []
        for chunk in chunks:
            if output_format == 'xml':
                # Reconstruct XML from elements
                result.append(self._elements_to_xml(chunk, root.tag, root.attrib))
            else:
                # Convert to text
                result.append(self._elements_to_text(chunk, include_paths))

        return result

    def chunk_with_metadata(self,
                            text: str,
                            max_size: int,
                            overlap: int = 0,
                            **options) -> list[ChunkResult]:
        """Chunk XML data and return results with source offsets."""
        if not self.validate_parameters(text, max_size, overlap):
            return []

        if len(text) > self.MAX_XML_SIZE:
            raise InvalidInputError(
                f"XML text size ({len(text)} bytes) exceeds maximum "
                f"allowed size ({self.MAX_XML_SIZE} bytes)"
            )

        try:
            try:
                import defusedxml.ElementTree as DefusedET
                from defusedxml import common as defused_common

                if _contains_unsafe_xml_decls(text):
                    logger.error("XML contains unsafe DOCTYPE SYSTEM or ENTITY declaration")
                    self._security_logger.log_xxe_attempt(text[:500], source="xml_chunk")
                    raise InvalidInputError("XML contains unsafe DOCTYPE SYSTEM or ENTITY declaration which is not allowed for security reasons")

                try:
                    root = DefusedET.fromstring(text)
                    logger.debug("Using defusedxml for secure XML parsing")
                except (defused_common.EntitiesForbidden,
                        defused_common.ExternalReferenceForbidden,
                        defused_common.DTDForbidden,
                        defused_common.NotSupportedError) as e:
                    logger.error(f"Blocked potential XXE attack: {e}")
                    raise InvalidInputError(f"XML contains forbidden constructs (potential XXE attack): {e}") from None
            except ImportError:
                raise InvalidInputError(
                    "Secure XML parsing requires the 'defusedxml' package."
                ) from None
        except ParseError as e:
            logger.error(f"Invalid XML data: {e}")
            raise InvalidInputError(f"Invalid XML data: {e}") from e

        output_format = options.get('output_format', 'text')
        include_paths = options.get('include_paths', True)

        elements_raw = self._extract_xml_elements_with_raw(root)
        if not elements_raw:
            return []

        # Map element content to source spans using rolling search for stability.
        cursor = 0
        element_entries: list[tuple[str, str, ET.Element, int, int]] = []
        for path, raw_text, elem in elements_raw:
            raw = raw_text
            if not raw:
                continue
            idx = text.find(raw, cursor)
            if idx == -1:
                stripped = raw.strip()
                if stripped:
                    idx = text.find(stripped, cursor)
                    if idx != -1:
                        raw = stripped
            if idx == -1:
                idx = cursor
            end = min(len(text), idx + len(raw))
            cursor = end
            element_entries.append((path, raw, elem, idx, end))

        # Chunk elements by word count, with overlap in element units.
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if overlap >= len(element_entries) and len(element_entries) > 0:
            logger.warning(f"Overlap {overlap} >= total elements. Setting to 0.")
            overlap = 0

        chunks: list[list[tuple[str, str, ET.Element, int, int]]] = []
        current_chunk: list[tuple[str, str, ET.Element, int, int]] = []
        current_word_count = 0

        for entry in element_entries:
            _path, raw, _elem, _s, _e = entry
            content = raw.strip()
            if not content:
                continue
            word_count = len(content.split())

            if current_word_count + word_count > max_size and current_chunk:
                chunks.append(current_chunk)
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:]
                    current_word_count = sum(len(e[1].strip().split()) for e in current_chunk)
                else:
                    current_chunk = []
                    current_word_count = 0

            current_chunk.append(entry)
            current_word_count += word_count

        if current_chunk:
            chunks.append(current_chunk)

        results: list[ChunkResult] = []
        for idx, chunk in enumerate(chunks):
            start_char = min(e[3] for e in chunk)
            end_char = max(e[4] for e in chunk)
            with contextlib.suppress(Exception):
                end_char = self._expand_end_to_grapheme_boundary(text, end_char, options=options)
            # Build chunk text in requested output format (offsets still refer to source).
            if output_format == 'xml':
                chunk_elements = [(p, t.strip(), el) for (p, t, el, _s, _e) in chunk]
                chunk_text = self._elements_to_xml(chunk_elements, root.tag, root.attrib)
            else:
                chunk_elements = [(p, t.strip(), el) for (p, t, el, _s, _e) in chunk]
                chunk_text = self._elements_to_text(chunk_elements, include_paths)
            md = ChunkMetadata(
                index=idx,
                start_char=start_char,
                end_char=end_char,
                word_count=len(chunk_text.split()) if chunk_text else 0,
                language=self.language,
                method='xml',
                options={'output_format': output_format, 'include_paths': include_paths},
            )
            results.append(ChunkResult(text=chunk_text, metadata=md))

        return results

    def _extract_xml_elements(self,
                             element: ET.Element,
                             path: str = "") -> list[tuple[str, str, ET.Element]]:
        """
        Recursively extract XML elements with paths.

        Args:
            element: XML element to process
            path: Current path in XML tree

        Returns:
            List of (path, text_content, element) tuples
        """
        results = []

        # Build current path
        current_path = f"{path}/{element.tag}" if path else element.tag

        # Get element text content
        text_content = ""
        if element.text:
            text_content = element.text.strip()

        # Add current element if it has content
        if text_content:
            results.append((current_path, text_content, element))

        # Process children
        for child in element:
            results.extend(self._extract_xml_elements(child, current_path))
            # Also get tail text
            if child.tail:
                tail_text = child.tail.strip()
                if tail_text:
                    results.append((f"{current_path}/{child.tag}[tail]", tail_text, child))

        return results

    def _extract_xml_elements_with_raw(self,
                                       element: ET.Element,
                                       path: str = "") -> list[tuple[str, str, ET.Element]]:
        """Extract XML elements preserving raw text for offset mapping."""
        results = []
        current_path = f"{path}/{element.tag}" if path else element.tag

        raw_text = element.text if element.text is not None else ""
        if raw_text and raw_text.strip():
            results.append((current_path, raw_text, element))

        for child in element:
            results.extend(self._extract_xml_elements_with_raw(child, current_path))
            if child.tail:
                tail_text = child.tail
                if tail_text and tail_text.strip():
                    results.append((f"{current_path}/{child.tag}[tail]", tail_text, child))

        return results

    def _chunk_xml_elements(self,
                           elements: list[tuple[str, str, ET.Element]],
                           max_size: int,
                           overlap: int,
                           **options) -> list[list[tuple[str, str, ET.Element]]]:
        """
        Chunk XML elements by content size.

        Args:
            elements: List of (path, content, element) tuples
            max_size: Maximum words per chunk
            overlap: Number of elements to overlap
            **options: Additional options

        Returns:
            List of chunked element lists
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        if overlap >= len(elements) and len(elements) > 0:
            logger.warning(f"Overlap {overlap} >= total elements. Setting to 0.")
            overlap = 0

        chunks = []
        current_chunk = []
        current_word_count = 0

        for _i, (path, content, elem) in enumerate(elements):
            word_count = len(content.split())

            # Check if adding this element exceeds max_size
            if current_word_count + word_count > max_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk)

                # Handle overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    # Keep last 'overlap' elements
                    current_chunk = current_chunk[-overlap:]
                    current_word_count = sum(len(c.split()) for _, c, _ in current_chunk)
                else:
                    current_chunk = []
                    current_word_count = 0

            # Add element to current chunk
            current_chunk.append((path, content, elem))
            current_word_count += word_count

        # Add remaining elements
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _elements_to_text(self,
                         elements: list[tuple[str, str, ET.Element]],
                         include_paths: bool) -> str:
        """
        Convert XML elements to text representation.

        Args:
            elements: List of (path, content, element) tuples
            include_paths: Whether to include element paths

        Returns:
            Text representation
        """
        lines = []

        for path, content, _ in elements:
            if include_paths:
                lines.append(f"[{path}]: {content}")
            else:
                lines.append(content)

        return '\n'.join(lines)

    def _elements_to_xml(self,
                        elements: list[tuple[str, str, ET.Element]],
                        root_tag: str,
                        root_attrib: dict[str, str]) -> str:
        """
        Reconstruct XML from elements, preserving hierarchy based on path.

        Args:
            elements: List of (path, content, element) tuples
            root_tag: Root element tag
            root_attrib: Root element attributes

        Returns:
            XML string
        """
        # Create new root
        new_root = ET.Element(root_tag, root_attrib)

        # Track added elements and path-to-element mapping
        orig_to_new: dict[ET.Element, ET.Element] = {}
        path_to_element: dict[str, ET.Element] = {root_tag: new_root}

        for path, content, elem in elements:
            is_tail = path.endswith("[tail]")
            base_path = path[:-6] if is_tail else path

            # Parse path to reconstruct hierarchy
            # Path format: "root/parent/child" or "root/parent[0]/child[1]"
            path_parts = base_path.split('/')

            # Build parent path and ensure all ancestors exist
            current_parent = new_root
            current_path = root_tag

            for _i, part in enumerate(path_parts[1:-1] if len(path_parts) > 1 else []):
                # Strip array index if present (e.g., "item[0]" -> "item")
                tag_name = part.split('[')[0] if '[' in part else part
                current_path = f"{current_path}/{part}"

                if current_path not in path_to_element:
                    # Create intermediate element
                    intermediate = ET.SubElement(current_parent, tag_name)
                    path_to_element[current_path] = intermediate

                current_parent = path_to_element[current_path]

            new_elem = orig_to_new.get(elem)
            if new_elem is None:
                # Add the actual element as child of its parent
                new_elem = ET.SubElement(current_parent, elem.tag, elem.attrib)
                orig_to_new[elem] = new_elem
                path_to_element[base_path] = new_elem

            if is_tail:
                if content:
                    new_elem.tail = (new_elem.tail or "") + content
            else:
                new_elem.text = content

        return ET.tostring(new_root, encoding='unicode')
