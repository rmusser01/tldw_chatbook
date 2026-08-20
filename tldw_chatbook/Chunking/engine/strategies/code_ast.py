"""
Python AST-based code chunking strategy.

Segments Python source files into logical blocks using the built-in ast module,
emitting import blocks and top-level classes/functions as primary units.

Falls back to greedy packing by character size with optional overlap.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from loguru import logger

from ..base import BaseChunkingStrategy, ChunkMetadata, ChunkResult


@dataclass
class _Block:
    kind: str  # 'imports' | 'class' | 'function'
    name: str | None
    start_line: int  # 0-based inclusive
    end_line: int    # 0-based exclusive


class PythonASTCodeChunkingStrategy(BaseChunkingStrategy):
    """AST-driven code chunker for Python files."""

    def __init__(self, language: str = 'python'):
        super().__init__(language='python')

    def _line_starts(self, text: str) -> list[int]:
        starts: list[int] = []
        pos = 0
        for _idx, ln in enumerate(text.splitlines(keepends=True)):
            starts.append(pos)
            pos += len(ln)
        # Ensure at least one start
        if not starts:
            starts.append(0)
        return starts

    def _span_chars(self, line_starts: list[int], total_chars: int, s_line: int, e_line: int) -> tuple[int, int]:
        if s_line < 0:
            s_line = 0
        if e_line < 0:
            e_line = 0
        s_char = total_chars if s_line >= len(line_starts) else line_starts[s_line]
        e_char = total_chars if e_line >= len(line_starts) else line_starts[e_line]
        if e_char < s_char:
            e_char = s_char
        return s_char, e_char

    def _extract_import_block(self, lines: list[str]) -> tuple[int, int]:
        # From file start until the first non-import/non-comment line (keep leading comments/docstring region contiguous)
        i = 0
        end = 0
        seen_non_import = False
        while i < len(lines):
            s = lines[i].strip()
            if not s or s.startswith(('#', '"""', "'''")):
                if not seen_non_import:
                    end = i + 1
                i += 1
                continue
            if s.startswith('import ') or s.startswith('from '):
                end = i + 1
                i += 1
                continue
            seen_non_import = True
            break
        return (0, end)

    def _collect_blocks(self, text: str) -> list[_Block]:
        blocks: list[_Block] = []
        try:
            tree = ast.parse(text)
        except (SyntaxError, TypeError, ValueError) as e:
            logger.warning(f"AST parse failed; falling back to single block: {e}")
            return [_Block('module', None, 0, len(text.splitlines()))]

        lines = text.splitlines()
        # Import/header block
        imp_s, imp_e = self._extract_import_block(lines)
        if imp_e > imp_s:
            blocks.append(_Block('imports', None, imp_s, imp_e))

        # Top-level class and function defs
        for node in tree.body:
            try:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    s = max(0, int(getattr(node, 'lineno', 1)) - 1)
                    e = int(getattr(node, 'end_lineno', s + 1))
                    blocks.append(_Block('function', node.name, s, e))
                elif isinstance(node, ast.ClassDef):
                    s = max(0, int(getattr(node, 'lineno', 1)) - 1)
                    e = int(getattr(node, 'end_lineno', s + 1))
                    blocks.append(_Block('class', node.name, s, e))
            except (AttributeError, TypeError, ValueError):
                continue

        if not blocks:
            # whole module fallback
            blocks.append(_Block('module', None, 0, len(lines)))

        # Ensure non-overlapping and sorted
        blocks.sort(key=lambda b: (b.start_line, b.end_line))
        dedup: list[_Block] = []
        for b in blocks:
            if not dedup:
                dedup.append(b)
            else:
                last = dedup[-1]
                s = max(b.start_line, last.end_line)
                e = max(s, b.end_line)
                dedup.append(_Block(b.kind, b.name, s, e))
        return dedup

    def _pack_blocks(self, blocks: list[_Block], lines_ke: list[str], max_chars: int, overlap_chars: int) -> list[str]:
        chunks: list[str] = []
        buf = ''
        current_len = 0
        line_starts = []
        pos = 0
        for ln in lines_ke:
            line_starts.append(pos)
            pos += len(ln)
        for b in blocks:
            seg = ''.join(lines_ke[b.start_line:b.end_line]).rstrip()
            if not seg:
                continue
            if not buf:
                buf = seg
                current_len = len(seg)
            elif current_len + 2 + len(seg) <= max_chars:
                buf = f"{buf}\n\n{seg}"
                current_len = len(buf)
            else:
                chunks.append(buf)
                if len(seg) > max_chars:
                    # Split oversized block into non-overlapping windows.
                    # Overlap is added later via prefixing tails.
                    t = seg
                    while t:
                        part = t[:max_chars]
                        chunks.append(part)
                        t = '' if len(t) <= max_chars else t[max_chars:]
                    buf = ''
                    current_len = 0
                else:
                    buf = seg
                    current_len = len(seg)
        if buf:
            chunks.append(buf)
        if overlap_chars > 0 and len(chunks) > 1:
            out: list[str] = []
            prev = ''
            for i, ch in enumerate(chunks):
                if i == 0:
                    out.append(ch)
                    prev = ch
                    continue
                tail = prev[-overlap_chars:] if prev else ''
                out.append(f"{tail}{ch}" if tail else ch)
                prev = ch
            return out
        return chunks

    def chunk(self, text: str, max_size: int, overlap: int = 0, **options) -> list[str]:
        if not self.validate_parameters(text, max_size, overlap):
            return []
        try:
            blocks = self._collect_blocks(text)
            lines_ke = text.splitlines(keepends=True)
            chunks = self._pack_blocks(blocks, lines_ke, max_chars=max_size, overlap_chars=overlap)
            logger.info(f"PythonASTCodeChunkingStrategy produced {len(chunks)} chunks")
            return chunks
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            logger.warning(f"AST code chunking failed, returning whole text: {e}")
            return [text]

    def chunk_with_metadata(self, text: str, max_size: int, overlap: int = 0, **options) -> list[ChunkResult]:
        if not self.validate_parameters(text, max_size, overlap):
            return []
        lines_ke = text.splitlines(keepends=True)
        total_chars = sum(len(ln) for ln in lines_ke)
        line_starts = []
        pos = 0
        for ln in lines_ke:
            line_starts.append(pos)
            pos += len(ln)

        results: list[ChunkResult] = []
        try:
            blocks = self._collect_blocks(text)
            buf_s = None
            buf_e = None
            buf_blocks: list[_Block] = []

            def flush():
                nonlocal buf_s, buf_e, buf_blocks
                if buf_s is None or buf_e is None:
                    return
                end_char = buf_e
                try:
                    end_char = self._expand_end_to_grapheme_boundary(text, end_char)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    end_char = buf_e
                ch_text = text[buf_s:end_char]
                md = ChunkMetadata(
                    index=len(results),
                    start_char=buf_s,
                    end_char=end_char,
                    word_count=len(ch_text.split()),
                    language='python',
                    method='code',
                    options={
                        'blocks': [
                            {'type': b.kind, 'name': b.name, 'start_line': b.start_line + 1, 'end_line': b.end_line}
                            for b in buf_blocks
                        ],
                        'mode': 'ast',
                    },
                )
                results.append(ChunkResult(text=ch_text, metadata=md))
                buf_s = None
                buf_e = None
                buf_blocks = []

            for b in blocks:
                s_char, e_char = self._span_chars(line_starts, total_chars, b.start_line, b.end_line)
                if buf_s is None:
                    if (e_char - s_char) <= max_size:
                        buf_s, buf_e = s_char, e_char
                        buf_blocks = [b]
                    else:
                        # Split oversized block
                        start = s_char
                        while start < e_char:
                            end = min(e_char, start + max_size)
                            try:
                                end_expanded = self._expand_end_to_grapheme_boundary(text, end)
                            except (AttributeError, RuntimeError, TypeError, ValueError):
                                end_expanded = end
                            piece = text[start:end_expanded]
                            md = ChunkMetadata(
                                index=len(results),
                                start_char=start,
                                end_char=end_expanded,
                                word_count=len(piece.split()),
                                language='python',
                                method='code',
                                options={'blocks': [{'type': b.kind, 'name': b.name}], 'partial_block': True, 'mode': 'ast'}
                            )
                            results.append(ChunkResult(text=piece, metadata=md))
                            if end_expanded >= e_char:
                                break
                            # Continue without intra-block overlap; prefixing tails handles overlap.
                            start = start + max_size
                else:
                    new_len = e_char - buf_s
                    if new_len <= max_size:
                        buf_e = e_char
                        buf_blocks.append(b)
                    else:
                        flush()
                        if (e_char - s_char) <= max_size:
                            buf_s, buf_e = s_char, e_char
                            buf_blocks = [b]
                        else:
                            start = s_char
                            while start < e_char:
                                end = min(e_char, start + max_size)
                                try:
                                    end_expanded = self._expand_end_to_grapheme_boundary(text, end)
                                except (AttributeError, RuntimeError, TypeError, ValueError):
                                    end_expanded = end
                                piece = text[start:end_expanded]
                                md = ChunkMetadata(
                                    index=len(results),
                                    start_char=start,
                                    end_char=end_expanded,
                                    word_count=len(piece.split()),
                                    language='python',
                                    method='code',
                                    options={'blocks': [{'type': b.kind, 'name': b.name}], 'partial_block': True, 'mode': 'ast'}
                                )
                                results.append(ChunkResult(text=piece, metadata=md))
                                if end_expanded >= e_char:
                                    break
                                # Continue without intra-block overlap; prefixing tails handles overlap.
                                start = start + max_size

            flush()

            # Apply prefix-tail overlap so chunks can grow beyond max_size.
            if overlap > 0 and results:
                prev = results[0]
                for cur in results[1:]:
                    try:
                        prev_start = int(prev.metadata.start_char)
                        prev_end = int(prev.metadata.end_char)
                        cur_start = int(cur.metadata.start_char)
                        cur_end = int(cur.metadata.end_char)
                    except (AttributeError, TypeError, ValueError):
                        prev = cur
                        continue
                    prev_len = max(0, prev_end - prev_start)
                    tail_len = min(overlap, prev_len)
                    if tail_len <= 0:
                        prev = cur
                        continue
                    new_start = max(0, prev_end - tail_len)
                    if new_start < cur_start and cur_end > new_start:
                        cur.metadata.start_char = new_start
                        cur.metadata.char_count = cur_end - new_start
                        cur.metadata.overlap_with_previous = tail_len
                        prev.metadata.overlap_with_next = tail_len
                        cur.text = text[new_start:cur_end]
                        cur.metadata.word_count = len(cur.text.split()) if cur.text else 0
                    prev = cur

            logger.info(f"PythonASTCodeChunkingStrategy produced {len(results)} chunks with metadata")
            return results
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            logger.warning(f"AST code chunking with metadata failed: {e}")
            # fallback: single chunk
            md = ChunkMetadata(
                index=0,
                start_char=0,
                end_char=len(text),
                word_count=len(text.split()),
                language='python',
                method='code',
                options={'mode': 'ast', 'fallback': True}
            )
            return [ChunkResult(text=text, metadata=md)]
