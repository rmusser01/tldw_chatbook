"""
World Info/Lorebook processor for character chat.
Handles keyword matching and injection of world info entries into conversations.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger


class WorldInfoProcessor:
    """Process and inject world info/lorebook entries into chat conversations."""
    
    def __init__(self, character_data: Optional[Dict[str, Any]] = None, 
                 world_books: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize with character data and/or standalone world books.
        
        Args:
            character_data: Character data containing embedded world info
            world_books: List of standalone world books with their entries
        """
        self.entries = []
        self.scan_depth = 3  # Default scan depth
        self.token_budget = 500  # Default token budget
        self.recursive_scanning = False

        # Parallel diagnostics candidate list: ALL entries (incl. disabled), each
        # tagged with source-book + id + enabled metadata. Built alongside
        # self.entries but never consumed by the plain process_messages() path.
        self._candidate_entries: List[Dict[str, Any]] = []

        # Process character book if available
        if character_data:
            self.character_book = self._extract_character_book(character_data)
            if self.character_book:
                self._process_character_book()
        
        # Process standalone world books if available
        if world_books:
            self._process_world_books(world_books)
    
    def _extract_character_book(self, character_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract character book from character data."""
        if not character_data:
            return None
            
        # Check extensions field for character_book
        extensions = character_data.get('extensions', {})
        if isinstance(extensions, dict):
            return extensions.get('character_book')
        
        return None
    
    def _process_character_book(self):
        """Process the character book and prepare entries for use."""
        if not self.character_book:
            return
            
        # Extract settings
        self.scan_depth = self.character_book.get('scan_depth', 3)
        self.token_budget = self.character_book.get('token_budget', 500)
        self.recursive_scanning = self.character_book.get('recursive_scanning', False)
        
        # Extract and prepare entries
        raw_entries = self.character_book.get('entries', [])
        character_book_name = self.character_book.get('name', 'Character book')
        for entry in raw_entries:
            if entry.get('enabled', True):
                processed_entry = self._process_entry(entry)
                if processed_entry:
                    self.entries.append(processed_entry)

            # Diagnostics: candidate list carries EVERY entry (enabled or not).
            # Does not affect self.entries / the plain path.
            candidate = self._make_candidate(entry, None, character_book_name)
            if candidate is not None:
                self._candidate_entries.append(candidate)

        # Sort by insertion order
        self.entries.sort(key=lambda x: x.get('insertion_order', 0))
        self._candidate_entries.sort(key=lambda x: x.get('insertion_order', 0))

        logger.debug(f"Loaded {len(self.entries)} active world info entries from character book")
    
    def _process_world_books(self, world_books: List[Dict[str, Any]]):
        """Process standalone world books and add their entries."""
        for book in world_books:
            if not book.get('enabled', True):
                continue
                
            # Update settings if book has higher values
            self.scan_depth = max(self.scan_depth, book.get('scan_depth', 3))
            self.token_budget = max(self.token_budget, book.get('token_budget', 500))
            self.recursive_scanning = self.recursive_scanning or book.get('recursive_scanning', False)
            
            # Process entries from this book
            book_entries = book.get('entries', [])
            priority_offset = book.get('priority', 0) * 1000  # Use priority to offset insertion order
            book_id = book.get('id')
            book_name = book.get('name', '')

            for entry in book_entries:
                if entry.get('enabled', True):
                    processed_entry = self._process_entry(entry)
                    if processed_entry:
                        # Adjust insertion order based on book priority
                        processed_entry['insertion_order'] += priority_offset
                        self.entries.append(processed_entry)

                # Diagnostics: candidate list carries EVERY entry (enabled or not).
                # Does not affect self.entries / the plain path.
                candidate = self._make_candidate(entry, book_id, book_name, priority_offset)
                if candidate is not None:
                    self._candidate_entries.append(candidate)

        # Re-sort all entries by insertion order
        self.entries.sort(key=lambda x: x.get('insertion_order', 0))
        self._candidate_entries.sort(key=lambda x: x.get('insertion_order', 0))

        logger.debug(f"Total {len(self.entries)} active world info entries after processing {len(world_books)} world books")
    
    def _process_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single world info entry."""
        # Extract keys (keywords)
        keys = entry.get('keys', [])
        if isinstance(keys, str):
            keys = [keys]
        elif not isinstance(keys, list):
            keys = []
        
        # Filter empty keys
        keys = [k.strip() for k in keys if k and k.strip()]
        
        if not keys:
            return None
        
        # Process secondary keys if selective is enabled
        secondary_keys = []
        if entry.get('selective', False):
            sec_keys = entry.get('secondary_keys', [])
            if isinstance(sec_keys, str):
                secondary_keys = [sec_keys]
            elif isinstance(sec_keys, list):
                secondary_keys = sec_keys
            secondary_keys = [k.strip() for k in secondary_keys if k and k.strip()]
        
        return {
            'keys': keys,
            'secondary_keys': secondary_keys,
            'content': entry.get('content', ''),
            'selective': entry.get('selective', False),
            'position': entry.get('position', 'before_char'),
            'insertion_order': entry.get('insertion_order', 0),
            'case_sensitive': entry.get('case_sensitive', False),
            'extensions': entry.get('extensions', {})
        }

    def _make_candidate(self, entry: Dict[str, Any], book_id: Optional[Any], book_name: str,
                         priority_offset: int = 0) -> Optional[Dict[str, Any]]:
        """Build a diagnostics candidate for ANY entry (enabled or not), carrying
        source-book + id + enabled metadata. Does NOT affect self.entries / the
        plain path. Returns None only if the entry has no usable keys."""
        processed = self._process_entry(entry)
        if processed is None:
            return None
        processed = dict(processed)
        processed['insertion_order'] = processed.get('insertion_order', 0) + priority_offset
        processed['_entry_id'] = entry.get('id')
        processed['_book_id'] = book_id
        processed['_book_name'] = book_name
        processed['_enabled'] = bool(entry.get('enabled', True))
        return processed

    def process_messages(self, current_message: str, conversation_history: List[Dict[str, str]],
                        scan_depth: Optional[int] = None, apply_token_budget: bool = True) -> Dict[str, Any]:
        """
        Process messages and find matching world info entries.
        
        Args:
            current_message: The current message being sent
            conversation_history: List of previous messages
            scan_depth: Override for scan depth (uses default if None)
            apply_token_budget: Whether to apply token budget limits
            
        Returns:
            Dict containing matched entries organized by position
        """
        if not self.entries:
            return {'injections': {}, 'matched_entries': [], 'tokens_used': 0}
        
        # Determine scan depth
        depth = scan_depth if scan_depth is not None else self.scan_depth
        
        # Build text to scan
        scan_text = self._build_scan_text(current_message, conversation_history, depth)
        
        # Find matching entries
        matched_entries = self._find_matching_entries(scan_text)
        
        # Apply token budget if enabled
        if apply_token_budget and self.token_budget > 0:
            matched_entries = self._apply_token_budget(matched_entries)
        
        # Organize by position
        injections = self._organize_by_position(matched_entries)
        
        # Calculate total tokens used (simple approximation)
        tokens_used = self._estimate_tokens(matched_entries)
        
        return {
            'injections': injections,
            'matched_entries': matched_entries,
            'tokens_used': tokens_used
        }
    
    def process_messages_with_diagnostics(self, current_message: str, conversation_history: List[Dict[str, str]],
                                          scan_depth: Optional[int] = None, apply_token_budget: bool = True
                                          ) -> Tuple[Dict[str, Any], "WorldBookScanDiagnostics"]:
        """Instrumented sibling of ``process_messages`` for the Try-it panel.

        ``result`` is byte-identical to ``process_messages`` (it IS the plain
        path's return value — the fired-set pin), and ``diagnostics`` classifies
        every candidate as fired / skipped:disabled / skipped:secondary /
        skipped:budget. Never raises on a bad entry.

        Args:
            current_message: The message being previewed/sent.
            conversation_history: Prior turns (oldest-first) scanned up to
                ``scan_depth``.
            scan_depth: Override for the effective scan depth; ``None`` uses the
                processor's own ``scan_depth``.
            apply_token_budget: Whether the plain path enforces the token budget.

        Returns:
            A ``(result, diagnostics)`` tuple where ``result`` is the plain
            ``process_messages`` dict (``injections`` / ``matched_entries`` /
            ``tokens_used``) and ``diagnostics`` is a ``WorldBookScanDiagnostics``
            whose ``fired`` set equals ``result["matched_entries"]``.
        """
        from .world_info_diagnostics import WorldBookScanDiagnostics, WorldBookEntryDiagnostic

        # Authoritative result — literally the plain path (guarantees agreement).
        result = self.process_messages(current_message, conversation_history,
                                       scan_depth, apply_token_budget)
        fired_list = result.get("matched_entries", [])

        depth = scan_depth if scan_depth is not None else self.scan_depth
        scan_text = self._build_scan_text(current_message, conversation_history, depth)
        scan_text_lower = scan_text.lower()

        # Identify entries by insertion_order + content + position (the stable
        # non-meta fields shared between self.entries/matched_entries and
        # self._candidate_entries — matched_entries carry no _entry_id).
        # Two distinct entries that share an identical (insertion_order,
        # content, position) triple collide on this signature (a degenerate
        # authoring case). Keep ALL candidates per signature in a list and
        # consume one per fired occurrence, so multiplicity and per-occurrence
        # attribution (entry_id / source book) are preserved and any leftover
        # candidates fall through to the near-miss pass.
        def sig(e):
            return (e.get("insertion_order", 0), e.get("content", ""), e.get("position", "before_char"))

        candidates_by_sig: Dict[Any, list] = {}
        for cand in self._candidate_entries:
            candidates_by_sig.setdefault(sig(cand), []).append(cand)

        records = []

        # Fired records are derived from the AUTHORITATIVE send result
        # (result["matched_entries"]), never from re-matching the original
        # scan text: when recursive_scanning is on, an entry can legitimately
        # fire because its key only appears inside ANOTHER entry's content,
        # not in the original message/history. Re-classifying against the
        # original text (as before) silently dropped those entries from the
        # diagnostics, under-reporting `fired` vs. the real send.
        for order, entry in enumerate(fired_list):
            bucket = candidates_by_sig.get(sig(entry))
            cand = bucket.pop(0) if bucket else entry
            try:
                primary_hit, pk, sec_req, sec_hit, sk = self._classify_entry_match(
                    cand, scan_text, scan_text_lower)
            except Exception:
                primary_hit, pk, sk = False, None, None

            if primary_hit:
                depth_level = 0
                reason = f"matched key '{pk}'" + (f" + secondary '{sk}'" if sk else "")
            else:
                # Fired, but its own key never hit the original scan text →
                # it only fired via recursive_scanning's rescan of other
                # matched entries' content.
                depth_level = 1
                reason = "matched in another entry's content (recursive)"

            records.append(WorldBookEntryDiagnostic(
                entry_id=cand.get("_entry_id"), source_book_id=cand.get("_book_id"),
                source_book_name=str(cand.get("_book_name") or ""),
                keys=list(cand.get("keys") or []), activation_reason=reason, status="fired",
                token_cost=self._estimate_entry_tokens(cand), injection_order=order,
                position=cand.get("position", "before_char"),
                content_preview=(cand.get("content", "") or "")[:80], depth_level=depth_level,
            ))

        # Near-misses: every candidate NOT consumed by the fired loop above
        # (whatever remains in the per-signature buckets), classified against
        # the original scan text.
        for cand in [c for bucket in candidates_by_sig.values() for c in bucket]:
            try:
                primary_hit, pk, sec_req, sec_hit, sk = self._classify_entry_match(
                    cand, scan_text, scan_text_lower)
            except Exception:
                continue
            if not primary_hit:
                continue  # key never appeared → neither fired nor a near-miss

            if cand.get("_enabled", True) is False:
                status, reason = "skipped:disabled", f"disabled (key '{pk}' matched)"
            elif sec_req and not sec_hit:
                status, reason = "skipped:secondary", "secondary key not found"
            else:
                # Matched originally, enabled, secondary-ok — but not part of
                # the fired set, so it was dropped by the budget hard-break.
                status, reason = "skipped:budget", "dropped by token budget"

            records.append(WorldBookEntryDiagnostic(
                entry_id=cand.get("_entry_id"), source_book_id=cand.get("_book_id"),
                source_book_name=str(cand.get("_book_name") or ""),
                keys=list(cand.get("keys") or []), activation_reason=reason, status=status,
                token_cost=self._estimate_entry_tokens(cand), injection_order=None,
                position=cand.get("position", "before_char"),
                content_preview=(cand.get("content", "") or "")[:80], depth_level=0,
            ))

        fired_count = len(fired_list)
        diagnostics = WorldBookScanDiagnostics(
            entries=records, matched=len(records), fired=fired_count,
            skipped=len(records) - fired_count, tokens_used=result.get("tokens_used", 0),
            token_budget=self.token_budget,
            # Truncation-derived (mirrors the dictionary diagnostics): true when the
            # budget dropped at least one matched entry. The fired set's tokens_used
            # is always <= budget by construction, so don't compare against it.
            budget_exceeded=any(r.status == "skipped:budget" for r in records),
            # Distinct source books, INCLUDING character-embedded books (whose
            # _book_id is None) — id alone would collapse every character book
            # into a single None bucket, so key on (id, name).
            books_scanned=len({(c.get("_book_id"), c.get("_book_name")) for c in self._candidate_entries}),
        )
        return result, diagnostics

    def _build_scan_text(self, current_message: str, conversation_history: List[Dict[str, str]],
                        depth: int) -> str:
        """Build the text to scan for keywords."""
        texts = [current_message]
        
        # Add recent messages from history
        for i, msg in enumerate(reversed(conversation_history)):
            if i >= depth:
                break
            content = msg.get('content', '')
            if content:
                texts.append(content)
        
        # Join with newlines to preserve structure
        return '\n'.join(reversed(texts))
    
    def _find_matching_entries(self, scan_text: str, _recursion_depth: int = 0) -> List[Dict[str, Any]]:
        """Find all world info entries that match the scan text."""
        matched = []
        scan_text_lower = scan_text.lower()
        
        for entry in self.entries:
            if self._entry_matches(entry, scan_text, scan_text_lower):
                matched.append(entry)
        
        # Handle recursive scanning if enabled with depth limit
        if self.recursive_scanning and matched and _recursion_depth < 3:
            # Check if any matched entries should trigger additional entries
            additional_text = '\n'.join([e['content'] for e in matched])
            additional_matches = self._find_matching_entries(additional_text, _recursion_depth + 1)
            
            # Add new matches that aren't already in the list
            for match in additional_matches:
                if match not in matched:
                    matched.append(match)
        
        return matched
    
    def _entry_matches(self, entry: Dict[str, Any], scan_text: str, scan_text_lower: str) -> bool:
        """Check if an entry matches the scan text."""
        # Check primary keys
        primary_match = False
        for key in entry['keys']:
            if entry.get('case_sensitive', False):
                if self._keyword_in_text(key, scan_text):
                    primary_match = True
                    break
            else:
                if self._keyword_in_text(key.lower(), scan_text_lower):
                    primary_match = True
                    break
        
        if not primary_match:
            return False
        
        # If not selective, primary match is enough
        if not entry.get('selective', False):
            return True
        
        # Check secondary keys for selective entries
        if not entry['secondary_keys']:
            return True  # No secondary keys means primary match is enough
        
        for key in entry['secondary_keys']:
            if entry.get('case_sensitive', False):
                if self._keyword_in_text(key, scan_text):
                    return True
            else:
                if self._keyword_in_text(key.lower(), scan_text_lower):
                    return True
        
        return False

    def _classify_entry_match(self, entry: Dict[str, Any], scan_text: str,
                               scan_text_lower: str) -> Tuple[bool, Optional[str], bool, bool, Optional[str]]:
        """Decompose an entry's match for diagnostics. Returns
        (primary_hit, primary_key, secondary_required, secondary_hit, secondary_key).
        Mirrors _entry_matches' logic exactly but reports WHICH key matched / why not."""
        case = entry.get('case_sensitive', False)

        def hit(key):
            return (self._keyword_in_text(key, scan_text) if case
                    else self._keyword_in_text(key.lower(), scan_text_lower))

        primary_key = next((k for k in (entry.get('keys') or []) if hit(k)), None)
        primary_hit = primary_key is not None
        if not primary_hit:
            return (False, None, False, False, None)
        secondary_required = bool(entry.get('selective', False) and entry.get('secondary_keys'))
        if not secondary_required:
            return (True, primary_key, False, False, None)
        secondary_key = next((k for k in (entry.get('secondary_keys') or []) if hit(k)), None)
        return (True, primary_key, True, secondary_key is not None, secondary_key)

    def _keyword_in_text(self, keyword: str, text: str) -> bool:
        """Check if a keyword appears in text with word boundary matching."""
        # Use word boundary regex for more accurate matching
        pattern = r'\b' + re.escape(keyword) + r'\b'
        return bool(re.search(pattern, text))
    
    def _organize_by_position(self, entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Organize matched entries by their injection position."""
        positions = {
            'before_char': [],
            'after_char': [],
            'at_start': [],
            'at_end': []
        }
        
        for entry in entries:
            position = entry.get('position', 'before_char')
            if position not in positions:
                position = 'before_char'  # Default position
            
            content = entry.get('content', '').strip()
            if content:
                positions[position].append(content)
        
        return positions
    
    def format_injections(self, injections: Dict[str, List[str]]) -> Dict[str, str]:
        """Format injection lists into strings for insertion."""
        formatted = {}
        
        # Ensure all positions are represented, even if empty
        all_positions = ['before_char', 'after_char', 'at_start', 'at_end']
        
        for position in all_positions:
            contents = injections.get(position, [])
            if contents:
                # Join with double newlines for clear separation
                formatted[position] = '\n\n'.join(contents)
        
        return formatted
    
    def _apply_token_budget(self, matched_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply token budget limits to matched entries."""
        if not matched_entries or self.token_budget <= 0:
            return matched_entries
        
        # Sort by insertion order (already done in initialization)
        # Estimate tokens for each entry
        selected_entries = []
        total_tokens = 0
        
        for entry in matched_entries:
            entry_tokens = self._estimate_entry_tokens(entry)
            if total_tokens + entry_tokens <= self.token_budget:
                selected_entries.append(entry)
                total_tokens += entry_tokens
            else:
                logger.debug(f"Token budget exceeded. Skipping entry: {entry.get('content', '')[:50]}...")
                break
        
        logger.debug(f"Token budget: {total_tokens}/{self.token_budget} tokens used for {len(selected_entries)} entries")
        return selected_entries
    
    def _estimate_tokens(self, entries: List[Dict[str, Any]]) -> int:
        """Estimate total tokens for all entries."""
        return sum(self._estimate_entry_tokens(entry) for entry in entries)
    
    def _estimate_entry_tokens(self, entry: Dict[str, Any]) -> int:
        """
        Estimate tokens for a single entry.
        Simple approximation: 1 token ≈ 4 characters
        """
        content = entry.get('content', '')
        # Rough estimate: 1 token per 4 characters
        return max(1, len(content) // 4)