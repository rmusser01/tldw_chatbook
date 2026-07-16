# Chat_Dictionary_Lib.py
# Description: Library for managing Chat Dictionaries - keyword/pattern-based text replacement system
#
# Imports
import json
import logging
import os
import random
import re
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from loguru import logger

# Local Imports
from ..Utils.input_validation import validate_text_input
from ..Utils.path_validation import validate_path
from ..DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, InputError, ConflictError
from ..Metrics.metrics import log_counter, log_histogram, log_gauge

#
#######################################################################################################################
#
# Chat Dictionary Classes and Functions

def _clean_dictionary_text(
    value: Optional[str],
    *,
    field_name: str,
    max_length: int,
    required: bool = False,
) -> str:
    text = "" if value is None else str(value).strip()
    if required and not text:
        raise InputError(f"{field_name} cannot be empty")
    if not validate_text_input(text, max_length=max_length):
        raise InputError(f"{field_name} is invalid or too long")
    return text


def _ensure_chat_dictionaries_fts_update_trigger(conn) -> None:
    conn.execute("DROP TRIGGER IF EXISTS chat_dictionaries_au")
    conn.execute(
        """
        CREATE TRIGGER chat_dictionaries_au
        AFTER UPDATE ON chat_dictionaries BEGIN
          INSERT INTO chat_dictionaries_fts(chat_dictionaries_fts, rowid, name, description, content)
          SELECT 'delete', OLD.id, OLD.name, OLD.description, OLD.content
          WHERE OLD.deleted = 0;

          INSERT INTO chat_dictionaries_fts(rowid, name, description, content)
          SELECT NEW.id, NEW.name, NEW.description, NEW.content
          WHERE NEW.deleted = 0;
        END;
        """
    )

class TokenBudgetExceededWarning(Warning):
    """Custom warning for token budget issues"""
    pass


def _coerce_int(value: Any, default: int = 0) -> int:
    """Best-effort int coercion for loosely-typed entry fields.

    Args:
        value: Raw value from a payload or persisted JSON.
        default: Fallback when the value is missing or malformed.

    Returns:
        The coerced int, or ``default`` on None/non-numeric input.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_FALSY_STRINGS = {"false", "0", "no", "off", ""}
_TRUTHY_STRINGS = {"true", "1", "yes", "on"}


def _coerce_bool(value: Any, default: bool) -> bool:
    """Best-effort bool coercion that treats quoted booleans honestly.

    Args:
        value: Raw value from a payload or persisted JSON.
        default: Fallback for None or unrecognized strings.

    Returns:
        ``value`` itself for real bools; a case-insensitive allowlist parse
        for strings ("false"/"0"/"no"/"off" are False); ``default`` for None
        or unrecognized strings; ``bool(value)`` otherwise.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUTHY_STRINGS:
            return True
        if lowered in _FALSY_STRINGS:
            return False
        return default
    return bool(value)


class ChatDictionary:
    def __init__(self, key: str, content: str, probability: int = 100, group: Optional[str] = None,
                 timed_effects: Optional[Dict[str, int]] = None, max_replacements: int = 1, enabled: bool = True,
                 case_sensitive: bool = False, priority: int = 0):
        """Initialize a single chat-dictionary entry.

        Args:
            key: The raw match key. A ``/pattern/flags`` form is compiled as
                a regex; any other string is treated as a literal match key.
            content: The replacement text substituted in for a match.
            probability: Percent chance (0-100) that a match actually fires.
            group: Optional group name used for mutually-exclusive
                group-scoring between entries.
            timed_effects: Optional ``{"sticky": int, "cooldown": int,
                "delay": int}`` mapping; defaults to all-zero effects.
            max_replacements: Maximum number of replacements to apply per
                invocation.
            enabled: Whether the entry participates in matching. Loosely
                typed values (e.g. ``"false"``) are coerced honestly instead
                of via truthy-string ``bool()``.
            case_sensitive: Whether literal-key matching is case sensitive.
                Ignored for regex keys, whose case handling comes from the
                pattern's own flags. Loosely typed values are coerced the
                same way as ``enabled``.
            priority: Tie-breaker used for group scoring, token-budget
                survival, and application order (higher wins). Malformed
                values fall back to ``0`` instead of raising.
        """
        self.raw_key = key # Store the original key string
        self.content = content
        self.is_regex = False
        self.key_pattern_str = "" # Store pattern string for regex for debugging
        self.key_flags = 0      # Store flags for regex for debugging
        self.key = self._compile_key_internal(key) # key will store re.Pattern or str

        self.probability = probability
        self.group = group
        self.timed_effects = timed_effects or {"sticky": 0, "cooldown": 0, "delay": 0}
        self.last_triggered: Optional[datetime] = None
        self.max_replacements = max_replacements
        self.enabled = _coerce_bool(enabled, True)
        self.case_sensitive = _coerce_bool(case_sensitive, False)
        self.priority = _coerce_int(priority, 0)

    def _compile_key_internal(self, key_str: str) -> Union[re.Pattern, str]:
        self.is_regex = False # Reset for this compilation
        self.key_flags = 0
        pattern_to_compile = key_str

        # Check for /pattern/flags format
        # Regex to capture pattern and flags: r^/(.+)/([ismx]*)$
        # Using string methods for simplicity here:
        if key_str.startswith("/") and len(key_str) > 1:
            last_slash_idx = key_str.rfind("/")
            if last_slash_idx > 0: # Found a second slash, potential flags
                pattern_to_compile = key_str[1:last_slash_idx]
                flag_chars = key_str[last_slash_idx+1:]
                if 'i' in flag_chars: self.key_flags |= re.IGNORECASE
                if 'm' in flag_chars: self.key_flags |= re.MULTILINE
                if 's' in flag_chars: self.key_flags |= re.DOTALL
                # Add other common flags if needed (e.g., 'x' for VERBOSE, 'u' for UNICODE automatically on in Py3)
                self.is_regex = True
            elif key_str.endswith("/") and len(key_str) > 2: # Only /pattern/, no flags after last /
                pattern_to_compile = key_str[1:-1]
                self.is_regex = True
            # else: it's like "/foo" or just "/" which are not valid regex delimiters here

        self.key_pattern_str = pattern_to_compile # Store for debugging

        if self.is_regex:
            try:
                # If pattern_to_compile is empty after stripping slashes (e.g. "//i"), it's an error
                if not pattern_to_compile:
                    logging.warning(f"Empty regex pattern from raw key '{self.raw_key}'. Treating as literal.")
                    self.is_regex = False
                    return self.raw_key
                return re.compile(pattern_to_compile, self.key_flags)
            except re.error as e:
                logging.warning(
                    f"Invalid regex '{pattern_to_compile}' with flags '{self.key_flags}' (from raw key '{self.raw_key}'): {e}. "
                    f"Treating as literal string."
                )
                self.is_regex = False # Fallback
                return self.raw_key # Return the original key string on error
        else: # Not a /regex/ or /regex/flags pattern, treat as plain string
            return key_str # Return the original string

    def matches(self, text: str) -> bool:
        if self.is_regex and isinstance(self.key, re.Pattern):
            return bool(self.key.search(text))
        elif not self.is_regex and isinstance(self.key, str):
            # For plain string, if you want case-insensitivity by default:
            # return self.key.lower() in text.lower()
            return self.key in text # Current: case-sensitive plain match
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert ChatDictionary instance to a dictionary for database storage.

        Returns:
            A dict with keys ``key``, ``content``, ``probability``, ``group``,
            ``timed_effects``, ``max_replacements``, ``is_regex``, ``enabled``,
            ``case_sensitive``, and ``priority``.
        """
        return {
            'key': self.raw_key,
            'content': self.content,
            'probability': self.probability,
            'group': self.group,
            'timed_effects': self.timed_effects,
            'max_replacements': self.max_replacements,
            'is_regex': self.is_regex,
            'enabled': self.enabled,
            'case_sensitive': self.case_sensitive,
            'priority': self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatDictionary':
        """Create ChatDictionary instance from dictionary data.

        Args:
            data: A mapping as produced by :meth:`to_dict` (or a legacy
                stored dict missing the newer keys). ``key`` and ``content``
                are required; all other keys are optional and fall back to
                the same defaults as :meth:`__init__`.

        Returns:
            A new :class:`ChatDictionary` built from ``data``.
        """
        return cls(
            key=data['key'],
            content=data['content'],
            probability=data.get('probability', 100),
            group=data.get('group'),
            timed_effects=data.get('timed_effects', {"sticky": 0, "cooldown": 0, "delay": 0}),
            max_replacements=data.get('max_replacements', 1),
            enabled=data.get('enabled', True),
            case_sensitive=data.get('case_sensitive', False),
            priority=data.get('priority', 0),
        )


def parse_user_dict_markdown_file(file_path: str, base_directory: Optional[str] = None) -> Dict[str, str]:
    """
    Parses a user-defined dictionary from a markdown-like file.

    The file format supports:
    - Single-line entries: `key: value`
    - Multi-line entries:
      ```
      key: |
      This is a
      multi-line value.
      ---@@@---
      ```
    Keys and single-line values are stripped of leading/trailing whitespace.
    Multi-line values preserve internal whitespace and newlines until the
    terminator `---@@@---` is encountered on its own line (stripped).
    Lines starting with a key pattern override previous multi-line contexts.

    Args:
        file_path: The path to the markdown dictionary file.
        base_directory: Optional base directory to restrict file access to. If None, 
                       uses the config directory or current working directory.

    Returns:
        A dictionary where keys are strings and values are the corresponding
        content strings. Returns an empty dictionary if the file is not found
        or an error occurs during parsing.
    """
    logger.debug(f"Parsing user dictionary file: {file_path}")
    
    # Validate the file path to prevent directory traversal
    if base_directory is None:
        # Default to a safe base directory - typically config or user data directory
        base_directory = os.path.expanduser("~/.config/tldw_cli/")
    
    try:
        validated_path = validate_path(file_path, base_directory)
        logger.debug(f"Validated file path: {validated_path}")
    except ValueError as e:
        logger.error(f"Invalid file path '{file_path}': {e}")
        return {}
    
    replacement_dict: Dict[str, str] = {}
    current_key: Optional[str] = None
    current_value_lines: List[str] = []

    new_key_pattern = re.compile(r'^\s*([^:\n]+?)\s*:(.*)$')
    termination_pattern = re.compile(r'^\s*---@@@---\s*$')

    try:
        with open(validated_path, 'r', encoding='utf-8') as file:
            for line_number, line_content_original in enumerate(file, 1):
                line_for_logic = line_content_original.strip()  # Use for terminator/blank checks

                if termination_pattern.match(line_for_logic):
                    if current_key:
                        replacement_dict[current_key] = '\n'.join(current_value_lines).strip()  # Final strip after join
                        logger.trace(f"L{line_number}: Terminated multi-line for '{current_key}'.")
                        current_key, current_value_lines = None, []
                    continue

                new_key_match = new_key_pattern.match(line_content_original)  # Match on original line

                if new_key_match:
                    if current_key:  # Finalize previous multi-line key if one was active
                        replacement_dict[current_key] = '\n'.join(current_value_lines).strip()
                        logger.trace(f"L{line_number}: New key, finalized old '{current_key}'.")

                    potential_new_key = new_key_match.group(1).strip()
                    value_part_after_colon = new_key_match.group(2).strip()  # Strip this part

                    if value_part_after_colon == '|':
                        current_key = potential_new_key
                        current_value_lines = []
                        logger.trace(f"L{line_number}: Starting multi-line for '{current_key}'.")
                    else:
                        replacement_dict[potential_new_key] = value_part_after_colon
                        logger.trace(f"L{line_number}: Parsed single-line key '{potential_new_key}'.")
                        current_key, current_value_lines = None, []  # Reset
                    continue

                if current_key:
                    # For multi-line content, append the line with only its trailing newline removed.
                    # Leading/internal whitespace should be preserved until the final .strip() after .join().
                    current_value_lines.append(line_content_original.rstrip('\n\r'))

            if current_key:  # Finalize any pending multi-line value at EOF
                replacement_dict[current_key] = '\n'.join(current_value_lines).strip()
                logger.debug(f"Finalizing last multi-line key '{current_key}' at EOF.")

    except FileNotFoundError:  # ...
        logger.error(f"Chat dictionary file not found: {file_path}")
        return {}
    except Exception as e:  # ...
        logger.opt(exception=True).error(f"Error parsing chat dictionary file {file_path}: {e}")
        return {}

    logger.debug(f"Finished parsing chat dictionary. Keys: {list(replacement_dict.keys())}")
    return replacement_dict


def apply_strategy(entries: List[ChatDictionary], strategy: str = "sorted_evenly") -> List[ChatDictionary]:
    """
    Sorts chat dictionary entries based on a given strategy.

    Strategies:
    - "sorted_evenly": Sorts entries alphabetically by their raw key.
    - "character_lore_first": Sorts "character" group entries first, then others, then by key.
    - "global_lore_first": Sorts "global" group entries first, then others, then by key.

    Args:
        entries: A list of `ChatDictionary` objects.
        strategy: The sorting strategy name. Defaults to "sorted_evenly".

    Returns:
        A new list of sorted `ChatDictionary` objects.
    """
    logging.debug(f"Applying strategy: {strategy}")
    if strategy == "sorted_evenly":
        return sorted(entries, key=lambda e: str(e.raw_key)) # Ensure raw_key is string for sort
    elif strategy == "character_lore_first":
        return sorted(entries, key=lambda e: (e.group != "character", str(e.raw_key)))
    elif strategy == "global_lore_first":
        return sorted(entries, key=lambda e: (e.group != "global", str(e.raw_key)))
    return entries # Fallback if strategy not recognized


def filter_by_probability(entries: List[ChatDictionary]) -> List[ChatDictionary]:
    """
    Filters a list of ChatDictionary entries based on their probability.

    Each entry has a `probability` attribute (0-100). This function
    includes an entry if a random number between 1 and 100 is less than
    or equal to its probability.

    Args:
        entries: A list of `ChatDictionary` objects.

    Returns:
        A new list containing only the entries that passed the probability check.
    """
    return [entry for entry in entries if random.randint(1, 100) <= entry.probability]


# Group Scoring - Situation where multiple entries are triggered in different groups in a single message
def group_scoring(entries: List[ChatDictionary]) -> List[ChatDictionary]:
    """
    Selects entries based on group scoring rules.

    - Entries without a group (group is None) are all included if matched.
    - For entries within the same named group, only the "best" entry (currently
      defined as the highest-priority entry, with raw-key length breaking ties)
      is selected from that group.

    Args:
        entries: A list of `ChatDictionary` objects that have already matched.

    Returns:
        A new list of selected `ChatDictionary` objects after group scoring.
    """
    logging.debug(f"Group scoring for {len(entries)} entries")
    if not entries: return []

    grouped_entries: Dict[Optional[str], List[ChatDictionary]] = {}
    for entry in entries:
        grouped_entries.setdefault(entry.group, []).append(entry)

    selected_entries: List[ChatDictionary] = []
    for group_name, group_entries_list in grouped_entries.items():
        if not group_entries_list: continue

        if group_name is None:  # For the default group (None)
            # Add all entries instead of just the "best" one.
            # This allows multiple ungrouped keywords to be processed if they all match.
            selected_entries.extend(group_entries_list)
        else:
            # For named groups, keep the original behavior of selecting the best.
            best_entry_in_group = max(
                group_entries_list,
                key=lambda e: (getattr(e, "priority", 0), len(str(e.raw_key)) if e.raw_key else 0),
            )
            selected_entries.append(best_entry_in_group)

    logging.debug(f"Selected {len(selected_entries)} entries after group scoring.")
    # Ensure the order is somewhat predictable if multiple entries come from the None group
    # The apply_strategy step later will sort them.
    return selected_entries


def apply_timed_effects(entry: ChatDictionary, current_time: datetime) -> bool:
    """
    Applies timed effects (delay, cooldown) to a ChatDictionary entry.

    - Delay: If `entry.timed_effects["delay"]` is positive, the entry is
      invalid if the time since `last_triggered` (or from epoch if never triggered)
      is less than the delay.
    - Cooldown: If `entry.timed_effects["cooldown"]` is positive, the entry is
      invalid if it was `last_triggered` and the time since then is less than
      the cooldown.

    If the entry is considered valid after checks, its `last_triggered` time is
    updated to `current_time`.

    Args:
        entry: The `ChatDictionary` entry to check.
        current_time: The current `datetime` object.

    Returns:
        True if the entry is valid after timed effect checks, False otherwise.
    """
    logging.debug(f"Applying timed effects for entry: {entry.raw_key}") # Use raw_key for logging
    if entry.timed_effects["delay"] > 0:
        # If never triggered, assume it's valid for delay unless delay is from program start
        # For simplicity, if last_triggered is None, it passes delay check.
        # A more complex interpretation might involve first_seen time.
        # Current logic: delay is from last trigger. If never triggered, passes delay.
        if entry.last_triggered is not None and \
           current_time - entry.last_triggered < timedelta(seconds=entry.timed_effects["delay"]):
            logging.debug(f"Entry {entry.raw_key} delayed.")
            return False
    if entry.timed_effects["cooldown"] > 0:
        if entry.last_triggered and \
           current_time - entry.last_triggered < timedelta(seconds=entry.timed_effects["cooldown"]):
            logging.debug(f"Entry {entry.raw_key} on cooldown.")
            return False

    # If checks pass, update last_triggered (conceptually, this happens if it *would* be used)
    # The actual update of last_triggered for active use is often done after selection.
    # Here, we return true, and `process_user_input` will update `last_triggered` for used entries.
    # For this function's purpose (filtering), we don't update here but assume it would be if selected.
    return True


def calculate_token_usage(entries: List[ChatDictionary]) -> int:
    """
    Calculates the approximate total token usage for a list of ChatDictionary entries.

    Token usage for each entry is estimated by splitting its `content` by spaces.

    Args:
        entries: A list of `ChatDictionary` objects.

    Returns:
        The total approximate token count for all entries' content.
    """
    logging.debug(f"Calculating token usage for {len(entries)} entries")
    return sum(len(entry.content.split()) for entry in entries)


def enforce_token_budget(entries: List[ChatDictionary], max_tokens: int) -> List[ChatDictionary]:
    """
    Filters a list of ChatDictionary entries to fit within a maximum token budget.

    Entries are added to the returned list one by one, accumulating their
    token count, until the `max_tokens` budget is reached. Entries are processed
    in their given order.

    Args:
        entries: A list of `ChatDictionary` objects, typically already sorted by priority/strategy.
        max_tokens: The maximum allowed total tokens for the content of selected entries.

    Returns:
        A new list of `ChatDictionary` objects whose combined content token count
        does not exceed `max_tokens`.
    """
    total_tokens = 0
    valid_entries = []
    for entry in entries:
        tokens = len(entry.content.split())
        if total_tokens + tokens <= max_tokens:
            valid_entries.append(entry)
            total_tokens += tokens
        else:
            logging.debug(f"Token budget exceeded with entry {entry.raw_key}. Total tokens: {total_tokens + tokens}, Max: {max_tokens}")
            break # Stop adding entries once budget is full
    return valid_entries


def match_whole_words(entries: List[ChatDictionary], text: str) -> List[ChatDictionary]:
    """
    Filters entries by matching their keys against text, ensuring whole word matches for string keys.

    - If an entry's key is a compiled regex, `re.search()` is used.
    - If an entry's key is a plain string, it's matched as a whole word
      (using `\\b` word boundaries) case per entry.case_sensitive.

    Args:
        entries: A list of `ChatDictionary` objects.
        text: The input text to match against.

    Returns:
        A new list of `ChatDictionary` objects that matched the text.
    """
    matched_entries = []
    for entry in entries:
        if isinstance(entry.key, re.Pattern): # Compiled regex
            if entry.key.search(text):
                matched_entries.append(entry)
                logging.debug(f"Chat Dictionary: Matched regex entry: {entry.key.pattern}")
        elif isinstance(entry.key, str): # Plain string key
            # Ensure whole word match for plain strings; case per entry.case_sensitive
            flags = 0 if getattr(entry, "case_sensitive", False) else re.IGNORECASE
            if re.search(rf'\b{re.escape(entry.key)}\b', text, flags):
                matched_entries.append(entry)
                logging.debug(f"Chat Dictionary: Matched string entry: {entry.key}")
    return matched_entries


def alert_token_budget_exceeded(entries: List[ChatDictionary], max_tokens: int):
    """
    Checks if the token usage of selected entries exceeds the budget and issues a warning.

    Args:
        entries: A list of `ChatDictionary` objects selected for use.
        max_tokens: The maximum allowed token budget.
    """
    token_usage = calculate_token_usage(entries)
    logging.debug(f"Token usage: {token_usage}, Max tokens: {max_tokens}")
    if token_usage > max_tokens:
        warning_msg = f"Alert: Token budget exceeded for chat dictionary! Used: {token_usage}, Allowed: {max_tokens}"
        warnings.warn(TokenBudgetExceededWarning(warning_msg))
        logging.warning(warning_msg)


def apply_replacement_once(text: str, entry: ChatDictionary) -> Tuple[str, int]:
    """
    Replaces the first occurrence of an entry's key in text with its content.

    - If `entry.key` is a regex pattern, `re.subn()` with `count=1` is used.
    - If `entry.key` is a string, a whole-word regex is constructed respecting
      entry.case_sensitive and used with `re.subn()` with `count=1`.

    Args:
        text: The input text where replacement should occur.
        entry: The `ChatDictionary` entry providing the key and content.

    Returns:
        A tuple containing:
        - `str`: The text after the first replacement (or original text if no match).
        - `int`: The number of replacements made (0 or 1).
    """
    logging.debug(f"Applying replacement for entry: {entry.raw_key} with content: {entry.content[:50]}... in text: {text[:50]}...")
    if isinstance(entry.key, re.Pattern):
        replaced_text, replaced_count = entry.key.subn(entry.content, text, count=1)
    else: # Plain string key
        flags = 0 if getattr(entry, "case_sensitive", False) else re.IGNORECASE
        pattern = re.compile(rf'\b{re.escape(str(entry.key))}\b', flags) # Ensure entry.key is str
        replaced_text, replaced_count = pattern.subn(entry.content, text, count=1)
    return replaced_text, replaced_count


@dataclass
class DictionaryEntryDiagnostic:
    """One matched entry's outcome in the substitution pipeline.

    Args:
        input_index: Position of the entry in the caller-provided list.
        pattern: The entry's raw key (slash-delimited for regex entries).
        status: ``"fired"``, ``"skipped:<stage>"``, or ``"no_replacement"``.
        replacements: Number of replacements this entry performed.
        token_cost: Approximate token cost of the entry's content.
        applied_order: 0-based position in the post-strategy application
            sequence for entries that reached the replacement loop, else None.
        content_preview: The first 40 characters of the entry's content.
    """

    input_index: int
    pattern: str
    status: str
    replacements: int = 0
    token_cost: int = 0
    applied_order: Optional[int] = None
    content_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Returns the record as a plain JSON-safe dict."""
        return {
            "input_index": self.input_index,
            "pattern": self.pattern,
            "status": self.status,
            "replacements": self.replacements,
            "token_cost": self.token_cost,
            "applied_order": self.applied_order,
            "content_preview": self.content_preview,
        }


@dataclass
class DictionaryProcessDiagnostics:
    """Aggregate diagnostics for one substitution run.

    Totals maintain the invariant ``matched == fired + skipped`` —
    ``no_replacement`` entries count as skipped. ``tokens_used`` is
    budget-stage accounting: the summed cost of entries that survived the
    token-budget stage (including no_replacement survivors), and
    ``budget_exceeded`` is truncation-derived (at least one matched entry
    was dropped at the budget stage).
    """

    entries: List[DictionaryEntryDiagnostic] = field(default_factory=list)
    matched: int = 0
    fired: int = 0
    skipped: int = 0
    total_replacements: int = 0
    tokens_used: int = 0
    token_budget: int = 0
    budget_exceeded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Returns the diagnostics as a plain JSON-safe dict."""
        return {
            "entries": [record.to_dict() for record in self.entries],
            "matched": self.matched,
            "fired": self.fired,
            "skipped": self.skipped,
            "total_replacements": self.total_replacements,
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "budget_exceeded": self.budget_exceeded,
        }


def process_user_input_with_diagnostics(
    user_input: str,
    entries: List[ChatDictionary],
    max_tokens: int = 5000,
    strategy: str = "sorted_evenly"
) -> Tuple[str, DictionaryProcessDiagnostics]:
    """Runs the substitution pipeline and reports per-entry diagnostics.

    Identical behavior to :func:`process_user_input` (which now wraps this
    function); diagnostics are collected purely from stage-boundary diffs,
    so no pipeline stage behaves differently.

    Args:
        user_input: The text input from the user.
        entries: A list of ``ChatDictionary`` objects to apply.
        max_tokens: The maximum token budget for applied entries' content.
        strategy: The sorting strategy for entries before replacement.

    Returns:
        A ``(processed_text, diagnostics)`` tuple. On critical pipeline
        failure the original input is returned with whatever diagnostics
        were collected before the failure.
    """
    current_time = datetime.now()
    original_input_for_fallback = user_input  # Save for critical error case
    temp_user_input = user_input

    diagnostics = DictionaryProcessDiagnostics(token_budget=max_tokens)
    # First-wins for pathological duplicate objects in the input list.
    index_by_id: Dict[int, int] = {}
    matched_snapshot: List[ChatDictionary] = []
    skip_reason_by_id: Dict[int, str] = {}
    replacements_by_id: Dict[int, int] = {}
    applied_order_by_id: Dict[int, int] = {}
    budget_survivor_ids: Set[int] = set()

    def _record_stage_drops(
        before: List[ChatDictionary], after: List[ChatDictionary], stage: str
    ) -> None:
        surviving = {id(e) for e in after}
        for candidate in before:
            if id(candidate) not in surviving and id(candidate) not in skip_reason_by_id:
                skip_reason_by_id[id(candidate)] = f"skipped:{stage}"

    def _finalize() -> None:
        # Same formula as calculate_token_usage; inlined to avoid a debug-log call per entry.
        for candidate in matched_snapshot:
            entry_id = id(candidate)
            replacements = replacements_by_id.get(entry_id, 0)
            if entry_id in skip_reason_by_id:
                status = skip_reason_by_id[entry_id]
            elif replacements > 0:
                status = "fired"
            else:
                status = "no_replacement"
            diagnostics.entries.append(
                DictionaryEntryDiagnostic(
                    input_index=index_by_id.get(entry_id, -1),
                    pattern=str(candidate.raw_key),
                    status=status,
                    replacements=replacements,
                    token_cost=len(candidate.content.split()) if candidate.content else 0,
                    applied_order=applied_order_by_id.get(entry_id),
                    content_preview=" ".join(str(candidate.content or "").split())[:40],
                )
            )
        diagnostics.matched = len(matched_snapshot)
        diagnostics.fired = sum(1 for r in diagnostics.entries if r.status == "fired")
        diagnostics.skipped = diagnostics.matched - diagnostics.fired
        diagnostics.total_replacements = sum(r.replacements for r in diagnostics.entries)
        # Budget-stage accounting: survivors of the budget stage, including
        # no_replacement survivors (they consumed budget without firing).
        diagnostics.tokens_used = calculate_token_usage([
            candidate for candidate in matched_snapshot
            if id(candidate) in budget_survivor_ids
        ])

    try:
        # 1. Match entries  (verbatim from the original body)
        logging.debug(f"Chat Dictionary: Initial matching for: {user_input[:100]}")
        # The original `entry.matches()` is a simple check. `match_whole_words` is more robust.
        # The original `process_user_input` had `entry.matches(user_input)` then later `match_whole_words`.
        # Consolidating to `match_whole_words` as the primary matching mechanism.
        try:
            # Ensure entries are ChatDictionary instances
            valid_initial_entries = []
            for input_index, candidate in enumerate(entries):
                index_by_id.setdefault(id(candidate), input_index)
                if isinstance(candidate, ChatDictionary):
                    valid_initial_entries.append(candidate)
            if len(valid_initial_entries) != len(entries):
                logging.warning("Some provided entries were not ChatDictionary instances and were skipped.")
            matched_entries = match_whole_words(valid_initial_entries, user_input)
        except re.error as e:
            log_counter("chat_dict_regex_error", labels={"key": "compilation_phase"})  # Generic key
            logging.error(f"Invalid regex pattern during initial matching. Error: {str(e)}")
            matched_entries = []
        except Exception as e_match:
            log_counter("chat_dict_match_error")
            logging.error(f"Error during initial matching: {str(e_match)}", exc_info=True)
            matched_entries = []

        matched_snapshot = list(matched_entries)                     # ADDED
        logging.debug(f"Matched entries after initial filtering: {[e.raw_key for e in matched_entries]}")

        # P1c: disabled entries stay visible as near-misses (filtered after match).
        stage_before = list(matched_entries)                          # ADDED
        matched_entries = [e for e in matched_entries if getattr(e, "enabled", True)]  # ADDED
        _record_stage_drops(stage_before, matched_entries, "disabled")               # ADDED

        # 2. Group scoring (verbatim try/except, with a before-list diff)
        stage_before = list(matched_entries)                          # ADDED
        try:
            logging.debug(f"Chat Dictionary: Applying group scoring for {len(matched_entries)} entries")
            matched_entries = group_scoring(matched_entries)
        except Exception as e_gs:  # More specific exception if defined (ChatProcessingError)
            log_counter("chat_dict_group_scoring_error")
            logging.error(f"Error in group scoring: {str(e_gs)}")
            matched_entries = []  # Fallback to empty list
        _record_stage_drops(stage_before, matched_entries, "group_scoring")   # ADDED

        # 3. Probability filter (same pattern)
        stage_before = list(matched_entries)                          # ADDED
        try:
            logging.debug(f"Chat Dictionary: Filtering by probability for {len(matched_entries)} entries")
            matched_entries = filter_by_probability(matched_entries)
        except Exception as e_prob:
            log_counter("chat_dict_probability_error")
            logging.error(f"Error in probability filtering: {str(e_prob)}")
            matched_entries = []  # Fallback to empty list
        _record_stage_drops(stage_before, matched_entries, "probability")     # ADDED

        # 4. Timed effects (same pattern around the original loop)
        # And update last_triggered for those that *will* be used
        stage_before = list(matched_entries)                          # ADDED
        active_timed_entries = []
        try:
            logging.debug("Chat Dictionary: Applying timed effects")
            for entry in matched_entries:
                if apply_timed_effects(entry, current_time):  # Checks if eligible
                    active_timed_entries.append(entry)
            matched_entries = active_timed_entries
        except Exception as e_time:
            log_counter("chat_dict_timed_effects_error")
            logging.error(f"Error applying timed effects: {str(e_time)}")
            matched_entries = []  # Fallback to empty list
        _record_stage_drops(stage_before, matched_entries, "timed_effects")   # ADDED

        # 5. Ordering (strategy sort + priority) (sort-only; drops are only possible via its except)
        stage_before = list(matched_entries)                          # ADDED
        try:
            logging.debug("Chat Dictionary: Applying replacement strategy")
            matched_entries = apply_strategy(matched_entries, strategy)
        except Exception as e_strategy:
            log_counter("chat_dict_strategy_error")
            logging.error(f"Error applying strategy: {str(e_strategy)}")
            matched_entries = []  # Fallback to empty list
        _record_stage_drops(stage_before, matched_entries, "strategy_error")  # ADDED (defensive)
        matched_entries.sort(key=lambda e: -int(getattr(e, "priority", 0) or 0))  # ADDED: stable — strategy order breaks ties

        # 6. Token budget (same pattern; truncation drives budget_exceeded)
        stage_before = list(matched_entries)                          # ADDED
        try:
            logging.debug(f"Chat Dictionary: Enforcing token budget for {len(matched_entries)} entries")
            matched_entries = enforce_token_budget(matched_entries, max_tokens)
            diagnostics.budget_exceeded = len(matched_entries) != len(stage_before)  # ADDED
        except TokenBudgetExceededWarning as e:
            log_counter("chat_dict_token_limit")
            logging.warning(str(e))
            matched_entries = []  # Fallback to empty list
        except Exception as e_budget:
            log_counter("chat_dict_token_budget_error")
            logging.error(f"Error enforcing token budget: {str(e_budget)}")
            matched_entries = []  # Fallback to empty list
        _record_stage_drops(stage_before, matched_entries, "token_budget")    # ADDED
        budget_survivor_ids = {id(e) for e in matched_entries}               # ADDED

        # Alert (dead code in practice — preserved verbatim, not used for diagnostics)
        try:
            alert_token_budget_exceeded(matched_entries, max_tokens)
        except Exception as e_alert:
            log_counter("chat_dict_token_alert_error")
            logging.error(f"Error in token budget alert: {str(e_alert)}")

        # 7. Replacements (verbatim loop + order/count recording)
        for applied_position, entry in enumerate(matched_entries):    # ADDED enumerate
            applied_order_by_id[id(entry)] = applied_position         # ADDED
            try:
                logging.debug("Chat Dictionary: Applying replacements")
                # Use a copy of max_replacements for this run if needed, or modify original for state
                replacements_done_for_this_entry = 0
                # Original code had `entry.max_replacements > 0` check outside loop.
                # If multiple replacements are allowed by one entry definition:
                current_max_replacements = entry.max_replacements  # Use current value
                while current_max_replacements > 0:
                    temp_user_input, replaced_count = apply_replacement_once(temp_user_input, entry)
                    if replaced_count > 0:
                        replacements_done_for_this_entry += 1
                        current_max_replacements -= 1
                        # Update last_triggered for entries that actually made a replacement
                        entry.last_triggered = current_time
                    else:
                        break  # No more matches for this key
                if replacements_done_for_this_entry > 0:
                    logging.debug(f"Replaced {replacements_done_for_this_entry} occurrences of '{entry.raw_key}'")
                replacements_by_id[id(entry)] = replacements_done_for_this_entry  # ADDED
            except Exception as e_replace:
                log_counter("chat_dict_replacement_error", labels={"key": entry.raw_key})
                logging.error(f"Error applying replacement for entry {entry.raw_key}: {str(e_replace)}", exc_info=True)
                skip_reason_by_id.setdefault(id(entry), "error:replacement")  # ADDED
                continue

    except Exception as e_crit:  # Catch-all for ChatProcessingError or other unexpected issues
        log_counter("chat_dict_processing_error")
        logging.error(f"Critical error in process_user_input: {str(e_crit)}", exc_info=True)
        _finalize()                                                   # ADDED
        return original_input_for_fallback, diagnostics  # Return original input on critical failure (CHANGED: now a tuple)

    _finalize()                                                       # ADDED
    return temp_user_input, diagnostics                               # CHANGED (tuple)


def process_user_input(
    user_input: str,
    entries: List[ChatDictionary],
    max_tokens: int = 5000,
    strategy: str = "sorted_evenly"
) -> str:
    """
    Processes user input by applying a series of chat dictionary transformations.

    The pipeline includes:
    1. Matching entries against the input text (regex and whole-word string matching).
    2. Filtering out disabled entries (kept visible as near-misses in diagnostics).
    3. Applying group scoring to select among matched entries from the same group.
    4. Filtering entries by probability.
    5. Applying timed effects (delay, cooldown).
    6. Sorting entries by the chosen strategy, then by priority (stable, descending;
       strategy order breaks ties among equal priorities).
    7. Enforcing a token budget by walking the sorted entries and stopping once the
       budget would be exceeded.
    8. Alerting if the token budget is exceeded by the (potentially filtered) entries.
    9. Applying replacements: each selected entry replaces its key in the user input
       (respecting `entry.max_replacements`).

    If any step in the pipeline encounters a significant error, it may log the error
    and continue with a potentially reduced set of entries or, in critical cases,
    return the original `user_input`.

    Args:
        user_input: The text input from the user.
        entries: A list of `ChatDictionary` objects to apply.
        max_tokens: The maximum token budget for the combined content of applied entries.
                    Defaults to 5000.
        strategy: The strategy for sorting entries before replacement.
                  Defaults to "sorted_evenly".

    Returns:
        The processed user input string after all applicable transformations.
        Returns the original input if critical errors occur.
        Diagnostics-aware callers should use process_user_input_with_diagnostics.
    """
    processed_text, _diagnostics = process_user_input_with_diagnostics(
        user_input, entries, max_tokens=max_tokens, strategy=strategy
    )
    return processed_text


#######################################################################################################################
#
# Database Operations for Chat Dictionaries
#
#######################################################################################################################

def save_chat_dictionary(
    db: CharactersRAGDB,
    name: str,
    description: str = "",
    content: Optional[str] = None,
    entries: Optional[List[ChatDictionary]] = None,
    strategy: str = "sorted_evenly",
    max_tokens: int = 1000,
    enabled: bool = True,
    file_path: Optional[str] = None
) -> Optional[int]:
    """
    Save a chat dictionary to the database.
    
    Args:
        db: Database instance
        name: Dictionary name (must be unique)
        description: Dictionary description
        content: Raw dictionary content (markdown format)
        entries: List of ChatDictionary objects
        strategy: Replacement strategy
        max_tokens: Maximum token budget
        enabled: Whether dictionary is enabled
        file_path: Optional path to source file
        
    Returns:
        Dictionary ID if successful, None otherwise
    """
    try:
        # Validate inputs
        name = _clean_dictionary_text(name, field_name="Dictionary name", max_length=255, required=True)
        description = (
            _clean_dictionary_text(description, field_name="Dictionary description", max_length=1000)
            if description
            else ""
        )
        
        # Convert entries to JSON if provided
        entries_json = None
        if entries:
            entries_list = [entry.to_dict() for entry in entries]
            entries_json = json.dumps(entries_list)
        
        # Insert into database
        cursor = db.get_connection().cursor()
        cursor.execute("""
            INSERT INTO chat_dictionaries 
            (name, description, file_path, content, entries_json, strategy, 
             max_tokens, enabled, client_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, description, file_path, content, entries_json, strategy, 
              max_tokens, enabled, db.client_id))
        
        dict_id = cursor.lastrowid
        db.get_connection().commit()
        
        logger.info(f"Saved chat dictionary '{name}' with ID {dict_id}")
        return dict_id
        
    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed" in str(e):
            raise ConflictError(f"Dictionary with name '{name}' already exists", 
                              entity="chat_dictionaries", entity_id=name)
        raise
    except Exception as e:
        logger.opt(exception=True).error(f"Error saving chat dictionary: {e}")
        db.get_connection().rollback()
        return None


def load_chat_dictionary(db: CharactersRAGDB, dict_id: int) -> Optional[Dict[str, Any]]:
    """
    Load a chat dictionary from the database.
    
    Args:
        db: Database instance
        dict_id: Dictionary ID
        
    Returns:
        Dictionary data including parsed entries, or None if not found
    """
    try:
        cursor = db.get_connection().cursor()
        cursor.execute("""
            SELECT id, name, description, file_path, content, entries_json,
                   strategy, max_tokens, enabled, created_at, last_modified,
                   deleted, client_id, version
            FROM chat_dictionaries 
            WHERE id = ? AND deleted = 0
        """, (dict_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        dict_data = {
            'id': row[0],
            'name': row[1],
            'description': row[2],
            'file_path': row[3],
            'content': row[4],
            'entries_json': row[5],
            'strategy': row[6],
            'max_tokens': row[7],
            'enabled': bool(row[8]),
            'created_at': row[9],
            'last_modified': row[10],
            'deleted': bool(row[11]),
            'client_id': row[12],
            'version': row[13],
        }
        
        # Parse entries from JSON
        if dict_data['entries_json']:
            entries_list = json.loads(dict_data['entries_json'])
            dict_data['entries'] = [ChatDictionary.from_dict(e) for e in entries_list]
        else:
            dict_data['entries'] = []
            
        return dict_data
        
    except Exception as e:
        logger.opt(exception=True).error(f"Error loading chat dictionary {dict_id}: {e}")
        return None


def load_character_dictionaries(char_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse a character card's embedded chat dictionaries into runtime blocks.

    Reads ``extensions['chat_dictionaries']`` (a list of ``export_json`` ``data``
    blocks) and returns one ``{"name", "enabled", "entries": [ChatDictionary...]}``
    per well-formed block. Malformed blocks/entries are skipped. This runs on the
    chat send path over untrusted (imported) card content, so it MUST NOT raise.
    """
    result: List[Dict[str, Any]] = []
    if not isinstance(char_data, dict):
        return result
    ext = char_data.get('extensions')
    if isinstance(ext, str):
        try:
            ext = json.loads(ext or "{}")
        except (TypeError, ValueError):
            ext = {}
    if not isinstance(ext, dict):
        return result
    raw = ext.get('chat_dictionaries') or []
    if not isinstance(raw, list):
        return result
    seen_names: set = set()
    for block in raw:
        if not isinstance(block, dict):
            continue
        name = block.get('name')
        if not name:
            continue
        # A hostile/crafted card can embed two blocks with the same name
        # (``attach_to_character`` dedups by name so it never creates this,
        # but a crafted import can). Without this guard,
        # ``collect_active_chatdict_entries`` would extend a dup-named
        # block's entries twice, and callers keyed by name (e.g. the
        # character-dictionaries panel) could crash on the duplicate. First
        # occurrence wins.
        if str(name) in seen_names:
            continue
        seen_names.add(str(name))
        entries: List[ChatDictionary] = []
        entries_raw = block.get('entries')
        if not isinstance(entries_raw, list):
            entries_raw = []
        for entry in entries_raw:
            try:
                entries.append(ChatDictionary.from_dict(entry))
            except Exception:
                continue
        result.append({
            "name": str(name),
            "enabled": bool(block.get('enabled', True)),
            "entries": entries,
        })
    return result


def collect_active_chatdict_entries(
    db: "CharactersRAGDB",
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
) -> List[ChatDictionary]:
    """Collect the ChatDictionary entries that apply to the current send.

    Additive union of the conversation's attached dictionaries (by id, from
    ``metadata.active_dictionaries``) and the active character's embedded
    dictionaries (snapshots in ``extensions.chat_dictionaries``), deduped at the
    dictionary level by name — the conversation's dictionary WINS a name
    collision. Only enabled dictionaries contribute. Never raises: any bad row is
    skipped so a chat send is never broken by dictionary loading.
    """
    entries: List[ChatDictionary] = []
    conversation_dict_names: set = set()
    if conversation_id and db is not None:
        try:
            conv_details = db.get_conversation_by_id(conversation_id)
        except Exception:
            conv_details = None
        if conv_details:
            try:
                metadata = json.loads(conv_details.get('metadata') or '{}')
            except (TypeError, ValueError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            active = metadata.get('active_dictionaries')
            if not isinstance(active, list):
                active = []
            for dict_id in active:
                try:
                    dict_data = load_chat_dictionary(db, dict_id)
                except Exception:
                    continue
                if dict_data and dict_data.get('enabled', True):
                    conversation_dict_names.add(dict_data.get('name'))
                    entries.extend(dict_data.get('entries') or [])
    for block in load_character_dictionaries(char_data):
        if not block.get('enabled', True):
            continue
        if block.get('name') in conversation_dict_names:
            continue
        entries.extend(block.get('entries') or [])
    return entries


def list_chat_dictionaries(
    db: CharactersRAGDB, 
    limit: int = 100,
    offset: int = 0,
    include_disabled: bool = False
) -> List[Dict[str, Any]]:
    """
    List all chat dictionaries in the database.
    
    Args:
        db: Database instance
        limit: Maximum number of results
        offset: Skip this many results
        include_disabled: Include disabled dictionaries
        
    Returns:
        List of dictionary metadata
    """
    try:
        cursor = db.get_connection().cursor()
        # entry_count is a cheap computed column (JSON1 json_array_length) so
        # the rail meta can show a real count without parsing every entry
        # into a ChatDictionary object here. save_chat_dictionary always
        # json.dumps()s entries_json, so a plain COALESCE-to-'[]' is safe.
        query = """
            SELECT id, name, description, strategy, max_tokens, enabled,
                   created_at, last_modified, client_id, version,
                   json_array_length(COALESCE(entries_json, '[]')) AS entry_count
            FROM chat_dictionaries
            WHERE deleted = 0
        """

        if not include_disabled:
            query += " AND enabled = 1"

        query += " ORDER BY name LIMIT ? OFFSET ?"

        cursor.execute(query, (limit, offset))

        dictionaries = []
        for row in cursor.fetchall():
            dictionaries.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'strategy': row[3],
                'max_tokens': row[4],
                'enabled': bool(row[5]),
                'created_at': row[6],
                'last_modified': row[7],
                'client_id': row[8],
                'version': row[9],
                'entries': [],
                'entry_count': int(row[10] or 0),
            })
            
        return dictionaries
        
    except Exception as e:
        logger.opt(exception=True).error(f"Error listing chat dictionaries: {e}")
        return []


def update_chat_dictionary(
    db: CharactersRAGDB,
    dict_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    content: Optional[str] = None,
    entries: Optional[List[ChatDictionary]] = None,
    strategy: Optional[str] = None,
    max_tokens: Optional[int] = None,
    enabled: Optional[bool] = None,
    expected_version: Optional[int] = None
) -> bool:
    """
    Update a chat dictionary in the database.
    
    Args:
        db: Database instance
        dict_id: Dictionary ID to update
        name: New name (optional)
        description: New description (optional)
        content: New content (optional)
        entries: New entries list (optional)
        strategy: New strategy (optional)
        max_tokens: New max tokens (optional)
        enabled: New enabled state (optional)
        expected_version: Expected version for optimistic locking
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Build update query dynamically
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(_clean_dictionary_text(name, field_name="Dictionary name", max_length=255, required=True))
        if description is not None:
            updates.append("description = ?")
            params.append(_clean_dictionary_text(description, field_name="Dictionary description", max_length=1000))
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if entries is not None:
            updates.append("entries_json = ?")
            entries_list = [entry.to_dict() for entry in entries]
            params.append(json.dumps(entries_list))
        if strategy is not None:
            updates.append("strategy = ?")
            params.append(strategy)
        if max_tokens is not None:
            updates.append("max_tokens = ?")
            params.append(max_tokens)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(int(enabled))
            
        if not updates:
            return True  # Nothing to update
            
        updates.append("last_modified = CURRENT_TIMESTAMP")
        updates.append("version = version + 1")
        
        query = f"UPDATE chat_dictionaries SET {', '.join(updates)} WHERE id = ? AND deleted = 0"
        params.append(dict_id)
        
        if expected_version is not None:
            query += " AND version = ?"
            params.append(expected_version)
            
        conn = db.get_connection()
        _ensure_chat_dictionaries_fts_update_trigger(conn)
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        if cursor.rowcount == 0:
            if expected_version is not None:
                raise ConflictError("Version mismatch - dictionary was modified by another process",
                                  entity="chat_dictionaries", entity_id=dict_id)
            return False
            
        conn.commit()
        logger.info(f"Updated chat dictionary {dict_id}")
        return True

    except ConflictError:
        db.get_connection().rollback()
        raise
    except Exception as e:
        logger.opt(exception=True).error(f"Error updating chat dictionary {dict_id}: {e}")
        db.get_connection().rollback()
        return False


def delete_chat_dictionary(
    db: CharactersRAGDB,
    dict_id: int,
    expected_version: Optional[int] = None
) -> bool:
    """
    Soft delete a chat dictionary.
    
    Args:
        db: Database instance
        dict_id: Dictionary ID to delete
        expected_version: Expected version for optimistic locking
        
    Returns:
        True if successful, False otherwise
    """
    try:
        query = """
            UPDATE chat_dictionaries 
            SET deleted = 1, last_modified = CURRENT_TIMESTAMP, version = version + 1
            WHERE id = ? AND deleted = 0
        """
        params = [dict_id]
        
        if expected_version is not None:
            query += " AND version = ?"
            params.append(expected_version)
            
        cursor = db.get_connection().cursor()
        cursor.execute(query, params)
        
        if cursor.rowcount == 0:
            if expected_version is not None:
                raise ConflictError("Version mismatch - dictionary was modified by another process",
                                  entity="chat_dictionaries", entity_id=dict_id)
            return False
            
        db.get_connection().commit()
        logger.info(f"Deleted chat dictionary {dict_id}")
        return True

    except ConflictError:
        db.get_connection().rollback()
        raise
    except Exception as e:
        logger.opt(exception=True).error(f"Error deleting chat dictionary {dict_id}: {e}")
        db.get_connection().rollback()
        return False


#######################################################################################################################
#
# File Management Functions
#
#######################################################################################################################

def get_chat_dicts_folder() -> Path:
    """Get the chat dictionaries folder path, creating it if needed."""
    from ..Utils.paths import get_user_data_dir
    
    chat_dicts_folder = get_user_data_dir() / "chat_dicts"
    chat_dicts_folder.mkdir(parents=True, exist_ok=True)
    
    return chat_dicts_folder


def import_dictionary_from_file(
    db: CharactersRAGDB,
    file_path: str,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Optional[int]:
    """
    Import a chat dictionary from a markdown file.
    
    Args:
        db: Database instance
        file_path: Path to the dictionary file
        name: Override name (uses filename if not provided)
        description: Override description
        
    Returns:
        Dictionary ID if successful, None otherwise
    """
    try:
        # Parse the file
        base_dir = str(get_chat_dicts_folder())
        dict_content = parse_user_dict_markdown_file(file_path, base_dir)
        
        if not dict_content:
            logger.error(f"No content found in dictionary file: {file_path}")
            return None
            
        # Convert to ChatDictionary objects
        entries = []
        for key, content in dict_content.items():
            entries.append(ChatDictionary(key=key, content=content))
            
        # Use filename as name if not provided
        if not name:
            name = Path(file_path).stem
            
        # Read the raw content for storage
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
            
        # Copy file to chat_dicts folder if not already there
        chat_dicts_folder = get_chat_dicts_folder()
        dest_path = chat_dicts_folder / Path(file_path).name
        
        if Path(file_path).resolve() != dest_path.resolve():
            import shutil
            shutil.copy2(file_path, dest_path)
            file_path = str(dest_path)
            
        # Save to database
        return save_chat_dictionary(
            db=db,
            name=name,
            description=description or f"Imported from {Path(file_path).name}",
            content=raw_content,
            entries=entries,
            file_path=file_path
        )
        
    except Exception as e:
        logger.opt(exception=True).error(f"Error importing dictionary from file: {e}")
        return None


def export_dictionary_to_file(
    db: CharactersRAGDB,
    dict_id: int,
    export_path: Optional[str] = None
) -> Optional[str]:
    """
    Export a chat dictionary to a markdown file.
    
    Args:
        db: Database instance
        dict_id: Dictionary ID to export
        export_path: Optional export path (uses default if not provided)
        
    Returns:
        Export file path if successful, None otherwise
    """
    try:
        # Load dictionary
        dict_data = load_chat_dictionary(db, dict_id)
        if not dict_data:
            logger.error(f"Dictionary {dict_id} not found")
            return None
            
        # Use original content if available, otherwise reconstruct
        if dict_data.get('content'):
            content = dict_data['content']
        else:
            # Reconstruct from entries
            lines = []
            lines.append(f"# {dict_data['name']}")
            lines.append(f"# {dict_data['description']}")
            lines.append("")
            
            for entry in dict_data.get('entries', []):
                if '\n' in entry.content:
                    lines.append(f"{entry.raw_key}: |")
                    lines.append(entry.content)
                    lines.append("---@@@---")
                else:
                    lines.append(f"{entry.raw_key}: {entry.content}")
                    
            content = '\n'.join(lines)
            
        # Determine export path
        if not export_path:
            chat_dicts_folder = get_chat_dicts_folder()
            safe_name = re.sub(r'[^\w\s-]', '', dict_data['name'])
            safe_name = re.sub(r'[-\s]+', '-', safe_name)
            export_path = str(chat_dicts_folder / f"{safe_name}.md")
            
        # Write file
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Exported dictionary to {export_path}")
        return export_path
        
    except Exception as e:
        logger.opt(exception=True).error(f"Error exporting dictionary: {e}")
        return None


def list_available_dictionary_files() -> List[Dict[str, str]]:
    """
    List all dictionary files in the chat_dicts folder.
    
    Returns:
        List of dictionaries with 'name' and 'path' keys
    """
    try:
        chat_dicts_folder = get_chat_dicts_folder()
        files = []
        
        for file_path in chat_dicts_folder.glob("*.md"):
            files.append({
                'name': file_path.stem,
                'path': str(file_path)
            })
            
        # Also check for .yaml files
        for file_path in chat_dicts_folder.glob("*.yaml"):
            files.append({
                'name': file_path.stem,
                'path': str(file_path)
            })
            
        return sorted(files, key=lambda x: x['name'])
        
    except Exception as e:
        logger.opt(exception=True).error(f"Error listing dictionary files: {e}")
        return []


#
# End of Chat_Dictionary_Lib.py
#######################################################################################################################
