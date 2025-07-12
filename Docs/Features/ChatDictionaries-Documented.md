# Chat Dictionaries - Comprehensive User Guide

## Table of Contents
- [What are Chat Dictionaries?](#what-are-chat-dictionaries)
- [Key Features](#key-features)
- [How Chat Dictionaries Work](#how-chat-dictionaries-work)
- [Creating Chat Dictionaries](#creating-chat-dictionaries)
- [Dictionary File Format](#dictionary-file-format)
- [Using Chat Dictionaries in the UI](#using-chat-dictionaries-in-the-ui)
- [Advanced Features](#advanced-features)
- [Configuration Settings](#configuration-settings)
- [Use Cases and Examples](#use-cases-and-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Comparison with World Books](#comparison-with-world-books)

## What are Chat Dictionaries?

Chat Dictionaries are a powerful text transformation system in tldw_chatbook that allows you to create keyword-based replacements and text expansions. Unlike World Books which inject context, Chat Dictionaries actively modify the text of messages both before sending to the AI (pre-processing) and after receiving the response (post-processing).

### Core Concept

Think of Chat Dictionaries as intelligent find-and-replace tools that can:
- Transform user input before the AI sees it
- Modify AI responses before displaying them
- Use simple keywords or complex regex patterns
- Apply probabilistically or conditionally
- Respect token budgets to avoid context overflow

## Key Features

### 1. **Dual-Stage Processing**
- **Pre-processing**: Modifies your message before sending to the AI
- **Post-processing**: Transforms the AI's response before display

### 2. **Flexible Pattern Matching**
- **Plain Keywords**: Simple word matching (case-insensitive by default)
- **Regular Expressions**: Complex pattern matching with `/pattern/flags` syntax
- **Whole Word Matching**: Prevents partial matches (e.g., "cat" won't match "category")

### 3. **Probability and Randomization**
- Set probability (0-100%) for each entry
- Useful for variety in responses
- Can create dynamic, non-repetitive interactions

### 4. **Grouping System**
- Group related entries together
- Only one entry per group is selected per processing
- Enables mutually exclusive replacements

### 5. **Timed Effects**
- **Sticky**: Entry remains active for X messages after triggering
- **Cooldown**: Entry can't trigger again for X messages
- **Delay**: Entry only triggers after X messages

### 6. **Token Budget Management**
- Set maximum tokens for dictionary content
- Prevents context overflow
- Automatically prioritizes entries when over budget

### 7. **Multiple Replacement Strategies**
- `sorted_evenly`: Balanced distribution
- `character_lore_first`: Prioritizes character-specific entries
- Custom strategies can be implemented

## How Chat Dictionaries Work

### Processing Pipeline

1. **User Input** → You type a message
2. **Pre-processing Stage**:
   - Dictionary entries are matched against your input
   - Matching entries are filtered by probability
   - Token budget is enforced
   - Replacements are applied based on strategy
   - Modified message is sent to AI
3. **AI Processing** → AI generates response
4. **Post-processing Stage**:
   - Dictionary entries are matched against AI output
   - Same filtering and budget process
   - Replacements applied to AI response
   - Modified response is displayed to you

### Example Flow

```
User types: "Tell me about the mcguffin"
↓ (Pre-processing)
Dictionary replaces: "mcguffin" → "ancient artifact of immense power"
↓
Sent to AI: "Tell me about the ancient artifact of immense power"
↓
AI responds: "The ancient artifact of immense power is fascinating..."
↓ (Post-processing)
Dictionary replaces: "ancient artifact" → "McGuffin"
↓
Displayed: "The McGuffin of immense power is fascinating..."
```

## Creating Chat Dictionaries

### Method 1: Through the UI

1. **Navigate to CCP Tab** (Conversations, Characters & Prompts)
2. **Open Chat Dictionaries Collapsible** in the left sidebar
3. **Click "Create Dictionary"**
4. **Fill in the details**:
   - Name: Unique identifier
   - Description: What this dictionary does
   - Strategy: How replacements are applied
   - Max Tokens: Token budget
5. **Add Entries** with the entry editor
6. **Save** the dictionary

### Method 2: Import from File

1. **Create a markdown file** with your entries
2. **Click "Import Dictionary"** in CCP tab
3. **Select your file**
4. **Review and confirm** import

### Method 3: Direct Database Creation

Chat dictionaries can be created programmatically through the API, useful for bulk operations or integrations.

## Dictionary File Format

Chat Dictionaries use a special markdown-like format for easy editing:

### Single-Line Entries

```markdown
keyword: replacement text
hello: Greetings, esteemed colleague!
/\bAI\b/i: artificial intelligence
```

### Multi-Line Entries

```markdown
description: |
This is a multi-line
replacement that preserves
line breaks and formatting.
---@@@---
```

### Regex Patterns

```markdown
# Simple regex (case-sensitive)
/pattern/: replacement

# Regex with flags
/pattern/i: case-insensitive replacement
/pattern/im: case-insensitive, multiline
/\b(quick|fast|speedy)\b/i: rapid
```

### Comments

```markdown
# This is a comment and will be ignored
keyword: replacement  # This part is the actual entry
```

### Complete Example

```markdown
# Fantasy Dictionary Example
# Replaces modern terms with fantasy equivalents

# Transportation
car: horseless carriage
airplane: sky ship
/\btrain\b/i: iron dragon

# Technology
computer: thinking machine
phone: speaking stone
/\binternet\b/i: the aethernet

# Multi-line lore entry
magic_explanation: |
Magic in this world flows from ley lines
that crisscross the land. Wizards tap into
these lines using special crystals.
---@@@---

# Regex for numbers
/\b(\d+)\s*dollars?\b/i: $1 gold pieces
```

## Using Chat Dictionaries in the UI

### In the Chat Tab

1. **Right Sidebar** → Open the chat right sidebar
2. **Chat Dictionaries Section** → Expand the collapsible
3. **Search** → Find dictionaries by name or description
4. **Add to Conversation**:
   - Select a dictionary from "Available"
   - Click "Add" button
   - Dictionary appears in "Active" list
5. **Enable/Disable** → Toggle the checkbox to turn processing on/off
6. **View Details** → Click a dictionary to see statistics and entries

### In the CCP Tab

1. **Left Sidebar** → Chat Dictionaries section
2. **Management Options**:
   - Import Dictionary
   - Create Dictionary
   - Load/Edit existing
   - Export dictionary
   - Clone dictionary
   - Delete dictionary
3. **Entry Management**:
   - Add new entries
   - Edit existing entries
   - Remove entries
   - Test replacements

### Dictionary Editor View

When editing a dictionary, you'll see:
- **Metadata Section**: Name, description, settings
- **Entries List**: All dictionary entries with:
  - Key (keyword/pattern)
  - Content (replacement)
  - Probability
  - Group
  - Max Replacements
- **Entry Editor**: Add/modify individual entries
- **Statistics**: Entry counts, token usage

## Advanced Features

### 1. Probability Control

Set probability for variety:
```markdown
# 50% chance to replace
greeting: hello|50

# In UI: Set probability slider to 50%
```

### 2. Grouping

Create mutually exclusive options:
```markdown
# Only one emotion will be selected
[emotion]happy: joyful
[emotion]sad: melancholic
[emotion]angry: furious
```

### 3. Timed Effects

Configure in the UI or via metadata:
- **Sticky**: Entry stays active for N messages
- **Cooldown**: Can't trigger again for N messages
- **Delay**: Only triggers after N messages

### 4. Max Replacements

Control how many times an entry can trigger:
- Set to 1 for single replacement
- Set higher for multiple replacements
- Useful for preventing over-replacement

### 5. Token Budget Management

The system automatically:
- Calculates token usage for all entries
- Prioritizes based on strategy
- Warns when over budget
- Removes lowest priority entries if needed

## Configuration Settings

In `config.toml`:

```toml
[ChatDictionaries]
enable_chat_dictionaries = true
post_gen_replacement = true
chat_dictionary_max_tokens = 1000
chat_dictionary_replacement_strategy = "sorted_evenly"

# File paths for dictionaries
chat_dictionary_chat_prompts = ""
chat_dictionary_RAG_prompts = ""
post_gen_replacement_dict = ""
```

### Key Settings

- **enable_chat_dictionaries**: Master on/off switch
- **post_gen_replacement**: Enable post-processing
- **chat_dictionary_max_tokens**: Default token budget
- **chat_dictionary_replacement_strategy**: Default strategy

## Use Cases and Examples

### 1. Writing Style Consistency

Create a dictionary for consistent terminology:
```markdown
# Technical Writing Standards
utilize: use
implement: create
leverage: use
/\bROI\b/: return on investment
```

### 2. Character Speech Patterns

Make characters speak distinctively:
```markdown
# Pirate Speak
you: ye
your: yer
yes: aye
/\bthe\b/: th'
friend: matey
```

### 3. World-Building Consistency

Maintain consistent naming:
```markdown
# Fantasy World Names
Earth: Terra
human: Terran
New York: Neo Arcanum
president: High Chancellor
```

### 4. Censorship/Content Filtering

```markdown
# Content Filter
/\b(damn|hell)\b/i: [redacted]
/\bstupid\b/i: silly
```

### 5. Abbreviation Expansion

```markdown
# Medical Abbreviations
BP: blood pressure
HR: heart rate
/\bDx\b/: diagnosis
/\bRx\b/: prescription
```

### 6. Dynamic Responses

Using probability for variety:
```markdown
# Greeting Variety (each 33% chance)
[greet]hello: Hey there!|33
[greet]hello: Howdy!|33
[greet]hello: What's up!|33
```

## Best Practices

### 1. **Organize with Clear Naming**
- Use descriptive dictionary names
- Group related transformations
- Document purpose in description

### 2. **Test Before Production**
- Use test conversations
- Verify replacements work as expected
- Check for unintended matches

### 3. **Mind the Token Budget**
- Monitor token usage
- Prioritize essential replacements
- Use groups to limit selections

### 4. **Use Regex Carefully**
- Test regex patterns thoroughly
- Use word boundaries `\b` to prevent partial matches
- Consider case-sensitivity needs

### 5. **Version Control**
- Export dictionaries regularly
- Keep backups of complex dictionaries
- Document changes

### 6. **Performance Optimization**
- Limit active dictionaries per conversation
- Use specific patterns over broad ones
- Disable unused dictionaries

### 7. **Combine with World Books**
- Use dictionaries for text transformation
- Use world books for context injection
- They work together seamlessly

## Troubleshooting

### Common Issues

#### 1. **Replacements Not Working**

**Symptoms**: Keywords aren't being replaced
**Solutions**:
- Check if dictionary is active in conversation
- Verify the enable checkbox is checked
- Ensure keyword matches exactly (check spaces, case)
- For regex, verify pattern syntax

#### 2. **Over-Replacement**

**Symptoms**: Too many replacements happening
**Solutions**:
- Use word boundaries in patterns: `\bword\b`
- Reduce max_replacements value
- Make patterns more specific
- Use grouping to limit selections

#### 3. **Token Budget Exceeded**

**Symptoms**: Warning messages, entries not applying
**Solutions**:
- Increase token budget in settings
- Reduce number of active dictionaries
- Shorten replacement content
- Use probability to reduce activation

#### 4. **Regex Errors**

**Symptoms**: Pattern treated as literal text
**Solutions**:
- Check regex syntax
- Verify flags are valid (i, m, s)
- Test in regex tester first
- Check logs for specific errors

#### 5. **Performance Issues**

**Symptoms**: Slow message processing
**Solutions**:
- Reduce number of entries
- Simplify regex patterns
- Disable unused dictionaries
- Use specific keywords over broad patterns

### Debug Mode

Enable debug logging in `config.toml`:
```toml
[logging]
log_level = "DEBUG"
```

Check logs at: `~/.share/tldw_cli/logs/`

## Comparison with World Books

| Feature | Chat Dictionaries | World Books |
|---------|------------------|-------------|
| **Purpose** | Text transformation | Context injection |
| **Processing** | Pre/post message | During message prep |
| **Trigger** | Pattern match | Keyword scan |
| **Effect** | Replaces text | Adds context |
| **Visibility** | Changes visible text | Adds hidden context |
| **Token Usage** | Reduces/maintains | Always increases |
| **Use Case** | Style, terminology | Lore, information |

### When to Use Which?

**Use Chat Dictionaries for**:
- Consistent terminology
- Speech patterns
- Text corrections
- Dynamic responses
- Style enforcement

**Use World Books for**:
- Background information
- Character details
- Setting descriptions
- Plot elements
- Knowledge injection

### Using Both Together

They complement each other perfectly:

1. **World Book** injects: "The kingdom of Eldoria is ruled by King Aldric"
2. **Chat Dictionary** transforms: "King" → "His Majesty"
3. **Result**: Consistent world-building with proper terminology

## Summary

Chat Dictionaries are a versatile tool for maintaining consistency, creating unique character voices, and managing terminology across your conversations. When combined with World Books, they provide a complete system for rich, immersive, and consistent AI interactions.

Key takeaways:
- Process text before and after AI interaction
- Support both simple keywords and complex patterns
- Respect token budgets automatically
- Work seamlessly with other tldw_chatbook features
- Highly customizable for any use case

For technical implementation details, see the developer documentation in the Development folder.