# World/Lore Books - Comprehensive User Guide

## Table of Contents
- [What are World/Lore Books?](#what-are-worldlore-books)
- [Key Features](#key-features)
- [How World Books Work](#how-world-books-work)
- [Creating World Books](#creating-world-books)
- [World Book Entry Format](#world-book-entry-format)
- [Using World Books in the UI](#using-world-books-in-the-ui)
- [Advanced Features](#advanced-features)
- [Configuration Settings](#configuration-settings)
- [Use Cases and Examples](#use-cases-and-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Comparison with Chat Dictionaries](#comparison-with-chat-dictionaries)

## What are World/Lore Books?

World Books (also called Lore Books) are a powerful context injection system in tldw_chatbook that automatically provides relevant background information to the AI based on keywords in your conversation. Unlike Chat Dictionaries which transform text, World Books add invisible context that helps the AI understand your world, characters, and settings better.

### Core Concept

Think of World Books as your AI's reference library that:
- Automatically provides relevant information when certain topics come up
- Works invisibly in the background (context isn't shown in chat)
- Can be shared across multiple characters and conversations
- Respects token limits to avoid overwhelming the AI
- Organizes information by priority and position

## Key Features

### 1. **Keyword-Triggered Activation**
- Entries activate when specific keywords appear in the conversation
- Supports multiple keywords per entry
- Can scan current message and conversation history
- Word boundary matching prevents false triggers

### 2. **Flexible Entry Types**
- **Character-Embedded**: Built into character cards for character-specific lore
- **Standalone**: Separate world books that can be shared across characters
- **Mixed Mode**: Use both types together seamlessly

### 3. **Smart Context Injection**
- **Position Control**: Inject before/after character definition or at start/end
- **Priority System**: Control which entries are processed first
- **Token Budget**: Automatically manages context size to prevent overflow
- **Selective Activation**: Require secondary keywords for context-sensitive entries

### 4. **Advanced Features**
- **Recursive Scanning**: Activated entries can trigger more entries
- **Regex Support**: Complex pattern matching for advanced users
- **Import/Export**: Compatible with SillyTavern format
- **Scan Depth**: Configure how many messages to scan for keywords

### 5. **Organization Tools**
- Group related lore into separate world books
- Set priorities for different books
- Enable/disable books per conversation
- Full CRUD operations with version control

## How World Books Work

### Processing Pipeline

1. **Keyword Detection**
   - System scans your current message and recent history
   - Looks for matching keywords in all active world books
   - Checks word boundaries to avoid partial matches

2. **Entry Selection**
   - Filters entries based on enable status
   - Applies selective activation rules
   - Sorts by priority and insertion order

3. **Token Budget Check**
   - Calculates tokens for each entry
   - Selects entries that fit within budget
   - Prioritizes based on configuration

4. **Context Injection**
   - Injects selected entries at specified positions
   - Builds complete context for AI
   - Sends enhanced prompt to LLM

### Example Flow

```
You type: "Tell me about the dragon's lair in the Dark Forest"
↓
World Book scans for keywords: "dragon", "lair", "Dark Forest"
↓
Finds matching entries:
- Dragon lore (500 tokens)
- Dark Forest description (300 tokens)
- Lair layouts (200 tokens)
↓
Checks token budget (1000 tokens) - all fit!
↓
Injects context (invisible to you):
[Dragon info][Dark Forest info][Lair info][Your actual message]
↓
AI responds with full knowledge of your world's lore
```

## Creating World Books

### Method 1: Through the CCP Tab UI

1. **Navigate to CCP Tab** (Conversations, Characters & Prompts)
2. **Open World/Lore Books Collapsible** in the left sidebar
3. **Click "Create World Book"**
4. **Fill in the details**:
   - **Name**: Unique identifier (e.g., "Fantasy World Lore")
   - **Description**: What this book contains
   - **Scan Depth**: How many messages to scan (default: 3)
   - **Token Budget**: Maximum tokens for entries (default: 500)
   - **Recursive Scanning**: Enable to let entries trigger other entries
5. **Save** the world book
6. **Add Entries** using the entry editor

### Method 2: Import from File

1. **Prepare your world book** in JSON format (SillyTavern compatible)
2. **Click "Import World Book"** in CCP tab
3. **Select your file**
4. **Review** the imported entries
5. **Save** to complete import

### Method 3: Character-Embedded

1. **Edit a character** in the CCP tab
2. **Add world info** to the character's extensions
3. **Save** the character
4. These entries are automatically available when using that character

## World Book Entry Format

### Basic Entry Structure

Each world book entry contains:
- **Keys**: Keywords that trigger the entry
- **Content**: The lore/information to inject
- **Position**: Where to inject (before_char, after_char, at_start, at_end)
- **Insertion Order**: Priority within the same position
- **Enabled**: Active/inactive status
- **Selective**: Whether secondary keys are required
- **Secondary Keys**: Additional required keywords
- **Case Sensitive**: Whether to match case exactly

### JSON Format Example

```json
{
  "entries": {
    "dragon_lore": {
      "keys": ["dragon", "wyrm", "drake"],
      "content": "Dragons in this world are ancient beings of immense magical power. They hoard knowledge rather than gold, and each dragon specializes in a different field of study. The eldest dragons can manipulate reality itself through their understanding of fundamental truths.",
      "position": "before_char",
      "insertion_order": 100,
      "enabled": true,
      "selective": false,
      "secondary_keys": [],
      "case_sensitive": false
    },
    "dark_forest": {
      "keys": ["Dark Forest", "Shadowwood"],
      "content": "The Dark Forest is a mystical woodland where the veil between worlds is thin. Time flows differently here, and those who enter may emerge days later having aged years, or vice versa. The forest is home to the Shadow Elves and various magical creatures.",
      "position": "before_char",
      "insertion_order": 90,
      "enabled": true,
      "selective": true,
      "secondary_keys": ["location", "travel", "journey", "explore"],
      "case_sensitive": true
    }
  }
}
```

## Using World Books in the UI

### In the Chat Tab

1. **Right Sidebar** → Expand the chat sidebar
2. **World Books Section** → Find the World Books collapsible
3. **Available Books** → See all world books in the system
4. **Search** → Filter books by name or description
5. **Add to Conversation**:
   - Select a book from the list
   - Click "Add" button
   - Book appears in active list
6. **Priority** → Books are processed in priority order
7. **View Details** → Click a book to see entry count and info

### In the CCP Tab

1. **Left Sidebar** → World/Lore Books section
2. **Management Options**:
   - **Import World Book** - Import from JSON file
   - **Create World Book** - Create new book
   - **Search** - Find books by name
   - **Load Selected** - Load book for editing
   - **Edit Selected** - Modify existing book
   - **Refresh List** - Update book list
3. **Entry Management** (when editing):
   - Add new entries
   - Edit existing entries
   - Delete entries
   - Test keyword matching
   - Reorder entries

### Character-Specific World Info

1. **Load a Character** in CCP tab
2. **Character World Info** section appears
3. **View/Edit** embedded world info
4. **Save** changes with character

## Advanced Features

### 1. Selective Activation

Use secondary keywords for context-sensitive entries:

```json
{
  "keys": ["sword"],
  "content": "Excalibur, the legendary sword of kings...",
  "selective": true,
  "secondary_keys": ["Arthur", "king", "legend", "Excalibur"]
}
```

This entry only activates when "sword" appears WITH one of the secondary keywords, preventing activation for generic sword mentions.

### 2. Recursive Scanning

Enable entries to trigger other entries:
- Set `recursive_scanning: true` on the world book
- When an entry activates, its content is scanned for more keywords
- Useful for interconnected lore where one topic leads to another

### 3. Position Strategy

Control exactly where information appears:
- **before_char**: Right before character definition (most common)
- **after_char**: After character but before conversation
- **at_start**: Very beginning of context
- **at_end**: Very end, closest to current message

### 4. Priority Management

Higher priority books/entries are processed first:
- Set priority when associating book with conversation
- Higher number = higher priority
- Affects token budget allocation
- Character-embedded entries have default priority 0

### 5. Token Budget Optimization

The system automatically:
- Estimates tokens per entry (~4 characters = 1 token)
- Prioritizes entries based on order and priority
- Stops adding entries when budget exceeded
- Warns in logs when entries are skipped

## Configuration Settings

### Global Settings (config.toml)

```toml
[character_chat]
enable_world_info = true  # Master on/off switch
```

### Per World Book Settings

- **scan_depth**: How many previous messages to scan (default: 3)
- **token_budget**: Maximum tokens for all entries (default: 500)
- **recursive_scanning**: Enable recursive keyword matching (default: false)
- **enabled**: Whether the book is active (default: true)

### Per Entry Settings

- **position**: Where to inject content
- **insertion_order**: Order within position (lower = earlier)
- **enabled**: Whether entry is active
- **selective**: Require secondary keywords
- **case_sensitive**: Exact case matching

## Use Cases and Examples

### 1. Fantasy World Building

Create consistent lore across multiple conversations:

```json
{
  "name": "Eldoria Campaign Setting",
  "entries": {
    "world_overview": {
      "keys": ["Eldoria", "world", "setting"],
      "content": "Eldoria is a vast continent divided into seven kingdoms, each blessed by a different deity. Magic flows through ley lines that converge at ancient temples."
    },
    "magic_system": {
      "keys": ["magic", "spell", "wizard", "mage"],
      "content": "Magic in Eldoria requires both innate talent and years of study. Spells are categorized into seven schools corresponding to the seven deities."
    },
    "political_landscape": {
      "keys": ["king", "queen", "politics", "kingdom"],
      "content": "The Seven Kingdoms maintain an uneasy peace through the Council of Crowns. Each kingdom specializes in different aspects of civilization."
    }
  }
}
```

### 2. Technical Documentation

Provide consistent technical information:

```json
{
  "name": "Project Technical Specs",
  "entries": {
    "architecture": {
      "keys": ["architecture", "system design", "components"],
      "content": "Our system uses a microservices architecture with Docker containers orchestrated by Kubernetes. The main services are: Auth Service, User Service, Content Service, and Analytics Service."
    },
    "api_standards": {
      "keys": ["API", "endpoint", "REST"],
      "content": "All APIs follow REST principles with JSON payloads. Authentication uses JWT tokens. Versioning is handled through URL paths (/api/v1/, /api/v2/)."
    },
    "database_schema": {
      "keys": ["database", "schema", "tables"],
      "content": "PostgreSQL database with schemas for each service. Main tables: users, sessions, content, analytics_events. All timestamps in UTC."
    }
  }
}
```

### 3. Character Relationships

Track complex character interactions:

```json
{
  "name": "Character Relationships",
  "entries": {
    "family_tree": {
      "keys": ["family", "related", "parent", "sibling"],
      "content": "The Ashford family: Lord Edmund (patriarch), Lady Catherine (matriarch), Children: William (heir), Elizabeth (scholar), Thomas (knight)."
    },
    "alliances": {
      "keys": ["alliance", "ally", "friend"],
      "content": "House Ashford allied with House Blackwood through marriage. Friendly with House Ravencrest. Neutral toward House Ironhold."
    },
    "conflicts": {
      "keys": ["enemy", "conflict", "rival"],
      "content": "Historical rivalry with House Crimson over border disputes. Current tensions with House Goldleaf over trade routes."
    }
  }
}
```

### 4. Scientific Research Context

Maintain consistency in technical discussions:

```json
{
  "name": "Research Project Context",
  "entries": {
    "methodology": {
      "keys": ["methodology", "method", "approach"],
      "content": "This study uses a mixed-methods approach combining quantitative analysis of sensor data with qualitative interviews. Sample size: n=500."
    },
    "previous_findings": {
      "keys": ["previous", "prior", "earlier research"],
      "content": "Smith et al. (2019) found a 23% correlation. Johnson (2020) challenged this with a larger sample showing 31%. Our preliminary data suggests 28%."
    },
    "terminology": {
      "keys": ["define", "definition", "terminology"],
      "content": "Key terms: 'Cognitive Load' - mental effort in working memory. 'Task Saturation' - point where performance degrades. 'Flow State' - optimal performance zone."
    }
  }
}
```

### 5. Game Campaign Management

For tabletop RPG sessions:

```json
{
  "name": "Current Campaign Status",
  "entries": {
    "party_inventory": {
      "keys": ["inventory", "items", "equipment"],
      "content": "Party inventory: 3x healing potions, Map of the Northern Wastes, Crystal of True Seeing, 450 gold pieces, Mysterious sealed letter."
    },
    "active_quests": {
      "keys": ["quest", "mission", "objective"],
      "content": "Active quests: 1) Find the missing prince (Day 12), 2) Investigate strange lights in the forest, 3) Deliver letter to Archmage Zeldin."
    },
    "npc_states": {
      "keys": ["NPC", "townspeople", "merchant"],
      "content": "Notable NPCs: Gareth (blacksmith) - friendly after party saved his daughter. Marina (innkeeper) - suspicious of strangers. Lord Blackthorne - secretly plotting against the party."
    }
  }
}
```

## Best Practices

### 1. **Organize Logically**
- Create separate books for different topics
- Use clear, descriptive names
- Group related entries together
- Document the purpose in descriptions

### 2. **Keyword Selection**
- Choose specific, unique keywords
- Avoid common words that trigger too often
- Use multiple keywords for important entries
- Consider case sensitivity for proper nouns

### 3. **Content Writing**
- Keep entries concise but informative
- Write in a style the AI can incorporate
- Avoid contradictions between entries
- Update entries as your world evolves

### 4. **Performance Optimization**
- Monitor token usage with larger books
- Disable unused entries rather than deleting
- Use selective activation for situational lore
- Limit recursive scanning depth

### 5. **Testing and Validation**
- Test keyword triggers in actual conversation
- Verify entries activate when expected
- Check token budget isn't exceeded
- Review AI responses for consistency

### 6. **Maintenance**
- Regularly review and update entries
- Remove outdated information
- Export books for backup
- Version control important changes

## Troubleshooting

### Common Issues

#### 1. **Entries Not Triggering**

**Symptoms**: Keywords in conversation don't activate entries
**Solutions**:
- Verify world book is associated with conversation
- Check entry is enabled
- Confirm keywords match exactly (including case if sensitive)
- Ensure scan depth covers the message with keyword
- Check token budget isn't already exceeded

#### 2. **Too Many Entries Triggering**

**Symptoms**: Context becomes cluttered, token limit exceeded
**Solutions**:
- Use more specific keywords
- Enable selective activation with secondary keys
- Reduce scan depth
- Adjust token budget limits
- Set better priorities

#### 3. **Wrong Information Injected**

**Symptoms**: AI uses outdated or incorrect lore
**Solutions**:
- Review and update entry content
- Check for conflicting entries
- Verify priority order is correct
- Disable outdated entries

#### 4. **Performance Issues**

**Symptoms**: Slow message processing
**Solutions**:
- Reduce number of active world books
- Limit entries per book
- Disable recursive scanning
- Optimize keyword patterns

#### 5. **Import/Export Problems**

**Symptoms**: Can't import SillyTavern books
**Solutions**:
- Verify JSON format is correct
- Check for name conflicts
- Ensure file permissions allow reading
- Try importing smaller chunks

### Debug Mode

Enable debug logging to see world book processing:

```toml
[logging]
log_level = "DEBUG"
```

Check logs at: `~/.share/tldw_cli/logs/`

Look for:
- "World info keyword found"
- "Token budget exceeded"
- "Applying world info injections"

## Comparison with Chat Dictionaries

| Aspect | World Books | Chat Dictionaries |
|--------|-------------|------------------|
| **Purpose** | Add context/lore | Transform text |
| **Visibility** | Invisible to user | Changes visible text |
| **Processing** | During prompt building | Pre/post message |
| **Token Usage** | Always increases | Can reduce or maintain |
| **Trigger** | Keyword presence | Pattern matching |
| **Effect** | AI has more knowledge | Text is modified |
| **Use Case** | Background info | Style, corrections |

### When to Use Each

**Use World Books for:**
- Character backstories
- Location descriptions
- Historical events
- Technical specifications
- Game rules and mechanics
- Research context

**Use Chat Dictionaries for:**
- Speech patterns
- Name consistency
- Technical term expansion
- Censorship/filtering
- Style enforcement
- Quick corrections

### Using Both Together

World Books and Chat Dictionaries complement each other perfectly:

1. **World Book** provides context: "The Kingdom of Aldora uses gold dragons as currency"
2. **Chat Dictionary** ensures consistency: "gold pieces" → "gold dragons"
3. **Result**: AI knows about the currency AND uses the correct term

Example workflow:
```
World Book Entry:
Keys: ["currency", "money", "gold"]
Content: "Aldora's currency is the gold dragon, worth 10 silver stags or 100 copper pennies."

Chat Dictionary Entry:
Key: "gold pieces"
Replacement: "gold dragons"

User: "How much do gold pieces cost?"
AI (with context): "Gold dragons, the primary currency of Aldora, are valued at..."
```

## Summary

World Books are an essential tool for maintaining consistency and depth in your AI conversations. They work invisibly to provide the AI with the context it needs to understand your world, characters, and scenarios.

Key takeaways:
- Inject contextual information based on keywords
- Work alongside character definitions
- Respect token budgets automatically
- Can be shared across conversations
- Complement Chat Dictionaries for complete control

Whether you're building a fantasy world, maintaining technical documentation, or managing complex character relationships, World Books ensure your AI always has the right information at the right time.

For technical implementation details, see the developer documentation in the Development folder.