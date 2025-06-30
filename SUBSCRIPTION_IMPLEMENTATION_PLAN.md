# Website Subscription & URL Monitoring Implementation Plan

## Overview
A comprehensive system for monitoring both RSS/Atom feeds and raw URLs for changes, with automatic content ingestion into tldw_chatbook.

## Core Architecture

### 1. Unified Subscription Model
Both RSS feeds and raw URLs are treated as "subscriptions" with different monitoring strategies:

```
Subscription Types:
├── RSS/Atom Feeds
│   ├── Standard RSS 2.0
│   ├── Atom 1.0
│   └── Custom feed formats
└── URL Watchlist
    ├── Single pages
    ├── URL patterns (with wildcards)
    └── URL lists (bulk import)
```

### 2. Database Schema

```sql
-- Unified subscription table
CREATE TABLE subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('rss', 'atom', 'url', 'url_list')),
    source TEXT NOT NULL, -- RSS URL or watched URL
    description TEXT,
    
    -- Monitoring configuration
    check_frequency INTEGER DEFAULT 3600, -- seconds
    last_checked DATETIME,
    last_error TEXT,
    error_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT 1,
    
    -- Processing options
    extraction_method TEXT DEFAULT 'auto', -- 'full', 'diff', 'metadata'
    processing_options TEXT, -- JSON: chunking, summarization, etc.
    auto_ingest BOOLEAN DEFAULT 0, -- Auto-accept new content
    
    -- Change detection for URLs
    change_threshold FLOAT DEFAULT 0.1, -- 10% change threshold
    ignore_selectors TEXT, -- CSS selectors to ignore (ads, timestamps)
    
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

-- Items from subscriptions (RSS entries or URL changes)
CREATE TABLE subscription_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id INTEGER NOT NULL,
    
    -- Common fields
    url TEXT NOT NULL,
    title TEXT,
    content_hash TEXT, -- For change detection
    published_date DATETIME,
    
    -- Status tracking
    status TEXT DEFAULT 'new' CHECK(status IN ('new', 'reviewed', 'ingested', 'ignored')),
    media_id INTEGER, -- Links to Media table when ingested
    
    -- Change tracking for URLs
    previous_hash TEXT,
    change_percentage FLOAT,
    diff_summary TEXT,
    
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id),
    FOREIGN KEY (media_id) REFERENCES Media(id),
    UNIQUE(subscription_id, url, content_hash)
);

-- URL monitoring snapshots
CREATE TABLE url_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id INTEGER NOT NULL,
    url TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    extracted_content TEXT,
    raw_html TEXT, -- Optional, for diff generation
    headers TEXT, -- JSON: last-modified, etag, etc.
    created_at DATETIME NOT NULL,
    
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id)
);

-- Create indexes
CREATE INDEX idx_subscriptions_active ON subscriptions(is_active, last_checked);
CREATE INDEX idx_subscription_items_status ON subscription_items(subscription_id, status);
CREATE INDEX idx_url_snapshots_lookup ON url_snapshots(subscription_id, url, created_at);
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Database Layer (`DB/Subscriptions_DB.py`)
```python
class SubscriptionsDB:
    """Database operations for subscription management"""
    
    def add_subscription(self, name, type, source, options=None):
        """Add RSS feed or URL to monitor"""
        
    def get_pending_checks(self, limit=10):
        """Get subscriptions due for checking"""
        
    def record_check_result(self, sub_id, items, error=None):
        """Update check status and add new items"""
        
    def get_new_items(self, subscription_id=None):
        """Get items pending review"""
        
    def mark_item_status(self, item_id, status, media_id=None):
        """Update item processing status"""
```

#### 1.2 Monitoring Engine (`Web_Scraping/Subscription_Monitor/`)

**Feed Monitor (`feed_monitor.py`)**:
```python
class FeedMonitor:
    """RSS/Atom feed monitoring"""
    
    async def check_feed(self, subscription):
        """Parse feed and extract new entries"""
        # Use feedparser
        # Extract: title, url, published date, content
        # Return list of new items
        
    def extract_content_from_entry(self, entry):
        """Extract relevant content from feed entry"""
```

**URL Monitor (`url_monitor.py`)**:
```python
class URLMonitor:
    """Raw URL change monitoring"""
    
    async def check_url(self, subscription):
        """Check URL for changes"""
        # Fetch current content
        # Compare with last snapshot
        # Calculate change percentage
        # Return change info if threshold met
        
    def calculate_change_percentage(self, old_content, new_content):
        """Smart diff calculation"""
        # Use difflib or similar
        # Ignore dynamic elements
        # Return percentage changed
        
    def extract_meaningful_diff(self, old_content, new_content):
        """Generate human-readable diff summary"""
```

**Subscription Manager (`subscription_manager.py`)**:
```python
class SubscriptionManager:
    """Orchestrates all subscription monitoring"""
    
    def __init__(self, db, feed_monitor, url_monitor):
        self.monitors = {
            'rss': feed_monitor,
            'atom': feed_monitor,
            'url': url_monitor,
            'url_list': url_monitor
        }
        
    async def check_subscriptions(self):
        """Main checking loop"""
        # Get pending subscriptions
        # Route to appropriate monitor
        # Handle results
        # Update database
```

### Phase 2: User Interface (Week 2-3)

#### 2.1 Subscriptions Tab (`UI/Subscriptions_Window.py`)

**Layout Structure**:
```
┌─────────────────────────────────────────────────────────────┐
│ Subscriptions                                               │
├──────────────┬──────────────────────────────────────────────┤
│              │  ┌─────────────────────────────────────────┐ │
│ [+ Add New]  │  │ Active Subscriptions                    │ │
│              │  ├─────────────────────────────────────────┤ │
│ Type Filter: │  │ 📰 Tech News RSS                        │ │
│ [All Types]  │  │    Last: 2h ago | New: 5 | Error: 0    │ │
│              │  ├─────────────────────────────────────────┤ │
│ Status:      │  │ 🔗 Product Docs URL                     │ │
│ ☑ Active     │  │    Last: 1h ago | Changed: 15%         │ │
│ ☐ Error      │  ├─────────────────────────────────────────┤ │
│ ☐ Paused     │  │ 📋 URL List: Competitor Sites           │ │
│              │  │    Last: 3h ago | New: 2 | Changed: 3  │ │
│              │  └─────────────────────────────────────────┘ │
│ Quick Actions│  ┌─────────────────────────────────────────┐ │
│ [Refresh All]│  │ New Items to Review (12)          [▼]  │ │
│ [Import OPML]│  ├─────────────────────────────────────────┤ │
│ [Export]     │  │ ☐ New Blog Post: "AI Trends 2025"      │ │
│              │  │   From: Tech News RSS | 2 hours ago    │ │
│              │  ├─────────────────────────────────────────┤ │
│              │  │ ☐ Page Updated: Documentation v2.5     │ │
│              │  │   From: Product Docs | 45% changed     │ │
│              │  └─────────────────────────────────────────┘ │
│              │  [Accept Selected] [Ignore] [Mark Reviewed] │
└──────────────┴──────────────────────────────────────────────┘
```

#### 2.2 Add Subscription Dialog (`UI/Dialogs/AddSubscriptionDialog.py`)

**Dynamic Form Based on Type**:
- **RSS/Atom**: URL, check frequency, auto-ingest option
- **Single URL**: URL, selectors to ignore, change threshold
- **URL List**: Text area for URLs, bulk import from file

#### 2.3 Item Review Interface

**Features**:
- Preview pane for selected items
- Diff viewer for URL changes
- Bulk selection tools
- Quick filters (new, changed, by source)
- Direct ingestion with processing options

### Phase 3: Background Service (Week 3)

#### 3.1 Subscription Service (`Services/subscription_service.py`)

```python
class SubscriptionService:
    """Background service for monitoring subscriptions"""
    
    def __init__(self, app, check_interval=300):
        self.app = app
        self.check_interval = check_interval
        self.is_running = False
        
    def start(self):
        """Start background monitoring"""
        self.is_running = True
        self.app.set_interval(
            self.check_interval,
            self._check_subscriptions_callback
        )
        
    async def _check_subscriptions_callback(self):
        """Periodic check callback"""
        if not self.is_running:
            return
            
        # Get subscriptions due for checking
        # Run checks in parallel (limited concurrency)
        # Post results to UI via events
        # Handle errors gracefully
```

#### 3.2 Event Integration

```python
# Event types
class NewSubscriptionItems(Message):
    """New items available for review"""
    def __init__(self, items: List[Dict]):
        self.items = items
        
class SubscriptionError(Message):
    """Subscription check failed"""
    def __init__(self, subscription_id: int, error: str):
        self.subscription_id = subscription_id
        self.error = error
```

### Phase 4: Processing Pipeline Integration (Week 4)

#### 4.1 Ingestion Adapter

```python
class SubscriptionIngestionAdapter:
    """Adapts subscription items to existing ingestion pipeline"""
    
    async def ingest_item(self, item, processing_options):
        """Process a subscription item"""
        if item.is_new_article:
            # Use existing article scraper
            return await self._ingest_article(item)
        elif item.is_changed_page:
            # Extract changed content
            return await self._ingest_changes(item)
            
    async def _ingest_article(self, item):
        """Ingest new article using existing pipeline"""
        # Call scrape_article()
        # Apply processing options
        # Store in Media DB
        
    async def _ingest_changes(self, item):
        """Ingest page changes"""
        # Extract meaningful changes
        # Create update record
        # Link to original if exists
```

### Phase 5: Advanced Features (Optional)

#### 5.1 Smart Filtering

```python
class SubscriptionFilter:
    """Rule-based filtering for subscription items"""
    
    def __init__(self):
        self.rules = []
        
    def add_rule(self, field, operator, value, action):
        """Add filtering rule"""
        # field: title, url, content, change_percentage
        # operator: contains, matches, greater_than, etc.
        # action: auto_accept, auto_ignore, tag
        
    def apply_filters(self, items):
        """Apply all rules to items"""
        # Return items with actions applied
```

#### 5.2 Change Intelligence

```python
class ChangeAnalyzer:
    """Intelligent change detection and summarization"""
    
    def analyze_changes(self, old_content, new_content):
        """Analyze what changed"""
        # Identify change types:
        # - New sections added
        # - Content updates
        # - Removals
        # - Structural changes
        
    def generate_change_summary(self, analysis):
        """Human-readable change summary"""
        # "Added 2 new sections about X"
        # "Updated pricing information"
        # "Removed deprecated API docs"
```

## Configuration Options

### In `config.toml`:
```toml
[subscriptions]
enabled = true
default_check_interval = 3600  # 1 hour
max_concurrent_checks = 5
timeout_seconds = 30

[subscriptions.rss]
auto_ingest = false
extract_full_content = true

[subscriptions.url_monitor]
default_change_threshold = 0.1  # 10%
store_snapshots = true
max_snapshots_per_url = 10

[subscriptions.notifications]
enabled = true
new_items_threshold = 5
```

## Error Handling & Edge Cases

### 1. Feed/URL Errors
- Network timeouts
- 404/403 errors
- Malformed RSS
- Authentication required
- Rate limiting

### 2. Change Detection Challenges
- Dynamic timestamps
- Random advertisement content
- Session-specific data
- JavaScript-rendered content
- Cookie banners

### 3. Performance Considerations
- Large feeds (1000+ items)
- Frequent updates (< 5 min)
- Many subscriptions (100+)
- Large page sizes

## Testing Strategy

### 1. Unit Tests
- Feed parsing with various formats
- Change detection algorithms
- URL validation and normalization
- Filter rule application

### 2. Integration Tests
- Full monitoring cycle
- Database operations
- Event propagation
- Error recovery

### 3. Performance Tests
- Concurrent subscription checking
- Large feed handling
- Memory usage with many snapshots

## Future Enhancements

### 1. AI-Powered Features
- Smart change summarization using LLMs
- Automatic categorization of new content
- Relevance scoring based on user history

### 2. Advanced Monitoring
- JavaScript-rendered page support
- API endpoint monitoring
- Custom extraction rules per site
- WebSocket/real-time monitoring

### 3. Collaboration Features
- Share subscription lists
- Community-curated feeds
- Change annotations
- Subscription recommendations

## Migration Path

### From Current System:
1. Existing ingested articles remain unchanged
2. URLs can be retroactively linked to subscriptions
3. Import browser bookmarks as URL watchlist
4. Convert existing scraped sites to monitored subscriptions

This implementation provides a robust foundation for both RSS feed monitoring and raw URL change tracking, with room for future enhancements and integration with the planned TLDW server audio/video processing.