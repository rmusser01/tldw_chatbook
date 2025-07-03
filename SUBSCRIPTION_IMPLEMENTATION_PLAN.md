# Website Subscription & URL Monitoring Implementation Plan

## Overview
A comprehensive system for monitoring both RSS/Atom feeds and raw URLs for changes, with automatic content ingestion into tldw_chatbook.

## Core Architecture

### 1. Unified Subscription Model (Enhanced)
Both RSS feeds and raw URLs are treated as "subscriptions" with different monitoring strategies:

```
Subscription Types:
├── Feed-based
│   ├── RSS 2.0 (Standard blog/news feeds)
│   ├── Atom 1.0 (Technical documentation)
│   ├── JSON Feed 1.1 (Modern feeds)
│   └── Podcast RSS (Audio/video with enclosures)
├── URL Monitoring
│   ├── Single pages (Track specific pages)
│   ├── URL patterns (Wildcards: /blog/*/comments)
│   ├── URL lists (Bulk import from CSV/TXT)
│   └── Sitemap.xml (Monitor entire sites)
└── API Endpoints
    ├── REST APIs (JSON responses)
    ├── GraphQL endpoints
    └── Webhook receivers
```

### 2. Database Schema (Enhanced)

```sql
-- Unified subscription table with enhanced features
CREATE TABLE subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('rss', 'atom', 'json_feed', 'url', 'url_list', 'podcast', 'sitemap', 'api')),
    source TEXT NOT NULL, -- RSS URL, watched URL, or API endpoint
    description TEXT,
    
    -- Organization
    tags TEXT, -- Comma-separated tags for categorization
    priority INTEGER DEFAULT 3 CHECK(priority BETWEEN 1 AND 5), -- 1=lowest, 5=highest
    folder TEXT, -- Folder/group organization
    
    -- Monitoring configuration
    check_frequency INTEGER DEFAULT 3600, -- seconds
    last_checked DATETIME,
    last_successful_check DATETIME, -- Track successful checks separately
    last_error TEXT,
    error_count INTEGER DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT 1,
    is_paused BOOLEAN DEFAULT 0, -- User-initiated pause
    auto_pause_threshold INTEGER DEFAULT 10, -- Auto-pause after N failures
    
    -- Authentication & Headers
    auth_config TEXT, -- JSON: {type: 'basic'|'bearer'|'oauth', credentials: {...}}
    custom_headers TEXT, -- JSON: additional headers for requests
    rate_limit_config TEXT, -- JSON: {requests_per_minute: 60, cooldown: 5}
    
    -- Processing options
    extraction_method TEXT DEFAULT 'auto', -- 'full', 'diff', 'metadata', 'template'
    extraction_rules TEXT, -- JSON: custom extraction patterns/selectors
    processing_options TEXT, -- JSON: chunking, summarization, etc.
    auto_ingest BOOLEAN DEFAULT 0, -- Auto-accept new content
    notification_config TEXT, -- JSON: webhook URLs, email settings
    
    -- Change detection for URLs
    change_threshold FLOAT DEFAULT 0.1, -- 10% change threshold
    ignore_selectors TEXT, -- CSS selectors to ignore (ads, timestamps)
    
    -- Performance optimization
    etag TEXT, -- For conditional requests
    last_modified TEXT, -- For If-Modified-Since
    
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    
    INDEX idx_priority_active (priority DESC, is_active, is_paused),
    INDEX idx_tags (tags),
    INDEX idx_folder (folder)
);

-- Items from subscriptions (RSS entries or URL changes) with enhanced metadata
CREATE TABLE subscription_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id INTEGER NOT NULL,
    
    -- Common fields
    url TEXT NOT NULL,
    title TEXT,
    content_hash TEXT, -- For change detection
    published_date DATETIME,
    
    -- Enhanced metadata
    author TEXT,
    categories TEXT, -- JSON array of categories/tags from feed
    enclosures TEXT, -- JSON: podcast/media attachments
    extracted_data TEXT, -- JSON: custom extracted fields
    
    -- Status tracking
    status TEXT DEFAULT 'new' CHECK(status IN ('new', 'reviewed', 'ingested', 'ignored', 'error')),
    media_id INTEGER, -- Links to Media table when ingested
    processing_error TEXT, -- Error details if processing failed
    
    -- Change tracking for URLs
    previous_hash TEXT,
    change_percentage FLOAT,
    diff_summary TEXT,
    change_type TEXT CHECK(change_type IN ('content', 'metadata', 'structural', 'new', 'removed')),
    
    -- Deduplication
    canonical_url TEXT, -- Cleaned URL for dedup
    duplicate_of INTEGER, -- Reference to original item if duplicate
    
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id),
    FOREIGN KEY (media_id) REFERENCES Media(id),
    FOREIGN KEY (duplicate_of) REFERENCES subscription_items(id),
    UNIQUE(subscription_id, url, content_hash),
    INDEX idx_status_created (subscription_id, status, created_at),
    INDEX idx_canonical_url (canonical_url)
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

-- Create indexes (already included inline above)

-- Subscription statistics for health monitoring
CREATE TABLE subscription_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id INTEGER NOT NULL,
    date DATE NOT NULL,
    
    -- Daily statistics
    checks_performed INTEGER DEFAULT 0,
    successful_checks INTEGER DEFAULT 0,
    new_items_found INTEGER DEFAULT 0,
    items_ingested INTEGER DEFAULT 0,
    errors_encountered INTEGER DEFAULT 0,
    
    -- Performance metrics
    avg_response_time_ms INTEGER,
    total_bytes_transferred INTEGER,
    
    created_at DATETIME NOT NULL,
    
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id),
    UNIQUE(subscription_id, date),
    INDEX idx_date (date)
);

-- Smart filters for automatic processing
CREATE TABLE subscription_filters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id INTEGER, -- NULL for global filters
    name TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    
    -- Filter conditions (JSON)
    conditions TEXT NOT NULL, -- {field: 'title', operator: 'contains', value: 'Breaking'}
    
    -- Actions
    action TEXT NOT NULL CHECK(action IN ('auto_ingest', 'auto_ignore', 'tag', 'priority', 'notify')),
    action_params TEXT, -- JSON: {tag: 'urgent', priority: 5}
    
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id)
);

-- Subscription templates for quick setup
CREATE TABLE subscription_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT, -- 'news', 'tech', 'documentation', etc.
    
    -- Template configuration
    type TEXT NOT NULL,
    check_frequency INTEGER,
    extraction_method TEXT,
    extraction_rules TEXT, -- JSON
    processing_options TEXT, -- JSON
    auth_config_template TEXT, -- JSON with placeholders
    
    -- Popularity tracking
    usage_count INTEGER DEFAULT 0,
    
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Database Layer (`DB/Subscriptions_DB.py`) - Enhanced
```python
class SubscriptionsDB:
    """Enhanced database operations for subscription management"""
    
    # Core subscription management
    def add_subscription(self, name, type, source, tags=None, priority=3, 
                        folder=None, auth_config=None, **kwargs):
        """Add subscription with enhanced metadata"""
        
    def get_pending_checks(self, limit=10, priority_order=True):
        """Get subscriptions due for checking, ordered by priority"""
        
    def get_subscriptions_by_tag(self, tag):
        """Filter subscriptions by tag"""
        
    def get_subscriptions_by_folder(self, folder):
        """Get all subscriptions in a folder"""
        
    # Check results and error handling
    def record_check_result(self, sub_id, items, error=None, stats=None):
        """Update check status with performance metrics"""
        
    def record_check_error(self, sub_id, error, should_pause=False):
        """Record error with auto-pause logic"""
        
    def reset_subscription_errors(self, sub_id):
        """Reset error count after successful check"""
        
    # Item management
    def get_new_items(self, subscription_id=None, status='new', limit=100):
        """Get items with filtering and pagination"""
        
    def mark_item_status(self, item_id, status, media_id=None, error=None):
        """Update item status with error tracking"""
        
    def find_duplicate_items(self, item_url, item_hash):
        """Check for existing duplicates"""
        
    def bulk_update_items(self, item_ids, status):
        """Efficient bulk status updates"""
        
    # Statistics and health monitoring
    def update_subscription_stats(self, sub_id, date, stats):
        """Record daily statistics"""
        
    def get_subscription_health(self, sub_id, days=30):
        """Get health metrics for dashboard"""
        
    def get_failing_subscriptions(self, threshold=5):
        """Find subscriptions needing attention"""
        
    # Filters and templates
    def add_filter(self, name, conditions, action, subscription_id=None):
        """Add smart filter rule"""
        
    def get_active_filters(self, subscription_id=None):
        """Get filters for processing"""
        
    def save_template(self, name, config, category=None):
        """Save subscription template"""
        
    def get_templates(self, category=None):
        """Retrieve available templates"""
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

## Enhanced Features & Improvements

### Smart Scheduling System
```python
class AdaptiveScheduler:
    """Intelligent scheduling based on update patterns"""
    
    def calculate_next_check(self, subscription, history):
        """Adaptive frequency based on:
        - Historical update patterns
        - Time of day/week patterns
        - Content type (news vs documentation)
        - Recent change frequency
        """
        
    def prioritize_checks(self, pending_subscriptions):
        """Priority-based checking order:
        - User-defined priority (1-5)
        - Time since last update
        - Historical reliability
        - Resource availability
        """
```

### Content Deduplication
```python
class DeduplicationEngine:
    """Cross-feed content deduplication"""
    
    def find_duplicates(self, new_item, existing_items):
        """Detect duplicates using:
        - URL canonicalization
        - Title similarity (fuzzy matching)
        - Content hash comparison
        - Publishing time proximity
        """
        
    def merge_duplicate_metadata(self, items):
        """Combine metadata from multiple sources"""
```

### Enhanced Error Recovery
```python
class ErrorRecovery:
    """Intelligent error handling and recovery"""
    
    def handle_error(self, subscription, error):
        """Error-specific strategies:
        - Network timeout: Exponential backoff
        - 404: Check redirects, mark dead
        - 401/403: Prompt for auth update
        - 429: Respect rate limits
        - Parse errors: Try alternate parsers
        """
        
    def should_auto_pause(self, subscription):
        """Auto-pause logic based on:
        - Consecutive failure count
        - Error types encountered
        - Time since last success
        """
```

### Performance Optimizations
```python
class PerformanceOptimizer:
    """Resource-efficient checking"""
    
    async def conditional_request(self, subscription):
        """Use HTTP caching headers:
        - ETag for content changes
        - If-Modified-Since for timestamps
        - Accept-Encoding for compression
        """
        
    def batch_similar_requests(self, subscriptions):
        """Group requests by:
        - Domain (connection reuse)
        - Authentication type
        - Rate limit buckets
        """
```

## Configuration Options

### In `config.toml` (Enhanced):
```toml
[subscriptions]
enabled = true
default_check_interval = 3600  # 1 hour
max_concurrent_checks = 5
timeout_seconds = 30
default_priority = 3
auto_pause_after_failures = 10

# Performance settings
[subscriptions.performance]
use_conditional_requests = true
connection_pool_size = 10
max_retries = 3
retry_backoff_factor = 2
respect_rate_limits = true
default_rate_limit_rpm = 60

# Feed processing
[subscriptions.rss]
auto_ingest = false
extract_full_content = true
autodiscover_feeds = true
parse_podcast_enclosures = true
max_items_per_check = 50

# URL monitoring
[subscriptions.url_monitor]
default_change_threshold = 0.1  # 10%
store_snapshots = true
max_snapshots_per_url = 10
use_visual_diff = true
ignore_common_dynamic_selectors = true

# Content processing
[subscriptions.processing]
enable_deduplication = true
similarity_threshold = 0.85
extract_structured_data = true
auto_categorize = true
summarize_changes = true

# Notifications
[subscriptions.notifications]
enabled = true
new_items_threshold = 5
digest_frequency = "daily"
webhook_timeout = 5000

# Templates
[subscriptions.templates]
enable_community_templates = true
template_repository = "https://templates.tldw.example.com"
cache_templates_days = 7
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