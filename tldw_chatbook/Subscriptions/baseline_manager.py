# baseline_manager.py
# Description: Baseline management system for content change detection
#
# This module provides intelligent change detection by maintaining baselines
# and comparing new content against them using multiple strategies:
# - Content hashing for exact matches
# - Structural analysis for layout changes
# - Semantic similarity for meaning changes
# - Token counting for length changes
#
# Imports
import hashlib
import json
import re
import zlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
#
# Third-Party Imports
from bs4 import BeautifulSoup, NavigableString
from loguru import logger
#
# Local Imports
from ..DB.Subscriptions_DB import SubscriptionsDB
from ..Metrics.metrics_logger import log_histogram, log_counter
#
########################################################################################################################
#
# Data Classes
#
########################################################################################################################

@dataclass
class ChangeReport:
    """Report of changes detected between baseline and new content."""
    has_changed: bool
    change_percentage: float
    change_type: str  # 'content', 'structural', 'semantic', 'new', 'removed'
    summary: str
    details: Dict[str, Any]
    should_update_baseline: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'has_changed': self.has_changed,
            'change_percentage': self.change_percentage,
            'change_type': self.change_type,
            'summary': self.summary,
            'details': self.details,
            'should_update_baseline': self.should_update_baseline
        }


@dataclass
class ContentBaseline:
    """Baseline snapshot of content for comparison."""
    subscription_id: int
    url: str
    content_hash: str
    structural_hash: str
    extracted_text: str
    token_count: int
    key_elements: Dict[str, Any]
    created_at: datetime
    compressed_content: Optional[bytes] = None
    
    @classmethod
    def from_content(cls, subscription_id: int, url: str, 
                    content: str, extract_structure: bool = True) -> 'ContentBaseline':
        """Create baseline from raw content."""
        # Extract text and structure
        extractor = ContentAnalyzer()
        extracted_text = extractor.extract_text(content)
        
        # Calculate hashes
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        structural_hash = extractor.calculate_structural_hash(content) if extract_structure else ""
        
        # Count tokens (simple word-based approximation)
        token_count = len(extracted_text.split())
        
        # Extract key elements
        key_elements = extractor.extract_key_elements(content)
        
        # Compress content for storage
        compressed = zlib.compress(content.encode('utf-8'))
        
        return cls(
            subscription_id=subscription_id,
            url=url,
            content_hash=content_hash,
            structural_hash=structural_hash,
            extracted_text=extracted_text,
            token_count=token_count,
            key_elements=key_elements,
            created_at=datetime.now(timezone.utc),
            compressed_content=compressed
        )


########################################################################################################################
#
# Content Analysis
#
########################################################################################################################

class ContentAnalyzer:
    """Analyze content structure and extract key elements."""
    
    def extract_text(self, html_content: str) -> str:
        """Extract clean text from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator=' ')
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return html_content
    
    def calculate_structural_hash(self, html_content: str) -> str:
        """Calculate hash of HTML structure (tags only, no content)."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Build structure string
            structure_parts = []
            for element in soup.find_all(True):  # All tags
                tag_info = f"<{element.name}"
                # Include key attributes that affect structure
                for attr in ['class', 'id', 'role']:
                    if element.get(attr):
                        tag_info += f" {attr}='{element.get(attr)}'"
                tag_info += ">"
                structure_parts.append(tag_info)
            
            structure_string = ''.join(structure_parts)
            return hashlib.md5(structure_string.encode('utf-8')).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating structural hash: {str(e)}")
            return ""
    
    def extract_key_elements(self, html_content: str) -> Dict[str, Any]:
        """Extract key elements like headers, links, images."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            elements = {
                'title': soup.find('title').get_text() if soup.find('title') else None,
                'h1_count': len(soup.find_all('h1')),
                'h2_count': len(soup.find_all('h2')),
                'paragraph_count': len(soup.find_all('p')),
                'link_count': len(soup.find_all('a', href=True)),
                'image_count': len(soup.find_all('img', src=True)),
                'headers': [h.get_text()[:100] for h in soup.find_all(['h1', 'h2', 'h3'])[:10]],
                'meta_description': None
            }
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                elements['meta_description'] = meta_desc.get('content', '')
            
            return elements
            
        except Exception as e:
            logger.error(f"Error extracting key elements: {str(e)}")
            return {}


########################################################################################################################
#
# Change Detection
#
########################################################################################################################

class ChangeDetector:
    """Detect and analyze changes between baseline and new content."""
    
    def __init__(self, change_threshold: float = 0.1, 
                 semantic_threshold: float = 0.85):
        """
        Initialize change detector.
        
        Args:
            change_threshold: Minimum change percentage to consider significant
            semantic_threshold: Similarity threshold for semantic comparison
        """
        self.change_threshold = change_threshold
        self.semantic_threshold = semantic_threshold
        self.analyzer = ContentAnalyzer()
    
    def detect_changes(self, baseline: ContentBaseline, new_content: str,
                      ignore_selectors: List[str] = None) -> ChangeReport:
        """
        Detect changes between baseline and new content.
        
        Args:
            baseline: Previous content baseline
            new_content: New content to compare
            ignore_selectors: CSS selectors for elements to ignore
            
        Returns:
            ChangeReport with detailed change information
        """
        start_time = datetime.now(timezone.utc)
        
        # Apply ignore selectors if provided
        if ignore_selectors:
            new_content = self._apply_ignore_selectors(new_content, ignore_selectors)
        
        # Quick hash check
        new_hash = hashlib.sha256(new_content.encode('utf-8')).hexdigest()
        if new_hash == baseline.content_hash:
            return ChangeReport(
                has_changed=False,
                change_percentage=0.0,
                change_type='none',
                summary='No changes detected',
                details={'exact_match': True},
                should_update_baseline=False
            )
        
        # Extract new content features
        new_text = self.analyzer.extract_text(new_content)
        new_structural_hash = self.analyzer.calculate_structural_hash(new_content)
        new_token_count = len(new_text.split())
        new_key_elements = self.analyzer.extract_key_elements(new_content)
        
        # Analyze different types of changes
        structural_changed = new_structural_hash != baseline.structural_hash
        
        # Calculate text similarity
        text_similarity = self._calculate_text_similarity(baseline.extracted_text, new_text)
        text_changed = text_similarity < self.semantic_threshold
        
        # Calculate change percentage
        change_percentage = 1.0 - text_similarity
        
        # Token count change
        token_change_ratio = abs(new_token_count - baseline.token_count) / max(baseline.token_count, 1)
        
        # Determine change type
        if not baseline.extracted_text:
            change_type = 'new'
        elif not new_text:
            change_type = 'removed'
        elif structural_changed and text_changed:
            change_type = 'structural'
        elif text_changed:
            change_type = 'content'
        else:
            change_type = 'metadata'
        
        # Generate change summary
        summary = self._generate_change_summary(
            baseline, new_key_elements, change_percentage, token_change_ratio
        )
        
        # Compile detailed changes
        details = {
            'text_similarity': text_similarity,
            'structural_changed': structural_changed,
            'token_count_before': baseline.token_count,
            'token_count_after': new_token_count,
            'token_change_ratio': token_change_ratio,
            'key_elements_changed': self._compare_key_elements(
                baseline.key_elements, new_key_elements
            )
        }
        
        # Determine if baseline should be updated
        should_update = change_percentage >= self.change_threshold
        
        # Log metrics
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        log_histogram("change_detection_duration", duration, labels={
            "change_type": change_type,
            "has_changed": str(text_changed or structural_changed)
        })
        log_counter("change_detection_performed", labels={
            "change_type": change_type,
            "significant": str(should_update)
        })
        
        return ChangeReport(
            has_changed=text_changed or structural_changed,
            change_percentage=change_percentage,
            change_type=change_type,
            summary=summary,
            details=details,
            should_update_baseline=should_update
        )
    
    def _apply_ignore_selectors(self, content: str, ignore_selectors: List[str]) -> str:
        """Remove elements matching ignore selectors."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            for selector in ignore_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            return str(soup)
            
        except Exception as e:
            logger.error(f"Error applying ignore selectors: {str(e)}")
            return content
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not text1 or not text2:
            return 0.0 if (text1 or text2) else 1.0
        
        # Normalize texts
        text1_normalized = self._normalize_text(text1)
        text2_normalized = self._normalize_text(text2)
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, text1_normalized, text2_normalized).ratio()
        
        return similarity
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common noise patterns
        noise_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # Dates
            r'\d{1,2}:\d{2}',      # Times
            r'[a-f0-9]{32,}',      # Hashes
            r'\b\d+\s*(views?|comments?|likes?)\b',  # Social metrics
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text)
        
        return text.strip()
    
    def _compare_key_elements(self, old_elements: Dict[str, Any], 
                             new_elements: Dict[str, Any]) -> Dict[str, Any]:
        """Compare key elements between versions."""
        changes = {}
        
        # Compare counts
        for key in ['h1_count', 'h2_count', 'paragraph_count', 'link_count', 'image_count']:
            old_val = old_elements.get(key, 0)
            new_val = new_elements.get(key, 0)
            if old_val != new_val:
                changes[key] = {'before': old_val, 'after': new_val}
        
        # Compare headers
        old_headers = set(old_elements.get('headers', []))
        new_headers = set(new_elements.get('headers', []))
        
        if old_headers != new_headers:
            changes['headers'] = {
                'added': list(new_headers - old_headers),
                'removed': list(old_headers - new_headers)
            }
        
        return changes
    
    def _generate_change_summary(self, baseline: ContentBaseline, 
                                new_elements: Dict[str, Any],
                                change_percentage: float,
                                token_change_ratio: float) -> str:
        """Generate human-readable change summary."""
        parts = []
        
        # Overall change level
        if change_percentage > 0.5:
            parts.append("Major changes detected")
        elif change_percentage > 0.2:
            parts.append("Moderate changes detected")
        elif change_percentage > 0:
            parts.append("Minor changes detected")
        else:
            parts.append("No significant changes")
        
        # Token count change
        if token_change_ratio > 0.2:
            old_tokens = baseline.token_count
            new_tokens = new_elements.get('token_count', old_tokens)
            if new_tokens > old_tokens:
                parts.append(f"content expanded by {token_change_ratio:.0%}")
            else:
                parts.append(f"content reduced by {token_change_ratio:.0%}")
        
        # Structural changes
        old_elements = baseline.key_elements
        if old_elements.get('h1_count', 0) != new_elements.get('h1_count', 0):
            parts.append("heading structure changed")
        
        return ". ".join(parts) + "."


########################################################################################################################
#
# Baseline Manager
#
########################################################################################################################

class BaselineManager:
    """Manage content baselines for change detection."""
    
    def __init__(self, db: SubscriptionsDB, retention_days: int = 30):
        """
        Initialize baseline manager.
        
        Args:
            db: Subscriptions database instance
            retention_days: Days to retain baseline history
        """
        self.db = db
        self.retention_days = retention_days
        self.change_detector = ChangeDetector()
    
    async def create_baseline(self, subscription_id: int, url: str, 
                            content: str) -> ContentBaseline:
        """
        Create initial baseline for subscription.
        
        Args:
            subscription_id: Subscription ID
            url: Content URL
            content: Raw content
            
        Returns:
            Created baseline
        """
        baseline = ContentBaseline.from_content(subscription_id, url, content)
        
        # Store in database
        self._store_baseline(baseline)
        
        logger.info(f"Created baseline for subscription {subscription_id}")
        log_counter("baseline_created", labels={
            "subscription_id": str(subscription_id)
        })
        
        return baseline
    
    async def check_for_changes(self, subscription_id: int, url: str,
                               new_content: str, 
                               ignore_selectors: List[str] = None) -> ChangeReport:
        """
        Check for changes against existing baseline.
        
        Args:
            subscription_id: Subscription ID
            url: Content URL
            new_content: New content to check
            ignore_selectors: Elements to ignore
            
        Returns:
            Change report
        """
        # Get existing baseline
        baseline = self._get_latest_baseline(subscription_id, url)
        
        if not baseline:
            # No baseline exists, create one
            baseline = await self.create_baseline(subscription_id, url, new_content)
            return ChangeReport(
                has_changed=True,
                change_percentage=1.0,
                change_type='new',
                summary='First time checking this content',
                details={'first_check': True},
                should_update_baseline=False
            )
        
        # Get subscription config for thresholds
        subscription = self.db.get_subscription(subscription_id)
        change_threshold = subscription.get('change_threshold', 0.1) if subscription else 0.1
        
        # Configure detector with subscription settings
        self.change_detector.change_threshold = change_threshold
        
        # Detect changes
        report = self.change_detector.detect_changes(
            baseline, new_content, ignore_selectors
        )
        
        # Update baseline if significant changes
        if report.should_update_baseline:
            await self.update_baseline(subscription_id, url, new_content)
        
        return report
    
    async def update_baseline(self, subscription_id: int, url: str, 
                            new_content: str) -> ContentBaseline:
        """Update baseline with new content."""
        # Create new baseline
        new_baseline = ContentBaseline.from_content(subscription_id, url, new_content)
        
        # Store new baseline
        self._store_baseline(new_baseline)
        
        # Clean up old baselines
        self._cleanup_old_baselines(subscription_id)
        
        logger.info(f"Updated baseline for subscription {subscription_id}")
        log_counter("baseline_updated", labels={
            "subscription_id": str(subscription_id)
        })
        
        return new_baseline
    
    def get_baseline_history(self, subscription_id: int, 
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Get baseline history for subscription."""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT id, url, content_hash, created_at, token_count
            FROM url_snapshots
            WHERE subscription_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (subscription_id, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def _get_latest_baseline(self, subscription_id: int, url: str) -> Optional[ContentBaseline]:
        """Get most recent baseline for URL."""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT * FROM url_snapshots
            WHERE subscription_id = ? AND url = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (subscription_id, url))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        # Decompress content if stored
        content = ""
        if row['raw_html']:
            try:
                content = zlib.decompress(row['raw_html']).decode('utf-8')
            except:
                content = row['raw_html']
        
        # Parse key elements
        key_elements = {}
        if row['key_elements']:
            try:
                key_elements = json.loads(row['key_elements'])
            except:
                pass
        
        return ContentBaseline(
            subscription_id=row['subscription_id'],
            url=row['url'],
            content_hash=row['content_hash'],
            structural_hash=row['structural_hash'] or "",
            extracted_text=row['extracted_content'] or "",
            token_count=row['token_count'] or 0,
            key_elements=key_elements,
            created_at=datetime.fromisoformat(row['created_at'])
        )
    
    def _store_baseline(self, baseline: ContentBaseline):
        """Store baseline in database."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Prepare compressed content
            compressed_html = baseline.compressed_content or b''
            
            cursor.execute("""
                INSERT INTO url_snapshots 
                (subscription_id, url, content_hash, extracted_content, raw_html,
                 structural_hash, token_count, key_elements, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                baseline.subscription_id,
                baseline.url,
                baseline.content_hash,
                baseline.extracted_text,
                compressed_html,
                baseline.structural_hash,
                baseline.token_count,
                json.dumps(baseline.key_elements),
                baseline.created_at.isoformat()
            ))
    
    def _cleanup_old_baselines(self, subscription_id: int):
        """Remove baselines older than retention period."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM url_snapshots
                WHERE subscription_id = ? 
                AND created_at < ?
                AND id NOT IN (
                    SELECT id FROM url_snapshots
                    WHERE subscription_id = ?
                    ORDER BY created_at DESC
                    LIMIT 5
                )
            """, (subscription_id, cutoff_date.isoformat(), subscription_id))
            
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old baselines for subscription {subscription_id}")


# End of baseline_manager.py