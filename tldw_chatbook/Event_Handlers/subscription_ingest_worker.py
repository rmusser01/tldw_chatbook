# subscription_ingest_worker.py
# Description: Worker for ingesting subscription items into the media database
#
# This module bridges subscription items with the existing media ingestion pipeline,
# handling the processing and storage of content from RSS feeds, URLs, and other sources.
#
# Imports
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
#
# Third-Party Imports
from loguru import logger
from textual.worker import Worker
#
# Local Imports
from ..DB.Subscriptions_DB import SubscriptionsDB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Subscriptions.content_processor import ContentProcessor
from ..Metrics.metrics_logger import log_histogram, log_counter
#
########################################################################################################################
#
# Worker Functions
#
########################################################################################################################

async def process_subscription_item(item: Dict[str, Any], subscription: Dict[str, Any],
                                  media_db: MediaDatabase, subscriptions_db: SubscriptionsDB,
                                  content_processor: Optional[ContentProcessor] = None) -> tuple[bool, Optional[int], str]:
    """
    Process a single subscription item for ingestion.
    
    Args:
        item: Subscription item to process
        subscription: Subscription configuration
        media_db: Media database instance
        subscriptions_db: Subscriptions database instance
        content_processor: Optional content processor for analysis
        
    Returns:
        Tuple of (success, media_id, message)
    """
    start_time = datetime.now()
    
    try:
        # Check if already ingested
        if item.get('media_id'):
            return True, item['media_id'], "Item already ingested"
        
        # Determine if we should analyze
        analyze = False
        save_analysis_only = False
        
        if subscription.get('processing_options'):
            try:
                options = json.loads(subscription['processing_options'])
                analyze = options.get('analyze_before_ingest', False)
                save_analysis_only = options.get('save_analysis_only', False)
            except:
                pass
        
        # Process content
        if content_processor and (analyze or subscription.get('auto_ingest')):
            processed = await content_processor.process_item(
                item, 
                subscription,
                analyze=analyze,
                save_analysis_only=save_analysis_only
            )
        else:
            # Basic processing without analysis
            processed = {
                'url': item.get('url', ''),
                'title': item.get('title', 'Untitled'),
                'content': item.get('content', ''),
                'author': item.get('author', ''),
                'published_date': item.get('published_date'),
                'keywords': _extract_basic_keywords(item, subscription)
            }
        
        # Map subscription type to media type
        media_type = _map_subscription_to_media_type(subscription['type'])
        
        # Prepare media data
        media_data = {
            'url': processed['url'],
            'title': processed['title'],
            'type': media_type,
            'content': processed.get('content', ''),
            'author': processed.get('author', ''),
            'keywords': ', '.join(processed.get('keywords', [])),
            'custom_metadata': {
                'subscription_id': subscription['id'],
                'subscription_name': subscription['name'],
                'subscription_type': subscription['type'],
                'published_date': processed.get('published_date'),
                'source_url': subscription['source']
            }
        }
        
        # Add analysis if available
        if 'analysis' in processed:
            media_data['analysis_content'] = processed['analysis']
        
        # Add media to database
        media_id, media_uuid, message = media_db.add_media_with_keywords(
            url=media_data['url'],
            title=media_data['title'],
            type=media_data['type'],
            content=media_data['content'],
            keywords=media_data['keywords'],
            author=media_data['author'],
            analysis_content=media_data.get('analysis_content'),
            custom_metadata=json.dumps(media_data['custom_metadata'])
        )
        
        if media_id:
            # Update subscription item with media ID
            subscriptions_db.mark_item_status(
                item['id'],
                'ingested',
                media_id=media_id
            )
            
            # Log success
            duration = (datetime.now() - start_time).total_seconds()
            log_histogram("subscription_item_ingestion_duration", duration, labels={
                "subscription_type": subscription['type'],
                "analyzed": str(analyze)
            })
            log_counter("subscription_item_ingestion_count", labels={
                "subscription_type": subscription['type'],
                "status": "success"
            })
            
            logger.info(f"Successfully ingested item '{processed['title']}' as media ID {media_id}")
            return True, media_id, message
        else:
            # Log failure
            log_counter("subscription_item_ingestion_count", labels={
                "subscription_type": subscription['type'],
                "status": "failed"
            })
            
            return False, None, message
            
    except Exception as e:
        logger.error(f"Error processing subscription item: {e}")
        log_counter("subscription_item_ingestion_count", labels={
            "subscription_type": subscription.get('type', 'unknown'),
            "status": "error"
        })
        return False, None, str(e)


def _map_subscription_to_media_type(subscription_type: str) -> str:
    """
    Map subscription type to media type.
    
    Args:
        subscription_type: Subscription type
        
    Returns:
        Corresponding media type
    """
    mapping = {
        'rss': 'article',
        'atom': 'article',
        'json_feed': 'article',
        'url': 'webpage',
        'url_list': 'webpage',
        'podcast': 'audio',
        'sitemap': 'webpage',
        'api': 'data'
    }
    
    return mapping.get(subscription_type, 'article')


def _extract_basic_keywords(item: Dict[str, Any], subscription: Dict[str, Any]) -> List[str]:
    """
    Extract basic keywords without LLM analysis.
    
    Args:
        item: Item data
        subscription: Subscription data
        
    Returns:
        List of keywords
    """
    keywords = []
    
    # Add subscription tags
    if subscription.get('tags'):
        tags = [tag.strip() for tag in subscription['tags'].split(',') if tag.strip()]
        keywords.extend(tags)
    
    # Add categories
    if item.get('categories'):
        keywords.extend(item['categories'])
    
    # Add subscription name as keyword
    keywords.append(subscription['name'].lower())
    
    # Add type
    keywords.append(f"subscription-{subscription['type']}")
    
    return list(set(keywords))[:15]  # Dedupe and limit


class SubscriptionIngestWorker(Worker):
    """Textual worker for processing subscription items in the background."""
    
    def __init__(self, items: List[Dict[str, Any]], subscription: Dict[str, Any],
                 media_db: MediaDatabase, subscriptions_db: SubscriptionsDB,
                 llm_provider: Optional[str] = None, llm_model: Optional[str] = None):
        """
        Initialize worker.
        
        Args:
            items: List of items to process
            subscription: Subscription configuration
            media_db: Media database instance
            subscriptions_db: Subscriptions database instance
            llm_provider: Optional LLM provider for analysis
            llm_model: Optional LLM model for analysis
        """
        super().__init__()
        self.items = items
        self.subscription = subscription
        self.media_db = media_db
        self.subscriptions_db = subscriptions_db
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        
    async def run(self):
        """Process subscription items."""
        try:
            # Create content processor if LLM is configured
            content_processor = None
            if self.llm_provider and self.llm_model:
                content_processor = ContentProcessor(self.llm_provider, self.llm_model)
            
            success_count = 0
            error_count = 0
            
            for item in self.items:
                if self.is_cancelled:
                    break
                    
                # Process item
                success, media_id, message = await process_subscription_item(
                    item,
                    self.subscription,
                    self.media_db,
                    self.subscriptions_db,
                    content_processor
                )
                
                if success:
                    success_count += 1
                else:
                    error_count += 1
                
                # Post progress update
                self.post_message(SubscriptionIngestProgress(
                    current=success_count + error_count,
                    total=len(self.items),
                    success_count=success_count,
                    error_count=error_count
                ))
            
            # Post completion
            self.post_message(SubscriptionIngestComplete(
                total_items=len(self.items),
                success_count=success_count,
                error_count=error_count
            ))
            
        except Exception as e:
            logger.error(f"Error in subscription ingest worker: {e}")
            self.post_message(SubscriptionIngestError(str(e)))


async def bulk_ingest_subscription_items(app, selected_item_ids: List[int]) -> tuple[int, int]:
    """
    Bulk ingest selected subscription items.
    
    Args:
        app: Application instance
        selected_item_ids: List of item IDs to ingest
        
    Returns:
        Tuple of (success_count, error_count)
    """
    if not hasattr(app, 'subscriptions_db') or not app.subscriptions_db:
        raise ValueError("Subscriptions database not initialized")
        
    if not hasattr(app, 'media_db') or not app.media_db:
        raise ValueError("Media database not initialized")
    
    success_count = 0
    error_count = 0
    
    # Get content processor if configured
    content_processor = None
    if hasattr(app, 'chat_provider') and hasattr(app, 'chat_model'):
        content_processor = ContentProcessor(app.chat_provider, app.chat_model)
    
    for item_id in selected_item_ids:
        try:
            # Get item details
            cursor = app.subscriptions_db.conn.cursor()
            cursor.execute("""
                SELECT i.*, s.* 
                FROM subscription_items i
                JOIN subscriptions s ON i.subscription_id = s.id
                WHERE i.id = ?
            """, (item_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Item {item_id} not found")
                error_count += 1
                continue
            
            # Convert row to dictionaries
            item = dict(row)
            subscription = {
                'id': item['subscription_id'],
                'name': item['name'],
                'type': item['type'],
                'source': item['source'],
                'tags': item.get('tags'),
                'processing_options': item.get('processing_options'),
                'auto_ingest': item.get('auto_ingest', False)
            }
            
            # Process item
            success, media_id, message = await process_subscription_item(
                item,
                subscription,
                app.media_db,
                app.subscriptions_db,
                content_processor
            )
            
            if success:
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            logger.error(f"Error processing item {item_id}: {e}")
            error_count += 1
    
    return success_count, error_count


# Custom message types for progress updates
from textual.message import Message

class SubscriptionIngestProgress(Message):
    """Progress update for subscription ingestion."""
    def __init__(self, current: int, total: int, success_count: int, error_count: int):
        super().__init__()
        self.current = current
        self.total = total
        self.success_count = success_count
        self.error_count = error_count


class SubscriptionIngestComplete(Message):
    """Completion message for subscription ingestion."""
    def __init__(self, total_items: int, success_count: int, error_count: int):
        super().__init__()
        self.total_items = total_items
        self.success_count = success_count
        self.error_count = error_count


class SubscriptionIngestError(Message):
    """Error message for subscription ingestion."""
    def __init__(self, error: str):
        super().__init__()
        self.error = error


# End of subscription_ingest_worker.py