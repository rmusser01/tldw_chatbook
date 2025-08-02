# content_processor.py
# Description: Content processing and LLM analysis for subscription items
#
# This module provides:
# - Content extraction from feeds and URLs
# - Optional LLM analysis before storage
# - Keyword extraction from metadata
# - Content summarization
# - Save only analysis results option
#
# Imports
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
#
# Third-Party Imports
from bs4 import BeautifulSoup
from loguru import logger
#
# Local Imports
from ..Chat.Chat_Functions import chat_api_call
from ..Metrics.metrics_logger import log_histogram, log_counter
#
########################################################################################################################
#
# Content Processing Classes
#
########################################################################################################################

class ContentProcessor:
    """Process subscription content for ingestion."""
    
    def __init__(self, llm_provider: Optional[str] = None, llm_model: Optional[str] = None):
        """
        Initialize content processor.
        
        Args:
            llm_provider: LLM provider for analysis
            llm_model: LLM model for analysis
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        
    async def process_item(self, item: Dict[str, Any], subscription: Dict[str, Any], 
                          analyze: bool = False, save_analysis_only: bool = False) -> Dict[str, Any]:
        """
        Process a subscription item for ingestion.
        
        Args:
            item: Raw item from feed/URL monitor
            subscription: Subscription configuration
            analyze: Whether to perform LLM analysis
            save_analysis_only: Whether to save only the analysis result
            
        Returns:
            Processed item ready for ingestion
        """
        start_time = datetime.now()
        
        # Extract base content
        processed = {
            'url': item.get('url', ''),
            'title': self._clean_title(item.get('title', 'Untitled')),
            'author': item.get('author', ''),
            'published_date': item.get('published_date'),
            'subscription_id': subscription['id'],
            'subscription_name': subscription['name'],
            'subscription_type': subscription['type']
        }
        
        # Extract and clean content
        content = self._extract_content(item, subscription)
        processed['original_content'] = content
        
        # Extract keywords from metadata
        keywords = self._extract_keywords(item, subscription)
        processed['keywords'] = keywords
        
        # Perform LLM analysis if requested
        if analyze and self.llm_provider and self.llm_model:
            try:
                analysis = await self._analyze_content(content, item, subscription)
                processed['analysis'] = analysis
                
                if save_analysis_only:
                    # Replace content with analysis
                    processed['content'] = analysis
                else:
                    # Keep both original and analysis
                    processed['content'] = content
                    
            except Exception as e:
                logger.error(f"Error analyzing content: {e}")
                processed['content'] = content
                processed['analysis_error'] = str(e)
        else:
            processed['content'] = content
        
        # Add processing metadata
        processed['processed_at'] = datetime.now().isoformat()
        processed['processing_duration_ms'] = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Log metrics
        log_histogram("subscription_item_processing_duration", processed['processing_duration_ms'] / 1000.0, labels={
            "analyzed": str(analyze),
            "subscription_type": subscription['type']
        })
        
        return processed
    
    def _extract_content(self, item: Dict[str, Any], subscription: Dict[str, Any]) -> str:
        """
        Extract clean content from item.
        
        Args:
            item: Raw item
            subscription: Subscription configuration
            
        Returns:
            Extracted content text
        """
        # Get raw content
        raw_content = item.get('content', '')
        
        # Handle different content types
        if subscription['type'] == 'url_change':
            # URL monitoring - content is already extracted
            return raw_content
        
        # For feeds, we might have HTML content
        if '<' in raw_content and '>' in raw_content:
            # Looks like HTML - extract text
            soup = BeautifulSoup(raw_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        else:
            # Plain text
            return raw_content
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize title."""
        # Remove excessive whitespace
        title = ' '.join(title.split())
        
        # Remove common prefixes
        prefixes = ['BREAKING:', 'UPDATE:', 'EXCLUSIVE:', 'JUST IN:']
        for prefix in prefixes:
            if title.upper().startswith(prefix):
                title = title[len(prefix):].strip()
        
        # Limit length
        if len(title) > 500:
            title = title[:497] + '...'
        
        return title
    
    def _extract_keywords(self, item: Dict[str, Any], subscription: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from item metadata.
        
        Args:
            item: Item data
            subscription: Subscription configuration
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # Add subscription tags
        if subscription.get('tags'):
            tags = [tag.strip() for tag in subscription['tags'].split(',') if tag.strip()]
            keywords.extend(tags)
        
        # Add subscription folder as keyword
        if subscription.get('folder'):
            keywords.append(subscription['folder'].lower())
        
        # Add item categories
        if item.get('categories'):
            keywords.extend([cat.lower() for cat in item['categories'] if cat])
        
        # Add source domain
        if item.get('url'):
            try:
                parsed = urlparse(item['url'])
                domain = parsed.netloc.replace('www.', '')
                if domain:
                    keywords.append(domain)
            except:
                pass
        
        # Extract keywords from title
        if item.get('title'):
            # Simple keyword extraction from title
            title_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', item['title'])
            keywords.extend([w.lower() for w in title_words if len(w) > 3])
        
        # Add content type
        keywords.append(f"subscription-{subscription['type']}")
        
        # Remove duplicates and clean
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw = kw.strip().lower()
            if kw and kw not in seen and len(kw) > 2:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:20]  # Limit to 20 keywords
    
    async def _analyze_content(self, content: str, item: Dict[str, Any], 
                             subscription: Dict[str, Any]) -> str:
        """
        Analyze content using LLM.
        
        Args:
            content: Content to analyze
            item: Item metadata
            subscription: Subscription configuration
            
        Returns:
            Analysis result
        """
        # Build analysis prompt
        prompt = self._build_analysis_prompt(content, item, subscription)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes and summarizes content from subscriptions."},
            {"role": "user", "content": prompt}
        ]
        
        # Call LLM
        # provider_model = get_provider_model_name(self.llm_provider, self.llm_model)
        
        start_time = datetime.now()
        
        response = chat_api_call(
            api_endpoint=self.llm_provider,
            messages_payload=messages,
            temp=0.3,
            max_tokens=1000
        )
        
        # Extract response
        if response and 'content' in response:
            analysis = response['content']
        else:
            raise ValueError("Invalid LLM response")
        
        # Log metrics
        duration = (datetime.now() - start_time).total_seconds()
        log_histogram("subscription_llm_analysis_duration", duration, labels={
            "provider": self.llm_provider,
            "model": self.llm_model
        })
        log_counter("subscription_llm_analysis_count", labels={
            "provider": self.llm_provider,
            "subscription_type": subscription['type']
        })
        
        return analysis
    
    def _build_analysis_prompt(self, content: str, item: Dict[str, Any], 
                              subscription: Dict[str, Any]) -> str:
        """
        Build analysis prompt based on subscription type.
        
        Args:
            content: Content to analyze
            item: Item metadata
            subscription: Subscription configuration
            
        Returns:
            Analysis prompt
        """
        # Get custom prompt if configured
        if subscription.get('processing_options'):
            try:
                options = json.loads(subscription['processing_options'])
                if 'analysis_prompt' in options:
                    # Use custom prompt
                    prompt = options['analysis_prompt']
                    # Replace variables
                    prompt = prompt.replace('{content}', content[:5000])
                    prompt = prompt.replace('{title}', item.get('title', ''))
                    prompt = prompt.replace('{source}', subscription['name'])
                    prompt = prompt.replace('{url}', item.get('url', ''))
                    return prompt
            except:
                pass
        
        # Build default prompt based on type
        if subscription['type'] in ['rss', 'atom', 'json_feed']:
            prompt = f"""Analyze this article from {subscription['name']}:

Title: {item.get('title', 'Untitled')}
URL: {item.get('url', 'N/A')}
Published: {item.get('published_date', 'Unknown')}

Content:
{content[:5000]}

Please provide:
1. A concise summary (2-3 sentences)
2. Key points or insights (bullet points)
3. Why this might be important or relevant
4. Any action items or implications

Keep the analysis focused and practical."""

        elif subscription['type'] == 'url_change':
            prompt = f"""A monitored webpage has changed:

URL: {item.get('url', subscription['source'])}
Change: {item.get('change_percentage', 0)*100:.1f}% of content changed

New content:
{content[:5000]}

Please:
1. Summarize what has changed
2. Highlight the most important updates
3. Assess the significance of these changes
4. Suggest any follow-up actions if needed"""

        elif subscription['type'] == 'podcast':
            prompt = f"""Analyze this podcast episode from {subscription['name']}:

Title: {item.get('title', 'Untitled')}
Published: {item.get('published_date', 'Unknown')}

Description:
{content[:3000]}

Please provide:
1. Episode summary
2. Main topics discussed
3. Key takeaways
4. Whether this episode is worth listening to and why"""

        else:
            # Generic prompt
            prompt = f"""Analyze this content from {subscription['name']}:

Title: {item.get('title', 'Untitled')}
Type: {subscription['type']}

Content:
{content[:5000]}

Please provide a helpful analysis including:
1. Summary
2. Key information
3. Relevance or importance
4. Any recommended actions"""
        
        return prompt


class KeywordExtractor:
    """Extract keywords using various methods."""
    
    @staticmethod
    def extract_from_text(text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text using simple frequency analysis.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum keywords to return
            
        Returns:
            List of keywords
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split
        words = re.findall(r'\b[a-z]+\b', text)
        
        # Filter stop words (simplified list)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
            'why', 'how', 'not', 'no', 'yes', 'all', 'each', 'every', 'some', 'any'
        }
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, freq in sorted_words[:max_keywords]]
    
    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """
        Extract named entities from text (simplified version).
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entities
        """
        entities = []
        
        # Find capitalized sequences (simple NER)
        # Matches: "Apple Inc" "San Francisco" "John Smith"
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # Filter out common words that start sentences
            if match.lower() not in ['the', 'this', 'that', 'these', 'those']:
                entities.append(match)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities[:15]  # Limit to 15 entities


class ContentSummarizer:
    """Summarize content without LLM (fallback option)."""
    
    @staticmethod
    def summarize(text: str, max_sentences: int = 3) -> str:
        """
        Create a simple extractive summary.
        
        Args:
            text: Text to summarize
            max_sentences: Maximum sentences in summary
            
        Returns:
            Summary text
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences by word frequency
        word_freq = {}
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score each sentence
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            words = sentence.lower().split()
            for word in words:
                score += word_freq.get(word, 0)
            
            # Boost first and last sentences
            if i == 0:
                score *= 1.5
            elif i == len(sentences) - 1:
                score *= 1.2
                
            sentence_scores.append((score, i, sentence))
        
        # Sort by score
        sentence_scores.sort(reverse=True)
        
        # Get top sentences in original order
        top_indices = sorted([idx for score, idx, sent in sentence_scores[:max_sentences]])
        summary_sentences = [sentences[idx] for idx in top_indices]
        
        return '. '.join(summary_sentences) + '.'


# End of content_processor.py