# recursive_summarizer.py
# Description: Recursive summarization pipeline for handling long content
#
# This module implements hierarchical summarization strategies to condense
# long content while preserving key information within token budgets.
#
# Imports
import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Callable
#
# Third-Party Imports
from loguru import logger
#
# Local Imports
from .token_manager import TokenCounter
from ..LLM_Calls.LLM_API_Calls import chat_with_provider
from ..Chat.Chat_Functions import get_provider_model_name
from ..Metrics.metrics_logger import log_histogram, log_counter
#
########################################################################################################################
#
# Data Classes
#
########################################################################################################################

@dataclass
class SummarizationConfig:
    """Configuration for recursive summarization."""
    # LLM settings
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.3  # Lower for more consistent summaries
    
    # Summarization parameters
    chunk_size: int = 2000  # Tokens per chunk
    summary_ratio: float = 0.2  # Target 20% of original size
    max_recursion_depth: int = 3
    min_chunk_size: int = 500  # Don't chunk below this
    
    # Content handling
    preserve_quotes: bool = True
    preserve_links: bool = True
    preserve_structure: bool = True
    
    # Summary styles
    style: str = 'balanced'  # 'concise', 'balanced', 'detailed'
    format: str = 'prose'  # 'prose', 'bullets', 'structured'
    
    # Fallback options
    use_fallback: bool = True  # Use extraction-based summary if LLM fails
    fallback_sentences: int = 5


@dataclass
class SummarizedChunk:
    """Represents a summarized chunk of content."""
    original_text: str
    summary: str
    original_tokens: int
    summary_tokens: int
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]


@dataclass
class SummarizationResult:
    """Result of recursive summarization."""
    final_summary: str
    original_tokens: int
    final_tokens: int
    compression_ratio: float
    levels_used: int
    chunks_processed: int
    method_used: str  # 'llm', 'fallback', 'truncation'
    processing_time: float
    metadata: Dict[str, Any]


########################################################################################################################
#
# Recursive Summarizer
#
########################################################################################################################

class RecursiveSummarizer:
    """
    Hierarchical summarization for long content.
    
    Implements multiple strategies:
    - Chunk-based summarization
    - Hierarchical merging
    - Structure-aware splitting
    - Fallback extraction methods
    """
    
    def __init__(self, config: Optional[SummarizationConfig] = None):
        """
        Initialize recursive summarizer.
        
        Args:
            config: Summarization configuration
        """
        self.config = config or SummarizationConfig()
        self.token_counter = TokenCounter()
        self._summary_cache = {}
    
    async def summarize_content(self, 
                              content: str, 
                              target_tokens: int,
                              context: Optional[str] = None) -> SummarizationResult:
        """
        Recursively summarize content to fit token budget.
        
        Args:
            content: Content to summarize
            target_tokens: Target token count
            context: Optional context about the content
            
        Returns:
            Summarization result
        """
        start_time = datetime.now(timezone.utc)
        
        # Check if summarization needed
        original_tokens = self.token_counter.count_tokens(content)
        if original_tokens <= target_tokens:
            return SummarizationResult(
                final_summary=content,
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                compression_ratio=1.0,
                levels_used=0,
                chunks_processed=0,
                method_used='none',
                processing_time=0,
                metadata={}
            )
        
        # Try LLM summarization if configured
        if self.config.provider and self.config.model:
            try:
                result = await self._llm_recursive_summarize(content, target_tokens, context)
                result.processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                # Log metrics
                log_histogram("recursive_summarization_duration", result.processing_time, labels={
                    "method": result.method_used,
                    "levels": str(result.levels_used)
                })
                log_counter("recursive_summarization_performed", labels={
                    "method": result.method_used,
                    "compression_ratio": f"{result.compression_ratio:.2f}"
                })
                
                return result
                
            except Exception as e:
                logger.error(f"LLM summarization failed: {str(e)}")
                if not self.config.use_fallback:
                    raise
        
        # Fallback to extraction-based summarization
        if self.config.use_fallback:
            result = await self._fallback_summarize(content, target_tokens)
            result.processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            return result
        
        # Last resort: intelligent truncation
        result = self._truncate_intelligently(content, target_tokens)
        result.processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        return result
    
    async def _llm_recursive_summarize(self, 
                                     content: str, 
                                     target_tokens: int,
                                     context: Optional[str] = None,
                                     level: int = 0) -> SummarizationResult:
        """Recursively summarize using LLM."""
        original_tokens = self.token_counter.count_tokens(content)
        
        # Base case: content fits or max depth reached
        if original_tokens <= target_tokens or level >= self.config.max_recursion_depth:
            return SummarizationResult(
                final_summary=content,
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                compression_ratio=1.0,
                levels_used=level,
                chunks_processed=0,
                method_used='llm',
                processing_time=0,
                metadata={'stopped_at': 'base_case'}
            )
        
        # Calculate chunking parameters
        chunk_params = self._calculate_chunk_params(original_tokens, target_tokens, level)
        
        # Split content into chunks
        chunks = self._split_into_chunks(content, chunk_params['chunk_size'])
        
        # Summarize each chunk
        summarized_chunks = []
        chunks_processed = 0
        
        for i, chunk in enumerate(chunks):
            chunk_summary = await self._summarize_single_chunk(
                chunk, 
                chunk_params['chunk_target'],
                context,
                i,
                len(chunks)
            )
            summarized_chunks.append(chunk_summary)
            chunks_processed += 1
        
        # Merge summaries
        merged_content = self._merge_summaries(summarized_chunks)
        merged_tokens = self.token_counter.count_tokens(merged_content)
        
        # Check if we need another level
        if merged_tokens > target_tokens and level < self.config.max_recursion_depth - 1:
            # Recursive call
            sub_result = await self._llm_recursive_summarize(
                merged_content,
                target_tokens,
                context,
                level + 1
            )
            
            return SummarizationResult(
                final_summary=sub_result.final_summary,
                original_tokens=original_tokens,
                final_tokens=sub_result.final_tokens,
                compression_ratio=original_tokens / sub_result.final_tokens,
                levels_used=level + 1,
                chunks_processed=chunks_processed + sub_result.chunks_processed,
                method_used='llm',
                processing_time=0,
                metadata={
                    'total_chunks': chunks_processed + sub_result.chunks_processed,
                    'levels': level + 1
                }
            )
        
        # Final result
        final_tokens = self.token_counter.count_tokens(merged_content)
        return SummarizationResult(
            final_summary=merged_content,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            compression_ratio=original_tokens / final_tokens if final_tokens > 0 else 0,
            levels_used=level + 1,
            chunks_processed=chunks_processed,
            method_used='llm',
            processing_time=0,
            metadata={'total_chunks': chunks_processed}
        )
    
    def _calculate_chunk_params(self, content_tokens: int, target_tokens: int, level: int) -> Dict[str, Any]:
        """Calculate optimal chunk size and targets for current level."""
        # Adjust chunk size based on level (smaller chunks at deeper levels)
        base_chunk_size = self.config.chunk_size
        chunk_size = max(
            int(base_chunk_size * (0.7 ** level)),  # Reduce by 30% each level
            self.config.min_chunk_size
        )
        
        # Calculate number of chunks
        num_chunks = math.ceil(content_tokens / chunk_size)
        
        # Calculate target size for each chunk summary
        # Account for merging overhead
        merge_overhead = 1.1  # 10% overhead for merge formatting
        chunk_target = int(target_tokens / (num_chunks * merge_overhead))
        
        # Ensure reasonable compression
        chunk_target = max(chunk_target, int(chunk_size * self.config.summary_ratio))
        
        return {
            'chunk_size': chunk_size,
            'chunk_target': chunk_target,
            'num_chunks': num_chunks
        }
    
    def _split_into_chunks(self, content: str, chunk_size: int) -> List[str]:
        """Split content into chunks intelligently."""
        chunks = []
        
        # Try to split at natural boundaries
        if self.config.preserve_structure:
            # Split by paragraphs first
            paragraphs = content.split('\n\n')
            current_chunk = []
            current_tokens = 0
            
            for para in paragraphs:
                para_tokens = self.token_counter.count_tokens(para)
                
                if current_tokens + para_tokens > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens
            
            # Add remaining
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
        
        else:
            # Simple token-based splitting
            words = content.split()
            current_chunk = []
            current_tokens = 0
            
            for word in words:
                current_chunk.append(word)
                current_tokens = self.token_counter.count_tokens(' '.join(current_chunk))
                
                if current_tokens >= chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks
    
    async def _summarize_single_chunk(self, 
                                    chunk: str, 
                                    target_tokens: int,
                                    context: Optional[str],
                                    chunk_index: int,
                                    total_chunks: int) -> SummarizedChunk:
        """Summarize a single chunk using LLM."""
        # Build prompt based on style
        prompt = self._build_chunk_prompt(chunk, target_tokens, context, chunk_index, total_chunks)
        
        try:
            # Get provider and model info
            provider_info = get_provider_model_name(self.config.provider)
            
            # Call LLM
            response = await chat_with_provider(
                provider=self.config.provider,
                api_key=self.config.api_key,
                model=self.config.model or provider_info['default_model'],
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=target_tokens * 2  # Allow some flexibility
            )
            
            summary = response.get('content', '').strip()
            
        except Exception as e:
            logger.error(f"Chunk summarization failed: {str(e)}")
            # Fallback to extraction
            summary = self._extract_key_sentences(chunk, target_tokens)
        
        return SummarizedChunk(
            original_text=chunk,
            summary=summary,
            original_tokens=self.token_counter.count_tokens(chunk),
            summary_tokens=self.token_counter.count_tokens(summary),
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            metadata={}
        )
    
    def _build_chunk_prompt(self, chunk: str, target_tokens: int, 
                          context: Optional[str], chunk_index: int, total_chunks: int) -> str:
        """Build prompt for chunk summarization."""
        parts = []
        
        # Add context if provided
        if context:
            parts.append(f"Context: {context}")
        
        # Add chunk info
        if total_chunks > 1:
            parts.append(f"This is part {chunk_index + 1} of {total_chunks}.")
        
        # Main instruction based on style
        if self.config.style == 'concise':
            parts.append(f"Summarize the following content in approximately {target_tokens} tokens. Be extremely concise, focusing only on the most critical information:")
        elif self.config.style == 'detailed':
            parts.append(f"Provide a comprehensive summary of the following content in approximately {target_tokens} tokens. Preserve important details and nuances:")
        else:  # balanced
            parts.append(f"Summarize the following content in approximately {target_tokens} tokens. Balance brevity with completeness:")
        
        # Format instruction
        if self.config.format == 'bullets':
            parts.append("Format as bullet points.")
        elif self.config.format == 'structured':
            parts.append("Use clear headings and structure.")
        
        # Special instructions
        if self.config.preserve_quotes:
            parts.append("Preserve important quotes.")
        if self.config.preserve_links:
            parts.append("Preserve all links and URLs.")
        
        # Add the chunk
        parts.append(f"\nContent to summarize:\n{chunk}")
        
        return '\n\n'.join(parts)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for summarization."""
        return """You are an expert content summarizer. Your task is to create clear, accurate summaries that preserve the key information while significantly reducing length. 

Key principles:
1. Maintain factual accuracy
2. Preserve the most important information
3. Use clear, concise language
4. Maintain logical flow
5. Respect the requested token limit"""
    
    def _merge_summaries(self, chunks: List[SummarizedChunk]) -> str:
        """Merge chunk summaries into coherent text."""
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0].summary
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x.chunk_index)
        
        # Merge based on format
        if self.config.format == 'bullets':
            # Combine bullet points
            all_bullets = []
            for chunk in chunks:
                # Extract bullets from summary
                lines = chunk.summary.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                        all_bullets.append(line)
                    elif line and not any(line.startswith(x) for x in ['#', '##']):
                        # Convert non-bullet lines to bullets
                        all_bullets.append(f"• {line}")
            
            return '\n'.join(all_bullets)
        
        elif self.config.format == 'structured':
            # Preserve structure
            sections = []
            for chunk in chunks:
                if chunk.chunk_index > 0:
                    sections.append("")  # Add spacing
                sections.append(chunk.summary)
            
            return '\n'.join(sections)
        
        else:  # prose
            # Join with transitions
            merged = []
            for i, chunk in enumerate(chunks):
                if i > 0 and not chunk.summary.startswith(('However', 'Additionally', 'Furthermore', 'Moreover')):
                    # Add simple transition
                    merged.append("Additionally, " + chunk.summary[0].lower() + chunk.summary[1:])
                else:
                    merged.append(chunk.summary)
            
            return ' '.join(merged)
    
    async def _fallback_summarize(self, content: str, target_tokens: int) -> SummarizationResult:
        """Fallback extraction-based summarization."""
        original_tokens = self.token_counter.count_tokens(content)
        
        # Extract key sentences
        summary = self._extract_key_sentences(content, target_tokens)
        final_tokens = self.token_counter.count_tokens(summary)
        
        return SummarizationResult(
            final_summary=summary,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            compression_ratio=original_tokens / final_tokens if final_tokens > 0 else 0,
            levels_used=0,
            chunks_processed=0,
            method_used='fallback',
            processing_time=0,
            metadata={'method': 'sentence_extraction'}
        )
    
    def _extract_key_sentences(self, content: str, target_tokens: int) -> str:
        """Extract key sentences using simple heuristics."""
        # Split into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        if not sentences:
            return content[:target_tokens * 4]  # Rough approximation
        
        # Score sentences
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position scoring (beginning and end are important)
            if i < 3:
                score += 2
            if i >= len(sentences) - 3:
                score += 1
            
            # Length scoring (prefer medium length)
            words = len(sentence.split())
            if 10 <= words <= 30:
                score += 1
            
            # Keyword scoring
            keywords = ['important', 'significant', 'key', 'main', 'primary', 'critical',
                       'however', 'therefore', 'conclusion', 'summary', 'result']
            for keyword in keywords:
                if keyword in sentence.lower():
                    score += 1
            
            # Contains numbers or statistics
            if any(char.isdigit() for char in sentence):
                score += 1
            
            scored_sentences.append((score, i, sentence))
        
        # Sort by score
        scored_sentences.sort(key=lambda x: (-x[0], x[1]))
        
        # Select sentences until target reached
        selected = []
        current_tokens = 0
        
        for score, idx, sentence in scored_sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            if current_tokens + sentence_tokens <= target_tokens:
                selected.append((idx, sentence))
                current_tokens += sentence_tokens
            
            if current_tokens >= target_tokens * 0.9:  # Allow 90% usage
                break
        
        # Sort by original position
        selected.sort(key=lambda x: x[0])
        
        # Join sentences
        return ' '.join(sentence for _, sentence in selected)
    
    def _truncate_intelligently(self, content: str, target_tokens: int) -> SummarizationResult:
        """Last resort: intelligent truncation."""
        truncated = self.token_counter.truncate_to_tokens(content, target_tokens)
        
        return SummarizationResult(
            final_summary=truncated,
            original_tokens=self.token_counter.count_tokens(content),
            final_tokens=self.token_counter.count_tokens(truncated),
            compression_ratio=len(content) / len(truncated) if truncated else 0,
            levels_used=0,
            chunks_processed=0,
            method_used='truncation',
            processing_time=0,
            metadata={'method': 'intelligent_truncation'}
        )


# End of recursive_summarizer.py