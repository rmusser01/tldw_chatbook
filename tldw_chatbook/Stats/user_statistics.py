"""
User Statistics Module

Calculates dynamic, privacy-focused user statistics from the database.
All calculations are performed on-demand without storing or logging results.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import re
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)


class UserStatistics:
    """Calculates user-focused statistics dynamically from the database."""
    
    def __init__(self, db_instance):
        """Initialize with a database instance."""
        self.db = db_instance
        
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get all user statistics in one call.
        Returns a dictionary with all calculated statistics.
        """
        stats = {}
        
        # Basic conversation stats
        stats['total_conversations'] = self._get_total_conversations()
        stats['total_messages'] = self._get_total_messages()
        stats['avg_messages_per_conversation'] = self._get_avg_messages_per_conversation()
        stats['avg_message_length'] = self._get_avg_message_length()
        
        # Time-based stats
        stats['data_history_length'] = self._get_data_history_length()
        stats['activity_last_30_days'] = self._get_activity_last_30_days()
        stats['most_active_time'] = self._get_most_active_time()
        stats['most_active_day'] = self._get_most_active_day_of_week()
        
        # User preferences
        stats['preferred_name'] = self._get_preferred_name()
        stats['preferred_device'] = self._get_preferred_device()
        stats['favorite_model'] = self._get_favorite_model()
        
        # Topic analysis
        stats['main_topics'] = self._get_main_topics()
        stats['top_topics_by_message_count'] = self._get_top_topics_by_message_count()
        
        # Fun stats
        stats['emoji_usage'] = self._get_emoji_usage_stats()
        stats['question_ratio'] = self._get_question_vs_statement_ratio()
        stats['longest_conversation'] = self._get_longest_conversation_info()
        stats['conversation_streaks'] = self._get_conversation_streaks()
        stats['vocabulary_diversity'] = self._get_vocabulary_diversity()
        stats['average_response_time'] = self._get_average_response_time()
        
        # Character chat stats
        stats['character_chat_stats'] = self._get_character_chat_stats()
        
        # Satisfaction rate (if ratings are available)
        stats['satisfaction_rate'] = self._get_satisfaction_rate()
        
        return stats
    
    def _get_total_conversations(self) -> int:
        """Get total number of conversations."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE deleted = 0"
            )
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting total conversations: {e}")
            return 0
    
    def _get_total_messages(self) -> int:
        """Get total number of messages."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE deleted = 0"
            )
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting total messages: {e}")
            return 0
    
    def _get_avg_messages_per_conversation(self) -> float:
        """Calculate average messages per conversation."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT AVG(msg_count) FROM (
                    SELECT COUNT(*) as msg_count 
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    WHERE m.deleted = 0 AND c.deleted = 0
                    GROUP BY m.conversation_id
                )
            """)
            result = cursor.fetchone()[0]
            return round(result, 2) if result else 0.0
        except Exception as e:
            logger.error(f"Error calculating avg messages per conversation: {e}")
            return 0.0
    
    def _get_avg_message_length(self) -> float:
        """Calculate average message length for user messages."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT AVG(LENGTH(content)) 
                FROM messages 
                WHERE deleted = 0 
                AND sender = 'user'
            """)
            result = cursor.fetchone()[0]
            return round(result, 2) if result else 0.0
        except Exception as e:
            logger.error(f"Error calculating avg message length: {e}")
            return 0.0
    
    def _get_data_history_length(self) -> Dict[str, Any]:
        """Get date range of user data."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT 
                    MIN(created_at) as earliest,
                    MAX(created_at) as latest
                FROM conversations
                WHERE deleted = 0
            """)
            row = cursor.fetchone()
            
            if row and row[0]:
                earliest = datetime.fromisoformat(row[0])
                latest = datetime.fromisoformat(row[1])
                days = (latest - earliest).days
                
                return {
                    'earliest_date': earliest.strftime('%Y-%m-%d'),
                    'latest_date': latest.strftime('%Y-%m-%d'),
                    'total_days': days,
                    'formatted': f"{days} days ({earliest.strftime('%b %d, %Y')} - {latest.strftime('%b %d, %Y')})"
                }
            return {'formatted': 'No data available'}
        except Exception as e:
            logger.error(f"Error getting data history length: {e}")
            return {'formatted': 'Error calculating'}
    
    def _get_activity_last_30_days(self) -> Dict[str, Any]:
        """Get user activity statistics for the last 30 days."""
        try:
            conn = self.db.get_or_create_connection()
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
            
            # Messages in last 30 days
            cursor = conn.execute("""
                SELECT COUNT(*) 
                FROM messages 
                WHERE deleted = 0 
                AND timestamp >= ?
                AND sender = 'user'
            """, (thirty_days_ago,))
            messages_count = cursor.fetchone()[0]
            
            # Conversations in last 30 days
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT conversation_id) 
                FROM messages 
                WHERE deleted = 0 
                AND timestamp >= ?
            """, (thirty_days_ago,))
            conversations_count = cursor.fetchone()[0]
            
            # Daily activity
            cursor = conn.execute("""
                SELECT DATE(timestamp) as day, COUNT(*) as count
                FROM messages
                WHERE deleted = 0 
                AND timestamp >= ?
                AND sender = 'user'
                GROUP BY DATE(timestamp)
                ORDER BY day
            """, (thirty_days_ago,))
            
            daily_activity = [(row[0], row[1]) for row in cursor.fetchall()]
            
            return {
                'messages': messages_count,
                'conversations': conversations_count,
                'daily_average': round(messages_count / 30, 2),
                'daily_activity': daily_activity
            }
        except Exception as e:
            logger.error(f"Error getting 30-day activity: {e}")
            return {'messages': 0, 'conversations': 0, 'daily_average': 0}
    
    def _get_preferred_name(self) -> str:
        """Detect user's preferred name from message patterns."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT content 
                FROM messages 
                WHERE deleted = 0 
                AND sender = 'user'
                AND (
                    content LIKE '%my name is%' 
                    OR content LIKE '%I am %'
                    OR content LIKE '%call me %'
                )
                LIMIT 50
            """)
            
            names = []
            patterns = [
                r"my name is (\w+)",
                r"I am (\w+)",
                r"call me (\w+)",
                r"I'm (\w+)"
            ]
            
            for row in cursor.fetchall():
                content = row[0].lower()
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        name = match.group(1).capitalize()
                        if len(name) > 2 and name.isalpha():
                            names.append(name)
            
            if names:
                name_counter = Counter(names)
                return name_counter.most_common(1)[0][0]
            
            return "User"
        except Exception as e:
            logger.error(f"Error detecting preferred name: {e}")
            return "User"
    
    def _get_preferred_device(self) -> str:
        """Estimate preferred device based on message patterns and length."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT 
                    AVG(LENGTH(content)) as avg_len,
                    COUNT(*) as msg_count,
                    AVG(CASE 
                        WHEN TIME(timestamp) BETWEEN '09:00' AND '17:00' THEN 1 
                        ELSE 0 
                    END) as business_hours_ratio
                FROM messages 
                WHERE deleted = 0 
                AND sender = 'user'
            """)
            
            row = cursor.fetchone()
            if row and row[0]:
                avg_length = row[0]
                business_ratio = row[2] if row[2] else 0
                
                # Heuristic: longer messages and business hours suggest desktop
                if avg_length > 100 or business_ratio > 0.6:
                    return "Desktop/Laptop"
                else:
                    return "Mobile"
            
            return "Unknown"
        except Exception as e:
            logger.error(f"Error detecting preferred device: {e}")
            return "Unknown"
    
    def _get_favorite_model(self) -> str:
        """Get the most frequently used AI model."""
        # This would need to be tracked in messages or conversations
        # For now, return a placeholder
        return "GPT-4o Mini"  # Placeholder
    
    def _get_main_topics(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Extract main topics from conversations using keyword extraction."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT content 
                FROM messages 
                WHERE deleted = 0 
                AND sender = 'user'
                ORDER BY timestamp DESC
                LIMIT 1000
            """)
            
            # Simple keyword extraction
            word_freq = Counter()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                         'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 
                         'was', 'are', 'were', 'been', 'be', 'have', 'has', 
                         'had', 'do', 'does', 'did', 'will', 'would', 'could',
                         'should', 'may', 'might', 'must', 'shall', 'can', 
                         'need', 'i', 'me', 'my', 'you', 'your', 'it', 'its',
                         'this', 'that', 'these', 'those', 'what', 'which',
                         'who', 'when', 'where', 'why', 'how'}
            
            for row in cursor.fetchall():
                words = re.findall(r'\b\w{4,}\b', row[0].lower())
                for word in words:
                    if word not in stop_words and word.isalpha():
                        word_freq[word] += 1
            
            # Group related words
            topic_groups = defaultdict(int)
            topic_keywords = {
                'programming': ['code', 'python', 'javascript', 'function', 'program', 'software', 'debug', 'error'],
                'data': ['data', 'database', 'analysis', 'statistics', 'chart', 'graph'],
                'ai': ['ai', 'machine', 'learning', 'model', 'neural', 'artificial'],
                'writing': ['write', 'writing', 'essay', 'article', 'story', 'document'],
                'business': ['business', 'company', 'market', 'customer', 'product', 'sales'],
                'education': ['learn', 'study', 'course', 'school', 'education', 'teach'],
                'health': ['health', 'medical', 'doctor', 'medicine', 'treatment', 'disease'],
                'technology': ['technology', 'computer', 'internet', 'digital', 'online', 'web']
            }
            
            for word, count in word_freq.items():
                for topic, keywords in topic_keywords.items():
                    if any(keyword in word for keyword in keywords):
                        topic_groups[topic] += count
                        break
                else:
                    # If no topic matched, use the word itself if frequent enough
                    if count > 5:
                        topic_groups[word] += count
            
            return sorted(topic_groups.items(), key=lambda x: x[1], reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Error extracting main topics: {e}")
            return []
    
    def _get_top_topics_by_message_count(self) -> Dict[str, List[Tuple[str, int]]]:
        """Get top topics for different message count ranges."""
        ranges = {
            'last_100': 100,
            'last_250': 250,
            'last_500': 500,
            'last_1000': 1000
        }
        
        results = {}
        for range_name, count in ranges.items():
            try:
                conn = self.db.get_or_create_connection()
                cursor = conn.execute("""
                    SELECT content 
                    FROM messages 
                    WHERE deleted = 0 
                    AND sender = 'user'
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (count,))
                
                # Similar keyword extraction as _get_main_topics but simpler
                word_freq = Counter()
                for row in cursor.fetchall():
                    words = re.findall(r'\b\w{5,}\b', row[0].lower())
                    word_freq.update(words)
                
                # Filter common words and get top 3
                filtered = [(w, c) for w, c in word_freq.most_common(20) 
                           if w not in {'would', 'could', 'should', 'about', 'there', 
                                       'where', 'which', 'think', 'really', 'something'}]
                results[range_name] = filtered[:3]
            except Exception as e:
                logger.error(f"Error getting topics for {range_name}: {e}")
                results[range_name] = []
        
        return results
    
    def _get_satisfaction_rate(self) -> Optional[float]:
        """Calculate satisfaction rate based on conversation ratings."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT 
                    COUNT(CASE WHEN rating >= 4 THEN 1 END) as satisfied,
                    COUNT(*) as total
                FROM conversations
                WHERE deleted = 0 
                AND rating IS NOT NULL
            """)
            
            row = cursor.fetchone()
            if row and row[1] > 0:
                return round((row[0] / row[1]) * 100, 1)
            return None
        except Exception as e:
            logger.error(f"Error calculating satisfaction rate: {e}")
            return None
    
    def _get_most_active_time(self) -> str:
        """Find the most active hour of the day."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT 
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as count
                FROM messages
                WHERE deleted = 0 
                AND sender = 'user'
                GROUP BY hour
                ORDER BY count DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                hour = int(row[0])
                if hour < 12:
                    period = "morning" if hour >= 6 else "night"
                elif hour < 17:
                    period = "afternoon"
                elif hour < 21:
                    period = "evening"
                else:
                    period = "night"
                
                return f"{hour}:00 ({period})"
            return "No data"
        except Exception as e:
            logger.error(f"Error finding most active time: {e}")
            return "Unknown"
    
    def _get_most_active_day_of_week(self) -> str:
        """Find the most active day of the week."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT 
                    CASE strftime('%w', timestamp)
                        WHEN '0' THEN 'Sunday'
                        WHEN '1' THEN 'Monday'
                        WHEN '2' THEN 'Tuesday'
                        WHEN '3' THEN 'Wednesday'
                        WHEN '4' THEN 'Thursday'
                        WHEN '5' THEN 'Friday'
                        WHEN '6' THEN 'Saturday'
                    END as day_name,
                    COUNT(*) as count
                FROM messages
                WHERE deleted = 0 
                AND sender = 'user'
                GROUP BY strftime('%w', timestamp)
                ORDER BY count DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            return row[0] if row else "No data"
        except Exception as e:
            logger.error(f"Error finding most active day: {e}")
            return "Unknown"
    
    def _get_emoji_usage_stats(self) -> Dict[str, Any]:
        """Calculate emoji usage statistics."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT content 
                FROM messages 
                WHERE deleted = 0 
                AND sender = 'user'
                LIMIT 5000
            """)
            
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE
            )
            
            emoji_counter = Counter()
            total_messages = 0
            messages_with_emoji = 0
            
            for row in cursor.fetchall():
                total_messages += 1
                emojis = emoji_pattern.findall(row[0])
                if emojis:
                    messages_with_emoji += 1
                    for emoji in emojis:
                        emoji_counter[emoji] += 1
            
            emoji_rate = round((messages_with_emoji / total_messages * 100), 1) if total_messages > 0 else 0
            
            return {
                'usage_rate': emoji_rate,
                'top_emojis': emoji_counter.most_common(5),
                'total_unique': len(emoji_counter)
            }
        except Exception as e:
            logger.error(f"Error calculating emoji stats: {e}")
            return {'usage_rate': 0, 'top_emojis': [], 'total_unique': 0}
    
    def _get_question_vs_statement_ratio(self) -> Dict[str, Any]:
        """Calculate ratio of questions to statements."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT content 
                FROM messages 
                WHERE deleted = 0 
                AND sender = 'user'
                LIMIT 5000
            """)
            
            questions = 0
            total = 0
            
            for row in cursor.fetchall():
                total += 1
                if '?' in row[0]:
                    questions += 1
            
            question_ratio = round((questions / total * 100), 1) if total > 0 else 0
            
            return {
                'question_percentage': question_ratio,
                'statement_percentage': round(100 - question_ratio, 1),
                'curiosity_level': 'High' if question_ratio > 40 else 'Moderate' if question_ratio > 20 else 'Low'
            }
        except Exception as e:
            logger.error(f"Error calculating question ratio: {e}")
            return {'question_percentage': 0, 'statement_percentage': 100, 'curiosity_level': 'Unknown'}
    
    def _get_longest_conversation_info(self) -> Dict[str, Any]:
        """Get information about the longest conversation."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT 
                    c.id,
                    c.title,
                    COUNT(m.id) as message_count,
                    c.created_at
                FROM conversations c
                JOIN messages m ON c.id = m.conversation_id
                WHERE c.deleted = 0 AND m.deleted = 0
                GROUP BY c.id
                ORDER BY message_count DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                return {
                    'title': row[1] or 'Untitled',
                    'message_count': row[2],
                    'date': datetime.fromisoformat(row[3]).strftime('%B %d, %Y')
                }
            return {'title': 'No conversations', 'message_count': 0}
        except Exception as e:
            logger.error(f"Error getting longest conversation: {e}")
            return {'title': 'Error', 'message_count': 0}
    
    def _get_conversation_streaks(self) -> Dict[str, Any]:
        """Calculate conversation streaks."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT DATE(timestamp) as day, COUNT(DISTINCT conversation_id) as convs
                FROM messages
                WHERE deleted = 0
                GROUP BY DATE(timestamp)
                ORDER BY day
            """)
            
            dates = []
            current_streak = 0
            longest_streak = 0
            last_date = None
            
            for row in cursor.fetchall():
                date = datetime.fromisoformat(row[0]).date()
                
                if last_date and (date - last_date).days == 1:
                    current_streak += 1
                else:
                    if current_streak > longest_streak:
                        longest_streak = current_streak
                    current_streak = 1
                
                last_date = date
            
            # Check if current streak is the longest
            if current_streak > longest_streak:
                longest_streak = current_streak
            
            # Check if still on streak
            today = datetime.now().date()
            if last_date == today or (last_date and (today - last_date).days == 1):
                active_streak = current_streak
            else:
                active_streak = 0
            
            return {
                'current_streak': active_streak,
                'longest_streak': longest_streak,
                'streak_status': 'Active' if active_streak > 0 else 'Inactive'
            }
        except Exception as e:
            logger.error(f"Error calculating streaks: {e}")
            return {'current_streak': 0, 'longest_streak': 0, 'streak_status': 'Unknown'}
    
    def _get_vocabulary_diversity(self) -> Dict[str, Any]:
        """Calculate vocabulary diversity score."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT content 
                FROM messages 
                WHERE deleted = 0 
                AND sender = 'user'
                LIMIT 1000
            """)
            
            all_words = []
            for row in cursor.fetchall():
                words = re.findall(r'\b\w+\b', row[0].lower())
                all_words.extend(words)
            
            if not all_words:
                return {'score': 0, 'level': 'No data'}
            
            unique_words = len(set(all_words))
            total_words = len(all_words)
            diversity_ratio = unique_words / total_words if total_words > 0 else 0
            
            # Score out of 100
            score = min(round(diversity_ratio * 200), 100)
            
            if score >= 80:
                level = "Excellent"
            elif score >= 60:
                level = "Good"
            elif score >= 40:
                level = "Average"
            else:
                level = "Limited"
            
            return {
                'score': score,
                'level': level,
                'unique_words': unique_words,
                'total_words': total_words
            }
        except Exception as e:
            logger.error(f"Error calculating vocabulary diversity: {e}")
            return {'score': 0, 'level': 'Error'}
    
    def _get_average_response_time(self) -> str:
        """Calculate average time between user messages and AI responses."""
        # This would require tracking response times in the database
        # For now, return a placeholder
        return "< 2 seconds"
    
    def _get_character_chat_stats(self) -> Dict[str, Any]:
        """Get statistics about character chats."""
        try:
            conn = self.db.get_or_create_connection()
            cursor = conn.execute("""
                SELECT 
                    cc.name,
                    COUNT(DISTINCT c.id) as conversation_count,
                    COUNT(m.id) as message_count
                FROM character_cards cc
                JOIN conversations c ON cc.id = c.character_id
                JOIN messages m ON c.id = m.conversation_id
                WHERE cc.deleted = 0 
                AND c.deleted = 0 
                AND m.deleted = 0
                AND cc.id != 1  -- Exclude default assistant
                GROUP BY cc.id
                ORDER BY message_count DESC
                LIMIT 5
            """)
            
            characters = []
            for row in cursor.fetchall():
                characters.append({
                    'name': row[0],
                    'conversations': row[1],
                    'messages': row[2]
                })
            
            # Get total character count
            cursor = conn.execute(
                "SELECT COUNT(*) FROM character_cards WHERE deleted = 0 AND id != 1"
            )
            total_characters = cursor.fetchone()[0]
            
            return {
                'total_characters': total_characters,
                'top_characters': characters
            }
        except Exception as e:
            logger.error(f"Error getting character stats: {e}")
            return {'total_characters': 0, 'top_characters': []}