"""
Simplified AI-Powered News Aggregation & Chatbot System
Focus on core requirements with clean, minimal code
"""

# Fix SQLite version issue for ChromaDB on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import re
import random

# Core dependencies
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# News extraction
import feedparser
import requests
from bs4 import BeautifulSoup
from newspaper import Article

# AI
import google.generativeai as genai

# Vector Database
import chromadb
from chromadb.config import Settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*CoreML.*")
warnings.filterwarnings("ignore", message=".*Context leak.*")

@dataclass
class NewsArticle:
    """Simple news article data model"""
    title: str
    content: str
    summary: str
    author: str
    source: str
    url: str
    category: str = "unknown"
    frequency: int = 1
    published_date: Optional[datetime] = None

@dataclass
class ChatMessage:
    """Represents a single message in the conversation history"""
    question: str
    answer: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[Dict] = field(default_factory=list)
    session_id: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'question': self.question,
            'answer': self.answer,
            'timestamp': self.timestamp.isoformat(),
            'sources': self.sources,
            'session_id': self.session_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        """Create from dictionary"""
        return cls(
            question=data['question'],
            answer=data['answer'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            sources=data.get('sources', []),
            session_id=data.get('session_id', '')
        )

class ConversationMemory:
    """Manages conversation history per session with automatic cleanup"""
    
    def __init__(self, max_messages: int = 10, max_age_hours: int = 24):
        self.max_messages = max_messages
        self.max_age_hours = max_age_hours
        self._conversations: Dict[str, List[ChatMessage]] = {}
        self._last_cleanup = datetime.now()
    
    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Add a message to the conversation history"""
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        
        message.session_id = session_id
        self._conversations[session_id].append(message)
        
        # Cleanup old messages
        self._cleanup_session(session_id)
        
        # Periodic cleanup of all sessions
        self._periodic_cleanup()
    
    def get_history(self, session_id: str, max_messages: Optional[int] = None) -> List[ChatMessage]:
        """Get conversation history for a session"""
        if session_id not in self._conversations:
            return []
        
        messages = self._conversations[session_id]
        if max_messages:
            return messages[-max_messages:]
        return messages
    
    def get_recent_context(self, session_id: str, max_messages: int = 3) -> str:
        """Get recent conversation context as formatted string for prompts"""
        messages = self.get_history(session_id, max_messages)
        if not messages:
            return ""
        
        context_parts = []
        for msg in messages:
            context_parts.append(f"User: {msg.question}")
            context_parts.append(f"Assistant: {msg.answer}")
        
        return "\n".join(context_parts)
    
    def get_last_answer(self, session_id: str) -> str:
        """Get the last assistant answer from conversation history"""
        messages = self.get_history(session_id, max_messages=1)
        if messages:
            return messages[-1].answer
        return ""
    
    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session"""
        if session_id in self._conversations:
            del self._conversations[session_id]
    
    def set_config(self, max_messages: int, max_age_hours: int) -> None:
        """Update memory configuration"""
        self.max_messages = max_messages
        self.max_age_hours = max_age_hours
    
    def _cleanup_session(self, session_id: str) -> None:
        """Clean up old messages for a specific session"""
        if session_id not in self._conversations:
            return
        
        messages = self._conversations[session_id]
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
        
        # Remove messages older than max_age_hours
        messages = [msg for msg in messages if msg.timestamp > cutoff_time]
        
        # Limit to max_messages
        if len(messages) > self.max_messages:
            messages = messages[-self.max_messages:]
        
        self._conversations[session_id] = messages
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of all sessions (every 10 minutes)"""
        if datetime.now() - self._last_cleanup < timedelta(minutes=10):
            return
        
        self._last_cleanup = datetime.now()
        
        # Clean up all sessions
        for session_id in list(self._conversations.keys()):
            self._cleanup_session(session_id)
            
            # Remove empty sessions
            if not self._conversations[session_id]:
                del self._conversations[session_id]

def normalize_text_for_matching(text: str) -> str:
    """
    Normalize text for better matching by handling special characters and punctuation.
    This function ensures that text with apostrophes, quotes, and other special characters
    is processed consistently for matching purposes.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle common special characters and punctuation
    replacements = {
        # Smart quotes and apostrophes
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...', # Horizontal ellipsis
        '\u00A0': ' ',  # Non-breaking space
        '\u200B': '',   # Zero-width space
        '\u200C': '',   # Zero-width non-joiner
        '\u200D': '',   # Zero-width joiner
        '\uFEFF': '',   # Zero-width no-break space
    }
    
    # Apply character replacements
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Handle common contractions first (before possessive forms)
    contractions = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "doesn't": "does not",
        "didn't": "did not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "mightn't": "might not",
        "mustn't": "must not",
        "shan't": "shall not",
        "let's": "let us",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
        "here's": "here is",
        "who's": "who is",
        "what's": "what is",
        "where's": "where is",
        "when's": "when is",
        "why's": "why is",
        "how's": "how is",
    }
    
    for contraction, expansion in contractions.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
    
    # Handle possessive forms after contractions
    # Convert "months'" to "months" for better matching
    text = re.sub(r"'s\b", "s", text)  # Convert "month's" to "months"
    text = re.sub(r"s'\b", "s", text)  # Convert "months'" to "months"
    
    # Additional possessive form handling
    text = re.sub(r"'s\s", "s ", text)  # Convert "month's " to "months "
    text = re.sub(r"s'\s", "s ", text)  # Convert "months' " to "months "
    text = re.sub(r"'s$", "s", text)    # Convert "month's" at end to "months"
    text = re.sub(r"s'$", "s", text)    # Convert "months'" at end to "months"
    
    # Remove question marks and other punctuation that might interfere with matching
    text = re.sub(r'[?!.,;:]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_words_for_matching(text: str) -> set:
    """
    Extract words from text for matching purposes, handling special characters properly.
    This function splits text into words while preserving important semantic information.
    """
    if not text:
        return set()
    
    # Normalize the text first
    normalized_text = normalize_text_for_matching(text)
    
    # Split on whitespace and filter out empty strings
    words = [word.strip() for word in normalized_text.split() if word.strip()]
    
    # Filter out very short words (likely noise), but keep numbers and important single characters
    filtered_words = []
    for word in words:
        if len(word) > 1 or word.isdigit() or word in ['a', 'i', 'o']:  # Keep numbers and common single letters
            filtered_words.append(word)
    
    return set(filtered_words)

class NewsAggregator:
    """Main news aggregation system - simplified"""
    
    def __init__(self, gemini_api_key: str):
        # Initialize Gemini API
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Initialize embedding model for vector database
        self.embedding_model = genai.embed_content
        
        # Initialize TF-IDF vectorizer for duplicate detection and chatbot RAG
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize ChromaDB for vector storage (new configuration)
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )
        
        # Initialize collections
        self.news_collection = self.chroma_client.get_or_create_collection(
            name="news_articles",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Keep TF-IDF for duplicate detection (as backup)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize conversation memory
        self.memory = ConversationMemory()
        
        
        # Australian news sources by category - Each source appears only once
        self.news_sources = {
            "sports": [
                "https://www.abc.net.au/news/feed/45910/rss.xml",  # ABC News Sport
                "https://www.smh.com.au/rss/sport.xml",            # Sydney Morning Herald Sport
                "https://www.theguardian.com/au/sport/rss",        # The Guardian Australia Sport
                "https://www.news.com.au/sport/rss",               # News.com.au Sport
            ],
            "lifestyle": [
                "https://www.abc.net.au/news/feed/51120/rss.xml",  # ABC News Main (includes lifestyle)
                "https://www.smh.com.au/rss/lifestyle.xml",        # Sydney Morning Herald Lifestyle
                "https://www.theguardian.com/au/lifeandstyle/rss", # The Guardian Australia Life & Style
                "https://www.news.com.au/lifestyle/rss",           # News.com.au Lifestyle
            ],
            "music": [
                "https://musicfeeds.com.au/feed/",                 # Music Feeds
                "https://www.theguardian.com/music/rss",           # The Guardian Music (international)
                "https://www.abc.net.au/triplej/feed/",            # Triple J Music
                "https://www.news.com.au/entertainment/music/rss", # News.com.au Music
            ],
            "finance": [
                "https://www.smh.com.au/rss/business.xml",         # Sydney Morning Herald Business
                "https://feeds.feedburner.com/afr",                # Australian Financial Review
                "https://www.news.com.au/business/rss",            # News.com.au Business
                "https://www.abc.net.au/news/business/rss",        # ABC News Business
            ]
        }
    
    def set_memory_config(self, max_messages: int = 10, max_age_hours: int = 24) -> None:
        """Configure memory settings"""
        self.memory.set_config(max_messages, max_age_hours)
        logger.info(f"Memory config updated: max_messages={max_messages}, max_age_hours={max_age_hours}")
    
    def get_conversation_history(self, session_id: str, max_messages: Optional[int] = None) -> List[Dict]:
        """Get conversation history for a session"""
        messages = self.memory.get_history(session_id, max_messages)
        return [msg.to_dict() for msg in messages]
    
    def clear_conversation_history(self, session_id: str) -> None:
        """Clear conversation history for a session"""
        self.memory.clear_history(session_id)
        logger.info(f"Cleared conversation history for session: {session_id}")
    
    def has_news_data(self) -> bool:
        """Check if the system has news data available for RAG"""
        try:
            return self.news_collection.count() > 0
        except:
            return False
    
    def get_news_summary(self) -> str:
        """Get a summary of available news data"""
        if not self.has_news_data():
            return "No news data available. Please run the news pipeline first."
        
        try:
            # Get all documents from collection
            results = self.news_collection.get()
            metadatas = results['metadatas']
            
            categories = {}
            for metadata in metadatas:
                category = metadata.get('category', 'unknown')
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1
            
            total_articles = len(metadatas)
            summary_parts = [f"Available news data: {total_articles} articles"]
            for category, count in categories.items():
                summary_parts.append(f"- {category.title()}: {count} articles")
            
            return "\n".join(summary_parts)
        except Exception as e:
            logger.warning(f"Error getting news summary: {e}")
            return "Error retrieving news data summary."
    
    
    def _advanced_retrieval(self, enhanced_question: str, original_question: str) -> Dict:
        """Advanced retrieval system using vector database semantic search"""
        logger.info(f"Vector database retrieval for: '{original_question}'")
        
        try:
            # Check if collection has data
            collection_count = self.news_collection.count()
            logger.info(f"Collection has {collection_count} documents")
            
            if collection_count == 0:
                logger.warning("Collection is empty, using fallback")
                return self._fallback_retrieval(original_question)
            
            # Generate query embedding
            query_embedding = self._generate_embedding(original_question)
            
            # Search vector database with more results
            results = self.news_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(10, collection_count),  # Get more results
                include=['documents', 'metadatas', 'distances']
            )
            
            # Extract results
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            # Convert distances to similarity scores (1 - distance)
            similarity_scores = [1 - dist for dist in distances]
            
            logger.info(f"Found {len(documents)} relevant documents")
            logger.info(f"Top similarity scores: {[f'{score:.3f}' for score in similarity_scores[:3]]}")
            logger.info(f"Top categories: {[meta.get('category', 'unknown') for meta in metadatas[:3]]}")
            
            # Filter by minimum similarity threshold
            min_similarity = 0.3  # Lower threshold for better recall
            filtered_docs = []
            filtered_metadatas = []
            
            for i, score in enumerate(similarity_scores):
                if score >= min_similarity:
                    filtered_docs.append(documents[i])
                    filtered_metadatas.append(metadatas[i])
            
            logger.info(f"After filtering (threshold {min_similarity}): {len(filtered_docs)} documents")
            
            return {
                'documents': [filtered_docs],
                'metadatas': [filtered_metadatas],
                'similarity_scores': [s for s in similarity_scores if s >= min_similarity]
            }
            
        except Exception as e:
            logger.warning(f"Vector database retrieval failed: {e}, using fallback")
            return self._fallback_retrieval(original_question)
    
    def _fallback_retrieval(self, question: str) -> Dict:
        """Fallback retrieval using keyword matching"""
        logger.info("Using fallback keyword-based retrieval")
        
        try:
            # Get all documents from collection
            results = self.news_collection.get()
            documents = results['documents']
            metadatas = results['metadatas']
            
            if not documents:
                return {'documents': [[]], 'metadatas': [[]]}
            
            # Simple keyword matching
            question_lower = question.lower()
            question_words = set(question_lower.split())
            
            scored_docs = []
            for i, doc in enumerate(documents):
                doc_lower = doc.lower()
                doc_words = set(doc_lower.split())
                
                # Calculate word overlap score
                overlap = len(question_words.intersection(doc_words))
                score = overlap / len(question_words) if question_words else 0
                
                scored_docs.append({
                    'index': i,
                    'score': score,
                    'document': doc,
                    'metadata': metadatas[i]
                })
            
            # Sort by score and take top 5
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            top_docs = scored_docs[:5]
            
            documents = [doc['document'] for doc in top_docs]
            metadatas = [doc['metadata'] for doc in top_docs]
            
            return {
                'documents': [documents],
                'metadatas': [metadatas]
            }
            
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {e}")
            return {'documents': [[]], 'metadatas': [[]]}
    
    
    
    
    async def extract_news(self) -> List[NewsArticle]:
        """Step 1: Extract news from all sources"""
        logger.info("Starting news extraction...")
        articles = []
        successful_sources = []
        failed_sources = []
        
        for category, sources in self.news_sources.items():
            logger.info(f"Processing {category} category with {len(sources)} sources...")
            for source_url in sources:
                try:
                    # Parse RSS feed
                    feed = feedparser.parse(source_url)
                    source_name = feed.feed.get('title', source_url)
                    
                    if not feed.entries:
                        logger.warning(f"No entries found in {source_url}")
                        failed_sources.append((source_url, "No entries found"))
                        continue
                    
                    articles_from_source = 0
                    for entry in feed.entries:  # Process all articles from source
                        # Get full article content
                        full_content = self._extract_article_content(entry.link)
                        
                        # Generate AI summary if we have content
                        ai_summary = self._generate_summary(full_content, entry.get('title', '')) if full_content else entry.get('summary', '')
                        
                        article = NewsArticle(
                            title=entry.get('title', ''),
                            content=full_content,
                            summary=ai_summary,
                            author=entry.get('author', 'Unknown'),
                            source=source_name,
                            url=entry.get('link', ''),
                            category=category,
                            published_date=self._parse_date(entry.get('published'))
                        )
                        articles.append(article)
                        articles_from_source += 1
                    
                    successful_sources.append((source_url, source_name, articles_from_source))
                    logger.info(f"Successfully extracted {articles_from_source} articles from {source_name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to extract from {source_url}: {e}")
                    failed_sources.append((source_url, str(e)))
        
        logger.info(f"Extraction complete: {len(articles)} articles from {len(successful_sources)} sources")
        logger.info(f"Successful sources: {[s[1] for s in successful_sources]}")
        if failed_sources:
            logger.warning(f"Failed sources: {len(failed_sources)} - {[s[0] for s in failed_sources]}")
        
        return articles
    
    def _extract_article_content(self, url: str) -> str:
        """Extract full article content"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except:
            return ""
    
    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Parse date string and ensure timezone consistency"""
        if not date_string:
            return datetime.now()
        try:
            from dateutil import parser
            parsed_date = parser.parse(date_string)
            # Make timezone-naive if it's timezone-aware
            if parsed_date.tzinfo is not None:
                parsed_date = parsed_date.replace(tzinfo=None)
            return parsed_date
        except:
            return datetime.now()
    
    def _generate_summary(self, content: str, title: str) -> str:
        """Generate AI summary using Gemini"""
        if not content or len(content) < 100:
            return ""
        
        # Truncate content if too long
        max_length = 2000
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        prompt = f"""
        Summarize this news article in 2-3 sentences. Focus on the key facts and main points.
        
        Title: {title}
        Content: {content}
        
        Provide a clear, concise summary:
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            # Fallback to simple truncation
            return content[:200] + "..." if len(content) > 200 else content
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini embedding model"""
        try:
            # Clean and prepare text for embedding
            clean_text = text.strip()
            if not clean_text:
                return [0.0] * 768  # Default embedding size
            
            # Generate embedding using the correct Gemini API
            result = self.embedding_model(
                model="models/embedding-001",
                content=clean_text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * 768
    
    def categorize_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Step 2: Categorize articles using Gemini"""
        logger.info("Categorizing articles with AI...")
        
        for article in articles:
            if article.category == "unknown":
                article.category = self._classify_article(article)
        
        return articles
    
    def _classify_article(self, article: NewsArticle) -> str:
        """Classify single article using Gemini"""
        prompt = f"""
        Classify this news article into one category: sports, lifestyle, music, or finance.
        
        Title: {article.title}
        Summary: {article.summary}
        
        Respond with only the category name (lowercase).
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            category = response.text.strip().lower()
            
            if category in ['sports', 'lifestyle', 'music', 'finance']:
                return category
            return 'unknown'
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return 'unknown'
    
    def detect_duplicates(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Step 2: Detect and merge duplicate articles using TF-IDF similarity"""
        logger.info("Detecting duplicates using TF-IDF similarity...")
        
        if len(articles) < 2:
            return articles
        
        try:
            # Use TF-IDF for reliable duplicate detection
            texts = [f"{article.title} {article.summary}" for article in articles]
            
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find similar articles (threshold 0.7 for TF-IDF)
            processed = set()
            unique_articles = []
            
            for i in range(len(articles)):
                if i in processed:
                    continue
                    
                # Find similar articles (lower threshold for better duplicate detection)
                similar_indices = [j for j in range(len(articles)) 
                                 if similarity_matrix[i][j] > 0.5 and j != i]
                
                if similar_indices:
                    # Merge similar articles
                    cluster_articles = [articles[i]] + [articles[j] for j in similar_indices]
                    merged = self._merge_articles(cluster_articles)
                    unique_articles.append(merged)
                    processed.update([i] + similar_indices)
                    logger.info(f"Merged {len(cluster_articles)} similar articles")
                else:
                    unique_articles.append(articles[i])
                    processed.add(i)
            
            logger.info(f"Reduced from {len(articles)} to {len(unique_articles)} unique articles using TF-IDF similarity")
            
            # Log frequency distribution
            frequency_counts = {}
            for article in unique_articles:
                freq = article.frequency
                frequency_counts[freq] = frequency_counts.get(freq, 0) + 1
            logger.info(f"Frequency distribution: {frequency_counts}")
            return unique_articles
            
        except Exception as e:
            logger.warning(f"Duplicate detection failed: {e}, returning original articles")
            return articles
    
    def _detect_duplicates_tfidf(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Fallback TF-IDF duplicate detection"""
        logger.info("Using TF-IDF fallback for duplicate detection...")
        
        # Create TF-IDF vectors for article titles + summaries
        texts = [f"{article.title} {article.summary}" for article in articles]
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find similar articles (threshold 0.7)
            processed = set()
            unique_articles = []
            
            for i in range(len(articles)):
                if i in processed:
                    continue
                    
                # Find similar articles (lower threshold for better duplicate detection)
                similar_indices = [j for j in range(len(articles)) 
                                 if similarity_matrix[i][j] > 0.5 and j != i]
                
                if similar_indices:
                    # Merge similar articles
                    cluster_articles = [articles[i]] + [articles[j] for j in similar_indices]
                    merged = self._merge_articles(cluster_articles)
                    unique_articles.append(merged)
                    processed.update([i] + similar_indices)
                else:
                    unique_articles.append(articles[i])
                    processed.add(i)
            
            logger.info(f"TF-IDF: Reduced from {len(articles)} to {len(unique_articles)} unique articles")
            return unique_articles
            
        except Exception as e:
            logger.warning(f"TF-IDF duplicate detection failed: {e}, returning original articles")
            return articles
    
    def _merge_articles(self, articles: List[NewsArticle]) -> NewsArticle:
        """Merge duplicate articles with improved source tracking"""
        # Use the article with most content as base
        base = max(articles, key=lambda a: len(a.content))
        
        # Collect all unique sources
        all_sources = set()
        for article in articles:
            # Extract source name from "Source +X others" format if needed
            source_name = article.source.split(' +')[0]
            all_sources.add(source_name)
        
        # Set frequency to actual number of unique sources
        base.frequency = len(all_sources)
        
        # Create better source description
        if len(all_sources) > 1:
            base.source = f"{base.source} +{len(all_sources)-1} others"
        
        return base
    
    def create_highlights(self, articles: List[NewsArticle]) -> Dict[str, List[NewsArticle]]:
        """Step 3: Create highlights prioritised by frequency and keywords"""
        logger.info("Creating news highlights...")
        
        highlights = {}
        breaking_keywords = ['breaking', 'urgent', 'major', 'exclusive', 'crisis']
        
        # Group by category
        by_category = {}
        for article in articles:
            if article.category not in by_category:
                by_category[article.category] = []
            by_category[article.category].append(article)
        
        # Create highlights for each category
        for category, category_articles in by_category.items():
            # Calculate importance scores with more sophisticated algorithm
            for article in category_articles:
                score = 0
                
                # Base score from frequency (more weight for higher frequency)
                score += article.frequency * 15
                
                # Boost for breaking news keywords
                text = f"{article.title} {article.summary}".lower()
                breaking_count = sum(1 for keyword in breaking_keywords if keyword in text)
                score += breaking_count * 25
                
                # Boost for recent articles (exponential decay)
                if article.published_date:
                    hours_ago = (datetime.now() - article.published_date).total_seconds() / 3600
                    if hours_ago < 1:
                        score += 30  # Very recent
                    elif hours_ago < 6:
                        score += 20  # Recent
                    elif hours_ago < 24:
                        score += 10  # Today
                    elif hours_ago < 48:
                        score += 5   # Yesterday
                
                # Boost for article length (more content = more important)
                content_length = len(article.content) if article.content else 0
                if content_length > 1000:
                    score += 10
                elif content_length > 500:
                    score += 5
                
                # Boost for title length (more descriptive titles)
                title_length = len(article.title) if article.title else 0
                if title_length > 50:
                    score += 5
                
                # Boost for specific important keywords by category
                category_keywords = {
                    'sports': ['championship', 'final', 'win', 'victory', 'record', 'olympics', 'world cup'],
                    'finance': ['market', 'stock', 'economy', 'inflation', 'rate', 'profit', 'revenue', 'ceo'],
                    'music': ['album', 'concert', 'tour', 'award', 'chart', 'hit', 'release'],
                    'lifestyle': ['health', 'food', 'travel', 'fashion', 'beauty', 'home', 'wellness']
                }
                
                if category in category_keywords:
                    keyword_matches = sum(1 for keyword in category_keywords[category] 
                                         if keyword in text)
                    score += keyword_matches * 8
                
                article.importance_score = score
            
            # Sort by importance and take top articles (no artificial limit)
            sorted_articles = sorted(category_articles, 
                                   key=lambda x: getattr(x, 'importance_score', 0), 
                                   reverse=True)
            # Take top articles with meaningful scores (above threshold)
            meaningful_articles = [article for article in sorted_articles 
                                 if getattr(article, 'importance_score', 0) > 10]
            highlights[category] = meaningful_articles if meaningful_articles else sorted_articles[:10]
        
        return highlights
    
    def update_chatbot_database(self, highlights: Dict[str, List[NewsArticle]]):
        """Update vector database for chatbot RAG"""
        logger.info("Updating vector database for chatbot RAG...")
        
        # Clear existing collection
        try:
            self.chroma_client.delete_collection("news_articles")
            logger.info("Deleted existing collection")
        except Exception as e:
            logger.info(f"Collection deletion: {e}")
        
        # Recreate collection
        self.news_collection = self.chroma_client.create_collection(
            name="news_articles",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Created new collection")
        
        # Prepare data for vector database
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        total_articles = 0
        for category, articles in highlights.items():
            logger.info(f"Processing {len(articles)} articles for category: {category}")
            for idx, article in enumerate(articles):
                # Create document text
                doc_text = f"Title: {article.title}\nSummary: {article.summary}\nSource: {article.source}\nCategory: {category}"
                
                # Generate embedding
                try:
                    embedding = self._generate_embedding(doc_text)
                except Exception as e:
                    logger.warning(f"Embedding generation failed for article {idx}: {e}")
                    continue  # Skip this article
                
                # Prepare metadata
                metadata = {
                    "category": category,
                    "title": article.title,
                    "source": article.source,
                    "url": article.url,
                    "frequency": article.frequency,
                    "author": article.author,
                    "importance_score": getattr(article, 'importance_score', 0)
                }
                
                # Add to collections
                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(f"{category}_{idx}_{total_articles}")
                embeddings.append(embedding)
                
                total_articles += 1
        
        # Add all documents to ChromaDB in batches
        if documents:
            try:
                self.news_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                logger.info(f"Successfully added {total_articles} articles to vector database")
                logger.info(f"Collection size: {self.news_collection.count()}")
                
                # Verify the data was added
                test_results = self.news_collection.get(limit=5)
                logger.info(f"Test query returned {len(test_results['documents'])} documents")
                
            except Exception as e:
                logger.error(f"Failed to add documents to ChromaDB: {e}")
        else:
            logger.warning("No articles to add to vector database")
    
    def chat_about_news(self, question: str, session_id: str = "default", include_memory: bool = True) -> str:
        """Step 5: Chatbot with RAG and conversation memory"""
        logger.info(f"Chat request: '{question}' for session: {session_id}")
        try:
            doc_count = self.news_collection.count()
            logger.info(f"Available documents: {doc_count}")
        except:
            logger.info("Available documents: 0")
        
        # Check if we have any news data at all
        if not self.has_news_data():
            logger.warning("No news data available for RAG")
            return "I don't have any news data available right now. Please run the news pipeline first to get the latest news data, then I'll be able to answer your questions about today's news."
        
        # Get conversation context if memory is enabled
        conversation_context = ""
        if include_memory:
            conversation_context = self.memory.get_recent_context(session_id, max_messages=3)
        
        # Enhance the question with conversation context if available
        enhanced_question = question
        if conversation_context and include_memory:
            enhanced_question = f"""
Previous conversation context:
{conversation_context}

Current question: {question}

Please consider the previous conversation when answering the current question. If the user is referring to something mentioned earlier, acknowledge that context in your response."""
        
        # Find relevant news articles using vector database semantic search
        try:
            results = self._advanced_retrieval(enhanced_question, question)
        except Exception as e:
            logger.error(f"Vector database search failed: {e}")
            return "I encountered an error searching the news database. Please try again."
        
        if not results['documents'] or not results['documents'][0]:
            logger.info("No relevant documents found in vector database")
            
            # Check for follow-up questions or provide general response
            if conversation_context and any(word in question.lower() for word in ['sure', 'really', 'that', 'it', 'this', 'what about', 'tell me more']):
                # Get the last answer from memory
                last_answer = self.memory.get_last_answer(session_id)
                
                if last_answer:
                    return f"Yes, I can confirm the information from our previous conversation: {last_answer}"
                else:
                    return "Yes, I can confirm the information I provided earlier is accurate based on the news highlights."
            else:
                return "I don't have information about that in today's news highlights. Please run the news pipeline first to get the latest news data."
        
        # Create context from relevant articles
        context = "\n\n".join(results['documents'][0])
        
        # Generate response with Gemini
        prompt = f"""
        You are an AI assistant specialising in news analysis and discussion.
        
        Based on these news articles, answer the user's question:
        
        {context}
        
        Question: {enhanced_question}
        
        Provide a helpful, conversational response based on the news information. If the user is asking a follow-up question, acknowledge the previous context naturally.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            answer = response.text.strip()
            
            # Store in memory if enabled
            if include_memory:
                # Extract sources from results
                sources = []
                if results.get('metadatas') and results['metadatas'][0]:
                    for i, metadata in enumerate(results['metadatas'][0]):
                        sources.append({
                            'title': metadata.get('title', ''),
                            'source': metadata.get('source', ''),
                            'url': metadata.get('url', ''),
                            'category': metadata.get('category', ''),
                            'frequency': metadata.get('frequency', 1)
                        })
                
                message = ChatMessage(
                    question=question,  # Store the original question, not the enhanced one
                    answer=answer,
                    sources=sources
                )
                self.memory.add_message(session_id, message)
            
            return answer
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return "Sorry, I encountered an error processing your question."

# Main pipeline function
async def run_news_pipeline(gemini_api_key: str) -> Dict[str, List[NewsArticle]]:
    """Run the complete news aggregation pipeline"""
    aggregator = NewsAggregator(gemini_api_key)
    
    # Step 1: Extract news
    articles = await aggregator.extract_news()
    
    # Step 2: Categorize and detect duplicates
    articles = aggregator.categorize_articles(articles)
    articles = aggregator.detect_duplicates(articles)
    
    # Step 3: Create highlights
    highlights = aggregator.create_highlights(articles)
    
    
    # Update chatbot database
    aggregator.update_chatbot_database(highlights)
    
    return highlights

if __name__ == "__main__":
    API_KEY = os.getenv('GEMINI_API_KEY', 'api-key-here')
    
    if API_KEY == 'api-key-here':
        print("Missing GEMINI_API_KEY environment variable")
        exit(1)
    
    # Run the pipeline
    highlights = asyncio.run(run_news_pipeline(API_KEY))
    
    # Display results
    total = sum(len(articles) for articles in highlights.values())
    print(f"\nGenerated {total} highlights across {len(highlights)} categories")
    
    for category, articles in highlights.items():
        print(f"\n{category.upper()} ({len(articles)} articles):")
        for article in articles:
            print(f"  • {article.title}")
            print(f"    Source: {article.source} | Frequency: {article.frequency}")
    
    # Test chatbot with memory
    print("\nTesting chatbot with memory...")
    aggregator = NewsAggregator(API_KEY)
    
    # First question
    response1 = aggregator.chat_about_news("What are the main sports stories today?", session_id="test_session")
    print(f"Response 1: {response1}")
    
    # Follow-up question to test memory
    response2 = aggregator.chat_about_news("Tell me more about that", session_id="test_session")
    print(f"Response 2: {response2}")
    
    # Show conversation history
    history = aggregator.get_conversation_history("test_session")
    print(f"\nConversation History ({len(history)} exchanges):")
    for i, msg in enumerate(history, 1):
        print(f"  {i}. Q: {msg['question']}")
        print(f"     A: {msg['answer'][:100]}...")
