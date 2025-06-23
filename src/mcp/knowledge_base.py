"""
DAX Knowledge Base - Scrapes and searches DAX optimization content
"""

import asyncio
import logging
import sqlite3
import requests
from bs4 import BeautifulSoup
import re
import time
import os
from typing import Dict, List, Any
from pathlib import Path
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class DAXKnowledgeBase:
    """Manages DAX optimization knowledge base from online sources"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "kb_cache")
        self.db_path = os.path.join(self.cache_dir, "dax_kb.db")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Knowledge sources
        self.sources = {
            "dax_optimizer": {
                "sitemap_url": "https://kb.daxoptimizer.com/sitemap.xml",
                "base_url": "https://kb.daxoptimizer.com",
                "name": "DAX Optimizer KB"
            },
            "sqlbi_patterns": {
                "base_url": "https://www.daxpatterns.com",
                "name": "DAX Patterns"
            }
        }
    
    def _init_database(self):
        """Initialize SQLite database for knowledge storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create articles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    category TEXT,
                    tags TEXT
                )
            """)
            
            # Create full-text search index
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                    title, content, tags, category,
                    content=articles,
                    content_rowid=id
                )
            """)
            
            # Create triggers to keep FTS index updated
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles
                BEGIN
                    INSERT INTO articles_fts(rowid, title, content, tags, category)
                    VALUES (new.id, new.title, new.content, new.tags, new.category);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS articles_ad AFTER DELETE ON articles
                BEGIN
                    DELETE FROM articles_fts WHERE rowid = old.id;
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE ON articles
                BEGIN
                    DELETE FROM articles_fts WHERE rowid = old.id;
                    INSERT INTO articles_fts(rowid, title, content, tags, category)
                    VALUES (new.id, new.title, new.content, new.tags, new.category);
                END
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Knowledge base database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Use FTS5 for full-text search
            search_query = """
                SELECT 
                    a.url,
                    a.title,
                    a.source,
                    a.category,
                    snippet(articles_fts, 1, '<b>', '</b>', '...', 64) as snippet,
                    bm25(articles_fts) as rank
                FROM articles_fts
                JOIN articles a ON articles_fts.rowid = a.id
                WHERE articles_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            
            # Prepare search query with FTS5 syntax
            fts_query = ' OR '.join(query.split())
            
            cursor.execute(search_query, (fts_query, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "url": row[0],
                    "title": row[1],
                    "source": row[2],
                    "category": row[3] or "Unknown",
                    "snippet": row[4],
                    "relevance_score": row[5]
                })
            
            conn.close()
            
            logger.info(f"Knowledge base search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []
    
    async def update_knowledge_base(self) -> str:
        """Update knowledge base from all sources"""
        results = []
        
        # Update DAX Optimizer KB
        try:
            result = await self._update_dax_optimizer_kb()
            results.append(f"DAX Optimizer KB: {result}")
        except Exception as e:
            results.append(f"DAX Optimizer KB: Error - {str(e)}")
        
        # Add other sources as needed
        # For now, focusing on DAX Optimizer KB as it's the main source
        
        return "\\n".join(results)
    
    async def _update_dax_optimizer_kb(self) -> str:
        """Update knowledge base from DAX Optimizer sitemap"""
        try:
            sitemap_url = self.sources["dax_optimizer"]["sitemap_url"]
            
            # Fetch sitemap
            logger.info(f"Fetching sitemap from {sitemap_url}")
            response = requests.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
            # Parse sitemap XML
            root = ET.fromstring(response.content)
            
            # Extract URLs (namespace handling)
            urls = []
            for sitemap in root:
                for child in sitemap:
                    if child.tag.endswith('loc'):
                        url = child.text
                        if url and url.startswith('https://kb.daxoptimizer.com/'):
                            urls.append(url)
            
            logger.info(f"Found {len(urls)} URLs in sitemap")
            
            # Limit to prevent overwhelming the server
            urls = urls[:50]  # Process first 50 URLs
            
            conn = sqlite3.connect(self.db_path)
            updated_count = 0
            skipped_count = 0
            
            for url in urls:
                try:
                    # Check if URL already exists
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM articles WHERE url = ?", (url,))
                    if cursor.fetchone():
                        skipped_count += 1
                        cursor.close()
                        continue
                    cursor.close()
                    
                    # Scrape the page
                    logger.info(f"Scraping: {url}")
                    page_response = requests.get(url, timeout=15)
                    page_response.raise_for_status()
                    
                    soup = BeautifulSoup(page_response.content, 'html.parser')
                    
                    # Extract title
                    title_elem = soup.find('title')
                    title = title_elem.text.strip() if title_elem else url.split('/')[-1]
                    
                    # Extract main content
                    content = self._extract_content(soup)
                    
                    # Extract category from URL or content
                    category = self._extract_category(url, soup)
                    
                    # Extract tags (DAX functions, keywords)
                    tags = self._extract_tags(content)
                    
                    if content and len(content) > 100:  # Only save if substantial content
                        # Insert into database
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO articles (url, title, content, source, category, tags)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (url, title, content, "DAX Optimizer KB", category, tags))
                        conn.commit()
                        cursor.close()
                        
                        updated_count += 1
                        logger.info(f"Added article: {title}")
                    
                    # Be respectful to the server
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
                    continue
            
            conn.close()
            
            return f"Updated {updated_count} articles, skipped {skipped_count} existing"
            
        except Exception as e:
            logger.error(f"Failed to update DAX Optimizer KB: {e}")
            raise
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML"""
        try:
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Try to find main content areas
            main_content = None
            
            # Common content selectors
            content_selectors = [
                'main', 'article', '.content', '.post-content', 
                '.entry-content', '#content', '.main-content'
            ]
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.find('body')
            
            if main_content:
                # Get text content
                text = main_content.get_text(separator=' ', strip=True)
                  # Clean up the text
                text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
                text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
                
                return text.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return ""
    
    def _extract_category(self, url: str, soup: BeautifulSoup) -> str:
        """Extract category from URL or page content"""
        try:
            # Try to extract from URL path
            path_parts = url.split('/')
            if len(path_parts) > 3:
                potential_category = path_parts[3]
                if potential_category and not potential_category.startswith('?'):
                    return potential_category.replace('-', ' ').title()
            
            # Try to extract from page breadcrumbs or headers
            breadcrumbs = soup.select('.breadcrumb li, .breadcrumbs a')
            if breadcrumbs and len(breadcrumbs) > 1:
                return breadcrumbs[1].get_text(strip=True)
            
            # Try category meta tags
            category_meta = soup.find('meta', {'name': 'category'})
            if category_meta:
                return category_meta.get('content', '')
            
            return "General"
            
        except Exception:
            return "General"
    
    def _extract_tags(self, content: str) -> str:
        """Extract DAX functions and relevant tags from content"""
        try:
            tags = set()
            
            # Common DAX functions
            dax_functions = [
                'CALCULATE', 'FILTER', 'SUM', 'SUMX', 'COUNT', 'COUNTROWS',
                'DISTINCTCOUNT', 'VALUES', 'ALL', 'ALLEXCEPT', 'RELATED',
                'RELATEDTABLE', 'USERELATIONSHIP', 'CROSSFILTER', 'TREATAS',
                'KEEPFILTERS', 'REMOVEFILTERS', 'SUMMARIZE', 'SUMMARIZECOLUMNS',
                'ADDCOLUMNS', 'SELECTCOLUMNS', 'TOPN', 'RANKX', 'EARLIER',
                'EARLIEST', 'HASONEVALUE', 'ISBLANK', 'ISERROR', 'IFERROR',
                'SWITCH', 'DIVIDE', 'CONCATENATEX', 'PATHCONTAINS'
            ]
            
            # Find DAX functions in content
            content_upper = content.upper()
            for func in dax_functions:
                if func in content_upper:
                    tags.add(func)
            
            # Find other relevant keywords
            keywords = [
                'optimization', 'performance', 'context transition', 'filter context',
                'row context', 'evaluation context', 'iterator', 'aggregation',
                'relationship', 'star schema', 'snowflake schema'
            ]
            
            content_lower = content.lower()
            for keyword in keywords:
                if keyword in content_lower:
                    tags.add(keyword)
            
            return ', '.join(sorted(tags))
            
        except Exception:
            return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total articles
            cursor.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0]
            
            # Articles by source
            cursor.execute("""
                SELECT source, COUNT(*) 
                FROM articles 
                GROUP BY source
            """)
            by_source = dict(cursor.fetchall())
            
            # Articles by category
            cursor.execute("""
                SELECT category, COUNT(*) 
                FROM articles 
                GROUP BY category
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """)
            by_category = dict(cursor.fetchall())
            
            # Most recent update
            cursor.execute("SELECT MAX(last_updated) FROM articles")
            last_update = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_articles": total_articles,
                "by_source": by_source,
                "by_category": by_category,
                "last_update": last_update
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear the knowledge base cache"""
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                logger.info("Knowledge base cache cleared")
                self._init_database()
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise
