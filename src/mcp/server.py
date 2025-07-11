import asyncio
from typing import Any, Dict, List, Optional
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import os
from dotenv import load_dotenv
import json
from datetime import datetime, date
from decimal import Decimal
import sys
import clr
import openai
import re
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import pandas as pd
import xml.etree.ElementTree as ET
from uuid import uuid4
import time
import sqlite3
import requests
from bs4 import BeautifulSoup
import hashlib
from pathlib import Path
from collections import deque

# Configure logging to stderr for MCP debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# DAX optimization classes and utilities
class DAXKnowledgeBase:
    """Handles scraping and searching the DAX Optimizer knowledge base"""
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "kb_cache")
        self.db_path = os.path.join(self.cache_dir, "dax_kb.db")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for knowledge base"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                title, content, content=articles, content_rowid=id
            )
        """)
        conn.commit()
        conn.close()
    
    async def update_knowledge_base(self) -> str:
        """Scrape the DAX Optimizer sitemap and update knowledge base"""
        try:
            sitemap_url = "https://kb.daxoptimizer.com/sitemap.xml"
            response = requests.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            urls = []
            for url_elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
                loc = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                if loc is not None:
                    urls.append(loc.text)
            
            conn = sqlite3.connect(self.db_path)
            updated_count = 0
            
            for url in urls[:50]:  # Limit to prevent overwhelming the server
                try:
                    # Check if we already have this URL
                    existing = conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()
                    if existing:
                        continue
                    
                    # Scrape the page
                    page_response = requests.get(url, timeout=10)
                    page_response.raise_for_status()
                    
                    soup = BeautifulSoup(page_response.content, 'html.parser')
                    title = soup.find('title').text if soup.find('title') else url.split('/')[-1]
                    
                    # Extract main content
                    content_div = soup.find('div', class_=['content', 'main', 'article']) or soup.find('main') or soup
                    content = content_div.get_text(separator=' ', strip=True) if content_div else ""
                    
                    # Insert into database
                    cursor = conn.execute(
                        "INSERT INTO articles (url, title, content) VALUES (?, ?, ?)",
                        (url, title, content)
                    )
                    conn.execute(
                        "INSERT INTO articles_fts (rowid, title, content) VALUES (?, ?, ?)",
                        (cursor.lastrowid, title, content)
                    )
                    updated_count += 1
                    
                    # Small delay to be respectful
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            return f"Updated knowledge base with {updated_count} new articles"
            
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}")
            return f"Error updating knowledge base: {str(e)}"
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Search the knowledge base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT a.url, a.title, snippet(articles_fts, 2, '<b>', '</b>', '...', 32) as snippet
                FROM articles_fts 
                JOIN articles a ON articles_fts.rowid = a.id
                WHERE articles_fts MATCH ?
                ORDER BY bm25(articles_fts)
                LIMIT ?
            """, (query, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "url": row[0],
                    "title": row[1],
                    "snippet": row[2]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []


class DAXTraceCollector:
    """Handles collecting server timings from Analysis Services traces"""
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.trace_id = None
        
    def start_trace(self) -> str:
        """Start a trace session and return trace ID"""
        try:
            with Pyadomd(self.connection_string) as conn:
                # Create trace definition
                trace_definition = """
                <Trace>
                    <ID>{trace_id}</ID>
                    <Name>DAX Optimization Trace</Name>
                    <ddl200:XEvent>
                        <event_session name="DAXOptimization" dispatchLatency="0">
                            <event package="AS" name="VertiPaqSEQueryEnd"/>
                            <event package="AS" name="VertiPaqSEQueryBegin"/>
                            <event package="AS" name="DirectQueryEnd"/>
                            <event package="AS" name="DirectQueryBegin"/>
                            <event package="AS" name="QueryEnd"/>
                            <event package="AS" name="QueryBegin"/>
                            <target package="package0" name="ring_buffer">
                                <parameter name="max_memory" value="4096"/>
                            </target>
                        </event_session>
                    </ddl200:XEvent>
                </Trace>
                """.format(trace_id=str(uuid4()))
                
                # Start the trace
                cursor = conn.cursor()
                cursor.execute(f"CALL SYSTEM$DISCOVER_TRACE_START('{trace_definition}')")
                result = cursor.fetchone()
                self.trace_id = result[0] if result else None
                cursor.close()
                
                return self.trace_id
                
        except Exception as e:
            logger.error(f"Failed to start trace: {e}")
            raise
    
    def get_trace_events(self) -> pd.DataFrame:
        """Get events from the active trace"""
        if not self.trace_id:
            return pd.DataFrame()
            
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM DISCOVER_TRACES WHERE TraceID = '{self.trace_id}'")
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                cursor.close()
                
                return pd.DataFrame(rows, columns=columns)
                
        except Exception as e:
            logger.error(f"Failed to get trace events: {e}")
            return pd.DataFrame()
    
    def stop_trace(self) -> pd.DataFrame:
        """Stop the trace and return collected events"""
        try:
            events = self.get_trace_events()
            
            if self.trace_id:
                with Pyadomd(self.connection_string) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"CALL SYSTEM$DISCOVER_TRACE_STOP('{self.trace_id}')")
                    cursor.close()
                
                self.trace_id = None
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to stop trace: {e}")
            return pd.DataFrame()


class DAXOptimizer:
    """Main DAX optimization engine"""
    def __init__(self, connector, analyzer):
        self.connector = connector
        self.analyzer = analyzer
        self.kb = DAXKnowledgeBase()
        
    async def optimize_measure(self, measure_name: str, original_dax: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Optimize a single DAX measure through iterative improvements"""
        results = {
            "measure_name": measure_name,
            "original_dax": original_dax,
            "iterations": [],
            "best_variant": None,
            "improvement_pct": 0
        }
        
        # Get baseline performance
        baseline_result, baseline_timings = await self._execute_with_timings(original_dax)
        baseline_duration = self._extract_total_duration(baseline_timings)
        
        results["baseline_duration_ms"] = baseline_duration
        
        current_best_dax = original_dax
        current_best_duration = baseline_duration
        
        for iteration in range(max_iterations):
            logger.info(f"Starting optimization iteration {iteration + 1}")
            
            # Generate optimization suggestions
            kb_context = self._get_relevant_kb_context(original_dax)
            optimized_dax = await self._generate_optimized_variant(
                current_best_dax, baseline_timings, kb_context, iteration
            )
            
            if not optimized_dax or optimized_dax.strip() == current_best_dax.strip():
                logger.info(f"No new variant generated for iteration {iteration + 1}")
                break
            
            try:
                # Test the optimized variant
                variant_result, variant_timings = await self._execute_with_timings(optimized_dax)
                variant_duration = self._extract_total_duration(variant_timings)
                
                # Verify results match
                results_match = self._verify_results_match(baseline_result, variant_result)
                
                iteration_result = {
                    "iteration": iteration + 1,
                    "dax": optimized_dax,
                    "duration_ms": variant_duration,
                    "results_match": results_match,
                    "improvement_pct": ((baseline_duration - variant_duration) / baseline_duration) * 100
                }
                
                if results_match and variant_duration < current_best_duration:
                    current_best_dax = optimized_dax
                    current_best_duration = variant_duration
                    results["best_variant"] = iteration_result
                    logger.info(f"New best variant found: {iteration_result['improvement_pct']:.1f}% improvement")
                
                results["iterations"].append(iteration_result)
                
            except Exception as e:
                logger.error(f"Iteration {iteration + 1} failed: {e}")
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "dax": optimized_dax,
                    "error": str(e)
                })
        
        if results["best_variant"]:
            results["improvement_pct"] = results["best_variant"]["improvement_pct"]
        
        return results
    
    async def _execute_with_timings(self, dax: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Execute DAX with server timing collection"""
        trace_collector = DAXTraceCollector(self.connector.connection_string)
        
        try:
            # Start trace
            trace_collector.start_trace()
            
            # Execute query
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.connector.execute_dax_query, dax
            )
            
            # Small delay to ensure trace events are captured
            await asyncio.sleep(1)
            
            # Stop trace and get timings
            timings = trace_collector.stop_trace()
            
            return pd.DataFrame(result), timings
            
        except Exception as e:
            trace_collector.stop_trace()
            raise
    
    def _extract_total_duration(self, timings_df: pd.DataFrame) -> float:
        """Extract total query duration from timings"""
        if timings_df.empty:
            return 0.0
        
        # Look for QueryEnd events
        query_end_events = timings_df[timings_df.get("EventClass", "") == "QueryEnd"]
        if not query_end_events.empty:
            return float(query_end_events["Duration"].iloc[0])
        
        # Fallback: sum all durations
        return float(timings_df.get("Duration", pd.Series([0])).sum())
    
    def _verify_results_match(self, baseline: pd.DataFrame, variant: pd.DataFrame) -> bool:
        """Verify that two result sets match"""
        try:
            if baseline.shape != variant.shape:
                return False
            
            # Sort both dataframes to handle potential ordering differences
            baseline_sorted = baseline.sort_values(baseline.columns.tolist()).reset_index(drop=True)
            variant_sorted = variant.sort_values(variant.columns.tolist()).reset_index(drop=True)
            
            return baseline_sorted.equals(variant_sorted)
            
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            return False
    
    def _get_relevant_kb_context(self, dax: str) -> str:
        """Get relevant knowledge base context for the DAX query"""
        # Extract key terms from DAX for search
        search_terms = []
        
        # Common DAX functions to look for
        dax_functions = re.findall(r'\b[A-Z]+\b', dax.upper())
        search_terms.extend(dax_functions[:5])  # Top 5 functions
        
        # Search knowledge base
        kb_results = []
        for term in search_terms[:3]:  # Limit searches
            results = self.kb.search(term, limit=2)
            kb_results.extend(results)
        
        # Format context
        context = "Relevant DAX optimization knowledge:\n"
        for result in kb_results[:5]:  # Top 5 results
            context += f"- {result['title']}: {result['snippet']}\n"
            context += f"  URL: {result['url']}\n\n"
        
        return context
    
    async def _generate_optimized_variant(self, dax: str, timings: pd.DataFrame, kb_context: str, iteration: int) -> str:
        """Generate an optimized DAX variant using AI"""
        timings_summary = self._summarize_timings(timings)
        
        prompt = f"""
        You are a DAX optimization expert. Analyze the following DAX measure and provide ONE optimized variant.
        
        Current DAX:
        {dax}
        
        Performance Analysis:
        {timings_summary}
        
        {kb_context}
        
        Requirements:
        1. Return ONLY the optimized DAX code, no explanations
        2. Ensure the measure returns semantically identical results
        3. Focus on performance improvements based on the timing analysis
        4. Use proven optimization patterns from SQLBI and DAX Patterns
        5. This is iteration {iteration + 1}, so try a different approach than before
        
        Common optimization patterns to consider:
        - Replace DISTINCTCOUNT with SUMX(VALUES(...), 1) when appropriate
        - Use KEEPFILTERS instead of FILTER when possible
        - Optimize context transitions
        - Reduce materialization through better filter placement
        - Use TREATAS instead of complex FILTER expressions where applicable
        
        Return only the optimized DAX code:
        """
        
        try:
            response = self.analyzer.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a DAX optimization expert. Return only optimized DAX code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            optimized_dax = response.choices[0].message.content.strip()
            # Clean up any markdown formatting
            optimized_dax = re.sub(r'^```(?:dax)?\s*', '', optimized_dax, flags=re.MULTILINE)
            optimized_dax = re.sub(r'\s*```$', '', optimized_dax, flags=re.MULTILINE)
            
            return optimized_dax.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate optimized variant: {e}")
            return ""
    
    def _summarize_timings(self, timings: pd.DataFrame) -> str:
        """Summarize timing information for AI analysis"""
        if timings.empty:
            return "No timing data available"
        
        # Basic summary
        total_duration = timings.get("Duration", pd.Series([0])).sum()
        event_counts = timings.get("EventClass", pd.Series()).value_counts()
        
        summary = f"Total Duration: {total_duration}ms\n"
        summary += f"Event Breakdown: {dict(event_counts)}\n"
        
        # Top slow operations
        if "TextData" in timings.columns and "Duration" in timings.columns:
            slow_ops = timings.nlargest(3, "Duration")[["EventClass", "Duration", "TextData"]]
            summary += "Slowest Operations:\n"
            for _, row in slow_ops.iterrows():
                text_data = str(row["TextData"])[:100] + "..." if len(str(row["TextData"])) > 100 else str(row["TextData"])
                summary += f"- {row['EventClass']} ({row['Duration']}ms): {text_data}\n"
        
        return summary

class PowerBIMCPServer:
    def __init__(self):
        self.server = Server("dax-optimizer-mcp-server")
        self.connector = PowerBIConnector()
        self.analyzer = None
        self.optimizer = None
        self.kb = DAXKnowledgeBase()
        self.is_connected = False
        self.connection_lock = threading.Lock()
        self.uploaded_files = {}  # Store uploaded file contents
        
        # Setup server handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="connect_powerbi",
                    description="Connect to a Power BI dataset using XMLA endpoint",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "xmla_endpoint": {"type": "string", "description": "Power BI XMLA endpoint URL"},
                            "tenant_id": {"type": "string", "description": "Azure AD Tenant ID"},
                            "client_id": {"type": "string", "description": "Service Principal Client ID"},
                            "client_secret": {"type": "string", "description": "Service Principal Client Secret"},
                            "initial_catalog": {"type": "string", "description": "Dataset name"}
                        },
                        "required": ["xmla_endpoint", "tenant_id", "client_id", "client_secret", "initial_catalog"]
                    }
                ),
                Tool(
                    name="optimize_measure",
                    description="Optimize a DAX measure for better performance while maintaining identical results",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "measure_name": {"type": "string", "description": "Name of the measure to optimize"},
                            "dax_expression": {"type": "string", "description": "Current DAX expression of the measure"},
                            "max_iterations": {"type": "integer", "description": "Maximum optimization iterations (default: 3)", "default": 3}
                        },
                        "required": ["measure_name", "dax_expression"]
                    }
                ),
                Tool(
                    name="analyze_query_performance",
                    description="Analyze the performance of a DAX query and provide detailed server timings",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "dax_query": {"type": "string", "description": "DAX query to analyze"}
                        },
                        "required": ["dax_query"]
                    }
                ),
                Tool(
                    name="search_dax_knowledge",
                    description="Search the DAX optimization knowledge base for best practices and patterns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query for DAX optimization knowledge"},
                            "limit": {"type": "integer", "description": "Maximum number of results to return", "default": 5}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="update_knowledge_base",
                    description="Update the local DAX optimization knowledge base from kb.daxoptimizer.com",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="upload_file_context",
                    description="Upload a file to use as context for DAX optimization",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Name of the file"},
                            "content": {"type": "string", "description": "Content of the file"}
                        },
                        "required": ["filename", "content"]
                    }
                ),
                Tool(
                    name="compare_measure_variants",
                    description="Compare multiple DAX measure variants for performance and correctness",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "measure_name": {"type": "string", "description": "Name of the measure"},
                            "variants": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "Variant name"},
                                        "dax": {"type": "string", "description": "DAX expression"}
                                    },
                                    "required": ["name", "dax"]
                                },
                                "description": "Array of measure variants to compare"
                            }
                        },
                        "required": ["measure_name", "variants"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[TextContent]:
            """Handle tool calls and return results as TextContent"""
            try:
                logger.info(f"Handling tool call: {name}")
                
                if name == "connect_powerbi":
                    result = await self._handle_connect(arguments)
                elif name == "list_tables":
                    result = await self._handle_list_tables()
                elif name == "get_table_info":
                    result = await self._handle_get_table_info(arguments)
                elif name == "query_data":
                    result = await self._handle_query_data(arguments)
                elif name == "execute_dax":
                    result = await self._handle_execute_dax(arguments)
                elif name == "suggest_questions":
                    result = await self._handle_suggest_questions()
                elif name == "optimize_measure":
                    result = await self._handle_optimize_measure(arguments)
                elif name == "analyze_query_performance":
                    result = await self._handle_analyze_performance(arguments)
                elif name == "search_dax_knowledge":
                    result = await self._handle_search_knowledge(arguments)
                elif name == "update_knowledge_base":
                    result = await self._handle_update_kb()
                elif name == "upload_file_context":
                    result = await self._handle_upload_file(arguments)
                elif name == "compare_measure_variants":
                    result = await self._handle_compare_variants(arguments)
                else:
                    logger.warning(f"Unknown tool: {name}")
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                
                # Convert string result to TextContent
                return [TextContent(type="text", text=result)]
                
            except Exception as e:
                logger.error(f"Error executing {name}: {str(e)}", exc_info=True)
                return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]
    
    async def _handle_optimize_measure(self, arguments: Dict[str, Any]) -> str:
        """Handle measure optimization requests"""
        if not self.is_connected:
            return "Not connected to Power BI. Please connect first."
        
        if not self.optimizer:
            return "Optimizer not initialized. Please ensure connection is established."
        
        measure_name = arguments.get("measure_name")
        dax_expression = arguments.get("dax_expression")
        max_iterations = arguments.get("max_iterations", 3)
        
        if not measure_name or not dax_expression:
            return "Please provide both measure_name and dax_expression."
        
        try:
            # Include uploaded file context if available
            context = self._get_file_context()
            if context:
                logger.info(f"Using uploaded file context: {len(context)} characters")
            
            optimization_result = await self.optimizer.optimize_measure(
                measure_name, dax_expression, max_iterations
            )
            
            # Format results
            result = f"DAX Measure Optimization Results for '{measure_name}'\n"
            result += "=" * 50 + "\n\n"
            
            result += f"Baseline Duration: {optimization_result['baseline_duration_ms']:.2f}ms\n\n"
            
            if optimization_result['best_variant']:
                best = optimization_result['best_variant']
                result += f"🎉 Best Optimization Found:\n"
                result += f"Improvement: {best['improvement_pct']:.1f}% faster\n"
                result += f"New Duration: {best['duration_ms']:.2f}ms\n"
                result += f"Results Match: {'✅ Yes' if best['results_match'] else '❌ No'}\n\n"
                result += f"Optimized DAX:\n{best['dax']}\n\n"
            else:
                result += "No improvements found.\n\n"
            
            # Iteration details
            result += "Iteration Details:\n"
            for iteration in optimization_result['iterations']:
                result += f"Iteration {iteration['iteration']}:\n"
                if 'error' in iteration:
                    result += f"  ❌ Error: {iteration['error']}\n"
                else:
                    result += f"  Duration: {iteration['duration_ms']:.2f}ms\n"
                    result += f"  Improvement: {iteration['improvement_pct']:.1f}%\n"
                    result += f"  Results Match: {'✅' if iteration['results_match'] else '❌'}\n"
                result += "\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return f"Optimization failed: {str(e)}"
    
    async def _handle_analyze_performance(self, arguments: Dict[str, Any]) -> str:
        """Handle query performance analysis"""
        if not self.is_connected:
            return "Not connected to Power BI. Please connect first."
        
        dax_query = arguments.get("dax_query")
        if not dax_query:
            return "Please provide a DAX query to analyze."
        
        try:
            trace_collector = DAXTraceCollector(self.connector.connection_string)
            
            # Start trace
            trace_id = trace_collector.start_trace()
            if not trace_id:
                return "Failed to start performance trace."
            
            # Execute query
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.connector.execute_dax_query, dax_query
            )
            
            # Wait a moment for trace events
            await asyncio.sleep(2)
            
            # Get trace events
            timings = trace_collector.stop_trace()
            
            if timings.empty:
                return "No timing data collected. Query may have been too fast or trace failed."
            
            # Generate performance analysis
            analysis = "DAX Query Performance Analysis\n"
            analysis += "=" * 40 + "\n\n"
            
            analysis += f"Query returned {len(results)} rows\n\n"
            
            # Add detailed timing analysis
            analysis += self._analyze_timings(timings)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return f"Performance analysis failed: {str(e)}"
    
    async def _handle_search_knowledge(self, arguments: Dict[str, Any]) -> str:
        """Handle knowledge base search"""
        query = arguments.get("query")
        limit = arguments.get("limit", 5)
        
        if not query:
            return "Please provide a search query."
        
        try:
            results = self.kb.search(query, limit)
            
            if not results:
                return f"No results found for '{query}'. Try updating the knowledge base first."
            
            response = f"DAX Optimization Knowledge Search Results for '{query}'\n"
            response += "=" * 50 + "\n\n"
            
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['title']}\n"
                response += f"   {result['snippet']}\n"
                response += f"   🔗 {result['url']}\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return f"Knowledge search failed: {str(e)}"
    
    async def _handle_update_kb(self) -> str:
        """Handle knowledge base update"""
        try:
            result = await self.kb.update_knowledge_base()
            return f"Knowledge Base Update Complete\n{result}"
        except Exception as e:
            logger.error(f"KB update failed: {e}")
            return f"Knowledge base update failed: {str(e)}"
    
    async def _handle_upload_file(self, arguments: Dict[str, Any]) -> str:
        """Handle file upload for context"""
        filename = arguments.get("filename")
        content = arguments.get("content")
        
        if not filename or not content:
            return "Please provide both filename and content."
        
        self.uploaded_files[filename] = content
        logger.info(f"Uploaded file: {filename} ({len(content)} characters)")
        
        return f"File '{filename}' uploaded successfully. Content will be used as context for DAX optimization."
    
    async def _handle_compare_variants(self, arguments: Dict[str, Any]) -> str:
        """Handle comparing multiple measure variants"""
        if not self.is_connected:
            return "Not connected to Power BI. Please connect first."
        
        measure_name = arguments.get("measure_name")
        variants = arguments.get("variants", [])
        
        if not measure_name or not variants:
            return "Please provide measure_name and variants array."
        
        if len(variants) < 2:
            return "Please provide at least 2 variants to compare."
        
        try:
            comparison_results = []
            baseline_result = None
            
            for i, variant in enumerate(variants):
                variant_name = variant.get("name", f"Variant {i+1}")
                dax = variant.get("dax")
                
                if not dax:
                    continue
                
                try:
                    # Execute with timing
                    if not self.optimizer:
                        return "Optimizer not initialized."
                    
                    result_df, timings_df = await self.optimizer._execute_with_timings(dax)
                    duration = self.optimizer._extract_total_duration(timings_df)
                    
                    variant_result = {
                        "name": variant_name,
                        "dax": dax,
                        "duration_ms": duration,
                        "row_count": len(result_df),
                        "success": True
                    }
                    
                    # Compare results with baseline
                    if baseline_result is None:
                        baseline_result = result_df
                        variant_result["results_match_baseline"] = True
                    else:
                        variant_result["results_match_baseline"] = self.optimizer._verify_results_match(
                            baseline_result, result_df
                        )
                    
                    comparison_results.append(variant_result)
                    
                except Exception as e:
                    comparison_results.append({
                        "name": variant_name,
                        "dax": dax,
                        "error": str(e),
                        "success": False
                    })
            
            # Format comparison report
            report = f"DAX Measure Variant Comparison: '{measure_name}'\n"
            report += "=" * 50 + "\n\n"
            
            # Sort by duration (successful variants only)
            successful_variants = [v for v in comparison_results if v.get("success")]
            successful_variants.sort(key=lambda x: x.get("duration_ms", float('inf')))
            
            if successful_variants:
                fastest = successful_variants[0]
                report += f"🏆 Fastest Variant: {fastest['name']} ({fastest['duration_ms']:.2f}ms)\n\n"
            
            report += "Detailed Results:\n"
            for result in comparison_results:
                report += f"\n{result['name']}:\n"
                if result.get("success"):
                    report += f"  ✅ Duration: {result['duration_ms']:.2f}ms\n"
                    report += f"  📊 Rows: {result['row_count']}\n"
                    report += f"  🔍 Results Match Baseline: {'✅' if result['results_match_baseline'] else '❌'}\n"
                else:
                    report += f"  ❌ Error: {result['error']}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Variant comparison failed: {e}")
            return f"Variant comparison failed: {str(e)}"
    
    def _get_file_context(self) -> str:
        """Get context from uploaded files"""
        if not self.uploaded_files:
            return ""
        
        context = "Uploaded File Context:\n"
        for filename, content in self.uploaded_files.items():
            context += f"\n--- {filename} ---\n"
            # Limit content to prevent overwhelming the context
            if len(content) > 5000:
                context += content[:5000] + "\n[... content truncated ...]\n"
            else:
                context += content + "\n"
        
        return context
    
    def _analyze_timings(self, timings_df: pd.DataFrame) -> str:
        """Analyze timing data and return formatted analysis"""
        if timings_df.empty:
            return "No timing data available."
        
        analysis = "Performance Breakdown:\n"
        
        # Total duration
        total_duration = timings_df.get("Duration", pd.Series([0])).sum()
        analysis += f"Total Duration: {total_duration:.2f}ms\n\n"
        
        # Event breakdown
        if "EventClass" in timings_df.columns:
            event_counts = timings_df["EventClass"].value_counts()
            analysis += "Event Breakdown:\n"
            for event, count in event_counts.items():
                event_duration = timings_df[timings_df["EventClass"] == event]["Duration"].sum()
                analysis += f"  {event}: {count} events, {event_duration:.2f}ms total\n"
            analysis += "\n"
        
        # Slowest operations
        if "Duration" in timings_df.columns and len(timings_df) > 0:
            slowest = timings_df.nlargest(5, "Duration")
            analysis += "Top 5 Slowest Operations:\n"
            for _, row in slowest.iterrows():
                event_class = row.get("EventClass", "Unknown")
                duration = row.get("Duration", 0)
                text_data = str(row.get("TextData", ""))[:100]
                analysis += f"  {event_class}: {duration:.2f}ms - {text_data}...\n"
        
        return analysis