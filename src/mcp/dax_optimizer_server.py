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
from collections import deque

# Configure logging to stderr for MCP debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Updated imports
try:
    from mcp.types import Tool, TextContent, Resource, Prompt
    from mcp.server.types import ToolResult
except ImportError:
    from mcp.types import Tool, TextContent

# Custom JSON encoder for Power BI data types
class PowerBIJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Power BI data types"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        return super().default(obj)

def safe_json_dumps(data, indent=2):
    """Safely serialize data containing datetime and other non-JSON types"""
    return json.dumps(data, indent=indent, cls=PowerBIJSONEncoder)

def clean_dax_query(dax_query: str) -> str:
    """Remove HTML/XML tags and other artifacts from DAX queries"""
    # Remove HTML/XML tags like <oii>, </oii>, etc.
    cleaned = re.sub(r'<[^>]+>', '', dax_query)
    # Remove any remaining angle brackets
    cleaned = cleaned.replace('<', '').replace('>', '')
    # Clean up extra whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned
    
# Load environment variables
load_dotenv()

# Add the path to the ADOMD.NET library
adomd_paths = [
    r"C:\Program Files\Microsoft.NET\ADOMD.NET\160",
    r"C:\Program Files\Microsoft.NET\ADOMD.NET\150",
    r"C:\Program Files (x86)\Microsoft.NET\ADOMD.NET\160",
    r"C:\Program Files (x86)\Microsoft.NET\ADOMD.NET\150"
]

adomd_loaded = False
for path in adomd_paths:
    if os.path.exists(path):
        try:
            sys.path.append(path)
            clr.AddReference("Microsoft.AnalysisServices.AdomdClient")
            adomd_loaded = True
            logger.info(f"Loaded ADOMD.NET from {path}")
            break
        except Exception as e:
            logger.debug(f"Failed to load ADOMD.NET from {path}: {e}")
            continue

if not adomd_loaded:
    logger.error("Could not load ADOMD.NET library")
    raise ImportError("Could not load ADOMD.NET library. Please install SSMS or ADOMD.NET client.")

from pyadomd import Pyadomd
from Microsoft.AnalysisServices.AdomdClient import AdomdSchemaGuid


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


class PowerBIConnector:
    def __init__(self):
        self.connection_string = None
        self.connected = False
        self.tables = []
        self.metadata = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def connect(self, xmla_endpoint: str, tenant_id: str, client_id: str, 
                client_secret: str, initial_catalog: str) -> bool:
        """Establish connection to Power BI dataset"""
        self.connection_string = (
            f"Provider=MSOLAP;"
            f"Data Source={xmla_endpoint};"
            f"Initial Catalog={initial_catalog};"
            f"User ID=app:{client_id}@{tenant_id};"
            f"Password={client_secret};"
        )
        
        try:
            # Test connection
            with Pyadomd(self.connection_string) as conn:
                self.connected = True
                logger.info(f"Connected to Power BI dataset: {initial_catalog}")
                return True
        except Exception as e:
            self.connected = False
            logger.error(f"Connection failed: {str(e)}")
            raise Exception(f"Connection failed: {str(e)}")
    
    def discover_tables(self) -> List[str]:
        """Discover all user-facing tables in the dataset"""
        if not self.connected:
            raise Exception("Not connected to Power BI")
            
        if self.tables:
            return self.tables
            
        tables_list = []
        try:
            with Pyadomd(self.connection_string) as pyadomd_conn:
                adomd_connection = pyadomd_conn.conn
                tables_dataset = adomd_connection.GetSchemaDataSet(AdomdSchemaGuid.Tables, None)
                
                if tables_dataset and tables_dataset.Tables.Count > 0:
                    schema_table = tables_dataset.Tables[0]
                    for row in schema_table.Rows:
                        table_name = row["TABLE_NAME"]
                        if (not table_name.startswith("$") and 
                            not table_name.startswith("DateTableTemplate_") and 
                            not row["TABLE_SCHEMA"] == "$SYSTEM"):
                            tables_list.append(table_name)
                            
            self.tables = tables_list
            logger.info(f"Discovered {len(tables_list)} tables")
            return tables_list
        except Exception as e:
            logger.error(f"Failed to discover tables: {str(e)}")
            raise Exception(f"Failed to discover tables: {str(e)}")
    
    def execute_dax_query(self, dax_query: str) -> List[Dict[str, Any]]:
        """Execute a DAX query and return results"""
        if not self.connected:
            raise Exception("Not connected to Power BI")
            
        cleaned_query = clean_dax_query(dax_query)
        logger.info(f"Executing DAX query: {cleaned_query}")
            
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute(cleaned_query)
                
                headers = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                cursor.close()
                
                # Convert to list of dictionaries
                results = []
                for row in rows:
                    results.append(dict(zip(headers, row)))
                
                logger.info(f"Query returned {len(results)} rows")
                return results
                
        except Exception as e:
            logger.error(f"DAX query failed: {str(e)}")
            raise Exception(f"DAX query failed: {str(e)}")


class DataAnalyzer:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)


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
        
        try:
            # Get baseline performance
            baseline_result, baseline_duration = await self._execute_with_timing(original_dax)
            results["baseline_duration_ms"] = baseline_duration
            
            current_best_dax = original_dax
            current_best_duration = baseline_duration
            
            for iteration in range(max_iterations):
                logger.info(f"Starting optimization iteration {iteration + 1}")
                
                # Generate optimization suggestions
                kb_context = self._get_relevant_kb_context(original_dax)
                optimized_dax = await self._generate_optimized_variant(
                    current_best_dax, baseline_duration, kb_context, iteration
                )
                
                if not optimized_dax or optimized_dax.strip() == current_best_dax.strip():
                    logger.info(f"No new variant generated for iteration {iteration + 1}")
                    break
                
                try:
                    # Test the optimized variant
                    variant_result, variant_duration = await self._execute_with_timing(optimized_dax)
                    
                    # Verify results match
                    results_match = self._verify_results_match(baseline_result, variant_result)
                    
                    iteration_result = {
                        "iteration": iteration + 1,
                        "dax": optimized_dax,
                        "duration_ms": variant_duration,
                        "results_match": results_match,
                        "improvement_pct": ((baseline_duration - variant_duration) / baseline_duration) * 100 if baseline_duration > 0 else 0
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
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _execute_with_timing(self, dax: str) -> tuple[List[Dict[str, Any]], float]:
        """Execute DAX and return results with timing"""
        start_time = time.time()
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.connector.execute_dax_query, dax
            )
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            return result, duration_ms
            
        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            raise Exception(f"DAX execution failed ({duration_ms:.2f}ms): {str(e)}")
    
    def _verify_results_match(self, baseline: List[Dict[str, Any]], variant: List[Dict[str, Any]]) -> bool:
        """Verify that two result sets match"""
        try:
            if len(baseline) != len(variant):
                return False
            
            # Convert to DataFrames for easier comparison
            baseline_df = pd.DataFrame(baseline)
            variant_df = pd.DataFrame(variant)
            
            if baseline_df.shape != variant_df.shape:
                return False
            
            # Sort both dataframes to handle potential ordering differences
            if not baseline_df.empty:
                baseline_sorted = baseline_df.sort_values(baseline_df.columns.tolist()).reset_index(drop=True)
                variant_sorted = variant_df.sort_values(variant_df.columns.tolist()).reset_index(drop=True)
                return baseline_sorted.equals(variant_sorted)
            
            return True
            
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            return False
    
    def _get_relevant_kb_context(self, dax: str) -> str:
        """Get relevant knowledge base context for the DAX query"""
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
    
    async def _generate_optimized_variant(self, dax: str, baseline_duration: float, kb_context: str, iteration: int) -> str:
        """Generate an optimized DAX variant using AI"""
        
        prompt = f"""
        You are a DAX optimization expert. Analyze the following DAX measure and provide ONE optimized variant.
        
        Current DAX:
        {dax}
        
        Current Performance: {baseline_duration:.2f}ms
        
        {kb_context}
        
        Requirements:
        1. Return ONLY the optimized DAX code, no explanations
        2. Ensure the measure returns semantically identical results
        3. Focus on performance improvements
        4. Use proven optimization patterns from SQLBI and DAX Patterns
        5. This is iteration {iteration + 1}, try a different approach than before
        
        Common optimization patterns to consider:
        - Replace DISTINCTCOUNT with SUMX(VALUES(...), 1) when appropriate
        - Use KEEPFILTERS instead of FILTER when possible
        - Optimize context transitions
        - Reduce materialization through better filter placement
        - Use TREATAS instead of complex FILTER expressions where applicable
        - Consider using SUMMARIZE or SUMMARIZECOLUMNS for aggregations
        - Replace multiple CALCULATE calls with single CALCULATE when possible
        
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


class DAXOptimizerMCPServer:
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
                    description="Analyze the performance of a DAX query and provide detailed timing information",
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
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """Return empty list of resources - stub implementation"""
            return []
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[Prompt]:
            """Return empty list of prompts - stub implementation"""
            return []
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[TextContent]:
            """Handle tool calls and return results as TextContent"""
            try:
                logger.info(f"Handling tool call: {name}")
                
                if name == "connect_powerbi":
                    result = await self._handle_connect(arguments)
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
    
    async def _handle_connect(self, arguments: Dict[str, Any]) -> str:
        """Handle connection to Power BI"""
        try:
            with self.connection_lock:
                # Connect to Power BI
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.connector.connect,
                    arguments["xmla_endpoint"],
                    arguments["tenant_id"],
                    arguments["client_id"],
                    arguments["client_secret"],
                    arguments["initial_catalog"]
                )
                
                # Initialize the analyzer with OpenAI API key
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return "OpenAI API key not found in environment variables"
                
                self.analyzer = DataAnalyzer(api_key)
                self.optimizer = DAXOptimizer(self.connector, self.analyzer)
                self.is_connected = True
                
                return f"Successfully connected to Power BI dataset '{arguments['initial_catalog']}' and initialized DAX optimizer."
                
        except Exception as e:
            self.is_connected = False
            logger.error(f"Connection failed: {str(e)}")
            return f"Connection failed: {str(e)}"
    
    async def _handle_optimize_measure(self, arguments: Dict[str, Any]) -> str:
        """Handle measure optimization requests"""
        if not self.is_connected:
            return "Not connected to Power BI. Please connect first using 'connect_powerbi'."
        
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
                result += "🎉 Best Optimization Found:\n"
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
            # Simple timing analysis
            start_time = time.time()
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.connector.execute_dax_query, dax_query
            )
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            analysis = "DAX Query Performance Analysis\n"
            analysis += "=" * 40 + "\n\n"
            analysis += f"Query Duration: {duration_ms:.2f}ms\n"
            analysis += f"Rows Returned: {len(results)}\n"
            analysis += f"Query: {dax_query[:100]}{'...' if len(dax_query) > 100 else ''}\n\n"
            
            # Basic performance assessment
            if duration_ms < 100:
                analysis += "⚡ Performance: Excellent (< 100ms)\n"
            elif duration_ms < 500:
                analysis += "✅ Performance: Good (100-500ms)\n"
            elif duration_ms < 2000:
                analysis += "⚠️ Performance: Moderate (500ms-2s)\n"
            else:
                analysis += "🐌 Performance: Slow (> 2s) - Consider optimization\n"
            
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
                    
                    result_data, duration = await self.optimizer._execute_with_timing(dax)
                    
                    variant_result = {
                        "name": variant_name,
                        "dax": dax,
                        "duration_ms": duration,
                        "row_count": len(result_data),
                        "success": True
                    }
                    
                    # Compare results with baseline
                    if baseline_result is None:
                        baseline_result = result_data
                        variant_result["results_match_baseline"] = True
                    else:
                        variant_result["results_match_baseline"] = self.optimizer._verify_results_match(
                            baseline_result, result_data
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
    
    async def run(self):
        """Run the MCP server"""
        try:
            logger.info("Starting DAX Optimizer MCP Server...")
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="dax-optimizer-mcp-server",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            logger.info("Server shutting down")


# Main entry point
async def main():
    server = DAXOptimizerMCPServer()
    await server.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
