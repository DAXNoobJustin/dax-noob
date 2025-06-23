"""
DAX Analyzer - Core DAX query execution and optimization
Uses DMV queries and XMLA connections without Fabric dependencies
"""

import logging
import pandas as pd
import re
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
import openai
from pyadomd import Pyadomd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DAXAnalyzer:
    """Core DAX analysis and optimization engine"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.current_database = None
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
        else:
            logger.warning("OpenAI API key not found - optimization features will be limited")
            self.openai_client = None
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection and return server information"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Get server version
                cursor.execute("SELECT @@VERSION")
                version = cursor.fetchone()[0] if cursor.rowcount > 0 else "Unknown"
                
                # Get server name
                cursor.execute("SELECT SERVERPROPERTY('ServerName')")
                server_name = cursor.fetchone()[0] if cursor.rowcount > 0 else "Unknown"
                
                cursor.close()
                
                return {
                    "version": version,
                    "server_name": server_name,
                    "connection_time": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise Exception(f"Connection test failed: {str(e)}")
    
    async def list_databases(self) -> List[Dict[str, Any]]:
        """List available databases using DMV queries"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Query for databases
                dmv_query = """
                SELECT 
                    [DATABASE_NAME] as name,
                    [DESCRIPTION] as description,
                    [DATE_MODIFIED] as last_update
                FROM $SYSTEM.DBSCHEMA_CATALOGS
                WHERE [CATALOG_NAME] IS NOT NULL
                ORDER BY [DATABASE_NAME]
                """
                
                cursor.execute(dmv_query)
                databases = []
                
                for row in cursor.fetchall():
                    databases.append({
                        "name": row[0],
                        "description": row[1] if row[1] else "",
                        "last_update": row[2].isoformat() if row[2] else "Unknown"
                    })
                
                cursor.close()
                return databases
                
        except Exception as e:
            logger.error(f"List databases failed: {e}")
            raise Exception(f"Failed to list databases: {str(e)}")
    
    async def connect_to_database(self, database_name: str):
        """Connect to a specific database"""
        # Update connection string with initial catalog
        if "Initial Catalog=" in self.connection_string:
            # Replace existing catalog
            self.connection_string = re.sub(
                r'Initial Catalog=[^;]*;?',
                f'Initial Catalog={database_name};',
                self.connection_string
            )
        else:
            # Add catalog to connection string
            self.connection_string += f"Initial Catalog={database_name};"
        
        self.current_database = database_name
        
        # Test connection to new database
        await self.test_connection()
        logger.info(f"Connected to database: {database_name}")
    
    async def execute_dax_query(self, dax_query: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute DAX query and return results with performance metrics
        Returns: (results, performance_metrics)
        """
        start_time = time.time()
        
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Clean the DAX query
                cleaned_query = self._clean_dax_query(dax_query)
                logger.info(f"Executing DAX query: {cleaned_query[:100]}...")
                
                # Execute query
                cursor.execute(cleaned_query)
                
                # Get results
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = []
                for row in rows:
                    results.append(dict(zip(columns, row)))
                
                cursor.close()
                
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                performance_metrics = {
                    "duration_ms": duration_ms,
                    "row_count": len(results),
                    "column_count": len(columns),
                    "execution_time": datetime.now().isoformat()
                }
                
                logger.info(f"Query executed successfully: {duration_ms:.2f}ms, {len(results)} rows")
                
                return results, performance_metrics
                
        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            logger.error(f"DAX query failed after {duration_ms:.2f}ms: {e}")
            raise Exception(f"DAX query execution failed: {str(e)}")
    
    async def optimize_measure(self, measure_name: str, dax_expression: str, 
                             max_iterations: int = 3, context: str = "") -> Dict[str, Any]:
        """
        Optimize a DAX measure using AI and performance testing
        """
        if not self.openai_client:
            return {
                "measure_name": measure_name,
                "error": "OpenAI API key not configured - optimization not available"
            }
        
        optimization_result = {
            "measure_name": measure_name,
            "original_dax": dax_expression,
            "iterations": [],
            "best_variant": None,
            "baseline_performance": None
        }
        
        try:
            # Get baseline performance
            logger.info(f"Getting baseline performance for {measure_name}")
            baseline_query = f"EVALUATE ROW(\"Result\", {dax_expression})"
            baseline_results, baseline_perf = await self.execute_dax_query(baseline_query)
            
            optimization_result["baseline_performance"] = baseline_perf
            baseline_duration = baseline_perf["duration_ms"]
            
            logger.info(f"Baseline performance: {baseline_duration:.2f}ms")
            
            current_best_dax = dax_expression
            current_best_duration = baseline_duration
            
            # Try optimization iterations
            for iteration in range(max_iterations):
                logger.info(f"Starting optimization iteration {iteration + 1}")
                
                # Generate optimized variant
                optimized_dax = await self._generate_optimized_dax(
                    measure_name, current_best_dax, baseline_duration, context, iteration
                )
                
                if not optimized_dax or optimized_dax.strip() == current_best_dax.strip():
                    logger.info(f"No new variant generated for iteration {iteration + 1}")
                    break
                
                try:
                    # Test optimized variant
                    test_query = f"EVALUATE ROW(\"Result\", {optimized_dax})"
                    variant_results, variant_perf = await self.execute_dax_query(test_query)
                    variant_duration = variant_perf["duration_ms"]
                    
                    # Verify results match
                    results_match = self._verify_results_match(baseline_results, variant_results)
                    
                    improvement_percent = ((baseline_duration - variant_duration) / baseline_duration) * 100
                    
                    iteration_result = {
                        "iteration": iteration + 1,
                        "dax": optimized_dax,
                        "duration_ms": variant_duration,
                        "improvement_percent": improvement_percent,
                        "results_match": results_match,
                        "performance": variant_perf
                    }
                    
                    if results_match and variant_duration < current_best_duration:
                        current_best_dax = optimized_dax
                        current_best_duration = variant_duration
                        optimization_result["best_variant"] = iteration_result
                        logger.info(f"New best variant found: {improvement_percent:.1f}% improvement")
                    
                    optimization_result["iterations"].append(iteration_result)
                    
                except Exception as e:
                    logger.error(f"Iteration {iteration + 1} failed: {e}")
                    optimization_result["iterations"].append({
                        "iteration": iteration + 1,
                        "dax": optimized_dax,
                        "error": str(e)
                    })
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            optimization_result["error"] = str(e)
        
        return optimization_result
    
    async def _generate_optimized_dax(self, measure_name: str, current_dax: str, 
                                    baseline_duration: float, context: str, iteration: int) -> str:
        """Generate optimized DAX variant using OpenAI"""
        
        prompt = f"""
You are an expert DAX optimization consultant. Analyze this DAX measure and provide ONE optimized variant.

Measure Name: {measure_name}
Current DAX:
{current_dax}

Current Performance: {baseline_duration:.2f}ms
Iteration: {iteration + 1}

{context}

Requirements:
1. Return ONLY the optimized DAX expression, no explanations
2. Ensure the measure returns semantically identical results
3. Focus on performance improvements
4. This is iteration {iteration + 1}, try a different optimization approach

Common DAX optimization patterns:
- Replace DISTINCTCOUNT with COUNTROWS(VALUES(...)) when appropriate
- Use KEEPFILTERS instead of FILTER when preserving filter context
- Optimize context transitions with CALCULATE
- Use variables to avoid repeated calculations
- Replace SUMX with SUM where possible
- Use TREATAS instead of complex FILTER expressions
- Consider SUMMARIZE or SUMMARIZECOLUMNS for aggregations
- Minimize iterator usage in favor of aggregation functions

Return only the optimized DAX expression:
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a DAX optimization expert. Return only optimized DAX code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            optimized_dax = response.choices[0].message.content.strip()
            
            # Clean up any markdown formatting
            optimized_dax = re.sub(r'^```(?:dax)?\s*', '', optimized_dax, flags=re.MULTILINE)
            optimized_dax = re.sub(r'\s*```$', '', optimized_dax, flags=re.MULTILINE)
            
            return optimized_dax.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate optimized DAX: {e}")
            return ""
    
    def _verify_results_match(self, baseline: List[Dict[str, Any]], 
                            variant: List[Dict[str, Any]]) -> bool:
        """Verify that two result sets are identical"""
        try:
            if len(baseline) != len(variant):
                return False
            
            if not baseline and not variant:
                return True
            
            # Convert to DataFrames for comparison
            baseline_df = pd.DataFrame(baseline)
            variant_df = pd.DataFrame(variant)
            
            if baseline_df.shape != variant_df.shape:
                return False
            
            # Sort both DataFrames to handle ordering differences
            if not baseline_df.empty:
                baseline_sorted = baseline_df.sort_values(
                    baseline_df.columns.tolist()
                ).reset_index(drop=True)
                variant_sorted = variant_df.sort_values(
                    variant_df.columns.tolist()
                ).reset_index(drop=True)
                
                # Compare with tolerance for floating point numbers
                return baseline_sorted.equals(variant_sorted)
            
            return True
            
        except Exception as e:
            logger.error(f"Result comparison failed: {e}")
            return False
    
    def _clean_dax_query(self, dax_query: str) -> str:
        """Clean DAX query by removing HTML tags and formatting"""
        # Remove HTML/XML tags
        cleaned = re.sub(r'<[^>]+>', '', dax_query)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    async def get_detailed_server_info(self) -> Dict[str, Any]:
        """Get detailed server information using DMV queries"""
        try:
            server_info = {}
            
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Server properties
                properties_query = """
                SELECT [PROPERTY_NAME], [PROPERTY_VALUE]
                FROM $SYSTEM.DISCOVER_PROPERTIES
                WHERE [PROPERTY_NAME] IN (
                    'ServerName', 'ProductName', 'ProductVersion', 
                    'ServerMode', 'Edition', 'BuildNumber'
                )
                """
                
                cursor.execute(properties_query)
                for row in cursor.fetchall():
                    server_info[row[0]] = row[1]
                  # Memory usage
                try:
                    cursor.execute("SELECT [MEMORY_USAGE_KB] FROM $SYSTEM.DISCOVER_MEMORYUSAGE")
                    memory_rows = cursor.fetchall()
                    if memory_rows:
                        total_memory_kb = sum(row[0] for row in memory_rows if row[0])
                        server_info["MemoryUsageMB"] = total_memory_kb / 1024
                except Exception:
                    server_info["MemoryUsageMB"] = "Unknown"
                
                # Connection info
                server_info["CurrentDatabase"] = self.current_database or "Not selected"
                server_info["ConnectionTime"] = datetime.now().isoformat()
                
                cursor.close()
                
            return server_info
            
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            return {"error": str(e)}
    
    async def execute_dmv_query(self, dmv_query: str) -> List[Dict[str, Any]]:
        """Execute a DMV (Dynamic Management View) query"""
        try:
            results, _ = await self.execute_dax_query(dmv_query)
            return results
        except Exception as e:
            logger.error(f"DMV query failed: {e}")
            raise Exception(f"DMV query failed: {str(e)}")
    
    async def get_query_plan(self, dax_query: str) -> Dict[str, Any]:
        """Get query execution plan for a DAX query (if supported)"""
        try:
            # This is a placeholder for query plan analysis
            # In practice, you might need to use SQL Profiler or similar tools
            # to capture query plans from Analysis Services
            
            logger.info("Query plan analysis not yet implemented")
            return {
                "message": "Query plan analysis requires additional tooling",
                "suggestion": "Use SQL Server Profiler or DAX Studio query plans for detailed analysis"
            }
            
        except Exception as e:
            logger.error(f"Query plan analysis failed: {e}")
            return {"error": str(e)}
    
    def define_query_measures(self, query: str) -> str:
        """
        Ensure every measure referenced in the query is declared via DEFINE MEASURE,
        including any nested measures in those definitions. Any existing DEFINE block
        (VARs, comments, etc.) is left untouched. Newly added measures appear immediately
        under the DEFINE keyword.
        """
        
        def normalize(name: str) -> str:
            # Strip out all non-alphanumeric characters and lowercase
            return re.sub(r"[^0-9A-Za-z]", "", name).lower()

        upper_q = query.upper()
        define_start = upper_q.find("DEFINE")
        eval_start = upper_q.find("EVALUATE")

        if 0 <= define_start < eval_start:
            # There is an existing DEFINE … EVALUATE block
            define_block = query[define_start:eval_start]
            main_query = query[eval_start:]
            has_define = True
        else:
            # No existing DEFINE; we will build one
            define_block = ""
            main_query = query
            has_define = False

        # Find all already‐defined measures (inside the DEFINE block)
        def_pattern = re.compile(
            r"MEASURE\s+(?:'[^']+'|\w+)\s*\[\s*([^\]]+)\]", re.IGNORECASE
        )
        raw_existing_defs = set(def_pattern.findall(define_block))
        normalized_existing = {normalize(m) for m in raw_existing_defs}

        # Build a catalog of all measures: MeasureName → (TableName, MeasureExpression)
        measures_info = {}
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                query_measures = """
                SELECT 
                    [MEASURE_NAME] as name,
                    [EXPRESSION] as expression,
                    [MEASURE_CAPTION] as table_name
                FROM $SYSTEM.MDSCHEMA_MEASURES
                WHERE [CUBE_NAME] = (
                    SELECT TOP 1 [CUBE_NAME] 
                    FROM $SYSTEM.MDSCHEMA_CUBES 
                    WHERE [CUBE_TYPE] = 3
                )
                ORDER BY [MEASURE_NAME]
                """
                
                cursor.execute(query_measures)
                
                for row in cursor.fetchall():
                    measure_name = row[0]
                    expression = row[1] if row[1] else ""
                    table_name = row[2] if row[2] else "Model"
                    measures_info[measure_name] = (table_name, expression)
                
                cursor.close()
                
        except Exception as e:
            logger.error(f"Failed to get measures for define_query_measures: {e}")
            return query  # Return original if we can't get measures

        # Create a lookup: normalized_name → actual_measure_name
        measure_lookup = {
            normalize(mname): mname for mname in measures_info.keys()
        }

        # Find every token that looks like [Something] in define_block + main_query
        bracket_pattern = re.compile(r"\[([^\]]+)\]")
        def extract_bracket_tokens(text: str) -> set:
            return set(bracket_pattern.findall(text))

        all_bracket_tokens = extract_bracket_tokens((define_block or "") + main_query)

        # BFS over tokens to collect all missing measures (and their nested children)
        from collections import deque
        to_define = []
        seen = set(normalized_existing)
        queue = deque()

        # Enqueue any bracket‐token that is not already in existing defs and exists in catalog
        for tok in all_bracket_tokens:
            norm_tok = normalize(tok)
            if norm_tok not in seen and norm_tok in measure_lookup:
                queue.append(norm_tok)

        while queue:
            norm_name = queue.popleft()
            if norm_name in seen:
                continue
            # Identify actual measure name from the lookup
            actual_name = measure_lookup[norm_name]
            seen.add(norm_name)

            table_name, expr = measures_info[actual_name]
            to_define.append((actual_name, table_name, expr))

            # Enqueue any child measure references inside this expression
            for child_tok in extract_bracket_tokens(expr):
                child_norm = normalize(child_tok)
                if child_norm not in seen and child_norm in measure_lookup:
                    queue.append(child_norm)

        # If there are no new measures to define, simply return the original query as‐is
        if not to_define and has_define:
            # Ensure the existing DEFINE block ends with exactly one newline
            trimmed = define_block.rstrip("\n")
            define_block_fixed = trimmed + "\n"
            return define_block_fixed + main_query
        elif not to_define and not has_define:
            return query  # no DEFINE needed

        # Build the new DEFINE block
        new_measure_lines = []
        for actual_name, table_name, expr in to_define:
            line = f"\tMEASURE '{table_name}'[{actual_name}] = {expr}"
            new_measure_lines.append(line)

        if has_define:
            # Split the existing DEFINE block into lines, preserving everything
            define_lines = define_block.splitlines()
            # Find the DEFINE line index
            first_idx = 0
            while first_idx < len(define_lines) and define_lines[first_idx].strip() == "":
                first_idx += 1

            # Insert new measures right after the "DEFINE" line
            rebuilt = []
            rebuilt.extend(define_lines[: first_idx + 1])  # up through the "DEFINE" line
            rebuilt.extend(new_measure_lines)              # new MEASURE lines
            rebuilt.extend(define_lines[first_idx + 1 :])  # the original content below

            # Ensure a single trailing newline
            new_define_block = "\n".join(rebuilt).rstrip("\n") + "\n"

        else:
            # No existing DEFINE → create one from scratch
            define_header = "DEFINE"
            new_define_block = define_header + "\n" + "\n".join(new_measure_lines) + "\n"

        # Return the newly constructed DEFINE + the original main query
        return new_define_block + main_query

    async def get_dax_query_dependencies(self, dax_query: str) -> pd.DataFrame:
        """
        Get DAX query dependencies using DMV queries instead of Fabric-specific functions.
        Returns a DataFrame with columns: ['Table Name', 'Column Name']
        """
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Escape quotes in DAX query
                escaped_dax = dax_query.replace('"', '""')
                
                # Use INFO.CALCDEPENDENCY to get dependencies
                dependency_query = f'''
                EVALUATE
                VAR source_query = "{escaped_dax}"
                VAR all_dependencies = SELECTCOLUMNS(
                    INFO.CALCDEPENDENCY("QUERY", source_query),
                        "Referenced Object Type",[REFERENCED_OBJECT_TYPE],
                        "Referenced Table", [REFERENCED_TABLE],
                        "Referenced Object", [REFERENCED_OBJECT]
                    )             
                RETURN all_dependencies
                '''
                
                cursor.execute(dependency_query)
                
                # Convert to DataFrame
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                if not data:
                    return pd.DataFrame(columns=['Table Name', 'Column Name'])
                
                df = pd.DataFrame(data, columns=columns)
                cursor.close()
                
                # Clean up column names (remove brackets if present)
                df.columns = df.columns.str.strip('[]')
                
                # Clean up object type values
                if 'Referenced Object Type' in df.columns:
                    df['Referenced Object Type'] = (
                        df['Referenced Object Type'].str.replace("_", " ").str.title()
                    )
                
                # Get model calc dependencies for nested dependencies
                calc_deps = await self._get_model_calc_dependencies()
                
                # Expand dependencies to include nested ones
                final_df = df.copy()
                for _, row in df.iterrows():
                    ot = row.get('Referenced Object Type', '')
                    object_name = row.get('Referenced Object', '')
                    table_name = row.get('Referenced Table', '')
                    
                    # Find nested dependencies
                    nested = calc_deps[
                        (calc_deps['Object Type'] == ot) &
                        (calc_deps['Object Name'] == object_name) &
                        (calc_deps['Table Name'] == table_name)
                    ]
                    
                    if len(nested) > 0:
                        subset = nested[['Referenced Object Type', 'Referenced Table', 'Referenced Object']]
                        final_df = pd.concat([final_df, subset], ignore_index=True)
                
                # Filter to only columns and calc columns
                final_df = final_df[
                    (final_df['Referenced Object Type'].isin(['Column', 'Calc Column'])) &
                    (~final_df['Referenced Object'].str.startswith('RowNumber-', na=False))
                ]
                
                # Rename columns to match expected format
                final_df = final_df.rename(columns={
                    'Referenced Table': 'Table Name',
                    'Referenced Object': 'Column Name'
                })
                
                # Remove duplicates and clean up
                final_df = final_df[['Table Name', 'Column Name']].drop_duplicates().reset_index(drop=True)
                
                return final_df
                
        except Exception as e:
            logger.error(f"Failed to get DAX query dependencies: {e}")
            # Return empty DataFrame on error
            return pd.DataFrame(columns=['Table Name', 'Column Name'])
    
    async def _get_model_calc_dependencies(self) -> pd.DataFrame:
        """
        Get all model calculation dependencies using DMV queries.
        This replaces the Fabric-specific get_model_calc_dependencies function.
        """
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT
                    [TABLE] AS [Table Name],
                    [OBJECT] AS [Object Name],
                    [OBJECT_TYPE] AS [Object Type],
                    [EXPRESSION] AS [Expression],
                    [REFERENCED_TABLE] AS [Referenced Table],
                    [REFERENCED_OBJECT] AS [Referenced Object],
                    [REFERENCED_OBJECT_TYPE] AS [Referenced Object Type]
                FROM $SYSTEM.DISCOVER_CALC_DEPENDENCY
                """
                
                cursor.execute(query)
                
                # Convert to DataFrame
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data, columns=columns)
                cursor.close()
                
                # Clean up object types
                df['Object Type'] = df['Object Type'].str.replace('_', ' ').str.title()
                df['Referenced Object Type'] = df['Referenced Object Type'].str.replace('_', ' ').str.title()
                
                # Add full object names
                df['Full Object Name'] = df['Table Name'] + '[' + df['Object Name'] + ']'
                df['Referenced Full Object Name'] = df['Referenced Table'] + '[' + df['Referenced Object'] + ']'
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to get model calc dependencies: {e}")
            return pd.DataFrame()
