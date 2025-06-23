"""
Performance Analyzer - Analyzes DAX query performance and compares variants
"""

import logging
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
from pyadomd import Pyadomd
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyzes DAX query performance and provides optimization insights"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.current_database = None
    
    async def connect_to_database(self, database_name: str):
        """Connect to specific database"""
        import re
        if "Initial Catalog=" in self.connection_string:
            self.connection_string = re.sub(
                r'Initial Catalog=[^;]*;?',
                f'Initial Catalog={database_name};',
                self.connection_string
            )
        else:
            self.connection_string += f"Initial Catalog={database_name};"
        
        self.current_database = database_name
    
    async def analyze_query(self, dax_query: str) -> Dict[str, Any]:
        """Comprehensive performance analysis of a DAX query"""
        analysis_result = {
            "query": dax_query,
            "analysis_time": datetime.now().isoformat(),
            "performance_metrics": {},
            "optimization_suggestions": [],
            "query_complexity": {},
            "resource_usage": {}
        }
        
        try:
            # Execute query with timing
            execution_result = await self._execute_with_detailed_timing(dax_query)
            analysis_result["performance_metrics"] = execution_result
            
            # Analyze query complexity
            complexity = self._analyze_query_complexity(dax_query)
            analysis_result["query_complexity"] = complexity
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(
                dax_query, execution_result, complexity
            )
            analysis_result["optimization_suggestions"] = suggestions
            
            # Get resource usage if available
            resource_usage = await self._get_resource_usage()
            analysis_result["resource_usage"] = resource_usage
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            analysis_result["error"] = str(e)
            return analysis_result
    
    async def _execute_with_detailed_timing(self, dax_query: str) -> Dict[str, Any]:
        """Execute query with detailed performance timing"""
        start_time = time.time()
        
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Clean the query
                cleaned_query = self._clean_dax_query(dax_query)
                
                # Execute with timing
                execution_start = time.time()
                cursor.execute(cleaned_query)
                
                # Fetch results
                fetch_start = time.time()
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                fetch_end = time.time()
                
                cursor.close()
                execution_end = time.time()
                
                # Calculate metrics
                total_duration = (execution_end - start_time) * 1000
                execution_duration = (fetch_start - execution_start) * 1000
                fetch_duration = (fetch_end - fetch_start) * 1000
                
                return {
                    "total_duration_ms": total_duration,
                    "execution_duration_ms": execution_duration,
                    "fetch_duration_ms": fetch_duration,
                    "row_count": len(rows),
                    "column_count": len(columns),
                    "rows_per_second": len(rows) / (total_duration / 1000) if total_duration > 0 else 0,
                    "performance_rating": self._get_performance_rating(total_duration),
                    "execution_timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            return {
                "error": str(e),
                "failed_duration_ms": duration,
                "execution_timestamp": datetime.now().isoformat()
            }
    
    def _get_performance_rating(self, duration_ms: float) -> str:
        """Get performance rating based on execution time"""
        if duration_ms < 100:
            return "Excellent"
        elif duration_ms < 500:
            return "Good"
        elif duration_ms < 2000:
            return "Moderate"
        elif duration_ms < 10000:
            return "Slow"
        else:
            return "Very Slow"
    
    def _analyze_query_complexity(self, dax_query: str) -> Dict[str, Any]:
        """Analyze DAX query complexity"""
        import re
        
        complexity = {
            "total_length": len(dax_query),
            "line_count": len(dax_query.split('\\n')),
            "function_count": 0,
            "iterator_functions": [],
            "aggregation_functions": [],
            "filter_functions": [],
            "context_functions": [],
            "table_references": [],
            "measure_references": [],
            "complexity_score": 0
        }
        
        try:            # Count functions
            function_pattern = r'\b([A-Z]+)\s*\('
            functions = re.findall(function_pattern, dax_query.upper())
            complexity["function_count"] = len(functions)
            
            # Categorize functions
            iterator_funcs = ['SUMX', 'AVERAGEX', 'COUNTX', 'MINX', 'MAXX', 'CONCATENATEX', 'PRODUCTX']
            aggregation_funcs = ['SUM', 'AVERAGE', 'COUNT', 'COUNTROWS', 'MIN', 'MAX', 'DISTINCTCOUNT']
            filter_funcs = ['FILTER', 'ALL', 'ALLEXCEPT', 'VALUES', 'DISTINCT', 'KEEPFILTERS', 'REMOVEFILTERS']
            context_funcs = ['CALCULATE', 'CALCULATETABLE', 'EARLIER', 'EARLIEST']
            
            for func in functions:
                if func in iterator_funcs:
                    complexity["iterator_functions"].append(func)
                elif func in aggregation_funcs:
                    complexity["aggregation_functions"].append(func)
                elif func in filter_funcs:
                    complexity["filter_functions"].append(func)
                elif func in context_funcs:
                    complexity["context_functions"].append(func)
              # Find table references
            table_pattern = r"'?([A-Za-z_][A-Za-z0-9_\s]*)'?\["
            tables = list(set(re.findall(table_pattern, dax_query)))
            complexity["table_references"] = tables
            
            # Calculate complexity score
            score = 0
            score += len(complexity["iterator_functions"]) * 3  # Iterators are expensive
            score += len(complexity["filter_functions"]) * 2   # Filters can be expensive
            score += len(complexity["context_functions"]) * 2  # Context transitions
            score += len(complexity["aggregation_functions"]) * 1
            score += len(tables) * 1
            score += complexity["line_count"] * 0.1
            
            complexity["complexity_score"] = round(score, 1)
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            complexity["error"] = str(e)
        
        return complexity
    
    def _generate_optimization_suggestions(self, dax_query: str, 
                                         performance: Dict[str, Any], 
                                         complexity: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on analysis"""
        suggestions = []
        
        try:
            duration = performance.get("total_duration_ms", 0)
            
            # Performance-based suggestions
            if duration > 2000:
                suggestions.append("⚠️ Query is slow (>2s) - Consider optimization")
            
            if duration > 10000:
                suggestions.append("🚨 Query is very slow (>10s) - Urgent optimization needed")
            
            # Complexity-based suggestions
            if complexity.get("complexity_score", 0) > 20:
                suggestions.append("🔍 High complexity query - Review for simplification opportunities")
            
            # Iterator function suggestions
            iterator_funcs = complexity.get("iterator_functions", [])
            if len(iterator_funcs) > 3:
                suggestions.append(f"⚡ Multiple iterator functions detected ({len(iterator_funcs)}) - Consider consolidation")
            
            if "SUMX" in iterator_funcs:
                suggestions.append("💡 Consider replacing SUMX with SUM where possible")
            
            # Filter function suggestions
            filter_funcs = complexity.get("filter_functions", [])
            if "FILTER" in filter_funcs:
                suggestions.append("🎯 Review FILTER usage - Consider KEEPFILTERS or direct filtering")
            
            # Table reference suggestions
            tables = complexity.get("table_references", [])
            if len(tables) > 5:
                suggestions.append(f"📊 Query references many tables ({len(tables)}) - Verify relationships are optimal")
            
            # Row count suggestions
            row_count = performance.get("row_count", 0)
            if row_count > 100000:
                suggestions.append(f"📈 Large result set ({row_count:,} rows) - Consider adding filters")
            
            # DAX pattern suggestions
            dax_upper = dax_query.upper()
            
            if "DISTINCTCOUNT" in dax_upper:
                suggestions.append("🔄 Consider replacing DISTINCTCOUNT with COUNTROWS(VALUES(...)) for better performance")
            
            if "CALCULATE" in dax_upper and "FILTER" in dax_upper:
                suggestions.append("⚙️ Review CALCULATE + FILTER patterns - May benefit from optimization")
            
            if not suggestions:
                if duration < 100:
                    suggestions.append("✅ Query performance is excellent - No immediate optimizations needed")
                else:
                    suggestions.append("✅ Query performance is acceptable - Monitor for future optimization needs")
        
        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            suggestions.append(f"❌ Could not generate suggestions: {str(e)}")
        
        return suggestions
    
    async def compare_variants(self, measure_name: str, 
                             variants: List[Dict[str, str]]) -> Dict[str, Any]:
        """Compare performance of multiple DAX variants"""
        comparison_result = {
            "measure_name": measure_name,
            "comparison_time": datetime.now().isoformat(),
            "variants": [],
            "summary": {}
        }
        
        try:
            baseline_results = None
            
            for i, variant in enumerate(variants):
                variant_name = variant.get("name", f"Variant {i+1}")
                dax_expression = variant.get("dax", "")
                
                if not dax_expression:
                    comparison_result["variants"].append({
                        "name": variant_name,
                        "error": "No DAX expression provided"
                    })
                    continue
                
                try:
                    # Create test query
                    test_query = f"EVALUATE ROW(\"Result\", {dax_expression})"
                    
                    # Execute with timing
                    performance = await self._execute_with_detailed_timing(test_query)
                    
                    # Extract result value for comparison
                    if "error" not in performance:
                        with Pyadomd(self.connection_string) as conn:
                            cursor = conn.cursor()
                            cursor.execute(test_query)
                            result_row = cursor.fetchone()
                            result_value = result_row[0] if result_row else None
                            cursor.close()
                        
                        variant_result = {
                            "name": variant_name,
                            "dax": dax_expression,
                            "duration_ms": performance["total_duration_ms"],
                            "result_value": result_value,
                            "performance_rating": performance["performance_rating"],
                            "results_match_baseline": True  # Will be updated below
                        }
                        
                        # Compare with baseline (first successful variant)
                        if baseline_results is None:
                            baseline_results = result_value
                            variant_result["is_baseline"] = True
                        else:
                            variant_result["is_baseline"] = False
                            variant_result["results_match_baseline"] = self._compare_values(
                                baseline_results, result_value
                            )
                        
                        comparison_result["variants"].append(variant_result)
                    
                    else:
                        comparison_result["variants"].append({
                            "name": variant_name,
                            "dax": dax_expression,
                            "error": performance["error"]
                        })
                
                except Exception as e:
                    comparison_result["variants"].append({
                        "name": variant_name,
                        "dax": dax_expression,
                        "error": str(e)
                    })
            
            # Generate summary
            successful_variants = [v for v in comparison_result["variants"] 
                                 if "error" not in v and "duration_ms" in v]
            
            if successful_variants:
                # Find fastest and slowest
                fastest = min(successful_variants, key=lambda x: x["duration_ms"])
                slowest = max(successful_variants, key=lambda x: x["duration_ms"])
                
                # Calculate improvement
                if slowest["duration_ms"] > 0:
                    improvement_pct = ((slowest["duration_ms"] - fastest["duration_ms"]) 
                                     / slowest["duration_ms"]) * 100
                else:
                    improvement_pct = 0
                
                comparison_result["summary"] = {
                    "total_variants": len(variants),
                    "successful_variants": len(successful_variants),
                    "fastest_variant": fastest["name"],
                    "fastest_duration_ms": fastest["duration_ms"],
                    "slowest_variant": slowest["name"],
                    "slowest_duration_ms": slowest["duration_ms"],
                    "max_improvement_pct": round(improvement_pct, 1),
                    "all_results_match": all(v.get("results_match_baseline", False) 
                                           for v in successful_variants)
                }
            
        except Exception as e:
            logger.error(f"Variant comparison failed: {e}")
            comparison_result["error"] = str(e)
        
        return comparison_result
    
    def _compare_values(self, value1, value2, tolerance: float = 1e-10) -> bool:
        """Compare two values with tolerance for floating point numbers"""
        try:
            if value1 is None and value2 is None:
                return True
            if value1 is None or value2 is None:
                return False
            
            # For numeric values, use tolerance
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                return abs(value1 - value2) <= tolerance
            
            # For other types, use direct comparison
            return value1 == value2
            
        except:
            return False
    
    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage from DMV queries"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                resource_info = {}
                
                # Try to get memory usage
                try:
                    cursor.execute("SELECT SUM([MEMORY_USAGE_KB]) FROM $SYSTEM.DISCOVER_MEMORYUSAGE")
                    memory_result = cursor.fetchone()
                    if memory_result and memory_result[0]:
                        resource_info["memory_usage_kb"] = memory_result[0]
                        resource_info["memory_usage_mb"] = memory_result[0] / 1024
                except:
                    resource_info["memory_usage"] = "Not available"
                
                # Try to get connection count
                try:
                    cursor.execute("SELECT COUNT(*) FROM $SYSTEM.DISCOVER_SESSIONS")
                    session_count = cursor.fetchone()[0]
                    resource_info["active_sessions"] = session_count
                except:
                    resource_info["active_sessions"] = "Unknown"
                
                cursor.close()
                return resource_info
                
        except Exception as e:
            logger.error(f"Resource usage query failed: {e}")
            return {"error": str(e)}
    
    def _clean_dax_query(self, dax_query: str) -> str:
        """Clean DAX query by removing HTML tags and formatting"""
        import re
        
        # Remove HTML/XML tags
        cleaned = re.sub(r'<[^>]+>', '', dax_query)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
