"""
Performance Analyzer - Analyzes DAX query performance and compares variants
"""

import logging
import time
import re
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
        """Analyze the complexity of a DAX query"""
        complexity = {
            'total_length': len(dax_query),
            'line_count': len(dax_query.split('\n')),
            'function_count': 0,
            'iterator_functions': [],
            'aggregation_functions': [],
            'filter_functions': [],
            'context_functions': [],
            'table_references': [],
            'measure_references': [],
            'complexity_score': 0,
            'complexity_rating': 'Simple'
        }
        
        # DAX functions categorization
        iterator_functions = ['SUMX', 'AVERAGEX', 'COUNTX', 'MINX', 'MAXX', 'PRODUCTX']
        aggregation_functions = ['SUM', 'AVERAGE', 'COUNT', 'MIN', 'MAX', 'DISTINCTCOUNT']
        filter_functions = ['FILTER', 'ALL', 'ALLEXCEPT', 'ALLSELECTED', 'KEEPFILTERS']
        context_functions = ['CALCULATE', 'CALCULATETABLE', 'EARLIER', 'EARLIEST']
        
        query_upper = dax_query.upper()
        
        # Count function usage
        for func in iterator_functions:
            if func in query_upper:
                complexity['iterator_functions'].append(func)
                complexity['function_count'] += len(re.findall(rf'\b{func}\b', query_upper))
        
        for func in aggregation_functions:
            if func in query_upper:
                complexity['aggregation_functions'].append(func)
                complexity['function_count'] += len(re.findall(rf'\b{func}\b', query_upper))
        
        for func in filter_functions:
            if func in query_upper:
                complexity['filter_functions'].append(func)
                complexity['function_count'] += len(re.findall(rf'\b{func}\b', query_upper))
        
        for func in context_functions:
            if func in query_upper:
                complexity['context_functions'].append(func)
                complexity['function_count'] += len(re.findall(rf'\b{func}\b', query_upper))
        
        # Find table and measure references
        table_pattern = r"'([^']+)'"
        measure_pattern = r'\[([^\]]+)\]'
        
        complexity['table_references'] = list(set(re.findall(table_pattern, dax_query)))
        complexity['measure_references'] = list(set(re.findall(measure_pattern, dax_query)))
        
        # Calculate complexity score
        score = 0
        score += len(complexity['iterator_functions']) * 3  # Iterator functions are expensive
        score += len(complexity['filter_functions']) * 2
        score += len(complexity['context_functions']) * 2
        score += len(complexity['aggregation_functions']) * 1
        score += complexity['total_length'] // 100  # Length factor
        
        complexity['complexity_score'] = score
        complexity['complexity_rating'] = self._get_complexity_rating(score)
        
        return complexity

    def _get_complexity_rating(self, score: int) -> str:
        """Get complexity rating based on score"""
        if score < 5:
            return "Simple"
        elif score < 15:
            return "Moderate"
        elif score < 30:
            return "Complex"
        else:
            return "Very Complex"

    def _generate_optimization_suggestions(self, dax_query: str, performance: Dict[str, Any], 
                                         complexity: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on analysis"""
        suggestions = []
        
        # Performance-based suggestions
        duration = performance.get('total_duration_ms', 0)
        if duration > 2000:
            suggestions.append("🐌 Query is slow (>2s) - Consider optimization")
            
        if duration > 500:
            suggestions.append("⚠️ Consider adding filters to reduce data volume")
        
        # Complexity-based suggestions
        if len(complexity.get('iterator_functions', [])) > 3:
            suggestions.append("🔄 Multiple iterator functions detected - Consider consolidating")
        
        if 'SUMX' in complexity.get('iterator_functions', []):
            suggestions.append("💡 Replace SUMX with SUM where possible")
        
        if 'DISTINCTCOUNT' in complexity.get('aggregation_functions', []):
            suggestions.append("🎯 Consider using COUNTROWS(VALUES(...)) instead of DISTINCTCOUNT")
        
        if len(complexity.get('filter_functions', [])) > 2:
            suggestions.append("🔍 Multiple filter functions - Consider combining filters")
        
        # Pattern-based suggestions
        query_upper = dax_query.upper()
        if 'FILTER(ALL(' in query_upper:
            suggestions.append("⚡ FILTER(ALL()) can be expensive - Consider alternatives")
        
        if query_upper.count('CALCULATE') > 2:
            suggestions.append("🎯 Multiple CALCULATE functions - Check for optimization opportunities")
        
        return suggestions

    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage information"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Try to get memory usage
                try:
                    cursor.execute("SELECT [MEMORY_USAGE_KB] FROM $SYSTEM.DISCOVER_MEMORYUSAGE")
                    memory_rows = cursor.fetchall()
                    total_memory_kb = sum(row[0] for row in memory_rows if row[0])
                    memory_mb = total_memory_kb / 1024
                except:
                    memory_mb = None
                
                cursor.close()
                
                return {
                    "memory_usage_mb": memory_mb,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.debug(f"Could not get resource usage: {e}")
            return {}

    async def compare_variants(self, measure_name: str, variants: List[Dict[str, str]]) -> Dict[str, Any]:
        """Compare multiple DAX variants for performance"""
        if not variants or len(variants) < 2:
            return {"error": "At least 2 variants required for comparison"}
        
        comparison_result = {
            "measure_name": measure_name,
            "comparison_time": datetime.now().isoformat(),
            "variants": [],
            "summary": {}
        }
        
        baseline_result = None
        
        for i, variant in enumerate(variants):
            variant_name = variant.get("name", f"Variant {i+1}")
            dax_expression = variant.get("dax", "")
            
            try:
                # Test the variant
                test_query = f"EVALUATE ROW(\"Result\", {dax_expression})"
                execution_result = await self._execute_with_detailed_timing(test_query)
                
                if "error" in execution_result:
                    comparison_result["variants"].append({
                        "name": variant_name,
                        "error": execution_result["error"]
                    })
                    continue
                
                # Set baseline (first successful variant)
                if baseline_result is None:
                    baseline_result = execution_result
                
                variant_info = {
                    "name": variant_name,
                    "dax": dax_expression,
                    "duration_ms": execution_result["total_duration_ms"],
                    "performance_rating": execution_result["performance_rating"],
                    "results_match": True,  # Simplified for now
                    "improvement_percent": 0
                }
                
                # Calculate improvement vs baseline
                if baseline_result and baseline_result["total_duration_ms"] > 0:
                    improvement = ((baseline_result["total_duration_ms"] - execution_result["total_duration_ms"]) 
                                 / baseline_result["total_duration_ms"]) * 100
                    variant_info["improvement_percent"] = round(improvement, 1)
                
                comparison_result["variants"].append(variant_info)
                
            except Exception as e:
                comparison_result["variants"].append({
                    "name": variant_name,
                    "error": str(e)
                })
        
        # Generate summary
        successful_variants = [v for v in comparison_result["variants"] if "error" not in v]
        if successful_variants:
            durations = [v["duration_ms"] for v in successful_variants]
            fastest = min(successful_variants, key=lambda x: x["duration_ms"])
            
            comparison_result["summary"] = {
                "fastest_variant": fastest["name"],
                "fastest_duration": fastest["duration_ms"],
                "max_improvement": max([v["improvement_percent"] for v in successful_variants])
            }
        
        return comparison_result

    def _clean_dax_query(self, dax_query: str) -> str:
        """Clean DAX query"""
        import re
        # Remove HTML/XML tags
        cleaned = re.sub(r'<[^>]+>', '', dax_query)
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
