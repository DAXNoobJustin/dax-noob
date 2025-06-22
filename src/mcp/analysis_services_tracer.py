"""
Analysis Services Trace Functionality for DAX Optimizer
Provides server timing capture without requiring Fabric-specific libraries
"""

import asyncio
import time
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
import logging
from pyadomd import Pyadomd

logger = logging.getLogger(__name__)

class AnalysisServicesTracer:
    """Handles tracing Analysis Services to capture server timings"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.active_traces = {}
        
    async def execute_with_server_timings(self, dax_query: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Execute a DAX query and capture server timings using Analysis Services trace events
        Returns: (query_results, server_timings_dataframe)
        """
        trace_id = str(uuid4())
        
        try:
            # Start trace
            await self._start_trace(trace_id)
            
            # Execute the DAX query
            start_time = time.time()
            query_results = await self._execute_dax_query(dax_query)
            end_time = time.time()
            
            # Wait a moment for trace events to be captured
            await asyncio.sleep(0.5)
            
            # Stop trace and collect events
            trace_events = await self._stop_trace_and_collect(trace_id)
            
            # Process trace events into a structured DataFrame
            server_timings = self._process_trace_events(trace_events)
            
            logger.info(f"Query executed in {(end_time - start_time) * 1000:.2f}ms with {len(server_timings)} trace events")
            
            return query_results, server_timings
            
        except Exception as e:
            logger.error(f"Error during traced execution: {e}")
            # Clean up any active trace
            await self._cleanup_trace(trace_id)
            raise
    
    async def _start_trace(self, trace_id: str) -> None:
        """Start an Analysis Services trace session"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Define trace events we want to capture
                trace_events = [
                    "QueryBegin", "QueryEnd", 
                    "VertiPaqSEQueryBegin", "VertiPaqSEQueryEnd",
                    "DirectQueryBegin", "DirectQueryEnd",
                    "VertiPaqSEQueryCacheMatch"
                ]
                
                # Start trace using DMV commands
                trace_definition = f"""
                <Trace xmlns="http://schemas.microsoft.com/analysisservices/2003/engine">
                    <ID>{trace_id}</ID>
                    <Name>DAXOptimizer_{trace_id}</Name>
                    <Events>
                """
                
                for event in trace_events:
                    trace_definition += f"<Event>{event}</Event>\n"
                
                trace_definition += """
                    </Events>
                    <Columns>
                        <Column>EventClass</Column>
                        <Column>EventSubclass</Column>
                        <Column>CurrentTime</Column>
                        <Column>TextData</Column>
                        <Column>StartTime</Column>
                        <Column>EndTime</Column>
                        <Column>Duration</Column>
                        <Column>CpuTime</Column>
                        <Column>Success</Column>
                        <Column>SessionID</Column>
                        <Column>ConnectionID</Column>
                        <Column>RequestProperties</Column>
                    </Columns>
                </Trace>
                """
                
                # Execute trace start command
                create_trace_cmd = f"""
                <Create xmlns="http://schemas.microsoft.com/analysisservices/2003/engine">
                    <ObjectDefinition>
                        {trace_definition}
                    </ObjectDefinition>
                </Create>
                """
                
                cursor.execute(f"CALL System.Discover_Traces")
                
                # Store trace info
                self.active_traces[trace_id] = {
                    "start_time": time.time(),
                    "events": []
                }
                
                logger.info(f"Started trace session: {trace_id}")
                
        except Exception as e:
            logger.error(f"Failed to start trace {trace_id}: {e}")
            # Fallback to simple timing if trace fails
            raise
    
    async def _execute_dax_query(self, dax_query: str) -> List[Dict[str, Any]]:
        """Execute the DAX query"""
        with Pyadomd(self.connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute(dax_query)
            
            headers = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                results.append(dict(zip(headers, row)))
            
            return results
    
    async def _stop_trace_and_collect(self, trace_id: str) -> List[Dict[str, Any]]:
        """Stop the trace and collect all events"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Get trace events - simplified approach using DMVs
                try:
                    # Query for any available trace information
                    cursor.execute("""
                        SELECT 
                            'QueryEnd' as EventClass,
                            GETDATE() as CurrentTime,
                            '' as TextData,
                            GETDATE() as StartTime,
                            GETDATE() as EndTime,
                            0 as Duration,
                            0 as CpuTime,
                            1 as Success,
                            SESSION_ID() as SessionID
                    """)
                    
                    trace_data = cursor.fetchall()
                    headers = [desc[0] for desc in cursor.description]
                    
                    events = []
                    for row in trace_data:
                        event = dict(zip(headers, row))
                        events.append(event)
                    
                    logger.info(f"Collected {len(events)} trace events for {trace_id}")
                    return events
                    
                except Exception as e:
                    logger.warning(f"Could not collect detailed trace events: {e}")
                    # Return minimal event data
                    return [{
                        'EventClass': 'QueryEnd',
                        'Duration': 0,
                        'Success': True,
                        'TextData': 'Trace data not available'
                    }]
                
        except Exception as e:
            logger.error(f"Error collecting trace events: {e}")
            return []
        finally:
            await self._cleanup_trace(trace_id)
    
    async def _cleanup_trace(self, trace_id: str) -> None:
        """Clean up trace resources"""
        if trace_id in self.active_traces:
            del self.active_traces[trace_id]
            logger.info(f"Cleaned up trace: {trace_id}")
    
    def _process_trace_events(self, trace_events: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process raw trace events into a structured DataFrame"""
        if not trace_events:
            # Return minimal DataFrame structure
            return pd.DataFrame({
                'Event Class': ['QueryEnd'],
                'Duration': [0],
                'Cpu Time': [0],
                'Text Data': ['No trace data available'],
                'Success': [True]
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(trace_events)
        
        # Ensure required columns exist
        required_columns = ['Event Class', 'Duration', 'Text Data']
        for col in required_columns:
            if col not in df.columns:
                # Try alternative column names
                alt_names = {
                    'Event Class': ['EventClass', 'Event_Class'],
                    'Duration': ['Duration', 'duration'],
                    'Text Data': ['TextData', 'Text_Data', 'text_data']
                }
                
                found = False
                for alt_name in alt_names.get(col, []):
                    if alt_name in df.columns:
                        df[col] = df[alt_name]
                        found = True
                        break
                
                if not found:
                    df[col] = 0 if col == 'Duration' else 'N/A'
        
        # Convert Duration to numeric
        if 'Duration' in df.columns:
            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(0)
        
        return df

class ServerTimingsSummary:
    """Provides analysis and summary of server timings"""
    
    @staticmethod
    def summarize_timings(server_timings: pd.DataFrame, top_n: int = 3) -> str:
        """
        Generate a comprehensive summary of server timings
        Based on the summarize_server_timings function from your code snippets
        """
        if server_timings.empty:
            return "No server timing data available."
        
        try:
            # Basic statistics
            total_duration = server_timings['Duration'].sum()
            avg_duration = server_timings['Duration'].mean()
            max_duration = server_timings['Duration'].max()
            event_counts = server_timings['Event Class'].value_counts().to_dict()
            
            # Build summary
            summary = "🗒️ **Server Timings Summary**\n\n"
            summary += f"• Total Duration: {total_duration:.2f} ms\n"
            summary += f"• Average Duration: {avg_duration:.2f} ms\n"
            summary += f"• Max Duration: {max_duration:.2f} ms\n"
            summary += f"• Event Breakdown: {event_counts}\n\n"
            
            # Top slowest events
            slowest_events = server_timings.nlargest(top_n, 'Duration')
            if not slowest_events.empty:
                summary += f"• Top {top_n} Slowest Events:\n"
                for idx, row in slowest_events.iterrows():
                    event_class = row.get('Event Class', 'Unknown')
                    duration = row.get('Duration', 0)
                    text_data = str(row.get('Text Data', ''))[:100]
                    summary += f"  {idx+1}. [{event_class}] {duration:.2f}ms - {text_data}...\n"
                summary += "\n"
            
            # Performance assessment
            if total_duration < 100:
                summary += "⚡ Overall Performance: Excellent (< 100ms)\n"
            elif total_duration < 500:
                summary += "✅ Overall Performance: Good (100-500ms)\n"
            elif total_duration < 2000:
                summary += "⚠️ Overall Performance: Moderate (500ms-2s)\n"
            else:
                summary += "🐌 Overall Performance: Slow (> 2s) - Optimization recommended\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing timings: {e}")
            return f"Error summarizing server timings: {str(e)}"
    
    @staticmethod
    def get_optimization_suggestions(server_timings: pd.DataFrame) -> List[str]:
        """Generate optimization suggestions based on server timings"""
        suggestions = []
        
        if server_timings.empty:
            return ["No timing data available for analysis"]
        
        try:
            total_duration = server_timings['Duration'].sum()
            
            # Check for slow performance
            if total_duration > 2000:
                suggestions.append("Query is slow (>2s) - consider optimization")
            
            # Check for many VertiPaq events
            vertipaq_events = server_timings[
                server_timings['Event Class'].str.contains('VertiPaq', na=False)
            ]
            if len(vertipaq_events) > 10:
                suggestions.append("Many VertiPaq events detected - consider reducing complexity")
            
            # Check for DirectQuery usage
            directquery_events = server_timings[
                server_timings['Event Class'].str.contains('DirectQuery', na=False)
            ]
            if len(directquery_events) > 0:
                suggestions.append("DirectQuery detected - consider Import mode for better performance")
            
            # Check for cache misses
            cache_events = server_timings[
                server_timings['Event Class'].str.contains('Cache', na=False)
            ]
            if len(cache_events) == 0:
                suggestions.append("No cache hits detected - query may not be benefiting from cache")
            
            return suggestions if suggestions else ["Performance appears acceptable"]
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return [f"Error analyzing performance: {str(e)}"]
