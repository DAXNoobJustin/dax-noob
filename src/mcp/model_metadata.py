"""
Model Metadata Extractor - Extracts comprehensive model information using DMV queries
"""

import logging
from typing import Dict, List, Any, Optional
from pyadomd import Pyadomd
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelMetadataExtractor:
    """Extracts model metadata using DMV (Dynamic Management View) queries"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.current_database = None
    
    async def connect_to_database(self, database_name: str):
        """Connect to specific database"""
        # Update connection string with initial catalog
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
    
    async def get_full_metadata(self) -> Dict[str, Any]:
        """Get comprehensive model metadata"""
        try:
            metadata = {
                "database_name": self.current_database,
                "extraction_time": datetime.now().isoformat(),
                "tables": await self.get_tables(),
                "columns": await self.get_columns(),
                "measures": await self.get_measures(),
                "relationships": await self.get_relationships(),
                "hierarchies": await self.get_hierarchies(),
                "partitions": await self.get_partitions()
            }
            
            logger.info(f"Extracted metadata: {len(metadata['tables'])} tables, "
                       f"{len(metadata['measures'])} measures, {len(metadata['relationships'])} relationships")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            raise Exception(f"Metadata extraction failed: {str(e)}")
    
    async def get_tables(self) -> List[Dict[str, Any]]:
        """Get table information"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT 
                    [TABLE_NAME] as name,
                    [TABLE_TYPE] as type,
                    [DESCRIPTION] as description
                FROM $SYSTEM.DBSCHEMA_TABLES
                WHERE [TABLE_TYPE] IN ('TABLE', 'CALCULATED_TABLE')
                ORDER BY [TABLE_NAME]
                """
                
                cursor.execute(query)
                
                tables = []
                for row in cursor.fetchall():
                    table_info = {
                        "name": row[0],
                        "type": row[1],
                        "description": row[2] if row[2] else ""
                    }
                    
                    # Get row count for each table
                    try:
                        row_count_query = f"EVALUATE COUNTROWS('{row[0]}')"
                        cursor.execute(row_count_query)
                        count_result = cursor.fetchone()
                        table_info["row_count"] = count_result[0] if count_result else 0
                    except:
                        table_info["row_count"] = "Unknown"
                    
                    tables.append(table_info)
                
                cursor.close()
                return tables
                
        except Exception as e:
            logger.error(f"Failed to get tables: {e}")
            return []
    
    async def get_columns(self) -> List[Dict[str, Any]]:
        """Get column information"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT 
                    [TABLE_NAME] as table_name,
                    [COLUMN_NAME] as column_name,
                    [DATA_TYPE] as data_type,
                    [IS_NULLABLE] as is_nullable,
                    [DESCRIPTION] as description
                FROM $SYSTEM.DBSCHEMA_COLUMNS
                WHERE [TABLE_NAME] NOT LIKE '$%'
                ORDER BY [TABLE_NAME], [ORDINAL_POSITION]
                """
                
                cursor.execute(query)
                
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        "table_name": row[0],
                        "column_name": row[1],
                        "data_type": row[2],
                        "is_nullable": row[3],
                        "description": row[4] if row[4] else ""
                    })
                
                cursor.close()
                return columns
                
        except Exception as e:
            logger.error(f"Failed to get columns: {e}")
            return []
    
    async def get_measures(self) -> List[Dict[str, Any]]:
        """Get measure information"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT 
                    [MEASURE_NAME] as name,
                    [EXPRESSION] as expression,
                    [DESCRIPTION] as description,
                    [MEASURE_DISPLAY_FOLDER] as display_folder
                FROM $SYSTEM.MDSCHEMA_MEASURES
                WHERE [CUBE_NAME] = (
                    SELECT TOP 1 [CUBE_NAME] 
                    FROM $SYSTEM.MDSCHEMA_CUBES 
                    WHERE [CUBE_TYPE] = 3
                )
                ORDER BY [MEASURE_NAME]
                """
                
                cursor.execute(query)
                
                measures = []
                for row in cursor.fetchall():
                    measures.append({
                        "name": row[0],
                        "expression": row[1] if row[1] else "",
                        "description": row[2] if row[2] else "",
                        "display_folder": row[3] if row[3] else ""
                    })
                
                cursor.close()
                return measures
                
        except Exception as e:
            logger.error(f"Failed to get measures: {e}")
            return []
    
    async def get_relationships(self) -> List[Dict[str, Any]]:
        """Get relationship information"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Try to get relationships from DMV
                query = """
                SELECT 
                    [PK_TABLE] as from_table,
                    [PK_COLUMN] as from_column,
                    [FK_TABLE] as to_table,
                    [FK_COLUMN] as to_column
                FROM $SYSTEM.DBSCHEMA_FOREIGN_KEYS
                ORDER BY [PK_TABLE], [FK_TABLE]
                """
                
                cursor.execute(query)
                
                relationships = []
                for row in cursor.fetchall():
                    relationships.append({
                        "from_table": row[0],
                        "from_column": row[1],
                        "to_table": row[2],
                        "to_column": row[3],
                        "cardinality": "Unknown"  # Would need additional DMV to get this
                    })
                
                cursor.close()
                return relationships
                
        except Exception as e:
            logger.error(f"Failed to get relationships: {e}")
            return []
    
    async def get_hierarchies(self) -> List[Dict[str, Any]]:
        """Get hierarchy information"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT 
                    [HIERARCHY_NAME] as name,
                    [HIERARCHY_CAPTION] as caption,
                    [DIMENSION_UNIQUE_NAME] as dimension,
                    [DESCRIPTION] as description
                FROM $SYSTEM.MDSCHEMA_HIERARCHIES
                WHERE [HIERARCHY_ORIGIN] = 2  -- User-defined hierarchies
                ORDER BY [DIMENSION_UNIQUE_NAME], [HIERARCHY_NAME]
                """
                
                cursor.execute(query)
                
                hierarchies = []
                for row in cursor.fetchall():
                    hierarchies.append({
                        "name": row[0],
                        "caption": row[1] if row[1] else row[0],
                        "dimension": row[2],
                        "description": row[3] if row[3] else ""
                    })
                
                cursor.close()
                return hierarchies
                
        except Exception as e:
            logger.error(f"Failed to get hierarchies: {e}")
            return []
    
    async def get_partitions(self) -> List[Dict[str, Any]]:
        """Get partition information"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # This DMV might not be available in all versions
                query = """
                SELECT 
                    [TABLE_NAME] as table_name,
                    [PARTITION_NAME] as partition_name,
                    [ROWS] as row_count,
                    [RIVIOLATION_COUNT] as ri_violations
                FROM $SYSTEM.DISCOVER_STORAGE_TABLE_COLUMNS
                WHERE [PARTITION_NAME] IS NOT NULL
                ORDER BY [TABLE_NAME], [PARTITION_NAME]
                """
                
                try:
                    cursor.execute(query)
                    
                    partitions = []
                    for row in cursor.fetchall():
                        partitions.append({
                            "table_name": row[0],
                            "partition_name": row[1],
                            "row_count": row[2] if row[2] else 0,
                            "ri_violations": row[3] if row[3] else 0
                        })
                    
                    cursor.close()
                    return partitions
                    
                except:
                    # DMV not available, return empty list
                    cursor.close()
                    return []
                
        except Exception as e:
            logger.error(f"Failed to get partitions: {e}")
            return []
    
    async def get_table_relationships(self, table_name: str) -> List[Dict[str, Any]]:
        """Get relationships for a specific table"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT 
                    [PK_TABLE] as from_table,
                    [PK_COLUMN] as from_column,
                    [FK_TABLE] as to_table,
                    [FK_COLUMN] as to_column
                FROM $SYSTEM.DBSCHEMA_FOREIGN_KEYS
                WHERE [PK_TABLE] = ? OR [FK_TABLE] = ?
                ORDER BY [PK_TABLE], [FK_TABLE]
                """
                
                cursor.execute(query, (table_name, table_name))
                
                relationships = []
                for row in cursor.fetchall():
                    relationships.append({
                        "from_table": row[0],
                        "from_column": row[1],
                        "to_table": row[2],
                        "to_column": row[3]
                    })
                
                cursor.close()
                return relationships
                
        except Exception as e:
            logger.error(f"Failed to get table relationships: {e}")
            return []
    
    async def get_measure_dependencies(self, measure_name: str) -> Dict[str, Any]:
        """Get dependencies for a specific measure"""
        try:
            # Get the measure expression first
            measures = await self.get_measures()
            measure_expression = None
            
            for measure in measures:
                if measure["name"] == measure_name:
                    measure_expression = measure["expression"]
                    break
            
            if not measure_expression:
                return {"error": f"Measure {measure_name} not found"}
            
            # Analyze dependencies from the expression
            dependencies = self._analyze_dax_dependencies(measure_expression)
            
            return {
                "measure_name": measure_name,
                "expression": measure_expression,
                "dependencies": dependencies
            }
            
        except Exception as e:
            logger.error(f"Failed to get measure dependencies: {e}")
            return {"error": str(e)}
    
    def _analyze_dax_dependencies(self, dax_expression: str) -> Dict[str, List[str]]:
        """Analyze DAX expression to find dependencies"""
        import re
        
        dependencies = {
            "tables": [],
            "columns": [],
            "measures": [],
            "functions": []
        }
        
        try:
            # Find table references (Table[Column] pattern)
            table_column_pattern = r"'?([A-Za-z_][A-Za-z0-9_\s]*)'?\[([A-Za-z_][A-Za-z0-9_\s]*)\]"
            matches = re.findall(table_column_pattern, dax_expression)
            
            for table, column in matches:
                table = table.strip()
                column = column.strip()
                
                if table not in dependencies["tables"]:
                    dependencies["tables"].append(table)
                
                column_ref = f"{table}[{column}]"
                if column_ref not in dependencies["columns"]:
                    dependencies["columns"].append(column_ref)
            
            # Find DAX functions
            function_pattern = r"\\b([A-Z]+)\\s*\\("
            function_matches = re.findall(function_pattern, dax_expression.upper())
            
            common_functions = [
                'CALCULATE', 'SUM', 'SUMX', 'COUNT', 'COUNTROWS', 'FILTER',
                'ALL', 'VALUES', 'RELATED', 'DISTINCTCOUNT', 'AVERAGE'
            ]
            
            for func in function_matches:
                if func in common_functions and func not in dependencies["functions"]:
                    dependencies["functions"].append(func)
            
            # Find measure references (measures typically don't have table prefix)
            # This is basic - more sophisticated parsing would be needed for complex cases
            measure_pattern = r"\\[([A-Za-z_][A-Za-z0-9_\\s]*)\\]"
            measure_matches = re.findall(measure_pattern, dax_expression)
            
            for measure in measure_matches:
                measure = measure.strip()
                # Check if this looks like a measure (not a column reference)
                if not any(f"{table}[{measure}]" in dependencies["columns"] for table in dependencies["tables"]):
                    if measure not in dependencies["measures"]:
                        dependencies["measures"].append(measure)
            
        except Exception as e:
            logger.error(f"DAX dependency analysis failed: {e}")
        
        return dependencies
    
    async def get_model_size_info(self) -> Dict[str, Any]:
        """Get model size and memory information"""
        try:
            with Pyadomd(self.connection_string) as conn:
                cursor = conn.cursor()
                
                size_info = {}
                
                # Try to get memory usage
                try:
                    cursor.execute("SELECT SUM([MEMORY_USAGE_KB]) FROM $SYSTEM.DISCOVER_MEMORYUSAGE")
                    memory_result = cursor.fetchone()
                    if memory_result and memory_result[0]:
                        size_info["total_memory_kb"] = memory_result[0]
                        size_info["total_memory_mb"] = memory_result[0] / 1024
                except:
                    size_info["memory_info"] = "Not available"
                
                # Get table counts
                try:
                    cursor.execute("SELECT COUNT(*) FROM $SYSTEM.DBSCHEMA_TABLES WHERE [TABLE_TYPE] = 'TABLE'")
                    table_count = cursor.fetchone()[0]
                    size_info["table_count"] = table_count
                except:
                    size_info["table_count"] = "Unknown"
                
                cursor.close()
                return size_info
                
        except Exception as e:
            logger.error(f"Failed to get model size info: {e}")
            return {"error": str(e)}

    async def get_dax_focused_metadata(self, dax_query: str) -> Dict[str, Any]:
        """
        Get model metadata focused on the dependencies of a specific DAX query.
        This method analyzes the DAX query to find dependencies and then expands
        the metadata to include related tables through relationships.
        """
        try:
            # Get the query dependencies
            deps_df = await self._get_dax_query_dependencies(dax_query)
            
            if deps_df.empty:
                # Fallback to full metadata if dependency analysis fails
                return await self.get_full_metadata()
            
            # Get the tables used directly by the query
            tables_used = set(deps_df["Table Name"].unique().tolist())
            
            # Get full metadata for expansion
            full_tables = await self.get_tables()
            full_columns = await self.get_columns()
            full_relationships = await self.get_relationships()
            
            # Expand tables to include related ones through relationships
            expanded_tables = self._expand_tables_via_relationships(
                tables_used, full_relationships
            )
            
            # Filter metadata to expanded tables
            filtered_metadata = {
                "database_name": self.current_database,
                "extraction_time": datetime.now().isoformat(),
                "query_dependencies": deps_df.to_dict('records'),
                "tables": [t for t in full_tables if t["name"] in expanded_tables],
                "columns": [c for c in full_columns if c["table_name"] in expanded_tables],
                "relationships": [r for r in full_relationships 
                               if r["from_table"] in expanded_tables and r["to_table"] in expanded_tables],
                "measures": [],  # We'll fully define measures in the query instead
                "hierarchies": [],  # Not critical for optimization
                "partitions": []   # Not critical for optimization
            }
            
            logger.info(f"Extracted focused metadata for {len(expanded_tables)} tables "
                       f"(originally {len(tables_used)} from query dependencies)")
            
            return filtered_metadata
            
        except Exception as e:
            logger.error(f"Failed to extract focused metadata: {e}")
            # Fallback to full metadata
            return await self.get_full_metadata()
    
    async def _get_dax_query_dependencies(self, dax_query: str) -> Any:
        """
        Get DAX query dependencies using DMV queries.
        Returns a DataFrame-like structure with columns: ['Table Name', 'Column Name']
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
                
                # Convert to list of dictionaries
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                if not data:
                    return {"Table Name": [], "Column Name": []}
                
                # Convert to a simple structure we can work with
                dependencies = []
                for row in data:
                    row_dict = dict(zip(columns, row))
                    # Clean up column names (remove brackets if present)
                    cleaned_dict = {}
                    for k, v in row_dict.items():
                        clean_key = k.strip('[]')
                        cleaned_dict[clean_key] = v
                    
                    # Only include columns and calc columns
                    obj_type = cleaned_dict.get('Referenced Object Type', '').replace('_', ' ').title()
                    if obj_type in ['Column', 'Calc Column']:
                        table_name = cleaned_dict.get('Referenced Table', '')
                        column_name = cleaned_dict.get('Referenced Object', '')
                        if table_name and column_name and not column_name.startswith('RowNumber-'):
                            dependencies.append({
                                'Table Name': table_name,
                                'Column Name': column_name
                            })
                cursor.close()
                
                # Create a simple DataFrame-like structure
                class SimpleDataFrame:
                    def __init__(self, data):
                        self.data = data
                        
                    def unique(self):
                        seen = set()
                        result = []
                        for item in self.data:
                            if item not in seen:
                                seen.add(item)
                                result.append(item)
                        return result
                    
                    def tolist(self):
                        return self.data
                    
                    @property
                    def empty(self):
                        return len(self.data) == 0
                    
                    def to_dict(self, orient='records'):
                        if orient == 'records':
                            return [dict(zip(['Table Name', 'Column Name'], [d['Table Name'], d['Column Name']])) 
                                   for d in self.data]
                        return self.data
                
                # Create result structure
                result = type('MockDataFrame', (), {})()
                result.data = dependencies
                result.empty = len(dependencies) == 0
                
                # Add column access
                table_names = [d['Table Name'] for d in dependencies]
                
                class TableNameColumn:
                    def __init__(self, data):
                        self.data = data
                    
                    def unique(self):
                        return SimpleDataFrame(list(set(self.data)))
                
                result.__dict__['Table Name'] = TableNameColumn(table_names)
                result.to_dict = lambda orient='records': dependencies if orient == 'records' else dependencies
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to get DAX query dependencies: {e}")
            # Return empty structure
            empty_result = type('EmptyDataFrame', (), {})()
            empty_result.empty = True
            empty_result.to_dict = lambda orient='records': []
            return empty_result
    
    def _expand_tables_via_relationships(self, initial_tables: set, relationships: List[Dict[str, Any]]) -> set:
        """
        Expand the set of tables to include related tables based on relationships.
        This follows filtering propagation through the relationship graph.
        """
        expanded_tables = set(initial_tables)
        changed = True
        
        # Keep expanding until no new tables are added
        while changed:
            changed = False
            for rel in relationships:
                from_table = rel.get("from_table", "")
                to_table = rel.get("to_table", "")
                
                # If we have one table in the relationship, add the other
                if from_table in expanded_tables and to_table not in expanded_tables:
                    expanded_tables.add(to_table)
                    changed = True
                elif to_table in expanded_tables and from_table not in expanded_tables:
                    expanded_tables.add(from_table)
                    changed = True
        
        return expanded_tables
