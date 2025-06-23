"""
Tests for DAX Analyzer
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.mcp.dax_analyzer import DAXAnalyzer


class TestDAXAnalyzer:
    """Test cases for DAX Analyzer"""
    
    @pytest.fixture
    def analyzer(self, mock_connection_string):
        """Create DAX analyzer instance"""
        return DAXAnalyzer(mock_connection_string)
    
    @pytest.mark.asyncio
    async def test_connection(self, analyzer, mock_pyadomd):
        """Test connection functionality"""
        result = await analyzer.test_connection()
        assert "version" in result
        assert "server_name" in result
    
    @pytest.mark.asyncio
    async def test_execute_dax_query(self, analyzer, mock_pyadomd):
        """Test DAX query execution"""
        query = "EVALUATE ROW(\"Test\", 123)"
        results, performance = await analyzer.execute_dax_query(query)
        
        assert isinstance(results, list)
        assert isinstance(performance, dict)
        assert "duration_ms" in performance
        assert "row_count" in performance
    
    def test_define_query_measures(self, analyzer):
        """Test measure definition functionality"""
        query = "EVALUATE ROW(\"Result\", [Total Sales])"
        
        with patch.object(analyzer, 'connection_string'):
            with patch('src.mcp.dax_analyzer.Pyadomd') as mock_pyadomd:
                # Mock the measures query
                mock_cursor = Mock()
                mock_cursor.fetchall.return_value = [
                    ("Total Sales", "SUM(Sales[Amount])", "Sales")
                ]
                mock_cursor.close.return_value = None
                
                mock_conn = Mock()
                mock_conn.__enter__.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                mock_pyadomd.return_value = mock_conn
                
                result = analyzer.define_query_measures(query)
                assert "DEFINE" in result
                assert "MEASURE" in result
                assert "Total Sales" in result
    
    def test_performance_rating(self, analyzer):
        """Test performance rating calculation"""
        assert analyzer._get_performance_rating(50) == "Excellent"
        assert analyzer._get_performance_rating(300) == "Good"
        assert analyzer._get_performance_rating(1000) == "Moderate"
        assert analyzer._get_performance_rating(5000) == "Slow"
        assert analyzer._get_performance_rating(15000) == "Very Slow"
