"""
Tests for Performance Analyzer
"""

import pytest
from unittest.mock import Mock, patch
from src.mcp.performance_analyzer import PerformanceAnalyzer


class TestPerformanceAnalyzer:
    """Test cases for Performance Analyzer"""
    
    @pytest.fixture
    def analyzer(self, mock_connection_string):
        """Create performance analyzer instance"""
        return PerformanceAnalyzer(mock_connection_string)
    
    @pytest.mark.asyncio
    async def test_analyze_query_performance(self, analyzer, mock_pyadomd):
        """Test query performance analysis"""
        query = "EVALUATE ROW(\"Test\", 123)"
        
        with patch('src.mcp.performance_analyzer.Pyadomd') as mock_pyadomd:
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = [["Test", 123]]
            mock_cursor.close.return_value = None
            
            mock_conn = Mock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_pyadomd.return_value = mock_conn
            
            result = await analyzer.analyze_query_performance(query)
            
            assert "execution_time" in result
            assert "complexity" in result
            assert "suggestions" in result
    
    def test_analyze_query_complexity(self, analyzer):
        """Test query complexity analysis"""
        query = "EVALUATE SUMMARIZE(Sales, Sales[Product], \"Total\", SUM(Sales[Amount]))"
        
        complexity = analyzer._analyze_query_complexity(query)
        
        assert "total_length" in complexity
        assert "function_count" in complexity
        assert "complexity_score" in complexity
        assert "complexity_rating" in complexity
    
    def test_get_complexity_rating(self, analyzer):
        """Test complexity rating calculation"""
        assert analyzer._get_complexity_rating(2) == "Simple"
        assert analyzer._get_complexity_rating(10) == "Moderate"
        assert analyzer._get_complexity_rating(25) == "Complex"
        assert analyzer._get_complexity_rating(35) == "Very Complex"
    
    def test_generate_optimization_suggestions(self, analyzer):
        """Test optimization suggestions generation"""
        query = "EVALUATE FILTER(ALL(Sales), Sales[Amount] > 1000)"
        performance = {"total_duration_ms": 3000}
        complexity = {"iterator_functions": ["SUMX", "AVERAGEX"]}
        
        suggestions = analyzer._generate_optimization_suggestions(query, performance, complexity)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
