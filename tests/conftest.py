"""
Test configuration and utilities for DAX Optimizer MCP Server
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import pandas as pd

# Test fixtures and utilities


@pytest.fixture
def mock_connection_string():
    """Mock connection string for testing"""
    return (
        "Provider=MSOLAP;"
        "Data Source=powerbi://api.powerbi.com/v1.0/myorg/TestWorkspace;"
        "Initial Catalog=TestDataset;"
        "User ID=;"
        "Password=mock_token;"
        "Persist Security Info=True;"
    )


@pytest.fixture
def mock_pyadomd():
    """Mock pyadomd connection"""
    with patch('src.mcp.dax_analyzer.Pyadomd') as mock:
        # Configure mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ("Test Result", 100, "2024-01-01")
        ]
        mock_cursor.fetchone.return_value = ("Test Single Result",)
        mock_cursor.description = [("Result", None), ("Value", None), ("Date", None)]
        mock_cursor.rowcount = 1
        
        # Configure mock connection
        mock_conn = Mock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        
        mock.return_value = mock_conn
        yield mock


@pytest.fixture
def sample_dax_query():
    """Sample DAX query for testing"""
    return """
    EVALUATE
    SUMMARIZE(
        Sales,
        Sales[Category],
        "Total Sales", SUM(Sales[Amount])
    )
    ORDER BY Sales[Category]
    """


@pytest.fixture
def sample_measure_expression():
    """Sample DAX measure expression"""
    return """
    Total Sales = 
    CALCULATE(
        SUM(Sales[Amount]),
        Sales[Status] = "Completed"
    )
    """


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    with patch('openai.OpenAI') as mock:
        mock_response = Mock()
        mock_response.choices[0].message.content = """
        Optimized Total Sales = 
        SUMX(
            FILTER(Sales, Sales[Status] = "Completed"),
            Sales[Amount]
        )
        """
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock.return_value = mock_client
        
        yield mock_client


class TestDataHelper:
    """Helper class for generating test data"""
    
    @staticmethod
    def create_mock_server_info() -> Dict[str, Any]:
        """Create mock server information"""
        return {
            "version": "Analysis Services 15.0.4000.0",
            "server_name": "TestServer",
            "connection_time": "2024-01-01T12:00:00Z"
        }
    
    @staticmethod
    def create_mock_databases() -> List[Dict[str, Any]]:
        """Create mock database list"""
        return [
            {
                "name": "TestDataset1",
                "description": "Test Dataset 1",
                "last_update": "2024-01-01T10:00:00Z"
            },
            {
                "name": "TestDataset2", 
                "description": "Test Dataset 2",
                "last_update": "2024-01-01T11:00:00Z"
            }
        ]
    
    @staticmethod
    def create_mock_query_results() -> List[Dict[str, Any]]:
        """Create mock query results"""
        return [
            {"Category": "Electronics", "Total Sales": 15000},
            {"Category": "Clothing", "Total Sales": 12000},
            {"Category": "Books", "Total Sales": 8000}
        ]
    
    @staticmethod
    def create_mock_performance_metrics() -> Dict[str, Any]:
        """Create mock performance metrics"""
        return {
            "duration_ms": 250.5,
            "row_count": 3,
            "column_count": 2,
            "execution_time": "2024-01-01T12:00:00Z"
        }
    
    @staticmethod
    def create_mock_optimization_result() -> Dict[str, Any]:
        """Create mock optimization result"""
        return {
            "measure_name": "Total Sales",
            "original_dax": "SUM(Sales[Amount])",
            "iterations": [
                {
                    "iteration": 1,
                    "dax": "SUMX(Sales, Sales[Amount])",
                    "duration_ms": 180.0,
                    "improvement_percent": 28.0,
                    "results_match": True
                }
            ],
            "best_variant": {
                "iteration": 1,
                "dax": "SUMX(Sales, Sales[Amount])",
                "duration_ms": 180.0,
                "improvement_percent": 28.0,
                "results_match": True
            },
            "baseline_performance": {
                "duration_ms": 250.0,
                "row_count": 1,
                "column_count": 1
            }
        }


# Async test utilities
def async_test(coro):
    """Decorator to run async tests"""
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    return wrapper


# Test base classes
class BaseDAXOptimizerTest:
    """Base test class with common setup"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.test_helper = TestDataHelper()
    
    def teardown_method(self):
        """Cleanup after each test method"""
        pass


# Custom assertions
def assert_dax_equivalent(dax1: str, dax2: str):
    """Assert that two DAX expressions are semantically equivalent"""
    # This would need a proper DAX parser for full implementation
    # For now, just normalize whitespace and compare
    normalized1 = ' '.join(dax1.split()).upper()
    normalized2 = ' '.join(dax2.split()).upper()
    assert normalized1 == normalized2, f"DAX expressions are not equivalent:\\n{dax1}\\n!=\\n{dax2}"


def assert_performance_improved(baseline: float, optimized: float, min_improvement: float = 0.05):
    """Assert that performance has improved by at least min_improvement (5% by default)"""
    improvement = (baseline - optimized) / baseline
    assert improvement >= min_improvement, f"Performance not improved enough: {improvement*100:.1f}% < {min_improvement*100:.1f}%"


def assert_results_identical(results1: List[Dict], results2: List[Dict]):
    """Assert that two result sets are identical"""
    df1 = pd.DataFrame(results1).sort_values(list(results1[0].keys()) if results1 else []).reset_index(drop=True)
    df2 = pd.DataFrame(results2).sort_values(list(results2[0].keys()) if results2 else []).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(df1, df2, check_exact=False)
