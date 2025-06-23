"""
Tests for Query Validator
"""

import pytest
from src.mcp.query_validator import DAXQueryValidator, ValidationSeverity


class TestDAXQueryValidator:
    """Test cases for DAX Query Validator"""
    
    @pytest.fixture
    def validator(self):
        """Create query validator instance"""
        return DAXQueryValidator()
    
    def test_validate_safe_query(self, validator):
        """Test validation of safe query"""
        query = "EVALUATE ROW(\"Test\", 123)"
        result = validator.validate_query(query)
        
        assert result.is_valid is True
        assert len(result.issues) == 0
    
    def test_validate_dangerous_query(self, validator):
        """Test validation of dangerous query"""
        query = "DROP TABLE Sales"
        result = validator.validate_query(query)
        
        assert result.is_valid is False
        assert len(result.issues) > 0
        assert any(issue['severity'] == 'critical' for issue in result.issues)
    
    def test_validate_unbalanced_parentheses(self, validator):
        """Test validation of unbalanced parentheses"""
        query = "EVALUATE ROW(\"Test\", SUM(Sales[Amount]"
        result = validator.validate_query(query)
        
        assert result.is_valid is False
        assert any("parentheses" in issue['message'].lower() for issue in result.issues)
    
    def test_validate_unbalanced_brackets(self, validator):
        """Test validation of unbalanced brackets"""
        query = "EVALUATE ROW(\"Test\", [Sales[Amount])"
        result = validator.validate_query(query)
        
        assert result.is_valid is False
        assert any("bracket" in issue['message'].lower() for issue in result.issues)
    
    def test_validate_long_query(self, validator):
        """Test validation of very long query"""
        query = "EVALUATE ROW(\"Test\", 123)" + " " * 15000
        result = validator.validate_query(query)
        
        # Should have a warning about length
        assert any(issue['severity'] == 'warning' for issue in result.issues)
    
    def test_get_validation_summary(self, validator):
        """Test validation summary generation"""
        query = "DROP TABLE Sales; EVALUATE ROW(\"Test\", SUM(Sales[Amount])"
        result = validator.validate_query(query)
        
        summary = validator.get_validation_summary(result)
        
        assert isinstance(summary, str)
        assert "🚨" in summary or "❌" in summary  # Should have critical or error icons
