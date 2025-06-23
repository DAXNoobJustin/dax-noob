"""
DAX Query Validator - Validates DAX queries for security and syntax
"""

import re
import logging
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationResult:
    """Result of DAX query validation"""
    
    def __init__(self):
        self.is_valid = True
        self.issues: List[Dict[str, Any]] = []
        self.sanitized_query: Optional[str] = None
    
    def add_issue(self, severity: ValidationSeverity, message: str, suggestion: str = None):
        """Add validation issue"""
        self.issues.append({
            "severity": severity.value,
            "message": message,
            "suggestion": suggestion
        })
        
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False


class DAXQueryValidator:
    """Validates DAX queries for security and best practices"""
    
    def __init__(self):        # Potentially dangerous patterns
        self.dangerous_patterns = [
            (r'(?i)\bEXECUTE\b', "EXECUTE statements are not allowed"),
            (r'(?i)\bSP_\w+', "System stored procedures are not allowed"),
            (r'(?i)\bXP_\w+', "Extended stored procedures are not allowed"),
            (r'(?i)\bOPENROWSET\b', "OPENROWSET is not allowed"),
            (r'(?i)\bOPENDATASOURCE\b', "OPENDATASOURCE is not allowed"),
        ]
          # Performance warning patterns
        self.performance_patterns = [
            (r'(?i)\bSUMX\s*\(\s*\w+\s*,', "SUMX over entire table - consider filtering first"),
            (r'(?i)\bFILTER\s*\(\s*ALL\s*\(', "FILTER(ALL()) can be expensive"),
            (r'(?i)\bCALCULATE\s*\([^)]*\bCALCULATE\b', "Nested CALCULATE statements"),
            (r'(?i)\bVALUES\s*\([^)]*\bVALUES\b', "Nested VALUES functions"),
        ]
        
        # Syntax patterns
        self.syntax_patterns = [
            (r'[{}]', "Curly braces are not valid in DAX"),
            (r'--.*(?:DROP|DELETE|UPDATE|INSERT)', "SQL comments with DDL/DML keywords"),
        ]
    
    def validate_query(self, dax_query: str) -> ValidationResult:
        """Validate a DAX query"""
        result = ValidationResult()
        
        if not dax_query or not dax_query.strip():
            result.add_issue(ValidationSeverity.ERROR, "Query cannot be empty")
            return result
        
        # Clean and prepare query
        cleaned_query = self._clean_query(dax_query)
        result.sanitized_query = cleaned_query
        
        # Security validation
        self._validate_security(cleaned_query, result)
        
        # Performance validation
        self._validate_performance(cleaned_query, result)
        
        # Syntax validation
        self._validate_syntax(cleaned_query, result)
          # Length validation
        self._validate_length(cleaned_query, result)
        
        return result
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
          # Remove comments (but preserve them for analysis)
        # This is basic - a full DAX parser would be better
        cleaned = re.sub(r'//.*?(?=\n|$)', '', cleaned)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        
        return cleaned
    
    def _validate_security(self, query: str, result: ValidationResult):
        """Validate query for security issues"""
        for pattern, message in self.dangerous_patterns:
            if re.search(pattern, query):
                result.add_issue(
                    ValidationSeverity.CRITICAL,
                    message,
                    "Remove dangerous SQL operations from DAX query"
                )
    
    def _validate_performance(self, query: str, result: ValidationResult):
        """Validate query for performance issues"""
        for pattern, message in self.performance_patterns:
            if re.search(pattern, query):
                result.add_issue(
                    ValidationSeverity.WARNING,
                    message,
                    "Consider optimizing for better performance"
                )
        
        # Check for excessive nesting
        nesting_level = self._calculate_nesting_level(query)
        if nesting_level > 5:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Deep nesting detected (level {nesting_level})",
                "Consider breaking into variables or simplifying"
            )
    
    def _validate_syntax(self, query: str, result: ValidationResult):
        """Validate query syntax"""
        for pattern, message in self.syntax_patterns:
            if re.search(pattern, query):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    message,
                    "Fix syntax error"
                )
        
        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            result.add_issue(
                ValidationSeverity.ERROR,
                "Unbalanced parentheses",
                "Ensure all parentheses are properly closed"
            )
        
        # Check for balanced brackets
        if query.count('[') != query.count(']'):
            result.add_issue(
                ValidationSeverity.ERROR,
                "Unbalanced brackets",
                "Ensure all brackets are properly closed"
            )
    
    def _validate_length(self, query: str, result: ValidationResult):
        """Validate query length"""
        if len(query) > 10000:
            result.add_issue(
                ValidationSeverity.WARNING,
                "Query is very long",
                "Consider breaking into smaller parts or using variables"
            )
        
        lines = query.split('\n')
        if len(lines) > 100:
            result.add_issue(
                ValidationSeverity.INFO,
                "Query has many lines",
                "Consider formatting for better readability"
            )

    def get_validation_summary(self, result: ValidationResult) -> str:
        """Get a formatted summary of validation results"""
        if not result.issues:
            return "✅ Query validation passed - no issues found"
        
        summary = "🔍 Query Validation Results:\n\n"
        
        # Group by severity
        by_severity = {}
        for issue in result.issues:
            severity = issue['severity']
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(issue)
        
        # Display by severity
        severity_icons = {
            'critical': '🚨',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        }
        
        for severity in ['critical', 'error', 'warning', 'info']:
            if severity in by_severity:
                summary += f"{severity_icons[severity]} {severity.upper()}:\n"
                for issue in by_severity[severity]:
                    summary += f"  • {issue['message']}\n"
                    if issue.get('suggestion'):
                        summary += f"    💡 {issue['suggestion']}\n"
                summary += "\n"
        
        return summary


# Global validator instance
query_validator = DAXQueryValidator()
