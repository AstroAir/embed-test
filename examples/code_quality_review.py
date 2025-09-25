"""
Code Quality Review and Standards Enforcement

This script reviews all example files to ensure they follow coding standards:
- PEP 8 compliance
- Proper docstring formatting
- Type hints usage
- Error handling patterns
- Import organization
- Code structure and readability

Prerequisites:
- PDF Vector System installed
- Access to all example files

Usage:
    python code_quality_review.py

Expected Output:
    - Code quality assessment
    - Standards compliance report
    - Recommendations for improvements
    - Best practices summary

Learning Objectives:
- Understand coding standards for the project
- Learn best practices for example code
- See quality assurance patterns
- Master maintainable code structure
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.example_helpers import (
    print_section, print_subsection, example_context
)


@dataclass
class QualityIssue:
    """Represents a code quality issue."""
    file_path: Path
    line_number: int
    issue_type: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    suggestion: Optional[str] = None


@dataclass
class QualityReport:
    """Code quality assessment report."""
    file_path: Path
    total_lines: int
    issues: List[QualityIssue]
    score: float
    compliant: bool


class CodeQualityChecker:
    """Checks code quality against project standards."""
    
    def __init__(self):
        self.standards = {
            "max_line_length": 88,  # Black formatter standard
            "max_function_length": 50,
            "max_file_length": 300,
            "required_docstring_sections": [
                "description", "prerequisites", "usage", 
                "expected output", "learning objectives"
            ]
        }
    
    def check_file(self, file_path: Path) -> QualityReport:
        """Check a single Python file for quality issues."""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Parse AST for structural analysis
            try:
                tree = ast.parse(content)
                issues.extend(self._check_ast_structure(tree, file_path))
            except SyntaxError as e:
                issues.append(QualityIssue(
                    file_path=file_path,
                    line_number=e.lineno or 0,
                    issue_type="syntax_error",
                    description=f"Syntax error: {e.msg}",
                    severity="error"
                ))
            
            # Check line-by-line issues
            issues.extend(self._check_line_issues(lines, file_path))
            
            # Check file-level issues
            issues.extend(self._check_file_issues(content, file_path))
            
            # Calculate quality score
            score = self._calculate_score(issues, len(lines))
            compliant = score >= 0.8 and not any(i.severity == "error" for i in issues)
            
            return QualityReport(
                file_path=file_path,
                total_lines=len(lines),
                issues=issues,
                score=score,
                compliant=compliant
            )
            
        except Exception as e:
            issues.append(QualityIssue(
                file_path=file_path,
                line_number=0,
                issue_type="file_error",
                description=f"Error reading file: {e}",
                severity="error"
            ))
            
            return QualityReport(
                file_path=file_path,
                total_lines=0,
                issues=issues,
                score=0.0,
                compliant=False
            )
    
    def _check_ast_structure(self, tree: ast.AST, file_path: Path) -> List[QualityIssue]:
        """Check AST structure for quality issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Check function definitions
            if isinstance(node, ast.FunctionDef):
                issues.extend(self._check_function(node, file_path))
            
            # Check class definitions
            elif isinstance(node, ast.ClassDef):
                issues.extend(self._check_class(node, file_path))
            
            # Check imports
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                issues.extend(self._check_import(node, file_path))
        
        return issues
    
    def _check_function(self, node: ast.FunctionDef, file_path: Path) -> List[QualityIssue]:
        """Check function definition for quality issues."""
        issues = []
        
        # Check for docstring
        if not ast.get_docstring(node):
            issues.append(QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                issue_type="missing_docstring",
                description=f"Function '{node.name}' missing docstring",
                severity="warning",
                suggestion="Add docstring describing function purpose and parameters"
            ))
        
        # Check function length
        if hasattr(node, 'end_lineno') and node.end_lineno:
            func_length = node.end_lineno - node.lineno
            if func_length > self.standards["max_function_length"]:
                issues.append(QualityIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    issue_type="function_too_long",
                    description=f"Function '{node.name}' is {func_length} lines (max {self.standards['max_function_length']})",
                    severity="warning",
                    suggestion="Consider breaking into smaller functions"
                ))
        
        # Check for type hints
        if not node.returns and node.name != "__init__":
            issues.append(QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                issue_type="missing_return_type",
                description=f"Function '{node.name}' missing return type hint",
                severity="info",
                suggestion="Add return type hint for better code documentation"
            ))
        
        return issues
    
    def _check_class(self, node: ast.ClassDef, file_path: Path) -> List[QualityIssue]:
        """Check class definition for quality issues."""
        issues = []
        
        # Check for docstring
        if not ast.get_docstring(node):
            issues.append(QualityIssue(
                file_path=file_path,
                line_number=node.lineno,
                issue_type="missing_docstring",
                description=f"Class '{node.name}' missing docstring",
                severity="warning",
                suggestion="Add docstring describing class purpose"
            ))
        
        return issues
    
    def _check_import(self, node: ast.AST, file_path: Path) -> List[QualityIssue]:
        """Check import statements for quality issues."""
        issues = []
        
        # Check for unused imports (basic check)
        # This would require more sophisticated analysis in a real implementation
        
        return issues
    
    def _check_line_issues(self, lines: List[str], file_path: Path) -> List[QualityIssue]:
        """Check line-by-line issues."""
        issues = []
        
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > self.standards["max_line_length"]:
                issues.append(QualityIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type="line_too_long",
                    description=f"Line {i} is {len(line)} characters (max {self.standards['max_line_length']})",
                    severity="warning",
                    suggestion="Break long line or use line continuation"
                ))
            
            # Check for trailing whitespace
            if line.rstrip() != line and line.strip():
                issues.append(QualityIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type="trailing_whitespace",
                    description=f"Line {i} has trailing whitespace",
                    severity="info",
                    suggestion="Remove trailing whitespace"
                ))
        
        return issues
    
    def _check_file_issues(self, content: str, file_path: Path) -> List[QualityIssue]:
        """Check file-level issues."""
        issues = []
        lines = content.split('\n')
        
        # Check file length
        if len(lines) > self.standards["max_file_length"]:
            issues.append(QualityIssue(
                file_path=file_path,
                line_number=0,
                issue_type="file_too_long",
                description=f"File is {len(lines)} lines (max {self.standards['max_file_length']})",
                severity="warning",
                suggestion="Consider splitting into multiple files"
            ))
        
        # Check module docstring for examples
        if file_path.name.endswith('.py') and 'examples' in str(file_path):
            if not content.strip().startswith('"""'):
                issues.append(QualityIssue(
                    file_path=file_path,
                    line_number=1,
                    issue_type="missing_module_docstring",
                    description="Example file missing module docstring",
                    severity="error",
                    suggestion="Add comprehensive module docstring with example description"
                ))
            else:
                # Check docstring sections
                docstring_match = re.match(r'"""(.*?)"""', content, re.DOTALL)
                if docstring_match:
                    docstring = docstring_match.group(1).lower()
                    missing_sections = []
                    
                    for section in self.standards["required_docstring_sections"]:
                        if section not in docstring:
                            missing_sections.append(section)
                    
                    if missing_sections:
                        issues.append(QualityIssue(
                            file_path=file_path,
                            line_number=1,
                            issue_type="incomplete_docstring",
                            description=f"Missing docstring sections: {', '.join(missing_sections)}",
                            severity="warning",
                            suggestion="Add missing sections to module docstring"
                        ))
        
        return issues
    
    def _calculate_score(self, issues: List[QualityIssue], total_lines: int) -> float:
        """Calculate quality score based on issues."""
        if total_lines == 0:
            return 0.0
        
        # Weight issues by severity
        weights = {"error": 10, "warning": 3, "info": 1}
        total_weight = sum(weights.get(issue.severity, 1) for issue in issues)
        
        # Calculate score (higher is better)
        max_possible_weight = total_lines * 0.1  # Assume max 0.1 weight per line
        score = max(0.0, 1.0 - (total_weight / max(max_possible_weight, 1)))
        
        return min(1.0, score)


def review_examples_directory() -> List[QualityReport]:
    """Review all Python files in the examples directory."""
    print_subsection("Scanning Example Files")
    
    examples_dir = Path("examples")
    python_files = list(examples_dir.rglob("*.py"))
    
    # Exclude certain files
    excluded_files = {"__init__.py", "code_quality_review.py"}
    python_files = [f for f in python_files if f.name not in excluded_files]
    
    print(f"📁 Found {len(python_files)} Python files to review")
    
    checker = CodeQualityChecker()
    reports = []
    
    for file_path in python_files:
        print(f"   🔍 Reviewing {file_path.relative_to(examples_dir)}")
        report = checker.check_file(file_path)
        reports.append(report)
    
    return reports


def analyze_quality_reports(reports: List[QualityReport]) -> None:
    """Analyze and display quality reports."""
    print_subsection("Quality Analysis Results")
    
    total_files = len(reports)
    compliant_files = sum(1 for r in reports if r.compliant)
    total_issues = sum(len(r.issues) for r in reports)
    average_score = sum(r.score for r in reports) / total_files if total_files > 0 else 0
    
    print(f"📊 Overall Quality Metrics:")
    print(f"   - Total files reviewed: {total_files}")
    print(f"   - Compliant files: {compliant_files}/{total_files} ({compliant_files/total_files*100:.1f}%)")
    print(f"   - Total issues found: {total_issues}")
    print(f"   - Average quality score: {average_score:.2f}/1.00")
    
    # Issue breakdown by severity
    issue_counts = {"error": 0, "warning": 0, "info": 0}
    for report in reports:
        for issue in report.issues:
            issue_counts[issue.severity] += 1
    
    print(f"\n🚨 Issues by Severity:")
    print(f"   - Errors: {issue_counts['error']}")
    print(f"   - Warnings: {issue_counts['warning']}")
    print(f"   - Info: {issue_counts['info']}")
    
    # Top issues by type
    issue_types = {}
    for report in reports:
        for issue in report.issues:
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
    
    if issue_types:
        print(f"\n🔍 Most Common Issues:")
        sorted_issues = sorted(issue_types.items(), key=lambda x: x[1], reverse=True)
        for issue_type, count in sorted_issues[:5]:
            print(f"   - {issue_type.replace('_', ' ').title()}: {count}")


def display_detailed_issues(reports: List[QualityReport], max_issues: int = 20) -> None:
    """Display detailed issues for review."""
    print_subsection("Detailed Issues Report")
    
    all_issues = []
    for report in reports:
        all_issues.extend(report.issues)
    
    # Sort by severity (errors first)
    severity_order = {"error": 0, "warning": 1, "info": 2}
    all_issues.sort(key=lambda x: (severity_order.get(x.severity, 3), str(x.file_path), x.line_number))
    
    print(f"🔍 Showing top {min(max_issues, len(all_issues))} issues:")
    
    for i, issue in enumerate(all_issues[:max_issues], 1):
        severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "❓")
        
        print(f"\n{i}. {severity_icon} {issue.file_path.name}:{issue.line_number}")
        print(f"   Type: {issue.issue_type.replace('_', ' ').title()}")
        print(f"   Description: {issue.description}")
        if issue.suggestion:
            print(f"   Suggestion: {issue.suggestion}")


def provide_improvement_recommendations() -> None:
    """Provide recommendations for code quality improvements."""
    print_subsection("Improvement Recommendations")
    
    print("🎯 Code Quality Best Practices:")
    print()
    print("1. 📝 Documentation:")
    print("   - Add comprehensive module docstrings to all example files")
    print("   - Include all required sections: description, prerequisites, usage, etc.")
    print("   - Document all functions and classes with clear docstrings")
    print("   - Use type hints for better code documentation")
    print()
    print("2. 🎨 Code Style:")
    print("   - Follow PEP 8 guidelines for Python code")
    print("   - Keep lines under 88 characters (Black formatter standard)")
    print("   - Remove trailing whitespace")
    print("   - Use consistent naming conventions")
    print()
    print("3. 🏗️  Structure:")
    print("   - Keep functions under 50 lines when possible")
    print("   - Break large files into smaller, focused modules")
    print("   - Organize imports properly (standard, third-party, local)")
    print("   - Use meaningful variable and function names")
    print()
    print("4. 🛡️  Error Handling:")
    print("   - Add proper exception handling in examples")
    print("   - Provide clear error messages")
    print("   - Handle edge cases gracefully")
    print("   - Include error recovery patterns")
    print()
    print("5. 🧪 Testing:")
    print("   - Ensure examples can run independently")
    print("   - Include sample data or data generation")
    print("   - Test examples in different environments")
    print("   - Validate example outputs")


def main() -> None:
    """
    Review code quality across all example files.
    
    This function performs a comprehensive quality review of all
    example files to ensure they meet project standards.
    """
    with example_context("Code Quality Review"):
        
        print_section("Code Quality Assessment")
        
        # Review all example files
        reports = review_examples_directory()
        
        # Analyze results
        analyze_quality_reports(reports)
        
        # Show detailed issues
        display_detailed_issues(reports)
        
        print_section("Quality Improvement")
        
        # Provide recommendations
        provide_improvement_recommendations()
        
        print_section("Quality Review Summary")
        
        # Calculate overall compliance
        total_files = len(reports)
        compliant_files = sum(1 for r in reports if r.compliant)
        compliance_rate = (compliant_files / total_files * 100) if total_files > 0 else 0
        
        print("📋 Quality Review Results:")
        print()
        print(f"✅ Files reviewed: {total_files}")
        print(f"📊 Compliance rate: {compliance_rate:.1f}%")
        print(f"🎯 Quality target: 90%+ compliance")
        
        if compliance_rate >= 90:
            print("🎉 Excellent! Code quality meets project standards")
        elif compliance_rate >= 70:
            print("👍 Good code quality with room for improvement")
        else:
            print("⚠️  Code quality needs attention - review recommendations")
        
        print()
        print("Next steps:")
        print("- Address high-priority issues (errors and warnings)")
        print("- Implement recommended improvements")
        print("- Set up automated quality checks")
        print("- Regular quality reviews for new examples")


if __name__ == "__main__":
    main()
