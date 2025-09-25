"""
Example Testing and Validation Suite

This script tests and validates all examples for correctness and completeness:
- Syntax validation
- Import verification
- Documentation completeness
- Functionality testing
- Error handling validation

Prerequisites:
- PDF Vector System installed
- All example dependencies available
- Sample data for testing

Usage:
    python test_all_examples.py

Expected Output:
    - Test results for each example
    - Validation status reports
    - Error summaries and recommendations
    - Overall test suite results

Learning Objectives:
- Understand example testing methodology
- Learn validation best practices
- See comprehensive testing patterns
- Master quality assurance for examples
"""

import sys
import ast
import importlib.util
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import traceback
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.example_helpers import (
    print_section, print_subsection, example_context
)


@dataclass
class TestResult:
    """Result of testing an example."""
    file_path: Path
    test_type: str
    success: bool
    duration: float
    message: str
    details: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for an example."""
    file_path: Path
    syntax_valid: bool
    imports_valid: bool
    documentation_complete: bool
    functionality_tested: bool
    overall_score: float
    test_results: List[TestResult]
    recommendations: List[str]


class ExampleTester:
    """Tests and validates example files."""
    
    def __init__(self):
        self.examples_dir = Path("examples")
        self.timeout_seconds = 120  # 2 minutes per example
    
    def test_syntax(self, file_path: Path) -> TestResult:
        """Test Python syntax validity."""
        start_time = time.time()
        
        try:
            content = file_path.read_text(encoding='utf-8')
            ast.parse(content)
            
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="syntax",
                success=True,
                duration=duration,
                message="Syntax is valid"
            )
            
        except SyntaxError as e:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="syntax",
                success=False,
                duration=duration,
                message=f"Syntax error: {e.msg}",
                details=f"Line {e.lineno}: {e.text}"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="syntax",
                success=False,
                duration=duration,
                message=f"Error reading file: {e}"
            )
    
    def test_imports(self, file_path: Path) -> TestResult:
        """Test if all imports can be resolved."""
        start_time = time.time()
        
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            import_errors = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            importlib.import_module(alias.name)
                        except ImportError as e:
                            import_errors.append(f"Cannot import {alias.name}: {e}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        try:
                            importlib.import_module(node.module)
                        except ImportError as e:
                            import_errors.append(f"Cannot import {node.module}: {e}")
            
            duration = time.time() - start_time
            
            if import_errors:
                return TestResult(
                    file_path=file_path,
                    test_type="imports",
                    success=False,
                    duration=duration,
                    message=f"Import errors found: {len(import_errors)}",
                    details="\n".join(import_errors[:5])  # Show first 5 errors
                )
            else:
                return TestResult(
                    file_path=file_path,
                    test_type="imports",
                    success=True,
                    duration=duration,
                    message="All imports resolved successfully"
                )
                
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="imports",
                success=False,
                duration=duration,
                message=f"Error checking imports: {e}"
            )
    
    def test_documentation(self, file_path: Path) -> TestResult:
        """Test documentation completeness."""
        start_time = time.time()
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Check for module docstring
            if not content.strip().startswith('"""'):
                duration = time.time() - start_time
                return TestResult(
                    file_path=file_path,
                    test_type="documentation",
                    success=False,
                    duration=duration,
                    message="Missing module docstring"
                )
            
            # Extract docstring
            docstring_match = content.split('"""')[1] if '"""' in content else ""
            docstring_lower = docstring_match.lower()
            
            # Required sections for examples
            required_sections = [
                "prerequisites", "usage", "expected output", "learning objectives"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in docstring_lower:
                    missing_sections.append(section)
            
            duration = time.time() - start_time
            
            if missing_sections:
                return TestResult(
                    file_path=file_path,
                    test_type="documentation",
                    success=False,
                    duration=duration,
                    message=f"Missing documentation sections: {', '.join(missing_sections)}"
                )
            else:
                return TestResult(
                    file_path=file_path,
                    test_type="documentation",
                    success=True,
                    duration=duration,
                    message="Documentation is complete"
                )
                
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="documentation",
                success=False,
                duration=duration,
                message=f"Error checking documentation: {e}"
            )
    
    def test_functionality(self, file_path: Path) -> TestResult:
        """Test basic functionality by running the example."""
        start_time = time.time()
        
        # Skip certain files that shouldn't be run directly
        skip_files = {
            "__init__.py", "test_all_examples.py", "code_quality_review.py"
        }
        
        if file_path.name in skip_files:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="functionality",
                success=True,
                duration=duration,
                message="Skipped (not executable)"
            )
        
        try:
            # Run the example with a timeout
            result = subprocess.run(
                [sys.executable, str(file_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=file_path.parent
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return TestResult(
                    file_path=file_path,
                    test_type="functionality",
                    success=True,
                    duration=duration,
                    message="Executed successfully",
                    details=f"Output length: {len(result.stdout)} chars"
                )
            else:
                return TestResult(
                    file_path=file_path,
                    test_type="functionality",
                    success=False,
                    duration=duration,
                    message=f"Execution failed (exit code {result.returncode})",
                    details=result.stderr[:500] if result.stderr else "No error output"
                )
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="functionality",
                success=False,
                duration=duration,
                message=f"Execution timed out after {self.timeout_seconds} seconds"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="functionality",
                success=False,
                duration=duration,
                message=f"Error running example: {e}"
            )
    
    def validate_example(self, file_path: Path) -> ValidationReport:
        """Perform comprehensive validation of an example."""
        print(f"   🔍 Validating {file_path.relative_to(self.examples_dir)}")
        
        test_results = []
        recommendations = []
        
        # Run all tests
        syntax_result = self.test_syntax(file_path)
        test_results.append(syntax_result)
        
        imports_result = self.test_imports(file_path)
        test_results.append(imports_result)
        
        docs_result = self.test_documentation(file_path)
        test_results.append(docs_result)
        
        # Only test functionality if syntax and imports are valid
        if syntax_result.success and imports_result.success:
            func_result = self.test_functionality(file_path)
            test_results.append(func_result)
        else:
            func_result = TestResult(
                file_path=file_path,
                test_type="functionality",
                success=False,
                duration=0,
                message="Skipped due to syntax/import errors"
            )
            test_results.append(func_result)
        
        # Generate recommendations
        if not syntax_result.success:
            recommendations.append("Fix syntax errors before proceeding")
        if not imports_result.success:
            recommendations.append("Resolve import issues or add missing dependencies")
        if not docs_result.success:
            recommendations.append("Complete documentation with all required sections")
        if not func_result.success and syntax_result.success and imports_result.success:
            recommendations.append("Debug runtime errors and improve error handling")
        
        # Calculate overall score
        scores = [r.success for r in test_results]
        overall_score = sum(scores) / len(scores) if scores else 0
        
        return ValidationReport(
            file_path=file_path,
            syntax_valid=syntax_result.success,
            imports_valid=imports_result.success,
            documentation_complete=docs_result.success,
            functionality_tested=func_result.success,
            overall_score=overall_score,
            test_results=test_results,
            recommendations=recommendations
        )


def discover_example_files() -> List[Path]:
    """Discover all Python example files."""
    examples_dir = Path("examples")
    python_files = list(examples_dir.rglob("*.py"))
    
    # Exclude certain files
    excluded_files = {"__init__.py"}
    python_files = [f for f in python_files if f.name not in excluded_files]
    
    return sorted(python_files)


def run_validation_suite() -> List[ValidationReport]:
    """Run the complete validation suite."""
    print_subsection("Running Validation Suite")
    
    example_files = discover_example_files()
    print(f"📁 Found {len(example_files)} example files to validate")
    
    tester = ExampleTester()
    reports = []
    
    for file_path in example_files:
        try:
            report = tester.validate_example(file_path)
            reports.append(report)
        except Exception as e:
            print(f"   ❌ Error validating {file_path.name}: {e}")
            # Create a failed report
            reports.append(ValidationReport(
                file_path=file_path,
                syntax_valid=False,
                imports_valid=False,
                documentation_complete=False,
                functionality_tested=False,
                overall_score=0.0,
                test_results=[],
                recommendations=[f"Fix validation error: {e}"]
            ))
    
    return reports


def analyze_validation_results(reports: List[ValidationReport]) -> None:
    """Analyze and display validation results."""
    print_subsection("Validation Results Analysis")
    
    total_files = len(reports)
    if total_files == 0:
        print("❌ No files to analyze")
        return
    
    # Calculate metrics
    syntax_valid = sum(1 for r in reports if r.syntax_valid)
    imports_valid = sum(1 for r in reports if r.imports_valid)
    docs_complete = sum(1 for r in reports if r.documentation_complete)
    functionality_working = sum(1 for r in reports if r.functionality_tested)
    
    average_score = sum(r.overall_score for r in reports) / total_files
    
    print(f"📊 Validation Metrics:")
    print(f"   - Total files: {total_files}")
    print(f"   - Syntax valid: {syntax_valid}/{total_files} ({syntax_valid/total_files*100:.1f}%)")
    print(f"   - Imports valid: {imports_valid}/{total_files} ({imports_valid/total_files*100:.1f}%)")
    print(f"   - Documentation complete: {docs_complete}/{total_files} ({docs_complete/total_files*100:.1f}%)")
    print(f"   - Functionality working: {functionality_working}/{total_files} ({functionality_working/total_files*100:.1f}%)")
    print(f"   - Average score: {average_score:.2f}/1.00")
    
    # Show files with issues
    failed_files = [r for r in reports if r.overall_score < 1.0]
    if failed_files:
        print(f"\n⚠️  Files with issues ({len(failed_files)}):")
        for report in failed_files[:10]:  # Show first 10
            score_percent = report.overall_score * 100
            print(f"   - {report.file_path.name}: {score_percent:.0f}% ({', '.join(report.recommendations[:2])})")
        
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more")


def display_detailed_results(reports: List[ValidationReport], max_details: int = 10) -> None:
    """Display detailed results for failed validations."""
    print_subsection("Detailed Validation Results")
    
    failed_reports = [r for r in reports if r.overall_score < 1.0]
    failed_reports.sort(key=lambda x: x.overall_score)  # Worst first
    
    print(f"🔍 Showing details for {min(max_details, len(failed_reports))} files with issues:")
    
    for i, report in enumerate(failed_reports[:max_details], 1):
        print(f"\n{i}. {report.file_path.name} (Score: {report.overall_score:.2f})")
        
        for test_result in report.test_results:
            status = "✅" if test_result.success else "❌"
            print(f"   {status} {test_result.test_type.title()}: {test_result.message}")
            if test_result.details and not test_result.success:
                print(f"      Details: {test_result.details[:100]}...")
        
        if report.recommendations:
            print(f"   💡 Recommendations:")
            for rec in report.recommendations:
                print(f"      - {rec}")


def provide_testing_summary() -> None:
    """Provide summary and recommendations for testing."""
    print_subsection("Testing Best Practices")
    
    print("🎯 Example Testing Guidelines:")
    print()
    print("1. 📝 Documentation Standards:")
    print("   - Include comprehensive module docstrings")
    print("   - Document prerequisites clearly")
    print("   - Provide usage instructions")
    print("   - Explain expected outputs")
    print("   - Define learning objectives")
    print()
    print("2. 🔧 Code Quality:")
    print("   - Ensure valid Python syntax")
    print("   - Resolve all import dependencies")
    print("   - Handle errors gracefully")
    print("   - Include meaningful error messages")
    print()
    print("3. 🧪 Functionality:")
    print("   - Examples should run without errors")
    print("   - Include sample data or data generation")
    print("   - Test with different configurations")
    print("   - Validate outputs and results")
    print()
    print("4. 🛡️  Error Handling:")
    print("   - Handle missing dependencies gracefully")
    print("   - Provide clear error messages")
    print("   - Include fallback options")
    print("   - Test edge cases")


def main() -> None:
    """
    Test and validate all examples for correctness and completeness.
    
    This function runs a comprehensive test suite to ensure all
    examples meet quality standards and work correctly.
    """
    with example_context("Example Testing and Validation"):
        
        print_section("Example Validation Suite")
        
        # Run validation tests
        reports = run_validation_suite()
        
        # Analyze results
        analyze_validation_results(reports)
        
        # Show detailed results
        display_detailed_results(reports)
        
        print_section("Testing Guidelines")
        
        # Provide testing summary
        provide_testing_summary()
        
        print_section("Validation Summary")
        
        # Calculate overall success rate
        total_files = len(reports)
        if total_files > 0:
            perfect_files = sum(1 for r in reports if r.overall_score == 1.0)
            success_rate = (perfect_files / total_files) * 100
            average_score = sum(r.overall_score for r in reports) / total_files * 100
            
            print("📋 Final Validation Results:")
            print()
            print(f"✅ Files tested: {total_files}")
            print(f"🎯 Perfect score: {perfect_files}/{total_files} ({success_rate:.1f}%)")
            print(f"📊 Average score: {average_score:.1f}%")
            print(f"🏆 Target score: 95%+")
            
            if success_rate >= 95:
                print("🎉 Excellent! All examples meet quality standards")
            elif success_rate >= 80:
                print("👍 Good validation results with minor issues to address")
            else:
                print("⚠️  Validation issues need attention - review failed examples")
            
            print()
            print("Next steps:")
            print("- Fix high-priority validation failures")
            print("- Improve documentation completeness")
            print("- Test examples in different environments")
            print("- Set up automated validation in CI/CD")
        else:
            print("❌ No examples found to validate")


if __name__ == "__main__":
    main()
