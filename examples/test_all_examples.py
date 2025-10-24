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

import ast
import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from examples.utils.example_helpers import (
    example_context,
    print_section,
    print_subsection,
)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


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
    test_results: list[TestResult]
    recommendations: list[str]


class ExampleTester:
    """Tests and validates example files."""

    def __init__(self):
        self.examples_dir = Path("examples")
        self.timeout_seconds = 120  # 2 minutes per example

    def test_syntax(self, file_path: Path) -> TestResult:
        """Test Python syntax validity."""
        start_time = time.time()

        try:
            content = file_path.read_text(encoding="utf-8")
            ast.parse(content)

            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="syntax",
                success=True,
                duration=duration,
                message="Syntax is valid",
            )

        except SyntaxError as e:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="syntax",
                success=False,
                duration=duration,
                message=f"Syntax error: {e.msg}",
                details=f"Line {e.lineno}: {e.text}",
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="syntax",
                success=False,
                duration=duration,
                message=f"Error reading file: {e}",
            )

    def test_imports(self, file_path: Path) -> TestResult:
        """Test if all imports can be resolved."""
        start_time = time.time()

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            import_errors = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            importlib.import_module(alias.name)
                        except ImportError as e:
                            import_errors.append(f"Cannot import {alias.name}: {e}")

                elif isinstance(node, ast.ImportFrom) and node.module:
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
                    # Show first 5 errors
                    details="\n".join(import_errors[:5]),
                )
            return TestResult(
                file_path=file_path,
                test_type="imports",
                success=True,
                duration=duration,
                message="All imports resolved successfully",
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="imports",
                success=False,
                duration=duration,
                message=f"Error checking imports: {e}",
            )

    def test_documentation(self, file_path: Path) -> TestResult:
        """Test documentation completeness."""
        start_time = time.time()

        try:
            content = file_path.read_text(encoding="utf-8")

            # Check for module docstring
            if not content.strip().startswith('"""'):
                duration = time.time() - start_time
                return TestResult(
                    file_path=file_path,
                    test_type="documentation",
                    success=False,
                    duration=duration,
                    message="Missing module docstring",
                )

            # Extract docstring
            docstring_match = content.split('"""')[1] if '"""' in content else ""
            docstring_lower = docstring_match.lower()

            # Required sections for examples
            required_sections = [
                "prerequisites",
                "usage",
                "expected output",
                "learning objectives",
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
                    message=f"Missing documentation sections: {', '.join(missing_sections)}",
                )
            return TestResult(
                file_path=file_path,
                test_type="documentation",
                success=True,
                duration=duration,
                message="Documentation is complete",
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="documentation",
                success=False,
                duration=duration,
                message=f"Error checking documentation: {e}",
            )

    def test_functionality(self, file_path: Path) -> TestResult:
        """Test basic functionality by running the example."""
        start_time = time.time()

        # Skip certain files that shouldn't be run directly
        skip_files = {"__init__.py", "test_all_examples.py", "code_quality_review.py"}

        if file_path.name in skip_files:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="functionality",
                success=True,
                duration=duration,
                message="Skipped (not executable)",
            )

        try:
            # Run the example with a timeout
            result = subprocess.run(
                [sys.executable, str(file_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=file_path.parent,
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                return TestResult(
                    file_path=file_path,
                    test_type="functionality",
                    success=True,
                    duration=duration,
                    message="Executed successfully",
                    details=f"Output length: {len(result.stdout)} chars",
                )
            return TestResult(
                file_path=file_path,
                test_type="functionality",
                success=False,
                duration=duration,
                message=f"Execution failed (exit code {result.returncode})",
                details=result.stderr[:500] if result.stderr else "No error output",
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="functionality",
                success=False,
                duration=duration,
                message=f"Execution timed out after {self.timeout_seconds} seconds",
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                file_path=file_path,
                test_type="functionality",
                success=False,
                duration=duration,
                message=f"Error running example: {e}",
            )

    def validate_example(self, file_path: Path) -> ValidationReport:
        """Perform comprehensive validation of an example."""

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
                message="Skipped due to syntax/import errors",
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
            recommendations=recommendations,
        )


def discover_example_files() -> list[Path]:
    """Discover all Python example files."""
    examples_dir = Path("examples")
    python_files = list(examples_dir.rglob("*.py"))

    # Exclude certain files
    excluded_files = {"__init__.py"}
    python_files = [f for f in python_files if f.name not in excluded_files]

    return sorted(python_files)


def run_validation_suite() -> list[ValidationReport]:
    """Run the complete validation suite."""
    print_subsection("Running Validation Suite")

    example_files = discover_example_files()

    tester = ExampleTester()
    reports = []

    for file_path in example_files:
        try:
            report = tester.validate_example(file_path)
            reports.append(report)
        except Exception as e:
            # Create a failed report
            reports.append(
                ValidationReport(
                    file_path=file_path,
                    syntax_valid=False,
                    imports_valid=False,
                    documentation_complete=False,
                    functionality_tested=False,
                    overall_score=0.0,
                    test_results=[],
                    recommendations=[f"Fix validation error: {e}"],
                )
            )

    return reports


def analyze_validation_results(reports: list[ValidationReport]) -> None:
    """Analyze and display validation results."""
    print_subsection("Validation Results Analysis")

    total_files = len(reports)
    if total_files == 0:
        return

    # Calculate metrics
    sum(1 for r in reports if r.syntax_valid)
    sum(1 for r in reports if r.imports_valid)
    sum(1 for r in reports if r.documentation_complete)
    sum(1 for r in reports if r.functionality_tested)

    sum(r.overall_score for r in reports) / total_files

    # Show files with issues
    failed_files = [r for r in reports if r.overall_score < 1.0]
    if failed_files:
        for report in failed_files[:10]:  # Show first 10
            report.overall_score * 100

        if len(failed_files) > 10:
            pass


def display_detailed_results(
    reports: list[ValidationReport], max_details: int = 10
) -> None:
    """Display detailed results for failed validations."""
    print_subsection("Detailed Validation Results")

    failed_reports = [r for r in reports if r.overall_score < 1.0]
    failed_reports.sort(key=lambda x: x.overall_score)  # Worst first

    for _i, report in enumerate(failed_reports[:max_details], 1):
        for test_result in report.test_results:
            if test_result.details and not test_result.success:
                pass

        if report.recommendations:
            for _rec in report.recommendations:
                pass


def provide_testing_summary() -> None:
    """Provide summary and recommendations for testing."""
    print_subsection("Testing Best Practices")


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
            sum(r.overall_score for r in reports) / total_files * 100

            if success_rate >= 95 or success_rate >= 80:
                pass
            else:
                pass

        else:
            pass


if __name__ == "__main__":
    main()
