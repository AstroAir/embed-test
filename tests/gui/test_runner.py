"""GUI Test Runner and Utilities."""

import subprocess
import sys
from pathlib import Path
from typing import Optional


class GUITestRunner:
    """Utility class for running GUI tests with various configurations."""

    def __init__(self, test_dir: Optional[Path] = None):
        """Initialize test runner.

        Args:
            test_dir: Directory containing GUI tests. Defaults to current directory.
        """
        self.test_dir = test_dir or Path(__file__).parent

    def run_all_tests(self, verbose: bool = True, coverage: bool = True) -> int:
        """Run all GUI tests.

        Args:
            verbose: Enable verbose output
            coverage: Enable coverage reporting

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        args = [
            "pytest",
            str(self.test_dir),
            "-v" if verbose else "",
            "--tb=short",
            "--strict-markers",
            "--strict-config",
        ]

        if coverage:
            args.extend(
                [
                    "--cov=vectorflow.gui",
                    "--cov-report=html:htmlcov/gui",
                    "--cov-report=term-missing",
                    "--cov-fail-under=80",
                ]
            )

        # Filter out empty strings
        args = [arg for arg in args if arg]

        return subprocess.call(args)

    def run_widget_tests(self, widget_name: Optional[str] = None) -> int:
        """Run widget tests.

        Args:
            widget_name: Specific widget to test (e.g., 'processing', 'search')

        Returns:
            Exit code
        """
        if widget_name:
            test_path = self.test_dir / "test_widgets" / f"test_{widget_name}_widget.py"
        else:
            test_path = self.test_dir / "test_widgets"

        return subprocess.call(["pytest", str(test_path), "-v", "-m", "widget"])

    def run_controller_tests(self, controller_name: Optional[str] = None) -> int:
        """Run controller tests.

        Args:
            controller_name: Specific controller to test

        Returns:
            Exit code
        """
        if controller_name:
            test_path = (
                self.test_dir
                / "test_controllers"
                / f"test_{controller_name}_controller.py"
            )
        else:
            test_path = self.test_dir / "test_controllers"

        return subprocess.call(["pytest", str(test_path), "-v", "-m", "controller"])

    def run_integration_tests(self) -> int:
        """Run integration tests.

        Returns:
            Exit code
        """
        return subprocess.call(
            [
                "pytest",
                str(self.test_dir / "test_integration"),
                "-v",
                "-m",
                "integration",
            ]
        )

    def run_dialog_tests(self) -> int:
        """Run dialog tests.

        Returns:
            Exit code
        """
        return subprocess.call(
            ["pytest", str(self.test_dir / "test_dialogs"), "-v", "-m", "dialog"]
        )

    def run_utils_tests(self) -> int:
        """Run utility tests.

        Returns:
            Exit code
        """
        return subprocess.call(
            ["pytest", str(self.test_dir / "test_utils"), "-v", "-m", "utils"]
        )

    def run_smoke_tests(self) -> int:
        """Run smoke tests (basic functionality).

        Returns:
            Exit code
        """
        return subprocess.call(["pytest", str(self.test_dir), "-v", "-m", "smoke"])

    def run_performance_tests(self) -> int:
        """Run performance tests.

        Returns:
            Exit code
        """
        return subprocess.call(
            ["pytest", str(self.test_dir), "-v", "-m", "performance"]
        )

    def generate_test_report(self, output_file: str = "gui_test_report.html") -> int:
        """Generate HTML test report.

        Args:
            output_file: Output HTML file name

        Returns:
            Exit code
        """
        return subprocess.call(
            [
                "pytest",
                str(self.test_dir),
                "--html=" + output_file,
                "--self-contained-html",
                "-v",
            ]
        )

    def list_test_markers(self) -> list[str]:
        """List all available test markers.

        Returns:
            List of marker names
        """
        result = subprocess.run(
            ["pytest", "--markers"], check=False, capture_output=True, text=True
        )

        markers = []
        for line in result.stdout.split("\n"):
            if line.startswith("@pytest.mark."):
                marker = line.split(".")[2].split(":")[0]
                markers.append(marker)

        return sorted(set(markers))

    def run_tests_by_marker(self, marker: str) -> int:
        """Run tests by marker.

        Args:
            marker: Test marker to run (e.g., 'gui', 'widget', 'integration')

        Returns:
            Exit code
        """
        return subprocess.call(["pytest", str(self.test_dir), "-v", "-m", marker])


def main():
    """Main entry point for test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="GUI Test Runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--widgets", action="store_true", help="Run widget tests")
    parser.add_argument(
        "--controllers", action="store_true", help="Run controller tests"
    )
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    parser.add_argument("--dialogs", action="store_true", help="Run dialog tests")
    parser.add_argument("--utils", action="store_true", help="Run utility tests")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests"
    )
    parser.add_argument("--marker", type=str, help="Run tests by marker")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument(
        "--list-markers", action="store_true", help="List available markers"
    )
    parser.add_argument(
        "--coverage", action="store_true", default=True, help="Enable coverage"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )

    args = parser.parse_args()

    runner = GUITestRunner()

    if args.list_markers:
        markers = runner.list_test_markers()
        for _marker in markers:
            pass
        return 0

    if args.all:
        return runner.run_all_tests(verbose=args.verbose, coverage=args.coverage)
    if args.widgets:
        return runner.run_widget_tests()
    if args.controllers:
        return runner.run_controller_tests()
    if args.integration:
        return runner.run_integration_tests()
    if args.dialogs:
        return runner.run_dialog_tests()
    if args.utils:
        return runner.run_utils_tests()
    if args.smoke:
        return runner.run_smoke_tests()
    if args.performance:
        return runner.run_performance_tests()
    if args.marker:
        return runner.run_tests_by_marker(args.marker)
    if args.report:
        return runner.generate_test_report()
    # Default: run all tests
    return runner.run_all_tests(verbose=args.verbose, coverage=args.coverage)


if __name__ == "__main__":
    sys.exit(main())


# Test discovery and validation utilities
class TestDiscovery:
    """Utilities for test discovery and validation."""

    @staticmethod
    def find_test_files(test_dir: Path) -> list[Path]:
        """Find all test files in directory.

        Args:
            test_dir: Directory to search

        Returns:
            List of test file paths
        """
        return list(test_dir.rglob("test_*.py"))

    @staticmethod
    def validate_test_structure(test_dir: Path) -> list[str]:
        """Validate test directory structure.

        Args:
            test_dir: Test directory to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check required directories exist
        required_dirs = [
            "test_widgets",
            "test_controllers",
            "test_dialogs",
            "test_utils",
            "test_integration",
        ]

        for dir_name in required_dirs:
            dir_path = test_dir / dir_name
            if not dir_path.exists():
                issues.append(f"Missing directory: {dir_name}")
            elif not (dir_path / "__init__.py").exists():
                issues.append(f"Missing __init__.py in: {dir_name}")

        # Check conftest.py exists
        if not (test_dir / "conftest.py").exists():
            issues.append("Missing conftest.py")

        return issues

    @staticmethod
    def count_tests_by_category(test_dir: Path) -> dict:
        """Count tests by category.

        Args:
            test_dir: Test directory

        Returns:
            Dictionary with test counts by category
        """
        counts = {
            "widgets": 0,
            "controllers": 0,
            "dialogs": 0,
            "utils": 0,
            "integration": 0,
            "total": 0,
        }

        for category in ["widgets", "controllers", "dialogs", "utils", "integration"]:
            category_dir = test_dir / f"test_{category}"
            if category_dir.exists():
                test_files = list(category_dir.glob("test_*.py"))
                counts[category] = len(test_files)
                counts["total"] += len(test_files)

        return counts
