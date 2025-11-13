#!/bin/bash
# VectorFlow Build Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
BUILD_DIR="dist"
CLEAN_BUILD=false
RUN_TESTS=true
CHECK_QUALITY=true
BUILD_DOCS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --no-tests)
            RUN_TESTS=false
            shift
            ;;
        --no-quality)
            CHECK_QUALITY=false
            shift
            ;;
        --docs)
            BUILD_DOCS=true
            shift
            ;;
        --help)
            echo "VectorFlow Build Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clean      Clean build directory before building"
            echo "  --no-tests   Skip running tests"
            echo "  --no-quality Skip code quality checks"
            echo "  --docs       Build documentation"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Clean build directory
clean_build() {
    if [ "$CLEAN_BUILD" = true ]; then
        print_status "Cleaning build directory..."
        rm -rf $BUILD_DIR build *.egg-info
        print_success "Build directory cleaned"
    fi
}

# Check dependencies
check_dependencies() {
    print_status "Checking build dependencies..."

    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi

    # Check for UV or pip
    if command -v uv &> /dev/null; then
        print_success "UV found"
        PACKAGE_MANAGER="uv"
    elif command -v pip &> /dev/null; then
        print_success "pip found"
        PACKAGE_MANAGER="pip"
    else
        print_error "No package manager found (uv or pip required)"
        exit 1
    fi

    # Install build dependencies
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        uv sync --extra dev
    else
        pip install -e ".[dev]"
    fi
}

# Run code quality checks
run_quality_checks() {
    if [ "$CHECK_QUALITY" = true ]; then
        print_status "Running code quality checks..."

        # Ruff linting
        print_status "Running ruff linter..."
        if [ "$PACKAGE_MANAGER" = "uv" ]; then
            uv run ruff check vectorflow tests examples
        else
            ruff check vectorflow tests examples
        fi

        # Ruff formatting
        print_status "Checking code formatting..."
        if [ "$PACKAGE_MANAGER" = "uv" ]; then
            uv run ruff format --check vectorflow tests examples
        else
            ruff format --check vectorflow tests examples
        fi

        # MyPy type checking
        print_status "Running type checks..."
        if [ "$PACKAGE_MANAGER" = "uv" ]; then
            uv run mypy vectorflow
        else
            mypy vectorflow
        fi

        # Security checks
        print_status "Running security checks..."
        if [ "$PACKAGE_MANAGER" = "uv" ]; then
            uv run bandit -r vectorflow
            uv run safety check
        else
            bandit -r vectorflow
            safety check
        fi

        print_success "Code quality checks passed"
    fi
}

# Run tests
run_tests() {
    if [ "$RUN_TESTS" = true ]; then
        print_status "Running tests..."

        if [ "$PACKAGE_MANAGER" = "uv" ]; then
            uv run pytest tests/ -v --cov=vectorflow --cov-report=term-missing
        else
            pytest tests/ -v --cov=vectorflow --cov-report=term-missing
        fi

        print_success "Tests passed"
    fi
}

# Build documentation
build_documentation() {
    if [ "$BUILD_DOCS" = true ]; then
        print_status "Building documentation..."

        if [ "$PACKAGE_MANAGER" = "uv" ]; then
            uv sync --extra docs
            uv run mkdocs build
        else
            pip install -e ".[docs]"
            mkdocs build
        fi

        print_success "Documentation built"
    fi
}

# Build package
build_package() {
    print_status "Building package..."

    # Install build tools
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        uv tool install build
        uv tool install twine
    else
        pip install build twine
    fi

    # Build the package
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        uvx build
    else
        python -m build
    fi

    # Check the package
    if [ "$PACKAGE_MANAGER" = "uv" ]; then
        uvx twine check dist/*
    else
        twine check dist/*
    fi

    print_success "Package built successfully"
}

# Display build information
show_build_info() {
    print_status "Build Information:"
    echo "  Build directory: $BUILD_DIR"
    echo "  Package manager: $PACKAGE_MANAGER"
    echo "  Clean build: $CLEAN_BUILD"
    echo "  Run tests: $RUN_TESTS"
    echo "  Quality checks: $CHECK_QUALITY"
    echo "  Build docs: $BUILD_DOCS"
    echo

    if [ -d "$BUILD_DIR" ]; then
        echo "Built packages:"
        ls -la $BUILD_DIR/
    fi
}

# Main build process
main() {
    echo "VectorFlow Build Script"
    echo "======================"
    echo

    clean_build
    check_dependencies
    run_quality_checks
    run_tests
    build_documentation
    build_package
    show_build_info

    print_success "Build completed successfully!"
    echo
    echo "Next steps:"
    echo "  1. Test the package: pip install dist/*.whl"
    echo "  2. Upload to PyPI: twine upload dist/*"
    echo "  3. Create a GitHub release"
    echo
}

# Run main function
main "$@"
