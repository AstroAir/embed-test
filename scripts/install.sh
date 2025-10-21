#!/bin/bash
# PDF Vector System Installation Script

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

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python"
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python not found. Please install Python 3.9+"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."

    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        print_success "pip found"
        PIP_CMD="pip"
    else
        print_error "pip not found. Please install pip"
        exit 1
    fi
}

# Install UV if available
install_uv() {
    print_status "Checking for UV package manager..."

    if command -v uv &> /dev/null; then
        print_success "UV found, using for faster installation"
        USE_UV=true
    else
        print_status "UV not found, installing..."
        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            export PATH="$HOME/.cargo/bin:$PATH"
            if command -v uv &> /dev/null; then
                print_success "UV installed successfully"
                USE_UV=true
            else
                print_warning "UV installation failed, falling back to pip"
                USE_UV=false
            fi
        else
            print_warning "UV installation failed, falling back to pip"
            USE_UV=false
        fi
    fi
}

# Install PDF Vector System
install_package() {
    print_status "Installing PDF Vector System..."

    if [ "$USE_UV" = true ]; then
        if uv pip install pdf-vector-system; then
            print_success "PDF Vector System installed successfully with UV"
        else
            print_error "Installation failed with UV, trying pip..."
            if $PIP_CMD install pdf-vector-system; then
                print_success "PDF Vector System installed successfully with pip"
            else
                print_error "Installation failed"
                exit 1
            fi
        fi
    else
        if $PIP_CMD install pdf-vector-system; then
            print_success "PDF Vector System installed successfully"
        else
            print_error "Installation failed"
            exit 1
        fi
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."

    if command -v pdf-vector &> /dev/null; then
        print_success "CLI command available"
        pdf-vector --version
    else
        print_error "CLI command not found"
        exit 1
    fi

    if $PYTHON_CMD -c "import pdf_vector_system; print(f'Version: {pdf_vector_system.__version__}')" 2>/dev/null; then
        print_success "Python package import successful"
    else
        print_error "Python package import failed"
        exit 1
    fi
}

# Main installation process
main() {
    echo "PDF Vector System Installation Script"
    echo "====================================="
    echo

    check_python
    check_pip
    install_uv
    install_package
    verify_installation

    echo
    print_success "Installation completed successfully!"
    echo
    echo "Next steps:"
    echo "  1. Run 'pdf-vector --help' to see available commands"
    echo "  2. Check the documentation at: https://your-username.github.io/pdf-vector-system/"
    echo "  3. Try the quick start guide: https://your-username.github.io/pdf-vector-system/getting-started/quickstart/"
    echo
}

# Run main function
main "$@"
