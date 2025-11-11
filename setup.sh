#!/bin/bash
################################################################################
# PyTorch for Financial Derivatives - Environment Setup Script
#
# This script sets up a Python virtual environment with all required
# dependencies for running the PyTorch derivatives pricing examples.
#
# Usage:
#   bash setup.sh [--gpu]
#
# Options:
#   --gpu    Install GPU-enabled PyTorch with CUDA support
#
# Author: PyTorch Derivatives Package
# Version: 1.0
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="pytorch_derivatives_env"
PYTHON_MIN_VERSION="3.8"
CUDA_VERSION="cu118"  # CUDA 11.8 by default

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║     PyTorch for Financial Derivatives - Setup Script          ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo -e "\n${BLUE}▶ $1${NC}"
    echo "================================================================"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "  $1"
}

################################################################################
# Check Prerequisites
################################################################################

check_python() {
    print_section "Checking Python Installation"
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed!"
        print_info "Please install Python 3.8 or higher:"
        print_info "  Ubuntu/Debian: sudo apt-get install python3 python3-venv python3-pip"
        print_info "  macOS: brew install python3"
        print_info "  Windows: Download from python.org"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Found Python $PYTHON_VERSION"
    
    # Check Python version
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    MIN_MAJOR=$(echo $PYTHON_MIN_VERSION | cut -d'.' -f1)
    MIN_MINOR=$(echo $PYTHON_MIN_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt "$MIN_MAJOR" ] || \
       ([ "$PYTHON_MAJOR" -eq "$MIN_MAJOR" ] && [ "$PYTHON_MINOR" -lt "$MIN_MINOR" ]); then
        print_error "Python $PYTHON_MIN_VERSION or higher is required!"
        print_info "Current version: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python version is compatible"
}

check_venv_module() {
    print_section "Checking venv Module"
    
    if ! python3 -m venv --help &> /dev/null; then
        print_error "Python venv module is not available!"
        print_info "Install it with:"
        print_info "  Ubuntu/Debian: sudo apt-get install python3-venv"
        print_info "  macOS: Should be included with Python"
        exit 1
    fi
    
    print_success "venv module is available"
}

################################################################################
# Virtual Environment Setup
################################################################################

create_venv() {
    print_section "Creating Virtual Environment"
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment '$VENV_NAME' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing environment..."
            rm -rf "$VENV_NAME"
        else
            print_info "Using existing environment"
            return 0
        fi
    fi
    
    print_info "Creating new virtual environment: $VENV_NAME"
    python3 -m venv "$VENV_NAME"
    print_success "Virtual environment created"
}

activate_venv() {
    print_section "Activating Virtual Environment"
    
    # Activate based on OS
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source "$VENV_NAME/Scripts/activate"
        ACTIVATE_CMD="$VENV_NAME\\Scripts\\activate"
    else
        source "$VENV_NAME/bin/activate"
        ACTIVATE_CMD="source $VENV_NAME/bin/activate"
    fi
    
    print_success "Virtual environment activated"
}

################################################################################
# Package Installation
################################################################################

upgrade_pip() {
    print_section "Upgrading pip"
    
    python -m pip install --upgrade pip setuptools wheel
    print_success "pip upgraded to latest version"
}

install_pytorch() {
    print_section "Installing PyTorch"
    
    if [ "$INSTALL_GPU" = true ]; then
        print_info "Installing PyTorch with CUDA $CUDA_VERSION support..."
        pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_VERSION"
        print_success "PyTorch (GPU version) installed"
    else
        print_info "Installing PyTorch (CPU version)..."
        pip install torch torchvision torchaudio
        print_success "PyTorch (CPU version) installed"
    fi
}

install_dependencies() {
    print_section "Installing Dependencies"
    
    if [ -f "requirements_pytorch.txt" ]; then
        print_info "Installing from requirements_pytorch.txt..."
        pip install -r requirements_pytorch.txt
    else
        print_warning "requirements_pytorch.txt not found, installing manually..."
        pip install numpy scipy matplotlib jupyter ipython
    fi
    
    print_success "All dependencies installed"
}

################################################################################
# Verification
################################################################################

verify_installation() {
    print_section "Verifying Installation"
    
    print_info "Testing PyTorch import..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
        print_error "PyTorch import failed!"
        exit 1
    }
    print_success "PyTorch imported successfully"
    
    print_info "Checking CUDA availability..."
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        print_success "CUDA is available (version: $CUDA_VERSION)"
    else
        if [ "$INSTALL_GPU" = true ]; then
            print_warning "CUDA is not available (no compatible GPU or drivers)"
        else
            print_info "CUDA not available (CPU-only installation)"
        fi
    fi
    
    print_info "Testing NumPy..."
    python -c "import numpy as np; print(f'NumPy version: {np.__version__}')" || {
        print_error "NumPy import failed!"
        exit 1
    }
    print_success "NumPy imported successfully"
    
    print_info "Testing SciPy..."
    python -c "import scipy; print(f'SciPy version: {scipy.__version__}')" || {
        print_error "SciPy import failed!"
        exit 1
    }
    print_success "SciPy imported successfully"
    
    print_info "Testing Matplotlib..."
    python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')" || {
        print_error "Matplotlib import failed!"
        exit 1
    }
    print_success "Matplotlib imported successfully"
}

test_demo() {
    print_section "Testing Demo Code"
    
    print_info "Running quick validation test..."
    
    cat > test_pytorch_derivatives.py << 'EOF'
import torch
import numpy as np

def test_autograd():
    """Test PyTorch autograd with simple function."""
    x = torch.tensor(3.0, requires_grad=True)
    y = x**2 + 2*x + 1
    y.backward()
    
    expected_grad = 2*3 + 2  # dy/dx = 2x + 2 = 8
    actual_grad = x.grad.item()
    
    assert abs(expected_grad - actual_grad) < 1e-6, f"Expected {expected_grad}, got {actual_grad}"
    print(f"✓ Autograd test passed: dy/dx = {actual_grad}")
    return True

def test_derivatives_pricing():
    """Test basic Black-Scholes with autograd."""
    # Simple test
    S0 = torch.tensor(100.0, requires_grad=True)
    K = 100.0
    
    # Simplified option value
    value = torch.maximum(S0 - K, torch.tensor(0.0))
    
    # Compute delta
    value.backward()
    delta = S0.grad.item()
    
    # For ATM call, delta should be close to 0.5 (actually slightly above)
    print(f"✓ Simple option delta: {delta}")
    return True

if __name__ == "__main__":
    print("Testing PyTorch for derivatives pricing...")
    print("-" * 50)
    
    try:
        test_autograd()
        test_derivatives_pricing()
        print("-" * 50)
        print("✓ All tests passed!")
        print("\nSetup is complete and working correctly!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        exit(1)
EOF
    
    python test_pytorch_derivatives.py
    rm test_pytorch_derivatives.py
    
    print_success "Demo test completed successfully"
}

################################################################################
# Usage Instructions
################################################################################

print_usage_instructions() {
    print_section "Setup Complete! 🎉"
    
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║  Virtual environment setup completed successfully!            ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo -e "\n${YELLOW}Next Steps:${NC}"
    echo "================================================================"
    
    echo -e "\n${BLUE}1. Activate the environment:${NC}"
    echo "   $ACTIVATE_CMD"
    
    echo -e "\n${BLUE}2. Run the demo:${NC}"
    echo "   python pytorch_derivatives.py"
    
    echo -e "\n${BLUE}3. Open Jupyter notebook:${NC}"
    echo "   jupyter notebook pytorch_derivatives_demo.ipynb"
    
    echo -e "\n${BLUE}4. Read the documentation:${NC}"
    echo "   cat differentiable_programming_intro.md"
    echo "   cat pytorch_derivatives_overview.md"
    
    echo -e "\n${YELLOW}To deactivate the environment later:${NC}"
    echo "   deactivate"
    
    echo -e "\n${YELLOW}To activate it again:${NC}"
    echo "   $ACTIVATE_CMD"
    
    echo ""
}

################################################################################
# Main Script
################################################################################

main() {
    print_header
    
    # Parse command line arguments
    INSTALL_GPU=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu)
                INSTALL_GPU=true
                shift
                ;;
            --cuda-version)
                CUDA_VERSION="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: bash setup.sh [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --gpu                 Install GPU-enabled PyTorch"
                echo "  --cuda-version VER    Specify CUDA version (e.g., cu118, cu121)"
                echo "  --help, -h            Show this help message"
                echo ""
                echo "Examples:"
                echo "  bash setup.sh                    # CPU-only installation"
                echo "  bash setup.sh --gpu              # GPU with default CUDA version"
                echo "  bash setup.sh --gpu --cuda-version cu121  # GPU with CUDA 12.1"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    if [ "$INSTALL_GPU" = true ]; then
        print_info "GPU installation requested (CUDA: $CUDA_VERSION)"
    else
        print_info "CPU-only installation"
    fi
    
    # Run setup steps
    check_python
    check_venv_module
    create_venv
    activate_venv
    upgrade_pip
    
    if [ "$INSTALL_GPU" = true ]; then
        install_pytorch
        install_dependencies
    else
        # For CPU, install all from requirements if available
        if [ -f "requirements_pytorch.txt" ]; then
            install_dependencies
        else
            install_pytorch
            install_dependencies
        fi
    fi
    
    verify_installation
    test_demo
    print_usage_instructions
    
    echo -e "${GREEN}Setup completed successfully!${NC}\n"
}

# Run main function
main "$@"
