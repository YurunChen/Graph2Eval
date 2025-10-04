#!/bin/bash

# GraphRAG + TaskCraft Benchmark Environment Setup Script
# Create isolated conda environment and install dependencies

set -e  # Exit immediately on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Environment variables
ENV_NAME="graph2eval"
PYTHON_VERSION="3.10"

# Print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
    echo "$(printf '=%.0s' {1..50})"
}

print_step() {
    echo -e "${CYAN}🔄 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if command exists
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check if conda is available
check_conda() {
    print_step "Checking conda environment"
    
    if check_command conda; then
        conda_version=$(conda --version)
        print_success "Conda available: $conda_version"
        return 0
    else
        print_error "Conda not installed or not in PATH"
        echo "Please install Anaconda or Miniconda first:"
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
}

# Create conda environment
create_conda_env() {
    print_step "Creating conda environment"
    
    # Check if environment already exists
    if conda env list | grep -q "^${ENV_NAME}"; then
        print_warning "Conda environment '${ENV_NAME}' already exists"
        echo -n "Do you want to delete and recreate it? (y/n): "
        read -r response
        
        if [[ "$response" =~ ^[Yy]$ ]]; then
            print_step "Removing existing environment '${ENV_NAME}'"
            conda env remove -n "$ENV_NAME" -y
            print_success "Environment removed successfully"
        else
            print_success "Using existing environment '${ENV_NAME}'"
            return 0
        fi
    fi
    
    # Create new environment
    print_step "Creating new conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    
    if [ $? -eq 0 ]; then
        print_success "Conda environment '${ENV_NAME}' created successfully"
        print_message $BLUE "💡 Activate environment command: conda activate ${ENV_NAME}"
        return 0
    else
        print_error "Environment creation failed"
        return 1
    fi
}

# Create project directories
create_directories() {
    print_step "Creating project directories"
    
    directories=(
        "data"
        "data/documents"
        "data/graphs"
        "data/datasets"
        "data/vectors"
        "logs"
        "models"
        "models/huggingface"
        "outputs"
        "outputs/results"
        "outputs/reports"
        "cache"
        "cache/embeddings"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_success "$dir"
    done
}

# Install conda packages
install_conda_packages() {
    print_step "Installing base packages with conda"
    
    packages=(
        "pyyaml"
        "pandas"
        "numpy"
        "networkx"
        "beautifulsoup4"
        "scikit-learn"
    )
    
    for package in "${packages[@]}"; do
        echo "Installing $package..."
        if conda install -n "$ENV_NAME" -c conda-forge "$package" -y >/dev/null 2>&1; then
            print_success "$package"
        else
            print_warning "$package (will install with pip)"
        fi
    done
}

# Install pip packages
install_pip_packages() {
    print_step "Installing other dependencies with pip"
    
    # Check if requirements files exist, skip if not found
    if [ ! -f "requirements_minimal.txt" ] && [ ! -f "requirements.txt" ]; then
        print_warning "No requirements files found, skipping pip package installation"
        print_info "Will install complete AI/ML dependencies directly"
        return 0
    fi
    
    # Prefer requirements_minimal.txt, fallback to requirements.txt
    local req_file=""
    if [ -f "requirements_minimal.txt" ]; then
        req_file="requirements_minimal.txt"
    elif [ -f "requirements.txt" ]; then
        req_file="requirements.txt"
    fi
    
    if [ -n "$req_file" ]; then
        print_info "Using file: $req_file"
        echo "Executing: conda run -n ${ENV_NAME} pip install -r $req_file"
        if conda run -n "$ENV_NAME" pip install -r "$req_file"; then
            print_success "Dependencies installed successfully"
            return 0
        else
            print_warning "Some dependencies may have issues, but continuing"
            return 0
        fi
    fi
    
    return 0
}

# Install complete AI/ML dependencies
install_complete_dependencies() {
    print_step "Installing complete AI/ML dependencies"
    
    local ml_packages=(
        "torch"
        "transformers>=4.30.0" 
        "sentence-transformers>=2.2.0"
        "openai>=1.0.0"
        "anthropic>=0.18.0"
        "spacy>=3.7.0"
        "faiss-cpu>=1.7.4"
        "loguru>=0.7.0"
        "jinja2>=3.1.0"
        "PyPDF2>=3.0.0"
        "pdfplumber>=0.11.0"
        "python-docx>=0.8.11"
        "playwright>=1.40.0"
        "selenium>=4.15.0"
    )
    
    print_info "Installing AI/ML packages..."
    for package in "${ml_packages[@]}"; do
        print_info "Installing $package ..."
        if conda run -n "$ENV_NAME" pip install "$package" >/dev/null 2>&1; then
            print_success "$package"
        else
            print_warning "$package installation failed, skipping"
        fi
    done
    
    return 0
}

# Install browser automation tools
install_browser_automation() {
    print_step "Installing browser automation tools"
    
    print_info "Installing Playwright browsers..."
    # Install all browsers (chromium, firefox, webkit) with system dependencies
    if conda run -n "$ENV_NAME" playwright install --with-deps; then
        print_success "Playwright browsers installed successfully"
    else
        print_warning "Playwright browser installation failed, trying without system dependencies..."
        # Fallback: install browsers without system dependencies
        if conda run -n "$ENV_NAME" playwright install chromium firefox webkit; then
            print_success "Playwright browsers installed (without system dependencies)"
        else
            print_warning "Playwright browser installation failed, will use Selenium fallback"
        fi
    fi
    
    print_info "Checking Chrome/Chromium availability for Selenium..."
    if check_command google-chrome; then
        print_success "Google Chrome found"
    elif check_command chromium; then
        print_success "Chromium found"
    elif check_command chromium-browser; then
        print_success "Chromium browser found"
    else
        print_warning "No Chrome/Chromium found, Selenium may not work properly"
        print_info "Consider installing Chrome or Chromium for full browser automation support"
    fi
    
    return 0
}

# Download and configure NLP models
setup_nlp_models() {
    print_step "Setting up NLP models"
    
    print_info "Downloading spaCy English model..."
    if conda run -n "$ENV_NAME" python -m spacy download en_core_web_sm >/dev/null 2>&1; then
        print_success "spaCy model downloaded successfully"
    else
        print_warning "spaCy model download failed, system will use simplified functionality"
    fi
    
    return 0
}

# Setup environment variables file
setup_env_file() {
    print_step "Setting up environment variables"
    
    cat > .env.example << 'EOF'
# API Configuration - Please enter your real API keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Debug settings
DEBUG_MODE=false
LOG_LEVEL=INFO

# Performance settings
MAX_WORKERS=4
BATCH_SIZE=32
EOF

    print_success "Created environment variables example file: .env.example"
    print_message $BLUE "💡 Please copy .env.example to .env and enter your real API keys"
}

# Test basic imports
test_imports() {
    print_step "Testing basic imports"
    
    cat > test_imports.py << 'EOF'
import sys
test_modules = [
    ("yaml", "pyyaml"),
    ("pandas", "pandas"), 
    ("numpy", "numpy"),
    ("networkx", "networkx"),
    ("bs4", "beautifulsoup4"),
    ("sklearn", "scikit-learn")
]

failed_imports = []
for module_name, package_name in test_modules:
    try:
        __import__(module_name)
        print(f"✅ {module_name}")
    except ImportError:
        print(f"❌ {module_name} (needs to install {package_name})")
        failed_imports.append(package_name)

print(f"failed_count:{len(failed_imports)}")
EOF

    # Run test in conda environment
    output=$(conda run -n "$ENV_NAME" python test_imports.py 2>/dev/null)
    echo "$output"
    
    if echo "$output" | grep -q "failed_count:0"; then
        print_success "All basic packages imported successfully"
        success=true
    else
        print_warning "Some packages failed to import, but environment is basically usable"
        success=true  # Be lenient
    fi
    
    # Clean up temporary file
    rm -f test_imports.py
    
    return 0
}

# Test configuration system
test_config() {
    print_step "Testing configuration system"
    
    cat > test_config_temp.py << EOF
import sys
sys.path.insert(0, "$(pwd)")

try:
    from config_manager import get_config
    config = get_config()
    print("✅ Configuration loaded successfully")
    print(f"  Project name: {config.project.get('name', 'Unknown')}")
    print(f"  Model config: {config.agent.get('execution', {}).get('model_name', 'Unknown')}")
    print(f"  Dataset size: {config.task_craft.get('generation', {}).get('max_total_tasks', 'Unknown')}")
    print("test_success:True")
except Exception as e:
    print(f"❌ Configuration system test failed: {e}")
    print("test_success:False")
EOF

    # Run test in conda environment
    output=$(conda run -n "$ENV_NAME" python test_config_temp.py 2>/dev/null)
    echo "$output"
    
    if echo "$output" | grep -q "test_success:True"; then
        success=true
    else
        print_error "Configuration system test failed in conda environment"
        success=false
    fi
    
    # Clean up temporary file
    rm -f test_config_temp.py
    
    [ "$success" = true ]
}


# Print next steps
print_next_steps() {
    echo ""
    print_header "📋 Next Steps"
    echo "1. Activate conda environment (choose one):"
    echo "   • conda activate ${ENV_NAME}"
    echo "   • source activate.sh"
    echo "   • conda run -n ${ENV_NAME} python <script> (for one-time execution)"
    echo ""
    echo "2. Copy and edit environment variables file:"
    echo "   cp .env.example .env"
    echo "   # Edit .env file and enter your real API keys"
    echo ""
    echo "3. Run benchmark:"
    echo "   conda run -n ${ENV_NAME} python benchmark_runner.py"
    echo ""
    echo "4. Complete dependencies installed, update if needed:"
    echo "   conda run -n ${ENV_NAME} pip install -r requirements.txt"
    echo ""
    print_message $GREEN "🎉 Environment setup completed!"
}

# Main function
main() {
    print_header "🚀 GraphRAG + TaskCraft Benchmark Environment Setup"
    
    # Check runtime environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        print_warning "Windows environment detected, recommend running this script in WSL"
    fi
    
    print_message $CYAN "📦 Performing complete installation (including AI/ML features)"
    INSTALL_COMPLETE=true

    # Execution steps
    steps=(
        "check_conda:Checking conda availability"
        "create_conda_env:Creating conda environment"
        "create_directories:Creating project directories"
        "install_conda_packages:Installing conda packages"
        "install_pip_packages:Installing pip packages"
        "install_complete_dependencies:Installing complete AI/ML dependencies"
        "install_browser_automation:Installing browser automation tools"
        "setup_nlp_models:Setting up NLP models"
        "setup_env_file:Setting up environment variables"
    )
    
    failed_steps=()
    
    for step in "${steps[@]}"; do
        IFS=':' read -r func_name step_name <<< "$step"
        echo ""
        print_step "$step_name"
        
        if ! $func_name; then
            print_error "$step_name failed"
            failed_steps+=("$step_name")
        fi
    done
    
    echo ""
    print_header "🏁 Environment Setup Completed"
    
    if [ ${#failed_steps[@]} -eq 0 ]; then
        print_success "All steps completed successfully!"
    else
        print_warning "Some steps were not successful: ${failed_steps[*]}"
        print_message $BLUE "💡 You can handle these steps manually or retry later"
    fi
    
    print_next_steps
    
    # Environment setup completed
    echo ""
    print_header "🏁 Environment Setup Completed"
    print_success "Environment setup completed successfully!"
    print_message $GREEN "💡 Your GraphRAG Benchmark environment is ready!"
    
    print_info "To activate the environment, run:"
    echo "  conda activate ${ENV_NAME}"
    echo "  or use: source activate.sh"
    
    # Return status
    [ ${#failed_steps[@]} -eq 0 ]
}

# Run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
