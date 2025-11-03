#!/bin/bash

# STEP Embedding - Build and Run Script

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
EXECUTABLE="$BUILD_DIR/step-embedding"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 <command> [arguments]"
    echo ""
    echo "Commands:"
    echo "  build                 Build the project"
    echo "  rebuild               Clean and rebuild"
    echo "  clean                 Remove build directory"
    echo "  extract <args>        Run feature extraction"
    echo "  example               Run example without Python"
    echo "  example-python        Run example with Python predictor"
    echo "  help                  Show this help message"
    echo ""
    echo "Extract arguments:"
    echo "  -i <file>       Input file path (required)"
    echo "  -o <file>       Output file path (required)"
    echo "  -m <int>        Max motif length (default: 3)"
    echo "  -d <double>     Delta C time window (default: auto)"
    echo "  --negative      Enable negative sampling"
    echo "  --neg-ratio <r> Negative sampling ratio (default: 1.0)"
    echo "  --no-python     Disable Python predictor"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 example"
    echo "  $0 example-python"
    echo "  $0 extract -i data.txt -o features.csv"
    echo "  $0 extract -i data.txt -o features.csv --negative --neg-ratio 1.0"
}

build_project() {
    echo -e "${GREEN}Building STEP Embedding...${NC}"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    echo -e "${GREEN}Build complete!${NC}"
}

clean_project() {
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}Clean complete!${NC}"
}

check_executable() {
    if [ ! -f "$EXECUTABLE" ]; then
        echo -e "${RED}Error: Executable not found. Please build first.${NC}"
        echo "Run: $0 build"
        exit 1
    fi
}

run_example() {
    echo -e "${GREEN}Running example (without Python)...${NC}"
    check_executable

    # Create example data if it doesn't exist
    if [ ! -f "$SCRIPT_DIR/example_data.txt" ]; then
        echo -e "${YELLOW}Creating example data file...${NC}"
        cat > "$SCRIPT_DIR/example_data.txt" << EOF
1 2 1000
2 3 1005
1 3 1010
3 4 1020
2 4 1025
1 4 1030
4 5 1040
3 5 1045
2 5 1050
1 5 1060
EOF
        echo -e "${GREEN}Example data created: example_data.txt${NC}"
    fi

    "$EXECUTABLE" \
        -i "$SCRIPT_DIR/example_data.txt" \
        -o "$SCRIPT_DIR/example_output.csv" \
        -m 3 \
        --no-python

    echo ""
    echo -e "${GREEN}Example complete!${NC}"
    echo "Output written to: example_output.csv"
    echo ""
    echo "First few lines of output:"
    head -n 10 "$SCRIPT_DIR/example_output.csv"
}

run_example_python() {
    echo -e "${GREEN}Running example with Python predictor...${NC}"
    check_executable

    # Check if example data exists
    if [ ! -f "$SCRIPT_DIR/example_data.txt" ]; then
        echo -e "${YELLOW}Creating example data file...${NC}"
        cat > "$SCRIPT_DIR/example_data.txt" << EOF
1 2 1000
2 3 1005
1 3 1010
3 4 1020
2 4 1025
1 4 1030
4 5 1040
3 5 1045
2 5 1050
1 5 1060
EOF
        echo -e "${GREEN}Example data created: example_data.txt${NC}"
    fi

    # Check if Python script exists
    if [ ! -f "$SCRIPT_DIR/python_predictor.py" ]; then
        echo -e "${RED}Error: python_predictor.py not found${NC}"
        echo "Please create the Python predictor script first."
        exit 1
    fi

    # Start Python predictor in background
    echo -e "${YELLOW}Starting Python predictor in background...${NC}"
    python3 "$SCRIPT_DIR/python_predictor.py" > /tmp/python_predictor.log 2>&1 &
    PYTHON_PID=$!

    # Give Python a moment to start, then run C++ (Python will wait for shared memory)
    sleep 1
    echo -e "${GREEN}Running C++ extractor...${NC}"
    "$EXECUTABLE" \
        -i "$SCRIPT_DIR/example_data.txt" \
        -o "$SCRIPT_DIR/example_output_python.csv" \
        -m 3

    # Kill Python process
    echo "Stopping Python predictor..."
    kill $PYTHON_PID 2>/dev/null || true
    wait $PYTHON_PID 2>/dev/null || true

    echo ""
    echo -e "${GREEN}Example complete!${NC}"
    echo "Output written to: example_output_python.csv"
    echo ""
    echo "First few lines of output:"
    head -n 10 "$SCRIPT_DIR/example_output_python.csv"
    echo ""
    echo "Python log (last 10 lines):"
    tail -n 10 /tmp/python_predictor.log 2>/dev/null || echo "(No log available)"
}

# Main command dispatcher
case "${1:-}" in
    build)
        build_project
        ;;
    rebuild)
        clean_project
        build_project
        ;;
    clean)
        clean_project
        ;;
    extract)
        check_executable
        shift
        "$EXECUTABLE" "$@"
        ;;
    example)
        run_example
        ;;
    example-python)
        run_example_python
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '${1:-}'${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
